"""P11-FE361 — SAE-disentangled L19 correctness probe vs F-2 (0.7731).

Goal: test whether F-2's single raw-direction DoM probe (OOF AUROC 0.7731 on
Qwen-2.5-1.5B MATH-500 L19 prefill) is ceiling-limited by L19 polysemanticity.
The LF-Steering recipe (Liu et al., LF-Steering Table 3) locates *key features*
in a sparse, disentangled code rather than the raw hidden state, then probes on
that masked feature set.

Pipeline per OOF fold:
  1. Standardize L19 prefill activations on the train fold.
  2. Obtain a sparse, overcomplete code z.
       - Preferred: a pretrained Qwen SAE supplied offline as an NPZ of encoder
         weights at SAE_NPZ (keys W_enc:(1536,F), b_enc:(F,), optional b_pre:(1536,)).
         Encoded once, fold-independent: z = relu((x - b_pre) @ W_enc + b_enc).
       - Fallback (no checkpoint, fully offline / CPU): an unsupervised sparse
         dictionary learned on the train fold only (MiniBatchDictionaryLearning),
         encoded with a non-negative sparse coder. This is a faithful surrogate
         for the SAE step — it shares the "overcomplete + sparse + label-free"
         structure — but is NOT the Goodfire-Ember checkpoint; reconnaissance for
         a downloadable Qwen-1.5B SAE is out of scope for an offline recompute.
  3. LF-Steering key-feature locator on the TRAIN codes:
         g_i = mean|z_i(correct) - z_i(incorrect)|, keep features with g_i >= t,
         t = 0.10 (codes are min-max normalized per-feature on train so t is
         comparable to the LF-Steering threshold).
  4. Fit a logistic probe on the masked feature set (train), score the held-out
     test fold. Aggregate scores into an OOF AUROC and compare to F-2's 0.7731.

Constraints: CPU only, numpy/scipy/sklearn, no network, no GPU/torch.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Optional pretrained-SAE encoder weights (offline NPZ). Absent -> local surrogate.
SAE_NPZ = ROOT / "pathway11_h100/sae_disentangle/qwen_sae.npz"
OUT_JSON = ROOT / "pathway11_h100/sae_disentangle/results.json"

SEED = 9999
N_FOLDS = 5
THRESHOLD = 0.10            # LF-Steering key-feature gate t
N_ATOMS = 512              # surrogate dictionary size (overcomplete vs train n)
DICT_ALPHA = 1.0          # sparsity penalty for surrogate dictionary learning
F2_BASELINE = 0.7731


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y, k, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def load_pretrained_sae():
    """Return (W_enc, b_enc, b_pre) if an offline SAE checkpoint NPZ exists."""
    if not SAE_NPZ.exists():
        return None
    blob = np.load(SAE_NPZ)
    if "W_enc" not in blob or "b_enc" not in blob:
        return None
    W_enc = blob["W_enc"].astype(np.float64)        # (1536, F)
    b_enc = blob["b_enc"].astype(np.float64)        # (F,)
    b_pre = blob["b_pre"].astype(np.float64) if "b_pre" in blob else np.zeros(W_enc.shape[0])
    if W_enc.shape[0] != 1536 or b_enc.shape[0] != W_enc.shape[1]:
        return None
    return W_enc, b_enc, b_pre


def encode_pretrained(X, sae):
    W_enc, b_enc, b_pre = sae
    z = (X - b_pre) @ W_enc + b_enc
    return np.maximum(z, 0.0)


def learn_surrogate_dict(X_train_std):
    """Unsupervised overcomplete sparse dictionary on the train fold only."""
    from sklearn.decomposition import MiniBatchDictionaryLearning
    n_atoms = min(N_ATOMS, max(64, X_train_std.shape[0]))
    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        alpha=DICT_ALPHA,
        n_iter=200,
        batch_size=64,
        transform_algorithm="lasso_lars",
        transform_alpha=DICT_ALPHA,
        positive_code=True,
        random_state=SEED,
    )
    dl.fit(X_train_std)
    return dl


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    from sklearn.linear_model import LogisticRegression

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    n = len(y)
    sae = load_pretrained_sae()
    mode = "pretrained_sae" if sae is not None else "local_dict_surrogate"

    # Pre-encode once if a fixed pretrained SAE is available.
    Z_full = encode_pretrained(X, sae) if sae is not None else None

    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof_scores = np.full(n, np.nan, dtype=np.float64)
    n_selected_per_fold = []

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]

        if sae is not None:
            Ztr = Z_full[train_mask]
            Zte = Z_full[test_idx]
        else:
            Xtr, Xte = X[train_mask], X[test_idx]
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
            dl = learn_surrogate_dict((Xtr - mu) / sd)
            Ztr = dl.transform((Xtr - mu) / sd)
            Zte = dl.transform((Xte - mu) / sd)

        # Per-feature min-max normalize on train so threshold t is comparable.
        zmin = Ztr.min(0)
        zrng = Ztr.max(0) - zmin + 1e-12
        Ztr_n = (Ztr - zmin) / zrng
        Zte_n = (Zte - zmin) / zrng

        # LF-Steering key-feature locator.
        g = np.abs(Ztr_n[ytr].mean(0) - Ztr_n[~ytr].mean(0))
        keep = g >= THRESHOLD
        if keep.sum() < 2:  # guarantee a non-degenerate probe
            keep = np.zeros_like(keep, dtype=bool)
            keep[np.argsort(g)[-2:]] = True
        n_selected_per_fold.append(int(keep.sum()))

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Ztr_n[:, keep], ytr)
        oof_scores[test_idx] = clf.predict_proba(Zte_n[:, keep])[:, 1]

    auroc_oof = auroc(oof_scores, y)
    out = {
        "experiment": "P11-FE361",
        "mode": mode,
        "sae_checkpoint_present": sae is not None,
        "threshold_t": THRESHOLD,
        "n_atoms": int(N_ATOMS if sae is None else sae[0].shape[1]),
        "n_selected_features_mean": float(np.mean(n_selected_per_fold)),
        "n_selected_features_per_fold": n_selected_per_fold,
        "auroc_oof": float(auroc_oof),
        "f2_baseline_auroc": F2_BASELINE,
        "delta_vs_f2": float(auroc_oof - F2_BASELINE),
        "beats_f2": bool(auroc_oof > F2_BASELINE),
        "note": (
            "local_dict_surrogate is an offline numpy/sklearn stand-in for the SAE "
            "encode step (overcomplete + sparse + label-free), NOT the Goodfire-Ember "
            "Qwen SAE; a true verdict requires dropping that checkpoint at SAE_NPZ."
            if sae is None else
            "Encoded with offline pretrained Qwen SAE supplied at SAE_NPZ."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())