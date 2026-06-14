"""FE689 — TopK SAE on cached 1.5B L19 prefill + TADA TF-IDF feature scoring.

Trains a TopK sparse autoencoder (expansion m=4 → dict size 4*1536, k=64 active
units) from scratch in NumPy on the 500 MATH-500 L19 prefill vectors (CPU-only,
manual Adam). Scores every dictionary feature with the TADA Eq. 6 TF-IDF
differential in two contrasts: (correct vs incorrect) and (D-bucket vs A-bucket,
difficulty proxied by seq_len quartiles). Reports the best single-feature AUROC
and a multi-feature OOF logistic AUROC against the L19 DoM ceiling (0.7731) and
CoE-60 (0.811).

Falsification logic (F-2 / F-9):
- If a single SAE feature beats AUROC 0.7731, the 'single L19 direction' framing
  is wrong.
- If multiple SAE features jointly beat both 0.7731 and 0.811, F-9's CoE-60 ==
  single-layer DoM equivalence dissolves into 'CoE-60 is a coarse projection of
  a sparse L19 feature set.'
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/sae_topk/results.json"

SEED = 9999
N_FOLDS = 5

# SAE hyperparameters
EXPANSION = 4            # m=4 → dict size = 4 * 1536 = 6144
TOPK = 64               # k active units per example
EPOCHS = 600
BATCH = 64
LR = 1e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

# Reference baselines (1024-tok canonical)
L19_DOM_AUROC = 0.7731
COE60_AUROC = 0.811

TOP_N = 10              # how many top TADA features to report / combine


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def signed_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC with the better-of-both-directions sign (features are unsigned)."""
    a = auroc(scores, labels)
    if np.isnan(a):
        return a
    return float(max(a, 1.0 - a))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def topk_mask(pre: np.ndarray, k: int) -> np.ndarray:
    """Per-row TopK mask over relu(pre) (selects k largest positive units)."""
    acts = np.maximum(pre, 0.0)
    if k >= acts.shape[1]:
        return (acts > 0.0).astype(acts.dtype)
    kth = np.partition(acts, -k, axis=1)[:, -k][:, None]
    mask = (acts >= kth) & (acts > 0.0)
    return mask.astype(acts.dtype)


def normalize_decoder(W_dec: np.ndarray) -> np.ndarray:
    """Unit-norm each dictionary atom (row of W_dec, shape (h, d))."""
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return W_dec / norms


def train_topk_sae(X: np.ndarray, h: int, k: int, seed: int):
    """Train a TopK SAE with manual Adam. Returns (acts, info)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Center / scale inputs for stable training; store stats for reporting only.
    mu = X.mean(axis=0)
    Xc = X - mu
    scale = float(np.sqrt((Xc ** 2).sum(axis=1).mean()) + 1e-8)
    Xn = Xc / scale

    W_enc = rng.standard_normal((d, h)).astype(np.float64) * (1.0 / np.sqrt(d))
    b_enc = np.zeros(h, dtype=np.float64)
    W_dec = normalize_decoder(W_enc.T.copy())   # (h, d)
    b_dec = np.zeros(d, dtype=np.float64)

    params = {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec}
    m = {kk: np.zeros_like(v) for kk, v in params.items()}
    v = {kk: np.zeros_like(v) for kk, v in params.items()}

    t = 0
    last_loss = float("nan")
    total_var = float((Xn ** 2).sum(axis=1).mean())

    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        nb = 0
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            xb = Xn[idx]
            B = xb.shape[0]

            pre = xb @ params["W_enc"] + params["b_enc"]      # (B,h)
            mask = topk_mask(pre, k)
            z = np.maximum(pre, 0.0) * mask                   # (B,h)
            recon = z @ params["W_dec"] + params["b_dec"]     # (B,d)

            err = recon - xb
            epoch_loss += float((err ** 2).sum(axis=1).mean())
            nb += 1

            dErr = (2.0 / B) * err                            # (B,d)
            g_W_dec = z.T @ dErr                              # (h,d)
            g_b_dec = dErr.sum(axis=0)                        # (d,)
            dz = dErr @ params["W_dec"].T                     # (B,h)
            dpre = dz * mask                                  # relu+topk straight-through
            g_W_enc = xb.T @ dpre                             # (d,h)
            g_b_enc = dpre.sum(axis=0)                        # (h,)

            grads = {"W_enc": g_W_enc, "b_enc": g_b_enc,
                     "W_dec": g_W_dec, "b_dec": g_b_dec}

            t += 1
            for kk in params:
                m[kk] = BETA1 * m[kk] + (1 - BETA1) * grads[kk]
                v[kk] = BETA2 * v[kk] + (1 - BETA2) * (grads[kk] ** 2)
                mhat = m[kk] / (1 - BETA1 ** t)
                vhat = v[kk] / (1 - BETA2 ** t)
                params[kk] -= LR * mhat / (np.sqrt(vhat) + EPS)

            params["W_dec"] = normalize_decoder(params["W_dec"])

        last_loss = epoch_loss / max(nb, 1)

    # Final dense forward pass for all examples.
    pre = Xn @ params["W_enc"] + params["b_enc"]
    mask = topk_mask(pre, k)
    acts = np.maximum(pre, 0.0) * mask
    recon = acts @ params["W_dec"] + params["b_dec"]
    resid = float(((recon - Xn) ** 2).sum(axis=1).mean())
    frac_var_unexplained = resid / (total_var + 1e-12)

    info = {
        "final_train_mse": last_loss,
        "frac_variance_unexplained": frac_var_unexplained,
        "frac_variance_explained": 1.0 - frac_var_unexplained,
        "dead_features": int((acts.sum(axis=0) == 0).sum()),
        "mean_active_per_example": float((acts > 0).sum(axis=1).mean()),
    }
    return acts, info


def tada_scores(acts: np.ndarray, group_pos: np.ndarray, group_neg: np.ndarray) -> np.ndarray:
    """TADA Eq. 6 TF-IDF differential contrast over features.

    tf_g(f) = mean activation of feature f within group g;
    idf(f)  = log(N / (1 + df(f))), df = #examples with feature active;
    score(f) = (tf_pos - tf_neg) * idf  (signed contrast).
    """
    n = acts.shape[0]
    df = (acts > 0).sum(axis=0).astype(np.float64)
    idf = np.log(n / (1.0 + df))
    tf_pos = acts[group_pos].mean(axis=0) if group_pos.any() else np.zeros(acts.shape[1])
    tf_neg = acts[group_neg].mean(axis=0) if group_neg.any() else np.zeros(acts.shape[1])
    return (tf_pos - tf_neg) * idf


def multi_feature_oof_auroc(acts: np.ndarray, y: np.ndarray, feat_idx: np.ndarray,
                            k: int, seed: int) -> float:
    """OOF logistic-regression AUROC over a fixed feature subset."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return float("nan")
    F = acts[:, feat_idx]
    folds = stratified_kfold(y, k, seed)
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        sc = StandardScaler().fit(F[train_mask])
        Xtr = sc.transform(F[train_mask]); Xte = sc.transform(F[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, y[train_mask])
        oof[test_idx] = clf.predict_proba(Xte)[:, 1]
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    dom_auroc = auroc(dom_score, y)

    h = EXPANSION * X.shape[1]
    acts, info = train_topk_sae(X, h, TOPK, SEED)

    # ---- Contrast 1: correct vs incorrect ----
    pos_mask = y
    neg_mask = ~y
    tada_corr = tada_scores(acts, pos_mask, neg_mask)

    # ---- Contrast 2: D-bucket vs A-bucket (difficulty proxied by seq_len) ----
    # Longer prefill ≈ harder problem. Quartiles A(easiest)..D(hardest).
    q = np.quantile(seq_len, [0.25, 0.5, 0.75])
    a_bucket = seq_len <= q[0]          # shortest → easiest proxy
    d_bucket = seq_len > q[2]           # longest  → hardest proxy
    tada_diff = tada_scores(acts, d_bucket, a_bucket)

    # Per-feature signed AUROC vs correctness.
    feat_auroc = np.array([signed_auroc(acts[:, f], y) for f in range(h)])
    feat_auroc = np.nan_to_num(feat_auroc, nan=0.5)

    # Best single feature by raw correctness AUROC.
    best_idx = int(np.argmax(feat_auroc))
    best_auroc = float(feat_auroc[best_idx])

    # Top-N features by |TADA correctness contrast|.
    order_corr = np.argsort(-np.abs(tada_corr))
    top_corr = order_corr[:TOP_N]
    top_corr_report = [
        {
            "feature": int(f),
            "tada_score": float(tada_corr[f]),
            "single_feat_auroc": float(feat_auroc[f]),
            "df": int((acts[:, f] > 0).sum()),
        }
        for f in top_corr
    ]

    # Top-N features by |TADA difficulty contrast|.
    order_diff = np.argsort(-np.abs(tada_diff))
    top_diff = order_diff[:TOP_N]
    top_diff_report = [
        {
            "feature": int(f),
            "tada_score": float(tada_diff[f]),
            "single_feat_auroc_vs_correct": float(feat_auroc[f]),
            "df": int((acts[:, f] > 0).sum()),
        }
        for f in top_diff
    ]

    # Multi-feature OOF logistic on the top-N TADA correctness features.
    multi_auroc = multi_feature_oof_auroc(acts, y, top_corr, N_FOLDS, SEED)

    # Verdicts.
    single_beats_dom = bool(best_auroc > L19_DOM_AUROC)
    multi_beats_dom = bool(not np.isnan(multi_auroc) and multi_auroc > L19_DOM_AUROC)
    multi_beats_coe60 = bool(not np.isnan(multi_auroc) and multi_auroc > COE60_AUROC)
    multi_beats_both = bool(multi_beats_dom and multi_beats_coe60)

    out = {
        "experiment": "P11-FE689",
        "sae_config": {
            "expansion": EXPANSION,
            "dict_size": h,
            "topk": TOPK,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
            "seed": SEED,
        },
        "sae_training": info,
        "reference_baselines": {
            "l19_dom_auroc_recomputed": float(dom_auroc),
            "l19_dom_auroc_canonical": L19_DOM_AUROC,
            "coe60_auroc_canonical": COE60_AUROC,
        },
        "best_single_feature": {
            "feature": best_idx,
            "auroc": best_auroc,
            "df": int((acts[:, best_idx] > 0).sum()),
        },
        "top_correctness_features_tada": top_corr_report,
        "top_difficulty_features_tada": top_diff_report,
        "multi_feature_oof_auroc": float(multi_auroc),
        "verdicts": {
            "single_feature_beats_l19_dom": single_beats_dom,
            "multi_feature_beats_l19_dom": multi_beats_dom,
            "multi_feature_beats_coe60": multi_beats_coe60,
            "multi_feature_beats_both_F9_dissolves": multi_beats_both,
            "interpretation": (
                "single_direction_framing_wrong" if single_beats_dom
                else "F9_equivalence_dissolves" if multi_beats_both
                else "single_direction_framing_survives"
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())