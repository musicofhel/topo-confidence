"""FE753 — LI-at-L19 vs LI-all-layers (Layerwise CP eq 6-8) tie-break for F-9.

Implements the LM-head-lens "usable information" recipe from Layerwise CP
(Kim et al. 2025a): for each transformer layer ℓ, take the cached prefill
residual h_ℓ, apply the model's final RMSNorm, project through the tied
unembedding (logit lens) to a vocab distribution p_ℓ, and read its entropy
H_ℓ (eq 6-7). Per-layer information gain is the entropy reduction
LI_ℓ = H_{ℓ-1} - H_ℓ (eq 8).

We then score MATH-500 correctness ranking by AUROC under two readouts:
  * LI restricted to ℓ=19 (LI_19 = H_18 - H_19), the single-layer L19 story;
  * LI aggregated over all layers — both the faithful summed reduction
    (telescopes to H_0 - H_last) and an OOF logistic over the full per-layer
    entropy vector, which is the genuine multi-layer Layerwise CP aggregate.

If LI-all >> LI-L19, F-9 (CoE-60 redundant with single-layer L19) is
contradicted; if they match, F-9 survives the cleanest available tie-break.

Requires an all-layers prefill cache + the tied unembedding, neither of which
is part of the L19-only headline cache. If absent, prints MISSING_REGEN_INPUT.
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
ALL_LAYERS = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz"
UNEMBED = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_unembed.npz"
OUT_JSON = ROOT / "pathway11_h100/layerwise_li/results.json"

SEED = 9999
N_FOLDS = 5
L19 = 19
RMS_EPS = 1e-6


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _first_key(blob, names):
    for n in names:
        if n in blob.files:
            return blob[n]
    return None


def rmsnorm(h: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
    """Qwen2.5 final RMSNorm: h / sqrt(mean(h^2)+eps) * weight."""
    var = np.mean(h * h, axis=-1, keepdims=True)
    out = h / np.sqrt(var + RMS_EPS)
    if weight is not None:
        out = out * weight[None, :]
    return out


def layer_entropy(h: np.ndarray, W_U: np.ndarray, norm_w: np.ndarray | None) -> np.ndarray:
    """Logit-lens entropy of each row's vocab distribution (stable, per-layer)."""
    x = rmsnorm(h.astype(np.float32), norm_w)
    logits = x @ W_U  # (n, V)
    logits -= logits.max(axis=1, keepdims=True)
    np.exp(logits, out=logits)
    Z = logits.sum(axis=1, keepdims=True)
    p = logits / Z
    logp = np.log(p + 1e-12)
    return -(p * logp).sum(axis=1)


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def oof_logistic_auroc(X: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return float("nan")
    oof = np.zeros(len(y), dtype=np.float64)
    for test_idx in stratified_kfold(y, N_FOLDS, SEED):
        train_mask = np.ones(len(y), dtype=bool); train_mask[test_idx] = False
        mu = X[train_mask].mean(axis=0)
        sd = X[train_mask].std(axis=0) + 1e-8
        Xtr = (X[train_mask] - mu) / sd
        Xte = (X[test_idx] - mu) / sd
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, y[train_mask])
        oof[test_idx] = clf.decision_function(Xte)
    return auroc(oof, y)


def main() -> int:
    for required in (ALL_LAYERS, UNEMBED):
        if not required.exists():
            print("MISSING_REGEN_INPUT", required, file=sys.stderr)
            return 2

    # correctness labels — prefer all-layers blob, fall back to L19 cache.
    al = np.load(ALL_LAYERS)
    hidden = _first_key(al, ["hidden", "hidden_states", "prefill_all", "residuals"])
    if hidden is None:
        print("MISSING_REGEN_INPUT", ALL_LAYERS, "(no hidden key)", file=sys.stderr)
        return 2
    hidden = np.asarray(hidden)
    if hidden.ndim != 3:
        print("MISSING_REGEN_INPUT", ALL_LAYERS, "(expected (N, L, D))", file=sys.stderr)
        return 2
    n, n_layers, d = hidden.shape
    if n_layers <= L19 or d != 1536:
        print("MISSING_REGEN_INPUT", ALL_LAYERS, f"(shape {hidden.shape})", file=sys.stderr)
        return 2

    y = _first_key(al, ["correct"])
    if y is None:
        if not CACHE.exists():
            print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
            return 2
        y = np.load(CACHE)["correct"]
    y = np.asarray(y).astype(bool)
    if y.shape[0] != n:
        print("MISSING_REGEN_INPUT", "label/hidden length mismatch", file=sys.stderr)
        return 2

    ue = np.load(UNEMBED)
    W_U = _first_key(ue, ["W_U", "unembed", "lm_head", "lm_head_weight", "embed_tokens"])
    if W_U is None:
        print("MISSING_REGEN_INPUT", UNEMBED, "(no unembed key)", file=sys.stderr)
        return 2
    W_U = np.asarray(W_U, dtype=np.float32)
    # want (D, V); transpose an (V, D) lm_head matrix.
    if W_U.shape[0] != d and W_U.shape[1] == d:
        W_U = W_U.T
    if W_U.shape[0] != d:
        print("MISSING_REGEN_INPUT", UNEMBED, f"(unembed shape {W_U.shape})", file=sys.stderr)
        return 2
    norm_w = _first_key(ue, ["ln_f", "norm_weight", "final_norm", "model_norm_weight"])
    if norm_w is not None:
        norm_w = np.asarray(norm_w, dtype=np.float32).reshape(-1)
        if norm_w.shape[0] != d:
            norm_w = None

    # Per-layer logit-lens entropy, processed one layer at a time (memory).
    H = np.zeros((n, n_layers), dtype=np.float64)
    for ell in range(n_layers):
        H[:, ell] = layer_entropy(hidden[:, ell, :], W_U, norm_w)

    # LI_ℓ = H_{ℓ-1} - H_ℓ (entropy reduction). Higher reduction -> more
    # confident -> more likely correct, so score = LI directly.
    li_l19 = H[:, L19 - 1] - H[:, L19]
    li_all_sum = H[:, 0] - H[:, n_layers - 1]  # telescoped total reduction
    neg_entropy_l19 = -H[:, L19]

    auroc_li_l19 = auroc(li_l19, y)
    auroc_li_all_sum = auroc(li_all_sum, y)
    auroc_neg_entropy_l19 = auroc(neg_entropy_l19, y)
    # Genuine multi-layer Layerwise CP aggregate: full per-layer entropy vector.
    auroc_li_all_logistic = oof_logistic_auroc(H, y)
    # Per-layer LI matrix as features (information-gain readout).
    LI = np.zeros((n, n_layers), dtype=np.float64)
    LI[:, 1:] = H[:, :-1] - H[:, 1:]
    auroc_li_all_logistic_gain = oof_logistic_auroc(LI[:, 1:], y)

    best_all = max(
        v for v in (auroc_li_all_sum, auroc_li_all_logistic, auroc_li_all_logistic_gain)
        if not np.isnan(v)
    )
    delta = best_all - auroc_li_l19 if not np.isnan(auroc_li_l19) else float("nan")
    # "LI-all >> LI-L19" => F-9 contradicted. Use a 0.02 AUROC margin.
    f9_contradicted = bool(not np.isnan(delta) and delta >= 0.02)

    out = {
        "experiment": "FE753",
        "n": int(n),
        "n_layers": int(n_layers),
        "l19_index": L19,
        "applied_final_norm": norm_w is not None,
        "auroc_li_l19": auroc_li_l19,
        "auroc_neg_entropy_l19": auroc_neg_entropy_l19,
        "auroc_li_all_sum": auroc_li_all_sum,
        "auroc_li_all_logistic_entropy": auroc_li_all_logistic,
        "auroc_li_all_logistic_gain": auroc_li_all_logistic_gain,
        "best_li_all": float(best_all),
        "delta_all_minus_l19": float(delta),
        "f9_contradicted": f9_contradicted,
        "verdict": (
            "F-9 CONTRADICTED: all-layer LI aggregation beats L19-only"
            if f9_contradicted
            else "F-9 SURVIVES: all-layer LI does not exceed single-layer L19"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())