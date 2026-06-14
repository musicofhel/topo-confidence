"""FE102 — Subspace-orthogonality test for F-3 via LEACE.

F-3 reports that the prefill-correctness direction (L19 prefill DoM) and the
final-token correctness direction are near-orthogonal (cos = 0.046). That rests
on a single-vector cosine. This script upgrades it to a rank-aware test:

  1. Per OOF fold, fit a LEACE eraser on Qwen-1.5B L19 *prefill* features for
     K=1 correctness — this deletes the entire prefill-correctness
     linear-readable subspace (rank-1 in whitened space for a binary concept).
  2. Apply that *prefill*-defined eraser to the *final-token* L19 features.
  3. Refit the final-token DoM on the erased final features and re-measure
     final-token DoM AUROC (raw ≈ 0.7186).

If F-3's orthogonal reading is correct, erasing the prefill subspace leaves the
final-token DoM AUROC essentially unchanged. A drop > 0.05 indicates a shared
rank-r linear subspace that the single-vector cos = 0.046 missed.
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

# Final-token L19 features are not in the documented prefill cache; they live in
# a pathway10 / pathway11 sibling NPZ. Try the known candidate locations and
# fall back to MISSING_REGEN_INPUT if none resolves to a (500, 1536) array.
FINAL_CANDIDATES = [
    (ROOT / "scratch/pathway10_final_l19.npz", ("final", "final_hidden", "hidden", "h", "X")),
    (ROOT / "scratch/pathway10_temporal_and_verifier.npz", ("final", "final_hidden", "final_l19")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz", ("final", "final_hidden", "hidden")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz", ("final", "final_hidden", "final_l19")),
]

OUT_JSON = ROOT / "pathway11_h100/leace_subspace_orthogonality/results.json"

N_FOLDS = 5
SEED = 9999
DROP_THRESHOLD = 0.05


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def fit_leace(X_train: np.ndarray, y_train: np.ndarray, alpha_rel: float = 1e-3):
    """Fit a rank-1 LEACE eraser for a binary concept.

    Returns (mu, d_vec, w): mu is the train mean, d_vec the raw mean-difference
    (concept direction), w the whitened direction Sigma^{-1} d_vec.
    """
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = alpha_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    mu_pos = X_train[y_train].mean(axis=0)
    mu_neg = X_train[~y_train].mean(axis=0)
    d_vec = mu_pos - mu_neg
    w = np.linalg.solve(Sigma_ridge, d_vec)
    return mu, d_vec, w


def apply_leace(X: np.ndarray, mu: np.ndarray, d_vec: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Oblique projection removing the d_vec subspace defined by the eraser."""
    denom = float(d_vec @ w)
    if abs(denom) < 1e-12:
        return X.copy()
    coeff = ((X - mu) @ w) / denom
    return X - np.outer(coeff, d_vec)


def dom_scores(X_train, y_train, X_test):
    """Refit a DoM direction on (X_train, y_train), score X_test."""
    d = X_train[y_train].mean(axis=0) - X_train[~y_train].mean(axis=0)
    return X_test @ d


def load_final_features() -> np.ndarray | None:
    for path, keys in FINAL_CANDIDATES:
        if not path.exists():
            continue
        try:
            blob = np.load(path)
        except Exception:
            continue
        for key in keys:
            if key in blob.files:
                arr = np.asarray(blob[key])
                if arr.shape == (500, 1536):
                    return arr.astype(np.float64)
        # Fall back: any (500, 1536) array in the file that is not the prefill.
        for key in blob.files:
            if key in ("prefill",):
                continue
            arr = np.asarray(blob[key])
            if arr.shape == (500, 1536):
                return arr.astype(np.float64)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X_pre = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X_pre.shape == (500, 1536) and y.shape == (500,)

    X_fin = load_final_features()
    if X_fin is None:
        tried = ", ".join(str(p) for p, _ in FINAL_CANDIDATES)
        print("MISSING_REGEN_INPUT final-token L19 features (tried:", tried, ")", file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)

    final_raw = np.zeros(n, dtype=np.float64)      # raw final-token DoM (OOF)
    final_erased = np.zeros(n, dtype=np.float64)    # prefill-erased final DoM (OOF)
    prefill_raw = np.zeros(n, dtype=np.float64)     # raw prefill DoM (OOF) — sanity
    prefill_erased = np.zeros(n, dtype=np.float64)  # erased prefill DoM (OOF) — sanity

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        Xp_tr, Xp_te = X_pre[train_mask], X_pre[test_idx]
        Xf_tr, Xf_te = X_fin[train_mask], X_fin[test_idx]
        ytr = y[train_mask]

        # Eraser fit on prefill correctness.
        mu, d_vec, w = fit_leace(Xp_tr, ytr)

        # Sanity: raw + erased prefill DoM.
        prefill_raw[test_idx] = dom_scores(Xp_tr, ytr, Xp_te)
        Xp_tr_e = apply_leace(Xp_tr, mu, d_vec, w)
        Xp_te_e = apply_leace(Xp_te, mu, d_vec, w)
        prefill_erased[test_idx] = dom_scores(Xp_tr_e, ytr, Xp_te_e)

        # Main test: apply the *prefill* eraser to *final* features, refit final DoM.
        final_raw[test_idx] = dom_scores(Xf_tr, ytr, Xf_te)
        Xf_tr_e = apply_leace(Xf_tr, mu, d_vec, w)
        Xf_te_e = apply_leace(Xf_te, mu, d_vec, w)
        final_erased[test_idx] = dom_scores(Xf_tr_e, ytr, Xf_te_e)

    auroc_final_raw = auroc(final_raw, y)
    auroc_final_erased = auroc(final_erased, y)
    delta_final = auroc_final_erased - auroc_final_raw

    out = {
        "experiment": "P11-FE102",
        "description": "LEACE prefill-subspace erasure applied to final-token features; F-3 rank-aware orthogonality test.",
        "n": int(n),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_final_raw_oof": float(auroc_final_raw),
        "auroc_final_erased_oof": float(auroc_final_erased),
        "delta_final_auroc": float(delta_final),
        "auroc_prefill_raw_oof": float(auroc(prefill_raw, y)),
        "auroc_prefill_erased_oof": float(auroc(prefill_erased, y)),
        "drop_threshold": DROP_THRESHOLD,
        "shared_subspace_detected": bool(delta_final < -DROP_THRESHOLD),
        "interpretation": (
            "If |delta| small, prefill and final correctness subspaces are "
            "orthogonal (F-3 holds at rank level). delta < -0.05 indicates a "
            "shared linear subspace the single-vector cos=0.046 missed."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())