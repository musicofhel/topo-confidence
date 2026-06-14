"""FE1231 — Tucker/HOSVD co-skewness basis vs PCA basis, matched-rank correctness probe.

Tests the Marraffini et al. claim that a κ₃-maximizing (co-skewness) Tucker basis
beats the κ₂-maximizing PCA basis for downstream prediction. Pipeline:

  1. Center L19 prefill activations, PCA-reduce + whiten to a P_WORK-dim working
     space (covariance eigenvectors ordered by eigenvalue → the κ₂ / PCA basis).
  2. Form the third-order co-skewness tensor S_{ijk} = E[y_i y_j y_k] in the
     whitened space (Gaussian → S ≈ 0; non-Gaussian structure shows up here).
  3. HOSVD: SVD of the mode-1 unfolding gives the symmetric Tucker factor; its
     top-r left singular vectors are the κ₃-maximizing basis.
  4. At each matched rank r, project onto the PCA subspace (first r whitened
     coords) vs the Tucker subspace, fit an OOF mean-difference (DoM) probe, and
     compare AUROC.
  5. Gaussian-null control (F-10): compare the empirical co-skewness tensor norm
     to a surrogate null of Gaussian-whitened tensors (third cumulant = 0), and
     report the z-score — if the Tucker lift is real, ||S|| must exceed the null.

Unsupervised bases (PCA, co-skewness) are fit on full data; only the DoM probe
touches labels and is fit out-of-fold, so the AUROC comparison is leakage-free.
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
OUT_JSON = ROOT / "pathway11_h100/coskewness_tucker/results.json"

SEED = 9999
N_FOLDS = 5
P_WORK = 100          # PCA-reduced working dimension for the tensor
RANKS = [2, 4, 8, 16, 32, 64, 100]
N_SURROGATE = 50      # Gaussian-null surrogates for the co-skewness norm test
WHITEN_EPS = 1e-8


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
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


def coskewness_unfolding(Y: np.ndarray) -> np.ndarray:
    """Mode-1 unfolding of the co-skewness tensor S_{ijk} = mean_n y_i y_j y_k.

    Returns a (p, p*p) matrix. Built via the row-wise Khatri-Rao of Y with Y.
    """
    n, p = Y.shape
    KR = (Y[:, :, None] * Y[:, None, :]).reshape(n, p * p)
    return (Y.T @ KR) / n


def oof_dom_auroc(F: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """Out-of-fold mean-difference (DoM) probe AUROC on feature matrix F."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Ftr, ytr = F[train_mask], y[train_mask]
        d = Ftr[ytr].mean(axis=0) - Ftr[~ytr].mean(axis=0)
        scores[test_idx] = F[test_idx] @ d
    return auroc(scores, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    n, d_full = X.shape
    p = min(P_WORK, d_full, n - 1)
    ranks = [r for r in RANKS if r <= p]

    # --- PCA reduction + whitening (covariance eigenbasis = the κ₂/PCA basis) ---
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / max(n - 1, 1)
    evals, evecs = np.linalg.eigh(Sigma)          # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order][:p]
    evecs = evecs[:, order][:, :p]
    evals = np.clip(evals, WHITEN_EPS, None)
    # Whitened PCA coordinates: column j = j-th covariance eigendirection.
    Y = (Xc @ evecs) / np.sqrt(evals)             # (n, p), identity covariance

    # --- Co-skewness tensor + HOSVD Tucker factor (κ₃-maximizing basis) ---
    S1 = coskewness_unfolding(Y)                  # (p, p*p)
    skew_norm = float(np.linalg.norm(S1))
    U_t, sing, _ = np.linalg.svd(S1, full_matrices=False)  # symmetric tensor → V = U_t

    # --- Gaussian-null control for F-10: norm of co-skewness under N(0, I) ---
    rng = np.random.default_rng(SEED)
    null_norms = np.empty(N_SURROGATE, dtype=np.float64)
    for s in range(N_SURROGATE):
        Yg = rng.standard_normal(size=(n, p))
        null_norms[s] = float(np.linalg.norm(coskewness_unfolding(Yg)))
    null_mean = float(null_norms.mean())
    null_std = float(null_norms.std(ddof=1))
    skew_z = float((skew_norm - null_mean) / null_std) if null_std > 0 else float("nan")

    # --- Matched-rank AUROC comparison: PCA subspace vs Tucker subspace ---
    folds = stratified_kfold(y, N_FOLDS, SEED)
    per_rank = []
    for r in ranks:
        F_pca = Y[:, :r]                          # top-r covariance directions
        F_tucker = Y @ U_t[:, :r]                 # top-r co-skewness directions
        a_pca = oof_dom_auroc(F_pca, y, folds)
        a_tucker = oof_dom_auroc(F_tucker, y, folds)
        per_rank.append({
            "rank": int(r),
            "auroc_pca": a_pca,
            "auroc_tucker": a_tucker,
            "delta_tucker_minus_pca": float(a_tucker - a_pca),
        })

    best = max(per_rank, key=lambda d: d["auroc_tucker"])

    out = {
        "experiment": "FE1231",
        "n": int(n),
        "p_work": int(p),
        "n_folds": N_FOLDS,
        "ranks": ranks,
        "per_rank": per_rank,
        "best_tucker": best,
        "max_delta_tucker_minus_pca": float(max(d["delta_tucker_minus_pca"] for d in per_rank)),
        "mean_delta_tucker_minus_pca": float(np.mean([d["delta_tucker_minus_pca"] for d in per_rank])),
        "coskewness_norm": skew_norm,
        "coskewness_top_singular": float(sing[0]),
        "gaussian_null_norm_mean": null_mean,
        "gaussian_null_norm_std": null_std,
        "coskewness_norm_z_vs_gaussian": skew_z,
        "n_surrogate": N_SURROGATE,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"FE1231 done: best Tucker AUROC {best['auroc_tucker']:.4f} @ rank {best['rank']}, "
          f"max delta {out['max_delta_tucker_minus_pca']:+.4f}, skew z={skew_z:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())