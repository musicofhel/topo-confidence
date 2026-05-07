"""FE919 — Diffusion Maps embedding for correctness prediction.

Builds renormalized diffusion kernel from L19 prefill activations,
eigendecomposes, uses diffusion coordinates as features for logistic
regression AUROC at k in {2, 5, 10, 20}.

Output: pathway11_h100/results/fe919_diffusion_maps.json
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe919_diffusion_maps.json"

K_VALUES = [2, 5, 10, 20]
SEED = 9999
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    n = X.shape[0]

    print("Computing pairwise distances...")
    dists = squareform(pdist(X, metric="euclidean"))  # (500, 500)

    # Gaussian kernel with epsilon = median distance
    epsilon = float(np.median(dists[np.triu_indices(n, k=1)]))
    K = np.exp(-dists ** 2 / (2 * epsilon ** 2))

    # Density debiasing (renormalized Laplacian, alpha=1)
    d = K.sum(axis=1)
    K_tilde = K / np.outer(d, d)
    d_tilde = K_tilde.sum(axis=1)
    P = K_tilde / d_tilde[:, None]

    # Eigendecompose transition matrix
    print("Eigendecomposing diffusion operator...")
    eigvals, eigvecs = np.linalg.eigh(P + P.T)  # symmetrize for stability
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Skip first trivial eigenvector (constant)
    dm_eigvals = eigvals[1:]
    dm_eigvecs = eigvecs[:, 1:]

    max_k = max(K_VALUES)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results_per_k = {}
    for k in K_VALUES:
        coords = dm_eigvecs[:, :k]  # (n, k) diffusion coordinates

        oof_scores = np.zeros(n, dtype=np.float64)
        for train_idx, test_idx in skf.split(X, correct):
            lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
            lr.fit(coords[train_idx], correct[train_idx])
            oof_scores[test_idx] = lr.predict_proba(coords[test_idx])[:, 1]

        dm_auroc = auroc(oof_scores, correct)

        # Spectral gap after k-th component
        spectral_gap = float(dm_eigvals[k - 1] - dm_eigvals[k]) if k < len(dm_eigvals) else 0.0

        results_per_k[str(k)] = {
            "k": k,
            "auroc_oof": float(dm_auroc),
            "eigenvalues": dm_eigvals[:k].tolist(),
            "spectral_gap_after_k": spectral_gap,
        }
        print(f"k={k}: AUROC={dm_auroc:.4f}")

    # DoM baseline in diffusion coords
    dom = X[correct].mean(0) - X[~correct].mean(0)
    dom_norm = dom / (np.linalg.norm(dom) + 1e-30)

    out = {
        "experiment": "FE919",
        "description": "Diffusion Maps embedding for correctness prediction",
        "n": n,
        "epsilon": epsilon,
        "centered": False,
        "results_per_k": results_per_k,
        "dom_auroc_reference": 0.7731,
        "top20_eigenvalues": dm_eigvals[:20].tolist(),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "interpretation": (
            "DM AUROC > DoM => nonlinear manifold structure encodes correctness "
            "beyond the linear DoM direction"
        ),
        "meta": {
            "cache": str(CACHE),
            "seed": SEED,
            "method": "diffusion_maps_renormalized_laplacian_logistic_regression",
            "kernel": "gaussian",
            "density_debiasing_alpha": 1,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nDoM AUROC reference: 0.7731")
    print(f"Epsilon (median distance): {epsilon:.4f}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
