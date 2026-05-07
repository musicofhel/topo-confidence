"""FE925 — Hyvärinen score difference for correctness prediction.

Gaussian baseline (= QDA via Ledoit-Wolf shrinkage) + sliced score matching
primary (200 slices, captures non-Gaussian moments).

Output: pathway11_h100/results/fe925_hyvarinen_score.json
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
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gaussian_kde

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe925_hyvarinen_score.json"

SEED = 9999
N_FOLDS = 5
N_SLICES = 200
PCA_DIM = 50


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def gaussian_score(x: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Gaussian score function: -Σ^{-1}(x - μ). Returns (n, d)."""
    return -(x - mu) @ precision


def hyvarinen_score_gaussian(x: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Per-sample Hyvärinen score under Gaussian model.
    H(x) = 0.5 ||s(x)||^2 + tr(ds/dx) = 0.5 ||Σ^{-1}(x-μ)||^2 - tr(Σ^{-1})
    """
    s = gaussian_score(x, mu, precision)
    return 0.5 * np.sum(s ** 2, axis=1) - np.trace(precision)


def sliced_score_1d(projections: np.ndarray, bw_method: str = "scott") -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D score and Laplacian via KDE for a set of projected values.
    Returns (score, laplacian) arrays of same shape as projections.
    """
    n = len(projections)
    kde = gaussian_kde(projections, bw_method=bw_method)
    bw = kde.factor * projections.std()

    # Score = d log p / dx = p'/p
    # For Gaussian KDE: p(x) = (1/n) sum K_h(x - x_i)
    # p'(x) = -(1/n) sum (x - x_i)/h^2 * K_h(x - x_i)
    # Laplacian = p''/p - (p'/p)^2
    eps = 1e-30
    diff = projections[:, None] - projections[None, :]  # (n, n)
    weights = np.exp(-0.5 * (diff / bw) ** 2)  # unnormalized kernel
    p = weights.sum(axis=1) + eps
    p_prime = -(diff / bw ** 2 * weights).sum(axis=1)
    p_double_prime = (((diff / bw ** 2) ** 2 - 1 / bw ** 2) * weights).sum(axis=1)

    score = p_prime / p
    laplacian = p_double_prime / p - (p_prime / p) ** 2

    return score, laplacian


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    n, d = X.shape
    X = X - X.mean(axis=0)

    X_corr = X[correct]
    X_incorr = X[~correct]

    # --- Gaussian baseline (QDA with Ledoit-Wolf) ---
    print("Computing Gaussian baseline (QDA with Ledoit-Wolf)...")
    lw_corr = LedoitWolf().fit(X_corr)
    lw_incorr = LedoitWolf().fit(X_incorr)

    mu_c, prec_c = lw_corr.location_, lw_corr.precision_
    mu_i, prec_i = lw_incorr.location_, lw_incorr.precision_

    H_corr_model = hyvarinen_score_gaussian(X, mu_c, prec_c)
    H_incorr_model = hyvarinen_score_gaussian(X, mu_i, prec_i)
    # Lower H = better fit. Correct samples have lower H under correct model.
    # Negate so positive diff → more likely correct.
    gauss_diff = H_incorr_model - H_corr_model

    gauss_auroc = auroc(gauss_diff, correct)
    print(f"Gaussian (QDA) AUROC: {gauss_auroc:.4f}")

    # --- Sliced score matching (primary) ---
    print(f"Computing sliced score matching ({N_SLICES} slices)...")
    rng = np.random.default_rng(SEED)
    directions = rng.standard_normal((N_SLICES, d)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Project data onto random directions
    sliced_H_diff = np.zeros(n, dtype=np.float64)

    for s_idx in range(N_SLICES):
        v = directions[s_idx]
        proj_all = X @ v  # (n,)
        proj_corr = X_corr @ v
        proj_incorr = X_incorr @ v

        # 1D score + Laplacian for each class model
        score_c, lap_c = sliced_score_1d(proj_corr)
        score_i, lap_i = sliced_score_1d(proj_incorr)

        # For each sample, compute 1D Hyvärinen score diff under each class model
        # Need to evaluate score at all n points under each class model
        # Use KDE from each class
        kde_c = gaussian_kde(proj_corr, bw_method="scott")
        kde_i = gaussian_kde(proj_incorr, bw_method="scott")

        bw_c = kde_c.factor * proj_corr.std()
        bw_i = kde_i.factor * proj_incorr.std()

        # Evaluate score at all n points under correct-class model
        for x_idx in range(n):
            x_val = proj_all[x_idx]

            diff_c = x_val - proj_corr
            w_c = np.exp(-0.5 * (diff_c / bw_c) ** 2)
            p_c = w_c.sum() + 1e-30
            pp_c = -(diff_c / bw_c ** 2 * w_c).sum()
            ppp_c = (((diff_c / bw_c ** 2) ** 2 - 1 / bw_c ** 2) * w_c).sum()
            sc_c = pp_c / p_c
            lap_c_val = ppp_c / p_c - sc_c ** 2
            h_c = 0.5 * sc_c ** 2 + lap_c_val

            diff_i = x_val - proj_incorr
            w_i = np.exp(-0.5 * (diff_i / bw_i) ** 2)
            p_i = w_i.sum() + 1e-30
            pp_i = -(diff_i / bw_i ** 2 * w_i).sum()
            ppp_i = (((diff_i / bw_i ** 2) ** 2 - 1 / bw_i ** 2) * w_i).sum()
            sc_i = pp_i / p_i
            lap_i_val = ppp_i / p_i - sc_i ** 2
            h_i = 0.5 * sc_i ** 2 + lap_i_val

            sliced_H_diff[x_idx] += h_i - h_c

        if (s_idx + 1) % 50 == 0:
            print(f"  Slices: {s_idx + 1}/{N_SLICES}")

    sliced_H_diff /= N_SLICES

    sliced_auroc = auroc(sliced_H_diff, correct)
    print(f"Sliced score AUROC: {sliced_auroc:.4f}")

    # --- OOF AUROC for both methods ---
    print("Computing OOF AUROCs...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    gauss_oof = np.zeros(n, dtype=np.float64)
    for train_idx, test_idx in skf.split(X, correct):
        corr_tr = correct[train_idx]
        X_tr = X[train_idx]
        lw_c = LedoitWolf().fit(X_tr[corr_tr])
        lw_i = LedoitWolf().fit(X_tr[~corr_tr])
        H_c = hyvarinen_score_gaussian(X[test_idx], lw_c.location_, lw_c.precision_)
        H_i = hyvarinen_score_gaussian(X[test_idx], lw_i.location_, lw_i.precision_)
        gauss_oof[test_idx] = H_i - H_c

    gauss_oof_auroc = auroc(gauss_oof, correct)

    out = {
        "experiment": "FE925",
        "description": "Hyvärinen score difference for correctness prediction",
        "n": n,
        "d": d,
        "centered": True,
        "gaussian_baseline_auroc_insample": float(gauss_auroc),
        "gaussian_baseline_auroc_oof": float(gauss_oof_auroc),
        "sliced_score_auroc_insample": float(sliced_auroc),
        "n_slices": N_SLICES,
        "dom_auroc_reference": 0.7731,
        "interpretation": (
            "Gaussian (QDA) captures second-order; sliced score captures higher-order. "
            "Gap between them = non-Gaussian contribution."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "cache": str(CACHE),
            "seed": SEED,
            "method": "hyvarinen_score_gaussian_baseline_plus_sliced_score_matching",
            "pca_dim": PCA_DIM,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nGaussian (QDA) AUROC in-sample: {gauss_auroc:.4f}")
    print(f"Gaussian (QDA) AUROC OOF: {gauss_oof_auroc:.4f}")
    print(f"Sliced score AUROC in-sample: {sliced_auroc:.4f}")
    print(f"DoM AUROC reference: 0.7731")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
