"""FE01172B — Signal-channel rank at L19 via Gram matrix eigenspectrum.

Computes Gram matrix eigenspectrum of L19 prefill activations, estimates
effective rank via Marchenko-Pastur (MP) upper edge with iterative
noise-level estimation. Compares to cov-spectrum probe dim (20) and
full dim (1536).

Output: pathway11_h100/results/fe01172B_gram_eigenspectrum.json
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe01172B_gram_eigenspectrum.json"


def mp_upper_edge(sigma2: float, n: int, p: int) -> float:
    return sigma2 * (1 + np.sqrt(p / n)) ** 2


def estimate_noise_iterative(eigvals: np.ndarray, n: int, p: int,
                              max_iter: int = 20, tol: float = 1e-6) -> float:
    sigma2 = float(np.median(eigvals) / n)
    for _ in range(max_iter):
        edge = mp_upper_edge(sigma2, n, p)
        bulk = eigvals[eigvals <= edge]
        if len(bulk) == 0:
            break
        new_sigma2 = float(bulk.mean() / n)
        if abs(new_sigma2 - sigma2) < tol * sigma2:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2
    return sigma2


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"]
    n, p = X.shape

    X = X - X.mean(axis=0)

    G = X @ X.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]
    eigvals_pos = eigvals[eigvals > 0]

    sigma2 = estimate_noise_iterative(eigvals_pos, n, p)
    edge = mp_upper_edge(sigma2, n, p)
    effective_rank = int((eigvals_pos > edge).sum())

    total_var = float(eigvals_pos.sum())
    signal_var = float(eigvals_pos[eigvals_pos > edge].sum())
    noise_var = float(eigvals_pos[eigvals_pos <= edge].sum())
    signal_frac = signal_var / total_var if total_var > 0 else 0.0

    top20_var = float(eigvals_pos[:20].sum()) / total_var if total_var > 0 else 0.0
    top50_var = float(eigvals_pos[:50].sum()) / total_var if total_var > 0 else 0.0

    # Per-class effective rank
    X_corr = X[correct] - X[correct].mean(axis=0)
    X_incorr = X[~correct] - X[~correct].mean(axis=0)

    G_corr = X_corr @ X_corr.T
    eig_corr = np.sort(np.linalg.eigvalsh(G_corr))[::-1]
    eig_corr_pos = eig_corr[eig_corr > 0]
    s2_corr = estimate_noise_iterative(eig_corr_pos, X_corr.shape[0], p)
    rank_corr = int((eig_corr_pos > mp_upper_edge(s2_corr, X_corr.shape[0], p)).sum())

    G_incorr = X_incorr @ X_incorr.T
    eig_incorr = np.sort(np.linalg.eigvalsh(G_incorr))[::-1]
    eig_incorr_pos = eig_incorr[eig_incorr > 0]
    s2_incorr = estimate_noise_iterative(eig_incorr_pos, X_incorr.shape[0], p)
    rank_incorr = int((eig_incorr_pos > mp_upper_edge(s2_incorr, X_incorr.shape[0], p)).sum())

    # Participation ratio
    pr_all = float(eigvals_pos.sum() ** 2 / (eigvals_pos ** 2).sum())

    out = {
        "experiment": "FE01172B",
        "description": "Gram eigenspectrum effective rank at L19",
        "n": n,
        "p": p,
        "centered": True,
        "noise_sigma2": float(sigma2),
        "mp_upper_edge": float(edge),
        "effective_rank": effective_rank,
        "participation_ratio": pr_all,
        "total_variance": total_var,
        "signal_variance": signal_var,
        "noise_variance": noise_var,
        "signal_fraction": signal_frac,
        "top20_variance_fraction": top20_var,
        "top50_variance_fraction": top50_var,
        "correct_effective_rank": rank_corr,
        "incorrect_effective_rank": rank_incorr,
        "eigenvalue_spectrum_top50": eigvals_pos[:50].tolist(),
        "meta": {
            "cache": str(CACHE),
            "seed": 9999,
            "method": "iterative_MP_noise_estimation",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"effective_rank={effective_rank}")
    print(f"participation_ratio={pr_all:.2f}")
    print(f"top20_variance_fraction={top20_var:.4f}")
    print(f"signal_fraction={signal_frac:.4f}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
