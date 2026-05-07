"""FE16 — Marchenko-Pastur bias-corrected participation ratio.

Implements the gamma_row finite-sample correction from Chun et al. 2509.26560
for PR in the n<d regime. Compares naive vs corrected PR for the full sample
and per-class/per-length-band splits.

Output: pathway11_h100/results/fe16_mp_bias_pr.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe16_mp_bias_pr.json"


def naive_pr(X: np.ndarray) -> tuple[float, np.ndarray]:
    """Returns (PR, eigenvalues) from centered Gram matrix."""
    X_c = X - X.mean(axis=0)
    G = X_c @ X_c.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, 0)
    sum_sq = (eigvals ** 2).sum()
    if sum_sq < 1e-30:
        return 0.0, eigvals
    pr = float(eigvals.sum() ** 2 / sum_sq)
    return pr, eigvals


def mp_corrected_pr(X: np.ndarray) -> dict:
    """Marchenko-Pastur bias-corrected PR via Chun et al. gamma_row.

    In the P < Q regime (samples < features), the naive PR overestimates
    the population PR due to eigenvalue spreading. The gamma_row correction
    analytically removes this bias.
    """
    P, Q = X.shape  # P=samples, Q=features
    X_c = X - X.mean(axis=0)

    # Naive PR
    pr_naive, eigvals = naive_pr(X)

    if P < 3:
        return {"pr_naive": pr_naive, "pr_corrected": float("nan"),
                "gamma_row": float("nan"), "aspect_ratio": P / Q, "P": P, "Q": Q}

    # Compute quantities for gamma_row
    # Phi = centered data (P, Q)
    Phi = X_c

    # ||Phi||_F^2 = sum of all squared entries = trace(Phi @ Phi.T)
    phi_fro_sq = float((Phi ** 2).sum())

    # ||Phi^T Phi||_F^2 = trace((Phi^T Phi)^2) = sum of squared eigenvalues of Phi^T Phi
    # Since Phi^T Phi is (Q, Q) and large, use: ||Phi^T Phi||_F^2 = ||Phi Phi^T||_F^2
    # because singular values are the same.
    # Gram = Phi @ Phi.T (P, P) — much smaller
    G = Phi @ Phi.T
    # ||Phi^T Phi||_F^2 = trace(G @ G) (since trace((AB)^T(AB)) = trace(B^T A^T A B))
    # Actually: ||Phi^T Phi||_F = ||Phi Phi^T||_F only for singular values.
    # trace((Phi^T Phi)^2) = trace(Phi^T Phi Phi^T Phi) = trace(G^2) when Phi is (P,Q)
    # NO: trace((Phi^T Phi)^2) = sum sigma_i^4 = trace(G^2) since G eigenvalues = Phi^T Phi eigenvalues
    gram_sq_trace = float(np.trace(G @ G))  # = ||Phi^T Phi||_F^2 in terms of eigenvalues

    # gamma_row formula from Chun et al. Appendix D (simplified for centered data):
    # gamma_row = [ (||Phi^T Phi||_F^2 * P - ||Phi||_F^4) / (P - 1) ] /
    #             [ (||Phi||_F^4 * Q - ||Phi^T Phi||_F^2 * P) / ((P-1)*(Q-1)) ]
    numerator = (gram_sq_trace * P - phi_fro_sq ** 2) / (P - 1)
    denominator = (phi_fro_sq ** 2 * Q - gram_sq_trace * P) / ((P - 1) * (Q - 1))

    if abs(denominator) < 1e-30:
        gamma_row = float("nan")
        pr_corrected = pr_naive
    else:
        gamma_row = numerator / denominator

        # Corrected PR: PR_corrected = PR_naive * (1 + gamma_row / (P-1))^{-1}
        # This removes the upward bias from finite-sample eigenvalue spreading
        correction_factor = 1 + gamma_row / (P - 1)
        if correction_factor <= 0:
            pr_corrected = float("nan")
        else:
            pr_corrected = pr_naive / correction_factor

    return {
        "pr_naive": float(pr_naive),
        "pr_corrected": float(pr_corrected),
        "gamma_row": float(gamma_row),
        "aspect_ratio": float(P / Q),
        "P": int(P),
        "Q": int(Q),
        "phi_fro_sq": float(phi_fro_sq),
        "gram_sq_trace": float(gram_sq_trace),
    }


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    seq_len = cache["seq_len"].astype(np.float64)
    n = X.shape[0]

    print("Computing MP-corrected PR for all groups...")

    # Full sample
    res_all = mp_corrected_pr(X)
    print(f"All (n={n}): naive={res_all['pr_naive']:.2f}  corrected={res_all['pr_corrected']:.2f}  "
          f"gamma={res_all['gamma_row']:.4f}")

    # Per class
    res_correct = mp_corrected_pr(X[correct])
    res_incorrect = mp_corrected_pr(X[~correct])
    print(f"Correct (n={correct.sum()}): naive={res_correct['pr_naive']:.2f}  "
          f"corrected={res_correct['pr_corrected']:.2f}")
    print(f"Incorrect (n={(~correct).sum()}): naive={res_incorrect['pr_naive']:.2f}  "
          f"corrected={res_incorrect['pr_corrected']:.2f}")

    # Per-length band
    p25 = np.percentile(seq_len, 25)
    p75 = np.percentile(seq_len, 75)
    bands = {
        "short": seq_len < p25,
        "medium": (seq_len >= p25) & (seq_len <= p75),
        "long": seq_len > p75,
    }
    band_results = {}
    for band_name, mask in bands.items():
        X_band = X[mask]
        y_band = correct[mask]
        res_band = mp_corrected_pr(X_band)

        res_band_correct = mp_corrected_pr(X_band[y_band]) if y_band.sum() >= 5 else None
        res_band_incorrect = mp_corrected_pr(X_band[~y_band]) if (~y_band).sum() >= 5 else None

        band_results[band_name] = {
            "all": res_band,
            "correct": res_band_correct,
            "incorrect": res_band_incorrect,
        }
        print(f"  {band_name} (n={mask.sum()}): naive={res_band['pr_naive']:.2f}  "
              f"corrected={res_band['pr_corrected']:.2f}")

    out = {
        "experiment": "FE16",
        "description": "Marchenko-Pastur bias-corrected participation ratio (Chun et al.)",
        "n": n,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "all_samples": res_all,
        "correct_only": res_correct,
        "incorrect_only": res_incorrect,
        "per_length_band": band_results,
        "pr_naive_reference": 19.86,
        "cross_check": {
            "naive_pr_matches": abs(res_all["pr_naive"] - 19.86) < 0.1,
        },
        "interpretation": (
            "If corrected PR > naive PR, finite-sample bias was deflating the "
            "estimate. If corrected ≈ naive, the sample is large enough that "
            "MP corrections are negligible. Per-class differences in corrected PR "
            "indicate genuine dimensionality asymmetry (F-4)."
        ),
        "meta": {
            "cache": str(CACHE),
            "method": "marchenko_pastur_gamma_row_correction",
            "reference": "Chun et al. 2509.26560 Appendix D",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
