"""FE15 — Length-band participation ratio control.

Splits samples into Short/Medium/Long by seq_len percentiles, computes PR
within each band (overall and per-class), plus per-band DoM AUROC.
Cross-checks global PR against FE01172B = 19.86.

Output: pathway11_h100/results/fe15_length_band_pr.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe15_length_band_pr.json"

SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² from the Gram matrix eigenvalues."""
    X_c = X - X.mean(axis=0)
    G = X_c @ X_c.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, 0)
    sum_sq = (eigvals ** 2).sum()
    if sum_sq < 1e-30:
        return 0.0
    return float(eigvals.sum() ** 2 / sum_sq)


def dom_auroc_insample(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample DoM AUROC (use for small-n bands)."""
    X_c = X - X.mean(axis=0)
    dom = X_c[y].mean(0) - X_c[~y].mean(0)
    dom = dom / (np.linalg.norm(dom) + 1e-30)
    scores = X_c @ dom
    return auroc(scores, y)


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    seq_len = cache["seq_len"].astype(np.float64)
    n = X.shape[0]

    # Global PR cross-check
    pr_global = participation_ratio(X)
    print(f"Global PR = {pr_global:.2f} (expected ~19.86)")

    # Define bands
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
        n_band = int(mask.sum())
        n_correct_band = int(y_band.sum())
        n_incorrect_band = n_band - n_correct_band

        pr_all = participation_ratio(X_band)

        # Per-class PR (flag small-n)
        pr_correct = participation_ratio(X_band[y_band]) if n_correct_band >= 5 else float("nan")
        pr_incorrect = participation_ratio(X_band[~y_band]) if n_incorrect_band >= 5 else float("nan")

        # In-sample DoM AUROC for band
        dom_auc = dom_auroc_insample(X_band, y_band) if n_correct_band >= 3 and n_incorrect_band >= 3 else float("nan")

        band_results[band_name] = {
            "n": n_band,
            "n_correct": n_correct_band,
            "n_incorrect": n_incorrect_band,
            "seq_len_range": [float(seq_len[mask].min()), float(seq_len[mask].max())],
            "pr_all": float(pr_all),
            "pr_correct": float(pr_correct),
            "pr_incorrect": float(pr_incorrect),
            "dom_auroc_insample": float(dom_auc),
            "rank_bounded": n_band < 1536,
            "max_rank": min(n_band, 1536),
        }
        print(f"  {band_name}: n={n_band} (c={n_correct_band}/i={n_incorrect_band})  "
              f"PR={pr_all:.2f}  PR_c={pr_correct:.2f}  PR_i={pr_incorrect:.2f}  "
              f"DoM AUROC={dom_auc:.4f}")

    out = {
        "experiment": "FE15",
        "description": "Length-band PR control: does PR=20 survive length stratification?",
        "n": n,
        "n_correct": int(correct.sum()),
        "percentiles": {"p25": float(p25), "p75": float(p75)},
        "pr_global": float(pr_global),
        "pr_global_reference": 19.86,
        "cross_check": {
            "global_pr_matches": abs(pr_global - 19.86) < 0.1,
        },
        "band_results": band_results,
        "interpretation": (
            "If PR is roughly constant across bands, F-4's asymmetric collapse "
            "is NOT a length artifact. If PR varies systematically with length, "
            "the spectral structure may be confounded."
        ),
        "meta": {
            "cache": str(CACHE),
            "seed": SEED,
            "method": "gram_eigenvalue_participation_ratio_per_length_band",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
