"""FE188 — LID-MLE local intrinsic dimension (Levina-Bickel estimator).

Computes per-sample local intrinsic dimension at multiple k, reports
mean LID per class, LID-as-correctness AUROC, and comparison to PR=19.86.

Output: pathway11_h100/results/fe188_lid_mle.json
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
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe188_lid_mle.json"

K_VALUES = [5, 10, 20, 50]
SEED = 9999
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def levina_bickel_lid(dists_sorted: np.ndarray, k: int) -> np.ndarray:
    """Compute LID-MLE for each sample.

    dists_sorted: (n, n-1) sorted distances to neighbors (excluding self).
    Returns: (n,) LID estimates.
    """
    r_k = np.maximum(dists_sorted[:, k - 1], 1e-30)  # (n,) k-th neighbor distance
    r_j = np.maximum(dists_sorted[:, :k], 1e-30)  # (n, k) first k neighbor distances
    log_ratios = np.log(r_j / r_k[:, None])  # negative since r_j < r_k
    lid = -k / log_ratios.sum(axis=1)
    return lid


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    n = X.shape[0]

    # Center
    X = X - X.mean(axis=0)

    print("Computing pairwise distances...")
    dists = squareform(pdist(X, metric="euclidean"))

    # Sort distances per sample (exclude self-distance=0)
    np.fill_diagonal(dists, np.inf)
    sorted_idx = np.argsort(dists, axis=1)
    dists_sorted = np.take_along_axis(dists, sorted_idx, axis=1)[:, :max(K_VALUES)]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results_per_k = {}

    for k in K_VALUES:
        lid = levina_bickel_lid(dists_sorted, k)

        lid_correct = lid[correct]
        lid_incorrect = lid[~correct]

        # LID as correctness predictor (OOF)
        oof_scores = np.zeros(n, dtype=np.float64)
        for train_idx, test_idx in skf.split(X, correct):
            oof_scores[test_idx] = lid[test_idx]

        auroc_lid = auroc(lid, correct)
        auroc_neg_lid = auroc(-lid, correct)

        results_per_k[str(k)] = {
            "k": k,
            "mean_lid": float(lid.mean()),
            "std_lid": float(lid.std()),
            "median_lid": float(np.median(lid)),
            "mean_lid_correct": float(lid_correct.mean()),
            "mean_lid_incorrect": float(lid_incorrect.mean()),
            "std_lid_correct": float(lid_correct.std()),
            "std_lid_incorrect": float(lid_incorrect.std()),
            "auroc_lid_positive": float(auroc_lid),
            "auroc_lid_negative": float(auroc_neg_lid),
            "best_auroc": float(max(auroc_lid, auroc_neg_lid)),
        }
        print(f"k={k}: mean LID={lid.mean():.2f}±{lid.std():.2f}  "
              f"correct={lid_correct.mean():.2f}  incorrect={lid_incorrect.mean():.2f}  "
              f"AUROC={max(auroc_lid, auroc_neg_lid):.4f}")

    out = {
        "experiment": "FE188",
        "description": "LID-MLE local intrinsic dimension (Levina-Bickel)",
        "n": n,
        "n_correct": int(correct.sum()),
        "centered": True,
        "results_per_k": results_per_k,
        "pr_reference": 19.86,
        "dom_auroc_reference": 0.7731,
        "interpretation": (
            "LID measures local intrinsic dimension around each sample. "
            "If mean LID ≈ PR (19.86), the global spectral dimension matches "
            "the local geometric dimension. Class-conditional LID differences "
            "would indicate correct/incorrect samples live on manifolds of "
            "different dimensionality."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "cache": str(CACHE),
            "seed": SEED,
            "method": "levina_bickel_mle",
            "distance_metric": "euclidean",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nPR reference: 19.86")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
