#!/usr/bin/env python3
"""Stage 4: DeLong AUROC comparison between model variants.

Provides paired statistical testing to determine whether adding non-Euclidean
features significantly improves AUROC over the 0.796 baseline.

Uses MLstatkit for the DeLong test. Falls back to a bootstrap comparison
if MLstatkit is not installed.

Run: python pathway7/validation/delong_comparison.py --probs_a FILE_A --probs_b FILE_B --labels LABELS
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score


def delong_test(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray):
    """Paired DeLong test for comparing two AUROCs.

    Returns (z_statistic, p_value). Positive z means probs_a > probs_b.
    """
    try:
        from MLstatkit.stats import Delong_test
        z, p = Delong_test(y_true, probs_a, probs_b)
        return float(z), float(p), "delong"
    except ImportError:
        pass

    # Fallback: paired bootstrap
    return _bootstrap_auroc_comparison(y_true, probs_a, probs_b)


def _bootstrap_auroc_comparison(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, str]:
    """Paired stratified bootstrap AUROC comparison."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y_b = y_true[idx]
        # Skip if bootstrap sample is single-class
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, probs_a[idx])
        auc_b = roc_auc_score(y_b, probs_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return 0.0, 1.0, "bootstrap"

    mean_diff = diffs.mean()
    std_diff = diffs.std()
    z = mean_diff / std_diff if std_diff > 0 else 0.0
    # Two-sided p-value from normal approximation
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))

    return float(z), float(p), "bootstrap"


def compare_models(
    y_true: np.ndarray,
    probs_baseline: np.ndarray,
    probs_new: np.ndarray,
    name_baseline: str = "Euclidean-44",
    name_new: str = "Extended",
) -> dict:
    """Compare two sets of predicted probabilities."""
    auc_base = roc_auc_score(y_true, probs_baseline)
    auc_new = roc_auc_score(y_true, probs_new)
    z, p, method = delong_test(y_true, probs_new, probs_baseline)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    result = {
        "baseline_name": name_baseline,
        "new_name": name_new,
        "auroc_baseline": round(auc_base, 4),
        "auroc_new": round(auc_new, 4),
        "auroc_diff": round(auc_new - auc_base, 4),
        "z_statistic": round(z, 4),
        "p_value": round(p, 6),
        "significance": sig,
        "method": method,
    }

    print(f"\n  {name_baseline} AUROC: {auc_base:.4f}")
    print(f"  {name_new} AUROC: {auc_new:.4f}")
    print(f"  Diff: {auc_new - auc_base:+.4f}")
    print(f"  {method} z={z:.3f}, p={p:.4f} {sig}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DeLong AUROC comparison")
    parser.add_argument("--probs_a", type=str, help="Path to baseline predicted probabilities (.npy)")
    parser.add_argument("--probs_b", type=str, help="Path to new model predicted probabilities (.npy)")
    parser.add_argument("--labels", type=str, help="Path to true labels (.npy)")
    parser.add_argument("--output", type=str, default="pathway7/validation/delong_results.json")
    args = parser.parse_args()

    if args.probs_a and args.probs_b and args.labels:
        y_true = np.load(args.labels)
        probs_a = np.load(args.probs_a)
        probs_b = np.load(args.probs_b)

        print("=" * 60)
        print("DeLong AUROC Comparison")
        print("=" * 60)

        result = compare_models(y_true, probs_a, probs_b)

        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {args.output}")
    else:
        print("DeLong comparison module loaded. Use compare_models() or run with --probs_a/--probs_b/--labels.")
        print("Example:")
        print("  python pathway7/validation/delong_comparison.py \\")
        print("    --probs_a baseline_probs.npy --probs_b new_probs.npy --labels holdout_labels.npy")


if __name__ == "__main__":
    main()
