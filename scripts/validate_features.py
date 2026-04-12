#!/usr/bin/env python3
"""Phase 3: Analyze 13-feature experiment results.

Loads cached data from experiment1_v2, computes:
1. Per-feature diagnostics (AUROC, stats, degeneracy)
2. Feature correlation matrix
3. Forward selection / backward elimination ablation
4. Speed benchmark under different configs

Run after experiment1_v2.py completes:
    python scripts/validate_features.py --data-dir data/experiment1_v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from topo_confidence.features import FEATURE_NAMES, TopologicalFeatureExtractor


def load_experiment_data(data_dir: Path):
    """Load features and labels from experiment1_v2 output."""
    per_problem_path = data_dir / "per_problem.jsonl"
    with open(per_problem_path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    n = len(items)
    n_features = len(FEATURE_NAMES)
    features = np.zeros((n, n_features))
    correct = np.zeros(n, dtype=int)

    for i, item in enumerate(items):
        correct[i] = item["correct"]
        for j, name in enumerate(FEATURE_NAMES):
            features[i, j] = item["features"][name]

    return features, correct


def auroc_cv(features, correct, n_cv=50):
    """Compute AUROC with cross-validated logistic regression."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_cv_actual = min(n_cv, len(features))
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_cv_actual, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, X, correct, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(correct, probs)


def feature_ablation(features, correct, feature_names, n_cv=50):
    """Forward selection and backward elimination."""
    n_features = features.shape[1]

    # Forward selection: start empty, add one at a time
    print("\nForward Selection:")
    print(f"  {'Step':>4s}  {'Feature Added':>35s}  {'AUROC':>8s}  {'Delta':>8s}")
    selected = []
    remaining = list(range(n_features))
    prev_auroc = 0.5

    for step in range(n_features):
        best_auroc = -1
        best_idx = -1
        for idx in remaining:
            candidate = selected + [idx]
            try:
                a = auroc_cv(features[:, candidate], correct, n_cv)
                if a > best_auroc:
                    best_auroc = a
                    best_idx = idx
            except Exception:
                continue

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        delta = best_auroc - prev_auroc
        print(f"  {step+1:>4d}  {feature_names[best_idx]:>35s}  {best_auroc:>8.3f}  {delta:>+8.3f}")
        prev_auroc = best_auroc

    print(f"\n  Best subset ({len(selected)} features): "
          f"{[feature_names[i] for i in selected]}")

    # Backward elimination: start with all, remove one at a time
    print("\nBackward Elimination:")
    print(f"  {'Feature Removed':>35s}  {'AUROC':>8s}  {'Delta':>8s}")
    full_auroc = auroc_cv(features, correct, n_cv)
    print(f"  {'(all features)':>35s}  {full_auroc:>8.3f}")

    current = list(range(n_features))
    for _ in range(n_features - 1):
        best_auroc = -1
        best_remove = -1
        for idx in current:
            candidate = [i for i in current if i != idx]
            if not candidate:
                continue
            try:
                a = auroc_cv(features[:, candidate], correct, n_cv)
                if a > best_auroc:
                    best_auroc = a
                    best_remove = idx
            except Exception:
                continue

        if best_remove < 0:
            break

        delta = best_auroc - full_auroc
        print(f"  {feature_names[best_remove]:>35s}  {best_auroc:>8.3f}  {delta:>+8.3f}")
        current.remove(best_remove)
        full_auroc = best_auroc

    return selected


def speed_benchmark(data_dir: Path, n_samples=50):
    """Benchmark feature extraction speed under different configs."""
    # Load trajectories
    traj_path = data_dir / "trajectories.npz"
    if not traj_path.exists():
        print("\nSpeed benchmark: SKIPPED (no cached trajectories)")
        return

    data = np.load(traj_path)
    n_avail = len([k for k in data.keys() if k.startswith("traj_")])
    n_use = min(n_samples, n_avail)
    trajectories = [data[f"traj_{i}"] for i in range(n_use)]

    configs = [
        ("OLD (max_dim=1, null_k=0)", 1, 0),
        ("H2 only (max_dim=2, null_k=0)", 2, 0),
        ("Null only (max_dim=1, null_k=100)", 1, 100),
        ("FULL (max_dim=2, null_k=100)", 2, 100),
    ]

    print(f"\nSpeed Benchmark ({n_use} samples):")
    print(f"  {'Config':>40s}  {'Total':>8s}  {'Per-sample':>10s}")

    for label, max_dim, null_k in configs:
        ext = TopologicalFeatureExtractor(max_dim=max_dim, null_k=null_k)
        t0 = time.perf_counter()
        ext.extract(token_trajectories=trajectories)
        elapsed = time.perf_counter() - t0
        per_sample = elapsed / n_use
        print(f"  {label:>40s}  {elapsed:>7.1f}s  {per_sample:>9.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment1_v2 results")
    parser.add_argument("--data-dir", default="data/experiment1_v2")
    parser.add_argument("--cv-folds", type=int, default=50)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Phase 3: Feature Analysis")
    print("=" * 60)

    # --- Load data ---
    features, correct = load_experiment_data(data_dir)
    print(f"\nLoaded {len(features)} problems, {correct.sum()} correct ({correct.mean():.1%})")
    print(f"Features: {features.shape[1]} ({len(FEATURE_NAMES)} expected)")

    # --- Step 3a: Per-feature diagnostics ---
    print("\n" + "-" * 60)
    print("Step 3a: Per-feature Diagnostics")
    print("-" * 60)

    print(f"\n  {'Feature':>35s}  {'AUROC':>7s}  {'mean':>10s}  {'std':>10s}  "
          f"{'min':>10s}  {'max':>10s}  {'frac0':>6s}")

    for j, name in enumerate(FEATURE_NAMES):
        col = features[:, j]
        mean_v = col.mean()
        std_v = col.std()
        min_v = col.min()
        max_v = col.max()
        frac_zero = (col == 0).mean()

        if std_v < 1e-10:
            auroc_str = "  N/A"
            flag = " [DEGENERATE]"
        else:
            try:
                a1 = roc_auc_score(correct, col)
                a2 = roc_auc_score(correct, -col)
                auroc = max(a1, a2)
                auroc_str = f"{auroc:.3f}"
                flag = " ***" if auroc > 0.6 else ""
            except Exception:
                auroc_str = "  ERR"
                flag = ""

        print(f"  {name:>35s}  {auroc_str:>7s}  {mean_v:>10.4f}  {std_v:>10.4f}  "
              f"{min_v:>10.4f}  {max_v:>10.4f}  {frac_zero:>6.2f}{flag}")

    # --- Correlation matrix ---
    print("\nFeature Correlations (|r| > 0.7 flagged):")
    corr = np.corrcoef(features.T)
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            if abs(corr[i, j]) > 0.7:
                print(f"  {FEATURE_NAMES[i]} <-> {FEATURE_NAMES[j]}: r = {corr[i, j]:.3f}")

    # --- Step 3b: Overall AUROC comparison ---
    print("\n" + "-" * 60)
    print("Step 3b: Overall AUROC Comparison")
    print("-" * 60)

    # Original 7 features
    original_idx = [0, 1, 2, 3, 4, 5, 9]  # H0-H1 features + bridge
    auroc_7 = auroc_cv(features[:, original_idx], correct, args.cv_folds)

    # All 13 features
    auroc_13 = auroc_cv(features, correct, args.cv_folds)

    # Original 6 (without bridge)
    auroc_6 = auroc_cv(features[:, [0, 1, 2, 3, 4, 5]], correct, args.cv_folds)

    delta = auroc_13 - auroc_7
    print(f"\n  6-feature (PH only):    {auroc_6:.3f}")
    print(f"  7-feature (+ bridge):   {auroc_7:.3f}")
    print(f"  13-feature (all):       {auroc_13:.3f}")
    print(f"  AUROC delta (13 vs 7):  {delta:+.3f}")

    if auroc_13 >= 0.75:
        print(f"  Target (>= 0.75): MET")
    elif auroc_13 >= 0.699:
        print(f"  Target (>= 0.75): NOT MET (but no regression)")
    else:
        print(f"  Target (>= 0.75): NOT MET (REGRESSION from baseline)")

    # --- Step 3c: Feature ablation ---
    if not args.skip_ablation:
        print("\n" + "-" * 60)
        print("Step 3c: Feature Ablation")
        print("-" * 60)
        feature_ablation(features, correct, FEATURE_NAMES, args.cv_folds)

    # --- Step 3d: Speed benchmark ---
    if not args.skip_speed:
        print("\n" + "-" * 60)
        print("Step 3d: Speed Benchmark")
        print("-" * 60)
        speed_benchmark(data_dir)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
