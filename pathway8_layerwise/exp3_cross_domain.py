#!/usr/bin/env python3
"""Experiment 3: Cross-domain generalization (HumanEval + BBH).

Applies the best method(s) from Experiments 1-2 to HumanEval and BBH data.
AUROC > 0.65 = cross-domain signal (meaningful above random).

CPU only — reads pre-extracted all-layer states.

Run: python pathway8_layerwise/exp3_cross_domain.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.config import (
    BBH_DATA_DIR,
    HUMANEVAL_DATA_DIR,
    LR_PARAMS,
    RESULTS_DIR,
    is_done,
    load_all_layer_states,
    load_manifest,
    mark_done,
)
from pathway8_layerwise.coe_features import compute_coe_batch
from pathway8_layerwise.d2hscore_features import compute_d2h_batch
from pathway8_layerwise.layerwise_features import (
    compute_layerwise_ph_batch,
    fit_per_layer_pca,
)

STAGE_NAME = "exp3_cross_domain"

BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]


def evaluate_benchmark(
    states_list: list[np.ndarray | None],
    labels: np.ndarray,
    benchmark_name: str,
    methods: list[str] | None = None,
) -> dict:
    """Evaluate one or more methods on a benchmark dataset.

    Returns dict with per-method AUROC results.
    """
    if methods is None:
        methods = ["layerwise_ph", "coe", "d2h_lite"]

    n = len(labels)
    if n < 20 or labels.sum() < 3 or (n - labels.sum()) < 3:
        print(f"  [WARN] {benchmark_name}: insufficient data (n={n}, pos={labels.sum()})")
        return {"benchmark": benchmark_name, "n": n, "n_correct": int(labels.sum()),
                "error": "insufficient data for meaningful AUROC"}

    # Train/test split (80/20)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9999)
    train_idx, hold_idx = next(sss.split(np.zeros(n), labels.astype(int)))
    y_train = labels[train_idx]
    y_holdout = labels[hold_idx]

    results_by_method = {}

    for method in methods:
        print(f"  Computing {method} features for {benchmark_name} ...")
        t0 = time.time()

        if method == "layerwise_ph":
            transforms = fit_per_layer_pca(states_list, train_idx)
            features = compute_layerwise_ph_batch(states_list, transforms, n_jobs=-1)
        elif method == "coe":
            features, _ = compute_coe_batch(states_list)
        elif method == "d2h_lite":
            features, _ = compute_d2h_batch(states_list, full=False)
        else:
            continue

        elapsed = time.time() - t0
        X_train = features[train_idx]
        X_holdout = features[hold_idx]

        # Train LR
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_ho = scaler.transform(X_holdout)

        lr = LogisticRegression(C=1.0, **LR_PARAMS)
        lr.fit(X_tr, y_train)
        probs = lr.predict_proba(X_ho)[:, 1]

        try:
            auroc = roc_auc_score(y_holdout, probs)
        except ValueError:
            auroc = 0.5

        # CV
        n_folds = min(10, max(2, int(y_train.sum())))
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        try:
            probs_cv = cross_val_predict(
                LogisticRegression(C=1.0, **LR_PARAMS),
                X_tr, y_train, cv=cv, method="predict_proba",
            )[:, 1]
            auroc_cv = roc_auc_score(y_train, probs_cv)
        except Exception:
            auroc_cv = 0.5

        results_by_method[method] = {
            "auroc_holdout": round(auroc, 4),
            "auroc_cv": round(auroc_cv, 4),
            "n_features": features.shape[1],
            "compute_time": round(elapsed, 1),
        }
        print(f"    {method}: AUROC={auroc:.4f} (CV={auroc_cv:.4f}), {elapsed:.1f}s")

    return {
        "benchmark": benchmark_name,
        "n": n,
        "n_correct": int(labels.sum()),
        "accuracy": round(float(labels.mean()), 4),
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "methods": results_by_method,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_done(STAGE_NAME):
        print(f"[SKIP] {STAGE_NAME} already done")
        return

    print("=" * 70)
    print("Experiment 3: Cross-Domain Generalization")
    print("=" * 70)

    # Determine which methods to test based on Exp 2 results
    exp2_path = RESULTS_DIR / "exp2_results.json"
    methods = ["layerwise_ph", "coe", "d2h_lite"]
    if exp2_path.exists():
        exp2 = json.loads(exp2_path.read_text())
        print(f"  Exp 2 best model: {exp2['best_model']} ({exp2['best_auroc']:.4f})")

    all_results = {}

    # ---- HumanEval ----
    print(f"\n{'='*70}")
    print("HumanEval (164 problems)")
    print(f"{'='*70}")

    if HUMANEVAL_DATA_DIR.exists() and list(HUMANEVAL_DATA_DIR.glob("problem_*.npz")):
        manifest = load_manifest(HUMANEVAL_DATA_DIR)
        n_he = manifest["n_problems"]

        states_he = load_all_layer_states(HUMANEVAL_DATA_DIR, n_he)
        labels_he = np.array([
            p.get("correct", False) for p in manifest["problems"]
        ], dtype=bool)

        print(f"  Accuracy: {labels_he.mean():.1%} ({labels_he.sum()}/{len(labels_he)})")
        all_results["humaneval"] = evaluate_benchmark(
            states_he, labels_he, "humaneval", methods
        )
    else:
        print("  [SKIP] No HumanEval data found")
        all_results["humaneval"] = {"error": "no data"}

    # ---- BBH (per-subset + pooled) ----
    print(f"\n{'='*70}")
    print("BBH (3 subsets × 250)")
    print(f"{'='*70}")

    bbh_all_states = []
    bbh_all_labels = []

    for subset in BBH_SUBSETS:
        subset_dir = BBH_DATA_DIR / subset
        if not subset_dir.exists() or not list(subset_dir.glob("problem_*.npz")):
            print(f"  [SKIP] No data for {subset}")
            all_results[f"bbh_{subset}"] = {"error": "no data"}
            continue

        n_sub = len(list(subset_dir.glob("problem_*.npz")))
        states_sub = load_all_layer_states(subset_dir, n_sub)

        # Load labels from BBH manifest
        bbh_manifest = load_manifest(BBH_DATA_DIR)
        labels_sub = np.array([
            p.get("correct", False)
            for p in bbh_manifest["problems"]
            if p.get("subset") == subset
        ], dtype=bool)

        if len(labels_sub) != len(states_sub):
            # Fallback: load from individual npz files
            labels_sub = []
            for i in range(n_sub):
                data = np.load(subset_dir / f"problem_{i:03d}.npz", allow_pickle=True)
                labels_sub.append(bool(data["correct"]))
            labels_sub = np.array(labels_sub, dtype=bool)

        print(f"\n  {subset}: {labels_sub.mean():.1%} ({labels_sub.sum()}/{len(labels_sub)})")
        all_results[f"bbh_{subset}"] = evaluate_benchmark(
            states_sub, labels_sub, subset, methods
        )

        bbh_all_states.extend(states_sub)
        bbh_all_labels.extend(labels_sub)

    # Pooled BBH
    if bbh_all_states:
        bbh_labels = np.array(bbh_all_labels, dtype=bool)
        print(f"\n  BBH pooled: {bbh_labels.mean():.1%} ({bbh_labels.sum()}/{len(bbh_labels)})")
        all_results["bbh_pooled"] = evaluate_benchmark(
            bbh_all_states, bbh_labels, "bbh_pooled", methods
        )

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("Cross-Domain Summary")
    print(f"{'='*70}")

    cross_domain_signal = False
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name}: {res['error']}")
            continue
        best_method = max(res["methods"].items(), key=lambda x: x[1]["auroc_holdout"])
        auroc = best_method[1]["auroc_holdout"]
        signal = "SIGNAL" if auroc > 0.65 else "no signal"
        if auroc > 0.65:
            cross_domain_signal = True
        print(f"  {name:<45s} best={best_method[0]:<15s} AUROC={auroc:.4f} [{signal}]")

    results = {
        "experiment": "exp3_cross_domain",
        "benchmarks": all_results,
        "cross_domain_signal": cross_domain_signal,
        "signal_threshold": 0.65,
    }

    out_path = RESULTS_DIR / "exp3_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    mark_done(STAGE_NAME)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
