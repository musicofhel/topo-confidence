#!/usr/bin/env python3
"""Phase 3 Step 3: Cross-Benchmark / Cross-Model Analysis.

CPU ONLY. Run after step1 (GSM8K) and step2 (MATH-7B).

Compares:
- 1.5B × MATH-500 (from Phase 1 rebuild)
- 1.5B × GSM8K (from step1)
- 7B × MATH-500 (from step2)

Produces the final cross-benchmark comparison table and transfer analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_V2_PATH,
    LR_PARAMS,
    get_abc_column_indices,
    get_train_holdout_indices,
    load_features_with_sss_fix,
    logger,
)

PHASE1_DIR = Path(__file__).parent.parent / "phase1_prompt_model"
PHASE2_DIR = Path(__file__).parent.parent / "phase2_completion"
GSM8K_DIR = Path(__file__).parent / "gsm8k"
MATH7B_DIR = Path(__file__).parent / "math7b"
OUTPUT_DIR = Path(__file__).parent


def load_math_1_5b_results():
    """Load 1.5B × MATH-500 results from Phase 1 rebuild."""
    with open(PHASE1_DIR / "holdout_metrics_v2.json") as f:
        metrics = json.load(f)

    correct = np.load(BASELINE_CORRECT_V2_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()

    # Load features
    old_correct = np.load(
        Path(__file__).parent.parent.parent / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
    )
    X_train, X_holdout, _, _, _, _ = load_features_with_sss_fix(baseline_correct=old_correct)

    # Load selection results
    selection = {}
    sel_path = PHASE2_DIR / "holdout_results_v2.json"
    if sel_path.exists():
        with open(sel_path) as f:
            selection = json.load(f)

    return {
        "greedy_acc": float(correct.mean()),
        "greedy_correct": int(correct.sum()),
        "topo_auroc": metrics["auroc"],
        "train_correct": int(correct[train_idx].sum()),
        "holdout_correct": int(correct[holdout_idx].sum()),
        "features_train": X_train,
        "features_holdout": X_holdout,
        "correct": correct,
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "selection": selection,
    }


def load_gsm8k_results():
    """Load GSM8K results from step1."""
    if not GSM8K_DIR.exists():
        logger.warning("GSM8K results not found at %s", GSM8K_DIR)
        return None

    with open(GSM8K_DIR / "summary.json") as f:
        summary = json.load(f)
    with open(GSM8K_DIR / "model_results.json") as f:
        model_results = json.load(f)
    with open(GSM8K_DIR / "selection_results.json") as f:
        selection = json.load(f)

    features = np.load(GSM8K_DIR / "features.npy")
    correct = np.load(GSM8K_DIR / "baseline_correct.npy")
    train_idx = np.load(GSM8K_DIR / "train_idx.npy")
    test_idx = np.load(GSM8K_DIR / "test_idx.npy")

    return {
        "greedy_acc": summary["greedy_accuracy"],
        "greedy_correct": summary["greedy_correct"],
        "topo_auroc": summary["topo_auroc"],
        "topo_auroc_ci": summary.get("topo_auroc_ci"),
        "best_baseline": model_results["best_baseline"],
        "best_baseline_auroc": model_results["best_baseline_auroc"],
        "features": features,
        "correct": correct,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "selection": selection,
    }


def load_math7b_results():
    """Load 7B × MATH-500 results from step2."""
    if not MATH7B_DIR.exists():
        logger.warning("MATH-7B results not found at %s", MATH7B_DIR)
        return None

    with open(MATH7B_DIR / "summary.json") as f:
        summary = json.load(f)
    with open(MATH7B_DIR / "model_results.json") as f:
        model_results = json.load(f)
    with open(MATH7B_DIR / "selection_results.json") as f:
        selection = json.load(f)

    features = np.load(MATH7B_DIR / "features.npy")
    correct = np.load(MATH7B_DIR / "baseline_correct.npy")
    train_idx = np.load(MATH7B_DIR / "train_idx.npy")
    holdout_idx = np.load(MATH7B_DIR / "holdout_idx.npy")

    return {
        "greedy_acc": summary["greedy_accuracy"],
        "greedy_correct": summary["greedy_correct"],
        "topo_auroc": summary["topo_auroc"],
        "topo_auroc_ci": summary.get("topo_auroc_ci"),
        "best_baseline": model_results["best_baseline"],
        "best_baseline_auroc": model_results["best_baseline_auroc"],
        "features": features,
        "correct": correct,
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "selection": selection,
    }


def feature_distribution_comparison(r1_5b, gsm8k, math7b, feature_names):
    """Compare topo feature distributions across models/benchmarks."""
    results = {}

    configs = []
    if r1_5b is not None:
        configs.append(("1.5B_MATH", r1_5b["features_train"], r1_5b["correct"][r1_5b["train_idx"]]))
    if gsm8k is not None:
        configs.append(("1.5B_GSM8K", gsm8k["features"][gsm8k["train_idx"]], gsm8k["correct"][gsm8k["train_idx"]]))
    if math7b is not None:
        configs.append(("7B_MATH", math7b["features"][math7b["train_idx"]], math7b["correct"][math7b["train_idx"]]))

    if len(configs) < 2:
        return results

    # Per-feature Cohen's d for correct vs incorrect
    for name, features, correct in configs:
        ds = []
        for fi in range(features.shape[1]):
            vals_correct = features[correct.astype(bool), fi]
            vals_wrong = features[~correct.astype(bool), fi]
            if len(vals_correct) > 1 and len(vals_wrong) > 1:
                pooled_std = np.sqrt(
                    ((len(vals_correct) - 1) * vals_correct.std() ** 2 +
                     (len(vals_wrong) - 1) * vals_wrong.std() ** 2) /
                    (len(vals_correct) + len(vals_wrong) - 2)
                )
                d = (vals_correct.mean() - vals_wrong.mean()) / max(pooled_std, 1e-10)
            else:
                d = 0.0
            ds.append(float(d))
        results[f"cohens_d_{name}"] = ds

    # Rank correlation of effect sizes between configs
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            name_i = configs[i][0]
            name_j = configs[j][0]
            d_i = results[f"cohens_d_{name_i}"]
            d_j = results[f"cohens_d_{name_j}"]
            rho, p = spearmanr(d_i, d_j)
            results[f"rank_corr_{name_i}_vs_{name_j}"] = {"rho": float(rho), "p": float(p)}

    return results


def transfer_experiment(r1_5b, gsm8k, math7b, feature_names):
    """Train on one model/benchmark, evaluate on another."""
    results = {}
    abc_idx = get_abc_column_indices(feature_names)

    def fit_and_score(X_train, y_train, X_test, y_test, label):
        X_tr = X_train[:, abc_idx]
        X_te = X_test[:, abc_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_tr_s, y_train.astype(int))
        scores = clf.predict_proba(X_te_s)[:, 1]
        if len(np.unique(y_test)) >= 2:
            auroc = roc_auc_score(y_test.astype(int), scores)
        else:
            auroc = None
        results[label] = float(auroc) if auroc is not None else None
        logger.info("  Transfer %s: AUROC = %s", label, f"{auroc:.4f}" if auroc else "N/A")

    # 1.5B MATH → 7B MATH
    if r1_5b is not None and math7b is not None:
        fit_and_score(
            r1_5b["features_train"], r1_5b["correct"][r1_5b["train_idx"]],
            math7b["features"][math7b["holdout_idx"]], math7b["correct"][math7b["holdout_idx"]],
            "1.5B_MATH_to_7B_MATH",
        )

    # 7B MATH → 1.5B MATH
    if math7b is not None and r1_5b is not None:
        fit_and_score(
            math7b["features"][math7b["train_idx"]], math7b["correct"][math7b["train_idx"]],
            r1_5b["features_holdout"], r1_5b["correct"][r1_5b["holdout_idx"]],
            "7B_MATH_to_1.5B_MATH",
        )

    # 1.5B MATH → 1.5B GSM8K
    if r1_5b is not None and gsm8k is not None:
        fit_and_score(
            r1_5b["features_train"], r1_5b["correct"][r1_5b["train_idx"]],
            gsm8k["features"][gsm8k["test_idx"]], gsm8k["correct"][gsm8k["test_idx"]],
            "1.5B_MATH_to_1.5B_GSM8K",
        )

    # 1.5B GSM8K → 1.5B MATH
    if gsm8k is not None and r1_5b is not None:
        fit_and_score(
            gsm8k["features"][gsm8k["train_idx"]], gsm8k["correct"][gsm8k["train_idx"]],
            r1_5b["features_holdout"], r1_5b["correct"][r1_5b["holdout_idx"]],
            "1.5B_GSM8K_to_1.5B_MATH",
        )

    return results


def build_comparison_table(r1_5b, gsm8k, math7b):
    """Build the final comparison table."""
    def get(d, *keys, default="—"):
        if d is None:
            return default
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    rows = []
    metrics = [
        ("Greedy accuracy", lambda d: f"{get(d, 'greedy_acc', default=0):.1%}"),
        ("Topo AUROC", lambda d: f"{get(d, 'topo_auroc', default=0):.4f}" if get(d, 'topo_auroc') != "—" else "—"),
        ("Best baseline AUROC", lambda d: f"{get(d, 'best_baseline_auroc', default=0):.4f}" if get(d, 'best_baseline_auroc') != "—" else "—"),
        ("Ungated MV net gain", lambda d: f"{get(d, 'selection', 'ungated_mv', 'net_gain', default='—')}"),
        ("Oracle ceiling", lambda d: f"+{get(d, 'selection', 'oracle_ceiling', default='—')}"),
    ]

    for label, fmt in metrics:
        row = {
            "metric": label,
            "1.5B_MATH": fmt(r1_5b) if r1_5b else "—",
            "1.5B_GSM8K": fmt(gsm8k) if gsm8k else "—",
            "7B_MATH": fmt(math7b) if math7b else "—",
        }
        rows.append(row)

    return rows


def main():
    logger.info("=" * 70)
    logger.info("PHASE 3 STEP 3: CROSS-BENCHMARK / CROSS-MODEL ANALYSIS")
    logger.info("=" * 70)

    # Load results
    logger.info("\n--- Loading results ---")
    r1_5b = load_math_1_5b_results()
    logger.info("1.5B MATH: loaded (accuracy %.1f%%)", 100 * r1_5b["greedy_acc"])

    gsm8k = load_gsm8k_results()
    if gsm8k:
        logger.info("1.5B GSM8K: loaded (accuracy %.1f%%)", 100 * gsm8k["greedy_acc"])
    else:
        logger.warning("1.5B GSM8K: NOT AVAILABLE")

    math7b = load_math7b_results()
    if math7b:
        logger.info("7B MATH: loaded (accuracy %.1f%%)", 100 * math7b["greedy_acc"])
    else:
        logger.warning("7B MATH: NOT AVAILABLE")

    # Load feature names (from whichever source is available)
    feature_names = None
    for src_dir in [GSM8K_DIR, MATH7B_DIR]:
        fn_path = src_dir / "feature_names.json"
        if fn_path.exists():
            with open(fn_path) as f:
                feature_names = json.load(f)
            break
    if feature_names is None:
        # Fall back to pathway1 feature names
        from common import WINNING_FEATURES_PATH
        sys.path.insert(0, str(WINNING_FEATURES_PATH.parent))
        try:
            from winning_features import FEATURE_NAMES
            feature_names = FEATURE_NAMES
        except ImportError:
            logger.warning("Could not load feature names")

    # Feature distribution comparison
    logger.info("\n--- Feature Distribution Comparison ---")
    dist_results = feature_distribution_comparison(r1_5b, gsm8k, math7b, feature_names)
    for key in sorted(dist_results):
        if key.startswith("rank_corr"):
            v = dist_results[key]
            logger.info("  %s: rho=%.3f (p=%.4f)", key, v["rho"], v["p"])

    # Transfer experiment
    logger.info("\n--- Transfer Experiment ---")
    transfer_results = transfer_experiment(r1_5b, gsm8k, math7b, feature_names)

    # Comparison table
    logger.info("\n--- Comparison Table ---")
    table = build_comparison_table(r1_5b, gsm8k, math7b)
    print(f"\n{'Metric':<30} {'1.5B×MATH':>12} {'1.5B×GSM8K':>12} {'7B×MATH':>12}")
    print("-" * 70)
    for row in table:
        print(f"{row['metric']:<30} {row['1.5B_MATH']:>12} {row['1.5B_GSM8K']:>12} {row['7B_MATH']:>12}")

    # Save all results
    all_results = {
        "comparison_table": table,
        "feature_distribution": {
            k: v for k, v in dist_results.items()
            if not k.startswith("cohens_d")  # don't save per-feature arrays
        },
        "feature_rank_correlations": {
            k: v for k, v in dist_results.items()
            if k.startswith("rank_corr")
        },
        "transfer": transfer_results,
        "summaries": {
            "1.5B_MATH": {
                "greedy_acc": r1_5b["greedy_acc"],
                "topo_auroc": r1_5b.get("topo_auroc"),
            } if r1_5b else None,
            "1.5B_GSM8K": {
                "greedy_acc": gsm8k["greedy_acc"],
                "topo_auroc": gsm8k["topo_auroc"],
            } if gsm8k else None,
            "7B_MATH": {
                "greedy_acc": math7b["greedy_acc"],
                "topo_auroc": math7b["topo_auroc"],
            } if math7b else None,
        },
    }

    with open(OUTPUT_DIR / "cross_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\nSaved to %s", OUTPUT_DIR / "cross_analysis.json")


if __name__ == "__main__":
    main()
