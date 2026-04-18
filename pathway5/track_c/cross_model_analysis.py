#!/usr/bin/env python3
"""Track C: Cross-Model Topology Comparison.

CPU-only analysis comparing topo features from Qwen2.5-1.5B and 7B on the
same MATH-500 problems. Answers: does correctness look topologically similar
across model scales?

Phase C1: Feature distribution comparison, correlation analysis
Phase C2: Transfer experiment (train on 1.5B → evaluate on 7B and vice versa)
Phase C3: Distillation feasibility assessment

Requires: Track A (or existing 1.5B data) + Track B outputs.
Runtime: ~10 min CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_PATH,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    N_PROBLEMS_MATH,
    SEED,
    TRACK_B_DIR,
    TRACK_C_DIR,
    bootstrap_auroc,
    get_abc_column_indices,
    get_train_holdout_indices,
    logger,
)

# Existing 1.5B artifacts
BASELINE_CORRECT_1_5B = Path(__file__).parent.parent.parent / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"

# Track B artifacts
PHASE_B1_DIR = TRACK_B_DIR / "phase_b1"
PHASE_B2_DIR = TRACK_B_DIR / "phase_b2"

PHASE_DIR = TRACK_C_DIR


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Load data ----
    logger.info("=== Loading 1.5B and 7B features ===")
    train_idx, holdout_idx = get_train_holdout_indices()

    # 1.5B features (from Pathway 1)
    features_1_5b_train = np.load(FEATURES_TRAIN_PATH)  # (400, 78) SSS order
    features_1_5b_holdout = np.load(FEATURES_HOLDOUT_PATH)  # (100, 78) SSS order
    correct_1_5b = np.load(BASELINE_CORRECT_1_5B)  # (500,)

    # Reconstruct SSS ordering for 1.5B
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    sss_train_idx, sss_holdout_idx = next(sss.split(np.arange(500), correct_1_5b))
    # sss_train_idx[i] → global index of i-th row in features_train400

    # 7B features (from Track B Phase B2)
    features_7b = np.load(PHASE_B2_DIR / "math500_7b_features.npy")  # (500, 78)
    correct_7b = np.load(PHASE_B1_DIR / "math500_7b_baseline_correct.npy")  # (500,)

    with open(PHASE_B2_DIR / "math500_7b_feature_names.json") as f:
        feature_names = json.load(f)

    # Load 1.5B feature names (should match)
    p1_feature_names_path = Path(__file__).parent.parent.parent / "pathway1" / "phase0" / "artifact_locks.json"
    # We'll use the 7B feature names since both should produce same features from winning_features.py

    logger.info("1.5B: %d correct (%.1f%%)", correct_1_5b.sum(), 100 * correct_1_5b.mean())
    logger.info("7B: %d correct (%.1f%%)", correct_7b.sum(), 100 * correct_7b.mean())

    # Build aligned 1.5B feature matrix (500, 78) in global order
    features_1_5b_global = np.zeros((500, features_1_5b_train.shape[1]))
    for i, gidx in enumerate(sss_train_idx):
        features_1_5b_global[gidx] = features_1_5b_train[i]
    for i, gidx in enumerate(sss_holdout_idx):
        features_1_5b_global[gidx] = features_1_5b_holdout[i]

    abc_indices = get_abc_column_indices(feature_names)
    abc_names = [feature_names[i] for i in abc_indices]

    # ================================================================
    # Phase C1: Feature Distribution Comparison
    # ================================================================
    logger.info("\n=== Phase C1: Feature Distribution Comparison ===")

    # Categories of problems
    both_correct = correct_1_5b & correct_7b
    both_wrong = ~correct_1_5b & ~correct_7b
    only_1_5b = correct_1_5b & ~correct_7b  # Unlikely but possible
    only_7b = ~correct_1_5b & correct_7b  # Most interesting: 7B solves what 1.5B can't

    logger.info("Both correct: %d", both_correct.sum())
    logger.info("Both wrong: %d", both_wrong.sum())
    logger.info("Only 1.5B correct: %d", only_1_5b.sum())
    logger.info("Only 7B correct: %d", only_7b.sum())

    # Per-feature comparison: correct vs incorrect within each model
    feature_comparisons = []
    for feat_idx in abc_indices:
        fname = feature_names[feat_idx]

        # 1.5B: correct vs incorrect
        f_1_5b = features_1_5b_global[:, feat_idx]
        f_7b = features_7b[:, feat_idx]

        try:
            # Effect size: Cohen's d for correct vs incorrect
            c1_correct = f_1_5b[correct_1_5b]
            c1_incorrect = f_1_5b[~correct_1_5b]
            pooled_std_1 = np.sqrt(
                ((len(c1_correct) - 1) * c1_correct.std() ** 2 + (len(c1_incorrect) - 1) * c1_incorrect.std() ** 2)
                / (len(c1_correct) + len(c1_incorrect) - 2)
            )
            cohen_d_1 = (c1_correct.mean() - c1_incorrect.mean()) / (pooled_std_1 + 1e-12)

            c7_correct = f_7b[correct_7b]
            c7_incorrect = f_7b[~correct_7b]
            pooled_std_7 = np.sqrt(
                ((len(c7_correct) - 1) * c7_correct.std() ** 2 + (len(c7_incorrect) - 1) * c7_incorrect.std() ** 2)
                / (len(c7_correct) + len(c7_incorrect) - 2)
            )
            cohen_d_7 = (c7_correct.mean() - c7_incorrect.mean()) / (pooled_std_7 + 1e-12)

            # Cross-model correlation on same problems
            r_all, p_all = stats.pearsonr(f_1_5b, f_7b)

            # Correlation within correct-for-both
            if both_correct.sum() > 5:
                r_correct, _ = stats.pearsonr(f_1_5b[both_correct], f_7b[both_correct])
            else:
                r_correct = float("nan")

            # Correlation within wrong-for-both
            if both_wrong.sum() > 5:
                r_wrong, _ = stats.pearsonr(f_1_5b[both_wrong], f_7b[both_wrong])
            else:
                r_wrong = float("nan")

            feature_comparisons.append({
                "feature": fname,
                "cohen_d_1_5b": round(cohen_d_1, 4),
                "cohen_d_7b": round(cohen_d_7, 4),
                "d_diff": round(cohen_d_7 - cohen_d_1, 4),
                "cross_model_r_all": round(r_all, 4),
                "cross_model_r_correct": round(r_correct, 4) if not np.isnan(r_correct) else None,
                "cross_model_r_wrong": round(r_wrong, 4) if not np.isnan(r_wrong) else None,
            })
        except Exception as e:
            feature_comparisons.append({"feature": fname, "error": str(e)})

    # Sort by absolute difference in Cohen's d
    feature_comparisons.sort(key=lambda x: abs(x.get("d_diff", 0)), reverse=True)

    # Summary statistics
    all_cross_r = [fc["cross_model_r_all"] for fc in feature_comparisons
                   if "cross_model_r_all" in fc and fc["cross_model_r_all"] is not None]
    all_d_1_5b = [fc["cohen_d_1_5b"] for fc in feature_comparisons if "cohen_d_1_5b" in fc]
    all_d_7b = [fc["cohen_d_7b"] for fc in feature_comparisons if "cohen_d_7b" in fc]

    # Rank correlation of effect sizes
    if len(all_d_1_5b) == len(all_d_7b) and len(all_d_1_5b) > 2:
        d_rank_rho, d_rank_p = stats.spearmanr(all_d_1_5b, all_d_7b)
    else:
        d_rank_rho, d_rank_p = 0, 1

    logger.info("Cross-model feature correlation (all): mean r=%.3f", np.mean(all_cross_r))
    logger.info("Effect size rank correlation: rho=%.3f (p=%.4f)", d_rank_rho, d_rank_p)
    logger.info("Top divergent features (by |Δd|):")
    for fc in feature_comparisons[:5]:
        logger.info("  %s: d_1.5B=%.3f, d_7B=%.3f, Δ=%.3f, cross_r=%.3f",
                     fc["feature"], fc.get("cohen_d_1_5b", 0), fc.get("cohen_d_7b", 0),
                     fc.get("d_diff", 0), fc.get("cross_model_r_all", 0))

    # ================================================================
    # Phase C2: Transfer Experiment
    # ================================================================
    logger.info("\n=== Phase C2: Transfer Experiment ===")

    # Experiment 1: Train on 1.5B train-400 → evaluate on 7B holdout-100
    X_train_1_5b = features_1_5b_global[sss_train_idx][:, abc_indices]
    y_train_1_5b = correct_1_5b[sss_train_idx]

    X_holdout_7b = features_7b[holdout_idx][:, abc_indices]
    y_holdout_7b = correct_7b[holdout_idx]

    scaler_1 = StandardScaler()
    X_train_1_5b_scaled = scaler_1.fit_transform(X_train_1_5b)

    clf_1 = LogisticRegression(**LR_PARAMS)
    clf_1.fit(X_train_1_5b_scaled, y_train_1_5b)

    # Score 7B holdout using 1.5B scaler and model
    X_holdout_7b_scaled = scaler_1.transform(X_holdout_7b)
    transfer_1to7_scores = clf_1.predict_proba(X_holdout_7b_scaled)[:, 1]
    try:
        transfer_1to7_auroc, t1to7_lo, t1to7_hi = bootstrap_auroc(y_holdout_7b, transfer_1to7_scores)
        logger.info("Transfer 1.5B→7B AUROC: %.3f [%.3f-%.3f]",
                     transfer_1to7_auroc, t1to7_lo, t1to7_hi)
    except ValueError as e:
        logger.warning("Transfer 1.5B→7B failed: %s", e)
        transfer_1to7_auroc, t1to7_lo, t1to7_hi = None, None, None

    # Experiment 2: Train on 7B train-400 → evaluate on 1.5B holdout-100
    X_train_7b = features_7b[train_idx][:, abc_indices]
    y_train_7b = correct_7b[train_idx]

    X_holdout_1_5b = features_1_5b_global[sss_holdout_idx][:, abc_indices]
    y_holdout_1_5b = correct_1_5b[sss_holdout_idx]

    scaler_7 = StandardScaler()
    X_train_7b_scaled = scaler_7.fit_transform(X_train_7b)

    clf_7 = LogisticRegression(**LR_PARAMS)
    clf_7.fit(X_train_7b_scaled, y_train_7b)

    X_holdout_1_5b_scaled = scaler_7.transform(X_holdout_1_5b)
    transfer_7to1_scores = clf_7.predict_proba(X_holdout_1_5b_scaled)[:, 1]
    try:
        transfer_7to1_auroc, t7to1_lo, t7to1_hi = bootstrap_auroc(y_holdout_1_5b, transfer_7to1_scores)
        logger.info("Transfer 7B→1.5B AUROC: %.3f [%.3f-%.3f]",
                     transfer_7to1_auroc, t7to1_lo, t7to1_hi)
    except ValueError as e:
        logger.warning("Transfer 7B→1.5B failed: %s", e)
        transfer_7to1_auroc, t7to1_lo, t7to1_hi = None, None, None

    # Native baselines for comparison
    # 1.5B native (from existing results)
    native_1_5b_auroc = 0.935  # Known from Pathway 4

    # 7B native
    X_holdout_7b_native = features_7b[holdout_idx][:, abc_indices]
    scaler_7n = StandardScaler()
    scaler_7n.fit(features_7b[train_idx][:, abc_indices])
    X_train_7b_n = scaler_7n.transform(features_7b[train_idx][:, abc_indices])
    X_holdout_7b_n = scaler_7n.transform(X_holdout_7b_native)

    clf_7n = LogisticRegression(**LR_PARAMS)
    clf_7n.fit(X_train_7b_n, y_train_7b)
    native_7b_scores = clf_7n.predict_proba(X_holdout_7b_n)[:, 1]
    try:
        native_7b_auroc, _, _ = bootstrap_auroc(y_holdout_7b, native_7b_scores)
    except ValueError:
        native_7b_auroc = None

    logger.info("\nTransfer comparison:")
    logger.info("  1.5B native AUROC:      %.3f", native_1_5b_auroc)
    logger.info("  7B native AUROC:        %.3f", native_7b_auroc or 0)
    logger.info("  1.5B→7B transfer AUROC: %.3f", transfer_1to7_auroc or 0)
    logger.info("  7B→1.5B transfer AUROC: %.3f", transfer_7to1_auroc or 0)

    # Compute transfer efficiency
    if native_7b_auroc and transfer_1to7_auroc:
        transfer_efficiency_1to7 = transfer_1to7_auroc / native_7b_auroc
    else:
        transfer_efficiency_1to7 = None

    if transfer_7to1_auroc:
        transfer_efficiency_7to1 = transfer_7to1_auroc / native_1_5b_auroc
    else:
        transfer_efficiency_7to1 = None

    # ================================================================
    # Phase C3: Distillation Feasibility Assessment
    # ================================================================
    logger.info("\n=== Phase C3: Distillation Feasibility ===")

    # Compute feature-level transfer: which features transfer best?
    feature_transfer = []
    coef_1_5b = np.abs(clf_1.coef_[0])
    coef_7b = np.abs(clf_7.coef_[0])

    for i, fname in enumerate(abc_names):
        feature_transfer.append({
            "feature": fname,
            "coef_1_5b": float(coef_1_5b[i]),
            "coef_7b": float(coef_7b[i]),
        })

    # Which features are important in BOTH models?
    coef_rank_rho, coef_rank_p = stats.spearmanr(coef_1_5b, coef_7b)
    logger.info("Feature importance rank correlation: rho=%.3f (p=%.4f)", coef_rank_rho, coef_rank_p)

    # Top features that are important in both
    combined_importance = coef_1_5b * coef_7b
    top_shared = np.argsort(combined_importance)[::-1][:10]
    logger.info("Top-10 shared important features:")
    for rank, idx in enumerate(top_shared):
        logger.info("  %d. %s (1.5B: %.4f, 7B: %.4f)",
                     rank + 1, abc_names[idx], coef_1_5b[idx], coef_7b[idx])

    # Determine distillation feasibility
    if transfer_efficiency_1to7 is not None and transfer_efficiency_1to7 > 0.85:
        distillation_verdict = "FEASIBLE"
        distillation_reason = (
            f"Transfer efficiency {transfer_efficiency_1to7:.1%} > 85%. "
            "The topological signature of correctness is largely model-universal. "
            "A distillation loss targeting PH feature alignment is viable."
        )
    elif transfer_efficiency_1to7 is not None and transfer_efficiency_1to7 > 0.65:
        distillation_verdict = "PARTIAL"
        distillation_reason = (
            f"Transfer efficiency {transfer_efficiency_1to7:.1%} (65-85%). "
            "Some features transfer, others are model-specific. "
            "Distillation would need feature selection or adapter layers."
        )
    else:
        eff_str = f"{transfer_efficiency_1to7:.1%}" if transfer_efficiency_1to7 else "N/A"
        distillation_verdict = "NOT_FEASIBLE"
        distillation_reason = (
            f"Transfer efficiency {eff_str} < 65%. "
            "The topological signature is model-specific. "
            "Per-model calibration needed; cross-model distillation unlikely to work."
        )

    logger.info("\nDistillation verdict: %s", distillation_verdict)
    logger.info("Reason: %s", distillation_reason)

    # ================================================================
    # Save results
    # ================================================================
    elapsed = time.time() - t0

    results = {
        "phase_c1_distribution": {
            "n_both_correct": int(both_correct.sum()),
            "n_both_wrong": int(both_wrong.sum()),
            "n_only_1_5b": int(only_1_5b.sum()),
            "n_only_7b": int(only_7b.sum()),
            "mean_cross_model_r": round(float(np.mean(all_cross_r)), 4),
            "effect_size_rank_rho": round(d_rank_rho, 4),
            "effect_size_rank_p": round(d_rank_p, 4),
            "top_divergent_features": feature_comparisons[:10],
        },
        "phase_c2_transfer": {
            "native_1_5b_auroc": native_1_5b_auroc,
            "native_7b_auroc": native_7b_auroc,
            "transfer_1to7_auroc": transfer_1to7_auroc,
            "transfer_1to7_ci": [t1to7_lo, t1to7_hi],
            "transfer_7to1_auroc": transfer_7to1_auroc,
            "transfer_7to1_ci": [t7to1_lo, t7to1_hi],
            "transfer_efficiency_1to7": (
                round(transfer_efficiency_1to7, 4) if transfer_efficiency_1to7 else None
            ),
            "transfer_efficiency_7to1": (
                round(transfer_efficiency_7to1, 4) if transfer_efficiency_7to1 else None
            ),
        },
        "phase_c3_distillation": {
            "verdict": distillation_verdict,
            "reason": distillation_reason,
            "coef_rank_rho": round(coef_rank_rho, 4),
            "coef_rank_p": round(coef_rank_p, 4),
            "top_shared_features": [
                {"feature": abc_names[i], "coef_1_5b": float(coef_1_5b[i]), "coef_7b": float(coef_7b[i])}
                for i in top_shared
            ],
            "feature_transfer_detail": feature_transfer,
        },
        "elapsed_minutes": round(elapsed / 60, 1),
    }

    with open(PHASE_DIR / "cross_model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(PHASE_DIR / "feature_comparisons.json", "w") as f:
        json.dump(feature_comparisons, f, indent=2)

    # ---- Final summary table ----
    logger.info("\n" + "=" * 70)
    logger.info("TRACK C: CROSS-MODEL TOPOLOGY COMPARISON")
    logger.info("=" * 70)
    logger.info("| Metric                      | Value          |")
    logger.info("|-----------------------------|----------------|")
    logger.info("| 1.5B accuracy               | %.1f%%           |", 100 * correct_1_5b.mean())
    logger.info("| 7B accuracy                 | %.1f%%           |", 100 * correct_7b.mean())
    logger.info("| Both correct                | %d              |", both_correct.sum())
    logger.info("| Only 7B correct             | %d             |", only_7b.sum())
    logger.info("| Mean cross-model feature r  | %.3f           |", np.mean(all_cross_r))
    logger.info("| Effect-size rank rho        | %.3f (p=%.3f) |", d_rank_rho, d_rank_p)
    logger.info("| 1.5B native AUROC           | %.3f           |", native_1_5b_auroc)
    logger.info("| 7B native AUROC             | %.3f           |", native_7b_auroc or 0)
    logger.info("| Transfer 1.5B→7B AUROC      | %.3f           |", transfer_1to7_auroc or 0)
    logger.info("| Transfer 7B→1.5B AUROC      | %.3f           |", transfer_7to1_auroc or 0)
    logger.info("| Distillation verdict        | %s      |", distillation_verdict)
    logger.info("| Coef rank correlation       | %.3f (p=%.3f) |", coef_rank_rho, coef_rank_p)
    logger.info("=" * 70)
    logger.info("Elapsed: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
