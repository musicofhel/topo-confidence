#!/usr/bin/env python3
"""Experiment 6: Feature Ablation — Tier Subsets + Importance + Leave-One-Out.

Answers "which features matter?" for the 44 A+B+C topo-confidence features
(AUROC=0.935 on holdout-100).

Three analyses:
  1. Tier subset ablation: all 7 A/B/C combinations + A+B+C+D reference,
     with holdout AUROC, bootstrap CIs, CV-train-400, and gating performance.
  2. Feature importance: LR coefficient magnitudes (standardized) + optional
     SHAP if available, + sklearn permutation importance.
  3. Leave-one-out: drop each of 44 features, measure AUROC change.

No GPU required — operates on precomputed features. ~30s runtime.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BASELINE_CORRECT_PATH,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    PHASE3_DIR,
    TIER_ASSIGNMENTS_PATH,
    check_correct,
    extract_answer,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    logger,
    normalize_answer,
)

OUTPUT_DIR = Path(__file__).parent / "experiment6_ablation"

# CV params matching Phase 2 exactly
CV_PARAMS_50 = dict(n_splits=50, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Helpers (reused from Exp 9)
# ---------------------------------------------------------------------------

def bootstrap_auroc(y_true: np.ndarray, scores: np.ndarray,
                    n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """AUROC with bootstrap 95% CI. Returns (auroc, ci_lo, ci_hi)."""
    rng = np.random.RandomState(seed)
    auroc = roc_auc_score(y_true, scores)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], scores[idx]))
    boot = np.array(boot)
    return float(auroc), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def majority_vote_answer(completions: list[str]) -> tuple[str, Counter]:
    """Extract majority-vote answer and vote distribution."""
    answers = [normalize_answer(extract_answer(c)) for c in completions]
    counter = Counter(answers)
    best_norm = counter.most_common(1)[0][0]
    for c in completions:
        if normalize_answer(extract_answer(c)) == best_norm:
            return extract_answer(c), counter
    return extract_answer(completions[0]), counter


def gating_sweep(scores: np.ndarray, holdout_correct: np.ndarray,
                 holdout_gen: dict, holdout_idx: np.ndarray,
                 ground_truths: list,
                 thresholds: list[float] | None = None) -> list[dict]:
    """Sweep gating thresholds: score >= tau -> trust greedy; else MV."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    greedy_arr = holdout_correct.astype(bool)
    results = []

    for tau in thresholds:
        gated = []
        for i, gi in enumerate(holdout_idx):
            if scores[i] >= tau:
                gated.append(bool(holdout_correct[i]))
            else:
                gi_str = str(gi)
                if gi_str in holdout_gen:
                    mv_ans, _ = majority_vote_answer(holdout_gen[gi_str]["completions"])
                    gated.append(check_correct(mv_ans, ground_truths[gi]))
                else:
                    gated.append(bool(holdout_correct[i]))

        gated_arr = np.array(gated, dtype=bool)
        w2r = int((~greedy_arr & gated_arr).sum())
        r2w = int((greedy_arr & ~gated_arr).sum())
        n_trust = int((scores >= tau).sum())

        results.append({
            "threshold": round(tau, 2),
            "n_trust_greedy": n_trust,
            "n_use_mv": len(holdout_idx) - n_trust,
            "correct": int(gated_arr.sum()),
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "net_gain": w2r - r2w,
        })

    return results


def fit_and_score(X_train: np.ndarray, y_train: np.ndarray,
                  X_holdout: np.ndarray, y_holdout: np.ndarray
                  ) -> tuple[float, float, float, np.ndarray, LogisticRegression, StandardScaler]:
    """Fit LR, return (auroc, ci_lo, ci_hi, scores, clf, scaler)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_holdout)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_tr, y_train)
    scores = clf.predict_proba(X_ho)[:, 1]

    auroc, ci_lo, ci_hi = bootstrap_auroc(y_holdout, scores)
    return auroc, ci_lo, ci_hi, scores, clf, scaler


def run_cv(X: np.ndarray, y: np.ndarray) -> float:
    """50-fold stratified CV AUROC on train data (matching Phase 2)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(**CV_PARAMS_50)
    clf = LogisticRegression(**LR_PARAMS)
    proba = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("Experiment 6: Feature Ablation (Tier Subsets + Importance + LOO)")
    logger.info("=" * 70)

    # ---- Step 1: Load & align data ----
    logger.info("\n[1] Loading data with SSS feature ordering fix...")

    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    _, holdout_idx = get_train_holdout_indices()
    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    holdout_correct = baseline_correct[holdout_idx].astype(int)
    logger.info("  Holdout: %d problems, %d greedy-correct (%.0f%%)",
                len(holdout_idx), holdout_correct.sum(), 100 * holdout_correct.mean())

    # SSS ordering fix (same as Exp 1/9)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(sss.split(
        np.zeros(len(baseline_correct)), baseline_correct.astype(int)
    ))

    train_sort_perm = np.argsort(train_idx_sss)
    hold_sort_perm = np.argsort(hold_idx_sss)

    train_features_78 = np.load(FEATURES_TRAIN_PATH)[train_sort_perm]
    holdout_features_78 = np.load(FEATURES_HOLDOUT_PATH)[hold_sort_perm]

    train_idx_sorted = np.sort(train_idx_sss)
    hold_idx_sorted = np.sort(hold_idx_sss)
    assert np.array_equal(hold_idx_sorted, holdout_idx), "Holdout index mismatch!"

    y_train = baseline_correct[train_idx_sorted].astype(int)

    # Load feature names and tier assignments
    with open(Path(__file__).parent / "phase1" / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(TIER_ASSIGNMENTS_PATH) as f:
        tier_map = json.load(f)

    # Load temperature generations for gating
    with open(PHASE3_DIR / "holdout_temperature_generations.json") as f:
        holdout_gen = json.load(f)

    logger.info("  Features: %d total, tier map: %d entries", len(feature_names), len(tier_map))

    # ---- Step 2: Build tier index maps ----
    logger.info("\n[2] Building tier index maps...")

    # Map feature names to tier labels for the 78-column feature space
    tier_indices = {"A": [], "B": [], "C": [], "D": []}
    for i, name in enumerate(feature_names):
        tier = tier_map.get(name, "D")
        tier_indices[tier].append(i)

    for t in "ABCD":
        logger.info("  Tier %s: %d features", t, len(tier_indices[t]))

    # Define all subsets
    subsets = {
        "A":       sorted(tier_indices["A"]),
        "B":       sorted(tier_indices["B"]),
        "C":       sorted(tier_indices["C"]),
        "A+B":     sorted(tier_indices["A"] + tier_indices["B"]),
        "A+C":     sorted(tier_indices["A"] + tier_indices["C"]),
        "B+C":     sorted(tier_indices["B"] + tier_indices["C"]),
        "A+B+C":   sorted(tier_indices["A"] + tier_indices["B"] + tier_indices["C"]),
        "A+B+C+D": list(range(len(feature_names))),
    }

    # Verify A+B+C matches get_abc_column_indices
    abc_check = get_abc_column_indices(feature_names)
    assert subsets["A+B+C"] == abc_check, (
        f"A+B+C mismatch: computed {len(subsets['A+B+C'])} vs common.py {len(abc_check)}"
    )
    logger.info("  A+B+C verified: %d features (matches common.py)", len(subsets["A+B+C"]))

    # ---- Step 3: Tier subset evaluation ----
    logger.info("\n[3] Evaluating tier subsets...")

    tier_results = {}
    per_problem_scores = {}

    for label, cols in subsets.items():
        X_tr = train_features_78[:, cols]
        X_ho = holdout_features_78[:, cols]

        auroc, ci_lo, ci_hi, scores, clf, scaler = fit_and_score(
            X_tr, y_train, X_ho, holdout_correct
        )

        # CV on train-400
        cv_auroc = run_cv(X_tr, y_train)

        # Gating sweep
        sweep = gating_sweep(scores, holdout_correct, holdout_gen, holdout_idx, ground_truths)
        best_gate = max(sweep, key=lambda x: x["net_gain"])

        tier_results[label] = {
            "n_features": len(cols),
            "feature_indices": cols,
            "holdout_auroc": round(auroc, 4),
            "holdout_auroc_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "cv_train400_auroc": round(cv_auroc, 6),
            "gating_sweep": sweep,
            "best_gate": best_gate,
        }

        per_problem_scores[label] = [round(float(s), 6) for s in scores]

        bg = best_gate
        logger.info("  %-8s %2d feat  AUROC=%.3f [%.3f-%.3f]  CV=%.3f  "
                     "best gate: net=%+d (%dW->R/%dR->W)",
                     label, len(cols), auroc, ci_lo, ci_hi, cv_auroc,
                     bg["net_gain"], bg["wrong_to_right"], bg["right_to_wrong"])

    # ---- Step 4: Feature importance on A+B+C model ----
    logger.info("\n[4] Feature importance analysis (A+B+C model)...")

    abc_cols = subsets["A+B+C"]
    X_tr_abc = train_features_78[:, abc_cols]
    X_ho_abc = holdout_features_78[:, abc_cols]

    scaler_abc = StandardScaler()
    X_tr_scaled = scaler_abc.fit_transform(X_tr_abc)
    X_ho_scaled = scaler_abc.transform(X_ho_abc)

    clf_abc = LogisticRegression(**LR_PARAMS)
    clf_abc.fit(X_tr_scaled, y_train)

    # (a) Standardized LR coefficients
    coefs = clf_abc.coef_[0]
    abs_coefs = np.abs(coefs)
    coef_ranking = np.argsort(abs_coefs)[::-1]

    abc_feature_names = [feature_names[c] for c in abc_cols]
    abc_feature_tiers = [tier_map.get(feature_names[c], "?") for c in abc_cols]

    logger.info("\n  Top-15 features by |coefficient| (standardized):")
    logger.info("  %4s  %-35s  %4s  %8s  %8s", "Rank", "Feature", "Tier", "|Coef|", "Coef")
    logger.info("  " + "-" * 68)
    for rank, j in enumerate(coef_ranking[:15]):
        logger.info("  %4d  %-35s  %4s  %8.4f  %+8.4f",
                     rank + 1, abc_feature_names[j], abc_feature_tiers[j],
                     abs_coefs[j], coefs[j])

    # (b) Try SHAP (optional)
    shap_values_mean = None
    try:
        import shap
        explainer = shap.LinearExplainer(clf_abc, X_tr_scaled)
        shap_vals = explainer.shap_values(X_ho_scaled)
        shap_values_mean = np.abs(shap_vals).mean(axis=0)
        logger.info("\n  SHAP values computed successfully (LinearExplainer)")
    except ImportError:
        logger.info("\n  SHAP not installed — using LR coefficients only (equivalent for linear model)")
    except Exception as e:
        logger.warning("  SHAP failed: %s — using LR coefficients only", e)

    # (c) Permutation importance on holdout
    logger.info("\n  Computing permutation importance (10 repeats)...")
    from sklearn.inspection import permutation_importance
    perm_result = permutation_importance(
        clf_abc, X_ho_scaled, holdout_correct,
        n_repeats=10, random_state=42, scoring="roc_auc"
    )
    perm_ranking = np.argsort(perm_result.importances_mean)[::-1]

    logger.info("\n  Top-15 features by permutation importance:")
    logger.info("  %4s  %-35s  %4s  %10s  %10s", "Rank", "Feature", "Tier", "Mean drop", "Std")
    logger.info("  " + "-" * 72)
    for rank, j in enumerate(perm_ranking[:15]):
        logger.info("  %4d  %-35s  %4s  %10.4f  %10.4f",
                     rank + 1, abc_feature_names[j], abc_feature_tiers[j],
                     perm_result.importances_mean[j], perm_result.importances_std[j])

    # ---- Step 5: Leave-one-out on 44 features ----
    logger.info("\n[5] Leave-one-out feature ablation (44 features)...")

    # Reference AUROC (full A+B+C)
    ref_scores = clf_abc.predict_proba(X_ho_scaled)[:, 1]
    ref_auroc = roc_auc_score(holdout_correct, ref_scores)

    loo_results = []
    for j in range(len(abc_cols)):
        cols_minus_j = [c for c in range(len(abc_cols)) if c != j]
        X_tr_sub = X_tr_abc[:, cols_minus_j]
        X_ho_sub = X_ho_abc[:, cols_minus_j]

        scaler_j = StandardScaler()
        X_tr_j = scaler_j.fit_transform(X_tr_sub)
        X_ho_j = scaler_j.transform(X_ho_sub)

        clf_j = LogisticRegression(**LR_PARAMS)
        clf_j.fit(X_tr_j, y_train)
        scores_j = clf_j.predict_proba(X_ho_j)[:, 1]
        auroc_j = roc_auc_score(holdout_correct, scores_j)

        loo_results.append({
            "feature_index": j,
            "feature_name": abc_feature_names[j],
            "tier": abc_feature_tiers[j],
            "auroc_without": round(auroc_j, 6),
            "delta_auroc": round(ref_auroc - auroc_j, 6),
        })

    # Sort by delta (largest drop first = most important)
    loo_sorted = sorted(loo_results, key=lambda x: x["delta_auroc"], reverse=True)

    logger.info("\n  Reference AUROC (all 44): %.4f", ref_auroc)
    logger.info("\n  Top-15 features by LOO impact (removing hurts most):")
    logger.info("  %4s  %-35s  %4s  %10s  %10s", "Rank", "Feature", "Tier", "AUROC w/o", "DAUROC")
    logger.info("  " + "-" * 72)
    for rank, entry in enumerate(loo_sorted[:15]):
        logger.info("  %4d  %-35s  %4s  %10.4f  %+10.4f",
                     rank + 1, entry["feature_name"], entry["tier"],
                     entry["auroc_without"], entry["delta_auroc"])

    # Also show bottom 5 (removing helps)
    logger.info("\n  Bottom-5 features (removing helps or is neutral):")
    for entry in loo_sorted[-5:]:
        logger.info("  %-35s  %4s  AUROC=%.4f  D=%+.4f",
                     entry["feature_name"], entry["tier"],
                     entry["auroc_without"], entry["delta_auroc"])

    # ---- Step 6: Output ----
    logger.info("\n[6] Saving results...")

    # Build importance table
    importance_table = []
    for j in range(len(abc_cols)):
        entry = {
            "feature_index": j,
            "feature_name": abc_feature_names[j],
            "tier": abc_feature_tiers[j],
            "lr_coef": round(float(coefs[j]), 6),
            "lr_abs_coef": round(float(abs_coefs[j]), 6),
            "lr_coef_rank": int(np.where(coef_ranking == j)[0][0]) + 1,
            "perm_importance_mean": round(float(perm_result.importances_mean[j]), 6),
            "perm_importance_std": round(float(perm_result.importances_std[j]), 6),
            "perm_rank": int(np.where(perm_ranking == j)[0][0]) + 1,
            "loo_auroc_without": next(r["auroc_without"] for r in loo_results if r["feature_index"] == j),
            "loo_delta_auroc": next(r["delta_auroc"] for r in loo_results if r["feature_index"] == j),
        }
        if shap_values_mean is not None:
            entry["shap_mean_abs"] = round(float(shap_values_mean[j]), 6)
        importance_table.append(entry)

    # Summary statistics
    tier_contribution = {}
    for t in "ABC":
        tier_feat_idx = [j for j in range(len(abc_cols)) if abc_feature_tiers[j] == t]
        tier_contribution[t] = {
            "n_features": len(tier_feat_idx),
            "mean_abs_coef": round(float(np.mean(abs_coefs[tier_feat_idx])), 6) if tier_feat_idx else 0,
            "total_abs_coef": round(float(np.sum(abs_coefs[tier_feat_idx])), 6) if tier_feat_idx else 0,
            "mean_perm_importance": round(float(np.mean(perm_result.importances_mean[tier_feat_idx])), 6) if tier_feat_idx else 0,
            "mean_loo_delta": round(float(np.mean([loo_results[j]["delta_auroc"] for j in tier_feat_idx])), 6) if tier_feat_idx else 0,
        }

    output = {
        "experiment": "feature_ablation",
        "description": (
            "Feature ablation: all 7 tier combinations (A/B/C) with holdout AUROC, "
            "bootstrap CIs, 50-fold CV, and gating sweep. Plus feature importance "
            "(LR coefficients, permutation importance, leave-one-out) on 44-feature A+B+C model."
        ),
        "n_holdout": len(holdout_idx),
        "n_greedy_correct": int(holdout_correct.sum()),
        "reference_auroc_abc": round(ref_auroc, 6),
        "tier_ablation": tier_results,
        "feature_importance": importance_table,
        "tier_contribution_summary": tier_contribution,
        "loo_results": loo_sorted,
        "phase2_comparison": {
            "note": (
                "Phase 2 ablation.py re-extracted features from trajectories with train-PCA. "
                "This experiment uses precomputed .npy features with SSS ordering fix. "
                "Small AUROC differences expected (different PCA basis)."
            ),
            "phase2_cv": {"A": 0.789302, "A+B": 0.907517, "A+B+C": 0.921887, "A+B+C+D": 0.927291},
            "phase2_holdout": {"A": 0.827375, "A+B": 0.931563, "A+B+C": 0.940756, "A+B+C+D": 0.949949},
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_DIR / "tier_ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(OUTPUT_DIR / "per_problem_tier_scores.json", "w") as f:
        json.dump({
            "holdout_indices": [int(gi) for gi in holdout_idx],
            "holdout_correct": [int(c) for c in holdout_correct],
            "scores_by_tier": per_problem_scores,
        }, f, indent=2)

    with open(OUTPUT_DIR / "feature_importance.json", "w") as f:
        json.dump({
            "abc_feature_names": abc_feature_names,
            "abc_feature_tiers": abc_feature_tiers,
            "importance_table": importance_table,
            "tier_contribution_summary": tier_contribution,
            "loo_sorted": loo_sorted,
        }, f, indent=2)

    # ---- Console summary tables ----
    logger.info("")
    logger.info("=" * 95)
    logger.info("TABLE 1: Tier Subset Comparison")
    logger.info("=" * 95)
    logger.info("")
    header = f"  {'Subset':<10} {'N_feat':>6} {'Holdout AUROC [95% CI]':<26} {'CV-train-400':>12} {'Best gate':>25}"
    logger.info(header)
    logger.info("  " + "-" * 90)

    for label in ["A", "B", "C", "A+B", "A+C", "B+C", "A+B+C", "A+B+C+D"]:
        r = tier_results[label]
        auroc_str = f"{r['holdout_auroc']:.3f} [{r['holdout_auroc_ci_95'][0]:.3f}-{r['holdout_auroc_ci_95'][1]:.3f}]"
        bg = r["best_gate"]
        gate_str = f"net={bg['net_gain']:+d} ({bg['wrong_to_right']}W->R/{bg['right_to_wrong']}R->W)"
        logger.info(f"  {label:<10} {r['n_features']:>6} {auroc_str:<26} {r['cv_train400_auroc']:>12.4f} {gate_str:>25}")

    logger.info("")
    logger.info("=" * 95)
    logger.info("TABLE 2: Top-10 Features by Importance (A+B+C model)")
    logger.info("=" * 95)
    logger.info("")
    logger.info("  %4s  %-30s  %4s  %8s  %10s  %10s",
                "Rank", "Feature", "Tier", "|Coef|", "Perm Imp", "LOO DAUROC")
    logger.info("  " + "-" * 80)

    # Combined ranking: use coef rank
    for rank, j in enumerate(coef_ranking[:10]):
        loo_d = next(r["delta_auroc"] for r in loo_results if r["feature_index"] == j)
        logger.info("  %4d  %-30s  %4s  %8.4f  %10.4f  %+10.4f",
                     rank + 1, abc_feature_names[j], abc_feature_tiers[j],
                     abs_coefs[j], perm_result.importances_mean[j], loo_d)

    logger.info("")
    logger.info("=" * 95)
    logger.info("TABLE 3: Tier Contribution Summary")
    logger.info("=" * 95)
    logger.info("")
    logger.info("  %4s  %6s  %12s  %12s  %14s  %12s",
                "Tier", "N_feat", "Mean|Coef|", "Total|Coef|", "Mean Perm Imp", "Mean LOO D")
    logger.info("  " + "-" * 70)
    for t in "ABC":
        tc = tier_contribution[t]
        logger.info("  %4s  %6d  %12.4f  %12.4f  %14.4f  %+12.4f",
                     t, tc["n_features"], tc["mean_abs_coef"], tc["total_abs_coef"],
                     tc["mean_perm_importance"], tc["mean_loo_delta"])

    logger.info("")
    logger.info("Results saved to: %s", OUTPUT_DIR)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
