#!/usr/bin/env python3
"""Phase A2: Cross-Validated Evaluation of Selection Strategies.

5-fold stratified CV over train-400. For each fold, fit completion scorer
on fold-train, evaluate 4 selection strategies on fold-test. Select and
lock the best strategy before holdout evaluation.

Input: Phase A1 artifacts (completion_scores.json, completion_features.npz)
Output: pathway4/track_a/phase2/ — cv_results.json, strategy_comparison.json,
        locked_strategy.json

Runtime: ~10 minutes CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import (
    BASELINE_CORRECT_PATH,
    COMPLETIONS_PER_PROBLEM,
    PHASE1_DIR,
    PHASE2_DIR,
    TEMP_GEN_PATH,
    bayesian_p_improvement,
    bootstrap_ci_net_gain,
    check_correct,
    create_train400_folds,
    extract_answer,
    fit_completion_scorer,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    logger,
    mcnemar_mid_p,
    normalize_answer,
    score_completions,
    select_confidence_weighted_vote,
    select_majority_vote,
    select_max_confidence,
    select_random,
)

STRATEGIES = ["random", "max_confidence", "majority_vote", "confidence_weighted_vote"]
N_RANDOM_TRIALS = 50  # Average over multiple random trials for stable baseline


def evaluate_fold(
    fold: dict,
    completion_scores: dict,
    completion_features: np.ndarray,
    completion_correct: np.ndarray,
    completion_map: np.ndarray,
    feature_names: list[str],
    baseline_correct: np.ndarray,
    train_idx: np.ndarray,
    ground_truths: list[str],
    temp_gen: dict,
) -> dict:
    """Evaluate all selection strategies on one CV fold.

    Args:
        fold: Fold dict with train_local, test_local, train_global, test_global
        completion_scores: {global_idx: {scores, correct, ...}} from Phase A1
        completion_features: (N_total_completions, 78) features
        completion_correct: (N_total_completions,) bool labels
        completion_map: (N_total_completions, 2) (global_idx, completion_idx)
        feature_names: list of 78 feature names
        baseline_correct: (500,) greedy correctness
        train_idx: (400,) global train indices
        ground_truths: list of 500 ground truth answers
        temp_gen: full temperature generations dict

    Returns:
        Dict with per-strategy results for this fold.
    """
    fold_idx = fold["fold"]
    test_global = fold["test_global"]
    train_global = set(fold["train_global"])

    # Build index mapping: global_idx -> rows in completion_features
    # completion_map has (global_idx, completion_idx) for each row
    idx_to_rows: dict[int, list[int]] = {}
    for row, (gi, ci) in enumerate(completion_map):
        gi = int(gi)
        if gi not in idx_to_rows:
            idx_to_rows[gi] = []
        idx_to_rows[gi].append(row)

    # Collect fold-train completion features and labels
    train_rows = []
    for gi in fold["train_global"]:
        if gi in idx_to_rows:
            train_rows.extend(idx_to_rows[gi])

    if len(train_rows) == 0:
        logger.warning("  Fold %d: no train completions found!", fold_idx)
        return {}

    train_features = completion_features[train_rows]
    train_correct = completion_correct[train_rows]
    logger.info(
        "  Fold %d: train completions=%d (%.1f%% correct), test problems=%d",
        fold_idx, len(train_rows), 100 * train_correct.mean(), len(test_global),
    )

    # Fit completion scorer on fold-train
    scaler, clf, abc_indices = fit_completion_scorer(
        train_features, train_correct, feature_names
    )

    # Evaluate on fold-test problems
    rng = np.random.default_rng(42 + fold_idx)
    results_by_strategy: dict[str, dict] = {}

    for strategy in STRATEGIES:
        greedy_correct_test = []
        selected_correct_test = []

        for gi in test_global:
            gc = bool(baseline_correct[gi])
            greedy_correct_test.append(gc)

            if gc:
                # Greedy is correct — keep it, don't risk selection
                selected_correct_test.append(True)
                continue

            # Greedy is wrong — try selection
            if str(gi) not in completion_scores:
                # No completions scored (shouldn't happen for wrong-greedy train problems)
                selected_correct_test.append(False)
                continue

            cs_data = completion_scores[str(gi)]
            completions = temp_gen[str(gi)]["completions"]
            gt = ground_truths[gi]

            if strategy == "random":
                # Average over N_RANDOM_TRIALS
                correct_count = 0
                for _ in range(N_RANDOM_TRIALS):
                    _, is_correct = select_random(
                        np.array(cs_data["scores"]), completions, gt, rng
                    )
                    correct_count += int(is_correct)
                # Probabilistic: use majority of trials
                selected_correct_test.append(correct_count > N_RANDOM_TRIALS / 2)

            elif strategy == "max_confidence":
                # Score with fold-fitted model (not Phase A1's full-data scorer)
                if gi in idx_to_rows:
                    rows = idx_to_rows[gi]
                    fold_scores = score_completions(
                        scaler, clf, abc_indices, completion_features[rows]
                    )
                else:
                    fold_scores = np.array(cs_data["scores"])
                _, is_correct = select_max_confidence(fold_scores, completions, gt)
                selected_correct_test.append(is_correct)

            elif strategy == "majority_vote":
                _, is_correct = select_majority_vote(completions, gt)
                selected_correct_test.append(is_correct)

            elif strategy == "confidence_weighted_vote":
                if gi in idx_to_rows:
                    rows = idx_to_rows[gi]
                    fold_scores = score_completions(
                        scaler, clf, abc_indices, completion_features[rows]
                    )
                else:
                    fold_scores = np.array(cs_data["scores"])
                _, is_correct = select_confidence_weighted_vote(
                    fold_scores, completions, gt
                )
                selected_correct_test.append(is_correct)

        greedy_arr = np.array(greedy_correct_test, dtype=bool)
        selected_arr = np.array(selected_correct_test, dtype=bool)

        w2r = int((~greedy_arr & selected_arr).sum())
        r2w = int((greedy_arr & ~selected_arr).sum())
        net_gain = w2r - r2w

        results_by_strategy[strategy] = {
            "net_gain": net_gain,
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "baseline_correct": int(greedy_arr.sum()),
            "selected_correct": int(selected_arr.sum()),
            "n_test": len(test_global),
        }

    return results_by_strategy


def main() -> None:
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Phase A2: 5-Fold CV Evaluation ===")

    # ---- Load data ----
    logger.info("Loading Phase A1 artifacts...")
    with open(PHASE1_DIR / "completion_scores.json") as f:
        completion_scores = json.load(f)
    logger.info("  Completion scores: %d problems", len(completion_scores))

    data = np.load(PHASE1_DIR / "completion_features.npz")
    completion_features = data["features"]
    completion_correct = data["correct"]
    completion_map = data["completion_map"]
    logger.info("  Completion features: %s", completion_features.shape)

    with open(PHASE1_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    logger.info("Loading temperature generations...")
    with open(TEMP_GEN_PATH) as f:
        temp_gen = json.load(f)

    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()
    y_train = baseline_correct[train_idx].astype(bool)

    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    # ---- Create folds ----
    logger.info("Creating 5-fold stratified splits over train-400...")
    folds = create_train400_folds(y_train, train_idx)
    folds_path = PHASE2_DIR / "fold_assignments_train400.json"
    with open(folds_path, "w") as f:
        json.dump(folds, f, indent=2)
    logger.info("  Saved fold assignments: %s", folds_path)

    # ---- Evaluate each fold ----
    all_fold_results: list[dict] = []

    for fold in folds:
        logger.info("--- Fold %d ---", fold["fold"])
        fold_results = evaluate_fold(
            fold, completion_scores, completion_features, completion_correct,
            completion_map, feature_names, baseline_correct, train_idx,
            ground_truths, temp_gen,
        )
        all_fold_results.append(fold_results)

    # ---- Aggregate across folds ----
    logger.info("\n=== Aggregating results across folds ===")
    strategy_summary: dict[str, dict] = {}

    for strategy in STRATEGIES:
        gains = [fr[strategy]["net_gain"] for fr in all_fold_results]
        w2r_total = sum(fr[strategy]["wrong_to_right"] for fr in all_fold_results)
        r2w_total = sum(fr[strategy]["right_to_wrong"] for fr in all_fold_results)

        mean_gain = float(np.mean(gains))
        std_gain = float(np.std(gains))

        # Bayesian P(improvement) on total counts
        p_improve = bayesian_p_improvement(w2r_total, r2w_total)

        strategy_summary[strategy] = {
            "per_fold_net_gain": gains,
            "mean_net_gain": round(mean_gain, 2),
            "std_net_gain": round(std_gain, 2),
            "total_wrong_to_right": w2r_total,
            "total_right_to_wrong": r2w_total,
            "total_net_gain": w2r_total - r2w_total,
            "bayesian_p_improvement": round(p_improve, 4),
        }

        logger.info(
            "  %-30s: mean=%.1f ± %.1f, W→R=%d, R→W=%d, P(improve)=%.3f",
            strategy, mean_gain, std_gain, w2r_total, r2w_total, p_improve,
        )

    # ---- Oracle ceiling ----
    logger.info("\nComputing oracle ceiling...")
    oracle_gains = []
    for fold in folds:
        n_solvable = 0
        for gi in fold["test_global"]:
            if not baseline_correct[gi]:
                data = temp_gen.get(str(gi))
                if data and sum(data["correct"]) > 0:
                    n_solvable += 1
        oracle_gains.append(n_solvable)

    oracle_summary = {
        "per_fold_oracle_gain": oracle_gains,
        "mean_oracle_gain": round(float(np.mean(oracle_gains)), 2),
        "total_oracle_gain": sum(oracle_gains),
        "description": "Maximum possible net gain if oracle selector always picks correct completion",
    }
    logger.info(
        "  Oracle ceiling: mean=%.1f per fold, total=%d across all folds",
        float(np.mean(oracle_gains)), sum(oracle_gains),
    )

    # ---- Save CV results ----
    cv_results = {
        "strategies": strategy_summary,
        "oracle": oracle_summary,
        "folds": all_fold_results,
        "n_folds": len(folds),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cv_results_path = PHASE2_DIR / "cv_results.json"
    with open(cv_results_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.info("  Saved CV results: %s", cv_results_path)

    # ---- Strategy comparison table ----
    comparison = {
        "ranking": sorted(
            strategy_summary.items(),
            key=lambda x: x[1]["mean_net_gain"],
            reverse=True,
        ),
        "best_strategy": max(
            strategy_summary.items(),
            key=lambda x: x[1]["mean_net_gain"],
        )[0],
        "oracle_mean": oracle_summary["mean_oracle_gain"],
    }
    comparison_path = PHASE2_DIR / "strategy_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("  Saved strategy comparison: %s", comparison_path)

    # ---- Lock best strategy ----
    best_name = comparison["best_strategy"]
    best_data = strategy_summary[best_name]
    locked = {
        "selected_strategy": best_name,
        "mean_net_gain": best_data["mean_net_gain"],
        "std_net_gain": best_data["std_net_gain"],
        "bayesian_p_improvement": best_data["bayesian_p_improvement"],
        "cv_n_splits": len(folds),
        "cv_random_state": 42,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    locked_path = PHASE2_DIR / "locked_strategy.json"
    with open(locked_path, "w") as f:
        json.dump(locked, f, indent=2)
    logger.info("  Locked strategy: %s (mean gain=%.1f)", best_name, best_data["mean_net_gain"])

    # ---- Final summary ----
    logger.info("\n=== Phase A2 Summary ===")
    logger.info("Strategies ranked by mean net gain:")
    for rank, (name, data) in enumerate(comparison["ranking"], 1):
        logger.info(
            "  %d. %-30s: %.1f ± %.1f (P=%.3f)",
            rank, name, data["mean_net_gain"], data["std_net_gain"],
            data["bayesian_p_improvement"],
        )
    logger.info("Oracle ceiling: %.1f per fold", oracle_summary["mean_oracle_gain"])
    logger.info("Locked: %s", best_name)
    logger.info("Total time: %.1f seconds", time.time() - t_start)
    logger.info("Phase A2 complete.")


if __name__ == "__main__":
    main()
