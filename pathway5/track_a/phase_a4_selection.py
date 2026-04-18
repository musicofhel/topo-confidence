#!/usr/bin/env python3
"""Phase A4: GSM8K Selection Strategy Evaluation.

CPU script. Computes per-completion topo features from Phase A3 trajectories,
evaluates all selection strategies (random, BoN, MV, weighted vote, gated),
and builds the final GSM8K comparison table.

Runtime: ~30 min - 2 hours CPU (PH on ~8K completions).
Input: pathway5/track_a/phase_a1/, phase_a2/, phase_a3/
Output: pathway5/track_a/phase_a4/
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    COMPLETIONS_PER_PROBLEM,
    LR_PARAMS,
    SEED,
    TRACK_A_DIR,
    bootstrap_auroc,
    check_correct,
    evaluate_all_strategies,
    evaluate_gating_strategies,
    extract_answer,
    extract_topo_features,
    fit_completion_scorer,
    fit_pca_on_trajectories,
    get_abc_column_indices,
    normalize_answer,
    score_completions,
    select_confidence_weighted_vote,
    select_majority_vote,
    select_max_confidence,
    select_random,
    logger,
)

PHASE_A1_DIR = TRACK_A_DIR / "phase_a1"
PHASE_A2_DIR = TRACK_A_DIR / "phase_a2"
PHASE_A3_DIR = TRACK_A_DIR / "phase_a3"
PHASE_DIR = TRACK_A_DIR / "phase_a4"


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load artifacts ----
    logger.info("=== Loading artifacts ===")

    baseline_correct = np.load(PHASE_A1_DIR / "gsm8k_baseline_correct.npy")
    train_idx = np.load(PHASE_A2_DIR / "gsm8k_train_idx.npy")
    test_idx = np.load(PHASE_A2_DIR / "gsm8k_test_idx.npy")
    topo_scores = np.load(PHASE_A2_DIR / "gsm8k_topo_scores_fresh.npy")

    with open(PHASE_A3_DIR / "temperature_generations.json") as f:
        generations = json.load(f)

    with open(PHASE_A2_DIR / "gsm8k_feature_names.json") as f:
        feature_names = json.load(f)

    # Load GSM8K for ground truths
    from common import load_gsm8k, get_gsm8k_prompts_and_ground_truths
    problems = load_gsm8k()
    prompts, ground_truths = get_gsm8k_prompts_and_ground_truths(problems)

    n_test = len(test_idx)
    n_greedy_correct_test = int(baseline_correct[test_idx].sum())
    logger.info(
        "Test split: %d problems, %d greedy correct",
        n_test, n_greedy_correct_test,
    )

    # ---- Step 2: Per-completion topo feature extraction ----
    logger.info("=== Per-completion topo feature extraction ===")

    # Load greedy trajectories for PCA fitting (use train split only)
    greedy_traj_data = np.load(PHASE_A1_DIR / "gsm8k_trajectories.npz")
    greedy_trajectories_train = [greedy_traj_data[f"traj_{i}"] for i in train_idx]
    pca = fit_pca_on_trajectories(greedy_trajectories_train)

    # Load completion trajectories
    comp_data = np.load(PHASE_A3_DIR / "completion_trajectories.npz")

    # Build per-problem completion features
    all_comp_trajectories = []
    all_comp_layer_states = []
    all_comp_correct = []
    all_comp_problem_ids = []

    for idx in test_idx:
        idx_str = str(int(idx))
        if idx_str not in generations:
            continue

        gen = generations[idx_str]
        for c_idx in range(len(gen["completions"])):
            traj_key = f"traj_{idx}_{c_idx}"
            ls_key = f"ls_{idx}_{c_idx}"
            if traj_key in comp_data and ls_key in comp_data:
                all_comp_trajectories.append(comp_data[traj_key])
                all_comp_layer_states.append(comp_data[ls_key])
                all_comp_correct.append(gen["correct"][c_idx])
                all_comp_problem_ids.append(int(idx))

    n_completions = len(all_comp_trajectories)
    logger.info("Loaded %d completion trajectories", n_completions)

    if n_completions == 0:
        logger.error("No completion trajectories found. Cannot proceed.")
        return

    # Stack layer states
    comp_layer_states = np.stack(all_comp_layer_states, axis=0)
    comp_correct = np.array(all_comp_correct)
    comp_problem_ids = np.array(all_comp_problem_ids)

    # Extract features using pre-fitted PCA
    logger.info("Extracting topo features for %d completions...", n_completions)
    t_feat = time.time()
    comp_features, comp_feature_names = extract_topo_features(
        all_comp_trajectories, comp_layer_states, prefitted_pca=pca,
    )
    logger.info(
        "Completion features: %s, took %.1f min",
        comp_features.shape, (time.time() - t_feat) / 60,
    )

    np.save(PHASE_DIR / "completion_features.npy", comp_features)
    np.save(PHASE_DIR / "completion_correct.npy", comp_correct)
    np.save(PHASE_DIR / "completion_problem_ids.npy", comp_problem_ids)

    # ---- Step 3: Fit per-completion scorer ----
    logger.info("=== Fitting per-completion scorer ===")

    # Split completions by problem: train on train-idx completions, score test-idx
    # But we only have completions for test_idx problems. Use 5-fold CV on test.
    abc_indices = get_abc_column_indices(comp_feature_names)

    # Global per-completion AUROC (5-fold CV within test problems)
    unique_problems = np.unique(comp_problem_ids)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Problem-level CV: split problems, not completions
    problem_correct = np.array([
        baseline_correct[p] for p in unique_problems
    ])

    cv_scores = np.zeros(n_completions)
    for fold_idx, (p_train, p_test) in enumerate(skf.split(unique_problems, problem_correct)):
        fold_train_problems = set(unique_problems[p_train])
        fold_test_problems = set(unique_problems[p_test])

        train_mask = np.array([pid in fold_train_problems for pid in comp_problem_ids])
        test_mask = np.array([pid in fold_test_problems for pid in comp_problem_ids])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(np.unique(comp_correct[train_mask])) < 2:
            continue

        scaler_f = StandardScaler()
        X_train_f = scaler_f.fit_transform(comp_features[train_mask][:, abc_indices])
        X_test_f = scaler_f.transform(comp_features[test_mask][:, abc_indices])

        clf_f = LogisticRegression(**LR_PARAMS)
        clf_f.fit(X_train_f, comp_correct[train_mask].astype(int))
        cv_scores[test_mask] = clf_f.predict_proba(X_test_f)[:, 1]

    # Global AUROC
    global_auroc = roc_auc_score(comp_correct, cv_scores)
    global_auroc_val, global_ci_lo, global_ci_hi = bootstrap_auroc(comp_correct, cv_scores)
    logger.info(
        "Global per-completion AUROC (CV): %.3f [%.3f-%.3f]",
        global_auroc_val, global_ci_lo, global_ci_hi,
    )

    # ---- Step 4: Within-problem AUROC ----
    logger.info("=== Within-problem AUROC ===")
    within_aurocs = []
    for pid in unique_problems:
        mask = comp_problem_ids == pid
        y_p = comp_correct[mask]
        s_p = cv_scores[mask]
        if len(np.unique(y_p)) < 2:
            continue
        within_aurocs.append(roc_auc_score(y_p, s_p))

    within_mean = np.mean(within_aurocs) if within_aurocs else 0.0
    within_median = np.median(within_aurocs) if within_aurocs else 0.0
    n_mixed = len(within_aurocs)
    frac_above_50 = np.mean(np.array(within_aurocs) > 0.5) if within_aurocs else 0.0
    logger.info(
        "Within-problem AUROC: mean=%.3f, median=%.3f, N_mixed=%d, %%>0.50=%.0f%%",
        within_mean, within_median, n_mixed, 100 * frac_above_50,
    )

    # ---- Step 5: Build per-problem scores for selection strategies ----
    logger.info("=== Building per-problem completion scores ===")

    # Fit final scorer on all completions for strategy evaluation
    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(comp_features[:, abc_indices])
    clf_all = LogisticRegression(**LR_PARAMS)
    clf_all.fit(X_all, comp_correct.astype(int))
    final_scores = clf_all.predict_proba(X_all)[:, 1]

    # Build per-problem score dicts
    completions_by_problem = {}
    scores_by_problem = {}
    for idx_str, gen in generations.items():
        completions_by_problem[idx_str] = gen["completions"]

    # Map completion scores back to problems
    for i, (pid, score) in enumerate(zip(comp_problem_ids, final_scores)):
        pid_str = str(pid)
        if pid_str not in scores_by_problem:
            scores_by_problem[pid_str] = []
        scores_by_problem[pid_str].append(float(score))

    # ---- Step 6: Evaluate all selection strategies ----
    logger.info("=== Selection strategy evaluation ===")

    strategy_results = evaluate_all_strategies(
        completions_by_problem, scores_by_problem,
        ground_truths, test_idx, baseline_correct,
    )

    for name, res in strategy_results.items():
        logger.info(
            "  %s: correct=%s, net=%s, W→R=%s, R→W=%s",
            name, res["correct"], res["net_gain"], res["w_to_r"], res["r_to_w"],
        )

    # ---- Step 7: Gating strategies ----
    logger.info("=== Gating strategy evaluation ===")

    gating_results = evaluate_gating_strategies(
        topo_scores, completions_by_problem,
        ground_truths, test_idx, baseline_correct,
    )

    for g in gating_results:
        logger.info(
            "  tau=%.1f: correct=%d, net=%+d, savings=%.1f%%",
            g["tau"], g["correct"], g["net_gain"], g["savings_pct"],
        )

    # ---- Step 8: Sampling-based baselines AUROC ----
    logger.info("=== Sampling-based baseline AUROC ===")
    with open(PHASE_A3_DIR / "gsm8k_sampling_baselines.json") as f:
        sampling_baselines = json.load(f)

    y_test_arr = baseline_correct[test_idx]
    sampling_aurocs = {}
    for metric_name in ["p_majority", "vote_margin", "inv_answer_diversity", "neg_agreement_entropy"]:
        metric_scores = []
        for idx in test_idx:
            idx_str = str(int(idx))
            if idx_str in sampling_baselines:
                metric_scores.append(sampling_baselines[idx_str][metric_name])
            else:
                metric_scores.append(0.0)
        metric_scores = np.array(metric_scores)
        try:
            auroc_val, ci_lo, ci_hi = bootstrap_auroc(y_test_arr, metric_scores)
            sampling_aurocs[metric_name] = {"auroc": auroc_val, "ci_lo": ci_lo, "ci_hi": ci_hi}
            logger.info("  %s: AUROC %.3f [%.3f-%.3f]", metric_name, auroc_val, ci_lo, ci_hi)
        except ValueError:
            sampling_aurocs[metric_name] = {"auroc": None}

    # ---- Step 9: Oracle ceiling ----
    oracle_correct = n_greedy_correct_test
    for idx in test_idx:
        idx_str = str(int(idx))
        if baseline_correct[idx]:
            continue
        if idx_str in generations and generations[idx_str]["n_correct"] > 0:
            oracle_correct += 1
    oracle_ceiling = oracle_correct - n_greedy_correct_test
    logger.info("Oracle ceiling: +%d (%d/%d correct)", oracle_ceiling, oracle_correct, n_test)

    # ---- Step 10: Final results ----
    elapsed = time.time() - t0

    # Load Phase A2 results for comparison
    with open(PHASE_A2_DIR / "phase_a2_results.json") as f:
        a2_results = json.load(f)

    results = {
        "dataset": "GSM8K",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_test": n_test,
        "n_greedy_correct_test": n_greedy_correct_test,
        "greedy_accuracy_test": round(n_greedy_correct_test / n_test, 4),
        "oracle_ceiling": oracle_ceiling,
        "oracle_correct": oracle_correct,
        "per_completion": {
            "global_auroc": global_auroc_val,
            "global_ci": [global_ci_lo, global_ci_hi],
            "within_problem_auroc_mean": round(within_mean, 3),
            "within_problem_auroc_median": round(within_median, 3),
            "n_mixed_problems": n_mixed,
            "frac_above_50": round(frac_above_50, 3),
        },
        "selection_strategies": strategy_results,
        "gating_strategies": gating_results,
        "sampling_baselines_auroc": sampling_aurocs,
        "topo_auroc_prompt_level": a2_results["topo_confidence"]["fresh_pca"]["test_auroc"],
        "elapsed_minutes": round(elapsed / 60, 1),
        "comparison_table": {
            "greedy_acc": round(n_greedy_correct_test / n_test, 3),
            "topo_auroc": a2_results["topo_confidence"]["fresh_pca"]["test_auroc"],
            "best_logprob_auroc": a2_results.get("comparison_with_math500", {}).get("gsm8k_best_logprob_auroc"),
            "mv_net_gain": strategy_results.get("majority_vote", {}).get("net_gain"),
            "weighted_vote_net_gain": strategy_results.get("confidence_weighted_vote", {}).get("net_gain"),
            "weighted_vote_r_to_w": strategy_results.get("confidence_weighted_vote", {}).get("r_to_w"),
            "oracle_ceiling": oracle_ceiling,
        },
    }

    with open(PHASE_DIR / "phase_a4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Print final comparison table ----
    logger.info("\n" + "=" * 70)
    logger.info("TRACK A FINAL COMPARISON: 1.5B × MATH-500 vs 1.5B × GSM8K")
    logger.info("=" * 70)
    logger.info("| Metric                    | MATH-500  | GSM8K     |")
    logger.info("|---------------------------|-----------|-----------|")
    logger.info("| Greedy accuracy           | 11.4%%     | %.1f%%     |",
                100 * n_greedy_correct_test / n_test)
    logger.info("| Topo AUROC                | 0.935     | %.3f     |",
                a2_results["topo_confidence"]["fresh_pca"]["test_auroc"])
    logger.info("| Best logprob AUROC        | 0.473     | %.3f     |",
                results["comparison_table"]["best_logprob_auroc"] or 0)
    logger.info("| Ungated MV net gain       | +8        | %+d       |",
                strategy_results.get("majority_vote", {}).get("net_gain", 0))
    logger.info("| Weighted vote net gain    | +14       | %+d       |",
                strategy_results.get("confidence_weighted_vote", {}).get("net_gain", 0))
    logger.info("| R→W (weighted vote)       | 0         | %d        |",
                strategy_results.get("confidence_weighted_vote", {}).get("r_to_w", 0))
    logger.info("| Oracle ceiling            | +28       | +%d       |", oracle_ceiling)
    logger.info("=" * 70)
    logger.info("Elapsed: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
