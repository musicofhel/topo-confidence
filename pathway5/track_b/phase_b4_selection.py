#!/usr/bin/env python3
"""Phase B4: 7B MATH-500 Selection Strategy Evaluation.

CPU script. Computes per-completion topo features from Phase B3 trajectories,
evaluates all selection strategies on holdout-100, and builds the final
cross-model comparison table.

Runtime: ~30 min - 2 hours CPU.
Input: pathway5/track_b/phase_b1/, phase_b2/, phase_b3/
Output: pathway5/track_b/phase_b4/
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
    N_PROBLEMS_MATH,
    SEED,
    TRACK_B_DIR,
    bootstrap_auroc,
    check_correct,
    evaluate_gating_strategies,
    extract_answer,
    extract_topo_features,
    fit_pca_on_trajectories,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    normalize_answer,
    select_confidence_weighted_vote,
    select_majority_vote,
    select_max_confidence,
    select_random,
    logger,
)

PHASE_B1_DIR = TRACK_B_DIR / "phase_b1"
PHASE_B2_DIR = TRACK_B_DIR / "phase_b2"
PHASE_B3_DIR = TRACK_B_DIR / "phase_b3"
PHASE_DIR = TRACK_B_DIR / "phase_b4"


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load artifacts ----
    logger.info("=== Loading artifacts ===")

    baseline_correct = np.load(PHASE_B1_DIR / "math500_7b_baseline_correct.npy")
    train_idx, holdout_idx = get_train_holdout_indices()
    topo_scores = np.load(PHASE_B2_DIR / "math500_7b_topo_scores.npy")

    with open(PHASE_B3_DIR / "temperature_generations.json") as f:
        generations = json.load(f)

    with open(PHASE_B2_DIR / "math500_7b_feature_names.json") as f:
        feature_names = json.load(f)

    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    n_holdout = len(holdout_idx)
    n_greedy_correct = int(baseline_correct[holdout_idx].sum())
    n_wrong_holdout = n_holdout - n_greedy_correct
    logger.info(
        "Holdout: %d problems, %d greedy correct, %d wrong",
        n_holdout, n_greedy_correct, n_wrong_holdout,
    )

    # ---- Step 2: Per-completion topo feature extraction ----
    logger.info("=== Per-completion topo feature extraction (7B) ===")

    # Fit PCA on greedy train trajectories
    greedy_traj_data = np.load(PHASE_B1_DIR / "math500_7b_trajectories.npz")
    greedy_trajectories_train = [greedy_traj_data[f"traj_{i}"] for i in train_idx]
    pca = fit_pca_on_trajectories(greedy_trajectories_train)

    # Load completion trajectories
    comp_data = np.load(PHASE_B3_DIR / "completion_trajectories.npz")

    all_comp_trajectories = []
    all_comp_layer_states = []
    all_comp_correct = []
    all_comp_problem_ids = []

    wrong_holdout = holdout_idx[~baseline_correct[holdout_idx]]

    for idx in wrong_holdout:
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
    logger.info("Loaded %d completion trajectories for %d problems",
                n_completions, len(set(all_comp_problem_ids)))

    if n_completions == 0:
        logger.error("No completion trajectories. Cannot proceed.")
        return

    comp_layer_states = np.stack(all_comp_layer_states, axis=0)
    comp_correct = np.array(all_comp_correct)
    comp_problem_ids = np.array(all_comp_problem_ids)

    # Extract features
    logger.info("Extracting topo features for %d completions...", n_completions)
    t_feat = time.time()
    comp_features, comp_feature_names = extract_topo_features(
        all_comp_trajectories, comp_layer_states, prefitted_pca=pca,
    )
    logger.info("Features: %s, took %.1f min", comp_features.shape, (time.time() - t_feat) / 60)

    np.save(PHASE_DIR / "completion_features.npy", comp_features)
    np.save(PHASE_DIR / "completion_correct.npy", comp_correct)
    np.save(PHASE_DIR / "completion_problem_ids.npy", comp_problem_ids)

    # ---- Step 3: Per-completion scoring (5-fold problem-level CV) ----
    logger.info("=== Per-completion scoring ===")
    abc_indices = get_abc_column_indices(comp_feature_names)
    unique_problems = np.unique(comp_problem_ids)

    # 5-fold CV
    problem_labels = np.array([
        int(comp_correct[comp_problem_ids == p].any()) for p in unique_problems
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.zeros(n_completions)
    for fold_idx, (p_train, p_test) in enumerate(skf.split(unique_problems, problem_labels)):
        fold_train_probs = set(unique_problems[p_train])
        fold_test_probs = set(unique_problems[p_test])

        train_mask = np.array([pid in fold_train_probs for pid in comp_problem_ids])
        test_mask = np.array([pid in fold_test_probs for pid in comp_problem_ids])

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

    # Global per-completion AUROC
    global_auroc, global_ci_lo, global_ci_hi = bootstrap_auroc(comp_correct, cv_scores)
    logger.info("Global per-completion AUROC (CV): %.3f [%.3f-%.3f]",
                global_auroc, global_ci_lo, global_ci_hi)

    # Within-problem AUROC
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
    logger.info("Within-problem AUROC: mean=%.3f, median=%.3f, N=%d",
                within_mean, within_median, len(within_aurocs))

    # ---- Step 4: Selection strategies ----
    logger.info("=== Selection strategies on holdout ===")

    # Fit final scorer on all completions
    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(comp_features[:, abc_indices])
    clf_all = LogisticRegression(**LR_PARAMS)
    clf_all.fit(X_all, comp_correct.astype(int))
    final_scores = clf_all.predict_proba(X_all)[:, 1]

    # Map scores back to problems
    scores_by_problem = {}
    completions_by_problem = {}
    for idx_str, gen in generations.items():
        completions_by_problem[idx_str] = gen["completions"]

    for i, (pid, score) in enumerate(zip(comp_problem_ids, final_scores)):
        pid_str = str(pid)
        if pid_str not in scores_by_problem:
            scores_by_problem[pid_str] = []
        scores_by_problem[pid_str].append(float(score))

    # Evaluate strategies
    rng = np.random.default_rng(SEED)
    strategy_results = {}

    for strategy_name in ["random", "max_confidence", "majority_vote", "confidence_weighted_vote"]:
        correct = n_greedy_correct  # greedy-correct always counted
        w_to_r = 0
        r_to_w = 0

        for idx in holdout_idx:
            idx_str = str(int(idx))
            if baseline_correct[idx]:
                continue
            if idx_str not in completions_by_problem:
                continue

            comps = completions_by_problem[idx_str]
            gt = ground_truths[idx]

            if strategy_name == "random":
                n_trials = 1000
                n_ok = sum(
                    1 for _ in range(n_trials)
                    if select_random(np.zeros(len(comps)), comps, gt, rng)[1]
                )
                correct += n_ok / n_trials
                continue

            scores = np.array(scores_by_problem.get(idx_str, [0.0] * len(comps)))
            if strategy_name == "max_confidence":
                _, is_correct = select_max_confidence(scores, comps, gt)
            elif strategy_name == "majority_vote":
                _, is_correct = select_majority_vote(comps, gt)
            elif strategy_name == "confidence_weighted_vote":
                _, is_correct = select_confidence_weighted_vote(scores, comps, gt)
            else:
                continue

            if is_correct:
                correct += 1
                w_to_r += 1

        strategy_results[strategy_name] = {
            "correct": correct if strategy_name != "random" else round(correct, 1),
            "net_gain": (correct - n_greedy_correct) if strategy_name != "random"
                else round(correct - n_greedy_correct, 1),
            "w_to_r": w_to_r,
            "r_to_w": r_to_w,
        }
        logger.info("  %s: correct=%s, net=%s", strategy_name,
                     strategy_results[strategy_name]["correct"],
                     strategy_results[strategy_name]["net_gain"])

    # ---- Step 5: Gating strategies ----
    logger.info("=== Gating strategies ===")
    gating_results = evaluate_gating_strategies(
        topo_scores, completions_by_problem,
        ground_truths, holdout_idx, baseline_correct,
    )
    for g in gating_results:
        logger.info("  tau=%.1f: correct=%d, net=%+d, savings=%.1f%%",
                     g["tau"], g["correct"], g["net_gain"], g["savings_pct"])

    # ---- Step 6: Oracle ceiling ----
    oracle_correct = n_greedy_correct
    for idx in holdout_idx:
        idx_str = str(int(idx))
        if baseline_correct[idx]:
            continue
        if idx_str in generations and generations[idx_str]["n_correct"] > 0:
            oracle_correct += 1
    oracle_ceiling = oracle_correct - n_greedy_correct

    # ---- Step 7: Final results ----
    elapsed = time.time() - t0

    with open(PHASE_B2_DIR / "phase_b2_results.json") as f:
        b2_results = json.load(f)

    results = {
        "dataset": "MATH-500",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "n_holdout": n_holdout,
        "n_greedy_correct": n_greedy_correct,
        "greedy_accuracy_holdout": round(n_greedy_correct / n_holdout, 4),
        "oracle_ceiling": oracle_ceiling,
        "oracle_correct": oracle_correct,
        "per_completion": {
            "global_auroc": global_auroc,
            "global_ci": [global_ci_lo, global_ci_hi],
            "within_problem_auroc_mean": round(within_mean, 3),
            "within_problem_auroc_median": round(within_median, 3),
            "n_mixed_problems": len(within_aurocs),
        },
        "selection_strategies": strategy_results,
        "gating_strategies": gating_results,
        "topo_auroc_prompt_level": b2_results["topo_confidence"]["holdout_auroc"],
        "elapsed_minutes": round(elapsed / 60, 1),
        "comparison_table": {
            "greedy_acc": round(n_greedy_correct / n_holdout, 3),
            "topo_auroc": b2_results["topo_confidence"]["holdout_auroc"],
            "best_logprob_auroc": (
                b2_results["baselines"][b2_results["best_baseline"]]["auroc"]
                if b2_results.get("best_baseline") else None
            ),
            "mv_net_gain": strategy_results.get("majority_vote", {}).get("net_gain"),
            "gated_mv_net_gain": max(
                (g["net_gain"] for g in gating_results), default=0,
            ),
            "weighted_vote_net_gain": strategy_results.get("confidence_weighted_vote", {}).get("net_gain"),
            "weighted_vote_r_to_w": strategy_results.get("confidence_weighted_vote", {}).get("r_to_w"),
            "oracle_ceiling": oracle_ceiling,
        },
    }

    with open(PHASE_DIR / "phase_b4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Print cross-model comparison ----
    logger.info("\n" + "=" * 70)
    logger.info("TRACK B FINAL: 1.5B vs 7B on MATH-500 holdout")
    logger.info("=" * 70)
    logger.info("| Metric                    | 1.5B      | 7B        |")
    logger.info("|---------------------------|-----------|-----------|")
    logger.info("| Greedy accuracy           | 11.4%%     | %.1f%%     |",
                100 * n_greedy_correct / n_holdout)
    logger.info("| Topo AUROC (holdout)      | 0.935     | %.3f     |",
                b2_results["topo_confidence"]["holdout_auroc"])
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
