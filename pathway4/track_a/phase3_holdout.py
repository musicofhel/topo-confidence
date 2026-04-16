#!/usr/bin/env python3
"""Phase A3: Holdout Evaluation.

Generate 32 temperature completions for holdout-100, extract trajectories,
compute topo features, score, and apply the locked selection strategy.

Input: Phase A2 locked strategy, Phase A1 features + scorer training data
Output: pathway4/track_a/phase3/ — holdout_temperature_generations.json,
        holdout_completion_scores.json, holdout_results.json,
        statistical_tests.json

Runtime: ~6 hours GPU (5h generation + 0.5h trajectory extraction +
         0.5h feature extraction).
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    BASELINE_CORRECT_PATH,
    COMPLETIONS_PER_PROBLEM,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    TEMP_GEN_PATH,
    bayesian_p_improvement,
    bootstrap_ci_net_gain,
    check_correct,
    extract_answer,
    extract_completion_trajectory,
    extract_features_with_prefitted_pca,
    fit_completion_scorer,
    fit_greedy_pca,
    generate_one_temperature,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_cached_trajectories,
    load_math500,
    load_model,
    logger,
    mcnemar_mid_p,
    score_completions,
    select_confidence_weighted_vote,
    select_majority_vote,
    select_max_confidence,
    select_random,
)

CHECKPOINT_EVERY_GEN = 10  # Save generation checkpoint every N problems
CHECKPOINT_EVERY_TRAJ = 20  # Save trajectory checkpoint every N problems


def generate_holdout_completions(
    model,
    tokenizer,
    holdout_idx: np.ndarray,
    prompts: list[str],
    ground_truths: list[str],
) -> dict:
    """Generate 32 temperature completions for each holdout problem.

    Returns:
        Dict keyed by global index: {completions: [32], correct: [32], n_correct: int}
    """
    gen_path = PHASE3_DIR / "holdout_temperature_generations.json"

    # Check for checkpoint
    existing = {}
    if gen_path.exists():
        with open(gen_path) as f:
            existing = json.load(f)
        logger.info("  Resuming from %d already-generated holdout problems", len(existing))

    n_total = len(holdout_idx)
    t0 = time.time()
    n_generated_this_run = 0

    for pi, gi in enumerate(holdout_idx):
        if str(gi) in existing:
            continue

        completions = []
        correct_flags = []
        gt = ground_truths[gi]
        prompt = prompts[gi]

        for ci in range(COMPLETIONS_PER_PROBLEM):
            comp = generate_one_temperature(
                model, tokenizer, prompt,
                temperature=0.8, top_p=0.95,
            )
            completions.append(comp)
            answer = extract_answer(comp)
            correct_flags.append(check_correct(answer, gt))

        existing[str(gi)] = {
            "completions": completions,
            "correct": correct_flags,
            "n_correct": sum(correct_flags),
        }
        n_generated_this_run += 1

        if n_generated_this_run % 5 == 0:
            elapsed = time.time() - t0
            rate = n_generated_this_run / elapsed
            remaining = (n_total - len(existing)) / rate if rate > 0 else 0
            logger.info(
                "  Generated %d/%d holdout problems (%.1f prob/min, ~%.0f min remaining)",
                len(existing), n_total, rate * 60, remaining / 60,
            )

        if n_generated_this_run % CHECKPOINT_EVERY_GEN == 0:
            with open(gen_path, "w") as f:
                json.dump(existing, f)
            logger.info("  Checkpoint saved: %d problems", len(existing))

    # Final save
    with open(gen_path, "w") as f:
        json.dump(existing, f)

    gen_time = time.time() - t0
    logger.info(
        "  Generation complete: %d problems in %.1fs",
        n_generated_this_run, gen_time,
    )

    return existing


def main() -> None:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Phase A3: Holdout Evaluation ===")

    # ---- Load prerequisites ----
    logger.info("Loading locked strategy...")
    with open(PHASE2_DIR / "locked_strategy.json") as f:
        locked = json.load(f)
    strategy_name = locked["selected_strategy"]
    logger.info("  Strategy: %s (CV mean gain=%.1f)", strategy_name, locked["mean_net_gain"])

    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()

    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    # ---- Step 1: Generate holdout temperature completions ----
    logger.info("Step 1: Generating holdout temperature completions...")
    model, tokenizer = load_model()

    holdout_gen = generate_holdout_completions(
        model, tokenizer, holdout_idx, prompts, ground_truths
    )

    # Count oracle ceiling for holdout
    holdout_oracle = 0
    for gi in holdout_idx:
        if not baseline_correct[gi]:
            data = holdout_gen.get(str(gi))
            if data and sum(data["correct"]) > 0:
                holdout_oracle += 1
    logger.info("  Holdout oracle ceiling: +%d", holdout_oracle)

    # ---- Step 2: Extract completion trajectories ----
    logger.info("Step 2: Extracting holdout completion trajectories...")
    t_extract = time.time()

    all_trajectories: list[np.ndarray] = []
    all_layer_states: list[np.ndarray] = []
    completion_map: list[tuple[int, int]] = []
    all_correct: list[bool] = []

    wrong_holdout = [gi for gi in holdout_idx if not baseline_correct[gi]]
    logger.info("  Wrong-greedy holdout problems: %d", len(wrong_holdout))

    # Check for trajectory extraction checkpoint
    traj_ckpt_path = PHASE3_DIR / "trajectory_checkpoint.npz"
    traj_meta_path = PHASE3_DIR / "trajectory_checkpoint_meta.json"
    start_pi = 0
    if traj_ckpt_path.exists() and traj_meta_path.exists():
        with open(traj_meta_path) as f:
            traj_meta = json.load(f)
        start_pi = traj_meta["n_problems_done"]
        ckpt_data = np.load(traj_ckpt_path, allow_pickle=True)
        all_trajectories = list(ckpt_data["trajectories"])
        all_layer_states = [ckpt_data["layer_states"][i] for i in range(len(ckpt_data["layer_states"]))]
        completion_map = [tuple(x) for x in ckpt_data["completion_map"].tolist()]
        all_correct = ckpt_data["correct"].tolist()
        logger.info("  Resuming trajectory extraction from problem %d/%d", start_pi, len(wrong_holdout))

    for pi, gi in enumerate(wrong_holdout):
        if pi < start_pi:
            continue

        data = holdout_gen[str(gi)]
        prompt = prompts[gi]

        for ci in range(COMPLETIONS_PER_PROBLEM):
            comp_text = data["completions"][ci]
            trajectory, layer_st = extract_completion_trajectory(
                model, tokenizer, prompt, comp_text
            )
            all_trajectories.append(trajectory)
            all_layer_states.append(layer_st)
            completion_map.append((int(gi), ci))
            all_correct.append(bool(data["correct"][ci]))

        if (pi + 1) % 10 == 0:
            elapsed = time.time() - t_extract
            logger.info(
                "  %d/%d holdout problems extracted (%.1fs elapsed)",
                pi + 1, len(wrong_holdout), elapsed,
            )

        if (pi + 1) % CHECKPOINT_EVERY_TRAJ == 0 and pi + 1 > start_pi:
            np.savez_compressed(
                traj_ckpt_path,
                trajectories=np.array(all_trajectories, dtype=object),
                layer_states=np.stack(all_layer_states),
                completion_map=np.array(completion_map),
                correct=np.array(all_correct),
            )
            with open(traj_meta_path, "w") as f:
                json.dump({"n_problems_done": pi + 1}, f)
            logger.info("  Trajectory checkpoint saved: %d problems", pi + 1)

    extract_time = time.time() - t_extract
    logger.info("  Extraction complete: %d completions in %.1fs", len(all_trajectories), extract_time)

    # Clean up checkpoint files
    if traj_ckpt_path.exists():
        traj_ckpt_path.unlink()
    if traj_meta_path.exists():
        traj_meta_path.unlink()

    # Free model VRAM
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Step 3: Compute topo features ----
    logger.info("Step 3: Computing topo features...")

    # Load greedy trajectories for PCA
    greedy_trajectories = load_cached_trajectories()
    greedy_pca = fit_greedy_pca(greedy_trajectories)
    del greedy_trajectories
    gc.collect()

    layer_states_array = np.stack(all_layer_states, axis=0)
    del all_layer_states
    gc.collect()

    t_features = time.time()
    holdout_features, feature_names = extract_features_with_prefitted_pca(
        all_trajectories, layer_states_array, greedy_pca
    )
    feature_time = time.time() - t_features
    logger.info("  Features computed: %s in %.1fs", holdout_features.shape, feature_time)

    # ---- Step 4: Fit scorer on ALL train-400 completions ----
    logger.info("Step 4: Fitting completion scorer on full train-400...")

    # Load train completion features from Phase A1
    train_data = np.load(PHASE1_DIR / "completion_features.npz")
    train_features = train_data["features"]
    train_correct = train_data["correct"]

    with open(PHASE1_DIR / "feature_names.json") as f:
        train_feature_names = json.load(f)

    scaler, clf, abc_indices = fit_completion_scorer(
        train_features, train_correct, train_feature_names
    )
    logger.info("  Scorer fitted on %d train completions", len(train_correct))

    # ---- Step 5: Score holdout completions and apply strategy ----
    logger.info("Step 5: Scoring and selecting...")

    holdout_correct_arr = np.array(all_correct, dtype=bool)
    holdout_scores = score_completions(scaler, clf, abc_indices, holdout_features)

    # Build per-problem score dict
    holdout_completion_scores: dict[str, dict] = {}
    idx = 0
    for gi in wrong_holdout:
        problem_scores = []
        problem_correct = []
        for ci in range(COMPLETIONS_PER_PROBLEM):
            problem_scores.append(float(holdout_scores[idx]))
            problem_correct.append(bool(all_correct[idx]))
            idx += 1
        holdout_completion_scores[str(gi)] = {
            "scores": problem_scores,
            "correct": problem_correct,
            "n_correct": sum(problem_correct),
        }

    # Save holdout completion scores
    with open(PHASE3_DIR / "holdout_completion_scores.json", "w") as f:
        json.dump(holdout_completion_scores, f, indent=2)

    # Apply selection strategy to all holdout problems
    rng = np.random.default_rng(9999)
    greedy_results = []
    selected_results = []

    for gi in holdout_idx:
        gc_flag = bool(baseline_correct[gi])
        greedy_results.append(gc_flag)

        if gc_flag:
            # Greedy correct — keep it
            selected_results.append(True)
            continue

        if str(gi) not in holdout_completion_scores:
            # Shouldn't happen for wrong-greedy holdout
            selected_results.append(False)
            continue

        cs_data = holdout_completion_scores[str(gi)]
        completions = holdout_gen[str(gi)]["completions"]
        gt = ground_truths[gi]
        scores_arr = np.array(cs_data["scores"])

        if strategy_name == "random":
            _, is_correct = select_random(scores_arr, completions, gt, rng)
        elif strategy_name == "max_confidence":
            _, is_correct = select_max_confidence(scores_arr, completions, gt)
        elif strategy_name == "majority_vote":
            _, is_correct = select_majority_vote(completions, gt)
        elif strategy_name == "confidence_weighted_vote":
            _, is_correct = select_confidence_weighted_vote(scores_arr, completions, gt)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        selected_results.append(is_correct)

    greedy_arr = np.array(greedy_results, dtype=bool)
    selected_arr = np.array(selected_results, dtype=bool)

    # ---- Step 6: Compute metrics ----
    w2r = int((~greedy_arr & selected_arr).sum())
    r2w = int((greedy_arr & ~selected_arr).sum())
    net_gain = w2r - r2w

    p_mcnemar = mcnemar_mid_p(r2w, w2r)
    p_improve = bayesian_p_improvement(w2r, r2w)
    ci_lo, ci_hi = bootstrap_ci_net_gain(
        greedy_arr.astype(int), selected_arr.astype(int)
    )

    holdout_results = {
        "strategy": strategy_name,
        "n_holdout": len(holdout_idx),
        "baseline_accuracy": round(float(greedy_arr.mean()), 4),
        "selected_accuracy": round(float(selected_arr.mean()), 4),
        "baseline_correct": int(greedy_arr.sum()),
        "selected_correct": int(selected_arr.sum()),
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net_gain": net_gain,
        "mcnemar_mid_p": round(p_mcnemar, 4),
        "bayesian_p_improvement": round(p_improve, 4),
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "oracle_ceiling": holdout_oracle,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save holdout results
    results_path = PHASE3_DIR / "holdout_results.json"
    with open(results_path, "w") as f:
        json.dump(holdout_results, f, indent=2)
    logger.info("  Saved holdout results: %s", results_path)

    # Statistical tests
    stat_tests = {
        "mcnemar_mid_p": round(p_mcnemar, 6),
        "bayesian_p_improvement": round(p_improve, 6),
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net_gain": net_gain,
    }
    with open(PHASE3_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # Comparison table
    comparison = {
        "greedy_baseline": {
            "accuracy": 0.11,
            "correct": 11,
            "net_gain": 0,
        },
        "track_a_steering": {
            "accuracy": 0.14,
            "correct": 14,
            "net_gain": 3,
            "wrong_to_right": 6,
            "right_to_wrong": 3,
        },
        "track_a_selection": {
            "accuracy": round(float(selected_arr.mean()), 4),
            "correct": int(selected_arr.sum()),
            "net_gain": net_gain,
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "strategy": strategy_name,
        },
    }
    with open(PHASE3_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ---- Go/No-Go ----
    if net_gain >= 5 and p_improve > 0.90:
        decision = "PUBLICATION_READY"
        rationale = (
            f"Selection achieves +{net_gain} holdout net gain with "
            f"P(improvement)={p_improve:.3f} > 0.90. Full P1→P4 arc is complete."
        )
    elif net_gain > 3:
        decision = "PROMISING"
        rationale = (
            f"Selection (+{net_gain}) improves over steering (+3) but "
            f"does not meet P(improvement) threshold. Consider Track B."
        )
    else:
        decision = "NO_IMPROVEMENT"
        rationale = (
            f"Selection (+{net_gain}) does not improve over steering (+3). "
            f"Proceed to Track B or accept routing-only paper."
        )

    go_no_go = {
        "decision": decision,
        "rationale": rationale,
        "holdout_net_gain": net_gain,
        "bayesian_p_improvement": round(p_improve, 4),
        "cv_mean_net_gain": locked["mean_net_gain"],
    }
    with open(PHASE3_DIR / "go_no_go_decision.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    # ---- Final report ----
    logger.info("\n=== Phase A3: Holdout Results ===")
    logger.info("Strategy: %s", strategy_name)
    logger.info("Baseline accuracy: %.1f%% (%d/100)", 100 * greedy_arr.mean(), greedy_arr.sum())
    logger.info("Selected accuracy: %.1f%% (%d/100)", 100 * selected_arr.mean(), selected_arr.sum())
    logger.info("Net gain: +%d (W→R=%d, R→W=%d)", net_gain, w2r, r2w)
    logger.info("Bayesian P(improvement): %.4f", p_improve)
    logger.info("Bootstrap 95%% CI: [%d, %d]", ci_lo, ci_hi)
    logger.info("Oracle ceiling: +%d", holdout_oracle)
    logger.info("Decision: %s", decision)
    logger.info("Total time: %.1f minutes", (time.time() - t_start) / 60)
    logger.info("Phase A3 complete.")


if __name__ == "__main__":
    main()
