#!/usr/bin/env python3
"""Phase A1: Score Temperature Completions.

For each wrong-greedy train problem, extract full prompt+completion
trajectories for all 32 temperature completions, compute topo features
with pre-fitted PCA, and score with a completion-level confidence model.

Input: temperature_generations.json (Pathway 3), greedy trajectories,
       baseline_correct.npy
Output: pathway4/track_a/phase1/ — completion_scores.json,
        completion_features.npz, phase1_summary.json

Runtime: ~2 hours GPU (11,328 forward passes + ripser feature extraction).
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
    TEMP_GEN_PATH,
    check_correct,
    extract_answer,
    extract_completion_trajectory,
    extract_features_with_prefitted_pca,
    fit_completion_scorer,
    fit_greedy_pca,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_cached_trajectories,
    load_math500,
    load_model,
    logger,
    score_completions,
)

# ---- Checkpointing ----

CHECKPOINT_EVERY = 50  # Save trajectories every N problems
CHECKPOINT_PATH = PHASE1_DIR / "trajectories_checkpoint.json"


def save_trajectory_checkpoint(
    problem_trajectories: dict,
    problem_layer_states: dict,
    processed_indices: list[int],
) -> None:
    """Save checkpoint of extracted trajectories (indices only, not data)."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "processed_indices": processed_indices,
            "n_processed": len(processed_indices),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    logger.info("  Checkpoint saved: %d problems processed", len(processed_indices))


def main() -> None:
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- Step 1: Load prerequisites ----
    logger.info("=== Phase A1: Score Temperature Completions ===")

    logger.info("Loading temperature generations...")
    with open(TEMP_GEN_PATH) as f:
        temp_gen = json.load(f)
    logger.info("  Loaded %d problems × %d completions", len(temp_gen), COMPLETIONS_PER_PROBLEM)

    logger.info("Loading baseline correctness...")
    baseline_correct = np.load(BASELINE_CORRECT_PATH)

    train_idx, holdout_idx = get_train_holdout_indices()
    y_train = baseline_correct[train_idx].astype(bool)

    # Identify wrong-greedy train problems
    wrong_mask = ~y_train
    wrong_local_indices = np.where(wrong_mask)[0]  # Local indices (0-399)
    wrong_global_indices = train_idx[wrong_local_indices]  # Global indices (0-499)
    n_wrong = len(wrong_global_indices)
    logger.info("  Wrong-greedy train problems: %d / %d", n_wrong, len(train_idx))

    # Count oracle ceiling
    oracle_count = 0
    for gi in wrong_global_indices:
        data = temp_gen[str(gi)]
        if sum(data["correct"]) > 0:
            oracle_count += 1
    logger.info("  Oracle ceiling (problems with ≥1 correct completion): +%d", oracle_count)

    logger.info("Loading MATH-500 problems...")
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    # ---- Step 2: Pre-fit PCA from greedy trajectories ----
    logger.info("Loading greedy trajectories for PCA fitting...")
    greedy_trajectories = load_cached_trajectories()
    greedy_pca = fit_greedy_pca(greedy_trajectories)
    del greedy_trajectories  # Free memory
    gc.collect()

    # ---- Step 3: Extract completion trajectories ----
    logger.info("Loading model for trajectory extraction...")
    model, tokenizer = load_model()

    # Storage: collect all trajectories and layer_states for batch feature extraction
    all_trajectories: list[np.ndarray] = []  # (n_tokens, 1536) per completion
    all_layer_states: list[np.ndarray] = []  # (29, 1536) per completion
    # Track which problem/completion each entry corresponds to
    completion_map: list[tuple[int, int]] = []  # (global_problem_idx, completion_idx)
    # Also store correctness labels for the completion scorer
    all_correct: list[bool] = []

    # Check for existing checkpoint
    processed_globals = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        processed_globals = set(ckpt["processed_indices"])
        logger.info("  Resuming from checkpoint: %d problems already done", len(processed_globals))
        # Load saved trajectories
        traj_ckpt_path = PHASE1_DIR / "trajectories_partial.npz"
        ls_ckpt_path = PHASE1_DIR / "layer_states_partial.npz"
        map_ckpt_path = PHASE1_DIR / "completion_map_partial.json"
        correct_ckpt_path = PHASE1_DIR / "correct_partial.json"
        if traj_ckpt_path.exists():
            traj_data = np.load(traj_ckpt_path, allow_pickle=True)
            all_trajectories = [traj_data[k] for k in sorted(traj_data.files, key=lambda x: int(x.split("_")[1]))]
            ls_data = np.load(ls_ckpt_path)
            all_layer_states = [ls_data[f"ls_{i}"] for i in range(len(all_trajectories))]
            with open(map_ckpt_path) as f:
                completion_map = [tuple(x) for x in json.load(f)]
            with open(correct_ckpt_path) as f:
                all_correct = json.load(f)

    logger.info("Extracting completion trajectories (%d problems)...", n_wrong - len(processed_globals))
    t_extract = time.time()
    problems_done_this_run = 0

    for pi, gi in enumerate(wrong_global_indices):
        if gi in processed_globals:
            continue

        data = temp_gen[str(gi)]
        prompt = prompts[gi]
        gt = ground_truths[gi]

        for ci in range(COMPLETIONS_PER_PROBLEM):
            comp_text = data["completions"][ci]
            trajectory, layer_st = extract_completion_trajectory(
                model, tokenizer, prompt, comp_text
            )
            all_trajectories.append(trajectory)
            all_layer_states.append(layer_st)
            completion_map.append((int(gi), ci))
            all_correct.append(bool(data["correct"][ci]))

        processed_globals.add(int(gi))
        problems_done_this_run += 1

        if problems_done_this_run % 10 == 0:
            elapsed = time.time() - t_extract
            rate = problems_done_this_run / elapsed
            remaining = (n_wrong - len(processed_globals)) / rate if rate > 0 else 0
            logger.info(
                "  %d/%d problems done (%.1f prob/min, ~%.0f min remaining)",
                len(processed_globals), n_wrong, rate * 60, remaining / 60,
            )

        if problems_done_this_run % CHECKPOINT_EVERY == 0:
            # Save partial results
            traj_dict = {f"traj_{i}": t for i, t in enumerate(all_trajectories)}
            np.savez_compressed(PHASE1_DIR / "trajectories_partial.npz", **traj_dict)
            ls_dict = {f"ls_{i}": ls for i, ls in enumerate(all_layer_states)}
            np.savez_compressed(PHASE1_DIR / "layer_states_partial.npz", **ls_dict)
            with open(PHASE1_DIR / "completion_map_partial.json", "w") as f:
                json.dump(completion_map, f)
            with open(PHASE1_DIR / "correct_partial.json", "w") as f:
                json.dump(all_correct, f)
            save_trajectory_checkpoint({}, {}, sorted(processed_globals))

    extract_time = time.time() - t_extract
    logger.info(
        "Trajectory extraction complete: %d completions in %.1fs (%.2fs/completion)",
        len(all_trajectories), extract_time,
        extract_time / max(1, problems_done_this_run * COMPLETIONS_PER_PROBLEM),
    )

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Step 4: Compute topo features ----
    n_completions = len(all_trajectories)
    logger.info("Computing topo features for %d completions...", n_completions)

    # Stack layer_states: (N, 29, 1536)
    layer_states_array = np.stack(all_layer_states, axis=0)
    del all_layer_states
    gc.collect()

    t_features = time.time()
    features, feature_names = extract_features_with_prefitted_pca(
        all_trajectories, layer_states_array, greedy_pca
    )
    feature_time = time.time() - t_features
    logger.info(
        "Feature extraction complete: shape %s in %.1fs (%.2fs/completion)",
        features.shape, feature_time, feature_time / n_completions,
    )

    # ---- Step 5: Score with completion-level confidence model ----
    logger.info("Fitting completion scorer on all extracted features...")
    correct_array = np.array(all_correct, dtype=bool)
    logger.info(
        "  Label distribution: %d correct (%.1f%%), %d wrong",
        correct_array.sum(), 100 * correct_array.mean(), (~correct_array).sum(),
    )

    scaler, clf, abc_indices = fit_completion_scorer(features, correct_array, feature_names)

    # Score all completions
    scores = score_completions(scaler, clf, abc_indices, features)
    logger.info(
        "  Scores: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        scores.mean(), scores.std(), scores.min(), scores.max(),
    )

    # ---- Step 6: Organize and save results ----
    logger.info("Organizing results per problem...")

    # Build per-problem results
    completion_scores: dict[str, dict] = {}
    idx = 0
    for gi in wrong_global_indices:
        problem_scores = []
        problem_correct = []
        for ci in range(COMPLETIONS_PER_PROBLEM):
            # Find the entry in completion_map
            assert completion_map[idx] == (int(gi), ci), (
                f"Map mismatch at idx={idx}: expected ({gi}, {ci}), got {completion_map[idx]}"
            )
            problem_scores.append(float(scores[idx]))
            problem_correct.append(bool(all_correct[idx]))
            idx += 1

        completion_scores[str(gi)] = {
            "scores": problem_scores,
            "correct": problem_correct,
            "n_correct": sum(problem_correct),
            "max_score": max(problem_scores),
            "max_score_correct": problem_correct[int(np.argmax(problem_scores))],
        }

    # Save completion scores
    scores_path = PHASE1_DIR / "completion_scores.json"
    with open(scores_path, "w") as f:
        json.dump(completion_scores, f, indent=2)
    logger.info("  Saved completion scores: %s", scores_path)

    # Save features (compressed)
    features_path = PHASE1_DIR / "completion_features.npz"
    np.savez_compressed(
        features_path,
        features=features,
        correct=correct_array,
        completion_map=np.array(completion_map),
    )
    logger.info("  Saved features: %s (%.1f MB)", features_path, features_path.stat().st_size / 1e6)

    # Save feature names for CV
    with open(PHASE1_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ---- Summary statistics ----
    n_solvable = sum(1 for v in completion_scores.values() if v["n_correct"] > 0)
    n_max_correct = sum(1 for v in completion_scores.values() if v["max_score_correct"])
    n_problems_scored = len(completion_scores)

    summary = {
        "n_wrong_greedy": n_wrong,
        "n_completions_scored": n_completions,
        "n_solvable_problems": n_solvable,
        "oracle_ceiling": f"+{n_solvable}",
        "max_confidence_correct": n_max_correct,
        "max_confidence_accuracy": round(n_max_correct / n_problems_scored, 4)
            if n_problems_scored > 0 else 0,
        "feature_shape": list(features.shape),
        "n_features": len(feature_names),
        "n_abc_features": len(abc_indices),
        "label_balance": {
            "n_correct": int(correct_array.sum()),
            "n_wrong": int((~correct_array).sum()),
            "correct_rate": round(float(correct_array.mean()), 4),
        },
        "score_stats": {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        },
        "timing": {
            "extraction_seconds": round(extract_time, 1),
            "feature_seconds": round(feature_time, 1),
            "total_seconds": round(time.time() - t_start, 1),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = PHASE1_DIR / "phase1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  Saved summary: %s", summary_path)

    # Quick analysis
    logger.info("\n=== Phase A1 Summary ===")
    logger.info("Wrong-greedy problems: %d", n_wrong)
    logger.info("Oracle ceiling: +%d (%.1f%% of wrong problems solvable)", n_solvable, 100 * n_solvable / n_wrong)
    logger.info("Max-confidence selection correct: %d / %d (%.1f%%)",
                n_max_correct, n_problems_scored, 100 * n_max_correct / n_problems_scored)
    logger.info("Total time: %.1f minutes", (time.time() - t_start) / 60)

    # Clean up partial checkpoints
    for p in [
        PHASE1_DIR / "trajectories_partial.npz",
        PHASE1_DIR / "layer_states_partial.npz",
        PHASE1_DIR / "completion_map_partial.json",
        PHASE1_DIR / "correct_partial.json",
        CHECKPOINT_PATH,
    ]:
        if p.exists():
            p.unlink()
            logger.info("  Cleaned up: %s", p.name)

    logger.info("Phase A1 complete.")


if __name__ == "__main__":
    main()
