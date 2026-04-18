#!/usr/bin/env python3
"""Phase B1: Qwen2.5-7B MATH-500 Greedy Baseline + Trajectory Extraction.

GPU script for RunPod H100. Generates greedy answers for all 500 MATH-500
problems using Qwen2.5-7B-Instruct, extracts hidden-state trajectories
(28 layers × seq_len × 3584) and layer states, and computes logprob features.

Uses the SAME train/holdout split as 1.5B experiments (seed 9999).
The 7B model needs ~15 GB VRAM in bf16.

Runtime: ~2-3 H100-hours.
Output: pathway5/track_b/phase_b1/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    MODEL_NAME_7B,
    N_PROBLEMS_MATH,
    SEED,
    TRACK_B_DIR,
    check_correct,
    extract_answer,
    extract_trajectory,
    format_prompt,
    generate_greedy_with_logprobs,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    load_model,
    logger,
    normalize_answer,
)

PHASE_DIR = TRACK_B_DIR / "phase_b1"


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load MATH-500 ----
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    train_idx, holdout_idx = get_train_holdout_indices()  # Same split as 1.5B
    logger.info(
        "Loaded MATH-500: %d problems, train=%d, holdout=%d",
        len(problems), len(train_idx), len(holdout_idx),
    )

    # ---- Step 2: Load 7B model ----
    model, tokenizer = load_model(MODEL_NAME_7B)

    # ---- Step 3: Greedy generation + logprob extraction ----
    checkpoint_path = PHASE_DIR / "greedy_checkpoint.json"
    answers = {}
    entropy_scores = {}

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        answers = ckpt.get("answers", {})
        entropy_scores = ckpt.get("entropy_scores", {})
        logger.info("Resumed from checkpoint: %d problems done", len(answers))

    logger.info("=== Greedy generation with 7B ===")
    for i in range(N_PROBLEMS_MATH):
        i_str = str(i)
        if i_str in answers:
            continue

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (N_PROBLEMS_MATH - i) / max(rate, 0.001)
            logger.info(
                "Greedy %d/%d | %.1f problems/s | ETA: %.0f min",
                i, N_PROBLEMS_MATH, rate, eta / 60,
            )

        text, entropy_feats = generate_greedy_with_logprobs(model, tokenizer, prompts[i])
        predicted = extract_answer(text)
        is_correct = check_correct(predicted, ground_truths[i])

        answers[i_str] = {
            "text": text[:1000],
            "predicted": predicted,
            "ground_truth": ground_truths[i],
            "correct": is_correct,
        }
        entropy_scores[i_str] = entropy_feats

        if (i + 1) % 100 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"answers": answers, "entropy_scores": entropy_scores}, f)
            logger.info("  Checkpoint saved (%d)", i + 1)

    # Save
    with open(PHASE_DIR / "math500_7b_baseline_answers.json", "w") as f:
        json.dump(answers, f, indent=2)

    baseline_correct = np.array([answers[str(i)]["correct"] for i in range(N_PROBLEMS_MATH)])
    np.save(PHASE_DIR / "math500_7b_baseline_correct.npy", baseline_correct)
    n_correct = int(baseline_correct.sum())
    logger.info(
        "7B greedy baseline: %d/%d correct (%.1f%%)",
        n_correct, N_PROBLEMS_MATH, 100 * n_correct / N_PROBLEMS_MATH,
    )

    with open(PHASE_DIR / "math500_7b_entropy_scores.json", "w") as f:
        json.dump(entropy_scores, f, indent=2)

    # ---- Step 4: Trajectory extraction ----
    logger.info("=== Trajectory extraction (7B, hidden_dim=3584) ===")
    traj_checkpoint_path = PHASE_DIR / "trajectory_checkpoint.npz"
    trajectories = {}
    layer_states_dict = {}

    if traj_checkpoint_path.exists():
        ckpt_data = np.load(traj_checkpoint_path, allow_pickle=True)
        done_indices = set(int(k.split("_")[1]) for k in ckpt_data.files if k.startswith("traj_"))
        for k in ckpt_data.files:
            if k.startswith("traj_"):
                trajectories[k] = ckpt_data[k]
            elif k.startswith("ls_"):
                layer_states_dict[k] = ckpt_data[k]
        logger.info("Resumed trajectories: %d done", len(done_indices))
    else:
        done_indices = set()

    for i in range(N_PROBLEMS_MATH):
        if i in done_indices:
            continue

        if i % 50 == 0:
            elapsed = time.time() - t0
            logger.info("Trajectory %d/%d | elapsed: %.0f min", i, N_PROBLEMS_MATH, elapsed / 60)

        traj, ls = extract_trajectory(model, tokenizer, prompts[i])
        trajectories[f"traj_{i}"] = traj
        layer_states_dict[f"ls_{i}"] = ls

        if (i + 1) % 100 == 0:
            np.savez_compressed(traj_checkpoint_path, **trajectories, **layer_states_dict)
            logger.info("  Trajectory checkpoint saved (%d)", i + 1)

    # Save trajectories
    np.savez_compressed(PHASE_DIR / "math500_7b_trajectories.npz", **trajectories)
    logger.info("Saved %d trajectories", len(trajectories))

    # Save layer states
    layer_states_array = np.stack(
        [layer_states_dict[f"ls_{i}"] for i in range(N_PROBLEMS_MATH)], axis=0,
    )
    np.save(PHASE_DIR / "math500_7b_layer_states.npy", layer_states_array)
    logger.info("Saved layer states: %s", layer_states_array.shape)

    # Verify hidden dim
    hidden_dim = trajectories["traj_0"].shape[1]
    logger.info("Hidden dim: %d (expected 3584)", hidden_dim)
    assert hidden_dim == 3584, f"Unexpected hidden_dim: {hidden_dim}"

    # ---- Step 5: Summary ----
    elapsed = time.time() - t0

    # Per-split accuracy
    train_acc = baseline_correct[train_idx].mean()
    holdout_acc = baseline_correct[holdout_idx].mean()

    summary = {
        "model": MODEL_NAME_7B,
        "dataset": "MATH-500",
        "n_problems": N_PROBLEMS_MATH,
        "n_correct": n_correct,
        "accuracy": round(n_correct / N_PROBLEMS_MATH, 4),
        "train_accuracy": round(float(train_acc), 4),
        "holdout_accuracy": round(float(holdout_acc), 4),
        "hidden_dim": hidden_dim,
        "n_layers": layer_states_array.shape[1],
        "trajectory_shape_example": list(trajectories["traj_0"].shape),
        "layer_states_shape": list(layer_states_array.shape),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(PHASE_DIR / "phase_b1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase B1 complete in %.1f min", elapsed / 60)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
