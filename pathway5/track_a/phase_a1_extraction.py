#!/usr/bin/env python3
"""Phase A1: GSM8K Greedy Baseline + Trajectory Extraction + Logprob Features.

GPU script for RunPod H100. Generates greedy answers for all 1319 GSM8K test
problems, extracts hidden-state trajectories and layer states, and computes
output-probability features for baselines.

Runtime: ~1-2 H100-hours.
Output: pathway5/track_a/phase_a1/
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
    MODEL_NAME_1_5B,
    SEED,
    TRACK_A_DIR,
    check_correct,
    extract_answer,
    extract_gsm8k_ground_truth,
    extract_trajectory,
    generate_greedy_with_logprobs,
    get_gsm8k_prompts_and_ground_truths,
    load_gsm8k,
    load_model,
    logger,
    normalize_answer,
)

PHASE_DIR = TRACK_A_DIR / "phase_a1"


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load GSM8K ----
    problems = load_gsm8k()
    n_problems = len(problems)
    logger.info("Loaded %d GSM8K test problems", n_problems)

    prompts, ground_truths = get_gsm8k_prompts_and_ground_truths(problems)

    # ---- Step 2: Load model ----
    model, tokenizer = load_model(MODEL_NAME_1_5B)

    # ---- Step 3: Greedy generation + logprob extraction ----
    # Resume from checkpoint if exists
    checkpoint_path = PHASE_DIR / "greedy_checkpoint.json"
    answers = {}
    entropy_scores = {}

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        answers = ckpt.get("answers", {})
        entropy_scores = ckpt.get("entropy_scores", {})
        logger.info("Resumed from checkpoint: %d problems done", len(answers))

    logger.info("=== Greedy generation + logprob extraction ===")
    for i in range(n_problems):
        i_str = str(i)
        if i_str in answers:
            continue

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (n_problems - i) / max(rate, 0.001)
            logger.info(
                "Greedy %d/%d | %.1f problems/s | ETA: %.0f min",
                i, n_problems, rate, eta / 60,
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

        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"answers": answers, "entropy_scores": entropy_scores}, f)
            logger.info("  Checkpoint saved (%d)", i + 1)

    # Save greedy answers
    with open(PHASE_DIR / "gsm8k_baseline_answers.json", "w") as f:
        json.dump(answers, f, indent=2)

    # Compute correctness array
    baseline_correct = np.array([answers[str(i)]["correct"] for i in range(n_problems)])
    np.save(PHASE_DIR / "gsm8k_baseline_correct.npy", baseline_correct)
    n_correct = int(baseline_correct.sum())
    logger.info(
        "Greedy baseline: %d/%d correct (%.1f%%)",
        n_correct, n_problems, 100 * n_correct / n_problems,
    )

    # Save entropy scores
    with open(PHASE_DIR / "gsm8k_entropy_scores.json", "w") as f:
        json.dump(entropy_scores, f, indent=2)

    # ---- Step 4: Trajectory extraction for ALL problems ----
    logger.info("=== Trajectory extraction ===")
    traj_checkpoint_path = PHASE_DIR / "trajectory_checkpoint.npz"
    trajectories = {}
    layer_states_dict = {}

    # Check for trajectory checkpoint
    if traj_checkpoint_path.exists():
        ckpt_data = np.load(traj_checkpoint_path, allow_pickle=True)
        done_indices = set(int(k.split("_")[1]) for k in ckpt_data.files if k.startswith("traj_"))
        for k in ckpt_data.files:
            if k.startswith("traj_"):
                trajectories[k] = ckpt_data[k]
            elif k.startswith("ls_"):
                layer_states_dict[k] = ckpt_data[k]
        logger.info("Resumed trajectories from checkpoint: %d done", len(done_indices))
    else:
        done_indices = set()

    for i in range(n_problems):
        if i in done_indices:
            continue

        if i % 50 == 0:
            elapsed = time.time() - t0
            logger.info("Trajectory %d/%d | elapsed: %.0f min", i, n_problems, elapsed / 60)

        traj, ls = extract_trajectory(model, tokenizer, prompts[i])
        trajectories[f"traj_{i}"] = traj
        layer_states_dict[f"ls_{i}"] = ls

        # Checkpoint every 200
        if (i + 1) % 200 == 0:
            np.savez_compressed(traj_checkpoint_path, **trajectories, **layer_states_dict)
            logger.info("  Trajectory checkpoint saved (%d)", i + 1)

    # Save final trajectories
    np.savez_compressed(PHASE_DIR / "gsm8k_trajectories.npz", **trajectories)
    logger.info("Saved trajectories for %d problems", len(trajectories))

    # Save layer states separately (large: 1319 × 29 × 1536)
    layer_states_array = np.stack(
        [layer_states_dict[f"ls_{i}"] for i in range(n_problems)], axis=0,
    )
    np.save(PHASE_DIR / "gsm8k_layer_states.npy", layer_states_array)
    logger.info("Saved layer states: %s", layer_states_array.shape)

    # ---- Step 5: Summary ----
    elapsed = time.time() - t0
    summary = {
        "model": MODEL_NAME_1_5B,
        "dataset": "GSM8K",
        "n_problems": n_problems,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_problems, 4),
        "trajectory_shape_example": list(trajectories["traj_0"].shape),
        "layer_states_shape": list(layer_states_array.shape),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(PHASE_DIR / "phase_a1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase A1 complete in %.1f min", elapsed / 60)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
