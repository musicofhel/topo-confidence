#!/usr/bin/env python3
"""Phase B3: 7B MATH-500 Temperature Sampling + Per-Completion Trajectories.

GPU script for RunPod H100. Generates 32 temperature completions for each of the
100 holdout MATH-500 problems using Qwen2.5-7B-Instruct, then extracts
per-completion trajectories and layer states.

Only generates for holdout-100 (not full train-400) to save compute.

Runtime: ~4-6 H100-hours (3,200 generations + 3,200 forward passes).
Input: pathway5/track_b/phase_b1/
Output: pathway5/track_b/phase_b3/
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
    COMPLETIONS_PER_PROBLEM,
    MODEL_NAME_7B,
    SEED,
    TRACK_B_DIR,
    check_correct,
    extract_answer,
    extract_completion_trajectory,
    generate_temperature_completions,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    load_model,
    logger,
    normalize_answer,
)

PHASE_B1_DIR = TRACK_B_DIR / "phase_b1"
PHASE_DIR = TRACK_B_DIR / "phase_b3"


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load data ----
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    train_idx, holdout_idx = get_train_holdout_indices()
    baseline_correct = np.load(PHASE_B1_DIR / "math500_7b_baseline_correct.npy")
    logger.info(
        "Loaded MATH-500: %d problems, holdout=%d, 7B greedy acc=%.1f%%",
        len(problems), len(holdout_idx), 100 * baseline_correct.mean(),
    )

    # ---- Step 2: Load 7B model ----
    model, tokenizer = load_model(MODEL_NAME_7B)

    # ---- Step 3: Temperature sampling for holdout ----
    # Only sample for wrong-greedy holdout problems (correct ones are gated)
    wrong_holdout = holdout_idx[~baseline_correct[holdout_idx]]
    logger.info(
        "=== Temperature sampling for %d wrong-greedy holdout problems ===",
        len(wrong_holdout),
    )

    generations = generate_temperature_completions(
        model, tokenizer, prompts, ground_truths,
        problem_indices=wrong_holdout,
        output_dir=PHASE_DIR,
        n_completions=COMPLETIONS_PER_PROBLEM,
        checkpoint_every=20,
    )

    # Stats
    n_with_correct = sum(1 for g in generations.values() if g["n_correct"] > 0)
    total_correct = sum(g["n_correct"] for g in generations.values())
    logger.info(
        "Sampling: %d/%d problems have >=1 correct, %d total correct (%.1f%%)",
        n_with_correct, len(wrong_holdout),
        total_correct, 100 * total_correct / (len(wrong_holdout) * COMPLETIONS_PER_PROBLEM),
    )

    # ---- Step 4: Per-completion trajectory extraction ----
    logger.info("=== Per-completion trajectory extraction (7B) ===")
    completion_data = {}

    total_completions = len(wrong_holdout) * COMPLETIONS_PER_PROBLEM
    count = 0

    for idx in wrong_holdout:
        idx_str = str(int(idx))
        if idx_str not in generations:
            continue

        comps = generations[idx_str]["completions"]

        for c_idx, comp_text in enumerate(comps):
            traj, ls = extract_completion_trajectory(
                model, tokenizer, prompts[idx], comp_text,
            )
            completion_data[f"traj_{idx}_{c_idx}"] = traj
            completion_data[f"ls_{idx}_{c_idx}"] = ls
            count += 1

            if count % 200 == 0:
                elapsed = time.time() - t0
                rate = count / max(elapsed, 1)
                eta = (total_completions - count) / max(rate, 0.001)
                logger.info(
                    "Completion traj %d/%d | %.1f/s | ETA: %.0f min",
                    count, total_completions, rate, eta / 60,
                )
                # Checkpoint
                np.savez_compressed(
                    PHASE_DIR / "completion_trajectories_checkpoint.npz",
                    **completion_data,
                )

    # Final save
    np.savez_compressed(PHASE_DIR / "completion_trajectories.npz", **completion_data)
    logger.info("Saved %d completion trajectories", count)

    # ---- Step 5: Sampling-based baselines ----
    logger.info("=== Sampling-based baseline features ===")
    from collections import Counter
    sampling_baselines = {}

    for idx in wrong_holdout:
        idx_str = str(int(idx))
        if idx_str not in generations:
            continue

        comps = generations[idx_str]["completions"]
        answers = [normalize_answer(extract_answer(c)) for c in comps]
        counter = Counter(answers)
        top_count = counter.most_common(1)[0][1]
        second_count = counter.most_common(2)[1][1] if len(counter) > 1 else 0
        n_unique = len(counter)
        probs = np.array([counter[a] / len(answers) for a in counter])
        agree_entropy = -np.sum(probs * np.log(probs + 1e-12))

        sampling_baselines[idx_str] = {
            "p_majority": top_count / len(answers),
            "vote_margin": (top_count - second_count) / len(answers),
            "inv_answer_diversity": 1.0 / n_unique,
            "neg_agreement_entropy": -agree_entropy,
            "n_correct": generations[idx_str]["n_correct"],
        }

    with open(PHASE_DIR / "math500_7b_sampling_baselines.json", "w") as f:
        json.dump(sampling_baselines, f, indent=2)

    # ---- Step 6: Summary ----
    elapsed = time.time() - t0
    summary = {
        "model": MODEL_NAME_7B,
        "n_holdout": len(holdout_idx),
        "n_wrong_holdout": len(wrong_holdout),
        "n_correct_holdout_greedy": int(baseline_correct[holdout_idx].sum()),
        "completions_per_problem": COMPLETIONS_PER_PROBLEM,
        "total_completions": count,
        "n_with_correct_completion": n_with_correct,
        "total_correct_completions": total_correct,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(PHASE_DIR / "phase_b3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase B3 complete in %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
