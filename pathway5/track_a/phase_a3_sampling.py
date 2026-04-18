#!/usr/bin/env python3
"""Phase A3: GSM8K Temperature Sampling + Per-Completion Trajectory Extraction.

GPU script for RunPod H100. Generates 32 temperature completions for each
GSM8K test-split problem, then extracts per-completion trajectories and
layer states for topo-confidence scoring.

Runtime: ~2-4 H100-hours (8,448 generations + 8,448 forward passes for trajectories).
Input: pathway5/track_a/phase_a1/, phase_a2/
Output: pathway5/track_a/phase_a3/
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
    MODEL_NAME_1_5B,
    SEED,
    TRACK_A_DIR,
    check_correct,
    extract_answer,
    extract_completion_trajectory,
    generate_temperature_completions,
    get_gsm8k_prompts_and_ground_truths,
    load_gsm8k,
    load_model,
    logger,
    normalize_answer,
)

PHASE_A1_DIR = TRACK_A_DIR / "phase_a1"
PHASE_A2_DIR = TRACK_A_DIR / "phase_a2"
PHASE_DIR = TRACK_A_DIR / "phase_a3"


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load data ----
    problems = load_gsm8k()
    prompts, ground_truths = get_gsm8k_prompts_and_ground_truths(problems)
    baseline_correct = np.load(PHASE_A1_DIR / "gsm8k_baseline_correct.npy")
    test_idx = np.load(PHASE_A2_DIR / "gsm8k_test_idx.npy")
    logger.info("Loaded GSM8K: %d problems, %d test split", len(problems), len(test_idx))

    # ---- Step 2: Load model ----
    model, tokenizer = load_model(MODEL_NAME_1_5B)

    # ---- Step 3: Temperature sampling for test split ----
    logger.info("=== Temperature sampling for %d test problems ===", len(test_idx))
    generations = generate_temperature_completions(
        model, tokenizer, prompts, ground_truths,
        problem_indices=test_idx,
        output_dir=PHASE_DIR,
        n_completions=COMPLETIONS_PER_PROBLEM,
        checkpoint_every=50,
    )

    # Compute sampling stats
    n_with_correct = sum(
        1 for idx_str, g in generations.items() if g["n_correct"] > 0
    )
    total_correct = sum(g["n_correct"] for g in generations.values())
    logger.info(
        "Sampling: %d/%d problems have >=1 correct completion, %d/%d total correct (%.1f%%)",
        n_with_correct, len(test_idx),
        total_correct, len(test_idx) * COMPLETIONS_PER_PROBLEM,
        100 * total_correct / (len(test_idx) * COMPLETIONS_PER_PROBLEM),
    )

    # ---- Step 4: Per-completion trajectory extraction ----
    logger.info("=== Per-completion trajectory extraction ===")
    completion_traj_path = PHASE_DIR / "completion_trajectories_checkpoint.npz"
    completion_data = {}

    # Check for checkpoint
    done_completions = set()
    if completion_traj_path.exists():
        ckpt = np.load(completion_traj_path, allow_pickle=True)
        for k in ckpt.files:
            completion_data[k] = ckpt[k]
            if k.startswith("traj_"):
                done_completions.add(k)
        logger.info("Resumed completion trajectories: %d done", len(done_completions))

    total_completions = len(test_idx) * COMPLETIONS_PER_PROBLEM
    count = 0

    for idx in test_idx:
        idx_str = str(int(idx))
        if idx_str not in generations:
            continue

        comps = generations[idx_str]["completions"]

        for c_idx, comp_text in enumerate(comps):
            key = f"traj_{idx}_{c_idx}"
            if key in done_completions:
                count += 1
                continue

            traj, ls = extract_completion_trajectory(
                model, tokenizer, prompts[idx], comp_text,
            )
            completion_data[f"traj_{idx}_{c_idx}"] = traj
            completion_data[f"ls_{idx}_{c_idx}"] = ls
            count += 1

            if count % 500 == 0:
                elapsed = time.time() - t0
                rate = count / max(elapsed, 1)
                eta = (total_completions - count) / max(rate, 0.001)
                logger.info(
                    "Completion traj %d/%d | %.1f/s | ETA: %.0f min",
                    count, total_completions, rate, eta / 60,
                )
                # Checkpoint
                np.savez_compressed(completion_traj_path, **completion_data)
                logger.info("  Checkpoint saved")

    # Final save
    np.savez_compressed(PHASE_DIR / "completion_trajectories.npz", **completion_data)
    logger.info("Saved %d completion trajectories", count)

    # ---- Step 5: Sampling-based baselines ----
    logger.info("=== Computing sampling-based baseline features ===")
    sampling_baselines = {}

    for idx in test_idx:
        idx_str = str(int(idx))
        if idx_str not in generations:
            continue

        comps = generations[idx_str]["completions"]
        correct_flags = generations[idx_str]["correct"]

        # Extract answers
        answers = [normalize_answer(extract_answer(c)) for c in comps]
        from collections import Counter
        counter = Counter(answers)
        top_count = counter.most_common(1)[0][1]
        second_count = counter.most_common(2)[1][1] if len(counter) > 1 else 0
        n_unique = len(counter)

        # Agreement entropy
        probs = np.array([counter[a] / len(answers) for a in counter])
        agree_entropy = -np.sum(probs * np.log(probs + 1e-12))

        sampling_baselines[idx_str] = {
            "p_majority": top_count / len(answers),
            "vote_margin": (top_count - second_count) / len(answers),
            "inv_answer_diversity": 1.0 / n_unique,
            "neg_agreement_entropy": -agree_entropy,
            "n_correct": sum(correct_flags),
        }

    with open(PHASE_DIR / "gsm8k_sampling_baselines.json", "w") as f:
        json.dump(sampling_baselines, f, indent=2)

    # ---- Step 6: Summary ----
    elapsed = time.time() - t0
    summary = {
        "n_test_problems": len(test_idx),
        "completions_per_problem": COMPLETIONS_PER_PROBLEM,
        "total_completions": len(test_idx) * COMPLETIONS_PER_PROBLEM,
        "n_with_correct_completion": n_with_correct,
        "total_correct_completions": total_correct,
        "correct_rate": round(total_correct / (len(test_idx) * COMPLETIONS_PER_PROBLEM), 4),
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(PHASE_DIR / "phase_a3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase A3 complete in %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
