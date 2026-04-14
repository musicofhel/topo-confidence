#!/usr/bin/env python3
"""Phase 1: Contrastive Set Expansion via Temperature Sampling.

1. Generate 32 completions per train-400 problem at T=0.8
2. Check correctness of each, tally expanded contrastive set
3. Extract prompt activations at layers {15, 20, 24, 25} for all correct problems
4. Extract layers 15, 25 for all 500 problems (Track A has 14,17,19,20,22,24)

Input: Track A phase0 artifacts (baseline_correct, baseline_answers, activations)
Output: pathway3/phase1/ — temperature_generations.json, expanded_activations/,
        additional_activations/, expansion_summary.json

Runtime: ~5-8 hours GPU (temperature sampling) + ~20 min (activation extraction).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    COMPLETIONS_PER_PROBLEM,
    N_PROBLEMS,
    P3_LAYERS,
    ADDITIONAL_LAYERS,
    PHASE1_DIR,
    SEED,
    TRACK_A_PHASE0,
    check_correct,
    extract_answer,
    extract_prompt_activations,
    generate_one_temperature,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    load_model,
    logger,
)

EXPANDED_ACT_DIR = PHASE1_DIR / "expanded_activations"
ADDITIONAL_ACT_DIR = PHASE1_DIR / "additional_activations"


def generate_temperature_completions(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    train_idx: np.ndarray,
    n_completions: int = COMPLETIONS_PER_PROBLEM,
    checkpoint_every: int = 50,
) -> dict:
    """Generate multiple temperature completions per train problem.

    Args:
        model: HF model
        tokenizer: Tokenizer
        prompts: All 500 prompts
        ground_truths: All 500 ground truths
        train_idx: Train problem indices
        n_completions: Completions per problem
        checkpoint_every: Save checkpoint every N problems

    Returns:
        generations dict: {problem_idx: {completions, correct, n_correct}}
    """
    checkpoint_path = PHASE1_DIR / "temperature_generations_checkpoint.json"
    generations = {}

    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            generations = json.load(f)
        logger.info("Resumed from checkpoint: %d problems done", len(generations))

    total = len(train_idx)
    t0 = time.time()

    for count, idx in enumerate(train_idx):
        idx_str = str(int(idx))
        if idx_str in generations:
            continue

        if count % 10 == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / max(elapsed, 1)
            eta = (total - count) / max(rate, 0.001)
            logger.info(
                "Problem %d/%d (idx=%d) | %.1f problems/s | ETA: %.0f min",
                count, total, idx, rate, eta / 60,
            )

        completions = []
        correct_flags = []

        for c in range(n_completions):
            text = generate_one_temperature(model, tokenizer, prompts[idx])
            predicted = extract_answer(text)
            is_correct = check_correct(predicted, ground_truths[idx])
            completions.append(text[:500])  # truncate for storage
            correct_flags.append(is_correct)

        generations[idx_str] = {
            "completions": completions,
            "correct": correct_flags,
            "n_correct": sum(correct_flags),
        }

        # Checkpoint
        if (count + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(generations, f)
            logger.info("  Checkpoint saved (%d problems)", count + 1)

    return generations


def tally_expanded_set(
    generations: dict,
    baseline_correct: np.ndarray,
    train_idx: np.ndarray,
) -> dict:
    """Tally the expanded contrastive set.

    Returns:
        Summary dict with counts and per-problem breakdown.
    """
    total_correct_solutions = 0
    problems_with_correct = set()
    new_problems_correct = []  # problems that were wrong under greedy but correct under sampling

    # Track which problems originally had correct greedy answers
    greedy_correct_problems = set(np.where(baseline_correct.astype(bool))[0])

    per_problem = {}
    for idx_str, data in generations.items():
        idx = int(idx_str)
        n_correct = data["n_correct"]
        n_total = len(data["correct"])

        if n_correct > 0:
            total_correct_solutions += n_correct
            problems_with_correct.add(idx)
            if idx not in greedy_correct_problems:
                new_problems_correct.append(idx)

        per_problem[idx_str] = {
            "n_correct": n_correct,
            "n_total": n_total,
            "pass_rate": round(n_correct / n_total, 4) if n_total > 0 else 0,
            "greedy_correct": idx in greedy_correct_problems,
        }

    n_greedy_correct_in_train = sum(
        1 for idx in train_idx if idx in greedy_correct_problems
    )

    summary = {
        "total_correct_solutions": total_correct_solutions,
        "n_problems_with_correct": len(problems_with_correct),
        "n_new_problems_correct": len(new_problems_correct),
        "n_greedy_correct_in_train": n_greedy_correct_in_train,
        "new_problem_indices": sorted(new_problems_correct),
        "completions_per_problem": COMPLETIONS_PER_PROBLEM,
        "temperature": 0.8,
        "per_problem": per_problem,
    }

    return summary


def extract_expanded_activations(
    model,
    tokenizer,
    prompts: list[str],
    generations: dict,
    layers: list[int] = P3_LAYERS,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Extract prompt activations for all problems that have correct completions.

    Since prompt activations don't depend on the completion, we only need
    one forward pass per unique prompt.

    Args:
        model: HF model
        tokenizer: Tokenizer
        prompts: All 500 prompts
        generations: Temperature generation results
        layers: Layers to extract

    Returns:
        layer_activations: {layer: (N_correct_problems, 1536)}
        problem_indices: (N_correct_problems,) which problems
    """
    # Collect unique problem indices that have at least one correct completion
    correct_problem_indices = sorted(
        int(idx) for idx, data in generations.items() if data["n_correct"] > 0
    )
    n_problems = len(correct_problem_indices)
    logger.info("Extracting activations for %d problems with correct completions", n_problems)

    hidden_dim = model.config.hidden_size  # 1536
    layer_activations = {layer: np.zeros((n_problems, hidden_dim), dtype=np.float32) for layer in layers}

    for i, idx in enumerate(correct_problem_indices):
        if i % 50 == 0:
            logger.info("  Activation extraction %d/%d (idx=%d)", i, n_problems, idx)

        acts = extract_prompt_activations(model, tokenizer, prompts[idx], layers)
        for layer in layers:
            layer_activations[layer][i] = acts[layer]

    return layer_activations, np.array(correct_problem_indices, dtype=int)


def extract_additional_layer_activations(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int] = ADDITIONAL_LAYERS,
) -> dict[int, np.ndarray]:
    """Extract activations at additional layers for all 500 problems.

    Track A already has layers {14, 17, 19, 20, 22, 24}.
    We need layers 15 and 25 for multi-layer experiments.

    Returns:
        {layer: (500, 1536)}
    """
    n = len(prompts)
    hidden_dim = model.config.hidden_size
    layer_activations = {layer: np.zeros((n, hidden_dim), dtype=np.float32) for layer in layers}

    for i in range(n):
        if i % 50 == 0:
            logger.info("  Additional layers %d/%d", i, n)

        acts = extract_prompt_activations(model, tokenizer, prompts[i], layers)
        for layer in layers:
            layer_activations[layer][i] = acts[layer]

    return layer_activations


def main():
    print("=" * 70)
    print("Phase 1: Contrastive Set Expansion via Temperature Sampling")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    assert len(prompts) == N_PROBLEMS

    train_idx, holdout_idx = get_train_holdout_indices()
    baseline_correct = np.load(TRACK_A_PHASE0 / "baseline_correct.npy")
    n_train_correct = baseline_correct[train_idx].sum()
    print(f"  Train-400: {n_train_correct} correct under greedy")

    # Load model
    print("\n[2] Loading model...")
    model, tokenizer = load_model()

    # ============================================
    # Step 1: Temperature sampling
    # ============================================
    print(f"\n[3] Generating {COMPLETIONS_PER_PROBLEM} completions per train problem at T=0.8...")
    t0 = time.time()
    generations = generate_temperature_completions(
        model, tokenizer, prompts, ground_truths, train_idx
    )
    gen_time = time.time() - t0
    print(f"  Temperature sampling: {gen_time:.0f}s ({gen_time / 3600:.1f}h)")

    # Save full generations
    with open(PHASE1_DIR / "temperature_generations.json", "w") as f:
        json.dump(generations, f)
    print(f"  Saved temperature_generations.json")

    # ============================================
    # Step 2: Tally expanded set
    # ============================================
    print("\n[4] Tallying expanded contrastive set...")
    summary = tally_expanded_set(generations, baseline_correct, train_idx)

    print(f"  Expanded contrastive set: {summary['total_correct_solutions']} correct "
          f"solutions across {summary['n_problems_with_correct']} problems "
          f"(was {n_train_correct} across {n_train_correct} problems)")
    print(f"  New problems with correct solutions: {summary['n_new_problems_correct']}")

    # Check if we hit the 150+ target
    if summary['n_problems_with_correct'] < 100:
        print(f"\n  *** WARNING: Only {summary['n_problems_with_correct']} problems have "
              f"correct completions (target: 150+). Consider T=1.0 or more completions. ***")
    elif summary['n_problems_with_correct'] < 150:
        print(f"\n  NOTE: {summary['n_problems_with_correct']} problems (target was 150+). "
              f"Proceeding but may be underpowered.")
    else:
        print(f"\n  TARGET MET: {summary['n_problems_with_correct']} >= 150")

    # ============================================
    # Step 3: Extract activations for correct problems
    # ============================================
    print(f"\n[5] Extracting activations at layers {P3_LAYERS} for correct problems...")
    t0 = time.time()
    layer_acts, correct_indices = extract_expanded_activations(
        model, tokenizer, prompts, generations
    )
    act_time = time.time() - t0
    print(f"  Extracted in {act_time:.0f}s")

    # Save
    EXPANDED_ACT_DIR.mkdir(parents=True, exist_ok=True)
    for layer, acts in layer_acts.items():
        path = EXPANDED_ACT_DIR / f"correct_activations_layer{layer}.npy"
        np.save(path, acts)
        print(f"  Saved {path.name}: {acts.shape}")

    np.save(EXPANDED_ACT_DIR / "correct_problem_indices.npy", correct_indices)

    # ============================================
    # Step 4: Extract additional layers for all 500
    # ============================================
    print(f"\n[6] Extracting layers {ADDITIONAL_LAYERS} for all 500 problems...")
    t0 = time.time()
    additional_acts = extract_additional_layer_activations(
        model, tokenizer, prompts
    )
    add_time = time.time() - t0
    print(f"  Extracted in {add_time:.0f}s")

    ADDITIONAL_ACT_DIR.mkdir(parents=True, exist_ok=True)
    for layer, acts in additional_acts.items():
        path = ADDITIONAL_ACT_DIR / f"layer_{layer}.npy"
        np.save(path, acts)
        print(f"  Saved {path.name}: {acts.shape}")

    # Save expansion summary
    summary["activation_extraction_time_s"] = round(act_time, 1)
    summary["additional_extraction_time_s"] = round(add_time, 1)
    summary["temperature_sampling_time_s"] = round(gen_time, 1)

    with open(PHASE1_DIR / "expansion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Cleanup checkpoint
    checkpoint_path = PHASE1_DIR / "temperature_generations_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Removed checkpoint file")

    # Summary
    print(f"\n" + "=" * 70)
    print("Phase 1 Summary")
    print("=" * 70)
    print(f"  Greedy correct (train): {n_train_correct}")
    print(f"  Problems with correct completions: {summary['n_problems_with_correct']}")
    print(f"  Total correct solutions: {summary['total_correct_solutions']}")
    print(f"  New problems (greedy-wrong, T-correct): {summary['n_new_problems_correct']}")
    print(f"  Activation layers: {P3_LAYERS}")
    print(f"  Artifacts saved to {PHASE1_DIR}/")
    print(f"\n  Next: Run phase2_prototypes.py")


if __name__ == "__main__":
    main()
