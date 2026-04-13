#!/usr/bin/env python3
"""Phase 1: Steering Vector Extraction.

Extract "correct reasoning" steering vector from train-400 contrastive pairs
at the probe-selected layer.

Input: phase0/selected_layer.txt, phase0/activations/layer_{L}.npy,
       phase0/confidence_scores.npy, phase0/baseline_correct.npy
Output: phase1/steering_vector.npy, phase1/sampling_cosines.json,
        phase1/probe_auroc.txt, phase1/projection_confidence_correlation.json

Runtime: ~seconds (no GPU needed, pure numpy).
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from common import (
    PHASE0_DIR,
    PHASE1_DIR,
    SEED,
    get_train_holdout_indices,
    logger,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compute_steering_vector(
    activations_train: np.ndarray,
    correct_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Compute mean-diff steering vector from contrastive pairs.

    Args:
        activations_train: (400, 1536)
        correct_train: (400,) boolean/int
        seed: RNG seed for sampling incorrect problems

    Returns:
        Unit-normalized steering vector (1536,)
    """
    correct_mask = correct_train.astype(bool)
    n_correct = correct_mask.sum()

    H_correct = activations_train[correct_mask]  # (~46, 1536)
    H_incorrect_all = activations_train[~correct_mask]  # (~354, 1536)

    # Sample matched count from incorrect
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(H_incorrect_all), size=n_correct, replace=False)
    H_incorrect = H_incorrect_all[sampled_idx]

    # Mean-diff
    mean_correct = H_correct.mean(axis=0)
    mean_incorrect = H_incorrect.mean(axis=0)
    v = mean_correct - mean_incorrect

    # Unit normalize
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def main():
    print("=" * 60)
    print("Phase 1: Steering Vector Extraction")
    print("=" * 60)

    # Load selected layer
    with open(PHASE0_DIR / "selected_layer.txt") as f:
        content = f.read().strip()
    if content.startswith("HALT"):
        print(f"\n  Phase 0.5 halted: {content}")
        print("  Cannot proceed with steering vector extraction.")
        return

    selected_layer = int(content)
    print(f"\n  Selected layer: {selected_layer}")

    # Load probe AUROC
    with open(PHASE0_DIR / "probe_aurocs.json") as f:
        probe_aurocs = json.load(f)
    probe_auroc = probe_aurocs[str(selected_layer)]
    print(f"  Probe AUROC: {probe_auroc:.4f}")

    # Load data
    train_idx, holdout_idx = get_train_holdout_indices()
    activations = np.load(
        PHASE0_DIR / "activations" / f"layer_{selected_layer}.npy"
    )  # (500, 1536)
    activations_train = activations[train_idx]  # (400, 1536)
    correct = np.load(PHASE0_DIR / "baseline_correct.npy")
    correct_train = correct[train_idx]
    confidence_scores = np.load(PHASE0_DIR / "confidence_scores.npy")
    confidence_train = confidence_scores[train_idx]

    n_correct_train = correct_train.sum()
    n_incorrect_train = len(correct_train) - n_correct_train
    print(f"\n  Train-400: {n_correct_train} correct, {n_incorrect_train} incorrect")

    # Step 1: Compute steering vectors across 5 seeds for stability check
    seeds = [SEED, 43, 44, 45, 46]
    vectors = {}
    for s in seeds:
        v = compute_steering_vector(activations_train, correct_train, seed=s)
        vectors[s] = v
        print(f"  Seed {s}: ||v|| = {np.linalg.norm(v):.6f}")

    # Step 2: Pairwise cosine similarity matrix
    cosine_matrix = {}
    all_cosines = []
    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            if j > i:
                cos = cosine_similarity(vectors[s1], vectors[s2])
                cosine_matrix[f"{s1}_vs_{s2}"] = round(cos, 6)
                all_cosines.append(cos)
                print(f"  cos(seed {s1}, seed {s2}) = {cos:.4f}")

    min_cosine = min(all_cosines)
    mean_cosine = float(np.mean(all_cosines))
    print(f"\n  Min pairwise cosine: {min_cosine:.4f}")
    print(f"  Mean pairwise cosine: {mean_cosine:.4f}")

    # Step 3: Select steering vector
    if min_cosine > 0.9:
        # All seeds agree — use seed 42
        steering_vector = vectors[SEED]
        vector_type = "mean_diff_seed42"
        print(f"  All cosines > 0.9 -> using seed {SEED} vector")
    else:
        # Average all 5 and re-normalize
        avg = np.mean([vectors[s] for s in seeds], axis=0)
        steering_vector = avg / (np.linalg.norm(avg) + 1e-12)
        vector_type = "averaged_5seeds"
        print(f"  Cosines below 0.9 -> averaging all 5 seeds")

    print(f"  Vector type: {vector_type}")
    print(f"  ||steering_vector|| = {np.linalg.norm(steering_vector):.6f}")

    # Step 4: Projection-confidence correlation
    projections = activations_train @ steering_vector  # (400,)
    rho, p_value = spearmanr(projections, confidence_train)
    print(f"\n  Spearman r(projection, confidence) = {rho:.4f} (p = {p_value:.2e})")

    if abs(rho) > 0.3:
        interpretation = "aligned (|r| > 0.3) — steering direction correlates with confidence"
    elif abs(rho) < 0.1:
        interpretation = "orthogonal (|r| < 0.1) — ideal for targeted steering"
    else:
        interpretation = "weakly related (0.1 < |r| < 0.3)"
    print(f"  Interpretation: {interpretation}")

    # Step 5: Save artifacts
    np.save(PHASE1_DIR / "steering_vector.npy", steering_vector)

    with open(PHASE1_DIR / "sampling_cosines.json", "w") as f:
        json.dump({
            "seeds": seeds,
            "pairwise_cosines": cosine_matrix,
            "min_cosine": round(min_cosine, 6),
            "mean_cosine": round(mean_cosine, 6),
            "selected_type": vector_type,
        }, f, indent=2)

    with open(PHASE1_DIR / "probe_auroc.txt", "w") as f:
        f.write(f"{probe_auroc:.6f}\n")

    with open(PHASE1_DIR / "projection_confidence_correlation.json", "w") as f:
        json.dump({
            "spearman_r": round(float(rho), 6),
            "p_value": float(p_value),
            "interpretation": interpretation,
            "n_train": len(projections),
            "n_correct_train": int(n_correct_train),
            "vector_type": vector_type,
        }, f, indent=2)

    # Summary
    print(f"\n" + "=" * 60)
    print("Phase 1 Summary")
    print("=" * 60)
    print(f"  Layer: {selected_layer}")
    print(f"  Probe AUROC: {probe_auroc:.4f}")
    print(f"  Vector type: {vector_type}")
    print(f"  Steering-confidence correlation: r={rho:.4f}")
    print(f"  Stability: min cos = {min_cosine:.4f}")
    print(f"  Artifacts saved to {PHASE1_DIR}/")
    print(f"\n  Next: Run phase2_calibration.py")


if __name__ == "__main__":
    main()
