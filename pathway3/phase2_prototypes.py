#!/usr/bin/env python3
"""Phase 2: Multi-Prototype Vector Extraction & Denoising.

1. Construct expanded contrastive dataset from Phase 1
2. Extract denoised single vectors (bootstrap + PCA + sparsify) at layers {15, 20, 25}
3. Extract multi-prototype vectors via K-means clustering
4. Configure multi-layer injection setups

Input: pathway3/phase1/ artifacts, Track A phase0 activations
Output: pathway3/phase2/ — denoised_vectors/, prototypes/

Runtime: ~1 hour CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import (
    ADDITIONAL_LAYERS,
    BOOTSTRAP_SAMPLES,
    MULTI_LAYER_CONFIGS,
    N_PROBLEMS,
    P3_LAYERS,
    PCA_K_SWEEP,
    PHASE1_DIR,
    PHASE2_DIR,
    PROTOTYPE_K_SWEEP,
    PROTOTYPE_TEMP_SWEEP,
    SEED,
    SPARSE_D_SWEEP,
    TRACK_A_PHASE0,
    bootstrap_mean_diff_vector,
    cluster_differences,
    get_train_holdout_indices,
    logger,
    pca_project_vector,
    sparsify_vector,
)

EXPANDED_ACT_DIR = PHASE1_DIR / "expanded_activations"
ADDITIONAL_ACT_DIR = PHASE1_DIR / "additional_activations"
DENOISED_DIR = PHASE2_DIR / "denoised_vectors"
PROTOTYPE_DIR = PHASE2_DIR / "prototypes"

# Working layers for vector extraction (not 24 — that's Track A baseline, handled separately)
EXTRACTION_LAYERS = [15, 20, 25]


def load_all_activations(
    train_idx: np.ndarray,
) -> dict[int, np.ndarray]:
    """Load activations at all needed layers for train-400.

    Combines Track A activations (layers 14,17,19,20,22,24) with
    Pathway 3 additional activations (layers 15, 25).

    Returns:
        {layer: (400, 1536)} for layers in EXTRACTION_LAYERS
    """
    activations = {}

    for layer in EXTRACTION_LAYERS:
        if layer in ADDITIONAL_LAYERS:
            # From Pathway 3 Phase 1
            path = ADDITIONAL_ACT_DIR / f"layer_{layer}.npy"
            all_acts = np.load(path)  # (500, 1536)
            activations[layer] = all_acts[train_idx]
        else:
            # From Track A Phase 0
            path = TRACK_A_PHASE0 / "activations" / f"layer_{layer}.npy"
            all_acts = np.load(path)  # (500, 1536)
            activations[layer] = all_acts[train_idx]

    return activations


def load_expanded_correct_activations(
    train_idx: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Load expanded correct activations from Phase 1.

    Returns only the activations for problems in the train set that have
    at least one correct completion.

    Returns:
        layer_acts: {layer: (N_correct_in_train, 1536)}
        correct_train_indices: which train positions have correct completions
    """
    correct_indices = np.load(EXPANDED_ACT_DIR / "correct_problem_indices.npy")
    train_set = set(train_idx.tolist())

    # Filter to train-only problems
    mask = np.array([idx in train_set for idx in correct_indices])
    train_correct_indices = correct_indices[mask]

    layer_acts = {}
    for layer in EXTRACTION_LAYERS:
        path = EXPANDED_ACT_DIR / f"correct_activations_layer{layer}.npy"
        if path.exists():
            all_acts = np.load(path)  # (N_correct_problems, 1536)
            layer_acts[layer] = all_acts[mask]
        else:
            # Fall back to full-500 activations
            if layer in ADDITIONAL_LAYERS:
                full_path = ADDITIONAL_ACT_DIR / f"layer_{layer}.npy"
            else:
                full_path = TRACK_A_PHASE0 / "activations" / f"layer_{layer}.npy"
            full_acts = np.load(full_path)
            layer_acts[layer] = full_acts[train_correct_indices]

    return layer_acts, train_correct_indices


def construct_contrastive_set(
    train_activations: dict[int, np.ndarray],
    expanded_correct_acts: dict[int, np.ndarray],
    baseline_correct: np.ndarray,
    train_idx: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Construct expanded contrastive dataset per layer.

    Positive set: activations of problems with correct completions (from Phase 1).
    Negative set: activations of all incorrect train problems under greedy.

    Returns:
        {layer: (H_correct, H_incorrect_mean, diff_matrix)}
        where diff_matrix[i] = H_correct[i] - H_incorrect_mean
    """
    result = {}
    incorrect_mask = ~baseline_correct[train_idx].astype(bool)

    for layer in EXTRACTION_LAYERS:
        H_correct = expanded_correct_acts[layer]  # (N_correct, 1536)
        H_incorrect = train_activations[layer][incorrect_mask]  # (~354, 1536)
        H_incorrect_mean = H_incorrect.mean(axis=0)  # (1536,)

        # Per-sample differences
        diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]  # (N_correct, 1536)

        result[layer] = (H_correct, H_incorrect_mean, diff_matrix)
        logger.info(
            "  Layer %d: %d correct, %d incorrect, diff_matrix %s",
            layer, len(H_correct), len(H_incorrect), diff_matrix.shape,
        )

    return result


def extract_denoised_vectors(
    contrastive: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    """Extract denoised single vectors at each layer.

    For each layer:
    (a) Bootstrap averaging (B=20)
    (b) PCA projection (sweep k)
    (c) Top-d sparsification (sweep d)

    Returns:
        Dict of {layer: {bootstrap, pca_k{k}, sparse_d{d}: vector}}
    """
    all_vectors = {}

    for layer, (H_correct, H_incorrect_mean, diff_matrix) in contrastive.items():
        logger.info("Denoising layer %d...", layer)
        layer_vectors = {}

        # (a) Bootstrap averaging
        v_bootstrap = bootstrap_mean_diff_vector(
            H_correct, H_incorrect_mean, n_bootstrap=BOOTSTRAP_SAMPLES, seed=SEED
        )
        layer_vectors["bootstrap"] = v_bootstrap
        np.save(DENOISED_DIR / f"layer{layer}_bootstrap.npy", v_bootstrap)

        # (b) PCA projection
        for k in PCA_K_SWEEP:
            if k > min(diff_matrix.shape):
                logger.info("  Skipping PCA k=%d (matrix too small: %s)", k, diff_matrix.shape)
                continue
            v_pca = pca_project_vector(v_bootstrap, diff_matrix, k)
            layer_vectors[f"pca_k{k}"] = v_pca
            np.save(DENOISED_DIR / f"layer{layer}_pca_k{k}.npy", v_pca)

        # (c) Top-d sparsification (applied after PCA)
        # Use the best PCA k (we'll select via CV later; for now save all)
        for k in PCA_K_SWEEP:
            key = f"pca_k{k}"
            if key not in layer_vectors:
                continue
            v_pca = layer_vectors[key]
            for d in SPARSE_D_SWEEP:
                v_sparse = sparsify_vector(v_pca, d)
                layer_vectors[f"pca_k{k}_sparse_d{d}"] = v_sparse
                np.save(DENOISED_DIR / f"layer{layer}_pca_k{k}_sparse_d{d}.npy", v_sparse)

        all_vectors[layer] = layer_vectors
        logger.info("  Layer %d: %d vector variants saved", layer, len(layer_vectors))

    return all_vectors


def extract_prototypes(
    contrastive: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    """Extract multi-prototype vectors via K-means clustering.

    For each layer, sweep K in {3, 5, 8}.

    Returns:
        {layer: {K: {centroids, labels, quality}}}
    """
    all_prototypes = {}

    for layer, (H_correct, H_incorrect_mean, diff_matrix) in contrastive.items():
        logger.info("Clustering layer %d...", layer)
        layer_protos = {}

        for K in PROTOTYPE_K_SWEEP:
            if len(diff_matrix) < K * 5:
                logger.info("  Skipping K=%d (need %d samples, have %d)", K, K * 5, len(diff_matrix))
                continue

            centroids, labels, quality = cluster_differences(diff_matrix, K, seed=SEED)

            layer_protos[K] = {
                "centroids": centroids,
                "labels": labels.tolist(),
                "quality": quality,
            }

            # Save
            np.save(PROTOTYPE_DIR / f"layer{layer}_K{K}_centroids.npy", centroids)
            with open(PROTOTYPE_DIR / f"layer{layer}_K{K}_cluster_sizes.json", "w") as f:
                json.dump(quality["cluster_sizes"], f)
            with open(PROTOTYPE_DIR / f"layer{layer}_K{K}_cosine_matrix.json", "w") as f:
                json.dump(quality["cosine_matrix"], f)

            logger.info(
                "  Layer %d, K=%d: silhouette=%.3f, sizes=%s, valid=%s",
                layer, K, quality["silhouette"], quality["cluster_sizes"], quality["valid"],
            )

        all_prototypes[layer] = layer_protos

    return all_prototypes


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def analyze_vectors(
    denoised_vectors: dict,
    prototypes: dict,
) -> dict:
    """Analyze relationship between denoised vectors and prototypes.

    Returns summary dict.
    """
    analysis = {}

    for layer in EXTRACTION_LAYERS:
        layer_analysis = {}

        # Compare bootstrap to Track A vector (if layer 24 is available)
        if layer in denoised_vectors and "bootstrap" in denoised_vectors[layer]:
            v_boot = denoised_vectors[layer]["bootstrap"]

            # Compare PCA-projected vs bootstrap
            for k in PCA_K_SWEEP:
                key = f"pca_k{k}"
                if key in denoised_vectors[layer]:
                    cos = cosine_similarity(v_boot, denoised_vectors[layer][key])
                    layer_analysis[f"cos_bootstrap_vs_pca_k{k}"] = round(cos, 4)

        # Prototype summary
        if layer in prototypes:
            for K, data in prototypes[layer].items():
                q = data["quality"]
                layer_analysis[f"proto_K{K}_silhouette"] = q["silhouette"]
                layer_analysis[f"proto_K{K}_valid"] = q["valid"]
                layer_analysis[f"proto_K{K}_sizes"] = q["cluster_sizes"]

        analysis[str(layer)] = layer_analysis

    return analysis


def main():
    print("=" * 70)
    print("Phase 2: Multi-Prototype Vector Extraction & Denoising")
    print("=" * 70)

    # Setup
    DENOISED_DIR.mkdir(parents=True, exist_ok=True)
    PROTOTYPE_DIR.mkdir(parents=True, exist_ok=True)

    train_idx, holdout_idx = get_train_holdout_indices()
    baseline_correct = np.load(TRACK_A_PHASE0 / "baseline_correct.npy")

    # Load expansion summary
    with open(PHASE1_DIR / "expansion_summary.json") as f:
        expansion = json.load(f)
    print(f"\n  Expanded set: {expansion['n_problems_with_correct']} problems with correct completions")

    # ============================================
    # Step 1: Load activations
    # ============================================
    print(f"\n[1] Loading activations at layers {EXTRACTION_LAYERS}...")
    t0 = time.time()
    train_activations = load_all_activations(train_idx)
    expanded_correct_acts, correct_train_indices = load_expanded_correct_activations(train_idx)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Correct problems in train: {len(correct_train_indices)}")

    # ============================================
    # Step 2: Construct contrastive set
    # ============================================
    print("\n[2] Constructing expanded contrastive dataset...")
    contrastive = construct_contrastive_set(
        train_activations, expanded_correct_acts, baseline_correct, train_idx
    )

    # ============================================
    # Step 3: Denoised single vectors
    # ============================================
    print(f"\n[3] Extracting denoised vectors (bootstrap={BOOTSTRAP_SAMPLES}, PCA k={PCA_K_SWEEP})...")
    t0 = time.time()
    denoised_vectors = extract_denoised_vectors(contrastive)
    print(f"  Denoised vectors: {time.time() - t0:.1f}s")

    # ============================================
    # Step 4: Multi-prototype clustering
    # ============================================
    print(f"\n[4] Extracting prototypes (K={PROTOTYPE_K_SWEEP})...")
    t0 = time.time()
    all_prototypes = extract_prototypes(contrastive)
    print(f"  Prototypes: {time.time() - t0:.1f}s")

    # ============================================
    # Step 5: Multi-layer configuration table
    # ============================================
    print("\n[5] Multi-layer configurations:")
    config_summary = {}
    for name, layers in MULTI_LAYER_CONFIGS.items():
        # Check all vectors exist
        available = all(
            layer in denoised_vectors or layer == 24
            for layer in layers
        )
        config_summary[name] = {
            "layers": layers,
            "available": available,
        }
        print(f"  {name}: layers={layers}, available={available}")

    # ============================================
    # Step 6: Analysis and summary
    # ============================================
    print("\n[6] Analyzing vectors...")
    analysis = analyze_vectors(denoised_vectors, all_prototypes)

    # Save summary
    phase2_summary = {
        "extraction_layers": EXTRACTION_LAYERS,
        "n_correct_train": len(correct_train_indices),
        "denoised_variants_per_layer": {
            str(layer): list(vecs.keys()) for layer, vecs in denoised_vectors.items()
        },
        "prototypes": {
            str(layer): {
                str(K): data["quality"] for K, data in protos.items()
            }
            for layer, protos in all_prototypes.items()
        },
        "multi_layer_configs": config_summary,
        "analysis": analysis,
    }

    with open(PHASE2_DIR / "phase2_summary.json", "w") as f:
        json.dump(phase2_summary, f, indent=2)

    # Compare Track A vector with expanded bootstrap at layer 24
    track_a_vec_path = TRACK_A_PHASE0.parent / "phase1" / "steering_vector.npy"
    if track_a_vec_path.exists() and 24 in EXTRACTION_LAYERS:
        track_a_vec = np.load(track_a_vec_path)
        # Load expanded bootstrap at layer 24 if it exists
        boot_24_path = DENOISED_DIR / "layer24_bootstrap.npy"
        if boot_24_path.exists():
            boot_24 = np.load(boot_24_path)
            cos = cosine_similarity(track_a_vec, boot_24)
            print(f"\n  cos(Track A vec, expanded bootstrap L24) = {cos:.4f}")
            phase2_summary["track_a_comparison"] = {"cosine_bootstrap_l24": round(cos, 4)}
            with open(PHASE2_DIR / "phase2_summary.json", "w") as f:
                json.dump(phase2_summary, f, indent=2)

    # Summary
    print(f"\n" + "=" * 70)
    print("Phase 2 Summary")
    print("=" * 70)
    print(f"  Layers: {EXTRACTION_LAYERS}")
    print(f"  Correct problems (train): {len(correct_train_indices)}")
    for layer in EXTRACTION_LAYERS:
        if layer in denoised_vectors:
            print(f"  Layer {layer}: {len(denoised_vectors[layer])} denoised variants")
        if layer in all_prototypes:
            for K, data in all_prototypes[layer].items():
                q = data["quality"]
                print(f"    K={K}: silhouette={q['silhouette']:.3f}, sizes={q['cluster_sizes']}, valid={q['valid']}")
    print(f"  Artifacts saved to {PHASE2_DIR}/")
    print(f"\n  Next: Run phase3_cv.py")


if __name__ == "__main__":
    main()
