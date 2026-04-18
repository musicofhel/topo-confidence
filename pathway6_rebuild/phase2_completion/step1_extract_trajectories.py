#!/usr/bin/env python3
"""Phase 2 Step 1: Re-extract per-completion trajectories with corrected labels.

GPU REQUIRED. Run on RunPod H100.

For each wrong-greedy problem (under corrected labels), forward-pass all 32
temperature completions through the model to extract:
  - Per-completion trajectory: (n_tokens, 1536) last-layer hidden states
  - Per-completion layer states: (29, 1536) at last token

Then compute topo features using train-400-only PCA (fixing the leakage).

Checkpoints every 50 problems. Resume from checkpoint on restart.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_V2_PATH,
    HOLDOUT_TEMP_GEN_V2_PATH,
    PATHWAY1_PHASE0_DIR,
    TEMP_GEN_V2_PATH,
    TRAJ_PATH,
    check_correct_v2,
    extract_completion_trajectory,
    extract_features_with_prefitted_pca,
    fit_greedy_pca_train_only,
    format_prompt,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_cached_trajectories,
    load_math500,
    load_model,
    logger,
)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_INTERVAL = 50


def extract_trajectories_for_split(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    temp_gen: dict,
    new_correct: np.ndarray,
    problem_indices: np.ndarray,
    checkpoint_prefix: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]], list[bool]]:
    """Extract per-completion trajectories for wrong-greedy problems in a split.

    Returns:
        all_trajectories: list of (n_tokens, hidden_dim) arrays
        all_layer_states: list of (n_layers, hidden_dim) arrays
        completion_map: list of (global_idx, completion_idx) tuples
        all_correct: list of bool correctness labels
    """
    all_trajectories = []
    all_layer_states = []
    completion_map = []
    all_correct = []

    # Find wrong-greedy problems
    wrong_global = [gi for gi in problem_indices if not new_correct[gi]]
    logger.info("  Wrong-greedy: %d problems × 32 completions", len(wrong_global))

    for batch_start in range(0, len(wrong_global), CHECKPOINT_INTERVAL):
        batch = wrong_global[batch_start : batch_start + CHECKPOINT_INTERVAL]
        batch_t0 = time.time()

        for gi in batch:
            key = str(gi)
            if key not in temp_gen:
                logger.warning("  Problem %d: no temperature generations, skipping", gi)
                continue

            data = temp_gen[key]
            completions = data["completions"]
            correct_labels = data["correct"]
            prompt = prompts[gi]
            gt = ground_truths[gi]

            for ci, completion in enumerate(completions):
                try:
                    traj, ls = extract_completion_trajectory(
                        model, tokenizer, prompt, completion
                    )
                    all_trajectories.append(traj)
                    all_layer_states.append(ls)
                    completion_map.append((gi, ci))
                    all_correct.append(bool(correct_labels[ci]))
                except Exception as e:
                    logger.error("  Problem %d, completion %d: %s", gi, ci, e)
                    # Use dummy data to maintain alignment
                    all_trajectories.append(np.zeros((1, 1536), dtype=np.float32))
                    all_layer_states.append(np.zeros((29, 1536), dtype=np.float32))
                    completion_map.append((gi, ci))
                    all_correct.append(False)

        # Progress log
        done = min(batch_start + CHECKPOINT_INTERVAL, len(wrong_global))
        logger.info(
            "  Progress: %d/%d problems done (batch took %.1fs)",
            done, len(wrong_global), time.time() - batch_t0,
        )

    return all_trajectories, all_layer_states, completion_map, all_correct


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 2 STEP 1: EXTRACT PER-COMPLETION TRAJECTORIES")
    logger.info("=" * 70)

    # Load data
    new_correct = np.load(BASELINE_CORRECT_V2_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()

    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    # Format prompts
    prompts = [format_prompt(p.get("problem", p.get("question", ""))) for p in problems]

    with open(TEMP_GEN_V2_PATH) as f:
        train_temp = json.load(f)
    with open(HOLDOUT_TEMP_GEN_V2_PATH) as f:
        holdout_temp = json.load(f)

    # Load model
    model, tokenizer = load_model()

    # ---- Extract train trajectories ----
    logger.info("\n--- Extracting train completion trajectories ---")
    train_trajs, train_ls, train_map, train_correct = extract_trajectories_for_split(
        model, tokenizer, prompts, ground_truths,
        train_temp, new_correct, train_idx,
        checkpoint_prefix="train",
    )
    logger.info("  Train: %d completions extracted", len(train_trajs))

    # ---- Extract holdout trajectories ----
    logger.info("\n--- Extracting holdout completion trajectories ---")
    hold_trajs, hold_ls, hold_map, hold_correct = extract_trajectories_for_split(
        model, tokenizer, prompts, ground_truths,
        holdout_temp, new_correct, holdout_idx,
        checkpoint_prefix="holdout",
    )
    logger.info("  Holdout: %d completions extracted", len(hold_trajs))

    # ---- Fit PCA on train-400 greedy trajectories only ----
    logger.info("\n--- Fitting PCA on train-400 greedy trajectories ---")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    greedy_trajectories = load_cached_trajectories()
    greedy_pca = fit_greedy_pca_train_only(greedy_trajectories, train_idx)
    del greedy_trajectories
    gc.collect()

    # ---- Extract topo features using corrected PCA ----
    logger.info("\n--- Computing topo features with corrected PCA ---")

    # Train features
    if train_trajs:
        train_ls_array = np.stack(train_ls, axis=0)  # (N, 29, 1536)
        train_features, feature_names = extract_features_with_prefitted_pca(
            train_trajs, train_ls_array, greedy_pca
        )
        logger.info("  Train features: %s", train_features.shape)

        np.savez_compressed(
            OUTPUT_DIR / "train_completion_features_v2.npz",
            features=train_features,
            correct=np.array(train_correct),
            completion_map=np.array(train_map),
        )

        with open(OUTPUT_DIR / "feature_names_v2.json", "w") as f:
            json.dump(feature_names, f)

    # Holdout features
    if hold_trajs:
        hold_ls_array = np.stack(hold_ls, axis=0)
        hold_features, _ = extract_features_with_prefitted_pca(
            hold_trajs, hold_ls_array, greedy_pca
        )
        logger.info("  Holdout features: %s", hold_features.shape)

        np.savez_compressed(
            OUTPUT_DIR / "holdout_completion_features_v2.npz",
            features=hold_features,
            correct=np.array(hold_correct),
            completion_map=np.array(hold_map),
        )

    # Save PCA
    import pickle
    with open(OUTPUT_DIR / "greedy_pca_train400.pkl", "wb") as f:
        pickle.dump(greedy_pca, f)

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 STEP 1 COMPLETE")
    logger.info("=" * 70)
    logger.info("Train completions: %d", len(train_trajs))
    logger.info("Holdout completions: %d", len(hold_trajs))
    logger.info("Elapsed: %.1f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
