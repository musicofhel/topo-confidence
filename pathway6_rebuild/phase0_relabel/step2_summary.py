#!/usr/bin/env python3
"""Step 2: Compute and print foundational counts after relabeling.

Computes:
  - New baseline accuracy (total, train, holdout)
  - New oracle ceiling (train, holdout)
  - New "solvable under temperature" count
  - New correct rate among temperature completions
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_PATH,
    get_train_holdout_indices,
    logger,
)

OUTPUT_DIR = Path(__file__).parent


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 0 STEP 2: FOUNDATIONAL COUNTS SUMMARY")
    logger.info("=" * 70)

    # Load data
    old_correct = np.load(BASELINE_CORRECT_PATH)
    new_correct = np.load(OUTPUT_DIR / "baseline_correct_v2.npy")
    train_idx, holdout_idx = get_train_holdout_indices()

    with open(OUTPUT_DIR / "temperature_generations_v2.json") as f:
        train_temp = json.load(f)
    with open(OUTPUT_DIR / "holdout_temperature_generations_v2.json") as f:
        holdout_temp = json.load(f)
    with open(OUTPUT_DIR / "label_diff_report.json") as f:
        diff_report = json.load(f)

    # ---- Baseline accuracy ----
    logger.info("\n--- Baseline Accuracy ---")
    for label, mask_or_idx in [
        ("Total", slice(None)),
        ("Train", train_idx),
        ("Holdout", holdout_idx),
    ]:
        old_n = int(old_correct[mask_or_idx].sum())
        new_n = int(new_correct[mask_or_idx].sum())
        total = len(old_correct[mask_or_idx])
        logger.info(
            "  %s: %d/%d (%.1f%%) -> %d/%d (%.1f%%) [+%d]",
            label.ljust(8), old_n, total, 100 * old_n / total,
            new_n, total, 100 * new_n / total, new_n - old_n,
        )

    # ---- Wrong-greedy counts ----
    logger.info("\n--- Wrong-Greedy Counts ---")
    old_train_wrong = int((~old_correct[train_idx].astype(bool)).sum())
    new_train_wrong = int((~new_correct[train_idx].astype(bool)).sum())
    old_holdout_wrong = int((~old_correct[holdout_idx].astype(bool)).sum())
    new_holdout_wrong = int((~new_correct[holdout_idx].astype(bool)).sum())
    logger.info("  Train wrong: %d -> %d", old_train_wrong, new_train_wrong)
    logger.info("  Holdout wrong: %d -> %d", old_holdout_wrong, new_holdout_wrong)

    # ---- Oracle ceiling (problems with ≥1 correct temperature completion) ----
    logger.info("\n--- Oracle Ceiling ---")

    # Train: problems that are greedy-wrong AND have ≥1 correct completion
    train_solvable = 0
    train_total_correct_completions = 0
    train_total_completions = 0
    for key, data in train_temp.items():
        prob_idx = int(key)
        is_wrong_greedy = not new_correct[prob_idx]
        n_correct = sum(data["correct"])
        train_total_completions += len(data["correct"])
        train_total_correct_completions += n_correct
        if is_wrong_greedy and n_correct > 0:
            train_solvable += 1

    logger.info("  Train solvable (wrong-greedy, ≥1 correct temp): %d (was ~153)", train_solvable)
    logger.info(
        "  Train correct completions: %d/%d (%.1f%%)",
        train_total_correct_completions, train_total_completions,
        100 * train_total_correct_completions / train_total_completions,
    )

    # Holdout
    holdout_solvable = 0
    holdout_total_correct_completions = 0
    holdout_total_completions = 0
    for key, data in holdout_temp.items():
        prob_idx = int(key)
        is_wrong_greedy = not new_correct[prob_idx]
        n_correct = sum(data["correct"])
        holdout_total_completions += len(data["correct"])
        holdout_total_correct_completions += n_correct
        if is_wrong_greedy and n_correct > 0:
            holdout_solvable += 1

    logger.info(
        "  Holdout solvable (wrong-greedy, ≥1 correct temp): %d (was ~28)",
        holdout_solvable,
    )
    logger.info(
        "  Holdout correct completions: %d/%d (%.1f%%)",
        holdout_total_correct_completions, holdout_total_completions,
        100 * holdout_total_correct_completions / holdout_total_completions,
    )

    # Note: holdout temp gen only covers problems that were greedy-wrong under OLD labels
    # Some of those are now greedy-correct. We need to track which holdout problems
    # have temperature completions available.
    holdout_with_temp = set(int(k) for k in holdout_temp.keys())
    holdout_now_correct_with_temp = sum(
        1 for idx in holdout_with_temp if new_correct[idx]
    )
    holdout_now_wrong_with_temp = sum(
        1 for idx in holdout_with_temp if not new_correct[idx]
    )
    holdout_wrong_without_temp = new_holdout_wrong - holdout_now_wrong_with_temp

    logger.info("\n--- Holdout Temperature Coverage ---")
    logger.info(
        "  Holdout problems with temp gens: %d (originally wrong-greedy)",
        len(holdout_with_temp),
    )
    logger.info(
        "  Of those, now greedy-correct: %d (can be gated, temp gens unused)",
        holdout_now_correct_with_temp,
    )
    logger.info(
        "  Of those, still greedy-wrong: %d (these have temp gens for selection)",
        holdout_now_wrong_with_temp,
    )
    logger.info(
        "  Holdout wrong WITHOUT temp gens: %d (need new temp gens on RunPod)",
        holdout_wrong_without_temp,
    )

    # ---- Flipped indices detail ----
    logger.info("\n--- Flipped Greedy Problems ---")
    greedy_diff = diff_report["greedy"]["diff_details"]
    fn_holdout = [d for d in greedy_diff if d["index"] in set(holdout_idx) and d["flip_type"] == "false_negative_fixed"]
    fn_train = [d for d in greedy_diff if d["index"] in set(train_idx) and d["flip_type"] == "false_negative_fixed"]

    logger.info("  Holdout false negatives fixed (%d):", len(fn_holdout))
    for d in fn_holdout:
        logger.info(
            "    idx=%d: pred_v2=%r, gt=%r",
            d["index"], d["predicted_v2"], d["ground_truth"],
        )

    logger.info("  Train false negatives fixed: %d", len(fn_train))

    # ---- Save foundational counts ----
    counts = {
        "greedy_baseline": {
            "old_total": int(old_correct.sum()),
            "new_total": int(new_correct.sum()),
            "old_train": int(old_correct[train_idx].sum()),
            "new_train": int(new_correct[train_idx].sum()),
            "old_holdout": int(old_correct[holdout_idx].sum()),
            "new_holdout": int(new_correct[holdout_idx].sum()),
        },
        "wrong_greedy": {
            "old_train_wrong": old_train_wrong,
            "new_train_wrong": new_train_wrong,
            "old_holdout_wrong": old_holdout_wrong,
            "new_holdout_wrong": new_holdout_wrong,
        },
        "oracle_ceiling": {
            "train_solvable": train_solvable,
            "holdout_solvable": holdout_solvable,
        },
        "temperature_completions": {
            "train_correct_rate": train_total_correct_completions / train_total_completions,
            "train_correct": train_total_correct_completions,
            "train_total": train_total_completions,
            "holdout_correct_rate": holdout_total_correct_completions / holdout_total_completions,
            "holdout_correct": holdout_total_correct_completions,
            "holdout_total": holdout_total_completions,
        },
        "holdout_temp_coverage": {
            "has_temp_gens": len(holdout_with_temp),
            "now_correct_with_temp": holdout_now_correct_with_temp,
            "still_wrong_with_temp": holdout_now_wrong_with_temp,
            "wrong_without_temp": holdout_wrong_without_temp,
        },
        "elapsed_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "foundational_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    logger.info("\nSaved foundational_counts.json")
    logger.info("Elapsed: %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
