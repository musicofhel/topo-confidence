#!/usr/bin/env python3
"""Step 1: Relabel ALL correctness labels using the corrected answer checker.

Relabels:
  1. 500 greedy baseline answers → baseline_correct_v2.npy
  2. 400 × 32 train temperature completions → temperature_generations_v2.json
  3. 100 × 32 holdout temperature completions → holdout_temperature_generations_v2.json

Produces:
  - baseline_correct_v2.npy
  - temperature_generations_v2.json
  - holdout_temperature_generations_v2.json
  - label_diff_report.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_PATH,
    TEMP_GEN_PATH,
    check_correct_v2,
    extract_answer_v2,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    logger,
)

OUTPUT_DIR = Path(__file__).parent
HOLDOUT_TEMP_GEN_PATH = (
    Path(__file__).parent.parent.parent
    / "pathway4"
    / "track_a"
    / "phase3"
    / "holdout_temperature_generations.json"
)
BASELINE_ANSWERS_PATH = (
    Path(__file__).parent.parent.parent
    / "pathway2"
    / "track_a"
    / "phase0"
    / "baseline_answers.json"
)


def relabel_greedy(ground_truths, baseline_answers):
    """Relabel 500 greedy answers with corrected checker."""
    old_correct = np.load(BASELINE_CORRECT_PATH)
    new_correct = np.zeros(len(baseline_answers), dtype=bool)
    diff_details = []

    for item in baseline_answers:
        idx = item["index"]
        gt = ground_truths[idx]
        gen_text = item["generated_text"]

        new_label = check_correct_v2(gen_text, gt)
        new_correct[idx] = new_label
        old_label = bool(old_correct[idx])

        if new_label != old_label:
            new_pred = extract_answer_v2(gen_text)
            diff_details.append(
                {
                    "index": idx,
                    "old_correct": old_label,
                    "new_correct": new_label,
                    "predicted_v2": new_pred,
                    "old_predicted": item["predicted_answer"],
                    "ground_truth": gt,
                    "flip_type": "false_negative_fixed"
                    if new_label
                    else "false_positive_found",
                }
            )

    return new_correct, old_correct, diff_details


def relabel_temperature(
    temp_gen: dict,
    ground_truths: list[str],
    label: str,
) -> tuple[dict, dict]:
    """Relabel temperature completions with corrected checker.

    Returns: (relabeled_data, stats_dict)
    """
    stats = {
        "total_completions": 0,
        "flipped_to_correct": 0,
        "flipped_to_wrong": 0,
        "unchanged": 0,
    }

    for prob_key, data in temp_gen.items():
        prob_idx = int(prob_key)
        gt = ground_truths[prob_idx]
        old_correct = data["correct"]  # list of 32 bools
        new_correct = []

        for i, completion in enumerate(data["completions"]):
            old_label = bool(old_correct[i])
            new_label = check_correct_v2(completion, gt)
            new_correct.append(new_label)
            stats["total_completions"] += 1
            if not old_label and new_label:
                stats["flipped_to_correct"] += 1
            elif old_label and not new_label:
                stats["flipped_to_wrong"] += 1
            else:
                stats["unchanged"] += 1

        data["correct"] = new_correct
        data["n_correct"] = sum(new_correct)

    return temp_gen, stats


def main():
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("PHASE 0 STEP 1: RELABEL ALL CORRECTNESS LABELS")
    logger.info("=" * 70)

    # Load MATH-500 ground truths
    logger.info("Loading MATH-500 dataset...")
    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)
    train_idx, holdout_idx = get_train_holdout_indices()

    # ---- 1. Relabel greedy baseline ----
    logger.info("\n--- Relabeling 500 greedy answers ---")
    with open(BASELINE_ANSWERS_PATH) as f:
        baseline_answers = json.load(f)

    new_correct, old_correct, diff_details = relabel_greedy(
        ground_truths, baseline_answers
    )

    old_total = int(old_correct.sum())
    new_total = int(new_correct.sum())
    fn_fixed = sum(1 for d in diff_details if d["flip_type"] == "false_negative_fixed")
    fp_found = sum(1 for d in diff_details if d["flip_type"] == "false_positive_found")

    logger.info("  Old baseline: %d/500 (%.1f%%)", old_total, 100 * old_total / 500)
    logger.info("  New baseline: %d/500 (%.1f%%)", new_total, 100 * new_total / 500)
    logger.info("  False negatives fixed: %d", fn_fixed)
    logger.info("  False positives found: %d", fp_found)

    # Per-split breakdown
    old_train = int(old_correct[train_idx].sum())
    new_train = int(new_correct[train_idx].sum())
    old_holdout = int(old_correct[holdout_idx].sum())
    new_holdout = int(new_correct[holdout_idx].sum())
    logger.info(
        "  Train: %d/%d -> %d/%d", old_train, len(train_idx), new_train, len(train_idx)
    )
    logger.info(
        "  Holdout: %d/%d -> %d/%d",
        old_holdout, len(holdout_idx), new_holdout, len(holdout_idx),
    )

    # Save greedy labels
    np.save(OUTPUT_DIR / "baseline_correct_v2.npy", new_correct)
    logger.info("  Saved baseline_correct_v2.npy")

    # ---- 2. Relabel train temperature completions ----
    logger.info("\n--- Relabeling train temperature completions ---")
    with open(TEMP_GEN_PATH) as f:
        train_temp = json.load(f)
    logger.info("  Loaded %d problems × 32 completions", len(train_temp))

    train_temp, train_stats = relabel_temperature(train_temp, ground_truths, "train")
    logger.info("  Total: %d", train_stats["total_completions"])
    logger.info("  Flipped to correct: %d", train_stats["flipped_to_correct"])
    logger.info("  Flipped to wrong: %d", train_stats["flipped_to_wrong"])
    logger.info("  Unchanged: %d", train_stats["unchanged"])

    with open(OUTPUT_DIR / "temperature_generations_v2.json", "w") as f:
        json.dump(train_temp, f)
    logger.info("  Saved temperature_generations_v2.json")

    # ---- 3. Relabel holdout temperature completions ----
    logger.info("\n--- Relabeling holdout temperature completions ---")
    with open(HOLDOUT_TEMP_GEN_PATH) as f:
        holdout_temp = json.load(f)
    logger.info("  Loaded %d problems × 32 completions", len(holdout_temp))

    holdout_temp, holdout_stats = relabel_temperature(
        holdout_temp, ground_truths, "holdout"
    )
    logger.info("  Total: %d", holdout_stats["total_completions"])
    logger.info("  Flipped to correct: %d", holdout_stats["flipped_to_correct"])
    logger.info("  Flipped to wrong: %d", holdout_stats["flipped_to_wrong"])
    logger.info("  Unchanged: %d", holdout_stats["unchanged"])

    with open(OUTPUT_DIR / "holdout_temperature_generations_v2.json", "w") as f:
        json.dump(holdout_temp, f)
    logger.info("  Saved holdout_temperature_generations_v2.json")

    # ---- 4. Save comprehensive diff report ----
    report = {
        "greedy": {
            "old_correct": old_total,
            "new_correct": new_total,
            "false_negatives_fixed": fn_fixed,
            "false_positives_found": fp_found,
            "old_train": old_train,
            "new_train": new_train,
            "old_holdout": old_holdout,
            "new_holdout": new_holdout,
            "diff_details": diff_details,
        },
        "train_temperature": train_stats,
        "holdout_temperature": holdout_stats,
        "elapsed_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "label_diff_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("\nSaved label_diff_report.json")

    # ---- Print summary ----
    logger.info("\n" + "=" * 70)
    logger.info("RELABELING COMPLETE")
    logger.info("=" * 70)
    logger.info("Greedy: %d/500 -> %d/500 (+%d)", old_total, new_total, new_total - old_total)
    logger.info(
        "  Train: %d -> %d, Holdout: %d -> %d",
        old_train, new_train, old_holdout, new_holdout,
    )
    logger.info(
        "Train temp: %d flipped (→correct: %d, →wrong: %d)",
        train_stats["flipped_to_correct"] + train_stats["flipped_to_wrong"],
        train_stats["flipped_to_correct"],
        train_stats["flipped_to_wrong"],
    )
    logger.info(
        "Holdout temp: %d flipped (→correct: %d, →wrong: %d)",
        holdout_stats["flipped_to_correct"] + holdout_stats["flipped_to_wrong"],
        holdout_stats["flipped_to_correct"],
        holdout_stats["flipped_to_wrong"],
    )
    logger.info("Elapsed: %.1f seconds", time.time() - t0)

    if fp_found > 0:
        logger.warning(
            "WARNING: %d false positives found! Investigate before proceeding.", fp_found
        )
        for d in diff_details:
            if d["flip_type"] == "false_positive_found":
                logger.warning(
                    "  idx=%d: old=%s -> new=%s, pred=%r, gt=%r",
                    d["index"], d["old_correct"], d["new_correct"],
                    d["predicted_v2"], d["ground_truth"],
                )


if __name__ == "__main__":
    main()
