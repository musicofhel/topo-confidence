#!/usr/bin/env python3
"""Phase B2: Evaluation.

Evaluate trained bias on train-400 and holdout-100. Test both ungated
(bias applied to all problems) and confidence-gated (bias only when
topo-confidence < tau=0.40). Compare with Track A steering, Track A
selection, and greedy baseline.

Input: trained_bias.pt, baseline_correct.npy, confidence_scores.npy
Output: pathway4/track_b/ — train_results.json, holdout_results.json,
        comparison.json

Runtime: ~45 min GPU (500 greedy generations × 2 modes).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    BASELINE_CORRECT_PATH,
    CONFIDENCE_SCORES_PATH,
    SteeringBias,
    bayesian_p_improvement,
    bootstrap_ci_net_gain,
    check_correct,
    compute_flip_metrics,
    extract_answer,
    generate_one,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_model,
    load_math500,
    logger,
    mcnemar_mid_p,
    register_bias_hooks,
    remove_hooks,
    TRACK_A_BEST_TAU,
    TRACK_B_DIR,
)


def evaluate_with_bias(
    model,
    tokenizer,
    steering_bias: SteeringBias,
    problem_indices: np.ndarray,
    prompts: list[str],
    ground_truths: list[str],
    baseline_correct: np.ndarray,
    confidence_scores: np.ndarray | None = None,
    tau: float | None = None,
    mode_name: str = "ungated",
) -> dict:
    """Evaluate trained bias on a set of problems.

    Args:
        model: HF model
        tokenizer: Tokenizer
        steering_bias: Trained SteeringBias
        problem_indices: Global indices to evaluate
        prompts: Full prompt list
        ground_truths: Full ground truth list
        baseline_correct: (500,) baseline correctness
        confidence_scores: (500,) confidence scores (needed for gated mode)
        tau: Confidence threshold (None = ungated)
        mode_name: Name for logging

    Returns:
        Dict with metrics
    """
    logger.info("  Evaluating %s on %d problems...", mode_name, len(problem_indices))

    handles = register_bias_hooks(model, steering_bias)
    steered_correct = []

    t0 = time.time()
    with torch.no_grad():
        for pi, gi in enumerate(problem_indices):
            # Decide whether to apply bias
            if tau is not None and confidence_scores is not None:
                conf = confidence_scores[gi]
                if conf >= tau:
                    # High confidence — use greedy (remove hooks temporarily)
                    remove_hooks(handles)
                    answer_text = generate_one(model, tokenizer, prompts[gi])
                    handles = register_bias_hooks(model, steering_bias)
                else:
                    # Low confidence — use steered model
                    answer_text = generate_one(model, tokenizer, prompts[gi])
            else:
                # Ungated — always use steered model
                answer_text = generate_one(model, tokenizer, prompts[gi])

            answer = extract_answer(answer_text)
            is_correct = check_correct(answer, ground_truths[gi])
            steered_correct.append(is_correct)

            if (pi + 1) % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "    %d/%d done (%.1fs elapsed)",
                    pi + 1, len(problem_indices), elapsed,
                )

    remove_hooks(handles)

    # Compute metrics
    baseline_arr = baseline_correct[problem_indices].astype(bool)
    steered_arr = np.array(steered_correct, dtype=bool)

    metrics = compute_flip_metrics(baseline_arr, steered_arr)

    # Statistical tests
    w2r = metrics["wrong_to_right"]
    r2w = metrics["right_to_wrong"]
    metrics["mcnemar_mid_p"] = round(mcnemar_mid_p(r2w, w2r), 6)
    metrics["bayesian_p_improvement"] = round(bayesian_p_improvement(w2r, r2w), 6)
    ci_lo, ci_hi = bootstrap_ci_net_gain(
        baseline_arr.astype(int), steered_arr.astype(int)
    )
    metrics["bootstrap_ci_95"] = [ci_lo, ci_hi]
    metrics["mode"] = mode_name
    metrics["tau"] = tau
    metrics["eval_time_s"] = round(time.time() - t0, 1)

    return metrics


def main() -> None:
    t_start = time.time()
    logger.info("=== Phase B2: Evaluation ===")

    # Load trained bias
    bias_path = TRACK_B_DIR / "trained_bias.pt"
    if not bias_path.exists():
        logger.error("No trained bias found at %s. Run phase1_train.py first.", bias_path)
        return

    steering_bias = SteeringBias(hidden_dim=1536, layers=[24])
    state_dict = torch.load(bias_path, weights_only=True)
    steering_bias.load_state_dict(state_dict)
    logger.info("Loaded trained bias: norm=%.6f", steering_bias.total_norm())

    # Load data
    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    confidence_scores = np.load(CONFIDENCE_SCORES_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()

    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    # Load model
    model, tokenizer = load_model()
    steering_bias.to(model.device)

    # ---- Train-400 evaluation ----
    logger.info("\n--- Train-400 Evaluation ---")

    train_ungated = evaluate_with_bias(
        model, tokenizer, steering_bias, train_idx,
        prompts, ground_truths, baseline_correct,
        mode_name="ungated",
    )
    logger.info(
        "  Ungated: net_gain=%d, W→R=%d, R→W=%d, acc=%.3f",
        train_ungated["net_gain"], train_ungated["wrong_to_right"],
        train_ungated["right_to_wrong"], train_ungated["steered_accuracy"],
    )

    train_gated = evaluate_with_bias(
        model, tokenizer, steering_bias, train_idx,
        prompts, ground_truths, baseline_correct,
        confidence_scores=confidence_scores,
        tau=TRACK_A_BEST_TAU,
        mode_name="gated_tau0.40",
    )
    logger.info(
        "  Gated: net_gain=%d, W→R=%d, R→W=%d, acc=%.3f",
        train_gated["net_gain"], train_gated["wrong_to_right"],
        train_gated["right_to_wrong"], train_gated["steered_accuracy"],
    )

    train_results = {
        "ungated": train_ungated,
        "gated": train_gated,
        "bias_norm": round(steering_bias.total_norm(), 6),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(TRACK_B_DIR / "train_results.json", "w") as f:
        json.dump(train_results, f, indent=2)
    logger.info("  Saved train results")

    # ---- Holdout-100 evaluation ----
    logger.info("\n--- Holdout-100 Evaluation ---")

    holdout_ungated = evaluate_with_bias(
        model, tokenizer, steering_bias, holdout_idx,
        prompts, ground_truths, baseline_correct,
        mode_name="ungated",
    )
    logger.info(
        "  Ungated: net_gain=%d, W→R=%d, R→W=%d, acc=%.3f",
        holdout_ungated["net_gain"], holdout_ungated["wrong_to_right"],
        holdout_ungated["right_to_wrong"], holdout_ungated["steered_accuracy"],
    )

    holdout_gated = evaluate_with_bias(
        model, tokenizer, steering_bias, holdout_idx,
        prompts, ground_truths, baseline_correct,
        confidence_scores=confidence_scores,
        tau=TRACK_A_BEST_TAU,
        mode_name="gated_tau0.40",
    )
    logger.info(
        "  Gated: net_gain=%d, W→R=%d, R→W=%d, acc=%.3f",
        holdout_gated["net_gain"], holdout_gated["wrong_to_right"],
        holdout_gated["right_to_wrong"], holdout_gated["steered_accuracy"],
    )

    holdout_results = {
        "ungated": holdout_ungated,
        "gated": holdout_gated,
        "bias_norm": round(steering_bias.total_norm(), 6),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(TRACK_B_DIR / "holdout_results.json", "w") as f:
        json.dump(holdout_results, f, indent=2)
    logger.info("  Saved holdout results")

    # ---- Comparison table ----
    logger.info("\n--- Comparison ---")

    # Load Track A steering results for comparison
    track_a_holdout_path = (
        Path(__file__).parent.parent.parent
        / "pathway2" / "track_a" / "phase3" / "holdout_results.json"
    )
    track_a_steering = {"net_gain": 3, "wrong_to_right": 6, "right_to_wrong": 3}
    if track_a_holdout_path.exists():
        with open(track_a_holdout_path) as f:
            ta_data = json.load(f)
        metrics = ta_data.get("metrics", ta_data)
        track_a_steering = {
            "net_gain": metrics.get("net_gain", 3),
            "wrong_to_right": metrics.get("wrong_to_right", 6),
            "right_to_wrong": metrics.get("right_to_wrong", 3),
        }

    # Load Track A selection results if available
    track_a_selection = {"net_gain": "N/A", "wrong_to_right": "N/A", "right_to_wrong": 0}
    selection_path = Path(__file__).parent.parent / "track_a" / "phase3" / "holdout_results.json"
    if selection_path.exists():
        with open(selection_path) as f:
            sel_data = json.load(f)
        track_a_selection = {
            "net_gain": sel_data.get("net_gain", "N/A"),
            "wrong_to_right": sel_data.get("wrong_to_right", "N/A"),
            "right_to_wrong": sel_data.get("right_to_wrong", 0),
        }

    best_holdout = max(
        holdout_ungated["net_gain"], holdout_gated["net_gain"]
    )
    best_mode = "ungated" if holdout_ungated["net_gain"] >= holdout_gated["net_gain"] else "gated"
    best_p = max(
        holdout_ungated["bayesian_p_improvement"],
        holdout_gated["bayesian_p_improvement"],
    )

    comparison = {
        "holdout_comparison": {
            "greedy_baseline": {
                "accuracy": 0.11,
                "correct": 11,
                "net_gain": 0,
            },
            "track_a_steering": track_a_steering,
            "track_a_selection": track_a_selection,
            "track_b_ungated": {
                "accuracy": holdout_ungated["steered_accuracy"],
                "correct": holdout_ungated["steered_correct"],
                "net_gain": holdout_ungated["net_gain"],
                "wrong_to_right": holdout_ungated["wrong_to_right"],
                "right_to_wrong": holdout_ungated["right_to_wrong"],
            },
            "track_b_gated": {
                "accuracy": holdout_gated["steered_accuracy"],
                "correct": holdout_gated["steered_correct"],
                "net_gain": holdout_gated["net_gain"],
                "wrong_to_right": holdout_gated["wrong_to_right"],
                "right_to_wrong": holdout_gated["right_to_wrong"],
            },
        },
        "best_track_b_mode": best_mode,
        "best_track_b_holdout_net_gain": best_holdout,
        "best_track_b_p_improvement": round(best_p, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(TRACK_B_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("  Saved comparison")

    # ---- Go/No-Go ----
    if best_holdout >= 5 and best_p > 0.90:
        decision = "PUBLICATION_READY"
        rationale = (
            f"GRPO bias ({best_mode}) achieves +{best_holdout} holdout net gain "
            f"with P(improvement)={best_p:.3f} > 0.90."
        )
    elif best_holdout > 3:
        decision = "PROMISING"
        rationale = (
            f"GRPO bias ({best_mode}) improves over contrastive steering (+3) "
            f"with +{best_holdout} but P(improvement)={best_p:.3f}."
        )
    else:
        decision = "NO_IMPROVEMENT"
        rationale = (
            f"GRPO bias ({best_mode}) achieves +{best_holdout}, not better than "
            f"contrastive steering (+3)."
        )

    go_no_go = {
        "decision": decision,
        "rationale": rationale,
        "best_holdout_net_gain": best_holdout,
        "best_mode": best_mode,
        "bayesian_p_improvement": round(best_p, 4),
    }
    with open(TRACK_B_DIR / "go_no_go_decision.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    # ---- Final report ----
    logger.info("\n=== Phase B2 Summary ===")
    logger.info("Bias norm: %.6f", steering_bias.total_norm())
    logger.info("")
    logger.info("Train-400:")
    logger.info("  Ungated: net_gain=%+d, W→R=%d, R→W=%d",
                train_ungated["net_gain"], train_ungated["wrong_to_right"], train_ungated["right_to_wrong"])
    logger.info("  Gated:   net_gain=%+d, W→R=%d, R→W=%d",
                train_gated["net_gain"], train_gated["wrong_to_right"], train_gated["right_to_wrong"])
    logger.info("")
    logger.info("Holdout-100:")
    logger.info("  Ungated: net_gain=%+d, W→R=%d, R→W=%d, P=%.3f",
                holdout_ungated["net_gain"], holdout_ungated["wrong_to_right"],
                holdout_ungated["right_to_wrong"], holdout_ungated["bayesian_p_improvement"])
    logger.info("  Gated:   net_gain=%+d, W→R=%d, R→W=%d, P=%.3f",
                holdout_gated["net_gain"], holdout_gated["wrong_to_right"],
                holdout_gated["right_to_wrong"], holdout_gated["bayesian_p_improvement"])
    logger.info("")
    logger.info("Decision: %s", decision)
    logger.info("Total time: %.1f minutes", (time.time() - t_start) / 60)
    logger.info("Phase B2 complete.")


if __name__ == "__main__":
    main()
