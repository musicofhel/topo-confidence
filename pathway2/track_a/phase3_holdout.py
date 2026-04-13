#!/usr/bin/env python3
"""Phase 3: Holdout Evaluation.

Test locked configuration on holdout-100 (honest answer).
Three conditions: no steering, uniform, targeted.
Compute metrics, bootstrap CIs, McNemar's test, per-problem analysis.
Make GO/NO-GO decision.

Input: phase2/locked_config.json, phase0/confidence_scores.npy,
       phase1/steering_vector.npy, holdout mask
Output: phase3/holdout_results.json, phase3/statistical_tests.json,
        phase3/per_problem_analysis.json, phase3/per_problem_analysis.txt,
        phase3/holdout_comparison_table.txt

Runtime: ~2 hours GPU.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from scipy import stats

from common import (
    PHASE0_DIR,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    SEED,
    check_correct,
    extract_answer,
    generate_one,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    load_model,
    logger,
    make_steering_hook,
)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of a binary array."""
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot_means.append(float(values[idx].mean()))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test for paired binary outcomes.

    Returns dict with test statistic, p-value, and contingency counts.
    """
    # 2x2 contingency: (both correct, a-only, b-only, both wrong)
    both_correct = int(((correct_a == 1) & (correct_b == 1)).sum())
    a_only = int(((correct_a == 1) & (correct_b == 0)).sum())
    b_only = int(((correct_a == 0) & (correct_b == 1)).sum())
    both_wrong = int(((correct_a == 0) & (correct_b == 0)).sum())

    b = a_only  # baseline correct, steered wrong
    c = b_only  # baseline wrong, steered correct

    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        # McNemar's with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    return {
        "both_correct": both_correct,
        "baseline_only_correct": b,
        "steered_only_correct": c,
        "both_wrong": both_wrong,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "n": len(correct_a),
    }


def run_holdout_condition(
    model, tokenizer, prompts, ground_truths, holdout_idx,
    steering_vec_gpu=None, layer_idx=None, t=None,
    confidence_scores=None, tau_steer=None,
    label="baseline"
) -> tuple[np.ndarray, list[dict]]:
    """Run one experimental condition on holdout-100.

    For 'baseline': generate normally.
    For 'uniform': apply steering to all.
    For 'targeted': apply steering only when confidence < tau_steer.

    Returns:
        correct_array: (100,) int
        per_problem: list of dicts with details
    """
    n = len(holdout_idx)
    correct_array = np.zeros(n, dtype=int)
    per_problem = []

    for i, idx in enumerate(holdout_idx):
        if i % 20 == 0:
            logger.info("  %s: %d/%d...", label, i, n)

        need_steering = False
        if label == "uniform":
            need_steering = True
        elif label == "targeted" and confidence_scores is not None and tau_steer is not None:
            need_steering = confidence_scores[idx] < tau_steer

        if need_steering and steering_vec_gpu is not None and layer_idx is not None:
            hook = make_steering_hook(steering_vec_gpu, t)
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            text = generate_one(model, tokenizer, prompts[idx])
            handle.remove()
        else:
            text = generate_one(model, tokenizer, prompts[idx])

        predicted = extract_answer(text)
        is_correct = check_correct(predicted, ground_truths[idx])
        correct_array[i] = int(is_correct)

        per_problem.append({
            "holdout_idx": i,
            "global_idx": int(idx),
            "correct": int(is_correct),
            "steered": int(need_steering),
            "generated_text": text[:500],
            "predicted_answer": predicted,
            "ground_truth": ground_truths[idx],
        })

    return correct_array, per_problem


def compute_condition_metrics(
    correct: np.ndarray,
    baseline_correct: np.ndarray,
    label: str,
    steered_flags: np.ndarray | None = None,
) -> dict:
    """Compute metrics for one condition relative to baseline."""
    n = len(correct)
    acc = float(correct.mean())
    ci_low, ci_high = bootstrap_ci(correct)

    orig_correct_mask = baseline_correct.astype(bool)
    orig_incorrect_mask = ~orig_correct_mask

    preserved = int(correct[orig_correct_mask].sum()) if orig_correct_mask.any() else 0
    recovered = int(correct[orig_incorrect_mask].sum()) if orig_incorrect_mask.any() else 0
    newly_wrong = int(orig_correct_mask.sum()) - preserved
    newly_correct = recovered
    net_gain = newly_correct - newly_wrong

    metrics = {
        "label": label,
        "accuracy": round(acc, 4),
        "n_correct": int(correct.sum()),
        "n_total": n,
        "bootstrap_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "orig_correct_preserved": preserved,
        "orig_correct_total": int(orig_correct_mask.sum()),
        "orig_correct_lost": newly_wrong,
        "orig_incorrect_recovered": recovered,
        "orig_incorrect_total": int(orig_incorrect_mask.sum()),
        "net_gain": net_gain,
    }

    if steered_flags is not None:
        steered_mask = steered_flags.astype(bool)
        metrics["n_steered"] = int(steered_mask.sum())
        if steered_mask.any():
            # Flips within steered subset
            steered_and_orig_wrong = steered_mask & orig_incorrect_mask
            steered_and_orig_correct = steered_mask & orig_correct_mask
            metrics["steered_wrong_to_right"] = int(correct[steered_and_orig_wrong].sum())
            metrics["steered_right_to_wrong"] = int((~correct.astype(bool))[steered_and_orig_correct].sum())
            metrics["steered_stayed_wrong"] = int(
                steered_and_orig_wrong.sum() - correct[steered_and_orig_wrong].sum()
            )

    return metrics


def go_no_go_decision(
    uniform_metrics: dict,
    targeted_metrics: dict,
    baseline_metrics: dict,
    probe_aurocs: dict,
) -> dict:
    """Determine GO/NO-GO based on holdout results."""
    uniform_gain = uniform_metrics["net_gain"]
    targeted_gain = targeted_metrics["net_gain"]
    any_gain = max(uniform_gain, targeted_gain)
    best_probe_auroc = max(probe_aurocs.values())

    if any_gain >= 3 and targeted_gain > uniform_gain:
        decision = "SKIP_TRACK_B_PROCEED_PATHWAY3"
        reason = (
            f"Net gain >= 3 ({any_gain}) AND targeted ({targeted_gain}) > "
            f"uniform ({uniform_gain}). Strong evidence for topology-guided steering."
        )
    elif any_gain >= 1:
        decision = "GO_TRACK_B"
        reason = (
            f"Net gain >= 1 ({any_gain}) under "
            f"{'uniform' if uniform_gain >= targeted_gain else 'targeted'} condition."
        )
    elif any_gain == 0 and best_probe_auroc >= 0.65:
        decision = "GO_TRACK_B_NO_EFFECT_VARIANT"
        reason = (
            f"Net gain = 0 but probe AUROC = {best_probe_auroc:.4f} >= 0.65. "
            f"Signal exists but steering may need different approach."
        )
    else:
        decision = "NO_GO"
        reason = (
            f"Net gain <= 0 under all conditions (uniform={uniform_gain}, "
            f"targeted={targeted_gain}). "
            f"Best probe AUROC = {best_probe_auroc:.4f}."
        )

    return {
        "decision": decision,
        "reason": reason,
        "uniform_net_gain": uniform_gain,
        "targeted_net_gain": targeted_gain,
        "best_probe_auroc": round(best_probe_auroc, 4),
    }


def main():
    print("=" * 60)
    print("Phase 3: Holdout Evaluation")
    print("=" * 60)

    # Load locked config
    with open(PHASE2_DIR / "locked_config.json") as f:
        config = json.load(f)
    print(f"\n  Locked config: {json.dumps(config, indent=2)}")

    selected_layer = config["layer"]
    best_t = config["best_t"]
    best_tau = config["best_tau_steer"]

    # Load data
    steering_vector = np.load(PHASE1_DIR / "steering_vector.npy")
    confidence_scores = np.load(PHASE0_DIR / "confidence_scores.npy")
    baseline_correct_all = np.load(PHASE0_DIR / "baseline_correct.npy")
    with open(PHASE0_DIR / "probe_aurocs.json") as f:
        probe_aurocs = json.load(f)

    train_idx, holdout_idx = get_train_holdout_indices()
    baseline_correct_holdout = baseline_correct_all[holdout_idx]

    # Step 1: Verify holdout mask
    print(f"\n[1] Holdout: {len(holdout_idx)} problems, "
          f"{baseline_correct_holdout.sum()} correct in baseline")
    assert len(holdout_idx) == 100

    # Load problems
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    # Load model
    model, tokenizer = load_model()
    steering_vec_gpu = torch.tensor(
        steering_vector, dtype=torch.float16, device=model.device
    )

    # ============================================
    # Step 2: Run three conditions
    # ============================================

    # (a) No steering
    print("\n[2a] Condition A: No steering...")
    t0 = time.time()
    correct_baseline, details_baseline = run_holdout_condition(
        model, tokenizer, prompts, ground_truths, holdout_idx,
        label="baseline"
    )
    print(f"  Done in {time.time() - t0:.0f}s. Accuracy: {correct_baseline.mean():.2%}")

    # Verify baseline matches Phase 0
    phase0_holdout_correct = baseline_correct_all[holdout_idx]
    baseline_diff = int(np.abs(correct_baseline - phase0_holdout_correct).sum())
    if baseline_diff > 0:
        logger.warning(
            "Baseline regeneration differs from Phase 0 on %d problems", baseline_diff
        )

    # (b) Uniform steering
    print(f"\n[2b] Condition B: Uniform steering (t={best_t:.2f})...")
    t0 = time.time()
    correct_uniform, details_uniform = run_holdout_condition(
        model, tokenizer, prompts, ground_truths, holdout_idx,
        steering_vec_gpu=steering_vec_gpu, layer_idx=selected_layer, t=best_t,
        label="uniform"
    )
    print(f"  Done in {time.time() - t0:.0f}s. Accuracy: {correct_uniform.mean():.2%}")

    # (c) Targeted steering
    print(f"\n[2c] Condition C: Targeted steering (t={best_t:.2f}, tau={best_tau:.2f})...")
    t0 = time.time()
    correct_targeted, details_targeted = run_holdout_condition(
        model, tokenizer, prompts, ground_truths, holdout_idx,
        steering_vec_gpu=steering_vec_gpu, layer_idx=selected_layer, t=best_t,
        confidence_scores=confidence_scores, tau_steer=best_tau,
        label="targeted"
    )
    print(f"  Done in {time.time() - t0:.0f}s. Accuracy: {correct_targeted.mean():.2%}")

    # Verify: un-steered targeted problems match baseline
    targeted_steered_flags = np.array([d["steered"] for d in details_targeted])
    unsteered_mask = ~targeted_steered_flags.astype(bool)
    if unsteered_mask.any():
        unsteered_match = (correct_targeted[unsteered_mask] == correct_baseline[unsteered_mask]).all()
        if not unsteered_match:
            logger.warning(
                "Targeted un-steered problems don't match baseline! "
                "This indicates a bug in the steering gating logic."
            )
        else:
            logger.info("  Targeted un-steered subset matches baseline (verified)")

    # ============================================
    # Step 3: Compute metrics
    # ============================================
    print("\n[3] Computing metrics...")

    metrics_baseline = compute_condition_metrics(
        correct_baseline, correct_baseline, "baseline"
    )
    metrics_uniform = compute_condition_metrics(
        correct_uniform, correct_baseline, "uniform"
    )
    metrics_targeted = compute_condition_metrics(
        correct_targeted, correct_baseline, "targeted",
        steered_flags=targeted_steered_flags
    )

    for m in [metrics_baseline, metrics_uniform, metrics_targeted]:
        print(f"  {m['label']}: acc={m['accuracy']:.4f} "
              f"CI=[{m['bootstrap_ci_95'][0]:.4f}, {m['bootstrap_ci_95'][1]:.4f}] "
              f"net_gain={m['net_gain']}")

    # ============================================
    # Step 4: McNemar's test
    # ============================================
    print("\n[4] McNemar's test...")

    mcnemar_uniform = mcnemar_test(correct_baseline, correct_uniform)
    mcnemar_targeted = mcnemar_test(correct_baseline, correct_targeted)

    print(f"  Baseline vs Uniform: chi2={mcnemar_uniform['chi2']:.4f}, "
          f"p={mcnemar_uniform['p_value']:.4f} "
          f"(b={mcnemar_uniform['baseline_only_correct']}, "
          f"c={mcnemar_uniform['steered_only_correct']})")
    print(f"  Baseline vs Targeted: chi2={mcnemar_targeted['chi2']:.4f}, "
          f"p={mcnemar_targeted['p_value']:.4f} "
          f"(b={mcnemar_targeted['baseline_only_correct']}, "
          f"c={mcnemar_targeted['steered_only_correct']})")
    print("  Note: n=100 limits statistical power.")

    # ============================================
    # Step 5: Per-problem analysis
    # ============================================
    print("\n[5] Per-problem analysis...")

    per_problem_analysis = []
    for i, idx in enumerate(holdout_idx):
        entry = {
            "holdout_idx": i,
            "global_idx": int(idx),
            "topo_confidence": round(float(confidence_scores[idx]), 4),
            "steered_in_targeted": int(details_targeted[i]["steered"]),
            "baseline_correct": int(correct_baseline[i]),
            "uniform_correct": int(correct_uniform[i]),
            "targeted_correct": int(correct_targeted[i]),
        }

        # Classify flip type (for uniform)
        if correct_baseline[i] == 0 and correct_uniform[i] == 1:
            entry["uniform_flip"] = "wrong_to_right"
        elif correct_baseline[i] == 1 and correct_uniform[i] == 0:
            entry["uniform_flip"] = "right_to_wrong"
        else:
            entry["uniform_flip"] = "no_change"

        # Classify flip type (for targeted)
        if correct_baseline[i] == 0 and correct_targeted[i] == 1:
            entry["targeted_flip"] = "wrong_to_right"
        elif correct_baseline[i] == 1 and correct_targeted[i] == 0:
            entry["targeted_flip"] = "right_to_wrong"
        else:
            entry["targeted_flip"] = "no_change"

        per_problem_analysis.append(entry)

    # Save per-problem JSON
    with open(PHASE3_DIR / "per_problem_analysis.json", "w") as f:
        json.dump(per_problem_analysis, f, indent=2)

    # Save per-problem TXT
    lines = [
        f"{'Idx':>4} {'Conf':>6} {'Steer':>5} {'Base':>4} {'Unif':>4} {'Targ':>4} {'U-Flip':>14} {'T-Flip':>14}",
        "-" * 65,
    ]
    for entry in per_problem_analysis:
        lines.append(
            f"{entry['global_idx']:4d} {entry['topo_confidence']:6.3f} "
            f"{'Y' if entry['steered_in_targeted'] else 'N':>5} "
            f"{entry['baseline_correct']:4d} {entry['uniform_correct']:4d} "
            f"{entry['targeted_correct']:4d} "
            f"{entry['uniform_flip']:>14} {entry['targeted_flip']:>14}"
        )
    with open(PHASE3_DIR / "per_problem_analysis.txt", "w") as f:
        f.write("\n".join(lines))

    # ============================================
    # Step 6: Go/No-Go decision
    # ============================================
    print("\n[6] Go/No-Go decision...")

    decision = go_no_go_decision(
        metrics_uniform, metrics_targeted, metrics_baseline, probe_aurocs
    )
    print(f"\n  DECISION: {decision['decision']}")
    print(f"  Reason: {decision['reason']}")

    # ============================================
    # Step 7: Save all artifacts
    # ============================================
    print("\n[7] Saving artifacts...")

    holdout_results = {
        "config": config,
        "conditions": {
            "baseline": metrics_baseline,
            "uniform": metrics_uniform,
            "targeted": metrics_targeted,
        },
        "decision": decision,
        "baseline_regeneration_diff": baseline_diff,
    }
    with open(PHASE3_DIR / "holdout_results.json", "w") as f:
        json.dump(holdout_results, f, indent=2)

    statistical_tests = {
        "mcnemar_baseline_vs_uniform": mcnemar_uniform,
        "mcnemar_baseline_vs_targeted": mcnemar_targeted,
        "note": "n=100 limits power. McNemar's with continuity correction.",
    }
    with open(PHASE3_DIR / "statistical_tests.json", "w") as f:
        json.dump(statistical_tests, f, indent=2)

    # Comparison table
    table_lines = [
        "Phase 3: Holdout-100 Evaluation Results",
        "=" * 60,
        "",
        f"Config: layer={config['layer']}, t={config['best_t']}, "
        f"tau={config['best_tau_steer']}, method={config['method']}",
        "",
        f"{'Condition':>12} {'Acc':>8} {'CI_low':>8} {'CI_high':>8} {'Gain':>6} {'Lost':>6} {'Recov':>6}",
        "-" * 60,
    ]
    for m in [metrics_baseline, metrics_uniform, metrics_targeted]:
        ci = m["bootstrap_ci_95"]
        table_lines.append(
            f"{m['label']:>12} {m['accuracy']:8.4f} {ci[0]:8.4f} {ci[1]:8.4f} "
            f"{m['net_gain']:+6d} {m.get('orig_correct_lost', 0):6d} "
            f"{m.get('orig_incorrect_recovered', 0):6d}"
        )
    table_lines += [
        "",
        f"McNemar (baseline vs uniform): chi2={mcnemar_uniform['chi2']:.4f}, "
        f"p={mcnemar_uniform['p_value']:.4f}",
        f"McNemar (baseline vs targeted): chi2={mcnemar_targeted['chi2']:.4f}, "
        f"p={mcnemar_targeted['p_value']:.4f}",
        "",
        f"DECISION: {decision['decision']}",
        f"  {decision['reason']}",
    ]
    comparison_text = "\n".join(table_lines)
    with open(PHASE3_DIR / "holdout_comparison_table.txt", "w") as f:
        f.write(comparison_text)

    # Final summary
    print(f"\n{comparison_text}")
    print(f"\n  All artifacts saved to {PHASE3_DIR}/")
    print(f"\n{'=' * 60}")
    print(f"  TRACK A COMPLETE")
    print(f"  Decision: {decision['decision']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
