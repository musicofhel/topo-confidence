#!/usr/bin/env python3
"""Phase 2: Steering Strength Calibration on Train-400.

1. Implement spherical steering hook
2. Sanity check: t=0 must match baseline on 10 problems
3. Sweep t in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50} on train-400
4. Select best uniform t (maximize net gain, <=2 correct-side losses)
5. Sweep tau_steer thresholds with best t
6. Lock full configuration

Input: phase1/steering_vector.npy, phase0/selected_layer.txt,
       phase0/baseline_answers.json, phase0/confidence_scores.npy,
       phase0/baseline_correct.npy
Output: phase2/uniform_sweep.json, phase2/targeted_sweep.json,
        phase2/best_t.txt, phase2/locked_config.json,
        phase2/comparison_table.txt

Runtime: ~6 hours GPU.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    PHASE0_DIR,
    PHASE1_DIR,
    PHASE2_DIR,
    SEED,
    YOUDEN_THRESHOLD,
    check_correct,
    extract_answer,
    generate_one,
    generate_one_with_scores,
    get_train_holdout_indices,
    load_math500,
    load_model,
    get_prompts_and_ground_truths,
    logger,
    make_steering_hook,
)

T_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
TAU_STEER_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
KL_SAMPLE_SIZE = 20


def compute_kl_divergence(
    model, tokenizer, prompts: list[str], steering_vec_gpu, t, layer_idx, n_samples=KL_SAMPLE_SIZE
) -> float:
    """Compute mean KL(steered || unsteered) on random sample of prompts.

    Uses output logits to compute per-token KL divergence, then averages.
    """
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(prompts), size=min(n_samples, len(prompts)), replace=False)

    kl_values = []
    for idx in sample_idx:
        prompt = prompts[idx]

        # Unsteered generation
        _, logits_base = generate_one_with_scores(model, tokenizer, prompt)
        if logits_base.numel() == 0:
            continue

        # Steered generation
        hook = make_steering_hook(steering_vec_gpu, t)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        _, logits_steered = generate_one_with_scores(model, tokenizer, prompt)
        handle.remove()

        if logits_steered.numel() == 0:
            continue

        # Align lengths (they may differ due to different generation paths)
        min_len = min(logits_base.shape[0], logits_steered.shape[0])
        if min_len == 0:
            continue

        p_base = F.softmax(logits_base[:min_len].float(), dim=-1)
        p_steer = F.softmax(logits_steered[:min_len].float(), dim=-1)

        # KL(steered || baseline) per token, then mean
        kl_per_token = (p_steer * (p_steer.log() - p_base.log())).sum(dim=-1)
        kl_values.append(float(kl_per_token.mean()))

    return float(np.mean(kl_values)) if kl_values else 0.0


def run_uniform_sweep_single_t(
    model, tokenizer, prompts, ground_truths, train_idx, baseline_correct,
    steering_vec_gpu, layer_idx, t
) -> dict:
    """Run uniform steering at strength t on all train-400 problems.

    Returns dict with accuracy metrics and per-problem results.
    """
    n_train = len(train_idx)
    results = []
    steered_correct = np.zeros(n_train, dtype=int)

    # Register hook
    if t > 0:
        hook = make_steering_hook(steering_vec_gpu, t)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)

    for i, idx in enumerate(train_idx):
        if i % 50 == 0:
            logger.info("  t=%.2f: generating %d/%d...", t, i, n_train)

        text = generate_one(model, tokenizer, prompts[idx])
        predicted = extract_answer(text)
        is_correct = check_correct(predicted, ground_truths[idx])
        steered_correct[i] = int(is_correct)

        results.append({
            "train_idx": i,
            "global_idx": int(idx),
            "correct": int(is_correct),
            "baseline_correct": int(baseline_correct[idx]),
            "generated_text": text[:500],
        })

    if t > 0:
        handle.remove()

    # Compute metrics
    orig_correct_mask = baseline_correct[train_idx].astype(bool)
    orig_incorrect_mask = ~orig_correct_mask

    overall_acc = float(steered_correct.mean())
    orig_correct_preserved = int(steered_correct[orig_correct_mask].sum())
    orig_correct_total = int(orig_correct_mask.sum())
    orig_incorrect_recovered = int(steered_correct[orig_incorrect_mask].sum())
    orig_incorrect_total = int(orig_incorrect_mask.sum())

    newly_correct = int(((steered_correct == 1) & orig_incorrect_mask).sum())
    newly_wrong = int(((steered_correct == 0) & orig_correct_mask).sum())
    net_gain = newly_correct - newly_wrong

    return {
        "t": t,
        "overall_accuracy": round(overall_acc, 4),
        "overall_correct": int(steered_correct.sum()),
        "orig_correct_preserved": orig_correct_preserved,
        "orig_correct_total": orig_correct_total,
        "orig_correct_lost": orig_correct_total - orig_correct_preserved,
        "orig_incorrect_recovered": orig_incorrect_recovered,
        "orig_incorrect_total": orig_incorrect_total,
        "newly_correct": newly_correct,
        "newly_wrong": newly_wrong,
        "net_gain": net_gain,
        "per_problem": results,
    }


def run_targeted_sweep_single_tau(
    model, tokenizer, prompts, ground_truths, train_idx, baseline_correct,
    baseline_answers, confidence_scores, steering_vec_gpu, layer_idx, best_t, tau_steer
) -> dict:
    """Run targeted steering: steer only when confidence < tau_steer.

    Optimization: reuse baseline answers for un-steered problems.
    """
    n_train = len(train_idx)
    steered_correct = np.zeros(n_train, dtype=int)
    steered_flags = np.zeros(n_train, dtype=int)
    results = []

    # Pre-compute which problems need steering
    conf_train = confidence_scores[train_idx]
    needs_steering = conf_train < tau_steer
    n_steered = int(needs_steering.sum())
    logger.info("  tau=%.2f: %d/%d problems need steering (%.1f%%)",
                tau_steer, n_steered, n_train, 100 * n_steered / n_train)

    # Build baseline answer lookup
    baseline_lookup = {}
    for entry in baseline_answers:
        baseline_lookup[entry["index"]] = entry

    # Register hook for steered problems
    hook = make_steering_hook(steering_vec_gpu, best_t)

    for i, idx in enumerate(train_idx):
        if needs_steering[i]:
            # Steer this problem
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            text = generate_one(model, tokenizer, prompts[idx])
            handle.remove()

            predicted = extract_answer(text)
            is_correct = check_correct(predicted, ground_truths[idx])
            steered_correct[i] = int(is_correct)
            steered_flags[i] = 1
        else:
            # Reuse baseline answer
            steered_correct[i] = int(baseline_correct[idx])
            steered_flags[i] = 0

        if i % 100 == 0 and needs_steering[i]:
            logger.info("  tau=%.2f: %d/%d (steered so far)", tau_steer, i, n_train)

        results.append({
            "train_idx": i,
            "global_idx": int(idx),
            "correct": int(steered_correct[i]),
            "baseline_correct": int(baseline_correct[idx]),
            "steered": int(steered_flags[i]),
            "confidence": round(float(conf_train[i]), 4),
        })

    # Compute metrics
    orig_correct_mask = baseline_correct[train_idx].astype(bool)
    orig_incorrect_mask = ~orig_correct_mask

    overall_acc = float(steered_correct.mean())
    newly_correct = int(((steered_correct == 1) & orig_incorrect_mask).sum())
    newly_wrong = int(((steered_correct == 0) & orig_correct_mask).sum())
    net_gain = newly_correct - newly_wrong

    # Steered-subset metrics
    steered_mask = steered_flags.astype(bool)
    if steered_mask.any():
        steered_subset_acc = float(steered_correct[steered_mask].mean())
        steered_baseline_acc = float(baseline_correct[train_idx][steered_mask].mean())
    else:
        steered_subset_acc = 0.0
        steered_baseline_acc = 0.0

    return {
        "tau_steer": tau_steer,
        "best_t": best_t,
        "n_steered": n_steered,
        "n_total": n_train,
        "coverage": round(n_steered / n_train, 4),
        "overall_accuracy": round(overall_acc, 4),
        "overall_correct": int(steered_correct.sum()),
        "newly_correct": newly_correct,
        "newly_wrong": newly_wrong,
        "net_gain": net_gain,
        "steered_subset_accuracy": round(steered_subset_acc, 4),
        "steered_subset_baseline_accuracy": round(steered_baseline_acc, 4),
        "per_problem": results,
    }


def main():
    print("=" * 60)
    print("Phase 2: Steering Strength Calibration on Train-400")
    print("=" * 60)

    # Load inputs
    with open(PHASE0_DIR / "selected_layer.txt") as f:
        content = f.read().strip()
    if content.startswith("HALT"):
        print(f"  Phase 0.5 halted: {content}")
        return

    selected_layer = int(content)
    print(f"\n  Selected layer: {selected_layer}")

    steering_vector = np.load(PHASE1_DIR / "steering_vector.npy")  # (1536,)
    baseline_correct = np.load(PHASE0_DIR / "baseline_correct.npy")  # (500,)
    confidence_scores = np.load(PHASE0_DIR / "confidence_scores.npy")  # (500,)
    with open(PHASE0_DIR / "baseline_answers.json") as f:
        baseline_answers = json.load(f)
    with open(PHASE0_DIR / "probe_aurocs.json") as f:
        probe_aurocs = json.load(f)

    train_idx, holdout_idx = get_train_holdout_indices()

    # Load problems
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    # Load model
    model, tokenizer = load_model()

    # Move steering vector to GPU
    steering_vec_gpu = torch.tensor(
        steering_vector, dtype=torch.float16, device=model.device
    )

    # ============================================
    # Step 1: Sanity check — t=0 must match baseline
    # ============================================
    print("\n[1] Sanity check: t=0 must match baseline...")
    rng = np.random.default_rng(SEED)
    sanity_indices = rng.choice(train_idx, size=10, replace=False)

    hook = make_steering_hook(steering_vec_gpu, 0.0)
    handle = model.model.layers[selected_layer].register_forward_hook(hook)

    mismatches = 0
    for idx in sanity_indices:
        text_steered = generate_one(model, tokenizer, prompts[idx])
        baseline_entry = baseline_answers[idx]
        text_baseline = baseline_entry["generated_text"]

        # Compare first 200 chars (baseline text may be truncated at 1000)
        if text_steered[:200] != text_baseline[:200]:
            mismatches += 1
            logger.warning(
                "  Mismatch at idx %d:\n    steered: %s\n    baseline: %s",
                idx, text_steered[:100], text_baseline[:100]
            )

    handle.remove()

    if mismatches > 0:
        print(f"  *** HALT: {mismatches}/10 mismatches at t=0 ***")
        print("  The steering hook is altering outputs even at t=0.")
        print("  Debug: verify hook target, slerp implementation, norm preservation.")
        return
    else:
        print(f"  PASS: All 10 sanity checks match baseline at t=0")

    # ============================================
    # Step 2: Uniform sweep on train-400
    # ============================================
    print("\n[2] Uniform sweep on train-400...")
    uniform_results = []

    for t in T_VALUES:
        # Check for checkpoint
        checkpoint_path = PHASE2_DIR / f"checkpoint_uniform_t{t:.2f}.json"
        if checkpoint_path.exists():
            logger.info("  Loading checkpoint for t=%.2f", t)
            with open(checkpoint_path) as f:
                result = json.load(f)
            uniform_results.append(result)
            print(f"  t={t:.2f}: net_gain={result['net_gain']}, "
                  f"acc={result['overall_accuracy']:.4f} (from checkpoint)")
            continue

        print(f"\n  --- t = {t:.2f} ---")
        t0 = time.time()
        result = run_uniform_sweep_single_t(
            model, tokenizer, prompts, ground_truths, train_idx, baseline_correct,
            steering_vec_gpu, selected_layer, t
        )
        elapsed = time.time() - t0

        # KL divergence on sample
        if t > 0:
            logger.info("  Computing KL divergence...")
            kl_div = compute_kl_divergence(
                model, tokenizer,
                [prompts[idx] for idx in train_idx],
                steering_vec_gpu, t, selected_layer
            )
            result["kl_divergence"] = round(kl_div, 6)
        else:
            result["kl_divergence"] = 0.0

        result["elapsed_s"] = round(elapsed, 1)
        uniform_results.append(result)

        print(f"  t={t:.2f}: net_gain={result['net_gain']}, "
              f"acc={result['overall_accuracy']:.4f}, "
              f"lost={result['orig_correct_lost']}, "
              f"recovered={result['orig_incorrect_recovered']}, "
              f"KL={result['kl_divergence']:.4f}, "
              f"time={elapsed:.0f}s")

        # Save checkpoint (without per_problem to save space)
        checkpoint = {k: v for k, v in result.items() if k != "per_problem"}
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # ============================================
    # Step 3: Select best uniform t
    # ============================================
    print("\n[3] Selecting best uniform t...")

    # Filter: originally-correct loses <= 2, then maximize net gain
    valid = [
        r for r in uniform_results
        if r["orig_correct_lost"] <= 2 and r["t"] > 0
    ]

    if not valid:
        print("  No t value satisfies constraint (correct-side loss <= 2).")
        print("  Relaxing to allow up to 5 losses...")
        valid = [
            r for r in uniform_results
            if r["orig_correct_lost"] <= 5 and r["t"] > 0
        ]

    if valid:
        best_uniform = max(valid, key=lambda r: r["net_gain"])
        best_t = best_uniform["t"]
    else:
        # Fall back to t with best net gain regardless
        non_zero = [r for r in uniform_results if r["t"] > 0]
        best_uniform = max(non_zero, key=lambda r: r["net_gain"])
        best_t = best_uniform["t"]

    print(f"  Best t: {best_t:.2f} (net_gain={best_uniform['net_gain']}, "
          f"lost={best_uniform['orig_correct_lost']})")

    with open(PHASE2_DIR / "best_t.txt", "w") as f:
        f.write(f"{best_t:.2f}\n")

    # Save uniform sweep (compact — strip per_problem)
    uniform_compact = [{k: v for k, v in r.items() if k != "per_problem"}
                       for r in uniform_results]
    with open(PHASE2_DIR / "uniform_sweep.json", "w") as f:
        json.dump(uniform_compact, f, indent=2)

    # ============================================
    # Step 4: Targeted sweep on train-400
    # ============================================
    print(f"\n[4] Targeted sweep with best_t={best_t:.2f}...")
    targeted_results = []

    for tau in TAU_STEER_VALUES:
        checkpoint_path = PHASE2_DIR / f"checkpoint_targeted_tau{tau:.2f}.json"
        if checkpoint_path.exists():
            logger.info("  Loading checkpoint for tau=%.2f", tau)
            with open(checkpoint_path) as f:
                result = json.load(f)
            targeted_results.append(result)
            print(f"  tau={tau:.2f}: net_gain={result['net_gain']}, "
                  f"coverage={result['coverage']:.2f} (from checkpoint)")
            continue

        print(f"\n  --- tau_steer = {tau:.2f} ---")
        t0 = time.time()
        result = run_targeted_sweep_single_tau(
            model, tokenizer, prompts, ground_truths, train_idx, baseline_correct,
            baseline_answers, confidence_scores, steering_vec_gpu, selected_layer,
            best_t, tau
        )
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        targeted_results.append(result)

        print(f"  tau={tau:.2f}: net_gain={result['net_gain']}, "
              f"coverage={result['coverage']:.2f}, "
              f"steered_acc={result['steered_subset_accuracy']:.4f}, "
              f"time={elapsed:.0f}s")

        # Save checkpoint
        checkpoint = {k: v for k, v in result.items() if k != "per_problem"}
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Save targeted sweep (compact)
    targeted_compact = [{k: v for k, v in r.items() if k != "per_problem"}
                        for r in targeted_results]
    with open(PHASE2_DIR / "targeted_sweep.json", "w") as f:
        json.dump(targeted_compact, f, indent=2)

    # ============================================
    # Step 5: Select best tau_steer
    # ============================================
    print("\n[5] Comparing uniform vs targeted...")

    best_targeted = max(targeted_results, key=lambda r: r["net_gain"])
    best_tau = best_targeted["tau_steer"]

    print(f"  Best uniform: t={best_t:.2f}, net_gain={best_uniform['net_gain']}")
    print(f"  Best targeted: tau={best_tau:.2f}, net_gain={best_targeted['net_gain']}, "
          f"coverage={best_targeted['coverage']:.2f}")

    targeted_beats_uniform = best_targeted["net_gain"] > best_uniform["net_gain"]
    if targeted_beats_uniform:
        print("  -> Targeted steering achieves higher net gain than uniform")
    else:
        print("  -> Uniform steering achieves equal or higher net gain than targeted")

    # ============================================
    # Step 6: Lock configuration
    # ============================================
    print("\n[6] Locking configuration...")

    with open(PHASE1_DIR / "sampling_cosines.json") as f:
        cosine_info = json.load(f)

    locked_config = {
        "method": "spherical",
        "layer": selected_layer,
        "layer_probe_auroc": probe_aurocs[str(selected_layer)],
        "best_t": best_t,
        "best_tau_steer": best_tau,
        "vector_type": cosine_info["selected_type"],
        "uniform_net_gain_train": best_uniform["net_gain"],
        "targeted_net_gain_train": best_targeted["net_gain"],
        "targeted_beats_uniform": targeted_beats_uniform,
    }
    with open(PHASE2_DIR / "locked_config.json", "w") as f:
        json.dump(locked_config, f, indent=2)
    print(f"  Config FROZEN: {json.dumps(locked_config, indent=2)}")

    # ============================================
    # Step 7: Comparison table
    # ============================================
    lines = [
        "Phase 2: Steering Calibration Results (Train-400)",
        "=" * 60,
        "",
        "Uniform Sweep:",
        f"{'t':>6} {'Acc':>8} {'Gain':>6} {'Lost':>6} {'Recovered':>10} {'KL':>8}",
        "-" * 50,
    ]
    for r in uniform_results:
        lines.append(
            f"{r['t']:6.2f} {r['overall_accuracy']:8.4f} {r['net_gain']:+6d} "
            f"{r['orig_correct_lost']:6d} {r['orig_incorrect_recovered']:10d} "
            f"{r.get('kl_divergence', 0):8.4f}"
        )
    lines += [
        "",
        "Targeted Sweep (best_t={:.2f}):".format(best_t),
        f"{'tau':>6} {'Cov':>6} {'Acc':>8} {'Gain':>6} {'Steered':>8} {'SteerAcc':>9}",
        "-" * 50,
    ]
    for r in targeted_results:
        lines.append(
            f"{r['tau_steer']:6.2f} {r['coverage']:6.2f} {r['overall_accuracy']:8.4f} "
            f"{r['net_gain']:+6d} {r['n_steered']:8d} "
            f"{r['steered_subset_accuracy']:9.4f}"
        )
    lines += [
        "",
        f"Best uniform: t={best_t:.2f}, net_gain={best_uniform['net_gain']}",
        f"Best targeted: tau={best_tau:.2f}, net_gain={best_targeted['net_gain']}",
        f"Targeted > Uniform: {targeted_beats_uniform}",
        "",
        "LOCKED CONFIG:",
        json.dumps(locked_config, indent=2),
    ]

    comparison_text = "\n".join(lines)
    with open(PHASE2_DIR / "comparison_table.txt", "w") as f:
        f.write(comparison_text)

    print(f"\n{comparison_text}")
    print(f"\n  Artifacts saved to {PHASE2_DIR}/")
    print(f"\n  Next: Run phase3_holdout.py")


if __name__ == "__main__":
    main()
