#!/usr/bin/env python3
"""Phase B1: Full GRPO Training.

Train per-layer bias vectors via GRPO reinforcement learning to maximize
correct math answers. The base model (Qwen2.5-1.5B-Instruct) stays frozen.

Input: MATH-500, baseline_correct.npy, pilot_report.json (validates pipeline)
Output: pathway4/track_b/ — trained_bias.pt, training_log.json,
        checkpoints/step_*.pt

Runtime: ~10 hours GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    BASELINE_CORRECT_PATH,
    CHECKPOINT_DIR,
    GRPO_BATCH_SIZE,
    GRPO_BETA_KL,
    GRPO_GRAD_CLIP,
    GRPO_GROUP_SIZE,
    GRPO_LR,
    GRPO_N_STEPS_FULL,
    GRPO_CHECKPOINT_EVERY,
    GRPO_EVAL_EVERY,
    GRPO_TEMPERATURE,
    GRPO_TOP_P,
    SteeringBias,
    check_correct,
    compute_group_advantages,
    compute_log_prob_with_grad,
    extract_answer,
    generate_one,
    generate_one_temperature,
    load_checkpoint,
    load_model,
    load_training_problems,
    logger,
    register_bias_hooks,
    remove_hooks,
    save_checkpoint,
    tokenize_prompt_completion,
    TRACK_B_DIR,
)


def quick_eval(
    model,
    tokenizer,
    steering_bias: SteeringBias,
    eval_indices: list[int],
    prompts: list[str],
    ground_truths: list[str],
    baseline_correct: np.ndarray,
) -> dict:
    """Quick evaluation on a subset of problems.

    Generates greedy answers with bias hooks active and compares
    against baseline.

    Args:
        model: HF model (hooks should already be registered)
        tokenizer: Tokenizer
        steering_bias: Current bias state
        eval_indices: Global indices to evaluate
        prompts: Full prompt list
        ground_truths: Full ground truth list
        baseline_correct: (500,) baseline correctness

    Returns:
        Dict with net_gain, w2r, r2w, accuracy
    """
    correct_count = 0
    w2r = 0
    r2w = 0

    with torch.no_grad():
        for gi in eval_indices:
            answer_text = generate_one(model, tokenizer, prompts[gi])
            answer = extract_answer(answer_text)
            is_correct = check_correct(answer, ground_truths[gi])
            correct_count += int(is_correct)

            was_correct = bool(baseline_correct[gi])
            if not was_correct and is_correct:
                w2r += 1
            elif was_correct and not is_correct:
                r2w += 1

    return {
        "n_eval": len(eval_indices),
        "correct": correct_count,
        "accuracy": round(correct_count / len(eval_indices), 4),
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net_gain": w2r - r2w,
    }


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Phase B1: Full GRPO Training ===")

    # Verify pilot passed
    pilot_path = TRACK_B_DIR / "pilot_report.json"
    if pilot_path.exists():
        with open(pilot_path) as f:
            pilot = json.load(f)
        if not pilot.get("all_passed", False):
            logger.error("Pilot validations did not pass! Run phase0_pilot.py first.")
            return
        if not pilot.get("pilot_training", {}).get("pilot_ok", False):
            logger.warning("Pilot training had issues. Proceeding with caution.")
        logger.info("Pilot report: validations passed, pilot OK")
    else:
        logger.warning("No pilot report found. Run phase0_pilot.py first for safety.")

    # Load data
    wrong_global, prompts, ground_truths = load_training_problems()
    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    rng = np.random.default_rng(42)

    # Select eval subset (50 random train problems including some correct ones)
    train_idx = np.where(~np.load(
        Path(__file__).parent.parent.parent / "pathway1" / "phase1" / "holdout_mask_seed9999.npy"
    ))[0]
    eval_indices = rng.choice(train_idx, size=50, replace=False).tolist()

    # Load model
    model, tokenizer = load_model()

    # Initialize trainable parameters
    steering_bias = SteeringBias(hidden_dim=1536, layers=[24])
    steering_bias.to(model.device)
    optimizer = torch.optim.Adam(steering_bias.parameters(), lr=GRPO_LR)

    # Check for existing checkpoint to resume from
    start_step = 0
    existing_logs = []
    latest_ckpt = None
    for ckpt_file in sorted(CHECKPOINT_DIR.glob("step_*.pt")):
        latest_ckpt = ckpt_file
    if latest_ckpt is not None:
        logger.info("Resuming from checkpoint: %s", latest_ckpt.name)
        ckpt_data = load_checkpoint(latest_ckpt, steering_bias, optimizer)
        start_step = ckpt_data["step"] + 1
        # Load existing log
        log_path = TRACK_B_DIR / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                existing_logs = json.load(f)
        logger.info("  Resuming from step %d, bias_norm=%.6f", start_step, steering_bias.total_norm())

    # Register hooks
    handles = register_bias_hooks(model, steering_bias)

    # Baseline eval before training
    if start_step == 0:
        logger.info("Baseline evaluation (before training)...")
        baseline_eval = quick_eval(
            model, tokenizer, steering_bias, eval_indices,
            prompts, ground_truths, baseline_correct,
        )
        logger.info("  Baseline: %d/%d correct, net_gain=%d",
                     baseline_eval["correct"], baseline_eval["n_eval"], baseline_eval["net_gain"])

    # ---- Training loop ----
    step_logs = existing_logs.copy()
    n_steps = GRPO_N_STEPS_FULL

    logger.info("Starting GRPO training: steps %d to %d...", start_step, n_steps - 1)

    for step in range(start_step, n_steps):
        t_step = time.time()

        # Sample batch of problems
        batch_indices = rng.choice(wrong_global, size=GRPO_BATCH_SIZE, replace=False).tolist()

        # ---- Phase 1: Generate rollouts (no grad) ----
        all_rollouts = []
        all_rewards = []

        with torch.no_grad():
            for gi in batch_indices:
                rollouts = []
                rewards = []
                for g in range(GRPO_GROUP_SIZE):
                    text = generate_one_temperature(
                        model, tokenizer, prompts[gi],
                        temperature=GRPO_TEMPERATURE, top_p=GRPO_TOP_P,
                    )
                    rollouts.append(text)
                    answer = extract_answer(text)
                    r = 1.0 if check_correct(answer, ground_truths[gi]) else 0.0
                    rewards.append(r)
                all_rollouts.append(rollouts)
                all_rewards.append(rewards)

        # ---- Phase 2: Compute advantages ----
        advantages = compute_group_advantages(all_rewards)

        # ---- Phase 3: Policy gradient + KL penalty ----
        # Per-rollout backward to avoid holding full graph in VRAM
        optimizer.zero_grad()
        total_policy_loss_value = 0.0
        total_kl = 0.0
        n_valid = 0

        for i, gi in enumerate(batch_indices):
            for g in range(GRPO_GROUP_SIZE):
                adv = advantages[i][g]
                if abs(adv) < 1e-8:
                    continue

                full_ids, prompt_len = tokenize_prompt_completion(
                    tokenizer, prompts[gi], all_rollouts[i][g]
                )
                full_ids = full_ids.to(model.device)

                # Log prob (with grad through hooks)
                avg_log_prob, comp_len = compute_log_prob_with_grad(
                    model, full_ids, prompt_len
                )
                if comp_len == 0:
                    continue

                # Policy loss — backward immediately to free graph
                policy_loss = -adv * avg_log_prob
                policy_loss.backward()
                total_policy_loss_value += policy_loss.item()
                n_valid += 1

        if n_valid > 0:
            # Scale accumulated gradients by 1/n_valid
            for p in steering_bias.parameters():
                if p.grad is not None:
                    p.grad.div_(n_valid)

            # KL penalty: bias norm squared as proxy
            # Add gradient manually: d/db [beta * sum(b^2)] = 2*beta*b
            kl_proxy_value = sum(
                b.data.pow(2).sum().item() for b in steering_bias.parameters()
            )
            for p in steering_bias.parameters():
                if p.grad is not None:
                    p.grad.add_(GRPO_BETA_KL * 2.0 * p.data)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                steering_bias.parameters(), GRPO_GRAD_CLIP
            ).item()
            optimizer.step()
            total_policy_loss_value /= n_valid
            total_kl = kl_proxy_value
            total_loss_value = total_policy_loss_value + GRPO_BETA_KL * kl_proxy_value
        else:
            total_loss_value = 0.0
            total_policy_loss_value = 0.0
            grad_norm = 0.0

        # ---- Logging ----
        reward_rate = float(np.mean([r for rewards in all_rewards for r in rewards]))
        bias_norm = steering_bias.total_norm()
        step_time = time.time() - t_step

        log_entry = {
            "step": step,
            "loss": round(total_loss_value, 6),
            "policy_loss": round(total_policy_loss_value, 6),
            "kl_proxy": round(total_kl, 8),
            "reward_rate": round(reward_rate, 4),
            "bias_norm": round(bias_norm, 6),
            "grad_norm": round(grad_norm, 6),
            "n_valid_rollouts": n_valid,
            "step_time_s": round(step_time, 1),
        }
        step_logs.append(log_entry)

        if step % 5 == 0:
            logger.info(
                "  Step %3d: loss=%.4f, reward=%.3f, bias=%.6f, grad=%.6f, "
                "kl=%.8f, time=%.1fs",
                step, log_entry["loss"], reward_rate, bias_norm, grad_norm,
                total_kl, step_time,
            )

        # ---- Checkpoint ----
        if (step + 1) % GRPO_CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"step_{step:04d}.pt"
            save_checkpoint(step, steering_bias, optimizer, log_entry, ckpt_path)
            logger.info("  Checkpoint saved: %s", ckpt_path.name)

            # Save training log
            with open(TRACK_B_DIR / "training_log.json", "w") as f:
                json.dump(step_logs, f, indent=2)

        # ---- Quick eval ----
        if (step + 1) % GRPO_EVAL_EVERY == 0:
            logger.info("  Quick eval at step %d...", step)
            eval_result = quick_eval(
                model, tokenizer, steering_bias, eval_indices,
                prompts, ground_truths, baseline_correct,
            )
            logger.info(
                "  Eval: %d/%d correct, net_gain=%d (W→R=%d, R→W=%d)",
                eval_result["correct"], eval_result["n_eval"],
                eval_result["net_gain"], eval_result["wrong_to_right"],
                eval_result["right_to_wrong"],
            )
            step_logs[-1]["eval"] = eval_result

    # ---- Save final artifacts ----
    remove_hooks(handles)

    # Save trained bias
    bias_path = TRACK_B_DIR / "trained_bias.pt"
    torch.save(steering_bias.state_dict(), bias_path)
    logger.info("Saved trained bias: %s", bias_path)

    # Save training log
    log_path = TRACK_B_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(step_logs, f, indent=2)
    logger.info("Saved training log: %s", log_path)

    # ---- Summary ----
    final_logs = step_logs[-10:]
    mean_reward = np.mean([log["reward_rate"] for log in final_logs])
    mean_loss = np.mean([log["loss"] for log in final_logs])

    summary = {
        "n_steps": n_steps,
        "final_bias_norm": round(steering_bias.total_norm(), 6),
        "final_reward_rate": round(mean_reward, 4),
        "final_loss": round(mean_loss, 6),
        "total_time_s": round(time.time() - t_start, 1),
        "total_time_hours": round((time.time() - t_start) / 3600, 2),
    }
    with open(TRACK_B_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Phase B1 Summary ===")
    logger.info("Steps completed: %d", n_steps)
    logger.info("Final bias norm: %.6f", steering_bias.total_norm())
    logger.info("Final reward rate: %.3f", mean_reward)
    logger.info("Total time: %.1f hours", (time.time() - t_start) / 3600)
    logger.info("Phase B1 complete.")


if __name__ == "__main__":
    main()
