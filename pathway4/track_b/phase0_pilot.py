#!/usr/bin/env python3
"""Phase B0: Pilot Validation.

Quick 50-step pilot to validate:
1. Qwen2DecoderLayer returns plain tensor (not tuple)
2. Bias=zeros produces identical outputs to unhooked model
3. Gradient flows from loss through log_prob through hook to bias
4. Loss is finite, bias norm increases from zero, some rollouts get reward

Input: MATH-500, baseline_correct.npy
Output: pathway4/track_b/pilot_report.json

Runtime: ~2.5 hours GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    CHECKPOINT_DIR,
    GRPO_BATCH_SIZE,
    GRPO_BETA_KL,
    GRPO_GRAD_CLIP,
    GRPO_GROUP_SIZE,
    GRPO_LR,
    GRPO_N_STEPS_PILOT,
    GRPO_TEMPERATURE,
    GRPO_TOP_P,
    MAX_LENGTH,
    SteeringBias,
    check_correct,
    compute_group_advantages,
    compute_log_prob_with_grad,
    extract_answer,
    generate_one_temperature,
    load_model,
    load_training_problems,
    logger,
    make_differentiable_bias_hook,
    register_bias_hooks,
    remove_hooks,
    save_checkpoint,
    tokenize_prompt_completion,
    TRACK_B_DIR,
)


def validate_hook_identity(model, tokenizer, prompt: str) -> bool:
    """Verify that bias=zeros produces identical outputs to unhooked model."""
    logger.info("Validation 1: Hook identity check...")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Unhooked forward
    with torch.no_grad():
        out_base = model(**inputs).logits

    # Hooked forward with zero bias
    bias = torch.nn.Parameter(torch.zeros(1536, dtype=torch.float32, device=model.device))
    hook = make_differentiable_bias_hook(bias)
    handle = model.model.layers[24].register_forward_hook(hook)
    with torch.no_grad():
        out_hooked = model(**inputs).logits
    handle.remove()

    max_diff = (out_base - out_hooked).abs().max().item()
    logger.info("  Max logit difference with zero bias: %.2e", max_diff)
    ok = max_diff < 1e-4
    if ok:
        logger.info("  PASS: Zero bias is identity")
    else:
        logger.warning("  FAIL: Zero bias changes outputs by %.2e", max_diff)
    return ok


def validate_output_type(model, tokenizer, prompt: str) -> bool:
    """Verify Qwen2DecoderLayer returns plain tensor."""
    logger.info("Validation 2: Output type check...")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = []
    def capture_hook(module, input, output):
        outputs.append(output)
        return output

    handle = model.model.layers[24].register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    output = outputs[0]
    is_tensor = isinstance(output, torch.Tensor)
    logger.info("  Layer 24 output type: %s", type(output).__name__)
    if is_tensor:
        logger.info("  PASS: Output is plain tensor, shape=%s", output.shape)
    else:
        logger.warning("  FAIL: Output is %s, not tensor", type(output).__name__)
    return is_tensor


def validate_gradient_flow(model, tokenizer, prompt: str, completion: str) -> bool:
    """Verify gradients flow from log_prob through hook to bias."""
    logger.info("Validation 3: Gradient flow check...")

    steering_bias = SteeringBias(hidden_dim=1536, layers=[24])
    steering_bias.to(model.device)
    handles = register_bias_hooks(model, steering_bias)

    full_ids, prompt_len = tokenize_prompt_completion(tokenizer, prompt, completion)
    full_ids = full_ids.to(model.device)

    avg_log_prob, comp_len = compute_log_prob_with_grad(model, full_ids, prompt_len)

    if comp_len == 0:
        logger.warning("  FAIL: No completion tokens")
        remove_hooks(handles)
        return False

    avg_log_prob.backward()

    bias_param = steering_bias.get_bias(24)
    grad = bias_param.grad
    if grad is None:
        logger.warning("  FAIL: No gradient on bias parameter")
        remove_hooks(handles)
        return False

    grad_norm = grad.norm().item()
    logger.info("  Bias gradient norm: %.6f", grad_norm)
    logger.info("  Log prob: %.4f, comp_len: %d", avg_log_prob.item(), comp_len)

    ok = grad_norm > 0
    if ok:
        logger.info("  PASS: Gradient flows to bias")
    else:
        logger.warning("  FAIL: Zero gradient")

    remove_hooks(handles)
    return ok


def run_pilot_training(
    model, tokenizer, n_steps: int = GRPO_N_STEPS_PILOT,
) -> list[dict]:
    """Run pilot training loop for n_steps."""
    logger.info("Running pilot training: %d steps...", n_steps)

    wrong_global, prompts, ground_truths = load_training_problems()
    rng = np.random.default_rng(42)

    # Initialize trainable parameters
    steering_bias = SteeringBias(hidden_dim=1536, layers=[24])
    steering_bias.to(model.device)
    optimizer = torch.optim.Adam(steering_bias.parameters(), lr=GRPO_LR)

    handles = register_bias_hooks(model, steering_bias)
    step_logs = []

    for step in range(n_steps):
        t_step = time.time()

        # Sample batch
        batch_indices = rng.choice(wrong_global, size=GRPO_BATCH_SIZE, replace=False).tolist()

        # Generate rollouts (no grad)
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

        # Compute advantages
        advantages = compute_group_advantages(all_rewards)

        # Compute policy gradient loss (per-rollout backward to save VRAM)
        optimizer.zero_grad()
        n_valid = 0
        total_loss_value = 0.0

        for i, gi in enumerate(batch_indices):
            for g in range(GRPO_GROUP_SIZE):
                adv = advantages[i][g]
                if abs(adv) < 1e-8:
                    continue  # Skip zero-advantage rollouts

                full_ids, prompt_len = tokenize_prompt_completion(
                    tokenizer, prompts[gi], all_rollouts[i][g]
                )
                full_ids = full_ids.to(model.device)

                avg_log_prob, comp_len = compute_log_prob_with_grad(
                    model, full_ids, prompt_len
                )
                if comp_len == 0:
                    continue

                policy_loss = -adv * avg_log_prob
                policy_loss.backward()  # Accumulate grads, free graph
                total_loss_value += policy_loss.item()
                n_valid += 1

        if n_valid > 0:
            # Scale accumulated gradients by 1/n_valid
            for p in steering_bias.parameters():
                if p.grad is not None:
                    p.grad.div_(n_valid)

            # KL penalty gradient: d/db [beta * sum(b^2)] = 2*beta*b
            for p in steering_bias.parameters():
                if p.grad is not None:
                    p.grad.add_(GRPO_BETA_KL * 2.0 * p.data)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                steering_bias.parameters(), GRPO_GRAD_CLIP
            ).item()
            optimizer.step()
            total_loss_value /= n_valid
        else:
            grad_norm = 0.0

        # Logging
        reward_rate = np.mean([r for rewards in all_rewards for r in rewards])
        bias_norm = steering_bias.total_norm()
        step_time = time.time() - t_step

        log_entry = {
            "step": step,
            "loss": round(total_loss_value, 6),
            "reward_rate": round(reward_rate, 4),
            "bias_norm": round(bias_norm, 6),
            "grad_norm": round(grad_norm, 6),
            "n_valid_rollouts": n_valid,
            "step_time_s": round(step_time, 1),
        }
        step_logs.append(log_entry)

        if step % 5 == 0:
            logger.info(
                "  Step %3d: loss=%.4f, reward=%.3f, bias_norm=%.6f, grad=%.6f, "
                "valid=%d, time=%.1fs",
                step, log_entry["loss"], reward_rate, bias_norm, grad_norm,
                n_valid, step_time,
            )

    remove_hooks(handles)
    return step_logs


def main() -> None:
    TRACK_B_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Phase B0: Pilot Validation ===")

    # Load model
    model, tokenizer = load_model()

    # Load a test problem
    wrong_global, prompts, ground_truths = load_training_problems()
    test_prompt = prompts[wrong_global[0]]
    test_completion = "Let me solve this step by step.\n\nFirst, we need to find the value.\n\nThe answer is #### 42"

    # Run validations
    results = {}
    results["output_type_ok"] = validate_output_type(model, tokenizer, test_prompt)
    results["identity_ok"] = validate_hook_identity(model, tokenizer, test_prompt)
    results["gradient_flow_ok"] = validate_gradient_flow(
        model, tokenizer, test_prompt, test_completion
    )

    all_ok = all(results.values())
    if not all_ok:
        logger.error("VALIDATION FAILED: %s", results)
        logger.error("Aborting pilot training.")
        report = {
            "validations": results,
            "all_passed": False,
            "step_logs": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(TRACK_B_DIR / "pilot_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return

    logger.info("\nAll validations passed. Starting pilot training...")

    # Run pilot
    step_logs = run_pilot_training(model, tokenizer, GRPO_N_STEPS_PILOT)

    # Analyze pilot results
    final_reward_rate = np.mean([log["reward_rate"] for log in step_logs[-10:]])
    final_bias_norm = step_logs[-1]["bias_norm"]
    initial_reward_rate = np.mean([log["reward_rate"] for log in step_logs[:5]])
    reward_improved = final_reward_rate > initial_reward_rate
    bias_grew = final_bias_norm > 1e-6
    losses_finite = all(np.isfinite(log["loss"]) for log in step_logs)

    pilot_ok = losses_finite and bias_grew

    report = {
        "validations": results,
        "all_passed": all_ok,
        "pilot_training": {
            "n_steps": GRPO_N_STEPS_PILOT,
            "initial_reward_rate": round(initial_reward_rate, 4),
            "final_reward_rate": round(final_reward_rate, 4),
            "reward_improved": reward_improved,
            "final_bias_norm": round(final_bias_norm, 6),
            "bias_grew": bias_grew,
            "all_losses_finite": losses_finite,
            "pilot_ok": pilot_ok,
        },
        "step_logs": step_logs,
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = TRACK_B_DIR / "pilot_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("  Saved pilot report: %s", report_path)

    logger.info("\n=== Phase B0 Summary ===")
    logger.info("Validations: %s", "ALL PASSED" if all_ok else "FAILED")
    logger.info("Pilot: %s", "OK" if pilot_ok else "ISSUES DETECTED")
    logger.info("  Initial reward rate: %.3f", initial_reward_rate)
    logger.info("  Final reward rate: %.3f", final_reward_rate)
    logger.info("  Final bias norm: %.6f", final_bias_norm)
    logger.info("  Losses finite: %s", losses_finite)
    logger.info("Total time: %.1f minutes", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
