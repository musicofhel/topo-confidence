"""Shared utilities for Pathway 4 Track B: Learned Steering via GRPO.

Differentiable bias hooks, log probability computation, KL penalty,
GRPO loss, and training hyperparameters.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Pathway 3 common (which re-exports Track A common)
import importlib.util

PATHWAY3_DIR = Path(__file__).parent.parent.parent / "pathway3"
_spec = importlib.util.spec_from_file_location("p3_common", PATHWAY3_DIR / "common.py")
_p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p3)

# Re-export everything we need
MODEL_NAME = _p3.MODEL_NAME
N_PROBLEMS = _p3.N_PROBLEMS
MAX_NEW_TOKENS = _p3.MAX_NEW_TOKENS
MAX_LENGTH = _p3.MAX_LENGTH
SEED = _p3.SEED
YOUDEN_THRESHOLD = _p3.YOUDEN_THRESHOLD
LR_PARAMS = _p3.LR_PARAMS
load_math500 = _p3.load_math500
format_prompt = _p3.format_prompt
get_prompts_and_ground_truths = _p3.get_prompts_and_ground_truths
extract_answer = _p3.extract_answer
normalize_answer = _p3.normalize_answer
check_correct = _p3.check_correct
get_train_holdout_indices = _p3.get_train_holdout_indices
load_model = _p3.load_model
generate_one = _p3.generate_one
generate_one_temperature = _p3.generate_one_temperature
compute_flip_metrics = _p3.compute_flip_metrics
mcnemar_mid_p = _p3.mcnemar_mid_p
bayesian_p_improvement = _p3.bayesian_p_improvement
bootstrap_ci_net_gain = _p3.bootstrap_ci_net_gain
fit_confidence_pipeline = _p3.fit_confidence_pipeline
predict_confidence = _p3.predict_confidence

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("pathway4_track_b")

# ---- Paths ----

PATHWAY4_DIR = Path(__file__).parent.parent
TRACK_B_DIR = PATHWAY4_DIR / "track_b"
CHECKPOINT_DIR = TRACK_B_DIR / "checkpoints"

BASELINE_CORRECT_PATH = (
    Path(__file__).parent.parent.parent / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
)
CONFIDENCE_SCORES_PATH = (
    Path(__file__).parent.parent.parent / "pathway2" / "track_a" / "phase0" / "confidence_scores.npy"
)

# ---- GRPO Hyperparameters ----

GRPO_BATCH_SIZE = 4          # Problems per step
GRPO_GROUP_SIZE = 8          # Rollouts per problem
GRPO_LR = 5e-4
GRPO_BETA_KL = 0.01         # KL penalty weight
GRPO_GRAD_CLIP = 1.0
GRPO_TEMPERATURE = 0.8
GRPO_TOP_P = 0.95
GRPO_INITIAL_LAYERS = [24]  # Start with layer 24 only
GRPO_N_STEPS_PILOT = 50
GRPO_N_STEPS_FULL = 200
GRPO_CHECKPOINT_EVERY = 25
GRPO_EVAL_EVERY = 50        # Quick eval on random subset

# Track A locked config for comparison
TRACK_A_BEST_T = 0.15
TRACK_A_BEST_TAU = 0.40
TRACK_A_LAYER = 24


# ---- Differentiable Steering Hook ----


class SteeringBias(nn.Module):
    """Learnable per-layer bias vectors for activation steering.

    Each bias is initialized to zeros (identity at start).
    During training, GRPO gradients update these biases to improve
    task performance (correct math answers).
    """

    def __init__(self, hidden_dim: int = 1536, layers: list[int] | None = None):
        super().__init__()
        if layers is None:
            layers = GRPO_INITIAL_LAYERS
        self.layers = layers
        self.biases = nn.ParameterDict({
            str(l): nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
            for l in layers
        })

    def get_bias(self, layer_idx: int) -> nn.Parameter:
        return self.biases[str(layer_idx)]

    def total_norm(self) -> float:
        """L2 norm across all bias vectors."""
        return float(sum(
            b.data.norm().item() ** 2 for b in self.biases.values()
        ) ** 0.5)


def make_differentiable_bias_hook(bias_param: nn.Parameter):
    """Create a forward hook that adds a learnable bias to hidden states.

    Fully differentiable: d(h + b)/d(b) = 1.
    At initialization (bias=zeros), this is identity.

    Qwen2DecoderLayer.forward() returns a plain torch.Tensor (not a tuple)
    in transformers 4.57.6.

    Args:
        bias_param: nn.Parameter of shape (hidden_dim,), requires_grad=True

    Returns:
        Hook function for register_forward_hook.
    """
    def hook(module, input, output):
        # output shape: (batch, seq_len, hidden_dim)
        return output + bias_param.to(output.dtype)
    return hook


def register_bias_hooks(
    model, steering_bias: SteeringBias,
) -> list[torch.utils.hooks.RemovableHook]:
    """Register differentiable bias hooks at all target layers.

    Args:
        model: HuggingFace model
        steering_bias: SteeringBias module with per-layer parameters

    Returns:
        List of hook handles for cleanup
    """
    handles = []
    for layer_idx in steering_bias.layers:
        bias_param = steering_bias.get_bias(layer_idx)
        hook = make_differentiable_bias_hook(bias_param)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)
    return handles


def remove_hooks(handles: list) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    handles.clear()


# ---- Log Probability Computation ----


def compute_log_prob_with_grad(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, int]:
    """Compute average per-token log probability of completion tokens.

    This runs a forward pass with autograd enabled. Hooks on the model
    participate in the computation graph, allowing gradients to flow
    back to bias parameters.

    Args:
        model: HF model with steering hooks registered
        full_ids: (1, seq_len) token ids [prompt + completion]
        prompt_len: Number of prompt tokens

    Returns:
        avg_log_prob: Scalar tensor with grad
        comp_len: Number of completion tokens
    """
    comp_len = full_ids.shape[1] - prompt_len
    if comp_len <= 0:
        return torch.tensor(0.0, device=full_ids.device, requires_grad=True), 0

    # Forward pass (hooks fire here, bias enters computation graph)
    logits = model(input_ids=full_ids).logits  # (1, seq_len, vocab_size)

    # Extract logits that predict completion tokens
    # logits[:, prompt_len-1] predicts the token at position prompt_len
    shift_logits = logits[:, prompt_len - 1 : full_ids.shape[1] - 1, :]  # (1, comp_len, vocab)
    shift_labels = full_ids[:, prompt_len:]  # (1, comp_len)

    # Cast to float32 for numerical stability in log_softmax
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, comp_len)

    # Average per-token log prob
    avg_log_prob = token_log_probs.sum() / comp_len

    return avg_log_prob, comp_len


# ---- KL Penalty ----


def compute_kl_penalty(
    model,
    steering_bias: SteeringBias,
    handles: list,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute KL(steered || base) using efficient per-token approximation.

    Instead of materializing the full vocab KL (expensive for 151K vocab),
    we use the per-token log-prob approximation:
        KL ≈ mean(log_p_steered - log_p_base) over completion tokens

    The steered forward pass must be done with grad (hooks active).
    The base forward pass is done without grad (hooks removed).

    Args:
        model: HF model
        steering_bias: SteeringBias module
        handles: Current hook handles (will be temporarily removed)
        full_ids: (1, seq_len) token ids
        prompt_len: Number of prompt tokens

    Returns:
        kl: Scalar tensor (has grad through steered log probs)
    """
    comp_len = full_ids.shape[1] - prompt_len
    if comp_len <= 0:
        return torch.tensor(0.0, device=full_ids.device, requires_grad=True)

    shift_labels = full_ids[:, prompt_len:]  # (1, comp_len)

    # Steered forward pass (with grad)
    logits_steered = model(input_ids=full_ids).logits
    steered_shift = logits_steered[:, prompt_len - 1 : full_ids.shape[1] - 1, :]
    log_p_steered = F.log_softmax(steered_shift.float(), dim=-1)
    steered_token_lp = log_p_steered.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Base forward pass (no grad, remove hooks temporarily)
    remove_hooks(handles)
    with torch.no_grad():
        logits_base = model(input_ids=full_ids).logits
    base_shift = logits_base[:, prompt_len - 1 : full_ids.shape[1] - 1, :]
    log_p_base = F.log_softmax(base_shift.float(), dim=-1)
    base_token_lp = log_p_base.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Re-register hooks
    new_handles = register_bias_hooks(model, steering_bias)
    handles.extend(new_handles)

    # KL approximation: mean(log_p_steered - log_p_base)
    kl = (steered_token_lp - base_token_lp.detach()).mean()

    return kl


# ---- GRPO Advantage Computation ----


def compute_group_advantages(
    rewards: list[list[float]],
    eps: float = 1e-8,
) -> list[list[float]]:
    """Compute group-relative advantages for GRPO.

    For each problem's group of rollouts:
        advantage = (reward - mean(rewards)) / (std(rewards) + eps)

    Args:
        rewards: List of lists, [batch_size][group_size]
        eps: Small constant for numerical stability

    Returns:
        Advantages with same shape as rewards.
    """
    advantages = []
    for group_rewards in rewards:
        r = np.array(group_rewards, dtype=np.float64)
        mean_r = r.mean()
        std_r = r.std() + eps
        advs = ((r - mean_r) / std_r).tolist()
        advantages.append(advs)
    return advantages


# ---- Tokenization ----


def tokenize_prompt_completion(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int = MAX_LENGTH,
) -> tuple[torch.Tensor, int]:
    """Tokenize prompt + completion, returning full_ids and prompt_len.

    Tokenizes the prompt alone first to get the exact boundary,
    then tokenizes the full concatenation.

    Args:
        tokenizer: HF tokenizer
        prompt: Full formatted prompt string
        completion: Generated completion text
        max_length: Max sequence length

    Returns:
        full_ids: (1, seq_len) tensor on CPU
        prompt_len: Number of prompt tokens
    """
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).input_ids
    prompt_len = prompt_ids.shape[1]

    full_text = prompt + completion
    full_ids = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length,
    ).input_ids

    return full_ids, prompt_len


# ---- Checkpointing ----


def save_checkpoint(
    step: int,
    steering_bias: SteeringBias,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    path: Path,
) -> None:
    """Save training checkpoint."""
    torch.save({
        "step": step,
        "steering_bias_state_dict": steering_bias.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def load_checkpoint(
    path: Path,
    steering_bias: SteeringBias,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load training checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    steering_bias.load_state_dict(ckpt["steering_bias_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---- Training data loading ----


def load_training_problems() -> tuple[list[int], list[str], list[str]]:
    """Load wrong-under-greedy train problems for GRPO training.

    Returns:
        wrong_global_indices: List of global indices (354 problems)
        prompts: Full list of 500 formatted prompts
        ground_truths: Full list of 500 ground truth answers
    """
    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    train_idx, _ = get_train_holdout_indices()
    y_train = baseline_correct[train_idx].astype(bool)

    wrong_local = np.where(~y_train)[0]
    wrong_global = train_idx[wrong_local].tolist()

    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)

    logger.info("Loaded %d wrong-under-greedy train problems", len(wrong_global))
    return wrong_global, prompts, ground_truths
