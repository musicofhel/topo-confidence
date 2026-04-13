"""Shared utilities for Track A activation steering experiment.

Constants, data loading, answer checking, slerp, model loading,
artifact verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("track_a")

# ---- Paths ----

TRACK_A_DIR = Path(__file__).parent
PATHWAY2_DIR = TRACK_A_DIR.parent
TOPO_DIR = PATHWAY2_DIR.parent
PATHWAY1_DIR = TOPO_DIR / "pathway1"

# Pathway 1 artifacts
HOLDOUT_MASK_PATH = PATHWAY1_DIR / "phase1" / "holdout_mask_seed9999.npy"
FEATURES_TRAIN_PATH = PATHWAY1_DIR / "phase1" / "features_train400.npy"
FEATURES_HOLDOUT_PATH = PATHWAY1_DIR / "phase1" / "features_holdout100.npy"
HOLDOUT_PREDICTIONS_PATH = PATHWAY1_DIR / "phase1" / "holdout_predictions.npy"
HOLDOUT_METRICS_PATH = PATHWAY1_DIR / "phase1" / "holdout_metrics.json"
TIER_ASSIGNMENTS_PATH = PATHWAY1_DIR / "phase2" / "tier_assignments.json"
BEST_TIER_PATH = PATHWAY1_DIR / "phase2" / "best_surviving_tier.txt"
ARTIFACT_LOCKS_PATH = PATHWAY1_DIR / "phase0" / "artifact_locks.json"
WINNING_FEATURES_PATH = PATHWAY1_DIR / "phase0" / "winning_features.py"

# Data
TRAJ_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectories.npz"
META_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectory_meta.json"
LAYER_STATES_PATH = (
    Path.home() / "att-docs" / "data" / "transformer" / "math500_hidden_states_aligned.npz"
)

# Phase output directories
PHASE0_DIR = TRACK_A_DIR / "phase0"
PHASE1_DIR = TRACK_A_DIR / "phase1"
PHASE2_DIR = TRACK_A_DIR / "phase2"
PHASE3_DIR = TRACK_A_DIR / "phase3"

# ---- Constants ----

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CANDIDATE_LAYERS = [14, 17, 19, 20, 22, 24]
YOUDEN_THRESHOLD = 0.3203
SEED = 42
MAX_NEW_TOKENS = 256
MAX_LENGTH = 512
N_PROBLEMS = 500

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Give your final answer after '####'.\n\n"
    "Problem: {problem}\n\nSolution:"
)

# LR and CV hyperparameters (matching Pathway 1 exactly)
LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42)
CV_PARAMS_10 = dict(n_splits=10, shuffle=True, random_state=42)
CV_PARAMS_50 = dict(n_splits=50, shuffle=True, random_state=42)


# ---- MATH-500 data loading ----


def load_math500() -> list[dict]:
    """Load MATH-500 from HuggingFace datasets."""
    logger.info("Loading MATH-500 from HuggingFace datasets...")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return list(ds)


def format_prompt(problem_text: str) -> str:
    """Apply the MATH prompt template."""
    return MATH_PROMPT_TEMPLATE.format(problem=problem_text)


def get_prompts_and_ground_truths(
    problems: list[dict],
) -> tuple[list[str], list[str]]:
    """Extract prompts and ground truth answers from MATH-500 problems."""
    prompts = []
    ground_truths = []
    for p in problems:
        text = p.get("problem", p.get("question", ""))
        prompts.append(format_prompt(text))
        answer = p.get("answer", p.get("solution", ""))
        boxed = re.search(r"\\boxed\{(.+?)\}", answer)
        if boxed:
            ground_truths.append(boxed.group(1))
        else:
            ground_truths.append(answer)
    return prompts, ground_truths


# ---- Answer checking (from scripts/experiment1_math500.py) ----


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from model output."""
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        return match.group(1).strip()
    match = re.search(
        r"(?:answer|result)\s+is\s+(.+?)(?:\.|,|\n|$)", text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    answer = answer.replace("$", "").replace(",", "").replace(" ", "")
    answer = answer.rstrip(".")
    try:
        val = float(answer)
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return answer


def check_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    return normalize_answer(predicted) == normalize_answer(ground_truth)


# ---- Train/holdout split ----


def get_train_holdout_indices() -> tuple[np.ndarray, np.ndarray]:
    """Load holdout mask and return (train_idx, holdout_idx)."""
    mask = np.load(HOLDOUT_MASK_PATH)
    assert mask.shape == (N_PROBLEMS,), f"Mask shape {mask.shape} != ({N_PROBLEMS},)"
    assert mask.sum() == 100, f"Holdout count {mask.sum()} != 100"
    train_idx = np.where(~mask)[0]
    holdout_idx = np.where(mask)[0]
    assert len(train_idx) == 400
    assert len(holdout_idx) == 100
    return train_idx, holdout_idx


# ---- Feature tier filtering ----


def get_abc_column_indices(feature_names: list[str]) -> list[int]:
    """Return indices of A+B+C tier features from the 78-column output.

    Args:
        feature_names: list of feature names from extract_features output
            (after drop_cols, so this is the ~78-element list).

    Returns:
        Sorted list of column indices where tier is A, B, or C.
    """
    with open(TIER_ASSIGNMENTS_PATH) as f:
        tier_map = json.load(f)
    abc_tiers = {"A", "B", "C"}
    indices = []
    for i, name in enumerate(feature_names):
        if name in tier_map and tier_map[name] in abc_tiers:
            indices.append(i)
    assert len(indices) == 44, (
        f"Expected 44 A+B+C features, got {len(indices)}. "
        f"Missing: {set(n for n, t in tier_map.items() if t in abc_tiers) - set(feature_names[i] for i in indices)}"
    )
    return sorted(indices)


# ---- Artifact verification ----


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pathway1_artifacts() -> None:
    """Verify all Pathway 1 artifacts exist and hashes match. Halt on failure."""
    required_paths = [
        HOLDOUT_MASK_PATH,
        FEATURES_TRAIN_PATH,
        FEATURES_HOLDOUT_PATH,
        HOLDOUT_PREDICTIONS_PATH,
        HOLDOUT_METRICS_PATH,
        TIER_ASSIGNMENTS_PATH,
        BEST_TIER_PATH,
        ARTIFACT_LOCKS_PATH,
        WINNING_FEATURES_PATH,
        TRAJ_PATH,
        META_PATH,
    ]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing Pathway 1 artifact: {p}")

    # Verify SHA-256 hashes
    with open(ARTIFACT_LOCKS_PATH) as f:
        locks = json.load(f)

    hash_checks = [
        ("winning_features_sha256", WINNING_FEATURES_PATH),
        ("trajectories_sha256", TRAJ_PATH),
        ("trajectory_meta_sha256", META_PATH),
    ]
    for key, path in hash_checks:
        expected = locks.get(key)
        if expected is None:
            logger.warning("No hash for %s in artifact_locks.json", key)
            continue
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {path.name}: expected {expected[:16]}..., "
                f"got {actual[:16]}..."
            )
        logger.info("  %s: hash OK", path.name)

    # Verify holdout metrics
    with open(HOLDOUT_METRICS_PATH) as f:
        metrics = json.load(f)
    assert abs(metrics["youden_threshold_from_train"] - YOUDEN_THRESHOLD) < 1e-4, (
        f"Youden threshold mismatch: {metrics['youden_threshold_from_train']} != {YOUDEN_THRESHOLD}"
    )

    logger.info("All Pathway 1 artifacts verified.")


# ---- Model loading ----


def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-1.5B-Instruct in fp16 on CUDA."""
    logger.info("Loading %s...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded on %s", next(model.parameters()).device)
    return model, tokenizer


# ---- Generation ----


@torch.no_grad()
def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a single answer with greedy decoding."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_one_with_scores(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, torch.Tensor]:
    """Generate with greedy decoding, also return output logits for KL computation."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    generated_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if outputs.scores:
        logits = torch.stack(outputs.scores, dim=0)  # (n_tokens, vocab_size)
    else:
        logits = torch.empty(0)
    return text, logits


# ---- Spherical linear interpolation ----


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between v0 and v1.

    Args:
        v0: source tensor, any shape (..., dim)
        v1: target tensor, same shape as v0
        t: interpolation factor in [0, 1]

    Returns:
        Interpolated tensor, same shape, unit-normalized.
    """
    v0_norm = F.normalize(v0, dim=-1)
    v1_norm = F.normalize(v1, dim=-1)
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # Near-parallel fallback: lerp when sin(omega) is tiny
    if sin_omega.abs().min() < 1e-6:
        return (1 - t) * v0_norm + t * v1_norm
    s0 = torch.sin((1 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega
    return s0 * v0_norm + s1 * v1_norm


def make_steering_hook(
    steering_vec_gpu: torch.Tensor, t: float
) -> callable:
    """Create a forward hook that applies spherical steering at strength t.

    The hook:
    1. Extracts hidden states from the layer output
    2. Saves original norms
    3. Applies slerp rotation toward steering vector
    4. Restores original norms (preserving magnitude)
    5. Returns a NEW tuple (critical: do not modify in-place)

    Args:
        steering_vec_gpu: unit-normalized steering vector on GPU, shape (1536,)
        t: rotation strength in [0, 1]. t=0 is identity.

    Returns:
        Hook function compatible with register_forward_hook.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        norms = h.norm(dim=-1, keepdim=True)
        # Expand steering vector to match h's shape (handle 2D and 3D)
        sv = steering_vec_gpu
        for _ in range(h.dim() - sv.dim()):
            sv = sv.unsqueeze(0)
        sv = sv.expand_as(h)
        rotated = slerp(h, sv, t)
        modified = rotated * norms  # restore original magnitudes
        if isinstance(output, tuple):
            return (modified,) + tuple(output[i] for i in range(1, len(output)))
        else:
            return modified

    return hook


# ---- Data loading helpers ----


def load_trajectory_meta() -> dict:
    """Load trajectory metadata (correct labels, ground truths, etc.)."""
    with open(META_PATH) as f:
        return json.load(f)


def load_cached_trajectories() -> list[np.ndarray]:
    """Load cached token trajectories from experiment1_v2."""
    t = np.load(TRAJ_PATH)
    return [t[f"traj_{i}"] for i in range(N_PROBLEMS)]
