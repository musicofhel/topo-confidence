"""Shared utilities for Pathway 5: Cross-Benchmark and Cross-Model Generalization.

Extends Pathway 4 Track A common with:
- Parameterized model loading (1.5B / 7B)
- GSM8K data loading and answer checking
- Variable hidden_dim trajectory extraction
- Train/test splitting for GSM8K
- Reusable evaluation and comparison helpers
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as OrigPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Import Pathway 4 Track A common (which chains through P3 → P2 common)
import importlib.util

PATHWAY4_TRACK_A_DIR = Path(__file__).parent.parent / "pathway4" / "track_a"
_spec = importlib.util.spec_from_file_location(
    "p4_common", PATHWAY4_TRACK_A_DIR / "common.py"
)
_p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p4)

# Re-export Pathway 4 Track A symbols
MODEL_NAME_1_5B = _p4.MODEL_NAME  # "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME_7B = "Qwen/Qwen2.5-7B-Instruct"
N_PROBLEMS_MATH = _p4.N_PROBLEMS  # 500
MAX_NEW_TOKENS = _p4.MAX_NEW_TOKENS  # 256
MAX_LENGTH = _p4.MAX_LENGTH  # 512
SEED = _p4.SEED  # 42
LR_PARAMS = _p4.LR_PARAMS
MATH_PROMPT_TEMPLATE = _p4.MATH_PROMPT_TEMPLATE
HOLDOUT_MASK_PATH = _p4.HOLDOUT_MASK_PATH
FEATURES_TRAIN_PATH = _p4.FEATURES_TRAIN_PATH
FEATURES_HOLDOUT_PATH = _p4.FEATURES_HOLDOUT_PATH
TIER_ASSIGNMENTS_PATH = _p4.TIER_ASSIGNMENTS_PATH
ARTIFACT_LOCKS_PATH = _p4.ARTIFACT_LOCKS_PATH
WINNING_FEATURES_PATH = _p4.WINNING_FEATURES_PATH
TRAJ_PATH = _p4.TRAJ_PATH
META_PATH = _p4.META_PATH
load_math500 = _p4.load_math500
format_prompt = _p4.format_prompt
get_prompts_and_ground_truths = _p4.get_prompts_and_ground_truths
extract_answer = _p4.extract_answer
normalize_answer = _p4.normalize_answer
check_correct = _p4.check_correct
get_train_holdout_indices = _p4.get_train_holdout_indices
get_abc_column_indices = _p4.get_abc_column_indices
verify_pathway1_artifacts = _p4.verify_pathway1_artifacts
load_cached_trajectories = _p4.load_cached_trajectories
load_trajectory_meta = _p4.load_trajectory_meta
compute_flip_metrics = _p4.compute_flip_metrics
BASELINE_CORRECT_PATH = _p4.BASELINE_CORRECT_PATH
mcnemar_mid_p = _p4.mcnemar_mid_p
bayesian_p_improvement = _p4.bayesian_p_improvement
bootstrap_ci_net_gain = _p4.bootstrap_ci_net_gain

# Selection strategies
select_random = _p4.select_random
select_max_confidence = _p4.select_max_confidence
select_majority_vote = _p4.select_majority_vote
select_confidence_weighted_vote = _p4.select_confidence_weighted_vote

# Per-completion scoring
fit_completion_scorer = _p4.fit_completion_scorer
score_completions = _p4.score_completions
extract_features_with_prefitted_pca = _p4.extract_features_with_prefitted_pca
fit_greedy_pca = _p4.fit_greedy_pca

# Pathway 1 Phase 0 (winning_features.py)
PATHWAY1_PHASE0_DIR = _p4.PATHWAY1_PHASE0_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("pathway5")

# ---- Paths ----

PATHWAY5_DIR = Path(__file__).parent
TOPO_DIR = PATHWAY5_DIR.parent
TRACK_A_DIR = PATHWAY5_DIR / "track_a"
TRACK_B_DIR = PATHWAY5_DIR / "track_b"
TRACK_C_DIR = PATHWAY5_DIR / "track_c"

# ---- Constants ----

COMPLETIONS_PER_PROBLEM = 32
TEMP_SAMPLING = dict(temperature=0.8, top_p=0.95)
GSM8K_SPLIT_SEED = 9999  # Match MATH-500 holdout seed for consistency
GSM8K_TEST_FRACTION = 0.20  # 80/20 train/test

GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem. Give your final answer after '####'.\n\n"
    "Problem: {problem}\n\nSolution:"
)

# Hidden dimensions per model
HIDDEN_DIM = {
    MODEL_NAME_1_5B: 1536,
    MODEL_NAME_7B: 3584,
}
N_LAYERS = {
    MODEL_NAME_1_5B: 28,  # + 1 embedding = 29 hidden_states entries
    MODEL_NAME_7B: 28,    # same architecture family
}


# ---- GSM8K data loading ----


def load_gsm8k() -> list[dict]:
    """Load GSM8K test set from HuggingFace datasets."""
    logger.info("Loading GSM8K test set from HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return list(ds)


def format_gsm8k_prompt(question: str) -> str:
    """Apply the GSM8K prompt template."""
    return GSM8K_PROMPT_TEMPLATE.format(problem=question)


def get_gsm8k_prompts_and_ground_truths(
    problems: list[dict],
) -> tuple[list[str], list[str]]:
    """Extract prompts and ground truth answers from GSM8K problems.

    GSM8K answer format: reasoning chain ending with '#### [number]'.
    We extract the number after ####.
    """
    prompts = []
    ground_truths = []
    for p in problems:
        prompts.append(format_gsm8k_prompt(p["question"]))
        # Extract ground truth: number after ####
        answer_text = p["answer"]
        gt = extract_gsm8k_ground_truth(answer_text)
        ground_truths.append(gt)
    return prompts, ground_truths


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K answer field.

    GSM8K answers end with '#### [number]'. The number may have commas.
    """
    import re
    match = re.search(r"####\s*(.+?)(?:\n|$)", answer_text)
    if match:
        ans = match.group(1).strip()
        # Remove commas from numbers (GSM8K uses "1,234" format)
        ans = ans.replace(",", "")
        return ans
    # Fallback: last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


def get_gsm8k_train_test_indices(
    n_problems: int,
    y: np.ndarray,
    test_fraction: float = GSM8K_TEST_FRACTION,
    seed: int = GSM8K_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 split for GSM8K.

    Returns (train_idx, test_idx) as sorted arrays.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    train_idx, test_idx = next(sss.split(np.arange(n_problems), y))
    return np.sort(train_idx), np.sort(test_idx)


# ---- Parameterized model loading ----


def load_model(model_name: str = MODEL_NAME_1_5B) -> tuple:
    """Load a Qwen2.5 model in fp16/bf16 on CUDA.

    Args:
        model_name: HuggingFace model name (1.5B or 7B)

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bf16 for 7B (better numerical stability at scale), fp16 for 1.5B
    dtype = torch.bfloat16 if "7B" in model_name else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(
        "Model loaded: %s, dtype=%s, device=%s",
        model_name, dtype, next(model.parameters()).device,
    )
    return model, tokenizer


# ---- Generation ----


@torch.no_grad()
def generate_greedy(
    model, tokenizer, prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_length: int = MAX_LENGTH,
) -> str:
    """Generate a single answer with greedy decoding."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_greedy_with_logprobs(
    model, tokenizer, prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_length: int = MAX_LENGTH,
) -> tuple[str, dict]:
    """Generate with greedy decoding and extract token-level logprob features.

    Returns:
        (generated_text, entropy_features_dict)
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    generated_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute token-level entropy features
    if outputs.scores:
        logits = torch.stack(outputs.scores, dim=0)  # (n_tokens, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        # Per-token entropy: H(p) = -sum(p * log(p))
        entropies = -(probs * log_probs).sum(dim=-1)  # (n_tokens,)
        max_probs = probs.max(dim=-1).values  # (n_tokens,)

        entropy_features = {
            "entropy": float(entropies.mean().cpu()),
            "max_token_prob": float(max_probs.mean().cpu()),
            "first_token_prob": float(max_probs[0].cpu()) if len(max_probs) > 0 else 0.0,
            "n_tokens": int(len(entropies)),
        }
    else:
        entropy_features = {
            "entropy": 0.0, "max_token_prob": 0.0,
            "first_token_prob": 0.0, "n_tokens": 0,
        }

    return text, entropy_features


@torch.no_grad()
def generate_temperature(
    model, tokenizer, prompt: str,
    temperature: float = 0.8, top_p: float = 0.95,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_length: int = MAX_LENGTH,
) -> str:
    """Generate a single completion with temperature sampling."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
    )
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---- Trajectory extraction (variable hidden_dim) ----


@torch.no_grad()
def extract_trajectory(
    model, tokenizer, prompt: str,
    max_length: int = MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass on prompt, extract per-token trajectory and layer states.

    Works for any hidden_dim (1.5B: 1536, 7B: 3584).

    Returns:
        trajectory: (n_tokens, hidden_dim) last-layer hidden states
        layer_states: (n_layers+1, hidden_dim) hidden states at last token across all layers
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs["attention_mask"].sum().item()
    last_pos = seq_len - 1

    # Last-layer trajectory: all token positions
    last_layer_hs = outputs.hidden_states[-1]
    trajectory = last_layer_hs[0, :seq_len].cpu().float().numpy()

    # Layer states at last token
    n_layers = len(outputs.hidden_states)
    hidden_dim = outputs.hidden_states[0].shape[-1]
    layer_states = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    for k in range(n_layers):
        layer_states[k] = outputs.hidden_states[k][0, last_pos].cpu().float().numpy()

    return trajectory, layer_states


@torch.no_grad()
def extract_completion_trajectory(
    model, tokenizer, prompt: str, completion_text: str,
    max_length: int = MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass on prompt+completion, extract trajectory and layer states.

    Returns:
        trajectory: (n_tokens, hidden_dim) last-layer hidden states
        layer_states: (n_layers+1, hidden_dim) hidden states at last token
    """
    full_text = prompt + completion_text
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs["attention_mask"].sum().item()
    last_pos = seq_len - 1

    last_layer_hs = outputs.hidden_states[-1]
    trajectory = last_layer_hs[0, :seq_len].cpu().float().numpy()

    n_layers = len(outputs.hidden_states)
    hidden_dim = outputs.hidden_states[0].shape[-1]
    layer_states = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    for k in range(n_layers):
        layer_states[k] = outputs.hidden_states[k][0, last_pos].cpu().float().numpy()

    return trajectory, layer_states


# ---- Temperature generation with checkpointing ----


def generate_temperature_completions(
    model, tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    problem_indices: np.ndarray,
    output_dir: Path,
    n_completions: int = COMPLETIONS_PER_PROBLEM,
    checkpoint_every: int = 50,
) -> dict:
    """Generate multiple temperature completions per problem with checkpointing.

    Args:
        model: HF model
        tokenizer: Tokenizer
        prompts: All prompts (indexed by global problem index)
        ground_truths: All ground truths
        problem_indices: Which problems to generate for
        output_dir: Directory for checkpoint and final output
        n_completions: Completions per problem
        checkpoint_every: Save checkpoint every N problems

    Returns:
        generations dict: {str(idx): {completions, correct, n_correct}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "temperature_generations_checkpoint.json"
    generations = {}

    # Resume from checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            generations = json.load(f)
        logger.info("Resumed from checkpoint: %d problems done", len(generations))

    total = len(problem_indices)
    t0 = time.time()

    for count, idx in enumerate(problem_indices):
        idx_str = str(int(idx))
        if idx_str in generations:
            continue

        if count % 10 == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / max(elapsed, 1)
            eta = (total - count) / max(rate, 0.001)
            logger.info(
                "Problem %d/%d (idx=%d) | %.1f problems/s | ETA: %.0f min",
                count, total, idx, rate, eta / 60,
            )

        completions = []
        correct_flags = []

        for c in range(n_completions):
            text = generate_temperature(model, tokenizer, prompts[idx])
            predicted = extract_answer(text)
            is_correct = check_correct(predicted, ground_truths[idx])
            completions.append(text[:500])  # truncate for storage
            correct_flags.append(is_correct)

        generations[idx_str] = {
            "completions": completions,
            "correct": correct_flags,
            "n_correct": sum(correct_flags),
        }

        # Checkpoint
        if (count + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(generations, f)
            logger.info("  Checkpoint saved (%d problems)", count + 1)

    # Final save
    final_path = output_dir / "temperature_generations.json"
    with open(final_path, "w") as f:
        json.dump(generations, f)
    logger.info("Saved %d problem generations to %s", len(generations), final_path)

    return generations


# ---- Feature extraction helpers ----


def fit_pca_on_trajectories(trajectories: list[np.ndarray], n_components: int = 45) -> OrigPCA:
    """Fit PCA on concatenated trajectories for a stable subspace.

    Works for any hidden_dim.
    """
    all_points = np.concatenate(trajectories, axis=0)
    n_comp = min(n_components, all_points.shape[1], all_points.shape[0])
    pca = OrigPCA(n_components=n_comp, svd_solver="full")
    pca.fit(all_points)
    logger.info(
        "Fitted PCA: %d components, %.1f%% variance explained",
        n_comp, 100 * pca.explained_variance_ratio_.sum(),
    )
    return pca


def extract_topo_features(
    trajectories: list[np.ndarray],
    layer_states: np.ndarray,
    prefitted_pca: OrigPCA | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract topological features using winning_features.py.

    If prefitted_pca is provided, uses it (no data leakage).
    Otherwise, fits PCA on the trajectories (for standalone use).

    Returns:
        features: (N, n_features) array
        feature_names: list of feature name strings
    """
    if prefitted_pca is not None:
        return extract_features_with_prefitted_pca(
            trajectories, layer_states, prefitted_pca,
        )
    else:
        # Import winning_features directly and call extract_features
        sys.path.insert(0, str(PATHWAY1_PHASE0_DIR))
        from winning_features import extract_features
        return extract_features(trajectories, layer_states)


# ---- Evaluation helpers ----


def bootstrap_auroc(
    y_true: np.ndarray, scores: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """Compute AUROC with bootstrap 95% CI. Returns (auroc, ci_lo, ci_hi)."""
    rng = np.random.RandomState(seed)
    auroc = roc_auc_score(y_true, scores)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], scores[idx]))
    boot_aurocs = np.array(boot_aurocs)
    return float(auroc), float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))


def auroc_permutation_test(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
    n_perm: int = 10000, seed: int = 42,
) -> float:
    """Bootstrap permutation test for AUROC difference. Returns p-value."""
    rng = np.random.RandomState(seed)
    observed_diff = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    count = 0
    for _ in range(n_perm):
        mask = rng.rand(len(y_true)) < 0.5
        mixed_a = np.where(mask, scores_a, scores_b)
        mixed_b = np.where(mask, scores_b, scores_a)
        if len(np.unique(y_true)) < 2:
            continue
        diff = roc_auc_score(y_true, mixed_a) - roc_auc_score(y_true, mixed_b)
        if abs(diff) >= abs(observed_diff):
            count += 1
    return count / n_perm


def evaluate_all_strategies(
    completions_by_problem: dict,
    scores_by_problem: dict,
    ground_truths: list[str],
    problem_indices: np.ndarray,
    greedy_correct: np.ndarray,
    n_random_trials: int = 1000,
) -> dict:
    """Evaluate all selection strategies on a set of problems.

    Args:
        completions_by_problem: {str(idx): list of completion texts}
        scores_by_problem: {str(idx): list of float scores}
        ground_truths: All ground truths (indexed by global index)
        problem_indices: Which problems to evaluate
        greedy_correct: (N_total,) boolean greedy correctness
        n_random_trials: Number of random trials for random baseline

    Returns:
        dict with per-strategy results
    """
    rng = np.random.default_rng(SEED)
    strategies = {}

    # Count greedy-correct in the evaluation set
    n_greedy_correct = int(greedy_correct[problem_indices].sum())

    for strategy_name in ["random", "max_confidence", "majority_vote", "confidence_weighted_vote"]:
        correct_count = n_greedy_correct  # greedy-correct always counted
        w_to_r = 0
        r_to_w = 0

        for idx in problem_indices:
            idx_str = str(int(idx))
            if greedy_correct[idx]:
                continue  # Already counted
            if idx_str not in completions_by_problem:
                continue

            comps = completions_by_problem[idx_str]
            gt = ground_truths[idx]

            if strategy_name == "random":
                # Average over multiple random trials
                n_correct_trials = 0
                for _ in range(n_random_trials):
                    _, is_correct = select_random(
                        np.zeros(len(comps)), comps, gt, rng,
                    )
                    if is_correct:
                        n_correct_trials += 1
                # Use expected value
                expected_correct = n_correct_trials / n_random_trials
                correct_count += expected_correct
                continue

            if strategy_name == "max_confidence":
                scores = np.array(scores_by_problem.get(idx_str, [0.0] * len(comps)))
                _, is_correct = select_max_confidence(scores, comps, gt)
            elif strategy_name == "majority_vote":
                _, is_correct = select_majority_vote(comps, gt)
            elif strategy_name == "confidence_weighted_vote":
                scores = np.array(scores_by_problem.get(idx_str, [0.0] * len(comps)))
                _, is_correct = select_confidence_weighted_vote(scores, comps, gt)
            else:
                continue

            if is_correct:
                correct_count += 1
                w_to_r += 1

        strategies[strategy_name] = {
            "correct": correct_count if strategy_name != "random" else round(correct_count, 1),
            "net_gain": correct_count - n_greedy_correct if strategy_name != "random"
                else round(correct_count - n_greedy_correct, 1),
            "w_to_r": w_to_r,
            "r_to_w": r_to_w,
        }

    return strategies


def evaluate_gating_strategies(
    topo_scores: np.ndarray,
    completions_by_problem: dict,
    ground_truths: list[str],
    problem_indices: np.ndarray,
    greedy_correct: np.ndarray,
    tau_values: list[float] = None,
) -> list[dict]:
    """Evaluate binary gating: trust greedy if confident, else MV.

    Args:
        topo_scores: (N_total,) prompt-level topo-confidence scores
        completions_by_problem: {str(idx): list of completions}
        ground_truths: All ground truths
        problem_indices: Which problems to evaluate
        greedy_correct: (N_total,) boolean greedy correctness
        tau_values: Threshold values to sweep

    Returns:
        List of dicts with per-tau results
    """
    if tau_values is None:
        tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    for tau in tau_values:
        correct = 0
        trust_greedy = 0
        use_mv = 0
        w_to_r = 0
        r_to_w = 0

        for idx in problem_indices:
            idx_str = str(int(idx))
            score = topo_scores[idx] if idx < len(topo_scores) else 0.0

            if score >= tau:
                # Trust greedy
                trust_greedy += 1
                if greedy_correct[idx]:
                    correct += 1
                # If greedy was correct, still correct. If wrong, still wrong.
            else:
                # Use majority vote
                use_mv += 1
                if greedy_correct[idx]:
                    correct += 1  # Greedy-correct, MV likely agrees
                elif idx_str in completions_by_problem:
                    comps = completions_by_problem[idx_str]
                    _, is_correct = select_majority_vote(comps, ground_truths[idx])
                    if is_correct:
                        correct += 1
                        w_to_r += 1

        n_greedy = int(greedy_correct[problem_indices].sum())
        results.append({
            "tau": tau,
            "trust_greedy": trust_greedy,
            "use_mv": use_mv,
            "correct": correct,
            "net_gain": correct - n_greedy,
            "w_to_r": w_to_r,
            "r_to_w": r_to_w,
            "savings_pct": round(100 * trust_greedy / len(problem_indices), 1),
        })

    return results


# ---- Comparison table builder ----


def build_comparison_table(
    math500_results: dict | None = None,
    gsm8k_results: dict | None = None,
    math500_7b_results: dict | None = None,
) -> str:
    """Build the final cross-benchmark/cross-model comparison table.

    Args:
        math500_results: Results from 1.5B × MATH-500 (from existing experiments)
        gsm8k_results: Results from Track A
        math500_7b_results: Results from Track B

    Returns:
        Formatted markdown table string
    """
    rows = [
        "| Metric | 1.5B × MATH-500 | 1.5B × GSM8K | 7B × MATH-500 |",
        "|--------|:---:|:---:|:---:|",
    ]

    def val(d, key, fmt="{}"): return fmt.format(d.get(key, "?")) if d else "—"

    metrics = [
        ("Greedy accuracy", "greedy_acc", "{:.1%}"),
        ("Topo AUROC", "topo_auroc", "{:.3f}"),
        ("Topo AUROC 95% CI", "topo_auroc_ci", "{}"),
        ("Best logprob AUROC", "best_logprob_auroc", "{:.3f}"),
        ("Ungated MV net gain", "mv_net_gain", "{:+d}"),
        ("Gated MV net gain (best τ)", "gated_mv_net_gain", "{:+d}"),
        ("Weighted vote net gain", "weighted_vote_net_gain", "{:+d}"),
        ("R→W (weighted vote)", "weighted_vote_r_to_w", "{}"),
        ("Oracle ceiling", "oracle_ceiling", "{:+d}"),
    ]

    for label, key, fmt in metrics:
        r1 = val(math500_results, key, fmt)
        r2 = val(gsm8k_results, key, fmt)
        r3 = val(math500_7b_results, key, fmt)
        rows.append(f"| {label} | {r1} | {r2} | {r3} |")

    return "\n".join(rows)
