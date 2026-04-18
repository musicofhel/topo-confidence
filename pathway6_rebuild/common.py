"""Corrected shared utilities for Pathway 6 Rebuild.

Single source of truth for answer checking, feature loading, and selection
strategies. Does NOT import extract_answer/normalize_answer/check_correct
from old code. Re-exports all non-buggy utilities via importlib.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as OrigPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("pathway6_rebuild")

# ---- Import non-buggy utilities from existing code ----

TOPO_DIR = Path(__file__).parent.parent
PATHWAY2_COMMON = TOPO_DIR / "pathway2" / "track_a" / "common.py"
PATHWAY4_COMMON = TOPO_DIR / "pathway4" / "track_a" / "common.py"
PATHWAY5_COMMON = TOPO_DIR / "pathway5" / "common.py"

_spec2 = importlib.util.spec_from_file_location("p2_common", PATHWAY2_COMMON)
_p2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_p2)

_spec4 = importlib.util.spec_from_file_location("p4_common", PATHWAY4_COMMON)
_p4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(_p4)

# Re-export non-buggy utilities
MODEL_NAME = _p2.MODEL_NAME
N_PROBLEMS = _p2.N_PROBLEMS
MAX_NEW_TOKENS = _p2.MAX_NEW_TOKENS
MAX_LENGTH = _p2.MAX_LENGTH
SEED = _p2.SEED
LR_PARAMS = _p2.LR_PARAMS
CV_PARAMS_10 = _p2.CV_PARAMS_10
CV_PARAMS_50 = _p2.CV_PARAMS_50
MATH_PROMPT_TEMPLATE = _p2.MATH_PROMPT_TEMPLATE
HOLDOUT_MASK_PATH = _p2.HOLDOUT_MASK_PATH
FEATURES_TRAIN_PATH = _p2.FEATURES_TRAIN_PATH
FEATURES_HOLDOUT_PATH = _p2.FEATURES_HOLDOUT_PATH
TIER_ASSIGNMENTS_PATH = _p2.TIER_ASSIGNMENTS_PATH
ARTIFACT_LOCKS_PATH = _p2.ARTIFACT_LOCKS_PATH
WINNING_FEATURES_PATH = _p2.WINNING_FEATURES_PATH
TRAJ_PATH = _p2.TRAJ_PATH
META_PATH = _p2.META_PATH
BASELINE_CORRECT_PATH = _p4.BASELINE_CORRECT_PATH
TEMP_GEN_PATH = _p4.TEMP_GEN_PATH

load_math500 = _p2.load_math500
format_prompt = _p2.format_prompt
get_prompts_and_ground_truths = _p2.get_prompts_and_ground_truths
get_train_holdout_indices = _p2.get_train_holdout_indices
get_abc_column_indices = _p2.get_abc_column_indices
verify_pathway1_artifacts = _p2.verify_pathway1_artifacts
load_trajectory_meta = _p2.load_trajectory_meta
load_cached_trajectories = _p2.load_cached_trajectories

# From pathway4 (non-buggy)
extract_completion_trajectory = _p4.extract_completion_trajectory
extract_features_with_prefitted_pca = _p4.extract_features_with_prefitted_pca
fit_completion_scorer = _p4.fit_completion_scorer
score_completions = _p4.score_completions
compute_flip_metrics = _p4.compute_flip_metrics
mcnemar_mid_p = _p4.mcnemar_mid_p
bayesian_p_improvement = _p4.bayesian_p_improvement
bootstrap_ci_net_gain = _p4.bootstrap_ci_net_gain
COMPLETIONS_PER_PROBLEM = _p4.COMPLETIONS_PER_PROBLEM
CV_N_SPLITS = _p4.CV_N_SPLITS
CV_RANDOM_STATE = _p4.CV_RANDOM_STATE
TEMP_SAMPLING = _p4.TEMP_SAMPLING

# Pathway 1 Phase 0 (winning_features)
PATHWAY1_PHASE0_DIR = _p4.PATHWAY1_PHASE0_DIR

# Pathway 6 rebuild directories
REBUILD_DIR = Path(__file__).parent
PHASE0_DIR = REBUILD_DIR / "phase0_relabel"
PHASE1_DIR = REBUILD_DIR / "phase1_prompt_model"
PHASE2_DIR = REBUILD_DIR / "phase2_completion"
PHASE3_DIR = REBUILD_DIR / "phase3_cross_benchmark"
PHASE4_DIR = REBUILD_DIR / "phase4_report"

# Pathway 6 rebuild artifact paths
BASELINE_CORRECT_V2_PATH = PHASE0_DIR / "baseline_correct_v2.npy"
TEMP_GEN_V2_PATH = PHASE0_DIR / "temperature_generations_v2.json"
HOLDOUT_TEMP_GEN_V2_PATH = PHASE0_DIR / "holdout_temperature_generations_v2.json"


# ============================================================================
# CORRECTED ANSWER EXTRACTION
# ============================================================================


def extract_boxed_balanced(text: str) -> str | None:
    """Extract content from \\boxed{...} using balanced-brace matching.

    Returns the content of the LAST \\boxed{} occurrence (models often box
    intermediate work then box the final answer last).

    Returns None if no \\boxed{} found.
    """
    results = []
    search_from = 0
    while True:
        idx = text.find("\\boxed{", search_from)
        if idx == -1:
            break
        # Start just after the opening brace
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[start : pos - 1])
        search_from = pos
    return results[-1] if results else None


def extract_answer_v2(text: str) -> str:
    """Extract the final answer from model output using corrected logic.

    Priority:
    1. \\boxed{} with balanced braces (LAST occurrence)
    2. #### marker (for GSM8K compatibility)
    3. "answer/result is" pattern
    4. Last number in text
    5. Full text stripped
    """
    # 1. Try \boxed{} with balanced braces
    boxed = extract_boxed_balanced(text)
    if boxed is not None:
        return boxed.strip()

    # 2. Try #### marker
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()

    # 3. Try "answer/result is" pattern
    match = re.search(
        r"(?:answer|result)\s+is\s+(.+?)(?:\.|,|\n|$)", text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # 4. Last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return text.strip()


def _parse_latex_to_sympy(expr_str: str):
    """Attempt to parse a LaTeX math expression into a sympy expression.

    Returns a sympy expression on success, None on failure.
    """
    import sympy
    from sympy import Rational, pi, sqrt

    s = expr_str.strip()

    # Strip surrounding $ signs
    s = s.strip("$")

    # Remove \left, \right
    s = s.replace("\\left", "").replace("\\right", "")

    # Remove \text{} wrapper
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # Remove trailing period or comma
    s = s.rstrip(".,")

    # Handle \frac{a}{b}
    def replace_frac(m):
        # m.group(0) is like \frac{a}{b} — but we need balanced braces
        return s  # placeholder, handled below

    # Iteratively replace \frac{a}{b} with (a)/(b) using balanced braces
    max_iter = 20
    for _ in range(max_iter):
        idx = s.find("\\frac{")
        if idx == -1:
            break
        # Extract first {a}
        start1 = idx + len("\\frac{")
        depth = 1
        pos = start1
        while pos < len(s) and depth > 0:
            if s[pos] == "{":
                depth += 1
            elif s[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0:
            break
        numer = s[start1 : pos - 1]
        # Next should be {b}
        if pos >= len(s) or s[pos] != "{":
            break
        start2 = pos + 1
        depth = 1
        pos2 = start2
        while pos2 < len(s) and depth > 0:
            if s[pos2] == "{":
                depth += 1
            elif s[pos2] == "}":
                depth -= 1
            pos2 += 1
        if depth != 0:
            break
        denom = s[start2 : pos2 - 1]
        s = s[:idx] + f"(({numer})/({denom}))" + s[pos2:]

    # Handle \sqrt{n} -> sqrt(n)
    for _ in range(max_iter):
        idx = s.find("\\sqrt{")
        if idx == -1:
            break
        start = idx + len("\\sqrt{")
        depth = 1
        pos = start
        while pos < len(s) and depth > 0:
            if s[pos] == "{":
                depth += 1
            elif s[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0:
            break
        content = s[start : pos - 1]
        s = s[:idx] + f"sqrt({content})" + s[pos:]

    # Handle \pi -> pi
    s = s.replace("\\pi", "pi")

    # Handle implicit multiplication: 3\sqrt -> 3*sqrt, 3pi -> 3*pi
    # Insert * between digit and letter/( where missing
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)

    # Handle \cdot
    s = s.replace("\\cdot", "*")
    # Handle \times
    s = s.replace("\\times", "*")

    # Remove remaining backslashes (e.g., \, for spacing)
    s = re.sub(r"\\[,;!]", "", s)
    s = s.replace("\\", "")

    # Try sympy parsing
    try:
        local_dict = {"pi": sympy.pi, "sqrt": sympy.sqrt, "I": sympy.I}
        expr = sympy.sympify(s, locals=local_dict, rational=True)
        return expr
    except Exception:
        return None


def normalize_answer_v2(answer: str) -> str:
    """Normalize an answer for comparison.

    Attempts sympy-based normalization for LaTeX expressions.
    Falls back to string normalization.
    """
    answer = answer.strip()

    # Strip surrounding $
    answer = answer.strip("$")

    # Basic cleaning
    cleaned = answer.replace(",", "").replace(" ", "").rstrip(".")

    # Try float first (simple numeric)
    try:
        val = float(cleaned)
        if val == int(val):
            return str(int(val))
        return f"{val:.10g}"
    except ValueError:
        pass

    # Return cleaned lowercase for string comparison
    return cleaned.lower()


def check_correct_v2(predicted_text: str, ground_truth: str) -> bool:
    """Check if a model prediction matches the ground truth.

    Uses a multi-level comparison strategy:
    1. Sympy-based comparison (handles LaTeX like \\frac, \\sqrt, \\pi)
    2. Float comparison with tolerance
    3. Normalized string comparison
    """
    import sympy

    # Extract answer from predicted text
    pred_answer = extract_answer_v2(predicted_text)

    # Level 1: Try sympy comparison
    pred_sym = _parse_latex_to_sympy(pred_answer)
    gt_sym = _parse_latex_to_sympy(ground_truth)

    if pred_sym is not None and gt_sym is not None:
        try:
            diff = sympy.simplify(pred_sym - gt_sym)
            if diff == 0:
                return True
            # Also try numerical evaluation for irrational numbers
            diff_float = complex(diff.evalf())
            if abs(diff_float) < 1e-6:
                return True
        except Exception:
            pass

    # Level 2: Try float comparison
    def try_float(s: str) -> float | None:
        s = s.strip().strip("$").replace(",", "").replace(" ", "").rstrip(".")
        try:
            return float(s)
        except ValueError:
            return None

    pred_f = try_float(pred_answer)
    gt_f = try_float(ground_truth)
    if pred_f is not None and gt_f is not None:
        if gt_f == 0:
            return abs(pred_f) < 1e-6
        return abs(pred_f - gt_f) / max(abs(gt_f), 1e-10) < 1e-4

    # Level 3: Normalized string comparison
    return normalize_answer_v2(pred_answer) == normalize_answer_v2(ground_truth)


# ============================================================================
# SSS ORDERING FIX
# ============================================================================


def load_features_with_sss_fix(
    baseline_correct: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features with SSS ordering fix applied.

    Features in features_train400.npy and features_holdout100.npy are stored
    in StratifiedShuffleSplit iteration order, NOT sorted by problem index.
    This function reconstructs the SSS ordering and returns features aligned
    with sorted global indices.

    Args:
        baseline_correct: (500,) bool correctness labels. If None, loads
            from BASELINE_CORRECT_PATH (old labels).

    Returns:
        X_train: (400, 78) features in sorted global index order
        X_holdout: (100, 78) features in sorted global index order
        train_idx: (400,) sorted global indices for train
        holdout_idx: (100,) sorted global indices for holdout
        train_sort_perm: permutation mapping SSS order -> sorted order
        holdout_sort_perm: permutation mapping SSS order -> sorted order
    """
    if baseline_correct is None:
        baseline_correct = np.load(BASELINE_CORRECT_PATH)

    # Reconstruct SSS ordering
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(
        sss.split(np.zeros(len(baseline_correct)), baseline_correct.astype(int))
    )

    # Permutations to go from SSS order to sorted order
    train_sort_perm = np.argsort(train_idx_sss)
    hold_sort_perm = np.argsort(hold_idx_sss)

    # Load and reorder features
    X_train = np.load(FEATURES_TRAIN_PATH)[train_sort_perm]
    X_holdout = np.load(FEATURES_HOLDOUT_PATH)[hold_sort_perm]

    # Sorted global indices
    train_idx = np.sort(train_idx_sss)
    holdout_idx = np.sort(hold_idx_sss)

    # Verify against mask-based indices
    _, holdout_idx_mask = get_train_holdout_indices()
    assert np.array_equal(holdout_idx, holdout_idx_mask), "Holdout index mismatch!"

    return X_train, X_holdout, train_idx, holdout_idx, train_sort_perm, hold_sort_perm


# ============================================================================
# PCA FIX: Train-400 only
# ============================================================================


def fit_greedy_pca_train_only(
    trajectories: list[np.ndarray], train_idx: np.ndarray
) -> OrigPCA:
    """Fit PCA on TRAIN-400 greedy trajectories only (no holdout leakage).

    Args:
        trajectories: list of 500 arrays, each (n_tokens, hidden_dim)
        train_idx: (400,) sorted global indices for train problems

    Returns:
        Fitted PCA with up to 45 components
    """
    train_trajs = [trajectories[i] for i in train_idx]
    all_points = np.concatenate(train_trajs, axis=0)
    n_components = min(45, all_points.shape[1], all_points.shape[0])
    pca = OrigPCA(n_components=n_components, svd_solver="full")
    pca.fit(all_points)
    logger.info(
        "Fitted greedy PCA (train-400 only): %d components, %.1f%% variance explained",
        n_components,
        100 * pca.explained_variance_ratio_.sum(),
    )
    return pca


# ============================================================================
# CORRECTED SELECTION STRATEGIES
# ============================================================================
# These use extract_answer_v2 + check_correct_v2 instead of old functions.


def select_random_v2(
    scores: np.ndarray,
    completions: list[str],
    ground_truth: str,
    rng: np.random.Generator,
) -> tuple[str, bool]:
    """Random baseline: pick a random completion."""
    idx = rng.integers(len(completions))
    return completions[idx], check_correct_v2(completions[idx], ground_truth)


def select_max_confidence_v2(
    scores: np.ndarray,
    completions: list[str],
    ground_truth: str,
) -> tuple[str, bool]:
    """Pick the completion with highest topo-confidence score."""
    idx = int(np.argmax(scores))
    return completions[idx], check_correct_v2(completions[idx], ground_truth)


def select_majority_vote_v2(
    completions: list[str],
    ground_truth: str,
) -> tuple[str, bool]:
    """Pick the most common answer among completions."""
    answers = [normalize_answer_v2(extract_answer_v2(c)) for c in completions]
    counter = Counter(answers)
    best_answer_norm = counter.most_common(1)[0][0]
    # Find a completion that produces this answer
    for c in completions:
        if normalize_answer_v2(extract_answer_v2(c)) == best_answer_norm:
            return c, check_correct_v2(c, ground_truth)
    return completions[0], check_correct_v2(completions[0], ground_truth)


def select_confidence_weighted_vote_v2(
    scores: np.ndarray,
    completions: list[str],
    ground_truth: str,
) -> tuple[str, bool]:
    """Weight each completion's vote by its confidence score."""
    answers = [normalize_answer_v2(extract_answer_v2(c)) for c in completions]
    answer_weights: dict[str, float] = {}
    for ans, score in zip(answers, scores):
        answer_weights[ans] = answer_weights.get(ans, 0.0) + float(score)
    best_answer_norm = max(answer_weights, key=answer_weights.get)
    for c in completions:
        if normalize_answer_v2(extract_answer_v2(c)) == best_answer_norm:
            return c, check_correct_v2(c, ground_truth)
    return completions[0], check_correct_v2(completions[0], ground_truth)


# ============================================================================
# CHAT TEMPLATE FORMATTING (for Phase 3)
# ============================================================================


def format_prompt_chat(problem_text: str, tokenizer, template: str = "math") -> str:
    """Format a problem with the model's chat template.

    Args:
        problem_text: The math problem text.
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        template: "math" for MATH-500 (use \\boxed{}), "gsm8k" for GSM8K (use ####).

    Returns:
        Formatted prompt string ready for tokenization.
    """
    if template == "math":
        content = (
            "Solve the following math problem. Show your work and put your "
            "final answer in \\boxed{}.\n\n" + problem_text
        )
    elif template == "gsm8k":
        content = (
            "Solve the following math problem. Give your final answer "
            "after '####'.\n\n" + problem_text
        )
    else:
        raise ValueError(f"Unknown template: {template}")

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================================
# MODEL LOADING (re-export with optional chat template support)
# ============================================================================

# Re-export existing model loading
load_model = _p2.load_model
generate_one = _p2.generate_one

# Pathway 5 model loading (parameterized for 1.5B/7B)
try:
    _spec5 = importlib.util.spec_from_file_location("p5_common", PATHWAY5_COMMON)
    _p5 = importlib.util.module_from_spec(_spec5)
    _spec5.loader.exec_module(_p5)
    load_model_parameterized = _p5.load_model
    generate_greedy = _p5.generate_greedy
    generate_greedy_with_logprobs = _p5.generate_greedy_with_logprobs
    generate_temperature = _p5.generate_temperature
    load_gsm8k = _p5.load_gsm8k
    format_gsm8k_prompt = _p5.format_gsm8k_prompt
    get_gsm8k_prompts_and_ground_truths = _p5.get_gsm8k_prompts_and_ground_truths
    extract_gsm8k_ground_truth = _p5.extract_gsm8k_ground_truth
    get_gsm8k_train_test_indices = _p5.get_gsm8k_train_test_indices
    HIDDEN_DIM = _p5.HIDDEN_DIM
    N_LAYERS = _p5.N_LAYERS
    MODEL_NAME_7B = _p5.MODEL_NAME_7B
    # Variable hidden-dim trajectory functions (override pathway4 hardcoded 1536)
    extract_trajectory = _p5.extract_trajectory
    extract_completion_trajectory = _p5.extract_completion_trajectory
    bootstrap_auroc = _p5.bootstrap_auroc
except Exception as e:
    logger.warning("Could not import pathway5 common: %s", e)


# ============================================================================
# TEMPERATURE GENERATION (re-export)
# ============================================================================

try:
    from pathlib import Path as _Path
    _spec3 = importlib.util.spec_from_file_location(
        "p3_common", TOPO_DIR / "pathway3" / "common.py"
    )
    _p3 = importlib.util.module_from_spec(_spec3)
    _spec3.loader.exec_module(_p3)
    generate_one_temperature = _p3.generate_one_temperature
except Exception as e:
    logger.warning("Could not import pathway3 common: %s", e)
