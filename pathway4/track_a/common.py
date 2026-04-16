"""Shared utilities for Pathway 4 Track A: Topo-Confidence as Best-of-N Selector.

Extends Pathway 3 common with completion trajectory extraction,
PreFittedPCA monkey-patching, and selection strategy helpers.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import Pathway 3 common (which re-exports Track A common)
import importlib.util

PATHWAY3_DIR = Path(__file__).parent.parent.parent / "pathway3"
_spec = importlib.util.spec_from_file_location("p3_common", PATHWAY3_DIR / "common.py")
_p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p3)

# Re-export everything we need from Pathway 3 / Track A
MODEL_NAME = _p3.MODEL_NAME
N_PROBLEMS = _p3.N_PROBLEMS
MAX_NEW_TOKENS = _p3.MAX_NEW_TOKENS
MAX_LENGTH = _p3.MAX_LENGTH
SEED = _p3.SEED
LR_PARAMS = _p3.LR_PARAMS
MATH_PROMPT_TEMPLATE = _p3.MATH_PROMPT_TEMPLATE
HOLDOUT_MASK_PATH = _p3.HOLDOUT_MASK_PATH
FEATURES_TRAIN_PATH = _p3.FEATURES_TRAIN_PATH
FEATURES_HOLDOUT_PATH = _p3.FEATURES_HOLDOUT_PATH
TIER_ASSIGNMENTS_PATH = _p3.TIER_ASSIGNMENTS_PATH
ARTIFACT_LOCKS_PATH = _p3.ARTIFACT_LOCKS_PATH
WINNING_FEATURES_PATH = _p3.WINNING_FEATURES_PATH
TRAJ_PATH = _p3.TRAJ_PATH
META_PATH = _p3.META_PATH
load_math500 = _p3.load_math500
format_prompt = _p3.format_prompt
get_prompts_and_ground_truths = _p3.get_prompts_and_ground_truths
extract_answer = _p3.extract_answer
normalize_answer = _p3.normalize_answer
check_correct = _p3.check_correct
get_train_holdout_indices = _p3.get_train_holdout_indices
get_abc_column_indices = _p3.get_abc_column_indices
verify_pathway1_artifacts = _p3.verify_pathway1_artifacts
load_model = _p3.load_model
generate_one = _p3.generate_one
generate_one_temperature = _p3.generate_one_temperature
load_trajectory_meta = _p3.load_trajectory_meta
load_cached_trajectories = _p3.load_cached_trajectories
compute_flip_metrics = _p3.compute_flip_metrics
mcnemar_mid_p = _p3.mcnemar_mid_p
bayesian_p_improvement = _p3.bayesian_p_improvement
bootstrap_ci_net_gain = _p3.bootstrap_ci_net_gain

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("pathway4_track_a")

# ---- Paths ----

PATHWAY4_DIR = Path(__file__).parent.parent
TRACK_A_DIR = PATHWAY4_DIR / "track_a"
PHASE1_DIR = TRACK_A_DIR / "phase1"
PHASE2_DIR = TRACK_A_DIR / "phase2"
PHASE3_DIR = TRACK_A_DIR / "phase3"

# Upstream artifacts
TEMP_GEN_PATH = PATHWAY3_DIR / "phase1" / "temperature_generations.json"
BASELINE_CORRECT_PATH = (
    Path(__file__).parent.parent.parent / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
)
PATHWAY1_PHASE0_DIR = Path(__file__).parent.parent.parent / "pathway1" / "phase0"

# ---- Constants ----

COMPLETIONS_PER_PROBLEM = 32
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
TEMP_SAMPLING = dict(temperature=0.8, top_p=0.95)


# ---- Completion trajectory extraction ----


@torch.no_grad()
def extract_completion_trajectory(
    model,
    tokenizer,
    prompt: str,
    completion_text: str,
    max_length: int = MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass on prompt+completion, extract per-token trajectory and layer states.

    Args:
        model: HuggingFace model (Qwen2.5-1.5B-Instruct)
        tokenizer: Tokenizer
        prompt: Full formatted prompt string
        completion_text: Temperature-sampled completion text
        max_length: Max sequence length for tokenization

    Returns:
        trajectory: (n_tokens, 1536) last-layer hidden states for all positions
        layer_states: (29, 1536) hidden states at last token across all layers
    """
    full_text = prompt + completion_text
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs["attention_mask"].sum().item()
    last_pos = seq_len - 1

    # Last-layer trajectory: all token positions
    # hidden_states: tuple of (embeddings, layer0, ..., layer27) = 29 elements
    last_layer_hs = outputs.hidden_states[-1]  # (1, seq_len, 1536)
    trajectory = last_layer_hs[0, :seq_len].cpu().float().numpy()  # (seq_len, 1536)

    # Layer states at last token: all 29 layers (including embeddings)
    n_layers = len(outputs.hidden_states)  # 29
    layer_states = np.zeros((n_layers, 1536), dtype=np.float32)
    for k in range(n_layers):
        layer_states[k] = outputs.hidden_states[k][0, last_pos].cpu().float().numpy()

    return trajectory, layer_states


# ---- PCA monkey-patching ----


def fit_greedy_pca(trajectories: list[np.ndarray]) -> OrigPCA:
    """Fit PCA on greedy prompt trajectories for a stable subspace.

    Args:
        trajectories: List of 500 arrays, each (n_tokens, 1536)

    Returns:
        Fitted PCA with 45 components
    """
    all_points = np.concatenate(trajectories, axis=0)
    n_components = min(45, all_points.shape[1], all_points.shape[0])
    pca = OrigPCA(n_components=n_components, svd_solver="full")
    pca.fit(all_points)
    logger.info(
        "Fitted greedy PCA: %d components, %.1f%% variance explained",
        n_components,
        100 * pca.explained_variance_ratio_.sum(),
    )
    return pca


def extract_features_with_prefitted_pca(
    trajectories: list[np.ndarray],
    layer_states: np.ndarray,
    prefitted_pca: OrigPCA,
) -> tuple[np.ndarray, list[str]]:
    """Extract topo features using a pre-fitted PCA (prevents data leakage).

    Monkey-patches winning_features.PCA to use the prefitted PCA, then
    calls extract_features, then restores the original PCA class.

    Args:
        trajectories: list of (n_tokens, 1536) arrays
        layer_states: (N, 29, 1536) per-sample layer states
        prefitted_pca: PCA fitted on greedy trajectories

    Returns:
        features: (N, n_features) array
        feature_names: list of feature name strings
    """
    # Import winning_features module
    sys.path.insert(0, str(PATHWAY1_PHASE0_DIR))
    import winning_features as wf_module
    from winning_features import extract_features

    # Create PreFittedPCA class
    class PreFittedPCA(OrigPCA):
        _prefitted = None

        def fit(self, X, y=None):
            self.components_ = PreFittedPCA._prefitted.components_.copy()
            self.mean_ = PreFittedPCA._prefitted.mean_.copy()
            self.explained_variance_ = PreFittedPCA._prefitted.explained_variance_.copy()
            self.explained_variance_ratio_ = PreFittedPCA._prefitted.explained_variance_ratio_.copy()
            self.singular_values_ = PreFittedPCA._prefitted.singular_values_.copy()
            self.n_components_ = PreFittedPCA._prefitted.n_components_
            self.n_samples_ = PreFittedPCA._prefitted.n_samples_
            self.n_features_in_ = PreFittedPCA._prefitted.n_features_in_
            self.noise_variance_ = PreFittedPCA._prefitted.noise_variance_
            return self

    PreFittedPCA._prefitted = prefitted_pca
    original_pca = wf_module.PCA
    wf_module.PCA = PreFittedPCA

    try:
        features, feature_names = extract_features(trajectories, layer_states)
    finally:
        wf_module.PCA = original_pca

    return features, feature_names


# ---- Completion scoring ----


def fit_completion_scorer(
    features: np.ndarray,
    correct: np.ndarray,
    feature_names: list[str],
) -> tuple[StandardScaler, LogisticRegression, list[int]]:
    """Fit a completion-level scorer on topo features.

    Args:
        features: (N_completions, n_features) raw topo features
        correct: (N_completions,) bool correctness labels
        feature_names: list of feature name strings

    Returns:
        scaler: Fitted StandardScaler
        clf: Fitted LogisticRegression
        abc_indices: Column indices for A+B+C features
    """
    abc_indices = get_abc_column_indices(feature_names)
    X = features[:, abc_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_scaled, correct.astype(int))

    return scaler, clf, abc_indices


def score_completions(
    scaler: StandardScaler,
    clf: LogisticRegression,
    abc_indices: list[int],
    features: np.ndarray,
) -> np.ndarray:
    """Score completions with a fitted scorer.

    Args:
        scaler: Fitted StandardScaler
        clf: Fitted LogisticRegression
        abc_indices: Column indices for A+B+C features
        features: (N, n_features) raw topo features

    Returns:
        scores: (N,) P(correct) in [0, 1]
    """
    X = features[:, abc_indices]
    X_scaled = scaler.transform(X)
    return clf.predict_proba(X_scaled)[:, 1]


# ---- Selection strategies ----


def select_random(
    scores: np.ndarray,
    completions: list[str],
    ground_truth: str,
    rng: np.random.Generator,
) -> tuple[str, bool]:
    """Random baseline: pick a random completion."""
    idx = rng.integers(len(completions))
    answer = extract_answer(completions[idx])
    return completions[idx], check_correct(answer, ground_truth)


def select_max_confidence(
    scores: np.ndarray,
    completions: list[str],
    ground_truth: str,
) -> tuple[str, bool]:
    """Pick the completion with highest topo-confidence score."""
    idx = int(np.argmax(scores))
    answer = extract_answer(completions[idx])
    return completions[idx], check_correct(answer, ground_truth)


def select_majority_vote(
    completions: list[str],
    ground_truth: str,
) -> tuple[str, bool]:
    """Pick the most common answer among completions."""
    answers = [normalize_answer(extract_answer(c)) for c in completions]
    counter = Counter(answers)
    best_answer_norm = counter.most_common(1)[0][0]
    # Find a completion that produces this answer
    for c in completions:
        if normalize_answer(extract_answer(c)) == best_answer_norm:
            answer = extract_answer(c)
            return c, check_correct(answer, ground_truth)
    # Fallback (shouldn't happen)
    return completions[0], check_correct(extract_answer(completions[0]), ground_truth)


def select_confidence_weighted_vote(
    scores: np.ndarray,
    completions: list[str],
    ground_truth: str,
) -> tuple[str, bool]:
    """Weight each completion's vote by its confidence score."""
    answers = [normalize_answer(extract_answer(c)) for c in completions]
    # Accumulate confidence per unique answer
    answer_weights: dict[str, float] = {}
    for ans, score in zip(answers, scores):
        answer_weights[ans] = answer_weights.get(ans, 0.0) + float(score)
    best_answer_norm = max(answer_weights, key=answer_weights.get)
    # Find a completion that produces this answer
    for c in completions:
        if normalize_answer(extract_answer(c)) == best_answer_norm:
            answer = extract_answer(c)
            return c, check_correct(answer, ground_truth)
    return completions[0], check_correct(extract_answer(completions[0]), ground_truth)


# ---- CV fold creation ----


def create_train400_folds(
    y_train: np.ndarray,
    train_idx: np.ndarray,
    n_splits: int = CV_N_SPLITS,
    seed: int = CV_RANDOM_STATE,
) -> list[dict]:
    """Create 5-fold stratified splits over the 400 train problems.

    Args:
        y_train: (400,) correctness labels for train problems
        train_idx: (400,) global indices of train problems
        n_splits: Number of folds
        seed: Random state

    Returns:
        List of fold dicts with 'fold', 'train_local', 'test_local',
        'train_global', 'test_global' indices.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (fold_train_local, fold_test_local) in enumerate(
        skf.split(np.arange(len(y_train)), y_train)
    ):
        fold_train_global = train_idx[fold_train_local]
        fold_test_global = train_idx[fold_test_local]
        folds.append({
            "fold": fold_idx,
            "train_local": fold_train_local.tolist(),
            "test_local": fold_test_local.tolist(),
            "train_global": fold_train_global.tolist(),
            "test_global": fold_test_global.tolist(),
            "n_train": len(fold_train_local),
            "n_test": len(fold_test_local),
            "n_correct_train": int(y_train[fold_train_local].sum()),
            "n_correct_test": int(y_train[fold_test_local].sum()),
        })
    return folds
