"""Shared utilities for Pathway 3: Closing the Steering Generalization Gap.

Extends Track A common.py with Pathway 3 constants, multi-layer hook support,
soft gating, and bootstrap/clustering utilities.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr

# Import Track A common utilities via importlib to avoid circular import
# (pathway3/common.py would shadow pathway2/track_a/common.py if we used sys.path)
import importlib.util

TRACK_A_DIR = Path(__file__).parent.parent / "pathway2" / "track_a"
_spec = importlib.util.spec_from_file_location("track_a_common", TRACK_A_DIR / "common.py")
_ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ta)

# Re-export Track A symbols
MODEL_NAME = _ta.MODEL_NAME
N_PROBLEMS = _ta.N_PROBLEMS
MAX_NEW_TOKENS = _ta.MAX_NEW_TOKENS
MAX_LENGTH = _ta.MAX_LENGTH
SEED = _ta.SEED
YOUDEN_THRESHOLD = _ta.YOUDEN_THRESHOLD
LR_PARAMS = _ta.LR_PARAMS
MATH_PROMPT_TEMPLATE = _ta.MATH_PROMPT_TEMPLATE
HOLDOUT_MASK_PATH = _ta.HOLDOUT_MASK_PATH
FEATURES_TRAIN_PATH = _ta.FEATURES_TRAIN_PATH
FEATURES_HOLDOUT_PATH = _ta.FEATURES_HOLDOUT_PATH
HOLDOUT_PREDICTIONS_PATH = _ta.HOLDOUT_PREDICTIONS_PATH
HOLDOUT_METRICS_PATH = _ta.HOLDOUT_METRICS_PATH
TIER_ASSIGNMENTS_PATH = _ta.TIER_ASSIGNMENTS_PATH
ARTIFACT_LOCKS_PATH = _ta.ARTIFACT_LOCKS_PATH
WINNING_FEATURES_PATH = _ta.WINNING_FEATURES_PATH
TRAJ_PATH = _ta.TRAJ_PATH
META_PATH = _ta.META_PATH
load_math500 = _ta.load_math500
format_prompt = _ta.format_prompt
get_prompts_and_ground_truths = _ta.get_prompts_and_ground_truths
extract_answer = _ta.extract_answer
normalize_answer = _ta.normalize_answer
check_correct = _ta.check_correct
get_train_holdout_indices = _ta.get_train_holdout_indices
get_abc_column_indices = _ta.get_abc_column_indices
verify_pathway1_artifacts = _ta.verify_pathway1_artifacts
load_model = _ta.load_model
generate_one = _ta.generate_one
generate_one_with_scores = _ta.generate_one_with_scores
slerp = _ta.slerp
make_steering_hook = _ta.make_steering_hook
load_trajectory_meta = _ta.load_trajectory_meta
load_cached_trajectories = _ta.load_cached_trajectories

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("pathway3")

# ---- Paths ----

PATHWAY3_DIR = Path(__file__).parent
TOPO_DIR = PATHWAY3_DIR.parent
PATHWAY1_DIR = TOPO_DIR / "pathway1"
PATHWAY2_DIR = TOPO_DIR / "pathway2"

# Track A artifacts
TRACK_A_PHASE0 = TRACK_A_DIR / "phase0"
TRACK_A_PHASE1 = TRACK_A_DIR / "phase1"
TRACK_A_PHASE2 = TRACK_A_DIR / "phase2"
TRACK_A_PHASE3 = TRACK_A_DIR / "phase3"

# Pathway 3 phase directories
PHASE1_DIR = PATHWAY3_DIR / "phase1"
PHASE2_DIR = PATHWAY3_DIR / "phase2"
PHASE3_DIR = PATHWAY3_DIR / "phase3"
PHASE4_DIR = PATHWAY3_DIR / "phase4"

# ---- Pathway 3 Constants ----

# Layers for multi-layer experiments
P3_LAYERS = [15, 20, 24, 25]
ADDITIONAL_LAYERS = [15, 25]  # Not in Track A's candidate set

# Temperature sampling config
TEMP_SAMPLING = dict(
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    max_new_tokens=MAX_NEW_TOKENS,
)
COMPLETIONS_PER_PROBLEM = 32

# Denoising config
BOOTSTRAP_SAMPLES = 20
PCA_K_SWEEP = [5, 10, 20, 50]
SPARSE_D_SWEEP = [50, 100, 200, 500]

# Prototype config
PROTOTYPE_K_SWEEP = [3, 5, 8]
PROTOTYPE_TEMP_SWEEP = [0.1, 0.5, 1.0]

# Multi-layer configs
MULTI_LAYER_CONFIGS = {
    "track_a_baseline": [24],
    "seal": [20],
    "non_adjacent_a": [15, 25],
    "non_adjacent_b": [20, 25],
    "triple": [15, 20, 25],
}

# CV config
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42

# Soft gating sigmoid sharpness sweep
SIGMOID_SHARPNESS_SWEEP = [5, 10, 20]

# Track A locked config (for baseline replication)
TRACK_A_BEST_T = 0.15
TRACK_A_BEST_TAU = 0.40
TRACK_A_LAYER = 24


# ---- Multi-layer steering hooks ----


def register_multi_layer_hooks(
    model,
    layer_vectors: dict[int, np.ndarray],
    t: float | dict[int, float],
) -> list:
    """Register steering hooks at multiple layers.

    Args:
        model: The HuggingFace model.
        layer_vectors: {layer_idx: unit-normalized steering vector (1536,)}
        t: Steering strength. Either a single float (applied to all layers)
           or a dict {layer_idx: t_value}.

    Returns:
        List of hook handles (call handle.remove() to clean up).
    """
    handles = []
    for layer_idx, vec in layer_vectors.items():
        t_val = t[layer_idx] if isinstance(t, dict) else t
        vec_gpu = torch.tensor(vec, dtype=torch.float16, device=next(model.parameters()).device)
        hook = make_steering_hook(vec_gpu, t_val)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)
    return handles


def remove_all_hooks(handles: list) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()


# ---- Prototype-based steering ----


def compute_prototype_steering_vector(
    h_input: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 0.5,
) -> np.ndarray:
    """Compute weighted steering vector from K prototypes.

    Args:
        h_input: Activation of input problem, shape (1536,)
        prototypes: K prototype centroids, shape (K, 1536)
        temperature: Softmax temperature (low = sharp selection)

    Returns:
        Unit-normalized weighted steering vector, shape (1536,)
    """
    projections = prototypes @ h_input  # (K,)
    weights = _softmax(projections / temperature)
    steering = weights @ prototypes  # (1536,)
    norm = np.linalg.norm(steering)
    if norm < 1e-12:
        return prototypes[0] / (np.linalg.norm(prototypes[0]) + 1e-12)
    return steering / norm


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


# ---- Soft confidence gating ----


def soft_gate_strength(
    confidence: float,
    t_base: float,
    threshold: float,
    sharpness: float = 10.0,
) -> float:
    """Compute per-problem steering strength via sigmoid gating.

    t_per_problem = t_base * sigmoid(sharpness * (threshold - confidence))

    When confidence << threshold: sigmoid ≈ 1, full steering.
    When confidence >> threshold: sigmoid ≈ 0, no steering.
    When confidence = threshold: sigmoid = 0.5, half steering.

    Args:
        confidence: P(correct) for this problem.
        t_base: Maximum steering strength.
        threshold: Confidence threshold (e.g., 0.40).
        sharpness: Sigmoid steepness (higher = sharper transition).

    Returns:
        Steering strength t in [0, t_base].
    """
    return float(t_base * sigmoid(sharpness * (threshold - confidence)))


# ---- Bootstrap averaging ----


def bootstrap_mean_diff_vector(
    H_correct: np.ndarray,
    H_incorrect_mean: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> np.ndarray:
    """Compute bootstrap-averaged mean-difference vector.

    Draws B bootstrap samples from H_correct, computes mean-diff for each,
    averages the B vectors, and unit-normalizes.

    Args:
        H_correct: Correct activations, shape (N_correct, hidden_dim)
        H_incorrect_mean: Mean of incorrect activations, shape (hidden_dim,)
        n_bootstrap: Number of bootstrap samples.
        seed: RNG seed.

    Returns:
        Unit-normalized averaged vector, shape (hidden_dim,)
    """
    rng = np.random.default_rng(seed)
    n_correct = len(H_correct)
    vectors = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_correct, size=n_correct, replace=True)
        mean_correct = H_correct[idx].mean(axis=0)
        v = mean_correct - H_incorrect_mean
        v = v / (np.linalg.norm(v) + 1e-12)
        vectors.append(v)

    avg = np.mean(vectors, axis=0)
    return avg / (np.linalg.norm(avg) + 1e-12)


# ---- PCA projection ----


def pca_project_vector(
    vector: np.ndarray,
    diff_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """Project vector onto top-k PCA subspace of the difference matrix.

    Args:
        vector: Steering vector, shape (hidden_dim,)
        diff_matrix: Per-sample differences, shape (N, hidden_dim)
        k: Number of principal components to retain.

    Returns:
        Unit-normalized projected vector, shape (hidden_dim,)
    """
    # SVD of difference matrix
    U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
    # Top-k components
    Vk = Vt[:k]  # (k, hidden_dim)
    # Project: vector onto span of Vk
    coords = Vk @ vector  # (k,)
    projected = coords @ Vk  # (hidden_dim,)
    norm = np.linalg.norm(projected)
    if norm < 1e-12:
        return vector  # fallback to original
    return projected / norm


# ---- Top-d sparsification ----


def sparsify_vector(vector: np.ndarray, d: int) -> np.ndarray:
    """Zero out all but top-d largest-magnitude dimensions.

    Args:
        vector: Input vector, shape (hidden_dim,)
        d: Number of dimensions to keep.

    Returns:
        Unit-normalized sparse vector, shape (hidden_dim,)
    """
    abs_vals = np.abs(vector)
    threshold = np.sort(abs_vals)[-d] if d < len(vector) else 0
    sparse = vector.copy()
    sparse[abs_vals < threshold] = 0.0
    norm = np.linalg.norm(sparse)
    if norm < 1e-12:
        return vector
    return sparse / norm


# ---- Clustering utilities ----


def cluster_differences(
    diff_matrix: np.ndarray,
    K: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """K-means clustering of activation differences.

    Args:
        diff_matrix: Per-sample differences, shape (N, hidden_dim)
        K: Number of clusters.
        seed: RNG seed.

    Returns:
        centroids: (K, hidden_dim), unit-normalized
        labels: (N,) cluster assignments
        quality: dict with silhouette, cluster_sizes, cosine_matrix
    """
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(diff_matrix)
    centroids = km.cluster_centers_.copy()

    # Unit-normalize centroids
    for i in range(K):
        norm = np.linalg.norm(centroids[i])
        if norm > 1e-12:
            centroids[i] /= norm

    # Cluster sizes
    cluster_sizes = [int((labels == k).sum()) for k in range(K)]

    # Silhouette score (skip if any cluster has < 2 samples)
    if min(cluster_sizes) >= 2 and K > 1:
        sil = float(silhouette_score(diff_matrix, labels))
    else:
        sil = -1.0

    # Cross-cluster cosine similarity matrix
    cosine_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cosine_matrix[i, j] = float(
                np.dot(centroids[i], centroids[j])
                / (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-12)
            )

    quality = {
        "silhouette": round(sil, 4),
        "cluster_sizes": cluster_sizes,
        "cosine_matrix": cosine_matrix.tolist(),
        "min_cluster_size": min(cluster_sizes),
        "valid": min(cluster_sizes) >= 5,
    }

    return centroids, labels, quality


# ---- Confidence pipeline for arbitrary train/test splits ----


def fit_confidence_pipeline(
    train_idx: np.ndarray,
    y_all: np.ndarray,
) -> tuple[StandardScaler, LogisticRegression, list[str], list[int]]:
    """Fit the CORAL confidence pipeline on a train subset.

    Uses cached Pathway 1 features (fallback approach from Track A).

    Args:
        train_idx: Indices of train problems.
        y_all: Correctness labels for all 500 problems.

    Returns:
        scaler: Fitted StandardScaler
        classifier: Fitted LogisticRegression
        abc_names: Feature names
        abc_indices: Column indices
    """
    # Load cached features (full 500)
    with open(ARTIFACT_LOCKS_PATH) as f:
        locks = json.load(f)
    feature_names = locks["feature_names"]

    # We need features for ALL problems. Load from Pathway 1 cached files.
    # features_train400.npy and features_holdout100.npy are for the P1 split.
    # For arbitrary CV folds, we need to reconstruct full 500.
    holdout_mask = np.load(HOLDOUT_MASK_PATH)
    p1_train_idx = np.where(~holdout_mask)[0]
    p1_holdout_idx = np.where(holdout_mask)[0]

    features_p1_train = np.load(FEATURES_TRAIN_PATH)  # (400, 78)
    features_p1_holdout = np.load(FEATURES_HOLDOUT_PATH)  # (100, 78)

    # Reconstruct full (500, 78) feature matrix
    n_features = features_p1_train.shape[1]
    features_all = np.zeros((N_PROBLEMS, n_features), dtype=np.float32)
    features_all[p1_train_idx] = features_p1_train
    features_all[p1_holdout_idx] = features_p1_holdout

    # Select A+B+C columns
    abc_indices = get_abc_column_indices(feature_names)
    abc_names = [feature_names[i] for i in abc_indices]

    features_abc = features_all[:, abc_indices]

    # Fit scaler on this fold's train set
    X_train = features_abc[train_idx]
    y_train = y_all[train_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)

    return scaler, clf, abc_names, abc_indices


def predict_confidence(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    abc_indices: list[int],
    problem_indices: np.ndarray,
) -> np.ndarray:
    """Predict P(correct) for given problem indices.

    Args:
        scaler: Fitted StandardScaler
        classifier: Fitted LogisticRegression
        abc_indices: Column indices for A+B+C features
        problem_indices: Which problems to predict for

    Returns:
        confidence_scores: (len(problem_indices),)
    """
    # Load features
    holdout_mask = np.load(HOLDOUT_MASK_PATH)
    p1_train_idx = np.where(~holdout_mask)[0]
    p1_holdout_idx = np.where(holdout_mask)[0]

    features_p1_train = np.load(FEATURES_TRAIN_PATH)
    features_p1_holdout = np.load(FEATURES_HOLDOUT_PATH)

    n_features = features_p1_train.shape[1]
    features_all = np.zeros((N_PROBLEMS, n_features), dtype=np.float32)
    features_all[p1_train_idx] = features_p1_train
    features_all[p1_holdout_idx] = features_p1_holdout

    features_abc = features_all[:, abc_indices]
    X = features_abc[problem_indices]
    X_scaled = scaler.transform(X)

    return classifier.predict_proba(X_scaled)[:, 1]


# ---- Statistical tests ----


def mcnemar_mid_p(b: int, c: int) -> float:
    """Mid-p McNemar's test.

    More appropriate than asymptotic chi-squared for small discordant counts.

    Args:
        b: Count of (correct under A, wrong under B)
        c: Count of (wrong under A, correct under B)

    Returns:
        Two-sided mid-p value.
    """
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided: P(X >= max(b,c)) + P(X <= min(b,c)) under Binom(n, 0.5)
    # Mid-p: subtract 0.5 * P(X = observed) from each tail
    x = max(b, c)
    tail = binom.sf(x - 1, n, 0.5)  # P(X >= x)
    mid = tail - 0.5 * binom.pmf(x, n, 0.5)
    return float(min(2 * mid, 1.0))  # two-sided


def bayesian_p_improvement(b: int, c: int) -> float:
    """Bayesian P(improvement) with flat prior.

    Models b ~ Binomial(b+c, p), flat prior on p.
    Posterior: p ~ Beta(b+1, c+1).
    P(improvement) = P(p > 0.5) = 1 - BetaCDF(0.5 | b+1, c+1).

    Args:
        b: wrong→right flips
        c: right→wrong flips

    Returns:
        P(improvement)
    """
    from scipy.stats import beta
    return float(1 - beta.cdf(0.5, b + 1, c + 1))


def bootstrap_ci_net_gain(
    correct_baseline: np.ndarray,
    correct_steered: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """BCa bootstrap 95% CI on net gain.

    Args:
        correct_baseline: (N,) baseline correctness
        correct_steered: (N,) steered correctness

    Returns:
        (lower, upper) CI bounds
    """
    rng = np.random.default_rng(seed)
    N = len(correct_baseline)
    gains = []

    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        b = correct_baseline[idx]
        s = correct_steered[idx]
        w2r = int(((b == 0) & (s == 1)).sum())
        r2w = int(((b == 1) & (s == 0)).sum())
        gains.append(w2r - r2w)

    gains = np.array(gains)
    alpha = 1 - ci
    lower = float(np.percentile(gains, 100 * alpha / 2))
    upper = float(np.percentile(gains, 100 * (1 - alpha / 2)))
    return lower, upper


# ---- Temperature generation ----


@torch.no_grad()
def generate_one_temperature(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a single completion with temperature sampling."""
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
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---- Activation extraction at specific layers ----


@torch.no_grad()
def extract_prompt_activations(
    model,
    tokenizer,
    prompt: str,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Extract last-token prompt activations at specified layers.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Input prompt
        layers: Layer indices to extract

    Returns:
        {layer_idx: activation (1536,)} as float32 numpy arrays
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs["attention_mask"].sum().item()
    last_pos = seq_len - 1

    result = {}
    for layer in layers:
        # hidden_states index: 0=embeddings, 1=layer0, ..., 28=layer27
        hs = outputs.hidden_states[layer + 1]  # +1 for embedding offset
        result[layer] = hs[0, last_pos].cpu().float().numpy()

    return result


# ---- Evaluation metrics ----


def compute_flip_metrics(
    baseline_correct: np.ndarray,
    steered_correct: np.ndarray,
) -> dict:
    """Compute steering flip metrics.

    Args:
        baseline_correct: (N,) int/bool
        steered_correct: (N,) int/bool

    Returns:
        Dict with accuracy, net_gain, wrong_to_right, right_to_wrong, etc.
    """
    b = baseline_correct.astype(bool)
    s = steered_correct.astype(bool)

    wrong_to_right = int((~b & s).sum())
    right_to_wrong = int((b & ~s).sum())
    net_gain = wrong_to_right - right_to_wrong

    return {
        "n_problems": len(b),
        "baseline_accuracy": float(b.mean()),
        "steered_accuracy": float(s.mean()),
        "baseline_correct": int(b.sum()),
        "steered_correct": int(s.sum()),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_gain": net_gain,
        "preserved": int((b & s).sum()),
        "both_wrong": int((~b & ~s).sum()),
    }
