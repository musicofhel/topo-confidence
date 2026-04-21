"""Per-layer PCA + 6 PH features per layer = 168 total features.

The core of Experiment 1. Computes persistent homology independently at each
of the 28 transformer layers after per-layer z-score + PCA(20) reduction.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from ripser import ripser
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathway8_layerwise.config import (
    N_PCA_PER_LAYER,
    N_TRANSFORMER_LAYERS,
    N_TOTAL_LAYERS,
    SUBSAMPLE,
    MAX_DIM,
    PH_FEATURE_NAMES,
)
from pathway8_layerwise.extraction_utils import subsample_tokens


# ---------------------------------------------------------------------------
# Per-layer PCA fitting
# ---------------------------------------------------------------------------


def fit_per_layer_pca(
    states_list: list[np.ndarray | None],
    train_idx: np.ndarray,
    n_components: int = N_PCA_PER_LAYER,
    layers: range | None = None,
) -> dict[int, tuple[StandardScaler, PCA]]:
    """Fit StandardScaler + PCA per layer on TRAIN problems only.

    Parameters
    ----------
    states_list : list of (29, n_tokens, 1536) arrays (or None for missing)
    train_idx : indices of training problems
    n_components : PCA components per layer
    layers : which layers to fit (default: 1-28, skipping embedding layer 0)

    Returns
    -------
    dict mapping layer_idx -> (scaler, pca)
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)

    transforms: dict[int, tuple[StandardScaler, PCA]] = {}

    for layer in layers:
        # Pool all train-problem tokens at this layer
        train_points = []
        for idx in train_idx:
            s = states_list[idx]
            if s is None:
                continue
            # s[layer]: (n_tokens, 1536) — convert from float16 to float32
            train_points.append(s[layer].astype(np.float32))

        if not train_points:
            raise ValueError(f"No train data for layer {layer}")

        pooled = np.concatenate(train_points, axis=0)
        n_comp = min(n_components, pooled.shape[0] - 1, pooled.shape[1])

        scaler = StandardScaler().fit(pooled)
        scaled = scaler.transform(pooled)
        pca = PCA(n_components=n_comp, random_state=0).fit(scaled)

        transforms[layer] = (scaler, pca)

        if layer == 1 or layer == N_TRANSFORMER_LAYERS:
            var_explained = 100 * pca.explained_variance_ratio_.sum()
            print(
                f"  Layer {layer:2d}: PCA({n_comp}) explains "
                f"{var_explained:.1f}% variance on {pooled.shape[0]} tokens"
            )

    return transforms


# ---------------------------------------------------------------------------
# PH feature computation
# ---------------------------------------------------------------------------


def _persistence_entropy(lifetimes: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of normalized persistence lifetimes."""
    L = lifetimes[lifetimes > eps]
    if len(L) < 2:
        return 0.0
    p = L / L.sum()
    return float(-np.sum(p * np.log(p + eps)))


def _lifetimes(dgm: np.ndarray) -> np.ndarray:
    """Extract finite lifetimes from a persistence diagram."""
    if dgm is None or len(dgm) == 0:
        return np.array([])
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return np.array([])
    return finite[:, 1] - finite[:, 0]


def compute_6_ph_features(
    X: np.ndarray,
    thresh: float | None = None,
    pers_thr: float | None = None,
) -> np.ndarray:
    """Compute 6 PH features from a point cloud.

    Parameters
    ----------
    X : (n_points, d) float32 — PCA-reduced, z-scored
    thresh : filtration threshold (default: 90th percentile of pairwise distances)
    pers_thr : persistence threshold for n_features counting

    Returns
    -------
    (6,) array: [H0_total_persistence, H0_n_features, H0_entropy,
                  H1_persistence_entropy, H1_n_features, H1_max_lifetime]
    """
    zeros = np.zeros(6, dtype=np.float64)

    if X is None or len(X) < 3 or not np.all(np.isfinite(X)):
        return zeros

    X = np.ascontiguousarray(X, dtype=np.float32)

    # Check for degenerate clouds
    if np.ptp(X, axis=0).sum() == 0:
        return zeros

    d = pdist(X)
    if len(d) == 0 or d.max() == 0:
        return zeros

    if thresh is None:
        thresh = float(np.quantile(d, 0.90))
    if pers_thr is None:
        pers_thr = 0.01 * float(np.median(d))

    try:
        res = ripser(X, maxdim=MAX_DIM, thresh=thresh, coeff=2)
        dgm0, dgm1 = res["dgms"][0], res["dgms"][1]
    except Exception:
        return zeros

    L0 = _lifetimes(dgm0)
    L1 = _lifetimes(dgm1)

    features = np.zeros(6, dtype=np.float64)
    features[0] = float(L0.sum()) if len(L0) else 0.0          # H0_total_persistence
    features[1] = int((L0 > pers_thr).sum())                    # H0_n_features
    features[2] = _persistence_entropy(L0)                       # H0_entropy
    features[3] = _persistence_entropy(L1)                       # H1_persistence_entropy
    features[4] = int((L1 > pers_thr).sum())                    # H1_n_features
    features[5] = float(L1.max()) if len(L1) else 0.0          # H1_max_lifetime

    return features


def compute_layerwise_ph_single(
    states: np.ndarray,
    per_layer_transforms: dict[int, tuple[StandardScaler, PCA]],
    layers: range | None = None,
    subsample_n: int = SUBSAMPLE,
) -> np.ndarray:
    """Compute 6 PH features per layer for a single problem.

    Parameters
    ----------
    states : (29, n_tokens, 1536) — all-layer hidden states for one problem
    per_layer_transforms : dict from fit_per_layer_pca()
    layers : which layers (default: 1-28)
    subsample_n : max tokens for ripser

    Returns
    -------
    (168,) feature vector (28 layers × 6 features)
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)

    all_features = []
    for layer in layers:
        layer_data = states[layer].astype(np.float32)  # (n_tokens, 1536)

        if layer not in per_layer_transforms:
            all_features.append(np.zeros(6, dtype=np.float64))
            continue

        scaler, pca = per_layer_transforms[layer]
        scaled = scaler.transform(layer_data)
        reduced = pca.transform(scaled)  # (n_tokens, 20)

        # Subsample
        sub = subsample_tokens(reduced, subsample_n)

        # Compute PH
        features = compute_6_ph_features(sub.astype(np.float32))
        all_features.append(features)

    return np.concatenate(all_features)


def compute_layerwise_ph_batch(
    states_list: list[np.ndarray | None],
    per_layer_transforms: dict[int, tuple[StandardScaler, PCA]],
    n_jobs: int = -1,
    layers: range | None = None,
) -> np.ndarray:
    """Compute layer-wise PH features for all problems in parallel.

    Returns (n_problems, 168) feature array.
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)

    n_features = len(layers) * 6

    def _process_one(idx: int) -> np.ndarray:
        s = states_list[idx]
        if s is None:
            return np.zeros(n_features, dtype=np.float64)
        return compute_layerwise_ph_single(s, per_layer_transforms, layers)

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one)(i) for i in range(len(states_list))
    )
    return np.stack(results)


# ---------------------------------------------------------------------------
# Z-score divergence analysis (REMA-style)
# ---------------------------------------------------------------------------


def compute_zscore_divergence(
    features_all: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    alpha: float = 2.0,
) -> dict:
    """Find earliest layer where PH features deviate from correct-problem distribution.

    For each layer L:
      - Compute mean/std of 6 features on CORRECT train problems
      - For each problem, compute max z-score of its L features vs correct dist
      - Earliest layer where max|z| > alpha is the "divergence layer"

    Returns dict with per-problem divergence layers and summary statistics.
    """
    n_problems = features_all.shape[0]
    n_layers = N_TRANSFORMER_LAYERS
    n_ph = 6

    # Reshape to (n_problems, n_layers, n_ph)
    F = features_all.reshape(n_problems, n_layers, n_ph)

    # Correct train problems
    correct_train_mask = labels[train_idx].astype(bool)
    correct_train_idx = train_idx[correct_train_mask]

    # Per-layer mean/std of correct train problems
    layer_means = np.zeros((n_layers, n_ph))
    layer_stds = np.zeros((n_layers, n_ph))
    for l in range(n_layers):
        correct_feats = F[correct_train_idx, l, :]  # (n_correct, 6)
        layer_means[l] = correct_feats.mean(axis=0)
        layer_stds[l] = correct_feats.std(axis=0) + 1e-12

    # Per-problem divergence layer
    divergence_layers = np.full(n_problems, n_layers, dtype=int)  # default: no divergence
    max_zscores = np.zeros((n_problems, n_layers))

    for i in range(n_problems):
        for l in range(n_layers):
            z = np.abs((F[i, l, :] - layer_means[l]) / layer_stds[l])
            max_zscores[i, l] = z.max()
            if z.max() > alpha and divergence_layers[i] == n_layers:
                divergence_layers[i] = l

    return {
        "divergence_layers": divergence_layers,
        "max_zscores": max_zscores,
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "mean_divergence_correct": float(
            divergence_layers[labels.astype(bool)].mean()
        ),
        "mean_divergence_incorrect": float(
            divergence_layers[~labels.astype(bool)].mean()
        ),
        "alpha": alpha,
    }
