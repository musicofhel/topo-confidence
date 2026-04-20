"""Extended feature extractor: original 44 ABC-tier + non-Euclidean PH features.

Composes with the frozen winning_features.py by either:
  (a) Loading pre-computed 44-feature arrays from pathway6_rebuild, or
  (b) Calling winning_features.extract_features() on raw trajectories

Then adds ~18 non-Euclidean features (eff-res PH, cosine PH, spectral).

The non-Euclidean features are computed on the SAME PCA-reduced, subsampled
point clouds that the original pipeline uses, ensuring fair comparison.
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathway7.distance_metrics import (
    compute_all_noneuclid_features,
    effective_resistance_ph,
    cosine_ph,
    features_from_diagrams,
)

# Match winning_features.py constants
SUBSAMPLE = 100
N_PCA = 45
SEED = 42


def subsample_points(points: np.ndarray, n: int = SUBSAMPLE) -> np.ndarray:
    """Evenly-spaced subsample preserving temporal coverage (deterministic)."""
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]


def extract_noneuclid_features_single(
    traj_reduced: np.ndarray,
    k_effres: int = 30,
    maxdim: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Compute non-Euclidean PH features for a single PCA-reduced trajectory.

    Parameters
    ----------
    traj_reduced : (n_tokens, n_pca) PCA-transformed trajectory
    k_effres : kNN connectivity for effective-resistance graph
    maxdim : max homology dimension

    Returns
    -------
    (~18,) feature array and list of feature names
    """
    subsampled = subsample_points(traj_reduced, SUBSAMPLE)
    return compute_all_noneuclid_features(
        subsampled,
        k_effres=k_effres,
        maxdim=maxdim,
    )


def extract_extended_features(
    trajectories: list[np.ndarray],
    layer_states: np.ndarray | None = None,
    pca: PCA | None = None,
    base_features: np.ndarray | None = None,
    base_feature_names: list[str] | None = None,
    k_effres: int = 30,
) -> tuple[np.ndarray, list[str]]:
    """Extract original 44 + ~18 non-Euclidean features.

    Two modes:
    1. **With pre-computed base features**: Pass `base_features` and
       `base_feature_names` (loaded from .npy files). Still needs trajectories
       for computing non-Euclidean features.
    2. **From scratch**: Calls winning_features.extract_features() internally
       (slower, requires full trajectory data).

    Parameters
    ----------
    trajectories : list of (n_tokens, hidden_dim) arrays per problem
    layer_states : optional (n_problems, n_layers, hidden_dim) array
    pca : pre-fitted PCA transform (train-only). If None, fits on all data
          (WARNING: only do this for exploration, not final results).
    base_features : (n_problems, 44) pre-computed ABC-tier features
    base_feature_names : list of 44 feature names
    k_effres : kNN for effective-resistance graph

    Returns
    -------
    (n_problems, 44 + ~18) feature array and combined feature names
    """
    n_problems = len(trajectories)

    # --- Base features (existing 44 ABC-tier) ---
    if base_features is not None and base_feature_names is not None:
        assert base_features.shape[0] == n_problems, (
            f"base_features has {base_features.shape[0]} rows, "
            f"expected {n_problems}"
        )
    else:
        # Import and run the frozen extractor
        from pathway1.phase0.winning_features import extract_features as _extract_base
        base_features, base_feature_names = _extract_base(trajectories, layer_states)

    # --- PCA for non-Euclidean features ---
    if pca is None:
        all_points = np.concatenate(trajectories, axis=0)
        n_comp = min(N_PCA, all_points.shape[1], all_points.shape[0])
        pca = PCA(n_components=n_comp, svd_solver="full")
        pca.fit(all_points)

    # --- Non-Euclidean features ---
    new_feats_list = []
    new_names = None

    for i, traj in enumerate(trajectories):
        reduced = pca.transform(traj)
        feats, names = extract_noneuclid_features_single(
            reduced, k_effres=k_effres
        )
        new_feats_list.append(feats)
        if new_names is None:
            new_names = names

    new_features = np.stack(new_feats_list)  # (n_problems, ~18)

    # --- Combine ---
    combined = np.hstack([base_features, new_features])
    combined_names = list(base_feature_names) + new_names

    return combined, combined_names


def extract_noneuclid_only(
    trajectories: list[np.ndarray],
    pca: PCA | None = None,
    k_effres: int = 30,
) -> tuple[np.ndarray, list[str]]:
    """Extract only the non-Euclidean features (no base features).

    Useful for ablation: test non-Euclidean features alone vs combined.

    Parameters
    ----------
    trajectories : list of (n_tokens, hidden_dim) arrays
    pca : pre-fitted PCA (train-only)
    k_effres : kNN for effective-resistance

    Returns
    -------
    (n_problems, ~18) feature array and feature names
    """
    if pca is None:
        all_points = np.concatenate(trajectories, axis=0)
        n_comp = min(N_PCA, all_points.shape[1], all_points.shape[0])
        pca = PCA(n_components=n_comp, svd_solver="full")
        pca.fit(all_points)

    feats_list = []
    names = None

    for traj in trajectories:
        reduced = pca.transform(traj)
        f, n = extract_noneuclid_features_single(reduced, k_effres=k_effres)
        feats_list.append(f)
        if names is None:
            names = n

    return np.stack(feats_list), names
