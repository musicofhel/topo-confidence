"""Null hypothesis testing for persistent homology features.

Implements shuffled-token null distribution (Machine 6 from topoyolo atlas).
Reference: Vejdemo-Johansson & Mukherjee (2018, arXiv: 1812.06491) —
universal empirical null for PH with FWER/FDR control.
"""

from __future__ import annotations

import numpy as np
from ripser import ripser


def shuffled_token_null(
    points: np.ndarray,
    max_dim: int = 1,
    k: int = 100,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Compute null distribution of total persistence via token-position shuffling.

    Randomly permutes token positions (rows) in the point cloud. This preserves
    each token's individual vector but destroys the spatial arrangement that PH
    measures. Generates K shuffled copies and computes total persistence for each.

    Parameters
    ----------
    points : array, shape (n_tokens, hidden_dim)
        The PCA-reduced point cloud.
    max_dim : int
        Maximum homology dimension to compute.
    k : int
        Number of shuffled copies to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict mapping dimension -> array of shape (k,) with null total persistence values.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    null_persistence = {dim: np.zeros(k) for dim in range(max_dim + 1)}

    for i in range(k):
        # Shuffle row indices — preserves per-token vectors, destroys arrangement
        perm = rng.permutation(n)
        shuffled = points[perm]
        result = ripser(shuffled, maxdim=max_dim)
        for dim in range(max_dim + 1):
            dgm = result["dgms"][dim]
            if len(dgm) > 0:
                lifetimes = dgm[:, 1] - dgm[:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                null_persistence[dim][i] = lifetimes.sum()

    return null_persistence


def ph_significance(
    points: np.ndarray,
    max_dim: int = 1,
    k: int = 100,
    seed: int = 42,
) -> dict[int, float]:
    """Compute z-score significance of PH features against shuffled null.

    For each homology dimension, computes:
        z = (real_total_persistence - mean(null)) / std(null)

    Values > 2 indicate genuine topological structure (p < 0.025 one-sided).
    Values near 0 mean features are artifacts of point density.

    Parameters
    ----------
    points : array, shape (n_tokens, hidden_dim)
        The PCA-reduced point cloud.
    max_dim : int
        Maximum homology dimension.
    k : int
        Number of null shuffles.
    seed : int
        Random seed.

    Returns
    -------
    dict mapping dimension -> z-score float.
    """
    # Compute real PH
    result = ripser(points, maxdim=max_dim)
    real_persistence = {}
    for dim in range(max_dim + 1):
        dgm = result["dgms"][dim]
        if len(dgm) > 0:
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            real_persistence[dim] = lifetimes.sum()
        else:
            real_persistence[dim] = 0.0

    # Compute null distribution
    null = shuffled_token_null(points, max_dim=max_dim, k=k, seed=seed)

    # Z-scores
    z_scores = {}
    for dim in range(max_dim + 1):
        null_vals = null[dim]
        std = null_vals.std()
        if std < 1e-12:
            z_scores[dim] = 0.0
        else:
            z_scores[dim] = (real_persistence[dim] - null_vals.mean()) / std

    return z_scores
