"""Compute topological features from hidden-state point clouds."""

from __future__ import annotations

import logging

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

from topo_confidence.utils import persistence_entropy, subsample_points

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "H0_persistence_entropy",
    "H1_max_lifetime",
    "H0_total_persistence",
    "H0_n_features",
    "H1_persistence_entropy",
    "H1_n_features",
]


class TopologicalFeatureExtractor:
    """Compute persistent homology features from token trajectory point clouds.

    For each problem, the token-by-token hidden states at the final layer
    form a point cloud in R^d. We PCA-reduce to n_pca dimensions, optionally
    subsample, compute persistent homology (H0 and H1), and extract 6 scalar
    features that summarize the topological structure.
    """

    def __init__(
        self,
        method: str = "token_trajectory",
        max_dim: int = 1,
        n_pca: int = 30,
        subsample: int = 100,
        seed: int = 42,
    ):
        self.method = method
        self.max_dim = max_dim
        self.n_pca = n_pca
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)
        self._pca: PCA | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)

    def _fit_pca(self, all_points: np.ndarray) -> None:
        """Fit PCA on the union of all token trajectories."""
        n_components = min(self.n_pca, all_points.shape[1], all_points.shape[0])
        self._pca = PCA(n_components=n_components)
        self._pca.fit(all_points)
        logger.info(
            "PCA: %d → %d components (%.1f%% variance)",
            all_points.shape[1],
            n_components,
            self._pca.explained_variance_ratio_.sum() * 100,
        )

    def _reduce(self, points: np.ndarray) -> np.ndarray:
        """PCA-reduce a point cloud."""
        if self._pca is None:
            raise RuntimeError("PCA not fitted. Call extract() first.")
        return self._pca.transform(points)

    def _compute_ph(self, points: np.ndarray) -> dict[int, np.ndarray]:
        """Run Ripser on a point cloud and return persistence diagrams."""
        points = subsample_points(points, self.subsample, self.rng)
        if len(points) < 3:
            return {0: np.empty((0, 2)), 1: np.empty((0, 2))}
        result = ripser(points, maxdim=self.max_dim)
        return {dim: result["dgms"][dim] for dim in range(self.max_dim + 1)}

    def _features_from_diagrams(self, diagrams: dict[int, np.ndarray]) -> np.ndarray:
        """Extract 6 scalar features from H0 and H1 diagrams."""
        features = np.zeros(6, dtype=np.float64)

        # H0 features
        h0 = diagrams.get(0, np.empty((0, 2)))
        h0_finite = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else np.empty((0, 2))
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([])

        features[0] = persistence_entropy(h0_lifetimes)  # H0_persistence_entropy
        features[2] = h0_lifetimes.sum() if len(h0_lifetimes) > 0 else 0.0  # H0_total_persistence
        features[3] = len(h0_lifetimes)  # H0_n_features

        # H1 features
        h1 = diagrams.get(1, np.empty((0, 2)))
        h1_finite = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
        h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([])

        features[1] = h1_lifetimes.max() if len(h1_lifetimes) > 0 else 0.0  # H1_max_lifetime
        features[4] = persistence_entropy(h1_lifetimes)  # H1_persistence_entropy
        features[5] = len(h1_lifetimes)  # H1_n_features

        return features

    def extract(
        self,
        hidden_states: np.ndarray | None = None,
        token_trajectories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute topological feature vector per problem.

        Args:
            hidden_states: (n_problems, hidden_dim) — unused for token_trajectory method
                but kept for API compatibility.
            token_trajectories: list of (n_tokens_i, hidden_dim) arrays, one per problem.
                Required for token_trajectory method.

        Returns:
            (n_problems, 6) feature array.
        """
        if self.method != "token_trajectory":
            raise NotImplementedError(f"Method {self.method!r} not implemented")

        if token_trajectories is None:
            raise ValueError("token_trajectories required for token_trajectory method")

        # Fit PCA on all tokens pooled together
        all_points = np.concatenate(token_trajectories, axis=0)
        self._fit_pca(all_points)

        # Extract features per problem
        features = np.zeros((len(token_trajectories), self.n_features))
        for i, traj in enumerate(token_trajectories):
            reduced = self._reduce(traj)
            diagrams = self._compute_ph(reduced)
            features[i] = self._features_from_diagrams(diagrams)

        return features

    def extract_single(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract features for a single problem (PCA must already be fitted).

        Args:
            trajectory: (n_tokens, hidden_dim) array.

        Returns:
            (6,) feature vector.
        """
        reduced = self._reduce(trajectory)
        diagrams = self._compute_ph(reduced)
        return self._features_from_diagrams(diagrams)
