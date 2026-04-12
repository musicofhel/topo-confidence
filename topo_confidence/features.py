"""Compute topological features from hidden-state point clouds."""

from __future__ import annotations

import logging

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

from topo_confidence.null import ph_significance
from topo_confidence.utils import persistence_entropy, subsample_points

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "H0_persistence_entropy",
    "H1_max_lifetime",
    "H0_total_persistence",
    "H0_n_features",
    "H1_persistence_entropy",
    "H1_n_features",
    "H2_n_features",
    "H2_total_persistence",
    "H2_persistence_entropy",
    "bridge_silhouette",
    "H0_ph_significance",
    "H1_ph_significance",
    "topological_sensitivity",
]


class TopologicalFeatureExtractor:
    """Compute persistent homology features from token trajectory point clouds.

    For each problem, the token-by-token hidden states at the final layer
    form a point cloud in R^d. We PCA-reduce to n_pca dimensions, optionally
    subsample, compute persistent homology (H0, H1, H2), and extract scalar
    features that summarize the topological structure.
    """

    def __init__(
        self,
        method: str = "token_trajectory",
        max_dim: int = 2,
        n_pca: int = 30,
        subsample: int = 100,
        null_k: int = 100,
        seed: int = 42,
    ):
        self.method = method
        self.max_dim = max_dim
        self.n_pca = n_pca
        self.subsample = subsample
        self.null_k = null_k
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

    def _compute_bridge_silhouette(self, reduced: np.ndarray) -> float:
        """Compute silhouette coefficient of position 0 in k=2 clustering.

        Position 0 is the singleton computational bridge during layers 4-24.
        Its geometry relative to the two clusters is a strong correctness signal.
        """
        if reduced.shape[0] < 10:
            return 0.0

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples

        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)

        if len(set(labels)) < 2:
            return 0.0

        sil_samples = silhouette_samples(reduced, labels)
        return float(sil_samples[0])  # position 0

    def _features_from_diagrams(self, diagrams: dict[int, np.ndarray]) -> np.ndarray:
        """Extract 9 scalar PH features from H0, H1, and H2 diagrams."""
        features = np.zeros(9, dtype=np.float64)

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

        # H2 features (cavities/voids — Varley et al. 2025 showed correlation
        # with information-theoretic synergy at rho = -0.55 to -0.65)
        h2 = diagrams.get(2, np.empty((0, 2)))
        h2_finite = h2[np.isfinite(h2[:, 1])] if len(h2) > 0 else np.empty((0, 2))
        h2_lifetimes = h2_finite[:, 1] - h2_finite[:, 0] if len(h2_finite) > 0 else np.array([])

        features[6] = len(h2_lifetimes)  # H2_n_features
        features[7] = h2_lifetimes.sum() if len(h2_lifetimes) > 0 else 0.0  # H2_total_persistence
        features[8] = persistence_entropy(h2_lifetimes)  # H2_persistence_entropy

        return features

    def _compute_topological_sensitivity(self, points: np.ndarray) -> float:
        """Compute sensitivity of H1 total persistence to Gaussian noise.

        Adds noise at epsilon in {0.01, 0.05, 0.1}, recomputes PH, and
        returns the OLS slope of total_H1_persistence vs epsilon.
        Steep slope = fragile topology = likely incorrect answer.
        Inspired by anti-stability in QEC (Machine 4).
        """
        epsilons = [0.0, 0.01, 0.05, 0.1]
        rng = np.random.default_rng(42)
        persistences = []
        for eps in epsilons:
            if eps == 0.0:
                noisy = points
            else:
                noisy = points + rng.normal(0, eps, size=points.shape)
            result = ripser(noisy, maxdim=1)
            h1 = result["dgms"][1]
            if len(h1) > 0:
                lifetimes = h1[:, 1] - h1[:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                persistences.append(lifetimes.sum())
            else:
                persistences.append(0.0)

        # OLS slope: persistence = a + b * epsilon
        eps_arr = np.array(epsilons)
        pers_arr = np.array(persistences)
        var_eps = np.var(eps_arr)
        if var_eps < 1e-15:
            return 0.0
        slope = np.cov(eps_arr, pers_arr)[0, 1] / var_eps
        return float(slope)

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
            (n_problems, n_features) feature array.
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
            subsampled = subsample_points(reduced, self.subsample, self.rng)
            diagrams = self._compute_ph(reduced)
            ph_features = self._features_from_diagrams(diagrams)
            bridge_sil = self._compute_bridge_silhouette(reduced)

            # Null significance z-scores (Machine 6: shuffled-token null)
            if self.null_k > 0 and len(subsampled) >= 3:
                z_scores = ph_significance(
                    subsampled, max_dim=min(self.max_dim, 1), k=self.null_k,
                    seed=42 + i,
                )
                h0_sig = z_scores.get(0, 0.0)
                h1_sig = z_scores.get(1, 0.0)
            else:
                h0_sig = 0.0
                h1_sig = 0.0

            # Perturbation sensitivity (Machine 4: Stability)
            sensitivity = self._compute_topological_sensitivity(subsampled)

            features[i] = np.concatenate([
                ph_features, [bridge_sil, h0_sig, h1_sig, sensitivity],
            ])

        return features

    def extract_single(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract features for a single problem (PCA must already be fitted).

        Args:
            trajectory: (n_tokens, hidden_dim) array.

        Returns:
            (n_features,) feature vector.
        """
        reduced = self._reduce(trajectory)
        subsampled = subsample_points(reduced, self.subsample, self.rng)
        diagrams = self._compute_ph(reduced)
        ph_features = self._features_from_diagrams(diagrams)
        bridge_sil = self._compute_bridge_silhouette(reduced)

        if self.null_k > 0 and len(subsampled) >= 3:
            z_scores = ph_significance(
                subsampled, max_dim=min(self.max_dim, 1), k=self.null_k,
                seed=42,
            )
            h0_sig = z_scores.get(0, 0.0)
            h1_sig = z_scores.get(1, 0.0)
        else:
            h0_sig = 0.0
            h1_sig = 0.0

        sensitivity = self._compute_topological_sensitivity(subsampled)
        return np.concatenate([ph_features, [bridge_sil, h0_sig, h1_sig, sensitivity]])
