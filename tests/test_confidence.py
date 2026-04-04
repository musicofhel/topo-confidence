"""Tests for topo-confidence core components.

These tests use synthetic data to verify the pipeline without
requiring a GPU or downloading model weights.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from topo_confidence.baselines import (
    first_token_probability,
    max_token_probability,
    output_entropy,
)
from topo_confidence.features import TopologicalFeatureExtractor
from topo_confidence.utils import persistence_entropy, subsample_points


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class TestPersistenceEntropy:
    def test_empty(self):
        assert persistence_entropy(np.array([])) == 0.0

    def test_single(self):
        # Single feature → entropy = 0 (p=1, log(1)=0)
        assert persistence_entropy(np.array([1.0])) == pytest.approx(0.0, abs=1e-10)

    def test_uniform(self):
        # N equal lifetimes → entropy = log(N)
        n = 10
        lifetimes = np.ones(n)
        expected = np.log(n)
        assert persistence_entropy(lifetimes) == pytest.approx(expected, rel=1e-6)

    def test_skewed(self):
        # One dominant + many small → low entropy
        lifetimes = np.array([100.0, 0.1, 0.1, 0.1])
        ent = persistence_entropy(lifetimes)
        # Compare to uniform case
        uniform_ent = persistence_entropy(np.ones(4))
        assert ent < uniform_ent

    def test_zeros_ignored(self):
        lifetimes = np.array([0.0, 0.0, 1.0])
        assert persistence_entropy(lifetimes) == pytest.approx(0.0, abs=1e-10)


class TestSubsample:
    def test_no_subsample_needed(self):
        rng = np.random.default_rng(42)
        points = rng.random((50, 10))
        result = subsample_points(points, 100, rng)
        assert result.shape == (50, 10)

    def test_subsample(self):
        rng = np.random.default_rng(42)
        points = rng.random((200, 10))
        result = subsample_points(points, 50, rng)
        assert result.shape == (50, 10)

    def test_deterministic(self):
        points = np.random.default_rng(0).random((200, 10))
        r1 = subsample_points(points, 50, np.random.default_rng(42))
        r2 = subsample_points(points, 50, np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class TestTopologicalFeatureExtractor:
    def _make_trajectories(self, n_problems: int = 20, n_tokens: int = 50,
                           hidden_dim: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        return [rng.random((n_tokens, hidden_dim)) for _ in range(n_problems)]

    def test_output_shape(self):
        trajectories = self._make_trajectories()
        ext = TopologicalFeatureExtractor(n_pca=10)
        features = ext.extract(token_trajectories=trajectories)
        assert features.shape == (20, 6)

    def test_feature_names(self):
        ext = TopologicalFeatureExtractor()
        assert len(ext.feature_names) == 6
        assert ext.feature_names[0] == "H0_persistence_entropy"

    def test_no_nans(self):
        trajectories = self._make_trajectories()
        ext = TopologicalFeatureExtractor(n_pca=10)
        features = ext.extract(token_trajectories=trajectories)
        assert not np.any(np.isnan(features))

    def test_different_data_different_features(self):
        ext = TopologicalFeatureExtractor(n_pca=10)

        # Clustered data (simple topology)
        rng = np.random.default_rng(42)
        simple = [rng.normal(0, 0.1, (50, 64)) for _ in range(10)]

        # Random data (complex topology)
        complex_ = [rng.random((50, 64)) for _ in range(10)]

        f_simple = ext.extract(token_trajectories=simple)

        # Need to re-create extractor since PCA is fitted
        ext2 = TopologicalFeatureExtractor(n_pca=10)
        f_complex = ext2.extract(token_trajectories=complex_)

        # H0 persistence entropy should differ
        assert not np.allclose(f_simple[:, 0], f_complex[:, 0])

    def test_requires_trajectories(self):
        ext = TopologicalFeatureExtractor()
        with pytest.raises(ValueError, match="token_trajectories required"):
            ext.extract(hidden_states=np.zeros((5, 64)))

    def test_short_trajectory(self):
        """Trajectories with very few tokens should not crash."""
        trajectories = [np.random.default_rng(42).random((3, 64)) for _ in range(5)]
        ext = TopologicalFeatureExtractor(n_pca=2)
        features = ext.extract(token_trajectories=trajectories)
        assert features.shape == (5, 6)

    def test_extract_single(self):
        trajectories = self._make_trajectories(n_problems=10)
        ext = TopologicalFeatureExtractor(n_pca=10)
        batch_features = ext.extract(token_trajectories=trajectories)

        # extract_single should give same result after PCA is fitted
        single = ext.extract_single(trajectories[0])
        # Note: not exactly equal because PCA is refit each time in extract(),
        # but extract_single reuses the fitted PCA
        assert single.shape == (6,)
        assert not np.any(np.isnan(single))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def _make_logits(self, n: int = 10, seq_len: int = 20, vocab: int = 100):
        rng = np.random.default_rng(42)
        return [rng.random((seq_len, vocab)) for _ in range(n)]

    def test_output_entropy_shape(self):
        logits = self._make_logits()
        ent = output_entropy(logits)
        assert ent.shape == (10,)
        assert np.all(ent >= 0)

    def test_max_token_probability_shape(self):
        logits = self._make_logits()
        probs = max_token_probability(logits)
        assert probs.shape == (10,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_first_token_probability_shape(self):
        logits = self._make_logits()
        probs = first_token_probability(logits)
        assert probs.shape == (10,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_empty_logits(self):
        logits = [np.array([])]
        assert output_entropy(logits).shape == (1,)
        assert max_token_probability(logits).shape == (1,)
        assert first_token_probability(logits).shape == (1,)

    def test_confident_vs_uncertain(self):
        # Very peaked logits (confident)
        confident = np.zeros((10, 50))
        confident[:, 0] = 100.0  # softmax will be ~1.0 for token 0

        # Flat logits (uncertain)
        uncertain = np.zeros((10, 50))

        conf_ent = output_entropy([confident])[0]
        unc_ent = output_entropy([uncertain])[0]
        assert conf_ent < unc_ent

        conf_prob = max_token_probability([confident])[0]
        unc_prob = max_token_probability([uncertain])[0]
        assert conf_prob > unc_prob


# ---------------------------------------------------------------------------
# integration test (synthetic, no model)
# ---------------------------------------------------------------------------

class TestSyntheticPipeline:
    """End-to-end test with synthetic data that simulates the full pipeline."""

    def test_separable_data_gives_high_auroc(self):
        """If correct/incorrect problems have genuinely different topology,
        the pipeline should detect it."""
        rng = np.random.default_rng(42)
        n = 100

        # Correct problems: tight clusters (low entropy)
        correct_trajs = [rng.normal(0, 0.3, (50, 32)) for _ in range(n // 2)]

        # Incorrect problems: spread out (high entropy)
        incorrect_trajs = [rng.uniform(-10, 10, (50, 32)) for _ in range(n // 2)]

        trajectories = correct_trajs + incorrect_trajs
        labels = np.array([1] * (n // 2) + [0] * (n // 2))

        ext = TopologicalFeatureExtractor(n_pca=10, subsample=50)
        features = ext.extract(token_trajectories=trajectories)

        # H0_persistence_entropy should separate these
        auroc = roc_auc_score(labels, -features[:, 0])  # lower entropy = correct
        assert auroc > 0.7, f"Expected AUROC > 0.7, got {auroc:.3f}"
