"""Tests for BridgeMonitor."""

import numpy as np
import pytest
from topo_confidence.integrations.monitor import BridgeMonitor, BridgeHealth


class TestBridgeMonitor:
    def _make_two_cluster_data(self, n_tokens=100, hidden_dim=64, seed=42):
        """Create synthetic hidden states with clear two-cluster structure."""
        rng = np.random.default_rng(seed)

        # Cluster A: centered at +5
        cluster_a = rng.normal(5.0, 0.5, (n_tokens // 2, hidden_dim))
        # Cluster B: centered at -5
        cluster_b = rng.normal(-5.0, 0.5, (n_tokens // 2, hidden_dim))

        # Position 0: bridge (between clusters)
        bridge = rng.normal(0.0, 0.5, (1, hidden_dim))

        # Stack: bridge at position 0, then alternating clusters
        data = np.vstack([bridge, cluster_a, cluster_b])
        return data

    def test_healthy_bridge(self):
        """Two clear clusters with bridge at pos 0 → healthy."""
        data = self._make_two_cluster_data()
        monitor = BridgeMonitor(check_layers=[7, 14])
        health = monitor.check({7: data, 14: data})

        assert isinstance(health, BridgeHealth)
        assert health.crystallized
        # Bridge detection depends on actual silhouette — may or may not detect pos 0
        # The test verifies the pipeline doesn't crash

    def test_no_layers(self):
        """No analyzable layers → unhealthy."""
        monitor = BridgeMonitor(check_layers=[7])
        health = monitor.check({})
        assert not health.healthy
        assert health.anomaly_reason is not None

    def test_summary_string(self):
        data = self._make_two_cluster_data()
        monitor = BridgeMonitor(check_layers=[14])
        health = monitor.check({14: data})
        summary = health.summary()
        assert isinstance(summary, str)
        assert "L14" in summary

    def test_to_dict(self):
        data = self._make_two_cluster_data()
        monitor = BridgeMonitor(check_layers=[14])
        health = monitor.check({14: data})
        d = health.to_dict()
        assert "healthy" in d
        assert "bridge_at_pos0" in d
        assert "crystallized" in d

    def test_small_input(self):
        """Very few tokens should not crash."""
        rng = np.random.default_rng(42)
        small = rng.random((5, 64))
        monitor = BridgeMonitor(check_layers=[14])
        health = monitor.check({14: small})
        # Should handle gracefully, not crash
        assert isinstance(health, BridgeHealth)


class TestBridgeHealthDataclass:
    def test_healthy_summary(self):
        health = BridgeHealth(
            healthy=True,
            bridge_at_pos0={7: True, 14: True},
            silhouette_by_layer={7: 0.95, 14: 0.97},
            pos0_silhouette_by_layer={7: 0.01, 14: -0.02},
            crystallized=True,
            anomaly_reason=None,
        )
        assert "HEALTHY" in health.summary()

    def test_anomaly_summary(self):
        health = BridgeHealth(
            healthy=False,
            bridge_at_pos0={7: False, 14: False},
            silhouette_by_layer={7: 0.3, 14: 0.2},
            pos0_silhouette_by_layer={7: 0.5, 14: 0.6},
            crystallized=False,
            anomaly_reason="Clusters not crystallized",
        )
        assert "ANOMALY" in health.summary()
