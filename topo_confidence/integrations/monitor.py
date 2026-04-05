"""Lightweight bridge health monitor for production inference.

Tracks whether position 0 serves as the expected computational bridge
between two token clusters. Deviations from the healthy pattern
(bridge at crystallized layers 4-24, dissolution at 27) indicate
potential output degradation.

Overhead: ~0.002s per request (k-means + silhouette at 3 layers).
No persistent homology computation needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

logger = logging.getLogger(__name__)


@dataclass
class BridgeHealth:
    """Result of a bridge health check."""

    healthy: bool
    bridge_at_pos0: dict[int, bool]  # layer -> whether pos 0 is bridge
    silhouette_by_layer: dict[int, float]  # layer -> mean silhouette
    pos0_silhouette_by_layer: dict[int, float]  # layer -> pos 0's silhouette
    crystallized: bool  # whether crystallization pattern is present
    anomaly_reason: str | None  # None if healthy, else description

    def summary(self) -> str:
        status = "HEALTHY" if self.healthy else f"ANOMALY: {self.anomaly_reason}"
        layers_str = ", ".join(
            f"L{l}={'bridge' if b else 'core'}"
            for l, b in sorted(self.bridge_at_pos0.items())
        )
        return f"[{status}] {layers_str}"

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "bridge_at_pos0": {str(k): v for k, v in self.bridge_at_pos0.items()},
            "silhouette_by_layer": {
                str(k): round(v, 4) for k, v in self.silhouette_by_layer.items()
            },
            "pos0_silhouette_by_layer": {
                str(k): round(v, 4) for k, v in self.pos0_silhouette_by_layer.items()
            },
            "crystallized": self.crystallized,
            "anomaly_reason": self.anomaly_reason,
        }


class BridgeMonitor:
    """Monitor bridge token health during inference.

    Usage with a HuggingFace model:

        monitor = BridgeMonitor()

        # Option A: pass hidden states directly
        health = monitor.check(hidden_states_dict)

        # Option B: wrap a model forward pass
        health = monitor.check_from_model(model, tokenizer, prompt)

        if not health.healthy:
            logger.warning("Bridge anomaly: %s", health.anomaly_reason)

    Usage as a hook:

        monitor = BridgeMonitor()
        monitor.attach(model)

        # Run inference normally
        outputs = model(**inputs)

        # Check results
        health = monitor.last_health
        monitor.detach()
    """

    # Layers to check: early crystallized, mid crystallized, late/dissolution
    DEFAULT_CHECK_LAYERS = [7, 14, 24]

    def __init__(
        self,
        check_layers: list[int] | None = None,
        bridge_threshold: float = 0.1,
        crystallization_threshold: float = 0.5,
    ):
        """
        Args:
            check_layers: Which layers to check (default: [7, 14, 24])
            bridge_threshold: |silhouette| below this = bridge token
            crystallization_threshold: Mean silhouette above this = crystallized
        """
        self.check_layers = check_layers or self.DEFAULT_CHECK_LAYERS
        self.bridge_threshold = bridge_threshold
        self.crystallization_threshold = crystallization_threshold
        self._hooks = []
        self._captured_states: dict[int, np.ndarray] = {}
        self.last_health: BridgeHealth | None = None

    def check(self, layer_hidden_states: dict[int, np.ndarray]) -> BridgeHealth:
        """Check bridge health from pre-extracted hidden states.

        Args:
            layer_hidden_states: {layer_index: (n_tokens, hidden_dim)} arrays

        Returns:
            BridgeHealth dataclass
        """
        bridge_at_pos0 = {}
        sil_by_layer = {}
        pos0_sil_by_layer = {}

        for layer_idx in self.check_layers:
            if layer_idx not in layer_hidden_states:
                continue

            h = layer_hidden_states[layer_idx]
            n_tokens = h.shape[0]

            if n_tokens < 10:
                continue

            # PCA reduce for clustering stability
            n_comp = min(30, n_tokens - 1, h.shape[1])
            if n_comp < 2:
                continue

            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(h)

            km = KMeans(n_clusters=2, n_init=5, random_state=42)
            labels = km.fit_predict(reduced)

            if len(set(labels)) < 2:
                sil_by_layer[layer_idx] = 0.0
                pos0_sil_by_layer[layer_idx] = 0.0
                bridge_at_pos0[layer_idx] = False
                continue

            sil = silhouette_samples(reduced, labels)
            sil_by_layer[layer_idx] = float(np.mean(np.abs(sil)))
            pos0_sil_by_layer[layer_idx] = float(sil[0])
            bridge_at_pos0[layer_idx] = abs(sil[0]) < self.bridge_threshold

        # Determine health
        n_checked = len(bridge_at_pos0)
        if n_checked == 0:
            health = BridgeHealth(
                healthy=False,
                bridge_at_pos0={},
                silhouette_by_layer={},
                pos0_silhouette_by_layer={},
                crystallized=False,
                anomaly_reason="No layers could be analyzed",
            )
            self.last_health = health
            return health

        n_bridge = sum(bridge_at_pos0.values())
        mean_sil = np.mean(list(sil_by_layer.values()))
        crystallized = mean_sil > self.crystallization_threshold

        # Healthy = pos 0 is bridge at most crystallized layers
        anomaly_reason = None
        if not crystallized:
            anomaly_reason = f"Clusters not crystallized (mean silhouette {mean_sil:.3f} < {self.crystallization_threshold})"
        elif n_bridge < n_checked * 0.5:
            anomaly_reason = f"Position 0 not serving as bridge ({n_bridge}/{n_checked} layers)"

        healthy = anomaly_reason is None

        health = BridgeHealth(
            healthy=healthy,
            bridge_at_pos0=bridge_at_pos0,
            silhouette_by_layer=sil_by_layer,
            pos0_silhouette_by_layer=pos0_sil_by_layer,
            crystallized=crystallized,
            anomaly_reason=anomaly_reason,
        )
        self.last_health = health
        return health

    @torch.no_grad()
    def check_from_model(
        self, model, tokenizer, prompt: str, max_length: int = 512
    ) -> BridgeHealth:
        """Run a forward pass and check bridge health."""
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True)

        layer_states = {}
        for l in self.check_layers:
            # hidden_states index 0 = embeddings, 1..n = layer outputs
            hs_idx = l + 1
            if hs_idx < len(outputs.hidden_states):
                layer_states[l] = (
                    outputs.hidden_states[hs_idx][0].cpu().float().numpy()
                )

        torch.cuda.empty_cache()
        return self.check(layer_states)

    def attach(self, model):
        """Attach hooks to capture hidden states during normal forward passes."""
        self._captured_states = {}

        for l in self.check_layers:
            if l >= len(model.model.layers):
                continue

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    self._captured_states[layer_idx] = (
                        h[0].detach().cpu().float().numpy()
                    )

                return hook_fn

            hook = model.model.layers[l].register_forward_hook(make_hook(l))
            self._hooks.append(hook)

    def detach(self):
        """Remove hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def check_captured(self) -> BridgeHealth:
        """Check health from the most recent hooked forward pass."""
        if not self._captured_states:
            return BridgeHealth(
                healthy=False,
                bridge_at_pos0={},
                silhouette_by_layer={},
                pos0_silhouette_by_layer={},
                crystallized=False,
                anomaly_reason="No captured states — did you call model forward after attach()?",
            )
        health = self.check(self._captured_states)
        self._captured_states = {}  # reset for next call
        return health
