"""LangChain-compatible evaluator using topological confidence.

Usage:
    from topo_confidence.integrations.langchain import TopoEvaluator

    evaluator = TopoEvaluator(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    # Use with LangChain
    result = evaluator.evaluate_strings(
        prediction="The answer is 42",
        input="What is the meaning of life?",
    )
    print(result)  # {"score": 0.73, "features": {...}, "bridge_healthy": True}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try importing LangChain — graceful fallback if not installed
try:
    from langchain.evaluation import StringEvaluator

    HAS_LANGCHAIN = True
except ImportError:

    class StringEvaluator:
        """Minimal fallback when LangChain is not installed."""

        @property
        def evaluation_name(self) -> str:
            return "topo_confidence"

        @property
        def requires_input(self) -> bool:
            return True

        @property
        def requires_reference(self) -> bool:
            return False

    HAS_LANGCHAIN = False


class TopoEvaluator(StringEvaluator):
    """Evaluate LLM outputs using topological features of hidden states.

    This evaluator runs the input prompt through a local model, extracts
    hidden states, computes topological features, and returns a confidence
    score predicting whether the output is correct.

    Works standalone or as a LangChain evaluator.

    Args:
        model_name: HuggingFace model identifier
        device: "auto", "cuda", or "cpu"
        calibration_path: Path to calibrated .pkl file. If None, uses heuristic scoring.
        monitor_bridge: If True, also runs bridge health check.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        calibration_path: str | None = None,
        monitor_bridge: bool = True,
    ):
        self.model_name = model_name
        self._monitor_bridge = monitor_bridge
        self._tc = None
        self._monitor = None
        self._device = device
        self._calibration_path = calibration_path
        # Lazy load — don't load model until first evaluation

    def _ensure_loaded(self):
        """Lazy-load model and components on first use."""
        if self._tc is not None:
            return

        from topo_confidence import TopoConfidence
        from topo_confidence.integrations.monitor import BridgeMonitor

        self._tc = TopoConfidence(model_name=self.model_name, device=self._device)

        if self._calibration_path:
            self._tc.load(self._calibration_path)

        if self._monitor_bridge:
            self._monitor = BridgeMonitor()

    @property
    def evaluation_name(self) -> str:
        return "topo_confidence"

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return False

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate a single prediction.

        Args:
            prediction: The model's output (not used for scoring — we re-run inference)
            input: The original prompt (required)
            reference: Ground truth (not used)

        Returns:
            dict with "score", "features", and optionally "bridge_health"
        """
        self._ensure_loaded()

        if input is None:
            return {"score": 0.0, "error": "input is required"}

        prompt = input

        # Extract hidden states
        result = self._tc.extractor.extract([prompt])
        traj = result["token_trajectories"][0]

        # Compute topo features
        if self._tc.feature_extractor._pca is None:
            # Fit PCA on this single example (not ideal, but works for one-off)
            _ = self._tc.feature_extractor.extract(token_trajectories=[traj])
        features = self._tc.feature_extractor.extract_single(traj)

        feature_dict = dict(
            zip(self._tc.feature_extractor.feature_names, features.tolist())
        )

        # Confidence score
        if self._tc.calibrated:
            confidence = float(
                self._tc._predict_from_features(features.reshape(1, -1))[0]
            )
        else:
            # Heuristic: lower H0 entropy → higher confidence
            h0_ent = features[0]
            confidence = float(np.clip(1.0 - (h0_ent - 1.5) / 3.0, 0.0, 1.0))

        output = {
            "score": confidence,
            "features": feature_dict,
        }

        # Bridge health check
        if self._monitor is not None:
            health = self._monitor.check_from_model(
                self._tc.extractor.model,
                self._tc.extractor.tokenizer,
                prompt,
            )
            output["bridge_healthy"] = health.healthy
            output["bridge_summary"] = health.summary()

        return output

    # LangChain compatibility
    def evaluate_strings(self, **kwargs) -> dict:
        return self._evaluate_strings(**kwargs)

    # Standalone convenience method
    def score(self, prompt: str) -> dict:
        """Score a single prompt. Convenience method for non-LangChain usage."""
        return self._evaluate_strings(prediction="", input=prompt)

    def score_batch(self, prompts: list[str]) -> list[dict]:
        """Score multiple prompts."""
        return [self.score(p) for p in prompts]
