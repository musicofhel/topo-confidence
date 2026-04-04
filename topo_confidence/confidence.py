"""Main TopoConfidence class: calibrate and predict."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from topo_confidence.extractor import HiddenStateExtractor
from topo_confidence.features import TopologicalFeatureExtractor
from topo_confidence.utils import timer

logger = logging.getLogger(__name__)


class TopoConfidence:
    """Training-free topological confidence estimator for LLM outputs.

    Usage:
        tc = TopoConfidence("Qwen/Qwen2.5-1.5B-Instruct")
        tc.calibrate(calibration_prompts, calibration_correct_labels)
        confidences = tc.predict_confidence(new_prompts)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        layers: list[int] | str = "last",
        calibration_size: int = 100,
    ):
        self.model_name = model_name
        self.extractor = HiddenStateExtractor(
            model_name, device=device, layers=layers
        )
        self.feature_extractor = TopologicalFeatureExtractor()
        self.calibrated = False
        self.calibration_size = calibration_size

        self._scaler: StandardScaler | None = None
        self._classifier: LogisticRegression | None = None
        self._calibrator: IsotonicRegression | None = None
        self._calibration_method: str = "logistic"

    def calibrate(
        self,
        prompts: list[str],
        correct: np.ndarray,
        method: str = "logistic",
    ) -> dict[str, float]:
        """Fit the confidence model on labeled examples.

        Args:
            prompts: problem statements.
            correct: boolean/int array (1 = model answered correctly).
            method: "logistic" or "isotonic".

        Returns:
            dict with calibration metrics: auroc, accuracy, brier_score, n_samples.
        """
        correct = np.asarray(correct, dtype=int)
        self._calibration_method = method

        with timer("Extracting hidden states"):
            result = self.extractor.extract(prompts)

        with timer("Computing topological features"):
            features = self.feature_extractor.extract(
                hidden_states=result["hidden_states"],
                token_trajectories=result["token_trajectories"],
            )

        with timer("Fitting classifier"):
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(features)

            self._classifier = LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            )
            self._classifier.fit(X, correct)

            # Cross-validated predictions for honest metrics
            n_cv = min(10, len(prompts))
            if n_cv >= 2 and len(np.unique(correct)) == 2:
                cv_probs = cross_val_predict(
                    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                    X, correct,
                    cv=n_cv,
                    method="predict_proba",
                )[:, 1]
            else:
                cv_probs = self._classifier.predict_proba(X)[:, 1]

            if method == "isotonic":
                self._calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                self._calibrator.fit(cv_probs, correct)

        self.calibrated = True

        # Compute metrics on cross-validated predictions
        metrics = {"n_samples": len(prompts)}
        if len(np.unique(correct)) == 2:
            metrics["auroc"] = float(roc_auc_score(correct, cv_probs))
            metrics["brier_score"] = float(brier_score_loss(correct, cv_probs))
            pred_labels = (cv_probs >= 0.5).astype(int)
            metrics["accuracy"] = float((pred_labels == correct).mean())
        else:
            metrics["auroc"] = float("nan")
            metrics["brier_score"] = float("nan")
            metrics["accuracy"] = float((correct == correct[0]).mean())

        # Log feature importances
        if self._classifier is not None:
            coefs = self._classifier.coef_[0]
            for name, coef in zip(self.feature_extractor.feature_names, coefs):
                logger.info("  %s: %.4f", name, coef)

        logger.info("Calibration AUROC: %.3f", metrics.get("auroc", float("nan")))
        return metrics

    def calibrate_from_features(
        self,
        features: np.ndarray,
        correct: np.ndarray,
        method: str = "logistic",
    ) -> dict[str, float]:
        """Fit the classifier directly from pre-computed features.

        Useful when features have already been extracted (e.g., in experiments).
        Does NOT fit PCA — the feature extractor must already be fitted.
        """
        correct = np.asarray(correct, dtype=int)
        self._calibration_method = method

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(features)

        self._classifier = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
        self._classifier.fit(X, correct)

        n_cv = min(10, len(features))
        if n_cv >= 2 and len(np.unique(correct)) == 2:
            cv_probs = cross_val_predict(
                LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                X, correct,
                cv=n_cv,
                method="predict_proba",
            )[:, 1]
        else:
            cv_probs = self._classifier.predict_proba(X)[:, 1]

        if method == "isotonic":
            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._calibrator.fit(cv_probs, correct)

        self.calibrated = True

        metrics = {"n_samples": len(features)}
        if len(np.unique(correct)) == 2:
            metrics["auroc"] = float(roc_auc_score(correct, cv_probs))
            metrics["brier_score"] = float(brier_score_loss(correct, cv_probs))
            pred_labels = (cv_probs >= 0.5).astype(int)
            metrics["accuracy"] = float((pred_labels == correct).mean())
        else:
            metrics["auroc"] = float("nan")

        return metrics

    def predict_confidence(self, prompts: list[str]) -> np.ndarray:
        """Predict P(correct) for each prompt.

        Returns:
            float array in [0, 1], shape (n_prompts,).
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict_confidence()")

        result = self.extractor.extract(prompts)
        features = np.zeros((len(prompts), self.feature_extractor.n_features))
        for i, traj in enumerate(result["token_trajectories"]):
            features[i] = self.feature_extractor.extract_single(traj)

        return self._predict_from_features(features)

    def _predict_from_features(self, features: np.ndarray) -> np.ndarray:
        """Predict confidence from pre-computed features."""
        X = self._scaler.transform(features)
        probs = self._classifier.predict_proba(X)[:, 1]

        if self._calibrator is not None:
            probs = self._calibrator.predict(probs)

        return probs

    def selective_predict(
        self,
        prompts: list[str],
        threshold: float = 0.7,
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        """Generate answers only for high-confidence problems.

        Returns:
            {
                "answers": list[str | None],
                "confidences": np.ndarray,
                "answered_fraction": float,
                "expected_accuracy_on_answered": float,
            }
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before selective_predict()")

        result = self.extractor.extract_with_output(
            prompts, max_new_tokens=max_new_tokens
        )
        features = np.zeros((len(prompts), self.feature_extractor.n_features))
        for i, traj in enumerate(result["token_trajectories"]):
            features[i] = self.feature_extractor.extract_single(traj)

        confidences = self._predict_from_features(features)
        mask = confidences >= threshold

        answers = []
        for i, text in enumerate(result["generated_texts"]):
            answers.append(text if mask[i] else None)

        return {
            "answers": answers,
            "confidences": confidences,
            "answered_fraction": float(mask.mean()),
            "expected_accuracy_on_answered": float(confidences[mask].mean())
            if mask.any()
            else 0.0,
        }

    def explain(self, prompt: str) -> dict[str, Any]:
        """Show which topological features drove the confidence score."""
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before explain()")

        result = self.extractor.extract([prompt])
        features = self.feature_extractor.extract_single(
            result["token_trajectories"][0]
        )
        X = self._scaler.transform(features.reshape(1, -1))
        confidence = self._classifier.predict_proba(X)[0, 1]

        coefs = self._classifier.coef_[0]
        contributions = X[0] * coefs

        explanation = {
            "confidence": float(confidence),
            "features": {},
            "top_contributor": "",
        }
        max_abs = 0.0
        for name, raw, scaled, coef, contrib in zip(
            self.feature_extractor.feature_names,
            features,
            X[0],
            coefs,
            contributions,
        ):
            explanation["features"][name] = {
                "raw_value": float(raw),
                "scaled_value": float(scaled),
                "coefficient": float(coef),
                "contribution": float(contrib),
            }
            if abs(contrib) > max_abs:
                max_abs = abs(contrib)
                explanation["top_contributor"] = name

        return explanation

    def save(self, path: str | Path) -> None:
        """Save calibrated model to disk."""
        path = Path(path)
        state = {
            "model_name": self.model_name,
            "scaler": self._scaler,
            "classifier": self._classifier,
            "calibrator": self._calibrator,
            "calibration_method": self._calibration_method,
            "pca": self.feature_extractor._pca,
            "feature_config": {
                "method": self.feature_extractor.method,
                "max_dim": self.feature_extractor.max_dim,
                "n_pca": self.feature_extractor.n_pca,
                "subsample": self.feature_extractor.subsample,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved calibrated model to %s", path)

    def load(self, path: str | Path) -> None:
        """Load calibrated model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._scaler = state["scaler"]
        self._classifier = state["classifier"]
        self._calibrator = state["calibrator"]
        self._calibration_method = state["calibration_method"]
        self.feature_extractor._pca = state["pca"]
        self.calibrated = True
        logger.info("Loaded calibrated model from %s", path)
