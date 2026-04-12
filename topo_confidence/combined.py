"""Combined topological + output-based confidence estimation."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from topo_confidence.baselines import max_token_probability, output_entropy
from topo_confidence.confidence import TopoConfidence

logger = logging.getLogger(__name__)


class CombinedConfidence(TopoConfidence):
    """Fuses topological features with output-level uncertainty signals.

    Adds output entropy and max token probability to the topo feature
    vector before fitting the classifier. Since topo features and
    output entropy are orthogonal (r=0.062), the combination captures
    strictly more information than either alone.

    Usage:
        cc = CombinedConfidence("Qwen/Qwen2.5-1.5B-Instruct")
        cc.calibrate(prompts, labels)  # extracts both topo + output features
        confidences = cc.predict_confidence(new_prompts)
    """

    def _extract_combined_features(self, prompts, max_new_tokens=256):
        """Extract topo features + output-based features in one pass."""
        result = self.extractor.extract_with_output(
            prompts, max_new_tokens=max_new_tokens, collect_logits=True
        )

        # Topo features — use batch extract (fits PCA) during calibration,
        # use extract_single (reuses PCA) during prediction
        if self.feature_extractor._pca is None:
            topo_features = self.feature_extractor.extract(
                token_trajectories=result["token_trajectories"]
            )
        else:
            topo_features = np.zeros((len(prompts), self.feature_extractor.n_features))
            for i, traj in enumerate(result["token_trajectories"]):
                topo_features[i] = self.feature_extractor.extract_single(traj)

        # Output-based features
        logits_list = result["output_logits"]
        ent = output_entropy(logits_list).reshape(-1, 1)
        mtp = max_token_probability(logits_list).reshape(-1, 1)

        # Concatenate: topo + 2 output features
        combined = np.hstack([topo_features, ent, mtp])

        return combined, result

    def calibrate(self, prompts, correct, method="logistic"):
        """Calibrate on topo + output features."""
        correct = np.asarray(correct, dtype=int)
        self._calibration_method = method

        combined, _ = self._extract_combined_features(prompts)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(combined)

        self._classifier = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
        self._classifier.fit(X, correct)

        # Cross-validated metrics
        n_cv = min(10, len(prompts))
        if n_cv >= 2 and len(np.unique(correct)) == 2:
            cv_probs = cross_val_predict(
                LogisticRegression(
                    max_iter=1000, random_state=42, class_weight="balanced"
                ),
                X,
                correct,
                cv=n_cv,
                method="predict_proba",
            )[:, 1]
        else:
            cv_probs = self._classifier.predict_proba(X)[:, 1]

        self.calibrated = True

        metrics = {"n_samples": len(prompts), "n_features": combined.shape[1]}
        if len(np.unique(correct)) == 2:
            metrics["auroc"] = float(roc_auc_score(correct, cv_probs))

        logger.info(
            "Combined calibration AUROC: %.3f", metrics.get("auroc", float("nan"))
        )
        return metrics

    def predict_confidence(self, prompts):
        """Predict P(correct) using combined topo + output features."""
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict_confidence()")

        combined, _ = self._extract_combined_features(prompts)
        return self._predict_from_features(combined)
