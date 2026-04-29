#!/usr/bin/env python3
"""Evaluate topo-confidence on a test set."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

from topo_confidence import TopoConfidence
from topo_confidence.utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(path: str) -> tuple[list[str], np.ndarray]:
    """Load test data from JSONL. Same format as calibration."""
    prompts = []
    correct = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item["prompt"])
            correct.append(int(item["correct"]))
    return prompts, np.array(correct)


def main():
    parser = argparse.ArgumentParser(description="Evaluate topo-confidence")
    parser.add_argument("data", help="Path to test JSONL file")
    parser.add_argument("--model-path", default="data/calibrated_model.pkl",
                        help="Path to calibrated model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model_name = args.model or config["model"]["name"]
    device = args.device or config["model"]["device"]

    tc = TopoConfidence(model_name=model_name, device=device)
    tc.load(args.model_path)

    prompts, correct = load_test_data(args.data)
    logger.info("Loaded %d test examples", len(prompts))

    confidences = tc.predict_confidence(prompts)

    auroc = roc_auc_score(correct, confidences)
    brier = brier_score_loss(correct, confidences)
    logger.info("Test AUROC: %.3f", auroc)
    logger.info("Test Brier: %.3f", brier)

    # Selective prediction at various thresholds
    thresholds = config["evaluation"]["confidence_thresholds"]
    logger.info("\nSelective prediction:")
    logger.info("%-10s %-12s %-20s", "Threshold", "Answered %", "Accuracy on Answered")
    for t in thresholds:
        mask = confidences >= t
        answered = mask.mean()
        acc = correct[mask].mean() if mask.any() else 0.0
        logger.info("%-10.2f %-12.1f%% %-20.3f", t, answered * 100, acc)


if __name__ == "__main__":
    main()
