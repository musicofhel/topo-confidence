#!/usr/bin/env python3
"""Calibrate topo-confidence on a labeled dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from topo_confidence import TopoConfidence
from topo_confidence.utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_calibration_data(path: str) -> tuple[list[str], np.ndarray]:
    """Load calibration data from a JSONL file.

    Expected format: {"prompt": "...", "correct": true/false}
    """
    prompts = []
    correct = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item["prompt"])
            correct.append(int(item["correct"]))
    return prompts, np.array(correct)


def main():
    parser = argparse.ArgumentParser(description="Calibrate topo-confidence")
    parser.add_argument("data", help="Path to calibration JSONL file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/calibrated_model.pkl")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--device", default=None, help="Override device")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model_name = args.model or config["model"]["name"]
    device = args.device or config["model"]["device"]

    prompts, correct = load_calibration_data(args.data)
    logger.info("Loaded %d calibration examples (%.1f%% correct)",
                len(prompts), correct.mean() * 100)

    tc = TopoConfidence(model_name=model_name, device=device)
    metrics = tc.calibrate(prompts, correct, method=config["calibration"]["method"])

    logger.info("Calibration results:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tc.save(args.output)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
