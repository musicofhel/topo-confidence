#!/usr/bin/env python3
"""Benchmark topo-confidence against baselines across datasets.

This script runs Experiment 4: compare topo-confidence to output entropy,
max token probability, and first-token probability baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from topo_confidence import TopoConfidence
from topo_confidence.baselines import (
    first_token_probability,
    max_token_probability,
    output_entropy,
)
from topo_confidence.utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str) -> tuple[list[str], np.ndarray]:
    prompts, correct = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item["prompt"])
            correct.append(int(item["correct"]))
    return prompts, np.array(correct)


def main():
    parser = argparse.ArgumentParser(description="Benchmark vs baselines")
    parser.add_argument("data", help="Path to test JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model_name = args.model or config["model"]["name"]
    device = args.device or config["model"]["device"]

    prompts, correct = load_data(args.data)
    logger.info("Loaded %d examples (%.1f%% correct)", len(prompts), correct.mean() * 100)

    tc = TopoConfidence(model_name=model_name, device=device)

    # Extract hidden states + generate outputs (one pass)
    logger.info("Extracting hidden states and generating outputs...")
    ext_result = tc.extractor.extract_with_output(prompts)

    # Compute topo features
    logger.info("Computing topological features...")
    features = tc.feature_extractor.extract(
        hidden_states=ext_result["hidden_states"],
        token_trajectories=ext_result["token_trajectories"],
    )
    cal_metrics = tc.calibrate_from_features(features, correct)

    topo_conf = tc._predict_from_features(features)
    topo_auroc = roc_auc_score(correct, topo_conf)

    # Compute baselines
    ent = output_entropy(ext_result["output_logits"])
    max_prob = max_token_probability(ext_result["output_logits"])
    first_prob = first_token_probability(ext_result["output_logits"])

    # For entropy, lower = more confident, so negate for AUROC
    ent_auroc = roc_auc_score(correct, -ent)
    max_prob_auroc = roc_auc_score(correct, max_prob)
    first_prob_auroc = roc_auc_score(correct, first_prob)

    results = {
        "model": model_name,
        "n_samples": len(prompts),
        "correct_rate": float(correct.mean()),
        "methods": {
            "topo_confidence": {"auroc": topo_auroc},
            "output_entropy": {"auroc": ent_auroc},
            "max_token_prob": {"auroc": max_prob_auroc},
            "first_token_prob": {"auroc": first_prob_auroc},
        },
        "correlation_topo_entropy": float(np.corrcoef(topo_conf, ent)[0, 1]),
    }

    logger.info("\nResults:")
    logger.info("%-20s AUROC", "Method")
    logger.info("-" * 35)
    for method, vals in results["methods"].items():
        logger.info("%-20s %.3f", method, vals["auroc"])
    logger.info("\nCorrelation (topo vs entropy): %.3f", results["correlation_topo_entropy"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
