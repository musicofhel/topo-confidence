#!/usr/bin/env python3
"""Experiment 2: Cross-model validation.

Run the topo-confidence pipeline on multiple models to test whether
H0_persistence_entropy dominates across architectures.

Models (all fit on RTX 2060 Super):
- Qwen/Qwen2.5-1.5B-Instruct (baseline)
- microsoft/phi-2 (2.7B)
- meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from topo_confidence.extractor import HiddenStateExtractor
from topo_confidence.features import TopologicalFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Give your final answer after '####'.\n\n"
    "Problem: {problem}\n\nSolution:"
)

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/phi-2",
    "meta-llama/Llama-3.2-1B-Instruct",
]


def load_math500(path: str | None = None, max_problems: int = 500) -> list[dict]:
    if path and Path(path).exists():
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
        return data[:max_problems]
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return list(ds)[:max_problems]


def extract_answer(text: str) -> str:
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:answer|result)\s+is\s+(.+?)(?:\.|,|\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower().replace("$", "").replace(",", "").replace(" ", "").rstrip(".")
    try:
        val = float(answer)
        return str(int(val)) if val == int(val) else f"{val:.6g}"
    except ValueError:
        return answer


def run_model(model_name: str, problems: list[dict], device: str,
              max_new_tokens: int = 256) -> dict:
    """Run full pipeline for one model."""
    logger.info("=" * 60)
    logger.info("Model: %s", model_name)
    logger.info("=" * 60)

    prompts = []
    ground_truths = []
    for p in problems:
        prompt_text = p.get("problem", p.get("question", ""))
        prompts.append(MATH_PROMPT_TEMPLATE.format(problem=prompt_text))
        answer = p.get("answer", p.get("solution", ""))
        boxed = re.search(r"\\boxed\{(.+?)\}", answer)
        ground_truths.append(boxed.group(1) if boxed else answer)

    t0 = time.perf_counter()
    extractor = HiddenStateExtractor(model_name, device=device)
    result = extractor.extract_with_output(prompts, max_new_tokens=max_new_tokens)
    inference_time = time.perf_counter() - t0

    correct = np.zeros(len(prompts), dtype=int)
    for i, (text, gt) in enumerate(zip(result["generated_texts"], ground_truths)):
        predicted = extract_answer(text)
        correct[i] = int(normalize_answer(predicted) == normalize_answer(gt))

    logger.info("Correctness: %.1f%% (%d/%d)", correct.mean() * 100, correct.sum(), len(correct))

    feat_extractor = TopologicalFeatureExtractor()
    features = feat_extractor.extract(
        hidden_states=result["hidden_states"],
        token_trajectories=result["token_trajectories"],
    )

    # Per-feature AUROC
    feature_aurocs = {}
    for j, name in enumerate(feat_extractor.feature_names):
        try:
            auroc = max(
                roc_auc_score(correct, features[:, j]),
                roc_auc_score(correct, -features[:, j]),
            )
            feature_aurocs[name] = auroc
        except ValueError:
            feature_aurocs[name] = float("nan")

    # Combined logistic regression
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_cv = min(50, len(prompts))
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)

    try:
        cv_probs = cross_val_predict(clf, X, correct, cv=cv, method="predict_proba")[:, 1]
        combined_auroc = roc_auc_score(correct, cv_probs)
    except ValueError:
        combined_auroc = float("nan")

    top_feature = max(feature_aurocs, key=lambda k: feature_aurocs.get(k, 0))

    logger.info("Combined AUROC: %.3f", combined_auroc)
    logger.info("Top feature: %s (%.3f)", top_feature, feature_aurocs[top_feature])

    # Free GPU memory
    del extractor
    import torch
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_problems": len(prompts),
        "correctness_rate": float(correct.mean()),
        "combined_auroc": float(combined_auroc),
        "per_feature_auroc": {k: float(v) for k, v in feature_aurocs.items()},
        "top_feature": top_feature,
        "top_feature_auroc": float(feature_aurocs[top_feature]),
        "inference_time_s": inference_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Cross-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default=None)
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="data/experiment2")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = load_math500(args.data, args.max_problems)
    logger.info("Loaded %d problems", len(problems))

    all_results = []
    for model_name in args.models:
        try:
            result = run_model(model_name, problems, args.device, args.max_new_tokens)
            all_results.append(result)
        except Exception as e:
            logger.error("Failed on %s: %s", model_name, e)
            all_results.append({"model": model_name, "error": str(e)})

    # Summary table
    print("\n" + "=" * 80)
    print("Experiment 2: Cross-Model Validation on MATH-500")
    print("=" * 80)
    print(f"{'Model':<40} {'Correct%':>8} {'AUROC':>7} {'Top Feature':<30}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<40} {'ERROR':>8}")
        else:
            print(f"{r['model']:<40} {r['correctness_rate']*100:>7.1f}% "
                  f"{r['combined_auroc']:>7.3f} {r['top_feature']:<30}")
    print("=" * 80)

    # Check if H0_persistence_entropy dominates across models
    h0_dominant = sum(
        1 for r in all_results
        if "error" not in r and r["top_feature"] == "H0_persistence_entropy"
    )
    print(f"\nH0_persistence_entropy is top feature in {h0_dominant}/{len(all_results)} models")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", results_path)


if __name__ == "__main__":
    main()
