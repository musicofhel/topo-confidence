#!/usr/bin/env python3
"""Experiment 4: Compare topo-confidence to baselines.

Methods:
1. Output entropy (negative, since lower entropy = more confident)
2. Max token probability (averaged over generated tokens)
3. First-token probability
4. Topo-confidence (our 6-feature logistic regression)
5. Combined (topo features + output entropy)

Reports AUROC for each, plus correlation between topo and entropy.
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

from topo_confidence.baselines import (
    first_token_probability,
    max_token_probability,
    output_entropy,
)
from topo_confidence.extractor import HiddenStateExtractor
from topo_confidence.features import TopologicalFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Give your final answer after '####'.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def load_math500(max_n: int = 500):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts, gts = [], []
    for item in list(ds)[:max_n]:
        problem = item.get("problem", item.get("question", ""))
        prompts.append(MATH_PROMPT_TEMPLATE.format(problem=problem))
        answer = item.get("answer", item.get("solution", ""))
        boxed = re.search(r"\\boxed\{(.+?)\}", answer)
        gts.append(boxed.group(1) if boxed else answer)
    return prompts, gts


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower().replace("$", "").replace(",", "").replace(" ", "").rstrip(".")
    try:
        val = float(answer)
        return str(int(val)) if val == int(val) else f"{val:.6g}"
    except ValueError:
        return answer


def extract_answer(text: str) -> str:
    for pattern in [r"####\s*(.+?)(?:\n|$)", r"\\boxed\{(.+?)\}",
                    r"(?:answer|result)\s+is\s+(.+?)(?:\.|,|\n|$)"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else text.strip()


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Baselines")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="data/experiment4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts, gts = load_math500(args.max_problems)
    logger.info("Loaded %d problems", len(prompts))

    extractor = HiddenStateExtractor(args.model, device=args.device)
    logger.info("Extracting hidden states + generating outputs...")
    t0 = time.perf_counter()
    result = extractor.extract_with_output(prompts, max_new_tokens=args.max_new_tokens, collect_logits=True)
    total_time = time.perf_counter() - t0
    logger.info("Total inference: %.1fs", total_time)

    # Evaluate correctness
    correct = np.zeros(len(prompts), dtype=int)
    for i, (text, gt) in enumerate(zip(result["generated_texts"], gts)):
        predicted = extract_answer(text)
        correct[i] = int(normalize_answer(predicted) == normalize_answer(gt))
    logger.info("Correctness: %.1f%%", correct.mean() * 100)

    # Topo features
    feat_ext = TopologicalFeatureExtractor()
    topo_features = feat_ext.extract(
        hidden_states=result["hidden_states"],
        token_trajectories=result["token_trajectories"],
    )

    # Baselines
    ent = output_entropy(result["output_logits"])
    max_prob = max_token_probability(result["output_logits"])
    first_prob = first_token_probability(result["output_logits"])

    # AUROC for each method
    n_cv = min(50, len(prompts))
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)

    methods = {}

    # Output entropy (negate: lower entropy = more confident = more likely correct)
    try:
        methods["output_entropy"] = {"auroc": float(roc_auc_score(correct, -ent))}
    except ValueError:
        methods["output_entropy"] = {"auroc": float("nan")}

    # Max token probability
    try:
        methods["max_token_prob"] = {"auroc": float(roc_auc_score(correct, max_prob))}
    except ValueError:
        methods["max_token_prob"] = {"auroc": float("nan")}

    # First token probability
    try:
        methods["first_token_prob"] = {"auroc": float(roc_auc_score(correct, first_prob))}
    except ValueError:
        methods["first_token_prob"] = {"auroc": float("nan")}

    # Topo-confidence (6-feature logistic regression, CV)
    scaler_topo = StandardScaler()
    X_topo = scaler_topo.fit_transform(topo_features)
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        cv_probs = cross_val_predict(clf, X_topo, correct, cv=cv, method="predict_proba")[:, 1]
        methods["topo_confidence"] = {"auroc": float(roc_auc_score(correct, cv_probs))}
    except ValueError:
        methods["topo_confidence"] = {"auroc": float("nan")}

    # Combined: topo features + entropy + max_prob + first_prob
    combined = np.column_stack([topo_features, ent.reshape(-1, 1),
                                 max_prob.reshape(-1, 1), first_prob.reshape(-1, 1)])
    scaler_comb = StandardScaler()
    X_comb = scaler_comb.fit_transform(combined)
    try:
        clf_comb = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        cv_probs_comb = cross_val_predict(clf_comb, X_comb, correct, cv=cv,
                                           method="predict_proba")[:, 1]
        methods["combined"] = {"auroc": float(roc_auc_score(correct, cv_probs_comb))}
    except ValueError:
        methods["combined"] = {"auroc": float("nan")}

    # Correlation
    topo_scores = cv_probs if "topo_confidence" in methods else np.zeros(len(correct))
    corr_topo_entropy = float(np.corrcoef(topo_scores, ent)[0, 1]) if len(topo_scores) > 1 else 0.0

    results = {
        "model": args.model,
        "n_problems": len(prompts),
        "correctness_rate": float(correct.mean()),
        "methods": methods,
        "correlation_topo_entropy": corr_topo_entropy,
        "inference_time_s": total_time,
    }

    print("\n" + "=" * 60)
    print("Experiment 4: Topo-Confidence vs Baselines (MATH-500)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Correctness rate: {correct.mean():.1%}")
    print()
    print(f"{'Method':<20} {'AUROC':>7}")
    print("-" * 30)
    for name, vals in methods.items():
        print(f"{name:<20} {vals['auroc']:>7.3f}")
    print("-" * 30)
    print(f"\nCorrelation (topo vs entropy): {corr_topo_entropy:.3f}")
    if abs(corr_topo_entropy) < 0.5:
        print("Low correlation → methods capture different signals, combination should help")
    print("=" * 60)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", results_path)


if __name__ == "__main__":
    main()
