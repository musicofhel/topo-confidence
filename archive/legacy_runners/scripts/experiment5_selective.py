#!/usr/bin/env python3
"""Experiment 5: Selective prediction.

Sweep confidence thresholds and show the accuracy vs coverage tradeoff.
If accuracy on answered problems increases meaningfully (e.g., baseline → 20%+
at 70% coverage), this is a useful selective prediction system.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    parser = argparse.ArgumentParser(description="Experiment 5: Selective prediction")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="data/experiment5")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts, gts = load_math500(args.max_problems)
    logger.info("Loaded %d problems", len(prompts))

    extractor = HiddenStateExtractor(args.model, device=args.device)
    result = extractor.extract_with_output(prompts, max_new_tokens=args.max_new_tokens)

    correct = np.zeros(len(prompts), dtype=int)
    for i, (text, gt) in enumerate(zip(result["generated_texts"], gts)):
        predicted = extract_answer(text)
        correct[i] = int(normalize_answer(predicted) == normalize_answer(gt))

    baseline_accuracy = correct.mean()
    logger.info("Baseline accuracy: %.1f%%", baseline_accuracy * 100)

    feat_ext = TopologicalFeatureExtractor()
    features = feat_ext.extract(
        hidden_states=result["hidden_states"],
        token_trajectories=result["token_trajectories"],
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_cv = min(50, len(prompts))
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    confidences = cross_val_predict(clf, X, correct, cv=cv, method="predict_proba")[:, 1]

    # Sweep thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows = []

    for t in thresholds:
        mask = confidences >= t
        answered_frac = mask.mean()
        if mask.any():
            acc_answered = correct[mask].mean()
        else:
            acc_answered = 0.0
        # "Accuracy on all" counts abstained as wrong
        acc_all = (correct * mask).sum() / len(correct)

        rows.append({
            "threshold": t,
            "answered_pct": float(answered_frac * 100),
            "accuracy_on_answered": float(acc_answered * 100),
            "accuracy_on_all": float(acc_all * 100),
            "n_answered": int(mask.sum()),
            "n_correct_answered": int((correct * mask).sum()),
        })

    # Also compute the "oracle" selective prediction (rank by actual confidence)
    # to see how close our method gets
    sorted_idx = np.argsort(-confidences)
    oracle_rows = []
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n = max(1, int(frac * len(correct)))
        top_idx = sorted_idx[:n]
        acc = correct[top_idx].mean()
        oracle_rows.append({
            "top_fraction": frac,
            "n_selected": n,
            "accuracy": float(acc * 100),
        })

    results = {
        "model": args.model,
        "n_problems": len(prompts),
        "baseline_accuracy": float(baseline_accuracy * 100),
        "auroc": float(roc_auc_score(correct, confidences)),
        "threshold_sweep": rows,
        "top_fraction_sweep": oracle_rows,
    }

    print("\n" + "=" * 75)
    print("Experiment 5: Selective Prediction (MATH-500)")
    print("=" * 75)
    print(f"Model: {args.model}")
    print(f"Baseline accuracy (all problems): {baseline_accuracy:.1%}")
    print(f"AUROC: {roc_auc_score(correct, confidences):.3f}")
    print()
    print(f"{'Threshold':>10} {'Answered%':>10} {'Acc(answered)':>14} {'Acc(all)':>10}")
    print("-" * 50)
    for r in rows:
        print(f"{r['threshold']:>10.2f} {r['answered_pct']:>9.1f}% "
              f"{r['accuracy_on_answered']:>13.1f}% {r['accuracy_on_all']:>9.1f}%")
    print("-" * 50)

    print("\nTop-k by confidence:")
    print(f"{'Top%':>6} {'N':>5} {'Accuracy':>10}")
    print("-" * 25)
    for r in oracle_rows:
        print(f"{r['top_fraction']*100:>5.0f}% {r['n_selected']:>5} {r['accuracy']:>9.1f}%")
    print("=" * 75)

    # Check if selective prediction is useful
    if rows:
        best_row = max(rows, key=lambda r: r["accuracy_on_answered"] if r["answered_pct"] > 30 else 0)
        improvement = best_row["accuracy_on_answered"] - baseline_accuracy * 100
        print(f"\nBest selective: {best_row['accuracy_on_answered']:.1f}% accuracy "
              f"at {best_row['answered_pct']:.1f}% coverage "
              f"(+{improvement:.1f}pp over baseline)")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", results_path)


if __name__ == "__main__":
    main()
