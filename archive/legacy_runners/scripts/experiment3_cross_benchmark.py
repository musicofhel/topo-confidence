#!/usr/bin/env python3
"""Experiment 3: Cross-benchmark validation.

Fix model (Qwen2.5-1.5B-Instruct), vary benchmark:
- MATH-500 (math reasoning)
- HumanEval (code generation)
- GSM8K (grade school math)
- MMLU (multiple choice)

Report AUROC per benchmark.
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


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_math500(max_n: int = 500) -> tuple[list[str], list[str], str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts, gts = [], []
    for item in list(ds)[:max_n]:
        problem = item.get("problem", item.get("question", ""))
        prompts.append(
            f"Solve the following math problem. Give your final answer after '####'.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
        answer = item.get("answer", item.get("solution", ""))
        boxed = re.search(r"\\boxed\{(.+?)\}", answer)
        gts.append(boxed.group(1) if boxed else answer)
    return prompts, gts, "math"


def load_gsm8k(max_n: int = 500) -> tuple[list[str], list[str], str]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts, gts = [], []
    for item in list(ds)[:max_n]:
        prompts.append(
            f"Solve this math problem step by step. Give your final answer after '####'.\n\n"
            f"Question: {item['question']}\n\nAnswer:"
        )
        # GSM8K answers end with #### <number>
        answer = item["answer"]
        match = re.search(r"####\s*(.+)", answer)
        gts.append(match.group(1).strip() if match else answer)
    return prompts, gts, "math"


def load_mmlu(max_n: int = 500) -> tuple[list[str], list[str], str]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    prompts, gts = [], []
    choices = ["A", "B", "C", "D"]
    for item in list(ds)[:max_n]:
        question = item["question"]
        options = "\n".join(f"{choices[i]}. {item['choices'][i]}" for i in range(4))
        prompts.append(
            f"Answer the following multiple choice question. Reply with just the letter (A, B, C, or D).\n\n"
            f"{question}\n{options}\n\nAnswer:"
        )
        gts.append(choices[item["answer"]])
    return prompts, gts, "mcq"


def load_humaneval(max_n: int = 164) -> tuple[list[str], list[str], str]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts, gts = [], []
    for item in list(ds)[:max_n]:
        prompts.append(
            f"Complete the following Python function. Only output the function body.\n\n"
            f"{item['prompt']}"
        )
        # For HumanEval, correctness requires execution. We use a simpler proxy:
        # check if the canonical solution's key elements appear in the output.
        gts.append(item.get("canonical_solution", ""))
    return prompts, gts, "code"


LOADERS = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
    "humaneval": load_humaneval,
}


# ── Answer checking ──────────────────────────────────────────────────────────

def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower().replace("$", "").replace(",", "").replace(" ", "").rstrip(".")
    try:
        val = float(answer)
        return str(int(val)) if val == int(val) else f"{val:.6g}"
    except ValueError:
        return answer


def extract_math_answer(text: str) -> str:
    for pattern in [r"####\s*(.+?)(?:\n|$)", r"\\boxed\{(.+?)\}",
                    r"(?:answer|result)\s+is\s+(.+?)(?:\.|,|\n|$)"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else text.strip()


def extract_mcq_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else text[:1].upper()


def check_correct(predicted: str, ground_truth: str, task_type: str) -> bool:
    if task_type == "math":
        return normalize_answer(extract_math_answer(predicted)) == normalize_answer(ground_truth)
    elif task_type == "mcq":
        return extract_mcq_answer(predicted) == ground_truth.upper()
    elif task_type == "code":
        # Simple heuristic: check if key lines from canonical solution appear
        # Real evaluation would use execution, but this is a proxy
        gt_lines = [l.strip() for l in ground_truth.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if not gt_lines:
            return False
        matches = sum(1 for l in gt_lines if l in predicted)
        return matches >= len(gt_lines) * 0.3
    return normalize_answer(predicted) == normalize_answer(ground_truth)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(extractor: HiddenStateExtractor, benchmark: str,
                  max_n: int, max_new_tokens: int) -> dict:
    logger.info("=" * 60)
    logger.info("Benchmark: %s", benchmark)
    logger.info("=" * 60)

    prompts, gts, task_type = LOADERS[benchmark](max_n)
    logger.info("Loaded %d problems (type: %s)", len(prompts), task_type)

    t0 = time.perf_counter()
    result = extractor.extract_with_output(prompts, max_new_tokens=max_new_tokens)
    inference_time = time.perf_counter() - t0

    correct = np.zeros(len(prompts), dtype=int)
    for i, (text, gt) in enumerate(zip(result["generated_texts"], gts)):
        correct[i] = int(check_correct(text, gt, task_type))

    logger.info("Correctness: %.1f%% (%d/%d)", correct.mean() * 100, correct.sum(), len(correct))

    feat_ext = TopologicalFeatureExtractor()
    features = feat_ext.extract(
        hidden_states=result["hidden_states"],
        token_trajectories=result["token_trajectories"],
    )

    feature_aurocs = {}
    for j, name in enumerate(feat_ext.feature_names):
        try:
            auroc = max(
                roc_auc_score(correct, features[:, j]),
                roc_auc_score(correct, -features[:, j]),
            )
            feature_aurocs[name] = auroc
        except ValueError:
            feature_aurocs[name] = float("nan")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_cv = min(50, len(prompts))

    try:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        cv_probs = cross_val_predict(clf, X, correct, cv=cv, method="predict_proba")[:, 1]
        combined_auroc = roc_auc_score(correct, cv_probs)
    except ValueError:
        combined_auroc = float("nan")

    top_feature = max(feature_aurocs, key=lambda k: feature_aurocs.get(k, 0))

    return {
        "benchmark": benchmark,
        "n_problems": len(prompts),
        "task_type": task_type,
        "correctness_rate": float(correct.mean()),
        "combined_auroc": float(combined_auroc),
        "per_feature_auroc": {k: float(v) for k, v in feature_aurocs.items()},
        "top_feature": top_feature,
        "top_feature_auroc": float(feature_aurocs[top_feature]),
        "inference_time_s": inference_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Cross-benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="data/experiment3")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["math500", "gsm8k", "mmlu", "humaneval"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = HiddenStateExtractor(args.model, device=args.device)

    all_results = []
    for benchmark in args.benchmarks:
        try:
            result = run_benchmark(extractor, benchmark, args.max_problems,
                                   args.max_new_tokens)
            all_results.append(result)
        except Exception as e:
            logger.error("Failed on %s: %s", benchmark, e, exc_info=True)
            all_results.append({"benchmark": benchmark, "error": str(e)})

    print("\n" + "=" * 80)
    print(f"Experiment 3: Cross-Benchmark Validation ({args.model})")
    print("=" * 80)
    print(f"{'Benchmark':<15} {'Type':<6} {'Correct%':>8} {'AUROC':>7} {'Top Feature':<30}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['benchmark']:<15} {'':>6} {'ERROR':>8}")
        else:
            print(f"{r['benchmark']:<15} {r['task_type']:<6} "
                  f"{r['correctness_rate']*100:>7.1f}% {r['combined_auroc']:>7.3f} "
                  f"{r['top_feature']:<30}")
    print("=" * 80)

    n_above_065 = sum(
        1 for r in all_results
        if "error" not in r and r["combined_auroc"] > 0.65
    )
    print(f"\nAUROC > 0.65 on {n_above_065}/{len(all_results)} benchmarks "
          f"(need 3+ to claim generalization)")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", results_path)


if __name__ == "__main__":
    main()
