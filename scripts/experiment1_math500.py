#!/usr/bin/env python3
"""Experiment 1: Reproduce Phase 5 MATH-500 correctness prediction.

Load Qwen2.5-1.5B-Instruct, extract token trajectories for MATH-500 problems,
compute topological features, evaluate correctness, fit logistic regression,
report AUROC.

Target: AUROC >= 0.75
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


def load_math500(path: str | None = None) -> list[dict]:
    """Load MATH-500 from HuggingFace datasets or local file."""
    if path and Path(path).exists():
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    logger.info("Loading MATH-500 from HuggingFace datasets...")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return list(ds)


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from model output.

    Looks for patterns like:
    - #### <answer>
    - The answer is <answer>
    - \\boxed{<answer>}
    - Last number in the text
    """
    # Try #### pattern first
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()

    # Try \\boxed{} pattern
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        return match.group(1).strip()

    # Try "answer is" pattern
    match = re.search(r"(?:answer|result)\s+is\s+(.+?)(?:\.|,|\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Last number in text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    answer = answer.replace("$", "").replace(",", "").replace(" ", "")
    # Remove trailing period
    answer = answer.rstrip(".")
    # Try to parse as number
    try:
        val = float(answer)
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return answer


def check_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm == gt_norm


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: MATH-500")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default=None, help="Path to local MATH-500 JSONL")
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="data/experiment1")
    parser.add_argument("--cv-folds", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    problems = load_math500(args.data)
    if args.max_problems and len(problems) > args.max_problems:
        problems = problems[: args.max_problems]
    logger.info("Loaded %d problems", len(problems))

    # Format prompts
    prompts = []
    ground_truths = []
    for p in problems:
        prompt_text = p.get("problem", p.get("question", ""))
        prompts.append(MATH_PROMPT_TEMPLATE.format(problem=prompt_text))
        # Ground truth answer
        answer = p.get("answer", p.get("solution", ""))
        # MATH dataset often has answer in \boxed{} within solution
        boxed = re.search(r"\\boxed\{(.+?)\}", answer)
        if boxed:
            ground_truths.append(boxed.group(1))
        else:
            ground_truths.append(answer)

    # Extract hidden states and generate outputs
    logger.info("Extracting hidden states and generating outputs...")
    t0 = time.perf_counter()
    extractor = HiddenStateExtractor(args.model, device=args.device)
    result = extractor.extract_with_output(prompts, max_new_tokens=args.max_new_tokens)
    extraction_time = time.perf_counter() - t0
    logger.info("Extraction + generation: %.1fs (%.2fs/problem)",
                extraction_time, extraction_time / len(prompts))

    # Evaluate correctness
    correct = np.zeros(len(prompts), dtype=int)
    for i, (text, gt) in enumerate(zip(result["generated_texts"], ground_truths)):
        predicted = extract_answer(text)
        correct[i] = int(check_correct(predicted, gt))

    correctness_rate = correct.mean()
    logger.info("Correctness rate: %.1f%% (%d/%d)",
                correctness_rate * 100, correct.sum(), len(correct))

    # Compute topological features
    logger.info("Computing topological features...")
    t0 = time.perf_counter()
    feat_extractor = TopologicalFeatureExtractor()
    features = feat_extractor.extract(
        hidden_states=result["hidden_states"],
        token_trajectories=result["token_trajectories"],
    )
    feature_time = time.perf_counter() - t0
    logger.info("Feature extraction: %.1fs (%.3fs/problem)",
                feature_time, feature_time / len(prompts))

    # Per-feature AUROC
    logger.info("\nPer-feature AUROC:")
    feature_aurocs = {}
    for j, name in enumerate(feat_extractor.feature_names):
        try:
            auroc = roc_auc_score(correct, features[:, j])
            # Try negated too (some features inversely predict)
            auroc_neg = roc_auc_score(correct, -features[:, j])
            best = max(auroc, auroc_neg)
            feature_aurocs[name] = best
            logger.info("  %s: %.3f", name, best)
        except ValueError:
            feature_aurocs[name] = float("nan")
            logger.info("  %s: N/A (constant)", name)

    # Logistic regression with cross-validation
    logger.info("\nFitting logistic regression...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_cv = min(args.cv_folds, len(prompts))
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(clf, X, correct, cv=cv, method="predict_proba")[:, 1]

    # Overall AUROC
    auroc = roc_auc_score(correct, cv_probs)
    logger.info("Overall AUROC (%d-fold CV): %.3f", n_cv, auroc)

    # Bootstrap CI
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    bootstrap_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(correct), size=len(correct), replace=True)
        if len(np.unique(correct[idx])) < 2:
            continue
        bootstrap_aurocs.append(roc_auc_score(correct[idx], cv_probs[idx]))
    bootstrap_aurocs = np.array(bootstrap_aurocs)
    ci_low = np.percentile(bootstrap_aurocs, 2.5)
    ci_high = np.percentile(bootstrap_aurocs, 97.5)
    logger.info("95%% CI: [%.3f, %.3f]", ci_low, ci_high)
    logger.info("Bootstrap mean: %.3f +/- %.3f", bootstrap_aurocs.mean(), bootstrap_aurocs.std())

    # Fit final model and get feature importance
    clf.fit(X, correct)
    logger.info("\nFeature coefficients:")
    for name, coef in zip(feat_extractor.feature_names, clf.coef_[0]):
        logger.info("  %s: %+.4f", name, coef)

    # Save results
    results = {
        "model": args.model,
        "n_problems": len(prompts),
        "correctness_rate": float(correctness_rate),
        "auroc": float(auroc),
        "auroc_ci_95": [float(ci_low), float(ci_high)],
        "auroc_bootstrap_mean": float(bootstrap_aurocs.mean()),
        "auroc_bootstrap_std": float(bootstrap_aurocs.std()),
        "cv_folds": n_cv,
        "per_feature_auroc": {k: float(v) for k, v in feature_aurocs.items()},
        "coefficients": {
            name: float(coef)
            for name, coef in zip(feat_extractor.feature_names, clf.coef_[0])
        },
        "extraction_time_s": extraction_time,
        "feature_time_s": feature_time,
        "target": "AUROC >= 0.75",
        "target_met": auroc >= 0.75,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", results_path)

    # Save per-problem data for further analysis
    per_problem = []
    for i in range(len(prompts)):
        per_problem.append({
            "index": i,
            "correct": int(correct[i]),
            "confidence": float(cv_probs[i]),
            "generated_text": result["generated_texts"][i][:500],
            "ground_truth": ground_truths[i],
            "features": {
                name: float(features[i, j])
                for j, name in enumerate(feat_extractor.feature_names)
            },
        })

    per_problem_path = output_dir / "per_problem.jsonl"
    with open(per_problem_path, "w") as f:
        for item in per_problem:
            f.write(json.dumps(item) + "\n")
    logger.info("Per-problem data saved to %s", per_problem_path)

    # Summary
    print("\n" + "=" * 60)
    print(f"Experiment 1: MATH-500 Correctness Prediction")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Problems: {len(prompts)}")
    print(f"Correctness rate: {correctness_rate:.1%}")
    print(f"AUROC: {auroc:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
    print(f"Target (>= 0.75): {'MET' if auroc >= 0.75 else 'NOT MET'}")
    print(f"Top feature: {max(feature_aurocs, key=feature_aurocs.get)} "
          f"(AUROC={max(feature_aurocs.values()):.3f})")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
