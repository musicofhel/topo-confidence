#!/usr/bin/env python3
"""Experiment 1 v2: MATH-500 correctness prediction with trajectory caching.

Key changes from experiment1_math500.py:
1. Caches token trajectories to trajectories.npz (~50-100MB)
2. --from-cache flag skips model inference and loads cached trajectories
3. Saves all 13 features (not just 6) to per_problem.jsonl
4. Saves manifest.json with experiment metadata

This allows iterating on features without re-running 45 min of GPU inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from topo_confidence.features import FEATURE_NAMES, TopologicalFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Give your final answer after '####'.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def load_math500(path: str | None = None) -> list[dict]:
    if path and Path(path).exists():
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    logger.info("Loading MATH-500 from HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return list(ds)


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
    answer = answer.strip().lower()
    answer = answer.replace("$", "").replace(",", "").replace(" ", "")
    answer = answer.rstrip(".")
    try:
        val = float(answer)
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return answer


def check_correct(predicted: str, ground_truth: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def save_trajectories(output_dir: Path, trajectories, generated_texts, ground_truths, correct):
    """Save token trajectories and metadata to disk."""
    traj_path = output_dir / "trajectories.npz"
    save_dict = {}
    for i, t in enumerate(trajectories):
        save_dict[f"traj_{i}"] = t

    np.savez_compressed(traj_path, **save_dict)
    logger.info("Trajectories saved to %s", traj_path)

    # Save metadata separately (strings can't go in npz cleanly)
    meta_path = output_dir / "trajectory_meta.json"
    meta = {
        "n_problems": len(trajectories),
        "generated_texts": [t[:500] for t in generated_texts],
        "ground_truths": ground_truths,
        "correct": correct.tolist(),
        "trajectory_shapes": [t.shape for t in trajectories],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    logger.info("Metadata saved to %s", meta_path)


def load_trajectories(output_dir: Path):
    """Load cached trajectories and metadata."""
    traj_path = output_dir / "trajectories.npz"
    meta_path = output_dir / "trajectory_meta.json"

    logger.info("Loading cached trajectories from %s", traj_path)
    data = np.load(traj_path)
    n_problems = len([k for k in data.keys() if k.startswith("traj_")])
    trajectories = [data[f"traj_{i}"] for i in range(n_problems)]

    with open(meta_path) as f:
        meta = json.load(f)

    correct = np.array(meta["correct"])
    generated_texts = meta["generated_texts"]
    ground_truths = meta["ground_truths"]

    logger.info("Loaded %d cached trajectories", len(trajectories))
    return trajectories, generated_texts, ground_truths, correct


def run_auroc_analysis(features, correct, feature_names, n_cv=50, label=""):
    """Run AUROC analysis: per-feature, cross-validated logistic regression, bootstrap CI."""
    prefix = f"[{label}] " if label else ""

    # Per-feature AUROC
    logger.info(f"\n{prefix}Per-feature AUROC:")
    feature_aurocs = {}
    for j, name in enumerate(feature_names):
        col = features[:, j]
        if col.std() < 1e-10:
            feature_aurocs[name] = float("nan")
            logger.info("  %s: N/A (constant)", name)
            continue
        try:
            auroc = roc_auc_score(correct, col)
            auroc_neg = roc_auc_score(correct, -col)
            best = max(auroc, auroc_neg)
            feature_aurocs[name] = best
            logger.info("  %s: %.3f", name, best)
        except ValueError:
            feature_aurocs[name] = float("nan")
            logger.info("  %s: N/A", name)

    # Cross-validated logistic regression
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_cv_actual = min(n_cv, len(features))

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_cv_actual, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(clf, X, correct, cv=cv, method="predict_proba")[:, 1]

    auroc = roc_auc_score(correct, cv_probs)
    logger.info(f"{prefix}Overall AUROC ({n_cv_actual}-fold CV): {auroc:.3f}")

    # Bootstrap CI
    rng = np.random.default_rng(42)
    bootstrap_aurocs = []
    for _ in range(1000):
        idx = rng.choice(len(correct), size=len(correct), replace=True)
        if len(np.unique(correct[idx])) < 2:
            continue
        bootstrap_aurocs.append(roc_auc_score(correct[idx], cv_probs[idx]))
    bootstrap_aurocs = np.array(bootstrap_aurocs)
    ci_low = np.percentile(bootstrap_aurocs, 2.5)
    ci_high = np.percentile(bootstrap_aurocs, 97.5)
    logger.info(f"{prefix}95%% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # Feature coefficients
    clf.fit(X, correct)
    coefficients = dict(zip(feature_names, clf.coef_[0]))

    return {
        "auroc": float(auroc),
        "auroc_ci_95": [float(ci_low), float(ci_high)],
        "auroc_bootstrap_mean": float(bootstrap_aurocs.mean()),
        "auroc_bootstrap_std": float(bootstrap_aurocs.std()),
        "per_feature_auroc": {k: float(v) for k, v in feature_aurocs.items()},
        "coefficients": {k: float(v) for k, v in coefficients.items()},
        "cv_probs": cv_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 v2: MATH-500 with caching")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default=None)
    parser.add_argument("--max-problems", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="data/experiment1_v2")
    parser.add_argument("--cv-folds", type=int, default=50)
    parser.add_argument("--from-cache", action="store_true",
                        help="Load cached trajectories instead of running model inference")
    parser.add_argument("--null-k", type=int, default=100,
                        help="Number of null shuffles for ph_significance (0 to skip)")
    parser.add_argument("--max-dim", type=int, default=2,
                        help="Maximum homology dimension (1=H0+H1, 2=H0+H1+H2)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        # --- Load cached trajectories ---
        trajectories, generated_texts, ground_truths, correct = load_trajectories(output_dir)
    else:
        # --- Full model inference ---
        problems = load_math500(args.data)
        if args.max_problems and len(problems) > args.max_problems:
            problems = problems[:args.max_problems]
        logger.info("Loaded %d problems", len(problems))

        prompts = []
        ground_truths = []
        for p in problems:
            prompt_text = p.get("problem", p.get("question", ""))
            prompts.append(MATH_PROMPT_TEMPLATE.format(problem=prompt_text))
            answer = p.get("answer", p.get("solution", ""))
            boxed = re.search(r"\\boxed\{(.+?)\}", answer)
            ground_truths.append(boxed.group(1) if boxed else answer)

        logger.info("Extracting hidden states and generating outputs...")
        t0 = time.perf_counter()
        from topo_confidence.extractor import HiddenStateExtractor
        extractor = HiddenStateExtractor(args.model, device=args.device)
        result = extractor.extract_with_output(prompts, max_new_tokens=args.max_new_tokens)
        extraction_time = time.perf_counter() - t0
        logger.info("Extraction + generation: %.1fs", extraction_time)

        trajectories = result["token_trajectories"]
        generated_texts = result["generated_texts"]

        # Evaluate correctness
        correct = np.zeros(len(prompts), dtype=int)
        for i, (text, gt) in enumerate(zip(generated_texts, ground_truths)):
            predicted = extract_answer(text)
            correct[i] = int(check_correct(predicted, gt))

        logger.info("Correctness rate: %.1f%% (%d/%d)",
                    correct.mean() * 100, correct.sum(), len(correct))

        # Cache trajectories
        save_trajectories(output_dir, trajectories, generated_texts, ground_truths, correct)

    # --- Feature extraction with 13-feature extractor ---
    logger.info("Computing 13-feature topological features (max_dim=%d, null_k=%d)...",
                args.max_dim, args.null_k)
    t0 = time.perf_counter()
    feat_extractor = TopologicalFeatureExtractor(
        max_dim=args.max_dim, null_k=args.null_k)
    features = feat_extractor.extract(token_trajectories=trajectories)
    feature_time = time.perf_counter() - t0
    logger.info("Feature extraction: %.1fs (%.3fs/problem)",
                feature_time, feature_time / len(trajectories))

    feature_names = feat_extractor.feature_names

    # --- Feature diagnostics ---
    logger.info("\nFeature statistics:")
    for j, name in enumerate(feature_names):
        col = features[:, j]
        logger.info("  %s: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  frac_zero=%.2f",
                    name, col.mean(), col.std(), col.min(), col.max(),
                    (col == 0).mean())

    # --- AUROC analysis: 13 features ---
    analysis_13 = run_auroc_analysis(features, correct, feature_names,
                                     n_cv=args.cv_folds, label="13-feature")

    # --- AUROC analysis: original 7 features (baseline comparison) ---
    # Original 7: H0_persistence_entropy, H1_max_lifetime, H0_total_persistence,
    #              H0_n_features, H1_persistence_entropy, H1_n_features, bridge_silhouette
    original_indices = [0, 1, 2, 3, 4, 5, 9]  # indices into FEATURE_NAMES
    original_names = [feature_names[i] for i in original_indices]
    features_7 = features[:, original_indices]
    analysis_7 = run_auroc_analysis(features_7, correct, original_names,
                                    n_cv=args.cv_folds, label="7-feature baseline")

    # --- Save results ---
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    results = {
        "model": args.model,
        "n_problems": len(trajectories),
        "correctness_rate": float(correct.mean()),
        "from_cache": args.from_cache,
        "max_dim": args.max_dim,
        "null_k": args.null_k,
        "feature_time_s": feature_time,
        "feature_time_per_problem_s": feature_time / len(trajectories),
        "feature_names": feature_names,
        "analysis_13_feature": {k: v for k, v in analysis_13.items() if k != "cv_probs"},
        "analysis_7_feature_baseline": {k: v for k, v in analysis_7.items() if k != "cv_probs"},
        "auroc_delta": analysis_13["auroc"] - analysis_7["auroc"],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", results_path)

    # Save per-problem data with all 13 features
    per_problem = []
    for i in range(len(trajectories)):
        per_problem.append({
            "index": i,
            "correct": int(correct[i]),
            "confidence_13": float(analysis_13["cv_probs"][i]),
            "confidence_7": float(analysis_7["cv_probs"][i]),
            "generated_text": generated_texts[i][:500] if i < len(generated_texts) else "",
            "ground_truth": ground_truths[i] if i < len(ground_truths) else "",
            "features": {
                name: float(features[i, j])
                for j, name in enumerate(feature_names)
            },
        })

    per_problem_path = output_dir / "per_problem.jsonl"
    with open(per_problem_path, "w") as f:
        for item in per_problem:
            f.write(json.dumps(item) + "\n")
    logger.info("Per-problem data saved to %s", per_problem_path)

    # Save manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "git_hash": git_hash,
        "model": args.model,
        "max_dim": args.max_dim,
        "null_k": args.null_k,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_problems": len(trajectories),
        "files": {
            "trajectories": "trajectories.npz",
            "metadata": "trajectory_meta.json",
            "results": "results.json",
            "per_problem": "per_problem.jsonl",
        },
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Experiment 1 v2: MATH-500 Correctness Prediction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Problems: {len(trajectories)}")
    print(f"Correctness rate: {correct.mean():.1%}")
    print(f"Feature config: max_dim={args.max_dim}, null_k={args.null_k}")
    print(f"Feature extraction: {feature_time:.1f}s ({feature_time/len(trajectories):.3f}s/problem)")
    print()
    print(f"7-feature AUROC (baseline): {analysis_7['auroc']:.3f} "
          f"(95% CI: [{analysis_7['auroc_ci_95'][0]:.3f}, {analysis_7['auroc_ci_95'][1]:.3f}])")
    print(f"13-feature AUROC:           {analysis_13['auroc']:.3f} "
          f"(95% CI: [{analysis_13['auroc_ci_95'][0]:.3f}, {analysis_13['auroc_ci_95'][1]:.3f}])")
    delta = analysis_13["auroc"] - analysis_7["auroc"]
    print(f"AUROC delta:                {delta:+.3f}")
    print(f"Target (>= 0.75): {'MET' if analysis_13['auroc'] >= 0.75 else 'NOT MET'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
