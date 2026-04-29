#!/usr/bin/env python3
"""Experiment 6: Speed benchmark.

Measure wall-clock time for each component of the pipeline:
1. Model forward pass (already required)
2. Hidden state extraction overhead
3. PCA reduction
4. Ripser PH computation
5. Feature extraction
6. Logistic regression prediction

Report total overhead per problem beyond what inference already costs.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from topo_confidence.extractor import HiddenStateExtractor
from topo_confidence.features import TopologicalFeatureExtractor
from topo_confidence.utils import subsample_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Speed benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-problems", type=int, default=50,
                        help="Number of problems to benchmark")
    parser.add_argument("--output-dir", default="data/experiment6")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test prompts
    prompts = [
        f"Solve: what is {i*7} + {i*13}? Give your answer after ####."
        for i in range(1, args.n_problems + 1)
    ]

    # 1. Model loading
    t0 = time.perf_counter()
    extractor = HiddenStateExtractor(args.model, device=args.device)
    model_load_time = time.perf_counter() - t0

    # 2. Forward pass + generation (the cost you'd pay anyway)
    t0 = time.perf_counter()
    result = extractor.extract_with_output(prompts, max_new_tokens=128)
    inference_time = time.perf_counter() - t0

    # 3. Forward pass WITHOUT generation (just hidden states)
    t0 = time.perf_counter()
    result_extract = extractor.extract(prompts)
    extract_only_time = time.perf_counter() - t0

    # Hidden state extraction overhead = extract_only - (what a bare forward pass would cost)
    # In practice, the overhead is nearly zero since we just save tensors that are already computed

    trajectories = result["token_trajectories"]
    n = len(trajectories)

    # 4. PCA fitting + transform
    all_points = np.concatenate(trajectories, axis=0)
    t0 = time.perf_counter()
    n_components = min(30, all_points.shape[1], all_points.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(all_points)
    pca_fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reduced_trajs = [pca.transform(t) for t in trajectories]
    pca_transform_time = time.perf_counter() - t0

    # 5. Ripser PH computation (per problem)
    from ripser import ripser as ripser_fn

    rng = np.random.default_rng(42)
    ripser_times = []
    for traj in reduced_trajs:
        points = subsample_points(traj, 100, rng)
        t0 = time.perf_counter()
        ripser_fn(points, maxdim=1)
        ripser_times.append(time.perf_counter() - t0)
    ripser_time = sum(ripser_times)

    # 6. Full feature extraction (PCA + PH + scalars)
    feat_ext = TopologicalFeatureExtractor()
    t0 = time.perf_counter()
    features = feat_ext.extract(
        hidden_states=result["hidden_states"],
        token_trajectories=trajectories,
    )
    feature_total_time = time.perf_counter() - t0

    # 7. Logistic regression prediction (trivial)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    fake_labels = np.random.randint(0, 2, n)
    clf.fit(X, fake_labels)

    t0 = time.perf_counter()
    for _ in range(100):  # repeat for measurable time
        clf.predict_proba(X)
    lr_time = (time.perf_counter() - t0) / 100

    # Summary
    overhead_per_problem = feature_total_time / n + lr_time / n
    inference_per_problem = inference_time / n

    results = {
        "model": args.model,
        "n_problems": n,
        "timings": {
            "model_load_s": model_load_time,
            "inference_with_generation_s": inference_time,
            "inference_per_problem_s": inference_per_problem,
            "extract_only_s": extract_only_time,
            "extract_per_problem_s": extract_only_time / n,
            "pca_fit_s": pca_fit_time,
            "pca_transform_s": pca_transform_time,
            "pca_transform_per_problem_s": pca_transform_time / n,
            "ripser_total_s": ripser_time,
            "ripser_per_problem_s": ripser_time / n,
            "ripser_per_problem_mean_s": float(np.mean(ripser_times)),
            "ripser_per_problem_max_s": float(np.max(ripser_times)),
            "feature_total_s": feature_total_time,
            "feature_per_problem_s": feature_total_time / n,
            "lr_prediction_s": lr_time,
            "lr_per_problem_s": lr_time / n,
            "total_overhead_per_problem_s": overhead_per_problem,
        },
        "overhead_fraction": overhead_per_problem / inference_per_problem
        if inference_per_problem > 0 else 0,
    }

    print("\n" + "=" * 65)
    print(f"Experiment 6: Speed Benchmark ({n} problems)")
    print("=" * 65)
    print(f"Model: {args.model}")
    print()
    print(f"{'Component':<35} {'Total':>10} {'Per Problem':>12}")
    print("-" * 60)
    print(f"{'Model loading (one-time)':<35} {model_load_time:>9.2f}s {'—':>12}")
    print(f"{'Inference + generation':<35} {inference_time:>9.2f}s {inference_per_problem:>11.3f}s")
    print(f"{'Hidden state extraction only':<35} {extract_only_time:>9.2f}s {extract_only_time/n:>11.3f}s")
    print(f"{'PCA fit (one-time per batch)':<35} {pca_fit_time:>9.3f}s {'—':>12}")
    print(f"{'PCA transform':<35} {pca_transform_time:>9.3f}s {pca_transform_time/n:>11.4f}s")
    print(f"{'Ripser PH':<35} {ripser_time:>9.2f}s {ripser_time/n:>11.3f}s")
    print(f"{'Feature extraction (PCA+PH+scalar)':<35} {feature_total_time:>9.2f}s {feature_total_time/n:>11.3f}s")
    print(f"{'Logistic regression prediction':<35} {lr_time:>9.5f}s {lr_time/n:>11.6f}s")
    print("-" * 60)
    print(f"{'TOTAL OVERHEAD (beyond inference)':<35} {'':>10} {overhead_per_problem:>11.3f}s")
    print(f"{'Overhead as % of inference':<35} {'':>10} {results['overhead_fraction']*100:>10.1f}%")
    print("=" * 65)

    if overhead_per_problem < 1.0:
        print("\nOverhead < 1s/problem: PRACTICAL for real-time use")
    else:
        print(f"\nOverhead {overhead_per_problem:.1f}s/problem: may need optimization")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", results_path)


if __name__ == "__main__":
    main()
