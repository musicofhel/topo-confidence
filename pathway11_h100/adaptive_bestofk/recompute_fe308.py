"""FE308 — Adaptive best-of-k Damani allocator.

Uses DoM scores as per-problem confidence λ̂ to allocate variable K across
problems. Compares adaptive majority-vote accuracy against uniform-K baselines
at matched average K. Note: this is an ALL-answer protocol, NOT selective
prediction (F-8's 71.6% at coverage 0.5 is a different comparison).

Output: pathway11_h100/results/fe308_adaptive_bestofk.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/musicofhel/topo-confidence")
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe308_adaptive_bestofk.json"

N_PROBLEMS = 500
K_MAX = 8
N_BINS = 5
TARGET_AVG_KS = [1.0, 2.0, 2.5, 4.0, 8.0]
UNIFORM_KS = [1, 2, 4, 8]
N_TIE_SEEDS = 1000
SEED = 9999


def majority_vote(correct_array: np.ndarray, k: int, rng=None) -> bool:
    """Majority vote of first k samples. Ties broken randomly."""
    votes = correct_array[:k]
    n_correct = votes.sum()
    n_incorrect = k - n_correct
    if n_correct > n_incorrect:
        return True
    elif n_incorrect > n_correct:
        return False
    else:
        if rng is not None:
            return bool(rng.random() < 0.5)
        return bool(np.random.random() < 0.5)


def main() -> int:
    # Load DoM scores
    dom_data = np.load(DOM_NPZ)
    dom_score = dom_data["prefill_score"].astype(np.float64)

    # Normalize to [0, 1] by rank percentile
    lambda_hat = (rankdata(dom_score) - 1) / (N_PROBLEMS - 1)

    # Load K=8 cache
    print("Loading K=8 cache (500 files)...")
    k8_correct = np.zeros((N_PROBLEMS, K_MAX), dtype=bool)

    for i in range(N_PROBLEMS):
        fp = K8_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            k8_correct[i] = d["correct"].astype(bool)[:K_MAX]

    print(f"K=8 majority vote accuracy: {sum(k8_correct[:, :8].sum(axis=1) > 4 for _ in [0])}")

    # Uniform baselines
    rng = np.random.default_rng(SEED)
    uniform_results = {}

    for k in UNIFORM_KS:
        if k % 2 == 1:
            # Odd K: no ties
            acc = sum(k8_correct[i, :k].sum() > k // 2 for i in range(N_PROBLEMS)) / N_PROBLEMS
            uniform_results[str(k)] = {
                "k": k, "accuracy": float(acc), "n_correct": int(acc * N_PROBLEMS),
            }
        else:
            # Even K: average over tie-break seeds
            accs = []
            for seed_offset in range(N_TIE_SEEDS):
                tie_rng = np.random.default_rng(SEED + seed_offset)
                n_right = 0
                for i in range(N_PROBLEMS):
                    if majority_vote(k8_correct[i], k, tie_rng):
                        n_right += 1
                accs.append(n_right / N_PROBLEMS)
            uniform_results[str(k)] = {
                "k": k,
                "accuracy": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "n_correct_mean": float(np.mean(accs) * N_PROBLEMS),
            }
        print(f"Uniform K={k}: accuracy={uniform_results[str(k)]['accuracy']:.4f}")

    # Damani adaptive allocator
    adaptive_results = {}

    for target_avg_k in TARGET_AVG_KS:
        total_budget = int(target_avg_k * N_PROBLEMS)

        # Sort by ascending lambda_hat (low confidence first)
        sorted_idx = np.argsort(lambda_hat)
        bin_size = N_PROBLEMS // N_BINS

        # Compute per-bin median lambda
        bin_medians = []
        bin_indices = []
        for b in range(N_BINS):
            start = b * bin_size
            end = start + bin_size if b < N_BINS - 1 else N_PROBLEMS
            bin_idx = sorted_idx[start:end]
            bin_indices.append(bin_idx)
            bin_medians.append(np.median(lambda_hat[bin_idx]))

        # Allocate K proportional to (1 - median_lambda)
        raw_weights = np.array([1 - m for m in bin_medians])
        raw_weights = np.maximum(raw_weights, 0.01)

        # Solve for K_b such that sum(K_b * n_b) = total_budget
        bin_sizes = [len(bi) for bi in bin_indices]
        weighted_sum = sum(w * s for w, s in zip(raw_weights, bin_sizes))
        scale = total_budget / weighted_sum

        k_per_bin = np.clip(np.round(raw_weights * scale).astype(int), 1, K_MAX)

        # Adjust to hit budget exactly
        actual_budget = sum(k_per_bin[b] * bin_sizes[b] for b in range(N_BINS))
        while actual_budget > total_budget:
            # Reduce from highest-confidence bin
            for b in range(N_BINS - 1, -1, -1):
                if k_per_bin[b] > 1:
                    k_per_bin[b] -= 1
                    actual_budget -= bin_sizes[b]
                    if actual_budget <= total_budget:
                        break

        actual_avg_k = sum(k_per_bin[b] * bin_sizes[b] for b in range(N_BINS)) / N_PROBLEMS

        # Evaluate
        accs = []
        for seed_offset in range(100):  # fewer seeds for adaptive since it's many configs
            tie_rng = np.random.default_rng(SEED + seed_offset)
            n_right = 0
            for b in range(N_BINS):
                kb = int(k_per_bin[b])
                for i in bin_indices[b]:
                    if majority_vote(k8_correct[i], kb, tie_rng):
                        n_right += 1
            accs.append(n_right / N_PROBLEMS)

        adaptive_results[str(target_avg_k)] = {
            "target_avg_k": target_avg_k,
            "actual_avg_k": float(actual_avg_k),
            "k_per_bin": k_per_bin.tolist(),
            "bin_sizes": bin_sizes,
            "bin_medians": [float(m) for m in bin_medians],
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
        }
        print(f"Adaptive avg_K={target_avg_k}: actual_avg_K={actual_avg_k:.2f}  "
              f"accuracy={np.mean(accs):.4f}  K_bins={k_per_bin.tolist()}")

    # Greedy K=1 reference (from cache, NOT canonical greedy T=0)
    cache_k1_acc = float(k8_correct[:, 0].mean())
    print(f"\nCache K=1 (T>0): {cache_k1_acc:.4f}")
    print(f"Canonical K=1 (T=0): 0.486")

    out = {
        "experiment": "FE308",
        "description": "Adaptive best-of-k Damani allocator: DoM-guided K allocation",
        "n": N_PROBLEMS,
        "k_max": K_MAX,
        "n_bins": N_BINS,
        "uniform_baselines": uniform_results,
        "adaptive_results": adaptive_results,
        "cache_k1_accuracy": cache_k1_acc,
        "canonical_k1_greedy_accuracy": 0.486,
        "canonical_k8_majority": 0.438,
        "references": {
            "f8_selective_71_6_at_cov_0_5": "DIFFERENT protocol (refuse bottom half)",
        },
        "interpretation": (
            "Adaptive allocation should outperform uniform at the same avg K. "
            "The margin quantifies how much DoM scores help allocate compute. "
            "Note: K=8 majority (43.8%) < K=1 greedy (48.6%), so naive majority "
            "vote HURTS — adaptive allocation must overcome this."
        ),
        "meta": {
            "dom_npz": str(DOM_NPZ),
            "k8_dir": str(K8_DIR),
            "seed": SEED,
            "n_tie_seeds_uniform": N_TIE_SEEDS,
            "n_tie_seeds_adaptive": 100,
            "method": "damani_greedy_bin_allocation",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
