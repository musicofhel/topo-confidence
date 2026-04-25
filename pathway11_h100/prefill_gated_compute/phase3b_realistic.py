#!/usr/bin/env python3
"""Experiment 2 / Phase 3b: realistic deployment variant.

In Phase 3, K=1 allocation used one sample from the T=0.7 pool (per-sample acc ≈ 0.41).
A more realistic deployment choice:
  * Confident problems → greedy K=1 at T=0 (Stage 2 labels: 0.486)
  * Uncertain problems → T=0.7 majority vote of K samples (Stage 3 data)

This is the canonical "spend extra compute on hard problems" setup.

Recomputes the same policies with this variant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/prefill_gated_compute"

RNG_SEED = 9999


def load_inputs():
    ph1 = np.load(OUT_DIR / "phase1_majority_vote.npz", allow_pickle=True)
    ph2 = np.load(OUT_DIR / "phase2_prefill_dom.npz", allow_pickle=True)
    common = np.intersect1d(ph1["problem_indices"], ph2["problem_indices"])
    im1 = {int(p): i for i, p in enumerate(ph1["problem_indices"])}
    im2 = {int(p): i for i, p in enumerate(ph2["problem_indices"])}
    sel1 = np.array([im1[int(p)] for p in common])
    sel2 = np.array([im2[int(p)] for p in common])
    return dict(
        problem_indices=common.astype(int),
        greedy_K1=ph2["correct_k1"][sel2].astype(float),  # 0/1 greedy correctness
        mean_maj_K2=ph1["mean_maj_at_K2"][sel1].astype(float),
        mean_maj_K4=ph1["mean_maj_at_K4"][sel1].astype(float),
        mean_maj_K8=ph1["mean_maj_at_K8"][sel1].astype(float),
        prefill_score=ph2["prefill_score"][sel2].astype(float),
        final_token_score=ph2["final_token_score"][sel2].astype(float),
        seq_len=ph2["seq_len"][sel2].astype(int),
    )


def acc_per_problem(gate, data):
    acc = np.zeros(len(gate))
    acc[gate == 1] = data["greedy_K1"][gate == 1]
    acc[gate == 2] = data["mean_maj_K2"][gate == 2]
    acc[gate == 4] = data["mean_maj_K4"][gate == 4]
    acc[gate == 8] = data["mean_maj_K8"][gate == 8]
    return acc


def evaluate(gate, data, name=""):
    n = len(gate)
    ans = gate > 0
    per = acc_per_problem(gate, data)
    return {
        "name": name,
        "n": n,
        "n_answered": int(ans.sum()),
        "coverage": float(ans.sum() / n),
        "avg_K_on_answered": float(gate[ans].mean()) if ans.any() else 0.0,
        "total_compute": float(gate[ans].sum()),
        "compute_per_problem": float(gate[ans].sum() / n),
        "accuracy_on_answered": float(per[ans].mean()) if ans.any() else 0.0,
        "overall_accuracy": float(per.sum() / n),
    }


def quartile_ranks(score):
    r = score.argsort().argsort()
    n = len(score)
    q = np.zeros(n, dtype=int)
    q[r >= (3 * n) // 4] = 3
    q[(r >= n // 2) & (r < (3 * n) // 4)] = 2
    q[(r >= n // 4) & (r < n // 2)] = 1
    q[r < n // 4] = 0
    return q


def quartile_gate(score, policy):
    q = quartile_ranks(score)
    g = np.zeros(len(score), dtype=int)
    if policy == "balanced":
        g[q == 3] = 1; g[q == 2] = 2; g[q == 1] = 4; g[q == 0] = 8
    elif policy == "refuse_and_spend":
        g[q == 3] = 1; g[q == 2] = 4; g[q == 1] = 0; g[q == 0] = 0
    elif policy == "top_heavy":
        g[q == 3] = 1; g[q == 2] = 1; g[q == 1] = 8; g[q == 0] = 8
    else:
        raise ValueError(policy)
    return g


def threshold_gate(score, pct):
    thr = np.percentile(score, pct)
    return np.where(score >= thr, 1, 8).astype(int)


def main():
    print("=" * 70)
    print("Experiment 2 / Phase 3b: REALISTIC variant (confident → greedy K=1 @ T=0)")
    print("=" * 70)
    data = load_inputs()
    n = len(data["problem_indices"])
    print(f"  greedy K=1 uniform = {data['greedy_K1'].mean():.4f}")
    print(f"  K=8 majority uniform = {data['mean_maj_K8'].mean():.4f}")
    print(f"  gap = {data['mean_maj_K8'].mean() - data['greedy_K1'].mean():+.4f}")

    policies = {}
    # Uniform baselines. (Uniform K=1 now uses greedy T=0, not sampled T=0.7.)
    for K, src in [(1, "greedy_K1"), (2, "mean_maj_K2"), (4, "mean_maj_K4"), (8, "mean_maj_K8")]:
        gate = np.full(n, K, dtype=int)
        policies[f"uniform_K{K}"] = evaluate(gate, data, f"uniform K={K}")

    # Threshold sweep (prefill)
    pareto_prefill = []
    for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        gate = threshold_gate(data["prefill_score"], pct)
        r = evaluate(gate, data, f"threshold_prefill_{pct}pct")
        r["tau_percentile"] = pct
        pareto_prefill.append(r)
    policies["threshold_prefill_sweep"] = pareto_prefill

    # Threshold sweeps for comparison signals
    for sn, sc in [
        ("final_token", data["final_token_score"]),
        ("neg_seq_len", -data["seq_len"].astype(float)),
    ]:
        sweep = []
        for pct in [10, 25, 50, 75, 90]:
            gate = threshold_gate(sc, pct)
            r = evaluate(gate, data, f"threshold_{sn}_{pct}pct")
            r["tau_percentile"] = pct
            sweep.append(r)
        policies[f"threshold_{sn}_sweep"] = sweep

    # Quartile policies
    for pol in ["balanced", "refuse_and_spend", "top_heavy"]:
        gate = quartile_gate(data["prefill_score"], pol)
        policies[f"quartile_prefill_{pol}"] = evaluate(gate, data, f"prefill/{pol}")
        for sn, sc in [
            ("final_token", data["final_token_score"]),
            ("neg_seq_len", -data["seq_len"].astype(float)),
        ]:
            gate = quartile_gate(sc, pol)
            policies[f"quartile_{sn}_{pol}"] = evaluate(gate, data, f"{sn}/{pol}")

    # Random gate (shuffled prefill score)
    rng = np.random.default_rng(RNG_SEED)
    for pol in ["balanced", "refuse_and_spend", "top_heavy"]:
        accs, covs, avgKs = [], [], []
        for _ in range(100):
            s = data["prefill_score"].copy()
            rng.shuffle(s)
            gate = quartile_gate(s, pol)
            r = evaluate(gate, data)
            accs.append(r["overall_accuracy"])
            covs.append(r["coverage"])
            avgKs.append(r["avg_K_on_answered"])
        policies[f"quartile_random_{pol}"] = {
            "name": f"random/{pol}", "overall_accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "coverage": float(np.mean(covs)),
            "avg_K_on_answered": float(np.mean(avgKs)),
            "n_draws": 100,
        }

    # Pretty print
    print("\n=== Policy comparison (REALISTIC: K=1 = greedy T=0) ===")
    print(f"{'policy':<50} {'cov':>6} {'avgK':>6} {'acc_ans':>8} {'overall':>8}")
    for key in [
        "uniform_K1", "uniform_K2", "uniform_K4", "uniform_K8",
        "quartile_prefill_balanced",
        "quartile_final_token_balanced",
        "quartile_neg_seq_len_balanced",
        "quartile_random_balanced",
        "quartile_prefill_refuse_and_spend",
        "quartile_final_token_refuse_and_spend",
        "quartile_neg_seq_len_refuse_and_spend",
        "quartile_random_refuse_and_spend",
        "quartile_prefill_top_heavy",
        "quartile_final_token_top_heavy",
        "quartile_neg_seq_len_top_heavy",
        "quartile_random_top_heavy",
    ]:
        r = policies[key]
        acc_ans = r.get("accuracy_on_answered",
                        r["overall_accuracy"] / max(r["coverage"], 1e-9))
        print(f"  {key:<50} {r['coverage']:>6.3f} {r['avg_K_on_answered']:>6.2f} "
              f"{acc_ans:>8.4f} {r['overall_accuracy']:>8.4f}")

    print("\n=== Threshold (prefill) Pareto ===")
    print(f"{'tau%':>5} {'avg_K':>7} {'overall':>9}")
    for r in pareto_prefill:
        print(f"  {r['tau_percentile']:>5} {r['avg_K_on_answered']:>7.2f} "
              f"{r['overall_accuracy']:>9.4f}")

    with open(OUT_DIR / "phase3b_realistic_policies.json", "w") as f:
        json.dump(policies, f, indent=2, default=lambda x: (float(x) if hasattr(x, "__float__") else str(x)))
    print(f"\nSaved: {OUT_DIR/'phase3b_realistic_policies.json'}")


if __name__ == "__main__":
    main()
