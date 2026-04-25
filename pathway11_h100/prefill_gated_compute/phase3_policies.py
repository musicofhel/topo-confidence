#!/usr/bin/env python3
"""Experiment 2 / Phase 3: compute-allocation policies on real K=8 majority-vote data.

Inputs:
  phase1_majority_vote.npz: per-problem expected majority-vote accuracy at K=1,2,4,8
  phase2_prefill_dom.npz: OOF prefill score, final-token score, seq_len, K=1 greedy correct

Key mechanic:
  A policy assigns each problem a K ∈ {1,2,4,8} (or "refuse"). The expected accuracy
  contribution of that problem is the cached mean_maj_at_K{k}[problem]. Answered
  accuracy averages over answered problems; overall accuracy = (sum of per-problem
  contributions on answered) / n_total (refused contribute 0).

Policies evaluated:
  (1) Uniform: every problem gets K ∈ {1,2,4,8}. Sanity baseline.
  (2) Threshold-gated by prefill score: τ sweep over 10..90th percentile.
      Above τ → K=1, below τ → K=8. Pareto (overall acc, avg K).
  (3) Quartile policies using prefill score quantiles:
        Balanced:  Q4→K=1, Q3→K=2, Q2→K=4, Q1→K=8         (avg K=3.75)
        Refuse-and-spend: Q4→K=1, Q3→K=4, Q2+Q1→refuse    (coverage=50%)
        Top-heavy: Q4+Q3→K=1, Q2+Q1→K=8                    (avg K=4.5)
      (Q4 = top quartile of prefill score = highest confidence)
  (4) Same quartile policies using:
        random gate (shuffled prefill score) — 100 seeds averaged
        final-token DoM score
        sequence-length score (shorter = higher confidence, consistent with seq_len↑ ⇒ correct↓)
        negative seq_len (longer = higher confidence; just a sanity inverse)
        mean_logprob (higher logprob = more confident)

Outputs phase3_policies.json with every policy's metrics + threshold sweep Pareto data.
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
    idx_map1 = {int(p): i for i, p in enumerate(ph1["problem_indices"])}
    idx_map2 = {int(p): i for i, p in enumerate(ph2["problem_indices"])}

    sel1 = np.array([idx_map1[int(p)] for p in common])
    sel2 = np.array([idx_map2[int(p)] for p in common])

    data = dict(
        problem_indices=common.astype(int),
        mean_maj_K1=ph1["mean_maj_at_K1"][sel1].astype(float),
        mean_maj_K2=ph1["mean_maj_at_K2"][sel1].astype(float),
        mean_maj_K4=ph1["mean_maj_at_K4"][sel1].astype(float),
        mean_maj_K8=ph1["mean_maj_at_K8"][sel1].astype(float),
        k8_maj_correct=np.asarray(ph1["k8_majority_correct"])[sel1].astype(bool),
        k1_greedy_correct=ph2["correct_k1"][sel2].astype(bool),
        prefill_score=ph2["prefill_score"][sel2].astype(float),
        final_token_score=ph2["final_token_score"][sel2].astype(float),
        seq_len=ph2["seq_len"][sel2].astype(int),
        mean_logprob=ph2["mean_logprob"][sel2].astype(float),
    )
    return data


def per_problem_acc_for_gate(gate: np.ndarray, data: dict) -> np.ndarray:
    """gate[p] ∈ {1,2,4,8,0=refuse}. Returns (n,) expected accuracy per problem.
    Refused → 0 contribution.
    """
    acc = np.zeros(len(gate), dtype=float)
    acc[gate == 1] = data["mean_maj_K1"][gate == 1]
    acc[gate == 2] = data["mean_maj_K2"][gate == 2]
    acc[gate == 4] = data["mean_maj_K4"][gate == 4]
    acc[gate == 8] = data["mean_maj_K8"][gate == 8]
    return acc


def evaluate(gate: np.ndarray, data: dict, name: str = "") -> dict:
    n = len(gate)
    answered = gate > 0
    n_answered = int(answered.sum())
    coverage = n_answered / n
    per_acc = per_problem_acc_for_gate(gate, data)
    acc_on_answered = float(per_acc[answered].mean()) if n_answered else 0.0
    overall_acc = float(per_acc.sum() / n)   # refused contribute 0
    avg_K_on_answered = float(gate[answered].mean()) if n_answered else 0.0
    # total compute = sum of K across all gated problems. For refused, we "save" the K we would have spent.
    total_compute = float(gate[answered].sum())  # in units of "one sample"
    return {
        "name": name,
        "n": n,
        "n_answered": n_answered,
        "coverage": coverage,
        "avg_K_on_answered": avg_K_on_answered,
        "total_compute": total_compute,
        "compute_per_problem": total_compute / n,
        "accuracy_on_answered": acc_on_answered,
        "overall_accuracy": overall_acc,
    }


def quartile_ranks(score: np.ndarray) -> np.ndarray:
    """Return an integer rank in {0,1,2,3} = {Q1, Q2, Q3, Q4} where Q4 is top."""
    # Use argsort-rank for stable, tie-broken buckets.
    ranks = score.argsort().argsort()  # 0..n-1
    n = len(score)
    q = np.zeros(n, dtype=int)
    q[ranks >= (3 * n) // 4] = 3  # Q4 (top 25%)
    q[(ranks >= n // 2) & (ranks < (3 * n) // 4)] = 2  # Q3
    q[(ranks >= n // 4) & (ranks < n // 2)] = 1  # Q2
    q[ranks < n // 4] = 0  # Q1 (bottom 25%)
    return q


def quartile_gate(score: np.ndarray, policy: str) -> np.ndarray:
    """Given a confidence score (higher = more confident), assign K per quartile."""
    q = quartile_ranks(score)
    gate = np.zeros(len(score), dtype=int)
    if policy == "balanced":
        gate[q == 3] = 1
        gate[q == 2] = 2
        gate[q == 1] = 4
        gate[q == 0] = 8
    elif policy == "refuse_and_spend":
        gate[q == 3] = 1
        gate[q == 2] = 4
        gate[q == 1] = 0  # refuse
        gate[q == 0] = 0  # refuse
    elif policy == "top_heavy":
        gate[q == 3] = 1
        gate[q == 2] = 1
        gate[q == 1] = 8
        gate[q == 0] = 8
    else:
        raise ValueError(policy)
    return gate


def threshold_gate(score: np.ndarray, tau_pct: float) -> np.ndarray:
    """Binary gate: if score >= percentile τ → K=1, else K=8."""
    thr = np.percentile(score, tau_pct)
    gate = np.where(score >= thr, 1, 8).astype(int)
    return gate


def main():
    print("=" * 70)
    print("Experiment 2 / Phase 3: allocation policies & Pareto sweep")
    print("=" * 70)

    data = load_inputs()
    n = len(data["problem_indices"])
    print(f"  n_problems (intersection) = {n}")
    print(f"  K=1 uniform acc = {data['mean_maj_K1'].mean():.4f}")
    print(f"  K=2 uniform acc = {data['mean_maj_K2'].mean():.4f}")
    print(f"  K=4 uniform acc = {data['mean_maj_K4'].mean():.4f}")
    print(f"  K=8 uniform acc = {data['mean_maj_K8'].mean():.4f}")

    policies = {}

    # 1) Uniform baselines
    for K in [1, 2, 4, 8]:
        gate = np.full(n, K, dtype=int)
        policies[f"uniform_K{K}"] = evaluate(gate, data, name=f"uniform K={K}")

    # 2) Threshold gate sweep on prefill score
    pareto = []
    for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        gate = threshold_gate(data["prefill_score"], pct)
        r = evaluate(gate, data, name=f"threshold_prefill_{pct}pct")
        r["tau_percentile"] = pct
        pareto.append(r)
    policies["threshold_prefill_sweep"] = pareto

    # Also sweep on the other scores for comparison
    for score_name, score in [
        ("final_token", data["final_token_score"]),
        ("neg_seq_len", -data["seq_len"].astype(float)),  # negative so "confidence" = short sequence
        ("mean_logprob", data["mean_logprob"]),
    ]:
        sweep = []
        for pct in [10, 25, 50, 75, 90]:
            gate = threshold_gate(score, pct)
            r = evaluate(gate, data, name=f"threshold_{score_name}_{pct}pct")
            r["tau_percentile"] = pct
            sweep.append(r)
        policies[f"threshold_{score_name}_sweep"] = sweep

    # 3) Quartile policies (prefill)
    for pol in ["balanced", "refuse_and_spend", "top_heavy"]:
        gate = quartile_gate(data["prefill_score"], pol)
        policies[f"quartile_prefill_{pol}"] = evaluate(gate, data, name=f"prefill/{pol}")

    # 4) Same quartile policies on other scores
    for score_name, score in [
        ("final_token", data["final_token_score"]),
        ("neg_seq_len", -data["seq_len"].astype(float)),
        ("mean_logprob", data["mean_logprob"]),
    ]:
        for pol in ["balanced", "refuse_and_spend", "top_heavy"]:
            gate = quartile_gate(score, pol)
            policies[f"quartile_{score_name}_{pol}"] = evaluate(
                gate, data, name=f"{score_name}/{pol}"
            )

    # 5) Random gate (shuffled prefill score): average over 100 seeds
    rng = np.random.default_rng(RNG_SEED)
    for pol in ["balanced", "refuse_and_spend", "top_heavy"]:
        n_draws = 100
        accs = []
        covs = []
        avg_Ks = []
        for _ in range(n_draws):
            shuffled = data["prefill_score"].copy()
            rng.shuffle(shuffled)
            gate = quartile_gate(shuffled, pol)
            r = evaluate(gate, data)
            accs.append(r["overall_accuracy"])
            covs.append(r["coverage"])
            avg_Ks.append(r["avg_K_on_answered"])
        policies[f"quartile_random_{pol}"] = {
            "name": f"random/{pol} (avg over {n_draws})",
            "overall_accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "coverage": float(np.mean(covs)),
            "avg_K_on_answered": float(np.mean(avg_Ks)),
            "n_draws": n_draws,
        }

    # Pretty print the headline comparison
    print("\n=== Policy comparison ===")
    print(f"{'policy':<55} {'cov':>6} {'avgK':>6} {'acc_ans':>8} {'overall':>8}")
    for key in [
        "uniform_K1", "uniform_K2", "uniform_K4", "uniform_K8",
        "quartile_prefill_balanced",
        "quartile_final_token_balanced",
        "quartile_neg_seq_len_balanced",
        "quartile_mean_logprob_balanced",
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
        print(f"  {key:<55} {r['coverage']:>6.3f} {r['avg_K_on_answered']:>6.2f} "
              f"{r.get('accuracy_on_answered', r['overall_accuracy']/max(r['coverage'],1e-9)):>8.4f} "
              f"{r['overall_accuracy']:>8.4f}")

    print("\n=== Threshold (prefill) Pareto ===")
    print(f"{'tau%':>5} {'avg_K':>7} {'overall_acc':>13}")
    for r in pareto:
        print(f"  {r['tau_percentile']:>5} {r['avg_K_on_answered']:>7.2f} {r['overall_accuracy']:>13.4f}")

    # Save.
    with open(OUT_DIR / "phase3_policies.json", "w") as f:
        json.dump(policies, f, indent=2, default=lambda x: (float(x) if hasattr(x, "__float__") else str(x)))
    print(f"\nSaved: {OUT_DIR/'phase3_policies.json'}")


if __name__ == "__main__":
    main()
