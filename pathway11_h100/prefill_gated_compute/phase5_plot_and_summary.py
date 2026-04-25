#!/usr/bin/env python3
"""Experiment 2 / Phase 5: Pareto plot + final summary.

Generates pareto_plot.png and results.json.
Also tests a "middle-heavy" quartile policy (spend K=8 on middle quartile, K=1 on extremes)
since Phase 4 showed B (recoverable) problems live at mid-prefill-score, not the bottom.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/prefill_gated_compute"


def load():
    ph1 = np.load(OUT_DIR / "phase1_majority_vote.npz", allow_pickle=True)
    ph2 = np.load(OUT_DIR / "phase2_prefill_dom.npz", allow_pickle=True)
    common = np.intersect1d(ph1["problem_indices"], ph2["problem_indices"])
    im1 = {int(p): i for i, p in enumerate(ph1["problem_indices"])}
    im2 = {int(p): i for i, p in enumerate(ph2["problem_indices"])}
    sel1 = np.array([im1[int(p)] for p in common])
    sel2 = np.array([im2[int(p)] for p in common])
    return dict(
        greedy_K1=ph2["correct_k1"][sel2].astype(float),
        k2=ph1["mean_maj_at_K2"][sel1].astype(float),
        k4=ph1["mean_maj_at_K4"][sel1].astype(float),
        k8=ph1["mean_maj_at_K8"][sel1].astype(float),
        prefill_score=ph2["prefill_score"][sel2].astype(float),
        final_token_score=ph2["final_token_score"][sel2].astype(float),
        seq_len=ph2["seq_len"][sel2].astype(int),
    )


def quartile_ranks(score):
    r = score.argsort().argsort()
    n = len(score)
    q = np.zeros(n, dtype=int)
    q[r >= (3 * n) // 4] = 3
    q[(r >= n // 2) & (r < (3 * n) // 4)] = 2
    q[(r >= n // 4) & (r < n // 2)] = 1
    q[r < n // 4] = 0
    return q


def evaluate(gate, data):
    n = len(gate)
    acc = np.zeros(n)
    acc[gate == 1] = data["greedy_K1"][gate == 1]
    acc[gate == 2] = data["k2"][gate == 2]
    acc[gate == 4] = data["k4"][gate == 4]
    acc[gate == 8] = data["k8"][gate == 8]
    ans = gate > 0
    return dict(
        coverage=float(ans.mean()),
        avg_K=float(gate[ans].mean()) if ans.any() else 0.0,
        compute_per_problem=float(gate[ans].sum() / n),
        accuracy_on_answered=float(acc[ans].mean()) if ans.any() else 0.0,
        overall_accuracy=float(acc.sum() / n),
    )


def threshold_gate(score, pct):
    thr = np.percentile(score, pct)
    return np.where(score >= thr, 1, 8).astype(int)


def main():
    d = load()
    n = len(d["greedy_K1"])

    # Middle-heavy: K=8 on middle quartiles, K=1 on extremes
    pf = d["prefill_score"]
    q = quartile_ranks(pf)
    gate_mid = np.zeros(n, dtype=int)
    gate_mid[q == 3] = 1
    gate_mid[q == 2] = 8
    gate_mid[q == 1] = 8
    gate_mid[q == 0] = 1
    r_mid = evaluate(gate_mid, d)
    print(f"middle-heavy (Q4,Q1→K=1; Q3,Q2→K=8) avg_K={r_mid['avg_K']:.2f} overall={r_mid['overall_accuracy']:.4f}")

    # Plot Pareto: avg compute per problem vs accuracy
    # Curves: threshold sweep on prefill, final_token, neg_seq_len. Uniform points.
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Uniform
    uniform_K = [1, 2, 4, 8]
    uniform_acc = [d["greedy_K1"].mean(), d["k2"].mean(), d["k4"].mean(), d["k8"].mean()]
    ax.plot(uniform_K, uniform_acc, "o-", color="black", lw=2, ms=8, label="uniform K", zorder=5)
    for k, a in zip(uniform_K, uniform_acc):
        ax.annotate(f"K={k}", (k, a), textcoords="offset points", xytext=(8, -2), fontsize=9)

    # Threshold sweeps
    sweep_specs = [
        ("prefill DoM", d["prefill_score"], "tab:blue"),
        ("final-token DoM", d["final_token_score"], "tab:orange"),
        ("neg seq-len", -d["seq_len"].astype(float), "tab:green"),
    ]
    pcts = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    for name, score, color in sweep_specs:
        xs, ys = [], []
        for pct in pcts:
            gate = threshold_gate(score, pct)
            r = evaluate(gate, d)
            xs.append(r["compute_per_problem"])
            ys.append(r["overall_accuracy"])
        ax.plot(xs, ys, "s-", color=color, alpha=0.7, label=f"threshold-{name}")

    # Quartile balanced (3 policies, single point each)
    for name, score, marker, color in [
        ("prefill quartile-balanced", d["prefill_score"], "D", "tab:blue"),
        ("final-token quartile-balanced", d["final_token_score"], "D", "tab:orange"),
        ("neg seq-len quartile-balanced", -d["seq_len"].astype(float), "D", "tab:green"),
    ]:
        q_ = quartile_ranks(score)
        gate = np.zeros(n, dtype=int)
        gate[q_ == 3] = 1; gate[q_ == 2] = 2; gate[q_ == 1] = 4; gate[q_ == 0] = 8
        r = evaluate(gate, d)
        ax.plot(r["compute_per_problem"], r["overall_accuracy"], marker,
                color=color, ms=12, mec="black", label=name)

    # Middle-heavy
    ax.plot(r_mid["compute_per_problem"], r_mid["overall_accuracy"], "P",
            color="tab:red", ms=14, mec="black", label="prefill middle-heavy")

    ax.set_xlabel("Compute per problem (avg # of samples)")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Pareto: compute-allocation policies on MATH-500, Qwen-2.5-1.5B-Instruct\n"
                 f"greedy K=1: {d['greedy_K1'].mean():.3f} | K=8 majority: {d['k8'].mean():.3f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.5, 8.5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "pareto_plot.png", dpi=140)
    print(f"Saved: {OUT_DIR/'pareto_plot.png'}")

    # Final consolidated summary
    summary = {
        "experiment": "prefill-gated compute allocation on MATH-500 × Qwen-2.5-1.5B",
        "n_problems": n,
        "data_sources": {
            "K_greedy_labels": "pathway8_layerwise/data/math500/ (1024-token, T=0, Stage 2 of Pathway 11)",
            "K_8_samples": "pathway11_h100/data/k8_selfconsistency/ (T=0.7, K=8, Stage 3 of Pathway 11)",
        },
        "accuracy_at_uniform_K": {
            "K=1 (greedy T=0)": float(d["greedy_K1"].mean()),
            "K=2 majority (T=0.7)": float(d["k2"].mean()),
            "K=4 majority (T=0.7)": float(d["k4"].mean()),
            "K=8 majority (T=0.7)": float(d["k8"].mean()),
        },
        "prefill_DoM_auroc_oof": 0.7731,
        "final_token_DoM_auroc_oof": 0.7186,
        "bucket_sizes": {
            "A_always_right": 207, "B_recoverable": 68, "C_never_right": 189, "D_pathological": 36,
            "K1_recovers_prob": 243 / 500, "K8_recovers_prob": 275 / 500,
        },
        "oracle_compute_per_problem": 1.01,
        "oracle_overall_accuracy": 0.589,
        "headline": {
            "does_prefill_gating_beat_uniform_K8_at_lower_K": False,
            "best_threshold_prefill_gate_at_avgK_~4": {
                "tau_40pct": {"avg_K": 3.80, "overall_acc": 0.4920},
                "tau_50pct": {"avg_K": 4.50, "overall_acc": 0.4920},
                "uniform_K4_reference": {"avg_K": 4.00, "overall_acc": 0.4950},
            },
            "refuse_and_spend_coverage_0.5": {
                "prefill": {"acc_on_answered": 0.716, "avg_K_on_answered": 2.5},
                "neg_seq_len": {"acc_on_answered": 0.707, "avg_K_on_answered": 2.5},
                "random": {"acc_on_answered": 0.494, "avg_K_on_answered": 2.5},
            },
            "finding_1": (
                "Prefill L19 DoM is a strong correctness signal (AUROC 0.77 OOF for K=1 correctness, "
                "0.82 for K=8-majority correctness). Sequence-length is a nearly-equal signal "
                "(AUROC 0.80 for K=1), but only available post-hoc."
            ),
            "finding_2": (
                "Despite the strong signal, prefill gating does NOT Pareto-dominate uniform K. "
                "At compute ~4 per problem, uniform K=4 gives 0.495; prefill threshold gating gives 0.492. "
                "Quartile-balanced prefill gating gives 0.463 at compute 3.75 — worse than uniform K=4."
            ),
            "finding_3": (
                "Root cause: K=8 majority vote only recovers 68/257 = 26.5% of K=1 failures on "
                "MATH-500 × 1.5B at T=0.7. 189/500 = 37.8% of problems are 'never-right' even at K=8, "
                "so routing K=8 to low-confidence problems mostly burns compute."
            ),
            "finding_4": (
                "Prefill score separates (A∪B) 'ever-right' from C 'hopeless' well (AUROC 0.82), "
                "BUT its relationship with oracle-K is weak (Spearman 0.175). Recoverable problems "
                "(B) live at MID prefill score, not bottom — they're hard-to-tell-apart from C with a "
                "monotonic policy."
            ),
            "finding_5": (
                "The one clean win for prefill gating: refuse-and-spend at coverage=0.5. "
                "Keep the top-half by prefill score, answer at K=1 (Q4) or K=4 (Q3) → accuracy on "
                "answered = 0.716 with avg K=2.5, versus random refuse-and-spend at 0.494. "
                "That's a 22pp lift in accuracy-on-answered for selective prediction."
            ),
            "finding_6": (
                "Pathological bucket D (K=1 right, K=8 majority wrong) is 36/500 = 7.2% — non-trivial. "
                "Spending K=8 on these problems ACTIVELY HURTS accuracy. Any gating policy needs to "
                "route D-bucket problems to K=1, not K=8."
            ),
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
