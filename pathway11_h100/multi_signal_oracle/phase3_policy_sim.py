#!/usr/bin/env python3
"""Exp 2b / Phase 3: policy simulation from classifier OOF predictions.

For each problem, use classifier prediction (class 0/1/2) to route:
  class 0 -> K=1 (cost=1; correct = k1_correct_greedy[i])
  class 1 -> K=8 (cost=8; correct = k8_majority_correct[i])
  class 2 -> refuse (cost=0; not counted in answered pool)

Metrics (matches Exp 2 conventions):
  - coverage           = answered / 500
  - avg_K_on_answered  = mean compute over answered
  - compute_per_problem = total compute / 500
  - accuracy_on_answered
  - overall_accuracy   = total_correct / 500  (refused count as not-correct)

Compare to Exp 2 policies at matched compute.
Compute D-bucket outcome: fraction of 36 D-problems routed to K=1 and got correct.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/multi_signal_oracle"
EXP2 = ROOT / "pathway11_h100/prefill_gated_compute"


def simulate(y_pred: np.ndarray, k1_correct: np.ndarray, k8_correct: np.ndarray,
             fallback: str = "none"):
    """fallback = 'none' (refuse = 0 compute, counted wrong)
                  'k1'   (refuse → run K=1, answer whatever)."""
    n = len(y_pred)
    compute = np.zeros(n, dtype=int)
    answered = np.zeros(n, dtype=bool)
    correct = np.zeros(n, dtype=bool)

    for i in range(n):
        if y_pred[i] == 0:
            compute[i] = 1; answered[i] = True; correct[i] = bool(k1_correct[i])
        elif y_pred[i] == 1:
            compute[i] = 8; answered[i] = True; correct[i] = bool(k8_correct[i])
        else:
            if fallback == "k1":
                compute[i] = 1; answered[i] = True; correct[i] = bool(k1_correct[i])
            else:
                compute[i] = 0; answered[i] = False; correct[i] = False

    n_ans = int(answered.sum())
    total_compute = int(compute.sum())
    n_correct = int(correct.sum())
    return dict(
        n=int(n), n_answered=n_ans,
        coverage=n_ans / n,
        avg_K_on_answered=float(compute[answered].mean()) if n_ans > 0 else 0.0,
        compute_per_problem=total_compute / n,
        accuracy_on_answered=(n_correct / n_ans) if n_ans > 0 else 0.0,
        overall_accuracy=n_correct / n,
        total_compute=total_compute,
        fallback=fallback,
        # Breakdown:
        n_routed_K1=int((y_pred == 0).sum()),
        n_routed_K8=int((y_pred == 1).sum()),
        n_refused=int((y_pred == 2).sum()),
        correct_on_K1_route=int((correct & (y_pred == 0)).sum()),
        correct_on_K8_route=int((correct & (y_pred == 1)).sum()),
    )


def d_bucket_analysis(y_pred: np.ndarray, bkt: np.ndarray,
                       k1_correct: np.ndarray, k8_correct: np.ndarray):
    d_idx = np.where(bkt == "D")[0]
    routed = y_pred[d_idx]
    n_k1 = int((routed == 0).sum())
    n_k8 = int((routed == 1).sum())
    n_ref = int((routed == 2).sum())
    # D problems: K=1 correct, K=8 wrong. So:
    #   routing to K=1 → correct; routing to K=8 → wrong; refuse → not answered
    correct_if_k1 = n_k1  # all D problems have k1_correct=True by definition
    correct_if_k8 = 0     # all D problems have k8_correct=False by definition
    # Sanity check
    assert k1_correct[d_idx].all(), "D-bucket should have all K=1 correct"
    assert not k8_correct[d_idx].any(), "D-bucket should have all K=8 majority wrong"
    return dict(
        n_D=int(len(d_idx)),
        routed_to_K1=n_k1, routed_to_K8=n_k8, refused=n_ref,
        D_recall_to_K1=n_k1 / len(d_idx) if len(d_idx) > 0 else 0.0,
        correct_after_routing=int(correct_if_k1),
    )


def a_bucket_analysis(y_pred, bkt):
    """Bucket A should route to K=1 ideally — same compute."""
    a_idx = np.where(bkt == "A")[0]
    r = y_pred[a_idx]
    return dict(
        n_A=int(len(a_idx)),
        routed_to_K1=int((r == 0).sum()),
        routed_to_K8=int((r == 1).sum()),
        refused=int((r == 2).sum()),
    )


def b_bucket_analysis(y_pred, bkt):
    """Bucket B should route to K=8 to recover correctness."""
    b_idx = np.where(bkt == "B")[0]
    r = y_pred[b_idx]
    return dict(
        n_B=int(len(b_idx)),
        routed_to_K1=int((r == 0).sum()),
        routed_to_K8=int((r == 1).sum()),
        refused=int((r == 2).sum()),
        B_recall_to_K8=int((r == 1).sum()) / len(b_idx) if len(b_idx) > 0 else 0.0,
    )


def c_bucket_analysis(y_pred, bkt):
    """Bucket C should ideally be refused (saves compute AND avoids wrong answers)."""
    c_idx = np.where(bkt == "C")[0]
    r = y_pred[c_idx]
    return dict(
        n_C=int(len(c_idx)),
        routed_to_K1=int((r == 0).sum()),
        routed_to_K8=int((r == 1).sum()),
        refused=int((r == 2).sum()),
        C_recall_to_refuse=int((r == 2).sum()) / len(c_idx) if len(c_idx) > 0 else 0.0,
    )


def main():
    print("=" * 70)
    print("Exp 2b / Phase 3: policy simulation")
    print("=" * 70)

    feats = np.load(OUT_DIR / "features.npz", allow_pickle=True)
    k1_correct = feats["k1_correct"].astype(bool)
    k8_correct = feats["k8_correct"].astype(bool)
    bkt = feats["bucket"]

    results = dict(
        n_problems=int(len(k1_correct)),
        baseline_metrics=dict(
            k1_acc=float(k1_correct.mean()),
            k8_acc=float(k8_correct.mean()),
        ),
    )

    for clf_kind in ["logreg", "rf"]:
        print(f"\n--- Policy simulation: {clf_kind} ---")
        oof = np.load(OUT_DIR / f"oof_{clf_kind}.npz")
        y_pred = oof["y_pred"]
        y_true = oof["y_true"]

        sim = simulate(y_pred, k1_correct, k8_correct, fallback="none")
        sim_fb = simulate(y_pred, k1_correct, k8_correct, fallback="k1")
        dA = a_bucket_analysis(y_pred, bkt)
        dB = b_bucket_analysis(y_pred, bkt)
        dC = c_bucket_analysis(y_pred, bkt)
        dD = d_bucket_analysis(y_pred, bkt, k1_correct, k8_correct)

        print(f"  routing: K=1:{sim['n_routed_K1']}, K=8:{sim['n_routed_K8']}, refuse:{sim['n_refused']}")
        print(f"  [refuse=no-answer] cov={sim['coverage']:.3f}, K/prob={sim['compute_per_problem']:.2f}, "
              f"answ_acc={sim['accuracy_on_answered']:.4f}, overall={sim['overall_accuracy']:.4f}")
        print(f"  [refuse→K=1 fb]    cov={sim_fb['coverage']:.3f}, K/prob={sim_fb['compute_per_problem']:.2f}, "
              f"answ_acc={sim_fb['accuracy_on_answered']:.4f}, overall={sim_fb['overall_accuracy']:.4f}")
        print(f"  A (always-right): {dA['routed_to_K1']}/{dA['n_A']} → K=1, "
              f"{dA['routed_to_K8']} → K=8, {dA['refused']} refused")
        print(f"  B (recoverable):  {dB['routed_to_K8']}/{dB['n_B']} → K=8 "
              f"(recall={dB['B_recall_to_K8']:.3f})")
        print(f"  C (never-right):  {dC['refused']}/{dC['n_C']} → refuse "
              f"(recall={dC['C_recall_to_refuse']:.3f})")
        print(f"  D (pathological): {dD['routed_to_K1']}/{dD['n_D']} → K=1 "
              f"(recall={dD['D_recall_to_K1']:.3f})")

        results[clf_kind] = dict(
            simulation_no_fallback=sim,
            simulation_k1_fallback=sim_fb,
            bucket_routing=dict(A=dA, B=dB, C=dC, D=dD),
        )

    # Compare to Exp 2 policies
    print("\n--- Comparison vs Exp 2 policies ---")
    with open(EXP2 / "phase3b_realistic_policies.json") as f:
        exp2 = json.load(f)
    compare_keys = [
        "uniform_K1", "uniform_K2", "uniform_K4", "uniform_K8",
        "quartile_prefill_balanced", "quartile_prefill_top_heavy",
        "quartile_prefill_refuse_and_spend",
        "quartile_neg_seq_len_balanced", "quartile_neg_seq_len_top_heavy",
        "quartile_neg_seq_len_refuse_and_spend",
    ]
    print(f"  {'policy':<44} {'cov':>5} {'K':>6} {'answ':>6} {'over':>6}")
    exp2_table = {}
    for k in compare_keys:
        v = exp2[k]
        print(f"  {k:<44} {v['coverage']:>5.2f} {v['compute_per_problem']:>6.2f} "
              f"{v['accuracy_on_answered']:>6.3f} {v['overall_accuracy']:>6.3f}")
        exp2_table[k] = dict(coverage=v["coverage"],
                             compute_per_problem=v["compute_per_problem"],
                             accuracy_on_answered=v["accuracy_on_answered"],
                             overall_accuracy=v["overall_accuracy"])
    for clf_kind in ["logreg", "rf"]:
        sim = results[clf_kind]["simulation_no_fallback"]
        sim_fb = results[clf_kind]["simulation_k1_fallback"]
        print(f"  multi_signal_{clf_kind}_nofb{'':<22} {sim['coverage']:>5.2f} "
              f"{sim['compute_per_problem']:>6.2f} "
              f"{sim['accuracy_on_answered']:>6.3f} {sim['overall_accuracy']:>6.3f}")
        print(f"  multi_signal_{clf_kind}_k1fb{'':<22} {sim_fb['coverage']:>5.2f} "
              f"{sim_fb['compute_per_problem']:>6.2f} "
              f"{sim_fb['accuracy_on_answered']:>6.3f} {sim_fb['overall_accuracy']:>6.3f}")

    results["exp2_policies"] = exp2_table
    # Include the exact Exp 2 oracle numbers
    with open(EXP2 / "phase4_oracle.json") as f:
        ex2_oracle = json.load(f)
    results["exp2_oracle"] = dict(
        compute_per_problem=ex2_oracle["oracle"]["compute_per_problem"],
        overall_accuracy=ex2_oracle["oracle"]["overall_accuracy"],
        K_distribution=ex2_oracle["oracle"]["K_distribution"],
    )
    print(f"  {'EXP2 oracle (upper bound)':<44} {1.0:>5.2f} "
          f"{ex2_oracle['oracle']['compute_per_problem']:>6.3f} "
          f"{'-':>6} {ex2_oracle['oracle']['overall_accuracy']:>6.3f}")

    with open(OUT_DIR / "phase3_policy.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR/'phase3_policy.json'}")


if __name__ == "__main__":
    main()
