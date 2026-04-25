#!/usr/bin/env python3
"""Experiment 2 / Phase 4: oracle bucket analysis.

Bucket each problem by (greedy K=1 correct, K=8 majority correct):
   A. always-right      : K=1 right (K=8 doesn't matter — call it "K=1 suffices")
   B. recoverable       : K=1 wrong, K=8 majority right
   C. never-right       : K=1 wrong, K=8 majority wrong
   D. pathological      : K=1 right, K=8 majority wrong (K=8 actively hurts — diagnostic)

Then:
  * report bucket sizes
  * AUROC of prefill score separating {A} vs {B} (does high confidence mean K=1 suffices
    vs K=8 helps?)
  * AUROC of prefill score separating {B} vs {C} (does low confidence distinguish
    recoverable from never-right? — answer to "can gating route K=8 to worthwhile cases?")
  * AUROC of prefill score separating {A∪B} vs {C} (separating "will eventually be right"
    from hopeless, at any K)
  * Pareto-optimal oracle: per-problem K that maximizes acc while minimizing compute.
    For each problem find the smallest K with majority-correct ≥ 0.5; that's the "just
    enough" K. Report the distribution.
  * Report how well prefill score predicts the oracle K.

Also:
  * Same three AUROC tests for: final_token score, seq_len (neg, so higher = more confident),
    mean_logprob.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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
        problem_indices=common.astype(int),
        greedy_K1=ph2["correct_k1"][sel2].astype(bool),
        k8_maj=np.asarray(ph1["k8_majority_correct"])[sel1].astype(bool),
        k2=ph1["mean_maj_at_K2"][sel1].astype(float),
        k4=ph1["mean_maj_at_K4"][sel1].astype(float),
        k8=ph1["mean_maj_at_K8"][sel1].astype(float),
        prefill_score=ph2["prefill_score"][sel2].astype(float),
        final_token_score=ph2["final_token_score"][sel2].astype(float),
        seq_len=ph2["seq_len"][sel2].astype(int),
        mean_logprob=ph2["mean_logprob"][sel2].astype(float),
    )


def main():
    print("=" * 70)
    print("Experiment 2 / Phase 4: Oracle bucket analysis")
    print("=" * 70)
    d = load()
    n = len(d["greedy_K1"])

    A = d["greedy_K1"] & d["k8_maj"]      # both right
    A_only_k1 = d["greedy_K1"] & ~d["k8_maj"]  # K=1 right, K=8 wrong (pathological D)
    B = ~d["greedy_K1"] & d["k8_maj"]     # recoverable
    C = ~d["greedy_K1"] & ~d["k8_maj"]    # never-right
    D = A_only_k1

    print(f"  n = {n}")
    print(f"  A always-right (K=1 ✓, K=8 ✓): {A.sum()}  ({A.mean():.3f})")
    print(f"  B recoverable  (K=1 ✗, K=8 ✓): {B.sum()}  ({B.mean():.3f})")
    print(f"  C never-right  (K=1 ✗, K=8 ✗): {C.sum()}  ({C.mean():.3f})")
    print(f"  D pathological (K=1 ✓, K=8 ✗): {D.sum()}  ({D.mean():.3f})")
    assert A.sum() + B.sum() + C.sum() + D.sum() == n

    print(f"\n  K=1 right:   {(A|D).sum()} ({(A|D).mean():.3f})")
    print(f"  K=8 maj right: {(A|B).sum()} ({(A|B).mean():.3f})")
    print(f"  K=8 recovers K=1 failures: B / (B+C) = {B.sum()}/{(B|C).sum()} = "
          f"{B.sum() / max((B|C).sum(), 1):.3f}")

    scores = {
        "prefill": d["prefill_score"],
        "final_token": d["final_token_score"],
        "neg_seq_len": -d["seq_len"].astype(float),
        "mean_logprob": d["mean_logprob"],
    }

    print("\n=== AUROC: does score separate buckets? ===")
    print(f"{'score':<18} {'A vs B':>12} {'B vs C':>12} {'(A∪B) vs C':>14} {'K=1 right':>12} {'K=8 right':>12}")
    auroc_results = {}
    for sname, score in scores.items():
        # A vs B: among K=1-ambiguous problems, does high confidence = K=1 suffices?
        mask_AB = A | B
        if mask_AB.sum() > 2 and A[mask_AB].sum() > 0 and B[mask_AB].sum() > 0:
            auc_AB = roc_auc_score(A[mask_AB].astype(int), score[mask_AB])
        else:
            auc_AB = np.nan
        # B vs C: among K=1-wrong problems, does low confidence = still wrong at K=8?
        mask_BC = B | C
        if mask_BC.sum() > 2 and B[mask_BC].sum() > 0 and C[mask_BC].sum() > 0:
            auc_BC = roc_auc_score(B[mask_BC].astype(int), score[mask_BC])
        else:
            auc_BC = np.nan
        # (A∪B) vs C: eventually right vs hopeless
        auc_ABC = roc_auc_score((A | B).astype(int), score)
        # K=1 right: usual correctness AUROC (already known for prefill from Phase 2)
        auc_k1 = roc_auc_score((A | D).astype(int), score)
        # K=8 right
        auc_k8 = roc_auc_score((A | B).astype(int), score)
        print(f"  {sname:<18} {auc_AB:>12.4f} {auc_BC:>12.4f} {auc_ABC:>14.4f} "
              f"{auc_k1:>12.4f} {auc_k8:>12.4f}")
        auroc_results[sname] = dict(A_vs_B=float(auc_AB), B_vs_C=float(auc_BC),
                                     AB_vs_C=float(auc_ABC), K1_correct=float(auc_k1),
                                     K8_maj_correct=float(auc_k8))

    # Oracle K assignment: smallest K s.t. majority-vote correct ≥ 0.5
    K_options = [1, 2, 4, 8]
    accs_per_K = {1: d["greedy_K1"].astype(float), 2: d["k2"], 4: d["k4"], 8: d["k8_maj"].astype(float)}
    oracle_K = np.zeros(n, dtype=int)
    for i in range(n):
        for K in K_options:
            if accs_per_K[K][i] >= 0.5:
                oracle_K[i] = K
                break
        else:
            oracle_K[i] = -1  # never reaches 0.5
    print("\n=== Oracle-optimal K distribution (smallest K with acc ≥ 0.5) ===")
    vals, counts = np.unique(oracle_K, return_counts=True)
    for v, c in zip(vals, counts):
        label = f"K={v}" if v > 0 else "never"
        print(f"  {label:>8}: {c:4d} ({c/n:.3f})")

    # Oracle's total compute per problem (refuse → pay K=1 to determine unsolvability, no, actually
    # oracle spends 0 on "never" problems — you just say "refuse" if you could identify them).
    oracle_compute = oracle_K.copy().astype(float)
    oracle_compute[oracle_K == -1] = 0  # refused
    oracle_acc = np.zeros(n)
    for i in range(n):
        if oracle_K[i] == -1:
            oracle_acc[i] = 0
        else:
            oracle_acc[i] = accs_per_K[oracle_K[i]][i]
    print(f"\n  Oracle compute-per-problem: {oracle_compute.mean():.2f}")
    print(f"  Oracle overall accuracy:    {oracle_acc.mean():.4f}")

    # Can the prefill score predict oracle K?
    print("\n=== Does prefill score predict oracle K? ===")
    # Spearman rank correlation between prefill score and -oracle_K (higher score → lower K needed)
    try:
        from scipy.stats import spearmanr
        answerable = oracle_K > 0
        rho, p = spearmanr(d["prefill_score"][answerable], -oracle_K[answerable])
        print(f"  Spearman(prefill_score, -oracle_K | answerable) = {rho:.3f} (p={p:.2e})")
        rho_f, p_f = spearmanr(d["final_token_score"][answerable], -oracle_K[answerable])
        print(f"  Spearman(final_token_score, -oracle_K | answerable) = {rho_f:.3f} (p={p_f:.2e})")
        rho_s, p_s = spearmanr(-d["seq_len"][answerable].astype(float), -oracle_K[answerable])
        print(f"  Spearman(-seq_len, -oracle_K | answerable) = {rho_s:.3f} (p={p_s:.2e})")
        rho_results = dict(prefill=(float(rho), float(p)), final_token=(float(rho_f), float(p_f)),
                           neg_seq_len=(float(rho_s), float(p_s)))
    except Exception as e:
        print(f"  scipy spearmanr failed: {e}")
        rho_results = {}

    # Also: AUROC for score separating (A) from (A ∪ B)? i.e., "K=1 suffices" among those who will
    # eventually be right. This is the policy-relevant question: if a problem is going to be right,
    # can we tell if we need K=1 or K=8?
    # The earlier A-vs-B AUROC covers this.

    summary = dict(
        bucket_sizes=dict(
            A_always_right=int(A.sum()),
            B_recoverable=int(B.sum()),
            C_never_right=int(C.sum()),
            D_pathological=int(D.sum()),
        ),
        k1_accuracy=float((A | D).mean()),
        k8_majority_accuracy=float((A | B).mean()),
        k8_recovers_K1_failures=float(B.sum() / max((B | C).sum(), 1)),
        auroc=auroc_results,
        oracle=dict(
            K_distribution={int(v): int(c) for v, c in zip(vals, counts)},
            compute_per_problem=float(oracle_compute.mean()),
            overall_accuracy=float(oracle_acc.mean()),
        ),
        spearman_with_oracle_K=rho_results,
    )
    with open(OUT_DIR / "phase4_oracle.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR/'phase4_oracle.json'}")


if __name__ == "__main__":
    main()
