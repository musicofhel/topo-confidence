#!/usr/bin/env python3
"""Tier-1 regen for §1 1.5B MATH-500 baseline accuracies + §1f oracle compute
+ phase4 bucket counts (A/B/C/D).

Loads `phase1_majority_vote.npz` (per-problem K=1..8 majority outcomes) and
`phase2_prefill_dom.npz` (K=1 greedy correctness + DoM scores). Both gitignored.

Stdout: key=value lines.
    baseline_K1_greedy, baseline_K2_maj, baseline_K4_maj, baseline_K8_maj,
    bucket_A, bucket_B, bucket_C, bucket_D,
    oracle_K, oracle_acc

Skips with MISSING_REGEN_INPUT if either cache absent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PH1 = ROOT / "pathway11_h100/prefill_gated_compute/phase1_majority_vote.npz"
PH2 = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()

    for fp in (PH1, PH2):
        if not fp.exists():
            sys.exit(f"MISSING_REGEN_INPUT: {fp}")

    ph1 = np.load(PH1, allow_pickle=True)
    ph2 = np.load(PH2, allow_pickle=True)

    # Align on common problem indices.
    common = np.intersect1d(ph1["problem_indices"], ph2["problem_indices"])
    im1 = {int(p): i for i, p in enumerate(ph1["problem_indices"])}
    im2 = {int(p): i for i, p in enumerate(ph2["problem_indices"])}
    sel1 = np.array([im1[int(p)] for p in common])
    sel2 = np.array([im2[int(p)] for p in common])

    greedy_K1 = ph2["correct_k1"][sel2].astype(bool)
    k8_maj = ph1["k8_majority_correct"][sel1].astype(bool)
    k2 = ph1["mean_maj_at_K2"][sel1].astype(float)
    k4 = ph1["mean_maj_at_K4"][sel1].astype(float)
    k8 = ph1["mean_maj_at_K8"][sel1].astype(float)
    n = len(common)

    # Baseline accs — match RESEARCH_SUMMARY §1 wording.
    print(f"baseline_K1_greedy={float(greedy_K1.mean()):.10f}")
    print(f"baseline_K2_maj={float(k2.mean()):.10f}")
    print(f"baseline_K4_maj={float(k4.mean()):.10f}")
    print(f"baseline_K8_maj={float(k8_maj.mean()):.10f}")

    # Buckets per phase4_oracle.py:
    A = greedy_K1 & k8_maj
    B = ~greedy_K1 & k8_maj
    C = ~greedy_K1 & ~k8_maj
    D = greedy_K1 & ~k8_maj
    assert A.sum() + B.sum() + C.sum() + D.sum() == n
    print(f"bucket_A={int(A.sum())}")
    print(f"bucket_B={int(B.sum())}")
    print(f"bucket_C={int(C.sum())}")
    print(f"bucket_D={int(D.sum())}")

    # Oracle: smallest K with maj_acc ≥ 0.5; -1 ⇒ refuse (compute=0, acc=0).
    K_OPTS = [1, 2, 4, 8]
    accs = {1: greedy_K1.astype(float), 2: k2, 4: k4, 8: k8_maj.astype(float)}
    oracle_K = np.full(n, -1, dtype=int)
    for K in K_OPTS:
        hit = (accs[K] >= 0.5) & (oracle_K == -1)
        oracle_K[hit] = K
    answerable = oracle_K > 0
    compute = np.where(answerable, oracle_K, 0).astype(float)
    acc = np.zeros(n)
    for K in K_OPTS:
        m = oracle_K == K
        acc[m] = accs[K][m]
    print(f"oracle_K={float(compute.mean()):.10f}")
    print(f"oracle_acc={float(acc.mean()):.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
