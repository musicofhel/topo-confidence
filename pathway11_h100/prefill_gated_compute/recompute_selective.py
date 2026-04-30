#!/usr/bin/env python3
"""Tier-1 regen for §7 selective prediction headlines.

Loads phase1_majority_vote.npz + phase2_prefill_dom.npz and re-implements the
two policies cited in narrative docs:
    refuse@0.5 (prefill): threshold gate at the 50th percentile of prefill_score;
                          answered top half gets K=1 (greedy) if K=1 ✓ else
                          K=4 majority. Reports acc_on_answered.
    refuse@0.5 (random):  same shape but with shuffled scores (avg over 100 draws).
    middle-heavy (K=4.5): quartile gate Q3,Q2 → K=8, Q4,Q1 → K=1, no refusals.

Stdout:
    refuse_prefill_acc, refuse_random_acc, middle_heavy_acc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PH1 = ROOT / "pathway11_h100/prefill_gated_compute/phase1_majority_vote.npz"
PH2 = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
SEED = 9999


def quartiles(score: np.ndarray) -> np.ndarray:
    r = score.argsort().argsort()
    n = len(score)
    q = np.zeros(n, dtype=int)
    q[r >= (3 * n) // 4] = 3
    q[(r >= n // 2) & (r < (3 * n) // 4)] = 2
    q[(r >= n // 4) & (r < n // 2)] = 1
    q[r < n // 4] = 0
    return q


def acc_per_problem(gate: np.ndarray, accs: dict[int, np.ndarray]) -> np.ndarray:
    a = np.zeros(len(gate))
    for K, vec in accs.items():
        a[gate == K] = vec[gate == K]
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()
    for fp in (PH1, PH2):
        if not fp.exists():
            sys.exit(f"MISSING_REGEN_INPUT: {fp}")

    ph1 = np.load(PH1, allow_pickle=True)
    ph2 = np.load(PH2, allow_pickle=True)
    common = np.intersect1d(ph1["problem_indices"], ph2["problem_indices"])
    im1 = {int(p): i for i, p in enumerate(ph1["problem_indices"])}
    im2 = {int(p): i for i, p in enumerate(ph2["problem_indices"])}
    sel1 = np.array([im1[int(p)] for p in common])
    sel2 = np.array([im2[int(p)] for p in common])
    accs = {
        1: ph2["correct_k1"][sel2].astype(float),
        2: ph1["mean_maj_at_K2"][sel1].astype(float),
        4: ph1["mean_maj_at_K4"][sel1].astype(float),
        8: ph1["k8_majority_correct"][sel1].astype(float),
    }
    score = ph2["prefill_score"][sel2].astype(float)
    n = len(common)

    # refuse@0.5 (prefill): top quartile → K=1, second-top → K=4, bottom half refused.
    q = quartiles(score)
    gate = np.zeros(n, dtype=int)
    gate[q == 3] = 1
    gate[q == 2] = 4
    answered = gate > 0
    acc_p = acc_per_problem(gate, accs)
    print(f"refuse_prefill_acc={float(acc_p[answered].mean()):.10f}")

    # Random shuffle baseline (mean over 100 draws).
    rng = np.random.default_rng(SEED)
    rand_accs = []
    for _ in range(100):
        s = score.copy()
        rng.shuffle(s)
        qr = quartiles(s)
        g = np.zeros(n, dtype=int)
        g[qr == 3] = 1
        g[qr == 2] = 4
        ans = g > 0
        a = acc_per_problem(g, accs)
        rand_accs.append(float(a[ans].mean()))
    print(f"refuse_random_acc={float(np.mean(rand_accs)):.10f}")

    # Middle-heavy (avg K=4.5): Q3,Q2 → K=8, Q4,Q1 → K=1, all answered.
    g = np.zeros(n, dtype=int)
    g[q == 3] = 1; g[q == 0] = 1
    g[q == 2] = 8; g[q == 1] = 8
    a = acc_per_problem(g, accs)
    print(f"middle_heavy_acc={float(a.mean()):.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
