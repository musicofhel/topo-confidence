#!/usr/bin/env python3
"""Shared cache loaders for FE19 (CPU). Single source of truth for buckets, the
uniform-K frontier, and the prefill-DoM C_infer baseline — all aligned to MATH-500
index 0..499 (verified problem_indices == arange(500)).

KEY INVARIANT (verified at impl): the published K=8 majority (0.55) and buckets
(A=207,B=68,C=189,D=36) come from phase1_majority_vote.npz's `k8_majority_correct`,
which is the correctness of the *plurality answer cluster* — NOT a majority over the
raw per-sample `correct` booleans (those differ when wrong answers are scattered).
Always use phase1's precomputed arrays; never recompute majority from bools.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
PGC = ROOT / "pathway11_h100/prefill_gated_compute"
HERE = Path(__file__).resolve().parent
N = 500


def load_k1_greedy():
    """(idx, correct) for the K=1 greedy answers (243/500 = 48.6%)."""
    rows = json.loads((HERE / "results" / "k1_greedy.json").read_text())["rows"]
    rows = sorted(rows, key=lambda r: r["idx"])
    idx = np.array([r["idx"] for r in rows], dtype=int)
    correct = np.array([r["correct"] for r in rows], dtype=bool)
    return idx, correct


def load_phase1():
    d = np.load(PGC / "phase1_majority_vote.npz", allow_pickle=True)
    assert np.array_equal(d["problem_indices"].astype(int), np.arange(N))
    return {
        "k8_majority_correct": d["k8_majority_correct"].astype(bool),
        "mean_maj_K1": d["mean_maj_at_K1"].astype(float),
        "mean_maj_K2": d["mean_maj_at_K2"].astype(float),
        "mean_maj_K4": d["mean_maj_at_K4"].astype(float),
        "mean_maj_K8": d["mean_maj_at_K8"].astype(float),
    }


def buckets(greedy_correct, k8_maj_correct):
    """Return dict of boolean masks A/B/C/D and assert published sizes."""
    g, k = greedy_correct, k8_maj_correct
    masks = {
        "A_always_right": g & k,
        "B_recoverable": (~g) & k,
        "C_never_right": (~g) & (~k),
        "D_pathological": g & (~k),
    }
    sizes = {n: int(m.sum()) for n, m in masks.items()}
    expected = {"A_always_right": 207, "B_recoverable": 68, "C_never_right": 189, "D_pathological": 36}
    assert sizes == expected, f"bucket mismatch: {sizes} != {expected} (cache loader bug)"
    return masks


def load_prefill_dom():
    """OOF prefill-L19-DoM score aligned to 0..499 (the C_infer baseline, AUROC 0.7731).
    Higher score = more correct-like (per phase2 sign). Returns (score, correct_k1)."""
    d = np.load(PGC / "phase2_prefill_dom.npz", allow_pickle=True)
    pidx = d["problem_indices"].astype(int)
    score = np.full(N, np.nan, dtype=float)
    score[pidx] = d["prefill_score"].astype(float)
    return score


def uniform_frontier(p1, greedy_correct):
    """Published uniform-K points: (avg_K, accuracy). K=1 greedy; K>=2 T=0.7 majority."""
    return {
        1: (1.0, float(greedy_correct.mean())),
        2: (2.0, float(p1["mean_maj_K2"].mean())),
        4: (4.0, float(p1["mean_maj_K4"].mean())),
        8: (8.0, float(p1["k8_majority_correct"].mean())),
    }
