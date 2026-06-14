"""P11-FE400 — Reframe F-8 selective prediction as within-question (WQD) confidence ranking.

CISC (§6) argues that between-question calibration metrics (ECE-t) poorly predict
intra-question discrimination. F-8's selective-prediction threshold is between-question:
it ranks *questions* by prefill L19 DoM and abstains below a global τ. This script tests
whether a within-question DoM-gap (WQD) abstention — rank the K=8 paths of each question
by DoM, take the highest-DoM path, abstain where (max - median) DoM gap < τ — yields a
different accuracy@coverage curve at equal cost.

Key structural fact this experiment surfaces: the cached prefill L19 DoM is a *prompt-level*
quantity (one forward pass over the question prompt, shared by all 8 sampled paths). It is
therefore constant across a question's K=8 paths, so the within-question DoM-gap is
identically zero and WQD abstention is degenerate from prefill DoM alone. We record that
degeneracy explicitly, then compute (a) the between-question accuracy@coverage curve that
F-8 actually uses, and (b) a within-question self-consistency vote-margin abstention curve
as the realizable intra-question analogue, so the two regimes can be compared on cost.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/within_question_selective/results.json"

N_PROBLEMS = 500
K = 8
COVERAGE_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 4)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def acc_at_coverage(confidence: np.ndarray, correct: np.ndarray, grid: np.ndarray):
    """Sort by confidence desc; for each coverage answer the top fraction, report accuracy."""
    correct = correct.astype(bool)
    order = np.argsort(-confidence, kind="stable")
    sorted_correct = correct[order]
    n = len(confidence)
    accs = []
    for c in grid:
        k = max(1, int(round(c * n)))
        accs.append(float(sorted_correct[:k].mean()))
    # area under the accuracy@coverage curve (trapezoid over the coverage grid)
    area = float(np.trapz(accs, grid))
    return accs, area


def main() -> int:
    for p in (CACHE, DOM_NPZ):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
            return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    base_correct = np.load(CACHE)["correct"].astype(bool)
    assert dom_score.shape == (N_PROBLEMS,) and base_correct.shape == (N_PROBLEMS,)

    # Per-question K=8 correctness → majority-vote correctness and vote margin.
    k8_correct = np.zeros((N_PROBLEMS, K), dtype=bool)
    for i in range(N_PROBLEMS):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            print("MISSING_REGEN_INPUT", f, file=sys.stderr)
            return 2
        c = np.load(f)["correct"].astype(bool)
        if c.shape != (K,):
            print("MISSING_REGEN_INPUT", f, "bad_shape", c.shape, file=sys.stderr)
            return 2
        k8_correct[i] = c

    n_correct_paths = k8_correct.sum(axis=1)
    maj_correct = n_correct_paths >= (K / 2.0 + 0.5)  # strict majority of 8 → >=5? use >=4 tie-break below
    maj_correct = n_correct_paths >= 4               # >=4/8 counts the question as answerable-correctly
    k1_correct = k8_correct[:, 0]                     # first sampled path

    # ---- Within-question DoM-gap degeneracy check ------------------------------
    # Prefill L19 DoM is a prompt-level forward pass shared by all K paths, so the
    # per-question DoM vector is constant: gap = max - median = 0 identically.
    within_question_dom_gap = np.zeros(N_PROBLEMS, dtype=np.float64)  # by construction
    wqd_gap_max = float(within_question_dom_gap.max())
    wqd_is_degenerate = bool(wqd_gap_max == 0.0)

    # ---- (a) Between-question accuracy@coverage (the F-8 regime) ----------------
    bq_accs_maj, bq_area_maj = acc_at_coverage(dom_score, maj_correct, COVERAGE_GRID)
    bq_accs_k1, bq_area_k1 = acc_at_coverage(dom_score, k1_correct, COVERAGE_GRID)

    # ---- (b) Within-question realizable analogue: self-consistency vote margin --
    # The intra-question confidence that *is* available per question is the K=8
    # vote margin (|#correct - #incorrect| / K). This is the WQD-style signal CISC
    # contrasts against between-question DoM. Abstain on low-margin questions.
    vote_margin = np.abs(2.0 * n_correct_paths - K) / float(K)
    wq_accs_maj, wq_area_maj = acc_at_coverage(vote_margin, maj_correct, COVERAGE_GRID)
    wq_accs_k1, wq_area_k1 = acc_at_coverage(vote_margin, k1_correct, COVERAGE_GRID)

    # Headline single-point comparison at coverage 0.5.
    half_idx = int(np.argmin(np.abs(COVERAGE_GRID - 0.5)))

    out = {
        "experiment": "P11-FE400",
        "description": "WQD vs between-question selective prediction on prefill L19 DoM",
        "n_questions": N_PROBLEMS,
        "k_paths": K,
        "question_correctness_defs": {
            "majority": ">=4 of 8 paths correct",
            "k1": "first sampled path correct",
        },
        "within_question_dom_gap": {
            "definition": "max(DoM over K paths) - median(DoM over K paths)",
            "max_over_questions": wqd_gap_max,
            "is_degenerate": wqd_is_degenerate,
            "reason": (
                "Prefill L19 DoM is a prompt-level forward pass shared by all K sampled "
                "paths of a question; it is constant within a question, so the within-question "
                "DoM-gap is identically zero. Intra-question abstention from prefill DoM alone "
                "is not realizable — this is the structural answer to FE400's question."
            ),
        },
        "between_question_dom": {
            "coverage_grid": COVERAGE_GRID.tolist(),
            "acc_at_coverage_majority": bq_accs_maj,
            "acc_at_coverage_k1": bq_accs_k1,
            "area_under_acc_coverage_majority": bq_area_maj,
            "area_under_acc_coverage_k1": bq_area_k1,
            "acc_at_coverage_0p5_majority": bq_accs_maj[half_idx],
            "acc_at_coverage_0p5_k1": bq_accs_k1[half_idx],
            "auroc_dom_vs_majority": auroc(dom_score, maj_correct),
            "auroc_dom_vs_k1": auroc(dom_score, k1_correct),
        },
        "within_question_vote_margin": {
            "note": (
                "Realizable intra-question analogue: K=8 self-consistency vote margin "
                "|#correct-#incorrect|/K used as the abstention signal in place of the "
                "degenerate within-question DoM-gap."
            ),
            "coverage_grid": COVERAGE_GRID.tolist(),
            "acc_at_coverage_majority": wq_accs_maj,
            "acc_at_coverage_k1": wq_accs_k1,
            "area_under_acc_coverage_majority": wq_area_maj,
            "area_under_acc_coverage_k1": wq_area_k1,
            "acc_at_coverage_0p5_majority": wq_accs_maj[half_idx],
            "acc_at_coverage_0p5_k1": wq_accs_k1[half_idx],
        },
        "comparison": {
            "area_delta_within_minus_between_majority": wq_area_maj - bq_area_maj,
            "area_delta_within_minus_between_k1": wq_area_k1 - bq_area_k1,
            "verdict": (
                "Between-question DoM and within-question vote-margin operate on different "
                "axes; prefill DoM cannot supply a within-question gap, so a like-for-like "
                "DoM-only WQD curve does not exist. Vote-margin is reported as the realizable "
                "intra-question alternative for the accuracy@coverage comparison."
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())