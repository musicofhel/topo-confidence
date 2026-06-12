"""Cascade — keep-on-small / escalate-to-large routing.

Ports the decision logic of
pathway11_h100/generalization_edge/exp_1b4_overlap.py (SPEC v6, EXP-82):
keep the small model's answer when the gate score >= tau, escalate to the
large model otherwise.

Cost model (FU2): keep = 1, escalate = 5 (pre-generation decision, no wasted
draft).  cost_rel = (5 - 4*pct_kept)/5 = 1 - 0.8*pct_kept, relative to
all-large.  Honest comparator is the RANDOM-MIX HULL (the FE19 lesson): the
line from (cost_rel 0.2, acc_small) to (cost_rel 1.0, acc_large).

Pinned v6 anchor (data/pinned_meta.json, from results/1b4_overlap.json): the
hybrid cascade beats the random-mix hull by +3.38pp (paired bootstrap SE
0.96pp, significant) at keep=0.30.
"""
from __future__ import annotations

import numpy as np

KEEP_COST = 1.0
ESCALATE_COST = 5.0

KEEP = "keep"
ESCALATE = "escalate"


def cost_rel(pct_kept: float) -> float:
    """Cost relative to all-large under the keep=1 / escalate=5 model."""
    return 1.0 - 0.8 * float(pct_kept)


class Cascade:
    """Route per-problem between a small (cheap) and a large (expensive) model."""

    def __init__(self, gate=None, tau: float = 0.5):
        self.gate = gate
        self.tau = float(tau)

    # --------------------------------------------------------------- routing
    def decide(self, scores) -> np.ndarray:
        """Per-problem decision: 'keep' if score >= tau else 'escalate'."""
        s = np.asarray(scores, dtype=np.float64)
        return np.where(s >= self.tau, KEEP, ESCALATE)

    def route(self, lengths, logprobs) -> np.ndarray:
        """Score with the attached gate, then decide."""
        if self.gate is None:
            raise ValueError("Cascade has no gate attached")
        return self.decide(self.gate.score(lengths, logprobs))

    # -------------------------------------------------------------- frontier
    @staticmethod
    def frontier(scores, y_15b, y_7b, n_tau: int = 51) -> list[dict]:
        """Accuracy/cost curve over tau (port of exp_1b4_overlap.cascade_sweep).

        Keep the highest-`scores` problems on the small model, escalate the
        rest.  Returns rows with pct_kept, accuracy, cost_rel, n_kept, tau
        (the gate score at the keep/escalate boundary), plus hull_acc and
        gap_pp vs the random-mix hull at matched cost.
        """
        scores = np.asarray(scores, dtype=np.float64)
        y15 = np.asarray(y_15b).astype(bool)
        y7 = np.asarray(y_7b).astype(bool)
        n = len(y15)
        if not (len(scores) == n == len(y7)):
            raise ValueError("scores, y_15b, y_7b must be aligned")
        acc15, acc7 = float(y15.mean()), float(y7.mean())
        order = np.argsort(-scores)  # highest score (most confident) first
        rows = []
        for k in np.linspace(0, n, n_tau).astype(int):
            keep = order[:k]
            esc = order[k:]
            correct = y15[keep].sum() + y7[esc].sum()
            acc = correct / n
            pct_kept = k / n
            c = cost_rel(pct_kept)
            hull = Cascade.hull_acc_at_cost(c, acc15, acc7)
            rows.append({
                "pct_kept": float(pct_kept),
                "accuracy": float(acc),
                "cost_rel": float(c),
                "n_kept": int(k),
                "tau": float(scores[order[k - 1]]) if k > 0 else float("inf"),
                "hull_acc": float(hull),
                "gap_pp": float(100 * (acc - hull)),
            })
        return rows

    @staticmethod
    def hull_acc_at_cost(c: float, acc15: float, acc7: float) -> float:
        """Random-mix line: pct_kept = (1-cost_rel)/0.8; acc linear in pct_kept."""
        pct_kept = (1.0 - c) / 0.8
        pct_kept = float(np.clip(pct_kept, 0.0, 1.0))
        return pct_kept * acc15 + (1.0 - pct_kept) * acc7

    @staticmethod
    def pick_operating_point(frontier_rows: list[dict], target: float) -> dict:
        """Cheapest frontier row whose accuracy >= target.

        If no row reaches `target`, returns the max-accuracy row (which is the
        most expensive achievable answer).  Use row['tau'] as the deployment
        threshold: keep when gate score >= tau.
        """
        feasible = [r for r in frontier_rows if r["accuracy"] >= target]
        if feasible:
            return min(feasible, key=lambda r: (r["cost_rel"], -r["accuracy"]))
        return max(frontier_rows, key=lambda r: r["accuracy"])

    @staticmethod
    def best_gap(frontier_rows: list[dict]) -> dict:
        """Interior frontier row with the largest gap over the random-mix hull
        (mirrors exp_1b4_overlap.adjudicate's best operating point)."""
        interior = [r for r in frontier_rows if 0.0 < r["pct_kept"] < 1.0]
        if not interior:
            raise ValueError("frontier has no interior operating points")
        return max(interior, key=lambda r: r["gap_pp"])
