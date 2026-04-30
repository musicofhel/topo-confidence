#!/usr/bin/env python3
"""Tier-1 regen for the no-CoT control PR at L19 2/3-depth.

Same shape as gibberish_control/recompute_pr.py — loads nocot data dir,
recomputes PR at target positions.

Stdout: nocot_pos1, nocot_posfinal (etc).
Skips with MISSING_REGEN_INPUT if data/nocot/problem_*.npz absent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "nocot"
TARGET_POSITIONS = {"1", "final"}


def participation_ratio(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    s2 = s.astype(np.float64) ** 2
    den = (s2 ** 2).sum()
    return float((s2.sum() ** 2) / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()
    files = sorted(DATA_DIR.glob("problem_*.npz"))
    if not files:
        sys.exit(f"MISSING_REGEN_INPUT: {DATA_DIR}")
    by_label: dict[str, list[np.ndarray]] = {lbl: [] for lbl in TARGET_POSITIONS}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        positions = d["positions_actual"].tolist()
        vecs = d["twothirds_positions"]
        n_gen = int(d["n_gen_tokens"])
        for p, v in zip(positions, vecs):
            if int(p) == n_gen:
                by_label["final"].append(v.astype(np.float32))
            if str(int(p)) in TARGET_POSITIONS:
                by_label[str(int(p))].append(v.astype(np.float32))
    for lbl, stack in by_label.items():
        if len(stack) >= 5:
            print(f"nocot_pos{lbl}={participation_ratio(np.stack(stack, 0)):.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
