#!/usr/bin/env python3
"""Tier-1 regen for §3 gibberish-control PR curves.

Reads `data/{math500,random,stream}/problem_*.npz` (gitignored) and
recomputes participation_ratio at each target position for each condition,
matching the logic of compute_pr.py.

Stdout: key=value lines like `gib_math_pos50=30.554`, `gib_random_pos1=11.32`,
`gib_math_final=22.19`, `gib_stream_pos10=15.66`.

Skips with MISSING_REGEN_INPUT if data dirs absent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"

CONDITIONS = {
    "math": "math500",
    "random": "random",
    "stream": "stream",
}

# Subset of position labels referenced by validate_claims.py
TARGET_POSITIONS = {"1", "10", "25", "50", "100", "200", "final"}


def participation_ratio(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    s2 = s.astype(np.float64) ** 2
    den = (s2 ** 2).sum()
    return float((s2.sum() ** 2) / den) if den > 0 else float("nan")


def compute(data_dir: Path) -> dict[str, float]:
    files = sorted(data_dir.glob("problem_*.npz"))
    if not files:
        sys.exit(f"MISSING_REGEN_INPUT: {data_dir} has no problem_*.npz")
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
    out: dict[str, float] = {}
    for lbl, stack in by_label.items():
        if len(stack) >= 5:
            out[lbl] = participation_ratio(np.stack(stack, axis=0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()

    for cond_short, cond_dir in CONDITIONS.items():
        d = DATA_ROOT / cond_dir
        if not d.exists():
            sys.exit(f"MISSING_REGEN_INPUT: {d}")
        prs = compute(d)
        for lbl, pr in prs.items():
            print(f"gib_{cond_short}_pos{lbl}={pr:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
