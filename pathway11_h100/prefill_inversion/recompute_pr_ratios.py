#!/usr/bin/env python3
"""Tier-1 regen for §9 prefill PR inversion.

Loads `cache/m{7b,15b}_prefill.npz` (committed-via-script, gitignored), recomputes
participation_ratio(H[correct]) / PR(H[~correct]) for prefill + final-token at
L19 on 7B and 1.5B, plus the balance-controlled 7B prefill ratio.

Skips with MISSING_REGEN_INPUT if the cache files are absent (fresh clone — run
phase1_bootstrap.py first to build them).

Stdout protocol — one `key=value` line per metric:
    pr_7b_prefill_correct, pr_7b_prefill_incorrect, ratio_7b_prefill,
    pr_7b_final_correct,   pr_7b_final_incorrect,   ratio_7b_final,
    pr_15b_prefill_correct, pr_15b_prefill_incorrect, ratio_15b_prefill,
    pr_15b_final_correct,   pr_15b_final_incorrect,   ratio_15b_final,
    n_7b_correct, n_7b_incorrect, n_15b_correct, n_15b_incorrect,
    ratio_7b_balance_controlled
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache"
SEED = 9999


def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum λ)² / sum λ², via SVD of centered X for numerical stability."""
    if len(X) < 2:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-30))


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = CACHE / f"{name}.npz"
    if not fp.exists():
        sys.exit(f"MISSING_REGEN_INPUT: {fp} — run phase1_bootstrap.py to build the cache")
    with np.load(fp) as d:
        return (d["prefill"].astype(np.float32),
                d["final_tok"].astype(np.float32),
                d["correct"].astype(bool))


def emit(model: str, prefill: np.ndarray, final_tok: np.ndarray, y: np.ndarray) -> None:
    print(f"n_{model}_correct={int(y.sum())}")
    print(f"n_{model}_incorrect={int((~y).sum())}")
    for tag, H in (("prefill", prefill), ("final", final_tok)):
        pr_c = participation_ratio(H[y])
        pr_i = participation_ratio(H[~y])
        print(f"pr_{model}_{tag}_correct={pr_c:.10f}")
        print(f"pr_{model}_{tag}_incorrect={pr_i:.10f}")
        print(f"ratio_{model}_{tag}={pr_c / max(pr_i, 1e-30):.10f}")


def balance_controlled(prefill: np.ndarray, y: np.ndarray, n_resample: int = 100) -> float:
    """Subsample majority class to minority size, average ratio over n_resample draws."""
    rng = np.random.default_rng(SEED)
    minority_is_correct = y.sum() < (~y).sum()
    n_min = int(min(y.sum(), (~y).sum()))
    if minority_is_correct:
        idx_min = np.where(y)[0]
        idx_maj = np.where(~y)[0]
    else:
        idx_min = np.where(~y)[0]
        idx_maj = np.where(y)[0]
    pr_min = participation_ratio(prefill[idx_min])
    ratios = []
    for _ in range(n_resample):
        sub = rng.choice(idx_maj, size=n_min, replace=False)
        pr_maj = participation_ratio(prefill[sub])
        # ratio_correct_over_incorrect orientation
        if minority_is_correct:
            ratios.append(pr_min / max(pr_maj, 1e-30))
        else:
            ratios.append(pr_maj / max(pr_min, 1e-30))
    return float(np.mean(ratios))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    args = ap.parse_args()

    pf7, ft7, y7 = load("m7b_prefill")
    pf15, ft15, y15 = load("m15b_prefill")
    emit("7b", pf7, ft7, y7)
    emit("15b", pf15, ft15, y15)
    print(f"ratio_7b_balance_controlled={balance_controlled(pf7, y7):.10f}")

    # Optional bootstrap CI (deterministic via SEED=9999, matches
    # phase1_bootstrap.py). Skipped by default — 1000 SVDs on 500×3584 takes
    # ~7 min on this machine, too slow for routine gate runs. Set
    # VALIDATE_BOOTSTRAP_CI=1 to opt in.
    if os.environ.get("VALIDATE_BOOTSTRAP_CI") == "1":
        rng = np.random.default_rng(SEED)
        n = len(y7)
        ratios_b = np.empty(1000)
        for b in range(1000):
            idx = rng.integers(0, n, size=n)
            yb = y7[idx]
            if yb.sum() < 5 or (~yb).sum() < 5:
                ratios_b[b] = np.nan
                continue
            Hb = pf7[idx]
            ratios_b[b] = participation_ratio(Hb[yb]) / max(participation_ratio(Hb[~yb]), 1e-30)
        print(f"ci95_7b_prefill_lo={float(np.nanpercentile(ratios_b, 2.5)):.10f}")
        print(f"ci95_7b_prefill_hi={float(np.nanpercentile(ratios_b, 97.5)):.10f}")

    # Three-way split — both right / only-7B / only-15B / both wrong.
    if len(y7) == len(y15):
        both = int((y7 & y15).sum())
        only7 = int((y7 & ~y15).sum())
        only15 = int((~y7 & y15).sum())
        neither = int((~y7 & ~y15).sum())
        print(f"three_way_both_right_n={both}")
        print(f"three_way_only_7b_n={only7}")
        print(f"three_way_only_15b_n={only15}")
        print(f"three_way_both_wrong_n={neither}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
