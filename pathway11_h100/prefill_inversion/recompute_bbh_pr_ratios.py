#!/usr/bin/env python3
"""Tier-1 regen for §9 BBH per-subset prefill PR ratios.

Loads `pathway8_layerwise/data/bbh/{subset}/problem_*.npz` (gitignored) and
recomputes pr(correct)/pr(incorrect) for prefill at L19 per BBH subset.

Stdout:
    pr_bbh_<subset>_prefill_ratio
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BBH_DIR = ROOT / "pathway8_layerwise/data/bbh"
CACHE_DIR = ROOT / "pathway11_h100/prefill_inversion/cache"
SUBSETS = ["tracking_shuffled_objects_seven_objects",
           "logical_deduction_seven_objects", "web_of_lies"]
STEERING_LAYER = 19


def participation_ratio(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()
    for sub in SUBSETS:
        # Fast path: small per-subset cache (~3MB) built once by build_bbh_cache.py.
        cache_fp = CACHE_DIR / f"bbh_{sub}.npz"
        if cache_fp.exists():
            with np.load(cache_fp) as cd:
                H = cd["prefill"].astype(np.float32)
                y = cd["correct"].astype(bool)
        else:
            # Fallback: per-problem NPZ scan (heavy, ~17GB total).
            d = BBH_DIR / sub
            files = sorted(d.glob("problem_*.npz"))
            if not files:
                sys.exit(f"MISSING_REGEN_INPUT: neither {cache_fp} nor {d}")
            prefill = []
            correct = []
            for fp in files:
                data = np.load(fp, allow_pickle=True)
                s = data["states"]
                prefill.append(s[STEERING_LAYER, 0, :].astype(np.float32))
                correct.append(bool(data["correct"]))
            H = np.stack(prefill)
            y = np.array(correct, dtype=bool)
        if y.sum() < 5 or (~y).sum() < 5:
            print(f"pr_bbh_{sub}_prefill_ratio=NaN")
            continue
        pr_c = participation_ratio(H[y])
        pr_i = participation_ratio(H[~y])
        print(f"pr_bbh_{sub}_prefill_ratio={pr_c / max(pr_i, 1e-30):.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
