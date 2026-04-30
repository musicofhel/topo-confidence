#!/usr/bin/env python3
"""One-time BBH L19 prefill+correctness cache builder.

Reads pathway8_layerwise/data/bbh/<subset>/problem_*.npz (heavy 27MB each),
extracts only the L19 prefill (position 0) + correctness, and writes one small
~3MB cache per subset to pathway11_h100/prefill_inversion/cache/bbh_<subset>.npz.

Run once after stage 4a; recompute_bbh_pr_ratios.py picks up the cache if
present (~ms), else falls back to per-problem NPZ scan (~3 min).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BBH_DIR = ROOT / "pathway8_layerwise/data/bbh"
CACHE_DIR = ROOT / "pathway11_h100/prefill_inversion/cache"
SUBSETS = ["tracking_shuffled_objects_seven_objects",
           "logical_deduction_seven_objects", "web_of_lies"]
STEERING_LAYER = 19


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in SUBSETS:
        d = BBH_DIR / sub
        files = sorted(d.glob("problem_*.npz"))
        if not files:
            print(f"[skip] {sub}: no problem_*.npz under {d}", file=sys.stderr)
            continue
        prefill = np.zeros((len(files), 1536), dtype=np.float16)
        correct = np.zeros(len(files), dtype=bool)
        for i, fp in enumerate(files):
            with np.load(fp, allow_pickle=True) as data:
                prefill[i] = data["states"][STEERING_LAYER, 0, :].astype(np.float16)
                correct[i] = bool(data["correct"])
        out = CACHE_DIR / f"bbh_{sub}.npz"
        np.savez_compressed(out, prefill=prefill, correct=correct)
        print(f"[ok] {sub}: {len(files)} problems → {out}  ({out.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
