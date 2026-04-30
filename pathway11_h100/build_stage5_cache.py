#!/usr/bin/env python3
"""One-time cache builder for stage5 BBH↔MATH DoM transfer regen.

Extracts L19 *last-token* activations + correctness from MATH-500 and BBH
per-subset NPZs into a single small cache file (~5MB). Subsequent runs of
recompute_stage5_bbh_transfer.py pick up the cache automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH_DIR = ROOT / "pathway8_layerwise/data/math500"
BBH_DIR = ROOT / "pathway8_layerwise/data/bbh"
CACHE = ROOT / "pathway11_h100/results/stage5_lasttoken_cache.npz"
SUBSETS = ["tracking_shuffled_objects_seven_objects",
           "logical_deduction_seven_objects", "web_of_lies"]
STEERING_LAYER = 19


def extract(d: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(d.glob("problem_*.npz"))
    if not files:
        return np.zeros((0, 1536), np.float16), np.zeros(0, bool)
    X = np.zeros((len(files), 1536), dtype=np.float16)
    y = np.zeros(len(files), dtype=bool)
    for i, fp in enumerate(files):
        with np.load(fp, allow_pickle=True) as data:
            X[i] = data["states"][STEERING_LAYER, -1, :].astype(np.float16)
            y[i] = bool(data["correct"])
    return X, y


def main() -> int:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    X_math, y_math = extract(MATH_DIR)
    print(f"math500: n={len(y_math)}, correct={int(y_math.sum())}", file=sys.stderr)
    bbh_arrays: dict[str, np.ndarray] = {}
    for sub in SUBSETS:
        X, y = extract(BBH_DIR / sub)
        if len(y) == 0:
            print(f"[skip] {sub}: no NPZs", file=sys.stderr)
            continue
        bbh_arrays[f"X_bbh_{sub}"] = X
        bbh_arrays[f"y_bbh_{sub}"] = y
        print(f"bbh/{sub}: n={len(y)}, correct={int(y.sum())}", file=sys.stderr)
    np.savez_compressed(CACHE, X_math=X_math, y_math=y_math, **bbh_arrays)
    print(f"[ok] {CACHE}  ({CACHE.stat().st_size//1024} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
