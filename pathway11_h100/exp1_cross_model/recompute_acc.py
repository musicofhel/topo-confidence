#!/usr/bin/env python3
"""Tier-1 regen for §exp1 cross-model accuracies + 2/3-depth layer indices.

Loads `data/{phi3mini,llama32_1b}/problem_*.npz` (gitignored), counts correct.

Stdout:
    phi3_acc, phi3_n_correct, phi3_2thirds_layer
    llama_acc, llama_n_correct, llama_2thirds_layer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SUBDIRS = {"phi3": "phi3mini", "llama": "llama32_1b"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()
    for short, sub in SUBDIRS.items():
        d = DATA / sub
        files = sorted(d.glob("problem_*.npz"))
        if not files:
            sys.exit(f"MISSING_REGEN_INPUT: {d}")
        n_correct = 0
        twothirds = None
        for fp in files:
            data = np.load(fp, allow_pickle=True)
            n_correct += int(bool(data["correct"]))
            if twothirds is None:
                twothirds = int(data["twothirds_layer_idx"])
        n = len(files)
        print(f"{short}_acc={n_correct / n:.10f}")
        print(f"{short}_n_correct={n_correct}")
        print(f"{short}_2thirds_layer={twothirds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
