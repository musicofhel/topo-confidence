#!/usr/bin/env python3
"""Tier-1 regen for stage 5 BBH ↔ MATH L19 DoM transfer.

Loads MATH-500 and BBH per-subset L19 last-token activations from
pathway8_layerwise/data/{math500,bbh/<subset>}/problem_*.npz (gitignored),
fits DoM on each domain, scores cross-domain, prints AUROCs.

Stdout:
    stage5_math_to_bbh_pooled, stage5_bbh_to_math, stage5_symmetric_avg,
    stage5_verdict_vs_coe, stage5_within_math, stage5_within_bbh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MATH_DIR = ROOT / "pathway8_layerwise/data/math500"
BBH_DIR = ROOT / "pathway8_layerwise/data/bbh"
CACHE = ROOT / "pathway11_h100/results/stage5_lasttoken_cache.npz"
BBH_SUBSETS = ["tracking_shuffled_objects_seven_objects",
               "logical_deduction_seven_objects", "web_of_lies"]
STEERING_LAYER = 19
COE_SYMMETRIC = 0.716


def load(d: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(d.glob("problem_*.npz"))
    if not files:
        sys.exit(f"MISSING_REGEN_INPUT: {d}")
    X, y = [], []
    for f in files:
        data = np.load(f, allow_pickle=True)
        X.append(data["states"][STEERING_LAYER, -1, :].astype(np.float32))
        y.append(bool(data["correct"]))
    return np.stack(X), np.array(y, dtype=int)


def dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()

    if CACHE.exists():
        # Fast path: small cache built once by build_stage5_cache.py.
        with np.load(CACHE) as cd:
            Xm = cd["X_math"].astype(np.float32)
            ym = cd["y_math"].astype(int)
            Xs, ys = [], []
            for sub in BBH_SUBSETS:
                xk, yk = f"X_bbh_{sub}", f"y_bbh_{sub}"
                if xk in cd.files:
                    Xs.append(cd[xk].astype(np.float32))
                    ys.append(cd[yk].astype(int))
        if not Xs:
            sys.exit(f"MISSING_REGEN_INPUT: cache {CACHE} missing BBH arrays")
    else:
        # Fallback: per-problem NPZ scan (heavy, ~31GB total).
        Xm, ym = load(MATH_DIR)
        Xs, ys = [], []
        for sub in BBH_SUBSETS:
            sd = BBH_DIR / sub
            if sd.exists() and list(sd.glob("problem_*.npz")):
                X_, y_ = load(sd)
                Xs.append(X_); ys.append(y_)
        if not Xs:
            sys.exit(f"MISSING_REGEN_INPUT: no BBH subsets under {BBH_DIR}")
    Xb = np.concatenate(Xs); yb = np.concatenate(ys)

    d_math = dom(Xm, ym)
    auroc_mb = float(roc_auc_score(yb, Xb @ d_math))
    print(f"stage5_math_to_bbh_pooled={auroc_mb:.10f}")

    d_bbh = dom(Xb, yb)
    auroc_bm = float(roc_auc_score(ym, Xm @ d_bbh))
    print(f"stage5_bbh_to_math={auroc_bm:.10f}")

    symmetric = (auroc_mb + auroc_bm) / 2
    print(f"stage5_symmetric_avg={symmetric:.10f}")
    print(f"stage5_verdict_vs_coe={symmetric - COE_SYMMETRIC:.10f}")

    auroc_mw = float(roc_auc_score(ym, Xm @ d_math))
    print(f"stage5_within_math={auroc_mw:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
