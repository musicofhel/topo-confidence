#!/usr/bin/env python3
"""Stage 5: cross-benchmark L19 DoM transfer (MATH ↔ BBH). CPU-only, ~10 min.

Depends on stage 2 (pathway8_layerwise/extract_math500.py) AND stage 4a
(pathway8_layerwise/extract_bbh.py) both complete.

Pathway 9 found CoE transfers symmetrically MATH↔BBH at ~0.72, while
layerwise-PH inverts. This stage asks: does RAW L19 activation DoM — no
trajectory features, no summaries — transfer just as well? If yes, CoE's
machinery isn't adding anything. If no, CoE's summaries are doing real work.

Directions tested:
  - MATH→BBH: fit DoM on all MATH-500 at L19 last-token, apply to each BBH
              subset + pooled BBH, compute AUROC vs BBH correctness.
  - BBH→MATH: fit DoM on pooled BBH at L19 last-token, apply to MATH,
              compute AUROC vs MATH correctness.
  - Symmetric avg: mean of both directions' AUROC.

Each AUROC gets a bootstrap 95% CI (n_boot=2000).

Pathway 9 baseline numbers for comparison (results/exp5_cross_domain.json):
  CoE  MATH→BBH=0.720, BBH→MATH=0.712, symmetric=0.716

Output: pathway11_h100/results/stage5_bbh_dom_transfer.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway11_h100.config import (
    BBH_DATA_DIR,
    BBH_SUBSETS,
    MATH500_DATA_DIR,
    RESULTS_DIR,
    STEERING_LAYER,
    mark_done,
)

SEED = 9999
N_BOOT = 2000
COE_BASELINE = {"math_to_bbh": 0.720, "bbh_to_math": 0.712, "symmetric": 0.716}


def load_l19_last_token(data_dir: Path, subset_subdir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load L19 last-token activations + correctness labels from per-problem npz.

    data_dir: e.g. pathway8_layerwise/data/math500/ or .../bbh/{subset}/
    Returns (X, y) where X: (n, 1536), y: (n,) int.
    """
    base = data_dir / subset_subdir if subset_subdir else data_dir
    files = sorted(base.glob("problem_*.npz"))
    if not files:
        raise FileNotFoundError(f"No problem_*.npz files in {base}")
    X, y = [], []
    for f in files:
        data = np.load(f, allow_pickle=True)
        s = data["states"]  # (29, n_tokens, 1536)
        X.append(s[STEERING_LAYER, -1, :].astype(np.float32))
        y.append(bool(data["correct"]))
    return np.stack(X), np.array(y, dtype=int)


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def bootstrap_auroc(scores: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, size=n)
        yb = y[bi]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        vals.append(roc_auc_score(yb, scores[bi]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))


def score_with_ci(X: np.ndarray, y: np.ndarray, direction: np.ndarray) -> dict:
    scores = X @ direction
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": None, "reason": "class imbalance"}
    auroc = float(roc_auc_score(y, scores))
    lo, med, hi = bootstrap_auroc(scores, y)
    return {"auroc": auroc, "ci95": [lo, hi], "median_boot": med, "n": int(len(y)), "n_correct": int(y.sum())}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 5: L19 DoM cross-benchmark transfer (MATH ↔ BBH)")
    print("=" * 70)
    t0 = time.time()

    # ---- Load MATH-500 -----------------------------------------------------
    print("\nLoading MATH-500 L19 last-token states...")
    X_math, y_math = load_l19_last_token(MATH500_DATA_DIR)
    print(f"  MATH-500: n={len(y_math)}, correct={int(y_math.sum())} ({y_math.mean():.1%})")

    # ---- Load BBH per subset ----------------------------------------------
    print("\nLoading BBH per-subset L19 last-token states...")
    bbh_per_subset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for subset in BBH_SUBSETS:
        subset_dir = BBH_DATA_DIR / subset
        if not subset_dir.exists() or not list(subset_dir.glob("problem_*.npz")):
            print(f"  [SKIP] {subset}: no data")
            continue
        X_s, y_s = load_l19_last_token(BBH_DATA_DIR, subset_subdir=subset)
        bbh_per_subset[subset] = (X_s, y_s)
        print(f"  {subset}: n={len(y_s)}, correct={int(y_s.sum())} ({y_s.mean():.1%})")

    if not bbh_per_subset:
        print("[ABORT] No BBH data — run stage 4a first.")
        return

    X_bbh_pooled = np.concatenate([X for X, _ in bbh_per_subset.values()])
    y_bbh_pooled = np.concatenate([y for _, y in bbh_per_subset.values()])
    print(f"  BBH pooled: n={len(y_bbh_pooled)}, correct={int(y_bbh_pooled.sum())} ({y_bbh_pooled.mean():.1%})")

    # ---- MATH → BBH --------------------------------------------------------
    print("\n--- MATH → BBH ---")
    d_math = dom_direction(X_math, y_math)
    math_to_bbh: dict[str, dict] = {}
    for subset, (X_s, y_s) in bbh_per_subset.items():
        r = score_with_ci(X_s, y_s, d_math)
        math_to_bbh[subset] = r
        if r["auroc"] is not None:
            print(f"  {subset:>42}: AUROC={r['auroc']:.3f} 95%CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
        else:
            print(f"  {subset:>42}: n/a ({r['reason']})")
    pooled_mb = score_with_ci(X_bbh_pooled, y_bbh_pooled, d_math)
    math_to_bbh["_pooled"] = pooled_mb
    if pooled_mb["auroc"] is not None:
        print(f"  {'POOLED':>42}: AUROC={pooled_mb['auroc']:.3f} 95%CI=[{pooled_mb['ci95'][0]:.3f}, {pooled_mb['ci95'][1]:.3f}]")

    # ---- BBH → MATH --------------------------------------------------------
    print("\n--- BBH → MATH ---")
    d_bbh = dom_direction(X_bbh_pooled, y_bbh_pooled)
    bbh_to_math = score_with_ci(X_math, y_math, d_bbh)
    if bbh_to_math["auroc"] is not None:
        print(f"  MATH-500: AUROC={bbh_to_math['auroc']:.3f} 95%CI=[{bbh_to_math['ci95'][0]:.3f}, {bbh_to_math['ci95'][1]:.3f}]")

    # ---- Within-benchmark sanity (DoM fit & eval on same data) -------------
    # Not an honest AUROC — upper bound since overfits — but useful as a ceiling
    print("\n--- Within-benchmark (train/eval same set — overfit ceiling) ---")
    d_math_self = dom_direction(X_math, y_math)
    math_self = score_with_ci(X_math, y_math, d_math_self)
    print(f"  MATH (overfit): AUROC={math_self['auroc']:.3f}")

    d_bbh_self = dom_direction(X_bbh_pooled, y_bbh_pooled)
    bbh_self = score_with_ci(X_bbh_pooled, y_bbh_pooled, d_bbh_self)
    print(f"  BBH pooled (overfit): AUROC={bbh_self['auroc']:.3f}")

    # ---- Symmetric avg + CoE comparison -----------------------------------
    symmetric = None
    if pooled_mb["auroc"] is not None and bbh_to_math["auroc"] is not None:
        symmetric = (pooled_mb["auroc"] + bbh_to_math["auroc"]) / 2
        print(f"\nSymmetric avg (pooled MATH↔BBH): {symmetric:.3f}")
        print(f"Pathway 9 CoE baseline (symmetric):  {COE_BASELINE['symmetric']:.3f}")
        delta = symmetric - COE_BASELINE["symmetric"]
        verdict = "matches CoE" if abs(delta) < 0.02 else (
            "beats CoE" if delta > 0 else "underperforms CoE"
        )
        print(f"Verdict: raw L19 DoM {verdict} ({delta:+.3f})")

    # ---- Save --------------------------------------------------------------
    out = {
        "meta": {
            "seed": SEED,
            "n_boot": N_BOOT,
            "steering_layer": STEERING_LAYER,
            "coe_baseline_pathway9": COE_BASELINE,
        },
        "math_to_bbh": math_to_bbh,
        "bbh_to_math": bbh_to_math,
        "within_benchmark_overfit": {"math": math_self, "bbh_pooled": bbh_self},
        "symmetric_avg": symmetric,
        "verdict_vs_coe": (
            None if symmetric is None
            else symmetric - COE_BASELINE["symmetric"]
        ),
    }
    out_path = RESULTS_DIR / "stage5_bbh_dom_transfer.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {out_path}")
    print(f"[done in {time.time()-t0:.1f}s]")

    mark_done("stage5", RESULTS_DIR)
    print("[DONE] stage5 marker written")


if __name__ == "__main__":
    main()
