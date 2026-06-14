"""P11-FE1 — Breathing analysis (temporal participation-ratio curve) on BBH subsets.

F-1 ("breathing": the residual-stream effective dimensionality expands then
contracts over a generation, and the correct/incorrect PR trajectories differ)
was measured on MATH-500 only. This script replays the same temporal-PR
analysis on BBH per-token activations (Stage 4a re-extract) to test whether
breathing is a reasoning-general phenomenon or math-specific.

For every BBH task NPZ we:
  1. Build a sliding-window participation-ratio curve PR(t) per problem from
     the per-token hidden states (window SVD → eigenvalues → PR = (Σλ)²/Σλ²).
  2. Resample each curve to a common normalized-position grid and average it
     separately over correct and incorrect problems.
  3. Report the correct/incorrect PR ratio (the F-1 readout) and the AUROC of
     a per-problem mean-PR scalar against correctness.

web_of_lies (≈54% accuracy, near-balanced) is the headline subset: a held PR
ratio there would be the cleanest non-math corroboration of F-1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE_DIR = ROOT / "pathway11_h100/bbh_breathing/cache"
OUT_JSON = ROOT / "pathway11_h100/bbh_breathing/results.json"

WINDOW = 16          # tokens per PR window
GRID = 50            # resampled curve length (normalized position)
MIN_WIN_TOKENS = 4   # skip windows shorter than this
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def participation_ratio(window: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² of the centered window covariance, via window SVD."""
    if window.shape[0] < MIN_WIN_TOKENS:
        return float("nan")
    Xc = window - window.mean(axis=0, keepdims=True)
    # singular values of the centered (W, d) window; λ_i = s_i².
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = s.astype(np.float64) ** 2
    denom = float((lam ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return float((lam.sum() ** 2) / denom)


def pr_curve(tokens: np.ndarray) -> np.ndarray:
    """Sliding-window PR curve, resampled to GRID points over [0, 1]."""
    T = tokens.shape[0]
    if T < MIN_WIN_TOKENS:
        return np.full(GRID, np.nan)
    half = WINDOW // 2
    centers, prs = [], []
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        pr = participation_ratio(tokens[lo:hi])
        if np.isfinite(pr):
            centers.append(t)
            prs.append(pr)
    if len(prs) < 2:
        return np.full(GRID, np.nan)
    centers = np.asarray(centers, dtype=np.float64)
    prs = np.asarray(prs, dtype=np.float64)
    pos = (centers - centers.min()) / max(centers.max() - centers.min(), 1e-9)
    grid = np.linspace(0.0, 1.0, GRID)
    return np.interp(grid, pos, prs)


def load_task(path: Path):
    """Return (list_of_token_arrays, correct_bool) or (None, None) on bad schema."""
    blob = np.load(path, allow_pickle=True)
    keys = set(blob.files)
    hkey = next((k for k in ("hidden", "tokens", "pertoken", "activations") if k in keys), None)
    if hkey is None or "correct" not in keys:
        return None, None
    hidden = blob[hkey]
    correct = blob["correct"].astype(bool)

    lengths = None
    for lk in ("lengths", "seq_len", "token_len"):
        if lk in keys:
            lengths = blob[lk].astype(int)
            break
    mask = blob["mask"].astype(bool) if "mask" in keys else None

    token_lists = []
    if hidden.dtype == object:  # ragged: array of (T_i, d) arrays
        for i in range(len(hidden)):
            token_lists.append(np.asarray(hidden[i], dtype=np.float64))
    else:  # padded (n, T, d)
        n = hidden.shape[0]
        for i in range(n):
            arr = np.asarray(hidden[i], dtype=np.float64)
            if mask is not None:
                arr = arr[mask[i]]
            elif lengths is not None:
                arr = arr[: lengths[i]]
            token_lists.append(arr)

    if len(token_lists) != len(correct):
        return None, None
    return token_lists, correct


def analyze_task(path: Path) -> dict | None:
    token_lists, correct = load_task(path)
    if token_lists is None:
        return None

    curves, mean_pr = [], []
    keep = []
    for arr, _ in zip(token_lists, correct):
        c = pr_curve(arr)
        if np.isfinite(c).all():
            curves.append(c)
            mean_pr.append(float(np.nanmean(c)))
            keep.append(True)
        else:
            keep.append(False)
    keep = np.asarray(keep, dtype=bool)
    if keep.sum() < 4:
        return None

    curves = np.asarray(curves, dtype=np.float64)
    mean_pr = np.asarray(mean_pr, dtype=np.float64)
    y = correct[keep]
    if y.sum() == 0 or (~y).sum() == 0:
        return None

    curve_correct = curves[y].mean(axis=0)
    curve_incorrect = curves[~y].mean(axis=0)
    pr_ratio = float(np.mean(curve_correct) / max(np.mean(curve_incorrect), 1e-9))
    # "Breathing" amplitude: peak-to-trough span of the mean curve.
    breath_correct = float(curve_correct.max() - curve_correct.min())
    breath_incorrect = float(curve_incorrect.max() - curve_incorrect.min())

    return {
        "n_problems": int(keep.sum()),
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "accuracy": float(y.mean()),
        "pr_ratio_correct_over_incorrect": pr_ratio,
        "mean_pr_correct": float(np.mean(curve_correct)),
        "mean_pr_incorrect": float(np.mean(curve_incorrect)),
        "breathing_amplitude_correct": breath_correct,
        "breathing_amplitude_incorrect": breath_incorrect,
        "auroc_mean_pr": auroc(mean_pr, y),
        "curve_correct": [float(x) for x in curve_correct],
        "curve_incorrect": [float(x) for x in curve_incorrect],
    }


def main() -> int:
    if not CACHE_DIR.exists():
        print("MISSING_REGEN_INPUT", CACHE_DIR, file=sys.stderr)
        return 2
    npz_files = sorted(CACHE_DIR.glob("*.npz"))
    if not npz_files:
        print("MISSING_REGEN_INPUT", CACHE_DIR / "*.npz", file=sys.stderr)
        return 2

    per_task = {}
    for path in npz_files:
        task = path.stem
        for prefix in ("bbh_", "stage4a_"):
            if task.startswith(prefix):
                task = task[len(prefix):]
        for suffix in ("_pertoken", "_breathing", "_activations"):
            if task.endswith(suffix):
                task = task[: -len(suffix)]
        res = analyze_task(path)
        if res is not None:
            per_task[task] = res

    if not per_task:
        print("MISSING_REGEN_INPUT", "no parseable per-token BBH NPZ", file=sys.stderr)
        return 2

    ratios = [v["pr_ratio_correct_over_incorrect"] for v in per_task.values()]
    aurocs = [v["auroc_mean_pr"] for v in per_task.values() if np.isfinite(v["auroc_mean_pr"])]

    out = {
        "experiment": "P11-FE1",
        "description": "BBH temporal-PR breathing analysis vs MATH-500 F-1",
        "window": WINDOW,
        "grid": GRID,
        "n_tasks": len(per_task),
        "tasks": per_task,
        "summary": {
            "mean_pr_ratio_across_tasks": float(np.mean(ratios)),
            "median_pr_ratio_across_tasks": float(np.median(ratios)),
            "mean_auroc_mean_pr_across_tasks": float(np.mean(aurocs)) if aurocs else float("nan"),
            "web_of_lies_pr_ratio": per_task.get("web_of_lies", {}).get(
                "pr_ratio_correct_over_incorrect", float("nan")
            ),
            "web_of_lies_auroc_mean_pr": per_task.get("web_of_lies", {}).get(
                "auroc_mean_pr", float("nan")
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())