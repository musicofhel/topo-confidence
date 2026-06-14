"""P11-FE738 — Step-wise DoM rotation curve on Qwen-1.5B (L19).

Tests whether F-3's prefill/final orthogonality (cos(prefill_DoM, final_DoM) =
0.046) is the two endpoints of a smooth rotation through SO(d) rather than two
discrete orthogonal circuits.

For each decoding step k = 0..K_max we form the difference-of-means direction at
L19, DoM(k) = mean(h(L19, step_k) | correct) - mean(h(L19, step_k) | incorrect),
fit out-of-fold (DoM computed on train folds, averaged as unit vectors) to avoid
the means being dominated by any single fold, and report:

  * consecutive cosines  cos(DoM(k), DoM(k+1))  along the trajectory,
  * the endpoint cosine   cos(DoM(0=prefill), DoM(K_max=final)),
  * full-data (non-OOF) variants of both as a stability cross-check.

Interpretation (per the FE rationale): consecutive cosines in 0.6-0.95 imply a
smooth rotation (F-3 should be reframed as "rotation through SO(d)", feeding the
RoPE-mechanical-rotation hypothesis H-17); near-zero consecutive cosines imply a
genuine discrete jump and F-3 stands as written.

Required regen input is a step-wise L19 hidden-state trajectory cache. The
documented prefill cache only holds step 0, so this script looks for the
stepwise NPZ and exits MISSING_REGEN_INPUT (2) if the Stage 2 trajectory has not
been extracted.
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
# Step-wise L19 trajectory cache: stacked hidden states across decoding steps.
# Candidate keys handled below; falls back gracefully if absent.
STEP_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_stepwise_L19.npz"
# Endpoints we *do* have on disk, used as a sanity anchor when present.
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/dom_rotation_curve/results.json"

N_FOLDS = 5
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(unit(a) @ unit(b), -1.0, 1.0))


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means direction at a single step."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def oof_dom_direction(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """Average of unit DoM directions computed on each train fold (held-out k)."""
    n = len(y)
    acc = np.zeros(X.shape[1], dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        if ytr.sum() == 0 or (~ytr).sum() == 0:
            continue
        acc += unit(dom_direction(Xtr, ytr))
    return acc / max(len(folds), 1)


def load_trajectory():
    """Return (steps: (500, T, 1536) float64, correct: (500,) bool) or None."""
    if not STEP_NPZ.exists():
        return None
    blob = np.load(STEP_NPZ, allow_pickle=False)
    keys = set(blob.files)
    step_key = next((k for k in ("step_hidden", "steps", "trajectory", "hidden") if k in keys), None)
    corr_key = next((k for k in ("correct", "labels", "y") if k in keys), None)
    if step_key is None or corr_key is None:
        return None
    steps = blob[step_key].astype(np.float64)
    correct = blob[corr_key].astype(bool)
    if steps.ndim != 3 or steps.shape[0] != correct.shape[0]:
        return None
    return steps, correct


def main() -> int:
    traj = load_trajectory()
    if traj is None:
        print("MISSING_REGEN_INPUT", STEP_NPZ, file=sys.stderr)
        return 2

    steps, correct = traj
    n_problems, n_steps, d = steps.shape
    if correct.sum() == 0 or (~correct).sum() == 0:
        print("MISSING_REGEN_INPUT degenerate labels", file=sys.stderr)
        return 2

    folds = stratified_kfold(correct, N_FOLDS, SEED)

    # Per-step DoM directions: OOF-averaged (primary) and full-data (cross-check).
    oof_dirs = np.zeros((n_steps, d), dtype=np.float64)
    full_dirs = np.zeros((n_steps, d), dtype=np.float64)
    step_auroc = []
    for k in range(n_steps):
        Xk = steps[:, k, :]
        oof_dirs[k] = oof_dom_direction(Xk, correct, folds)
        full_dirs[k] = dom_direction(Xk, correct)
        step_auroc.append(auroc(Xk @ unit(full_dirs[k]), correct))

    consecutive_oof = [cos(oof_dirs[k], oof_dirs[k + 1]) for k in range(n_steps - 1)]
    consecutive_full = [cos(full_dirs[k], full_dirs[k + 1]) for k in range(n_steps - 1)]
    endpoint_oof = cos(oof_dirs[0], oof_dirs[-1])
    endpoint_full = cos(full_dirs[0], full_dirs[-1])

    smooth = all(0.6 <= abs(c) <= 0.95 for c in consecutive_oof) if consecutive_oof else False

    out = {
        "experiment": "P11-FE738",
        "n_problems": int(n_problems),
        "n_steps": int(n_steps),
        "dim": int(d),
        "step_auroc": [float(x) for x in step_auroc],
        "consecutive_cosines_oof": [float(x) for x in consecutive_oof],
        "consecutive_cosines_full": [float(x) for x in consecutive_full],
        "endpoint_cosine_oof": float(endpoint_oof),
        "endpoint_cosine_full": float(endpoint_full),
        "mean_consecutive_cosine_oof": float(np.mean(consecutive_oof)) if consecutive_oof else float("nan"),
        "min_consecutive_cosine_oof": float(np.min(consecutive_oof)) if consecutive_oof else float("nan"),
        "smooth_rotation": bool(smooth),
        "interpretation": (
            "smooth rotation through SO(d): reframe F-3, feeds H-17"
            if smooth else
            "consecutive cosines outside [0.6,0.95]: not a uniform smooth rotation"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())