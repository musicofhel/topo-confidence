"""P11-FE524 — Cosine-stability matrix of L19 DoM across generation token positions.

Refits the difference-of-means (DoM) correctness direction at a set of token
positions {0, 5, 9, 19, 29, 39, 49, 59, 69, 200, 500} from cached Stage 2
per-position hidden-state NPZs, then computes the full pairwise cosine matrix
between the refit directions. Direct test of whether F-3 orthogonality
(cos(prefill_DoM, final_DoM) = 0.046) is a prefill-vs-final artifact of
measuring only the two trajectory extremes, or a property of the entire CoT
trajectory. Trigger: Chan et al. 2507.12428 Fig 3b (a single linear probe
transfers across observed/foresight positions on safety refusal) — if
correctness geometry behaved the same way the cosine matrix would be near-1
everywhere; F-3 predicts a rapid decorrelation toward the final token.

Per-position NPZs are expected to carry an L19 hidden-state matrix (500, 1536)
and a (500,) bool correctness vector. If any required position cache is
missing, the script prints MISSING_REGEN_INPUT and returns 2.
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
# Stage 2 per-position trajectory caches. Position 500 (final token) and
# position 0 (prefill) also resolve to the canonical Stage 2 caches below.
TRAJ_DIR = ROOT / "pathway11_h100/dom_trajectory/cache"
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/dom_cosine_stability/results.json"

POSITIONS = [0, 5, 9, 19, 29, 39, 49, 59, 69, 200, 500]
HIDDEN_KEYS = ("hidden", "prefill", "states", "h", "resid", "act")
CORRECT_KEYS = ("correct", "labels", "y")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _pick(blob, keys):
    for k in keys:
        if k in blob.files:
            return blob[k]
    return None


def _candidate_paths(pos: int) -> list[Path]:
    cands = [
        TRAJ_DIR / f"pos_{pos}.npz",
        TRAJ_DIR / f"pos_{pos:03d}.npz",
        TRAJ_DIR / f"m15b_pos{pos}.npz",
        TRAJ_DIR / f"token_{pos}.npz",
    ]
    # The prefill cache doubles as the position-0 (prompt-final) Stage 2 NPZ.
    if pos == 0:
        cands.append(PREFILL_CACHE)
    return cands


def load_position(pos: int):
    """Return (X (500,1536) float64, y (500,) bool) or None if not found."""
    for path in _candidate_paths(pos):
        if not path.exists():
            continue
        blob = np.load(path)
        X = _pick(blob, HIDDEN_KEYS)
        y = _pick(blob, CORRECT_KEYS)
        if X is None or y is None:
            continue
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(bool)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            continue
        return X, y
    return None


def dom_direction(X: np.ndarray, y: np.ndarray):
    """Difference-of-means direction (unit-normalized) and its in-sample AUROC."""
    if y.sum() == 0 or (~y).sum() == 0:
        return None, float("nan")
    d = X[y].mean(axis=0) - X[~y].mean(axis=0)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return None, float("nan")
    unit = d / nrm
    return unit, auroc(X @ unit, y)


def main() -> int:
    directions = {}
    aurocs = {}
    found_positions = []
    missing = []

    for pos in POSITIONS:
        loaded = load_position(pos)
        if loaded is None:
            missing.append(pos)
            continue
        X, y = loaded
        unit, au = dom_direction(X, y)
        if unit is None:
            missing.append(pos)
            continue
        directions[pos] = unit
        aurocs[pos] = au
        found_positions.append(pos)

    if missing:
        print("MISSING_REGEN_INPUT", "positions=" + ",".join(str(p) for p in missing),
              file=sys.stderr)
        return 2

    # Pairwise cosine matrix over the (unit) DoM directions.
    P = found_positions
    n = len(P)
    cos_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            cos_mat[i, j] = float(directions[P[i]] @ directions[P[j]])

    # F-3 diagnostic: cos between the earliest (prefill-like) and final token.
    cos_prefill_final = float(directions[P[0]] @ directions[P[-1]])
    off_diag = cos_mat[~np.eye(n, dtype=bool)]

    out = {
        "experiment": "P11-FE524",
        "positions": P,
        "auroc_by_position": {str(p): float(aurocs[p]) for p in P},
        "cosine_matrix": cos_mat.tolist(),
        "cos_prefill_final": cos_prefill_final,
        "cos_offdiag_min": float(off_diag.min()),
        "cos_offdiag_max": float(off_diag.max()),
        "cos_offdiag_mean": float(off_diag.mean()),
        "f3_reference_cos": 0.046,
        "f3_artifact_verdict": (
            "TRAJECTORY_WIDE_DECORRELATION" if off_diag.mean() < 0.3
            else "STABLE_TRAJECTORY" if off_diag.min() > 0.7
            else "MIXED"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())