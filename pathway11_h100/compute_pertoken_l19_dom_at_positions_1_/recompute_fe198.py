"""FE198 — Per-token L19 DoM direction stability across the trailing answer window.

Settles whether F-3's prefill/final orthogonality (cos = 0.046) is a
prefill-vs-answer cleavage or an across-the-window phenomenon. We recompute
the L19 difference-of-means (DoM) direction independently at each of the last
eight answer-span token positions [-1, -2, ..., -8] from cached pathway11
per-token activations, then build the 8x8 pairwise cosine matrix of those
directions.

Decision rule (NL-ITI rho=6/tau=4 motivation — a quasi-stationary truthful
direction across the trailing window):
  - ROTATION:     all off-diagonal |cos| < 0.1  (per-token rotating direction)
  - QUASI_STATIONARY: most off-diagonal cos > 0.6 (smooth aligned subspace)
If quasi-stationary, F-3 reframes from "two orthogonal vectors" to
"prefill direction vs answer-side subspace."
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
# Per-token L19 activation cache. Candidate locations / key names are probed so
# the script survives minor naming drift in the regen output.
PERTOKEN_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_pertoken.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_pertoken_l19.npz",
    ROOT / "pathway11_h100/data/pertoken/m15b_pertoken_l19.npz",
]
MAIN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/pertoken_dom_window/results.json"

N_POS = 8  # positions [-1, -2, ..., -8]
HIDDEN = 1536


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _find_pertoken_npz() -> Path | None:
    for p in PERTOKEN_CANDIDATES:
        if p.exists():
            return p
    return None


def _extract_pertoken_array(blob) -> np.ndarray | None:
    """Return a (N, W, 1536) per-token activation array from the npz."""
    for key in ("pertoken", "tokens", "hidden", "hidden_states", "acts", "prefill_tokens"):
        if key in blob.files:
            arr = np.asarray(blob[key])
            if arr.ndim == 3 and arr.shape[-1] == HIDDEN:
                return arr.astype(np.float64)
    # Fall back: any 3D array with trailing dim 1536.
    for key in blob.files:
        arr = np.asarray(blob[key])
        if arr.ndim == 3 and arr.shape[-1] == HIDDEN:
            return arr.astype(np.float64)
    return None


def _extract_labels(blob, n: int) -> np.ndarray | None:
    if "correct" in blob.files:
        y = np.asarray(blob["correct"]).astype(bool)
        if y.shape[0] == n:
            return y
    if MAIN_CACHE.exists():
        y = np.load(MAIN_CACHE)["correct"].astype(bool)
        if y.shape[0] == n:
            return y
    return None


def main() -> int:
    pt_path = _find_pertoken_npz()
    if pt_path is None:
        print("MISSING_REGEN_INPUT", PERTOKEN_CANDIDATES[0], file=sys.stderr)
        return 2

    blob = np.load(pt_path)
    acts = _extract_pertoken_array(blob)
    if acts is None:
        print("MISSING_REGEN_INPUT", pt_path, "(no (N,W,1536) array)", file=sys.stderr)
        return 2

    n, window, _ = acts.shape
    if window < N_POS:
        print("MISSING_REGEN_INPUT", pt_path,
              f"(window {window} < {N_POS} positions)", file=sys.stderr)
        return 2

    y = _extract_labels(blob, n)
    if y is None:
        print("MISSING_REGEN_INPUT", pt_path, "(no correct labels)", file=sys.stderr)
        return 2

    if y.sum() == 0 or (~y).sum() == 0:
        print("MISSING_REGEN_INPUT", pt_path, "(degenerate labels)", file=sys.stderr)
        return 2

    # Window assumed ordered earliest -> latest, so position -1 is index window-1.
    positions = [-(k + 1) for k in range(N_POS)]          # [-1, -2, ..., -8]
    pos_index = [window - 1 - k for k in range(N_POS)]     # window-1 .. window-8

    # DoM direction (unit-normalized) at each position, plus its projection AUROC.
    dom = np.zeros((N_POS, HIDDEN), dtype=np.float64)
    dom_norms = np.zeros(N_POS, dtype=np.float64)
    pos_auroc = np.zeros(N_POS, dtype=np.float64)
    for i, idx in enumerate(pos_index):
        X = acts[:, idx, :]
        d = X[y].mean(axis=0) - X[~y].mean(axis=0)
        nrm = float(np.linalg.norm(d))
        dom_norms[i] = nrm
        if nrm > 0:
            dom[i] = d / nrm
        pos_auroc[i] = auroc(X @ d, y)

    # 8x8 pairwise cosine matrix of the (unit) DoM directions.
    cos_mat = dom @ dom.T
    cos_mat = np.clip(cos_mat, -1.0, 1.0)

    iu = np.triu_indices(N_POS, k=1)
    off = cos_mat[iu]
    abs_off = np.abs(off)

    all_rotation = bool(np.all(abs_off < 0.1))
    frac_aligned = float(np.mean(off > 0.6))
    mostly_quasi_stationary = bool(frac_aligned >= 0.5)

    if all_rotation:
        verdict = "ROTATION"
    elif mostly_quasi_stationary:
        verdict = "QUASI_STATIONARY"
    else:
        verdict = "MIXED"

    out = {
        "experiment": "P11-FE198",
        "pertoken_cache": str(pt_path),
        "n_problems": int(n),
        "window_len": int(window),
        "positions": positions,
        "pos_index": [int(x) for x in pos_index],
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "dom_norms": [float(x) for x in dom_norms],
        "pos_auroc": [float(x) for x in pos_auroc],
        "cosine_matrix": [[float(c) for c in row] for row in cos_mat],
        "offdiag_cos": [float(c) for c in off],
        "offdiag_cos_mean": float(off.mean()),
        "offdiag_cos_median": float(np.median(off)),
        "offdiag_abs_cos_max": float(abs_off.max()),
        "offdiag_abs_cos_min": float(abs_off.min()),
        "frac_offdiag_gt_0p6": frac_aligned,
        "frac_offdiag_abs_lt_0p1": float(np.mean(abs_off < 0.1)),
        "all_offdiag_abs_lt_0p1": all_rotation,
        "mostly_offdiag_gt_0p6": mostly_quasi_stationary,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())