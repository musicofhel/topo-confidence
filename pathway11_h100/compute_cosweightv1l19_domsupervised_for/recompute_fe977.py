"""P11-FE977 — Is the supervised DoM direction a weight-structural artifact?

Computes |cos(weight_v1_L19, DoM_supervised)| for each L19 projection type
(Gate, Down, Q, K, O), where weight_v1 is the leading singular vector of the
projection weight matrix restricted to the 1536-d residual-stream (hidden)
basis, and DoM_supervised is the supervised difference-of-means direction
mean(prefill[correct]) - mean(prefill[~correct]).

If |cos(weight_v1, DoM)| > 0.85 for any projection, the DoM direction is
plausibly the leading weight direction of that projection — a training-mechanics
artifact rather than a learned correctness signal. Context: cos(DoM, PC1)=0.9216
already suggests PC1 ≈ a dominant weight/covariance direction.

Weight matrices for L19 are read from a CPU-only NPZ export (no torch/
transformers at runtime). For a weight W of shape (a, b): if b == HIDDEN we take
the leading right singular vector (lives in hidden space); if a == HIDDEN we take
the leading left singular vector. Missing inputs -> MISSING_REGEN_INPUT, exit 2.
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
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# CPU-only export of the L19 projection weight matrices. Keys map to the
# transformer projection names; values are the raw 2-D weight arrays.
WEIGHTS_NPZ = ROOT / "pathway11_h100/weight_structure/l19_weights.npz"
OUT_JSON = ROOT / "pathway11_h100/weight_structure/results.json"

HIDDEN = 1536
THRESHOLD = 0.85

# Canonical projection key aliases -> friendly label. The export may use either
# HF-style (`gate_proj`) or short (`gate`) keys; we accept both.
PROJ_ALIASES = {
    "Gate": ("gate_proj", "gate", "mlp_gate", "gate_proj_weight"),
    "Down": ("down_proj", "down", "mlp_down", "down_proj_weight"),
    "Q": ("q_proj", "q", "self_attn_q", "q_proj_weight"),
    "K": ("k_proj", "k", "self_attn_k", "k_proj_weight"),
    "O": ("o_proj", "o", "self_attn_o", "o_proj_weight"),
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def hidden_space_v1(W: np.ndarray) -> np.ndarray | None:
    """Leading singular vector of W that lives in the HIDDEN-dim basis."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        return None
    a, b = W.shape
    # Economy SVD: W = U (a,k) @ diag(s) @ Vt (k,b), k = min(a,b).
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    if b == HIDDEN:
        v1 = Vt[0]            # right singular vector, length b == HIDDEN
    elif a == HIDDEN:
        v1 = U[:, 0]          # left singular vector, length a == HIDDEN
    else:
        return None
    nrm = np.linalg.norm(v1)
    if nrm < 1e-12:
        return None
    return v1 / nrm


def resolve_key(blob, aliases) -> str | None:
    keys = set(blob.files)
    for cand in aliases:
        if cand in keys:
            return cand
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not WEIGHTS_NPZ.exists():
        print("MISSING_REGEN_INPUT", WEIGHTS_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, HIDDEN) and y.shape == (500,)

    # Supervised difference-of-means direction in residual-stream space.
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    dom_norm = np.linalg.norm(dom)
    if dom_norm < 1e-12:
        print("DEGENERATE_DOM", file=sys.stderr)
        return 3
    dom_unit = dom / dom_norm
    # Sanity: DoM should recover the headline ~0.77 AUROC on the cache.
    dom_auroc = auroc(X @ dom_unit, y)

    weights = np.load(WEIGHTS_NPZ)
    per_proj = {}
    max_abs_cos = 0.0
    max_proj = None
    for label, aliases in PROJ_ALIASES.items():
        key = resolve_key(weights, aliases)
        if key is None:
            per_proj[label] = {"status": "missing_key", "abs_cos": None}
            continue
        v1 = hidden_space_v1(weights[key])
        if v1 is None:
            per_proj[label] = {
                "status": "no_hidden_basis",
                "shape": list(np.asarray(weights[key]).shape),
                "abs_cos": None,
            }
            continue
        cos = float(v1 @ dom_unit)
        abs_cos = abs(cos)
        per_proj[label] = {
            "status": "ok",
            "key": key,
            "shape": list(np.asarray(weights[key]).shape),
            "cos_signed": cos,
            "abs_cos": abs_cos,
            "exceeds_threshold": bool(abs_cos > THRESHOLD),
        }
        if abs_cos > max_abs_cos:
            max_abs_cos = abs_cos
            max_proj = label

    out = {
        "experiment": "P11-FE977",
        "hypothesis": "DoM_supervised is leading weight direction (artifact) iff |cos|>0.85",
        "threshold": THRESHOLD,
        "dom_self_auroc": float(dom_auroc),
        "per_projection": per_proj,
        "max_abs_cos": float(max_abs_cos),
        "max_abs_cos_projection": max_proj,
        "any_exceeds_threshold": bool(max_abs_cos > THRESHOLD),
        "verdict": (
            "ARTIFACT_PLAUSIBLE" if max_abs_cos > THRESHOLD else "NOT_WEIGHT_ARTIFACT"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())