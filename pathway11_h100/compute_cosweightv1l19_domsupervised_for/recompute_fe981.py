"""P11-FE981 — Is the supervised DoM direction a weight-structural artifact?

Computes the leading singular direction (in residual-stream / hidden space) of
each L19 projection weight matrix — gate, down, q, k, o — and measures its
cosine alignment with the supervised Difference-of-Means (DoM) direction fit on
the L19 prefill hidden states.

Logic: the supervised DoM is d = mean(prefill | correct) - mean(prefill |
incorrect), a (1536,) residual-stream direction. For each projection weight
matrix W, the leading singular vector restricted to the hidden (1536-d) axis is
the dominant direction that projection reads from / writes to the residual
stream — call it weight_v1. If |cos(weight_v1, DoM)| > 0.85 for any projection,
the DoM is plausibly the leading weight direction (a training-mechanics
artifact) rather than a learned correctness signal. Context: cos(DoM, PC1) =
0.9216 already suggests PC1 ≈ a structural weight direction.

Read projections (gate, q, k) take the right-singular (input-axis) direction;
write projections (down, o) take the left-singular (output-axis) direction —
selected automatically by matching the 1536-d hidden axis.
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
WEIGHTS = ROOT / "pathway11_h100/weight_structure/l19_weights.npz"
OUT_JSON = ROOT / "pathway11_h100/weight_structure/results.json"

HIDDEN_DIM = 1536
ALIGN_THRESHOLD = 0.85

# Candidate NPZ key aliases for each L19 projection weight matrix. The script
# accepts the first present alias so it is robust to minor extractor naming.
PROJ_KEYS = {
    "gate": ["gate_proj", "mlp.gate_proj", "gate"],
    "down": ["down_proj", "mlp.down_proj", "down"],
    "q": ["q_proj", "self_attn.q_proj", "q"],
    "k": ["k_proj", "self_attn.k_proj", "k"],
    "o": ["o_proj", "self_attn.o_proj", "o"],
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def hidden_leading_singular_vector(W: np.ndarray, hidden_dim: int):
    """Leading singular vector of W restricted to the hidden-dim axis.

    If the input axis (columns) is hidden_dim, returns the top right-singular
    vector via eigh(W^T W). If the output axis (rows) is hidden_dim, returns the
    top left-singular vector via eigh(W W^T). Returns (vec, axis) or (None,None).
    """
    W = W.astype(np.float64)
    if W.ndim != 2:
        return None, None
    m, n = W.shape
    if n == hidden_dim:
        gram = W.T @ W
        axis = "input"
    elif m == hidden_dim:
        gram = W @ W.T
        axis = "output"
    else:
        return None, None
    vals, vecs = np.linalg.eigh(gram)  # ascending eigenvalues
    return vecs[:, -1], axis


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not WEIGHTS.exists():
        print("MISSING_REGEN_INPUT", WEIGHTS, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    if X.shape != (500, HIDDEN_DIM) or y.shape != (500,):
        print("UNEXPECTED_CACHE_SHAPE", X.shape, y.shape, file=sys.stderr)
        return 2

    # Supervised DoM direction on the L19 prefill hidden states.
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    dom_auroc = auroc(X @ dom, y)

    wblob = np.load(WEIGHTS)
    available = set(wblob.files)

    per_proj = {}
    max_abs_cos = 0.0
    max_abs_proj = None
    any_artifact = False

    for proj, aliases in PROJ_KEYS.items():
        key = next((a for a in aliases if a in available), None)
        if key is None:
            per_proj[proj] = {"status": "MISSING_WEIGHT_KEY", "tried": aliases}
            continue
        W = wblob[key]
        v1, axis = hidden_leading_singular_vector(np.asarray(W), HIDDEN_DIM)
        if v1 is None:
            per_proj[proj] = {
                "status": "NO_HIDDEN_AXIS",
                "key": key,
                "shape": list(np.asarray(W).shape),
            }
            continue
        c = cos(v1, dom)
        abs_c = abs(c)
        artifact = bool(abs_c > ALIGN_THRESHOLD)
        any_artifact = any_artifact or artifact
        if abs_c > max_abs_cos:
            max_abs_cos = abs_c
            max_abs_proj = proj
        per_proj[proj] = {
            "status": "OK",
            "key": key,
            "weight_shape": list(np.asarray(W).shape),
            "hidden_axis": axis,
            "cos_weight_v1_dom": c,
            "abs_cos_weight_v1_dom": abs_c,
            "artifact_above_threshold": artifact,
        }

    out = {
        "experiment": "P11-FE981",
        "hidden_dim": HIDDEN_DIM,
        "align_threshold": ALIGN_THRESHOLD,
        "dom_supervised_auroc_in_sample": dom_auroc,
        "dom_norm": float(np.linalg.norm(dom)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "per_projection": per_proj,
        "max_abs_cos": max_abs_cos,
        "max_abs_cos_projection": max_abs_proj,
        "any_projection_is_artifact": any_artifact,
        "verdict": (
            "WEIGHT_STRUCTURAL_ARTIFACT" if any_artifact else "LEARNED_SIGNAL_NOT_WEIGHT_v1"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())