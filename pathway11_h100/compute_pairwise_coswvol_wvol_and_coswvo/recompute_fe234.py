"""P11-FE234 — Pairwise W_vo Frobenius cosines + cos(W_vo, I) for Qwen-2.5-1.5B-Instruct.

Tests the buffer-orthogonality prediction of the buffer-mechanism reading of F-3
(cos(prefill_DoM, final_DoM) = 0.046). A trained transformer that implements a
read/write buffer pair should show near-zero Frobenius cosines between distinct
per-layer OV (value-output) circuits W_vo^(l) = W_o^(l) @ W_v^(l), mirroring the
3-layer toy-model Fig-D-equivalent in the trigger paper.

If real W_vo matrices are pairwise near-orthogonal, F-3's tiny cross-direction
cosine becomes the *predicted* signature of a buffer pair. If they are not
orthogonal, the buffer framing fails to lift and F-3's "orthogonal because
uncorrelated" reading is preferred.

Weights-only, CPU. Expects a pre-extracted NPZ of per-layer W_vo matrices (the
HF weights cannot be loaded here — no torch/transformers). Cache schema:
  W_vo: (L, d, d) float32 — per-layer OV circuit  W_o @ W_v
        (or supply  W_v: (L, d_kv, d)  and  W_o: (L, d, d_kv)  to build it)
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
CACHE = ROOT / "pathway11_h100/wvo_orthogonality/wvo_qwen15b.npz"
OUT_JSON = ROOT / "pathway11_h100/wvo_orthogonality/results.json"

NEAR_ORTHO_THRESH = 0.10  # |cos| below this counts as near-orthogonal


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _build_wvo(blob) -> np.ndarray:
    """Return per-layer W_vo of shape (L, d, d) from the NPZ blob."""
    if "W_vo" in blob.files:
        return blob["W_vo"].astype(np.float64)
    if "W_v" in blob.files and "W_o" in blob.files:
        W_v = blob["W_v"].astype(np.float64)  # (L, d_kv, d)
        W_o = blob["W_o"].astype(np.float64)  # (L, d, d_kv)
        L = W_v.shape[0]
        return np.stack([W_o[l] @ W_v[l] for l in range(L)], axis=0)
    raise KeyError("NPZ lacks 'W_vo' or ('W_v','W_o')")


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    try:
        W = _build_wvo(blob)
    except KeyError as e:
        print("MISSING_REGEN_INPUT", CACHE, str(e), file=sys.stderr)
        return 2

    if W.ndim != 3 or W.shape[1] != W.shape[2]:
        print("MISSING_REGEN_INPUT", CACHE, f"bad W_vo shape {W.shape}", file=sys.stderr)
        return 2

    L, d, _ = W.shape

    # Flatten each W_vo to a Frobenius-inner-product vector and unit-normalize.
    flat = W.reshape(L, -1)
    norms = np.linalg.norm(flat, axis=1)
    if np.any(norms == 0):
        print("MISSING_REGEN_INPUT", CACHE, "zero-norm W_vo layer", file=sys.stderr)
        return 2
    unit = flat / norms[:, None]

    # Pairwise Frobenius cosine matrix (L, L).
    cos_mat = unit @ unit.T
    np.fill_diagonal(cos_mat, 1.0)

    iu = np.triu_indices(L, k=1)
    off = cos_mat[iu]
    off_abs = np.abs(off)

    # cos(W_vo^(l), I) = <W_vo, I>_F / (||W_vo||_F * ||I||_F) = trace(W_vo) / (||W_vo||_F * sqrt(d))
    traces = np.array([np.trace(W[l]) for l in range(L)], dtype=np.float64)
    cos_I = traces / (norms * np.sqrt(d))

    frac_near_ortho = float(np.mean(off_abs < NEAR_ORTHO_THRESH))
    buffer_orthogonality_supported = bool(
        float(off_abs.mean()) < NEAR_ORTHO_THRESH and frac_near_ortho >= 0.90
    )

    out = {
        "experiment": "P11-FE234",
        "model": "Qwen-2.5-1.5B-Instruct",
        "n_layers": int(L),
        "hidden_dim": int(d),
        "n_pairs": int(off.size),
        "pairwise_cos": {
            "mean": float(off.mean()),
            "mean_abs": float(off_abs.mean()),
            "median_abs": float(np.median(off_abs)),
            "max_abs": float(off_abs.max()),
            "p95_abs": float(np.percentile(off_abs, 95)),
            "std": float(off.std()),
            "frac_near_ortho": frac_near_ortho,
        },
        "cos_with_identity": {
            "mean": float(cos_I.mean()),
            "mean_abs": float(np.abs(cos_I).mean()),
            "max_abs": float(np.abs(cos_I).max()),
            "per_layer": [float(c) for c in cos_I],
        },
        "near_ortho_thresh": NEAR_ORTHO_THRESH,
        "buffer_orthogonality_supported": buffer_orthogonality_supported,
        "f3_reference_cosine": 0.046,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())