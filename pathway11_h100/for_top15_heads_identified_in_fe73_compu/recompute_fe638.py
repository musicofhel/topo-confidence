"""P11-FE638 — Per-head prefill/final DoM orthogonality (F-3 rescue test).

F-3 reports cos(prefill_DoM, final_DoM) = 0.046 at the *residual* level (L19,
1536-d). MeltRTL applies the same per-head angle θ_h at every token position,
implicitly asserting that each head's correctness direction is stable across
positions. This script tests whether F-3's near-orthogonality is a
residual-aggregation artifact: for the top-15 correctness-critical heads
identified in FE73, it slices the L19 residual stream into per-head blocks,
computes the prefill-position and final-position DoM direction *within each
head block*, and reports the per-head cosine distribution.

If mean per-head cos >= 0.5 for these heads, the residual-level orthogonality
is an aggregation artifact and per-circuit directions are in fact stable; if
the per-head cosines hug 0.046, F-3 holds at the head level too.

Inputs (all cached, no GPU/network):
  - L19 prefill residual + labels  (m15b_prefill.npz)
  - L19 final-token residual       (m15b_final.npz)
  - FE73 top-head indices          (fe73_top_heads.json)
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
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
FE73_JSON = ROOT / "pathway11_h100/head_attribution/fe73_top_heads.json"
OUT_JSON = ROOT / "pathway11_h100/head_dom_orthogonality/results.json"

D_MODEL = 1536
F3_RESIDUAL_COS = 0.046  # F-3 anchor (1024-tok canonical)
ARTIFACT_THRESHOLD = 0.5
DEFAULT_N_HEADS = 12  # Qwen2.5-1.5B: 1536 / 128 = 12 attention heads at L19


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means direction (mean over correct - mean over incorrect)."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def main() -> int:
    for path in (PREFILL_CACHE, FINAL_CACHE, FE73_JSON):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2

    pre_blob = np.load(PREFILL_CACHE)
    fin_blob = np.load(FINAL_CACHE)

    X_pre = pre_blob["prefill"].astype(np.float64)
    y = pre_blob["correct"].astype(bool)
    # final cache may store the residual under "final" or "prefill"; accept either
    fin_key = "final" if "final" in fin_blob.files else "prefill"
    X_fin = fin_blob[fin_key].astype(np.float64)

    if X_pre.shape != (500, D_MODEL) or X_fin.shape != (500, D_MODEL):
        print("BAD_SHAPE", X_pre.shape, X_fin.shape, file=sys.stderr)
        return 3
    if y.shape != (500,):
        print("BAD_SHAPE", y.shape, file=sys.stderr)
        return 3

    fe73 = json.loads(FE73_JSON.read_text())
    n_heads = int(fe73.get("n_heads", DEFAULT_N_HEADS))
    if D_MODEL % n_heads != 0:
        print("BAD_HEAD_PARTITION", n_heads, file=sys.stderr)
        return 3
    head_dim = D_MODEL // n_heads

    # FE73 top-head indices (head-block indices into the L19 residual stream).
    top_heads = None
    for key in ("top_heads", "top15_heads", "heads", "head_indices"):
        if key in fe73:
            top_heads = [int(h) for h in fe73[key]]
            break
    if top_heads is None:
        print("MISSING_TOP_HEADS", file=sys.stderr)
        return 3
    top_heads = [h for h in top_heads if 0 <= h < n_heads][:15]
    if not top_heads:
        print("NO_VALID_HEADS", file=sys.stderr)
        return 3

    # Reference: residual-level prefill/final DoM cosine (should reproduce ~0.046).
    dom_pre_full = dom_direction(X_pre, y)
    dom_fin_full = dom_direction(X_fin, y)
    residual_cos = cosine(dom_pre_full, dom_fin_full)

    per_head = []
    for h in top_heads:
        sl = slice(h * head_dim, (h + 1) * head_dim)
        dp = dom_direction(X_pre[:, sl], y)
        df = dom_direction(X_fin[:, sl], y)
        c = cosine(dp, df)
        per_head.append({
            "head": h,
            "cos_prefill_final_dom": c,
            "prefill_head_dom_auroc": auroc(X_pre[:, sl] @ dp, y),
        })

    cos_vals = np.array([d["cos_prefill_final_dom"] for d in per_head], dtype=np.float64)
    finite = cos_vals[np.isfinite(cos_vals)]
    mean_cos = float(finite.mean()) if finite.size else float("nan")

    is_artifact = bool(np.isfinite(mean_cos) and mean_cos >= ARTIFACT_THRESHOLD)

    out = {
        "experiment": "P11-FE638",
        "description": "Per-head prefill/final DoM cosine vs F-3 residual cos=0.046",
        "n_heads": n_heads,
        "head_dim": head_dim,
        "top_heads": top_heads,
        "f3_residual_cos_anchor": F3_RESIDUAL_COS,
        "residual_cos_recomputed": residual_cos,
        "artifact_threshold": ARTIFACT_THRESHOLD,
        "per_head": per_head,
        "mean_per_head_cos": mean_cos,
        "median_per_head_cos": float(np.median(finite)) if finite.size else float("nan"),
        "std_per_head_cos": float(finite.std(ddof=0)) if finite.size else float("nan"),
        "min_per_head_cos": float(finite.min()) if finite.size else float("nan"),
        "max_per_head_cos": float(finite.max()) if finite.size else float("nan"),
        "n_heads_above_threshold": int((finite >= ARTIFACT_THRESHOLD).sum()),
        "f3_orthogonality_is_aggregation_artifact": is_artifact,
        "verdict": (
            "ARTIFACT: per-head DoM directions are stable across positions; "
            "F-3 residual orthogonality is an aggregation effect"
            if is_artifact else
            "F-3 HOLDS: prefill/final DoM near-orthogonal at the head level too"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())