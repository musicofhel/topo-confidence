"""P11-FE874 — ϕ_{1/2} superposition-regime diagnostic on the Qwen2.5-1.5B unembedding.

Loads the cached Qwen2.5-1.5B unembedding matrix W_U (vocab × hidden = 152064 × 1536)
and computes the ϕ_{1/2} statistic Liu et al. use to place a model in the strong- vs
weak-superposition regime: the fraction of vectors whose L2 norm exceeds half the
median norm. We report it both column-wise (per the literal FE description, over the
1536 hidden axes) and feature-wise (over the 152064 vocab rows, the superposition-
relevant direction), alongside the loading factor m/n = vocab/hidden.

Regime call: strong superposition packs many comparably-normed feature directions
(ϕ_{1/2} → 1 over features, effective feature count ≫ ambient dim); weak superposition
leaves a long tail of near-zero feature norms (ϕ_{1/2} markedly below 1). All four
refutations and the framing pivot assume Qwen sits in the strong regime Liu et al.
infer for OPT/Pythia/GPT-2; this script tests that premise directly. No new data.
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
# Cached unembedding W_U, exported once from the HF checkpoint (no torch at runtime).
WU_CANDIDATES = [
    ROOT / "pathway11_h100/superposition/qwen15b_unembed.npz",
    ROOT / "pathway11_h100/superposition/qwen15b_unembed.npy",
    ROOT / "pathway11_h100/superposition/W_U.npz",
    ROOT / "pathway11_h100/superposition/W_U.npy",
]
OUT_JSON = ROOT / "pathway11_h100/superposition/results.json"

VOCAB = 152064
HIDDEN = 1536


def load_wu() -> np.ndarray | None:
    """Return W_U as (vocab, hidden) float32, or None if no cache is present."""
    path = next((p for p in WU_CANDIDATES if p.exists()), None)
    if path is None:
        return None
    if path.suffix == ".npy":
        W = np.load(path)
    else:
        blob = np.load(path)
        key = next((k for k in ("W_U", "unembed", "lm_head", "weight", "arr_0")
                    if k in blob.files), blob.files[0])
        W = blob[key]
    W = np.asarray(W, dtype=np.float32)
    # Normalize orientation to (vocab, hidden): larger axis is the vocab axis.
    if W.ndim != 2:
        raise ValueError(f"expected 2-D W_U, got shape {W.shape}")
    if W.shape[0] < W.shape[1]:
        W = W.T
    return W


def phi_half(norms: np.ndarray) -> float:
    """Fraction of vectors with L2 norm > 0.5 * median(norm)."""
    med = float(np.median(norms))
    if med == 0.0:
        return float("nan")
    return float(np.mean(norms > 0.5 * med))


def main() -> int:
    W = load_wu()
    if W is None:
        print("MISSING_REGEN_INPUT", " | ".join(str(p) for p in WU_CANDIDATES),
              file=sys.stderr)
        return 2

    n_features, n_dims = W.shape  # (vocab, hidden)
    m_over_n = n_features / n_dims

    # Feature-wise: each row is a token's embedding direction in hidden space.
    feature_norms = np.linalg.norm(W, axis=1)
    # Column-wise (literal FE description): norm of each hidden-axis column.
    column_norms = np.linalg.norm(W, axis=0)

    phi_half_feature = phi_half(feature_norms)
    phi_half_column = phi_half(column_norms)

    # Effective feature count via participation ratio of squared feature norms:
    # (Σ s²)² / Σ s⁴  — a basis-free estimate of how many directions are active.
    s2 = feature_norms.astype(np.float64) ** 2
    denom = float(np.sum(s2 ** 2))
    participation_ratio = float((np.sum(s2) ** 2) / denom) if denom > 0 else float("nan")
    pr_over_dim = participation_ratio / n_dims

    # Regime verdict. Strong superposition: nearly all feature directions carry
    # comparable, substantial norm (ϕ_{1/2}→1) and the effective feature count far
    # exceeds the ambient dimension (participation ratio ≫ n_dims is impossible by
    # construction, so we read "strong" off the near-uniform feature-norm spectrum).
    if not np.isnan(phi_half_feature):
        if phi_half_feature >= 0.90:
            regime = "strong"
        elif phi_half_feature >= 0.70:
            regime = "intermediate"
        else:
            regime = "weak"
    else:
        regime = "undetermined"

    out = {
        "experiment": "P11-FE874",
        "wu_shape": [int(n_features), int(n_dims)],
        "vocab": int(n_features),
        "hidden": int(n_dims),
        "m_over_n": float(m_over_n),
        "phi_half_feature": phi_half_feature,
        "phi_half_column": phi_half_column,
        "feature_norm_median": float(np.median(feature_norms)),
        "feature_norm_mean": float(np.mean(feature_norms)),
        "feature_norm_min": float(np.min(feature_norms)),
        "feature_norm_max": float(np.max(feature_norms)),
        "column_norm_median": float(np.median(column_norms)),
        "n_features_above_half_median": int(np.sum(feature_norms > 0.5 * np.median(feature_norms))),
        "participation_ratio_features": participation_ratio,
        "participation_ratio_over_dim": pr_over_dim,
        "regime": regime,
        "note": (
            "phi_half is a fraction in [0,1]; m/n (vocab/hidden) is reported "
            "alongside it as the loading factor. Strong superposition <=> "
            "phi_half_feature near 1 (near-uniform feature norms, dense packing); "
            "weak <=> a long tail of near-zero feature norms drags phi_half down. "
            "Liu et al.'s regime claim applies to our model only if regime=='strong'."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())