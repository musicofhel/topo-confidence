"""P11-FE457 — So-Im unembedding direction vs prefill / final-token DoM.

Tests whether F-3's prefill/final-token orthogonality (cos 0.046) reflects two
distinct circuits or a single rising-confidence axis sampled at two timesteps.

Pulls the final-layer unembedding rows for the confidence tokens {`confident`,
`sure`}, folds the final RMSNorm scale into them (γ ⊙ W_U[t]) to recover the
residual-stream direction each token reads ("projecting through the inverse of
the final RMSNorm"), averages + normalizes, then cosines that direction against
the prefill DoM (F-2, computed here from the main cache) and the final-token DoM
(F-3, supplied by the extractor in the inputs cache).

Decision rule:
  - high cos with final DoM, low cos with prefill DoM  -> F-3 holds (two circuits)
  - both cosines high                                  -> collapses to one axis

The unembedding rows, RMSNorm γ, and the final-token DoM direction are all
model-weight / final-token-state dependent, so they must be produced by the
torch extractor and cached in SOIM_NPZ. This recompute is CPU/numpy-only and
returns MISSING_REGEN_INPUT until that cache exists.
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
# Extractor-produced inputs (torch-dependent): confidence-token unembedding
# rows, final RMSNorm gamma, and the L19 final-token DoM direction.
SOIM_NPZ = ROOT / "pathway11_h100/soim_unembedding/inputs.npz"
OUT_JSON = ROOT / "pathway11_h100/soim_unembedding/results.json"

CONFIDENCE_TOKENS = ["confident", "sure"]

# F-3 orthogonality reference: cos(prefill_DoM, final_DoM) reported at 0.046.
HIGH_COS = 0.30  # |cos| above this counts as "aligned"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not SOIM_NPZ.exists():
        print("MISSING_REGEN_INPUT", SOIM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # F-2: prefill L19 DoM direction (correct - incorrect mean difference).
    prefill_dom = X[y].mean(axis=0) - X[~y].mean(axis=0)

    blob = np.load(SOIM_NPZ)
    # Required keys; missing any -> treat as un-regenerated input.
    required = ["rmsnorm_gamma", "final_dom"] + [
        f"W_U_{tok}" for tok in CONFIDENCE_TOKENS
    ]
    missing = [k for k in required if k not in blob.files]
    if missing:
        print("MISSING_REGEN_INPUT", SOIM_NPZ, "keys:", ",".join(missing),
              file=sys.stderr)
        return 2

    gamma = blob["rmsnorm_gamma"].astype(np.float64)
    final_dom = blob["final_dom"].astype(np.float64)
    assert gamma.shape == (1536,) and final_dom.shape == (1536,)

    # Per-token residual-stream direction: g_t = gamma (elementwise) W_U[t].
    # (RMSNorm logit_t = (x/rms) (gamma . W_U[t]) => token t reads gamma*W_U[t].)
    per_token = {}
    dirs = []
    for tok in CONFIDENCE_TOKENS:
        w = blob[f"W_U_{tok}"].astype(np.float64)
        assert w.shape == (1536,)
        g = unit(gamma * w)
        per_token[tok] = {
            "cos_prefill_dom": cosine(g, prefill_dom),
            "cos_final_dom": cosine(g, final_dom),
        }
        dirs.append(g)

    soim = unit(np.mean(np.stack(dirs, axis=0), axis=0))

    cos_soim_prefill = cosine(soim, prefill_dom)
    cos_soim_final = cosine(soim, final_dom)
    cos_prefill_final = cosine(prefill_dom, final_dom)

    soim_aligns_final = abs(cos_soim_final) >= HIGH_COS
    soim_aligns_prefill = abs(cos_soim_prefill) >= HIGH_COS

    if soim_aligns_final and not soim_aligns_prefill:
        verdict = "F3_HOLDS_two_circuits"
    elif soim_aligns_final and soim_aligns_prefill:
        verdict = "ONE_AXIS_collapse"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "P11-FE457",
        "confidence_tokens": CONFIDENCE_TOKENS,
        "high_cos_threshold": HIGH_COS,
        "cos_soim_prefill_dom": cos_soim_prefill,
        "cos_soim_final_dom": cos_soim_final,
        "cos_prefill_dom_final_dom": cos_prefill_final,
        "per_token": per_token,
        "soim_aligns_final": bool(soim_aligns_final),
        "soim_aligns_prefill": bool(soim_aligns_prefill),
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())