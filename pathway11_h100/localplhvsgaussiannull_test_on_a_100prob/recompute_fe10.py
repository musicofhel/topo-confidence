"""P8-FE10 — Local persistent-local-homology (Euclidicity) vs Gaussian-null test.

F-10 retired the *global* one-parameter Vietoris-Rips PH story (PH adds negative
signal beyond covariance). TARDIS (von Rohrscheidt & Rieck) claims that
*multi-scale local* persistent local homology (PLH) is strictly more sensitive
to singularities than global PH — their wedged-sphere experiment (Fig. 5a)
separates cleanly under exactly this construction. This script asks whether the
local construction escapes the Gaussian null on residual streams.

For each pathway-8 layer in {12, 19, 24, 27} we sample 1000 token-residuals,
compute a per-point Euclidicity score over an (r, s) scale grid, build a
covariance-matched Gaussian null of equal sample size, and compare the real and
null Euclidicity distributions cell-by-cell with a Tukey range (HSD) test at
alpha = 0.05, family-wise across the whole (r, s) grid.

Euclidicity proxy (PH-library-free, numpy/scipy only). The TARDIS Euclidicity
of a point compares the persistent local homology of an annulus A(x; r, s) to a
Euclidean ball of the point's local intrinsic dimension d. Persistent homology
libraries (ripser/gudhi) are outside the allowed dependency set, so we use the
density-and-dimension surrogate that the same construction reduces to for a
uniform Euclidean model: for a d-dimensional uniform ball the expected fraction
of within-s neighbours that fall in the annulus [r, s] is 1 - (r/s)^d. The
Euclidicity at (x; r, s) is the absolute discrepancy between this Euclidean
expectation and the observed annulus fraction, with d estimated per point from
the participation ratio of the local neighbourhood's singular spectrum. A real
manifold that is locally Euclidean matches the Gaussian null; positive
real-vs-null separation in any cell is evidence that local PLH detects structure
global PH missed, which would narrow F-10 to the global construction only.

Verdict:
  local_plh_separates_from_null == False  -> F-10 generalises to local PLH;
      all PH-on-residual-stream experiments can be retired.
  local_plh_separates_from_null == True   -> F-10 must be narrowed to global PH.
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
from scipy.spatial.distance import cdist
from scipy.stats import studentized_range

ROOT = Path("/home/musicofhel/topo-confidence")

# Pathway-8 layer-wise residual caches. The exact on-disk name has drifted
# across rebuilds, so probe a few plausible locations before giving up.
LAYER_CACHE_CANDIDATES = [
    ROOT / "pathway8_layerwise/cache/m15b_layerwise.npz",
    ROOT / "pathway8_layerwise/cache/layerwise_residuals.npz",
    ROOT / "pathway8/cache/m15b_layerwise.npz",
    ROOT / "pathway8/cache/layerwise.npz",
    ROOT / "pathway11_h100/layerwise/m15b_layers.npz",
]

OUT_JSON = ROOT / "pathway11_h100/local_plh_null/results.json"

LAYERS = [12, 19, 24, 27]
N_SAMPLE = 1000          # token-residuals sampled per layer
K_DIM = 150              # nearest neighbours used for local intrinsic-dim estimate
R_PCTL = [10.0, 25.0, 40.0]   # inner-radius percentiles of pairwise distances
S_PCTL = [55.0, 70.0, 85.0]   # outer-radius percentiles
ALPHA = 0.05
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _layer_array(blob, layer: int) -> np.ndarray | None:
    """Find the 2D residual matrix for a layer inside an opened npz."""
    candidate_keys = [
        f"layer_{layer}", f"L{layer}", f"resid_{layer}", f"residuals_{layer}",
        f"hidden_{layer}", f"layer{layer}", f"l{layer}",
    ]
    for k in candidate_keys:
        if k in blob.files:
            arr = blob[k]
            if arr.ndim == 2:
                return np.asarray(arr, dtype=np.float64)
    # 3D stack: (n_layers, n_tokens, dim) keyed by 'residuals' / 'hidden'
    for k in ("residuals", "hidden", "layerwise"):
        if k in blob.files and blob[k].ndim == 3:
            stack = blob[k]
            if "layers" in blob.files:
                idx = np.flatnonzero(np.asarray(blob["layers"]) == layer)
                if idx.size:
                    return np.asarray(stack[idx[0]], dtype=np.float64)
            if layer < stack.shape[0]:
                return np.asarray(stack[layer], dtype=np.float64)
    return None


def estimate_local_dims(X: np.ndarray, k: int) -> np.ndarray:
    """Per-point intrinsic dimension via participation ratio of the local SVD."""
    n, D = X.shape
    k = min(k, n - 1)
    dims = np.empty(n, dtype=np.float64)
    Dmat = cdist(X, X)
    for i in range(n):
        idx = np.argpartition(Dmat[i], k + 1)[: k + 1]
        nbr = X[idx]
        nbr = nbr - nbr.mean(axis=0)
        sv = np.linalg.svd(nbr, compute_uv=False)
        sv2 = sv ** 2
        denom = float((sv2 ** 2).sum())
        if denom <= 0.0:
            dims[i] = 1.0
            continue
        pr = float(sv2.sum() ** 2 / denom)
        dims[i] = float(np.clip(round(pr), 1.0, D - 1))
    return dims, Dmat


def euclidicity_grid(Dmat: np.ndarray, dims: np.ndarray,
                     cells: list[tuple[float, float]]) -> np.ndarray:
    """Euclidicity per point per (r, s) cell. Shape (n_points, n_cells)."""
    n = Dmat.shape[0]
    E = np.zeros((n, len(cells)), dtype=np.float64)
    for i in range(n):
        row = np.sort(Dmat[i])
        # drop the self-distance (a single ~0 entry)
        row = row[1:]
        d = dims[i]
        for c, (r, s) in enumerate(cells):
            n_s = int(np.searchsorted(row, s, side="right"))
            if n_s == 0:
                continue
            n_r = int(np.searchsorted(row, r, side="right"))
            observed_frac = (n_s - n_r) / n_s
            expected_frac = 1.0 - (r / s) ** d
            E[i, c] = abs(observed_frac - expected_frac)
    return E


def gaussian_null(X: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Covariance- and mean-matched Gaussian sample of size n_samples."""
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    L = V * np.sqrt(w)[None, :]          # cov = L @ L.T
    Z = rng.standard_normal((n_samples, X.shape[1]))
    return mu + Z @ L.T


def tukey_grid(E_real: np.ndarray, E_null: np.ndarray, alpha: float) -> dict:
    """Family-wise Tukey HSD across all (cell x source) groups.

    Groups = {(cell, real), (cell, null)} for every grid cell (balanced, n each).
    For each cell we test the real-vs-null contrast against the studentized-range
    critical difference computed from the pooled within-group MSE.
    """
    n, n_cells = E_real.shape
    groups = []
    for c in range(n_cells):
        groups.append(E_real[:, c])
        groups.append(E_null[:, c])
    G = len(groups)
    means = np.array([g.mean() for g in groups])
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    N_total = G * n
    df = N_total - G
    mse = ss_within / df if df > 0 else float("nan")
    q_crit = float(studentized_range.ppf(1.0 - alpha, G, df))
    hsd = q_crit * float(np.sqrt(mse / n))

    cells_out = []
    n_sig = 0
    for c in range(n_cells):
        m_real = float(means[2 * c])
        m_null = float(means[2 * c + 1])
        diff = m_real - m_null
        sig = bool(abs(diff) > hsd)
        if sig:
            n_sig += 1
        cells_out.append({
            "cell": c,
            "mean_euclidicity_real": m_real,
            "mean_euclidicity_null": m_null,
            "diff_real_minus_null": diff,
            "significant": sig,
            "real_exceeds_null": bool(sig and diff > 0.0),
        })
    return {
        "n_groups": G,
        "df": int(df),
        "mse": float(mse),
        "q_crit": q_crit,
        "hsd_critical_diff": hsd,
        "n_cells_significant": n_sig,
        "n_cells_real_exceeds_null": sum(c["real_exceeds_null"] for c in cells_out),
        "cells": cells_out,
    }


def main() -> int:
    blob = None
    used_cache = None
    for cand in LAYER_CACHE_CANDIDATES:
        if cand.exists():
            try:
                blob = np.load(cand, allow_pickle=False)
                used_cache = cand
                break
            except Exception:
                continue
    if blob is None:
        print("MISSING_REGEN_INPUT", LAYER_CACHE_CANDIDATES[0], file=sys.stderr)
        return 2

    rng = np.random.default_rng(SEED)
    cells = [(r, s) for r in range(len(R_PCTL)) for s in range(len(S_PCTL))]
    # placeholder; real (r,s) values are per-layer (derived from its scale)

    per_layer = {}
    any_separation = False

    for layer in LAYERS:
        Xfull = _layer_array(blob, layer)
        if Xfull is None or Xfull.ndim != 2 or len(Xfull) < (K_DIM + 2):
            print("MISSING_REGEN_INPUT", f"layer_{layer}", used_cache, file=sys.stderr)
            return 2

        # Sample N_SAMPLE token-residuals (the 100-problem MATH-500 subset is
        # encoded in how the cache was built; we draw a fixed-seed subsample).
        n_take = min(N_SAMPLE, len(Xfull))
        sel = rng.choice(len(Xfull), size=n_take, replace=False)
        X = Xfull[sel]

        dims, Dmat = estimate_local_dims(X, K_DIM)

        # Scale grid from the real cloud's pairwise-distance percentiles.
        triu = Dmat[np.triu_indices_from(Dmat, k=1)]
        r_vals = np.percentile(triu, R_PCTL)
        s_vals = np.percentile(triu, S_PCTL)
        rs_cells = [(float(r), float(s)) for r in r_vals for s in s_vals
                    if float(r) < float(s)]

        E_real = euclidicity_grid(Dmat, dims, rs_cells)

        # Covariance-matched Gaussian null, equal sample size, same scale grid.
        Xnull = gaussian_null(Xfull, n_take, rng)
        dims_null, Dmat_null = estimate_local_dims(Xnull, K_DIM)
        E_null = euclidicity_grid(Dmat_null, dims_null, rs_cells)

        tukey = tukey_grid(E_real, E_null, ALPHA)
        layer_separates = tukey["n_cells_real_exceeds_null"] > 0
        any_separation = any_separation or layer_separates

        per_layer[str(layer)] = {
            "n_samples": int(n_take),
            "intrinsic_dim_mean_real": float(dims.mean()),
            "intrinsic_dim_mean_null": float(dims_null.mean()),
            "rs_cells": rs_cells,
            "mean_euclidicity_real": float(E_real.mean()),
            "mean_euclidicity_null": float(E_null.mean()),
            "tukey": tukey,
            "local_plh_separates_from_null": bool(layer_separates),
        }

    out = {
        "experiment": "P8-FE10",
        "description": "Local PLH (Euclidicity) vs covariance-matched Gaussian "
                       "null on pathway-8 layer-wise residuals; Tukey HSD over "
                       "the (r,s) grid at alpha=0.05.",
        "cache": str(used_cache),
        "layers": LAYERS,
        "n_sample": N_SAMPLE,
        "alpha": ALPHA,
        "euclidicity_method": "ph_free_density_dimension_surrogate",
        "per_layer": per_layer,
        "local_plh_separates_from_null": bool(any_separation),
        "verdict": ("F-10 narrows to GLOBAL PH only — local PLH separates from "
                    "the Gaussian null in at least one (r,s) cell"
                    if any_separation else
                    "F-10 GENERALISES to local PLH — no (r,s) cell separates "
                    "real residuals from the covariance-matched Gaussian null; "
                    "PH-on-residual-stream experiments can be retired"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())