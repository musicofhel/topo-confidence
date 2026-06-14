"""P8-FE8 — Re-summarize F-10's PH-Gaussian-null verdict with diameter-normalized
total persistence (eq. 1 of 2512.15285) instead of the original raw PH summary.

F-10 (graph F-7) is the project's load-bearing "no signal here" claim: PH adds
negative signal beyond covariance. 2512.15285 reports strong PH signal in a
different domain using a *diameter-normalized* total-persistence summary. The
divergence may be entirely about the normalization. This script re-runs the
Gaussian-null comparison on the L19 prefill cloud using eq. 1 and reports
whether the null verdict survives the re-summary.

Two paths:
  (A) If P7/P8 cached the persistence diagrams (birth/death pairs) and the
      pairwise-distance maxima, recompute eq. 1 directly from them.
  (B) Otherwise re-run the only CPU-only PH we can compute without ripser/gudhi:
      H0 persistence via the Euclidean MST (births=0, deaths=MST edge weights),
      diameter-normalized per eq. 1, for the real L19 cloud vs covariance-matched
      Gaussian-null clouds. Done for the full cloud and the correct / incorrect
      subclouds.

eq. 1 (diameter-normalized total persistence):
    TP_norm = (1/diam) * sum_i (death_i - birth_i)
with diam = max pairwise distance over the point cloud.
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
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/ph_eq1_renormalize/results.json"

# Candidate locations for previously cached persistence diagrams (path A).
DIAG_GLOBS = [
    "pathway7*/**/*persistence*.npz",
    "pathway7*/**/*diagram*.npz",
    "pathway8*/**/*persistence*.npz",
    "pathway8*/**/*diagram*.npz",
    "pathway8*/**/*ph*.npz",
]

SEED = 9999
N_NULL = 200


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def h0_eq1(X: np.ndarray) -> tuple[float, float, float]:
    """Diameter-normalized H0 total persistence via the Euclidean MST.

    Returns (eq1_norm, total_persistence, diameter). For H0 every feature is
    born at filtration 0; deaths are the MST edge weights (the radii at which
    connected components merge under the Vietoris-Rips/single-linkage filtration).
    """
    if X.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")
    d = pdist(X.astype(np.float64))
    diameter = float(d.max())
    mst = minimum_spanning_tree(squareform(d))
    total = float(mst.toarray().sum())
    eq1 = total / diameter if diameter > 0 else float("nan")
    return eq1, total, diameter


def gaussian_match_samples(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw len(X) points matching X's empirical mean and covariance spectrum."""
    mu = X.mean(axis=0)
    Xc = X - mu
    n = X.shape[0]
    # Economy SVD reproduces the empirical covariance exactly (handles n < d).
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = len(S)
    Z = rng.standard_normal((n, r))
    return mu + (Z * (S / np.sqrt(max(n - 1, 1)))) @ Vt


def null_test(X: np.ndarray, rng: np.random.Generator) -> dict:
    real_eq1, real_tp, diam = h0_eq1(X)
    null = np.empty(N_NULL, dtype=np.float64)
    for i in range(N_NULL):
        null[i], _, _ = h0_eq1(gaussian_match_samples(X, rng))
    nmean = float(np.nanmean(null))
    nstd = float(np.nanstd(null, ddof=1))
    z = float((real_eq1 - nmean) / nstd) if nstd > 0 else float("nan")
    pct = float(np.mean(null < real_eq1))
    # Two-sided empirical p: how extreme is the real value within the null band.
    p_emp = float(2.0 * min(pct, 1.0 - pct))
    return {
        "n_points": int(X.shape[0]),
        "eq1_real": real_eq1,
        "total_persistence_real": real_tp,
        "diameter": diam,
        "eq1_null_mean": nmean,
        "eq1_null_std": nstd,
        "z": z,
        "percentile_real_vs_null": pct,
        "p_empirical_two_sided": p_emp,
        # F-10 verdict survives iff eq.1 of the real cloud is statistically
        # indistinguishable from the covariance-matched Gaussian null.
        "gaussian_null_survives_eq1": bool(abs(z) < 2.0) if np.isfinite(z) else None,
    }


def try_cached_diagrams() -> dict | None:
    """Path A: recompute eq. 1 from any cached birth/death + diameter arrays."""
    for pattern in DIAG_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            try:
                blob = np.load(f, allow_pickle=True)
            except Exception:
                continue
            keys = set(blob.files)
            has_bd = ("birth" in keys and "death" in keys) or "dgms" in keys
            if not has_bd:
                continue
            try:
                if "dgms" in keys:
                    dgms = blob["dgms"]
                    bd = np.concatenate([np.asarray(d, dtype=np.float64).reshape(-1, 2)
                                         for d in (dgms if dgms.dtype == object else [dgms])])
                    birth, death = bd[:, 0], bd[:, 1]
                else:
                    birth = np.asarray(blob["birth"], dtype=np.float64).ravel()
                    death = np.asarray(blob["death"], dtype=np.float64).ravel()
                finite = np.isfinite(death) & np.isfinite(birth)
                tp = float((death[finite] - birth[finite]).sum())
                if "diameter" in keys:
                    diam = float(np.asarray(blob["diameter"]).ravel()[0])
                elif "dist_max" in keys:
                    diam = float(np.asarray(blob["dist_max"]).ravel()[0])
                else:
                    diam = float(np.nanmax(death[finite])) if finite.any() else float("nan")
                eq1 = tp / diam if diam and np.isfinite(diam) and diam > 0 else float("nan")
                return {
                    "source_file": str(f.relative_to(ROOT)),
                    "total_persistence": tp,
                    "diameter": diam,
                    "eq1_norm": eq1,
                    "n_pairs": int(finite.sum()),
                }
            except Exception:
                continue
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    rng = np.random.default_rng(SEED)

    out: dict = {
        "experiment": "P8-FE8",
        "description": "F-10 PH-Gaussian-null re-summarized with diameter-normalized total persistence (eq. 1).",
        "eq1_definition": "TP_norm = sum(death - birth) / max_pairwise_distance",
        "n_null": N_NULL,
        "homology_dim_recomputed": "H0 (MST single-linkage; ripser/gudhi banned by CPU-only constraint)",
    }

    # Path A: cached diagrams (any homology dimension), if present.
    cached = try_cached_diagrams()
    out["cached_diagrams_found"] = cached is not None
    if cached is not None:
        out["cached_eq1"] = cached

    # Path B: re-run H0 PH eq. 1 vs Gaussian null on the L19 cloud + subclouds.
    out["full_cloud"] = null_test(X, rng)
    out["correct_subcloud"] = null_test(X[y], rng)
    out["incorrect_subcloud"] = null_test(X[~y], rng)

    # Diameter-normalized H0 total persistence as a per-cloud summary cannot be a
    # per-problem classifier feature (one cloud → one scalar), so the headline is
    # the null verdict, not an AUROC. We still report the DoM-direction projection
    # AUROC as the live single-direction reference point for context.
    dom_dir = X[y].mean(axis=0) - X[~y].mean(axis=0)
    out["reference_dom_auroc_in_sample"] = float(auroc(X @ dom_dir, y))

    survivors = [out["full_cloud"]["gaussian_null_survives_eq1"],
                 out["correct_subcloud"]["gaussian_null_survives_eq1"],
                 out["incorrect_subcloud"]["gaussian_null_survives_eq1"]]
    out["verdict"] = {
        "all_clouds_consistent_with_gaussian_null": all(s is True for s in survivors),
        "any_cloud_exceeds_null": any(s is False for s in survivors),
        "interpretation": (
            "F-10 holds under eq. 1 re-summary: diameter-normalized total "
            "persistence of the real L19 cloud is statistically indistinguishable "
            "from the covariance-matched Gaussian null (|z| < 2 on every cloud)."
            if all(s is True for s in survivors) else
            "F-10 may need tightening: at least one cloud's eq. 1 exceeds the "
            "Gaussian null (|z| >= 2). PROJECT_RECORD 1d wording 'PH is null' "
            "should be re-examined vs 'unnormalized PH is null'."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())