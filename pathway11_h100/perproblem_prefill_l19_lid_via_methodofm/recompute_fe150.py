"""P11-FE150 — Per-problem prefill L19 local intrinsic dimensionality (LID)
via the Method-of-Moments (MoM) estimator at k=32 and k=64, on cached P11 H100
Qwen-2.5-1.5B L19 prefill activations.

Refutation 3 (F-7 reframing). F-7 reports a gap between *group* participation
ratio (PR computed over all points in a bucket: D 14.49, A 18.72, B 16.65,
C 20.04) and *per-problem* PR (local covariance PR averaged within a bucket:
D 12.84, A 12.66, B 12.80, C 12.97). That group-vs-local PR gap is exactly the
local-vs-global intrinsic-dimensionality phenomenon of Huang et al. This script
estimates per-problem MoM LID, takes the geometric mean per A/B/C/D bucket and
cross-bucket, and re-derives both group PR and mean local PR per bucket so the
ordering can be checked against F-7. If the D bucket shows the lowest
geometric-mean LID while the bucket-PR ordering reproduces, F-7 reframes as a
bucket-level local-LID collapse rather than a "collective signature."

Buckets A/B/C/D are defined here as ascending DoM-score quartiles (A = lowest
DoM confidence quartile … D = highest); the F-7 reference values are carried as
constants for comparison only.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/lid_mom/results.json"

K_VALUES = (32, 64)
BUCKETS = ("A", "B", "C", "D")

# F-7 reference participation ratios (carried for comparison only).
F7_GROUP_PR = {"A": 18.72, "B": 16.65, "C": 20.04, "D": 14.49}
F7_PERPROBLEM_PR = {"A": 12.66, "B": 12.80, "C": 12.97, "D": 12.84}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def geomean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(x))))


def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² of the covariance spectrum of point set X (rows=points)."""
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    # eigenvalues of covariance via singular values of centered data
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s.astype(np.float64) ** 2) / max(X.shape[0] - 1, 1)
    denom = float(np.sum(lam ** 2))
    if denom <= 0:
        return float("nan")
    return float((float(np.sum(lam)) ** 2) / denom)


def mom_lid(sorted_dists: np.ndarray, k: int) -> np.ndarray:
    """Method-of-Moments LID per point.

    sorted_dists: (n, n-1) per-point neighbor distances, ascending (self removed).
    LID_i = wbar / (wmax - wbar), with wbar the mean of the k nearest distances
    and wmax the k-th (largest of the k) distance.
    """
    nn = sorted_dists[:, :k]
    wbar = nn.mean(axis=1)
    wmax = nn[:, -1]
    denom = wmax - wbar
    out = np.full(wbar.shape, np.nan, dtype=np.float64)
    ok = denom > 1e-12
    out[ok] = wbar[ok] / denom[ok]
    return out


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    correct = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and correct.shape == (500,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (500,)

    n = X.shape[0]

    # Pairwise Euclidean distances; remove self by setting diagonal to +inf.
    D = cdist(X, X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)  # neighbor indices, nearest first
    sorted_dists = np.take_along_axis(D, order, axis=1)[:, : n - 1]

    # Ascending DoM-score quartile buckets A..D.
    edges = np.quantile(dom_score, [0.25, 0.50, 0.75])
    bucket_idx = np.digitize(dom_score, edges)  # 0..3 -> A..D
    bucket_of = np.array(BUCKETS)[bucket_idx]

    results: dict = {
        "experiment": "P11-FE150",
        "n": int(n),
        "bucket_definition": "ascending DoM-score quartiles (A=lowest .. D=highest)",
        "bucket_counts": {b: int(np.sum(bucket_of == b)) for b in BUCKETS},
        "dom_auroc": auroc(dom_score, correct),
        "f7_reference": {"group_pr": F7_GROUP_PR, "per_problem_pr": F7_PERPROBLEM_PR},
        "lid_mom": {},
        "group_pr": {},
        "per_problem_pr": {},
    }

    # Group PR per bucket and cross-bucket (global).
    for b in BUCKETS:
        results["group_pr"][b] = participation_ratio(X[bucket_of == b])
    results["group_pr"]["cross_bucket"] = participation_ratio(X)

    # MoM LID per problem at each k; geometric mean per bucket + cross-bucket.
    lid_by_k: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        kk = min(k, n - 1)
        lid = mom_lid(sorted_dists, kk)
        lid_by_k[k] = lid
        block = {b: geomean(lid[bucket_of == b]) for b in BUCKETS}
        block["cross_bucket"] = geomean(lid)
        # D-vs-rest contrast: is D the lowest geometric-mean LID bucket?
        non_d = [block[b] for b in ("A", "B", "C") if np.isfinite(block[b])]
        block["d_is_lowest"] = bool(
            np.isfinite(block["D"]) and non_d and block["D"] < min(non_d)
        )
        results["lid_mom"][f"k{k}"] = block

    # Per-problem (local) PR: PR of each point's k-NN neighborhood covariance,
    # averaged (arithmetic + geometric mean) per bucket. Uses k=32 neighborhoods.
    k_local = min(32, n - 1)
    local_pr = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        nbr = order[i, :k_local]
        local_pr[i] = participation_ratio(X[np.append(nbr, i)])
    for b in BUCKETS:
        vals = local_pr[bucket_of == b]
        results["per_problem_pr"][b] = {
            "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
            "geomean": geomean(vals),
        }
    results["per_problem_pr"]["cross_bucket"] = {
        "mean": float(np.nanmean(local_pr)),
        "geomean": geomean(local_pr),
        "k_local": int(k_local),
    }

    # Ordering-reproduction check: does our group-PR bucket ordering match F-7's?
    our_order = sorted(BUCKETS, key=lambda b: results["group_pr"][b])
    f7_order = sorted(BUCKETS, key=lambda b: F7_GROUP_PR[b])
    results["group_pr_ordering_matches_f7"] = bool(our_order == f7_order)
    results["our_group_pr_ascending_order"] = our_order
    results["f7_group_pr_ascending_order"] = f7_order

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())