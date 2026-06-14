"""P11-FE509 — PH-vs-dispersion reconciliation for F-10.

F-10 found that the trained L19 residual-stream point cloud sits at a Gaussian
null under persistent homology (PH adds negative signal beyond covariance).
The triggering paper claims scalar *dispersion* (mean pairwise cosine distance)
is highly informative on the same architecture family. PH is a higher-moment
(connectivity) statistic; dispersion is a first/second-moment scalar. This
script tests *which moment the geometry lives in* on the exact same point cloud:

  1. Compute, for each problem, its mean pairwise cosine distance to every other
     problem's L19 prefill vector (the paper's dispersion scalar).
  2. Bin problems into 10% strata of dispersion; measure the correctness rate
     per stratum and test monotonicity (Spearman + adjacent-step sign count).
  3. AUROC of dispersion vs correctness.

Verdict:
  - Monotonic (Spearman p < 0.05)  -> REFINES F-10: "PH null but mean-distance
    informative" — the signal lives in a lower moment than PH probes.
  - Flat (Spearman p >= 0.05)      -> REPLICATES F-10 and weakens the paper's
    universality claim: dispersion is uninformative here too.
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
from scipy.stats import spearmanr

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/ph_dispersion_reconciliation/results.json"

N_STRATA = 10
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def mean_pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """For each row i, mean cosine distance (1 - cos sim) to all other rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T                      # (n, n) cosine similarities
    n = X.shape[0]
    np.fill_diagonal(sim, 0.0)
    # mean over the n-1 off-diagonal entries; distance = 1 - sim
    sim_sum = sim.sum(axis=1)
    mean_sim = sim_sum / (n - 1)
    return 1.0 - mean_sim


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or y.shape != (500,):
        print("MISSING_REGEN_INPUT", "unexpected shapes", X.shape, y.shape,
              file=sys.stderr)
        return 2

    dispersion = mean_pairwise_cosine_distance(X)

    # 10% strata of dispersion (equal-count bins on sorted dispersion).
    order = np.argsort(dispersion, kind="stable")
    bins = np.array_split(order, N_STRATA)
    strata = []
    for b_idx, idx in enumerate(bins):
        strata.append({
            "stratum": b_idx,
            "n": int(len(idx)),
            "dispersion_lo": float(dispersion[idx].min()),
            "dispersion_hi": float(dispersion[idx].max()),
            "dispersion_mean": float(dispersion[idx].mean()),
            "correctness_rate": float(y[idx].mean()),
        })

    rates = np.array([s["correctness_rate"] for s in strata])
    mids = np.array([s["dispersion_mean"] for s in strata])

    # Monotonicity diagnostics.
    steps = np.diff(rates)
    n_up = int((steps > 0).sum())
    n_down = int((steps < 0).sum())
    stratum_rho, stratum_p = spearmanr(mids, rates)
    point_rho, point_p = spearmanr(dispersion, y.astype(np.float64))

    auroc_disp = auroc(dispersion, y)
    auroc_neg_disp = auroc(-dispersion, y)
    auroc_oriented = max(auroc_disp, auroc_neg_disp)

    monotonic = bool(np.isfinite(point_p) and point_p < 0.05)
    if monotonic:
        verdict = "REFINES_F10"
        interpretation = (
            "PH null but mean-distance informative: correctness varies "
            "monotonically with scalar dispersion on the same point cloud, so "
            "the signal lives in a lower moment than PH probes."
        )
    else:
        verdict = "REPLICATES_F10"
        interpretation = (
            "Dispersion is flat across correctness strata, replicating the "
            "F-10 Gaussian-null result and weakening the paper's universality "
            "claim: scalar dispersion is uninformative on this architecture."
        )

    out = {
        "experiment": "P11-FE509",
        "n_problems": int(len(y)),
        "overall_correctness_rate": float(y.mean()),
        "n_strata": N_STRATA,
        "strata": strata,
        "stratum_level_spearman_rho": float(stratum_rho),
        "stratum_level_spearman_p": float(stratum_p),
        "point_level_spearman_rho": float(point_rho),
        "point_level_spearman_p": float(point_p),
        "adjacent_steps_up": n_up,
        "adjacent_steps_down": n_down,
        "auroc_dispersion": float(auroc_disp),
        "auroc_neg_dispersion": float(auroc_neg_disp),
        "auroc_dispersion_oriented": float(auroc_oriented),
        "monotonic": monotonic,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())