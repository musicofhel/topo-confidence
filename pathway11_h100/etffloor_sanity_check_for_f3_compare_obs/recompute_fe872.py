"""P11-FE872 — ETF-floor sanity check for F-3.

F-3 reports cos(prefill_DoM, final_DoM) = 0.046 (cos^2 = 0.0021), framed as
near-orthogonality of the prefill and final-token Difference-of-Means
directions in the L19 residual stream. Liu et al. 2505.10465 predict ETF-like
residual geometry in which two unrelated unit directions have squared overlap
distributed as Beta(1/2, (m-1)/2), with mean 1/m. For Qwen2.5-1.5B (m=1536)
that is E[cos^2] ~ 6.5e-4, whereas the observed cos^2 ~ 0.0021 ~ 3.2/m.

This script formalises the null test: it reconstructs m from the cached prefill
activations, draws / evaluates the Beta(1/2, (m-1)/2) random-ETF null both
analytically and by Monte Carlo, and reports the z-score and survival p-value
of the observed cos^2. A large positive z means the prefill/final overlap is
*above* the ETF floor — i.e. the directions are less orthogonal than a random
pair, so F-3's orthogonality is a genuine (computational) finding rather than a
geometric default. A z near 0 means the observed overlap is indistinguishable
from random-ETF chance.

The final_DoM direction is not in any local cache, so the observed cosine is
read from scratch/pathway10_temporal_and_verifier_results.json when available
and otherwise falls back to the documented F-3 value. The cached prefill
activations are still loaded to (a) ground the dimensionality m and (b) confirm
the prefill DoM direction exists.
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
from scipy.stats import beta as beta_dist

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
SCRATCH = ROOT / "scratch/pathway10_temporal_and_verifier_results.json"
OUT_JSON = ROOT / "pathway11_h100/etf_floor/results.json"

SEED = 9999
N_MC = 2_000_000

# Documented F-3 value (cos(prefill_DoM, final_DoM)); used if scratch JSON has
# no recognisable key.
DEFAULT_OBSERVED_COS = 0.046


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _find_cos(obj) -> float | None:
    """Recursively search a JSON blob for a prefill/final DoM cosine value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, (int, float)) and "cos" in kl and (
                "final" in kl or "temporal" in kl or "prefill_final" in kl
            ):
                return float(v)
        for v in obj.values():
            hit = _find_cos(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_cos(v)
            if hit is not None:
                return hit
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.ndim == 2 and X.shape[0] == 500, X.shape
    m = int(X.shape[1])

    # Ground the prefill DoM direction (confirms it is non-degenerate).
    prefill_dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    prefill_dom_norm = float(np.linalg.norm(prefill_dom))

    # Sanity AUROC of the cached DoM scores, if present.
    dom_auroc = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom_score.shape[0] == y.shape[0]:
            dom_auroc = auroc(dom_score, y)

    # Observed cos(prefill_DoM, final_DoM).
    observed_cos = DEFAULT_OBSERVED_COS
    observed_cos_source = "documented_default"
    if SCRATCH.exists():
        try:
            found = _find_cos(json.loads(SCRATCH.read_text()))
            if found is not None and abs(found) <= 1.0:
                observed_cos = float(found)
                observed_cos_source = "scratch_json"
        except (ValueError, OSError):
            pass

    observed_cos2 = observed_cos ** 2

    # Analytic Beta(1/2, (m-1)/2) random-ETF null on cos^2.
    a = 0.5
    b = (m - 1) / 2.0
    null_mean = a / (a + b)                       # == 1/m
    null_var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    null_std = float(np.sqrt(null_var))
    z_cos2 = (observed_cos2 - null_mean) / null_std
    p_analytic = float(beta_dist.sf(observed_cos2, a, b))

    # Monte Carlo confirmation: cos^2 of a random unit vector against a fixed
    # axis is (g_0^2 / ||g||^2) for g ~ N(0, I_m), distributed Beta(1/2,(m-1)/2).
    rng = np.random.default_rng(SEED)
    g = rng.standard_normal((N_MC, m))
    cos2_mc = g[:, 0] ** 2 / np.einsum("ij,ij->i", g, g)
    mc_mean = float(cos2_mc.mean())
    mc_std = float(cos2_mc.std(ddof=1))
    p_mc = float((cos2_mc >= observed_cos2).mean())
    z_mc = (observed_cos2 - mc_mean) / mc_std

    significant = bool(p_analytic < 0.05)
    verdict = (
        "ABOVE_ETF_FLOOR" if (z_cos2 > 0 and significant)
        else "AT_ETF_FLOOR" if not significant
        else "BELOW_ETF_FLOOR"
    )

    out = {
        "experiment": "P11-FE872",
        "m": m,
        "observed_cos": observed_cos,
        "observed_cos2": observed_cos2,
        "observed_cos_source": observed_cos_source,
        "observed_cos2_times_m": observed_cos2 * m,
        "prefill_dom_norm": prefill_dom_norm,
        "prefill_dom_auroc": dom_auroc,
        "null": {
            "distribution": "Beta(1/2, (m-1)/2)",
            "beta_a": a,
            "beta_b": b,
            "expected_cos2": null_mean,
            "expected_cos2_check_1_over_m": 1.0 / m,
            "std_cos2": null_std,
        },
        "z_cos2_analytic": float(z_cos2),
        "p_survival_analytic": p_analytic,
        "monte_carlo": {
            "n_samples": N_MC,
            "mean_cos2": mc_mean,
            "std_cos2": mc_std,
            "z_cos2": float(z_mc),
            "p_survival": p_mc,
        },
        "significant_at_0p05": significant,
        "verdict": verdict,
        "interpretation": (
            "Observed prefill/final DoM overlap exceeds the random-ETF floor: "
            "F-3 orthogonality is computational, not a geometric default."
            if verdict == "ABOVE_ETF_FLOOR" else
            "Observed overlap is indistinguishable from the random-ETF null: "
            "F-3 orthogonality is consistent with a geometric default."
            if verdict == "AT_ETF_FLOOR" else
            "Observed overlap is below the random-ETF floor (anti-aligned)."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())