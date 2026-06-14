"""P11-FE151 — Geometric-mean re-aggregation of the F-4 asymmetric-collapse claim.

F-4 reports that correct generations occupy a lower-dimensional final-token
manifold than incorrect ones ("asymmetric collapse"), quantified by participation
ratio (PR) and local intrinsic dimensionality (LID, k=32) and aggregated by the
arithmetic mean of the log quantity. LDReg (Theorem 4) argues the *geometric*
mean is the correct aggregator for dimensionality CDFs because the arithmetic
mean is dominated by heavy upper tails — so F-4's gap might be a heavy-tail
artifact.

This script recomputes per-point local LID (Levina-Bickel / MacKay MLE, k=32)
and per-point local PR (k=32 neighborhood covariance) on the cached L19 1.5B
final-token activations, aggregates each quantity within the correct / incorrect
groups by BOTH arithmetic and geometric mean, and reports:
  - the correct-vs-incorrect gap under each aggregator,
  - a standardized (Cohen's d, log scale) effect size,
  - a nonparametric bootstrap SE + 95% CI of the geometric-mean gap.

Decision rule (per the rationale): if the geometric-mean gap survives at
>1 bootstrap sigma the F-4 collapse strengthens; if it shrinks below 1 sigma the
F-4 arithmetic-mean gap was an aggregation / heavy-tail artifact.

Final-token activations are loaded from a candidate list; if no final-token NPZ
is present the script falls back to the L19 prefill cache and records the source
in the output JSON so the substrate is never silently misreported.
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
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/musicofhel/topo-confidence")
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Candidate final-token activation caches, tried in priority order.
FINAL_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
     ("final", "final_token", "hidden_final", "hidden", "h")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz",
     ("final", "final_token")),
]
OUT_JSON = ROOT / "pathway11_h100/fe151_geomean_collapse/results.json"

K = 32           # neighbors for LID + local PR
N_BOOT = 2000
SEED = 9999
EPS = 1e-12


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_activations() -> tuple[np.ndarray, str]:
    """Return (X, source_tag). Prefer a real final-token cache; fall back to
    the L19 prefill cache (guaranteed by the schema) and tag the fallback."""
    for path, keys in FINAL_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in keys:
            if key in blob.files:
                X = np.asarray(blob[key], dtype=np.float64)
                return X, f"{path.name}:{key}"
    # Fallback: L19 prefill states (only guaranteed substrate).
    if PREFILL_CACHE.exists():
        X = np.asarray(np.load(PREFILL_CACHE)["prefill"], dtype=np.float64)
        return X, f"{PREFILL_CACHE.name}:prefill_L19_FALLBACK"
    return None, "MISSING"  # type: ignore[return-value]


def local_lid_mle(dist: np.ndarray) -> np.ndarray:
    """Levina-Bickel / MacKay MLE local intrinsic dimensionality.

    `dist` is (n, K) distances to the K nearest neighbors (self excluded),
    sorted ascending. m_hat(i) = [ (1/(K-1)) sum_{j<K} log(T_K / T_j) ]^-1.
    """
    d = np.maximum(dist, EPS)
    Tk = d[:, -1][:, None]
    ratios = np.log(Tk / d[:, :-1])           # (n, K-1)
    inv = ratios.mean(axis=1)                  # (1/(K-1)) sum log-ratios
    inv = np.maximum(inv, EPS)
    return 1.0 / inv


def local_pr(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-point local participation ratio over the (self + K)-neighborhood.

    PR = (sum eig)^2 / sum(eig^2), computed from the neighborhood Gram matrix
    (scale-free, so Gram eigenvalues = covariance eigenvalues up to a constant).
    """
    n = X.shape[0]
    pr = np.empty(n, dtype=np.float64)
    for i in range(n):
        N = X[idx[i]]                          # (K+1, d), includes self at col 0
        Nc = N - N.mean(axis=0, keepdims=True)
        G = Nc @ Nc.T                          # (K+1, K+1)
        w = np.linalg.eigvalsh(G)
        w = np.clip(w, 0.0, None)
        s1 = w.sum()
        s2 = (w * w).sum()
        pr[i] = (s1 * s1) / (s2 + EPS)
    return pr


def _agg(vals: np.ndarray) -> tuple[float, float]:
    """Return (arithmetic_mean, geometric_mean) of strictly-positive vals."""
    v = np.maximum(vals, EPS)
    return float(v.mean()), float(np.exp(np.log(v).mean()))


def metric_stats(values: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Correct-vs-incorrect aggregation comparison for one per-point metric.

    Gap convention: incorrect - correct (positive => incorrect higher dim,
    i.e. the asymmetric-collapse direction of F-4)."""
    v = np.maximum(values, y.astype(np.float64) * 0.0 + values, )  # no-op guard
    v = np.maximum(values, EPS)
    cor = v[y]
    inc = v[~y]

    arith_c, geom_c = _agg(cor)
    arith_i, geom_i = _agg(inc)
    gap_arith = arith_i - arith_c
    gap_geom = geom_i - geom_c

    # Standardized effect size on the log scale (Cohen's d).
    lc, li = np.log(cor), np.log(inc)
    nc, ni = len(lc), len(li)
    pooled_sd = np.sqrt(
        ((nc - 1) * lc.var(ddof=1) + (ni - 1) * li.var(ddof=1)) / max(nc + ni - 2, 1)
    )
    cohen_d_log = float((li.mean() - lc.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")

    # Nonparametric bootstrap of both gaps (resample within each group).
    boot_arith = np.empty(N_BOOT, dtype=np.float64)
    boot_geom = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        bc = cor[rng.integers(0, nc, nc)]
        bi = inc[rng.integers(0, ni, ni)]
        a_c, g_c = _agg(bc)
        a_i, g_i = _agg(bi)
        boot_arith[b] = a_i - a_c
        boot_geom[b] = g_i - g_c

    se_geom = float(boot_geom.std(ddof=1))
    se_arith = float(boot_arith.std(ddof=1))
    ci_geom = [float(np.percentile(boot_geom, 2.5)), float(np.percentile(boot_geom, 97.5))]
    ci_arith = [float(np.percentile(boot_arith, 2.5)), float(np.percentile(boot_arith, 97.5))]
    sigma_geom = float(abs(gap_geom) / se_geom) if se_geom > 0 else float("inf")
    sigma_arith = float(abs(gap_arith) / se_arith) if se_arith > 0 else float("inf")

    return {
        "arith_mean_correct": arith_c,
        "arith_mean_incorrect": arith_i,
        "geom_mean_correct": geom_c,
        "geom_mean_incorrect": geom_i,
        "gap_arith_incorrect_minus_correct": float(gap_arith),
        "gap_geom_incorrect_minus_correct": float(gap_geom),
        "cohen_d_log_scale": cohen_d_log,
        "bootstrap_se_arith": se_arith,
        "bootstrap_se_geom": se_geom,
        "bootstrap_ci95_arith": ci_arith,
        "bootstrap_ci95_geom": ci_geom,
        "gap_sigma_arith": sigma_arith,
        "gap_sigma_geom": sigma_geom,
        "survives_geom_1sigma": bool(sigma_geom >= 1.0),
        "ci95_geom_excludes_zero": bool(ci_geom[0] > 0.0 or ci_geom[1] < 0.0),
        # AUROC of the raw per-point metric predicting CORRECT (sign-revealing).
        "auroc_metric_predicts_correct": auroc(values, y),
    }


def main() -> int:
    if not PREFILL_CACHE.exists():
        print("MISSING_REGEN_INPUT", PREFILL_CACHE, file=sys.stderr)
        return 2
    y = np.load(PREFILL_CACHE)["correct"].astype(bool)

    X, source = load_activations()
    if X is None:
        print("MISSING_REGEN_INPUT", "no final-token or prefill activation cache",
              file=sys.stderr)
        return 2
    if X.shape[0] != y.shape[0]:
        print("MISSING_REGEN_INPUT",
              f"activation/label length mismatch {X.shape} vs {y.shape}",
              file=sys.stderr)
        return 2
    n = X.shape[0]
    if n <= K + 1:
        print("MISSING_REGEN_INPUT", f"too few points ({n}) for k={K}", file=sys.stderr)
        return 2

    # K nearest neighbors (self at col 0).
    nn = NearestNeighbors(n_neighbors=K + 1).fit(X)
    dist, idx = nn.kneighbors(X)

    lid = local_lid_mle(dist[:, 1:])           # (n,) k=32 LID per point
    pr = local_pr(X, idx)                      # (n,) local PR per point

    rng = np.random.default_rng(SEED)
    out = {
        "experiment": "P11-FE151",
        "description": "Geometric- vs arithmetic-mean re-aggregation of F-4 "
                       "asymmetric-collapse (final-token PR + L19 LID k=32).",
        "activation_source": source,
        "n_total": int(n),
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "k_neighbors": K,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "lid": metric_stats(lid, y, np.random.default_rng(SEED)),
        "pr": metric_stats(pr, y, np.random.default_rng(SEED + 1)),
    }

    # Headline verdict: does the F-4 collapse survive geometric-mean aggregation?
    out["verdict"] = {
        "lid_geom_survives_1sigma": out["lid"]["survives_geom_1sigma"],
        "pr_geom_survives_1sigma": out["pr"]["survives_geom_1sigma"],
        "f4_strengthens": bool(out["lid"]["survives_geom_1sigma"]
                               and out["pr"]["survives_geom_1sigma"]),
        "f4_aggregation_artifact": bool(not out["lid"]["survives_geom_1sigma"]
                                        and not out["pr"]["survives_geom_1sigma"]),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())