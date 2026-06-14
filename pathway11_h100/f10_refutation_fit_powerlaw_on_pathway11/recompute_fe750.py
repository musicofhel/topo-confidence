"""P11-FE750 — F-10 refutation via power-law α vs Marchenko–Pastur Gaussian null.

The external paper claims (a) residual-stream singular-value spectra follow a
power law s_k ∝ k^{-α} with R²>0.85, and (b) a task-dependent α that separates
reasoning from factual prompts at p<10⁻⁵. F-10 ("PH on trained residual streams
sits at the Gaussian null") implies the residual stream is spectrally
Gaussian-equivalent — which a stable, task-dependent power-law α would directly
contradict, because a random T×d Gaussian matrix has Marchenko–Pastur singular
values, NOT a stable power law.

This script fits a per-question power-law α to every layer of the cached
pathway11 residual streams (1.5B and, when present, 7B; 500 problems × 28 layers)
and to a Marchenko–Pastur-matched Gaussian baseline drawn at the SAME T×d aspect
ratio and total variance per question (the per-question control). For every layer
it runs a Welch two-sample t-test of the real-α distribution against the
Gaussian-baseline-α distribution and reports the fraction of layers separating at
p<0.001. If real and Gaussian-baseline α distributions differ at p<0.001 across
most layers, F-10 should be narrowed to "PH cannot detect non-Gaussian structure
that α can."

Expected per-question layerwise caches (regenerate if MISSING_REGEN_INPUT):
  pathway11_h100/data/layerwise_resid/problem_NNN.npz      (1.5B)
  pathway11_h100/data/layerwise_resid_7b/problem_NNN.npz   (7B, optional)
each with key "resid": (n_layers, T_q, d) float32 — per-layer, per-token residual
stream for problem NNN (T_q varies per problem; d=1536 for 1.5B, 3584 for 7B).
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

try:
    from scipy.stats import ttest_ind
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
LAYERWISE_DIRS = {
    "1.5B": ROOT / "pathway11_h100/data/layerwise_resid",
    "7B": ROOT / "pathway11_h100/data/layerwise_resid_7b",
}
OUT_JSON = ROOT / "pathway11_h100/powerlaw_alpha_null/results.json"

SEED = 9999
P_THRESH = 1e-3
KMIN = 2            # skip the leading singular value (bulk/scale outlier)
R2_FLOOR = 0.0      # keep all fits; report distribution of R²


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _f(x):
    """JSON-safe float: NaN/inf -> None."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def fit_alpha(svals: np.ndarray):
    """Power-law exponent of the singular-value spectrum: s_k ∝ k^{-α}.

    OLS of log(s_k) on log(k) over ranks KMIN..kmax, where kmax is the midpoint
    of the spectrum (the power-law body, before the MP/noise floor). Returns
    (alpha, r2); alpha = -slope.
    """
    s = np.sort(svals[np.isfinite(svals)])[::-1]
    s = s[s > 0]
    r = s.size
    if r < KMIN + 5:
        return np.nan, np.nan
    kmax = max(KMIN + 5, r // 2)
    kmax = min(kmax, r)
    k = np.arange(KMIN, kmax + 1, dtype=np.float64)
    logs = np.log(s[KMIN - 1:kmax])
    logk = np.log(k)
    A = np.column_stack([logk, np.ones_like(logk)])
    coef, _, _, _ = np.linalg.lstsq(A, logs, rcond=None)
    pred = A @ coef
    ss_res = float(((logs - pred) ** 2).sum())
    ss_tot = float(((logs - logs.mean()) ** 2).sum())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(-coef[0]), float(r2)


def singular_values(X: np.ndarray) -> np.ndarray:
    """Centered (per-feature) economy singular values of a (T, d) matrix."""
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        return np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        # Fall back to eigvals of the smaller Gram matrix.
        T, d = Xc.shape
        G = (Xc @ Xc.T) if T <= d else (Xc.T @ Xc)
        ev = np.linalg.eigvalsh(G)
        return np.sqrt(np.clip(ev[::-1], 0.0, None))


def gaussian_baseline_svals(T: int, d: int, scale: float, rng) -> np.ndarray:
    """MP-matched Gaussian: same T×d aspect ratio and per-entry scale."""
    G = rng.standard_normal((T, d)).astype(np.float64) * scale
    return singular_values(G)


def welch(real: np.ndarray, base: np.ndarray):
    real = real[np.isfinite(real)]
    base = base[np.isfinite(base)]
    if real.size < 3 or base.size < 3:
        return np.nan, np.nan
    if _HAVE_SCIPY:
        t, p = ttest_ind(real, base, equal_var=False)
        return float(t), float(p)
    # Manual Welch t + normal-approx p (scipy absent).
    m1, m2 = real.mean(), base.mean()
    v1, v2 = real.var(ddof=1), base.var(ddof=1)
    n1, n2 = real.size, base.size
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return np.nan, np.nan
    t = (m1 - m2) / se
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))
    return float(t), float(p)


def list_problem_files(d: Path):
    return sorted(d.glob("problem_*.npz")) if d.exists() else []


def analyze_model(name: str, prob_files, correct: np.ndarray | None, rng) -> dict | None:
    if not prob_files:
        return None

    real_alpha = {}   # layer -> list[float]
    gauss_alpha = {}
    real_r2 = {}
    n_layers = None
    used = 0

    for pf in prob_files:
        try:
            blob = np.load(pf)
        except Exception:
            continue
        if "resid" not in blob.files:
            continue
        resid = blob["resid"]
        if resid.ndim != 3:
            continue
        L = resid.shape[0]
        if n_layers is None:
            n_layers = L
            for ell in range(L):
                real_alpha[ell] = []
                gauss_alpha[ell] = []
                real_r2[ell] = []
        if L != n_layers:
            continue
        for ell in range(L):
            X = resid[ell].astype(np.float64)
            if X.shape[0] < KMIN + 6:
                continue
            sv = singular_values(X)
            a, r2 = fit_alpha(sv)
            real_alpha[ell].append(a)
            real_r2[ell].append(r2)
            Xc = X - X.mean(axis=0, keepdims=True)
            scale = float(Xc.std()) if Xc.size else 0.0
            gsv = gaussian_baseline_svals(X.shape[0], X.shape[1], scale, rng)
            ga, _ = fit_alpha(gsv)
            gauss_alpha[ell].append(ga)
        used += 1

    if n_layers is None or used == 0:
        return None

    per_layer = []
    n_sig = 0
    all_real_r2 = []
    for ell in range(n_layers):
        ra = np.asarray(real_alpha[ell], dtype=np.float64)
        ga = np.asarray(gauss_alpha[ell], dtype=np.float64)
        r2a = np.asarray(real_r2[ell], dtype=np.float64)
        all_real_r2.extend([v for v in r2a if np.isfinite(v)])
        t, p = welch(ra, ga)
        sig = bool(np.isfinite(p) and p < P_THRESH)
        if sig:
            n_sig += 1
        per_layer.append({
            "layer": ell,
            "n": int(np.isfinite(ra).sum()),
            "mean_real_alpha": _f(np.nanmean(ra)) if ra.size else None,
            "mean_gauss_alpha": _f(np.nanmean(ga)) if ga.size else None,
            "mean_real_r2": _f(np.nanmean(r2a)) if r2a.size else None,
            "welch_t": _f(t),
            "welch_p": _f(p),
            "sig_p001": sig,
        })

    all_real_r2 = np.asarray(all_real_r2, dtype=np.float64)
    return {
        "n_problems": int(used),
        "n_layers": int(n_layers),
        "n_layers_sig_p001": int(n_sig),
        "frac_layers_sig_p001": _f(n_sig / n_layers),
        "mean_real_r2": _f(all_real_r2.mean()) if all_real_r2.size else None,
        "frac_real_r2_above_0.85": _f((all_real_r2 > 0.85).mean()) if all_real_r2.size else None,
        "per_layer": per_layer,
        "correct_auroc_check": (
            _f(auroc(np.asarray([np.nanmean(real_alpha[e]) for e in range(n_layers)]),
                     np.ones(n_layers, dtype=bool)))
            if correct is not None else None
        ),
    }


def main() -> int:
    dirs = {k: v for k, v in LAYERWISE_DIRS.items()}
    files = {k: list_problem_files(v) for k, v in dirs.items()}

    if not any(files.values()):
        missing = ", ".join(str(v) for v in dirs.values())
        print("MISSING_REGEN_INPUT", missing, file=sys.stderr)
        return 2

    correct = None
    if CACHE.exists():
        try:
            correct = np.load(CACHE)["correct"].astype(bool)
        except Exception:
            correct = None

    rng = np.random.default_rng(SEED)
    models = {}
    for name in ("1.5B", "7B"):
        res = analyze_model(name, files.get(name, []), correct, rng)
        if res is not None:
            models[name] = res

    if not models:
        print("MISSING_REGEN_INPUT no usable resid arrays", file=sys.stderr)
        return 2

    tot_layers = sum(m["n_layers"] for m in models.values())
    tot_sig = sum(m["n_layers_sig_p001"] for m in models.values())

    out = {
        "experiment": "P11-FE750",
        "premise": "F-10 refutation: power-law alpha vs Marchenko-Pastur Gaussian null",
        "p_threshold": P_THRESH,
        "kmin": KMIN,
        "seed": SEED,
        "models": models,
        "headline": {
            "total_layers": int(tot_layers),
            "total_layers_sig_p001": int(tot_sig),
            "frac_layers_sig_p001": _f(tot_sig / tot_layers) if tot_layers else None,
            "interpretation": (
                "If frac_layers_sig_p001 is high, real-alpha separates from the "
                "Gaussian/MP null -> F-10 narrows to 'PH cannot detect non-Gaussian "
                "structure that alpha can.' If near zero, residual spectra are "
                "Gaussian-equivalent and the paper's power-law/task-alpha claim fails."
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())