"""P11-FE117 — Participation-ratio breathing curve on pos+ctx-residualized
hidden states, four-model comparison (Qwen-2.5-1.5B, Qwen-2.5-7B, Phi-3-mini,
Llama-3.2-1B).

Tests whether F-1's "breathing is universal across architectures" survives when
the smooth positional basis (Song-Zhong positional spiral) is removed. For each
model we load the per-position breathing-curve NPZ (N samples x T positions x d),
form the two-way residual

    resid_{c,t} = X_{c,t} - pos_t - ctx_c + grand_mean

where pos_t is the per-position mean over samples (the positional basis P) and
ctx_c is the per-sample mean over positions, then recompute:

  (a) ScreeNOT-family rank of the positional basis P (Gavish-Donoho unknown-noise
      optimal hard threshold) — compared against Song-Zhong Table 1 (9-12),
  (b) PR(t) on resid_{c,t} (the de-positioned breathing curve),
  (c) the raw PR(t) curve (no subtraction) reproducing EXP-031/032/040.

F-1 universality survives only if the breathing SHAPE (normalized curvature /
peak-to-trough ratio) is preserved on residuals. If de-positioning flattens the
curve, breathing was driven by positional-spiral curvature, not content geometry.
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
BREATHING_DIR = ROOT / "pathway11_h100/breathing"
OUT_JSON = ROOT / "pathway11_h100/breathing_residual_pr/results.json"

# Per-position breathing-curve caches written by EXP-031/032/040. Each NPZ is
# expected to hold a 3D hidden-state tensor (N, T, d). We try a few likely keys.
MODELS = {
    "qwen_1_5b": BREATHING_DIR / "qwen15b_breathing.npz",
    "qwen_7b": BREATHING_DIR / "qwen7b_breathing.npz",
    "phi3_mini": BREATHING_DIR / "phi3mini_breathing.npz",
    "llama_3_2_1b": BREATHING_DIR / "llama32_1b_breathing.npz",
}
HIDDEN_KEYS = ("hidden", "resid", "states", "h", "hiddens", "activations")

SEED = 9999
SCREENOT_KMAX = 64  # upper bound on positional-basis rank


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _load_hidden(npz_path: Path):
    """Load a (N, T, d) hidden tensor from a breathing NPZ, key-agnostic."""
    blob = np.load(npz_path)
    for key in HIDDEN_KEYS:
        if key in blob.files:
            arr = blob[key]
            if arr.ndim == 3:
                return arr.astype(np.float64)
    # fall back to the first 3D array present
    for key in blob.files:
        arr = blob[key]
        if getattr(arr, "ndim", 0) == 3:
            return arr.astype(np.float64)
    return None


def _mp_median(beta: float) -> float:
    """Numerical median of the Marchenko-Pastur law with ratio beta in (0,1]."""
    beta = min(max(beta, 1e-6), 1.0)
    a = (1.0 - np.sqrt(beta)) ** 2
    b = (1.0 + np.sqrt(beta)) ** 2
    xs = np.linspace(a, b, 200000)
    dens = np.sqrt(np.clip((b - xs) * (xs - a), 0.0, None)) / (2.0 * np.pi * beta * xs)
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) * np.diff(xs) / 2.0)])
    cdf /= cdf[-1]
    return float(xs[np.searchsorted(cdf, 0.5)])


def screenot_rank(M: np.ndarray, k_max: int = SCREENOT_KMAX) -> int:
    """ScreeNOT-family rank estimate via the Gavish-Donoho unknown-noise optimal
    hard threshold (median-of-singular-values calibration). Counts singular
    values of M exceeding tau = omega(beta) * median(singular values)."""
    M = M - M.mean(axis=0, keepdims=True)
    n, p = M.shape
    if min(n, p) < 2:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 0]
    if sv.size == 0:
        return 0
    beta = min(n, p) / max(n, p)
    lam = np.sqrt(
        2.0 * (beta + 1.0)
        + (8.0 * beta) / ((beta + 1.0) + np.sqrt(beta ** 2 + 14.0 * beta + 1.0))
    )
    omega = lam / np.sqrt(_mp_median(beta))
    tau = omega * float(np.median(sv))
    rank = int((sv > tau).sum())
    return int(min(rank, k_max))


def participation_ratio(cov: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2) on eigenvalues of a covariance."""
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    s1 = float(ev.sum())
    s2 = float((ev ** 2).sum())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def pr_curve(H: np.ndarray) -> np.ndarray:
    """PR(t) across token positions: covariance over the sample axis per t."""
    n, T, d = H.shape
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        Xt = H[:, t, :]
        Xt = Xt - Xt.mean(axis=0, keepdims=True)
        cov = (Xt.T @ Xt) / max(n - 1, 1)
        out[t] = participation_ratio(cov)
    return out


def residualize(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-way (position + context) subtraction. Returns (resid, pos_basis)."""
    pos = H.mean(axis=0)           # (T, d) positional basis P
    ctx = H.mean(axis=1)           # (N, d) per-sample context mean
    grand = H.mean(axis=(0, 1))    # (d,)
    resid = H - pos[None, :, :] - ctx[:, None, :] + grand[None, None, :]
    return resid, pos


def _curve_stats(curve: np.ndarray) -> dict:
    c = np.asarray(curve, dtype=np.float64)
    peak = float(c.max())
    trough = float(c.min())
    return {
        "T": int(c.size),
        "peak_pr": peak,
        "trough_pr": trough,
        "mean_pr": float(c.mean()),
        "peak_to_trough": float(peak / trough) if trough > 0 else float("nan"),
        "argmax_pos": int(c.argmax()),
        # normalized breathing amplitude — the shape-preservation diagnostic
        "amplitude_norm": float((peak - trough) / peak) if peak > 0 else 0.0,
        "curve": [float(v) for v in c],
    }


def main() -> int:
    available = {name: p for name, p in MODELS.items() if p.exists()}
    if not available:
        print("MISSING_REGEN_INPUT", BREATHING_DIR, file=sys.stderr)
        return 2

    per_model = {}
    for name, path in MODELS.items():
        if name not in available:
            per_model[name] = {"status": "missing", "npz": str(path)}
            continue
        H = _load_hidden(path)
        if H is None:
            per_model[name] = {"status": "no_3d_array", "npz": str(path)}
            continue

        raw_curve = pr_curve(H)
        resid, pos_basis = residualize(H)
        resid_curve = pr_curve(resid)
        p_rank = screenot_rank(pos_basis)

        raw_stats = _curve_stats(raw_curve)
        resid_stats = _curve_stats(resid_curve)

        # Shape preservation: correlation of the two curves + amplitude ratio.
        if raw_curve.std() > 0 and resid_curve.std() > 0:
            shape_corr = float(np.corrcoef(raw_curve, resid_curve)[0, 1])
        else:
            shape_corr = float("nan")
        amp_ratio = (
            resid_stats["amplitude_norm"] / raw_stats["amplitude_norm"]
            if raw_stats["amplitude_norm"] > 0
            else float("nan")
        )

        per_model[name] = {
            "status": "ok",
            "npz": str(path),
            "n_samples": int(H.shape[0]),
            "n_positions": int(H.shape[1]),
            "d_model": int(H.shape[2]),
            "screenot_rank_P": int(p_rank),
            "raw_pr_curve": raw_stats,
            "resid_pr_curve": resid_stats,
            "shape_corr_raw_vs_resid": shape_corr,
            "amplitude_ratio_resid_over_raw": amp_ratio,
            # breathing survives de-positioning if residual amplitude is a
            # meaningful fraction of raw amplitude AND shapes still correlate
            "breathing_survives_residual": bool(
                np.isfinite(shape_corr)
                and shape_corr > 0.5
                and np.isfinite(amp_ratio)
                and amp_ratio > 0.5
            ),
        }

    ok = {k: v for k, v in per_model.items() if v.get("status") == "ok"}
    ranks = {k: v["screenot_rank_P"] for k, v in ok.items()}
    surviving = [k for k, v in ok.items() if v["breathing_survives_residual"]]

    out = {
        "experiment": "P11-FE117",
        "description": "PR breathing curve on pos+ctx-residualized hidden states, "
        "four-model comparison",
        "seed": SEED,
        "screenot_kmax": SCREENOT_KMAX,
        "models_available": sorted(available.keys()),
        "per_model": per_model,
        "screenot_rank_P_by_model": ranks,
        "songzhong_table1_rank_range": [9, 12],
        "screenot_rank_in_songzhong_range": {
            k: bool(9 <= r <= 12) for k, r in ranks.items()
        },
        "n_models_breathing_survives": len(surviving),
        "n_models_ok": len(ok),
        # F-1 verdict: universality survives only if breathing shape is preserved
        # on residuals in (essentially) all evaluated models
        "f1_universality_survives": bool(len(ok) > 0 and len(surviving) == len(ok)),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())