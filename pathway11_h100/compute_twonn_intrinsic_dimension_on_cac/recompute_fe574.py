"""P11-FE574 — TwoNN intrinsic-dimension robustness check for breathing / asymmetric collapse.

2510.01105 uses TwoNN (Facco et al. 2017) as its primary intrinsic-dimension (ID)
estimator and frames the PR/PCA eigenvalue-based ID literature as one specific
(potentially biased) family. This experiment recomputes ID on the cached
pathway11_h100 L19 activations with TwoNN — a non-eigenvalue estimator — instead
of participation-ratio (PR), and compares against PR on the same point clouds.

Two questions:
  (1) Asymmetric collapse (F-4): is TwoNN-ID(correct) < TwoNN-ID(incorrect) at the
      prefill (and, if a final-token cache is found, the final) position, matching
      the PR-based direction?  If TwoNN flips the sign (correct shows *higher* ID),
      Refutation 1 fires.
  (2) Temporal breathing (F-1): if a per-token / temporal activation cache is
      present (3-D array of token trajectories), reproduce the per-token ID curve
      with TwoNN and report whether it tracks the PR curve.

The only guaranteed input is the 1.5B L19 prefill cache. Final-token and 7B and
per-token caches are auto-discovered by globbing pathway11_h100; whatever is found
is added to the report, whatever is missing is recorded as skipped. The script
fails (return 2) only if the guaranteed prefill cache is absent.
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
SEARCH_DIR = ROOT / "pathway11_h100"
OUT_JSON = ROOT / "pathway11_h100/twonn_id/results.json"

SEED = 9999
N_BOOT = 200
DISCARD_FRAC = 0.10  # drop largest-mu tail, standard TwoNN bias mitigation

# Candidate keys for a feature matrix / label vector inside an arbitrary NPZ.
FEATURE_KEYS = ("prefill", "final", "final_token", "hidden", "states", "X", "acts", "activations")
LABEL_KEYS = ("correct", "labels", "y")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def twonn_id(X: np.ndarray, discard_frac: float = DISCARD_FRAC) -> float:
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017).

    For each point, mu = r2/r1 (ratio of 2nd to 1st nearest-neighbour distance).
    The empirical CDF F(mu) satisfies -log(1 - F(mu)) = d * log(mu); the slope of
    a through-origin linear fit is the ID estimate. The largest-mu tail is dropped
    to mitigate curvature bias from finite-sample / non-constant density.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 10:
        return float("nan")
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    Dsort = np.sort(D, axis=1)
    r1 = Dsort[:, 0]
    r2 = Dsort[:, 1]
    valid = (r1 > 0) & np.isfinite(r1) & np.isfinite(r2)
    mu = r2[valid] / r1[valid]
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if len(mu) < 10:
        return float("nan")
    mu = np.sort(mu)
    keep = int(np.floor(len(mu) * (1.0 - discard_frac)))
    keep = max(keep, 10)
    mu = mu[:keep]
    N = len(mu)
    F = np.arange(1, N + 1, dtype=np.float64) / (N + 1)
    x = np.log(mu)
    yv = -np.log1p(-F)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * yv) / denom)


def participation_ratio(X: np.ndarray) -> float:
    """Eigenvalue-family ID proxy: PR = (sum λ)^2 / sum(λ^2) of the covariance."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    # eigenvalues of covariance via singular values (avoids forming 1536x1536)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / max(X.shape[0] - 1, 1)
    num = float(lam.sum()) ** 2
    den = float((lam ** 2).sum())
    if den <= 0:
        return float("nan")
    return num / den


def bootstrap_id_diff(X_corr: np.ndarray, X_inc: np.ndarray, n_boot: int, seed: int):
    """Bootstrap CI for TwoNN-ID(incorrect) - TwoNN-ID(correct)."""
    rng = np.random.default_rng(seed)
    nc, ni = X_corr.shape[0], X_inc.shape[0]
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ic = rng.integers(0, nc, size=nc)
        ii = rng.integers(0, ni, size=ni)
        dc = twonn_id(X_corr[ic])
        di = twonn_id(X_inc[ii])
        diffs[b] = di - dc
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return {"ci_lo": None, "ci_hi": None, "mean": None, "frac_positive": None}
    return {
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "mean": float(diffs.mean()),
        "frac_positive": float((diffs > 0).mean()),
    }


def class_id_report(X: np.ndarray, y: np.ndarray, name: str, seed: int) -> dict:
    """TwoNN + PR ID for all / correct / incorrect, plus the F-4 collapse verdict."""
    Xc = X[y]
    Xi = X[~y]
    twonn_all = twonn_id(X)
    twonn_c = twonn_id(Xc)
    twonn_i = twonn_id(Xi)
    pr_all = participation_ratio(X)
    pr_c = participation_ratio(Xc)
    pr_i = participation_ratio(Xi)
    boot = bootstrap_id_diff(Xc, Xi, N_BOOT, seed)

    twonn_dir = None
    if np.isfinite(twonn_c) and np.isfinite(twonn_i):
        twonn_dir = "correct_lower" if twonn_c < twonn_i else "correct_higher"
    pr_dir = None
    if np.isfinite(pr_c) and np.isfinite(pr_i):
        pr_dir = "correct_lower" if pr_c < pr_i else "correct_higher"

    # F-4 collapse direction is "correct_lower" under PR. Refutation 1 fires if
    # TwoNN reverses it while PR holds the canonical direction.
    refutation_1 = (
        twonn_dir is not None
        and pr_dir is not None
        and pr_dir == "correct_lower"
        and twonn_dir == "correct_higher"
    )

    return {
        "name": name,
        "n_total": int(X.shape[0]),
        "n_correct": int(Xc.shape[0]),
        "n_incorrect": int(Xi.shape[0]),
        "twonn_id_all": twonn_all,
        "twonn_id_correct": twonn_c,
        "twonn_id_incorrect": twonn_i,
        "twonn_collapse_direction": twonn_dir,
        "twonn_diff_inc_minus_corr": (
            float(twonn_i - twonn_c) if np.isfinite(twonn_c) and np.isfinite(twonn_i) else None
        ),
        "twonn_diff_bootstrap": boot,
        "pr_id_all": pr_all,
        "pr_id_correct": pr_c,
        "pr_id_incorrect": pr_i,
        "pr_collapse_direction": pr_dir,
        "estimators_agree_on_direction": (
            (twonn_dir == pr_dir) if (twonn_dir and pr_dir) else None
        ),
        "refutation_1_fires": bool(refutation_1),
    }


def _find_arr(blob, keys, ndim=None, length=None):
    """Return the first array in `blob` matching a candidate key / shape filter."""
    files = list(blob.files)
    for k in keys:
        if k in files:
            a = blob[k]
            if ndim is not None and a.ndim != ndim:
                continue
            if length is not None and a.shape[0] != length:
                continue
            return k, a
    return None, None


def load_features_labels(path: Path):
    """Best-effort: pull a 2-D feature matrix + bool label vector from an NPZ."""
    blob = np.load(path, allow_pickle=False)
    lk, y = _find_arr(blob, LABEL_KEYS)
    fk, X = _find_arr(blob, FEATURE_KEYS, ndim=2)
    if X is None:
        # fall back: largest 2-D float array
        best = None
        for k in blob.files:
            a = blob[k]
            if a.ndim == 2 and np.issubdtype(a.dtype, np.floating):
                if best is None or a.size > blob[best].size:
                    best = k
        if best is not None:
            fk, X = best, blob[best]
    if X is None or y is None:
        return None
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    if X.shape[0] != y.shape[0]:
        return None
    return {"feature_key": fk, "label_key": lk, "X": X, "y": y}


def temporal_breathing(path: Path, seed: int):
    """If an NPZ holds a 3-D (problems, tokens, dim) trajectory + labels, build the
    per-token TwoNN / PR breathing curves for correct vs incorrect."""
    blob = np.load(path, allow_pickle=False)
    lk, y = _find_arr(blob, LABEL_KEYS)
    traj = None
    tk = None
    for k in blob.files:
        a = blob[k]
        if a.ndim == 3 and np.issubdtype(a.dtype, np.floating):
            tk = k
            traj = a
            break
    if traj is None or y is None or traj.shape[0] != y.shape[0]:
        return None
    y = np.asarray(y).astype(bool)
    n_tok = traj.shape[1]
    curve = []
    for t in range(n_tok):
        Xt = np.asarray(traj[:, t, :], dtype=np.float64)
        Xc, Xi = Xt[y], Xt[~y]
        curve.append({
            "token_index": t,
            "twonn_id_correct": twonn_id(Xc),
            "twonn_id_incorrect": twonn_id(Xi),
            "pr_id_correct": participation_ratio(Xc),
            "pr_id_incorrect": participation_ratio(Xi),
        })
    # Does TwoNN track PR across tokens? Correlate the two correct-class curves.
    tw = np.array([c["twonn_id_correct"] for c in curve], dtype=np.float64)
    pr = np.array([c["pr_id_correct"] for c in curve], dtype=np.float64)
    m = np.isfinite(tw) & np.isfinite(pr)
    corr = (
        float(np.corrcoef(tw[m], pr[m])[0, 1])
        if m.sum() >= 3 and np.std(tw[m]) > 0 and np.std(pr[m]) > 0
        else None
    )
    return {
        "trajectory_key": tk,
        "n_tokens": int(n_tok),
        "twonn_pr_curve_correlation_correct": corr,
        "curve": curve,
    }


def discover_optional(primary: Path):
    """Glob pathway11_h100 for final-token / 7B / temporal caches, excluding the
    guaranteed primary prefill cache."""
    out = {"class_caches": [], "temporal_caches": []}
    patterns_2d = ["*final*.npz", "*m7b*.npz", "*7b*.npz", "*7B*.npz"]
    patterns_3d = ["*temporal*.npz", "*per_token*.npz", "*per-token*.npz", "*breathing*.npz", "*trajector*.npz"]
    seen = set()
    for pat in patterns_2d:
        for p in sorted(SEARCH_DIR.rglob(pat)):
            if p.resolve() == primary.resolve() or p in seen:
                continue
            seen.add(p)
            out["class_caches"].append(p)
    for pat in patterns_3d:
        for p in sorted(SEARCH_DIR.rglob(pat)):
            if p in seen:
                continue
            seen.add(p)
            out["temporal_caches"].append(p)
    return out


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), f"unexpected shapes {X.shape} {y.shape}"

    results = {
        "experiment": "P11-FE574",
        "estimator": "TwoNN (Facco 2017)",
        "discard_frac": DISCARD_FRAC,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "reference_paper": "2510.01105",
        "positions": {},
        "temporal": {},
        "skipped": [],
    }

    # ---- Primary: 1.5B L19 prefill, the guaranteed input. -------------------
    results["positions"]["m15b_prefill"] = class_id_report(X, y, "m15b_prefill", SEED)
    # Sanity: DoM direction should still separate classes (orientation check only).
    dom = X[y].mean(0) - X[~y].mean(0)
    results["positions"]["m15b_prefill"]["dom_inproject_auroc"] = float(auroc(X @ dom, y))

    # ---- Optional: final-token / 7B class caches. ---------------------------
    found = discover_optional(CACHE)
    for path in found["class_caches"]:
        rel = str(path.relative_to(ROOT))
        try:
            loaded = load_features_labels(path)
        except Exception as e:  # corrupt / unreadable NPZ
            results["skipped"].append({"path": rel, "reason": f"load_error:{type(e).__name__}"})
            continue
        if loaded is None:
            results["skipped"].append({"path": rel, "reason": "no_feature_label_pair"})
            continue
        rep = class_id_report(loaded["X"], loaded["y"], rel, SEED)
        rep["source_path"] = rel
        rep["feature_key"] = loaded["feature_key"]
        rep["label_key"] = loaded["label_key"]
        results["positions"][rel] = rep

    # ---- Optional: temporal breathing curves. -------------------------------
    for path in found["temporal_caches"]:
        rel = str(path.relative_to(ROOT))
        try:
            curve = temporal_breathing(path, SEED)
        except Exception as e:
            results["skipped"].append({"path": rel, "reason": f"temporal_error:{type(e).__name__}"})
            continue
        if curve is None:
            results["skipped"].append({"path": rel, "reason": "no_3d_trajectory_or_labels"})
            continue
        results["temporal"][rel] = curve

    if not results["temporal"]:
        results["skipped"].append({
            "path": "(temporal breathing)",
            "reason": "no per-token trajectory cache found; breathing curve not reproduced",
        })

    # ---- Top-line verdict across all positions that have a class comparison. -
    refutations = [p for p in results["positions"].values() if p.get("refutation_1_fires")]
    agree = [
        p for p in results["positions"].values()
        if p.get("estimators_agree_on_direction") is True
    ]
    results["summary"] = {
        "positions_evaluated": list(results["positions"].keys()),
        "n_positions": len(results["positions"]),
        "any_refutation_1": bool(refutations),
        "refutation_1_positions": [p["name"] for p in refutations],
        "twonn_pr_direction_agreement_count": len(agree),
        "verdict": (
            "REFUTATION_1: TwoNN reverses F-4 collapse direction where PR holds it"
            if refutations
            else "TwoNN reproduces PR collapse direction (F-4 robust to estimator family)"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())