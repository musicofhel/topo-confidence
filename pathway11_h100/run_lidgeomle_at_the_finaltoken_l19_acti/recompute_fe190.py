"""P11-FE190 — LID-GeoMLE at the final-token L19 activation vs final-token DoM.

Yin et al. measure a per-token Local Intrinsic Dimension (LID) at the *last
generated token* and obtain 0.746 AUROC for correctness — higher than this
project's final-token DoM AUROC of 0.7186 (F-3). This experiment recomputes a
GeoMLE-style per-point LID estimate (Levina–Bickel MLE over a range of k,
bootstrap-averaged, polynomial-extrapolated to zero scale — Gomtsyan et al.)
on the cached final-token L19 activations and scores it against ground-truth
correctness.

Logic: F-3 infers the final token is "information-poor" because a linear DoM
probe reads only 0.7186 there. If a non-linear, geometry-based LID estimate
reaches >= 0.74, that inference is wrong: the final token is information-rich
along an axis the linear probe cannot read, and the prefill/final orthogonality
(cos = 0.046) is a coordinate-frame artefact rather than a signal-orthogonality
finding.

CPU only, cached NPZ only. Writes pathway11_h100/lid_geomle_final/results.json.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/lid_geomle_final/results.json"

# Candidate locations / keys for the cached final-token L19 activations.
# The schema documents only the prefill cache, so we probe a handful of
# plausible final-token caches and degrade gracefully if none exists.
FINAL_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_last_token.npz",
    CACHE,  # last resort: a 'final'/'last' key may live in the prefill cache
]
FINAL_KEY_CANDIDATES = (
    "final", "final_token", "final_hidden", "finaltoken",
    "last_token", "last", "hidden_final", "final_l19",
)

N_FOLDS = 5
SEED = 9999

# GeoMLE hyperparameters.
K1 = 15            # smallest neighborhood size
K2 = 45            # largest neighborhood size
N_BOOTSTRAP = 20   # bootstrap resamples (variance reduction, à la GeoMLE)
POLY_DEG = 2       # polynomial extrapolation degree (scale -> 0)
EPS = 1e-12

# F-3 reference numbers (1024-tok canonical).
FINAL_DOM_REF = 0.7186
LID_THRESHOLD = 0.74  # Yin et al. last-token LID AUROC


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def load_final_activations() -> "np.ndarray | None":
    """Return (500, 1536) final-token activations, or None if unavailable."""
    for path in FINAL_CANDIDATES:
        if not path.exists():
            continue
        try:
            blob = np.load(path)
        except Exception:
            continue
        keys = list(blob.files)
        # Prefer an explicitly-named final-token key.
        for key in FINAL_KEY_CANDIDATES:
            if key in keys:
                arr = blob[key]
                if arr.ndim == 2 and arr.shape == (500, 1536):
                    return arr.astype(np.float64)
        # Otherwise accept any (500, 1536) array that isn't 'prefill'.
        for key in keys:
            if key == "prefill":
                continue
            arr = blob[key]
            if getattr(arr, "ndim", 0) == 2 and arr.shape == (500, 1536):
                return arr.astype(np.float64)
    return None


def mle_per_point(dist: np.ndarray) -> np.ndarray:
    """Levina–Bickel MLE LID for each point over its k-NN distances.

    dist: (n, k) sorted ascending neighbor distances (self excluded).
    Returns (n,) LID estimate using all k neighbors.
    """
    k = dist.shape[1]
    logr = np.log(np.clip(dist, EPS, None))
    # sum_{j=1}^{k-1} ln(r_k / r_j) = (k-1)*ln r_k - sum_{j<k} ln r_j
    s = (k - 1) * logr[:, k - 1] - logr[:, : k - 1].sum(axis=1)
    s = np.where(s > EPS, s, np.nan)
    return (k - 1) / s


def geomle_lid(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """GeoMLE-style per-point LID.

    For a range of neighborhood sizes k, bootstrap-average the MLE estimate and
    the mean neighbor distance, then fit a degree-POLY_DEG polynomial of
    estimate vs scale and extrapolate to zero scale (the intercept).
    """
    n = X.shape[0]
    ks = list(range(K1, K2 + 1, 2))
    # Accumulators: per-point, per-k sums across bootstraps.
    mle_acc = np.zeros((n, len(ks)), dtype=np.float64)
    dist_acc = np.zeros((n, len(ks)), dtype=np.float64)
    cnt = np.zeros((n, len(ks)), dtype=np.float64)

    for _ in range(N_BOOTSTRAP):
        boot = rng.integers(0, n, size=n)
        base = X[boot]
        nn = NearestNeighbors(n_neighbors=K2 + 1, algorithm="auto")
        nn.fit(base)
        d, _ = nn.kneighbors(X)  # (n, K2+1), includes a possible self/zero
        # Drop the nearest column (self or duplicate) -> (n, K2)
        d = d[:, 1:]
        d = np.clip(d, EPS, None)
        for ci, k in enumerate(ks):
            dk = d[:, :k]
            m = mle_per_point(dk)
            meand = dk.mean(axis=1)
            valid = np.isfinite(m)
            mle_acc[valid, ci] += m[valid]
            dist_acc[valid, ci] += meand[valid]
            cnt[valid, ci] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        mle_mean = mle_acc / np.where(cnt > 0, cnt, np.nan)
        dist_mean = dist_acc / np.where(cnt > 0, cnt, np.nan)

    lid = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        ymask = np.isfinite(mle_mean[i]) & np.isfinite(dist_mean[i])
        if ymask.sum() < POLY_DEG + 1:
            continue
        xs = dist_mean[i, ymask]
        ys = mle_mean[i, ymask]
        try:
            coeffs = np.polyfit(xs, ys, POLY_DEG)
            lid[i] = coeffs[-1]  # intercept == extrapolation to scale 0
        except Exception:
            lid[i] = np.nanmean(ys)
    # Fallback for any unresolved points: mean MLE over k.
    nanmask = ~np.isfinite(lid)
    if nanmask.any():
        lid[nanmask] = np.nanmean(mle_mean[nanmask], axis=1)
    return lid


def oof_dom_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Out-of-fold linear DoM AUROC (matches the F-3 final-token probe)."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    y = np.load(CACHE)["correct"].astype(bool)
    assert y.shape == (500,)

    X_final = load_final_activations()
    if X_final is None:
        print("MISSING_REGEN_INPUT final-token L19 activations "
              "(tried: %s)" % ", ".join(str(p) for p in FINAL_CANDIDATES),
              file=sys.stderr)
        return 2

    rng = np.random.default_rng(SEED)
    lid = geomle_lid(X_final, rng)

    # LID orientation w.r.t. correctness is a priori unknown — report both.
    auroc_lid_raw = auroc(lid, y)
    auroc_lid_flip = auroc(-lid, y)
    auroc_lid = max(auroc_lid_raw, auroc_lid_flip)
    orientation = "low_lid_correct" if auroc_lid_flip >= auroc_lid_raw else "high_lid_correct"

    auroc_dom_final = oof_dom_auroc(X_final, y)

    f3_overturned = bool(auroc_lid >= LID_THRESHOLD)

    out = {
        "experiment": "P11-FE190",
        "description": "GeoMLE LID at final-token L19 vs final-token DoM (F-3)",
        "n": int(len(y)),
        "n_correct": int(y.sum()),
        "geomle_params": {
            "k1": K1, "k2": K2, "n_bootstrap": N_BOOTSTRAP,
            "poly_degree": POLY_DEG, "seed": SEED,
        },
        "auroc_lid_geomle": float(auroc_lid),
        "auroc_lid_geomle_raw": float(auroc_lid_raw),
        "auroc_lid_geomle_flipped": float(auroc_lid_flip),
        "lid_orientation": orientation,
        "auroc_final_dom_oof_recomputed": float(auroc_dom_final),
        "auroc_final_dom_reference_f3": FINAL_DOM_REF,
        "lid_minus_dom": float(auroc_lid - auroc_dom_final),
        "lid_threshold": LID_THRESHOLD,
        "f3_information_poor_overturned": f3_overturned,
        "verdict": (
            "LID-GeoMLE >= 0.74: final-token is information-rich along a "
            "non-linear axis; F-3 'information-poor' inference is wrong and "
            "prefill/final orthogonality (cos=0.046) is plausibly a "
            "coordinate-frame artefact."
            if f3_overturned else
            "LID-GeoMLE < 0.74: no non-linear geometric signal beyond the "
            "linear DoM probe; F-3 'final-token information-poor' inference "
            "holds and prefill/final orthogonality stands as a signal finding."
        ),
        "lid_stats": {
            "mean": float(np.nanmean(lid)),
            "std": float(np.nanstd(lid)),
            "min": float(np.nanmin(lid)),
            "max": float(np.nanmax(lid)),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT_JSON)
    print("auroc_lid_geomle=%.4f  auroc_final_dom_oof=%.4f  overturned=%s"
          % (auroc_lid, auroc_dom_final, f3_overturned))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())