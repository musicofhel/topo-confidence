"""P11-FE51 — Heat / Silhouette PD-vectorization of Pathway-9 PH features vs covariance null.

Re-runs the EXP-026 Pathway-9 persistent-homology pipeline, but replaces the raw
H0/H1 summary statistics with giotto-tda-style *vectorized* persistence diagrams
(Heat kernel + power-weighted Silhouette), under an η=0.01 lifespan filter and a
normalized weight filtration. For each representation we measure OOF AUROC against
correctness and compare it to a matched-covariance Gaussian null (per-problem mean
and covariance preserved, point cloud re-sampled). The paper's headline 0.82–0.89
correlations come from Heat-distance over vectorized PDs, not raw summaries; if the
discretized features lift past the Gaussian null while the raw summary statistics do
not, then F-10's "PH = covariance" verdict was vectorization-specific.

Constraints note: giotto-tda / ripser are disallowed (no extra deps), so the
persistence diagram is the *exact* single-linkage H0 diagram computed from the
Euclidean MST (births = 0, deaths = MST edge weights) via scipy. H1 needs a Rips
backend and is out of scope here; H0 lifespans are the component whose distribution
is most directly covariance-determined, so this is a faithful test of the F-10 claim
for the part of the diagram we can compute without a TDA library.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
# Candidate locations / keys for the cached Pathway-9 L19 per-problem point clouds.
PATHWAY9_CANDIDATES = [
    ROOT / "pathway9/cache/pathway9_l19_pointclouds.npz",
    ROOT / "pathway9/cache/pathway9_l19_activations.npz",
    ROOT / "pathway9/cache/coe_l19_clouds.npz",
    ROOT / "pathway11_h100/pathway9_ph/cache/l19_pointclouds.npz",
]
CLOUD_KEYS = ("point_clouds", "clouds", "activations", "hidden", "X")
LABEL_KEYS = ("correct", "labels", "y")
SEQLEN_KEYS = ("seq_len", "lengths", "n_tokens")
MAIN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/ph_vectorization/results.json"

SEED = 9999
N_FOLDS = 5
N_NULL = 20
MAX_TOKENS = 200       # deterministic per-problem subsample for tractability
ETA = 0.01             # lifespan filter: drop points with lifespan < ETA * max
RES = 64               # vectorization grid resolution
SILH_P = 1.0           # silhouette lifespan weighting power
HEAT_SIGMA_FRAC = 0.05 # heat-kernel bandwidth as a fraction of global scale


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_pathway9():
    """Return (clouds, correct) where clouds is a list of (n_i, d) float64 arrays.

    Tries several candidate caches/keys; returns None if nothing usable is found.
    """
    for path in PATHWAY9_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path, allow_pickle=True)
        keys = set(blob.files)

        cloud_key = next((k for k in CLOUD_KEYS if k in keys), None)
        if cloud_key is None:
            continue
        raw = blob[cloud_key]

        clouds = None
        if raw.dtype == object:
            clouds = [np.asarray(c, dtype=np.float64) for c in raw]
        elif raw.ndim == 3:
            seq_key = next((k for k in SEQLEN_KEYS if k in keys), None)
            if seq_key is not None:
                lens = blob[seq_key].astype(int)
                clouds = [np.asarray(raw[i, : int(lens[i])], dtype=np.float64)
                          for i in range(raw.shape[0])]
            else:
                clouds = [np.asarray(raw[i], dtype=np.float64)
                          for i in range(raw.shape[0])]
        if clouds is None:
            continue

        label_key = next((k for k in LABEL_KEYS if k in keys), None)
        if label_key is not None:
            correct = blob[label_key].astype(bool)
        elif MAIN_CACHE.exists():
            correct = np.load(MAIN_CACHE)["correct"].astype(bool)
        else:
            continue

        if len(clouds) != len(correct):
            continue
        return clouds, correct
    return None


def subsample(X: np.ndarray, idx: int) -> np.ndarray:
    n = X.shape[0]
    if n <= MAX_TOKENS:
        return X
    rng = np.random.default_rng(SEED + idx)
    pick = rng.choice(n, size=MAX_TOKENS, replace=False)
    pick.sort()
    return X[pick]


def h0_deaths(X: np.ndarray) -> np.ndarray:
    """Exact single-linkage H0 persistence deaths (births are all 0)."""
    if X.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D)
    deaths = np.asarray(mst[mst.nonzero()]).ravel().astype(np.float64)
    return deaths


def filter_deaths(deaths: np.ndarray) -> np.ndarray:
    """η lifespan filter (births=0 so lifespan == death)."""
    if deaths.size == 0:
        return deaths
    thresh = ETA * float(deaths.max())
    return deaths[deaths >= thresh]


def matched_gaussian(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Re-sample a point cloud with the same per-problem mean and covariance."""
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    G = rng.standard_normal((n, n))
    return mu + (G @ Xc) / np.sqrt(max(n - 1, 1))


def vectorize(deaths: np.ndarray, grid: np.ndarray, sigma: float):
    """Return (heat_vec, silhouette_vec) over the shared grid; normalized weights."""
    heat = np.zeros(grid.shape[0], dtype=np.float64)
    silh = np.zeros(grid.shape[0], dtype=np.float64)
    if deaths.size == 0:
        return heat, silh
    life = deaths  # births == 0
    w = life / max(float(life.sum()), 1e-12)  # normalized weight filtration

    # Heat kernel: weighted Gaussian bumps centered on each death.
    for d_i, w_i in zip(deaths, w):
        heat += w_i * np.exp(-0.5 * ((grid - d_i) / sigma) ** 2)

    # Power-weighted silhouette: tents peak at d/2 with height d/2.
    sw = life ** SILH_P
    sw_sum = max(float(sw.sum()), 1e-12)
    for d_i, sw_i in zip(deaths, sw):
        m = 0.5 * d_i
        h = 0.5 * d_i
        silh += sw_i * np.maximum(0.0, h - np.abs(grid - m))
    silh /= sw_sum
    return heat, silh


def raw_summary(deaths: np.ndarray) -> np.ndarray:
    """EXP-026-style raw H0 summary statistics."""
    if deaths.size == 0:
        return np.zeros(6, dtype=np.float64)
    return np.array([
        deaths.size,
        deaths.sum(),
        deaths.mean(),
        deaths.std(),
        deaths.max(),
        np.median(deaths),
    ], dtype=np.float64)


def featurize(deaths_list, grid, sigma):
    n = len(deaths_list)
    raw = np.zeros((n, 6), dtype=np.float64)
    heat = np.zeros((n, RES), dtype=np.float64)
    silh = np.zeros((n, RES), dtype=np.float64)
    for i, deaths in enumerate(deaths_list):
        raw[i] = raw_summary(deaths)
        h, s = vectorize(deaths, grid, sigma)
        heat[i] = h
        silh[i] = s
    return {"raw": raw, "heat": heat, "silhouette": silh}


def oof_auroc(feat: np.ndarray, y: np.ndarray) -> float:
    if not np.isfinite(feat).all() or np.allclose(feat.std(axis=0), 0.0):
        feat = np.nan_to_num(feat)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(feat, y):
        scaler = StandardScaler().fit(feat[tr])
        Xtr = scaler.transform(feat[tr])
        Xte = scaler.transform(feat[te])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.decision_function(Xte)
    return auroc(oof, y)


def main() -> int:
    loaded = load_pathway9()
    if loaded is None:
        print("MISSING_REGEN_INPUT", "pathway9 L19 point-cloud cache",
              file=sys.stderr)
        return 2
    clouds, correct = loaded

    # Deterministic subsample + real persistence diagrams.
    clouds = [subsample(c, i) for i, c in enumerate(clouds)]
    real_deaths = [filter_deaths(h0_deaths(c)) for c in clouds]

    all_deaths = np.concatenate([d for d in real_deaths if d.size]) \
        if any(d.size for d in real_deaths) else np.array([1.0])
    scale = float(np.percentile(all_deaths, 99.0)) or 1.0
    grid = np.linspace(0.0, scale, RES)
    sigma = HEAT_SIGMA_FRAC * scale

    real_feats = featurize(real_deaths, grid, sigma)
    real_auroc = {k: oof_auroc(v, correct) for k, v in real_feats.items()}

    # Matched-covariance Gaussian null.
    null_auroc = {k: [] for k in real_feats}
    for t in range(N_NULL):
        rng = np.random.default_rng(SEED + 1000 + t)
        null_deaths = [filter_deaths(h0_deaths(matched_gaussian(c, rng)))
                       for c in clouds]
        nfeats = featurize(null_deaths, grid, sigma)
        for k, v in nfeats.items():
            null_auroc[k].append(oof_auroc(v, correct))

    summary = {}
    for k in real_feats:
        arr = np.array(null_auroc[k], dtype=np.float64)
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        z = float((real_auroc[k] - mu) / sd) if sd > 1e-9 else float("nan")
        p_emp = float((arr >= real_auroc[k]).mean())
        summary[k] = {
            "auroc_real": float(real_auroc[k]),
            "null_mean": mu,
            "null_std": sd,
            "z": z,
            "p_empirical_one_sided": p_emp,
            "lifts_past_null": bool(np.isfinite(z) and z > 2.0 and p_emp <= 0.05),
        }

    vectorized_lift = (summary["heat"]["lifts_past_null"]
                       or summary["silhouette"]["lifts_past_null"])
    raw_lift = summary["raw"]["lifts_past_null"]
    breaks_f10 = bool(vectorized_lift and not raw_lift)

    out = {
        "experiment": "P11-FE51",
        "description": "Heat/Silhouette PD-vectorization vs matched-covariance null",
        "n_problems": int(len(correct)),
        "params": {
            "eta": ETA, "res": RES, "n_null": N_NULL, "max_tokens": MAX_TOKENS,
            "silhouette_p": SILH_P, "heat_sigma_frac": HEAT_SIGMA_FRAC,
            "scale_p99": scale, "homology_dim": "H0_single_linkage",
        },
        "results": summary,
        "vectorized_lifts_past_null": bool(vectorized_lift),
        "raw_summary_lifts_past_null": bool(raw_lift),
        "breaks_f10_ph_eq_covariance": breaks_f10,
        "verdict": ("VECTORIZATION_SPECIFIC: discretized PDs beat the covariance "
                    "null where raw summaries do not — F-10 framing was "
                    "vectorization-specific" if breaks_f10 else
                    "F-10 holds: vectorized PDs do not beat the matched-covariance "
                    "Gaussian null beyond what raw summaries already capture"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())