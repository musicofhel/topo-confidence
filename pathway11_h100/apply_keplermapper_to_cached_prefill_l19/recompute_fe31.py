"""P11-FE31 — KeplerMapper on cached prefill L19 activations (Goldfarb 1811.00852 params).

Self-contained Mapper implementation (numpy/scipy/sklearn only — no kmapper dep).
Lens = PCA-2 on variance-normalized prefill activations; metric =
VarianceNormalizedEuclidean (Euclidean on per-feature variance-normalized X);
cover resolution=50 per lens axis, gain=3 (cube width = 3 * step); within-cube
clusterer = DBSCAN on the normalized space. Nodes coloured by 1.5B K=1
correctness.

Reports (a) leave-one-out cluster-purity AUROC of unsupervised Mapper node
topology vs the supervised prefill-DoM AUROC 0.7731 (F-2), and (b) D-bucket
node concentration vs A/B/C buckets (DoM-score quartiles) — does the
incorrect-leaning D bucket form one coherent cluster or scatter across many
nodes (F-7)?
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
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/mapper_keplermapper/results.json"

DOM_AUROC_REF = 0.7731  # F-2 supervised prefill-DoM OOF AUROC (1024tok)
RESOLUTION = 50
GAIN = 3.0
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def variance_normalize(X: np.ndarray) -> np.ndarray:
    """VarianceNormalizedEuclidean: standardize each feature to unit variance."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def build_cover(lens: np.ndarray, resolution: int, gain: float):
    """Product cover over 2D lens. Interval i on each axis is centered at
    lo + (i+0.5)*step with half-width = step*gain/2 (gain as width multiplier)."""
    intervals = []
    for axis in range(lens.shape[1]):
        lo = float(lens[:, axis].min())
        hi = float(lens[:, axis].max())
        step = (hi - lo) / resolution
        if step <= 0:
            step = 1.0
        half = step * gain / 2.0
        axis_intervals = []
        for i in range(resolution):
            center = lo + (i + 0.5) * step
            axis_intervals.append((center - half, center + half))
        intervals.append(axis_intervals)
    return intervals


def cluster_cube(Xn_cube: np.ndarray, eps: float):
    """Cluster points inside a cube. <=2 pts -> single node; else DBSCAN
    (noise points dropped, KeplerMapper-style)."""
    n = Xn_cube.shape[0]
    if n == 0:
        return []
    if n <= 2:
        return [np.arange(n)]
    labels = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit_predict(Xn_cube)
    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        clusters.append(np.flatnonzero(labels == lab))
    return clusters


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    n = len(y)
    Xn = variance_normalize(X)

    # PCA-2 lens on the variance-normalized space.
    lens = PCA(n_components=2, random_state=SEED).fit_transform(Xn)

    # DBSCAN eps from typical neighbor scale in the normalized space.
    rng = np.random.default_rng(SEED)
    sub = rng.choice(n, size=min(200, n), replace=False)
    dists = pdist(Xn[sub])
    eps = float(np.median(dists)) * 0.7
    if not np.isfinite(eps) or eps <= 0:
        eps = 1.0

    # Build cover and Mapper nodes.
    intervals = build_cover(lens, RESOLUTION, GAIN)
    nodes = []  # each node = np.ndarray of global point indices
    for ax_lo, ax_hi in intervals[0]:
        in_x = np.flatnonzero((lens[:, 0] >= ax_lo) & (lens[:, 0] <= ax_hi))
        if in_x.size == 0:
            continue
        for ay_lo, ay_hi in intervals[1]:
            mask = (lens[in_x, 1] >= ay_lo) & (lens[in_x, 1] <= ay_hi)
            cube_idx = in_x[mask]
            if cube_idx.size == 0:
                continue
            for local in cluster_cube(Xn[cube_idx], eps):
                nodes.append(cube_idx[local])

    n_nodes = len(nodes)

    # Point -> node membership and leave-one-out node correctness.
    membership = [[] for _ in range(n)]
    node_correct_sum = np.zeros(n_nodes, dtype=np.float64)
    node_size = np.zeros(n_nodes, dtype=np.float64)
    for ni, idx in enumerate(nodes):
        node_correct_sum[ni] = y[idx].sum()
        node_size[ni] = idx.size
        for p in idx:
            membership[p].append(ni)

    global_mean = float(y.mean())
    purity_score = np.full(n, global_mean, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    for p in range(n):
        nids = membership[p]
        if not nids:
            continue
        covered[p] = True
        vals = []
        for ni in nids:
            if node_size[ni] > 1:
                vals.append((node_correct_sum[ni] - float(y[p])) / (node_size[ni] - 1))
            else:
                vals.append(global_mean)
        purity_score[p] = float(np.mean(vals))

    auroc_purity_all = auroc(purity_score, y)
    auroc_purity_covered = (
        auroc(purity_score[covered], y[covered]) if covered.sum() > 0 else float("nan")
    )

    # D-bucket concentration. Buckets from DoM-score quartiles:
    #   A = highest DoM (most correct-leaning) ... D = lowest (incorrect-leaning).
    order = np.argsort(dom_score)
    bucket = np.empty(n, dtype="<U1")
    quart = np.array_split(order, 4)
    for label, idxs in zip(["D", "C", "B", "A"], quart):  # ascending DoM
        bucket[idxs] = label

    bucket_stats = {}
    for label in ["A", "B", "C", "D"]:
        members = np.flatnonzero(bucket == label)
        # Distribution of this bucket's node *memberships* across nodes.
        counts = {}
        total_mem = 0
        for p in members:
            for ni in membership[p]:
                counts[ni] = counts.get(ni, 0) + 1
                total_mem += 1
        distinct_nodes = len(counts)
        if total_mem > 0:
            shares = np.array(list(counts.values()), dtype=np.float64) / total_mem
            herfindahl = float((shares ** 2).sum())  # 1.0 = one cluster
            max_share = float(shares.max())
            ent = float(-(shares * np.log(shares + 1e-12)).sum())
        else:
            herfindahl = float("nan")
            max_share = float("nan")
            ent = float("nan")
        # Of the distinct nodes this bucket touches, how many lean incorrect?
        incorrect_nodes = sum(
            1 for ni in counts if node_size[ni] > 0 and (node_correct_sum[ni] / node_size[ni]) < 0.5
        )
        bucket_stats[label] = {
            "n_members": int(members.size),
            "mean_correct": float(y[members].mean()) if members.size else float("nan"),
            "distinct_nodes": int(distinct_nodes),
            "node_concentration_herfindahl": herfindahl,
            "max_node_share": max_share,
            "node_entropy": ent,
            "incorrect_leaning_nodes_touched": int(incorrect_nodes),
        }

    out = {
        "experiment": "P11-FE31",
        "params": {
            "metric": "VarianceNormalizedEuclidean",
            "lens": "PCA-2",
            "resolution": RESOLUTION,
            "gain": GAIN,
            "dbscan_eps": eps,
            "dbscan_min_samples": 2,
        },
        "n_points": n,
        "n_nodes": int(n_nodes),
        "n_points_covered": int(covered.sum()),
        "coverage_fraction": float(covered.mean()),
        "dom_auroc_ref": DOM_AUROC_REF,
        # (a) cluster-purity AUROC vs supervised DoM
        "auroc_cluster_purity_loo_all": float(auroc_purity_all),
        "auroc_cluster_purity_loo_covered": float(auroc_purity_covered),
        "purity_minus_dom": float(auroc_purity_all - DOM_AUROC_REF),
        # (b) D-bucket node concentration vs A/B/C
        "bucket_node_concentration": bucket_stats,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())