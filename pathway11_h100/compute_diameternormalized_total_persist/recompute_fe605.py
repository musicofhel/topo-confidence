"""P11-FE605 — Diameter-normalized total persistence on the prefill point cloud.

Tests whether the F-10 "PH = Gaussian null" verdict is summary-specific by
swapping raw total persistence for the *diameter-normalized* total persistence
of eq. (1) of arXiv 2512.15285:

    TP(X) = ( Persistence_0(X) + Persistence_1(X) ) / diam(X)

where Persistence_d is the sum of finite (death - birth) lifetimes of dimension-d
features in the Vietoris-Rips filtration, and diam is the max pairwise distance.

Two probes on the P11 Qwen-2.5-1.5B L19 prefill cloud (500 problems x 1536 dim):

  (a) Global separation. TP of the correct subset (n=243) vs the incorrect
      subset (n=257), against a Gaussian-matched-covariance permutation null
      (sample 500 pts ~ N(mu, Sigma) of the pooled cloud, random 243/257 split,
      recompute the diff). Significant if two-sided p < 0.01.

  (b) Label-free per-problem probe. For each problem take its k-NN neighborhood,
      compute the neighborhood's diameter-normalized TP, and rank correctness by
      that scalar. Viable contender for H-12 if max(AUROC, 1-AUROC) >= 0.65,
      rankable against F-2's DoM AUROC 0.7731.

Persistent homology (H0 via the standard Z2 boundary-matrix reduction over the
VR 2-skeleton; no gudhi/ripser) is implemented from scratch in numpy. Global TDA
is run in a PCA-reduced space on fixed-size subsamples with a scale cap to keep
the 2-skeleton tractable on CPU; the per-problem probe runs on the raw cloud.
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
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/persistence_diam_norm/results.json"

SEED = 9999
DOM_AUROC_REF = 0.7731       # F-2 prefill L19 DoM, OOF 5-fold (1024-tok)

# --- global separation (part a) ---
D_PCA = 12                   # PCA dims for the global TDA space
SUB_GLOBAL = 50              # subsample size per subset (bounds the 2-skeleton)
R_SUB = 3                    # subsamples averaged per TP estimate
N_PERM = 100                 # Gaussian permutations
MAX_SCALE_Q = 0.7            # cap VR edge length at this quantile of pairdists

# --- per-problem probe (part b) ---
K_NN = 16                    # neighbors per problem (neighborhood = K_NN + self)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def pairwise(X: np.ndarray) -> np.ndarray:
    g = (X * X).sum(axis=1)
    d2 = g[:, None] + g[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def rips_persistence(D: np.ndarray, max_scale: float | None):
    """Finite H0 and H1 total persistence of the VR filtration of D (Z2).

    Returns (sum_H0_lifetimes, sum_H1_lifetimes) over finite pairs only
    (the single essential H0 class is excluded).
    """
    n = D.shape[0]
    simplices: list[tuple[float, tuple]] = [(0.0, (v,)) for v in range(n)]
    edge_filt: dict[tuple, float] = {}
    neigh = [set() for _ in range(n)]
    for i in range(n):
        Di = D[i]
        for j in range(i + 1, n):
            d = float(Di[j])
            if max_scale is None or d <= max_scale:
                simplices.append((d, (i, j)))
                edge_filt[(i, j)] = d
                neigh[i].add(j)
                neigh[j].add(i)

    # triangles: i < j < k with all three edges present
    for i in range(n):
        ni = neigh[i]
        for j in ni:
            if j <= i:
                continue
            common = ni & neigh[j]
            for k in common:
                if k <= j:
                    continue
                f = max(edge_filt[(i, j)], edge_filt[(i, k)], edge_filt[(j, k)])
                simplices.append((f, (i, j, k)))

    simplices.sort(key=lambda s: (s[0], len(s[1]), s[1]))
    index = {s[1]: idx for idx, s in enumerate(simplices)}
    filt = [s[0] for s in simplices]
    dim = [len(s[1]) - 1 for s in simplices]

    columns: list = [None] * len(simplices)
    low_to_col: dict[int, int] = {}
    p0 = 0.0
    p1 = 0.0

    for idx, (f, verts) in enumerate(simplices):
        d = len(verts) - 1
        if d == 0:
            columns[idx] = set()
            continue
        if d == 1:
            a, b = verts
            col = {index[(a,)], index[(b,)]}
        else:
            a, b, c = verts
            col = {index[(a, b)], index[(a, c)], index[(b, c)]}

        while col:
            low = max(col)
            owner = low_to_col.get(low)
            if owner is None:
                break
            col ^= columns[owner]

        if col:
            low = max(col)
            low_to_col[low] = idx
            columns[idx] = col
            life = f - filt[low]
            if dim[low] == 0:
                p0 += life
            elif dim[low] == 1:
                p1 += life
        else:
            columns[idx] = set()

    return p0, p1


def total_persistence(X: np.ndarray, max_scale_q: float | None):
    """Diameter-normalized total persistence (P0 + P1) / diam of point cloud X."""
    D = pairwise(X)
    n = len(X)
    diam = float(D.max()) if n > 1 else 0.0
    if diam <= 0.0:
        return 0.0, 0.0, 0.0
    ms = None
    if max_scale_q is not None:
        ms = float(np.quantile(D[np.triu_indices(n, 1)], max_scale_q))
    p0, p1 = rips_persistence(D, ms)
    return (p0 + p1) / diam, p0 / diam, p1 / diam


def subsample_tp(pts: np.ndarray, m: int, reps: int, rng) -> float:
    vals = []
    for _ in range(reps):
        P = pts
        if len(pts) > m:
            sel = rng.choice(len(pts), m, replace=False)
            P = pts[sel]
        tp, _, _ = total_persistence(P, MAX_SCALE_Q)
        vals.append(tp)
    return float(np.mean(vals))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or y.shape != (500,):
        print("MISSING_REGEN_INPUT bad shapes", X.shape, y.shape, file=sys.stderr)
        return 2

    n_correct = int(y.sum())
    n_incorrect = int((~y).sum())
    rng = np.random.default_rng(SEED)

    # -------- part (b): label-free per-problem k-NN persistence probe --------
    Dfull = pairwise(X)
    knn_score = np.zeros(len(X), dtype=np.float64)
    for i in range(len(X)):
        order = np.argsort(Dfull[i], kind="stable")[: K_NN + 1]  # self + K_NN
        tp, _, _ = total_persistence(X[order], None)
        knn_score[i] = tp
    auroc_knn = auroc(knn_score, y)
    auroc_knn_best = float(max(auroc_knn, 1.0 - auroc_knn))

    # -------- part (a): global correct-vs-incorrect separation --------
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:D_PCA].T

    obs_correct_tp = subsample_tp(Z[y], SUB_GLOBAL, R_SUB, rng)
    obs_incorrect_tp = subsample_tp(Z[~y], SUB_GLOBAL, R_SUB, rng)
    obs_diff = obs_correct_tp - obs_incorrect_tp

    mu = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    L = (V * np.sqrt(w)) @ V.T  # symmetric sqrt: L @ L == cov

    null_diffs = np.zeros(N_PERM, dtype=np.float64)
    for p in range(N_PERM):
        G = mu + rng.standard_normal((500, D_PCA)) @ L
        perm = rng.permutation(500)
        c_idx, i_idx = perm[:n_correct], perm[n_correct:]
        tp_c = subsample_tp(G[c_idx], SUB_GLOBAL, R_SUB, rng)
        tp_i = subsample_tp(G[i_idx], SUB_GLOBAL, R_SUB, rng)
        null_diffs[p] = tp_c - tp_i

    p_value = float((1 + int((np.abs(null_diffs) >= abs(obs_diff)).sum())) / (1 + N_PERM))

    separation_significant = bool(p_value < 0.01)
    knn_viable = bool(auroc_knn_best >= 0.65)

    out = {
        "experiment": "P11-FE605",
        "metric": "diameter_normalized_total_persistence_H0_plus_H1",
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "params": {
            "D_PCA": D_PCA,
            "SUB_GLOBAL": SUB_GLOBAL,
            "R_SUB": R_SUB,
            "N_PERM": N_PERM,
            "MAX_SCALE_Q": MAX_SCALE_Q,
            "K_NN": K_NN,
            "seed": SEED,
        },
        "global_separation": {
            "tp_correct": obs_correct_tp,
            "tp_incorrect": obs_incorrect_tp,
            "observed_diff": float(obs_diff),
            "null_diff_mean": float(null_diffs.mean()),
            "null_diff_std": float(null_diffs.std()),
            "p_value_two_sided": p_value,
            "significant_p_lt_0.01": separation_significant,
        },
        "knn_persistence_probe": {
            "auroc": float(auroc_knn),
            "auroc_best_oriented": auroc_knn_best,
            "dom_auroc_reference": DOM_AUROC_REF,
            "viable_ge_0.65": knn_viable,
        },
        "verdict": {
            "f10_partially_refuted": separation_significant or knn_viable,
            "summary": (
                "diameter-normalized PH separates correct/incorrect and/or gives a "
                "viable label-free probe — F-10 (PH=Gaussian null) is summary-specific"
                if (separation_significant or knn_viable)
                else "diameter-normalized PH neither separates nor probes correctness — "
                "F-10 holds under this summary too"
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("OK", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())