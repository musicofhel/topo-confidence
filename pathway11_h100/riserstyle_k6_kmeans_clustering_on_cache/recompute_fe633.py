"""P11-FE633 — RISER-style K=6 mixture-of-linear-probes vs single-direction L19 DoM.

Tests whether the L19 prefill correctness signal is single-direction or
multi-cluster. RISER reports a 6-vector library beats Top-1 single-vector by
1.7-3.4pp; F-9 claims CoE-60 is redundant with single-layer L19 DoM. This is the
cheapest direct test: K=6 K-means partitions the cached prefill features, a
linear probe is fit per cluster (hard 1-of-6 gating by nearest centroid), and the
mixture's OOF 5-fold AUROC is compared against the single-direction DoM baseline
(0.7731). All clustering/probe fitting is per-fold (train-only) to keep the OOF
estimate honest.
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
OUT_JSON = ROOT / "pathway11_h100/riser_mixture_probe/results.json"

N_FOLDS = 5
K_CLUSTERS = 6
SEED = 9999
RIDGE_ALPHA_REL = 1e-3


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


def fit_ridge_probe(X: np.ndarray, y: np.ndarray, alpha_rel: float) -> np.ndarray:
    """Ridge-regularized covariance-whitened DoM probe; returns weight vector."""
    mu_pos = X[y].mean(axis=0) if y.any() else X.mean(axis=0)
    mu_neg = X[~y].mean(axis=0) if (~y).any() else X.mean(axis=0)
    d_vec = mu_pos - mu_neg
    Xc = X - X.mean(axis=0)
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = alpha_rel * trace / d if d > 0 else alpha_rel
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    try:
        w = np.linalg.solve(Sigma_ridge, d_vec)
    except np.linalg.LinAlgError:
        w = d_vec
    return w


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("MISSING_REGEN_INPUT sklearn", file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (500,)

    n = len(y)
    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Baselines: pre-computed DoM scores, and a per-fold single-direction DoM
    # refit (so the comparison against the mixture uses identical folds/data).
    single_dom_oof = np.zeros(n, dtype=np.float64)
    mixture_oof = np.zeros(n, dtype=np.float64)
    cluster_assignment = np.full(n, -1, dtype=int)

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # Single-direction DoM refit on this train fold.
        w_single = fit_ridge_probe(Xtr, ytr, RIDGE_ALPHA_REL)
        single_dom_oof[test_idx] = Xte @ w_single

        # Standardize features for K-means (train statistics only).
        feat_mu = Xtr.mean(axis=0)
        feat_sd = Xtr.std(axis=0) + 1e-8
        Ztr = (Xtr - feat_mu) / feat_sd
        Zte = (Xte - feat_mu) / feat_sd

        km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
        tr_labels = km.fit_predict(Ztr)

        # One linear probe per cluster (fit on raw features within cluster).
        probes = {}
        for c in range(K_CLUSTERS):
            sel = tr_labels == c
            if sel.sum() < 4 or not (ytr[sel].any() and (~ytr[sel]).any()):
                # Degenerate cluster: fall back to the global single-DoM probe.
                probes[c] = w_single
            else:
                probes[c] = fit_ridge_probe(Xtr[sel], ytr[sel], RIDGE_ALPHA_REL)

        # Hard 1-of-6 gating: assign each test problem to its nearest centroid.
        te_labels = km.predict(Zte)
        cluster_assignment[test_idx] = te_labels
        for j, ti in enumerate(test_idx):
            mixture_oof[ti] = Xte[j] @ probes[te_labels[j]]

    # Per-cluster scores live on different linear scales; rank-normalize within
    # each test fold's cluster so the pooled mixture AUROC is well-defined.
    mixture_oof_z = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        seg = mixture_oof[test_idx]
        order = seg.argsort().argsort().astype(np.float64)
        mixture_oof_z[test_idx] = (order - order.mean()) / (order.std() + 1e-8)

    auroc_single_precomputed = auroc(dom_score, y)
    auroc_single_refit = auroc(single_dom_oof, y)
    auroc_mixture_raw = auroc(mixture_oof, y)
    auroc_mixture_ranknorm = auroc(mixture_oof_z, y)

    cluster_sizes = [int((cluster_assignment == c).sum()) for c in range(K_CLUSTERS)]

    out = {
        "experiment": "P11-FE633",
        "n": int(n),
        "k_clusters": K_CLUSTERS,
        "n_folds": N_FOLDS,
        "auroc_single_dom_precomputed": float(auroc_single_precomputed),
        "auroc_single_dom_refit_oof": float(auroc_single_refit),
        "auroc_mixture_oof_raw": float(auroc_mixture_raw),
        "auroc_mixture_oof_ranknorm": float(auroc_mixture_ranknorm),
        "lift_vs_single_refit_pp": float(
            100.0 * (auroc_mixture_ranknorm - auroc_single_refit)
        ),
        "lift_vs_precomputed_pp": float(
            100.0 * (auroc_mixture_ranknorm - auroc_single_precomputed)
        ),
        "baseline_dom_0p7731": 0.7731,
        "cluster_test_sizes": cluster_sizes,
        "verdict": (
            "MIXTURE_LIFTS"
            if auroc_mixture_ranknorm > auroc_single_refit + 0.005
            else "SINGLE_DIRECTION_SUFFICIENT"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())