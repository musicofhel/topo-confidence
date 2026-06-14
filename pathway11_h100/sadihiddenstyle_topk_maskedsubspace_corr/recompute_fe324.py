"""FE324 — SADI-HIDDEN-style top-K masked-subspace correctness probe on L19 prefill.

Per-fold contrastive feature selection: compute per-dimension
mean(correct) − mean(incorrect) at L19 prefill on the train fold, binarize to
the top-K dims by |diff|, and project each problem onto that K-dim subspace
(i.e. keep those K columns). Train a logistic-regression correctness probe on
the masked subspace for K ∈ {64, 128, 256, 512}, and compare against the F-2
baseline: a logistic probe on the single full-residual L19 DoM projection.

All selection (mask, DoM direction) is fit on train folds only; 5-fold OOF
AUROC is reported for each K and for the F-2 baseline. Direct test of whether
F-2's 0.7731 lives in a rank-1 fixed direction or a sparse subspace of L19.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/sadi_topk_subspace/results.json"

N_FOLDS = 5
SEED = 9999
K_VALUES = [64, 128, 256, 512]


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


def fit_probe(Xtr, ytr, Xte):
    """Standardize on train, fit LR, return test decision scores."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)
    return clf.decision_function(Xte_s)


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

    n, d = X.shape
    folds = stratified_kfold(y, N_FOLDS, SEED)

    oof_topk = {K: np.zeros(n, dtype=np.float64) for K in K_VALUES}
    oof_dom = np.zeros(n, dtype=np.float64)
    # track mask stability: how often each dim is selected across folds, per K
    sel_counts = {K: np.zeros(d, dtype=np.int64) for K in K_VALUES}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # contrastive per-dim mean difference on train fold only
        diff_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)

        # (b) F-2 baseline: probe on the single full-residual DoM projection
        dom_tr = (Xtr @ diff_vec).reshape(-1, 1)
        dom_te = (Xte @ diff_vec).reshape(-1, 1)
        oof_dom[test_idx] = fit_probe(dom_tr, ytr, dom_te)

        # (a) top-K masked subspace probes
        order = np.argsort(np.abs(diff_vec))[::-1]
        for K in K_VALUES:
            cols = order[:K]
            sel_counts[K][cols] += 1
            oof_topk[K][test_idx] = fit_probe(Xtr[:, cols], ytr, Xte[:, cols])

    per_k = {}
    for K in K_VALUES:
        stable = int((sel_counts[K] == N_FOLDS).sum())
        per_k[str(K)] = {
            "auroc_oof": float(auroc(oof_topk[K], y)),
            "dims_selected_all_folds": stable,
            "frac_subspace": float(K) / d,
        }

    out = {
        "experiment": "P11-FE324",
        "description": "SADI top-K masked-subspace vs F-2 rank-1 DoM probe on L19 prefill",
        "n_problems": n,
        "n_dims": d,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_f2_dom_baseline_oof": float(auroc(oof_dom, y)),
        "topk": per_k,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())