"""P11-FE1094 — Orthogonality-constrained residual probe for mechanism plurality.

Tests whether the correctness signal in L19 prefill activations lives on a single
direction (the DoM direction) or is distributed across many. Per OOF fold:
  1. Fit the DoM direction (mean_pos - mean_neg) on the train fold.
  2. Project the DoM component out of both train and test activations
     (orthogonal complement of the DoM direction).
  3. Train an L2-regularized logistic regression on the residual subspace.
  4. Score the held-out test fold.

A residual AUROC near 0.5 confirms the signal is concentrated on the single DoM
direction; a residual AUROC > 0.65 confirms mechanism plurality (signal spread
across many directions). For reference we also report the raw full-space and raw
DoM-only OOF AUROCs.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/orthogonal_residual_probe/results.json"

N_FOLDS = 5
SEED = 9999
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]


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


def project_out(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of each row of X along the unit `direction`."""
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-12:
        return X.copy()
    u = direction / nrm
    coeff = X @ u
    return X - np.outer(coeff, u)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), (X.shape, y.shape)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)

    raw_full_scores = np.zeros(n, dtype=np.float64)
    raw_dom_scores = np.zeros(n, dtype=np.float64)
    residual_scores = np.zeros(n, dtype=np.float64)

    # Fixed C chosen by a quick inner sweep on full residual data per fold would
    # add cost; instead use a single moderate C and report the best over a small
    # grid evaluated on the aggregated OOF residual scores per C.
    residual_scores_by_c = {c: np.zeros(n, dtype=np.float64) for c in C_GRID}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # DoM direction on train fold.
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)

        # Raw full-space probe (DoM as linear scorer).
        raw_dom_scores[test_idx] = Xte @ d_vec

        # Full-space logistic regression (reference ceiling), standardized.
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        clf_full = LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs")
        clf_full.fit((Xtr - mu) / sd, ytr)
        raw_full_scores[test_idx] = clf_full.decision_function((Xte - mu) / sd)

        # Project DoM direction out of train + test.
        Xtr_res = project_out(Xtr, d_vec)
        Xte_res = project_out(Xte, d_vec)
        mu_r = Xtr_res.mean(axis=0)
        sd_r = Xtr_res.std(axis=0) + 1e-8
        Xtr_rs = (Xtr_res - mu_r) / sd_r
        Xte_rs = (Xte_res - mu_r) / sd_r

        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
            clf.fit(Xtr_rs, ytr)
            residual_scores_by_c[c][test_idx] = clf.decision_function(Xte_rs)

    auroc_by_c = {f"C={c}": float(auroc(residual_scores_by_c[c], y)) for c in C_GRID}
    best_c = max(C_GRID, key=lambda c: auroc(residual_scores_by_c[c], y))
    residual_scores = residual_scores_by_c[best_c]

    auroc_residual = float(auroc(residual_scores, y))
    auroc_raw_full = float(auroc(raw_full_scores, y))
    auroc_raw_dom = float(auroc(raw_dom_scores, y))

    if auroc_residual <= 0.55:
        verdict = "single_direction"
    elif auroc_residual >= 0.65:
        verdict = "mechanism_plurality"
    else:
        verdict = "intermediate"

    out = {
        "experiment": "P11-FE1094",
        "n": int(n),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_raw_dom_oof": auroc_raw_dom,
        "auroc_raw_full_logreg_oof": auroc_raw_full,
        "auroc_residual_logreg_oof": auroc_residual,
        "best_C": best_c,
        "auroc_residual_by_C": auroc_by_c,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())