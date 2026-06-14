"""P11-FE613 — Top-k SVD of the L19 prefill correctness contrast.

Forms the correctness contrast matrix M = (X_correct - mu_incorrect) on each
training fold, takes its top-k right singular vectors, and evaluates the
cumulative out-of-fold AUROC explained by k = 1, 2, 4, 8 components (a
logistic combiner is fit on the k projections per fold).

Motivation: paper 2512.16602 finds refusal signals 'distributed across many
dimensions' in Qwen3-80B, whereas F-2/F-3 quietly assume a rank-1 DoM
direction. If k=4 lifts OOF AUROC by >= 0.02 over k=1 (the F-2 DoM), the
rank-1 framing is an under-fit and correctness is multi-dimensional like
refusal. Cheap, falsifiable check on existing cached prefill activations.
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
OUT_JSON = ROOT / "pathway11_h100/svd_contrast/results.json"

N_FOLDS = 5
SEED = 9999
K_LIST = [1, 2, 4, 8]
LIFT_THRESHOLD = 0.02


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


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), (X.shape, y.shape)

    n = len(y)
    folds = stratified_kfold(y, N_FOLDS, SEED)
    max_k = max(K_LIST)

    # OOF projection scores: one column per requested k.
    oof_scores = {k: np.zeros(n, dtype=np.float64) for k in K_LIST}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # Correctness contrast: correct rows centered on the incorrect mean.
        mu_inc = Xtr[~ytr].mean(axis=0)
        M = Xtr[ytr] - mu_inc  # (n_correct, 1536)

        # Top-k right singular vectors span the correctness contrast subspace.
        # Vt rows are the directions; columns of V = Vt.T are unit vectors.
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        V = Vt[:max_k].T  # (1536, max_k)

        # Center both splits on the train mean before projecting onto the basis.
        mu_tr = Xtr.mean(axis=0)
        Ptr_full = (Xtr - mu_tr) @ V  # (n_train, max_k)
        Pte_full = (Xte - mu_tr) @ V  # (n_test, max_k)

        for k in K_LIST:
            Ptr = Ptr_full[:, :k]
            Pte = Pte_full[:, :k]
            if k == 1:
                # Single direction: oriented projection, no combiner needed.
                d = Ptr[:, 0]
                sign = 1.0 if (d[ytr].mean() - d[~ytr].mean()) >= 0 else -1.0
                oof_scores[k][test_idx] = sign * Pte[:, 0]
            else:
                clf = LogisticRegression(C=1.0, max_iter=2000)
                clf.fit(Ptr, ytr)
                oof_scores[k][test_idx] = clf.decision_function(Pte)

    aurocs = {k: float(auroc(oof_scores[k], y)) for k in K_LIST}
    lift_k4_vs_k1 = aurocs[4] - aurocs[1]

    out = {
        "experiment": "P11-FE613",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "k_list": K_LIST,
        "auroc_oof_by_k": {str(k): aurocs[k] for k in K_LIST},
        "lift_k4_vs_k1": float(lift_k4_vs_k1),
        "lift_threshold": LIFT_THRESHOLD,
        "multidimensional": bool(lift_k4_vs_k1 >= LIFT_THRESHOLD),
        "interpretation": (
            "k=4 lifts >= 0.02 over k=1: correctness is multi-dimensional, "
            "rank-1 F-2/F-3 framing is an under-fit"
            if lift_k4_vs_k1 >= LIFT_THRESHOLD
            else "k=4 lift < 0.02: rank-1 DoM framing survives"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())