"""P11-FE721 — Concept-cone replacement for single-direction DoM.

Replaces the mean-diff F-2 vector with a rank-k subspace ("concept cone",
Wollschlager et al. via paper 2603.18280) extracted by SVD of the per-example
(correct - incorrect) difference matrix at L19 prefill. Sweeps
k in {1, 2, 4, 8, 16} and scores each test example by the maximum signed
projection onto the k oriented cone directions. All AUROCs are 5-fold OOF.

Test: does the cone AUROC rise monotonically to >0.83 by rank>=4? If so, the
F-2/F-3 single-direction framing is incomplete.
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
OUT_JSON = ROOT / "pathway11_h100/concept_cone/results.json"

N_FOLDS = 5
SEED = 9999
RANKS = [1, 2, 4, 8, 16]
MONO_RANK = 4
HIT_THRESH = 0.83


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


def cone_basis(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the (correct - mean_incorrect) matrix.

    Uncentered, so the rank-1 cone reproduces the mean-diff DoM direction;
    higher ranks add the spread of correct examples around that mean.
    Each basis vector is oriented so correct examples project higher than
    incorrect ones on the training fold.
    """
    mu_neg = X_train[~y_train].mean(axis=0)
    M = X_train[y_train] - mu_neg            # (n_correct, d) difference matrix
    # economy SVD; right singular vectors are rows of Vt
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt[:k].copy()                        # (k, d)
    for j in range(k):
        proj = X_train @ V[j]
        sgn = np.sign(proj[y_train].mean() - proj[~y_train].mean())
        if sgn == 0.0:
            sgn = 1.0
        V[j] *= sgn
    return V


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)

    oof_scores = {k: np.zeros(n, dtype=np.float64) for k in RANKS}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        for k in RANKS:
            V = cone_basis(Xtr, ytr, k)      # (k, d), oriented
            proj = Xte @ V.T                 # (n_test, k)
            oof_scores[k][test_idx] = proj.max(axis=1)

    auroc_by_rank = {str(k): float(auroc(oof_scores[k], y)) for k in RANKS}

    vals = [auroc_by_rank[str(k)] for k in RANKS]
    monotone = all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1))
    rank_ge4 = [auroc_by_rank[str(k)] for k in RANKS if k >= MONO_RANK]
    hits_thresh = any(v > HIT_THRESH for v in rank_ge4)

    out = {
        "experiment": "P11-FE721",
        "description": "concept-cone (rank-k SVD subspace) vs single-direction DoM, max-projection score, 5-fold OOF",
        "ranks": RANKS,
        "auroc_by_rank": auroc_by_rank,
        "dom_rank1_auroc": auroc_by_rank["1"],
        "best_rank": int(RANKS[int(np.argmax(vals))]),
        "best_auroc": float(max(vals)),
        "monotone_nondecreasing": bool(monotone),
        "exceeds_0_83_by_rank_ge4": bool(hits_thresh),
        "verdict": (
            "single-direction-incomplete" if (monotone and hits_thresh)
            else "single-direction-sufficient"
        ),
        "n_folds": N_FOLDS,
        "seed": SEED,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())