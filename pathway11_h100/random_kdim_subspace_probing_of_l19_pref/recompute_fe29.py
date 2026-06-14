"""P11-FE29 — Random k-dim subspace probing of L19 prefill activations.

Direct analog of Li et al.'s d_int90 sweep applied to the probing problem.
Cached 1.5B L19 prefill activations (500, 1536) are projected through a
Gaussian random matrix P ∈ ℝ^(1536×k) for k ∈ {1, 5, 50, 500, 1536}. In each
random subspace a 5-fold out-of-fold logistic regression is fit and scored by
AUROC, averaged over several random projections per k. These are compared to
the supervised L19 DoM baseline (mean-difference direction, OOF) and the
published F-2 figure (0.7731). d_probe is reported as the smallest k whose mean
random-subspace AUROC reaches 90% of the supervised DoM OOF AUROC.

If random k=50 already matches supervised DoM, F-2's "this direction is
privileged" claim weakens to "the correctness subspace is low effective dim and
any random sketch finds it."
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/random_subspace_probe/results.json"

SEED = 9999
N_FOLDS = 5
N_PROJ = 10                      # random projections averaged per k
K_VALUES = [1, 5, 50, 500, 1536]
SUPERVISED_DOM_PUBLISHED = 0.7731


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oof_logreg_auroc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """5-fold OOF L2 logistic regression AUROC on standardized features."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr_s, y[train_idx])
        oof[test_idx] = clf.decision_function(Xte_s)
    return auroc(oof, y)


def supervised_dom_oof_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """OOF mean-difference (DoM) direction AUROC — the F-2 baseline."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d_vec
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    d = X.shape[1]
    supervised_auroc = supervised_dom_oof_auroc(X, y)
    target_auroc = 0.90 * supervised_auroc

    rng = np.random.default_rng(SEED)
    per_k = {}
    for k in K_VALUES:
        proj_aurocs = []
        for p in range(N_PROJ):
            # Gaussian random projection P ∈ R^(d x k), columns ~ N(0, 1/d)
            P = rng.standard_normal((d, k)) / np.sqrt(d)
            Xp = X @ P
            proj_aurocs.append(oof_logreg_auroc(Xp, y, seed=SEED + p))
        arr = np.asarray(proj_aurocs, dtype=np.float64)
        per_k[k] = {
            "mean_auroc": float(arr.mean()),
            "std_auroc": float(arr.std()),
            "min_auroc": float(arr.min()),
            "max_auroc": float(arr.max()),
            "frac_of_supervised": float(arr.mean() / supervised_auroc),
            "n_proj": N_PROJ,
        }

    # d_probe = smallest k whose mean random-subspace AUROC reaches 90% of supervised
    d_probe = None
    for k in K_VALUES:
        if per_k[k]["mean_auroc"] >= target_auroc:
            d_probe = k
            break

    out = {
        "experiment": "P11-FE29",
        "supervised_dom_oof_auroc": float(supervised_auroc),
        "supervised_dom_published": SUPERVISED_DOM_PUBLISHED,
        "target_auroc_90pct": float(target_auroc),
        "k_values": K_VALUES,
        "n_proj_per_k": N_PROJ,
        "per_k": {str(k): v for k, v in per_k.items()},
        "d_probe": d_probe,
        "d_probe_note": (
            "smallest random-subspace dim k whose mean OOF logreg AUROC "
            "reaches 90% of the supervised L19 DoM OOF AUROC"
        ),
        "interpretation": (
            "If d_probe is small (e.g. <=50), the correctness signal lives in a "
            "low effective-dim subspace findable by any random sketch, weakening "
            "F-2's 'privileged direction' framing."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())