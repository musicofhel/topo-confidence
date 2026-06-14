"""P11-FE557 — Random-vector AUROC null distribution against prefill-L19 DoM.

Rogue Scalpel (2509.22067) shows that most random directions in a residual
stream meaningfully perturb behavior; by analogy the prefill-DoM correctness
AUROC of 0.7731 (F-2) might be reachable by generic random directions,
falsifying the "privileged correctness direction" framing.

For 1000 random unit vectors v_i drawn isotropically in the d=1536
Qwen-2.5-1.5B L19 residual space, project the cached prefill activations onto
v_i and compute the OOF 5-fold AUROC of <prefill, v_i> for MATH-500
correctness. Because AUROC is sign-asymmetric, score each random direction at
its better orientation (max(a, 1-a)) so the null reflects the best achievable
read from a generic direction. Report the 50/95/99 percentiles of the null
distribution and compare against F-2's 0.7731.
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
OUT_JSON = ROOT / "pathway11_h100/random_vector_null/results.json"

N_FOLDS = 5
N_RANDOM = 1000
SEED = 9999
F2_DOM_AUROC = 0.7731


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


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """OOF supervised difference-of-means AUROC — the F-2 reference readout."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr = X[train_mask]; ytr = y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d_vec
    return auroc(scores, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    n, d = X.shape
    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Reference: OOF supervised DoM AUROC on the same folds (sanity vs F-2).
    dom_auroc = oof_dom_auroc(X, y, folds)

    # Pre-build per-fold test masks for the random sweep.
    test_masks = []
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        test_masks.append((train_mask, test_idx))

    rng = np.random.default_rng(SEED)
    null_aurocs = np.zeros(N_RANDOM, dtype=np.float64)

    for i in range(N_RANDOM):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        # Random directions are fixed (data-independent), so the projection is
        # the same regardless of fold; the OOF structure is preserved because
        # the direction never sees the labels. Score once across all points.
        proj = X @ v
        a = auroc(proj, y)
        # Sign-symmetric: a generic direction has no privileged orientation.
        null_aurocs[i] = max(a, 1.0 - a)

    p50 = float(np.percentile(null_aurocs, 50))
    p95 = float(np.percentile(null_aurocs, 95))
    p99 = float(np.percentile(null_aurocs, 99))
    p_max = float(null_aurocs.max())
    frac_ge_f2 = float((null_aurocs >= F2_DOM_AUROC).mean())

    out = {
        "experiment": "P11-FE557",
        "description": "Random-vector AUROC null distribution vs prefill-L19 DoM (F-2)",
        "n_random": N_RANDOM,
        "n_folds": N_FOLDS,
        "d": int(d),
        "seed": SEED,
        "f2_dom_auroc_reference": F2_DOM_AUROC,
        "dom_auroc_oof_this_run": dom_auroc,
        "null_random_auroc_mean": float(null_aurocs.mean()),
        "null_random_auroc_std": float(null_aurocs.std()),
        "null_random_auroc_p50": p50,
        "null_random_auroc_p95": p95,
        "null_random_auroc_p99": p99,
        "null_random_auroc_max": p_max,
        "frac_random_ge_f2": frac_ge_f2,
        "verdict": (
            "NULL_REFUTED_PRIVILEGE" if p_max >= F2_DOM_AUROC
            else "PRIVILEGE_SURVIVES"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())