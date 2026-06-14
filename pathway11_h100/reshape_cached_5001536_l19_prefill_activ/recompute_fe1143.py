"""P11-FE1143 — Per-head-block DoM AUROC on the L19 prefill matrix.

The 1536-d L19 prefill hidden state is the concatenation of 12 attention-head
blocks of 128 dims each (12 * 128 = 1536). This script reshapes the cached
500x1536 activation matrix into 500x12x128, fits a within-block Difference-of-
Means (DoM) direction per head over 5 stratified out-of-fold splits, scores each
head block, and reports its OOF AUROC for correctness prediction.

The goal is to test whether the 0.7731 full-space AUROC is carried by a small
"retrieval-head" circuit (RTPurbo hypothesis) rather than spread across all 12
blocks. We rank heads by single-block AUROC and report a greedy cumulative
sweep (top-k blocks concatenated, OOF DoM) to find the minimal head set.
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
OUT_JSON = ROOT / "pathway11_h100/head_block_dom/results.json"

N_HEADS = 12
HEAD_DIM = 128
N_FOLDS = 5
SEED = 9999


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
    """OOF AUROC of a within-block DoM direction over the supplied folds."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = Xte @ d_vec
    return auroc(scores, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), f"unexpected shapes {X.shape} {y.shape}"
    assert N_HEADS * HEAD_DIM == X.shape[1]

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Full-space OOF DoM (reference baseline).
    full_auroc = oof_dom_auroc(X, y, folds)

    # Reshape into per-head blocks: (500, 12, 128).
    Xh = X.reshape(X.shape[0], N_HEADS, HEAD_DIM)

    per_head = []
    for h in range(N_HEADS):
        a = oof_dom_auroc(Xh[:, h, :], y, folds)
        per_head.append({"head": h, "auroc": a})

    # Rank heads by single-block AUROC (NaN sorts last).
    order = sorted(range(N_HEADS), key=lambda h: (per_head[h]["auroc"]
                   if not np.isnan(per_head[h]["auroc"]) else -np.inf), reverse=True)

    # Greedy cumulative sweep: concatenate the top-k blocks and OOF-DoM score.
    cumulative = []
    for k in range(1, N_HEADS + 1):
        cols = np.concatenate([np.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in order[:k]])
        a = oof_dom_auroc(X[:, cols], y, folds)
        cumulative.append({"top_k": k, "heads": [int(h) for h in order[:k]], "auroc": a})

    best_head = order[0]
    out = {
        "experiment": "P11-FE1143",
        "n_samples": int(len(y)),
        "n_heads": N_HEADS,
        "head_dim": HEAD_DIM,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "full_space_oof_dom_auroc": full_auroc,
        "per_head_auroc": per_head,
        "head_rank_by_auroc": [int(h) for h in order],
        "best_head": int(best_head),
        "best_head_auroc": per_head[best_head]["auroc"],
        "cumulative_topk_auroc": cumulative,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())