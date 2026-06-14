"""P11-FE202 — Head-wise logistic-regression probe panel on Qwen-2.5-1.5B L19.

SMITIN (per-head probe accuracy varies dramatically within a layer) motivates
splitting the L19 prefill residual stream into its constituent attention-head
sub-blocks and probing each independently. Qwen-2.5-1.5B has hidden_size=1536
= 12 attention heads x 128 head_dim; the cached residual concatenates the
head sub-spaces, so we slice it into 12 contiguous 128-d blocks and fit an
OOF logistic-regression probe per head for prefill correctness on MATH-500.

We then compare the best single-head OOF AUROC against the layer-aggregate
F-2 number (0.7731). If best-head AUROC > 0.85 the "L19 is the locus" framing
is refuted in favor of "a small subset of L19 heads is the locus".
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
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/headwise_probe/results.json"

SEED = 9999
N_FOLDS = 5
NUM_HEADS = 12
HEAD_DIM = 128
F2_LAYER_AGGREGATE_AUROC = 0.7731
REFUTE_THRESHOLD = 0.85
C_REG = 1.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oof_logreg_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold logistic-regression decision scores for one feature block."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C_REG, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.decision_function(Xte)
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), f"unexpected shapes {X.shape} {y.shape}"
    assert NUM_HEADS * HEAD_DIM == X.shape[1]

    per_head_auroc = []
    for h in range(NUM_HEADS):
        block = X[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        oof = oof_logreg_scores(block, y)
        per_head_auroc.append(float(auroc(oof, y)))

    # Full-layer logistic probe (all 1536 dims) for an apples-to-apples
    # reference against the head-wise panel and the F-2 DoM number.
    full_oof = oof_logreg_scores(X, y)
    full_layer_auroc = float(auroc(full_oof, y))

    best_head = int(np.argmax(per_head_auroc))
    best_head_auroc = float(per_head_auroc[best_head])
    worst_head = int(np.argmin(per_head_auroc))
    worst_head_auroc = float(per_head_auroc[worst_head])

    refuted = best_head_auroc > REFUTE_THRESHOLD

    out = {
        "experiment": "P11-FE202",
        "description": "Head-wise logistic-regression probe panel on L19 (12 heads x 128 dim)",
        "n": int(len(y)),
        "n_folds": N_FOLDS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "per_head_auroc": per_head_auroc,
        "best_head": best_head,
        "best_head_auroc": best_head_auroc,
        "worst_head": worst_head,
        "worst_head_auroc": worst_head_auroc,
        "head_auroc_mean": float(np.mean(per_head_auroc)),
        "head_auroc_std": float(np.std(per_head_auroc)),
        "full_layer_logreg_auroc": full_layer_auroc,
        "f2_layer_aggregate_auroc": F2_LAYER_AGGREGATE_AUROC,
        "best_head_minus_f2": best_head_auroc - F2_LAYER_AGGREGATE_AUROC,
        "refute_threshold": REFUTE_THRESHOLD,
        "f2_locus_refuted": bool(refuted),
        "interpretation": (
            "best-head AUROC exceeds 0.85: signal localizes to a head subset, F-2 'L19 is the locus' refined"
            if refuted else
            "no single head beats 0.85: F-2 layer-aggregate framing holds; signal is distributed across heads"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())