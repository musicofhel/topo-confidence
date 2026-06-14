"""P11-FE205 — Head-wise logistic-regression probe panel on Qwen-2.5-1.5B L19.

Motivated by SMITIN's finding that per-head probe accuracy varies dramatically
within a single layer (94.3% drums vs near-chance neighbours). F-2's layer-mean
DoM (AUROC 0.7731) may average over a sparse signal concentrated in a small set
of L19 heads.

Qwen-2.5-1.5B L19 residual width is 1536 = 12 heads x 128 head_dim. The cached
`prefill` activation is the post-attention residual-stream slice at L19, so a
contiguous 128-d block is a *proxy* for a single attention head's contribution
(the W_O output mixing is not inverted — we slice the residual, not the raw
head outputs). For each head-slice we fit an OOF logistic-regression probe and
report per-head AUROC. If the best single-head OOF AUROC exceeds 0.85 the
"L19 is the locus" framing is refuted in favour of "a small subset of L19 heads
is the locus".
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
OUT_JSON = ROOT / "pathway11_h100/headwise_probe_panel/results.json"

N_HEADS = 12
HEAD_DIM = 128
HIDDEN = N_HEADS * HEAD_DIM  # 1536
N_FOLDS = 5
SEED = 9999
C_REG = 1.0
F2_LAYER_AGGREGATE = 0.7731
REFUTE_THRESHOLD = 0.85


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oof_probe_auroc(X: np.ndarray, y: np.ndarray, skf: StratifiedKFold) -> float:
    """Out-of-fold logistic-regression decision scores -> AUROC."""
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C_REG, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.decision_function(Xte)
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, HIDDEN) or y.shape != (500,):
        print(f"UNEXPECTED_SHAPE X={X.shape} y={y.shape}", file=sys.stderr)
        return 3

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    per_head = []
    for h in range(N_HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        head_auroc = oof_probe_auroc(X[:, sl], y, skf)
        per_head.append({"head": h, "auroc": float(head_auroc)})

    # Full-layer probe for reference (matches the F-2 aggregate readout family).
    full_layer_auroc = oof_probe_auroc(X, y, skf)

    aurocs = np.array([d["auroc"] for d in per_head], dtype=np.float64)
    best_idx = int(np.nanargmax(aurocs))
    best_auroc = float(aurocs[best_idx])
    worst_idx = int(np.nanargmin(aurocs))

    refuted = bool(best_auroc > REFUTE_THRESHOLD)

    out = {
        "experiment": "P11-FE205",
        "description": "Head-wise logistic-regression probe panel on L19 attention heads",
        "n_examples": int(len(y)),
        "n_heads": N_HEADS,
        "head_dim": HEAD_DIM,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "C_reg": C_REG,
        "caveat": (
            "Head-slice is a proxy: cached prefill is the post-attention "
            "residual-stream slice; W_O head mixing is not inverted."
        ),
        "per_head_auroc": per_head,
        "best_head": best_idx,
        "best_head_auroc": best_auroc,
        "worst_head": worst_idx,
        "worst_head_auroc": float(aurocs[worst_idx]),
        "mean_head_auroc": float(np.nanmean(aurocs)),
        "std_head_auroc": float(np.nanstd(aurocs)),
        "full_layer_probe_auroc": float(full_layer_auroc),
        "f2_layer_aggregate_auroc": F2_LAYER_AGGREGATE,
        "best_head_minus_f2": float(best_auroc - F2_LAYER_AGGREGATE),
        "refute_threshold": REFUTE_THRESHOLD,
        "f2_locus_refuted": refuted,
        "verdict": (
            "REFUTED: a small subset of L19 heads carries the signal"
            if refuted
            else "NOT_REFUTED: no single head beats the layer aggregate by enough"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(
        f"best head {best_idx} AUROC={best_auroc:.4f} "
        f"vs F-2 aggregate {F2_LAYER_AGGREGATE:.4f} "
        f"(full-layer probe {full_layer_auroc:.4f}) refuted={refuted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())