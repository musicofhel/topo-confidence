"""P11-FE415 — Non-linear MLP probe vs F-2's linear DoM on L19 prefill.

Trains a 3-layer MLP (1536 -> 256 -> 128 -> 1, ReLU, BCE/log-loss) on cached
pathway11_h100 L19 prefill activations to predict K=1 MATH-500 correctness,
scored 5-fold out-of-fold. Compares OOF AUROC and positive likelihood ratio
LR+ at FPR=0.01 against the linear mass-mean DoM direction (F-2, 0.7731 AUROC),
computed OOF in the same folds for a fair head-to-head.

Falsification (per TruthPrInt §3.1 motivation): the 'single linear direction is
the signal' framing is falsified if the MLP lifts AUROC above 0.80 OR delivers
an LR+@FPR=0.01 ratio above 3x relative to linear DoM.
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
OUT_JSON = ROOT / "pathway11_h100/mlp_probe/results.json"

N_FOLDS = 5
SEED = 9999
HIDDEN = (256, 128)
F2_LINEAR_AUROC = 0.7731
FPR_TARGET = 0.01

AUROC_FALSIFY = 0.80
LRPLUS_RATIO_FALSIFY = 3.0


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


def lrplus_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> dict:
    """LR+ = TPR / FPR at the highest threshold whose FPR does not exceed target.

    Picks a threshold so that at most ceil(fpr_target * n_neg) negatives score
    above it, then reads off TPR. Returns realized FPR/TPR for honesty since the
    grid is discrete with ~250 negatives.
    """
    y = labels.astype(bool)
    pos = scores[y]
    neg = scores[~y]
    n_pos = len(pos); n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return {"lrplus": float("nan"), "fpr": float("nan"), "tpr": float("nan"),
                "threshold": float("nan")}
    k = max(1, int(np.ceil(fpr_target * n_neg)))
    neg_sorted = np.sort(neg)[::-1]
    # threshold just below the k-th largest negative -> exactly k negatives above
    thr = neg_sorted[k - 1]
    fp = int((neg > thr).sum())
    tp = int((pos > thr).sum())
    # handle ties at threshold by counting >= would over-count; use strict > and
    # fall back if no negatives strictly above (degenerate ties)
    if fp == 0:
        fp = int((neg >= thr).sum())
        tp = int((pos >= thr).sum())
    fpr = fp / n_neg
    tpr = tp / n_pos
    lrplus = (tpr / fpr) if fpr > 0 else float("inf")
    return {"lrplus": float(lrplus), "fpr": float(fpr), "tpr": float(tpr),
            "threshold": float(thr), "n_pos": n_pos, "n_neg": n_neg}


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    try:
        from sklearn.neural_network import MLPClassifier
    except Exception as exc:  # pragma: no cover
        print("MISSING_REGEN_INPUT sklearn", exc, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_cached = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    n = len(y)
    folds = stratified_kfold(y, N_FOLDS, SEED)

    mlp_oof = np.zeros(n, dtype=np.float64)
    dom_oof = np.zeros(n, dtype=np.float64)

    for fi, test_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # standardize on train statistics
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd < 1e-8] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd

        clf = MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=False,
            random_state=SEED + fi,
        )
        clf.fit(Xtr_s, ytr.astype(int))
        # P(correct); decision = logit of positive class probability
        proba = clf.predict_proba(Xte_s)
        pos_col = list(clf.classes_).index(1)
        mlp_oof[test_idx] = proba[:, pos_col]

        # linear mass-mean DoM, OOF in the same folds, on raw activations
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        dom_oof[test_idx] = Xte @ d_vec

    mlp_auroc = auroc(mlp_oof, y)
    dom_oof_auroc = auroc(dom_oof, y)
    dom_cached_auroc = auroc(dom_cached, y)

    mlp_lr = lrplus_at_fpr(mlp_oof, y, FPR_TARGET)
    dom_lr = lrplus_at_fpr(dom_oof, y, FPR_TARGET)

    lr_ratio = (mlp_lr["lrplus"] / dom_lr["lrplus"]
                if dom_lr["lrplus"] > 0 and np.isfinite(dom_lr["lrplus"])
                else float("inf"))

    falsifies = bool(mlp_auroc > AUROC_FALSIFY or lr_ratio > LRPLUS_RATIO_FALSIFY)

    out = {
        "experiment": "P11-FE415",
        "n": n,
        "n_correct": int(y.sum()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "mlp_hidden_layers": list(HIDDEN),
        "mlp_auroc_oof": float(mlp_auroc),
        "linear_dom_auroc_oof": float(dom_oof_auroc),
        "linear_dom_auroc_cached": float(dom_cached_auroc),
        "f2_reference_auroc": F2_LINEAR_AUROC,
        "fpr_target": FPR_TARGET,
        "mlp_lrplus": mlp_lr,
        "linear_dom_lrplus": dom_lr,
        "lrplus_ratio_mlp_over_dom": float(lr_ratio),
        "auroc_falsify_threshold": AUROC_FALSIFY,
        "lrplus_ratio_falsify_threshold": LRPLUS_RATIO_FALSIFY,
        "falsifies_single_linear_direction": falsifies,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())