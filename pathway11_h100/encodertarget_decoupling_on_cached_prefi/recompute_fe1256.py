"""P11-FE1256 — Encoder-Target Decoupling on cached prefill geometry.

Paper claim under test: a *foreign* encoder predicts a target model's
correctness as well as (or better than) the target's own states. We have both
Qwen-1.5B and Qwen-7B L19 prefill activations cached over the same 500 aligned
MATH-500 problems, so this is a zero-new-forward-pass test.

For each (encoder, target) pair we train an L2-regularized logistic regression
on the ENCODER's 1536-d prefill features and predict the TARGET model's
correctness labels, scored out-of-fold (5-fold stratified on the target label).
Four cells: in-family 1.5B->1.5B and 7B->7B (diagonal), and the decoupled
cross-family 1.5B->7B and 7B->1.5B (off-diagonal). Each is compared against the
"free" out-of-family length(+logprob) baseline — features that cost no extra
forward pass — to decide whether cross-family geometry beats trivial cues.

A cross-family AUROC that matches or exceeds both the in-family diagonal and the
free baseline would resurrect the REFUTED geometry-portable-signal premise and
challenge F-2's single-direction model-idiosyncrasy framing.
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
CACHE_15B = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
CACHE_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/encoder_target_decoupling/results.json"

SEED = 9999
N_FOLDS = 5
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def oof_lr_scores(X, y, C, seed=SEED, n_folds=N_FOLDS):
    """Out-of-fold L2-LR decision scores. Folds are stratified on y (target)."""
    y = y.astype(bool)
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=2000
        ).fit(Xtr, y[tr])
        oof[te] = clf.decision_function(Xte)
    return oof


def best_C_auroc(X, y, seed=SEED):
    """Sweep C, return (best_C, best_auroc, per_C dict) on OOF scores."""
    per_c = {}
    best = (None, -1.0)
    for C in C_GRID:
        a = auroc(oof_lr_scores(X, y, C, seed=seed), y)
        per_c[f"C={C}"] = a
        if not np.isnan(a) and a > best[1]:
            best = (C, a)
    return best[0], best[1], per_c


def load_cache(path):
    blob = np.load(path)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    # logprob is optional in the schema; include it in the free baseline if present.
    logprob = None
    for key in ("logprob", "mean_logprob", "seq_logprob", "avg_logprob"):
        if key in blob.files:
            logprob = blob[key].astype(np.float64)
            break
    return X, y, seq_len, logprob


def free_baseline_auroc(seq_len, logprob, y, seed=SEED):
    """OOF 1-2 feature LR over free cues (length, optional logprob)."""
    cols = [seq_len]
    if logprob is not None:
        cols.append(logprob)
    F = np.column_stack(cols)
    return auroc(oof_lr_scores(F, y, C=1.0, seed=seed), y), F.shape[1]


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2
    if not CACHE_7B.exists():
        print("MISSING_REGEN_INPUT", CACHE_7B, file=sys.stderr)
        return 2

    X15, y15, len15, lp15 = load_cache(CACHE_15B)
    X7, y7, len7, lp7 = load_cache(CACHE_7B)

    if X15.shape[0] != X7.shape[0]:
        print(
            f"MISALIGNED_CACHES {X15.shape[0]} vs {X7.shape[0]}",
            file=sys.stderr,
        )
        return 2
    assert X15.shape[1] == 1536, X15.shape
    assert X7.shape[1] == 1536, X7.shape

    out = {"experiment": "P11-FE1256", "n": int(X15.shape[0])}
    out["base_rates"] = {
        "y_1.5b_acc": float(y15.mean()),
        "y_7b_acc": float(y7.mean()),
    }

    # Four encoder->target cells. Diagonal = in-family, off-diagonal = decoupled.
    cells = {
        "enc1.5b_tgt1.5b": (X15, y15),  # in-family diagonal
        "enc7b_tgt7b": (X7, y7),        # in-family diagonal
        "enc1.5b_tgt7b": (X15, y7),     # decoupled: foreign encoder -> 7B labels
        "enc7b_tgt1.5b": (X7, y15),     # decoupled: foreign encoder -> 1.5B labels
    }
    out["geometry"] = {}
    for name, (Xenc, ytgt) in cells.items():
        bC, bA, per_c = best_C_auroc(Xenc, ytgt)
        out["geometry"][name] = {
            "best_C": bC,
            "auroc_oof": bA,
            "per_C": per_c,
        }

    # Free out-of-family baselines: encoder's length(+logprob) -> target labels.
    a_15to7, nf_15 = free_baseline_auroc(len15, lp15, y7)
    a_7to15, nf_7 = free_baseline_auroc(len7, lp7, y15)
    a_15to15, _ = free_baseline_auroc(len15, lp15, y15)
    a_7to7, _ = free_baseline_auroc(len7, lp7, y7)
    out["free_baseline"] = {
        "n_features": int(nf_15),
        "logprob_available": bool(lp15 is not None and lp7 is not None),
        "len1.5b_tgt7b": a_15to7,
        "len7b_tgt1.5b": a_7to15,
        "len1.5b_tgt1.5b": a_15to15,
        "len7b_tgt7b": a_7to7,
    }

    # Headline deltas: does decoupled geometry beat (a) the free out-of-family
    # baseline and (b) the target's own in-family geometry?
    g = out["geometry"]
    out["verdict"] = {
        "cross_1.5b_to_7b_vs_free": g["enc1.5b_tgt7b"]["auroc_oof"] - a_15to7,
        "cross_7b_to_1.5b_vs_free": g["enc7b_tgt1.5b"]["auroc_oof"] - a_7to15,
        "cross_1.5b_to_7b_vs_own_7b": (
            g["enc1.5b_tgt7b"]["auroc_oof"] - g["enc7b_tgt7b"]["auroc_oof"]
        ),
        "cross_7b_to_1.5b_vs_own_1.5b": (
            g["enc7b_tgt1.5b"]["auroc_oof"] - g["enc1.5b_tgt1.5b"]["auroc_oof"]
        ),
    }
    v = out["verdict"]
    out["portable_signal_supported"] = bool(
        v["cross_1.5b_to_7b_vs_free"] > 0
        and v["cross_7b_to_1.5b_vs_free"] > 0
        and g["enc1.5b_tgt7b"]["auroc_oof"] >= 0.60
        and g["enc7b_tgt1.5b"]["auroc_oof"] >= 0.60
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())