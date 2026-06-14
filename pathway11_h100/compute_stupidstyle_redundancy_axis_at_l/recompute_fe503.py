"""P11-FE503 — STU-PID-style redundancy axis vs F-2 prefill DoM.

STU-PID frames "redundancy" as a steerable axis in residual space: the linear
direction that best explains how verbose / repetitive a generation is. We have
no separate redundancy annotation on the cached MATH-500 prefills, so we use the
generation length (`seq_len`) as the standard redundancy proxy and fit the
redundancy axis as the ridge-regression direction mapping L19 prefill activations
onto length. We then ask two questions that bear on whether F-2's headline
AUROC 0.7731 is partly a length/redundancy artifact:

  (1) cos(redundancy_axis, F-2 prefill DoM) — alignment of the two directions.
      The F-2 DoM axis is the class-mean difference of L19 prefill activations.
  (2) AUROC of the redundancy classifier (OOF projection onto the redundancy
      axis) for MATH-500 correctness.

Reframe trips if cos > 0.4 OR redundancy-classifier AUROC > 0.65.

Only L19 prefill activations are cached (the schema has no L20), so "L19/L20" is
served entirely by the L19 prefill tensor; this is noted in the output JSON.
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
OUT_JSON = ROOT / "pathway11_h100/stupid_redundancy/results.json"

N_FOLDS = 5
SEED = 9999
RIDGE_REL = 1e-3  # ridge alpha relative to trace/d, STU-PID-style regularization


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


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def fit_redundancy_axis(X: np.ndarray, target: np.ndarray, ridge_rel: float) -> np.ndarray:
    """Ridge regression of mean-centered activations onto a redundancy target.

    Returns the redundancy axis w (unit-norm-free) s.t. (X-mu) @ w approximates
    the centered target. This is the STU-PID redundancy direction.
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = ridge_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    t = target - target.mean()
    rhs = Xc.T @ t / max(n - 1, 1)
    w = np.linalg.solve(Sigma_ridge, rhs)
    return w


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536) and y.shape == (500,) and seq_len.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    # F-2 prefill DoM axis = class-mean difference of L19 prefill activations.
    dom_axis = X[y].mean(axis=0) - X[~y].mean(axis=0)

    # Full-data redundancy axis for the cosine alignment measurement.
    redund_axis = fit_redundancy_axis(X, seq_len, RIDGE_REL)
    cos_redund_dom = cos(redund_axis, dom_axis)

    # OOF redundancy classifier: fit axis on train (regress on seq_len), project
    # test activations, score correctness. Sign-align projection with length on
    # the training fold so AUROC sign is well-defined.
    n = len(y)
    oof_redund_proj = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(y, N_FOLDS, SEED)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        w = fit_redundancy_axis(Xtr, seq_len[train_mask], RIDGE_REL)
        mu = Xtr.mean(axis=0)
        oof_redund_proj[test_idx] = (Xte - mu) @ w

    auroc_redund = auroc(oof_redund_proj, y)
    # AUROC is direction-sensitive; report the orientation-agnostic strength too.
    auroc_redund_abs = max(auroc_redund, 1.0 - auroc_redund)

    # Reference numbers for the reframe judgement.
    auroc_dom = auroc(dom_score, y)
    auroc_len = auroc(seq_len, y)
    auroc_len_abs = max(auroc_len, 1.0 - auroc_len)

    reframe_triggered = bool(
        (not np.isnan(cos_redund_dom) and abs(cos_redund_dom) > 0.4)
        or (not np.isnan(auroc_redund_abs) and auroc_redund_abs > 0.65)
    )

    out = {
        "experiment": "P11-FE503",
        "note": "Only L19 prefill is cached; 'L19/L20' redundancy axis computed on L19 prefill. seq_len used as redundancy proxy (no STU-PID redundancy annotation cached).",
        "n": int(n),
        "ridge_rel": RIDGE_REL,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "cos_redundancy_dom": cos_redund_dom,
        "auroc_redundancy_classifier_oof": float(auroc_redund),
        "auroc_redundancy_classifier_oof_abs": float(auroc_redund_abs),
        "auroc_f2_dom_reference": float(auroc_dom),
        "auroc_seq_len_reference": float(auroc_len),
        "auroc_seq_len_reference_abs": float(auroc_len_abs),
        "reframe_thresholds": {"cos_gt": 0.4, "auroc_gt": 0.65},
        "reframe_triggered": reframe_triggered,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())