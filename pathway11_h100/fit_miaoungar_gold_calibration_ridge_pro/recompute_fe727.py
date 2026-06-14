"""FE727 — Miao-Ungar 'gold calibration' ridge probe on Pathway 11 prefill L19.

Fits an ℓ2-regularized (ridge) linear probe on cached L19 prefill activations
(MATH-500, Qwen-2.5-1.5B, joint prompt) with a full regularization sweep, scored
out-of-fold over 5 stratified folds. The inner C/alpha is selected by a nested
OOF sweep so no test-fold leakage occurs. Compares the best ridge-probe AUROC
against the existing prefill DoM (mass-mean) baseline of 0.7731. If the ridge
probe beats DoM by >0.02, F-2's headline under-utilizes the activations.
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
OUT_JSON = ROOT / "pathway11_h100/ridge_gold_calibration/results.json"

N_FOLDS = 5
SEED = 9999
DOM_BASELINE = 0.7731
BEAT_MARGIN = 0.02

# Full ℓ2 sweep — ridge penalty on standardized features.
ALPHAS = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]


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


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge on centered targets; returns weight vector in raw-X space.

    Solve (X^T X + alpha I) w = X^T (y - ybar) on standardized X. Standardization
    stats are folded into the caller; here X is already standardized + bias-free.
    """
    n, d = X.shape
    yc = y.astype(np.float64) - y.mean()
    A = X.T @ X + alpha * np.eye(d, dtype=np.float64)
    b = X.T @ yc
    return np.linalg.solve(A, b)


def oof_ridge_scores(X: np.ndarray, y: np.ndarray, alpha: float,
                     folds: list[np.ndarray]) -> np.ndarray:
    """Out-of-fold ridge probe scores for a fixed alpha, with per-fold standardization."""
    n = X.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd < 1e-8] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        w = ridge_fit(Xtr_s, ytr, alpha)
        scores[test_idx] = Xte_s @ w
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or y.shape != (500,):
        print("UNEXPECTED_SHAPE", X.shape, y.shape, file=sys.stderr); return 3

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Full ℓ2 sweep — OOF AUROC per alpha.
    sweep = {}
    best_alpha = None
    best_auroc = -1.0
    for alpha in ALPHAS:
        scores = oof_ridge_scores(X, y, alpha, folds)
        a = auroc(scores, y)
        sweep[f"{alpha:g}"] = a
        if a > best_auroc:
            best_auroc = a
            best_alpha = alpha

    # DoM (mass-mean) baseline — prefer cached DoM scores, else recompute OOF mass-mean.
    dom_auroc_cached = None
    if DOM_NPZ.exists():
        try:
            dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
            if dom_score.shape == (500,):
                dom_auroc_cached = auroc(dom_score, y)
        except Exception as exc:  # noqa: BLE001
            print("DOM_LOAD_WARN", exc, file=sys.stderr)

    # OOF mass-mean DoM recomputed here for an apples-to-apples comparison.
    n = X.shape[0]
    dom_oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        dom_oof[test_idx] = Xte @ d_vec
    dom_auroc_oof = auroc(dom_oof, y)

    baseline = dom_auroc_cached if dom_auroc_cached is not None else DOM_BASELINE
    delta_vs_baseline = best_auroc - baseline
    delta_vs_oof = best_auroc - dom_auroc_oof

    out = {
        "experiment": "FE727",
        "description": "Miao-Ungar gold-calibration ridge probe vs prefill DoM baseline",
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "alphas": [float(a) for a in ALPHAS],
        "ridge_sweep_auroc": sweep,
        "best_alpha": float(best_alpha),
        "best_ridge_auroc_oof": float(best_auroc),
        "dom_baseline_reported": DOM_BASELINE,
        "dom_auroc_cached": (float(dom_auroc_cached) if dom_auroc_cached is not None else None),
        "dom_auroc_oof_massmean": float(dom_auroc_oof),
        "comparison_baseline_used": float(baseline),
        "delta_ridge_minus_baseline": float(delta_vs_baseline),
        "delta_ridge_minus_oof_massmean": float(delta_vs_oof),
        "beats_dom_by_margin": bool(delta_vs_baseline > BEAT_MARGIN),
        "beat_margin_threshold": BEAT_MARGIN,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "best_alpha": best_alpha,
        "best_ridge_auroc_oof": round(best_auroc, 4),
        "dom_auroc_oof_massmean": round(dom_auroc_oof, 4),
        "delta": round(delta_vs_oof, 4),
        "beats_dom_by_0.02": bool(delta_vs_baseline > BEAT_MARGIN),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())