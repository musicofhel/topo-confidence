"""FE120 — IID mass-mean correction (Σ⁻¹·(μ⁺−μ⁻), Fisher LDA) for L19 prefill DoM.

Marks & Tegmark §5 establishes that raw DoM (μ⁺−μ⁻) is sub-optimal versus the
Σ⁻¹-corrected mass-mean direction under non-spherical covariance. F-1's
dimensional breathing implies anisotropic Σ, so the correction may lift AUROC
above the raw-DoM baseline of 0.7731.

This script fits the IID mass-mean direction per fold (ridge-stabilized Σ⁻¹·d)
on the cached L19 prefill activations and recomputes 5-fold OOF AUROC, comparing
against per-fold raw DoM scored on the same OOF splits.
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
OUT_JSON = ROOT / "pathway11_h100/mass_mean_correction/results.json"

N_FOLDS = 5
SEED = 9999
RAW_DOM_BASELINE = 0.7731

# Ridge stabilization sweep (relative to trace(Σ)/d) for the Σ⁻¹ solve.
ALPHA_REL_GRID = [1e-1, 1e-2, 1e-3, 1e-4]


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


def fit_raw_dom(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Raw mass-mean difference direction μ⁺ − μ⁻."""
    return X_train[y_train].mean(axis=0) - X_train[~y_train].mean(axis=0)


def fit_mass_mean(X_train: np.ndarray, y_train: np.ndarray, alpha_rel: float) -> np.ndarray:
    """IID mass-mean correction: Σ⁻¹·(μ⁺ − μ⁻), ridge-stabilized.

    Fisher-LDA-equivalent direction under a shared (IID) within-class covariance,
    here estimated as the pooled covariance about the global mean.
    """
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = alpha_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    d_vec = fit_raw_dom(X_train, y_train)
    return np.linalg.solve(Sigma_ridge, d_vec)


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

    raw_scores = np.zeros(n, dtype=np.float64)
    mm_scores = {a: np.zeros(n, dtype=np.float64) for a in ALPHA_REL_GRID}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        d_raw = fit_raw_dom(Xtr, ytr)
        raw_scores[test_idx] = Xte @ d_raw

        for alpha_rel in ALPHA_REL_GRID:
            w_mm = fit_mass_mean(Xtr, ytr, alpha_rel)
            mm_scores[alpha_rel][test_idx] = Xte @ w_mm

    auroc_raw = float(auroc(raw_scores, y))
    mm_aurocs = {f"{a:.0e}": float(auroc(mm_scores[a], y)) for a in ALPHA_REL_GRID}
    best_alpha = max(ALPHA_REL_GRID, key=lambda a: auroc(mm_scores[a], y))
    auroc_mm_best = float(auroc(mm_scores[best_alpha], y))

    out = {
        "experiment": "P11-FE120",
        "description": "IID mass-mean Σ⁻¹·(μ⁺−μ⁻) correction vs raw L19 prefill DoM, 5-fold OOF",
        "n": n,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "raw_dom_baseline_reference": RAW_DOM_BASELINE,
        "auroc_raw_oof": auroc_raw,
        "auroc_mass_mean_by_alpha_rel": mm_aurocs,
        "best_alpha_rel": f"{best_alpha:.0e}",
        "auroc_mass_mean_best_oof": auroc_mm_best,
        "lift_over_raw_oof": auroc_mm_best - auroc_raw,
        "lift_over_reference_baseline": auroc_mm_best - RAW_DOM_BASELINE,
        "verdict": (
            "MM_LIFTS" if auroc_mm_best - auroc_raw > 0.005
            else "MM_NEUTRAL" if auroc_mm_best - auroc_raw > -0.005
            else "MM_HURTS"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())