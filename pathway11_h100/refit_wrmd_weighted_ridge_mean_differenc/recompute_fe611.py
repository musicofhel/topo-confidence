"""P11-FE611 — Weighted Ridge Mean Difference (WRMD) refit on cached prefill L19.

Paper 2512.16602 (García-Ferrero et al., LREC 2026) augments the plain
Difference-of-Means (DoM / WMD) readout with Tikhonov-regularized covariance
whitening: instead of scoring with d = μ_pos - μ_neg, they score with
w = (Σ + λI)^-1 (μ_pos - μ_neg). On their refusal task WRMD beat plain WMD by
~10pp at λ=1e-2.

Refutation test for F-2: does covariance whitening lift OOF AUROC above the
plain-DoM headline (0.7731 for Qwen-2.5-1.5B)? We refit WRMD per fold (μ, Σ, and
the mean-difference are estimated on the training fold only, then applied to the
held-out fold), sweep λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1e0}, and report 5-fold OOF
AUROC for both Qwen-2.5-1.5B and (if its cache is present) Qwen-2.5-7B against
the plain-DoM baseline. λ is scaled relative to trace(Σ)/d so it is comparable
across models of different activation scale, matching the LEACE convention.
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
CACHE_15B = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
CACHE_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/wrmd_whitening/results.json"

N_FOLDS = 5
SEED = 9999
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
F2_BASELINE_15B = 0.7731


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


def fit_wrmd(X_train: np.ndarray, y_train: np.ndarray, lam_rel: float):
    """Whitened mean-difference direction w = (Σ + αI)^-1 (μ_pos - μ_neg).

    α = lam_rel * trace(Σ)/d so λ is comparable across activation scales.
    """
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = lam_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    mu_pos = X_train[y_train].mean(axis=0)
    mu_neg = X_train[~y_train].mean(axis=0)
    d_vec = mu_pos - mu_neg
    w = np.linalg.solve(Sigma_ridge, d_vec)
    return w, d_vec, alpha


def run_model(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)

    # Plain-DoM (WMD) OOF baseline refit per fold.
    plain_oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        d_vec = X[train_mask][y[train_mask]].mean(0) - X[train_mask][~y[train_mask]].mean(0)
        plain_oof[test_idx] = X[test_idx] @ d_vec
    plain_auroc = auroc(plain_oof, y)

    # WRMD OOF AUROC across the λ sweep.
    sweep = {}
    best_lam, best_auroc = None, -1.0
    for lam in LAMBDAS:
        wrmd_oof = np.zeros(n, dtype=np.float64)
        for test_idx in folds:
            train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
            w, _, _ = fit_wrmd(X[train_mask], y[train_mask], lam)
            wrmd_oof[test_idx] = X[test_idx] @ w
        a = auroc(wrmd_oof, y)
        sweep[f"{lam:g}"] = a
        if a > best_auroc:
            best_auroc, best_lam = a, lam

    return {
        "model": name,
        "n": int(n),
        "n_correct": int(y.sum()),
        "auroc_plain_dom_oof": float(plain_auroc),
        "wrmd_lambda_sweep_oof": {k: float(v) for k, v in sweep.items()},
        "best_lambda": float(best_lam),
        "best_wrmd_auroc_oof": float(best_auroc),
        "lift_over_plain_dom": float(best_auroc - plain_auroc),
    }


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr); return 2

    blob = np.load(CACHE_15B)
    X15 = blob["prefill"].astype(np.float64)
    y15 = blob["correct"].astype(bool)
    assert X15.shape[1] == 1536 and y15.shape[0] == X15.shape[0]

    results = {"experiment": "P11-FE611", "f2_baseline_15b": F2_BASELINE_15B, "models": {}}
    r15 = run_model("qwen2.5-1.5b", X15, y15)
    results["models"]["qwen2.5-1.5b"] = r15

    if CACHE_7B.exists():
        blob7 = np.load(CACHE_7B)
        X7 = blob7["prefill"].astype(np.float64)
        y7 = blob7["correct"].astype(bool)
        results["models"]["qwen2.5-7b"] = run_model("qwen2.5-7b", X7, y7)
    else:
        results["models"]["qwen2.5-7b"] = {"status": "cache_absent", "path": str(CACHE_7B)}

    # Refutation verdict: does whitening clear the F-2 headline by >= 0.01?
    results["refutation"] = {
        "threshold_lift": 0.01,
        "whitening_lifts_15b": bool(r15["best_wrmd_auroc_oof"] - F2_BASELINE_15B >= 0.01),
        "best_wrmd_auroc_15b": r15["best_wrmd_auroc_oof"],
        "delta_vs_f2_headline": float(r15["best_wrmd_auroc_oof"] - F2_BASELINE_15B),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())