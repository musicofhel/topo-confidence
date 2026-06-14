"""P11-FE137 — Causal-inner-product (Mahalanobis-whitened) refit of the prefill
L19 correctness probe.

Theorem 2.2 of arXiv:2311.03658 (Linear Representation Hypothesis) states that a
measurement representation is logit-linear in the causal inner product
λᵀγ̄_W, i.e. coordinates whitened by the (centered) feature covariance. Our raw
DoM probe (AUROC 0.7731) uses Euclidean projections, not the causal inner
product the paper prescribes. This script whitens the L19 prefill activations
with Cov(γ)^{-1/2} (per-fold, train-fit) and refits a 5-fold OOF logistic
regression on Qwen-2.5-1.5B MATH-500 correctness labels, comparing AUROC and
Brier score against vanilla DoM.

Three discriminative outcomes:
  (a) AUROC > 0.80   — LRH frame strengthens F-2, explains F-9 redundancy.
  (b) AUROC within ±0.01 of 0.7731 — vanilla DoM already near-optimal.
  (c) AUROC drops materially — correctness is not LRH-linear at L19.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/causal_inner_product/results.json"

SEED = 9999
N_FOLDS = 5
VANILLA_DOM_AUROC = 0.7731
RIDGE_REL = 1e-3  # relative ridge on covariance, as in the LEACE convention


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    y = labels.astype(np.float64)
    return float(np.mean((probs - y) ** 2))


def whitener(X_train: np.ndarray, ridge_rel: float = RIDGE_REL):
    """Return (mu, W) where W = Cov(γ)^{-1/2} computed on centered train data."""
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = ridge_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Sigma.dtype)
    # Symmetric inverse square root via eigendecomposition (Sigma is SPD).
    evals, evecs = np.linalg.eigh(Sigma_ridge)
    evals = np.clip(evals, 1e-12, None)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu, inv_sqrt


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), (X.shape, y.shape)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (500,), dom_score.shape

    n = len(y)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Whitened-coordinate OOF logistic regression.
    oof_white_score = np.zeros(n, dtype=np.float64)
    oof_white_prob = np.zeros(n, dtype=np.float64)
    # Euclidean (raw-feature) logistic regression control, same folds.
    oof_euclid_score = np.zeros(n, dtype=np.float64)
    # DoM raw projection, refit OOF for a like-for-like internal baseline.
    oof_dom_score = np.zeros(n, dtype=np.float64)

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]

        # --- causal inner product: whiten then logistic regression ---
        mu, W = whitener(Xtr)
        Ztr = (Xtr - mu) @ W
        Zte = (Xte - mu) @ W
        clf_w = LogisticRegression(max_iter=2000, C=1.0)
        clf_w.fit(Ztr, ytr)
        oof_white_score[test_idx] = clf_w.decision_function(Zte)
        oof_white_prob[test_idx] = clf_w.predict_proba(Zte)[:, 1]

        # --- Euclidean logistic regression control (no whitening) ---
        clf_e = LogisticRegression(max_iter=2000, C=1.0)
        clf_e.fit(Xtr - mu, ytr)
        oof_euclid_score[test_idx] = clf_e.decision_function(Xte - mu)

        # --- vanilla DoM (difference-of-means Euclidean projection) ---
        d_raw = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof_dom_score[test_idx] = Xte @ d_raw

    auroc_white = auroc(oof_white_score, y)
    auroc_euclid = auroc(oof_euclid_score, y)
    auroc_dom_oof = auroc(oof_dom_score, y)
    auroc_dom_cached = auroc(dom_score, y)
    brier_white = brier(oof_white_prob, y)

    delta_vs_vanilla = auroc_white - VANILLA_DOM_AUROC
    if auroc_white > 0.80:
        verdict = "a_lrh_strengthens"
    elif abs(delta_vs_vanilla) <= 0.01:
        verdict = "b_neutral"
    else:
        verdict = "c_not_lrh_linear"

    out = {
        "experiment": "P11-FE137",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "ridge_rel": RIDGE_REL,
        "vanilla_dom_auroc_reference": VANILLA_DOM_AUROC,
        "auroc_whitened_logreg_oof": auroc_white,
        "auroc_euclidean_logreg_oof": auroc_euclid,
        "auroc_dom_refit_oof": auroc_dom_oof,
        "auroc_dom_cached_score": auroc_dom_cached,
        "brier_whitened_logreg_oof": brier_white,
        "delta_auroc_vs_vanilla": float(delta_vs_vanilla),
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())