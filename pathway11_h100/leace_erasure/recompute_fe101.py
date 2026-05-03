"""FE101 — LEACE linear-erasure null test for F-2.

Per-fold LEACE: fit μ, Σ, DoM on train; apply to test; refit DoM on erased
train fold; score on erased test fold; aggregate AUROC over 5 OOF folds.

If F-2's correctness signal lives in a linear subspace of L19 prefill,
LEACE should collapse the OOF AUROC from 0.7731 to ~0.5 (within sampling
noise). A residual AUROC > 0.55 would indicate either (a) numerical
incompleteness of LEACE under ridge stabilization, or (b) a non-linear
correctness signal the DoM probe was failing to pick up cleanly.

Method (binary closed-form LEACE, equivalent to Belrose-Smith for d-dim
features and 1-dim binary label):
  1. μ, Σ_ridge from train fold (Σ_ridge = X_c^T X_c / (n-1) + α I)
  2. d = μ_pos − μ_neg     (train-fold DoM direction)
  3. w = Σ_ridge^(-1) d    (whitened concept direction)
  4. Erase X[i]: X_erased[i] = X[i] − ((X[i] − μ) · w / (d · w)) · d
  5. Refit DoM on erased train (zero by construction); score erased test.

Reference: Belrose et al., "LEACE: perfect linear concept erasure in
closed form" (NeurIPS 2023, 2306.03819).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/leace_erasure/results.json"

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


def fit_leace(X_train: np.ndarray, y_train: np.ndarray, alpha_rel: float = 1e-3):
    """Returns (mu, d, w) such that X_erased = X − ((X − μ) · w / (d · w)) · d."""
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = alpha_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    mu_pos = X_train[y_train].mean(axis=0)
    mu_neg = X_train[~y_train].mean(axis=0)
    d_vec = mu_pos - mu_neg
    # w = Σ_ridge^{-1} d_vec, computed via solve for stability.
    w = np.linalg.solve(Sigma_ridge, d_vec)
    return mu, d_vec, w, alpha


def apply_leace(X: np.ndarray, mu: np.ndarray, d_vec: np.ndarray, w: np.ndarray) -> np.ndarray:
    denom = float(d_vec @ w)
    if abs(denom) < 1e-12:
        return X.copy()
    coeff = ((X - mu) @ w) / denom  # (n,)
    return X - np.outer(coeff, d_vec)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)  # higher precision for matrix solve
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)

    raw_scores = np.zeros(n, dtype=np.float64)
    erased_scores = np.zeros(n, dtype=np.float64)
    erasure_ratios = []  # ‖DoM_erased(test)‖ / ‖DoM_raw(test)‖ — sanity

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr, yte = y[train_mask], y[test_idx]

        # Raw DoM AUROC (train DoM scored on test) — anchor.
        d_raw = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        raw_scores[test_idx] = Xte @ d_raw

        # LEACE eraser fit on train, applied to both
        mu, d_vec, w, alpha = fit_leace(Xtr, ytr, alpha_rel=1e-3)
        Xtr_e = apply_leace(Xtr, mu, d_vec, w)
        Xte_e = apply_leace(Xte, mu, d_vec, w)

        d_erased = Xtr_e[ytr].mean(axis=0) - Xtr_e[~ytr].mean(axis=0)
        # Should be ~0 in train space — verify magnitude
        erasure_ratios.append(
            float(np.linalg.norm(d_erased) / (np.linalg.norm(d_raw) + 1e-12))
        )

        erased_scores[test_idx] = Xte_e @ d_erased

    auroc_raw_oof = auroc(raw_scores, y)
    auroc_erased_oof = auroc(erased_scores, y) if np.std(erased_scores) > 1e-12 else 0.5

    out = {
        "n": n,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "alpha_rel": 1e-3,
        "auroc_raw_oof": float(auroc_raw_oof),
        "auroc_erased_oof": float(auroc_erased_oof),
        "auroc_collapse_pp": float(auroc_raw_oof - auroc_erased_oof),
        "erasure_ratio_mean": float(np.mean(erasure_ratios)),
        "erasure_ratio_max": float(np.max(erasure_ratios)),
        "f2_linear_holds_threshold": 0.55,
        "f2_linear_holds": auroc_erased_oof < 0.55,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"auroc_raw_oof={out['auroc_raw_oof']:.10f}")
    print(f"auroc_erased_oof={out['auroc_erased_oof']:.10f}")
    print(f"auroc_collapse_pp={out['auroc_collapse_pp']:.10f}")
    print(f"erasure_ratio_mean={out['erasure_ratio_mean']:.10f}")
    print(f"erasure_ratio_max={out['erasure_ratio_max']:.10f}")
    print(f"f2_linear_holds={int(out['f2_linear_holds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
