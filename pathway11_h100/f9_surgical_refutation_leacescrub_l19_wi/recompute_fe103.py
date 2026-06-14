"""P11-FE103 — F-9 surgical refutation via LEACE-scrubbed L19 inside CoE-60.

F-9 claims CoE-60 (multi-layer Chain-of-Embeddings trajectory probe) is
nothing more than a re-expression of the single-layer L19 DoM direction. This
script provides the counterfactual that probe-vs-probe comparison cannot: it
LEACE-erases the linear K=1-correctness content from the L19 slice of the
layer trajectory, recomputes the CoE-60 trajectory features from the scrubbed
trajectory, refits the CoE-60 probe out-of-fold, and reports its AUROC.

Reading:
  - If CoE-60 AUROC collapses toward ~0.5 once L19's linear correctness signal
    is erased, F-9 is corroborated (CoE-60 was riding on L19).
  - If CoE-60 retains AUROC >= 0.70, it draws on multi-layer information not
    contained in L19's linear correctness direction, and F-9's framing is wrong.

All erasure and probe fitting is per-fold (no test leakage). The LEACE fit and
DoM direction for scrubbing use only the training fold's K=1 labels.
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
# Multi-layer trajectory cache required to (re)compute CoE-60 features.
TRAJ_CACHE = ROOT / "pathway11_h100/coe_trajectory/cache/m15b_alllayers.npz"
# Fallback single-layer L19 cache, used only to cross-check labels if present.
L19_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/coe_leace_refutation/results.json"

N_FOLDS = 5
SEED = 9999
L19_INDEX = 19  # index of the L19 slice within the layer trajectory


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
    """Closed-form LEACE oblique projection params for one binary concept."""
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
    w = np.linalg.solve(Sigma_ridge, d_vec)
    return mu, d_vec, w


def apply_leace(X: np.ndarray, mu: np.ndarray, d_vec: np.ndarray, w: np.ndarray) -> np.ndarray:
    denom = float(d_vec @ w)
    if abs(denom) < 1e-12:
        return X.copy()
    coeff = ((X - mu) @ w) / denom
    return X - np.outer(coeff, d_vec)


def coe_features(H: np.ndarray) -> np.ndarray:
    """CoE-60 trajectory features from a (N, L, D) layer-state tensor.

    Per Chain-of-Embeddings: per-layer hidden-state norms, consecutive-layer
    update magnitudes, and consecutive-layer cosines. Concatenated across the
    L layers, this yields the multi-layer trajectory descriptor that the CoE-60
    probe consumes.
    """
    norms = np.linalg.norm(H, axis=2)                      # (N, L)
    diffs = H[:, 1:, :] - H[:, :-1, :]                     # (N, L-1, D)
    diff_norms = np.linalg.norm(diffs, axis=2)             # (N, L-1)
    num = (H[:, 1:, :] * H[:, :-1, :]).sum(axis=2)         # (N, L-1)
    denom = norms[:, 1:] * norms[:, :-1] + 1e-8
    cos = num / denom                                      # (N, L-1)
    return np.concatenate([norms, diff_norms, cos], axis=1)


def standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-8
    out = [(train - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out


def logreg_oof_scores(feat: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """OOF decision scores from a logistic-regression probe on CoE features."""
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr_s, Xte_s = standardize(feat[train_mask], feat[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr_s, y[train_mask])
        scores[test_idx] = clf.decision_function(Xte_s)
    return scores


def main() -> int:
    if not TRAJ_CACHE.exists():
        print("MISSING_REGEN_INPUT", TRAJ_CACHE, file=sys.stderr)
        return 2

    blob = np.load(TRAJ_CACHE)
    if "hidden" not in blob or "correct" not in blob:
        print("MISSING_REGEN_INPUT", TRAJ_CACHE, "(missing keys)", file=sys.stderr)
        return 2

    H = blob["hidden"].astype(np.float64)   # (N, L, D)
    y = blob["correct"].astype(bool)        # (N,)
    if H.ndim != 3 or H.shape[0] != y.shape[0]:
        print("MISSING_REGEN_INPUT", TRAJ_CACHE, "(bad shape)", file=sys.stderr)
        return 2
    n, L, d = H.shape
    if not (0 <= L19_INDEX < L):
        print("MISSING_REGEN_INPUT", TRAJ_CACHE, "(no L19 slice)", file=sys.stderr)
        return 2

    # Optional label cross-check against the canonical single-layer cache.
    label_match = None
    if L19_CACHE.exists():
        l19 = np.load(L19_CACHE)
        if "correct" in l19 and l19["correct"].shape[0] == n:
            label_match = bool(np.array_equal(l19["correct"].astype(bool), y))

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # --- Baseline CoE-60 (intact trajectory) ---
    feat_baseline = coe_features(H)
    coe_scores_baseline = logreg_oof_scores(feat_baseline, y, folds)
    auroc_coe_baseline = auroc(coe_scores_baseline, y)

    # --- Scrubbed CoE-60: per-fold LEACE erase the L19 slice, recompute CoE ---
    # We erase the L19 linear correctness direction per train fold, applying the
    # fold's projection to its test rows, then recompute CoE features from the
    # scrubbed trajectory and refit the probe out-of-fold.
    H_scrubbed = H.copy()
    L19 = H[:, L19_INDEX, :]
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        mu, d_vec, w = fit_leace(L19[train_mask], y[train_mask])
        H_scrubbed[test_idx, L19_INDEX, :] = apply_leace(L19[test_idx], mu, d_vec, w)
        H_scrubbed[train_mask, L19_INDEX, :] = apply_leace(L19[train_mask], mu, d_vec, w)

    feat_scrubbed = coe_features(H_scrubbed)
    coe_scores_scrubbed = logreg_oof_scores(feat_scrubbed, y, folds)
    auroc_coe_scrubbed = auroc(coe_scores_scrubbed, y)

    # --- Reference: single-layer L19 DoM AUROC, intact vs scrubbed (OOF) ---
    dom_intact = np.zeros(n, dtype=np.float64)
    dom_scrubbed = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]
        d_raw = L19[train_mask][ytr].mean(0) - L19[train_mask][~ytr].mean(0)
        dom_intact[test_idx] = L19[test_idx] @ d_raw
        L19s = H_scrubbed[:, L19_INDEX, :]
        d_s = L19s[train_mask][ytr].mean(0) - L19s[train_mask][~ytr].mean(0)
        dom_scrubbed[test_idx] = L19s[test_idx] @ d_s

    out = {
        "experiment": "P11-FE103",
        "n": int(n),
        "n_layers": int(L),
        "dim": int(d),
        "l19_index": int(L19_INDEX),
        "n_coe_features": int(feat_baseline.shape[1]),
        "label_match_with_l19_cache": label_match,
        "auroc_coe60_baseline_oof": float(auroc_coe_baseline),
        "auroc_coe60_l19scrubbed_oof": float(auroc_coe_scrubbed),
        "auroc_coe60_delta": float(auroc_coe_baseline - auroc_coe_scrubbed),
        "auroc_l19_dom_intact_oof": float(auroc(dom_intact, y)),
        "auroc_l19_dom_scrubbed_oof": float(auroc(dom_scrubbed, y)),
        "f9_verdict": (
            "F9_CORROBORATED_coe_collapses"
            if auroc_coe_scrubbed < 0.60
            else (
                "F9_REFUTED_coe_retains_multilayer_signal"
                if auroc_coe_scrubbed >= 0.70
                else "AMBIGUOUS_0.60_to_0.70"
            )
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())