"""P11-FE1046 — Direction-suppression analysis: does the DoM-orthogonal residual
carry independent linear correctness signal?

Treats DoM as the 'shared direction μ' from Prop. 2's load-balancing framework.
We project DoM out of the L19 prefill activations (residualize each row to the
DoM-orthogonal subspace), then ask whether a logistic-regression probe fit on
that residual subspace still separates correct from incorrect answers. This
tests whether the single DoM direction captures ALL linear correctness info or
only the dominant 'shared' component — the analog of Prop. 2's 'retained energy'.

Pipeline (all 5-fold OOF, DoM + residual projection fit on train only):
  1. DoM-only score (Xte @ d_train)                          -> baseline AUROC
  2. Full-1536-d L2 logistic regression                       -> ceiling AUROC
  3. Residualize: remove unit-DoM component from X (train-fit
     direction), then L2 logistic regression on the residual  -> residual AUROC
  4. Retained-energy diagnostic: fraction of total activation
     variance carried by the DoM axis vs. the residual.

If the residual AUROC stays well above 0.5, the 0.7731->0.7928 gap between DoM
and the cov-spectrum ceiling plausibly lives in DoM-orthogonal directions.
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
OUT_JSON = ROOT / "pathway11_h100/direction_suppression/results.json"

N_FOLDS = 5
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
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


def residualize(X: np.ndarray, unit_dir: np.ndarray) -> np.ndarray:
    """Remove the component of each row along unit_dir (||unit_dir|| == 1)."""
    coeff = X @ unit_dir
    return X - np.outer(coeff, unit_dir)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print("MISSING_REGEN_INPUT", "sklearn:", e, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), (X.shape, y.shape)

    # Optional sanity: the canonical global DoM score, if present.
    dom_global_auroc = None
    if DOM_NPZ.exists():
        try:
            dom_global = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
            if dom_global.shape == (500,):
                dom_global_auroc = auroc(dom_global, y)
        except Exception:
            dom_global_auroc = None

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    dom_scores = np.zeros(n, dtype=np.float64)
    full_scores = np.zeros(n, dtype=np.float64)
    resid_scores = np.zeros(n, dtype=np.float64)

    energy_dom = []
    energy_resid = []

    C = 1e-3  # matches the L2-reg ceiling regime (FE882 best C)

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # --- supervised DoM direction (train-only) ---
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        nrm = np.linalg.norm(d_vec)
        unit_dir = d_vec / nrm if nrm > 1e-12 else d_vec
        dom_scores[test_idx] = Xte @ d_vec

        # --- retained-energy diagnostic (train activations, mean-centered) ---
        mu = Xtr.mean(axis=0)
        Xtr_c = Xtr - mu
        total_var = float((Xtr_c ** 2).sum())
        along = float(((Xtr_c @ unit_dir) ** 2).sum())
        energy_dom.append(along / total_var if total_var > 0 else float("nan"))
        energy_resid.append((total_var - along) / total_var if total_var > 0 else float("nan"))

        # --- full-1536-d L2 logistic regression (ceiling) ---
        scaler_f = StandardScaler().fit(Xtr)
        Xtr_fs = scaler_f.transform(Xtr)
        Xte_fs = scaler_f.transform(Xte)
        clf_full = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf_full.fit(Xtr_fs, ytr)
        full_scores[test_idx] = clf_full.decision_function(Xte_fs)

        # --- residualize onto DoM-orthogonal subspace (train-fit dir) ---
        Xtr_r = residualize(Xtr, unit_dir)
        Xte_r = residualize(Xte, unit_dir)
        scaler_r = StandardScaler().fit(Xtr_r)
        Xtr_rs = scaler_r.transform(Xtr_r)
        Xte_rs = scaler_r.transform(Xte_r)
        clf_resid = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf_resid.fit(Xtr_rs, ytr)
        resid_scores[test_idx] = clf_resid.decision_function(Xte_rs)

    out = {
        "experiment": "P11-FE1046",
        "n": int(n),
        "n_folds": N_FOLDS,
        "C": C,
        "auroc_dom_oof": auroc(dom_scores, y),
        "auroc_full_logreg_oof": auroc(full_scores, y),
        "auroc_residual_logreg_oof": auroc(resid_scores, y),
        "dom_global_auroc": dom_global_auroc,
        "retained_energy_dom_mean": float(np.nanmean(energy_dom)),
        "retained_energy_residual_mean": float(np.nanmean(energy_resid)),
        "gap_full_minus_dom": float(auroc(full_scores, y) - auroc(dom_scores, y)),
        "gap_residual_minus_chance": float(auroc(resid_scores, y) - 0.5),
        "interpretation": (
            "If auroc_residual_logreg_oof >> 0.5, the DoM-orthogonal subspace "
            "carries independent linear correctness signal; DoM does not capture "
            "ALL linear info, and the DoM->ceiling gap can live off the DoM axis."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())