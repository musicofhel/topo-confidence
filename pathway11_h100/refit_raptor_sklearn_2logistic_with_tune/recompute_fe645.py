"""P11-FE645 — RAPTOR (ridge-tuned ℓ2-logistic) refit at L19 prefill vs F-2 DoM.

RAPTOR: sklearn ℓ2-logistic with the regularization strength λ (= 1/C) tuned on
an inner validation split, the fitted weight vector L2-normalized into a concept
vector, and test points scored by projection onto that vector. We compute the
5-fold OOF AUROC against K=1 MATH-500 correctness and compare head-to-head
against F-2's DoM 0.7731 (DoM is the unregularized, covariance-free special
case). Decision rule: if RAPTOR beats DoM by ≥0.02 OOF, F-2's headline is
conservative and should be revised.

We also report a best-layer (validation-averaged) sweep across L0–L27 when a
per-layer cache is available; the documented single-layer cache only carries
L19, so the sweep degrades to "unavailable" rather than failing when the
all-layer NPZ is absent.
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
# Optional per-layer cache for the L0–L27 sweep (best-effort; not in the
# documented schema). Expected key "prefill_all" of shape (500, 28, 1536).
ALLLAYER_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz"
OUT_JSON = ROOT / "pathway11_h100/raptor_refit/results.json"

N_FOLDS = 5
SEED = 9999
F2_DOM_HEADLINE = 0.7731
REVISE_DELTA = 0.02
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _tune_C(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> float:
    """Pick C maximizing inner-CV validation AUROC (concept-vector projection)."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_C, best_auc = C_GRID[0], -1.0
    for C in C_GRID:
        val_scores = np.zeros(len(ytr), dtype=np.float64)
        for itr, ite in inner.split(Xtr, ytr):
            mu = Xtr[itr].mean(0)
            sd = Xtr[itr].std(0) + 1e-8
            Xa = (Xtr[itr] - mu) / sd
            Xb = (Xtr[ite] - mu) / sd
            clf = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000)
            clf.fit(Xa, ytr[itr])
            w = clf.coef_.ravel()
            nrm = np.linalg.norm(w)
            if nrm < 1e-12:
                continue
            val_scores[ite] = Xb @ (w / nrm)
        auc = auroc(val_scores, ytr)
        if auc > best_auc:
            best_auc, best_C = auc, C
    return best_C


def _raptor_oof(X: np.ndarray, y: np.ndarray, seed: int):
    """5-fold OOF: per-fold tune λ, fit ℓ2-logistic, normalize w → concept vector."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    chosen_C = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte = X[test_idx]
        C = _tune_C(Xtr, ytr, seed)
        chosen_C.append(C)
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-8
        clf = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000)
        clf.fit((Xtr - mu) / sd, ytr)
        w = clf.coef_.ravel()
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            nrm = 1.0
        oof[test_idx] = ((Xte - mu) / sd) @ (w / nrm)
    return oof, chosen_C


def _dom_oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Plain difference-of-means direction, refit per fold (F-2 special case)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        d = Xtr[ytr].mean(0) - Xtr[~ytr].mean(0)
        oof[test_idx] = X[test_idx] @ d
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # --- L19 RAPTOR vs DoM head-to-head ---
    raptor_oof, chosen_C = _raptor_oof(X, y, SEED)
    raptor_auc = auroc(raptor_oof, y)

    dom_oof = _dom_oof(X, y, SEED)
    dom_auc_refit = auroc(dom_oof, y)

    # Reference DoM score from the canonical phase-2 cache, if present.
    dom_auc_cached = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auc_cached = auroc(dom_score, y)

    delta_vs_headline = raptor_auc - F2_DOM_HEADLINE
    delta_vs_refit = raptor_auc - dom_auc_refit
    revise_f2 = bool(delta_vs_headline >= REVISE_DELTA)

    # --- Best-layer (val-averaged) sweep across L0–L27, best-effort ---
    layer_sweep = {"status": "unavailable",
                   "reason": f"per-layer cache not found at {ALLLAYER_CACHE.name}"}
    if ALLLAYER_CACHE.exists():
        all_blob = np.load(ALLLAYER_CACHE)
        if "prefill_all" in all_blob:
            Xall = all_blob["prefill_all"].astype(np.float64)  # (500, 28, 1536)
            n_layers = Xall.shape[1]
            per_layer = {}
            best_layer, best_layer_auc = -1, -1.0
            for L in range(n_layers):
                oofL, _ = _raptor_oof(Xall[:, L, :], y, SEED)
                aL = auroc(oofL, y)
                per_layer[f"L{L}"] = aL
                if aL > best_layer_auc:
                    best_layer_auc, best_layer = aL, L
            layer_sweep = {
                "status": "computed",
                "n_layers": int(n_layers),
                "per_layer_raptor_oof_auroc": per_layer,
                "best_layer": int(best_layer),
                "best_layer_auroc": float(best_layer_auc),
            }
        else:
            layer_sweep["reason"] = "all-layer cache present but missing 'prefill_all' key"

    out = {
        "experiment": "P11-FE645",
        "method": "RAPTOR (ridge-tuned l2-logistic, weights L2-normalized -> concept vector)",
        "n": int(len(y)),
        "p": int(X.shape[1]),
        "base_rate": float(y.mean()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "C_grid": C_GRID,
        "raptor_oof_auroc_L19": float(raptor_auc),
        "raptor_chosen_C_per_fold": [float(c) for c in chosen_C],
        "dom_oof_auroc_L19_refit": float(dom_auc_refit),
        "dom_auroc_cached_score": (None if dom_auc_cached is None else float(dom_auc_cached)),
        "f2_dom_headline": F2_DOM_HEADLINE,
        "delta_raptor_vs_f2_headline": float(delta_vs_headline),
        "delta_raptor_vs_dom_refit": float(delta_vs_refit),
        "revise_delta_threshold": REVISE_DELTA,
        "revise_f2_headline": revise_f2,
        "verdict": ("RAPTOR beats DoM by >= 0.02; F-2 headline is conservative"
                    if revise_f2 else
                    "RAPTOR does not clear the +0.02 revision bar over F-2 DoM"),
        "layer_sweep": layer_sweep,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("raptor_oof_auroc_L19", "dom_oof_auroc_L19_refit",
                       "delta_raptor_vs_f2_headline", "revise_f2_headline")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())