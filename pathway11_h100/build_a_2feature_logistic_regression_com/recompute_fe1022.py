"""FE1022 — DoM + output-diversity 2-feature logistic regression.

Tests complementarity of internal residual-stream geometry (DoM score) and an
output-side signal (answer-agreement rate / cluster count from K=8 self-
consistency samples). Builds a 2-feature OOF logistic regression and asks
whether its AUROC exceeds the 0.7928 cov-spectrum ceiling and beats each
single-feature baseline — evidence that DoM carries information orthogonal to
output diversity.
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
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/dom_plus_diversity/results.json"

COV_SPECTRUM_CEILING = 0.7928
SEED = 9999
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_k8_features(n: int):
    """Per-problem output-diversity features from K=8 self-consistency runs.

    agreement_rate = fraction of the 8 samples that match the modal verdict
                     (a model-free proxy for answer-cluster concentration).
    cluster_count  = number of distinct outcome groups (here a coarse 2-way
                     correct/incorrect split, in [1, 2]); higher = more diverse.
    Returns (agreement_rate, cluster_count, n_found).
    """
    agreement = np.full(n, np.nan, dtype=np.float64)
    clusters = np.full(n, np.nan, dtype=np.float64)
    n_found = 0
    for i in range(n):
        fp = K8_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            continue
        c = np.load(fp)["correct"].astype(bool)
        if c.size == 0:
            continue
        n_found += 1
        n_true = int(c.sum())
        n_false = int(c.size - n_true)
        modal = max(n_true, n_false)
        agreement[i] = modal / c.size
        clusters[i] = float((n_true > 0) + (n_false > 0))
    return agreement, clusters, n_found


def oof_logreg(X: np.ndarray, y: np.ndarray, seed: int, n_folds: int) -> np.ndarray:
    """Out-of-fold predicted P(correct) from a standardized logistic regression."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(Xtr_s, y[train_idx])
        oof[test_idx] = clf.predict_proba(Xte_s)[:, 1]
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    y = cache["correct"].astype(bool)
    n = len(y)
    assert y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (n,)

    agreement, clusters, n_found = load_k8_features(n)
    if n_found == 0:
        print("MISSING_REGEN_INPUT", K8_DIR, "(no problem_NNN.npz found)", file=sys.stderr)
        return 2

    # Restrict to problems with a K=8 cache so every model sees the same rows.
    have = np.isfinite(agreement) & np.isfinite(clusters)
    y_e = y[have]
    dom_e = dom_score[have]
    agree_e = agreement[have]
    clust_e = clusters[have]
    n_eval = int(have.sum())

    # Single-feature OOF baselines.
    dom_oof = oof_logreg(dom_e[:, None], y_e, SEED, N_FOLDS)
    agree_oof = oof_logreg(agree_e[:, None], y_e, SEED, N_FOLDS)
    clust_oof = oof_logreg(clust_e[:, None], y_e, SEED, N_FOLDS)

    auroc_dom = auroc(dom_oof, y_e)
    auroc_agree = auroc(agree_oof, y_e)
    auroc_clust = auroc(clust_oof, y_e)
    # DoM AUROC directly from the raw score, for reference (no OOF logistic).
    auroc_dom_raw = auroc(dom_e, y_e)

    # 2-feature combinations.
    X_dom_agree = np.column_stack([dom_e, agree_e])
    X_dom_clust = np.column_stack([dom_e, clust_e])
    combo_agree_oof = oof_logreg(X_dom_agree, y_e, SEED, N_FOLDS)
    combo_clust_oof = oof_logreg(X_dom_clust, y_e, SEED, N_FOLDS)
    auroc_combo_agree = auroc(combo_agree_oof, y_e)
    auroc_combo_clust = auroc(combo_clust_oof, y_e)

    best_combo = max(auroc_combo_agree, auroc_combo_clust)
    best_single = max(auroc_dom, auroc_agree, auroc_clust)

    out = {
        "experiment": "FE1022",
        "n_eval": n_eval,
        "n_total": n,
        "n_k8_found": n_found,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "cov_spectrum_ceiling": COV_SPECTRUM_CEILING,
        "auroc_dom_raw": auroc_dom_raw,
        "auroc_dom_oof": auroc_dom,
        "auroc_agreement_oof": auroc_agree,
        "auroc_cluster_oof": auroc_clust,
        "auroc_combo_dom_agreement_oof": auroc_combo_agree,
        "auroc_combo_dom_cluster_oof": auroc_combo_clust,
        "best_single_feature": best_single,
        "best_combo_feature": best_combo,
        "combo_beats_ceiling": bool(best_combo > COV_SPECTRUM_CEILING),
        "combo_lift_over_ceiling": float(best_combo - COV_SPECTRUM_CEILING),
        "combo_lift_over_best_single": float(best_combo - best_single),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())