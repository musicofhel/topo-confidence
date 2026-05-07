"""FE421 — Ridge-LR on [prefill, final] concat.

Tests whether properly regularized logistic regression on the concatenated
(500, 3072) feature space exceeds DoM-only 0.7731 — the key follow-up from
EXP-51's finding that naive DoM concat = 0.6946.

Output: pathway11_h100/results/fe421_regularized_concat.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe421_regularized_concat.json"

SEED = 9999
N_FOLDS = 5
CS = [0.001, 0.01, 0.1, 1.0, 10.0]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def oof_ridge_auroc(X: np.ndarray, y: np.ndarray, skf, label: str) -> dict:
    """Run OOF LogisticRegressionCV and return AUROC + best C."""
    n = len(y)
    oof_scores = np.zeros(n, dtype=np.float64)
    best_cs = []

    for train_idx, test_idx in skf.split(X, y):
        lr = LogisticRegressionCV(
            Cs=CS, penalty="l2", solver="lbfgs",
            max_iter=5000, random_state=SEED, cv=3,
        )
        lr.fit(X[train_idx], y[train_idx])
        oof_scores[test_idx] = lr.predict_proba(X[test_idx])[:, 1]
        best_cs.append(float(lr.C_[0]))

    auc = auroc(oof_scores, y)
    print(f"  {label}: AUROC={auc:.4f}  best_Cs={best_cs}")
    return {"auroc_oof": float(auc), "best_Cs": best_cs}


def dom_auroc(X: np.ndarray, y: np.ndarray, skf) -> float:
    """OOF DoM AUROC (mean-diff direction, project, threshold)."""
    n = len(y)
    oof_scores = np.zeros(n, dtype=np.float64)

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        dom = X_train[y_train].mean(0) - X_train[~y_train].mean(0)
        dom = dom / (np.linalg.norm(dom) + 1e-30)
        oof_scores[test_idx] = X[test_idx] @ dom

    return float(auroc(oof_scores, y))


def main() -> int:
    cache = np.load(CACHE)
    prefill = cache["prefill"].astype(np.float32)
    final_tok = cache["final_tok"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    n = len(correct)

    X_concat = np.concatenate([prefill, final_tok], axis=1)  # (500, 3072)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print("Running Ridge-LR (LogisticRegressionCV, l2)...")
    res_concat = oof_ridge_auroc(X_concat, correct, skf, "concat(3072)")
    res_prefill = oof_ridge_auroc(prefill, correct, skf, "prefill(1536)")
    res_final = oof_ridge_auroc(final_tok, correct, skf, "final(1536)")

    print("\nRunning DoM baselines...")
    dom_concat_auroc = dom_auroc(X_concat, correct, skf)
    dom_prefill_auroc = dom_auroc(prefill, correct, skf)
    dom_final_auroc = dom_auroc(final_tok, correct, skf)
    print(f"  DoM concat: {dom_concat_auroc:.4f}")
    print(f"  DoM prefill: {dom_prefill_auroc:.4f}")
    print(f"  DoM final: {dom_final_auroc:.4f}")

    out = {
        "experiment": "FE421",
        "description": "Ridge-LR on [prefill, final] concat vs single-source",
        "n": n,
        "n_correct": int(correct.sum()),
        "ridge_lr": {
            "concat_3072": res_concat,
            "prefill_1536": res_prefill,
            "final_1536": res_final,
        },
        "dom_baselines": {
            "concat_3072": float(dom_concat_auroc),
            "prefill_1536": float(dom_prefill_auroc),
            "final_1536": float(dom_final_auroc),
        },
        "references": {
            "dom_prefill_canonical": 0.7731,
            "dom_concat_exp51_fe254": 0.6946,
        },
        "cross_check": {
            "dom_concat_matches_fe254": abs(dom_concat_auroc - 0.6946) < 0.01,
        },
        "interpretation": (
            "If ridge concat > prefill DoM 0.7731, the final-token carries "
            "complementary signal that DoM alone misses. The gap between "
            "ridge and DoM on concat quantifies how much regularization "
            "helps vs naive mean-diff."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "cache": str(CACHE),
            "seed": SEED,
            "Cs": CS,
            "method": "logistic_regression_cv_l2",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
