"""P11-FE301 — LEAT (last exact answer token) probe vs F-2 prefill baseline.

Refutation #1 of triage brief 2410.02707 (Orgad et al.). They report that a
probe on the *exact-answer token* (LEAT, the token emitted immediately after
`\\boxed{...}` is closed) beats a probe on earlier/prefill hidden states by
10-20 AUROC points on Mistral/Llama. This script tests the claim on
Qwen-1.5B + MATH-500: for each of the 500 problems, the L19 hidden state at the
post-\\boxed answer token is loaded from a Stage-2 LEAT NPZ, an OOF 5-fold
L2-regularized logistic probe is trained on those features, and its AUROC is
compared against the F-2 prefill DoM baseline (0.7731, recomputed OOF here from
the prefill cache for an apples-to-apples comparison).

If LEAT AUROC materially exceeds the prefill baseline, F-2 must be reframed as
"L19 LEAT > L19 prefill". If it does not, the Orgad finding does not transfer
to Qwen-1.5B/MATH-500 and F-2 stands.

Requires the LEAT activation cache (per-token L19 states at the boxed answer
token), which is NOT one of the standard P11 NPZs. If absent, this prints
MISSING_REGEN_INPUT and returns 2 so the regen step can produce it.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
LEAT_CACHE = ROOT / "pathway11_h100/leat_probe/cache/m15b_leat.npz"
OUT_JSON = ROOT / "pathway11_h100/leat_probe/results.json"

N_FOLDS = 5
SEED = 9999
C_GRID = (0.001, 0.01, 0.1, 1.0)
F2_PREFILL_BASELINE = 0.7731


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oof_logistic(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Out-of-fold standardized L2-logistic decision scores."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        scores[test_idx] = clf.decision_function(Xte)
    return scores


def oof_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold difference-of-means (DoM) projection scores."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        ytr = y[train_idx]
        d = X[train_idx][ytr].mean(axis=0) - X[train_idx][~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return scores


def best_oof_logistic(X: np.ndarray, y: np.ndarray):
    best_auc = -1.0
    best_C = None
    best_scores = None
    per_C = {}
    for C in C_GRID:
        s = oof_logistic(X, y, C)
        a = auroc(s, y)
        per_C[str(C)] = a
        if a > best_auc:
            best_auc, best_C, best_scores = a, C, s
    return best_C, best_auc, best_scores, per_C


def main() -> int:
    if not PREFILL_CACHE.exists():
        print("MISSING_REGEN_INPUT", PREFILL_CACHE, file=sys.stderr)
        return 2
    if not LEAT_CACHE.exists():
        print("MISSING_REGEN_INPUT", LEAT_CACHE, file=sys.stderr)
        return 2

    pre = np.load(PREFILL_CACHE)
    X_pre = pre["prefill"].astype(np.float64)
    y = pre["correct"].astype(bool)
    assert X_pre.shape == (500, 1536) and y.shape == (500,)

    leat = np.load(LEAT_CACHE)
    if "leat" not in leat.files:
        print("MISSING_REGEN_INPUT", LEAT_CACHE, "(no 'leat' key)", file=sys.stderr)
        return 2
    X_leat = leat["leat"].astype(np.float64)
    if X_leat.shape != (500, 1536):
        print("MISSING_REGEN_INPUT", LEAT_CACHE,
              f"(bad shape {X_leat.shape})", file=sys.stderr)
        return 2
    # Prefer LEAT-cache labels if present; otherwise reuse prefill labels.
    y_leat = leat["correct"].astype(bool) if "correct" in leat.files else y

    # --- LEAT probe (logistic, C-grid OOF) ---
    leat_C, leat_auc, _, leat_per_C = best_oof_logistic(X_leat, y_leat)
    leat_dom_auc = auroc(oof_dom(X_leat, y_leat), y_leat)

    # --- Prefill baseline, recomputed OOF on identical folds ---
    pre_C, pre_logit_auc, _, pre_per_C = best_oof_logistic(X_pre, y)
    pre_dom_auc = auroc(oof_dom(X_pre, y), y)

    delta_vs_baseline = leat_auc - F2_PREFILL_BASELINE
    delta_vs_recomputed = leat_auc - pre_dom_auc
    leat_beats = bool(delta_vs_recomputed > 0.01)

    out = {
        "experiment": "P11-FE301",
        "description": "LEAT (post-\\boxed answer token) L19 probe vs F-2 prefill baseline",
        "n_problems": int(len(y)),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "leat": {
            "auroc_logistic_best": float(leat_auc),
            "best_C": leat_C,
            "auroc_logistic_per_C": leat_per_C,
            "auroc_dom": float(leat_dom_auc),
            "n_correct": int(y_leat.sum()),
        },
        "prefill_baseline": {
            "f2_reported_dom": F2_PREFILL_BASELINE,
            "auroc_dom_recomputed_oof": float(pre_dom_auc),
            "auroc_logistic_best": float(pre_logit_auc),
            "best_C": pre_C,
            "auroc_logistic_per_C": pre_per_C,
        },
        "comparison": {
            "delta_leat_vs_f2_reported": float(delta_vs_baseline),
            "delta_leat_vs_recomputed_prefill_dom": float(delta_vs_recomputed),
            "leat_beats_prefill": leat_beats,
            "verdict": (
                "REFRAME_F2_LEAT_GT_PREFILL" if leat_beats
                else "ORGAD_DOES_NOT_TRANSFER_F2_STANDS"
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())