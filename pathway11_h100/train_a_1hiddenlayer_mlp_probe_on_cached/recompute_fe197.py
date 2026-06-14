"""P11-FE197 — Non-linear MLP probe vs linear-probe ceiling for F-2.

NL-ITI (Hoscilowicz et al.) reports a +16% relative MC1 gain over linear ITI
by swapping the linear probe for a 1-hidden-layer MLP, with the largest gains
in early layers — implying linear probes systematically underestimate
non-linearly encoded concept information. F-2 fixed L19 as the optimal layer
and reports AUROC 0.7731 from a linear (LR / DoM) probe on cached Qwen-2.5-1.5B
prefill L19 activations. This script trains a 1-hidden-layer MLP probe under the
same 5-fold OOF protocol and compares OOF AUROC against the linear baselines
(DoM 0.7731, L2-regularized LR). If the MLP gains >=0.02 AUROC OOF, F-2's
number is a linear-probe ceiling, not a model property; otherwise the L19
prefill correctness signal is (to first order) linearly encoded.
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
OUT_JSON = ROOT / "pathway11_h100/mlp_probe/results.json"

N_FOLDS = 5
SEED = 9999
LR_BASELINE = 0.7731  # F-2 reported linear-probe (DoM) OOF AUROC
GAIN_THRESHOLD = 0.02  # MLP must beat the better linear probe by this to flip F-2


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


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        print("MISSING_REGEN_INPUT sklearn:", exc, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # Optional DoM reference (the exact F-2 readout); fall back to per-fold DoM.
    dom_score = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    n = len(y)
    folds = stratified_kfold(y, N_FOLDS, SEED)

    oof_dom = np.zeros(n, dtype=np.float64)      # per-fold linear DoM readout
    oof_lr = np.zeros(n, dtype=np.float64)       # L2-regularized logistic regression
    oof_mlp = np.zeros(n, dtype=np.float64)      # 1-hidden-layer MLP probe

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # Standardize on train statistics only.
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # Linear DoM readout (difference-of-means) on standardized features.
        d_vec = Xtr_s[ytr].mean(axis=0) - Xtr_s[~ytr].mean(axis=0)
        oof_dom[test_idx] = Xte_s @ d_vec

        # L2-regularized logistic regression baseline.
        lr = LogisticRegression(C=0.001, max_iter=2000, solver="lbfgs")
        lr.fit(Xtr_s, ytr)
        oof_lr[test_idx] = lr.decision_function(Xte_s)

        # 1-hidden-layer MLP probe (matched OOF protocol).
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-2,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        )
        mlp.fit(Xtr_s, ytr)
        # decision-function proxy: log-odds of the positive class.
        proba = mlp.predict_proba(Xte_s)
        pos_col = list(mlp.classes_).index(True) if True in mlp.classes_ else 1
        oof_mlp[test_idx] = proba[:, pos_col]

    auroc_dom = float(auroc(oof_dom, y))
    auroc_lr = float(auroc(oof_lr, y))
    auroc_mlp = float(auroc(oof_mlp, y))
    auroc_dom_precomputed = (
        float(auroc(dom_score, y)) if dom_score is not None else None
    )

    best_linear = max(auroc_dom, auroc_lr)
    gain_vs_best_linear = auroc_mlp - best_linear
    gain_vs_f2 = auroc_mlp - LR_BASELINE
    flips_f2 = bool(gain_vs_best_linear >= GAIN_THRESHOLD)

    out = {
        "experiment": "P11-FE197",
        "n": n,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "lr_baseline_f2": LR_BASELINE,
        "gain_threshold": GAIN_THRESHOLD,
        "auroc_dom_oof": auroc_dom,
        "auroc_lr_oof": auroc_lr,
        "auroc_mlp_oof": auroc_mlp,
        "auroc_dom_precomputed_oof": auroc_dom_precomputed,
        "best_linear_oof": float(best_linear),
        "mlp_gain_vs_best_linear": float(gain_vs_best_linear),
        "mlp_gain_vs_f2_baseline": float(gain_vs_f2),
        "mlp_flips_f2_linear_ceiling": flips_f2,
        "verdict": (
            "MLP beats best linear probe by >=0.02 OOF AUROC: F-2 is a "
            "linear-probe ceiling, not a model property."
            if flips_f2 else
            "MLP does not beat best linear probe by >=0.02 OOF AUROC: L19 "
            "prefill correctness signal is (to first order) linearly encoded."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())