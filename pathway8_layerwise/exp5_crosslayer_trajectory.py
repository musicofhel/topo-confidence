#!/usr/bin/env python3
"""Experiment 5: Cross-layer single-token trajectory PH.

Conditional on Exp 1 AND Exp 2 showing layer-wise > single-layer.
Uses H0-only (n=28 points per trajectory is too small for H1).
Bootstrap: m=20 tokens from ~100, repeat 50×.

CPU only.

Run: python pathway8_layerwise/exp5_crosslayer_trajectory.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.config import (
    LR_PARAMS,
    MATH500_DATA_DIR,
    N_PROBLEMS,
    RESULTS_DIR,
    is_done,
    load_all_layer_states,
    load_labels_and_split,
    mark_done,
)
from pathway8_layerwise.crosslayer_features import compute_crosslayer_batch
from pathway8_layerwise.exp1_layerwise_ph import bootstrap_auroc_ci, run_delong

STAGE_NAME = "exp5_crosslayer_trajectory"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_done(STAGE_NAME):
        print(f"[SKIP] {STAGE_NAME} already done")
        return

    # Check prerequisites
    exp1_path = RESULTS_DIR / "exp1_results.json"
    exp2_path = RESULTS_DIR / "exp2_results.json"

    if exp1_path.exists() and exp2_path.exists():
        exp1 = json.loads(exp1_path.read_text())
        exp2 = json.loads(exp2_path.read_text())
        # Check if layer-wise beats single-layer
        if exp1["auroc_holdout"] < 0.75 and exp2["best_auroc"] < 0.75:
            print("[SKIP] Layer-wise methods did not show sufficient signal (Exp 1+2)")
            print(f"  Exp 1 AUROC: {exp1['auroc_holdout']}, Exp 2 best: {exp2['best_auroc']}")
            mark_done(STAGE_NAME)
            return

    print("=" * 70)
    print("Experiment 5: Cross-Layer Single-Token Trajectory PH")
    print("=" * 70)

    # Load data
    labels, train_idx, holdout_idx = load_labels_and_split()
    y_train = labels[train_idx]
    y_holdout = labels[holdout_idx]

    states_list = load_all_layer_states(MATH500_DATA_DIR, N_PROBLEMS)

    # Load Exp 1 features for comparison
    exp1_features_path = RESULTS_DIR / "exp1_features_all.npy"
    ph_features = np.load(exp1_features_path) if exp1_features_path.exists() else None

    # Compute cross-layer trajectory PH features
    print("\nComputing cross-layer token trajectory PH (bootstrap) ...")
    t0 = time.time()
    xl_features, xl_names = compute_crosslayer_batch(
        states_list, n_bootstrap=50, m_tokens=20
    )
    print(f"  Cross-layer features: {xl_features.shape}, {time.time()-t0:.1f}s")

    # Evaluate standalone
    def eval_model(X_all, name, C=1.0):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_idx])
        X_ho = scaler.transform(X_all[holdout_idx])
        lr = LogisticRegression(C=C, **LR_PARAMS)
        lr.fit(X_tr, y_train)
        probs = lr.predict_proba(X_ho)[:, 1]
        auroc = roc_auc_score(y_holdout, probs)
        n_folds = min(50, max(2, int(y_train.sum())))
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        probs_cv = cross_val_predict(
            LogisticRegression(C=C, **LR_PARAMS),
            X_tr, y_train, cv=cv, method="predict_proba",
        )[:, 1]
        auroc_cv = roc_auc_score(y_train, probs_cv)
        return {"name": name, "auroc_holdout": round(auroc, 4),
                "auroc_cv": round(auroc_cv, 4), "n_features": X_all.shape[1],
                "probs_holdout": probs}

    print("\nEvaluating ...")
    r_xl = eval_model(xl_features, f"CrossLayer ({xl_features.shape[1]})")

    models = [r_xl]

    # Combined with Exp 1 PH features
    if ph_features is not None:
        combined = np.hstack([ph_features, xl_features])
        r_combined = eval_model(combined, f"PH+CrossLayer ({combined.shape[1]})")
        models.append(r_combined)

        # DeLong
        delong = run_delong(y_holdout, models[0]["probs_holdout"],
                           r_combined["probs_holdout"], "CrossLayer", "PH+CrossLayer")
    else:
        delong = None

    print(f"\n  {'Model':<35s} {'AUROC':>8s} {'CV':>8s}")
    for m in models:
        print(f"  {m['name']:<35s} {m['auroc_holdout']:8.4f} {m['auroc_cv']:8.4f}")

    results = {
        "experiment": "exp5_crosslayer_trajectory",
        "models": [
            {k: v for k, v in m.items() if k != "probs_holdout"}
            for m in models
        ],
        "delong": delong,
        "feature_names": xl_names,
        "n_bootstrap": 50,
        "m_tokens": 20,
    }

    out_path = RESULTS_DIR / "exp5_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    np.save(RESULTS_DIR / "exp5_crosslayer_features.npy", xl_features)
    mark_done(STAGE_NAME)

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
