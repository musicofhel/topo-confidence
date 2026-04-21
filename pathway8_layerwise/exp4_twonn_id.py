#!/usr/bin/env python3
"""Experiment 4: TwoNN intrinsic dimension (conditional on Exp 1).

Adds 28 TwoNN ID features (one per layer) to Exp 1's 168 features.
Tests marginal AUROC lift.

CPU only.

Run: python pathway8_layerwise/exp4_twonn_id.py
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
from pathway8_layerwise.exp1_layerwise_ph import bootstrap_auroc_ci, run_delong
from pathway8_layerwise.twonn_features import compute_twonn_batch

STAGE_NAME = "exp4_twonn_id"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_done(STAGE_NAME):
        print(f"[SKIP] {STAGE_NAME} already done")
        return

    # Check Exp 1 prerequisite
    exp1_path = RESULTS_DIR / "exp1_features_all.npy"
    if not exp1_path.exists():
        print("[SKIP] Exp 1 features not found — run exp1 first")
        return

    print("=" * 70)
    print("Experiment 4: TwoNN Intrinsic Dimension")
    print("=" * 70)

    # Load data
    labels, train_idx, holdout_idx = load_labels_and_split()
    y_train = labels[train_idx]
    y_holdout = labels[holdout_idx]

    ph_features = np.load(exp1_path)
    states_list = load_all_layer_states(MATH500_DATA_DIR, N_PROBLEMS)

    # Compute TwoNN per layer
    print("\nComputing TwoNN ID per layer ...")
    t0 = time.time()
    twonn_features, twonn_names = compute_twonn_batch(states_list)
    print(f"  TwoNN: {twonn_features.shape[1]} features, {time.time()-t0:.1f}s")
    print(f"  Mean ID per layer: {np.nanmean(twonn_features, axis=0)[:5]}... (first 5)")

    # Combined: PH + TwoNN
    combined = np.hstack([ph_features, twonn_features])

    # Evaluate
    def eval_model(X_all, name, C=1.0):
        X_train = X_all[train_idx]
        X_holdout = X_all[holdout_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_ho = scaler.transform(X_holdout)
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

    print("\nEvaluating models ...")
    r_ph = eval_model(ph_features, f"PH only ({ph_features.shape[1]})")
    r_twonn = eval_model(twonn_features, f"TwoNN only ({twonn_features.shape[1]})")
    r_combined = eval_model(combined, f"PH+TwoNN ({combined.shape[1]})")

    print(f"  {'Model':<30s} {'AUROC':>8s} {'CV':>8s}")
    for r in [r_ph, r_twonn, r_combined]:
        print(f"  {r['name']:<30s} {r['auroc_holdout']:8.4f} {r['auroc_cv']:8.4f}")

    # DeLong: combined vs PH-only
    delong = run_delong(y_holdout, r_ph["probs_holdout"], r_combined["probs_holdout"],
                        "PH-only", "PH+TwoNN")

    results = {
        "experiment": "exp4_twonn_id",
        "models": [
            {k: v for k, v in r.items() if k != "probs_holdout"}
            for r in [r_ph, r_twonn, r_combined]
        ],
        "marginal_lift": round(r_combined["auroc_holdout"] - r_ph["auroc_holdout"], 4),
        "delong_combined_vs_ph": delong,
        "twonn_feature_names": twonn_names,
        "mean_id_per_layer": np.nanmean(twonn_features, axis=0).tolist(),
    }

    out_path = RESULTS_DIR / "exp4_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    np.save(RESULTS_DIR / "exp4_twonn_features.npy", twonn_features)
    mark_done(STAGE_NAME)

    print(f"\n  Marginal lift: {results['marginal_lift']:+.4f}")
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
