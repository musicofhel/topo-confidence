#!/usr/bin/env python3
"""Experiment 2: PH cloud-shape vs CoE vs D2HScore on MATH-500.

Head-to-head comparison of layer-wise methods:
  (a) Layer-wise PH: 168 features (from Exp 1)
  (b) CoE: 60 features (4 scalar + 28 mag + 28 ang)
  (c) D2HScore-lite: 58 features (2 scalar + 28 dispersion + 28 drift)
  (d) D2HScore-full: 86+ features (add 28 attention entropy)
  (e-h) Combinations + ABC-44 baseline

CPU only — reads pre-extracted all-layer states and Exp 1 features.

Run: python pathway8_layerwise/exp2_method_comparison.py
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
    load_existing_abc_features,
    load_labels_and_split,
    mark_done,
)
from pathway8_layerwise.coe_features import compute_coe_batch
from pathway8_layerwise.d2hscore_features import compute_d2h_batch
from pathway8_layerwise.exp1_layerwise_ph import bootstrap_auroc_ci, run_delong

STAGE_NAME = "exp2_method_comparison"


def evaluate_model(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    y_train: np.ndarray,
    y_holdout: np.ndarray,
    name: str,
    C: float = 1.0,
) -> dict:
    """Fit LR, return AUROC + predicted probabilities."""
    if X_train.shape[1] == 0:
        return {
            "name": name, "n_features": 0,
            "auroc_holdout": 0.5, "auroc_cv": 0.5,
            "probs_holdout": np.full(len(y_holdout), 0.5),
        }

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_holdout)

    lr = LogisticRegression(C=C, **LR_PARAMS)
    lr.fit(X_tr, y_train)
    probs_holdout = lr.predict_proba(X_ho)[:, 1]
    auroc_holdout = roc_auc_score(y_holdout, probs_holdout)

    n_folds = min(50, max(2, int(y_train.sum())))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    probs_cv = cross_val_predict(
        LogisticRegression(C=C, **LR_PARAMS),
        X_tr, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    auroc_cv = roc_auc_score(y_train, probs_cv)

    return {
        "name": name,
        "n_features": X_train.shape[1],
        "auroc_holdout": round(auroc_holdout, 4),
        "auroc_cv": round(auroc_cv, 4),
        "probs_holdout": probs_holdout,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_done(STAGE_NAME):
        print(f"[SKIP] {STAGE_NAME} already done")
        return

    print("=" * 70)
    print("Experiment 2: PH vs CoE vs D2HScore Method Comparison")
    print("=" * 70)

    # Load data
    print("\nLoading data ...")
    labels, train_idx, holdout_idx = load_labels_and_split()
    y_train = labels[train_idx]
    y_holdout = labels[holdout_idx]

    states_list = load_all_layer_states(MATH500_DATA_DIR, N_PROBLEMS)

    # Load D2H attention entropy from npz files
    d2h_attn_list = []
    for i in range(N_PROBLEMS):
        path = MATH500_DATA_DIR / f"problem_{i:03d}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            if "d2h_attn_entropy" in data:
                d2h_attn_list.append(data["d2h_attn_entropy"])
            else:
                d2h_attn_list.append(None)
        else:
            d2h_attn_list.append(None)

    # Load Exp 1 layer-wise PH features (if available)
    exp1_path = RESULTS_DIR / "exp1_features_all.npy"
    if exp1_path.exists():
        ph_features = np.load(exp1_path)
        print(f"  Loaded Exp 1 PH features: {ph_features.shape}")
    else:
        print("  [WARN] Exp 1 features not found, computing from scratch ...")
        from pathway8_layerwise.layerwise_features import (
            compute_layerwise_ph_batch,
            fit_per_layer_pca,
        )
        transforms = fit_per_layer_pca(states_list, train_idx)
        ph_features = compute_layerwise_ph_batch(states_list, transforms)
        np.save(exp1_path, ph_features)

    # Compute CoE features
    print("\nComputing CoE features ...")
    t0 = time.time()
    coe_features, coe_names = compute_coe_batch(states_list)
    print(f"  CoE: {coe_features.shape[1]} features, {time.time()-t0:.1f}s")

    # Compute D2HScore-lite features (from hidden states only)
    print("Computing D2HScore-lite features ...")
    t0 = time.time()
    d2h_lite_features, d2h_lite_names = compute_d2h_batch(states_list, full=False)
    print(f"  D2H-lite: {d2h_lite_features.shape[1]} features, {time.time()-t0:.1f}s")

    # Compute D2HScore-full features (with attention)
    print("Computing D2HScore-full features ...")
    t0 = time.time()
    d2h_full_features, d2h_full_names = compute_d2h_batch(
        states_list, d2h_attn_list, full=True,
    )
    print(f"  D2H-full: {d2h_full_features.shape[1]} features, {time.time()-t0:.1f}s")

    # Load ABC-44 baseline
    abc_X_train, abc_X_holdout = load_existing_abc_features()

    # ---- Build all model variants ----
    print(f"\n{'='*70}")
    print("Model Comparison")
    print(f"{'='*70}")

    models = []

    # (a) ABC-44 baseline
    models.append(evaluate_model(
        abc_X_train, abc_X_holdout, y_train, y_holdout, "ABC-44 (baseline)", C=1.0,
    ))

    # (b) Layer-wise PH (168)
    models.append(evaluate_model(
        ph_features[train_idx], ph_features[holdout_idx],
        y_train, y_holdout, "LayerwisePH (168)", C=1.0,
    ))

    # (c) CoE
    models.append(evaluate_model(
        coe_features[train_idx], coe_features[holdout_idx],
        y_train, y_holdout, f"CoE ({coe_features.shape[1]})", C=1.0,
    ))

    # (d) D2HScore-lite
    models.append(evaluate_model(
        d2h_lite_features[train_idx], d2h_lite_features[holdout_idx],
        y_train, y_holdout, f"D2H-lite ({d2h_lite_features.shape[1]})", C=1.0,
    ))

    # (e) D2HScore-full
    models.append(evaluate_model(
        d2h_full_features[train_idx], d2h_full_features[holdout_idx],
        y_train, y_holdout, f"D2H-full ({d2h_full_features.shape[1]})", C=1.0,
    ))

    # (f) PH + CoE
    ph_coe = np.hstack([ph_features, coe_features])
    models.append(evaluate_model(
        ph_coe[train_idx], ph_coe[holdout_idx],
        y_train, y_holdout, f"PH+CoE ({ph_coe.shape[1]})", C=1.0,
    ))

    # (g) PH + D2H-full
    ph_d2h = np.hstack([ph_features, d2h_full_features])
    models.append(evaluate_model(
        ph_d2h[train_idx], ph_d2h[holdout_idx],
        y_train, y_holdout, f"PH+D2H ({ph_d2h.shape[1]})", C=1.0,
    ))

    # (h) All combined
    all_combined = np.hstack([ph_features, coe_features, d2h_full_features])
    models.append(evaluate_model(
        all_combined[train_idx], all_combined[holdout_idx],
        y_train, y_holdout, f"All combined ({all_combined.shape[1]})", C=1.0,
    ))

    # Print comparison table
    print(f"\n  {'Model':<35s} {'AUROC':>8s} {'CV':>8s} {'Features':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for m in models:
        print(f"  {m['name']:<35s} {m['auroc_holdout']:8.4f} {m['auroc_cv']:8.4f} {m['n_features']:8d}")

    # ---- DeLong pairwise tests vs baseline ----
    print(f"\n{'='*70}")
    print("DeLong Tests vs Baseline")
    print(f"{'='*70}")

    baseline_probs = models[0]["probs_holdout"]
    delong_results = []
    for m in models[1:]:
        dl = run_delong(y_holdout, baseline_probs, m["probs_holdout"],
                        "ABC-44", m["name"])
        delong_results.append(dl)
        sig = "*" if dl.get("significant", False) else ""
        print(f"  vs {m['name']:<30s}  z={dl.get('z_statistic', 0):+.3f}  p={dl.get('p_value', 1):.4f} {sig}")

    # ---- Best model bootstrap CI ----
    best = max(models, key=lambda m: m["auroc_holdout"])
    ci = bootstrap_auroc_ci(y_holdout, best["probs_holdout"])

    # ---- Save results ----
    results = {
        "experiment": "exp2_method_comparison",
        "models": [
            {k: v for k, v in m.items() if k != "probs_holdout"}
            for m in models
        ],
        "best_model": best["name"],
        "best_auroc": best["auroc_holdout"],
        "best_ci": ci,
        "baseline_auroc": models[0]["auroc_holdout"],
        "delong_vs_baseline": delong_results,
        "beats_baseline": best["auroc_holdout"] > models[0]["auroc_holdout"],
        "proceed_to_exp3": best["auroc_holdout"] > 0.75,
        "feature_counts": {
            "ph": ph_features.shape[1],
            "coe": coe_features.shape[1],
            "d2h_lite": d2h_lite_features.shape[1],
            "d2h_full": d2h_full_features.shape[1],
        },
        "coe_feature_names": coe_names,
        "d2h_lite_feature_names": d2h_lite_names,
        "d2h_full_feature_names": d2h_full_names,
    }

    out_path = RESULTS_DIR / "exp2_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))

    # Save feature arrays for downstream use
    np.save(RESULTS_DIR / "exp2_coe_features.npy", coe_features)
    np.save(RESULTS_DIR / "exp2_d2h_full_features.npy", d2h_full_features)

    mark_done(STAGE_NAME)

    print(f"\n{'='*70}")
    print("Experiment 2 Summary")
    print(f"{'='*70}")
    print(f"  Best model:    {best['name']} AUROC={best['auroc_holdout']:.4f}")
    print(f"  Baseline:      ABC-44 AUROC={models[0]['auroc_holdout']:.4f}")
    print(f"  Proceed to Exp 3: {results['proceed_to_exp3']}")
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
