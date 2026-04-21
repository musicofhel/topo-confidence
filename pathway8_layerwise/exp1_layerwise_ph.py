#!/usr/bin/env python3
"""Experiment 1: Layer-wise PH on MATH-500.

Computes 6 PH features per each of 28 transformer layers (168 total) using
per-layer PCA at 20 components. Compares AUROC against the 0.796 ABC-44 baseline.
Includes z-score divergence analysis (REMA-style) and DeLong test.

CPU only — reads pre-extracted all-layer states from data/math500/.

Run: python pathway8_layerwise/exp1_layerwise_ph.py
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
    BASELINE_PROBS_PATH,
    LR_PARAMS,
    MATH500_DATA_DIR,
    N_PROBLEMS,
    N_TRANSFORMER_LAYERS,
    RESULTS_DIR,
    is_done,
    load_all_layer_states,
    load_existing_abc_features,
    load_labels_and_split,
    make_layerwise_feature_names,
    mark_done,
)
from pathway8_layerwise.diagnostics import (
    diagnostic_battery,
    sanity_gate_10,
    validate_ph_pipeline,
)
from pathway8_layerwise.layerwise_features import (
    compute_layerwise_ph_batch,
    compute_zscore_divergence,
    fit_per_layer_pca,
)

STAGE_NAME = "exp1_layerwise_ph"


def run_delong(y_true, probs_a, probs_b, name_a="A", name_b="B") -> dict:
    """Run DeLong test comparing two sets of predictions."""
    try:
        from MLstatkit.stats import Delong_test
        z, p = Delong_test(y_true, probs_a, probs_b)
        return {
            "z_statistic": float(z),
            "p_value": float(p),
            "significant": float(p) < 0.05,
            "models": [name_a, name_b],
        }
    except Exception as e:
        return {"error": str(e), "models": [name_a, name_b]}


def bootstrap_auroc_ci(y_true, probs, n_boot=2000, seed=42) -> dict:
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], probs[idx]))
    aurocs = np.array(aurocs)
    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_std": float(aurocs.std()),
        "ci_lower": float(np.percentile(aurocs, 2.5)),
        "ci_upper": float(np.percentile(aurocs, 97.5)),
        "n_bootstrap": len(aurocs),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_done(STAGE_NAME):
        print(f"[SKIP] {STAGE_NAME} already done")
        return

    print("=" * 70)
    print("Experiment 1: Layer-wise PH on MATH-500")
    print("=" * 70)

    # Step 1: Validate PH pipeline
    validate_ph_pipeline()

    # Step 2: Load data
    print("\nLoading data ...")
    labels, train_idx, holdout_idx = load_labels_and_split()
    y_train = labels[train_idx]
    y_holdout = labels[holdout_idx]
    print(f"  Labels: {labels.sum()}/{len(labels)} correct")
    print(f"  Train: {len(train_idx)} ({y_train.sum()} correct)")
    print(f"  Holdout: {len(holdout_idx)} ({y_holdout.sum()} correct)")

    states_list = load_all_layer_states(MATH500_DATA_DIR, N_PROBLEMS)
    n_loaded = sum(1 for s in states_list if s is not None)
    assert n_loaded >= 450, f"Only {n_loaded} problems loaded, need >= 450"

    # Step 3: Fit per-layer PCA on train only
    print("\nFitting per-layer PCA ...")
    t0 = time.time()
    transforms = fit_per_layer_pca(states_list, train_idx)
    print(f"  PCA fitting took {time.time() - t0:.1f}s")

    # Step 4: Compute 168 layer-wise PH features
    print("\nComputing layer-wise PH features (168 = 28 layers × 6 features) ...")
    t0 = time.time()

    # Quick sanity gate on first 10
    from pathway8_layerwise.layerwise_features import compute_layerwise_ph_single
    early_feats = []
    for i in range(min(10, len(states_list))):
        if states_list[i] is not None:
            f = compute_layerwise_ph_single(states_list[i], transforms)
            early_feats.append(f)
    if early_feats:
        early_arr = np.stack(early_feats)
        if not sanity_gate_10(early_arr, labels[:len(early_feats)]):
            print("[ABORT] Sanity gate failed after 10 problems")
            return

    # Full batch computation with parallel processing
    features_all = compute_layerwise_ph_batch(states_list, transforms, n_jobs=-1)
    print(f"  Feature computation took {time.time() - t0:.1f}s")
    print(f"  Feature shape: {features_all.shape}")

    X_train = features_all[train_idx]
    X_holdout = features_all[holdout_idx]

    # Step 5: Train and evaluate LR classifier
    print("\nTraining LogisticRegression ...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_holdout)

    lr = LogisticRegression(C=1.0, **LR_PARAMS)
    lr.fit(X_tr, y_train)
    probs_holdout = lr.predict_proba(X_ho)[:, 1]
    auroc_holdout = roc_auc_score(y_holdout, probs_holdout)

    # 50-fold CV on train
    n_folds = min(50, int(y_train.sum()))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    probs_train_cv = cross_val_predict(
        LogisticRegression(C=1.0, **LR_PARAMS),
        X_tr, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    auroc_cv = roc_auc_score(y_train, probs_train_cv)

    print(f"\n  Layer-wise PH (168 features):")
    print(f"    Holdout AUROC: {auroc_holdout:.4f}")
    print(f"    CV AUROC:      {auroc_cv:.4f}")

    # Also try C sweep
    c_results = {}
    for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        lr_c = LogisticRegression(C=C, **LR_PARAMS)
        lr_c.fit(X_tr, y_train)
        p = lr_c.predict_proba(X_ho)[:, 1]
        c_results[str(C)] = round(roc_auc_score(y_holdout, p), 4)
    best_C = max(c_results, key=c_results.get)
    print(f"    C sweep: {c_results}")
    print(f"    Best C={best_C}: AUROC={c_results[best_C]}")

    # Step 6: Reproduce baseline for comparison
    print("\nReproducing ABC-44 baseline ...")
    abc_X_train, abc_X_holdout = load_existing_abc_features()
    abc_scaler = StandardScaler()
    abc_Xtr = abc_scaler.fit_transform(abc_X_train)
    abc_Xho = abc_scaler.transform(abc_X_holdout)
    abc_lr = LogisticRegression(C=1.0, **LR_PARAMS)
    abc_lr.fit(abc_Xtr, y_train)
    abc_probs = abc_lr.predict_proba(abc_Xho)[:, 1]
    abc_auroc = roc_auc_score(y_holdout, abc_probs)
    print(f"  ABC-44 baseline AUROC: {abc_auroc:.4f}")

    # Step 7: DeLong test
    print("\nDeLong test (layer-wise vs baseline) ...")
    delong = run_delong(
        y_holdout, abc_probs, probs_holdout,
        name_a="ABC-44", name_b="LayerwisePH-168",
    )
    print(f"  DeLong z={delong.get('z_statistic', 'N/A')}, p={delong.get('p_value', 'N/A')}")

    # Bootstrap CI
    ci = bootstrap_auroc_ci(y_holdout, probs_holdout)
    print(f"  Bootstrap 95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Step 8: Z-score divergence analysis
    print("\nZ-score divergence analysis ...")
    divergence = compute_zscore_divergence(features_all, labels, train_idx)
    print(f"  Mean divergence layer (correct):   {divergence['mean_divergence_correct']:.1f}")
    print(f"  Mean divergence layer (incorrect): {divergence['mean_divergence_incorrect']:.1f}")

    # Step 9: Diagnostic battery if needed
    diag_report = None
    if auroc_holdout < 0.796:
        print("\n[WARN] AUROC < 0.796 baseline — running diagnostic battery ...")
        diag_report = diagnostic_battery(
            features_all, labels, train_idx, holdout_idx,
            baseline_auroc=0.796,
            abc_X_train=abc_X_train, abc_X_holdout=abc_X_holdout,
        )
        for d in diag_report["diagnostics"]:
            print(f"  {d['step']}: {json.dumps({k: v for k, v in d.items() if k != 'step'}, default=float)[:200]}")

    # Step 10: Save results
    results = {
        "experiment": "exp1_layerwise_ph",
        "n_features": int(features_all.shape[1]),
        "n_layers": N_TRANSFORMER_LAYERS,
        "n_pca_per_layer": 20,
        "auroc_holdout": round(auroc_holdout, 4),
        "auroc_cv": round(auroc_cv, 4),
        "baseline_auroc": round(abc_auroc, 4),
        "improvement": round(auroc_holdout - abc_auroc, 4),
        "delong": delong,
        "bootstrap_ci": ci,
        "c_sweep": c_results,
        "best_C": best_C,
        "divergence": {
            "mean_correct": round(divergence["mean_divergence_correct"], 2),
            "mean_incorrect": round(divergence["mean_divergence_incorrect"], 2),
            "alpha": divergence["alpha"],
        },
        "diagnostic_battery": diag_report,
        "feature_names": make_layerwise_feature_names(),
    }

    out_path = RESULTS_DIR / "exp1_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    np.save(RESULTS_DIR / "exp1_features_all.npy", features_all)
    np.save(RESULTS_DIR / "exp1_probs_holdout.npy", probs_holdout)

    mark_done(STAGE_NAME)

    print(f"\n{'='*70}")
    print(f"Experiment 1 Summary")
    print(f"{'='*70}")
    print(f"  Layer-wise PH AUROC: {auroc_holdout:.4f} (CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
    print(f"  ABC-44 baseline:     {abc_auroc:.4f}")
    print(f"  Improvement:         {auroc_holdout - abc_auroc:+.4f}")
    print(f"  DeLong p-value:      {delong.get('p_value', 'N/A')}")
    print(f"  Beats baseline:      {auroc_holdout > abc_auroc}")
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
