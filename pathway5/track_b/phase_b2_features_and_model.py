#!/usr/bin/env python3
"""Phase B2: 7B MATH-500 Feature Extraction + Confidence Model + Baselines.

CPU script. Computes 78 topo features from Phase B1 7B trajectories (hidden_dim=3584),
fits confidence model on train-400, evaluates on holdout-100. Compares with 1.5B results.

PCA handles the different hidden_dim automatically (projects 3584-D → 45-D).
Feature definitions from winning_features.py remain the same.

Runtime: ~1-4 hours CPU (PH computation for 500 problems).
Input: pathway5/track_b/phase_b1/
Output: pathway5/track_b/phase_b2/
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    N_PROBLEMS_MATH,
    PATHWAY1_PHASE0_DIR,
    SEED,
    TRACK_B_DIR,
    bootstrap_auroc,
    auroc_permutation_test,
    extract_topo_features,
    fit_pca_on_trajectories,
    get_abc_column_indices,
    get_train_holdout_indices,
    logger,
)

PHASE_B1_DIR = TRACK_B_DIR / "phase_b1"
PHASE_DIR = TRACK_B_DIR / "phase_b2"
QWEN_VOCAB_SIZE = 151936


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load Phase B1 artifacts ----
    logger.info("=== Loading Phase B1 artifacts ===")

    baseline_correct = np.load(PHASE_B1_DIR / "math500_7b_baseline_correct.npy")
    train_idx, holdout_idx = get_train_holdout_indices()

    y_train = baseline_correct[train_idx]
    y_holdout = baseline_correct[holdout_idx]
    logger.info(
        "7B MATH-500: %d/%d correct (%.1f%%), train=%d correct, holdout=%d correct",
        baseline_correct.sum(), N_PROBLEMS_MATH,
        100 * baseline_correct.mean(),
        y_train.sum(), y_holdout.sum(),
    )

    # Load trajectories
    traj_data = np.load(PHASE_B1_DIR / "math500_7b_trajectories.npz")
    trajectories = [traj_data[f"traj_{i}"] for i in range(N_PROBLEMS_MATH)]
    logger.info("Loaded %d trajectories (hidden_dim=%d)", len(trajectories), trajectories[0].shape[1])

    # Load layer states
    layer_states = np.load(PHASE_B1_DIR / "math500_7b_layer_states.npy")
    logger.info("Layer states: %s", layer_states.shape)

    # Load entropy scores
    with open(PHASE_B1_DIR / "math500_7b_entropy_scores.json") as f:
        entropy_scores = json.load(f)

    # ---- Step 2: Feature extraction (fresh PCA on 7B train trajectories) ----
    logger.info("=== Feature extraction (PCA on 7B train data) ===")
    t_feat = time.time()

    train_trajectories = [trajectories[i] for i in train_idx]
    pca = fit_pca_on_trajectories(train_trajectories)

    features, feature_names = extract_topo_features(
        trajectories, layer_states, prefitted_pca=pca,
    )
    logger.info(
        "Features: %s, %d names, took %.1f min",
        features.shape, len(feature_names), (time.time() - t_feat) / 60,
    )

    np.save(PHASE_DIR / "math500_7b_features.npy", features)
    with open(PHASE_DIR / "math500_7b_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ---- Step 3: Confidence model ----
    logger.info("=== Confidence model ===")
    abc_indices = get_abc_column_indices(feature_names)

    X_train = features[train_idx][:, abc_indices]
    X_holdout = features[holdout_idx][:, abc_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)

    # Holdout AUROC
    holdout_scores = clf.predict_proba(X_holdout_scaled)[:, 1]
    holdout_auroc, ho_ci_lo, ho_ci_hi = bootstrap_auroc(y_holdout, holdout_scores)
    logger.info(
        "7B Topo AUROC (holdout): %.3f [%.3f-%.3f]",
        holdout_auroc, ho_ci_lo, ho_ci_hi,
    )

    # Train CV
    cv_scores = cross_val_predict(
        LogisticRegression(**LR_PARAMS),
        X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    cv_auroc = roc_auc_score(y_train, cv_scores)
    logger.info("7B Topo AUROC (5-fold train CV): %.3f", cv_auroc)

    # Save all scores
    all_scores = clf.predict_proba(
        scaler.transform(features[:, abc_indices])
    )[:, 1]
    np.save(PHASE_DIR / "math500_7b_topo_scores.npy", all_scores)

    # ---- Step 4: Output-probability baselines ----
    logger.info("=== Output-probability baselines ===")

    baseline_names = ["neg_mean_entropy", "mean_max_token_prob", "first_token_prob", "self_certainty"]
    baseline_scores_dict = {}

    for i in range(N_PROBLEMS_MATH):
        es = entropy_scores.get(str(i), {})
        baseline_scores_dict.setdefault("neg_mean_entropy", []).append(-es.get("entropy", 0.0))
        baseline_scores_dict.setdefault("mean_max_token_prob", []).append(es.get("max_token_prob", 0.0))
        baseline_scores_dict.setdefault("first_token_prob", []).append(es.get("first_token_prob", 0.0))
        baseline_scores_dict.setdefault("self_certainty", []).append(
            np.log(QWEN_VOCAB_SIZE) - es.get("entropy", 0.0)
        )

    baseline_aurocs = {}
    for name in baseline_names:
        scores_arr = np.array(baseline_scores_dict[name])
        ho_scores = scores_arr[holdout_idx]
        try:
            bl_auroc, bl_ci_lo, bl_ci_hi = bootstrap_auroc(y_holdout, ho_scores)
            baseline_aurocs[name] = {"auroc": bl_auroc, "ci_lo": bl_ci_lo, "ci_hi": bl_ci_hi}
            logger.info("  %s: AUROC %.3f [%.3f-%.3f]", name, bl_auroc, bl_ci_lo, bl_ci_hi)
        except ValueError as e:
            logger.warning("  %s: AUROC failed: %s", name, e)
            baseline_aurocs[name] = {"auroc": None, "error": str(e)}

    # Best baseline
    best_baseline_name = max(
        (n for n in baseline_aurocs if baseline_aurocs[n].get("auroc") is not None),
        key=lambda n: baseline_aurocs[n]["auroc"],
        default=None,
    )

    # Statistical comparison
    if best_baseline_name:
        p_val = auroc_permutation_test(
            y_holdout, holdout_scores,
            np.array(baseline_scores_dict[best_baseline_name])[holdout_idx],
        )
        logger.info("Topo vs best baseline (%s): p=%.4f", best_baseline_name, p_val)
    else:
        p_val = None

    # ---- Step 5: Feature importance comparison with 1.5B ----
    logger.info("=== Feature importance comparison ===")

    # 7B top features
    coef_7b = np.abs(clf.coef_[0])
    top_7b = np.argsort(coef_7b)[::-1][:10]
    abc_names = [feature_names[i] for i in abc_indices]
    logger.info("7B top-10 features by |coef|:")
    for rank, idx in enumerate(top_7b):
        logger.info("  %d. %s (|coef|=%.4f)", rank + 1, abc_names[idx], coef_7b[idx])

    # Save feature importance
    feature_importance = {
        "feature_names": abc_names,
        "coef_abs": coef_7b.tolist(),
        "top_10": [(abc_names[i], float(coef_7b[i])) for i in top_7b],
    }
    with open(PHASE_DIR / "feature_importance_7b.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    # ---- Step 6: Results ----
    elapsed = time.time() - t0
    results = {
        "dataset": "MATH-500",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "n_problems": N_PROBLEMS_MATH,
        "n_train": len(train_idx),
        "n_holdout": len(holdout_idx),
        "greedy_accuracy": {
            "total": round(float(baseline_correct.mean()), 4),
            "train": round(float(y_train.mean()), 4),
            "holdout": round(float(y_holdout.mean()), 4),
        },
        "topo_confidence": {
            "holdout_auroc": holdout_auroc,
            "holdout_ci": [ho_ci_lo, ho_ci_hi],
            "train_cv_auroc": cv_auroc,
        },
        "baselines": baseline_aurocs,
        "best_baseline": best_baseline_name,
        "topo_vs_best_baseline_p": p_val,
        "n_features": len(feature_names),
        "n_abc_features": len(abc_indices),
        "elapsed_minutes": round(elapsed / 60, 1),
        "comparison_with_1_5b": {
            "1_5b_topo_auroc": 0.935,
            "1_5b_best_logprob": 0.473,
            "1_5b_greedy_acc": 0.114,
            "7b_topo_auroc": holdout_auroc,
            "7b_best_logprob": (
                baseline_aurocs[best_baseline_name]["auroc"]
                if best_baseline_name else None
            ),
            "7b_greedy_acc": round(float(baseline_correct.mean()), 3),
        },
    }

    with open(PHASE_DIR / "phase_b2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Print comparison ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE B2 RESULTS — 7B MATH-500 Topo-Confidence")
    logger.info("=" * 60)
    logger.info("7B greedy accuracy: %d/%d (%.1f%%)",
                baseline_correct.sum(), N_PROBLEMS_MATH, 100 * baseline_correct.mean())
    logger.info("7B topo AUROC (holdout): %.3f [%.3f-%.3f]", holdout_auroc, ho_ci_lo, ho_ci_hi)
    logger.info("7B topo AUROC (train CV): %.3f", cv_auroc)
    logger.info("--- vs 1.5B ---")
    logger.info("1.5B topo AUROC: 0.935, 1.5B greedy acc: 11.4%%")
    logger.info("Elapsed: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
