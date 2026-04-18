#!/usr/bin/env python3
"""Phase A2: GSM8K Feature Extraction + Confidence Model + Baselines.

CPU script. Computes 78 topo features from Phase A1 trajectories, fits
confidence model, evaluates AUROC against baselines. Tests both fresh
PCA and transfer PCA (from MATH-500 training data).

Runtime: ~1-4 hours CPU (PH computation for 1319 problems).
Input: pathway5/track_a/phase_a1/
Output: pathway5/track_a/phase_a2/
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
    FEATURES_TRAIN_PATH,
    GSM8K_SPLIT_SEED,
    GSM8K_TEST_FRACTION,
    LR_PARAMS,
    PATHWAY1_PHASE0_DIR,
    SEED,
    TRAJ_PATH,
    TRACK_A_DIR,
    bootstrap_auroc,
    auroc_permutation_test,
    extract_topo_features,
    fit_pca_on_trajectories,
    get_abc_column_indices,
    get_gsm8k_train_test_indices,
    load_cached_trajectories,
    logger,
)

PHASE_A1_DIR = TRACK_A_DIR / "phase_a1"
PHASE_DIR = TRACK_A_DIR / "phase_a2"
QWEN_VOCAB_SIZE = 151936


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: Load Phase A1 artifacts ----
    logger.info("=== Loading Phase A1 artifacts ===")

    baseline_correct = np.load(PHASE_A1_DIR / "gsm8k_baseline_correct.npy")
    n_problems = len(baseline_correct)
    logger.info("Loaded %d problems, %d correct (%.1f%%)",
                n_problems, baseline_correct.sum(), 100 * baseline_correct.mean())

    # Load trajectories
    traj_data = np.load(PHASE_A1_DIR / "gsm8k_trajectories.npz")
    trajectories = [traj_data[f"traj_{i}"] for i in range(n_problems)]
    logger.info("Loaded %d trajectories", len(trajectories))

    # Load layer states
    layer_states = np.load(PHASE_A1_DIR / "gsm8k_layer_states.npy")
    logger.info("Loaded layer states: %s", layer_states.shape)

    # Load entropy scores
    with open(PHASE_A1_DIR / "gsm8k_entropy_scores.json") as f:
        entropy_scores = json.load(f)

    # ---- Step 2: Train/test split ----
    train_idx, test_idx = get_gsm8k_train_test_indices(
        n_problems, baseline_correct, GSM8K_TEST_FRACTION, GSM8K_SPLIT_SEED,
    )
    y_train = baseline_correct[train_idx]
    y_test = baseline_correct[test_idx]
    logger.info(
        "Split: train=%d (%d correct), test=%d (%d correct)",
        len(train_idx), y_train.sum(), len(test_idx), y_test.sum(),
    )

    # Save split
    np.save(PHASE_DIR / "gsm8k_train_idx.npy", train_idx)
    np.save(PHASE_DIR / "gsm8k_test_idx.npy", test_idx)

    # ---- Step 3: Feature extraction (FRESH PCA on GSM8K) ----
    logger.info("=== Feature extraction (Fresh PCA) ===")
    t_feat = time.time()

    # Fit PCA on GSM8K train trajectories only (prevent test leakage)
    train_trajectories = [trajectories[i] for i in train_idx]
    fresh_pca = fit_pca_on_trajectories(train_trajectories)

    # Extract features for ALL problems (using train PCA)
    features_fresh, feature_names = extract_topo_features(
        trajectories, layer_states, prefitted_pca=fresh_pca,
    )
    logger.info(
        "Fresh PCA features: %s, %d feature names, took %.1f min",
        features_fresh.shape, len(feature_names), (time.time() - t_feat) / 60,
    )

    np.save(PHASE_DIR / "gsm8k_features_fresh_pca.npy", features_fresh)
    with open(PHASE_DIR / "gsm8k_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ---- Step 4: Feature extraction (TRANSFER PCA from MATH-500) ----
    logger.info("=== Feature extraction (Transfer PCA from MATH-500) ===")
    t_xfer = time.time()

    # Load MATH-500 training trajectories and fit PCA on them
    math_trajectories = load_cached_trajectories()  # 500 MATH-500 trajectories
    # Need to check hidden_dim compatibility
    math_hidden_dim = math_trajectories[0].shape[1]
    gsm8k_hidden_dim = trajectories[0].shape[1]

    if math_hidden_dim == gsm8k_hidden_dim:
        # Same model → transfer PCA makes sense
        math_pca = fit_pca_on_trajectories(math_trajectories)
        features_transfer, _ = extract_topo_features(
            trajectories, layer_states, prefitted_pca=math_pca,
        )
        np.save(PHASE_DIR / "gsm8k_features_transfer_pca.npy", features_transfer)
        transfer_pca_available = True
        logger.info(
            "Transfer PCA features: %s, took %.1f min",
            features_transfer.shape, (time.time() - t_xfer) / 60,
        )
    else:
        logger.warning(
            "Hidden dim mismatch: MATH=%d, GSM8K=%d. Skipping transfer PCA.",
            math_hidden_dim, gsm8k_hidden_dim,
        )
        features_transfer = None
        transfer_pca_available = False

    # ---- Step 5: Confidence model (Fresh PCA) ----
    logger.info("=== Confidence model fitting ===")
    abc_indices = get_abc_column_indices(feature_names)

    # Train on GSM8K train split
    X_train = features_fresh[train_idx][:, abc_indices]
    X_test = features_fresh[test_idx][:, abc_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)

    # Predict on test
    test_scores = clf.predict_proba(X_test_scaled)[:, 1]
    fresh_auroc, fresh_ci_lo, fresh_ci_hi = bootstrap_auroc(y_test, test_scores)
    logger.info(
        "Fresh PCA AUROC (test): %.3f [%.3f-%.3f]",
        fresh_auroc, fresh_ci_lo, fresh_ci_hi,
    )

    # CV on train for robustness
    cv_scores = cross_val_predict(
        LogisticRegression(**LR_PARAMS),
        X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    cv_auroc = roc_auc_score(y_train, cv_scores)
    logger.info("Fresh PCA AUROC (5-fold train CV): %.3f", cv_auroc)

    # Save per-problem scores
    all_scores_fresh = clf.predict_proba(
        scaler.transform(features_fresh[:, abc_indices])
    )[:, 1]
    np.save(PHASE_DIR / "gsm8k_topo_scores_fresh.npy", all_scores_fresh)

    # ---- Step 6: Transfer PCA confidence model ----
    transfer_results = {}
    if transfer_pca_available:
        X_train_xfer = features_transfer[train_idx][:, abc_indices]
        X_test_xfer = features_transfer[test_idx][:, abc_indices]

        scaler_xfer = StandardScaler()
        X_train_xfer_scaled = scaler_xfer.fit_transform(X_train_xfer)
        X_test_xfer_scaled = scaler_xfer.transform(X_test_xfer)

        clf_xfer = LogisticRegression(**LR_PARAMS)
        clf_xfer.fit(X_train_xfer_scaled, y_train)

        test_scores_xfer = clf_xfer.predict_proba(X_test_xfer_scaled)[:, 1]
        xfer_auroc, xfer_ci_lo, xfer_ci_hi = bootstrap_auroc(y_test, test_scores_xfer)
        logger.info(
            "Transfer PCA AUROC (test): %.3f [%.3f-%.3f]",
            xfer_auroc, xfer_ci_lo, xfer_ci_hi,
        )

        transfer_results = {
            "auroc": xfer_auroc,
            "ci_lo": xfer_ci_lo,
            "ci_hi": xfer_ci_hi,
        }

        all_scores_xfer = clf_xfer.predict_proba(
            scaler_xfer.transform(features_transfer[:, abc_indices])
        )[:, 1]
        np.save(PHASE_DIR / "gsm8k_topo_scores_transfer.npy", all_scores_xfer)

    # ---- Step 7: Output-probability baselines ----
    logger.info("=== Output-probability baselines ===")

    # Greedy-pass baselines from entropy scores
    baseline_names = ["neg_mean_entropy", "mean_max_token_prob", "first_token_prob", "self_certainty"]
    baseline_scores = {}

    for i in range(n_problems):
        es = entropy_scores.get(str(i), {})
        baseline_scores.setdefault("neg_mean_entropy", []).append(-es.get("entropy", 0.0))
        baseline_scores.setdefault("mean_max_token_prob", []).append(es.get("max_token_prob", 0.0))
        baseline_scores.setdefault("first_token_prob", []).append(es.get("first_token_prob", 0.0))
        # Self-certainty: log(vocab_size) - entropy
        baseline_scores.setdefault("self_certainty", []).append(
            np.log(QWEN_VOCAB_SIZE) - es.get("entropy", 0.0)
        )

    baseline_aurocs = {}
    for name in baseline_names:
        scores_arr = np.array(baseline_scores[name])
        test_scores_bl = scores_arr[test_idx]
        try:
            bl_auroc, bl_ci_lo, bl_ci_hi = bootstrap_auroc(y_test, test_scores_bl)
            baseline_aurocs[name] = {
                "auroc": bl_auroc, "ci_lo": bl_ci_lo, "ci_hi": bl_ci_hi,
            }
            logger.info("  %s: AUROC %.3f [%.3f-%.3f]", name, bl_auroc, bl_ci_lo, bl_ci_hi)
        except ValueError as e:
            logger.warning("  %s: AUROC failed: %s", name, e)
            baseline_aurocs[name] = {"auroc": None, "error": str(e)}

    # ---- Step 8: Statistical comparison ----
    best_baseline_name = max(
        (n for n in baseline_aurocs if baseline_aurocs[n].get("auroc") is not None),
        key=lambda n: baseline_aurocs[n]["auroc"],
        default=None,
    )
    if best_baseline_name:
        p_val = auroc_permutation_test(
            y_test, test_scores,
            np.array(baseline_scores[best_baseline_name])[test_idx],
        )
        logger.info(
            "Topo vs best baseline (%s): p=%.4f",
            best_baseline_name, p_val,
        )
    else:
        p_val = None

    # ---- Step 9: Results ----
    elapsed = time.time() - t0
    results = {
        "dataset": "GSM8K",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_problems": n_problems,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "greedy_accuracy": {
            "total": round(baseline_correct.mean(), 4),
            "train": round(y_train.mean(), 4),
            "test": round(y_test.mean(), 4),
        },
        "topo_confidence": {
            "fresh_pca": {
                "test_auroc": fresh_auroc,
                "test_ci": [fresh_ci_lo, fresh_ci_hi],
                "train_cv_auroc": cv_auroc,
            },
            "transfer_pca": transfer_results if transfer_pca_available else "skipped",
        },
        "baselines": baseline_aurocs,
        "best_baseline": best_baseline_name,
        "topo_vs_best_baseline_p": p_val,
        "n_features": len(feature_names),
        "n_abc_features": len(abc_indices),
        "elapsed_minutes": round(elapsed / 60, 1),
        "comparison_with_math500": {
            "math500_topo_auroc": 0.935,
            "math500_best_logprob_auroc": 0.473,
            "gsm8k_topo_auroc": fresh_auroc,
            "gsm8k_best_logprob_auroc": (
                baseline_aurocs[best_baseline_name]["auroc"]
                if best_baseline_name else None
            ),
        },
    }

    with open(PHASE_DIR / "phase_a2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Print summary ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE A2 RESULTS — GSM8K Topo-Confidence")
    logger.info("=" * 60)
    logger.info("Greedy accuracy: %d/%d (%.1f%%)", baseline_correct.sum(), n_problems, 100 * baseline_correct.mean())
    logger.info("Topo AUROC (fresh PCA, test): %.3f [%.3f-%.3f]", fresh_auroc, fresh_ci_lo, fresh_ci_hi)
    if transfer_pca_available:
        logger.info("Topo AUROC (transfer PCA, test): %.3f [%.3f-%.3f]",
                     transfer_results["auroc"], transfer_results["ci_lo"], transfer_results["ci_hi"])
    logger.info("Topo AUROC (5-fold train CV): %.3f", cv_auroc)
    for name in baseline_names:
        if baseline_aurocs[name].get("auroc") is not None:
            logger.info("  %s: %.3f", name, baseline_aurocs[name]["auroc"])
    if p_val is not None:
        logger.info("Topo vs best baseline p-value: %.4f", p_val)
    logger.info("Elapsed: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
