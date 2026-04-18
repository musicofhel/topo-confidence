#!/usr/bin/env python3
"""Phase 1: Refit prompt-level model + rerun Experiments 9, 6, 10 with corrected labels.

Features don't change (from hidden states). Only labels change.
CPU-only, ~2 minutes.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    TIER_ASSIGNMENTS_PATH,
    get_abc_column_indices,
    load_features_with_sss_fix,
    logger,
)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE0_DIR = Path(__file__).parent.parent / "phase0_relabel"
BASELINE_CORRECT_V2 = PHASE0_DIR / "baseline_correct_v2.npy"

# Old experiment 9 scores (pre-computed, don't depend on labels)
EXP9_SCORES_PATH = (
    Path(__file__).parent.parent.parent
    / "pathway4"
    / "track_a"
    / "experiment9_baselines"
    / "per_problem_scores.json"
)

# Feature names from Phase 1 extraction
FEATURE_NAMES_PATH = (
    Path(__file__).parent.parent.parent
    / "pathway4"
    / "track_a"
    / "phase1"
    / "feature_names.json"
)


def bootstrap_auroc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """AUROC with bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    auroc = roc_auc_score(y_true, scores)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], scores[idx]))
    boot = sorted(boot)
    return auroc, boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 1: REFIT PROMPT-LEVEL MODEL + RERUN EXPERIMENTS")
    logger.info("=" * 70)

    # ---- Load data ----
    new_correct = np.load(BASELINE_CORRECT_V2)
    old_correct = np.load(
        Path(__file__).parent.parent.parent
        / "pathway2"
        / "track_a"
        / "phase0"
        / "baseline_correct.npy"
    )

    # Load features with SSS fix (using OLD labels for SSS split reconstruction,
    # since the SSS split was done with old labels)
    X_train, X_holdout, train_idx, holdout_idx, _, _ = load_features_with_sss_fix(
        baseline_correct=old_correct
    )

    # NEW labels aligned with sorted indices
    y_train = new_correct[train_idx].astype(int)
    y_holdout = new_correct[holdout_idx].astype(int)

    # OLD labels for comparison
    y_train_old = old_correct[train_idx].astype(int)
    y_holdout_old = old_correct[holdout_idx].astype(int)

    with open(FEATURE_NAMES_PATH) as f:
        feature_names_78 = json.load(f)

    abc_indices = get_abc_column_indices(feature_names_78)
    abc_names = [feature_names_78[i] for i in abc_indices]

    X_train_abc = X_train[:, abc_indices]
    X_holdout_abc = X_holdout[:, abc_indices]

    logger.info("  Train: %d problems, %d correct (was %d)", len(y_train), y_train.sum(), y_train_old.sum())
    logger.info("  Holdout: %d problems, %d correct (was %d)", len(y_holdout), y_holdout.sum(), y_holdout_old.sum())
    logger.info("  Features: %d ABC-tier", len(abc_indices))

    # ==================================================================
    # SECTION 1: REFIT CONFIDENCE MODEL
    # ==================================================================
    logger.info("\n--- Section 1: Refit Confidence Model ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_abc)
    X_holdout_scaled = scaler.transform(X_holdout_abc)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)

    # Holdout scores
    holdout_scores = clf.predict_proba(X_holdout_scaled)[:, 1]
    auroc, ci_lo, ci_hi = bootstrap_auroc(y_holdout, holdout_scores)

    logger.info("  NEW holdout AUROC: %.4f [%.4f - %.4f]", auroc, ci_lo, ci_hi)
    logger.info("  (was 0.935 with old labels)")

    # Train CV AUROC (50-fold for consistency with original)
    cv_scores = cross_val_predict(
        LogisticRegression(**LR_PARAMS),
        X_train_scaled,
        y_train,
        cv=StratifiedKFold(n_splits=50, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    train_cv_auroc = roc_auc_score(y_train, cv_scores)
    logger.info("  Train CV-50 AUROC: %.4f", train_cv_auroc)

    # Save model
    with open(OUTPUT_DIR / "model_v2.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "abc_indices": abc_indices}, f)

    # Save per-problem scores
    per_problem_scores = {
        "holdout_indices": holdout_idx.tolist(),
        "holdout_correct_v2": y_holdout.tolist(),
        "holdout_correct_old": y_holdout_old.tolist(),
        "topo_scores": holdout_scores.tolist(),
    }
    with open(OUTPUT_DIR / "per_problem_scores_v2.json", "w") as f:
        json.dump(per_problem_scores, f, indent=2)

    holdout_metrics = {
        "auroc": auroc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "train_cv_auroc": train_cv_auroc,
        "old_auroc": 0.935,
        "n_train_correct": int(y_train.sum()),
        "n_holdout_correct": int(y_holdout.sum()),
    }

    # ==================================================================
    # SECTION 2: EXPERIMENT 9 — OUTPUT PROBABILITY BASELINES
    # ==================================================================
    logger.info("\n--- Section 2: Experiment 9 (Baselines) with Corrected Labels ---")

    with open(EXP9_SCORES_PATH) as f:
        exp9_data = json.load(f)

    # Verify holdout indices match
    assert exp9_data["holdout_indices"] == holdout_idx.tolist(), "Holdout index mismatch!"

    exp9_results = {}
    all_baseline_names = [
        "topo_confidence",
        "neg_mean_entropy",
        "mean_max_token_prob",
        "first_token_prob",
        "self_certainty",
        "p_majority",
        "vote_margin",
        "inv_answer_diversity",
        "neg_agreement_entropy",
    ]

    for name in all_baseline_names:
        if name not in exp9_data["scores"]:
            continue
        scores = np.array(exp9_data["scores"][name])

        # With NEW labels
        new_auroc, new_lo, new_hi = bootstrap_auroc(y_holdout, scores)

        # With OLD labels (for comparison)
        old_auroc = roc_auc_score(y_holdout_old, scores)

        exp9_results[name] = {
            "new_auroc": new_auroc,
            "new_ci": [new_lo, new_hi],
            "old_auroc": old_auroc,
            "delta": new_auroc - old_auroc,
        }

        marker = " ***" if name == "topo_confidence" else ""
        logger.info(
            "  %s: OLD=%.4f -> NEW=%.4f (delta=%+.4f)%s",
            name.ljust(25), old_auroc, new_auroc, new_auroc - old_auroc, marker,
        )

    # Topo vs best baseline gap
    topo_new = exp9_results["topo_confidence"]["new_auroc"]
    baselines_new = {
        k: v["new_auroc"]
        for k, v in exp9_results.items()
        if k != "topo_confidence"
    }
    best_baseline_name = max(baselines_new, key=baselines_new.get)
    best_baseline_auroc = baselines_new[best_baseline_name]
    gap = topo_new - best_baseline_auroc
    logger.info(
        "\n  Topo vs best baseline (%s): %.4f - %.4f = %.4f (was +0.208)",
        best_baseline_name, topo_new, best_baseline_auroc, gap,
    )

    with open(OUTPUT_DIR / "experiment9_v2.json", "w") as f:
        json.dump(
            {
                "baselines": exp9_results,
                "best_baseline": best_baseline_name,
                "topo_vs_best_gap": gap,
            },
            f,
            indent=2,
        )

    # ==================================================================
    # SECTION 3: EXPERIMENT 6 — FEATURE TIER ABLATION
    # ==================================================================
    logger.info("\n--- Section 3: Experiment 6 (Feature Ablation) with Corrected Labels ---")

    with open(TIER_ASSIGNMENTS_PATH) as f:
        tier_map = json.load(f)

    # Build tier index maps
    tier_indices = {"A": [], "B": [], "C": [], "D": []}
    for i, name in enumerate(feature_names_78):
        t = tier_map.get(name, "D")
        if t in tier_indices:
            tier_indices[t].append(i)

    # All 7 tier combinations + D reference
    tier_combos = {
        "A": tier_indices["A"],
        "B": tier_indices["B"],
        "C": tier_indices["C"],
        "A+B": tier_indices["A"] + tier_indices["B"],
        "A+C": tier_indices["A"] + tier_indices["C"],
        "B+C": tier_indices["B"] + tier_indices["C"],
        "A+B+C": sorted(tier_indices["A"] + tier_indices["B"] + tier_indices["C"]),
        "A+B+C+D": list(range(len(feature_names_78))),
    }

    tier_results = {}
    for combo_name, indices in tier_combos.items():
        if not indices:
            continue
        X_tr = X_train[:, indices]
        X_ho = X_holdout[:, indices]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_ho_s = sc.transform(X_ho)
        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(X_tr_s, y_train)
        scores = lr.predict_proba(X_ho_s)[:, 1]
        a, lo, hi = bootstrap_auroc(y_holdout, scores)
        tier_results[combo_name] = {
            "auroc": a,
            "ci": [lo, hi],
            "n_features": len(indices),
        }
        logger.info(
            "  %s (%d features): AUROC=%.4f [%.4f-%.4f]",
            combo_name.ljust(10), len(indices), a, lo, hi,
        )

    # LOO (Leave-One-Out feature ablation) on ABC tier
    logger.info("\n  LOO ablation (drop each of %d features):", len(abc_indices))
    full_auroc = auroc  # from Section 1
    loo_results = []
    for drop_i, drop_name in enumerate(abc_names):
        keep = [j for j in range(len(abc_indices)) if j != drop_i]
        X_tr = X_train_abc[:, keep]
        X_ho = X_holdout_abc[:, keep]
        sc = StandardScaler()
        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(sc.fit_transform(X_tr), y_train)
        a = roc_auc_score(y_holdout, lr.predict_proba(sc.transform(X_ho))[:, 1])
        delta = full_auroc - a
        loo_results.append({"feature": drop_name, "auroc_without": a, "delta": delta})

    loo_results.sort(key=lambda x: -x["delta"])
    for r in loo_results[:10]:
        logger.info(
            "    Drop %s: AUROC=%.4f (delta=%.4f)",
            r["feature"].ljust(30), r["auroc_without"], r["delta"],
        )

    with open(OUTPUT_DIR / "experiment6_v2.json", "w") as f:
        json.dump(
            {"tier_ablation": tier_results, "loo_top10": loo_results[:10]},
            f,
            indent=2,
        )

    # ==================================================================
    # SECTION 4: EXPERIMENT 10 — CALIBRATION
    # ==================================================================
    logger.info("\n--- Section 4: Experiment 10 (Calibration) with Corrected Labels ---")

    # ECE (Expected Calibration Error) — equal-width bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_ew = 0.0
    reliability_data = []
    for b in range(n_bins):
        mask = (holdout_scores >= bin_edges[b]) & (holdout_scores < bin_edges[b + 1])
        if b == n_bins - 1:
            mask |= holdout_scores == bin_edges[b + 1]
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            reliability_data.append({"bin": b, "count": 0, "mean_pred": None, "mean_obs": None})
            continue
        mean_pred = holdout_scores[mask].mean()
        mean_obs = y_holdout[mask].mean()
        ece_ew += n_in_bin * abs(mean_pred - mean_obs)
        reliability_data.append({
            "bin": b,
            "count": int(n_in_bin),
            "mean_pred": float(mean_pred),
            "mean_obs": float(mean_obs),
        })
    ece_ew /= len(y_holdout)

    # Brier score
    brier = brier_score_loss(y_holdout, holdout_scores)

    logger.info("  ECE (equal-width): %.4f", ece_ew)
    logger.info("  Brier score: %.4f", brier)
    logger.info("  (old ECE=0.118, old Brier=0.087)")

    calibration_results = {
        "ece_equal_width": ece_ew,
        "brier": brier,
        "reliability_data": reliability_data,
        "old_ece": 0.118,
        "old_brier": 0.087,
    }
    with open(OUTPUT_DIR / "experiment10_v2.json", "w") as f:
        json.dump(calibration_results, f, indent=2)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 70)
    logger.info("Headline AUROC: 0.935 -> %.4f (delta=%+.4f)", auroc, auroc - 0.935)
    logger.info(
        "Topo vs best baseline: +0.208 -> %+.4f (%s)",
        gap, best_baseline_name,
    )
    logger.info("ECE: 0.118 -> %.4f", ece_ew)
    logger.info("Brier: 0.087 -> %.4f", brier)
    logger.info("Elapsed: %.1f seconds", time.time() - t0)

    # Save combined metrics
    holdout_metrics["experiment9"] = exp9_results
    holdout_metrics["experiment6_tier_a_auroc"] = tier_results.get("A", {}).get("auroc")
    holdout_metrics["experiment10"] = calibration_results
    with open(OUTPUT_DIR / "holdout_metrics_v2.json", "w") as f:
        json.dump(holdout_metrics, f, indent=2)


if __name__ == "__main__":
    main()
