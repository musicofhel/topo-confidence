#!/usr/bin/env python3
"""Phase 2 Step 2: Selection strategy evaluation with corrected labels.

This script handles the CPU-only parts of Phase 2:
- Holdout selection evaluation (MV, weighted vote, etc.) with corrected labels
- Experiment 1 (MV ablation) with corrected labels
- Experiment 8 (adaptive sampling) with corrected labels

For holdout selection, we use existing temperature completions (all 85 wrong-holdout
problems already have temp gens) and the retrained prompt-level model for gating.

NOTE: Per-completion features need PCA fix + GPU re-extraction (step1).
The Experiment 5 (per-completion AUROC) section is DEFERRED until after RunPod.
This script runs the parts that DON'T depend on per-completion features.
"""

from __future__ import annotations

import json
import pickle  # noqa: F401
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_PATH,
    LR_PARAMS,
    check_correct_v2,
    extract_answer_v2,
    get_abc_column_indices,
    get_train_holdout_indices,
    get_prompts_and_ground_truths,
    load_features_with_sss_fix,
    load_math500,
    logger,
    normalize_answer_v2,
    select_confidence_weighted_vote_v2,
    select_majority_vote_v2,
    select_max_confidence_v2,
    select_random_v2,
)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE0_DIR = Path(__file__).parent.parent / "phase0_relabel"
PHASE1_DIR = Path(__file__).parent.parent / "phase1_prompt_model"
BASELINE_CORRECT_V2 = PHASE0_DIR / "baseline_correct_v2.npy"
HOLDOUT_TEMP_V2 = PHASE0_DIR / "holdout_temperature_generations_v2.json"
TRAIN_TEMP_V2 = PHASE0_DIR / "temperature_generations_v2.json"

# Old experiment results for comparison
OLD_BASELINE_CORRECT = (
    Path(__file__).parent.parent.parent
    / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
)


# ---- Per-completion score management ----
# Populated by main() if step1 features are available; otherwise uniform (= MV)
_COMP_SCORES_BY_PROBLEM: dict[int, np.ndarray] = {}
_HAS_REAL_SCORES = False


def _get_completion_scores(problem_idx: int, n_completions: int) -> np.ndarray:
    """Get per-completion topo scores for a problem. Falls back to uniform."""
    if problem_idx in _COMP_SCORES_BY_PROBLEM:
        return _COMP_SCORES_BY_PROBLEM[problem_idx]
    return np.ones(n_completions)


def load_and_score_completions(train_idx, holdout_idx):
    """Load per-completion features from step1 and fit scorer.

    Returns True if real scores were loaded, False if step1 artifacts missing.
    """
    global _COMP_SCORES_BY_PROBLEM, _HAS_REAL_SCORES

    train_feat_path = OUTPUT_DIR / "train_completion_features_v2.npz"
    holdout_feat_path = OUTPUT_DIR / "holdout_completion_features_v2.npz"
    feat_names_path = OUTPUT_DIR / "feature_names_v2.json"

    if not train_feat_path.exists() or not holdout_feat_path.exists():
        logger.info("Step1 features not found — using uniform scores (= MV)")
        return False

    logger.info("Loading per-completion features from step1...")
    train_data = np.load(train_feat_path)
    holdout_data = np.load(holdout_feat_path)
    with open(feat_names_path) as f:
        feature_names = json.load(f)

    train_features = train_data["features"]
    train_correct = train_data["correct"]
    train_map = train_data["completion_map"]
    holdout_features = holdout_data["features"]
    holdout_correct = holdout_data["correct"]
    holdout_map = holdout_data["completion_map"]

    logger.info("  Train completions: %d, Holdout completions: %d",
                len(train_features), len(holdout_features))

    # Fit scorer on train completions
    abc_idx = get_abc_column_indices(feature_names)
    X_train = train_features[:, abc_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_s, train_correct.astype(int))

    # Score holdout completions
    X_holdout = holdout_features[:, abc_idx]
    X_holdout_s = scaler.transform(X_holdout)
    holdout_scores = clf.predict_proba(X_holdout_s)[:, 1]

    # Build per-problem score lookup
    for i, (pidx, cidx) in enumerate(holdout_map):
        pidx = int(pidx)
        if pidx not in _COMP_SCORES_BY_PROBLEM:
            _COMP_SCORES_BY_PROBLEM[pidx] = []
        _COMP_SCORES_BY_PROBLEM[pidx].append(float(holdout_scores[i]))

    # Convert lists to arrays
    for pidx in _COMP_SCORES_BY_PROBLEM:
        _COMP_SCORES_BY_PROBLEM[pidx] = np.array(_COMP_SCORES_BY_PROBLEM[pidx])

    _HAS_REAL_SCORES = True

    # Save scorer
    with open(OUTPUT_DIR / "completion_scorer_v2.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "abc_idx": abc_idx}, f)

    return True


def run_experiment5(train_idx, holdout_idx):
    """Experiment 5: Per-completion verifier analysis.

    Requires step1 features. Returns None if unavailable.
    """
    train_feat_path = OUTPUT_DIR / "train_completion_features_v2.npz"
    holdout_feat_path = OUTPUT_DIR / "holdout_completion_features_v2.npz"
    feat_names_path = OUTPUT_DIR / "feature_names_v2.json"

    if not train_feat_path.exists() or not holdout_feat_path.exists():
        return None

    train_data = np.load(train_feat_path)
    holdout_data = np.load(holdout_feat_path)
    with open(feat_names_path) as f:
        feature_names = json.load(f)

    train_features = train_data["features"]
    train_correct = train_data["correct"]
    train_map = train_data["completion_map"]
    holdout_features = holdout_data["features"]
    holdout_correct = holdout_data["correct"]
    holdout_map = holdout_data["completion_map"]

    abc_idx = get_abc_column_indices(feature_names)

    # Global AUROC via 5-fold CV on train completions
    X = train_features[:, abc_idx]
    y = train_correct.astype(int)
    problem_ids = train_map[:, 0].astype(int)

    # Group by problem for stratified folds
    unique_problems = np.unique(problem_ids)
    problem_labels = np.array([y[problem_ids == p].mean() > 0.5 for p in unique_problems]).astype(int)

    if len(np.unique(problem_labels)) < 2:
        logger.warning("Experiment 5: single class in train, skipping CV")
        return None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.zeros(len(X))

    for fold_train, fold_test in skf.split(unique_problems, problem_labels):
        fold_train_problems = set(unique_problems[fold_train])
        fold_test_problems = set(unique_problems[fold_test])
        train_mask = np.array([p in fold_train_problems for p in problem_ids])
        test_mask = np.array([p in fold_test_problems for p in problem_ids])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])

        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_tr, y[train_mask])
        cv_scores[test_mask] = clf.predict_proba(X_te)[:, 1]

    if len(np.unique(y)) >= 2:
        global_auroc = float(roc_auc_score(y, cv_scores))
    else:
        global_auroc = None

    # Within-problem AUROC
    within_aurocs = []
    for p in unique_problems:
        mask = problem_ids == p
        y_p = y[mask]
        s_p = cv_scores[mask]
        if len(np.unique(y_p)) >= 2:
            within_aurocs.append(float(roc_auc_score(y_p, s_p)))

    # Holdout AUROC (train on all train, score holdout)
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X)
    clf_final = LogisticRegression(**LR_PARAMS)
    clf_final.fit(X_all_s, y)

    X_hold = holdout_features[:, abc_idx]
    X_hold_s = scaler_final.transform(X_hold)
    hold_scores = clf_final.predict_proba(X_hold_s)[:, 1]

    if len(np.unique(holdout_correct)) >= 2:
        holdout_auroc = float(roc_auc_score(holdout_correct, hold_scores))
    else:
        holdout_auroc = None

    results = {
        "global_auroc_cv": global_auroc,
        "within_problem_auroc_mean": float(np.mean(within_aurocs)) if within_aurocs else None,
        "within_problem_auroc_n": len(within_aurocs),
        "holdout_auroc": holdout_auroc,
        "n_train_completions": int(len(train_features)),
        "n_holdout_completions": int(len(holdout_features)),
        "n_train_correct": int(train_correct.sum()),
        "n_holdout_correct": int(holdout_correct.sum()),
    }
    return results


def evaluate_holdout_selection(
    new_correct: np.ndarray,
    holdout_idx: np.ndarray,
    holdout_temp: dict,
    ground_truths: list[str],
    holdout_scores: np.ndarray | None = None,
    tau: float = 0.3,
) -> dict:
    """Evaluate all selection strategies on holdout with corrected labels.

    Strategies evaluated:
    - greedy baseline (no selection)
    - ungated majority vote
    - gated majority vote (trust greedy if score >= tau)
    - ungated confidence-weighted vote (if scores available)
    - gated confidence-weighted vote
    - random (averaged over 100 trials)
    """
    results = {}

    # Build holdout problem data
    holdout_greedy_correct = new_correct[holdout_idx].astype(bool)
    n_greedy_correct = int(holdout_greedy_correct.sum())
    n_greedy_wrong = int((~holdout_greedy_correct).sum())

    results["baseline"] = {
        "n_correct": n_greedy_correct,
        "n_wrong": n_greedy_wrong,
    }

    # For each strategy, track: n_correct_after, W->R (wrong to right), R->W (right to wrong)
    strategies_to_run = ["ungated_mv", "gated_mv"]
    if holdout_scores is not None:
        strategies_to_run.extend(["ungated_weighted", "gated_weighted"])

    for strategy_name in strategies_to_run:
        correct_after = 0
        w_to_r = 0
        r_to_w = 0
        gated_correct = 0
        gated_count = 0

        for local_i, gi in enumerate(holdout_idx):
            greedy_correct = bool(new_correct[gi])
            gt = ground_truths[gi]

            # Check if this problem has temperature completions
            if str(gi) not in holdout_temp:
                # No temp gens — use greedy result
                if greedy_correct:
                    correct_after += 1
                continue

            data = holdout_temp[str(gi)]
            completions = data["completions"]

            # Gated strategies: trust greedy if score >= tau
            is_gated = strategy_name.startswith("gated")
            if is_gated and holdout_scores is not None:
                score = holdout_scores[local_i]
                if score >= tau:
                    # Trust greedy
                    if greedy_correct:
                        correct_after += 1
                        gated_correct += 1
                    gated_count += 1
                    continue

            # Apply selection strategy
            if "mv" in strategy_name:
                _, selected_correct = select_majority_vote_v2(completions, gt)
            elif "weighted" in strategy_name:
                # Use per-completion topo scores if available from step1
                scores_for_completions = _get_completion_scores(gi, len(completions))
                _, selected_correct = select_confidence_weighted_vote_v2(
                    scores_for_completions, completions, gt
                )
            else:
                _, selected_correct = select_majority_vote_v2(completions, gt)

            if selected_correct:
                correct_after += 1
                if not greedy_correct:
                    w_to_r += 1
            else:
                if greedy_correct:
                    r_to_w += 1

        net_gain = correct_after - n_greedy_correct
        results[strategy_name] = {
            "n_correct": correct_after,
            "net_gain": net_gain,
            "W_to_R": w_to_r,
            "R_to_W": r_to_w,
        }
        if is_gated:
            results[strategy_name]["gated_count"] = gated_count
            results[strategy_name]["tau"] = tau

    # Random baseline (100 trials)
    rng = np.random.default_rng(42)
    random_corrects = []
    for trial in range(100):
        trial_correct = 0
        for local_i, gi in enumerate(holdout_idx):
            greedy_correct = bool(new_correct[gi])
            gt = ground_truths[gi]
            if str(gi) not in holdout_temp:
                if greedy_correct:
                    trial_correct += 1
                continue
            data = holdout_temp[str(gi)]
            completions = data["completions"]
            _, sel_correct = select_random_v2(
                np.ones(len(completions)), completions, gt, rng
            )
            if sel_correct:
                trial_correct += 1
        random_corrects.append(trial_correct)

    results["random"] = {
        "mean_correct": float(np.mean(random_corrects)),
        "std_correct": float(np.std(random_corrects)),
        "net_gain": float(np.mean(random_corrects)) - n_greedy_correct,
    }

    return results


def evaluate_adaptive_sampling(
    new_correct: np.ndarray,
    holdout_idx: np.ndarray,
    holdout_temp: dict,
    ground_truths: list[str],
    holdout_scores: np.ndarray,
) -> dict:
    """Experiment 8: Adaptive sampling with corrected labels.

    Three-tier routing: easy (trust greedy), medium (N=8 MV), hard (N=32 MV).
    """
    configs = [
        {"tau_e": 0.3, "tau_h": 0.05, "n_med": 8},
        {"tau_e": 0.3, "tau_h": 0.05, "n_med": 16},
        {"tau_e": 0.5, "tau_h": 0.1, "n_med": 8},
    ]

    results = []
    for cfg in configs:
        tau_e = cfg["tau_e"]
        tau_h = cfg["tau_h"]
        n_med = cfg["n_med"]

        correct_after = 0
        total_samples = 0
        n_easy = 0
        n_med_count = 0
        n_hard = 0
        w_to_r = 0
        r_to_w = 0

        for local_i, gi in enumerate(holdout_idx):
            greedy_correct = bool(new_correct[gi])
            gt = ground_truths[gi]
            score = holdout_scores[local_i]

            if score >= tau_e:
                # Easy: trust greedy, 0 samples
                if greedy_correct:
                    correct_after += 1
                n_easy += 1
            elif str(gi) not in holdout_temp:
                # No temp gens available
                if greedy_correct:
                    correct_after += 1
                n_hard += 1
                total_samples += 32
            else:
                completions = holdout_temp[str(gi)]["completions"]
                if score < tau_h:
                    # Hard: full 32 samples
                    n_hard += 1
                    total_samples += 32
                    _, sel_correct = select_majority_vote_v2(completions, gt)
                else:
                    # Medium: n_med samples
                    n_med_count += 1
                    total_samples += n_med
                    subset = completions[:n_med]
                    _, sel_correct = select_majority_vote_v2(subset, gt)

                if sel_correct:
                    correct_after += 1
                    if not greedy_correct:
                        w_to_r += 1
                else:
                    if greedy_correct:
                        r_to_w += 1

        uniform_samples = len(holdout_idx) * 32
        savings = 1 - total_samples / uniform_samples if uniform_samples > 0 else 0

        results.append({
            **cfg,
            "n_correct": correct_after,
            "net_gain": correct_after - int(new_correct[holdout_idx].sum()),
            "W_to_R": w_to_r,
            "R_to_W": r_to_w,
            "total_samples": total_samples,
            "uniform_samples": uniform_samples,
            "savings_pct": savings * 100,
            "n_easy": n_easy,
            "n_medium": n_med_count,
            "n_hard": n_hard,
        })

    return results


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 2 STEP 2: SELECTION STRATEGIES WITH CORRECTED LABELS")
    logger.info("=" * 70)

    # Load data
    new_correct = np.load(BASELINE_CORRECT_V2)
    old_correct = np.load(OLD_BASELINE_CORRECT)
    train_idx, holdout_idx = get_train_holdout_indices()

    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    with open(HOLDOUT_TEMP_V2) as f:
        holdout_temp = json.load(f)

    # Load retrained prompt-level model scores
    with open(PHASE1_DIR / "per_problem_scores_v2.json") as f:
        scores_data = json.load(f)
    holdout_scores = np.array(scores_data["topo_scores"])

    # Try to load per-completion features from step1 (GPU)
    has_real_scores = load_and_score_completions(train_idx, holdout_idx)

    logger.info("  Holdout: %d problems, %d correct (was %d)",
                len(holdout_idx), new_correct[holdout_idx].sum(),
                old_correct[holdout_idx].sum())
    logger.info("  Holdout temp gens: %d problems", len(holdout_temp))

    # ---- Holdout selection evaluation ----
    logger.info("\n--- Holdout Selection Evaluation ---")

    selection_results = evaluate_holdout_selection(
        new_correct, holdout_idx, holdout_temp, ground_truths,
        holdout_scores=holdout_scores, tau=0.3,
    )

    logger.info("  Baseline: %d/100 correct", selection_results["baseline"]["n_correct"])
    for name in ["random", "ungated_mv", "gated_mv", "ungated_weighted", "gated_weighted"]:
        if name not in selection_results:
            continue
        r = selection_results[name]
        if name == "random":
            logger.info("  %s: %.1f +/- %.1f correct (net %+.1f)",
                        name.ljust(20), r["mean_correct"], r["std_correct"], r["net_gain"])
        else:
            logger.info("  %s: %d correct (net %+d, W->R=%d, R->W=%d)",
                        name.ljust(20), r["n_correct"], r["net_gain"],
                        r["W_to_R"], r["R_to_W"])

    # Compare with old results
    logger.info("\n  Old results (wrong labels):")
    logger.info("    MV: +8 net (11 W->R, 3 R->W)")
    logger.info("    Gated MV (tau=0.3): +10 net (0 R->W)")
    logger.info("    Weighted vote: +14 (0 R->W)")

    with open(OUTPUT_DIR / "holdout_results_v2.json", "w") as f:
        json.dump(selection_results, f, indent=2)

    # ---- Experiment 1: MV Ablation ----
    logger.info("\n--- Experiment 1: MV Ablation with Corrected Labels ---")

    # Sweep tau values for gated MV
    tau_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ablation_results = {"tau_sweep": []}

    for tau in tau_sweep:
        res = evaluate_holdout_selection(
            new_correct, holdout_idx, holdout_temp, ground_truths,
            holdout_scores=holdout_scores, tau=tau,
        )
        gated = res.get("gated_mv", {})
        ablation_results["tau_sweep"].append({
            "tau": tau,
            "n_correct": gated.get("n_correct", 0),
            "net_gain": gated.get("net_gain", 0),
            "W_to_R": gated.get("W_to_R", 0),
            "R_to_W": gated.get("R_to_W", 0),
            "gated_count": gated.get("gated_count", 0),
        })
        logger.info(
            "  tau=%.1f: %d correct (net %+d, W->R=%d, R->W=%d, gated=%d)",
            tau,
            gated.get("n_correct", 0),
            gated.get("net_gain", 0),
            gated.get("W_to_R", 0),
            gated.get("R_to_W", 0),
            gated.get("gated_count", 0),
        )

    with open(OUTPUT_DIR / "experiment1_v2.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # ---- Experiment 8: Adaptive Sampling ----
    logger.info("\n--- Experiment 8: Adaptive Sampling with Corrected Labels ---")

    adaptive_results = evaluate_adaptive_sampling(
        new_correct, holdout_idx, holdout_temp, ground_truths, holdout_scores,
    )

    for r in adaptive_results:
        logger.info(
            "  tau_e=%.1f, tau_h=%.2f, N_med=%d: %d correct (net %+d), "
            "W->R=%d, R->W=%d, savings=%.1f%%",
            r["tau_e"], r["tau_h"], r["n_med"],
            r["n_correct"], r["net_gain"],
            r["W_to_R"], r["R_to_W"], r["savings_pct"],
        )

    with open(OUTPUT_DIR / "experiment8_v2.json", "w") as f:
        json.dump(adaptive_results, f, indent=2)

    # ---- Experiment 5: Per-Completion Verifier Analysis ----
    if has_real_scores:
        logger.info("\n--- Experiment 5: Per-Completion Verifier Analysis ---")
        exp5 = run_experiment5(train_idx, holdout_idx)
        if exp5:
            logger.info("  Global AUROC (CV): %s", f"{exp5['global_auroc_cv']:.4f}" if exp5['global_auroc_cv'] else "N/A")
            logger.info("  Within-problem AUROC: %s (n=%d)",
                        f"{exp5['within_problem_auroc_mean']:.4f}" if exp5['within_problem_auroc_mean'] else "N/A",
                        exp5['within_problem_auroc_n'])
            logger.info("  Holdout AUROC: %s", f"{exp5['holdout_auroc']:.4f}" if exp5['holdout_auroc'] else "N/A")
            with open(OUTPUT_DIR / "experiment5_v2.json", "w") as f:
                json.dump(exp5, f, indent=2)
    else:
        logger.info("\n  Experiment 5 deferred — no per-completion features from step1.")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 STEP 2 COMPLETE")
    logger.info("=" * 70)
    if has_real_scores:
        logger.info("Per-completion features loaded — weighted vote uses REAL topo scores.")
    else:
        logger.info("NOTE: Weighted vote uses uniform scores (= MV) as placeholder.")
        logger.info("Real weighted vote needs per-completion features from RunPod.")
    logger.info("Elapsed: %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
