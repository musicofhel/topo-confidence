#!/usr/bin/env python3
"""Experiment 5: Per-Completion Topo Features as Verifier.

Can topo-confidence distinguish correct from incorrect completions WITHIN
a problem?  If yes, it's a verifier (like PRM).  If no, it's a routing
mechanism that captures problem-level difficulty.

Metrics:
  1. Global per-completion AUROC (train-CV and holdout)
  2. Within-problem AUROC (per mixed problem, then averaged)
  3. Score distribution analysis (Cohen's d, KS)
  4. Variance decomposition (between vs within problem, eta-squared)
  5. Rank analysis (MRR for correct completions)
  6. BoN@K + MV hybrid
  7. All 4 strategies on holdout
  8. Prompt-level vs completion-level correlation
  9. Combined gate + verifier
  10. Feature importance comparison

Zero GPU required -- operates on precomputed features and scores.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BASELINE_CORRECT_PATH,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    TEMP_GEN_PATH,
    check_correct,
    extract_answer,
    fit_completion_scorer,
    get_abc_column_indices,
    get_train_holdout_indices,
    load_math500,
    get_prompts_and_ground_truths,
    normalize_answer,
    score_completions,
    select_confidence_weighted_vote,
    select_majority_vote,
    select_max_confidence,
    select_random,
    logger,
)

OUTPUT_DIR = Path(__file__).parent / "experiment5_per_completion"
N_BOOTSTRAP = 500
N_MRR_SHUFFLES = 10_000
EXP10_SCORES_PATH = (
    Path(__file__).parent / "experiment10_calibration" / "per_problem_scores.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bootstrap_auroc(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 42
) -> tuple[float, float, float]:
    """AUROC with bootstrap 95% CI."""
    auroc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    boot_aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        boot_aurocs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    ci_lo = float(np.percentile(boot_aurocs, 2.5))
    ci_hi = float(np.percentile(boot_aurocs, 97.5))
    return auroc, ci_lo, ci_hi


def majority_vote_from_completions(
    completions: list[str], ground_truth: str
) -> bool:
    """Standard MV on a list of completion strings."""
    answers = [normalize_answer(extract_answer(c)) for c in completions]
    counter = Counter(answers)
    best = counter.most_common(1)[0][0]
    for c in completions:
        if normalize_answer(extract_answer(c)) == best:
            return check_correct(extract_answer(c), ground_truth)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("EXPERIMENT 5: PER-COMPLETION VERIFIER ANALYSIS")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load all data
    # ------------------------------------------------------------------
    logger.info("\n--- Step 1: Loading data ---")

    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()
    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    # Train completion features
    cf_data = np.load(PHASE1_DIR / "completion_features.npz")
    comp_features = cf_data["features"]       # (11328, 78)
    comp_correct = cf_data["correct"]         # (11328,) bool
    comp_map = cf_data["completion_map"]      # (11328, 2) [global_idx, comp_idx]

    with open(PHASE1_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    # Build idx_to_rows: global_idx -> list of row indices
    idx_to_rows: dict[int, list[int]] = {}
    for row, (gi, ci) in enumerate(comp_map):
        gi = int(gi)
        idx_to_rows.setdefault(gi, []).append(row)

    # Holdout completion scores (already scored by full-train model)
    with open(PHASE3_DIR / "holdout_completion_scores.json") as f:
        holdout_comp_scores = json.load(f)

    # Temperature generations (text completions)
    with open(TEMP_GEN_PATH) as f:
        train_temp_gen = json.load(f)
    with open(PHASE3_DIR / "holdout_temperature_generations.json") as f:
        holdout_temp_gen = json.load(f)

    # CV fold assignments
    with open(PHASE2_DIR / "fold_assignments_train400.json") as f:
        folds = json.load(f)

    # Prompt-level scores (holdout, from experiment 10)
    with open(EXP10_SCORES_PATH) as f:
        exp10_data = json.load(f)
    exp10_holdout_indices = exp10_data["holdout_indices"]
    exp10_uncalibrated = exp10_data["uncalibrated"]

    # SSS ordering for prompt-level features
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(
        sss.split(np.zeros(len(baseline_correct)), baseline_correct.astype(int))
    )
    train_sort_perm = np.argsort(train_idx_sss)
    hold_sort_perm = np.argsort(hold_idx_sss)
    train_features_78 = np.load(FEATURES_TRAIN_PATH)[train_sort_perm]
    holdout_features_78 = np.load(FEATURES_HOLDOUT_PATH)[hold_sort_perm]

    logger.info("  Completion features: %s (%d correct, %.1f%%)",
                comp_features.shape, comp_correct.sum(),
                100 * comp_correct.mean())
    logger.info("  Holdout completion scores: %d problems", len(holdout_comp_scores))
    logger.info("  Train idx: %d, Holdout idx: %d", len(train_idx), len(holdout_idx))

    # ------------------------------------------------------------------
    # Step 1b: Reconstruct out-of-fold per-completion scores (train)
    # ------------------------------------------------------------------
    logger.info("\n--- Step 1b: CV score reconstruction (5-fold) ---")

    cv_scores_arr = np.full(len(comp_correct), np.nan)
    cv_problem_ids = np.full(len(comp_correct), -1, dtype=int)

    # Pre-fill problem_ids (these don't depend on fold)
    for row, (gi, ci) in enumerate(comp_map):
        cv_problem_ids[row] = int(gi)

    for fold in folds:
        fold_idx = fold["fold"]
        fold_train_global = set(fold["train_global"])
        fold_test_global = set(fold["test_global"])

        # Collect fold-train rows
        train_rows = []
        for gi in fold["train_global"]:
            if gi in idx_to_rows:
                train_rows.extend(idx_to_rows[gi])

        if not train_rows:
            logger.warning("  Fold %d: no train completions", fold_idx)
            continue

        # Fit completion scorer on fold-train
        fold_train_feat = comp_features[train_rows]
        fold_train_corr = comp_correct[train_rows]
        scaler, clf, abc_indices = fit_completion_scorer(
            fold_train_feat, fold_train_corr, feature_names
        )

        # Score fold-test completions
        test_rows = []
        for gi in fold["test_global"]:
            if gi in idx_to_rows:
                test_rows.extend(idx_to_rows[gi])

        if test_rows:
            fold_test_feat = comp_features[test_rows]
            fold_test_scores = score_completions(
                scaler, clf, abc_indices, fold_test_feat
            )
            cv_scores_arr[test_rows] = fold_test_scores

        logger.info("  Fold %d: train=%d completions, test=%d completions",
                     fold_idx, len(train_rows), len(test_rows))

    # Remove any NaN entries (should be 0 if all problems covered)
    valid_mask = ~np.isnan(cv_scores_arr)
    n_nan = (~valid_mask).sum()
    if n_nan > 0:
        logger.warning("  %d completions without CV scores (removing)", n_nan)
    cv_scores = cv_scores_arr[valid_mask]
    cv_correct = comp_correct[valid_mask]
    cv_pids = cv_problem_ids[valid_mask]

    logger.info("  CV scores: %d completions, %d correct",
                len(cv_scores), cv_correct.sum())

    # Flatten holdout scores
    holdout_all_scores = []
    holdout_all_correct = []
    holdout_all_pids = []
    for gi_str, data in holdout_comp_scores.items():
        holdout_all_scores.extend(data["scores"])
        holdout_all_correct.extend(data["correct"])
        holdout_all_pids.extend([int(gi_str)] * 32)
    holdout_all_scores = np.array(holdout_all_scores)
    holdout_all_correct = np.array(holdout_all_correct, dtype=bool)
    holdout_all_pids = np.array(holdout_all_pids)

    logger.info("  Holdout scores: %d completions, %d correct",
                len(holdout_all_scores), holdout_all_correct.sum())

    # ------------------------------------------------------------------
    # Step 2: Global per-completion AUROC
    # ------------------------------------------------------------------
    logger.info("\n--- Step 2: Global per-completion AUROC ---")

    train_auroc, train_auroc_lo, train_auroc_hi = bootstrap_auroc(
        cv_correct.astype(int), cv_scores
    )
    holdout_auroc, holdout_auroc_lo, holdout_auroc_hi = bootstrap_auroc(
        holdout_all_correct.astype(int), holdout_all_scores
    )

    logger.info("  Train-CV:  AUROC = %.3f [%.3f - %.3f]  (N=%d, %d correct)",
                train_auroc, train_auroc_lo, train_auroc_hi,
                len(cv_scores), cv_correct.sum())
    logger.info("  Holdout:   AUROC = %.3f [%.3f - %.3f]  (N=%d, %d correct)",
                holdout_auroc, holdout_auroc_lo, holdout_auroc_hi,
                len(holdout_all_scores), holdout_all_correct.sum())

    global_auroc = {
        "train_cv": {
            "auroc": round(train_auroc, 4),
            "ci_95": [round(train_auroc_lo, 4), round(train_auroc_hi, 4)],
            "n_completions": int(len(cv_scores)),
            "n_correct": int(cv_correct.sum()),
        },
        "holdout": {
            "auroc": round(holdout_auroc, 4),
            "ci_95": [round(holdout_auroc_lo, 4), round(holdout_auroc_hi, 4)],
            "n_completions": int(len(holdout_all_scores)),
            "n_correct": int(holdout_all_correct.sum()),
        },
        "prompt_level_auroc_reference": 0.935,
    }

    # ------------------------------------------------------------------
    # Step 3: Within-problem AUROC
    # ------------------------------------------------------------------
    logger.info("\n--- Step 3: Within-problem AUROC ---")

    def compute_within_aurocs(
        scores: np.ndarray, correct: np.ndarray, pids: np.ndarray
    ) -> dict:
        unique_pids = np.unique(pids)
        within_aurocs = []
        per_problem = {}
        for gi in unique_pids:
            mask = pids == gi
            y = correct[mask].astype(int)
            s = scores[mask]
            n_correct = int(y.sum())
            n_total = int(len(y))
            if n_correct == 0 or n_correct == n_total:
                per_problem[int(gi)] = {
                    "n_correct": n_correct, "n_total": n_total,
                    "within_auroc": None, "reason": "degenerate",
                }
                continue
            auc = float(roc_auc_score(y, s))
            within_aurocs.append(auc)
            per_problem[int(gi)] = {
                "n_correct": n_correct, "n_total": n_total,
                "within_auroc": round(auc, 4),
            }

        arr = np.array(within_aurocs) if within_aurocs else np.array([])
        return {
            "mean": round(float(arr.mean()), 4) if len(arr) else None,
            "median": round(float(np.median(arr)), 4) if len(arr) else None,
            "std": round(float(arr.std()), 4) if len(arr) else None,
            "n_eligible": len(within_aurocs),
            "n_total_problems": len(unique_pids),
            "fraction_above_050": round(float((arr > 0.5).mean()), 4) if len(arr) else None,
            "fraction_above_060": round(float((arr > 0.6).mean()), 4) if len(arr) else None,
            "quartiles": {
                "q25": round(float(np.percentile(arr, 25)), 4),
                "q50": round(float(np.percentile(arr, 50)), 4),
                "q75": round(float(np.percentile(arr, 75)), 4),
            } if len(arr) else None,
            "per_problem": per_problem,
        }

    within_train = compute_within_aurocs(cv_scores, cv_correct, cv_pids)
    within_holdout = compute_within_aurocs(
        holdout_all_scores, holdout_all_correct, holdout_all_pids
    )

    logger.info("  Train-CV:  mean=%.3f, median=%.3f (N=%d mixed problems, %.0f%% > 0.50)",
                within_train["mean"] or 0, within_train["median"] or 0,
                within_train["n_eligible"],
                100 * (within_train["fraction_above_050"] or 0))
    logger.info("  Holdout:   mean=%.3f, median=%.3f (N=%d mixed problems, %.0f%% > 0.50)",
                within_holdout["mean"] or 0, within_holdout["median"] or 0,
                within_holdout["n_eligible"],
                100 * (within_holdout["fraction_above_050"] or 0))

    # ------------------------------------------------------------------
    # Step 4: Score distribution analysis
    # ------------------------------------------------------------------
    logger.info("\n--- Step 4: Score distributions ---")

    def score_distribution(scores: np.ndarray, correct: np.ndarray) -> dict:
        correct_s = scores[correct.astype(bool)]
        incorrect_s = scores[~correct.astype(bool)]
        mean_c = float(correct_s.mean())
        mean_i = float(incorrect_s.mean())
        std_c = float(correct_s.std())
        std_i = float(incorrect_s.std())
        pooled_std = np.sqrt(
            ((len(correct_s) - 1) * std_c**2 + (len(incorrect_s) - 1) * std_i**2)
            / (len(correct_s) + len(incorrect_s) - 2)
        )
        cohens_d = (mean_c - mean_i) / pooled_std if pooled_std > 0 else 0.0
        ks_stat, ks_p = ks_2samp(correct_s, incorrect_s)
        return {
            "correct_mean": round(mean_c, 4),
            "correct_std": round(std_c, 4),
            "correct_n": int(len(correct_s)),
            "incorrect_mean": round(mean_i, 4),
            "incorrect_std": round(std_i, 4),
            "incorrect_n": int(len(incorrect_s)),
            "cohens_d": round(cohens_d, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 6),
        }

    dist_train = score_distribution(cv_scores, cv_correct)
    dist_holdout = score_distribution(holdout_all_scores, holdout_all_correct)

    logger.info("  Train-CV:  Cohen's d = %.3f (correct: %.3f +/- %.3f, incorrect: %.3f +/- %.3f)",
                dist_train["cohens_d"], dist_train["correct_mean"],
                dist_train["correct_std"], dist_train["incorrect_mean"],
                dist_train["incorrect_std"])
    logger.info("  Holdout:   Cohen's d = %.3f (correct: %.3f +/- %.3f, incorrect: %.3f +/- %.3f)",
                dist_holdout["cohens_d"], dist_holdout["correct_mean"],
                dist_holdout["correct_std"], dist_holdout["incorrect_mean"],
                dist_holdout["incorrect_std"])

    # ------------------------------------------------------------------
    # Step 5: Variance decomposition (eta-squared)
    # ------------------------------------------------------------------
    logger.info("\n--- Step 5: Variance decomposition ---")

    def variance_decomposition(scores: np.ndarray, pids: np.ndarray) -> dict:
        grand_mean = scores.mean()
        unique_pids = np.unique(pids)
        ssb = 0.0
        ssw = 0.0
        for gi in unique_pids:
            mask = pids == gi
            group = scores[mask]
            group_mean = group.mean()
            n_g = len(group)
            ssb += n_g * (group_mean - grand_mean) ** 2
            ssw += ((group - group_mean) ** 2).sum()
        sst = ssb + ssw
        eta_sq = ssb / sst if sst > 0 else 0.0
        return {
            "eta_squared": round(eta_sq, 4),
            "between_pct": round(100 * eta_sq, 1),
            "within_pct": round(100 * (1 - eta_sq), 1),
            "ssb": round(ssb, 4),
            "ssw": round(ssw, 4),
            "n_groups": int(len(unique_pids)),
        }

    var_train = variance_decomposition(cv_scores, cv_pids)
    var_holdout = variance_decomposition(holdout_all_scores, holdout_all_pids)

    logger.info("  Train-CV:  eta^2 = %.3f (%.1f%% between-problem, %.1f%% within)",
                var_train["eta_squared"], var_train["between_pct"],
                var_train["within_pct"])
    logger.info("  Holdout:   eta^2 = %.3f (%.1f%% between-problem, %.1f%% within)",
                var_holdout["eta_squared"], var_holdout["between_pct"],
                var_holdout["within_pct"])

    # ------------------------------------------------------------------
    # Step 6: Rank analysis
    # ------------------------------------------------------------------
    logger.info("\n--- Step 6: Rank analysis ---")

    def rank_analysis(
        scores_dict: dict, rng_seed: int = 42
    ) -> dict:
        """Compute MRR and rank stats for problems with >= 1 correct completion."""
        mrr_list = []
        rank_list = []
        top_k_hits = {1: 0, 3: 0, 5: 0, 10: 0}
        n_eligible = 0

        # Random baseline MRR via Monte Carlo
        random_mrr_list = []
        rng = np.random.default_rng(rng_seed)

        for gi_str, data in scores_dict.items():
            s = np.array(data["scores"])
            c = np.array(data["correct"])
            n_correct = int(c.sum())
            if n_correct == 0:
                continue
            n_eligible += 1

            # Rank by score descending
            ranking = np.argsort(-s)
            for rank_pos, idx in enumerate(ranking):
                if c[idx]:
                    rank_list.append(rank_pos + 1)
                    mrr_list.append(1.0 / (rank_pos + 1))
                    for k in top_k_hits:
                        if rank_pos < k:
                            top_k_hits[k] += 1
                    break

            # Random baseline MRR
            for _ in range(N_MRR_SHUFFLES // max(n_eligible, 1)):
                perm = rng.permutation(len(s))
                for rank_pos, idx in enumerate(perm):
                    if c[idx]:
                        random_mrr_list.append(1.0 / (rank_pos + 1))
                        break

        if not mrr_list:
            return {"mrr": None, "n_eligible": 0}

        return {
            "mrr": round(float(np.mean(mrr_list)), 4),
            "mean_rank": round(float(np.mean(rank_list)), 2),
            "median_rank": round(float(np.median(rank_list)), 1),
            "n_eligible": n_eligible,
            "top1_pct": round(100 * top_k_hits[1] / n_eligible, 1),
            "top3_pct": round(100 * top_k_hits[3] / n_eligible, 1),
            "top5_pct": round(100 * top_k_hits[5] / n_eligible, 1),
            "top10_pct": round(100 * top_k_hits[10] / n_eligible, 1),
            "random_baseline_mrr": round(float(np.mean(random_mrr_list)), 4)
                if random_mrr_list else None,
        }

    # Build train scores dict from CV scores
    train_scores_dict: dict[str, dict] = {}
    for gi in np.unique(cv_pids):
        mask = cv_pids == gi
        train_scores_dict[str(int(gi))] = {
            "scores": cv_scores[mask].tolist(),
            "correct": cv_correct[mask].tolist(),
        }

    rank_train = rank_analysis(train_scores_dict, rng_seed=42)
    rank_holdout = rank_analysis(holdout_comp_scores, rng_seed=99)

    logger.info("  Train-CV:  MRR=%.3f (random=%.3f), mean rank=%.1f, "
                "top-1=%.0f%%, top-5=%.0f%%, top-10=%.0f%% (N=%d)",
                rank_train["mrr"], rank_train["random_baseline_mrr"],
                rank_train["mean_rank"], rank_train["top1_pct"],
                rank_train["top5_pct"], rank_train["top10_pct"],
                rank_train["n_eligible"])
    logger.info("  Holdout:   MRR=%.3f (random=%.3f), mean rank=%.1f, "
                "top-1=%.0f%%, top-5=%.0f%%, top-10=%.0f%% (N=%d)",
                rank_holdout["mrr"], rank_holdout["random_baseline_mrr"],
                rank_holdout["mean_rank"], rank_holdout["top1_pct"],
                rank_holdout["top5_pct"], rank_holdout["top10_pct"],
                rank_holdout["n_eligible"])

    # ------------------------------------------------------------------
    # Step 7: BoN@K hybrid (top-K by score, then MV among K)
    # ------------------------------------------------------------------
    logger.info("\n--- Step 7: BoN@K hybrid ---")

    def bon_k_evaluation(
        holdout_idx_arr: np.ndarray,
        baseline_correct_arr: np.ndarray,
        holdout_comp_scores_dict: dict,
        holdout_gen: dict,
        ground_truths_list: list[str],
        k_values: list[int],
    ) -> dict:
        results = {}
        for k in k_values:
            greedy_list = []
            selected_list = []
            for gi in holdout_idx_arr:
                gc = bool(baseline_correct_arr[gi])
                greedy_list.append(gc)
                if gc:
                    selected_list.append(True)
                    continue
                gi_str = str(gi)
                if gi_str not in holdout_comp_scores_dict:
                    selected_list.append(False)
                    continue
                scores = np.array(holdout_comp_scores_dict[gi_str]["scores"])
                completions = holdout_gen[gi_str]["completions"]
                gt = ground_truths_list[gi]
                if k >= 32:
                    # Full MV
                    ok = majority_vote_from_completions(completions, gt)
                else:
                    top_k_idx = np.argsort(-scores)[:k]
                    top_k_comps = [completions[i] for i in top_k_idx]
                    if k == 1:
                        ok = check_correct(extract_answer(top_k_comps[0]), gt)
                    else:
                        ok = majority_vote_from_completions(top_k_comps, gt)
                selected_list.append(ok)

            g = np.array(greedy_list, dtype=bool)
            s = np.array(selected_list, dtype=bool)
            w2r = int((~g & s).sum())
            r2w = int((g & ~s).sum())
            results[f"K{k}"] = {
                "k": k,
                "n_correct": int(s.sum()),
                "net_gain": w2r - r2w,
                "w2r": w2r,
                "r2w": r2w,
            }
        return results

    bon_k_holdout = bon_k_evaluation(
        holdout_idx, baseline_correct, holdout_comp_scores,
        holdout_temp_gen, ground_truths, [1, 3, 5, 10, 32]
    )

    # Also do BoN@K with RANDOM K from 32 (no score guidance) for comparison
    rng_bonk = np.random.default_rng(42)
    random_k_holdout = {}
    for k in [1, 3, 5, 10]:
        n_trials = 200
        trial_correct = np.zeros(n_trials)
        for trial in range(n_trials):
            correct_count = 0
            for gi in holdout_idx:
                if baseline_correct[gi]:
                    correct_count += 1
                    continue
                gi_str = str(gi)
                if gi_str not in holdout_temp_gen:
                    continue
                completions = holdout_temp_gen[gi_str]["completions"]
                gt = ground_truths[gi]
                rand_idx = rng_bonk.choice(32, size=k, replace=False)
                rand_comps = [completions[i] for i in rand_idx]
                if k == 1:
                    ok = check_correct(extract_answer(rand_comps[0]), gt)
                else:
                    ok = majority_vote_from_completions(rand_comps, gt)
                correct_count += int(ok)
            trial_correct[trial] = correct_count
        random_k_holdout[f"K{k}"] = {
            "k": k,
            "mean_correct": round(float(trial_correct.mean()), 2),
            "std": round(float(trial_correct.std()), 2),
        }

    for k_str, res in bon_k_holdout.items():
        random_ref = random_k_holdout.get(k_str, {}).get("mean_correct", "N/A")
        logger.info("  BoN@%s (holdout): %d correct, net +%d  (random@%s: %s)",
                     k_str, res["n_correct"], res["net_gain"],
                     k_str, random_ref)

    # ------------------------------------------------------------------
    # Step 8: All 4 strategies on holdout
    # ------------------------------------------------------------------
    logger.info("\n--- Step 8: All strategies on holdout ---")

    all_strategies_holdout = {}
    rng_strat = np.random.default_rng(42)
    n_random_trials = 200

    for strategy in ["random", "max_confidence", "majority_vote",
                     "confidence_weighted_vote"]:
        if strategy == "random":
            # Average over trials
            trial_results = []
            for _ in range(n_random_trials):
                greedy_list = []
                selected_list = []
                for gi in holdout_idx:
                    gc = bool(baseline_correct[gi])
                    greedy_list.append(gc)
                    if gc:
                        selected_list.append(True)
                        continue
                    gi_str = str(gi)
                    if gi_str not in holdout_comp_scores:
                        selected_list.append(False)
                        continue
                    scores_arr = np.array(holdout_comp_scores[gi_str]["scores"])
                    comps = holdout_temp_gen[gi_str]["completions"]
                    gt = ground_truths[gi]
                    _, ok = select_random(scores_arr, comps, gt, rng_strat)
                    selected_list.append(ok)
                g = np.array(greedy_list, dtype=bool)
                s = np.array(selected_list, dtype=bool)
                trial_results.append(int(s.sum()))
            mean_correct = float(np.mean(trial_results))
            all_strategies_holdout[strategy] = {
                "n_correct": round(mean_correct, 1),
                "net_gain": round(mean_correct - 11, 1),
                "note": f"averaged over {n_random_trials} trials",
            }
        else:
            greedy_list = []
            selected_list = []
            for gi in holdout_idx:
                gc = bool(baseline_correct[gi])
                greedy_list.append(gc)
                if gc:
                    selected_list.append(True)
                    continue
                gi_str = str(gi)
                if gi_str not in holdout_comp_scores:
                    selected_list.append(False)
                    continue
                scores_arr = np.array(holdout_comp_scores[gi_str]["scores"])
                comps = holdout_temp_gen[gi_str]["completions"]
                gt = ground_truths[gi]
                if strategy == "max_confidence":
                    _, ok = select_max_confidence(scores_arr, comps, gt)
                elif strategy == "majority_vote":
                    _, ok = select_majority_vote(comps, gt)
                elif strategy == "confidence_weighted_vote":
                    _, ok = select_confidence_weighted_vote(scores_arr, comps, gt)
                else:
                    ok = False
                selected_list.append(ok)
            g = np.array(greedy_list, dtype=bool)
            s = np.array(selected_list, dtype=bool)
            w2r = int((~g & s).sum())
            r2w = int((g & ~s).sum())
            all_strategies_holdout[strategy] = {
                "n_correct": int(s.sum()),
                "net_gain": w2r - r2w,
                "w2r": w2r,
                "r2w": r2w,
            }

        logger.info("  %-30s: %s correct, net +%s",
                     strategy,
                     all_strategies_holdout[strategy]["n_correct"],
                     all_strategies_holdout[strategy]["net_gain"])

    # ------------------------------------------------------------------
    # Step 9: Prompt-level vs completion-level correlation
    # ------------------------------------------------------------------
    logger.info("\n--- Step 9: Prompt vs completion-level correlation ---")

    # Holdout: mean per-completion score vs prompt-level topo-confidence
    holdout_prompt_scores = {}
    for i, gi in enumerate(exp10_holdout_indices):
        holdout_prompt_scores[gi] = exp10_uncalibrated[i]

    prompt_vals = []
    comp_mean_vals = []
    for gi_str, data in holdout_comp_scores.items():
        gi = int(gi_str)
        if gi in holdout_prompt_scores:
            prompt_vals.append(holdout_prompt_scores[gi])
            comp_mean_vals.append(float(np.mean(data["scores"])))

    prompt_vals = np.array(prompt_vals)
    comp_mean_vals = np.array(comp_mean_vals)

    if len(prompt_vals) > 2:
        r_pear, p_pear = pearsonr(prompt_vals, comp_mean_vals)
        r_spear, p_spear = spearmanr(prompt_vals, comp_mean_vals)
    else:
        r_pear = p_pear = r_spear = p_spear = None

    corr_holdout = {
        "pearson_r": round(float(r_pear), 4) if r_pear is not None else None,
        "pearson_p": round(float(p_pear), 6) if p_pear is not None else None,
        "spearman_r": round(float(r_spear), 4) if r_spear is not None else None,
        "spearman_p": round(float(p_spear), 6) if p_spear is not None else None,
        "n_problems": len(prompt_vals),
    }

    # Train: reconstruct prompt-level CV probs
    y_train = baseline_correct[train_idx].astype(int)
    abc_prompt = get_abc_column_indices(feature_names)
    X_train_prompt = train_features_78[:, abc_prompt]
    scaler_prompt = StandardScaler()
    X_train_scaled = scaler_prompt.fit_transform(X_train_prompt)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_prompt_probs = cross_val_predict(
        LogisticRegression(**LR_PARAMS),
        X_train_scaled, y_train, cv=skf, method="predict_proba"
    )[:, 1]

    # Build local_idx -> global_idx mapping
    train_prompt_by_global = {}
    for local_i, gi in enumerate(train_idx):
        train_prompt_by_global[int(gi)] = float(cv_prompt_probs[local_i])

    train_prompt_vals = []
    train_comp_mean_vals = []
    for gi_str, data in train_scores_dict.items():
        gi = int(gi_str)
        if gi in train_prompt_by_global:
            train_prompt_vals.append(train_prompt_by_global[gi])
            train_comp_mean_vals.append(float(np.mean(data["scores"])))

    train_prompt_vals = np.array(train_prompt_vals)
    train_comp_mean_vals = np.array(train_comp_mean_vals)

    if len(train_prompt_vals) > 2:
        r_pear_t, p_pear_t = pearsonr(train_prompt_vals, train_comp_mean_vals)
        r_spear_t, p_spear_t = spearmanr(train_prompt_vals, train_comp_mean_vals)
    else:
        r_pear_t = p_pear_t = r_spear_t = p_spear_t = None

    corr_train = {
        "pearson_r": round(float(r_pear_t), 4) if r_pear_t is not None else None,
        "pearson_p": round(float(p_pear_t), 6) if p_pear_t is not None else None,
        "spearman_r": round(float(r_spear_t), 4) if r_spear_t is not None else None,
        "spearman_p": round(float(p_spear_t), 6) if p_spear_t is not None else None,
        "n_problems": len(train_prompt_vals),
    }

    logger.info("  Holdout:  Pearson r=%.3f (p=%.4f), Spearman rho=%.3f (N=%d)",
                corr_holdout["pearson_r"] or 0, corr_holdout["pearson_p"] or 0,
                corr_holdout["spearman_r"] or 0, corr_holdout["n_problems"])
    logger.info("  Train:    Pearson r=%.3f (p=%.4f), Spearman rho=%.3f (N=%d)",
                corr_train["pearson_r"] or 0, corr_train["pearson_p"] or 0,
                corr_train["spearman_r"] or 0, corr_train["n_problems"])

    # ------------------------------------------------------------------
    # Step 10: Combined gate + verifier
    # ------------------------------------------------------------------
    logger.info("\n--- Step 10: Combined gate + verifier ---")

    combined_results = []
    for tau in [0.3, 0.5]:
        for selection in ["majority_vote", "max_confidence", "bon_k5_mv"]:
            greedy_list = []
            selected_list = []
            n_gated = 0
            for gi in holdout_idx:
                gc = bool(baseline_correct[gi])
                greedy_list.append(gc)
                # Gate: if prompt-level score >= tau, trust greedy
                prompt_score = holdout_prompt_scores.get(int(gi), 0.0)
                if prompt_score >= tau:
                    selected_list.append(gc)
                    n_gated += 1
                    continue
                gi_str = str(gi)
                if gi_str not in holdout_comp_scores:
                    selected_list.append(False)
                    continue
                scores_arr = np.array(holdout_comp_scores[gi_str]["scores"])
                comps = holdout_temp_gen[gi_str]["completions"]
                gt = ground_truths[gi]
                if selection == "majority_vote":
                    _, ok = select_majority_vote(comps, gt)
                elif selection == "max_confidence":
                    _, ok = select_max_confidence(scores_arr, comps, gt)
                elif selection == "bon_k5_mv":
                    top5 = np.argsort(-scores_arr)[:5]
                    ok = majority_vote_from_completions(
                        [comps[i] for i in top5], gt
                    )
                else:
                    ok = False
                selected_list.append(ok)

            g = np.array(greedy_list, dtype=bool)
            s = np.array(selected_list, dtype=bool)
            w2r = int((~g & s).sum())
            r2w = int((g & ~s).sum())

            entry = {
                "tau": tau,
                "selection": selection,
                "n_gated": n_gated,
                "n_correct": int(s.sum()),
                "net_gain": w2r - r2w,
                "w2r": w2r,
                "r2w": r2w,
            }
            combined_results.append(entry)
            logger.info("  tau=%.1f + %-20s: %d correct, net +%d "
                        "(%d gated, W->R=%d, R->W=%d)",
                        tau, selection, entry["n_correct"], entry["net_gain"],
                        n_gated, w2r, r2w)

    # ------------------------------------------------------------------
    # Step 11: Feature importance comparison
    # ------------------------------------------------------------------
    logger.info("\n--- Step 11: Feature importance comparison ---")

    # Prompt-level LR coefficients
    scaler_p = StandardScaler()
    X_p = train_features_78[:, abc_prompt]
    X_p_scaled = scaler_p.fit_transform(X_p)
    clf_p = LogisticRegression(**LR_PARAMS)
    clf_p.fit(X_p_scaled, y_train)
    prompt_coefs = np.abs(clf_p.coef_[0])

    # Completion-level LR coefficients (full data, for importance comparison)
    abc_comp = get_abc_column_indices(feature_names)
    scaler_c = StandardScaler()
    X_c = comp_features[:, abc_comp]
    X_c_scaled = scaler_c.fit_transform(X_c)
    clf_c = LogisticRegression(**LR_PARAMS)
    clf_c.fit(X_c_scaled, comp_correct.astype(int))
    comp_coefs = np.abs(clf_c.coef_[0])

    # Feature names for ABC subset
    abc_names = [feature_names[i] for i in abc_comp]

    # Rank correlation of coefficient magnitudes
    r_coef, p_coef = spearmanr(prompt_coefs, comp_coefs)

    # Top-10 for each
    prompt_top10_idx = np.argsort(-prompt_coefs)[:10]
    comp_top10_idx = np.argsort(-comp_coefs)[:10]

    feature_importance = {
        "spearman_rank_corr": round(float(r_coef), 4),
        "spearman_p": round(float(p_coef), 6),
        "n_abc_features": len(abc_names),
        "prompt_top10": [
            {"feature": abc_names[i], "abs_coef": round(float(prompt_coefs[i]), 4)}
            for i in prompt_top10_idx
        ],
        "completion_top10": [
            {"feature": abc_names[i], "abs_coef": round(float(comp_coefs[i]), 4)}
            for i in comp_top10_idx
        ],
    }

    logger.info("  Coefficient rank correlation: Spearman rho=%.3f (p=%.4f)",
                r_coef, p_coef)
    logger.info("  Prompt top-3: %s",
                ", ".join(abc_names[i] for i in prompt_top10_idx[:3]))
    logger.info("  Completion top-3: %s",
                ", ".join(abc_names[i] for i in comp_top10_idx[:3]))

    # ------------------------------------------------------------------
    # Step 12: Save results
    # ------------------------------------------------------------------
    logger.info("\n--- Step 12: Saving results ---")

    # Determine conclusion
    within_mean = within_holdout["mean"] or within_train["mean"] or 0.5
    eta_sq = var_train["eta_squared"]
    if within_mean < 0.55 and eta_sq > 0.5:
        conclusion = ("Topo-confidence discriminates at the PROBLEM level "
                       "(AUROC=0.935) but NOT at the COMPLETION level "
                       f"(within-problem AUROC={within_mean:.3f}). "
                       f"{var_train['between_pct']:.0f}% of score variance "
                       "is between-problem. This is a routing mechanism, "
                       "not a verifier.")
    elif within_mean >= 0.55:
        conclusion = (f"Moderate within-problem discrimination "
                       f"(AUROC={within_mean:.3f}), suggesting some "
                       "completion-level signal exists.")
    else:
        conclusion = ("Results are mixed; see detailed metrics.")

    results = {
        "experiment": "per_completion_verifier_analysis",
        "global_auroc": global_auroc,
        "within_problem_auroc": {
            "train_cv": {k: v for k, v in within_train.items() if k != "per_problem"},
            "holdout": {k: v for k, v in within_holdout.items() if k != "per_problem"},
        },
        "score_distributions": {
            "train_cv": dist_train,
            "holdout": dist_holdout,
        },
        "variance_decomposition": {
            "train_cv": var_train,
            "holdout": var_holdout,
        },
        "rank_analysis": {
            "train_cv": rank_train,
            "holdout": rank_holdout,
        },
        "bon_k_hybrid": {
            "score_guided": bon_k_holdout,
            "random_baseline": random_k_holdout,
        },
        "holdout_all_strategies": all_strategies_holdout,
        "prompt_completion_correlation": {
            "holdout": corr_holdout,
            "train_cv": corr_train,
        },
        "combined_gate_verifier": combined_results,
        "feature_importance_comparison": feature_importance,
        "existing_phase2_cv_reference": {
            "max_confidence_net_gain": 29,
            "majority_vote_net_gain": 46,
            "confidence_weighted_vote_net_gain": 45,
            "random_net_gain": 2,
            "oracle_net_gain": 153,
        },
        "paper_narrative": {
            "conclusion": conclusion,
            "key_evidence": [
                f"Global per-completion AUROC: {train_auroc:.3f} (train-CV), {holdout_auroc:.3f} (holdout)",
                f"Within-problem AUROC: {within_train['mean']:.3f} (train, N={within_train['n_eligible']}), "
                f"{within_holdout['mean']:.3f} (holdout, N={within_holdout['n_eligible']})" if within_train["mean"] else "N/A",
                f"Variance: {var_train['between_pct']:.0f}% between-problem (train), "
                f"{var_holdout['between_pct']:.0f}% (holdout)",
                f"MRR: {rank_train['mrr']:.3f} (train, random={rank_train['random_baseline_mrr']:.3f})",
                f"BoN@1 holdout: {bon_k_holdout['K1']['n_correct']} correct "
                f"vs MV@32: {bon_k_holdout['K32']['n_correct']} correct",
                f"Prompt vs completion correlation: r={corr_holdout['pearson_r']} (holdout)",
            ],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_DIR / "verifier_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  Saved: verifier_results.json")

    # Within-problem AUROC details
    within_detail = {
        "train_cv": within_train["per_problem"],
        "holdout": within_holdout["per_problem"],
    }
    with open(OUTPUT_DIR / "within_problem_aurocs.json", "w") as f:
        json.dump(within_detail, f, indent=2)
    logger.info("  Saved: within_problem_aurocs.json")

    # Per-problem score summary
    per_problem_summary = {}
    for gi_str, data in train_scores_dict.items():
        gi = int(gi_str)
        scores = np.array(data["scores"])
        correct = np.array(data["correct"])
        entry = {
            "mean_score": round(float(scores.mean()), 4),
            "std_score": round(float(scores.std()), 4),
            "n_correct": int(correct.sum()),
            "n_total": int(len(correct)),
            "split": "train",
        }
        wp = within_train["per_problem"].get(str(gi), {})
        if wp.get("within_auroc") is not None:
            entry["within_auroc"] = wp["within_auroc"]
        if gi in train_prompt_by_global:
            entry["prompt_level_score"] = round(train_prompt_by_global[gi], 4)
        per_problem_summary[gi_str] = entry

    for gi_str, data in holdout_comp_scores.items():
        gi = int(gi_str)
        scores = np.array(data["scores"])
        entry = {
            "mean_score": round(float(scores.mean()), 4),
            "std_score": round(float(np.std(scores)), 4),
            "n_correct": data["n_correct"],
            "n_total": 32,
            "split": "holdout",
        }
        wp = within_holdout["per_problem"].get(str(gi), {})
        if wp.get("within_auroc") is not None:
            entry["within_auroc"] = wp["within_auroc"]
        if gi in holdout_prompt_scores:
            entry["prompt_level_score"] = round(holdout_prompt_scores[gi], 4)
        per_problem_summary[gi_str] = entry

    with open(OUTPUT_DIR / "per_problem_score_summary.json", "w") as f:
        json.dump(per_problem_summary, f, indent=2)
    logger.info("  Saved: per_problem_score_summary.json")

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("--- Global AUROC ---")
    logger.info("  Train-CV:  %.3f [%.3f-%.3f]  (N=%d, %d correct)",
                train_auroc, train_auroc_lo, train_auroc_hi,
                len(cv_scores), cv_correct.sum())
    logger.info("  Holdout:   %.3f [%.3f-%.3f]  (N=%d, %d correct)",
                holdout_auroc, holdout_auroc_lo, holdout_auroc_hi,
                len(holdout_all_scores), holdout_all_correct.sum())
    logger.info("  (Prompt-level AUROC = 0.935 for reference)")
    logger.info("")
    logger.info("--- Within-Problem AUROC (KEY METRIC) ---")
    logger.info("  Train-CV:  mean=%.3f, median=%.3f (%d mixed problems, %.0f%% > 0.50)",
                within_train["mean"] or 0, within_train["median"] or 0,
                within_train["n_eligible"],
                100 * (within_train["fraction_above_050"] or 0))
    logger.info("  Holdout:   mean=%.3f, median=%.3f (%d mixed problems, %.0f%% > 0.50)",
                within_holdout["mean"] or 0, within_holdout["median"] or 0,
                within_holdout["n_eligible"],
                100 * (within_holdout["fraction_above_050"] or 0))
    logger.info("")
    logger.info("--- Score Distributions ---")
    logger.info("  Cohen's d (train-CV):  %.3f", dist_train["cohens_d"])
    logger.info("  Cohen's d (holdout):   %.3f", dist_holdout["cohens_d"])
    logger.info("")
    logger.info("--- Variance Decomposition ---")
    logger.info("  eta^2 (train-CV):  %.3f (%.0f%% between-problem)",
                var_train["eta_squared"], var_train["between_pct"])
    logger.info("  eta^2 (holdout):   %.3f (%.0f%% between-problem)",
                var_holdout["eta_squared"], var_holdout["between_pct"])
    logger.info("")
    logger.info("--- Rank Analysis ---")
    logger.info("  MRR (train-CV):  %.3f  (random: %.3f)",
                rank_train["mrr"], rank_train["random_baseline_mrr"])
    logger.info("  MRR (holdout):   %.3f  (random: %.3f)",
                rank_holdout["mrr"], rank_holdout["random_baseline_mrr"])
    logger.info("")
    logger.info("--- BoN@K + MV (holdout, net gain) ---")
    for k_str in ["K1", "K3", "K5", "K10", "K32"]:
        res = bon_k_holdout[k_str]
        rand = random_k_holdout.get(k_str, {}).get("mean_correct", "-")
        logger.info("  %s: +%d (%d correct)  [random: %s correct]",
                     k_str, res["net_gain"], res["n_correct"], rand)
    logger.info("")
    logger.info("--- All Strategies (holdout) ---")
    for strat, res in all_strategies_holdout.items():
        logger.info("  %-30s: %s correct, net +%s",
                     strat, res["n_correct"], res["net_gain"])
    logger.info("")
    logger.info("--- Prompt vs Completion Correlation ---")
    logger.info("  Holdout: Pearson r=%.3f, Spearman rho=%.3f (N=%d)",
                corr_holdout["pearson_r"] or 0,
                corr_holdout["spearman_r"] or 0,
                corr_holdout["n_problems"])
    logger.info("  Train:   Pearson r=%.3f, Spearman rho=%.3f (N=%d)",
                corr_train["pearson_r"] or 0,
                corr_train["spearman_r"] or 0,
                corr_train["n_problems"])
    logger.info("")
    logger.info("CONCLUSION: %s", conclusion)
    logger.info("")
    logger.info("Total time: %.1f seconds", elapsed)
    logger.info("Experiment 5 complete.")


if __name__ == "__main__":
    main()
