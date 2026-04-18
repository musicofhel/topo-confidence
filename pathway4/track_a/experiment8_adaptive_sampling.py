#!/usr/bin/env python3
"""Experiment 8: Adaptive Sampling Budget (Parcae Demo).

Partition problems by topo-confidence into tiers (easy/medium/hard) and
allocate different sampling budgets per tier. Compare accuracy and token
savings vs uniform N=32 sampling.

Cost model: greedy pass is sunk cost (always needed for topo features).
Extra cost = number of temperature samples. Savings = 1 - sum(N_i) / (100*32).

No GPU required — operates on existing holdout_temperature_generations.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    PHASE3_DIR,
    TEMP_GEN_PATH,
    check_correct,
    extract_answer,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    logger,
    normalize_answer,
)

OUTPUT_DIR = Path(__file__).parent / "experiment8_adaptive_sampling"


# ---- Helper functions ----


def majority_vote_at_n(
    completions: list[str], ground_truth: str, n: int
) -> tuple[bool, str, float, int]:
    """Majority vote on first n completions.

    Returns: (correct, answer, vote_margin, n_unique_answers)
    """
    subset = completions[:n]
    answers = [normalize_answer(extract_answer(c)) for c in subset]
    counter = Counter(answers)
    best_norm = counter.most_common(1)[0][0]
    vote_margin = counter.most_common(1)[0][1] / len(subset)
    n_unique = len(counter)
    # Find raw answer
    for c in subset:
        if normalize_answer(extract_answer(c)) == best_norm:
            raw_answer = extract_answer(c)
            return check_correct(raw_answer, ground_truth), raw_answer, vote_margin, n_unique
    raw_answer = extract_answer(subset[0])
    return check_correct(raw_answer, ground_truth), raw_answer, vote_margin, n_unique


def majority_vote_bootstrap(
    completions: list[str],
    ground_truth: str,
    n: int,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap MV accuracy by randomly drawing n from 32 completions.

    Returns: (mean_correct, ci_lo, ci_hi)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    results = []
    for _ in range(n_trials):
        idx = rng.choice(len(completions), size=n, replace=False)
        subset = [completions[i] for i in idx]
        answers = [normalize_answer(extract_answer(c)) for c in subset]
        counter = Counter(answers)
        best_norm = counter.most_common(1)[0][0]
        for c in subset:
            if normalize_answer(extract_answer(c)) == best_norm:
                results.append(check_correct(extract_answer(c), ground_truth))
                break
        else:
            results.append(check_correct(extract_answer(subset[0]), ground_truth))
    arr = np.array(results, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compute_strategy(
    holdout_idx: np.ndarray,
    holdout_gen: dict,
    ground_truths: list[str],
    baseline_correct: np.ndarray,
    assignments: dict[int, int],
) -> dict:
    """Evaluate an adaptive strategy given per-problem sample assignments.

    Args:
        assignments: maps local index (0-99) -> N (0 = trust greedy, else MV@N)

    Returns dict with correct_arr, total_samples, per_problem, flip metrics.
    """
    per_problem = []
    for local_i, gi in enumerate(holdout_idx):
        n_samples = assignments[local_i]
        greedy_ok = bool(baseline_correct[gi])
        gt = ground_truths[gi]

        if n_samples == 0:
            # Trust greedy
            correct = greedy_ok
            answer = "greedy"
            margin = None
            n_unique = None
        else:
            gi_str = str(gi)
            if gi_str not in holdout_gen:
                correct = greedy_ok
                answer = "missing"
                margin = None
                n_unique = None
            else:
                completions = holdout_gen[gi_str]["completions"]
                n_use = min(n_samples, len(completions))
                correct, answer, margin, n_unique = majority_vote_at_n(
                    completions, gt, n_use
                )

        # Classify flip
        if greedy_ok and correct:
            cat = "R->R"
        elif greedy_ok and not correct:
            cat = "R->W"
        elif not greedy_ok and correct:
            cat = "W->R"
        else:
            cat = "W->W"

        per_problem.append({
            "global_idx": int(gi),
            "local_idx": local_i,
            "n_samples": n_samples,
            "greedy_correct": greedy_ok,
            "strategy_correct": correct,
            "category": cat,
            "vote_margin": round(margin, 3) if margin is not None else None,
        })

    correct_arr = np.array([p["strategy_correct"] for p in per_problem], dtype=bool)
    greedy_arr = np.array([p["greedy_correct"] for p in per_problem], dtype=bool)
    total_samples = sum(assignments.values())

    w2r = int((~greedy_arr & correct_arr).sum())
    r2w = int((greedy_arr & ~correct_arr).sum())
    net_gain = w2r - r2w

    return {
        "n_correct": int(correct_arr.sum()),
        "accuracy": round(float(correct_arr.mean()), 4),
        "total_samples": total_samples,
        "savings_pct": round(100 * (1 - total_samples / 3200), 1),
        "relative_cost_pct": round(100 * total_samples / 3200, 1),
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net_gain": net_gain,
        "per_problem": per_problem,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Experiment 8: Adaptive Sampling Budget ===")

    # ---- Step 1: Data loading & model fitting ----
    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    _, holdout_idx = get_train_holdout_indices()
    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    with open(PHASE3_DIR / "holdout_temperature_generations.json") as f:
        holdout_gen = json.load(f)
    logger.info("Loaded %d holdout problems with 32 completions each", len(holdout_gen))

    # SSS ordering fix
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(
        sss.split(np.zeros(len(baseline_correct)), baseline_correct.astype(int))
    )
    train_sort_perm = np.argsort(train_idx_sss)
    hold_sort_perm = np.argsort(hold_idx_sss)

    train_features_78 = np.load(FEATURES_TRAIN_PATH)[train_sort_perm]
    holdout_features_78 = np.load(FEATURES_HOLDOUT_PATH)[hold_sort_perm]

    with open(Path(__file__).parent / "phase1" / "feature_names.json") as f:
        feature_names = json.load(f)

    abc_indices = get_abc_column_indices(feature_names)
    X_train = train_features_78[:, abc_indices]
    X_holdout = holdout_features_78[:, abc_indices]

    train_idx_sorted = np.sort(train_idx_sss)
    hold_idx_sorted = np.sort(hold_idx_sss)
    assert np.array_equal(hold_idx_sorted, holdout_idx), "Holdout index mismatch!"

    y_train = baseline_correct[train_idx_sorted].astype(int)
    y_holdout = baseline_correct[holdout_idx].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)
    holdout_probs = clf.predict_proba(X_holdout_scaled)[:, 1]

    logger.info(
        "Holdout probs: min=%.3f, median=%.3f, max=%.3f",
        holdout_probs.min(), np.median(holdout_probs), holdout_probs.max(),
    )

    # ---- Step 3: Uniform baselines ----
    logger.info("\nStep 3: Uniform MV baselines...")
    uniform_results = []
    for n_samples in [1, 2, 4, 8, 16, 32]:
        assignments = {i: n_samples for i in range(len(holdout_idx))}
        res = compute_strategy(
            holdout_idx, holdout_gen, ground_truths, baseline_correct, assignments
        )
        uniform_results.append({
            "N": n_samples,
            "n_correct": res["n_correct"],
            "accuracy": res["accuracy"],
            "total_samples": res["total_samples"],
            "relative_cost_pct": res["relative_cost_pct"],
            "net_gain": res["net_gain"],
            "wrong_to_right": res["wrong_to_right"],
            "right_to_wrong": res["right_to_wrong"],
        })
        logger.info(
            "  N=%2d: %d/100 correct (net %+d, W->R=%d, R->W=%d) | %d samples (%.1f%%)",
            n_samples, res["n_correct"], res["net_gain"],
            res["wrong_to_right"], res["right_to_wrong"],
            res["total_samples"], res["relative_cost_pct"],
        )

    # Verify MV@32 matches experiment 1
    mv32 = [u for u in uniform_results if u["N"] == 32][0]
    logger.info("  MV@32 check: %d correct (exp1 reference: 19)", mv32["n_correct"])

    # ---- Step 4: Binary gated strategy ----
    logger.info("\nStep 4: Binary gated (trust greedy if prob >= tau, else MV@32)...")
    binary_results = []
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
        assignments = {}
        for i in range(len(holdout_idx)):
            assignments[i] = 0 if holdout_probs[i] >= tau else 32
        res = compute_strategy(
            holdout_idx, holdout_gen, ground_truths, baseline_correct, assignments
        )
        n_easy = sum(1 for i in range(len(holdout_idx)) if holdout_probs[i] >= tau)
        binary_results.append({
            "tau": tau,
            "n_trust_greedy": n_easy,
            "n_use_mv32": 100 - n_easy,
            "n_correct": res["n_correct"],
            "accuracy": res["accuracy"],
            "total_samples": res["total_samples"],
            "savings_pct": res["savings_pct"],
            "net_gain": res["net_gain"],
            "wrong_to_right": res["wrong_to_right"],
            "right_to_wrong": res["right_to_wrong"],
        })
        logger.info(
            "  tau=%.1f: %d trust greedy, %d MV@32 | %d correct (net %+d, R->W=%d) | %d samples (%.1f%% savings)",
            tau, n_easy, 100 - n_easy, res["n_correct"], res["net_gain"],
            res["right_to_wrong"], res["total_samples"], res["savings_pct"],
        )

    # ---- Step 5: Three-tier adaptive ----
    logger.info("\nStep 5: Three-tier adaptive sweep...")
    three_tier_results = []

    tau_easy_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    tau_hard_vals = [0.05, 0.1, 0.15, 0.2]
    n_med_vals = [4, 8, 16]

    for tau_easy in tau_easy_vals:
        for tau_hard in tau_hard_vals:
            if tau_hard >= tau_easy:
                continue
            for n_med in n_med_vals:
                assignments = {}
                n_easy = n_medium = n_hard = 0
                for i in range(len(holdout_idx)):
                    p = holdout_probs[i]
                    if p >= tau_easy:
                        assignments[i] = 0
                        n_easy += 1
                    elif p <= tau_hard:
                        assignments[i] = 32
                        n_hard += 1
                    else:
                        assignments[i] = n_med
                        n_medium += 1

                res = compute_strategy(
                    holdout_idx, holdout_gen, ground_truths, baseline_correct, assignments
                )
                three_tier_results.append({
                    "tau_easy": tau_easy,
                    "tau_hard": tau_hard,
                    "n_med": n_med,
                    "n_easy": n_easy,
                    "n_medium": n_medium,
                    "n_hard": n_hard,
                    "n_correct": res["n_correct"],
                    "accuracy": res["accuracy"],
                    "total_samples": res["total_samples"],
                    "savings_pct": res["savings_pct"],
                    "relative_cost_pct": res["relative_cost_pct"],
                    "net_gain": res["net_gain"],
                    "wrong_to_right": res["wrong_to_right"],
                    "right_to_wrong": res["right_to_wrong"],
                })

    # Sort by net gain desc, then savings desc
    three_tier_results.sort(key=lambda x: (-x["net_gain"], -x["savings_pct"]))

    logger.info("  Top 10 three-tier configurations:")
    logger.info(
        "  tau_e | tau_h | N_med | easy | med | hard | correct | net | W->R | R->W | samples | savings"
    )
    logger.info("  " + "-" * 95)
    for r in three_tier_results[:10]:
        logger.info(
            "  %.2f  | %.2f  |  %2d   | %3d  | %3d | %3d  |   %3d   | %+3d |  %2d  |  %2d  |  %5d  | %.1f%%",
            r["tau_easy"], r["tau_hard"], r["n_med"],
            r["n_easy"], r["n_medium"], r["n_hard"],
            r["n_correct"], r["net_gain"],
            r["wrong_to_right"], r["right_to_wrong"],
            r["total_samples"], r["savings_pct"],
        )

    # ---- Step 5b: Four-tier variant ----
    logger.info("\nStep 5b: Four-tier variant (N=4 and N=16 medium tiers)...")
    four_tier_results = []

    tau_boundaries = [
        (0.5, 0.3, 0.1),  # easy >= 0.5, med-easy 0.3-0.5, med-hard 0.1-0.3, hard <= 0.1
        (0.6, 0.3, 0.1),
        (0.5, 0.2, 0.1),
        (0.7, 0.4, 0.15),
        (0.5, 0.3, 0.05),
    ]

    for tau_e, tau_me, tau_h in tau_boundaries:
        assignments = {}
        counts = {"easy": 0, "med_easy": 0, "med_hard": 0, "hard": 0}
        for i in range(len(holdout_idx)):
            p = holdout_probs[i]
            if p >= tau_e:
                assignments[i] = 0
                counts["easy"] += 1
            elif p >= tau_me:
                assignments[i] = 4
                counts["med_easy"] += 1
            elif p > tau_h:
                assignments[i] = 16
                counts["med_hard"] += 1
            else:
                assignments[i] = 32
                counts["hard"] += 1

        res = compute_strategy(
            holdout_idx, holdout_gen, ground_truths, baseline_correct, assignments
        )
        four_tier_results.append({
            "tau_easy": tau_e,
            "tau_med_easy": tau_me,
            "tau_hard": tau_h,
            "n_per_tier": counts,
            "n_correct": res["n_correct"],
            "accuracy": res["accuracy"],
            "total_samples": res["total_samples"],
            "savings_pct": res["savings_pct"],
            "net_gain": res["net_gain"],
            "wrong_to_right": res["wrong_to_right"],
            "right_to_wrong": res["right_to_wrong"],
        })
        logger.info(
            "  [%.1f/%.1f/%.1f]: %s | %d correct (net %+d) | %d samples (%.1f%% savings)",
            tau_e, tau_me, tau_h,
            "/".join(f"{v}" for v in counts.values()),
            res["n_correct"], res["net_gain"],
            res["total_samples"], res["savings_pct"],
        )

    # ---- Step 6: Continuous budget ----
    logger.info("\nStep 6: Continuous proportional allocation...")
    assignments = {}
    for i in range(len(holdout_idx)):
        raw_n = 32 * (1 - holdout_probs[i])
        # Round to nearest power of 2 (1, 2, 4, 8, 16, 32), or 0 if very high conf
        if raw_n < 0.5:
            assignments[i] = 0
        elif raw_n < 1.5:
            assignments[i] = 1
        elif raw_n < 3:
            assignments[i] = 2
        elif raw_n < 6:
            assignments[i] = 4
        elif raw_n < 12:
            assignments[i] = 8
        elif raw_n < 24:
            assignments[i] = 16
        else:
            assignments[i] = 32

    res_continuous = compute_strategy(
        holdout_idx, holdout_gen, ground_truths, baseline_correct, assignments
    )
    # Distribution of N values
    n_dist = Counter(assignments.values())
    continuous_result = {
        "method": "proportional_32*(1-p)_pow2",
        "n_distribution": {str(k): v for k, v in sorted(n_dist.items())},
        "n_correct": res_continuous["n_correct"],
        "accuracy": res_continuous["accuracy"],
        "total_samples": res_continuous["total_samples"],
        "savings_pct": res_continuous["savings_pct"],
        "net_gain": res_continuous["net_gain"],
        "wrong_to_right": res_continuous["wrong_to_right"],
        "right_to_wrong": res_continuous["right_to_wrong"],
    }
    logger.info(
        "  Continuous: %d correct (net %+d, R->W=%d) | %d samples (%.1f%% savings)",
        res_continuous["n_correct"], res_continuous["net_gain"],
        res_continuous["right_to_wrong"],
        res_continuous["total_samples"], res_continuous["savings_pct"],
    )
    logger.info("  N distribution: %s", dict(sorted(n_dist.items())))

    # ---- Step 7: Bootstrap robustness for best 3-tier ----
    logger.info("\nStep 7: Bootstrap robustness for best 3-tier...")
    best_3t = three_tier_results[0]
    tau_e_best = best_3t["tau_easy"]
    tau_h_best = best_3t["tau_hard"]
    n_med_best = best_3t["n_med"]

    rng = np.random.default_rng(42)
    n_boot_trials = 500
    boot_correct_counts = []

    for trial in range(n_boot_trials):
        trial_correct = 0
        for local_i, gi in enumerate(holdout_idx):
            p = holdout_probs[local_i]
            greedy_ok = bool(baseline_correct[gi])
            gt = ground_truths[gi]
            gi_str = str(gi)

            if p >= tau_e_best:
                trial_correct += greedy_ok
            elif p <= tau_h_best:
                # Full N=32, deterministic
                if gi_str in holdout_gen:
                    correct, _, _, _ = majority_vote_at_n(
                        holdout_gen[gi_str]["completions"], gt, 32
                    )
                    trial_correct += correct
                else:
                    trial_correct += greedy_ok
            else:
                # Medium tier: random subsample of n_med from 32
                if gi_str in holdout_gen:
                    completions = holdout_gen[gi_str]["completions"]
                    idx = rng.choice(32, size=n_med_best, replace=False)
                    subset = [completions[j] for j in idx]
                    answers = [normalize_answer(extract_answer(c)) for c in subset]
                    counter = Counter(answers)
                    best_norm = counter.most_common(1)[0][0]
                    for c in subset:
                        if normalize_answer(extract_answer(c)) == best_norm:
                            trial_correct += check_correct(extract_answer(c), gt)
                            break
                else:
                    trial_correct += greedy_ok

        boot_correct_counts.append(trial_correct)

    boot_arr = np.array(boot_correct_counts)
    bootstrap_result = {
        "config": f"tau_e={tau_e_best}, tau_h={tau_h_best}, N_med={n_med_best}",
        "n_trials": n_boot_trials,
        "mean_correct": round(float(boot_arr.mean()), 2),
        "std_correct": round(float(boot_arr.std()), 2),
        "ci_95": [float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))],
        "min_correct": int(boot_arr.min()),
        "max_correct": int(boot_arr.max()),
    }
    logger.info(
        "  Best 3-tier bootstrap (%d trials): %.1f +/- %.1f correct, 95%% CI [%.0f, %.0f]",
        n_boot_trials, boot_arr.mean(), boot_arr.std(),
        np.percentile(boot_arr, 2.5), np.percentile(boot_arr, 97.5),
    )

    # ---- Step 8: CV on train-400 ----
    logger.info("\nStep 8: Cross-validation on train-400...")

    cv_probs = cross_val_predict(
        LogisticRegression(**LR_PARAMS),
        X_train_scaled, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]

    # Load train temperature generations
    with open(TEMP_GEN_PATH) as f:
        train_gen = json.load(f)

    train_idx_sorted_list = train_idx_sorted.tolist()

    # Apply best 3-tier to train-400
    cv_results_list = []
    for label, tau_e, tau_h, n_m in [
        ("best_3tier", tau_e_best, tau_h_best, n_med_best),
        ("binary_0.3", 0.3, -1, 0),  # binary: everything below 0.3 gets MV@32
    ]:
        cv_correct = 0
        cv_total_samples = 0
        cv_w2r = 0
        cv_r2w = 0
        n_cv_easy = 0
        n_cv_med = 0
        n_cv_hard = 0

        for local_i, gi in enumerate(train_idx_sorted_list):
            p = cv_probs[local_i]
            greedy_ok = bool(baseline_correct[gi])
            gt = ground_truths[gi]
            gi_str = str(gi)

            if p >= tau_e:
                cv_correct += greedy_ok
                n_cv_easy += 1
            elif label.startswith("binary") or p <= tau_h:
                # MV@32
                cv_total_samples += 32
                n_cv_hard += 1
                if gi_str in train_gen:
                    correct, _, _, _ = majority_vote_at_n(
                        train_gen[gi_str]["completions"], gt, 32
                    )
                    cv_correct += correct
                    if not greedy_ok and correct:
                        cv_w2r += 1
                    elif greedy_ok and not correct:
                        cv_r2w += 1
                else:
                    cv_correct += greedy_ok
            else:
                # Medium
                cv_total_samples += n_m
                n_cv_med += 1
                if gi_str in train_gen:
                    correct, _, _, _ = majority_vote_at_n(
                        train_gen[gi_str]["completions"], gt, n_m
                    )
                    cv_correct += correct
                    if not greedy_ok and correct:
                        cv_w2r += 1
                    elif greedy_ok and not correct:
                        cv_r2w += 1
                else:
                    cv_correct += greedy_ok

        cv_results_list.append({
            "label": label,
            "tau_easy": tau_e,
            "tau_hard": tau_h if tau_h >= 0 else None,
            "n_med": n_m if n_m > 0 else None,
            "n_problems": 400,
            "n_correct": cv_correct,
            "accuracy": round(cv_correct / 400, 4),
            "total_samples": cv_total_samples,
            "savings_pct": round(100 * (1 - cv_total_samples / (400 * 32)), 1),
            "net_gain": cv_w2r - cv_r2w,
            "wrong_to_right": cv_w2r,
            "right_to_wrong": cv_r2w,
            "n_easy": n_cv_easy,
            "n_medium": n_cv_med,
            "n_hard": n_cv_hard,
        })

    # Also compute uniform baselines on train-400
    cv_uniform = []
    for n_samples in [1, 8, 32]:
        cv_correct = 0
        cv_w2r = 0
        cv_r2w = 0
        for local_i, gi in enumerate(train_idx_sorted_list):
            greedy_ok = bool(baseline_correct[gi])
            gt = ground_truths[gi]
            gi_str = str(gi)
            if gi_str in train_gen:
                correct, _, _, _ = majority_vote_at_n(
                    train_gen[gi_str]["completions"], gt, n_samples
                )
                cv_correct += correct
                if not greedy_ok and correct:
                    cv_w2r += 1
                elif greedy_ok and not correct:
                    cv_r2w += 1
            else:
                cv_correct += greedy_ok
        cv_uniform.append({
            "N": n_samples,
            "n_correct": cv_correct,
            "accuracy": round(cv_correct / 400, 4),
            "net_gain": cv_w2r - cv_r2w,
        })

    for r in cv_results_list:
        logger.info(
            "  CV %s: %d/400 correct (net %+d) | %d samples (%.1f%% savings) | easy=%d med=%d hard=%d",
            r["label"], r["n_correct"], r["net_gain"],
            r["total_samples"], r["savings_pct"],
            r["n_easy"], r["n_medium"], r["n_hard"],
        )
    for r in cv_uniform:
        logger.info(
            "  CV uniform N=%d: %d/400 correct (net %+d)",
            r["N"], r["n_correct"], r["net_gain"],
        )

    # ---- Step 9: Pareto frontier ----
    logger.info("\nStep 9: Computing Pareto frontier...")

    all_strategies = []

    # Greedy only
    greedy_correct = int(y_holdout.sum())
    all_strategies.append({
        "label": "greedy_only",
        "n_correct": greedy_correct,
        "accuracy": round(greedy_correct / 100, 4),
        "total_samples": 0,
        "relative_cost_pct": 0.0,
    })

    # Uniform
    for u in uniform_results:
        all_strategies.append({
            "label": f"uniform_N{u['N']}",
            "n_correct": u["n_correct"],
            "accuracy": u["accuracy"],
            "total_samples": u["total_samples"],
            "relative_cost_pct": u["relative_cost_pct"],
        })

    # Binary gated
    for b in binary_results:
        all_strategies.append({
            "label": f"binary_tau{b['tau']}",
            "n_correct": b["n_correct"],
            "accuracy": b["accuracy"],
            "total_samples": b["total_samples"],
            "relative_cost_pct": round(100 * b["total_samples"] / 3200, 1),
        })

    # Top 3-tier results (top 20 by net gain)
    for r in three_tier_results[:20]:
        all_strategies.append({
            "label": f"3tier_e{r['tau_easy']}_h{r['tau_hard']}_m{r['n_med']}",
            "n_correct": r["n_correct"],
            "accuracy": r["accuracy"],
            "total_samples": r["total_samples"],
            "relative_cost_pct": r["relative_cost_pct"],
        })

    # Four-tier
    for r in four_tier_results:
        all_strategies.append({
            "label": f"4tier_{r['tau_easy']}_{r['tau_med_easy']}_{r['tau_hard']}",
            "n_correct": r["n_correct"],
            "accuracy": r["accuracy"],
            "total_samples": r["total_samples"],
            "relative_cost_pct": round(100 * r["total_samples"] / 3200, 1),
        })

    # Continuous
    all_strategies.append({
        "label": "continuous_proportional",
        "n_correct": continuous_result["n_correct"],
        "accuracy": continuous_result["accuracy"],
        "total_samples": continuous_result["total_samples"],
        "relative_cost_pct": round(100 * continuous_result["total_samples"] / 3200, 1),
    })

    # Find Pareto-optimal: no other strategy has both higher accuracy AND lower cost
    pareto_optimal = []
    for s in all_strategies:
        dominated = False
        for other in all_strategies:
            if other is s:
                continue
            if (other["n_correct"] >= s["n_correct"]
                    and other["total_samples"] <= s["total_samples"]
                    and (other["n_correct"] > s["n_correct"]
                         or other["total_samples"] < s["total_samples"])):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(s)

    pareto_optimal.sort(key=lambda x: x["total_samples"])

    logger.info("  Pareto-optimal strategies:")
    for p in pareto_optimal:
        logger.info(
            "    %-45s %d correct | %5d samples (%.1f%%)",
            p["label"], p["n_correct"], p["total_samples"], p["relative_cost_pct"],
        )

    # ---- Step 10: Build paper claims ----
    best_binary = max(binary_results, key=lambda x: x["net_gain"])
    best_3tier = three_tier_results[0]

    # Find best 3-tier that matches or beats binary accuracy
    matching_3tier = [
        r for r in three_tier_results
        if r["n_correct"] >= best_binary["n_correct"]
    ]
    if matching_3tier:
        # Among those matching accuracy, pick highest savings
        best_matching = max(matching_3tier, key=lambda x: x["savings_pct"])
    else:
        best_matching = best_3tier

    # Find best 3-tier that matches uniform MV@32 accuracy
    mv32_correct = mv32["n_correct"]
    matching_mv32 = [
        r for r in three_tier_results
        if r["n_correct"] >= mv32_correct
    ]
    if matching_mv32:
        best_matching_mv32 = max(matching_mv32, key=lambda x: x["savings_pct"])
    else:
        best_matching_mv32 = None

    paper_claims = {
        "greedy_baseline": greedy_correct,
        "uniform_mv32": mv32["n_correct"],
        "binary_gated_best": {
            "tau": best_binary["tau"],
            "n_correct": best_binary["n_correct"],
            "savings_pct": best_binary["savings_pct"],
        },
        "best_3tier_overall": {
            "config": f"tau_e={best_3tier['tau_easy']}, tau_h={best_3tier['tau_hard']}, N_med={best_3tier['n_med']}",
            "n_correct": best_3tier["n_correct"],
            "savings_pct": best_3tier["savings_pct"],
            "net_gain": best_3tier["net_gain"],
        },
    }
    if best_matching and best_matching != best_3tier:
        paper_claims["best_3tier_matching_binary"] = {
            "config": f"tau_e={best_matching['tau_easy']}, tau_h={best_matching['tau_hard']}, N_med={best_matching['n_med']}",
            "n_correct": best_matching["n_correct"],
            "savings_pct": best_matching["savings_pct"],
        }
    if best_matching_mv32:
        paper_claims["best_3tier_matching_mv32"] = {
            "config": f"tau_e={best_matching_mv32['tau_easy']}, tau_h={best_matching_mv32['tau_hard']}, N_med={best_matching_mv32['n_med']}",
            "n_correct": best_matching_mv32["n_correct"],
            "savings_pct": best_matching_mv32["savings_pct"],
        }

    # ---- Save results ----
    logger.info("\nSaving results...")

    # Get per-problem assignments for best 3-tier
    best_assignments = {}
    best_tiers = {}
    for i in range(len(holdout_idx)):
        p = holdout_probs[i]
        if p >= tau_e_best:
            best_assignments[i] = 0
            best_tiers[i] = "easy"
        elif p <= tau_h_best:
            best_assignments[i] = 32
            best_tiers[i] = "hard"
        else:
            best_assignments[i] = n_med_best
            best_tiers[i] = "medium"

    best_res = compute_strategy(
        holdout_idx, holdout_gen, ground_truths, baseline_correct, best_assignments
    )

    per_problem_out = []
    for pp in best_res["per_problem"]:
        pp["tier"] = best_tiers[pp["local_idx"]]
        pp["topo_confidence"] = round(float(holdout_probs[pp["local_idx"]]), 4)
        per_problem_out.append(pp)

    results = {
        "experiment": "adaptive_sampling_budget",
        "description": (
            "Partition holdout problems by topo-confidence into tiers "
            "and allocate different sampling budgets. Compare to uniform N=32."
        ),
        "n_holdout": 100,
        "greedy_baseline_correct": greedy_correct,
        "cost_model": "temperature_samples_only (greedy is sunk cost)",
        "uniform_baselines": uniform_results,
        "binary_gated": binary_results,
        "three_tier_sweep": three_tier_results[:30],  # top 30
        "four_tier": four_tier_results,
        "continuous": continuous_result,
        "bootstrap_robustness": bootstrap_result,
        "cv_train400": {
            "adaptive_strategies": cv_results_list,
            "uniform_baselines": cv_uniform,
        },
        "pareto_frontier": pareto_optimal,
        "paper_claims": paper_claims,
        "best_three_tier_config": {
            "tau_easy": tau_e_best,
            "tau_hard": tau_h_best,
            "n_med": n_med_best,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_DIR / "adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "per_problem_assignments.json", "w") as f:
        json.dump(per_problem_out, f, indent=2)

    with open(OUTPUT_DIR / "pareto_data.json", "w") as f:
        json.dump(all_strategies, f, indent=2)

    # ---- Console summary ----
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 8 RESULTS: Adaptive Sampling Budget")
    logger.info("=" * 70)

    logger.info("\n--- Uniform Baselines ---")
    logger.info("  N  | Correct | Net Gain | W->R | R->W | Samples | Cost%%")
    logger.info("  " + "-" * 60)
    for u in uniform_results:
        logger.info(
            "  %2d |  %3d    |   %+3d    |  %2d  |  %2d  |  %5d  | %5.1f%%",
            u["N"], u["n_correct"], u["net_gain"],
            u["wrong_to_right"], u["right_to_wrong"],
            u["total_samples"], u["relative_cost_pct"],
        )

    logger.info("\n--- Binary Gated ---")
    for b in binary_results:
        logger.info(
            "  tau=%.1f: %d correct (net %+d, R->W=%d) | %.1f%% savings",
            b["tau"], b["n_correct"], b["net_gain"], b["right_to_wrong"], b["savings_pct"],
        )

    logger.info("\n--- Best Three-Tier ---")
    b3 = three_tier_results[0]
    logger.info(
        "  Config: tau_e=%.2f, tau_h=%.2f, N_med=%d",
        b3["tau_easy"], b3["tau_hard"], b3["n_med"],
    )
    logger.info(
        "  Tiers: %d easy, %d medium, %d hard",
        b3["n_easy"], b3["n_medium"], b3["n_hard"],
    )
    logger.info(
        "  Result: %d correct (net %+d, W->R=%d, R->W=%d) | %d samples (%.1f%% savings)",
        b3["n_correct"], b3["net_gain"], b3["wrong_to_right"], b3["right_to_wrong"],
        b3["total_samples"], b3["savings_pct"],
    )

    logger.info("\n--- Pareto Frontier ---")
    for p in pareto_optimal:
        logger.info(
            "  %-45s %d correct | %5d samples",
            p["label"], p["n_correct"], p["total_samples"],
        )

    logger.info("\n--- Paper Claims ---")
    for k, v in paper_claims.items():
        logger.info("  %s: %s", k, v)

    logger.info("\nResults saved to: %s", OUTPUT_DIR)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
