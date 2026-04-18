#!/usr/bin/env python3
"""Experiment 1: Pure Majority Vote Ablation.

The paper's +11 holdout result uses a GATED strategy: trust greedy for
problems where greedy was correct, apply majority vote only to wrong-greedy
problems. This guarantees R→W = 0 by construction.

This experiment removes the gate and applies majority vote to ALL 100
holdout problems, answering: does the +11 come from gating (protecting
correct-greedy answers) or is majority vote alone sufficient?

Key question: How many of the 11 correct-greedy problems does majority
vote get WRONG (R→W regressions)?

No GPU required — operates entirely on existing holdout_temperature_generations.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Add pathway4/track_a to path so common.py is importable
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BASELINE_CORRECT_PATH,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    PHASE3_DIR,
    TIER_ASSIGNMENTS_PATH,
    bayesian_p_improvement,
    bootstrap_ci_net_gain,
    check_correct,
    extract_answer,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    logger,
    mcnemar_mid_p,
    normalize_answer,
)

OUTPUT_DIR = Path(__file__).parent / "experiment1_ablation"


def majority_vote_answer(completions: list[str]) -> tuple[str, Counter]:
    """Extract the majority-vote answer and vote distribution.

    Returns:
        (best_answer_raw, counter_of_normalized_answers)
    """
    answers = [normalize_answer(extract_answer(c)) for c in completions]
    counter = Counter(answers)
    best_norm = counter.most_common(1)[0][0]
    # Find a completion producing this answer (for raw form)
    for c in completions:
        if normalize_answer(extract_answer(c)) == best_norm:
            return extract_answer(c), counter
    return extract_answer(completions[0]), counter


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Experiment 1: Pure Majority Vote Ablation ===")

    # ---- Load data ----
    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    _, holdout_idx = get_train_holdout_indices()
    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    with open(PHASE3_DIR / "holdout_temperature_generations.json") as f:
        holdout_gen = json.load(f)
    logger.info("Loaded %d holdout problems with 32 completions each", len(holdout_gen))

    # Load existing gated result for comparison
    with open(PHASE3_DIR / "holdout_results.json") as f:
        gated_results = json.load(f)

    # ---- Apply majority vote to ALL holdout problems ----
    per_problem = []

    for gi in holdout_idx:
        gi_str = str(gi)
        greedy_correct = bool(baseline_correct[gi])
        gt = ground_truths[gi]

        if gi_str not in holdout_gen:
            logger.warning("Problem %d missing from holdout generations!", gi)
            per_problem.append({
                "global_idx": int(gi),
                "greedy_correct": greedy_correct,
                "mv_correct": greedy_correct,  # fallback to greedy
                "mv_answer": None,
                "vote_margin": None,
                "n_unique_answers": None,
                "category": "missing",
            })
            continue

        completions = holdout_gen[gi_str]["completions"]
        mv_answer, vote_counter = majority_vote_answer(completions)
        mv_correct = check_correct(mv_answer, gt)

        # Vote analysis
        top_count = vote_counter.most_common(1)[0][1]
        vote_margin = top_count / len(completions)
        n_unique = len(vote_counter)

        # Classify flip
        if greedy_correct and mv_correct:
            category = "R→R"
        elif greedy_correct and not mv_correct:
            category = "R→W"
        elif not greedy_correct and mv_correct:
            category = "W→R"
        else:
            category = "W→W"

        per_problem.append({
            "global_idx": int(gi),
            "greedy_correct": greedy_correct,
            "mv_correct": mv_correct,
            "mv_answer": mv_answer,
            "gt_answer": gt,
            "vote_margin": round(vote_margin, 3),
            "top_vote_count": top_count,
            "n_unique_answers": n_unique,
            "vote_distribution": dict(vote_counter.most_common()),
            "category": category,
        })

    # ---- Compute metrics ----
    greedy_arr = np.array([p["greedy_correct"] for p in per_problem], dtype=bool)
    mv_arr = np.array([p["mv_correct"] for p in per_problem], dtype=bool)

    w2r = int((~greedy_arr & mv_arr).sum())
    r2w = int((greedy_arr & ~mv_arr).sum())
    r2r = int((greedy_arr & mv_arr).sum())
    w2w = int((~greedy_arr & ~mv_arr).sum())
    net_gain = w2r - r2w

    p_mcnemar = mcnemar_mid_p(r2w, w2r)
    p_improve = bayesian_p_improvement(w2r, r2w)
    ci_lo, ci_hi = bootstrap_ci_net_gain(
        greedy_arr.astype(int), mv_arr.astype(int)
    )

    # ---- Vote margin analysis ----
    r2w_problems = [p for p in per_problem if p["category"] == "R→W"]
    r2r_problems = [p for p in per_problem if p["category"] == "R→R"]
    w2r_problems = [p for p in per_problem if p["category"] == "W→R"]
    w2w_problems = [p for p in per_problem if p["category"] == "W→W"]

    # Mean vote margin by category
    def mean_margin(problems):
        margins = [p["vote_margin"] for p in problems if p["vote_margin"] is not None]
        return round(np.mean(margins), 3) if margins else None

    # ---- Build results ----
    ablation_results = {
        "experiment": "pure_majority_vote_ablation",
        "description": (
            "Apply majority vote to ALL 100 holdout problems (no greedy gating). "
            "Compare to the gated result which skips MV for correct-greedy problems."
        ),
        "n_holdout": len(holdout_idx),
        "ungated_majority_vote": {
            "baseline_accuracy": round(float(greedy_arr.mean()), 4),
            "selected_accuracy": round(float(mv_arr.mean()), 4),
            "baseline_correct": int(greedy_arr.sum()),
            "selected_correct": int(mv_arr.sum()),
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "right_to_right": r2r,
            "wrong_to_wrong": w2w,
            "net_gain": net_gain,
            "mcnemar_mid_p": round(p_mcnemar, 6),
            "bayesian_p_improvement": round(p_improve, 4),
            "bootstrap_ci_95": [ci_lo, ci_hi],
        },
        "gated_majority_vote": {
            "baseline_correct": gated_results["baseline_correct"],
            "selected_correct": gated_results["selected_correct"],
            "wrong_to_right": gated_results["wrong_to_right"],
            "right_to_wrong": gated_results["right_to_wrong"],
            "net_gain": gated_results["net_gain"],
            "note": "R→W=0 by construction (greedy-correct problems never touched)",
        },
        "comparison": {
            "gating_prevents_regressions": r2w,
            "gating_effect_on_net_gain": gated_results["net_gain"] - net_gain,
            "conclusion": None,  # filled below
        },
        "vote_margin_analysis": {
            "R→R": {"count": len(r2r_problems), "mean_vote_margin": mean_margin(r2r_problems)},
            "R→W": {"count": len(r2w_problems), "mean_vote_margin": mean_margin(r2w_problems)},
            "W→R": {"count": len(w2r_problems), "mean_vote_margin": mean_margin(w2r_problems)},
            "W→W": {"count": len(w2w_problems), "mean_vote_margin": mean_margin(w2w_problems)},
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Determine conclusion
    if r2w == 0:
        conclusion = (
            "MAJORITY VOTE ALONE achieves the same +{} with 0 regressions. "
            "Gating adds no value for selection — topology's contribution "
            "must be reframed around verification/calibration."
        ).format(net_gain)
    elif net_gain >= gated_results["net_gain"]:
        conclusion = (
            "Majority vote gets +{} net (W→R={}, R→W={}). Despite {} regressions, "
            "net gain matches or exceeds gated result. Gating protects {} problems "
            "but doesn't improve net accuracy."
        ).format(net_gain, w2r, r2w, r2w, r2w)
    elif net_gain > 0 and r2w > 0:
        conclusion = (
            "Majority vote gets +{} net (W→R={}, R→W={}), below gated +{}. "
            "Gating prevents {} regressions, adding +{} net. "
            "TOPOLOGY GATING PROVIDES REAL VALUE for selection."
        ).format(net_gain, w2r, r2w, gated_results["net_gain"], r2w,
                 gated_results["net_gain"] - net_gain)
    else:
        conclusion = (
            "Majority vote HURTS: net gain={} (W→R={}, R→W={}). "
            "Gating is essential — without it, majority vote creates more "
            "regressions than improvements."
        ).format(net_gain, w2r, r2w)

    ablation_results["comparison"]["conclusion"] = conclusion

    # ---- Save results ----
    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    with open(OUTPUT_DIR / "per_problem_detail.json", "w") as f:
        json.dump(per_problem, f, indent=2)

    # ---- Topo-confidence gated analysis ----
    logger.info("Step 2: Topo-confidence gated majority vote...")

    # CRITICAL: features_train400.npy and features_holdout100.npy are stored
    # in StratifiedShuffleSplit order (unsorted), NOT np.where(mask) order.
    # three_number.py (pathway1/phase1) used SSS with seed=9999, and saved
    # features indexed by SSS order. We must reconstruct this ordering to
    # align features with labels correctly.
    logger.info("  Reconstructing SSS feature ordering...")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(sss.split(
        np.zeros(len(baseline_correct)), baseline_correct.astype(int)
    ))

    # Permutations to convert SSS order → sorted order
    train_sort_perm = np.argsort(train_idx_sss)
    hold_sort_perm = np.argsort(hold_idx_sss)

    train_features_78 = np.load(FEATURES_TRAIN_PATH)
    holdout_features_78 = np.load(FEATURES_HOLDOUT_PATH)

    # Reorder from SSS → sorted to match get_train_holdout_indices() labels
    train_features_78 = train_features_78[train_sort_perm]
    holdout_features_78 = holdout_features_78[hold_sort_perm]

    # Verify alignment
    train_idx_sorted = np.sort(train_idx_sss)
    hold_idx_sorted = np.sort(hold_idx_sss)
    assert np.array_equal(hold_idx_sorted, holdout_idx), "Holdout index mismatch!"

    with open(
        Path(__file__).parent / "phase1" / "feature_names.json"
    ) as f:
        feature_names = json.load(f)

    abc_indices = get_abc_column_indices(feature_names)
    X_train = train_features_78[:, abc_indices]
    X_holdout = holdout_features_78[:, abc_indices]

    y_train = baseline_correct[train_idx_sorted].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)
    holdout_probs = clf.predict_proba(X_holdout_scaled)[:, 1]

    # Sweep thresholds for topo-confidence gating
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gating_sweep = []

    for tau in thresholds:
        gated_results_list = []
        for i, gi in enumerate(holdout_idx):
            greedy_ok = bool(baseline_correct[gi])
            conf = holdout_probs[i]

            if conf >= tau:
                # Trust greedy
                gated_results_list.append(greedy_ok)
            else:
                # Use majority vote
                gi_str = str(gi)
                if gi_str in holdout_gen:
                    completions = holdout_gen[gi_str]["completions"]
                    mv_ans, _ = majority_vote_answer(completions)
                    gated_results_list.append(check_correct(mv_ans, ground_truths[gi]))
                else:
                    gated_results_list.append(greedy_ok)

        gated_arr = np.array(gated_results_list, dtype=bool)
        g_w2r = int((~greedy_arr & gated_arr).sum())
        g_r2w = int((greedy_arr & ~gated_arr).sum())
        g_net = g_w2r - g_r2w
        n_gated_greedy = int((holdout_probs >= tau).sum())

        gating_sweep.append({
            "threshold": tau,
            "n_trust_greedy": n_gated_greedy,
            "n_use_mv": 100 - n_gated_greedy,
            "correct": int(gated_arr.sum()),
            "wrong_to_right": g_w2r,
            "right_to_wrong": g_r2w,
            "net_gain": g_net,
        })

    ablation_results["topo_confidence_gating"] = {
        "description": (
            "Use P1 topo-confidence model to decide per-problem: "
            "if P(correct) >= tau, trust greedy; else use majority vote."
        ),
        "threshold_sweep": gating_sweep,
    }

    # Find best threshold
    best_gate = max(gating_sweep, key=lambda x: x["net_gain"])
    ablation_results["topo_confidence_gating"]["best_threshold"] = best_gate

    # Re-save with gating results
    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # ---- Console report ----
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: Pure Majority Vote Ablation")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Greedy baseline: %d/100 correct", int(greedy_arr.sum()))
    logger.info("")
    logger.info("--- Ungated Majority Vote (ALL problems) ---")
    logger.info("  Correct: %d/100 (%.0f%%)", int(mv_arr.sum()), 100 * mv_arr.mean())
    logger.info("  W→R: %d  |  R→W: %d  |  Net: %+d", w2r, r2w, net_gain)
    logger.info("  R→R: %d  |  W→W: %d", r2r, w2w)
    logger.info("  McNemar mid-p: %.4f", p_mcnemar)
    logger.info("  P(improvement): %.4f", p_improve)
    logger.info("  Bootstrap 95%% CI: [%d, %d]", ci_lo, ci_hi)
    logger.info("")
    logger.info("--- Gated Majority Vote (phase3 result) ---")
    logger.info("  Correct: %d/100", gated_results["selected_correct"])
    logger.info("  W→R: %d  |  R→W: %d  |  Net: %+d",
                gated_results["wrong_to_right"], gated_results["right_to_wrong"],
                gated_results["net_gain"])
    logger.info("  (R→W=0 by construction: greedy-correct never touched)")
    logger.info("")
    logger.info("--- Vote Margin by Category ---")
    for cat in ["R→R", "R→W", "W→R", "W→W"]:
        data = ablation_results["vote_margin_analysis"][cat]
        logger.info("  %s: n=%d, mean margin=%.3f",
                    cat, data["count"], data["mean_vote_margin"] or 0)
    logger.info("")

    if r2w_problems:
        logger.info("--- R→W Regression Details ---")
        for p in r2w_problems:
            logger.info("  Problem %d: MV answer=%s, GT=%s, margin=%.3f, %d unique answers",
                        p["global_idx"], p["mv_answer"], p["gt_answer"],
                        p["vote_margin"], p["n_unique_answers"])
        logger.info("")

    logger.info("--- Topo-Confidence Gated MV (threshold sweep) ---")
    logger.info("  tau  | trust_greedy | use_mv | correct | W→R | R→W | net")
    logger.info("  " + "-" * 62)
    for g in gating_sweep:
        logger.info("  %.1f  |     %3d      |  %3d   |   %3d   |  %2d |  %2d | %+3d",
                    g["threshold"], g["n_trust_greedy"], g["n_use_mv"],
                    g["correct"], g["wrong_to_right"], g["right_to_wrong"],
                    g["net_gain"])
    logger.info("  Best: tau=%.1f → %d correct, net %+d",
                best_gate["threshold"], best_gate["correct"], best_gate["net_gain"])
    logger.info("")
    logger.info("CONCLUSION: %s", conclusion)
    logger.info("")
    logger.info("Results saved to: %s", OUTPUT_DIR)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
