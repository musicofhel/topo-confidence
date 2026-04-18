#!/usr/bin/env python3
"""Experiment 9: Output-Probability Baselines.

Head-to-head comparison of topo-confidence (AUROC 0.948 on train-400 CV)
against cheap output-probability baselines for predicting greedy correctness
on MATH-500 holdout.

Two classes of baselines:
  (A) Greedy-pass metrics (same 1-forward-pass budget as topo-confidence):
      neg_mean_entropy, mean_max_token_prob, first_token_prob, self_certainty
  (B) Sampling-based metrics (32 forward passes, from existing text):
      p_majority, vote_margin, inv_answer_diversity, neg_agreement_entropy

Each baseline is evaluated as:
  (i)   Correctness AUROC (predicting greedy correctness on holdout-100)
  (ii)  Gating signal (trust greedy if confident, else use majority vote)
  (iii) Statistical comparison vs topo-confidence (bootstrap permutation)

No GPU required — operates on precomputed entropy scores and existing text.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BASELINE_CORRECT_PATH,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    PHASE3_DIR,
    TIER_ASSIGNMENTS_PATH,
    bootstrap_ci_net_gain,
    check_correct,
    extract_answer,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    logger,
    mcnemar_mid_p,
    bayesian_p_improvement,
    normalize_answer,
)

OUTPUT_DIR = Path(__file__).parent / "experiment9_baselines"
REPO_ROOT = Path(__file__).parent.parent.parent
ENTROPY_SCORES_PATH = REPO_ROOT / "data" / "experiment1_v2" / "output_entropy_scores.json"
QWEN_VOCAB_SIZE = 151936  # Qwen2.5-1.5B-Instruct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_auroc(y_true: np.ndarray, scores: np.ndarray,
                    n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Compute AUROC with bootstrap 95% CI.

    Returns (auroc, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    auroc = roc_auc_score(y_true, scores)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        # Need both classes in sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], scores[idx]))
    boot_aurocs = np.array(boot_aurocs)
    ci_lo = float(np.percentile(boot_aurocs, 2.5))
    ci_hi = float(np.percentile(boot_aurocs, 97.5))
    return float(auroc), ci_lo, ci_hi


def auroc_permutation_test(y_true: np.ndarray, scores_a: np.ndarray,
                           scores_b: np.ndarray, n_perm: int = 10000,
                           seed: int = 42) -> float:
    """Bootstrap permutation test for AUROC difference (A vs B).

    Returns p-value for H0: AUROC_A = AUROC_B.
    """
    rng = np.random.RandomState(seed)
    observed_diff = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    count = 0
    for _ in range(n_perm):
        # Randomly swap scores between A and B for each sample
        mask = rng.rand(len(y_true)) < 0.5
        perm_a = np.where(mask, scores_a, scores_b)
        perm_b = np.where(mask, scores_b, scores_a)
        if len(np.unique(y_true)) < 2:
            continue
        perm_diff = roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_perm


def majority_vote_answer(completions: list[str]) -> tuple[str, Counter]:
    """Extract majority-vote answer and vote distribution."""
    answers = [normalize_answer(extract_answer(c)) for c in completions]
    counter = Counter(answers)
    best_norm = counter.most_common(1)[0][0]
    for c in completions:
        if normalize_answer(extract_answer(c)) == best_norm:
            return extract_answer(c), counter
    return extract_answer(completions[0]), counter


def gating_sweep(baseline_scores: np.ndarray, holdout_correct: np.ndarray,
                 holdout_gen: dict, holdout_idx: np.ndarray,
                 ground_truths: list, thresholds: list[float] | None = None,
                 use_percentiles: bool = False) -> list[dict]:
    """Sweep gating thresholds: if score >= tau, trust greedy; else use MV.

    If use_percentiles=True, thresholds are interpreted as percentile values
    of the score distribution and converted to actual score cutoffs.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if use_percentiles:
        actual_thresholds = [float(np.percentile(baseline_scores, p * 100)) for p in thresholds]
    else:
        actual_thresholds = thresholds

    greedy_arr = holdout_correct.astype(bool)
    results = []

    for tau_label, tau_actual in zip(thresholds, actual_thresholds):
        gated = []
        for i, gi in enumerate(holdout_idx):
            if baseline_scores[i] >= tau_actual:
                gated.append(bool(holdout_correct[i]))
            else:
                gi_str = str(gi)
                if gi_str in holdout_gen:
                    mv_ans, _ = majority_vote_answer(holdout_gen[gi_str]["completions"])
                    gated.append(check_correct(mv_ans, ground_truths[gi]))
                else:
                    gated.append(bool(holdout_correct[i]))

        gated_arr = np.array(gated, dtype=bool)
        w2r = int((~greedy_arr & gated_arr).sum())
        r2w = int((greedy_arr & ~gated_arr).sum())
        n_trust = int((baseline_scores >= tau_actual).sum())

        results.append({
            "threshold_label": round(tau_label, 2),
            "threshold_actual": round(tau_actual, 4),
            "n_trust_greedy": n_trust,
            "n_use_mv": len(holdout_idx) - n_trust,
            "correct": int(gated_arr.sum()),
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "net_gain": w2r - r2w,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=== Experiment 9: Output-Probability Baselines ===")

    # ---- Step 1: Load all data ----
    logger.info("Step 1: Loading data...")

    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    _, holdout_idx = get_train_holdout_indices()  # sorted order (for generation lookups)
    problems = load_math500()
    _, ground_truths = get_prompts_and_ground_truths(problems)

    holdout_correct = baseline_correct[holdout_idx].astype(int)
    n_holdout = len(holdout_idx)
    logger.info("Holdout: %d problems, %d greedy-correct (%.0f%%)",
                n_holdout, holdout_correct.sum(), 100 * holdout_correct.mean())

    with open(PHASE3_DIR / "holdout_temperature_generations.json") as f:
        holdout_gen = json.load(f)

    with open(ENTROPY_SCORES_PATH) as f:
        entropy_data = json.load(f)
    entropy_scores = entropy_data["scores"]  # list of 500 dicts
    logger.info("Loaded entropy scores for %d problems (model: %s)",
                len(entropy_scores), entropy_data.get("model", "unknown"))

    # ---- Step 2: Topo-confidence reference ----
    # CRITICAL: features_train400.npy and features_holdout100.npy are stored
    # in StratifiedShuffleSplit order (unsorted), NOT np.where(mask) order.
    # three_number.py (pathway1/phase1) used SSS with seed=9999, and saved
    # features indexed by SSS order. We must reconstruct this ordering to
    # align features with labels correctly.
    logger.info("Step 2: Topo-confidence reference AUROC...")
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
    topo_scores = clf.predict_proba(X_holdout_scaled)[:, 1]

    topo_auroc, topo_ci_lo, topo_ci_hi = bootstrap_auroc(holdout_correct, topo_scores)
    logger.info("Topo-confidence holdout AUROC: %.3f [%.3f–%.3f]",
                topo_auroc, topo_ci_lo, topo_ci_hi)

    # ---- Step 3: Greedy-pass baselines ----
    logger.info("Step 3: Greedy-pass baselines...")

    # Extract holdout scores from precomputed entropy data
    holdout_entropy = np.array([entropy_scores[gi]["entropy"] for gi in holdout_idx])
    holdout_max_prob = np.array([entropy_scores[gi]["max_token_prob"] for gi in holdout_idx])
    holdout_first_prob = np.array([entropy_scores[gi]["first_token_prob"] for gi in holdout_idx])

    # Derived: self-certainty = KL(P||Uniform) = log(V) - H(P)
    # Monotonic with neg_entropy → same AUROC, different scale
    holdout_self_certainty = np.log(QWEN_VOCAB_SIZE) - holdout_entropy

    greedy_baselines = {
        "neg_mean_entropy": -holdout_entropy,          # higher = more confident
        "mean_max_token_prob": holdout_max_prob,        # higher = more confident
        "first_token_prob": holdout_first_prob,         # higher = more confident
        "self_certainty": holdout_self_certainty,       # higher = more confident
    }

    # ---- Step 4: Sampling-based baselines ----
    logger.info("Step 4: Sampling-based baselines...")

    p_majority_arr = np.zeros(n_holdout)
    vote_margin_arr = np.zeros(n_holdout)
    inv_diversity_arr = np.zeros(n_holdout)
    neg_agree_entropy_arr = np.zeros(n_holdout)

    for i, gi in enumerate(holdout_idx):
        gi_str = str(gi)
        if gi_str not in holdout_gen:
            logger.warning("Problem %d missing from holdout generations", gi)
            continue

        completions = holdout_gen[gi_str]["completions"]
        answers = [normalize_answer(extract_answer(c)) for c in completions]
        counter = Counter(answers)
        n_total = len(completions)
        n_unique = len(counter)

        top_count = counter.most_common(1)[0][1]
        second_count = counter.most_common(2)[1][1] if len(counter) > 1 else 0

        p_majority_arr[i] = top_count / n_total
        vote_margin_arr[i] = (top_count - second_count) / n_total
        inv_diversity_arr[i] = 1.0 - n_unique / n_total

        # Shannon entropy of answer distribution
        probs = np.array([c for c in counter.values()], dtype=float) / n_total
        agree_entropy = -np.sum(probs * np.log(probs + 1e-12))
        neg_agree_entropy_arr[i] = -agree_entropy

    sampling_baselines = {
        "p_majority": p_majority_arr,
        "vote_margin": vote_margin_arr,
        "inv_answer_diversity": inv_diversity_arr,
        "neg_agreement_entropy": neg_agree_entropy_arr,
    }

    # ---- Step 5: Evaluate all baselines ----
    logger.info("Step 5: Evaluating all baselines...")

    all_baselines = {"topo_confidence": topo_scores}
    all_baselines.update(greedy_baselines)
    all_baselines.update(sampling_baselines)

    # Pre-compute majority vote results for gating
    # (same as Experiment 1's ungated MV)
    mv_correct_arr = np.zeros(n_holdout, dtype=bool)
    for i, gi in enumerate(holdout_idx):
        gi_str = str(gi)
        if gi_str in holdout_gen:
            mv_ans, _ = majority_vote_answer(holdout_gen[gi_str]["completions"])
            mv_correct_arr[i] = check_correct(mv_ans, ground_truths[gi])
        else:
            mv_correct_arr[i] = bool(holdout_correct[i])

    greedy_arr = holdout_correct.astype(bool)
    ungated_w2r = int((~greedy_arr & mv_correct_arr).sum())
    ungated_r2w = int((greedy_arr & ~mv_correct_arr).sum())
    ungated_net = ungated_w2r - ungated_r2w

    results_by_baseline = {}
    per_problem_scores = {}

    for name, scores in all_baselines.items():
        # (i) AUROC with bootstrap CI
        auroc, ci_lo, ci_hi = bootstrap_auroc(holdout_correct, scores)

        # (ii) Permutation test vs topo-confidence (skip for topo itself)
        if name != "topo_confidence":
            p_perm = auroc_permutation_test(holdout_correct, topo_scores, scores)
        else:
            p_perm = None

        # (iii) Gating sweep
        # For topo-confidence, use raw probability thresholds
        # For other baselines, use percentile-based thresholds
        is_topo = name == "topo_confidence"
        sweep = gating_sweep(
            scores, holdout_correct, holdout_gen, holdout_idx, ground_truths,
            thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            use_percentiles=not is_topo,
        )

        best_gate = max(sweep, key=lambda x: x["net_gain"])

        # Determine baseline type and compute cost
        if name == "topo_confidence":
            btype, compute = "supervised", "1 fwd + PH"
        elif name in greedy_baselines:
            btype, compute = "unsupervised", "1 fwd"
        else:
            btype, compute = "unsupervised", "32 fwd"

        results_by_baseline[name] = {
            "type": btype,
            "compute": compute,
            "auroc": round(auroc, 4),
            "auroc_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "auroc_perm_p_vs_topo": round(p_perm, 4) if p_perm is not None else None,
            "gating_sweep": sweep,
            "best_gate": best_gate,
        }

        per_problem_scores[name] = [round(float(s), 6) for s in scores]

        logger.info("  %-25s AUROC=%.3f [%.3f–%.3f]  best gate: net=%+d (W→R=%d, R→W=%d)%s",
                     name, auroc, ci_lo, ci_hi,
                     best_gate["net_gain"], best_gate["wrong_to_right"],
                     best_gate["right_to_wrong"],
                     f"  p={p_perm:.4f}" if p_perm is not None else "")

    # ---- Step 6: Output ----
    logger.info("\nStep 6: Saving results...")

    # Verify self-certainty == neg_entropy AUROC
    sc_auroc = results_by_baseline["self_certainty"]["auroc"]
    ne_auroc = results_by_baseline["neg_mean_entropy"]["auroc"]
    assert sc_auroc == ne_auroc, (
        f"Self-certainty AUROC ({sc_auroc}) should equal neg_entropy ({ne_auroc}) — "
        f"they are monotonic transforms. Bug in computation."
    )

    # Build output JSON
    output = {
        "experiment": "output_probability_baselines",
        "description": (
            "Head-to-head comparison of topo-confidence vs cheap output-probability "
            "baselines for predicting greedy correctness on MATH-500 holdout-100. "
            "Greedy-pass baselines use 1 forward pass (same as topo). "
            "Sampling-based baselines use 32 forward passes (existing data)."
        ),
        "n_holdout": n_holdout,
        "n_greedy_correct": int(holdout_correct.sum()),
        "reference": {
            "ungated_mv": {
                "correct": int(mv_correct_arr.sum()),
                "wrong_to_right": ungated_w2r,
                "right_to_wrong": ungated_r2w,
                "net_gain": ungated_net,
            },
            "oracle_gated_mv": {
                "correct": int(holdout_correct.sum()) + ungated_w2r,
                "wrong_to_right": ungated_w2r,
                "right_to_wrong": 0,
                "net_gain": ungated_w2r,
                "note": "R→W=0 by construction",
            },
        },
        "baselines": results_by_baseline,
        "notes": {
            "self_certainty_equivalence": (
                "self_certainty = log(151936) - entropy = KL(P||Uniform). "
                "Monotonic transform of neg_mean_entropy → identical AUROC and ranking. "
                "Included per Kang et al. 2025 terminology."
            ),
            "entropy_source": (
                f"Precomputed in {ENTROPY_SCORES_PATH.relative_to(REPO_ROOT)}. "
                "Greedy-pass mean per-token H(softmax(logits)), NOT cross-entropy. "
                "Perplexity (exp of cross-entropy) is a different signal requiring Phase B."
            ),
            "gating_thresholds": (
                "Topo-confidence uses raw P(correct) thresholds [0.3–0.9]. "
                "Other baselines use percentile-based thresholds at matching coverage levels."
            ),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(OUTPUT_DIR / "per_problem_scores.json", "w") as f:
        json.dump({
            "holdout_indices": [int(gi) for gi in holdout_idx],
            "holdout_correct": [int(c) for c in holdout_correct],
            "scores": per_problem_scores,
        }, f, indent=2)

    # ---- Console comparison table ----
    logger.info("")
    logger.info("=" * 90)
    logger.info("COMPARISON TABLE: Output-Probability Baselines vs Topo-Confidence")
    logger.info("=" * 90)
    logger.info("")
    header = f"{'Baseline':<25} {'Type':<12} {'Compute':<10} {'AUROC [95% CI]':<22} {'Best gate':>20}"
    logger.info(header)
    logger.info("-" * 90)

    # Sort by AUROC descending
    sorted_baselines = sorted(results_by_baseline.items(),
                              key=lambda x: x[1]["auroc"], reverse=True)

    for name, res in sorted_baselines:
        auroc_str = f"{res['auroc']:.3f} [{res['auroc_ci_95'][0]:.3f}–{res['auroc_ci_95'][1]:.3f}]"
        bg = res["best_gate"]
        gate_str = f"net={bg['net_gain']:+d} ({bg['wrong_to_right']}W→R/{bg['right_to_wrong']}R→W)"

        # Mark self-certainty as equivalent
        if name == "self_certainty":
            auroc_str = f"= neg_entropy ({res['auroc']:.3f})"

        p_str = ""
        if res["auroc_perm_p_vs_topo"] is not None:
            p_str = f"  p={res['auroc_perm_p_vs_topo']:.3f}"

        logger.info(f"  {name:<23} {res['type']:<12} {res['compute']:<10} {auroc_str:<22} {gate_str}{p_str}")

    logger.info("-" * 90)
    logger.info("  %-23s %-12s %-10s %-22s net=%+d (%dW→R/%dR→W)",
                "Ungated MV", "—", "32 fwd", "—",
                ungated_net, ungated_w2r, ungated_r2w)
    logger.info("  %-23s %-12s %-10s %-22s net=%+d (%dW→R/%dR→W)",
                "Oracle-gated MV", "—", "32 fwd", "—",
                ungated_w2r, ungated_w2r, 0)
    logger.info("")

    # Summary
    topo_res = results_by_baseline["topo_confidence"]
    best_greedy = max(
        [(n, r) for n, r in results_by_baseline.items()
         if n in greedy_baselines and n != "self_certainty"],
        key=lambda x: x[1]["auroc"]
    )
    best_sampling = max(
        [(n, r) for n, r in results_by_baseline.items() if n in sampling_baselines],
        key=lambda x: x[1]["auroc"]
    )

    logger.info("SUMMARY:")
    logger.info("  Topo-confidence AUROC:        %.3f (supervised, 1 fwd + PH)", topo_res["auroc"])
    logger.info("  Best greedy-pass baseline:    %.3f (%s, 1 fwd)",
                best_greedy[1]["auroc"], best_greedy[0])
    logger.info("  Best sampling baseline:       %.3f (%s, 32 fwd)",
                best_sampling[1]["auroc"], best_sampling[0])
    logger.info("  Topo advantage over greedy:   %+.3f",
                topo_res["auroc"] - best_greedy[1]["auroc"])
    logger.info("  Topo advantage over sampling: %+.3f",
                topo_res["auroc"] - best_sampling[1]["auroc"])

    # Phase B trigger check
    if best_greedy[1]["auroc"] > 0.85:
        logger.info("")
        logger.info("⚠ PHASE B TRIGGER: %s achieved AUROC %.3f > 0.85. "
                     "Consider running experiment9b for perplexity/min-token-prob.",
                     best_greedy[0], best_greedy[1]["auroc"])

    logger.info("")
    logger.info("Results saved to: %s", OUTPUT_DIR)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
