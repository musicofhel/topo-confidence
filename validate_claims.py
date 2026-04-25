"""
validate_claims.py — ground-truth every quantitative claim in PROJECT_RECORD.md
and RESEARCH_SUMMARY.md against the source JSONs on disk.

Each Claim entry carries:
  - a short ID and description (mirrors PROJECT_RECORD §1b)
  - source file path (relative to repo root)
  - jq-style key path into the JSON
  - expected value as stated in the narrative docs
  - tolerance (abs) for float comparison
  - label_scheme tag — 256tok_NEW / 256tok_manifest / 1024tok — so we can
    flag anything computed against the superseded truncation-artifact labels.

Run:  python validate_claims.py > validation_report.txt
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _drill(d: Any, path: list[str | int]) -> Any:
    """Walk a nested dict/list by a list of keys/indices; raise KeyError on miss."""
    cur = d
    for p in path:
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"missing key {p!r} in {list(cur.keys())[:8]}...")
            cur = cur[p]
        elif isinstance(cur, list):
            if not isinstance(p, int) or p >= len(cur):
                raise KeyError(f"index {p} out of range (list len={len(cur)})")
            cur = cur[p]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at key {p!r}")
    return cur


@dataclass
class Claim:
    cid: str
    description: str
    file: str
    path: list
    expected: Any
    tol: float = 0.005
    labels: str = "1024tok"  # 256tok_NEW | 256tok_manifest | 1024tok | mixed
    note: str = ""
    result: str = field(default="", init=False)
    actual: Any = field(default=None, init=False)


# --- CLAIMS ------------------------------------------------------------------
# Grouped roughly in the order of sections in RESEARCH_SUMMARY.md.

CLAIMS: list[Claim] = [
    # ----- Section 1 / §1: baselines -----
    Claim("baseline-1.5B-acc", "1.5B MATH-500 K=1 accuracy 0.486",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=1 (greedy T=0)"], 0.486, 0.002, "1024tok"),
    Claim("baseline-K2", "1.5B K=2 majority accuracy 0.413",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=2 majority (T=0.7)"], 0.4132, 0.005, "1024tok"),
    Claim("baseline-K4", "1.5B K=4 majority 0.495",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=4 majority (T=0.7)"], 0.495, 0.003, "1024tok"),
    Claim("baseline-K8", "1.5B K=8 majority 0.550",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=8 majority (T=0.7)"], 0.55, 0.005, "1024tok"),
    Claim("oracle-K", "Oracle compute/problem 1.01",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["oracle_compute_per_problem"], 1.01, 0.02, "1024tok"),
    Claim("oracle-acc", "Oracle accuracy 0.589",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["oracle_overall_accuracy"], 0.589, 0.005, "1024tok"),

    # ----- Bucket sizes (used across §7, §9, §10) -----
    Claim("buckets-A", "Bucket A always-right n=207",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "A_always_right"], 207, 0, "1024tok"),
    Claim("buckets-B", "Bucket B recoverable n=68",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "B_recoverable"], 68, 0, "1024tok"),
    Claim("buckets-C", "Bucket C never-right n=189",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "C_never_right"], 189, 0, "1024tok"),
    Claim("buckets-D", "Bucket D pathological n=36",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "D_pathological"], 36, 0, "1024tok"),

    # ----- §2 / §4 DoM signal -----
    Claim("prefill-dom-auroc", "Prefill L19 DoM AUROC (OOF 5-fold) = 0.7731",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["prefill_DoM_auroc_oof"], 0.7731, 0.002, "1024tok"),
    Claim("finaltok-dom-auroc", "Final-token L19 DoM AUROC = 0.7186",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["final_token_DoM_auroc_oof"], 0.7186, 0.002, "1024tok"),

    # ----- §7 Selective prediction / refuse-and-spend -----
    Claim("refuse-prefill-acc", "Prefill refuse@0.5 answered acc = 0.716",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["headline", "refuse_and_spend_coverage_0.5", "prefill", "acc_on_answered"],
          0.716, 0.005, "1024tok"),
    Claim("refuse-random-acc", "Random refuse@0.5 answered acc = 0.494",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["headline", "refuse_and_spend_coverage_0.5", "random", "acc_on_answered"],
          0.494, 0.010, "1024tok"),
    Claim("middle-heavy-acc", "Prefill middle-heavy K=4.5 acc = 0.526",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["headline", "prefill_middle_heavy", "overall_accuracy"],
          0.526, 0.005, "1024tok"),

    # ----- §9 Prefill inversion (7B) -----
    Claim("inv-7B-ratio", "7B prefill PR ratio correct/incorrect = 1.377",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "7B_prefill_PR_correct_over_incorrect"],
          1.377113639689133, 1e-5, "1024tok"),
    Claim("inv-15B-ratio", "1.5B prefill PR ratio = 0.946",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "1.5B_prefill_PR_correct_over_incorrect"],
          0.9456399969402085, 1e-5, "1024tok"),
    Claim("inv-7B-final-ratio", "7B final-token PR ratio = 0.384",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "7B_final_token_PR_correct_over_incorrect"],
          0.38397476790580454, 1e-5, "1024tok"),
    Claim("inv-15B-final-ratio", "1.5B final-token PR ratio = 0.509",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "1.5B_final_token_PR_correct_over_incorrect"],
          0.5086845084799126, 1e-5, "1024tok"),
    Claim("inv-7B-ci-lo", "7B prefill ratio 95% CI lo = 1.226",
          "pathway11_h100/prefill_inversion/results.json",
          ["bootstrap_95_CI_on_ratio", "7B_prefill", 0], 1.2254645586938757, 1e-5, "1024tok"),
    Claim("inv-7B-ci-hi", "7B prefill ratio 95% CI hi = 1.757",
          "pathway11_h100/prefill_inversion/results.json",
          ["bootstrap_95_CI_on_ratio", "7B_prefill", 1], 1.7569904867197708, 1e-5, "1024tok"),
    Claim("inv-7B-balance-ratio", "7B prefill balance-controlled ratio = 1.227",
          "pathway11_h100/prefill_inversion/results.json",
          ["balance_controlled_ratios", "7B_prefill"], 1.2274623411828771, 1e-5, "1024tok"),

    # Phase 1 bootstrap PR values (7B prefill, used in §3 aggregate table)
    Claim("7B-prefill-PR-correct", "7B prefill PR correct = 26.87",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "point", "pr_correct"], 26.872772216796875, 0.01, "1024tok"),
    Claim("7B-prefill-PR-incorrect", "7B prefill PR incorrect = 19.51",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "point", "pr_incorrect"], 19.513837814331055, 0.01, "1024tok"),
    Claim("7B-final-PR-correct", "7B final-token PR correct = 2.63",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_final_token", "point", "pr_correct"], 2.629302501678467, 0.01, "1024tok"),
    Claim("7B-final-PR-incorrect", "7B final-token PR incorrect = 6.85",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_final_token", "point", "pr_incorrect"], 6.847591876983643, 0.01, "1024tok"),
    Claim("15B-prefill-PR-correct", "1.5B prefill PR correct = 19.41",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "point", "pr_correct"], 19.405458450317383, 0.01, "1024tok"),
    Claim("15B-prefill-PR-incorrect", "1.5B prefill PR incorrect = 20.52",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "point", "pr_incorrect"], 20.520978927612305, 0.01, "1024tok"),
    Claim("15B-final-PR-correct", "1.5B final-token PR correct = 4.33",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_final_token", "point", "pr_correct"], 4.329, 0.05, "1024tok"),
    Claim("15B-final-PR-incorrect", "1.5B final-token PR incorrect = 8.51",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_final_token", "point", "pr_incorrect"], 8.51, 0.05, "1024tok"),

    # Phase 1 counts
    Claim("7B-n-correct", "7B n_correct = 366",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "n_correct"], 366, 0, "1024tok"),
    Claim("7B-n-incorrect", "7B n_incorrect = 134",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "n_incorrect"], 134, 0, "1024tok"),
    Claim("15B-n-correct", "1.5B n_correct = 243",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "n_correct"], 243, 0, "1024tok"),
    Claim("15B-n-incorrect", "1.5B n_incorrect = 257",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "n_incorrect"], 257, 0, "1024tok"),

    # Three-way split
    Claim("three-way-both-right-n", "Both-right n=230",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "both_right", "n"], 230, 0, "1024tok"),
    Claim("three-way-only7b-n", "Only-7B-right n=136",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "only_7b", "n"], 136, 0, "1024tok"),
    Claim("three-way-only15b-n", "Only-1.5B-right n=13",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "only_15b", "n"], 13, 0, "1024tok"),
    Claim("three-way-both-wrong-n", "Both-wrong n=121",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "both_wrong", "n"], 121, 0, "1024tok"),

    # BBH prefill ratios (§9)
    Claim("bbh-tracking-ratio", "BBH tracking_shuffled PR prefill ratio = 0.930",
          "pathway11_h100/prefill_inversion/results.json",
          ["bbh", "tracking_shuffled_objects_seven_objects", "pr_prefill_ratio"],
          0.9304261877758804, 1e-4, "1024tok"),
    Claim("bbh-logical-ratio", "BBH logical_deduction PR prefill ratio = 0.824",
          "pathway11_h100/prefill_inversion/results.json",
          ["bbh", "logical_deduction_seven_objects", "pr_prefill_ratio"],
          0.8240621037129747, 1e-4, "1024tok"),
    Claim("bbh-web-ratio", "BBH web_of_lies PR prefill ratio = 1.009",
          "pathway11_h100/prefill_inversion/results.json",
          ["bbh", "web_of_lies", "pr_prefill_ratio"], 1.00919347747066, 1e-4, "1024tok"),

    # ----- §10 Multi-signal oracle (Exp 2b) -----
    Claim("exp2b-logreg-f1", "Exp 2b logreg macro-F1 = 0.548",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_logreg", "macro_f1"], 0.5477024403341905, 1e-4, "1024tok"),
    Claim("exp2b-rf-f1", "Exp 2b RF macro-F1 = 0.494",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_rf", "macro_f1"], 0.49397192549427976, 1e-4, "1024tok"),
    Claim("exp2b-logreg-acc", "Exp 2b logreg overall acc = 0.620",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_logreg", "overall_acc_oof"], 0.62, 0.005, "1024tok"),
    Claim("exp2b-rf-acc", "Exp 2b RF overall acc = 0.636",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_rf", "overall_acc_oof"], 0.636, 0.005, "1024tok"),
    Claim("exp2b-logreg-Drecall", "Logreg D-recall to K=1 = 0.417",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_logreg", "D_recall_to_K1"], 0.4166666666666667, 1e-4, "1024tok"),
    Claim("exp2b-rf-Drecall", "RF D-recall to K=1 = 0.444",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_rf", "D_recall_to_K1"], 0.4444444444444444, 1e-4, "1024tok"),
    Claim("exp2b-multi-vs-single", "Multi-signal vs best single = +0.4pp",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["multi_signal_minus_single_signal_pp"], 0.4, 0.05, "1024tok"),
    Claim("exp2b-2pp-met", "2pp bar met = False",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["user_threshold_2pp_met"], False, 0, "1024tok"),
    Claim("exp2b-best-single", "Best single-signal compute 2.74, acc 0.516",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["best_single_signal_at_matched_compute", "overall_accuracy"], 0.516, 0.003, "1024tok"),

    # ----- §3 Gibberish control -----
    Claim("gib-math-pos50", "MATH-500 PR pos 50 = 30.55",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "math500_baseline", "pr_by_position", "50", "pr"],
          30.554419380797377, 0.01, "1024tok"),
    Claim("gib-math-pos1", "MATH-500 PR pos 1 = 13.74",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "math500_baseline", "pr_by_position", "1", "pr"],
          13.73524962240093, 0.01, "1024tok"),
    Claim("gib-math-final", "MATH-500 PR final = 22.19",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "math500_baseline", "pr_by_position", "final", "pr"],
          22.186663572792018, 0.01, "1024tok"),
    Claim("gib-random-pos1", "Random PR pos 1 = 11.32",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "random_tokens", "pr_by_position", "1", "pr"],
          11.315935166610068, 0.01, "1024tok"),
    Claim("gib-random-final", "Random PR final = 8.96",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "random_tokens", "pr_by_position", "final", "pr"],
          8.963097360506318, 0.01, "1024tok"),
    Claim("gib-stream-pos10", "Stream-of-consc PR pos 10 = 15.66",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "stream_of_consciousness", "pr_by_position", "10", "pr"],
          15.662493885959867, 0.01, "1024tok"),
    Claim("gib-stream-pos1", "Stream-of-consc PR pos 1 = 4.60",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "stream_of_consciousness", "pr_by_position", "1", "pr"],
          4.5966970751104155, 0.01, "1024tok"),

    # ----- No-CoT control -----
    Claim("nocot-pos1", "No-CoT PR pos 1 = 15.78",
          "pathway11_h100/no_cot_control/pr_curves.json",
          ["conditions", "nocot", "pr_by_position", "1", "pr"],
          15.78, 0.1, "1024tok"),
    Claim("nocot-final", "No-CoT PR final = 19.80",
          "pathway11_h100/no_cot_control/pr_curves.json",
          ["conditions", "nocot", "pr_by_position", "final", "pr"],
          19.80, 0.1, "1024tok"),

    # ----- Exp 1 cross-model -----
    Claim("phi3-acc", "Phi-3-mini MATH-500 acc = 0.448",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 0, "accuracy"], 0.448, 0.002, "1024tok"),
    Claim("phi3-n-correct", "Phi-3-mini n_correct = 224",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 0, "n_correct"], 224, 0, "1024tok"),
    Claim("llama-acc", "Llama-3.2-1B MATH-500 acc = 0.252",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 1, "accuracy"], 0.252, 0.002, "1024tok"),
    Claim("llama-n-correct", "Llama-3.2-1B n_correct = 126",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 1, "n_correct"], 126, 0, "1024tok"),
    Claim("phi3-2thirds-layer", "Phi-3 2/3-depth layer = 21",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 0, "twothirds_layer_idx"], 21, 0, "1024tok"),
    Claim("llama-2thirds-layer", "Llama 2/3-depth layer = 11",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 1, "twothirds_layer_idx"], 11, 0, "1024tok"),

    # ----- Stage 5 BBH L19 DoM transfer -----
    Claim("stage5-math-to-bbh-pooled", "MATH→BBH pooled AUROC = 0.747",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["math_to_bbh", "_pooled", "auroc"], 0.7471963025177446, 1e-4, "1024tok"),
    Claim("stage5-bbh-to-math", "BBH→MATH AUROC = 0.693",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["bbh_to_math", "auroc"], 0.6926230164448928, 1e-4, "1024tok"),
    Claim("stage5-symmetric", "Symmetric avg = 0.720",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["symmetric_avg"], 0.7199096594813187, 1e-4, "1024tok"),
    Claim("stage5-verdict-vs-coe", "Delta vs CoE symmetric = +0.004",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["verdict_vs_coe"], 0.003909659481318717, 1e-4, "1024tok"),
    Claim("stage5-within-math", "MATH within-benchmark AUROC = 0.731",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["within_benchmark_overfit", "math", "auroc"], 0.7305727690509358, 1e-4, "1024tok"),

    # ----- Pathway 9 claims (256-tok labels, many superseded) -----
    Claim("p9-abc44-holdout-raw", "ABC-44 holdout AUROC (raw) = 0.7961",
          "pathway9/results/exp1_raw_length_v2.json",
          ["baseline_auroc"], 0.7960784313725491, 1e-4, "256tok_NEW",
          note="SUPERSEDED: computed against 104/500 NEW labels; 1024-tok re-run gives prefill+final DoM ≈0.794"),
    Claim("p9-abc44-holdout-deconfounded", "ABC-44 deconfounded (raw length) AUROC = 0.7459",
          "pathway9/results/exp1_raw_length_v2.json",
          ["deconfounded_auroc"], 0.7458823529411764, 1e-4, "256tok_NEW",
          note="SUPERSEDED: length-confound was feature-engineering residue; raw residual-stream DoM is not length-confounded (cos=-0.09)"),
    Claim("p9-length-drop", "Drop from raw-length deconfounding = 0.050",
          "pathway9/results/exp1_raw_length_v2.json",
          ["drop_from_deconfounding"], 0.050, 0.001, "256tok_NEW"),
    Claim("p9-gaussian-null-real", "Real PH 5-feat AUROC = 0.690",
          "pathway9/results/exp3a_gaussian_null_v2.json",
          ["classifiers", "real_ph", "auroc_holdout"], 0.6901960784313725, 1e-4, "256tok_NEW"),
    Claim("p9-gaussian-null-null", "Null (empirical-cov) PH AUROC = 0.693",
          "pathway9/results/exp3a_gaussian_null_v2.json",
          ["classifiers", "null_ph", "auroc_holdout"], 0.6925490196078431, 1e-4, "256tok_NEW"),
    Claim("p9-lr-vs-xgb-lift", "XGB-LR lift CV5 = -0.005",
          "pathway9/results/exp4_xgboost_oinfo_v2.json",
          ["xgboost_lift_cv5"], -0.005228758169934511, 1e-4, "256tok_NEW"),
    Claim("p9-coe-math-to-bbh", "CoE MATH→BBH = 0.720",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "coe", "MATH_to_BBH_auroc"], 0.7204625669180751, 1e-4, "256tok_manifest"),
    Claim("p9-coe-bbh-to-math", "CoE BBH→MATH = 0.712",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "coe", "BBH_to_MATH_auroc"], 0.7122901881266194, 1e-4, "256tok_manifest"),
    Claim("p9-ph-math-to-bbh", "PH-168 MATH→BBH = 0.354",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "layerwise_ph", "MATH_to_BBH_auroc"], 0.3539202363906647, 1e-4, "256tok_manifest"),
    Claim("p9-ph-bbh-to-math", "PH-168 BBH→MATH = 0.497",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "layerwise_ph", "BBH_to_MATH_auroc"], 0.4967572699914708, 1e-4, "256tok_manifest"),
    Claim("p9-d2h-math-to-bbh", "D2H-lite MATH→BBH = 0.740",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "d2h_lite", "MATH_to_BBH_auroc"], 0.7403350691752775, 1e-4, "256tok_manifest"),
    Claim("p9-d2h-bbh-to-math", "D2H-lite BBH→MATH = 0.529",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "d2h_lite", "BBH_to_MATH_auroc"], 0.5287339673956775, 1e-4, "256tok_manifest"),

    # ----- Pathway 8 layer-wise results (256-tok NEW labels) -----
    # Model array index: 0=ABC-44, 1=LayerwisePH-168, 2=CoE-60, 3=D2H-lite, 4=D2H-full, 5=PH+CoE, 6=PH+D2H, 7=All
    Claim("p8-abc44-auroc", "Pathway 8 ABC-44 holdout = 0.796",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 0, "auroc_holdout"], 0.7961, 1e-3, "256tok_NEW"),
    Claim("p8-coe-auroc", "Pathway 8 CoE-60 holdout = 0.811",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 2, "auroc_holdout"], 0.811, 1e-3, "256tok_NEW"),
    Claim("p8-d2h-lite-auroc", "Pathway 8 D2H-lite holdout = 0.806",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 3, "auroc_holdout"], 0.8055, 1e-3, "256tok_NEW"),
    Claim("p8-layerwise-ph", "Pathway 8 layer-wise PH = 0.646",
          "pathway8_layerwise/results/exp1_results.json",
          ["auroc_holdout"], 0.6463, 1e-3, "256tok_NEW"),
    Claim("p8-ph-coe-ensemble", "Pathway 8 PH+CoE combined = 0.758",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 5, "auroc_holdout"], 0.7584, 1e-3, "256tok_NEW"),

    # ----- Scratch pathway10 old-labels numbers (for §4 "superseded" provenance) -----
    Claim("scratch-rc-auroc", "Pathway10 rc 5fold prefill AUROC (old labels) = 0.7525",
          "scratch/pathway10_rc_5fold_results.json",
          ["methods", "1.5B_L19_DoM", "auroc"], 0.7524766899766899, 1e-4, "256tok_NEW"),
    Claim("scratch-rc-aurc", "Pathway10 rc 5fold AURC = 0.6346",
          "scratch/pathway10_rc_5fold_results.json",
          ["methods", "1.5B_L19_DoM", "aurc"], 0.6346043846421335, 1e-4, "256tok_NEW"),
    Claim("scratch-temporal-prefill", "Pathway10 temporal prefill AUROC (old) = 0.3921",
          "scratch/pathway10_temporal_and_verifier_results.json",
          ["T1_temporal_dom", "positions", 0, "dom_holdout_auroc"], 0.39210367691380343, 1e-4, "256tok_NEW",
          note="SUPERSEDED: reversal was label-distribution artifact at 104/500 correct"),
    Claim("scratch-temporal-cos-prefill-final", "Old prefill vs final DoM cosine = 0.046",
          "scratch/pathway10_temporal_and_verifier_results.json",
          ["T1_temporal_dom", "positions", 0, "cos_with_final_token_dom"], 0.046196311712265015, 1e-4, "256tok_NEW"),
]


# --- Driver ------------------------------------------------------------------

def _load(path: str) -> Any:
    return json.load(open(ROOT / path))


def _check(c: Claim) -> None:
    full = ROOT / c.file
    if not full.exists():
        c.result = "MISSING_FILE"
        return
    try:
        data = _load(c.file)
    except Exception as exc:  # noqa: BLE001
        c.result = f"LOAD_ERROR:{exc}"
        return
    try:
        val = _drill(data, c.path)
    except (KeyError, IndexError, TypeError) as exc:
        c.result = f"KEY_MISSING:{exc}"
        return
    c.actual = val
    exp = c.expected
    if isinstance(exp, bool) or isinstance(val, bool):
        c.result = "PASS" if bool(val) == bool(exp) else "FAIL"
    elif isinstance(exp, (int, float)) and isinstance(val, (int, float)):
        if math.isclose(val, exp, abs_tol=c.tol):
            c.result = "PASS"
        else:
            c.result = f"FAIL(Δ={val - exp:+.5g})"
    elif exp == val:
        c.result = "PASS"
    else:
        c.result = f"FAIL(got={val!r})"


def main() -> int:
    for c in CLAIMS:
        _check(c)

    # Summary counts
    pass_n = sum(1 for c in CLAIMS if c.result == "PASS")
    fail_n = sum(1 for c in CLAIMS if c.result.startswith("FAIL"))
    missing = sum(1 for c in CLAIMS if "MISSING" in c.result or "ERROR" in c.result)
    tok256 = sum(1 for c in CLAIMS if c.labels.startswith("256tok"))

    print(f"# validate_claims.py — {len(CLAIMS)} claims")
    print(f"PASS:   {pass_n}")
    print(f"FAIL:   {fail_n}")
    print(f"MISSING/ERROR: {missing}")
    print(f"(of which: {tok256} use the superseded 256-tok label scheme and should be re-baselined against 1024-tok if cited as current)\n")

    hdr = ("ID", "labels", "result", "expected", "actual", "description", "file")
    widths = (28, 18, 22, 14, 16, 56, 70)
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))

    for c in CLAIMS:
        actual = c.actual
        if isinstance(actual, float):
            actual_s = f"{actual:.6g}"
        else:
            actual_s = str(actual)[:16]
        exp = c.expected
        if isinstance(exp, float):
            exp_s = f"{exp:.6g}"
        else:
            exp_s = str(exp)[:14]
        desc = c.description[:56]
        path_s = c.file[:70]
        row = (c.cid[:28], c.labels[:18], c.result[:22], exp_s[:14],
               actual_s[:16], desc, path_s)
        print("  ".join(r.ljust(w) for r, w in zip(row, widths)))

    # Detail section for failures and notes
    had_notes = [c for c in CLAIMS if c.note]
    any_issue = [c for c in CLAIMS if c.result != "PASS"]

    if any_issue:
        print("\n## Issues requiring attention\n")
        for c in any_issue:
            print(f"- [{c.cid}] {c.result}: {c.description}")
            print(f"    file: {c.file}")
            print(f"    path: {c.path}")
            print(f"    expected: {c.expected!r}  actual: {c.actual!r}")
            if c.note:
                print(f"    note: {c.note}")

    if had_notes:
        print("\n## Claims with superseded/outdated notes\n")
        for c in had_notes:
            print(f"- [{c.cid}] labels={c.labels}: {c.note}")

    return 0 if fail_n == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
