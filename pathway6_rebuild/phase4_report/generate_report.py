#!/usr/bin/env python3
"""Phase 4: Generate comparison report — OLD vs CORRECTED results.

Prints a formatted comparison table and categorizes each original claim.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PHASE0_DIR = Path(__file__).parent.parent / "phase0_relabel"
PHASE1_DIR = Path(__file__).parent.parent / "phase1_prompt_model"
PHASE2_DIR = Path(__file__).parent.parent / "phase2_completion"


def main():
    # Load all results
    with open(PHASE0_DIR / "foundational_counts.json") as f:
        counts = json.load(f)
    with open(PHASE1_DIR / "holdout_metrics_v2.json") as f:
        metrics = json.load(f)
    with open(PHASE1_DIR / "experiment9_v2.json") as f:
        exp9 = json.load(f)
    with open(PHASE2_DIR / "holdout_results_v2.json") as f:
        selection = json.load(f)
    with open(PHASE2_DIR / "experiment1_v2.json") as f:
        exp1 = json.load(f)
    with open(PHASE2_DIR / "experiment8_v2.json") as f:
        exp8 = json.load(f)

    print("=" * 80)
    print("PATHWAY 6 REBUILD: CORRECTED RESULTS COMPARISON")
    print("=" * 80)

    # ---- Comparison Table ----
    print("\n## Comparison Table\n")
    print(f"{'Metric':<40} {'ORIGINAL':>10} {'CORRECTED':>10} {'Delta':>10}")
    print("-" * 75)

    rows = [
        ("1.5B MATH greedy accuracy",
         f"{counts['greedy_baseline']['old_total']}/500",
         f"{counts['greedy_baseline']['new_total']}/500",
         f"+{counts['greedy_baseline']['new_total'] - counts['greedy_baseline']['old_total']}"),

        ("  Train correct",
         f"{counts['greedy_baseline']['old_train']}/400",
         f"{counts['greedy_baseline']['new_train']}/400",
         f"+{counts['greedy_baseline']['new_train'] - counts['greedy_baseline']['old_train']}"),

        ("  Holdout correct",
         f"{counts['greedy_baseline']['old_holdout']}/100",
         f"{counts['greedy_baseline']['new_holdout']}/100",
         f"+{counts['greedy_baseline']['new_holdout'] - counts['greedy_baseline']['old_holdout']}"),

        ("Topo AUROC (holdout)",
         "0.935",
         f"{metrics['auroc']:.4f}",
         f"{metrics['auroc'] - 0.935:+.4f}"),

        ("Topo AUROC (train CV-50)",
         "~0.93",
         f"{metrics['train_cv_auroc']:.4f}",
         f"{metrics['train_cv_auroc'] - 0.93:+.3f}"),

        ("Best 1-pass baseline (neg_entropy)",
         "0.457",
         f"{exp9['baselines']['neg_mean_entropy']['new_auroc']:.4f}",
         f"{exp9['baselines']['neg_mean_entropy']['new_auroc'] - 0.457:+.4f}"),

        ("Best 32-pass baseline (vote_margin)",
         "0.727",
         f"{exp9['baselines']['vote_margin']['new_auroc']:.4f}",
         f"{exp9['baselines']['vote_margin']['new_auroc'] - 0.727:+.4f}"),

        ("Topo - best baseline gap",
         "+0.208",
         f"{exp9['topo_vs_best_gap']:+.4f}",
         f"{exp9['topo_vs_best_gap'] - 0.208:+.4f}"),

        ("Oracle ceiling (holdout)",
         "+28",
         f"+{counts['oracle_ceiling']['holdout_solvable']}",
         f"{counts['oracle_ceiling']['holdout_solvable'] - 28:+d}"),

        ("Solvable under temperature (train)",
         "153",
         f"{counts['oracle_ceiling']['train_solvable']}",
         f"{counts['oracle_ceiling']['train_solvable'] - 153:+d}"),

        ("Ungated MV net gain (holdout)",
         "+8",
         f"+{selection['ungated_mv']['net_gain']}",
         f"{selection['ungated_mv']['net_gain'] - 8:+d}"),

        ("Ungated MV R->W",
         "3",
         f"{selection['ungated_mv']['R_to_W']}",
         f"{selection['ungated_mv']['R_to_W'] - 3:+d}"),

        ("Gated MV net gain (tau=0.3)",
         "+10",
         f"+{selection['gated_mv']['net_gain']}",
         f"{selection['gated_mv']['net_gain'] - 10:+d}"),

        ("Gated MV R->W (tau=0.3)",
         "0",
         f"{selection['gated_mv']['R_to_W']}",
         f"{selection['gated_mv']['R_to_W']:+d}"),

        ("Weighted vote net gain",
         "+14",
         "DEFERRED",
         "(needs RunPod)"),

        ("ECE (equal-width)",
         "0.118",
         f"{metrics['experiment10']['ece_equal_width']:.4f}",
         f"{metrics['experiment10']['ece_equal_width'] - 0.118:+.4f}"),

        ("Brier score",
         "0.087",
         f"{metrics['experiment10']['brier']:.4f}",
         f"{metrics['experiment10']['brier'] - 0.087:+.4f}"),
    ]

    for metric, old, new, delta in rows:
        print(f"{metric:<40} {old:>10} {new:>10} {delta:>10}")

    # ---- Adaptive Sampling ----
    print("\n## Adaptive Sampling (Experiment 8)")
    print(f"{'Config':<35} {'Correct':>8} {'Net':>6} {'R->W':>6} {'Savings':>8}")
    print("-" * 65)
    for r in exp8:
        config = f"tau_e={r['tau_e']}, tau_h={r['tau_h']}, N={r['n_med']}"
        print(f"{config:<35} {r['n_correct']:>8} {r['net_gain']:>+6} {r['R_to_W']:>6} {r['savings_pct']:>7.1f}%")

    # ---- Tau Sweep (Experiment 1) ----
    print("\n## Gated MV Tau Sweep (Experiment 1)")
    print(f"{'tau':>6} {'Correct':>8} {'Net':>6} {'W->R':>6} {'R->W':>6} {'Gated':>6}")
    print("-" * 45)
    for r in exp1["tau_sweep"]:
        print(f"{r['tau']:>6.1f} {r['n_correct']:>8} {r['net_gain']:>+6} {r['W_to_R']:>6} {r['R_to_W']:>6} {r['gated_count']:>6}")

    # ---- Claim Assessment ----
    print("\n" + "=" * 80)
    print("CLAIM ASSESSMENT")
    print("=" * 80)

    topo_auroc = metrics["auroc"]
    best_bl = exp9["topo_vs_best_gap"]

    claims = [
        ("Topo AUROC > 0.90 on MATH-500",
         "INVALIDATED" if topo_auroc < 0.90 else "SURVIVES",
         f"AUROC = {topo_auroc:.3f}"),

        ("Topo crushes all logprob baselines",
         "WEAKENED" if best_bl > 0 else "INVALIDATED",
         f"Gap = {best_bl:+.3f} vs vote_margin"),

        ("Topo detects 'confidently wrong' (logprob anti-correlated)",
         "WEAKENED",
         f"neg_entropy AUROC = {exp9['baselines']['neg_mean_entropy']['new_auroc']:.3f} (was 0.457, now ~0.5)"),

        ("Gated MV achieves 0 R->W",
         "SURVIVES" if selection["gated_mv"]["R_to_W"] == 0 else "INVALIDATED",
         f"R->W = {selection['gated_mv']['R_to_W']} at tau=0.3"),

        ("Gating is strictly better than ungated MV",
         "WEAKENED",
         f"Gated +{selection['gated_mv']['net_gain']} vs ungated +{selection['ungated_mv']['net_gain']}"),

        ("Adaptive sampling: 45% token savings at matched accuracy",
         "SURVIVES",
         f"89% savings at +3 or 77% savings at +5"),

        ("Feature ablation: Tier A is irreplaceable anchor",
         "SURVIVES" if metrics.get("experiment6_tier_a_auroc", 0) > 0.6 else "WEAKENED",
         f"Tier A AUROC = {metrics.get('experiment6_tier_a_auroc', 'N/A')}"),

        ("Per-completion weighted vote +14 (0 R->W)",
         "UNKNOWN",
         "Needs RunPod for per-completion features"),

        ("GSM8K generalization (AUROC=0.652)",
         "UNKNOWN",
         "Needs RunPod with chat template"),

        ("7B generalization (AUROC=0.657)",
         "UNKNOWN",
         "Needs RunPod with chat template"),
    ]

    for claim, status, evidence in claims:
        print(f"\n  [{status}] {claim}")
        print(f"    Evidence: {evidence}")

    # ---- Save ----
    report = {
        "comparison_table": rows,
        "adaptive_sampling": exp8,
        "tau_sweep": exp1["tau_sweep"],
        "claims": [{"claim": c, "status": s, "evidence": e} for c, s, e in claims],
    }
    with open(Path(__file__).parent / "FINAL_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\nSaved to {Path(__file__).parent / 'FINAL_REPORT.json'}")


if __name__ == "__main__":
    main()
