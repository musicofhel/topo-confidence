#!/usr/bin/env python3
"""Phase 6.5 Step 4: Generate FINAL_SUMMARY.md.

CPU ONLY. Run after step3_cross_analysis.py.

Produces the definitive results summary for the topo-confidence project:
- Complete corrected results table (all configs)
- Whether truncation was the sole cause of low accuracy
- Deconfounded AUROC numbers
- Whether topo-confidence generalizes beyond MATH-500
- Updated claim assessment
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

P65_DIR = Path(__file__).parent
GSM8K_DIR = P65_DIR / "gsm8k"
MATH7B_DIR = P65_DIR / "math7b"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    # Load all results
    gsm8k_summary = load_json(GSM8K_DIR / "summary.json")
    math7b_summary = load_json(MATH7B_DIR / "summary.json")
    gsm8k_model = load_json(GSM8K_DIR / "model_results.json")
    math7b_model = load_json(MATH7B_DIR / "model_results.json")
    gsm8k_selection = load_json(GSM8K_DIR / "selection_results.json")
    math7b_selection = load_json(MATH7B_DIR / "selection_results.json")
    cross = load_json(P65_DIR / "cross_analysis.json")

    # Phase 3 (truncated) for comparison
    p3_dir = P65_DIR.parent / "phase3_cross_benchmark"
    gsm8k_p3 = load_json(p3_dir / "gsm8k" / "summary.json")
    math7b_p3 = load_json(p3_dir / "math7b" / "summary.json")

    lines = []
    lines.append("# Pathway 6.5: Deconfounded Cross-Benchmark Results")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # --- Section 1: What Changed ---
    lines.append("## 1. What Changed")
    lines.append("")
    lines.append("The e2e audit (Phase 6 audit) found that `max_new_tokens=256` truncated")
    lines.append("64% of GSM8K outputs and 90% of 7B MATH outputs. This caused:")
    lines.append("- GSM8K accuracy: 36.7% (expected ~73%)")
    lines.append("- 7B MATH accuracy: 16.2% (expected ~75-77%)")
    lines.append("- AUROC confounded by truncation (binary 'not truncated' predictor beat topo)")
    lines.append("")
    lines.append("Phase 6.5 reruns both configs with `max_new_tokens=1024`.")
    lines.append("The 1.5B x MATH-500 result (AUROC 0.796) is unchanged (20.8% accuracy, no truncation).")
    lines.append("")

    # --- Section 2: Results Table ---
    lines.append("## 2. Complete Results Table")
    lines.append("")
    lines.append("| Config | max_tokens | Greedy Acc | Topo AUROC | Best Baseline | Gap | MV | Oracle |")
    lines.append("|--------|-----------|-----------|------------|---------------|-----|-----|--------|")

    # 1.5B MATH (unchanged)
    lines.append(
        f"| 1.5B x MATH-500 | 256 | 104/500 (20.8%) | **0.796** | vote_margin 0.767 | +0.057 | +12 | +30 |"
    )

    # GSM8K Phase 3
    if gsm8k_p3:
        lines.append(
            f"| 1.5B x GSM8K (truncated) | 256 | {gsm8k_p3['greedy_correct']}/{gsm8k_p3['n_problems']} "
            f"({gsm8k_p3['greedy_accuracy']:.1%}) | {gsm8k_p3['topo_auroc']:.3f} | "
            f"{gsm8k_p3.get('best_baseline', '?')} {gsm8k_p3.get('best_baseline_auroc', 0):.3f} | "
            f"+{gsm8k_p3['topo_auroc'] - gsm8k_p3.get('best_baseline_auroc', 0):.3f} | "
            f"+{gsm8k_p3.get('mv_net_gain', '?')} | +{gsm8k_p3.get('oracle_ceiling', '?')} |"
        )

    # GSM8K Phase 6.5
    if gsm8k_summary:
        bl = gsm8k_model or {}
        sel = gsm8k_selection or {}
        mv = sel.get("ungated_mv", {})
        lines.append(
            f"| **1.5B x GSM8K (fixed)** | 1024 | {gsm8k_summary['greedy_correct']}/{gsm8k_summary['n_problems']} "
            f"({gsm8k_summary['greedy_accuracy']:.1%}) | **{gsm8k_summary['topo_auroc']:.3f}** | "
            f"{bl.get('best_baseline', '?')} {bl.get('best_baseline_auroc', 0):.3f} | "
            f"{bl.get('topo_vs_best_gap', 0):+.3f} | "
            f"+{mv.get('net_gain', '?')} | +{sel.get('oracle_ceiling', '?')} |"
        )

    # 7B Phase 3
    if math7b_p3:
        lines.append(
            f"| 7B x MATH-500 (truncated) | 256 | {math7b_p3['greedy_correct']}/{math7b_p3['n_problems']} "
            f"({math7b_p3['greedy_accuracy']:.1%}) | {math7b_p3['topo_auroc']:.3f} | "
            f"{math7b_p3.get('best_baseline', '?')} {math7b_p3.get('best_baseline_auroc', 0):.3f} | "
            f"+{math7b_p3['topo_auroc'] - math7b_p3.get('best_baseline_auroc', 0):.3f} | "
            f"+{math7b_p3.get('mv_net_gain', '?')} | +{math7b_p3.get('oracle_ceiling', '?')} |"
        )

    # 7B Phase 6.5
    if math7b_summary:
        bl = math7b_model or {}
        sel = math7b_selection or {}
        mv = sel.get("ungated_mv", {})
        lines.append(
            f"| **7B x MATH-500 (fixed)** | 1024 | {math7b_summary['greedy_correct']}/{math7b_summary['n_problems']} "
            f"({math7b_summary['greedy_accuracy']:.1%}) | **{math7b_summary['topo_auroc']:.3f}** | "
            f"{bl.get('best_baseline', '?')} {bl.get('best_baseline_auroc', 0):.3f} | "
            f"{bl.get('topo_vs_best_gap', 0):+.3f} | "
            f"+{mv.get('net_gain', '?')} | +{sel.get('oracle_ceiling', '?')} |"
        )

    lines.append("")

    # Truncation check
    if gsm8k_summary:
        lines.append(f"GSM8K truncation at 1024: {gsm8k_summary.get('n_truncated', '?')}/{gsm8k_summary['n_problems']} "
                      f"({gsm8k_summary.get('truncation_pct', '?'):.1f}%)")
    if math7b_summary:
        lines.append(f"7B MATH truncation at 1024: {math7b_summary.get('n_truncated', '?')}/{math7b_summary['n_problems']} "
                      f"({math7b_summary.get('truncation_pct', '?'):.1f}%)")
    lines.append("")

    # --- Section 3: Was Truncation the Sole Cause? ---
    lines.append("## 3. Was Truncation the Sole Cause of Low Accuracy?")
    lines.append("")
    if gsm8k_summary and math7b_summary:
        gsm_ok = gsm8k_summary["greedy_accuracy"] > 0.70
        m7b_ok = math7b_summary["greedy_accuracy"] > 0.70
        if gsm_ok and m7b_ok:
            lines.append("**YES.** Both configs now match expected model capabilities:")
            lines.append(f"- GSM8K: {gsm8k_summary['greedy_accuracy']:.1%} (expected ~73%)")
            lines.append(f"- 7B MATH: {math7b_summary['greedy_accuracy']:.1%} (expected ~75-77%)")
        elif gsm_ok or m7b_ok:
            lines.append("**PARTIALLY.** One config matches, the other needs investigation.")
        else:
            lines.append("**NO.** Both configs still below expected — truncation was not the only issue.")
    else:
        lines.append("Results not yet available.")
    lines.append("")

    # --- Section 4: Deconfounded AUROCs ---
    lines.append("## 4. Deconfounded AUROC Numbers")
    lines.append("")
    lines.append("| Config | Phase 3 (confounded) | Phase 6.5 (clean) | Change |")
    lines.append("|--------|---------------------|-------------------|--------|")
    if gsm8k_summary and gsm8k_p3:
        old = gsm8k_p3["topo_auroc"]
        new = gsm8k_summary["topo_auroc"]
        lines.append(f"| GSM8K AUROC | {old:.3f} | {new:.3f} | {new - old:+.3f} |")
    if math7b_summary and math7b_p3:
        old = math7b_p3["topo_auroc"]
        new = math7b_summary["topo_auroc"]
        lines.append(f"| 7B MATH AUROC | {old:.3f} | {new:.3f} | {new - old:+.3f} |")
    lines.append("")

    if gsm8k_summary:
        ci = gsm8k_summary.get("topo_auroc_ci", [0, 1])
        lines.append(f"GSM8K 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    if math7b_summary:
        ci = math7b_summary.get("topo_auroc_ci", [0, 1])
        lines.append(f"7B MATH 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    lines.append("")

    # --- Section 5: Does Topo Generalize? ---
    lines.append("## 5. Does Topo-Confidence Generalize Beyond MATH-500?")
    lines.append("")
    if cross and "transfer" in cross:
        lines.append("### Transfer AUROC (train on X, evaluate on Y)")
        lines.append("")
        lines.append("| Source → Target | AUROC | CI |")
        lines.append("|----------------|-------|-----|")
        for key, val in cross["transfer"].items():
            auroc = f"{val['auroc']:.3f}" if val.get("auroc") else "N/A"
            ci = f"[{val['ci'][0]:.3f}, {val['ci'][1]:.3f}]" if val.get("ci") else ""
            lines.append(f"| {key.replace('_to_', ' → ')} | {auroc} | {ci} |")
        lines.append("")

    if cross and "feature_rank_correlations" in cross:
        lines.append("### Feature Effect-Size Correlations")
        lines.append("")
        for key, val in cross["feature_rank_correlations"].items():
            lines.append(f"- {key.replace('_vs_', ' vs ')}: rho={val['rho']:.3f} (p={val['p']:.2e})")
        lines.append("")

    # --- Section 6: Claim Assessment ---
    lines.append("## 6. Updated Claim Assessment")
    lines.append("")
    if cross and "claims" in cross:
        for c in cross["claims"]:
            lines.append(f"- **[{c['status']}]** {c['claim']}")
            lines.append(f"  - {c['evidence']}")
        lines.append("")

    # --- Section 7: Cost and Runtime ---
    lines.append("## 7. Cost and Runtime")
    lines.append("")
    total_min = 0
    if gsm8k_summary:
        total_min += gsm8k_summary.get("elapsed_minutes", 0)
        lines.append(f"- GSM8K pipeline: {gsm8k_summary.get('elapsed_minutes', 0):.0f} min")
    if math7b_summary:
        total_min += math7b_summary.get("elapsed_minutes", 0)
        lines.append(f"- 7B MATH pipeline: {math7b_summary.get('elapsed_minutes', 0):.0f} min")
    if total_min > 0:
        cost = (total_min / 60) * 2.99
        lines.append(f"- Total: {total_min:.0f} min ({total_min / 60:.1f} hours)")
        lines.append(f"- Estimated cost: ~${cost:.0f} (H100 SXM at $2.99/hr)")
    lines.append("")

    # --- Write ---
    text = "\n".join(lines)
    with open(P65_DIR / "FINAL_SUMMARY.md", "w") as f:
        f.write(text)

    print(text)
    print(f"\nSaved to {P65_DIR / 'FINAL_SUMMARY.md'}")


if __name__ == "__main__":
    main()
