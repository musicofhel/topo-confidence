#!/usr/bin/env python3
"""Exp 2b / Phase 5: consolidated results.json + Pareto plot + SUMMARY.md write assist."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/multi_signal_oracle"
EXP2 = ROOT / "pathway11_h100/prefill_gated_compute"


def main():
    clf = json.loads((OUT_DIR / "phase2_classifier.json").read_text())
    pol = json.loads((OUT_DIR / "phase3_policy.json").read_text())
    abl = json.loads((OUT_DIR / "phase4_ablation.json").read_text())
    exp2 = json.loads((EXP2 / "phase3b_realistic_policies.json").read_text())

    # Gather points for Pareto plot
    points = []
    # Exp 2 full-coverage policies
    for key in ["uniform_K1", "uniform_K2", "uniform_K4", "uniform_K8"]:
        v = exp2[key]
        points.append(("uniform", key.replace("uniform_", ""),
                       v["compute_per_problem"], v["overall_accuracy"]))
    for key in ["threshold_prefill_sweep", "threshold_neg_seq_len_sweep",
                "threshold_final_token_sweep"]:
        for entry in exp2[key]:
            name = key.replace("threshold_", "").replace("_sweep", "")
            points.append((f"threshold_{name}", str(entry["tau_percentile"]),
                           entry["compute_per_problem"], entry["overall_accuracy"]))
    for key, label in [("quartile_prefill_top_heavy", "prefill_top_heavy"),
                       ("quartile_neg_seq_len_top_heavy", "seq_len_top_heavy")]:
        v = exp2[key]
        points.append(("quartile", label,
                       v["compute_per_problem"], v["overall_accuracy"]))

    # Multi-signal classifier (with K=1 fallback)
    for clf_kind in ["logreg", "rf"]:
        sim = pol[clf_kind]["simulation_k1_fallback"]
        points.append(("multi_signal", f"{clf_kind}_k1fb",
                       sim["compute_per_problem"], sim["overall_accuracy"]))
        sim_nofb = pol[clf_kind]["simulation_no_fallback"]
        # We'll also annotate the no-fallback variant but it's not full coverage
        # so on an overall-accuracy plot it should still appear.
        points.append(("multi_signal", f"{clf_kind}_nofb",
                       sim_nofb["compute_per_problem"], sim_nofb["overall_accuracy"]))

    # Oracle
    oracle_K = pol["exp2_oracle"]["compute_per_problem"]
    oracle_acc = pol["exp2_oracle"]["overall_accuracy"]
    points.append(("oracle", "oracle", oracle_K, oracle_acc))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "uniform": "black",
        "threshold_prefill": "tab:blue",
        "threshold_neg_seq_len": "tab:green",
        "threshold_final_token": "tab:purple",
        "quartile": "tab:olive",
        "multi_signal": "tab:red",
        "oracle": "gold",
    }
    markers = {
        "uniform": "s",
        "threshold_prefill": "o",
        "threshold_neg_seq_len": "^",
        "threshold_final_token": "v",
        "quartile": "D",
        "multi_signal": "X",
        "oracle": "*",
    }

    # Plot by group
    groups = {}
    for g, name, k, acc in points:
        groups.setdefault(g, []).append((name, k, acc))
    for g, items in groups.items():
        ks = [x[1] for x in items]; accs = [x[2] for x in items]
        ax.scatter(ks, accs, s=120 if g == "oracle" else 70,
                   c=colors.get(g, "gray"),
                   marker=markers.get(g, "o"),
                   label=g, edgecolors="black",
                   linewidths=0.6, alpha=0.9, zorder=5 if g == "multi_signal" else 3)
        for name, k, acc in items:
            if g in ("uniform", "multi_signal", "oracle"):
                ax.annotate(name, (k, acc), fontsize=8,
                            xytext=(3, 3), textcoords="offset points")

    ax.axhline(oracle_acc, color="gold", ls="--", lw=1, alpha=0.5,
               label=f"oracle upper bound ({oracle_acc:.3f})")
    ax.set_xlabel("Compute per problem (K-samples)")
    ax.set_ylabel("Overall accuracy (refused = 0)")
    ax.set_title("Pareto: compute vs accuracy — Exp 2b multi-signal vs Exp 2 single-signal policies")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "pareto_comparison.png", dpi=140)
    plt.close(fig)
    print(f"Saved plot: {OUT_DIR/'pareto_comparison.png'}")

    # Consolidated results.json
    best_single_signal_at_K_2_74 = dict(
        policy="neg_seq_len_threshold_25pct",
        compute_per_problem=2.74, overall_accuracy=0.516,
    )
    multi_signal_logreg_k1fb = dict(
        compute_per_problem=pol["logreg"]["simulation_k1_fallback"]["compute_per_problem"],
        overall_accuracy=pol["logreg"]["simulation_k1_fallback"]["overall_accuracy"],
    )
    gain = multi_signal_logreg_k1fb["overall_accuracy"] - best_single_signal_at_K_2_74["overall_accuracy"]

    # D-bucket story
    d_story = dict(
        n_D=36,
        D_recall_to_K1_logreg=pol["logreg"]["bucket_routing"]["D"]["D_recall_to_K1"],
        D_recall_to_K1_rf=pol["rf"]["bucket_routing"]["D"]["D_recall_to_K1"],
        class_0_prior=243/500,
        baseline_random_assignment_D_K1=243/500,
    )

    summary = {
        "experiment": "Exp 2b — Multi-signal oracle-K classifier",
        "features_primary": ["prefill_dom", "prefill_lpr", "seq_len"],
        "features_extended": ["prefill_dom", "prefill_lpr", "seq_len",
                              "finaltok_dom", "finaltok_lpr", "mean_logprob"],
        "n_problems": 500,
        "target_classes": {
            "0_K1_AD": 243, "1_K8_B": 68, "2_refuse_C": 189,
        },
        "classifier_logreg": dict(
            macro_f1=clf["logreg"]["macro_f1"],
            overall_acc_oof=clf["logreg"]["accuracy"],
            D_recall_to_K1=clf["logreg"]["D_bucket_recall_to_K1"],
            confusion_matrix=clf["logreg"]["confusion_matrix"],
        ),
        "classifier_rf": dict(
            macro_f1=clf["rf"]["macro_f1"],
            overall_acc_oof=clf["rf"]["accuracy"],
            D_recall_to_K1=clf["rf"]["D_bucket_recall_to_K1"],
            confusion_matrix=clf["rf"]["confusion_matrix"],
        ),
        "policy_logreg_with_K1_fallback": pol["logreg"]["simulation_k1_fallback"],
        "policy_rf_with_K1_fallback": pol["rf"]["simulation_k1_fallback"],
        "best_single_signal_at_matched_compute": best_single_signal_at_K_2_74,
        "multi_signal_minus_single_signal_pp": round(gain * 100, 2),
        "user_threshold_2pp_met": bool(gain >= 0.02),
        "d_bucket": d_story,
        "ablation_drop_one_out_logreg": {
            r["label"]: dict(
                macro_f1=r["macro_f1"],
                overall_accuracy=r["overall_accuracy"],
                D_recall_to_K1=r["D_recall_to_K1"],
            )
            for r in abl["runs"]
            if r["clf"] == "logreg"
        },
        "exp2_policies_comparison": pol["exp2_policies"],
        "exp2_oracle": pol["exp2_oracle"],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {OUT_DIR/'results.json'}")
    print()
    print("=" * 70)
    print("HEADLINE")
    print("=" * 70)
    print(f"  Best single-signal at K≈2.74: neg_seq_len threshold → overall {best_single_signal_at_K_2_74['overall_accuracy']:.3f}")
    print(f"  Multi-signal logreg (+K=1 fallback) at K={multi_signal_logreg_k1fb['compute_per_problem']:.2f}: overall {multi_signal_logreg_k1fb['overall_accuracy']:.3f}")
    print(f"  Gain: {gain*100:+.2f}pp  (user's ≥2pp bar: {'MET' if gain >= 0.02 else 'NOT MET'})")
    print(f"  D-bucket recall to K=1: {d_story['D_recall_to_K1_logreg']:.3f} "
          f"(vs class-0 prior {d_story['class_0_prior']:.3f} — at chance)")
    print(f"  prefill_lpr contribution (drop-one-out): macro-F1 +0.000, overall +0.000 — DEAD WEIGHT")


if __name__ == "__main__":
    main()
