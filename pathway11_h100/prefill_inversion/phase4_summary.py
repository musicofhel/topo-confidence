#!/usr/bin/env python3
"""Experiment 3 / Phase 4: summary + plots.

Consumes phase1_bootstrap.json, phase2_cross_scale.json, phase3_mechanistic.json
and writes results.json + SUMMARY.md + three plots:
  - bootstrap_ci.png: PR ratio bootstrap distributions (7B vs 1.5B, prefill vs final-tok)
  - three_way_split.png: bar chart of PR per (model × bucket × layer_position)
  - d_bucket_signature.png: D-bucket PR + DoM score + seq_len distributions
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/prefill_inversion"


def load():
    p1 = json.loads((OUT_DIR / "phase1_bootstrap.json").read_text())
    p2 = json.loads((OUT_DIR / "phase2_cross_scale.json").read_text())
    p3 = json.loads((OUT_DIR / "phase3_mechanistic.json").read_text())
    return p1, p2, p3


def plot_bootstrap_ci(p1, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Prefill
    ax = axes[0]
    models = ["7B_prefill", "1.5B_prefill"]
    for i, m in enumerate(models):
        r = p1[m]["bootstrap"]["ratio"]
        mean = r["mean"]
        lo, hi = r["ci95"]
        ax.errorbar([i], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o",
                    ms=12, capsize=8, lw=2)
        ax.plot([i], [p1[m]["point"]["ratio"]], "rx", ms=10, label="point" if i == 0 else None)
        ax.text(i, hi + 0.05, f"P(>1)={r['pct_ratio_above_1']:.2f}",
                ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", lw=1, label="ratio=1 (no inversion)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Ratio PR_correct / PR_incorrect")
    ax.set_title("Prefill — bootstrap 95% CI")
    ax.legend()
    ax.grid(alpha=0.3)

    # Final token
    ax = axes[1]
    models = ["7B_final_token", "1.5B_final_token"]
    for i, m in enumerate(models):
        r = p1[m]["bootstrap"]["ratio"]
        mean = r["mean"]
        lo, hi = r["ci95"]
        ax.errorbar([i], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o",
                    ms=12, capsize=8, lw=2)
        ax.plot([i], [p1[m]["point"]["ratio"]], "rx", ms=10, label="point" if i == 0 else None)
    ax.axhline(1.0, color="gray", linestyle="--", lw=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Ratio PR_correct / PR_incorrect")
    ax.set_title("Final token — bootstrap 95% CI")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_three_way(p2, out_path):
    tw = p2["three_way"]
    groups = ["both_right", "only_7b", "only_15b", "both_wrong"]
    colors = {"both_right": "tab:green", "only_7b": "tab:orange",
              "only_15b": "tab:purple", "both_wrong": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, model in enumerate(["7b", "15b"]):
        ax = axes[i]
        xs = np.arange(len(groups))
        prefill_vals = [tw[g].get(f"pr_{model}_prefill") for g in groups]
        final_vals = [tw[g].get(f"pr_{model}_finaltok") for g in groups]
        width = 0.35
        ax.bar(xs - width / 2, prefill_vals, width, label="prefill",
               color=[colors[g] for g in groups], edgecolor="black")
        ax.bar(xs + width / 2, final_vals, width, label="final-tok",
               color=[colors[g] for g in groups], edgecolor="black",
               hatch="///", alpha=0.6)
        for j, g in enumerate(groups):
            n = tw[g]["n"]
            ax.text(j, max(prefill_vals[j] or 0, final_vals[j] or 0) + 1.0,
                    f"n={n}", ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(groups, rotation=15)
        ax.set_ylabel("Participation ratio")
        ax.set_title(f"{'Qwen-2.5-7B' if model == '7b' else 'Qwen-2.5-1.5B'} prefill/final PR by three-way bucket")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_d_bucket(p2, out_path):
    d = p2["d_bucket"]
    buckets = ["A_always_right", "B_recoverable", "C_never_right", "D_pathological"]
    colors = ["tab:green", "tab:blue", "tab:red", "tab:orange"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) PR per bucket
    ax = axes[0]
    vals_pre = [d["per_bucket"][b].get("pr_prefill", 0) for b in buckets]
    vals_fin = [d["per_bucket"][b].get("pr_finaltok", 0) for b in buckets]
    xs = np.arange(len(buckets))
    w = 0.35
    ax.bar(xs - w/2, vals_pre, w, label="prefill", color=colors, edgecolor="black")
    ax.bar(xs + w/2, vals_fin, w, label="final-tok", color=colors, edgecolor="black",
           hatch="///", alpha=0.6)
    for i, b in enumerate(buckets):
        n = d["per_bucket"][b]["n"]
        ax.text(i, max(vals_pre[i], vals_fin[i]) + 0.5, f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([b.split("_", 1)[0] for b in buckets])
    ax.set_ylabel("Participation ratio")
    ax.set_title("1.5B PR per bucket")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # 2) DoM score mean ± std
    ax = axes[1]
    means = [d["per_bucket"][b].get("prefill_score_mean", 0) for b in buckets]
    stds = [d["per_bucket"][b].get("prefill_score_std", 0) for b in buckets]
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="black", capsize=6)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([b.split("_", 1)[0] for b in buckets])
    ax.set_ylabel("Prefill DoM score (OOF, mean±std)")
    ax.set_title("Prefill confidence by bucket")
    ax.grid(alpha=0.3, axis="y")

    # 3) Seq len mean ± std
    ax = axes[2]
    means = [d["per_bucket"][b].get("seq_len_mean", 0) for b in buckets]
    stds = [d["per_bucket"][b].get("seq_len_std", 0) for b in buckets]
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="black", capsize=6)
    ax.set_xticks(xs); ax.set_xticklabels([b.split("_", 1)[0] for b in buckets])
    ax.set_ylabel("Generation length (tokens, mean±std)")
    ax.set_title("Greedy K=1 generation length by bucket")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p1, p2, p3 = load()
    plot_bootstrap_ci(p1, OUT_DIR / "bootstrap_ci.png")
    plot_three_way(p2, OUT_DIR / "three_way_split.png")
    plot_d_bucket(p2, OUT_DIR / "d_bucket_signature.png")
    print("Wrote plots.")

    # Build consolidated results.json
    b7p = p1["7B_prefill"]
    b15p = p1["1.5B_prefill"]
    b7f = p1["7B_final_token"]
    b15f = p1["1.5B_final_token"]

    summary = {
        "experiment": "7B prefill PR inversion",
        "headline_ratios_point_estimates": {
            "7B_prefill_PR_correct_over_incorrect": b7p["point"]["ratio"],
            "1.5B_prefill_PR_correct_over_incorrect": b15p["point"]["ratio"],
            "7B_final_token_PR_correct_over_incorrect": b7f["point"]["ratio"],
            "1.5B_final_token_PR_correct_over_incorrect": b15f["point"]["ratio"],
        },
        "bootstrap_95_CI_on_ratio": {
            "7B_prefill": b7p["bootstrap"]["ratio"]["ci95"],
            "1.5B_prefill": b15p["bootstrap"]["ratio"]["ci95"],
            "7B_final_token": b7f["bootstrap"]["ratio"]["ci95"],
            "1.5B_final_token": b15f["bootstrap"]["ratio"]["ci95"],
        },
        "balance_controlled_ratios": {
            "7B_prefill": b7p["balance_control"]["ratio_correct_over_incorrect_mean"],
            "1.5B_prefill": b15p["balance_control"]["ratio_correct_over_incorrect_mean"],
            "7B_final_token": b7f["balance_control"]["ratio_correct_over_incorrect_mean"],
            "1.5B_final_token": b15f["balance_control"]["ratio_correct_over_incorrect_mean"],
        },
        "three_way_split": p2["three_way"],
        "bbh": p2["bbh"],
        "d_bucket": p2["d_bucket"],
        "mechanistic": {
            "have_levels": p3.get("have_levels"),
            "level_distribution": p3.get("level_distribution"),
            "pr_per_level": p3.get("pr_per_level"),
            "kmeans_7b_ari_correct": {f"k{k}": p3["kmeans_7b"][f"k{k}"]["ari_correct"]
                                       for k in [5, 6, 8, 10]},
            "kmeans_15b_ari_correct": {f"k{k}": p3["kmeans_15b"][f"k{k}"]["ari_correct"]
                                        for k in [5, 6, 8, 10]},
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
