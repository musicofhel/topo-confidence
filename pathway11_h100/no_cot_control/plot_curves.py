#!/usr/bin/env python3
"""Overlay no-CoT and CoT MATH-500 PR curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POS_LABELS = ["1", "10", "25", "50", "100", "200", "final"]


def main() -> None:
    data = json.loads((HERE / "pr_curves.json").read_text())
    conditions = data["conditions"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    style = {
        "cot_baseline": {"label": "CoT (step-by-step)", "color": "tab:blue", "marker": "o"},
        "nocot": {"label": "No-CoT (answer only)", "color": "tab:red", "marker": "s"},
    }

    for name, cfg in style.items():
        if name not in conditions or "error" in conditions[name]:
            continue
        prs = conditions[name]["pr_by_position"]
        xs, ys, ns = [], [], []
        for i, lbl in enumerate(POS_LABELS):
            entry = prs[lbl]
            if entry["pr"] is None:
                continue
            xs.append(i)
            ys.append(entry["pr"])
            ns.append(entry["n"])
        ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"], label=cfg["label"])
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 6), fontsize=7, ha="center", color=cfg["color"])

    ax.set_xticks(range(len(POS_LABELS)))
    ax.set_xticklabels(POS_LABELS)
    ax.set_xlabel("Generation position")
    ax.set_ylabel("Participation Ratio at L19")
    ax.set_title("PR breathing with/without CoT (Qwen2.5-1.5B, MATH-500 first 50)")
    ax.legend()
    ax.grid(alpha=0.3)

    out = HERE / "nocot_vs_cot.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
