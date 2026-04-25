#!/usr/bin/env python3
"""Overlay PR curves for all three conditions onto one figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

POS_LABELS = ["1", "10", "25", "50", "100", "200", "final"]
# X-axis numeric mapping: 'final' sits just past 200 for visual layout.
X_VALUES = [1, 10, 25, 50, 100, 200, 260]

STYLES = {
    "math500_baseline": {"color": "#1f77b4", "marker": "o", "label": "MATH-500 (reasoning)"},
    "stream_of_consciousness": {"color": "#ff7f0e", "marker": "s", "label": "Stream-of-consciousness"},
    "random_tokens": {"color": "#2ca02c", "marker": "^", "label": "Random tokens"},
}


def main() -> None:
    data = json.loads((HERE / "pr_curves.json").read_text())

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, res in data["conditions"].items():
        if "error" in res:
            continue
        pr_vals = []
        x_vals = []
        for x, lbl in zip(X_VALUES, POS_LABELS):
            entry = res["pr_by_position"][lbl]
            if entry["pr"] is None:
                continue
            pr_vals.append(entry["pr"])
            x_vals.append(x)

        style = STYLES.get(name, {"color": "gray", "marker": "x", "label": name})
        ax.plot(
            x_vals,
            pr_vals,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=7,
            label=f"{style['label']} (n={res['n_problems']})",
        )

    ax.set_xlabel("Generation-token position")
    ax.set_ylabel("Participation Ratio at L19 (2/3-depth)")
    ax.set_xscale("log")
    ax.set_xticks(X_VALUES)
    ax.set_xticklabels(POS_LABELS)
    ax.set_title(
        "Qwen2.5-1.5B: temporal PR at L19 — MATH-500 vs gibberish controls\n"
        "Does breathing happen on non-reasoning content?"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = HERE / "gibberish_vs_math500.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
