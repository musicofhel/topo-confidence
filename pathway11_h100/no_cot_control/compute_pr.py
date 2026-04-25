#!/usr/bin/env python3
"""Compute temporal PR curves for no-CoT vs CoT baseline.

Reads the no-CoT extraction at data/nocot/ and the existing CoT MATH-500
baseline at ../gibberish_control/data/math500/. Writes pr_curves.json.

Reuses participation_ratio from exp1_cross_model.analyze.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pathway11_h100.exp1_cross_model.analyze import participation_ratio  # noqa: E402

TARGET_POSITIONS = [1, 10, 25, 50, 100, 200]
POS_LABELS = ["1", "10", "25", "50", "100", "200", "final"]


def compute_condition(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("problem_*.npz"))
    if not files:
        return {"error": f"No npz files in {data_dir}"}

    by_label: dict[str, list[np.ndarray]] = {lbl: [] for lbl in POS_LABELS}
    n_gen_tokens_all: list[int] = []
    correct_count = 0
    twothirds_layer_idx = None

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        positions = d["positions_actual"].tolist()
        vecs = d["twothirds_positions"]
        n_gen = int(d["n_gen_tokens"])
        n_gen_tokens_all.append(n_gen)
        correct_count += int(bool(d["correct"]))
        if twothirds_layer_idx is None:
            twothirds_layer_idx = int(d["twothirds_layer_idx"])

        for p, v in zip(positions, vecs):
            if int(p) == n_gen:
                by_label["final"].append(v.astype(np.float32))
            if int(p) in TARGET_POSITIONS:
                by_label[str(int(p))].append(v.astype(np.float32))

    pr_by_label: dict[str, dict] = {}
    for lbl in POS_LABELS:
        stack = by_label[lbl]
        if len(stack) < 5:
            pr_by_label[lbl] = {"n": len(stack), "pr": None}
            continue
        X = np.stack(stack, axis=0)
        pr_by_label[lbl] = {"n": len(stack), "pr": participation_ratio(X)}

    return {
        "data_dir": str(data_dir),
        "n_problems": len(files),
        "n_correct": correct_count,
        "twothirds_layer_idx": twothirds_layer_idx,
        "n_gen_tokens": {
            "min": int(min(n_gen_tokens_all)),
            "median": float(np.median(n_gen_tokens_all)),
            "max": int(max(n_gen_tokens_all)),
            "p25": float(np.percentile(n_gen_tokens_all, 25)),
            "p75": float(np.percentile(n_gen_tokens_all, 75)),
        },
        "pr_by_position": pr_by_label,
    }


def main() -> None:
    conditions = {
        "nocot": HERE / "data" / "nocot",
        "cot_baseline": HERE.parent / "gibberish_control" / "data" / "math500",
    }
    results: dict = {"conditions": {}}
    for name, d in conditions.items():
        if not d.exists():
            print(f"[skip] {name}: {d} does not exist")
            continue
        print(f"[compute] {name} from {d}")
        results["conditions"][name] = compute_condition(d)

    out_path = HERE / "pr_curves.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}")

    print("\nPR curve summary (position : PR [n=])")
    for name, res in results["conditions"].items():
        if "error" in res:
            print(f"  {name}: {res['error']}")
            continue
        tok = res["n_gen_tokens"]
        print(
            f"  {name} (L{res['twothirds_layer_idx']}, n_problems={res['n_problems']}, "
            f"correct={res['n_correct']}, n_gen_tokens median={tok['median']:.0f} "
            f"p25={tok['p25']:.0f} p75={tok['p75']:.0f}):"
        )
        for lbl in POS_LABELS:
            entry = res["pr_by_position"][lbl]
            pr = entry["pr"]
            if pr is None:
                print(f"    pos {lbl:>5}: -- [n={entry['n']}]")
            else:
                print(f"    pos {lbl:>5}: {pr:6.2f} [n={entry['n']}]")


if __name__ == "__main__":
    main()
