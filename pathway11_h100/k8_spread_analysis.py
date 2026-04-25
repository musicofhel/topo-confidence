#!/usr/bin/env python3
"""K=8 within-problem spread analysis on mixed-outcome MATH-500 problems.

For problems with >=2 correct AND >=2 incorrect samples (first 50), measure:
  1. Activation norm: mean ||x|| for correct vs incorrect (within problem)
  2. Cosine alignment with per-problem DoM direction
     (trivially separable by construction; included as sanity/magnitude check)
  3. Cosine alignment with GLOBAL DoM direction (non-circular)
  4. Within-group spread: mean ||x - centroid_group|| for each group
     (proxy for per-sample PR — can't compute covariance PR on single vectors)

Writes pathway11_h100/results/k8_spread_analysis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "k8_selfconsistency"
OUT = Path(__file__).parent / "results" / "k8_spread_analysis.json"


def load_mixed(n_target: int = 50) -> list[dict]:
    files = sorted(DATA_DIR.glob("problem_*.npz"))
    selected = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        correct = d["correct"].astype(bool)
        n_c = int(correct.sum())
        if 2 <= n_c <= 6:
            X = d["L19_samples"].astype(np.float32)
            selected.append({
                "problem_id": int(f.stem.split("_")[1]),
                "X": X,
                "correct": correct,
                "n_correct": n_c,
                "n_incorrect": int((~correct).sum()),
            })
            if len(selected) >= n_target:
                break
    return selected


def per_problem_stats(prob: dict, global_dom: np.ndarray) -> dict:
    X = prob["X"]
    correct = prob["correct"]
    Xc = X[correct]
    Xi = X[~correct]

    norms_c = np.linalg.norm(Xc, axis=1)
    norms_i = np.linalg.norm(Xi, axis=1)

    # Per-problem DoM (within-problem)
    dom_pp = Xc.mean(0) - Xi.mean(0)
    dom_pp_unit = dom_pp / (np.linalg.norm(dom_pp) + 1e-9)
    X_unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cos_pp = X_unit @ dom_pp_unit
    cos_pp_c = cos_pp[correct]
    cos_pp_i = cos_pp[~correct]

    # Global DoM alignment (non-circular)
    cos_gl = X_unit @ global_dom
    cos_gl_c = cos_gl[correct]
    cos_gl_i = cos_gl[~correct]

    # Within-group spread: distance from own-group centroid
    cent_c = Xc.mean(0)
    cent_i = Xi.mean(0)
    d_c = np.linalg.norm(Xc - cent_c, axis=1)
    d_i = np.linalg.norm(Xi - cent_i, axis=1)

    return {
        "problem_id": prob["problem_id"],
        "n_correct": prob["n_correct"],
        "n_incorrect": prob["n_incorrect"],
        "norm_correct_mean": float(norms_c.mean()),
        "norm_incorrect_mean": float(norms_i.mean()),
        "norm_delta": float(norms_c.mean() - norms_i.mean()),
        "cos_pp_correct_mean": float(cos_pp_c.mean()),
        "cos_pp_incorrect_mean": float(cos_pp_i.mean()),
        "cos_global_correct_mean": float(cos_gl_c.mean()),
        "cos_global_incorrect_mean": float(cos_gl_i.mean()),
        "spread_correct": float(d_c.mean()),
        "spread_incorrect": float(d_i.mean()),
        "spread_delta": float(d_c.mean() - d_i.mean()),
    }


def main():
    probs = load_mixed(n_target=50)
    print(f"Selected {len(probs)} mixed-outcome problems")

    # Global DoM pooled across selected problems
    all_correct = np.vstack([p["X"][p["correct"]] for p in probs])
    all_incorrect = np.vstack([p["X"][~p["correct"]] for p in probs])
    print(f"Pooled: {len(all_correct)} correct, {len(all_incorrect)} incorrect samples")
    global_dom = all_correct.mean(0) - all_incorrect.mean(0)
    global_dom_unit = global_dom / (np.linalg.norm(global_dom) + 1e-9)

    rows = [per_problem_stats(p, global_dom_unit) for p in probs]

    def agg(key):
        vals = np.array([r[key] for r in rows])
        return {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "n_positive": int((vals > 0).sum()),
            "n": len(vals),
        }

    # Paired sign test: within-problem, is correct tighter than incorrect?
    spread_sign = np.array([r["spread_delta"] for r in rows])
    norm_sign = np.array([r["norm_delta"] for r in rows])

    summary = {
        "n_problems": len(rows),
        "n_correct_samples_total": int(sum(r["n_correct"] for r in rows)),
        "n_incorrect_samples_total": int(sum(r["n_incorrect"] for r in rows)),
        "norm_correct": agg("norm_correct_mean"),
        "norm_incorrect": agg("norm_incorrect_mean"),
        "norm_delta_correct_minus_incorrect": agg("norm_delta"),
        "cos_per_problem_DoM_correct": agg("cos_pp_correct_mean"),
        "cos_per_problem_DoM_incorrect": agg("cos_pp_incorrect_mean"),
        "cos_global_DoM_correct": agg("cos_global_correct_mean"),
        "cos_global_DoM_incorrect": agg("cos_global_incorrect_mean"),
        "within_group_spread_correct": agg("spread_correct"),
        "within_group_spread_incorrect": agg("spread_incorrect"),
        "within_group_spread_delta_C_minus_I": agg("spread_delta"),
        "paired_sign_tests": {
            "spread_correct_less_than_incorrect": {
                "n_problems_where_correct_is_tighter": int((spread_sign < 0).sum()),
                "n_problems_total": len(spread_sign),
                "fraction": float((spread_sign < 0).mean()),
            },
            "norm_correct_less_than_incorrect": {
                "n_problems_where_correct_is_smaller": int((norm_sign < 0).sum()),
                "n_problems_total": len(norm_sign),
                "fraction": float((norm_sign < 0).mean()),
            },
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump({"summary": summary, "per_problem": rows}, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
