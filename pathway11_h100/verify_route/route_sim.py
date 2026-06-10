#!/usr/bin/env python3
"""S5 — Routing simulation (CPU): accuracy vs avg compute (K-equivalent) for every policy.

Compute metric = avg K-equivalent generated-token cost, CHARGING verification (~0.1) and
correction (~1.0) so the matched-compute comparison is honest. Each problem already paid
K=1 (1.0). The D-bucket trap is automatic: routing a K=1-right problem to K=8/correct uses
that problem's k8_majority / corrected outcome, which may be wrong — the frontier nets it out.

Arms:
  A_uniform   : uniform K in {1,2,4,8}  (published frontier points; cost=K)
  A_gate      : prefill-DoM pre-hoc gate (C_infer) — route low-conf to K=8, no verify cost
  B_resample  : verify, then route-fail -> K=8 majority   (cost 1.1 + 7*frac_routed)
  C_correct   : verify, then route-fail -> corrected ans  (cost 1.1 + 1*frac_routed)

Routers (the "flag as wrong" signal; higher = more likely K=1-wrong):
  panl_best, panl_l19, prefill_dom (negated), verbalized (binary No).

Output: results/route_frontier.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cache_utils as cu  # noqa: E402

N = 500
VERIFY_COST = 0.1
CORRECT_COST = 1.0


def trace_arm(route_score, greedy, k8maj, corrected, arm):
    """Sweep #routed (top-k by route_score desc) -> list of (avg_cost, acc, frac_routed)."""
    rs = np.where(np.isfinite(route_score), route_score, -np.inf)
    order = np.argsort(-rs)  # most-wrong-looking first
    pts = []
    for k in range(N + 1):
        routed = np.zeros(N, bool)
        routed[order[:k]] = True
        if arm == "A_gate":
            cost = np.where(routed, 8.0, 1.0)
            acc = np.where(routed, k8maj, greedy)
        elif arm == "B_resample":
            cost = 1.0 + VERIFY_COST + np.where(routed, 7.0, 0.0)
            acc = np.where(routed, k8maj, greedy)
        elif arm == "C_correct":
            cost = 1.0 + VERIFY_COST + np.where(routed, CORRECT_COST, 0.0)
            acc = np.where(routed, corrected, greedy)
        else:
            raise ValueError(arm)
        pts.append([round(float(cost.mean()), 4), round(float(acc.mean()), 4), round(k / N, 4)])
    return pts


def best_below_budget(pts, budget):
    """Highest-accuracy point with avg_cost <= budget."""
    cand = [p for p in pts if p[0] <= budget + 1e-9]
    return max(cand, key=lambda p: p[1]) if cand else None


def main() -> int:
    idx, greedy = cu.load_k1_greedy()
    p1 = cu.load_phase1()
    k8maj = p1["k8_majority_correct"]
    cu.buckets(greedy, k8maj)  # sanity assert 207/68/189/36

    corr = json.loads((HERE / "results" / "correct_pass.json").read_text())["rows"]
    corr = sorted(corr, key=lambda r: r["idx"])
    corrected = np.array([r["corrected_correct"] for r in corr], dtype=bool)

    panl = np.load(HERE / "results" / "panl_oof.npz", allow_pickle=True)
    panl_best = panl["panl_score_best"].astype(float)   # higher = wrong
    panl_l19 = panl["panl_score_l19"].astype(float)
    best_layer = int(panl["best_layer"])

    prefill_dom = cu.load_prefill_dom()                  # higher = correct
    prefill_wrong = -prefill_dom                         # higher = wrong

    vrows = json.loads((HERE / "results" / "verify.json").read_text())["rows"]
    vrows = sorted(vrows, key=lambda r: r["idx"])
    verdict_no = np.array([not r["verdict_bool"] for r in vrows], dtype=bool)  # True = route
    # binary router score: 1.0 where verdict==No (route), else 0.0
    verb_score = verdict_no.astype(float)

    routers = {"panl_best": panl_best, "panl_l19": panl_l19,
               "prefill_dom": prefill_wrong, "verbalized": verb_score}

    uni = cu.uniform_frontier(p1, greedy)  # {K:(cost,acc)}
    uni_pts = [[v[0], round(v[1], 4)] for v in uni.values()]
    uniform_K8 = uni[8]  # (8.0, 0.55)

    curves = {}
    for arm in ("A_gate", "B_resample", "C_correct"):
        for rname, rs in routers.items():
            # A_gate only makes sense for the prefill-DoM C_infer baseline
            if arm == "A_gate" and rname != "prefill_dom":
                continue
            curves[f"{arm}|{rname}"] = trace_arm(rs, greedy, k8maj, corrected, arm)

    # the single verbalized operating point (k = #flagged-No) for B and C
    nflag = int(verdict_no.sum())
    verb_points = {}
    for arm in ("B_resample", "C_correct"):
        pts = curves[f"{arm}|verbalized"]
        verb_points[arm] = pts[nflag]  # [cost, acc, frac] at exactly the flagged set

    out = {
        "uniform_K_frontier": {f"K={k}": uni_pts[i] for i, k in enumerate([1, 2, 4, 8])},
        "uniform_K8": list(uniform_K8),
        "oracle": [1.01, 0.589],
        "best_panl_layer": best_layer,
        "n_flagged_by_verbalized_verdict": nflag,
        "verbalized_operating_points": verb_points,
        "curves": curves,
        # headline summaries: best acc reachable under matched budgets vs uniform/gating
        "summary": {},
    }

    # For budgets matching uniform K points, report best acc per arm/router
    for budget, label in [(2.0, "budget~2"), (4.0, "budget~4"), (8.0, "budget~8")]:
        row = {"uniform": round(best_below_budget(
            [[c, a] for c, a in [uni[1], uni[2], uni[4], uni[8]]], budget)[1], 4)}
        for key, pts in curves.items():
            b = best_below_budget(pts, budget)
            if b:
                row[key] = b[1]
        out["summary"][label] = row

    (HERE / "results" / "route_frontier.json").write_text(json.dumps(out, indent=1))

    print("  uniform-K frontier:", {f"K{k}": uni[k] for k in (1, 2, 4, 8)})
    print(f"  best PANL layer = L{best_layer}")
    print("  --- best accuracy at matched budget (charging verify+correct) ---")
    for label, row in out["summary"].items():
        print(f"  {label}: " + "  ".join(f"{k}={v}" for k, v in sorted(row.items())))
    print("  wrote results/route_frontier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
