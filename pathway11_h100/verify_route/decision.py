#!/usr/bin/env python3
"""S6 — Decision (CPU): apply pre-registered criteria -> verdict.json.

Pre-registered (encoded BEFORE running, per the plan):

  C_exact WINS (H-19 confirmed): some Arm-B or Arm-C policy reaches
    (i)  accuracy >= uniform-K8's 55.0% at avg-K < 8, OR
    (ii) sits >= 2pp above the uniform-K frontier at matched avg-K,
    AND beats prefill-gating (A_gate) at matched avg-K.
  C_exact LOSES: no verification-routed policy beats the uniform-K frontier.

  Routing-signal claim: report PANL-probe OOF AUROC (K=1-wrong) vs verbalized-verdict
    and prefill-DoM. Kumaran predicts PANL > verbalized.
  Verify-then-correct claim: full-coverage corrected accuracy vs K=1 48.6%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def _upper_hull(points):
    """Upper convex hull (max achievable acc per cost). Dominated uniform points
    (e.g. K=2 majority = 0.413, worse than K=1 at higher cost; and K=4 below the
    K1->K8 line) are dropped. The hull == what RANDOM routing between the surviving
    K's achieves, so it is the honest baseline a real router must beat."""
    pts = sorted(points)
    h = []
    for p in pts:
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                h.pop()
            else:
                break
        h.append(tuple(p))
    return h


def uniform_acc_at(frontier_pts, avg_k):
    """Accuracy of the achievable uniform frontier (UPPER CONVEX HULL = random routing)
    at a given avg compute. NOT raw interpolation — that would pass through dominated
    points (K=2/K=4) and spuriously inflate any 'win'."""
    hull = _upper_hull([(p[0], p[1]) for p in frontier_pts])
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    return float(np.interp(avg_k, xs, ys))


def main() -> int:
    fr = json.loads((R / "route_frontier.json").read_text())
    panl = json.loads((R / "panl_probe.json").read_text())
    corr = json.loads((R / "correct_pass.json").read_text())

    uni_pts = [fr["uniform_K_frontier"][k] for k in ("K=1", "K=2", "K=4", "K=8")]
    uni_K8_acc = fr["uniform_K8"][1]
    curves = fr["curves"]

    # Gate-arm (C_infer) frontier — the thing we must beat
    gate = curves.get("A_gate|prefill_dom", [])

    def gate_acc_at(avg_k):
        if not gate:
            return None
        return uniform_acc_at([[p[0], p[1]] for p in gate], avg_k)

    # --- honesty guards (added at impl, BEFORE reading curve values, to avoid p-hacking) ---
    # A "win" must be (a) statistically real: beat the matched uniform-frontier point by
    #   >= 1.96 SE (n=500), not just nominally; and (b) NON-DEGENERATE: not a router that
    #   sends almost everything to resampling (frac_routed <= 0.70) — such a policy is just
    #   uniform-K8 in disguise and inherits its accuracy by construction, not by routing.
    # These make the verdict MORE conservative than the loose pre-registered cond_i
    # ("acc >= 55% at avg-K < 8"), which a near-K=8 degenerate policy can satisfy on noise.
    N = 500
    SE = (uni_K8_acc * (1 - uni_K8_acc) / N) ** 0.5  # ~0.0222
    MARGIN = 1.96 * SE                               # 95% one-sided-ish significance band
    FRAC_CAP = 0.70

    wins = []                 # significant, non-degenerate (the honest bar)
    nominal_passes = []       # loose cond_i/cond_ii passes (recorded for transparency)
    for key, pts in curves.items():
        arm = key.split("|")[0]
        if arm not in ("B_resample", "C_correct"):
            continue
        for cost, acc, frac in pts:
            if cost >= 8.0:
                continue
            uni_here = uniform_acc_at(uni_pts, cost)
            gate_here = gate_acc_at(cost)
            beats_gate = (gate_here is None) or (acc >= gate_here + 1e-9)
            cond_i = acc >= uni_K8_acc                         # loose: reaches K=8 acc
            cond_ii = acc >= uni_here + 0.02                   # loose: +2pp over frontier
            rec = {
                "policy": key, "avg_K": cost, "accuracy": acc, "frac_routed": frac,
                "above_uniform_frontier_by": round(acc - uni_here, 4),
                "uniform_at_this_K": round(uni_here, 4),
                "gate_at_this_K": None if gate_here is None else round(gate_here, 4),
                "exceeds_uniform_by_SE": round((acc - uni_here) / SE, 2),
                "degenerate_router": bool(frac > FRAC_CAP),
            }
            if (cond_i or cond_ii) and beats_gate:
                nominal_passes.append(rec)
            # honest win: significant AND non-degenerate AND beats gate
            if (acc >= uni_here + MARGIN) and (frac <= FRAC_CAP) and beats_gate:
                wins.append(rec)

    wins.sort(key=lambda w: (-w["accuracy"], w["avg_K"]))
    nominal_passes.sort(key=lambda w: (-w["accuracy"], w["avg_K"]))
    c_exact_wins = len(wins) > 0
    best = wins[0] if wins else None
    # best signal-driven, meaningful-compute point (avg-K <= 4, non-degenerate) for reporting
    meaningful = [r for k, pts in curves.items() if k.split("|")[0] in ("B_resample", "C_correct")
                  for (cost, acc, frac) in pts
                  for r in [{"policy": k, "avg_K": cost, "accuracy": acc, "frac_routed": frac,
                             "vs_uniform_SE": round((acc - uniform_acc_at(uni_pts, cost)) / SE, 2)}]
                  if cost <= 4.0 and frac <= FRAC_CAP and "verbalized" not in k]
    meaningful.sort(key=lambda w: -w["accuracy"])
    best_meaningful = meaningful[0] if meaningful else None

    # routing-signal comparison
    panl_best = panl["best_dom_auroc_wrong"]
    panl_l19 = panl["L19_dom_auroc_wrong"]
    verb_auroc = panl["comparators"]["verbalized_verdict_auroc_wrong"]
    prefill_auroc = panl["comparators"]["prefill_DoM_auroc_wrong"]

    verdict = {
        "experiment": "P11-FE19",
        "date": "2026-06-10",
        "C_exact_verdict": "WINS" if c_exact_wins else "LOSES",
        "verdict_basis": ("a non-degenerate (frac_routed<=0.70) verification-routed policy "
                          "beats the matched uniform-K point by >=1.96 SE" if c_exact_wins else
                          "no non-degenerate verification-routed policy beats the uniform-K "
                          "frontier beyond noise (1.96 SE = +/-%.3f at n=500)" % (1.96 * SE)),
        "best_winning_policy": best,
        "n_winning_operating_points": len(wins),
        "best_meaningful_signal_point": best_meaningful,
        "loose_preregistered_passes": {
            "note": ("cond_i ('acc>=55% at avg-K<8') / cond_ii ('+2pp over frontier') WITHOUT "
                     "the significance+degeneracy guards. Recorded for transparency; these are "
                     "gameable by a near-K=8 degenerate router on noise and do NOT establish a win."),
            "n": len(nominal_passes),
            "top": nominal_passes[0] if nominal_passes else None,
        },
        "comparison": {
            "uniform_K8_accuracy": uni_K8_acc,
            "oracle_accuracy": fr["oracle"][1],
            "binomial_SE_n500": round(SE, 4),
            "significance_margin_1p96SE": round(1.96 * SE, 4),
            "prefill_gating_is_C_infer_baseline": True,
        },
        "routing_signal": {
            "panl_best_layer": panl["best_layer_for_wrong"],
            "panl_best_auroc_wrong": panl_best,
            "panl_l19_auroc_wrong": panl_l19,
            "verbalized_verdict_auroc_wrong": verb_auroc,
            "prefill_dom_auroc_wrong": prefill_auroc,
            "panl_beats_verbalized": (verb_auroc != verb_auroc) or (panl_best > verb_auroc),  # nan-safe
            "panl_beats_prefill_dom": panl_best > prefill_auroc,
        },
        "verify_then_correct": {
            "full_coverage_corrected_accuracy": corr["full_coverage_corrected_accuracy"],
            "k1_accuracy": corr["k1_accuracy"],
            "delta_vs_k1": corr["delta_vs_k1"],
            "self_correction_helps": corr["delta_vs_k1"] > 0,
            "flipped_right_to_wrong": corr["n_flipped_right_to_wrong"],
            "flipped_wrong_to_right": corr["n_flipped_wrong_to_right"],
        },
        "budget_summary": fr["summary"],
    }

    (R / "verdict.json").write_text(json.dumps(verdict, indent=1))

    print("=" * 72)
    print(f"  P11-FE19 VERDICT: C_exact {verdict['C_exact_verdict']}")
    print(f"    basis: {verdict['verdict_basis']}")
    if best:
        print(f"    best (significant, non-degenerate): {best['policy']}  acc={best['accuracy']} "
              f"@ avg-K={best['avg_K']} (+{best['exceeds_uniform_by_SE']} SE over uniform)")
    else:
        print("    no significant, non-degenerate verification-routed win.")
    if best_meaningful:
        bm = best_meaningful
        print(f"    best signal point @avg-K<=4: {bm['policy']} acc={bm['accuracy']} "
              f"@K={bm['avg_K']} ({bm['vs_uniform_SE']} SE vs uniform)")
    if nominal_passes:
        np0 = nominal_passes[0]
        print(f"    (loose cond_i/ii pass exists: {np0['policy']} acc={np0['accuracy']} "
              f"@K={np0['avg_K']} frac={np0['frac_routed']} degenerate={np0['degenerate_router']} "
              f"— noise-level, not a real win)")
    print(f"    uniform-K8 = {uni_K8_acc}  oracle = {fr['oracle'][1]}")
    print(f"  routing signal: PANL best L{panl['best_layer_for_wrong']} AUROC(wrong)={panl_best:.4f} | "
          f"PANL L19={panl_l19:.4f} | verbalized={verb_auroc:.4f} | prefill-DoM={prefill_auroc:.4f}")
    print(f"  verify-then-correct: {corr['full_coverage_corrected_accuracy']:.4f} "
          f"(K=1 {corr['k1_accuracy']:.4f}, Δ={corr['delta_vs_k1']:+.4f})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
