#!/usr/bin/env python3
"""Phase 1 — CPU transfer bake-off across the cache-ready panel.

Runs run_panel for every implemented probe across in-domain (OOF + incremental
gate), T1 (LOCO leave-one-MATH-category-out), T2 (MATH<->BBH cross-domain),
T3 (1.5B<->7B cross-scale; 'all'-tier probes only). Prints the adjudication
table and saves results/phase1_panel.json.

Not yet implemented (logged, not silently skipped): arm-3 GRIDE/PHD intrinsic-dim
scalars, arm-2A prompt-token-cloud (needs Phase-2 prefill pass), cov-spectrum
closure row (dry run already refuted it). T4/T5 need Phase 2/3 generation.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

import transfer as TR

PROBES = ["free_baseline", "length_only", "mean_logprob_only",
          "prefill_dom", "last_dom", "mean_dom", "concat_ridge",
          "coe_profile", "coe_depthgrid"]

NOT_IMPLEMENTED = {
    "arm3_gride_phd": "intrinsic-dim scalars (FE238/FE55) — estimator not ported",
    "arm2A_prompt_cloud": "needs Phase-2 prefill-only re-extraction",
    "cov_spectrum_closure": "dry-run already refuted (spectrum+len 0.761<len 0.786)",
}


def main():
    res = TR.run_panel(PROBES)
    res["_not_implemented"] = NOT_IMPLEMENTED
    (RESULTS / "phase1_panel.json").write_text(json.dumps(res, indent=2, default=float))

    idom = res["in_domain"]
    print("=== IN-DOMAIN (OOF) + incremental gate over free baseline ===")
    print(f"{'probe':18s} {'AUROC':>7s}  {'feat+free':>9s} {'gate_dp':>8s} {'p':>7s}")
    for name in PROBES:
        r = idom[name]
        g = r.get("gate")
        if g:
            print(f"{name:18s} {r['auroc']:7.4f}  {g['auroc_feat_plus_free']:9.4f} "
                  f"{g['delta']:+8.4f} {g['p']:7.3f}")
        else:
            print(f"{name:18s} {r['auroc']:7.4f}  {'(free)':>9s}")

    for tier, title in [("T1_loco", "T1 LOCO cross-category"),
                        ("T2_crossdomain", "T2 MATH<->BBH cross-domain"),
                        ("T3_crossscale", "T3 1.5B<->7B cross-scale")]:
        print(f"\n=== {title} ===")
        t = res[tier]
        for name in PROBES:
            r = t.get(name, {})
            if "skipped" in r:
                print(f"{name:18s} SKIPPED ({r['skipped']})")
            elif "aggregate" in r:
                gap = idom[name]["auroc"] - r["aggregate"]
                extra = (f"  worst={r['worst']:.4f}"
                         + (f" @{r['worst_cell']}" if "worst_cell" in r else ""))
                print(f"{name:18s} agg={r['aggregate']:.4f}{extra}  gap={gap:+.4f}")
    print(f"\n→ {RESULTS / 'phase1_panel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
