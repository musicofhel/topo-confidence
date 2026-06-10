#!/usr/bin/env python3
"""FE269 — apply the PRE-REGISTERED verdict criteria to the result JSONs.

Pre-registered (committed before the run):
  ΔMATH = MATH_off - MATH_ablate (percentage points). Capability drops measured under
  the SAME ablation config that produced ΔMATH (L19-only or all-layer).

  LEVER          : ΔMATH ≥ 15 AND GSM8K drop ≤ 5 AND MMLU drop ≤ 5
                   AND best-α addition flips ≥ 15% of incorrect with retention ≥ 85%
  DIAGNOSTIC     : ablated MATH survives ≥ 45% (ΔMATH ≤ ~3.6 pts) → reads out, doesn't cause
  GENERAL_DAMAGE : ΔMATH ≥ 15 BUT GSM8K and/or MMLU also drop ≥ 15 → broke model broadly
  INCONCLUSIVE   : ΔMATH 5–15 with ≤5 capability drop (weak lever, needs replication)

Usage: python decision.py --results results/ --out results/verdict.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

PP = 100.0  # fraction -> percentage points


def load(results: Path, name: str):
    fp = results / name
    return json.loads(fp.read_text()) if fp.exists() else None


def acc(j):
    return None if j is None else j["accuracy"] * PP


def classify_config(label, math_off, math_ab, gsm_off, gsm_ab, mmlu_off, mmlu_ab):
    """Classify one ablation config (necessity + specificity, no addition yet)."""
    if math_off is None or math_ab is None:
        return None
    d_math = math_off - math_ab
    d_gsm = (gsm_off - gsm_ab) if (gsm_off is not None and gsm_ab is not None) else None
    d_mmlu = (mmlu_off - mmlu_ab) if (mmlu_off is not None and mmlu_ab is not None) else None
    cap_drops = [d for d in (d_gsm, d_mmlu) if d is not None]
    rec = {
        "config": label, "math_off": math_off, "math_ablate": math_ab,
        "delta_math": d_math, "gsm8k_drop": d_gsm, "mmlu_drop": d_mmlu,
        "math_survives_pct": math_ab,
    }
    if math_ab >= 45.0:
        rec["necessity_class"] = "DIAGNOSTIC"          # survives ablation
    elif d_math >= 15.0:
        if cap_drops and max(cap_drops) >= 15.0:
            rec["necessity_class"] = "GENERAL_DAMAGE"
        elif all(d <= 5.0 for d in cap_drops) and cap_drops:
            rec["necessity_class"] = "NECESSARY_SPECIFIC"
        else:
            rec["necessity_class"] = "NECESSARY_UNCLEAR_CAP"  # 5<drop<15 capability
    elif d_math >= 5.0:
        rec["necessity_class"] = "INCONCLUSIVE"
    else:
        rec["necessity_class"] = "DIAGNOSTIC"          # ΔMATH < 5
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = Path(args.results)

    math_off = acc(load(R, "math_off.json"))
    gsm_off = acc(load(R, "gsm8k_off.json"))
    mmlu_off = acc(load(R, "mmlu_off.json"))

    configs = []
    for label, mfile, gfile, mmfile in [
        ("L19", "math_ablate_L19.json", "gsm8k_ablate_L19.json", "mmlu_ablate_L19.json"),
        ("all", "math_ablate_all.json", "gsm8k_ablate_all.json", "mmlu_ablate_all.json"),
    ]:
        rec = classify_config(
            label, math_off, acc(load(R, mfile)),
            gsm_off, acc(load(R, gfile)), mmlu_off, acc(load(R, mmfile)),
        )
        if rec is not None:
            configs.append(rec)

    # Addition arm: best α by flip rate (subject to retention ≥ 85%)
    add_runs = []
    for fp in sorted(glob.glob(str(R / "math_add_*.json"))):
        j = json.loads(Path(fp).read_text())
        add_runs.append({
            "file": Path(fp).name, "alpha_mult": j.get("alpha_mult"),
            "flip_rate": j.get("flip_rate_on_incorrect"),
            "retention": j.get("retention_on_correct"),
            "accuracy": j["accuracy"] * PP,
        })
    valid_add = [a for a in add_runs if a["flip_rate"] is not None
                 and a["retention"] is not None and a["retention"] >= 0.85]
    best_add = max(valid_add, key=lambda a: a["flip_rate"]) if valid_add else None
    addition_sufficient = bool(best_add and best_add["flip_rate"] >= 0.15)

    # Overall verdict: prefer the config with the largest specific ΔMATH.
    necessary_specific = [c for c in configs if c["necessity_class"] == "NECESSARY_SPECIFIC"]
    general_damage = [c for c in configs if c["necessity_class"] == "GENERAL_DAMAGE"]
    diagnostic = [c for c in configs if c["necessity_class"] == "DIAGNOSTIC"]
    inconclusive = [c for c in configs if c["necessity_class"] in
                    ("INCONCLUSIVE", "NECESSARY_UNCLEAR_CAP")]

    if necessary_specific and addition_sufficient:
        verdict = "LEVER"
    elif necessary_specific and not addition_sufficient:
        verdict = "NECESSARY_BUT_NOT_SUFFICIENT"   # ablation specific, addition doesn't induce
    elif general_damage and not necessary_specific:
        verdict = "GENERAL_DAMAGE"
    elif diagnostic and not necessary_specific and not general_damage:
        verdict = "DIAGNOSTIC"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "FE269_causal_dom",
        "baselines": {"math_off": math_off, "gsm8k_off": gsm_off, "mmlu_off": mmlu_off},
        "gate_ok": (math_off is not None and 47.0 <= math_off <= 50.2),
        "ablation_configs": configs,
        "addition_runs": add_runs, "best_addition": best_add,
        "addition_sufficient": addition_sufficient,
        "verdict": verdict,
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\n-> {args.out}")
    print(f"\n=== FE269 VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
