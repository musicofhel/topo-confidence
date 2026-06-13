#!/usr/bin/env python3
"""v8_phase2_distill.py — SPEC v8 Phase 2 H-R adjudication (CPU).

Consumes the four SFT'd-arm evals + the un-adapted base reference, grades
MATH-500 with the pinned grader, and adjudicates H-R (confidence-curated
distillation) against the pre-registered decision rules:

  CONFIRM : b >= c+1.5pp AND b >= d+1pp AND b >= a-1pp AND b >= base+3pp
  PARTIAL : b >= c+1.5pp but b < d+1pp   (length-artifact verdict)
  REFUTE  : b < c+0.5pp                   (filter adds nothing over volume)
  else    : INCONCLUSIVE

Guardrails (per arm; a failure means that arm cannot re-pin anything):
  - BBH-750 greedy >= base_bbh - 1pp                 (no catastrophic forgetting)
  - post-SFT free-gate OOF AUROC >= 0.849 - 0.02     (confidence signal survives)

Inputs:
  results/v8_filter_arms_report.json          (N, gate-vs-truth confusion)
  pod_results/v8_evalsft_{a,b,c,d}_math.json.gz
  pod_results/v8_evalsft_{a,b,c,d}_bbh.json.gz
  pod_results/v8_evalbase_math.json.gz / v8_evalbase_bbh.json.gz
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
POD = HERE / "pod_results"
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/musicofhel/topo-confidence")
from metrics import frozen_folds                                # noqa: E402
from pathway8_layerwise.extract_math500 import check_correct    # noqa: E402

BASE_MATH_PIN = 0.486          # pinned Qwen2.5-1.5B-Instruct MATH-500 K=1
GATE_DEV_AUROC = 0.849         # dev free-gate anchor; floor = anchor - 0.02
ARMS = ("a", "b", "c", "d")


def load_gz(path: Path):
    if not path.exists():
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def grade_math(rows) -> np.ndarray:
    return np.array([bool(check_correct(t, a))
                     for t, a in zip(rows["text"], rows["answer"])], bool)


def gate_features(rows) -> np.ndarray:
    return np.column_stack([np.array(rows["n_gen"], float),
                            np.nan_to_num(np.array(rows["mean_lp"], float))])


def oof_auroc(X, y) -> float:
    if y.sum() in (0, len(y)):
        return float("nan")
    sc = np.zeros(len(y))
    for tr, te in frozen_folds(y):
        s = StandardScaler().fit(X[tr])
        est = LogisticRegression(C=1.0, max_iter=2000).fit(s.transform(X[tr]),
                                                           y[tr])
        sc[te] = est.predict_proba(s.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, sc))


def main():
    report_path = RESULTS / "v8_filter_arms_report.json"
    filt = json.loads(report_path.read_text()) if report_path.exists() else {}

    # base reference (un-adapted)
    base_m = load_gz(POD / "v8_evalbase_math.json.gz")
    base_b = load_gz(POD / "v8_evalbase_bbh.json.gz")
    base_math = float(grade_math(base_m).mean()) if base_m else BASE_MATH_PIN
    base_bbh = float(np.mean(base_b["correct"])) if base_b else None
    if base_m:
        print(f"  base MATH-500 (reconfirm 0.486): {base_math:.4f}")
    if base_b:
        print(f"  base BBH-750: {base_bbh:.4f}")

    out = {"spec": "v8 Phase 2 (EXP-91) — H-R confidence-curated distillation",
           "base_math_acc": base_math, "base_math_pin": BASE_MATH_PIN,
           "base_bbh_acc": base_bbh,
           "matched_N": filt.get("matched_N"),
           "gate_vs_truth": filt.get("gate_vs_truth"),
           "filter_arms_report": filt.get("arms"), "arms": {}}

    acc, bbh, gauroc = {}, {}, {}
    for arm in ARMS:
        m = load_gz(POD / f"v8_evalsft_{arm}_math.json.gz")
        b = load_gz(POD / f"v8_evalsft_{arm}_bbh.json.gz")
        if m is None:
            print(f"  !! missing evalsft for arm {arm}; skipping")
            continue
        y = grade_math(m)
        acc[arm] = float(y.mean())
        au = oof_auroc(gate_features(m), y)
        gauroc[arm] = au
        bbh[arm] = float(np.mean(b["correct"])) if b else None
        out["arms"][arm] = {
            "math_acc": acc[arm], "bbh_acc": bbh[arm],
            "postsft_gate_oof_auroc": au,
            "gate_guardrail_pass": (au >= GATE_DEV_AUROC - 0.02),
            "bbh_guardrail_pass": (bbh[arm] is None or base_bbh is None or
                                   bbh[arm] >= base_bbh - 0.01),
            "mean_gen_tokens": float(np.mean(m["n_gen"]))}
        print(f"  arm {arm}: MATH {acc[arm]:.4f}  BBH "
              f"{(bbh[arm] if bbh[arm] is not None else float('nan')):.4f}  "
              f"gate-AUROC {au:.4f}")

    # ---------------- H-R verdict (needs at least b and c) ----------------
    if "b" in acc and "c" in acc:
        b_, c_ = acc["b"], acc["c"]
        d_ = acc.get("d")
        a_ = acc.get("a")
        beats_vol = b_ >= c_ + 0.015
        beats_len = (d_ is None) or (b_ >= d_ + 0.01)
        near_skyline = (a_ is None) or (b_ >= a_ - 0.01)
        beats_base = b_ >= base_math + 0.03
        if beats_vol and beats_len and near_skyline and beats_base:
            verdict = "CONFIRMED"
        elif beats_vol and not beats_len:
            verdict = "PARTIAL-length-artifact"
        elif b_ < c_ + 0.005:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"
        # guardrail veto: the winning arm (b) cannot re-pin if a guardrail fails
        b_guard = (out["arms"]["b"]["gate_guardrail_pass"] and
                   out["arms"]["b"]["bbh_guardrail_pass"])
        out["H_R"] = {
            "verdict": verdict, "winning_arm_guardrails_pass": b_guard,
            "acc": {"a": a_, "b": b_, "c": c_, "d": d_},
            "deltas": {"b_minus_c": (b_ - c_), "b_minus_d":
                       (None if d_ is None else b_ - d_),
                       "b_minus_a": (None if a_ is None else b_ - a_),
                       "b_minus_base": b_ - base_math},
            "rules": {"beats_volume_c+1.5pp": beats_vol,
                      "beats_length_d+1pp": beats_len,
                      "near_skyline_a-1pp": near_skyline,
                      "beats_base+3pp": beats_base}}
        print(f"  H-R: {verdict} (b={b_:.3f} c={c_:.3f} "
              f"d={d_ if d_ is None else round(d_,3)} "
              f"a={a_ if a_ is None else round(a_,3)} base={base_math:.3f}; "
              f"winning-arm guardrails {'PASS' if b_guard else 'FAIL'})")

    f = RESULTS / "v8_phase2_distill.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
