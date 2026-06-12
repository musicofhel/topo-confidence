#!/usr/bin/env python3
"""v7_phase1b_costmatched.py — SPEC v7 Phase 1b (EXP-85): matched-cost frontier.

Cost axis: token-FLOP units. 1 unit = one token through the 1.5B model; a 7B
token costs 4.7 units. Per-problem pipeline cost sums prompt + generated
tokens of every model pass (prefill counted once per request; K samples of
one request share their prefill with the greedy pass).

Deployment cost accounting (pre-registered, SPEC v7 deviation D-2): the
token-logprob family (min/p10/entropy/margin/answer-span) reads off the
decoder's own logits at generation time, so its marginal deployment cost is
0 — the pod rescore pass was cache reconstruction, not a deployment cost.
P(True)/verbalized/K-sample/PRM arms pay real extra forwards, measured from
the pod token counts.

H-I: no arm adds >= +0.01 combined AUROC (DeLong p < 0.05) over the free
gate at <= 1.05x cell cost on dev. Adjudicated per dev cell.
H-J: K-majority(+agreement-ranked selective prediction) loses to the
1.5B->7B free-gate cascade on selective accuracy at every shared cost point
on MATH (paired bootstrap; K wins a point iff Delta > 1pp, p < 0.05).
Comparators: cascade curve + random-mix hull (FE19 lesson).

Inputs: pod_results/v7_{ptrue,verbalized}_*.json, v7_k8bbh_qwen1.5b.json.gz,
optional v7_prm_qwen1.5b_math.json, rescore sidecars in results/, the K=8
MATH cache, v7_phase1a.py helpers (identical LR/fold protocol).
Outputs: results/v7_phase1b_costmatched.json (full analysis) and
results/v7_frontier_table.json (the Phase-4 router's contracted menu).
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import activation_loader as AL                                   # noqa: E402
import k8_lib                                                    # noqa: E402
from metrics import frozen_folds, auroc_sym                      # noqa: E402
from v7_phase1a import arm_row, impute_median, oof_scores        # noqa: E402
from v7_runpod_arms import bbh_extract_answer                    # noqa: E402

RESULTS = HERE / "results"
POD = HERE / "pod_results"
F7B = 4.7        # token-FLOP multiplier, 7B vs 1.5B
PRM_MULT = 4.7   # Qwen2.5-Math-PRM-7B is 7B-class
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
BOOT = 2000
RNG_SEED = 0


def pod_json(name: str) -> dict:
    f = POD / name
    if name.endswith(".gz"):
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    return json.loads(f.read_text())


def payload_rows(name: str) -> list[dict]:
    rows = []
    with gzip.open(HERE / "pod_payload" / f"{name}.jsonl.gz", "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def selective_curve(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Risk-coverage: answer the top-c fraction by confidence."""
    order = np.argsort(-conf)
    out = {}
    for c in COVERAGES:
        m = max(1, int(round(c * len(conf))))
        out[f"{c:.1f}"] = float(correct[order[:m]].mean())
    return out


class DevCell:
    """Free-gate base + cost basis for one dev cell."""

    def __init__(self, model: str, cellname: str, mult: float):
        self.name = f"{model}_{cellname}"
        self.mult = mult
        c = AL.load_cell(model, cellname)
        self.y = c.y
        self.folds = frozen_folds(self.y)
        self.base_X = np.column_stack([c.n_gen_tokens, c.mean_logprob])
        self.free_oof = oof_scores(self.base_X, self.y, self.folds)
        self.free_auroc = auroc_sym(self.y, self.free_oof)
        side = np.load(RESULTS / f"v7_rescore_{model}_{cellname}.npz")
        assert bool((side["y"] == self.y).all())
        self.n_gen = side["n_gen"].astype(float)
        if "n_prompt" in side:
            self.n_prompt = side["n_prompt"].astype(float)
        else:
            # local MATH sidecar lacks n_prompt; Qwen2.5 1.5B/7B share the
            # tokenizer + chat template, so the 7B MATH counts are identical
            self.n_prompt = np.load(
                RESULTS / "v7_rescore_qwen7b_math.npz")["n_prompt"].astype(
                    float)
        self.c0 = float((self.n_prompt + self.n_gen).mean())   # cell units
        self.c0_abs = self.c0 * mult                           # 1.5B units

    def row(self, name, x, rel_extra_cell_units: float, note: str = ""):
        """arm_row + cost columns + system-view risk-coverage curve."""
        xi, cov = impute_median(np.asarray(x, dtype=np.float64))
        r = arm_row(name, xi, self.base_X, self.y, self.folds,
                    self.free_oof, self.free_auroc)
        s_alone = oof_scores(xi.reshape(-1, 1) if xi.ndim == 1 else xi,
                             self.y, self.folds)
        r["risk_coverage"] = selective_curve(s_alone, self.y)
        r["finite_fraction"] = cov
        r["rel_cost"] = 1.0 + rel_extra_cell_units / self.c0
        r["abs_cost_per_problem"] = (self.c0 + rel_extra_cell_units) * \
            self.mult
        if note:
            r["note"] = note
        return r

    def free_row(self) -> dict:
        return {"arm": "free_gate", "standalone_auroc": self.free_auroc,
                "combined_auroc": self.free_auroc,
                "free_auroc": self.free_auroc, "incremental_delta": 0.0,
                "risk_coverage": selective_curve(self.free_oof, self.y),
                "rel_cost": 1.0, "abs_cost_per_problem": self.c0_abs}


def token_arm_rows(cell: DevCell, model: str, cellname: str) -> list[dict]:
    side = np.load(RESULTS / f"v7_rescore_{model}_{cellname}.npz")
    arms = ("min_lp", "p10_lp", "mean_entropy", "mean_top2_margin",
            "ans_span_lp")
    rows = [cell.row(f"token_{a}", side[a], 0.0,
                     note="deployment-free: read off decoder logits")
            for a in arms]
    X = np.column_stack([impute_median(side[a].astype(float))[0]
                         for a in arms])
    rows.append(cell.row("token_family_joint", X, 0.0,
                         note="deployment-free joint"))
    return rows


def ptrue_rows(cell: DevCell, model: str, cellname: str) -> list[dict]:
    d = pod_json(f"v7_ptrue_{model}_{cellname}.json")
    assert list(map(bool, d["y"])) == list(map(bool, cell.y))
    rows = []
    for fmt in ("A", "B"):
        extra = float(np.mean(d[f"n_prompt_{fmt}"]))  # one forward, no decode
        rows.append(cell.row(f"ptrue_fmt{fmt}", d[f"p_true_{fmt}"], extra))
    return rows


def verbalized_rows(cell: DevCell, model: str, cellname: str) -> list[dict]:
    d = pod_json(f"v7_verbalized_{model}_{cellname}.json")
    assert list(map(bool, d["y"])) == list(map(bool, cell.y))
    extra = float(d["cost_tokens"]) / len(d["y"]) / 2.0   # per question
    rows = []
    for arm, key, pf in (("verb_binary", "verb_binary",
                          d["parse_failure_binary"]),
                         ("verb_score_0to100", "verb_score",
                          d["parse_failure_score"])):
        r = cell.row(arm, d[key], extra)
        r["parse_failure"] = pf
        rows.append(r)
    return rows


def k8_math_features() -> dict:
    """Per-K agreement/entropy/majority labels/sampled tokens (K=8 cache)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    n = AL.n_k8_problems()
    greedy_ans = []
    for i in range(n):
        d = np.load(AL.TRAJ_1P5B / f"problem_{i:03d}.npz", allow_pickle=True)
        greedy_ans.append(k8_lib.extract_answers([str(d["text"])])[0])
    out = {K: {"agreement": np.zeros(n), "entropy": np.zeros(n),
               "majority_correct": np.zeros(n, dtype=bool),
               "sampled_tokens": np.zeros(n)} for K in (2, 4, 8)}
    for i in range(n):
        k8 = AL.load_k8(i)
        answers = k8_lib.extract_answers(k8["texts"])
        ntok = [len(tok(str(t), add_special_tokens=False).input_ids)
                for t in k8["texts"]]
        for K in (2, 4, 8):
            sub = answers[:K]
            o = out[K]
            o["agreement"][i] = float(np.mean(
                [a == greedy_ans[i] and a != "" for a in sub]))
            _, counts = np.unique(sub, return_counts=True)
            p = counts / K
            o["entropy"][i] = float(-(p * np.log(p)).sum())
            o["majority_correct"][i] = k8_lib.plain_majority_correct(
                sub, list(k8["correct"][:K]))
            o["sampled_tokens"][i] = float(np.sum(ntok[:K]))
    return out


def k8_bbh_features(cell: DevCell) -> tuple[dict, dict]:
    """Same features from the pod K=8 BBH run; greedy answers re-extracted."""
    d = pod_json("v7_k8bbh_qwen1.5b.json.gz")
    rows = sorted(d["rows"], key=lambda r: r["idx"])
    n = len(rows)
    assert n == len(cell.y)
    payload = payload_rows("qwen1.5b_bbh")
    greedy_ans = [bbh_extract_answer(p["text"]) for p in payload]
    out = {K: {"agreement": np.zeros(n), "entropy": np.zeros(n),
               "majority_correct": np.zeros(n, dtype=bool),
               "sampled_tokens": np.zeros(n)} for K in (2, 4, 8)}
    for i, r in enumerate(rows):
        for K in (2, 4, 8):
            sub = r["answers"][:K]
            o = out[K]
            o["agreement"][i] = float(np.mean(
                [a == greedy_ans[i] and a != "" for a in sub]))
            _, counts = np.unique(sub, return_counts=True)
            p = counts / K
            o["entropy"][i] = float(-(p * np.log(p)).sum())
            nonempty = [a for a in sub if a != ""]
            mode = Counter(nonempty).most_common(1)[0][0] if nonempty else ""
            o["majority_correct"][i] = (mode == r["target"] and mode != "")
            o["sampled_tokens"][i] = float(np.sum(r["n_gen"][:K]))
    meta = {"K": d["K"], "temperature": d["temperature"], "seed": d["seed"],
            "cost_tokens_generated": d["cost_tokens_generated"]}
    return out, meta


def consistency_rows(cell: DevCell, feats: dict) -> list[dict]:
    rows = []
    for K in (2, 4, 8):
        o = feats[K]
        extra = float(o["sampled_tokens"].mean())   # prefill shared w/ greedy
        rows.append(cell.row(f"k{K}_agreement", o["agreement"], extra))
        rows.append(cell.row(f"k{K}_cluster_entropy", o["entropy"], extra))
    return rows


def prm_rows(cell: DevCell) -> list[dict]:
    f = POD / "v7_prm_qwen1.5b_math.json"
    if not f.exists():
        return []
    d = pod_json(f.name)
    assert list(map(bool, d["y"])) == list(map(bool, cell.y))
    extra = float(d["cost_tokens_scored"]) / len(d["y"]) * PRM_MULT
    return [cell.row(f"prm_{k}", d[f"prm_{k}"], extra,
                     note="7B-class PRM; honest comparator is escalation")
            for k in ("min", "mean", "last")]


def h_i(cells: dict) -> dict:
    """H-I per cell: any arm with rel_cost<=1.05, delta>=+0.01, p<0.05?"""
    verdicts = {}
    for cname, table in cells.items():
        winners = [r for r in table["arms"]
                   if r.get("delong_p") is not None
                   and r["rel_cost"] <= 1.05
                   and r["incremental_delta"] >= 0.01
                   and r["delong_p"] < 0.05]
        verdicts[cname] = {
            "refuted_by": [{k: w[k] for k in
                            ("arm", "incremental_delta", "delong_p",
                             "rel_cost")} for w in winners],
            "H_I_holds": not winners}
    verdicts["overall_H_I_holds"] = all(
        v["H_I_holds"] for v in verdicts.values() if isinstance(v, dict))
    return verdicts


def h_j(m15: DevCell, feats_math: dict, m7: DevCell) -> dict:
    """K-majority selective prediction vs cost-matched cascade (MATH).

    Shared cost points = the K-system's native costs. At each, the cascade
    escalates the least-confident (free-gate) fraction that matches cost.
    Selective accuracy at each coverage: K-system ranks by agreement,
    cascade by the answering model's own free-gate OOF score. Paired
    bootstrap over problems; K wins a point iff Delta > 1pp at p < 0.05.
    """
    rng = np.random.default_rng(RNG_SEED)
    y1, y7 = m15.y, m7.y
    n = len(y1)
    per_cost_15 = m15.n_prompt + m15.n_gen                  # 1.5B units
    per_cost_7 = (m7.n_prompt + m7.n_gen) * F7B
    base_cost = float(per_cost_15.mean())
    order = np.argsort(m15.free_oof)                        # least conf first

    # cascade curve (for the frontier + cost->f inversion)
    curve = []
    for m in range(0, n + 1, 5):
        esc = order[:m]
        acc = float((y7[esc].sum() + y1[order[m:]].sum()) / n)
        cost = base_cost + float(per_cost_7[esc].sum()) / n
        curve.append({"escalate_frac": m / n, "abs_cost": cost, "acc": acc})
    costs = np.array([p["abs_cost"] for p in curve])
    cmax = float(costs[-1])
    acc15, acc7 = float(y1.mean()), float(y7.mean())

    points, k_wins_anywhere = [], False
    for K in (2, 4, 8):
        o = feats_math[K]
        kcost = base_cost + float(o["sampled_tokens"].mean())
        # cascade matched to this cost (capped at full escalation)
        m = int(np.interp(min(kcost, cmax), costs,
                          [p["escalate_frac"] * n for p in curve]))
        esc_mask = np.zeros(n, dtype=bool)
        esc_mask[order[:m]] = True
        cas_correct = np.where(esc_mask, y7, y1)
        cas_conf = np.where(esc_mask, m7.free_oof, m15.free_oof)
        k_correct = o["majority_correct"]
        k_conf = o["agreement"]

        cov_rows = {}
        for c in COVERAGES:
            mm = max(1, int(round(c * n)))
            ik = np.argsort(-k_conf)[:mm]
            ic = np.argsort(-cas_conf)[:mm]
            dacc = float(k_correct[ik].mean() - cas_correct[ic].mean())
            # paired bootstrap over problems
            wins = 0
            for _ in range(BOOT):
                bs = rng.integers(0, n, n)
                kk, cc = k_correct[bs], cas_correct[bs]
                kcf, ccf = k_conf[bs], cas_conf[bs]
                d = (kk[np.argsort(-kcf)[:mm]].mean()
                     - cc[np.argsort(-ccf)[:mm]].mean())
                wins += d > 0.01
            p_win = 1.0 - wins / BOOT     # small p => K reliably wins >1pp
            cov_rows[f"{c:.1f}"] = {
                "k_selective_acc": float(k_correct[ik].mean()),
                "cascade_selective_acc": float(cas_correct[ic].mean()),
                "delta": dacc, "p_k_wins_gt_1pp": p_win}
            if dacc > 0.01 and p_win < 0.05:
                k_wins_anywhere = True
        rand = acc15 + (min(kcost, cmax) - base_cost) / (cmax - base_cost) \
            * (acc7 - acc15)
        points.append({"K": K, "abs_cost": kcost,
                       "cascade_escalate_frac": m / n,
                       "acc_k_majority_full_cov": float(k_correct.mean()),
                       "acc_cascade_full_cov": float(cas_correct.mean()),
                       "random_mix_acc_at_cost": rand,
                       "by_coverage": cov_rows})
    return {"acc_1p5b": acc15, "acc_7b": acc7,
            "base_cost_1p5b": base_cost, "full_escalation_cost": cmax,
            "cascade_curve": curve[::10],
            "k_points": points,
            "H_J_confirmed_K_loses_everywhere": not k_wins_anywhere}


def main():
    m15 = DevCell("qwen1.5b", "math", 1.0)
    b15 = DevCell("qwen1.5b", "bbh", 1.0)
    m7 = DevCell("qwen7b", "math", F7B)

    feats_math = k8_math_features()
    feats_bbh, k8bbh_meta = k8_bbh_features(b15)

    cells = {
        "qwen1.5b_math": {
            "free_gate_oof_auroc": m15.free_auroc, "c0_abs": m15.c0_abs,
            "arms": ([m15.free_row()]
                     + token_arm_rows(m15, "qwen1.5b", "math")
                     + ptrue_rows(m15, "qwen1.5b", "math")
                     + verbalized_rows(m15, "qwen1.5b", "math")
                     + consistency_rows(m15, feats_math)
                     + prm_rows(m15))},
        "qwen1.5b_bbh": {
            "free_gate_oof_auroc": b15.free_auroc, "c0_abs": b15.c0_abs,
            "arms": ([b15.free_row()]
                     + token_arm_rows(b15, "qwen1.5b", "bbh")
                     + ptrue_rows(b15, "qwen1.5b", "bbh")
                     + verbalized_rows(b15, "qwen1.5b", "bbh")
                     + consistency_rows(b15, feats_bbh))},
        "qwen7b_math": {
            "free_gate_oof_auroc": m7.free_auroc, "c0_abs": m7.c0_abs,
            "arms": ([m7.free_row()]
                     + token_arm_rows(m7, "qwen7b", "math")
                     + ptrue_rows(m7, "qwen7b", "math")
                     + verbalized_rows(m7, "qwen7b", "math"))}}

    # P(True) format freeze rule: better format on 1.5B MATH dev only
    pt = {r["arm"]: r["standalone_auroc"]
          for r in cells["qwen1.5b_math"]["arms"]
          if r["arm"].startswith("ptrue_")}
    frozen_fmt = max(pt, key=pt.get)

    out = {"spec": "v7 Phase 1b (EXP-85)",
           "cost_units": "token-FLOPs, 1 = one 1.5B token; 7B token = 4.7",
           "k8bbh_meta": k8bbh_meta,
           "cells": cells,
           "ptrue_format_frozen_for_T5": {
               "rule": "better standalone on 1.5B MATH dev only",
               "choice": frozen_fmt, "dev_aurocs": pt},
           "H_I": h_i(cells),
           "H_J_math": h_j(m15, feats_math, m7),
           "bbh_k8_system_view": {
               "acc_greedy": float(b15.y.mean()),
               **{f"acc_k{K}_majority":
                  float(feats_bbh[K]["majority_correct"].mean())
                  for K in (2, 4, 8)}}}

    f = RESULTS / "v7_phase1b_costmatched.json"
    f.write_text(json.dumps(out, indent=1, default=float))

    # the contracted Phase-4 router menu
    menu = {"cost_units": out["cost_units"],
            "cells": {cn: {"c0_abs": t["c0_abs"],
                           "free_gate_oof_auroc": t["free_gate_oof_auroc"],
                           "arms": [{k: r.get(k) for k in
                                     ("arm", "rel_cost",
                                      "abs_cost_per_problem",
                                      "standalone_auroc", "combined_auroc",
                                      "incremental_delta", "delong_p",
                                      "risk_coverage", "parse_failure")}
                                    for r in t["arms"]]}
                      for cn, t in cells.items()},
            "cascade_curve_math": out["H_J_math"]["cascade_curve"],
            "k_points_math": out["H_J_math"]["k_points"]}
    f2 = RESULTS / "v7_frontier_table.json"
    f2.write_text(json.dumps(menu, indent=1, default=float))
    print(f"-> {f}\n-> {f2}\n")

    for cname, t in cells.items():
        print(f"{cname}: free={t['free_gate_oof_auroc']:.4f} "
              f"c0={t['c0_abs']:.0f}")
        for r in t["arms"]:
            if r["arm"] == "free_gate":
                continue
            pf = f" pf={r['parse_failure']:.2f}" if "parse_failure" in r \
                else ""
            print(f"  {r['arm']:24s} x{r['rel_cost']:5.2f} "
                  f"alone={r['standalone_auroc']:.4f} "
                  f"comb={r['combined_auroc']:.4f} "
                  f"d={r['incremental_delta']:+.4f} p={r['delong_p']:.3f}"
                  f"{pf}")
    print(f"\nptrue format frozen for T5: {frozen_fmt} ({pt})")
    print(f"H-I holds overall: {out['H_I']['overall_H_I_holds']}")
    for cname in cells:
        v = out["H_I"][cname]
        if not v["H_I_holds"]:
            print(f"  REFUTED in {cname}: {v['refuted_by']}")
    hj = out["H_J_math"]
    print(f"\nH-J (MATH): 1.5B={hj['acc_1p5b']:.3f} 7B={hj['acc_7b']:.3f} "
          f"K-loses-everywhere={hj['H_J_confirmed_K_loses_everywhere']}")
    for p in hj["k_points"]:
        full = p["by_coverage"]["1.0"]
        print(f"  K={p['K']}: cost={p['abs_cost']:.0f} "
              f"(cascade f={p['cascade_escalate_frac']:.2f}) "
              f"maj={p['acc_k_majority_full_cov']:.3f} "
              f"cas={p['acc_cascade_full_cov']:.3f} "
              f"rand={p['random_mix_acc_at_cost']:.3f} "
              f"d@1.0={full['delta']:+.3f} p={full['p_k_wins_gt_1pp']:.3f}")
    print(f"\nBBH K=8 system view: {out['bbh_k8_system_view']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
