#!/usr/bin/env python3
"""v7_phase2_pseudolabel.py — SPEC v7 Phase 2 (EXP-87): label-free adaptation.

Pseudo-labels (zero true labels):
  BBH @1.5B: y-hat_K(i) = 1 iff the greedy answer equals the modal answer of
  the first K in {4,8} T=0.7 samples (pod k8bbh run). Dev = tracking +
  logical_deduction; confirmatory = web_of_lies (pre-pinned).
  Cross-scale (H-L3): y-hat7(i) = 1 iff the 7B greedy MATH answer equals the
  1.5B K=8 modal answer (the only label-free cross-scale proxy available;
  its noise rate vs truth is reported, guard 2).

Hypotheses (all adjudicated on TRUE labels — guard 1):
  H-L1 threshold transfer: tau chosen on pseudo-labels hits target coverage
    within +/-5pp and answered-accuracy within 2pp of tau-from-32-TRUE-labels
    (stratified, R=200, mirroring phase4_recalibration). Targets =
    greedy_acc(subset) + {10pp, 20pp} answered-accuracy improvements
    (pre-registered here before the k8bbh results existed).
  H-L2 refit no-harm: 2-D head refit on dev-subset pseudo-labels scores
    true-label AUROC on web_of_lies >= frozen zero-shot gate - 0.01.
  H-L3 conformal (cross-scale cell ONLY, per Phase-0 resolution): pseudo
    calibration achieves validity >= 0.9 at every (eps, k in {32,64}) cell
    where TRUE-label calibration is feasible with validity >= 0.9.

Guards: (2) confusion matrix vs truth, noise rate, corr(pseudo-error, gate
score), coverage-bias note; (3) K=4 vs K=8 monotonicity on H-L1/H-L2.

Inputs: pod_results/v7_k8bbh_qwen1.5b.json.gz, pod_payload/*.jsonl.gz,
the K=8 MATH cache, npz cells. Output: results/v7_phase2_pseudolabel.json.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import activation_loader as AL                                   # noqa: E402
import k8_lib                                                    # noqa: E402
from metrics import auroc_sym, SEED                              # noqa: E402
from exp_1b3_conformal import choose_tau                         # noqa: E402
from phase4_recalibration import (_stratified_support,           # noqa: E402
                                  EPS_LIST, DELTA)
from v7_runpod_arms import bbh_extract_answer                    # noqa: E402

RESULTS = HERE / "results"
POD = HERE / "pod_results"
R = 200
DEV_SUBSETS = ["tracking_shuffled_objects_seven_objects",
               "logical_deduction_seven_objects"]
CONF_SUBSET = "web_of_lies"
K_CONFORMAL = [32, 64]
MIN_ANSWERED = 5


def payload_rows(name: str) -> list[dict]:
    rows = []
    with gzip.open(HERE / "pod_payload" / f"{name}.jsonl.gz", "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def modal(answers: list[str]) -> str:
    nonempty = [a for a in answers if a != ""]
    return Counter(nonempty).most_common(1)[0][0] if nonempty else ""


def frozen_gate(src_cell, tgt_cell) -> np.ndarray:
    """MATH-fit free gate applied to the target cell (the 0.785 transfer)."""
    Xs = np.column_stack([src_cell.n_gen_tokens, src_cell.mean_logprob])
    Xt = np.column_stack([tgt_cell.n_gen_tokens, tgt_cell.mean_logprob])
    sc = StandardScaler().fit(Xs)
    est = LogisticRegression(max_iter=2000).fit(sc.transform(Xs), src_cell.y)
    return est.predict_proba(sc.transform(Xt))[:, 1]


def tau_for_target_acc(scores, labels, target):
    """Smallest threshold whose empirical answered-accuracy >= target
    (>= MIN_ANSWERED answered). None if unreachable."""
    order = np.argsort(-scores)
    best = None
    for m in range(MIN_ANSWERED, len(scores) + 1):
        idx = order[:m]
        if labels[idx].mean() >= target:
            best = float(scores[order[m - 1]])
    return best


def realized(scores, y_true, tau):
    ans = scores >= tau
    if ans.sum() == 0:
        return {"coverage": 0.0, "answered_acc": None}
    return {"coverage": float(ans.mean()),
            "answered_acc": float(y_true[ans].mean())}


def h_l1_subset(s, y_true, pseudo: dict, subset: str) -> dict:
    """Threshold transfer on one subset; pseudo = {K: y_hat array}."""
    rng = np.random.default_rng(SEED)
    base = float(y_true.mean())
    out = {"greedy_acc": base, "targets": {}}
    for bump in (0.10, 0.20):
        target = base + bump
        row = {"target_answered_acc": target}
        # reference: tau from 32 TRUE labels, R resamples, scored on remainder
        covs, accs, feas = [], [], 0
        for _ in range(R):
            sup, rem = _stratified_support(y_true, 32, rng)
            tau = tau_for_target_acc(s[sup], y_true[sup], target)
            if tau is None:
                continue
            feas += 1
            r = realized(s[rem], y_true[rem], tau)
            if r["answered_acc"] is not None:
                covs.append(r["coverage"])
                accs.append(r["answered_acc"])
        row["true32"] = {"feasible_frac": feas / R,
                         "coverage": float(np.mean(covs)) if covs else None,
                         "answered_acc": float(np.mean(accs)) if accs
                         else None}
        for K, yhat in pseudo.items():
            tau = tau_for_target_acc(s, yhat, target)
            if tau is None:
                row[f"pseudo_k{K}"] = {"tau": None, "feasible": False}
                continue
            r = realized(s, y_true, tau)
            ok = (row["true32"]["coverage"] is not None
                  and r["answered_acc"] is not None
                  and abs(r["coverage"] - row["true32"]["coverage"]) <= 0.05
                  and abs(r["answered_acc"]
                          - row["true32"]["answered_acc"]) <= 0.02)
            row[f"pseudo_k{K}"] = {"tau": tau, "feasible": True, **r,
                                   "within_tolerance": bool(ok)}
        out["targets"][f"+{int(bump*100)}pp"] = row
    return out


def guard_confusion(yhat, y_true, s) -> dict:
    tp = int((yhat & y_true).sum())
    fp = int((yhat & ~y_true).sum())
    fn = int((~yhat & y_true).sum())
    tn = int((~yhat & ~y_true).sum())
    err = yhat != y_true
    med = float(np.median(s))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "noise_rate": float(err.mean()),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "corr_pseudoerror_gatescore":
                float(np.corrcoef(err.astype(float), s)[0, 1]),
            "error_rate_above_median_score": float(err[s >= med].mean()),
            "error_rate_below_median_score": float(err[s < med].mean())}


def h_l3_crossscale(math_cell, m7_cell) -> dict:
    """Conformal recalibration with cross-model pseudo-labels (1.5B K=8
    majority as pseudo-truth for 7B greedy), protocol mirrors
    phase4_recalibration.recalibrate_free at k in {32,64}."""
    n = AL.n_k8_problems()
    modal_15 = []
    for i in range(n):
        k8 = AL.load_k8(i)
        modal_15.append(modal(k8_lib.extract_answers(k8["texts"])))
    rows7 = payload_rows("qwen7b_math")
    ans7 = k8_lib.extract_answers([r["text"] for r in rows7])
    yhat7 = np.array([a == m and a != "" for a, m in zip(ans7, modal_15)])
    y7 = m7_cell.y
    s = frozen_gate(math_cell, m7_cell)

    guard = guard_confusion(yhat7, y7, s)
    rng = np.random.default_rng(SEED)
    cells = {}
    for k in K_CONFORMAL:
        for eps in EPS_LIST:
            tcov, tval, tfeas = [], [], 0
            pcov, pval, pfeas = [], [], 0
            for _ in range(R):
                sup, rem = _stratified_support(y7, k, rng)
                yr = y7[rem]
                tau_t = choose_tau(s[sup], y7[sup], eps, DELTA)
                if tau_t is not None:
                    tfeas += 1
                    a = s[rem] >= tau_t
                    if a.sum():
                        tcov.append(float(a.mean()))
                        tval.append((1 - yr[a].mean()) <= eps)
                tau_p = choose_tau(s[sup], yhat7[sup], eps, DELTA)
                if tau_p is not None:
                    pfeas += 1
                    a = s[rem] >= tau_p
                    if a.sum():
                        pcov.append(float(a.mean()))
                        pval.append((1 - yr[a].mean()) <= eps)
            cells[f"k{k}_eps{eps}"] = {
                "true": {"feasible_frac": tfeas / R,
                         "coverage": float(np.mean(tcov)) if tcov else 0.0,
                         "validity": float(np.mean(tval)) if tval else None},
                "pseudo": {"feasible_frac": pfeas / R,
                           "coverage": float(np.mean(pcov)) if pcov else 0.0,
                           "validity": float(np.mean(pval)) if pval
                           else None}}
    live = {k: v for k, v in cells.items()
            if v["true"]["feasible_frac"] > 0
            and v["true"]["validity"] is not None
            and v["true"]["validity"] >= 0.9}
    holds = bool(live) and all(
        v["pseudo"]["validity"] is not None and v["pseudo"]["validity"] >= 0.9
        for v in live.values())
    return {"pseudo_label_guard": guard, "cells": cells,
            "live_cells": sorted(live),
            "H_L3_holds": holds}


def main():
    math = AL.load_cell("qwen1.5b", "math")
    bbh = AL.load_cell("qwen1.5b", "bbh")
    m7 = AL.load_cell("qwen7b", "math")

    # --- BBH pseudo-labels from the pod K=8 run ---
    with gzip.open(POD / "v7_k8bbh_qwen1.5b.json.gz", "rt") as f:
        k8 = json.load(f)
    rows = sorted(k8["rows"], key=lambda r: r["idx"])
    payload = payload_rows("qwen1.5b_bbh")
    assert [r["idx"] for r in rows] == [p["idx"] for p in payload]
    greedy = [bbh_extract_answer(p["text"]) for p in payload]
    subsets = np.array([p["subset"] for p in payload])
    yhat = {K: np.array([greedy[i] == modal(r["answers"][:K])
                         and greedy[i] != ""
                         for i, r in enumerate(rows)]) for K in (4, 8)}
    s_bbh = frozen_gate(math, bbh)
    frozen_auroc_pooled = auroc_sym(bbh.y, s_bbh)

    out = {"spec": "v7 Phase 2 (EXP-87)",
           "frozen_gate_bbh_auroc_pooled": frozen_auroc_pooled,
           "guards": {}, "H_L1": {}, "H_L2": {}, "monotonicity": {}}

    # guard 2 per subset/K
    for K in (4, 8):
        for sub in DEV_SUBSETS + [CONF_SUBSET]:
            m = subsets == sub
            out["guards"][f"k{K}_{sub}"] = guard_confusion(
                yhat[K][m], bbh.y[m], s_bbh[m])

    # --- H-L1 per subset ---
    for sub in DEV_SUBSETS + [CONF_SUBSET]:
        m = subsets == sub
        out["H_L1"][sub] = h_l1_subset(
            s_bbh[m], bbh.y[m], {K: yhat[K][m] for K in (4, 8)}, sub)
    conf_rows = out["H_L1"][CONF_SUBSET]["targets"]
    l1_holds = all(r.get("pseudo_k8", {}).get("within_tolerance", False)
                   for r in conf_rows.values()
                   if r["true32"]["coverage"] is not None)
    out["H_L1"]["H_L1_holds_confirmatory_k8"] = bool(l1_holds)

    # --- H-L2: refit on dev pseudo-labels, true AUROC on web_of_lies ---
    Xb = np.column_stack([bbh.n_gen_tokens, bbh.mean_logprob])
    dev_m = np.isin(subsets, DEV_SUBSETS)
    conf_m = subsets == CONF_SUBSET
    frozen_conf_auroc = auroc_sym(bbh.y[conf_m], s_bbh[conf_m])
    for K in (4, 8):
        sc = StandardScaler().fit(Xb[dev_m])
        est = LogisticRegression(max_iter=2000).fit(
            sc.transform(Xb[dev_m]), yhat[K][dev_m])
        s_refit = est.predict_proba(sc.transform(Xb[conf_m]))[:, 1]
        a = auroc_sym(bbh.y[conf_m], s_refit)
        out["H_L2"][f"k{K}"] = {
            "refit_true_auroc_web_of_lies": a,
            "frozen_true_auroc_web_of_lies": frozen_conf_auroc,
            "no_harm": bool(a >= frozen_conf_auroc - 0.01)}
    out["H_L2"]["H_L2_holds_k8"] = out["H_L2"]["k8"]["no_harm"]

    # guard 3: monotonicity in K (H-L2 AUROC and H-L1 tolerance hits)
    out["monotonicity"] = {
        "H_L2_auroc_k4_le_k8": bool(
            out["H_L2"]["k4"]["refit_true_auroc_web_of_lies"]
            <= out["H_L2"]["k8"]["refit_true_auroc_web_of_lies"] + 1e-9),
        "note": "true-label benefit must not decrease from K=4 to K=8"}

    # --- H-L3 cross-scale only ---
    out["H_L3_crossscale"] = h_l3_crossscale(math, m7)

    f = RESULTS / "v7_phase2_pseudolabel.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {f}\n")
    print(f"frozen gate BBH pooled AUROC: {frozen_auroc_pooled:.4f} "
          f"(anchor 0.785)")
    for sub in DEV_SUBSETS + [CONF_SUBSET]:
        g = out["guards"][f"k8_{sub}"]
        print(f"  {sub}: noise={g['noise_rate']:.3f} "
              f"prec={g['precision']:.3f} rec={g['recall']:.3f} "
              f"corr(err,s)={g['corr_pseudoerror_gatescore']:+.3f}")
    print(f"\nH-L1 holds (confirmatory, K=8): "
          f"{out['H_L1']['H_L1_holds_confirmatory_k8']}")
    for tname, r in conf_rows.items():
        t32, p8 = r["true32"], r.get("pseudo_k8", {})
        print(f"  {tname}: true32 cov={t32['coverage']} "
              f"acc={t32['answered_acc']} | pseudo8 {p8}")
    print(f"H-L2 no-harm (K=8): {out['H_L2']['H_L2_holds_k8']} "
          f"({out['H_L2']['k8']})")
    print(f"H-L3 holds: {out['H_L3_crossscale']['H_L3_holds']} "
          f"live cells: {out['H_L3_crossscale']['live_cells']}")
    print(f"  7B pseudo-label guard: "
          f"{out['H_L3_crossscale']['pseudo_label_guard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
