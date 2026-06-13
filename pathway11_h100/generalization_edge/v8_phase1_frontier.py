#!/usr/bin/env python3
"""v8_phase1_frontier.py — SPEC v8 Phase 1 local adjudication (CPU).

Consumes pod_results/v8_frontier_{key}.json.gz (texts + gen-time features),
grades with the pinned harness grader (math_verify via pathway8), places every
arm on the matched-cost frontier, and adjudicates H-N / H-O / H-P / H-Q.

Cost units: token-FLOPs, 1 = one 1.5B token; 7B-class = 4.7/token.
Cascade replication is the v7 Phase-4 construction exactly: frozen 5-fold
outer-OOF, free gate (n_gen, mean_logprob) LR fit per train fold, escalate
the lowest-gate test problems within budget 0.5*c_esc. The escalation MASK is
target-independent, so target swaps (H-N) re-read the same mask with the new
target's y-vector. Anchor check: incumbent mask must reproduce 0.648.
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
import activation_loader as AL                        # noqa: E402
from metrics import frozen_folds                      # noqa: E402
from pathway8_layerwise.extract_math500 import check_correct  # noqa: E402

RNG = np.random.default_rng(1234)
B_BOOT = 2000
UNIT = {"math7b": 4.7, "mathstral": 4.7, "math1.5b": 1.0, "r1distill": 1.0}
CASCADE_ACC = 0.648
ORACLE_ACC = 0.758
BUDGET_TOTAL = 2220.7134      # c0 + 0.5*c_esc


def load_pod(key: str, suffix: str = ""):
    f = POD / f"v8_frontier_{key}{suffix}.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def answer_segment(text: str, key: str) -> str:
    """Pre-registered parse rule: r1distill answers come from the segment
    after the last </think> when present (else the full text); other models
    use the full text. Downstream, the pinned grader's boxed-extraction with
    final-number fallback applies; unparseable = wrong."""
    if key == "r1distill":
        idx = text.rfind("</think>")
        if idx != -1:
            return text[idx + len("</think>"):]
    return text


def grade(rows: dict, key: str):
    y, parse_fail = [], []
    for text, ans in zip(rows["text"], rows["answer"]):
        seg = answer_segment(text, key)
        parse_fail.append("\\boxed{" not in seg)
        y.append(bool(check_correct(seg, ans)))
    return np.array(y, dtype=bool), np.array(parse_fail, dtype=bool)


def fit_lr(X, y):
    sc = StandardScaler().fit(X)
    est = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: est.predict_proba(sc.transform(Z))[:, 1]


def cascade_mask(y1, X_free, c_esc, budget_extra):
    """v7 Phase-4 Cascade policy, outer-OOF. Mask is target-independent."""
    n = len(y1)
    mask = np.zeros(n, dtype=bool)
    for tr, te in frozen_folds(y1):
        g = fit_lr(X_free[tr], y1[tr])
        n_esc = int(budget_extra * len(te) // c_esc)
        s = g(X_free[te])
        m = np.zeros(len(te), dtype=bool)
        m[np.argsort(s)[:n_esc]] = True
        mask[te] = m
    return mask


def cascade_curve(y1, y_target, X_free, c0, c_esc, fracs):
    """(cost, acc) points of the gate cascade at escalation budgets f*c_esc."""
    pts = []
    for f in fracs:
        m = cascade_mask(y1, X_free, c_esc, f * c_esc)
        acc = float(np.where(m, y_target, y1).mean())
        pts.append({"esc_budget_frac": f,
                    "cost": c0 + float(m.mean()) * c_esc, "acc": acc})
    return pts


def paired_boot_p(y_new, y_ref):
    """One-sided paired bootstrap: P(delta <= 0), B=2000, RNG(1234)."""
    n = len(y_new)
    d = y_new.astype(float) - y_ref.astype(float)
    deltas = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(B_BOOT)])
    return float(np.mean(deltas <= 0)), float(d.mean()), float(deltas.std())


def oof_gate(X, y):
    sc = np.zeros(len(y))
    for tr, te in frozen_folds(y):
        g = fit_lr(X[tr], y[tr])
        sc[te] = g(X[te])
    return sc


def risk_coverage(scores, y, covs=(0.9, 0.8, 0.7)):
    order = np.argsort(-scores)
    out = {}
    for c in covs:
        k = int(np.ceil(c * len(y)))
        out[str(c)] = float(y[order[:k]].mean())
    return out


def length_coef_sign(X, y):
    sc = StandardScaler().fit(X)
    est = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return float(est.coef_[0][0])


def main():
    out = {"spec": "v8 Phase 1 (EXP-90) — frontier raisers H-N/H-O/H-P/H-Q",
           "cost_units": "token-FLOPs, 1 = one 1.5B token; 7B-class = 4.7"}

    # pinned incumbents
    m15 = AL.load_cell("qwen1.5b", "math")
    y1 = m15.y.astype(bool)
    y7 = np.load(RESULTS / "v7_rescore_qwen7b_math.npz",
                 allow_pickle=True)["y"].astype(bool)
    X_free = np.column_stack([m15.n_gen_tokens.astype(float),
                              m15.mean_logprob.astype(float)])
    ft = json.loads((RESULTS / "v7_frontier_table.json").read_text())
    c0 = ft["cells"]["qwen1.5b_math"]["c0_abs"]
    c_esc = ft["cells"]["qwen7b_math"]["c0_abs"]
    unresc = (~y1) & (~y7)
    assert unresc.sum() == 121 and (~y1).sum() == 257

    # anchor: incumbent cascade mask reproduces 0.648
    mask = cascade_mask(y1, X_free, c_esc, 0.5 * c_esc)
    y_casc_inc = np.where(mask, y7, y1)
    acc_inc = float(y_casc_inc.mean())
    out["anchor_cascade_incumbent"] = acc_inc
    assert abs(acc_inc - CASCADE_ACC) < 1e-9, f"cascade anchor {acc_inc}"

    # grade all pod arms
    arms = {}
    for key in ("math7b", "mathstral", "math1.5b", "r1distill"):
        rows = load_pod(key)
        if rows is None:
            print(f"  !! missing pod result for {key}; skipping")
            continue
        y, pf = grade(rows, key)
        n_gen = np.array(rows["n_gen"], dtype=float)
        hit_max = np.array(rows["hit_max"], dtype=bool)
        arms[key] = {"y": y, "parse_fail": pf, "n_gen": n_gen,
                     "hit_max": hit_max,
                     "mean_lp": np.array(rows["mean_lp"], dtype=float),
                     "rows": rows}
        arms[key]["summary"] = {
            "model": rows["model_name"], "decoding": rows["decoding"],
            "accuracy": float(y.mean()),
            "parse_failure_rate": float(pf.mean()),
            "hit_max_frac": float(hit_max.mean()),
            "mean_gen_tokens": float(n_gen.mean()),
            "mean_cost_flops": float(n_gen.mean()) * UNIT[key],
        }
        print(f"  {key}: acc {y.mean():.3f}, cost "
              f"{n_gen.mean() * UNIT[key]:.0f}, parse-fail {pf.mean():.3f}, "
              f"hit-max {hit_max.mean():.3f}")

    # ---- H-P pre-registered fallback check (r1distill greedy repetition) ----
    if "r1distill" in arms:
        a = arms["r1distill"]
        frac_bad = float((a["hit_max"] & a["parse_fail"]).mean())
        a["summary"]["hit_max_unparseable_frac"] = frac_bad
        use_fallback = frac_bad > 0.20
        a["summary"]["fallback_triggered"] = use_fallback
        if use_fallback:
            fb = load_pod("r1distill", "_t0.6_s9999")
            if fb is None:
                print("  !! r1distill fallback TRIGGERED but no fallback run "
                      "present — run v8_run_task.sh r1distill --temp 0.6")
            else:
                y, pf = grade(fb, "r1distill")
                arms["r1distill_fallback"] = {
                    "y": y, "parse_fail": pf,
                    "n_gen": np.array(fb["n_gen"], dtype=float),
                    "hit_max": np.array(fb["hit_max"], dtype=bool),
                    "mean_lp": np.array(fb["mean_lp"], dtype=float),
                    "rows": fb,
                    "summary": {"model": fb["model_name"],
                                "decoding": fb["decoding"],
                                "accuracy": float(y.mean()),
                                "parse_failure_rate": float(pf.mean()),
                                "mean_gen_tokens": float(np.mean(fb["n_gen"])),
                                "mean_cost_flops":
                                    float(np.mean(fb["n_gen"])) * 1.0}}

    # ---------------- H-N: target swap ----------------
    if "math7b" in arms:
        ym = arms["math7b"]["y"]
        y_casc_new = np.where(mask, ym, y1)
        p, delta, se = paired_boot_p(y_casc_new, y_casc_inc)
        new_oracle = float((y1 | ym).mean())
        resc_121 = int(ym[unresc].sum())
        both = int((ym & y7).sum())
        only_new = int((ym & ~y7).sum())
        only_inc = int((~ym & y7).sum())
        confirm = (delta >= 0.03) and (p < 0.05)
        refute = delta < 0.01
        out["H_N"] = {
            "target": arms["math7b"]["summary"]["model"],
            "target_standalone_acc": float(ym.mean()),
            "cascade_acc_new_target": float(y_casc_new.mean()),
            "cascade_acc_incumbent": acc_inc,
            "delta": delta, "boot_se": se, "p_one_sided": p,
            "new_oracle": new_oracle, "incumbent_oracle": ORACLE_ACC,
            "rescues_of_121_unrescuable": resc_121,
            "overlap_matrix": {"both_correct": both, "only_new": only_new,
                               "only_incumbent": only_inc,
                               "neither": 500 - both - only_new - only_inc},
            "verdict": ("CONFIRMED" if confirm else
                        "REFUTED" if refute else "INCONCLUSIVE"),
            "teacher_for_phase2": (arms["math7b"]["summary"]["model"]
                                   if confirm else
                                   "Qwen/Qwen2.5-7B-Instruct"),
        }
        print(f"  H-N: {out['H_N']['verdict']} (delta {delta*100:+.2f}pp, "
              f"p {p:.4f}, rescues {resc_121}/121)")

    # ---------------- H-O: family-diverse rescues ----------------
    if "mathstral" in arms:
        ys = arms["mathstral"]["y"]
        resc = int(ys[unresc].sum())
        verdict = ("CONFIRMED" if resc >= 25 else
                   "REFUTED" if resc < 0.10 * 121 else "INCONCLUSIVE")
        out["H_O"] = {"target": arms["mathstral"]["summary"]["model"],
                      "standalone_acc": float(ys.mean()),
                      "rescues_of_121_unrescuable": resc,
                      "confirm_threshold": 25,
                      "refute_threshold_lt": 0.10 * 121,
                      "verdict": verdict}
        print(f"  H-O: {verdict} ({resc}/121 disjoint rescues)")

    # ---------------- H-P / H-Q: base swaps ----------------
    curve = cascade_curve(y1, y7, X_free, c0, c_esc,
                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    out["incumbent_cascade_curve"] = curve

    out["H_P"], out["H_Q"] = {}, {}
    hp_arms = [k for k in ("math1.5b", "r1distill", "r1distill_fallback")
               if k in arms]
    for key in hp_arms:
        a = arms[key]
        y, n_gen = a["y"], a["n_gen"]
        cost = float(n_gen.mean()) * UNIT.get(key, 1.0)
        p, delta, se = paired_boot_p(y, y_casc_inc)
        within = cost <= BUDGET_TOTAL
        if within:
            verdict = ("CONFIRMED" if (delta > 0 and p < 0.05 and
                                       float(y.mean()) > CASCADE_ACC)
                       else "REFUTED")
            comparator = {"type": "matched_budget", "acc": acc_inc}
        else:
            cs = sorted(curve, key=lambda r: r["cost"])
            costs = [r["cost"] for r in cs]
            accs = [r["acc"] for r in cs]
            acc_eq = float(np.interp(cost, costs, accs))
            verdict = ("REFUTED" if float(y.mean()) <= acc_eq
                       else "CONFIRMED-AT-HIGHER-COST")
            comparator = {"type": "cost_equivalent_cascade", "acc": acc_eq}
        out["H_P"][key] = {**a["summary"], "delta_vs_cascade": delta,
                           "boot_se": se, "p_one_sided": p,
                           "within_budget": within,
                           "budget": BUDGET_TOTAL,
                           "comparator": comparator, "verdict": verdict}
        print(f"  H-P[{key}]: {verdict} (acc {y.mean():.3f}, "
              f"cost {cost:.0f}, budget {BUDGET_TOTAL:.0f})")

        # H-Q: frozen free-gate recipe on the arm's own outputs
        fin = np.isfinite(a["mean_lp"])
        Xa = np.column_stack([n_gen, np.where(fin, a["mean_lp"], 0.0)])
        if y.sum() in (0, len(y)):
            out["H_Q"][key] = {"auroc": None,
                               "note": "degenerate labels, gate unfittable"}
            continue
        sc = oof_gate(Xa, y)
        au = float(roc_auc_score(y, sc))
        # descriptive: cascade ON TOP of this base (escalate to incumbent 7B)
        m2 = cascade_mask(y, Xa, c_esc, 0.5 * c_esc)
        acc_on_top = float(np.where(m2, y7, y).mean())
        out["H_Q"][key] = {
            "oof_auroc": au, "pass_070": au >= 0.70,
            "length_coef_standardized": length_coef_sign(Xa, y),
            "risk_coverage": risk_coverage(sc, y),
            "cascade_on_top_acc_desc": acc_on_top,
            "base_alone_acc": float(y.mean()),
        }
        print(f"  H-Q[{key}]: AUROC {au:.4f} "
              f"({'PASS' if au >= 0.70 else 'FAIL'} >=0.70), "
              f"len-coef {out['H_Q'][key]['length_coef_standardized']:+.3f}")

    # H-Q roll-up over the two primary arms (fallback replaces greedy if used)
    prim = [k for k in ("math1.5b",
                        ("r1distill_fallback"
                         if "r1distill_fallback" in arms else "r1distill"))
            if k in out["H_Q"] and out["H_Q"][k].get("oof_auroc") is not None]
    passes = [out["H_Q"][k]["oof_auroc"] >= 0.70 for k in prim]
    out["H_Q"]["verdict"] = ("CONFIRMED" if len(passes) == 2 and all(passes)
                             else "PARTIAL" if any(passes) else "REFUTED")
    print(f"  H-Q roll-up: {out['H_Q']['verdict']}")

    out["arm_summaries"] = {k: a["summary"] for k, a in arms.items()}
    f = RESULTS / "v8_phase1_frontier.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
