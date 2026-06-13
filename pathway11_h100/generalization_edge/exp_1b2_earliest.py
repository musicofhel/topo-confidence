#!/usr/bin/env python3
"""Phase 1B.2 — earliest-decision curve (prefix-truncated probes).

On the full Qwen-1.5B (29,T,1536) MATH cache: for each prefix length m, compute
the L19 mean-pooled DoM and the CoE depth-grid readout RESTRICTED to the first m
generated tokens, OOF (frozen folds). Deliverable: AUROC(m) — when does prefill
(0.77) become full-gen (0.85)?

Pre-registered trigger: if some readout reaches >=0.82 by m<=128, simulate
abort-and-escalate (draft on 1.5B to m tokens, escalate flagged problems to 7B)
and adjudicate vs the random-mix hull AND the FU2 hybrid frontier at matched
accuracy. Cost of a drafted-then-escalated problem = (m/median_full)*1 + 5.

Output: results/1b2_earliest.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

import activation_loader as AL
import probes as PR
from metrics import oof_dom_scores, frozen_folds, auroc_sym, SEED

M_GRID = [25, 50, 100, 200, 400, None]   # None = full generation
L19 = 19


def collect_prefix_features(model="qwen1.5b"):
    """One pass over the trajectory cache: per-problem, per-m, L19 prefix-mean
    and all-layer prefix-mean (for CoE). Returns dicts keyed by m."""
    man = json.loads((AL.TRAJ_1P5B / "manifest.json").read_text())
    n = man["n_problems"]
    H = 1536
    L19_feat = {m: np.zeros((n, H)) for m in M_GRID}
    coe_feat = {m: np.zeros((n, 29, H)) for m in M_GRID}
    y = np.zeros(n, dtype=bool)
    Tlens = np.zeros(n, dtype=int)
    for i in range(n):
        states, corr, _ = AL.load_trajectory(model, i)   # (29,T,H)
        y[i] = corr
        T = states.shape[1]
        Tlens[i] = T
        for m in M_GRID:
            mm = T if (m is None or m >= T) else m
            pm = states[:, :mm, :].mean(axis=1)          # (29,H)
            coe_feat[m][i] = pm
            L19_feat[m][i] = pm[L19]
        if (i + 1) % 100 == 0:
            print(f"  loaded {i+1}/{n}")
    return L19_feat, coe_feat, y, Tlens


def main():
    L19_feat, coe_feat, y, Tlens = collect_prefix_features()
    folds = frozen_folds(y)
    median_full = int(np.median(Tlens))

    curve = {}
    for m in M_GRID:
        key = "full" if m is None else str(m)
        # L19 mean-pool DoM
        s_dom = oof_dom_scores(L19_feat[m], y)
        a_dom = auroc_sym(y, s_dom)
        # CoE depth-grid (the portable arm-2 readout) over the prefix
        Xcoe = PR.coe_depthgrid_features(coe_feat[m], n_grid=16)
        s_coe = np.zeros(len(y))
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        for tr, te in folds:
            sc = StandardScaler().fit(Xcoe[tr])
            lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xcoe[tr]), y[tr])
            s_coe[te] = lr.predict_proba(sc.transform(Xcoe[te]))[:, 1]
        a_coe = auroc_sym(y, s_coe)
        curve[key] = {"m": (median_full if m is None else m),
                      "L19_meanpool_dom": float(a_dom),
                      "coe_depthgrid": float(a_coe),
                      "_dom_scores": s_dom.tolist()}
        print(f"  m={key:>4}: L19-DoM={a_dom:.4f}  CoE-depthgrid={a_coe:.4f}")

    # prefill anchor (m=0, the pre-flight gate)
    m15 = AL.load_cell("qwen1.5b", "math")
    s_pf = oof_dom_scores(m15.X("prefill"), y)
    a_pf = auroc_sym(y, s_pf)
    print(f"  m=   0 (prefill): L19-DoM={a_pf:.4f}")

    # ---- pre-registered trigger ----
    best_early = max((curve[k]["L19_meanpool_dom"], curve[k]["coe_depthgrid"],
                      curve[k]["m"], k) for k in curve if curve[k]["m"] <= 128)
    early_best_auroc = max(curve[k]["L19_meanpool_dom"] for k in curve
                           if curve[k]["m"] <= 128)
    triggered = early_best_auroc >= 0.82
    full_auroc = curve["full"]["L19_meanpool_dom"]
    frac_gain = ((early_best_auroc - a_pf) / (full_auroc - a_pf)
                 if full_auroc > a_pf else float("nan"))

    out = {
        "prefill_auroc": float(a_pf),
        "median_full_tokens": median_full,
        "curve": {k: {kk: vv for kk, vv in v.items() if kk != "_dom_scores"}
                  for k, v in curve.items()},
        "H_G_fraction_gain_by_m128": float(frac_gain),
        "trigger_threshold": 0.82,
        "trigger_fired_by_m128": bool(triggered),
        "best_early_auroc_m_le_128": float(early_best_auroc),
    }

    # ---- abort-and-escalate sim (only if triggered, else report counterfactual) ----
    m7 = AL.load_cell("qwen7b", "math")
    y7 = m7.y
    # pick the readout/m: best early if triggered, else full as reference
    if triggered:
        m_star = min(k for k in curve if curve[k]["m"] <= 128
                     and curve[k]["L19_meanpool_dom"] == early_best_auroc)
    else:
        m_star = "full"
    s_star = np.array(curve[m_star]["_dom_scores"])
    m_tokens = curve[m_star]["m"]
    # cascade: keep highest-score on 1.5B (full draft cost ~1), escalate rest to
    # 7B but having paid the partial draft (m_tokens/median_full) first.
    order = np.argsort(-s_star)
    n = len(y)
    draft_frac = m_tokens / median_full
    rows = []
    for k in np.linspace(0, n, 51).astype(int):
        keep = order[:k]; esc = order[k:]
        acc = (y[keep].sum() + y7[esc].sum()) / n
        pct_kept = k / n
        # kept: full 1.5B draft = 1; escalated: partial draft + 7B = draft_frac + 5
        cost = pct_kept * 1.0 + (1 - pct_kept) * (draft_frac + 5.0)
        cost_rel = cost / 5.0
        rows.append({"pct_kept": float(pct_kept), "accuracy": float(acc),
                     "cost_rel": float(cost_rel)})
    # compare to random-mix hull at matched cost_rel
    acc15, acc7 = float(y.mean()), float(y7.mean())
    for r in rows:
        # invert cost_rel -> equivalent random keep frac for the SAME cost model
        # random: cost_rel = [p*1 + (1-p)*(draft_frac+5)]/5
        # solve p: cost_rel*5 = p + (1-p)(draft_frac+5)
        c = r["cost_rel"] * 5.0
        denom = (1.0 - (draft_frac + 5.0))
        p = (c - (draft_frac + 5.0)) / denom if denom != 0 else 1.0
        p = float(np.clip(p, 0, 1))
        r["hull_acc"] = p * acc15 + (1 - p) * acc7
        r["gap_pp"] = 100 * (r["accuracy"] - r["hull_acc"])
    best_pt = max((r for r in rows if 0 < r["pct_kept"] < 1), key=lambda r: r["gap_pp"])
    out["abort_escalate_sim"] = {
        "readout": f"L19-meanpool-DoM @ m={m_tokens}",
        "draft_frac_of_median": float(draft_frac),
        "best_point": best_pt,
        "note": ("trigger fired" if triggered else
                 "trigger NOT fired (no readout >=0.82 by m<=128); sim uses full-gen "
                 "readout as a reference, not a deployable early-abort claim"),
    }

    (RESULTS / "1b2_earliest.json").write_text(json.dumps(out, indent=2))
    print(f"\nH-G: fraction of prefill->full gain available by m<=128 = {frac_gain:.2%}")
    print(f"Trigger (>=0.82 by m<=128): {'FIRED' if triggered else 'NOT fired'} "
          f"(best early {early_best_auroc:.4f})")
    bp = out["abort_escalate_sim"]["best_point"]
    print(f"Abort-escalate best: keep={bp['pct_kept']:.2f} cost_rel={bp['cost_rel']:.3f} "
          f"acc={bp['accuracy']:.3f} hull={bp['hull_acc']:.3f} gap={bp['gap_pp']:+.2f}pp")
    print(f"\n→ {RESULTS / '1b2_earliest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
