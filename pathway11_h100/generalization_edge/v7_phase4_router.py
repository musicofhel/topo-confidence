#!/usr/bin/env python3
"""v7_phase4_router.py — SPEC v7 Phase 4 (EXP-89): cost-aware router vs H-M.

Operating point (disambiguation recorded in SPEC_v7.md BEFORE this script
ran): the 11.2pp oracle gap is the ACCURACY gap at matched cost = the
free-gate cascade's escalation-fraction-0.5 point (cascade 0.6460 vs oracle
ceiling 0.7580 on 1.5B->7B MATH-500). H-M primary adjudication: paired
accuracy delta router-vs-cascade at that exact budget, router strictly OOF
on the frozen 5-fold split; confirm iff delta >= +3pp with bootstrap p<0.05
(B=2000), refute iff delta < +1pp.

Router action menu (costs from results/v7_frontier_table.json, D-2
semantics): token-feature arms free; K=2 consistency probe pays its measured
extra tokens; escalation pays the 7B pass. K=4/8 probes are cost-dominated
near this budget (k4 extra ~2241 ~ 0.72 escalations, k8 extra ~4474 > one
escalation) and the H-J verdict already establishes K-majority loses to
escalation, so the probe action is K=2 only. PRM-7B scoring costs ~4.7x the
full 1.5B transcript ~ MORE than one escalation -> cost-dominated, reported
descriptively only (resolves P11-FE403's comparator question).

Pre-named policies (each fit on train folds only; per-fold meta-selection
uses inner 5-fold OOF simulation on the train fold, so the deployed
"router" is itself a learned object evaluated once per outer test fold):
  cascade   comparator: free gate (n_gen, mean_logprob) LR, escalate lowest.
  rich      LR(y_1.5B | free+prompt_len+token arms), escalate lowest.
  rescue    LR(rescued = ~y_1.5B & y_7B | same feats), escalate highest.
  two_model LR(~y_1.5B)*LR(y_7B), escalate highest product.
  k2_band   probe the bottom-f (by rich score) with K=2 samples, re-rank the
            probed band by LR(y | feats+k2_agreement+k2_entropy), escalate
            within the residual budget; f in {0.1..1.0} chosen on the inner
            simulation.

Comparators reported: free-gate cascade (primary, paired), random-mix hull
at matched cost, best single component, oracle ceiling. Secondary view:
escalation cost at the 90%-of-7B accuracy target.

Output: results/v7_phase4_router.json
"""
from __future__ import annotations

import json
import sys
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
from metrics import frozen_folds                                 # noqa: E402
from v7_phase1a import impute_median                             # noqa: E402

RESULTS = HERE / "results"
POD = HERE / "pod_results"
RNG = np.random.default_rng(1234)
B_BOOT = 2000
F_GRID = [round(0.1 * i, 1) for i in range(1, 11)]


def fit_lr(X, y):
    sc = StandardScaler().fit(X)
    est = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: est.predict_proba(sc.transform(Z))[:, 1]


def load_data():
    m15 = AL.load_cell("qwen1.5b", "math")
    m7 = AL.load_cell("qwen7b", "math")
    y1, y7 = m15.y.astype(bool), m7.y.astype(bool)
    side = np.load(RESULTS / "v7_rescore_qwen1.5b_math.npz")
    # prompt length: the local MATH sidecar predates the pod schema and has
    # no n_prompt; borrow the 7B pod sidecar's (same tokenizer, same
    # prompts) — the same convention phase1b used for the cost basis
    side7 = np.load(RESULTS / "v7_rescore_qwen7b_math.npz")
    cols = [m15.n_gen_tokens.astype(float), m15.mean_logprob.astype(float),
            side7["n_prompt"].astype(float)]
    for k in ("min_lp", "p10_lp", "mean_entropy", "mean_top2_margin",
              "ans_span_lp"):
        cols.append(impute_median(side[k].astype(float))[0])
    X = np.column_stack(cols)
    X_free = X[:, :2]                       # the pinned 2-feature gate

    n = AL.n_k8_problems()
    assert n == len(y1)
    k2a, k2e = np.zeros(n), np.zeros(n)
    for i in range(n):
        d = np.load(AL.TRAJ_1P5B / f"problem_{i:03d}.npz", allow_pickle=True)
        greedy = k8_lib.extract_answers([str(d["text"])])[0]
        sub = k8_lib.extract_answers(AL.load_k8(i)["texts"])[:2]
        k2a[i] = float(np.mean([a == greedy and a != "" for a in sub]))
        _, counts = np.unique(sub, return_counts=True)
        p = counts / 2
        k2e[i] = float(-(p * np.log(p)).sum())
    K2 = np.column_stack([k2a, k2e])

    ft = json.loads((RESULTS / "v7_frontier_table.json").read_text())
    arms = {r["arm"]: r for r in ft["cells"]["qwen1.5b_math"]["arms"]}
    c0 = ft["cells"]["qwen1.5b_math"]["c0_abs"]
    c_esc = ft["cells"]["qwen7b_math"]["c0_abs"]
    c_k2_extra = arms["k2_agreement"]["abs_cost_per_problem"] - c0
    return y1, y7, X, X_free, K2, c0, c_esc, c_k2_extra


# ---------------------------------------------------------------------------
# Policies. fit(tr) -> state; escalate(state, te, budget_total) -> bool mask
# over te (len(te) array). budget_total is in abs token-FLOP units for the
# te set, on TOP of the mandatory 1.5B pass.
# ---------------------------------------------------------------------------

def esc_lowest(score, n_esc):
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(score)[:n_esc]] = True
    return mask


class Cascade:
    name = "cascade_free_gate"

    def __init__(self, d): self.d = d

    def fit(self, tr):
        y1, X_free = self.d["y1"], self.d["X_free"]
        return fit_lr(X_free[tr], y1[tr])

    def escalate(self, g, te, budget):
        n_esc = int(budget // self.d["c_esc"])
        return esc_lowest(g(self.d["X_free"][te]), n_esc)


class Rich(Cascade):
    name = "rich_gate"

    def fit(self, tr):
        return fit_lr(self.d["X"][tr], self.d["y1"][tr])

    def escalate(self, g, te, budget):
        n_esc = int(budget // self.d["c_esc"])
        return esc_lowest(g(self.d["X"][te]), n_esc)


class Rescue(Cascade):
    name = "rescue_ranker"

    def fit(self, tr):
        d = self.d
        rescued = (~d["y1"]) & d["y7"]
        return fit_lr(d["X"][tr], rescued[tr])

    def escalate(self, g, te, budget):
        n_esc = int(budget // self.d["c_esc"])
        return esc_lowest(-g(self.d["X"][te]), n_esc)


class TwoModel(Cascade):
    name = "two_model"

    def fit(self, tr):
        d = self.d
        return (fit_lr(d["X"][tr], d["y1"][tr]),
                fit_lr(d["X"][tr], d["y7"][tr]))

    def escalate(self, gs, te, budget):
        g1, g7 = gs
        X = self.d["X"][te]
        score = (1.0 - g1(X)) * g7(X)
        n_esc = int(budget // self.d["c_esc"])
        return esc_lowest(-score, n_esc)


class K2Band(Cascade):
    """Probe the bottom-f band (by rich gate) with K=2, re-rank, escalate."""
    name = "k2_band"

    def __init__(self, d, f=None):
        self.d, self.f = d, f

    def fit(self, tr):
        d = self.d
        g_free = fit_lr(d["X"][tr], d["y1"][tr])
        Xk = np.column_stack([d["X"], d["K2"]])
        g_k2 = fit_lr(Xk[tr], d["y1"][tr])
        return (g_free, g_k2)

    def escalate(self, gs, te, budget):
        g_free, g_k2 = gs
        d = self.d
        m = len(te)
        p_free = g_free(d["X"][te])
        order = np.argsort(p_free)                    # worst first
        n_probe = int(np.ceil(self.f * m))
        band = order[:n_probe]                        # probed band
        rest = order[n_probe:]
        budget_left = budget - n_probe * d["c_k2_extra"]
        n_esc = max(0, int(budget_left // d["c_esc"]))
        Xk = np.column_stack([d["X"], d["K2"]])
        p_band = g_k2(Xk[te][band])
        mask = np.zeros(m, dtype=bool)
        take = band[np.argsort(p_band)[:n_esc]]
        mask[take] = True
        if n_esc > len(band):                         # spill to unprobed
            extra = n_esc - len(band)
            mask[band] = True
            mask[rest[:extra]] = True
        return mask


def policy_menu(d):
    return ([Cascade(d), Rich(d), Rescue(d), TwoModel(d)]
            + [K2Band(d, f) for f in F_GRID])


def run_policy_oof(pol, d, folds, budget_per_problem):
    """Outer-OOF escalate mask for one policy."""
    n = len(d["y1"])
    mask = np.zeros(n, dtype=bool)
    cost = np.zeros(n)
    for tr, te in folds:
        st = pol.fit(tr)
        m = pol.escalate(st, te, budget_per_problem * len(te))
        mask[te] = m
        cost[te] = d["c_esc"] * m
        if isinstance(pol, K2Band):
            n_probe = int(np.ceil(pol.f * len(te)))
            band = np.argsort(pol.fit(tr)[0](d["X"][te]))[:n_probe]
            cost[te[band]] += d["c_k2_extra"]
    return mask, cost


def inner_select(d, tr, budget_per_problem):
    """Pick the best policy on the train fold via inner 5-fold OOF acc."""
    y1 = d["y1"]
    inner = frozen_folds(y1[tr])
    best, best_acc = None, -1.0
    for pol in policy_menu(d):
        esc = np.zeros(len(tr), dtype=bool)
        for itr, ite in inner:
            st = pol.fit(tr[itr])
            esc[ite] = pol.escalate(st, tr[ite],
                                    budget_per_problem * len(ite))
        acc = float(np.where(esc, d["y7"][tr], y1[tr]).mean())
        if acc > best_acc:
            best, best_acc = pol, acc
    return best, best_acc


def main():
    y1, y7, X, X_free, K2, c0, c_esc, c_k2_extra = load_data()
    d = {"y1": y1, "y7": y7, "X": X, "X_free": X_free, "K2": K2,
         "c0": c0, "c_esc": c_esc, "c_k2_extra": c_k2_extra}
    n = len(y1)
    folds = frozen_folds(y1)
    budget = 0.5 * c_esc                              # the 11.2pp gap point

    # comparator: free-gate cascade, outer-OOF
    cas_mask, _ = run_policy_oof(Cascade(d), d, folds, budget)
    acc_cascade = float(np.where(cas_mask, y7, y1).mean())

    # descriptive: every pre-named policy at the same budget, outer-OOF
    rows = []
    for pol in policy_menu(d):
        m, cost = run_policy_oof(pol, d, folds, budget)
        rows.append({"policy": pol.name,
                     **({"f": pol.f} if isinstance(pol, K2Band) else {}),
                     "esc_frac": float(m.mean()),
                     "mean_extra_cost": float(cost.mean()),
                     "acc": float(np.where(m, y7, y1).mean())})

    # the router: per-outer-fold inner selection
    router_mask = np.zeros(n, dtype=bool)
    selected = []
    for tr, te in folds:
        pol, inner_acc = inner_select(d, tr, budget)
        st = pol.fit(tr)
        router_mask[te] = pol.escalate(st, te, budget * len(te))
        selected.append({"policy": pol.name,
                         **({"f": pol.f} if isinstance(pol, K2Band) else {}),
                         "inner_oof_acc": inner_acc})
    acc_router = float(np.where(router_mask, y7, y1).mean())

    # paired bootstrap on per-problem outcomes
    out_r = np.where(router_mask, y7, y1).astype(float)
    out_c = np.where(cas_mask, y7, y1).astype(float)
    deltas = np.empty(B_BOOT)
    for b in range(B_BOOT):
        idx = RNG.integers(0, n, n)
        deltas[b] = out_r[idx].mean() - out_c[idx].mean()
    delta = acc_router - acc_cascade
    p_gt0 = float((deltas <= 0).mean())

    # anchors at this budget
    rescue_frac = float(((~y1) & y7).mean())
    acc_oracle = float(y1.mean() + min(0.5, rescue_frac))
    acc_hull = float(y1.mean() + 0.5 * (y7.mean() - y1.mean()))

    # secondary: escalation cost at the 90%-of-7B accuracy target
    target = 0.9 * float(y7.mean())

    def esc_frac_for_target(mask_order_score):
        order = np.argsort(mask_order_score)
        run_y = y1.copy().astype(float)
        for k in range(n + 1):
            esc = np.zeros(n, dtype=bool)
            esc[order[:k]] = True
            if np.where(esc, y7, y1).mean() >= target:
                return k / n
        return 1.0

    free_oof = np.zeros(n)
    rtr_score = np.zeros(n)
    for tr, te in folds:
        free_oof[te] = fit_lr(X_free[tr], y1[tr])(X_free[te])
        rtr_score[te] = fit_lr(X[tr], (~y1[tr]) & y7[tr])(X[te])
    secondary = {
        "target_acc_90pct_of_7b": target,
        "cascade_esc_frac": esc_frac_for_target(free_oof),
        "rescue_ranked_esc_frac": esc_frac_for_target(-rtr_score),
        "oracle_esc_frac": float(np.ceil((target - y1.mean()) * n) / n)}

    # PRM descriptive row (cost-dominated; resolves P11-FE403 comparator)
    prm = None
    prm_f = POD / "v7_prm_qwen1.5b_math.json"
    if prm_f.exists():
        from v7_phase1a import arm_row, oof_scores
        from metrics import auroc_sym
        pd_ = json.loads(prm_f.read_text())
        assert list(map(bool, pd_["y"])) == list(map(bool, y1))
        free_s = free_oof
        free_a = auroc_sym(y1, free_s)
        prm = {"cost_tokens_scored": pd_["cost_tokens_scored"],
               "abs_cost_per_problem_7b_flops":
                   4.7 * pd_["cost_tokens_scored"] / n,
               "rel_cost_vs_escalation":
                   4.7 * pd_["cost_tokens_scored"] / n / c_esc,
               "arms": []}
        for k in ("prm_min", "prm_mean", "prm_last"):
            x, cov = impute_median(np.array(pd_[k], dtype=float))
            r = arm_row(k, x, X_free, y1, folds, free_s, free_a)
            r["finite_fraction"] = cov
            prm["arms"].append(r)

    holds = bool(delta >= 0.03 and p_gt0 < 0.05)
    refuted = bool(delta < 0.01)
    out = {
        "spec": "v7 Phase 4 (EXP-89) — cost-aware router, H-M",
        "operating_point": {
            "budget_extra_per_problem": budget, "c_esc": c_esc,
            "c_k2_extra": c_k2_extra,
            "acc_cascade_free_gate": acc_cascade,
            "acc_oracle": acc_oracle, "acc_random_mix_hull": acc_hull,
            "oracle_gap_pp": (acc_oracle - acc_cascade) * 100},
        "policies_oof": rows,
        "router": {"acc": acc_router,
                   "esc_frac": float(router_mask.mean()),
                   "selected_per_fold": selected,
                   "delta_vs_cascade": delta,
                   "bootstrap_p_delta_le_0": p_gt0,
                   "ci95": [float(np.percentile(deltas, 2.5)),
                            float(np.percentile(deltas, 97.5))]},
        "secondary_90pct_view": secondary,
        "prm_descriptive": prm,
        "H_M": {"confirmed_ge_3pp": holds,
                "refuted_lt_1pp": refuted,
                "verdict": ("CONFIRMED" if holds else
                            "REFUTED" if refuted else "PARTIAL")}}
    f = RESULTS / "v7_phase4_router.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {f}\n")
    print(f"cascade {acc_cascade:.4f}  oracle {acc_oracle:.4f}  "
          f"hull {acc_hull:.4f}  gap {(acc_oracle-acc_cascade)*100:.1f}pp")
    for r in rows:
        tag = f" f={r.get('f')}" if "f" in r else ""
        print(f"  {r['policy']:18s}{tag:7s} esc={r['esc_frac']:.3f} "
              f"acc={r['acc']:.4f}")
    print(f"\nROUTER acc {acc_router:.4f}  Δ={delta*100:+.2f}pp "
          f"p={p_gt0:.4f}  -> H-M {out['H_M']['verdict']}")
    print(f"selected: {[s['policy'] for s in selected]}")
    print(f"secondary 90% view: {secondary}")
    if prm:
        print(f"PRM rel cost vs escalation: "
              f"{prm['rel_cost_vs_escalation']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
