#!/usr/bin/env python3
"""C3 (conformal sweep, 2026-06-13) — group-conditional (Mondrian) certificates.

Marginal split-conformal certifies the *pooled* answered-error <= eps, but that
guarantee can hide wide per-group disparity: some categories are silently over the
budget while others are far under. This experiment promotes C1's Mondrian arm into
its own deliverable and asks the in-domain question C1 only touched on BBH:

    Does per-group (Mondrian) calibration deliver VALID per-category certificates
    where a single marginal threshold leaves a coverage/error GAP across groups,
    and what per-group TRUE-label budget k does it cost?

Two domains, deliberately different base accuracy:
  - MATH-500 (base acc 0.486)  : 7 LOCO categories. Honest in-domain scores via
                                 5-fold OUT-OF-FOLD gate (gate never sees its own
                                 eval point). The higher-accuracy regime where
                                 conditional certs may actually be FEASIBLE.
  - BBH-750   (base acc 0.241) : 3 tasks. Cross-domain gate (fit on MATH), matching
                                 C1. The cliff regime (C1: marginal validity 0.0).

For each domain + eps:
  - marginal : one tau on a pooled in-domain calibration bootstrap; report each
               group's coverage + answered-error under that single tau, and the
               cross-group GAP (max-min coverage; #groups whose answered-err > eps).
  - mondrian : per group, calibrate tau_g on k in-group TRUE labels (k grid),
               deploy on the group remainder; per-group validity@eps + coverage,
               and the label BUDGET = smallest k giving feasible & valid (>=0.9)
               conditional coverage.

Decision rule (pre-registered): Mondrian "delivers" for a (domain, eps) iff >=1
group reaches conditional validity >= 1-delta = 0.9 with coverage > 0 at some k,
AND marginal CP leaves a real gap there (some group violates eps or coverage gap
> 0.2). Report which groups are certifiable and the budget. Honest expectation:
MATH categories (higher base acc) are the half where conditional certs can exist;
BBH tasks stay infeasible (the cliff is per-group too).

CPU-only on cached scalars; no GPU. All adjudication on TRUE labels.
Output: results/c3_group_conditional_cp.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

import activation_loader as AL                       # noqa: E402
from exp_1b3_conformal import choose_tau             # LTT parity w/ C1/phase4  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "confgate"))
from confgate.gate import FreeGate                   # noqa: E402

EPS_LIST = [0.1, 0.2, 0.3]
DELTA = 0.1
R = 200
K_GRID = (8, 16, 32, 64)
TARGET_VALIDITY = 1 - DELTA       # 0.9
SEED = 20260613


# ----------------------------------------------------------------- data layer
def _load(npz, domain):
    r = np.load(RESULTS / npz, allow_pickle=True)
    cell = AL.load_cell("qwen1.5b", domain)
    assert np.array_equal(np.asarray(cell.y).astype(bool), r["y"].astype(bool))
    return dict(L=r["n_gen"].astype(float), LP=r["mean_lp_rescored"].astype(float),
                y=r["y"].astype(bool), g=np.asarray(cell.subjects))


def math_oof_scores(math):
    """Honest in-domain MATH gate scores: 5-fold out-of-fold (gate never scores its
    own training point). Mirrors the pinned LOCO protocol's leakage discipline."""
    L, LP, y = math["L"], math["LP"], math["y"]
    s = np.empty(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(L, y):
        g = FreeGate.fit(L[tr], LP[tr], y[tr], family="qwen2.5-1.5b")
        s[te] = np.ravel(g.score(L[te], LP[te]))
    return s


# ----------------------------------------------------------------- measures
def deploy(scores, ys, tau, eps):
    if tau is None:
        return 0, 0.0, None, None
    ans = scores >= tau
    if ans.sum() == 0:
        return 1, 0.0, None, None
    err = float(1 - ys[ans].mean())
    return 1, float(ans.mean()), bool(err <= eps), err


def arm_marginal(scores, ys, groups, glist, eps, rng):
    """One pooled tau on a calibration bootstrap; per-group coverage + answered-err
    under it. Reports the cross-group disparity a marginal cert hides."""
    n = len(scores)
    per_group = {g: {"cov": [], "err": [], "valid": []} for g in glist}
    for _ in range(R):
        idx = rng.integers(0, n, n)
        tau = choose_tau(scores[idx], ys[idx], eps, DELTA)
        if tau is None:
            continue
        for g in glist:
            m = groups == g
            f, c, v, e = deploy(scores[m], ys[m], tau, eps)
            per_group[g]["cov"].append(c)
            if e is not None:
                per_group[g]["err"].append(e); per_group[g]["valid"].append(int(v))
    out = {}
    covs = []
    for g in glist:
        pg = per_group[g]
        mc = float(np.mean(pg["cov"])) if pg["cov"] else 0.0
        me = float(np.mean(pg["err"])) if pg["err"] else None
        mv = float(np.mean(pg["valid"])) if pg["valid"] else None
        out[g] = {"mean_coverage": mc, "mean_answered_err": me, "validity": mv}
        covs.append(mc)
    out["_coverage_gap"] = float(max(covs) - min(covs)) if covs else 0.0
    out["_n_groups_violating"] = int(sum(
        1 for g in glist if out[g]["mean_answered_err"] is not None
        and out[g]["mean_answered_err"] > eps))
    return out


def arm_mondrian(scores, ys, groups, glist, eps, rng):
    """Per-group: calibrate tau_g on k in-group TRUE labels, deploy on remainder."""
    out = {}
    for g in glist:
        m = groups == g
        sg, yg = scores[m], ys[m]; ng = len(yg)
        pos, neg = np.where(yg)[0], np.where(~yg)[0]
        per_k = {}
        for k in K_GRID:
            covs, valids, feas = [], [], []
            kp = min(len(pos), k // 2); kn = min(len(neg), k - kp)
            if kp == 0 or kn == 0 or kp + kn >= ng:
                per_k[f"k{k}"] = {"feasible_frac": 0.0, "mean_coverage": 0.0,
                                  "validity": None, "skipped": True}
                continue
            for _ in range(R):
                sup = np.concatenate([rng.choice(pos, kp, replace=False),
                                      rng.choice(neg, kn, replace=False)])
                mask = np.zeros(ng, bool); mask[sup] = True
                tau = choose_tau(sg[sup], yg[sup], eps, DELTA)
                f, c, v, e = deploy(sg[~mask], yg[~mask], tau, eps)
                feas.append(f); covs.append(c); valids.append(v)
            vv = [v for v in valids if v is not None]
            per_k[f"k{k}"] = {
                "feasible_frac": float(np.mean(feas)),
                "mean_coverage": float(np.mean(covs)),
                "validity": float(np.mean(vv)) if vv else None}
        # label budget: smallest k with validity >= target and coverage > 0
        budget = None
        for k in K_GRID:
            pk = per_k[f"k{k}"]
            if (pk.get("validity") or 0) >= TARGET_VALIDITY and pk["mean_coverage"] > 0:
                budget = k; break
        out[g] = {"n": int(ng), "base_acc": float(yg.mean()),
                  "by_k": per_k, "label_budget": budget}
    return out


def run_domain(name, scores, ys, groups, rng):
    glist = sorted(set(groups.tolist()))
    out = {"domain": name, "n": int(len(ys)), "base_acc": float(ys.mean()),
           "groups": glist,
           "group_base_acc": {g: float(ys[groups == g].mean()) for g in glist}}
    for eps in EPS_LIST:
        marg = arm_marginal(scores, ys, groups, glist,
                            eps, np.random.default_rng(SEED + 11))
        mond = arm_mondrian(scores, ys, groups, glist,
                            eps, np.random.default_rng(SEED + 12))
        certifiable = {g: mond[g]["label_budget"] for g in glist
                       if mond[g]["label_budget"] is not None}
        out[f"eps{eps}"] = {
            "marginal": marg, "mondrian": mond,
            "marginal_coverage_gap": marg["_coverage_gap"],
            "marginal_n_groups_violating": marg["_n_groups_violating"],
            "mondrian_certifiable_groups": certifiable,
            "mondrian_delivers": bool(
                len(certifiable) > 0 and
                (marg["_n_groups_violating"] > 0 or marg["_coverage_gap"] > 0.2)),
        }
    return out


# ----------------------------------------------------------------------- main
def main():
    math = _load("v7_rescore_qwen1.5b_math.npz", "math")
    bbh = _load("v7_rescore_qwen1.5b_bbh.npz", "bbh")

    s_math = math_oof_scores(math)                          # honest in-domain OOF
    gate_x = FreeGate.fit(math["L"], math["LP"], math["y"], family="qwen2.5-1.5b")
    s_bbh = np.ravel(gate_x.score(bbh["L"], bbh["LP"]))     # cross-domain transfer

    res = {
        "experiment": "C3 group-conditional (Mondrian) certificates",
        "gate": "free_baseline length+logprob logistic (pinned recipe)",
        "math_scores": "5-fold OOF (in-domain, leakage-safe)",
        "bbh_scores": "MATH-fit gate (cross-domain transfer, matches C1)",
        "delta": DELTA, "R": R, "k_grid": list(K_GRID), "eps_list": EPS_LIST,
        "target_validity": TARGET_VALIDITY,
        "domains": {
            "math": run_domain("MATH-500", s_math, math["y"], math["g"],
                               np.random.default_rng(SEED + 1)),
            "bbh": run_domain("BBH-750", s_bbh, bbh["y"], bbh["g"],
                              np.random.default_rng(SEED + 2)),
        },
    }
    res["verdict"] = {
        dom: {f"eps{eps}": {
                "marginal_coverage_gap": res["domains"][dom][f"eps{eps}"]["marginal_coverage_gap"],
                "marginal_n_groups_violating": res["domains"][dom][f"eps{eps}"]["marginal_n_groups_violating"],
                "mondrian_certifiable_groups": res["domains"][dom][f"eps{eps}"]["mondrian_certifiable_groups"],
                "mondrian_delivers": res["domains"][dom][f"eps{eps}"]["mondrian_delivers"],
              } for eps in EPS_LIST}
        for dom in ("math", "bbh")
    }

    (RESULTS / "c3_group_conditional_cp.json").write_text(json.dumps(res, indent=2))
    for dom in ("math", "bbh"):
        d = res["domains"][dom]
        print(f"\n=== {d['domain']} (base acc {d['base_acc']:.3f}) ===")
        for eps in EPS_LIST:
            b = d[f"eps{eps}"]
            cert = b["mondrian_certifiable_groups"]
            print(f" eps={eps}: marginal cov-gap={b['marginal_coverage_gap']:.2f} "
                  f"viol-groups={b['marginal_n_groups_violating']}/{len(d['groups'])} | "
                  f"mondrian certifiable={ {g.split('_')[0][:10]:k for g,k in cert.items()} or 'NONE'} "
                  f"delivers={b['mondrian_delivers']}")
    print(f"\n-> {RESULTS / 'c3_group_conditional_cp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
