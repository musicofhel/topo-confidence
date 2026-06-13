#!/usr/bin/env python3
"""Phase 1B.3 — conformal / Learn-then-Test risk control on F-8 (the guarantee).

Standard split-conformal selective prediction (method: Angelopoulos LTT;
LLM application prior: 2402.10978 conformal-factual-lm, already graphed — NO
novelty claim is made here). On calibration data, pick the threshold tau that
maximises coverage subject to a Clopper-Pearson upper bound on the answered-set
error <= eps at confidence 1-delta. Validate empirical coverage over >=1000
calib/test resplits.

Readouts compared: incumbent prefill-DoM (raw + isotonic-calibrated), the free
baseline (length+logprob), and the best in-domain readout (CoE profile). Reports
the (eps, delta, tau, validity, coverage, answered-accuracy) table.

Transfer-shift restatement of the overfit gap: also runs calib=MATH / test=BBH
to show conformal validity DEGRADES under distribution shift (H-H).

Output: results/1b3_conformal.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

import activation_loader as AL
import probes as PR
from metrics import oof_dom_scores, frozen_folds, SEED

EPS_GRID = [0.1, 0.2, 0.3]
DELTA = 0.1
N_SPLITS = 1000


def cp_upper(k_err, n, delta):
    """Clopper-Pearson upper bound on error rate given k_err errors in n trials."""
    if n == 0:
        return 1.0
    if k_err == n:
        return 1.0
    return float(beta.ppf(1 - delta, k_err + 1, n - k_err))


def choose_tau(scores_cal, y_cal, eps, delta):
    """Smallest tau (=> max coverage) s.t. CP-upper(answered error) <= eps.
    'answer if score >= tau'. Returns tau or None if infeasible."""
    order = np.argsort(-scores_cal)            # high score first
    s_sorted = scores_cal[order]
    y_sorted = y_cal[order].astype(bool)
    best_tau = None
    # grow the answered set from the most confident; track when CP-upper exceeds eps
    n_ans = 0; n_err = 0
    for i in range(len(s_sorted)):
        n_ans += 1
        n_err += int(not y_sorted[i])
        ub = cp_upper(n_err, n_ans, delta)
        if ub <= eps:
            best_tau = s_sorted[i]              # can answer down to here
    return best_tau


def evaluate(scores, y, eps, delta, n_splits=N_SPLITS, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    risks, covs, accs, valid = [], [], [], []
    feasible = 0
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, te = perm[:n // 2], perm[n // 2:]
        tau = choose_tau(scores[cal], y[cal], eps, delta)
        if tau is None:
            continue
        feasible += 1
        ans = scores[te] >= tau
        if ans.sum() == 0:
            covs.append(0.0); continue
        err = 1.0 - y[te][ans].mean()
        risks.append(err)
        covs.append(ans.mean())
        accs.append(y[te][ans].mean())
        valid.append(err <= eps)
    return {
        "eps": eps, "delta": delta,
        "feasible_frac": feasible / n_splits,
        "mean_test_risk": float(np.mean(risks)) if risks else None,
        "validity_frac_risk_le_eps": float(np.mean(valid)) if valid else None,
        "target_validity": 1 - delta,
        "mean_coverage": float(np.mean(covs)) if covs else 0.0,
        "mean_answered_accuracy": float(np.mean(accs)) if accs else None,
    }


def transfer_conformal(scores_cal, y_cal, scores_te, y_te, eps, delta):
    """Calibrate on one distribution, deploy on another (H-H)."""
    tau = choose_tau(scores_cal, y_cal, eps, delta)
    if tau is None:
        return {"eps": eps, "feasible": False}
    ans = scores_te >= tau
    if ans.sum() == 0:
        return {"eps": eps, "feasible": True, "coverage": 0.0}
    return {"eps": eps, "feasible": True,
            "tau": float(tau),
            "test_risk": float(1 - y_te[ans].mean()),
            "risk_le_eps": bool((1 - y_te[ans].mean()) <= eps),
            "coverage": float(ans.mean()),
            "answered_accuracy": float(y_te[ans].mean())}


def main():
    m15 = AL.load_cell("qwen1.5b", "math")
    y = m15.y
    folds = frozen_folds(y)

    # readouts (OOF, in-domain)
    s_dom = oof_dom_scores(m15.X("prefill"), y)
    s_free = PR.fit_score_oof(PR.REGISTRY["free_baseline"], m15, folds)
    s_coe = PR.fit_score_oof(PR.REGISTRY["coe_profile"], m15, folds)
    readouts = {"prefill_dom": s_dom, "free_baseline": s_free, "coe_profile": s_coe}

    out = {"method": "split-conformal LTT (CP-upper); prior 2402.10978 (graphed)",
           "delta": DELTA, "n_splits": N_SPLITS, "in_domain": {}, "transfer_HH": {}}

    print("In-domain conformal (calib/test resplits, MATH):")
    for name, s in readouts.items():
        out["in_domain"][name] = {}
        print(f"  {name}:")
        for eps in EPS_GRID:
            r = evaluate(s, y, eps, DELTA)
            out["in_domain"][name][f"eps{eps}"] = r
            print(f"    eps={eps}: cov={r['mean_coverage']:.3f} "
                  f"ans_acc={r['mean_answered_accuracy'] and round(r['mean_answered_accuracy'],3)} "
                  f"validity={r['validity_frac_risk_le_eps'] and round(r['validity_frac_risk_le_eps'],3)} "
                  f"(target {r['target_validity']}) feasible={r['feasible_frac']:.2f}")

    # H-H: calibrate on MATH, deploy on BBH (distribution shift)
    bbh = AL.load_cell("qwen1.5b", "bbh")
    # prefill-DoM ports best cross-domain (verified): fit on MATH, score both
    s_math_dom = oof_dom_scores(m15.X("prefill"), y)
    s_bbh_dom = PR.fit_score_transfer(PR.REGISTRY["prefill_dom"], m15, bbh)
    print("\nH-H transfer conformal (calib=MATH prefill-DoM, deploy=BBH):")
    for eps in EPS_GRID:
        r = transfer_conformal(s_math_dom, y, s_bbh_dom, bbh.y, eps, DELTA)
        out["transfer_HH"][f"eps{eps}"] = r
        if r.get("feasible") and "test_risk" in r:
            print(f"  eps={eps}: BBH risk={r['test_risk']:.3f} "
                  f"(<=eps? {r['risk_le_eps']}) cov={r['coverage']:.3f} "
                  f"ans_acc={r['answered_accuracy']:.3f}")
        else:
            print(f"  eps={eps}: {r}")

    (RESULTS / "1b3_conformal.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ {RESULTS / '1b3_conformal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
