#!/usr/bin/env python3
"""C1 (conformal sweep, 2026-06-13) — crack the cross-domain certificate wall.

SPEC v6 Phase 4 / v7 declared confgate's cross-domain risk certificate INFEASIBLE:
calibrate a selective-risk threshold on MATH, deploy on BBH, and the answered-set
error blows past eps -- validity 0.0 at eps=0.2 even after refitting on k=32/64 true
BBH labels (results/phase4_recalibration.json::math_to_bbh; 1b3_conformal.json::
transfer_HH). BUT every one of those tests used STATIC split-conformal LTT. The
distribution-shift conformal toolkit was never tried. This experiment asks:

    Does ANY adaptive method restore a valid cross-domain certificate
    (validity >= 1-delta = 0.9 at eps=0.2) where static gives 0.0?

Arms (gate = the pinned free-baseline length+logprob logistic; fit ONCE on MATH,
fixed across arms -- only the certificate machinery changes):
  0. static            -- split-conformal LTT (the wall; reproduces validity 0.0).
  1. weighted          -- Tibshirani covariate-shift weighted CP: reweight MATH
                          calibration by a density ratio p_bbh(x)/p_math(x) from a
                          domain classifier on the 2 free scalars. Tests "is it pure
                          covariate shift in x?" (fixable) vs concept/non-overlap.
  2. nonexch_barber    -- Barber et al. "beyond exchangeability" robustness: the
                          coverage-gap penalty Delta (TV of the reweighted calib
                          weights from uniform); does a distribution-free guarantee
                          survive eps - Delta? If Delta is large, NO cross-domain cert
                          exists from MATH alone -- a decisive formal negative.
  3. online_aci        -- Gibbs&Candes / Zaffran adaptive conformal: stream BBH WITH
                          labels, adapt the rate online to a fixed long-run error.
                          A different (online-feedback) deployment model.
  4. mondrian          -- group-conditional: per-BBH-task calibration with k true
                          labels/group. Tests whether the marginal-mixing wall hides
                          valid per-group coverage, and the label budget it needs.

Honest framing: this is a GUARANTEES lever (does confgate ship `certify_cross_domain`),
not an accuracy lever. CPU-only on cached scalars; no GPU. All adjudication on TRUE
BBH labels.

Output: results/c1_cross_domain_adaptive_cp.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

import activation_loader as AL                       # noqa: E402
from exp_1b3_conformal import choose_tau, cp_upper   # exact LTT parity w/ phase4  # noqa: E402

# confgate gate (the deployment scoring function) -- import the in-repo copy
sys.path.insert(0, str(HERE.parents[1] / "confgate"))
from confgate.gate import FreeGate                   # noqa: E402

EPS_LIST = [0.1, 0.2, 0.3]
DELTA = 0.1
R = 200                       # calibration resamples (bootstrap of MATH calib)
TARGET_VALIDITY = 1 - DELTA   # 0.9 -> an arm "wins" iff validity >= this with cov>0
SEED = 20260613


# ----------------------------------------------------------------- data layer
def load_domains():
    """Index-aligned (length, logprob, y, group) for MATH (calib) and BBH (deploy).

    Uses HONEST rescored lengths (cached BBH n_gen_tokens is the D-3 degenerate-zero
    defect); subset/category labels come from the activation loader (same order).
    """
    rm = np.load(RESULTS / "v7_rescore_qwen1.5b_math.npz", allow_pickle=True)
    rb = np.load(RESULTS / "v7_rescore_qwen1.5b_bbh.npz", allow_pickle=True)
    mcell = AL.load_cell("qwen1.5b", "math")
    bcell = AL.load_cell("qwen1.5b", "bbh")
    assert np.array_equal(np.asarray(mcell.y).astype(bool), rm["y"].astype(bool))
    assert np.array_equal(np.asarray(bcell.y).astype(bool), rb["y"].astype(bool))
    math = dict(L=rm["n_gen"].astype(float), LP=rm["mean_lp_rescored"].astype(float),
                y=rm["y"].astype(bool), g=np.asarray(mcell.subjects))
    bbh = dict(L=rb["n_gen"].astype(float), LP=rb["mean_lp_rescored"].astype(float),
               y=rb["y"].astype(bool), g=np.asarray(bcell.subjects))
    return math, bbh


# ------------------------------------------------------- weighted LTT machinery
def cp_upper_continuous(k_eff: float, n_eff: float, delta: float) -> float:
    """Clopper-Pearson upper bound generalized to a (continuous) effective count.
    Reduces to cp_upper for integer (k_eff, n_eff)."""
    if n_eff <= 0:
        return 1.0
    if k_eff >= n_eff:
        return 1.0
    return float(_beta.ppf(1 - delta, k_eff + 1.0, n_eff - k_eff))


def choose_tau_weighted(scores, y, w, eps, delta):
    """Weighted split-conformal LTT: smallest tau (max coverage) s.t. the WEIGHTED
    answered-set error has a CP-upper bound (via Kish effective n) <= eps.

    Weights w approximate the deploy (BBH) distribution over the calibration (MATH)
    points -> the weighted error rate estimates the deploy answered error."""
    scores = np.asarray(scores, float); y = np.asarray(y).astype(bool)
    w = np.asarray(w, float)
    order = np.argsort(-scores)
    s_s, y_s, w_s = scores[order], y[order], w[order]
    err = (~y_s).astype(float)
    W = np.cumsum(w_s)                       # weighted answered mass
    W_err = np.cumsum(w_s * err)             # weighted error mass
    W_sq = np.cumsum(w_s ** 2)
    best_tau = None
    for i in range(len(s_s)):
        if W[i] <= 0:
            continue
        n_eff = W[i] ** 2 / W_sq[i]          # Kish effective sample size
        p_hat = W_err[i] / W[i]
        k_eff = p_hat * n_eff
        if cp_upper_continuous(k_eff, n_eff, delta) <= eps:
            best_tau = s_s[i]
    return best_tau


def domain_weights(math, bbh):
    """Density-ratio weights w_i = p_bbh(x_i)/p_math(x_i) for MATH calibration points,
    from a logistic domain classifier on the 2 free scalars. Returns (w, clf_auc)."""
    Xm = np.column_stack([math["L"], math["LP"]])
    Xb = np.column_stack([bbh["L"], bbh["LP"]])
    X = np.vstack([Xm, Xb]); d = np.r_[np.zeros(len(Xm)), np.ones(len(Xb))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), d)
    pm = clf.predict_proba(sc.transform(Xm))[:, 1]            # P(bbh | x) on MATH pts
    pm = np.clip(pm, 1e-6, 1 - 1e-6)
    w = (pm / (1 - pm)) * (len(Xm) / len(Xb))                 # density ratio
    # domain-classifier AUC (separability of MATH vs BBH on the 2 scalars)
    from metrics import auroc_sym
    pall = clf.predict_proba(sc.transform(X))[:, 1]
    clf_auc = float(auroc_sym(d.astype(bool), pall))
    return w, clf_auc


# --------------------------------------------------------------- deploy/measure
def deploy(scores_bbh, y_bbh, tau, eps):
    """Apply tau to BBH; return (feasible, coverage, valid_bool, answered_err)."""
    if tau is None:
        return 0, 0.0, None, None
    ans = scores_bbh >= tau
    if ans.sum() == 0:
        return 1, 0.0, None, None
    err = float(1 - y_bbh[ans].mean())
    return 1, float(ans.mean()), bool(err <= eps), err


def _summ(feas, covs, valids, errs):
    vv = [v for v in valids if v is not None]
    return {"feasible_frac": float(np.mean(feas)) if feas else 0.0,
            "mean_coverage": float(np.mean(covs)) if covs else 0.0,
            "validity": float(np.mean(vv)) if vv else None,
            "mean_answered_err": float(np.mean([e for e in errs if e is not None]))
                                 if any(e is not None for e in errs) else None}


# ------------------------------------------------------------------------ arms
def arm_static(math, bbh, s_math, s_bbh, rng):
    """Reproduce the wall: MATH-calibrated split-conformal LTT, deployed on BBH."""
    out = {}
    n = len(s_math)
    for eps in EPS_LIST:
        feas, covs, valids, errs = [], [], [], []
        # point estimate on full MATH calib (matches phase4 k=0 single draw)
        tau0 = choose_tau(s_math, math["y"], eps, DELTA)
        f0, c0, v0, e0 = deploy(s_bbh, bbh["y"], tau0, eps)
        for _ in range(R):                                   # bootstrap MATH calib
            idx = rng.integers(0, n, n)
            tau = choose_tau(s_math[idx], math["y"][idx], eps, DELTA)
            f, c, v, e = deploy(s_bbh, bbh["y"], tau, eps)
            feas.append(f); covs.append(c); valids.append(v); errs.append(e)
        out[f"eps{eps}"] = {**_summ(feas, covs, valids, errs),
                            "point_tau": None if tau0 is None else float(tau0),
                            "point_coverage": c0, "point_valid": v0,
                            "point_answered_err": e0}
    return out


def arm_weighted(math, bbh, s_math, s_bbh, w, rng):
    """Tibshirani covariate-shift weighted CP."""
    out = {}; n = len(s_math)
    for eps in EPS_LIST:
        feas, covs, valids, errs = [], [], [], []
        for _ in range(R):
            idx = rng.integers(0, n, n)
            tau = choose_tau_weighted(s_math[idx], math["y"][idx], w[idx], eps, DELTA)
            f, c, v, e = deploy(s_bbh, bbh["y"], tau, eps)
            feas.append(f); covs.append(c); valids.append(v); errs.append(e)
        out[f"eps{eps}"] = _summ(feas, covs, valids, errs)
    return out


def arm_nonexch_barber(math, w):
    """Barber et al. (2023) 'conformal prediction beyond exchangeability': with
    normalized weights w~ (sum=1), the distribution-free coverage gap is bounded by
    the total-variation mass the weighting places away from uniform. Report Delta and
    whether a guarantee survives the eps budget (eps - Delta > 0 is necessary)."""
    wn = w / w.sum()
    delta_tv = float(0.5 * np.sum(np.abs(wn - 1.0 / len(wn))) * 2)  # TV(w~, uniform)
    # Barber's gap term: sum of the (normalized) weights' deviation; equivalently the
    # influence of the unweighted tail. Use the standard 1 - sum(min(w~,1/n))*... form:
    gap = float(np.sum(np.maximum(wn - 1.0 / len(wn), 0.0)))        # mass moved up
    out = {"weight_tv_from_uniform": delta_tv, "barber_coverage_gap": gap,
           "kish_eff_n": float(w.sum() ** 2 / np.sum(w ** 2)),
           "n_calib": int(len(w))}
    for eps in EPS_LIST:
        out[f"eps{eps}"] = {
            "robust_budget_eps_minus_gap": float(eps - gap),
            "guarantee_possible": bool(eps - gap > 0)}
    return out


def arm_online_aci(bbh, s_bbh, rng, gamma=0.05):
    """Gibbs & Candes (2021) ACI / Zaffran online conformal. Stream BBH WITH labels;
    maintain an adaptive error-rate target alpha_t and answer iff score >= running
    (1-alpha_t)-quantile of the WRONG-class scores seen so far. Report realized
    long-run answered error and coverage, averaged over R stream orders."""
    out = {}; n = len(s_bbh); y = bbh["y"]
    for eps in EPS_LIST:
        errs, covs = [], []
        for _ in range(R):
            order = rng.permutation(n)
            alpha = eps
            seen_s, seen_y = [], []
            n_ans = n_err = 0
            for t in order:
                # threshold = current (1-alpha) quantile of seen scores; answer high
                if seen_s:
                    tau = np.quantile(np.asarray(seen_s), 1 - np.clip(alpha, 0, 1))
                else:
                    tau = -np.inf
                answered = s_bbh[t] >= tau
                if answered:
                    n_ans += 1
                    err_t = int(not y[t])
                    n_err += err_t
                    # ACI update: push alpha toward eps using realized loss
                    alpha = alpha + gamma * (eps - err_t)
                seen_s.append(s_bbh[t]); seen_y.append(y[t])
            if n_ans:
                errs.append(n_err / n_ans); covs.append(n_ans / n)
        out[f"eps{eps}"] = {
            "mean_long_run_err": float(np.mean(errs)) if errs else None,
            "mean_coverage": float(np.mean(covs)) if covs else 0.0,
            "validity_err_le_eps": float(np.mean([e <= eps for e in errs]))
                                   if errs else None,
            "gamma": gamma}
    return out


def arm_mondrian(bbh, s_bbh, rng, k_grid=(8, 16, 32, 64)):
    """Group-conditional (Mondrian) CP: per-BBH-task, calibrate tau on k TRUE in-group
    labels, deploy on the group remainder. Tests whether valid per-group coverage
    exists (vs the failed marginal transfer), and the label budget it needs."""
    groups = sorted(set(bbh["g"].tolist()))
    out = {"groups": groups}
    for eps in EPS_LIST:
        per_k = {}
        for k in k_grid:
            g_valids, g_covs, g_feas = [], [], []
            for g in groups:
                m = bbh["g"] == g
                sg, yg = s_bbh[m], bbh["y"][m]
                ng = len(yg)
                for _ in range(R):
                    pos = np.where(yg)[0]; neg = np.where(~yg)[0]
                    kp = min(len(pos), k // 2); kn = min(len(neg), k - kp)
                    if kp == 0 or kn == 0:
                        continue
                    sup = np.concatenate([rng.choice(pos, kp, replace=False),
                                          rng.choice(neg, kn, replace=False)])
                    mask = np.zeros(ng, bool); mask[sup] = True
                    rem = ~mask
                    tau = choose_tau(sg[sup], yg[sup], eps, DELTA)
                    f, c, v, e = deploy(sg[rem], yg[rem], tau, eps)
                    g_feas.append(f); g_covs.append(c); g_valids.append(v)
            vv = [v for v in g_valids if v is not None]
            per_k[f"k{k}"] = {
                "feasible_frac": float(np.mean(g_feas)) if g_feas else 0.0,
                "mean_coverage": float(np.mean(g_covs)) if g_covs else 0.0,
                "validity": float(np.mean(vv)) if vv else None}
        out[f"eps{eps}"] = per_k
    return out


# ------------------------------------------------------------------------ main
def main():
    math, bbh = load_domains()
    gate = FreeGate.fit(math["L"], math["LP"], math["y"], family="qwen2.5-1.5b")
    s_math = np.ravel(gate.score(math["L"], math["LP"]))
    s_bbh = np.ravel(gate.score(bbh["L"], bbh["LP"]))
    w, clf_auc = domain_weights(math, bbh)

    rng = np.random.default_rng(SEED)
    res = {
        "experiment": "C1 cross-domain adaptive conformal",
        "calib_domain": "MATH-500 (qwen2.5-1.5b)", "deploy_domain": "BBH-750",
        "gate": "free_baseline length+logprob logistic (pinned recipe), fit on MATH",
        "delta": DELTA, "R": R, "eps_list": EPS_LIST,
        "target_validity": TARGET_VALIDITY,
        "domain_classifier_auc": clf_auc,
        "kish_eff_n_weighted": float(w.sum() ** 2 / np.sum(w ** 2)),
        "math_mean_y": float(math["y"].mean()), "bbh_mean_y": float(bbh["y"].mean()),
        "arms": {},
    }
    print(f"domain-classifier AUC (MATH vs BBH on 2 scalars) = {clf_auc:.3f}")
    print(f"weighted Kish effective n = {res['kish_eff_n_weighted']:.1f} / {len(w)}")

    res["arms"]["static"] = arm_static(math, bbh, s_math, s_bbh,
                                       np.random.default_rng(SEED + 1))
    res["arms"]["weighted"] = arm_weighted(math, bbh, s_math, s_bbh, w,
                                           np.random.default_rng(SEED + 2))
    res["arms"]["nonexch_barber"] = arm_nonexch_barber(math, w)
    res["arms"]["online_aci"] = arm_online_aci(bbh, s_bbh,
                                               np.random.default_rng(SEED + 3))
    res["arms"]["mondrian"] = arm_mondrian(bbh, s_bbh,
                                           np.random.default_rng(SEED + 4))

    # verdict @ eps=0.2 (the deployable target)
    e = "eps0.2"
    static_v = res["arms"]["static"][e]["validity"]
    wins = []
    for arm in ("weighted", "online_aci", "mondrian"):
        a = res["arms"][arm][e]
        if arm == "mondrian":
            v = max((a[k]["validity"] or 0) for k in a if k.startswith("k"))
            cov = max((a[k]["mean_coverage"] or 0) for k in a if k.startswith("k"))
        elif arm == "online_aci":
            v = a.get("validity_err_le_eps") or 0; cov = a["mean_coverage"]
        else:
            v = a["validity"] or 0; cov = a["mean_coverage"]
        if v >= TARGET_VALIDITY and cov > 0:
            wins.append(arm)
    res["verdict"] = {
        "static_validity_eps0.2": static_v,
        "static_reproduces_wall": bool((static_v or 0) < 0.5),
        "winning_arms_eps0.2": wins,
        "cross_domain_cert_feasible": len(wins) > 0,
    }

    (RESULTS / "c1_cross_domain_adaptive_cp.json").write_text(json.dumps(res, indent=2))
    print("\n=== verdict @ eps=0.2 ===")
    print(f"  static validity         = {static_v}  (wall reproduced: "
          f"{res['verdict']['static_reproduces_wall']})")
    print(f"  weighted validity       = {res['arms']['weighted'][e]['validity']} "
          f"cov={res['arms']['weighted'][e]['mean_coverage']:.2f}")
    print(f"  online_aci validity     = {res['arms']['online_aci'][e].get('validity_err_le_eps')} "
          f"cov={res['arms']['online_aci'][e]['mean_coverage']:.2f} "
          f"err={res['arms']['online_aci'][e].get('mean_long_run_err')}")
    md = res['arms']['mondrian'][e]
    print(f"  mondrian validity by k  = "
          + ", ".join(f"{k}:{md[k]['validity']}(cov{md[k]['mean_coverage']:.2f})"
                      for k in md if k.startswith('k')))
    print(f"  Barber coverage gap     = {res['arms']['nonexch_barber']['barber_coverage_gap']:.3f}"
          f" -> guarantee possible @e=.2: "
          f"{res['arms']['nonexch_barber']['eps0.2']['guarantee_possible']}")
    print(f"\n  WINNING ARMS @ eps=0.2: {wins or 'NONE (wall holds)'}")
    print(f"→ {RESULTS / 'c1_cross_domain_adaptive_cp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
