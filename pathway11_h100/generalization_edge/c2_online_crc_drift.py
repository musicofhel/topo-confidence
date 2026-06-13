#!/usr/bin/env python3
"""C2 (conformal sweep, 2026-06-13) — online Conformal Risk Control under drift.

confgate ships a STATIC one-shot selective-risk threshold tau: calibrate once
(split-conformal LTT), freeze, deploy. Real deployments drift. This experiment
asks the deployment-robustness question C1 did not:

    When the input distribution drifts (MATH -> BBH mid-stream), does an ONLINE
    threshold controller (Gibbs & Candes / Zaffran ACI, in score space, targeting
    the selective answered-error) keep confgate VALID where the frozen tau
    silently violates eps?

Two stream regimes, both streamed item-by-item with label feedback (the standard
online-conformal assumption: you learn whether each *answered* item was right):
  - drift     : MATH block, then BBH block. A covariate+concept shift at a known
                onset index. The realistic "domain changed under you" case.
  - shuffled  : MATH+BBH fully interleaved (stationary mixture). CONTROL: isolates
                whether a frozen-tau violation is a *non-stationarity* problem
                (online adaptation fixes) vs a *stationary hard-mixture* problem
                (online adaptation cannot fix -- it is then just re-calibration).

Two controllers on each stream:
  - frozen    : tau from split-conformal LTT on a MATH calibration bootstrap, held
                FIXED for the whole stream (what confgate ships today).
  - online    : warm-started at the SAME frozen tau, then ACI/CRC-updated in score
                space ONLY on answered items: tau <- tau + gamma*sigma*(err_t - eps).
                Wrong-when-answered -> tau rises (stricter); right -> tau falls
                slightly (answer more). Directly targets answered-error = eps.

Headline metric = the post-drift trailing-window VIOLATION RATE (fraction of steps
whose trailing answered-error exceeds eps) and the fail-open/fail-safe coverage
contrast. Honest expectation (C1 showed BBH base acc 0.241 => no threshold holds
eps=0.2 with useful coverage): the win is not "online holds eps at high coverage"
on the cliff -- it is "online FAILS SAFE (detects drift, clamps coverage) where
frozen FAILS OPEN (keeps answering at high error, silently)." At looser/feasible
eps the online controller additionally restores long-run validity. Both reported.

CPU-only on cached scalars; no GPU. All adjudication on TRUE labels.
Output: results/c2_online_crc_drift.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

import activation_loader as AL                       # noqa: E402
from exp_1b3_conformal import choose_tau             # exact LTT parity w/ C1/phase4  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "confgate"))
from confgate.gate import FreeGate                   # noqa: E402

EPS_LIST = [0.1, 0.2, 0.3, 0.4]   # incl. a looser eps where the cliff may relent
DELTA = 0.1
R = 200                            # stream orders (bootstrap of calib + shuffle)
WINDOW = 100                       # trailing window for the violation-rate metric
GAMMA = 0.05                       # ACI learning rate (score-space, x sigma)
N_BINS = 25                        # coarse time-series bins for figures
SEED = 20260613


# ----------------------------------------------------------------- data layer
def load_domains():
    """Index-aligned (length, logprob, y) for MATH (calib) and BBH (deploy).
    HONEST rescored lengths (cached BBH n_gen is the D-3 degenerate-zero defect)."""
    rm = np.load(RESULTS / "v7_rescore_qwen1.5b_math.npz", allow_pickle=True)
    rb = np.load(RESULTS / "v7_rescore_qwen1.5b_bbh.npz", allow_pickle=True)
    mcell = AL.load_cell("qwen1.5b", "math")
    bcell = AL.load_cell("qwen1.5b", "bbh")
    assert np.array_equal(np.asarray(mcell.y).astype(bool), rm["y"].astype(bool))
    assert np.array_equal(np.asarray(bcell.y).astype(bool), rb["y"].astype(bool))
    math = dict(L=rm["n_gen"].astype(float), LP=rm["mean_lp_rescored"].astype(float),
                y=rm["y"].astype(bool))
    bbh = dict(L=rb["n_gen"].astype(float), LP=rb["mean_lp_rescored"].astype(float),
               y=rb["y"].astype(bool))
    return math, bbh


# --------------------------------------------------------------- controllers
def run_frozen(scores, ys, tau):
    """Stream with a fixed tau. Returns per-step (answered, err_if_answered)."""
    ans = scores >= tau
    err = np.where(ans, (~ys).astype(float), np.nan)
    return ans, err


def run_online(scores, ys, tau0, eps, sigma, gamma=GAMMA):
    """ACI/CRC in score space, warm-started at tau0. Updates ONLY on answered items:
        tau <- tau + gamma*sigma*(err_t - eps)
    so a wrong answer raises tau (stricter) and a right answer lowers it. Returns
    per-step (answered, err_if_answered, tau_t)."""
    n = len(scores)
    ans = np.zeros(n, bool); err = np.full(n, np.nan); taus = np.empty(n)
    tau = float(tau0)
    lo, hi = float(scores.min()) - 1e-3, float(scores.max()) + 1e-3
    for t in range(n):
        taus[t] = tau
        a = scores[t] >= tau
        ans[t] = a
        if a:
            e = float(not ys[t])
            err[t] = e
            tau = min(hi, max(lo, tau + gamma * sigma * (e - eps)))
    return ans, err, taus


# ----------------------------------------------------------------- measures
def trailing_violation_rate(ans, err, eps, window, start=0):
    """Fraction of steps t>=start whose trailing-`window` answered-error exceeds eps.
    Steps with no answered item in the window are skipped (no risk asserted)."""
    n = len(ans)
    viol = 0; counted = 0
    for t in range(start, n):
        a = ans[max(0, t - window + 1): t + 1]
        e = err[max(0, t - window + 1): t + 1]
        m = a & ~np.isnan(e)
        if m.sum() == 0:
            continue
        counted += 1
        if np.nanmean(e[m]) > eps:
            viol += 1
    return (viol / counted) if counted else None


def seg_stats(ans, err, lo, hi):
    """Coverage + answered-error over stream slice [lo:hi)."""
    a = ans[lo:hi]; e = err[lo:hi]
    cov = float(a.mean()) if len(a) else 0.0
    m = a & ~np.isnan(e)
    aerr = float(np.nanmean(e[m])) if m.sum() else None
    return cov, aerr


def timeseries(ans, err, window, n_bins):
    """Coarse (trailing-err, coverage) series for figures."""
    n = len(ans); idx = np.linspace(window, n - 1, n_bins).astype(int)
    te, tc = [], []
    for t in idx:
        a = ans[t - window + 1: t + 1]; e = err[t - window + 1: t + 1]
        m = a & ~np.isnan(e)
        te.append(float(np.nanmean(e[m])) if m.sum() else None)
        tc.append(float(a.mean()))
    return {"step": idx.tolist(), "trailing_err": te, "coverage": tc}


# ----------------------------------------------------------------- one regime
def run_regime(regime, math, bbh, gate, rng):
    """For each eps: average frozen vs online over R stream orders; report pre/post
    drift coverage+err, post-drift violation rate, and one representative series."""
    sM = np.ravel(gate.score(math["L"], math["LP"])); yM = math["y"]
    sB = np.ravel(gate.score(bbh["L"], bbh["LP"])); yB = bbh["y"]
    nM, nB = len(sM), len(sB)
    sigma = float(np.std(np.r_[sM, sB]))
    out = {"regime": regime, "n_math": nM, "n_bbh": nB, "sigma_score": sigma}

    for eps in EPS_LIST:
        agg = {c: {"pre": {"cov": [], "err": []}, "post": {"cov": [], "err": []},
                   "overall": {"cov": [], "err": []}, "viol": []}
               for c in ("frozen", "online")}
        rep = {}
        for r in range(R):
            # frozen tau: split-conformal LTT on a MATH calibration bootstrap
            ci = rng.integers(0, nM, nM)
            tau = choose_tau(sM[ci], yM[ci], eps, DELTA)
            if tau is None:                       # infeasible calib draw -> skip
                continue
            permM = rng.permutation(nM); permB = rng.permutation(nB)
            if regime == "drift":
                scores = np.r_[sM[permM], sB[permB]]
                ys = np.r_[yM[permM], yB[permB]]
                onset = nM
            else:                                  # shuffled stationary mixture
                allp = rng.permutation(nM + nB)
                scores = np.r_[sM, sB][allp]; ys = np.r_[yM, yB][allp]
                onset = (nM + nB) // 2             # nominal midpoint (no real drift)

            fa, fe = run_frozen(scores, ys, tau)
            oa, oe, _ = run_online(scores, ys, tau, eps, sigma)
            for cname, (a, e) in (("frozen", (fa, fe)), ("online", (oa, oe))):
                pc, pe = seg_stats(a, e, 0, onset)
                qc, qe = seg_stats(a, e, onset, len(a))
                oc, oee = seg_stats(a, e, 0, len(a))
                agg[cname]["pre"]["cov"].append(pc)
                if pe is not None: agg[cname]["pre"]["err"].append(pe)
                agg[cname]["post"]["cov"].append(qc)
                if qe is not None: agg[cname]["post"]["err"].append(qe)
                agg[cname]["overall"]["cov"].append(oc)
                if oee is not None: agg[cname]["overall"]["err"].append(oee)
                agg[cname]["viol"].append(
                    trailing_violation_rate(a, e, eps, WINDOW, start=onset))
                if r == 0:
                    rep[cname] = timeseries(a, e, WINDOW, N_BINS)

        def _m(xs):
            xs = [x for x in xs if x is not None]
            return float(np.mean(xs)) if xs else None
        block = {}
        for c in ("frozen", "online"):
            block[c] = {
                "pre_cov": _m(agg[c]["pre"]["cov"]), "pre_err": _m(agg[c]["pre"]["err"]),
                "post_cov": _m(agg[c]["post"]["cov"]), "post_err": _m(agg[c]["post"]["err"]),
                "overall_cov": _m(agg[c]["overall"]["cov"]),
                "overall_err": _m(agg[c]["overall"]["err"]),
                "post_violation_rate": _m(agg[c]["viol"]),
                "series_r0": rep.get(c),
            }
        # honest tags
        fz, on = block["frozen"], block["online"]
        block["frozen"]["fails_open"] = bool(
            fz["post_err"] is not None and fz["post_err"] > eps and (fz["post_cov"] or 0) > 0.2)
        block["online"]["fails_safe"] = bool(
            on["post_err"] is not None and fz["post_err"] is not None
            and on["post_cov"] is not None and fz["post_cov"] is not None
            and on["post_cov"] < fz["post_cov"]            # answered fewer post-drift
            and (on["post_err"] <= fz["post_err"] + 1e-9)) # at no worse error
        block["online"]["holds_eps_post"] = bool(
            on["post_err"] is not None and on["post_err"] <= eps + 0.01
            and (on["post_cov"] or 0) > 0.0)
        block["online_violation_lt_frozen"] = bool(
            on["post_violation_rate"] is not None and fz["post_violation_rate"] is not None
            and on["post_violation_rate"] < fz["post_violation_rate"])
        out[f"eps{eps}"] = block
    return out


# ----------------------------------------------------------------------- main
def main():
    math, bbh = load_domains()
    gate = FreeGate.fit(math["L"], math["LP"], math["y"], family="qwen2.5-1.5b")
    res = {
        "experiment": "C2 online conformal risk control under drift",
        "calib_domain": "MATH-500 (qwen2.5-1.5b)", "deploy_domain": "BBH-750",
        "gate": "free_baseline length+logprob logistic (pinned recipe), fit on MATH",
        "delta": DELTA, "R": R, "window": WINDOW, "gamma": GAMMA,
        "eps_list": EPS_LIST,
        "math_mean_y": float(math["y"].mean()), "bbh_mean_y": float(bbh["y"].mean()),
        "regimes": {},
    }
    for regime in ("drift", "shuffled"):
        rng = np.random.default_rng(SEED + (1 if regime == "drift" else 2))
        res["regimes"][regime] = run_regime(regime, math, bbh, gate, rng)
        print(f"\n=== regime: {regime} ===")
        for eps in EPS_LIST:
            b = res["regimes"][regime][f"eps{eps}"]
            fz, on = b["frozen"], b["online"]
            print(f" eps={eps}: "
                  f"FROZEN post cov={fz['post_cov']:.2f} err={fz['post_err']} "
                  f"viol={fz['post_violation_rate']:.2f} open={fz['fails_open']} | "
                  f"ONLINE post cov={on['post_cov']:.2f} err={on['post_err']} "
                  f"viol={on['post_violation_rate']:.2f} safe={on['fails_safe']} "
                  f"holds={on['holds_eps_post']}")

    # verdict: on the DRIFT stream, does online beat frozen (lower violation rate),
    # and is it drift-specific (frozen NOT systematically violating in shuffled)?
    drift = res["regimes"]["drift"]; shuf = res["regimes"]["shuffled"]
    verdict = {}
    for eps in EPS_LIST:
        d, s = drift[f"eps{eps}"], shuf[f"eps{eps}"]
        verdict[f"eps{eps}"] = {
            "online_lowers_violation_drift": d["online_violation_lt_frozen"],
            "frozen_fails_open_drift": d["frozen"]["fails_open"],
            "online_fails_safe_drift": d["online"]["fails_safe"],
            "online_holds_eps_post_drift": d["online"]["holds_eps_post"],
            "frozen_violation_drift": d["frozen"]["post_violation_rate"],
            "frozen_violation_shuffled": s["frozen"]["post_violation_rate"],
            "drift_specific": bool(
                d["frozen"]["post_violation_rate"] is not None
                and s["frozen"]["post_violation_rate"] is not None
                and d["frozen"]["post_violation_rate"]
                    > s["frozen"]["post_violation_rate"] + 0.1),
        }
    # an eps "wins for online" iff it lowers violation on drift AND online holds eps
    feasible_wins = [eps for eps in EPS_LIST
                     if verdict[f"eps{eps}"]["online_lowers_violation_drift"]
                     and verdict[f"eps{eps}"]["online_holds_eps_post_drift"]]
    failsafe_wins = [eps for eps in EPS_LIST
                     if verdict[f"eps{eps}"]["online_fails_safe_drift"]]
    res["verdict"] = {
        "per_eps": verdict,
        "eps_where_online_restores_validity": feasible_wins,
        "eps_where_online_fails_safe": failsafe_wins,
        "online_value": (
            "restores validity under drift" if feasible_wins else
            "fail-safe only (clamps coverage; cliff blocks valid useful coverage)"
            if failsafe_wins else "no measurable online benefit"),
    }

    (RESULTS / "c2_online_crc_drift.json").write_text(json.dumps(res, indent=2))
    print("\n=== VERDICT ===")
    print(f"  eps where online RESTORES validity under drift: "
          f"{feasible_wins or 'NONE'}")
    print(f"  eps where online FAILS SAFE (frozen fails open): {failsafe_wins or 'NONE'}")
    print(f"  online value: {res['verdict']['online_value']}")
    print(f"-> {RESULTS / 'c2_online_crc_drift.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
