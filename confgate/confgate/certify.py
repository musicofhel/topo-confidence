"""Split-conformal LTT certificates (Clopper-Pearson upper bound).

Math is an exact port of
pathway11_h100/generalization_edge/exp_1b3_conformal.py (SPEC v6, EXP-82):
on calibration data, pick the smallest threshold tau (=> max coverage) such
that the Clopper-Pearson upper bound on the answered-set error is <= eps at
confidence 1 - delta ("answer if score >= tau").

SUPPORTED transfer path (pinned, results/phase4_recalibration.json,
math1.5b_to_7b / free_baseline / eps=0.2):
  * cross-scale ZERO-SHOT (k=0): calibrate on Qwen2.5-1.5B MATH, deploy the
    same thresholded gate on Qwen2.5-7B MATH -> feasible_frac 1.0,
    validity 1.0 at 0.60 coverage.
  * k-label recalibration is FEASIBILITY-RESTORING ONLY past k>=32 (CP-upper
    small-sample penalty makes k=8/16 infeasible); it NEVER improves coverage
    over zero-shot (k=64 @ eps=0.2 coverage 0.29 < 0.60). Prefer k=0.

UNSUPPORTED (documented product limit, NOT a bug): cross-domain certificates.
See certify_cross_domain().

scipy is used for the Clopper-Pearson bound; it ships transitively with
scikit-learn (already a hard dependency).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta

DEFAULT_DELTA = 0.1

_CROSS_DOMAIN_MSG = (
    "Cross-domain conformal certificates are infeasible with any light head "
    "(documented product limit, SPEC v7 Phase 0 resolution). Pinned evidence: "
    "pathway11_h100/generalization_edge/results/phase4_recalibration.json "
    "(math_to_bbh, free_baseline, CP-upper light-head recalibration, R=200): "
    "validity 0.0 at eps=0.2 for ALL k in {0, 32, 64} where feasible "
    "(k=8/16 entirely infeasible); best anywhere = validity 0.41 at "
    "(k=32, eps=0.3) — never near the 0.9 target. 1b3_conformal.json::"
    "transfer_HH (calib MATH -> deploy BBH) agrees: infeasible at eps<=0.2, "
    "coverage 0.0 at eps=0.3. The supported path is the cross-scale zero-shot "
    "(k=0) certificate: calibrate-on-1.5B deploy-on-7B is valid (validity 1.0 "
    "at 0.60 coverage, eps=0.2). For a NEW DOMAIN, recalibrate on >=32 true "
    "in-domain labels and certify in-domain — do not transfer the certificate."
)


def cp_upper(k_err: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper bound on error rate given k_err errors in n trials."""
    if n == 0:
        return 1.0
    if k_err == n:
        return 1.0
    return float(_beta.ppf(1 - delta, k_err + 1, n - k_err))


def choose_tau(scores_cal, y_cal, eps: float, delta: float):
    """Smallest tau (=> max coverage) s.t. CP-upper(answered error) <= eps.
    'answer if score >= tau'. Returns tau or None if infeasible."""
    scores_cal = np.asarray(scores_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal)
    order = np.argsort(-scores_cal)            # high score first
    s_sorted = scores_cal[order]
    y_sorted = y_cal[order].astype(bool)
    best_tau = None
    n_ans = 0
    n_err = 0
    for i in range(len(s_sorted)):
        n_ans += 1
        n_err += int(not y_sorted[i])
        ub = cp_upper(n_err, n_ans, delta)
        if ub <= eps:
            best_tau = s_sorted[i]              # can answer down to here
    return best_tau


def certify(cal_scores, cal_y, eps: float, delta: float = DEFAULT_DELTA) -> dict:
    """Split-conformal LTT certificate from a labelled calibration set.

    Returns a dict with the threshold and the guarantee. Deployment rule:
    answer when gate score >= cert['tau'], abstain (or escalate) otherwise.
    Guarantee: with probability >= 1-delta over the calibration draw, the
    error rate of the answered set is <= eps.
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    cal_y = np.asarray(cal_y).astype(bool)
    if len(cal_scores) != len(cal_y):
        raise ValueError("cal_scores and cal_y must be aligned")
    tau = choose_tau(cal_scores, cal_y, eps, delta)
    out = {
        "method": "split-conformal LTT (Clopper-Pearson upper bound)",
        "eps": float(eps),
        "delta": float(delta),
        "n_cal": int(len(cal_y)),
        "feasible": tau is not None,
    }
    if tau is None:
        out.update({
            "tau": None,
            "guarantee": None,
            "note": (f"infeasible: no threshold achieves CP-upper <= {eps} at "
                     f"confidence {1 - delta} on n={len(cal_y)} calibration "
                     "points (small-sample penalty); add labels or raise eps"),
        })
        return out
    ans = cal_scores >= tau
    out.update({
        "tau": float(tau),
        "cal_coverage": float(ans.mean()),
        "cal_answered_accuracy": float(cal_y[ans].mean()) if ans.sum() else None,
        "guarantee": (f"P(answered-set error <= {eps}) >= {1 - delta} "
                      "over the calibration draw; answer iff score >= tau"),
    })
    return out


def apply_certificate(cert: dict, scores, y=None) -> dict:
    """Apply a certificate's threshold to deployment scores.

    Pass `y` (true labels) only for auditing; deployment does not need it.
    """
    if not cert.get("feasible"):
        raise ValueError("cannot apply an infeasible certificate")
    scores = np.asarray(scores, dtype=np.float64)
    ans = scores >= cert["tau"]
    out = {"tau": cert["tau"], "eps": cert["eps"], "delta": cert["delta"],
           "coverage": float(ans.mean()), "n": int(len(scores)),
           "answered_idx": np.nonzero(ans)[0]}
    if y is not None:
        y = np.asarray(y).astype(bool)
        if ans.sum():
            risk = float(1 - y[ans].mean())
            out.update({"test_risk": risk,
                        "risk_le_eps": bool(risk <= cert["eps"]),
                        "answered_accuracy": float(y[ans].mean())})
    return out


def certify_grouped(cal_scores, cal_y, cal_groups, eps: float,
                    delta: float = DEFAULT_DELTA) -> dict:
    """Group-conditional (Mondrian) split-conformal LTT certificate.

    A *marginal* certificate (`certify`) bounds the POOLED answered-set error
    <= eps, but that single guarantee can hide wide per-group disparity: some
    groups are silently over budget while others are far under. This computes
    a separate per-group threshold tau_g (Mondrian CP) so the answered-set
    error of EACH group g is certified <= eps, and reports the cross-group
    coverage gap the marginal certificate hides.

    Deployment rule: for an item in group g, answer iff its gate score >=
    cert['groups'][g]['tau'] (abstain/escalate otherwise, including for any
    group whose certificate is infeasible). Use `apply_grouped`.

    Honest expectation (pinned: in-domain MATH categories, EXP-86/C3): the
    conditional certificate is VALID for high-accuracy groups but typically
    LOW coverage, and needs a per-group label budget (~16-32 true in-group
    labels — fewer trips the CP-upper small-sample penalty to infeasible).
    Cross-domain it stays infeasible (the accuracy cliff is per-group too);
    see certify_cross_domain().

    Parameters
    ----------
    cal_scores, cal_y, cal_groups : aligned 1-D arrays
        Calibration gate scores, TRUE labels, and a group key per item.
    eps, delta : float
        Per-group answered-error target and 1-delta calibration confidence.
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    cal_y = np.asarray(cal_y).astype(bool)
    cal_groups = np.asarray(cal_groups)
    if not (len(cal_scores) == len(cal_y) == len(cal_groups)):
        raise ValueError("cal_scores, cal_y, cal_groups must be aligned")

    glist = sorted({str(g) for g in cal_groups.tolist()})
    keys = cal_groups.astype(str)
    marg_tau = choose_tau(cal_scores, cal_y, eps, delta)   # pooled threshold

    groups: dict[str, dict] = {}
    marg_covs: list[float] = []
    n_violating = 0
    for g in glist:
        m = keys == g
        sg, yg = cal_scores[m], cal_y[m]
        cert_g = certify(sg, yg, eps, delta)               # conditional cert
        cert_g["n_group"] = int(m.sum())
        cert_g["base_acc"] = float(yg.mean())
        # diagnostic: this group's coverage/error UNDER the marginal threshold
        if marg_tau is None:
            cert_g.update({"marginal_coverage": 0.0,
                           "marginal_answered_err": None,
                           "marginal_violates_eps": False})
        else:
            ans = sg >= marg_tau
            mc = float(ans.mean())
            me = float(1 - yg[ans].mean()) if ans.sum() else None
            mv = bool(me is not None and me > eps)
            n_violating += int(mv)
            marg_covs.append(mc)
            cert_g.update({"marginal_coverage": mc,
                           "marginal_answered_err": me,
                           "marginal_violates_eps": mv})
        groups[g] = cert_g

    certifiable = [g for g in glist
                   if groups[g]["feasible"]
                   and groups[g].get("cal_coverage", 0.0) > 0]
    out = {
        "method": "group-conditional (Mondrian) split-conformal LTT",
        "eps": float(eps),
        "delta": float(delta),
        "n_cal": int(len(cal_y)),
        "n_groups": len(glist),
        "groups": groups,
        "marginal_tau": None if marg_tau is None else float(marg_tau),
        "marginal_coverage_gap": (float(max(marg_covs) - min(marg_covs))
                                  if marg_covs else 0.0),
        "n_groups_marginal_violating": int(n_violating),
        "certifiable_groups": certifiable,
        "guarantee": (
            f"For each group g, P(group-g answered-set error <= {eps}) >= "
            f"{1 - delta} over the calibration draw; for an item in group g, "
            "answer iff score >= groups[g]['tau']. NOTE: conditional certs "
            "are valid but typically LOW coverage and need ~16-32 true "
            "in-group labels; groups with feasible=False must abstain."),
    }
    return out


def apply_grouped(cert: dict, scores, groups, y=None) -> dict:
    """Apply a group-conditional certificate: per-group threshold per item.

    Items whose group has an infeasible certificate (or an unseen group key)
    abstain. Pass `y` (true labels) only for a per-group risk audit.
    """
    scores = np.asarray(scores, dtype=np.float64)
    keys = np.asarray(groups).astype(str)
    if len(scores) != len(keys):
        raise ValueError("scores and groups must be aligned")
    ans = np.zeros(len(scores), dtype=bool)
    for g, cg in cert["groups"].items():
        if not cg.get("feasible"):
            continue
        m = keys == g
        ans[m] = scores[m] >= cg["tau"]
    out = {"eps": cert["eps"], "delta": cert["delta"],
           "coverage": float(ans.mean()), "n": int(len(scores)),
           "answered_idx": np.nonzero(ans)[0]}
    if y is not None:
        y = np.asarray(y).astype(bool)
        if ans.sum():
            out["test_risk"] = float(1 - y[ans].mean())
            out["risk_le_eps"] = bool(out["test_risk"] <= cert["eps"])
        per_group = {}
        for g in cert["groups"]:
            gm = keys == g
            am = gm & ans
            if am.sum():
                per_group[g] = {
                    "coverage": float(am.sum() / max(int(gm.sum()), 1)),
                    "risk": float(1 - y[am].mean()),
                    "risk_le_eps": bool((1 - y[am].mean()) <= cert["eps"])}
        out["per_group"] = per_group
    return out


def certify_cross_scale(cal_scores, cal_y, eps: float = 0.2,
                        delta: float = DEFAULT_DELTA) -> dict:
    """Cross-scale ZERO-SHOT certificate (the supported transfer path).

    Calibrate on the small model's labelled data (e.g. Qwen2.5-1.5B MATH) and
    deploy the SAME threshold on the larger model (Qwen2.5-7B), k=0 — no
    target labels. Pinned validation (phase4_recalibration.json,
    math1.5b_to_7b, free_baseline, eps=0.2): validity 1.0 at 0.60 coverage.
    k-label recalibration on the target is feasibility-restoring only past
    k>=32 and never coverage-improving over zero-shot — prefer k=0.
    """
    cert = certify(cal_scores, cal_y, eps, delta)
    cert["transfer"] = "cross-scale zero-shot (k=0); pinned valid at eps=0.2"
    return cert


def certify_cross_domain(*args, **kwargs):
    """Cross-domain conformal certificates: REFUSED (documented product limit)."""
    raise NotImplementedError(_CROSS_DOMAIN_MSG)
