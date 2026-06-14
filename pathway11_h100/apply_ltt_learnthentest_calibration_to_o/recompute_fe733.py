"""P11-FE733 — Learn-then-Test (LTT) calibration of the prefill-DoM
selective-prediction pipeline on MATH-500.

F-8 reports 71.6% answered accuracy at 50% coverage but carries NO finite-sample
risk guarantee. ORCA's LTT recipe supplies one. We treat the prefill-DoM score as
a confidence signal: a problem is *answered* (accept the cheap K=1 path) iff its
DoM score >= lambda, otherwise *abstained* (escalate compute). The selective risk
is the error rate on the answered set, R(lambda) = P(incorrect | answered).

For each threshold lambda in a grid Lambda we form the LTT null H0(lambda):
R(lambda) > delta. Under the worst-case null the answered errors are stochastically
larger than Binomial(n_lambda, delta), so a valid super-uniform p-value is the
binomial lower tail p(lambda) = P(Bin(n_lambda, delta) <= observed_errors). We
reject (declare lambda risk-controlling) when p(lambda) <= eps. Fixed-sequence FWER
control orders thresholds from most conservative (highest lambda, smallest answered
set) toward most liberal (lowest lambda, largest coverage) and stops at the first
non-rejection; lambda* is the last rejected threshold, i.e. the maximal coverage /
savings that still certifies P(R <= delta) >= 1 - eps.

We sweep delta in {0.05, 0.10, 0.15, 0.20} at eps = 0.05 and report
(savings, error rate) tuples on a held-out test split, averaged over random
calibration/test splits, directly comparable to ORCA's Table 3 MATH-500 row.
Direct test of Refutation 3: if LTT-certified coverage at error <= 0.10
substantially exceeds 71.6%, F-8's headline number is superseded.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from scipy.stats import binom

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/ltt_calibration/results.json"

SEED = 9999
EPS = 0.05
DELTAS = [0.05, 0.10, 0.15, 0.20]
N_TRIALS = 200
CAL_FRAC = 0.5
MIN_N = 10  # smallest answered set for which a binomial test has any power


def auroc(scores, labels):
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_split(y, cal_frac, rng):
    """Stratified split into calibration / test index arrays."""
    cal, test = [], []
    for cls in (False, True):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = int(round(len(idx) * cal_frac))
        cal.append(idx[:cut])
        test.append(idx[cut:])
    return np.concatenate(cal), np.concatenate(test)


def ltt_fixed_sequence(scores_cal, y_cal, delta, eps, grid_desc):
    """Fixed-sequence FWER LTT. Returns lambda* (max-coverage risk-controlling
    threshold) or None if no threshold is certified."""
    selected = None
    for lam in grid_desc:
        answered = scores_cal >= lam
        n = int(answered.sum())
        if n < MIN_N:
            continue
        errors = int((~y_cal[answered]).sum())
        # p-value for H0: R(lambda) > delta  (super-uniform under the null)
        pval = float(binom.cdf(errors, n, delta))
        if pval <= eps:
            selected = lam  # certified; advance toward higher coverage
        else:
            break  # fixed-sequence: stop at the first non-rejection
    return selected


def evaluate(scores, y, lam):
    answered = scores >= lam
    n = int(answered.sum())
    if n == 0:
        return 0.0, float("nan"), 0
    err = float((~y[answered]).mean())
    coverage = float(n / len(y))
    return coverage, err, n


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct.shape == (500,) and dom_score.shape == (500,)

    # Higher DoM score => more likely correct (sanity check on direction).
    auroc_full = auroc(dom_score, correct)
    if not np.isnan(auroc_full) and auroc_full < 0.5:
        dom_score = -dom_score  # orient so that high score == answer/accept

    # F-8 reference: error on the answered set at fixed 50% coverage (no guarantee).
    order = np.argsort(-dom_score)
    top_half = order[: len(order) // 2]
    f8_ref = {
        "coverage": float(len(top_half) / len(correct)),
        "answered_error": float((~correct[top_half]).mean()),
        "answered_accuracy": float(correct[top_half].mean()),
    }

    # Full-data LTT (no held-out split) — uses every point for both calib & report.
    grid_full = np.unique(dom_score)[::-1]  # descending
    full_data = {}
    for delta in DELTAS:
        lam = ltt_fixed_sequence(dom_score, correct, delta, EPS, grid_full)
        if lam is None:
            full_data[f"delta_{delta:.2f}"] = {"certified": False}
            continue
        cov, err, n = evaluate(dom_score, correct, lam)
        full_data[f"delta_{delta:.2f}"] = {
            "certified": True,
            "lambda_star": float(lam),
            "savings": cov,
            "error_rate": err,
            "answered_n": n,
        }

    # Split-calibrated LTT: calibrate lambda* on calibration half, report on test
    # half. Averaged over N_TRIALS random stratified splits for stability, with
    # the realized test-risk-violation frequency as an empirical check that the
    # P(R <= delta) >= 1 - eps guarantee holds out of sample.
    rng = np.random.default_rng(SEED)
    split_results = {}
    for delta in DELTAS:
        covs, errs, ns = [], [], []
        n_certified = 0
        n_test_violations = 0  # test error > delta among certified trials
        for _ in range(N_TRIALS):
            cal_idx, test_idx = stratified_split(correct, CAL_FRAC, rng)
            grid_cal = np.unique(dom_score[cal_idx])[::-1]
            lam = ltt_fixed_sequence(
                dom_score[cal_idx], correct[cal_idx], delta, EPS, grid_cal
            )
            if lam is None:
                continue
            n_certified += 1
            cov, err, n = evaluate(dom_score[test_idx], correct[test_idx], lam)
            covs.append(cov)
            errs.append(err)
            ns.append(n)
            if not np.isnan(err) and err > delta:
                n_test_violations += 1
        if n_certified == 0:
            split_results[f"delta_{delta:.2f}"] = {
                "certified_frac": 0.0,
                "note": "no threshold certified on any calibration split",
            }
            continue
        split_results[f"delta_{delta:.2f}"] = {
            "certified_frac": float(n_certified / N_TRIALS),
            "savings_mean": float(np.mean(covs)),
            "savings_std": float(np.std(covs)),
            "test_error_mean": float(np.nanmean(errs)),
            "test_error_std": float(np.nanstd(errs)),
            "answered_n_mean": float(np.mean(ns)),
            "test_violation_frac": float(n_test_violations / n_certified),
            "guarantee_holds": bool(n_test_violations / n_certified <= EPS + 0.02),
        }

    out = {
        "experiment": "P11-FE733",
        "method": "LTT fixed-sequence FWER selective-risk calibration",
        "eps": EPS,
        "deltas": DELTAS,
        "n_trials": N_TRIALS,
        "cal_frac": CAL_FRAC,
        "min_answered_n": MIN_N,
        "auroc_dom_full": float(auroc_full),
        "f8_reference_coverage_0.5": f8_ref,
        "ltt_full_data": full_data,
        "ltt_split_calibrated": split_results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())