"""metrics.py — scoring utilities for the edge program.

Re-exports the verified nocompute/lib.py substrate (DomProbe, oof_dom_scores,
bootstrap_auroc, delong_ci single-AUROC CI, ece_score, risk_coverage_curve,
aurc, oracle_aurc) and ADDS what the spec's adjudication needs but lib lacks:
  - frozen_folds(n, y, seed)         one fixed StratifiedKFold map, reused by all (V2-5)
  - auroc_sym(y, s)                  sign-agnostic AUROC (free baselines flip sign)
  - delong_paired_test(y, sa, sb)    paired DeLong z/p for two scorers on SAME items
  - mcnemar_test(correct_a, correct_b)  paired McNemar for selection (1B.1)
  - overfit_gap(in_domain, transfer) the alarm metric
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import the verified substrate.
sys.path.insert(0, str(Path("/home/musicofhel/topo-confidence/nocompute")))
from lib import (  # noqa: E402
    DomProbe, oof_dom_scores, oof_dom_calibrated,
    bootstrap_auroc, delong_ci, ece_score,
    risk_coverage_curve, aurc, oracle_aurc, SEED, N_FOLDS,
)

__all__ = [
    "DomProbe", "oof_dom_scores", "oof_dom_calibrated",
    "bootstrap_auroc", "delong_ci", "ece_score",
    "risk_coverage_curve", "aurc", "oracle_aurc", "SEED", "N_FOLDS",
    "frozen_folds", "auroc_sym", "auroc_with_ci",
    "delong_paired_test", "mcnemar_test", "overfit_gap",
]


def frozen_folds(y: np.ndarray, seed: int = SEED, n_folds: int = N_FOLDS):
    """One fixed StratifiedKFold partition, reused by every method (V2-5).

    Returns list of (train_idx, test_idx). Stratified on the binary label so the
    paired DeLong always compares identical test items across methods.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def auroc_sym(y: np.ndarray, scores: np.ndarray) -> float:
    """Sign-agnostic AUROC = max(a, 1-a). Free baselines (length, logprob) carry
    signal whose sign is conventional, not learned — report the magnitude."""
    a = roc_auc_score(y, scores)
    return float(max(a, 1.0 - a))


def auroc_with_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 2000,
                  seed: int = SEED) -> dict:
    """Point AUROC + bootstrap BCa/percentile CI + DeLong CI + n. Thin wrapper
    pairing lib.bootstrap_auroc and lib.delong_ci with the per-cell n."""
    b = bootstrap_auroc(y, scores, n_resamples=n_boot, seed=seed)
    dl = delong_ci(y, scores)
    return {**b, "delong_lo": dl["delong_lo"], "delong_hi": dl["delong_hi"],
            "se": dl["se"], "n": int(len(y)),
            "n_pos": int(y.sum()), "n_neg": int((~y.astype(bool)).sum())}


def _delong_structural(y: np.ndarray, score: np.ndarray):
    """Return (V10, V01, auc) DeLong placement components for one scorer."""
    yb = y.astype(bool)
    pos = score[yb]
    neg = score[~yb]
    m, n = len(pos), len(neg)
    V10 = np.array([((neg < p).sum() + 0.5 * (neg == p).sum()) / n for p in pos])
    V01 = np.array([((pos > q).sum() + 0.5 * (pos == q).sum()) / m for q in neg])
    return V10, V01, V10.mean()


def delong_paired_test(y: np.ndarray, score_a: np.ndarray,
                       score_b: np.ndarray) -> dict:
    """Paired DeLong test that AUROC_a != AUROC_b on the SAME items (Sun & Xu).

    Returns auc_a, auc_b, delta, se_delta, z, p (two-sided). score_a/score_b must
    be aligned to the same y vector (same test items)."""
    yb = y.astype(bool)
    Va10, Va01, auc_a = _delong_structural(yb, score_a)
    Vb10, Vb01, auc_b = _delong_structural(yb, score_b)
    m = int(yb.sum())
    n = int((~yb).sum())
    # 2x2 covariance of (auc_a, auc_b) via DeLong placement covariances.
    S10 = np.cov(np.vstack([Va10, Vb10]))  # (2,2) over the m positives
    S01 = np.cov(np.vstack([Va01, Vb01]))  # (2,2) over the n negatives
    S = S10 / m + S01 / n
    var_delta = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = float(np.sqrt(max(var_delta, 0.0)))
    delta = float(auc_a - auc_b)
    z = delta / se if se > 0 else 0.0
    p = float(2 * (1 - norm.cdf(abs(z))))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "delta": delta,
            "se": se, "z": float(z), "p": p}


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar for two selection policies (1B.1). correct_* are per-item
    booleans (did policy X answer item i correctly). Tests whether the policies
    differ in accuracy. Uses the exact binomial on discordant pairs."""
    from scipy.stats import binomtest
    a = correct_a.astype(bool)
    b = correct_b.astype(bool)
    b01 = int((~a & b).sum())   # a wrong, b right
    b10 = int((a & ~b).sum())   # a right, b wrong
    nd = b01 + b10
    if nd == 0:
        p = 1.0
    else:
        p = float(binomtest(b01, nd, 0.5).pvalue)
    return {"acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "b_a_wrong_b_right": b01, "b_a_right_b_wrong": b10,
            "n_discordant": nd, "p": p}


def overfit_gap(in_domain_auroc: float, transfer_auroc: float) -> float:
    """The alarm: in-domain OOF minus transfer. Lower = more robust."""
    return float(in_domain_auroc - transfer_auroc)
