"""probes.py — candidate readouts, uniform fit/score interface.

Every probe exposes:
    name              : str
    tiers             : "all" (fixed-dim/scalar, enters T1-T5) or
                        "t1t2" (hidden-dim-bound supervised, no cross-model)
    featurize(cell)   : Cell -> X (n, d) feature matrix
    kind              : "dom" | "logreg"  — how the d-dim X is turned into a score

transfer.py composes these into OOF / cross-cell scores and runs the incremental
gate. Hidden-dim-bound probes (DoM/ridge-LR on raw activations) are "t1t2";
scalar and depth-grid features are "all".

Feature provenance:
  - DoM / ridge-LR        : raw L19 activation (prefill or last), hidden-dim-bound
  - free_baseline         : [n_gen_tokens, mean_logprob]                  (Tier-B)
  - length_only           : [n_gen_tokens]                                (Tier-B)
  - mean_logprob_only     : [mean_logprob]                                (Tier-B)
  - coe_profile           : per-layer mag/ang of the depth trajectory (X_mean)
  - coe_depthgrid         : coe_profile resampled to a fixed depth-fraction grid
                            (the only cross-model-eligible geometric form)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, RidgeClassifier,
)
from sklearn.preprocessing import StandardScaler

# C grid for the regularised concat (FE421 recipe).
_RIDGE_LR_CS = [0.001, 0.01, 0.1, 1.0, 10.0]

from metrics import DomProbe


# ---------------------------------------------------------------------------
# Feature constructors
# ---------------------------------------------------------------------------

def coe_magang_profile(X_layers: np.ndarray) -> np.ndarray:
    """CoE depth-axis profile from per-layer mean states (n, L, H) -> (n, 2*(L-1)).

    Mirrors pathway8_layerwise/coe_features.compute_coe_single but vectorised over
    problems and fed the already-mean-pooled per-layer states (X_mean[i] is exactly
    that script's `per_layer_mean`). Per-layer magnitude-of-change and angle-change.
    """
    n, L, H = X_layers.shape
    dif = X_layers[:, 1:, :] - X_layers[:, :-1, :]          # (n, L-1, H)
    mag = np.linalg.norm(dif, axis=2)                        # (n, L-1)
    norms = np.linalg.norm(X_layers, axis=2)                 # (n, L)
    denom = norms[:, :-1] * norms[:, 1:] + 1e-12
    cos = np.einsum("nld,nld->nl", X_layers[:, :-1, :], X_layers[:, 1:, :]) / denom
    ang = np.arccos(np.clip(cos, -1.0, 1.0))                 # (n, L-1)
    return np.concatenate([mag, ang], axis=1)               # (n, 2(L-1))


def depth_grid_resample(profile: np.ndarray, n_grid: int = 16) -> np.ndarray:
    """Resample a per-layer profile (n, L) onto a fixed depth-fraction grid.

    Makes 29-, 33-, 17-layer models share a feature space (the depth analog of
    fixed-K eigvals). Linear interpolation on layer-fraction in [0,1].
    """
    n, L = profile.shape
    src = np.linspace(0.0, 1.0, L)
    grid = np.linspace(0.0, 1.0, n_grid)
    out = np.empty((n, n_grid), dtype=np.float64)
    for i in range(n):
        out[i] = np.interp(grid, src, profile[i])
    return out


def coe_depthgrid_features(X_layers: np.ndarray, n_grid: int = 16,
                           normalize: bool = True) -> np.ndarray:
    """Cross-model-portable CoE: split mag/ang, resample each onto the depth grid.

    Magnitudes are trace-normalised per problem (scale-invariant); angles are
    scale-invariant already. Output (n, 2*n_grid).
    """
    n, L, H = X_layers.shape
    half = L - 1
    ma = coe_magang_profile(X_layers)
    mag = ma[:, :half]
    ang = ma[:, half:]
    if normalize:
        # per-problem total displacement normalisation (chosen on train only in
        # transfer.py via the scaler; here we use a within-row norm that is
        # itself scale-free).
        scale = np.linalg.norm(mag, axis=1, keepdims=True) + 1e-12
        mag = mag / scale
    mag_g = depth_grid_resample(mag, n_grid)
    ang_g = depth_grid_resample(ang, n_grid)
    return np.concatenate([mag_g, ang_g], axis=1)


# ---------------------------------------------------------------------------
# Probe registry
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    name: str
    tiers: str                      # "all" | "t1t2"
    kind: str                       # "dom" | "logreg" | "ridge"
    featurize: Callable             # Cell -> (n, d)
    standardize: bool = True        # z-score features (fit on train) before logreg
    meta: dict = field(default_factory=dict)


def _build_registry():
    P = {}

    # --- Baselines / free signals (enter every tier) ---
    P["free_baseline"] = Probe(
        "free_baseline", "all", "logreg",
        lambda c: np.column_stack([c.n_gen_tokens, c.mean_logprob]),
        meta={"desc": "length + mean_logprob (pinned Tier-B free baseline)"})
    P["length_only"] = Probe(
        "length_only", "all", "logreg",
        lambda c: c.n_gen_tokens.reshape(-1, 1),
        meta={"desc": "generation length only"})
    P["mean_logprob_only"] = Probe(
        "mean_logprob_only", "all", "logreg",
        lambda c: c.mean_logprob.reshape(-1, 1),
        meta={"desc": "mean token logprob only (P10-FE23)"})

    # --- Arm 1: hidden-dim-bound supervised (T1/T2 only) ---
    P["prefill_dom"] = Probe(
        "prefill_dom", "t1t2", "dom",
        lambda c: c.X("prefill"), standardize=False,
        meta={"desc": "L19 prefill DoM — PINNED INCUMBENT (0.7731)"})
    P["last_dom"] = Probe(
        "last_dom", "t1t2", "dom",
        lambda c: c.X("last"), standardize=False,
        meta={"desc": "L19 final-token DoM"})
    P["mean_dom"] = Probe(
        "mean_dom", "t1t2", "dom",
        lambda c: c.X("mean"), standardize=False,
        meta={"desc": "L19 mean-pooled DoM (Tier-B)"})
    P["concat_ridge"] = Probe(
        "concat_ridge", "t1t2", "ridge_lr",
        lambda c: np.concatenate([c.X("prefill"), c.X("last")], axis=1),
        standardize=False,
        meta={"desc": "L19 prefill+final concat, LogisticRegressionCV L2 (FE421, 0.8509)"})

    # --- Arm 2: depth-axis layer-profile family ---
    P["coe_profile"] = Probe(
        "coe_profile", "t1t2", "logreg",   # raw per-layer-index form: in-domain only
        lambda c: coe_magang_profile(c.X_all_layers("mean")),
        meta={"desc": "CoE mag/ang per-layer profile (raw, in-domain parity)"})
    P["coe_depthgrid"] = Probe(
        "coe_depthgrid", "all", "logreg",  # the cross-model-eligible form
        lambda c: coe_depthgrid_features(c.X_all_layers("mean"), n_grid=16),
        meta={"desc": "CoE depth-grid resampled + scale-normalised (T3-T5 eligible)"})

    return P


REGISTRY = _build_registry()


# ---------------------------------------------------------------------------
# Fit / score over a frozen fold (OOF) or train->test cell
# ---------------------------------------------------------------------------

def _make_estimator(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0)
    if kind == "ridge":
        return RidgeClassifier(alpha=1.0)
    if kind == "ridge_lr":  # FE421 recipe: L2 LogisticRegressionCV over the C grid
        return LogisticRegressionCV(Cs=_RIDGE_LR_CS, penalty="l2", solver="lbfgs",
                                    max_iter=5000, random_state=9999, cv=3)
    raise ValueError(kind)


def _score_estimator(est, X):
    """Probability-like score for AUROC from a fitted sklearn estimator."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)  # RidgeClassifier


def fit_score_oof(probe: Probe, cell, folds) -> np.ndarray:
    """Out-of-fold scores on one cell using the frozen fold map."""
    X = probe.featurize(cell)
    y = cell.y
    scores = np.zeros(len(y), dtype=np.float64)
    for tr, te in folds:
        if probe.kind == "dom":
            p = DomProbe().fit(X[tr], y[tr])
            scores[te] = p.score(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
            if probe.standardize:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            est = _make_estimator(probe.kind).fit(Xtr, y[tr])
            scores[te] = _score_estimator(est, Xte)
    return scores


def fit_score_transfer(probe: Probe, train_cell, test_cell) -> np.ndarray:
    """Fit on the whole train cell, score the whole test cell (cross-distribution).

    For hidden-dim-bound (t1t2) probes the dims must match (same model) — the
    caller guarantees this by tier eligibility.
    """
    Xtr = probe.featurize(train_cell)
    Xte = probe.featurize(test_cell)
    ytr = train_cell.y
    if probe.kind == "dom":
        return DomProbe().fit(Xtr, ytr).score(Xte)
    if probe.standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    est = _make_estimator(probe.kind).fit(Xtr, ytr)
    return _score_estimator(est, Xte)
