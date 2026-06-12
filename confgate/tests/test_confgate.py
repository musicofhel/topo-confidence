"""confgate test suite.

Tests that need the topo-confidence repo caches skip gracefully when the
caches are absent (installed-package scenario).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from confgate import (FreeGate, PreflightGate, certify, certify_cross_domain,
                      pinned_families, pinned_meta)

REPO = Path(os.environ.get("CONFGATE_REPO", Path.home() / "topo-confidence"))
CACHE_1P5B = REPO / "nocompute" / "cache" / "math500_1p5b.npz"
PHASE3_GEMMA = (REPO / "pathway11_h100" / "generalization_edge" / "results"
                / "phase3_gemma.npz")

EXPECTED_FAMILIES = {"qwen2.5-1.5b", "qwen2.5-7b", "smollm2-1.7b",
                     "gemma-2-2b", "olmo-2-1b"}


def _need(path: Path):
    if not path.exists():
        pytest.skip(f"repo cache not available: {path}")


# --------------------------------------------------------------- (a) pins load
def test_pinned_gates_load_all_families():
    fams = set(pinned_families())
    assert fams == EXPECTED_FAMILIES
    for fam in fams:
        g = FreeGate.from_pinned(fam)
        assert g.coef.shape == (2,)
        assert g.scaler_mean.shape == (2,)
        assert np.all(g.scaler_scale > 0)
        assert 0.5 < g.meta["resub_auroc"] <= 1.0
        assert g.meta["n"] >= 500


def test_pinned_preflight_loads():
    pf = PreflightGate.from_pinned("qwen2.5-1.5b")
    assert pf.scaler_scale > 0
    s = pf.score([10, 100, 1000])
    assert s.shape == (3,)
    assert np.all((s > 0) & (s < 1))


def test_pinned_meta_anchors_present():
    meta = pinned_meta()
    a = meta["anchors"]
    assert set(a["t5_heldout_oof"]) == {"smollm2-1.7b", "gemma-2-2b", "olmo-2-1b"}
    for fam in a["t5_heldout_oof"].values():
        assert fam["free_baseline_auroc"] >= 0.70 and fam["T5_pass_0.70"]
    assert a["cascade"]["hybrid_significant"] is True
    assert a["conformal"]["cross_scale_math1.5b_to_7b_eps0.2"]["k0"]["validity"] == 1.0
    assert a["conformal"]["cross_domain_math_to_bbh_eps0.2"]["k0"]["validity"] == 0.0


# ------------------------------------------------- (b) determinism + shape
def test_score_deterministic_and_shaped():
    g = FreeGate.from_pinned("qwen2.5-1.5b")
    rng = np.random.default_rng(0)
    L = rng.uniform(1, 1024, 64)
    lp = rng.uniform(-3, 0, 64)
    s1 = g.score(L, lp)
    s2 = g.score(L, lp)
    assert s1.shape == (64,)
    assert np.array_equal(s1, s2)
    assert np.all((s1 > 0) & (s1 < 1))


# --------------------------------------- (c) refit reproduces the pinned resub
def test_refit_reproduces_pinned_resub_auroc():
    _need(CACHE_1P5B)
    from sklearn.metrics import roc_auc_score

    d = np.load(CACHE_1P5B, allow_pickle=True)
    y = d["y"].astype(bool)
    L = d["n_gen_tokens"].astype(np.float64)
    lp = d["mean_logprob"].astype(np.float64)
    g = FreeGate.fit(L, lp, y)
    auroc = roc_auc_score(y, g.score(L, lp))
    pinned = FreeGate.from_pinned("qwen2.5-1.5b").meta["resub_auroc"]
    assert abs(auroc - pinned) < 1e-6


# ------------------------------------------------ (d) cross-domain refusal
def test_certify_cross_domain_raises():
    with pytest.raises(NotImplementedError, match="phase4_recalibration.json"):
        certify_cross_domain()


def test_certify_in_domain_basic():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random(n) < 0.5
    scores = np.clip(0.5 * y + rng.normal(0.25, 0.15, n), 0, 1)
    cert = certify(scores, y, eps=0.2, delta=0.1)
    assert cert["feasible"]
    ans = scores >= cert["tau"]
    assert (1 - y[ans].mean()) <= 0.2 + 1e-12   # holds on calibration by design


# ------------------------- (e) OOF protocol reproduces the pinned T5 anchor
def test_oof_gemma_reproduces_pinned_anchor():
    _need(PHASE3_GEMMA)
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    d = np.load(PHASE3_GEMMA, allow_pickle=True)
    y = d["y"].astype(bool)
    L = d["n_gen_tokens"].astype(np.float64)
    lp = d["mean_logprob"].astype(np.float64)
    X = np.column_stack([L, lp])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        g = FreeGate.fit(L[tr], lp[tr], y[tr])
        oof[te] = g.score(L[te], lp[te])
    auroc = roc_auc_score(y, oof)

    anchor = pinned_meta()["anchors"]["t5_heldout_oof"]["gemma-2-2b"][
        "free_baseline_auroc"]
    assert abs(anchor - 0.8439) < 1e-3          # the pinned number itself
    assert abs(auroc - anchor) < 1e-3           # our recipe reproduces it
