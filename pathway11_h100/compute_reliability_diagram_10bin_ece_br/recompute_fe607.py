"""P11-FE607 — Calibration audit of prefill L19 DoM scores (F-2).

F-2 reports AUROC=0.7731, which is invariant to monotone rescaling and silent
on calibration. This script measures whether the prefill-DoM ranking signal
corresponds to a well-calibrated *probability*, or whether it needs post-hoc
Platt scaling, by computing a 10-bin reliability diagram (ECE), Brier score,
Brier Skill Score (BSS vs base-rate predictor), MCE, and over-confidence ratios
OCR@{0.7,0.9} on the cached Qwen-2.5-1.5B MATH-500 1024-tok 5-fold OOF predictions.

Procedure (per fold, leak-free): fit a Platt sigmoid (1-D logistic regression,
score -> P(correct)) on the training fold, apply to the held-out test fold,
aggregate the OOF calibrated probabilities, then bin. We report metrics for the
raw min-max-mapped DoM (no calibration) and the Platt-calibrated DoM so the lift
from post-hoc scaling is explicit, and compare ECE against KalshiBench's
frontier-model calibration table (acc 0.64-0.69, ECE 0.120-0.395).
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/dom_calibration/results.json"

N_FOLDS = 5
N_BINS = 10
SEED = 9999
OCR_THRESHOLDS = (0.7, 0.9)

# KalshiBench frontier-model calibration table (external reference, KalshiBench
# paper). Accuracies are decoupled from ECE: similar accuracy, ~3x ECE range.
KALSHIBENCH = {
    "accuracy_range": [0.64, 0.69],
    "ece_range": [0.120, 0.395],
    "note": "frontier models: similar accuracy, 3x ECE spread (acc/calibration decoupled)",
}


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y, k, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def platt_fit(scores, labels, iters=500, lr=0.5):
    """1-D logistic regression P(y=1) = sigmoid(a*z + b) on standardized scores."""
    mu = float(scores.mean()); sd = float(scores.std()) or 1.0
    z = (scores - mu) / sd
    y = labels.astype(np.float64)
    a, b = 0.0, 0.0
    n = len(z)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        ga = float(((p - y) * z).mean())
        gb = float((p - y).mean())
        a -= lr * ga
        b -= lr * gb
    return a, b, mu, sd


def platt_apply(scores, a, b, mu, sd):
    z = (scores - mu) / sd
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def reliability(probs, labels, n_bins):
    """Equal-width 10-bin reliability diagram -> ECE, MCE, per-bin table."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    mce = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "conf": None, "acc": None, "gap": None})
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        gap = abs(conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt,
                     "conf": conf, "acc": acc, "gap": float(gap)})
    return float(ece), float(mce), bins


def brier(probs, labels):
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


def brier_skill_score(probs, labels):
    base = float(labels.mean())
    bs = brier(probs, labels)
    bs_ref = brier(np.full_like(probs, base), labels)
    if bs_ref <= 0:
        return float("nan")
    return float(1.0 - bs / bs_ref)


def ocr(probs, labels, t):
    """Over-confidence ratio at threshold t: among high-confidence predictions
    (p>=t), the mean(confidence) - mean(accuracy) gap. Positive = overconfident."""
    mask = probs >= t
    cnt = int(mask.sum())
    if cnt == 0:
        return {"threshold": t, "count": 0, "conf": None, "acc": None, "ocr": None}
    conf = float(probs[mask].mean())
    acc = float(labels[mask].mean())
    return {"threshold": t, "count": cnt, "conf": conf, "acc": acc,
            "ocr": float(conf - acc)}


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    cache = np.load(CACHE)
    y = cache["correct"].astype(bool)
    dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert y.shape == (500,) and dom.shape == (500,)

    n = len(y)
    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Leak-free OOF Platt calibration.
    oof_platt = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        a, b, mu, sd = platt_fit(dom[train_mask], y[train_mask])
        oof_platt[test_idx] = platt_apply(dom[test_idx], a, b, mu, sd)

    # Raw "probability" baseline: global min-max of DoM into [0,1] (no calibration).
    lo, hi = float(dom.min()), float(dom.max())
    raw_prob = (dom - lo) / (hi - lo) if hi > lo else np.full(n, 0.5)

    yf = y.astype(np.float64)
    base_rate = float(y.mean())

    ece_raw, mce_raw, bins_raw = reliability(raw_prob, y, N_BINS)
    ece_cal, mce_cal, bins_cal = reliability(oof_platt, y, N_BINS)

    out = {
        "experiment": "P11-FE607",
        "description": "Calibration audit (ECE/Brier/BSS/MCE/OCR) of prefill L19 DoM, 1024tok",
        "n": n,
        "n_folds": N_FOLDS,
        "n_bins": N_BINS,
        "base_rate_correct": base_rate,
        "auroc_dom": auroc(dom, y),
        "auroc_oof_platt": auroc(oof_platt, y),
        "raw_minmax": {
            "ece": ece_raw,
            "mce": mce_raw,
            "brier": brier(raw_prob, y),
            "brier_skill_score": brier_skill_score(raw_prob, y),
            "ocr": [ocr(raw_prob, y, t) for t in OCR_THRESHOLDS],
            "bins": bins_raw,
        },
        "platt_calibrated_oof": {
            "ece": ece_cal,
            "mce": mce_cal,
            "brier": brier(oof_platt, y),
            "brier_skill_score": brier_skill_score(oof_platt, y),
            "ocr": [ocr(oof_platt, y, t) for t in OCR_THRESHOLDS],
            "bins": bins_cal,
        },
        "ece_reduction_from_platt": float(ece_raw - ece_cal),
        "kalshibench_reference": KALSHIBENCH,
        "platt_calibrated_ece_within_kalshibench_range": bool(
            KALSHIBENCH["ece_range"][0] <= ece_cal <= KALSHIBENCH["ece_range"][1]
        ),
        "verdict": (
            "well_calibrated_after_platt" if ece_cal <= 0.10
            else "requires_post_hoc_platt_scaling" if ece_raw - ece_cal > 0.02
            else "miscalibrated"
        ),
        "f8_caveat": (
            "If platt_calibrated_oof.ocr at 0.9 shows positive OCR (overconfident "
            "high-DoM regime), F-8 selective-prediction deployment should be "
            "restricted to mid-coverage."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())