"""FE778 — PANL-equivalent probe on cached Qwen 1.5B MATH-500 1024-tok residuals.

Refutation #1 test. The PANL paper reports a post-answer-newline (PANL) hidden-state
probe at AUROC 0.986 (Gemma TriviaQA verification) and 0.774 (Qwen 7B A2-correctness).
Here we replicate the PANL recipe on the same-architecture-family Qwen 1.5B: locate the
post-answer-newline token in each K=1 MATH-500 generation, extract its L19 residual
(plus a full layer sweep), and train an OOF L2-logistic probe against K=1 correctness.

Decision rule: if the PANL L19 (or any swept layer) probe AUROC beats the prefill L19
DoM headline (0.7731), F-2's headline-position claim must be revised. We also print the
gap to the final-token L19 DoM (0.7186).

Regen input: a per-token / per-layer PANL residual cache extracted from the Stage-2
1024-tok run. It carries the post-answer-newline residual for each of the 500 problems
across the captured layer set. If that cache has not been materialized this script exits
cleanly with MISSING_REGEN_INPUT (return 2) — the documented prefill cache alone does not
contain the PANL token, so the probe cannot be faked from it.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
# PANL post-answer-newline per-layer residual cache (regen input).
PANL_NPZ = ROOT / "pathway11_h100/panl_probe/cache/m15b_panl_layers.npz"
# Baseline prefill cache for the DoM cross-check.
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/panl_probe/results.json"

SEED = 9999
N_FOLDS = 5
TARGET_LAYER = 19
C_GRID = [0.001, 0.01, 0.1, 1.0]
SWEEP_C = 0.01  # fixed regularization for the per-layer sweep

# Published comparison anchors (external / prior-internal, not recomputed here).
PREFILL_L19_DOM = 0.7731
FINAL_L19_DOM = 0.7186


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def oof_logistic_auroc(X: np.ndarray, y: np.ndarray, C: float) -> float:
    """Standardize-per-fold L2-logistic, return OOF AUROC over decision scores."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        clf = LogisticRegression(penalty="l2", C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        oof[te] = clf.decision_function(Xte)
    return auroc(oof, y)


def load_panl(blob) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (resid[N, L, D], layers[L], correct[N]) from a flexible PANL npz."""
    keys = set(blob.files)
    if "correct" not in keys:
        raise KeyError("PANL cache missing 'correct'")
    correct = blob["correct"].astype(bool)

    resid_key = next((k for k in ("resid", "panl_resid", "panl", "hidden") if k in keys), None)
    if resid_key is None:
        raise KeyError("PANL cache missing residual array")
    resid = blob[resid_key].astype(np.float64)

    if resid.ndim == 2:
        # Single-layer (assumed L19) cache: lift to a length-1 layer axis.
        resid = resid[:, None, :]
        layers = np.array([TARGET_LAYER], dtype=int)
    elif resid.ndim == 3:
        if "layers" in keys:
            layers = blob["layers"].astype(int)
        else:
            layers = np.arange(resid.shape[1], dtype=int)
    else:
        raise ValueError(f"unexpected PANL residual ndim={resid.ndim}")
    return resid, layers, correct


def main() -> int:
    if not PANL_NPZ.exists():
        print("MISSING_REGEN_INPUT", PANL_NPZ, file=sys.stderr)
        return 2

    blob = np.load(PANL_NPZ)
    try:
        resid, layers, correct = load_panl(blob)
    except (KeyError, ValueError) as exc:
        print("MISSING_REGEN_INPUT", PANL_NPZ, str(exc), file=sys.stderr)
        return 2

    n, n_layers, d = resid.shape
    if correct.shape[0] != n:
        print("MISSING_REGEN_INPUT", PANL_NPZ, "N mismatch", file=sys.stderr)
        return 2

    # Per-layer PANL sweep at fixed regularization.
    layer_sweep = {}
    for li, layer in enumerate(layers):
        layer_sweep[int(layer)] = oof_logistic_auroc(resid[:, li, :], correct, SWEEP_C)

    best_layer = max(layer_sweep, key=lambda k: layer_sweep[k])
    best_layer_auroc = layer_sweep[best_layer]

    # Full C sweep on the target L19 layer (or nearest available).
    if TARGET_LAYER in set(int(x) for x in layers):
        l19_idx = int(np.flatnonzero(layers == TARGET_LAYER)[0])
        l19_layer = TARGET_LAYER
    else:
        l19_idx = int(np.argmin(np.abs(layers - TARGET_LAYER)))
        l19_layer = int(layers[l19_idx])
    panl_l19_by_C = {
        str(C): oof_logistic_auroc(resid[:, l19_idx, :], correct, C) for C in C_GRID
    }
    panl_l19_best = max(panl_l19_by_C.values())

    # Cross-check: recompute the prefill L19 DoM AUROC from the documented caches.
    prefill_dom_recomputed = float("nan")
    if CACHE.exists() and DOM_NPZ.exists():
        pre_correct = np.load(CACHE)["correct"].astype(bool)
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom_score.shape[0] == pre_correct.shape[0]:
            prefill_dom_recomputed = auroc(dom_score, pre_correct)

    headline = max(panl_l19_best, best_layer_auroc)
    beats_prefill = bool(headline > PREFILL_L19_DOM)
    beats_final = bool(headline > FINAL_L19_DOM)

    out = {
        "experiment": "FE778",
        "n": int(n),
        "n_layers": int(n_layers),
        "feature_dim": int(d),
        "panl_l19_layer_used": int(l19_layer),
        "panl_l19_auroc_by_C": panl_l19_by_C,
        "panl_l19_auroc_best": float(panl_l19_best),
        "layer_sweep_auroc": {str(k): float(v) for k, v in layer_sweep.items()},
        "best_layer": int(best_layer),
        "best_layer_auroc": float(best_layer_auroc),
        "panl_headline_auroc": float(headline),
        "baseline_prefill_l19_dom_auroc": PREFILL_L19_DOM,
        "baseline_final_l19_dom_auroc": FINAL_L19_DOM,
        "prefill_dom_auroc_recomputed": prefill_dom_recomputed,
        "gap_vs_prefill_dom": float(headline - PREFILL_L19_DOM),
        "gap_vs_final_dom": float(headline - FINAL_L19_DOM),
        "beats_prefill_dom": beats_prefill,
        "beats_final_dom": beats_final,
        "refutation_1_supported": beats_prefill,
        "verdict": (
            "REFUTES_F2_HEADLINE_POSITION" if beats_prefill
            else "F2_HEADLINE_POSITION_HOLDS"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())