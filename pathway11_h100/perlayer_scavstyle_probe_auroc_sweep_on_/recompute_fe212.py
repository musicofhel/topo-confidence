"""P11-FE212 — Per-layer SCAV-style probe AUROC sweep on Qwen-1.5B prefill activations.

Trains an L2-regularized logistic probe sigmoid(w·e + b) at every transformer
layer L0..L27 on the prefill (mean-pooled / last-token) hidden states, scoring
correctness via 5-fold out-of-fold (OOF) AUROC. Reports per-layer OOF AUROC, the
peak layer, and the L19 value, comparing the peak against the canonical
L19 DoM AUROC = 0.7731 (F-2's layer-anchor assumption).

Mirrors SCAV Fig 1 (per-layer concept-vector probe accuracy) but targets
correctness instead of safety. SCAV reports >95% TestAcc from layer ~10-11 on
safety with ~140+140 samples; our correctness probe at L19 saturates near 0.77
with ~500 samples — this sweep tests whether L19 is actually the optimal layer.

Requires an all-layer prefill cache (500, n_layers, 1536). If only the single
L19 cache is present, that layer alone is still reported; absent all inputs the
script prints MISSING_REGEN_INPUT and returns 2.
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
# Preferred: an all-layer prefill cache. Fall back to the single-L19 cache.
ALL_LAYER_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz"
L19_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/scav_layer_sweep/results.json"

SEED = 9999
N_FOLDS = 5
L19_INDEX = 19
L19_REFERENCE_AUROC = 0.7731  # canonical OOF DoM AUROC at L19 (1024-tok labels)
PROBE_C = 1.0  # inverse L2 strength for the sigmoid probe


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def probe_oof_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold OOF AUROC for a standardized L2-regularized logistic probe."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        clf = LogisticRegression(
            penalty="l2", C=PROBE_C, solver="lbfgs", max_iter=2000
        )
        clf.fit(Xtr_s, y[train_idx])
        oof[test_idx] = clf.predict_proba(Xte_s)[:, 1]
    return auroc(oof, y)


def load_activations():
    """Return (acts, y) where acts is (500, n_layers, 1536) and y is (500,) bool.

    Prefers the all-layer cache; falls back to wrapping the single L19 cache as a
    one-layer stack (placed at index L19 for reporting consistency).
    """
    if ALL_LAYER_CACHE.exists():
        blob = np.load(ALL_LAYER_CACHE)
        # Find a 3D (samples, layers, hidden) array among the keys.
        acts = None
        for key in ("prefill_all", "prefill", "hidden_all", "activations"):
            if key in blob.files and blob[key].ndim == 3:
                acts = blob[key].astype(np.float64)
                break
        if acts is None:
            for key in blob.files:
                if blob[key].ndim == 3 and blob[key].shape[0] == 500:
                    acts = blob[key].astype(np.float64)
                    break
        if acts is None:
            return None, None
        y = blob["correct"].astype(bool)
        return acts, y
    if L19_CACHE.exists():
        blob = np.load(L19_CACHE)
        X = blob["prefill"].astype(np.float64)  # (500, 1536)
        y = blob["correct"].astype(bool)
        # Single available layer; report it at the L19 slot.
        acts = X[:, None, :]
        return acts, y, {"single_layer_only": True}
    return None, None


def main() -> int:
    if not ALL_LAYER_CACHE.exists() and not L19_CACHE.exists():
        print("MISSING_REGEN_INPUT", ALL_LAYER_CACHE, L19_CACHE, file=sys.stderr)
        return 2

    loaded = load_activations()
    single_layer_only = False
    if loaded is None or loaded[0] is None:
        print("MISSING_REGEN_INPUT", ALL_LAYER_CACHE, file=sys.stderr)
        return 2
    if len(loaded) == 3:
        acts, y, meta = loaded
        single_layer_only = bool(meta.get("single_layer_only", False))
    else:
        acts, y = loaded

    n_samples, n_layers, hidden = acts.shape
    assert n_samples == 500 and hidden == 1536, f"unexpected shape {acts.shape}"

    per_layer = []
    if single_layer_only:
        # Only L19 is available — report it at its canonical index.
        a = probe_oof_auroc(acts[:, 0, :], y)
        per_layer = [
            {"layer": L19_INDEX, "auroc_oof": a}
        ]
        layer_indices = [L19_INDEX]
        aurocs = [a]
    else:
        layer_indices = list(range(n_layers))
        aurocs = []
        for L in layer_indices:
            a = probe_oof_auroc(acts[:, L, :], y)
            per_layer.append({"layer": L, "auroc_oof": a})
            aurocs.append(a)

    valid = [(L, a) for L, a in zip(layer_indices, aurocs) if not np.isnan(a)]
    if valid:
        peak_layer, peak_auroc = max(valid, key=lambda t: t[1])
    else:
        peak_layer, peak_auroc = None, float("nan")

    l19_auroc = next(
        (a for L, a in zip(layer_indices, aurocs) if L == L19_INDEX), float("nan")
    )

    out = {
        "experiment": "P11-FE212",
        "description": "Per-layer SCAV-style L2 logistic probe AUROC sweep for correctness",
        "n_samples": int(n_samples),
        "n_layers_scanned": len(layer_indices),
        "probe_C": PROBE_C,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "single_layer_only": single_layer_only,
        "per_layer_auroc": per_layer,
        "peak_layer": peak_layer,
        "peak_auroc_oof": float(peak_auroc),
        "l19_auroc_oof": float(l19_auroc),
        "l19_reference_dom_auroc": L19_REFERENCE_AUROC,
        "peak_minus_l19": (
            float(peak_auroc - l19_auroc) if not np.isnan(peak_auroc) and not np.isnan(l19_auroc) else None
        ),
        "peak_beats_l19_reference": (
            bool(peak_auroc > L19_REFERENCE_AUROC) if not np.isnan(peak_auroc) else None
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())