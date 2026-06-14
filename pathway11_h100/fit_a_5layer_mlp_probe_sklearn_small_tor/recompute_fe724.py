"""FE724 — 5-layer MLP probe vs linear baseline on prefill L19 activations.

Tests whether F-2's AUROC ceiling (linear DoM L19 = 0.7731) is information-bound
or probe-architecture-bound. Fits a 5-layer (4-hidden) MLP probe on cached
mean-pooled prefill activations and compares 5-fold OOF AUROC against a linear
logistic-regression baseline and the raw DoM direction. Repeats at L18/L20/L21
for layer-sensitivity *if* per-layer caches exist (missing layers are skipped,
not fatal — only L19 is required since that is the canonical cache).

If MLP OOF AUROC > 0.82, F-2's 'DoM is the signal' downgrades to 'L19 contains
the signal' and H-13 (head-level attribution) gains priority.
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/mlp_probe/results.json"

LINEAR_BASELINE = 0.7731  # canonical L19 linear DoM OOF AUROC (F-2)
MLP_HIDDEN = (512, 256, 128, 64)  # 4 hidden + output = 5 weight layers
N_FOLDS = 5
SEED = 9999

# Candidate (path, npz-key) locations for each layer's prefill activations.
# L19 is the canonical cache; other layers are best-effort and skipped if absent.
LAYER_SOURCES = {
    18: [
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_L18.npz", "prefill"),
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_l18.npz", "prefill"),
    ],
    19: [
        (CACHE, "prefill"),
    ],
    20: [
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_L20.npz", "prefill"),
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_l20.npz", "prefill"),
    ],
    21: [
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_L21.npz", "prefill"),
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_l21.npz", "prefill"),
    ],
}


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def load_layer(layer):
    """Return (X, key_path) for a layer, or (None, None) if no cache is present."""
    for path, key in LAYER_SOURCES[layer]:
        if path.exists():
            blob = np.load(path)
            if key in blob:
                X = blob[key].astype(np.float64)
                if X.ndim == 2 and X.shape[0] == 500:
                    return X, str(path)
    return None, None


def oof_mlp(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=SEED,
        )
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.predict_proba(Xte)[:, 1]
    return oof


def oof_linear(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.decision_function(Xte)
    return oof


def oof_dom(X, y):
    """OOF raw difference-of-means projection (the canonical DoM probe)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        ytr = y[train_idx]
        d_vec = X[train_idx][ytr].mean(axis=0) - X[train_idx][~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d_vec
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    base = np.load(CACHE)
    y = base["correct"].astype(bool)
    assert y.shape == (500,)

    per_layer = {}
    for layer in sorted(LAYER_SOURCES):
        X, src = load_layer(layer)
        if X is None:
            per_layer[str(layer)] = {"status": "MISSING_CACHE", "source": None}
            continue
        if X.shape[0] != len(y):
            per_layer[str(layer)] = {"status": "SHAPE_MISMATCH", "source": src}
            continue
        mlp = oof_mlp(X, y)
        lin = oof_linear(X, y)
        dom = oof_dom(X, y)
        per_layer[str(layer)] = {
            "status": "OK",
            "source": src,
            "n": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "auroc_mlp_oof": auroc(mlp, y),
            "auroc_linear_oof": auroc(lin, y),
            "auroc_dom_oof": auroc(dom, y),
        }

    l19 = per_layer.get("19", {})
    mlp19 = l19.get("auroc_mlp_oof", float("nan"))
    out = {
        "experiment": "P11-FE724",
        "description": "5-layer MLP probe vs linear/DoM baseline on prefill activations",
        "mlp_hidden_layer_sizes": list(MLP_HIDDEN),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "linear_baseline_reference": LINEAR_BASELINE,
        "per_layer": per_layer,
        "l19_mlp_auroc": mlp19,
        "mlp_beats_082": bool(mlp19 == mlp19 and mlp19 > 0.82),
        "mlp_vs_linear_l19_delta": (
            float(mlp19 - l19["auroc_linear_oof"])
            if l19.get("status") == "OK"
            else None
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())