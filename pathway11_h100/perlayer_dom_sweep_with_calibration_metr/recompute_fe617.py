"""P11-FE617 — Per-layer DoM sweep with calibration metrics.

Refit centroid-difference (Difference-of-Means) probes on cached 1.5B prefill
activations at every available layer L0..L27 from the Pathway-8 layer-wise
cache. For each layer compute OOF (5-fold) AUROC, plus ECE / Brier-skill-score
under (a) an unscaled probability mapping (sigmoid of z-scored OOF score) and
(b) a Platt-scaled mapping (logistic fit on the OOF scores).

Motivation: Gros & Devanbu (2512.24560) Table 6 + Section 6.1 find the LAST
layer worst, MIDDLE best for ECE. F-2 selected L19 on AUROC alone. We identify
the minimum-ECE layer whose AUROC matches L19's (within tolerance); if such a
layer exists, F-2 is calibration-suboptimal and the headline number earns a
calibration companion.
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
MAIN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/perlayer_dom_calibration/results.json"

# Candidate single-file Pathway-8 per-layer activation caches. The first that
# exists and yields >=2 distinct layers wins.
SINGLE_NPZ_CANDIDATES = [
    ROOT / "pathway8_layerwise/cache/perlayer_prefill.npz",
    ROOT / "pathway8_layerwise/cache/layerwise_prefill.npz",
    ROOT / "pathway8_layerwise/cache/layerwise_activations.npz",
    ROOT / "pathway8_layerwise/cache/all_layers.npz",
    ROOT / "pathway8/cache/perlayer_prefill.npz",
    ROOT / "pathway8/cache/layerwise_prefill.npz",
    ROOT / "pathway8/cache/all_layers.npz",
    ROOT / "pathway11_h100/perlayer/cache/all_layers.npz",
]
# Candidate directories holding one NPZ per layer (layer_NN.npz / L19.npz etc).
PER_LAYER_DIR_CANDIDATES = [
    ROOT / "pathway8_layerwise/cache",
    ROOT / "pathway8/cache",
    ROOT / "pathway11_h100/perlayer/cache",
]

N_LAYERS = 28
N_FOLDS = 5
SEED = 9999
L19 = 19
AUROC_MATCH_TOL = 0.01  # "matched AUROC" tolerance vs L19
N_BINS = 15


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    labels = labels.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        total += (m / n) * abs(conf - acc)
    return float(total)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


def bss(probs: np.ndarray, labels: np.ndarray) -> float:
    y = labels.astype(np.float64)
    base = float(y.mean())
    ref = float(np.mean((base - y) ** 2))
    if ref < 1e-12:
        return float("nan")
    return float(1.0 - brier(probs, y) / ref)


def platt_probs(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Logistic (Platt) fit of P(correct) on a 1-D score vector."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    clf.fit(scores.reshape(-1, 1), labels.astype(int))
    return clf.predict_proba(scores.reshape(-1, 1))[:, 1]


def oof_dom_scores(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """Out-of-fold centroid-difference (DoM) projection scores."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = Xte @ d_vec
    return scores


def layer_idx_from_key(key: str) -> int | None:
    kl = key.lower()
    if "layer" not in kl and not (kl.startswith("l") and any(c.isdigit() for c in kl)):
        return None
    digits = "".join(c for c in key if c.isdigit())
    if digits == "":
        return None
    return int(digits)


def _layers_from_blob(blob) -> dict[int, np.ndarray]:
    """Extract {layer_idx: (500,1536)} from an opened NPZ blob."""
    layers: dict[int, np.ndarray] = {}
    # (a) a single 3-D array stacking all layers.
    for key in blob.files:
        a = blob[key]
        if getattr(a, "ndim", 0) == 3 and N_LAYERS in a.shape:
            lax = a.shape.index(N_LAYERS)
            a = np.moveaxis(a, lax, 0)  # -> (n_layers, ?, ?)
            if a.shape[1:] == (500, 1536):
                for li in range(a.shape[0]):
                    layers[li] = a[li].astype(np.float64)
                return layers
            if a.shape[1:] == (1536, 500):
                for li in range(a.shape[0]):
                    layers[li] = a[li].T.astype(np.float64)
                return layers
    # (b) one 2-D array per layer, keyed by name.
    for key in blob.files:
        a = blob[key]
        if getattr(a, "ndim", 0) != 2:
            continue
        li = layer_idx_from_key(key)
        if li is None:
            continue
        if a.shape == (500, 1536):
            layers[li] = a.astype(np.float64)
        elif a.shape == (1536, 500):
            layers[li] = a.T.astype(np.float64)
    return layers


def load_layers() -> dict[int, np.ndarray]:
    for path in SINGLE_NPZ_CANDIDATES:
        if path.exists():
            with np.load(path) as blob:
                layers = _layers_from_blob(blob)
            if len(layers) >= 2:
                return layers
    for d in PER_LAYER_DIR_CANDIDATES:
        if not d.is_dir():
            continue
        layers: dict[int, np.ndarray] = {}
        for fp in sorted(d.glob("*.npz")):
            li = layer_idx_from_key(fp.stem)
            if li is None:
                continue
            with np.load(fp) as b:
                arr = None
                for key in b.files:
                    a = b[key]
                    if getattr(a, "ndim", 0) == 2 and a.shape in ((500, 1536), (1536, 500)):
                        arr = a if a.shape == (500, 1536) else a.T
                        break
                if arr is not None:
                    layers[li] = arr.astype(np.float64)
        if len(layers) >= 2:
            return layers
    return {}


def main() -> int:
    if not MAIN_CACHE.exists():
        print("MISSING_REGEN_INPUT", MAIN_CACHE, file=sys.stderr)
        return 2

    main_blob = np.load(MAIN_CACHE)
    y = main_blob["correct"].astype(bool)
    assert y.shape == (500,)

    layers = load_layers()
    # Fall back to the canonical L19 prefill if it wasn't found in the sweep cache.
    if L19 not in layers:
        layers[L19] = main_blob["prefill"].astype(np.float64)

    if len(layers) < 2:
        print("MISSING_REGEN_INPUT", "no Pathway-8 per-layer activation cache found",
              file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)

    per_layer = {}
    for li in sorted(layers):
        X = layers[li]
        if X.shape != (500, 1536):
            continue
        s = oof_dom_scores(X, y, folds)
        au = auroc(s, y)

        # Unscaled probabilities: sigmoid of z-scored OOF scores (no label fit).
        sd = s.std()
        z = (s - s.mean()) / sd if sd > 1e-12 else np.zeros_like(s)
        p_unscaled = sigmoid(z)

        # Platt-scaled probabilities: logistic fit on the OOF scores.
        try:
            p_platt = platt_probs(s, y)
        except Exception as exc:  # pragma: no cover - defensive
            print("PLATT_FAILED", li, exc, file=sys.stderr)
            p_platt = p_unscaled

        per_layer[li] = {
            "layer": int(li),
            "auroc_oof": float(au),
            "ece_unscaled": float(ece(p_unscaled, y)),
            "bss_unscaled": float(bss(p_unscaled, y)),
            "brier_unscaled": float(brier(p_unscaled, y)),
            "ece_platt": float(ece(p_platt, y)),
            "bss_platt": float(bss(p_platt, y)),
            "brier_platt": float(brier(p_platt, y)),
        }

    if L19 not in per_layer:
        print("MISSING_REGEN_INPUT", "L19 layer unavailable", file=sys.stderr)
        return 2

    l19_auroc = per_layer[L19]["auroc_oof"]

    # Layers whose AUROC matches L19 within tolerance (>= L19 - tol).
    matched = [
        li for li in per_layer
        if per_layer[li]["auroc_oof"] >= l19_auroc - AUROC_MATCH_TOL
    ]
    min_ece_matched = None
    if matched:
        min_ece_matched = min(matched, key=lambda li: per_layer[li]["ece_platt"])
    min_ece_overall = min(per_layer, key=lambda li: per_layer[li]["ece_platt"])
    best_auroc_layer = max(per_layer, key=lambda li: per_layer[li]["auroc_oof"])

    f2_calibration_suboptimal = bool(
        min_ece_matched is not None
        and min_ece_matched != L19
        and per_layer[min_ece_matched]["ece_platt"] < per_layer[L19]["ece_platt"]
    )

    out = {
        "experiment": "P11-FE617",
        "description": "Per-layer DoM sweep with calibration metrics (AUROC/ECE/BSS, unscaled + Platt).",
        "n_layers_found": len(per_layer),
        "layers_found": sorted(per_layer),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_bins_ece": N_BINS,
        "auroc_match_tol": AUROC_MATCH_TOL,
        "l19_auroc_oof": float(l19_auroc),
        "l19_ece_platt": float(per_layer[L19]["ece_platt"]),
        "l19_ece_unscaled": float(per_layer[L19]["ece_unscaled"]),
        "best_auroc_layer": int(best_auroc_layer),
        "best_auroc_value": float(per_layer[best_auroc_layer]["auroc_oof"]),
        "min_ece_matched_layer": (int(min_ece_matched) if min_ece_matched is not None else None),
        "min_ece_matched_ece_platt": (
            float(per_layer[min_ece_matched]["ece_platt"]) if min_ece_matched is not None else None
        ),
        "min_ece_matched_auroc": (
            float(per_layer[min_ece_matched]["auroc_oof"]) if min_ece_matched is not None else None
        ),
        "min_ece_overall_layer": int(min_ece_overall),
        "f2_calibration_suboptimal": f2_calibration_suboptimal,
        "per_layer": [per_layer[li] for li in sorted(per_layer)],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())