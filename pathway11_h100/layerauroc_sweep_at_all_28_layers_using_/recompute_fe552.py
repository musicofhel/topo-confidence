"""FE552 — Layer-AUROC sweep across all 28 layers (refutation #4 of L19).

For every transformer layer's cached prefill hidden state we fit an out-of-fold
DiffInMeans (DoM) direction (μ_correct − μ_incorrect on the train folds), project
the held-out fold onto it, and aggregate the OOF AUROC. Projecting onto a 1-D DoM
score and then fitting a 1-D logistic probe leaves AUROC unchanged (logistic is a
monotone transform), so the OOF DoM-projection AUROC IS the probe AUROC.

The output is the full AUROC-over-layers curve plus L19's rank, peak layer, and
the plateau width (layers within 0.01 AUROC of the peak). This is the smallest
test of whether 'L19' (F-2) is a sharp peak or a flat plateau — SteeringSafety
finds best layers drift by model/task, so we publish the curve, not a point.

Requires a per-layer prefill cache (n, n_layers, 1536). The single-layer L19
cache (m15b_prefill.npz) only carries one layer and is insufficient on its own;
if no all-layer cache is present we emit MISSING_REGEN_INPUT and stop.
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
OUT_JSON = ROOT / "pathway11_h100/layer_auroc_sweep/results.json"

# Candidate per-layer caches: (path, key) pairs. The array under <key> must be
# 3-D, shaped (n_samples, n_layers, hidden_dim). First match wins.
LAYER_CACHE_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz", "prefill_layers"),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz", "prefill"),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz", "prefill_layers"),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz", "prefill"),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_per_layer.npz", "prefill"),
    (ROOT / "pathway11_h100/data/m15b_prefill_alllayers.npz", "prefill_layers"),
]

L19_INDEX = 19          # nominal index of the published F-2 layer
PLATEAU_TOL = 0.01      # layers within this AUROC of the peak count as plateau
N_FOLDS = 5
SEED = 9999


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


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """OOF DiffInMeans-projection AUROC for a single layer's features."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof[test_idx] = Xte @ d_vec
    return auroc(oof, y)


def load_layer_cache():
    """Return (X_all, y) where X_all is (n, n_layers, dim), or None if missing."""
    for path, key in LAYER_CACHE_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        if key not in blob.files:
            continue
        arr = blob[key]
        if arr.ndim != 3:
            continue
        X_all = arr.astype(np.float64)
        if "correct" in blob.files:
            y = blob["correct"].astype(bool)
        elif CACHE.exists():
            y = np.load(CACHE)["correct"].astype(bool)
        else:
            continue
        if X_all.shape[0] != y.shape[0]:
            continue
        return X_all, y, str(path), key
    return None


def main() -> int:
    loaded = load_layer_cache()
    if loaded is None:
        print("MISSING_REGEN_INPUT", "no per-layer prefill cache "
              "(n, n_layers, 1536) found among:", file=sys.stderr)
        for path, key in LAYER_CACHE_CANDIDATES:
            print("  ", path, "::", key, file=sys.stderr)
        return 2

    X_all, y, cache_path, cache_key = loaded
    n, n_layers, dim = X_all.shape
    folds = stratified_kfold(y, N_FOLDS, SEED)

    per_layer = []
    for layer in range(n_layers):
        a = oof_dom_auroc(X_all[:, layer, :], y, folds)
        per_layer.append({"layer": layer, "auroc_oof_dom": a})

    aurocs = np.array([d["auroc_oof_dom"] for d in per_layer], dtype=np.float64)
    valid = ~np.isnan(aurocs)
    peak_layer = int(np.nanargmax(aurocs))
    peak_auroc = float(aurocs[peak_layer])
    plateau_layers = [int(i) for i in np.flatnonzero(valid & (aurocs >= peak_auroc - PLATEAU_TOL))]

    # Rank of L19 (1 = best). Guard against caches that omit L19.
    l19_auroc = float(aurocs[L19_INDEX]) if L19_INDEX < n_layers else float("nan")
    if np.isnan(l19_auroc):
        l19_rank = None
    else:
        l19_rank = int((aurocs[valid] > l19_auroc).sum()) + 1

    out = {
        "experiment": "P11-FE552",
        "description": "Layer-AUROC sweep, OOF DiffInMeans probe, all layers",
        "cache_path": cache_path,
        "cache_key": cache_key,
        "n_samples": int(n),
        "n_layers": int(n_layers),
        "hidden_dim": int(dim),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "plateau_tol": PLATEAU_TOL,
        "per_layer_auroc": per_layer,
        "peak_layer": peak_layer,
        "peak_auroc": peak_auroc,
        "plateau_layers": plateau_layers,
        "plateau_width": len(plateau_layers),
        "l19_index": L19_INDEX,
        "l19_auroc": l19_auroc,
        "l19_rank": l19_rank,
        "l19_is_peak": bool(L19_INDEX == peak_layer),
        "verdict": ("L19_SHARP_PEAK" if len(plateau_layers) <= 2 and L19_INDEX in plateau_layers
                    else "L19_ON_PLATEAU" if L19_INDEX in plateau_layers
                    else "L19_NOT_NEAR_PEAK"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())