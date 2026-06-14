"""P11-FE634 — Layer sensitivity sweep for the L19 DoM probe (F-2).

RISER picks L20/L25/L40 as optimal Qwen2.5 injection layers, with L19-L21 the
lower edge. F-2 anchors the prefill DoM probe at L19 (OOF AUROC 0.7731) on the
assumption that L19 is THE predictive layer. This recomputes the difference-of-
means (DoM) direction and its 5-fold out-of-fold AUROC at deeper layers
(L20, L22, L25) and at L13 (Llama-comparison analog) on cached all-layer prefill
activations, to test whether L19 is locally optimal or merely where we first
probed. L19 from the canonical single-layer cache is recomputed as the reference.

DoM per fold: d = mean(X_train[correct]) - mean(X_train[~correct]); score test
fold as X_test @ d; aggregate OOF scores; AUROC over all 500.
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

# Canonical single-layer L19 cache holds `correct` labels + the L19 reference.
# The all-layer prefill cache is needed for the deeper-layer probes; probe a few
# plausible filenames so the script is robust to the cache's exact name.
ALLLAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_perlayer.npz",
    ROOT / "pathway11_h100/data/m15b_prefill_alllayers.npz",
]

OUT_JSON = ROOT / "pathway11_h100/layer_sensitivity_sweep/results.json"

REFERENCE_LAYER = 19
TARGET_LAYERS = [13, 20, 22, 25]  # L13 = Llama-comparison analog; L20/L22/L25 = RISER deep edge
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


def dom_oof_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """5-fold OOF AUROC for the difference-of-means probe at one layer."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d
    return auroc(oof, y)


def find_alllayer_cache() -> Path | None:
    for p in ALLLAYER_CANDIDATES:
        if p.exists():
            return p
    return None


def extract_layer(blob, layer: int) -> np.ndarray | None:
    """Pull a (500, 1536) slice for `layer` from an all-layer npz.

    Supports a 3D array keyed by (samples, layers, hidden) or per-layer keys
    such as 'L20' / 'layer_20' / 'prefill_20'.
    """
    # Per-layer key patterns first.
    for key in (f"L{layer}", f"layer_{layer}", f"prefill_{layer}", f"l{layer}"):
        if key in blob.files:
            arr = blob[key]
            if arr.ndim == 2 and arr.shape == (500, 1536):
                return arr.astype(np.float64)
    # 3D stacked array.
    for key in blob.files:
        arr = blob[key]
        if arr.ndim == 3 and arr.shape[0] == 500 and arr.shape[-1] == 1536:
            if 0 <= layer < arr.shape[1]:
                return arr[:, layer, :].astype(np.float64)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    cache = np.load(CACHE)
    y = cache["correct"].astype(bool)
    assert y.shape == (500,)
    X19 = cache["prefill"].astype(np.float64)
    assert X19.shape == (500, 1536)

    folds = stratified_kfold(y, N_FOLDS, SEED)

    per_layer = {}
    # L19 reference from the canonical single-layer cache.
    per_layer[REFERENCE_LAYER] = float(dom_oof_auroc(X19, y, folds))

    alllayer_path = find_alllayer_cache()
    missing_layers = []
    if alllayer_path is None:
        # Deeper-layer probes require the all-layer prefill cache, which is not
        # present on this machine. The L19 reference still recomputes; the sweep
        # is reported as unavailable so the run does not silently claim a null.
        missing_layers = list(TARGET_LAYERS)
    else:
        blob = np.load(alllayer_path)
        # Sanity-check the all-layer cache's labels match if it carries them.
        if "correct" in blob.files:
            y_alt = blob["correct"].astype(bool)
            if y_alt.shape == y.shape and not np.array_equal(y_alt, y):
                print("LABEL_MISMATCH between caches", file=sys.stderr); return 3
        for layer in TARGET_LAYERS:
            Xl = extract_layer(blob, layer)
            if Xl is None:
                missing_layers.append(layer)
                continue
            per_layer[layer] = float(dom_oof_auroc(Xl, y, folds))

    ref_auroc = per_layer[REFERENCE_LAYER]
    swept = {L: a for L, a in per_layer.items() if L != REFERENCE_LAYER}
    best_layer = max(per_layer, key=per_layer.get)

    out = {
        "experiment": "P11-FE634",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_samples": int(len(y)),
        "reference_layer": REFERENCE_LAYER,
        "reference_auroc_oof": ref_auroc,
        "alllayer_cache": str(alllayer_path) if alllayer_path else None,
        "per_layer_auroc_oof": {str(L): per_layer[L] for L in sorted(per_layer)},
        "missing_layers": sorted(missing_layers),
        "best_layer": int(best_layer),
        "best_auroc_oof": float(per_layer[best_layer]),
        "deeper_layer_beats_l19": bool(
            swept and max(swept.values()) > ref_auroc
        ),
        "l19_locally_optimal": bool(
            best_layer == REFERENCE_LAYER if swept else True
        ),
        "note": (
            "Deeper-layer sweep unavailable: all-layer prefill cache not found; "
            "only the L19 reference was recomputed."
            if not swept
            else "DoM 5-fold OOF AUROC swept across cached prefill layers."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())