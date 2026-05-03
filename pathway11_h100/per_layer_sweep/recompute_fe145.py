"""FE145 — Per-layer prefill DoM AUROC sweep on Qwen-2.5-1.5B MATH-500.

For each layer L in 0..28 (29 entries: input embeds + 28 transformer
blocks), extract the prefill activation at position 0 from every per-
problem NPZ, run 5-fold OOF DoM, report AUROC. Verify that L19 is the
argmax within ±0.005 of any other layer.

CAA (Panickssery et al.) reports optimal steering at ~⅓ depth with a
sharp drop-off at L17 (53% depth). F-2 picks L19/28 (~68%) — well past
CAA's drop-off. This sweep settles whether L19 is a real peak or a
post-hoc lucky pick.

Inputs: pathway8_layerwise/data/math500/problem_{i:03d}.npz × 500
Output: pathway11_h100/per_layer_sweep/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
DATA_DIR = ROOT / "pathway8_layerwise/data/math500"
OUT_JSON = ROOT / "pathway11_h100/per_layer_sweep/results.json"

N_PROBLEMS = 500
HIDDEN_DIM = 1536  # Qwen-2.5-1.5B
N_LAYERS = 29
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


def stratified_kfold_indices(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Stratified k-fold indices on a binary y, deterministic given seed."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def load_per_layer_prefill() -> tuple[np.ndarray, np.ndarray]:
    """Returns prefill (N_LAYERS, N_PROBLEMS, HIDDEN_DIM) and correct (N_PROBLEMS,)."""
    X = np.zeros((N_LAYERS, N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    y = np.zeros(N_PROBLEMS, dtype=bool)
    for i in range(N_PROBLEMS):
        fp = DATA_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing per-problem NPZ: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, T, 1536)
            X[:, i, :] = s[:, 0, :].astype(np.float32)
            y[i] = bool(d["correct"])
    return X, y


def oof_dom_auroc(features: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """5-fold OOF DoM AUROC. features (n, d), y (n,) bool."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_pos = train_mask & y
        train_neg = train_mask & ~y
        if not train_pos.any() or not train_neg.any():
            continue
        dom = features[train_pos].mean(axis=0) - features[train_neg].mean(axis=0)
        scores[test_idx] = features[test_idx] @ dom
    return auroc(scores, y)


def main() -> int:
    if not DATA_DIR.exists():
        print("MISSING_REGEN_INPUT", DATA_DIR, file=sys.stderr)
        return 2

    X, y = load_per_layer_prefill()
    folds = stratified_kfold_indices(y, N_FOLDS, SEED)

    aurocs = {}
    for layer in range(N_LAYERS):
        a = oof_dom_auroc(X[layer], y, folds)
        aurocs[layer] = a

    sorted_layers = sorted(aurocs.items(), key=lambda kv: kv[1], reverse=True)
    peak_layer, peak_auroc = sorted_layers[0]
    second_layer, second_auroc = sorted_layers[1]

    # Within ±0.005 of L19?
    auroc_l19 = aurocs[19]
    within_band = peak_auroc - auroc_l19
    is_l19_peak = peak_layer == 19
    is_l19_within_band = within_band <= 0.005

    # CAA drop-off: L17 vs L19 vs argmax
    caa_dropoff_layer = 17
    auroc_caa = aurocs[caa_dropoff_layer]

    out = {
        "n": N_PROBLEMS,
        "n_layers": N_LAYERS,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_per_layer": {str(k): float(v) for k, v in aurocs.items()},
        "peak_layer": int(peak_layer),
        "peak_auroc": float(peak_auroc),
        "second_layer": int(second_layer),
        "second_auroc": float(second_auroc),
        "auroc_l19": float(auroc_l19),
        "auroc_l17": float(auroc_caa),
        "is_l19_peak": is_l19_peak,
        "is_l19_within_5e-3_of_peak": is_l19_within_band,
        "peak_minus_l19": float(within_band),
        "f2_layer_validation_passed": is_l19_peak or is_l19_within_band,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"peak_layer={peak_layer}")
    print(f"peak_auroc={peak_auroc:.10f}")
    print(f"auroc_l19={auroc_l19:.10f}")
    print(f"auroc_l17={auroc_caa:.10f}")
    print(f"peak_minus_l19={within_band:.10f}")
    print(f"is_l19_peak={int(is_l19_peak)}")
    print(f"is_l19_within_5em3_of_peak={int(is_l19_within_band)}")
    for L in range(N_LAYERS):
        print(f"auroc_l{L:02d}={aurocs[L]:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
