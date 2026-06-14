"""P11-FE288 — Mid-layer prefill DoM AUROC sweep on Qwen2.5-1.5B.

Trains a mass-mean (difference-of-means, DoM) probe at every transformer layer
L4..L27 on the cached pathway11_h100 prefill hidden states and reports OOF
5-fold AUROC vs layer. This is a refutation test for F-2's L19-specificity
claim, motivated by Zhang/Chosen/Andreas 2409.02228, which finds probe accuracy
is roughly uniform across mid-layers in Llama-2-7B / GPT-J-6B / GPT-2-124M.

Interpretation:
  * If the AUROC-vs-layer curve is flat (L19 not separated from the mid-layer
    band), F-2's emphasis on L19 is overfit to a single layer choice and the
    finding generalizes to "any mid-layer".
  * If L19 stands out above the mid-layer band, F-2 strengthens and we have
    evidence of Qwen-specific layer specialization.

Input: a per-layer prefill cache. The headline m15b_prefill.npz only stores the
L19 slice, so this sweep requires an all-layers extraction. The script probes a
few plausible cache layouts and degrades gracefully (MISSING_REGEN_INPUT) if no
multi-layer cache is present, so it can be wired up once the extraction lands.
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
# Candidate all-layers caches (any one is sufficient).
ALLLAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz",
]
OUT_JSON = ROOT / "pathway11_h100/layer_sweep_dom/results.json"

LAYERS = list(range(4, 28))  # L4..L27 inclusive
L19 = 19
N_FOLDS = 5
SEED = 9999
N_PROBLEMS = 500
HIDDEN = 1536


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
    """Out-of-fold mass-mean DoM AUROC for one layer's (500, d) activations."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d_vec
    return auroc(scores, y)


def load_layer_stack() -> tuple[np.ndarray, dict[int, int]] | None:
    """Return (stack (500, n_layers, 1536), {layer_index: stack_col}) or None.

    Supports three layouts:
      (a) a 3D array under one of several keys, with an optional "layers" index;
      (b) per-layer 2D arrays keyed prefill_L<l> / L<l> / layer_<l>;
      (c) a 3D array whose layer axis covers all 28 layers (axis indexed by L).
    """
    cache_path = next((p for p in ALLLAYER_CANDIDATES if p.exists()), None)
    if cache_path is None:
        return None
    blob = np.load(cache_path)
    keys = set(blob.files)

    # (b) per-layer 2D arrays
    per_layer = {}
    for l in range(0, 28):
        for tmpl in (f"prefill_L{l}", f"L{l}", f"layer_{l}", f"prefill_{l}"):
            if tmpl in keys:
                arr = blob[tmpl]
                if arr.shape == (N_PROBLEMS, HIDDEN):
                    per_layer[l] = arr.astype(np.float64)
                break
    if per_layer:
        idx = sorted(per_layer)
        stack = np.stack([per_layer[l] for l in idx], axis=1)
        return stack, {l: c for c, l in enumerate(idx)}

    # (a)/(c) 3D array
    arr3d = None
    for k in ("prefill_layers", "prefill", "hidden", "hidden_states", "activations"):
        if k in keys and blob[k].ndim == 3:
            arr3d = blob[k].astype(np.float64)
            break
    if arr3d is None:
        return None
    # Orient so axis 0 is problems, axis 2 is hidden.
    if arr3d.shape[0] != N_PROBLEMS and arr3d.shape[1] == N_PROBLEMS:
        arr3d = np.transpose(arr3d, (1, 0, 2))
    if arr3d.shape[0] != N_PROBLEMS or arr3d.shape[2] != HIDDEN:
        return None
    n_layers = arr3d.shape[1]
    if "layers" in keys:
        layer_index = [int(v) for v in blob["layers"]]
    else:
        # Assume the layer axis is dense and 0-based over the model's layers.
        layer_index = list(range(n_layers))
    mapping = {l: c for c, l in enumerate(layer_index)}
    return arr3d, mapping


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    base = np.load(CACHE)
    y = base["correct"].astype(bool)
    assert y.shape == (N_PROBLEMS,)

    loaded = load_layer_stack()
    if loaded is None:
        print(
            "MISSING_REGEN_INPUT all-layers prefill cache (tried: "
            + ", ".join(str(p) for p in ALLLAYER_CANDIDATES) + ")",
            file=sys.stderr,
        )
        return 2
    stack, mapping = loaded

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Sanity cross-check: the L19 slice in the all-layers cache should agree
    # with the headline L19-only cache if both are present.
    l19_consistency = None
    if L19 in mapping and "prefill" in base.files and base["prefill"].ndim == 2:
        base_l19 = base["prefill"].astype(np.float64)
        if base_l19.shape == (N_PROBLEMS, HIDDEN):
            a = oof_dom_auroc(base_l19, y, folds)
            b = oof_dom_auroc(stack[:, mapping[L19], :], y, folds)
            l19_consistency = {"headline_l19_auroc": a, "stack_l19_auroc": b,
                               "abs_diff": abs(a - b)}

    per_layer_auroc = {}
    for l in LAYERS:
        if l not in mapping:
            continue
        per_layer_auroc[l] = oof_dom_auroc(stack[:, mapping[l], :], y, folds)

    if not per_layer_auroc:
        print("MISSING_REGEN_INPUT no L4..L27 layers found in cache", file=sys.stderr)
        return 2

    vals = np.array([per_layer_auroc[l] for l in sorted(per_layer_auroc)])
    layers_present = sorted(per_layer_auroc)
    best_layer = int(layers_present[int(np.argmax(vals))])
    l19_auroc = per_layer_auroc.get(L19, float("nan"))

    # How far above the mid-layer band does L19 sit? Mid-band excludes L19.
    others = np.array([per_layer_auroc[l] for l in layers_present if l != L19])
    band_mean = float(others.mean()) if others.size else float("nan")
    band_std = float(others.std(ddof=1)) if others.size > 1 else float("nan")
    l19_z = (float(l19_auroc) - band_mean) / band_std if (others.size > 1 and band_std > 0) else float("nan")
    spread = float(vals.max() - vals.min())

    out = {
        "experiment": "P11-FE288",
        "model": "Qwen2.5-1.5B",
        "n_problems": int(N_PROBLEMS),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "layers_swept": layers_present,
        "auroc_by_layer": {str(l): float(per_layer_auroc[l]) for l in layers_present},
        "l19_auroc": float(l19_auroc),
        "best_layer": best_layer,
        "best_layer_auroc": float(vals.max()),
        "midband_mean_excl_l19": band_mean,
        "midband_std_excl_l19": band_std,
        "l19_zscore_vs_band": float(l19_z),
        "auroc_spread_max_minus_min": spread,
        # L19 "stands out" if it is the argmax AND >2 SD above the rest of the band.
        "l19_stands_out": bool(best_layer == L19 and np.isfinite(l19_z) and l19_z > 2.0),
        "l19_consistency_check": l19_consistency,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())