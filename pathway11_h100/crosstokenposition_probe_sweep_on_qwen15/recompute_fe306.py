"""P11-FE306 — Cross-token-position × layer probe sweep on Qwen-1.5B + MATH-500.

Builds an AUROC heatmap (Orgad et al. 2024 Figure 2 analogue) by sweeping a
mean-difference (DoM) probe across the last 50 generated token positions × the
layer band {L17, L18, L19, L20, L21}. For each (layer, position) cell we fit a
DoM direction out-of-fold (5-fold stratified) and score held-out problems,
giving a leakage-free AUROC per cell.

The point is to test F-2's "prefill is the peak" claim: Orgad et al. report
truthfulness encoding peaking at the *exact answer tokens* rather than the
prefill. We compare the best sweep cell (the empirical LEAT-style peak) against
the L19 prefill AUROC from the canonical Stage 2 cache and report whether any
intermediate token/layer beats prefill.

Required input is a multi-position × multi-layer activation cache:
  pathway11_h100/prefill_inversion/cache/m15b_tokenpos_layers.npz
    hidden:        (500, P, L, 1536) float32  — [problem, token_pos, layer, dim]
    correct:       (500,)            bool
    layer_ids:     (L,)              int   — e.g. [17,18,19,20,21]   (optional)
    token_offsets: (P,)             int   — offsets from final token, e.g. -50..-1 (optional)
This cache is NOT one of the committed Stage 2 NPZs; if absent the script emits
MISSING_REGEN_INPUT and exits 2 (regenerate via the H100 multi-layer extract).
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
SWEEP_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_tokenpos_layers.npz"
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/token_position_sweep/results.json"

N_FOLDS = 5
SEED = 9999
EXPECTED_LAYERS = [17, 18, 19, 20, 21]
N_LAST_TOKENS = 50


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
    """Out-of-fold mean-difference probe AUROC for one (layer, position) cell."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        if ytr.all() or (~ytr).any() is False or ytr.sum() == 0 or (~ytr).sum() == 0:
            oof[test_idx] = Xte @ np.zeros(X.shape[1])
            continue
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof[test_idx] = Xte @ d_vec
    return auroc(oof, y)


def prefill_reference(y: np.ndarray, folds: list[np.ndarray]) -> dict:
    """L19 prefill DoM AUROC, for the prefill-vs-sweep comparison."""
    ref = {"prefill_oof_dom_auroc": None, "prefill_phase2_dom_auroc": None}
    if PREFILL_CACHE.exists():
        Xp = np.load(PREFILL_CACHE)["prefill"].astype(np.float64)
        if Xp.shape == (len(y), 1536):
            ref["prefill_oof_dom_auroc"] = oof_dom_auroc(Xp, y, folds)
    if DOM_NPZ.exists():
        dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom.shape == (len(y),):
            ref["prefill_phase2_dom_auroc"] = auroc(dom, y)
    return ref


def main() -> int:
    if not SWEEP_CACHE.exists():
        print("MISSING_REGEN_INPUT", SWEEP_CACHE, file=sys.stderr)
        return 2

    blob = np.load(SWEEP_CACHE)
    if "hidden" not in blob or "correct" not in blob:
        print("MISSING_REGEN_INPUT", "hidden/correct keys", SWEEP_CACHE, file=sys.stderr)
        return 2

    hidden = blob["hidden"]
    y = blob["correct"].astype(bool)
    if hidden.ndim != 4 or hidden.shape[0] != len(y) or hidden.shape[-1] != 1536:
        print("MISSING_REGEN_INPUT", "bad hidden shape", hidden.shape, file=sys.stderr)
        return 2

    n_prob, n_pos, n_layer, _ = hidden.shape
    layer_ids = (blob["layer_ids"].astype(int).tolist()
                 if "layer_ids" in blob else EXPECTED_LAYERS[:n_layer])
    token_offsets = (blob["token_offsets"].astype(int).tolist()
                     if "token_offsets" in blob
                     else list(range(-min(n_pos, N_LAST_TOKENS), 0)))

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # AUROC heatmap: rows = layers, cols = token positions (last 50).
    pos_slice = range(max(0, n_pos - N_LAST_TOKENS), n_pos)
    heatmap = np.full((n_layer, len(list(pos_slice))), np.nan, dtype=np.float64)
    pos_list = list(pos_slice)
    for li in range(n_layer):
        for ci, pi in enumerate(pos_list):
            X = hidden[:, pi, li, :].astype(np.float64)
            heatmap[li, ci] = oof_dom_auroc(X, y, folds)

    # Peak cell across the whole sweep.
    flat = np.nanargmax(heatmap)
    peak_li, peak_ci = np.unravel_index(flat, heatmap.shape)
    peak_auroc = float(heatmap[peak_li, peak_ci])
    peak_layer = int(layer_ids[peak_li]) if peak_li < len(layer_ids) else int(peak_li)
    peak_offset = int(token_offsets[pos_list[peak_ci]]) if pos_list[peak_ci] < len(token_offsets) else int(pos_list[peak_ci])

    ref = prefill_reference(y, folds)
    prefill_auroc = ref["prefill_oof_dom_auroc"]
    beats_prefill = (prefill_auroc is not None and not np.isnan(peak_auroc)
                     and peak_auroc > prefill_auroc)

    verdict = "INCONCLUSIVE"
    if prefill_auroc is not None and not np.isnan(peak_auroc):
        margin = peak_auroc - prefill_auroc
        if margin > 0.01:
            verdict = "REFUTES_F2_PREFILL_PEAK"   # an intermediate token/layer is stronger
        elif margin < -0.005:
            verdict = "CONFIRMS_F2_PREFILL_PEAK"   # prefill strictly dominates the sweep
        else:
            verdict = "PREFILL_TIES_SWEEP_PEAK"

    out = {
        "experiment": "P11-FE306",
        "n_problems": int(n_prob),
        "n_positions_swept": len(pos_list),
        "layer_ids": layer_ids,
        "token_offsets": [int(token_offsets[p]) if p < len(token_offsets) else int(p)
                          for p in pos_list],
        "auroc_heatmap": [[None if np.isnan(v) else float(v) for v in row]
                          for row in heatmap],
        "peak_auroc": peak_auroc,
        "peak_layer": peak_layer,
        "peak_token_offset": peak_offset,
        "prefill_oof_dom_auroc": prefill_auroc,
        "prefill_phase2_dom_auroc": ref["prefill_phase2_dom_auroc"],
        "peak_beats_prefill": bool(beats_prefill),
        "peak_minus_prefill": (None if prefill_auroc is None or np.isnan(peak_auroc)
                               else float(peak_auroc - prefill_auroc)),
        "verdict": verdict,
        "n_folds": N_FOLDS,
        "seed": SEED,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())