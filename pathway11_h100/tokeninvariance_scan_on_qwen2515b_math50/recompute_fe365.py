"""P11-FE365 — Token-invariance scan of the DoM direction across layers.

Tests whether F-3's cos(prefill_DoM, final_DoM) = 0.046 (orthogonal) at L19 is a
layer-local phenomenon or persists at the readout layer. OPENIA Section 5.2.2
reports that at deep layers probe F1 is stable across token positions, implying
the correctness direction becomes approximately token-invariant at depth. If so,
cos(DoM_first-prompt-token(L), DoM_final-decoded-token(L)) should climb toward 1
for L >= 24, which would make F-3's orthogonality narrative layer-local and
overstated.

For every layer L = 0..N-1 we compute:
  d_first(L)  = mean_{correct} h_first(L) - mean_{incorrect} h_first(L)
  d_final(L)  = mean_{correct} h_final(L) - mean_{incorrect} h_final(L)
  cos(L)      = <d_first(L), d_final(L)> / (||d_first(L)|| ||d_final(L)||)

and report the per-layer cosine curve plus the L19 and last-layer values.

Inputs are the cached layerwise activations (first-prompt-token and
final-decoded-token hidden states at every layer). If the layerwise cache is
absent the script prints MISSING_REGEN_INPUT and returns 2.
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
# Candidate locations for the layerwise (all-layer) activation cache.
LAYERWISE_CANDIDATES = [
    ROOT / "pathway11_h100/layerwise_dom/cache/m15b_layerwise.npz",
    ROOT / "pathway11_h100/layerwise_dom/cache/m15b_layerwise_prefill.npz",
    ROOT / "pathway11_h100/layerwise/cache/m15b_layerwise.npz",
    ROOT / "pathway11_h100/data/layerwise_dom/m15b_layerwise.npz",
]
# Fallback label source (canonical correctness labels) if the layerwise cache
# does not carry its own.
LABEL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/layerwise_dom/results.json"

L19 = 19

FIRST_KEYS = ["prefill_layers", "prefill_all", "first_token_layers",
              "first_layers", "prompt_layers", "prefill"]
FINAL_KEYS = ["final_layers", "final_all", "final_token_layers",
              "decoded_layers", "last_token_layers", "final"]
LABEL_KEYS = ["correct", "labels", "y"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _find_layerwise() -> Path | None:
    for p in LAYERWISE_CANDIDATES:
        if p.exists():
            return p
    return None


def _pick(blob, keys):
    for k in keys:
        if k in blob.files:
            return blob[k]
    return None


def _as_layered(arr: np.ndarray) -> np.ndarray:
    """Normalize to shape (n_samples=500, n_layers, hidden) with layer axis=1."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D layerwise array, got shape {arr.shape}")
    # Put the 500-sample axis first.
    sample_axis = int(np.argmax([d == 500 for d in arr.shape]))
    if sample_axis != 0:
        arr = np.moveaxis(arr, sample_axis, 0)
    # Remaining two axes are (n_layers, hidden); hidden is the larger (1536).
    if arr.shape[1] > arr.shape[2]:
        arr = np.swapaxes(arr, 1, 2)
    return arr


def main() -> int:
    cache = _find_layerwise()
    if cache is None:
        print("MISSING_REGEN_INPUT", LAYERWISE_CANDIDATES[0], file=sys.stderr)
        return 2

    blob = np.load(cache)
    first = _pick(blob, FIRST_KEYS)
    final = _pick(blob, FINAL_KEYS)
    if first is None or final is None:
        print("MISSING_REGEN_INPUT", f"{cache} (no first/final layerwise arrays)",
              file=sys.stderr)
        return 2

    first = _as_layered(first).astype(np.float64)
    final = _as_layered(final).astype(np.float64)

    y = _pick(blob, LABEL_KEYS)
    if y is None:
        if not LABEL_CACHE.exists():
            print("MISSING_REGEN_INPUT", LABEL_CACHE, file=sys.stderr)
            return 2
        y = np.load(LABEL_CACHE)["correct"]
    y = np.asarray(y).astype(bool)

    n, n_layers, hidden = first.shape
    if final.shape != first.shape:
        # Align on the common number of layers if they differ.
        n_layers = min(first.shape[1], final.shape[1])
        first = first[:, :n_layers, :]
        final = final[:, :n_layers, :]
    if y.shape[0] != n:
        print("MISSING_REGEN_INPUT",
              f"label/activation sample mismatch {y.shape} vs {n}", file=sys.stderr)
        return 2

    pos = y
    neg = ~y
    per_layer = []
    cos_curve = []
    for L in range(n_layers):
        hf = first[:, L, :]
        hl = final[:, L, :]
        d_first = hf[pos].mean(axis=0) - hf[neg].mean(axis=0)
        d_final = hl[pos].mean(axis=0) - hl[neg].mean(axis=0)
        c = cosine(d_first, d_final)
        cos_curve.append(c)
        per_layer.append({
            "layer": L,
            "cos_dom_first_final": c,
            "auroc_first": auroc(hf @ d_first, y),
            "auroc_final": auroc(hl @ d_final, y),
        })

    cos_arr = np.array(cos_curve, dtype=np.float64)
    deep = [c["cos_dom_first_final"] for c in per_layer if c["layer"] >= 24
            and not np.isnan(c["cos_dom_first_final"])]
    l19_cos = (per_layer[L19]["cos_dom_first_final"]
               if n_layers > L19 else float("nan"))
    last_cos = per_layer[-1]["cos_dom_first_final"]

    deep_mean = float(np.mean(deep)) if deep else float("nan")
    # F-3 is "layer-local / overstated" if cos climbs toward 1 at depth while
    # remaining near-orthogonal at L19.
    f3_layer_local = bool(
        (not np.isnan(l19_cos)) and abs(l19_cos) < 0.15
        and (not np.isnan(deep_mean)) and deep_mean > 0.5
    )

    out = {
        "experiment": "P11-FE365",
        "cache": str(cache.relative_to(ROOT)),
        "n_samples": int(n),
        "n_layers": int(n_layers),
        "hidden": int(hidden),
        "cos_dom_first_final_l19": l19_cos,
        "cos_dom_first_final_last": last_cos,
        "cos_dom_first_final_deep_mean_L>=24": deep_mean,
        "cos_dom_first_final_max": float(np.nanmax(cos_arr)),
        "cos_dom_first_final_argmax_layer": int(np.nanargmax(cos_arr)),
        "f3_cos046_reference_l19": 0.046,
        "f3_layer_local_verdict": f3_layer_local,
        "per_layer": per_layer,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())