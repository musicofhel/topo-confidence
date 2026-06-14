"""P11-FE469 — Project Sinii et al. (2505.18706) RL-trained per-layer steering
vectors onto our cached F-2 probe directions.

Sinii et al. train per-layer steering vectors s_ℓ for Qwen2.5-1.5B (public
checkpoint, github.com/corl-team/steering-reasoning). This script loads those
vectors from a LOCAL cache (no network), reconstructs our prefill_L19_DoM and
final_L19_DoM directions from cached activations, and computes
cos(s_ℓ, prefill_DoM) and cos(s_ℓ, final_DoM) for every layer ℓ.

Verdict logic: a high |cos(s_17, prefill_DoM)| or |cos(s_19, prefill_DoM)|
corroborates F-2 as a causal direction (RL training rediscovered the probe
axis); near-zero cosine at every layer demotes F-2 to a correlational probe and
flags Sinii's vectors as the distinct, actually-causal direction.

Required local inputs (the Sinii checkpoint must be pre-exported to an NPZ
offline — this script never touches the network):
  pathway11_h100/sinii_steering/sinii_steering_vectors.npz
    one 2-D array of shape (n_layers, 1536) of per-layer steering vectors,
    optionally a 1-D "layers" array of the corresponding layer indices.
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
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
SINII_NPZ = ROOT / "pathway11_h100/sinii_steering/sinii_steering_vectors.npz"
OUT_JSON = ROOT / "pathway11_h100/sinii_steering/results.json"

HIDDEN = 1536
KEY_LAYERS = (17, 19)  # L17, L19 are F-2's prefill / candidate causal layers
CORROBORATE_THRESH = 0.30  # |cos| above this at a key layer => RL rediscovered F-2
DEMOTE_THRESH = 0.10       # |cos| below this at every layer => probe is correlational


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means direction (correct minus incorrect), unit-normalized."""
    return unit(X[y].mean(axis=0) - X[~y].mean(axis=0))


def load_steering_vectors(blob) -> tuple[np.ndarray, np.ndarray]:
    """Return (vectors (L,1536), layer_indices (L,)) from a flexibly-keyed NPZ."""
    mat = None
    for key in blob.files:
        arr = np.asarray(blob[key])
        if arr.ndim == 2 and arr.shape[1] == HIDDEN:
            mat = arr.astype(np.float64)
            break
    if mat is None:
        raise ValueError(f"no (L,{HIDDEN}) steering matrix in {SINII_NPZ}")
    if "layers" in blob.files:
        layers = np.asarray(blob["layers"]).astype(int).ravel()
        if layers.shape[0] != mat.shape[0]:
            layers = np.arange(mat.shape[0], dtype=int)
    else:
        layers = np.arange(mat.shape[0], dtype=int)
    return mat, layers


def main() -> int:
    if not SINII_NPZ.exists():
        print("MISSING_REGEN_INPUT", SINII_NPZ, file=sys.stderr)
        return 2
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    # --- F-2 prefill_L19_DoM direction (and sanity AUROC) -------------------
    cache = np.load(CACHE)
    Xp = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert Xp.shape == (500, HIDDEN) and y.shape == (500,)
    prefill_dom = dom_direction(Xp, y)

    sanity_auroc = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        sanity_auroc = auroc(dom_score, y)

    # --- final_L19_DoM direction (optional; null if final cache absent) -----
    final_dom = None
    final_cache_present = FINAL_CACHE.exists()
    if final_cache_present:
        fblob = np.load(FINAL_CACHE)
        fkey = "final" if "final" in fblob.files else (
            "prefill" if "prefill" in fblob.files else None)
        if fkey is not None:
            Xf = fblob[fkey].astype(np.float64)
            yf = (fblob["correct"].astype(bool)
                  if "correct" in fblob.files else y)
            if Xf.shape[1] == HIDDEN and Xf.shape[0] == yf.shape[0]:
                final_dom = dom_direction(Xf, yf)

    # --- Sinii per-layer steering vectors -----------------------------------
    sinii = np.load(SINII_NPZ)
    S, layers = load_steering_vectors(sinii)
    S_unit = np.array([unit(s) for s in S])

    cos_prefill = (S_unit @ prefill_dom).tolist()
    cos_final = ((S_unit @ final_dom).tolist()
                 if final_dom is not None else None)

    per_layer = []
    for i, lyr in enumerate(layers.tolist()):
        per_layer.append({
            "layer": int(lyr),
            "cos_prefill_dom": float(cos_prefill[i]),
            "cos_final_dom": (float(cos_final[i])
                              if cos_final is not None else None),
        })

    # --- verdict ------------------------------------------------------------
    layer_to_cos = {int(l): float(c) for l, c in zip(layers, cos_prefill)}
    key_cos = {l: layer_to_cos.get(l) for l in KEY_LAYERS}
    key_abs = [abs(v) for v in key_cos.values() if v is not None]
    all_abs = [abs(c) for c in cos_prefill]
    max_abs_any = float(max(all_abs)) if all_abs else float("nan")
    argmax_layer = int(layers[int(np.argmax(np.abs(cos_prefill)))]) if all_abs else None

    if key_abs and max(key_abs) >= CORROBORATE_THRESH:
        verdict = "CORROBORATES_F2_CAUSAL"
    elif all_abs and max_abs_any < DEMOTE_THRESH:
        verdict = "DEMOTES_F2_TO_CORRELATIONAL"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "P11-FE469",
        "n_layers": int(S.shape[0]),
        "sanity_prefill_dom_auroc": sanity_auroc,
        "final_dom_available": final_dom is not None,
        "key_layers": list(KEY_LAYERS),
        "key_layer_cos_prefill_dom": {str(k): v for k, v in key_cos.items()},
        "max_abs_cos_prefill_dom": max_abs_any,
        "argmax_layer_prefill_dom": argmax_layer,
        "corroborate_thresh": CORROBORATE_THRESH,
        "demote_thresh": DEMOTE_THRESH,
        "verdict": verdict,
        "per_layer": per_layer,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())