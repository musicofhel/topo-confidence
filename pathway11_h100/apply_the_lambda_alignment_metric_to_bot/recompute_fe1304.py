"""P11-FE1304 — Lambda weight-read alignment for prefill-DoM and final-token-DoM.

The "lambda" alignment metric measures how strongly a correctness read direction
(a DoM vector in the L19 residual stream, 1536-d) aligns with the *leading right
singular direction* v1 of an MLP weight matrix W (input dim 1536). We report two
flavours:

  lambda_cos  = |cos(DoM_unit, v1)|                     (pure direction overlap)
  lambda_gain = sqrt(DoM_unit^T (W^T W) DoM_unit) / s1   (spectral gain vs s1)

Two questions:
  (a) Can the two near-orthogonal correctness reads (prefill-DoM and final-DoM,
      cos ~ 0.046 per F-3) BOTH align highly with the *same* L19 v1? If so the
      single-dominant-axis premise is in trouble.
  (b) Sweeping the L19 read against all 28 layers' MLP weights, is L19 the argmax
      of weight-read alignment? Independent check on the F-2 hand-pick / H-921
      Fisher-J layer choice. If an all-layer prefill cache is present we also run
      the diagonal per-layer sweep lambda(DoM_layer, W_layer).

CPU only, no network, no GPU. Leading right singular vector is obtained from the
1536x1536 Gram matrix W^T W via eigh (fast, exact for the top direction) rather
than a full SVD of the tall (m x 1536) weight matrix.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
WEIGHTS_NPZ = ROOT / "pathway11_h100/data/mlp_weights.npz"
ALL_PREFILL = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz"
OUT_JSON = ROOT / "pathway11_h100/lambda_weight_alignment/results.json"

D = 1536
N_LAYERS = 28
DOM_LAYER = 19  # layers indexed 0..27; L19 == index 19


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
    return v / n if n > 0 else v


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means correctness direction (correct minus incorrect)."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


def detect_layers(npz) -> dict | None:
    """Return {layer_idx -> W} for the 28 MLP weight matrices, or None."""
    schemes = [
        lambda i: f"W_L{i:02d}",
        lambda i: f"layer_{i}",
        lambda i: f"mlp_{i}",
        lambda i: f"gate_proj_L{i:02d}",
        lambda i: f"L{i}",
    ]
    for scheme in schemes:
        keys = [scheme(i) for i in range(N_LAYERS)]
        if all(k in npz.files for k in keys):
            return {i: npz[keys[i]] for i in range(N_LAYERS)}
    # fallback: exactly 28 two-dimensional arrays, ordered by (zero-padded) key
    arrs = [(k, npz[k]) for k in npz.files if np.asarray(npz[k]).ndim == 2]
    if len(arrs) == N_LAYERS:
        def keyfn(item):
            m = re.search(r"(\d+)", item[0])
            return (int(m.group(1)) if m else 0, item[0])
        arrs.sort(key=keyfn)
        return {i: a for i, (k, a) in enumerate(arrs)}
    return None


def orient(W: np.ndarray) -> np.ndarray:
    """Return W with the 1536 residual-stream (input) dimension on axis 1."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"weight not 2D: {W.shape}")
    if W.shape[1] == D:
        return W
    if W.shape[0] == D:
        return W.T
    raise ValueError(f"weight has no dim == {D}: {W.shape}")


def v1_and_gram(W: np.ndarray):
    """Leading right singular vector v1, sigma1, and the Gram matrix W^T W."""
    G = W.T @ W  # (D, D)
    evals, evecs = np.linalg.eigh(G)  # ascending
    v1 = evecs[:, -1]
    sigma1 = float(np.sqrt(max(evals[-1], 0.0)))
    return v1, sigma1, G


def lambdas_against_layers(d_unit, grams, v1s, sigma1s):
    """lambda_cos and lambda_gain of a unit read vector against every layer."""
    lam_cos, lam_gain = [], []
    for i in range(N_LAYERS):
        lam_cos.append(abs(float(d_unit @ v1s[i])))
        s1 = sigma1s[i]
        gain = float(np.sqrt(max(d_unit @ grams[i] @ d_unit, 0.0)))
        lam_gain.append(gain / s1 if s1 > 0 else float("nan"))
    return lam_cos, lam_gain


def main() -> int:
    for path in (PREFILL_CACHE, FINAL_CACHE, WEIGHTS_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2

    pre = np.load(PREFILL_CACHE)
    Xp = pre["prefill"].astype(np.float64)
    y = pre["correct"].astype(bool)
    if Xp.shape != (500, D) or y.shape != (500,):
        print("BAD_PREFILL_SHAPE", Xp.shape, y.shape, file=sys.stderr)
        return 2

    fin = np.load(FINAL_CACHE)
    fin_key = "final" if "final" in fin.files else fin.files[0]
    Xf = fin[fin_key].astype(np.float64)
    if Xf.shape != (500, D):
        print("BAD_FINAL_SHAPE", Xf.shape, file=sys.stderr)
        return 2

    # --- L19 reads -----------------------------------------------------------
    prefill_dom = dom_direction(Xp, y)
    final_dom = dom_direction(Xf, y)
    pre_u = unit(prefill_dom)
    fin_u = unit(final_dom)
    cos_prefill_final = cos(prefill_dom, final_dom)

    # AUROC sanity of each read (in-sample projection; OOF lives elsewhere)
    if DOM_NPZ.exists():
        prefill_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    else:
        prefill_score = Xp @ prefill_dom
    final_score = Xf @ final_dom

    # --- MLP weights → per-layer v1 ------------------------------------------
    wnpz = np.load(WEIGHTS_NPZ)
    layer_W = detect_layers(wnpz)
    if layer_W is None:
        print("UNRECOGNIZED_WEIGHTS_LAYOUT", list(wnpz.files)[:8], file=sys.stderr)
        return 2

    v1s, sigma1s, grams = [], [], []
    for i in range(N_LAYERS):
        W = orient(layer_W[i])
        v1, s1, G = v1_and_gram(W)
        v1s.append(v1)
        sigma1s.append(s1)
        grams.append(G)

    pre_cos, pre_gain = lambdas_against_layers(pre_u, grams, v1s, sigma1s)
    fin_cos, fin_gain = lambdas_against_layers(fin_u, grams, v1s, sigma1s)

    pre_argmax = int(np.argmax(pre_cos))
    fin_argmax = int(np.argmax(fin_cos))
    pre_argmax_gain = int(np.argmax(pre_gain))
    fin_argmax_gain = int(np.argmax(fin_gain))

    # --- optional diagonal per-layer DoM sweep -------------------------------
    diagonal = None
    if ALL_PREFILL.exists():
        allnpz = np.load(ALL_PREFILL)
        akey = "prefill_all" if "prefill_all" in allnpz.files else allnpz.files[0]
        Xall = allnpz[akey].astype(np.float64)  # expected (500, 28, 1536)
        if Xall.ndim == 3 and Xall.shape[0] == 500 and Xall.shape[2] == D:
            n_avail = min(Xall.shape[1], N_LAYERS)
            diag_cos, diag_gain = [], []
            for i in range(n_avail):
                d_u = unit(dom_direction(Xall[:, i, :], y))
                diag_cos.append(abs(float(d_u @ v1s[i])))
                s1 = sigma1s[i]
                gain = float(np.sqrt(max(d_u @ grams[i] @ d_u, 0.0)))
                diag_gain.append(gain / s1 if s1 > 0 else float("nan"))
            diagonal = {
                "n_layers_available": int(n_avail),
                "lambda_cos_per_layer": [float(x) for x in diag_cos],
                "lambda_gain_per_layer": [float(x) for x in diag_gain],
                "argmax_layer_cos": int(np.argmax(diag_cos)),
                "l19_is_argmax_cos": bool(int(np.argmax(diag_cos)) == DOM_LAYER),
            }

    out = {
        "experiment": "P11-FE1304",
        "n": int(len(y)),
        "dim": D,
        "n_layers": N_LAYERS,
        "dom_layer_index": DOM_LAYER,
        "layer_indexing": "0..27; L19 == index 19",
        "cos_prefill_final_dom": cos_prefill_final,
        "auroc_prefill_read": float(auroc(prefill_score, y)),
        "auroc_final_read": float(auroc(final_score, y)),
        "sigma1_per_layer": [float(s) for s in sigma1s],
        "prefill": {
            "lambda_cos_v1_per_layer": [float(x) for x in pre_cos],
            "lambda_gain_per_layer": [float(x) for x in pre_gain],
            "lambda_cos_v1_L19": float(pre_cos[DOM_LAYER]),
            "lambda_gain_L19": float(pre_gain[DOM_LAYER]),
            "argmax_layer_cos": pre_argmax,
            "argmax_layer_gain": pre_argmax_gain,
            "l19_is_argmax_cos": bool(pre_argmax == DOM_LAYER),
            "l19_is_argmax_gain": bool(pre_argmax_gain == DOM_LAYER),
        },
        "final": {
            "lambda_cos_v1_per_layer": [float(x) for x in fin_cos],
            "lambda_gain_per_layer": [float(x) for x in fin_gain],
            "lambda_cos_v1_L19": float(fin_cos[DOM_LAYER]),
            "lambda_gain_L19": float(fin_gain[DOM_LAYER]),
            "argmax_layer_cos": fin_argmax,
            "argmax_layer_gain": fin_argmax_gain,
            "l19_is_argmax_cos": bool(fin_argmax == DOM_LAYER),
            "l19_is_argmax_gain": bool(fin_argmax_gain == DOM_LAYER),
        },
        "diagonal_per_layer_dom_sweep": diagonal,
        "interpretation": {
            "both_align_same_v1_L19": bool(
                pre_cos[DOM_LAYER] >= 0.3 and fin_cos[DOM_LAYER] >= 0.3
            ),
            "single_axis_premise_at_risk": bool(
                pre_cos[DOM_LAYER] >= 0.3
                and fin_cos[DOM_LAYER] >= 0.3
                and abs(cos_prefill_final) < 0.2
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())