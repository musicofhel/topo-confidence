"""FE284 — Does F-3's prefill/final-token orthogonality reduce to Lad et al.'s
prediction/suppression neuron taxonomy?

Lad et al. (2406.19384) Stage 3 (prediction ensembling) and Stage 4 (residual
sharpening), with Gurnee et al.'s neuron taxonomy (2401.12181), predict that
late-layer MLP neurons split into a Pred-set (boost specific next tokens —
heavy positive logit-attribution tails) and a Supp-set (suppress tokens —
negative skew). The hypothesis under test: our cached L19 *prefill* DoM
direction lives mostly in the Pred-set sub-basis at L19-L24, while the
*final-token* DoM direction lives mostly in the Supp-set sub-basis at L24-L27.
If both hold, the cos≈0.046 orthogonality F-3 reports is a re-description of a
known prediction/suppression duality, not a novel geometric claim.

Method (no forward passes — pure weight projection):
  1. For each MLP layer L, read w_out (d_mlp, d_model). Per neuron j compute
     its logit-attribution vector a_j = W_U @ w_out[j] over the vocab, then its
     kurtosis and skew (Lad/Gurnee statistics).
  2. Pred-set(L) = neurons with kurtosis above the per-layer 90th pct;
     Supp-set(L) = neurons with skew below the per-layer 10th pct (neg skew).
  3. Compute the prefill DoM direction from the cached L19 activations
     (mean_correct - mean_incorrect) and load the final-token DoM direction.
  4. Project each DoM direction into both neuron sub-bases: report the fraction
     of the direction's energy captured by the unit w_out vectors of each set
     (sum of squared cosines), per layer.
  5. Adjudicate the prediction.

Inputs that must be regenerated on a fresh machine (no transformers here — a
separate GPU pass dumps the raw weights to NPZ): the Qwen2.5-1.5B MLP w_out
matrices + unembedding W_U, and the final-token DoM direction. If absent we
emit MISSING_REGEN_INPUT and exit 2.
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
from scipy.stats import kurtosis, skew

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# GPU-side dump (no forward passes needed at recompute time): per-layer MLP
# down-projection w_out matrices keyed "w_out_<L>" : (d_mlp, d_model), plus the
# unembedding "W_U" : (vocab, d_model).
WEIGHTS = ROOT / "pathway11_h100/lad_neuron_basis/qwen15b_mlp_weights.npz"
# Final-token DoM direction in residual-stream coords: "final_dom" : (1536,).
FINAL_DOM_NPZ = ROOT / "pathway11_h100/lad_neuron_basis/final_token_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/lad_neuron_basis/results.json"

PREFILL_LAYER = 19
PRED_LAYERS = list(range(19, 25))   # L19-L24: prefill ↦ Pred-set prediction
SUPP_LAYERS = list(range(24, 28))   # L24-L27: final  ↦ Supp-set prediction
PRED_PCT = 90.0                     # kurtosis upper percentile → Pred-set
SUPP_PCT = 10.0                     # skew lower percentile → Supp-set
OVERLAP_GAP = 0.05                  # min energy gap to call a directional bias


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def energy_in_set(dom_unit: np.ndarray, w_out_unit: np.ndarray, idx: np.ndarray) -> dict:
    """Fraction of dom_unit's energy captured by neuron output directions in idx.

    w_out_unit rows are unit-normalized neuron output directions. Energy is the
    sum of squared cosines over the (generally non-orthogonal) set — a bounded
    overlap proxy, plus the mean |cos| for interpretability.
    """
    if len(idx) == 0:
        return {"energy": 0.0, "mean_abs_cos": 0.0, "n": 0}
    cos = w_out_unit[idx] @ dom_unit
    return {
        "energy": float(np.sum(cos ** 2)),
        "mean_abs_cos": float(np.mean(np.abs(cos))),
        "n": int(len(idx)),
    }


def main() -> int:
    for p in (CACHE, WEIGHTS, FINAL_DOM_NPZ):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
            return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # Prefill DoM direction from the L19 cache (the supervised mean-difference).
    prefill_dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    prefill_dom_u = unit(prefill_dom)
    # Sanity: this direction is the one F-3 reports (AUROC ≈ 0.77 in-sample).
    prefill_dom_auroc = auroc(X @ prefill_dom, y)

    final_blob = np.load(FINAL_DOM_NPZ)
    final_dom = final_blob["final_dom"].astype(np.float64).reshape(-1)
    if final_dom.shape != (1536,):
        print("MISSING_REGEN_INPUT", FINAL_DOM_NPZ, "bad shape", final_dom.shape, file=sys.stderr)
        return 2
    final_dom_u = unit(final_dom)

    cos_prefill_final = float(prefill_dom_u @ final_dom_u)

    weights = np.load(WEIGHTS)
    if "W_U" not in weights:
        print("MISSING_REGEN_INPUT", WEIGHTS, "no W_U", file=sys.stderr)
        return 2
    W_U = weights["W_U"].astype(np.float64)   # (vocab, d_model)
    if W_U.shape[1] != 1536:
        # tolerate (d_model, vocab) orientation
        if W_U.shape[0] == 1536:
            W_U = W_U.T
        else:
            print("MISSING_REGEN_INPUT", WEIGHTS, "W_U dim", W_U.shape, file=sys.stderr)
            return 2

    # Discover available MLP layers.
    layer_keys = {}
    for k in weights.files:
        if k.startswith("w_out_"):
            try:
                layer_keys[int(k.split("_")[-1])] = k
            except ValueError:
                continue
    if not layer_keys:
        print("MISSING_REGEN_INPUT", WEIGHTS, "no w_out_<L>", file=sys.stderr)
        return 2

    per_layer = {}
    for L in sorted(layer_keys):
        w_out = weights[layer_keys[L]].astype(np.float64)   # (d_mlp, d_model)
        if w_out.shape[1] != 1536:
            if w_out.shape[0] == 1536:
                w_out = w_out.T
            else:
                continue
        # Logit-attribution vector per neuron, then Lad/Gurnee tail statistics.
        attrib = w_out @ W_U.T                  # (d_mlp, vocab)
        kurt = kurtosis(attrib, axis=1, fisher=True, bias=False)
        skw = skew(attrib, axis=1, bias=False)

        pred_thr = float(np.percentile(kurt, PRED_PCT))
        supp_thr = float(np.percentile(skw, SUPP_PCT))
        pred_idx = np.flatnonzero(kurt >= pred_thr)
        supp_idx = np.flatnonzero(skw <= supp_thr)

        norms = np.linalg.norm(w_out, axis=1)
        keep = norms > 1e-12
        w_out_u = np.zeros_like(w_out)
        w_out_u[keep] = w_out[keep] / norms[keep, None]

        per_layer[str(L)] = {
            "n_neurons": int(w_out.shape[0]),
            "n_pred": int(len(pred_idx)),
            "n_supp": int(len(supp_idx)),
            "prefill_in_pred": energy_in_set(prefill_dom_u, w_out_u, pred_idx),
            "prefill_in_supp": energy_in_set(prefill_dom_u, w_out_u, supp_idx),
            "final_in_pred": energy_in_set(final_dom_u, w_out_u, pred_idx),
            "final_in_supp": energy_in_set(final_dom_u, w_out_u, supp_idx),
        }

    def mean_energy(layers, dom_key, set_key):
        vals = [per_layer[str(L)][f"{dom_key}_in_{set_key}"]["energy"]
                for L in layers if str(L) in per_layer]
        return float(np.mean(vals)) if vals else float("nan")

    prefill_pred_L19_24 = mean_energy(PRED_LAYERS, "prefill", "pred")
    prefill_supp_L19_24 = mean_energy(PRED_LAYERS, "prefill", "supp")
    final_supp_L24_27 = mean_energy(SUPP_LAYERS, "final", "supp")
    final_pred_L24_27 = mean_energy(SUPP_LAYERS, "final", "pred")

    # Prediction holds iff prefill prefers Pred-set in L19-L24 AND final prefers
    # Supp-set in L24-L27, each by a non-trivial energy gap.
    prefill_prefers_pred = (prefill_pred_L19_24 - prefill_supp_L19_24) > OVERLAP_GAP
    final_prefers_supp = (final_supp_L24_27 - final_pred_L24_27) > OVERLAP_GAP
    duality_explains_f3 = bool(prefill_prefers_pred and final_prefers_supp)

    out = {
        "experiment": "P11-FE284",
        "description": "Project prefill/final DoM into Lad pred/supp neuron sub-bases",
        "cos_prefill_final_dom": cos_prefill_final,
        "prefill_dom_auroc_insample": float(prefill_dom_auroc),
        "prefill_layer": PREFILL_LAYER,
        "pred_layers": PRED_LAYERS,
        "supp_layers": SUPP_LAYERS,
        "pred_percentile_kurtosis": PRED_PCT,
        "supp_percentile_skew": SUPP_PCT,
        "summary": {
            "prefill_energy_in_pred_L19_24": prefill_pred_L19_24,
            "prefill_energy_in_supp_L19_24": prefill_supp_L19_24,
            "final_energy_in_supp_L24_27": final_supp_L24_27,
            "final_energy_in_pred_L24_27": final_pred_L24_27,
            "prefill_prefers_pred": bool(prefill_prefers_pred),
            "final_prefers_supp": bool(final_prefers_supp),
        },
        "duality_explains_f3_orthogonality": duality_explains_f3,
        "verdict": (
            "F-3 orthogonality REDUCES to pred/supp duality (not novel)"
            if duality_explains_f3 else
            "F-3 orthogonality NOT explained by pred/supp duality (survives)"
        ),
        "per_layer": per_layer,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())