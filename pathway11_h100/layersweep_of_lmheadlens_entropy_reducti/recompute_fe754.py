"""P11-FE754 — Layer-sweep of LM-head-lens (logit-lens) entropy reduction I_ℓ.

For each transformer layer ℓ ∈ {0..27} of Qwen-2.5-1.5B on MATH-500 prefill we
take the LM-head ("logit lens") next-token distribution read off layer ℓ's
residual stream and form the per-example entropy-reduction signal

    I_ℓ(x→y) = H_ℓ(y | ∅) − H_ℓ(y | x)

where H_ℓ(y | x) is the entropy of the layer-ℓ logit-lens distribution under the
full prefill context x, and H_ℓ(y | ∅) is the entropy of the same head under the
empty / marginal context (a probe-free "how much did context sharpen the
prediction" quantity). We score AUROC of I_ℓ against ground-truth correctness for
every layer and take argmax_ℓ.

Decision rule (FE754): F-2 located L19 as special via a *linear DoM probe* sweep.
Layerwise CP entropy reduction is a probe-free per-layer signal. If
argmax_ℓ AUROC(I_ℓ) ≠ 19, then F-2's "L19 is special" is probe-method-dependent
and should weaken to "late-mid block (L18–L24) is special".

This is a recompute over cached per-layer logit-lens entropies — no model weights,
no GPU. It requires a per-layer entropy cache produced by the extraction job
(keys H_cond (500,28) and H_uncond (500,28)). If that cache is absent the script
reports MISSING_REGEN_INPUT and exits 2.
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
ENTROPY_NPZ = ROOT / "pathway11_h100/layerlens_entropy/all_layer_entropy.npz"
OUT_JSON = ROOT / "pathway11_h100/layerlens_entropy/results.json"

N_LAYERS = 28
DOM_PEAK_LAYER = 19
# late-mid block per F-2 weakening rule
LATE_MID_BLOCK = (18, 24)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _load_layer_entropies() -> tuple[np.ndarray, np.ndarray]:
    """Return (H_cond, H_uncond), each (500, N_LAYERS) float64.

    Accepts either an explicit pair of entropy matrices (H_cond / H_uncond) or a
    precomputed reduction matrix (I_layer); in the latter case H_uncond is taken
    as zero so that I_ℓ = H_cond branch yields the stored reduction directly.
    """
    blob = np.load(ENTROPY_NPZ)
    keys = set(blob.files)
    if {"H_cond", "H_uncond"} <= keys:
        h_cond = np.asarray(blob["H_cond"], dtype=np.float64)
        h_uncond = np.asarray(blob["H_uncond"], dtype=np.float64)
    elif "I_layer" in keys:
        # already reduced: treat I directly (H_uncond - H_cond == I_layer)
        i_layer = np.asarray(blob["I_layer"], dtype=np.float64)
        h_uncond = i_layer
        h_cond = np.zeros_like(i_layer)
    else:
        raise KeyError(
            f"entropy cache {ENTROPY_NPZ} lacks (H_cond,H_uncond) or I_layer; "
            f"found {sorted(keys)}"
        )
    return h_cond, h_uncond


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not ENTROPY_NPZ.exists():
        print("MISSING_REGEN_INPUT", ENTROPY_NPZ, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    assert correct.shape == (500,), correct.shape

    try:
        h_cond, h_uncond = _load_layer_entropies()
    except KeyError as exc:
        print("MISSING_REGEN_INPUT", str(exc), file=sys.stderr)
        return 2

    if h_cond.shape != (500, N_LAYERS) or h_uncond.shape != (500, N_LAYERS):
        print(
            "MISSING_REGEN_INPUT",
            f"expected (500,{N_LAYERS}) entropy matrices, got "
            f"{h_cond.shape} / {h_uncond.shape}",
            file=sys.stderr,
        )
        return 2

    # I_ℓ(x→y) = H_ℓ(y|∅) - H_ℓ(y|x). Higher reduction => context sharpened the
    # next-token distribution; orientation vs correctness resolved by AUROC.
    i_layer = h_uncond - h_cond  # (500, N_LAYERS)

    per_layer = []
    aurocs = np.full(N_LAYERS, np.nan, dtype=np.float64)
    for ell in range(N_LAYERS):
        s = i_layer[:, ell]
        a = auroc(s, correct)
        # use the better orientation so argmax compares informativeness, not sign
        a_oriented = max(a, 1.0 - a) if np.isfinite(a) else a
        aurocs[ell] = a_oriented
        per_layer.append(
            {
                "layer": ell,
                "auroc_raw": float(a),
                "auroc_oriented": float(a_oriented),
                "mean_I": float(np.mean(s)),
            }
        )

    finite = np.isfinite(aurocs)
    if not finite.any():
        print("MISSING_REGEN_INPUT", "no finite AUROC across layers", file=sys.stderr)
        return 2

    argmax_layer = int(np.nanargmax(aurocs))
    peak_auroc = float(aurocs[argmax_layer])
    l19_auroc = float(aurocs[DOM_PEAK_LAYER])

    agrees_with_dom = argmax_layer == DOM_PEAK_LAYER
    in_late_mid_block = LATE_MID_BLOCK[0] <= argmax_layer <= LATE_MID_BLOCK[1]

    if agrees_with_dom:
        verdict = (
            "AGREES: probe-free entropy-reduction peak coincides with L19; "
            "F-2 L19 specificity is robust to probe method."
        )
    elif in_late_mid_block:
        verdict = (
            f"WEAKEN: peak at L{argmax_layer} differs from L19 but lies in the "
            f"late-mid block {LATE_MID_BLOCK}; F-2 claim should soften from "
            "'L19 is special' to 'late-mid block (L18-L24) is special'."
        )
    else:
        verdict = (
            f"DISAGREE: probe-free peak at L{argmax_layer} lies outside L19 and "
            f"the late-mid block {LATE_MID_BLOCK}; F-2 L19 specificity is "
            "probe-method-dependent."
        )

    out = {
        "experiment": "P11-FE754",
        "n_layers": N_LAYERS,
        "n_examples": int(correct.shape[0]),
        "dom_peak_layer": DOM_PEAK_LAYER,
        "argmax_layer": argmax_layer,
        "peak_auroc_oriented": peak_auroc,
        "l19_auroc_oriented": l19_auroc,
        "peak_minus_l19": peak_auroc - l19_auroc,
        "agrees_with_dom_l19": bool(agrees_with_dom),
        "argmax_in_late_mid_block": bool(in_late_mid_block),
        "late_mid_block": list(LATE_MID_BLOCK),
        "verdict": verdict,
        "per_layer": per_layer,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(
        f"argmax L{argmax_layer} AUROC={peak_auroc:.4f} | "
        f"L19 AUROC={l19_auroc:.4f} | {verdict.split(':')[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())