"""P11-FE1198 — Layer-wise RSA profile for correctness geometry across all 29 layers.

Representational Similarity Analysis (RSA) per layer: build an activation
representational dissimilarity matrix (RDM = 1 - Pearson corr between the 500
sample patterns) for each layer's hidden states, and a categorical *model* RDM
encoding correctness (0 if two problems share a correctness label, 1 otherwise).
The RSA score for a layer is the Spearman correlation between the upper-triangle
of the layer RDM and the model RDM.

This directly tests the transient-emergence prediction: if correctness-predictive
geometry peaks at intermediate layers (~10-14) and is *attenuated* by L19, that
refutes the geometric interpretation of F-2 (the L19 prefill DoM direction). If
RSA instead peaks at/near L19, the geometric reading is corroborated. A per-layer
in-sample mean-difference (DoM) AUROC is reported alongside as a complementary,
direction-based discriminability profile.
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
from scipy.stats import rankdata

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# All-layer hidden states cache (500, n_layers, 1536). Path/key not fixed across
# regenerations, so probe a few candidates and locate the 3D array by shape.
ALL_LAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_all_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz",
    ROOT / "pathway11_h100/data/all_layers/m15b_all_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_layers.npz",
]
OUT_JSON = ROOT / "pathway11_h100/layerwise_rsa/results.json"

L19_INDEX = 19
INTERMEDIATE_WINDOW = (10, 14)  # inclusive layer-index window for "transient" peak


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson on ranks. Inputs are 1-D vectors."""
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra @ ra) * (rb @ rb)))
    if denom < 1e-12:
        return float("nan")
    return float((ra @ rb) / denom)


def load_all_layers():
    """Return (array (500, n_layers, dim), path) or (None, None) if absent."""
    for p in ALL_LAYER_CANDIDATES:
        if not p.exists():
            continue
        blob = np.load(p)
        for k in blob.files:
            arr = blob[k]
            if arr.ndim == 3 and 500 in arr.shape:
                # orient to (500, n_layers, dim)
                if arr.shape[0] == 500:
                    return arr.astype(np.float64), p
                if arr.shape[1] == 500:
                    return np.transpose(arr, (1, 0, 2)).astype(np.float64), p
    return None, None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    hidden, used_path = load_all_layers()
    if hidden is None:
        print("MISSING_REGEN_INPUT", "all-layer hidden-state cache "
              f"(tried: {[str(p) for p in ALL_LAYER_CANDIDATES]})", file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    assert correct.shape == (500,), correct.shape
    n_samples, n_layers, dim = hidden.shape
    assert n_samples == 500, hidden.shape

    # Categorical model RDM: 0 if same correctness label, else 1. Upper triangle.
    iu = np.triu_indices(n_samples, k=1)
    same = (correct[:, None] == correct[None, :]).astype(np.float64)
    model_rdm = (1.0 - same)[iu]

    rsa_per_layer = []
    auroc_insample = []
    for l in range(n_layers):
        X = hidden[:, l, :]
        # Activation RDM = 1 - Pearson correlation between the 500 sample patterns.
        corr = np.corrcoef(X)
        corr = np.nan_to_num(corr, nan=0.0)
        rdm = (1.0 - corr)[iu]
        rsa_per_layer.append(spearman(rdm, model_rdm))

        # Complementary direction-based discriminability (in-sample DoM AUROC).
        dom = X[correct].mean(axis=0) - X[~correct].mean(axis=0)
        auroc_insample.append(auroc(X @ dom, correct))

    rsa = np.asarray(rsa_per_layer, dtype=np.float64)
    peak_layer = int(np.nanargmax(rsa))
    l19_rsa = float(rsa[L19_INDEX]) if n_layers > L19_INDEX else float("nan")
    lo, hi = INTERMEDIATE_WINDOW
    peak_in_window = bool(lo <= peak_layer <= hi)
    l19_is_peak = bool(peak_layer == L19_INDEX)
    # "Attenuated by L19" = L19 RSA notably below the peak.
    l19_attenuated = bool(
        not np.isnan(l19_rsa) and (float(np.nanmax(rsa)) - l19_rsa) > 0.02
    )

    if peak_in_window and l19_attenuated:
        interpretation = ("transient-emergence: RSA peaks intermediate and is "
                          "attenuated by L19 — refutes geometric reading of F-2")
    elif l19_is_peak or (abs(peak_layer - L19_INDEX) <= 2 and not l19_attenuated):
        interpretation = ("RSA peaks at/near L19 — corroborates geometric reading "
                          "of F-2")
    else:
        interpretation = "mixed/inconclusive layer-wise RSA profile"

    out = {
        "experiment": "P11-FE1198",
        "cache_used": str(used_path),
        "n_layers": int(n_layers),
        "dim": int(dim),
        "rsa_per_layer": [float(v) for v in rsa_per_layer],
        "auroc_insample_per_layer": [float(v) for v in auroc_insample],
        "peak_rsa_layer": peak_layer,
        "peak_rsa": float(np.nanmax(rsa)),
        "l19_index": L19_INDEX,
        "l19_rsa": l19_rsa,
        "intermediate_window": list(INTERMEDIATE_WINDOW),
        "peak_in_intermediate_window": peak_in_window,
        "l19_is_peak": l19_is_peak,
        "l19_attenuated_vs_peak": l19_attenuated,
        "interpretation": interpretation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())