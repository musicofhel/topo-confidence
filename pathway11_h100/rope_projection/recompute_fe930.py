"""FE930 — RoPE plane projection via q_proj pullback (per-head decomposition).

Loads q_proj.weight from L19, pulls RoPE rotation planes back to residual
space per head, measures DoM variance in each head's subspace.

Output: pathway11_h100/results/fe930_rope_projection.json
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from safetensors import safe_open

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe930_rope_projection.json"

SAFETENSORS_PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"

NUM_HEADS = 12
HEAD_DIM = 128
HIDDEN_DIM = 1536
PLANES_PER_HEAD = 64


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    X = X - X.mean(axis=0)

    dom = X[correct].mean(0) - X[~correct].mean(0)
    dom_norm = dom / (np.linalg.norm(dom) + 1e-30)
    dom_var_total = float(np.var(X @ dom_norm))

    f = safe_open(str(SAFETENSORS_PATH), framework="pt")
    import torch
    W_q = f.get_tensor("model.layers.19.self_attn.q_proj.weight").float().numpy()
    # W_q shape: (num_heads * head_dim, hidden_dim) = (1536, 1536)

    per_head_results = []
    per_head_dom_var = []

    for h in range(NUM_HEADS):
        W_h = W_q[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]  # (128, 1536)
        # Project dom into this head's Q-space
        dom_in_head = W_h @ dom_norm  # (128,)
        # Variance of X projected through this head, along dom direction
        # Pullback: project each activation into head's 128-dim subspace
        X_head = X @ W_h.T  # (500, 128)
        head_dom_dir = W_h @ dom_norm
        head_dom_dir = head_dom_dir / (np.linalg.norm(head_dom_dir) + 1e-30)
        projections = X_head @ head_dom_dir  # (500,)
        head_var = float(np.var(projections))

        # Per-plane analysis for this head
        plane_vars = []
        for plane_idx in range(PLANES_PER_HEAD):
            i0 = 2 * plane_idx
            i1 = 2 * plane_idx + 1
            # Pullback plane directions to residual space
            e0 = W_h[i0, :]  # (1536,)
            e1 = W_h[i1, :]  # (1536,)
            # Project DoM onto this 2D plane
            d0 = float(dom_norm @ e0)
            d1 = float(dom_norm @ e1)
            plane_var = d0 ** 2 + d1 ** 2
            plane_vars.append(plane_var)

        plane_vars = np.array(plane_vars)
        top_planes = np.argsort(plane_vars)[::-1]

        per_head_results.append({
            "head": h,
            "dom_variance_in_head": head_var,
            "total_dom_projection_sq": float(np.sum(plane_vars)),
            "top5_planes": top_planes[:5].tolist(),
            "top5_plane_vars": plane_vars[top_planes[:5]].tolist(),
        })
        per_head_dom_var.append(head_var)

    per_head_dom_var = np.array(per_head_dom_var)
    head_var_fracs = per_head_dom_var / (per_head_dom_var.sum() + 1e-30)

    # Concentration metrics
    sorted_fracs = np.sort(head_var_fracs)[::-1]
    top1_frac = float(sorted_fracs[0])
    top3_frac = float(sorted_fracs[:3].sum())
    gini = float(np.sum((2 * np.arange(1, NUM_HEADS + 1) - NUM_HEADS - 1) * np.sort(head_var_fracs)) / (NUM_HEADS * head_var_fracs.sum() + 1e-30))

    out = {
        "experiment": "FE930",
        "description": "RoPE plane projection via q_proj pullback (per-head decomposition)",
        "n": X.shape[0],
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "centered": True,
        "dom_variance_total": dom_var_total,
        "per_head_variance_fractions": head_var_fracs.tolist(),
        "top1_head_fraction": top1_frac,
        "top3_head_fraction": top3_frac,
        "gini_coefficient": gini,
        "per_head_details": per_head_results,
        "interpretation": (
            "Concentrated in few heads => positional coupling in specific attention patterns; "
            "uniform => no special positional structure in DoM"
        ),
        "meta": {
            "cache": str(CACHE),
            "safetensors": str(SAFETENSORS_PATH),
            "method": "q_proj_pullback_per_head_RoPE_plane_decomposition",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Per-head DoM variance fractions: {head_var_fracs.round(4).tolist()}")
    print(f"top1={top1_frac:.4f} top3={top3_frac:.4f} gini={gini:.4f}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
