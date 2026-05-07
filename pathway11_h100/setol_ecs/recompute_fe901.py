"""FE901 — SETOL Effective Correlation Space (ECS) projection.

Loads L19 o_proj and down_proj weights, computes W @ W^T (residual-space),
eigendecomposes, fits power-law to eigenvalue tail, projects DoM/PC1 onto
ECS (power-law tail eigenvectors).

Output: pathway11_h100/results/fe901_setol_ecs.json
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
from sklearn.decomposition import PCA

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe901_setol_ecs.json"

SAFETENSORS_PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"

WEIGHT_KEYS = {
    "o_proj": "model.layers.19.self_attn.o_proj.weight",
    "down_proj": "model.layers.19.mlp.down_proj.weight",
}


def fit_power_law_tail(eigvals_desc: np.ndarray) -> dict:
    """Fit power-law to eigenvalue spectrum via log-log linear regression.

    Returns alpha, r2, ecs_rank (number of eigenvalues in power-law tail).
    """
    pos = eigvals_desc[eigvals_desc > 0]
    if len(pos) < 10:
        return {"alpha": float("nan"), "r2": 0.0, "ecs_rank": 0}

    log_rank = np.log(np.arange(1, len(pos) + 1))
    log_eig = np.log(pos)

    # Try different tail cutoffs, pick best R^2
    best_alpha, best_r2, best_start = 0.0, -1.0, 0
    for start_frac in [0.1, 0.2, 0.3, 0.5]:
        start_idx = int(len(pos) * start_frac)
        if len(pos) - start_idx < 5:
            continue
        x = log_rank[start_idx:]
        y = log_eig[start_idx:]
        coeffs = np.polyfit(x, y, 1)
        alpha = -coeffs[0]
        y_pred = np.polyval(coeffs, x)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-30)
        if r2 > best_r2:
            best_alpha, best_r2, best_start = alpha, r2, start_idx

    ecs_rank = len(pos) - best_start
    return {"alpha": float(best_alpha), "r2": float(best_r2), "ecs_rank": ecs_rank, "tail_start_idx": best_start}


def analyze_weight_matrix(W: np.ndarray, name: str, dom_norm: np.ndarray, pc1_norm: np.ndarray) -> dict:
    """Compute W @ W^T, eigendecompose, fit PL, project DoM/PC1 onto ECS."""
    # W @ W^T gives residual-space eigenvectors
    WWT = W @ W.T  # (1536, 1536)
    eigvals, eigvecs = np.linalg.eigh(WWT)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pl = fit_power_law_tail(eigvals)

    # ECS = power-law tail eigenvectors
    if pl["ecs_rank"] > 0:
        ecs_vecs = eigvecs[:, pl["tail_start_idx"]:]  # (1536, ecs_rank)
        dom_proj = ecs_vecs.T @ dom_norm  # (ecs_rank,)
        dom_ecs_frac = float((dom_proj ** 2).sum())
        pc1_proj = ecs_vecs.T @ pc1_norm
        pc1_ecs_frac = float((pc1_proj ** 2).sum())
    else:
        dom_ecs_frac = 0.0
        pc1_ecs_frac = 0.0

    # Also report top-k variance fractions
    dom_proj_all = eigvecs.T @ dom_norm
    cum_dom = np.cumsum(dom_proj_all ** 2)
    top_k_dom = {str(k): float(cum_dom[min(k - 1, len(cum_dom) - 1)]) for k in [1, 5, 10, 20, 50]}

    return {
        "name": name,
        "shape": list(W.shape),
        "wwt_shape": list(WWT.shape),
        "power_law_alpha": pl["alpha"],
        "power_law_r2": pl["r2"],
        "ecs_rank": pl["ecs_rank"],
        "dom_ecs_variance_fraction": dom_ecs_frac,
        "pc1_ecs_variance_fraction": pc1_ecs_frac,
        "dom_top_k_variance": top_k_dom,
        "eigenvalue_spectrum_top50": eigvals[:50].tolist(),
        "eigenvalue_spectrum_bottom20": eigvals[-20:].tolist(),
    }


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    X = X - X.mean(axis=0)

    dom = X[correct].mean(0) - X[~correct].mean(0)
    dom_norm = dom / (np.linalg.norm(dom) + 1e-30)

    pca = PCA(n_components=1, random_state=9999).fit(X)
    pc1 = pca.components_[0]
    pc1_norm = pc1 / (np.linalg.norm(pc1) + 1e-30)

    cos_dom_pc1 = float(np.abs(dom_norm @ pc1_norm))

    f = safe_open(str(SAFETENSORS_PATH), framework="pt")
    import torch

    weight_results = {}
    for name, key in WEIGHT_KEYS.items():
        W = f.get_tensor(key).float().numpy()
        print(f"Analyzing {name}: shape={W.shape}")
        weight_results[name] = analyze_weight_matrix(W, name, dom_norm, pc1_norm)

    out = {
        "experiment": "FE901",
        "description": "SETOL ECS projection of DoM and PC1 onto L19 weight eigenspaces",
        "n": X.shape[0],
        "hidden_dim": X.shape[1],
        "centered": True,
        "cos_dom_pc1": cos_dom_pc1,
        "weight_analyses": weight_results,
        "interpretation": (
            "High DoM ECS fraction => DoM aligns with weight matrix's effective correlation structure; "
            "power-law alpha 2-6 typical for well-trained models"
        ),
        "meta": {
            "cache": str(CACHE),
            "safetensors": str(SAFETENSORS_PATH),
            "seed": 9999,
            "method": "W_WTranspose_eigendecomp_powerlaw_tail_projection",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    for name, r in weight_results.items():
        print(f"{name}: alpha={r['power_law_alpha']:.3f} R2={r['power_law_r2']:.3f} "
              f"ECS_rank={r['ecs_rank']} DoM_ECS={r['dom_ecs_variance_fraction']:.4f} "
              f"PC1_ECS={r['pc1_ecs_variance_fraction']:.4f}")
    print(f"cos(DoM, PC1)={cos_dom_pc1:.4f}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
