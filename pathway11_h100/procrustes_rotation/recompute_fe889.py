"""FE889 — Procrustes rotation-axis projection (base vs instruct o_proj).

Computes Procrustes alignment R between base and instruct L19 o_proj
weights, extracts rotation planes via matrix logarithm, projects DoM
onto top rotation planes.

Output: pathway11_h100/results/fe889_procrustes_rotation.json
"""
from __future__ import annotations

import json
import sys
import os
import glob
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from scipy.linalg import logm
from safetensors import safe_open
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe889_procrustes_rotation.json"

INSTRUCT_PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"
BASE_GLOB = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*/model.safetensors")

SEED = 9999
N_FOLDS = 5

WEIGHT_KEYS = {
    "o_proj": "model.layers.19.self_attn.o_proj.weight",
    "down_proj": "model.layers.19.mlp.down_proj.weight",
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def extract_rotation_planes(Q: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Extract rotation angles and 2D planes from skew-symmetric Q = logm(R).

    Returns (angles sorted descending, list of (e1, e2) orthonormal plane basis pairs).
    """
    eigvals, eigvecs = np.linalg.eig(Q)

    # Find conjugate pairs (±iθ)
    used = set()
    angles = []
    planes = []

    for j in range(len(eigvals)):
        if j in used:
            continue
        lam = eigvals[j]
        if abs(lam.imag) < 1e-10:
            continue
        # Find conjugate
        for k in range(j + 1, len(eigvals)):
            if k in used:
                continue
            if abs(eigvals[k] - lam.conjugate()) < 1e-8 * (abs(lam) + 1):
                used.add(j)
                used.add(k)
                theta = abs(lam.imag)
                angles.append(theta)
                v = eigvecs[:, j]
                e1 = np.real(v)
                e1 = e1 / (np.linalg.norm(e1) + 1e-30)
                e2 = np.imag(v)
                e2 = e2 - (e2 @ e1) * e1  # Gram-Schmidt
                e2 = e2 / (np.linalg.norm(e2) + 1e-30)
                planes.append((e1, e2))
                break

    order = np.argsort(angles)[::-1]
    angles = np.array(angles)[order]
    planes = [planes[i] for i in order]
    return angles, planes


def analyze_procrustes(W_base: np.ndarray, W_instruct: np.ndarray, name: str,
                       X: np.ndarray, dom_norm: np.ndarray, correct: np.ndarray) -> dict:
    """Procrustes analysis for a pair of weight matrices."""
    # For non-square matrices (down_proj is 1536×8960), compute in residual space
    # Product W_instruct @ W_base.T is (1536, 1536)
    M = W_instruct @ W_base.T
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    det_R = float(np.linalg.det(R))
    if det_R < 0:
        U[:, -1] *= -1
        R = U @ Vt
        det_R = float(np.linalg.det(R))

    print(f"  {name}: det(R) = {det_R:.6f}, computing logm...")
    Q = logm(R)
    # Q should be real skew-symmetric; take real part
    Q = np.real(Q)
    skew_err = float(np.linalg.norm(Q + Q.T) / (np.linalg.norm(Q) + 1e-30))

    angles, planes = extract_rotation_planes(Q)
    n_planes = len(angles)

    # Project DoM onto rotation planes
    k_values = [1, 5, 10, 50]
    dom_cum_var = {}
    for k in k_values:
        k_eff = min(k, n_planes)
        var = 0.0
        for i in range(k_eff):
            e1, e2 = planes[i]
            var += (dom_norm @ e1) ** 2 + (dom_norm @ e2) ** 2
        dom_cum_var[str(k)] = float(var)

    # Use rotation-plane projections as features for AUROC
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    auroc_per_k = {}
    for k in k_values:
        k_eff = min(k, n_planes)
        if k_eff == 0:
            auroc_per_k[str(k)] = float("nan")
            continue
        feats = np.zeros((X.shape[0], 2 * k_eff), dtype=np.float32)
        for i in range(k_eff):
            e1, e2 = planes[i]
            feats[:, 2 * i] = X @ e1
            feats[:, 2 * i + 1] = X @ e2
        oof_scores = np.zeros(X.shape[0], dtype=np.float64)
        for train_idx, test_idx in skf.split(X, correct):
            lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
            lr.fit(feats[train_idx], correct[train_idx])
            oof_scores[test_idx] = lr.predict_proba(feats[test_idx])[:, 1]
        auroc_per_k[str(k)] = float(auroc(oof_scores, correct))

    return {
        "name": name,
        "det_R": det_R,
        "skew_symmetry_error": skew_err,
        "n_rotation_planes": n_planes,
        "top10_rotation_angles_rad": angles[:10].tolist() if n_planes > 0 else [],
        "top10_rotation_angles_deg": np.degrees(angles[:10]).tolist() if n_planes > 0 else [],
        "dom_cumulative_variance_in_top_k_planes": dom_cum_var,
        "auroc_top_k_planes": auroc_per_k,
        "dom_auroc_reference": 0.7731,
    }


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    X = X - X.mean(axis=0)

    dom = X[correct].mean(0) - X[~correct].mean(0)
    dom_norm = dom / (np.linalg.norm(dom) + 1e-30)

    base_paths = sorted(glob.glob(BASE_GLOB))
    if not base_paths:
        print("ERROR: No base model safetensors found", file=sys.stderr)
        return 1
    base_path = base_paths[0]

    f_instruct = safe_open(str(INSTRUCT_PATH), framework="pt")
    f_base = safe_open(str(base_path), framework="pt")

    import torch

    results = {}
    for name, key in WEIGHT_KEYS.items():
        print(f"Loading {name}...")
        W_instruct = f_instruct.get_tensor(key).float().numpy()
        W_base = f_base.get_tensor(key).float().numpy()
        results[name] = analyze_procrustes(W_base, W_instruct, name, X, dom_norm, correct)

    out = {
        "experiment": "FE889",
        "description": "Procrustes rotation-axis projection (base vs instruct L19 weights)",
        "n": X.shape[0],
        "hidden_dim": X.shape[1],
        "centered": True,
        "weight_analyses": results,
        "interpretation": (
            "If DoM variance concentrated in top-k rotation planes with small k, "
            "DoM aligns with RLHF/SFT rotation axes — correctness signal is an imprint "
            "of instruction tuning."
        ),
        "meta": {
            "cache": str(CACHE),
            "instruct_safetensors": str(INSTRUCT_PATH),
            "base_safetensors": base_path,
            "seed": SEED,
            "method": "Procrustes_logm_rotation_plane_extraction",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  det(R)={r['det_R']:.6f} skew_err={r['skew_symmetry_error']:.6f}")
        print(f"  n_planes={r['n_rotation_planes']}")
        if r['top10_rotation_angles_deg']:
            print(f"  top5_angles_deg={r['top10_rotation_angles_deg'][:5]}")
        print(f"  DoM var in top planes: {r['dom_cumulative_variance_in_top_k_planes']}")
        print(f"  AUROC from planes: {r['auroc_top_k_planes']}")
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
