"""FE136 — Park-Choe-Veitch causal inner product whitened cosine.

Apply M = Cov(γ)^{-1} to the L19 prefill and final-token DoM directions;
recompute cos similarity in whitened coordinates and compare against
raw cos = 0.046 (F-3).  If the whitened cos jumps toward 1, LRH+causal
inner product unifies the two directions as one concept; if it stays
near zero, the orthogonality is representationally real.
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
OUT_JSON = ROOT / "pathway11_h100/causal_inner_product/results.json"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def cos_raw(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cos_mahalanobis(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
    """Cosine similarity under inner product <u, v>_M = u^T M v."""
    aMb = float(a @ M @ b)
    aMa = float(a @ M @ a)
    bMb = float(b @ M @ b)
    if aMa <= 0 or bMb <= 0:
        return float("nan")
    return aMb / np.sqrt(aMa * bMb)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X_pre = blob["prefill"].astype(np.float64)
    X_fin = blob["final_tok"].astype(np.float64)
    y = blob["correct"].astype(bool)
    n, d = X_pre.shape

    # --- DoM directions (uncentered, matching pathway10 convention) ---
    dom_pre = X_pre[y].mean(axis=0) - X_pre[~y].mean(axis=0)
    dom_fin = X_fin[y].mean(axis=0) - X_fin[~y].mean(axis=0)

    raw_cos = cos_raw(dom_pre, dom_fin)

    # --- Also compute centered DoM for completeness ---
    dom_pre_c = (X_pre - X_pre.mean(0))[y].mean(0) - (X_pre - X_pre.mean(0))[~y].mean(0)
    dom_fin_c = (X_fin - X_fin.mean(0))[y].mean(0) - (X_fin - X_fin.mean(0))[~y].mean(0)
    raw_cos_centered = cos_raw(dom_pre_c, dom_fin_c)

    # --- Covariance matrices ---
    X_pre_c = X_pre - X_pre.mean(axis=0)
    Cov_pre = X_pre_c.T @ X_pre_c / max(n - 1, 1)

    X_fin_c = X_fin - X_fin.mean(axis=0)
    Cov_fin = X_fin_c.T @ X_fin_c / max(n - 1, 1)

    Cov_pool = (Cov_pre + Cov_fin) / 2.0

    results = {
        "experiment": "FE136",
        "description": "Park-Choe-Veitch causal inner product whitened cosine",
        "n_samples": int(n),
        "d_hidden": int(d),
        "raw_cos_uncentered": raw_cos,
        "raw_cos_centered": raw_cos_centered,
    }

    # --- Whitened cosine under each M = Cov^{-1} ---
    for label, Cov in [("prefill", Cov_pre), ("final", Cov_fin), ("pooled", Cov_pool)]:
        eigvals = np.linalg.eigvalsh(Cov)
        trace = float(np.trace(Cov))
        cond = float(eigvals[-1] / max(eigvals[0], 1e-30))

        for alpha_rel in [1e-2, 1e-3, 1e-4]:
            alpha = alpha_rel * trace / d
            M = np.linalg.inv(Cov + alpha * np.eye(d))

            wcos_unc = cos_mahalanobis(dom_pre, dom_fin, M)
            wcos_cen = cos_mahalanobis(dom_pre_c, dom_fin_c, M)

            tag = f"cov_{label}_alpha{alpha_rel}"
            results[f"whitened_cos_uncentered_{tag}"] = wcos_unc
            results[f"whitened_cos_centered_{tag}"] = wcos_cen

        # Truncated pseudo-inverse (keep top-k eigenvectors by variance)
        eigvals_full, eigvecs = np.linalg.eigh(Cov)
        for keep_frac in [0.99, 0.95, 0.90]:
            cumvar = np.cumsum(eigvals_full[::-1]) / eigvals_full.sum()
            k = int(np.searchsorted(cumvar, keep_frac) + 1)
            k = min(k, d)
            top_vecs = eigvecs[:, -k:]
            top_vals = eigvals_full[-k:]
            M_trunc = top_vecs @ np.diag(1.0 / top_vals) @ top_vecs.T

            wcos_unc = cos_mahalanobis(dom_pre, dom_fin, M_trunc)
            wcos_cen = cos_mahalanobis(dom_pre_c, dom_fin_c, M_trunc)

            tag = f"cov_{label}_trunc{keep_frac}"
            results[f"whitened_cos_uncentered_{tag}"] = wcos_unc
            results[f"whitened_cos_centered_{tag}"] = wcos_cen
            results[f"n_components_{tag}"] = int(k)

        results[f"cov_{label}_condition_number"] = cond
        results[f"cov_{label}_trace"] = trace

    # --- AUROC sanity checks under whitened DoM ---
    alpha = 1e-3 * float(np.trace(Cov_pre)) / d
    M_ref = np.linalg.inv(Cov_pre + alpha * np.eye(d))

    w_pre = M_ref @ dom_pre
    scores_pre_whitened = X_pre @ w_pre
    results["auroc_prefill_dom_raw"] = auroc(X_pre @ dom_pre, y)
    results["auroc_prefill_dom_whitened"] = auroc(scores_pre_whitened, y)

    w_fin = M_ref @ dom_fin
    scores_fin_whitened = X_fin @ w_fin
    results["auroc_final_dom_raw"] = auroc(X_fin @ dom_fin, y)
    results["auroc_final_dom_whitened"] = auroc(scores_fin_whitened, y)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"raw cos (uncentered): {raw_cos:.6f}")
    print(f"raw cos (centered):   {raw_cos_centered:.6f}")
    best_key = "whitened_cos_centered_cov_prefill_alpha0.001"
    print(f"whitened cos (prefill cov, alpha=1e-3, centered): {results[best_key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())