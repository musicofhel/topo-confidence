"""FE903 — Regularized CCA between prefill and final-token L19 activations.

PCA-reduces both to k dims, then sklearn CCA. Reports mean canonical
correlation for k in {20, 50, 100, 200}.

Output: pathway11_h100/results/fe903_mcca_prefill_final.json
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
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe903_mcca_prefill_final.json"

K_VALUES = [20, 50, 100, 200]
N_CCA_COMPONENTS = 20


def main() -> int:
    cache = np.load(CACHE)
    X_pre = cache["prefill"].astype(np.float32)
    X_fin = cache["final_tok"].astype(np.float32)
    correct = cache["correct"]
    n = X_pre.shape[0]

    X_pre = X_pre - X_pre.mean(axis=0)
    X_fin = X_fin - X_fin.mean(axis=0)

    results_per_k = {}
    for k in K_VALUES:
        pca_pre = PCA(n_components=k, random_state=9999).fit(X_pre)
        pca_fin = PCA(n_components=k, random_state=9999).fit(X_fin)
        Z_pre = pca_pre.transform(X_pre)
        Z_fin = pca_fin.transform(X_fin)

        n_cc = min(k, N_CCA_COMPONENTS)
        cca = CCA(n_components=n_cc, max_iter=1000)
        cca.fit(Z_pre, Z_fin)
        U, V = cca.transform(Z_pre, Z_fin)

        canon_corrs = []
        for i in range(n_cc):
            r = float(np.corrcoef(U[:, i], V[:, i])[0, 1])
            canon_corrs.append(r)

        mcca = float(np.mean(canon_corrs))
        var_pre = float(pca_pre.explained_variance_ratio_.sum())
        var_fin = float(pca_fin.explained_variance_ratio_.sum())

        # Per-class mCCA
        mask_c = correct.astype(bool)
        U_c, V_c = U[mask_c], V[mask_c]
        U_i, V_i = U[~mask_c], V[~mask_c]
        corrs_c = [float(np.corrcoef(U_c[:, j], V_c[:, j])[0, 1]) for j in range(n_cc)]
        corrs_i = [float(np.corrcoef(U_i[:, j], V_i[:, j])[0, 1]) for j in range(n_cc)]

        results_per_k[str(k)] = {
            "k": k,
            "n_cca_components": n_cc,
            "canonical_correlations": canon_corrs,
            "mean_canonical_correlation": mcca,
            "pca_variance_explained_prefill": var_pre,
            "pca_variance_explained_final": var_fin,
            "mcca_correct_class": float(np.mean(corrs_c)),
            "mcca_incorrect_class": float(np.mean(corrs_i)),
        }

    # Cosine similarity between prefill and final-token mean directions
    dom_pre = X_pre[correct].mean(0) - X_pre[~correct].mean(0)
    dom_fin = X_fin[correct].mean(0) - X_fin[~correct].mean(0)
    cos_dom = float(dom_pre @ dom_fin / (np.linalg.norm(dom_pre) * np.linalg.norm(dom_fin) + 1e-30))

    out = {
        "experiment": "FE903",
        "description": "Regularized CCA between prefill and final-token L19 activations",
        "n": n,
        "centered": True,
        "results_per_k": results_per_k,
        "cosine_dom_prefill_vs_final": cos_dom,
        "interpretation": (
            "mCCA near 1 => prefill/final share the same linear subspace; "
            "mCCA < 0.5 => genuine structural separation (consistent with F-3 cos=0.046)"
        ),
        "meta": {
            "cache": str(CACHE),
            "seed": 9999,
            "method": "PCA_reduction_then_sklearn_CCA",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    for k in K_VALUES:
        r = results_per_k[str(k)]
        print(f"k={k}: mCCA={r['mean_canonical_correlation']:.4f} "
              f"(var_pre={r['pca_variance_explained_prefill']:.4f}, "
              f"var_fin={r['pca_variance_explained_final']:.4f})")
    print(f"cos(DoM_pre, DoM_fin)={cos_dom:.4f}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
