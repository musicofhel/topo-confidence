"""FE899 — Principal subspace angles between correct and incorrect groups.

Splits L19 prefill activations by correctness, PCA-reduces each group,
computes principal angles via scipy.linalg.subspace_angles.

Output: pathway11_h100/results/fe899_subspace_angles.json
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
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe899_subspace_angles.json"

K_VALUES = [5, 10, 20, 50]
SEED = 9999


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)

    X_corr = X[correct]
    X_incorr = X[~correct]
    X_corr = X_corr - X_corr.mean(axis=0)
    X_incorr = X_incorr - X_incorr.mean(axis=0)

    n_corr, n_incorr = X_corr.shape[0], X_incorr.shape[0]

    results_per_k = {}
    for k in K_VALUES:
        pca_c = PCA(n_components=k, random_state=SEED).fit(X_corr)
        pca_i = PCA(n_components=k, random_state=SEED).fit(X_incorr)

        # subspace_angles expects (ambient_dim, k) — PCA components are (k, ambient_dim)
        V_c = pca_c.components_.T  # (1536, k)
        V_i = pca_i.components_.T  # (1536, k)

        angles_rad = subspace_angles(V_c, V_i)  # sorted descending
        angles_deg = np.degrees(angles_rad)

        grassmann_dist = float(np.sqrt((angles_rad ** 2).sum()))
        mean_angle = float(angles_deg.mean())
        max_angle = float(angles_deg.max())
        min_angle = float(angles_deg.min())
        frac_below_10 = float((angles_deg < 10).mean())
        frac_below_5 = float((angles_deg < 5).mean())

        var_c = float(pca_c.explained_variance_ratio_.sum())
        var_i = float(pca_i.explained_variance_ratio_.sum())

        results_per_k[str(k)] = {
            "k": k,
            "angles_degrees": angles_deg.tolist(),
            "mean_angle_deg": mean_angle,
            "max_angle_deg": max_angle,
            "min_angle_deg": min_angle,
            "grassmann_distance": grassmann_dist,
            "fraction_below_10deg": frac_below_10,
            "fraction_below_5deg": frac_below_5,
            "variance_explained_correct": var_c,
            "variance_explained_incorrect": var_i,
        }

    out = {
        "experiment": "FE899",
        "description": "Principal subspace angles between correct and incorrect groups",
        "n_correct": n_corr,
        "n_incorrect": n_incorr,
        "centered_per_class": True,
        "results_per_k": results_per_k,
        "interpretation": (
            "All angles near 0 => same subspace (DoM is within-class variation); "
            "large angles => different geometry between classes"
        ),
        "meta": {
            "cache": str(CACHE),
            "seed": SEED,
            "method": "PCA_per_class_then_scipy_subspace_angles",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    for k in K_VALUES:
        r = results_per_k[str(k)]
        print(f"k={k}: mean_angle={r['mean_angle_deg']:.2f}° "
              f"max={r['max_angle_deg']:.2f}° "
              f"Grassmann={r['grassmann_distance']:.4f} "
              f"<10°={r['fraction_below_10deg']:.2%}")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
