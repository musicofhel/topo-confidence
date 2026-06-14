"""P11-FE948 — Weight-spectral origin of the prefill DoM/PC1 direction.

Projects the cached L19 prefill activations (500x1536) onto the top-k singular
vectors of L19's attention-out (o_proj) weight matrix and measures what
fraction of activation variance each weight singular vector captures, against
the PCA eigenspectrum of the same activations.

Theory under test (paper): activation geometry is dominated by weight spectral
structure. If the weight-SVD basis explains >90% of activation variance at
k=1-2, the supervised DoM direction (cos(DoM,PC1)=0.9216 per FE291) is the
residual-stream image of the weight's dominant singular structure. We also
report the cosine of the top weight singular vector against the DoM direction
and PC1, and the AUROC of the top-weight-SV projection vs correctness.

Requires a cached, CPU-side copy of the L19 o_proj weight (no torch / no GPU):
  pathway11_h100/weight_spectral/l19_attn_out_weight.npz  with key "W"
If absent, prints MISSING_REGEN_INPUT and returns 2.
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
WEIGHT_NPZ = ROOT / "pathway11_h100/weight_spectral/l19_attn_out_weight.npz"
OUT_JSON = ROOT / "pathway11_h100/weight_svd_projection/results.json"

TOPK = 20


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(abs(a @ b) / (na * nb))


def main() -> int:
    for required in (CACHE, WEIGHT_NPZ):
        if not required.exists():
            print("MISSING_REGEN_INPUT", required, file=sys.stderr)
            return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n, d = X.shape

    wblob = np.load(WEIGHT_NPZ)
    W = wblob["W"].astype(np.float64)
    if W.ndim != 2:
        print("MISSING_REGEN_INPUT", "W not 2-D", file=sys.stderr)
        return 2

    # SVD of the weight: W = U @ diag(S) @ Vt. Right singular vectors are rows
    # of Vt (input space, length W.shape[1]); left singular vectors are columns
    # of U (output/residual space, length W.shape[0]). Pick whichever basis
    # lives in the d=1536 activation space so the projection is well-defined.
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    if Vt.shape[1] == d:
        basis = Vt                      # right singular vectors (as specified)
        basis_kind = "right_singular_vectors"
    elif U.shape[0] == d:
        basis = U.T                     # left singular vectors fallback
        basis_kind = "left_singular_vectors"
    else:
        print("MISSING_REGEN_INPUT", f"W shape {W.shape} incompatible with d={d}",
              file=sys.stderr)
        return 2

    # Center activations; total variance = trace of covariance.
    mu = X.mean(axis=0)
    Xc = X - mu
    denom = max(n - 1, 1)
    total_var = float((Xc * Xc).sum() / denom)

    # Variance of activation captured by each weight singular vector (orthonormal).
    proj = Xc @ basis.T                                   # (n, r)
    per_dir_var = (proj * proj).sum(axis=0) / denom       # (r,)
    order = np.argsort(per_dir_var)[::-1]                 # by captured variance
    sorted_var = per_dir_var[order]
    weight_cumvar = np.cumsum(sorted_var) / total_var

    # In native singular-value order (k=1 is the dominant SV of the weight).
    weight_cumvar_svorder = np.cumsum(per_dir_var) / total_var

    # PCA eigenspectrum of the same activations.
    cov = (Xc.T @ Xc) / denom
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    pca_cumvar = np.cumsum(eigvals) / max(float(eigvals.sum()), 1e-12)

    # Supervised DoM direction and PC1.
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    _, _, pca_vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = pca_vt[0]
    top_weight_sv = basis[0]            # singular vector for largest weight SV

    top_proj = Xc @ top_weight_sv

    out = {
        "experiment": "P11-FE948",
        "basis_kind": basis_kind,
        "weight_shape": list(W.shape),
        "n_samples": int(n),
        "activation_dim": int(d),
        "top_singular_values": [float(v) for v in S[:TOPK]],
        "spectral_gap_sv1_sv2": float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float("nan"),
        # Cumulative variance captured, weight SVs ranked by captured variance.
        "weight_svd_cumvar_frac_top20": [float(v) for v in weight_cumvar[:TOPK]],
        # Cumulative variance in native singular-value order (k=1 = dominant SV).
        "weight_svd_cumvar_frac_svorder_top20": [float(v) for v in weight_cumvar_svorder[:TOPK]],
        "pca_cumvar_frac_top20": [float(v) for v in pca_cumvar[:TOPK]],
        "weight_frac_k1": float(weight_cumvar[0]),
        "weight_frac_k2": float(weight_cumvar[1]) if len(weight_cumvar) > 1 else float("nan"),
        "weight_frac_k1_svorder": float(weight_cumvar_svorder[0]),
        "weight_frac_k2_svorder": float(weight_cumvar_svorder[1]) if len(weight_cumvar_svorder) > 1 else float("nan"),
        "pca_frac_k1": float(pca_cumvar[0]),
        "pca_frac_k2": float(pca_cumvar[1]) if len(pca_cumvar) > 1 else float("nan"),
        "meets_90pct_at_k1": bool(weight_cumvar[0] >= 0.90),
        "meets_90pct_at_k2": bool(len(weight_cumvar) > 1 and weight_cumvar[1] >= 0.90),
        "cos_dom_top_weight_sv": cosine(dom, top_weight_sv),
        "cos_pc1_top_weight_sv": cosine(pc1, top_weight_sv),
        "cos_dom_pc1": cosine(dom, pc1),
        "auroc_top_weight_sv_proj": auroc(top_proj, y),
    }

    # Optional: tie to the canonical DoM score if the cache is present.
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        out["auroc_dom_score"] = auroc(dom_score, y)
        out["cos_topproj_domscore_rank"] = float(
            np.corrcoef(top_proj, dom_score)[0, 1]
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())