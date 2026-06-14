"""P11-FE318 — Bilinear approximation of L19 SwiGLU + eigendecomposition of the
correctness output direction.

Qwen-2.5-1.5B's L19 MLP is SwiGLU: out(x) = down_proj( SiLU(gate_proj·x) ⊙ (up_proj·x) ).
Drop the SiLU σ and treat the gate as bilinear:  g(x) = (W x) ⊙ (V x), with
W = gate_proj (intermediate×hidden), V = up_proj (intermediate×hidden),
D = down_proj (hidden×intermediate).

The MLP output writes into the residual stream, so the scalar correctness readout
of the probe u_correct (the residual-space pullback of the prefill DoM direction)
is a quadratic form in x:

    s(x) = u_correct · out(x)
         = sum_i u_correct[i] * x^T B_i x          (B_i = sum_k D[i,k] w_k v_k^T)
         = x^T Q x,   Q = sum_k (u_correct · D[:,k]) w_k v_k^T = W^T diag(c) V,
                      c = D^T u_correct.

We symmetrize Q, eigendecompose, and report cos(top_eigenvector, prefill_DoM)
plus the top-10 eigenvalue magnitudes.

Refutation bands (rationale):
  cos > 0.5    -> F-2 direction is a static weight feature (demotes "averaging" framing)
  0.2 <= cos <= 0.5 -> partial alignment, hybrid story
  cos < 0.2    -> bilinear approximation fails; refutation does not fire (negative result)

Requires pre-extracted L19 MLP weights (no torch/transformers in this harness).
Expected at pathway11_h100/bilinear_swiglu/l19_mlp_weights.npz with keys
{gate_proj, up_proj, down_proj} (or {W, V, D}). If absent: MISSING_REGEN_INPUT.
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
WEIGHTS = ROOT / "pathway11_h100/bilinear_swiglu/l19_mlp_weights.npz"
OUT_JSON = ROOT / "pathway11_h100/bilinear_swiglu/results.json"

HIDDEN = 1536


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _pick(blob, names):
    for nm in names:
        if nm in blob.files:
            return blob[nm]
    return None


def _orient(M: np.ndarray, n_int: int) -> np.ndarray:
    """Return weight in (intermediate, hidden) orientation."""
    if M.shape == (n_int, HIDDEN):
        return M
    if M.shape == (HIDDEN, n_int):
        return M.T
    raise ValueError(f"unexpected weight shape {M.shape}")


def main() -> int:
    for p in (CACHE, DOM_NPZ, WEIGHTS):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
            return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, HIDDEN) and y.shape == (500,)

    # Prefill DoM direction (residual space). u_correct is its unit pullback.
    dom_dir = _unit(X[y].mean(axis=0) - X[~y].mean(axis=0))
    # Sanity: stored prefill_score should track X @ dom_dir.
    stored_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    cos_dom_score = float(
        np.dot(_unit(X @ dom_dir), _unit(stored_score))
    )
    u_correct = dom_dir

    # Load L19 MLP weights, drop SiLU -> bilinear gate.
    wb = np.load(WEIGHTS)
    W_raw = _pick(wb, ("gate_proj", "W", "gate_proj.weight"))
    V_raw = _pick(wb, ("up_proj", "V", "up_proj.weight"))
    D_raw = _pick(wb, ("down_proj", "D", "down_proj.weight"))
    if W_raw is None or V_raw is None or D_raw is None:
        print("MISSING_REGEN_INPUT", WEIGHTS, "missing gate/up/down keys", file=sys.stderr)
        return 2

    # Infer intermediate dim from the larger axis of down_proj.
    D_raw = D_raw.astype(np.float64)
    n_int = D_raw.shape[1] if D_raw.shape[0] == HIDDEN else D_raw.shape[0]
    D = D_raw if D_raw.shape == (HIDDEN, n_int) else D_raw.T  # (hidden, intermediate)
    W = _orient(W_raw.astype(np.float64), n_int)  # (intermediate, hidden)
    V = _orient(V_raw.astype(np.float64), n_int)  # (intermediate, hidden)

    # Q = W^T diag(c) V,  c = D^T u_correct.
    c = D.T @ u_correct                       # (intermediate,)
    Q = (W * c[:, None]).T @ V                 # (hidden, hidden)
    Q_sym = 0.5 * (Q + Q.T)

    # Eigendecomposition (symmetric).
    eigvals, eigvecs = np.linalg.eigh(Q_sym)   # ascending
    order = np.argsort(-np.abs(eigvals))       # by magnitude, descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    top_vec = _unit(eigvecs[:, 0])
    cos_top = float(np.dot(top_vec, u_correct))
    abs_cos_top = abs(cos_top)

    # cos of DoM with each of the top-10 eigenvectors (abs).
    cos_top10 = [abs(float(np.dot(_unit(eigvecs[:, i]), u_correct))) for i in range(min(10, eigvecs.shape[1]))]
    top10_mags = [float(abs(eigvals[i])) for i in range(min(10, len(eigvals)))]
    top10_signed = [float(eigvals[i]) for i in range(min(10, len(eigvals)))]

    if abs_cos_top > 0.5:
        verdict = "STATIC_WEIGHT_FEATURE"
    elif abs_cos_top >= 0.2:
        verdict = "PARTIAL_ALIGNMENT_HYBRID"
    else:
        verdict = "BILINEAR_FAILS_NO_REFUTATION"

    out = {
        "experiment": "P11-FE318",
        "intermediate_dim": int(n_int),
        "hidden_dim": HIDDEN,
        "cos_dom_vs_storedscore_sanity": cos_dom_score,
        "cos_top_eigvec_prefill_dom": cos_top,
        "abs_cos_top_eigvec_prefill_dom": abs_cos_top,
        "abs_cos_dom_top10_eigvecs": cos_top10,
        "eigval_top10_magnitudes": top10_mags,
        "eigval_top10_signed": top10_signed,
        "best_abs_cos_within_top10": float(max(cos_top10)) if cos_top10 else float("nan"),
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())