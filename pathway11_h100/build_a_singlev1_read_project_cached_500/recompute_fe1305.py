"""P11-FE1305 — Single weight-v1 read vs. the spectrum findings.

Tests the paper's single-principal-direction sufficiency claim. We project the
cached 500x1536 L19 prefill activations onto the *leading right singular vector
of W_L19* (computed by power iteration on the cached L19 weight matrix, NOT on
the activation covariance) and score correctness AUROC. We compare this
weight-v1 read against:
  - PC1 of the activations (the data principal direction, ~0.7731),
  - the PC1-residualized cov-spectrum top-20 read (0.7928, FE881),
  - the full-1536d L2-reg ceiling (0.7847, FE882).

If the weight-v1 read ceilings well below the residualized cov-spectrum, the
weight's principal direction is NOT the most informative object for
correctness — reinforcing the off-principal-tail story (Refutation 2).
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
# L19 weight matrix cache. We do not load the model (no torch); we read a
# pre-dumped weight NPZ if one exists. Try a few plausible keys/paths.
W_NPZ_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/w_l19.npz",
    ROOT / "pathway11_h100/data/w_l19.npz",
    ROOT / "pathway11_h100/weight_v1_read/w_l19.npz",
]
W_KEYS = ["W_L19", "W", "weight", "w_l19", "W_l19"]

OUT_JSON = ROOT / "pathway11_h100/weight_v1_read/results.json"

SEED = 9999
PI_ITERS = 500
PI_TOL = 1e-10

# Reference numbers from prior experiments (for side-by-side reporting).
REF_PC1 = 0.7731
REF_COV_SPECTRUM_FE881 = 0.7928
REF_L2_CEILING_FE882 = 0.7847


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oriented_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC with sign ambiguity resolved (a 'read' has no fixed polarity)."""
    a = auroc(scores, labels)
    if np.isnan(a):
        return a
    return max(a, 1.0 - a)


def power_iter_leading_right_sv(W: np.ndarray, iters: int, tol: float,
                                seed: int) -> np.ndarray:
    """Leading right singular vector v1 of W via power iteration on W^T W."""
    rng = np.random.default_rng(seed)
    n_in = W.shape[1]
    v = rng.standard_normal(n_in)
    v /= np.linalg.norm(v)
    prev = v
    for _ in range(iters):
        w = W @ v          # (n_out,)
        v = W.T @ w        # (n_in,)
        nrm = np.linalg.norm(v)
        if nrm < 1e-30:
            break
        v = v / nrm
        if np.linalg.norm(v - prev) < tol or np.linalg.norm(v + prev) < tol:
            prev = v
            break
        prev = v
    return v


def power_iter_pc1(Xc: np.ndarray, iters: int, tol: float,
                   seed: int) -> np.ndarray:
    """Leading eigenvector (PC1) of the centered-data covariance Xc^T Xc."""
    rng = np.random.default_rng(seed)
    d = Xc.shape[1]
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    prev = v
    for _ in range(iters):
        v = Xc.T @ (Xc @ v)
        nrm = np.linalg.norm(v)
        if nrm < 1e-30:
            break
        v = v / nrm
        if np.linalg.norm(v - prev) < tol or np.linalg.norm(v + prev) < tol:
            prev = v
            break
        prev = v
    return v


def load_weight_matrix() -> np.ndarray | None:
    for path in W_NPZ_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for k in W_KEYS:
            if k in blob.files:
                return blob[k].astype(np.float64)
        # fall back to the first 2-D array in the archive
        for k in blob.files:
            arr = blob[k]
            if arr.ndim == 2:
                return arr.astype(np.float64)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    W = load_weight_matrix()
    if W is None:
        print("MISSING_REGEN_INPUT", "w_l19.npz (no weight cache found)",
              file=sys.stderr)
        return 2
    if W.shape[1] != X.shape[1]:
        # right singular vector must live in the activation (input) space
        if W.shape[0] == X.shape[1]:
            W = W.T
        else:
            print("MISSING_REGEN_INPUT",
                  f"W_L19 shape {W.shape} incompatible with activations "
                  f"{X.shape}", file=sys.stderr)
            return 2

    # --- weight-v1 read ---
    v1_w = power_iter_leading_right_sv(W, PI_ITERS, PI_TOL, SEED)
    proj_w = X @ v1_w
    auroc_weight_v1 = oriented_auroc(proj_w, y)

    # --- activation PC1 read (recompute locally for apples-to-apples) ---
    Xc = X - X.mean(axis=0)
    v1_x = power_iter_pc1(Xc, PI_ITERS, PI_TOL, SEED)
    proj_pc1 = Xc @ v1_x
    auroc_pc1_local = oriented_auroc(proj_pc1, y)

    # alignment between the weight principal direction and the data PC1
    cos_w_pc1 = float(abs(v1_w @ v1_x) /
                      (np.linalg.norm(v1_w) * np.linalg.norm(v1_x)))

    # --- DoM reference (single supervised direction) ---
    auroc_dom = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        auroc_dom = oriented_auroc(dom_score, y)

    out = {
        "experiment": "P11-FE1305",
        "n": int(len(y)),
        "auroc_weight_v1_read": auroc_weight_v1,
        "auroc_pc1_local": auroc_pc1_local,
        "auroc_dom_prefill": auroc_dom,
        "cos_weight_v1_vs_pc1": cos_w_pc1,
        "reference": {
            "pc1": REF_PC1,
            "cov_spectrum_pc1resid_fe881": REF_COV_SPECTRUM_FE881,
            "l2_ceiling_fe882": REF_L2_CEILING_FE882,
        },
        "gap_weight_v1_minus_cov_spectrum":
            auroc_weight_v1 - REF_COV_SPECTRUM_FE881,
        "weight_v1_below_cov_spectrum":
            bool(auroc_weight_v1 < REF_COV_SPECTRUM_FE881),
        "interpretation": (
            "If auroc_weight_v1 << cov_spectrum (0.7928), the weight's leading "
            "singular direction is not the most informative object for "
            "correctness — supports the off-principal-tail story (Refutation 2)."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())