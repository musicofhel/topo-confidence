"""P11-FE696 — Coordinate-invariance audit of F-2 and F-3 at L19.

Park et al. (2602.15293) argue that probe directions are dual/covector objects
and that cosines / additions taken in *primal* residual-stream coordinates are
"type errors" for softmax-induced geometries. Two of our headline numbers are
computed in primal coordinates:

  F-3:  cos(prefill_DoM, final_DoM) = 0.046  (orthogonality claim)
  F-2:  prefill L19 DoM AUROC       = 0.7731 (OOF 5-fold)

This script re-derives both under the natural geometry to test whether they are
coordinate-conditional:

  (i)  cos(prefill_DoM, final_DoM) under three inner products
         - Euclidean (current ~0.046)
         - Fisher metric, logit-lens approximation evaluated at the mean
           activation:  <a,b>_G = Cov_p(U a, U b)  with p = softmax(U h0)
         - dual-coordinate cosine: map each residual h to its dual
           phi(h) = softmax(U h) @ U, take the correct-minus-incorrect
           mean-difference in dual space, then cosine.
  (ii) prefill DoM AUROC for K=1 correctness in primal vs dual coordinates.

Reports the deltas so we can see whether F-2/F-3 are geometry-invariant
(strengthens the findings) or collapse under the softmax geometry (forces a
reframing). Inputs: cached pathway11_h100 NPZs + Qwen-2.5-1.5B unembed weights.
CPU-only, no network, no GPU.
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/coord_invariance/results.json"

# Final-token L19 hidden states (needed for final_DoM). Several plausible cache
# locations are probed; first existing wins.
FINAL_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltok.npz",
    ROOT / "pathway11_h100/final_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/data/m15b_final.npz",
]

# Qwen-2.5-1.5B unembedding (lm_head) weights, saved as an NPZ on a prior run.
UNEMBED_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/unembed_m15b.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/lm_head_m15b.npz",
    ROOT / "pathway11_h100/data/qwen15b_unembed.npz",
    ROOT / "pathway11_h100/data/m15b_unembed.npz",
]

HID = 1536
N_FOLDS = 5
SEED = 9999
BATCH = 25  # rows per dual-projection batch (keeps the b x V logit block small)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def first_existing(paths) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_states(path: Path) -> np.ndarray | None:
    """Pull the (500, 1536) hidden-state array out of an NPZ, key-agnostic."""
    blob = np.load(path)
    for k in ("final", "prefill", "hidden", "states", "h", "arr_0"):
        if k in blob.files:
            a = np.asarray(blob[k])
            if a.ndim == 2 and a.shape[1] == HID:
                return a.astype(np.float64)
    for k in blob.files:
        a = np.asarray(blob[k])
        if a.ndim == 2 and a.shape[1] == HID:
            return a.astype(np.float64)
    return None


def load_unembed(path: Path) -> np.ndarray | None:
    """Return unembedding as (V, HID) float32 regardless of stored orientation."""
    blob = np.load(path)
    cand = None
    for k in blob.files:
        a = np.asarray(blob[k])
        if a.ndim != 2:
            continue
        if a.shape[1] == HID and a.shape[0] > HID:
            cand = a
            break
        if a.shape[0] == HID and a.shape[1] > HID:
            cand = a.T
            break
    if cand is None:
        return None
    return np.ascontiguousarray(cand.astype(np.float32))


def softmax_rows(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z


def dual_project(H: np.ndarray, U: np.ndarray) -> np.ndarray:
    """phi(h) = softmax(U h) @ U for each row of H. Returns (n, HID)."""
    n = H.shape[0]
    Phi = np.empty((n, HID), dtype=np.float64)
    Hf = H.astype(np.float32)
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        Z = Hf[s:e] @ U.T            # (b, V) float32
        P = softmax_rows(Z.astype(np.float64))
        Phi[s:e] = P @ U             # (b, HID)
    return Phi


def fisher_cos(a: np.ndarray, b: np.ndarray, U: np.ndarray, p: np.ndarray) -> float:
    """Cosine under the logit-lens Fisher metric at base distribution p:
    <a,b>_G = Cov_p(U a, U b)."""
    za = (U @ a.astype(np.float32)).astype(np.float64)
    zb = (U @ b.astype(np.float32)).astype(np.float64)
    ma = float(p @ za)
    mb = float(p @ zb)
    ab = float((p * za) @ zb) - ma * mb
    aa = float((p * za) @ za) - ma * ma
    bb = float((p * zb) @ zb) - mb * mb
    if aa <= 0 or bb <= 0:
        return float("nan")
    return ab / np.sqrt(aa * bb)


def euclid_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def oof_dom_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """OOF 5-fold DoM-projection AUROC in whatever coordinates X is given."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        d = dom(X[tr], y[tr])
        scores[te] = X[te] @ d
    return auroc(scores, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    final_path = first_existing(FINAL_CANDIDATES)
    if final_path is None:
        print("MISSING_REGEN_INPUT", "m15b_final.npz (final-token L19 states)",
              file=sys.stderr)
        return 2

    unembed_path = first_existing(UNEMBED_CANDIDATES)
    if unembed_path is None:
        print("MISSING_REGEN_INPUT", "unembed_m15b.npz (Qwen-2.5-1.5B lm_head)",
              file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    Xpre = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if Xpre.shape != (500, HID) or y.shape != (500,):
        print("MISSING_REGEN_INPUT", "unexpected prefill cache shape", file=sys.stderr)
        return 2

    Xfin = load_states(final_path)
    if Xfin is None or Xfin.shape[0] != Xpre.shape[0]:
        print("MISSING_REGEN_INPUT", f"final states unreadable in {final_path}",
              file=sys.stderr)
        return 2

    U = load_unembed(unembed_path)
    if U is None:
        print("MISSING_REGEN_INPUT", f"unembed unreadable in {unembed_path}",
              file=sys.stderr)
        return 2
    vocab = int(U.shape[0])

    # ---- Primal DoM directions (full-data, descriptive) ----
    d_pre = dom(Xpre, y)
    d_fin = dom(Xfin, y)

    # ---- (i) cos(prefill_DoM, final_DoM) under three inner products ----
    cos_euclid = euclid_cos(d_pre, d_fin)

    # Fisher base point: mean activation (combined prefill+final mean), logit-lens.
    h0 = 0.5 * (Xpre.mean(axis=0) + Xfin.mean(axis=0))
    z0 = (U @ h0.astype(np.float32)).astype(np.float64)
    z0 -= z0.max()
    p0 = np.exp(z0)
    p0 /= p0.sum()
    cos_fisher = fisher_cos(d_pre, d_fin, U, p0)

    # Dual-coordinate cosine: map residuals to phi(h), DoM in dual space.
    Phi_pre = dual_project(Xpre, U)
    Phi_fin = dual_project(Xfin, U)
    d_pre_dual = dom(Phi_pre, y)
    d_fin_dual = dom(Phi_fin, y)
    cos_dual = euclid_cos(d_pre_dual, d_fin_dual)

    # ---- (ii) prefill DoM AUROC, primal vs dual coordinates (OOF 5-fold) ----
    auroc_primal = oof_dom_auroc(Xpre, y)
    auroc_dual = oof_dom_auroc(Phi_pre, y)

    out = {
        "experiment": "P11-FE696",
        "n": int(len(y)),
        "n_correct": int(y.sum()),
        "vocab": vocab,
        "hidden": HID,
        "inputs": {
            "prefill_cache": str(CACHE),
            "final_cache": str(final_path),
            "unembed": str(unembed_path),
        },
        "F3_cos_prefill_final_DoM": {
            "euclidean": cos_euclid,
            "fisher_logitlens": cos_fisher,
            "dual_coord": cos_dual,
            "delta_fisher_minus_euclid": cos_fisher - cos_euclid,
            "delta_dual_minus_euclid": cos_dual - cos_euclid,
            "reference_euclidean": 0.046,
        },
        "F2_prefill_DoM_auroc": {
            "primal_oof": auroc_primal,
            "dual_oof": auroc_dual,
            "delta_dual_minus_primal": auroc_dual - auroc_primal,
            "reference_primal": 0.7731,
        },
        "verdict_note": (
            "If fisher/dual cosines stay near euclidean and dual AUROC stays "
            "near primal, F-2/F-3 are coordinate-invariant (strengthened). "
            "Large deltas indicate coordinate-conditional findings."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())