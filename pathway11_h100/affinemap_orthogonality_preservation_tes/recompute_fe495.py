"""P11-FE495 — Affine-map orthogonality preservation test for F-3.

F-3 reports cos(prefill_DoM, final_DoM) = 0.046 in 1.5B-native space. This is
currently a 1.5B-specific number. LRT predicts feature-space angles are
preserved if features live in a universal V_U. This experiment applies the
affine map A (with bias p) trained in P11-FE494 to BOTH the 1.5B prefill DoM
and the 1.5B final-token DoM, then compares three cosines:

  (a) 1.5B native    cos(prefill_DoM_15b,      final_DoM_15b)        ~ 0.046
  (b) affine-proj    cos(A·prefill_DoM_15b+p,  A·final_DoM_15b+p)
  (c) 7B native      cos(prefill_DoM_7b,       final_DoM_7b)         (L21)

Interpretation:
  - 7B native ~ 0.046          -> F-3 generalises (universal property).
  - affine ~ 0.046 but 7B big  -> affine preserves a 1.5B artefact, not a law.
  - neither ~ 0.046            -> F-3 is purely 1.5B-specific.
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

# --- 1.5B caches (prefill from main cache, final from final-token cache) ---
CACHE_15B_PREFILL = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
CACHE_15B_FINAL = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"

# --- 7B native caches at L21 (prefill + final-token) ---
CACHE_7B_PREFILL = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
CACHE_7B_FINAL = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_final.npz"

# --- Affine map trained in P11-FE494 (A and bias p) ---
AFFINE_NPZ = ROOT / "pathway11_h100/affine_alignment/fe494_affine_map.npz"

OUT_JSON = ROOT / "pathway11_h100/affine_orthogonality/results.json"

F3_NATIVE_15B = 0.046  # FINDINGS F-3, scratch/pathway10_temporal_and_verifier_results.json
TOL = 0.03  # |cos - 0.046| < TOL counts as "approximately reproduces F-3"

PREFILL_KEYS = ("prefill", "prefill_hidden", "hidden", "states")
FINAL_KEYS = ("final", "final_token", "final_hidden", "hidden", "states")
LABEL_KEYS = ("correct", "labels", "y")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _pick(blob, candidates):
    for k in candidates:
        if k in blob.files:
            return blob[k]
    return None


def dom_direction(npz_path: Path, hidden_candidates) -> np.ndarray | None:
    """DoM = mean(correct) - mean(incorrect) of the hidden states in npz_path."""
    if not npz_path.exists():
        print("MISSING_REGEN_INPUT", npz_path, file=sys.stderr)
        return None
    blob = np.load(npz_path)
    X = _pick(blob, hidden_candidates)
    y = _pick(blob, LABEL_KEYS)
    if X is None or y is None:
        print("MISSING_REGEN_INPUT", npz_path, "(no hidden/label key)", file=sys.stderr)
        return None
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        print("MISSING_REGEN_INPUT", npz_path, "(shape mismatch)", file=sys.stderr)
        return None
    if y.sum() == 0 or (~y).sum() == 0:
        print("MISSING_REGEN_INPUT", npz_path, "(degenerate labels)", file=sys.stderr)
        return None
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def apply_affine(A: np.ndarray, p: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Compute A @ v + p, tolerating A stored in either orientation."""
    if A.shape[1] == v.shape[0]:
        out = A @ v
    elif A.shape[0] == v.shape[0]:
        out = A.T @ v
    else:
        return None
    if p is not None:
        if p.shape[0] != out.shape[0]:
            return None
        out = out + p
    return out


def main() -> int:
    # --- (a) 1.5B native ---
    pre_15b = dom_direction(CACHE_15B_PREFILL, PREFILL_KEYS)
    fin_15b = dom_direction(CACHE_15B_FINAL, FINAL_KEYS)
    if pre_15b is None or fin_15b is None:
        return 2
    cos_native_15b = cosine(pre_15b, fin_15b)

    # --- affine map from P11-FE494 ---
    if not AFFINE_NPZ.exists():
        print("MISSING_REGEN_INPUT", AFFINE_NPZ, file=sys.stderr)
        return 2
    ablob = np.load(AFFINE_NPZ)
    A = _pick(ablob, ("A", "W", "map", "affine"))
    p = _pick(ablob, ("p", "b", "bias", "offset"))
    if A is None:
        print("MISSING_REGEN_INPUT", AFFINE_NPZ, "(no A key)", file=sys.stderr)
        return 2
    A = np.asarray(A, dtype=np.float64)
    p = None if p is None else np.asarray(p, dtype=np.float64).ravel()

    # --- (b) affine-projected ---
    pre_proj = apply_affine(A, p, pre_15b)
    fin_proj = apply_affine(A, p, fin_15b)
    if pre_proj is None or fin_proj is None:
        print("MISSING_REGEN_INPUT", AFFINE_NPZ, "(A/p dim mismatch)", file=sys.stderr)
        return 2
    cos_affine = cosine(pre_proj, fin_proj)

    # --- (c) 7B native at L21 ---
    pre_7b = dom_direction(CACHE_7B_PREFILL, PREFILL_KEYS)
    fin_7b = dom_direction(CACHE_7B_FINAL, FINAL_KEYS)
    if pre_7b is None or fin_7b is None:
        return 2
    cos_7b_native = cosine(pre_7b, fin_7b)

    # --- verdict ---
    affine_matches = abs(cos_affine - F3_NATIVE_15B) < TOL
    sevenb_matches = abs(cos_7b_native - F3_NATIVE_15B) < TOL
    if sevenb_matches:
        verdict = "F3_GENERALISES"
    elif affine_matches and not sevenb_matches:
        verdict = "AFFINE_PRESERVES_15B_ARTEFACT"
    else:
        verdict = "F3_IS_15B_SPECIFIC"

    out = {
        "experiment": "P11-FE495",
        "f3_native_15b_reference": F3_NATIVE_15B,
        "tol": TOL,
        "cos_native_15b": cos_native_15b,
        "cos_affine_projected": cos_affine,
        "cos_7b_native": cos_7b_native,
        "affine_matches_f3": bool(affine_matches),
        "sevenb_matches_f3": bool(sevenb_matches),
        "verdict": verdict,
        "dims": {
            "prefill_dom_15b": int(pre_15b.shape[0]),
            "final_dom_15b": int(fin_15b.shape[0]),
            "affine_A": list(A.shape),
            "affine_p": (None if p is None else int(p.shape[0])),
            "prefill_dom_7b": int(pre_7b.shape[0]),
            "final_dom_7b": int(fin_7b.shape[0]),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())