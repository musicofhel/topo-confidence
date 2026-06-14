"""P11-FE587 — Per-layer matrix-based entropy curve on Qwen-2.5-1.5B MATH-500.

Replicates Queipo-de-Llano 2510.06477 Figure 1 (Mix-Compress-Refine) on our
reasoning task. For each layer ℓ we form the representation matrix X^(ℓ) (N×d)
and compute the matrix-based entropy of its normalized squared singular-value
spectrum:

    p_j = σ_j² / ‖X‖_F²      (note ‖X‖_F² = Σ_k σ_k²)
    H(X^(ℓ)) = -Σ_j p_j log p_j

The curve is computed separately for correct and incorrect K=1 trajectories and
over the pooled set. We then locate the compression valley (argmin of the pooled
curve) and report whether L19 — the layer on which F-1/F-2 were discovered — sits
inside that valley, testing the mid-layer-DoM-peak ↔ compression-valley-bottom
hypothesis.

Requires a per-layer prefill cache (a 3-D array shaped roughly
(n_layers, N, d) with d=1536). If no such cache is present this prints
MISSING_REGEN_INPUT and returns 2 — the single documented L19 cache is not
sufficient to draw the full 28-layer curve.
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
CACHE_DIR = ROOT / "pathway11_h100/prefill_inversion/cache"
OUT_JSON = ROOT / "pathway11_h100/matrix_entropy/results.json"

# Layer on which F-1 (breathing) / F-2 (DoM) were established.
L19_INDEX = 19
N_EXPECTED = 500
D_EXPECTED = 1536
VALLEY_TOL = 2  # L19 counts as "in the valley" if within this many layers of bottom

# Candidate per-layer caches, in priority order; CACHE_DIR is globbed as fallback.
PERLAYER_CANDIDATES = [
    CACHE_DIR / "m15b_perlayer.npz",
    CACHE_DIR / "m15b_all_layers.npz",
    CACHE_DIR / "m15b_prefill_perlayer.npz",
    ROOT / "pathway11_h100/data/m15b_perlayer.npz",
]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def matrix_entropy(X: np.ndarray) -> float:
    """Matrix-based entropy of the normalized squared singular spectrum of X."""
    if X.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(X.astype(np.float64), compute_uv=False)
    ssq = s * s
    total = float(ssq.sum())
    if total <= 0.0:
        return float("nan")
    p = ssq / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def _find_perlayer_array() -> np.ndarray | None:
    """Locate a (n_layers, N, d) prefill tensor in the candidate/glob caches."""
    paths: list[Path] = [p for p in PERLAYER_CANDIDATES if p.exists()]
    if CACHE_DIR.exists():
        for p in sorted(CACHE_DIR.glob("*.npz")):
            if p not in paths:
                paths.append(p)
    for path in paths:
        try:
            blob = np.load(path, allow_pickle=False)
        except Exception:
            continue
        for key in blob.files:
            arr = blob[key]
            if arr.ndim == 3 and arr.shape[1] == N_EXPECTED and arr.shape[2] == D_EXPECTED:
                return arr.astype(np.float64)
            # Some caches store (N, n_layers, d) — normalize to layer-first.
            if arr.ndim == 3 and arr.shape[0] == N_EXPECTED and arr.shape[2] == D_EXPECTED:
                return np.transpose(arr, (1, 0, 2)).astype(np.float64)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    base = np.load(CACHE)
    correct = base["correct"].astype(bool)
    if correct.shape[0] != N_EXPECTED:
        print("MISSING_REGEN_INPUT", "unexpected correct shape", correct.shape, file=sys.stderr)
        return 2

    layers = _find_perlayer_array()
    if layers is None:
        print("MISSING_REGEN_INPUT", "no per-layer prefill cache (need (n_layers,500,1536))", file=sys.stderr)
        return 2

    n_layers = layers.shape[0]
    inc = ~correct

    H_all = np.full(n_layers, np.nan)
    H_correct = np.full(n_layers, np.nan)
    H_incorrect = np.full(n_layers, np.nan)

    for ell in range(n_layers):
        X = layers[ell]
        Xc = X - X.mean(axis=0, keepdims=True)
        H_all[ell] = matrix_entropy(Xc)
        H_correct[ell] = matrix_entropy(Xc[correct] - Xc[correct].mean(axis=0, keepdims=True))
        H_incorrect[ell] = matrix_entropy(Xc[inc] - Xc[inc].mean(axis=0, keepdims=True))

    valley_idx = int(np.nanargmin(H_all))
    l19_present = L19_INDEX < n_layers
    l19_in_valley = bool(l19_present and abs(valley_idx - L19_INDEX) <= VALLEY_TOL)

    # Correct-vs-incorrect entropy gap as a per-layer separability proxy.
    gap = H_incorrect - H_correct

    out = {
        "experiment": "P11-FE587",
        "n_layers": int(n_layers),
        "n_correct": int(correct.sum()),
        "n_incorrect": int(inc.sum()),
        "entropy_all": [float(x) for x in H_all],
        "entropy_correct": [float(x) for x in H_correct],
        "entropy_incorrect": [float(x) for x in H_incorrect],
        "entropy_gap_incorrect_minus_correct": [float(x) for x in gap],
        "valley_layer": valley_idx,
        "valley_entropy": float(H_all[valley_idx]),
        "l19_index": L19_INDEX,
        "l19_present": bool(l19_present),
        "l19_entropy": float(H_all[L19_INDEX]) if l19_present else None,
        "l19_in_compression_valley": l19_in_valley,
        "l19_offset_from_valley": int(L19_INDEX - valley_idx) if l19_present else None,
        "valley_tolerance_layers": VALLEY_TOL,
        "l19_entropy_gap": float(gap[L19_INDEX]) if l19_present else None,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())