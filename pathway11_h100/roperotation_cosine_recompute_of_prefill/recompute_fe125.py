"""P11-FE125 — RoPE-rotation cosine of prefill vs final L19 DoM.

F-3 reports cos(prefill_DoM, final_DoM) = 0.046, read as "prefill and final
encode different signals." Quirke-Barez §7 (one-layer models run different
sub-algorithms at different output positions) motivates an alternative: the two
directions could be the *same* signal expressed in position-rotated bases. RoPE
applies a per-position rotation to the residual representation; if rotating the
prefill DoM direction by RoPE(position) for some position p lifts its cosine
against the final-token DoM materially above 0.046 (threshold 0.3), F-3 reframes
from "semantic independence" to "same signal, rotated basis."

Method: derive d_prefill = mean(correct) - mean(incorrect) over the L19 prefill
activations and d_final from the parallel final-token cache. Apply HF-Qwen2
rotate_half RoPE (per-head, head_dim=128, num_heads=12) for every position
0..max(seq_len)-1, sweeping rope_theta in {1e4, 1e6}. Report min/median/max
cosine and the argmax position per theta. Position 0 reproduces the 0.046
baseline (identity rotation) as a sanity check.
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
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Parallel final-token L19 cache (same schema as the prefill cache, but the
# hidden state is taken at the final answer token). Try a few plausible paths.
FINAL_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/final_inversion/cache/m15b_final.npz",
]
OUT_JSON = ROOT / "pathway11_h100/rope_rotation_dom/results.json"

HIDDEN = 1536
NUM_HEADS = 12
HEAD_DIM = HIDDEN // NUM_HEADS  # 128
ROPE_THETAS = [1.0e4, 1.0e6]
MATERIAL_THRESHOLD = 0.3
BASELINE_COS = 0.046


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _find_2d_500x1536(blob) -> np.ndarray | None:
    """Return the first (500, 1536) array in an NPZ, preferring named keys."""
    preferred = ["final", "final_token", "hidden", "hidden_states", "prefill"]
    for key in preferred:
        if key in blob.files:
            arr = blob[key]
            if arr.ndim == 2 and arr.shape == (500, HIDDEN):
                return arr.astype(np.float64)
    for key in blob.files:
        arr = blob[key]
        if getattr(arr, "ndim", 0) == 2 and arr.shape == (500, HIDDEN):
            return arr.astype(np.float64)
    return None


def dom_direction(X: np.ndarray, correct: np.ndarray) -> np.ndarray:
    return X[correct].mean(axis=0) - X[~correct].mean(axis=0)


def rope_rotate_alldim(direction: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Apply HF-Qwen2 rotate_half RoPE to a (HIDDEN,) direction at each position.

    Per-head: reshape to (NUM_HEADS, HEAD_DIM); inv_freq over HEAD_DIM; cos/sin
    duplicated across the two halves; rotate_half(v) = concat(-v[d/2:], v[:d/2]).
    Returns (n_pos, HIDDEN).
    """
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))  # (half,)
    angles = positions[:, None] * inv_freq[None, :]                                       # (n_pos, half)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=1)                        # (n_pos, HEAD_DIM)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=1)                        # (n_pos, HEAD_DIM)

    V = direction.reshape(NUM_HEADS, HEAD_DIM)                                            # (H, D)
    rot_half = np.concatenate([-V[:, half:], V[:, :half]], axis=1)                        # (H, D)

    # (n_pos, H, D)
    rotated = V[None, :, :] * cos[:, None, :] + rot_half[None, :, :] * sin[:, None, :]
    return rotated.reshape(positions.shape[0], HIDDEN)


def main() -> int:
    if not PREFILL_CACHE.exists():
        print("MISSING_REGEN_INPUT", PREFILL_CACHE, file=sys.stderr)
        return 2

    final_cache = next((p for p in FINAL_CACHE_CANDIDATES if p.exists()), None)
    if final_cache is None:
        print("MISSING_REGEN_INPUT", " | ".join(str(p) for p in FINAL_CACHE_CANDIDATES), file=sys.stderr)
        return 2

    pblob = np.load(PREFILL_CACHE)
    X_pre = pblob["prefill"].astype(np.float64)
    correct = pblob["correct"].astype(bool)
    seq_len = pblob["seq_len"].astype(np.int64)
    assert X_pre.shape == (500, HIDDEN) and correct.shape == (500,)

    fblob = np.load(final_cache)
    X_fin = _find_2d_500x1536(fblob)
    if X_fin is None:
        print("MISSING_REGEN_INPUT", f"{final_cache}: no (500,{HIDDEN}) array", file=sys.stderr)
        return 2

    d_pre = dom_direction(X_pre, correct)
    d_fin = dom_direction(X_fin, correct)
    nfin = float(np.linalg.norm(d_fin))
    if nfin < 1e-12 or float(np.linalg.norm(d_pre)) < 1e-12:
        print("MISSING_REGEN_INPUT", "degenerate DoM direction", file=sys.stderr)
        return 2
    d_fin_hat = d_fin / nfin

    base_cos = float((d_pre / np.linalg.norm(d_pre)) @ d_fin_hat)

    max_pos = int(seq_len.max())
    positions = np.arange(0, max_pos, dtype=np.float64)

    per_theta = {}
    overall_max = -1.0
    overall_max_theta = None
    overall_max_pos = None
    for theta in ROPE_THETAS:
        rotated = rope_rotate_alldim(d_pre, positions, theta)               # (n_pos, HIDDEN)
        norms = np.linalg.norm(rotated, axis=1)
        norms[norms < 1e-12] = 1.0
        cosines = (rotated @ d_fin_hat) / norms                             # (n_pos,)
        imax = int(np.argmax(cosines))
        imin = int(np.argmin(cosines))
        key = f"theta_{theta:.0e}"
        per_theta[key] = {
            "rope_theta": float(theta),
            "cos_min": float(cosines[imin]),
            "cos_median": float(np.median(cosines)),
            "cos_max": float(cosines[imax]),
            "argmax_position": imax,
            "argmin_position": imin,
            "cos_at_pos0": float(cosines[0]),
        }
        if cosines[imax] > overall_max:
            overall_max = float(cosines[imax])
            overall_max_theta = float(theta)
            overall_max_pos = imax

    out = {
        "experiment": "P11-FE125",
        "description": "RoPE-rotation cosine of prefill vs final L19 DoM",
        "final_cache_used": str(final_cache.relative_to(ROOT)),
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "n_positions": int(max_pos),
        "baseline_cos_unrotated": base_cos,
        "f3_reference_cos": BASELINE_COS,
        "material_threshold": MATERIAL_THRESHOLD,
        "per_theta": per_theta,
        "overall_max_rotated_cos": overall_max,
        "overall_max_theta": overall_max_theta,
        "overall_max_position": overall_max_pos,
        "reframes_f3": bool(overall_max > MATERIAL_THRESHOLD),
        "auroc_prefill_dom_oof_check": float(auroc(X_pre @ d_pre, correct)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"overall_max_rotated_cos={overall_max:.4f} "
          f"(theta={overall_max_theta:.0e}, pos={overall_max_pos}) "
          f"baseline={base_cos:.4f} reframes_f3={out['reframes_f3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())