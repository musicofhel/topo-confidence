"""P11-FE17 — RoPE de-rotation of the temporal DoM direction.

For the 1.5B model's per-token L19 residual-stream activations, undo the RoPE
rotation at each token's absolute position (deterministic from the model config:
head_dim=128, GPT-NeoX/Llama-style rotate_half, fixed inv-freq base), then
recompute the temporal Direction-of-Mean (DoM) AUROC curve and the
cosine-to-final-token table on both the raw and de-rotated activations.

F-3 reports that the prefill DoM and final-token DoM are nearly orthogonal
(cos ~= 0.046). RoPE applies position-dependent rotations to dimension pairs, so
two tokens at very different absolute positions will have their shared signal
rotated apart. If de-rotating every token to position 0 lifts cos(DoM_t,
DoM_final) above 0.8, the orthogonality is a mechanical RoPE artifact and E1
fixed-vector steering becomes viable again. L19 is post-attention/MLP, so a
partial lift (cos 0.2 -> 0.6) is still informative.

Needs per-token L19 activations (not in the headline prefill cache). Looks for a
per-problem cache directory; reports MISSING_REGEN_INPUT and returns 2 if absent.
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
# Per-token L19 activations, one file per problem: key "hidden" (T, 1536),
# optional key "positions" (T,) absolute token indices (defaults to arange(T)).
PERTOKEN_DIR = ROOT / "pathway11_h100/data/pertoken_l19"
OUT_JSON = ROOT / "pathway11_h100/rope_derotation/results.json"

# Qwen2.5-Math-1.5B RoPE config (deterministic, no model load needed).
HIDDEN = 1536
N_HEADS = 12
HEAD_DIM = HIDDEN // N_HEADS  # 128
ROPE_THETA = 10000.0

# End-aligned token offsets at which to measure the temporal DoM curve.
# offset 0 = last (final) token; larger = further back toward the prefill.
OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64, 128]

N_FOLDS = 5
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def rope_derotate(hidden: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Undo RoPE on each row of `hidden` (N, 1536) at its absolute position.

    GPT-NeoX/Llama rotate_half convention: within each head_dim block, dim i is
    paired with dim i+head_dim/2 and rotated by angle p * inv_freq[i]. The
    forward map is  a' = a*cos - b*sin ;  b' = b*cos + a*sin. De-rotation rotates
    by -angle:  a = a'*cos + b'*sin ;  b = b'*cos - a'*sin.
    """
    n = hidden.shape[0]
    half = HEAD_DIM // 2
    inv_freq = ROPE_THETA ** (-(2.0 * np.arange(half, dtype=np.float64)) / HEAD_DIM)
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]  # (N, half)
    cos = np.cos(angles)[:, None, :]  # (N, 1, half)
    sin = np.sin(angles)[:, None, :]

    blocks = hidden.astype(np.float64).reshape(n, N_HEADS, HEAD_DIM)
    a = blocks[:, :, :half]
    b = blocks[:, :, half:]
    derot_a = a * cos + b * sin
    derot_b = b * cos - a * sin
    out = np.concatenate([derot_a, derot_b], axis=2)
    return out.reshape(n, HIDDEN)


def oof_dom_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Out-of-fold AUROC of the DoM direction fit on each training fold."""
    if y.sum() < N_FOLDS or (~y).sum() < N_FOLDS:
        return float("nan")
    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        if ytr.sum() == 0 or (~ytr).sum() == 0:
            scores[test_idx] = X[test_idx] @ np.zeros(X.shape[1])
            continue
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return auroc(scores, y)


def full_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float((u @ v) / (nu * nv))


def gather_offset(records, offset):
    """Return (X_raw, X_derot, y, abs_positions) for tokens `offset` from the end."""
    Xr, Xd, ys, ps = [], [], [], []
    for hidden, positions, correct in records:
        t = hidden.shape[0]
        idx = t - 1 - offset
        if idx < 0:
            continue
        Xr.append(hidden[idx])
        Xd.append(rope_derotate(hidden[idx:idx + 1], positions[idx:idx + 1])[0])
        ys.append(bool(correct))
        ps.append(int(positions[idx]))
    if not Xr:
        return None
    return (np.asarray(Xr, dtype=np.float64),
            np.asarray(Xd, dtype=np.float64),
            np.asarray(ys, dtype=bool),
            np.asarray(ps, dtype=np.int64))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not PERTOKEN_DIR.exists():
        print("MISSING_REGEN_INPUT", PERTOKEN_DIR, file=sys.stderr); return 2

    labels = np.load(CACHE)["correct"].astype(bool)
    assert labels.shape == (500,)

    records = []
    for i in range(500):
        f = PERTOKEN_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        blob = np.load(f)
        hidden = blob["hidden"].astype(np.float32)
        if hidden.ndim != 2 or hidden.shape[1] != HIDDEN or hidden.shape[0] == 0:
            continue
        positions = (blob["positions"] if "positions" in blob.files
                     else np.arange(hidden.shape[0])).astype(np.int64)
        records.append((hidden, positions, bool(labels[i])))

    if not records:
        print("MISSING_REGEN_INPUT", PERTOKEN_DIR, "(no usable per-problem files)",
              file=sys.stderr); return 2

    # Final-token DoM (offset 0), on raw and de-rotated activations.
    final = gather_offset(records, 0)
    if final is None:
        print("MISSING_REGEN_INPUT", PERTOKEN_DIR, "(no final tokens)", file=sys.stderr)
        return 2
    Xr0, Xd0, y0, _ = final
    dom_final_raw = full_dom(Xr0, y0)
    dom_final_derot = full_dom(Xd0, y0)

    auroc_raw, auroc_derot = {}, {}
    cos_raw, cos_derot = {}, {}
    n_per_offset = {}

    for off in OFFSETS:
        g = gather_offset(records, off)
        if g is None:
            continue
        Xr, Xd, y, _ = g
        n_per_offset[off] = int(len(y))
        auroc_raw[off] = oof_dom_auroc(Xr, y)
        auroc_derot[off] = oof_dom_auroc(Xd, y)
        if y.sum() == 0 or (~y).sum() == 0:
            cos_raw[off] = float("nan"); cos_derot[off] = float("nan"); continue
        cos_raw[off] = cosine(full_dom(Xr, y), dom_final_raw)
        cos_derot[off] = cosine(full_dom(Xd, y), dom_final_derot)

    # Verdict: does de-rotation lift cos-to-final above 0.8 at the deepest
    # offset available (the most prefill-like, largest positional gap)?
    far_offsets = [o for o in OFFSETS if o in cos_derot and o > 0
                   and np.isfinite(cos_derot[o])]
    if far_offsets:
        deep = max(far_offsets)
        deep_cos_raw = cos_raw.get(deep, float("nan"))
        deep_cos_derot = cos_derot[deep]
        mechanical = bool(deep_cos_derot > 0.8)
        verdict = "MECHANICAL_ROPE_ARTIFACT" if mechanical else "NOT_MECHANICAL"
    else:
        deep = None
        deep_cos_raw = deep_cos_derot = float("nan")
        mechanical = False
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "P11-FE17",
        "description": "RoPE de-rotation of temporal DoM; test if prefill/final "
                       "orthogonality (F-3) is a mechanical RoPE artifact",
        "rope_config": {"hidden": HIDDEN, "n_heads": N_HEADS,
                        "head_dim": HEAD_DIM, "theta": ROPE_THETA,
                        "convention": "gpt_neox_rotate_half"},
        "n_problems_loaded": len(records),
        "offsets": OFFSETS,
        "n_per_offset": n_per_offset,
        "auroc_raw": {str(k): v for k, v in auroc_raw.items()},
        "auroc_derot": {str(k): v for k, v in auroc_derot.items()},
        "cos_to_final_raw": {str(k): v for k, v in cos_raw.items()},
        "cos_to_final_derot": {str(k): v for k, v in cos_derot.items()},
        "deep_offset": deep,
        "deep_cos_to_final_raw": deep_cos_raw,
        "deep_cos_to_final_derot": deep_cos_derot,
        "mechanical_threshold": 0.8,
        "mechanical": mechanical,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())