"""P11-FE1003 — Last-token-only vs mean-pooled prefill DoM AUROC.

Paper Ablation 3 reports that last-token pooling yields d ≈ 0 (no signal) for
semantic-identity geometry. This experiment tests whether our F-2 correctness
direction (mean-pooled L19 prefill DoM, OOF AUROC 0.7731) survives a switch to
last-token-only pooling.

The committed main cache (`m15b_prefill.npz`) stores a single pooled (500, 1536)
`prefill` vector per problem — the mean-pooled representation that produces the
0.7731 baseline. Recovering a last-token-only vector requires per-token L19
activations. We look for those in a per-token NPZ; if absent we cannot
re-extract on CPU (would need the model on GPU), so we print MISSING_REGEN_INPUT
and exit 2 rather than fabricating a number.

Per-token NPZ schema expected (any one of these layouts):
  - key "prefill_tokens": object/ragged array, [i] -> (T_i, 1536) float
  - key "prefill_tokens" (3D): (500, T, 1536) with key "seq_len" giving valid T_i
  - key "prefill_last": (500, 1536) precomputed last-token vectors

Both pooled and last-token DoM directions are fit per train fold and scored OOF
over 5 stratified folds, so the comparison is apples-to-apples.
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
# Candidate per-token caches (first existing one is used).
PERTOK_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_tokens.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_pertoken.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_lasttok.npz",
]
OUT_JSON = ROOT / "pathway11_h100/lasttok_pooling/results.json"

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


def dom_oof_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """5-fold OOF AUROC for a fit-per-train-fold difference-of-means direction."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d_vec
    return float(auroc(scores, y))


def extract_last_token(blob, n: int, dim: int) -> np.ndarray | None:
    """Return (n, dim) last-token L19 vectors from a per-token NPZ, or None."""
    keys = set(blob.files)
    # Layout C: precomputed last-token vectors.
    if "prefill_last" in keys:
        arr = np.asarray(blob["prefill_last"], dtype=np.float64)
        if arr.shape == (n, dim):
            return arr
    # Layouts A/B: per-token activations.
    if "prefill_tokens" in keys:
        tok = blob["prefill_tokens"]
        seq_len = blob["seq_len"].astype(int) if "seq_len" in keys else None
        # Dense 3D (n, T, dim).
        if getattr(tok, "ndim", 0) == 3 and tok.shape[0] == n and tok.shape[2] == dim:
            out = np.zeros((n, dim), dtype=np.float64)
            for i in range(n):
                t_last = (seq_len[i] - 1) if seq_len is not None else (tok.shape[1] - 1)
                t_last = int(max(0, min(t_last, tok.shape[1] - 1)))
                out[i] = tok[i, t_last].astype(np.float64)
            return out
        # Ragged object array of (T_i, dim).
        if getattr(tok, "dtype", None) == object and len(tok) == n:
            out = np.zeros((n, dim), dtype=np.float64)
            for i in range(n):
                ti = np.asarray(tok[i], dtype=np.float64)
                if ti.ndim != 2 or ti.shape[1] != dim or ti.shape[0] == 0:
                    return None
                out[i] = ti[-1]
            return out
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    blob = np.load(CACHE)
    X_mean = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X_mean.shape == (500, 1536) and y.shape == (500,)
    n, dim = X_mean.shape

    pertok_path = next((p for p in PERTOK_CANDIDATES if p.exists()), None)
    if pertok_path is None:
        print("MISSING_REGEN_INPUT", "no per-token L19 cache; "
              "last-token-only requires GPU re-extraction (not CPU-runnable):",
              " | ".join(str(p) for p in PERTOK_CANDIDATES), file=sys.stderr)
        return 2

    tok_blob = np.load(pertok_path, allow_pickle=True)
    X_last = extract_last_token(tok_blob, n, dim)
    if X_last is None:
        print("MISSING_REGEN_INPUT", "per-token cache present but no usable "
              "last-token layout:", pertok_path, file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)
    auroc_mean = dom_oof_auroc(X_mean, y, folds)
    auroc_last = dom_oof_auroc(X_last, y, folds)

    out = {
        "experiment": "P11-FE1003",
        "pertoken_cache": str(pertok_path),
        "n": int(n),
        "n_correct": int(y.sum()),
        "auroc_mean_pooled_dom_oof": auroc_mean,
        "auroc_last_token_dom_oof": auroc_last,
        "delta_last_minus_mean": float(auroc_last - auroc_mean),
        "mean_pooled_reference_0p7731": 0.7731,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())