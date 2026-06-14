"""P11-FE539 — D²HScore (label-free, all-layer) vs single-layer L19 DoM at 7B scale.

Direct test of whether a label-free, all-layer hidden-state score (D²HScore)
beats the single-layer L19 DoM probe on the cached Qwen2.5-7B Stage-2 NPZs.
This is a refutation test of F-9 ("all-layer signals collapse to single-layer
at 1.5B"): if D²HScore_AUROC_7B > L19-DoM_AUROC_7B at the scale we publish at,
F-9 is contradicted.

D²HScore here is computed from the per-layer pooled prefill representations
available in the cached 7B Stage-2 NPZ. The published D²HScore mixes an
intra-layer (token-dispersion) term with an inter-layer (second-difference of
the layer trajectory) term; token-level dispersion is not recoverable from the
pooled prefill cache, so this script implements the inter-layer / trajectory
component faithfully (second-order layer-difference magnitude, the "D²H"
curvature term) and documents the adaptation. The score is label-free: it never
sees `correct`. L19 DoM, by contrast, is fit supervised out-of-fold.

Refutes/contradicts F-9 at 7B iff d2hscore_auroc_7b > l19_dom_auroc_oof_7b.
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

# 7B all-layer cache (label-free D²HScore needs every layer). Try the known
# Stage-2 layouts in priority order; first existing wins.
ALLLAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_alllayer.npz",
    ROOT / "pathway11_h100/data/m7b_alllayer.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill_alllayer.npz",
]
# Fallback single-layer 7B prefill cache (L19 already extracted).
L19_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz",
    ROOT / "pathway11_h100/data/m7b_prefill.npz",
]

OUT_JSON = ROOT / "pathway11_h100/d2hscore_7b/results.json"

N_FOLDS = 5
SEED = 9999
L19_INDEX = 19  # single-layer L19 probe, matching the 1.5B convention


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
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


def first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_alllayer(path: Path):
    """Return (hidden, correct, seq_len) with hidden shape (N, L, D).

    Accepts a few key namings; transposes (L, N, D) -> (N, L, D) if needed.
    """
    blob = np.load(path)
    keys = set(blob.files)
    hkey = next((k for k in ("hidden", "hidden_states", "all_layers", "prefill_all") if k in keys), None)
    if hkey is None:
        return None
    H = np.asarray(blob[hkey]).astype(np.float64)
    if H.ndim != 3:
        return None
    correct = blob["correct"].astype(bool) if "correct" in keys else None
    seq_len = blob["seq_len"].astype(np.float64) if "seq_len" in keys else None
    if correct is None:
        return None
    n = correct.shape[0]
    # Orient to (N, L, D): the axis equal to n is the sample axis.
    if H.shape[0] != n and H.shape[1] == n:
        H = np.transpose(H, (1, 0, 2))
    if H.shape[0] != n:
        return None
    return H, correct, seq_len


def d2hscore(H: np.ndarray) -> np.ndarray:
    """Label-free inter-layer D²H (trajectory-curvature) score, one per sample.

    H: (N, L, D). Each layer rep is L2-normalized so the score reflects
    directional reorientation of the residual-stream trajectory rather than raw
    norm growth. The D²H term is the mean magnitude of the second-order
    layer-to-layer difference (discrete curvature of the trajectory). Higher
    curvature = a less stable, more "searching" trajectory; we negate so that
    higher score = more confident/correct, matching the L19-DoM orientation.
    """
    Hn = H / (np.linalg.norm(H, axis=2, keepdims=True) + 1e-12)
    # second difference along the layer axis: Hn[:, l+1] - 2 Hn[:, l] + Hn[:, l-1]
    d2 = Hn[:, 2:, :] - 2.0 * Hn[:, 1:-1, :] + Hn[:, :-2, :]
    curvature = np.linalg.norm(d2, axis=2).mean(axis=1)  # (N,)
    return -curvature


def l19_dom_oof(X19: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold L19 DoM projection scores (supervised difference-of-means)."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in stratified_kfold(y, N_FOLDS, SEED):
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X19[train_mask], y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X19[test_idx] @ d_vec
    return scores


def main() -> int:
    allpath = first_existing(ALLLAYER_CANDIDATES)
    l19path = first_existing(L19_CANDIDATES)

    if allpath is None and l19path is None:
        print("MISSING_REGEN_INPUT", ALLLAYER_CANDIDATES[0], file=sys.stderr)
        return 2

    H = correct = seq_len = X19 = None

    if allpath is not None:
        loaded = load_alllayer(allpath)
        if loaded is not None:
            H, correct, seq_len = loaded

    # Resolve L19 single-layer reps + labels from whichever source is available.
    if H is not None:
        if H.shape[1] <= L19_INDEX:
            print("MISSING_REGEN_INPUT", f"all-layer cache has only {H.shape[1]} layers (<{L19_INDEX+1})", file=sys.stderr)
            return 2
        X19 = H[:, L19_INDEX, :]
    if (X19 is None or correct is None) and l19path is not None:
        blob = np.load(l19path)
        if "prefill" in blob.files and "correct" in blob.files:
            X19 = blob["prefill"].astype(np.float64)
            if correct is None:
                correct = blob["correct"].astype(bool)

    if correct is None or X19 is None:
        print("MISSING_REGEN_INPUT", "no usable 7B labels/L19 reps", file=sys.stderr)
        return 2

    n = correct.shape[0]

    # --- single-layer L19 DoM (supervised, OOF) ---
    l19_auroc = float(auroc(l19_dom_oof(X19, correct), correct))

    out: dict = {
        "experiment": "P11-FE539",
        "scale": "7B",
        "n": int(n),
        "n_correct": int(correct.sum()),
        "l19_index": L19_INDEX,
        "l19_dom_auroc_oof_7b": l19_auroc,
        "alllayer_cache": str(allpath) if allpath is not None else None,
        "l19_cache": str(l19path) if l19path is not None else None,
    }

    # --- D²HScore (label-free, all-layer) ---
    if H is not None:
        score = d2hscore(H)
        d2h_auroc = float(auroc(score, correct))
        # label-free orientation is not fixed a priori; report both for honesty.
        out.update({
            "n_layers": int(H.shape[1]),
            "hidden_dim": int(H.shape[2]),
            "d2hscore_auroc_7b": d2h_auroc,
            "d2hscore_auroc_flipped": float(auroc(-score, correct)),
            "d2hscore_auroc_oriented": float(max(d2h_auroc, 1.0 - d2h_auroc)),
            "delta_vs_l19": float(d2h_auroc - l19_auroc),
            "f9_contradicted_at_7b": bool(d2h_auroc > l19_auroc),
            "f9_contradicted_oriented": bool(max(d2h_auroc, 1.0 - d2h_auroc) > l19_auroc),
        })
    else:
        out.update({
            "d2hscore_auroc_7b": None,
            "note": "all-layer 7B cache unavailable; D²HScore not computable. "
                    "Reporting L19 DoM only. Re-run after regenerating the "
                    "all-layer Stage-2 NPZ (see DATA_MANIFEST.md).",
            "f9_contradicted_at_7b": None,
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())