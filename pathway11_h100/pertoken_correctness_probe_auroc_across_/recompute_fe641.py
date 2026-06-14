"""P11-FE641 — Per-token correctness-probe AUROC across the full CoT trajectory.

Refutation 2 + 3 in one pass. The survey cites two literatures that predict a
shape for the probe-AUROC-vs-token-position curve that the F-2 (prefill DoM,
AUROC 0.7731) / F-3 (final-token DoM) pair never engages because we only ever
sampled two of the T positions:

  * Refusal-cliff (Yin 2025): a sharp position at which a linear readout snaps
    from uninformative to informative.
  * ARES mid-CoT failure probing (You 2025): an intermediate plateau where a
    mid-reasoning probe already predicts final correctness.

Method: load cached per-token L19 hidden states for all 500 MATH-500 problems
(P11 Stage 2 cache). Because CoT lengths vary, align tokens by *normalized*
position into N_BINS equal-width bins of [0,1]; for each problem take the mean
L19 activation of the tokens falling in each bin, giving a (n_problems, 1536)
matrix per bin. Score each bin with a 5-fold OOF DoM (mean-difference) probe
against ground-truth correctness, yielding AUROC(position). Prefill (from the
canonical prefill cache) and the final token are scored explicitly and marked.

Decision rule: if any intermediate normalized position beats AUROC 0.7731, the
F-2 ranking changes — there is a better single-layer readout than prefill DoM.
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
# Per-token L19 trajectory cache from P11 Stage 2 (one NPZ per problem).
PER_TOKEN_DIR = ROOT / "pathway11_h100/data/per_token_l19"
OUT_JSON = ROOT / "pathway11_h100/per_token_probe/results.json"

N_PROBLEMS = 500
HID_DIM = 1536
N_BINS = 20
N_FOLDS = 5
SEED = 9999
F2_THRESHOLD = 0.7731

# Candidate array keys for the per-token hidden states inside each problem NPZ.
HIDDEN_KEYS = ("hidden", "hidden_l19", "l19", "acts", "states", "tokens")


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


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """5-fold out-of-fold DoM (mean-difference) probe AUROC.

    Rows with any non-finite value are treated as missing and dropped before
    scoring (variable-length bins leave some problems with no token in a bin).
    """
    finite = np.isfinite(X).all(axis=1)
    if finite.sum() < 2 * k:
        return float("nan")
    Xf = X[finite]; yf = y[finite]
    if yf.sum() == 0 or (~yf).sum() == 0:
        return float("nan")
    n = len(yf)
    scores = np.full(n, np.nan, dtype=np.float64)
    folds = stratified_kfold(yf, k, seed)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = yf[train_mask]
        if ytr.sum() == 0 or (~ytr).sum() == 0:
            continue
        Xtr = Xf[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = Xf[test_idx] @ d_vec
    valid = np.isfinite(scores)
    if valid.sum() < 2:
        return float("nan")
    return auroc(scores[valid], yf[valid])


def _load_hidden(blob) -> np.ndarray | None:
    for key in HIDDEN_KEYS:
        if key in blob.files:
            arr = np.asarray(blob[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == HID_DIM and arr.shape[0] >= 1:
                return arr
    # Fall back to the sole 2-D, correctly-widthed array if present.
    for key in blob.files:
        arr = np.asarray(blob[key])
        if arr.ndim == 2 and arr.shape[1] == HID_DIM and arr.shape[0] >= 1:
            return arr.astype(np.float64)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not PER_TOKEN_DIR.exists():
        print("MISSING_REGEN_INPUT", PER_TOKEN_DIR, file=sys.stderr); return 2

    cache = np.load(CACHE)
    prefill = cache["prefill"].astype(np.float64)
    correct = cache["correct"].astype(bool)
    assert prefill.shape == (N_PROBLEMS, HID_DIM) and correct.shape == (N_PROBLEMS,)

    # Per-bin mean activation (mean over tokens in the normalized-position bin)
    # and the final-token activation, accumulated across all 500 problems.
    bin_acts = np.full((N_BINS, N_PROBLEMS, HID_DIM), np.nan, dtype=np.float64)
    final_acts = np.full((N_PROBLEMS, HID_DIM), np.nan, dtype=np.float64)

    n_loaded = 0
    n_missing = 0
    for i in range(N_PROBLEMS):
        fpath = PER_TOKEN_DIR / f"problem_{i:03d}.npz"
        if not fpath.exists():
            n_missing += 1
            continue
        with np.load(fpath) as blob:
            hidden = _load_hidden(blob)
        if hidden is None:
            n_missing += 1
            continue
        n_loaded += 1
        T = hidden.shape[0]
        final_acts[i] = hidden[-1]
        # Normalized position in [0,1); single-token sequences land in bin 0.
        if T == 1:
            pos = np.zeros(1, dtype=np.float64)
        else:
            pos = np.arange(T, dtype=np.float64) / float(T)
        bin_idx = np.clip((pos * N_BINS).astype(int), 0, N_BINS - 1)
        for b in range(N_BINS):
            sel = bin_idx == b
            if sel.any():
                bin_acts[b, i] = hidden[sel].mean(axis=0)

    if n_loaded == 0:
        print("MISSING_REGEN_INPUT", PER_TOKEN_DIR, "(no readable per-token files)",
              file=sys.stderr)
        return 2

    # Prefill and final-token reference AUROCs.
    prefill_auroc = oof_dom_auroc(prefill, correct, N_FOLDS, SEED)
    final_auroc = oof_dom_auroc(final_acts, correct, N_FOLDS, SEED)

    # Trajectory: AUROC at each normalized-position bin.
    trajectory = []
    best_intermediate = {"bin": None, "norm_pos_center": None, "auroc": float("-inf")}
    for b in range(N_BINS):
        Xb = bin_acts[b]
        n_present = int(np.isfinite(Xb).all(axis=1).sum())
        a = oof_dom_auroc(Xb, correct, N_FOLDS, SEED)
        center = (b + 0.5) / N_BINS
        trajectory.append({
            "bin": b,
            "norm_pos_center": round(center, 4),
            "n_problems": n_present,
            "auroc": (None if not np.isfinite(a) else round(float(a), 6)),
        })
        if np.isfinite(a) and a > best_intermediate["auroc"]:
            best_intermediate = {
                "bin": b,
                "norm_pos_center": round(center, 4),
                "auroc": round(float(a), 6),
            }

    if best_intermediate["bin"] is None:
        best_intermediate = {"bin": None, "norm_pos_center": None, "auroc": None}
    best_auroc = best_intermediate["auroc"]
    f2_ranking_changes = bool(best_auroc is not None and best_auroc > F2_THRESHOLD)

    out = {
        "experiment": "P11-FE641",
        "n_problems": N_PROBLEMS,
        "n_loaded": n_loaded,
        "n_missing": n_missing,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "f2_threshold": F2_THRESHOLD,
        "prefill_auroc": (None if not np.isfinite(prefill_auroc) else round(float(prefill_auroc), 6)),
        "final_token_auroc": (None if not np.isfinite(final_auroc) else round(float(final_auroc), 6)),
        "trajectory": trajectory,
        "best_intermediate": best_intermediate,
        "f2_ranking_changes": f2_ranking_changes,
        "interpretation": (
            "Refusal-cliff / mid-CoT readout beats prefill DoM"
            if f2_ranking_changes else
            "No intermediate normalized position beats prefill DoM (AUROC 0.7731); "
            "F-2 ranking unchanged"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())