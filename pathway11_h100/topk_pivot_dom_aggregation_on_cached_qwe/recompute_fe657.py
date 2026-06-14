"""P11-FE657 — Top-K pivot DoM aggregation vs single-layer prefill DoM and CoE-60.

On cached Qwen-2.5-1.5B MATH-500 trajectories, compute per-token final-answer
entropy from saved logprobs, select the K lowest-entropy ("pivot") positions per
trajectory, aggregate the L19 DoM projection at those positions, and measure
out-of-fold correctness AUROC. The DoM direction is refit per training fold from
the single-vector prefill cache (mean_correct - mean_incorrect) so the only thing
that changes across the sweep is the aggregator.

Sweep K in {1, 5, 10, 20, all}; K=all is the uniform-aggregation baseline. The
output is compared against the F-2 single-layer-prefill number (0.7731, OOF) and
the F-9 CoE-60 number (0.811). Tele-Lens (2602.02103) reports +9pp from K=5 pivot
selection over uniform aggregation; if pivot DoM clears 0.7731, F-9's "CoE-60 ≈
single-layer DoM" reading would need revisiting (right direction, wrong aggregator).

Requires a per-token cache (token-level L19 hidden states + per-token entropy);
the single-vector prefill cache alone is insufficient for pivot selection, so the
script returns MISSING_REGEN_INPUT (2) if that cache is absent.
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
PERTOKEN = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_pertoken.npz"
OUT_JSON = ROOT / "pathway11_h100/pivot_dom/results.json"

SEED = 9999
N_FOLDS = 5
K_SWEEP = [1, 5, 10, 20, -1]  # -1 == "all" (uniform aggregation)

REF_SINGLE_PREFILL = 0.7731  # F-2, OOF 5-fold
REF_COE60 = 0.811            # F-9, CoE-60


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _traj_arrays(blob, i: int):
    """Return (hidden (T,1536) float64, entropy (T,) float64) for trajectory i,
    supporting both ragged object arrays and padded (N,Tmax,*) arrays."""
    hidden = blob["hidden"]
    if "entropy" in blob.files:
        ent = blob["entropy"]
    elif "logprobs" in blob.files:
        ent = None  # derive below
    else:
        raise KeyError("per-token cache missing 'entropy' and 'logprobs'")

    if hidden.dtype == object:
        h = np.asarray(hidden[i], dtype=np.float64)
        if ent is not None and ent.dtype == object:
            e = np.asarray(ent[i], dtype=np.float64)
        elif ent is not None:
            e = np.asarray(ent[i], dtype=np.float64)
        else:
            lp = np.asarray(blob["logprobs"][i], dtype=np.float64)
            e = _entropy_from_logprobs(lp)
        return h, e

    # padded: need a length key
    if "tok_len" in blob.files:
        n = int(blob["tok_len"][i])
    elif "seq_len" in blob.files:
        n = int(blob["seq_len"][i])
    else:
        n = hidden.shape[1]
    h = np.asarray(hidden[i, :n], dtype=np.float64)
    if ent is not None:
        e = np.asarray(ent[i, :n], dtype=np.float64)
    else:
        lp = np.asarray(blob["logprobs"][i, :n], dtype=np.float64)
        e = _entropy_from_logprobs(lp)
    return h, e


def _entropy_from_logprobs(lp: np.ndarray) -> np.ndarray:
    """Shannon entropy per position from a (T, V) log-prob array."""
    lp = np.atleast_2d(lp)
    p = np.exp(lp - lp.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)


def aggregate(hidden: np.ndarray, entropy: np.ndarray, d_vec: np.ndarray, k: int) -> float:
    """Mean L19 DoM projection over the K lowest-entropy positions (all if k<0)."""
    dom = hidden @ d_vec
    n = len(dom)
    if n == 0:
        return 0.0
    if k < 0 or k >= n:
        return float(dom.mean())
    order = np.argsort(entropy, kind="stable")[:k]
    return float(dom[order].mean())


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not PERTOKEN.exists():
        print("MISSING_REGEN_INPUT", PERTOKEN, file=sys.stderr)
        return 2

    base = np.load(CACHE)
    X = base["prefill"].astype(np.float64)
    y = base["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    pt = np.load(PERTOKEN, allow_pickle=True)
    if "hidden" not in pt.files:
        print("MISSING_REGEN_INPUT", PERTOKEN, "(no 'hidden')", file=sys.stderr)
        return 2

    n = len(y)
    trajs = [_traj_arrays(pt, i) for i in range(n)]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Single-vector prefill DoM baseline (OOF), same fitting protocol.
    oof_single = np.zeros(n, dtype=np.float64)
    pivot_oof = {k: np.zeros(n, dtype=np.float64) for k in K_SWEEP}

    for train_idx, test_idx in skf.split(X, y):
        ytr = y[train_idx]
        d_vec = X[train_idx][ytr].mean(0) - X[train_idx][~ytr].mean(0)
        oof_single[test_idx] = X[test_idx] @ d_vec
        for ti in test_idx:
            h, e = trajs[ti]
            for k in K_SWEEP:
                pivot_oof[k][ti] = aggregate(h, e, d_vec, k)

    auroc_single = float(auroc(oof_single, y))
    sweep = {}
    best_k, best_auroc = None, -1.0
    for k in K_SWEEP:
        a = float(auroc(pivot_oof[k], y))
        key = "all" if k < 0 else str(k)
        sweep[key] = a
        if not np.isnan(a) and a > best_auroc:
            best_auroc, best_k = a, key

    uniform_auroc = sweep.get("all", float("nan"))
    pivot_lift_vs_uniform = (
        best_auroc - uniform_auroc if not np.isnan(uniform_auroc) else float("nan")
    )

    out = {
        "experiment": "P11-FE657",
        "title": "Top-K pivot DoM aggregation",
        "n": n,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_single_prefill_oof": auroc_single,
        "auroc_pivot_sweep": sweep,
        "best_k": best_k,
        "best_pivot_auroc": best_auroc,
        "uniform_aggregation_auroc": uniform_auroc,
        "pivot_lift_vs_uniform_pp": pivot_lift_vs_uniform,
        "ref_single_prefill_oof": REF_SINGLE_PREFILL,
        "ref_coe60": REF_COE60,
        "pivot_beats_single_prefill": bool(best_auroc > REF_SINGLE_PREFILL),
        "pivot_beats_coe60": bool(best_auroc > REF_COE60),
        "verdict": (
            "F9_AT_RISK pivot>single-prefill" if best_auroc > REF_SINGLE_PREFILL
            else "F9_HOLDS pivot<=single-prefill"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())