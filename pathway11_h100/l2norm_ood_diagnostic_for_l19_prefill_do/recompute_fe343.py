"""FE343 — L2-norm OOD diagnostic for the L19 prefill DoM direction.

Mayne et al. argue that contrastive steering vectors (CAA) are out-of-distribution
"by construction": the L2 norm of the steering direction sits far below the
histogram of per-example activation L2 norms ||a_L(x)||_2, so adding alpha*v
pushes activations off the data manifold and produces non-monotonic alpha-sweeps.

This is the cheapest empirical test of that claim against our L19 prefill DoM.
We compute:
  - ||DoM||_2, the L2 norm of the difference-of-means direction
    (mean(prefill[correct]) - mean(prefill[~correct])), full-data and as the
    mean over 5 OOF train folds (the quantity an alpha-sweep would actually use);
  - the histogram of ||a_L19(x)||_2 over all 500 MATH-500 prefill activations,
    with summary stats, percentiles, and bin counts replicating the Mayne
    Fig. 1 / Fig. 5 layout;
  - where ||DoM||_2 falls relative to that histogram (percentile, z-score,
    ratio to median activation norm).

A Pile-token reference histogram is overlaid if a cache is present; otherwise
the field is recorded as null and the MATH-500 histogram stands alone.

Verdict logic: if ||DoM||_2 lands inside the activation-norm histogram, Mayne's
OOD-by-construction refutation against H-1 weakens; if it sits far below (as
Mayne reports for all 7 CAA behaviors), H-1 must add an in-distribution scaling
step or expect non-monotonic alpha-sweeps.
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
# Optional held-out Pile-token reference (not assumed to exist).
PILE_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/pile_l19_reference.npz"
OUT_JSON = ROOT / "pathway11_h100/dom_l2_ood/results.json"

N_FOLDS = 5
SEED = 9999
N_BINS = 40


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def percentile_of(value: float, sample: np.ndarray) -> float:
    """Fraction of sample strictly below value, plus half the ties (in [0,1])."""
    below = (sample < value).sum() + 0.5 * (sample == value).sum()
    return float(below / len(sample))


def histogram_block(values: np.ndarray, bins: np.ndarray) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "percentiles": {
            "p1": float(np.percentile(values, 1)),
            "p5": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # --- DoM direction (full data) and its L2 norm ---
    dom_full = X[y].mean(axis=0) - X[~y].mean(axis=0)
    dom_norm_full = float(np.linalg.norm(dom_full))

    # --- DoM L2 norm averaged over OOF train folds (the alpha-sweep quantity) ---
    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    fold_norms = []
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        fold_norms.append(float(np.linalg.norm(d)))
    fold_norms = np.array(fold_norms, dtype=np.float64)
    dom_norm_oof_mean = float(fold_norms.mean())
    dom_norm_oof_std = float(fold_norms.std(ddof=1))

    # Sanity: confirm the cached DoM scores reproduce the expected signal if present.
    dom_score_auroc = None
    if DOM_NPZ.exists():
        try:
            dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
            if dom_score.shape == (500,):
                dom_score_auroc = auroc(dom_score, y)
        except Exception:
            dom_score_auroc = None

    # --- Activation L2-norm histogram across the 500 MATH-500 problems ---
    act_norms = np.linalg.norm(X, axis=1)
    lo = float(min(act_norms.min(), dom_norm_full))
    hi = float(max(act_norms.max(), dom_norm_full))
    span = max(hi - lo, 1e-9)
    bins = np.linspace(lo - 0.02 * span, hi + 0.02 * span, N_BINS + 1)
    math_hist = histogram_block(act_norms, bins)

    # --- Optional Pile-token reference overlay ---
    pile_hist = None
    pile_dom_position = None
    if PILE_NPZ.exists():
        try:
            pblob = np.load(PILE_NPZ)
            pkey = "prefill" if "prefill" in pblob else (
                "activations" if "activations" in pblob else list(pblob.keys())[0])
            P = pblob[pkey].astype(np.float64)
            if P.ndim == 2 and P.shape[1] == X.shape[1]:
                pile_norms = np.linalg.norm(P, axis=1)
                pmin = float(min(pile_norms.min(), dom_norm_full))
                pmax = float(max(pile_norms.max(), dom_norm_full))
                pspan = max(pmax - pmin, 1e-9)
                pbins = np.linspace(pmin - 0.02 * pspan, pmax + 0.02 * pspan, N_BINS + 1)
                pile_hist = histogram_block(pile_norms, pbins)
                pile_dom_position = {
                    "percentile": percentile_of(dom_norm_full, pile_norms),
                    "z_score": float((dom_norm_full - pile_norms.mean()) / pile_norms.std(ddof=1)),
                    "ratio_to_median": float(dom_norm_full / np.median(pile_norms)),
                }
        except Exception:
            pile_hist = None

    # --- Where does ||DoM||_2 sit relative to the activation histogram? ---
    median_act = float(np.median(act_norms))
    dom_position = {
        "percentile": percentile_of(dom_norm_full, act_norms),
        "z_score": float((dom_norm_full - act_norms.mean()) / act_norms.std(ddof=1)),
        "ratio_to_median": float(dom_norm_full / median_act),
        "ratio_to_min": float(dom_norm_full / float(act_norms.min())),
        "inside_histogram": bool(act_norms.min() <= dom_norm_full <= act_norms.max()),
    }

    # Mayne reports CAA vectors sitting "far below" the activation histogram.
    # Operationalize: below the 1st percentile of activation norms => far below.
    far_below = bool(dom_norm_full < np.percentile(act_norms, 1))

    if dom_position["inside_histogram"]:
        verdict = ("DoM_L2_inside_activation_histogram: Mayne OOD-by-construction "
                   "refutation against H-1 WEAKENS")
    elif far_below:
        verdict = ("DoM_L2_far_below_activation_histogram: matches Mayne; H-1 needs "
                   "in-distribution scaling step / expect non-monotonic alpha-sweeps")
    else:
        verdict = ("DoM_L2_outside_but_not_far_below: ambiguous; inspect ratio_to_median")

    out = {
        "experiment": "P11-FE343",
        "description": "L2-norm OOD diagnostic for L19 prefill DoM vs Mayne et al.",
        "dom_norm_full": dom_norm_full,
        "dom_norm_oof_mean": dom_norm_oof_mean,
        "dom_norm_oof_std": dom_norm_oof_std,
        "dom_norm_per_fold": [float(v) for v in fold_norms],
        "dom_score_auroc_sanity": dom_score_auroc,
        "activation_norm_histogram": math_hist,
        "dom_position_vs_activation": dom_position,
        "far_below_p1": far_below,
        "pile_reference_available": pile_hist is not None,
        "pile_norm_histogram": pile_hist,
        "dom_position_vs_pile": pile_dom_position,
        "verdict": verdict,
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT_JSON)
    print("dom_norm_full=%.4f  median_act_norm=%.4f  percentile=%.4f  verdict=%s"
          % (dom_norm_full, median_act, dom_position["percentile"], verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())