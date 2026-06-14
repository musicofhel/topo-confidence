"""P11-FE540 — Sign-check intra-layer dispersion at final token, correct vs incorrect.

Cross-paper sign test. D²HScore (hidden-state dispersion) implies hallucinations
(incorrect answers) exhibit HIGHER intra-layer dispersion. F-4 (asymmetric
collapse) implies correct trajectories collapse HARDER, i.e. lower spread. The
two framings should AGREE on direction: incorrect problems sit farther from the
class/global centroid than correct ones. This script tests that empirically on
the cached P11 1.5B L19 prefill NPZs.

Data caveat: the committed P11 caches store a single L19 hidden vector per
problem (no per-token sequence, no attention matrices), so a true token-axis
attention-weighted centroid-distance cannot be recomputed here. We use the
recoverable proxy: per-problem L2 distance of the prefill vector to (a) the
global centroid and (b) a leave-one-out own-class centroid, in the 1536-d L19
space. Both are valid "dispersion of a point from the population center" readouts
and preserve the sign question. Buckets are K=1 correctness; we report bucket
means, histograms, a Mann-Whitney U rank test, and a directional verdict.
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
from scipy.stats import mannwhitneyu

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/dispersion_signcheck/results.json"

N_BINS = 30


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def loo_class_centroid_distance(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row L2 distance to its own-class centroid, leave-one-out.

    mask selects the class members. For a member i the centroid is computed
    over the other members of the same class (avoids self-inclusion bias);
    non-members get the full-class centroid as reference.
    """
    idx = np.flatnonzero(mask)
    n_cls = len(idx)
    dist = np.full(X.shape[0], np.nan, dtype=np.float64)
    if n_cls == 0:
        return dist
    cls_sum = X[idx].sum(axis=0)
    full_centroid = cls_sum / n_cls
    for i in range(X.shape[0]):
        if mask[i] and n_cls > 1:
            centroid = (cls_sum - X[i]) / (n_cls - 1)
        else:
            centroid = full_centroid
        dist[i] = float(np.linalg.norm(X[i] - centroid))
    return dist


def hist_pair(vals: np.ndarray, correct: np.ndarray, n_bins: int):
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    edges = np.linspace(lo, hi, n_bins + 1)
    c_counts, _ = np.histogram(vals[correct], bins=edges)
    i_counts, _ = np.histogram(vals[~correct], bins=edges)
    return {
        "bin_edges": [float(e) for e in edges],
        "correct_counts": [int(x) for x in c_counts],
        "incorrect_counts": [int(x) for x in i_counts],
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or correct.shape != (500,):
        print("MISSING_REGEN_INPUT bad shapes", X.shape, correct.shape, file=sys.stderr)
        return 2

    # Dispersion proxy A: distance to the global centroid.
    global_centroid = X.mean(axis=0)
    dist_global = np.linalg.norm(X - global_centroid, axis=1)

    # Dispersion proxy B: leave-one-out distance to own-class centroid.
    dist_loo = loo_class_centroid_distance(X, correct)

    results = {"experiment": "P11-FE540", "n": int(len(correct)),
               "n_correct": int(correct.sum()), "n_incorrect": int((~correct).sum()),
               "proxies": {}}

    for name, dist in (("global_centroid", dist_global), ("loo_class_centroid", dist_loo)):
        c_vals = dist[correct]
        i_vals = dist[~correct]
        mean_c = float(c_vals.mean())
        mean_i = float(i_vals.mean())
        # Rank test: alternative "incorrect has greater dispersion".
        u_stat, p_two = mannwhitneyu(i_vals, c_vals, alternative="two-sided")
        _, p_incorrect_greater = mannwhitneyu(i_vals, c_vals, alternative="greater")

        # AUROC of dispersion predicting INCORRECT (higher dist -> incorrect).
        auroc_incorrect = auroc(dist, ~correct)

        # Sign verdict: do D2HScore (incorrect higher) and F-4 (correct lower) agree?
        incorrect_more_dispersed = mean_i > mean_c
        verdict = ("AGREE_incorrect_more_dispersed" if incorrect_more_dispersed
                   else "DISAGREE_correct_more_dispersed")

        results["proxies"][name] = {
            "mean_dispersion_correct": mean_c,
            "mean_dispersion_incorrect": mean_i,
            "median_dispersion_correct": float(np.median(c_vals)),
            "median_dispersion_incorrect": float(np.median(i_vals)),
            "mean_gap_incorrect_minus_correct": mean_i - mean_c,
            "mannwhitney_u": float(u_stat),
            "p_two_sided": float(p_two),
            "p_incorrect_greater": float(p_incorrect_greater),
            "auroc_dispersion_predicts_incorrect": auroc_incorrect,
            "incorrect_more_dispersed": bool(incorrect_more_dispersed),
            "sign_verdict": verdict,
            "histogram": hist_pair(dist, correct, N_BINS),
        }

    # Top-line verdict from the global-centroid proxy.
    g = results["proxies"]["global_centroid"]
    results["headline_sign_verdict"] = g["sign_verdict"]
    results["headline_p_incorrect_greater"] = g["p_incorrect_greater"]
    results["note"] = ("attention-weighted token-axis dispersion not recoverable "
                       "from cached single-vector NPZs; centroid-distance proxy used")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())