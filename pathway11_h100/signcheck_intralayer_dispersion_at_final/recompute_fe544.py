"""P11-FE544 — Sign-check intra-layer dispersion at final-token, correct vs incorrect.

Two prior framings make a sign prediction about hidden-state dispersion:

  * D²HScore: hallucinations (incorrect answers) carry HIGHER dispersion in the
    residual stream — incorrect should sit FARTHER from the population centroid.
  * F-4 (asymmetric collapse): correct trajectories collapse HARDER (lower
    spread) — correct should sit CLOSER to the population centroid.

Both framings therefore predict the same observable sign:
    dispersion(correct) < dispersion(incorrect).
If the data inverts this, one of the two framings is measuring something else.

Data limitation / honest proxy:
  The committed P11 cache stores a SINGLE L19 vector per problem (the final
  prefill token), not the per-token trajectory, and carries no attention
  weights. A literal "attention-weighted centroid-distance over the layer's
  token positions" is therefore not computable from this NPZ. We compute the
  defensible population-level proxy: per-problem distance of the final-token
  L19 representation from the population centroid (Euclidean + ridge
  Mahalanobis), bucket by K=1 correctness, run a rank test (Mann-Whitney U),
  and report the SIGN of the effect plus histograms. The seq_len-normalized
  Euclidean distance is reported as a robustness variant (length stand-in for
  the missing attention weighting).
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
OUT_JSON = ROOT / "pathway11_h100/signcheck_intralayer_dispersion_at_final/results.json"

N_BINS = 30
RIDGE_REL = 1e-3


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def rank_test(dist: np.ndarray, correct: np.ndarray) -> dict:
    """Mann-Whitney U on dispersion (incorrect vs correct) + effect sizes.

    AUROC uses distance as a score predicting INCORRECT (label = ~correct):
    >0.5 means incorrect carries more dispersion (D²HScore / F-4 sign).
    """
    d_correct = dist[correct]
    d_incorrect = dist[~correct]
    if len(d_correct) == 0 or len(d_incorrect) == 0:
        return {"error": "one class empty"}
    # Alternative: incorrect dispersion greater than correct.
    u_stat, p_greater = mannwhitneyu(d_incorrect, d_correct, alternative="greater")
    _, p_two = mannwhitneyu(d_incorrect, d_correct, alternative="two-sided")
    n1, n2 = len(d_incorrect), len(d_correct)
    rank_biserial = float(2.0 * u_stat / (n1 * n2) - 1.0)  # +1 incorrect>correct
    incorrect_label = (~correct)
    return {
        "median_dist_correct": float(np.median(d_correct)),
        "median_dist_incorrect": float(np.median(d_incorrect)),
        "mean_dist_correct": float(d_correct.mean()),
        "mean_dist_incorrect": float(d_incorrect.mean()),
        "mann_whitney_u": float(u_stat),
        "p_incorrect_greater": float(p_greater),
        "p_two_sided": float(p_two),
        "rank_biserial": rank_biserial,
        "cohens_d_incorrect_minus_correct": cohens_d(d_incorrect, d_correct),
        "auroc_dist_predicts_incorrect": auroc(dist, incorrect_label),
    }


def histogram(dist: np.ndarray, correct: np.ndarray) -> dict:
    lo, hi = float(dist.min()), float(dist.max())
    edges = np.linspace(lo, hi, N_BINS + 1)
    counts_correct, _ = np.histogram(dist[correct], bins=edges)
    counts_incorrect, _ = np.histogram(dist[~correct], bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "counts_correct": counts_correct.astype(int).tolist(),
        "counts_incorrect": counts_incorrect.astype(int).tolist(),
    }


def verdict(t: dict) -> dict:
    """Both D²HScore and F-4 predict dispersion(correct) < dispersion(incorrect).

    AGREE if the observed median/mean dispersion is lower for correct AND the
    one-sided 'incorrect greater' test is significant at 0.05.
    """
    if "error" in t:
        return {"sign_agreement": "UNDEFINED", "note": t["error"]}
    correct_lower = (t["median_dist_correct"] < t["median_dist_incorrect"]) and (
        t["mean_dist_correct"] < t["mean_dist_incorrect"]
    )
    significant = t["p_incorrect_greater"] < 0.05
    if correct_lower and significant:
        agreement = "AGREE"
    elif correct_lower and not significant:
        agreement = "AGREE_WEAK"
    else:
        agreement = "DISAGREE"
    return {
        "sign_agreement": agreement,
        "correct_has_lower_dispersion": bool(correct_lower),
        "significant_at_0.05": bool(significant),
        "interpretation": (
            "Both D2HScore (hallucinations higher dispersion) and F-4 (correct "
            "collapse harder) predict dispersion(correct) < dispersion(incorrect). "
            f"Observed agreement: {agreement}."
        ),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    if "prefill" not in blob or "correct" not in blob:
        print("MISSING_REGEN_INPUT", CACHE, "(missing prefill/correct keys)", file=sys.stderr)
        return 2

    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64) if "seq_len" in blob else None
    if X.shape != (500, 1536) or correct.shape != (500,):
        print("MISSING_REGEN_INPUT", CACHE, f"(unexpected shapes {X.shape} {correct.shape})", file=sys.stderr)
        return 2

    n, d = X.shape
    centroid = X.mean(axis=0)
    Xc = X - centroid

    # Euclidean centroid-distance per problem.
    dist_euclid = np.sqrt((Xc ** 2).sum(axis=1))

    # Ridge-Mahalanobis centroid-distance (covariance-whitened dispersion).
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    ridge = RIDGE_REL * float(np.trace(Sigma)) / d
    Sigma_r = Sigma + ridge * np.eye(d)
    sol = np.linalg.solve(Sigma_r, Xc.T)  # (d, n)
    dist_mahal = np.sqrt(np.einsum("ij,ji->i", Xc, sol))

    out = {
        "experiment": "P11-FE544",
        "n_problems": int(n),
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "data_note": (
            "Cache stores one L19 final-prefill-token vector per problem; "
            "dispersion computed as distance from population centroid (no "
            "per-token trajectory / attention weights available)."
        ),
        "euclidean": {
            "rank_test": rank_test(dist_euclid, correct),
            "histogram": histogram(dist_euclid, correct),
            "verdict": verdict(rank_test(dist_euclid, correct)),
        },
        "mahalanobis_ridge": {
            "ridge_rel": RIDGE_REL,
            "ridge_abs": float(ridge),
            "rank_test": rank_test(dist_mahal, correct),
            "histogram": histogram(dist_mahal, correct),
            "verdict": verdict(rank_test(dist_mahal, correct)),
        },
    }

    # Robustness variant: seq_len-normalized Euclidean (length as attn stand-in).
    if seq_len is not None and np.all(seq_len > 0):
        dist_norm = dist_euclid / seq_len
        out["euclidean_seqlen_normalized"] = {
            "rank_test": rank_test(dist_norm, correct),
            "verdict": verdict(rank_test(dist_norm, correct)),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())