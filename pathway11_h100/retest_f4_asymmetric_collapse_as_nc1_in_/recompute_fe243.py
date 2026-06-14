"""P11-FE243 — Re-test F-4 (asymmetric collapse) as NC1 in disguise.

Computes Wu & Papyan's Class-Distance Normalized Variance (CDNV, eqn 4) on
cached pathway11_h100 final-token L19 activations partitioned into two classes
by ground-truth correctness (correct vs incorrect). The "k=2 augmentation"
estimator splits each class into two disjoint halves and forms a cross
mean/variance estimate (mean from one half, squared deviations referenced to
that mean computed on the other half), averaged over many random splits, to
debias the small-sample variance term.

Within-class variance (Wu & Papyan):
    Var_c = (1/N_c) * sum_i || h_{i,c} - mu_c ||^2          (= trace of class cov, 1/N)
CDNV between classes c, c':
    V_{c,c'} = (Var_c + Var_c') / (2 * || mu_c - mu_c' ||^2)

F-4 demotion test: the correct/incorrect within-class variance RATIO is the
"asymmetric collapse magnitude". If that ratio is the same quantity NC1 would
produce when measured per-correctness-class, F-4 is a special case of generic
neural collapse rather than a reasoning-specific geometric finding.

Activation source: the experiment targets final-token L19 states. If a
final-token cache is present it is used; otherwise the script falls back to the
L19 prefill states in the main cache and records the source in the output.
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
# Candidate final-token L19 caches (used if present; else fall back to prefill).
FINAL_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/data/m15b_final.npz",
]
OUT_JSON = ROOT / "pathway11_h100/neural_collapse_cdnv/results.json"

SEED = 9999
N_SPLITS = 200  # random k=2 augmentation splits for the debiased estimator


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def within_class_var(X: np.ndarray) -> float:
    """Mean squared Euclidean distance of samples to their class mean (1/N)."""
    if X.shape[0] == 0:
        return float("nan")
    mu = X.mean(axis=0)
    return float(np.mean(np.sum((X - mu) ** 2, axis=1)))


def naive_cdnv(Xc: np.ndarray, Xi: np.ndarray) -> dict:
    var_c = within_class_var(Xc)
    var_i = within_class_var(Xi)
    mu_c = Xc.mean(axis=0)
    mu_i = Xi.mean(axis=0)
    dist_sq = float(np.sum((mu_c - mu_i) ** 2))
    cdnv = (var_c + var_i) / (2.0 * dist_sq) if dist_sq > 0 else float("nan")
    return {
        "var_correct": var_c,
        "var_incorrect": var_i,
        "between_class_dist_sq": dist_sq,
        "cdnv": cdnv,
        "var_ratio_correct_over_incorrect": (var_c / var_i) if var_i > 0 else float("nan"),
    }


def k2_cdnv(Xc: np.ndarray, Xi: np.ndarray, n_splits: int, seed: int) -> dict:
    """k=2 augmentation estimator.

    For each class, randomly split into halves A,B. Estimate the class mean from
    half A and reference half B's squared deviations to it (cross estimate);
    symmetrize over A<->B. Between-class distance uses independent halves of the
    two classes to remove the mean-overlap bias. Average over n_splits draws.
    """
    rng = np.random.default_rng(seed)
    var_c_acc = []
    var_i_acc = []
    cdnv_acc = []
    ratio_acc = []

    def split_halves(n):
        idx = rng.permutation(n)
        h = n // 2
        return idx[:h], idx[h:]

    for _ in range(n_splits):
        ca, cb = split_halves(Xc.shape[0])
        ia, ib = split_halves(Xi.shape[0])
        if len(ca) == 0 or len(cb) == 0 or len(ia) == 0 or len(ib) == 0:
            continue

        mu_c_a, mu_c_b = Xc[ca].mean(0), Xc[cb].mean(0)
        mu_i_a, mu_i_b = Xi[ia].mean(0), Xi[ib].mean(0)

        # cross variance: deviations of one half about the OTHER half's mean,
        # symmetrized — unbiased for the true within-class second moment.
        var_c = 0.5 * (
            np.mean(np.sum((Xc[cb] - mu_c_a) ** 2, axis=1))
            + np.mean(np.sum((Xc[ca] - mu_c_b) ** 2, axis=1))
        )
        var_i = 0.5 * (
            np.mean(np.sum((Xi[ib] - mu_i_a) ** 2, axis=1))
            + np.mean(np.sum((Xi[ia] - mu_i_b) ** 2, axis=1))
        )
        # between-class distance from independent halves (debiased).
        dist_sq = 0.5 * (
            np.sum((mu_c_a - mu_i_b) ** 2) + np.sum((mu_c_b - mu_i_a) ** 2)
        )

        var_c_acc.append(float(var_c))
        var_i_acc.append(float(var_i))
        if dist_sq > 0:
            cdnv_acc.append(float((var_c + var_i) / (2.0 * dist_sq)))
        if var_i > 0:
            ratio_acc.append(float(var_c / var_i))

    def agg(a):
        return (float(np.mean(a)), float(np.std(a))) if a else (float("nan"), float("nan"))

    var_c_m, var_c_s = agg(var_c_acc)
    var_i_m, var_i_s = agg(var_i_acc)
    cdnv_m, cdnv_s = agg(cdnv_acc)
    ratio_m, ratio_s = agg(ratio_acc)
    return {
        "n_splits_used": len(cdnv_acc),
        "var_correct_mean": var_c_m,
        "var_correct_std": var_c_s,
        "var_incorrect_mean": var_i_m,
        "var_incorrect_std": var_i_s,
        "cdnv_mean": cdnv_m,
        "cdnv_std": cdnv_s,
        "var_ratio_correct_over_incorrect_mean": ratio_m,
        "var_ratio_correct_over_incorrect_std": ratio_s,
    }


def load_activations() -> tuple[np.ndarray, str]:
    """Return (X, source). Prefer a final-token cache; fall back to prefill."""
    for cand in FINAL_CACHE_CANDIDATES:
        if cand.exists():
            blob = np.load(cand)
            for key in ("final", "final_token", "hidden", "prefill"):
                if key in blob.files:
                    return blob[key].astype(np.float64), f"{cand.name}:{key}"
    # fall back to L19 prefill states in the main cache
    blob = np.load(CACHE)
    return blob["prefill"].astype(np.float64), f"{CACHE.name}:prefill"


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    X, source = load_activations()
    y = np.load(CACHE)["correct"].astype(bool)
    if X.shape[0] != y.shape[0]:
        print("MISSING_REGEN_INPUT shape-mismatch", X.shape, y.shape, file=sys.stderr)
        return 2

    Xc = X[y]      # correct class
    Xi = X[~y]     # incorrect class

    naive = naive_cdnv(Xc, Xi)
    k2 = k2_cdnv(Xc, Xi, N_SPLITS, SEED)

    # Sanity: DoM-style mean-difference AUROC on these activations, as a check
    # that the correctness partition still carries the directional signal.
    d_vec = Xc.mean(0) - Xi.mean(0)
    dom_auroc = auroc(X @ d_vec, y)

    out = {
        "experiment": "P11-FE243",
        "description": "CDNV (Wu&Papyan eqn 4, k=2 augmentation) per-correctness-class; F-4 vs NC1",
        "activation_source": source,
        "n_correct": int(Xc.shape[0]),
        "n_incorrect": int(Xi.shape[0]),
        "dim": int(X.shape[1]),
        "naive_cdnv": naive,
        "k2_augmentation_cdnv": k2,
        "dom_meandiff_auroc": float(dom_auroc),
        # The asymmetric-collapse magnitude under test is the within-class
        # variance ratio (correct collapses more => ratio < 1). If this matches
        # the F-4 observed magnitude, F-4 demotes to a per-class NC1 measurement.
        "asymmetric_collapse_ratio_naive": naive["var_ratio_correct_over_incorrect"],
        "asymmetric_collapse_ratio_k2": k2["var_ratio_correct_over_incorrect_mean"],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())