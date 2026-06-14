"""P11-FE566 — Bias-corrected, sample-matched participation-ratio (PR) inversion
refutation test for F-6 (prefill PR inversion) and F-4 (asymmetric collapse).

F-6 compares peak participation ratio between the correct and incorrect prefill
groups, which have very unequal sizes (e.g. 7B: 366 vs 134). Chun 2509.26560's
harmonic-mean law  γ_naive ≈ 1 / (1/P + 1/Q + 1/γ)  (P = #samples, Q = #features,
γ = true PR) implies the *naive* PR floor differs between groups of unequal size
even when the underlying geometry is identical — so a raw PR gap can be a pure
sample-asymmetry artifact.

This script, following the badooki/dimensionality bias-corrected estimator:
  • computes naive PR per group via the n×n centred Gram trick
        PR = trace(G)^2 / ||G||_F^2 ,  G = Xc Xc^T   (= (Σλ)^2 / Σλ^2)
  • applies two corrections from the harmonic-mean law:
        γ_row  : 1/γ = 1/γ_naive - 1/P            (sample-size correction only)
        γ_both : 1/γ = 1/γ_naive - 1/P - 1/Q      (sample + feature correction)
  • bootstraps both groups down to a matched P = min(n_c, n_i) and re-tests the
    correct-vs-incorrect PR gap under naive and both corrected estimators.

Refutation rule (F-6 specifically): the inversion is an artifact if its sign
vanishes under EITHER matched-N OR bias correction. It is robust only if it
survives both.
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
CACHE_15B = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# 7B prefill cache is optional — probe a few plausible paths; skip if absent.
CACHE_7B_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill_7b.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m72b_prefill.npz",
]
OUT_JSON = ROOT / "pathway11_h100/pr_bias_correction_pergroup/results.json"

SEED = 9999
N_BOOT = 2000


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def naive_pr(X: np.ndarray) -> float:
    """Naive participation ratio via the centred n×n Gram matrix.

    PR = (Σ λ_i)^2 / Σ λ_i^2 = trace(Σ)^2 / ||Σ||_F^2 = trace(G)^2 / ||G||_F^2
    where G = Xc Xc^T shares the nonzero spectrum of (n-1)Σ. Avoids the 1536×1536
    eigendecomposition entirely, so bootstrap is cheap.
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0)
    G = Xc @ Xc.T
    tr = float(np.trace(G))
    fro2 = float(np.sum(G * G))
    if fro2 <= 0.0:
        return float("nan")
    return tr * tr / fro2


def correct_pr(pr_naive: float, n_samples: int, n_features: int) -> tuple[float, float]:
    """Return (γ_row, γ_both) bias-corrected PRs from the harmonic-mean law.

    1/γ_row  = 1/γ_naive - 1/P
    1/γ_both = 1/γ_naive - 1/P - 1/Q
    A non-positive corrected inverse means the naive estimate is at/below the
    finite-size floor — the geometry is unresolved at this N — reported as +inf.
    """
    if not np.isfinite(pr_naive) or pr_naive <= 0:
        return float("nan"), float("nan")
    inv = 1.0 / pr_naive
    inv_row = inv - 1.0 / n_samples
    inv_both = inv - 1.0 / n_samples - 1.0 / n_features
    g_row = (1.0 / inv_row) if inv_row > 0 else float("inf")
    g_both = (1.0 / inv_both) if inv_both > 0 else float("inf")
    return float(g_row), float(g_both)


def hm_floor(n_samples: int, n_features: int) -> float:
    """Naive-PR floor as γ_true → ∞: γ_naive → 1/(1/P + 1/Q) = P·Q/(P+Q)."""
    return float(n_samples * n_features / (n_samples + n_features))


def ci(arr: np.ndarray) -> tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def analyze_model(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    D = X.shape[1]
    Xc_grp = X[y]
    Xi_grp = X[~y]
    n_c, n_i = len(Xc_grp), len(Xi_grp)

    # ---- full-sample (unmatched) estimates -------------------------------
    pr_c = naive_pr(Xc_grp)
    pr_i = naive_pr(Xi_grp)
    g_row_c, g_both_c = correct_pr(pr_c, n_c, D)
    g_row_i, g_both_i = correct_pr(pr_i, n_i, D)

    # full-sample signed gaps (correct - incorrect)
    full_naive_diff = pr_c - pr_i
    full_row_diff = g_row_c - g_row_i
    full_both_diff = g_both_c - g_both_i
    sign0 = float(np.sign(full_naive_diff))  # original F-6 inversion direction

    # ---- matched-P bootstrap --------------------------------------------
    m = min(n_c, n_i)
    rng = np.random.default_rng(SEED)
    naive_d = np.empty(N_BOOT)
    row_d = np.empty(N_BOOT)
    both_d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ic = rng.choice(n_c, m, replace=False)
        ii = rng.choice(n_i, m, replace=False)
        pc = naive_pr(Xc_grp[ic])
        pi = naive_pr(Xi_grp[ii])
        rc, bc = correct_pr(pc, m, D)
        ri, bi = correct_pr(pi, m, D)
        naive_d[b] = pc - pi
        row_d[b] = rc - ri
        both_d[b] = bc - bi

    naive_ci = ci(naive_d)
    row_ci = ci(row_d)
    both_ci = ci(both_d)

    def excludes_zero(c):
        lo, hi = c
        return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))

    def sign_consistency(arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0 or sign0 == 0:
            return float("nan")
        return float(np.mean(np.sign(a) == sign0))

    # ---- refutation verdict (F-6) ---------------------------------------
    survives_matched_n = excludes_zero(naive_ci) and (
        np.sign(np.nanmedian(naive_d)) == sign0
    )
    # bias-correction survival: full-sample γ_both gap keeps the original sign
    # and is finite/non-trivial.
    survives_bias_corr = bool(
        np.isfinite(full_both_diff)
        and sign0 != 0
        and np.sign(full_both_diff) == sign0
    )
    collapses_to_artifact = not (survives_matched_n and survives_bias_corr)

    return {
        "model": name,
        "n_features": int(D),
        "n_correct": int(n_c),
        "n_incorrect": int(n_i),
        "matched_P": int(m),
        "auroc_pr_direction_sanity": None,  # PR is a group statistic, not per-sample
        "full_sample": {
            "pr_naive_correct": pr_c,
            "pr_naive_incorrect": pr_i,
            "hm_floor_correct": hm_floor(n_c, D),
            "hm_floor_incorrect": hm_floor(n_i, D),
            "gamma_row_correct": g_row_c,
            "gamma_row_incorrect": g_row_i,
            "gamma_both_correct": g_both_c,
            "gamma_both_incorrect": g_both_i,
            "naive_gap_correct_minus_incorrect": full_naive_diff,
            "gamma_row_gap": full_row_diff,
            "gamma_both_gap": full_both_diff,
            "inversion_sign": sign0,
        },
        "matched_bootstrap": {
            "n_boot": N_BOOT,
            "naive_gap_mean": float(np.nanmean(naive_d)),
            "naive_gap_ci95": list(naive_ci),
            "naive_sign_consistency": sign_consistency(naive_d),
            "gamma_row_gap_mean": float(np.nanmean(row_d)),
            "gamma_row_gap_ci95": list(row_ci),
            "gamma_row_sign_consistency": sign_consistency(row_d),
            "gamma_both_gap_mean": float(np.nanmean(both_d)),
            "gamma_both_gap_ci95": list(both_ci),
            "gamma_both_sign_consistency": sign_consistency(both_d),
        },
        "verdict": {
            "survives_matched_N": bool(survives_matched_n),
            "survives_bias_correction": bool(survives_bias_corr),
            "collapses_to_sample_asymmetry_artifact": bool(collapses_to_artifact),
        },
    }


def load_model(path: Path) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(path)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    return X, y


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2

    results = {"experiment": "P11-FE566", "seed": SEED, "models": {}}

    X15, y15 = load_model(CACHE_15B)
    assert X15.shape[1] == 1536, X15.shape
    results["models"]["qwen2.5-1.5b"] = analyze_model(X15, y15, "qwen2.5-1.5b")

    cache_7b = next((p for p in CACHE_7B_CANDIDATES if p.exists()), None)
    if cache_7b is not None:
        X7, y7 = load_model(cache_7b)
        results["models"]["qwen2.5-7b"] = analyze_model(X7, y7, "qwen2.5-7b")
        results["seven_b_cache"] = str(cache_7b)
    else:
        results["seven_b_cache"] = None
        results["seven_b_note"] = (
            "7B prefill cache not found on this machine; 1.5B-only run. "
            "See DATA_MANIFEST.md for 7B regeneration."
        )

    # cross-model summary of the refutation verdict
    results["summary"] = {
        m: r["verdict"]["collapses_to_sample_asymmetry_artifact"]
        for m, r in results["models"].items()
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())