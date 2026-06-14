"""P11-FE570 — γ_row / γ_both bias-corrected peak participation-ratio (PR) per group.

Refutation test for F-6 (prefill PR inversion) and re-test of F-4 (asymmetric
collapse). F-6 compares peak PR between the correct and incorrect groups, which
have very unequal sizes (1.5B 243/257; 7B 366/134). Chun 2509.26560's
harmonic-mean law, γ_naive ≈ 1/(1/P + 1/Q + 1/γ), implies the naive PR floor
differs between groups of different N even when the true γ is identical — so a
raw PR "inversion" could be a pure sample-asymmetry artifact.

This script, per model:
  1. computes the naive participation ratio per group (γ_naive),
  2. bias-corrects it two ways — γ_row (sample-count P only) and
     γ_both (sample-count P and ambient dim Q), following the badooki/
     dimensionality harmonic-mean reference relation,
  3. bootstraps both groups to a matched P (= min group size) and re-tests the
     inversion under naive, γ_row and γ_both estimators.

Verdict: if the correct-vs-incorrect PR inversion vanishes under either
matched-N bootstrap or bias correction, the F-6 geometric story collapses to a
sample-asymmetry artifact.
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
# 7B prefill cache is optional — the asymmetric 366/134 split lives here when present.
CACHE_7B_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill_l19.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/q7b_prefill.npz",
]
OUT_JSON = ROOT / "pathway11_h100/gamma_pergroup_bias/results.json"

SEED = 9999
N_BOOT = 1000


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def participation_ratio(X: np.ndarray) -> float:
    """Naive participation ratio (effective dimension) of the covariance of X.

    PR = (Σ λ_i)^2 / Σ λ_i^2 over the eigenvalues of the sample covariance.
    Computed via the (n×n) Gram matrix, which shares the nonzero spectrum with
    the (d×d) covariance and is far cheaper for d=1536, n≲500.
    """
    n = X.shape[0]
    if n < 3:
        return float("nan")
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T) / (n - 1)
    eig = np.linalg.eigvalsh(G)
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return float("nan")
    s1 = float(eig.sum())
    s2 = float((eig * eig).sum())
    if s2 <= 0:
        return float("nan")
    return s1 * s1 / s2


def correct_pr(gamma_naive: float, P: int, Q: int | None) -> float:
    """Harmonic-mean bias correction (Chun 2509.26560 / badooki reference).

    γ_naive ≈ 1/(1/P + 1/Q + 1/γ)  ⇒  1/γ = 1/γ_naive − 1/P [− 1/Q].
    γ_row corrects for the finite sample count P only (Q=None);
    γ_both additionally corrects for the ambient dimension Q.
    Returns inf if the correction over-shoots (1/γ ≤ 0), nan if γ_naive invalid.
    """
    if not np.isfinite(gamma_naive) or gamma_naive <= 0:
        return float("nan")
    inv = 1.0 / gamma_naive - 1.0 / P
    if Q is not None:
        inv -= 1.0 / Q
    if inv <= 0:
        return float("inf")
    return 1.0 / inv


def _signed(diff: float) -> int:
    if not np.isfinite(diff) or diff == 0:
        return 0
    return 1 if diff > 0 else -1


def process_model(X: np.ndarray, y: np.ndarray, model: str) -> dict:
    D = X.shape[1]
    Xc = X[y]
    Xi = X[~y]
    n_c, n_i = len(Xc), len(Xi)

    # ---- full-sample naive + corrected PR per group ----
    gn_c = participation_ratio(Xc)
    gn_i = participation_ratio(Xi)
    gr_c = correct_pr(gn_c, n_c, None)
    gr_i = correct_pr(gn_i, n_i, None)
    gb_c = correct_pr(gn_c, n_c, D)
    gb_i = correct_pr(gn_i, n_i, D)

    def inv_block(c, i):
        diff = float(i - c)  # inversion = incorrect − correct
        return {"correct": float(c), "incorrect": float(i),
                "inversion_incorrect_minus_correct": diff,
                "inversion_sign": _signed(diff)}

    full = {
        "pr_naive": inv_block(gn_c, gn_i),
        "pr_row": inv_block(gr_c, gr_i),
        "pr_both": inv_block(gb_c, gb_i),
    }

    # ---- matched-N bootstrap (resample both groups to m = min group size) ----
    rng = np.random.default_rng(SEED)
    m = min(n_c, n_i)
    boot = {k: {"naive": [], "row": [], "both": []} for k in ("correct", "incorrect")}
    for _ in range(N_BOOT):
        ic = rng.integers(0, n_c, size=m)
        ii = rng.integers(0, n_i, size=m)
        gnc = participation_ratio(Xc[ic])
        gni = participation_ratio(Xi[ii])
        boot["correct"]["naive"].append(gnc)
        boot["incorrect"]["naive"].append(gni)
        boot["correct"]["row"].append(correct_pr(gnc, m, None))
        boot["incorrect"]["row"].append(correct_pr(gni, m, None))
        boot["correct"]["both"].append(correct_pr(gnc, m, D))
        boot["incorrect"]["both"].append(correct_pr(gni, m, D))

    matched = {"m": int(m), "n_boot": int(N_BOOT)}
    survives = {}
    for est in ("naive", "row", "both"):
        c = np.asarray(boot["correct"][est], dtype=np.float64)
        i = np.asarray(boot["incorrect"][est], dtype=np.float64)
        diff = i - c
        finite = np.isfinite(diff)
        diff_f = diff[finite]
        full_sign = full[f"pr_{est}"]["inversion_sign"]
        if diff_f.size:
            lo, hi = np.percentile(diff_f, [2.5, 97.5])
            mean_diff = float(np.mean(diff_f))
            frac_same = float(np.mean(_vsign(diff_f) == full_sign)) if full_sign != 0 else float("nan")
            ci_excludes_zero = bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
        else:
            lo = hi = mean_diff = frac_same = float("nan")
            ci_excludes_zero = False
        matched[f"pr_{est}"] = {
            "correct_mean": float(np.nanmean(c)) if np.isfinite(c).any() else float("nan"),
            "incorrect_mean": float(np.nanmean(i)) if np.isfinite(i).any() else float("nan"),
            "inversion_mean": mean_diff,
            "inversion_ci95": [float(lo), float(hi)],
            "ci_excludes_zero": ci_excludes_zero,
            "frac_bootstraps_matching_full_sign": frac_same,
            "n_finite": int(diff_f.size),
        }
        survives[est] = ci_excludes_zero

    inversion_survives_matched_n_naive = survives["naive"]
    inversion_survives_bias_correction = survives["row"] and survives["both"]
    # F-6 collapses to artifact unless the inversion survives BOTH stresses.
    artifact = not (inversion_survives_matched_n_naive and inversion_survives_bias_correction)

    return {
        "model": model,
        "ambient_dim": int(D),
        "n_correct": int(n_c),
        "n_incorrect": int(n_i),
        "full_sample": full,
        "matched_n_bootstrap": matched,
        "refutation": {
            "inversion_survives_matched_n_naive": bool(inversion_survives_matched_n_naive),
            "inversion_survives_bias_correction_row": bool(survives["row"]),
            "inversion_survives_bias_correction_both": bool(survives["both"]),
            "f6_collapses_to_sample_asymmetry_artifact": bool(artifact),
        },
    }


def _vsign(a: np.ndarray) -> np.ndarray:
    s = np.sign(a)
    s[~np.isfinite(a)] = 0
    return s


def _load_model(npz_path: Path, model: str) -> dict | None:
    if not npz_path.exists():
        return None
    blob = np.load(npz_path)
    if "prefill" not in blob or "correct" not in blob:
        return None
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        return None
    return process_model(X, y, model)


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2

    results = {"experiment": "P11-FE570", "seed": SEED, "models": {}}

    res_15b = _load_model(CACHE_15B, "Qwen2.5-1.5B")
    if res_15b is None:
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2
    results["models"]["1.5B"] = res_15b

    cache_7b = next((p for p in CACHE_7B_CANDIDATES if p.exists()), None)
    if cache_7b is not None:
        res_7b = _load_model(cache_7b, "Qwen2.5-7B")
        if res_7b is not None:
            results["models"]["7B"] = res_7b
            results["cache_7b_path"] = str(cache_7b)
    else:
        results["cache_7b_path"] = None
        results["note_7b"] = (
            "No 7B prefill NPZ found among candidates; 7B 366/134 asymmetry test "
            "skipped. Provide one of: " + ", ".join(str(p) for p in CACHE_7B_CANDIDATES)
        )

    # Cross-model verdict for narrative docs.
    results["verdict"] = {
        m: results["models"][m]["refutation"]["f6_collapses_to_sample_asymmetry_artifact"]
        for m in results["models"]
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())