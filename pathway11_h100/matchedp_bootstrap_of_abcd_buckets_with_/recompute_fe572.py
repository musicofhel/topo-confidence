"""P11-FE572 — Matched-|P| bootstrap of A/B/C/D buckets with naive participation
ratio (γ_naive), no estimator change.

Cheap pre-test for F-7. F-7 anchors on the D-bucket (K=1 right / K=8 wrong)
showing a distinctive *concentrated* geometric signature. D is by far the
smallest of the four buckets. Chun 2509.26560's harmonic-mean law bounds
γ_naive above by ~min(P, Q, γ), so a tiny |P| alone forces a bucket's naive
participation ratio to look "concentrated" even when its true γ matches the
others.

This script defines the four buckets from K=1 correctness (prefill cache) and
K=8 majority correctness (self-consistency cache), then for each bucket
recomputes γ_naive of the L19 prefill covariance after bootstrap-resampling
*down to the D-bucket size*. If, at matched |P|, A/B/C bootstrap CIs overlap
D's, the "distinctive D-bucket signature" is a small-N artifact and the full
γ_both bias-correction pipeline for F-7 can be deprioritized.

γ_naive(X) = (Σ λ_i)^2 / Σ λ_i^2  over the eigenvalues of the (centered)
sample covariance — the naive participation ratio, no bias correction.
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
K8_DIR_15B = ROOT / "pathway11_h100/data/k8_selfconsistency"
# 7B caches are optional — recorded as missing if absent rather than aborting.
CACHE_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
K8_DIR_7B = ROOT / "pathway11_h100/data/k8_selfconsistency_7b"
OUT_JSON = ROOT / "pathway11_h100/matchedp_bootstrap_abcd/results.json"

SEED = 9999
N_BOOT = 2000
LAYER = 19  # only L19 prefill is cached; "peak per layer" reduces to this layer


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def gamma_naive(X: np.ndarray) -> float:
    """Naive participation ratio of the centered sample covariance of X (m, d).

    PR = (Σ λ)^2 / Σ λ^2. Computed from singular values of the centered matrix
    (λ ∝ s^2); the (m-1) scale cancels in the ratio.
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = s.astype(np.float64) ** 2
    denom = float((lam ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return float((lam.sum() ** 2) / denom)


def load_k8_majority(k8_dir: Path, n: int) -> np.ndarray | None:
    """K=8 majority correctness per problem, aligned to prefill row order.

    Returns a length-n float array with NaN where the problem_NNN.npz is
    missing, or None if the directory itself is absent.
    """
    if not k8_dir.exists():
        return None
    maj = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        f = k8_dir / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        c = np.load(f)["correct"].astype(bool)
        if c.size == 0:
            continue
        maj[i] = 1.0 if c.mean() > 0.5 else 0.0
    return maj


def bucket_masks(k1: np.ndarray, k8: np.ndarray) -> dict[str, np.ndarray]:
    """A=right/right, B=wrong/right, C=wrong/wrong, D=right/wrong (K=1 / K=8)."""
    return {
        "A": k1 & k8,
        "B": (~k1) & k8,
        "C": (~k1) & (~k8),
        "D": k1 & (~k8),
    }


def bootstrap_matched(X: np.ndarray, members: np.ndarray, m_target: int,
                      rng: np.random.Generator) -> dict:
    """Bootstrap γ_naive of bucket `members` resampled (with replacement) to
    size m_target. Returns native γ plus bootstrap mean/std/95% CI."""
    Xb = X[members]
    out = {
        "n_native": int(members.size),
        "gamma_native": gamma_naive(Xb),
        "m_target": int(m_target),
    }
    if members.size < 2 or m_target < 2:
        out.update({"boot_mean": float("nan"), "boot_std": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan")})
        return out
    vals = np.empty(N_BOOT, dtype=np.float64)
    n_b = members.size
    for b in range(N_BOOT):
        idx = rng.integers(0, n_b, size=m_target)
        vals[b] = gamma_naive(Xb[idx])
    vals = vals[np.isfinite(vals)]
    out.update({
        "boot_mean": float(vals.mean()),
        "boot_std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
    })
    return out


def ci_overlap(a: dict, b: dict) -> bool:
    if not all(np.isfinite([a["ci_lo"], a["ci_hi"], b["ci_lo"], b["ci_hi"]])):
        return False
    return (a["ci_lo"] <= b["ci_hi"]) and (b["ci_lo"] <= a["ci_hi"])


def analyse_model(prefill: np.ndarray, k1: np.ndarray, k8: np.ndarray,
                  rng: np.random.Generator) -> dict:
    valid = np.isfinite(k8)
    X = prefill[valid]
    k1v = k1[valid]
    k8v = k8[valid].astype(bool)

    masks = bucket_masks(k1v, k8v)
    members = {b: np.flatnonzero(m) for b, m in masks.items()}
    m_D = int(members["D"].size)

    res = {
        "layer": LAYER,
        "n_used": int(valid.sum()),
        "n_dropped_missing_k8": int((~valid).sum()),
        "bucket_sizes": {b: int(v.size) for b, v in members.items()},
        "matched_to_D_size": m_D,
        "buckets": {},
    }

    if m_D < 2:
        res["verdict"] = "INSUFFICIENT_D"
        res["note"] = "D-bucket has < 2 members; matched bootstrap undefined."
        return res

    for b, mem in members.items():
        res["buckets"][b] = bootstrap_matched(X, mem, m_D, rng)

    # F-7 pre-test: does D still stand out once A/B/C are sampled at |D|?
    d = res["buckets"]["D"]
    overlaps = {b: ci_overlap(res["buckets"][b], d) for b in ("A", "B", "C")}
    res["matched_ci_overlaps_D"] = overlaps
    all_overlap = all(overlaps.values())
    res["verdict"] = "ARTIFACT_LIKELY" if all_overlap else "DISTINCTIVE_SURVIVES"
    res["interpretation"] = (
        "At matched |P|=|D| the A/B/C participation ratios are statistically "
        "indistinguishable from D's — the 'distinctive D-bucket signature' is a "
        "small-N artifact; deprioritize the full γ_both bias-correction for F-7."
        if all_overlap else
        "D's participation ratio remains separable from at least one of A/B/C "
        "even after matching |P| — the distinctive signature is not purely a "
        "small-N artifact; the full γ_both pipeline for F-7 is still warranted."
    )
    return res


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2
    if not K8_DIR_15B.exists():
        print("MISSING_REGEN_INPUT", K8_DIR_15B, file=sys.stderr)
        return 2

    out: dict = {"experiment": "P11-FE572", "seed": SEED, "n_boot": N_BOOT,
                 "models": {}}

    # ---- Qwen-1.5B (primary) ----
    blob = np.load(CACHE_15B)
    prefill = blob["prefill"].astype(np.float64)
    k1 = blob["correct"].astype(bool)
    assert prefill.shape == (500, 1536) and k1.shape == (500,)
    k8 = load_k8_majority(K8_DIR_15B, prefill.shape[0])
    if k8 is None:
        print("MISSING_REGEN_INPUT", K8_DIR_15B, file=sys.stderr)
        return 2
    rng = np.random.default_rng(SEED)
    out["models"]["1.5B"] = analyse_model(prefill, k1, k8, rng)
    # sanity readout: native DoM AUROC on the used subset (informational)
    out["models"]["1.5B"]["dom_auroc_sanity"] = auroc(
        prefill[np.isfinite(k8)] @ (
            prefill[np.isfinite(k8)][k1[np.isfinite(k8)]].mean(0)
            - prefill[np.isfinite(k8)][~k1[np.isfinite(k8)]].mean(0)),
        k1[np.isfinite(k8)],
    )

    # ---- Qwen-7B (optional) ----
    if CACHE_7B.exists() and K8_DIR_7B.exists():
        b7 = np.load(CACHE_7B)
        pf7 = b7["prefill"].astype(np.float64)
        k1_7 = b7["correct"].astype(bool)
        k8_7 = load_k8_majority(K8_DIR_7B, pf7.shape[0])
        if k8_7 is not None:
            rng7 = np.random.default_rng(SEED + 1)
            out["models"]["7B"] = analyse_model(pf7, k1_7, k8_7, rng7)
        else:
            out["models"]["7B"] = {"status": "MISSING_K8_CACHE"}
    else:
        out["models"]["7B"] = {
            "status": "MISSING_CACHE",
            "note": "No 7B prefill/K=8 cache present locally; 1.5B result stands alone.",
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())