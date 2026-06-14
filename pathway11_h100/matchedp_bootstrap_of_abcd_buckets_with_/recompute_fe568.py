"""P11-FE568 — Matched-|P| bootstrap of A/B/C/D buckets with γ_naive.

Pre-test for F-7. F-7 anchors on the D-bucket (K=1 right, K=8 wrong) showing a
distinctive *concentrated* geometric signature in the L19 prefill stream. D is
by far the smallest of the four buckets. Chun 2509.26560's harmonic-mean law
bounds the naive participation ratio γ_naive ≲ min(P, Q, γ): D's tiny sample
count P alone would force its naive PR to look "concentrated" even if its true γ
matched A/B/C. This script bootstraps A, B, C down to |D| (matched-|P|), with no
estimator change (γ_naive only), and recomputes peak γ_naive per bucket. If A/B/C
γ_naive collapses to D's level once sample-size-matched, F-7's distinctive
D-bucket signature is a small-N artifact and the full γ_both bias-correction
pipeline can be deprioritized.

Buckets, by (K=1 correct, K=8 majority correct):
  A: K1 wrong, K8 wrong       C: K1 right, K8 right
  B: K1 wrong, K8 right       D: K1 right, K8 wrong   <- distinctive

γ_naive(bucket) = (Σλ)² / Σλ²  (participation ratio of the centered covariance
spectrum), computed from SVD of the centered prefill matrix. Only the L19
prefill layer is cached, so "per layer" reduces to the one cached layer here.
7B is processed if a 7B prefill cache + K=8 dir are present; otherwise skipped.
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
# Optional 7B inputs (schema-unspecified; processed only if present).
CACHE_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
K8_DIR_7B = ROOT / "pathway11_h100/data/k8_selfconsistency_7b"

OUT_JSON = ROOT / "pathway11_h100/matched_p_bootstrap/results.json"

SEED = 9999
N_BOOT = 1000
K8_MAJORITY_MIN = 5  # of 8 generations correct -> K8 "right"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def participation_ratio(X: np.ndarray) -> float:
    """γ_naive = (Σλ)²/Σλ² of the centered-covariance eigenspectrum.

    Computed from singular values of the centered matrix:
    λ_i = s_i² / (n-1). The (n-1)² scaling cancels in the ratio, so PR is taken
    directly from s². Returns nan for n < 2.
    """
    n = X.shape[0]
    if n < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s.astype(np.float64) ** 2
    denom = float((lam ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return float((lam.sum() ** 2) / denom)


def load_k8_majority(k8_dir: Path, n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (k8_right, valid) boolean arrays over problem index 0..n_expected-1.

    k8_right[i] = (#correct of 8 >= K8_MAJORITY_MIN). valid[i] = file was found
    and well-formed. Problems with missing/malformed K=8 files are excluded from
    bucketing via valid=False.
    """
    k8_right = np.zeros(n_expected, dtype=bool)
    valid = np.zeros(n_expected, dtype=bool)
    for i in range(n_expected):
        f = k8_dir / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        try:
            c = np.load(f)["correct"].astype(bool)
        except Exception:
            continue
        if c.size == 0:
            continue
        valid[i] = True
        k8_right[i] = int(c.sum()) >= K8_MAJORITY_MIN
    return k8_right, valid


def bucket_masks(k1_right: np.ndarray, k8_right: np.ndarray, valid: np.ndarray) -> dict:
    """A/B/C/D buckets restricted to problems with valid K=8 labels."""
    v = valid
    return {
        "A": v & ~k1_right & ~k8_right,
        "B": v & ~k1_right & k8_right,
        "C": v & k1_right & k8_right,
        "D": v & k1_right & ~k8_right,
    }


def bootstrap_matched(X: np.ndarray, target_n: int, n_boot: int, seed: int) -> dict:
    """Subsample (without replacement) to target_n and recompute γ_naive.

    Without-replacement matching keeps P (distinct points) — and hence the rank
    cap that drives the harmonic-mean bound — exactly equal to |D|.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < target_n or target_n < 2:
        return {"n_boot": 0, "gamma_naive_mean": float("nan"),
                "gamma_naive_std": float("nan"), "ci2.5": float("nan"),
                "ci97.5": float("nan"), "samples": []}
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n, size=target_n, replace=False)
        vals[b] = participation_ratio(X[idx])
    return {
        "n_boot": int(n_boot),
        "target_n": int(target_n),
        "gamma_naive_mean": float(np.nanmean(vals)),
        "gamma_naive_std": float(np.nanstd(vals)),
        "ci2.5": float(np.nanpercentile(vals, 2.5)),
        "ci97.5": float(np.nanpercentile(vals, 97.5)),
    }


def analyze_model(name: str, prefill: np.ndarray, k1_right: np.ndarray,
                  k8_dir: Path) -> dict:
    n = prefill.shape[0]
    k8_right, valid = load_k8_majority(k8_dir, n)
    if valid.sum() == 0:
        return {"status": "MISSING_K8", "n_valid": 0}

    masks = bucket_masks(k1_right, k8_right, valid)
    sizes = {b: int(m.sum()) for b, m in masks.items()}
    d_size = sizes["D"]

    full = {b: participation_ratio(prefill[m]) for b, m in masks.items()}

    out = {
        "status": "OK",
        "model": name,
        "n_total": int(n),
        "n_valid": int(valid.sum()),
        "bucket_sizes": sizes,
        "gamma_naive_full": {b: full[b] for b in ("A", "B", "C", "D")},
        "matched_p_bootstrap": {},
        "verdict_per_bucket": {},
    }

    if d_size < 2:
        out["status"] = "D_TOO_SMALL"
        return out

    gamma_d_full = full["D"]
    overlaps = []
    for b in ("A", "B", "C"):
        bs = bootstrap_matched(prefill[masks[b]], d_size, N_BOOT, SEED + ord(b))
        out["matched_p_bootstrap"][b] = bs
        lo, hi = bs.get("ci2.5", float("nan")), bs.get("ci97.5", float("nan"))
        overlap = bool(np.isfinite(lo) and np.isfinite(hi) and
                       lo <= gamma_d_full <= hi)
        out["verdict_per_bucket"][b] = {
            "matched_gamma_mean": bs.get("gamma_naive_mean"),
            "d_full_gamma": gamma_d_full,
            "d_in_matched_ci": overlap,
        }
        overlaps.append(overlap)

    out["all_buckets_match_D_after_matched_P"] = bool(all(overlaps))
    out["verdict"] = (
        "F7_LIKELY_SMALL_N_ARTIFACT" if all(overlaps)
        else "F7_D_SIGNATURE_SURVIVES_MATCHED_P"
    )
    return out


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2
    if not K8_DIR_15B.exists():
        print("MISSING_REGEN_INPUT", K8_DIR_15B, file=sys.stderr)
        return 2

    blob = np.load(CACHE_15B)
    X15 = blob["prefill"].astype(np.float64)
    k1_15 = blob["correct"].astype(bool)
    assert X15.shape == (500, 1536) and k1_15.shape == (500,)

    results = {
        "experiment": "P11-FE568",
        "description": "Matched-|P| bootstrap of A/B/C/D buckets, γ_naive only.",
        "k8_majority_min": K8_MAJORITY_MIN,
        "n_boot": N_BOOT,
        "seed": SEED,
        "models": {},
    }

    results["models"]["qwen_1.5b"] = analyze_model("qwen_1.5b", X15, k1_15, K8_DIR_15B)
    # Sanity anchor: K=1 DoM-free check that buckets read the right labels.
    results["k1_accuracy_1.5b"] = float(k1_15.mean())

    # Optional 7B pass.
    if CACHE_7B.exists() and K8_DIR_7B.exists():
        blob7 = np.load(CACHE_7B)
        X7 = blob7["prefill"].astype(np.float64)
        k1_7 = blob7["correct"].astype(bool)
        results["models"]["qwen_7b"] = analyze_model("qwen_7b", X7, k1_7, K8_DIR_7B)
        results["k1_accuracy_7b"] = float(k1_7.mean())
    else:
        results["models"]["qwen_7b"] = {"status": "MISSING_CACHE_SKIPPED"}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())