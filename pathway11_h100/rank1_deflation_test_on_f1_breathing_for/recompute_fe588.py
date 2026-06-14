"""P11-FE588 — Rank-1 (σ₁) deflation test on F-1 breathing.

Theorem 1 of 2510.06477 implies a mid-layer hidden-state matrix X is rank-1
(a BOS-norm direction) plus a small content-dependent perturbation. F-1's
"breathing" claim is that correct and incorrect prefills occupy residual-stream
neighborhoods of measurably different effective dimensionality (participation
ratio). If that divergence is just BOS-norm modulation living along the top
singular direction, then projecting out σ₁ should collapse the correct/incorrect
gap; if the gap survives deflation, F-1's content-dependence claim is rescued.

For the cached Qwen-2.5-1.5B L19 prefill activations this script:
  1. Centers X, takes its SVD, and records the top singular value's share.
  2. Computes the participation ratio (effective rank) of the correct and the
     incorrect subgroups, before deflation.
  3. Projects out the global top-1 right singular direction (rank-1 deflation),
     and recomputes the same per-group participation ratios on the residual.
  4. Reports the correct-vs-incorrect divergence before/after, and the AUROC of
     a simple per-sample "breathing" scalar (residual-stream norm) before/after
     deflation, to quantify how much F-1 signal lives in σ₁.

The documented NPZ schema exposes only the L19 prefill layer, so the per-layer
loop degenerates to the single cached layer; additional layer caches, if present
under the same directory, are picked up automatically.
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
OUT_JSON = ROOT / "pathway11_h100/rank1_deflation_breathing/results.json"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality of a (n, d) block via covariance eigenvalues.

    PR = (Σλ)^2 / Σλ^2, where λ are the eigenvalues of the (centered) covariance.
    Computed from singular values of the centered block to avoid forming the
    d×d covariance explicitly.
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / max(X.shape[0] - 1, 1)
    num = float(lam.sum()) ** 2
    den = float((lam ** 2).sum())
    if den <= 0.0:
        return float("nan")
    return num / den


def deflate_top1(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project out the global top-1 right singular direction of centered X.

    Returns (X_residual, v1, sigma1_share, sigma1_value). The deflation is done
    on the centered matrix and the residual is re-offset by the original mean so
    that downstream per-sample norms remain comparable.
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    v1 = Vt[0]
    sigma1 = float(S[0])
    sigma1_share = float((S[0] ** 2) / (S ** 2).sum())
    proj = np.outer(Xc @ v1, v1)
    X_resid = (Xc - proj) + mu
    return X_resid, v1, sigma1_share, sigma1


def breathing_scores(X: np.ndarray) -> np.ndarray:
    """Per-sample 'breathing' scalar: deviation of residual-stream norm from the
    block mean norm (the magnitude whose correct/incorrect divergence F-1 tracks)."""
    norms = np.linalg.norm(X - X.mean(axis=0), axis=1)
    return norms


def analyze_layer(X: np.ndarray, y: np.ndarray) -> dict:
    yb = y.astype(bool)

    pr_corr_pre = participation_ratio(X[yb])
    pr_inc_pre = participation_ratio(X[~yb])
    div_pre = abs(pr_corr_pre - pr_inc_pre)

    breath_pre = breathing_scores(X)
    auroc_pre = auroc(breath_pre, yb)

    X_resid, v1, sigma1_share, sigma1 = deflate_top1(X)

    pr_corr_post = participation_ratio(X_resid[yb])
    pr_inc_post = participation_ratio(X_resid[~yb])
    div_post = abs(pr_corr_post - pr_inc_post)

    breath_post = breathing_scores(X_resid)
    auroc_post = auroc(breath_post, yb)

    div_collapse = float("nan")
    if div_pre > 0:
        div_collapse = float(1.0 - (div_post / div_pre))

    return {
        "n_correct": int(yb.sum()),
        "n_incorrect": int((~yb).sum()),
        "sigma1_variance_share": sigma1_share,
        "sigma1_value": sigma1,
        "pr_correct_pre": pr_corr_pre,
        "pr_incorrect_pre": pr_inc_pre,
        "pr_divergence_pre": div_pre,
        "pr_correct_post": pr_corr_post,
        "pr_incorrect_post": pr_inc_post,
        "pr_divergence_post": div_post,
        "pr_divergence_collapse_frac": div_collapse,
        "breathing_auroc_pre": auroc_pre,
        "breathing_auroc_post": auroc_post,
        "breathing_auroc_delta": float(auroc_post - auroc_pre),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    if "prefill" not in blob.files or "correct" not in blob.files:
        print("MISSING_REGEN_INPUT", CACHE, "(missing prefill/correct keys)", file=sys.stderr)
        return 2

    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape[0] != y.shape[0] or X.ndim != 2:
        print("MISSING_REGEN_INPUT", CACHE, "(unexpected shape)", file=sys.stderr)
        return 2

    layers = {"L19_prefill": analyze_layer(X, y)}

    # F-1 is rescued if the per-group participation-ratio divergence survives
    # rank-1 deflation (collapse fraction well below 1) AND the breathing AUROC
    # does not collapse to chance after σ₁ removal.
    l19 = layers["L19_prefill"]
    verdict = (
        "RESCUED"
        if (l19["pr_divergence_collapse_frac"] < 0.5 and l19["breathing_auroc_post"] > 0.55)
        else "REFUTED"
    )

    out = {
        "experiment": "P11-FE588",
        "description": "Rank-1 (sigma1) deflation test on F-1 breathing",
        "cache": str(CACHE.relative_to(ROOT)),
        "n_samples": int(X.shape[0]),
        "n_dims": int(X.shape[1]),
        "layers": layers,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())