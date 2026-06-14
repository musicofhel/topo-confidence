"""P11-FE1219 — ε_latent geometry classification of L19 prefill latent space.

Background. A prior paper classifies Qwen-2.5-1.5B *embedding* space as
"Class 1" (ε = 0.26, indeterminate linear-representation capacity). Its key
open question is whether ε generalizes from embeddings to *latents*. This
script answers that question for our model by computing an analogous ε from the
L19 prefill residual stream and classifying the latent space:

  * Class 1 (LRH weakened)  — ε_latent is statistically indistinguishable from
    or worse than the embedding anchor 0.26: the latent space is no more
    linearly structured than the embeddings, so the LRH reading of DoM (F-2)
    gains no extra support.
  * Class 2 (LRH rescued)   — ε_latent is meaningfully below 0.26: the latent
    space has cleaner linear structure than the embeddings.

ε definition (covariance-spectrum capacity, documented here so the number is
reinterpretable). Let Σ be the sample covariance of the centered L19 prefill
activations with eigenvalues λ_1 ≥ … ≥ λ_d ≥ 0. The participation ratio
PR = (Σλ)² / Σλ² is the effective number of variance-carrying directions, and
ε = PR / d ∈ (0, 1] is its fraction of the ambient dimension. Low ε ⇒ variance
concentrated in few directions ⇒ a single linear axis (DoM) plausibly captures
the dominant concept (good LRH structure); ε near the embedding anchor ⇒ the
diffuse, indeterminate regime. A nonparametric bootstrap over samples gives a
95% CI used for the class decision.
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
OUT_JSON = ROOT / "pathway11_h100/eps_latent_classification/results.json"

SEED = 9999
N_BOOT = 2000
PAPER_EMBED_EPS = 0.26  # Qwen-2.5-1.5B embedding-space ε (Class 1, indeterminate)
MARGIN = 0.05           # decision band around the embedding anchor


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def participation_ratio(eigvals: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² over the nonnegative eigenvalue spectrum."""
    lam = np.clip(eigvals, 0.0, None)
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def eps_from_X(X: np.ndarray) -> float:
    """ε = participation_ratio(cov(X)) / d on centered activations."""
    d = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    Sigma = (Xc.T @ Xc) / max(n - 1, 1)
    eigvals = np.linalg.eigvalsh(Sigma)
    return participation_ratio(eigvals) / d


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n, d = X.shape

    # Point estimate of ε on the full latent space.
    eps_latent = eps_from_X(X)

    # Auxiliary spectral diagnostics for human reinterpretation.
    Xc = X - X.mean(axis=0, keepdims=True)
    Sigma = (Xc.T @ Xc) / max(n - 1, 1)
    eigvals = np.clip(np.linalg.eigvalsh(Sigma), 0.0, None)[::-1]
    total = float(eigvals.sum())
    pr = participation_ratio(eigvals)
    top1_frac = float(eigvals[0] / total) if total > 0 else float("nan")
    anisotropy = 1.0 - top1_frac  # 1 - λ1/Σλ

    # DoM AUROC sanity anchor (F-2 readout this classification bears on).
    dom_auroc = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auroc = auroc(dom_score, y)

    # Nonparametric bootstrap over samples for a 95% CI on ε.
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot[b] = eps_from_X(X[idx])
    ci_lo, ci_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    # Classification vs the embedding anchor.
    if eps_latent < PAPER_EMBED_EPS - MARGIN:
        latent_class = "Class 2"
        lrh_verdict = "rescued"
        interpretation = (
            "ε_latent is meaningfully below the embedding anchor (0.26): the "
            "L19 latent space is more linearly structured than embeddings, "
            "supporting the LRH reading of DoM (F-2)."
        )
    elif eps_latent > PAPER_EMBED_EPS + MARGIN:
        latent_class = "Class 1"
        lrh_verdict = "weakened"
        interpretation = (
            "ε_latent is above the embedding anchor (0.26): the latent space is "
            "even more diffuse than embeddings — the LRH reading of DoM (F-2) "
            "is weakened."
        )
    else:
        latent_class = "Class 1"
        lrh_verdict = "weakened"
        interpretation = (
            "ε_latent is statistically indistinguishable from the embedding "
            "anchor (0.26): ε generalizes from embeddings to latents, so the "
            "latent space is no better structured and the LRH reading of DoM "
            "(F-2) is weakened."
        )

    # CI-based robustness flag: does the anchor fall inside the bootstrap CI?
    anchor_in_ci = bool(ci_lo <= PAPER_EMBED_EPS <= ci_hi)

    out = {
        "experiment": "P11-FE1219",
        "n_samples": int(n),
        "ambient_dim": int(d),
        "eps_latent": float(eps_latent),
        "eps_latent_ci95": [ci_lo, ci_hi],
        "eps_definition": "participation_ratio(cov) / ambient_dim",
        "participation_ratio": float(pr),
        "top1_eigval_fraction": top1_frac,
        "anisotropy_1_minus_top1": float(anisotropy),
        "paper_embedding_eps": PAPER_EMBED_EPS,
        "decision_margin": MARGIN,
        "anchor_in_bootstrap_ci": anchor_in_ci,
        "latent_class": latent_class,
        "lrh_verdict": lrh_verdict,
        "interpretation": interpretation,
        "dom_auroc_sanity": dom_auroc,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())