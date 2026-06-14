"""P11-FE1299 — DySIB MI-saturation intrinsic-dimension estimator vs the F-1
PR/d_eff breathing curve.

F-1 reads "dimensional breathing" off the participation ratio (PR) and the
effective dimension d_eff — both pure second-moment (covariance-eigenvalue)
quantities, hence magnitude-sensitive. DySIB (Fig.3, DPI criterion) instead
estimates intrinsic latent dimension k_z from where an InfoNCE mutual-information
lower bound *saturates* as a function of latent dimension: a second-moment-free
read on the number of genuinely shared dynamical degrees of freedom.

Method per layer: split the residual feature axis into two disjoint random
"views", whiten each view's top-k PCA scores (kills per-dim magnitude), and
measure InfoNCE MI between the paired k-dim projections. MI rises with k then
saturates; k_z is the saturation knee. Averaged over several seeded feature
splits. PR / d_eff are reported alongside for the overlay.

Interpretation: if k_z stays ~constant across layers while PR / d_eff "breathe",
the breathing is a second-moment (variance-rescaling) artifact, not a change in
the number of dynamical DoF. Descriptive geometry only — no claim that geometry
beats free baselines.

Only L19 is present in the canonical cache; the script scans an optional
per-layer cache dir and degrades to the single L19 layer when that is absent.
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
PER_LAYER_DIR = ROOT / "pathway11_h100/prefill_inversion/cache/per_layer"
OUT_JSON = ROOT / "pathway11_h100/dysib_mi_saturation/results.json"

SEED = 9999
N_SPLITS = 4          # seeded random feature-view splits per layer
TAU = 0.1             # InfoNCE temperature (cosine critic)
SAT_FRAC = 0.90       # k_z = smallest k reaching SAT_FRAC of the MI ceiling
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def cov_eigs(X: np.ndarray) -> np.ndarray:
    """Descending covariance eigenvalues (non-negative, clipped)."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    w = np.linalg.eigvalsh(Sigma)
    return np.clip(w[::-1], 0.0, None)


def participation_ratio(eigs: np.ndarray) -> float:
    s1 = float(eigs.sum())
    s2 = float((eigs ** 2).sum())
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def shannon_eff_dim(eigs: np.ndarray) -> float:
    s = float(eigs.sum())
    if s <= 0:
        return float("nan")
    p = eigs / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def dim_for_variance(eigs: np.ndarray, frac: float) -> int:
    s = float(eigs.sum())
    if s <= 0:
        return 0
    c = np.cumsum(eigs) / s
    return int(np.searchsorted(c, frac) + 1)


def pca_scores(X: np.ndarray, kmax: int) -> np.ndarray:
    """Top-kmax PCA scores (N, kmax) via thin SVD on the centered data."""
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S
    return scores[:, :kmax]


def infonce_mi(Za: np.ndarray, Zb: np.ndarray, tau: float) -> float:
    """Symmetric InfoNCE lower bound on I(Za; Zb) in nats (cosine critic)."""
    Za = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12)
    Zb = Zb / (np.linalg.norm(Zb, axis=1, keepdims=True) + 1e-12)
    s = (Za @ Zb.T) / tau
    n = s.shape[0]
    diag = np.diag(s)
    m_row = s.max(axis=1, keepdims=True)
    lse_row = m_row[:, 0] + np.log(np.exp(s - m_row).sum(axis=1))
    m_col = s.max(axis=0, keepdims=True)
    lse_col = m_col[0, :] + np.log(np.exp(s - m_col).sum(axis=0))
    fwd = float(np.mean(diag - lse_row))
    bwd = float(np.mean(diag - lse_col))
    return 0.5 * (fwd + bwd) + float(np.log(n))


def mi_saturation_kz(X: np.ndarray, seed: int) -> dict:
    """MI-vs-latent-dimension curve and its saturation knee k_z."""
    n, d = X.shape
    half = d // 2
    kmax = min(max(K_GRID), half, n - 1)
    ks = [k for k in K_GRID if k <= kmax]
    rng = np.random.default_rng(seed)

    mi_acc = np.zeros(len(ks), dtype=np.float64)
    for _ in range(N_SPLITS):
        perm = rng.permutation(d)
        a_idx, b_idx = perm[:half], perm[half:2 * half]
        Sa = pca_scores(X[:, a_idx], kmax)
        Sb = pca_scores(X[:, b_idx], kmax)
        Sa = Sa / (Sa.std(axis=0, keepdims=True) + 1e-12)   # whiten -> 2nd-moment-free
        Sb = Sb / (Sb.std(axis=0, keepdims=True) + 1e-12)
        for i, k in enumerate(ks):
            mi_acc[i] += infonce_mi(Sa[:, :k], Sb[:, :k], TAU)
    mi = mi_acc / N_SPLITS

    mi_max = float(mi.max())
    thresh = SAT_FRAC * mi_max
    above = np.flatnonzero(mi >= thresh)
    k_z = int(ks[int(above[0])]) if above.size else int(ks[-1])
    return {
        "k_grid": ks,
        "mi_curve_nats": [float(v) for v in mi],
        "mi_max_nats": mi_max,
        "k_z": k_z,
        "sat_frac": SAT_FRAC,
    }


def load_layers() -> list[tuple[str, np.ndarray]]:
    """Return [(layer_name, activations)] — per-layer dir if present, else L19."""
    layers: list[tuple[str, np.ndarray]] = []
    if PER_LAYER_DIR.exists():
        for npz in sorted(PER_LAYER_DIR.glob("m15b_prefill_L*.npz")):
            blob = np.load(npz)
            if "prefill" in blob:
                name = npz.stem.split("_")[-1]
                layers.append((name, blob["prefill"].astype(np.float64)))
    if not layers:
        blob = np.load(CACHE)
        layers.append(("L19", blob["prefill"].astype(np.float64)))
    return layers


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    layers = load_layers()

    per_layer = []
    for name, X in layers:
        eigs = cov_eigs(X)
        sat = mi_saturation_kz(X, SEED)
        per_layer.append({
            "layer": name,
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "pr": participation_ratio(eigs),       # F-1 breathing readout (2nd moment)
            "d_eff_shannon": shannon_eff_dim(eigs),
            "d_eff_90pct": dim_for_variance(eigs, 0.90),
            "k_z": sat["k_z"],                      # DySIB MI-saturation (2nd-moment-free)
            "mi_max_nats": sat["mi_max_nats"],
            "k_grid": sat["k_grid"],
            "mi_curve_nats": sat["mi_curve_nats"],
        })

    kz = np.array([d["k_z"] for d in per_layer], dtype=np.float64)
    pr = np.array([d["pr"] for d in per_layer], dtype=np.float64)
    n_layers = len(per_layer)

    summary = {
        "n_layers": n_layers,
        "single_layer_only": bool(n_layers == 1),
        "k_z_mean": float(kz.mean()),
        "k_z_std": float(kz.std()),
        "k_z_cv": float(kz.std() / kz.mean()) if kz.mean() > 0 else float("nan"),
        "pr_min": float(pr.min()),
        "pr_max": float(pr.max()),
        "pr_breathing_range": float(pr.max() - pr.min()),
    }
    if n_layers > 1:
        # second-moment-free k_z constant while PR breathes => breathing is artifact
        pr_breathes = summary["pr_breathing_range"] > 0.10 * float(pr.mean())
        kz_constant = summary["k_z_cv"] < 0.10
        summary["pr_breathes"] = bool(pr_breathes)
        summary["k_z_constant"] = bool(kz_constant)
        summary["breathing_is_second_moment_artifact"] = bool(pr_breathes and kz_constant)
        summary["corr_kz_pr"] = float(np.corrcoef(kz, pr)[0, 1]) if kz.std() > 0 else float("nan")

    out = {
        "experiment": "P11-FE1299",
        "description": "DySIB InfoNCE-MI-saturation intrinsic dim (k_z) vs F-1 PR/d_eff breathing",
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "tau": TAU,
            "sat_frac": SAT_FRAC,
            "k_grid": K_GRID,
        },
        "sanity_dom_proxy_auroc": auroc(np.load(CACHE)["prefill"].astype(np.float64)
                                        @ (np.load(CACHE)["prefill"][correct].mean(0)
                                           - np.load(CACHE)["prefill"][~correct].mean(0)),
                                        correct),
        "per_layer": per_layer,
        "summary": summary,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())