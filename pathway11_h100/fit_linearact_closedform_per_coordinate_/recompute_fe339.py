"""FE339 — Linear-AcT closed-form (ω, β) σ-aware correctness probe.

Activation Transport (AcT, arXiv:2410.23054) shows that on real LLM activations
σ_source ≠ σ_target, and that Linear-AcT (affine ω-scaled + β-shifted transport)
consistently beats Mean-AcT (β-shift only). Our F-2 DoM probe (AUROC 0.7731 on the
EXP-037 5-fold OOF split) is Mean-AcT-equivalent: it uses only the per-coordinate
mean difference μ_correct − μ_incorrect and ignores variance asymmetry.

This script fits, per coordinate and per fold, the closed-form Linear-AcT transport
between the incorrect (source) and correct (target) class-conditional distributions
of the cached L19 prefill activations:

    ω_d = σ_correct,d / σ_incorrect,d
    β_d = μ_correct,d − ω_d · μ_incorrect,d            (so T_d(x) = ω_d·x + β_d)

The σ-aware probe is the per-coordinate diagonal-Gaussian log-likelihood ratio
(the Bayes-optimal score that *uses* ω; Mean-AcT collapses to it when ω≡1):

    s(x) = Σ_d [ −½((x_d−μ_c,d)/σ_c,d)² − log σ_c,d
                  +½((x_d−μ_i,d)/σ_i,d)² + log σ_i,d ]

Both the σ-aware probe and the mean-only DoM are scored out-of-fold on the same
stratified 5-fold split (SEED=9999) used for EXP-037, then compared to F-2 (0.7731).

Predicts: σ-aware probe beats mean-only DoM by ≥ 0.02 AUROC iff variance asymmetry
(mean_d |log(σ_c,d/σ_i,d)|) is non-negligible.
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
OUT_JSON = ROOT / "pathway11_h100/linear_act/results.json"

N_FOLDS = 5
SEED = 9999
F2_DOM_AUROC = 0.7731
BEAT_MARGIN = 0.02
# Relative floor on per-coordinate std to keep the Gaussian LLR finite when a
# coordinate is (near-)degenerate within a class on a given fold.
SIGMA_FLOOR_REL = 1e-6


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def _class_stats(X: np.ndarray, floor: float):
    """Per-coordinate mean and floored std for one class."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.maximum(sigma, floor)
    return mu, sigma


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), (X.shape, y.shape)

    # Global relative floor scaled to the data so SIGMA_FLOOR_REL is unit-free.
    global_scale = float(X.std())
    sigma_floor = SIGMA_FLOOR_REL * global_scale if global_scale > 0 else SIGMA_FLOOR_REL

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    mean_scores = np.zeros(n, dtype=np.float64)   # Mean-AcT / F-2 DoM
    sigma_scores = np.zeros(n, dtype=np.float64)  # Linear-AcT σ-aware LLR

    # Variance-asymmetry diagnostics aggregated over folds.
    log_omega_abs_means = []   # mean_d |log(σ_c/σ_i)| per fold
    omega_means = []           # mean_d ω_d per fold

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        mu_c, sig_c = _class_stats(Xtr[ytr], sigma_floor)   # target = correct
        mu_i, sig_i = _class_stats(Xtr[~ytr], sigma_floor)  # source = incorrect

        # Linear-AcT closed form (per coordinate).
        omega = sig_c / sig_i
        # beta = mu_c - omega * mu_i  (retained for diagnostics; not needed to score)
        omega_means.append(float(np.mean(omega)))
        log_omega_abs_means.append(float(np.mean(np.abs(np.log(omega)))))

        # Mean-only DoM (Mean-AcT-equivalent): raw mean-difference projection,
        # matching the F-2 / EXP-037 convention.
        d_raw = mu_c - mu_i
        mean_scores[test_idx] = Xte @ d_raw

        # σ-aware diagonal-Gaussian LLR (uses ω via the separate σ_c, σ_i).
        zc = (Xte - mu_c) / sig_c
        zi = (Xte - mu_i) / sig_i
        log_sig_term = np.log(sig_i) - np.log(sig_c)   # +log σ_i − log σ_c per coord
        sigma_scores[test_idx] = (
            -0.5 * np.sum(zc * zc, axis=1)
            + 0.5 * np.sum(zi * zi, axis=1)
            + np.sum(log_sig_term)
        )

    auroc_mean = auroc(mean_scores, y)
    auroc_sigma = auroc(sigma_scores, y)
    delta = auroc_sigma - auroc_mean
    var_asymmetry = float(np.mean(log_omega_abs_means))

    out = {
        "experiment": "FE339",
        "description": "Linear-AcT closed-form (omega, beta) sigma-aware probe vs mean-only DoM",
        "n": int(n),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_mean_only_dom_oof": float(auroc_mean),
        "auroc_sigma_aware_act_oof": float(auroc_sigma),
        "delta_sigma_minus_mean": float(delta),
        "f2_reference_auroc": F2_DOM_AUROC,
        "delta_vs_f2": float(auroc_sigma - F2_DOM_AUROC),
        "beat_margin": BEAT_MARGIN,
        "sigma_aware_beats_mean_by_margin": bool(delta >= BEAT_MARGIN),
        "variance_asymmetry_mean_abs_log_omega": var_asymmetry,
        "omega_mean_per_fold": [float(v) for v in omega_means],
        "log_omega_abs_mean_per_fold": [float(v) for v in log_omega_abs_means],
        "sigma_floor": float(sigma_floor),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())