"""P11-FE773 — Fisher LDA decomposition of L19 prefill activations.

Tests whether the 0.7731 prefill-DoM AUROC (F-2, Qwen 1.5B, MATH-500) rests on a
clean linear discriminant axis or on between-class mean dispersion buried in
anisotropic, near-singular within-class noise.

Computes, from cached L19 prefill hidden states:
  - within-class scatter  S_W = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T
  - between-class scatter  S_B = (n_pos n_neg / N) d d^T,  d = mu_pos - mu_neg
  - lambda_max(S_W^-1 S_B)  (the single nonzero generalized eigenvalue;
        for two classes equals (n_pos n_neg / N) * d^T S_W^-1 d)
  - Tr(S_B)/N, Tr(S_W), cond(S_W) (raw + ridged), numerical rank(S_W)
  - OOF LDA-axis AUROC vs the published DoM AUROC
  - fold-to-fold cosine stability of the discriminant axis (robustness proxy)
  - AUROC degradation under isotropic axis perturbation

Theorem 1 (Fu et al. 2604.20817) framing: Fourier sparsity / class-mean
dispersion (Tr(S_B) > 0) is *necessary but not sufficient* for linear
separability. The decisive quantity is the discriminant ratio
lambda_max(S_W^-1 S_B) — between-class scatter projected through the inverse
within-class scatter. A large Tr(S_B)/N with a small lambda_max and a near-
singular, ill-conditioned S_W is the signature of dispersion that does not
translate into a robust separating axis. We report the bound ratio
Tr(S_B)/N divided by lambda_max as the "dispersion-not-discriminability" gap.
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
OUT_JSON = ROOT / "pathway11_h100/fisher_lda/results.json"

N_FOLDS = 5
SEED = 9999
RIDGE_REL = 1e-3
PERTURB_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


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


def within_scatter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pooled within-class scatter S_W (sum of squares, not normalized)."""
    d = X.shape[1]
    S = np.zeros((d, d), dtype=np.float64)
    for cls in (True, False):
        Xc = X[y == cls]
        Xc = Xc - Xc.mean(axis=0)
        S += Xc.T @ Xc
    return S


def ridge_sw(S_W: np.ndarray) -> tuple[np.ndarray, float]:
    d = S_W.shape[0]
    alpha = RIDGE_REL * float(np.trace(S_W)) / d
    return S_W + alpha * np.eye(d, dtype=np.float64), alpha


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    N = len(y)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    d_dim = X.shape[1]

    # --- Full-data scatter decomposition --------------------------------
    mu_pos = X[y].mean(axis=0)
    mu_neg = X[~y].mean(axis=0)
    d_vec = mu_pos - mu_neg
    sb_coeff = (n_pos * n_neg) / N
    tr_SB = float(sb_coeff * (d_vec @ d_vec))

    S_W = within_scatter(X, y)
    tr_SW = float(np.trace(S_W))

    # cond(S_W): raw (rank-deficient: N < d, so expect near-singular) + ridged
    evals = np.linalg.eigvalsh(S_W)
    lam_max_sw = float(evals[-1])
    lam_min_sw = float(evals[0])
    tol = lam_max_sw * max(S_W.shape) * np.finfo(np.float64).eps
    rank_SW = int((evals > tol).sum())
    cond_SW_raw = float(lam_max_sw / lam_min_sw) if lam_min_sw > 0 else float("inf")

    S_W_ridge, alpha = ridge_sw(S_W)
    evals_r = np.linalg.eigvalsh(S_W_ridge)
    cond_SW_ridged = float(evals_r[-1] / evals_r[0])

    # lambda_max(S_W^-1 S_B) = sb_coeff * d^T S_W^-1 d  (rank-1 S_B)
    w_full = np.linalg.solve(S_W_ridge, d_vec)
    fisher_discriminant = float(d_vec @ w_full)            # d^T S_W^-1 d
    lambda_max_SWinv_SB = float(sb_coeff * fisher_discriminant)

    # Dispersion-not-discriminability gap (Theorem 1 framing): large when
    # between-class scatter per sample dwarfs the realized discriminant.
    tr_SB_over_N = tr_SB / N
    dispersion_gap = float(tr_SB_over_N / lambda_max_SWinv_SB) if lambda_max_SWinv_SB > 0 else float("inf")

    # --- OOF axis AUROCs -------------------------------------------------
    folds = stratified_kfold(y, N_FOLDS, SEED)
    lda_oof = np.zeros(N, dtype=np.float64)
    dom_oof = np.zeros(N, dtype=np.float64)
    fold_axes = []
    for test_idx in folds:
        train_mask = np.ones(N, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_tr = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        S_W_tr, _ = ridge_sw(within_scatter(Xtr, ytr))
        w_tr = np.linalg.solve(S_W_tr, d_tr)
        lda_oof[test_idx] = Xte @ w_tr
        dom_oof[test_idx] = Xte @ d_tr               # mean-difference (DoM) axis
        fold_axes.append(w_tr / (np.linalg.norm(w_tr) + 1e-12))

    auroc_lda_oof = auroc(lda_oof, y)
    auroc_dom_meandiff_oof = auroc(dom_oof, y)
    auroc_dom_published = auroc(dom_score, y)

    # Fold-to-fold cosine stability of the LDA discriminant axis.
    cosines = []
    for i in range(len(fold_axes)):
        for j in range(i + 1, len(fold_axes)):
            cosines.append(float(fold_axes[i] @ fold_axes[j]))
    lda_axis_fold_cos_mean = float(np.mean(cosines))
    lda_axis_fold_cos_min = float(np.min(cosines))

    # --- Robustness under isotropic perturbation of the discriminant ----
    # Add Gaussian noise scaled to a fraction of the axis RMS magnitude;
    # a robust axis keeps AUROC, a noise-supported one collapses to 0.5.
    rng = np.random.default_rng(SEED)
    w_unit = w_full / (np.linalg.norm(w_full) + 1e-12)
    base_scores = X @ w_unit
    rms = float(np.sqrt(np.mean(w_unit ** 2)))
    perturb = []
    for lvl in PERTURB_LEVELS:
        if lvl == 0.0:
            perturb.append({"noise_rel": 0.0, "auroc": auroc(base_scores, y)})
            continue
        accum = []
        for _ in range(20):
            noise = rng.normal(0.0, lvl * rms, size=w_unit.shape)
            w_p = w_unit + noise
            accum.append(auroc(X @ w_p, y))
        perturb.append({"noise_rel": float(lvl), "auroc_mean": float(np.mean(accum)),
                        "auroc_std": float(np.std(accum))})

    # --- Verdict ---------------------------------------------------------
    clean_axis = (
        auroc_lda_oof >= 0.70
        and lda_axis_fold_cos_mean >= 0.5
        and np.isfinite(lambda_max_SWinv_SB)
    )
    if clean_axis:
        verdict = ("CLEAN_DISCRIMINANT: DoM AUROC is supported by a stable, "
                   "well-conditioned separating axis.")
    else:
        verdict = ("DISPERSION_DOMINATED: AUROC rests on between-class mean "
                   "dispersion buried in anisotropic within-class noise; the "
                   "discriminant axis is unstable across folds / S_W is "
                   "near-singular.")

    out = {
        "experiment": "P11-FE773",
        "description": "Fisher LDA decomposition of L19 prefill activations vs Theorem 1 (Fu et al. 2604.20817)",
        "N": N, "n_pos": n_pos, "n_neg": n_neg, "d": d_dim,
        "ridge_rel": RIDGE_REL, "ridge_alpha_fulldata": float(alpha),
        "lambda_max_SWinv_SB": lambda_max_SWinv_SB,
        "fisher_discriminant_dT_SWinv_d": fisher_discriminant,
        "tr_SB": tr_SB,
        "tr_SB_over_N": tr_SB_over_N,
        "tr_SW": tr_SW,
        "cond_SW_raw": cond_SW_raw,
        "cond_SW_ridged": cond_SW_ridged,
        "rank_SW_numerical": rank_SW,
        "lam_max_SW": lam_max_sw,
        "lam_min_SW": lam_min_sw,
        "dispersion_not_discriminability_gap": dispersion_gap,
        "auroc_lda_oof": auroc_lda_oof,
        "auroc_dom_meandiff_oof": auroc_dom_meandiff_oof,
        "auroc_dom_published": auroc_dom_published,
        "lda_axis_fold_cos_mean": lda_axis_fold_cos_mean,
        "lda_axis_fold_cos_min": lda_axis_fold_cos_min,
        "perturbation_robustness": perturb,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "lambda_max_SWinv_SB", "tr_SB_over_N", "cond_SW_raw", "rank_SW_numerical",
        "auroc_lda_oof", "auroc_dom_published", "lda_axis_fold_cos_mean", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())