"""FE291 follow-up #2 — per-problem cov-spectrum probe on PC1-residualized
L19 trajectory clouds.

Question. Move #2 (`recompute_pc1_resid_ph.py`) found that a matched-cov
Gaussian null on PC1-residualized clouds reaches OOF AUROC **0.763**, while
the actual data's 5-PH features hit only 0.688. The null AUROC is a property
of per-problem covariance spectra: the null draws Gaussians from each
problem's residualized covariance, so 5 PH features on the null sample are
just a randomized read of the cov-spectrum signature.

This script extracts that signature directly. For each of the 500 pathway8
problems we:
  1. load states[19] (T_i, 1536),
  2. project out the FE291 PC1 direction (rederived from the prefill cache),
  3. compute the eigenvalue spectrum of the residualized cloud's covariance,
  4. take the top-K eigenvalues as a feature vector,
  5. fit a 5-fold StratifiedKFold OOF logistic on (top-K log-eigvals).

Reads:
  pathway11_h100/prefill_inversion/cache/m15b_prefill.npz   — for FE291 PC1
  pathway8_layerwise/data/math500/problem_*.npz             — for L19 clouds

Output: pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json

Comparison anchors:
  - DoM (supervised, 1-d):                          0.7679 (FE291 SEED=0)
  - 2-feature [PC1, PC9] OOF logistic:              0.7856 (this session)
  - PC1-resid 5-PH real OOF:                        0.6884 (move #2)
  - PC1-resid 5-PH null OOF (matched-cov Gaussian): 0.7628 (move #2)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
NPZ_DIR = ROOT / "pathway8_layerwise/data/math500"
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json"
EIGVAL_CACHE = ROOT / "pathway11_h100/cov_spectrum/eigval_cache.npz"

LAYER = 19
N_PROBLEMS = 500
TOP_K_GRID = [5, 10, 20, 50, 100]
N_FOLDS = 5
SEED = 42


def derive_fe291_pc1() -> tuple[np.ndarray, float]:
    d = np.load(PREFILL_CACHE)
    X = d["prefill"].astype(np.float64)  # (500, 1536)
    n, _ = X.shape
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (n - 1)
    pc1_var_share = float(eigvals[0] / eigvals.sum())
    pc1 = Vt[0]
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)
    return pc1, pc1_var_share


def cloud_eigvals(cloud: np.ndarray) -> np.ndarray:
    """Top eigvals (descending) of cloud's sample covariance via SVD."""
    n = cloud.shape[0]
    centered = cloud - cloud.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    eigvals = (s ** 2) / max(n - 1, 1)
    return np.sort(eigvals)[::-1]


def fit_lr_oof_auroc(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED)),
    ])
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc_oof = float(roc_auc_score(y, oof))
    fold_aurocs: list[float] = []
    for tr, te in cv.split(X, y):
        p = pipe.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fold_aurocs.append(float(roc_auc_score(y[te], p)))
    return auroc_oof, float(np.mean(fold_aurocs)), float(np.std(fold_aurocs))


def _extract_eigvals(files, pc1: np.ndarray, K_MAX: int):
    """Cold-pass: load every per-problem NPZ, project out PC1, extract top-K
    eigvals of the residualized cov."""
    n = len(files)
    correct = np.zeros(n, dtype=bool)
    cloud_T = np.zeros(n, dtype=np.int64)
    eigval_mat = np.zeros((n, K_MAX), dtype=np.float64)
    t0 = time.time()
    for i, f in enumerate(files):
        d = np.load(f)
        s = d["states"][LAYER].astype(np.float64)  # (T_i, 1536)
        correct[i] = bool(d["correct"])
        cloud_T[i] = s.shape[0]

        proj_coef = s @ pc1
        cloud = s - np.outer(proj_coef, pc1)

        eigs = cloud_eigvals(cloud)
        if len(eigs) >= K_MAX:
            eigval_mat[i] = eigs[:K_MAX]
        else:
            eigval_mat[i, :len(eigs)] = eigs

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n}  T_i={s.shape[0]:>4}  "
                  f"top1={eigs[0]:.3f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)
    elapsed_total = time.time() - t0
    print(f"eigval extraction done in {elapsed_total:.0f}s", flush=True)
    return eigval_mat, correct, cloud_T, elapsed_total


def _save_cache(eigval_mat: np.ndarray, correct: np.ndarray, cloud_T: np.ndarray,
                pc1_var_share: float, elapsed_seconds: float) -> None:
    EIGVAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EIGVAL_CACHE,
             eigval_mat=eigval_mat,
             correct=correct,
             cloud_T=cloud_T,
             pc1_var_share=np.float64(pc1_var_share),
             elapsed_seconds=np.float64(elapsed_seconds))
    print(f"saved eigval cache to {EIGVAL_CACHE.relative_to(ROOT)}", flush=True)


def main() -> int:
    files = sorted(NPZ_DIR.glob("problem_*.npz"))
    if len(files) != N_PROBLEMS:
        print(f"MISSING_REGEN_INPUT expected {N_PROBLEMS} in {NPZ_DIR}, "
              f"got {len(files)}", file=sys.stderr)
        return 2

    pc1, pc1_var_share = derive_fe291_pc1()
    print(f"derived FE291 PC1: ‖pc1‖={np.linalg.norm(pc1):.6f}, "
          f"var_share={pc1_var_share:.4f}", flush=True)

    K_MAX = max(TOP_K_GRID)

    if EIGVAL_CACHE.exists():
        cache = np.load(EIGVAL_CACHE)
        if (cache["eigval_mat"].shape == (N_PROBLEMS, K_MAX)
                and float(cache["pc1_var_share"]) == pc1_var_share):
            eigval_mat = cache["eigval_mat"]
            correct = cache["correct"]
            cloud_T = cache["cloud_T"]
            elapsed_total = float(cache["elapsed_seconds"])
            print(f"loaded eigval cache from {EIGVAL_CACHE.relative_to(ROOT)} "
                  f"(skip {N_PROBLEMS}-problem extraction; original took "
                  f"{elapsed_total:.0f}s)", flush=True)
        else:
            print(f"cache shape/PC1 mismatch — re-extracting", flush=True)
            eigval_mat, correct, cloud_T, elapsed_total = _extract_eigvals(
                files, pc1, K_MAX)
            _save_cache(eigval_mat, correct, cloud_T, pc1_var_share, elapsed_total)
    else:
        eigval_mat, correct, cloud_T, elapsed_total = _extract_eigvals(
            files, pc1, K_MAX)
        _save_cache(eigval_mat, correct, cloud_T, pc1_var_share, elapsed_total)
    print(f"cloud T_i: min={cloud_T.min()}, median={int(np.median(cloud_T))}, "
          f"max={cloud_T.max()}", flush=True)
    print(f"correctness rate: {correct.mean():.4f}  ({correct.sum()}/{N_PROBLEMS})",
          flush=True)

    # Use log-eigvals so dynamic range doesn't dominate the linear fit.
    log_eig = np.log(eigval_mat + 1e-12)

    classifiers: dict[str, dict] = {}
    print("\nClassifier sweep over K:")
    for K in TOP_K_GRID:
        feats = log_eig[:, :K]
        auroc_oof, cv_mean, cv_std = fit_lr_oof_auroc(feats, correct)
        classifiers[f"top{K}_log_eigvals"] = {
            "auroc_oof": auroc_oof,
            "auroc_cv_mean": cv_mean,
            "auroc_cv_std": cv_std,
            "n_features": K,
        }
        print(f"  K={K:>3}  oof={auroc_oof:.4f}  cv5={cv_mean:.4f}±{cv_std:.4f}",
              flush=True)

    # Single-feature baselines: each of the top-3 log-eigvals alone.
    for k in (1, 2, 3):
        feats = log_eig[:, k - 1:k]
        auroc_oof, cv_mean, cv_std = fit_lr_oof_auroc(feats, correct)
        classifiers[f"log_eigval_{k}_alone"] = {
            "auroc_oof": auroc_oof,
            "auroc_cv_mean": cv_mean,
            "auroc_cv_std": cv_std,
            "n_features": 1,
        }
        print(f"  log_eig_{k} alone     oof={auroc_oof:.4f}  "
              f"cv5={cv_mean:.4f}±{cv_std:.4f}", flush=True)

    # Headline pick: highest OOF among the K-grid.
    grid_results = {k: classifiers[f"top{k}_log_eigvals"]["auroc_oof"]
                    for k in TOP_K_GRID}
    best_k = max(grid_results, key=grid_results.get)
    best_auroc = grid_results[best_k]

    # Comparison anchors.
    A_DOM = 0.7679
    A_PH_NULL = 0.7628
    A_PH_REAL = 0.6884
    A_TWO_FEAT = 0.7856

    if best_auroc > A_DOM + 0.005:
        verdict = (
            f"Per-problem cov spectrum (top-{best_k} log-eigvals on PC1-"
            f"residualized clouds) reaches OOF AUROC {best_auroc:.4f} — "
            f"BEATS supervised DoM ({A_DOM}) by "
            f"{(best_auroc - A_DOM) * 100:+.2f} pp. F-2's signal is at "
            f"least partly spectral, not purely directional: per-problem "
            f"second-order structure (after PC1 removal) carries correctness "
            f"information beyond the L19 mean shift along PC1+PC9."
        )
    elif best_auroc > A_PH_NULL - 0.01:
        verdict = (
            f"Per-problem cov spectrum (top-{best_k} log-eigvals on PC1-"
            f"residualized clouds) reaches OOF AUROC {best_auroc:.4f} — "
            f"matches the matched-cov Gaussian PH-null AUROC ({A_PH_NULL}) "
            f"to within ~1 pp. The null's PH-feature signal is faithfully "
            f"captured by the spectrum it samples from. PH features add no "
            f"information beyond the spectrum, but the spectrum itself "
            f"falls short of supervised DoM ({A_DOM}). F-2's signal is "
            f"primarily directional, not spectral."
        )
    else:
        verdict = (
            f"Per-problem cov spectrum (top-{best_k} log-eigvals on PC1-"
            f"residualized clouds) only reaches OOF AUROC {best_auroc:.4f}, "
            f"below both the matched-cov PH null ({A_PH_NULL}) and "
            f"supervised DoM ({A_DOM}). The null's 0.763 is being driven by "
            f"something other than top-K eigvals alone (perhaps eigval "
            f"ratios or shape descriptors not captured by linear-on-log-"
            f"eigvals)."
        )

    print(f"\nVERDICT: {verdict}", flush=True)

    out = {
        "experiment": "FE291-followup-PC1-resid-cov-spectrum",
        "depends_on": ["F-2", "F-10", "FE291"],
        "n_problems": N_PROBLEMS,
        "layer": LAYER,
        "pc1_source": "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz "
                      "(FE291 protocol)",
        "pc1_var_share": pc1_var_share,
        "top_k_grid": TOP_K_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "cloud_T_min": int(cloud_T.min()),
        "cloud_T_median": int(np.median(cloud_T)),
        "cloud_T_max": int(cloud_T.max()),
        "correctness_rate": float(correct.mean()),
        "classifiers": classifiers,
        "best_top_k": int(best_k),
        "best_auroc_oof": float(best_auroc),
        "anchors": {
            "supervised_dom_auroc": A_DOM,
            "two_feature_pc1_pc9_auroc": A_TWO_FEAT,
            "pc1_resid_5ph_real_auroc": A_PH_REAL,
            "pc1_resid_5ph_null_auroc": A_PH_NULL,
        },
        "verdict": verdict,
        "elapsed_seconds": float(elapsed_total),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}")

    # Tier-1 regen-readback (key=value lines on stdout) for future Claim wiring.
    for K in TOP_K_GRID:
        a = classifiers[f"top{K}_log_eigvals"]["auroc_oof"]
        print(f"cov_spectrum.top{K}_log_eigvals_auroc={a:.10f}")
    print(f"cov_spectrum.best_top_k={best_k}")
    print(f"cov_spectrum.best_auroc_oof={best_auroc:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
