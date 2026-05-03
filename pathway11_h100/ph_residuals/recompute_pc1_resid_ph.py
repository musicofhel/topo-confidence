"""FE291 follow-up — PH on PC1-residualized L19 trajectory clouds.

Question: F-10's strongest form (post-FE291) is "topology adds no signal
beyond covariance, where the covariance pathway reduces to a single
named direction PC1." This script tests that literally — projects out
the FE291 PC1 direction from every L19 trajectory token, then runs the
5-feature PH null-vs-real pipeline. If real ≈ null still, F-10's
strongest form holds. If real > null + 0.05, there's residual
orthogonal-to-PC1 correctness topology that PH features can recover.

PC1 source: rederived here from `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
(500 × 1536 prefill activations) via the same SVD as FE291. Sub-second.

Trajectory source: `pathway8_layerwise/data/math500/problem_*.npz`,
states[19] = (T_i, 1536) per problem.

Output: pathway11_h100/ph_residuals/pc1_resid_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
NPZ_DIR = ROOT / "pathway8_layerwise/data/math500"
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/ph_residuals/pc1_resid_results.json"
PH_CACHE = ROOT / "pathway11_h100/ph_residuals/ph_cache.npz"

LAYER = 19
N_PROBLEMS = 500
N_NULLS_PER_PROBLEM = 5
SUBSAMPLE = 100
PCA_COMPONENTS = 45
N_FOLDS = 5
SEED = 42

PH_FEATURE_NAMES = [
    "H0_max_lifetime",
    "H0_entropy",
    "H1_pers_entropy",
    "H1_n_features",
    "H1_max_lifetime",
]


def compute_5_ph_features(cloud: np.ndarray) -> np.ndarray:
    if len(cloud) < 5:
        return np.zeros(5)
    n_comp = min(PCA_COMPONENTS, cloud.shape[0] - 1, cloud.shape[1])
    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)
    if len(cloud_pca) > SUBSAMPLE:
        idx = np.linspace(0, len(cloud_pca) - 1, SUBSAMPLE, dtype=int)
        cloud_pca = cloud_pca[idx]
    dgms = ripser(cloud_pca, maxdim=1)["dgms"]
    feats = np.zeros(5)
    h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
    if len(h0) > 0:
        lt = h0[:, 1] - h0[:, 0]
        feats[0] = lt.max()
        ltn = lt / lt.sum() if lt.sum() > 0 else lt
        ltn = ltn[ltn > 0]
        feats[1] = -np.sum(ltn * np.log(ltn)) if len(ltn) > 0 else 0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
        if len(h1) > 0:
            lt1 = h1[:, 1] - h1[:, 0]
            lt1n = lt1 / lt1.sum() if lt1.sum() > 0 else lt1
            lt1n = lt1n[lt1n > 0]
            feats[2] = -np.sum(lt1n * np.log(lt1n)) if len(lt1n) > 0 else 0
            feats[3] = len(h1)
            feats[4] = lt1.max()
    return feats


def low_rank_gaussian_null(cloud: np.ndarray, n_nulls: int, seed: int) -> np.ndarray:
    n, _ = cloud.shape
    mean = cloud.mean(axis=0)
    centered = cloud - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    scale = s / np.sqrt(max(n - 1, 1))
    rng = np.random.default_rng(seed)
    feats_all = []
    for _ in range(n_nulls):
        z = rng.standard_normal((n, len(s)))
        synthetic = mean + (z * scale) @ vt
        feats_all.append(compute_5_ph_features(synthetic))
    return np.mean(feats_all, axis=0)


def fit_lr_oof_auroc(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> tuple[float, float, float]:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc_oof = float(roc_auc_score(y, oof))
    fold_aurocs: list[float] = []
    for train, test in cv.split(X, y):
        p = pipe.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        fold_aurocs.append(float(roc_auc_score(y[test], p)))
    return auroc_oof, float(np.mean(fold_aurocs)), float(np.std(fold_aurocs))


def _extract_ph(files: list[Path], pc1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    correct = np.zeros(N_PROBLEMS, dtype=bool)
    real_ph = np.zeros((N_PROBLEMS, 5), dtype=np.float64)
    null_ph = np.zeros((N_PROBLEMS, 5), dtype=np.float64)
    t0 = time.time()
    for i, f in enumerate(files):
        d = np.load(f)
        s = d["states"][LAYER].astype(np.float64)
        correct[i] = bool(d["correct"])
        proj_coef = s @ pc1
        cloud = s - np.outer(proj_coef, pc1)
        real_ph[i] = compute_5_ph_features(cloud)
        null_ph[i] = low_rank_gaussian_null(cloud, n_nulls=N_NULLS_PER_PROBLEM, seed=SEED + i)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_PROBLEMS - i - 1)
            print(f"  {i+1}/{N_PROBLEMS}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    elapsed_total = time.time() - t0
    print(f"PH compute done in {elapsed_total:.0f}s", flush=True)
    return real_ph, null_ph, correct, elapsed_total


def _save_ph_cache(real_ph: np.ndarray, null_ph: np.ndarray, correct: np.ndarray,
                    pc1_var_share: float, elapsed_seconds: float) -> None:
    np.savez(PH_CACHE, real_ph=real_ph, null_ph=null_ph, correct=correct,
             pc1_var_share=pc1_var_share, elapsed_seconds=elapsed_seconds)
    print(f"saved PH cache to {PH_CACHE.relative_to(ROOT)} "
          f"({PH_CACHE.stat().st_size // 1024} KB)", flush=True)


def derive_fe291_pc1() -> tuple[np.ndarray, float, float]:
    """Re-derive PC1 from prefill aggregate (matches FE291)."""
    d = np.load(PREFILL_CACHE)
    X = d["prefill"].astype(np.float64)  # (500, 1536)
    n, _ = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (n - 1)
    pc1_var_share = float(eigvals[0] / eigvals.sum())
    pc1 = Vt[0]  # (1536,) unit vector
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)
    return pc1, pc1_var_share, float(mu @ pc1)


def main() -> int:
    files = sorted(NPZ_DIR.glob("problem_*.npz"))
    if len(files) != N_PROBLEMS:
        print(f"MISSING_REGEN_INPUT expected {N_PROBLEMS} in {NPZ_DIR}, got {len(files)}",
              file=sys.stderr)
        return 2

    pc1, pc1_var_share, _ = derive_fe291_pc1()
    print(f"derived FE291 PC1: ‖pc1‖={np.linalg.norm(pc1):.6f}, var_share={pc1_var_share:.4f}",
          flush=True)

    if PH_CACHE.exists():
        cache = np.load(PH_CACHE)
        if (cache["real_ph"].shape == (N_PROBLEMS, 5)
                and cache["null_ph"].shape == (N_PROBLEMS, 5)
                and abs(float(cache["pc1_var_share"]) - pc1_var_share) < 1e-9):
            real_ph = cache["real_ph"]
            null_ph = cache["null_ph"]
            correct = cache["correct"]
            elapsed_total = float(cache["elapsed_seconds"])
            print(f"loaded PH cache from {PH_CACHE.relative_to(ROOT)} "
                  f"(skipped ~{elapsed_total:.0f}s of PH compute)", flush=True)
        else:
            print("PH cache shape/PC1 mismatch — re-extracting", flush=True)
            real_ph, null_ph, correct, elapsed_total = _extract_ph(files, pc1)
            _save_ph_cache(real_ph, null_ph, correct, pc1_var_share, elapsed_total)
    else:
        real_ph, null_ph, correct, elapsed_total = _extract_ph(files, pc1)
        _save_ph_cache(real_ph, null_ph, correct, pc1_var_share, elapsed_total)

    feature_comparison = {}
    print("\nFeature comparison (PC1-residualized real vs full-cov null):")
    for j, name in enumerate(PH_FEATURE_NAMES):
        rm, rs = float(real_ph[:, j].mean()), float(real_ph[:, j].std())
        nm, ns = float(null_ph[:, j].mean()), float(null_ph[:, j].std())
        print(f"  {name:20s} real={rm:8.3f}±{rs:.3f}  null={nm:8.3f}±{ns:.3f}  diff={rm-nm:+.3f}")
        feature_comparison[name] = {"real_mean": rm, "real_std": rs,
                                    "null_mean": nm, "null_std": ns,
                                    "diff_mean": rm - nm}

    ph_results = {}
    for name, feats in (("real_ph_pc1resid", real_ph),
                        ("null_ph_pc1resid", null_ph),
                        ("diff_ph_pc1resid", real_ph - null_ph)):
        auroc_oof, cv_mean, cv_std = fit_lr_oof_auroc(feats, correct, N_FOLDS, SEED)
        ph_results[name] = {"auroc_oof": auroc_oof, "auroc_cv_mean": cv_mean, "auroc_cv_std": cv_std}
        print(f"  {name:18s}: oof={auroc_oof:.4f}, cv5={cv_mean:.4f}±{cv_std:.4f}")

    real_auroc = ph_results["real_ph_pc1resid"]["auroc_oof"]
    null_auroc = ph_results["null_ph_pc1resid"]["auroc_oof"]
    diff_auroc = ph_results["diff_ph_pc1resid"]["auroc_oof"]
    gap = real_auroc - null_auroc

    F10_RAW_REAL = 0.690
    F10_RAW_NULL = 0.693

    if abs(gap) < 0.02:
        verdict = (
            f"PC1-residualized PH features behave like a Gaussian null "
            f"(real={real_auroc:.3f}, null={null_auroc:.3f}, gap={gap:+.3f}). F-10's "
            f"strongest form holds: removing the FE291 PC1 direction does not "
            f"expose orthogonal topology-level correctness signal."
        )
    elif gap > 0.02:
        verdict = (
            f"PC1 residualization exposes residual topology-level signal "
            f"(real={real_auroc:.3f} > null={null_auroc:.3f}, gap={gap:+.3f}). F-10's "
            f"strongest form FAILS: there is correctness topology orthogonal "
            f"to the FE291 PC1 direction."
        )
    else:
        verdict = (
            f"Inverted gap on PC1 residuals (real={real_auroc:.3f} < null={null_auroc:.3f}, "
            f"gap={gap:+.3f}). Residualization weakens topology signal below null."
        )

    print(f"\nVERDICT: {verdict}")

    out = {
        "experiment": "FE291-followup-PC1-resid-PH",
        "depends_on": ["F-10", "FE291"],
        "n_problems": N_PROBLEMS,
        "layer": LAYER,
        "pc1_source": "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz (FE291 protocol)",
        "pc1_var_share": pc1_var_share,
        "n_features": 5,
        "n_nulls_per_problem": N_NULLS_PER_PROBLEM,
        "subsample": SUBSAMPLE,
        "pca_components": PCA_COMPONENTS,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "feature_comparison": feature_comparison,
        "classifiers": ph_results,
        "auroc_real_ph_pc1resid": real_auroc,
        "auroc_null_ph_pc1resid": null_auroc,
        "auroc_diff_ph_pc1resid": diff_auroc,
        "gap_real_minus_null": gap,
        "f10_raw_real": F10_RAW_REAL,
        "f10_raw_null": F10_RAW_NULL,
        "verdict": verdict,
        "elapsed_seconds": float(elapsed_total),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}")

    # Tier-1 regen-readback (key=value lines on stdout) for future Claim wiring.
    print(f"pc1_resid.auroc_real_ph_pc1resid={real_auroc:.10f}")
    print(f"pc1_resid.auroc_null_ph_pc1resid={null_auroc:.10f}")
    print(f"pc1_resid.gap_real_minus_null={gap:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
