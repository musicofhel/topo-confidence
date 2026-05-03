"""FE116 — PH-null check on Song-Zhong residualized L19 prefill clouds.

Re-runs the Pathway-9 F-10 PH-null pipeline (5 PH features, low-rank-Gaussian
null, 5-fold OOF LR) but on per-problem clouds residualized via Song-Zhong:
    resid_{c,t} = h_{c,t} − μ − pos_t − ctx_c
where μ is the global mean, pos_t is the per-position mean (minus μ), and
ctx_c is the per-problem mean (minus μ).

F-10 currently has trained=0.690 vs null=0.693 on RAW L19 last-token
trajectories (Pathway 8 cache). The question for FE116 is whether the same
trained=null collapse holds once positional structure is removed — or
whether residualization exposes a topology-level correctness signal that the
raw pipeline missed.

Output: pathway11_h100/ph_residuals/results.json
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
OUT_JSON = ROOT / "pathway11_h100/ph_residuals/results.json"

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
    """Mean of compute_5_ph_features over n_nulls Gaussian samples matching cloud's empirical mean+cov."""
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
    """Return (OOF AUROC, mean fold AUROC, std fold AUROC)."""
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


def main() -> int:
    files = sorted(NPZ_DIR.glob("problem_*.npz"))
    if len(files) != N_PROBLEMS:
        print(f"MISSING_REGEN_INPUT expected {N_PROBLEMS} in {NPZ_DIR}, got {len(files)}",
              file=sys.stderr)
        return 2

    # Pass 0: lengths + labels
    lengths = np.zeros(N_PROBLEMS, dtype=np.int64)
    correct = np.zeros(N_PROBLEMS, dtype=bool)
    for i, f in enumerate(files):
        d = np.load(f)
        lengths[i] = d["states"].shape[1]
        correct[i] = bool(d["correct"])
    T_max = int(lengths.max())
    hidden = 1536
    print(f"n={N_PROBLEMS}, T_min={lengths.min()}, T_median={int(np.median(lengths))}, T_max={T_max}")
    print(f"correct: {int(correct.sum())}/{N_PROBLEMS}")

    # Pass 1: streaming μ and pos_t accumulators (Song-Zhong)
    pos_sum = np.zeros((T_max, hidden), dtype=np.float64)
    pos_count = np.zeros(T_max, dtype=np.int64)
    global_sum = np.zeros(hidden, dtype=np.float64)
    global_count = 0
    for f in files:
        s = np.load(f)["states"][LAYER].astype(np.float64)
        T_i = s.shape[0]
        pos_sum[:T_i] += s
        pos_count[:T_i] += 1
        global_sum += s.sum(axis=0)
        global_count += T_i
    mu = global_sum / global_count
    pos_t = np.zeros((T_max, hidden), dtype=np.float64)
    valid = pos_count > 0
    pos_t[valid] = pos_sum[valid] / pos_count[valid, None] - mu
    print(f"Song-Zhong: μ ‖={np.linalg.norm(mu):.3f}, pos_t variance ‖={np.linalg.norm(pos_t):.3f}")

    # Passes 2+3: per-problem PH on residualized cloud + matched-cov Gaussian null
    real_ph = np.zeros((N_PROBLEMS, 5), dtype=np.float64)
    null_ph = np.zeros((N_PROBLEMS, 5), dtype=np.float64)
    t0 = time.time()
    for i, f in enumerate(files):
        s = np.load(f)["states"][LAYER].astype(np.float64)  # (T_i, 1536)
        T_i = s.shape[0]
        ctx_i = s.mean(axis=0) - mu                         # (1536,)
        # resid_{c,t} = h_{c,t} − μ − pos_t − ctx_c
        cloud = s - mu - pos_t[:T_i] - ctx_i               # (T_i, 1536)

        real_ph[i] = compute_5_ph_features(cloud)
        null_ph[i] = low_rank_gaussian_null(cloud, n_nulls=N_NULLS_PER_PROBLEM, seed=SEED + i)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_PROBLEMS - i - 1)
            print(f"  {i+1}/{N_PROBLEMS}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    elapsed_total = time.time() - t0
    print(f"PH compute done in {elapsed_total:.0f}s")

    # Feature comparison
    feature_comparison = {}
    print("\nFeature comparison (residualized real vs full-cov null):")
    for j, name in enumerate(PH_FEATURE_NAMES):
        rm, rs = float(real_ph[:, j].mean()), float(real_ph[:, j].std())
        nm, ns = float(null_ph[:, j].mean()), float(null_ph[:, j].std())
        print(f"  {name:20s} real={rm:8.3f}±{rs:.3f}  null={nm:8.3f}±{ns:.3f}  diff={rm-nm:+.3f}")
        feature_comparison[name] = {"real_mean": rm, "real_std": rs,
                                    "null_mean": nm, "null_std": ns,
                                    "diff_mean": rm - nm}

    # 5-fold OOF LR per feature set
    ph_results = {}
    for name, feats in (("real_ph_resid", real_ph),
                        ("null_ph_resid", null_ph),
                        ("diff_ph_resid", real_ph - null_ph)):
        auroc_oof, cv_mean, cv_std = fit_lr_oof_auroc(feats, correct, N_FOLDS, SEED)
        ph_results[name] = {"auroc_oof": auroc_oof, "auroc_cv_mean": cv_mean, "auroc_cv_std": cv_std}
        print(f"  {name:18s}: oof={auroc_oof:.4f}, cv5={cv_mean:.4f}±{cv_std:.4f}")

    real_auroc = ph_results["real_ph_resid"]["auroc_oof"]
    null_auroc = ph_results["null_ph_resid"]["auroc_oof"]
    diff_auroc = ph_results["diff_ph_resid"]["auroc_oof"]
    gap = real_auroc - null_auroc

    F10_RAW_REAL = 0.690
    F10_RAW_NULL = 0.693
    F10_RAW_GAP = F10_RAW_REAL - F10_RAW_NULL  # −0.003

    if abs(gap) < 0.02:
        verdict = (
            f"Residualized PH features behave like a Gaussian null "
            f"(real={real_auroc:.3f}, null={null_auroc:.3f}, gap={gap:+.3f}). F-10 "
            f"survives Song-Zhong residualization — topology-level correctness "
            f"signal is not exposed by removing positional structure."
        )
    elif gap > 0.02:
        verdict = (
            f"Residualization exposes a topology-level correctness signal "
            f"(real={real_auroc:.3f} > null={null_auroc:.3f}, gap={gap:+.3f}). F-10 "
            f"is narrower than stated: PH-null collapse is a positional artifact."
        )
    else:
        verdict = (
            f"Inverted gap on residuals (real={real_auroc:.3f} < null={null_auroc:.3f}, "
            f"gap={gap:+.3f}). Residualization weakens topology signal below null — "
            f"unusual but consistent with F-10 being null-bound."
        )

    print(f"\nVERDICT: {verdict}")

    out = {
        "experiment": "P11-FE116",
        "depends_on": ["F-10"],
        "n_problems": N_PROBLEMS,
        "layer": LAYER,
        "n_features": 5,
        "n_nulls_per_problem": N_NULLS_PER_PROBLEM,
        "subsample": SUBSAMPLE,
        "pca_components": PCA_COMPONENTS,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "song_zhong_norms": {
            "mu_norm": float(np.linalg.norm(mu)),
            "pos_t_norm": float(np.linalg.norm(pos_t)),
        },
        "feature_comparison": feature_comparison,
        "classifiers": ph_results,
        "auroc_real_ph_resid": real_auroc,
        "auroc_null_ph_resid": null_auroc,
        "auroc_diff_ph_resid": diff_auroc,
        "gap_real_minus_null": gap,
        "f10_raw_real": F10_RAW_REAL,
        "f10_raw_null": F10_RAW_NULL,
        "f10_raw_gap": F10_RAW_GAP,
        "gap_change_vs_raw": gap - F10_RAW_GAP,
        "verdict": verdict,
        "elapsed_seconds": float(elapsed_total),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
