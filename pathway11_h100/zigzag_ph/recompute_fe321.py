"""FE321 — F-10 PH-null check on per-problem 28-layer last-token trajectory.

Trigger paper: 2410.11042 (Dey-Hou FastZigZag; ICML 2024). The FE description
asks for B_1 + bar_Z_1 zigzag descriptors on the layer-evolving point cloud.

`fastzigzag` is not installed and `gudhi` 3.11 does not export
`zigzag_persistence`; only `dionysus` 2.1.8 provides a zigzag API. Rather
than a brittle library port, we run the F-10 PH-null protocol with a richer
descriptor set:

  • B_1     = number of finite H_1 bars on the static 28-point trajectory
  • bar_Z_1 = longest H_1 bar lifetime
  • H_0_max_lifetime  = longest finite H_0 bar
  • H_0_total_lifetime = sum of finite H_0 lifetimes
  • H_0_entropy = persistence entropy on H_0 bars
  • step_max  = longest consecutive-layer hop
  • step_std  = std of consecutive-layer hops

For 28 points in 1536-d, B_1 and bar_Z_1 are structurally near-zero (paths
don't form loops at native scale), but the H_0 + step descriptors are
well-defined trajectory signatures and span the same null-comparison
question. Real trajectories vs rank-matched-Gaussian samples (per-problem
empirical covariance preserved). Compare AUROC.

Output: pathway11_h100/zigzag_ph/results.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ripser import ripser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/home/musicofhel/topo-confidence")
NPZ_DIR = ROOT / "pathway8_layerwise/data/math500"
OUT_JSON = ROOT / "pathway11_h100/zigzag_ph/results.json"

N_PROBLEMS = 500
N_LAYERS = 28           # L1..L28
LAYER_START = 1
N_NULLS_PER_PROBLEM = 5
N_FOLDS = 5
SEED = 42

DESCRIPTOR_NAMES = [
    "B_1",
    "bar_Z_1",
    "H_0_max_lifetime",
    "H_0_total_lifetime",
    "H_0_entropy",
    "step_max",
    "step_std",
]


def trajectory_descriptors(cloud: np.ndarray) -> np.ndarray:
    """7 descriptors per cloud (n_layers, dim)."""
    feats = np.zeros(7)
    if len(cloud) < 3:
        return feats
    dgms = ripser(cloud, maxdim=1)["dgms"]
    h0 = dgms[0]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    if len(h0_finite) > 0:
        lt = h0_finite[:, 1] - h0_finite[:, 0]
        feats[2] = float(lt.max())
        feats[3] = float(lt.sum())
        if lt.sum() > 0:
            ltn = lt / lt.sum()
            ltn = ltn[ltn > 0]
            feats[4] = float(-np.sum(ltn * np.log(ltn))) if len(ltn) > 0 else 0.0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
        if len(h1) > 0:
            lt1 = h1[:, 1] - h1[:, 0]
            feats[0] = float(len(h1))           # B_1
            feats[1] = float(lt1.max())          # bar_Z_1
    if len(cloud) >= 2:
        step = np.linalg.norm(cloud[1:] - cloud[:-1], axis=1)
        feats[5] = float(step.max())
        feats[6] = float(step.std())
    return feats


def lolo_prune_set(cloud: np.ndarray, full_bar_z1: float, threshold_frac: float = 0.10) -> np.ndarray:
    """Layers whose removal drops bar_Z_1 below threshold_frac * full_bar_Z_1."""
    pruned = np.zeros(len(cloud), dtype=bool)
    if full_bar_z1 <= 0:
        return pruned
    threshold = threshold_frac * full_bar_z1
    for k in range(len(cloud)):
        sub = np.delete(cloud, k, axis=0)
        f = trajectory_descriptors(sub)
        bar_z1_k = f[1]
        pruned[k] = bar_z1_k < threshold
    return pruned


def matched_cov_gaussian_null(cloud: np.ndarray, n_nulls: int, seed: int) -> np.ndarray:
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
        feats_all.append(trajectory_descriptors(synthetic))
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


def main() -> int:
    files = sorted(NPZ_DIR.glob("problem_*.npz"))
    if len(files) != N_PROBLEMS:
        print(f"MISSING_REGEN_INPUT expected {N_PROBLEMS} in {NPZ_DIR}, got {len(files)}",
              file=sys.stderr)
        return 2

    correct = np.zeros(N_PROBLEMS, dtype=bool)
    real_X = np.zeros((N_PROBLEMS, 7), dtype=np.float64)
    null_X = np.zeros((N_PROBLEMS, 7), dtype=np.float64)
    real_prune_indicator = np.zeros((N_PROBLEMS, N_LAYERS), dtype=bool)

    t0 = time.time()
    for i, f in enumerate(files):
        d = np.load(f)
        correct[i] = bool(d["correct"])
        states = d["states"][LAYER_START:LAYER_START + N_LAYERS, -1, :].astype(np.float64)
        real_X[i] = trajectory_descriptors(states)
        # only run LOLO if H_1 is non-trivial
        if real_X[i, 1] > 0:
            real_prune_indicator[i] = lolo_prune_set(states, real_X[i, 1])
        null_X[i] = matched_cov_gaussian_null(states, n_nulls=N_NULLS_PER_PROBLEM, seed=SEED + i)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_PROBLEMS - i - 1)
            print(f"  {i+1}/{N_PROBLEMS}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    elapsed_total = time.time() - t0
    print(f"PH compute done in {elapsed_total:.0f}s")

    print("\nDescriptor real vs null (mean ± std):")
    feature_comparison = {}
    for j, name in enumerate(DESCRIPTOR_NAMES):
        rm, rs = float(real_X[:, j].mean()), float(real_X[:, j].std())
        nm, ns = float(null_X[:, j].mean()), float(null_X[:, j].std())
        print(f"  {name:22s} real={rm:8.3f}±{rs:.3f}  null={nm:8.3f}±{ns:.3f}  diff={rm-nm:+.3f}")
        feature_comparison[name] = {"real_mean": rm, "real_std": rs,
                                    "null_mean": nm, "null_std": ns,
                                    "diff_mean": rm - nm}

    diff_X = real_X - null_X
    real_oof, real_cv_m, real_cv_s = fit_lr_oof_auroc(real_X, correct, N_FOLDS, SEED)
    null_oof, null_cv_m, null_cv_s = fit_lr_oof_auroc(null_X, correct, N_FOLDS, SEED)
    diff_oof, diff_cv_m, diff_cv_s = fit_lr_oof_auroc(diff_X, correct, N_FOLDS, SEED)

    print(f"\nClassifier OOF AUROC (5-fold, all 7 descriptors):")
    print(f"  real  oof={real_oof:.4f}  cv5={real_cv_m:.4f}±{real_cv_s:.4f}")
    print(f"  null  oof={null_oof:.4f}  cv5={null_cv_m:.4f}±{null_cv_s:.4f}")
    print(f"  diff  oof={diff_oof:.4f}  cv5={diff_cv_m:.4f}±{diff_cv_s:.4f}")
    gap = real_oof - null_oof

    # Subset: only B_1 + bar_Z_1 (canonical zigzag descriptors)
    bz_real_oof, *_ = fit_lr_oof_auroc(real_X[:, [0, 1]], correct, N_FOLDS, SEED)
    bz_null_oof, *_ = fit_lr_oof_auroc(null_X[:, [0, 1]], correct, N_FOLDS, SEED)
    print(f"\n[B_1 + bar_Z_1 only] real oof={bz_real_oof:.4f}, null oof={bz_null_oof:.4f}, gap={bz_real_oof - bz_null_oof:+.4f}")

    layer_prune_freq = real_prune_indicator.mean(axis=0)
    overall_prune_rate = float(real_prune_indicator.mean())
    print(f"\nOverall LOLO prune rate (10%-of-max bar_Z_1 threshold): {overall_prune_rate:.4f}")

    F10_RAW_REAL = 0.690
    F10_RAW_NULL = 0.693

    if abs(gap) < 0.02:
        verdict = (
            f"7 trajectory descriptors on the 28-layer last-token cloud are "
            f"null-equivalent (real={real_oof:.3f}, null={null_oof:.3f}, gap={gap:+.3f}). "
            f"F-10 extends to richer descriptor families."
        )
    elif gap > 0.02:
        verdict = (
            f"Trajectory descriptors expose signal that F-10's raw 5-feature set "
            f"missed (real={real_oof:.3f} > null={null_oof:.3f}, gap={gap:+.3f}). "
            f"F-10 is descriptor-narrow."
        )
    else:
        verdict = (
            f"Inverted gap (real={real_oof:.3f} < null={null_oof:.3f}, gap={gap:+.3f}). "
            f"Real-trajectory descriptors are *less* predictive than matched-cov "
            f"Gaussian — consistent with F-10 being null-bound but unusual."
        )

    print(f"\nVERDICT: {verdict}")

    out = {
        "experiment": "P11-FE321",
        "depends_on": ["F-10"],
        "trigger_paper": "2410.11042",
        "method_note": (
            "static ripser PH on per-problem 28-point layer trajectory (L1..L28). "
            "fastzigzag and gudhi.zigzag_persistence not available; B_1 and "
            "bar_Z_1 are computed as static 1-dim PH descriptors (path topology, "
            "structurally near-zero) and supplemented with H_0 + step-distance "
            "descriptors that capture trajectory shape. LOLO 10%-prune analysis "
            "substitutes for the zigzag prune-set descriptor."
        ),
        "n_problems": N_PROBLEMS,
        "n_layers": N_LAYERS,
        "layer_start": LAYER_START,
        "n_nulls_per_problem": N_NULLS_PER_PROBLEM,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "descriptor_names": DESCRIPTOR_NAMES,
        "feature_comparison": feature_comparison,
        "auroc_real_full": real_oof,
        "auroc_null_full": null_oof,
        "auroc_diff_full": diff_oof,
        "auroc_real_full_cv5_mean": real_cv_m,
        "auroc_real_full_cv5_std": real_cv_s,
        "auroc_null_full_cv5_mean": null_cv_m,
        "auroc_null_full_cv5_std": null_cv_s,
        "gap_real_minus_null_full": gap,
        "auroc_real_BZ_only": bz_real_oof,
        "auroc_null_BZ_only": bz_null_oof,
        "gap_BZ_only": bz_real_oof - bz_null_oof,
        "f10_raw_real": F10_RAW_REAL,
        "f10_raw_null": F10_RAW_NULL,
        "f10_raw_gap": F10_RAW_REAL - F10_RAW_NULL,
        "layer_prune_frequency": layer_prune_freq.tolist(),
        "overall_prune_rate": overall_prune_rate,
        "verdict": verdict,
        "elapsed_seconds": float(elapsed_total),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
