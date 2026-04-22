"""Exp 3a v2 rerun — full-covariance Gaussian null, H0_n_features dropped.

Addresses audit defects D1 + D2:
  D1 — original used diagonal variance, not full covariance.
  D2 — original included H0_n_features which is structurally n_points-1
       (identical in real and null by construction).

Uses low-rank SVD sampling: cov of an (n_tokens, 1536) cloud has rank
<= n_tokens-1, so we sample in that low-rank subspace instead of
Cholesky-decomposing a 1536x1536 cov matrix.

Output: pathway9/results/exp3a_gaussian_null_v2.json + exp3a_v2.done
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def compute_5_ph_features(cloud, subsample=100, pca_components=45):
    """5 PH features (H0 max_lifetime, H0 entropy, H1 pers_entropy, H1 n_features, H1 max_lifetime)."""
    if len(cloud) < 5:
        return np.zeros(5)
    n_comp = min(pca_components, cloud.shape[0] - 1, cloud.shape[1])
    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)
    if len(cloud_pca) > subsample:
        idx = np.linspace(0, len(cloud_pca) - 1, subsample, dtype=int)
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


def low_rank_gaussian_null(cloud, n_nulls=5, seed=None):
    """Sample from N(mean, cov) using SVD of centered cloud (low-rank, fast).

    For an (n, d) cloud with n << d, cov has rank <= n-1. Sample as:
        synthetic = mean + randn(n, rank) @ (s/sqrt(n-1))[:, None] * Vt
    where (U, s, Vt) = svd(cloud - mean).
    This gives a Gaussian with exactly the same mean and covariance as the
    empirical distribution, at O(n * d) per draw instead of O(d^3).
    """
    n, d = cloud.shape
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


def load_data():
    new_labels = np.load(REPO / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy")
    old_labels = np.load(REPO / "pathway2/track_a/phase0/baseline_correct.npy")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_sss, hold_sss = next(sss.split(np.zeros(500), old_labels.astype(int)))
    return dict(
        new_labels=new_labels,
        train_idx=np.sort(train_sss),
        holdout_idx=np.sort(hold_sss),
    )


def fit_lr(X_train, y_train, X_holdout, y_holdout, cv=5):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    auroc_ho = roc_auc_score(y_holdout, pipe.predict_proba(X_holdout)[:, 1])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    return float(auroc_ho), float(cv_scores.mean()), float(cv_scores.std())


def main():
    print("=" * 60)
    print("EXP 3A v2: Full-covariance Gaussian null (SVD low-rank sampler)")
    print("=" * 60, flush=True)

    d = load_data()
    traj_file = np.load(REPO / "data/experiment1_v2/trajectories.npz", allow_pickle=True)
    n_problems = 500

    real_ph = np.zeros((n_problems, 5))
    null_ph = np.zeros((n_problems, 5))

    t0 = time.time()
    for i in range(n_problems):
        cloud = traj_file[f"traj_{i}"].astype(np.float64)
        real_ph[i] = compute_5_ph_features(cloud)
        null_ph[i] = low_rank_gaussian_null(cloud, n_nulls=5, seed=42 + i)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_problems - i - 1)
            print(f"  {i+1}/{n_problems}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    names = ["H0_max_lifetime", "H0_entropy", "H1_pers_entropy", "H1_n_features", "H1_max_lifetime"]
    feature_comparison = {}
    print("\n  Feature comparison (real vs full-cov null):")
    for j, name in enumerate(names):
        rm, rs = float(real_ph[:, j].mean()), float(real_ph[:, j].std())
        nm, ns = float(null_ph[:, j].mean()), float(null_ph[:, j].std())
        print(f"    {name:20s} real={rm:8.2f}±{rs:.2f}  null={nm:8.2f}±{ns:.2f}  diff={rm-nm:+.2f}", flush=True)
        feature_comparison[name] = {"real_mean": rm, "real_std": rs, "null_mean": nm, "null_std": ns, "diff_mean": rm - nm}

    y_tr = d["new_labels"][d["train_idx"]]
    y_ho = d["new_labels"][d["holdout_idx"]]

    ph_results = {}
    for name, feats in (("real_ph", real_ph), ("null_ph", null_ph), ("diff_ph", real_ph - null_ph)):
        f_tr, f_ho = feats[d["train_idx"]], feats[d["holdout_idx"]]
        auroc_ho, cv_mean, cv_std = fit_lr(f_tr, y_tr, f_ho, y_ho, cv=5)
        ph_results[name] = {"auroc_holdout": auroc_ho, "auroc_cv5": cv_mean, "cv_std": cv_std}
        print(f"  {name:10s}: holdout={auroc_ho:.3f}, CV5={cv_mean:.3f}±{cv_std:.3f}", flush=True)

    real_auroc = ph_results["real_ph"]["auroc_holdout"]
    null_auroc = ph_results["null_ph"]["auroc_holdout"]
    diff_auroc = ph_results["diff_ph"]["auroc_holdout"]

    if null_auroc >= real_auroc - 0.02:
        verdict = (
            f"These 5 raw PH features (H0_n_features dropped) do NOT outperform a "
            f"full-covariance Gaussian null (real={real_auroc:.3f}, null={null_auroc:.3f}). "
            f"The predictive content in these summary statistics is explained by matched "
            f"covariance structure. Narrow claim: these 5 features do not carry "
            f"topology-specific signal. The ABC-44 production features include geometric "
            f"terms (cosines, norms, velocities) NOT tested here."
        )
    elif diff_auroc > real_auroc + 0.01:
        verdict = f"Topology-specific signal exists: diff beats raw ({diff_auroc:.3f} > {real_auroc:.3f})."
    else:
        verdict = f"Mixed — real={real_auroc:.3f}, null={null_auroc:.3f}, diff={diff_auroc:.3f}."

    print(f"\n  VERDICT: {verdict}", flush=True)

    results = {
        "experiment": "exp3a_full_cov_null_v2",
        "n_problems": n_problems,
        "n_features": 5,
        "n_nulls_per_problem": 5,
        "null_type": "full_covariance_via_SVD_of_centered_cloud",
        "dropped_feature": "H0_n_features",
        "classifiers": ph_results,
        "feature_comparison": feature_comparison,
        "verdict": verdict,
    }
    (RESULTS / "exp3a_gaussian_null_v2.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp3a_v2.done").touch()
    print("\n  Saved: pathway9/results/exp3a_gaussian_null_v2.json")


if __name__ == "__main__":
    main()
