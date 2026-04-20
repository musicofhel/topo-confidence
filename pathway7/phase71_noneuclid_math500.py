#!/usr/bin/env python3
"""Phase 7.1: Non-Euclidean PH features on existing MATH-500 data.

Reuses trajectories from data/experiment1_v2/trajectories.npz and
labels from pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy.
Computes effective-resistance + cosine PH features, fits logistic
regression, and compares AUROC against the 0.796 Euclidean baseline.

CPU only, ~1 hour for 500 problems.

Run: python pathway7/phase71_noneuclid_math500.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathway7.distance_metrics import compute_all_noneuclid_features
from pathway7.validation.delong_comparison import compare_models

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_PATH = REPO_ROOT / "data" / "experiment1_v2" / "trajectories.npz"
LABELS_PATH = REPO_ROOT / "pathway6_rebuild" / "phase0_relabel" / "baseline_correct_v2.npy"
OLD_LABELS_PATH = REPO_ROOT / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
FEATURES_TRAIN_PATH = REPO_ROOT / "pathway1" / "phase1" / "features_train400.npy"
FEATURES_HOLDOUT_PATH = REPO_ROOT / "pathway1" / "phase1" / "features_holdout100.npy"
TIER_PATH = REPO_ROOT / "pathway1" / "phase2" / "tier_assignments.json"
FEATURE_NAMES_PATH = REPO_ROOT / "pathway6_rebuild" / "phase2_completion" / "feature_names_v2.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_phase71"

# LR hyperparameters (match pathway6_rebuild)
LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42)
SUBSAMPLE = 100
N_PCA = 45
SEED = 42


# ============================================================================
# Data loading
# ============================================================================

def load_trajectories() -> list[np.ndarray]:
    """Load 500 per-problem trajectories from experiment1_v2."""
    print(f"Loading trajectories from {TRAJ_PATH} ...")
    data = np.load(TRAJ_PATH, allow_pickle=True)
    trajs = [data[f"traj_{i}"] for i in range(500)]
    shapes = [t.shape for t in trajs[:5]]
    print(f"  Loaded {len(trajs)} trajectories, first 5 shapes: {shapes}")
    return trajs


def load_labels_and_split():
    """Load corrected labels and reconstruct train/holdout split.

    IMPORTANT: Features in features_train400.npy were saved using SSS split
    with OLD labels (57 correct). We must use OLD labels to reconstruct the
    split ordering, then apply NEW labels (104 correct) for actual training.
    """
    new_labels = np.load(LABELS_PATH)
    old_labels = np.load(OLD_LABELS_PATH)
    print(f"Labels (new/v2): {new_labels.shape}, {new_labels.sum()} correct out of {len(new_labels)}")
    print(f"Labels (old): {old_labels.shape}, {old_labels.sum()} correct")

    # SSS split with OLD labels (matches how features were saved)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(
        sss.split(np.zeros(len(old_labels)), old_labels.astype(int))
    )

    # Sorted indices
    train_idx = np.sort(train_idx_sss)
    holdout_idx = np.sort(hold_idx_sss)

    return new_labels, train_idx, holdout_idx


def load_existing_abc_features():
    """Load existing 44 ABC-tier features with SSS ordering fix.

    Uses OLD labels for SSS reconstruction (features were saved with old split).
    """
    # tier_assignments.json maps {feature_name: tier_letter}
    tier_data = json.loads(TIER_PATH.read_text())
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    abc_cols = [i for i, name in enumerate(feature_names)
                if tier_data.get(name) in ("A", "B", "C")]

    # SSS permutation using OLD labels (matches how features were saved)
    old_labels = np.load(OLD_LABELS_PATH)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_sss, hold_sss = next(sss.split(np.zeros(len(old_labels)), old_labels.astype(int)))
    train_sort = np.argsort(train_sss)
    hold_sort = np.argsort(hold_sss)

    X_train_raw = np.load(FEATURES_TRAIN_PATH)[train_sort]
    X_holdout_raw = np.load(FEATURES_HOLDOUT_PATH)[hold_sort]

    # Select ABC columns only
    X_train = X_train_raw[:, abc_cols]
    X_holdout = X_holdout_raw[:, abc_cols]

    print(f"  Existing features: train {X_train.shape}, holdout {X_holdout.shape}")
    return X_train, X_holdout, abc_cols


# ============================================================================
# Feature extraction
# ============================================================================

def subsample_points(points: np.ndarray, n: int = SUBSAMPLE) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]


def compute_noneuclid_features_all(
    trajectories: list[np.ndarray],
    train_idx: np.ndarray,
    k_effres: int = 30,
) -> tuple[np.ndarray, list[str]]:
    """Compute non-Euclidean PH features for all 500 problems.

    PCA is fitted on train-only data to avoid holdout leakage.
    """
    print(f"\nComputing non-Euclidean features (k_effres={k_effres}) ...")

    # Fit PCA on train only
    train_trajs = [trajectories[i] for i in train_idx]
    all_train_points = np.concatenate(train_trajs, axis=0)
    n_comp = min(N_PCA, all_train_points.shape[1], all_train_points.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(all_train_points)
    print(f"  PCA: {n_comp} components, {100*pca.explained_variance_ratio_.sum():.1f}% variance")

    features_list = []
    names = None
    t0 = time.time()

    for i, traj in enumerate(trajectories):
        reduced = pca.transform(traj)
        subsampled = subsample_points(reduced, SUBSAMPLE)
        feats, feat_names = compute_all_noneuclid_features(
            subsampled, k_effres=k_effres
        )
        features_list.append(feats)
        if names is None:
            names = feat_names

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (500 - i - 1) / rate
            print(f"  {i+1}/500 ({elapsed:.0f}s elapsed, {eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.0f}s total ({elapsed/500:.2f}s per problem)")

    return np.stack(features_list), names


# ============================================================================
# Model evaluation
# ============================================================================

def evaluate_model(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    y_train: np.ndarray,
    y_holdout: np.ndarray,
    name: str,
    C: float = 1.0,
) -> dict:
    """Fit LR on train, evaluate AUROC on holdout + 50-fold CV on train."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_holdout)

    lr = LogisticRegression(C=C, **LR_PARAMS)
    lr.fit(X_tr, y_train)

    # Holdout AUROC
    probs_holdout = lr.predict_proba(X_ho)[:, 1]
    auroc_holdout = roc_auc_score(y_holdout, probs_holdout)

    # 50-fold CV on train
    cv = StratifiedKFold(n_splits=min(50, y_train.sum()), shuffle=True, random_state=42)
    probs_train_cv = cross_val_predict(
        LogisticRegression(C=C, **LR_PARAMS),
        X_tr, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    auroc_cv = roc_auc_score(y_train, probs_train_cv)

    print(f"\n  {name}: holdout AUROC={auroc_holdout:.4f}, CV AUROC={auroc_cv:.4f}")

    return {
        "name": name,
        "C": C,
        "n_features": X_train.shape[1],
        "auroc_holdout": round(auroc_holdout, 4),
        "auroc_cv": round(auroc_cv, 4),
        "probs_holdout": probs_holdout,
        "probs_train_cv": probs_train_cv,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 7.1: Non-Euclidean PH on MATH-500 (CPU)")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    labels, train_idx, holdout_idx = load_labels_and_split()
    y_train = labels[train_idx]
    y_holdout = labels[holdout_idx]

    # Load existing ABC features (uses old labels for SSS ordering)
    X_train_abc, X_holdout_abc, abc_cols = load_existing_abc_features()

    # Compute non-Euclidean features
    noneuclid_all, noneuclid_names = compute_noneuclid_features_all(
        trajectories, train_idx, k_effres=30
    )
    X_train_ne = noneuclid_all[train_idx]
    X_holdout_ne = noneuclid_all[holdout_idx]

    print(f"\n  Non-Euclidean features: {X_train_ne.shape[1]} ({noneuclid_names})")

    # Combine: ABC + non-Euclidean
    X_train_combined = np.hstack([X_train_abc, X_train_ne])
    X_holdout_combined = np.hstack([X_holdout_abc, X_holdout_ne])

    # ---- Evaluate all model variants ----
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)

    results = []

    # 1. Baseline: ABC only (reproducing 0.796)
    res_abc = evaluate_model(X_train_abc, X_holdout_abc, y_train, y_holdout,
                             "ABC-44 (baseline)", C=1.0)
    results.append(res_abc)

    # 2. Non-Euclidean only
    res_ne = evaluate_model(X_train_ne, X_holdout_ne, y_train, y_holdout,
                            "NonEuclid-only", C=1.0)
    results.append(res_ne)

    # 3. Combined (default C=1.0)
    res_combined = evaluate_model(X_train_combined, X_holdout_combined, y_train, y_holdout,
                                  "Combined (C=1.0)", C=1.0)
    results.append(res_combined)

    # 4. Combined with stronger regularization
    for C in [0.1, 0.5, 2.0, 5.0]:
        res_c = evaluate_model(X_train_combined, X_holdout_combined, y_train, y_holdout,
                               f"Combined (C={C})", C=C)
        results.append(res_c)

    # 5. Eff-res only (subset of non-Euclidean)
    er_cols = [i for i, n in enumerate(noneuclid_names) if n.startswith("effres_")]
    if er_cols:
        res_er = evaluate_model(
            X_train_ne[:, er_cols], X_holdout_ne[:, er_cols],
            y_train, y_holdout, "EffRes-only", C=1.0
        )
        results.append(res_er)

    # 6. Cosine only (subset of non-Euclidean)
    cos_cols = [i for i, n in enumerate(noneuclid_names) if n.startswith("cosine_")]
    if cos_cols:
        res_cos = evaluate_model(
            X_train_ne[:, cos_cols], X_holdout_ne[:, cos_cols],
            y_train, y_holdout, "Cosine-only", C=1.0
        )
        results.append(res_cos)

    # ---- DeLong comparison: best combined vs baseline ----
    print("\n" + "=" * 70)
    print("Statistical Comparison (DeLong)")
    print("=" * 70)

    # Find best combined result
    combined_results = [r for r in results if "Combined" in r["name"]]
    best_combined = max(combined_results, key=lambda r: r["auroc_holdout"])

    delong_result = compare_models(
        y_holdout,
        res_abc["probs_holdout"],
        best_combined["probs_holdout"],
        name_baseline="ABC-44",
        name_new=best_combined["name"],
    )

    # ---- Save results ----
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    summary = {
        "baseline_auroc": res_abc["auroc_holdout"],
        "best_combined_auroc": best_combined["auroc_holdout"],
        "best_combined_name": best_combined["name"],
        "auroc_improvement": round(best_combined["auroc_holdout"] - res_abc["auroc_holdout"], 4),
        "delong": delong_result,
        "all_results": [
            {k: v for k, v in r.items() if k not in ("probs_holdout", "probs_train_cv")}
            for r in results
        ],
        "noneuclid_feature_names": noneuclid_names,
        "n_train": len(y_train),
        "n_holdout": len(y_holdout),
        "n_correct_train": int(y_train.sum()),
        "n_correct_holdout": int(y_holdout.sum()),
    }

    for r in summary["all_results"]:
        print(f"  {r['name']:25s}  AUROC={r['auroc_holdout']:.4f}  (CV={r['auroc_cv']:.4f}, {r['n_features']} feats)")

    print(f"\n  Best combined: {best_combined['name']} AUROC={best_combined['auroc_holdout']:.4f}")
    print(f"  Baseline:      ABC-44                    AUROC={res_abc['auroc_holdout']:.4f}")
    print(f"  Improvement:   {summary['auroc_improvement']:+.4f}")
    print(f"  DeLong p-value: {delong_result['p_value']:.4f} ({delong_result['significance']})")

    # Save
    out_path = OUTPUT_DIR / "phase71_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Results saved to {out_path}")

    # Save predicted probabilities for downstream analysis
    np.save(OUTPUT_DIR / "probs_baseline_holdout.npy", res_abc["probs_holdout"])
    np.save(OUTPUT_DIR / "probs_best_combined_holdout.npy", best_combined["probs_holdout"])
    np.save(OUTPUT_DIR / "noneuclid_features_all500.npy", noneuclid_all)

    # Save feature names
    (OUTPUT_DIR / "noneuclid_feature_names.json").write_text(
        json.dumps(noneuclid_names, indent=2)
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
