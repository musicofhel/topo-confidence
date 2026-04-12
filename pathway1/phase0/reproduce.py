"""Phase 0: Reproduce CORAL's 0.9779 AUROC and measure statefulness leakage.

Steps:
1. Load all data, call extract_features on all 500 samples (matching CORAL).
2. StandardScaler.fit_transform(features).
3. StratifiedKFold(50, shuffle=True, random_state=42).
4. LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42).
5. cross_val_predict -> roc_auc_score.
6. Measure PCA/rank-bin leakage via throwaway train-400/holdout-100 split.
7. Lock artifacts by hash.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---- Paths ----
PHASE0_DIR = Path(__file__).parent
TOPO_DIR = PHASE0_DIR.parent.parent  # ~/topo-confidence
TRAJ_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectories.npz"
META_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectory_meta.json"
LAYER_PATH = Path.home() / "att-docs" / "data" / "transformer" / "math500_hidden_states_aligned.npz"
WF_PATH = PHASE0_DIR / "winning_features.py"

# ---- Import winning_features from local copy ----
sys.path.insert(0, str(PHASE0_DIR))
from winning_features import extract_features


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    """Load trajectories, labels, and layer states."""
    t = np.load(TRAJ_PATH)
    trajectories = [t[f"traj_{i}"] for i in range(500)]
    with open(META_PATH) as f:
        meta = json.load(f)
    y = np.array(meta["correct"], dtype=int)
    ls = np.load(LAYER_PATH)["layer_hidden_states"]
    return trajectories, y, ls


def run_cv(X, y, label=""):
    """Run 50-fold stratified CV with LR, return AUROC."""
    cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs",
    )
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, proba)
    print(f"  {label} AUROC = {auroc:.6f}")
    return auroc


def main():
    print("=" * 60)
    print("Phase 0: Reproduce & Lock")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1] Loading data...")
    t0 = time.time()
    trajectories, y, layer_states = load_data()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Trajectories: {len(trajectories)}, labels sum={y.sum()}")
    print(f"  Layer states: {layer_states.shape}")

    # Step 2: Extract features on all 500 (matching CORAL)
    print("\n[2] Extracting features (all-500 PCA, matching CORAL)...")
    t0 = time.time()
    features_all, names_all = extract_features(trajectories, layer_states)
    print(f"  Extracted in {time.time() - t0:.1f}s")
    print(f"  Feature matrix: {features_all.shape}, names: {len(names_all)}")
    assert np.isfinite(features_all).all(), "Non-finite features detected!"

    # Step 3: Filter degenerate features (matching CORAL grader exactly)
    good_mask = np.array([
        features_all[:, j].std() > 1e-10 for j in range(features_all.shape[1])
    ])
    n_good = good_mask.sum()
    n_degen = features_all.shape[1] - n_good
    if n_degen > 0:
        degen_names = [names_all[j] for j in range(len(names_all)) if not good_mask[j]]
        print(f"  Degenerate features ({n_degen}): {degen_names}")
    else:
        print(f"  No degenerate features (all {n_good} have std > 1e-10)")
    X_good = features_all[:, good_mask]

    # StandardScaler fit_transform on good features only
    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(X_good)

    # Step 4: Reproduce CORAL's AUROC
    print("\n[3] Reproducing CORAL's 0.9779 AUROC (50-fold CV on all 500)...")
    auroc_cv500 = run_cv(X_all, y, label="CV-500")

    if abs(auroc_cv500 - 0.9779) > 0.001:
        print(f"\n  *** HALT: reproduced AUROC {auroc_cv500:.6f} is outside "
              f"tolerance of 0.9779 +/- 0.001 ***")
        # Don't sys.exit — still save what we have for debugging
    else:
        print(f"  MATCH: {auroc_cv500:.6f} is within tolerance of 0.9779 +/- 0.001")

    # Step 5: Measure statefulness leakage
    print("\n[4] Measuring statefulness leakage (throwaway split, seed=12345)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=12345)
    train_idx, hold_idx = next(sss.split(np.zeros(500), y))
    print(f"  Train: {len(train_idx)}, Holdout: {len(hold_idx)}")
    print(f"  Train correct: {y[train_idx].sum()}/400, Holdout correct: {y[hold_idx].sum()}/100")

    # (a) All-500 path: features already computed above
    feat_a_train = features_all[train_idx]

    # (b) Train-only PCA path: need to re-extract with only train trajectories
    # We need to rebuild extract_features with train-only PCA.
    # Since extract_features is monolithic and fits PCA on all input,
    # we can call it on just the train trajectories, then separately on holdout
    # using a modified approach. But the function is frozen — we can't modify it.
    #
    # Instead: call extract_features on all 500, but also call it on just the
    # 400 train trajectories. Compare the train-400 feature values.
    print("  Extracting features (train-400 only PCA)...")
    t0 = time.time()
    traj_train = [trajectories[i] for i in train_idx]
    ls_train = layer_states[train_idx]
    feat_b_train, _ = extract_features(traj_train, ls_train)
    print(f"  Extracted in {time.time() - t0:.1f}s")
    print(f"  Train-only feature matrix: {feat_b_train.shape}")

    # Compare: mean absolute difference per feature column
    # feat_a_train: train-400 rows from all-500 extraction
    # feat_b_train: train-400 rows from train-only extraction
    abs_diff = np.abs(feat_a_train - feat_b_train)
    mean_abs_diff = abs_diff.mean(axis=0)
    max_feature_drift = float(mean_abs_diff.max())

    print(f"\n  Leakage analysis (400 train samples, {len(names_all)} features):")
    print(f"  Mean abs diff per feature (min/median/max): "
          f"{mean_abs_diff.min():.6f} / {np.median(mean_abs_diff):.6f} / {max_feature_drift:.6f}")

    flagged = []
    for j, (name, mad) in enumerate(zip(names_all, mean_abs_diff)):
        if mad > 0.1:
            flagged.append((name, float(mad)))
            print(f"  ** FLAGGED: {name} (col {j}): mean_abs_diff = {mad:.4f}")

    if not flagged:
        print("  No features flagged (all mean_abs_diff <= 0.1)")
    else:
        print(f"  {len(flagged)} feature(s) flagged with mean_abs_diff > 0.1")

    # Step 6: Lock artifacts by hash
    print("\n[5] Computing artifact hashes...")
    hashes = {
        "winning_features_sha256": sha256_file(WF_PATH),
        "trajectories_sha256": sha256_file(TRAJ_PATH),
        "trajectory_meta_sha256": sha256_file(META_PATH),
        "layer_states_sha256": sha256_file(LAYER_PATH),
    }
    for name, h in hashes.items():
        print(f"  {name}: {h[:16]}...")

    locks = {
        **hashes,
        "reproduced_auroc": round(float(auroc_cv500), 6),
        "lr_params": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "solver": "lbfgs",
        },
        "cv_params": {"n_splits": 50, "shuffle": True, "random_state": 42},
        "pca_params": {"n_components": 45, "svd_solver": "full"},
        "scaler": "StandardScaler",
        "leakage_magnitude": {
            "mean_abs_diff_per_feature": [round(float(x), 6) for x in mean_abs_diff],
            "max_feature_drift": round(max_feature_drift, 6),
            "flagged_features": flagged,
        },
        "n_output_features": features_all.shape[1],
        "feature_names": names_all,
    }

    locks_path = PHASE0_DIR / "artifact_locks.json"
    with open(locks_path, "w") as f:
        json.dump(locks, f, indent=2)
    print(f"\n  Saved artifact_locks.json ({locks_path})")

    # Summary
    print("\n" + "=" * 60)
    print("Phase 0 Summary")
    print("=" * 60)
    print(f"  Reproduced AUROC:  {auroc_cv500:.6f}")
    print(f"  Target:            0.9779 +/- 0.001")
    match = abs(auroc_cv500 - 0.9779) <= 0.001
    print(f"  Match:             {'YES' if match else 'NO'}")
    print(f"  Features:          {features_all.shape[1]}")
    print(f"  Max feature drift: {max_feature_drift:.6f}")
    print(f"  Flagged features:  {len(flagged)}")
    if match:
        print("\n  Phase 0 PASSED. Proceed to Phase 1.")
    else:
        print(f"\n  Phase 0 FAILED. Investigate before proceeding.")

    return auroc_cv500, match


if __name__ == "__main__":
    auroc, passed = main()
    sys.exit(0 if passed else 1)
