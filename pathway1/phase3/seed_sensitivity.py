"""Phase 3: Fold Seed Sensitivity.

Test whether CORAL's result is brittle to the specific CV fold assignment
(random_state=42) held fixed across 676 attempts.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler

PHASE3_DIR = Path(__file__).parent
PHASE1_DIR = PHASE3_DIR.parent / "phase1"
PHASE0_DIR = PHASE3_DIR.parent / "phase0"
TOPO_DIR = PHASE3_DIR.parent.parent
TRAJ_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectories.npz"
META_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectory_meta.json"
LAYER_PATH = Path.home() / "att-docs" / "data" / "transformer" / "math500_hidden_states_aligned.npz"

sys.path.insert(0, str(PHASE0_DIR))
from winning_features import extract_features, PCA as OrigPCA

LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs")


def load_data():
    t = np.load(TRAJ_PATH)
    trajectories = [t[f"traj_{i}"] for i in range(500)]
    with open(META_PATH) as f:
        meta = json.load(f)
    y = np.array(meta["correct"], dtype=int)
    ls = np.load(LAYER_PATH)["layer_hidden_states"]
    return trajectories, y, ls


def run_cv(X, y, cv_seed=42, label=""):
    """Run 50-fold CV with a specific fold seed."""
    cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=cv_seed)
    clf = LogisticRegression(**LR_PARAMS)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, proba)
    return auroc


def make_prefitted_pca(traj_list):
    """Create a PCA fitted on the given trajectories."""
    from sklearn.decomposition import PCA as SkPCA
    concat = np.concatenate(traj_list, axis=0)
    n_comp = min(45, concat.shape[1], concat.shape[0])
    pca = SkPCA(n_components=n_comp, svd_solver="full")
    pca.fit(concat)
    return pca


def extract_with_train_pca(trajectories, layer_states, train_pca):
    """Extract features with monkey-patched train PCA."""
    from sklearn.decomposition import PCA as SkPCA

    class PreFittedPCA(SkPCA):
        _prefitted = None
        def fit(self, X, y=None):
            for attr in ['components_', 'mean_', 'explained_variance_',
                         'explained_variance_ratio_', 'singular_values_',
                         'n_components_', 'n_samples_', 'n_features_in_',
                         'noise_variance_']:
                setattr(self, attr, getattr(PreFittedPCA._prefitted, attr))
            return self

    PreFittedPCA._prefitted = train_pca

    import winning_features as wf_module
    original = wf_module.PCA
    wf_module.PCA = PreFittedPCA
    try:
        features, names = extract_features(trajectories, layer_states)
    finally:
        wf_module.PCA = original
    return features, names


def main():
    print("=" * 60)
    print("Phase 3: Fold Seed Sensitivity")
    print("=" * 60)

    trajectories, y, layer_states = load_data()

    # ==============================
    # Group 1: Vary CV fold seed on CV-500 (CORAL protocol)
    # ==============================
    print("\n[1] CV-500: varying fold seed (all-500 PCA as in CORAL)...")
    t0 = time.time()
    features_all, _ = extract_features(trajectories, layer_states)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")

    # Filter degenerate
    good_mask = np.array([features_all[:, j].std() > 1e-10 for j in range(features_all.shape[1])])
    X_all = features_all[:, good_mask]
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_all)

    cv500_seeds = [42, 43, 44, 45, 46]
    cv500_results = {}
    for seed in cv500_seeds:
        auroc = run_cv(X_all_scaled, y, cv_seed=seed)
        cv500_results[seed] = auroc
        print(f"  seed={seed}: AUROC={auroc:.6f}")

    # ==============================
    # Group 2: Vary CV fold seed on CV-train-400
    # ==============================
    print("\n[2] CV-train-400: varying fold seed (train-only PCA, seed=9999 split)...")
    mask = np.load(PHASE1_DIR / "holdout_mask_seed9999.npy")
    train_idx_9999 = np.where(~mask)[0]
    y_train_9999 = y[train_idx_9999]
    traj_train_9999 = [trajectories[i] for i in train_idx_9999]
    ls_train_9999 = layer_states[train_idx_9999]

    t0 = time.time()
    features_train_9999, _ = extract_features(traj_train_9999, ls_train_9999)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")

    good_mask_tr = np.array([features_train_9999[:, j].std() > 1e-10 for j in range(features_train_9999.shape[1])])
    X_train_9999 = features_train_9999[:, good_mask_tr]
    scaler_tr = StandardScaler()
    X_train_9999_scaled = scaler_tr.fit_transform(X_train_9999)

    cv_train_seeds = [42, 43, 44, 45, 46]
    cv_train_results = {}
    for seed in cv_train_seeds:
        auroc = run_cv(X_train_9999_scaled, y_train_9999, cv_seed=seed)
        cv_train_results[seed] = auroc
        print(f"  seed={seed}: AUROC={auroc:.6f}")

    # ==============================
    # Group 3: Vary holdout split seed
    # ==============================
    print("\n[3] Holdout: varying split seed...")
    split_seeds = [9999, 9998, 9997, 9996, 9995]
    holdout_results = {}

    for split_seed in split_seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=split_seed)
        tr_idx, ho_idx = next(sss.split(np.zeros(500), y))
        y_tr = y[tr_idx]
        y_ho = y[ho_idx]

        # Extract train features
        traj_tr = [trajectories[i] for i in tr_idx]
        ls_tr = layer_states[tr_idx]
        t0 = time.time()
        feat_tr, _ = extract_features(traj_tr, ls_tr)
        print(f"  seed={split_seed}: train extraction {time.time() - t0:.1f}s", end="")

        # Extract holdout features with train PCA
        train_pca = make_prefitted_pca(traj_tr)
        t0 = time.time()
        feat_all_trainpca, _ = extract_with_train_pca(trajectories, layer_states, train_pca)
        feat_ho = feat_all_trainpca[ho_idx]
        print(f", holdout extraction {time.time() - t0:.1f}s", end="")

        # Filter degenerate on train
        good = np.array([feat_tr[:, j].std() > 1e-10 for j in range(feat_tr.shape[1])])
        X_tr = feat_tr[:, good]
        X_ho = feat_ho[:, good]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_ho_s = scaler.transform(X_ho)

        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_ho_s)[:, 1]
        auroc = roc_auc_score(y_ho, proba)
        holdout_results[split_seed] = auroc
        print(f", AUROC={auroc:.6f}")

    # ==============================
    # Summary statistics
    # ==============================
    print("\n" + "=" * 60)
    print("Seed Sensitivity Summary")
    print("=" * 60)

    for label, results in [
        ("CV-500", cv500_results),
        ("CV-train-400", cv_train_results),
        ("Holdout (split seed)", holdout_results),
    ]:
        vals = np.array(list(results.values()))
        print(f"\n  {label}:")
        print(f"    Seeds: {list(results.keys())}")
        print(f"    AUROCs: {[f'{v:.6f}' for v in vals]}")
        print(f"    Mean: {vals.mean():.6f}")
        print(f"    Std:  {vals.std():.6f}")
        print(f"    Min:  {vals.min():.6f}")
        print(f"    Max:  {vals.max():.6f}")

    # Interpret
    cv500_std = np.std(list(cv500_results.values()))
    holdout_std = np.std(list(holdout_results.values()))

    print("\n  Interpretation:")
    if cv500_std < 0.005:
        print(f"  CV-500 std={cv500_std:.4f} < 0.005: CORAL AUROC is stable across fold assignments.")
    elif cv500_std < 0.02:
        print(f"  CV-500 std={cv500_std:.4f} in [0.005, 0.02]: moderate fold sensitivity.")
    else:
        print(f"  CV-500 std={cv500_std:.4f} > 0.02: CORAL AUROC is fold-fragile.")

    if holdout_std > 0.03:
        print(f"  Holdout std={holdout_std:.4f} > 0.03: n=100 holdout is noisy for fine-grained decisions.")
    else:
        print(f"  Holdout std={holdout_std:.4f} <= 0.03: holdout noise is manageable.")

    # Save
    with open(PHASE3_DIR / "seed_sensitivity_cv500.json", "w") as f:
        json.dump({str(k): v for k, v in cv500_results.items()}, f, indent=2)
    with open(PHASE3_DIR / "seed_sensitivity_cv_train400.json", "w") as f:
        json.dump({str(k): v for k, v in cv_train_results.items()}, f, indent=2)
    with open(PHASE3_DIR / "seed_sensitivity_holdout.json", "w") as f:
        json.dump({str(k): v for k, v in holdout_results.items()}, f, indent=2)

    summary = []
    summary.append("Seed Sensitivity Summary (Phase 3)")
    summary.append("=" * 50)
    for label, results in [
        ("CV-500 (fold seeds 42-46)", cv500_results),
        ("CV-train-400 (fold seeds 42-46)", cv_train_results),
        ("Holdout (split seeds 9999-9995)", holdout_results),
    ]:
        vals = np.array(list(results.values()))
        summary.append(f"\n{label}:")
        for k, v in results.items():
            summary.append(f"  seed={k}: {v:.6f}")
        summary.append(f"  Mean={vals.mean():.6f}, Std={vals.std():.6f}, "
                       f"Min={vals.min():.6f}, Max={vals.max():.6f}")

    with open(PHASE3_DIR / "seed_sensitivity_summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    print(f"\n  Artifacts saved to {PHASE3_DIR}/")


if __name__ == "__main__":
    main()
