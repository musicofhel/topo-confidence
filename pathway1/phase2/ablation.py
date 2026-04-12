"""Phase 2: Feature Ablation by Tier.

Tier A: Baseline PH + geometric features (pre-CORAL)
Tier B: Single transforms, layer features, simple products (depth-1)
Tier C: Depth-2 products (products of Tier B with Tier A/B)
Tier D: Depth-3+ recursive products (p3_, p4_, p5_, p6_, p7_, p8_)
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

PHASE2_DIR = Path(__file__).parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"
PHASE0_DIR = PHASE2_DIR.parent / "phase0"
TOPO_DIR = PHASE2_DIR.parent.parent
TRAJ_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectories.npz"
META_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectory_meta.json"
LAYER_PATH = Path.home() / "att-docs" / "data" / "transformer" / "math500_hidden_states_aligned.npz"

sys.path.insert(0, str(PHASE0_DIR))
from winning_features import extract_features, FEATURE_NAMES, PCA as OrigPCA

LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs")
CV_PARAMS = dict(n_splits=50, shuffle=True, random_state=42)

# ---- Tier assignments ----
# Based on analysis of winning_features.py:
# Output feature index -> (name, tier)
# drop_cols from winning_features.py
DROP_COLS = [0, 10, 11, 12, 15, 17, 19, 28, 29, 34, 42, 47, 50, 55, 56, 57, 58, 61, 64, 67, 69, 80]

# Get output feature names
OUT_NAMES = [nm for k, nm in enumerate(FEATURE_NAMES) if k not in DROP_COLS]

# Tier assignments by output feature name
# Tier A: baseline PH + geometric (pre-CORAL, raw measurements)
TIER_A = {
    "H0_max_lifetime",            # PH baseline (squared transform, but raw PH)
    "H1_persistence_entropy",     # PH baseline (exp transform)
    "first_token_norm",           # geometric baseline (rank-binned)
    "H1_total_persistence",       # PH baseline (log transform)
    "pairwise_dist_mean",         # geometric baseline (rank-binned)
    "norm_mean",                  # geometric baseline (rank-binned)
    "last_token_centroid_dist",   # geometric baseline (rank-binned, scale-normalized)
    "last5_centroid_dist",        # geometric baseline (rank-binned, scale-normalized)
    "last_token_layer_alignment", # geometric baseline (rank-binned)
}

# Tier B: depth-1 (single transforms of A, pairwise products of A, novel layer features)
TIER_B = {
    # Novel layer-state features (first exploited by CORAL)
    "layer_pca_var_ratio",        # SVD of layer_states
    "layer_pc2_early_ratio",      # SVD of early layers
    "tok_layer_min_dist",         # cross-modal geometry (rank-binned)
    "layer_angle_min_plus_max",   # layer curvature (rank-binned)
    "crossmodal_sq",              # tok_layer_min_dist squared (depth-1: sq of B)
    # Layer cosines (raw layer-state geometry)
    "cos_l9_l28",                 # layer cosine (rank-binned)
    "cos_l0_l5",                  # layer cosine (rank-binned)
    "cos_l11_l23",                # layer cosine (rank-binned)
    "cos_l12_avg_l24_l28",        # avg of two cosines (rank-binned)
    "cos_l23_l26",                # layer cosine (rank-binned)
    "cos_l5_l13",                 # layer cosine (rank-binned)
    "cos_l27_l28",                # layer cosine (rank-binned)
    "cos_l17_l27",                # layer cosine raw
    "cos_l7_l11",                 # layer cosine raw
    "cos_l0_l13",                 # layer cosine raw
    "cos_l4_l11",                 # layer cosine raw
    "cos_l4_l10",                 # layer cosine raw
    "cos_l13_l15",                # layer cosine raw
    "cos_l16_l28_bin",            # layer cosine binned
    "cos_l14_l28_bin",            # layer cosine binned
    # Norm ratios, velocities, accelerations (raw layer dynamics)
    "norm_ratio_l19_l18",         # norm ratio raw
    "vel_l26_l27",                # velocity (rank-binned)
    "accel_l14",                  # acceleration raw
    # SVD spectrum features × baseline PH (products of B × A)
    "sv3_x_h0max_sq",            # sv3 (B) × H0_max_lifetime_sq (A) -> depth-1
    "jerk_l7_x_h0tot_sq",        # jerk (B) × H0_tot_sq (A) -> depth-1
    "jerk_l8_x_h0maxl_sq",       # jerk (B) × H0_max_sq (A) -> depth-1
    # Single-cross products of layer feature × baseline PH
    "accel_l4_x_h0ent35_bin",    # accel (B) × h0ent35 (A, dropped but used) -> depth-1
    "accel_l15_x_h1tot_raw",     # accel (B) × H1tot_log (A) -> depth-1
    # Norm ratio × baseline PH products
    "nr28_15_x_h0ent35",         # nratio (B) × h0ent35 (A) -> depth-1
    "nr28_14_x_h0ent35",         # nratio (B) × h0ent35 (A) -> depth-1
}

# Tier C: depth-2 products (products of B×B or B×A, ADD #287-#288 era)
TIER_C = {
    # Cross-products of layer cosine (B) × PH (A)
    "bc25_27_x_h1ent",           # bin(cos25_27) × h1ent_exp: B × A -> depth-2
    "c27_28_x_h0tot",            # cos27_28 × h0tot_sq: B × A -> depth-2
    "bc7_26_x_h0tightfine",      # bin(cos7_26 × h0tight_fine): B × A -> depth-2 (h0tight*fine is A×A=A)
    # Interaction products involving tk_vel_max or snap/jerk
    "interact_tk_x_s21_rb",      # rb(tk*h0e35 * snap*h0tf): (A×A) × (B×A) -> depth-2
    "interact_tk_x_j15_rb",      # rb(tk*h0tf * jerk*h0tot): (A×A) × (B×A) -> depth-2
}

# Everything else is Tier D (depth-3+)
# Verify assignments cover all features
_assigned = TIER_A | TIER_B | TIER_C
TIER_D = set(OUT_NAMES) - _assigned


def load_data():
    t = np.load(TRAJ_PATH)
    trajectories = [t[f"traj_{i}"] for i in range(500)]
    with open(META_PATH) as f:
        meta = json.load(f)
    y = np.array(meta["correct"], dtype=int)
    ls = np.load(LAYER_PATH)["layer_hidden_states"]
    return trajectories, y, ls


def get_tier_indices(feature_names, tier_set):
    """Get column indices for features in the given tier set."""
    return [i for i, nm in enumerate(feature_names) if nm in tier_set]


def run_cv_subset(X, y, col_indices, label=""):
    """Run CV on a subset of feature columns."""
    if len(col_indices) == 0:
        print(f"  {label}: no features, skipping")
        return None

    X_sub = X[:, col_indices]
    # Filter degenerate
    good = np.array([X_sub[:, j].std() > 1e-10 for j in range(X_sub.shape[1])])
    if good.sum() == 0:
        print(f"  {label}: all {X_sub.shape[1]} features degenerate, skipping")
        return None
    X_sub = X_sub[:, good]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    cv = StratifiedKFold(**CV_PARAMS)
    clf = LogisticRegression(**LR_PARAMS)
    proba = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, proba)
    print(f"  {label}: AUROC={auroc:.6f} ({good.sum()}/{X_sub.shape[1] + (len(col_indices) - X_sub.shape[1] - (~good).sum())} features)")
    return auroc


def holdout_eval(features_train, y_train, features_holdout, y_holdout, col_indices, label=""):
    """Train on train, score on holdout for a subset of features."""
    if len(col_indices) == 0:
        return None

    X_tr = features_train[:, col_indices]
    X_ho = features_holdout[:, col_indices]

    # Filter degenerate on train
    good = np.array([X_tr[:, j].std() > 1e-10 for j in range(X_tr.shape[1])])
    if good.sum() == 0:
        return None
    X_tr = X_tr[:, good]
    X_ho = X_ho[:, good]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_ho_scaled = scaler.transform(X_ho)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_tr_scaled, y_train)
    proba = clf.predict_proba(X_ho_scaled)[:, 1]
    auroc = roc_auc_score(y_holdout, proba)
    print(f"  {label} holdout: AUROC={auroc:.6f}")
    return auroc


def main():
    print("=" * 60)
    print("Phase 2: Feature Ablation by Tier")
    print("=" * 60)

    # Save tier assignments
    tier_assignments = {}
    for nm in OUT_NAMES:
        if nm in TIER_A:
            tier_assignments[nm] = "A"
        elif nm in TIER_B:
            tier_assignments[nm] = "B"
        elif nm in TIER_C:
            tier_assignments[nm] = "C"
        else:
            tier_assignments[nm] = "D"

    with open(PHASE2_DIR / "tier_assignments.json", "w") as f:
        json.dump(tier_assignments, f, indent=2)

    tier_counts = {}
    for t in "ABCD":
        c = sum(1 for v in tier_assignments.values() if v == t)
        tier_counts[t] = c
        print(f"  Tier {t}: {c} features")
    print(f"  Total: {sum(tier_counts.values())}")

    # Load data
    print("\n[1] Loading data and extracting features...")
    trajectories, y, layer_states = load_data()

    # Load holdout mask from Phase 1
    mask = np.load(PHASE1_DIR / "holdout_mask_seed9999.npy")
    train_idx = np.where(~mask)[0]
    hold_idx = np.where(mask)[0]
    y_train = y[train_idx]
    y_holdout = y[hold_idx]

    # Extract train features (train-only PCA)
    print("  Extracting train-400 features...")
    t0 = time.time()
    traj_train = [trajectories[i] for i in train_idx]
    ls_train = layer_states[train_idx]
    features_train, names_train = extract_features(traj_train, ls_train)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Extract holdout features with train PCA (monkey-patch)
    print("  Extracting holdout features (train PCA)...")
    import sklearn.decomposition
    from sklearn.decomposition import PCA as SkPCA

    traj_train_concat = np.concatenate(traj_train, axis=0)
    n_components = min(45, traj_train_concat.shape[1], traj_train_concat.shape[0])
    train_pca = SkPCA(n_components=n_components, svd_solver="full")
    train_pca.fit(traj_train_concat)

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
    original_pca = wf_module.PCA
    wf_module.PCA = PreFittedPCA

    try:
        t0 = time.time()
        features_all_trainpca, _ = extract_features(trajectories, layer_states)
        print(f"  Done in {time.time() - t0:.1f}s")
    finally:
        wf_module.PCA = original_pca

    features_holdout = features_all_trainpca[hold_idx]

    # Build tier indices
    idx_A = get_tier_indices(OUT_NAMES, TIER_A)
    idx_B = get_tier_indices(OUT_NAMES, TIER_B)
    idx_C = get_tier_indices(OUT_NAMES, TIER_C)
    idx_D = get_tier_indices(OUT_NAMES, TIER_D)

    # Cumulative tier indices
    idx_AB = idx_A + idx_B
    idx_ABC = idx_A + idx_B + idx_C
    idx_ABCD = list(range(len(OUT_NAMES)))  # all

    tiers_cumulative = [
        ("A", idx_A),
        ("A+B", idx_AB),
        ("A+B+C", idx_ABC),
        ("A+B+C+D", idx_ABCD),
    ]
    tiers_isolated = [
        ("A only", idx_A),
        ("B only", idx_B),
        ("C only", idx_C),
        ("D only", idx_D),
    ]

    # ---- CV-train-400 for all tier combos ----
    print("\n[2] Tier selection via nested CV on train-400...")
    cv_results = {}

    print("\n  Cumulative tiers:")
    for label, indices in tiers_cumulative:
        auroc = run_cv_subset(features_train, y_train, indices, label=label)
        cv_results[f"cv_{label}"] = auroc

    print("\n  Isolated tiers:")
    for label, indices in tiers_isolated:
        auroc = run_cv_subset(features_train, y_train, indices, label=label)
        cv_results[f"cv_{label}"] = auroc

    # Lock best surviving tier
    full_auroc = cv_results["cv_A+B+C+D"]
    best_tier = None
    for label in ["A", "A+B", "A+B+C", "A+B+C+D"]:
        tier_auroc = cv_results[f"cv_{label}"]
        if tier_auroc is not None and full_auroc is not None:
            if full_auroc - tier_auroc <= 0.01:
                best_tier = label
                break

    if best_tier is None:
        best_tier = "A+B+C+D"

    print(f"\n  Best surviving tier (within 0.01 of full): {best_tier}")
    print(f"  Full AUROC: {full_auroc:.6f}, {best_tier} AUROC: {cv_results[f'cv_{best_tier}']:.6f}")

    with open(PHASE2_DIR / "best_surviving_tier.txt", "w") as f:
        f.write(f"Best surviving tier: {best_tier}\n")
        f.write(f"Rationale: Smallest cumulative tier within 0.01 AUROC of full set on train-400 CV.\n")
        f.write(f"Full (A+B+C+D) CV-train-400 AUROC: {full_auroc:.6f}\n")
        f.write(f"{best_tier} CV-train-400 AUROC: {cv_results[f'cv_{best_tier}']:.6f}\n")
        f.write(f"Gap: {full_auroc - cv_results[f'cv_{best_tier}']:.6f}\n")

    # ---- Single holdout touch for reporting ----
    print("\n[3] Holdout evaluation (single touch, for reporting only)...")
    holdout_results = {}
    for label, indices in tiers_cumulative:
        auroc = holdout_eval(features_train, y_train, features_holdout, y_holdout,
                            indices, label=label)
        holdout_results[f"holdout_{label}"] = auroc

    # ---- Produce table ----
    print("\n" + "=" * 60)
    print("Ablation Table")
    print("=" * 60)
    print(f"\n  {'Tier':<12} {'N_feat':>7} {'CV-train-400':>13} {'Holdout':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*13} {'-'*10}")

    for label, indices in tiers_cumulative:
        n = len(indices)
        cv_val = cv_results.get(f"cv_{label}")
        ho_val = holdout_results.get(f"holdout_{label}")
        cv_str = f"{cv_val:.6f}" if cv_val is not None else "N/A"
        ho_str = f"{ho_val:.6f}" if ho_val is not None else "N/A"
        print(f"  {label:<12} {n:>7} {cv_str:>13} {ho_str:>10}")

    print(f"\n  Isolated tiers:")
    for label, indices in tiers_isolated:
        n = len(indices)
        cv_val = cv_results.get(f"cv_{label}")
        cv_str = f"{cv_val:.6f}" if cv_val is not None else "N/A"
        print(f"  {label:<12} {n:>7} {cv_str:>13}")

    # Save results
    with open(PHASE2_DIR / "ablation_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    with open(PHASE2_DIR / "ablation_holdout_results.json", "w") as f:
        json.dump(holdout_results, f, indent=2)

    # Save table
    table_lines = []
    table_lines.append(f"{'Tier':<12} {'N_feat':>7} {'CV-train-400':>13} {'Holdout':>10}")
    table_lines.append(f"{'-'*12} {'-'*7} {'-'*13} {'-'*10}")
    for label, indices in tiers_cumulative:
        n = len(indices)
        cv_val = cv_results.get(f"cv_{label}")
        ho_val = holdout_results.get(f"holdout_{label}")
        cv_str = f"{cv_val:.6f}" if cv_val is not None else "N/A"
        ho_str = f"{ho_val:.6f}" if ho_val is not None else "N/A"
        table_lines.append(f"{label:<12} {n:>7} {cv_str:>13} {ho_str:>10}")
    table_lines.append("")
    table_lines.append("Isolated tiers:")
    for label, indices in tiers_isolated:
        n = len(indices)
        cv_val = cv_results.get(f"cv_{label}")
        cv_str = f"{cv_val:.6f}" if cv_val is not None else "N/A"
        table_lines.append(f"{label:<12} {n:>7} {cv_str:>13}")
    table_lines.append("")
    table_lines.append(f"Best surviving tier: {best_tier}")
    table_lines.append(f"(Locked before holdout consultation)")

    with open(PHASE2_DIR / "ablation_table.txt", "w") as f:
        f.write("\n".join(table_lines) + "\n")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tier_labels = ["A", "A+B", "A+B+C", "A+B+C+D"]
        cv_vals = [cv_results.get(f"cv_{l}") for l in tier_labels]
        ho_vals = [holdout_results.get(f"holdout_{l}") for l in tier_labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(tier_labels))
        if all(v is not None for v in cv_vals):
            ax.plot(x, cv_vals, "bo-", label="CV-train-400", markersize=8)
        if all(v is not None for v in ho_vals):
            ax.plot(x, ho_vals, "rs-", label="Holdout-100", markersize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tier_labels)
        ax.set_xlabel("Cumulative Feature Tier")
        ax.set_ylabel("AUROC")
        ax.set_title("Feature Ablation by Tier")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.6, 1.0)
        plt.tight_layout()
        plt.savefig(PHASE2_DIR / "ablation_plot.png", dpi=150)
        print(f"\n  Plot saved to {PHASE2_DIR / 'ablation_plot.png'}")
    except ImportError:
        print("\n  matplotlib not available, skipping plot")

    print(f"\n  All artifacts saved to {PHASE2_DIR}/")


if __name__ == "__main__":
    main()
