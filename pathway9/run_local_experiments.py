"""Pathway 9: Diagnostic Decomposition — CPU-local experiments.

Runs Exp 1 (length deconfound), Exp 3a (Gaussian null), Exp 3c (count control),
and Exp 4 (XGBoost + O-information) using cached features and trajectories.

All results saved to pathway9/results/*.json with .done markers.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# ───────────────────── DATA LOADING ─────────────────────


def load_data():
    """Load ABC-44 features, labels, token counts, and reconstruct SSS split."""
    # Labels
    new_labels = np.load(REPO / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy")
    old_labels = np.load(REPO / "pathway2/track_a/phase0/baseline_correct.npy")

    # SSS split (use OLD labels for split, NEW labels for training)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_sss, hold_sss = next(sss.split(np.zeros(500), old_labels.astype(int)))
    train_sort = np.argsort(train_sss)
    hold_sort = np.argsort(hold_sss)
    train_idx = np.sort(train_sss)
    holdout_idx = np.sort(hold_sss)

    # ABC-44 features (first 44 columns are ABC tier)
    with open(REPO / "pathway1/phase2/tier_assignments.json") as f:
        tiers = json.load(f)
    with open(REPO / "pathway6_rebuild/phase2_completion/feature_names_v2.json") as f:
        all_names = json.load(f)
    abc_cols = [i for i, name in enumerate(all_names) if tiers.get(name) in ("A", "B", "C")]
    abc_names = [all_names[i] for i in abc_cols]

    X_train = np.load(REPO / "pathway1/phase1/features_train400.npy")[train_sort][:, abc_cols]
    X_holdout = np.load(REPO / "pathway1/phase1/features_holdout100.npy")[hold_sort][:, abc_cols]
    y_train = new_labels[train_idx]
    y_holdout = new_labels[holdout_idx]

    # Token counts from trajectory shapes
    with open(REPO / "data/experiment1_v2/trajectory_meta.json") as f:
        meta = json.load(f)
    all_token_counts = np.array([s[0] for s in meta["trajectory_shapes"]])
    token_counts_train = all_token_counts[train_idx]
    token_counts_holdout = all_token_counts[holdout_idx]

    print(f"  Data loaded: train={X_train.shape}, holdout={X_holdout.shape}")
    print(f"  Train correct: {y_train.sum()}/{len(y_train)}, Holdout correct: {y_holdout.sum()}/{len(y_holdout)}")
    print(f"  Token counts: train mean={token_counts_train.mean():.0f}, holdout mean={token_counts_holdout.mean():.0f}")

    return dict(
        X_train=X_train, X_holdout=X_holdout,
        y_train=y_train, y_holdout=y_holdout,
        train_idx=train_idx, holdout_idx=holdout_idx,
        abc_names=abc_names, abc_cols=abc_cols,
        all_names=all_names,
        token_counts_train=token_counts_train,
        token_counts_holdout=token_counts_holdout,
        all_token_counts=all_token_counts,
        new_labels=new_labels,
    )


def fit_lr(X_train, y_train, X_holdout, y_holdout, n_cv=50):
    """Train LR, return holdout AUROC + CV AUROC."""
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    pipe.fit(X_train, y_train)
    probs_ho = pipe.predict_proba(X_holdout)[:, 1]
    auroc_ho = roc_auc_score(y_holdout, probs_ho)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=min(n_cv, 50), scoring="roc_auc")
    return auroc_ho, cv_scores.mean(), cv_scores.std(), probs_ho


# ───────────────────── EXPERIMENT 1: LENGTH DECONFOUND ─────────────────────


def exp1_length_deconfound(d):
    """Test whether ABC-44 AUROC is inflated by output length."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Length Deconfound on ABC-44 MATH-500")
    print("=" * 60)

    X_tr, X_ho = d["X_train"], d["X_holdout"]
    y_tr, y_ho = d["y_train"], d["y_holdout"]
    tc_tr, tc_ho = d["token_counts_train"], d["token_counts_holdout"]
    names = d["abc_names"]

    # 1a. Length-only AUROC
    # Shorter completions might be more correct (got answer quickly) or less correct (ran out of tokens)
    auroc_length_pos = roc_auc_score(y_ho, tc_ho)
    auroc_length_neg = roc_auc_score(y_ho, -tc_ho)
    length_auroc = max(auroc_length_pos, auroc_length_neg)
    length_sign = "shorter=correct" if auroc_length_neg > auroc_length_pos else "longer=correct"
    print(f"\n  Length-only AUROC: {length_auroc:.3f} ({length_sign})")

    # Also on train
    auroc_len_tr_pos = roc_auc_score(y_tr, tc_tr)
    auroc_len_tr_neg = roc_auc_score(y_tr, -tc_tr)
    print(f"  Length-only AUROC (train): {max(auroc_len_tr_pos, auroc_len_tr_neg):.3f}")

    # 1b. Per-feature correlations with length
    print(f"\n  Per-feature correlation with token count (|r| > 0.1):")
    corrs = []
    high_corr_features = []
    for i, name in enumerate(names):
        all_feats = np.concatenate([X_tr[:, i], X_ho[:, i]])
        all_tc = np.concatenate([tc_tr, tc_ho])
        r = np.corrcoef(all_feats, all_tc)[0, 1]
        corrs.append(r)
        if abs(r) > 0.1:
            high_corr_features.append((name, r))
            marker = " ***" if abs(r) > 0.3 else ""
            print(f"    {name:40s}  r={r:+.3f}{marker}")

    n_above_03 = sum(1 for _, r in high_corr_features if abs(r) > 0.3)
    n_above_01 = len(high_corr_features)
    print(f"  Features with |r|>0.3: {n_above_03}/{len(names)}")
    print(f"  Features with |r|>0.1: {n_above_01}/{len(names)}")

    # 1c. Baseline AUROC (undeconfounded)
    auroc_base, cv_base, cv_std, probs_base = fit_lr(X_tr, y_tr, X_ho, y_ho)
    print(f"\n  Baseline AUROC (ABC-44): holdout={auroc_base:.3f}, CV={cv_base:.3f}±{cv_std:.3f}")

    # 1d. Residualize features against length
    X_tr_deconf = np.zeros_like(X_tr)
    X_ho_deconf = np.zeros_like(X_ho)
    for i in range(X_tr.shape[1]):
        lr = LinearRegression().fit(tc_tr.reshape(-1, 1), X_tr[:, i])
        X_tr_deconf[:, i] = X_tr[:, i] - lr.predict(tc_tr.reshape(-1, 1))
        X_ho_deconf[:, i] = X_ho[:, i] - lr.predict(tc_ho.reshape(-1, 1))

    auroc_deconf, cv_deconf, cv_std_deconf, probs_deconf = fit_lr(X_tr_deconf, y_tr, X_ho_deconf, y_ho)
    drop = auroc_base - auroc_deconf
    print(f"  Deconfounded AUROC: holdout={auroc_deconf:.3f}, CV={cv_deconf:.3f}±{cv_std_deconf:.3f}")
    print(f"  Drop from deconfounding: {drop:+.3f}")

    # 1e. Length as additional feature (does adding length help?)
    X_tr_plus = np.column_stack([X_tr, tc_tr])
    X_ho_plus = np.column_stack([X_ho, tc_ho])
    auroc_plus, cv_plus, cv_std_plus, _ = fit_lr(X_tr_plus, y_tr, X_ho_plus, y_ho)
    print(f"  ABC-44 + length feature: holdout={auroc_plus:.3f}, CV={cv_plus:.3f}±{cv_std_plus:.3f}")

    # 1f. Length-only classifier
    auroc_length_lr, cv_length_lr, cv_std_length_lr, _ = fit_lr(
        tc_tr.reshape(-1, 1), y_tr, tc_ho.reshape(-1, 1), y_ho
    )
    print(f"  Length-only (LR): holdout={auroc_length_lr:.3f}, CV={cv_length_lr:.3f}±{cv_std_length_lr:.3f}")

    # Decision
    if abs(drop) < 0.02:
        verdict = "CLEAN — length confound is minimal (<0.02 drop)"
    elif abs(drop) < 0.05:
        verdict = f"MILD — length confound contributes ~{abs(drop):.3f} to AUROC"
    else:
        verdict = f"SIGNIFICANT — length confound inflates AUROC by {abs(drop):.3f}"
    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "exp1_length_deconfound",
        "length_only_auroc": float(length_auroc),
        "length_sign": length_sign,
        "length_only_lr_auroc": float(auroc_length_lr),
        "baseline_auroc": float(auroc_base),
        "baseline_cv": float(cv_base),
        "deconfounded_auroc": float(auroc_deconf),
        "deconfounded_cv": float(cv_deconf),
        "drop_from_deconfounding": float(drop),
        "abc44_plus_length_auroc": float(auroc_plus),
        "n_features_corr_above_03": n_above_03,
        "n_features_corr_above_01": n_above_01,
        "feature_length_correlations": {name: float(r) for name, r in zip(names, corrs)},
        "high_corr_features": [(name, float(r)) for name, r in high_corr_features],
        "verdict": verdict,
    }
    (RESULTS / "exp1_length_deconfound.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp1.done").touch()
    return results


# ───────────────────── EXPERIMENT 3a: GAUSSIAN NULL ─────────────────────


def exp3a_gaussian_null(d):
    """Test whether PH features measure topology or just covariance structure."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3a: Gaussian-Matched Null Test")
    print("=" * 60)

    from ripser import ripser
    from sklearn.decomposition import PCA

    X_tr, X_ho = d["X_train"], d["X_holdout"]
    y_tr, y_ho = d["y_train"], d["y_holdout"]
    train_idx, holdout_idx = d["train_idx"], d["holdout_idx"]

    # Load raw trajectories
    print("  Loading trajectories...")
    traj_file = np.load(REPO / "data/experiment1_v2/trajectories.npz", allow_pickle=True)

    def compute_6_ph_features(cloud, subsample=100, pca_components=45):
        """Compute the 6 PH features matching ABC-tier extraction.

        Uses the same pipeline: PCA → subsample → ripser → 6 features.
        """
        if len(cloud) < 5:
            return np.zeros(6)

        # PCA (match pathway1 pipeline: 45 components)
        n_comp = min(pca_components, cloud.shape[0] - 1, cloud.shape[1])
        pca = PCA(n_components=n_comp)
        cloud_pca = pca.fit_transform(cloud)

        # Subsample
        if len(cloud_pca) > subsample:
            idx = np.linspace(0, len(cloud_pca) - 1, subsample, dtype=int)
            cloud_pca = cloud_pca[idx]

        # Ripser
        result = ripser(cloud_pca, maxdim=1)
        dgms = result["dgms"]

        feats = np.zeros(6)
        # H0
        h0 = dgms[0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        if len(h0_finite) > 0:
            lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
            feats[0] = lifetimes.max()  # H0_max_lifetime
            feats[1] = len(h0_finite)   # H0_n_features (count)
            # Entropy
            lt_norm = lifetimes / lifetimes.sum() if lifetimes.sum() > 0 else lifetimes
            lt_norm = lt_norm[lt_norm > 0]
            feats[2] = -np.sum(lt_norm * np.log(lt_norm)) if len(lt_norm) > 0 else 0  # H0_entropy
        # H1
        if len(dgms) > 1 and len(dgms[1]) > 0:
            h1 = dgms[1]
            h1_finite = h1[np.isfinite(h1[:, 1])]
            if len(h1_finite) > 0:
                lifetimes_h1 = h1_finite[:, 1] - h1_finite[:, 0]
                lt_norm_h1 = lifetimes_h1 / lifetimes_h1.sum() if lifetimes_h1.sum() > 0 else lifetimes_h1
                lt_norm_h1 = lt_norm_h1[lt_norm_h1 > 0]
                feats[3] = -np.sum(lt_norm_h1 * np.log(lt_norm_h1)) if len(lt_norm_h1) > 0 else 0  # H1_persistence_entropy
                feats[4] = len(h1_finite)   # H1_n_features
                feats[5] = lifetimes_h1.max()  # H1_max_lifetime
        return feats

    def gaussian_null_features(cloud, n_nulls=5):
        """Generate Gaussian-matched nulls and compute PH features."""
        n, d_dim = cloud.shape
        mean = cloud.mean(axis=0)
        # Use diagonal covariance for stability (full cov singular with n < d)
        var = cloud.var(axis=0) + 1e-8
        null_feats = []
        for _ in range(n_nulls):
            synthetic = np.random.randn(n, d_dim) * np.sqrt(var) + mean
            null_feats.append(compute_6_ph_features(synthetic))
        return np.mean(null_feats, axis=0)

    # Compute real and null PH features for all 500 problems
    n_problems = 500
    real_ph = np.zeros((n_problems, 6))
    null_ph = np.zeros((n_problems, 6))
    diff_ph = np.zeros((n_problems, 6))

    np.random.seed(42)
    print("  Computing PH features for real and Gaussian-null clouds...")
    t0 = time.time()
    for i in range(n_problems):
        cloud = traj_file[f"traj_{i}"].astype(np.float64)
        real_ph[i] = compute_6_ph_features(cloud)
        null_ph[i] = gaussian_null_features(cloud, n_nulls=5)
        diff_ph[i] = real_ph[i] - null_ph[i]
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_problems} ({elapsed:.0f}s)")

    print(f"  Done in {time.time() - t0:.0f}s")

    # Check: are real features different from null features?
    ph_names = ["H0_max_lifetime", "H0_n_features", "H0_entropy",
                "H1_pers_entropy", "H1_n_features", "H1_max_lifetime"]
    print("\n  Real vs Null feature comparison (mean ± std):")
    for j, name in enumerate(ph_names):
        real_m, real_s = real_ph[:, j].mean(), real_ph[:, j].std()
        null_m, null_s = null_ph[:, j].mean(), null_ph[:, j].std()
        diff_m = diff_ph[:, j].mean()
        print(f"    {name:25s}  real={real_m:8.2f}±{real_s:.2f}  null={null_m:8.2f}±{null_s:.2f}  diff={diff_m:+.2f}")

    # Train classifiers on: (a) real PH, (b) null PH, (c) diff PH
    y_all = d["new_labels"]
    y_tr_ph = y_all[train_idx]
    y_ho_ph = y_all[holdout_idx]

    configs = {
        "real_ph": real_ph,
        "null_ph": null_ph,
        "diff_ph": diff_ph,
    }
    ph_results = {}
    for name, feats in configs.items():
        f_tr = feats[train_idx]
        f_ho = feats[holdout_idx]
        auroc_ho, cv_mean, cv_std, _ = fit_lr(f_tr, y_tr_ph, f_ho, y_ho_ph)
        ph_results[name] = {"auroc_holdout": float(auroc_ho), "auroc_cv": float(cv_mean), "cv_std": float(cv_std)}
        print(f"\n  {name:12s}: holdout={auroc_ho:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")

    # Decision
    real_auroc = ph_results["real_ph"]["auroc_holdout"]
    null_auroc = ph_results["null_ph"]["auroc_holdout"]
    diff_auroc = ph_results["diff_ph"]["auroc_holdout"]

    if null_auroc >= real_auroc - 0.02:
        verdict = "PH measures COVARIANCE, not topology — Gaussian null matches real PH"
    elif diff_auroc > real_auroc + 0.01:
        verdict = "Topology-specific signal exists — diff features beat raw features"
    else:
        verdict = f"Mixed — real={real_auroc:.3f}, null={null_auroc:.3f}, diff={diff_auroc:.3f}"
    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "exp3a_gaussian_null",
        "n_problems": n_problems,
        "n_nulls_per_problem": 5,
        "classifiers": ph_results,
        "feature_comparison": {
            name: {
                "real_mean": float(real_ph[:, j].mean()),
                "real_std": float(real_ph[:, j].std()),
                "null_mean": float(null_ph[:, j].mean()),
                "null_std": float(null_ph[:, j].std()),
                "diff_mean": float(diff_ph[:, j].mean()),
            }
            for j, name in enumerate(ph_names)
        },
        "verdict": verdict,
    }
    (RESULTS / "exp3a_gaussian_null.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp3a.done").touch()
    return results


# ───────────────────── EXPERIMENT 3c: COUNT CONTROL ─────────────────────


def exp3c_count_control(d):
    """Test whether removing count-based features changes AUROC.

    ABC-44 already excludes H0_n_features and H1_n_features (they're tier D).
    But check: what features in ABC-44 are most like counts?
    Also test with all 78 features including counts.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3c: Feature-Count Control")
    print("=" * 60)

    names = d["abc_names"]
    all_names = d["all_names"]
    y_tr, y_ho = d["y_train"], d["y_holdout"]
    train_idx, holdout_idx = d["train_idx"], d["holdout_idx"]

    # Check: are H0_n_features, H1_n_features in ABC-44?
    count_in_abc = [n for n in names if "n_features" in n.lower()]
    print(f"  Count features in ABC-44: {count_in_abc if count_in_abc else 'NONE'}")

    # Find count features in full 78
    with open(REPO / "pathway1/phase2/tier_assignments.json") as f:
        tiers = json.load(f)
    count_cols_78 = [i for i, n in enumerate(all_names) if "n_features" in n.lower()]
    count_names_78 = [all_names[i] for i in count_cols_78]
    count_tiers = [tiers.get(n, "?") for n in count_names_78]
    print(f"  Count features in full 78: {list(zip(count_names_78, count_tiers))}")

    # Load full 78-feature set
    old_labels = np.load(REPO / "pathway2/track_a/phase0/baseline_correct.npy")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_sss, hold_sss = next(sss.split(np.zeros(500), old_labels.astype(int)))
    train_sort = np.argsort(train_sss)
    hold_sort = np.argsort(hold_sss)

    X78_tr = np.load(REPO / "pathway1/phase1/features_train400.npy")[train_sort]
    X78_ho = np.load(REPO / "pathway1/phase1/features_holdout100.npy")[hold_sort]

    # Full 78 features
    auroc_78, cv_78, cv_std_78, _ = fit_lr(X78_tr, y_tr, X78_ho, y_ho)
    print(f"\n  Full 78 features: holdout={auroc_78:.3f}, CV={cv_78:.3f}±{cv_std_78:.3f}")

    # Full 78 minus count features
    non_count_cols = [i for i in range(78) if i not in count_cols_78]
    auroc_no_counts, cv_no_counts, std_no_counts, _ = fit_lr(
        X78_tr[:, non_count_cols], y_tr, X78_ho[:, non_count_cols], y_ho
    )
    print(f"  78 minus counts ({len(non_count_cols)} feat): holdout={auroc_no_counts:.3f}, CV={cv_no_counts:.3f}±{std_no_counts:.3f}")

    # Count features only
    if count_cols_78:
        auroc_counts_only, cv_counts_only, std_counts, _ = fit_lr(
            X78_tr[:, count_cols_78], y_tr, X78_ho[:, count_cols_78], y_ho
        )
        print(f"  Count features only ({len(count_cols_78)} feat): holdout={auroc_counts_only:.3f}, CV={cv_counts_only:.3f}±{std_counts:.3f}")
    else:
        auroc_counts_only = None

    # ABC-44 (baseline)
    auroc_abc, cv_abc, std_abc, _ = fit_lr(d["X_train"], y_tr, d["X_holdout"], y_ho)
    print(f"  ABC-44: holdout={auroc_abc:.3f}, CV={cv_abc:.3f}±{std_abc:.3f}")

    # Verdict
    if not count_in_abc:
        verdict = "ABC-44 already EXCLUDES count features — the 0.796 AUROC is count-free"
    elif auroc_no_counts > auroc_78 - 0.02:
        verdict = "Count features don't matter — removal doesn't change AUROC"
    else:
        verdict = f"Count features contribute — removing them drops AUROC by {auroc_78 - auroc_no_counts:.3f}"

    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "exp3c_count_control",
        "count_features_in_abc44": count_in_abc,
        "count_features_in_78": list(zip(count_names_78, count_tiers)),
        "full_78_auroc": float(auroc_78),
        "full_78_cv": float(cv_78),
        "minus_counts_auroc": float(auroc_no_counts),
        "minus_counts_cv": float(cv_no_counts),
        "counts_only_auroc": float(auroc_counts_only) if auroc_counts_only is not None else None,
        "abc44_auroc": float(auroc_abc),
        "abc44_cv": float(cv_abc),
        "verdict": verdict,
    }
    (RESULTS / "exp3c_count_control.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp3c.done").touch()
    return results


# ───────────────────── EXPERIMENT 4: XGBOOST + O-INFO ─────────────────────


def exp4_xgboost_oinfo(d):
    """Compare XGBoost vs LR and compute O-information."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: XGBoost Comparison + O-Information")
    print("=" * 60)

    from xgboost import XGBClassifier

    X_tr, X_ho = d["X_train"], d["X_holdout"]
    y_tr, y_ho = d["y_train"], d["y_holdout"]

    # LR baseline (with StandardScaler)
    auroc_lr, cv_lr, std_lr, probs_lr = fit_lr(X_tr, y_tr, X_ho, y_ho)
    print(f"\n  LR: holdout={auroc_lr:.3f}, CV={cv_lr:.3f}±{std_lr:.3f}")

    # XGBoost with various configs
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    xgb_configs = {
        "xgb_default": XGBClassifier(
            n_estimators=100, max_depth=3, random_state=42,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            verbosity=0,
        ),
        "xgb_shallow": XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=42, scale_pos_weight=scale_pos,
            eval_metric="logloss", verbosity=0,
        ),
        "xgb_regularized": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, scale_pos_weight=scale_pos,
            eval_metric="logloss", verbosity=0,
        ),
    }

    xgb_results = {}
    for name, clf in xgb_configs.items():
        # Scale features for XGBoost too (for fair comparison)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_ho_s = scaler.transform(X_ho)

        clf.fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_ho_s)[:, 1]
        auroc_ho = roc_auc_score(y_ho, probs)

        cv_scores = cross_val_score(
            Pipeline([("scaler", StandardScaler()), ("xgb", clf.__class__(**clf.get_params()))]),
            X_tr, y_tr, cv=5, scoring="roc_auc"
        )
        xgb_results[name] = {
            "auroc_holdout": float(auroc_ho),
            "auroc_cv": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        }
        print(f"  {name:20s}: holdout={auroc_ho:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    # Best XGBoost
    best_xgb_name = max(xgb_results, key=lambda k: xgb_results[k]["auroc_cv"])
    best_xgb = xgb_results[best_xgb_name]
    lift = best_xgb["auroc_cv"] - cv_lr
    print(f"\n  Best XGBoost ({best_xgb_name}): CV={best_xgb['auroc_cv']:.3f}")
    print(f"  LR CV: {cv_lr:.3f}")
    print(f"  XGBoost lift: {lift:+.3f}")

    # Feature importance from best XGBoost
    best_clf = xgb_configs[best_xgb_name]
    scaler = StandardScaler()
    best_clf.fit(scaler.fit_transform(X_tr), y_tr)
    importances = best_clf.feature_importances_
    top_10 = np.argsort(importances)[-10:][::-1]
    print(f"\n  Top 10 XGBoost features:")
    for idx in top_10:
        print(f"    {d['abc_names'][idx]:40s}  importance={importances[idx]:.3f}")

    # O-Information
    o_info_result = None
    try:
        from hoi.metrics import Oinfo as OinfoMetric
        print("\n  Computing O-information...")
        # hoi expects (n_samples, n_features) with the last column as the target
        X_with_y = np.column_stack([StandardScaler().fit_transform(X_tr), y_tr.astype(float)])
        # Discretize for O-info (continuous features need binning)
        n_bins = 5
        X_disc = np.zeros_like(X_with_y, dtype=int)
        for j in range(X_with_y.shape[1]):
            X_disc[:, j] = np.digitize(X_with_y[:, j], np.percentile(X_with_y[:, j], np.linspace(0, 100, n_bins + 1)[1:-1]))

        oi = OinfoMetric(X_disc.T)  # hoi expects (n_variables, n_samples)
        oi_values = oi.fit(minsize=3, maxsize=5)
        mean_oi = float(np.nanmean(oi_values))
        print(f"  Mean O-information (order 3-5): {mean_oi:.4f}")
        if mean_oi > 0:
            print(f"  → REDUNDANCY-dominated (positive O-info)")
        else:
            print(f"  → SYNERGY-dominated (negative O-info)")
        o_info_result = {
            "mean_o_information": mean_oi,
            "interpretation": "redundancy" if mean_oi > 0 else "synergy",
        }
    except Exception as e:
        print(f"  O-information computation failed: {e}")
        o_info_result = {"error": str(e)}

    # Decision
    if lift > 0.03:
        verdict = f"XGBoost significantly beats LR (+{lift:.3f} CV) — signal is NONLINEAR"
    elif lift > 0.01:
        verdict = f"XGBoost mildly beats LR (+{lift:.3f} CV) — minor nonlinear component"
    elif lift > -0.01:
        verdict = f"XGBoost ≈ LR (diff={lift:+.3f} CV) — LR is appropriate"
    else:
        verdict = f"XGBoost WORSE than LR ({lift:+.3f} CV) — features are linear, XGBoost overfits"
    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "exp4_xgboost_oinfo",
        "lr_auroc_holdout": float(auroc_lr),
        "lr_auroc_cv": float(cv_lr),
        "xgboost_results": xgb_results,
        "best_xgboost": best_xgb_name,
        "xgboost_lift_cv": float(lift),
        "top_10_features": [(d["abc_names"][i], float(importances[i])) for i in top_10],
        "o_information": o_info_result,
        "verdict": verdict,
    }
    (RESULTS / "exp4_xgboost_oinfo.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp4.done").touch()
    return results


# ───────────────────── MAIN ─────────────────────


def main():
    print("=" * 60)
    print("PATHWAY 9: Diagnostic Decomposition — Local Experiments")
    print("=" * 60)

    # Validate baseline reproduction first
    d = load_data()

    auroc_base, cv_base, _, _ = fit_lr(d["X_train"], d["y_train"], d["X_holdout"], d["y_holdout"])
    print(f"\n  Baseline check: holdout={auroc_base:.3f} (expected ~0.796)")
    if abs(auroc_base - 0.796) > 0.01:
        print(f"  WARNING: Baseline drifted by {auroc_base - 0.796:+.3f}!")

    # Run experiments
    results = {}

    if not (RESULTS / "exp1.done").exists():
        results["exp1"] = exp1_length_deconfound(d)
    else:
        print("\n  Exp 1 already done, skipping.")
        results["exp1"] = json.loads((RESULTS / "exp1_length_deconfound.json").read_text())

    if not (RESULTS / "exp3a.done").exists():
        results["exp3a"] = exp3a_gaussian_null(d)
    else:
        print("\n  Exp 3a already done, skipping.")
        results["exp3a"] = json.loads((RESULTS / "exp3a_gaussian_null.json").read_text())

    if not (RESULTS / "exp3c.done").exists():
        results["exp3c"] = exp3c_count_control(d)
    else:
        print("\n  Exp 3c already done, skipping.")
        results["exp3c"] = json.loads((RESULTS / "exp3c_count_control.json").read_text())

    if not (RESULTS / "exp4.done").exists():
        results["exp4"] = exp4_xgboost_oinfo(d)
    else:
        print("\n  Exp 4 already done, skipping.")
        results["exp4"] = json.loads((RESULTS / "exp4_xgboost_oinfo.json").read_text())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {r.get('verdict', 'N/A')}")

    print("\nAll local experiments complete. Results in pathway9/results/")


if __name__ == "__main__":
    main()
