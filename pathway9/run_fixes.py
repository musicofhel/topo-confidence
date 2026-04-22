"""Pathway 9 audit fixes — reruns affected experiments per AUDIT.md.

Fixes:
  D4 — Exp 4 rerun with cv=5 for both LR and XGBoost (apples-to-apples).
  D1+D2 — Exp 3a rerun with full-covariance null and without H0_n_features.
  D6 — Exp 1 cross-check with raw tokenizer-encoded completion length.
  D5 — Exp 4 O-information rerun with correct dtype.

Outputs: pathway9/results/*_v2.json + *_v2.done markers.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Shared loaders (copied minimally from run_local_experiments.py)
# ---------------------------------------------------------------------------


def load_data():
    new_labels = np.load(REPO / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy")
    old_labels = np.load(REPO / "pathway2/track_a/phase0/baseline_correct.npy")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_sss, hold_sss = next(sss.split(np.zeros(500), old_labels.astype(int)))
    train_sort = np.argsort(train_sss)
    hold_sort = np.argsort(hold_sss)
    train_idx = np.sort(train_sss)
    holdout_idx = np.sort(hold_sss)

    with open(REPO / "pathway1/phase2/tier_assignments.json") as f:
        tiers = json.load(f)
    with open(REPO / "pathway6_rebuild/phase2_completion/feature_names_v2.json") as f:
        all_names = json.load(f)
    abc_cols = [i for i, n in enumerate(all_names) if tiers.get(n) in ("A", "B", "C")]
    abc_names = [all_names[i] for i in abc_cols]

    X_train = np.load(REPO / "pathway1/phase1/features_train400.npy")[train_sort][:, abc_cols]
    X_holdout = np.load(REPO / "pathway1/phase1/features_holdout100.npy")[hold_sort][:, abc_cols]
    y_train = new_labels[train_idx]
    y_holdout = new_labels[holdout_idx]

    with open(REPO / "data/experiment1_v2/trajectory_meta.json") as f:
        meta = json.load(f)
    all_traj_rows = np.array([s[0] for s in meta["trajectory_shapes"]])
    generated_texts = meta["generated_texts"]

    return dict(
        X_train=X_train, X_holdout=X_holdout,
        y_train=y_train, y_holdout=y_holdout,
        train_idx=train_idx, holdout_idx=holdout_idx,
        abc_names=abc_names, new_labels=new_labels,
        all_traj_rows=all_traj_rows,
        generated_texts=generated_texts,
    )


def _run_oinfo_subprocess(X_tr, y_tr, timeout_sec=120):
    """Run O-information in isolated subprocess so JAX OOM doesn't kill parent."""
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    np.save(tmp / "X.npy", X_tr.astype(np.float64))
    np.save(tmp / "y.npy", y_tr.astype(np.float64))
    out_path = tmp / "oinfo_result.json"

    script = f'''
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

tmp = Path("{tmp}")
X = np.load(tmp / "X.npy")
y = np.load(tmp / "y.npy")
try:
    from hoi.metrics import Oinfo
    X_std = StandardScaler().fit_transform(X)
    X_with_y = np.column_stack([X_std, y])
    n_bins = 5
    X_disc = np.zeros_like(X_with_y, dtype=float)
    for j in range(X_with_y.shape[1]):
        bins = np.percentile(X_with_y[:, j], np.linspace(0, 100, n_bins + 1)[1:-1])
        X_disc[:, j] = np.digitize(X_with_y[:, j], bins).astype(float)
    oi = Oinfo(X_disc.T)
    oi_values = oi.fit(minsize=3, maxsize=5)
    mean_oi = float(np.nanmean(oi_values))
    (tmp / "oinfo_result.json").write_text(json.dumps({{
        "mean_o_information": mean_oi,
        "interpretation": "redundancy" if mean_oi > 0 else "synergy"
    }}))
except Exception as e:
    (tmp / "oinfo_result.json").write_text(json.dumps({{
        "error": f"{{type(e).__name__}}: {{e}}"
    }}))
'''
    script_path = tmp / "run_oinfo.py"
    script_path.write_text(script)

    print(f"\n  Attempting O-information (subprocess, timeout {timeout_sec}s)...")
    try:
        proc = subprocess.run(
            ["python3", str(script_path)],
            timeout=timeout_sec, capture_output=True, text=True,
        )
        if out_path.exists():
            result = json.loads(out_path.read_text())
            if "mean_o_information" in result:
                print(f"  Mean O-information: {result['mean_o_information']:.4f} → {result['interpretation']}")
            else:
                print(f"  O-info subprocess failed: {result.get('error', 'unknown')}")
            return result
        else:
            return {"error": f"subprocess exit {proc.returncode}, no output file",
                    "stderr_tail": proc.stderr[-500:] if proc.stderr else ""}
    except subprocess.TimeoutExpired:
        return {"error": f"subprocess timed out after {timeout_sec}s"}
    except Exception as e:
        return {"error": f"subprocess launch failed: {type(e).__name__}: {e}"}


def fit_lr_cv(X_train, y_train, X_holdout, y_holdout, cv=5):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    probs_ho = pipe.predict_proba(X_holdout)[:, 1]
    auroc_ho = roc_auc_score(y_holdout, probs_ho)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    return auroc_ho, float(cv_scores.mean()), float(cv_scores.std()), probs_ho


# ---------------------------------------------------------------------------
# D6 — Exp 1 raw tokenizer length cross-check
# ---------------------------------------------------------------------------


def fix_d6_tokenizer_length(d):
    print("\n" + "=" * 60)
    print("D6: Raw tokenizer-length cross-check (Exp 1)")
    print("=" * 60)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    texts = d["generated_texts"]
    raw_token_counts = np.array([len(tok.encode(t, add_special_tokens=False)) for t in texts])
    traj_rows = d["all_traj_rows"]

    r_traj_vs_raw = np.corrcoef(traj_rows, raw_token_counts)[0, 1]
    print(f"  trajectory_rows vs raw-tokenizer count:  r={r_traj_vs_raw:+.3f}")
    print(f"  trajectory_rows: mean={traj_rows.mean():.0f}  median={np.median(traj_rows):.0f}  max={traj_rows.max()}")
    print(f"  raw token count: mean={raw_token_counts.mean():.0f}  median={np.median(raw_token_counts):.0f}  max={raw_token_counts.max()}")

    tc_tr = raw_token_counts[d["train_idx"]]
    tc_ho = raw_token_counts[d["holdout_idx"]]

    auroc_len_pos = roc_auc_score(d["y_holdout"], tc_ho)
    auroc_len_neg = roc_auc_score(d["y_holdout"], -tc_ho)
    length_auroc = max(auroc_len_pos, auroc_len_neg)
    length_sign = "shorter=correct" if auroc_len_neg > auroc_len_pos else "longer=correct"
    print(f"  Raw-length-only AUROC: {length_auroc:.3f} ({length_sign})")

    X_tr, X_ho = d["X_train"], d["X_holdout"]
    X_tr_deconf = np.zeros_like(X_tr)
    X_ho_deconf = np.zeros_like(X_ho)
    for i in range(X_tr.shape[1]):
        lr = LinearRegression().fit(tc_tr.reshape(-1, 1), X_tr[:, i])
        X_tr_deconf[:, i] = X_tr[:, i] - lr.predict(tc_tr.reshape(-1, 1))
        X_ho_deconf[:, i] = X_ho[:, i] - lr.predict(tc_ho.reshape(-1, 1))

    auroc_base, cv_base, _, _ = fit_lr_cv(X_tr, d["y_train"], X_ho, d["y_holdout"], cv=5)
    auroc_deconf, cv_deconf, _, _ = fit_lr_cv(X_tr_deconf, d["y_train"], X_ho_deconf, d["y_holdout"], cv=5)
    drop = auroc_base - auroc_deconf
    print(f"  Baseline AUROC: holdout={auroc_base:.3f}, CV5={cv_base:.3f}")
    print(f"  Deconfounded (raw length): holdout={auroc_deconf:.3f}, CV5={cv_deconf:.3f}")
    print(f"  Drop from deconfounding: {drop:+.3f}")

    if abs(drop) < 0.02:
        verdict = "CLEAN — raw-length confound is minimal (<0.02 drop). Consistent with Exp 1."
    elif abs(drop) < 0.05:
        verdict = f"MILD — raw-length confound contributes ~{abs(drop):.3f}."
    else:
        verdict = f"SIGNIFICANT — raw-length inflates AUROC by {abs(drop):.3f}. Exp 1 underestimated the confound."

    print(f"  VERDICT: {verdict}")

    results = {
        "experiment": "exp1_raw_length_cross_check",
        "traj_rows_vs_raw_tokenizer_r": float(r_traj_vs_raw),
        "traj_rows_stats": {
            "mean": float(traj_rows.mean()),
            "median": float(np.median(traj_rows)),
            "max": int(traj_rows.max()),
        },
        "raw_token_count_stats": {
            "mean": float(raw_token_counts.mean()),
            "median": float(np.median(raw_token_counts)),
            "max": int(raw_token_counts.max()),
        },
        "raw_length_only_auroc": float(length_auroc),
        "raw_length_sign": length_sign,
        "baseline_auroc": float(auroc_base),
        "baseline_cv5": float(cv_base),
        "deconfounded_auroc": float(auroc_deconf),
        "deconfounded_cv5": float(cv_deconf),
        "drop_from_deconfounding": float(drop),
        "verdict": verdict,
    }
    (RESULTS / "exp1_raw_length_v2.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp1_v2.done").touch()
    return results


# ---------------------------------------------------------------------------
# D1 + D2 — Exp 3a with full-covariance null and no H0_n_features
# ---------------------------------------------------------------------------


def fix_d1d2_full_cov_null(d):
    print("\n" + "=" * 60)
    print("D1+D2: Exp 3a rerun (full-covariance null, no H0_n_features)")
    print("=" * 60)

    from ripser import ripser
    from sklearn.decomposition import PCA

    def compute_5_ph_features(cloud, subsample=100, pca_components=45):
        """Same as exp3a's 6-feature function but drops H0_n_features (index 1)."""
        if len(cloud) < 5:
            return np.zeros(5)
        n_comp = min(pca_components, cloud.shape[0] - 1, cloud.shape[1])
        pca = PCA(n_components=n_comp)
        cloud_pca = pca.fit_transform(cloud)
        if len(cloud_pca) > subsample:
            idx = np.linspace(0, len(cloud_pca) - 1, subsample, dtype=int)
            cloud_pca = cloud_pca[idx]
        result = ripser(cloud_pca, maxdim=1)
        dgms = result["dgms"]
        feats = np.zeros(5)
        h0 = dgms[0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        if len(h0_finite) > 0:
            lt = h0_finite[:, 1] - h0_finite[:, 0]
            feats[0] = lt.max()
            lt_norm = lt / lt.sum() if lt.sum() > 0 else lt
            lt_norm = lt_norm[lt_norm > 0]
            feats[1] = -np.sum(lt_norm * np.log(lt_norm)) if len(lt_norm) > 0 else 0
        if len(dgms) > 1 and len(dgms[1]) > 0:
            h1 = dgms[1]
            h1_finite = h1[np.isfinite(h1[:, 1])]
            if len(h1_finite) > 0:
                lt1 = h1_finite[:, 1] - h1_finite[:, 0]
                lt1n = lt1 / lt1.sum() if lt1.sum() > 0 else lt1
                lt1n = lt1n[lt1n > 0]
                feats[2] = -np.sum(lt1n * np.log(lt1n)) if len(lt1n) > 0 else 0
                feats[3] = len(h1_finite)
                feats[4] = lt1.max()
        return feats

    def full_cov_null(cloud, n_nulls=5, seed=None):
        """Null using FULL covariance, not diagonal variance."""
        n, dim = cloud.shape
        mean = cloud.mean(axis=0)
        cov = np.cov(cloud, rowvar=False)
        cov = cov + 1e-6 * np.eye(dim)
        rng = np.random.default_rng(seed)
        feats_all = []
        for _ in range(n_nulls):
            synthetic = rng.multivariate_normal(mean, cov, size=n)
            feats_all.append(compute_5_ph_features(synthetic))
        return np.mean(feats_all, axis=0)

    traj_file = np.load(REPO / "data/experiment1_v2/trajectories.npz", allow_pickle=True)
    n_problems = 500
    real_ph = np.zeros((n_problems, 5))
    null_ph = np.zeros((n_problems, 5))

    print("  Computing PH on real + full-cov null clouds (500 problems)...")
    t0 = time.time()
    for i in range(n_problems):
        cloud = traj_file[f"traj_{i}"].astype(np.float64)
        real_ph[i] = compute_5_ph_features(cloud)
        null_ph[i] = full_cov_null(cloud, n_nulls=5, seed=42 + i)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_problems} ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.0f}s")

    ph_names = ["H0_max_lifetime", "H0_entropy", "H1_pers_entropy", "H1_n_features", "H1_max_lifetime"]
    print("\n  Real vs Full-Cov Null feature comparison:")
    feature_comparison = {}
    for j, name in enumerate(ph_names):
        rm, rs = float(real_ph[:, j].mean()), float(real_ph[:, j].std())
        nm, ns = float(null_ph[:, j].mean()), float(null_ph[:, j].std())
        print(f"    {name:20s}  real={rm:8.2f}±{rs:.2f}  null={nm:8.2f}±{ns:.2f}  diff={rm-nm:+.2f}")
        feature_comparison[name] = {"real_mean": rm, "real_std": rs, "null_mean": nm, "null_std": ns, "diff_mean": rm - nm}

    train_idx, holdout_idx = d["train_idx"], d["holdout_idx"]
    y_all = d["new_labels"]
    y_tr = y_all[train_idx]
    y_ho = y_all[holdout_idx]

    configs = {"real_ph": real_ph, "null_ph": null_ph, "diff_ph": real_ph - null_ph}
    ph_results = {}
    for name, feats in configs.items():
        f_tr = feats[train_idx]
        f_ho = feats[holdout_idx]
        auroc_ho, cv_mean, cv_std, _ = fit_lr_cv(f_tr, y_tr, f_ho, y_ho, cv=5)
        ph_results[name] = {"auroc_holdout": float(auroc_ho), "auroc_cv5": float(cv_mean), "cv_std": float(cv_std)}
        print(f"  {name:10s}: holdout={auroc_ho:.3f}, CV5={cv_mean:.3f}±{cv_std:.3f}")

    real_auroc = ph_results["real_ph"]["auroc_holdout"]
    null_auroc = ph_results["null_ph"]["auroc_holdout"]
    diff_auroc = ph_results["diff_ph"]["auroc_holdout"]

    if null_auroc >= real_auroc - 0.02:
        verdict = (
            "These 5 raw PH features do NOT outperform a full-covariance "
            "Gaussian null on this data (real=%.3f, null=%.3f). The predictive "
            "content in these 6-feature summary statistics is explained by "
            "matched covariance structure — they do not capture topology-specific signal. "
            "Implication is narrow: the ABC-44 production features include "
            "geometric terms (cosines, norms, velocities) that were NOT tested here "
            "and may carry genuine structure."
        ) % (real_auroc, null_auroc)
    elif diff_auroc > real_auroc + 0.01:
        verdict = "Topology-specific signal exists: residual features beat raw."
    else:
        verdict = f"Mixed — real={real_auroc:.3f}, null={null_auroc:.3f}, diff={diff_auroc:.3f}."
    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "exp3a_full_cov_null_v2",
        "n_problems": n_problems,
        "n_features": 5,
        "n_nulls_per_problem": 5,
        "null_type": "full_covariance_multivariate_normal_with_1e-6_I_regularization",
        "dropped_features": ["H0_n_features"],
        "classifiers": ph_results,
        "feature_comparison": feature_comparison,
        "verdict": verdict,
    }
    (RESULTS / "exp3a_gaussian_null_v2.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp3a_v2.done").touch()
    return results


# ---------------------------------------------------------------------------
# D4 + D5 — Exp 4 rerun with matched CV and O-info dtype fix
# ---------------------------------------------------------------------------


def fix_d4d5_matched_cv_and_oinfo(d):
    print("\n" + "=" * 60)
    print("D4+D5: Exp 4 rerun (matched cv=5, O-info dtype fix)")
    print("=" * 60)

    from xgboost import XGBClassifier

    X_tr, X_ho = d["X_train"], d["X_holdout"]
    y_tr, y_ho = d["y_train"], d["y_holdout"]

    # LR with cv=5 (matched)
    auroc_lr, cv_lr, std_lr, _ = fit_lr_cv(X_tr, y_tr, X_ho, y_ho, cv=5)
    print(f"\n  LR: holdout={auroc_lr:.3f}, CV5={cv_lr:.3f}±{std_lr:.3f}")

    scale_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def xgb_config(**kw):
        base = dict(
            random_state=42, scale_pos_weight=scale_pos,
            eval_metric="logloss", verbosity=0,
        )
        base.update(kw)
        return base

    xgb_configs = {
        "xgb_default": xgb_config(n_estimators=100, max_depth=3),
        "xgb_shallow": xgb_config(n_estimators=200, max_depth=2, learning_rate=0.05),
        "xgb_regularized": xgb_config(n_estimators=100, max_depth=3, learning_rate=0.1, reg_alpha=1.0, reg_lambda=5.0),
    }

    xgb_results = {}
    for name, params in xgb_configs.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier(**params))])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_ho)[:, 1]
        auroc_ho = roc_auc_score(y_ho, probs)

        fresh_pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier(**params))])
        cv_scores = cross_val_score(fresh_pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
        xgb_results[name] = {
            "auroc_holdout": float(auroc_ho),
            "auroc_cv5": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        }
        print(f"  {name:20s}: holdout={auroc_ho:.3f}, CV5={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    best_name = max(xgb_results, key=lambda k: xgb_results[k]["auroc_cv5"])
    best = xgb_results[best_name]
    lift = best["auroc_cv5"] - cv_lr
    print(f"\n  Best XGBoost ({best_name}): CV5={best['auroc_cv5']:.3f}")
    print(f"  LR CV5: {cv_lr:.3f}")
    print(f"  XGBoost lift (matched cv=5): {lift:+.3f}")

    # Save XGBoost results BEFORE attempting O-info (O-info OOMs in WSL2 via JAX)
    xgb_partial = {
        "experiment": "exp4_xgboost_oinfo_v2",
        "cv_scheme": "StratifiedKFold(n_splits=5) for both LR and XGBoost",
        "lr_auroc_holdout": float(auroc_lr),
        "lr_auroc_cv5": float(cv_lr),
        "lr_cv_std": float(std_lr),
        "xgboost_results": xgb_results,
        "best_xgboost": best_name,
        "xgboost_lift_cv5": float(lift),
        "o_information": {"status": "not_attempted_yet"},
    }
    (RESULTS / "exp4_xgboost_oinfo_v2.json").write_text(json.dumps(xgb_partial, indent=2))
    print("  (XGBoost results saved; attempting O-info in isolated subprocess)")

    oinfo_result = _run_oinfo_subprocess(X_tr, y_tr)

    if lift > 0.03:
        verdict = f"XGBoost significantly beats LR (+{lift:.3f} CV5) — signal is NONLINEAR."
    elif lift > 0.01:
        verdict = f"XGBoost mildly beats LR (+{lift:.3f} CV5) — minor nonlinear component."
    elif lift > -0.01:
        verdict = f"XGBoost ≈ LR (diff={lift:+.3f} CV5) — LR is appropriate (matched CV)."
    else:
        verdict = f"XGBoost worse than LR ({lift:+.3f} CV5) — features are linear, XGBoost overfits."

    if oinfo_result and "mean_o_information" in oinfo_result:
        verdict += f" O-info {oinfo_result['mean_o_information']:+.4f} → {oinfo_result['interpretation']}."

    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "exp4_xgboost_oinfo_v2",
        "cv_scheme": "StratifiedKFold(n_splits=5) for both LR and XGBoost",
        "lr_auroc_holdout": float(auroc_lr),
        "lr_auroc_cv5": float(cv_lr),
        "lr_cv_std": float(std_lr),
        "xgboost_results": xgb_results,
        "best_xgboost": best_name,
        "xgboost_lift_cv5": float(lift),
        "o_information": oinfo_result,
        "verdict": verdict,
    }
    (RESULTS / "exp4_xgboost_oinfo_v2.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp4_v2.done").touch()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("PATHWAY 9 AUDIT FIX RUNS")
    print("=" * 60)

    d = load_data()
    print(f"  Data loaded: train={d['X_train'].shape}, holdout={d['X_holdout'].shape}")
    print(f"  Train correct: {d['y_train'].sum()}/{len(d['y_train'])}, "
          f"Holdout correct: {d['y_holdout'].sum()}/{len(d['y_holdout'])}")

    # Baseline sanity check with cv=5
    auroc_base, cv5_base, _, _ = fit_lr_cv(d["X_train"], d["y_train"], d["X_holdout"], d["y_holdout"], cv=5)
    print(f"\n  Baseline check: holdout={auroc_base:.4f} (expected 0.7961)")
    print(f"  Baseline CV5 (new!): {cv5_base:.4f}")
    if abs(auroc_base - 0.7961) > 0.005:
        raise SystemExit(f"BASELINE DRIFT: {auroc_base:.4f} vs expected 0.7961. Aborting.")

    results = {}

    # D6 first (fast: just tokenizer + 44 LRs)
    if not (RESULTS / "exp1_v2.done").exists():
        results["exp1_v2"] = fix_d6_tokenizer_length(d)
    else:
        print("\n  Exp 1 v2 already done, skipping.")
        results["exp1_v2"] = json.loads((RESULTS / "exp1_raw_length_v2.json").read_text())

    # D4+D5 (fast: 3 XGBoost fits + O-info)
    if not (RESULTS / "exp4_v2.done").exists():
        results["exp4_v2"] = fix_d4d5_matched_cv_and_oinfo(d)
    else:
        print("\n  Exp 4 v2 already done, skipping.")
        results["exp4_v2"] = json.loads((RESULTS / "exp4_xgboost_oinfo_v2.json").read_text())

    # D1+D2 last (slow: 500 problems × 5 full-cov null draws)
    if not (RESULTS / "exp3a_v2.done").exists():
        results["exp3a_v2"] = fix_d1d2_full_cov_null(d)
    else:
        print("\n  Exp 3a v2 already done, skipping.")
        results["exp3a_v2"] = json.loads((RESULTS / "exp3a_gaussian_null_v2.json").read_text())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n  {name}: {r.get('verdict', 'N/A')}")

    summary = {
        name: {"verdict": r.get("verdict")}
        for name, r in results.items()
    }
    summary["key_numbers"] = {
        "exp1_v2_baseline_auroc": results["exp1_v2"].get("baseline_auroc"),
        "exp1_v2_deconfounded_auroc": results["exp1_v2"].get("deconfounded_auroc"),
        "exp1_v2_raw_length_only_auroc": results["exp1_v2"].get("raw_length_only_auroc"),
        "exp4_v2_lr_cv5": results["exp4_v2"].get("lr_auroc_cv5"),
        "exp4_v2_xgb_best_cv5": results["exp4_v2"]["xgboost_results"][results["exp4_v2"]["best_xgboost"]]["auroc_cv5"],
        "exp4_v2_lift_cv5": results["exp4_v2"].get("xgboost_lift_cv5"),
        "exp4_v2_o_info": results["exp4_v2"].get("o_information"),
        "exp3a_v2_real_ph_auroc": results["exp3a_v2"]["classifiers"]["real_ph"]["auroc_holdout"],
        "exp3a_v2_null_ph_auroc": results["exp3a_v2"]["classifiers"]["null_ph"]["auroc_holdout"],
    }
    (RESULTS / "fixes_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAll fix runs complete. Results in pathway9/results/*_v2.json")


if __name__ == "__main__":
    main()
