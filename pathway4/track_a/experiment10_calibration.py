#!/usr/bin/env python3
"""Experiment 10: Calibration Analysis.

Answers: are topo-confidence predicted probabilities well-calibrated?
AUROC measures discrimination (ranking); calibration measures whether
P(correct)=0.7 really means 70% accuracy. This feeds Experiment 8
(adaptive sampling budget) — rational thresholding requires calibrated scores.

Metrics:
  - ECE (Expected Calibration Error) with 15 equal-width AND 15 adaptive bins
  - Brier score
  - NLL (negative log-likelihood)
  - Reliability diagram data (JSON, both binning strategies)

Post-hoc recalibration:
  - Temperature scaling (manual, scipy)
  - Platt scaling (sklearn CalibratedClassifierCV, method="sigmoid")
  - Isotonic regression (sklearn CalibratedClassifierCV, method="isotonic")

Zero GPU required — operates on precomputed features.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BASELINE_CORRECT_PATH,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    LR_PARAMS,
    get_abc_column_indices,
    get_train_holdout_indices,
    logger,
)

OUTPUT_DIR = Path(__file__).parent / "experiment10_calibration"
N_BINS = 15
N_BOOTSTRAP = 2000
CLIP_EPS = 1e-7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
    strategy: str = "uniform",
) -> tuple[float, list[dict]]:
    """Expected Calibration Error with per-bin details.

    strategy="uniform": equal-width bins (standard ECE).
    strategy="quantile": adaptive bins (equal-count, more robust for skewed probs).
    """
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        # Deduplicate edges from ties
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    bins_detail = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:  # last bin includes upper edge
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_in = int(mask.sum())
        if n_in == 0:
            bins_detail.append({
                "lower": round(float(lo), 6),
                "upper": round(float(hi), 6),
                "center": round(float((lo + hi) / 2), 6),
                "mean_predicted": None,
                "observed_fraction": None,
                "n_samples": 0,
                "gap": None,
            })
            continue

        mean_pred = float(y_prob[mask].mean())
        obs_frac = float(y_true[mask].mean())
        gap = mean_pred - obs_frac
        ece += n_in * abs(gap)

        bins_detail.append({
            "lower": round(float(lo), 6),
            "upper": round(float(hi), 6),
            "center": round(float((lo + hi) / 2), 6),
            "mean_predicted": round(mean_pred, 6),
            "observed_fraction": round(obs_frac, 6),
            "n_samples": n_in,
            "gap": round(gap, 6),
        })

    ece /= len(y_true)
    return float(ece), bins_detail


def temperature_scale(
    logits_train: np.ndarray,
    y_train: np.ndarray,
    logits_holdout: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit temperature T on train logits, apply to holdout.

    T > 1: model is overconfident → soften probabilities.
    T < 1: model is underconfident → sharpen probabilities.
    """
    def nll(T: float) -> float:
        p = expit(logits_train / T)
        p = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
        return log_loss(y_train, p)

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T_opt = float(result.x)
    probs_cal = expit(logits_holdout / T_opt)
    return probs_cal, T_opt


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for a metric function(y_true, y_prob) -> float."""
    rng = np.random.RandomState(seed)
    point = metric_fn(y_true, y_prob)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(metric_fn(y_true[idx], y_prob[idx]))
    boots = np.array(boots)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return point, ci_lo, ci_hi


def bootstrap_auroc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """AUROC with bootstrap 95% CI (same as exp 9)."""
    return bootstrap_metric(y_true, scores, roc_auc_score, n_boot, seed)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    do_bootstrap: bool = True,
) -> dict:
    """Compute ECE (both), Brier, NLL, AUROC for one set of probabilities."""
    y_prob_clipped = np.clip(y_prob, CLIP_EPS, 1 - CLIP_EPS)

    ece_ew, bins_ew = compute_ece(y_true, y_prob, N_BINS, "uniform")
    ece_ad, bins_ad = compute_ece(y_true, y_prob, N_BINS, "quantile")
    brier = float(brier_score_loss(y_true, y_prob_clipped))
    nll = float(log_loss(y_true, y_prob_clipped))
    auroc = float(roc_auc_score(y_true, y_prob))

    n_nonempty_ew = sum(1 for b in bins_ew if b["n_samples"] > 0)

    result = {
        "label": label,
        "ece_equal_width_15": round(ece_ew, 4),
        "ece_adaptive_15": round(ece_ad, 4),
        "brier": round(brier, 4),
        "nll": round(nll, 4),
        "auroc": round(auroc, 4),
        "n_nonempty_bins_ew": n_nonempty_ew,
        "reliability_equal_width": bins_ew,
        "reliability_adaptive": bins_ad,
    }

    if do_bootstrap:
        def ece_ew_fn(yt, yp):
            return compute_ece(yt, yp, N_BINS, "uniform")[0]

        def ece_ad_fn(yt, yp):
            return compute_ece(yt, yp, N_BINS, "quantile")[0]

        _, ece_ew_lo, ece_ew_hi = bootstrap_metric(y_true, y_prob, ece_ew_fn)
        _, ece_ad_lo, ece_ad_hi = bootstrap_metric(y_true, y_prob, ece_ad_fn)
        _, brier_lo, brier_hi = bootstrap_metric(
            y_true, y_prob_clipped,
            lambda yt, yp: float(brier_score_loss(yt, yp)),
        )
        _, auroc_lo, auroc_hi = bootstrap_auroc(y_true, y_prob)

        result["ece_equal_width_15_ci"] = [round(ece_ew_lo, 4), round(ece_ew_hi, 4)]
        result["ece_adaptive_15_ci"] = [round(ece_ad_lo, 4), round(ece_ad_hi, 4)]
        result["brier_ci"] = [round(brier_lo, 4), round(brier_hi, 4)]
        result["auroc_ci"] = [round(auroc_lo, 4), round(auroc_hi, 4)]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("=" * 75)
    logger.info("EXPERIMENT 10: CALIBRATION ANALYSIS")
    logger.info("=" * 75)

    # ---- Step 1: Load data & fit model (identical to exp 9) ----
    logger.info("\nStep 1: Loading data and fitting topo-confidence model...")

    baseline_correct = np.load(BASELINE_CORRECT_PATH)
    _, holdout_idx = get_train_holdout_indices()

    # SSS ordering reconstruction
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_idx_sss, hold_idx_sss = next(sss.split(
        np.zeros(len(baseline_correct)), baseline_correct.astype(int)
    ))
    train_sort_perm = np.argsort(train_idx_sss)
    hold_sort_perm = np.argsort(hold_idx_sss)

    train_features_78 = np.load(FEATURES_TRAIN_PATH)[train_sort_perm]
    holdout_features_78 = np.load(FEATURES_HOLDOUT_PATH)[hold_sort_perm]

    train_idx_sorted = np.sort(train_idx_sss)
    hold_idx_sorted = np.sort(hold_idx_sss)
    assert np.array_equal(hold_idx_sorted, holdout_idx), "Holdout index mismatch!"

    with open(Path(__file__).parent / "phase1" / "feature_names.json") as f:
        feature_names = json.load(f)

    abc_indices = get_abc_column_indices(feature_names)
    X_train = train_features_78[:, abc_indices]
    X_holdout = holdout_features_78[:, abc_indices]
    y_train = baseline_correct[train_idx_sorted].astype(int)
    y_holdout = baseline_correct[hold_idx_sorted].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_scaled, y_train)
    probs_uncal = clf.predict_proba(X_holdout_scaled)[:, 1]

    n_train_correct = int(y_train.sum())
    n_holdout_correct = int(y_holdout.sum())
    logger.info("  Train: %d problems, %d correct (%.1f%%)",
                len(y_train), n_train_correct, 100 * y_train.mean())
    logger.info("  Holdout: %d problems, %d correct (%.1f%%)",
                len(y_holdout), n_holdout_correct, 100 * y_holdout.mean())
    logger.info("  Prob distribution: min=%.4f, median=%.4f, mean=%.4f, max=%.4f",
                probs_uncal.min(), np.median(probs_uncal), probs_uncal.mean(), probs_uncal.max())
    logger.info("  Probs < 0.1: %d/%d, Probs > 0.5: %d/%d",
                (probs_uncal < 0.1).sum(), len(probs_uncal),
                (probs_uncal > 0.5).sum(), len(probs_uncal))

    # ---- Step 2: Uncalibrated metrics ----
    logger.info("\nStep 2: Uncalibrated calibration metrics on holdout...")
    uncal = compute_all_metrics(y_holdout, probs_uncal, "uncalibrated")
    logger.info("  ECE (15 equal-width): %.4f [%.4f–%.4f]  (%d/%d bins non-empty)",
                uncal["ece_equal_width_15"], uncal["ece_equal_width_15_ci"][0],
                uncal["ece_equal_width_15_ci"][1], uncal["n_nonempty_bins_ew"], N_BINS)
    logger.info("  ECE (15 adaptive):    %.4f [%.4f–%.4f]",
                uncal["ece_adaptive_15"], uncal["ece_adaptive_15_ci"][0],
                uncal["ece_adaptive_15_ci"][1])
    logger.info("  Brier:                %.4f [%.4f–%.4f]",
                uncal["brier"], uncal["brier_ci"][0], uncal["brier_ci"][1])
    logger.info("  NLL:                  %.4f", uncal["nll"])
    logger.info("  AUROC:                %.4f [%.4f–%.4f]",
                uncal["auroc"], uncal["auroc_ci"][0], uncal["auroc_ci"][1])

    # Naive baseline for context: predict base rate for everyone
    base_rate = y_holdout.mean()
    naive_brier = float(brier_score_loss(y_holdout, np.full(len(y_holdout), base_rate)))
    logger.info("  Naive Brier (predict base rate %.2f): %.4f", base_rate, naive_brier)

    # ---- Step 3: CV calibration on train-400 ----
    logger.info("\nStep 3: Cross-validated calibration on train-400...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs_train = cross_val_predict(
        LogisticRegression(**LR_PARAMS),
        X_train_scaled, y_train, cv=skf, method="predict_proba",
    )[:, 1]

    cv_train = compute_all_metrics(y_train, cv_probs_train, "cv_train400", do_bootstrap=False)
    logger.info("  ECE (15 equal-width): %.4f  (%d/%d bins non-empty)",
                cv_train["ece_equal_width_15"], cv_train["n_nonempty_bins_ew"], N_BINS)
    logger.info("  ECE (15 adaptive):    %.4f", cv_train["ece_adaptive_15"])
    logger.info("  Brier:                %.4f", cv_train["brier"])
    logger.info("  NLL:                  %.4f", cv_train["nll"])

    # ---- Step 4: Temperature scaling ----
    logger.info("\nStep 4: Temperature scaling...")

    # Get CV logits on train for fitting T
    cv_logits_train = logit(np.clip(cv_probs_train, CLIP_EPS, 1 - CLIP_EPS))
    holdout_logits = logit(np.clip(probs_uncal, CLIP_EPS, 1 - CLIP_EPS))

    probs_temp, T_opt = temperature_scale(cv_logits_train, y_train, holdout_logits)
    temp = compute_all_metrics(y_holdout, probs_temp, "temperature_scaled")
    temp["optimal_T"] = round(T_opt, 4)

    interp = "overconfident" if T_opt > 1 else "underconfident"
    logger.info("  Optimal T = %.4f (%s)", T_opt, interp)
    logger.info("  ECE (15 equal-width): %.4f [%.4f–%.4f]",
                temp["ece_equal_width_15"], temp["ece_equal_width_15_ci"][0],
                temp["ece_equal_width_15_ci"][1])
    logger.info("  ECE (15 adaptive):    %.4f [%.4f–%.4f]",
                temp["ece_adaptive_15"], temp["ece_adaptive_15_ci"][0],
                temp["ece_adaptive_15_ci"][1])
    logger.info("  Brier:                %.4f", temp["brier"])

    # ---- Step 5: Platt scaling ----
    logger.info("\nStep 5: Platt scaling (CalibratedClassifierCV, sigmoid)...")
    cal_platt = CalibratedClassifierCV(
        LogisticRegression(**LR_PARAMS), method="sigmoid", cv=5,
    )
    cal_platt.fit(X_train_scaled, y_train)
    probs_platt = cal_platt.predict_proba(X_holdout_scaled)[:, 1]

    platt = compute_all_metrics(y_holdout, probs_platt, "platt")
    logger.info("  ECE (15 equal-width): %.4f [%.4f–%.4f]",
                platt["ece_equal_width_15"], platt["ece_equal_width_15_ci"][0],
                platt["ece_equal_width_15_ci"][1])
    logger.info("  ECE (15 adaptive):    %.4f [%.4f–%.4f]",
                platt["ece_adaptive_15"], platt["ece_adaptive_15_ci"][0],
                platt["ece_adaptive_15_ci"][1])
    logger.info("  Brier:                %.4f", platt["brier"])

    # ---- Step 6: Isotonic regression ----
    logger.info("\nStep 6: Isotonic regression (CalibratedClassifierCV)...")
    logger.info("  WARNING: n=100 holdout with 11 positive — isotonic may overfit.")
    cal_iso = CalibratedClassifierCV(
        LogisticRegression(**LR_PARAMS), method="isotonic", cv=5,
    )
    cal_iso.fit(X_train_scaled, y_train)
    probs_iso = cal_iso.predict_proba(X_holdout_scaled)[:, 1]

    iso = compute_all_metrics(y_holdout, probs_iso, "isotonic")
    iso["warning"] = "Isotonic may overfit with n=100 (11 positive)"
    logger.info("  ECE (15 equal-width): %.4f [%.4f–%.4f]",
                iso["ece_equal_width_15"], iso["ece_equal_width_15_ci"][0],
                iso["ece_equal_width_15_ci"][1])
    logger.info("  ECE (15 adaptive):    %.4f [%.4f–%.4f]",
                iso["ece_adaptive_15"], iso["ece_adaptive_15_ci"][0],
                iso["ece_adaptive_15_ci"][1])
    logger.info("  Brier:                %.4f", iso["brier"])

    # ---- Step 7: Comparison table ----
    logger.info("\n" + "=" * 75)
    logger.info("CALIBRATION COMPARISON TABLE")
    logger.info("=" * 75)
    logger.info("  %-18s  %10s  %13s  %8s  %8s  %8s",
                "Method", "ECE-EW(15)", "ECE-Adapt(15)", "Brier", "NLL", "AUROC")
    logger.info("  " + "-" * 71)

    for m in [uncal, temp, platt, iso]:
        logger.info("  %-18s  %10.4f  %13.4f  %8.4f  %8.4f  %8.4f",
                     m["label"], m["ece_equal_width_15"], m["ece_adaptive_15"],
                     m["brier"], m["nll"], m["auroc"])

    logger.info("  " + "-" * 71)
    logger.info("  %-18s  %10.4f  %13s  %8.4f  %8s  %8s",
                "Naive (base rate)", naive_brier, "—", naive_brier, "—", "—")
    logger.info("")
    logger.info("  Temperature T = %.4f (%s)", T_opt, interp)
    logger.info("  Holdout: %d samples (%d correct, %d incorrect)",
                len(y_holdout), n_holdout_correct, len(y_holdout) - n_holdout_correct)
    logger.info("  Equal-width bins: %d/%d non-empty (uncalibrated)",
                uncal["n_nonempty_bins_ew"], N_BINS)

    # ---- Step 8: Experiment 8 feed-forward ----
    # Determine best calibration method and quality
    methods = {"uncalibrated": uncal, "temperature_scaled": temp,
               "platt": platt, "isotonic": iso}
    best_method_name = min(methods, key=lambda k: methods[k]["ece_adaptive_15"])
    best_ece = methods[best_method_name]["ece_adaptive_15"]

    if best_ece < 0.05:
        quality = "adequate"
    elif best_ece < 0.10:
        quality = "marginal"
    else:
        quality = "poor"

    # Threshold reliability: what's the observed accuracy at key thresholds?
    best_probs = {"uncalibrated": probs_uncal, "temperature_scaled": probs_temp,
                  "platt": probs_platt, "isotonic": probs_iso}[best_method_name]
    threshold_reliability = {}
    for tau in [0.3, 0.5, 0.7]:
        trusted = best_probs >= tau
        n_trusted = int(trusted.sum())
        if n_trusted > 0:
            obs_acc = float(y_holdout[trusted].mean())
            mean_cal_prob = float(best_probs[trusted].mean())
        else:
            obs_acc = None
            mean_cal_prob = None
        threshold_reliability[f"tau_{tau}"] = {
            "n_trusted": n_trusted,
            "mean_calibrated_prob": round(mean_cal_prob, 4) if mean_cal_prob is not None else None,
            "observed_accuracy": round(obs_acc, 4) if obs_acc is not None else None,
        }

    exp8_feed = {
        "calibration_quality": quality,
        "best_ece_adaptive": best_ece,
        "recommended_method": best_method_name,
        "threshold_reliability": threshold_reliability,
    }

    logger.info("\n" + "=" * 75)
    logger.info("EXPERIMENT 8 FEED-FORWARD")
    logger.info("=" * 75)
    logger.info("  Calibration quality: %s (best adaptive ECE = %.4f via %s)",
                quality, best_ece, best_method_name)
    for tau_key, tr in threshold_reliability.items():
        tau_val = tau_key.replace("tau_", "")
        if tr["observed_accuracy"] is not None:
            logger.info("  %s: %d problems trusted, mean P(correct)=%.3f, observed accuracy=%.3f",
                        tau_val, tr["n_trusted"], tr["mean_calibrated_prob"],
                        tr["observed_accuracy"])
        else:
            logger.info("  %s: 0 problems trusted", tau_val)

    # ---- Step 9: Save JSON outputs ----
    logger.info("\nSaving results...")

    # Strip reliability data from main results (stored separately)
    def strip_reliability(d: dict) -> dict:
        return {k: v for k, v in d.items()
                if k not in ("reliability_equal_width", "reliability_adaptive")}

    main_results = {
        "experiment": "calibration_analysis",
        "description": "Calibration analysis for topo-confidence logistic regression "
                       "(44 ABC features). ECE, Brier, NLL, reliability data, "
                       "and post-hoc recalibration (temperature, Platt, isotonic).",
        "n_holdout": len(y_holdout),
        "n_correct_holdout": n_holdout_correct,
        "n_train": len(y_train),
        "n_correct_train": n_train_correct,
        "naive_brier_baseline": round(naive_brier, 4),
        "uncalibrated": strip_reliability(uncal),
        "cv_train400": strip_reliability(cv_train),
        "temperature_scaled": strip_reliability(temp),
        "platt": strip_reliability(platt),
        "isotonic": strip_reliability(iso),
        "comparison_table": [
            {
                "method": m["label"],
                "ece_ew_15": m["ece_equal_width_15"],
                "ece_adapt_15": m["ece_adaptive_15"],
                "brier": m["brier"],
                "nll": m["nll"],
                "auroc": m["auroc"],
            }
            for m in [uncal, temp, platt, iso]
        ],
        "experiment8_feed": exp8_feed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_DIR / "calibration_results.json", "w") as f:
        json.dump(main_results, f, indent=2)

    # Reliability diagram data
    reliability_data = {}
    for m, probs, label in [
        (uncal, probs_uncal, "uncalibrated"),
        (temp, probs_temp, "temperature_scaled"),
        (platt, probs_platt, "platt"),
        (iso, probs_iso, "isotonic"),
    ]:
        reliability_data[label] = {
            "equal_width_15": m["reliability_equal_width"],
            "adaptive_15": m["reliability_adaptive"],
        }

    with open(OUTPUT_DIR / "reliability_diagram_data.json", "w") as f:
        json.dump(reliability_data, f, indent=2)

    # Per-problem scores
    per_problem = {
        "holdout_indices": hold_idx_sorted.tolist(),
        "holdout_correct": y_holdout.tolist(),
        "uncalibrated": np.round(probs_uncal, 6).tolist(),
        "temperature_scaled": np.round(probs_temp, 6).tolist(),
        "platt": np.round(probs_platt, 6).tolist(),
        "isotonic": np.round(probs_iso, 6).tolist(),
    }

    with open(OUTPUT_DIR / "per_problem_scores.json", "w") as f:
        json.dump(per_problem, f, indent=2)

    logger.info("Results saved to: %s", OUTPUT_DIR)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
