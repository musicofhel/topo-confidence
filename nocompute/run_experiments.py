#!/usr/bin/env python3
"""No-compute experiment plan: A2–G1.

Run after A1_cache.py has built the mean-pool caches.
Writes results to results/ and figures to figs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lib import (
    CACHE_DIR, FIGS_DIR, N_FOLDS, N_LAYERS, RESULTS_DIR, SEED, STEERING_LAYER,
    DomProbe, aurc, bootstrap_auroc, delong_ci, ece_score, ensure_dirs,
    load_cache, oof_dom_calibrated, oof_dom_scores, oracle_aurc,
    risk_coverage_curve, save_fig, save_json, setup_matplotlib, timer,
)


# ===== TIER A: Statistical Hardening =========================================

def A2_bootstrap_ci():
    """Bootstrap 95% CIs on AUROC at L19 (1.5B, 7B)."""
    results = {}
    for tag, cache_name in [("1p5b", "math500_1p5b"), ("7b", "math500_7b")]:
        c = load_cache(cache_name)
        X_prefill_L19 = c["X_prefill"][:, STEERING_LAYER, :]
        y = c["y"].astype(bool)

        raw_scores = oof_dom_scores(X_prefill_L19, y)
        cal_probs, _ = oof_dom_calibrated(X_prefill_L19, y)

        raw_auroc = float(roc_auc_score(y, raw_scores))
        cal_auroc = float(roc_auc_score(y, cal_probs))

        bstrap = bootstrap_auroc(y, raw_scores)
        dl = delong_ci(y, raw_scores)

        results[tag] = {
            "raw_auroc": raw_auroc,
            "calibrated_auroc": cal_auroc,
            **{f"bootstrap_{k}": v for k, v in bstrap.items()},
            **{f"delong_{k}": v for k, v in dl.items()},
            "n": int(len(y)),
            "n_pos": int(y.sum()),
        }
        ci_w = bstrap["bca_hi"] - bstrap["bca_lo"]
        print(f"  {tag}: AUROC={raw_auroc:.4f}, 95% BCa [{bstrap['bca_lo']:.4f}, {bstrap['bca_hi']:.4f}] (width={ci_w:.4f})")
        print(f"       DeLong [{dl['delong_lo']:.4f}, {dl['delong_hi']:.4f}]")

        np.savez(
            CACHE_DIR / f"oof_scores_{tag}.npz",
            raw_scores=raw_scores, cal_probs=cal_probs, y=y,
        )

    bca_agree = all(
        abs(results[t][f"bootstrap_bca_{end}"] - results[t][f"delong_delong_{end}"]) < 0.005
        for t in results for end in ["lo", "hi"]
    )
    # Use approximate check — just compare point estimates
    results["bca_delong_agree_within_005"] = bca_agree
    save_json(RESULTS_DIR / "A2_auroc_ci.json", results)


def A3_seed_variance():
    """Multi-seed DomProbe variance (15 seeds)."""
    seeds = [SEED] + list(range(1, 15))
    results = {}

    for tag, cache_name in [("1p5b", "math500_1p5b"), ("7b", "math500_7b")]:
        c = load_cache(cache_name)
        X = c["X_prefill"][:, STEERING_LAYER, :]
        y = c["y"].astype(bool)

        aurocs = []
        for seed in seeds:
            scores = oof_dom_scores(X, y, seed=seed)
            aurocs.append(float(roc_auc_score(y, scores)))

        results[tag] = {
            "seeds": seeds,
            "aurocs": aurocs,
            "mean": float(np.mean(aurocs)),
            "std": float(np.std(aurocs)),
            "min": float(np.min(aurocs)),
            "max": float(np.max(aurocs)),
        }
        print(f"  {tag}: mean={np.mean(aurocs):.4f} ± {np.std(aurocs):.4f} (range [{np.min(aurocs):.4f}, {np.max(aurocs):.4f}])")

    save_json(RESULTS_DIR / "A3_seed_variance.json", results)


def A4_calibration():
    """Calibration analysis: ECE, Brier, reliability diagrams."""
    plt = setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    results = {}

    for ax_idx, (tag, cache_name) in enumerate([("1p5b", "math500_1p5b"), ("7b", "math500_7b")]):
        c = load_cache(cache_name)
        X = c["X_prefill"][:, STEERING_LAYER, :]
        y = c["y"].astype(bool)

        cal_probs, raw_scores = oof_dom_calibrated(X, y)

        # Sigmoid-normalize raw scores for pre-isotonic calibration metrics
        from scipy.special import expit
        raw_probs = expit(raw_scores / (raw_scores.std() + 1e-12))

        pre_iso = {
            "ece_uniform": ece_score(y, raw_probs, 15, "uniform"),
            "ece_quantile": ece_score(y, raw_probs, 15, "quantile"),
            "brier": float(brier_score_loss(y, raw_probs)),
        }
        post_iso = {
            "ece_uniform": ece_score(y, cal_probs, 15, "uniform"),
            "ece_quantile": ece_score(y, cal_probs, 15, "quantile"),
            "brier": float(brier_score_loss(y, cal_probs)),
        }
        results[tag] = {"pre_isotonic": pre_iso, "post_isotonic": post_iso}
        print(f"  {tag} pre-iso:  ECE(eq-mass)={pre_iso['ece_quantile']:.4f}, Brier={pre_iso['brier']:.4f}")
        print(f"  {tag} post-iso: ECE(eq-mass)={post_iso['ece_quantile']:.4f}, Brier={post_iso['brier']:.4f}")

        # Reliability diagram
        ax = axes[ax_idx]
        for label, probs, ls in [("Pre-isotonic", raw_probs, "--"), ("Post-isotonic", cal_probs, "-")]:
            fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, fraction_pos, ls, marker="o", label=label)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability — {tag}")
        ax.legend()

        # Histogram of predictions
        ax_hist = ax.inset_axes([0.55, 0.05, 0.4, 0.25])
        ax_hist.hist(cal_probs, bins=20, alpha=0.7, color="steelblue")
        ax_hist.set_xlabel("P(correct)", fontsize=8)
        ax_hist.set_ylabel("Count", fontsize=8)
        ax_hist.tick_params(labelsize=7)

    fig.tight_layout()
    save_fig(fig, "A4_reliability.png")
    plt.close(fig)
    save_json(RESULTS_DIR / "A4_calibration.json", results)


def A5_per_subject():
    """Per-subject AUROC on MATH-500 (1.5B and 7B)."""
    plt = setup_matplotlib()
    results = {}

    for tag, cache_name in [("1p5b", "math500_1p5b"), ("7b", "math500_7b")]:
        c = load_cache(cache_name)
        X = c["X_prefill"][:, STEERING_LAYER, :]
        y = c["y"].astype(bool)
        subjects = c["subjects"]

        scores = oof_dom_scores(X, y)
        unique_subjects = sorted(set(subjects))

        rows = []
        for subj in unique_subjects:
            mask = subjects == subj
            n_s = mask.sum()
            n_pos = y[mask].sum()
            n_neg = n_s - n_pos
            if n_pos < 5 or n_neg < 5:
                rows.append({"subject": subj, "n": int(n_s), "n_pos": int(n_pos),
                             "auroc": None, "note": "too few of one class"})
                continue
            ba = bootstrap_auroc(y[mask], scores[mask], n_resamples=1000)
            rows.append({"subject": subj, "n": int(n_s), "n_pos": int(n_pos),
                         **ba})
            print(f"  {tag} {subj:30s} n={n_s:3d} AUROC={ba['point']:.3f} [{ba['bca_lo']:.3f}, {ba['bca_hi']:.3f}]")

        results[tag] = rows

    save_json(RESULTS_DIR / "A5_per_subject.json", results)

    # Forest plot for 1.5B
    rows_1p5b = [r for r in results["1p5b"] if r.get("point") is not None]
    if rows_1p5b:
        fig, ax = plt.subplots(figsize=(8, len(rows_1p5b) * 0.6 + 1))
        subjects = [r["subject"] for r in rows_1p5b]
        points = [r["point"] for r in rows_1p5b]
        lo = [r["bca_lo"] for r in rows_1p5b]
        hi = [r["bca_hi"] for r in rows_1p5b]
        y_pos = range(len(subjects))
        ax.errorbar(points, y_pos, xerr=[np.array(points) - np.array(lo),
                     np.array(hi) - np.array(points)],
                     fmt="o", capsize=4, color="steelblue")
        ax.axvline(0.5, color="gray", ls=":", alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(subjects)
        ax.set_xlabel("AUROC")
        ax.set_title("Per-subject AUROC — 1.5B MATH-500")
        fig.tight_layout()
        save_fig(fig, "A5_per_subject_forest.png")
        plt.close(fig)


# ===== TIER B: Layer-Sweep Methodology ========================================

def _layer_sweep(X_all_layers, y, name, n_bootstrap=1000):
    """Sweep AUROC across all layers. Returns list of dicts."""
    rows = []
    for L in range(N_LAYERS):
        X_L = X_all_layers[:, L, :]
        scores = oof_dom_scores(X_L, y)
        auroc_val = float(roc_auc_score(y, scores))
        ba = bootstrap_auroc(y, scores, n_resamples=n_bootstrap)
        brier = float(brier_score_loss(y, (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)))
        rows.append({"layer": L, "auroc": auroc_val, "bca_lo": ba["bca_lo"],
                      "bca_hi": ba["bca_hi"], "brier": brier})
        if L % 5 == 0:
            print(f"  {name} L{L:02d}: AUROC={auroc_val:.4f}")
    return rows


def B1_layer_sweep_1p5b():
    """Per-layer AUROC sweep, 1.5B MATH-500 (mean-pool)."""
    plt = setup_matplotlib()
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)

    rows_mean = _layer_sweep(c["X_mean"], y, "1.5B-meanpool")
    rows_prefill = _layer_sweep(c["X_prefill"], y, "1.5B-prefill")

    peak_mean = max(rows_mean, key=lambda r: r["auroc"])
    peak_prefill = max(rows_prefill, key=lambda r: r["auroc"])

    result = {
        "meanpool": rows_mean,
        "prefill": rows_prefill,
        "peak_mean": {"layer": peak_mean["layer"], "auroc": peak_mean["auroc"],
                       "depth_frac": peak_mean["layer"] / 28},
        "peak_prefill": {"layer": peak_prefill["layer"], "auroc": peak_prefill["auroc"],
                          "depth_frac": peak_prefill["layer"] / 28},
        "L19_mean_auroc": rows_mean[19]["auroc"],
        "L19_prefill_auroc": rows_prefill[19]["auroc"],
    }
    print(f"  Peak (meanpool): L{peak_mean['layer']} AUROC={peak_mean['auroc']:.4f} (depth={peak_mean['layer']/28:.2f})")
    print(f"  Peak (prefill):  L{peak_prefill['layer']} AUROC={peak_prefill['auroc']:.4f}")

    # Plot
    fig, ax = plt.subplots()
    for label, rows, c_color in [("Mean-pool", rows_mean, "steelblue"), ("Prefill", rows_prefill, "coral")]:
        layers = [r["layer"] for r in rows]
        aurocs = [r["auroc"] for r in rows]
        lo = [r["bca_lo"] for r in rows]
        hi = [r["bca_hi"] for r in rows]
        ax.plot(layers, aurocs, "-o", ms=4, label=label, color=c_color)
        ax.fill_between(layers, lo, hi, alpha=0.15, color=c_color)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Layer sweep — 1.5B MATH-500")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "B1_layer_sweep_1p5b.png")
    plt.close(fig)

    save_json(RESULTS_DIR / "B1_layer_sweep_1p5b.json", result)
    return result


def B2_layer_sweep_7b():
    """Per-layer AUROC sweep, 7B MATH-500 (mean-pool)."""
    plt = setup_matplotlib()
    c = load_cache("math500_7b")
    y = c["y"].astype(bool)

    rows_mean = _layer_sweep(c["X_mean"], y, "7B-meanpool")
    rows_prefill = _layer_sweep(c["X_prefill"], y, "7B-prefill")

    peak_mean = max(rows_mean, key=lambda r: r["auroc"])
    peak_prefill = max(rows_prefill, key=lambda r: r["auroc"])

    result = {
        "meanpool": rows_mean,
        "prefill": rows_prefill,
        "peak_mean": {"layer": peak_mean["layer"], "auroc": peak_mean["auroc"],
                       "depth_frac": peak_mean["layer"] / 28},
        "peak_prefill": {"layer": peak_prefill["layer"], "auroc": peak_prefill["auroc"],
                          "depth_frac": peak_prefill["layer"] / 28},
    }
    print(f"  Peak (meanpool): L{peak_mean['layer']} AUROC={peak_mean['auroc']:.4f} (depth={peak_mean['layer']/28:.2f})")

    save_json(RESULTS_DIR / "B2_layer_sweep_7b.json", result)

    # Overlay plot (depth-fraction x-axis)
    fig, ax = plt.subplots()
    depth_frac = np.arange(N_LAYERS) / 28
    for label, rows_1p5b_file, rows_7b, color in [
        ("1.5B", None, None, "steelblue"),
        ("7B", None, None, "coral"),
    ]:
        pass  # will plot below

    # Load B1 results for overlay
    b1_path = RESULTS_DIR / "B1_layer_sweep_1p5b.json"
    if b1_path.exists():
        b1 = json.loads(b1_path.read_text())
        aurocs_1p5b = [r["auroc"] for r in b1["meanpool"]]
        ax.plot(depth_frac, aurocs_1p5b, "-o", ms=4, label="1.5B", color="steelblue")

    aurocs_7b = [r["auroc"] for r in rows_mean]
    ax.plot(depth_frac, aurocs_7b, "-s", ms=4, label="7B", color="coral")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Depth fraction (layer / 28)")
    ax.set_ylabel("AUROC")
    ax.set_title("Layer sweep — 1.5B vs 7B (mean-pool)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "B12_overlay.png")
    plt.close(fig)

    return result


def B3_layer_sweep_bbh():
    """Per-layer sweep on BBH (1.5B, per-subset & pooled)."""
    c = load_cache("bbh_1p5b")
    subsets_arr = c["subsets"]
    y_all = c["y"].astype(bool)
    X_mean_all = c["X_mean"]

    results = {}
    for subset in list(set(subsets_arr)) + ["pooled"]:
        if subset == "pooled":
            mask = np.ones(len(y_all), dtype=bool)
        else:
            mask = subsets_arr == subset
        y_sub = y_all[mask]
        X_sub = X_mean_all[mask]

        n_pos = y_sub.sum()
        n_neg = len(y_sub) - n_pos
        if n_pos < 10 or n_neg < 10:
            print(f"  {subset}: skipping (n_pos={n_pos}, n_neg={n_neg})")
            results[subset] = {"skipped": True, "n_pos": int(n_pos), "n_neg": int(n_neg)}
            continue

        rows = []
        for L in range(N_LAYERS):
            scores = oof_dom_scores(X_sub[:, L, :], y_sub)
            a = float(roc_auc_score(y_sub, scores))
            rows.append({"layer": L, "auroc": a})

        peak = max(rows, key=lambda r: r["auroc"])
        results[subset] = {
            "layers": rows,
            "peak_layer": peak["layer"],
            "peak_auroc": peak["auroc"],
            "n": int(len(y_sub)),
            "n_pos": int(n_pos),
        }
        short = subset.split("_")[0] if "_" in subset else subset
        print(f"  {short:20s} peak=L{peak['layer']} AUROC={peak['auroc']:.4f} (n={len(y_sub)})")

    save_json(RESULTS_DIR / "B3_layer_sweep_bbh.json", results)
    return results


def B4_cross_layer_transfer():
    """Probe-direction transfer across nearby layers."""
    plt = setup_matplotlib()
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    X_mean = c["X_mean"]

    src_layers = [15, 17, 19, 21, 23]
    n = len(y)

    # Use 80/20 split for train/test
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(n, size=n // 5, replace=False)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    train_mask = ~test_mask

    matrix = np.zeros((len(src_layers), N_LAYERS))
    for si, L_src in enumerate(src_layers):
        probe = DomProbe().fit(X_mean[train_mask, L_src, :], y[train_mask])
        w = probe.direction
        for L_tgt in range(N_LAYERS):
            scores = X_mean[test_mask, L_tgt, :] @ w
            y_test = y[test_mask]
            if len(np.unique(y_test)) < 2:
                matrix[si, L_tgt] = 0.5
            else:
                matrix[si, L_tgt] = float(roc_auc_score(y_test, scores))

    result = {
        "src_layers": src_layers,
        "matrix": matrix.tolist(),
        "diagonal": {L: float(matrix[si, L]) for si, L in enumerate(src_layers)},
    }
    save_json(RESULTS_DIR / "B4_cross_layer.json", result)

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0.45, vmax=0.85)
    ax.set_yticks(range(len(src_layers)))
    ax.set_yticklabels([f"L{L}" for L in src_layers])
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([str(i) for i in range(N_LAYERS)], fontsize=7)
    ax.set_xlabel("Target layer")
    ax.set_ylabel("Source layer (trained on)")
    ax.set_title("Cross-layer probe transfer — 1.5B MATH-500")
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    save_fig(fig, "B4_cross_layer_heatmap.png")
    plt.close(fig)

    for si, L_src in enumerate(src_layers):
        neighbors = [matrix[si, L_src + d] for d in [-1, 0, 1]
                      if 0 <= L_src + d < N_LAYERS]
        print(f"  L{L_src}: self={matrix[si, L_src]:.4f}, neighbors={[f'{v:.4f}' for v in neighbors]}")


# ===== TIER C: Cross-Domain Transfer =========================================

def C1_math_to_bbh():
    """MATH-500 → BBH transfer at L19 (and L* from B1)."""
    c_math = load_cache("math500_1p5b")
    c_bbh = load_cache("bbh_1p5b")

    y_math = c_math["y"].astype(bool)
    y_bbh = c_bbh["y"].astype(bool)
    subsets = c_bbh["subsets"]

    # Determine L* from B1 if available
    b1_path = RESULTS_DIR / "B1_layer_sweep_1p5b.json"
    L_star = STEERING_LAYER
    if b1_path.exists():
        b1 = json.loads(b1_path.read_text())
        L_star = b1["peak_mean"]["layer"]

    results = {}
    for L, label in [(STEERING_LAYER, "L19"), (L_star, f"L{L_star}")]:
        X_train = c_math["X_mean"][:, L, :]
        X_test = c_bbh["X_mean"][:, L, :]

        probe = DomProbe().fit(X_train, y_math)
        scores = probe.score(X_test)

        # Overall
        ba = bootstrap_auroc(y_bbh, scores, n_resamples=1000)
        results[f"pooled_{label}"] = ba
        print(f"  MATH→BBH ({label}) pooled: AUROC={ba['point']:.4f} [{ba['bca_lo']:.4f}, {ba['bca_hi']:.4f}]")

        # Per-subset
        for subset in sorted(set(subsets)):
            mask = subsets == subset
            y_s = y_bbh[mask]
            s_s = scores[mask]
            n_pos, n_neg = y_s.sum(), (~y_s).sum()
            if n_pos < 5 or n_neg < 5:
                results[f"{subset}_{label}"] = {"skipped": True, "n_pos": int(n_pos)}
                continue
            ba_s = bootstrap_auroc(y_s, s_s, n_resamples=1000)
            short = subset.split("_")[0]
            results[f"{subset}_{label}"] = ba_s
            print(f"    {short}: AUROC={ba_s['point']:.4f}")

    save_json(RESULTS_DIR / "C1_math_to_bbh.json", results)


def C2_bbh_to_math():
    """BBH → MATH-500 reverse transfer."""
    c_math = load_cache("math500_1p5b")
    c_bbh = load_cache("bbh_1p5b")

    y_math = c_math["y"].astype(bool)
    y_bbh = c_bbh["y"].astype(bool)

    results = {}
    for L in [STEERING_LAYER]:
        X_train = c_bbh["X_mean"][:, L, :]
        X_test = c_math["X_mean"][:, L, :]

        probe = DomProbe().fit(X_train, y_bbh)
        scores = probe.score(X_test)

        ba = bootstrap_auroc(y_math, scores, n_resamples=1000)
        results[f"L{L}"] = ba
        print(f"  BBH→MATH (L{L}): AUROC={ba['point']:.4f} [{ba['bca_lo']:.4f}, {ba['bca_hi']:.4f}]")

    save_json(RESULTS_DIR / "C2_bbh_to_math.json", results)


def C3_loso_bbh():
    """Within-BBH leave-one-subset-out."""
    c = load_cache("bbh_1p5b")
    y = c["y"].astype(bool)
    subsets = c["subsets"]
    X = c["X_mean"][:, STEERING_LAYER, :]

    unique_subsets = sorted(set(subsets))
    results = {}

    for held_out in unique_subsets:
        test_mask = subsets == held_out
        train_mask = ~test_mask
        y_tr, y_te = y[train_mask], y[test_mask]
        X_tr, X_te = X[train_mask], X[test_mask]

        n_pos_tr, n_neg_tr = y_tr.sum(), (~y_tr).sum()
        n_pos_te, n_neg_te = y_te.sum(), (~y_te).sum()
        if n_pos_tr < 5 or n_neg_tr < 5 or n_pos_te < 5 or n_neg_te < 5:
            results[held_out] = {"skipped": True}
            continue

        probe = DomProbe().fit(X_tr, y_tr)
        scores = probe.score(X_te)
        ba = bootstrap_auroc(y_te, scores, n_resamples=1000)
        results[held_out] = {**ba, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
        short = held_out.split("_")[0]
        print(f"  held-out {short}: AUROC={ba['point']:.4f} [{ba['bca_lo']:.4f}, {ba['bca_hi']:.4f}]")

    save_json(RESULTS_DIR / "C3_loso_bbh.json", results)


def C4_pooled_domain():
    """Domain-pooled probe vs per-domain probes."""
    c_math = load_cache("math500_1p5b")
    c_bbh = load_cache("bbh_1p5b")

    X_math = c_math["X_mean"][:, STEERING_LAYER, :]
    y_math = c_math["y"].astype(bool)
    X_bbh = c_bbh["X_mean"][:, STEERING_LAYER, :]
    y_bbh = c_bbh["y"].astype(bool)

    X_all = np.vstack([X_math, X_bbh])
    y_all = np.concatenate([y_math, y_bbh])
    domain = np.array(["math"] * len(y_math) + ["bbh"] * len(y_bbh))

    # Pooled OOF
    scores_all = oof_dom_scores(X_all, y_all)
    auroc_math_half = float(roc_auc_score(y_all[domain == "math"], scores_all[domain == "math"]))
    auroc_bbh_half = float(roc_auc_score(y_all[domain == "bbh"], scores_all[domain == "bbh"]))

    # Within-domain OOF for comparison
    scores_math_wd = oof_dom_scores(X_math, y_math)
    auroc_math_wd = float(roc_auc_score(y_math, scores_math_wd))
    scores_bbh_wd = oof_dom_scores(X_bbh, y_bbh)
    auroc_bbh_wd = float(roc_auc_score(y_bbh, scores_bbh_wd))

    results = {
        "pooled_on_math": auroc_math_half,
        "pooled_on_bbh": auroc_bbh_half,
        "within_domain_math": auroc_math_wd,
        "within_domain_bbh": auroc_bbh_wd,
        "math_delta": auroc_math_half - auroc_math_wd,
        "bbh_delta": auroc_bbh_half - auroc_bbh_wd,
    }
    print(f"  Pooled on MATH: {auroc_math_half:.4f} (within-domain: {auroc_math_wd:.4f}, Δ={auroc_math_half - auroc_math_wd:+.4f})")
    print(f"  Pooled on BBH:  {auroc_bbh_half:.4f} (within-domain: {auroc_bbh_wd:.4f}, Δ={auroc_bbh_half - auroc_bbh_wd:+.4f})")

    save_json(RESULTS_DIR / "C4_pooled.json", results)


# ===== TIER D: Cross-Tier Transfer (1.5B ↔ 7B) ==============================

def D1_pca256():
    """PCA-256 shared subspace — refit and compare."""
    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    results = {}
    for tag, c_cache in [("1p5b", c_1p5b), ("7b", c_7b)]:
        y = c_cache["y"].astype(bool)

        # Use best layer from B1/B2 or default L19
        bx_path = RESULTS_DIR / f"B{'1' if tag == '1p5b' else '2'}_layer_sweep_{tag}.json"
        L = STEERING_LAYER
        if bx_path.exists():
            bx = json.loads(bx_path.read_text())
            L = bx["peak_mean"]["layer"]

        X_native = c_cache["X_mean"][:, L, :]

        # Native-dim AUROC
        scores_native = oof_dom_scores(X_native, y)
        auroc_native = float(roc_auc_score(y, scores_native))

        # PCA-256
        pca = PCA(n_components=256, random_state=SEED)
        X_pca = pca.fit_transform(X_native)
        scores_pca = oof_dom_scores(X_pca, y)
        auroc_pca = float(roc_auc_score(y, scores_pca))

        # Cosine of probe direction in PCA space
        probe_native = DomProbe().fit(X_native, y)
        probe_pca = DomProbe().fit(X_pca, y)

        var_explained = float(pca.explained_variance_ratio_.sum())

        results[tag] = {
            "layer": L,
            "auroc_native": auroc_native,
            "auroc_pca256": auroc_pca,
            "drop": auroc_native - auroc_pca,
            "pca_var_explained": var_explained,
        }
        print(f"  {tag} L{L}: native={auroc_native:.4f}, PCA-256={auroc_pca:.4f} (Δ={auroc_native - auroc_pca:+.4f}, var={var_explained:.3f})")

    save_json(RESULTS_DIR / "D1_pca256.json", results)
    return results


def D2_procrustes():
    """Procrustes alignment of probe directions between 1.5B and 7B."""
    from scipy.linalg import orthogonal_procrustes

    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)

    L = STEERING_LAYER
    X_1p5b = c_1p5b["X_mean"][:, L, :]
    X_7b = c_7b["X_mean"][:, L, :]

    # PCA-256 both
    pca_1p5b = PCA(n_components=256, random_state=SEED)
    pca_7b = PCA(n_components=256, random_state=SEED)
    A1 = pca_1p5b.fit_transform(StandardScaler().fit_transform(X_1p5b))
    A7 = pca_7b.fit_transform(StandardScaler().fit_transform(X_7b))

    # Orthogonal Procrustes: find R such that A1 @ R ≈ A7
    R, scale = orthogonal_procrustes(A1, A7)
    residual = np.linalg.norm(A1 @ R - A7, 'fro') / np.linalg.norm(A7, 'fro')

    # Probe directions in PCA space
    w1 = DomProbe().fit(A1, y_1p5b).direction
    w7 = DomProbe().fit(A7, y_7b).direction

    # Cosine of R*w1 with w7
    Rw1 = R @ w1
    Rw1_unit = Rw1 / (np.linalg.norm(Rw1) + 1e-12)
    cosine = float(np.dot(Rw1_unit, w7))

    results = {
        "procrustes_residual_frac": float(residual),
        "cosine_Rw1_w7": cosine,
        "cosine_w1_w7_unaligned": float(np.dot(w1, w7)),
        "scale": float(scale),
    }
    print(f"  Procrustes residual: {residual:.4f}")
    print(f"  cos(R·w_1.5B, w_7B) = {cosine:.4f}")
    print(f"  cos(w_1.5B, w_7B) unaligned = {results['cosine_w1_w7_unaligned']:.4f}")

    save_json(RESULTS_DIR / "D2_procrustes.json", results)


def D3_cross_tier():
    """Linear-projection cross-tier prediction (honest holdout)."""
    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)

    L = STEERING_LAYER
    X_1p5b = c_1p5b["X_mean"][:, L, :]
    X_7b = c_7b["X_mean"][:, L, :]

    n = len(y_1p5b)
    n_holdout = 100
    n_repeats = 20
    rng = np.random.default_rng(SEED)

    aurocs_7b_labels = []
    aurocs_1p5b_labels = []

    for rep in range(n_repeats):
        test_idx = rng.choice(n, size=n_holdout, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask

        # Fit ridge map 7B→1.5B on train
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_7b[train_mask], X_1p5b[train_mask])

        # Fit DomProbe on 1.5B train
        probe = DomProbe().fit(X_1p5b[train_mask], y_1p5b[train_mask])

        # Project 7B test through ridge → score with 1.5B probe
        X_7b_proj = ridge.predict(X_7b[test_mask])
        scores = probe.score(X_7b_proj)

        # AUROC against 7B labels
        if len(np.unique(y_7b[test_mask])) >= 2:
            aurocs_7b_labels.append(float(roc_auc_score(y_7b[test_mask], scores)))
        # AUROC against 1.5B labels
        if len(np.unique(y_1p5b[test_mask])) >= 2:
            aurocs_1p5b_labels.append(float(roc_auc_score(y_1p5b[test_mask], scores)))

    results = {
        "n_holdout": n_holdout,
        "n_repeats": n_repeats,
        "vs_7b_labels": {
            "mean": float(np.mean(aurocs_7b_labels)),
            "std": float(np.std(aurocs_7b_labels)),
            "min": float(np.min(aurocs_7b_labels)),
            "max": float(np.max(aurocs_7b_labels)),
        },
        "vs_1p5b_labels": {
            "mean": float(np.mean(aurocs_1p5b_labels)),
            "std": float(np.std(aurocs_1p5b_labels)),
            "min": float(np.min(aurocs_1p5b_labels)),
            "max": float(np.max(aurocs_1p5b_labels)),
        },
    }
    print(f"  Cross-tier (7B→1.5B probe) vs 7B labels: {np.mean(aurocs_7b_labels):.4f} ± {np.std(aurocs_7b_labels):.4f}")
    print(f"  Cross-tier (7B→1.5B probe) vs 1.5B labels: {np.mean(aurocs_1p5b_labels):.4f} ± {np.std(aurocs_1p5b_labels):.4f}")

    save_json(RESULTS_DIR / "D3_cross_tier.json", results)


# ===== TIER E: Baseline Comparisons ==========================================

def E1_logprob():
    """Logprob baseline AUROC."""
    results = {}
    for tag, cache_name in [("1p5b", "math500_1p5b"), ("7b", "math500_7b")]:
        c = load_cache(cache_name)
        y = c["y"].astype(bool)
        logprob = c["mean_logprob"]

        # Higher logprob → more confident → more likely correct
        ba = bootstrap_auroc(y, logprob)

        # Isotonic-calibrated logprob
        probs_cal, _ = oof_calibrated_1d(logprob, y)
        brier = float(brier_score_loss(y, probs_cal)) if probs_cal is not None else None

        results[tag] = {**ba, "brier_calibrated": brier}
        print(f"  {tag} logprob AUROC: {ba['point']:.4f} [{ba['bca_lo']:.4f}, {ba['bca_hi']:.4f}]")

    save_json(RESULTS_DIR / "E1_logprob.json", results)


def oof_calibrated_1d(scores: np.ndarray, y: np.ndarray,
                      n_folds: int = N_FOLDS, seed: int = SEED):
    """1D isotonic calibration on pre-computed scores."""
    from sklearn.isotonic import IsotonicRegression as IsoReg
    probs = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(scores.reshape(-1, 1), y):
        iso = IsoReg(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(scores[tr], y[tr])
        probs[te] = iso.predict(scores[te])
    return probs, scores


def E2_token_count():
    """Token-count baseline AUROC."""
    results = {}
    for tag, cache_name in [("1p5b", "math500_1p5b"), ("7b", "math500_7b")]:
        c = load_cache(cache_name)
        y = c["y"].astype(bool)
        tokens = c["n_gen_tokens"].astype(float)
        # Shorter answers → more likely correct? Use negative
        ba_neg = bootstrap_auroc(y, -tokens)
        ba_pos = bootstrap_auroc(y, tokens)
        best = ba_neg if ba_neg["point"] > ba_pos["point"] else ba_pos
        results[tag] = {**best, "sign": "negative" if ba_neg["point"] > ba_pos["point"] else "positive"}
        print(f"  {tag} token-count AUROC: {best['point']:.4f} (sign={results[tag]['sign']})")

    save_json(RESULTS_DIR / "E2_tokens.json", results)


def E3_attn_entropy():
    """Attention-entropy baseline (1.5B only)."""
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    entropy = c["d2h_attn_entropy"]  # (500, 28)

    results = {}
    for L in [STEERING_LAYER, 18, 20]:
        if L >= entropy.shape[1]:
            continue
        e_L = entropy[:, L]
        if np.all(e_L == 0):
            results[f"L{L}"] = {"skipped": True, "reason": "all-zeros"}
            continue
        ba_pos = bootstrap_auroc(y, e_L)
        ba_neg = bootstrap_auroc(y, -e_L)
        best = ba_pos if ba_pos["point"] > ba_neg["point"] else ba_neg
        sign = "positive" if ba_pos["point"] > ba_neg["point"] else "negative"
        results[f"L{L}"] = {**best, "sign": sign}
        print(f"  L{L} attn-entropy AUROC: {best['point']:.4f} (sign={sign})")

    save_json(RESULTS_DIR / "E3_attn_entropy.json", results)


def E4_combined():
    """Combined logistic regression of all signals."""
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)

    # Load OOF DomProbe scores from A2
    oof_path = CACHE_DIR / "oof_scores_1p5b.npz"
    if not oof_path.exists():
        print("  WARNING: oof_scores_1p5b.npz not found, recomputing")
        X_L19 = c["X_prefill"][:, STEERING_LAYER, :]
        dom_scores = oof_dom_scores(X_L19, y)
    else:
        oof_data = np.load(oof_path)
        dom_scores = oof_data["raw_scores"]

    logprob = c["mean_logprob"]
    tokens = c["n_gen_tokens"].astype(float)
    entropy_L19 = c["d2h_attn_entropy"][:, STEERING_LAYER]

    feature_names = ["dom_score", "mean_logprob", "n_gen_tokens", "attn_entropy_L19"]
    features = np.column_stack([dom_scores, logprob, tokens, entropy_L19])

    # 5-fold OOF logistic
    scaler = StandardScaler()
    scores_combined = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(features, y):
        X_tr = scaler.fit_transform(features[tr])
        X_te = scaler.transform(features[te])
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr.fit(X_tr, y[tr])
        scores_combined[te] = lr.predict_proba(X_te)[:, 1]

    auroc_combined = float(roc_auc_score(y, scores_combined))

    # Per-feature ablation
    ablations = {}
    for feat_idx, fname in enumerate(feature_names):
        feat_1d = features[:, feat_idx]
        ba = bootstrap_auroc(y, feat_1d)
        ablations[fname] = ba["point"]

    # Fit on all data for coefficients
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_full.fit(scaler.fit_transform(features), y)
    coefs = {fname: float(lr_full.coef_[0, i]) for i, fname in enumerate(feature_names)}

    dom_only = float(roc_auc_score(y, dom_scores))
    results = {
        "auroc_combined": auroc_combined,
        "auroc_dom_only": dom_only,
        "delta": auroc_combined - dom_only,
        "coefs": coefs,
        "per_feature_auroc": ablations,
    }
    print(f"  Combined AUROC: {auroc_combined:.4f} (DoM-only: {dom_only:.4f}, Δ={auroc_combined - dom_only:+.4f})")
    print(f"  Coefficients: {coefs}")

    save_json(RESULTS_DIR / "E4_combined.json", results)


# ===== TIER F: Calibration & Operating-Point =================================

def F1_risk_coverage():
    """Risk-coverage curve with bootstrap bands."""
    plt = setup_matplotlib()

    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    X = c["X_prefill"][:, STEERING_LAYER, :]
    scores = oof_dom_scores(X, y)
    unconditional_acc = y.mean()

    cov_grid = np.arange(0.05, 1.001, 0.05)
    curve = risk_coverage_curve(y, scores, cov_grid)

    # Bootstrap bands
    n_boot = 2000
    rng = np.random.default_rng(SEED)
    boot_accs = np.zeros((n_boot, len(cov_grid)))

    for b in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        y_b = y[idx]
        s_b = scores[idx]
        order = np.argsort(-s_b)
        y_sorted = y_b[order]
        for ci, c_val in enumerate(cov_grid):
            k = max(1, int(round(c_val * len(y_b))))
            boot_accs[b, ci] = y_sorted[:k].mean()

    accs = np.array([r["accuracy"] for r in curve])
    lo = np.percentile(boot_accs, 2.5, axis=0)
    hi = np.percentile(boot_accs, 97.5, axis=0)

    lift_50 = None
    for r in curve:
        if abs(r["coverage"] - 0.5) < 0.01:
            lift_50 = r["accuracy"] - float(unconditional_acc)

    result = {
        "curve": curve,
        "unconditional_accuracy": float(unconditional_acc),
        "lift_at_50pct_coverage": lift_50,
        "boot_lo": lo.tolist(),
        "boot_hi": hi.tolist(),
    }

    if lift_50 is not None:
        lo_50_idx = np.argmin(np.abs(cov_grid - 0.5))
        print(f"  Lift at 50% coverage: +{lift_50:.4f} (acc={accs[lo_50_idx]:.4f} vs {unconditional_acc:.4f})")
        print(f"  95% CI on acc@50%: [{lo[lo_50_idx]:.4f}, {hi[lo_50_idx]:.4f}]")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(cov_grid, accs, "-o", ms=4, color="steelblue", label="DomProbe gate")
    ax.fill_between(cov_grid, lo, hi, alpha=0.15, color="steelblue")
    ax.axhline(unconditional_acc, color="gray", ls="--", label=f"Unconditional ({unconditional_acc:.3f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("Risk-coverage — 1.5B MATH-500 (prefill L19 DoM)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "F1_risk_coverage.png")
    plt.close(fig)

    save_json(RESULTS_DIR / "F1_rc_curve.json", result)
    return result


def F2_match_7b():
    """'Match-7B-accuracy at X% coverage' reframing."""
    c_7b = load_cache("math500_7b")
    acc_7b = float(c_7b["y"].astype(bool).mean())

    f1_path = RESULTS_DIR / "F1_rc_curve.json"
    if not f1_path.exists():
        print("  WARNING: F1 not run yet, running now")
        F1_risk_coverage()
    f1 = json.loads(f1_path.read_text())

    curve = f1["curve"]
    c_star = None
    for row in curve:
        if row["accuracy"] >= acc_7b:
            c_star = row["coverage"]
            break

    c_95 = None
    for row in curve:
        if row["accuracy"] >= 0.95 * acc_7b:
            c_95 = row["coverage"]
            break

    results = {
        "accuracy_7b": acc_7b,
        "c_star_match_7b": c_star,
        "c_95_of_7b": c_95,
    }
    if c_star:
        print(f"  Matches 7B accuracy ({acc_7b:.3f}) at coverage={c_star:.2f}")
    else:
        print(f"  Cannot match 7B accuracy ({acc_7b:.3f}) at any coverage point")
    if c_95:
        print(f"  95% of 7B accuracy at coverage={c_95:.2f}")

    save_json(RESULTS_DIR / "F2_match_7b.json", results)


def F3_aurc():
    """AURC and E-AURC."""
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    X = c["X_prefill"][:, STEERING_LAYER, :]
    scores = oof_dom_scores(X, y)

    aurc_val = aurc(y, scores)
    oracle_val = oracle_aurc(y)
    eaurc = aurc_val - oracle_val

    # Logprob baseline for comparison
    logprob = c["mean_logprob"]
    aurc_logprob = aurc(y, logprob)

    results = {
        "aurc_domprobe": aurc_val,
        "aurc_oracle": oracle_val,
        "eaurc": eaurc,
        "aurc_logprob": aurc_logprob,
    }
    print(f"  AURC (DomProbe): {aurc_val:.4f}")
    print(f"  AURC (Oracle):   {oracle_val:.4f}")
    print(f"  E-AURC:          {eaurc:.4f}")
    print(f"  AURC (Logprob):  {aurc_logprob:.4f}")

    save_json(RESULTS_DIR / "F3_aurc.json", results)


# ===== TIER G: Cascade Simulation ============================================

def G1_cascade():
    """Two-tier 1.5B → 7B cascade simulation."""
    plt = setup_matplotlib()

    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)

    X = c_1p5b["X_prefill"][:, STEERING_LAYER, :]
    gate_scores = oof_dom_scores(X, y_1p5b)

    cost_1p5b = 1.0
    cost_7b = 5.0
    n = len(y_1p5b)

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []

    for tau_pct in thresholds:
        tau = np.percentile(gate_scores, (1 - tau_pct) * 100)
        keep_1p5b = gate_scores >= tau
        escalate = ~keep_1p5b

        n_kept = keep_1p5b.sum()
        n_esc = escalate.sum()

        correct_kept = y_1p5b[keep_1p5b].sum() if n_kept > 0 else 0
        correct_esc = y_7b[escalate].sum() if n_esc > 0 else 0
        accuracy = (correct_kept + correct_esc) / n

        cost = (n_kept * cost_1p5b + n_esc * cost_7b) / n
        cost_rel = cost / cost_7b  # relative to all-7B

        rows.append({
            "tau_pct": float(tau_pct),
            "accuracy": float(accuracy),
            "cost_per_problem": float(cost),
            "cost_relative_to_all_7b": float(cost_rel),
            "n_kept_1p5b": int(n_kept),
            "n_escalated_7b": int(n_esc),
            "pct_kept": float(n_kept / n),
        })

    # Baselines
    acc_all_1p5b = float(y_1p5b.mean())
    acc_all_7b = float(y_7b.mean())

    # Find threshold retaining ≥95% of 7B accuracy at min cost
    target = 0.95 * acc_all_7b
    valid = [r for r in rows if r["accuracy"] >= target]
    best = min(valid, key=lambda r: r["cost_per_problem"]) if valid else None

    results = {
        "sweep": rows,
        "acc_all_1p5b": acc_all_1p5b,
        "acc_all_7b": acc_all_7b,
        "cost_weights": {"1p5b": cost_1p5b, "7b": cost_7b},
        "best_95pct_7b": best,
    }

    if best:
        print(f"  Best @ ≥95% 7B acc: τ_pct={best['tau_pct']:.2f}, acc={best['accuracy']:.4f}, cost={best['cost_relative_to_all_7b']:.3f}× all-7B")

    # Pareto plot
    fig, ax = plt.subplots()
    costs = [r["cost_relative_to_all_7b"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    ax.plot(costs, accs, "-o", ms=5, color="steelblue", label="Cascade")
    ax.axhline(acc_all_7b, color="coral", ls="--", label=f"All-7B ({acc_all_7b:.3f})")
    ax.axhline(acc_all_1p5b, color="gray", ls=":", label=f"All-1.5B ({acc_all_1p5b:.3f})")
    ax.axhline(target, color="coral", ls=":", alpha=0.5, label=f"95% of 7B ({target:.3f})")
    ax.set_xlabel("Cost (relative to all-7B)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Two-tier cascade — 1.5B → 7B (MATH-500)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "G1_pareto.png")
    plt.close(fig)

    save_json(RESULTS_DIR / "G1_cascade.json", results)


# ===== MAIN ==================================================================

def main():
    ensure_dirs()

    experiments = [
        ("E1", "Logprob baseline", E1_logprob),
        ("E2", "Token-count baseline", E2_token_count),
        ("A2", "Bootstrap CI on AUROC at L19", A2_bootstrap_ci),
        ("A3", "Multi-seed DomProbe variance", A3_seed_variance),
        ("A4", "Calibration analysis", A4_calibration),
        ("A5", "Per-subject AUROC", A5_per_subject),
        ("B1", "Layer sweep 1.5B MATH-500", B1_layer_sweep_1p5b),
        ("B2", "Layer sweep 7B MATH-500", B2_layer_sweep_7b),
        ("B3", "Layer sweep BBH (1.5B)", B3_layer_sweep_bbh),
        ("B4", "Cross-layer probe transfer", B4_cross_layer_transfer),
        ("E3", "Attention-entropy baseline", E3_attn_entropy),
        ("E4", "Combined logistic regression", E4_combined),
        ("C1", "MATH→BBH transfer", C1_math_to_bbh),
        ("C2", "BBH→MATH reverse transfer", C2_bbh_to_math),
        ("C3", "Within-BBH LOSO", C3_loso_bbh),
        ("C4", "Pooled-domain probe", C4_pooled_domain),
        ("D1", "PCA-256 refit", D1_pca256),
        ("D2", "Procrustes alignment", D2_procrustes),
        ("D3", "Cross-tier projection", D3_cross_tier),
        ("F1", "Risk-coverage curve", F1_risk_coverage),
        ("F2", "Match-7B at X% coverage", F2_match_7b),
        ("F3", "AURC / E-AURC", F3_aurc),
        ("G1", "Two-tier cascade", G1_cascade),
    ]

    # Allow running a subset: python run_experiments.py A2 B1 E1
    if len(sys.argv) > 1:
        selected = set(sys.argv[1:])
        experiments = [(tid, desc, fn) for tid, desc, fn in experiments if tid in selected]

    for tid, desc, fn in experiments:
        with timer(f"{tid}: {desc}"):
            try:
                fn()
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("All experiments complete. Results in:")
    print(f"  {RESULTS_DIR}/")
    print(f"  {FIGS_DIR}/")


if __name__ == "__main__":
    main()
