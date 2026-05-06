#!/usr/bin/env python3
"""Diagnostic questions Q1–Q8 on existing nocompute caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lib import (
    CACHE_DIR, FIGS_DIR, N_FOLDS, N_LAYERS, RESULTS_DIR, SEED, STEERING_LAYER,
    DomProbe, bootstrap_auroc, ece_score, ensure_dirs, load_cache,
    oof_dom_calibrated, oof_dom_scores, save_fig, save_json, setup_matplotlib,
    timer,
)


# ═══════════════════════════════════════════════════════════════
# Q1. MATH→BBH sign flip check
# ═══════════════════════════════════════════════════════════════

def Q1_sign_flip():
    c_math = load_cache("math500_1p5b")
    c_bbh = load_cache("bbh_1p5b")
    y_math = c_math["y"].astype(bool)
    y_bbh = c_bbh["y"].astype(bool)
    subsets = c_bbh["subsets"]

    X_math = c_math["X_mean"][:, STEERING_LAYER, :]
    X_bbh = c_bbh["X_mean"][:, STEERING_LAYER, :]

    probe = DomProbe().fit(X_math, y_math)
    scores_bbh = probe.score(X_bbh)

    auroc_pos = float(roc_auc_score(y_bbh, scores_bbh))
    auroc_neg = float(roc_auc_score(y_bbh, -scores_bbh))
    rho, pval = spearmanr(scores_bbh, y_bbh.astype(float))

    results = {
        "auroc_original": auroc_pos,
        "auroc_negated": auroc_neg,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "diagnosis": "sign_flip" if auroc_neg > 0.55 else "genuine_anti_correlation",
    }

    print(f"  AUROC (original):  {auroc_pos:.4f}")
    print(f"  AUROC (negated):   {auroc_neg:.4f}")
    print(f"  Spearman ρ:        {rho:+.4f} (p={pval:.4f})")
    print(f"  Diagnosis:         {results['diagnosis']}")

    # Per-subset breakdown
    for subset in sorted(set(subsets)):
        mask = subsets == subset
        y_s = y_bbh[mask]
        s_s = scores_bbh[mask]
        n_pos, n_neg = y_s.sum(), (~y_s).sum()
        if n_pos < 3 or n_neg < 3:
            continue
        a_pos = float(roc_auc_score(y_s, s_s))
        a_neg = float(roc_auc_score(y_s, -s_s))
        rho_s, _ = spearmanr(s_s, y_s.astype(float))
        short = subset.split("_")[0]
        print(f"    {short:12s}: AUROC={a_pos:.3f} / negated={a_neg:.3f}, ρ={rho_s:+.3f}")
        results[subset] = {"auroc": a_pos, "auroc_neg": a_neg, "rho": float(rho_s)}

    # Also check prefill (not just mean-pool)
    X_math_pf = c_math["X_prefill"][:, STEERING_LAYER, :]
    X_bbh_pf = c_bbh["X_prefill"][:, STEERING_LAYER, :]
    probe_pf = DomProbe().fit(X_math_pf, y_math)
    scores_pf = probe_pf.score(X_bbh_pf)
    auroc_pf = float(roc_auc_score(y_bbh, scores_pf))
    auroc_pf_neg = float(roc_auc_score(y_bbh, -scores_pf))
    rho_pf, _ = spearmanr(scores_pf, y_bbh.astype(float))
    results["prefill"] = {"auroc": auroc_pf, "auroc_neg": auroc_pf_neg, "rho": float(rho_pf)}
    print(f"  Prefill:           AUROC={auroc_pf:.4f} / negated={auroc_pf_neg:.4f}, ρ={rho_pf:+.4f}")

    save_json(RESULTS_DIR / "Q1_sign_flip.json", results)


# ═══════════════════════════════════════════════════════════════
# Q2. Token count AUROC on BBH
# ═══════════════════════════════════════════════════════════════

def Q2_token_count_bbh():
    c_bbh = load_cache("bbh_1p5b")
    y = c_bbh["y"].astype(bool)
    subsets = c_bbh["subsets"]

    # BBH cache has n_gen_tokens=0 (placeholder). Derive from states shape.
    # Actually, we need to get token counts from the NPZ files or compute
    # from the cached X_mean vs X_last difference. Let's get them from raw NPZs.
    from lib import DATA_BBH, BBH_SUBSETS
    tokens = []
    for subset in BBH_SUBSETS:
        subset_dir = DATA_BBH / subset
        n = len(list(subset_dir.glob("problem_*.npz")))
        for i in range(n):
            fp = subset_dir / f"problem_{i:03d}.npz"
            with np.load(fp, allow_pickle=True) as d:
                tokens.append(d["states"].shape[1])
    tokens = np.array(tokens, dtype=float)

    results = {}

    # Pooled
    ba_neg = bootstrap_auroc(y, -tokens, n_resamples=1000)
    ba_pos = bootstrap_auroc(y, tokens, n_resamples=1000)
    best = ba_neg if ba_neg["point"] > ba_pos["point"] else ba_pos
    sign = "negative" if ba_neg["point"] > ba_pos["point"] else "positive"
    results["pooled"] = {**best, "sign": sign, "n": len(y)}
    print(f"  BBH pooled: token-count AUROC={best['point']:.4f} (sign={sign}) [{best['bca_lo']:.4f}, {best['bca_hi']:.4f}]")

    # Per-subset
    for subset in sorted(set(subsets)):
        mask = subsets == subset
        y_s = y[mask]
        t_s = tokens[mask]
        n_pos, n_neg = y_s.sum(), (~y_s).sum()
        if n_pos < 3 or n_neg < 3:
            short = subset.split("_")[0]
            results[subset] = {"skipped": True, "n_pos": int(n_pos)}
            print(f"    {short}: skipped (n_pos={n_pos})")
            continue
        ba_n = bootstrap_auroc(y_s, -t_s, n_resamples=500)
        ba_p = bootstrap_auroc(y_s, t_s, n_resamples=500)
        best_s = ba_n if ba_n["point"] > ba_p["point"] else ba_p
        sign_s = "negative" if ba_n["point"] > ba_p["point"] else "positive"
        short = subset.split("_")[0]
        results[subset] = {**best_s, "sign": sign_s}
        print(f"    {short:12s}: AUROC={best_s['point']:.4f} (sign={sign_s})")

    # Mean token counts per class
    results["mean_tokens_correct"] = float(tokens[y].mean())
    results["mean_tokens_incorrect"] = float(tokens[~y].mean())
    print(f"  Mean tokens: correct={tokens[y].mean():.0f}, incorrect={tokens[~y].mean():.0f}")

    # Compare to MATH-500
    c_math = load_cache("math500_1p5b")
    y_math = c_math["y"].astype(bool)
    t_math = c_math["n_gen_tokens"].astype(float)
    results["math_mean_correct"] = float(t_math[y_math].mean())
    results["math_mean_incorrect"] = float(t_math[~y_math].mean())
    print(f"  MATH tokens: correct={t_math[y_math].mean():.0f}, incorrect={t_math[~y_math].mean():.0f}")

    save_json(RESULTS_DIR / "Q2_token_count_bbh.json", results)


# ═══════════════════════════════════════════════════════════════
# Q3. Combined logistic on BBH
# ═══════════════════════════════════════════════════════════════

def Q3_combined_bbh():
    c_bbh = load_cache("bbh_1p5b")
    y = c_bbh["y"].astype(bool)
    subsets = c_bbh["subsets"]

    X_mean_L19 = c_bbh["X_mean"][:, STEERING_LAYER, :]
    logprob = c_bbh["mean_logprob"]
    entropy_L19 = c_bbh["d2h_attn_entropy"][:, STEERING_LAYER]

    # Get token counts (same approach as Q2)
    from lib import DATA_BBH, BBH_SUBSETS
    tokens = []
    for subset in BBH_SUBSETS:
        subset_dir = DATA_BBH / subset
        n = len(list(subset_dir.glob("problem_*.npz")))
        for i in range(n):
            fp = subset_dir / f"problem_{i:03d}.npz"
            with np.load(fp, allow_pickle=True) as d:
                tokens.append(d["states"].shape[1])
    tokens = np.array(tokens, dtype=float)

    # OOF DomProbe scores on BBH (in-domain)
    dom_scores = oof_dom_scores(X_mean_L19, y)
    auroc_dom_only = float(roc_auc_score(y, dom_scores))

    feature_names = ["dom_score", "mean_logprob", "n_gen_tokens", "attn_entropy_L19"]
    features = np.column_stack([dom_scores, logprob, tokens, entropy_L19])

    # OOF combined logistic
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

    # Without DomProbe
    features_no_dom = np.column_stack([logprob, tokens, entropy_L19])
    scores_no_dom = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(features_no_dom, y):
        X_tr = scaler.fit_transform(features_no_dom[tr])
        X_te = scaler.transform(features_no_dom[te])
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr.fit(X_tr, y[tr])
        scores_no_dom[te] = lr.predict_proba(X_te)[:, 1]
    auroc_no_dom = float(roc_auc_score(y, scores_no_dom))

    # Full-data fit for coefficients
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_full.fit(scaler.fit_transform(features), y)
    coefs = {fname: float(lr_full.coef_[0, i]) for i, fname in enumerate(feature_names)}

    results = {
        "auroc_dom_only": auroc_dom_only,
        "auroc_combined": auroc_combined,
        "auroc_no_dom": auroc_no_dom,
        "domprobe_delta": auroc_combined - auroc_no_dom,
        "coefs": coefs,
    }
    print(f"  BBH DomProbe-only:  {auroc_dom_only:.4f}")
    print(f"  BBH combined:       {auroc_combined:.4f}")
    print(f"  BBH w/o DomProbe:   {auroc_no_dom:.4f}")
    print(f"  DomProbe Δ:         {auroc_combined - auroc_no_dom:+.4f}")
    print(f"  Coefficients:       {coefs}")

    save_json(RESULTS_DIR / "Q3_combined_bbh.json", results)


# ═══════════════════════════════════════════════════════════════
# Q4. 1.5B ECE calibration deep dive
# ═══════════════════════════════════════════════════════════════

def Q4_calibration_deep():
    plt = setup_matplotlib()
    c = load_cache("math500_1p5b")
    X = c["X_prefill"][:, STEERING_LAYER, :]
    y = c["y"].astype(bool)

    cal_probs, raw_scores = oof_dom_calibrated(X, y)

    # Platt scaling alternative
    platt_probs = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        probe = DomProbe().fit(X[tr], y[tr])
        raw_tr = probe.score(X[tr])
        raw_te = probe.score(X[te])
        lr = LogisticRegression(max_iter=1000)
        lr.fit(raw_tr.reshape(-1, 1), y[tr])
        platt_probs[te] = lr.predict_proba(raw_te.reshape(-1, 1))[:, 1]

    # Sigmoid-normalized raw for pre-isotonic
    from scipy.special import expit
    raw_probs = expit(raw_scores / (raw_scores.std() + 1e-12))

    methods = {
        "pre_isotonic": raw_probs,
        "post_isotonic": cal_probs,
        "platt": platt_probs,
    }

    results = {}
    for name, probs in methods.items():
        results[name] = {
            "ece_uniform_15": ece_score(y, probs, 15, "uniform"),
            "ece_quantile_15": ece_score(y, probs, 15, "quantile"),
            "brier": float(brier_score_loss(y, probs)),
            "auroc": float(roc_auc_score(y, probs)),
        }
        print(f"  {name:16s}: ECE(ew)={results[name]['ece_uniform_15']:.4f}, ECE(em)={results[name]['ece_quantile_15']:.4f}, Brier={results[name]['brier']:.4f}")

    # Where is the miscalibration? Bin analysis for isotonic
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, probs) in zip(axes, methods.items()):
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "-o", color="steelblue", label="Observed")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5)

        # Annotate each bin with count and gap
        n_bins = 10
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(probs, quantiles)
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1] + (1e-8 if i == n_bins - 1 else 0))
            if not mask.any():
                continue
            acc = y[mask].mean()
            conf = probs[mask].mean()
            gap = abs(acc - conf)
            if gap > 0.05:
                ax.annotate(f"n={mask.sum()}\ngap={gap:.2f}",
                           (conf, acc), fontsize=7, ha="center")

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"{name}\nECE(em)={results[name]['ece_quantile_15']:.3f}")
        ax.legend(fontsize=9)

        # Inset histogram
        ax_h = ax.inset_axes([0.55, 0.05, 0.4, 0.22])
        ax_h.hist(probs, bins=20, alpha=0.7, color="steelblue")
        ax_h.set_xlabel("P(correct)", fontsize=7)
        ax_h.tick_params(labelsize=6)

    fig.tight_layout()
    save_fig(fig, "Q4_calibration_deep.png")
    plt.close(fig)

    # Identify which region drives miscalibration
    for name, probs in methods.items():
        low_mask = probs < 0.3
        mid_mask = (probs >= 0.3) & (probs <= 0.7)
        high_mask = probs > 0.7
        for region, mask in [("low (<0.3)", low_mask), ("mid (0.3-0.7)", mid_mask), ("high (>0.7)", high_mask)]:
            if mask.sum() < 10:
                continue
            acc = y[mask].mean()
            conf = probs[mask].mean()
            gap = abs(acc - conf)
            results[f"{name}_{region}"] = {"n": int(mask.sum()), "acc": float(acc), "conf": float(conf), "gap": float(gap)}
            if gap > 0.04:
                print(f"    {name} {region}: n={mask.sum()}, acc={acc:.3f}, conf={conf:.3f}, gap={gap:.3f}")

    save_json(RESULTS_DIR / "Q4_calibration_deep.json", results)


# ═══════════════════════════════════════════════════════════════
# Q5. 7B per-subject AUROC + accuracy pattern
# ═══════════════════════════════════════════════════════════════

def Q5_per_subject_7b():
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
            acc = float(y[mask].mean())
            if n_pos < 5 or n_neg < 5:
                rows.append({"subject": subj, "n": int(n_s), "accuracy": acc, "auroc": None})
                continue
            ba = bootstrap_auroc(y[mask], scores[mask], n_resamples=1000)
            rows.append({"subject": subj, "n": int(n_s), "accuracy": acc, **ba})

        results[tag] = rows

    # AUROC vs base accuracy scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for tag, marker, color in [("1p5b", "o", "steelblue"), ("7b", "s", "coral")]:
        rows = [r for r in results[tag] if r.get("point") is not None]
        accs = [r["accuracy"] for r in rows]
        aurocs = [r["point"] for r in rows]
        labels = [r["subject"] for r in rows]
        ax.scatter(accs, aurocs, marker=marker, s=60, color=color, label=tag, zorder=5)
        for a, au, lab in zip(accs, aurocs, labels):
            ax.annotate(lab[:6], (a, au), fontsize=7, ha="center", va="bottom",
                       xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Subject accuracy")
    ax.set_ylabel("AUROC")
    ax.set_title("Gate AUROC vs base accuracy per subject")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "Q5_auroc_vs_accuracy.png")
    plt.close(fig)

    # Print table
    print(f"  {'Subject':30s} {'1.5B acc':>8s} {'1.5B AUC':>8s} {'7B acc':>8s} {'7B AUC':>8s}")
    for r1, r7 in zip(results["1p5b"], results["7b"]):
        a1 = f"{r1.get('point', 0):.3f}" if r1.get("point") else "  n/a"
        a7 = f"{r7.get('point', 0):.3f}" if r7.get("point") else "  n/a"
        print(f"  {r1['subject']:30s} {r1['accuracy']:8.3f} {a1:>8s} {r7['accuracy']:8.3f} {a7:>8s}")

    save_json(RESULTS_DIR / "Q5_per_subject_7b.json", results)


# ═══════════════════════════════════════════════════════════════
# Q6. Cascade bottleneck — oracle upper bound
# ═══════════════════════════════════════════════════════════════

def Q6_cascade_oracle():
    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")
    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)

    X = c_1p5b["X_prefill"][:, STEERING_LAYER, :]
    gate_scores = oof_dom_scores(X, y_1p5b)

    cost_1p5b = 1.0
    cost_7b = 5.0
    n = len(y_1p5b)
    acc_all_7b = float(y_7b.mean())

    def cascade_sweep(scores, label):
        """Sweep threshold, report accuracy and cost at each."""
        rows = []
        thresholds = np.arange(0.01, 1.0, 0.01)
        for tau_pct in thresholds:
            tau = np.percentile(scores, (1 - tau_pct) * 100)
            keep = scores >= tau
            esc = ~keep
            correct = y_1p5b[keep].sum() + y_7b[esc].sum()
            accuracy = correct / n
            cost = (keep.sum() * cost_1p5b + esc.sum() * cost_7b) / n
            cost_rel = cost / cost_7b
            rows.append({
                "tau_pct": float(tau_pct),
                "accuracy": float(accuracy),
                "cost_rel": float(cost_rel),
                "pct_kept": float(keep.sum() / n),
            })
        return rows

    # Real gate
    real = cascade_sweep(gate_scores, "real")

    # Oracle gate: perfect knowledge of 1.5B correctness
    oracle_scores = y_1p5b.astype(float) + np.random.default_rng(SEED).uniform(0, 0.01, n)
    oracle = cascade_sweep(oracle_scores, "oracle")

    results = {"acc_all_7b": acc_all_7b}

    for threshold_pct, label in [(0.90, "90%"), (0.95, "95%"), (0.98, "98%")]:
        target = threshold_pct * acc_all_7b

        # Find best operating point for real gate
        valid_real = [r for r in real if r["accuracy"] >= target]
        best_real = min(valid_real, key=lambda r: r["cost_rel"]) if valid_real else None

        # Oracle
        valid_oracle = [r for r in oracle if r["accuracy"] >= target]
        best_oracle = min(valid_oracle, key=lambda r: r["cost_rel"]) if valid_oracle else None

        results[f"target_{label}"] = {
            "target_acc": target,
            "real": best_real,
            "oracle": best_oracle,
        }

        real_cost = f"{best_real['cost_rel']:.3f}" if best_real else "impossible"
        oracle_cost = f"{best_oracle['cost_rel']:.3f}" if best_oracle else "impossible"
        real_kept = f"{best_real['pct_kept']*100:.0f}%" if best_real else "n/a"
        oracle_kept = f"{best_oracle['pct_kept']*100:.0f}%" if best_oracle else "n/a"
        print(f"  ≥{label} 7B acc ({target:.3f}):")
        print(f"    Real gate:   cost={real_cost}× all-7B, kept={real_kept}")
        print(f"    Oracle:      cost={oracle_cost}× all-7B, kept={oracle_kept}")

        if best_real and best_oracle:
            gap = best_real["cost_rel"] - best_oracle["cost_rel"]
            print(f"    Gate overhead: {gap:+.3f}× (oracle saves {(1-best_oracle['cost_rel'])*100:.0f}%, gate saves {(1-best_real['cost_rel'])*100:.0f}%)")

    # Diagnosis
    oracle_95 = results["target_95%"]["oracle"]
    real_95 = results["target_95%"]["real"]
    if oracle_95 and real_95:
        oracle_saving = 1 - oracle_95["cost_rel"]
        real_saving = 1 - real_95["cost_rel"]
        if oracle_saving < 0.20:
            results["diagnosis"] = "accuracy_gap_is_bottleneck"
            print(f"\n  DIAGNOSIS: Accuracy gap is the bottleneck (oracle max saving={oracle_saving:.0%})")
        elif real_saving / oracle_saving < 0.3:
            results["diagnosis"] = "gate_is_bottleneck"
            print(f"\n  DIAGNOSIS: Gate is the bottleneck (capturing {real_saving/oracle_saving:.0%} of oracle potential)")
        else:
            results["diagnosis"] = "mixed"
            print(f"\n  DIAGNOSIS: Mixed ({real_saving/oracle_saving:.0%} of oracle potential)")

    save_json(RESULTS_DIR / "Q6_cascade_oracle.json", results)


# ═══════════════════════════════════════════════════════════════
# Q7. Token count vs DomProbe disagreement quadrants
# ═══════════════════════════════════════════════════════════════

def Q7_disagreement():
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    X = c["X_prefill"][:, STEERING_LAYER, :]
    tokens = c["n_gen_tokens"].astype(float)

    dom_scores = oof_dom_scores(X, y)
    # For token count: shorter = more confident = higher "correctness score"
    tc_scores = -tokens

    # Median split
    dom_med = np.median(dom_scores)
    tc_med = np.median(tc_scores)

    dom_high = dom_scores >= dom_med
    tc_high = tc_scores >= tc_med

    quadrants = {
        "both_high":     dom_high & tc_high,
        "dom_high_tc_low": dom_high & ~tc_high,
        "dom_low_tc_high": ~dom_high & tc_high,
        "both_low":      ~dom_high & ~tc_high,
    }

    results = {}
    print(f"  {'Quadrant':25s} {'n':>5s} {'acc':>7s} {'dom_better':>10s}")
    for name, mask in quadrants.items():
        n_q = mask.sum()
        acc_q = float(y[mask].mean())
        results[name] = {"n": int(n_q), "accuracy": acc_q}
        print(f"  {name:25s} {n_q:5d} {acc_q:7.3f}")

    # On disagreement quadrants: which signal predicts better?
    for q_name in ["dom_high_tc_low", "dom_low_tc_high"]:
        mask = quadrants[q_name]
        y_q = y[mask]
        if mask.sum() < 20 or y_q.sum() < 5 or (~y_q).sum() < 5:
            continue
        # Within this quadrant, does higher dom_score still predict?
        a_dom = float(roc_auc_score(y_q, dom_scores[mask]))
        a_tc = float(roc_auc_score(y_q, tc_scores[mask]))
        results[f"{q_name}_within_dom_auroc"] = a_dom
        results[f"{q_name}_within_tc_auroc"] = a_tc
        winner = "DomProbe" if a_dom > a_tc else "TokenCount"
        print(f"    Within {q_name}: DomProbe AUROC={a_dom:.3f}, TokenCount AUROC={a_tc:.3f} → {winner}")

    # Overall: what fraction of DomProbe's "value" comes from agreement with TC?
    agree = (dom_high == tc_high)
    results["agreement_rate"] = float(agree.mean())
    results["accuracy_agree"] = float(y[agree].mean())
    results["accuracy_disagree"] = float(y[~agree].mean())
    print(f"\n  Agreement rate: {agree.mean():.1%}")
    print(f"  Accuracy on agree: {y[agree].mean():.3f}")
    print(f"  Accuracy on disagree: {y[~agree].mean():.3f}")

    # Correlation between the two scores
    from scipy.stats import pearsonr
    r, p = pearsonr(dom_scores, tc_scores)
    results["pearson_r"] = float(r)
    results["pearson_p"] = float(p)
    print(f"  Pearson(dom, tc): r={r:.3f} (p={p:.4f})")

    save_json(RESULTS_DIR / "Q7_disagreement.json", results)


# ═══════════════════════════════════════════════════════════════
# Q8. Mean-pool vs token count R²
# ═══════════════════════════════════════════════════════════════

def Q8_meanpool_length():
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    X_mean = c["X_mean"][:, STEERING_LAYER, :]
    X_prefill = c["X_prefill"][:, STEERING_LAYER, :]
    tokens = c["n_gen_tokens"].astype(float)

    # Mean-pool OOF scores
    dom_mean = oof_dom_scores(X_mean, y)
    dom_prefill = oof_dom_scores(X_prefill, y)

    # R² of mean-pool score vs token count
    from scipy.stats import pearsonr
    r_mean, _ = pearsonr(dom_mean, -tokens)
    r_prefill, _ = pearsonr(dom_prefill, -tokens)

    r2_mean = r_mean ** 2
    r2_prefill = r_prefill ** 2

    results = {
        "r2_meanpool_vs_tokencount": float(r2_mean),
        "r2_prefill_vs_tokencount": float(r2_prefill),
        "r_meanpool": float(r_mean),
        "r_prefill": float(r_prefill),
        "auroc_meanpool": float(roc_auc_score(y, dom_mean)),
        "auroc_prefill": float(roc_auc_score(y, dom_prefill)),
        "auroc_tokencount": float(roc_auc_score(y, -tokens)),
    }

    print(f"  Mean-pool DoM vs −token_count: R²={r2_mean:.3f} (r={r_mean:.3f})")
    print(f"  Prefill DoM vs −token_count:   R²={r2_prefill:.3f} (r={r_prefill:.3f})")
    print(f"  AUROC: mean-pool={results['auroc_meanpool']:.4f}, prefill={results['auroc_prefill']:.4f}, tokens={results['auroc_tokencount']:.4f}")

    if r2_mean > 0.5:
        results["diagnosis"] = "meanpool_mostly_length"
        print(f"  DIAGNOSIS: Mean-pool gain is mostly length information (R²={r2_mean:.3f} > 0.5)")
    elif r2_mean < 0.2:
        results["diagnosis"] = "meanpool_genuinely_different"
        print(f"  DIAGNOSIS: Mean-pool captures genuinely different signal (R²={r2_mean:.3f} < 0.2)")
    else:
        results["diagnosis"] = "mixed"
        print(f"  DIAGNOSIS: Mixed (R²={r2_mean:.3f})")

    save_json(RESULTS_DIR / "Q8_meanpool_length.json", results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    experiments = [
        ("Q1", "Sign flip check", Q1_sign_flip),
        ("Q2", "Token count on BBH", Q2_token_count_bbh),
        ("Q3", "Combined logistic on BBH", Q3_combined_bbh),
        ("Q6", "Cascade oracle bound", Q6_cascade_oracle),
        ("Q4", "Calibration deep dive", Q4_calibration_deep),
        ("Q5", "Per-subject 7B", Q5_per_subject_7b),
        ("Q7", "Disagreement quadrants", Q7_disagreement),
        ("Q8", "Mean-pool vs length", Q8_meanpool_length),
    ]

    import sys
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

    print("\nAll diagnostics complete.")


if __name__ == "__main__":
    main()
