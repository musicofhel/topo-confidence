#!/usr/bin/env python3
"""Final diagnostic experiments G1–G4 — paper-defense quality.

G1: 7B AUROC decomposition (difficulty vs model-specific)
G2: Cross-domain cosine-0.12 deep dive (independent signal?)
G3: Latency decomposition for cascade variants
G4: Calibrator shootout (pick one for the paper)
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lib import (
    CACHE_DIR, FIGS_DIR, N_FOLDS, RESULTS_DIR, SEED, STEERING_LAYER,
    DomProbe, bootstrap_auroc, ece_score, ensure_dirs, load_cache,
    oof_dom_scores, save_fig, save_json, setup_matplotlib, timer,
)


# ═══════════════════════════════════════════════════════════════
# G1. 7B AUROC decomposition — difficulty vs model-specific
# ═══════════════════════════════════════════════════════════════

def G1_7b_decomposition():
    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)

    X_7b = c_7b["X_prefill"][:, STEERING_LAYER, :]
    X_1p5b = c_1p5b["X_prefill"][:, STEERING_LAYER, :]

    # Partition into 4 quadrants
    both_right = y_1p5b & y_7b
    only_7b_right = ~y_1p5b & y_7b
    only_1p5b_right = y_1p5b & ~y_7b
    both_wrong = ~y_1p5b & ~y_7b

    n_br = int(both_right.sum())
    n_o7 = int(only_7b_right.sum())
    n_o1 = int(only_1p5b_right.sum())
    n_bw = int(both_wrong.sum())

    print(f"  Partition (n=500):")
    print(f"    Both right:       {n_br:4d} ({n_br/500:.1%})")
    print(f"    Only 7B right:    {n_o7:4d} ({n_o7/500:.1%})")
    print(f"    Only 1.5B right:  {n_o1:4d} ({n_o1/500:.1%})")
    print(f"    Both wrong:       {n_bw:4d} ({n_bw/500:.1%})")

    results = {
        "partition": {
            "both_right": n_br,
            "only_7b_right": n_o7,
            "only_1p5b_right": n_o1,
            "both_wrong": n_bw,
        },
    }

    # Disagreement set: problems where 1.5B and 7B disagree
    disagree = only_7b_right | only_1p5b_right
    n_disagree = int(disagree.sum())
    print(f"\n  Disagreement set: {n_disagree} problems")

    # 7B probe AUROC on disagreement set, predicting 7B correctness
    scores_7b_all = oof_dom_scores(X_7b, y_7b)

    if n_o7 >= 5 and n_o1 >= 5:
        y_disagree_7b = y_7b[disagree]
        scores_disagree = scores_7b_all[disagree]
        auroc_7b_disagree = float(roc_auc_score(y_disagree_7b, scores_disagree))
        ba_7b_disagree = bootstrap_auroc(y_disagree_7b, scores_disagree, n_resamples=2000)
        print(f"  7B probe AUROC on disagreement (predicting 7B correct):")
        print(f"    {auroc_7b_disagree:.4f} [{ba_7b_disagree['bca_lo']:.4f}, {ba_7b_disagree['bca_hi']:.4f}]")
        results["7b_probe_disagreement"] = {
            "auroc": auroc_7b_disagree,
            "bca_lo": ba_7b_disagree["bca_lo"],
            "bca_hi": ba_7b_disagree["bca_hi"],
            "n": n_disagree,
            "n_7b_correct": n_o7,
            "n_7b_wrong": n_o1,
        }
    else:
        print(f"  Too few disagreements for reliable AUROC (n_o7={n_o7}, n_o1={n_o1})")
        results["7b_probe_disagreement"] = {"skipped": True}

    # 1.5B probe AUROC on disagreement set, predicting 1.5B correctness
    scores_1p5b_all = oof_dom_scores(X_1p5b, y_1p5b)

    if n_o7 >= 5 and n_o1 >= 5:
        y_disagree_1p5b = y_1p5b[disagree]
        scores_disagree_1p5b = scores_1p5b_all[disagree]
        auroc_1p5b_disagree = float(roc_auc_score(y_disagree_1p5b, scores_disagree_1p5b))
        ba_1p5b_disagree = bootstrap_auroc(y_disagree_1p5b, scores_disagree_1p5b, n_resamples=2000)
        print(f"  1.5B probe AUROC on disagreement (predicting 1.5B correct):")
        print(f"    {auroc_1p5b_disagree:.4f} [{ba_1p5b_disagree['bca_lo']:.4f}, {ba_1p5b_disagree['bca_hi']:.4f}]")
        results["1p5b_probe_disagreement"] = {
            "auroc": auroc_1p5b_disagree,
            "bca_lo": ba_1p5b_disagree["bca_lo"],
            "bca_hi": ba_1p5b_disagree["bca_hi"],
        }

    # Per-quadrant mean probe scores
    for label, mask in [("both_right", both_right), ("only_7b", only_7b_right),
                         ("only_1p5b", only_1p5b_right), ("both_wrong", both_wrong)]:
        if mask.sum() > 0:
            s7 = float(scores_7b_all[mask].mean())
            s1 = float(scores_1p5b_all[mask].mean())
            print(f"  {label:16s}: mean 7B score={s7:+.4f}, mean 1.5B score={s1:+.4f}")
            results[f"mean_scores_{label}"] = {"7b": s7, "1p5b": s1}

    # Reference: full-data AUROC
    auroc_7b_full = float(roc_auc_score(y_7b, scores_7b_all))
    results["7b_probe_full"] = auroc_7b_full
    print(f"\n  Reference: 7B full AUROC = {auroc_7b_full:.4f}")

    # "Difficulty-only" baseline: use 1.5B correctness to predict 7B correctness
    auroc_difficulty = float(roc_auc_score(y_7b, y_1p5b.astype(float) +
                                            np.random.default_rng(SEED).uniform(0, 0.001, 500)))
    results["difficulty_baseline_auroc"] = auroc_difficulty
    print(f"  Difficulty baseline (1.5B label → 7B): {auroc_difficulty:.4f}")

    # Diagnosis
    if "auroc" in results.get("7b_probe_disagreement", {}):
        d_auroc = results["7b_probe_disagreement"]["auroc"]
        if d_auroc >= 0.80:
            results["diagnosis"] = "strong_model_specific"
            print(f"\n  DIAGNOSIS: 7B probe has strong model-specific signal ({d_auroc:.3f} on disagreement)")
            print(f"  Headline: '0.874 overall, {d_auroc:.3f} on model-disagreement set'")
        elif d_auroc >= 0.65:
            results["diagnosis"] = "moderate_model_specific"
            print(f"\n  DIAGNOSIS: Model-specific signal exists but moderate ({d_auroc:.3f})")
            print(f"  Headline: '0.874 overall, {d_auroc:.3f} controlling for difficulty'")
        else:
            results["diagnosis"] = "mostly_difficulty"
            print(f"\n  DIAGNOSIS: Probe is mostly difficulty detection ({d_auroc:.3f} on disagreement)")
            print(f"  Reviewer 2 wins this round.")

    save_json(RESULTS_DIR / "G1_7b_decomposition.json", results)


# ═══════════════════════════════════════════════════════════════
# G2. Cross-domain cosine-0.12 deep dive
# ═══════════════════════════════════════════════════════════════

def G2_cross_domain_directions():
    plt = setup_matplotlib()

    c_math = load_cache("math500_1p5b")
    c_bbh = load_cache("bbh_1p5b")

    y_math = c_math["y"].astype(bool)
    y_bbh = c_bbh["y"].astype(bool)

    X_math_pf = c_math["X_prefill"][:, STEERING_LAYER, :]
    X_bbh_pf = c_bbh["X_prefill"][:, STEERING_LAYER, :]

    # Train probes on each domain
    probe_math = DomProbe().fit(X_math_pf, y_math)
    probe_bbh = DomProbe().fit(X_bbh_pf, y_bbh)

    w_math = probe_math.direction
    w_bbh = probe_bbh.direction
    cos_angle = float(np.dot(w_math, w_bbh))

    # Project BBH data onto BOTH directions
    score_bbh_via_math = probe_math.score(X_bbh_pf)
    score_bbh_via_bbh = oof_dom_scores(X_bbh_pf, y_bbh)

    # AUROC of each direction on BBH
    auroc_math_dir = float(roc_auc_score(y_bbh, score_bbh_via_math))
    auroc_bbh_dir = float(roc_auc_score(y_bbh, score_bbh_via_bbh))

    # 2-feature logistic: do the two directions capture INDEPENDENT signal?
    features_2d = np.column_stack([score_bbh_via_math, score_bbh_via_bbh])

    scaler = StandardScaler()
    scores_combined = np.zeros(len(y_bbh), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(features_2d, y_bbh):
        X_tr = scaler.fit_transform(features_2d[tr])
        X_te = scaler.transform(features_2d[te])
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_tr, y_bbh[tr])
        scores_combined[te] = lr.predict_proba(X_te)[:, 1]
    auroc_combined = float(roc_auc_score(y_bbh, scores_combined))

    r_scores, _ = pearsonr(score_bbh_via_math, score_bbh_via_bbh)

    print(f"  cos(w_math, w_bbh) = {cos_angle:.4f}")
    print(f"  r(score_math_dir, score_bbh_dir) on BBH data = {r_scores:.4f}")
    print(f"\n  BBH AUROC via MATH direction:   {auroc_math_dir:.4f}")
    print(f"  BBH AUROC via BBH direction:    {auroc_bbh_dir:.4f}")
    print(f"  BBH AUROC via 2-feature combo:  {auroc_combined:.4f}")

    delta = auroc_combined - max(auroc_math_dir, auroc_bbh_dir)
    print(f"  Δ(combo vs best single):        {delta:+.4f}")

    results = {
        "cosine_angle": cos_angle,
        "score_correlation": float(r_scores),
        "auroc_math_direction_on_bbh": auroc_math_dir,
        "auroc_bbh_direction_on_bbh": auroc_bbh_dir,
        "auroc_2feature_combined": auroc_combined,
        "delta_combo_vs_best_single": float(delta),
    }

    # Per-subset breakdown
    subsets = c_bbh["subsets"]
    print(f"\n  Per-subset AUROC (MATH dir / BBH dir / combined):")
    for subset in sorted(set(subsets)):
        mask = subsets == subset
        y_s = y_bbh[mask]
        if y_s.sum() < 5 or (~y_s).sum() < 5:
            continue
        a_m = float(roc_auc_score(y_s, score_bbh_via_math[mask]))
        a_b = float(roc_auc_score(y_s, score_bbh_via_bbh[mask]))
        a_c = float(roc_auc_score(y_s, scores_combined[mask]))
        short = subset.split("_")[0]
        print(f"    {short:12s}: {a_m:.3f} / {a_b:.3f} / {a_c:.3f}")
        results[f"subset_{short}"] = {"math_dir": a_m, "bbh_dir": a_b, "combined": a_c}

    # 2D scatter plot
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {True: "forestgreen", False: "coral"}
    labels_done = set()
    for correct_val, label, marker in [(True, "Correct", "o"), (False, "Incorrect", "x")]:
        mask = y_bbh == correct_val
        ax.scatter(score_bbh_via_math[mask], score_bbh_via_bbh[mask],
                   c=colors[correct_val], marker=marker, s=25, alpha=0.5,
                   label=label, zorder=5 if correct_val else 4)
    ax.set_xlabel(f"MATH-trained probe score (AUROC={auroc_math_dir:.3f})")
    ax.set_ylabel(f"BBH-trained probe score (AUROC={auroc_bbh_dir:.3f})")
    ax.set_title(f"BBH problems projected onto two probe directions\ncos(w_math, w_bbh)={cos_angle:.3f}, r(scores)={r_scores:.3f}")
    ax.legend()
    ax.axhline(0, color="gray", ls=":", alpha=0.3)
    ax.axvline(0, color="gray", ls=":", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "G2_cross_domain_scatter.png")
    plt.close(fig)

    # Diagnosis
    if delta > 0.02:
        results["diagnosis"] = "independent_signal"
        print(f"\n  DIAGNOSIS: Two directions capture independent signal (Δ={delta:+.4f})")
        print("  Defensible claim: 'MATH probe hits a useful component in BBH space'")
    elif delta > 0.005:
        results["diagnosis"] = "marginal_independence"
        print(f"\n  DIAGNOSIS: Marginal independence (Δ={delta:+.4f})")
    else:
        results["diagnosis"] = "redundant"
        print(f"\n  DIAGNOSIS: Directions are operationally redundant despite low cosine")

    save_json(RESULTS_DIR / "G2_cross_domain_directions.json", results)


# ═══════════════════════════════════════════════════════════════
# G3. Latency decomposition for cascade variants
# ═══════════════════════════════════════════════════════════════

def G3_latency_decomposition():
    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)
    t_1p5b = c_1p5b["n_gen_tokens"].astype(float)
    t_7b = c_7b["n_gen_tokens"].astype(float)

    n = len(y_1p5b)

    # === Cost model (1.5B-token-equivalents) ===
    # 1 unit = cost of decoding 1 token on 1.5B
    # prefill_frac = 0.14 (given: prefill ≈ 14% of full 1.5B forward pass)
    # R = 5 (7B/1.5B cost ratio from cascade code)
    R = 5.0
    prefill_frac = 0.14
    mean_t1 = float(t_1p5b.mean())  # ~575

    # P₁ = prefill_frac / (1 - prefill_frac) × mean_gen_tokens
    # Because: P₁ / (P₁ + mean_T) = 0.14 → P₁ = 0.14/(1-0.14) × mean_T
    P1 = prefill_frac / (1 - prefill_frac) * mean_t1
    P7 = P1 * R  # same prompt, bigger model

    # Gate scores
    X_pf = c_1p5b["X_prefill"][:, STEERING_LAYER, :]
    gate_pregen = oof_dom_scores(X_pf, y_1p5b)

    # Hybrid gate (same as FU2)
    logprob = c_1p5b["mean_logprob"]
    tokens = c_1p5b["n_gen_tokens"].astype(float)
    entropy_L19 = c_1p5b["d2h_attn_entropy"][:, STEERING_LAYER]
    dom_scores = oof_dom_scores(X_pf, y_1p5b)
    features = np.column_stack([dom_scores, logprob, tokens, entropy_L19])

    scaler = StandardScaler()
    gate_hybrid = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(features, y_1p5b):
        X_tr = scaler.fit_transform(features[tr])
        X_te = scaler.transform(features[te])
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr.fit(X_tr, y_1p5b[tr])
        gate_hybrid[te] = lr.predict_proba(X_te)[:, 1]

    acc_all_7b = float(y_7b.mean())
    target_acc = 0.95 * acc_all_7b

    def find_best_threshold(scores, target):
        """Find tau_pct that achieves ≥target accuracy at MINIMUM per-query cost."""
        best = None
        for tau_pct in np.arange(0.01, 1.0, 0.01):
            tau = np.percentile(scores, (1 - tau_pct) * 100)
            keep = scores >= tau
            acc = (y_1p5b[keep].sum() + y_7b[~keep].sum()) / n
            if acc < target:
                continue
            # Compute per-query cost in token-equivalents
            cost_per_q = np.where(keep, P1 + t_1p5b, P1 + P7 + t_7b * R)
            mean_cost = float(cost_per_q.mean())
            if best is None or mean_cost < best[0]:
                best = (mean_cost, tau_pct, tau, keep)
        if best is None:
            return None, None, None
        return best[1], best[2], best[3]

    results = {
        "cost_model": {
            "P1_prefill": float(P1),
            "P7_prefill": float(P7),
            "R_gen_ratio": R,
            "mean_t_1p5b": float(mean_t1),
            "mean_t_7b": float(t_7b.mean()),
            "prefill_frac": prefill_frac,
        },
    }

    scenarios = {}

    # === All-7B baseline ===
    cost_all_7b = P7 + t_7b * R
    latency_all_7b = P7 + t_7b * R  # same as cost (single model)
    scenarios["all_7b"] = {
        "mean_cost": float(cost_all_7b.mean()),
        "p50_latency": float(np.percentile(latency_all_7b, 50)),
        "p95_latency": float(np.percentile(latency_all_7b, 95)),
        "accuracy": float(y_7b.mean()),
    }
    print(f"  All-7B:      mean_cost={cost_all_7b.mean():.0f}  p50={np.percentile(latency_all_7b, 50):.0f}  p95={np.percentile(latency_all_7b, 95):.0f}  acc={y_7b.mean():.3f}")

    # === All-1.5B baseline ===
    cost_all_1p5b = P1 + t_1p5b
    scenarios["all_1p5b"] = {
        "mean_cost": float(cost_all_1p5b.mean()),
        "p50_latency": float(np.percentile(cost_all_1p5b, 50)),
        "p95_latency": float(np.percentile(cost_all_1p5b, 95)),
        "accuracy": float(y_1p5b.mean()),
    }
    print(f"  All-1.5B:    mean_cost={cost_all_1p5b.mean():.0f}  p50={np.percentile(cost_all_1p5b, 50):.0f}  p95={np.percentile(cost_all_1p5b, 95):.0f}  acc={y_1p5b.mean():.3f}")

    # === Pre-gen cascade ===
    tau_pct, tau, keep_pregen = find_best_threshold(gate_pregen, target_acc)
    if keep_pregen is not None:
        esc_pregen = ~keep_pregen
        # Per-query cost:
        # Kept: P1 (already paid for gate) + t_1p5b[i]
        # Escalated: P1 (wasted prefill) + P7 + t_7b[i] * R
        cost_pregen = np.where(keep_pregen, P1 + t_1p5b, P1 + P7 + t_7b * R)
        latency_pregen = np.where(keep_pregen, P1 + t_1p5b, P1 + P7 + t_7b * R)

        acc_pregen = (y_1p5b[keep_pregen].sum() + y_7b[esc_pregen].sum()) / n
        pct_kept = keep_pregen.sum() / n

        scenarios["pregen_cascade"] = {
            "mean_cost": float(cost_pregen.mean()),
            "p50_latency": float(np.percentile(latency_pregen, 50)),
            "p95_latency": float(np.percentile(latency_pregen, 95)),
            "accuracy": float(acc_pregen),
            "pct_kept": float(pct_kept),
            "tau_pct": float(tau_pct),
            "cost_relative_to_all_7b": float(cost_pregen.mean() / cost_all_7b.mean()),
        }
        print(f"  Pre-gen:     mean_cost={cost_pregen.mean():.0f}  p50={np.percentile(latency_pregen, 50):.0f}  p95={np.percentile(latency_pregen, 95):.0f}  acc={acc_pregen:.3f}  kept={pct_kept:.0%}")

    # === Post-hoc (hybrid) cascade ===
    # For hybrid, cost model differs: 1.5B always runs fully, so escalated
    # queries pay 1.5B full + 7B full. Override cost calc for threshold search.
    def find_best_threshold_hybrid(scores, target):
        best = None
        for tau_pct in np.arange(0.01, 1.0, 0.01):
            tau = np.percentile(scores, (1 - tau_pct) * 100)
            keep = scores >= tau
            acc = (y_1p5b[keep].sum() + y_7b[~keep].sum()) / n
            if acc < target:
                continue
            cost_per_q = np.where(keep, P1 + t_1p5b, P1 + t_1p5b + P7 + t_7b * R)
            mean_cost = float(cost_per_q.mean())
            if best is None or mean_cost < best[0]:
                best = (mean_cost, tau_pct, tau, keep)
        if best is None:
            return None, None, None
        return best[1], best[2], best[3]

    tau_pct_h, tau_h, keep_hybrid = find_best_threshold_hybrid(gate_hybrid, target_acc)
    if keep_hybrid is not None:
        esc_hybrid = ~keep_hybrid
        # Per-query cost:
        # Kept: P1 + t_1p5b[i] (1.5B ran fully, answer accepted)
        # Escalated: P1 + t_1p5b[i] (1.5B ran fully, wasted) + P7 + t_7b[i] * R
        cost_hybrid = np.where(keep_hybrid,
                                P1 + t_1p5b,
                                P1 + t_1p5b + P7 + t_7b * R)
        latency_hybrid = cost_hybrid  # sequential

        acc_hybrid = (y_1p5b[keep_hybrid].sum() + y_7b[esc_hybrid].sum()) / n
        pct_kept_h = keep_hybrid.sum() / n

        scenarios["hybrid_cascade"] = {
            "mean_cost": float(cost_hybrid.mean()),
            "p50_latency": float(np.percentile(latency_hybrid, 50)),
            "p95_latency": float(np.percentile(latency_hybrid, 95)),
            "accuracy": float(acc_hybrid),
            "pct_kept": float(pct_kept_h),
            "tau_pct": float(tau_pct_h),
            "cost_relative_to_all_7b": float(cost_hybrid.mean() / cost_all_7b.mean()),
        }
        print(f"  Hybrid:      mean_cost={cost_hybrid.mean():.0f}  p50={np.percentile(latency_hybrid, 50):.0f}  p95={np.percentile(latency_hybrid, 95):.0f}  acc={acc_hybrid:.3f}  kept={pct_kept_h:.0%}")

    results["scenarios"] = scenarios

    # === Summary table ===
    print(f"\n  {'Scenario':<20s} {'Mean cost':>10s} {'p50 lat':>10s} {'p95 lat':>10s} {'Acc':>7s} {'Cost/7B':>8s}")
    print("  " + "-" * 70)
    ref_cost = scenarios["all_7b"]["mean_cost"]
    for name, s in scenarios.items():
        cost_frac = s["mean_cost"] / ref_cost
        print(f"  {name:<20s} {s['mean_cost']:>10.0f} {s['p50_latency']:>10.0f} {s['p95_latency']:>10.0f} {s['accuracy']:>7.3f} {cost_frac:>8.2f}×")

    # === Buyer recommendation ===
    pregen = scenarios.get("pregen_cascade", {})
    hybrid = scenarios.get("hybrid_cascade", {})
    if pregen and hybrid:
        pregen_p95 = pregen["p95_latency"]
        hybrid_p95 = hybrid["p95_latency"]
        pregen_cost = pregen["mean_cost"]
        hybrid_cost = hybrid["mean_cost"]

        print(f"\n  Code assistant (p95-latency-sensitive):")
        print(f"    Pre-gen p95={pregen_p95:.0f} vs Hybrid p95={hybrid_p95:.0f}")
        if pregen_p95 < hybrid_p95:
            pct_better = (1 - pregen_p95 / hybrid_p95) * 100
            print(f"    → Pre-gen wins p95 by {pct_better:.0f}%")
        else:
            print(f"    → Hybrid is faster (unexpected)")

        print(f"\n  Batch inference (mean-cost-sensitive):")
        print(f"    Pre-gen cost={pregen_cost:.0f} vs Hybrid cost={hybrid_cost:.0f}")
        if hybrid_cost < pregen_cost:
            pct_better = (1 - hybrid_cost / pregen_cost) * 100
            print(f"    → Hybrid wins cost by {pct_better:.0f}%")
        else:
            print(f"    → Pre-gen is cheaper (unexpected)")

        results["buyer_recommendation"] = {
            "latency_sensitive": "pregen" if pregen_p95 < hybrid_p95 else "hybrid",
            "cost_sensitive": "hybrid" if hybrid_cost < pregen_cost else "pregen",
            "pregen_p95_advantage_pct": float((1 - pregen_p95 / hybrid_p95) * 100) if hybrid_p95 > 0 else 0,
            "hybrid_cost_advantage_pct": float((1 - hybrid_cost / pregen_cost) * 100) if pregen_cost > 0 else 0,
        }

    save_json(RESULTS_DIR / "G3_latency_decomposition.json", results)


# ═══════════════════════════════════════════════════════════════
# G4. Calibrator shootout
# ═══════════════════════════════════════════════════════════════

def _histogram_binning_calibrate(scores_tr, y_tr, scores_te, n_bins=15):
    """Equal-mass histogram binning calibrator."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(scores_tr, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    bin_probs = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (scores_tr >= bins[i]) & (scores_tr < bins[i + 1])
        if mask.sum() > 0:
            bin_probs[i] = y_tr[mask].mean()
        else:
            bin_probs[i] = 0.5

    cal = np.zeros(len(scores_te))
    for i in range(n_bins):
        mask = (scores_te >= bins[i]) & (scores_te < bins[i + 1])
        cal[mask] = bin_probs[i]
    return cal


def _beta_calibrate(scores_tr, y_tr, scores_te):
    """Beta calibration (Kull et al. 2017): logistic regression on [log(s), log(1-s)].

    Requires input scores normalized to (0, 1).
    """
    eps = 1e-6
    s_tr = np.clip(scores_tr, eps, 1 - eps)
    s_te = np.clip(scores_te, eps, 1 - eps)

    feat_tr = np.column_stack([np.log(s_tr), np.log(1 - s_tr)])
    feat_te = np.column_stack([np.log(s_te), np.log(1 - s_te)])

    lr = LogisticRegression(max_iter=1000, C=1e6)
    lr.fit(feat_tr, y_tr)
    return lr.predict_proba(feat_te)[:, 1]


def G4_calibrator_shootout():
    plt = setup_matplotlib()
    from sklearn.isotonic import IsotonicRegression

    results = {}

    for tag, cache_name in [("1p5b", "math500_1p5b"), ("7b", "math500_7b")]:
        c = load_cache(cache_name)
        X = c["X_prefill"][:, STEERING_LAYER, :]
        y = c["y"].astype(bool)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # Collect OOF predictions for each calibrator
        cals = {
            "raw_sigmoid": np.zeros(len(y)),
            "isotonic": np.zeros(len(y)),
            "platt": np.zeros(len(y)),
            "beta": np.zeros(len(y)),
            "histogram_15": np.zeros(len(y)),
        }

        for tr, te in skf.split(X, y):
            probe = DomProbe().fit(X[tr], y[tr])
            raw_tr = probe.score(X[tr])
            raw_te = probe.score(X[te])

            std = raw_tr.std() + 1e-12

            # 1. Raw sigmoid
            cals["raw_sigmoid"][te] = expit(raw_te / std)

            # 2. Isotonic
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_tr, y[tr])
            cals["isotonic"][te] = iso.predict(raw_te)

            # 3. Platt
            lr = LogisticRegression(max_iter=1000)
            lr.fit(raw_tr.reshape(-1, 1), y[tr])
            cals["platt"][te] = lr.predict_proba(raw_te.reshape(-1, 1))[:, 1]

            # 4. Beta (input must be (0,1))
            sig_tr = expit(raw_tr / std)
            sig_te = expit(raw_te / std)
            cals["beta"][te] = _beta_calibrate(sig_tr, y[tr], sig_te)

            # 5. Histogram binning
            cals["histogram_15"][te] = _histogram_binning_calibrate(raw_tr, y[tr], raw_te, n_bins=15)

        # Evaluate each
        tag_results = {}
        print(f"\n  {tag} calibrator comparison:")
        print(f"  {'Calibrator':<16s} {'ECE(ew)':>8s} {'ECE(em)':>8s} {'Brier':>8s} {'AUROC':>8s}")
        print("  " + "-" * 52)

        for name, probs in cals.items():
            ece_ew = ece_score(y, probs, 15, "uniform")
            ece_em = ece_score(y, probs, 15, "quantile")
            brier = float(brier_score_loss(y, probs))
            auroc = float(roc_auc_score(y, probs))
            tag_results[name] = {
                "ece_uniform_15": ece_ew,
                "ece_quantile_15": ece_em,
                "brier": brier,
                "auroc": auroc,
            }
            marker = " ← best" if ece_ew == min(tag_results[n]["ece_uniform_15"] for n in tag_results) else ""
            print(f"  {name:<16s} {ece_ew:>8.4f} {ece_em:>8.4f} {brier:>8.4f} {auroc:>8.4f}{marker}")

        # Pick lowest ECE(ew)
        best_name = min(tag_results, key=lambda n: tag_results[n]["ece_uniform_15"])
        best_ece = tag_results[best_name]["ece_uniform_15"]
        tag_results["recommended"] = best_name
        tag_results["recommended_ece"] = best_ece
        print(f"\n  Recommended for {tag}: {best_name} (ECE={best_ece:.4f})")

        if best_ece > 0.05:
            print(f"  WARNING: best ECE={best_ece:.4f} > 0.05 target")
        else:
            print(f"  PASS: ECE < 0.05 target")

        results[tag] = tag_results

        # Reliability diagram for all calibrators
        fig, axes = plt.subplots(1, len(cals), figsize=(4 * len(cals), 4))
        for ax, (name, probs) in zip(axes, cals.items()):
            try:
                frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
                ax.plot(mean_pred, frac_pos, "-o", color="steelblue", ms=5)
            except ValueError:
                pass
            ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
            ece_val = tag_results[name]["ece_uniform_15"]
            ax.set_title(f"{name}\nECE={ece_val:.3f}", fontsize=9)
            ax.set_xlabel("Predicted", fontsize=8)
            if ax == axes[0]:
                ax.set_ylabel("Observed", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        fig.suptitle(f"Calibration comparison — {tag}", fontsize=11)
        fig.tight_layout()
        save_fig(fig, f"G4_calibration_{tag}.png")
        plt.close(fig)

    # Final recommendation
    rec_1p5b = results["1p5b"]["recommended"]
    rec_7b = results["7b"]["recommended"]
    print(f"\n  PAPER RECOMMENDATION:")
    print(f"    1.5B: use {rec_1p5b} (ECE={results['1p5b']['recommended_ece']:.4f})")
    print(f"    7B:   use {rec_7b} (ECE={results['7b']['recommended_ece']:.4f})")

    save_json(RESULTS_DIR / "G4_calibrator_shootout.json", results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    experiments = [
        ("G1", "7B AUROC decomposition", G1_7b_decomposition),
        ("G3", "Latency decomposition", G3_latency_decomposition),
        ("G2", "Cross-domain directions", G2_cross_domain_directions),
        ("G4", "Calibrator shootout", G4_calibrator_shootout),
    ]

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
    print("All final diagnostics complete. Results in:")
    print(f"  {RESULTS_DIR}/")
    print(f"  {FIGS_DIR}/")


if __name__ == "__main__":
    main()
