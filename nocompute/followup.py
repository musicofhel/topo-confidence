#!/usr/bin/env python3
"""Follow-up experiments F1–F6 on existing nocompute caches.

F4: 7B AUROC sanity check (rebalanced + difficulty leakage)
F1: Pre-generation baseline leaderboard
F2: Cascade with combined gate (three variants)
F3: Compound cross-tier + cross-domain transfer
F5: Depth-fraction overlay figure
F6: Mean-pool residualized against token count
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import entr, softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lib import (
    CACHE_DIR, DATA_1P5B, DATA_BBH, BBH_SUBSETS, FIGS_DIR, N_FOLDS, N_LAYERS,
    RESULTS_DIR, SEED, STEERING_LAYER,
    DomProbe, bootstrap_auroc, ensure_dirs, load_cache, oof_dom_scores,
    save_fig, save_json, setup_matplotlib, timer,
)


# ═══════════════════════════════════════════════════════════════
# F4. 7B AUROC sanity check
# ═══════════════════════════════════════════════════════════════

def FU4_7b_sanity():
    c_7b = load_cache("math500_7b")
    c_1p5b = load_cache("math500_1p5b")
    X_7b = c_7b["X_prefill"][:, STEERING_LAYER, :]
    y_7b = c_7b["y"].astype(bool)
    y_1p5b = c_1p5b["y"].astype(bool)

    # --- Action 1: rebalanced subsampling to 50/50 ---
    rng = np.random.default_rng(SEED)
    n_correct = int(y_7b.sum())
    n_incorrect = int((~y_7b).sum())
    n_each = min(n_correct, n_incorrect)

    idx_incorrect = np.where(~y_7b)[0]
    idx_correct_all = np.where(y_7b)[0]

    aurocs_rebal = []
    for rep in range(50):
        idx_correct_sub = rng.choice(idx_correct_all, size=n_each, replace=False)
        idx = np.concatenate([idx_correct_sub, idx_incorrect])
        rng.shuffle(idx)
        y_sub = y_7b[idx]
        X_sub = X_7b[idx]
        scores = oof_dom_scores(X_sub, y_sub, seed=SEED + rep)
        aurocs_rebal.append(float(roc_auc_score(y_sub, scores)))

    print(f"  Rebalanced 50/50 (n_each={n_each}): {np.mean(aurocs_rebal):.4f} ± {np.std(aurocs_rebal):.4f}")
    print(f"    range [{np.min(aurocs_rebal):.4f}, {np.max(aurocs_rebal):.4f}]")

    # --- Action 2: difficulty leakage ---
    scores_7b = oof_dom_scores(X_7b, y_7b)
    r_diff, p_diff = pearsonr(scores_7b, y_1p5b.astype(float))
    rho_diff, p_rho = spearmanr(scores_7b, y_1p5b.astype(float))

    print(f"  Difficulty leakage: Pearson(7B_probe, 1.5B_correct) = {r_diff:.4f} (p={p_diff:.4f})")
    print(f"    Spearman ρ = {rho_diff:.4f}")

    # AUROC of 7B probe scores against 1.5B labels
    auroc_7b_vs_1p5b = float(roc_auc_score(y_1p5b, scores_7b))
    print(f"  AUROC(7B probe scores, 1.5B labels) = {auroc_7b_vs_1p5b:.4f}")

    # How much overlap in the labels?
    agree = (y_7b == y_1p5b)
    both_correct = (y_7b & y_1p5b).sum()
    both_wrong = (~y_7b & ~y_1p5b).sum()
    print(f"  Label agreement: {agree.mean():.1%} (both correct: {both_correct}, both wrong: {both_wrong})")

    results = {
        "rebalanced": {
            "n_each": n_each,
            "n_reps": 50,
            "mean": float(np.mean(aurocs_rebal)),
            "std": float(np.std(aurocs_rebal)),
            "min": float(np.min(aurocs_rebal)),
            "max": float(np.max(aurocs_rebal)),
            "all_aurocs": aurocs_rebal,
        },
        "difficulty_leakage": {
            "pearson_r": float(r_diff),
            "pearson_p": float(p_diff),
            "spearman_rho": float(rho_diff),
            "spearman_p": float(p_rho),
            "auroc_7b_probe_vs_1p5b_labels": auroc_7b_vs_1p5b,
        },
        "label_overlap": {
            "agreement_rate": float(agree.mean()),
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "only_7b_correct": int((y_7b & ~y_1p5b).sum()),
            "only_1p5b_correct": int((~y_7b & y_1p5b).sum()),
        },
        "headline_7b_auroc": float(roc_auc_score(y_7b, scores_7b)),
        "n_correct_7b": n_correct,
        "n_incorrect_7b": n_incorrect,
        "class_ratio": f"{n_correct}:{n_incorrect}",
    }

    if float(r_diff) > 0.7:
        results["diagnosis"] = "difficulty_detector"
        print(f"\n  DIAGNOSIS: 7B probe is largely a difficulty detector (r={r_diff:.3f} > 0.7)")
    elif float(r_diff) < 0.4:
        results["diagnosis"] = "genuinely_7b_specific"
        print(f"\n  DIAGNOSIS: 7B probe is genuinely 7B-specific (r={r_diff:.3f} < 0.4)")
    else:
        results["diagnosis"] = "mixed"
        print(f"\n  DIAGNOSIS: Mixed difficulty/model-specific signal (r={r_diff:.3f})")

    save_json(RESULTS_DIR / "FU4_7b_sanity.json", results)


# ═══════════════════════════════════════════════════════════════
# F1. Pre-generation baseline leaderboard
# ═══════════════════════════════════════════════════════════════

def FU1_pregen_leaderboard():
    c = load_cache("math500_1p5b")
    X_pf = c["X_prefill"][:, STEERING_LAYER, :].astype(np.float64)
    y = c["y"].astype(bool)

    baselines = {}

    # (b) L1 norm of prefill activation
    l1_norm = np.abs(X_pf).sum(axis=1)
    ba_l1_pos = bootstrap_auroc(y, l1_norm, n_resamples=2000)
    ba_l1_neg = bootstrap_auroc(y, -l1_norm, n_resamples=2000)
    best_l1 = ba_l1_pos if ba_l1_pos["point"] > ba_l1_neg["point"] else ba_l1_neg
    sign_l1 = "positive" if ba_l1_pos["point"] > ba_l1_neg["point"] else "negative"
    baselines["prefill_L1_norm"] = {**best_l1, "sign": sign_l1, "pre_generation": True}

    # (c) L2 norm
    l2_norm = np.linalg.norm(X_pf, axis=1)
    ba_l2_pos = bootstrap_auroc(y, l2_norm, n_resamples=2000)
    ba_l2_neg = bootstrap_auroc(y, -l2_norm, n_resamples=2000)
    best_l2 = ba_l2_pos if ba_l2_pos["point"] > ba_l2_neg["point"] else ba_l2_neg
    sign_l2 = "positive" if ba_l2_pos["point"] > ba_l2_neg["point"] else "negative"
    baselines["prefill_L2_norm"] = {**best_l2, "sign": sign_l2, "pre_generation": True}

    # (d) Activation entropy: softmax over hidden dim, then Shannon H
    probs = softmax(X_pf, axis=1)
    act_entropy = entr(probs).sum(axis=1)
    ba_ae_pos = bootstrap_auroc(y, act_entropy, n_resamples=2000)
    ba_ae_neg = bootstrap_auroc(y, -act_entropy, n_resamples=2000)
    best_ae = ba_ae_pos if ba_ae_pos["point"] > ba_ae_neg["point"] else ba_ae_neg
    sign_ae = "positive" if ba_ae_pos["point"] > ba_ae_neg["point"] else "negative"
    baselines["prefill_activation_entropy"] = {**best_ae, "sign": sign_ae, "pre_generation": True}

    # (e) Attention entropy at L19 — NOTE: d2h_attn_entropy aggregates
    # over all generated positions, making it a post-generation signal.
    # Including for comparison but flagged as post-gen.
    attn_ent = c["d2h_attn_entropy"][:, STEERING_LAYER]
    if not np.all(attn_ent == 0):
        ba_at_pos = bootstrap_auroc(y, attn_ent, n_resamples=2000)
        ba_at_neg = bootstrap_auroc(y, -attn_ent, n_resamples=2000)
        best_at = ba_at_pos if ba_at_pos["point"] > ba_at_neg["point"] else ba_at_neg
        sign_at = "positive" if ba_at_pos["point"] > ba_at_neg["point"] else "negative"
        baselines["d2h_attn_entropy_L19"] = {
            **best_at, "sign": sign_at,
            "pre_generation": False,
            "caveat": "aggregated over generation positions, not prefill-only",
        }

    # (f) DomProbe prefill (reference)
    dom_scores = oof_dom_scores(X_pf, y)
    ba_dom = bootstrap_auroc(y, dom_scores, n_resamples=2000)
    baselines["prefill_DomProbe_L19"] = {**ba_dom, "pre_generation": True}

    # Rank by AUROC
    ranked = sorted(baselines.items(), key=lambda x: x[1]["point"], reverse=True)

    print(f"\n  {'Rank':<5} {'Baseline':<30} {'AUROC':>7} {'95% BCa CI':>20} {'Pre-gen':>8}")
    print("  " + "-" * 75)
    for rank, (name, vals) in enumerate(ranked, 1):
        pre = "YES" if vals.get("pre_generation", False) else "NO"
        ci = f"[{vals['bca_lo']:.4f}, {vals['bca_hi']:.4f}]"
        print(f"  {rank:<5} {name:<30} {vals['point']:>7.4f} {ci:>20} {pre:>8}")

    # Compute gap from DomProbe to next-best pre-gen signal
    pregen_only = [(n, v) for n, v in ranked if v.get("pre_generation", False) and n != "prefill_DomProbe_L19"]
    if pregen_only:
        next_best_name, next_best = pregen_only[0]
        gap = ba_dom["point"] - next_best["point"]
        print(f"\n  DomProbe lead over next-best pre-gen ({next_best_name}): +{gap:.4f}")

    results = {
        "baselines": {k: v for k, v in baselines.items()},
        "ranking": [{"rank": i + 1, "name": n, "auroc": v["point"]} for i, (n, v) in enumerate(ranked)],
        "notes": {
            "prompt_token_length": "NOT COMPUTED — original MATH-500 problem texts not stored in cache. Since AUROC is rank-based, character length would suffice but prompt text is unavailable in NPZ/manifest.",
            "prefill_attn_entropy": "d2h_attn_entropy aggregates over all generation positions. True prefill-position attention entropy would require per-position attention weights not in the cache.",
        },
    }

    save_json(RESULTS_DIR / "FU1_pregen_leaderboard.json", results)


# ═══════════════════════════════════════════════════════════════
# F2. Cascade with combined gate
# ═══════════════════════════════════════════════════════════════

def FU2_cascade_combined():
    c_1p5b = load_cache("math500_1p5b")
    c_7b = load_cache("math500_7b")

    y_1p5b = c_1p5b["y"].astype(bool)
    y_7b = c_7b["y"].astype(bool)
    n = len(y_1p5b)

    cost_1p5b = 1.0
    cost_7b = 5.0
    acc_all_7b = float(y_7b.mean())
    acc_all_1p5b = float(y_1p5b.mean())

    # --- Gate (i): pre-generation — DomProbe prefill only ---
    X_pf = c_1p5b["X_prefill"][:, STEERING_LAYER, :]
    gate_pregen = oof_dom_scores(X_pf, y_1p5b)

    # --- Gate (ii): hybrid — DomProbe + logprob + token count (post-hoc) ---
    logprob = c_1p5b["mean_logprob"]
    tokens = c_1p5b["n_gen_tokens"].astype(float)
    entropy_L19 = c_1p5b["d2h_attn_entropy"][:, STEERING_LAYER]

    dom_scores_pf = oof_dom_scores(X_pf, y_1p5b)
    features = np.column_stack([dom_scores_pf, logprob, tokens, entropy_L19])

    scaler = StandardScaler()
    gate_hybrid = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(features, y_1p5b):
        X_tr = scaler.fit_transform(features[tr])
        X_te = scaler.transform(features[te])
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr.fit(X_tr, y_1p5b[tr])
        gate_hybrid[te] = lr.predict_proba(X_te)[:, 1]

    # --- Gate (iii): oracle — perfect knowledge ---
    rng = np.random.default_rng(SEED)
    gate_oracle = y_1p5b.astype(float) + rng.uniform(0, 0.01, n)

    def cascade_sweep(scores, label):
        thresholds = np.arange(0.01, 1.0, 0.01)
        rows = []
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

    gates = {
        "pregen_domprobe": (gate_pregen, "Pre-gen (DomProbe prefill)"),
        "hybrid_combined": (gate_hybrid, "Hybrid (DomProbe+logprob+tokens+entropy)"),
        "oracle": (gate_oracle, "Oracle (perfect 1.5B knowledge)"),
    }

    results = {"acc_all_7b": acc_all_7b, "acc_all_1p5b": acc_all_1p5b}

    for target_pct, target_label in [(0.95, "95%"), (0.90, "90%")]:
        target_acc = target_pct * acc_all_7b
        results[f"target_{target_label}"] = {"target_acc": float(target_acc)}

        print(f"\n  ≥{target_label} of 7B accuracy ({target_acc:.3f}):")
        for gate_name, (gate_scores, desc) in gates.items():
            sweep = cascade_sweep(gate_scores, gate_name)
            valid = [r for r in sweep if r["accuracy"] >= target_acc]
            best = min(valid, key=lambda r: r["cost_rel"]) if valid else None

            if best:
                saving = (1 - best["cost_rel"]) * 100
                print(f"    {desc:48s}: cost={best['cost_rel']:.3f}× (saves {saving:.0f}%), kept={best['pct_kept']*100:.0f}%")
            else:
                print(f"    {desc:48s}: IMPOSSIBLE at this threshold")

            results[f"target_{target_label}"][gate_name] = best

    # AUROC of each gate against 1.5B correctness
    for gate_name, (gate_scores, desc) in gates.items():
        auroc = float(roc_auc_score(y_1p5b, gate_scores))
        results[f"auroc_{gate_name}"] = auroc
    print(f"\n  Gate AUROCs (vs 1.5B correct):")
    print(f"    Pre-gen DomProbe: {results['auroc_pregen_domprobe']:.4f}")
    print(f"    Hybrid combined:  {results['auroc_hybrid_combined']:.4f}")
    print(f"    Oracle:           {results['auroc_oracle']:.4f}")

    # Key comparison: pre-gen vs hybrid savings gap
    t95 = results.get("target_95%", {})
    pregen_95 = t95.get("pregen_domprobe")
    hybrid_95 = t95.get("hybrid_combined")
    if pregen_95 and hybrid_95:
        pregen_saving = (1 - pregen_95["cost_rel"]) * 100
        hybrid_saving = (1 - hybrid_95["cost_rel"]) * 100
        gap_pp = hybrid_saving - pregen_saving
        results["savings_gap_95_pp"] = float(gap_pp)
        print(f"\n  Hybrid saves {gap_pp:+.0f}pp more than pre-gen at 95% threshold")
        if gap_pp <= 3:
            print("  → Prefill story holds: hybrid adds ≤3pp")
        else:
            print(f"  → Post-hoc cascade is the product story (+{gap_pp:.0f}pp)")

    # Figure: Pareto for all three gates
    plt = setup_matplotlib()
    fig, ax = plt.subplots()

    colors = {"pregen_domprobe": "steelblue", "hybrid_combined": "forestgreen", "oracle": "coral"}
    labels = {"pregen_domprobe": "Pre-gen (DomProbe)", "hybrid_combined": "Hybrid (combined)", "oracle": "Oracle"}

    for gate_name, (gate_scores, _) in gates.items():
        sweep = cascade_sweep(gate_scores, gate_name)
        costs = [r["cost_rel"] for r in sweep]
        accs = [r["accuracy"] for r in sweep]
        ax.plot(costs, accs, "-o", ms=3, color=colors[gate_name], label=labels[gate_name])

    ax.axhline(acc_all_7b, color="gray", ls="--", alpha=0.7, label=f"All-7B ({acc_all_7b:.3f})")
    ax.axhline(acc_all_1p5b, color="gray", ls=":", alpha=0.5, label=f"All-1.5B ({acc_all_1p5b:.3f})")
    ax.axhline(0.95 * acc_all_7b, color="coral", ls=":", alpha=0.3, label="95% of 7B")
    ax.set_xlabel("Cost (relative to all-7B)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cascade Pareto — three gate variants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "FU2_cascade_pareto.png")
    plt.close(fig)

    save_json(RESULTS_DIR / "FU2_cascade_combined.json", results)


# ═══════════════════════════════════════════════════════════════
# F3. Compound cross-tier + cross-domain
# ═══════════════════════════════════════════════════════════════

def FU3_compound_transfer():
    """Cross-tier AND cross-domain: 1.5B MATH → 7B BBH.

    Requires 7B BBH activations. Since we only have bbh_1p5b and
    math500_7b, we test the closest available compound case:
    train on 1.5B MATH prefill, ridge-project to 7B MATH space,
    then score with that probe direction on 7B data.

    Also reports the decomposed single-transfer cases for comparison.
    """
    c_math_1p5b = load_cache("math500_1p5b")
    c_math_7b = load_cache("math500_7b")
    c_bbh_1p5b = load_cache("bbh_1p5b")

    y_math_1p5b = c_math_1p5b["y"].astype(bool)
    y_math_7b = c_math_7b["y"].astype(bool)
    y_bbh_1p5b = c_bbh_1p5b["y"].astype(bool)

    L = STEERING_LAYER
    X_math_1p5b = c_math_1p5b["X_prefill"][:, L, :]
    X_math_7b = c_math_7b["X_prefill"][:, L, :]
    X_bbh_1p5b = c_bbh_1p5b["X_prefill"][:, L, :]

    results = {}

    # --- Single transfers (decomposed components) ---

    # Cross-domain only (same tier): 1.5B MATH → 1.5B BBH prefill
    probe_math = DomProbe().fit(X_math_1p5b, y_math_1p5b)
    scores_bbh = probe_math.score(X_bbh_1p5b)
    ba_cd = bootstrap_auroc(y_bbh_1p5b, scores_bbh, n_resamples=2000)
    results["cross_domain_only"] = {**ba_cd, "desc": "1.5B MATH → 1.5B BBH (prefill)"}
    print(f"  Cross-domain (1.5B MATH→BBH prefill): AUROC={ba_cd['point']:.4f} [{ba_cd['bca_lo']:.4f}, {ba_cd['bca_hi']:.4f}]")

    # Cross-tier only (same domain): 1.5B MATH → 7B MATH via ridge
    n = len(y_math_1p5b)
    n_holdout = 100
    n_repeats = 20
    rng = np.random.default_rng(SEED)

    aurocs_ct = []
    for rep in range(n_repeats):
        test_idx = rng.choice(n, size=n_holdout, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_math_7b[train_mask], X_math_1p5b[train_mask])
        probe = DomProbe().fit(X_math_1p5b[train_mask], y_math_1p5b[train_mask])

        X_7b_proj = ridge.predict(X_math_7b[test_mask])
        scores = probe.score(X_7b_proj)
        if len(np.unique(y_math_7b[test_mask])) >= 2:
            aurocs_ct.append(float(roc_auc_score(y_math_7b[test_mask], scores)))

    results["cross_tier_only"] = {
        "mean": float(np.mean(aurocs_ct)),
        "std": float(np.std(aurocs_ct)),
        "desc": "1.5B MATH → 7B MATH via ridge (prefill)",
    }
    print(f"  Cross-tier (1.5B→7B MATH prefill):    AUROC={np.mean(aurocs_ct):.4f} ± {np.std(aurocs_ct):.4f}")

    # --- Compound: 1.5B MATH probe → applied on 7B space (no BBH 7B data) ---
    results["compound_cross_tier_cross_domain"] = {
        "status": "NOT_COMPUTED",
        "reason": "7B BBH activations not in cache (only bbh_1p5b, math500_7b exist). Requires running BBH through Qwen2.5-7B on GPU pod.",
        "desc": "Would be: train 1.5B MATH → ridge to 7B → score 7B BBH",
    }
    print(f"  Compound (1.5B MATH→7B BBH):          NOT COMPUTED — missing 7B BBH data")

    # --- Closest approximation: direct probe cosine alignment ---
    # Train probe on 1.5B MATH, train probe on 1.5B BBH, measure cosine
    probe_bbh = DomProbe().fit(X_bbh_1p5b, y_bbh_1p5b)
    cos_math_bbh = float(np.dot(probe_math.direction, probe_bbh.direction))
    results["probe_direction_cosine_math_bbh_1p5b"] = cos_math_bbh
    print(f"  cos(MATH probe, BBH probe) at 1.5B prefill: {cos_math_bbh:.4f}")

    # Within-domain reference
    scores_math_oof = oof_dom_scores(X_math_1p5b, y_math_1p5b)
    ba_wd = bootstrap_auroc(y_math_1p5b, scores_math_oof, n_resamples=2000)
    results["within_domain_reference"] = {**ba_wd, "desc": "1.5B MATH within-domain prefill OOF"}
    print(f"  Within-domain (1.5B MATH OOF prefill): AUROC={ba_wd['point']:.4f}")

    save_json(RESULTS_DIR / "FU3_compound_transfer.json", results)


# ═══════════════════════════════════════════════════════════════
# F5. Depth-fraction overlay figure
# ═══════════════════════════════════════════════════════════════

def FU5_depth_fraction():
    plt = setup_matplotlib()

    # Load layer sweep results
    b1_path = RESULTS_DIR / "B1_layer_sweep_1p5b.json"
    b2_path = RESULTS_DIR / "B2_layer_sweep_7b.json"
    b3_path = RESULTS_DIR / "B3_layer_sweep_bbh.json"

    if not b1_path.exists() or not b2_path.exists():
        print("  ERROR: B1/B2 results not found. Run run_experiments.py B1 B2 first.")
        return

    b1 = json.loads(b1_path.read_text())
    b2 = json.loads(b2_path.read_text())
    b3 = json.loads(b3_path.read_text()) if b3_path.exists() else None

    total_layers = 28  # both 1.5B and 7B are 28-layer

    curves = {
        "1.5B MATH (prefill)": (b1["prefill"], "steelblue", "-o"),
        "7B MATH (prefill)": (b2["prefill"], "coral", "-s"),
    }
    if b3 and "pooled" in b3 and "layers" in b3["pooled"]:
        curves["1.5B BBH pooled (mean-pool)"] = (b3["pooled"]["layers"], "forestgreen", "-^")

    fig, ax = plt.subplots(figsize=(10, 6))

    plateau_info = {}

    for label, (rows, color, marker) in curves.items():
        layers = [r["layer"] for r in rows]
        aurocs = np.array([r["auroc"] for r in rows])
        depth_fracs = np.array(layers) / total_layers

        ax.plot(depth_fracs, aurocs, marker, ms=5, label=label, color=color)

        # Fill CI bands if available
        if "bca_lo" in rows[0]:
            lo = [r["bca_lo"] for r in rows]
            hi = [r["bca_hi"] for r in rows]
            ax.fill_between(depth_fracs, lo, hi, alpha=0.1, color=color)

        # Find peak and 99% plateau (within 1pp of peak)
        peak_auroc = aurocs.max()
        peak_layer = layers[aurocs.argmax()]
        threshold_99 = peak_auroc - 0.01
        plateau_mask = aurocs >= threshold_99
        plateau_layers = np.array(layers)[plateau_mask]
        plateau_lo = plateau_layers.min() / total_layers
        plateau_hi = plateau_layers.max() / total_layers

        plateau_info[label] = {
            "peak_layer": int(peak_layer),
            "peak_depth_frac": float(peak_layer / total_layers),
            "peak_auroc": float(peak_auroc),
            "plateau_99_lo_frac": float(plateau_lo),
            "plateau_99_hi_frac": float(plateau_hi),
            "plateau_99_width": float(plateau_hi - plateau_lo),
            "plateau_99_layers": [int(l) for l in plateau_layers],
        }

        # Mark plateau with shading
        ax.axvspan(plateau_lo, plateau_hi, alpha=0.05, color=color)
        ax.axvline(peak_layer / total_layers, color=color, ls=":", alpha=0.3)

    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Depth fraction (layer / 28)")
    ax.set_ylabel("AUROC")
    ax.set_title("Layer sweep on depth-fraction axis — peak and 99% plateau")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "FU5_depth_fraction_overlay.png")
    plt.close(fig)

    # Print summary
    print(f"\n  {'Curve':<30s} {'Peak':>10s} {'Frac':>6s} {'Plateau (99%)':>20s} {'Width':>7s}")
    print("  " + "-" * 78)
    for label, info in plateau_info.items():
        plat = f"[{info['plateau_99_lo_frac']:.2f}, {info['plateau_99_hi_frac']:.2f}]"
        print(f"  {label:<30s} L{info['peak_layer']:<3d} ({info['peak_auroc']:.3f}) {info['peak_depth_frac']:>6.2f} {plat:>20s} {info['plateau_99_width']:>7.2f}")

    # Qwen3 sweep recommendation
    all_plateaus = [(info["plateau_99_lo_frac"], info["plateau_99_hi_frac"]) for info in plateau_info.values()]
    overall_lo = min(lo for lo, _ in all_plateaus)
    overall_hi = max(hi for _, hi in all_plateaus)
    results = {
        "plateau_info": plateau_info,
        "qwen3_recommendation": {
            "sweep_range": f"[{overall_lo:.2f}, {overall_hi:.2f}]",
            "for_1.7B_28L": [int(round(f * 28)) for f in np.arange(max(0.5, overall_lo - 0.05), min(1.0, overall_hi + 0.05), 0.07)],
            "for_8B_36L": [int(round(f * 36)) for f in np.arange(max(0.5, overall_lo - 0.05), min(1.0, overall_hi + 0.05), 0.07)],
        },
    }

    print(f"\n  Qwen3 sweep recommendation: depth fraction [{overall_lo:.2f}, {overall_hi:.2f}]")
    print(f"    1.7B (28L): layers {results['qwen3_recommendation']['for_1.7B_28L']}")
    print(f"    8B  (36L):  layers {results['qwen3_recommendation']['for_8B_36L']}")

    save_json(RESULTS_DIR / "FU5_depth_fraction.json", results)


# ═══════════════════════════════════════════════════════════════
# F6. Mean-pool residualized against token count
# ═══════════════════════════════════════════════════════════════

def FU6_meanpool_residualized():
    c = load_cache("math500_1p5b")
    y = c["y"].astype(bool)
    X_mean = c["X_mean"][:, STEERING_LAYER, :].astype(np.float64)
    X_prefill = c["X_prefill"][:, STEERING_LAYER, :].astype(np.float64)
    tokens = c["n_gen_tokens"].astype(float)

    neg_tokens = -tokens

    # Step 1: regress each hidden dim against -n_gen_tokens
    # X_resid[i, d] = X_mean[i, d] - (a_d * (-tokens[i]) + b_d)
    n, d = X_mean.shape
    X_resid = np.zeros_like(X_mean)

    for dim in range(d):
        x_col = neg_tokens.reshape(-1, 1)
        y_col = X_mean[:, dim]
        # Simple linear regression
        x_mean = neg_tokens.mean()
        y_mean = y_col.mean()
        ss_xy = ((neg_tokens - x_mean) * (y_col - y_mean)).sum()
        ss_xx = ((neg_tokens - x_mean) ** 2).sum()
        slope = ss_xy / (ss_xx + 1e-12)
        intercept = y_mean - slope * x_mean
        predicted = slope * neg_tokens + intercept
        X_resid[:, dim] = y_col - predicted

    # Step 2: fit DomProbe on residuals
    scores_resid = oof_dom_scores(X_resid, y)
    auroc_resid = float(roc_auc_score(y, scores_resid))
    ba_resid = bootstrap_auroc(y, scores_resid, n_resamples=2000)

    # Reference: raw mean-pool and prefill
    scores_mean = oof_dom_scores(X_mean, y)
    auroc_mean = float(roc_auc_score(y, scores_mean))
    ba_mean = bootstrap_auroc(y, scores_mean, n_resamples=2000)

    scores_prefill = oof_dom_scores(X_prefill, y)
    auroc_prefill = float(roc_auc_score(y, scores_prefill))
    ba_prefill = bootstrap_auroc(y, scores_prefill, n_resamples=2000)

    auroc_tc = float(roc_auc_score(y, -tokens))

    # Verify residualization worked
    r_resid_tc, _ = pearsonr(scores_resid, neg_tokens)
    r_mean_tc, _ = pearsonr(scores_mean, neg_tokens)

    print(f"  Raw mean-pool AUROC:          {auroc_mean:.4f} [{ba_mean['bca_lo']:.4f}, {ba_mean['bca_hi']:.4f}]")
    print(f"  Residualized mean-pool AUROC: {auroc_resid:.4f} [{ba_resid['bca_lo']:.4f}, {ba_resid['bca_hi']:.4f}]")
    print(f"  Prefill AUROC:                {auroc_prefill:.4f} [{ba_prefill['bca_lo']:.4f}, {ba_prefill['bca_hi']:.4f}]")
    print(f"  Token count AUROC:            {auroc_tc:.4f}")
    print(f"  r(resid_scores, -tokens):     {r_resid_tc:.4f} (should be ~0)")
    print(f"  r(raw_mean_scores, -tokens):  {r_mean_tc:.4f}")

    results = {
        "auroc_raw_meanpool": auroc_mean,
        "auroc_residualized": auroc_resid,
        "auroc_prefill": auroc_prefill,
        "auroc_token_count": auroc_tc,
        "ci_residualized": {"bca_lo": ba_resid["bca_lo"], "bca_hi": ba_resid["bca_hi"]},
        "ci_raw_meanpool": {"bca_lo": ba_mean["bca_lo"], "bca_hi": ba_mean["bca_hi"]},
        "ci_prefill": {"bca_lo": ba_prefill["bca_lo"], "bca_hi": ba_prefill["bca_hi"]},
        "r_resid_vs_tokens": float(r_resid_tc),
        "r_raw_vs_tokens": float(r_mean_tc),
    }

    if abs(auroc_resid - auroc_prefill) < 0.01:
        results["diagnosis"] = "meanpool_adds_nothing_beyond_length"
        print(f"\n  DIAGNOSIS: Residualized ≈ prefill ({auroc_resid:.4f} vs {auroc_prefill:.4f}) — mean-pool adds nothing beyond length")
    elif auroc_resid > auroc_prefill + 0.02:
        results["diagnosis"] = "meanpool_has_length_independent_signal"
        print(f"\n  DIAGNOSIS: Residualized > prefill by {auroc_resid - auroc_prefill:+.4f} — some length-independent generation signal exists")
    else:
        results["diagnosis"] = "small_residual_gain"
        print(f"\n  DIAGNOSIS: Small residual gain ({auroc_resid - auroc_prefill:+.4f}) — mostly length, tiny additional signal")

    save_json(RESULTS_DIR / "FU6_meanpool_residualized.json", results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    experiments = [
        ("FU4", "7B AUROC sanity check", FU4_7b_sanity),
        ("FU1", "Pre-generation leaderboard", FU1_pregen_leaderboard),
        ("FU2", "Cascade with combined gate", FU2_cascade_combined),
        ("FU3", "Compound cross-tier+domain", FU3_compound_transfer),
        ("FU5", "Depth-fraction overlay", FU5_depth_fraction),
        ("FU6", "Mean-pool residualized", FU6_meanpool_residualized),
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
    print("All follow-up experiments complete. Results in:")
    print(f"  {RESULTS_DIR}/")
    print(f"  {FIGS_DIR}/")


if __name__ == "__main__":
    main()
