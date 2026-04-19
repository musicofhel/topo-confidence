#!/usr/bin/env python3
"""Phase 6.5 Step 3: Cross-Benchmark / Cross-Model Analysis (Deconfounded).

CPU ONLY. Run after step1 (GSM8K) and step2 (MATH-7B).

Compares deconfounded results (max_new_tokens=1024) against:
- 1.5B × MATH-500 (Phase 1 rebuild, AUROC 0.796 — unchanged)
- 1.5B × GSM8K at 256 tokens (Phase 3, confounded)
- 7B × MATH-500 at 256 tokens (Phase 3, confounded)

Produces transfer experiments, feature correlations, and updated claim assessment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    BASELINE_CORRECT_V2_PATH,
    LR_PARAMS,
    bootstrap_auroc,
    get_abc_column_indices,
    get_train_holdout_indices,
    load_features_with_sss_fix,
    logger,
)

# Directories
PHASE1_DIR = Path(__file__).parent.parent / "phase1_prompt_model"
PHASE2_DIR = Path(__file__).parent.parent / "phase2_completion"
PHASE3_DIR = Path(__file__).parent.parent / "phase3_cross_benchmark"
P65_DIR = Path(__file__).parent
P65_GSM8K = P65_DIR / "gsm8k"
P65_MATH7B = P65_DIR / "math7b"

# Phase 3 directories (truncated results for comparison)
P3_GSM8K = PHASE3_DIR / "gsm8k"
P3_MATH7B = PHASE3_DIR / "math7b"


def load_math_1_5b():
    """Load 1.5B × MATH-500 results (Phase 1 rebuild, unchanged)."""
    with open(PHASE1_DIR / "holdout_metrics_v2.json") as f:
        metrics = json.load(f)

    correct = np.load(BASELINE_CORRECT_V2_PATH)
    train_idx, holdout_idx = get_train_holdout_indices()

    old_correct = np.load(
        Path(__file__).parent.parent.parent / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
    )
    X_train, X_holdout, _, _, _, _ = load_features_with_sss_fix(baseline_correct=old_correct)

    selection = {}
    sel_path = PHASE2_DIR / "holdout_results_v2.json"
    if sel_path.exists():
        with open(sel_path) as f:
            selection = json.load(f)

    return {
        "label": "1.5B × MATH-500",
        "max_new_tokens": 256,
        "greedy_acc": float(correct.mean()),
        "greedy_correct": int(correct.sum()),
        "n_problems": len(correct),
        "topo_auroc": metrics["auroc"],
        "topo_auroc_ci": [metrics.get("ci_lo", 0), metrics.get("ci_hi", 1)],
        "features_train": X_train,
        "features_holdout": X_holdout,
        "correct": correct,
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "selection": selection,
    }


def _load_benchmark_dir(d, label, idx_key="test_idx"):
    """Generic loader for GSM8K or 7B MATH directories."""
    if not d.exists():
        return None
    summary_path = d / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    model_results_path = d / "model_results.json"
    model_results = {}
    if model_results_path.exists():
        with open(model_results_path) as f:
            model_results = json.load(f)
    sel_path = d / "selection_results.json"
    selection = {}
    if sel_path.exists():
        with open(sel_path) as f:
            selection = json.load(f)

    # Binary files may not exist (gitignored) — handle gracefully
    features_path = d / "features.npy"
    correct_path = d / "baseline_correct.npy"
    train_path = d / "train_idx.npy"
    eval_path = d / f"{idx_key}.npy"
    if not all(p.exists() for p in [features_path, correct_path, train_path, eval_path]):
        logger.warning("  %s: binary files missing (gitignored), skipping transfer experiments", label)
        return {
            "label": label,
            "max_new_tokens": summary.get("max_new_tokens", 256),
            "greedy_acc": summary["greedy_accuracy"],
            "greedy_correct": summary["greedy_correct"],
            "n_problems": summary["n_problems"],
            "topo_auroc": summary["topo_auroc"],
            "topo_auroc_ci": summary.get("topo_auroc_ci", [0, 1]),
            "best_baseline": model_results.get("best_baseline"),
            "best_baseline_auroc": model_results.get("best_baseline_auroc"),
            "topo_vs_best_gap": model_results.get("topo_vs_best_gap"),
            "selection": selection,
        }

    features = np.load(features_path)
    correct = np.load(correct_path)
    train_idx = np.load(train_path)
    eval_idx = np.load(eval_path)

    return {
        "label": label,
        "max_new_tokens": summary.get("max_new_tokens", 256),
        "greedy_acc": summary["greedy_accuracy"],
        "greedy_correct": summary["greedy_correct"],
        "n_problems": summary["n_problems"],
        "topo_auroc": summary["topo_auroc"],
        "topo_auroc_ci": summary.get("topo_auroc_ci", [0, 1]),
        "best_baseline": model_results.get("best_baseline"),
        "best_baseline_auroc": model_results.get("best_baseline_auroc"),
        "topo_vs_best_gap": model_results.get("topo_vs_best_gap"),
        "features": features,
        "correct": correct,
        "train_idx": train_idx,
        "eval_idx": eval_idx,
        "selection": selection,
    }


def transfer_experiment(configs, feature_names):
    """Train on one config, evaluate on another. All pairwise combinations."""
    results = {}
    abc_idx = get_abc_column_indices(feature_names)

    # Filter to configs that have features (binary files present)
    configs_with_features = [c for c in configs if "features" in c or "features_train" in c]
    if len(configs_with_features) < 2:
        logger.warning("  Not enough configs with features for transfer experiments")
        return results

    def get_train_eval(c):
        """Get train/eval features and labels."""
        if "features_train" in c:  # 1.5B MATH (pre-split)
            return c["features_train"][:, abc_idx], c["correct"][c["train_idx"]], \
                   c["features_holdout"][:, abc_idx], c["correct"][c["holdout_idx"]]
        else:
            return c["features"][c["train_idx"]][:, abc_idx], c["correct"][c["train_idx"]], \
                   c["features"][c["eval_idx"]][:, abc_idx], c["correct"][c["eval_idx"]]

    for i, src in enumerate(configs_with_features):
        X_tr, y_tr, _, _ = get_train_eval(src)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_tr_s, y_tr.astype(int))

        for j, tgt in enumerate(configs_with_features):
            if i == j:
                continue
            _, _, X_te, y_te = get_train_eval(tgt)
            X_te_s = scaler.transform(X_te)
            scores = clf.predict_proba(X_te_s)[:, 1]

            if len(np.unique(y_te)) >= 2:
                auroc = roc_auc_score(y_te.astype(int), scores)
                _, ci_lo, ci_hi = bootstrap_auroc(y_te.astype(int), scores)
            else:
                auroc, ci_lo, ci_hi = None, None, None

            key = f"{src['label']}_to_{tgt['label']}"
            results[key] = {
                "auroc": float(auroc) if auroc else None,
                "ci": [float(ci_lo), float(ci_hi)] if ci_lo else None,
            }
            logger.info("  Transfer %s: AUROC = %s", key,
                        f"{auroc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]" if auroc else "N/A")

    return results


def feature_rank_correlation(configs, feature_names):
    """Spearman correlation of per-feature Cohen's d across configs."""
    results = {}

    def cohens_d(features, correct):
        ds = []
        for fi in range(features.shape[1]):
            vals_c = features[correct.astype(bool), fi]
            vals_w = features[~correct.astype(bool), fi]
            if len(vals_c) > 1 and len(vals_w) > 1:
                pooled = np.sqrt(
                    ((len(vals_c) - 1) * vals_c.std() ** 2 +
                     (len(vals_w) - 1) * vals_w.std() ** 2) /
                    (len(vals_c) + len(vals_w) - 2)
                )
                d = (vals_c.mean() - vals_w.mean()) / max(pooled, 1e-10)
            else:
                d = 0.0
            ds.append(d)
        return np.array(ds)

    ds_per_config = {}
    for c in configs:
        if "features_train" in c:
            feats = c["features_train"]
            correct = c["correct"][c["train_idx"]]
        elif "features" in c:
            feats = c["features"][c["train_idx"]]
            correct = c["correct"][c["train_idx"]]
        else:
            continue  # No features for this config
        ds_per_config[c["label"]] = cohens_d(feats, correct)

    labels = list(ds_per_config.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rho, p = spearmanr(ds_per_config[labels[i]], ds_per_config[labels[j]])
            key = f"{labels[i]}_vs_{labels[j]}"
            results[key] = {"rho": float(rho), "p": float(p)}
            logger.info("  Feature corr %s: rho=%.3f (p=%.2e)", key, rho, p)

    return results


def build_comparison_table(configs):
    """Build comparison table across all configs."""
    rows = []
    for c in configs:
        row = {
            "config": c["label"],
            "max_new_tokens": c.get("max_new_tokens", 256),
            "greedy_acc": f"{c['greedy_correct']}/{c['n_problems']} ({c['greedy_acc']:.1%})",
            "topo_auroc": f"{c['topo_auroc']:.4f}",
            "best_baseline": f"{c.get('best_baseline', '—')} ({c.get('best_baseline_auroc', 0):.4f})" if c.get('best_baseline') else "—",
            "gap": f"{c.get('topo_vs_best_gap', 0):+.4f}" if c.get("topo_vs_best_gap") is not None else "—",
        }
        sel = c.get("selection", {})
        mv = sel.get("ungated_mv", {})
        row["mv_net_gain"] = f"+{mv.get('net_gain', '?')}" if mv else "—"
        row["oracle"] = f"+{sel.get('oracle_ceiling', '?')}" if sel else "—"
        rows.append(row)
    return rows


def claim_assessment(r1_5b, gsm8k_65, math7b_65, gsm8k_p3=None, math7b_p3=None):
    """Updated claim assessment with deconfounded numbers."""
    claims = []

    # 1. AUROC 0.796 (unchanged)
    claims.append({
        "claim": "1.5B MATH topo AUROC = 0.796",
        "status": "ESTABLISHED",
        "evidence": f"Unchanged from Phase 1 rebuild. No truncation confound (20.8% correct).",
    })

    # 2. GSM8K generalization
    if gsm8k_65:
        old_auroc = gsm8k_p3["topo_auroc"] if gsm8k_p3 else 0.731
        new_auroc = gsm8k_65["topo_auroc"]
        status = "SURVIVES" if new_auroc > 0.55 else "WEAKENED"
        claims.append({
            "claim": "GSM8K cross-benchmark generalization",
            "status": status,
            "evidence": (
                f"Deconfounded AUROC: {new_auroc:.3f} (was {old_auroc:.3f} with truncation confound). "
                f"Accuracy: {gsm8k_65['greedy_acc']:.1%} (was {gsm8k_p3['greedy_acc']:.1%} truncated)."
            ),
        })

    # 3. 7B generalization
    if math7b_65:
        old_auroc = math7b_p3["topo_auroc"] if math7b_p3 else 0.682
        new_auroc = math7b_65["topo_auroc"]
        status = "SURVIVES" if new_auroc > 0.55 else "WEAKENED"
        claims.append({
            "claim": "7B cross-model generalization",
            "status": status,
            "evidence": (
                f"Deconfounded AUROC: {new_auroc:.3f} (was {old_auroc:.3f} with truncation confound). "
                f"Accuracy: {math7b_65['greedy_acc']:.1%} (was {math7b_p3['greedy_acc']:.1%} truncated)."
            ),
        })

    # 4. Truncation was the sole cause of low accuracy
    if gsm8k_65 and math7b_65:
        gsm_fixed = gsm8k_65["greedy_acc"] > 0.70
        m7b_fixed = math7b_65["greedy_acc"] > 0.70
        if gsm_fixed and m7b_fixed:
            status = "CONFIRMED"
            ev = "Both configs now match expected model capabilities."
        elif gsm_fixed or m7b_fixed:
            status = "PARTIALLY CONFIRMED"
            ev = "One config fixed, other still low — investigate further."
        else:
            status = "REFUTED"
            ev = "Neither config reached expected accuracy — truncation was not the sole cause."
        claims.append({
            "claim": "max_new_tokens=256 truncation was sole cause of low accuracy",
            "status": status,
            "evidence": ev,
        })

    # 5. Topo adds value beyond truncation detection
    if gsm8k_65:
        gap = gsm8k_65.get("topo_vs_best_gap", 0)
        status = "SURVIVES" if gap > 0.02 else ("MARGINAL" if gap > 0 else "REFUTED")
        claims.append({
            "claim": "Topo features add value beyond logprob baselines (deconfounded)",
            "status": status,
            "evidence": (
                f"GSM8K gap: {gap:+.3f}. "
                f"With truncation removed, baselines may improve too."
            ),
        })

    return claims


def main():
    logger.info("=" * 70)
    logger.info("PHASE 6.5 STEP 3: CROSS-BENCHMARK ANALYSIS (DECONFOUNDED)")
    logger.info("=" * 70)

    # Load all results
    logger.info("\n--- Loading results ---")

    r1_5b = load_math_1_5b()
    logger.info("1.5B MATH (Phase 1): %.1f%%, AUROC %.4f", 100 * r1_5b["greedy_acc"], r1_5b["topo_auroc"])

    gsm8k_65 = _load_benchmark_dir(P65_GSM8K, "1.5B × GSM8K (1024)", idx_key="test_idx")
    if gsm8k_65:
        logger.info("GSM8K (1024): %.1f%%, AUROC %.4f", 100 * gsm8k_65["greedy_acc"], gsm8k_65["topo_auroc"])

    math7b_65 = _load_benchmark_dir(P65_MATH7B, "7B × MATH (1024)", idx_key="holdout_idx")
    if math7b_65:
        logger.info("7B MATH (1024): %.1f%%, AUROC %.4f", 100 * math7b_65["greedy_acc"], math7b_65["topo_auroc"])

    gsm8k_p3 = _load_benchmark_dir(P3_GSM8K, "1.5B × GSM8K (256)", idx_key="test_idx")
    math7b_p3 = _load_benchmark_dir(P3_MATH7B, "7B × MATH (256)", idx_key="holdout_idx")

    # Feature names
    feature_names = None
    for src_dir in [P65_GSM8K, P65_MATH7B, P3_GSM8K]:
        fn_path = src_dir / "feature_names.json"
        if fn_path.exists():
            with open(fn_path) as f:
                feature_names = json.load(f)
            break

    # --- Comparison Table ---
    logger.info("\n--- Comparison Table ---")
    configs_for_table = [r1_5b]
    if gsm8k_p3:
        configs_for_table.append(gsm8k_p3)
    if gsm8k_65:
        configs_for_table.append(gsm8k_65)
    if math7b_p3:
        configs_for_table.append(math7b_p3)
    if math7b_65:
        configs_for_table.append(math7b_65)

    table = build_comparison_table(configs_for_table)
    print(f"\n{'Config':<30} {'Accuracy':>18} {'AUROC':>8} {'Gap':>8} {'MV':>6} {'Oracle':>8}")
    print("-" * 82)
    for row in table:
        print(f"{row['config']:<30} {row['greedy_acc']:>18} {row['topo_auroc']:>8} {row['gap']:>8} {row['mv_net_gain']:>6} {row['oracle']:>8}")

    # --- Transfer experiments ---
    transfer_results = {}
    if feature_names:
        logger.info("\n--- Transfer Experiments ---")
        configs_for_transfer = [r1_5b]
        if gsm8k_65:
            configs_for_transfer.append(gsm8k_65)
        if math7b_65:
            configs_for_transfer.append(math7b_65)

        if len(configs_for_transfer) >= 2:
            transfer_results = transfer_experiment(configs_for_transfer, feature_names)

    # --- Feature rank correlations ---
    rank_corr_results = {}
    if feature_names:
        logger.info("\n--- Feature Rank Correlations ---")
        configs_for_corr = [r1_5b]
        if gsm8k_65:
            configs_for_corr.append(gsm8k_65)
        if math7b_65:
            configs_for_corr.append(math7b_65)

        if len(configs_for_corr) >= 2:
            rank_corr_results = feature_rank_correlation(configs_for_corr, feature_names)

    # --- Claim assessment ---
    logger.info("\n--- Claim Assessment ---")
    claims = claim_assessment(r1_5b, gsm8k_65, math7b_65, gsm8k_p3, math7b_p3)
    for c in claims:
        logger.info("  [%s] %s", c["status"], c["claim"])
        logger.info("    %s", c["evidence"])

    # --- Save ---
    all_results = {
        "comparison_table": table,
        "transfer": transfer_results,
        "feature_rank_correlations": rank_corr_results,
        "claims": claims,
        "phase3_vs_phase65": {
            "gsm8k": {
                "p3_auroc": gsm8k_p3["topo_auroc"] if gsm8k_p3 else None,
                "p3_acc": gsm8k_p3["greedy_acc"] if gsm8k_p3 else None,
                "p65_auroc": gsm8k_65["topo_auroc"] if gsm8k_65 else None,
                "p65_acc": gsm8k_65["greedy_acc"] if gsm8k_65 else None,
            },
            "math7b": {
                "p3_auroc": math7b_p3["topo_auroc"] if math7b_p3 else None,
                "p3_acc": math7b_p3["greedy_acc"] if math7b_p3 else None,
                "p65_auroc": math7b_65["topo_auroc"] if math7b_65 else None,
                "p65_acc": math7b_65["greedy_acc"] if math7b_65 else None,
            },
        },
    }
    with open(P65_DIR / "cross_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\nSaved to %s", P65_DIR / "cross_analysis.json")


if __name__ == "__main__":
    main()
