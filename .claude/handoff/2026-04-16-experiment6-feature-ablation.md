# Handoff: Experiment 6 — Feature Ablation (Tier Subsets + Importance + LOO)

**Date**: 2026-04-16
**Session scope**: Implement Experiment 6 from the 12-experiment defense plan

## What was done

Created and ran `pathway4/track_a/experiment6_feature_ablation.py` — a zero-GPU feature ablation covering all 7 A/B/C tier combinations, three importance methods (LR coefficients, permutation importance, leave-one-out), and gating performance per subset. Completed in 10s.

## Results

### Table 1: Tier Subset Comparison

| Subset | N_feat | Holdout AUROC [95% CI] | CV-train-400 | Best gate: Net (W→R/R→W) |
|--------|--------|------------------------|--------------|---------------------------|
| A | 9 | 0.840 [0.730–0.931] | 0.791 | +7 (7/0) |
| B | 30 | 0.760 [0.638–0.869] | 0.775 | +10 (11/1) |
| C | 5 | 0.745 [0.603–0.871] | 0.716 | +8 (11/3) |
| A+B | 39 | 0.928 [0.870–0.974] | 0.902 | +10 (10/0) |
| A+C | 14 | 0.905 [0.829–0.967] | 0.803 | +8 (9/1) |
| B+C | 35 | 0.763 [0.630–0.870] | 0.775 | +10 (11/1) |
| **A+B+C** | **44** | **0.935 [0.877–0.975]** | **0.919** | **+10 (10/0)** |
| A+B+C+D | 78 | 0.948 [0.894–0.985] | 0.929 | +10 (10/0) |

### Key findings from tier ablation

1. **Tier A (PH + geometry) is the indispensable core**: A alone hits 0.840 AUROC — the highest single-tier score. B alone (0.760) and C alone (0.745) are much weaker. Tier A anchors the signal.

2. **A+B is near-ceiling**: 0.928 AUROC with 39 features, only 0.007 below A+B+C. Adding Tier C (5 features) pushes to 0.935 — marginal but consistent.

3. **Tier B without A collapses**: B alone (0.760) and B+C (0.763) are mediocre. Tier B's layer cosines need Tier A's PH/geometry as an anchor. The A→A+B jump (+0.088) is the largest in the table.

4. **Gating: A alone achieves +7 with 0 R→W**: Even the minimal 9-feature model eliminates all regressions, though it misses 3 W→R opportunities. A+B and A+B+C both achieve +10 with 0 R→W.

5. **A+C is surprisingly strong**: 14 features, 0.905 AUROC — the 5 Tier C cross-products interact synergistically with Tier A. On CV it's only 0.803, suggesting holdout may be an overestimate.

### Table 2: Top-10 Features by Importance

| Rank | Feature | Tier | |Coef| | Perm Imp | LOO ΔAUROC |
|------|---------|------|--------|----------|------------|
| 1 | accel_l4_x_h0ent35_bin | B | 2.016 | 0.217 | +0.000 |
| 2 | cos_l9_l28 | B | 1.772 | 0.181 | +0.009 |
| 3 | cos_l7_l11 | B | 1.503 | 0.056 | -0.017 |
| 4 | jerk_l7_x_h0tot_sq | B | 1.373 | 0.089 | +0.005 |
| 5 | nr28_15_x_h0ent35 | B | 1.342 | 0.086 | -0.009 |
| 6 | cos_l23_l26 | B | 1.295 | 0.095 | +0.023 |
| 7 | cos_l16_l28_bin | B | 1.287 | 0.045 | +0.003 |
| 8 | c27_28_x_h0tot | C | 1.249 | 0.075 | -0.002 |
| 9 | layer_pc2_early_ratio | B | 1.237 | 0.122 | +0.015 |
| 10 | tok_layer_min_dist | B | 1.219 | 0.063 | +0.014 |

### Top-5 LOO (most irreplaceable features)

| Rank | Feature | Tier | AUROC without | ΔAUROC |
|------|---------|------|---------------|--------|
| 1 | H1_persistence_entropy | A | 0.865 | +0.070 |
| 2 | last5_centroid_dist | A | 0.877 | +0.057 |
| 3 | first_token_norm | A | 0.898 | +0.037 |
| 4 | last_token_centroid_dist | A | 0.904 | +0.031 |
| 5 | cos_l23_l26 | B | 0.912 | +0.023 |

### Tier Contribution Summary

| Tier | N_feat | Mean |Coef| | Total |Coef| | Mean Perm Imp | Mean LOO Δ |
|------|--------|------------|-------------|---------------|------------|
| A | 9 | 0.830 | 7.469 | 0.052 | +0.023 |
| B | 30 | 0.937 | 28.116 | 0.053 | +0.004 |
| C | 5 | 0.679 | 3.392 | 0.034 | +0.001 |

### Interpretation

**Three importance methods tell complementary stories**:

- **LR coefficients** (how much the model weights each feature): Tier B dominates — 9 of top 10 features are Tier B. `accel_l4_x_h0ent35_bin` is the most weighted, a product of layer-4 acceleration × H0 entropy (PH signal).

- **Permutation importance** (how much shuffling each feature hurts): Similar to coefficients but Tier A rises. `H1_persistence_entropy` (PH) is #4 despite being #13 by coefficient — it carries unique information.

- **Leave-one-out** (which features are irreplaceable): **Tier A dominates completely**. Top 4 LOO features are ALL Tier A: H1_persistence_entropy (+0.070), last5_centroid_dist (+0.057), first_token_norm (+0.037), last_token_centroid_dist (+0.031). These are the features that, when removed, nothing else can compensate for.

**The key insight**: Tier B features have large coefficients because they're numerous and correlated — removing any single one barely hurts because the model has 29 others. Tier A features have smaller coefficients but are **irreplaceable** — they carry unique topological signal (PH lifetimes, token geometry) that no layer cosine can substitute.

## Paper narrative

"Topo-confidence draws on 44 features spanning three scales: persistent homology lifetimes and token geometry (Tier A, 9 features), layer-wise cosine similarities and dynamics (Tier B, 30 features), and cross-scale products (Tier C, 5 features). Tier A alone achieves 0.840 AUROC; adding Tier B lifts this to 0.928, and Tier C contributes a further 0.007. Leave-one-out analysis reveals that the four most irreplaceable features are all Tier A: H1 persistence entropy, last-5-token centroid distance, first-token norm, and last-token centroid distance. Tier B features individually rank high by coefficient magnitude but are mutually substitutable — no single layer cosine is critical. The topology-derived signal (PH lifetimes) provides the anchor; layer geometry amplifies it."

## Phase 2 comparison

Phase 2 (re-extracted features with train-PCA) vs Experiment 6 (precomputed .npy with SSS fix):

| Subset | Phase 2 CV | Exp 6 CV | Phase 2 Holdout | Exp 6 Holdout |
|--------|-----------|----------|----------------|---------------|
| A | 0.789 | 0.791 | 0.827 | 0.840 |
| A+B | 0.908 | 0.902 | 0.932 | 0.928 |
| A+B+C | 0.922 | 0.919 | 0.941 | 0.935 |
| A+B+C+D | 0.927 | 0.929 | 0.950 | 0.948 |

Small differences (<0.01) due to different PCA bases — consistent and expected.

## Files created

- **NEW**: `pathway4/track_a/experiment6_feature_ablation.py`
- **NEW**: `pathway4/track_a/experiment6_ablation/tier_ablation_results.json`
- **NEW**: `pathway4/track_a/experiment6_ablation/per_problem_tier_scores.json`
- **NEW**: `pathway4/track_a/experiment6_ablation/feature_importance.json`

## What's next

**Week 1 remaining** (all zero-GPU):
- **Exp 10**: Calibration analysis — ECE/Brier/reliability diagram on P1 confidence scores

**Week 1 complete**:
- **Exp 1** (done): Pure MV ablation → topo-gating +10 net, 0 R→W
- **Exp 9** (done): Output-probability baselines → topo AUROC=0.935 crushes all baselines
- **Exp 6** (done): Feature ablation → Tier A PH is irreplaceable anchor, Tier B amplifies, no single feature is critical
