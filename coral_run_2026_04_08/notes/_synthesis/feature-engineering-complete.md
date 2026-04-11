---
creator: agent-1
created: 2026-04-08T13:35:00+00:00
---
# Feature Engineering Synthesis: 190+ Evals Across 3 Agents

## Conclusion
The optimal feature set has been found: 11 features, PCA=48, with 6-PC cosine 
alignment on late layers (17:). Score ceiling: ~0.800 (CI: [0.73, 0.86]).

## The 11 Optimal Features (in order of univariate AUROC)
1. H0_persistence_entropy: 0.717 — strongest individual predictor
2. H0_total_persistence: 0.695
3. H1_persistence_entropy: 0.682
4. mean_step_size/pairwise_dist_mean: 0.658/0.609 — step_size higher uni but correlated
5. H1_total_persistence: 0.648
6. last_token_centroid_dist: 0.640
7. H0_max_lifetime: 0.617
8. last_token_layer_alignment: 0.606 (6-PC cosine, late layers)
9. norm_mean: 0.596
10. first_token_norm: 0.575
11. last5_centroid_dist: 0.507 — lowest univariate but IRREPLACEABLE

## Feature Count: U-shaped Optimum
- 7 features (baseline): 0.713
- 11 features: 0.796-0.800 (sweet spot)
- 12 features: 0.793-0.796 (neutral to worse)
- 14 features: 0.787 (too many)

## Why 11 is the Ceiling
57 positive examples in 500 samples. LogisticRegression(balanced) gives each 
positive ~7.8x weight. With 11 coefficients + intercept = 12 parameters, 
the effective sample size for the minority class is ~57*7.8 ≈ 445 weighted 
observations. Adding more features risks overfitting the minority class.

## Key Lessons (validated across all agents)
1. **Univariate AUROC doesn't predict multivariate value.** mean_step_size 
   (0.658 uni) HURT when swapped for pairwise_dist_mean (0.609 uni) because 
   it was more correlated with existing features.
2. **Independence > quality** for the Nth feature. Correlation-swap technique 
   (replacing r>0.95 correlated features) gave biggest gain (+0.008).
3. **Late layers > all layers** for alignment. Layers 17-28 capture output-
   relevant processing; early layers add noise.
4. **6 PCs > 3 PCs** for cosine alignment. More PCs = better subspace approximation.
5. **Cosine > SV-weighted** alignment. Simpler is better.
6. **PCA=48 is optimal.** Range 45-50 all similar; outside that range degrades.

## Dead Ends (do not retry)
- Manhattan/L1 distance for PH: -0.028
- Cosine distance for PH: -0.052
- Maxmin subsampling: neutral to negative
- Position-augmented PH: neutral
- Log transforms: -0.015+
- Feature interactions: dilution
- UMAP: untested but likely too slow
- Any 12th feature: wall confirmed 5+ times

## What Might Still Work (low confidence, <15% each)
1. Different PCA fitting (e.g., fit on correct problems only — but we don't know labels)
2. Fundamentally different DR before PH (kernel PCA, random projection)
3. Multi-resolution PH (PH at two PCA scales)
4. Better alignment: different token weighting, different normalization
