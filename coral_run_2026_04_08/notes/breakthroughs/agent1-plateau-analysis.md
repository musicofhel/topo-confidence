---
creator: agent-1
created: 2026-04-08T13:38:00+00:00
---
# Plateau Analysis: 0.796-0.800 ceiling confirmed

## What I'm stuck on
11-feature set (PH + geometry + alignment) with LogReg. Score oscillates 0.763-0.796. Personal best 0.7963, global best 0.8004.

## Why it plateaued
1. **57 positives in 500 samples** — severe constraint on effective feature count
2. **11 features is the optimal count** — every 12th feature addition hurts (confirmed 5+ times)
3. **Feature swaps always regress** — even features with higher univariate AUROC hurt multivariate due to correlation with existing features
4. **last5_centroid_dist (0.505 univariate) is essential** — its orthogonality matters more than its individual predictive power

## What I've tried that failed (evals 61-67)
- 12th feature (layer_norm_slope): 0.7927
- Swap pairwise_dist_mean: 0.7858
- Swap H0_max_lifetime for velocity_H0_entropy: 0.7913
- Feature interaction (entropy*alignment): 0.7651 (WORST — redundant despite 0.726 uni)
- Deterministic subsampling: 0.7919
- Alignment slope (per-token trend): 0.7631

## Key insight
The 0.004 gap between 0.7963 and 0.8004 is CV noise (95% CI spans 0.12). All approaches that score 0.796+ are statistically equivalent.

## New direction: late_start=15
Agent-2 got 0.7977 and agent-3 got 0.7966 with layers 15: instead of 17:. More late layers = richer processing subspace. Simple parameter change worth testing.
