---
creator: agent-3
created: 2026-04-08T04:36:00+00:00
---
# Feature Engineering Findings — Synthesis of 20 evals across 3 agents

## Top-level conclusion
Best AUROC: **0.7581** (13 features). The 13-feature set is at a local optimum —
ALL modifications by ALL agents hurt. To improve further, must change the
underlying representation (PCA dims, distance metric, subsample), not just swap features.

## What works
1. **Selective addition** of high-signal features (AUROC > 0.60) to baseline
2. **Position-dependent features** — biggest single breakthrough (+0.025 from 0.733 → 0.758)
3. **H0 features** dominate over H1 for this task
4. **Interaction effects matter** — last5_centroid_dist has 0.502 univariate but contributes +0.019 in combination

## What doesn't work
1. **Raw layer-state features** — dominated by model architecture, near-zero problem variance (3 agents confirmed)
2. **Adding many features** — with 57 positives, >13 features overfits logistic regression
3. **PCA variance features** — redundant with PH features, hurt multivariate AUROC
4. **Standardization** — neutral effect (LogisticRegression handles scale internally)
5. **H1_max_lifetime** — consistently weakest PH feature (0.599), replaced with H0_max_lifetime + H1_total_persistence

## Current optimal feature set (13 features, 0.7581)
| Feature | Univariate AUROC | Category |
|---------|-----------------|----------|
| n_tokens | 0.723 | Cloud size |
| H0_n_features | 0.722 | PH-H0 |
| H0_persistence_entropy | 0.717 | PH-H0 |
| H0_total_persistence | 0.696 | PH-H0 |
| H1_n_features | 0.687 | PH-H1 |
| H1_persistence_entropy | 0.682 | PH-H1 |
| last_token_centroid_dist | 0.650 | Position |
| H1_total_persistence | 0.646 | PH-H1 |
| H0_max_lifetime | 0.624 | PH-H0 |
| bridge_silhouette | 0.618 | Clustering |
| pairwise_dist_mean | 0.612 | Cloud geometry |
| norm_mean | 0.601 | Cloud geometry |
| last5_centroid_dist | 0.502 | Position (interaction) |

## Untried approaches with potential
1. **Change PCA dimensions** (50 instead of 30) — different representation entirely
2. **Cosine distance in ripser** — captures angular structure, not just magnitude
3. **Subsample=200** — more points for finer PH resolution
4. **Features in raw 1536D** — some metrics may work better before PCA
5. **Persistence landscapes/Betti curves** — WHERE in filtration events happen
6. **Normalized layer deviations** — z-score layer states across problems (partially tried, weak signal)
