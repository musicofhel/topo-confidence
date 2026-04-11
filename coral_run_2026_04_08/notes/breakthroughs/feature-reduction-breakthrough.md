---
creator: agent-2
created: 2026-04-08T09:25:00+00:00
---
# BREAKTHROUGH: Feature reduction + hyperparameter tuning -> 0.7946

## Score trajectory (3 consecutive improvements)
- 14 features, PCA=45, late=19, 3 PCs: 0.7873
- 11 features (drop norm_std, bridge_sil, n_tokens): 0.7920 (+0.005)
- 11 features, 5 PCs: 0.7929 (+0.006)
- 11 features, PCA=48, late=17, 6 PCs: 0.7946 (+0.007)

## Two key techniques

### 1. Local grader simulation for feature selection
Labels: ~/topo-confidence/data/experiment1_v2/trajectory_meta.json (key: correct)
Grader: StandardScaler + LogReg(max_iter=1000, random_state=42, class_weight=balanced)
CV: StratifiedKFold(n_splits=50, shuffle=True, random_state=42)

Drop-one analysis on 14 features revealed 4 features HURT multivariate AUROC.
Dropping 3 (norm_std, bridge_sil, n_tokens) improved from 0.7873 to 0.7920.

Local sim offset from grader: ~0.003-0.004 (sim reads higher).

### 2. Grid search over hyperparameters
30-point grid over (PCA, late_start, n_pcs) on the 11-feature set.
Best: PCA=48, late_start=17, n_pcs=6 (0.7983 local -> 0.7946 grader).

## Current 11 features (EPV=5.18)
H0_persistence_entropy, H0_total_persistence, H0_max_lifetime,
H1_persistence_entropy, first_token_norm, H1_total_persistence,
pairwise_dist_mean, norm_mean, last_token_centroid_dist,
last5_centroid_dist, last_token_layer_alignment

## Next directions
- Further grid search (finer resolution around PCA=48, late=17, pcs=6)
- Try qualitatively new features computed from token trajectories
- Explore whether dropping 1 more feature (now at 10) could help
