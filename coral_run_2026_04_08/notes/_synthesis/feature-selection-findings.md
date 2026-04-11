---
creator: agent-2
updated: 2026-04-08T08:25:00+00:00
updater: agent-3
---
# Feature Selection and Pipeline Findings -- Synthesis of 90+ evals across 3 agents

## Current best: 0.7789 (agent-3, 14 features, PCA=45, cosine 3-PC subspace alignment)

## Evolution of the ceiling
1. Baseline (7 features): 0.713
2. Trajectory-only (evals 1-40): 0.770 (PCA=45, 13 features)
3. Single-direction alignment (eval ~55): 0.777 (agent-3, dot(last_token, layer_dir))
4. 3-PC subspace alignment (eval 65): 0.778 (agent-2, norm(top3_PCs @ last_token))
5. Cosine subspace alignment (eval 82): **0.779** (agent-3, proj_norm/token_norm)

## The 14-feature set is LOCKED (VERY HIGH CONFIDENCE)
Across 90+ evals, 3 agents have independently confirmed:
- Cannot add 15th feature: path_tortuosity (0.712 uni), crystallization, coherence,
  layer_ph, late_convergence ALL give ~0.775 or worse
- Cannot swap ANY feature: H0_max_lifetime (0.614) -> path_tortuosity (0.712) = 0.770
  last5_centroid_dist (0.507) -> traj_speed_std (0.586) = 0.743
  last5_centroid_dist (0.507) -> norm_std = 0.744 (3 agents confirmed)
- Cannot preprocess: rank-transform (0.729), z-score (0.731), log-transform (0.761)
- Cannot change PCA: 45 is optimal (40=0.769, 50=0.764, 43/47 = noise)
- Cannot change subsampling: FPS (0.755), biased (0.769), 80pt (0.741), 120pt (0.761)
- One exception: agent-1 replaced H0_n_features (r=0.985 with H0_entropy) with
  norm_std → 0.7788. Only works because the replaced features were 98.5% correlated.

## Cross-modal alignment details
- SVD of per-problem centered layer states (29 x 1536)
- Top 3 PCs define processing subspace (K=3 optimal)
- Cosine alignment: ||Vt[:3] @ last_token|| / ||last_token|| 
- Captures what fraction of last token lies in the layer processing manifold

## Feature count ceiling (VERY HIGH CONFIDENCE)
| Count | Best AUROC | Status |
|-------|-----------|--------|
| 13 | 0.770 | Optimal before cross-modal |
| 14 | **0.779** | Current optimal |
| 15 | 0.775 | Consistent wall across 10+ attempts |

## ALL confirmed dead ends
- Feature additions, swaps, preprocessing, PCA changes, subsampling changes
- PCA on layer_states instead of tokens: 0.748 (disaster)
- PCA whitening: 0.698 (disaster)
- Different PC counts: 1/2/4/5 all worse or equal to 3
- Layer PH features on top of base: always 0.774-0.775

## Remaining untried directions (low confidence)
1. Non-linear kernel for PH distance matrix
2. Ensemble of PCA dimensions for PH
3. Per-problem adaptive features
4. Fundamentally different dimensionality reduction (UMAP not available)
