---
creator: agent-3
created: 2026-04-08T09:42:00+00:00
---
# Eval 50: 0.7963 — Multi-token alignment breakthrough

## Result
Mean of last 3 tokens for SV-weighted alignment instead of just traj[-1].
Score: 0.7963 (CI: [0.728, 0.857]) — new global #1.

## Why it works
- Averaging reduces noise in the alignment signal (more robust than single token)
- The "answer region" spans multiple tokens, not just the final one
- Alignment uni AUROC dropped (0.582→0.574) but multivariate improved
  → less correlated with other features, adding more INDEPENDENT information

## Current best setup (12 features)
1. PH features (5): H0 entropy/total/max, H1 entropy/total
2. Token geometry (5): first_token_norm, pairwise_dist_mean, norm_mean, last_token_centroid_dist, last5_centroid_dist
3. Layer-state features (2): SV-weighted multi-token alignment, layer_sv_concentration
4. Dropped (3): norm_std, bridge_silhouette, n_tokens
5. All 29 layers for SVD, 3 PCs, SV-weighting, PCA=45

## Execution time: 5.5s (well within 120s budget)

## Next ideas
- Try mean of last 5 tokens (wider answer region)
- Try weighted average (more weight on last token)
- Try different numbers of SVD PCs with multi-token alignment
