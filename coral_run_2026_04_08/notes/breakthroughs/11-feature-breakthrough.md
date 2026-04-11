---
creator: agent-1
created: 2026-04-08T09:29:00+00:00
---
# 11-feature breakthrough: 0.787 → 0.791-0.793

## What happened
Agent-2 discovered that dropping 3 features (norm_std, bridge_silhouette, 
n_tokens) from 14→11 features improved from 0.787 to 0.792. This was
shocking because:
- bridge_silhouette was the biggest position-feature breakthrough
- n_tokens had the highest univariate AUROC (0.723)
- All previous "all features are load-bearing" analysis was wrong

## Why previous feature-drop tests failed
My eval #42 dropped first_token_norm + last5_centroid_dist → 0.754 (-0.033).
Agent-2 dropped norm_std + bridge_silhouette + n_tokens → 0.792 (+0.005).
**WHICH features you drop matters enormously.** Dropping redundant/noisy 
features helps; dropping truly orthogonal ones hurts.

## Why these 3 are droppable
- **norm_std**: was a correlation-swap replacement, not original signal
- **bridge_silhouette**: relies on KMeans (stochastic, noisy)
- **n_tokens**: highest univariate but likely correlated with PH features 
  (more tokens → different PH characteristics)

## Combined with SV-weighted alignment
Agent-3 and I both combined 11 features + SV-weighted alignment:
- Agent-3: 0.7933 (rank 1)
- Agent-1: 0.7908 (rank 3)  
- Agent-2: 0.7920 (rank 2, standard alignment)
Differences are within CV noise.

## Next directions
- Try adding back one feature (e.g., n_tokens for 12 features)
- PCA/layer parameter tuning on 11-feature base
- Refine other existing features (pairwise_dist, centroid_dist)
