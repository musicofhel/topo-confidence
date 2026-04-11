---
creator: agent-1
created: 2026-04-08T09:10:00+00:00
---
# Plateau Analysis: 9 consecutive failures from 0.7862 base

## Failed approaches (evals #31-39):
1. Late-layer alignment: 0.7857 (neutral)
2. Equal-weight PCA: 0.7720 (-0.014)
3. layer_norm_slope swap: 0.7763 (-0.010)
4. Maxmin subsampling: 0.7786 (-0.008) + too slow
5. Layer-trajectory PH: 0.7764 (-0.010)
6. Global scaling k=0.5: 0.7850 (-0.001)
7. AUROC-weighted scaling: 0.7843 (-0.002)
8. trajectory_curvature swap: 0.7581 (-0.029)

## Diagnosis
ALL 14 features are deeply load-bearing. No single swap improves.
Scaling/transforms don't help — raw scales are already near-optimal.
The 14-feature logistic regression on 57 positives is at capacity.

## What hasn't been tried
- Subsample only answer-portion tokens for PH
- Layer SVD singular value concentration (S[0]/sum(S))
- Feature count REDUCTION (drop to 10-12)
- Completely different PCA basis (answer-weighted)
