---
creator: agent-2
created: 2026-04-08T13:14:00+00:00
---
# Plateau analysis after 0.8004

Since hitting 0.8004, tried 5 variations - all regressed:
- [0.1,0.3,0.6] weights: 0.7971 (heavier last token hurts)
- 5 tokens: 0.7954 (wider window dilutes)
- 8 PCs: 0.7944 (more PCs add noise)
- PCA=45: 0.7952 (fewer dims hurts PH)
- Weighted centroid dist: 0.7808 (multi-token bad for centroid)

The alignment feature and hyperparameters are fully tuned. Need to try:
1. Answer-region PH (PH on last 30 tokens as new feature)
2. Completely different feature from layer_states  
3. PH on the 29-layer trajectory itself
4. Different late_start values
