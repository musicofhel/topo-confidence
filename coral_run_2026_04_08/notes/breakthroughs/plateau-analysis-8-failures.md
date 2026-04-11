---
creator: agent-3
created: 2026-04-08T13:20:00+00:00
---
# Plateau: 8 consecutive failures from 0.7963/0.7972 base

## Failed attempts (evals 51-58)
All feature additions, PH changes, and feature swaps fail.
11 features is optimal. All features load-bearing.

## Radical direction: separate PCA dims for PH vs geometric features
PH in 48D suffers curse of dimensionality. Use low-D PCA (10-15) 
specifically for PH, keep 48D for geometric features.
