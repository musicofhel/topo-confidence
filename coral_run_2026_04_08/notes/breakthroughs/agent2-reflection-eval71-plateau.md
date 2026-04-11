---
creator: agent-2
created: 2026-04-08T18:15:00+00:00
---
# Reflection: 25+ Consecutive Regressions from 0.8004

## Score trajectory
All 25+ attempts since 0.8004 have regressed. Closest was squared alignment at 0.7985.

## What was tried and failed
- Token weight variants, PC counts, PCA dims, late_start tuning
- Feature additions (12th feature wall), replacements (all load-bearing)
- Alignment variants (squared, skip PC1, early-late contrast)
- PH modifications (cosine distance, per-problem PCA, standardization)
- Transforms (Yeo-Johnson, variance-weighted PCA)
- Layer curvature as replacement: 0.527 univariate AUROC, worse than first_token_norm 0.575

## Why 0.8004 is hard to beat
- EPV = 57/11 = 5.2 (low for LogReg)
- CI = [0.735, 0.858] - improvements invisible in noise
- Every feature is load-bearing with unique signal

## Next direction: Layer-state PH topology
The 29-layer trajectory has never been processed with PH. This is genuinely new info.
Replace 2 weak token-PH features with layer-PH features to mix two independent signals.
