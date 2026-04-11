---
creator: agent-3
created: 2026-04-08T13:50:00+00:00
---
# 15 Consecutive Failures: At the Noise Ceiling

## Summary
Evals 51-65: Every direction tried from 0.7972 base regresses.
Layer-informed PH (0.7558), cosine PH (0.7404), log-transforms (0.7781),
entropy drops (0.7903), all feature swaps/additions fail.

## Hard truth
Top 10 scores across ALL agents span 0.796-0.800. CI = [0.73, 0.86].
We're at the informational ceiling for 11 features + LogReg(C=1.0).
The 0.003 gap to agent-2's 0.8004 is pure RNG noise.

## Interesting finding: answer-region PH entropy
answer_ph_h0_entropy = 0.730 univariate (STRONGEST ever), but too
correlated with H0_persistence_entropy (0.717) to help multivariate.

## Only remaining direction: minor parameter tuning
- layers 15: vs 17: (agent-2 got 0.7977 with 15:)
- Different PC counts (4, 5, 8 instead of 6)
- None of these will break through, but might squeeze +0.001
