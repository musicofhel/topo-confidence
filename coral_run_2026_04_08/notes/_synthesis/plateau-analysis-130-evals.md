---
creator: agent-1
updated: 2026-04-08T09:32:00+00:00
---
# Plateau Analysis: Synthesis of 140+ Evals Across 3 Agents

## Summary
After 100+ evals stuck at 0.787, agent-2 discovered dropping 3 features 
(norm_std, bridge_silhouette, n_tokens) improved to 0.792. Combined with
agent-1's SV-weighted alignment and all-layer SVD → 0.7945 (new #1).

## The 7 breakthroughs (in order)
| # | What | Gain | Who |
|---|------|------|-----|
| 1 | Geometric features | +0.014 | agent-3 |
| 2 | Position features | +0.031 | agent-3 |
| 3 | PCA=45 | +0.011 | agent-3 |
| 4 | Layer alignment | +0.008 | agent-3/2 |
| 5 | Correlation swaps | +0.008 | agent-1 |
| 6 | Drop 3 features (14→11) | +0.005 | agent-2 |
| 7 | SV-weighted + all layers | +0.002 | agent-1 |

Total: 0.713 → 0.795 = +0.082 AUROC.

## Key insight: feature count has a U-shaped optimum
- 7 features (baseline): 0.713
- 11 features (current best): 0.795
- 14 features (double swap): 0.787
- Adding features beyond 11 adds noise; removing below 11 loses signal.
- The "all features load-bearing" test was misleading — it tested single 
  drops from 14, not joint drops of the RIGHT 3 features.

## Current best config
11 features, PCA=45, all 29 layers SVD, SV-weighted K=3, subsample=100.
