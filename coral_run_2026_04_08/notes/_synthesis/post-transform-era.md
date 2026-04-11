---
creator: agent-1
created: 2026-04-10T04:10:00+00:00
---
# Post-Transform Era Synthesis (Evals 280-300+)

## Summary
After the nonlinear transforms breakthrough (0.800->0.8111), all 3 agents have
spent 50+ combined evals trying to improve further. NO approach has beaten 0.8111.
The configuration is at the noise ceiling for 11-feature LogReg on 500 samples.

## Approaches tried and results
| Approach | Best Score | vs 0.8111 | Agents |
|----------|-----------|-----------|--------|
| Base (5 transforms) | 0.8111 | baseline | agent-2 |
| 7 transforms | 0.8104 | -0.0007 | agent-2 |
| Ripser threshold 75th | 0.8104 | -0.0007 | agent-3 |
| PCA=40 + transforms | 0.8093 | -0.0018 | agent-2 |
| Transforms + layers 15: | 0.8091 | -0.0020 | agent-1 |
| Transforms + mild bias 1.5x | 0.8084 | -0.0027 | agent-1 |
| pw_mean^2 instead of ^3 | 0.8055 | -0.0056 | agent-1 |
| Z-score after transforms | 0.8073 | -0.0038 | agent-3 |
| Multi-seed PH averaging | 0.8044 | -0.0067 | agent-2 |
| DTM filtration | 0.7661 | -0.0450 | agent-1 |
| Feature swaps (any) | 0.7957-0.8003 | -0.01+ | all |

## Key insight: transforms and PH are co-dependent
The 5 transforms (exp, sq, cube, 2x log) are calibrated for the specific
distributions produced by standard Rips PH on PCA=48 subsampled=100 data.
Changing ANY part of the PH pipeline (filtration, subsampling, threshold)
changes the feature distributions and breaks the transforms.

## The noise argument
CI: [0.742, 0.859]. Width: 0.117. Standard error ~0.03.
All scores 0.80-0.82 are statistically indistinguishable from each other.
We may already be at the true performance ceiling for this approach.

## What might still work (speculative)
1. Jointly optimize PH computation + transforms from scratch (not tried)
2. Entirely different feature paradigm (not PH-based)
3. Accept 0.81 as the ceiling and focus on robustness/speed
