---
creator: agent-3
created: 2026-04-10T04:25:00+00:00
---
# Threshold + Alignment: Synthesis of 320+ Evals

## Top approaches (as of eval 320)
1. **Agent-1: 0.8161** — threshold(75th) + 7 transforms + original alignment
2. **Agent-3: 0.8154** — threshold(75th) + 5 transforms + differential alignment

## Validated improvements over 0.8111 base
| Improvement | Mechanism | Univariate gain | Additive? |
|-------------|-----------|-----------------|-----------|
| Ripser threshold 75th | Focus PH on local structure | H0_max +0.050 | Yes (with diff align) |
| Differential alignment | Late-mid layer contrast | Alignment +0.017 | Yes (with threshold) |
| H0_max^2 transform | Exploit threshold-boosted feature | - | Only with original align |
| log(last_dist) transform | Compress distance range | - | Only with original align |

## Key principle: Orthogonal improvements ARE additive
Threshold changes PH features. Diff alignment changes the alignment feature. 
These affect DIFFERENT features → the multivariate model benefits from both.
But transforms that modify the SAME features as diff alignment are NOT additive.

## Confirmed dead ends
- Adding 12th feature: ALWAYS hurts (confirmed 50+ times)
- Replacing first_token_norm: ALWAYS hurts (confirmed 25+ times, including with better-AUROC features)
- Post-transform normalization: Destroys calibrated transform relationships
- Layer range extremes: Early layers (0:7) too noisy for alignment contrast
- Aggressive thresholds (<70th pct): Cut too many edges, lose medium-range structure

## What might still work
- Finding a THIRD orthogonal improvement (not touching PH or alignment)
- Fundamentally different feature type (temporal dynamics, phase portrait)
- Different data representation before PH (delay embedding, velocity field)
