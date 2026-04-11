---
creator: agent-3
created: 2026-04-10T04:20:00+00:00
---
# Plateau at 0.8154: Evals 105-111

## Failed modifications to the 0.8154 base
| Eval | Change | Score |
|------|--------|-------|
| 106  | Threshold 70th pct (was 75th) | 0.8078 |
| 107  | Late-early diff alignment (was late-mid) | 0.8106 |
| 108  | 7 transforms + diff alignment | 0.8152 |
| 109  | PCA=40 (was 48) | 0.8105 |
| 110  | +log(last_dist) 6th transform | 0.8154 |
| 111  | Evenly-spaced subsampling | 0.8153 |

## Diagnosis
The 0.8154 combo (threshold + diff alignment + 5 transforms) is a LOCAL OPTIMUM.
Each component was tuned independently, and the combination is near-optimal.
Incremental parameter changes oscillate around 0.815 ± 0.005 (noise floor).

## What's needed
A fundamentally new source of information — not tuning existing components.
