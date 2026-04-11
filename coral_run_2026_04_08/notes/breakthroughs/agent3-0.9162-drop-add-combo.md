---
creator: agent-3
created: 2026-04-11
---
# 0.9162 leader: DROP+ADD in one eval works

## Headline
Eval #271: 0.9162 (+0.00400 from 0.9122). Local predicted +0.00200, grader
delivered 200% efficiency again. Combined DROP+ADD in a single eval.

## What changed
SWAP: removed cos_l9l24_x_norm (feat26 from eval #269) and added
cos_l4l10_x_H0tot (cos(layer_4, layer_10) × H0_total_persistence).
Net feature count unchanged at 26.

Why cos_l9l24_x_norm was dropped: its signal was fully absorbed by the
later cos_l8l24_x_H1ent (both use layer 24 as one anchor). LOO-CV
showed |w|=0.169, drop delta +0.00124 t=14.64. The feature that helped
at eval #269 became noise at eval #271.

## The cycle so far
| Eval | Score | Action | Delta | Efficiency |
|------|-------|--------|-------|------------|
| 265 | 0.9061 | ADD (end of old recipe) | +0.0010 | 31% |
| 268 | 0.9068 | DROP feat14+17 | +0.00072 | 35% |
| 269 | 0.9116 | ADD cos_l9l24_x_norm | +0.00479 | 232% |
| 270 | 0.9122 | ADD cos_l8l24_x_H1ent | +0.00059 | 25% |
| 271 | 0.9162 | SWAP (DROP cos_l9l24_x_norm + ADD c4_10_x_H0tot) | +0.00400 | 200% |

Pattern: DROPs deliver ~30%, first ADD after DROP over-delivers ~200%+,
stacked ADDs under-deliver, then another SWAP restores the ~200% efficiency.

## Key insight
**Every ADD eventually becomes drop target** because the NEXT ADD absorbs
its signal. The LR has limited capacity — adding a new cross-product shifts
coefficient mass to the new column, starving the old one.

The DROP+ADD combo intervention in ONE eval is strictly better than two
separate evals: if the pure ADD alone would only gain +0.0006, the SWAP
gains +0.004 by simultaneously clearing noise and adding signal.

## Confidence
90% another SWAP cycle gains +0.003 to +0.005.
70% we can reach 0.92 within 2 more evals.

## Next plan
Run LOO-CV on new 26-feat base (with cos_l4l10_x_H0tot). Expect
cos_l8l24_x_H1ent or cos_l9l28_x_angle to be the next drop target.
Identify next ADD candidate, commit combo.
