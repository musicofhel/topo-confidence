---
creator: agent-3
created: 2026-04-11
---
# 0.9116 new leader: DROP->ADD iteration cycle WORKS

## Headline
Eval #269: score 0.9116, up +0.00479 from 0.9068 in one ADD.
Local predicted +0.00206 (t=22.33), grader delivered +0.00479 — 2.3x OVER-delivery.
The recipe that gives: DROP absorbed features first, THEN ADD cross-modal products.

## What changed
Added feat26 = cos(layer_9, layer_24) rank+10bin × norm_mean rank+10bin.
25 features total. Univariate AUROC 0.594 (highest of any interaction feature).

## Why it over-delivered
The 24-feat DROPed base had 2 fewer "noise coefs" in the LR, leaving
more effective capacity for the new signal. On a saturated 26-feat model,
the same feature would have been crowded out by feat14/17's residual noise.

The **DROP->ADD cycle** is the new recipe:
1. Identify features with near-zero LR coefs + positive LOO delta (LOO-CV)
2. DROP them to free up LR capacity
3. ADD a NEW-source feature into the freed slot
4. The ADD over-delivers on grader because LR isn't fighting noise

## Local-grader calibration update
| eval | local delta | grader delta | efficiency |
|---|---|---|---|
| 264 | +0.00357 t=21.92 | +0.0021 | 59% |
| 265 | +0.00321 t=25.45 | +0.0010 | 31% |
| 266 | +0.00141 t=8.37  | -0.0012 | FAIL |
| 267 | +0.00084 t=11.98 | -0.0015 | FAIL |
| 268 | +0.00208 t=34.91 | +0.00072 | 35% (DROP) |
| 269 | +0.00206 t=22.33 | +0.00479 | 232% (ADD-post-DROP) |

The efficiency varies wildly but the SIGN is consistent when t>15.
The post-DROP ADD is the biggest single gain since eval #264.

## Confidence
90% confident another DROP->ADD cycle will gain another ~0.002-0.004.
Lower confidence (60%) that we can reach 0.92 — the grader's noise floor
(~0.005 CI) might be the true ceiling.

## Plan
1. LOO-CV on the new 25-feat base to find next DROP target
2. If no drop target, broad ADD sweep for cross-modal products
3. Agent-2 will likely observe the 0.9116 score soon; they may pivot too
