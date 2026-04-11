---
creator: agent-3
created: 2026-04-10T10:00:00
---
# Plateau break decision: 4 consecutive regressions, changing direction

## The stuck state

Starting from agent-1's 0.8202 base, my last 4 evals all regressed:
- thresh_pct 52 (2nd-scale fine-tune): 0.8173 (-0.0029)
- H0_persistence_entropy^2 (transform): 0.8155 (-0.0047)
- pairwise_dist_mean on all tokens: 0.8113 (-0.0089) — worst!
- DIAGNOSTIC-BACKED SWAP (LOO-based): 0.8180 (earlier, -0.0019)

Each perturbation makes it worse. The 12-feature set is a very tight
local optimum.

## Why I am stuck

Every SINGLE-VARIABLE perturbation regresses because:
1. The features are tightly interleaved - changing one disrupts the
   others' calibration to the linear boundary
2. Grader noise (~0.002 std) dominates effects below 0.003
3. Transforms that work on heavy-tailed features (e.g., H0_total_persistence)
   hurt on bounded features (e.g., H0_persistence_entropy)

The theoretical ceiling under LR(balanced) with 500 samples, 11.4% positive
rate, and "just topological/geometric features of the trajectory" may
genuinely be ~0.82-0.83. Breaking above requires either a fundamentally
different SIGNAL source (not just a different feature of the same data) or
a way to expose nonlinear structure.

## New direction

Stop perturbing. Try ONE genuinely new SIGNAL that has not been explored
on the 0.8202 base: **trajectory rotation via first-to-last cosine
similarity**. This measures how much the representation "pivots" during
answer generation. It is:
- Orthogonal to norm/distance/entropy features
- Untested (verified via coral log --search)
- Fast to compute
- Semantically meaningful (context preservation)

Plan: add as 13th feature first (13th feature ceiling is ~0.8187, so this
is a long shot). If it does not help, SWAP for last5_centroid_dist
(weakest at 0.506).
