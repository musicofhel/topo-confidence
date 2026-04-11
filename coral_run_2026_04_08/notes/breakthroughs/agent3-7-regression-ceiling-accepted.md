---
creator: agent-3
created: 2026-04-10
---
# 7-regression chain — ceiling accepted, one final creative attempt

After 7 consecutive regressions on the 0.8202 base, combined with agent-2's
explicit "ceiling at 0.8202 ± 0.005" synthesis, I'm accepting the plateau.
Agent-1 has 163 attempts, most clustered in 0.80-0.82 noise band. Agent-2
stopped at 0.8195. I'm at 0.8199.

## Complete regression chain on 0.8202

| # | Change                                         | Score   | Delta    |
|---|------------------------------------------------|---------|----------|
| 1 | thresh_pct 50 → 52                             | 0.8173  | -0.0029  |
| 2 | H0_persistence_entropy^2 transform             | 0.8155  | -0.0047  |
| 3 | pairwise_dist_mean on ALL tokens               | 0.8113  | -0.0089  |
| 4 | answer_depth_layer feature                     | DEGEN   | -        |
| 5 | last_token_global_pc1 swap                     | 0.7894  | -0.0308  |
| 6 | early-weighted centroid (exp decay tau=n/3)    | 0.8166  | -0.0036  |
| 7 | (pending: 13th feature intrinsic dimension)    | ?       | ?        |

## Why early-weighted centroid failed

**Surprising finding**: the swap made last_token_centroid_dist uni WORSE
(0.639 → 0.561) while making last5_centroid_dist uni BETTER (0.507 → 0.597).
Net multivariate effect: regression.

**Mechanism**: the two features (last_dist, last5_dist) share a centroid.
When I changed the centroid computation, both features shifted correlatedly.
The original uniform-mean centroid produces uncorrelated information between
the two features; the early-weighted centroid makes them share too much.
**Redundancy between features hurts more than individual univariate strength.**

## Confidence assessment

- **95% confident**: 0.8202 is the signal ceiling under LR(balanced) + 50-fold
  CV + 500 samples + 11.4% positives. Evidence: (a) agent-2's 20+ regressions;
  (b) agent-1's 163 attempts plateau; (c) my 155+ attempts plateau; (d) CI
  width 0.11 vs top-3 gap 0.001; (e) no new feature with uni < 0.6 ever helps.

- **5% hope**: a truly orthogonal signal type (manifold dimension, topological
  invariant not yet tried) might add +0.001-0.003 via robust multivariate
  contribution. Worth ONE more attempt: twoNN intrinsic dimension of reduced
  trajectory. If this regresses, STOP for good.

## What would change my mind

Finding a feature with uni ≥ 0.6 that's also NOT correlated with any of the 12
existing features (max abs correlation < 0.5). This is the condition for
genuine additive signal. Everything tried so far fails at least one of these.
