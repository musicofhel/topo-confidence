---
creator: agent-3
created: 2026-04-10
---
# Plateau after 15+ regressions on 0.8790 leader — shifting to transform tuning

## The plateau
Every single-feature change to the 19-feat leader has regressed:

| Type | Examples | Best regression |
|------|----------|-----------------|
| ADD 20th | token_angle_min, layer_accel_angle_min | -0.0015 |
| SWAP (same family) | velocity->accel curvature | -0.0152 |
| SWAP (new family) | -> layer_arc_length | -0.0077 |
| REMOVE weakest | -tok_layer_min_dist (uni 0.508) | -0.0111 |
| TARGETED transform | bin=5 on tok_layer_min_dist | -0.0140 |

## Diagnosis
The 19-feat mix is Pareto-optimal: every feature carries unique multivariate
signal even when uni AUROC = 0.508. EPV is 3.0, exactly at the empirical
floor. No feature-level modification can win because:
1. ADDs hit EPV (57/20 = 2.85)
2. SWAPs lose specific multivariate contribution of dropped feature
3. REMOVEs cost ~0.01 AUROC even for uni-0.508 features

## New direction: transform-pipeline hyperparameter tuning
I've been treating rank+10bin as fixed. It's actually a hyperparameter never
tuned at the 19-feat level — agent-2 found it at the 12-feat era. Candidates
to try (each a single eval):
- bin count: try 15, 20, 7, 5 (currently 10)
- transform: gaussian-rank (norm.ppf) instead of discrete bins
- selective: keep bin=10 on PH features, try different bin on curvature

## Rationale for direction shift
Transform tweaks are STRUCTURALLY different from feature adds/swaps — they
don't hit EPV (no new coefs) and they don't break load-bearing contributions
(features keep their orthogonal information, just different encoding). Lower
regression risk, potential for small but real gains.
