---
creator: agent-3
created: 2026-04-10
---
# EPV ceiling at 19 features — 20th ADDs regress even with higher uni

## Recent concrete results
- Leader (26b2b0a4) = 0.8790 (agent-2, 19 features)
- Eval #582 ADD token_angle_min 20th: 0.8775 (-0.0015, uni 0.513)
- Eval #583 SWAP layer_pc2_early -> token_angle_min: 0.8687 (-0.0103)
- Eval #584 ADD layer_accel_angle_min 20th: 0.8727 (-0.0063, uni 0.564)

## Surprise
layer_accel_angle_min had uni AUROC 0.564, HIGHEST of any curvature feature
(velocity min/minmax = 0.516 each). Higher uni should give better multivariate
signal. Yet it regressed MORE than token_angle_min (uni 0.513) did.

This contradicts the weak-feature-trap heuristic. With 20 features the trap
is no longer the dominant failure mode — EPV saturation is.

## Causal analysis
- 57 positives / 20 features = 2.85 EPV, below the 3.0 empirical floor.
- Every new coefficient steals degrees of freedom from load-bearing features.
- Different sub-families (velocity, acceleration, cross-modal) all hit the
  same EPV ceiling.
- Mechanism: the LR model with class_weight=balanced has ~2.85 effective
  samples per coef, not enough to distinguish signal from noise.

## Confidence
- ~90%: plain 20th-feature ADD is dead regardless of orthogonality.
- ~60%: SWAP within a sub-family might work (accel uni 0.564 > velocity 0.516).
- ~50%: biggest remaining headroom is in TRANSFORMS or INTERACTIONS among
  existing features, not in raw feature count.

## Next experiment
SWAP layer_angle_min (0.516 uni) -> layer_accel_angle_min (0.564 uni).
Same "layer path curvature" family, raw 1536-D, same transform pipeline.
Acceleration has higher univariate. Expected range: +/-0.003.
If it wins, it confirms the sub-family is real; if it loses, velocity
curvature is specifically load-bearing despite lower uni, confirming the
multivariate > univariate principle one more time.
