---
creator: agent-2
created: 2026-04-10T21:10:00
---
# Layer curvature feature family — new breakthrough (0.8682 -> 0.8703)

## Result
Eval #577: 0.8703 (+0.0021 grader) from adding layer_angle_min as 18th
feature. Local 50-seed pair CV predicted +0.0038 (se=0.0013, t=+2.89,
pos=66%). Grader delivered 55% of predicted effect. New overall leader.

## The feature
d1 = np.diff(layer_states[i], axis=0)        # (28, 1536)
d1_unit = d1 / ||d1||
cos_angles = sum(d1_unit[:-1] * d1_unit[1:], axis=1)  # (27,)
layer_angle_min = min(cos_angles)
Rank+10bin transform.

Geometric meaning: at each layer the hidden state moves in some direction.
layer_angle_min is the MIN cosine between consecutive layer movement
directions - the single sharpest turn in the layer trajectory.

## Why it works - NEW feature family
"Layer curvature" - distinct from PH, norm, distance, PCA variance, and
cross-modal features. It measures the SHAPE of the layer path.

Univariate AUROC 0.516 (near chance) but multivariate delta reliable
(t=2.89, pos=66% over 50 seeds).

## Why raw 1536-D beats PCA reduction
layer_angle_min_pca (on PCA-45-reduced): -0.00347 (vs +0.00380 raw).
Complete reversal.

Hypothesis: sharp-turn signal lives in high-frequency directions
orthogonal to token PC directions. PCA fit on tokens projects them out.
Opposite to tok_layer_min_dist which REQUIRES PCA. Different features
need different spaces.

## Failed variants (30-50 seed CV)
- layer_angle_mean, p10, std, argmin - all regressed
- layer_angle_min_early (first 13 angles) - -0.00485
- layer_angle_min_pca - -0.00347
- layer_d1/d2 norm max/mean/std - all regressed
- 15+ cross-modal variants (Hausdorff, mean-min, cosine, early/late
  splits, argmin) - all regressed or unreliable

Only MIN of all 27 cosines in raw space works.

## Session progress
Start: 0.8234. Now: 0.8703. Target 0.75 exceeded by +0.1203.

## Next experiment
Try angle between velocity d1 and acceleration d2 - actual curvature
direction, different from angle-between-velocities. Expect +0.001-0.003.
