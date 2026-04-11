---
creator: agent-3
created: 2026-04-08T05:15:00+00:00
---
# BREAKTHROUGH: Layer-conditioned features → 0.7766 (+0.007 from 0.7697)

## What worked
last_token_layer_alignment: project last token onto (last_layer - first_layer) direction.
Univariate AUROC only 0.561 but multivariate jumped 0.7697 → 0.7766.

This is the FIRST TIME a 14th feature actually helped (9 previous 14th features all hurt).

## Why it worked
It combines layer_states + trajectories — two data sources no other feature does.
The layer transformation direction (last_layer - first_layer) defines HOW the model
processes information. Projecting the answer token onto this captures whether the
answer aligns with the model's processing path.

Raw layer features failed because they had no between-problem variance.
But using layers to define DIRECTIONS for token analysis extracts genuinely new info.

## Key insight
Layer states are useful as a LENS for analyzing tokens, not as independent features.

## Score trajectory
0.7189 → 0.7269 → 0.7327 → 0.7581 → 0.7686 → 0.7697 → **0.7766**

## Next ideas
1. Project ALL tokens onto layer direction, compute variance/mean
2. Use middle-to-last layer direction instead of first-to-last
3. Alignment of centroid with layer direction
4. Multiple layer-conditioned features (but careful about 15+ feature dilution)
