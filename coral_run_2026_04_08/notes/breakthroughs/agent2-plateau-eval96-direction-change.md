---
creator: agent-2
created: 2026-04-10T08:25:00-04:00
---
# Plateau at 0.8192 — Direction change to layer trajectory topology

## Stuck approach: multi-scale PH on token trajectories
- Best: 0.8192 (13 features, three-scale PH 75/50/35, commit f8951975)
- 5 consecutive regressions trying to extend this:
  - 14th feat H0_thresh20: 0.8173
  - 14th feat H0_total_thresh50: 0.8180
  - Drop last5_centroid_dist: 0.7905 (catastrophic)
  - 14th feat H1_entropy_thresh50: 0.8172
  - exp transform H0_entropy_thresh35: 0.8149

**Why stuck**: 13 features on 57-positive class is near capacity. Multi-scale H0 features all share the same underlying cluster signal — redundant. The topological signal from token trajectories is essentially saturated.

## New direction: Layer trajectory topology
CLAUDE.md explicitly flags: "layer_states gives you 29 points per problem in 1536D — topology of how representations evolve across layers. This data is UNEXPLOITED."

Currently I only use layer_states for the SVD alignment subspace. I don't compute ANY topology on the 29-point layer trajectory itself. This is genuinely orthogonal to token-trajectory PH.

## Hypothesis
If a model "converges" on an answer early and stays consistent across late layers, its 29-point layer trajectory in representation space should be short and compact (low-entropy H0 diagram). If it's uncertain or contradictory, the trajectory should be longer and more spread out.

Prior layer-PH attempts failed because they added 14-25 features at once. I'll add ONE minimal feature.

## Plan
1. Revert to 0.8192
2. Add 14th feature: `layer_traj_h0_entropy` = H0 persistence entropy of the 29-layer trajectory in the same PCA subspace
3. Eval.

Expected: unknown — this is genuine exploration. Even if flat/slightly negative, I learn whether layer-trajectory topology carries signal.
