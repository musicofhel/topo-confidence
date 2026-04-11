---
creator: agent-3
created: 2026-04-08T05:08:00+00:00
---
# Plateau at 0.7686 — Direction Change Needed

## Diagnosis
7 consecutive modifications to the 13-feature PCA(40) set all hurt:
- Adding features → dilution (14-15 features too many)
- Removing features → each one uniquely contributes 
- Swapping features → interactions are fragile
- Changing subsample → more variance or different geometry
- Changing distance metric → Euclidean is correct for PCA space

## Why I'm stuck
ALL features are computed from the SAME PCA(40)-reduced token point cloud.
Every new feature I try is correlated with existing ones because they share
the same underlying representation.

## New direction: features from DIFFERENT representations
1. **Raw 1536D space features** — PCA discards variance; raw norms/angles carry 
   information PCA throws away (directions of low variance may still be discriminative)
2. **Betti curve at characteristic scale** — instead of scalar PH summaries (entropy, 
   total), count features alive at the MEDIAN distance. Different information.
3. **Layer-state conditioned features** — use layer states to define DIRECTIONS 
   in 1536D space, project tokens onto these directions. Combines both data sources.

## Plan
Try raw-space features first (simplest, fastest). If that doesn't work,
try Betti curves, then layer-conditioned features.
