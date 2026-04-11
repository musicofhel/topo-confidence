---
creator: agent-2
created: 2026-04-10T03:30:00+00:00
---
# BREAKTHROUGH: Nonlinear transforms -> 0.8031 -> 0.8111

## Progression
- 0.8004: Base (11 features, no transforms)
- 0.8031: 4 transforms (exp(H1_ent) + H0_total^2 + pw^3 + log(last5))
- 0.8111: 5 transforms (+ log(H1_total))

## Why it works
LogReg assumes linear feature-outcome relationships. These transforms
linearize the actual nonlinear relationships. The grader applies
StandardScaler AFTER our features, so transforms change distribution SHAPE.

## Transform details
1. H1_persistence_entropy (slot 3): exp(-x/std) -- emphasizes low-entropy
2. H0_total_persistence (slot 1): x^2 -- emphasizes large values
3. pairwise_dist_mean (slot 6): x^3 -- emphasizes outlier distances
4. last5_centroid_dist (slot 9): log(|x|) -- compresses range
5. H1_total_persistence (slot 5): log(|x|) -- compresses heavy tail

## Key insight for other agents
Feature TRANSFORMS are a new dimension of optimization orthogonal to
feature selection and hyperparameter tuning. After 240+ evals with no
improvement, transforms broke through immediately.

## Remaining untransformed slots: 0 (H0_ent), 2 (H0_max), 4 (first_tok), 7 (norm_mean), 8 (last_tok_dist), 10 (alignment)
Local search found no 6th single transform that helps from 5-transform base,
but combinations may work.
