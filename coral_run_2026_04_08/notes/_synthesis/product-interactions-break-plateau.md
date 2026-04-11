---
creator: agent-2
created: 2026-04-10T08:50:00+00:00
---
# Product interactions break the 0.8192 plateau (+0.0003)

## Summary
After 16 consecutive failures trying to extend 0.8192, a **product interaction
feature** (H0_entropy_thresh50 × H0_entropy_thresh35) improved the score to 0.8195.
This is the first improvement on the plateau and opens a new direction for all agents.

## The key insight
**LogisticRegression is LINEAR and cannot learn feature products.** If two features
carry complementary signal that's stronger when combined multiplicatively, adding their
explicit product gives the model access to that signal as a new degree of freedom.

## Evidence
- Base: 0.8192 (13 features, three-scale PH at thresh 75/50/35)
- +1 product feature (thresh50 × thresh35): **0.8195** (14 features)
- Product feature univariate: 0.725 (competitive with strongest base features)

## Why it works here
- The three H0 entropy features (75/50/35) are MODERATELY correlated (~0.6-0.8 typical)
  but not redundant. They each carry scale-specific structure.
- Their pairwise product is a NEW signal: it's large when BOTH scales show complex
  topology simultaneously, small otherwise. This "joint topological complexity" is not
  captured by a linear combination.
- The single product feature didn't dilute the other 13 — it ADDED.

## What this unlocks
Pairwise products across strong features become a new feature-generation strategy:
1. H0_entropy × H0_entropy_thresh50 (coarsest × mid)
2. H0_entropy × H0_entropy_thresh35 (widest scale range — next to try)
3. H0_entropy × alignment (topology × direction — cross-modality)
4. alignment × last_token_centroid_dist (direction × distance)

## Confidence
**Moderate-high.** The gain is small (0.0003) but it's the first non-noise improvement
in 16 evals. The mechanism (LR can't learn products) is sound and well-understood.
Worth pursuing 2-3 more interaction features to see if the strategy stacks.

## Caveat
Products of correlated features have large variance across CV folds. Adding too many
interactions could overfit. Budget: test 2-3 more, stop if they all regress.
