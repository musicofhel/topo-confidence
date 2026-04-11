---
creator: agent-3
created: 2026-04-10
supersedes: product-interactions-break-plateau.md (expands it)
---
# Interaction-recipe scaling: 0.8192 -> 0.9051 in ~7 features

## Summary
The product-interaction insight that broke 0.8192 generalizes far further than
initially expected. Adding 7 derived features (products, squares, centered
squares, and cross-modal cosines) on top of the 18-feat raw-feature base took
the score from 0.8682 -> 0.9051 (+0.037). Every addition followed the same
recipe: find two rank+10bin features LR cannot jointly model, multiply them.

The EPV ceiling (57 positives / 2.85 = ~20 feature cap) applies ONLY to
independent raw features. Derived columns (products, self-transforms) can
push well past 25 features without overfitting.

## Evidence chain (agent-3 + agent-2, in chronological order)
| Score | Feature added | Type | Local delta | Grader delta |
|-------|---------------|------|-------------|--------------|
| 0.8682 | (18-feat base) | -- | -- | -- |
| 0.8779 | layer_angle_min (raw) | Raw layer curvature | -- | +0.0097 |
| 0.8790 | layer_angle_min_plus_max | Raw curvature | -- | +0.0011 |
| 0.8826 | angle_x_H1total | INTERACTION (layer x H1) | +0.0019 | +0.0036 |
| 0.8856 | crossmodal_sq | SELF-TRANSFORM (feat16 ** 2) | +0.00766 | +0.0030 |
| 0.8949 | last5_x_pc2early | INTERACTION (cross-modal) | +0.01228 | +0.0093 |
| 0.9024 | norm_x_layernorm | INTERACTION (cross-modal) | +0.00234 | +0.0075 |
| 0.9030 | angle_plus_csq | SELF-TRANSFORM ((x-4.5)^2) | +0.00253 | +0.0006 |
| 0.9051 | cos_l9l28_x_angle | NEW SOURCE + INTERACTION | +0.00357 | +0.0021 |

## Recipe rules (validated)
1. **Features must be rank+10bin ints (0-9) before multiplying.** Otherwise
   one column's magnitude dominates the product. All the winning products
   involve two rank+10bin columns, yielding int values in [0, 81].
2. **Cross-modal products dominate.** The three biggest gains (last5*pc2early,
   norm*layernorm, cos*angle) all multiplied a TOKEN-TRAJECTORY feature with
   a LAYER-STATE feature. Within-modal products are weaker.
3. **Don't over-mine one source.** feat18 (layer_angle_min_plus_max) now has
   3 derived features — adding a 4th (angle_plus_csq) under-delivered on the
   grader even though local CV said +0.00253. Rule of thumb: 2-3 derivatives
   per source before diminishing returns.
4. **Self-transforms work but only for high-signal features.** feat16 (tok_layer
   cross-modal distance, uni AUROC 0.506) benefited from squaring +0.0077.
   Low-uni features don't benefit from non-linear spacing alone.
5. **Grader over-delivery correlates with layer-geometry signals.** Three out
   of four layer_angle / layer_cos features over-delivered on the grader
   relative to local CV — consistent pattern across both agents.

## The sweep methodology
1. Extract features.py output ONCE (6-12s) and cache in memory.
2. Generate ~150 candidate 26th columns via in-memory matrix ops:
   - All pairwise products of rank+10bin cols (not yet used)
   - Self-squares and centered squares
   - Triples (a*b*c) from cross-modal combos
   - NEW sources (layer cosines at different layer pairs)
3. Run 20-seed paired CV (50 folds each) ~= 5-7 minutes per sweep.
4. Commit the top candidate with t > 10 and pos > 0.9.
5. Re-sweep after each commit (the previous-best stackers shift).

## Confidence
**Very high** that 2-4 more features of this type can be added before the
recipe exhausts. The 25-feat local CV is 0.9007 → grader 0.9051. Each
successive feature yields diminishing returns (0.01 → 0.008 → 0.007 → 0.0006
→ 0.002). At some point the interaction space will saturate.

**Moderate** that 0.92+ is reachable with this family alone. After 0.91, we
may need a completely new source (e.g., per-token PCA, layer cosine chains,
or full layer PH) rather than more interactions.

## Open questions
- Can we stack multiple features per eval instead of one-at-a-time?
  Local CV suggests yes (23+24 both improved), but the under-delivery on
  feat18_csq warns that local stacking over-predicts grader stacking.
- Do RIDGE-regularized products help more than raw products? (Untested.)
- What's the ceiling if we exclusively mine layer_cosine features? Agent-2
  is exploring this direction; their cos_l0_l5 and cos_l9_l28 are promising.
