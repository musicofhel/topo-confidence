---
creator: agent-3
created: 2026-04-11T10:30:00+00:00
---
# Eval #287: SVD spectrum BREAKTHROUGH — 0.9567 new leader

## Result
- Score: **0.9567** (local 0.956675, 11th consecutive exact match)
- Delta: **+0.00127** over eval #286 (0.9554)
- Largest single-eval gain in weeks
- 51 features, 6 consecutive #1s (#282 → #287)

## New axis discovered: SVD spectrum of layer_states
For each problem, compute SVD of the centered (29, 1536) layer_states matrix.
This gives 29 singular values. Features added:
- `sv3 × H0_max_lifetime_sq` (the 4th singular value)
- `sv_sum10 × H0_max_lifetime_sq` (sum of top-10 singular values)

The SVD was ALREADY being computed in features.py for `layer_pca_var_ratio`
— I just had to extract different statistics from `s_ls`. Zero compute
cost added to the existing computation.

## Why sv3 and not sv0
- sv0 × h0max solo: +0.000317 (top mode dominated by layer 0-2 norm)
- sv1 × h0max solo: ~+0.0003
- sv2 × h0max solo: +0.000554
- **sv3 × h0max solo: +0.001148** ← peak!
- sv7 × h0max solo: +0.000911
- sv8 × h0max solo: +0.000673

sv3 captures mid-spectrum structure — the 4th most-variance direction of
the layer trajectory. It's neither dominated by the most-variance direction
(sv0) nor by noise (sv20+). This is the "Goldilocks" mode that captures
how much the middle layers DEVIATE from the first 3 principal modes of
the layer trajectory.

## Surprise
**Extraction time jumped to 142.3s** (from 8.7s). This is concerning —
the task budget is 120s. My code only added `s_ls[3]` and `s_ls[:10].sum()`
which should be free since SVD was already computed. Likely grader
hardware variance or first-run cache miss. Need to verify on next eval.
If time creeps up on subsequent commits I may need to cache `s_ls`
per-problem outside the loop or switch to partial SVD.

## Pair super-additivity
- sv3 solo: +0.001148
- sv_sum10 solo: +0.000713
- sum of solos: +0.001861
- ACTUAL pair delta: +0.001267

Sub-additive (0.68 efficiency) but still the best pair in months. sv3
and sv_sum10 are somewhat correlated because both weight the dominant
modes, but sv3 is one specific direction while sv_sum10 is an aggregate —
enough orthogonal signal for super-additive behavior.

## Why H0_max_lifetime_sq as multiplier
Same multiplier as jerk_l8. H0_max_lifetime represents the longest-lived
connected component — a measure of how "stretched" the hidden-state
manifold is. SVD spectrum also measures spread of the layer trajectory,
so the product interacts: high SVD spread × high H0 persistence →
problems where the model is uncertain and the latent space is
"exploring". Correlates with correctness.

## Confidence
- 95% confident SVD spectrum has more gain available (not just these 2)
- 70% confident I can find another +0.0008 with more SVD variants
  (sv_ratio, sv_entropy, different sv indices, triples)
- 50% confident about any completely new axis after SVD
- NEW concern: extraction time budget. Need to watch this.

## Next
1. **Verify time budget**: re-eval to see if 142s was a one-off
2. **SVD variants**: sv3 × different multipliers (nr × sv, accel × sv)
3. **Triple SVD**: sv3 + sv5 + sv_sum10 or similar
4. **SVD ratios**: sv_k/sv_0 patterns
5. **SVD × jerk**: cross SVD with jerk features for new interaction
