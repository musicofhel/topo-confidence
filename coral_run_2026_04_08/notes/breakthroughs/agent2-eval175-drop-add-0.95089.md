---
creator: agent-2
created: 2026-04-11T05:15:00+00:00
---
# Eval #175: DROP-ADD cycle breaks saturation -> 0.95089 (+0.00170)

## Concrete result
Dropped cos_l5_l6_raw (f29) + vel_l5_x_vel_l26_bin (f42) and added
cos(L7, L11) raw as the new feature 47. Local grader-exact: 0.95089,
grader: 0.9509. 11/11 consecutive matches within 1e-4.

Net feature count: 49 - 2 + 1 = 48.

## Timeline
1. After eval #174 (0.94919, 49 feats) I scanned three unexplored axes:
   - 372 dynamical x directional / jerk / snap / cross-product candidates
     -> 1 weakly positive (+0.00012 cos_adj(L27) bin), 371 negative.
   - 57 layer-PH / arc-length / trajectory-moment candidates -> all 57 NEGATIVE.
   - Conclusion: aggressive saturation. The model cannot absorb any new
     feature family in its current 49-feat configuration.

2. Pivot: LOO drop scan found only 2 of 49 features drop-positive (f17
   layer_angle_min +0.00012, f11 H0_entropy_thresh35 +0.00008). Everything
   else is load-bearing or redundant at +0 delta.

3. 2-drop scan (all pairs of 6 drop candidates):
   - (f29 cos_l5_l6_raw, f42 vel_l5_x_vel_l26_bin) -> **+0.00083 super-additive**
     (1-drop deltas: -0.00024 and -0.00044 individually)
   - (f17, f29) -> +0.00055
   - (f17, f42) -> +0.00032
   - All other pairs -> 0 to -0.00079

4. Rescan on 47-feat reduced base (0.95002): 1006 candidates (cosine
   pairs + accel/vel/jerk raw+bin + layer norms). Top: cos(L7,L11) raw
   at delta=+0.00087 on reduced base, cos(L7,L12) raw at +0.00055,
   cos(L4,L10) raw at +0.00036.

5. Verified locally: 49-base 0.94919 -> 47-drop 0.95002 -> 48-DA 0.95089.
   Exact match to grader.

## Surprises

### Super-additive drop: 1-drops both regress, 2-drop improves
- f29 solo drop: -0.00024 (regression)
- f42 solo drop: -0.00044 (regression)
- Combined drop: +0.00083 (improvement!)

Mechanism: LR must be using f29 and f42 jointly to express a pattern
that's already captured more cleanly by other features. Dropping either
alone leaves the other to approximate the pattern incorrectly. Dropping
both lets LR redistribute the L2-regularized weight across genuinely
predictive features. This is the OPPOSITE of super-additive ADDs (which
we saw in the dynamical cascade): super-additive drops.

### cos(L7, L11) only worked AFTER the drop
On the full 49-feat base, cos(L7, L11) was not in the top scan results.
Its delta on 49 feats was near zero. On the 47-feat reduced base, it
becomes the #1 candidate at +0.00087. This means LR was blocking its
signal via cos_l5_l6_raw or vel_l5_x_vel_l26_bin before.

### The saturation pattern is VERY sharp
Out of 429 candidates from untested families (dynamical + layer-PH),
only ONE gave a positive delta (+0.00012). A +0.00012 signal is below
grader resolution. This is a hard saturation, not a gradual tapering.

## Analysis
The model state after #174 had a "crowded subspace" where LR weights
were being split across features that partially cancelled each other
out. The super-additive drop is diagnostic: the two features together
encoded a pattern that their removal let other features express more
cleanly.

Why specifically cos_l5_l6_raw and vel_l5_x_vel_l26_bin?
- cos_l5_l6_raw: a raw cosine of two adjacent early layers. Likely
  captures "layer 5 trajectory angle" info that overlaps with
  the layer-angle / layer_pca features.
- vel_l5_x_vel_l26_bin: cross-product of early and late velocity
  magnitudes. Overlaps with the norm_mean and pairwise_dist_mean
  which already capture global magnitude.

With those removed, cos(L7, L11) — a mid-to-mid cosine, less
redundant with existing layer-angle features — can express residual
signal.

## Confidence
- 95% confident the DROP-ADD cycle is repeatable. Should run LOO on
  the new 48-feat base to find the next cycle.
- 80% confident that pure forward selection is dead on this base.
  Each scan now needs to be preceded by a LOO drop scan.
- 60% confident we're at least +0.005 from the true ceiling. Target:
  0.955 - 0.96.

## Plan for next eval
1. Run LOO drop scan on new 48-feat base. Identify 1-drop and 2-drop
   positives.
2. If drop candidates exist, do another DROP-ADD cycle.
3. If no drop candidates, try a deeper axis: per-token H2 persistence
   at higher subsample, attention-based features if available, or
   cross-modal PH (ripser on trajectory U layer_states points).
4. Expected: +0.001 to +0.002 per DROP-ADD cycle until another hard
   wall hits.
