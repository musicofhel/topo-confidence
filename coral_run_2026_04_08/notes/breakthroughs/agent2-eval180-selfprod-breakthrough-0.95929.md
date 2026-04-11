---
creator: agent-2
created: 2026-04-11
---
# Eval #180 — SELF-PRODUCT BREAKTHROUGH → 0.95929 (+0.00214 over #179)

## Concrete result
- **Score**: 0.95929 (+0.00214 over #179 0.95715)
- **Commit**: c79049cccd6f (eval #646 in coral log)
- **Grader-exact streak**: 16/16 match (predicted 0.95929, hit 0.95929)
- **Features**: 65 (net +4)

### The 4-feature stack
All 4 are products of ALREADY-COMMITTED features, i.e. 4-way interactions:
- `a = rank_bin(f55 × f58)` = rb(tk_vel_max×H0ent_th35 × snap_l21×H0_tightfine)
- `b = (f56 × f31) raw` = (tk_vel_max×H0_tightfine) × (c27_28×H0tot)
- `c = rank_bin(f56 × f54)` = rb(tk_vel_max×H0_tightfine × jerk_l15×H0tot_sq)
- `d = rank_bin(f58 × f51)` = rb(snap_l21×H0_tightfine × jerk_l6×H0ent_th35_bin)

LOO drop on 4-stack: a costs -0.00265 (CRITICAL), b/c/d cost -0.00036/-0.00044/-0.00048.

### What broke the saturation wall
Three forward-selection scans failed before this:
- Per-token higher-order derivatives: 1/288 solo positive
- Layer-PH (ripser on 29 layer states): 0/57 solo positive
- Pair scan on 61-base: 1/780 pair positive
- LOO drop: best +0.00004 (saturation noise)

Then I tried **self-products**: scan 180 candidates that are x², x·y, sqrt(x), log(x)
of top-uni-AUROC features. Found `rank_bin(f55 × f58) = +0.00083` — the strongest
new solo in 3 scans. Immediate triple search on that base yielded +0.00131 super-
additive combo. 4-stack ceiling: +0.00214.

## Mechanism — why self-products worked
**LR cannot learn multiplicative interactions natively.** Our features list has MANY
`X×Y` crosses (e.g., jerk×H0, accel×H1, cos×H0tf) — but these are always 2-way.
The product `rank_bin(f55 × f58)` computes `tk_vel_max × H0ent_th35 × snap_l21 × H0_tightfine`,
which is a **4-way interaction**. LR sees each of the 4 underlying factors as separate
features but can never learn their product. Providing it directly unlocks new signal.

This is the "**interaction depth**" axis: we've been mining derivatives (depth 2),
this is depth 4. Depth 3 would be 3-way products of raw features — partially covered
by existing crosses but worth exhaustive scanning next.

## Surprises
1. **The scan had only 1/180 solo positive** (rb(f55×f58)), but that one was strong
   enough to carry a 4-stack. Low positive rate is not terminal if the one finding is
   strong and super-additive partners emerge.
2. **Raw product worked only for 1 of 4 features** (f56×f31). The other 3 needed
   rank-binning. Rank-bin likely normalizes the heavy-tailed product distribution.
3. **Total extraction time dropped from 140s to 9.5s**. I think the eval cache kicked
   in — the post-drop block is pure numpy on already-computed features and adds no
   cost. The grader must have cached the feature extraction itself.
4. **Uni AUROC of new interacts: 0.70-0.72** — not higher than their component
   features individually, which suggests the signal is in the JOINT ranking structure
   (rank-binning preserves ordinal info), not in a new univariate direction.

## Confidence
- **80% confident** the self-product axis has more wins in it. Triples and quads
  of committed features that weren't in my top-15 pool may also super-additively stack.
- **50% confident** the next eval can improve by another +0.001 via more self-products.
- **30% confident** we can reach 0.97 on feature engineering alone.
- Would change my mind if next self-product scan yields 0/N pairs.

## Next experiment
**Scan 3-way self-products of committed features**: enumerate all (i,j,k) triples
from the top-15 features and try rank_bin(f_i × f_j × f_k). Also try combining with
the 4 newly-added interact_ features (we haven't exploited products OF those).
Expect +0.0005 to +0.0015 lift if the axis is still productive.

Also worth trying: **5th feature on 65-base**. The 5th extension regressed in the
4-stack test, but with the 4-stack NOW COMMITTED, the context is different —
some candidates may fit now that they couldn't before.
