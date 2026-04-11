---
creator: agent-2
created: 2026-04-11T14:10:00-07:00
topic: composition-depth feature engineering
scores: 0.95715 -> 0.96404 (+0.00689)
evals: 179-183
---
# Self-Product / Composition-Depth Era: 0.95715 -> 0.96404

**Summary:** After forward-selection on raw base features saturated at #179
(0.95715), I unlocked 4 consecutive evals of gain by building **products of
existing committed features** — specifically, recursive rank-binned
multiplicative interactions that a linear classifier cannot learn natively.
Total gain +0.00689 in 4 evals.

## Conclusion (read-this-first)
When forward-selection on raw features saturates, DON'T conclude the data is
exhausted. Instead: try multiplicative interactions of your top committed
features, rank-bin the result, and add as new columns. If that pays, escalate
to **products of products** for another cycle. Each escalation level opens
a new combinatorial universe.

LR with N base features can natively learn any linear combination of f_i,
but CANNOT learn f_i * f_j without explicit cross-product features. Pre-
computing the cross-product as a scalar and rank-binning (to handle the
heavy tail from multiplying real-valued features) exposes the 2-way signal
to LR. Recursing (product of products) exposes 4-way, 8-way, etc.

## The numerical story
| Eval | Composition | Stack width | Delta | Cum score | Hit rate |
|------|-------------|-------------|-------|-----------|----------|
| #179 | (pre-pivot: raw j23 jerk-extension) | 4 | +0.00182 | 0.95715 | 2.0% |
| #180 | 2-way self-product (depth 1) | 4 | +0.00214 | 0.95929 | 1.1% |
| #181 | 3-way self-product + 1 recursive (depth 2) | 4 | +0.00123 | 0.96052 | 0.7% |
| #182 | 4-way product + 3 recursive (depth 2) | 5 | +0.00221 | 0.96273 | 3.6% |
| #183 | depth-4 recursive triple | 3 | +0.00131 | 0.96404 | 5.0% |

**Stack widths have shrunk** in #183 — first signal of axis exhaustion.

## Key findings

### Finding 1: rank_bin is essential for products of >= 3 features
Products of 3+ real-valued features become extremely heavy-tailed. LR with
`class_weight=balanced` collapses or destabilizes on such features. Rank-
binning (`np.digitize(np.argsort(np.argsort(x))/n, 10_bin_edges)`) restores
a uniform scale and dramatically improves the solo delta (often from 0 to
+0.0003).

Cite: Every #180-#183 added feature has `_rb_` or `_b_` suffix. Raw (non-rank-
binned) 3-way products were scanned and almost always underperformed — only
2 raw 2-way products survived (#180 `int_b`, #181 `int_g`, `int_h`).

### Finding 2: Recursive features are stronger than pure ones
Recursive = product that includes a prior interact_* feature (itself a
product) as an operand. In #182 and #183, the *top solo* was always recursive:
- #182 top: `p4_32_56_53_51_b` was pure 4-way but 4 of top 7 were recursive.
- #183 top: `rec4_11_39_55_i72_b` uses f72 = #182 `interact4_32_10_x_ii_rb`.
- #183: all 3 committed features are recursive.

**Why:** Recursive features encode higher-order moments (e.g., depth-4 recursive
= rb(f_a * f_b * rb(f_c*f_d*f_e*f_f)) which is effectively a 6-way interaction).
LR would need O(N^6) explicit cross-terms to learn the same signal.

### Finding 3: Stack width decays as depth increases
| Eval | Depth | Stack width |
|------|-------|-------------|
| #180 | 1 | 4 |
| #181 | 2 | 4 |
| #182 | 2 | 5 |
| #183 | 3-4 | 3 |

**Mechanism:** At higher composition depths, new features become linear
combinations of earlier deep features after scaling + rank-binning. They
duplicate signal rather than adding it.

### Finding 4: Hit-rate is deceptive — saturation shows in stack width, not hit rate
#183 had a 5.0% hit rate (highest of the era), but the committed stack
shrunk from 4/5 to 3. Hit rate counts solo positives; stack width measures
independence of signal. **Use stack-width regression as the stopping signal**,
not hit rate.

## What we tried and failed (fair reporting)

### Post-#179 dead ends (3 scans, 0 meaningful positives)
1. **Layer-PH on 29 layer states as a point cloud** (`scan_layer_ph.py`):
   0/57 positive. The point cloud is too small (29 points in 1536D) for
   stable PH.
2. **Per-token higher derivatives × PH** (`scan_token_higher_derivs.py`):
   1/288 positive, max +0.00004. Token-level signal is saturated.
3. **Pair scan of existing committed features** (`scan_pair_existing.py`):
   1/780 positive. Raw products pay only with rank-binning.

### Dead ends inside the self-product axis
- **Raw (non-rank-binned) products** for 3-way and higher: too heavy-tailed.
- **4-stacks after #183**: adding a 4th/5th feature regresses the triple.
- **Products of #181 interact_g/int_h (raw)** — did not make top-40.

## When to apply this technique (reusable recipe)

See skill `.claude/skills/recursive-interaction-scan/`. Trigger conditions:
1. Forward selection on raw features yields < 5% hit rate in a 500+ cand scan.
2. Current top feature's uni AUROC ≥ 0.70 (strong enough to survive binning).
3. You have ≥ 10 committed features with varied axes (so products sample
   independent modes).
4. Budget for a ~1500-cand scan (~10-15 min wall time).

## Confidence and remaining headroom
- **80% confident** this axis has 0-1 more evals left. Stack-width shrinkage
  is unambiguous. Maximum additional gain: ~+0.002.
- **40% confident** we can still reach 0.97 without a completely new feature
  family. Would reconsider if a 5-way / double-rb-nested variant pays >+0.0015.
- **Next experiment to run first**: "double-rank-bin" = `rb(rb(f_a*f_b) * rb(f_c*f_d))`
  — if this pays, axis is alive; if not, pivot to Mapper/graph-Laplacian on
  layer states.

## Evidence
All attempts logged in `.coral/attempts/` under commits:
- 6ef1fb61 #179: j23 jerk-extension (pre-pivot baseline)
- c79049cc #180: self-product breakthrough (+0.00214)
- 51a3c945 #181: 3-way self-product + recursive (+0.00123)
- 3975e5ad #182: 4-way product + recursive (+0.00221)
- 0f9c7405 #183: depth-4 recursive triple (+0.00131)

Total sequence gain: +0.00689 over 4 evals (0.95715 -> 0.96404).
Grader-exact match streak: 19/19.
