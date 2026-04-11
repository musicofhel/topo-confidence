---
name: drop-add-cycle
description: Break saturation by dropping load-bearing-in-name-only features and adding new signal in their place. Use when forward selection stalls despite testing multiple feature families.
creator: agent-2
created: 2026-04-11T05:30:00+00:00
---

# DROP-ADD cycle skill

## What it does
When forward-selection single-ADD scans return <+0.001 deltas across
hundreds of candidates in multiple families, the LR model has
saturated its current feature subspace. Dropping 1-2 redundant features
(even ones with negative individual drop-deltas) can free model capacity,
enabling subsequent ADDs that were blocked.

## When to use
Trigger conditions (all must hold):
1. Current base > 0.90 AUROC with 30+ features.
2. Last 2+ single-ADD scans (200+ candidates each) returned fewer
   than 3 candidates with delta > +0.0005.
3. The family rotation strategy ("switch feature axis") has already
   been tried and also saturated.

**Do not use** if single-ADD scans are still delivering +0.001 or
better — pure forward selection is more efficient when it works.

## How to use

### Phase 1: LOO drop scan
Run `scripts/scan_drop.py` (adapted from scan_drop.py in root). For each
feature index `i`, compute `grader_exact_cv(X_base[:, keep_all_except_i], y)`.
Sort by delta descending. Note any positive-delta features.

**Expected output** at a saturated base: 1-3 features with positive
delta (often +0.00005 to +0.00050), rest negative.

### Phase 2: 2-drop super-additive scan
Take the top 6-10 drop candidates (by delta, including marginal
negatives). Try ALL pairs — compute `grader_exact_cv` on
`X_base[:, all_except_d1_d2]`.

**Look for super-additive pairs**: pairs where the combined drop is
GREATER than the sum of 1-drop deltas. These indicate features that
were jointly encoding a pattern now expressible by other features.

In eval #175 agent-2 observed:
- f29 solo drop: -0.00024
- f42 solo drop: -0.00044
- (f29, f42) combined drop: **+0.00083** (super-additive by +0.00151)

### Phase 3: Rescan ADDs on reduced base
After dropping the best pair (or single if no super-additive pair), run
the full add-scan (1000+ candidates) on the reduced base. Candidates
that were neutral/negative on the full base may now be positive because
LR has room to express their signal.

In eval #175: `cos(L7, L11) raw` had near-zero delta on the 49-feat
base but +0.00087 on the 47-feat reduced base.

### Phase 4: Commit the full cycle
One eval commits: DROP d1 + DROP d2 + ADD new_feature. Expected total
delta = drop_delta + add_delta. Verify locally via `grader_exact_cv` on
the final feature matrix before committing.

## Implementation notes

### Clean index management
When the extract_features code is already built on fixed column indices,
the simplest DROP implementation is:
```python
# At end of extract_features, before return:
drop_cols = [29, 42]
features = np.delete(features, drop_cols, axis=1)
out_names = [n for k, n in enumerate(FEATURE_NAMES) if k not in drop_cols]
return features, out_names
```
This keeps ALL post-transform cross-product references valid (they use
pre-drop indices) and only affects the final output.

### Why super-additive drops happen
When two features f_a and f_b together encode a pattern that's already
partially captured by f_c, LR assigns split coefficients to f_a and
f_b with opposite signs to approximate the residual. Dropping only
one leaves the other with the wrong sign for the full pattern.
Dropping both lets LR redistribute to f_c cleanly.

### Common DROP candidates
- Raw cosines between adjacent layers (they overlap with layer-angle
  features)
- Cross-products involving both early and late layer data (overlap
  with multiple isolated features)
- Features added late in forward selection (often marginal)

## Track record
Eval #175 (agent-2): 49-feat 0.94919 -> 48-feat 0.95089 via DROP-ADD
cycle (+0.00170 total). The single-ADD scans had saturated to <+0.0005
across 372 + 57 = 429 candidates.
