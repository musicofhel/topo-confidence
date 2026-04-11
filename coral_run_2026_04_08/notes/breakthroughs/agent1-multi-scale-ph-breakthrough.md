---
creator: agent-1
created: 2026-04-10
---
# BREAKTHROUGH: Multi-scale PH breaks the 0.8171 ceiling

## Result
New best: 0.8202 (was 0.8171, +0.0031) via adding H0_entropy_thresh50 as 12th feature.

## What changed
Added a SECOND H0 persistence entropy at a TIGHTER threshold (50th pct of distances)
alongside the existing H0 entropy at 75th pct. Same PH diagram computation, just
thrown-away at shorter distance.

## Why it works
- H0 entropy at thresh 75: 0.717 univariate
- H0 entropy at thresh 50: 0.721 univariate (the new feature!)
- These capture DIFFERENT structural scales:
  - 75th pct: medium-range connectivity between token clusters
  - 50th pct: tight local connectivity (only nearby tokens connect)
- The two features are correlated but not redundant; the LR can exploit the
  difference between them as a richer signal.

## Implication
The 11-feature recipe was saturated because all features described a single
filtration scale. Adding a second scale opened a new axis. Multi-scale PH is
the unlock.

## Next ideas
- Add H0_entropy at thresh 25 (even tighter) as 13th feature
- Add H0_total_persistence at thresh 50 (total, not entropy)
- Add a nonlinear transform to H0_entropy_thresh50
- Try H0 entropy at thresh 100 (no threshold) too

## Key takeaway
"Tight local optimum" usually means you're optimizing along ONE axis. Adding a
fundamentally different axis (here: scale) bypassed the ceiling.
