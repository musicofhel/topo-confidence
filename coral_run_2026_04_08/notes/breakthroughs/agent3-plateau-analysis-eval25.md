---
creator: agent-3
created: 2026-04-08T05:40:00+00:00
---
# Plateau Analysis: Stuck at 0.776 after 25 evals

## Score trajectory (last 6 evals)
- Eval 20: 0.7766 (single direction alignment — my best)
- Eval 21: 0.7705 (different layer pair — worse)
- Eval 22: 0.7721 (15th feature — worse)
- Eval 23: 0.7653 (feature swap — worse)
- Eval 24: 0.7456 (PH orthogonal — disaster)
- Eval 25: 0.7764 (SVD 2 PCs — noise)

## What I've exhausted
1. Feature addition (15th feature wall)
2. Feature replacement (all 14 load-bearing)
3. Layer pair tuning (full span is best)
4. PH subspace modification (layer-orthogonal killed PH)
5. SVD vs single direction (essentially same score)

## Why I'm stuck
The 14 features + LogReg combination has a STRUCTURAL ceiling at ~0.777.
With 57 positives and 14 features, LogReg can't fit more complex boundaries.
The features themselves are near-optimal for LINEAR classification.

## New direction: Feature preprocessing
Instead of changing WHICH features, change HOW they're represented.
Rank-transform all features to uniform [0,1] → removes outliers,
normalizes scales, makes L2 regularization act uniformly.
This is the most different approach I haven't tried.

## Alternative: Multi-resolution PH
Compute PH at multiple PCA dimensions and select best.
Or: completely restructure to use 10 features (more regularization room).
