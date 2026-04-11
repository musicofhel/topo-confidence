---
creator: agent-3
created: 2026-04-08T09:18:00+00:00
---
# Plateau Analysis: 4 consecutive failures (evals 39-42)

## Results
| Eval | Change | Score | Delta |
|------|--------|-------|-------|
| 39 | layer_cosine_min swap | 0.7772 | -0.010 |
| 40 | sqrt(n_tokens) | 0.7852 | -0.002 |
| 41 | Angular PH (unit sphere) | 0.7552 | -0.032 |
| 42 | Mahalanobis meta-feature | 0.7582 | -0.029 |

## Root cause: feature set is LOCKED, ALL modifications are noise
- 14-feature set is an optimized package. ANY change = net loss.
- last5_centroid_dist (uni=0.507) confirmed irreplaceable AGAIN (eval 42)
- Even agent-1's high-AUROC replacement (answer_H0_max_lifetime, uni=0.668) → 0.7818 (regression)
- The features are load-bearing not for univariate signal but for INDEPENDENCE structure

## What HASN'T been tried
- Persistence images (persim library): data-driven PH representation via PCA
- Different DR before PH (kernel PCA, random projections)
- Fundamentally different feature engineering paradigm (not hand-crafted scalars)

## New direction: Persistence images → PCA → replace PH scalars
Instead of 5 hand-crafted PH scalar features, use persim to convert diagrams to
persistence images and PCA them. This finds the directions of maximum VARIANCE
in diagram space, which might extract MORE signal from the same PH computation.
