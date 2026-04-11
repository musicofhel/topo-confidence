---
creator: agent-2
created: 2026-04-10T20:20:00
---
# Cross-modal geometry breakthrough (0.8618 -> 0.8682)

## Result
Eval #151: 0.8682 (+0.0064 grader) from adding tok_layer_min_dist as 17th
feature. Local 30-seed CV predicted +0.0052 (se=0.0016, pos=0.70). Grader
delivered 123% of predicted effect.

## The feature
Compute PCA-reduced layer states (same N_PCA=45 as tokens), then take the
minimum pairwise distance between any token and any layer state. Apply
rank+10bin quantile transform.

Measures: how close the token cloud comes to the "layer manifold".

## Why it works: orthogonality, not univariate strength
Univariate AUROC: 0.506 (near random). The signal is entirely multivariate.
- raw 1536-D version: NEGATIVE delta (-0.0019, pos=0.00)
- PCA-reduced version: +0.0052 delta

PCA reduction makes the distance focus on dimensions that matter. In raw
1536-D, cross-modal distance is dominated by norm differences which the
existing features already capture.

## Why this is a new direction
FIRST cross-modal feature combining token trajectory AND layer states in a
single scalar. Previous layer features used layer_states alone (SVD, PC2,
norm_ratio). Previous token features used trajectories alone. This is the
first feature that is a function of BOTH.

The "token cloud approaches layer manifold" signal is conceptually
different from either source's internal structure. It's a geometric
relationship between two representations.

## Failed variants
From verify_tok_lay.py:
- min_dist_raw1536: NEGATIVE (no PCA)
- min_dist_last_tok only: NEGATIVE
- min_dist_first_tok only: NEGATIVE
- median_dist_pca: NEGATIVE
- 5th_pct_dist_pca: NEGATIVE
- tok_lay_mean_dist: NEGATIVE (-0.0040)

Only the MINIMUM works, and only in PCA space. Median/5th-pct/mean all
fail. Signal is specifically in "closest-approach point", not typical
separation.

## What this opens up (next direction)
Cross-modal geometric features are a new family:
- argmin along layer axis (which layer is closest-approach at)
- argmin along token axis (which token is closest)
- Per-layer min-dist histogram stats
- Cosine-based cross (max cos between token dir and layer dir)
- Hausdorff (max of mins)
- Wasserstein between token and layer distributions

## Weak-feature-trap: another counterexample
uni=0.506, would be rejected by "drop if uni < 0.60" heuristic. Delivered
LARGEST single-feature gain of session (+0.0064). Mental model: univariate
AUROC is actively misleading for multivariate features. Multi-seed CV pair
delta is the ONLY criterion.

## Session progress
Start of session: 0.8234
After 5 prior breakthroughs: 0.8618
Eval #151: 0.8682 (+0.0448 total session)
Distance from target 0.75: +0.1182

## Confidence
HIGH that more cross-modal features will work. First one I tried hit big.
Search space is large (~20+ natural variants); only 7 tested. Underlying
insight (token/layer geometric interaction) is unexploited.

## Next experiment
Sweep cross-modal variants:
1. Which-layer-closest (argmin along layer axis)
2. Token-index-at-closest (argmin along token axis, normalized to [0,1])
3. Per-layer min-dist: std, range across layers
4. Max cosine between any token dir and any layer dir
5. Hausdorff (max of mins)
