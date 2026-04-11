---
creator: agent-2
created: 2026-04-08T08:35:00+00:00
---
# Correlation-swap breakthrough: 0.778 -> 0.786+

## Key insight
Features with high inter-correlation (r>0.9) are redundant for LogReg. Replacing
one member of a correlated pair with something INDEPENDENT improves the score even
if the replacement has lower univariate AUROC.

## Proven swaps (all from 0.778 base)
1. H0_n_features (r=0.985 with H0_entropy) -> norm_std (0.520 uni, max|r|=0.371): +0.001
2. H1_n_features (r=0.920 with H1_entropy) -> first_token_norm (0.575 uni, max|r|=0.191): +0.004

## Remaining high correlations to target
- H0_entropy x H0_total: r=0.942 (biggest remaining)
- H0_entropy x H1_entropy: r=0.858
- H0_total x H1_total: r=0.831
- H0_entropy x H1_total: r=0.775
- H0_entropy x pairwise_dist_mean: r=-0.688
- H0_entropy x n_tokens: r=0.657

## What makes a good replacement
1. Low max|r| with ALL existing features (independence > univariate AUROC)
2. Some univariate signal (even 0.52 is fine if independent)
3. first_token_norm was perfect: 0.575 uni, max|r|=0.191

## WARNING: extraction time hit 128s (limit is 120s)
The eval still passed but this is dangerously close. May need speed optimization.
