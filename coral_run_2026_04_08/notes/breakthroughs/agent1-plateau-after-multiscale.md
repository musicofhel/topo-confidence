---
creator: agent-1
created: 2026-04-10
---
# Plateau at 0.8202 after multi-scale breakthrough

## Recent consecutive regressions (all from 0.8202 base):
- thresh 60 for 2nd scale: 0.8158
- thresh 80 for 1st scale: 0.8113
- transform on thresh50: 0.8119
- H1 entropy at thresh 50 as 13th: 0.8111
- H0 total at thresh 50 as 13th: 0.8170
- H0 entropy thresh 25/35 as 13th: 0.8141, 0.8187
- Replace last5 with thresh35: 0.7923
- Early-layer alignment as 13th: 0.8109
- N_PCA=52: 0.8076

## Key finding
The 12-feature state is a VERY tight local optimum. Even features with 0.727
univariate AUROC (thresh35) hurt multivariate. Pattern: any 13th feature adds
redundancy to the PH-entropy cluster and disrupts the balance.

## Unexplored directions
- **Trajectory dynamics**: velocity, acceleration, turning angle (non-PH)
- **Distance metrics**: cosine/Manhattan for PH (Euclidean is default)
- **Layer-state PH**: compute PH on 29 layer states directly (only tried once)
- **Intrinsic dimensionality**: MLE / TwoNN estimator
- **Graph spectral features**: Laplacian of token similarity graph

## New direction: trajectory velocity
Next: add mean token-to-token distance as a 13th feature. This is
structurally orthogonal to PH (which ignores token ordering) and to
position-based features (last_token_*). Could provide the missing axis.
