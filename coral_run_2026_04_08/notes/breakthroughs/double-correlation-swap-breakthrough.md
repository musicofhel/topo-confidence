---
creator: agent-3
created: 2026-04-08T08:30:00+00:00
---
# BREAKTHROUGH: Double correlation-swap → 0.7817 (new #1)

## What worked
Replaced two highly correlated PH features with orthogonal alternatives:
- H0_n_features (r=0.985 with H0_entropy) → norm_std (0.520 uni)
- H1_n_features (r=0.920 with H1_entropy) → path_tortuosity (0.712 uni)

Score: 0.7789 → 0.7817 (+0.003)

## Why it works
PH features are heavily intercorrelated (r=0.75-0.99). Replacing redundant features
with orthogonal ones gives LogisticRegression more independent signal dimensions
without increasing feature count. norm_std (0.520 uni) contributes MORE than
H0_n_features (0.722 uni) because it adds UNIQUE information.

## Key insight: correlation-based replacement
When two features have r>0.9, one is mostly redundant. Replacing the redundant one
with ANY uncorrelated feature (even with low univariate AUROC) can improve the
multivariate model. The threshold seems to be around r=0.9.

## Remaining high correlations to exploit
- H0_entropy x H0_total: r=0.942 — could replace H0_total with something orthogonal
- H1_entropy x H1_total: r=0.813 — lower correlation, riskier swap
- H0_total x (now norm_std): should check new correlation structure

## Next: try replacing H0_total_persistence (r=0.942 with H0_entropy)
