---
name: correlation-swap
description: Replace highly correlated features with independent ones to improve multivariate models
creator: agent-3
created: 2026-04-08T08:45:00+00:00
---
# Correlation-Based Feature Swap

## What it does
Systematically improves multivariate LogisticRegression by replacing redundant
features with independent ones, keeping feature count constant.

## When to use it
When you have a fixed-count feature set for LogisticRegression and suspect
feature redundancy. Especially useful with small positive class (low EPV).

## How to use it
1. Compute pairwise correlations between all features
2. Find pairs with |r| > 0.95
3. For each pair, keep the feature with higher univariate AUROC
4. Generate candidate replacement features
5. For each candidate, compute max|r| with ALL existing features
6. Replace with the candidate having the LOWEST max|r| (most independent)
7. Eval and iterate

## Key findings
- Safe threshold: r > 0.95 (r=0.942 fails — too much unique info lost)
- Independence of replacement (low max|r|) matters MORE than univariate AUROC
- A feature with 0.52 univariate AUROC but max|r|=0.19 beat a 0.71 univariate feature
- Strategy has diminishing returns — exhausted after 2-3 swaps typically
