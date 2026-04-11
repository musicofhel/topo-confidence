---
creator: agent-3
created: 2026-04-08T04:38:00+00:00
---
# PCA(50) improves AUROC to 0.7639 but has timing issue

## Result
PCA(30) → PCA(50): AUROC 0.7581 → 0.7639 (+0.006)
But extraction time 122.1s exceeds 120s limit.

## Why PCA(50) helps
More PCA components preserve more variance from the 1536D space. The PH computation
captures finer geometric structure that was lost at 30D. H1 features improved most:
H1_n_features 0.687→0.696, H1_total_persistence 0.646→0.656.

## Fix needed
Reduce subsample from 100 to ~80 to keep ripser fast in 50D.
Or try PCA(40) as a compromise (faster, still more than 30).
