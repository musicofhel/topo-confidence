---
creator: agent-3
created: 2026-04-08T04:30:00+00:00
---
# BREAKTHROUGH: Position-dependent features → 0.7581 (+0.025)

## What happened
Added last_token_centroid_dist (distance of final answer token from cloud centroid)
and last5_centroid_dist (mean distance of last 5 tokens from centroid).

Score jumped from 0.7327 to 0.7581 — the largest single improvement so far.

## Why this works
All previous features treated tokens as an UNORDERED point cloud. But tokens have
sequential structure — the last token is the answer. How far the answer token sits
from the cloud centroid captures whether the model's output is "typical" or "outlier"
relative to its internal representation. This is genuinely orthogonal to PH features.

## Surprise
last_token_centroid_dist has univariate AUROC of only 0.650, and last5_centroid_dist
is 0.502 (near chance). Yet together they boosted multivariate AUROC by 0.025.
This means the value is in INTERACTION with other features, not standalone prediction.

## Key insight
Sequential/positional structure of tokens is a rich, unexploited signal source.
The model's answer (last tokens) vs its reasoning (earlier tokens) have different
geometric properties that correlate with correctness.

## Current best: 13 features, 0.7581
H0 PH (4) + H1 PH (3) + bridge_sil + n_tokens + pw_dist + norm_mean + last_tok_dist + last5_dist

## Next ideas
1. Drop last5_centroid_dist (0.502 univariate) — might hurt or help
2. More position features: first_token_centroid_dist, first_last_distance
3. Ratio: last_token_dist / pairwise_dist_mean (normalized answer outlier-ness)
4. Cluster assignment of last token (which k=2 cluster does the answer land in?)
5. Try answer token distance in RAW space (not PCA-reduced)
