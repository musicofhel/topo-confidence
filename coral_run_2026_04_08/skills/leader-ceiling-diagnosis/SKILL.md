---
name: leader-ceiling-diagnosis
description: Diagnose the 17-feat leader (0.8682) ceiling — what SWAPs/REFRAMEs have been tried and failed, so future agents avoid wasting evals
creator: agent-3
created: 2026-04-10T21:30:00Z
---

# Leader ceiling at 0.8682 — what doesn't work

## Current leader (as of 2026-04-10)
- Commit `e61608b8` (agent-2) — ADD `tok_layer_min_dist` as 17th feature.
- Score: 0.8682 (50-fold StratifiedKFold LR balanced).
- Weakest uni AUROC: tok_layer_min_dist (0.506), layer_norm_ratio_midlast
  (0.508), layer_pc2_early_ratio (0.520), layer_pca_var_ratio (0.538),
  first_token_norm (0.577). These are ALL load-bearing despite weak uni.

## Confirmed-failing moves on the leader base (all regressed)

| Change | Score | Delta |
|---|---|---|
| Euclidean -> cosine tok_layer_min_dist | 0.8650 | -0.0032 |
| ADD n_tokens (18th) | 0.8621 | -0.0061 |
| ADD tok_layer_last_min_dist (18th) | 0.8621 | -0.0061 |
| ADD tok_svd_entropy (18th) | 0.8621 | -0.0061 |
| ADD H0_entropy_cosine (18th+19th) | 0.8631 | -0.0051 |
| SWAP layer_pc2_early -> tok_layer_mean_min_dist | 0.8631 | -0.0051 |
| SWAP layer_pc2_early -> layer_h1_total (H1 on layer path) | 0.8589 | -0.0093 |
| DROP layer_norm_ratio_midlast (16 feats) | 0.8629 | -0.0053 |
| SWAP first_token_norm -> bridge_silhouette | 0.8448 | -0.0234 |
| REFRAME: joint PCA (tokens + layer_states) | 0.8453 | -0.0229 |
| HYPERPARAM: N_PCA 45->60 | 0.8371 | -0.031 |
| HYPERPARAM: SUBSAMPLE 100->120 | 0.8442 | -0.024 |
| HYPERPARAM: rank bins 10->12 | 0.8569 | -0.0113 |
| HYPERPARAM: rank bins 10->8 | 0.8505 | -0.0177 |

## Core pattern
**Every feature is load-bearing.** Features with uni < 0.55 contribute
orthogonal multivariate signal and cannot be dropped without regression.
New features with higher uni AUROC regress because they correlate with
existing features (especially the cross-modal family: tok_layer_min_dist
and its variants share the same cross_dists matrix).

## What NOT to try next
1. SWAP any feature with a new mean/median/percentile of the same
   cross_dists matrix — these correlate with tok_layer_min_dist.
2. DROP any "weak uni" feature — it's load-bearing.
3. Per-feature hyperparameter tuning (PCA dim, subsample, bin count).
4. Bridge silhouette, layer H1, layer pc2 late variants — tried, regressed.
5. Cosine variant of tok_layer_min_dist — worse than Euclidean (0.8650 vs 0.8682).
6. Joint PCA fit (tokens + layer_states) — regressed to 0.8453.

## What COULD work (untried)
1. A feature from a genuinely new DATA SOURCE (e.g., attention weights if
   available, token-identity strings, positional encodings).
2. A feature that MEASURES orthogonality, e.g., residual after projecting
   onto existing features' linear span — a synthetic "orthogonal signal".
3. Grader noise is real — the 0.8682 may itself have a lucky seed. Running
   the exact leader twice might give 0.866-0.870 range. Reproducibility
   check could calibrate expectations.
4. Problem-level meta features that aren't trajectory statistics (e.g.,
   gradient-based saliency, loss curvature at eval-time).

## Key insight: the EPV wall
57 positives / 17 variables = 3.35 EPV. Adding an 18th (3.16 EPV) has
regressed in every attempt. LR balanced penalizes overfit harshly at
this EPV. The only path to >0.8682 is STRICTLY BETTER 17 features,
not more features.

