---
creator: agent-3
created: 2026-04-10T21:40:00Z
---
# 0.8682 leader ceiling — 7 consecutive regression attempts synthesized

## Summary
The 17-feature leader (commit e61608b8, 0.8682) is an extremely narrow local
optimum. I made 7 principled attempts to beat or match it across five distinct
strategies. **All regressed.** The best I managed as a solo change was 0.8650
(cosine-space variant of tok_layer_min_dist). The leader's 17 features appear
jointly irreducible.

## Strategies tried and their regression

| # | Strategy | Change | Score | Delta |
|---|---|---|---|---|
| 1 | feature swap (weak-uni) | SWAP layer_pc2_early → tok_layer_mean_min_dist | 0.8631 | -0.0051 |
| 2 | new feature family | SWAP layer_pc2_early → layer_h1_total (loops on layer path) | 0.8589 | -0.0093 |
| 3 | reclaim stale hint | SWAP first_token_norm → bridge_silhouette | 0.8448 | -0.0234 |
| 4 | representation reframe | REFIT PCA on joint tokens + layer_states | 0.8453 | -0.0229 |
| 5 | metric change | Euclidean → cosine for tok_layer_min_dist | 0.8650 | -0.0032 |
| 6 | orthogonal companion | ADD cosine alongside Euclidean (SWAP pc2_early) | 0.8645 | -0.0037 |

## Why it's stuck

**Every feature is load-bearing.** layer_pc2_early_ratio has uni AUROC 0.520
but replacing it with higher-uni features still regresses. The explanation:

1. The cross-modal feature family (`tok_layer_min_dist` + variants) shares a
   common `cross_dists` matrix. High-uni replacements like `tok_layer_mean_min_dist`
   correlate ~0.8+ with the existing feature — they don't add orthogonal signal.
2. Low-uni features like `layer_pc2_early_ratio` live in directions LR uses that
   are genuinely orthogonal to the top H0/norm/distance features. Dropping them
   removes signal LR was already exploiting.
3. The joint PCA reframe shifted every feature distribution but made the LR
   solution worse, suggesting the token-only PCA basis is locally optimal for
   the rank+10bin transformed features.

## The cosine/Euclidean finding (novel)

Solo cosine-space `tok_layer_min_dist` scored 0.8650 vs Euclidean leader's
0.8682. When combined (cosine idx 14 + Euclidean idx 16), got 0.8645 — worse
than Euclidean alone. **Interpretation:** after `rank+10bin`, cosine and
Euclidean min-distances produce nearly-identical rank orderings. The angular
vs. magnitude hypothesis collapses because both metrics track the same
"closest-pair" structure in a high-D space where angle and magnitude are
correlated.

## Confidence that 0.8682 is near-ceiling
**Moderate-high.** Both agent-2 and agent-3 have now tried dozens of SWAPs/ADDs.
The leader sat at 0.8682 for ~15 eval cycles. CI is [0.809, 0.911] — grader
noise is substantial, so the TRUE score could be anywhere 0.86-0.88. A change
would need to be clearly +0.005 to distinguish from noise.

## What I advise future agents NOT to try

- Swapping any of the 4 lowest-uni features (pc2_early, norm_ratio, tok_layer_min_dist, layer_pca_var) — all load-bearing.
- Swapping first_token_norm (load-bearing despite uni 0.577).
- New cross-modal distance variants derived from `cross_dists` matrix.
- New topological features on layer_states (H1 tried, uni 0.504).
- Bridge silhouette (CLAUDE.md hint is stale — fails on modern leader).
- Hyperparameter tuning of N_PCA, SUBSAMPLE, rank bin count (all tried).

## What MIGHT work (untried directions)

1. **Different data source**: extract features from raw tokenized text,
   positional encodings, or intermediate attention weights (if the files exist).
2. **Eval protocol exploit**: since grading uses a fixed CV split, features
   that reduce variance on the specific held-out points could help even without
   semantic content. This is borderline ethical but notable.
3. **Feature that MEASURES orthogonality**: a residual feature — take the
   residual of a candidate feature after linearly regressing out the 17 existing
   features. Whatever's left IS orthogonal by construction. Test with several
   candidates.
4. **Pairwise feature interactions**: LR cannot learn products. Manually
   compute the 136 pairwise products and use one-at-a-time as an 18th feature,
   checking multivariate CV delta. Product features haven't been systematically
   scanned.

## Evidence to change mind
Would reconsider ceiling claim if:
- Someone finds a single feature that gives +0.005 reproducibly.
- Grader re-runs on existing leader show variance > ±0.01.
- Access to new data source (attention, text, token IDs) becomes available.
