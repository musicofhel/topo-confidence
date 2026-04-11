---
creator: agent-3
created: 2026-04-10T21:05:00Z
---
# 17-feature ceiling at 0.8682 — swap failures diagnosed

## Context
Leader is agent-2's commit e61608b8 at 0.8682 (17 features, last added
tok_layer_min_dist — cross-modal PCA distance, uni=0.40 but +0.0052 multivariate).

## What I tried
- **All 18th-feature ADDs regressed** to 0.862x (tok_layer_last_min_dist,
  tok_nearest_layer_mean_idx, tok_svd_entropy, n_tokens, H0_entropy_cosine).
  EPV ceiling: 57 pos / 17 vars = 3.35.
- **DROP** layer_norm_ratio_midlast -> 16 feats -> 0.8629 (regression).
- **SWAP** layer_norm_ratio_midlast -> n_tokens: 0.8596 (regression).
- **SWAP** layer_pc2_early_ratio (uni 0.520, "weakest") -> tok_layer_mean_min_dist
  (uni 0.664): 0.8631 (regression from leader).

## Surprise
The "weakest" features by univariate AUROC (layer_pc2_early_ratio 0.520,
layer_norm_ratio_midlast 0.508, tok_layer_min_dist 0.506) are NOT droppable.
Their multivariate contribution is orthogonal signal. New features with much
higher uni (0.66+) regressed when replacing them.

## Why
- High-uni new features like tok_layer_mean_min_dist are CORRELATED with
  existing cross-modal features (tok_layer_min_dist). Both built from the
  same cross_dists matrix.
- Low-uni features like layer_pc2_early_ratio exist in ORTHOGONAL directions
  that LR uses despite their weak standalone signal.
- Conclusion: adding features from the same family as existing top features
  rarely helps at the ceiling.

## What I will try next
Fundamentally different direction: layer-state H1 persistence entropy or
layer velocity entropy. Every current feature is H0-based (connected
components) or magnitude/distance-based. Neither H1 on layer_states nor
layer-velocity-distribution-entropy is exploited by the leader.

## Confidence
60% one of these will improve score. 40% they'll also regress, in which case
the true ceiling may be ~0.87 for this feature paradigm, and we need a
non-topological breakthrough.

## Update after 3 SWAP attempts on leader (all regressed)
- SWAP layer_pc2_early → tok_layer_mean_min_dist: 0.8631 (-0.0051)
- SWAP layer_pc2_early → layer_h1_total (H1 on layer path): 0.8589 (-0.0093).
  layer_h1 uni 0.504 — layer trajectories don't form loops in PCA space.
- SWAP first_token_norm → bridge_silhouette (uni 0.613 > 0.577): 0.8448 (-0.0234).
  first_token_norm despite weak uni is MORE load-bearing than bridge_silhouette.

## Key insight
CLAUDE.md's "bridge silhouette is best structural feature" claim is STALE —
true only relative to pre-leader (0.75 era). On the 0.8682 leader, swapping it
in for first_token_norm is a 0.023 regression. first_token_norm's multivariate
contribution matters despite its weak uni.

## Next direction
Trying traj-vs-layer direction cosine: cos(token_end - token_start,
layer_end - layer_start) in shared PCA space. A single scalar that captures
whether token path and layer path move in aligned directions. Orthogonal to
all distance/magnitude/topology features.
