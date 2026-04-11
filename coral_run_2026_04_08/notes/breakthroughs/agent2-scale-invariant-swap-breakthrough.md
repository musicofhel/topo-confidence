---
creator: agent-2
created: 2026-04-10T15:20:00-04:00
---
# Scale-invariant ratio swap breakthrough: 0.8353 → 0.8382

## What worked

Replaced `last_token_centroid_dist` (idx 7) with the ratio
`last_dist / pw_mean` (last-token distance from centroid, normalized by
trajectory spread). Both feature 5 (pairwise_dist_mean) and feature 7 are
then rank-normalized and 10-bin digitized as before.

- Local 20-seed paired CV on 0.8353 base: **delta=+0.0031, se=0.0002, pos=1.00**
- Grader: **0.8353 → 0.8382 (+0.0029)** — near-perfect calibration
- Commit: 8f43a05c

## Why

`last_dist` and `pw_mean` are both spread-dependent. Ratio removes the
trajectory-length confound and isolates "how unusual is the last token
relative to spread". Raw last_dist was redundant with pw_mean.

This is a correlation swap: replace a redundant feature with an
independent-ish derivative.

## Session progress

| base | → | new | Δ | technique |
|---|---|---|---|---|
| 0.8234 | → | 0.8304 | +0.0070 | rank-normalize 3 dist features |
| 0.8304 | → | 0.8353 | +0.0049 | + 10-bin quantize |
| 0.8353 | → | 0.8382 | +0.0029 | swap last_dist → last/pw ratio |

Total: **+0.0148** across session (0.8234 → 0.8382).

## Key methodology insight

Local multi-seed paired CV with precomputed `base_scores` dict is
reliable. Delta threshold of **+0.0025 with 100% pos fraction across 20
seeds** predicts grader improvement reliably. Weaker signals (<+0.002)
are mostly noise vs grader.

The search pattern: for each promising candidate, try SWAPPING it in for
each existing feature. A candidate that is NULL as a 14th addition
(+0.0006) can become a strong SWAP (+0.0031) — because the feature it
replaces was carrying redundant information.

## Next ideas

- Swap `last5_centroid_dist` (idx 8) with `last5/pw` ratio
- Swap with `last_dist/last5_dist` (change over final tokens)
- Swap lower-performing features (first_token_norm, norm_mean) with other ratios
- Try alignment × distance interactions

---

## UPDATE: 4th breakthrough at 0.8382 → 0.8496 (+0.0114)

Swap `last5_centroid_dist` (idx 8) → `last5_dist / pw_max` (rank+10bin).

- Local 20-seed CV on 0.8382: **delta=+0.0136, se=0.0002, pos=1.00**
- Grader: 0.8382 → 0.8496 (+0.0114)
- Commit: 6dcc7bd1

### Why so huge?

Previous `last5_centroid_dist` had uni AUROC **0.508** — DEAD after
rank+digitize (same distribution as pw_mean, became redundant).
Swapping to `last5/pw_max` revived it: uni 0.508 → 0.623.

pw_max is heavier-tailed than pw_mean — breaks the correlation-collapse.

### Session progress

| base | → | new | Δ | technique |
|---|---|---|---|---|
| 0.8234 | → | 0.8304 | +0.0070 | rank-normalize 3 dist features |
| 0.8304 | → | 0.8353 | +0.0049 | + 10-bin quantize |
| 0.8353 | → | 0.8382 | +0.0029 | swap last_dist → last/pw_mean ratio |
| 0.8382 | → | 0.8496 | +0.0114 | swap last5_dist → last5/pw_max ratio |

**Total session delta: +0.0262** (0.8234 → 0.8496).

### Methodology insight: hunt for "dead" features

Check uni AUROC after each transform. If a feature drops to ~0.5, it
became degenerate via rank+digitize collapsing onto another feature's
distribution. Fix: swap with a scale-normalized version using a
DIFFERENT denominator (max vs mean, std vs mean) to break correlation.

Dead feature signature: uni ~0.5 AND same frac_zero/range as another.

---

## UPDATE: 5th breakthrough at 0.8532 → 0.8582 (+0.0050)

Added `layer_pca_var_ratio` as 14th feature (rank+10bin): fraction of
total SVD variance captured by the top PC of the (29, 1536) layer_states
trajectory per problem.

- Local 10-seed CV on 0.8532 base: +0.0044 (se=0.0002, pos=1.0)
- Grader: 0.8532 → 0.8582 (+0.0050) — slight UNDER-prediction (nice!)
- Commit: 7d77cbc5

### Why this worked — orthogonal weak signal

**Uni AUROC: only 0.538** — barely above random. Yet adds +0.0050
multivariate. This violates the "weak feature trap" (<0.60 usually
regresses). The reason: its signal is **orthogonal** to all existing
features, which are derived from the TOKEN trajectory (not layer states).

layer_states (29, 1536) was the key unexploited data source per task
description. The first layer_states-derived feature to actually help.

### Methodology update: weak-feature-trap is about correlation, not uni AUROC

A weak uni AUROC feature CAN help — if its information is orthogonal to
the existing feature basis. The "trap" only applies when a weak feature
is ALSO correlated with existing features (then it just adds noise to
their scale).

### Session progress

| base | → | new | Δ | technique |
|---|---|---|---|---|
| 0.8234 | → | 0.8304 | +0.0070 | rank-normalize 3 dist features |
| 0.8304 | → | 0.8353 | +0.0049 | + 10-bin quantize |
| 0.8353 | → | 0.8382 | +0.0029 | swap last_dist → last/pw_mean ratio |
| 0.8382 | → | 0.8496 | +0.0114 | swap last5_dist → last5/pw_max ratio |
| 0.8496 | → | 0.8532 | +0.0036 | agent-3: rank+digitize first_token_norm+norm_mean |
| 0.8532 | → | 0.8582 | +0.0050 | add layer_pca_var_ratio (layer_states SVD) |

**Total session delta: +0.0348** (0.8234 → 0.8582).

### Next: More layer_states features

Now that we know layer_states has orthogonal signal, search aggressively:
- PC2/PC1 ratio (second-PC importance)
- Layer-to-layer cosine similarities
- PH on layer trajectory
- Layer norm profile features
- Early vs late layer divergence

---

## UPDATE: 6th breakthrough at 0.8582 → 0.8618 (+0.0036)

Added TWO more layer_states features together:
- `layer_pc2_early_ratio` (15th): variance share of 2nd PC in layers 0-10
- `layer_norm_ratio_midlast` (16th): norm[-1] / norm[middle layer]

Both rank+10bin digitized.

- Local 10-seed pair CV on 0.8582 base: +0.0049 (se=0.0002, pos=1.0)
- Grader: 0.8582 → 0.8618 (+0.0036) — 74% efficiency (lower than usual)
- Commit: d4228c77

### Weak-signal dilution observation

When adding PAIRS of weak features (uni ~0.5), local→grader efficiency
drops from 100% (single strong feature) to ~74% (weak pair). Each
feature carries some noise that compounds. Going forward: prioritize
single candidates with local delta > 0.003, don't pile multiple weak
features per eval.

### Session progress

| base | → | new | Δ | technique |
|---|---|---|---|---|
| 0.8234 | → | 0.8304 | +0.0070 | rank-normalize 3 dist features |
| 0.8304 | → | 0.8353 | +0.0049 | + 10-bin quantize |
| 0.8353 | → | 0.8382 | +0.0029 | swap last_dist → last/pw_mean ratio |
| 0.8382 | → | 0.8496 | +0.0114 | swap last5_dist → last5/pw_max ratio |
| 0.8496 | → | 0.8532 | +0.0036 | rank+digitize first_token_norm+norm_mean |
| 0.8532 | → | 0.8582 | +0.0050 | add layer_pca_var_ratio (SVD) |
| 0.8582 | → | 0.8618 | +0.0036 | add pc2_early + norm_ratio_midlast |

**Total session delta: +0.0384** (0.8234 → 0.8618).

All 3 layer_states features have uni AUROC 0.5-0.54 (near random) yet
together contribute +0.0086 multivariate. Orthogonal signal in action.
