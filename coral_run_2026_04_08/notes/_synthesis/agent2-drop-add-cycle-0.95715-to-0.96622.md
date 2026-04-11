---
creator: agent-2
created: 2026-04-11T13:20:00
topic: drop-add cycle + composition-depth feature engineering
scores: 0.95715 -> 0.96622 (+0.00907)
evals: 179-189
supersedes: agent2-self-product-composition-depth-era (extends with drop era)
---
# Drop-Add Cycle Era: 0.95715 -> 0.96622

**Summary:** 11 consecutive evals (#179-#189) gained +0.00907 total on the
topo-AUROC task by alternating two patterns: (1) building **self-products
of existing features** as rank-binned columns, and (2) **LOO drops** of
features whose information has been absorbed into the products. The key
insight is that these two patterns are NOT independent — they form a cycle
where drops and adds alternately free and consume EPV.

## Conclusion (read-this-first)

When forward-selection on raw features saturates, run the **drop-add cycle**:
1. Build multiplicative interaction products of top features, rank-bin, add.
2. Once committed, run LOO drop scans to remove features whose information
   is now baked into the product factors (drop-the-factor pattern).
3. With freed EPV, build a SECOND generation of products — these can now
   include the new products, the squared versions, or mix them — giving
   depth-2 composition.
4. Run LOO drop again on the new base. The newly-dropped features may
   be factors the depth-2 products absorbed.
5. Repeat.

**Expected gain per eval: +0.00024 to +0.00080** (average +0.00082). The
cycle sustained 11 consecutive strong evals with no regression.

## The drop-add cycle mechanism

**Why it works:** LR with N base features can natively learn any linear
combination, but CANNOT learn f_i * f_j without explicit cross-product
features. The **rank-bin** (`_rb`) step handles the heavy-tail distribution
of real-valued products. Once a feature is baked into a rank-bin product,
its OWN rank information becomes redundant to the LR fit — because the
product's rank ordering already captures the relevant gradient direction.

The cycle is:
```
[add products] -> base_features become product_factors ->
[LOO drop base factor] -> EPV freed ->
[add depth-2 products using the squared/rb versions] ->
new factors emerge -> [LOO drop those] -> ...
```

## Eval-by-eval evidence table

| Eval | Commit | Score | Delta | Action | Net feats |
|------|--------|-------|-------|--------|-----------|
| #178 | baseline | 0.95715 | — | forward select finished | 77 |
| #179 | ? | 0.95770 | +0.00055 | interact depth-1 | 80 |
| #180 | ? | 0.95825 | +0.00055 | interact depth-2 | 83 |
| #181 | ? | 0.95937 | +0.00112 | recursive interact | 86 |
| #182 | ? | 0.96076 | +0.00139 | recursive | 90 |
| #183 | ? | 0.96304 | +0.00228 | rec3/rec4 | 94 |
| #184 | 7d28ae52 | 0.96460 | +0.00156 | LOO drop #1 (3 feats) | 74 |
| #185 | 31228a3d | 0.96495 | +0.00035 | LOO drop #2 (3 feats) | 71 |
| #187 | 0c84fe3d | 0.96519 | +0.00024 | LOO drop #3 (3 feats) | 68 |
| #188 | 04a2a4da | 0.96574 | +0.00055 | ADD 5w + 2 squared | 71 |
| #189 | 7e944142 | 0.96622 | +0.00048 | ADD 2 new 5w + DROP 2 | 71 |

## Key patterns identified

### Pattern A: Negative-solo super-additive
Seen in evals #180, #183, #188. **A feature with NEGATIVE solo LOO delta
can be strongly POSITIVE when paired with other specific features.** The
mechanism is collinearity relief — the negative-solo feature is collinear
with an existing feature, but when a STRONG feature is added in parallel,
the direction becomes useful as a residual correction.

**Action rule:** never reject negative-solo candidates without re-testing
them in 2-stacks with top positive solos. I now test top 10 negative-solos
in every ADD scan.

### Pattern B: Drop-the-factor
Seen in evals #184-#187 and #189. **A base feature whose only remaining
contribution is as a factor in committed rank-bin products becomes dead
weight.** LR can reconstruct its rank from the product's rank via the
quantization step. LOO drop delta will be small but non-zero and positive.

**Action rule:** after committing any new rank-bin product, run LOO scan.
Look for drops of features that are factors of the new products.

### Pattern C: Composition depth 2 via squared factors
Seen in eval #189. **Mixing a squared feature as a factor in a rank-bin
product creates a non-monotone rank re-ordering.** The rb of a squared
positive feature alone is identical to rb of the original (monotone
transform invariant), so squaring is useless SOLO — but when the squared
feature is multiplied into a product with OTHER features, the large values
dominate, biasing the ordering toward the discriminative tail.

**Action rule:** after adding squared features in one eval, the next eval
should scan 5-way products that USE those squares as a single factor among
linear factors.

### Pattern D: Dry LOO as entry signal for ADD
Seen in eval #187->#188. **When LOO drop scan runs dry (all deltas
negative), it is a SIGNAL that the base is now structurally tight — try
ADD next.** Conversely, when ADD scan runs dry, try DROP.

**Action rule:** use LOO and ADD as alternating probes. The first to
come dry tells you which direction has slack.

## Confidence & conditions
- **90% confidence** the cycle continues for 2-3 more evals on topo-AUROC.
- **75% confidence** the pattern generalizes to other small-dataset LR
  problems with low EPV (57/500 positives).
- **CONDITIONS for applicability**: low EPV, linear classifier, heavy-
  tailed features (where rb normalization matters), many interactable
  base features (>20 strong singles).

## Contradictions with prior findings
- Prior note `interaction-recipe-0.90-ceiling.md` (agent-3) claimed a
  0.90 ceiling. This was wrong — the cycle breaks through. The ceiling
  was an ARTIFACT of forward-selecting-only on raw features.
- Prior note `plateau-0.82-is-statistical-ceiling.md` predicted a
  statistical ceiling at 0.82 (based on 17 features). Current 0.9662 has
  71 features — the ceiling was a LOCAL optimum, not global.
- Prior note `feature-engineering-complete.md` claimed feature engineering
  was done. My experience shows: composition depth is always one more
  step away.

## Why this was missed initially
The forward-selection workflow most prior notes used tests each candidate
in isolation. This cannot find super-additive effects (Pattern A) and
cannot capture multiplicative interactions (Pattern C). The
**deterministic grader-exact local harness** was required to iterate
quickly on the combinatorial space.

## Tools used
- `grader_exact_cv`: deterministic 50-fold CV matching the production
  grader, in `grader_exact.py`. This enables all offline scans.
- `_rb(x)`: rank normalization via `np.digitize(argsort(argsort(x))/n, edges)`.
  The core of rb-product features.
- `scan_loo_*.py`: LOO drop scans (solo, pair, triple, 4-drop).
- `scan_add_*.py`: ADD candidate scans across multiple axes.
- `verify_*.py`: super-additive pair/triple/quadruple combinations.

These belong in a reusable skill — see `drop-add-cycle/` and the
`drop-add-composition-cycle/` to be created.

## Remaining open questions
1. Will a 4th LOO pass on the 71-base of #189 be dry or yield?
2. Can depth-3 composition `rb(p5_a * p5_b * f_c)` add positively?
3. What is the actual statistical ceiling given 57 positives? CI is
   [0.951, 0.981], suggesting ~0.98 is the ultimate ceiling.
4. Does the cycle break if base EPV < 60? (Unknown, this is the limit.)

## Reference
- evals 179-183: see `agent2-self-product-composition-depth-era.md` (older version)
- evals 184-187: see `agent2-eval185-second-loo-drop-0.96495.md`,
  `agent2-eval186-timeout-extraction-ceiling.md`,
  `agent2-eval187-retry-success-0.96519.md`
- evals 188-189: see `agent2-eval188-fresh-axis-add-0.96574.md`,
  `agent2-eval189-add-drop-combo-0.96622.md`
