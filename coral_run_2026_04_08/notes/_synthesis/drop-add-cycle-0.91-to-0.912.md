---
creator: agent-3
created: 2026-04-11
supersedes: N/A (extends interaction-recipe-0.90-ceiling.md)
---
# DROP->ADD iteration cycle: 0.9061 -> 0.9122 in 3 evals

## Headline
After the add-only interaction recipe exhausted at 0.9061 with two consecutive
regressions (evals 266-267), a DROP->ADD iteration cycle broke through to
0.9122 in 3 evals. The insight: once LR collapses a feature's coefficient
toward zero, the column is adding NOISE, not signal. Dropping it frees
regression capacity for the next ADD.

## Evidence chain (agent-3, evals 266-270)
| Eval | Score | Action | Local delta | Grader delta | Efficiency |
|------|-------|--------|-------------|--------------|------------|
| 265 | 0.9061 | ADD cos_l0l20_x_pc2early | +0.00321 t=25.45 | +0.0010 | 31% |
| 266 | 0.9049 | ADD normratio_l14l28_x_toklayer | +0.00141 t=8.37 | -0.0012 | FAIL |
| 267 | 0.9046 | ADD tok_count | +0.00084 t=11.98 | -0.0015 | FAIL |
| 268 | 0.9068 | **DROP feat14+feat17** | +0.00208 t=34.91 | +0.00072 | 35% |
| 269 | 0.9116 | **ADD cos_l9l24_x_norm** | +0.00206 t=22.33 | +0.00479 | 232% |
| 270 | 0.9122 | **ADD cos_l8l24_x_H1ent** | +0.00233 t=24.71 | +0.00059 | 25% |

Total gain 265→270: +0.0061 (3 regressions, then 3 improvements).

## The mechanism
**On the 26-feat saturated base:** LR had collapsed feat14 (|w|=0.042) and
feat17 (|w|=0.199) to near-zero coefs. Their signal was fully absorbed by
cross-product features that used them as factors (feat19, feat21, feat24,
feat25). The standalone columns contributed noise to coefficient estimation.

**Dropping them** lowered the feature count from 26 to 24, improving the
effective EPV from 2.19 to 2.375. More importantly, it reduced the LR
covariance matrix conditioning — existing features' coefficients are now
tighter, the model "sees" signal better.

**The first ADD after a DROP over-delivers** (232% efficiency on #269)
because the LR finally has capacity to incorporate the new feature.
**The second ADD under-delivers** (25% on #270) as diminishing returns
set in, but is still positive.

## Recipe (updated)
0. Apply interaction-recipe-0.90-ceiling.md rules first (rank+10bin products,
   cross-modal dominance, 2-3 derivatives per source).
1. **When an ADD regresses twice in a row**, switch to DROP mode.
2. **Run leave-one-out (LOO) CV** on all current features. For each feat j:
   - Drop column j, re-run 20-30 seed CV
   - Paired delta = auroc(drop_j) - auroc(base)
   - Look for features with delta > +0.0005 AND LR |w| < 0.3
3. **DROP candidates** are features where the LOO-CV delta is *positive* (their
   removal helps). These have been absorbed by interactions or have redundant
   signal.
4. **Commit a single-DROP eval** or a multi-DROP if t > 15 (strong signal).
   DROPs are more robust than ADDs because they don't add new variance.
5. **After the DROP**, run a broad ADD sweep on the cleaner base. The first
   ADD tends to over-deliver. The second ADD underdelivers.
6. **Alternate DROP and ADD** when both recipes lose efficiency.

## Strategic observations
- **DROP deltas have tighter SE than ADD deltas** (see eval #268: se=0.00006
  for DROP vs typical ADD se=0.00015). Removing noise is more reproducible
  than adding signal.
- **Efficiency (grader/local) varies 25-232%** post-DROP. Mean ~100%.
  Before DROP, efficiency was 30-60%.
- **First ADD post-DROP is the most valuable move.** The second ADD is
  probably at a reduced ceiling already.
- **Topological×layer cross-products** (eval #270: H1_persistence_entropy ×
  cos_l8_l24, uni AUROC 0.651) may carry fresher signal than the norm×layer
  products that dominated the earlier recipe. The H1_entropy factor is
  topologically orthogonal to cosine alignment.

## Open questions
- How many DROP->ADD cycles remain before absolute ceiling? (guess: 2-3)
- Can we DROP MORE than 2 features at once? feat14+feat17+feat23 was tested
  and gave +0.00217 vs +0.00208 for feat14+feat17 alone — marginal.
- Does the cycle work for other agents' feature bases? Worth sharing.

## Confidence
85% that one more DROP->ADD cycle reaches 0.914+.
60% that the absolute ceiling on topological features is 0.92.
