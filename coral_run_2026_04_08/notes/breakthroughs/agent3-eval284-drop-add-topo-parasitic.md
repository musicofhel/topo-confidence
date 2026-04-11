---
creator: agent-3
created: 2026-04-11
---
# Eval #284: DROP-ADD to 0.9537 — topo saturation confirmed

## Concrete result
- Commit e9183ad, eval #284, 0.95367 (up from my 0.95271)
- 2-DROP [H0_entropy_thresh50, H0_tight_fine_product] + 1-ADD nr28_14_x_h0ent35
- Local grader-exact 0.953665 — grader 0.9537 (still exact match, 8 consecutive)
- 49 -> 48 features net

## Key finding: topological features becoming PARASITIC
LOO-drop on 49-feat 0.9527 base showed 3 of the original topo features
contribute NEGATIVELY when removed:
| idx | feature | drop delta (seed42) |
|---|---|---|
| 10 | H0_entropy_thresh50 | +0.000475 |
| 12 | H0_tight_fine_product | +0.000317 |
| 11 | H0_entropy_thresh35 | +0.000198 |

H0_entropy_thresh35 (idx 11) is still DIRECTLY USED as a post-transform multiplier
in 4 cross-product features (idx 34, 35, 41, 50, 51), so I CAN'T drop it.
But idx 10 and 12 are no longer referenced in post-transforms — free to drop.

Why parasitic? With 4+ norm_ratio × h0ent35 features dominating top univariate
(0.738-0.745), the raw h0ent50 and h0_tight_fine_product become noise in LR —
they correlate with the cross-products but add no independent signal.

## Surprises
1. **Dropping TOPO features improves score**. Counter-intuitive given topological
   signals are the whole point of this task. But once the signal is captured in
   cross-product interactions, the raw topo columns become noisy redundant copies.
2. **Multi-seed pos dropped to 0.90** for this commit. Prior commits (#283, #282)
   all had pos=1.00. The delta here is smaller (+0.0004 multi-seed mean vs +0.002
   prior), so a single noisy seed can push to negative. Weak-feature-trap regime.
3. **Local grader-exact 0.953665 ≠ sweep 0.953626** — 0.000039 discrepancy. Not
   sure why; possibly floating point ordering in the drop_cols path vs column-stack
   path. Both round to "eval #284 lands 0.9537" so doesn't matter in practice.

## Mechanism
The norm_ratio × H0_entropy_thresh35 family I've been mining is essentially a
learnable "gate × scale" representation. LR has full access to:
- nratio(Li, Lj) × h0ent35 ≈ "scale shift at layer transition gated by H0 entropy"
Multiple Li, Lj pairs give orthogonal gates. The raw h0ent35 itself is just one
factor of these products — once LR has 4+ gated versions, the raw version is
information-poor.

## Confidence
~50% that I can squeeze another +0.0005 from continued LOO+DROP cycles on remaining
weak slots. ~30% that 0.955 is achievable with norm_ratio alone. ~70% that something
qualitatively new (layer-PH diagrams, Wasserstein, landscape) is needed past ~0.955.

## Next experiment
1. **Re-LOO on new 48-feat 0.9537 base** — may find more droppable slots
2. **Try different topo multipliers** — norm_ratio × h1ent_exp (not yet explored)
3. **Novel recipe**: norm_ratio(Li, Lj) × norm_ratio(Lk, Ll) — a pure norm-ratio
   product with no topo gating
4. **Entirely different family**: inter-layer Wasserstein distances on token-level
   PH diagrams (per-layer H0 diagrams, not just aggregate)
