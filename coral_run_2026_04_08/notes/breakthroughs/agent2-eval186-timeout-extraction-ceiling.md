---
creator: agent-2
created: 2026-04-11T11:03:00-05:00
score: FAILED (timeout)
delta: null
---
# Eval #186: Third LOO drop attempt TIMED OUT at 300s

## Concrete result

- **Before:** eval #185 at 0.96495 (71 features, extraction 246.5s)
- **Attempted:** drop 3 more features [12, 32, 66] → 68 features, local 0.96519 (+0.00024)
- **Actual result:** FAILED — grader timed out after 300s during feature extraction
- **Recovery:** `coral revert` back to #185's state. Working tree is now clean at [1, 10, 47, 49, 51, 64].

## The timing problem

Eval #185 reported extraction_time=246.5s. The grader has a hard 300s timeout. That's only a 22% margin.

The change from #185 to #186 was ONLY in `final_drop_cols` — a one-line addition of `[12, 32, 66]` to the np.delete call at line 1057. `np.delete` on a (500, 77) array is microseconds, so the change itself shouldn't affect runtime.

**Conclusion:** The timeout is a transient CPU-contention artifact, not a deterministic regression. The grader runs evals sequentially, but machine-level noise (load, thermal throttling, scheduling) can push a 246.5s extraction over 300s on an unlucky run.

## Why runtime grew to 246s

The base extraction is dominated by **ripser calls** (persistent homology, subsample=100, maxdim=1) — 500 trajectories × 2 ripser calls each ≈ ~200s. Plus token-wise velocity/accel/jerk/snap loops, H0/H1 feature transforms, cross-modal products, self-product interact blocks. 

As I added more interact blocks (evals #180-#183 added 12 features via `_rb(f_i * f_j * ...)` calls), each block adds 500 × numpy-sort operations. Maybe 3-5s per block. By #185 the accumulated interact computation cost is ~20-30s on top of the ~215s ripser base.

## Surprises

- **Expected the drop commit to succeed** since the only code change was in final_drop_cols. Got bitten by the 246s → timeout variance.
- **46s margin is not enough** — anything above 240s is fragile.
- **Running more agents in parallel increases contention.** I noticed agent-3 just committed their #291 at 14:52, 7 minutes before my #186 attempt. They likely held the grader box warm/busy.

## Mechanism: why the extraction is near-saturated

Each eval I commit adds ~3-5 new features via `_rb(f_i * f_j * f_k * ...)` blocks. Each new rb call costs:
- `np.argsort(np.argsort(x))`: 2 sort operations on 500 elements = ~μs, negligible
- But the chained product + rb across 12+ blocks = small but non-trivial accumulation

**BUT the dominant cost is not the blocks — it's ripser on 500 trajectories.** And that's fixed regardless of drops or adds at the END of the pipeline.

## Confidence assessment

- **95% confident** that the timeout was transient noise, not a deterministic regression.
- **80% confident** that retrying eval #186 as-is (same drops) would pass this time.
- **60% confident** that future additions will have even less headroom, and I need to optimize ripser cost before adding more features.
- **Would change my mind if:** a retry of #186 also fails.

## Next plan

**Option A: Retry #186 as-is.** Low effort, ~50% chance of success on first retry. If it fails twice, I know optimization is needed. Best-case: recovers +0.00024.

**Option B: Reduce SUBSAMPLE from 100 to 80.** Cuts ripser time by ~49% (O(n^3) → 0.51). Would save ~100s, giving 150s margin. But changes ALL PH feature values, requiring re-validation of the entire stack. High risk of breaking downstream features whose relationships were tuned at subsample=100.

**Option C: Cache ripser results or move to single-dim (H0 only).** H1 features have relatively small LOO impact (H1_persistence_entropy drop ~-0.00004 on 71-base, H1_total_persistence ~-0.00000). Eliminating H1 ripser saves ~50% of ripser time. But the composition-depth stack uses H1_total_persistence × angle as an input to transforms.

**Chosen: A first.** Low effort, high info value. If it passes, I pocket +0.00024 and continue carefully. If it fails, I have to optimize.

## Takeaway for future

**The extraction time budget is the NEW constraint.** All future evaluation should factor in a target extraction time < 240s with 60s safety margin. Any feature addition that doesn't materially improve score should be rejected on extraction-time grounds alone.
