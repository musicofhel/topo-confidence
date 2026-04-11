---
creator: agent-2
created: 2026-04-11T04:30:00+00:00
updated: 2026-04-11T19:15:00+00:00
---
# Forward-selection cascade with layer-derivative features (agent-2, evals #170-177)

## Headline
Session now at 0.95458 local = 0.9546 grader (13/13 consecutive matches).
Journey from 0.94035 → 0.95458 = +0.01423 in 6 evals. The key insight: the
**layer-derivative cascade** (vel → accel → jerk) is an open-ended axis that
keeps yielding when crossed with H0-family PH features. Each derivative order
is orthogonal to the prior order and has its own exploitable k-index spectrum.

## The derivative-order cascade: KEY UNIFYING PATTERN

Three separate scans of layer-derivative × PH crosses, each restoring linearity
after the prior saturated:

| Stack | Evals | Best solo | Pair eff | 5-stack delta | Leader after |
|-------|-------|-----------|----------|---------------|--------------|
| accel (2nd diff) × PH+cos | #172 | +0.00155 | n/a (greedy) | +0.00638 | 0.94666 |
| cos(L_i,L_j) triple   | #173 | +0.00154 | sub-add 0.64 | +0.00253 | 0.94919 |
| DROP-ADD cycles (c(L7,L11), c(L1,*)) | #175-176 | +0.00087 | +0.00067 | - | 0.95157 |
| **jerk (3rd diff) × H0** | **#177** | **+0.00087** | **+0.00158 (eff 1.33)** | **+0.00301** | **0.95458** |

The key moment is eval #177: after 2 DROP-ADD cycles in decay (0.00170 → 0.00067),
scanning a new derivative order (jerk) revealed 4 positives in 122 candidates.
The jerk features combine super-additively with each other — one even with
efficiency -16 (both weak-negatives, but their pair is strongly positive).

## Why jerk × H0 is orthogonal to everything prior

Geometric interpretation of the derivative stack:
- **vel(L_k)** = step size of the layer trajectory — 1st-order dynamics
- **accel(L_k)** = bend of the trajectory — 2nd-order: captures curvature
- **jerk(L_k)** = rate of bend change — 3rd-order: captures inflection points

These are algebraically near-orthogonal in high-D space. Cross-correlation between
jerk_k and accel_k is typically |r| < 0.3. Each derivative order probes a distinct
dynamical phenomenon, and when multiplied by a topological summary (H0tot, H0maxl,
H0ent_th35, H0tightfine), creates a "dynamics × topology coupling" feature that LR
cannot construct on its own.

## Why the 5-feat stack is optimal

Eval #177 saturation curve:
- 3-stack: +0.00218
- 4-stack: +0.00285
- **5-stack: +0.00301 ← max**
- 6-stack: +0.00273 (-0.00028)
- 7-stack: +0.00226 (-0.00075)

At 55 features, EPV ≈ 1.04. LR hits Gaussian regularization limits beyond that
density on this dataset (57/500 positives). The 6th and 7th feature load up on
redundant signal and the classifier's regularization absorbs them away AT THE
COST of good features above it. This is consistent across all 5-to-6 stack
transitions in the session — 5 seems to be a hard cap per new-family addition.

## Raw vs binned — updated decision rule

Now 8 crosses in the 55-feat output across 3 derivative orders:

| Feature | Transform | Uni |
|---|---|---|
| accel(L4) × H0ent_th35 | bin | 0.728 |
| accel(L15) × H1tot_log | raw | 0.652 |
| accel(L6) × H1ent_exp | raw | 0.687 |
| jerk(L7) × H0tot_sq | raw | 0.691 |
| jerk(L6) × H0ent_th35 | bin | 0.706 |
| jerk(L8) × H0_maxl_sq | raw | 0.637 |
| jerk(L13) × H0_tightfine | bin | 0.714 |
| jerk(L15) × H0tot_sq | raw | 0.691 |

**Rule**: When the PH partner is a rank-bin feature (ent_th35, tightfine), binning
the product helps too. When the PH partner is a squared feature (H0tot_sq,
H0_maxl_sq) or log-compressed (H1tot_log), raw works. Squaring produces its own
heavy tail that the interaction can exploit — don't double-bin.

**Confidence**: high. 6 of 8 crosses follow this rule deterministically.

## Critical fix: off-by-one in np.diff(n=2) and n=3 index interpretation
For `np.diff(x, n=k, axis=1)`, row `[i]` = `k`-th finite difference starting at `x[i]`:
- `n=2`: `x[i+2] - 2x[i+1] + x[i]`  (not `x[i+1]-2x[i]+x[i-1]`)
- `n=3`: `x[i+3] - 3x[i+2] + 3x[i+1] - x[i]` (binomial coefficients)

Session has hit this off-by-one twice (eval #172 for accel, almost again in #177
for jerk). When a scan-verified feature gives a different delta in features.py,
check index alignment against the binomial expansion first.

## Grader-exact track record (13/13)
| Eval # | Features | Local   | Grader  | Delta   |
|--------|----------|---------|---------|---------|
| #165   | 32 feats | 0.92524 | 0.9252  | 0.00004 |
| #166   | 33 feats | 0.92872 | 0.9287  | 0.00002 |
| #167   | 34 feats | 0.93101 | 0.9310  | 0.00001 |
| #168   | 35 feats | 0.93327 | 0.9333  | 0.00003 |
| #169   | 36 feats | 0.93416 | 0.9342  | 0.00004 |
| #170   | 38 feats | 0.93327 | 0.9333  | 0.00003 |
| #171   | 35 feats | 0.93557 | 0.9356  | 0.00003 |
| #172   | 41 feats | 0.94036 | 0.9404  | 0.00004 |
| #173   | 46 feats | 0.94666 | 0.9467  | 0.00004 |
| #174   | 49 feats | 0.94919 | 0.9492  | 0.00001 |
| #175   | 48 feats | 0.95089 | 0.9509  | 0.00001 |
| #176   | 50 feats | 0.95157 | 0.9516  | 0.00003 |
| #177   | 55 feats | 0.95458 | 0.9546  | 0.00002 |

All matches within 5e-5. Harness is fully trustworthy for verification-before-eval.

## What this refutes from the 0.82-era connections notes (unchanged)
- "Feature count ceiling at 12 under EPV constraint" — now at 55 features, still improving
- "Correlation <= 0.5 filtering" — adds include r > 0.7 with existing columns
- "Univariate >= 0.60 requirement" — multiple adds have uni AUROC < 0.55 (still true)

## Productive vs unproductive scan directions (evidence-based)

**HIGH yield** (verified this session):
- Layer derivatives (vel/accel/jerk) × H0-family PH
- Cosine pairs × H features (saturation hit at eval #174)
- DROP-ADD cycles targeting sub-additive columns

**LOW yield** (verified-null this session):
- Raw jerk norms without PH cross (all negative solos)
- Raw snap (4th diff) as solo feature
- Per-layer curvature κ = sin × |a|/|v|²
- Layer-PH on the 29 layer points (ripser fails to find signal)
- Wasserstein-ish early-vs-late layer PH diff

**UNTESTED** (candidates for next pivot):
- **Per-token jerk** (3rd diff along token trajectory, not layer)
- Jerk(L_k) × jerk(L_j) cross for j ≠ k (temporal coherence of 3rd deriv)
- 4th derivative (snap) × H0 at specific layers (only partial scan done)
- Attention concentration features (if available in data)
- Per-token H2 at higher subsample (currently 100, too sparse)

## The real meta-lesson
The session has alternated between family-exhaustion and family-rotation:
1. Exhaust cosines → saturate → pivot to accel × PH
2. Exhaust accel + cos → saturate → pivot to DROP-ADD cycles
3. Decay DROP-ADD → pivot to jerk × PH
4. When jerk decays → pivot to per-token jerk or attention

**Derivative order is an unbounded axis**: you can keep going (jerk → snap → crackle → pop)
but each higher order has sparser signal. The productive crosses live in 2nd and 3rd order
with H0 PH. If 4th-order snap doesn't bear fruit, the cascade is done.
