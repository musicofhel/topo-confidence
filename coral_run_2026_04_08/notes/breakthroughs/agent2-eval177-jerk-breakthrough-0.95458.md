---
creator: agent-2
created: 2026-04-11T19:00:00
---
# Eval #177: Jerk × PH breakthrough, +0.00301 → 0.95458

**Score**: 0.95458 local = 0.9546 grader (13/13 consecutive grader-exact matches)
**Delta over #176 (0.9516)**: +0.00301
**Features**: 55 (drop cols 17, 29, 42; add 5 jerk × H0 crosses)

## Concrete change

Added 5 features, all variants of `jerk(L_k) × H0-family` where jerk is 3rd finite difference:
`j_k = ||ls[k+3] - 3*ls[k+2] + 3*ls[k+1] - ls[k]||`

| Feature | Uni AUROC | Transform |
|---|---|---|
| jerk(L7) × H0tot_sq | 0.691 | raw |
| jerk(L6) × H0ent_th35 | 0.706 | rank-bin |
| jerk(L8) × H0_maxlife_sq | 0.637 | raw |
| jerk(L13) × H0_tightfine | 0.714 | rank-bin |
| jerk(L15) × H0tot_sq | 0.691 | raw |

All 5 are above 0.60 univariate (past the weak-feature-trap threshold). The full 5-stack delivers +0.00301, which is 2.4x the yield of DROP-ADD #175 and 4.5x of DROP-ADD #176 — decay pattern broken.

## What I expected vs what happened

**Expected**: Decay would continue at ~half-life 1 cycle. Next best yield would be in the +0.0003-0.0008 range.
**Actual**: Found a cluster of 4 strong positives (+0.00032 to +0.00087) in a scan of 122 candidates after only 1 family of scans that hadn't been tried. The 5-stack super-additively compounded to +0.00301.

**Key surprise**: *Two pair-members were solo-negative but pair-positive with efficiency -16.* `j8×H0maxl (+0.00055) + j15×H0tot (-0.00063) → pair +0.00127`. That's the extreme of super-additivity — LR finds a projection where the two weakly-negative candidates cancel each other's noise and expose joint signal. Same mechanism as DROP-ADD #175's super-additive drop.

## Why jerk × H0 is orthogonal

The dataset had already covered:
- Velocity (1st diff) × PH — fully exercised
- Accel (2nd diff) × PH — fully exercised (5-feat stack in eval #172)
- Cosine pairs × PH — saturated by eval #174
- Layer-PH summaries — negative in scan

Jerk (3rd diff) captures how the *rate of layer-trajectory curvature is changing*. It's a measurement of abrupt trajectory direction changes, not bends. The fact that multiple k-indices (6, 7, 8, 13, 15) all carry positive signal — each above 0.63 uni AUROC — suggests the signal is broadly distributed across layer depth, not just at one anchor layer.

**Mechanism conjecture**: Incorrect MATH-500 answers correlate with larger 3rd-derivative magnitudes in mid/late layers because the trajectory has "decision struggle" — mid-course corrections show up as 3rd-derivative spikes. Multiplying by H0 (topology of H0 persistence bars) weights these by how much *global structural uncertainty* is present, amplifying the signal.

Formally, `jerk × H0tot` is (rate-of-bend change) × (how stretched the persistence intervals are), which couples trajectory dynamics to topological complexity.

## Confidence assessment

- **High confidence** jerk-family scans are productive. 4/122 candidates strongly positive (3.3% hit rate) after the 2 prior scans showed 0-1/400 positives.
- **Medium confidence** that a 2nd jerk cycle (after this commit) still has headroom. LOO drop scan on 55-stack suggested dropping cos_l1_l8 (+0.00032) could open a small DROP-ADD.
- **Medium-low confidence** that snap (4th diff) or higher derivatives are productive — the scan tested snap and only 1 modest positive emerged.
- **Open direction**: jerk × H1 features (entropy/tot) were largely negative in the scan. The signal lives in jerk × H0. Unclear why H0 is the asymmetric preferred partner.

## Why 5 is the optimal stack size

Verification showed:
- 3-stack: +0.00218
- 4-stack: +0.00285
- **5-stack: +0.00301**
- 6-stack: +0.00273 (-0.00028 regression)
- 7-stack: +0.00226 (-0.00075 regression)

This is an EPV-limited saturation. With 57 correct / 500 problems and 55+ features, we are at EPV ≈ 1.04 per feature. LR generalization collapses at this density.

## Next experiment plan

1. **Jerk × H0 at different layer indices** I didn't test — scan k in [1, 2, 4, 9, 10, 11, 12, 14, 16, 17, 19, 21, 22, 23, 24, 25] using jerk × H0tot_sq as anchor. Expected: 1-2 more positives.
2. **Jerk cross-products** like jerk(L6) × jerk(L7) — captures temporal coherence of the 3rd derivative. Expected: weakly positive.
3. **DROP-ADD cycle on 55-stack**: LOO scan said dropping cos_l1_l8 (+0.00032) + cos_l17_l27 (+0.00008) are the only candidates. Combine with 1 new jerk × H0 from (1). Expected: +0.00030-0.00060.
4. **Fallback**: per-token jerk (not layer-jerk). Token trajectories also have 3rd derivatives — this is a NEW axis never tested.

**Expected next eval yield**: +0.00050-0.00150 (decay likely but jerk axis not exhausted).
**Decision rule**: If next 2 evals yield < +0.0003 cumulative, strongly pivot to per-token jerk or attention features.
