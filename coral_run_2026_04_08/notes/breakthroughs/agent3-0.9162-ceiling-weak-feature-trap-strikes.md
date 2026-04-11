---
creator: agent-3
created: 2026-04-11T04:55:00+00:00
---
# Eval #272 regression: weak-feature-trap fires at 0.9162 ceiling

## Result
- Leader: 0.9162 (eval #271, SWAP cos_l9l24_x_norm -> cos_l4l10_x_H0tot)
- Eval #272: ADD cos_l23_l27 (27th, raw layer cosine)
- **Regressed: 0.9153 (-0.0009)**
- Reverted to 0.9162 leader (d4246122)

## Local CV said "go"
- 773-candidate sweep. Winner: cos_l23_l27
- Pass1 (5 seeds): +0.00482
- Pass2 (20 seeds): +0.00415, t=+35.86
- Pass3 (50 seeds): +0.00236, t=+23.28, pos=1.00
- DROP+ADD combo tested WORSE than ADD-only (-0.00011)
- So committed ADD-only

## Grader said "no"
- cos_l23_l27 univariate AUROC: **0.514** (from eval feedback)
- That's the `weak-feature-trap` skill firing EXACTLY as predicted:
  > "At the signal ceiling, any Nth feature with univariate AUROC < 0.60
  > will regress the multivariate score, regardless of correlation profile."
- This is now the 4th regression in a row from ADD-only attempts at 0.91+.

## Why the local CV missed it
- Local CV evaluates multivariate fit, which can find tiny residual signal
  from weak-uni features IF the model has low variance noise on that particular
  seed.
- Grader's CV uses different random state than our seeds — variance cancels
  differently and weak signals are swamped by estimation noise.
- At 0.91+, the grader-local gap grows because ALL signal is already extracted.

## Key observation: raw layer cosines are TRAP features at this regime
- All previous successful layer-cosine ADDs were CROSS-MODAL products:
  - cos_l9l28_x_angle (24th): uni 0.524 but succeeded when base was ~0.82
  - cos_l0l20_x_pc2early (25th): uni 0.521, succeeded at 0.89 base
  - cos_l9l24_x_norm (26th): uni ? then absorbed
  - cos_l8l24_x_H1ent (27th): uni 0.649 (strong!)
  - cos_l4l10_x_H0tot (28th): uni 0.594 (borderline)
- At 0.91+, even cross-modal products with uni ~0.59 are risky.
- Raw layer cosines (uni ~0.51) are REJECTED by the grader.

## Updated strategy
STOP looking for new layer cosines, RAW or cross-modal, unless their uni AUROC
is >= 0.65. The 0.91+ regime demands STRONG univariate signal.

New directions to explore:
1. **NEW topological features with uni >= 0.65**: variations on H0/H1 metrics
   we haven't computed yet (e.g., H0 with different distance metrics,
   persistence landscapes, Betti curves).
2. **DROP-only cycles**: find features with positive LOO drop delta and drop
   them pure (no ADD), shrinking to 25 or even 24 features to reduce
   overfitting. The last successful DROP (eval #268) dropped 2 features.
3. **Variance-based REFRAME**: replace an existing weakest feature (|w|<0.3,
   uni<0.55) with a transformed version that preserves the signal.
4. **Multi-scale H0**: our H0 features all use the same subsample/threshold.
   Try H0_total at subsample=50 or H0_entropy at thresh=25%.

## Confidence
- 85% confident that any new ADD candidate with uni < 0.60 will regress
  (strong prior from weak-feature-trap skill + now a 4th confirmation).
- 60% confident that DROP-only cycles will find net gains (LOO showed only
  1 weakly positive candidate).
- 40% confident that new topological features (different metrics/scales) will
  beat the 0.9162 ceiling.

## Next experiment
Compute topological features using cosine distance metric (instead of
Euclidean) for the trajectory PH. Check univariate AUROC FIRST. Only commit
if uni >= 0.65.
