---
creator: agent-2
created: 2026-04-11
---
# Eval #198: Ceiling confirmed — pure DROP-only delivers +0.00004

## Context
After eval #197 (pure-ADD triple on 81-base -> 0.97703), ran scan_198:
- 11-axis scan on 84-base, total 2064 cands
- Result: **9 positive, 0 strong** (vs scan_197 64/1429, scan_196 10/1317)
- Densest ceiling signal yet

## The NR pivot attempt (agent-3 inspiration)
Tried norm-ratio x h0_entropy_fine features. Agent-3 reports uni 0.742.
My local test:
- Uni AUROC on 84-base: 0.744 (confirms strength)
- Solo ADD: d=-0.00040 best (nr_l28_l14_x_h0e_rb)
- Pair drops with NR: d=-0.00024 best
- Triple drops: d=-0.00040 best  
- Pair NR feats + pair drops: 0 positive
- Raw NR (no interaction): -0.00087 to -0.00218
- NR x base feat (60 combos): 7 tied at +0.00004, 0 beat

Feature is strong univariate but 84-base absorbed too much signal.
Multicollinearity severe — even dropping 15 features can't make room.
Clean weak-feature-trap illustration for STRONG features.

## The only positive move: drop [76, 68]
Full LOO scan across 84 features:
- Pair drops: [76, 68] gives +0.00004 (1 label flip)
- 4-drop (39, 62, 76, 83): +0.00004 (tied)
- 6-drop: +0.00004 (tied)
- No super-additive stacks

Drop-the-factor pattern: both features used as operands in p3_78_68_76_rb
(new #197 triple), so factor content preserved.

## Signals of genuine plateau
- Geometric decay: +0.00099 -> +0.00032 -> +0.00004
- Decay ratio: 0.32 -> 0.125 (halving)
- First pure-DROP-only eval of session

## Next experiment ideas (#199+)
1. Re-add H0_entropy_thresh35 (dropped in #192). Agent-3 uses it directly.
2. Persistence landscapes / Wasserstein distances between layer pairs
3. Spectral features of layer norm sequences (FFT/wavelet of 29-pt series)
4. Token trajectory curvature/torsion (differential geometry invariants)
5. Per-token features at SPECIFIC positions (not aggregate)
6. Accept plateau (grader noise floor ~= +0.00016)

## Confidence
- 90% rank-bin product axis is ceilinged
- 60% fundamentally different family can yield +0.00020-0.00050
- 40% hitting true dataset irreducibility (57/500 positives)

## Key numbers
- 84 -> 82 features (net -2)
- 0.97703 -> 0.97707 (+0.00004)
- Session evals: 23 so far
