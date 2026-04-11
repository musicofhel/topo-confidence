---
creator: agent-2
created: 2026-04-10
---
# Plateau Synthesis: Agent-2 at 0.8195 (rank 2), global ceiling ~0.8202

## Summary
After ~20 evals since breaking plateau at 0.8192→0.8195 with product interaction
(commit 22660faa), all subsequent experiments regressed. Multiple fundamentally
different directions ALL failed on this peak:

- **Layer_states PH** (uni 0.554 → score 0.8118, -0.008)
- **Layer velocity ratio** (uni 0.523 → score 0.8098, -0.010)  
- **Trajectory directional coherence** (uni 0.509 → score 0.8150, -0.005)
- **SUBSAMPLE=150** (univariates *improved* but multivariate dropped to 0.8045, -0.015)
- **sqrt transform on product** (score 0.8178, -0.002)
- **Stacked product interactions** (all regress)

## Key pattern
New features with univariate AUROC near 0.5 (random) always regress the
ensemble, even when mathematically independent from existing features.
Feature space has saturated: any noisy feature shifts the LR fit noise-ward.

## Key insight
SUBSAMPLE=150 IMPROVED univariate AUROCs on all topology features but
REGRESSED multivariate. This means the features are highly correlated and
"better" univariates make the correlation structure worse, not better.

**Implication:** The remaining signal is NOT in getting better individual
topology features — they're saturated. The remaining signal would need to
come from:
1. A new feature family that's independent of existing ones (all 3 tried failed)
2. Feature interactions (product works once, but doesn't stack)
3. Noise — the CI is [0.76, 0.87], width 0.11 — gap to leader 0.0007 = likely noise

## Confidence
**High** (>80%) that the 0.82 plateau is near the statistical ceiling for
this 500-sample, 11.4% positive-class dataset with 50-fold CV and LR-balanced.
The ±0.005 gap between top scores is within expected cross-validation noise.

## Untested direction
Multi-scale H1 (loops) at tight threshold — all multi-scale work has been H0.
H1 topology is orthogonal to H0 in the simplicial sense. Worth one attempt.
