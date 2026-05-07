# Result Brief — 2026-05-06 — Overnight Sweep (10 CPU Experiments)

## Summary

Ran 10 CPU-only experiments from the 2026-05-06 paper triage overnight sweep.
All use cached L19 prefill activations (500×1536, Qwen-2.5-1.5B, MATH-500, 1024-tok).
Baseline: DoM OOF AUROC = 0.7731.

## Results

### FE01172B — Gram Eigenspectrum (Effective Rank)
- **Participation ratio: 19.86** — matches cov-spectrum probe dimensionality (20)
- Top-20 eigenvalues explain 68.15% of total variance
- MP effective rank = 499 (essentially full rank in 500-sample regime)
- Per-class PR similar: correct and incorrect groups have comparable rank
- **Interpretation**: The ~20 effective dimensions explain why the 20-feature
  cov-spectrum probe (AUROC 0.7928) nearly saturates the full 1536-d ceiling (0.7847)

### FE903 — CCA Prefill vs Final-Token
- **mCCA = 0.98 at k=200** — prefill and final-token activations share the same
  linear subspace
- But **cos(DoM_prefill, DoM_final) = -0.062** — the discriminative directions
  within that shared subspace are nearly orthogonal
- Per-class mCCA: correct=0.970, incorrect=0.974 (both high)
- **Interpretation**: F-3 (prefill/final DoM orthogonality, cos=0.046) is NOT a
  between-subspace artifact. The subspaces overlap almost completely; the
  discriminative directions genuinely rotate within the shared subspace.

### FE899 — Principal Subspace Angles (Correct vs Incorrect)
- k=5: mean angle 22.5°, max 44.7°, Grassmann=1.09
- k=10: mean 28.2°, max 62.9°, Grassmann=1.91
- k=20: mean 34.1°, max 88.9°, Grassmann=3.17
- k=50: mean 39.0°, max 87.1°, Grassmann=5.53
- **Zero angles below 10° at k≤20** — classes occupy measurably different subspaces
- **Interpretation**: Correct and incorrect activations diverge geometrically, not
  just in mean (DoM). The subspace divergence grows with k, suggesting the
  discriminative structure has many dimensions, not just the DoM direction.

### FE930 — RoPE Plane Projection via q_proj Pullback
- Per-head DoM variance fractions: [0.07, 0.10, 0.05, **0.16**, 0.08, 0.07, 0.05, 0.07, 0.10, 0.07, 0.08, 0.10]
- **Gini coefficient = 0.17** (nearly uniform)
- Top-1 head fraction = 16.4%, top-3 = 36.5%
- **Interpretation**: DoM has no special coupling to any attention head's RoPE
  rotation planes. The correctness signal is not position-encoded through a
  specific head — it lives in the isotropic part of residual space.

### FE901 — SETOL ECS Projection (ROI 9, Highest Priority)
- **o_proj**: PL alpha=1.14, R²=0.921, ECS rank=768, DoM ECS variance=50%, PC1 ECS=49%
- **down_proj**: PL alpha=1.72, R²=0.926, ECS rank=768, DoM ECS variance=53%, PC1 ECS=53%
- cos(DoM, PC1) = 0.9216 (consistent with F-2)
- DoM top-k variance in o_proj eigenvectors: k=1: 5.4%, k=5: 16.9%, k=10: 22.6%, k=50: 48.3%
- **Interpretation**: DoM partially aligns with the weight effective correlation
  structure (~50% variance), but is not concentrated in the top few eigenvectors.
  PL exponents (1.1-1.7) are on the low end of the 2-6 range typical for
  well-trained models, suggesting heavy-tailed weight spectra.

### FE26841a — Logit-Lens Entropy
- **Entropy OOF AUROC (LogReg) = 0.633** — informative but well below DoM 0.773
- **Correct samples have HIGHER entropy** (8.76 ± 0.90 vs 8.33 ± 0.94)
- Raw negated entropy AUROC = 0.363 (confirms positive entropy-correctness association)
- Spearman(entropy, DoM) = 0.339 (p < 1e-14) — moderate correlation, non-redundant
- **Interpretation**: At L19, correct solutions have MORE diffuse logit distributions
  (exploring more possibilities) while incorrect ones are prematurely committed.
  This is an intermediate-layer phenomenon — final-layer entropy likely reverses
  (correct answers converge). Entropy is a complementary but weaker signal to DoM.

### FE889 — Procrustes Rotation-Axis Projection
- **o_proj**: det(R)≈1.0, largest rotation=76° (!), but DoM has only 0.3% variance
  in top-1 plane, 1.6% in top-10 planes
- o_proj AUROC from top-k planes: k=1: 0.667, k=5: 0.742, k=10: 0.752, k=50: 0.745
- **down_proj**: tiny rotations (max 0.79°), DoM 2.9% in top-10 planes
- down_proj AUROC from top-k planes: k=1: 0.720, k=10: 0.757, k=50: 0.721
- **Interpretation**: DoM is NOT aligned with the largest RLHF/SFT rotation axes.
  The 76° o_proj rotation is large, but DoM lives in a different subspace. The
  correctness signal is not a direct imprint of the largest instruction-tuning
  changes. However, rotation-plane features do achieve 0.752 AUROC, suggesting
  some indirect relationship.

### FE909 — Per-Layer Alignment Profile
- **Peak at L19** (alignment=1.0 by construction, since DoM_L19 is the reference)
- **Half-max width = 7 layers (L15-L21)**, CI confirms significance
- Gradual ramp from L0 (0.0) through L12 (0.38) to L19 (1.0), then decay to L28 (0.15)
- L18: 0.795, L20: 0.752 — sharp but not ultra-narrow peak
- **Interpretation**: The correctness direction emerges gradually starting around
  L12 (~40% depth), peaks at L19 (~66% depth), and persists through L21 before
  decaying. The 7-layer half-max width is "medium" — consistent with a 5-8 layer
  computational window rather than a single-layer imprint.

### FE919 — Diffusion Maps Embedding
- **k=2: AUROC 0.757, k=5: 0.758, k=10: 0.788, k=20: 0.784**
- **k=10 AUROC = 0.788 > DoM 0.773** — nonlinear manifold structure encodes
  correctness beyond the linear DoM direction
- Spectral gap after k=2 is largest (0.051), then plateaus
- **Interpretation**: There IS correctness-relevant nonlinear structure in the
  activation manifold. 10 diffusion coordinates capture enough curvature to beat
  the linear DoM. This is the strongest result from the sweep — suggests that
  manifold-aware methods could improve on the linear cov-spectrum probe.

### FE925 — Hyvärinen Score Difference
- Gaussian (QDA) in-sample AUROC = 1.0 (overfit in 1536d with 500 samples)
- **Gaussian (QDA) OOF AUROC = 0.736** — below DoM 0.773
- Sliced score in-sample = 0.996 (also overfit)
- **Interpretation**: Full Gaussian QDA with Ledoit-Wolf shrinkage cannot beat the
  simple linear DoM in 1536 dimensions with only 500 samples. The cov-spectrum
  probe (0.793) succeeds by using feature selection (20 log-eigenvalues) rather
  than the full quadratic form. The lesson: feature engineering > model complexity
  in the low-n/high-d regime.

## Cross-Experiment Synthesis

1. **Effective dimensionality ≈ 20** (FE01172B PR=19.86), explaining why the
   20-feature cov-spectrum probe (0.793) nearly saturates the full-space ceiling.

2. **Nonlinear structure matters** (FE919 DM=0.788 > DoM=0.773), but not
   dramatically — the linear DoM captures most of the signal.

3. **Subspace structure is rich** (FE899 angles 34-39°, FE903 mCCA=0.98 with
   orthogonal DoMs) — correct and incorrect activations share the same ~20-dim
   subspace but occupy it differently in ways beyond the DoM direction.

4. **DoM is NOT an RLHF artifact** (FE889 DoM orthogonal to rotation axes) and
   NOT position-encoded (FE930 uniform across heads). It is an emergent
   statistical regularity in the activation geometry.

5. **L19 is the correct layer** (FE909 peak at L19, 7-layer half-max width),
   with the correctness direction ramping up from L12.

6. **Entropy adds complementary signal** (FE26841a AUROC=0.633, Spearman=0.34
   with DoM) — correct samples are LESS committed at L19.

## Data

All result JSONs in `pathway11_h100/results/fe{01172B,903,899,930,901,26841a,889,909,919,925}_*.json`
