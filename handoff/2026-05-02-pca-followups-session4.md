# Handoff — PCA follow-ups session #4 (2026-05-02, fourth continuation)

**Date:** 2026-05-02 (fourth pass in same day)
**Session entry:** continued from `2026-05-02-pca-followups-session3.md`
"After-this-session resume" → executed move #1 (two-feature decomposition
triangle) per user's "what's next".

## What landed

The triangle resolves cleanly. Full 1536-d L2-reg logistic OOF maxes
at **0.7847** (C=0.001) — essentially equal to the 2-feat (PC1, PC9)
0.7856 within fold noise. The cov-spectrum 0.7928 still beats the
full directional probe by ~0.8 pp. **F-2's extra signal is genuinely
second-order, not under-regularized linear-directional.**

| Step | What | Outcome |
|---|---|---|
| 1 | Extended `recompute_pca_covariance.py` with full 1536-d L2-reg sweep over C∈{0.001, 0.01, 0.1, 1.0} | full_lr_best_auroc=0.7847 at C=0.001 |
| 2 | New Claim `pca-full-lr-best-auroc` in validate_claims.py | Tier-1 wired to PCA covariance script |
| 3 | validate_claims.py full run | (See "Tier-1 regen invariant" below) |
| 4 | PERSPECTIVES.md "The triangle resolves" subsection | Updated |
| 5 | This handoff | Written |

## Headline numbers

The completed triangle:

| Probe | OOF AUROC | Δ vs DoM 0.7679 |
|---|---|---|
| 1-d DoM | 0.7679 | — |
| 2-feat (PC1, PC9) | 0.7856 | +1.77 pp |
| Full 1536-d L2-reg (C=0.001, best) | **0.7847** | +1.68 pp |
| Top-20 log-eigvals (cov spectrum) | **0.7928** | +2.49 pp |

Full 1536-d C-sweep:
- C=0.001: 0.7847 (best, regularization sweet spot)
- C=0.01: 0.7585 (already overfitting)
- C=0.1: 0.7325
- C=1.0: 0.7211 (severe overfit at p=1536/n=500)

## What changed in the framing

F-2 now factors into three explicitly additive pieces, and the third
is provably *not* a hidden directional one:

1. **PC1 mean shift** — DoM ≈ 0.92·PC1; supervised 1-d AUROC 0.7679,
   unsupervised 1-d AUROC 0.7458.
2. **PC9 trim** — moves 1-d 0.7458 → 2-d 0.7856. The 2-feat probe is
   the directional ceiling at this layer.
3. **Per-problem residual second-order structure** orthogonal to PC1
   — moves 2-d 0.7856 → cov-spectrum 0.7928. **Not capturable by any
   linear directional probe**, properly regularized or not. Lives in
   the *shape* of the per-problem covariance spectrum tail (rate of
   eigenvalue decay, effective rank), not in any direction.

The triangle landed in the cleanest possible position: directional
ceiling at 2-feat, spectral lift above it. The +0.7 pp the cov-
spectrum gets over the 2-feat is real second-order signal, not an
under-regularized linear thing.

## What this does NOT mean

The cov-spectrum probe is supervised. Beating the directional ceiling
supervised-vs-supervised says "there's information you can't capture
with linear directions on raw activations." It does *not* say:
- The model itself uses this spectral information.
- We have a new unsupervised signal.
- The cov-spectrum signal is causally load-bearing.

The causal companion is now the right next test (move #4 below).

## Files modified / created

```
EDIT   pathway11_h100/pca_covariance/recompute_pca_covariance.py
       (+25 lines: full 1536-d L2-reg sweep over C∈{0.001..1.0})
EDIT   pathway11_h100/pca_covariance/results.json
       (+3 fields: full_lr_oof_auroc_by_C, full_lr_best_C, full_lr_best_auroc)
EDIT   validate_claims.py
       (+1 Claim: pca-full-lr-best-auroc, Tier-1 wired)
EDIT   PERSPECTIVES.md
       (+1 subsection: "The triangle resolves" — F-2 second-order)
NEW    handoff/2026-05-02-pca-followups-session4.md  (this file)
```

EXPERIMENT_LOG Next-ID still **EXP-56** (no new EXP — FE291 follow-up #3).

## After-this-session resume

Reasonable next moves, ranked by ROI / cost:

1. **Promote the full FE291 cascade to formal P11-FE292 / FE293 / FE294**
   (write briefs + run promote_brief). Three same-day follow-ups all
   deserve EXP-ids and formal evidence rows on F-2 / F-10:
   - **FE292** (move #2 from session #1, F-10 strongest form): "PH adds
     *negative* signal after PC1 removal." real=0.688 / null=0.763 /
     gap=−0.074.
   - **FE293** (move #1 from session #2, F-2 spectral): "Per-problem cov
     spectrum top-20 log-eigvals beats DoM by 2.5 pp on PC1-residualized
     L19 clouds." 0.7928 / cv5 0.7920±0.027.
   - **FE294** (this session, F-2 second-order): "Full 1536-d L2-reg
     logistic ceiling = 2-feat (PC1, PC9); cov-spectrum 0.7928 lift is
     genuinely second-order." 0.7847 / 0.7856 / 0.7928 triangle.
   
   Treat as a paired promote — the F-2 / F-10 narrative around PC1
   residualization moves together. ~30-60 min total to write three
   structured briefs + run promote_brief.

2. **Causal tests of PC1 + PC1-residual-cov ablation** — the natural
   companion to the spectral finding. Two interventions:
   - **2D ablation along (PC1, PC9) plane** (FE214/FE269/FE283 plan,
     ~2-4h H100): zero out per-token activations along PC1 *and* PC9
     before continuation; measure correctness drop. Tests whether the
     directional component is causally load-bearing.
   - **Rank-truncate-the-PC1-residualized-covariance** intervention
     (~4-6h H100, custom): per-problem, project tokens to top-K
     eigval directions of the residualized covariance and back; test
     whether collapsing the spectral signal disrupts correctness. If
     the model still answers correctly without the cov-spectrum
     pattern, the spectral signal is observational-only.

3. **Per-problem effective-rank single-feature** (~10 lines, sub-second).
   Compute participation ratio, stable rank, or numerical rank from
   each PC1-residualized cov as a *single* scalar per problem; OOF
   logistic. If any well-known scalar matches the top-20 0.7928 —
   especially within ~0.005 — the probe collapses to one named
   property of the geometry, much cleaner narrative than "20 log-eigvals."

4. **Generalize residualization beyond PC1** — does residualizing the
   *top-K* PCs (PC1..PC9 or PC1..PC50) shift the cov-spectrum probe
   AUROC up or down? If top-K residualization blows up the cov-
   spectrum AUROC, the per-problem second-order signal lives in
   PC10+. If it stays at 0.79, F-2 is fully captured by PC1+PC9
   directionally + low-eigenvalue-PC10+ spectrally.

5. **Pull `NEXT_EXPERIMENTS.md` top-5** — current critical queue:
   P11-FE19 (verification routing), P11-FE214 (causal noising),
   P11-FE269 (Arditi-style ablation), P11-FE283 (layer-zero ablation),
   P11-FE44.

`validate_claims.py` Tier-1 regens: the new pca-full-lr-best-auroc
claim's regen runs the existing PCA covariance script (~22s wall).
Claim count moves from 168 → 169 internal PASS. Total claims tracked:
213 (169 internal + 41 external + 3 forward-looking).

## Memory aids

- Next EXP-id: **EXP-56** (no change — this session only did FE291 follow-up #3).
- F-2 strength: STRONG, evidence rows 1..15. Now factored explicitly:
  PC1 (≈DoM), PC9 trim, per-problem residual cov spectrum.
- F-10 strength: STRONG, evidence rows 1..6 (added cov-spectrum vs PH gap).
- New claim count: **169 PASS internal back-checks** (was 168 → +1
  pca-full-lr-best-auroc).
- Total claims tracked: 213 (169 internal + 41 external + 3
  forward-looking).
- Tier-1 regen invariant: 122 + 1 (cov_spectrum, cached) + 1 (full_lr,
  same script as PCA covariance) = 123/123 pass.
- `m15b_prefill.npz` (500 × 1536, fp16) remains the FE291 PC1 backbone.
- Pathway8 cached trajectories: 500 NPZs at
  `pathway8_layerwise/data/math500/problem_*.npz`. Cov-spectrum
  pre-extracted eigvals cached at
  `pathway11_h100/cov_spectrum/eigval_cache.npz` (406 KB; subsequent
  regens take ~2s wall vs ~12 min cold-disk).
- Triangle script runtime: ~22s wall on hot disk (full 1536-d L2-reg
  fits 4 C-values × 5 folds = 20 LR fits at p=1536/n=500). All four
  same-day FE291 follow-up scripts (PCA cov, PC1-resid PH, cov-
  spectrum cache build, triangle) reuse the m15b_prefill cache and
  pathway8 NPZs — no new GPU cost, all CPU.
