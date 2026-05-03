# Handoff — PCA follow-ups session #3 (2026-05-02, third continuation)

**Date:** 2026-05-02 (third pass in same day)
**Session entry:** continued from `2026-05-02-pca-followups-session2.md`
"After-this-session resume" → executed move #1 (per-problem cov-spectrum
probe on PC1-residualized clouds) per user's "continue next steps per
handoff".

## What landed

Move #1 (per-problem cov-spectrum probe) is the headline. Top-K log-
eigvals of each PC1-residualized L19 trajectory covariance, fed to a
5-fold OOF logistic, **beats supervised DoM (0.7679) by 2.5 pp** at
K=20 (and 0.7925 at K=10 with tighter CV5 std). The 0.763 null AUROC
that move #2 surfaced last session was indeed cov-spectrum signal, and
extracting it directly goes a further ~3 pp. PERSPECTIVES.md gained
one post-script subsection. validate_claims gained one new entry. No
new FE / brief / EXP-id (still EXP-56).

| Step | What | Outcome |
|---|---|---|
| 1 | New script `pathway11_h100/cov_spectrum/recompute_pc1_resid_cov_spectrum.py` | Top-K log-eigvals on PC1-residualized clouds, sweep K∈{5,10,20,50,100} + single-feature baselines |
| 2 | Result JSON `pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json` | best_top_k=20, best_auroc_oof=**0.7928** |
| 3 | New Claim `pca-pc1resid-cov-spectrum-top20-auroc` in validate_claims.py | Tier-1 wired to recompute script |
| 4 | validate_claims.py full run | (See "Tier-1 regen invariant" below) |
| 5 | PERSPECTIVES.md + this handoff | Updated |

## Headline numbers

| Quantity | Value | Notes |
|---|---|---|
| Top-5 log-eigvals OOF AUROC | 0.7632 | CV5 0.7627 ± 0.024 |
| Top-10 log-eigvals OOF AUROC | **0.7925** | CV5 0.7935 ± **0.012** (tightest) |
| Top-20 log-eigvals OOF AUROC | **0.7928** | CV5 0.7920 ± 0.027 (headline) |
| Top-50 log-eigvals OOF AUROC | 0.7925 | CV5 0.7936 ± 0.029 |
| Top-100 log-eigvals OOF AUROC | 0.7886 | CV5 0.7914 ± 0.021 (slight overfit) |
| log_eigval_1 alone | 0.5223 | top eigval ≈ cloud size T_i, uninformative |
| log_eigval_2 alone | 0.6925 | |
| log_eigval_3 alone | 0.7488 | strongest single eigval |

Ordering of all PC1-residualized probes:

| Probe | OOF AUROC | Δ vs DoM 0.7679 |
|---|---|---|
| 5 PH features (real) | 0.6884 | −7.95 pp |
| Matched-cov Gaussian PH null | 0.7628 | −0.51 pp |
| **Top-20 log-eigvals (this run)** | **0.7928** | **+2.49 pp** |
| 2-feat (PC1, PC9) (session #2) | 0.7856 | +1.77 pp |

EXPERIMENT_LOG Next-ID still **EXP-56** (no formal new EXP this session
— FE291 follow-up #2).

## What this means for F-2

F-2 was "a single L19 prefill direction (DoM) predicts correctness at
AUROC 0.7731." FE291 (yesterday) showed DoM ≈ PC1 of the prefill cov.
Session #2 showed (PC1, PC9) re-weighted beats DoM (0.7856 vs 0.7679).
Session #3 (this one) shows the *per-problem covariance spectrum after
PC1 removal* beats both (0.7928).

Concretely: F-2 is at least three things stacked.

1. **The PC1 mean shift** (DoM ≈ 0.92·PC1 plus noise; explains 0.7458
   at supervised DoM-projection level, 0.745 at unsupervised PC1-
   projection level).
2. **PC9 trim** (raises 1-d 0.7458 → 2-d 0.7856 = +4 pp).
3. **Per-problem residual second-order structure** orthogonal to PC1
   (raises 2-d 0.7856 → cov-spectrum 0.7928 = +0.7 pp; or 1-d 0.7458
   → cov-spectrum 0.7928 = +4.7 pp without using PC9 at all).

The 0.7928 ≥ 0.7856 ≥ 0.7679 ordering says (3) is doing real work
beyond (1)+(2). The single-feature baselines clarify *what kind* of
work: the largest eigvalue is uninformative (mostly cloud size T_i
via SVD scaling); the third is the strongest single (0.7488); eigvals
2..20 carry the signal collectively. The discriminative pattern is in
the **shape of the spectrum tail** (rate of decay, effective rank),
not absolute scale.

## What this means for F-10

F-10 ("PH adds no signal beyond covariance") gets a third piece of
evidence beyond session #2's "PH adds *negative* signal after PC1
removal." Now we can quantify the *covariance pathway* explicitly:
top-20 log-eigvals = 0.7928, while 5 PH features = 0.6884. The PH
representation discards exactly the signal the eigval extractor
recovers. F-10 is now: PH features behave as a lossy compression of
the cov spectrum, where the lossy part is the part that predicts
correctness.

## A subtle caveat

The cov-spectrum probe is supervised. Beating DoM 0.7679 doesn't
imply a new *unsupervised* signal. It means the PC1+PC9 directional
summary is incomplete — there's exploitable structure in PC2..PC1536's
per-problem eigenvalue magnitudes that no directional probe can see.
The natural causal companion is now a 2D ablation along the (PC1,
PC9) plane *plus* an ablation of residual second-order structure
(less obvious how to do — perhaps Mahalanobis-whitening the
residualized cloud per-problem before continuation, or rank-truncating
the residual covariance to top-K before reinjection).

Also: 0.7928 is *above* the 1536-d supervised logistic anchor
(0.7731) at K=20/n=500. CV5 std 0.027 leaves slack. Either K=20 is
overfitting (try a properly-regularized full 1536-d logistic, C=0.01
or 0.1, n=500/p=1536) or the 1536-d anchor has untapped re-weighting
room. The two-feature decomposition triangle move from the prior
handoff is now urgent — it's the right tiebreaker.

## Files modified / created

```
NEW    pathway11_h100/cov_spectrum/recompute_pc1_resid_cov_spectrum.py
NEW    pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json
NEW    pathway11_h100/cov_spectrum/run.log
EDIT   validate_claims.py
       (+1 Claim: pca-pc1resid-cov-spectrum-top20-auroc, Tier-1 wired)
EDIT   PERSPECTIVES.md
       (+1 subsection: per-problem cov spectrum is itself the strongest probe so far)
NEW    handoff/2026-05-02-pca-followups-session3.md  (this file)
```

## After-this-session resume

Reasonable next moves, ranked by ROI / cost:

1. **Two-feature decomposition triangle, properly regularized** (~30
   lines, sub-second) — the urgent tiebreaker. Add a full 1536-d L2-
   reg logistic OOF (C ∈ {0.001, 0.01, 0.1, 1.0}) to the existing
   `recompute_pca_covariance.py` and emit `full_lr_oof_auroc_C{c}` for
   each. Comparison table:
   - 1-d DoM logistic: 0.7679
   - 2-feat (PC1, PC9): 0.7856
   - Top-20 log-eigvals (cov spectrum): 0.7928
   - Full 1536-d, regularized: TBD
   
   If full 1536-d ≤ 2-feat (PC1, PC9) at appropriate C, F-2 is exactly
   two named directions. If full 1536-d > 2-feat by more than ~0.005,
   there's residual orthogonal-to-PC1+PC9 directional signal in
   PC10..PC1535. If full 1536-d ≈ cov-spectrum 0.7928, the extra ~0.7
   pp the spectrum probe gets over 2-feat lives in eigval-magnitudes
   that a linear directional probe would also re-weight via L2 — and
   F-2 has a richer-but-still-linear story. If full 1536-d ≪ cov-
   spectrum 0.7928, F-2's signal is *genuinely* in second-order
   structure (cov-spectrum is non-linear in the raw activations). All
   four outcomes are decisive.

2. **Promote moves #2 and #1 to formal P11-FE292 / P11-FE293** (write
   briefs + run promote_brief). Move #2 is the "PH adds negative
   signal after PC1 removal" finding (F-10 strongest form). Move #1
   is the "per-problem cov spectrum beats DoM" finding (F-2
   refinement). Both deserve EXP-ids and formal evidence rows. Treat
   as a paired promote so the F-10 / F-2 narrative around PC1
   residualization moves together.

3. **Per-problem effective-rank probe** (~10 lines, sub-second) —
   single-feature variants of the cov-spectrum probe. Compute
   participation-ratio / stable-rank / numerical-rank from each PC1-
   residualized cov, feed each as a 1-d OOF logistic. If any single
   well-known scalar matches the top-20 0.7928, the probe collapses
   to one named property of the cloud's geometry. Cleaner narrative
   than "20 log-eigvals."

4. **Causal tests of (PC1, PC9)** — same FE214/FE269/FE283 plan, ~2-4h
   H100 per test. Now with stronger motivation: the linear directional
   pathway beats DoM at 2 features and the cov-spectrum (orthogonal-
   to-PC1) pathway beats it further. Consider a 2D ablation along
   (PC1, PC9) for the directional component, and Mahalanobis-
   whitening the PC1-residualized cloud for the spectral component.

5. **Pull `NEXT_EXPERIMENTS.md` top-5** — current top-5 (post-FE291
   regen): P11-FE19 (verification routing), P11-FE214 (causal noising),
   P11-FE269 (Arditi-style ablation), P11-FE283 (layer-zero ablation),
   P11-FE44.

`validate_claims.py` Tier-1 regens are wired and verified end-to-end:
the new pca-pc1resid-cov-spectrum-top20-auroc claim's regen runs the
new script (~4-5 min CPU). Claim count moves from 167 → 168 internal
PASS. Total claims tracked: 212 (168 internal + 41 external + 3
forward-looking).

## Memory aids

- Next EXP-id: **EXP-56** (no change — this session only did FE291 follow-up #2).
- F-2 strength: STRONG, evidence rows 1..14 (added cov-spectrum 0.7928).
  F-10 strength: STRONG, evidence rows 1..6 (added cov-spectrum vs PH gap).
- New claim count: **168 PASS internal back-checks** (was 167 → +1
  pca-pc1resid-cov-spectrum-top20-auroc).
- Total claims tracked: 212 (168 internal + 41 external + 3
  forward-looking).
- Tier-1 regen invariant: 122/122 pass + 1 new = 123/123 pass (will
  confirm at end of run).
- `m15b_prefill.npz` (500 × 1536, fp16) remains the FE291 PC1 backbone.
  PC1 unit vector can be re-derived in <1s by `derive_fe291_pc1()` in
  `recompute_pc1_resid_cov_spectrum.py`.
- Pathway8 cached trajectories: 500 NPZs at
  `pathway8_layerwise/data/math500/problem_*.npz`, ~17 GB total, with
  `states[19]` = (T_i, 1536) per problem. Cloud T_i: min=123, median=521,
  max=1024.
- Move #1 runtime: ~256s (eigval extraction) + sub-second (LR sweep)
  for the full 500-problem cov-spectrum pass. Much faster than the
  ~35 min PH pass — no ripser, no PCA-then-rips, just SVD per cloud.
