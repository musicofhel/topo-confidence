# Handoff — PCA follow-ups session #2 (2026-05-02, continuation)

**Date:** 2026-05-02 (later in same day)
**Session entry:** continued from `2026-05-02-pca-covariance-fe291-landed.md`
"After-this-session resume" → executed move #1 (two-feature [PC1, PC9] OOF)
and move #2 (PH on PC1-residualized clouds) per user's "execute next steps".

## What landed

Move #1 (two-feature [PC1, PC9] OOF logistic) completed in-script, plus
one new validate_claims entry. Move #2 (PH on PC1-residualized
trajectory clouds) ran on cached pathway8 data; results in
`pathway11_h100/ph_residuals/pc1_resid_results.json`. PERSPECTIVES.md
gained one post-script section. No new FE / brief / EXP-id.

| Step | What | Outcome |
|---|---|---|
| 1 | Extended `recompute_pca_covariance.py` with 2-feature [PC1, PC9] OOF | `pc1_pc9_oof_auroc=0.7856` (added to results.json + readback) |
| 2 | New Claim `pca-pc1-pc9-oof-auroc` in validate_claims.py | Tier-1 wired to recompute script |
| 3 | validate_claims.py full run | **167 PASS / 0 FAIL / 122/122 Tier-1 regens** |
| 4 | New script `pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py` | PC1-residualized PH on pathway8 L19 trajectories |
| 5 | Move #2 result JSON | `pathway11_h100/ph_residuals/pc1_resid_results.json` |
| 6 | PERSPECTIVES.md + this handoff | Updated |

EXPERIMENT_LOG Next-ID still **EXP-56** (no new EXP this session — these
were FE291 follow-ups, not formal new FE entries).

## Headline numbers

| Quantity | Value | Notes |
|---|---|---|
| Two-feature [PC1, PC9] OOF logistic AUROC | **0.7856** | Exceeds supervised DoM 0.7679 by 1.8 pp |
| Predicted (handoff #1 said "0.76 ± 0.005, matching DoM") | 0.76 ± 0.005 | Overshot by ~2.5 pp |
| validate_claims invariant post-add | 167 PASS / 0 FAIL | Was 166 before this session |
| Total claims tracked | 211 | (167 internal + 41 external + 3 forward-looking) |
| PC1-resid PH real AUROC | **0.6884** | actual residualized data 5-PH features, OOF |
| PC1-resid PH null AUROC | **0.7628** | matched-cov Gaussian null on residualized cov |
| gap_real_minus_null | **−0.074** | null *exceeds* real by 7.4 pp |
| Verdict on F-10 strongest form | **STRENGTHENED** | covariance spectrum carries signal, PH adds nothing on top |

## What changed in the framing

**Move #1 surprise.** I expected the 2-feature [PC1, PC9] OOF logistic
to land near DoM 0.7679 — that would have closed the basis-decomposition
story cleanly: "DoM = 0.92·PC1 + 0.225·PC9 plus noise, identical
projection AUROC." Instead 0.7856 *exceeds* DoM by 1.8 pp.

The mechanism: DoM = μ⁺ − μ⁻ optimizes mean separation, not
classification AUROC. In PC basis its weights are (0.922, …, 0.225 at
PC9, …) — the natural mean-difference projection. A 2-feature OOF
logistic in (PC1, PC9) is free to re-weight, and what it learns gives
PC9 disproportionately more weight than DoM does. PC9 is *signal-dense
per unit variance*: 0.658 single-feature AUROC at 2.5% var share
compared to 0.7457 / 14.7% for PC1.

**Implication for F-2:** the supervised 1536-d probe at AUROC 0.7731
(F-2's anchor number) is closer to the 2-feature 0.7856 than to plain
DoM 0.7679 — both 1536-d logistic and 2-feature (PC1, PC9) are
re-weighting projections, both beat DoM, and the gap (0.7731 − 0.7856 =
−0.013) is small. **Open question, cheap follow-up:** how much of F-2's
1536-d gap-over-DoM is captured by just (PC1, PC9)? If most of it,
F-2's "supervised structure" is two named, unsupervised-identifiable
directions plus a re-weighting. If a meaningful fraction lives in
PC10..PC1535, F-2 has more to it.

**Move #2 result.** Tested F-10's strongest form: project out the FE291
PC1 from every L19 trajectory token, compute 5 PH features on the
residualized cloud, compare against matched-cov Gaussian null. The
actual outcome was the third branch (and the most interesting one):

- Real PH features (PC1-residualized) → OOF AUROC **0.688**
- Matched-cov Gaussian null → OOF AUROC **0.763**
- Gap = real − null = **−0.074** (null *exceeds* real by 7.4 pp)

**Reading.** F-10 in its strongest form *strengthens*, not weakens.
After PC1 removal:
- The residualized cloud's *covariance spectrum* carries correctness
  signal (the null AUROC moved from 0.693 raw to 0.763 PC1-residualized
  — a ~7 pp jump).
- The actual data's PH features carry *less* signal than what a
  Gaussian with matching covariance would. Real clouds are
  *topologically simpler than their cov-matched Gaussians* (real has
  ~27% fewer H1 loops; lower H0 entropy; lower H1 persistence
  entropy). Real clouds are a low-d submanifold of the cov ellipsoid,
  not a Gaussian fill.
- Net: PH features are *strictly worse* than covariance-spectrum
  features at predicting correctness. F-10 sharpens to: **"PH adds no
  signal beyond covariance, and on PC1-residualized clouds, PH adds
  *negative* signal — covariance alone wins by 7.4 pp."**

**Why the null AUROC jumped.** The null draws Gaussians from each
problem's residualized covariance. Per-problem cov spectrum (after PC1
projection-out) varies systematically with correctness. Removing PC1
exposes per-problem-specific eigenvalue structure of PC2..PC1536 that
the null's PH features pick up. Plausible mechanism: PC1 dominates the
cloud's cov in a roughly correctness-uniform way (it's the dominant
variance direction across all problems), and removing it amplifies the
signal-to-variance ratio of PC2..PC1536 axes whose spectrum encodes
problem-difficulty / trajectory-shape patterns that correlate with
correctness.

**What this does NOT do.** It does not weaken F-10 ("PH features
predict correctness no better than a matched-cov Gaussian null"). It
does not contradict FE291's PC1 dominance result. It does not give us
a new exploitable signal for selective prediction (the 0.763 null
AUROC is a feature of the *null*, not of any practical extractor — to
exploit it you'd extract per-problem-cov-spectrum features directly,
not via PH on a Gaussian sample).

## Files modified / created

```
EDIT   pathway11_h100/pca_covariance/recompute_pca_covariance.py
       (+15 lines: 2-feature [PC1, PC9] OOF + readback)
EDIT   pathway11_h100/pca_covariance/results.json
       (+1 field: pc1_pc9_oof_auroc=0.7856)
NEW    pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py
NEW    pathway11_h100/ph_residuals/pc1_resid_results.json
NEW    pathway11_h100/ph_residuals/pc1_resid_run.log
EDIT   validate_claims.py
       (+1 Claim: pca-pc1-pc9-oof-auroc, Tier-1 wired)
EDIT   PERSPECTIVES.md
       (+2 subsections: post-script update on 2-feat beats DoM, and
        F-10 strongest form: PH adds negative signal after PC1 removal)
NEW    handoff/2026-05-02-pca-followups-session2.md  (this file)
```

## After-this-session resume

Reasonable next moves, ranked by ROI / cost:

1. **Per-problem cov-spectrum probe** (~30 min CPU, *new*) — direct
   follow-up to move #2's surprise. The 0.763 null AUROC implies
   per-problem cov spectrum (post-PC1-residualization) is itself
   correctness-predictive. Pipeline: for each pathway8 problem L19
   trajectory, project out FE291 PC1, compute eigenvalues of residual
   cov (top-k or all), feed top-k spectrum as a feature vector to a
   5-fold OOF logistic. If this beats 0.763 — and especially if it
   beats DoM's 0.7679 — F-2's signal is fully spectral, not just
   directional. Cheap, well-motivated, no new GPU cost.

2. **Move #2 promotion to formal FE** (write brief + run promote_result):
   The verdict is informative enough to deserve a brief + EXP-56 + F-10
   evidence row. Treat as P11-FE292; reuse the FE291 brief structure;
   add 3 claims (real_auroc, null_auroc, gap). validate_claims Tier-1
   regen would just be the script we already wrote.

3. **Two-feature decomposition triangle** (~10 lines, sub-second) —
   compare on a *matched* protocol: 1-d DoM logistic (0.7679, FE291
   SEED=0), 2-feat (PC1, PC9) logistic (0.7856), full 1536-d L2-reg
   logistic (TBD). If 1536-d ≈ 2-feat, F-2 is exactly two named
   directions. If 1536-d > 2-feat by > 0.01, residual signal in
   PC10..PC1535. Watch overfitting at n=500/p=1536; use C=0.01–0.1.

4. **Causal tests of PC1** — same FE214/FE269/FE283 plan, ~2-4h H100
   per test. Now with stronger motivation: PC1 carries 92% of DoM and
   84.9% of DoM-energy; PC9 adds the rest; PH on residual is below
   null. The "F-2 lives in (PC1, PC9)" story is concrete enough that
   ablating PC1 alone is a meaningful causal test. Consider 2D
   ablation along the (PC1, PC9) plane for the strongest version.

5. **Pull `NEXT_EXPERIMENTS.md` top-5** — current top-5 (post-FE291
   regen): P11-FE19 (verification routing), P11-FE214 (causal noising),
   P11-FE269 (Arditi-style ablation), P11-FE283 (layer-zero ablation),
   P11-FE44.

`validate_claims.py` Tier-1 regens are wired and verified end-to-end:
122/122 pass, full run completes within reasonable time (multiple
minutes for FE181 / FE115). The new pca-pc1-pc9-oof-auroc claim is
included.

## Memory aids

- Next EXP-id: **EXP-56** (no change — this session only did FE291 follow-ups).
- F-2 strength: STRONG, evidence rows 1..13. F-10 strength: STRONG,
  evidence rows 1..5. Unchanged from prior handoff.
- New claim count: **167 PASS internal back-checks** (was 166 → +1
  pca-pc1-pc9-oof-auroc).
- Total claims tracked: 211 (167 internal + 41 external + 3
  forward-looking).
- Tier-1 regen invariant: **122/122 pass**.
- `m15b_prefill.npz` (500 × 1536, fp16) remains the FE291 PC1 backbone.
  PC1 unit vector can be re-derived in <1s by `derive_fe291_pc1()` in
  `recompute_pc1_resid_ph.py`.
- Pathway8 cached trajectories: 500 NPZs at
  `pathway8_layerwise/data/math500/problem_*.npz`, ~17 GB total, with
  `states[19]` = (T_i, 1536) per problem.
- Move #2 runtime: ~35 min CPU on this machine for the full 500-problem
  PC1-residualized PH pass.
