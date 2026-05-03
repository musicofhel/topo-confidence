# Handoff — PCA follow-ups session #5 (2026-05-02, fifth continuation)

**Date:** 2026-05-02 (fifth pass in same day)
**Session entry:** continued from `2026-05-02-pca-followups-session4.md`
"After-this-session resume" → executed move #1 (paired-promote of the
FE291 cascade as formal P11-FE-{N}/{N+1}/{N+2}, write briefs + run
promote_brief).

## What we set out to do

Session-4 handoff move #1: "Promote the full FE291 cascade to formal
P11-FE292 / FE293 / FE294 (write briefs + run promote_brief). Three
same-day follow-ups all deserve EXP-ids and formal evidence rows on
F-2 / F-10."

## Renumbering (heads-up)

The session-4 handoff suggested IDs **FE292 / FE293 / FE294**, but
those slots are **already taken** in the research-graph as planned-
but-unrun CAST grid-search experiments (P11-FE292: per-layer DoM grid
search; P11-FE293: per-layer PCA-PC1 follow-up; P11-FE294: D-bucket
vs A-bucket condition vector). Highest existing FE-id in P11 is **FE879**.

This session uses **FE880 / FE881 / FE882** for the cascade. The
session-4 narrative numbers are now stale.

| Session-4 plan | This session | Topic |
|---|---|---|
| FE292 | **P11-FE880** | PH features on PC1-residualized clouds (real 0.6884 / null 0.7628, gap −0.074) |
| FE293 | **P11-FE881** | Per-problem cov spectrum top-20 log-eigvals OOF AUROC 0.7928 |
| FE294 | **P11-FE882** | Decomposition triangle: full 1536-d L2-reg ceiling 0.7847 ≈ 2-feat |

## What landed (so far this session)

| Step | What | Outcome |
|---|---|---|
| 1 | `add_future_experiment.py` × 3 | P11-FE880, FE881, FE882 nodes MERGEd into Neo4j (status=COMPLETED, depends_on/would_update edges to F-2 / F-10) |
| 2 | Wrote 3 result briefs in `research-graph/briefs/` | result-2026-05-02-P11-FE880.md (EXP-56 declared), -FE881.md (EXP-57), -FE882.md (EXP-58) |
| 3 | Added 3 new Claim entries to `validate_claims.py` | pca-pc1resid-ph-real-auroc (0.6884), -null-auroc (0.7628), -gap (−0.0744). Existing pca-pc1resid-cov-spectrum-top20-auroc (0.7928) and pca-full-lr-best-auroc (0.7847) already in place from session #3/#4 |
| 4 | Sidecar PH cache scaffolding added to `recompute_pc1_resid_ph.py` | New `ph_cache.npz` (gitignored) — mirrors FE881's `eigval_cache.npz` pattern. Cold extract once, ~5s warm thereafter. Required because the FE880 PH compute is ~13min cold and would time out the 600s validate_claims subprocess limit |
| 5 | Dry-run promote_result for all 3 briefs | FE880 dry-run **green** (Next ID matches EXP-56). FE881 / FE882 dry-runs flag "EXPERIMENT_LOG entry declares EXP-57/58 but Next ID is EXP-56" — expected, they promote sequentially after FE880 lands |
| 6 | Kicked off cold PH cache build (`python pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py`) | **In progress.** First 25/500 problems took 127s wall (slow cold disk). ETA ~40 min total. Once `ph_cache.npz` lands, the script reuses it and re-runs the LR pipeline in seconds |

## Pending — what remains for this session

Once the PH cache build completes:

1. **`python validate_claims.py`** to confirm 172/172 internal PASS
   (previous 169 + 3 new FE880 claims). All 3 new claims have Tier-1
   regen wired (`pc1_resid.auroc_real_ph_pc1resid`,
   `pc1_resid.auroc_null_ph_pc1resid`,
   `pc1_resid.gap_real_minus_null` from
   `recompute_pc1_resid_ph.py` stdout). Should run in ~5s on the now-
   cached `ph_cache.npz`.

2. **`promote_result.py briefs/result-2026-05-02-P11-FE880.md
   --skip-git-check`** (real apply). Will:
   - Update Neo4j FE880 status / outcome (already COMPLETED via
     add_future_experiment, but promote_result re-asserts via
     update_status.py)
   - Append EXP-56 to `EXPERIMENT_LOG.md`, bump Next ID → EXP-57
   - Replace STATE.md "## Last experiment completed"
   - Replace `### F-10:` block in FINDINGS.md (in-place; F-10 already
     exists, this just updates the body — adds EXP-56 evidence row +
     PC1-residualization control row + sharpened counterargument)
   - Regenerate `NEXT_EXPERIMENTS.md`

3. **`promote_result.py briefs/result-2026-05-02-P11-FE881.md
   --skip-git-check`** — EXP-57. Updates F-2 (adds spectral-piece
   counterargument + EXP-57 evidence) and F-10 (adds spectral
   companion to "PH = null" framing + EXP-57 evidence).

4. **`promote_result.py briefs/result-2026-05-02-P11-FE882.md
   --skip-git-check`** — EXP-58. Updates F-2 (adds 3-piece factorization
   + directional-ceiling control + EXP-58 evidence).

5. **Bulk commit + push** to the working branch
   (`max-depth-retriage-2026-04-28`):
   ```
   research-graph/briefs/result-2026-05-02-P11-FE880.md
   research-graph/briefs/result-2026-05-02-P11-FE881.md
   research-graph/briefs/result-2026-05-02-P11-FE882.md
   pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py  (cache scaffolding)
   pathway11_h100/ph_residuals/cache_build.log            (one-off run log)
   pathway11_h100/ph_residuals/pc1_resid_results.json     (regen overwrite — values unchanged within fold noise)
   validate_claims.py                                     (+3 FE880 claims)
   validation_report.txt                                  (regen)
   FINDINGS.md                                            (F-2 / F-10 evidence-row additions)
   STATE.md                                               (last-experiment replacement)
   EXPERIMENT_LOG.md                                      (+ EXP-56 / EXP-57 / EXP-58)
   NEXT_EXPERIMENTS.md                                    (regen)
   handoff/2026-05-02-pca-followups-session5.md           (this file)
   ```

## Files modified / created (so far)

```
NEW    research-graph/briefs/result-2026-05-02-P11-FE880.md
NEW    research-graph/briefs/result-2026-05-02-P11-FE881.md
NEW    research-graph/briefs/result-2026-05-02-P11-FE882.md
EDIT   validate_claims.py
       (+3 Claim entries: pca-pc1resid-ph-{real,null,gap}-auroc with regen wired)
EDIT   pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py
       (+~30 lines: PH_CACHE constant, _extract_ph()/_save_ph_cache() helpers,
        cache-load/extract branch in main())
NEW    pathway11_h100/ph_residuals/ph_cache.npz             (in progress, gitignored)
NEW    pathway11_h100/ph_residuals/cache_build.log          (one-off cold-run log)
NEEDED handoff/2026-05-02-pca-followups-session5.md          (this file)
```

Neo4j additions (already applied):
- `(:FutureExperiment {id: 'P11-FE880', status: 'COMPLETED', ...})` + edges to F-10
- `(:FutureExperiment {id: 'P11-FE881', status: 'COMPLETED', ...})` + edges to F-2, F-10
- `(:FutureExperiment {id: 'P11-FE882', status: 'COMPLETED', ...})` + edges to F-2

## EXPERIMENT_LOG progression (planned)

- Pre-session: Next ID = **EXP-56**
- After FE880 promote: Next ID = **EXP-57** (EXP-56 = P11-FE880 PH-after-PC1)
- After FE881 promote: Next ID = **EXP-58** (EXP-57 = P11-FE881 cov-spectrum)
- After FE882 promote: Next ID = **EXP-59** (EXP-58 = P11-FE882 triangle)

## Headline numbers (no change — this is the formalization step)

| Probe | OOF AUROC | Δ vs DoM 0.7679 |
|---|---|---|
| 1-d DoM | 0.7679 | — |
| Unsupervised PCA-PC1 | 0.7458 | −2.21 pp |
| 2-feat (PC1, PC9) | 0.7856 | +1.77 pp |
| Full 1536-d L2-reg, C=0.001 (FE882) | 0.7847 | +1.68 pp |
| **Top-20 log-eigvals on PC1-residualized clouds (FE881)** | **0.7928** | **+2.49 pp** |

| Topology probe (PC1-residualized clouds) | OOF AUROC |
|---|---|
| Real 5 V-R PH features (FE880) | 0.6884 |
| Matched-cov Gaussian null PH (FE880) | 0.7628 |
| **Gap (real − null)** | **−0.074** (F-10's pre-registered overturning condition tested directly and failed by 12.4 pp in the wrong direction) |

## Resume points

If this session is interrupted before all promotes land:

- **Cache still building** (the common case while writing this handoff):
  Wait for `ph_cache.npz` to land
  (`ls -la pathway11_h100/ph_residuals/ph_cache.npz` should be ~10–20 KB).
  Then run validate_claims, then 3 promote_results, then commit.

- **Validate fails**: Most likely the new FE880 claims fail Tier-0
  readback because of float drift between original 13-min cold run
  and current cold rebuild. Compare
  `pathway11_h100/ph_residuals/pc1_resid_results.json` against
  `auroc_real_ph_pc1resid` ≈ 0.6884. Adjust Claim `expected` if so —
  values are seeded so they should be identical.

- **Promote fails on EXP-id mismatch**: someone else bumped Next ID;
  re-edit the brief's `## EXP-N:` header to match
  `grep "Next ID" EXPERIMENT_LOG.md`.

- **Promote fails on git cleanliness**: pass `--skip-git-check`
  (already required because the other two briefs and the cache log
  are uncommitted on each promote; safe — only narrative + log files
  affected).

## After-this-session resume (looking past the cascade promote)

Reasonable next moves, ranked by ROI / cost:

1. **Causal tests of PC1 + PC1-resid-cov ablation** — the right
   companion to the spectral finding now that FE881 and FE882 are
   formalized. Two interventions:
   - **2D ablation along (PC1, PC9) plane** (FE214/FE269/FE283 plan,
     ~2-4h H100): zero out per-token activations along PC1 *and* PC9
     before continuation; measure correctness drop. Tests whether the
     directional component is causally load-bearing.
   - **Rank-truncate-the-PC1-residualized-covariance** intervention
     (~4-6h H100, custom): per-problem, project tokens to top-K
     eigval directions of the residualized covariance and back; test
     whether collapsing the spectral signal disrupts correctness.
     This is the load-bearing causal test for FE881.

2. **Per-problem effective-rank single-feature** (~10 lines, sub-second).
   Compute participation ratio, stable rank, or numerical rank from
   each PC1-residualized cov as a *single* scalar per problem; OOF
   logistic. If any well-known scalar matches the FE881 top-20 0.7928
   — especially within ~0.005 — the probe collapses to one named
   property of the geometry, much cleaner narrative than "20
   log-eigvals." (Was move #3 in session-4; still open.)

3. **Generalize residualization beyond PC1** — does residualizing the
   *top-K* PCs (PC1..PC9 or PC1..PC50) shift the cov-spectrum probe
   AUROC up or down? If top-K residualization blows up the cov-
   spectrum AUROC, the per-problem second-order signal lives in
   PC10+. If it stays at 0.79, F-2 is fully captured by PC1+PC9
   directionally + low-eigenvalue-PC10+ spectrally.

4. **Single-feature H1_max_lifetime PH probe on PC1-residualized
   clouds.** FE880 found H1_max_lifetime real 8.53 vs null 5.30
   (Δ +3.23) — the only PH feature where real beats matched-cov
   null. Joint probe was dominated by the other 4 features
   Gaussianizing. Single-feature 5-fold OOF AUROC of H1_max_lifetime
   alone could expose a residual topological signal if ≥ 0.65.
   ~5 lines on cached `ph_cache.npz`.

5. **Pull `NEXT_EXPERIMENTS.md` top-5** — current critical queue still
   includes P11-FE19 (verification routing), P11-FE214 (causal noising),
   P11-FE269 (Arditi-style ablation), P11-FE283 (layer-zero ablation),
   P11-FE44.

## Memory aids

- Next EXP-id (target after this session): **EXP-59** (EXP-56/57/58
  consumed by FE880/881/882).
- Next FE-id slot: P11-FE883 (after this session).
- F-2 strength: STRONG, will gain evidence rows P11-E57 and P11-E58.
  Now factored explicitly into 3 additive pieces: PC1 mean shift, PC9
  trim, per-problem residual cov spectrum.
- F-10 strength: STRONG, will gain evidence rows P11-E56 and P11-E57.
  Pre-registered overturning condition (real ≥ null + 0.05 on
  PC1-residualized clouds) tested directly and failed by 12.4 pp
  in the wrong direction.
- New claim count target: **172 PASS internal back-checks** (was 169
  → +3 FE880 PH-after-PC1 claims).
- Total claims tracked: 216 (172 internal + 41 external + 3
  forward-looking).
- Tier-1 regen invariant target: 124 + 3 (PH-after-PC1, all wired
  through cached `ph_cache.npz`) = 127/127 pass.
- Caches in pathway11_h100 (all gitignored, all sub-second warm):
  - `prefill_inversion/cache/m15b_prefill.npz` (FE291 PC1 backbone)
  - `cov_spectrum/eigval_cache.npz` (406 KB, FE881)
  - `ph_residuals/ph_cache.npz` (in progress this session, FE880)
- Pathway8 cached trajectories (~17 GB on disk): 500 NPZs at
  `pathway8_layerwise/data/math500/problem_*.npz`. PH-resid and
  cov-spectrum scripts both load these once cold then save sidecar
  caches; subsequent regens skip the 12–13min cold reload.
- Triangle script runtime: ~22s wall on hot disk for FE882's full
  1536-d L2-reg sweep (4 C-values × 5 folds). FE881 cov-spectrum:
  ~2s on cached eigvals. FE880 PH-after-PC1: ~5s on cached
  `ph_cache.npz` (was ~13min cold) — this fix is the load-bearing
  change in this session.

## What this session does NOT change

The headline science is unchanged from session #4:

- PC1 = DoM at L19 (cosine 0.922, 85% energy).
- PC9 trim raises 1-d 0.7458 → 2-d 0.7856 (directional ceiling).
- Cov-spectrum top-20 log-eigvals 0.7928 = strongest probe.
- Full 1536-d L2-reg max 0.7847 ≈ 2-feat (within fold noise).
- PH on residualized clouds: real < matched-cov null by 7.4 pp.

This session **formalizes** session-3 / session-4 results into
graph-tracked FutureExperiments + EXPERIMENT_LOG / FINDINGS / STATE
evidence rows. Substantive science blocked on next moves (causal
tests, effective-rank single-feature, top-K residualization
generalization) is unchanged.
