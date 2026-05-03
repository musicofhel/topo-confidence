# STATE.md — where was I

*Overwritten at the end of every session. Not appended. For append-only history see `EXPERIMENT_LOG.md`.*

**Date:** 2026-05-01

## Where the project actually is

Mid-pipeline-rebuild. Three things landed since the 2026-04-24 STATE:

1. **Max-depth research-graph re-triage (commit `509cead` on branch
   `max-depth-retriage-2026-04-28`).** All 221 papers reach `status='graphed'`,
   queue empty (`pending_triage = 0`). 24 new structured briefs promoted
   through `promote_brief.py` end-to-end (FE/H/PAPER_INDEX nodes + queue
   regen). The "two-stage cheap+deep" framing was dropped — every admitted
   paper now gets the same maximum-depth pass.
2. **`validate_claims.py` extended with `kind` field.** Three Claim modes:
   - `internal` — back-checked against committed JSONs (PASS/FAIL)
   - `external` — paper-cited anchor, registered without readback (REGISTERED)
   - `forward_looking` — H-N acceptance threshold, becomes live when the
     corresponding FE result JSON lands (PENDING_FE)
   Current state (post-2026-05-01 Phase 2/3 cheap-wins promote chain):
   **178 claims = 134 PASS / 41 REGISTERED / 3 PENDING_FE / 0 FAIL.**
   Tier-1 regen annotates 105 of the 134 internals with cached-intermediate
   recompute commands (FE749 spectral α excluded — regen takes ~2h45m,
   exceeds the 600s per-claim timeout; readback only on Tier-0).
3. **13 new Tier-1 regen scripts** under `pathway11_h100/`. CLAUDE.md, working
   norms, and the claims-gate description in `promote_brief.py` updated to
   reflect the claims invariant (now 134/41/3 = 178 after this session's
   Phase 2/3 promote chain; was 91/41/3 = 135 before).

## Last experiment completed

EXP-58 (P11-FE882 decomposition triangle: full 1536-d L2-reg
directional ceiling). Full 1536-d L2-reg logistic OOF AUROC sweep
over C ∈ {0.001, 0.01, 0.1, 1.0}: max **0.7847 at C=0.001**, then
degrades to 0.7211 at C=1.0 (severe overfit at p=1536/n=500). The
directional ceiling at L19 is the 2-feat (PC1, PC9) probe at 0.7856
— full 1536-d L2-reg recovers 0.7847 ≈ 2-feat within fold noise.
The cov-spectrum (FE881 top-20 log-eigvals) 0.7928 lift above
0.7856/0.7847 is **genuinely second-order**, not under-regularized
linear-directional. F-2 now factors into 3 explicit additive pieces:
(1) PC1 mean shift, (2) PC9 trim (raises 1-d → 2-d 0.7856 directional
ceiling), (3) per-problem residual second-order eigenvalue decay
structure orthogonal to PC1 (raises 2-d → spectral 0.7928, NOT
capturable by any linear direction probe). The causal companion
test (rank-truncate PC1-residualized covariance per-problem before
continuation) is the right next move.

## Queued — next session

**`/home/musicofhel/topo-confidence/PLAN_cheap_wins.md`** is the action queue.
Phased plan for ~25 CPU-only / sub-2h-on-2060-Super FEs that exploit cached
RunPod NPZs. Targets F-2 (the load-bearing prefill L19 DoM AUROC 0.7731) with
the cheapest possible refutations first.

Plan structure:
- **Phase -1** (15 min) — re-validate the pipeline rails before kickoff
  (validate_claims invariant, Tier-1 regens, Neo4j health, queue determinism,
  promote_brief imports clean, working tree clean). Seven explicit checks.
  Block on any failure.
- **Phase 0** (15 min) — verify two cache assumptions: (a) sequence-likelihoods
  in `pathway11_h100/prefill_gated_compute/generations.npz`; (b) per-layer vs
  L19-only prefill in `pathway11_h100/prefill_inversion/cache/`.
- **Phase 1** (~3h CPU) — sub-1h sanity battery: FE719 (LOCO-CV by subject),
  FE448 (length partial-correlation), FE145 (per-layer sweep, skip if Phase 0(b)
  shows L19-only), FE299, FE101.
- **Phase 2** (~3.5h CPU) — ROI-10 anchors: FE115 (Song-Zhong pos/ctx decomp),
  FE749 (spectral α head-to-head), FE181 (token-prob baseline).
- **Phase 3** (~2.5h CPU bundle) — causal corroborators: FE110, FE136, FE244,
  FE319, FE331, FE339, FE254. Run only if F-2 survived Phase 1.
- **Phase 4** (≤2h each on 2060) — FE455 (ConCISE), FE23 (softmax confidence),
  FE282-partial (per-layer extract).
- **Phase 5** (background) — FE116 (PH on residuals, 1 day CPU), FE321
  (zigzag PH).

**Decision branch**: if Phase 1 softens F-2 below AUROC 0.70 on ≥3 of 5 checks,
demote F-2 in FINDINGS.md / STATE.md, skip Phase 3, jump to Phase 2 + Phase 4
to look for a stronger signal.

## Architectural gap closed

`research-graph/promote_result.py` was built 2026-04-30 (smoke-tested via
`briefs/result-2026-04-30-P11-FE115.md --dry-run`). It mirrors
`promote_brief.py`: author writes a structured brief
(`briefs/result-YYYY-MM-DD-<fe-id>.md`) with required sections (`## FE`,
`## EXPERIMENT_LOG entry`, `## STATE.md last-experiment update`,
`## FINDINGS.md updates`, `## New claims`, `## Sources`), and the script
auto-writes EXPERIMENT_LOG.md (append + Next-ID bump), STATE.md ("Last
experiment completed" section replace), FINDINGS.md (in-place block replace
or append before `## Honorable mentions`), Neo4j (FE status + Finding props
+ PRODUCED edges), and regenerates NEXT_EXPERIMENTS.md. PERSPECTIVES.md is
intentionally never touched. Same claims-gate as `promote_brief.py:322`.
Phase 1 kicks off with `promote_result.py` already in place.

## Pod status

- `y687b9z2dgukcj` (Pathway 11 main pipeline): **REMOVED** on 2026-04-24 at
  exp1_cross_model end-of-session. volumeInGb=0 so stop wouldn't have preserved
  disk. Re-create from scratch if needed; ~15 min setup.
- `lsuoka6bo8io7m` (gibberish + no-CoT): **STOPPED, volumeInGb=50, disk
  preserved**. Resume with `runpodctl pod start lsuoka6bo8io7m`. pip packages
  live on container disk (not volume) and need reinstalling after stop/start.

The cheap-wins plan does NOT require pod resumption — all CPU and 2060-local.

## Disk usage summary

- `pathway11_h100/data/math500_7b/` — 42 GB (7B MATH-500, 1024 tok)
- `pathway8_layerwise/data/math500/` — 19 GB (1.5B MATH-500, 1024 tok)
- `pathway8_layerwise/data/bbh/` — 18 GB (BBH 3×250, 1024 tok)
- All other caches combined — <1 GB.
- **Total project** — ~94 GB on `~/topo-confidence/`.

## Open threads for next session

1. **Kickoff**: Phase -1 re-validation in PLAN_cheap_wins.md. Block on any
   failure, fix root cause, then proceed.
2. **First cheap-win lands through `promote_result.py`** — built and
   dry-run smoke-tested 2026-04-30. Run end-to-end (without --dry-run) on
   the first real FE result so any remaining rough edges surface
   immediately.
3. **`pathway11_h100/` is NOT in the GitHub remote** — back up if the local
   disk is at risk.
4. **Branch state**: `max-depth-retriage-2026-04-28` is pushed but not merged
   to main. Decide whether to merge before starting Phase 1 work or keep the
   branch live.
