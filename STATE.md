# STATE.md — where was I

*Overwritten at the end of every session. Not appended. For append-only history see `EXPERIMENT_LOG.md`.*

**Date:** 2026-05-03

## Where the project actually is

Synthesis cascade complete. Three practitioner-facing docs landed
2026-05-03 and have now been propagated through the intent-layer narrative
docs (this session). Branch `max-depth-retriage-2026-04-28` is pushed.
`validate_claims.py` invariant: **172/172 internal PASS, 127/127 Tier-1
regen PASS, 216 claims tracked total** (172 internal + 41 external + 3
PENDING_FE). No experiments ran this session — narrative-doc work only.

The FE291 PCA cascade (FE880, FE881, FE882) landed 2026-05-02 as
commit 39c264f. The synthesis docs distill those + every prior finding
into one practitioner briefing.

## Last documents completed

- `SYNTHESIS.md` (2026-05-03, 375 lines) — 10-min practitioner briefing.
  What survived (DoM 0.7731, selective 71.6%, decomposition triangle),
  what was overturned (PH=null, 256-tok artifacts, fixed-vector steering,
  CoE redundant), what is new (cov-spectrum 0.7928, F-10 strengthening,
  CAST PC1 ≈ supervised DoM), 8 honest open questions.
- `NOVELTY_AUDIT.md` (2026-05-03, 469 lines) — F-1..F-10 ranked novel /
  refines-prior-work / parallel-discovery against the 220-paper
  research-graph. Cov-spectrum probe ranked #1 novel; F-2 AUROC
  parallel-discovery on Qwen-VL-7B (Zhu 2509.12886).
- `APPLICATIONS.md` (2026-05-03, 387 lines) — 6 deployment surfaces with
  honest "what would have to be true for production" checklists.
  Selective serving for small-model APIs ranked first to ship.
- `handoff/2026-05-03-intent-layer-and-githubio-update.md` — the queued
  next-session plan that drove this session's work.

The last experiment was EXP-58 (P11-FE882 decomposition triangle: full
1536-d L2-reg directional ceiling, 0.7847 at C=0.001) on 2026-05-02.
Cov-spectrum 0.7928 above 0.7856/0.7847 is genuinely second-order.

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

1. **Causal companion test for F-2.** The rank-truncate-PC1-residualized-cov
   ablation (the natural follow-up to the cov-spectrum 0.7928 result) is
   not yet a `:FutureExperiment` node — file as one. Top priority on the
   queue once the next experiment session starts.
2. **Cross-architecture replication of F-2.** Phi-3-mini and Llama-3.2-1B
   1024-tok caches exist in `pathway11_h100/exp1_cross_model/`; the prefill
   DoM AUROC has not been computed for them yet. Cheap (~1h CPU each).
3. **12 untriaged Discord-admitted papers** in research-graph Neo4j
   (status NULL, missed by `query.py pending` CLI filter). Reconciliation
   procedure documented in
   `handoff/2026-05-03-findings-synthesis-and-applications.md` "Open queue
   item 1".
4. **`pathway11_h100/` is NOT in the GitHub remote** — back up if the local
   disk is at risk.
5. **Branch state**: `max-depth-retriage-2026-04-28` is pushed (head
   1bd0c1b after synthesis cascade). Not merged to main. Decide whether
   to merge before starting Phase 1 / causal-companion work or keep the
   branch live.
