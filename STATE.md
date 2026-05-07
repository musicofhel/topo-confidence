# STATE.md — where was I

*Overwritten at the end of every session. Not appended. For append-only history see `EXPERIMENT_LOG.md`.*

**Date:** 2026-05-06

## Where the project actually is

**Methodology phase complete.** The `nocompute/` experiment suite landed
2026-05-06 (commit 1921e3a, merged to main). Branch
`max-depth-retriage-2026-04-28` is merged and up to date with `main`.
`validate_claims.py` invariant: **172/172 internal PASS, 127/127 Tier-1
regen PASS, 216 claims tracked total** (172 internal + 41 external + 3
PENDING_FE).

The `nocompute/` suite runs all paper-defense experiments against committed
NPZ caches without GPU: 6 scripts, 39 result JSONs, 14 figures. Key
paper-narrative results from this phase:

- 7B AUROC 0.874 is NOT an imbalance artifact (rebalanced: 0.875±0.012),
  but IS substantially a difficulty detector (disagreement-set AUROC: 0.61)
- Prefill DomProbe dominates all pre-generation baselines by +0.18 AUROC
- Pre-gen cascade wins on BOTH cost (0.92×) and latency (p95 14% better)
  vs hybrid — no tradeoff, pre-gen is universally better
- Cross-domain transfer: cos(directions)=0.12 but r(scores)=0.986 —
  operationally redundant, "transfers at 0.796" is the defensible claim
- Mean-pool residualized AUROC = 0.68, below prefill (0.77) — deprecate
- Calibration locked: histogram_15 for 1.5B (ECE=0.041), isotonic for 7B
  (ECE=0.038)

## Last experiment completed

EXP-59 through EXP-68 (overnight sweep, 10 CPU experiments) on 2026-05-06.
Standout: FE919 Diffusion Maps AUROC=0.788 > DoM 0.773 (nonlinear manifold
structure contributes). FE903 mCCA=0.98 confirms shared subspace with
orthogonal DoM directions. FE909 alignment profile peaks at L19 with 7-layer
half-max width (L15-L21). FE889 rules out RLHF rotation alignment. FE930
rules out positional coupling. Full brief:
`research-graph/briefs/result-2026-05-06-P11-overnight-sweep.md`

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
