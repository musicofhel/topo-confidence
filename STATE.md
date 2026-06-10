# STATE.md — where was I

*Overwritten at the end of every session. Not appended. For append-only history see `EXPERIMENT_LOG.md`.*

**Date:** 2026-06-10

## Where the project actually is

**FE19 verification-routing test complete — verdict C_exact LOSES (2026-06-10, local 2060).**
The post-hoc routing follow-up that FE269 pointed at is now closed too. Self-grade
the K=1 answer and route verify-failures to K=8 / self-correction (Kumaran 2604.22271
PANL signal) does **NOT** beat the achievable uniform-K frontier (upper convex hull
= random K=1/K=8 routing) beyond noise — best routed point +0.1 SE (n=500, SE=0.022).
PANL "K=1-wrong" probe peaks at **L22 AUROC 0.7555, below pre-hoc prefill-DoM 0.7731**
(post-hoc weaker than pre-hoc — opposite of Kumaran's 7B/27B claim). Verbalized
self-verdict is anti-informative (AUROC 0.480); **verify-then-correct HURTS**
(0.474 vs K=1 0.486; 46 right→wrong, 40 wrong→right). **H-19 refuted at 1.5B.**
Two false-win artifacts were caught by fresh-eyes scrutiny (a degenerate verbalized
router sending 85%→K=8, and dominated-point frontier interpolation through K=2/K=4).
Net: the applied story stays **prefill-DoM refuse-and-spend selective prediction (F-8)**,
not verify-and-route. Harness + results under `pathway11_h100/verify_route/`
(run_fe19.sh, verdict.json); brief `research-graph/briefs/result-2026-06-10-P11-FE19.md`.
**Graph bookkeeping deferred** (Docker Desktop was down): run `update_status.py P11-FE19
COMPLETED`, `promote_result.py briefs/result-2026-06-10-P11-FE19.md`, and
`generate_next_experiments.py` once topo-research-graph (bolt 7688) is back up.

**FE269 causal test complete — verdict DIAGNOSTIC (2026-06-09).** The L19
prefill DoM is a *correlational readout, not a causal lever*. This closed the
question that gated the finish line: pursue selective-prediction / routing
(F-8), NOT steering (H-1). Run on a now-removed H100 SXM pod;
harness + 14 result JSONs committed under `pathway11_h100/causal_dom/`.
- Ablation (necessity) FAILED: removing the direction is a no-op on math —
  L19-only ΔMATH −0.2pp (survives 48.4%), all-layer ΔMATH +2.0pp (46.2%);
  GSM8K/MMLU drops ≤2.6pp (no general damage either).
- Addition (sufficiency) FAILED: α-sweep degrades MATH 47.8→33.4%. Entangled,
  not surgical — flips 18–20% of incorrect but retention falls 80→54%; no α
  meets flip≥15% with retention≥85%. First DIRECT test of fixed-vector steering
  on this direction; confirms the prior F-3-rotation *inference* (P10-v2 E1 was
  never executed). See `causal_dom/results/verdict.json`.

**Also locked this session:** F-2 cross-architecture generalization SATISFIED
(Phi-3-mini ⅔-depth 0.8089, Llama-3.2-1B 0.7458 vs Qwen-1.5B ref 0.7731) — F-2
is now STRONG across 3 architectures. `exp1_cross_model/dom_auroc_results.json`.

**Methodology phase complete.** The `nocompute/` experiment suite landed
2026-05-06 (commit 1921e3a, merged to main). Branch
`max-depth-retriage-2026-04-28` is up to date with origin.
`validate_claims.py` invariant: **214/214 internal PASS, 258 claims tracked total**
(214 internal + 41 external + 3 PENDING_FE) — +6 internal this session (FE19: PANL
best/L19 AUROC, verbalized AUROC, verify-then-correct acc + helps-flag, C_exact verdict).
Verify with `python validate_claims.py`.

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

**FE19 (P11-FE19) — verification routing C_exact vs prefill-gating C_infer, + verify-then-correct.
Verdict C_exact LOSES.** Kumaran 2604.22271 PANL post-hoc signal on Qwen-2.5-1.5B, MATH-500 n=500,
local. No non-degenerate verify-routed policy beats the achievable uniform-K frontier (upper convex
hull = random K=1/K=8 routing) beyond noise — best routed point +0.1 SE (SE=0.022). PANL "K=1-wrong"
probe peaks L22 AUROC 0.7555 < pre-hoc prefill-DoM 0.7731 (post-hoc weaker than pre-hoc). Verbalized
self-verdict anti-informative (0.4801). Verify-then-correct HURTS (0.474 vs K=1 0.486). **H-19
refuted at 1.5B**; applied story stays prefill-DoM selective prediction (F-8). Graph node P11-FE19 →
COMPLETED. Results in `pathway11_h100/verify_route/results/` (verdict.json). No GPU; all local.

## Refutes
- **H-19** (verification routing beats prefill gating): refuted at 1.5B.
- **Kumaran 2604.22271** at small scale: verify-then-correct and verbalized
  self-verification both fail to transfer to 1.5B (the PANL *activation* signal
  partially transfers, but is dominated by the pre-hoc prefill signal).

## Cross-paper
- Confirms and extends the prefill-gating finding (`results.json:headline.
  does_prefill_gating_beat_uniform_K8_at_lower_K=false`): C_exact joins C_infer in
  failing to beat random routing. Rybin C_infer/C_exact framing: neither pre-hoc nor
  post-hoc per-problem routing beats the mixed-strategy baseline for this model/benchmark.

## Queued — next session

**`/home/musicofhel/topo-confidence/PLAN_cheap_wins.md`** is the action queue.
Phased plan for ~25 CPU-only / sub-2h-on-2060-Super FEs that exploit cached
cached NPZs. Targets F-2 (the load-bearing prefill L19 DoM AUROC 0.7731) with
the cheapest possible refutations first.

Plan structure:
- **Phase -1** (15 min) — re-validate the pipeline rails before kickoff
  (validate_claims invariant, Tier-1 regens, Neo4j health, queue determinism,
  promote_brief imports clean, working tree clean). Seven explicit checks.
  Block on any failure.
- **Phase 0** (15 min) — verify two cache assumptions: (a) sequence-likelihoods
  in `pathway11_h100/prefill_gated_compute/generations.npz`; (b) per-layer vs
  L19-only prefill in `pathway11_h100/prefill_inversion/cache/`.
- **Phase 1** ✓ COMPLETE — sub-1h sanity battery: FE719, FE448, FE145, FE299,
  FE101 all run in EXP-59..78 sweeps. F-2 survived (not softened below 0.70).
- **Phase 2** (NEXT, ~3.5h CPU) — ROI-10 anchors: FE115 (Song-Zhong pos/ctx
  decomp), FE749 (spectral α head-to-head), FE181 (token-prob baseline).
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

## Compute

**Local only.** 2060 Super + CPU. No cloud GPU. Historical RunPod pods
(`y687b9z2dgukcj` removed, `lsuoka6bo8io7m` stopped) are not active.
All current and future experiments run on local hardware.

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
3. **11 former status-NULL Discord papers** — reconciled 2026-05-09: 10 set
   to `pending_triage` (will be picked up by triage daemon), 1 rejected
   (malformed ID `2503.dual-route-induction`).
4. **`pathway11_h100/` is NOT in the GitHub remote** — back up if the local
   disk is at risk.
