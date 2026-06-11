# STATE.md — where was I

*Overwritten at the end of every session. Not appended. For append-only history see `EXPERIMENT_LOG.md`.*

**Date:** 2026-06-11

## Where the project actually is

**Generalization-First Edge Program EXECUTED end-to-end (EXP-81 + EXP-82, SPEC v6,
2026-06-11, local 2060+CPU).** The headline finding flips which readout the F-8 gate
should READ. A transfer bake-off (`pathway11_h100/generalization_edge/`, 15/15 anchors
reproduce) over a (model × domain) panel shows the **free length+mean_logprob baseline
is the most GENERALIZING correctness readout** — T1 LOCO 0.845 (vs prefill-DoM 0.743),
T3 cross-scale 0.865, and on a **held-out family (SmolLM2-1.7B) OOF AUROC 0.810 ≥ 0.70
(T5 PASS)** — needs no activations, so it is the shippable model-agnostic gate. **All
hidden-state geometry is now refuted as portable value-add:** CoE depth-grid (H-C) and
the last untested arm, prompt-token-cloud (H-E), both lose — prompt-cloud 0.694 < a
plain prompt-length prior 0.706 (gate p=0.42). Positive byproduct: **prompt-length is a
free 0.71 pre-flight (Tier-A) signal.** Applied: the **1.5B→7B hybrid cascade beats the
FE19 random-mix hull +3.38±0.96pp (sig)** — the one routing surface that wins; same-model
reranking (H-F), early-abort (H-G) refuted; conformal valid in-domain, collapses under
MATH→BBH shift (H-H). Phase 4 recalibration: a guarantee transfers **zero-shot to a
stronger model** (1.5B→7B valid at ε=0.2) but **no k≤64 recalibration rescues a valid
cert on harder BBH** — task accuracy is the ceiling. Re-pins F-8 readout (→length+logprob)
and extends F-2 (transfer ceiling; geometry hidden-dim-bound). Brief
`research-graph/briefs/result-2026-06-11-P11-FE-EDGEGEN.md`; full Phase 2/3/4 record
`pathway11_h100/generalization_edge/RESULTS_v6_phase234.md`. Open: far-family T5 (Gemma,
HF-gated, deferred); cross-scale arm-2A (7B prompt clouds GPU-blocked locally — moot,
cloud loses in-domain).

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
`validate_claims.py` invariant: **220/220 internal PASS, 264 claims tracked total**
(220 internal + 41 external + 3 PENDING_FE) — +6 internal in EXP-82 (edge-v6: SmolLM2
T5 free/length, arm-2A prompt-length + gate-null, Phase-4 cross-scale ceiling + conformal
validity). Prior +6 (FE19: PANL best/L19 AUROC, verbalized AUROC, verify-then-correct).
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

**EXP-81/82 (Generalization-First Edge Program, SPEC v6) — free length+logprob baseline is the most GENERALIZING correctness readout; geometry fully closed; cross-model cascade revived.** Local, Qwen 1.5B/7B + SmolLM2-1.7B held-out, MATH+BBH. Harness `pathway11_h100/generalization_edge/` (15/15 anchors). length+logprob: T1 LOCO 0.845, T3 cross-scale 0.865, and **held-out SmolLM2 T5 0.810 ≥ 0.70 (PASS)** — the shippable, zero-activation model-agnostic gate. No hidden-state geometry adds portable value: raw CoE +1.75pp in-domain (hidden-dim-bound), depth-grid n.s./≤0.63 (**H-C**), prompt-cloud 0.694 < prompt-length 0.706 (**H-E refuted**); geometry fully closed. New free Tier-A signal: prompt-length 0.71. Applied: hybrid 1.5B→7B cascade beats FE19 hull +3.38pp (sig); reranking (H-F), early-abort (H-G) refuted; conformal valid in-domain, collapses under shift (H-H); Phase-4 recalibration restores a guarantee zero-shot cross-scale but not cross-domain (task accuracy is the ceiling). Re-pins F-8 readout (→length+logprob), extends F-2 (transfer ceiling; geometry hidden-dim-bound). Full Phase 2/3/4 record `pathway11_h100/generalization_edge/RESULTS_v6_phase234.md`. Open: far-family T5 (Gemma, deferred); cross-scale arm-2A (GPU-blocked, moot).

---

## Addendum — Phase 2/3/4 executed (EXP-82, 2026-06-11, local 2060)

The forward phases ran locally (per CLAUDE.md "Local only"). Full record:
`pathway11_h100/generalization_edge/RESULTS_v6_phase234.md`.

**Phase 3 (V5-2, T5 held-out) — PASS.** Generated MATH-500 K=1 greedy on
**SmolLM2-1.7B-Instruct** (near-family holdout, zero role in selection; acc 21.0%).
Free length+logprob OOF AUROC **0.8097 ≥ 0.70** — the shippable, model-agnostic,
zero-activation generate-then-abstain gate generalizes to an unseen family. On
SmolLM2 **length carries (0.781), logprob is near-chance (0.518)** — the mirror of
the MATH→BBH cell (logprob carries, length degenerate); pinning *both* scalars is
what makes the gate robust to the flip. Far-family Gemma-2-2b-it deferred (HF-gated,
token 403s). `results/phase3_t5_free_baseline.json`.

**Phase 2 arm-2A (V5-3, H-E) — REFUTED.** Prefill pass over the 500 MATH prompts on
Qwen-1.5B; L19 prompt-token-cloud top-20 covariance log-eigvals gated against
prompt-length. Prompt-cloud 0.694 < prompt-length **0.706**; gate Δ=−0.0096, p=0.42;
cloud PC1 ↔ prompt-length r=−0.89 (the V3-1 length confound, now on the prompt side).
**Geometry fully closed** — H-A, H-C, H-E all refuted; no activation feature beats
free signals anywhere. Positive byproduct: **prompt-length is a free 0.71 pre-flight
(Tier-A) correctness signal** (previously unmeasured). Cross-scale arm-2A (Qwen-7B
prompt clouds) is GPU-blocked locally (7B bf16 ~15GB > 8GB) and moot (cloud loses
in-domain). `results/phase2_arm2a.json`.

**Phase 4 (V5-5, recalibration) — nuanced.** Freeze source extractor, refit a light
head on k∈{0,8,16,32,64} target labels (R=200, large cells only). Cross-scale
1.5B→7B: the conformal guarantee transfers **zero-shot** (k=0, ε=0.2 → 60% coverage,
valid) because a threshold calibrated on the harder source is conservative on the
easier target; tiny k hurts (CP small-sample), recalibrate only past k=32. Cross-domain
MATH→BBH: ranking ports (0.785) but **no light-head recalibration up to k=64 buys a
valid certificate even at ε=0.3** — BBH's 24% base rate caps answered-accuracy; target-
task accuracy, not calibration, is the ceiling. `results/phase4_recalibration.json`.

**Net (v6 complete, local-constrained):** ship the free length+logprob gate (T5 0.81);
geometry fully closed; new Tier-A prompt-length signal (0.71); guarantees transfer
zero-shot to a stronger model but not to a harder domain. Open: far-family T5 (Gemma,
deferred), cross-scale arm-2A (GPU-blocked, moot). H-E refuted (added).

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
