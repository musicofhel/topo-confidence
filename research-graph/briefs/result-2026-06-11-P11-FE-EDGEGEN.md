# Result brief — Generalization-First Edge Program (SPEC v5), Phase 0 + Phase 1 + Phase 1B

SPEC v5's CPU-only phases, executed end-to-end on existing local caches
(2060 Super, no generation). A reusable transfer-bake-off harness
(`pathway11_h100/generalization_edge/`) was built on the verified
`nocompute/lib.py` substrate and validated against 15 pinned anchors, then used
to (1) adjudicate the readout's **cross-distribution transfer** (Phase 1) and
(2) attack four **applied uses** of the readout (Phase 1B: selection, cascade,
timing, guarantees). Phases 2–4 (cross-arch + held-out-family generation,
recalibration) require GPU generation and were NOT run.

## Headline

**The free two-scalar baseline (generation length + mean token logprob) is the
most *generalizing* correctness readout we have — and no hidden-state geometry
feature adds significant portable value over it.** In-domain the supervised
geometry can match or slightly exceed it (CoE layer-profile +1.75pp, p=0.018),
but the part of the geometry that *adds* signal does not *port*, and the part
that ports (the cross-model depth-grid form) does not add. Per the SPEC v5
V3-4 fork ("ship the simplest winner"), **the deliverable is the free gate.**

Separately, the **cross-model 1.5B→7B cascade is revived** (V4-1): the hybrid
post-gen gate beats the FE19-honest random-mix hull by **+3.38 ± 0.96 pp**
(significant), while same-model selection/reranking and early-abort drafting do
NOT beat their honest baselines.

## What was run

- **Phase 0 harness** — `activation_loader.py` / `probes.py` / `transfer.py` /
  `metrics.py` over the consolidated `nocompute/cache/*.npz` caches. 15/15
  verification anchors reproduce exactly (prefill-DoM 0.7731, concat-ridge-LR
  0.8509, length 0.7986, logprob 0.6721, LOCO number_theory 0.6032/n=62, dry-run
  length+logprob 0.834/n=200, CoE 0.825, K=8 majority 0.554, oracle 0.704,
  FU2 hybrid 0.852 / gap 11.2pp).
- **Phase 1** — transfer bake-off across in-domain (OOF + incremental gate),
  T1 LOCO (leave-one-MATH-category-out), T2 MATH↔BBH cross-domain, T3 1.5B↔7B
  cross-scale.
- **Phase 1B.1–1B.4** — probe-reranking on the K=8 cache (executes H-40/FE41),
  earliest-decision curve, conformal risk control, and the 1.5B×7B overlap
  table + hull-honest cascade re-adjudication.

## Phase 1 — transfer panel (Qwen-2.5, MATH/BBH, n: MATH 500, BBH 750, 7B 500)

In-domain OOF AUROC and the **incremental gate** (readout-score + free vs free,
paired DeLong) — the corrected gate tests the 1-D readout, not raw activations:

| readout | in-dom OOF | +free | Δ over free | p |
|---|---|---|---|---|
| free_baseline (len+logprob) | 0.849 | — | — | — |
| coe_profile (CoE layer-profile) | 0.854 | 0.866 | **+0.0175** | **0.018** |
| concat_ridge (L19 pre+final) | 0.851 | 0.861 | +0.0117 | 0.113 |
| mean_dom (L19) | 0.819 | 0.858 | +0.0092 | 0.129 |
| coe_depthgrid (portable form) | 0.830 | 0.857 | +0.0081 | 0.137 |
| prefill_dom (incumbent) | 0.773 | 0.853 | +0.0042 | 0.413 |
| last_dom (L19) | 0.719 | 0.849 | −0.0004 | 0.650 |

Only the **raw** CoE layer-profile clears the gate (in-domain). Its
cross-model-portable depth-grid form does not.

Transfer (robust aggregate = mean off-diagonal AUROC; gap = in-domain − transfer):

| readout | T1 LOCO agg (worst) | T2 cross-domain agg (worst) | T3 cross-scale agg (worst) |
|---|---|---|---|
| **free_baseline** | **0.845** (0.735 NT) | 0.728 (0.672) | **0.865** (0.837) |
| length_only | 0.797 (0.640) | **0.500** (chance!) | 0.831 (0.799) |
| mean_logprob_only | 0.665 | 0.728 (0.672) | 0.653 |
| prefill_dom | 0.743 (0.603) | 0.726 (0.656) | — (dim-bound) |
| mean_dom | 0.820 (0.684) | 0.547 (0.510) | — |
| concat_ridge | 0.847 (0.661) | 0.727 (0.713) | — |
| coe_profile | 0.833 (0.722) | 0.732 (0.719) | — (dim-bound) |
| coe_depthgrid | 0.838 (0.770) | 0.588 (0.585) | 0.627 (0.619) |

Reading:
- **Cross-category (T1) & cross-scale (T3): the free baseline dominates** (0.845
  / 0.865, near-zero or negative gap). coe_depthgrid is competitive on T1
  (best worst-cell 0.770) but collapses on T3 (0.627, +0.20 gap).
- **Cross-domain (T2): everything clusters ~0.73, and the transferable core is
  logprob, not length** — length_only falls to **chance (0.500)** MATH↔BBH while
  mean_logprob holds 0.728. No geometry beats logprob on T2.
- **Central bet refuted (H-C):** the geometry that adds in-domain (raw CoE
  profile) is hidden-dim-bound and cannot cross scale/arch; the portable form
  (depthgrid) neither adds in-domain (p=0.137) nor transfers (T2 0.588, T3
  0.627). Geometry arms = **LOSE**, the pre-acknowledged modal outcome.

## Phase 1B — applied uses

**1B.4 — overlap table + hull-honest cascade (V4-1, V4-5).** Built the missing
`[K1, K8maj, 7B]` per-problem table (CSV). Of 257 K=1 failures, **7B rescues
136 (52.9%)** vs K=8-majority's 70 (27.2%); rescues concentrate in number_theory
(74%), algebra/prealgebra (~75%); the unrescuable hard core is
intermediate_algebra / precalculus / geometry. Re-scored cascades vs the
**random-mix hull** (FE19 comparator, cost: keep=1, escalate=5):
- **Hybrid (post-gen) cascade: +3.38pp, boot +3.38 ± 0.96pp — SIGNIFICANT.**
- Pre-gen prefill-DoM cascade: +1.70pp, boot +1.73 ± 0.87pp — **n.s.** (marginal,
  as V4-1 predicted). Cross-model routing is real; the pre-gen-only form is not.

**1B.1 — probe-reranking on K=8 (executes H-40/FE41). H-F REFUTED.** `L19_samples`
confirmed = final generated token. Per-sample probe AUROC 0.720 (problem-grouped
OOF), but **no arm beats K=8 plain majority (0.554)**: probe-argmax 0.480
(−7.4pp, McNemar p<0.001, sig WORSE), probe-weighted majority 0.538 (n.s.),
confidence-fallback 0.552 (n.s., no D-bucket regression: 34 vs majority's 36).
The probe carries correctness signal but not the cross-sibling *comparative*
signal that unweighted voting discards — selection at fixed compute does not win
at 1.5B (mirrors FE19).

**1B.2 — earliest-decision curve. H-G REFUTED.** L19 mean-pool DoM AUROC(m):
prefill 0.773 → m=25 0.704 → m=100 0.758 → m=200 0.783 → m=400 0.809 → full
0.819. The mean-pool signal is **below the prefill gate until ~m=200** — the
decision signal does NOT arrive early. Trigger (≥0.82 by m≤128) did not fire;
the abort-and-escalate counterfactual costs 0.96× for +3.7pp (useless).
Decision-relevant: skip early-abort drafting; the pre-gen prefill gate already
matches the first ~200 generated tokens.

**1B.3 — conformal risk control. H-H CONFIRMED.** Split-conformal LTT (CP-upper;
method prior 2402.10978, graphed — no novelty claim). In-domain (δ=0.1, 1000
resplits) a valid **(ε=0.2)** guarantee is achievable: free_baseline validity
0.855 at 28% coverage / 85% answered-accuracy; coe_profile 0.846 at 32%. Under
shift (calibrate MATH → deploy BBH) the guarantee **collapses: infeasible at
ε=0.1/0.2, coverage 0.0 at ε=0.3.** Guarantee validity degrades to nothing
cross-domain — the conformal restatement of the overfit gap; motivates Phase 4
target-domain recalibration.

## Net

The applied goal ("confidence in correctness + routing") is best served by:
(1) **a free, model-agnostic length+logprob gate** that ports across category
(0.845) and scale (0.865) better than any activation probe — the shippable
generalizing readout; (2) **the cross-model hybrid cascade** for compute routing
(+3.38pp over honest random mixing). Same-model reranking, early-abort drafting,
and zero-shot conformal transfer all fail their honest baselines. T5 (held-out
families) is now **closed and confirmed cross-architecture**: the free gate clears
≥0.70 on SmolLM2-1.7B (near, 0.810), Gemma-2-2b-it (far, 0.844), and OLMo-2-1B
(far, 0.838) — because the winning readout needs no activations, it was trivially
testable on any model (Phase 3).

## FE

```yaml
fe_id: P11-FE-EDGEGEN
status: COMPLETED
outcome: "Free length+logprob baseline is the most generalizing correctness readout (T1 LOCO 0.845, T3 cross-scale 0.865, near-zero gap); no hidden-state geometry adds significant PORTABLE value over it. In-domain only the raw CoE layer-profile clears the incremental gate (+1.75pp, p=0.018) but it is hidden-dim-bound; the cross-model depth-grid form neither adds in-domain (p=0.137) nor transfers (T2 0.588, T3 0.627) — H-C refuted, geometry arms LOSE. Cross-domain (T2) transfer is carried by logprob (0.728); length alone collapses to chance (0.500) MATH<->BBH. Applied (Phase 1B): cross-model 1.5B->7B HYBRID cascade beats the FE19 random-mix hull by +3.38+-0.96pp (significant); pre-gen-only cascade marginal (+1.7pp n.s.). H-40/FE41 probe-reranking REFUTED (no arm beats K=8 majority 0.554; probe-argmax -7.4pp). Earliest-decision: mean-pool signal below prefill gate until m~200 (H-G refuted). Conformal valid in-domain at eps=0.2 (~30% cov) but collapses to 0 coverage under MATH->BBH shift (H-H confirmed). Phase 2/3/4 EXECUTED (EXP-82, local): held-out T5 free-baseline OOF AUROC clears >=0.70 on THREE families spanning near AND far architectures — SmolLM2-1.7B (near/Llama 0.810, local), Gemma-2-2b-it (far/distinct-arch 0.844, H100), OLMo-2-1B (far/distinct-arch 0.838, H100) — the shippable zero-activation gate generalizes CROSS-ARCHITECTURE, not just within the Llama family; which scalar carries is family-dependent (SmolLM2 length-only, Gemma both, OLMo-2 length-led), so pin both. arm-2A prompt-token-cloud REFUTED (0.694 < prompt-length 0.706, gate p=0.42, H-E) — geometry fully closed, with a new free Tier-A prompt-length signal at 0.71; Phase-4 recalibration restores a conformal guarantee zero-shot cross-scale (1.5B->7B valid at eps=0.2) but no k<=64 light-head rescues a valid cert cross-domain (BBH 24% base rate is the ceiling, not calibration). Cross-scale arm-2A GPU-blocked locally."
result_json: pathway11_h100/generalization_edge/results/phase1_panel.json
result_json_all:
  - pathway11_h100/generalization_edge/results/phase1_panel.json
  - pathway11_h100/generalization_edge/results/1b1_reranking.json
  - pathway11_h100/generalization_edge/results/1b2_earliest.json
  - pathway11_h100/generalization_edge/results/1b3_conformal.json
  - pathway11_h100/generalization_edge/results/1b4_overlap.json
  - pathway11_h100/generalization_edge/results/phase3_t5_free_baseline.json
  - pathway11_h100/generalization_edge/results/phase2_arm2a.json
  - pathway11_h100/generalization_edge/results/phase4_recalibration.json
triggered_by: ["2402.10978", "2406.15927"]
executes: ["H-40"]
refutes: ["H-F", "H-G", "H-C", "H-E"]
confirms: ["H-H"]
would_update: ["F-2", "F-8"]
```

## EXPERIMENT_LOG entry

## EXP-81: Generalization-First Edge Program (SPEC v6) — Phase 0/1/1B (transfer bake-off + applied uses)

**Date:** 2026-06-11. **Compute:** local (2060 Super, CPU only), Qwen-2.5-1.5B/7B, MATH-500 + BBH×3, existing caches, no generation. **Harness:** `pathway11_h100/generalization_edge/` (15/15 anchors reproduce).

**Phase 1 (transfer bake-off):** the free length+logprob baseline is the most generalizing readout — T1 LOCO agg 0.845 (worst 0.735), T3 cross-scale agg 0.865 (worst 0.837), near-zero/negative gap; dominates incumbent prefill-DoM (T1 0.743 / worst 0.603). No activation geometry adds significant portable value: in-domain only raw CoE layer-profile clears the incremental gate (+1.75pp, p=0.018) but is hidden-dim-bound; portable depth-grid does not add (p=0.137) and transfers poorly (T2 0.588, T3 0.627). Cross-domain T2 (~0.73) carried by logprob (0.728); length → chance (0.500). Geometry arms LOSE (modal). **H-C refuted.**

**Phase 1B:** (1B.4) 7B rescues 52.9% of K=1 fails vs voting 27.2%; hybrid cascade beats random-mix hull +3.38±0.96pp (sig), pre-gen marginal (+1.7pp n.s.). (1B.1) H-40/FE41 probe-reranking refuted — no arm beats K=8 majority 0.554. (1B.2) signal not early — mean-pool below prefill until m~200 (**H-G refuted**). (1B.3) conformal valid in-domain (ε=0.2, ~30% cov) but 0 coverage under MATH→BBH shift (**H-H confirmed**). Phase 2/3/4 logged separately as EXP-82.

## STATE.md last-experiment update

**EXP-81/82 (Generalization-First Edge Program, SPEC v6) — free length+logprob baseline is the most GENERALIZING correctness readout; geometry fully closed; cross-model cascade revived.** Local, Qwen 1.5B/7B + SmolLM2-1.7B held-out, MATH+BBH. Harness `pathway11_h100/generalization_edge/` (15/15 anchors). length+logprob: T1 LOCO 0.845, T3 cross-scale 0.865, and **held-out T5 ≥ 0.70 on THREE families — SmolLM2-1.7B (near/Llama 0.810), Gemma-2-2b-it (far 0.844), OLMo-2-1B (far 0.838) — confirmed CROSS-ARCHITECTURE (PASS)** — the shippable, zero-activation model-agnostic gate. No hidden-state geometry adds portable value: raw CoE +1.75pp in-domain (hidden-dim-bound), depth-grid n.s./≤0.63 (**H-C**), prompt-cloud 0.694 < prompt-length 0.706 (**H-E refuted**); geometry fully closed. New free Tier-A signal: prompt-length 0.71. Applied: hybrid 1.5B→7B cascade beats FE19 hull +3.38pp (sig); reranking (H-F), early-abort (H-G) refuted; conformal valid in-domain, collapses under shift (H-H); Phase-4 recalibration restores a guarantee zero-shot cross-scale but not cross-domain (task accuracy is the ceiling). Re-pins F-8 readout (→length+logprob), extends F-2 (transfer ceiling; geometry hidden-dim-bound). Full Phase 2/3/4 record `pathway11_h100/generalization_edge/RESULTS_v6_phase234.md`. Far-family T5 closed (Gemma+OLMo-2, H100 follow-on); open: cross-scale arm-2A (GPU-blocked, moot).

---

## Addendum — Phase 2/3/4 executed (EXP-82, 2026-06-11, local 2060)

The forward phases ran locally (per CLAUDE.md "Local only"). Full record:
`pathway11_h100/generalization_edge/RESULTS_v6_phase234.md`.

**Phase 3 (V5-2, T5 held-out) — PASS, cross-architecture.** Generated MATH-500 K=1
greedy on **three** held-out families that played zero role in selection. Free
length+logprob OOF AUROC clears ≥0.70 on every one — **SmolLM2-1.7B** (near/Llama,
acc 21.0%) **0.8097**, **Gemma-2-2b-it** (far/distinct-arch, acc 25.6%) **0.8439**,
**OLMo-2-1B** (far/distinct-arch, acc 21.6%) **0.8382** — so the shippable,
model-agnostic, zero-activation gate generalizes cross-architecture, not just within
the Llama family. *Which* scalar carries is family-dependent (SmolLM2 length-only,
logprob→chance 0.518; Gemma both, 0.783/0.756; OLMo-2 length-led 0.816/0.654), which
is exactly why pinning *both* scalars makes the gate robust. SmolLM2 ran locally
(2060); the two far-family holdouts ran on a RunPod H100 SXM (EXP-82 follow-on,
user-authorized for speed — greedy decoding is deterministic so the AUROCs are
hardware-independent; canonical JSON re-evaluated locally from all three `.npz`).
`results/phase3_t5_free_baseline.json`.

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

**Net (v6 complete):** ship the free length+logprob gate — generalizes
cross-architecture to three unseen families (T5 SmolLM2 0.81 / Gemma 0.84 / OLMo-2
0.84); geometry fully closed; new Tier-A prompt-length signal (0.71); guarantees
transfer zero-shot to a stronger model but not to a harder domain. Far-family T5
closed (Gemma+OLMo-2, H100 follow-on); open: cross-scale arm-2A (GPU-blocked, moot).
H-E refuted (added).
