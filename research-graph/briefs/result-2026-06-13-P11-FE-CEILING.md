# Result brief — "Raise the Ceiling" (SPEC v8), EXP-90/91

SPEC v8 asked whether any lever *untouched* by v6/v7 could move the matched-cost
MATH-500 ceiling above the free-gate cascade's operating point (0.648 @ budget
2220.7 token-FLOPs/problem; oracle 0.758). Four levers, none of them a confidence
*probe*: the escalation-target identity (H-N), a family-diverse second target
(H-O), the base-model identity (H-P/H-Q), and — the centerpiece — the base-model
*weights* via confidence-curated distillation (H-R). One RunPod H100 SXM session
(~6 GPU-hrs, user-authorized $30-capped key); all grading/adjudication local CPU.
Selection firewall held: single-shot per arm; Gate G1 froze the entire Phase-2
recipe before any Phase-2 GPU minute; one eval per SFT checkpoint.

## Headline

**The ceiling didn't move by curating data — it moves by swapping the base.**
The biggest result is the cheapest: the off-the-shelf **Qwen2.5-Math-1.5B-Instruct**
scores **0.740 standalone at ~¼ the budget** (528.6 vs 2220.7 token-FLOPs),
beating the entire budget-matched free-gate cascade by **+9.2pp** (H-P CONFIRMED,
p≈0). Escalating to **Qwen2.5-Math-7B-Instruct** instead of the generic 7B lifts
the cascade **+4.2pp** (0.648→0.690, p=0.0025) and rescues **41 of the 121**
previously-unrescuable base failures (oracle 0.758→0.800; H-N CONFIRMED). The free
gate (length + mean-logprob) **generalizes to every new base** (OOF AUROC 0.863 /
0.950 / 0.936; H-Q PASS). But the centerpiece **failed**: a zero-label confidence
filter curating 4000 Math-7B distillation traces into the generic 1.5B does **not**
beat unfiltered at matched N (gate arm 0.494 < unfiltered 0.500 < perfect-label
skyline 0.504; **H-R REFUTED**) — and the gate arm even sits *below* its own
length-matched control (0.502), so at selection the gate behaves as a noisy length
proxy, not a quality signal. Worse, MATH-only SFT **catastrophically forgets BBH**
on every arm (best 0.208 < un-adapted base 0.267). **Curation is a free-rider;
the lever is the base, not the data engine.**

## What was run

- **Phase 1 (EXP-90)** — frontier raisers, local-graded single-shot:
  - **H-N** target swap: full free-gate cascade re-run with Qwen2.5-Math-7B as the
    escalation target (mask is target-independent; only the escalated answers change).
  - **H-O** second target: Mathstral-7B-v0.1 disjoint-rescue count vs the 121.
  - **H-P/H-Q** base swap: Qwen2.5-Math-1.5B-Instruct and DeepSeek-R1-Distill-Qwen-1.5B
    standalone MATH-500 greedy + free-gate OOF AUROC on each. R1-Distill greedy
    degenerated (36% hit-max unparseable) → pre-registered one-shot T=0.6 fallback.
- **Phase 2 (EXP-91)** — confidence-curated distillation, H-R, four matched-N arms:
  teacher = Math-7B greedy on MATH-train n=4000 (train-acc 0.766); free gate fit
  once on the Math-7B MATH-500 dev cell at the precision-0.90 op-point, applied
  frozen to the 4000 train traces (zero train labels touched, H-L3 pattern); arms
  (a) GT-correct skyline, (b) gate-keep zero-label, (c) unfiltered random,
  (d) length-matched control — all subsampled to N=3065. LoRA SFT ×4 on generic
  Qwen2.5-1.5B-Instruct (r16 α32, 2 ep, eff-batch 32, seed 0, final ckpt), one
  greedy MATH-500 + BBH-750 eval each, plus the un-adapted base reference.

## Hypothesis scoreboard

| ID | Statement | Verdict |
|---|---|---|
| H-N | Math-7B is a better escalation target than generic 7B | **CONFIRMED** (cascade +4.20pp, 0.648→0.690, p=0.0025; 41/121 rescued; oracle 0.758→0.800) |
| H-O | Mathstral-7B adds a family-diverse second rescue tier | **INCONCLUSIVE** (15/121 disjoint; confirm≥25, refute<12.1) |
| H-P | An off-the-shelf small base beats the cascade at matched budget | **CONFIRMED** for Math-1.5B (0.740 @ cost 529, +9.2pp, p≈0); **REFUTED** for R1-Distill (greedy 0.634 / T=0.6 0.678, cost 2742 > budget) |
| H-Q | The free gate works on the new bases | **CONFIRMED** (OOF AUROC Math-1.5B 0.863, R1-Distill 0.950/0.936; all PASS ≥0.70) |
| H-R | A zero-label confidence filter curates distillation data, beating unfiltered at matched N | **REFUTED** (gate 0.494 < unfiltered 0.500 < skyline 0.504; gate even < length-control 0.502; winning-arm BBH guardrail FAILs) |

## H-R in one paragraph

At matched N=3065, the zero-label gate filter keeps a set whose train-truth
precision is 0.855 / recall 0.992 — i.e. it *does* preferentially keep correct
traces — yet the resulting SFT'd model (0.494 MATH-500) does **not** beat the
unfiltered model (0.500) or its own length-matched control (0.502), and barely
trails the perfect-ground-truth skyline (0.504). The decision rule fires REFUTE
(b−c = −0.006, below the +0.5pp floor; beats-volume, beats-length, and
beats-base-by-3pp all fail). Two mechanisms: (1) at this recipe and scale,
distilling Math-7B traces into the generic 1.5B barely helps at all — even
*perfect* label filtering buys only +0.4pp over unfiltered — so there is no
curation increment for any filter to capture; (2) the gate's length feature makes
it a step-length selector, and the length-matched control (d) reproduces (b)
almost exactly, confirming the "Step Length Confounding" risk pre-registered as
the reason arm (d) exists. Independently, MATH-only SFT collapses BBH on every arm
(a 0.109 / b 0.148 / c 0.208 / d 0.139, all < un-adapted base 0.267), so no arm
clears the no-catastrophic-forgetting guardrail regardless of MATH. The honest
recipe: **don't curate distillation data with the gate — use all traces, or
better, skip distillation and swap to the off-the-shelf domain base (H-P).**

## Deviations

- **D-1**: all GPU work on RunPod H100 SXM (no-local-GPU directive); deterministic
  greedy gen → hardware-independent, JSON re-graded locally with the pinned grader.
- **D-2** (R1-Distill greedy degeneracy): greedy R1-Distill hit 4096-max-unparseable
  on 36% of MATH-500; the pre-registered one-shot T=0.6/seed-9999 fallback was run
  (0.634→0.678) — still over budget and below cascade, H-P verdict unchanged. The
  high hit-max persists at T=0.6 (32%), so the length is intrinsic, not a decoding
  artifact; this is exactly what the free gate reads (length-coef −3.08).

## FE

```yaml
fe_id: P11-FE-CEILING
status: COMPLETED
outcome: "SPEC v8 'Raise the Ceiling' executed end-to-end (EXP-90/91, ~6 H100 hrs): four non-probe levers tested against the free-gate cascade's matched-cost MATH-500 ceiling (0.648 @ budget 2220.7). H-N CONFIRMED: escalating to Qwen2.5-Math-7B-Instruct instead of generic 7B lifts the cascade +4.20pp (0.648->0.690, p=0.0025), rescues 41/121 previously-unrescuable base failures, oracle 0.758->0.800 -> new pinned escalation target. H-O INCONCLUSIVE: Mathstral-7B adds 15/121 disjoint rescues (below the 25 confirm bar). H-P CONFIRMED: off-the-shelf Qwen2.5-Math-1.5B-Instruct scores 0.740 standalone at ~1/4 budget (cost 529), beating the whole budget-matched cascade by +9.2pp (p~0) -> the real product lever is the base, not introspection; REFUTED for DeepSeek-R1-Distill-1.5B (greedy 0.634 / T=0.6 0.678 @ cost 2742, over budget, reasoning length intrinsic). H-Q CONFIRMED: the free gate (length+logprob) generalizes to all new bases (OOF AUROC 0.863 Math-1.5B, 0.950/0.936 R1-Distill). H-R REFUTED (the centerpiece): a zero-label confidence filter curating 4000 Math-7B distillation traces into the generic 1.5B does NOT beat unfiltered at matched N=3065 (gate arm 0.494 < unfiltered 0.500 < perfect-label skyline 0.504; gate arm even < length-matched control 0.502 -> gate is a noisy step-length proxy at selection, not a quality signal). Even perfect label filtering buys only +0.4pp over unfiltered, so there is no curation increment to capture; and MATH-only SFT catastrophically forgets BBH on every arm (best 0.208 < base 0.267), failing the no-forgetting guardrail. GENERAL CLOSURE: the matched-cost ceiling is raised by swapping the base/target (free, off-the-shelf), not by curating data with the confidence stack; confidence-curated distillation is a free-rider. Honest recipe: use the off-the-shelf domain base + free-gate cascade on top, escalation target = Math-7B."
result_json: pathway11_h100/generalization_edge/results/v8_phase2_distill.json
result_json_all:
  - pathway11_h100/generalization_edge/results/v8_phase1_frontier.json
  - pathway11_h100/generalization_edge/results/v8_filter_arms_report.json
  - pathway11_h100/generalization_edge/results/v8_phase2_distill.json
executes: ["P11-FE5"]
answers: ["P11-FE1214"]
refutes: ["H-R"]
confirms: ["H-N", "H-P", "H-Q"]
confirms_premise: [free-baseline-strongest-readout]
relies_on: [escalation-beats-introspection]
would_update: ["F-2", "F-8"]
```

## EXPERIMENT_LOG entry

## EXP-85: SPEC v8 "Raise the Ceiling" — frontier raisers + confidence-curated distillation (spec-internal phase labels EXP-90/91)

**Date:** 2026-06-13. **Compute:** local CPU + 1 RunPod H100 SXM session
(~6 GPU-hrs; D-1 zero local GPU, $30-capped key). Qwen2.5 1.5B/7B + Math-1.5B/7B,
DeepSeek-R1-Distill-Qwen-1.5B, Mathstral-7B; MATH-500 + MATH-train n=4000 + BBH-750.

**Four non-probe levers vs the free-gate cascade's matched-cost ceiling
(0.648 @ budget 2220.7).** H-N CONFIRMED: Math-7B escalation target lifts the
cascade +4.20pp (0.648→0.690, p=0.0025), 41/121 rescued, oracle 0.758→0.800.
H-O INCONCLUSIVE (Mathstral 15/121). **H-P CONFIRMED — the headline lever:**
off-the-shelf Qwen2.5-Math-1.5B-Instruct 0.740 standalone at ~¼ budget beats the
budget-matched cascade by +9.2pp (p≈0); REFUTED for R1-Distill (greedy 0.634 /
T=0.6 0.678 @ cost 2742, intrinsic length). H-Q CONFIRMED: free gate generalizes
to all bases (0.863 / 0.950 / 0.936). **H-R REFUTED (centerpiece):** zero-label
gate-curated distillation (4000 Math-7B traces → generic 1.5B, matched N=3065)
gate arm 0.494 < unfiltered 0.500 < perfect-label skyline 0.504, and < its own
length control 0.502 — curation is a free-rider, the gate is a step-length proxy
at selection; MATH-only SFT catastrophically forgets BBH (all arms < base 0.267).
Selection firewall held (G1 froze the recipe before any Phase-2 GPU minute; one
eval per checkpoint). Claims 245→264 internal PASS (+19 edge-v8-*).

## STATE.md last-experiment update

**EXP-85 (SPEC v8 "Raise the Ceiling") — the ceiling moves by swapping the base,
not by curating data.** Four levers untouched by v6/v7 tested against the
free-gate cascade's matched-cost MATH-500 ceiling. **Pin the base/target, not a
probe:** off-the-shelf Qwen2.5-Math-1.5B-Instruct scores 0.740 at ¼ budget (+9.2pp
over the whole cascade, H-P); escalating to Qwen2.5-Math-7B instead of generic 7B
lifts the cascade +4.20pp and rescues 41/121 (H-N, new pinned target). The free
gate generalizes to every new base (0.86/0.95/0.94, H-Q). **The centerpiece
failed:** zero-label confidence-curated distillation does NOT beat unfiltered at
matched N (gate 0.494 < unfiltered 0.500 < skyline 0.504, < length-control 0.502;
H-R refuted) and MATH-only SFT catastrophically forgets BBH (all arms < base
0.267) — curation is a free-rider, the gate is a step-length proxy at selection.
Honest recipe: off-the-shelf domain base + free-gate cascade on top, escalation
target = Math-7B. Claims 245→264 internal PASS (+19 edge-v8-*). Artifacts:
`generalization_edge/results/v8_*.json`, SPEC_v8.md (G1 freeze).

## Sources

- `pathway11_h100/generalization_edge/SPEC_v8.md` (spec + G1 freeze block)
- `pathway11_h100/generalization_edge/results/v8_phase1_frontier.json` (H-N/H-O/H-P/H-Q)
- `pathway11_h100/generalization_edge/results/v8_filter_arms_report.json` (teacher, gate-vs-truth, N)
- `pathway11_h100/generalization_edge/results/v8_phase2_distill.json` (H-R)
