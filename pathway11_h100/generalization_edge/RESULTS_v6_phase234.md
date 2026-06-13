# SPEC v6 — Phase 2/3/4 execution results (EXP-82)

Forward phases of SPEC v6, executed locally on the 2060 Super (per CLAUDE.md
"Local only"). Continues EXP-81 (Phase 0/1/1B). Date 2026-06-11.

Scripts: `phase3_heldout.py`, `phase2_arm2a.py`, `phase4_recalibration.py`.
Results: `results/phase3_t5_free_baseline.json`, `results/phase2_arm2a.json`,
`results/phase4_recalibration.json`; held-out generation cached at
`results/phase3_smollm2.npz`; prompt clouds at
`cache/arm2a_prompt_cloud_qwen1.5b_math.npz`.

## Phase 3 (V5-2) — T5 held-out free-baseline test: **PASS** (headline)

Generated MATH-500 K=1 greedy on **SmolLM2-1.7B-Instruct** (near-family,
Llama-arch; near-family holdout per V2-4) — a model that played **zero role** in
any selection. Acc 21.0% (105/500), mean 426 gen tokens.

Free-baseline OOF 5-fold AUROC on **three** held-out families that played zero
role in any selection (free = length + mean_logprob; all PASS the ≥0.70 bar):

| held-out model | family | acc | **free** | length | logprob | T5≥0.70 |
|---|---|---|---|---|---|---|
| Gemma-2-2b-it | **far** (distinct arch) | 25.6% | **0.8439** | 0.7828 | 0.7558 | ✅ |
| OLMo-2-1B-Instruct | **far** (distinct arch) | 21.6% | **0.8382** | 0.8158 | 0.6535 | ✅ |
| SmolLM2-1.7B-Instruct | near (Llama-arch) | 21.0% | **0.8097** | 0.7813 | 0.5175 | ✅ |

**The free length+logprob recipe clears the ≥0.70 ship bar on every held-out
family, near AND far.** The shippable, model-agnostic, zero-activation
generate-then-abstain F-8 gate generalizes **cross-architecture**, not just within
the Llama family — the load-bearing far-family number is now measured, not open.
Notable: *which* scalar carries is family-dependent — on SmolLM2 length carries
and logprob is near-chance; on Gemma **both** carry (0.783 / 0.756); on OLMo-2
length leads (0.816). Pinning *both* scalars is what makes the gate robust across
these flips.

**Provenance:** SmolLM2 generated locally (2060, EXP-82). The two far-family
holdouts were generated on a RunPod H100 SXM (EXP-82 follow-on, 2026-06-11) — a
deliberate, user-authorized exception to the local-only norm for speed; greedy
decoding is deterministic so the AUROCs are hardware-independent, and the canonical
`phase3_t5_free_baseline.json` was re-evaluated locally from all three pulled
`.npz` caches.

## Phase 2 arm-2A (V5-3, H-E) — prompt-token-cloud: **REFUTED**

Prefill pass over the 500 MATH prompts on Qwen-1.5B; L19 prompt-token cloud →
top-20 covariance log-eigvals, gated against **prompt-length** (the exogenous
difficulty prior, previously unmeasured — V3-5).

| feature (in-domain OOF) | AUROC |
|---|---|
| prompt_length_only | **0.7056** |
| prompt_cloud_spectrum_only | 0.6938 |
| cloud + length | 0.6960 |

Gate Δ(cloud+len vs len) = **−0.0096, p=0.42** → **prompt-cloud does NOT add over
prompt-length. H-E refuted.** The prompt-cloud top eigenvalue is ~collinear with
prompt length (r = −0.89) — the same length-confound that killed the generated-
token cloud (V3-1), now on the prompt side. **Geometry is now fully closed:**
every arm (token-cloud spectrum, layer-profile depth-grid, prompt-cloud) refuted
as portable value-add over free signals.

**Positive byproduct:** prompt-length alone is a **0.71 AUROC pre-flight (Tier-A)
correctness signal** — exogenous, pre-generation, model-agnostic, zero generation
cost. That, not geometry, is the deployable pre-flight gate (weaker than the
post-gen 0.85 but genuinely Tier-A).

*Local-compute limit:* cross-scale arm-2A (Qwen-7B prompt clouds) is unrunnable —
7B bf16 (~15GB) exceeds the 2060's 8GB. The in-domain 1.5B gate is decisive for
H-E (cloud loses even in-domain), so the cross-scale cell would not rescue it.

## Phase 4 (V5-5) — per-domain recalibration: nuanced (sharpens the premise)

Freeze the source-fit extractor; re-fit only a light head on k ∈ {0,8,16,32,64}
target labels (R=200 resamples), large cells only (BBH n=750, Qwen-7B n=500).

**Cross-scale → stronger model (1.5B→7B), free baseline:** ranking ports (AUROC
k0 0.893 → ceiling 0.899). The conformal guarantee **transfers zero-shot** (k=0:
ε=0.2 → 60% coverage, validity 1.0) because a threshold calibrated on the harder
source is conservative on the easier target. Tiny target sets (k≤16) *hurt*
(CP-upper small-sample penalty → infeasible); recalibrate only past k=32
(k=64: ε=0.2 → 29% cov valid 1.0).

**Cross-domain → harder task (MATH→BBH):** ranking ports (AUROC 0.785,
logprob-carried; BBH `n_gen_tokens` degenerate so length adds nothing), but **no
light-head recalibration up to k=64 buys a valid certificate even at ε=0.3.**
BBH's 24.1% base rate (→ ~38% selective@50%) caps answered-accuracy below the
bound. Honest lesson: recalibration restores ranking-based abstention but **cannot
manufacture a tight guarantee on a hard domain** — target-task accuracy is the
ceiling, not calibration.

## Net (v6 complete, local-constrained)

- **Ship:** the free length+logprob gate — generalizes **cross-architecture** to
  three unseen families (T5: SmolLM2 0.81 near, Gemma 0.84 far, OLMo-2 0.84 far),
  needs no activations. Pin both scalars (length/logprob carry on different models).
- **Geometry: fully closed** (H-C, H-E, H-A all refuted) — no portable activation
  feature beats free signals anywhere.
- **New Tier-A signal:** prompt-length 0.71 (pre-generation, free).
- **Guarantees:** transfer zero-shot to a stronger model; require — but are not
  rescued by — recalibration on a harder domain.
- **Closed:** far-family T5 — Gemma + OLMo-2 both PASS (0.84/0.84), the gate is
  confirmed cross-architecture (EXP-82 follow-on, H100).
- **Open:** cross-scale arm-2A — GPU-blocked locally (moot, cloud loses in-domain).
