# PAPER_INDEX.md — paper-to-repo bridge

Every external paper that influenced an experiment or finding. Full textual
lit index with 80+ papers is at `pathway10-papers.md` (has duplication / noise
— treat as unfiltered). This file is the curated set that directly informed
the work.

**Status key.**
- REPLICATED — we ran their method on our data and it worked.
- CONTRADICTED — we ran their method and got the opposite result.
- PARTIALLY CONFIRMED — we reproduced a weakened version.
- CITED ONLY — we read it and it motivated a direction but we didn't test their specific claim.
- TO TEST — listed here because it's on HYPOTHESES.md.

---

## 2306.03341 — Inference-Time Intervention (Li et al., NeurIPS 2023)

**Relevance:** Canonical probe-as-steering-vector paper. The blueprint for
Pathway 10 v2 E1.

**Key claim we tested:** A linear probe fit to predict correctness from
residual-stream activations has a weight vector `w` that, when injected at
inference (`h' = h + α · w_norm`), raises accuracy on the probed task.

**Our result:** **CONTRADICTED for a reasoning task via direction-rotation.**
Our finding: the "correctness axis" *rotates* through generation.
`cos(prefill_DoM, final_DoM) ≈ 0.046`; cos with intermediate positions stays
< 0.2. A final-token-fit ITI vector injected at position 15 is in the wrong
subspace. On TruthfulQA (Li et al.'s benchmark) the question/answer
structure is shorter and this may not manifest. We did not replicate on
TruthfulQA.

**Related experiments:** EXP-006, EXP-007 (superseded Pathway 2 steering);
EXP-038 direction-rotation; HYPOTHESES H-1 (per-position DoM bank would
rescue).

**Delta from their setup:** Qwen-2.5-1.5B vs their LLaMA-7B; MATH reasoning
vs TruthfulQA QA; per-token vs single-position probe.

**Status:** CONTRADICTED (partial — their method appears to assume a static
direction that doesn't hold for long-generation reasoning tasks).

---

## 2509.18116 — Adaptive Layer-wise Steering / ALS

**Relevance:** Per-layer DoM steering explicitly on Qwen-family models.

**Key claim we tested:** Steering direction varies by layer; an adaptive
per-layer injection beats fixed-layer ITI.

**Our result:** NOT TESTED directly — but our direction-rotation finding
(cos changes by position) is a stronger version of ALS's layer-wise story.
ALS doesn't address the temporal axis. If we'd run E1, ALS would be the
natural control.

**Related experiments:** HYPOTHESES H-1 (per-position DoM bank) is a
generalization of ALS from per-layer to per-(layer, position).

**Status:** CITED ONLY.

---

## 2509.12886 — "The LLM Already Knows"

**Relevance:** Prompt-only / pre-generation signals predict downstream
output quality.

**Key claim we tested:** Hidden states at prompt-end encode information
about whether the model will succeed at a task.

**Our result:** **REPLICATED and SHARPENED.** Prefill L19 DoM AUROC on
Qwen-2.5-1.5B = 0.7731 (OOF 5-fold, 243/500 labels). On 7B ≈ 0.876. Our
additional finding: the prefill direction is *orthogonal* to the
final-token direction (cos ≈ 0.046), not just a weaker version of it. This
reframes their "LLM already knows" as a two-circuits claim.

**Related experiments:** EXP-037 (prefill DoM 0.7731 headline), the
"Prefill Knows Best" section of RESEARCH_SUMMARY.

**Delta from their setup:** Different model family; math benchmark vs their
general QA benchmarks; explicitly measures cos(prefill, final) direction
divergence which they don't.

**Status:** REPLICATED.

---

## 2604.18419 — "Knowing When to Quit" / probe-driven abstention

**Relevance:** Uses a linear probe to drive selective prediction / abstention.

**Key claim we tested:** A correctness probe can support a refusal policy
that trades coverage for precision.

**Our result:** **REPLICATED** with a specific headline: at coverage 0.5,
the 1.5B hits 71.6% accuracy on answered (+23 pp over 48.6% unconditional).
This is our cleanest inference-time intervention win.

**Related experiments:** EXP-037 (Section 8 of RESEARCH_SUMMARY).

**Delta from their setup:** We use prefill L19 DoM, they use final-hidden;
we evaluate on MATH-500 reasoning.

**Status:** REPLICATED.

---

## 2604.19740 — Sharpness Dimension / EoS (Tuci et al.)

**Relevance:** Explains why persistent homology on post-training residual
streams should be Gaussian-null. Also motivates the dimensional-breathing
inference-time analog.

**Key claim we tested:** A trained model's hidden states are approximately
Gaussian in their top directions (edge-of-stability dynamics).

**Our result:** **PARTIALLY CONFIRMED** — EXP-026 shows 5 raw PH features
at AUROC 0.690 vs rank-matched empirical-covariance Gaussian null at
0.693. Matched covariance fully explains the PH signal. This is the
Sharpness-Dimension prediction applied to inference-time activations.

**Related experiments:** EXP-026 Pathway 9 Exp 3a v2.

**Open:** Whether inference-time breathing (our PR rise-then-collapse) and
training-time Sharpness Dimension share formal structure is **HYPOTHESES
H-4** — the one theoretical experiment.

**Status:** PARTIALLY CONFIRMED for the PH-at-null prediction; TO TEST for
the breathing/EoS correspondence.

---

## 2506.06609 — Cross-scale stitching (Chen et al.)

**Relevance:** Relevant to the deferred E4 distillation — stitches small-model
layers into a big-model backbone.

**Key claim we tested:** Cross-scale representation transfer is feasible.

**Our result:** NOT TESTED — E4 deferred because the 256-tok artifact
invalidated the 7B-superior-verifier motivator. Remains viable as a
"distill 7B self-prediction into 1.5B" direction if a new motivator
appears.

**Status:** CITED ONLY.

---

## 2410.13640 — Chain-of-Embedding (Wang ICLR 2025)

**Relevance:** Source of the 60-dim trajectory features used in Pathway 8
Exp 2 and Pathway 9 Exp 5.

**Key claim we tested:** Per-layer magnitude/angle changes (60 features)
beat single-layer embeddings for correctness prediction.

**Our result:** **REPLICATED** within-domain (CoE-60 = 0.811 vs ABC-44 =
0.7961 at 256-tok labels). **PARTIALLY CONTRADICTED** by EXP-036 at 1024
tokens: single-layer L19 DoM symmetric cross-benchmark transfer = 0.720 vs
CoE-60's 0.716 (+0.004). Trajectory features are redundant with a single
well-chosen layer.

**Related experiments:** EXP-020, EXP-030, EXP-036.

**Delta from their setup:** Same model family (Qwen); same MATH benchmark;
we re-measured against 1024-tok labels whereas they used a different
default.

**Status:** PARTIALLY CONFIRMED within-domain, CONTRADICTED on the "CoE >
single-layer" cross-domain claim.

---

## 2509.11569 — D²HScore

**Relevance:** Dispersion + drift trajectory features, comparable to CoE.

**Key claim we tested:** Dispersion/drift features predict correctness
better than single-layer baselines.

**Our result:** **PARTIALLY CONFIRMED** within-domain (D2H-lite = 0.806 vs
ABC-44 = 0.7961). But **asymmetric** cross-domain (MATH→BBH 0.740, BBH→MATH
0.529 — see EXP-030).

**Related experiments:** EXP-020, EXP-030.

**Status:** PARTIALLY CONFIRMED.

---

## 2402.03744 — INSIDE / EigenScore (Chen et al., ICLR 2024)

**Relevance:** Eigenvalue-based hallucination detection on hidden-state
covariance. Same family as D²HScore and the spectral arm of our Pathway 9
work. Cited in `pathway10_v1.md` and `pathway9/literature-sweep.md` as the
alternative spectral baseline to D²HScore.

**Key claim:** The log-determinant of the hidden-state covariance across
sampled responses is a training-free hallucination signal.

**Our result:** NOT TESTED head-to-head. Pathway 9 Exp 5 compared CoE
against D²HScore cross-domain but did not include INSIDE/EigenScore.
Remains the cleanest remaining spectral comparison if Pathway 9 is revisited.

**Delta from their setup:** We measure a single-response covariance
structure (DoM + breathing), not across multiple sampled responses.
INSIDE's multi-sample covariance is a different axis.

**Status:** CITED ONLY.

---

## 2509.15735 — EigenTrack

**Relevance:** 2025 spectral statistics of activation covariance for
hallucination detection. Sister paper to D²HScore (2509.11569). Catalogued
in `pathway10-papers.md` under the spectral/eigenvalue cluster.

**Key claim:** Time-series of covariance spectral statistics (dominant
eigenvalue magnitudes, effective rank) carry hallucination signal.

**Our result:** NOT TESTED directly. Our participation-ratio breathing
curve is effectively an effective-rank time-series, so the conceptual
overlap is high — but we have not run the specific statistics they
propose on our cached states.

**Status:** CITED ONLY. If Pathway 11 breathing is ever written up as a
standalone contribution, this paper is a required cite + likely baseline.

---

## 2410.11414 — ReDeEP + AARF

**Relevance:** Head-level mechanistic interpretation + mitigation method.
Would be the natural next step if E1 is revived and goes head-specific.

**Our result:** NOT TESTED.

**Status:** CITED ONLY.

---

## 2505.21772 — CCPS (Contrastive Calibration via Probe Sensitivity)

**Relevance:** Perturbation-based calibration probe. Could robustify our E3
refusal threshold.

**Our result:** NOT TESTED. Our E3 uses the raw probe output without
perturbation analysis.

**Status:** CITED ONLY.

---

## 2406.15927 — Semantic Entropy Probes (SEPs)

**Relevance:** Label-free alternative to the DoM probe. Uses sample-level
semantic entropy as training signal.

**Key claim we tested:** Semantic entropy is recoverable from a single
forward pass via a probe.

**Our result:** NOT TESTED — we used supervised DoM probes throughout.
SEPs would be an ablation for E3 (label-free refusal).

**Status:** CITED ONLY. Placed in HYPOTHESES as a low-priority extension.

---

## 2510.07364 — Brumm et al. (inference-time scaling of CoT)

**Relevance:** Framework paper for how inference-time compute scaling works.
Places our EXP-037 compute-allocation results in context.

**Key claim we tested:** Inference-time compute scales accuracy
sub-linearly, with specific shape by benchmark difficulty.

**Our result:** **PARTIALLY CONFIRMED** — K=8 majority gives +6.4 pp over
K=1 at 8× compute (sub-linear, consistent with Brumm's framework). Our
D-bucket (7.2% of problems where K=8 < K=1) is a novel observation their
framework doesn't predict.

**Related experiments:** EXP-037, EXP-033.

**Status:** PARTIALLY CONFIRMED.

---

## 2504.05419 — Zhang et al. self-verification

**Relevance:** Multi-stage self-verification as an alternative to K=8
majority vote. Relevant for the D-bucket where majority voting hurts.

**Our result:** NOT TESTED. A promising direction for the 36 D-bucket
problems (HYPOTHESES H-7 variant).

**Status:** CITED ONLY.

---

## 2203.11171 — Self-Consistency (Wang et al., ICLR 2023)

**Relevance:** The K=10 self-consistency baseline we tried to beat.

**Key claim we tested:** Majority vote over K samples at T=0.7 boosts
accuracy on reasoning benchmarks.

**Our result:** **REPLICATED** (K=8 gives +6.4 pp on MATH-500 × 1.5B).
**Sharpened** with D-bucket discovery: 7.2% of problems are *hurt* by going
from K=1 to K=8.

**Related experiments:** EXP-033, EXP-037.

**Status:** REPLICATED + EXTENDED.

---

## 2512.19399 — Brain-Grounded Axes (external-coordinate steering)

**Relevance:** Speculative reading for E4 (pathway10_v2 suggestion).

**Our result:** NOT TESTED.

**Status:** CITED ONLY.

---

## Pope et al., "Intrinsic Dimension" — classic TwoNN paper

**Relevance:** TwoNN intrinsic dimension estimator.

**Key claim we tested:** TwoNN ID is a per-cloud summary statistic that
can predict model behavior.

**Our result:** **CONTRADICTED** — EXP-021 (Pathway 8 Exp 3) showed TwoNN
AUROC = 0.407 on MATH-500 correctness prediction. Worse than chance (not
just weaker).

**Status:** CONTRADICTED for correctness prediction specifically; the ID
estimator itself is fine, it just doesn't carry predictive signal for this
task.

---

## ATT Phase 5 (internal reference)

**Relevance:** Pre-existed topo-confidence. Original observation: H0
persistence entropy of last-layer residual stream predicts LLM correctness.

**Our result:** **REPLICATED** as the seed for Pathway 1 CORAL. Later
REINTERPRETED in Pathway 9: the signal is covariance structure, not
topology-specific.

**Status:** PARTIALLY CONFIRMED (signal real; mechanism was misattributed).

---

## Overall lit-to-repo summary

- Supports our headline findings: 2509.12886 (prefill knows best),
  2604.18419 (selective prediction), 2604.19740 (PH at null predicted).
- Contradicted by us: 2306.03341 ITI (on long-generation tasks — direction
  rotates); Pope et al. TwoNN (doesn't predict correctness on our data).
- Partially confirmed: 2410.13640 CoE (within-domain yes, but redundant
  with single-layer DoM); 2509.11569 D²HScore (asymmetric cross-domain).
- Cited only (TO TEST or TO RESPOND TO): 2509.18116 ALS, 2410.11414 ReDeEP,
  2505.21772 CCPS, 2406.15927 SEPs, 2506.06609 cross-scale stitching,
  2504.05419 self-verification, 2512.19399 brain-grounded, 2510.07364 Brumm
  (partially), 2402.03744 INSIDE/EigenScore, 2509.15735 EigenTrack.
