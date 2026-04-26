# Next Experiments — Priority Queue

_Auto-generated from the research graph. Do not edit directly._
_Regenerate: `cd research-graph && python generate_next_experiments.py`_
_Generated: 2026-04-26 18:13 UTC_

---

## CRITICAL (ROI 9-10) — Do these first

### P8-FE1 (P8) — [ROI: 10, READY, CRITICAL]

**What:** Re-extract CoE-60 features using 1024-token generations instead of the original 256-token truncated runs. Refit the logistic regression classifier and report AUROC under the corrected label distribution (243/257 instead of 104/396).

**Why:** F-6's headline 0.811 AUROC was measured on truncated labels. The corrected baseline accuracy is 48.6% not 20.8%, which means the class balance is ~50/50 instead of ~20/80. CoE AUROC may go up (more signal) or down (less separation) — we don't know until we test.

**Cost:** 2h H100 + 1h CPU
**Blocked by:** H100 pod for re-extraction
**Depends on:** F-13, F-6
**Would update:** F-8, F-6
**Trigger condition:** Already triggered — F-13 invalidated the labels.

### P9-FE1 (P9) — [ROI: 10, READY, CRITICAL]

**What:** Run the full 5-experiment battery (length deconfound, orthogonality, Gaussian null, XGBoost-vs-LR, cross-domain transfer) using 1024-token corrected labels. Report which findings survive and which change.

**Why:** Every P9 number was computed on the wrong labels. This is the single most important re-validation in the project.

**Cost:** 2h H100 + 1 day CPU
**Blocked by:** H100 pod for fresh feature extraction
**Depends on:** F-13
**Would update:** F-10, F-9, F-8, F-7, F-6
**Trigger condition:** Already triggered — F-13.

### P10-FE1 (P10) — [ROI: 9, TRIGGERED, CRITICAL]

**What:** Run E1 steering experiment with three parallel arms: (a) probe-weight vector, (b) mass-mean (difference-of-means) vector, (c) trained per-layer bias (Sinii-style, 2505.18706). Sweep alpha, report accuracy on MATH-500 holdout for each. Use SEAL's thought-type vector (2504.07986) as a fourth comparison if feasible.

**Why:** The literature strongly predicts mass-mean >= probe-weight, and RL-trained biases exceed both (AxBench ranking, AdaRAS, Sinii). Running all three arms on the same model and benchmark gives a definitive comparison that doesn't exist in the literature.

**Cost:** 1 day H100
**Blocked by:** H100 pod
**Triggered by:** [AxBench: Steering Benchmark](https://arxiv.org/abs/2501.17148), [SEAL: Steerable Reasoning Calibration](https://arxiv.org/abs/2504.07986), [Bias-Only Steering (Sinii)](https://arxiv.org/abs/2505.18706), [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)
**Depends on:** F-4, F-3
**Would update:** F-4, F-3
**Trigger condition:** Sinii and SEAL published with results on same model family.

### P11-FE5 (P11) — [ROI: 9, TRIGGERED, HIGH]

**What:** Use DeepSeek-R1-Distill-Qwen-1.5B (same architecture as project's base model, but distilled from R1 with long-CoT training) as a comparison model. Extract L19 activations on MATH-500. Compare: (a) Does breathing still occur? (b) Is the asymmetric collapse stronger? (c) Does the prefill signal improve (R1-Distill has 83.9% accuracy vs base 48.6%)? (d) Does the DoM direction transfer between base and R1-Distill?

**Why:** R1-Distill is the same architecture with massively improved reasoning via SFT on R1 traces. Comparing base vs R1-Distill isolates the effect of reasoning training on the hidden-state geometry. If breathing/collapse are stronger in R1-Distill, they're tied to reasoning capability, not just architecture.

**Cost:** 4h H100 + 2h CPU
**Blocked by:** H100 pod
**Triggered by:** [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
**Depends on:** F-4, F-2, F-1
**Would update:** F-4, F-2, F-1
**Trigger condition:** R1-Distill-Qwen-1.5B publicly available.

### P3-FE1 (P3) — [ROI: 9, TRIGGERED, CRITICAL]

**What:** Re-run P3's steering-for-accuracy using adaptive per-token steering (PID or STU-PID) instead of fixed-vector injection. Use the proper L19 DoM direction, apply PID controller across generation positions.

**Why:** P3 failed because the steering direction rotates during generation (F-3). PID steering (2510.04309) and STU-PID (2506.18831) are designed to handle exactly this — they accumulate error and damp overshoot. STU-PID was tested on the project's exact model (DeepSeek-R1-Distill-Qwen-1.5B).

**Cost:** 1 day H100
**Triggered by:** [STU-PID Steering](https://arxiv.org/abs/2506.18831), [PID Steering](https://arxiv.org/abs/2510.04309)
**Depends on:** F-4, F-3
**Would update:** F-3
**Trigger condition:** PID Steering and STU-PID papers published.

### P11-FE3 (P11) — [ROI: 9, READY, HIGH]

**What:** Compute the causal effect of prefill geometry on downstream correctness using activation patching. Swap prefill activations between a correct-predicted and incorrect-predicted problem pair at L19. If swapping flips the outcome, the prefill signal is causal. If not, it's correlational.

**Why:** All findings so far are correlational probes. Kudo et al. (2412.01113) used causal interventions to confirm models compute sub-answers during CoT. The same methodology applied to prefill would determine whether the 'can I solve this' signal actually drives behavior or merely correlates with it.

**Cost:** 4h H100
**Blocked by:** H100 pod for generation with interventions
**Triggered by:** [How to Use Activation Patching](https://arxiv.org/abs/2404.15255), [LLMs Faithfully Compute During CoT (Kudo)](https://arxiv.org/abs/2412.01113)
**Depends on:** F-2
**Would update:** F-2
**Trigger condition:** Methodological — activation patching infra exists.

---

## HIGH (ROI 7-8)

### P10-FE2 (P10) — [ROI: 8, TRIGGERED, HIGH]

**What:** Implement E3 refusal with conformal prediction wrapper (Mohri & Hashimoto, 2402.10978). Instead of a fixed threshold tau, use conformal calibration to provide distribution-free coverage guarantees. Report: at 90% coverage, what accuracy? At 95% precision on answered, what coverage?

**Why:** F-11's 71.6%@50% coverage is strong but uses a fixed threshold. Conformal prediction gives the same operating point with a statistical guarantee that holds on future data. This is the publishable version of E3.

**Cost:** 4h CPU
**Triggered by:** [Conformal Prediction via Internal Representations](https://arxiv.org/abs/2604.16217), [Conformal Factuality (Mohri & Hashimoto)](https://arxiv.org/abs/2402.10978)
**Depends on:** F-2, F-11
**Would update:** F-11
**Trigger condition:** Conformal factuality paper published with MATH numbers.

### P10-FE3 (P10) — [ROI: 8, TRIGGERED, HIGH]

**What:** Implement E2 (confidence-gated compute) using Damani et al.'s 'Learning How Hard to Think' framework (2410.04707). Train a lightweight predictor on prefill embeddings to estimate marginal reward of additional samples. Compare against the project's monotonic/non-monotonic gating baselines.

**Why:** F-14 showed monotonic gating fails. Damani's framework is purpose-built for this — it trains a meta-predictor of when more compute helps, which is exactly the bucket-B recovery problem.

**Cost:** 1 day CPU + 4h H100 for sample generation
**Triggered by:** [Learning How Hard to Think (Damani)](https://arxiv.org/abs/2410.04707)
**Depends on:** F-2, F-14
**Would update:** F-14
**Trigger condition:** Damani et al. published.

### P11-FE2 (P11) — [ROI: 8, TRIGGERED, HIGH]

**What:** Test whether the prefill direction encodes training-set familiarity or problem structure. Use MATH-500 difficulty level metadata (5 levels in the original dataset). Compute prefill DoM score vs difficulty level. If prefill tracks difficulty perfectly, it's familiarity. If orthogonal to difficulty but still predicts correctness, it's something more interesting.

**Why:** F-2's strongest counterargument is that prefill encodes familiarity, not structure. This test directly addresses it. Lugoloobi & Russell (2510.18147) showed Qwen2.5-Math-1.5B encodes human difficulty with rho=0.88 — is our prefill DoM the same signal or a different one?

**Cost:** 1h CPU
**Triggered by:** [LLMs Encode Problem Difficulty](https://arxiv.org/abs/2510.18147)
**Depends on:** F-2
**Would update:** F-2
**Trigger condition:** Lugoloobi & Russell published on exact model.

### P4-FE2 (P4) — [ROI: 8, TRIGGERED, HIGH]

**What:** Replace TwoNN global ID with GeoMLE local intrinsic dimension estimation per-problem. Compute local LID at L19 for each MATH-500 problem's activation neighborhood (k=20). Test as correctness predictor.

**Why:** F-10 showed global TwoNN has near-zero predictive power (AUROC 0.407). But Yin et al. (2402.18048) showed LOCAL LID predicts truthfulness with 5-8 AUROC points above baselines using GeoMLE. The discrepancy is likely global-vs-local — local LID may recover the signal TwoNN misses.

**Cost:** 4h CPU on cached activations
**Triggered by:** [Truthfulness via Local ID](https://arxiv.org/abs/2402.18048)
**Depends on:** F-10
**Would update:** F-10
**Trigger condition:** Already triggered — Yin et al. published with code.

### P8-FE2 (P8) — [ROI: 8, TRIGGERED, HIGH]

**What:** Extract CoE features at step/chunk boundaries during chain-of-thought (every 50 tokens) rather than only at the final token. Train a step-level correctness probe. Compare against Zhang et al.'s chunk-boundary probing (2504.05419) which achieved >0.9 AUROC on AIME.

**Why:** Current CoE features are final-token only. Zhang et al. showed mid-trajectory probes can beat final-token probes substantially. Step-level CoE also feeds directly into E2 (confidence-gated compute) and E3 (refusal) — you need temporal resolution to gate during generation, not after.

**Cost:** 4h H100 + 2h CPU
**Blocked by:** H100 pod
**Triggered by:** [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419)
**Depends on:** F-2, F-6
**Would update:** F-2, F-6
**Trigger condition:** Zhang et al. published with strong results.

### P11-FE1 (P11) — [ROI: 8, READY, HIGH]

**What:** Run breathing analysis (temporal PR curve) on BBH subsets using Stage 4a per-token activations. Does breathing occur on non-math tasks? Does the correct/incorrect PR ratio hold on web_of_lies (54% accuracy, near-balanced)?

**Why:** F-1 was measured on MATH-500 only. BBH data is already cached on the pod. If breathing occurs on BBH, it's a reasoning phenomenon. If not, it's math-specific.

**Cost:** 2h CPU
**Depends on:** F-1
**Would update:** F-1
**Trigger condition:** Data already cached.

### P7-FE1 (P7) — [ROI: 7, TRIGGERED, HIGH]

**What:** Run zigzag persistence on the cached pathway 8 layer-wise activations using the Dey-Hou fast zigzag implementation (github.com/taohou01/fzz) or the RitaSciencePark/topo_llm pipeline. Compare zigzag PH features against the standard Rips PH that hit the Gaussian null.

**Why:** F-7 killed standard Rips PH. But zigzag PH captures transition dynamics between layers — the exact signal that CoE captures geometrically. If zigzag PH survives the Gaussian null where Rips didn't, it means PH *can* work on LLM activations but only when the filtration respects layer ordering.

**Cost:** 3 days CPU
**Triggered by:** [Persistent Topological Features in LLMs](https://arxiv.org/abs/2410.11042), [Fast Zigzag Persistence (Dey-Hou)](https://arxiv.org/abs/2103.07353)
**Depends on:** F-8, F-7
**Would update:** F-8, F-7
**Trigger condition:** Fast zigzag implementations now available. Dionysus2 install blocker removed.

### P11-FE4 (P11) — [ROI: 7, READY, MEDIUM]

**What:** Characterize D-bucket (F-12) problems by mid-generation features. Extract L19 activations at token positions 50, 100, 150 for all 500 MATH-500 problems. Train a classifier to distinguish bucket D (K=1 right, K=8 wrong) from bucket A (K=1 right, K=8 right). If mid-generation features detect D-bucket, the signal appears during reasoning, not before.

**Why:** F-12 showed D-bucket is undetectable from prefill features. But D-bucket is about self-consistency failure during generation — the model starts right and talks itself into a wrong answer. The signal should appear when the reasoning trajectory diverges.

**Cost:** 4h H100 + 2h CPU
**Blocked by:** Per-token activations from K=8 runs (may need pod)
**Triggered by:** [Soft Self-Consistency](https://arxiv.org/abs/2402.13212)
**Depends on:** F-3, F-12
**Would update:** F-12
**Trigger condition:** Soft Self-Consistency (2402.13212) documented SC failure modes on diverse answers — same phenomenon.

### P2-FE1 (P2) — [ROI: 7, READY, HIGH]

**What:** Re-derive the steering vector from scratch using 1024-token labels and proper DoM at L19. Compare against the original P2 cached vector (cos=0.05 with current L19 DoM).

**Why:** The P2 steering vector was derived with truncated labels. F-3 showed it's unusable (cos=0.05). A properly derived vector may still fail due to direction rotation (F-3), but at least it tests steering with a non-stale direction.

**Cost:** 2h CPU + 4h H100 for generation with intervention
**Triggered by:** [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)
**Depends on:** F-13, F-3
**Would update:** F-3
**Trigger condition:** Already triggered — F-3 and F-13 together show the old vector was doubly wrong (wrong labels + wrong subspace).

### P9-FE3 (P9) — [ROI: 7, READY, HIGH]

**What:** Run cross-domain transfer experiment (P9-E5) on individual BBH subsets instead of pooled. Report per-subset AUROC for CoE, D2H, and DoM. Identify which subset drives the D2H asymmetry.

**Why:** The pooled BBH transfer (0.720/0.712 for CoE) may be driven by one subset. web_of_lies has 54% accuracy (near-balanced), tracking_shuffled_objects has 12.4%, logical_deduction has 11.2%. The subset heterogeneity is extreme and pooling masks it.

**Cost:** 2h CPU on cached features
**Depends on:** F-6
**Would update:** F-6
**Trigger condition:** Already identified as open question in pathway 9.

---

## MEDIUM (ROI 5-6)

### P10-FE4 (P10) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Implement E4 distillation using TIP's Q3-token-focused loss (2604.14084). Fine-tune Qwen2.5-1.5B with auxiliary loss matching its L19 activations to 7B's L20 activations, but only at Q3 tokens (low student entropy, high teacher-student divergence). Compare against uniform distillation.

**Why:** TIP showed Q3 tokens carry disproportionate corrective signal on the exact model pair (Qwen2.5-14B->1.5B). Combining with the project's finding that L19 carries the correctness signal gives a focused distillation target: match the correctness direction at positions where the student is overconfidently wrong.

**Cost:** 2+ weeks, multi-day H100
**Blocked by:** Multi-day H100 allocation
**Triggered by:** [TIP: Token Importance in OPD](https://arxiv.org/abs/2604.14084)
**Depends on:** F-5, F-4
**Would update:** F-5
**Trigger condition:** TIP published with code and Qwen results.

### P6-FE1 (P6) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Test cross-scale probe transfer using affine alignment (Bello et al. LRT, 2506.00653) instead of raw transfer. Train a lightweight affine map from 1.5B L19 activations to 7B L20 activations on a shared problem set, then test whether the 7B's DoM direction transfers to 1.5B through the map.

**Why:** F-5 showed raw cross-scale transfer is flat (0.717 vs self 0.719). But LRT (2506.00653) showed affine maps between model sizes preserve steering-relevant structure. The question is whether an aligned 7B correctness direction transfers better than unaligned.

**Cost:** 4h H100 + 2h CPU
**Blocked by:** H100 pod for 7B activations
**Triggered by:** [Linear Representation Transferability (LRT)](https://arxiv.org/abs/2506.00653)
**Depends on:** F-5
**Would update:** F-5
**Trigger condition:** LRT paper published with theoretical framework.

### P7-FE2 (P7) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Compute PH on attention graphs (token-token attention matrices per layer) instead of residual-stream point clouds. Use the TOHA/HalluZig approach: build a weighted graph from attention maps, compute PH of the graph filtration.

**Why:** Every successful PH-on-LLM paper in the literature (Kushnareva 2021, Bazarova 2025) works on attention graphs, not residual-stream point clouds. The project tested only the latter. PH may be valid for LLM correctness detection — just on the wrong object.

**Cost:** 1 week CPU (attention extraction + PH computation)
**Triggered by:** [HalluZig: Zigzag PH on Attention](https://arxiv.org/abs/2601.01552), [TOHA: Topological Divergence on Attention](https://arxiv.org/abs/2504.10063)
**Depends on:** F-7
**Would update:** F-7
**Trigger condition:** TOHA (2504.10063) and HalluZig (2601.01552) published showing attention-graph PH works.

### P5-FE1 (P5) — [ROI: 6, READY, MEDIUM]

**What:** Run dimensional breathing analysis (temporal PR curve) on Gemma-2-2B and Mistral-7B on the same MATH-500 problems. Confirm breathing is architecture-universal beyond the Qwen/Phi/Llama families already tested.

**Why:** F-1 replicated across Phi-3-mini and Llama-3.2-1B but all three are decoder-only with similar training recipes. Gemma-2 uses a different attention variant and Mistral uses sliding window attention — testing these extends the universality claim substantially.

**Cost:** 4h H100
**Blocked by:** H100 pod time
**Depends on:** F-1
**Would update:** F-1
**Trigger condition:** Whenever H100 pod is next active for any other experiment.

### P8-FE3 (P8) — [ROI: 5, TRIGGERED, MEDIUM]

**What:** Compute representation dispersion (average pairwise cosine distance) per layer on the cached activations. Correlate with CoE magnitude features and with correctness. Test whether dispersion alone (1 feature per layer, 28 total) matches D2H-lite's 58-dim AUROC.

**Why:** Li & Li (2506.24106) showed dispersion strongly predicts perplexity and downstream accuracy across Qwen family. This is exactly what D2H's intra-layer dispersion measures but in a simpler formulation. If 28-dim dispersion matches 58-dim D2H, the extra complexity is unnecessary.

**Cost:** 2h CPU on cached activations
**Triggered by:** [Representation Dispersion Predictive Power](https://arxiv.org/abs/2506.24106)
**Depends on:** F-6
**Trigger condition:** Dispersion paper published with Qwen validation.

### P1-FE1 (P1) — [ROI: 5, READY, MEDIUM]

**What:** Revisit initial geometry observations with proper 1024-token labels and participation ratio instead of ad-hoc metrics. Re-extract on the same problems with current infrastructure.

**Why:** P1 observations predate the truncation correction (F-13). Any early geometry finding may have been measuring truncation artifacts. A clean re-run with current methodology would either validate or retire P1 claims.

**Cost:** 4h CPU on cached activations
**Depends on:** F-13
**Would update:** F-1
**Trigger condition:** Already triggered — F-13 (truncation artifact) invalidated the label distribution P1 used.

### P4-FE1 (P4) — [ROI: 5, READY, MEDIUM]

**What:** Rebuild the ABC-44 feature pipeline with LEACE-style concept erasure for the length direction instead of regression deconfounding. Compare LEACE-deconfounded AUROC against the regression-deconfounded 0.746.

**Why:** P9-E1 used regression deconfounding which only removes linear length dependence. LEACE (2306.03819) provides closed-form perfect linear erasure — it's the gold standard for representation-level deconfounding.

**Cost:** 2h CPU
**Triggered by:** [LEACE: Linear Concept Erasure](https://arxiv.org/abs/2306.03819)
**Depends on:** F-9
**Would update:** F-9
**Trigger condition:** LEACE paper and codebase available.

---

## LOW (ROI 1-4)

### P9-FE2 (P9) — [ROI: 4, READY, LOW]

**What:** Add persistence landscapes and persistence images as PH feature representations (instead of 5 summary statistics). Re-run the Gaussian null test. Persistence landscapes (Bubenik 2015) have statistical testing guarantees that raw summaries lack.

**Why:** F-7 killed 5 raw PH summary stats. But the Gaussian null test was on a specific lossy representation. Persistence landscapes vectorize the full diagram and have known hypothesis testing properties. If landscapes also fail, PH is comprehensively dead for this use case.

**Cost:** 2 days CPU
**Triggered by:** [Persistence Landscapes (Bubenik)](https://arxiv.org/abs/1207.6437)
**Depends on:** F-7
**Would update:** F-7
**Trigger condition:** Methodological — always applicable.

---

## Completed

_(none yet)_

---

## Watchlist — Papers to monitor for new triggers

These papers are referenced as triggers for future experiments. When a
follow-up appears (or the original methodology gets a public implementation),
check whether any experiment's status should change.

| arxiv_id | title | triggers experiment(s) | status |
|---|---|---|---|
| [2601.01552](https://arxiv.org/abs/2601.01552) | HalluZig: Zigzag PH on Attention | P7-FE2 | TRIGGERED |
| [2604.14084](https://arxiv.org/abs/2604.14084) | TIP: Token Importance in OPD | P10-FE4 | TRIGGERED |
| [2604.16217](https://arxiv.org/abs/2604.16217) | Conformal Prediction via Internal Representations | P10-FE2 | TRIGGERED |
| [2501.12948](https://arxiv.org/abs/2501.12948) | DeepSeek-R1 | P11-FE5 | TRIGGERED |
| [2501.17148](https://arxiv.org/abs/2501.17148) | AxBench: Steering Benchmark | P10-FE1 | TRIGGERED |
| [2504.05419](https://arxiv.org/abs/2504.05419) | Reasoning Models Know When They're Right | P8-FE2 | TRIGGERED |
| [2504.07986](https://arxiv.org/abs/2504.07986) | SEAL: Steerable Reasoning Calibration | P10-FE1 | TRIGGERED |
| [2504.10063](https://arxiv.org/abs/2504.10063) | TOHA: Topological Divergence on Attention | P7-FE2 | TRIGGERED |
| [2505.18706](https://arxiv.org/abs/2505.18706) | Bias-Only Steering (Sinii) | P10-FE1 | TRIGGERED |
| [2506.00653](https://arxiv.org/abs/2506.00653) | Linear Representation Transferability (LRT) | P6-FE1 | TRIGGERED |
| [2506.18831](https://arxiv.org/abs/2506.18831) | STU-PID Steering | P3-FE1 | TRIGGERED |
| [2506.24106](https://arxiv.org/abs/2506.24106) | Representation Dispersion Predictive Power | P8-FE3 | TRIGGERED |
| [2510.04309](https://arxiv.org/abs/2510.04309) | PID Steering | P3-FE1 | TRIGGERED |
| [2510.18147](https://arxiv.org/abs/2510.18147) | LLMs Encode Problem Difficulty | P11-FE2 | TRIGGERED |
| [2402.10978](https://arxiv.org/abs/2402.10978) | Conformal Factuality (Mohri & Hashimoto) | P10-FE2 | TRIGGERED |
| [2402.13212](https://arxiv.org/abs/2402.13212) | Soft Self-Consistency | P11-FE4 | READY |
| [2402.18048](https://arxiv.org/abs/2402.18048) | Truthfulness via Local ID | P4-FE2 | TRIGGERED |
| [2404.15255](https://arxiv.org/abs/2404.15255) | How to Use Activation Patching | P11-FE3 | READY |
| [2410.04707](https://arxiv.org/abs/2410.04707) | Learning How Hard to Think (Damani) | P10-FE3 | TRIGGERED |
| [2410.11042](https://arxiv.org/abs/2410.11042) | Persistent Topological Features in LLMs | P7-FE1 | TRIGGERED |
| [2412.01113](https://arxiv.org/abs/2412.01113) | LLMs Faithfully Compute During CoT (Kudo) | P11-FE3 | READY |
| [2306.03341](https://arxiv.org/abs/2306.03341) | Inference-Time Intervention (ITI) | P2-FE1, P10-FE1 | READY, TRIGGERED |
| [2306.03819](https://arxiv.org/abs/2306.03819) | LEACE: Linear Concept Erasure | P4-FE1 | READY |
| [2103.07353](https://arxiv.org/abs/2103.07353) | Fast Zigzag Persistence (Dey-Hou) | P7-FE1 | TRIGGERED |
| [1207.6437](https://arxiv.org/abs/1207.6437) | Persistence Landscapes (Bubenik) | P9-FE2 | READY |
