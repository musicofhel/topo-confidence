# Next Experiments — Priority Queue

_Auto-generated from the research graph. Do not edit directly._
_Regenerate: `cd research-graph && python generate_next_experiments.py`_
_Generated: 2026-04-28 23:31 UTC_

---

## CRITICAL (ROI 9-10) — Do these first

### P11-FE19 (P11) — [ROI: 10, READY, CRITICAL]

**What:** C_exact verification routing — Desktop's highest-leverage proposal. For 500 MATH-500 problems: take K=1 greedy answer, prompt model to verify its own answer, extract PANL-equivalent activation, use as routing signal. Route verification-failed problems to K=8 sampling, keep verification-passed at K=1. Compare against (a) pure C_infer (uniform K=8), (b) pure C_exact (verify-then-correct, no sampling) at matched compute. Total avg K = 3-4.

**Why:** Combines Rybin compute-allocation framework (C_train / C_infer / C_exact decomposition) with Kumaran PANL signal. E2 prefill-gated compute allocation failed because B-bucket lives at mid-confidence and sits within C_infer. Verification routing is C_exact, where the framework predicts the win. If C_exact > C_infer at matched compute, the entire selective-prediction story (F-8) shifts from predict-difficulty-at-prefill to verify-and-route-failures.

**Cost:** 30min H100 + verification pass per problem
**Triggered by:** [How LLMs Detect and Correct Their Own Errors: Internal Confidence Signals](https://arxiv.org/abs/2604.22271)
**Depends on:** F-8, F-7, F-2
**Would update:** F-8
**Trigger condition:** Rybin compute-allocation blog provides C_infer / C_exact framework. Kumaran 2604.22271 provides the PANL routing signal.

### P8-FE1 (P8) — [ROI: 10, READY, CRITICAL]

**What:** Re-extract CoE-60 features using 1024-token generations instead of the original 256-token truncated runs. Refit the logistic regression classifier and report AUROC under the corrected label distribution (243/257 instead of 104/396).

**Why:** F-6's headline 0.811 AUROC was measured on truncated labels. The corrected baseline accuracy is 48.6% not 20.8%, which means the class balance is ~50/50 instead of ~20/80. CoE AUROC may go up (more signal) or down (less separation) — we don't know until we test.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 2h H100 + 1h CPU
**Blocked by:** H100 pod for re-extraction
**Depends on:** F-13, F-6
**Would update:** F-8, F-6
**Trigger condition:** Already triggered — F-13 invalidated the labels.

### P9-FE1 (P9) — [ROI: 10, READY, CRITICAL]

**What:** Run the full 5-experiment battery (length deconfound, orthogonality, Gaussian null, XGBoost-vs-LR, cross-domain transfer) using 1024-token corrected labels. Report which findings survive and which change.

**Why:** Every P9 number was computed on the wrong labels. This is the single most important re-validation in the project.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 2h H100 + 1 day CPU
**Blocked by:** H100 pod for fresh feature extraction
**Depends on:** F-13
**Would update:** F-10, F-9, F-8, F-7, F-6
**Trigger condition:** Already triggered — F-13.

### P10-FE1 (P10) — [ROI: 9, TRIGGERED, CRITICAL]

**What:** Run E1 steering experiment with three parallel arms: (a) probe-weight vector, (b) mass-mean (difference-of-means) vector, (c) trained per-layer bias (Sinii-style, 2505.18706). Sweep alpha, report accuracy on MATH-500 holdout for each. Use SEAL's thought-type vector (2504.07986) as a fourth comparison if feasible.

**Why:** The literature strongly predicts mass-mean >= probe-weight, and RL-trained biases exceed both (AxBench ranking, AdaRAS, Sinii). Running all three arms on the same model and benchmark gives a definitive comparison that doesn't exist in the literature.

**Source paper:** [AxBench: Steering Benchmark](https://arxiv.org/abs/2501.17148)
- *What they did:* Large-scale benchmark on Gemma-2-2B/9B comparing prompting, finetuning, SAEs, supervised steering vectors, linear probes, and ReFT-r1.
- *Their result:* Prompting > finetuning > all representation methods for steering; difference-in-means wins concept detection; SAEs are not competitive.
- *What we'll do:* Use AxBench's ranking as the prior that orders our three E1 arms (probe-weight, mass-mean, trained bias) and predicts mass-mean >= probe-weight.
- *Same:* Same comparison axis (probe-weight vs difference-in-means vs trained methods).
- *Differs:* Single-model + math-benchmark setup vs a multi-method benchmark on Gemma; we test the literature ranking on a model and benchmark the original paper does not cover.

**Source paper:** [SEAL: Steerable Reasoning Calibration](https://arxiv.org/abs/2504.07986)
- *What they did:* Categorizes CoT into execution / reflection / transition thoughts; offline-extracts a thought-type steering vector and applies it during generation.
- *Their result:* +11% accuracy and 11.8-50.4% token reduction on Math500 + GSM8K + LiveCodeBench using DeepSeek-R1-Distill and QwQ-32B.
- *What we'll do:* Run SEAL's thought-type vector as a fourth arm of E1 (when feasible).
- *Same:* Same offline-extraction-then-online-intervention recipe; same family of math benchmarks.
- *Differs:* Target is correctness (not reasoning-style efficiency); model is Qwen-2.5-1.5B base (not R1-Distill or QwQ-32B); the comparison is against probe-weight / mass-mean / bias arms.

**Source paper:** [Bias-Only Steering (Sinii)](https://arxiv.org/abs/2505.18706)
- *What they did:* Trains a single d-dim per-layer steering bias with reinforcement learning while freezing all base weights.
- *Their result:* On 8B models, +0.0016% extra params matches full RL-tuned reasoning accuracy on math benchmarks; reduces optimizer memory and inter-GPU communication.
- *What we'll do:* Run an analogous RL-trained per-layer bias arm at L19 in E1, alongside ITI mass-mean and probe-weight, on MATH-500 holdout.
- *Same:* Same per-layer bias-only trainable adapter; same math-reasoning benchmark family.
- *Differs:* Backbone is Qwen-2.5-1.5B (not 8B); we ablate it as one arm of a three-arm comparison rather than as a standalone result.

**Source paper:** [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)
- *What they did:* ITI applies an activation-shift intervention to a small set of attention heads at inference, using directions derived from labeled probes.
- *Their result:* Alpaca's TruthfulQA score nearly doubles (32.5% to 65.1%) with a few hundred examples and minimal compute.
- *What we'll do:* Treat ITI's mass-mean direction as one arm of E1 (alongside probe-weight and trained-bias) and apply it at L19 on MATH-500 with sweeps over alpha.
- *Same:* Same idea of mass-mean / difference-in-means as a steering direction.
- *Differs:* Site is the residual stream at L19 (not selected attention heads); target is math correctness (not truthfulness); evaluation is MATH-500 accuracy across an alpha sweep.

**Cost:** 1 day H100
**Blocked by:** H100 pod
**Triggered by:** [AxBench: Steering Benchmark](https://arxiv.org/abs/2501.17148), [SEAL: Steerable Reasoning Calibration](https://arxiv.org/abs/2504.07986), [Bias-Only Steering (Sinii)](https://arxiv.org/abs/2505.18706), [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)
**Depends on:** F-4, F-3
**Would update:** F-4, F-3
**Trigger condition:** Sinii and SEAL published with results on same model family.

### P11-FE5 (P11) — [ROI: 9, TRIGGERED, HIGH]

**What:** Use DeepSeek-R1-Distill-Qwen-1.5B (same architecture as project's base model, but distilled from R1 with long-CoT training) as a comparison model. Extract L19 activations on MATH-500. Compare: (a) Does breathing still occur? (b) Is the asymmetric collapse stronger? (c) Does the prefill signal improve (R1-Distill has 83.9% accuracy vs base 48.6%)? (d) Does the DoM direction transfer between base and R1-Distill?

**Why:** R1-Distill is the same architecture with massively improved reasoning via SFT on R1 traces. Comparing base vs R1-Distill isolates the effect of reasoning training on the hidden-state geometry. If breathing/collapse are stronger in R1-Distill, they're tied to reasoning capability, not just architecture.

**Source paper:** [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- *What they did:* Trains long-CoT reasoning capability via pure reinforcement learning without human-annotated reasoning trajectories; the distilled R1-Distill series transfers this to smaller backbones via SFT on R1 traces.
- *Their result:* Emergent self-reflection and verification behaviors; R1-Distill-Qwen-1.5B reaches 83.9% on MATH-500 vs 48.6% for the base.
- *What we'll do:* Run breathing / collapse / DoM analysis on R1-Distill-Qwen-1.5B and compare against the base Qwen-2.5-1.5B on MATH-500.
- *Same:* Same architecture; same MATH-500 benchmark; same residual-stream geometry analysis pipeline.
- *Differs:* Model is the SFT'd R1-Distill (not RL-on-base); we ask whether reasoning training amplifies F-1 / F-2 / F-4 or leaves them unchanged — a comparison the original R1 paper does not perform.

**Cost:** 4h H100 + 2h CPU
**Blocked by:** H100 pod
**Triggered by:** [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
**Depends on:** F-4, F-2, F-1
**Would update:** F-4, F-2, F-1
**Trigger condition:** R1-Distill-Qwen-1.5B publicly available.

### P3-FE1 (P3) — [ROI: 9, TRIGGERED, CRITICAL]

**What:** Re-run P3's steering-for-accuracy using adaptive per-token steering (PID or STU-PID) instead of fixed-vector injection. Use the proper L19 DoM direction, apply PID controller across generation positions.

**Why:** P3 failed because the steering direction rotates during generation (F-3). PID steering (2510.04309) and STU-PID (2506.18831) are designed to handle exactly this — they accumulate error and damp overshoot. STU-PID was tested on the project's exact model (DeepSeek-R1-Distill-Qwen-1.5B).

**Source paper:** [STU-PID Steering](https://arxiv.org/abs/2506.18831)
- *What they did:* Trains a chunk-level classifier for redundant reasoning patterns and uses a PID controller to adaptively modulate steering strength based on predicted redundancy.
- *Their result:* On GSM8K: +6% accuracy and -32% tokens vs static-steering baselines, training-free at inference.
- *What we'll do:* Plug STU-PID's controller into our pipeline using the L19 DoM correctness direction as the steering target on MATH-500.
- *Same:* Same per-chunk PID-controlled steering recipe targeting an LLM that overshoots without dynamic adjustment.
- *Differs:* Target signal is correctness (not redundancy); benchmark is MATH-500 not GSM8K; we layer it on an existing correctness probe rather than a fresh redundancy classifier.

**Source paper:** [PID Steering](https://arxiv.org/abs/2510.04309)
- *What they did:* Casts activation steering as a P controller, then proposes a full PID controller where I accumulates layer-wise error and D damps overshoot.
- *Their result:* Closed-loop design with stability guarantees; consistently outperforms existing steering across multiple LLM families and benchmarks.
- *What we'll do:* Apply a PID controller using the project's L19 prefill DoM as the reference signal during MATH-500 generation, instead of P3's failed fixed-vector injection.
- *Same:* Same closed-loop control framing; same idea of using a learned semantic direction as the feedback signal.
- *Differs:* Reference signal is a correctness direction (not generic semantic targets); benchmark is MATH-500; we evaluate accuracy and AUROC lift against the P3 baseline that failed due to direction rotation (F-3).

**Cost:** 1 day H100
**Triggered by:** [STU-PID Steering](https://arxiv.org/abs/2506.18831), [PID Steering](https://arxiv.org/abs/2510.04309)
**Depends on:** F-4, F-3
**Would update:** F-3
**Trigger condition:** PID Steering and STU-PID papers published.

### P11-FE15 (P11) — [ROI: 9, READY, HIGH]

**What:** Length-band PR control on MATH-500. Split 500 problems into three generation-length bands (<25th, 25-75th, >75th percentile). Compute final-token PR per band, separately for correct vs incorrect. Refutation test for whether the asymmetric-collapse finding (F-4) survives within length-matched subsets.

**Why:** Ríos-García 2604.18805 anti-pattern: 68% of agent runs ignore disconfirming evidence. Self-applied refutation test for F-4. If correct/incorrect PR ratio approaches 1.0 within bands, the collapse is a length-mixing artifact, not a real signal.

**Cost:** 20min CPU on cached Stage 2 NPZs
**Triggered by:** [AI scientists produce results without reasoning scientifically](https://arxiv.org/abs/2604.18805)
**Depends on:** F-4, F-1
**Would update:** F-4, F-1
**Trigger condition:** Ríos-García 2604.18805 motivates self-applied refutation tests for headline findings.

### P11-FE16 (P11) — [ROI: 9, READY, HIGH]

**What:** Implement Marchenko-Pastur bias correction for participation ratio. Recompute (a) temporal PR curve for Qwen 1.5B + 7B, (b) correct vs incorrect PR at prefill + final token, (c) cross-model peak-PR comparisons. Critical question: does asymmetric collapse (F-4) and cross-model breathing magnitude (F-1) survive correction?

**Why:** Chun et al. 2509.26560 show naive PR is biased by P/Q ratio. Our P/Q values: Qwen 1.5B 0.33, Qwen 7B 0.14, Llama 0.24, Phi-3 0.16 — all in the bias-significant regime. Cross-model PR comparisons at fixed n=500 are exactly the scenario the paper warns against.

**Cost:** 1h CPU on cached Stage 2 NPZs
**Triggered by:** [Estimating Dimensionality of Neural Representations from Finite Samples](https://arxiv.org/abs/2509.26560)
**Depends on:** F-4, F-1
**Would update:** F-4, F-1
**Trigger condition:** Chun et al. 2509.26560 published bias-corrected estimator with code.

### P11-FE18 (P11) — [ROI: 9, READY, HIGH]

**What:** PANL-equivalent correctability gate. In K=1 greedy generations, locate the post-answer-newline token per problem, extract L19 activations, compute DoM AUROC predicting (a) K=1 correctness, (b) K=8 majority correctness, (c) B-bucket membership (K=1 wrong, K=8 right — recoverable), (d) D-bucket membership (K=1 right, K=8 wrong — pathological, F-7). Compare against prefill (F-2) and final-token AUROC for the same four targets.

**Why:** Kumaran et al. 2604.22271 PANL signal predicts error correction (AUROC 0.986 Gemma, 0.961 Qwen 7B) — orthogonal to verification logprobs (cos=0.007). Our prefill DoM is the pre-hoc analog (cos=-0.06 with final DoM). The B/D-bucket question prefill couldn't answer (Exp 2 monotonic gating fails because B lives at mid-confidence) might be answerable via PANL-equivalent.

**Cost:** 30min H100 (2060 sufficient)
**Triggered by:** [How LLMs Detect and Correct Their Own Errors: Internal Confidence Signals](https://arxiv.org/abs/2604.22271)
**Depends on:** F-7, F-2
**Would update:** F-7
**Trigger condition:** Kumaran et al. 2604.22271 introduces the PANL signal.

### P11-FE3 (P11) — [ROI: 9, READY, HIGH]

**What:** Compute the causal effect of prefill geometry on downstream correctness using activation patching. Swap prefill activations between a correct-predicted and incorrect-predicted problem pair at L19. If swapping flips the outcome, the prefill signal is causal. If not, it's correlational.

**Why:** All findings so far are correlational probes. Kudo et al. (2412.01113) used causal interventions to confirm models compute sub-answers during CoT. The same methodology applied to prefill would determine whether the 'can I solve this' signal actually drives behavior or merely correlates with it.

**Source paper:** [How to Use Activation Patching](https://arxiv.org/abs/2404.15255)
- *What they did:* Survey + best-practice tutorial on activation patching: how to choose metrics, how to interpret causal effects, common pitfalls.
- *Their result:* Methodology document with no headline number — provides the right-tool guidance for circuit-level interventions.
- *What we'll do:* Follow the tutorial's metric and validation guidance when designing the prefill-swap intervention so the causal claim is interpretable.
- *Same:* Same activation-patching methodology and same concern about how to interpret the result.
- *Differs:* We are users, not contributors; the focus is whether prefill geometry is causal for correctness rather than circuit identification.

**Source paper:** [LLMs Faithfully Compute During CoT (Kudo)](https://arxiv.org/abs/2412.01113)
- *What they did:* Multi-step arithmetic + activation patching to determine when LLMs commit to an answer relative to CoT generation.
- *Their result:* LLMs do not pre-determine the answer; they iteratively compute sub-answers during CoT — CoT is a faithful reflection of internal computation.
- *What we'll do:* Apply the same activation-patching methodology to swap prefill activations between correct- and incorrect-predicted MATH-500 problem pairs at L19.
- *Same:* Same causal-intervention recipe (activation patching across CoT positions).
- *Differs:* We test prefill (not mid-CoT) and correctness (not arithmetic-answer faithfulness); the swap is between problem pairs, not corrupted-vs-clean copies of the same problem.

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

**Source paper:** [Conformal Prediction via Internal Representations](https://arxiv.org/abs/2604.16217)
- *What they did:* Defines Layer-Wise Information (LI) scores from internal representations and uses them as conformal nonconformity scores in split conformal prediction.
- *Their result:* LI scores beat token-probability and entropy baselines on the validity-efficiency tradeoff, especially under cross-domain shift.
- *What we'll do:* Substitute the L19 DoM score for the LI score and report conformal coverage at fixed risk levels on MATH-500.
- *Same:* Same internal-rep nonconformity score; same conformal pipeline.
- *Differs:* Score is single-layer DoM (not layer-wise information); benchmark is math reasoning (not closed-book / open-domain QA); we have a published probe AUROC (0.7731) to compare against.

**Source paper:** [Conformal Factuality (Mohri & Hashimoto)](https://arxiv.org/abs/2402.10978)
- *What they did:* Conformal factuality: a back-off algorithm that progressively makes LM outputs less specific until conformal prediction guarantees high-probability correctness.
- *Their result:* 80-90% correctness guarantees on FActScore, NaturalQuestions, MATH while preserving most of the original output content.
- *What we'll do:* Wrap the project's L19 DoM probe as a conformal nonconformity score and report coverage-vs-accuracy tradeoffs on MATH-500.
- *Same:* Same conformal-prediction wrapper around an LM-correctness signal; same MATH benchmark.
- *Differs:* Score is an internal-representation probe (not back-off entailment); we report a single operating point (e.g. 71.6%@50% coverage) with a guarantee, not a back-off cascade.

**Cost:** 4h CPU
**Triggered by:** [Conformal Prediction via Internal Representations](https://arxiv.org/abs/2604.16217), [Conformal Factuality (Mohri & Hashimoto)](https://arxiv.org/abs/2402.10978)
**Depends on:** F-2, F-11
**Would update:** F-11
**Trigger condition:** Conformal factuality paper published with MATH numbers.

### P10-FE3 (P10) — [ROI: 8, TRIGGERED, HIGH]

**What:** Implement E2 (confidence-gated compute) using Damani et al.'s 'Learning How Hard to Think' framework (2410.04707). Train a lightweight predictor on prefill embeddings to estimate marginal reward of additional samples. Compare against the project's monotonic/non-monotonic gating baselines.

**Why:** F-14 showed monotonic gating fails. Damani's framework is purpose-built for this — it trains a meta-predictor of when more compute helps, which is exactly the bucket-B recovery problem.

**Source paper:** [Learning How Hard to Think (Damani)](https://arxiv.org/abs/2410.04707)
- *What they did:* Trains a meta-predictor of the reward distribution given an input + budget; uses it for adaptive best-of-k and decoder routing.
- *Their result:* Up to -50% compute at fixed quality, or +10% quality at fixed compute, across programming + math + dialog suites.
- *What we'll do:* Train a Damani-style predictor on prefill embeddings to estimate marginal reward of additional samples; gate K dynamically against the project's bucket-B problems.
- *Same:* Same meta-predictor framing; same input-adaptive compute-allocation goal.
- *Differs:* Predictor input is L19 prefill (not text features); gating policy targets the bucket-B recovery problem (F-14) rather than generic best-of-k; benchmark is MATH-500 with the project's K-vs-accuracy curve.

**Cost:** 1 day CPU + 4h H100 for sample generation
**Triggered by:** [Learning How Hard to Think (Damani)](https://arxiv.org/abs/2410.04707)
**Depends on:** F-2, F-14
**Would update:** F-14
**Trigger condition:** Damani et al. published.

### P11-FE2 (P11) — [ROI: 8, TRIGGERED, HIGH]

**What:** Test whether the prefill direction encodes training-set familiarity or problem structure. Use MATH-500 difficulty level metadata (5 levels in the original dataset). Compute prefill DoM score vs difficulty level. If prefill tracks difficulty perfectly, it's familiarity. If orthogonal to difficulty but still predicts correctness, it's something more interesting.

**Why:** F-2's strongest counterargument is that prefill encodes familiarity, not structure. This test directly addresses it. Lugoloobi & Russell (2510.18147) showed Qwen2.5-Math-1.5B encodes human difficulty with rho=0.88 — is our prefill DoM the same signal or a different one?

**Source paper:** [LLMs Encode Problem Difficulty](https://arxiv.org/abs/2510.18147)
- *What they did:* Trains linear probes across layers and token positions on 60 models and evaluates on Easy2HardBench math + coding subsets.
- *Their result:* Human-difficulty rho approx 0.88 on AMC with clear model-size scaling; steering toward 'easier' reduces hallucination; GRPO on Qwen2.5-Math-1.5B amplifies the human-difficulty probe.
- *What we'll do:* Compute the project's prefill DoM score against MATH-500 difficulty levels (1-5) on Qwen-2.5-1.5B and check whether the prefill direction is the same as Lugoloobi's difficulty direction or orthogonal to it.
- *Same:* Same model family (Qwen2.5-Math-1.5B), same kind of math benchmark, same probe-the-residual-stream method.
- *Differs:* We are testing whether prefill-DoM-correctness != difficulty (orthogonality); they showed difficulty itself is encoded — we want to know if our 'can I solve this' signal is a different direction.

**Cost:** 1h CPU
**Triggered by:** [LLMs Encode Problem Difficulty](https://arxiv.org/abs/2510.18147)
**Depends on:** F-2
**Would update:** F-2
**Trigger condition:** Lugoloobi & Russell published on exact model.

### P11-FE6 (P11) — [ROI: 8, TRIGGERED, HIGH]

**What:** Compute spectral alpha (heavy-tailed self-regularization exponent) of L19 prefill activations on the 500 MATH-500 problems. Use it as a label-free correctness predictor and compare AUROC head-to-head against supervised prefill DoM (F-2's 0.7731).

**Why:** F-2's L19 prefill DoM is a supervised probe requiring 243+257 labels. The Spectral Geometry of Thought paper (2604.15350) shows alpha of hidden activations predicts correctness pre-generation. If alpha matches DoM AUROC, refusal becomes label-free (closes H-12). If it loses substantially, F-2's supervised structure is genuinely informative beyond the universal alpha signal.

**Source paper:** [The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason](https://arxiv.org/abs/2604.15350)
- *What they did:* Computes spectral alpha (heavy-tailed self-regularization exponent) of hidden activations across layers and uses it as a label-free correctness predictor pre-generation.
- *Their result:* Alpha exhibits universal phase transitions across model families and predicts correctness on math benchmarks without supervision.
- *What we'll do:* Compute alpha at L19 on the project's cached prefill activations and report AUROC head-to-head against F-2's supervised prefill DoM (0.7731).
- *Same:* Same idea of a layer-localized scalar feature on residual-stream activations as a correctness predictor.
- *Differs:* Alpha is unsupervised (no labels needed); DoM is supervised (243+257-label probe). Same model (Qwen-2.5-1.5B) and benchmark (MATH-500) — direct head-to-head AUROC comparison.

**Cost:** 2h CPU on cached activations
**Triggered by:** [The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason](https://arxiv.org/abs/2604.15350)
**Depends on:** F-2
**Would update:** F-2
**Trigger condition:** 2604.15350 Spectral Geometry of Thought claims alpha is universal across model families and predicts correctness pre-generation.

### P4-FE2 (P4) — [ROI: 8, TRIGGERED, HIGH]

**What:** Replace TwoNN global ID with GeoMLE local intrinsic dimension estimation per-problem. Compute local LID at L19 for each MATH-500 problem's activation neighborhood (k=20). Test as correctness predictor.

**Why:** F-10 showed global TwoNN has near-zero predictive power (AUROC 0.407). But Yin et al. (2402.18048) showed LOCAL LID predicts truthfulness with 5-8 AUROC points above baselines using GeoMLE. The discrepancy is likely global-vs-local — local LID may recover the signal TwoNN misses.

**Source paper:** [Truthfulness via Local ID](https://arxiv.org/abs/2402.18048)
- *What they did:* Estimates per-token local intrinsic dimension (LID) of LLM activations via GeoMLE on a k-NN neighborhood and uses it as a truthfulness score.
- *Their result:* On four QA datasets, local LID outperforms entropy and verbalized-confidence baselines by 5-8 AUROC points.
- *What we'll do:* Compute GeoMLE local LID at L19 per MATH-500 problem (k=20) on cached activations and test it as a correctness predictor.
- *Same:* Same LID-on-activations methodology and same GeoMLE estimator; same intent of recovering signal that global ID misses.
- *Differs:* Local instead of global TwoNN (the project's failed P9 setup); benchmark is MATH-500 correctness rather than QA truthfulness; layer is L19, not the paper's mid-layer choice.

**Cost:** 4h CPU on cached activations
**Triggered by:** [Truthfulness via Local ID](https://arxiv.org/abs/2402.18048)
**Depends on:** F-10
**Would update:** F-10
**Trigger condition:** Already triggered — Yin et al. published with code.

### P8-FE2 (P8) — [ROI: 8, TRIGGERED, HIGH]

**What:** Extract CoE features at step/chunk boundaries during chain-of-thought (every 50 tokens) rather than only at the final token. Train a step-level correctness probe. Compare against Zhang et al.'s chunk-boundary probing (2504.05419) which achieved >0.9 AUROC on AIME.

**Why:** Current CoE features are final-token only. Zhang et al. showed mid-trajectory probes can beat final-token probes substantially. Step-level CoE also feeds directly into E2 (confidence-gated compute) and E3 (refusal) — you need temporal resolution to gate during generation, not after.

**Source paper:** [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419)
- *What they did:* Trains linear probes on hidden states at intermediate-answer positions during long-CoT reasoning to verify correctness mid-generation.
- *Their result:* Probes verify intermediate answers with high accuracy and calibration; using the probe as an early-exit verifier cuts inference tokens by 24% with no quality drop.
- *What we'll do:* Extract CoE features at chunk boundaries (every 50 tokens) on cached MATH-500 generations and train a step-level correctness probe.
- *Same:* Same step-level probing target (correctness encoded in hidden states) and same goal of beating final-token-only probes.
- *Differs:* We probe CoE-style geometry features (not raw hidden state); benchmark is MATH-500 (not AIME); the result feeds into selective-prediction (E2/E3), not just early exit.

**Cost:** 4h H100 + 2h CPU
**Blocked by:** H100 pod
**Triggered by:** [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419)
**Depends on:** F-2, F-6
**Would update:** F-2, F-6
**Trigger condition:** Zhang et al. published with strong results.

### P11-FE1 (P11) — [ROI: 8, READY, HIGH]

**What:** Run breathing analysis (temporal PR curve) on BBH subsets using Stage 4a per-token activations. Does breathing occur on non-math tasks? Does the correct/incorrect PR ratio hold on web_of_lies (54% accuracy, near-balanced)?

**Why:** F-1 was measured on MATH-500 only. BBH data is already cached on the pod. If breathing occurs on BBH, it's a reasoning phenomenon. If not, it's math-specific.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 2h CPU
**Depends on:** F-1
**Would update:** F-1
**Trigger condition:** Data already cached.

### P11-FE17 (P11) — [ROI: 8, READY, HIGH]

**What:** RoPE de-rotation of the DoM direction. For 1.5B per-token L19 activations, undo the RoPE rotation at each token position (deterministic from model config), recompute the temporal DoM AUROC curve and cosine-to-final table. If cos(DoM_t, DoM_final) rises above 0.8 after de-rotation, the rotation is mechanical and E1 fixed-vector steering becomes viable again.

**Why:** Puranik (Jane Street) shows all valid positional encodings are matrix groups exp(M·tau). RoPE applies constant-frequency rotations to dimension pairs. F-3 (orthogonal prefill/final DoM) might be purely a RoPE consequence rather than a computational orthogonality. L19 is post-attention/MLP so partial explanation (cos 0.2 to 0.6) is still informative.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 1h CPU on cached per-token activations
**Depends on:** F-3
**Would update:** F-3
**Trigger condition:** Jane Street blog (Puranik) on group theory and positional encodings frames RoPE as a deterministic per-position rotation.

### P10-FE5 (P10) — [ROI: 7, TRIGGERED, HIGH]

**What:** Train an SAE on L19 prefill activations and compute cosine of L19 DoM against (a) the SAE feature most predictive of correctness and (b) the SAE feature most predictive of model self-reported uncertainty. If DoM is parallel to (a) and orthogonal to (b), uncertainty and correctness live in different residual-stream subspaces. If DoM is parallel to both, the project's headline AUROC may be partly inheriting an uncertainty signal.

**Why:** Hofmann et al. (2604.19974) used SAEs to dissociate uncertainty vs correctness features at activation level. Our F-2 supervised probe doesn't distinguish these — the prefill direction may be 'I'm uncertain' rather than 'I can't solve this.' If the two features are confusable in residual space at L19, F-2 needs a counterargument response; if they're cleanly orthogonal, F-2's structural-prediction interpretation is supported.

**Source paper:** [Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders](https://arxiv.org/abs/2604.19974)
- *What they did:* Train SAEs on hidden activations across multiple LMs; identify the most-predictive features for self-reported uncertainty and for correctness; check if they're the same feature.
- *Their result:* In some models, uncertainty and correctness load on different SAE features; in others, they're confusable. Whether they dissociate is model-dependent.
- *What we'll do:* Train an SAE on L19 prefill activations on Qwen-2.5-1.5B; compute cos(L19 DoM, top-correctness-feature) and cos(L19 DoM, top-uncertainty-feature). Report whether DoM aligns with one or both.
- *Same:* Same SAE-on-residual-stream method; same uncertainty-vs-correctness dissociation question.
- *Differs:* We test whether the project's specific F-2 direction (L19 prefill DoM) aligns with the correctness or uncertainty SAE feature on Qwen-2.5-1.5B — Hofmann et al. didn't run this exact pairing.

**Cost:** 4h H100 for SAE training + 1d CPU for analysis
**Triggered by:** [Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders](https://arxiv.org/abs/2604.19974)
**Depends on:** F-2
**Would update:** F-2
**Trigger condition:** 2604.19974 demonstrates the SAE-based dissociation methodology and shows in some models the two features are not separable; same analysis on Qwen-2.5-1.5B at L19 hasn't been done.

### P11-FE7 (P11) — [ROI: 7, TRIGGERED, HIGH]

**What:** Test whether L19 prefill-DoM is a single direction or a low-dim manifold. Compute SVD of the per-problem prefill activations within each correctness class. If the top-k>1 components carry comparable AUROC to k=1, F-2 should be reframed as 'low-dim manifold mediates correctness' rather than 'a single direction'.

**Why:** Refusal in LMs (Wollschlaeger 2511.08379, Arditi 2406.11717) was first claimed to be a single direction, then revised to a multi-direction manifold. F-2's framing as 'a single direction' inherits the same potential issue. This test uses the project's existing cached activations and would update F-2's framing without new compute. It's a quick check that could substantially change how the headline 0.7731 AUROC is interpreted.

**Source paper:** [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- *What they did:* Activation-difference analysis on residual streams identifying a single 'refusal direction' via mean-difference of harmful-vs-harmless prompts.
- *Their result:* Refusal in LMs is mediated by a single residual-stream direction; ablating it bypasses safety training.
- *What we'll do:* Apply the same single-direction framing as a baseline (top-1 SVD component) and check whether higher components add meaningful correctness AUROC.
- *Same:* Same 'single residual-stream direction mediates a behavior' framing as F-2/F-3 currently use for L19 prefill DoM.
- *Differs:* Behavior is correctness rather than refusal; the test is whether the single-direction story holds (replicates Arditi at L19 for correctness) or breaks (motivates a manifold reframing).

**Source paper:** [SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](https://arxiv.org/abs/2511.08379)
- *What they did:* Self-organizing-maps applied to LM activations identify multiple parallel directions encoding refusal behavior; argues refusal is a low-dim manifold not a single direction.
- *Their result:* Multi-directional refusal suppression succeeds where single-direction ablation fails on adversarial prompts.
- *What we'll do:* Run SVD on per-problem prefill activations within each correctness class on Qwen-2.5-1.5B L19; check whether top-k>1 components carry meaningful correctness AUROC vs k=1 only.
- *Same:* Same single-direction-vs-manifold question applied to a residual-stream behavior signal.
- *Differs:* Behavior is correctness (not refusal); analysis is SVD on prefill activations (not SOM); test is whether F-2's headline 0.7731 should be reframed as a manifold rather than a single direction.

**Cost:** 4h CPU on cached prefill activations
**Triggered by:** [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), [SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](https://arxiv.org/abs/2511.08379)
**Depends on:** F-3, F-2
**Would update:** F-2
**Trigger condition:** 2511.08379 SOM Directions and 2406.11717 Refusal Single Direction together motivate the manifold-vs-direction question; the same test on prefill DoM is straightforward and decisive.

### P7-FE1 (P7) — [ROI: 7, TRIGGERED, HIGH]

**What:** Run zigzag persistence on the cached pathway 8 layer-wise activations using the Dey-Hou fast zigzag implementation (github.com/taohou01/fzz) or the RitaSciencePark/topo_llm pipeline. Compare zigzag PH features against the standard Rips PH that hit the Gaussian null.

**Why:** F-7 killed standard Rips PH. But zigzag PH captures transition dynamics between layers — the exact signal that CoE captures geometrically. If zigzag PH survives the Gaussian null where Rips didn't, it means PH *can* work on LLM activations but only when the filtration respects layer ordering.

**Source paper:** [Persistent Topological Features in LLMs](https://arxiv.org/abs/2410.11042)
- *What they did:* Builds a zigzag-persistence filtration across LLM layers and extracts topological descriptors that track holes evolving through the network.
- *Their result:* Descriptors are sensitive to model and dataset and enable layer pruning competitive with state-of-the-art while preserving a system-level view.
- *What we'll do:* Run zigzag PH on the cached pathway-8 layer-wise activations and check whether zigzag features survive the Gaussian null where Rips-PH didn't (F-7).
- *Same:* Same zigzag-across-layers framing applied to LLM residual streams.
- *Differs:* We test for correctness signal (not pruning); we apply a Gaussian-null control because Rips-PH failed it; we compare against per-layer DoM rather than to layer-pruning baselines.

**Source paper:** [Fast Zigzag Persistence (Dey-Hou)](https://arxiv.org/abs/2103.07353)
- *What they did:* Near-linear-time algorithms for 0- and 1-dimensional zigzag persistence on graphs (O(m log^2 n) and O(m log^4 n)).
- *Their result:* Practical zigzag computation on real-world graphs for the first time, removing the cubic-matrix-multiplication bottleneck.
- *What we'll do:* Use Dey-Hou's fzz library to make P7-FE1 computationally feasible on 28-layer activations across 500 problems.
- *Same:* Same zigzag-persistence algorithm we depend on for the experiment to terminate.
- *Differs:* We are a downstream user, not a competitor; the project's contribution is whether the resulting features predict correctness, not the algorithm itself.

**Cost:** 3 days CPU
**Triggered by:** [Persistent Topological Features in LLMs](https://arxiv.org/abs/2410.11042), [Fast Zigzag Persistence (Dey-Hou)](https://arxiv.org/abs/2103.07353)
**Depends on:** F-8, F-7
**Would update:** F-8, F-7
**Trigger condition:** Fast zigzag implementations now available. Dionysus2 install blocker removed.

### P9-FE4 (P9) — [ROI: 7, TRIGGERED, HIGH]

**What:** Re-run the prefill-DoM-gated selective prediction experiment (E3) using PDS (the verifier signal from Verify-Step-by-Step, 2402.10528) instead of L19 DoM as the gating score. Report acc@coverage 0.5 and compare against F-8's 71.6% baseline.

**Why:** F-2/F-8 use a single supervised DoM probe. PDS is a fundamentally different signal (CoT-derived verifier output, not a residual-stream probe). If PDS matches DoM at 71.6%@50%, the DoM advantage is competing-signal-bounded; if PDS is much worse, F-8 is robustly the strongest correctness signal. Either result locks in the F-8 baseline against an obvious alternative.

**Source paper:** [Can We Verify Step by Step for Incorrect Answer Detection?](https://arxiv.org/abs/2402.10528)
- *What they did:* PDS (Predictive Decoding Score) — a verifier signal derived from CoT reasoning chains used for incorrect-answer detection.
- *Their result:* PDS competitively detects incorrect answers across math benchmarks; sometimes outperforms model-internal probes.
- *What we'll do:* Use PDS as the gating score in F-8's refuse-and-spend protocol at coverage 0.5 and report acc-on-answered against the 71.6% DoM baseline.
- *Same:* Same selective-prediction problem; same coverage-vs-accuracy tradeoff framing.
- *Differs:* Gating score is CoT-derived (post-generation) rather than prefill (pre-generation) — comparison reveals whether F-8's pre-generation gating is bounded by post-generation alternatives.

**Cost:** 4h CPU + 2h H100 for PDS extraction
**Triggered by:** [Can We Verify Step by Step for Incorrect Answer Detection?](https://arxiv.org/abs/2402.10528)
**Depends on:** F-8, F-2
**Would update:** F-8
**Trigger condition:** 2402.10528 introduces PDS as an alternative to model-internal probes; the comparison hasn't been run on MATH-500 with K-allocation accounting.

### P11-FE20 (P11) — [ROI: 7, READY, MEDIUM]

**What:** Cross-model kernel alignment fluctuation during inference. Compute mutual k-NN alignment between all 6 model pairs (Qwen 1.5B, Qwen 7B, Phi-3, Llama) at each of 7 temporal positions using cached 2/3-depth activations. Predicts: alignment peaks at low-PR moments (prefill, final token), reaches min at mid-generation peak PR. Extends Platonic Representation Hypothesis from static convergence (Huh et al.) to dynamic convergence.

**Why:** Huh et al. 2405.07987 measure representational convergence via mutual k-NN alignment of kernel matrices. Our F-1 cross-arch breathing universality is evidence for the Platonic Hypothesis applied to inference-time dynamics — not just static representations but their temporal evolution converges.

**Cost:** 2h CPU, all data cached
**Triggered by:** [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)
**Depends on:** F-1
**Would update:** F-1
**Trigger condition:** Huh et al. 2405.07987 Platonic Representation Hypothesis (existing — re-saved 2026-04-28).

### P11-FE4 (P11) — [ROI: 7, READY, MEDIUM]

**What:** Characterize D-bucket (F-12) problems by mid-generation features. Extract L19 activations at token positions 50, 100, 150 for all 500 MATH-500 problems. Train a classifier to distinguish bucket D (K=1 right, K=8 wrong) from bucket A (K=1 right, K=8 right). If mid-generation features detect D-bucket, the signal appears during reasoning, not before.

**Why:** F-12 showed D-bucket is undetectable from prefill features. But D-bucket is about self-consistency failure during generation — the model starts right and talks itself into a wrong answer. The signal should appear when the reasoning trajectory diverges.

**Source paper:** [Soft Self-Consistency](https://arxiv.org/abs/2402.13212)
- *What they did:* Replaces self-consistency majority voting with a continuous likelihood-based score, enabling selection on long-horizon agentic tasks where answers are sparsely distributed.
- *Their result:* +1.3% (bash), +6.6% (WebShop), +4.7% (ALFWorld) absolute success-rate gains; matches SC quality with half the samples.
- *What we'll do:* Train a classifier on mid-generation L19 features (token positions 50/100/150) to detect bucket-D problems where K=1 right but K=8 wrong (self-consistency failure).
- *Same:* Same target (self-consistency failure modes when answers diverge); same diagnosis that majority voting masks signal.
- *Differs:* Probe-on-hidden-states, not output-likelihood scoring; benchmark is MATH-500 K=8 sampling (not interactive agents); detection target is the bucket-D self-consistency-failure case (F-12).

**Cost:** 4h H100 + 2h CPU
**Blocked by:** Per-token activations from K=8 runs (may need pod)
**Triggered by:** [Soft Self-Consistency](https://arxiv.org/abs/2402.13212)
**Depends on:** F-3, F-12
**Would update:** F-12
**Trigger condition:** Soft Self-Consistency (2402.13212) documented SC failure modes on diverse answers — same phenomenon.

### P2-FE1 (P2) — [ROI: 7, READY, HIGH]

**What:** Re-derive the steering vector from scratch using 1024-token labels and proper DoM at L19. Compare against the original P2 cached vector (cos=0.05 with current L19 DoM).

**Why:** The P2 steering vector was derived with truncated labels. F-3 showed it's unusable (cos=0.05). A properly derived vector may still fail due to direction rotation (F-3), but at least it tests steering with a non-stale direction.

**Source paper:** [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)
- *What they did:* ITI shifts activations across a small set of attention heads at inference, using directions learned from a few hundred labeled examples.
- *Their result:* Alpaca's TruthfulQA score nearly doubles (32.5% to 65.1%) with minimal data and compute.
- *What we'll do:* Re-derive the steering vector at L19 from 1024-token correctness labels and test it on MATH-500 generation, comparing cos against the project's stale P2 vector.
- *Same:* Same idea of a label-derived directional intervention on residual-stream geometry.
- *Differs:* Target is correctness on math (not truthfulness on QA); intervention site is residual stream not attention heads; the goal is to test whether re-derivation rescues a known-stale steering vector.

**Cost:** 2h CPU + 4h H100 for generation with intervention
**Triggered by:** [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)
**Depends on:** F-13, F-3
**Would update:** F-3
**Trigger condition:** Already triggered — F-3 and F-13 together show the old vector was doubly wrong (wrong labels + wrong subspace).

### P9-FE3 (P9) — [ROI: 7, READY, HIGH]

**What:** Run cross-domain transfer experiment (P9-E5) on individual BBH subsets instead of pooled. Report per-subset AUROC for CoE, D2H, and DoM. Identify which subset drives the D2H asymmetry.

**Why:** The pooled BBH transfer (0.720/0.712 for CoE) may be driven by one subset. web_of_lies has 54% accuracy (near-balanced), tracking_shuffled_objects has 12.4%, logical_deduction has 11.2%. The subset heterogeneity is extreme and pooling masks it.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 2h CPU on cached features
**Depends on:** F-6
**Would update:** F-6
**Trigger condition:** Already identified as open question in pathway 9.

---

## MEDIUM (ROI 5-6)

### P10-FE4 (P10) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Implement E4 distillation using TIP's Q3-token-focused loss (2604.14084). Fine-tune Qwen2.5-1.5B with auxiliary loss matching its L19 activations to 7B's L20 activations, but only at Q3 tokens (low student entropy, high teacher-student divergence). Compare against uniform distillation.

**Why:** TIP showed Q3 tokens carry disproportionate corrective signal on the exact model pair (Qwen2.5-14B->1.5B). Combining with the project's finding that L19 carries the correctness signal gives a focused distillation target: match the correctness direction at positions where the student is overconfidently wrong.

**Source paper:** [TIP: Token Importance in OPD](https://arxiv.org/abs/2604.14084)
- *What they did:* Distillation token-importance taxonomy across student-entropy x teacher-student-divergence; Q3 = low-entropy + high-divergence (overconfident-wrong) tokens carry dense corrective signal.
- *Their result:* Q3-only training on <20% of tokens surpasses full-token OPD on MATH-500 + AIME 2024/2025 + DeepPlanning across Qwen3 / Llama / Qwen2.5 teacher-student pairs.
- *What we'll do:* Distill Qwen-2.5-1.5B from the 7B teacher with a Q3-token loss matching L19 (student) to L20 (teacher) only at correctness-relevant positions.
- *Same:* Same Q3 token-selection rule; same Qwen2.5 teacher-student family.
- *Differs:* Loss target is the L19 correctness direction (not the teacher's full distribution); we apply distillation to a single layer pair at the token positions where the student's correctness probe disagrees with the teacher's.

**Cost:** 2+ weeks, multi-day H100
**Blocked by:** Multi-day H100 allocation
**Triggered by:** [TIP: Token Importance in OPD](https://arxiv.org/abs/2604.14084)
**Depends on:** F-5, F-4
**Would update:** F-5
**Trigger condition:** TIP published with code and Qwen results.

### P11-FE8 (P11) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Replicate Mostafazadeh et al.'s question-only-probe protocol (2509.10625) on Qwen-2.5-1.5B at L19. Extract activations on MATH-500 question text only (no answer/CoT) and fit a logistic probe; compare AUROC against the project's prefill L19 DoM (0.7731 from F-2).

**Why:** F-2 is one model, one benchmark. Mostafazadeh et al. show question-only probes generalize across models — running their exact protocol on Qwen-2.5-1.5B is a direct independent-replication test. Match → F-2 is robust and externally validated. Mismatch → the L19 DoM result may be Qwen-specific or methodology-specific.

**Source paper:** [No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](https://arxiv.org/abs/2509.10625)
- *What they did:* Linear probes on question-only activations (no answer, no CoT) to predict downstream answer accuracy on math benchmarks.
- *Their result:* Question-only probes generalize across models; AUROCs are competitive with answer-aware probes despite using less information.
- *What we'll do:* Apply the same protocol on Qwen-2.5-1.5B L19 prefill activations and compare AUROC against F-2's 0.7731 (which uses prefill = last prompt token, equivalent to question-end).
- *Same:* Same prefill/question-only-probe → linear-classifier → MATH-500 setup.
- *Differs:* We use the exact F-2 layer (L19) and probe-fit protocol; the result is an independent replication that strengthens or weakens F-2's external validity.

**Cost:** 2h CPU on cached prefill activations + 1h H100 if re-extracting on question-only prompts
**Triggered by:** [No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](https://arxiv.org/abs/2509.10625)
**Depends on:** F-2
**Would update:** F-2
**Trigger condition:** 2509.10625 publishes a question-only probe protocol that produces comparable AUROCs across models; same protocol on the project's exact model is a 1-day independent replication.

### P6-FE1 (P6) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Test cross-scale probe transfer using affine alignment (Bello et al. LRT, 2506.00653) instead of raw transfer. Train a lightweight affine map from 1.5B L19 activations to 7B L20 activations on a shared problem set, then test whether the 7B's DoM direction transfers to 1.5B through the map.

**Why:** F-5 showed raw cross-scale transfer is flat (0.717 vs self 0.719). But LRT (2506.00653) showed affine maps between model sizes preserve steering-relevant structure. The question is whether an aligned 7B correctness direction transfers better than unaligned.

**Source paper:** [Linear Representation Transferability (LRT)](https://arxiv.org/abs/2506.00653)
- *What they did:* Learns affine maps between hidden states of different-sized models on shared inputs, then transfers steering vectors through the map (the LRT hypothesis).
- *Their result:* Steering vectors transferred small-to-large preserve their semantic effect, supporting linear representation transferability across scales.
- *What we'll do:* Train an affine map from 1.5B L19 to 7B L20 activations, then test whether the 7B's correctness DoM transfers back to 1.5B and beats raw cross-scale transfer (0.717).
- *Same:* Same affine-alignment-then-transfer recipe; same hypothesis that representations across scales share linear structure.
- *Differs:* Direction is a correctness probe (not a generic steering target); we measure AUROC lift on MATH-500 vs the raw-transfer baseline; mapping direction is large-to-small (1.5B inheriting from 7B).

**Cost:** 4h H100 + 2h CPU
**Blocked by:** H100 pod for 7B activations
**Triggered by:** [Linear Representation Transferability (LRT)](https://arxiv.org/abs/2506.00653)
**Depends on:** F-5
**Would update:** F-5
**Trigger condition:** LRT paper published with theoretical framework.

### P7-FE2 (P7) — [ROI: 6, TRIGGERED, MEDIUM]

**What:** Compute PH on attention graphs (token-token attention matrices per layer) instead of residual-stream point clouds. Use the TOHA/HalluZig approach: build a weighted graph from attention maps, compute PH of the graph filtration.

**Why:** Every successful PH-on-LLM paper in the literature (Kushnareva 2021, Bazarova 2025) works on attention graphs, not residual-stream point clouds. The project tested only the latter. PH may be valid for LLM correctness detection — just on the wrong object.

**Source paper:** [HalluZig: Zigzag PH on Attention](https://arxiv.org/abs/2601.01552)
- *What they did:* Models per-layer attention matrices as a zigzag graph filtration and extracts a topological signature from the dynamic graph evolution.
- *Their result:* Outperforms strong hallucination-detection baselines across benchmarks; signatures generalize across models and across partial network depth.
- *What we'll do:* Apply the same zigzag-on-attention-graphs pipeline to MATH-500 correctness instead of factuality, on the project's cached attention matrices.
- *Same:* Same object (attention graphs) and same zigzag-on-attention filtration.
- *Differs:* Target is correctness (not factual hallucination); benchmark is math reasoning (not RAG QA); we compare against the residual-stream-PH null result (F-7) to test whether attention graphs are the right object.

**Source paper:** [TOHA: Topological Divergence on Attention](https://arxiv.org/abs/2504.10063)
- *What they did:* Computes a topological-divergence metric between prompt-side and response-side attention subgraphs in the RAG setting.
- *Their result:* Per-head divergence values correlate with hallucinations; achieves SOTA-or-competitive results on QA + summarization with minimal labeled data.
- *What we'll do:* Use the same attention-graph object on MATH-500 and test correctness rather than RAG-grounding divergence.
- *Same:* Same conclusion that attention graphs (not residual streams) are the right object for PH-on-LLMs.
- *Differs:* We are not in the RAG setting (no prompt-vs-response divergence); we use a probe directly, not a divergence metric; benchmark is math correctness.

**Cost:** 1 week CPU (attention extraction + PH computation)
**Triggered by:** [HalluZig: Zigzag PH on Attention](https://arxiv.org/abs/2601.01552), [TOHA: Topological Divergence on Attention](https://arxiv.org/abs/2504.10063)
**Depends on:** F-7
**Would update:** F-7
**Trigger condition:** TOHA (2504.10063) and HalluZig (2601.01552) published showing attention-graph PH works.

### P11-FE21 (P11) — [ROI: 6, READY, MEDIUM]

**What:** D-bucket attention entropy at prefill. Extract attention weights at prefill for the 36 D-bucket and 207 A-bucket problems, compute entropy of attention across prompt regions, compare. Mechanism test: D-bucket (F-7) might rely on a single reasoning pathway triggered by a specific prompt feature, making it fragile to sampling perturbation.

**Why:** Akli et al. 2604.24712 Figure 3 shows that on HumanEval (structurally simple) 60-86% of attention concentrates on the description, while LiveCodeBench (structurally rich) distributes attention across description, I/O format, and sample I/O. D-bucket fragility may be the same single-specification fragility expressed at sampling time rather than prompt time.

**Cost:** small GPU (~30 min), depends on whether Stage 2 saved attention
**Triggered by:** [When Prompt Under-Specification Improves Code Correctness](https://arxiv.org/abs/2604.24712)
**Depends on:** F-7
**Would update:** F-7
**Trigger condition:** Akli et al. 2604.24712 attention-distribution analysis.

### P5-FE1 (P5) — [ROI: 6, READY, MEDIUM]

**What:** Run dimensional breathing analysis (temporal PR curve) on Gemma-2-2B and Mistral-7B on the same MATH-500 problems. Confirm breathing is architecture-universal beyond the Qwen/Phi/Llama families already tested.

**Why:** F-1 replicated across Phi-3-mini and Llama-3.2-1B but all three are decoder-only with similar training recipes. Gemma-2 uses a different attention variant and Mistral uses sliding window attention — testing these extends the universality claim substantially.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 4h H100
**Blocked by:** H100 pod time
**Depends on:** F-1
**Would update:** F-1
**Trigger condition:** Whenever H100 pod is next active for any other experiment.

### P11-FE22 (P11) — [ROI: 6, BLOCKED, MEDIUM]

**What:** Abstract-CoT compressed breathing curve. Fine-tune Qwen2.5-1.5B with Abstract-CoT on MATH-500, extract per-token L19 activations during abstract reasoning, compute the temporal PR curve. Compare peak PR, inflation rate, and collapse asymmetry against natural-language CoT. If flat or monotonically decreasing PR, breathing (F-1, F-5) is specific to verbalized reasoning, not reasoning itself — i.e., dimensional inflation is the cost of verbalization.

**Why:** Ramji et al. 2604.22709 achieve 11.6x token compression via discrete latent CoT. Cleanly separates reasoning-as-computation from reasoning-as-verbalization. Heaviest experiment in the queue (training pipeline + GPU week).

**Cost:** ~1 week H100 for fine-tune + extraction
**Blocked by:** Awaiting Abstract-CoT training pipeline release from authors.
**Triggered by:** [Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought](https://arxiv.org/abs/2604.22709)
**Depends on:** F-5, F-1
**Would update:** F-5, F-1
**Trigger condition:** Ramji et al. 2604.22709 release of Abstract-CoT method.

### P8-FE3 (P8) — [ROI: 5, TRIGGERED, MEDIUM]

**What:** Compute representation dispersion (average pairwise cosine distance) per layer on the cached activations. Correlate with CoE magnitude features and with correctness. Test whether dispersion alone (1 feature per layer, 28 total) matches D2H-lite's 58-dim AUROC.

**Why:** Li & Li (2506.24106) showed dispersion strongly predicts perplexity and downstream accuracy across Qwen family. This is exactly what D2H's intra-layer dispersion measures but in a simpler formulation. If 28-dim dispersion matches 58-dim D2H, the extra complexity is unnecessary.

**Source paper:** [Representation Dispersion Predictive Power](https://arxiv.org/abs/2506.24106)
- *What they did:* Computes representation dispersion (mean pairwise cosine distance among hidden vectors) per layer and uses it as an unsupervised quality signal.
- *Their result:* Dispersion correlates strongly and negatively with perplexity across LLaMA + Qwen on Wikipedia / news / scientific abstracts; helps rank examples by difficulty and select layers for kNN-LM.
- *What we'll do:* Compute per-layer dispersion (1 number per layer, 28 total) on cached activations and test against D2H-lite's 58-dim AUROC for correctness.
- *Same:* Same metric (mean pairwise cosine distance) on the same kind of representations.
- *Differs:* Target is supervised correctness (not perplexity); benchmark is MATH-500; we compare against an existing 58-dim D2H baseline rather than to perplexity itself.

**Cost:** 2h CPU on cached activations
**Triggered by:** [Representation Dispersion Predictive Power](https://arxiv.org/abs/2506.24106)
**Depends on:** F-6
**Trigger condition:** Dispersion paper published with Qwen validation.

### P1-FE1 (P1) — [ROI: 5, READY, MEDIUM]

**What:** Revisit initial geometry observations with proper 1024-token labels and participation ratio instead of ad-hoc metrics. Re-extract on the same problems with current infrastructure.

**Why:** P1 observations predate the truncation correction (F-13). Any early geometry finding may have been measuring truncation artifacts. A clean re-run with current methodology would either validate or retire P1 claims.

**Source:** Internal re-validation — no external trigger paper.

**Cost:** 4h CPU on cached activations
**Depends on:** F-13
**Would update:** F-1
**Trigger condition:** Already triggered — F-13 (truncation artifact) invalidated the label distribution P1 used.

### P4-FE1 (P4) — [ROI: 5, READY, MEDIUM]

**What:** Rebuild the ABC-44 feature pipeline with LEACE-style concept erasure for the length direction instead of regression deconfounding. Compare LEACE-deconfounded AUROC against the regression-deconfounded 0.746.

**Why:** P9-E1 used regression deconfounding which only removes linear length dependence. LEACE (2306.03819) provides closed-form perfect linear erasure — it's the gold standard for representation-level deconfounding.

**Source paper:** [LEACE: Linear Concept Erasure](https://arxiv.org/abs/2306.03819)
- *What they did:* Closed-form least-squares concept erasure that provably prevents any linear classifier from recovering a target concept while minimally perturbing the embedding.
- *Their result:* Reduces gender bias in BERT and POS-information leakage across LLMs via 'concept scrubbing' applied at every layer.
- *What we'll do:* Use LEACE to erase length from L19 activations, then refit the ABC-44 correctness probe and compare against the regression-deconfounded 0.746 AUROC.
- *Same:* Both deconfound a target representation against a nuisance direction before downstream prediction.
- *Differs:* Nuisance is generation length (not gender / POS); we use it as a baseline upgrade for an existing regression-deconfounded probe; we evaluate AUROC, not bias metrics.

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

**Source paper:** [Persistence Landscapes (Bubenik)](https://arxiv.org/abs/1207.6437)
- *What they did:* Defines persistence landscapes — a Banach-space vector representation of persistence diagrams that obeys a strong law of large numbers and a central limit theorem, enabling standard hypothesis tests.
- *Their result:* Landscapes are stable, give bottleneck and Wasserstein lower bounds, and unlock parametric statistical inference on PH features.
- *What we'll do:* Replace the 5-stat PH summary with persistence landscapes (and persistence images) and re-run the F-7 Gaussian null test.
- *Same:* Same Rips-PH on the same point clouds; same Gaussian-null control.
- *Differs:* Feature representation is a full vectorization (not 5 lossy stats); the comparison is whether a richer summary recovers signal lost to summarization, not whether PH itself works.

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
| [2402.10528](https://arxiv.org/abs/2402.10528) | Can We Verify Step by Step for Incorrect Answer Detection? | P9-FE4 | TRIGGERED |
| [2406.11717](https://arxiv.org/abs/2406.11717) | Refusal in Language Models Is Mediated by a Single Direction | P11-FE7 | TRIGGERED |
| [2509.10625](https://arxiv.org/abs/2509.10625) | No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes | P11-FE8 | TRIGGERED |
| [2511.08379](https://arxiv.org/abs/2511.08379) | SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models | P11-FE7 | TRIGGERED |
| [2604.15350](https://arxiv.org/abs/2604.15350) | The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason | P11-FE6 | TRIGGERED |
| [2604.19974](https://arxiv.org/abs/2604.19974) | Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders | P10-FE5 | TRIGGERED |
| [2601.01552](https://arxiv.org/abs/2601.01552) | HalluZig: Zigzag PH on Attention | P7-FE2 | TRIGGERED |
| [2604.14084](https://arxiv.org/abs/2604.14084) | TIP: Token Importance in OPD | P10-FE4 | TRIGGERED |
| [2604.16217](https://arxiv.org/abs/2604.16217) | Conformal Prediction via Internal Representations | P10-FE2 | TRIGGERED |
| [2604.18805](https://arxiv.org/abs/2604.18805) | AI scientists produce results without reasoning scientifically | P11-FE15 | READY |
| [2604.22271](https://arxiv.org/abs/2604.22271) | How LLMs Detect and Correct Their Own Errors: Internal Confidence Signals | P11-FE18, P11-FE19 | READY |
| [2604.22709](https://arxiv.org/abs/2604.22709) | Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought | P11-FE22 | BLOCKED |
| [2604.24712](https://arxiv.org/abs/2604.24712) | When Prompt Under-Specification Improves Code Correctness | P11-FE21 | READY |
| [2501.12948](https://arxiv.org/abs/2501.12948) | DeepSeek-R1 | P11-FE5 | TRIGGERED |
| [2501.17148](https://arxiv.org/abs/2501.17148) | AxBench: Steering Benchmark | P10-FE1 | TRIGGERED |
| [2504.05419](https://arxiv.org/abs/2504.05419) | Reasoning Models Know When They're Right | P8-FE2 | TRIGGERED |
| [2504.07986](https://arxiv.org/abs/2504.07986) | SEAL: Steerable Reasoning Calibration | P10-FE1 | TRIGGERED |
| [2504.10063](https://arxiv.org/abs/2504.10063) | TOHA: Topological Divergence on Attention | P7-FE2 | TRIGGERED |
| [2505.18706](https://arxiv.org/abs/2505.18706) | Bias-Only Steering (Sinii) | P10-FE1 | TRIGGERED |
| [2506.00653](https://arxiv.org/abs/2506.00653) | Linear Representation Transferability (LRT) | P6-FE1 | TRIGGERED |
| [2506.18831](https://arxiv.org/abs/2506.18831) | STU-PID Steering | P3-FE1 | TRIGGERED |
| [2506.24106](https://arxiv.org/abs/2506.24106) | Representation Dispersion Predictive Power | P8-FE3 | TRIGGERED |
| [2509.26560](https://arxiv.org/abs/2509.26560) | Estimating Dimensionality of Neural Representations from Finite Samples | P11-FE16 | READY |
| [2510.04309](https://arxiv.org/abs/2510.04309) | PID Steering | P3-FE1 | TRIGGERED |
| [2510.18147](https://arxiv.org/abs/2510.18147) | LLMs Encode Problem Difficulty | P11-FE2 | TRIGGERED |
| [2402.10978](https://arxiv.org/abs/2402.10978) | Conformal Factuality (Mohri & Hashimoto) | P10-FE2 | TRIGGERED |
| [2402.13212](https://arxiv.org/abs/2402.13212) | Soft Self-Consistency | P11-FE4 | READY |
| [2402.18048](https://arxiv.org/abs/2402.18048) | Truthfulness via Local ID | P4-FE2 | TRIGGERED |
| [2404.15255](https://arxiv.org/abs/2404.15255) | How to Use Activation Patching | P11-FE3 | READY |
| [2405.07987](https://arxiv.org/abs/2405.07987) | The Platonic Representation Hypothesis | P11-FE20 | READY |
| [2410.04707](https://arxiv.org/abs/2410.04707) | Learning How Hard to Think (Damani) | P10-FE3 | TRIGGERED |
| [2410.11042](https://arxiv.org/abs/2410.11042) | Persistent Topological Features in LLMs | P7-FE1 | TRIGGERED |
| [2412.01113](https://arxiv.org/abs/2412.01113) | LLMs Faithfully Compute During CoT (Kudo) | P11-FE3 | READY |
| [2306.03341](https://arxiv.org/abs/2306.03341) | Inference-Time Intervention (ITI) | P2-FE1, P10-FE1 | READY, TRIGGERED |
| [2306.03819](https://arxiv.org/abs/2306.03819) | LEACE: Linear Concept Erasure | P4-FE1 | READY |
| [2103.07353](https://arxiv.org/abs/2103.07353) | Fast Zigzag Persistence (Dey-Hou) | P7-FE1 | TRIGGERED |
| [1207.6437](https://arxiv.org/abs/1207.6437) | Persistence Landscapes (Bubenik) | P9-FE2 | READY |
