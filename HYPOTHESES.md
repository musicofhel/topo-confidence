# HYPOTHESES.md — prioritized queue of untested hypotheses

Each hypothesis is actionable: a concrete test, an estimated cost, and a
resolution that would change our understanding. Ranked by how much they'd
change the story.

**Priority key.** HIGH | MEDIUM | LOW | PARKED.

**Numbering continues monotonically. Next ID: H-941.**

## Cost + time summary (at a glance)

Grouped by rung so you can pick a session's worth of work without scrolling.

| Rung | # | Experiment | Compute | $ | Notes |
|---|---|---|---|---|---|
| **Free / local** | H-14 | Qwen breathing headline figure | 30 min CPU | $0 | Script over cached NPZs; validates vs RESEARCH_SUMMARY §3 table. |
| **Free / local** | H-6 | CoE-60 re-baseline at 1024 tok | 30 min CPU | $0 | Validates whether CoE > ABC-44 headline survives label correction. |
| **Cheap GPU** | H-2 | Short-CoT breathing (~20–40 tok) | ~5 min H100 | ~$0.25 | Decides length vs CoT-structure confound. Blocks EXP-042 follow-up. |
| **Cheap GPU** | H-3 | Code-generation breathing (HumanEval) | ~1 H100-hr | ~$2 | Generalizes breathing beyond math. |
| **Cheap GPU** | H-11 | Within-problem K=8 PR spread | ~1 H100-hr | ~$2 | Closes the 4th control gap in RESEARCH_SUMMARY §3. |
| **Big swing** | H-1 | Per-position DoM steering | ~3 H100-days | ~$200 | Most decisive — diagnostic vs lever question for the whole program. |
| **Blocked** | H-4 | Breathing ↔ EoS correspondence | ~3 H100-days | ~$200 | Needs Qwen-2.5-1.5B training checkpoints (HF hub + budget). |
| **Blocked** | H-8 | Prefill direction across training checkpoints | ~3 H100-days | ~$200 | Same checkpoint blocker as H-4. |
| **Free / local** | H-15 | Length-band PR control (refutation) | 20 min CPU | $0 | Self-applied refutation test for F-4. Cached NPZs. |
| **Free / local** | H-16 | Marchenko-Pastur PR bias correction | 1 h CPU | $0 | Methodology check on F-1, F-4. Cached NPZs. |
| **Free / local** | H-17 | RoPE de-rotation of DoM | 1 h CPU | $0 | Could revive E1 steering and retire H-1 if F-3 is mechanical. |
| **Cheap GPU** | H-18 | PANL-equivalent correctability gate | ~30 min H100 | ~$1 | Could explain F-7 (D-bucket) via second-order confidence. |
| ~~Cheap GPU~~ | ~~H-19~~ | ~~C_exact verification routing ★~~ | done | — | **REFUTED (FE19, 2026-06-10):** routing ≯ uniform beyond noise; F-8 stays a predictor, not a recipe. |
| **Free / local** | H-20 | Cross-model kernel alignment fluctuation | 2 h CPU | $0 | Extends F-1 to dynamic Platonic hypothesis. |
| **Cheap GPU** | H-21 | D-bucket prefill attention entropy | ~30 min GPU | ~$1 | Mechanism for F-7 fragility. |
| **Big swing** | H-22 | Abstract-CoT compressed breathing | ~1 week H100 | ~$300 | Tests whether breathing is verbalization or computation. |

**Pod state** (from `STATE.md`):
- `lsuoka6bo8io7m` — STOPPED, volume preserved. Resume with `runpodctl pod
  start lsuoka6bo8io7m`; pip packages live on container disk, reinstall after
  start. Best target for H-2 / H-3 / H-11.
- `y687b9z2dgukcj` — REMOVED (volumeInGb=0 at end of `exp1_cross_model`).
  Re-create from scratch if needed (~15 min).

**One-session suggestions:**
- "Local afternoon": H-14 + H-6 → ~1 hr, $0, closes the headline-figure
  gap and the CoE label-correction question.
- "Cheap-GPU evening": H-2 + H-3 + H-11 → ~2 H100-hr, ~$5, closes the
  length-vs-structure ambiguity and generalizes the claim beyond math.
- "Commit the program": H-1 alone → ~3 H100-days, ~$200, decides whether
  the project ships as "good selective predictor" or "actionable lever."

---

### H-1: Per-position DoM steering moves MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** EXP-037 (prefill AUROC 0.7731) + direction-rotation
analysis (cos(prefill, final) ≈ 0.05).
**Test:** Extract L19 activations at every token position on 200 MATH-500
problems (≈28 positions × full generation × 1536 dims). Fit a separate DoM
direction per position (or per 10-position bucket). During inference, at
each generated token `t`, project the hidden state onto the direction
appropriate to position `t` and inject `α · w_t_norm`. Sweep α over
{−4,−2,−1,0,1,2,4}; negative controls = random direction and negated probe.
Score final answer on held-out 100 problems.
**Requires:** ~1 H100-day for extraction, ~2 H100-day for sweep. ~$200.
Uses Pathway 11 Stage 2 code (1.5B extraction) extended to save per-token
states (already does).
**Would change:** If accuracy moves ≥ 3pp → hidden-state signal is
actionable at inference and the project's steering goal has a concrete
recipe. If it doesn't → 0.77 prefill is diagnostic only, and the program
closes at "good selective predictor, not a lever."
**Blocks:** H-11 (distillation feasibility depends on whether steering
even works with a moving direction).

### H-2: Breathing happens on short CoT reasoning
**Priority:** HIGH
**Motivated by:** EXP-042 no-CoT control was inconclusive (median 2 tokens
— trajectory physically absent). Need a short-but-non-zero trajectory.
**Test:** System prompt "Solve this in one sentence, then give the answer
in \boxed{}" on MATH-500 first-50. Target generation length 20–40 tokens.
Extract L19 per-token at positions {1, 5, 10, 15, 20, final}. Compare PR
shape.
**Requires:** ~5 min H100 + CPU analysis. ~$0.25. Uses
`no_cot_control/run.sh` + new system prompt.
**Would change:** If peak PR ≥ 25 in a 20-40 token span → breathing tracks
generation length. If PR stays ≤ 15 → breathing requires extended
reasoning and is absent on short explanations.
**Blocks:** nothing downstream. This is the "cheap and decisive" test.

### H-3: Breathing replicates on code generation
**Priority:** HIGH
**Motivated by:** Universality has only been tested on MATH-500 and
MATH-500-flavored gibberish. Code is a different reasoning mode.
**Test:** Run Exp 1's extract.py against HumanEval (164 Python problems)
on Qwen-2.5-1.5B. Compare temporal PR shape.
**Requires:** ~15 min H100 + analysis. ~$0.75. Reuses extract.py with new
dataset config.
**Would change:** If code shows the same three-phase structure → breathing
is a general reasoning property. If flat → breathing is math-specific
(possibly driven by the structured "derive and arrive at an answer" shape).

### H-4: Breathing ↔ Sharpness Dimension / EoS correspond formally
**Priority:** HIGH (theoretical impact)
**Motivated by:** Tuci et al. 2604.19740 predicts PH at Gaussian null (our
EXP-026 confirms). Their framework also shows Hessian top eigenvalues
expand-then-collapse around edge of stability. Our PR curve shows
inference-time covariance expand-then-collapse during CoT.
**Test:** Fine-tune Qwen-2.5-1.5B on MATH for 100 steps from a recent
checkpoint. Record (a) covariance PR trajectory at L19 during inference
breathing at each checkpoint, (b) top-Hessian eigenvalue trajectory during
training. Measure Pearson correlation between normalized PR(position) and
normalized λ_1(step) across matched checkpoints.
**Requires:** Fine-tuning access to Qwen-2.5-1.5B. ~3 H100-days. ~$215.
**Would change:** A correlation ≥ 0.6 is evidence of a formal
correspondence between inference-time covariance and training-time
Hessian. It'd be the closest thing to a theoretical contribution the
project has produced.
**Blocks:** nothing, but unlocks a potential paper.

### H-5: Prefill signal encodes problem familiarity vs decomposability
**Priority:** MEDIUM
**Motivated by:** 0.7731 (1.5B) and ~0.876 (7B) prefill AUROCs are
strong enough that the mechanism matters. Is it memorization or
structural prediction?
**Test:** Two-axis regression on the 243 correct + 257 incorrect problems:
(a) familiarity: nearest-neighbor cosine to MATH training set in embedding
space (proxy), (b) decomposability: expected CoT length on a held-out
7B run. Regress prefill DoM score on both + interaction.
**Requires:** Embedding access to MATH training set (~10k problems);
7B-per-problem length extraction (cheap, uses existing 7B cache). Mostly
CPU. ~$5.
**Would change:** Mechanism clarity for the "prefill knows best" finding.
Familiarity-dominated → memorization caveat on the 0.77 / 0.88 numbers.
Decomposability-dominated → the signal is structural and likely
generalizable.

### H-6: CoE-60 still beats single-layer DoM at 1024-tok labels
**Priority:** MEDIUM
**Motivated by:** EXP-020 showed CoE > ABC-44 at 256-tok labels (0.811 vs
0.7961), but CoE was never re-measured at 1024 tokens. The Stage-5 L19 DoM
transfer matches Pathway 9 CoE transfer (0.720 ≈ 0.716), suggesting CoE
isn't *richer*, but within-domain hasn't been re-checked.
**Test:** Extract CoE-60 features from the 1024-tok Stage 2 data (already
cached). Re-run 5-fold OOF at new labels.
**Requires:** CPU only, ~30 min. Uses `pathway8_layerwise/coe_features.py`.
Zero dollar.
**Would change:** If CoE still ≥ 0.80 at 1024 tok, the CoE-beats-ABC-44
headline survives and we have two ~equivalently-strong methods (DoM,
CoE). If CoE ≤ 0.72 (matching DoM), CoE adds nothing — the single
direction is the full story.

### H-7: Density-based outlier detection finds D-bucket
**Priority:** MEDIUM
**Motivated by:** EXP-039 showed per-problem k-NN local PR doesn't detect
the D-bucket (flat across buckets). But the group-level signature exists.
A *density*-aware detector might recover it.
**Test:** Train Local Outlier Factor (LOF) + Isolation Forest on the 500
prefill activations with D labels. Evaluate D-recall at matched
false-positive rate vs the 41.7% Exp 2b baseline.
**Requires:** CPU only, ~15 min. Zero dollar.
**Would change:** If D-recall > 60%, "route D to K=1" becomes a viable
policy and EXP-039's conclusion on multi-signal gating needs revisiting
(not via kNN but via density).

### H-8: The prefill direction is stable across training checkpoints
**Priority:** MEDIUM (mechanism clarification)
**Motivated by:** If "prefill knows best" reflects memorized familiarity
(H-5), the direction should emerge only after training; if it reflects
structural decomposability, it should be present earlier. Also feeds into
H-4.
**Test:** Extract L19 prefill activations on 200 MATH-500 problems using
Qwen-2.5-1.5B-Instruct at checkpoint steps 0, 1k, 10k, 100k, final.
Measure cosine of prefill_DoM at each checkpoint against the final model.
**Requires:** Checkpoint access (HF usually has mid-training).
~1 H100-day. ~$75.
**Would change:** If checkpoint DoMs are nearly parallel → direction is
architectural, not memorized. If they rotate through training → direction
is learned late (supports memorization interpretation).

### H-9: Breathing amplitude predicts problem difficulty
**Priority:** MEDIUM
**Motivated by:** EXP-040 showed Phi-3 and Llama's cross-bucket PR
differences; EXP-038 showed MATH-level stratification affects the 7B
prefill ratio (Level 2: 3.98, Level 5: 1.11). Suggests breathing amplitude
tracks difficulty.
**Test:** Per-problem peak PR (max over positions 30–100) vs MATH-500
level 1–5. Also per-problem peak PR vs whether K=8 majority vote was
correct (bucket membership).
**Requires:** CPU only on existing Stage 2 data, ~15 min. Zero dollar.
**Would change:** If peak PR correlates with level (Spearman > 0.3),
breathing amplitude is a difficulty signal and could be used for
adaptive compute gating independent of DoM score.

### H-10: Breathing pattern changes after fine-tuning
**Priority:** LOW
**Motivated by:** H-4 parent question. Fine-tuning should concentrate
problem-difficulty distribution and either sharpen or flatten the three
phases.
**Test:** SFT Qwen-2.5-1.5B on MATH for 1000 steps. Compare pre/post
breathing curves (7 positions × 500 problems).
**Requires:** ~1 H100-day. ~$75.
**Would change:** If breathing flattens → it's a problem-difficulty
diversity effect that fine-tuning specializes. If it sharpens → fine-tuning
makes the commitment-dynamics more pronounced. Connects to H-4.

### H-11: Big-to-small distillation of 7B self-prediction into 1.5B
**Priority:** LOW (motivator eliminated by EXP-031)
**Motivated by:** The deferred E4 of Pathway 10 v2. Motivator was
"7B residual stream encodes correctness more strongly than 1.5B's" — now
shown false at 1024 tokens (0.717 vs 0.719 transfer, no asymmetry).
Weaker surviving motivator: 7B self-prediction is 0.782 vs 1.5B's 0.719.
**Test:** Fine-tune 1.5B with auxiliary loss matching its L15 projection
onto the 7B L15 DoM direction.
**Requires:** Multi-day H100. ~$400+.
**Would change:** Accuracy bump on 1.5B if transfer works. Negative result
if 7B self-knowledge doesn't distill meaningfully.
**Status:** PARKED unless H-5 shows decomposability-signal in prefill
(would suggest transferable structure).

### H-12: Semantic-entropy probes (label-free) replace supervised DoM for refusal
**Priority:** LOW
**Motivated by:** SEPs paper (2406.15927) gives a label-free alternative
to the DoM probe. Would eliminate the 243-label dependency in E3.
**Test:** Sample K=4 at T=0.7, compute semantic entropy. Compare to
supervised prefill DoM for selective-prediction risk-coverage curve.
**Requires:** Uses EXP-033 K=8 cache. CPU-only analysis, ~1 h. Zero
dollar.
**Would change:** If SEP gives comparable coverage-accuracy → refusal is
label-free; if worse → supervised labels carry meaningful extra signal.

### H-13: Head-level attribution of prefill vs final-token circuits
**Priority:** LOW
**Motivated by:** cos(prefill, final) ≈ 0.05 is consistent with different
attention heads contributing to each probe. If the contributing head sets
are disjoint, we have evidence for two circuits.
**Test:** For the trained prefill probe and final-token probe (same on
the 1.5B, L19 DoM), compute each attention head's contribution via
direct logit attribution. Check whether the top-K contributing heads are
disjoint across the two probes.
**Requires:** Access to attention outputs (cached in Stage 2 for 1.5B).
CPU only, ~2 h. Zero dollar.
**Would change:** Mechanism clarity on the orthogonality finding. If
heads are disjoint → two-circuits story. If heads overlap but
coefficients differ → same heads, different readouts.

### H-14: Generate the missing `breathing_temporal_qwen15b_7b.png` headline figure
**Priority:** HIGH (headline-communicability, user-reprioritized 2026-04-24)
**Motivated by:** User's goal list called it out as the headline figure
for the whole project; it doesn't exist as a PNG, only as a RESEARCH_SUMMARY
inlined table. Phi-3 and Llama cross-arch analogs already exist — the
Qwen figure that started the project is the one we lack.
**Test:** Matplotlib script over `pathway8_layerwise/data/math500/` and
`pathway11_h100/data/math500_7b/` computing per-position cohort PR (L19)
at {1, 10, 25, 50, 100, 200, final} for both scales, plotting both curves
on one log-x axes. Validate against the RESEARCH_SUMMARY §3 inlined table
(1.5B peak ~67 at pos 50, final ~8; 7B peak ~88 at pos 50, final ~6).
**Requires:** CPU only, ~20 min writing script + ~10 min rendering.
Zero dollar.
**Would change:** Nothing quantitative. Makes the headline communicable
in a single image — it's the one figure most people will look at first.

---


### H-15: Length-band PR control validates asymmetric-collapse claim
**Priority:** HIGH (refutation test, user-prioritized 2026-04-28)
**Motivated by:** Ríos-García 2604.18805 (agents ignore disconfirming
evidence in 68% of traces). Self-applied refutation test for F-4.
**Test:** Split 500 MATH-500 problems by generation length (<25th,
25-75th, >75th percentile). Compute final-token PR per band, separately
for correct vs incorrect. Uses cached Stage 2 NPZs.
**Requires:** CPU only, ~20 min. Zero dollar.
**Would change:** If correct/incorrect PR ratio approaches 1.0 within
length-matched bands, asymmetric collapse is a length-mixing artifact and
F-4 retracts. If ratio holds within each band, F-4 is corroborated against
the strongest available within-data refutation.
**Blocks:** nothing. Cheap, decisive validity check.

### H-16: Marchenko-Pastur bias correction preserves breathing magnitude
**Priority:** HIGH (methodological, user-prioritized 2026-04-28)
**Motivated by:** Chun et al. 2509.26560. Our cross-model P/Q ratios
(Qwen 1.5B 0.33, Qwen 7B 0.14, Llama 0.24, Phi-3 0.16) are all in the
bias-significant regime; cross-model peak-PR comparisons at fixed n=500
are the scenario the paper warns against.
**Test:** Implement bias-corrected PR estimator (Section 4 of the paper
or their reference code). Recompute (a) temporal PR curve for Qwen 1.5B
and 7B, (b) correct vs incorrect PR at prefill and final token,
(c) cross-model peak-PR comparison. Report naive and corrected
side-by-side.
**Requires:** CPU only, ~1 h. Uses cached Stage 2 NPZs.
**Would change:** If correct-collapses-harder (F-4) and cross-arch
breathing magnitude (F-1) survive correction, the headline numbers are
robust. If correction shifts ratios materially, narrative claims need
re-stating with the corrected estimator.
**Blocks:** nothing.

### H-17: DoM rotation during generation is RoPE-mechanical, not computed
**Priority:** HIGH (could revive E1 steering)
**Motivated by:** Puranik (Jane Street) shows positional encodings are
matrix groups exp(M·tau); RoPE applies deterministic constant-frequency
rotations. F-3 (cos(prefill, final) ≈ 0.046) might be a RoPE consequence
rather than a computational orthogonality.
**Test:** For 1.5B per-token L19 activations, undo RoPE rotation at each
position (rotation angles deterministic from model config), recompute
temporal DoM AUROC curve and cosine-to-final table.
**Requires:** CPU only, ~1 h. Uses cached per-token L19 activations and
the model config.
**Would change:** If cos(DoM_t, DoM_final) rises above 0.8 after
de-rotation, F-3 becomes a positional-encoding artifact and the
direction-rotation argument against E1 steering collapses — H-1
per-position bank loses its motivation, and a single un-rotated DoM is
sufficient for steering. Partial explanation (cos 0.2 to 0.6) still
informative about how much of F-3 is mechanical.
**Blocks:** H-1 (per-position steering bank) becomes unnecessary if this
confirms; reframes if it partially confirms.

### H-18: Post-answer-newline activation predicts B/D-bucket membership
**Priority:** HIGH
**Motivated by:** Kumaran et al. 2604.22271 (PANL = orthogonal
second-order confidence signal, AUROC 0.986). Our prefill DoM is the
pre-hoc analog. The B/D-bucket question (F-7) — recoverable vs
pathological — sits at mid-confidence where prefill gating fails (Exp 2).
**Test:** In K=1 greedy generations, locate the post-answer-newline
token per problem, extract L19 activations at that position. Compute DoM
AUROC predicting (a) K=1 correctness, (b) K=8 majority correctness,
(c) B-bucket membership (K=1 wrong, K=8 right), (d) D-bucket membership
(K=1 right, K=8 wrong). Compare against prefill (F-2) and final-token
AUROC.
**Requires:** Model loaded (2060 sufficient), ~30 min. Uses cached
per-token trajectories from Stage 2.
**Would change:** If PANL-equivalent predicts B-bucket > prefill does,
post-hoc correctability gating becomes feasible — route only genuinely
recoverable problems to K=8. F-7 gains a mechanism.
**Blocks:** H-19 (verification routing builds on this signal).

### H-19: C_exact verification routing beats C_infer scaling at matched compute
**STATUS: REFUTED (P11-FE19, 2026-06-10).** Verification routing does NOT beat the
achievable uniform-K frontier (upper convex hull = random K=1/K=8 routing) beyond noise
(best routed point +0.1 SE, n=500). The post-hoc PANL "K=1-wrong" probe peaks at L22 AUROC
0.7555 — *below* the pre-hoc prefill-DoM 0.7731, so the C_exact signal is weaker than the
C_infer one it was meant to beat. Verbalized self-verdict is anti-informative (AUROC 0.480)
and verify-then-correct HURTS at 1.5B (0.474 vs K=1 0.486). F-8 stays predict-difficulty-
at-prefill, NOT verify-and-route. See `pathway11_h100/verify_route/results/verdict.json`
and brief `result-2026-06-10-P11-FE19.md`. (Original hypothesis below, for the record.)

**Priority:** CRITICAL (Desktop's highest-leverage call, user-prioritized 2026-04-28)
**Motivated by:** Rybin compute-allocation framework
(C_train / C_infer / C_exact decomposition) + Kumaran 2604.22271 PANL
signal. E2 prefill-gated compute (pure C_infer) failed because B-bucket
lives at mid-confidence; the right axis is C_infer ↔ C_exact.
**Test:** For 500 MATH-500 problems: take K=1 greedy answer, prompt the
model to verify its own answer (verify-then-correct paradigm), extract
PANL-equivalent activation, use as routing signal. Route
verification-failed problems to K=8 sampling, keep verification-passed at
K=1. Compare against (a) pure C_infer (uniform K=8), (b) pure C_exact
(verify-then-correct, no sampling) at matched compute. Total avg K ≈ 3-4.
**Requires:** Model loaded (2060 sufficient), ~30 min plus verification
pass per problem.
**Would change:** If C_exact routing beats C_infer routing at matched
compute, the selective-prediction story (F-8) shifts from predict-
difficulty-at-prefill to verify-after-generation-and-route-failures. This
is the result that converts the program from "good selective predictor"
to "actionable inference recipe."
**Blocks:** nothing downstream, but it's the highest-impact lever the
program has access to without GPU-week investment.

### H-20: Cross-model kernel alignment fluctuates during inference
**Priority:** MEDIUM
**Motivated by:** Huh et al. 2405.07987 (Platonic Representation
Hypothesis). Our F-1 cross-arch breathing universality is evidence for
the hypothesis applied to inference-time dynamics — not just trained
representations but their temporal evolution converges.
**Test:** Compute mutual k-NN alignment between all 6 model pairs (Qwen
1.5B, Qwen 7B, Phi-3, Llama) at each of the 7 temporal positions using
cached 2/3-depth activations. Predicts: alignment peaks at low-PR
moments (prefill, final token), reaches min at mid-generation peak PR.
**Requires:** CPU only, ~2 h, all data cached.
**Would change:** If alignment fluctuates as predicted, F-1 extends
from "all transformers breathe" to "all transformers explore differently
but compress to the same place" — a temporal version of the Platonic
hypothesis. If alignment is flat or anti-correlated with PR, F-1 is
universality of *shape* but not of *content* (each model breathes through
its own subspace).
**Blocks:** nothing.

### H-21: D-bucket has narrower prefill attention than A-bucket
**Priority:** MEDIUM
**Motivated by:** Akli et al. 2604.24712 single-specification fragility:
HumanEval prompts concentrate 60-86% attention on the description while
LiveCodeBench distributes across description, I/O format, sample I/O.
D-bucket fragility (F-7) may be the same effect at sampling time.
**Test:** Extract attention weights at prefill for the 36 D-bucket and
207 A-bucket problems, compute entropy of attention across prompt
regions, compare distributions.
**Requires:** Model loaded (small GPU, ~30 min), or approximate from
cached Stage 2 attention if saved.
**Would change:** If D-bucket prefill attention entropy is lower than
A-bucket, F-7 gains a mechanistic explanation: D-bucket relies on a
single prompt feature triggering a single reasoning pathway, fragile to
sampling perturbation. If equivalent, F-7 fragility lives elsewhere
(generation-time stochasticity, not prompt-side concentration).
**Blocks:** nothing.

### H-22: Abstract-CoT shows compressed or qualitatively different breathing
**Priority:** MEDIUM (heaviest experiment in the queue)
**Motivated by:** Ramji et al. 2604.22709 (Abstract Chain-of-Thought,
11.6x token compression at comparable accuracy). Cleanly separates
reasoning-as-computation from reasoning-as-verbalization.
**Test:** Fine-tune Qwen2.5-1.5B with Abstract-CoT on MATH-500. Extract
per-token L19 activations during abstract reasoning. Compute temporal PR
curve. Compare peak PR, inflation rate, and collapse asymmetry against
natural-language CoT.
**Requires:** GPU (training pipeline release pending from authors),
~1 week H100. ~$300.
**Would change:** If Abstract-CoT shows flat or monotonically decreasing
PR, breathing (F-1, F-5) is specific to verbalized reasoning, not
reasoning itself — dimensional inflation is the computational cost of
verbalization, not of computation. If breathing replicates with
compressed shape, breathing is a property of sequential autoregressive
generation regardless of vocabulary.
**Blocks:** nothing. Long-tail experiment.

---


### H-23: Persistence-landscape two-sample test rejects F-10's Gaussian null
**Priority:** HIGH
**Motivated by:** 1207.6437 + F-10, EXP-026, EXP-018
**Test:** Compute λ_1…λ_5 persistence landscapes per problem on the cached
Pathway 8 layer-wise PH outputs (trained-stream and matched Gaussian baseline);
run an L²-landscape permutation two-sample test per layer (B = 10 000
permutations). 2h CPU.
**Requires:** CPU only; existing cached PH outputs (Pathway 8 NPZs); `gudhi`
landscape module.
**Would change:** If p < 0.01 in ≥ 3 layers, F-10's "PH at Gaussian null" is
demoted to "scalar PH summaries are at Gaussian null" and the PH graveyard
entry in PROJECT_RECORD §1d is reopened. If null replicates under the more
powerful landscape statistic, F-10 is *strengthened* (it's not a power
artefact).
**Blocks:** would-block H-24 (landscape AUROC) if landscapes can't separate
trained from Gaussian — no point trying to classify correctness if the
features can't even separate the two distributions.

### H-24: PH-landscape features at L19 beat 0.7731 prefill-DoM AUROC
**Priority:** MEDIUM
**Motivated by:** 1207.6437 + F-9, F-2
**Test:** Vectorise λ_1…λ_5 on a global (b, d) grid for the cached Qwen-1.5B
L19 PH outputs across the 500 MATH-500 problems; fit OOF 5-fold logistic
regression vs K=1 correctness; report AUROC and ΔAUROC vs single-layer L19
DoM. 30min CPU.
**Requires:** CPU only; cached L19 PH outputs from Pathway 8 / Pathway 11.
**Would change:** ΔAUROC > 0 refutes F-9's "CoE-60 redundant with single-layer
DoM" framing (the redundancy was conditional on scalar PH features) and
revives PH as a candidate residual-stream featurisation. ΔAUROC ≤ 0
strengthens F-9 — DoM is the relevant primitive even under a richer PH
summary.
**Blocks:** nothing.

---


### H-25: Labelled-support persistent homology of the L19 prefill cloud is non-trivial and explains the F-2 AUROC ceiling
**Priority:** HIGH
**Motivated by:** 1802.04443 (Guss & Salakhutdinov 2018) + F-2 + F-10
**Test:** Compute β₀, β₁ of the L19 prefill activation supports for the
correct (243) and incorrect (257) MATH-500 splits separately, using
ripser on ≤500-sample subsamples (optionally LLE→R³ for tractability),
with shuffled-label and Gaussian-rank-matched baselines. Compare
persistence-barcode summary statistics. ~30 min CPU on cached
`pathway11_h100/prefill_gated_compute/` extractions.
**Requires:** CPU only; cached prefill activations from P11; ripser /
giotto-tda; matched-rank Gaussian baseline tooling already used in EXP-026.
**Would change:** If H(X⁺_correct) has β_p ≠ Gaussian for any p, F-10's
null result needs scoping to "null on aggregate clouds, non-trivial on
labelled supports," and F-2's AUROC ceiling gains a structural
explanation via Theorem 3.1 of the paper. If null on both supports too,
the topological-program door is now closed for this regime, not just for
the aggregate.
**Blocks:** H-26.

### H-26: A non-linear probe over L19 features crosses a topological phase transition above the linear DoM AUROC
**Priority:** MEDIUM
**Motivated by:** 1802.04443 (Section 3.2 phase-transition results) +
F-2 + historical kill of XGBoost-over-ABC-44
**Test:** Sweep MLP probe widths h₀ ∈ {2, 4, 8, 16, 32, 64, 128} on
cached L19 features with the same 5-fold OOF split used for F-2's
0.7731. Plot AUROC vs h₀. Look for a discontinuous jump (paper's
prediction) vs smooth flat (Bayes-noise prediction).
**Requires:** ~2h CPU on cached features.
**Would change:** If a phase transition appears, the historical
conclusion "signal is linear" gets scoped to "linear is enough to
recover most of the signal but not the homologically-distinct subset."
If no transition, the topological-ceiling story for F-2 is refuted and
the gap is Bayes / noise.
**Blocks:** nothing.

---


### H-27: Random k-dim subspaces of L19 prefill activations match supervised L19 DoM AUROC at low k
**Priority:** HIGH
**Motivated by:** 1804.08838 + F-2
**Test:** Project cached `pathway11_h100/prefill_gated_compute` L19 prefill activations through Gaussian random P ∈ ℝ^(1536×k) for k ∈ {1, 5, 50, 500}; fit 5-fold OOF logistic regression; compare AUROC against the 0.7731 baseline. Identify d_probe = smallest k achieving 90% of supervised AUROC. ~20 min CPU.
**Requires:** CPU, cached 1.5B L19 prefill activations from P11.
**Would change:** Confirm (k=50 ≈ 0.74) → reframe F-2 from "L19 DoM is special" to "L19 prefill activations contain a low-d subspace and supervised DoM is one of many directions that find it." Reject (only k=1536 reaches 0.74) → strengthens F-2's "this direction is privileged" claim.
**Blocks:** nothing (cheap to run independently)

### H-28: Per-problem breathing amplitude is model-specific, not dataset-determined
**Priority:** HIGH
**Motivated by:** 1804.08838 + F-7 + H-9
**Test:** Compute per-problem prefill L19 PR (and L19 DoM score) on 1.5B and 7B for the same 500 MATH-500 problems from P11 caches. Spearman-rank-correlate. ~10 min CPU.
**Requires:** CPU, cached 1.5B + 7B prefill L19 activations.
**Would change:** Confirm cross-model rank correlation r > 0.5 → "MATH-500 has dataset-level difficulty axes that both models track" supports F-7. Reject (r < 0.3) → F-7's collective geometric signature is a model-specific artifact, not a problem-determined property; H-9 weakens.
**Blocks:** H-9 reformulation if rejected.

### H-29: Low-d random-subspace steering at L19 outperforms single-direction DoM steering
**Priority:** MEDIUM
**Motivated by:** 1804.08838 + H-1 + EXP-028 (E1 fixed-vector failure)
**Test:** Sample random k-dim subspace of 1536-d residual at L19 for k ∈ {1, 10, 50}; train a steering vector in this subspace; generate K=1 MATH-500 on Qwen-2.5-1.5B with steered activations across magnitude sweep; measure accuracy delta. ~4h H100.
**Requires:** H100, Qwen-2.5-1.5B, cached MATH-500 prompts.
**Would change:** Confirm (k=10 lifts ≥1pp, k=1 flat) → H-1's "single-direction DoM" mechanism is wrong; rewrite as "low-rank steering manifold." Reject (all k flat) → corroborates EXP-028's E1 result and pushes H-1 deeper into PARKED.
**Blocks:** H-1 mechanism reformulation if confirmed.

---


### H-30: Mapper-cluster membership on prefill L19 activations matches supervised DoM AUROC
**Priority:** HIGH
**Motivated by:** 1811.00852 + F-2
**Test:** Run KeplerMapper (VarNormEuclidean + PCA-2 lens, resolution=50, gain=3) on `pathway11_h100/prefill_gated_compute/` cached prefill L19 activations for the 500 MATH-500 problems on Qwen-2.5-1.5B. Compute purity AUROC of cluster-membership vs 1.5B K=1 correctness with 5-fold cross-validation. Compare to F-2's 0.7731.
**Requires:** CPU only; uses cached NPZ. ~30min.
**Would change:** If Mapper-cluster AUROC ≥ 0.74 (within noise of 0.7731), F-2's "supervised DoM direction is the channel" framing weakens — the predictive content is a property of local activation density, not of any single linear axis. If Mapper-cluster AUROC ≪ 0.7731, F-2 is strengthened.
**Blocks:** H-31, H-32

### H-31: D-bucket forms a single coherent Mapper cluster, not one-of-many incorrect clusters
**Priority:** MEDIUM
**Motivated by:** 1811.00852 + F-7
**Test:** From the Mapper graph in H-30, identify all nodes whose contained problems are >70% incorrect. Check whether the 36 D-bucket problems (K=1 right, K=8 wrong) concentrate in a single such node-group disjoint from other incorrect-leaning groups, or whether they scatter across multiple incorrect clusters alongside other incorrect-bucket types.
**Requires:** CPU only; same Mapper output as H-30. ~10min.
**Would change:** If D-bucket forms one coherent group, F-7's "distinctive collective signature" is corroborated. If D-bucket scatters across several Goldfarb-style failure clusters, F-7's specificity is refuted — D-bucket would just be one of several incorrect-leaning groups any TDA on activations surfaces, and its "distinctiveness" is a labeling artifact.
**Blocks:** nothing

### H-32: Mapper exposes correctness-correlated structure in residuals where PH found Gaussian null
**Priority:** MEDIUM
**Motivated by:** 1811.00852 + F-10
**Test:** Apply KeplerMapper to the residual-stream cache that produced F-10's null result (per-layer or per-trajectory, same as PH input). Color nodes by correctness. Measure cluster→correctness association (purity AUROC, χ² of node-correctness contingency). Compare to F-10's PH-Betti null.
**Requires:** CPU only; same residual cache as F-10. ~1h.
**Would change:** If Mapper finds correctness-correlated cluster structure (purity AUROC > 0.6, χ² p < 0.01), F-10's interpretation must be narrowed: trained residuals are at the Gaussian null *for PH-Betti* but not for local-connectivity TDA. If Mapper is also at chance, F-10's "no topology" reading is robustly corroborated.
**Blocks:** nothing

---


### H-33: Weight-graph neural persistence at L19 of Qwen-2.5-1.5B exceeds Gaussian null and rank-correlates with the layer's prefill-DoM AUROC
**Priority:** HIGH
**Motivated by:** 1812.09764 + F-10 + F-2
**Test:** Implement Algorithm 1 of Rieck et al. on Qwen-2.5-1.5B per-layer MLP and attention weight matrices (28 layers × {W_up, W_down, W_q, W_k, W_v, W_o}). Compare NP against Gaussian and uniform nulls of the same shape (100 nulls per matrix, as in Figure 2 of the paper). Then plot per-layer NP vs per-layer prefill-DoM AUROC from `pathway11_h100/prefill_gated_compute/results.json`. ~1.5h CPU total.
**Requires:** CPU only; public Qwen-2.5-1.5B weights (already cached); existing per-layer DoM-AUROC table.
**Would change:** On *confirm with strong gap*, F-10 must be sharpened to "PH on activation point clouds at null" rather than "PH at null"; on *confirm with strong rank-correlation to DoM*, F-2 is partially confounded with weight-graph complexity and the 0.7731 AUROC needs a NP-controlled re-evaluation. On *reject* (NP at null on Qwen weights), F-10's null generalizes from activations to weights and the project's no-PH stance hardens.
**Blocks:** H-34 (filtration-design re-test of F-10) is downstream — only worth running if H-33 confirms a non-null gap at L19.

### H-34: Magnitude-filtered (Rieck-style) PH on activation graphs recovers a non-null signal that distance-filtered PH on activation point clouds (F-10) missed
**Priority:** MEDIUM
**Motivated by:** 1812.09764 + F-10
**Test:** Build per-token activation graphs at L19 prefill where edge weights are attention scores between token pairs; filter by descending edge weight (Rieck-style), compute 0-dim persistent homology, take p=2 norm of the diagram. Compute this scalar per problem on the 36 D-bucket and 207 A-bucket problems (1024-tok labels). Test for a significant difference in distribution.
**Requires:** Cached prefill activations + attention weights (Stage 2 of P11 if saved); ~4h CPU; ~200 LOC for graph-construction + union-find filtration.
**Would change:** On *confirm*, F-10's null is filtration-specific not regime-specific, and Pathway 7's NO-GO ("non-Euclidean PH 0.774 < 0.796") may be revivable with a magnitude filtration. On *reject*, F-10 generalizes across filtration choices, hardening the activation-side null.
**Blocks:** Pathway-7 revival.

---


### H-35: F-10's PH-null result is summary-statistic-bound, not topology-bound
**Priority:** MEDIUM
**Motivated by:** 1906.00722 (Moor et al., differentiable-PH loss on edge-level pairwise distances) + EXP-026 (F-10).
**Test:** Replace the 5 scalar PH summaries used in EXP-026 with the |π|-length edge-distance vector from the 0-dim persistence pairing. Refit logistic regression on the same cached `pathway9/results/exp3a_gaussian_null_v2` real-vs-null residuals; report AUROC delta. Also compute bottleneck distance `d_b(D_real, D_null)` directly. ~1-2h CPU.
**Requires:** CPU (gudhi or ripser-py for full diagram; scikit-learn for LR), cached EXP-026 residuals.
**Would change:** If real − null AUROC ≥ 0.03 OR bottleneck-distance ratio (real/null) ≥ 1.5, F-10 weakens from "PH is null-matched" to "PH-summary scalars are null-matched". If null still ties at edge-vector level, F-10's null robustness is *strengthened* (it survives a richer PH summarization).
**Blocks:** further H-N hypotheses about non-trivial PH structure on residual streams.

### H-36: Prefill/final-token DoM coordinate-orthogonality is rotation-only, not topology-different
**Priority:** MEDIUM
**Motivated by:** 1906.00722 (rotation-invariance of distance-based topology preservation) + F-3.
**Test:** On cached `pathway11_h100/prefill_gated_compute` NPZs, extract (correct, incorrect) sub-clouds at L19 prefill and L19 final. Compute 0-dim persistence diagrams per group per stage; report `d_b(D_prefill_correct, D_final_correct)` and `d_b(D_prefill_incorrect, D_final_incorrect)`. ~1h CPU.
**Requires:** CPU, gudhi, cached pathway11_h100 NPZs.
**Would change:** If both bottleneck distances are small (< 0.05 normalized) while `cos(DoM_prefill, DoM_final) = 0.046`, F-3's "orthogonal therefore different mechanism" reading is *wrong* — same topology, different rotation. If both bottleneck distances are large, F-3 stands and gains a topological corroboration.
**Blocks:** any mechanism-level claim about prefill vs final-token correctness signals (currently relevant for H-13 head-level attribution).

### H-37: CoE-60 redundancy with L19 DoM is single-scale; multi-scale evaluation reveals scale-specific signal
**Priority:** LOW
**Motivated by:** 1906.00722 (KL_σ at σ ∈ {0.01, 0.1, 1.0}) + F-9.
**Test:** Compute KL_σ density divergence between (correct, incorrect) classes at σ ∈ {0.01, 0.1, 1.0, 10.0} for two feature families: (a) CoE-60 trajectory vector (Pathway 9), (b) L19 DoM scalar (Pathway 11). Report KL_σ profile per family. ~30 min CPU.
**Requires:** CPU, cached CoE-60 features and DoM scores.
**Would change:** If CoE-60 wins at any extremal σ by ≥ 0.05 KL units, F-9 needs a "redundant-at-AUROC-scale" qualifier — a 60-D trajectory feature carries genuine multi-scale information not captured at one classifier head, even when AUROCs match.
**Blocks:** nothing.

---


### H-38: Per-token PPLM-style gradient steering on F-2's probe outperforms static-DoM steering at matched fluency
**Priority:** HIGH
**Motivated by:** 1912.02164 + F-2, F-3
**Test:** Add a fourth arm to P10-FE1: PPLM with α ∈ {0.02, 0.04}, m=3 inner steps, λ_KL=0.01, γ_gm ∈ {0.8, 0.9}, F-2's L19 logistic probe as p(a|H). Hold out 100 MATH-500 problems; compare K=1 accuracy and mean perplexity to probe-weight / mass-mean / trained-bias arms at matched perplexity budget.
**Requires:** H100 pod; F-2 probe; cached pathway11 prefill NPZs; ~1 day H100 (4 sweep cells × ~6h each).
**Would change:** If PPLM-arm > best static arm by ≥1% absolute MATH-500 accuracy at matched perplexity, F-3's "static direction is fine" framing is wrong and the project pivots to gradient-field characterization. If PPLM-arm ≤ static arms, F-3 holds and we can drop the gradient arm from future steering work.
**Blocks:** nothing.

### H-39: F-2's probe gradient is locally informative on the F-8-abstained decile
**Priority:** HIGH
**Motivated by:** 1912.02164 + F-2, F-8
**Test:** P11-FE40 — offline one-step gradient perturbation on cached prefill activations of the bottom-decile (50 MATH-500 problems). Compute Δ H_19 = α · ∇_{H_19} log σ(probe(H_19)), re-run cached forward pass L20→L27, measure probe AUROC pre/post.
**Requires:** 4h CPU on cached NPZs; no generation needed.
**Would change:** Confirm: full PPLM generation experiment (H-38) is justified. Reject: F-2's probe is a global-direction object and gradient-based steering is dead.
**Blocks:** H-38 (gating sanity check).

### H-40: Best-of-K reranking by F-2's probe recovers D-bucket errors
**Priority:** MEDIUM
**Motivated by:** 1912.02164 + F-7, F-8
**Test:** P11-FE41 — for problems with cached K=8 samples, score each sample's prefill by F-2's probe, take argmax; report K=1-greedy vs best-of-8-by-probe accuracy stratified by F-7's bucket.
**Requires:** 1h CPU on cached samples (only valid where K=8 cache exists).
**Would change:** If best-of-8-by-probe lifts D-bucket above K=1-greedy, the prefill probe is informative *across* siblings of the same problem (not just across problems), strengthening F-7 and giving us the cheapest selective-prediction lever. If it does not, F-7's "collective signature, not per-problem" caveat tightens.
**Blocks:** nothing.

---


### H-41: Head-level probe ensemble beats single-layer L19 DoM on prefill correctness
**Priority:** HIGH
**Motivated by:** 2025.35346 + F-2
**Test:** Train 448 logistic-regression probes (one per (layer, head)) on the
cached Qwen-1.5B Stage-2 prefill activations, predicting K=1 MATH-500
correctness. Compare two ensemble AUROCs against F-2's 0.7731 layer-mean L19
baseline on the same OOF 5-fold split: (a) top-K=16 vote, (b) SMITIN
soft-weighted (c=3) sum of probe sigmoids. Est: 30 min CPU.
**Requires:** CPU; cached Stage-2 NPZs (already exist).
**Would change:** If best ensemble AUROC ≥ 0.79, F-2 is restated as "L19 DoM
captures one *projection* of the correctness signal", and F-9's "CoE-60 is
redundant with single-layer DoM" must be retested against the head ensemble
(CoE-60 may genuinely add over single-layer but be subsumed by head-level).
If ensemble ≤ 0.78, F-2 strengthens — single-layer DoM is the correctness
sufficient statistic at the prefill stage.
**Blocks:** H-13 (head-level attribution gets actual numbers from this
experiment), H-42 (the controller hypothesis below).

### H-42: DoM steering on math reasoning requires SMITIN-style self-monitoring, not constant α
**Priority:** CRITICAL
**Motivated by:** 2025.35346 + H-1
**Test:** Two-arm comparison on Qwen-1.5B MATH-500 K=1: Arm A applies prefill
L19 DoM as a constant-α residual add at every generation token; Arm B applies
the same direction sparsely (every 5 tokens) with the SMITIN three-rule
controller (decay if probe confidence below τ = median(acc) − std(acc), reset
when stuck at zero, zero out when above τ). Sweep α ∈ {2, 5, 10}. Baseline
48.6%. Est: H100 day.
**Requires:** H100 pod; SMITIN controller code ported (~50 lines around
existing forward hooks).
**Would change:** If Arm A accuracy collapses below 30% at α = 5 while Arm B
preserves or improves on the 48.6% baseline, H-1 must be rewritten as a
controller hypothesis — the *direction* is necessary but a *static add* is
the wrong intervention. If both arms preserve baseline accuracy, the
self-monitoring machinery is unnecessary and H-1 in its current static-vector
form is admitted to the queue. If both collapse, the prefill direction is not
causally usable for steering and H-1 dies.
**Blocks:** H-1 in its current form (this experiment decides which form
H-1 should take), and any future per-position-steering FE.

---


### H-43: Prefill DoM signal is in the anticausal direction (correctness → prefill state)
**Priority:** HIGH
**Motivated by:** 2102.11107 §VII.A + F-2 + H-1 open status
**Test:** Train an unsupervised contrastive head on unlabeled cached prefill
L19 activations (500 MATH-500 problems × 4096 dims); evaluate on F-2's
correctness-prediction task on a 5-fold held-out split. If SSL pretraining
strictly improves DoM AUROC over supervised-only (0.7731), Schölkopf's
§VII.A diagnostic places the probe in the anticausal direction. If no
improvement, the probe is causal-direction. (~1 day CPU.)
**Requires:** Cached `pathway11_h100/prefill_gated_compute/*.npz`; small
contrastive-pretrain implementation; existing 5-fold split from F-2.
**Would change:** Confirm ⇒ prefill DoM is a *readout* of an upstream
correctness commitment (not a cause); H-1 / P10-FE1 / P11-FE3 reframed as
intervention-on-readout (likely null result expected). Reject ⇒ probe is
causal-direction; H-1 steering more likely to move accuracy. Either outcome
sharply narrows the next experiments.
**Blocks:** P11-FE3 priority — if H-43 confirms anticausal direction,
P11-FE3 patching cost is hard to justify for a null result.

### H-44: D-bucket membership is interventionally stable across decoding temperature (SMS test)
**Priority:** MEDIUM
**Motivated by:** 2102.11107 §VI (coarse-grained causal variables) + F-7
**Test:** Re-run K=1 + K=8 MATH-500 protocol at temperature 0.3 (canonical
F-7 numbers are at temp 0.0); compute Jaccard(D@temp=0.0, D@temp=0.3).
Threshold: Jaccard > 0.7 supports F-7 as a real coarse-grained causal
variable; Jaccard 0.4–0.7 = mixed; Jaccard < 0.4 = fitting artifact.
(~4h H100.)
**Requires:** H100 pod for generation; existing K=1 + K=8 pipeline
parametrized by temperature.
**Would change:** Low Jaccard ⇒ F-7's "distinctive collective signature"
narrative needs scoping to a single decoding regime; downstream claims
about D-bucket as a route for selective routing (P11-FE19) need
temperature-marginalization. High Jaccard ⇒ F-7 strengthens; D-bucket
becomes a candidate coarse-grained causal variable in Schölkopf's sense.
**Blocks:** nothing.

### H-45: Cross-architecture replication of breathing — Llama-3 / Mistral / GPT-2 XL
**Priority:** MEDIUM
**Motivated by:** 2102.11107 §VI Problem 2 (transferable mechanisms) + F-1
universality framing
**Test:** Replicate F-1's temporal PR curve protocol on a non-Qwen model
family — Llama-3 8B, Mistral 7B, GPT-2 XL — on 200 MATH-500 problems.
Measure peak PR, inflation rate, and asymmetric-collapse signature against
Qwen-2.5 1.5B / 7B baselines. (~1 day H100 across 3 models.)
**Requires:** H100 pod; HF model checkpoints; existing Stage 2 PR pipeline
generalized to non-Qwen tokenizers.
**Would change:** Replication ⇒ F-1's universality claim survives
cross-architecture interventional check (Schölkopf-style). Failure (no
breathing or qualitatively different signature) ⇒ F-1 narrows to Qwen-tradition
decoder-only models trained on overlapping corpora; the "universal" framing
is downgraded.
**Blocks:** nothing; supersedes weaker H-3 (code generation) as the
strongest external test of F-1.

### H-46: SMS-sparseness of prefill_DoM under prompt perturbation
**Priority:** MEDIUM
**Motivated by:** 2102.11107 §IV (SMS) + F-3 mechanism-independence reading
**Test:** Perturb 50 MATH-500 problems with 4 controlled prompt edits each
(paraphrase, numeric swap, distractor injection, problem restatement).
Re-extract L19 prefill + final-token activations. Compute cross-sensitivity
matrix S_ij = corr(Δ_prefill, Δ_final) across perturbation types and problems.
SMS predicts S_ij ≈ 0 if prefill_DoM and final_DoM are independent ICM
modules. Compute ℓ₀ / ℓ₁ ratio of the change vector as a sparseness
quantification. (~4h H100 for re-extraction + 1h CPU for analysis.)
**Requires:** H100 pod; GPT-4 paraphrasing pipeline; existing extraction
script generalized to perturbed-prompt batches.
**Would change:** Low S_ij ⇒ F-3's "independent mechanisms" reading
strengthens. High S_ij ⇒ orthogonality is geometric coincidence; F-3
recategorized as "two correlated readouts of a shared upstream cause"
rather than "two pathways."
**Blocks:** nothing.

---


### H-47: Zigzag persistent homology on residual-stream trajectories beats static VR-PH and rivals L19 DoM
**Priority:** HIGH
**Motivated by:** 2103.07353 + F-10 + F-2
**Test:** Build sliding-window k-NN graph filtration over L19 hidden states across decoding steps for cached MATH-500 1.5B trajectories. Run fzz (github.com/taohou01/fzz) to get 0-dim and 1-dim zigzag barcodes per problem. Train a logistic probe on barcode summary features (longest interval, count, total persistence) predicting K=1 correctness with 5-fold OOF. Compare AUROC against F-2 (0.7731) and F-9/CoE-60 (0.7185). Estimated time: 1-2 days CPU.
**Requires:** CPU only, fzz binary built locally (Boost + PHAT deps), cached `pathway11_h100/` L19 NPZs (already on disk per DATA_MANIFEST.md).
**Would change:** If zigzag AUROC > 0.78, F-10's "PH = null" generalizes only to *static* PH, and we have a new probe family worth pursuing for selective-prediction (F-8). If zigzag AUROC ≈ Gaussian null, F-10 generalizes from static to dynamic — strong evidence that residual-stream topology is genuinely uninformative for correctness, regardless of filtration design.
**Blocks:** Any future zigzag-PH-based experiment (P8-FE5 attention-graph variant; H-22 abstract-CoT zigzag comparison).

---


### H-48: F-10's PH-at-null verdict is a subsampling artifact of random/maxmin landmarks
**Priority:** MEDIUM
**Motivated by:** 2103.14743 + F-10
**Test:** Re-run Pathway 8 layer-wise PH on Pathway 11 H100 1024-tok residuals with Stolz Algorithm 1 (PH landmarks I and II, `out_PH^1`-driven) replacing the random/maxmin landmark step. Compare persistence diagrams to matched-Gaussian baseline per layer. ~2h CPU. (P8-FE6.)
**Requires:** Cached Stage 2 prefill+per-layer residuals (already on disk per DATA_MANIFEST), Python Ripser, sanity-check from P8-FE7.
**Would change:** If null-rejection survives PH-landmark subsampling, F-10 hardens to a strong negative claim. If null-rejection collapses, F-10 is downgraded to "PH-at-null under standard subsampling only" and the v1 topological-homology framing becomes a live possibility again — invalidating part of PROJECT_RECORD §1d's graveyard entry.
**Blocks:** any clean writeup of F-10; H-22 (Abstract-CoT compressed breathing) interpretation.

### H-49: Per-point PH-outlierness separates D-bucket from A/B/C buckets
**Priority:** MEDIUM
**Motivated by:** 2103.14743 + F-7 + H-7
**Test:** On Pathway 11 H100 prefill L19 residuals (1.5B MATH-500 K=1, N=500), compute `out_PH^1(y)` per problem (Ripser on each problem's δ-ball at δ = median nearest-neighbour distance). Mann-Whitney U test for D-bucket vs other-bucket distributions. AUROC for D-bucket membership prediction. Head-to-head against KNN density (H-7's mechanism). ~1h CPU. (P11-FE50.)
**Requires:** Cached prefill L19 NPZs, Python Ripser, bucket labels from existing pipeline.
**Would change:** Confirm: refutes F-7's "collective only" framing and provides a per-point alternative to the H-7 density mechanism. Reject: F-7 strengthens, H-7 stays as the D-bucket detector candidate.
**Blocks:** F-7 negative-half writeup; clean retirement of H-7 if both per-point candidates fail.

---


### H-50: Heat-distance vectorization closes the F-10 PH-vs-null gap
**Priority:** HIGH
**Motivated by:** 2106.00012 + EXP-026
**Test:** Re-extract Pathway 9 PH features from cached L19 activations using giotto-tda's `Heat` and `Silhouette` PD-vectorizations (η=0.01 lifespan filter, Z₂ field, H0–H2). Compute AUROC and matched-covariance Gaussian null. ~2h CPU.
**Requires:** Cached `pathway9/cache/L19_qwen15b_*.npz`; giotto-tda already in env.
**Would change:** If Heat-vectorized AUROC > null + 0.05, F-10's "PH at Gaussian null" framing is vectorization-specific — null applies to raw H0/H1 summary statistics but not to discretized PD distance. F-10 narrows to "raw-summary-PH at null"; vectorized form gets re-tested. If null gap stays ≤ 0.005, F-10 is robust to giotto-tda's preferred discretizations.
**Blocks:** H-51 (between-state framing rests on having a sane within-state baseline first).

### H-51: Between-token PH-Heat distance adds AUROC over single-layer DoM
**Priority:** HIGH
**Motivated by:** 2106.00012 + F-9
**Test:** On the 500 MATH-500 prefill→answer L19 token sequences (cached in `pathway11_h100/extracted_states/`), build a Vietoris-Rips PD per 16-token sliding window, compute Heat-distance between consecutive-window PDs, take per-problem mean as scalar feature. Logistic-regression OOF 5-fold AUROC vs correctness. ~4h CPU.
**Requires:** Cached 1024-tok L19 sequences; giotto-tda.
**Would change:** If AUROC > 0.79 (DoM=0.7731), F-9's claim that "trajectory features are redundant with single-layer DoM" was depth-trajectory-specific — time-trajectory PH carries independent signal. If AUROC ≤ 0.78, F-9 generalizes properly to both trajectory axes.
**Blocks:** Promote-or-reject of paper's whole between-state framing.

### H-52: PH-on-weights of Qwen LoRA fine-tune checkpoints correlates with MATH-500 val acc
**Priority:** LOW (heaviest experiment in this brief; depends on availability of training checkpoints we don't currently have)
**Motivated by:** 2106.00012
**Test:** Save 10 LoRA-fine-tuning checkpoints during a Qwen-2.5-1.5B SFT run on MATH-500. Build directed flag complex on each checkpoint's LoRA adapter weight-graph (manageable size, unlike full model). Compute Heat-distance between consecutive checkpoint PDs. Pearson-correlate with held-out K=1 MATH-500 accuracy at each checkpoint. Compare to MNIST r=0.89.
**Requires:** Qwen-2.5-1.5B SFT pipeline (not currently part of the project), 1× H100 day, ~$60.
**Would change:** If r > 0.7, paper's between-state PH framing transfers from MLP-on-MNIST to LoRA-on-Qwen-MATH — adds a holdout-free training-monitor for our setting. If r < 0.4, paper's CNN-failure-mode generalizes (PH-on-weights is MLP-specific and doesn't capture transformer/LoRA).
**Blocks:** nothing — long-tail.

---


### H-53: Birdal-style PHD on residuals detects structure F-10's PH features missed
**Priority:** HIGH
**Motivated by:** 2111.13171 + F-10
**Test:** Run `calculate_ph_dim` (github.com/tolgabirdal/PHDimGeneralization) on
cached pathway 8 L19 residual NPZs (per-problem 1024×1536 clouds). Compare
PHD distribution against PHD computed on a matched-shape Gaussian point cloud.
~30 min CPU.
**Requires:** CPU only; cached pathway 8 NPZs; `ripser` Python package.
**Would change:** If PHD significantly differs from Gaussian PHD (>2σ), F-10's
"PH = Gaussian null" headline is downgraded to "PH = Gaussian null *under the
fingerprint statistics we tried*" and reopens P7. If PHD ≈ Gaussian PHD, F-10
hardens substantially.
**Blocks:** Nothing (P7-FE3 is self-contained).

### H-54: Per-problem PHD on prefill L19 is a label-free competitor to F-2's supervised DoM
**Priority:** HIGH
**Motivated by:** 2111.13171 + F-2 + F-8
**Test:** Compute per-problem PHD on cached prefill L19 (pathway11_h100 Stage 2
NPZs). AUROC against K=1 correctness. If AUROC ≥ 0.70, ship as label-free
gating signal alongside F-8's selective-prediction pipeline.
**Requires:** CPU only; cached Stage 2 NPZs; `ripser`.
**Would change:** If PHD-AUROC ≥ 0.70, the supervised-probe framing in F-2/F-8
is significantly weakened — there exists a probe-free, training-free
correctness predictor of comparable strength. If PHD-AUROC < 0.55, the
supervised framing stands; PHD is captured by but not replaceable by DoM.
**Blocks:** P11-FE55 confirmation needed before claiming label-free selective
prediction.

### H-55: Per-problem PHD distinguishes D-bucket from A-bucket on prefill L19
**Priority:** MEDIUM
**Motivated by:** 2111.13171 + F-7
**Test:** KS test between A-bucket (K=1 right, K=8 right) and D-bucket (K=1 right,
K=8 wrong) per-problem PHD distributions on prefill L19. From P11-FE55 output.
**Requires:** Same as H-54; reuses output of P11-FE55.
**Would change:** If KS rejects (p<0.01), F-7's "collective not per-problem"
hedge collapses — D-bucket has a per-problem PHD signature. If KS doesn't
reject, F-7 stands and PHD-as-D-detector is ruled out.
**Blocks:** Nothing.

---


### H-56: K-sample consistency-rate matches or exceeds prefill-DoM selective prediction at K=8 compute
**Priority:** HIGH
**Motivated by:** 2203.11171 + F-8 + F-2
**Test:** On cached MATH-500 × Qwen-2.5-1.5B K=8 outputs, compute per-problem consistency rate (% of K paths agreeing with majority), sort, take top-50% answered set, report accuracy. Compare to F-8's 71.6% at coverage 0.5, K=2.5 avg. (Note: K=8 vs K=2.5 is *not* matched compute — also report consistency-rate at K=2 and K=4 for the apples-to-apples version.) Est: 30min CPU.
**Requires:** Cached K=8 path-by-path answers from `pathway11_h100/prefill_gated_compute/`. No new GPU.
**Would change:** Confirm → F-8 reframes as "matches output-space consistency at K=1 inference, loses at K≥4" and is downgraded from a primary finding to a low-compute alternative; should also re-prioritize H-12 (SE-probes). Reject → F-8 is *strengthened* as the only label-free / one-shot selector that beats Wang's consistency-rate, a much stronger claim than currently made.
**Blocks:** any further claim that prefill DoM is "the" correctness signal; prerequisite for promoting F-8 to PUBLICATION_READY.

### H-57: D-bucket vanishes (size < 1% of MATH-500) at K=40 majority vote
**Priority:** MEDIUM
**Motivated by:** 2203.11171 + F-7
**Test:** Extend K=8 MATH-500 1.5B run to K=40 (T=0.7, top-k=40); recompute D-bucket (problems hurt by majority vote) at K=8, K=20, K=40. Plot D-bucket(K). Est: 8h H100.
**Requires:** H100 pod, fresh generation pass on same MATH-500 prompts.
**Would change:** Confirm (D-bucket ~ 0 at K=40) → F-7's "collective geometric signature" claim is reframed as a K=8-specific noise pocket and downgraded; the geometric story for those problems may still be valid descriptively but is no longer load-bearing for selective prediction. Reject (D-bucket persists at K=40) → F-7 is sharpened: there are problems where *no amount* of self-consistency rescues, motivating the next round of mechanistic probes (H-21, H-19).
**Blocks:** H-21, H-19 — if D-bucket vanishes, the per-problem mechanistic stories these hypotheses test become much smaller-N experiments.

### H-58: Self-consistency at K=8 closes most of the H-19 verification-vs-scaling gap on Qwen-2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2203.11171 + H-19
**Test:** At matched FLOPs ~ 8× single-pass inference, run (a) plain K=8 SC, (b) K=4 SC + small-model self-verifier rerank, (c) K=1 + prefill-DoM-gated K=4 fallback (the H-19 design). Report MATH-500 accuracy and answered-set accuracy at coverage 0.5. Est: 6h H100 + 1h CPU analysis.
**Requires:** H100 pod, verifier prompt template (can re-use Cobbe-style verifier prompts).
**Would change:** Confirm (a within 1pp of b) → H-19 is *provisionally refuted* at this scale: smart routing doesn't beat brute-force sampling. The interesting question becomes "does H-19 win at K≥40 ceiling, where SC plateaus?" and we should run that. Reject → H-19 strengthened: smart-budget routing dominates SC even at modest scale, which is the scaling-laws story we want.
**Blocks:** any "verification routing" framing of compute-allocation experiments.

---


### H-59: K-sample consistency-rate matches or exceeds prefill-DoM selective prediction at K=8 compute
**Priority:** HIGH
**Motivated by:** 2203.11171 + F-8 + F-2
**Test:** On cached MATH-500 × Qwen-2.5-1.5B K=8 outputs, compute per-problem consistency rate (% of K paths agreeing with majority), sort, take top-50% answered set, report accuracy. Compare to F-8's 71.6% at coverage 0.5, K=2.5 avg. (Note: K=8 vs K=2.5 is *not* matched compute — also report consistency-rate at K=2 and K=4 for the apples-to-apples version.) Est: 30min CPU.
**Requires:** Cached K=8 path-by-path answers from `pathway11_h100/prefill_gated_compute/`. No new GPU.
**Would change:** Confirm → F-8 reframes as "matches output-space consistency at K=1 inference, loses at K≥4" and is downgraded from a primary finding to a low-compute alternative; should also re-prioritize H-12 (SE-probes). Reject → F-8 is *strengthened* as the only label-free / one-shot selector that beats Wang's consistency-rate, a much stronger claim than currently made.
**Blocks:** any further claim that prefill DoM is "the" correctness signal; prerequisite for promoting F-8 to PUBLICATION_READY.

### H-60: D-bucket vanishes (size < 1% of MATH-500) at K=40 majority vote
**Priority:** MEDIUM
**Motivated by:** 2203.11171 + F-7
**Test:** Extend K=8 MATH-500 1.5B run to K=40 (T=0.7, top-k=40); recompute D-bucket (problems hurt by majority vote) at K=8, K=20, K=40. Plot D-bucket(K). Est: 8h H100.
**Requires:** H100 pod, fresh generation pass on same MATH-500 prompts.
**Would change:** Confirm (D-bucket ~ 0 at K=40) → F-7's "collective geometric signature" claim is reframed as a K=8-specific noise pocket and downgraded; the geometric story for those problems may still be valid descriptively but is no longer load-bearing for selective prediction. Reject (D-bucket persists at K=40) → F-7 is sharpened: there are problems where *no amount* of self-consistency rescues, motivating the next round of mechanistic probes (H-21, H-19).
**Blocks:** H-21, H-19 — if D-bucket vanishes, the per-problem mechanistic stories these hypotheses test become much smaller-N experiments.

### H-61: Self-consistency at K=8 closes most of the H-19 verification-vs-scaling gap on Qwen-2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2203.11171 + H-19
**Test:** At matched FLOPs ~ 8× single-pass inference, run (a) plain K=8 SC, (b) K=4 SC + small-model self-verifier rerank, (c) K=1 + prefill-DoM-gated K=4 fallback (the H-19 design). Report MATH-500 accuracy and answered-set accuracy at coverage 0.5. Est: 6h H100 + 1h CPU analysis.
**Requires:** H100 pod, verifier prompt template (can re-use Cobbe-style verifier prompts).
**Would change:** Confirm (a within 1pp of b) → H-19 is *provisionally refuted* at this scale: smart routing doesn't beat brute-force sampling. The interesting question becomes "does H-19 win at K≥40 ceiling, where SC plateaus?" and we should run that. Reject → H-19 strengthened: smart-budget routing dominates SC even at modest scale, which is the scaling-laws story we want.
**Blocks:** any "verification routing" framing of compute-allocation experiments.

---


### H-62: Attention-graph TDA carries correctness signal where residual-stream PH does not
**Priority:** HIGH
**Motivated by:** 2205.09630 + F-10 (EXP-026)
**Test:** Run H0M / β0 / β1 / #cycles on per-head Qwen 1.5B attention maps for
MATH-500 prefill position, fit LR vs K=1 correctness, report AUROC and compare
to (a) random-graph null with matched edge density, (b) F-2's prefill DoM
0.7731. ~2h CPU if attention cached.
**Requires:** CPU only; cached Qwen 1.5B attention (verify), MATH-500
correctness labels (cached); Ripser/Ripser++ install.
**Would change:** Confirm → F-10's "PH at Gaussian null" must be re-scoped to
residual-stream Rips PH only; opens an entire label-free attention-graph probe
family. Reject → F-10 generalises to attention graphs and the negative result
is stronger.
**Blocks:** P11-FE64 (RTD selective prediction needs the basic attention-TDA
pipeline working first).

### H-63: A single Qwen attention head's H0M AUROC rivals or beats prefill L19 DoM (0.7731) for MATH-500 correctness
**Priority:** MEDIUM
**Motivated by:** 2205.09630 + F-2
**Test:** From the H-62 per-head AUROC table, identify max-AUROC head; report
head identity, AUROC, and bootstrap CI. Compare head-by-head against F-2's
prefill DoM at L19.
**Requires:** Same data as H-62.
**Would change:** Confirm → F-2's "L19 DoM is the best correctness predictor"
framing is incomplete; per-head topology may be a stronger probe class.
Restate F-2 as "best-residual-stream-direction predictor" rather than "best
predictor". Reject → DoM remains the cheapest strong probe.
**Blocks:** nothing.

---


### H-64: Prefill L19 DoM signal is concentrated in a sparse attention-head circuit (≤10 heads carry ≥80% of AUROC)
**Priority:** HIGH
**Motivated by:** 2211.00593 + F-2 + F-3
**Test:** Path patching (or AtP* approximation) on Qwen-2.5-1.5B, MATH-500 prefill activations, target = L19 DoM probe logit. Rank (layer, head) contributions; find smallest set whose ablation drops AUROC from 0.7731 to ≤0.65 (F-2 floor for "no signal"). 1–2 days H100, or 4h CPU + 1h H100 with AtP*.
**Requires:** H100 pod (or CPU + cached per-head activations); Qwen-2.5-1.5B HookedTransformer config; corrupted MATH-500 prompt set (perturb one number per problem).
**Would change:** On confirm, F-2 reframed as circuit-mediated rather than single-direction; H-13 partially answered. On reject (signal genuinely distributed across many heads), F-2's single-direction framing is mechanistically validated and the geometric reading wins over the circuit reading.
**Blocks:** H-65 (depends on identified head set)

### H-65: Prefill and final-token DoMs share head support despite orthogonal residual-stream projection
**Priority:** HIGH
**Motivated by:** 2211.00593 + F-3
**Test:** Take top-3 heads from H-64. Mean-ablate them on MATH-500 forward pass. Re-extract L19 final-token activations. Refit final-token DoM probe. Compare AUROC to unablated baseline (0.7186). If within 0.02, F-3's "two independent circuits" reading is wrong. ~2h H100.
**Requires:** H100, MATH-500 mean activations for ablation reference (length-matched to 1024-tok), output of H-64.
**Would change:** Reframes F-3 from "two circuits" to "one circuit with two positional readouts" — this changes which interventions could plausibly affect both signals simultaneously.
**Blocks:** nothing

### H-66: Path-patching faithfulness criterion can be applied to F-2/F-3 to establish a mechanistic explanation
**Priority:** MEDIUM
**Motivated by:** 2211.00593 (faithfulness/completeness/minimality framework)
**Test:** Apply Wang et al.'s three quantitative criteria — faithfulness (circuit reproduces ≥95% of full-model logit difference), completeness (ablating circuit complement leaves task signal intact), minimality (no head can be removed without faithfulness drop) — to whatever circuit H-64 identifies. Codify pass/fail thresholds. ~4h CPU on top of H-64 outputs.
**Requires:** H-64 outputs.
**Would change:** Provides go/no-go criteria for declaring a "MATH-500 prefill DoM circuit" found, so H-13 has a concrete success metric.
**Blocks:** nothing

---


### H-67: Prefill L19 DoM is a task-recognition probe, not a per-problem correctness probe
**Priority:** HIGH
**Motivated by:** 2301.00234 + Pan et al. 2023b (cited in survey §5.2.2) + F-2
**Test:** Construct two MATH-500 subsets matched on gold K=1 accuracy: (a) canonical format, (b) rephrased (e.g., equation-only or unusual word-problem framings). Extract prefill L19 activations on both. Compare DoM AUROC. ~1 day H100 to generate variants and re-extract.
**Requires:** H100 (1 day for re-extraction), Qwen-2.5-1.5B, MATH-500 with rephrased variants (new data)
**Would change:** On confirm (AUROC drops sharply on rephrased set despite matched accuracy), F-2's narrative shifts from "prefill knows correctness" to "prefill recognizes task family." F-8 (selective prediction at 50% coverage) keeps its number but the *mechanism* becomes "abstain on novel-format problems," which has a sharper actionable handle. On reject (AUROC holds), prefill carries genuine per-problem information and TR-only reading is wrong.
**Blocks:** Sharper version of H-5; prerequisite for any cross-task transfer claim about prefill DoM.

### H-68: F-3 prefill/final-token DoM orthogonality is an anchor-aggregation artifact, not two independent circuits
**Priority:** HIGH
**Motivated by:** 2301.00234 + Wang et al. 2023b (cited in survey §5.2.1) + F-3
**Test:** On a re-extracted 4-shot MATH-500 prefill set, identify label-anchor token positions; zero-ablate their residual contribution at L19 and recompute prefill DoM, final-token DoM, and cos(prefill, final). ~1 day H100 for the 4-shot extraction, then 2h CPU for ablation + probe.
**Requires:** H100 day (re-extraction), Qwen-2.5-1.5B, 4-shot MATH-500 prompts (new data)
**Would change:** On confirm (cos jumps from 0.046 to >0.5 under anchor ablation), F-3's "two orthogonal circuits" framing collapses to "one induction-head circuit at two timesteps gated by anchor aggregation"; H-13 (head-level attribution) becomes the natural follow-up. On reject (cos stays low under ablation), prefill and final-token DoM survive as genuinely independent computations and F-3 is confirmed.
**Blocks:** H-13 head-attribution interpretation depends on which way this resolves.

### H-69: Residual-stream class-conditional difference at L19 has a clean 1-2 PC concept axis hidden under the F-10 PH null
**Priority:** MEDIUM
**Motivated by:** 2301.00234 + Han et al. 2023a kernel-regression limit (cited survey §5.2.2) + F-10
**Test:** Project cached 1024-tok L19 activations onto top-2 PCs of the correct-vs-incorrect class-conditional mean difference; measure linear / planar separability; compare against prefill DoM AUROC. ~20 min CPU on existing NPZs.
**Requires:** CPU only, cached L19 1024-tok NPZs (already on disk)
**Would change:** On confirm (a clean 1-D / 2-D concept axis with separation comparable to DoM AUROC), F-10's "PH at Gaussian null = no structure" framing is misleading — the right framing is "no higher-order topological structure but a planar concept axis." On reject (PCA also at-null-level), F-10 is reinforced as a stronger negative.
**Blocks:** Affects how F-10 is described in any writeup.

---


### H-70: L19 is not at the Qwen-2.5-1.5B prefill ID minimum
**Priority:** HIGH
**Motivated by:** 2302.00294 + F-2 + F-9
**Test:** Run TwoNN (DADApy) on cached per-layer prefill activations for Qwen-2.5-1.5B on MATH-500; locate argmin ID(L). If argmin ≠ L19, fit a logistic probe at argmin and compare AUROC vs F-2's 0.7731. ~1h CPU on existing cache (if all-layer prefill cache exists; otherwise pre-extract step is ~6h H100). Owner: P11.
**Requires:** pathway11_h100 prefill cache extended to all 28 layers, DADApy >= 0.3, sklearn.
**Would change:** *Confirm* (argmin ≠ L19 with higher AUROC) → F-2's "L19 is privileged" recast as "the supervised search lands near the ID minimum, but the geometric rule finds a better layer," with downstream implications for FE-driven layer selection in 7B and other families. *Reject* (argmin = L19) → F-2's L19 choice is *also* the unsupervised geometric optimum, strengthening F-2 considerably.
**Blocks:** H-71, H-72.

### H-71: Qwen-family ID profiles diverge from Pythia/Llama on MATH-500 prefill
**Priority:** MEDIUM
**Motivated by:** 2302.00294 + F-1
**Test:** Compute TwoNN ID(L) per layer on MATH-500 prefill for Qwen-2.5-1.5B, Pythia-1.4B, Llama-3-8B. Count number of local minima and compare qualitative shape (single-peak vs double-peak vs triple-peak). ~10h H100 for cross-family prefill extraction + ~2h CPU for analysis.
**Requires:** Pythia-1.4B and Llama-3-8B weights, MATH-500 with matched chat templates, H100 for extraction.
**Would change:** *Confirm* (different families show different peak counts) → F-1's "universal" weakened to "Qwen-family-internal"; opens a new sub-direction on what determines ID-profile shape across families. *Reject* (all three show same shape) → strengthens F-1 across the AR-LM family, makes "breathing is universal" a much harder claim to dismiss.
**Blocks:** nothing (independent test).

### H-72: D-bucket emerges at the ID minimum, not at L19
**Priority:** MEDIUM
**Motivated by:** 2302.00294 + F-7
**Test:** Compute χ^{l,gt}_k=10 with y = {A-bucket, D-bucket} label at every layer of Qwen-2.5-1.5B prefill (using cached activations from H-70). Identify argmax_L χ^{l,gt} and compare against L19 and against argmin_L ID(L).
**Requires:** All-layer prefill cache (from H-70), bucket labels (already in `pathway11_h100/prefill_gated_compute/results.json`).
**Would change:** *Confirm* (D-bucket separability peaks at ID-min layer, not L19) → F-7 reframed: D-bucket is a *layer-geometry* phenomenon visible at the encoder-output layer, not an L19-specific signature. *Reject* (D-bucket peaks at L19, not the ID min) → F-7 strengthened as a non-trivially layer-specific phenomenon, possibly post-encoder.
**Blocks:** nothing.

---


### H-73: Tuned-lens trajectory beats single-layer L19 DoM for MATH-500 correctness prediction
**Priority:** HIGH
**Motivated by:** 2303.08112 + F-2
**Test:** Train per-layer tuned-lens translators on Qwen-2.5-1.5B with KL distillation loss; compute per-problem prediction trajectory on cached MATH-500 1024-tok residuals; fit iForest/LOF on correct-answer trajectories only; evaluate AUROC for predicting K=1 incorrectness. Compare against F-2's 0.7731. ~4h H100 train + 30min CPU eval.
**Requires:** H100 for translator training (~50K-token Pile/MATH slice), cached MATH-500 residuals (already exist), `tuned-lens` library, no labels.
**Would change:** Confirm → F-2's "prefill L19 DoM is the strong predictor" reframes as "L19 DoM is a useful low-dim projection of a richer label-free trajectory signal"; H-12 (semantic entropy probes) is upstaged; selective prediction (F-8) gets a label-free upgrade. Reject → F-2's localization is robust against the strongest known label-free baseline; the supervised DoM is not redundant with iterative-inference draft trajectories.
**Blocks:** H-74, H-75 (depend on trained translators).

### H-74: Prediction-depth distinguishes D-bucket from A/B/C-bucket on Qwen-2.5-1.5B MATH-500
**Priority:** MEDIUM
**Motivated by:** 2303.08112 §5.4 + F-7
**Test:** After H-73's tuned lens is trained, compute prediction-depth (smallest layer L* such that argmax(TunedLens_l) is constant for all l >= L*) for every MATH-500 problem. Compare prediction-depth distributions across A-bucket (K=1 right, K=8 right), B-bucket (K=1 wrong, K=8 right — recoverable), C-bucket (K=1 wrong, K=8 wrong), D-bucket (K=1 right, K=8 wrong — pathological). KS-test for distinct distributions. ~30min CPU.
**Requires:** trained Qwen-2.5-1.5B tuned lens (from H-73); cached MATH-500 K=1 + K=8 results; B/D-bucket assignments.
**Would change:** Confirm → F-7's "D-bucket has distinctive collective signature" gets a per-problem mechanistic interpretation (D-bucket has distinctively early/late prediction depth); selects D-bucket detection as a candidate for label-free routing. Reject → F-7 stays a collective-only finding; D-bucket cannot be detected without K=8 sampling.
**Blocks:** nothing.

### H-75: Causal Basis Extraction reveals L19 DoM is rank-1 of a multi-direction causal subspace
**Priority:** MEDIUM
**Motivated by:** 2303.08112 §4.1 + F-2
**Test:** Run CBE on cached Qwen-2.5-1.5B L19 prefill residuals (correct-supervised correctness logit as functional, mean-ablation erasure, k=5 directions). Compute (a) cumulative AUROC of top-k directions, (b) cosine of each CBE direction with the supervised L19 DoM. Stimulus-response alignment: do CBE directions also degrade the model's correctness, or only the probe's? ~2h CPU.
**Requires:** cached L19 residuals (exist); supervised correctness probe (exists); H100 *not* needed for CBE itself but needed for stimulus-response forward-passes (~30min H100 for k=5 directions).
**Would change:** Confirm DoM is rank-1 with cumulative AUROC saturating quickly → F-2's localization is robust; H-1's per-position DoM steering plan can use a single direction. Reject (multiple equally-effective orthogonal directions) → F-2 needs to be reframed as a *subspace* finding, and H-1 should steer over a basis rather than a single vector.
**Blocks:** H-1 (per-position DoM steering scope depends on whether DoM is a direction or subspace).

---


### H-76: ACDC produces a sparse circuit (<30% of edges) for the prefill-DoM signal at L19 on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2304.14997 + F-2 + H-13
**Test:** Run ACDC with prefill-DoM as the metric on Qwen-2.5-1.5B, MATH-500 problems with random-other-MATH-500 as corrupted baseline, KL ≤ 0.05 threshold. Report edge retention at the largest τ that preserves KL. ~1 day H100 + 1-2 days port.
**Requires:** H100 pod, `transformer_lens` port for Qwen-2.5, per-head L19 activations (likely re-extraction), pre-registered τ and edge-fraction success criterion.
**Would change:** On confirm — H-13 is provable, F-2 has a localized causal handle, downstream interventions become possible. On reject (edge retention ≥ 50% at any KL-preserving τ) — F-2 is a population-code signal; the AUROC 0.7731 number stands but the interpretive narrative shifts to "distributed correctness representation," and H-1 (per-position DoM steering) becomes correspondingly less likely to work.
**Blocks:** H-77 (overlap test), any future causal intervention experiment that targets specific heads.

### H-77: prefill-DoM and final-token-DoM circuits share ≥60% of heads despite cos = 0.046 orthogonality
**Priority:** MEDIUM
**Motivated by:** 2304.14997 + F-3
**Test:** Two ACDC runs (prefill-DoM, final-token-DoM as metrics), compute Jaccard of recovered head sets. ~2 days H100 (after H-76 establishes ACDC works on our task).
**Requires:** H-76 must have run successfully; same H100 + port prerequisites.
**Would change:** On confirm (Jaccard ≥ 0.6) — F-3's "two distinct mechanisms" framing must be revised; the cos = 0.046 number stands but its narrative interpretation shifts to "shared circuit writing into orthogonal subspaces via output-projection geometry." On reject (Jaccard ≤ 0.2) — F-3 strengthens; truly disjoint circuits.
**Blocks:** any narrative claim built on F-3 (currently F-3 backstops the prefill/final-token separation in the headline narrative).

### H-78: HISP-style head importance scoring on cached prefill activations recovers a 5-10 head subset preserving AUROC ≥ 0.74
**Priority:** HIGH
**Motivated by:** 2304.14997 + F-2
**Test:** Compute per-head gradient × activation importance at L19, rank, ablate top-k zero-shot, measure AUROC. 2-4h CPU if per-head data is cached, ~6h H100 if re-extraction needed.
**Requires:** Cached per-head L19 activations (need to verify in DATA_MANIFEST) OR a re-extraction run.
**Would change:** On confirm — F-2 has a localized causal handle even before full ACDC; informs experimental design (intervene on those heads). On reject (no sparse subset preserves AUROC) — F-2 is distributed; H-76 will likely fail; reframes the project's narrative around F-2 toward "population-coded correctness signal" rather than "single-direction handle."
**Blocks:** H-76 (HISP is a cheaper screen that should run first).

---


### H-79: Prefill L19 DoM is a non-causal linear correlate, removable by LEACE without harming MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2305.00586 + F-2 + F-8
**Test:** LEACE-ablate the prefill DoM direction on cached L19 residuals; (a) refit a probe on residualized features and measure AUROC; (b) project residualized residuals through `lm_head` and compute approximate K=1 MATH-500 accuracy. Compare against baselines (probe AUROC 0.7731, accuracy 48.6%). 1h CPU for projection-only; 4h H100 for true regen-test.
**Requires:** CPU + cached `pathway11_h100/prefill_gated_compute/` activations + EleutherAI/concept-erasure library. Optional: H100 pod for true regeneration.
**Would change:** If AUROC drops to ≤ 0.55 while accuracy holds ≥ 45%, F-2 must be reframed as "linearly-extractable correlate, not causal correctness signal" — and F-8's selective-prediction headline becomes a probing-artifact result. If both AUROC and accuracy collapse together, F-2 is causally vindicated.
**Blocks:** nothing; informs P11-FE85, P11-FE86, P11-FE88.

### H-80: A multi-layer late-MLP circuit (L17–L21) implements MATH-500 correctness in Qwen-1.5B, mirroring Hanna et al.'s GPT-2 MLP 8–11 greater-than circuit
**Priority:** MEDIUM
**Motivated by:** 2305.00586 + F-2 + F-9
**Test:** Iterative path patching on Qwen-1.5B with counterfactual MATH-500 dataset (correct → wrong-final-answer minimal edits); build IPP heatmap of (layer × head) and (layer × MLP) contributions to a prefill-DoM-projected scalar across L15–L23; report the minimal sub-circuit recovering ≥ 80% of the DoM scalar.
**Requires:** H100 pod (1d full sweep, 4h restricted) + counterfactual MATH-500 generation.
**Would change:** If the discovered circuit fits inside L19 (≤ 2 MLPs, ≤ 6 heads), F-2's single-layer framing is mechanistically vindicated. If it spans L17–L21 (4+ MLPs), F-9's "CoE redundant with single-layer DoM" needs revision in the multi-layer composition direction.
**Blocks:** any rigorous H-13 circuit claim.

### H-81: Prefill DoM over-activates on adversarial MATH-500 with minimal operator swaps (largest↔smallest), producing high-confidence wrong-direction predictions
**Priority:** MEDIUM
**Motivated by:** 2305.00586 + F-8
**Test:** Construct ~50 MATH-500 variants with minimal operator swaps preserving surface form; measure prefill-DoM AUROC and selective-prediction accuracy at coverage 0.5. Compare against unmodified MATH-500 baseline.
**Requires:** 2h CPU for variant construction + 4h H100 for K=2.5 regeneration.
**Would change:** If AUROC drops > 10 points while DoM scalar magnitudes stay high, F-8 has a critical robustness gap and selective-prediction guarantees are surface-pattern-bounded. If AUROC holds, F-8 is robust to operator-swap adversaries (a strong claim).
**Blocks:** any deployment-style claim about selective prediction.

---


### H-82: Prefill L19 DoM signal is concentrated on a small fraction of input tokens, not a "global" direction
**Priority:** HIGH
**Motivated by:** 2306.02873 (DecompX per-token decomposition) + F-2 + F-7
**Test:** Run P11-FE90: compute gradient×input of (L19 prefill activation · DoM) wrt input embeddings on cached activations, sum per input token, report attribution entropy and top-5% mass fraction across all 500 MATH-500 problems. Compare D-bucket vs A-bucket entropy distributions. ~30 min CPU on cached NPZs.
**Requires:** CPU, cached prefill activations from `pathway11_h100/prefill_gated_compute/`, fitted DoM probe.
**Would change:** If top-5% of tokens carry ≥70% of |DoM-projection| mass on a majority of problems, F-2 should be reframed: the "single direction" finding is partly a per-prompt sparse-token-attribution phenomenon — converges with H-21 (D-bucket attention concentration) and Akli 2604.24712 (under-specification fragility). If attribution is uniform (entropy near log N), F-2's geometric framing survives and DecompX-style decomposition is unnecessary.
**Blocks:** H-83 (only worth doing if H-82 shows non-trivial concentration).

### H-83: FFN-block dominates the prefill DoM signal more than any single attention head
**Priority:** MEDIUM
**Motivated by:** 2306.02873 (DecompX leave-one-out shows FFN ablation > bias ablation in AOPC drop) + H-13
**Test:** Run P11-FE92: zero out FFN_19 contribution and re-extract prefill activations vs zero out attention_19 head-by-head. Measure AUROC drop of L19 DoM probe under each ablation. ~1h H100 on top of the H-13 setup.
**Requires:** H100 for activation re-extraction with selective component zeroing; cached DoM probe.
**Would change:** If FFN ablation drops AUROC by more than the worst single-head ablation (say > 0.05 difference), the F-2 signal is FFN-mediated and H-13's head-only attribution is the wrong granularity — should be reframed to "sub-block-level circuit attribution" with FFN as a first-class entity. If individual attention heads dominate, H-13 stands.
**Blocks:** nothing.

---


### H-84: Per-head, post-MHA Qwen probing exceeds residual-stream L19 DoM for prefill correctness
**Priority:** HIGH
**Motivated by:** 2306.03341 + F-2
**Test:** Extract per-head pre-output-projection activations across all (l, h) pairs on Qwen-2.5-1.5B for the cached MATH-500 prefill set, fit per-head linear probes for K=1 correctness, rank by OOF AUROC. Compute (a) max single-head AUROC, (b) top-K head DoM bank AUROC for K ∈ {16, 32, 48}. Compare to F-2's 0.7731.
**Requires:** 30min H100 (one fresh forward pass to capture per-head activations) + 1h CPU; cached MATH-500 prefill set + correctness labels.
**Would change:** Confirm → F-2 should be re-anchored on the canonical (l, h) head, not residual stream L19; F-9 (CoE redundancy) needs revisiting against multi-head bank. Reject → residual-stream L19 is genuinely the best target and ITI's head-wise advantage is LLaMA-architecture-specific.
**Blocks:** H-13 (head-level attribution).

### H-85: A single TruthfulQA-style fixed correctness direction transfers across reasoning tasks on Qwen
**Priority:** HIGH
**Motivated by:** 2306.03341 + F-2 + H-5
**Test:** Fit Qwen-2.5-1.5B L19 prefill DoM on a TriviaQA closed-book correctness set (~1k items), apply zero-shot to cached MATH-500 prefill activations, measure AUROC on MATH-500 K=1 correctness; also measure cos(MATH_DoM, TriviaQA_DoM).
**Requires:** 30min H100 (TriviaQA closed-book generation) + 30min CPU (probe fitting + transfer).
**Would change:** Confirm (AUROC > 0.65, cos > 0.5) → correctness axis is partly task-general; H-5's "domain familiarity vs decomposability" framing is too narrow; F-2 generalizes. Reject (AUROC ≈ 0.5, cos < 0.2) → MATH-500 prefill correctness is genuinely task-specific and ITI's cross-task lift is a TruthfulQA artifact.
**Blocks:** H-5 (familiarity framing).

### H-86: A position-invariant per-head ITI intervention raises MATH-500 K=1 accuracy on Qwen
**Priority:** MEDIUM
**Motivated by:** 2306.03341 + F-3 + H-1
**Test:** Extract Qwen per-head mass-mean shift directions on top-K=32 prefill-correctness heads, inject σ-scaled at every autoregressive token during MATH-500 greedy generation (subset 100 problems), sweep α ∈ {5, 10, 15, 20}. Measure accuracy delta vs 48.6% baseline + CE/KL drift on OWT subset.
**Requires:** 4h H100 with intervention hooks. Blocks on intervention infra (same blocker as P11-FE3).
**Would change:** Confirm (≥+5pp accuracy at α=15, KL<0.5) → F-3's rotation finding is incomplete at the per-head level; per-head correctness subspace is position-stable; H-1 weakens (no per-position bank needed). Reject → F-3 holds for reasoning tasks, position-invariant injection insufficient, per-position DoM bank (H-1) is the right move.
**Blocks:** H-1 (per-position DoM bank).

### H-87: Qwen prefill correctness DoM saturates at fewer than 80 labeled examples
**Priority:** LOW
**Motivated by:** 2306.03341 + F-2 + F-9
**Test:** Sub-sample MATH-500 prefill train to N ∈ {20, 40, 80, 160, 320}, fit L19 DoM, compute OOF AUROC + cos(N-DoM, full-DoM) on held-out. Plot saturation curve.
**Requires:** 20min CPU on cached NPZs.
**Would change:** Confirm → prefill direction is low-rank, label-cheap; supports unsupervised label-free extension (H-12). Reject → high label complexity; CoE-style trajectory features (F-9) may carry independent information after all.
**Blocks:** nothing.

---


### H-88: Per-head, post-MHA Qwen probing exceeds residual-stream L19 DoM for prefill correctness
**Priority:** HIGH
**Motivated by:** 2306.03341 + F-2
**Test:** Extract per-head pre-output-projection activations across all (l, h) pairs on Qwen-2.5-1.5B for the cached MATH-500 prefill set, fit per-head linear probes for K=1 correctness, rank by OOF AUROC. Compute (a) max single-head AUROC, (b) top-K head DoM bank AUROC for K ∈ {16, 32, 48}. Compare to F-2's 0.7731.
**Requires:** 30min H100 (one fresh forward pass to capture per-head activations) + 1h CPU; cached MATH-500 prefill set + correctness labels.
**Would change:** Confirm → F-2 should be re-anchored on the canonical (l, h) head, not residual stream L19; F-9 (CoE redundancy) needs revisiting against multi-head bank. Reject → residual-stream L19 is genuinely the best target and ITI's head-wise advantage is LLaMA-architecture-specific.
**Blocks:** H-13 (head-level attribution).

### H-89: A single TruthfulQA-style fixed correctness direction transfers across reasoning tasks on Qwen
**Priority:** HIGH
**Motivated by:** 2306.03341 + F-2 + H-5
**Test:** Fit Qwen-2.5-1.5B L19 prefill DoM on a TriviaQA closed-book correctness set (~1k items), apply zero-shot to cached MATH-500 prefill activations, measure AUROC on MATH-500 K=1 correctness; also measure cos(MATH_DoM, TriviaQA_DoM).
**Requires:** 30min H100 (TriviaQA closed-book generation) + 30min CPU (probe fitting + transfer).
**Would change:** Confirm (AUROC > 0.65, cos > 0.5) → correctness axis is partly task-general; H-5's "domain familiarity vs decomposability" framing is too narrow; F-2 generalizes. Reject (AUROC ≈ 0.5, cos < 0.2) → MATH-500 prefill correctness is genuinely task-specific and ITI's cross-task lift is a TruthfulQA artifact.
**Blocks:** H-5 (familiarity framing).

### H-90: A position-invariant per-head ITI intervention raises MATH-500 K=1 accuracy on Qwen
**Priority:** MEDIUM
**Motivated by:** 2306.03341 + F-3 + H-1
**Test:** Extract Qwen per-head mass-mean shift directions on top-K=32 prefill-correctness heads, inject σ-scaled at every autoregressive token during MATH-500 greedy generation (subset 100 problems), sweep α ∈ {5, 10, 15, 20}. Measure accuracy delta vs 48.6% baseline + CE/KL drift on OWT subset.
**Requires:** 4h H100 with intervention hooks. Blocks on intervention infra (same blocker as P11-FE3).
**Would change:** Confirm (≥+5pp accuracy at α=15, KL<0.5) → F-3's rotation finding is incomplete at the per-head level; per-head correctness subspace is position-stable; H-1 weakens (no per-position bank needed). Reject → F-3 holds for reasoning tasks, position-invariant injection insufficient, per-position DoM bank (H-1) is the right move.
**Blocks:** H-1 (per-position DoM bank).

### H-91: Qwen prefill correctness DoM saturates at fewer than 80 labeled examples
**Priority:** LOW
**Motivated by:** 2306.03341 + F-2 + F-9
**Test:** Sub-sample MATH-500 prefill train to N ∈ {20, 40, 80, 160, 320}, fit L19 DoM, compute OOF AUROC + cos(N-DoM, full-DoM) on held-out. Plot saturation curve.
**Requires:** 20min CPU on cached NPZs.
**Would change:** Confirm → prefill direction is low-rank, label-cheap; supports unsupervised label-free extension (H-12). Reject → high label complexity; CoE-style trajectory features (F-9) may carry independent information after all.
**Blocks:** nothing.

---


### H-92: LEACE-scrubbed L19 prefill features still predict correctness better than chance
**Priority:** HIGH
**Motivated by:** 2306.03819 + F-2
**Test:** Fit LEACE on Qwen-1.5B prefill L19 train fold with K=1 correctness label; transform held-out fold; refit linear probe; compute OOF AUROC. Compare against the 0.7731 baseline. If post-LEACE AUROC ≥ 0.55, F-2 is showing a non-linear residual the DoM linear probe is silently leveraging. If ~0.5, F-2's linear framing holds. ~20min CPU, all data cached.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/` NPZ; `pip install concept-erasure`.
**Would change:** ON CONFIRM (≥0.55) — F-2's interpretation must be downgraded to "linear-readable + non-linear residual"; new hypothesis space about what kind of non-linear correctness signal exists at L19. ON REJECT (~0.5) — F-2 strengthened with a strict-null lower bound; we can claim "all linear correctness information at L19 is captured by the DoM direction" rather than just "DoM finds it."
**Blocks:** Determines whether non-linear-probe extensions of F-2 are worth pursuing.

### H-93: Concept-scrubbing L19 correctness across all layers degrades MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2306.03819 + F-2 + H-1
**Test:** Run Algorithm 1 (LEACE concept-scrubbing) across all 28 transformer blocks of Qwen-1.5B with K=1 correctness as the target concept. Generate K=1 outputs for MATH-500 with scrubbed forward pass. Compare accuracy (baseline 48.6%) and bits-per-byte perplexity. Test against (a) random-projection control, (b) single-layer-only LEACE at L19. ~1h H100.
**Requires:** H100 pod; forward-hook injection of LEACE projection at each block input post-LN; concept-erasure repo.
**Would change:** ON CONFIRM (accuracy collapses to ≤10%) — correctness is causally encoded as a linear concept distributed across the residual stream; H-1 (steering moves accuracy) gets a closed-form alternative; F-2 gets a causal counterpart. ON PARTIAL (accuracy drops 10–30%) — correctness is partially linear / distributed; refines the spatial extent of F-2. ON NULL (accuracy unchanged) — the L19 linear DoM is correlational, not causal — major reframing of F-2 and the entire DoM line.
**Blocks:** H-1 (per-position DoM steering) — if scrubbing has no causal effect, steering is unlikely to either.

---


### H-94: Prefill L19 DoM AUROC localizes to FFN-output, not MHSA-output
**Priority:** HIGH
**Motivated by:** 2308.08742 + F-2
**Test:** Train OOF logistic DoMs on MHSA-out (a^L19), FFN-out (m^L19), and pre-L19 residual (h^L18) separately on the cached MATH-500 prefill set; compare AUROCs against the 0.7731 whole-residual baseline. PMET predicts FFN-only AUROC ≈ 0.77, MHSA-only ≈ 0.55. ~30min CPU if streams cached, 3h H100 if re-extracting.
**Requires:** Cached MHSA-out and FFN-out streams at L19 (or 3h H100 to extract); existing MATH-500 correctness labels.
**Would change:** If FFN-only AUROC dominates, F-2 is confirmed as a knowledge-content axis and PMET's framework transfers cleanly. If MHSA-only AUROC dominates, F-2 is reinterpreted as an extraction/routing signal — connects strongly to H-13 (head-level attribution) and elevates its priority to CRITICAL. If both are weak individually but the residual-stream sum is high, PMET's clean MHSA/FFN separation breaks down for Qwen and the framework needs a non-additive component model.
**Blocks:** H-13 priority depends on outcome.

### H-95: FFN states in Qwen-2.5-1.5B stabilize before L19 (PMET-style transition exists)
**Priority:** MEDIUM
**Motivated by:** 2308.08742 + F-2 + F-9
**Test:** Per-layer cosine + top-50 Jaccard (vocab-projected) analysis of last-token hidden states before/after MHSA and before/after FFN, across all 28 layers, on 1209 factual statements (PMET's protocol) AND on the MATH-500 prefill set. Identify the layer at which FFN-out cosine similarity to its input exceeds 0.95 and stays there. PMET found L15/28 in GPT-J. ~1h CPU if streams cached.
**Requires:** Either cached MHSA-out and FFN-out streams at every layer, or 3h H100 to extract; LM head matrix for vocab projection.
**Would change:** Confirms whether L19 is at or past the FFN-stabilization boundary, gating whether F-2's framing as a "knowledge axis" is internally consistent. If FFN never stabilizes in Qwen, the GPT-J finding doesn't transfer and PMET-derived predictions for F-2/F-3 weaken. If stabilization happens at L19±2, our canonical layer choice is independently grounded.
**Blocks:** H-94, H-96.

### H-96: FFN-out-targeted DoM steering shifts MATH-500 accuracy more than MHSA-out-targeted steering
**Priority:** MEDIUM
**Motivated by:** 2308.08742 + H-1
**Test:** Two-arm ablation on 100 MATH-500 problems: (A) hook adds α·v_DoM to a^L19 (MHSA-out) only, (B) hook adds α·v_DoM to m^L19 (FFN-out) only. Compare K=1 accuracy delta against residual-stream-injection baseline. ~4h H100.
**Requires:** Qwen-2.5-1.5B + transformer-lens-style hooks + cached prefix activations + the v_DoM vector from F-2.
**Would change:** If FFN-out steering has a larger accuracy delta, replicates PMET's "FFN = factual content" framing for our regime and validates H-1 as a knowledge-content intervention. If MHSA-out has a larger delta, H-1 is an extraction-pattern intervention — reframes the steering story toward attention/routing, connects to H-21. If neither component alone reproduces the residual-stream baseline delta, the signal is genuinely distributed — refutes PMET's clean separation.
**Blocks:** nothing.

---


### H-97: A single ActAdd contrast-pair vector at L19, position a=last_user_token, moves MATH-500 K=1 accuracy
**Priority:** HIGH
**Motivated by:** 2308.10248 + F-2 + F-3
**Test:** Build ActAdd vector at Qwen-1.5B L19 from 3-5 correctness-targeted contrast prompt pairs; inject at last user-prompt position with c ∈ {3,5,8,12,15}; rerun MATH-500 K=1 against the cached `pathway11_h100/prefill_gated_compute` baseline. Measure accuracy delta and ConceptNet off-target P@1. ~1h H100.
**Requires:** H100, Qwen-1.5B base + cached Stage 2 extractor; no new training data.
**Would change:** If +Δacc > 3pp at any c, F-3's orthogonality is decoupled from steerability and H-1's per-position machinery is unmotivated (H-1 demoted to PARKED). If Δacc ≈ 0 across c, ActAdd-style fixed-position steering does not transfer to math correctness; H-1 strengthened as the *necessary* extension; refutation 1 of this brief is itself refuted.
**Blocks:** H-1 (until resolved).

### H-98: Our supervised L19 prefill DoM is the same direction a contrast-pair ActAdd would have discovered
**Priority:** HIGH
**Motivated by:** 2308.10248 + F-2
**Test:** Compute cos(F-2 supervised DoM, ActAdd vector) and head-to-head AUROC on cached MATH-500 prefill states for 3-5 contrast pair candidates. ~20 min CPU.
**Requires:** Cached Stage 2 NPZs only; no GPU.
**Would change:** If cos > 0.5 and AUROC within 0.02 of 0.7731 for any pair, F-2's supervision is rediscovering a direction one prompt pair would have surfaced — supervised-probe framing collapses to "ActAdd with a clever pair" and PAPER_INDEX status of 2308.10248 promotes from CITED ONLY → REPLICATED + EXTENDED. If max cos < 0.2, F-2 is genuinely orthogonal to all our contrast-pair guesses and has independent content.
**Blocks:** nothing.

### H-99: ActAdd's mid-layer optimum predicts the *steering-optimal* layer, which is upstream of our L19 *correctness-prediction-optimal* layer
**Priority:** MEDIUM
**Motivated by:** 2308.10248 (mid-layer optimum across GPT-2-XL/OPT/LLaMA-3) + F-2
**Test:** Layer sweep of ActAdd-vector AUROC across L0..L27 on cached Qwen-1.5B prefill states. ~1h CPU.
**Requires:** Per-layer cached NPZs; no GPU.
**Would change:** If the AUROC curve peaks meaningfully earlier than L19 (e.g. L13-L16), F-2's "L19 is the right layer" assumes correctness prediction = steering target, which is false — H-1 should target the steering-optimal layer, not L19. If the curve also peaks at L19, F-2's layer choice is convergently validated.
**Blocks:** H-1's layer choice.

---


### H-100: Prefill L19 DoM AUROC retains ≥ 0.7 within length-balanced MATH-500 subsets
**Priority:** HIGH
**Motivated by:** 2310.03716 + F-2 + EXP-038 (or whichever EXP delivered the 0.7731 number)
**Test:** Run P11-FE112. Bucket MATH-500 prompts by token count (5 quantile bins), compute prefill-DoM AUROC per bin and length-weighted within-bucket AUROC. Also run univariate length-only LR (P11-FE112 part 2) and shallow-feature LR (P11-FE113). Confirm condition: weighted within-bucket AUROC ≥ 0.7 AND length-only LR < 0.65 AND shallow-feature LR ≤ DoM AUROC by ≥ 0.05. Time: 30min CPU.
**Requires:** cached `pathway11_h100/prefill_gated_compute/` NPZs, MATH-500 prompt token counts (regenerate from cached prompts).
**Would change:** confirms F-2 is a substantively geometric (not shallow-lexical) signal; rejection means F-2 needs to be re-stated as "L19 DoM partially captures a length-confound that explains [X]% of AUROC, with [Y]% residual geometric content".
**Blocks:** any cross-model AUROC interpretation that assumes the within-1.5B 0.7731 is a clean geometric quantity (relevant for the breathing comparison and for cross-domain transfer in P9-FE3).

---


### H-101: Pathway 11 DoM signal lives in the non-positional (cvec/resid) subspace
**Priority:** HIGH
**Motivated by:** 2310.04861 + F-2 + F-3 + EXP-037
**Test:** On P11 cached L19 residual streams (Qwen-2.5-1.5B), compute Song-Zhong decomposition `h = μ + pos_t + ctx_c + resid`. Project prefill and final DoM directions onto span(P) (positional subspace, top-K right singular vectors of `P = [pos_1, …, pos_T]`) and onto span(P)^⊥ (non-positional). Report fraction of DoM-norm in each subspace. Refit DoM probe on `resid_{c,t}` only; report new AUROC.
**Requires:** CPU only; cached P11 NPZ (~2h)
**Would change:** *Confirm:* DoM is mostly in span(P)^⊥, AUROC on resid stays ≥0.75 — F-2 strengthened (the DoM is a real content signal, orthogonal to positional spiral by construction). *Reject:* DoM is mostly in span(P), AUROC on resid drops to <0.65 — F-2's "prefill DoM is a correctness predictor" was partly a positional/length-position artifact, and we need to re-examine the headline 0.7731 number.
**Blocks:** H-13, H-17, H-102

### H-102: F-10's "PH = Gaussian null" is a false negative caused by positional-basis dominance, not a true topological null
**Priority:** HIGH
**Motivated by:** 2310.04861 + F-10 + EXP-026
**Test:** Recompute the five PH summary features (H0_max_lifetime, H0_entropy, H1_pers_entropy, H1_n_features, H1_max_lifetime) on `resid_{c,t} = h - μ - pos_t - ctx_c` for L19, Qwen-2.5-1.5B, MATH-500. Compare classifier AUROC (5-fold OOF) against a rank-matched empirical-covariance Gaussian null built from the same `resid` covariance. Use Pathway 9 pipeline.
**Requires:** ~1 day CPU; existing pathway 9 ripser pipeline; cached P11 NPZ
**Would change:** *Confirm:* PH on `resid` beats null by >0.02 AUROC — F-10 is overturned, "no topology" was a methodology bug, and the topology-as-correctness-predictor program is back on the table. *Reject:* gap stays ≤0.005 — F-10 strengthened, null result is robust to pos+ctx subtraction.
**Blocks:** anything depending on F-10 staying at null (Pathway 7/8 abandonment justification)

---


### H-103: Prefill ⊥ final-token DoM is a mid-rotation snapshot at L19, not a global property
**Priority:** HIGH
**Motivated by:** 2310.06824 (App. C layer-wise rotation of cities/neg_cities truth axes) + F-3
**Test:** Compute cos(prefill_DoM_ℓ, final_DoM_ℓ) at every layer ℓ ∈ {0…27} of Qwen-2.5-1.5B using cached or freshly-extracted activations. Plot cos as a function of layer.
**Requires:** Cached prefill+final activations across all layers (or 30 min H100 to re-extract); 5 min CPU for plotting.
**Would change:** If cos(·) passes near ±1 at any layer, F-3's "directions are orthogonal" framing must be replaced with "directions rotate during the forward pass; orthogonality at L19 is one moment in the rotation." If cos remains near 0 at every layer, F-3 is robust and the orthogonality is genuinely geometric.
**Blocks:** nothing — but if confirmed, makes H-17 (DoM rotation as RoPE-mechanical) far more interesting because we'd have a layer-wise companion to the position-wise rotation.

### H-104: Σ⁻¹-corrected prefill DoM lifts AUROC above 0.7731
**Priority:** HIGH
**Motivated by:** 2310.06824 §5.1 (IID mass-mean as Σ⁻¹·(μ⁺−μ⁻)) + F-2
**Test:** Compute residual covariance Σ on the training fold of cached L19 prefill activations; compute θ_iid = Σ⁻¹·(μ⁺−μ⁻); rerun 5-fold OOF AUROC with θ_iid scoring. Compare against the existing raw-DoM 0.7731.
**Requires:** Cached L19 prefill activations (already exist); ~20 min CPU; numerical stability for 1536-dim Σ inversion (regularize with εI if needed).
**Would change:** If θ_iid lifts AUROC by ≥ 0.02, F-2's headline number gets revised upward and the project's "raw DoM is enough" claim weakens. If lift < 0.005, raw DoM is genuinely near the linear-classifier ceiling for Qwen-2.5-1.5B at L19.
**Blocks:** H-1 — should be settled before any DoM-steering experiment, since the steering direction we care about is whichever has the highest causal NIE, and Marks & Tegmark show that raw vs IID can differ.

### H-105: Prefill DoM steering NIE is < 0.20 without paired-contrast training data
**Priority:** CRITICAL
**Motivated by:** 2310.06824 Table 2 (cities-only NIE = 0.13/0.19 vs cities+neg_cities NIE = 0.85/0.97) + H-1 + F-8
**Test:** Implement the §6.1 NIE protocol on a 50-problem MATH-500 held-out set. Normalize θ_prefill_DoM so that p(μ⁻+θ)=p(μ⁺); intervene at prefill[-1] for false→true and true→false directions; measure NIE on log-P(correct-token).
**Requires:** ~1 h H100; existing L19 DoM and cached extractor; small inference harness.
**Would change:** Confirm (NIE < 0.20): H-1 should be PARKED until a paired contrast set exists (motivates P11-FE123). Reject (NIE > 0.50): single-distribution DoM is causal *and* predictive — strengthens F-8 substantially and suggests MATH correctness has a different paired-vs-unpaired profile than truth-statement classification (publishable contrast).
**Blocks:** H-1 (per-position DoM steering rollout — should not run until NIE is calibrated).

---


### H-106: Prefill rows 0..(prefill_len-1) at L19 are causally inert for K=1 accuracy on MATH-500
**Priority:** HIGH
**Motivated by:** 2310.13121 §7 (mean-ablation showed rows 0-10 unused in 1-layer addition transformer) + F-2
**Test:** Mean-ablate (replace with dataset-mean activation) prefill rows at L19 in cached `pathway11_h100/prefill_gated_compute/` activations; rerun the prefill-gated-compute pipeline and recompute K=1 accuracy + L19-DoM AUROC. ~30min CPU + 30min H100 if re-rollout needed.
**Requires:** Cached Stage 2 NPZs (already on disk per DATA_MANIFEST.md), Qwen-2.5-1.5B for re-rollout if accuracy changes.
**Would change:** If accuracy unchanged but DoM AUROC drops to ~0.5, F-2 reframes from "prefill encodes correctness" to "prefill DoM is a problem-identity probe correlated with eventual correctness" (H-5). If both unchanged, prefill is an active reasoning trace and F-2 stands. If accuracy degrades, prefill is structurally load-bearing and Quirke-Barez's toy result does not generalize.
**Blocks:** H-1 (per-position bank steering) is predicated on the prefill direction being meaningful — if H-106 confirms inertness, H-1 priors must shift.

### H-107: Prefill–final DoM cosine becomes ≥ 0.3 after per-position RoPE rotation
**Priority:** HIGH
**Motivated by:** 2310.13121 §7 (position-specific algorithms in 1-layer addition) + F-3
**Test:** Apply per-position RoPE rotation matrices to the prefill L19 DoM direction; sweep all token positions in the answer; compute cosine vs final-token L19 DoM at each position; report min / median / max. ~1h CPU on cached activations.
**Requires:** RoPE matrices from Qwen-2.5-1.5B config (deterministic), cached prefill + final L19 DoM directions.
**Would change:** Confirm → F-3 reframes from "different signals" to "same signal in rotated bases" and cos 0.046 is the predicted consequence of RoPE not a finding. Reject → F-3's independent-signal interpretation strengthens.
**Blocks:** H-1, H-17.

### H-108: Quirke-Barez 1-layer addition transformer shows no breathing
**Priority:** MEDIUM
**Motivated by:** 2310.13121 (release of 1-layer interpretable addition transformers) + F-1
**Test:** Clone `apartresearch/Integer_Addition`, load 5/10/15-digit checkpoints, extract per-token residual-stream activations, compute PR per token. Half-day CPU.
**Requires:** Public checkpoints (released), CPU only.
**Would change:** Confirm (no breathing in their 1-layer model) → F-1's universality narrows to multi-layer / depth-dependent. Reject (clear inverted-U PR curve) → F-1 universality strengthens; the simplest interpretable transformer breathes.
**Blocks:** nothing.

---


### H-109: Sparse-discrete codes at L19 outperform the DoM linear probe for MATH-500 correctness prediction
**Priority:** HIGH
**Motivated by:** 2310.17230 + F-2 + F-8
**Test:** Train a single-layer codebook bottleneck at L19 on Qwen-2.5-1.5B
(frozen body, cosine top-k retrieval with k=8, C=4096, MSE-aux loss with λ=1
and stop-gradient, straight-through estimator), per the Tamkin et al. recipe.
Score the highest-precision correctness-code at matched recall and compare
against the L19 prefill DoM probe AUROC of 0.7731. Free no-train sanity check
first via k-means at k ∈ {64, 256, 1024}.
**Requires:** ~6h H100 for codebook fine-tune; 30 min CPU for k-means
sanity. All MATH-500 activations cached (`pathway11_h100/`).
**Would change:** If a discrete code beats DoM by ≥3 pp AUROC, F-2's
"DoM is the signal" reading becomes "DoM is the linear shadow of a
sparse code" and F-8's 71.6% selective-prediction number is no longer
the residual-stream ceiling. If codes match DoM but don't exceed it,
F-2 / F-8 are robust to the structural-prior change.
**Blocks:** H-110 (codebook required for steering test).

### H-110: Codebook substitution at L19 produces a larger MATH-500 K=1 accuracy shift than per-position DoM addition at the same layer
**Priority:** HIGH
**Motivated by:** 2310.17230 + H-1 + 2308.10248 (ActAdd)
**Test:** Using the H-109 codebook, identify the top correctness-codes and
activate them during forward-pass generation on the K=1 MATH-500 pipeline,
matching the per-position DoM-steering protocol of H-1. Compare ΔAccuracy.
**Requires:** ~1d H100 (depends on H-109), pathway-11 generation harness.
**Would change:** If codebook steering beats DoM steering by ≥1.5x lift,
H-1's intervention vehicle (dense vector addition) is the wrong abstraction
and the steering program should pivot to discrete-code substitution.
If DoM matches or beats codebook steering, the dense-direction account
is vindicated and H-1 stays as designed.
**Blocks:** nothing.

### H-111: Per-layer codebooks at L13, L19, L25 capture non-overlapping correctness-relevant code sets, even though their linear readouts are correlated
**Priority:** MEDIUM
**Motivated by:** 2310.17230 + F-9
**Test:** Train independent codebooks at L13, L19, L25 (same recipe as H-109).
Compute the Jaccard overlap of "high-precision correctness codes" across
layers. Separately, refit the F-9 CoE-60 ablation but using each layer's
correctness-codes as features. Compare predictive AUROC of the 3-layer
code-feature stack against single-layer L19 DoM.
**Requires:** ~18h H100 (3 codebooks); cached activations.
**Would change:** If layer-specific codes have low Jaccard overlap (< 0.3)
but their linear readouts are highly correlated, F-9's "redundancy" claim
must be qualified to "linear redundancy in the residual stream, not
feature-level redundancy in the codebook decomposition." If overlap is
high, F-9 is robust.
**Blocks:** nothing.

---


### H-112: Diagonal hidden-state extraction beats single-layer DoM on MATH-500 correctness probing
**Priority:** HIGH
**Motivated by:** 2311.01460 + F-9 + F-2
**Test:** On cached Qwen-2.5-1.5B P11 prefill activations (28 layers × prompt tokens for each MATH-500 problem), apply Deng et al.'s diagonal indexing `t_l = ⌊1 + ∆(l−1)⌋` with `∆ = (T−1)/(L−1)`, normalize each selected vector per the paper (zero-mean unit-std), fit a logistic probe on the 28-layer concatenation. Compare AUROC against F-2's 0.7731 single-layer L19 DoM. Time: 4h CPU.
**Requires:** Cached P11 1.5B 1024-tok prefill activations; CPU only.
**Would change:** If diagonal AUROC ≥ 0.79, F-9's "CoE-60 redundant with single-layer L19" needs domain qualifier — a different *type* of multi-layer feature (paper-style diagonal, with normalization) is non-redundant on within-domain MATH-500 even if CoE-60 was redundant on MATH↔BBH transfer. Also rules out whether we have been leaving headline-AUROC lift on the table for two months.
**Blocks:** P11-FE132; weakly blocks H-6 (CoE-60 vs DoM at 1024-tok) since diagonal is a third candidate.

### H-113: Prefill and final-token DoM projections are both input-determined
**Priority:** MEDIUM
**Motivated by:** 2311.01460 + F-3
**Test:** Train a 2-layer MLP from prompt-token-mean L0 activations on cached 1.5B 1024-tok data to predict (a) prefill L19 DoM-projection and (b) final-token L19 DoM-projection. Report R² for each. Time: 2h CPU.
**Requires:** Cached 1.5B 1024-tok activations (L0 + L19); CPU only.
**Would change:** If R²_a > 0.5 and R²_b > 0.5, F-3's cos = 0.046 stops being evidence of "new computation across positions" and becomes evidence of "two near-orthogonal residual subspaces both written from input alone". F-3's narrative shifts from "trajectory rotation" to "static dual-channel encoding". If both R² < 0.3, F-3 stands and the rotation is non-trivial.
**Blocks:** P11-FE133; relevant to H-17 (DoM rotation is RoPE-mechanical) — if both directions are input-determined, RoPE mechanics aren't doing the work either.

### H-114: F-2's AUROC ceiling is probe-bound
**Priority:** HIGH
**Motivated by:** 2311.01460 + F-2
**Test:** Fit a 2-layer MLP with all 28 layer-mean activations concatenated as input on cached 1.5B 1024-tok data, 5-fold CV for binary correctness. Compare AUROC against F-2's 0.7731. Time: 6h CPU.
**Requires:** Cached 1.5B 1024-tok activations across all 28 layers; CPU only.
**Would change:** If MLP AUROC ≥ 0.82, F-2 reframes as "the *current linear single-layer probe* extracts 0.7731; the prefill state encodes substantially more correctness signal", and F-8's selective-prediction ceiling at 50% coverage may be loose. If AUROC ≤ 0.78, F-2's ceiling is signal-bound, which is also a useful finding (suggests prefill-DoM near optimal).
**Blocks:** P11-FE134; relevant to F-8 selective-prediction interpretation.

---


### H-115: Mahalanobis-whitened prefill probe matches or beats raw DoM AUROC of 0.7731 on Qwen-2.5-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2311.03658 (Theorem 2.2) + F-2 + F-9
**Test:** Compute Cov(γ_Qwen2.5-1.5B) from `lm_head.weight`. Build features Cov(γ)^{-1/2} · h_L19_prefill. Refit logistic regression on cached prefill activations with 5-fold OOF, compare AUROC and Brier score against vanilla DoM (0.7731, 0.205). Est time: 1h CPU.
**Requires:** CPU only. Cached prefill activation NPZs from P11. Qwen-2.5-1.5B unembedding matrix from HF.
**Would change:** If AUROC > 0.80 with better calibration, F-2 narrative gains a theoretical scaffold and F-9 redundancy is *predicted* by LRH unification — also strengthens the case that L19 is the canonical layer for correctness. If AUROC unchanged within ±0.01, vanilla DoM was already saturating LRH limit. If AUROC drops, correctness fails the LRH frame at L19 and we close the LRH chapter for reasoning-grade concepts.
**Blocks:** H-116 (depends on the implementation of Cov(γ)^{-1} being validated)

### H-116: Cos(prefill_DoM, final_DoM) under causal inner product is ≫ 0.046
**Priority:** HIGH
**Motivated by:** 2311.03658 (Theorem 3.2) + F-3
**Test:** After P11-FE136 / H-115, compute ⟨γ̄_prefill, γ̄_final⟩_C / sqrt(⟨γ̄_prefill, γ̄_prefill⟩_C · ⟨γ̄_final, γ̄_final⟩_C). Expectation under LRH: cos ≫ 0 (probably > 0.5). Observation under Euclidean: 0.046. Est time: minutes.
**Requires:** CPU only. Same cached artifacts as H-115.
**Would change:** If whitened cos > 0.5, F-3 is reframed: prefill and final DoM are *the same direction* in the right coordinates, and the orthogonality was a Euclidean-coordinate artifact. If whitened cos still < 0.15, F-3 is a *true* representational fact and the LRH unification fails for chain-of-thought correctness — large negative result for the LRH program's applicability beyond word-level concepts.
**Blocks:** nothing

### H-117: LRH-prescribed steering λ̄_W = Cov(γ)^{-1} γ̄_DoM raises K=1 accuracy on incorrect-labeled MATH-500 problems with bounded off-target drift
**Priority:** MEDIUM
**Motivated by:** 2311.03658 (Theorem 2.5) + H-1
**Test:** P11-FE138 sweep (α ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4}), L19 prefill of 257 incorrect 1.5B problems, K=1 generation. Score: flip rate, length distribution drift, language drift, on-topic preservation. Est time: 1 H100-day.
**Requires:** H100 pod, Qwen-2.5-1.5B, prefill activation hook at L19, cached DoM direction from F-2.
**Would change:** Confirm: H-1 promoted from "unconfirmed" to "confirmed" with a principled LRH-derived recipe — major win for the project narrative. Reject: with no flip rate increase, both H-1 and the paper's Theorem 2.5 fail for chain-of-thought correctness — significant negative result.
**Blocks:** any future steering-based intervention proposals (P11-FE3, P11-FE5)

---


### H-118: Label-free mean-centred direction matches supervised DoM AUROC at L19
**Priority:** HIGH
**Motivated by:** 2312.03813 + F-2
**Test:** Cache 50–100 OpenWebText documents through Qwen-2.5-1.5B at L19, compute `µ_OWT`. Compute `µ_correct` over the 243 correct-K=1 problems' prefill activations; let `f_mc = µ_correct − µ_OWT`. Project all 500 prefill activations onto `f_mc`, report AUROC. Also compute `cos(f_mc, DoM_supervised)` and logit-lens top-5 tokens for `f_mc`. ~2h CPU.
**Requires:** Cached L19 prefill NPZ (DATA_MANIFEST item), cached or freshly-computed OWT activations at L19 (no GPU needed for forward pass on 50 docs × 100 tokens), supervised DoM JSON.
**Would change:** If AUROC(f_mc) ≥ 0.74 and cos > 0.9, F-2's "supervised contrast is what matters" framing collapses; the predictive signal is "MATH-prompt centroid offset from generic text," and supervision is unnecessary. Strengthens H-12. If AUROC < 0.7, the supervised contrast is doing real work, and the paper's "implicit mean-centring" conjecture fails for our position-conditioned regime.
**Blocks:** H-12 evaluation framework (this is the cheapest label-free probe candidate).

### H-119: F-3's prefill/final-token DoM orthogonality survives position-bias correction
**Priority:** MEDIUM
**Motivated by:** 2312.03813 + F-3
**Test:** Compute `b_prefill = µ(prefill, all 500)` and `b_final = µ(final, all 500)` at L19. Compute residual DoMs after projecting out the bias direction `(b_prefill − b_final)`. Report cos(residual_DoM_prefill, residual_DoM_final). ~1h CPU on cached NPZs.
**Requires:** Cached prefill + final-token L19 activations.
**Would change:** If residual cos < 0.1, F-3's "orthogonal computations" claim is robust to anisotropy correction. If residual cos > 0.3, F-3 must be reframed as "orthogonal modulo position-bias" and the prefill/final-token decoupling story weakens.
**Blocks:** Any future intervention experiment that assumes prefill and final-token DoMs operate on disjoint subspaces (currently nothing depends on it directly, but it underpins H-1 design).

---


### H-120: CAA-style position-uniform application of prefill-DoM moves MATH-500 K=1 accuracy at multiplier ±1 (or it doesn't, refuting H-1)
**Priority:** HIGH
**Motivated by:** 2312.06681 + F-2 + F-8
**Test:** Build v_correct = mean(prefill_L19[correct]) − mean(prefill_L19[incorrect]) on cached 1024-tok activations, apply at every post-prompt position with multiplier ∈ {−2, −1, 0, +1, +2} on 100 held-out MATH-500 problems. Compare K=1 accuracy + GPT-4 fluency rating across multipliers. Est: 8h H100.
**Requires:** H100 + cached prefill activations + Qwen-2.5-1.5B chat checkpoint.
**Would change:** Confirm → H-1 receives first positive evidence and CAA-uniform becomes our default intervention shape. Reject → H-1 must be reframed (per-position is *required*, not just one option), and F-2 is downgraded to "diagnostic-only".
**Blocks:** H-1, P10-FE1.

### H-121: Optimal layer for prefill correctness MD on Qwen-2.5-1.5B is L19 (or it isn't, weakening F-2)
**Priority:** HIGH
**Motivated by:** 2312.06681 + F-2
**Test:** Per-layer prefill MD-vector AUROC sweep on cached 1024-tok activations for L ∈ {0..27}, 5-fold OOF. Argmax + ±0.005 confidence band. Compare to CAA's "⅓-depth" pattern (would predict L9-L11 on Qwen). Est: 20min CPU.
**Requires:** Cached pathway11_h100 NPZs.
**Would change:** Confirm L19 → F-2's layer choice has provenance. Reject → F-2 headline number and figures need re-rendering at the true argmax layer; H-1/H-13/H-17 all shift.
**Blocks:** H-1, F-2 narrative claim about "L19 is the operating depth".

### H-122: MATH-500 prefill activations cluster *behaviourally* (correct/incorrect) at L19 in 2D PCA, not only by emitted answer letter
**Priority:** MEDIUM
**Motivated by:** 2312.06681 (Section 3.2 diagnostic) + F-2
**Test:** PCA(prefill_L19_activations) on cached 500 problems, scatter coloured by (a) emitted answer letter, (b) correct/incorrect. Manual inspection per CAA Figure 2. Est: 30min CPU.
**Requires:** Cached pathway11_h100 prefill NPZs.
**Would change:** Confirm → F-2 has the missing dataset-quality diagnostic. Reject → F-2 is letter-emission signal rather than correctness signal, and the 0.7731 number must be reinterpreted.
**Blocks:** Nothing — quick provenance check.

---


### H-123: D-bucket's group-vs-per-problem PR gap is local dimensional collapse
**Priority:** HIGH
**Motivated by:** 2401.10474 + F-7
**Test:** Compute per-problem prefill L19 LID (Method-of-Moments, k=32) on cached P11 H100 1.5B activations. Compare geometric-mean LID across A/B/C/D buckets. If D-bucket geometric-mean LID is significantly lower than A/B/C while bucket-level PR ordering reproduces, F-7 is reframed as bucket-level local-LID collapse — a known phenomenon with a clean theoretical home (Huang et al. 2024) rather than a novel "collective vs per-problem" dichotomy.
**Requires:** CPU only (~1h on cached `pathway11_h100/prefill_gated_compute/*.npz`).
**Would change:** confirm → F-7 narrative simplifies to "D-bucket has lower local LID at prefill," removes the "dissolves per-problem" framing, and suggests H-7 (density-based outlier detection) is best implemented as LID rather than k-NN density. Reject → F-7 retains its novelty as a non-local geometric phenomenon.
**Blocks:** any future write-up that cites F-7 as a novel signature.

### H-124: F-4 asymmetric-collapse gap is robust to geometric-mean aggregation
**Priority:** MEDIUM
**Motivated by:** 2401.10474 (Theorem 4) + F-4
**Test:** Re-aggregate correct-vs-incorrect final-token PR (and final-token L19 LID, k=32) using geometric mean of log-values, with bootstrap CIs. Compare effect size against arithmetic mean.
**Requires:** CPU only (~20min on cached final-token NPZs).
**Would change:** confirm → F-4 strengthens with theoretical backing (Fisher-Rao geometry); narrative gains "asymmetric collapse is robust to aggregation choice." Reject → F-4 is partially an arithmetic-mean artifact; need to revisit which PR statistic to use throughout the paper.
**Blocks:** nothing; runs in parallel to H-16 (Marchenko-Pastur correction).

### H-125: Local-LID-conditioned DoM probes beat the global F-2 DoM
**Priority:** MEDIUM
**Motivated by:** 2401.10474 + F-2 + F-7
**Test:** Median-split MATH-500 by per-problem prefill L19 LID (k=32). Train 5-fold OOF DoMs separately per half. Compare combined AUROC against global F-2 DoM AUROC (0.7731). Significance via paired bootstrap.
**Requires:** CPU only (~2h on cached prefill activations).
**Would change:** confirm → F-2's single-direction story is incomplete; suggests per-cluster or LID-conditioned probes as a path to higher selective-prediction accuracy (currently 71.6% at coverage 0.5, F-8). Reject → F-2's global DoM is robust to local geometry heterogeneity.
**Blocks:** any selective-prediction follow-up that wants to push past F-8's 71.6%.

---


### H-126: Prefill L19 DoM is primarily an input-geometry detector, with correctness AUROC inherited from input↔correctness correlation
**Priority:** HIGH
**Motivated by:** 2401.13558 + F-2
**Test:** Run P11-FE154 (CKA) and P11-FE155 (CCGP across topics) on cached Pathway 11 NPZs. If CKA(L19, problem-tokens) >> CKA(L19, correctness), and CCGP-across-topics drops below 0.65, hypothesis confirmed.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/results.json` and raw MATH-500 problem statements.
**Would change:** F-2 reframes from "correctness direction" to "problem-feature axis"; F-8 (selective prediction) reinterpreted as input-feature gating; recasts the entire prefill-DoM finding.
**Blocks:** revising the F-2 narrative in PROJECT_RECORD §1b and FINDINGS.md F-2 conclusion.

### H-127: Modern transformer residual streams obey the paper's "ReLU-input-preservation" prediction; cos(prefill, final) ≈ 0.046 is generic, not exceptional
**Priority:** MEDIUM
**Motivated by:** 2401.13558 + F-3
**Test:** Run P11-FE157 (pairwise cos for many probes at L19). If pairwise cos for arbitrary-label probes clusters in 0.02–0.10, F-3's value is not informative. If cos(prefill, final) is uniquely low while same-label probes from different layers stay >0.3, F-3 stands.
**Requires:** CPU; cached L19 activations + MATH-500 metadata for ≥5 binary labels.
**Would change:** F-3's "asymmetric circuits" framing; either confirms F-3 stands as a real result, or downgrades it to expected null behavior.
**Blocks:** any paper-write-up sections framing F-3 as causally meaningful.

### H-128: F-4 (asymmetric collapse) is mediated by RMSNorm saturation, not by raw residual-stream dynamics
**Priority:** MEDIUM
**Motivated by:** 2401.13558 + F-4
**Test:** Run P11-FE158 (pre-norm vs post-norm PR). Confirms if post-norm asymmetric-collapse magnitude is ≥1.5× pre-norm; rejects if both streams show identical asymmetry.
**Requires:** CPU; both pre-norm and post-norm residual cached at the answer-final position. May need re-extraction if only one is currently cached.
**Would change:** Adds a saturation-mechanism paragraph to F-4. Connects F-4 to literature on normalization-induced collapse (a separate research direction we have not engaged with).
**Blocks:** nothing.

---


### H-129: Label-free ISFI (Zhao et al. closed-form) matches supervised L19 DoM AUROC on Qwen-2.5-1.5B prefill
**Priority:** HIGH
**Motivated by:** 2402.00865 + F-2 (prefill L19 DoM AUROC 0.7731 OOF)
**Test:** Compute ISFI vectors I(z) for Qwen-2.5-1.5B L19 prefill activations using
the unembedding row for the gold/top-predicted token as w^max. Form
θ* = sqrt(K)/||E[I(z)]|| · E[I(z)] on the same 5-fold split as F-2. Score test
folds via shaped MLS/Energy. Compare AUROC against 0.7731.
**Requires:** ~30min CPU on cached `pathway11_h100/prefill_gated_compute` NPZs.
No new data, no GPU.
**Would change:** If AUROC ≥ 0.74, F-2's claim that supervised label-fitting
beats final-token DoM is downgraded — the gain is from picking the right
*layer* (L19) and the right *bin reweighting*, not from labels per se. If
AUROC < 0.65, F-2 is robustly supervised and the closed-form is insufficient.
**Blocks:** H-130 (a positive H-129 result motivates extending ISFI to other layers).

### H-130: Coarse-K percentile clipping (K≤10) isolates D-bucket as well as density-based outlier detection
**Priority:** MEDIUM
**Motivated by:** 2402.00865 (Fig. 4c K-plateau) + F-7 + H-7
**Test:** Sweep K ∈ {1, 2, 5, 10, 20, 50, 100} of ISFI on the 36 D-bucket vs 207
A-bucket prefill activations. Plot AUROC vs K. Compare K=10 AUROC to LOF /
IsolationForest baselines (H-7 candidates) on the same subset.
**Requires:** ~1h CPU on cached activations.
**Would change:** If K=10 ISFI matches LOF within ±0.02, H-7 is downgraded to
LOW or merged with H-129; we'd recommend percentile-clipping as the canonical
D-bucket isolator for its cost and interpretability.
**Blocks:** nothing.

### H-131: Negative-magnitude L19 prefill features carry incorrectness signal (sign-flip hypothesis)
**Priority:** LOW
**Motivated by:** 2402.00865 (Sec. 3.2 sign-flip on low-z bins) + F-4
**Test:** Compute ISFI on incorrect-trajectory prefill activations restricted to
z<0 entries. Check sign of optimal θ_k for low-z bins. Score AUROC of (1) ReAct
clip baseline, (2) zero-low VRA-P style, (3) sign-flip ISFI on the same probes.
**Requires:** ~1h CPU.
**Would change:** Confirms or refutes that asymmetric collapse (F-4) is partly
driven by negative-magnitude information being discarded; suggests a one-line
fix to existing percentile-clip baselines.
**Blocks:** nothing.

---


### H-132: Cross-sample covariance LogDet (EigenScore) outperforms single-pass DoM on MATH-500

**Priority:** HIGH
**Motivated by:** 2402.03744 + F-2 + F-9
**Test:** Resample K=10 at T=0.5 on Qwen-2.5-1.5B over MATH-500.
Extract last-token L14 hidden states; compute EigenScore =
(1/K) Σ log λ_i(Σ+0.001·I); AUROC against ground-truth correctness. Same
cache → also compute K=10 majority-vote accuracy as control. Compare
EigenScore AUROC to F-2's 0.7731. ~6h H100 + 30min CPU.
**Requires:** H100 pod, Qwen-2.5-1.5B-Instruct, fresh K=10 generation cache
(MATH-500, 1024-tok labels protocol).
**Would change:** If EigenScore AUROC ≥ 0.80, F-9 ("CoE-60 redundant with
DoM") and F-8 ("prefill DoM is best selective predictor") both need revision —
multi-sample variance is a separate, dominant signal. If EigenScore AUROC ≤
0.70, the method doesn't transfer to math reasoning and the cross-sample axis
gets parked under H-12.
**Blocks:** H-133.

### H-133: Multi-sample EigenScore peaks at the int(L/2) middle layer (L14), not at L19

**Priority:** MEDIUM
**Motivated by:** 2402.03744 (Fig 3b) + F-2
**Test:** Per-layer EigenScore on the H-132 K=10 cache, layers {5, 10, 14, 19,
23, 27}. ~1h CPU.
**Requires:** K=10 hidden-state cache from H-132 (all layers).
**Would change:** If layer-best ≠ L19, F-2's "L19 is special" claim is
specific to supervised-DoM extraction, not a general property of the
representation. Forces re-running every "L19 is the canonical layer"
experiment under the multi-sample protocol. If layer-best = L19, F-2
generalizes.
**Blocks:** nothing.

### H-134: Penultimate-layer feature clipping (top/bottom 0.2%) shrinks the D-bucket without harming accuracy

**Priority:** MEDIUM
**Motivated by:** 2402.03744 (§3.2) + F-7
**Test:** Apply per-neuron clipping to penultimate-layer activations during
K=8 generation on Qwen-1.5B MATH-500. Compare D-bucket fraction (K=1 right,
K=8 wrong) and overall MATH-500 K=1 / K=8 accuracy against unclipped baseline.
~4h H100 + 30min CPU.
**Requires:** H100 pod, custom forward hook on Qwen layer L26 (penultimate of
28), memory bank of N=3000 token activations.
**Would change:** If D-bucket fraction drops by ≥30% without hurting K=1
accuracy, F-7's pathological-problem geometry is mechanistically linked to
penultimate-layer activation outliers — a generation-time intervention exists.
If no D-bucket change, F-7 is decoupled from INSIDE's overconfidence
mechanism.
**Blocks:** nothing.

---


### H-135: Cross-sample covariance LogDet (EigenScore) outperforms single-pass DoM on MATH-500

**Priority:** HIGH
**Motivated by:** 2402.03744 + F-2 + F-9
**Test:** Resample K=10 at T=0.5 on Qwen-2.5-1.5B over MATH-500.
Extract last-token L14 hidden states; compute EigenScore =
(1/K) Σ log λ_i(Σ+0.001·I); AUROC against ground-truth correctness. Same
cache → also compute K=10 majority-vote accuracy as control. Compare
EigenScore AUROC to F-2's 0.7731. ~6h H100 + 30min CPU.
**Requires:** H100 pod, Qwen-2.5-1.5B-Instruct, fresh K=10 generation cache
(MATH-500, 1024-tok labels protocol).
**Would change:** If EigenScore AUROC ≥ 0.80, F-9 ("CoE-60 redundant with
DoM") and F-8 ("prefill DoM is best selective predictor") both need revision —
multi-sample variance is a separate, dominant signal. If EigenScore AUROC ≤
0.70, the method doesn't transfer to math reasoning and the cross-sample axis
gets parked under H-12.
**Blocks:** H-136.

### H-136: Multi-sample EigenScore peaks at the int(L/2) middle layer (L14), not at L19

**Priority:** MEDIUM
**Motivated by:** 2402.03744 (Fig 3b) + F-2
**Test:** Per-layer EigenScore on the H-135 K=10 cache, layers {5, 10, 14, 19,
23, 27}. ~1h CPU.
**Requires:** K=10 hidden-state cache from H-135 (all layers).
**Would change:** If layer-best ≠ L19, F-2's "L19 is special" claim is
specific to supervised-DoM extraction, not a general property of the
representation. Forces re-running every "L19 is the canonical layer"
experiment under the multi-sample protocol. If layer-best = L19, F-2
generalizes.
**Blocks:** nothing.

### H-137: Penultimate-layer feature clipping (top/bottom 0.2%) shrinks the D-bucket without harming accuracy

**Priority:** MEDIUM
**Motivated by:** 2402.03744 (§3.2) + F-7
**Test:** Apply per-neuron clipping to penultimate-layer activations during
K=8 generation on Qwen-1.5B MATH-500. Compare D-bucket fraction (K=1 right,
K=8 wrong) and overall MATH-500 K=1 / K=8 accuracy against unclipped baseline.
~4h H100 + 30min CPU.
**Requires:** H100 pod, custom forward hook on Qwen layer L26 (penultimate of
28), memory bank of N=3000 token activations.
**Would change:** If D-bucket fraction drops by ≥30% without hurting K=1
accuracy, F-7's pathological-problem geometry is mechanistically linked to
penultimate-layer activation outliers — a generation-time intervention exists.
If no D-bucket change, F-7 is decoupled from INSIDE's overconfidence
mechanism.
**Blocks:** nothing.

---


### H-138: K=5 black-box NLI agreement (PDS) beats the prefill L19 DoM probe at matched compute on MATH-500
**Priority:** HIGH
**Motivated by:** 2402.10528 + F-2 + F-8
**Test:** Run SummaC-PDS (`min_j max_i (ent − con)` aggregated to PDS = (norm_ADS + PSS)/2) on the K=5 Qwen-2.5-1.5B MATH-500 rollouts in `pathway11_h100/prefill_gated_compute/`. Compute F1 / AUC-PR / risk-coverage; compare to prefill-DoM (AUROC 0.7731 OOF) and to F-8's 71.6% / 50%-coverage point. ~30–60min CPU.
**Requires:** CPU only; SummaC NLI weights (HF, ≤500M params); the K=5 rollout text from Stage 2.
**Would change:** Confirm → F-8's compute-savings story is wrong; rewrite as "DoM is good at K=1, PDS at K=5". Reject → F-8 strengthens; selective prediction via hidden-state probe is genuinely compute-efficient.
**Blocks:** H-139 (only meaningful if PDS is established as the relevant K=5 baseline).

### H-139: PDS contributes independent variance to correctness prediction beyond L19 DoM
**Priority:** HIGH
**Motivated by:** 2402.10528 + F-9 (CoE redundant with DoM)
**Test:** Residualize PDS against L19 DoM probe predictions via OLS; refit a logistic regressor on (DoM, residual_PDS) and on DoM alone; report ΔAUROC. ~30min CPU after H-138.
**Requires:** CPU; outputs of H-138 + cached `prefill_DoM_oof_predictions.npz`.
**Would change:** Confirm (ΔAUROC ≥ 0.02) → F-9's redundancy claim should be narrowed to "within hidden-state features", and PDS becomes a third orthogonal modality. Reject → DoM saturates the available signal across modalities, strengthening F-9.
**Blocks:** nothing.

### H-140: Asymmetric final-token collapse (F-4) is mediated by cross-chain content agreement
**Priority:** MEDIUM
**Motivated by:** 2402.10528 + F-4
**Test:** Per problem on MATH-500, compute (i) cross-chain PDS over K=5, (ii) final-token L19 PR. Test whether the correct-vs-incorrect difference in PR survives partialing out PDS. ~1h CPU after H-138.
**Requires:** CPU; PR_final from cached pathway 11 final-token NPZs; PDS from H-138.
**Would change:** Confirm (PR's marginal AUROC drops ≥0.04 with PDS partialed out) → F-4's asymmetric-collapse mechanism should be reframed as a downstream consequence of content agreement, not a primary geometric signal. Reject → F-4 is robust to a content-agreement confound; geometric mechanism stands.
**Blocks:** nothing.

### H-141: PDS recovers D-bucket membership without any geometric features
**Priority:** MEDIUM
**Motivated by:** 2402.10528 + F-7
**Test:** Compute PDS for the 36 D-bucket + 207 A-bucket problems (1024-tok labels); fit logistic on PDS only; report AUROC for bucket classification. ~30min CPU after H-138.
**Requires:** CPU; PDS column from H-138; D-bucket / A-bucket labels from F-7's working set.
**Would change:** Confirm (AUROC ≥0.7) → F-7's geometric framing is unnecessarily narrow; D-bucket has a textual fingerprint too. Reject → F-7's claim that geometry is the right tool stands.
**Blocks:** nothing.

---


### H-142: Conformal-wrapping prefill DoM yields valid 1−α correctness guarantees on MATH-500 with only 50 calibration examples
**Priority:** HIGH
**Motivated by:** 2402.10978 + F-2 + F-8
**Test:** Take pathway11_h100/prefill_gated_compute/results.json. Random-split 500 MATH-500 problems 1000 times into 50-cal / 450-test. For each split, compute conformal threshold q̂_α at α∈{0.1, 0.2, 0.3} over DoM scores. Measure empirical correctness on test set above q̂_α. Expected: empirical correctness within ±0.09 of (1−α), matching paper's reported calibration-set stdev. Time: ~20 min CPU.
**Requires:** CPU, cached prefill_gated_compute/results.json, no new generation.
**Would change:** Confirm → F-8 gets a probabilistic-guarantee twin (selective prediction with marginal coverage); we can publish "DoM-conformal" as a usable wrapper. Reject → conformal calibration fails on DoM scores at n=50 (probably score-distribution heavy-tailed), pushing us to bigger calibration sets or to score transformation.
**Blocks:** H-143 (would-be hybrid only makes sense if both sides calibrate).

### H-143: Frequency scoring (5-sample sub-claim agreement) matches or beats prefill L19 DoM on MATH-500 sub-claim correctness AUROC
**Priority:** HIGH
**Motivated by:** 2402.10978 + F-2 + F-9
**Test:** Regenerate K=5 alt-completions at T=1.0 for first 50 MATH-500 problems on Qwen-2.5-1.5B-Instruct. Run GPT-4 with paper's S prompt to extract sub-claims from each greedy answer. Run GPT-4 with paper's frequency-counter prompt (Appendix A) to score each sub-claim. Compute per-sub-claim AUROC against ground-truth step entailment (reuse paper's annotation method or align to MATH stepwise solutions). Compare against per-problem prefill DoM AUROC on the same 50 problems (already cached). Time: 3 min H100 + ~$2 GPT-4 + 30 min compute.
**Requires:** H100 (~3 min), GPT-4 API access (~$2), Qwen-2.5-1.5B-Instruct, MATH-500 problems 0–49.
**Would change:** Confirm → F-2 reframes ("DoM is strongest *supervised* predictor; frequency wins label-free"), opens path to label-free DoM-replacement. Reject → DoM dominates even at sub-claim grain, strengthens F-2 and weakens H-12. Either outcome is publishable.
**Blocks:** P11-FE175 (the hybrid only makes sense if frequency carries new information).

---


### H-144: L19 prefill DoM is causally linked to correctness, not merely correlated
**Priority:** HIGH
**Motivated by:** 2402.11917 + F-2
**Test:** Causal-scrubbing protocol per Brinkmann et al. §3: project out L19 prefill DoM on 100 originally-correct MATH-500 problems, re-run greedy decode, measure accuracy delta against (a) random-direction control, (b) cross-problem prefill swap. Loss recovery `(L_scrubbed − L_random)/(L_model − L_random)` ≥ 0.8 means causal; < 0.4 means correlational. ~30min H100 + cached activations.
**Requires:** H100 pod (lsuoka6bo8io7m), TransformerLens hooks on Qwen-2.5, cached prefill activations from P11.
**Would change:** If confirmed, F-2/F-8 upgrade from "correlational" to "causal" status and become defensible mechanistic claims. If rejected, F-2's AUROC is reading a confounder (most likely problem difficulty), and the entire prefill-DoM framing needs the disclaimer "covaries with correctness, mechanism unknown."
**Blocks:** H-1 (per-position DoM steering — pointless if DoM is just a difficulty proxy), H-19 (C_exact verification routing — same reason).

### H-145: The prefill / final-token orthogonality (F-3) is a path-merge residual, not independence
**Priority:** MEDIUM
**Motivated by:** 2402.11917 §5.3 + F-3
**Test:** Identify low-information positions in MATH-500 prompts (separators, instruction tokens). Train F-2's correctness probe at each. If any non-prefill, non-final position scores AUROC ≥ 0.65, it is acting as a register-token. Then test whether the L19 final-token DoM equals approximately `prefill_DoM − register_DoM` (a merge equation), measured by cosine of residual. Cosine ≥ 0.5 supports the merge interpretation. ~2h CPU on existing per-token activations.
**Requires:** Per-token cached activations (already exist for some positions, need extension to all prompt positions).
**Would change:** Reframes F-3 from "two orthogonal signals" to "merge-residual artifact." Suggests there are 3+ correctness-encoding positions, not 2.
**Blocks:** Nothing.

### H-146: Effective reasoning depth in Qwen-2.5-1.5B is bounded by ~L−1 deduction-head layers, capping AUROC on long-chain problems
**Priority:** MEDIUM
**Motivated by:** 2402.11917 §5.2 + F-2 + F-8
**Test:** Annotate minimum-chain-depth on 100 MATH-500 problems (rough — count required substitutions). Stratify F-2 AUROC and F-8 selective-prediction accuracy into depth bins. Brinkmann et al.'s mechanism predicts a sharp knee at the model's effective depth bound; F-2's "AUROC reflects knowability" reading predicts uniform AUROC across bins. ~4h annotation + 30min eval.
**Requires:** CPU only.
**Would change:** If knee found, F-2/F-8 headline numbers should be reported per-depth and "0.7731 overall" becomes misleading. Predicts Qwen-7B (more layers, deeper effective chain) should have higher AUROC on long-chain problems specifically — testable with existing 7B prefill data.
**Blocks:** Nothing directly, but reframes how H-1 and H-19 are evaluated.

### H-147: A specific Qwen-2.5-1.5B attention head implements a deduction-head motif on MATH-500 prefill
**Priority:** LOW
**Motivated by:** 2402.11917 §5.2 + H-13
**Test:** For each of Qwen-2.5-1.5B's ~28 layers × ~16 heads, check whether the QK circuit attends preferentially to the most-recent number / equation token, and whether the OV circuit increases the logit of the next computation step. Score each head with the deduction-head pattern-match score from §5.2. Heads scoring > 0.5 are candidate deduction heads. ~1 day analysis.
**Requires:** TransformerLens hooks on Qwen, MATH-500 prefill activations.
**Would change:** Identifying a deduction head in Qwen would be the first piece of mechanistic (not just correlational) evidence that the model performs structured reasoning rather than pattern-matching, and would give H-13 a concrete attribution target.
**Blocks:** Nothing.

---


### H-148: Mean-token-log-prob is a sufficient correctness ranker, matching prefill L19 DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2402.13212 + F-2
**Test:** Recover per-token P_LM for each Qwen-2.5-1.5B K=1 MATH-500 generation; score = mean / min / product over answer-span tokens; compute AUROC against the F-2 labels. ≤ 1h CPU if log-probs cached, ~1h H100 otherwise.
**Requires:** Qwen-2.5-1.5B; the 500 MATH-500 K=1 outputs already cached in `pathway11_h100/prefill_gated_compute/`; either stored token log-probs or one re-forward pass.
**Would change:** **Confirm** → F-2 reframes from "discovered geometric direction" to "redundant restatement of observable token likelihood"; F-8's selective-prediction story collapses to a cheap baseline. **Reject** → confirms prefill DoM carries information not present in output likelihoods, strengthening F-2.
**Blocks:** H-149, H-150, H-19 reframing.

### H-149: Adaptive sequence-likelihood stopping matches C_exact verification routing on the accuracy/compute Pareto
**Priority:** HIGH
**Motivated by:** 2402.13212 + H-19
**Test:** Simulate SOFT-SC adaptive procedure on cached K=8 MATH-500 generations: walk K in order, stop when cumulative min-log-prob ≥ τ. Sweep τ on dev split. Compare accuracy at average K against H-19's planned verification routing. 30 min CPU.
**Requires:** Cached K=8 generations + per-token log-probs.
**Would change:** **Confirm** → H-19 deprioritizes; the verification-pass step is unnecessary overhead. **Reject** → strengthens C_exact case; verification adds value beyond raw likelihood.
**Blocks:** H-19.

### H-150: F-3 orthogonality is downstream of shared output-likelihood signal
**Priority:** MEDIUM
**Motivated by:** 2402.13212 + F-3
**Test:** Compute Pearson + Spearman of (prefill DoM score, final DoM score) against (mean / min / product token log-prob) on 500 MATH-500 problems. If both DoM scores correlate >0.6 with the same aggregation, the geometric orthogonality is epiphenomenal to output likelihood. 30 min CPU.
**Requires:** Cached DoM scores (F-2, F-3) + token log-probs.
**Would change:** **Confirm** → H-13 (head-level attribution) and H-17 (RoPE de-rotation) lose motivation; F-3's mechanistic story weakens. **Reject** → confirms prefill and final-token directions encode information not reducible to output likelihood, strengthening F-3.
**Blocks:** H-13, H-17 (motivation only — they may still be testable).

### H-151: D-bucket cannot be flagged by sequence-likelihood scoring alone
**Priority:** MEDIUM
**Motivated by:** 2402.13212 + F-7
**Test:** Per-bucket distributions of mean / min / product token log-prob over the K=1 answer. If D-bucket K=1 answers have likelihoods statistically indistinguishable from A-bucket, but the F-7 collective geometric signature still separates them, the geometric finding is information beyond the logits. 30 min CPU.
**Requires:** Cached log-probs and bucket labels.
**Would change:** **Confirm** → F-7 is the unique signal for D-bucket; sequence-likelihood routing is insufficient and prefill geometry is essential. **Reject** → D-bucket is detectable from output likelihoods alone, simplifying the C_exact / verification story.
**Blocks:** F-7's positioning in the next writeup.

---


### H-152: Agentic clarification baseline matches or beats prefill DoM selective prediction at 50% coverage
**Priority:** HIGH
**Motivated by:** 2402.15610 + F-8
**Test:** On the bottom-50% prefill-DoM MATH-500 problems (where F-8 abstains), run one extra K=1 Qwen-2.5-1.5B generation conditioned on a self-asked rephrase prompt. Compare answered-set accuracy and effective coverage against F-8's 71.6% / 50%. ~1h H100 + 2h CPU.
**Requires:** Cached `pathway11_h100/prefill_gated_compute/` results + H100 for one K=1 pass on the rephrased subset; no new probes, no new labels.
**Would change:** If the clarification wrapper matches F-8's 71.6% on a comparable answered set without using DoM, F-8's framing as "the probe enables selective prediction" weakens to "the probe and a simple wrapper both enable it"; H-12's design must add a non-probe arm.
**Blocks:** Cleaner H-12 evaluation.

### H-153: Prefill L19 DoM score is paraphrase-sensitive within problem
**Priority:** MEDIUM
**Motivated by:** 2402.15610 + F-2
**Test:** Bottom-quartile prefill-DoM MATH-500 problems, 3 paraphrases each (Qwen-2.5-1.5B at temp 0.7), re-extract L19 prefill activations, recompute DoM. Report within-problem SD of DoM scores. ~1h H100 + 30min CPU.
**Requires:** Cached frozen DoM direction from F-2; H100 for prefill re-extraction on ≈ 375 paraphrased prompts.
**Would change:** If within-problem SD ≥ 0.5 × across-problem SD, F-2's reading of "prefill encodes intrinsic problem difficulty" must narrow to "prefill encodes problem-as-presented difficulty" — a meaningful shift in what the AUROC 0.7731 means.
**Blocks:** Cleaner F-2 interpretation; potential reframing of F-3 (orthogonality may be presentation-dependent).

---


### H-154: Local kNN intrinsic dimension at L19 prefill is a Gaussian-defying correctness signal where TwoNN failed
**Priority:** HIGH
**Motivated by:** 2402.18048 + F-10 + EXP-021
**Test:** Compute LID-MLE (Levina–Bickel 2004) and LID-GeoMLE (Gomtsyan
2019) at k=500 on the cached `pathway11_h100/` Qwen-2.5-1.5B L19
prefill-final-token activations across all 500 MATH-500 problems.
Predict K=1 correctness; report AUROC. Run TwoNN on the *same* NPZ as
the head-to-head bias control. ~30 min CPU.
**Requires:** CPU only; cached H100 NPZs (DATA_MANIFEST L19 prefill
final-token).
**Would change:** If LID-GeoMLE AUROC ≥ 0.65, F-10 narrows from
"residual geometry is null" to "global topology is null; local
neighbourhood dimension is informative" — and the TwoNN graveyard entry
becomes "TwoNN-specific failure" rather than "intrinsic-dimension
failure". If LID-GeoMLE ≤ 0.55, F-10 generalises to all dimensional
estimators we have tried, which is itself a strong claim.
**Blocks:** H-155, H-156 (downstream LID experiments depend on the basic
result).

### H-155: Last-token LID exceeds last-token DoM, refuting the "final-token information-poor" reading of F-3
**Priority:** HIGH
**Motivated by:** 2402.18048 + F-3 + F-2
**Test:** Compute LID-GeoMLE (k=500) on cached final-token L19
activations for MATH-500 K=1 generations. Predict K=1 correctness.
Compare AUROC against final-token DoM (F-3, 0.7186) and prefill DoM
(F-2, 0.7731). ~30 min CPU.
**Requires:** Cached final-token NPZ from pathway11_h100; CPU only.
**Would change:** If LID-final-token AUROC ≥ 0.74, the prefill/final
orthogonality result (cos = 0.046) is a *coordinate-frame* fact, not a
signal-content fact — the final token carries the signal but on an axis
our linear probe misses. F-3 has to be reframed accordingly. If
LID-final-token AUROC ≤ 0.72, F-3's information-asymmetry claim
strengthens.
**Blocks:** nothing.

### H-156: D-bucket prefill LID > A-bucket prefill LID (signal-mixing signature for F-7)
**Priority:** MEDIUM
**Motivated by:** 2402.18048 mixing experiment + F-7
**Test:** On cached pathway11_h100 prefill L19 activations, split
problems into A/B/C/D buckets per F-7. Compute LID-GeoMLE per problem;
compare bucket distributions (Mann-Whitney). Replicate Yin's mixing
control: take 30 D-bucket problems, prepend ground-truth solution
prefix, regenerate; compare prefill LID before/after.
**Requires:** CPU for the LID computation (~1 h); ~3 h H100 for the
mixing-control regeneration.
**Would change:** If D-bucket prefill LID is significantly elevated,
F-7's "distinctive collective signature" gains a generative mechanism
(distributional mixing) and connects to a published Yin-style
diagnostic. If A and D bucket LIDs overlap, F-7's signature is more
specific than a bulk dimensionality property.
**Blocks:** nothing.

---


### H-157: Multi-layer prefill DoM mosaic exceeds single-layer L19 DoM at correctness prediction
**Priority:** HIGH
**Motivated by:** 2403.05767 + F-9 + F-2
**Test:** Train logistic-regression probe on concatenated prefill DoM projections at {L11, L13, L15, L17, L19, L21, L23} for Qwen-2.5-1.5B on 1024-tok-labeled MATH-500, OOF 5-fold. Compare against F-2's L19-only AUROC=0.7731 and F-9's CoE-60 AUROC=0.811. Est: 20min CPU on cached NPZs.
**Requires:** Cached prefill activations at the 7 listed layers (re-extract if missing — 30min H100). No new model runs.
**Would change:** If multi-layer ≥ L19 + 1.5 pts, F-9's redundancy claim is wrong (CoE-60's lift was real-signal that L19 misses); F-2 is reframed as "best single-layer probe" rather than "the privileged layer." If multi-layer = L19 within noise, F-9 stands and L19 is genuinely sufficient.
**Blocks:** any further single-layer-DoM-only experiments in P11.

### H-158: DoM steering carries non-trivial alignment tax on non-MATH text
**Priority:** CRITICAL (precondition for honest H-1 reporting)
**Motivated by:** 2403.05767 §3.1 Fig 3b + H-1
**Test:** For each injection coefficient c ∈ {0, 0.5, 1, 2, 5, 10} of L19 DoM steering on Qwen-2.5-1.5B, jointly measure: (a) MATH-500 accuracy, (b) PPL on a 500k-token Pile text slab, (c) PPL on GSM8K dev, (d) fraction of off-grammar / non-numeric MATH-500 outputs, (e) entropy of predicted final-answer distribution. Plot accuracy gain vs each tax metric. Est: 4h H100.
**Requires:** H100 pod with intervention hooks; the Pile slab + GSM8K dev splits.
**Would change:** If accuracy gain at any c comes with >10% PPL increase or >5% off-grammar outputs, H-1's headline is a mode-collapse / capability-suppression artifact rather than a real correctness shift. If tax is genuinely <3% relative, H-1 is reportable.
**Blocks:** H-1 (H-1 cannot publish a number without H-158 controls attached).

### H-159: Prefill/final-DoM near-orthogonality is generic, not correctness-specific
**Priority:** MEDIUM
**Motivated by:** 2403.05767 §3.2 simultaneous-steering near-additivity + F-3
**Test:** At Qwen-2.5-1.5B L19, build 20 arbitrary contrast directions on prefill activations from random 250/250 splits of MATH-500 (random index splits, length splits, problem-id splits). Compute pairwise cosines among the 20. Report median, IQR, and rank of cos(prefill_DoM, final_DoM)=0.046 within this null. Est: 20min CPU.
**Requires:** Cached Stage 2 prefill NPZ for 500 MATH-500 problems at L19.
**Would change:** If 0.046 is at or below the median of the null, F-3 is reframed from "surprising orthogonality" to "expected null behavior of two unrelated mass-mean directions" — F-3 stops being load-bearing for the prefill/final-distinct-mechanism narrative. If 0.046 is below the 5th percentile of the null, F-3 *is* a genuine specific finding.
**Blocks:** any narrative claim that F-3 is evidence for distinct prefill / final mechanisms.

---


### H-160: Non-linear probing reveals an earlier-layer "best layer" than F-2's LR-derived L19
**Priority:** HIGH
**Motivated by:** 2403.18680 + F-2
**Test:** Fit a 1-hidden-layer MLP probe (5-fold OOF, identical splits to F-2) on cached pathway11 prefill activations at every layer 0..27 of Qwen-2.5-1.5B. Plot per-layer MLP-AUROC vs LR-AUROC. ~30min CPU.
**Requires:** CPU-only, cached prefill NPZs.
**Would change:** If any layer < L19 reaches MLP-AUROC ≥ 0.78 with LR-AUROC < 0.7 there, F-2 is reframed: "L19 is the LR-best layer; the MLP-best layer is L_k" — and our headline 0.7731 becomes a probe-class ceiling, not a model fact. If MLP at every layer is within 0.01 of LR, NL-ITI's non-linearity claim doesn't transfer to math reasoning, and F-2 is robust.
**Blocks:** P11-FE199, H-1's choice of intervention layer.

### H-161: The truthful direction is quasi-stationary across the trailing answer-token window
**Priority:** HIGH
**Motivated by:** 2403.18680 + F-3
**Test:** From cached pathway11 per-token activations, compute L19 DoM at answer-token positions [-1, -2, ..., -8]. Build the 8×8 pairwise-cosine matrix. ~20min CPU.
**Requires:** CPU-only, cached per-token activations.
**Would change:** If off-diagonals cluster >0.6, F-3 reframes as prefill-vs-answer-subspace (not prefill-vs-final-token); the orthogonality is *between* the prefill and the entire answer manifold, and our two-vector picture undercounts the answer side. If off-diagonals are <0.1, the answer-side direction rotates per-token and NL-ITI's averaging-helps result is paradoxical for our regime; F-3 stands.
**Blocks:** any future answer-side steering experiment.

### H-162: Head-level prefill-DoM steering moves MATH-500 accuracy where residual-stream-level steering doesn't
**Priority:** HIGH (subsumes part of H-1)
**Motivated by:** 2403.18680 + H-1
**Test:** Two-arm sweep, matched α-on-KL budget: (a) residual-stream prefill-DoM add at L19; (b) Top-K=48 head-level prefill-DoM add via NL-ITI methodology. Measure MATH-500 K=1 accuracy delta. ~4h H100 + 2h CPU.
**Requires:** H100 pod, per-head cached activations (or re-extract); P11-FE199 must complete first.
**Would change:** If (b) moves MATH-500 by ≥3pp while (a) moves <1pp, H-1's likely null is a granularity artifact and head-level steering becomes the canonical intervention. If both arms move similarly, granularity isn't the bottleneck — direction itself is the issue and H-1's paradigm is correct.
**Blocks:** publication scope of H-1 / F-8 selective-prediction extension.

---


### H-163: Head-wise L19 probes outperform layer-aggregate L19 DoM
**Priority:** HIGH
**Motivated by:** 2404.02252 + F-2
**Test:** Fit per-head logistic-regression probes on Qwen-2.5-1.5B L19
attention head outputs (12 heads) using cached prefill activations and
the same OOF 5-fold CV used for F-2 (AUROC 0.7731, 1024-tok). Report
best single-head AUROC and the top-3 head AUROC ensemble. ~1h CPU.
**Requires:** CPU, cached `pathway11_h100/...prefill_activations.npz`
(may need per-head dis-aggregation if currently stored as residual stream).
**Would change:** If best-head > 0.85, F-2's "layer 19 is the locus"
becomes "a small subset of L19 heads is the locus", and selective
prediction (F-8) at 50% coverage may improve from 71.6% by switching to
head-wise routing.
**Blocks:** H-13.

### H-164: Self-monitored gating beats unconditional per-position steering for MATH-500
**Priority:** HIGH
**Motivated by:** 2404.02252 + H-1 + 2306.03341
**Test:** P10-FE10 — extend the P10-FE1 three-direction sweep with a
SMITIN-style gate `w(t+s) = w(t)·(1−Δ)` if probe confidence C̄(t) < τ,
with τ = med(A) − std(A) over top-K=16 head-probes. Sweep
(gated, ungated) × (probe-weight, mass-mean, trained-bias) on MATH-500.
1 day H100.
**Requires:** H100 pod (currently stopped — `lsuoka6bo8io7m`), cached
prefill probes (H-163 must run first or run in parallel), Qwen-2.5-1.5B.
**Would change:** If gated > ungated by >2 points MATH-500 accuracy and
ungated ≈ baseline, H-1's framing shifts from "direction bank suffices"
to "direction + monitor required". If ungated also moves accuracy, the
SMITIN-style monitor is redundant for text-domain CoT (audio coherence
constraint does not port).
**Blocks:** H-1 reformulation.

### H-165: Prefill L19 DoM projection sign-flips during CoT, correlated with correctness
**Priority:** MEDIUM
**Motivated by:** 2404.02252 + F-3
**Test:** P11-FE204 — compute per-token ⟨θ_L19_prefill, residual_t⟩ on
cached MATH-500 generations; count sign-flip events per problem;
correlate flip count with correctness. 2h CPU.
**Requires:** CPU, cached Stage 2 NPZs.
**Would change:** High flip rate on incorrect trajectories supports a
"third regime" (in-flight direction rotation) between prefill and final,
which would explain F-9's CoE-60 redundancy as the in-flight monitor
signal smuggled in via aggregation. Low flip rate confirms F-3's
stationary-direction-orthogonality story.
**Blocks:** nothing.

---


### H-166: Head-wise L19 probes outperform layer-aggregate L19 DoM
**Priority:** HIGH
**Motivated by:** 2404.02252 + F-2
**Test:** Fit per-head logistic-regression probes on Qwen-2.5-1.5B L19
attention head outputs (12 heads) using cached prefill activations and
the same OOF 5-fold CV used for F-2 (AUROC 0.7731, 1024-tok). Report
best single-head AUROC and the top-3 head AUROC ensemble. ~1h CPU.
**Requires:** CPU, cached `pathway11_h100/...prefill_activations.npz`
(may need per-head dis-aggregation if currently stored as residual stream).
**Would change:** If best-head > 0.85, F-2's "layer 19 is the locus"
becomes "a small subset of L19 heads is the locus", and selective
prediction (F-8) at 50% coverage may improve from 71.6% by switching to
head-wise routing.
**Blocks:** H-13.

### H-167: Self-monitored gating beats unconditional per-position steering for MATH-500
**Priority:** HIGH
**Motivated by:** 2404.02252 + H-1 + 2306.03341
**Test:** P10-FE11 — extend the P10-FE1 three-direction sweep with a
SMITIN-style gate `w(t+s) = w(t)·(1−Δ)` if probe confidence C̄(t) < τ,
with τ = med(A) − std(A) over top-K=16 head-probes. Sweep
(gated, ungated) × (probe-weight, mass-mean, trained-bias) on MATH-500.
1 day H100.
**Requires:** H100 pod (currently stopped — `lsuoka6bo8io7m`), cached
prefill probes (H-166 must run first or run in parallel), Qwen-2.5-1.5B.
**Would change:** If gated > ungated by >2 points MATH-500 accuracy and
ungated ≈ baseline, H-1's framing shifts from "direction bank suffices"
to "direction + monitor required". If ungated also moves accuracy, the
SMITIN-style monitor is redundant for text-domain CoT (audio coherence
constraint does not port).
**Blocks:** H-1 reformulation.

### H-168: Prefill L19 DoM projection sign-flips during CoT, correlated with correctness
**Priority:** MEDIUM
**Motivated by:** 2404.02252 + F-3
**Test:** P11-FE207 — compute per-token ⟨θ_L19_prefill, residual_t⟩ on
cached MATH-500 generations; count sign-flip events per problem;
correlate flip count with correctness. 2h CPU.
**Requires:** CPU, cached Stage 2 NPZs.
**Would change:** High flip rate on incorrect trajectories supports a
"third regime" (in-flight direction rotation) between prefill and final,
which would explain F-9's CoE-60 redundancy as the in-flight monitor
signal smuggled in via aggregation. Low flip rate confirms F-3's
stationary-direction-orthogonality story.
**Blocks:** nothing.

---


### H-169: F-2's L19 specificity is a Qwen-2.5 quirk; the universal correctness-probing
locus is the middle-of-depth band (≈0.4–0.6 of layers).

**Priority:** HIGH
**Motivated by:** 2404.05971 (universal middle-layer finding across Mamba/RWKV/BTLM/Pythia) + F-2
**Test:** Per-layer DiffInMeans probe AUROC sweep on cached P11 H100 prefill NPZs
(P11-FE208). Plot AUROC vs layer for Qwen-2.5-1.5B; check whether argmax falls in
L13–L15 (middle 0.5 of 28 layers). 30min CPU.
**Requires:** CPU-only; cached P11 H100 NPZs.
**Would change:** On confirm: rewrite F-2 as "middle-layer DoM" not "L19 DoM"; revisit
H-1 per-position steering layer choice; recheck H-13 head-attribution layer scope.
On reject: F-2 stands; L19 has Qwen-specific significance worth investigating mechanistically.
**Blocks:** Definitive layer-choice for any future per-position steering bank (H-1).

### H-170: A Mahalanobis anomaly detector on concatenated per-layer probe log-odds
identifies D-bucket at AUROC > 0.8, matching the literature ceiling for ELK-style
distribution-shift detection.

**Priority:** HIGH
**Motivated by:** 2404.05971 (Mamba 0.82 / BTLM 0.85 anomaly AUROC) + F-7 + H-7
**Test:** Implement the Paulo construction (concat per-layer probe log-odds across all
28 layers, fit Gaussian on ABC-bucket activations, compute Mahalanobis distance for
D-bucket) on cached P11 H100 NPZs. Compute AUROC for D-bucket detection.
Depends on H-169 / P11-FE208 having computed per-layer probes. ~1h CPU.
**Requires:** CPU-only; cached P11 H100 NPZs; per-layer probes from H-169.
**Would change:** On confirm: F-7's "distinctive geometric signature" claim is
quantitatively supported at literature-grade level; H-7 (density-based outlier
detection) is validated. On reject (AUROC < 0.7): F-7 must be softened — geometric
signature is real but weaker than ELK distribution-shift detection.
**Blocks:** H-7.

### H-171: Tuned Lens per-layer perplexity on Qwen-2.5-1.5B is monotonically decreasing,
implying multi-layer ensembles must strictly dominate single-layer probes (refuting F-9).

**Priority:** MEDIUM
**Motivated by:** 2404.05971 (monotonic tuned-lens perplexity across Mamba/RWKV/Pythia)
+ F-9
**Test:** Train tuned-lens translators on Qwen-2.5-1.5B using
github.com/AlignmentResearch/tuned-lens against Pile validation (or MATH-500 prompts).
Plot per-layer perplexity. Then compute per-layer DoM AUROC on translated activations
and check whether stacked L13–L24 strictly improves over L19-alone via bootstrap CI.
~4h H100 + 30min CPU.
**Requires:** H100 for translator training; cached P11 NPZs.
**Would change:** On confirm: F-9 demoted to "redundancy at our sample size"; CoE-60
rehabilitated as a depth-aware feature. On reject: F-9 stands as a genuine redundancy
finding.
**Blocks:** any future re-validation of CoE-60 vs single-layer comparisons.

---


### H-172: Per-layer probe AUROC peaks at a layer earlier than L19 on Qwen-1.5B
**Priority:** HIGH
**Motivated by:** 2404.12038 + F-2
**Test:** Train sigmoid(w·e+b) probes at every layer L0..L27 on cached Qwen-1.5B prefill activations, report 5-fold OOF AUROC per layer. Confirm if any layer L < 19 gives AUROC ≥ 0.80; refute if L19 is at or near the global peak.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/` per-layer activations (or quick re-extract from H100 pod if not cached at every layer).
**Would change:** Confirm → demote L19 anchor in F-2; promote earlier layer; rerun all selective-prediction (F-8) results at the new layer. Refute → strengthens F-2's layer specificity claim and bounds the linear-probe ceiling.
**Blocks:** P11-FE213 (need this to know which layers have signal), P10-FE12 (probe weights re-derived at chosen layer)

### H-173: A multi-layer prefill probe ensemble exceeds single-L19 AUROC by ≥0.03
**Priority:** MEDIUM
**Motivated by:** 2404.12038 + F-9
**Test:** Stack per-layer probes (from H-172) into a logit-mean ensemble and a trained meta-classifier; report OOF AUROC. If ensemble − L19 ≥ 0.03 with bootstrap CI not crossing zero, F-9's redundancy claim is incomplete.
**Requires:** CPU; outputs of H-172.
**Would change:** Confirm → F-9 gets a "redundant *given L19*, not in general" caveat; opens a CoE-style feature-aggregation pathway that is not redundant. Refute → strengthens F-9.
**Blocks:** nothing

### H-174: Late-layer DoM is the negation of prefill DoM, not an orthogonal concept
**Priority:** MEDIUM
**Motivated by:** 2404.12038 + F-3
**Test:** P10-FE13 — patch final-token L19 with projection along (−DoM_prefill) at fixed magnitude vs random-direction ablation of same magnitude on a 50-problem held-out subset.
**Requires:** H100 pod for intervened generation; ~4h.
**Would change:** Confirm → F-3 narrative shifts from "different concepts" to "active cancellation"; mechanistic interpretability of prefill becomes more tractable. Refute → F-3 retains its current interpretation.
**Blocks:** nothing

---


### H-175: L19 prefill DoM is causally implicated in correctness, not merely correlated
**Priority:** CRITICAL
**Motivated by:** 2404.15255 + F-2 + EXP-Pathway11
**Test:** P11-FE214 — noising sweep replacing L19 prefill activations of correctly-answered MATH-500 problems with activations from incorrectly-answered problems; measure accuracy drop on the originally-correct subset.
**Requires:** 1 H100-day (cached activations + decoder pass), Qwen-2.5-1.5B, no new data
**Would change:** Confirm = F-2 upgrades from correlational to causal, validates H-1 steering premise. Reject = F-2 is a problem-difficulty-correlated probe artifact, H-1 should be deprioritized, headline narrative shifts from "the model uses L19 to know it's right" to "L19 contains a difficulty signal that we can read out."
**Blocks:** H-1, H-13, P11-FE215

### H-176: F-9's CoE-60 ↔ L19 redundancy is OR-gate, not mechanism-identity
**Priority:** MEDIUM
**Motivated by:** 2404.15255 + F-9 + 2307.15771 (Hydra effect)
**Test:** P11-FE217 — noise L19 prefill, re-fit CoE-60 probe on the noised activations, compare AUROC to clean CoE-60 baseline.
**Requires:** 4h H100 + 1h CPU, cached activations
**Would change:** Confirm (CoE-60 AUROC preserved under L19 noise) = F-9 is wrong about redundancy; CoE-60 is a parallel backup pathway and should be retained as a separate signal. Reject (CoE-60 AUROC collapses with L19 noise) = F-9 upgraded from correlational redundancy to causal redundancy.
**Blocks:** nothing

---


### H-177: Per-problem mutual k-NN alignment matches supervised L19 DoM as a label-free correctness probe on MATH-500
**Priority:** HIGH
**Motivated by:** 2405.07987 + F-2 + EXP-001..EXP-042 (any prefill DoM AUROC reference)
**Test:** Compute Qwen-1.5B ↔ Qwen-7B mutual k-NN(k=10) at L19 prefill, last-prompt-token, on cached MATH-500 NPZs. Use the per-problem alignment score (mean overlap of its k-NN set across the two models) as a 1-D classifier; report ROC AUROC vs the F-2 baseline of 0.7731. ~30 min CPU.
**Requires:** CPU, both 1.5B and 7B L19 prefill caches from `pathway11_h100/prefill_gated_compute`. No new extraction.
**Would change:** If AUROC ≥ 0.74 (within noise of 0.7731), F-2 reframes as a particular projection of the broader platonic kernel; the supervised DoM is then an efficient single-direction summary of an unsupervised geometric signal. If AUROC < 0.6, the F-2 signal is *not* in the kernel-alignment subspace and F-3's orthogonality framing strengthens into a directional claim about what is and isn't shared across scales.
**Blocks:** P11-FE218 must run before H-178 makes sense.

### H-178: F-6 incorrect-group concentration corresponds to *lower* cross-scale alignment in the 7B-incorrect bin
**Priority:** HIGH
**Motivated by:** 2405.07987 §2.2 + F-6
**Test:** Within-bin mutual k-NN alignment, 1.5B vs 7B at L19 prefill, on the 7B-correct (366) and 7B-incorrect (134) MATH-500 splits. Report bin-conditional alignment + bootstrap CI. ~1h CPU.
**Requires:** CPU, cached NPZs.
**Would change:** Confirm → F-6 reframes as "incorrect items concentrate into a 7B-specific subspace that is not aligned with 1.5B's neighborhood structure," giving a kernel-level interpretation of the PR inversion. Reject (alignment higher in incorrect bin) → F-6's concentration is alignment-preserving, and the PR inversion is purely a within-scale geometric tightening.
**Blocks:** P11-FE220.

### H-179: L19 is *not* the cross-scale-alignment-optimal layer for Qwen 1.5B↔7B
**Priority:** MEDIUM
**Motivated by:** 2405.07987 BrainScore-max protocol + F-2
**Test:** Layer-pair scan (28×28) of mutual k-NN alignment on cached MATH-500 prefill residuals if all layers cached; otherwise restrict to (5, 10, 15, 19, 24). Report argmax (i,j) and value vs (19, 19).
**Requires:** CPU, full per-layer cache (may need partial re-extract — check DATA_MANIFEST.md before promoting).
**Would change:** If argmax ≠ (19, 19), F-2's layer choice is correctness-driven not alignment-driven, which is a *positive* refinement (the supervised signal is not a confound of generic representational similarity). If argmax = (19, 19), L19 is a privileged layer for both signals.
**Blocks:** nothing.

---


### H-180: Per-problem mutual k-NN alignment matches supervised L19 DoM as a label-free correctness probe on MATH-500
**Priority:** HIGH
**Motivated by:** 2405.07987 + F-2 + EXP-001..EXP-042 (any prefill DoM AUROC reference)
**Test:** Compute Qwen-1.5B ↔ Qwen-7B mutual k-NN(k=10) at L19 prefill, last-prompt-token, on cached MATH-500 NPZs. Use the per-problem alignment score (mean overlap of its k-NN set across the two models) as a 1-D classifier; report ROC AUROC vs the F-2 baseline of 0.7731. ~30 min CPU.
**Requires:** CPU, both 1.5B and 7B L19 prefill caches from `pathway11_h100/prefill_gated_compute`. No new extraction.
**Would change:** If AUROC ≥ 0.74 (within noise of 0.7731), F-2 reframes as a particular projection of the broader platonic kernel; the supervised DoM is then an efficient single-direction summary of an unsupervised geometric signal. If AUROC < 0.6, the F-2 signal is *not* in the kernel-alignment subspace and F-3's orthogonality framing strengthens into a directional claim about what is and isn't shared across scales.
**Blocks:** P11-FE222 must run before H-181 makes sense.

### H-181: F-6 incorrect-group concentration corresponds to *lower* cross-scale alignment in the 7B-incorrect bin
**Priority:** HIGH
**Motivated by:** 2405.07987 §2.2 + F-6
**Test:** Within-bin mutual k-NN alignment, 1.5B vs 7B at L19 prefill, on the 7B-correct (366) and 7B-incorrect (134) MATH-500 splits. Report bin-conditional alignment + bootstrap CI. ~1h CPU.
**Requires:** CPU, cached NPZs.
**Would change:** Confirm → F-6 reframes as "incorrect items concentrate into a 7B-specific subspace that is not aligned with 1.5B's neighborhood structure," giving a kernel-level interpretation of the PR inversion. Reject (alignment higher in incorrect bin) → F-6's concentration is alignment-preserving, and the PR inversion is purely a within-scale geometric tightening.
**Blocks:** P11-FE224.

### H-182: L19 is *not* the cross-scale-alignment-optimal layer for Qwen 1.5B↔7B
**Priority:** MEDIUM
**Motivated by:** 2405.07987 BrainScore-max protocol + F-2
**Test:** Layer-pair scan (28×28) of mutual k-NN alignment on cached MATH-500 prefill residuals if all layers cached; otherwise restrict to (5, 10, 15, 19, 24). Report argmax (i,j) and value vs (19, 19).
**Requires:** CPU, full per-layer cache (may need partial re-extract — check DATA_MANIFEST.md before promoting).
**Would change:** If argmax ≠ (19, 19), F-2's layer choice is correctness-driven not alignment-driven, which is a *positive* refinement (the supervised signal is not a confound of generic representational similarity). If argmax = (19, 19), L19 is a privileged layer for both signals.
**Blocks:** nothing.

---


### H-183: F-2's single-direction reading is a linear summary of a sparse correctness population, not one circuit
**Priority:** HIGH
**Motivated by:** 2405.10928 + F-2 + F-3
**Test:** Train a TopK SAE on cached L19 prefill residuals (P11-FE227). Single-feature ablation sweep; report AUROC degradation curve. Confirm if AUROC robust to single-feature ablation but smooth multi-feature decay; refute if any single feature drops AUROC ≫ 0.05.
**Requires:** 1 day H100 for SAE training, cached `pathway11_h100/...l19_prefill.npz`, ~2h CPU for sweep.
**Would change:** On confirm, F-2 reframes from "single L19 prefill direction" to "linear projection over a sparse correctness code"; AUROC unchanged but the mechanistic narrative weakens. On reject, F-2's single-direction reading survives a stronger test than any we currently have.
**Blocks:** H-184 (would build on the SAE).

### H-184: F-3 orthogonality (cos=0.046) is a basis artefact, not evidence of distinct prefill/final circuits
**Priority:** HIGH
**Motivated by:** 2405.10928 + F-3
**Test:** Decompose prefill_DoM and final_DoM in the SAE basis from H-183 (P10-FE14). Report Jaccard / SAE-cosine of top-K active features for each direction.
**Requires:** SAE from H-183, ~1h CPU.
**Would change:** On confirm (high SAE-feature overlap), F-3 narrative changes from "two circuits" to "two linear views of one circuit." On reject (low SAE overlap), F-3's distinct-computation reading survives.
**Blocks:** nothing.

### H-185: Jacobian-importance (Λ) basis identifies a subspace where prefill DoM is *not* a top-importance direction
**Priority:** MEDIUM
**Motivated by:** 2405.10928 + F-2 + F-9
**Test:** L18→L19 Jacobian Gram matrix on cached residuals (stochastic-sources sketch). Project prefill_DoM into V; report Λ-rank distribution of squared dot products (P11-FE226).
**Requires:** Cached L18 + L19 prefill NPZs, ~6h CPU.
**Would change:** On confirm (DoM lives on low-Λ directions), F-2's predictive AUROC needs reinterpretation as a correlate rather than a computational driver; F-9 redundancy claim needs a basis caveat. On reject (DoM concentrates on high-Λ directions), F-2's mechanistic reading is strengthened.
**Blocks:** nothing.

---


### H-186: F-2's L19-uniqueness lives in the non-linear residual of the 0.99-Procrustes inter-layer fit
**Priority:** HIGH
**Motivated by:** 2405.12250 + F-2
**Test:** Fit per-layer linear A via Razzhigaev's generalized Procrustes; compute per-layer prefill DoM AUROC; check whether AUROC peak at L19 coincides with a non-linearity peak (low linearity_score) or is statistically indistinguishable from a plateau across L17-L21.
**Requires:** CPU only, ~30 min, all data cached in `pathway11_h100/`.
**Would change:** If L19 is a non-linearity peak, F-2 gains a mechanistic anchor (the prefill correctness signal lives precisely in the layers that *do* non-linear work). If L19 is on a plateau, F-2 must be re-stated as "any of L17-L21 works equally well; layer choice was arbitrary," which weakens the paper-readiness of the headline AUROC = 0.7731 number.
**Blocks:** P11-FE228, partially P11-FE230.

### H-187: Dimensional breathing (F-1) survives residual-stream subtraction
**Priority:** HIGH
**Motivated by:** 2405.12250 + F-1 + F-5
**Test:** Re-compute participation-ratio / breathing curve on Qwen-2.5-1.5B + 7B P11 NPZs after subtracting residual stream component (the paper's Figure 1 right-panel recipe). Check (a) whether breathing magnitude collapses to MP null and (b) whether content-dependence (F-5) survives.
**Requires:** CPU only, ~1 h, cached NPZs. Folds into P11-FE16's MP-correction work.
**Would change:** If breathing survives residual subtraction, F-1 is a real layer-wise reorganization signal independent of slow drift — robust to the paper's "block contributions are tiny" framing. If breathing collapses, F-1 must be re-cast as "residual stream drift on a low-dim subspace," which weakens the universality claim and forces F-5 (content-dependence) to be re-tested on the residual-removed signal.
**Blocks:** nothing; complements P11-FE16.

### H-188: D-bucket prefill activations are the long-tail of layer-linearization L2 error
**Priority:** MEDIUM
**Motivated by:** 2405.12250 Figure 9 + F-7
**Test:** Fit per-layer linear A on prefill residual streams; compute per-problem L2 residual ||xA − y|| at L18→L19 and L19→L20; compare D-bucket (n=36) vs A-bucket (n=207) distributions with Mann-Whitney U.
**Requires:** CPU only, ~30 min, cached NPZs.
**Would change:** If D-bucket is significantly higher (p < 0.05), F-7's collective signature unifies with the paper's "feature triggering regime" — D-bucket pathology is the rare-but-load-bearing non-linear event the paper hypothesizes. If indistinguishable, the bridge fails and F-7's mechanism is something else (attention concentration per H-21, or sampling-time stochasticity).
**Blocks:** nothing.

### H-189: Inter-layer maps in Qwen-2.5-1.5B are dominated by RoPE rotation, not learned non-linearity
**Priority:** MEDIUM
**Motivated by:** 2405.12250 + H-17
**Test:** Fit best linear A_k between each consecutive layer pair on prefill residual streams; compute cos(A_k, R(θ_k)) where R(θ_k) is the local RoPE rotation. Paper's 0.99 Procrustes plus H-17's "DoM rotation is RoPE-mechanical" jointly predict cos ≥ 0.9 across middle layers.
**Requires:** CPU only, ~1 h, cached NPZs (RoPE matrices come from model config).
**Would change:** If cos ≥ 0.9, the paper's "transformer is secretly linear" reduces to "transformer is mostly RoPE on a slow residual," a much stronger mechanistic claim with implications for F-2, F-3, and H-17. If cos is much lower, the inter-layer linear map is *learned* non-RoPE structure and H-17 is contradicted while paper's claim still stands.
**Blocks:** P11-FE232.

---


### H-190: Prefill and final-token DoM directions are W_vo-buffer reads of complementary reasoning intermediates
**Priority:** HIGH
**Motivated by:** 2405.15302 + F-3 + F-2
**Test:** P11-FE234 (pairwise W_vo cosine) + P11-FE235 (project DoM onto W_vo per
layer). Predicts prefill_DoM and final_DoM align with **different** W_vo matrices,
matching the paper's buffer-orthogonality prediction. ~40min CPU, weights-only.
**Requires:** Qwen-2.5-1.5B head weights (cached); prefill_DoM and final_DoM
directions (cached in `pathway11_h100/prefill_gated_compute/`).
**Would change:** if confirmed, F-3's interpretation flips from "two
uncorrelated probes" to "two W_vo-buffer reads of one reasoning circuit",
strengthening F-2 + F-3 as a *coupled* signal rather than two independent ones.
If refuted (both DoMs align with same W_vo or with no W_vo), the buffer-mechanism
framing of large-LLM correctness signals fails and we keep the current reading.
**Blocks:** H-191 (KS-peak layer test depends on the buffer reading being valid)

### H-191: KS(Ker^(l)) peak coincides with the prefill-DoM AUROC peak (L19) on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2405.15302 + F-2
**Test:** P11-FE233 — compute KS per layer from cached head weights only.
Predicts: KS peaks at l=19 if L19 is the structurally-correct buffer-extraction
layer. ~1h CPU, no GPU.
**Requires:** Qwen-2.5-1.5B head weights (cached, gitignored — see DATA_MANIFEST).
**Would change:** confirms gives F-2 a label-free structural anchor — we picked
the right layer for non-trivial mechanistic reasons. Refutes → re-extract DoM at
the KS-peak layer; if AUROC > 0.7731, F-2 updates to a different L_*.
**Blocks:** nothing

### H-192: Matching Score MS(h^(L19)) is itself a label-free correctness signal on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2405.15302 + F-2
**Test:** P11-FE237 — compute MS on cached prefill activations per problem;
test whether mean MS is higher on correct than incorrect K=1 generations. ~1h
CPU on cached NPZ.
**Requires:** cached Qwen-2.5-1.5B Stage 2 prefill activations
(`pathway11_h100/stage2/`).
**Would change:** if MS separates correct from incorrect at AUROC > 0.65 without
training, we have a *zero-shot, weights-only* alternative to supervised DoM.
Refute → MS is layer-selection only.
**Blocks:** nothing

### H-193: F-9's CoE-60 / L19-DoM redundancy holds on long-CoT generations but breaks on short greedy generations
**Priority:** MEDIUM
**Motivated by:** 2405.15302 + F-9
**Test:** P11-FE236 — partition MATH-500 K=1 generations by token length;
re-compute CoE-60 AUROC vs L19-DoM AUROC per partition. ~2h CPU on cached Stage 2.
**Requires:** cached Stage 2 NPZs with per-token features.
**Would change:** if redundancy reverses with length, F-9 is reframed as a
mixing-of-regimes artefact and CoE-60 becomes the right signal in VTS-only
contexts (short greedy, K=1). Buffer mechanism predicts this directly.
**Blocks:** nothing

---


### H-194: L19 prefill DoM is a generic abstract-syntax probe, not a correctness-specific probe
**Priority:** HIGH
**Motivated by:** 2405.15471 + F-2
**Test:** Train Qwen-2.5-1.5B L19 DoM on Conneau (2018) Bigram-Shift labels and on MATH-500 K=1 correctness labels separately; compare cosines and AUROC-overlap. P11-FE239.
**Requires:** ~5h CPU + 1h H100 to extract Conneau-task L19 activations.
**Would change:** If `cos(syntax_DoM, correctness_DoM) > 0.5`, F-2 needs the qualifier "via abstract-syntax abstraction" and the selective-prediction argument (F-8) inherits the qualifier. If cosines are near zero, F-2 is genuinely correctness-specific.
**Blocks:** nothing.

### H-195: Per-problem GRIDE intrinsic dimension at the depth-ID-peak layer is a label-free correctness predictor competitive with supervised L19 DoM
**Priority:** HIGH
**Motivated by:** 2405.15471 + F-2 + F-10
**Test:** Compute GRIDE-ID per problem at the layer Cheng et al. would predict as the peak (TBD by P11-FE238); compute AUROC vs MATH-500 K=1 correctness; compare to F-2's 0.7731 AUROC.
**Requires:** 1h CPU on cached prefill NPZs (after P11-FE238 fixes peak layer).
**Would change:** If GRIDE-ID AUROC ≥ 0.7, an unsupervised geometric scalar competes with our supervised DoM, and the "what does the DoM measure" question becomes "what is the per-problem ID measuring." If GRIDE-ID AUROC < 0.55, ID is informative at the *population* level (across models/corpora) but not at the *per-problem* level, distinguishing Cheng's contribution from F-2's.
**Blocks:** H-9 (Cheng's surprisal-ID correlation is the population analog of this hypothesis).

### H-196: F-10's PH-at-null does not extend to GRIDE intrinsic dimension on the same residual streams
**Priority:** MEDIUM
**Motivated by:** 2405.15471 + F-10
**Test:** Run GRIDE on the residual-stream point clouds where EXP-026 returned PH-at-null. If GRIDE gives non-null ID, F-10 must be narrowed to "PH-on-our-clouds is at the null" not "geometry on our clouds is at the null."
**Requires:** 1h CPU on cached EXP-026 NPZs.
**Would change:** If GRIDE separates correct/incorrect, F-10 narrows; if both PH and GRIDE return null on the same clouds, F-10 broadens.
**Blocks:** nothing.

### H-197: ID-peak depth-onset within the Qwen family correlates with model quality, mirroring Cheng's surprisal-vs-onset finding
**Priority:** LOW
**Motivated by:** 2405.15471
**Test:** Compute GRIDE-ID per layer for Qwen-2.5-1.5B and Qwen-2.5-7B on MATH-500 prefills. Predict: 7B ID-peak onset is at *earlier* relative depth than 1.5B (since 7B is the better LM at MATH-500: 73.2% vs 48.6% K=1). Tests whether Cheng's cross-model finding generalizes within a family.
**Requires:** 2h CPU on cached prefill NPZs for both models.
**Would change:** If 7B onset is earlier-relative-depth than 1.5B, Cheng's finding extends within-family. If the relation reverses or is null, the cross-model finding is not generalizable to within-family scale comparisons.
**Blocks:** nothing.

---


### H-198: F-4 asymmetric collapse is NC1 in disguise — correct/incorrect CDNV ratio matches observed effect size
**Priority:** HIGH
**Motivated by:** 2405.17767 + F-4 + EXP-related to pathway11_h100 collapse data
**Test:** Compute Wu & Papyan CDNV (eqn 4, k=2 power) on cached pathway11_h100
final-token L19 activations partitioned by ground-truth correctness (1.5B and
7B). Compare correct/incorrect CDNV ratio against the qualitative magnitude
that F-4 reports. ~20min CPU.
**Requires:** CPU only, cached `pathway11_h100/.../final_activations.npz` and
the `neural-collapse` PyPI package.
**Would change:** If CDNV ratio explains F-4, F-4 demotes from a
reasoning-specific finding to a special case of NC1; downstream claims about
"asymmetric collapse as a correctness signature" need rephrasing in NC1
language. If CDNV ratio does NOT explain F-4, the asymmetric-collapse claim
strengthens with a quantitative residual.
**Blocks:** publication framing of F-4

### H-199: F-3 prefill⊥final orthogonality is the structural NC3 violation Wu & Papyan document, not a reasoning-circuit finding
**Priority:** HIGH
**Motivated by:** 2405.17767 + F-3
**Test:** Compute cos(final_token_L19_DoM, W_unembed_row_k) for k = top-50
most-frequent MATH-500 answer tokens; same for prefill_DoM. Compare
distributions. If final_DoM concentrates near top-K W_unembed rows (cos > 0.3
median) and prefill_DoM doesn't, F-3 is the projection-vs-non-projection
structural effect. ~10min on H100 weights file.
**Requires:** CPU + W_unembed weights from Qwen-2.5-1.5B, existing DoM JSON.
**Would change:** Reframes F-3 as a corollary of the published NC3-violation
result rather than a novel orthogonality finding about reasoning circuits.
**Blocks:** any F-3-anchored claims about "two distinct correctness circuits."

### H-200: Hidden geometric structure in residual streams lives in the top-K class-mean subspace, invisible to PH on raw activations (refutes F-10 as basis artifact)
**Priority:** MEDIUM
**Motivated by:** 2405.17767 §4.2 + F-10
**Test:** Select top-K confusable class-mean directions via CDNV ranking
(K=32, 64, 128). Project cached residual streams (pathway11_h100, multiple
layers) onto this subspace and the orthogonal complement separately. Run
Vietoris-Rips persistent homology on each. Compare H_0/H_1 lifetimes against
Gaussian null sampled in matched dimension. ~4h CPU.
**Requires:** CPU + cached residual NPZs.
**Would change:** If projected-subspace PH shows non-null structure while
complement is null, F-10's "PH = Gaussian null" demotes to "PH on the
isotropic bulk = null" — leaves room for a positive PH finding on
NC-selected subspace.
**Blocks:** definitive PH-graveyard framing.

### H-201: Single-layer NCC classifier agreement (Wu&Papyan NC4) on prefill activations matches prefill DoM AUROC 0.7731
**Priority:** MEDIUM
**Motivated by:** 2405.17767 Table 1 (NC4 R²=0.49) + F-2 + F-9
**Test:** Build NCC classifier from prefill L19 means of correct vs incorrect
trajectories on training fold; evaluate agreement with logistic-DoM
classifier on held-out fold. If agreement > 90% and AUROCs within 0.01,
F-9's "single-layer DoM is sufficient" claim strengthens with a second
formulation. ~30min CPU.
**Requires:** CPU + cached pathway11_h100 prefill activations + 5-fold split.
**Would change:** Confirms F-9 from a second mathematical framing; provides
a published-method baseline (NC4) for any future selective-prediction claim.
**Blocks:** nothing.

---


### H-202: Cluster-size consensus confidence is correlated-but-not-redundant with prefill L19 DoM
**Priority:** HIGH
**Motivated by:** 2405.20974 + F-2 + F-9
**Test:** On cached pathway11_h100 K=8 samples for MATH-500, derive per-problem cluster-size-of-modal-answer / 8 as a SaySelf-style consensus confidence. Compute (a) Pearson and Spearman correlation with prefill L19 DoM score, (b) AUROC of consensus-confidence alone vs prefill DoM alone vs both jointly (logistic regression), (c) likelihood ratio test for the joint model. ~30 min CPU.
**Requires:** CPU only, cached K=8 generations + prefill DoM scores from `pathway11_h100/prefill_gated_compute/results.json`. No new data.
**Would change:** If correlation > 0.5 *and* joint LR test fails to reject univariate, the consensus-confidence signal is redundant with prefill DoM — strengthens F-2 and F-9 (single-layer DoM is the universal correctness signal). If correlation < 0.2, F-2's prefill signal measures something *other* than what SaySelf-style training extracts, weakening the universality framing and opening a refutation channel.
**Blocks:** H-203 (only worth pursuing the heavier verbalized-confidence test if cheap consensus-frequency proxy already shows independence).

### H-203: Stock Qwen verbalized confidence is materially weaker than prefill L19 DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2405.20974 + F-2
**Test:** Re-generate K=1 on MATH-500 with a confidence-elicitation suffix ("On a scale of 1-10, how confident are you in this answer?"). Parse integer confidence, compute ECE and AUROC against K=1 correctness for both Qwen-2.5-1.5B and 7B. Compare directly against F-2's prefill DoM AUROC 0.7731 (1.5B) and 0.7186 (final-token).
**Requires:** ~2h H100 for re-generation on both models, then CPU analysis.
**Would change:** If verbalized AUROC ≥ 0.77, F-2's "the probe is needed because output-space confidence is weak" assumption is falsified — output-space verbalized confidence on un-trained Qwen already matches the probe. If AUROC < 0.6, F-2 is robust against the SaySelf-style baseline and the prefill probe earns its keep on reasoning tasks even without RL.
**Blocks:** Decision on whether full SaySelf replication (FE76) is worth the H100 budget.

### H-204: Post-SaySelf-RL on Qwen, cos(prefill_DoM, final_DoM) rises from 0.046 toward shared-task-direction alignment
**Priority:** MEDIUM
**Motivated by:** 2405.20974 + F-3
**Test:** Train SaySelf (SFT on HotpotQA + PPO on cluster-size confidence with quadratic reward) on Qwen-2.5-1.5B. On MATH-500, re-extract per-problem prefill_L19_DoM and final_L19_DoM, recompute cosine. Predict cos > 0.2 post-training (substantially above the 0.046 baseline).
**Requires:** ~10h H100 + $50 GPT-4 API for rationale generation.
**Would change:** If cosine rises substantially, F-3's orthogonality claim is a property of un-RL'd stock Qwen, not of the architecture or task. If cosine stays at ~0.05, F-3 is structurally robust to output-space training and the orthogonality is geometrically deeper than reward-driven reorganization.
**Blocks:** nothing.

---


### H-205: BiPO-optimized L19 vector beats mean-of-means prefill DoM at correctness AUROC on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2406.00045 + F-2
**Test:** Train BiPO `v` at L19 on (q, correct CoT, incorrect CoT) triples from MATH-500 K=8 cached rollouts (β=0.1, lr=5e-4, AdamW, 20 epochs, batch 4 — paper's defaults). Compute 5-fold OOF AUROC for correctness prediction on the same fold split as F-2's 0.7731. ΔAUROC > +0.02 = confirm; |ΔAUROC| < 0.005 = mean-of-means is enough.
**Requires:** H100 pod (~30 min), Qwen-2.5-1.5B, cached prefill activations + K=8 CoT JSONs.
**Would change:** If confirmed, F-9 becomes "CoE-60 redundant with *optimized* L19 vector" rather than "redundant with single-layer DoM"; F-2 number gets a successor; H-1 gains a stronger candidate vector. If rejected, F-2 mechanism is mean-extraction-robust and the H-1 program can keep using DoM.
**Blocks:** H-206, P11-FE251, P11-FE252.

### H-206: BiPO L19 vector has high cosine with both prefill_DoM and final_DoM, contradicting F-3 orthogonality
**Priority:** HIGH
**Motivated by:** 2406.00045 + F-3
**Test:** Compute cos(`v_BiPO`, `prefill_DoM`) and cos(`v_BiPO`, `final_DoM`). If both > 0.5, F-3's "two orthogonal circuits" claim is an artifact of probing position; if one > 0.5 and other near zero, BiPO is implicitly choosing one of the two.
**Requires:** CPU, 5 min once `v_BiPO` exists from H-205.
**Would change:** Confirm = F-3 reframed as "mean-extracted prefill and final are orthogonal but a non-mean-extracted L19 direction can drive both, so prefill/final are projections of one underlying axis." Reject = F-3 strengthened (BiPO picks one circuit, the other is genuinely separate).
**Blocks:** nothing; informs H-13 head-level attribution work.

---


### H-207: Joint (prefill, final) probe AUROC exceeds prefill-only DoM AUROC by ≥0.01 on MATH-500
**Priority:** HIGH
**Motivated by:** 2406.04028 + F-2 + F-9
**Test:** Concatenate cached L19 prefill and final-token residuals for the 500 MATH-500 1.5B problems. Fit a 5-fold OOF logistic probe with l2. Predict: AUROC ≥ 0.7831 (vs F-2's 0.7731). If true, F-2 should be re-stated as "best single-state predictor" and F-9's CoE-60 redundancy claim is refuted (joint-state probes carry information that single-layer L19 does not). If false, single-state framing of F-2 is correct and CSAE-flavoured methods are unlikely to help further.
**Requires:** 30 min CPU; cached pathway11_h100 NPZs only; no GPU, no new data.
**Would change:** F-2 statement (single-state best vs overall best); F-9 redundancy claim; whether to invest in a full CSAE (P11-FE255).
**Blocks:** P11-FE255 should not run if H-207 fails decisively (joint probe ≤ prefill alone).

### H-208: Restricted to CSAE d-features, the prefill-final DoM cosine exceeds 0.3
**Priority:** HIGH
**Motivated by:** 2406.04028 + F-3
**Test:** Train a small CSAE on (h(prefill_L19); h(final_L19)) pairs from 500 MATH-500 problems with correct/incorrect labels driving the BCE auxiliary (eq. 9). Compute cos(DoM_d_prefill, DoM_d_final). Predict: cos ≥ 0.3 (vs raw cos=0.046 from F-3). If true, F-3's "orthogonal directions" framing is a partition artefact: the correctness-relevant subspace is shared, with orthogonality concentrated in c-features (context). If false, F-3 stands as a genuine subspace claim.
**Requires:** 4-8h H100 for CSAE training; cached prefill+final activations; standard SAE training pipeline (build it; ~200 LOC).
**Would change:** F-3 framing; H-1 (per-position steering) interpretation — if d-features align, the right intervention is at d-feature level, not residual level; H-13 head-level attribution gains a complementary feature-level decomposition.
**Blocks:** H-1 interpretation depends on whether H-208 confirms or refutes.

### H-209: D-bucket problems form a coherent cluster in agglomerative clustering of L19 prefill activations (ARI ≥ 0.3)
**Priority:** MEDIUM
**Motivated by:** 2406.04028 + F-7
**Test:** Cluster L19 prefill activations from all 500 MATH-500 1.5B problems using NMF → t-SNE → agglomerative (Ward) with k ∈ {10, 20, 40}. Compute ARI between cluster labels and D-bucket membership (36 D-bucket vs 207 A-bucket). Predict: ARI ≥ 0.3 at some k. If true, F-7's "distinctive collective signature" gains a quantitative anchor and the centroid becomes a candidate D-bucket direction. If ARI ≤ 0.1 across all k, F-7 weakens to "tail signature, not separable cluster."
**Requires:** 30 min CPU; cached prefill NPZs only.
**Would change:** F-7 framing; H-21 (D-bucket attention entropy) gains or loses a prerequisite — a coherent cluster is a much more useful target than a tail; informs the design of any D-bucket detector.
**Blocks:** nothing critical; but H-21 becomes more interpretable after H-209.

---


### H-210: A PaCE-1M-style sparse-code probe at L19 lifts correctness AUROC above 0.7731 on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2406.04331 + F-2
**Test:** Build a 1k–40k atom PaCE-1M dictionary at Qwen-1.5B L19, sparse-code the cached 500-problem prefill activations, fit 5-fold OOF logistic regression on correctness labels with the sparse codes as features. Compare AUROC against F-2's 0.7731. Threshold for confirmation: AUROC ≥ 0.79 with the same OOF protocol.
**Requires:** 4h H100 (concept extraction) + 1h CPU (sparse coding + LR); cached `pathway11_h100/prefill_gated_compute/` activations (already on disk); PaCE-1M stimuli (open).
**Would change:** Confirms → F-2 is a single-direction ceiling, not an info ceiling; suggests upgrading our DoM probe to a sparse-code probe and rewriting the headline number. Refutes → F-2 is robust to parametrization, strengthens the "single direction is the right object" framing.
**Blocks:** H-211 (the steering version is harder to interpret if the probe version doesn't lift)

### H-211: Multi-concept oblique-projection steering moves MATH-500 accuracy more than single-DoM steering at matched compute
**Priority:** MEDIUM
**Motivated by:** 2406.04331 + H-1
**Test:** On 100 MATH-500 problems, run three intervention arms at matched generation budget: (a) vanilla, (b) single-DoM L19 subtraction (H-1 design), (c) PaCE-style multi-atom oblique projection that zeros a 50-atom "tied-to-incorrect" partition while preserving "tied-to-correct" atoms. Threshold for confirmation: arm (c) > arm (b) by ≥5pp on accuracy with overlapping CIs only at one boundary.
**Requires:** 1 day H100; concept dictionary from H-210; LLM-judge partition over 1k atoms (~$10 GPT-4 cost or local model).
**Would change:** Confirms → H-1's single-direction design is wrong; the right object is a *partition* of concept atoms, not a single direction. Refutes → simpler is sufficient; multi-atom decomposition is overkill for steering.
**Blocks:** nothing

---


### H-212: Black-box LR-on-perturbations matches or exceeds L19 prefill DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2406.04370 + F-2 + F-8
**Test:** Replicate Pedapati et al. Algorithm 7 (six perturbations × 5 generations + semantic-set count + rouge similarity + NLI contradiction features + logistic regression) on Qwen-2.5-1.5B / MATH-500. Use majority-of-K=8 as binary label. Compare AUROC and AUARC against F-2's 0.7731 and F-8's selective-pred curve. Estimated ~14h H100 + 1h CPU.
**Requires:** H100 for ~28 generations × 500 problems; deberta-large-nli; Helsinki-NLP MT; NLTK; spaCy NER. Existing 1.5B greedy cache is reusable for SD-mode.
**Would change:** Confirm → F-2's "white-box prefill DoM is the predictive signal" framing weakens; selective-prediction H-19/F-8 should be revisited with black-box features as a baseline. Reject (LR << DoM on MATH-500) → strengthens F-2 by showing 1024-tok CoT is where internal geometry uniquely beats response-diversity baselines that work on short QA.
**Blocks:** P11-FE260

### H-213: SRC NLI-contradiction over the cached CoT alone predicts MATH-500 correctness
**Priority:** MEDIUM
**Motivated by:** 2406.04370 + F-8
**Test:** Apply deberta-large-nli to 5 random splits of each existing K=1 CoT in pathway11_h100/prefill_gated_compute/, take max contradiction probability, compute AUROC vs majority-of-K=8 label. ~20min CPU.
**Requires:** CPU only; cached generations; deberta-large-nli (~430MB).
**Would change:** AUROC ≥ 0.65 → cheap output-only feature folds into F-8 selective-prediction stack and shrinks the marginal value of internal access. AUROC < 0.55 → CoT-internal contradiction is QA-domain-specific and doesn't translate to MATH-500's symbolic/numeric reasoning.
**Blocks:** nothing

### H-214: Qwen-2.5-1.5B-trained L19 prefill DoM transfers zero-shot to Qwen-2.5-7B
**Priority:** MEDIUM
**Motivated by:** 2406.04370 + F-2
**Test:** Project Qwen-7B prefill activations onto the L19 DoM direction learned on Qwen-1.5B (handle width mismatch via shared problem-pair structure or learned aligner). Compute AUROC on Qwen-7B MATH-500 K=1 correctness labels and compare to 7B-native DoM AUROC. ~30-60min CPU.
**Requires:** CPU; cached pathway11_h100 activations for 1.5B and 7B.
**Would change:** Within ~5pp of native → F-2 reframes as dataset-driven direction encoded in residual stream; opens H-11 (1.5B-as-7B-verifier) variant. Drops >10pp → F-2's direction is model-specific and the cross-scale transfer claim Pedapati et al. make for *output features* doesn't carry over to *internal directions*.
**Blocks:** nothing

---


### H-215: L19 prefill correctness signal localizes to <10 MLP `W_V` columns whose vocabulary projections promote answer-formatting / numeric tokens
**Priority:** HIGH
**Motivated by:** 2406.11614 (Hong et al., Concept Vectors) + F-2 + F-7
**Test:** Run P11-FE263 (vocab-projection sweep over L19 `W_V` columns of
Qwen2.5-1.5B, GPT-4-scored for math-correctness coherence) followed by
P11-FE264 (Gaussian-noise causal ablation of top-10 columns). Predicts: a
small number (<10) of L19 columns have vocab projections clustered around
`\boxed`, digits, `=`, `\frac`, and ablating them drops prefill DoM AUROC
substantially while random-column ablation does not.
**Requires:** 1 day CPU (projections) + 5h H100 (ablations). Qwen2.5-1.5B
weights already cached.
**Would change:** If <10 columns account for >50% of the AUROC drop, F-2
gains a sparse parametric mechanism and H-13's head-level attribution unit
shifts to MLP-column. If no localization, F-2's "dense residual-stream
direction" reading is reinforced and concept-vector framework does not
transport to correctness signals (interesting null — Hong et al. might
not generalize beyond entity concepts).
**Blocks:** any mechanistic claim in P11 about "what the prefill direction
is doing."

### H-216: D-bucket fragility (F-7) is mediated by ablation-vulnerability of one or two L19 `W_V` columns
**Priority:** MEDIUM
**Motivated by:** 2406.11614 + F-7 (D-bucket peak PR 67/88 collective
signature)
**Test:** From the 36 D-bucket and 207 A-bucket problems
(`pathway11_h100/prefill_gated_compute/`), compute mean L19 MLP activation
`m^L19` per group, identify top-10 columns with largest differential. Apply
single-column Gaussian-noise ablation, re-run K=1 + K=8 generation on D and
A subsets, measure D-bucket → A-bucket transitions vs A-bucket → D-bucket
transitions.
**Requires:** ~3h H100 (10 columns × 18 min/column for D ∪ A subset = 243
problems), Qwen2.5-1.5B cached.
**Would change:** If one column's ablation moves a meaningful fraction
(>20%) of D-bucket problems to A-bucket while leaving A-bucket stable, F-7's
"collective" wording weakens to "parametrically localized." If no single
column does this, F-7's collective framing strengthens (no parametric
shortcut).
**Blocks:** nothing.

---


### H-217: SAE-refined L19 prefill correctness feature exceeds dense-DoM AUROC 0.7731
**Priority:** HIGH
**Motivated by:** 2406.11624 + F-2
**Test:** Train a JumpReLU-SAE (~16k features, k=64-128) on cached pathway11_h100 L19 prefill residuals. Identify the SAE feature(s) most aligned with the raw correctness DoM; compute OOF 5-fold AUROC for that feature (or small feature ensemble) on MATH-500 correctness. ~4-8h H100 + 1h CPU.
**Requires:** H100 for SAE training; cached pathway11_h100 NPZs (already on disk).
**Would change:** **Confirm** ⇒ 0.7731 is the dense-DoM ceiling, *not* the correctness-signal ceiling; F-2's numerical anchor demotes from "the prefill correctness signal" to "the dense-vector readout of a sparser underlying feature." **Reject** ⇒ dense L19 DoM is already near-optimal, SAE refinement does not add discriminative power, and the H-1 / P10-FE1 raw-DoM arms remain the right baseline.
**Blocks:** H-218, P10-FE16, P10-FE17

### H-218: F-3 prefill / final-token orthogonality is a dense-DoM artifact
**Priority:** HIGH
**Motivated by:** 2406.11624 + F-3
**Test:** Train SAEs on cached L19 prefill and final-token residuals (same architecture as H-217). For each, find the correctness-aligned sparse feature(s). Compute cos similarity between the prefill SAE feature(s) and the final-token SAE feature(s) in the SAE feature basis (and projected back to residual space). ~+1h CPU on top of H-217 training.
**Requires:** SAE artifacts from H-217.
**Would change:** **Confirm** (cos in SAE space ≥ 0.3) ⇒ F-3's "two distinct circuits" reading is wrong — there is one underlying correctness feature with two surface readouts, and the orthogonality at the dense level is confound mass cancelling. **Reject** ⇒ orthogonality survives sparsification, strengthening F-3 as a substantive geometric finding rather than a probe-extraction artifact.
**Blocks:** P10-FE17 fourth-arm interpretation

### H-219: L19 prefill DoM is partially confounded by surface-feature axes
**Priority:** HIGH
**Motivated by:** 2406.11624 + F-2 + F-7
**Test:** Linear probes for length-tier (3 buckets), MATH-500 topic cluster (5 clusters via Sentence-BERT on question text), and formula density (regex-counted) on cached L19 prefill residuals. OOF 5-fold AUROC per axis. 20min CPU. Cheap version of P11-FE266.
**Requires:** Cached pathway11_h100 L19 prefill NPZs; question-text embeddings.
**Would change:** **Confirm** (any axis AUROC ≥ 0.7) ⇒ the 0.7731 correctness AUROC partly reflects surface features the model knows about prefill; F-2's "correctness signal" framing is partly confounded; the P10-FE1 steering experiment must per-axis-orthogonalize the DoM before alpha sweep. **Reject** ⇒ correctness DoM is largely surface-feature-orthogonal and F-2's reading stands.
**Blocks:** none

---


### H-220: Position×layer-swept DoM beats hand-picked L19 prefill
**Priority:** HIGH
**Motivated by:** 2406.11717 + F-2
**Test:** Re-extract Qwen-2.5-1.5B-Instruct activations at every layer × every post-instruction-template token position on the MATH-500 train split. Compute DoM at each (l, i). Score each candidate by OOF correctness AUROC on held-out folds. Compare peak AUROC against current L19 prefill AUROC of 0.7731. ~1.5h H100 + CPU.
**Requires:** H100 for re-extraction (cached prefill NPZs are L19-only); CPU for sweep.
**Would change:** If peak AUROC > 0.7731 at (l, i) ≠ (L19, end-of-user-prompt), F-2 needs to be re-stated as "the canonical correctness direction lives at (l*, i*)" rather than "L19 is special." If peak AUROC ≈ 0.7731 at L19, F-2 hardens.
**Blocks:** H-13 (head-level attribution should be done at the canonical layer, not arbitrarily at L19).

### H-221: Prefill DoM is causally sufficient for MATH-500 correctness
**Priority:** CRITICAL
**Motivated by:** 2406.11717 + F-2 + F-8
**Test:** Two paired interventions on Qwen-2.5-1.5B-Instruct. (1) Directional ablation of unit-norm prefill_L19_DoM at every (layer, position) during prefill+generation; measure MATH-500 K=1 accuracy delta from baseline 48.6%. (2) Activation addition of prefill_L19_DoM at L19 across all positions on K=1 incorrect problems; measure flip rate to correct. Capability control: MMLU + GSM8K. ~4h H100.
**Requires:** H100 with intervention hooks; cached MATH-500 prompts.
**Would change:** If ablation drops MATH-500 by ≥10pp AND MMLU preserved within 1pp, F-2 is causally load-bearing for correctness specifically. If ablation no-ops, F-2 is read-out-only and the "encodes correctness" framing is wrong (correlation only). If MMLU also collapses, the direction is general-capability, not correctness-specific.
**Blocks:** H-1 (this is a stronger version of H-1's per-position steering test).

### H-222: Rank-1 weight orthogonalization w.r.t. prefill DoM removes correctness signal in one shot
**Priority:** HIGH
**Motivated by:** 2406.11717 + F-9
**Test:** Apply W' = W - r̂r̂ᵀW to all output-writing matrices of Qwen-2.5-1.5B-Instruct (embedding, attention-out per head per layer, MLP-out per layer, output biases) where r̂ is unit-norm prefill_L19_DoM. Eval MATH-500 K=1 + MMLU + GSM8K + ARC. ~30min H100.
**Requires:** H100; ability to save modified weights.
**Would change:** Cleanest possible one-shot causal test. Three diagnostic outcomes per the FE-73 description. Confirms or refutes the "all of CoE-60 + L19 DoM collapses to a single rank-1 weight subspace" reading of F-9.
**Blocks:** nothing.

### H-223: Top correctness-attributing attention heads cluster at mid-depth (L10-L16), not at L19
**Priority:** MEDIUM
**Motivated by:** 2406.11717 + F-2 + H-13
**Test:** For Qwen-2.5-1.5B-Instruct (28 layers), compute direct feature attribution (DFA) per (layer, head) by projecting each head's output at the prefill end-of-prompt position onto unit-norm prefill_L19_DoM. Rank heads by |attribution|; check spatial distribution of top-K. ~2h H100 + CPU.
**Requires:** H100 for per-head output extraction; CPU for projections.
**Would change:** If top-8 heads cluster at L10-L16 (40-58% depth, analogue of Arditi's L10-L14 of 24-layer model), F-2's L19 framing is post-hoc — the *production* layer is mid-depth and L19 is just where the direction has accumulated maximum projection. This reframes F-2 from "L19 encodes" to "mid-depth heads write, L19 reads."
**Blocks:** nothing.

---


### H-224: SE-supervised probes generalize across MATH-500 categories better than accuracy-supervised DoM
**Priority:** HIGH
**Motivated by:** 2406.15927 + F-2 + F-8
**Test:** Sample K=10 completions per MATH-500 problem at T=1 on Qwen-2.5-1.5B. NLI-cluster via DeBERTa-Large. Compute discrete SE per problem, binarize via best-split (Eq. 5). Train two LR probes at L19 prefill: (a) target = binarized SE, (b) target = K=1 correctness. Evaluate within-distribution AUROC and leave-one-category-out OOD AUROC. Estimated time: ~2h H100 + 1h CPU.
**Requires:** H100 for K=10 sampling on Qwen-1.5B (DeepSeek-R1 distill if cached), CPU for NLI + LR fitting. NPZs already cached for prefill L19; need K=10 completions only.
**Would change:** If SE-supervised probe matches in-distribution AUROC and improves OOD by ≥3 AUROC points, F-2 / F-8 should adopt SE supervision as the canonical recipe and our 0.7731 figure is an under-trained baseline. If the OOD gap is null on math (vs 7-10 on QA), the SEP advantage is QA-specific and our framing survives.
**Blocks:** nothing.

### H-225: TBG and SLT probe directions are aligned (cos > 0.5), refuting F-3 in the uncertainty regime
**Priority:** HIGH
**Motivated by:** 2406.15927 + F-3
**Test:** Train two LR probes on Qwen-2.5-1.5B / MATH-500 at L19: one on TBG (= last-input-token, our prefill), one on SLT (= last-generated-token, our final-token). Both supervised on the same target (binarized SE first; then accuracy as control). Compute cos(w_TBG, w_SLT). Estimated time: ~30min CPU.
**Requires:** cached prefill NPZs (have); SLT NPZs (need extraction). LR fits trivial.
**Would change:** If cos > 0.5 for SE-target and ≈ 0 for accuracy-target, F-3's 0.046 is a property of *correctness supervision* and the residual stream encodes a stable uncertainty axis across positions. Reframes F-3 from "geometry rotates" to "correctness is high-frequency, uncertainty is low-frequency." If both targets give cos ≈ 0, F-3 is robust and SEP behavior must be explained otherwise.
**Blocks:** H-1 (per-position DoM steering — if directions align, fixed-vector steering may work).

---


### H-226: SE-supervised probes generalize across MATH-500 categories better than accuracy-supervised DoM
**Priority:** HIGH
**Motivated by:** 2406.15927 + F-2 + F-8
**Test:** Sample K=10 completions per MATH-500 problem at T=1 on Qwen-2.5-1.5B. NLI-cluster via DeBERTa-Large. Compute discrete SE per problem, binarize via best-split (Eq. 5). Train two LR probes at L19 prefill: (a) target = binarized SE, (b) target = K=1 correctness. Evaluate within-distribution AUROC and leave-one-category-out OOD AUROC. Estimated time: ~2h H100 + 1h CPU.
**Requires:** H100 for K=10 sampling on Qwen-1.5B (DeepSeek-R1 distill if cached), CPU for NLI + LR fitting. NPZs already cached for prefill L19; need K=10 completions only.
**Would change:** If SE-supervised probe matches in-distribution AUROC and improves OOD by ≥3 AUROC points, F-2 / F-8 should adopt SE supervision as the canonical recipe and our 0.7731 figure is an under-trained baseline. If the OOD gap is null on math (vs 7-10 on QA), the SEP advantage is QA-specific and our framing survives.
**Blocks:** nothing.

### H-227: TBG and SLT probe directions are aligned (cos > 0.5), refuting F-3 in the uncertainty regime
**Priority:** HIGH
**Motivated by:** 2406.15927 + F-3
**Test:** Train two LR probes on Qwen-2.5-1.5B / MATH-500 at L19: one on TBG (= last-input-token, our prefill), one on SLT (= last-generated-token, our final-token). Both supervised on the same target (binarized SE first; then accuracy as control). Compute cos(w_TBG, w_SLT). Estimated time: ~30min CPU.
**Requires:** cached prefill NPZs (have); SLT NPZs (need extraction). LR fits trivial.
**Would change:** If cos > 0.5 for SE-target and ≈ 0 for accuracy-target, F-3's 0.046 is a property of *correctness supervision* and the residual stream encodes a stable uncertainty axis across positions. Reframes F-3 from "geometry rotates" to "correctness is high-frequency, uncertainty is low-frequency." If both targets give cos ≈ 0, F-3 is robust and SEP behavior must be explained otherwise.
**Blocks:** H-1 (per-position DoM steering — if directions align, fixed-vector steering may work).

---


### H-228: Prefill–final orthogonality is a saturation artefact, not a subspace artefact
**Priority:** HIGH
**Motivated by:** 2406.17563 + F-3
**Test:** Apply the project's L19 prefill DoM at α=2 during 1.5B greedy generation on 50 MATH-500 problems. At each generation step i, compute KL(f(y_<i) ‖ f_Δ(y_<i)) using top-p_top=0.5 truncation. Predict: KL drops monotonically and is ≤ 0.2× its initial value by token 10 across both correct and incorrect problems.
**Requires:** 2h H100; cached prefill prompts; existing P2 / P11 DoM vector at L19; standard logit-injection pipeline (no new infrastructure).
**Would change:** If KL collapses as predicted, F-3's "orthogonal directions" framing is replaced with "single-direction saturation" — the direction is the same throughout generation but the activation has advanced along it. This reshapes H-1 (per-position steering): the right interpretation is "early-token injection," not "different direction at each position." If KL stays high, F-3 stands and the orthogonality is genuine.
**Blocks:** H-1, H-17 (rotation hypothesis becomes harder to disentangle from saturation if H-228 confirms).

### H-229: Per-head L19 DoM AUROC exceeds residual-stream L19 DoM AUROC
**Priority:** HIGH
**Motivated by:** 2406.17563 + F-2
**Test:** Re-extract Stage 2 prefill activations as per-head attention outputs (H=14 or 28 depending on Qwen 2.5 1.5B head count, d_h≈128). Compute per-head DoM AUROC on 500 MATH-500 1024-tok labels. Predict: max_head AUROC ≥ 0.78 (vs residual 0.7731), and the top-3 heads cluster in the middle-to-late layers consistent with Scalena Fig 3.
**Requires:** 2h H100 to re-run prefill forward pass with per-head output caching; 30min CPU for per-head AUROC scoring; existing labels.
**Would change:** If max_head AUROC > residual AUROC by ≥ 0.01, F-2 should be re-stated as "the L19 prefill correctness direction is concentrated in 3-5 attention heads"; the 0.7731 headline number can be raised; H-13 (head-level attribution) gains a quantitative anchor; downstream selective-prediction (F-8, H-19) gains a sharper feature. If max_head ≤ residual, the residual-stream framing is correct and F-2 stays as-is.
**Blocks:** H-13 (becomes well-defined once per-head AUROCs are tabulated).

### H-230: KL-trace produces a label-free correctness signal competitive with semantic-entropy probes
**Priority:** MEDIUM
**Motivated by:** 2406.17563 + F-8 + H-12
**Test:** For each MATH-500 problem, compute KL(f(prefill) ‖ f_Δ(prefill)) using a *random* contrastive direction injected at L19, α=2, top-p_top=0.5. Rank problems by KL. Compare AUROC against (a) supervised prefill DoM, (b) Semantic Entropy Probes (2406.15927).
**Requires:** ~4h H100 for the prefill-only forward passes (single token per problem); existing infrastructure.
**Would change:** If KL-rank AUROC ≥ 0.70 (matching SEP) without any label supervision in direction discovery, H-12 framing of SEPs as the canonical label-free path is weakened — KL-trace becomes a competitor or complement. F-8 gains a label-free baseline arm.
**Blocks:** H-12 needs a comparator before being closed.

---


### H-231: L19 is one of several layers in a "Stage 3" prediction-ensembling block, not a uniquely necessary layer
**Priority:** HIGH
**Motivated by:** 2406.19384 + F-2
**Test:** Single-layer DoM AUROC sweep across L0-L27 on cached MATH-500 prefill set, plus adjacent-layer swap (P11-FE282 + P11-FE286). Confirm: AUROC plateau across L17-L23 ≥0.75; AUROC under L18↔L19 swap within ±0.02 of 0.7731. Reject: AUROC peaks sharply only at L19, drops ≥0.05 at L18 and L20.
**Requires:** Re-extract activations at all 28 layers (30min H100); CPU for 28 logistic regressions; 1h H100 for swap experiments
**Would change:** On confirm, F-2 is reframed as "Stage-3-block DoM" rather than "L19 DoM"; H-1 (per-position steering) targets multi-layer rather than single-layer steering. On reject, L19 specificity is robust and elevates F-2's mechanistic content.
**Blocks:** H-1 (steering experiments shouldn't proceed at single layer until block question resolved), H-13 (head-level attribution scope depends on whether to scope to L19 only or L17-L23)

### H-232: F-3's prefill/final-token orthogonality reduces to the prediction-neuron / suppression-neuron duality
**Priority:** HIGH
**Motivated by:** 2406.19384 + F-3
**Test:** Compute prediction-neuron and suppression-neuron sets per layer on Qwen 2.5-1.5B (Lad et al. + Gurnee 2401.12181 kurtosis/skew classifier on W_U·w_out). Project cached L19 prefill DoM and final-token DoM into Pred-set and Supp-set. Confirm: prefill DoM cos with Pred-set basis ≥0.4 at L19-L24, ≤0.1 with Supp-set; reverse for final-token DoM. Reject: low cos with both sets.
**Requires:** 5h CPU on saved weights (no forward passes)
**Would change:** On confirm, F-3's cos=0.046 is no longer a novel geometric finding but a re-statement of the universal-neurons taxonomy; the standalone "DoM rotates 90°" framing in the Pathway 10 narrative needs revision.
**Blocks:** Nothing directly, but bears on H-1 (steering target choice) and H-17 (RoPE-mechanical rotation question — if rotation is suppression-neuron-driven, it's not RoPE-mechanical).

### H-233: L19 ablation preserves K=1 MATH-500 accuracy at ≥45% (F-2 is correlational, not causal)
**Priority:** CRITICAL
**Motivated by:** 2406.19384 + F-2 + F-8
**Test:** Run K=1 greedy generation on 500 MATH-500 problems on Qwen 2.5-1.5B with (a) baseline (b) L19 zero-ablated (c) L18-L19-L20 zero-ablated. Compare top-1 accuracy and re-fit prefill DoM AUROC for each condition. Confirm (Lad et al. consistent): (b) accuracy 45-48% (within 3 pts of baseline 48.6%). Reject: (b) accuracy <40%.
**Requires:** 2h H100, no new data — 500-problem set already cached
**Would change:** On confirm, F-2 becomes "L19 is the readout, not the compute" — H-1 (per-position steering at L19) loses much of its motivation, since steering an epiphenomenal layer won't change behavior; selective-prediction (F-8) reading is unaffected. On reject, F-2 gains causal content and supports H-1.
**Blocks:** H-1 (steering experiments should wait for causality answer)

### H-234: Prefill-DoM AUROC peak (L19) lags WiC-probe peak (~L14) — signal is post-feature
**Priority:** MEDIUM
**Motivated by:** 2406.19384 + F-2 + F-9
**Test:** Train WiC-task linear probe at L0-L27 on Qwen 2.5-1.5B; locate peak. Overlay against prefill-DoM AUROC curve from H-231/FE70. Confirm: WiC peaks at L13-L15, DoM peaks at L19-L21 (≥4 layer separation). Reject: peaks within 2 layers of each other.
**Requires:** 2h CPU + WiC dataset
**Would change:** On confirm, F-9 ("CoE-60 redundant with single-layer DoM") narrows: the redundancy is within the prediction-ensembling stage but CoE-60 may also capture earlier feature-stage content that single L19 DoM misses; reframes F-2's "feature" interpretation.
**Blocks:** Nothing.

### H-235: Breathing PR-curve inflections co-locate with CKA block boundaries
**Priority:** MEDIUM
**Motivated by:** 2406.19384 + F-1
**Test:** Compute CKA heatmap across L0-L27 on Qwen 2.5-1.5B; identify block boundaries (e.g. via spectral clustering of CKA matrix). Overlay PR(layer) breathing curve. Confirm: breathing-curve inflection points within ±1 layer of CKA boundaries. Reject: PR curve smooth, CKA matrix block-structured (mismatch).
**Requires:** 1h CPU on cached activations
**Would change:** On confirm, F-1's "universal breathing" claim is partly subsumed by Lad et al.'s "universal stages" — re-statement rather than independent finding. On reject, breathing is genuinely orthogonal to representational-similarity block structure and remains independent.
**Blocks:** Nothing.

---


### H-236: Prefill-DoM AUROC is uniform across mid-layers in Qwen2.5-1.5B (F-2's L19 specificity is a layer-search artifact)
**Priority:** HIGH
**Motivated by:** 2409.02228 + F-2
**Test:** Train mass-mean DoM probes at every mid-layer L4–L27 on the
cached `pathway11_h100/prefill_gated_compute/` 1.5B prefill
activations. Plot OOF AUROC vs layer. Compare against L19's 0.7731.
30min CPU.
**Requires:** CPU, Qwen2.5-1.5B, cached NPZs (DATA_MANIFEST §
prefill activations).
**Would change:** If AUROC varies by ≤0.02 across L13–L25, F-2 is
downgraded from "L19 is special" to "any mid-layer DoM gives ~0.77,"
and our story becomes about *the protocol*, not *the layer*. If L19
stands out by ≥0.03 over its neighbors, F-2 strengthens with concrete
layer-specialization evidence — and Qwen2.5-1.5B becomes interesting
*relative to* Llama-2-7B / GPT-J / GPT-2.
**Blocks:** P11-FE289 (the fine-tune test should target whatever layer
this hypothesis identifies as best, not L19 by assumption).

### H-237: Randomized-label fine-tuning of Qwen2.5-1.5B does not materially change the L19 prefill DoM direction or AUROC (shallow forgetting on MATH)
**Priority:** HIGH
**Motivated by:** 2409.02228 + H-8 + H-10
**Test:** Fine-tune Qwen2.5-1.5B on MATH-500-train with labels
randomized per problem for ≥1k steps. Verify behavioral collapse
(MATH-500 accuracy drops <10%). Re-extract L19 prefill activations on
MATH-500 holdout. Report cos(DoM_pre, DoM_post), AUROC drop, and PR
profile delta. ~half H100 day.
**Requires:** H100, training pipeline (TRL / standard SFT loop),
MATH-500-train split.
**Would change:** If cos > 0.9 and AUROC drop < 0.03, H-8 is
*confirmed* (prefill direction stable through destructive fine-tune)
and H-10 is *rejected* (breathing structure does not change with
fine-tuning, at least not with this destructive variant). Adds a clean
"shallow forgetting" replication for our pipeline. If cos < 0.5 or
AUROC drop > 0.1, breathing *is* fine-tune-fragile and H-10 stands —
and the paper's finding does not generalize to math reasoning.
**Blocks:** any future "stability through fine-tuning" claims.

---


### H-238: PCA-PC1 of mean-centered prefill activations beats DoM as a correctness predictor on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2409.05907 + F-2 + F-4
**Test:** Re-extract the L19 prefill direction on cached P11 H100 activations using CAST's PCA-on-alternating-rows protocol. Recompute 5-fold OOF AUROC. Compare against F-2's 0.7731. Also report at all 28 layers under the same protocol. ~30min CPU.
**Requires:** CPU only; cached prefill activations from P11.
**Would change:** If PC1 AUROC ≥ 0.79 at L19, F-2's headline number must be re-stated as a lower bound and the operator changed in PROJECT_RECORD §1c. If PC1 ≈ DoM (within 0.005), F-4's asymmetric-collapse story does not actually induce class-covariance asymmetry strong enough to matter for the linear extractor — that is informative.
**Blocks:** H-239, H-240 (which assume the same operator).

### H-239: A non-L19 layer wins under a real (layer × θ × direction) grid search for the correctness gate
**Priority:** HIGH
**Motivated by:** 2409.05907 + F-2 + F-8
**Test:** Run CAST's full grid search over all 28 layers, ~50 threshold values, both > and < directions, F1-optimised on the correct/incorrect contrast. Report best (layer, θ, direction) and the AUROC achieved. ~1h CPU.
**Requires:** CPU only; per-layer cached activations.
**Would change:** F-2's "L19 is privileged" claim. If a layer in [L4, L12] wins, the prefill correctness story moves earlier in the network and forces re-interpretation of F-2 in terms of input-context features rather than mid-network reasoning state.
**Blocks:** nothing.

### H-240: CAST-style tanh-projection thresholder lifts F-8 selective prediction above 71.6% answered accuracy at 50% coverage
**Priority:** MEDIUM
**Motivated by:** 2409.05907 + F-8
**Test:** Drop CAST's `f(sim(h, tanh(proj_c h))) > θ` thresholder onto F-8's selective-prediction operating point. Sweep θ; report best (coverage, answered-acc, K-budget). ~20min CPU.
**Requires:** CPU only; cached prefill activations.
**Would change:** F-8's headline 71.6% number if CAST's thresholder lifts ≥ 1pp at matched coverage. Also gives F-8 a principled (F1-optimal) operating point selection rule rather than the current ad-hoc 50% coverage choice.
**Blocks:** nothing.

### H-241: F-7's D-bucket signature is per-prompt extractable via CAST's PCA + grid search (refutation test)
**Priority:** HIGH
**Motivated by:** 2409.05907 + F-7
**Test:** Apply CAST's full pipeline with D⁺ = 36 D-bucket prefill activations, D⁻ = 207 A-bucket prefill activations, across all 28 layers, both directions. Report best AUROC. ~1h CPU.
**Requires:** CPU only; cached prefill activations + bucket assignments.
**Would change:** If best AUROC > 0.75, F-7's "collective signature" framing is wrong and D-bucket has a per-prompt linear feature at a specific layer (which we report). If best AUROC < 0.6 across all layers, F-7 is corroborated against the strongest available supervised baseline — important provenance update.
**Blocks:** P11-FE21 (D-bucket attention entropy) — outcome of H-241 may obviate the attention-entropy mechanism test.

### H-242: CAST's open-source IBM toolkit successfully steers Qwen-2.5-1.5B refusal on Sorry-Bench, validating the harness for H-1
**Priority:** HIGH
**Motivated by:** 2409.05907 + H-1 (currently blocked on harness validity)
**Test:** Install `github.com/IBM/activation-steering`, port to Qwen-2.5-1.5B, replicate their Qwen-1.5-Chat-1.8B harmful-refusal numbers (>85% harmful refusal, <5% harmless refusal) using their published (cond_layer=8, θ>0.031, behavior layers 10-20, α=4) hyperparameters as a starting point. ~4h H100 + 2h CPU.
**Requires:** H100 pod for inference; Sorry-Bench (450 harmful prompts) + 500 Alpaca harmless prompts.
**Would change:** Provides a validated steering harness on our base model so that any subsequent failure of H-1 (per-position DoM steering) cannot be attributed to a broken harness. Also gives empirical (layer, α) hyperparameters as the search starting point for H-1.
**Blocks:** H-1, H-12, P11-FE3 (activation patching needs same harness).

---


### H-243: DoM-projection probes recover ground-truth circuit axes on TracrBench compiled transformers
**Priority:** MEDIUM
**Motivated by:** 2409.13714 + F-2 + F-3
**Test:** Download TracrBench (Thurnherr & Scheurer 2024), select 20 binary-output programs, fit mass-mean DoM at the residual layer Tracr compiles the decision into, compute cos(DoM, ground-truth feature axis from the compilation map). Predicts: if F-2's interpretation of DoM as "computationally meaningful direction" is correct, median cos > 0.5; if DoM is a training-dynamics correlate that incidentally tracks correctness on Qwen, median cos ≈ 0.
**Requires:** CPU only (~1 day), Tracr library, public TracrBench release.
**Would change:** Confirm → F-2 gets its first ground-truth validation; H-13 head-level attribution becomes substantially more credible. Reject → F-2's headline AUROC stays as a phenomenological correlation but the "mechanistically meaningful direction" framing must be dropped from FINDINGS.md and HYPOTHESES.md. Also gates whether H-13 is worth the GPU spend.
**Blocks:** H-13.

---


### H-244: Counterfactual prompt-perturbation matches supervised L19 DoM on MATH-500 selective prediction
**Priority:** HIGH
**Motivated by:** 2409.16146 + F-8 + H-12
**Test:** Implement CF-decomposition and CF-premise prompts on Qwen-2.5-1.5B / MATH-500, score answer-change for each problem, build risk-coverage curve, compare against F-8's prefill-DoM curve at every coverage from 0.3 to 0.7. Use cached `pathway11_h100/prefill_gated_compute/results.json` as the "initial answer"; only the two CF rounds need fresh inference. Est. 2h H100 + 1h CPU.
**Requires:** H100 pod (cheap — 1.5B), no new training data, cached P11 generations.
**Would change:** Confirm → F-8 hardens against the strongest behavioural baseline; H-12 priority drops; we have a label-free fallback that ships without geometry. Reject (CF dominates DoM by >5pp at coverage 0.5) → F-8's "supervised hidden-state probe is necessary" justification breaks; pivot toward CF-prompting variants.
**Blocks:** H-12, partially F-8.

### H-245: Prefill L19 DoM AUROC is dominated by topic-familiarity base rates rather than per-problem decomposability
**Priority:** HIGH
**Motivated by:** 2409.16146 + F-2 + H-5
**Test:** Stratify MATH-500 by topic (7 categories from MATH benchmark labels) and compute within-topic prefill L19 DoM AUROC. If overall AUROC stays at 0.7731 but mean within-topic AUROC drops below 0.6, the signal is base-rate familiarity, not decomposability. Free check on cached pathway11 NPZs. Est. 20min CPU.
**Requires:** No new compute; cached prefill activations already include topic labels in MATH-500 metadata.
**Would change:** Confirm → H-5 framing rewritten ("prefill is a topic-familiarity prior, not a decomposability signal"); F-2 reframed; selective-prediction reports must include topic-stratified curves; F-8's "71.6% at coverage 0.5" headline becomes suspect because answered subset is topic-imbalanced. Reject → H-5 hardens.
**Blocks:** H-5, partially F-2.

---


### H-246: LEAT probe beats prefill DoM on MATH-500
**Priority:** HIGH
**Motivated by:** 2410.02707 + F-2
**Test:** Extract Qwen-1.5B L19 hidden state at the last token before `\boxed{...}` close per MATH-500 problem; train OOF 5-fold logistic on cached P11 H100 activations; compare AUROC to F-2's 0.7731. Estimated 30min CPU.
**Requires:** CPU; cached Stage 2 NPZs from `pathway11_h100/prefill_gated_compute/`.
**Would change:** If LEAT AUROC ≥ 0.85, F-2's headline claim ("prefill > final-token") must be reframed as "LEAT > prefill > final-token" — and the L19 DoM rebranded as a token-localized rather than prefill-localized signal. Cascading update to F-9 (CoE-60 redundancy comparison) and H-12 (label-free probe motivation).
**Blocks:** H-247, H-248.

### H-247: Qwen-1.5B prefill DoM does not generalize from non-MATH training tasks to MATH-500
**Priority:** MEDIUM
**Motivated by:** 2410.02707 + F-2
**Test:** Generate Qwen-1.5B answers on TriviaQA, Winogrande, MNLI subsets (~1k each); train L19 prefill DoM probes per dataset; evaluate on MATH-500; subtract logit-min-exact baseline. Estimated 3h H100 + 1h CPU.
**Requires:** H100 pod, cached MATH-500 activations.
**Would change:** If transfer AUROC ≤ logit baseline, F-2's 0.7731 is MATH-specific (matches Orgad et al.'s skill-specific finding). Forces H-1 (per-position steering) to be reformulated as topic-conditional. Validates Section 4 multifaceted-truthfulness within Qwen-1.5B.
**Blocks:** H-1 reformulation, H-3 (code generation transfer).

### H-248: Probe-as-selector across K=8 resamples beats F-8's gated-compute selective prediction
**Priority:** HIGH
**Motivated by:** 2410.02707 + F-7 + F-8
**Test:** Generate K=8 MATH-500 samples per problem on Qwen-1.5B; score each with F-2's prefill DoM probe; pick argmax probe-score; measure accuracy; compare to (a) majority vote (b) F-8's K=2.5 selective. Estimated 6h H100 + 30min CPU.
**Requires:** H100 pod for K=8 generation.
**Would change:** If probe-selector accuracy is 5+ points above majority vote and 10+ above F-8 on the matched-coverage operating point, F-8 must be reframed as "gating ≪ selecting" and the selective-prediction operating regime shifts. Cascades to H-19 (C_exact verification routing) — selector might be cheaper than verification.
**Blocks:** H-19 redesign.

### H-249: D-bucket (F-7) is a mixture of Orgad-types {C2, D, E1, E2}
**Priority:** MEDIUM
**Motivated by:** 2410.02707 + F-7
**Test:** Generate K=30 MATH-500 samples on Qwen-1.5B; classify each problem into A/B1/B2/C1/C2/D/E1/E2 per Section 5; intersect with current K=8 D-bucket membership; compute mixture proportions and per-type internal-representation distances at L19. Estimated 1 day H100 + 2h CPU.
**Requires:** H100 pod for K=30, cached prefill NPZs.
**Would change:** If D-bucket is dominated by one type (e.g., C2), F-7's "distinctive collective signature" sharpens to that type. If it's a mixture, F-7 must be subdivided. Either way, H-7 (density-based outlier detection) gets a finer target.
**Blocks:** H-7 refinement.

### H-250: Token-position × layer AUROC heatmap shows prefill ≠ peak on Qwen-1.5B
**Priority:** HIGH
**Motivated by:** 2410.02707 + F-2 + F-3
**Test:** On cached Qwen-1.5B + MATH-500 P11 activations, sweep probe across (last 50 generated tokens) × (L17–L21). Build AUROC heatmap; identify the (token, layer) argmax. Estimated 2h CPU.
**Requires:** CPU; cached NPZs.
**Would change:** If the heatmap argmax is *not* at prefill, F-2 must be relocalized. If the argmax is at LEAT, that aligns with Orgad et al. Figure 2 and lifts H-246 confidence. If argmax is at a third position (e.g., post-`\boxed` close), it suggests a Qwen-specific decoding signature.
**Blocks:** F-2 relocalization PR.

---


### H-251: A Damani-style MLP/LoRA probe on prefill hidden state lifts F-2's correctness-prediction AUROC by ≥0.02 on Qwen-2.5-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2410.04707 + F-2
**Test:** Train (a) 2-layer MLP on L19 prefill activations targeting Bernoulli λ_i (K=8 soft targets); (b) LoRA on Qwen-2.5-1.5B with same target. Compare AUROC for K=1 correctness against F-2's linear logistic probe (0.7731 OOF 5-fold). Cached prefill activations + K=8 cache already exist. Est: 30min CPU (MLP), 4h H100 (LoRA).
**Requires:** Cached `pathway11_h100/prefill_gated_compute/` L19 activations + K=8 outcomes; H100 only for LoRA variant.
**Would change:** On confirm — F-2's probe-architecture choice is suboptimal; project should use MLP/LoRA probes by default and 0.7731 is a floor. On reject — linear logistic at L19 is near-ceiling for the prefill signal; gains seen in Damani's Math come from soft-target training or LoRA fine-tuning of the base LM, not from probe capacity.
**Blocks:** nothing

### H-252: Offline Ada-BoK on prefill DoM beats F-8 selective prediction at matched compute on MATH-500
**Priority:** HIGH
**Motivated by:** 2410.04707 + F-8
**Test:** Implement Damani's greedy matroid allocator + offline binning on cached K=8 MATH-500 generations using F-2 prefill DoM as λ̂. Compare expected accuracy at average K=2.5 to F-8's 71.6% answered-accuracy at coverage 0.5. Est: 1h CPU.
**Requires:** Cached MATH-500 K=8 outputs + prefill DoM scores. No new GPU.
**Would change:** On confirm — F-8's selective-prediction framing should be retired in favor of Ada-BoK; the right primitive is continuous K-allocation. On reject — abstain-vs-answer is competitive with continuous allocation under the project's signal quality, and the Damani gains come from richer probe families (covered by H-251).
**Blocks:** H-19 (FE19's verification routing must clear the Ada-BoK bar before justification)

### H-253: F-3 prefill ⊥ final-token orthogonality is target-specific and disappears under a marginal-reward target
**Priority:** MEDIUM
**Motivated by:** 2410.04707 + F-3
**Test:** Train two probes on cached L19: target=K=1 correctness (replicates F-3) and target=empirical λ_i over K=8 (Damani-style). Compute cos(probe_K1, probe_λ) and cos(probe_λ, final-token DoM). If the λ-target probe has cos > 0.5 with final-token DoM, F-3's geometric claim is target-dependent. Est: 1h CPU.
**Requires:** Cached prefill + final-token L19 activations; K=8 outcomes.
**Would change:** On confirm — F-3 needs reframing: orthogonality is a property of the K=1 functional, not of residual-stream geometry. On reject — F-3 is geometrically robust; both probe targets recover the same orthogonal structure.
**Blocks:** nothing

---


### H-254: Prefill L19 DoM has direction-unstable / norm-stable structure across re-fits
**Priority:** HIGH
**Motivated by:** 2410.04962 + F-2
**Test:** Re-fit prefill DoM 5× with different fold seeds on cached pathway11_h100 residuals. Compute pairwise cosine matrix of the 5 directions and Kendall ρ on per-token DoM-projection magnitudes across runs. ~30 min CPU.
**Requires:** CPU only; pathway11_h100/stage2/*.npz cached.
**Would change:** If pairwise cos < 0.5 and ρ_norm > 0.9, F-2 must be reframed as "L19 prefill activation magnitude, not direction, predicts correctness" — and the project's geometric narrative (DoM = direction) shifts to magnitude-based. Confirms a deeper alignment with Stoehr's "norm stable, direction not" finding.
**Blocks:** H-1 reformulation (would split into magnitude-only H-1a vs vector H-1b).

### H-255: Last-token gate-keeping is reasoning-specific (last-5 prefill tokens recover full-prefill AUROC)
**Priority:** MEDIUM
**Motivated by:** 2410.04962 + F-3
**Test:** Subset cached prefill activations to final 5 prompt tokens only; refit prefill DoM; report MATH-500 OOF AUROC and compare against 0.7731. ~20 min CPU.
**Requires:** CPU only; cached residuals.
**Would change:** If last-5 ≈ 0.77 AUROC, F-3 weakens to "prefill last-token vs generation last-token are orthogonal" — much smaller claim. If last-5 << 0.77 (e.g., < 0.65), F-3 is strengthened: prefill signal is genuinely distributed across the prompt and not concentrated at the final position, contra Stoehr's last-token gate-keeping for factual recall.
**Blocks:** nothing.

---


### H-256: The L19 prefill DoM is recoverable from layer-19 SwiGLU weights alone (no MATH-500 activations needed)
**Priority:** HIGH
**Motivated by:** 2410.08417 + F-2 + F-3
**Test:** Bilinear approximation of L19 SwiGLU on Qwen-2.5-1.5B (drop SiLU
nonlinearity), construct interaction tensor B from gate_proj W and up_proj V,
contract with u_correct (residual-space pullback of the prefill DoM probe
itself), symmetrize, eigendecompose, report cos(top eigenvector, prefill_DoM).
~30 min CPU once weights are in memory.
**Requires:** CPU, Qwen-2.5-1.5B weights (already on H100 pod), cached prefill
DoM probe vector from `pathway11_h100/prefill_gated_compute/`.
**Would change:** If cos > 0.5, F-2 is reframed as "L19 weights pre-determine
the correctness direction; activation averaging just retrieves it" —
strengthens cross-checkpoint stability claim H-8 and weakens novelty of the DoM
discovery itself. If cos < 0.2, bilinear approximation of SwiGLU fails on Qwen
and any future weight-based circuit work needs a true bilinear surrogate model
trained from scratch.
**Blocks:** H-257 partial dependency; FE-118 independent.

### H-257: A quadratic probe at L19 substantially exceeds the linear DoM AUROC ceiling 0.7731
**Priority:** HIGH
**Motivated by:** 2410.08417 + F-2 + F-8
**Test:** Train logistic(α + Σ_i β_i (v_i^T h)²) on the same OOF 5-fold prefill
data, with v_i = top-k SVD components of the L19 prefill activation matrix, k
∈ {1, 2, 5, 10, 20}. Compare AUROC against linear DoM 0.7731 and against the
selective-prediction accuracy curve (F-8: 71.6% at coverage 0.5).
**Requires:** CPU only, ~10 min on cached prefill activations.
**Would change:** If quadratic AUROC > 0.85, the linear-probe ceiling is a
probe-class artefact and all selective-prediction / gated-compute results
should be re-run with the quadratic head. If quadratic AUROC ≈ 0.78, F-2's
ceiling is the real geometric ceiling — confirms the linear DoM frame and
demotes the bilinear interpretability angle.
**Blocks:** nothing.

---


### H-258: Zigzag-PH layer descriptors are at the Gaussian null on layer pruning
**Priority:** HIGH
**Motivated by:** 2410.11042 + F-10 + EXP-026
**Test:** Run paper's pipeline (kNN=4, m=4, p=1, FastZigZag) on (a) Qwen-2.5-1.5B
trained activations, (b) rank-matched Gaussian samples with same per-layer
empirical covariance, on cached pathway11_h100 NPZs (500 MATH-500 prompts × 28
layers). Compute bar Z_1 curves and 10%-of-max prune sets. ~1h CPU.
**Requires:** Dionysus2 + FastZigZag (pip-installable), cached NPZs, no GPU.
**Would change:** Confirm → F-10's "PH null on trained streams" generalizes to
zigzag and to pruning, killing zigzag-PH as a candidate signal for any
LLM-internal task. Reject → F-10 is narrower than the claim states (null on
correctness, not null on pruning); zigzag-PH becomes a candidate alternative
trajectory featurization worth keeping.
**Blocks:** H-259.

### H-259: L19 is in the prunable-plateau set per Gardinazzi 10%-of-max bar Z_1 criterion on Qwen-2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2410.11042 + F-2
**Test:** Apply paper's algorithm to Qwen-2.5-1.5B (Pile proxy, 500 prompts);
check (a) whether L19 ∈ prune set, (b) MATH-500 K=1 accuracy with prune-set
layers removed (single-pass H100 generation), (c) re-run prefill DoM AUROC on
the pruned model. ~1h CPU + 30min H100.
**Requires:** Cached NPZs + 30min H100 generation; Qwen-2.5-1.5B already
loaded in pathway11 pipeline.
**Would change:** Confirm L19-prunable + accuracy-preserved → F-2's "L19 is
the correctness layer" is overstated; AUROC reflects content available across
several plateau layers, not L19-specific. Confirm L19-prunable but
accuracy drops → paper's pruning criterion mis-identifies L19. L19 not in
prune set → F-2 + paper's framework consistent.
**Blocks:** nothing.

---


### H-260: A sparse top-K mask of L19 prefill dimensions matches or exceeds the full L19 DoM AUROC for correctness prediction
**Priority:** HIGH
**Motivated by:** 2410.12299 + F-2
**Test:** Compute per-dimension mean(correct − incorrect) at L19 prefill on the 500 MATH-500 problems; binarize to top-K with K ∈ {64, 128, 256, 512, 1024}; 5-fold OOF logistic regression on the masked-subspace projection of L19. AUROC vs F-2's 0.7731 baseline. ~30min CPU.
**Requires:** Cached prefill L19 NPZs, sklearn.
**Would change:** If top-K AUROC ≥ 0.7731 for K ≪ d (e.g. K=128 of d=2048), F-2's framing shifts from "the direction" to "this sparse subspace." If it's strictly worse, F-2's rank-1 framing is reinforced and SADI's claim doesn't transfer to the regression-of-correctness setting.
**Blocks:** H-261 weakly (informs whether per-cluster analysis is worth the next CPU hour).

### H-261: The prefill/final orthogonality (cos = 0.046, F-3) is partly explained by population averaging over instance-conditional DoM directions
**Priority:** MEDIUM
**Motivated by:** 2410.12299 + F-3
**Test:** KMeans cluster 500 problems on L19 prefill activations into K ∈ {4, 8, 16} clusters; per-cluster compute prefill_DoM_k and final_DoM_k; report distribution of cos(prefill_DoM_k, final_DoM_k). ~1h CPU.
**Requires:** Cached prefill + final-token L19 NPZs.
**Would change:** If mean per-cluster cos ≥ 0.3 with low variance, F-3's "two genuinely different directions" reading needs to be narrowed to "global means are nearly orthogonal because they shrink heterogeneous local directions to zero." If per-cluster cosines remain near 0, F-3 is robust and SADI's averaging-artifact hypothesis doesn't apply here.
**Blocks:** nothing.

### H-262: A SADI-HIDDEN-style dynamic-mask intervention moves MATH-500 K=1 accuracy more than a CAA-style fixed-DoM add at matched K and δ
**Priority:** HIGH
**Motivated by:** 2410.12299 + H-1 + F-2
**Test:** Two intervention arms during 1.5B greedy decoding: (a) fixed `A_q + δ × DoM_L19` add (CAA form), (b) dynamic `A_q + δ(A_q ⊙ M_topK)` (SADI form), same top-K identification. δ ∈ {0.1, 0.5, 1.0, 2.0}; K ∈ {64, 256}. Report MATH-500 K=1 accuracy vs no-intervention baseline (48.6%). Report per-arm best-cell delta. ~4h H100.
**Requires:** H100 pod for inference, cached prefill L19 NPZs for mask construction.
**Would change:** If dynamic > fixed by ≥3pt accuracy at the best (K, δ) cell, H-1's implementation form needs to be rewritten — fixed-DoM steering is dominated by sparse-subspace dynamic scaling. If the two arms tie or fixed wins, SADI's transfer claim from short multiple-choice to long-CoT MATH doesn't hold and our F-2 fixed-direction framing is reinforced.
**Blocks:** H-1 implementation form.

---


### H-263: A learned per-layer least-squares matrix W_l (incorrect-residual → correct-residual) applied at the last prompt token causally raises K=1 accuracy on Qwen-2.5-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2410.12462 (INCLINE) + F-2 + F-3
**Test:** Fit W_l per layer on cached P11 prefill last-token activations split by K=1 correctness; apply h_mix = h + α·W_l·h at each layer's last prompt token with α grid-searched on a held-out validation set; re-run MATH-500 with intervention; measure ΔK=1 accuracy. ~1h H100. Closed-form fit is 5 min CPU.
**Requires:** Qwen-2.5-1.5B loaded on H100, cached prefill last-token residuals at all 28 layers (P11 Stage 4), MATH-500 evaluation harness.
**Would change:** If +ΔK=1 ≥ +0.02 absolute, the prefill channel is causal (not just predictive) and H-1 is validated under a learned-matrix formulation rather than fixed-vector. If +ΔK=1 ≈ 0 or negative, F-3's two-channel framing is reinforced (prefill predicts what the model will do, final-token computes the answer; rotating prefill doesn't change the answer).
**Blocks:** H-1 reformulation; FE-125 outcome determines whether H-1 stays as written or is rewritten as "signed per-cluster matrix-bank steering".

### H-264: The optimal α in INCLINE-style intervention is signed-different across MATH-500 problem clusters (algebra vs counting vs geometry vs prealgebra)
**Priority:** MEDIUM
**Motivated by:** 2410.12462 Figure 4 (α positive for Spanish, negative for Chinese on the same task) + F-7
**Test:** Cluster MATH-500 into the 7 standard subjects. For each cluster fit α via grid search on validation residuals using the W_l from H-263. Report sign and magnitude of α per cluster.
**Requires:** Output of H-263, cluster labels (already in MATH-500 metadata).
**Would change:** If α flips sign across clusters, H-1's per-position bank must become a *signed* per-cluster bank. If α is uniformly positive, H-1's original formulation survives.
**Blocks:** H-1 final formulation.

---


### H-265: Label-free Stolfo-style DoM matches the supervised L19 prefill probe within ±0.01 AUROC
**Priority:** HIGH
**Motivated by:** 2410.12877 + F-2
**Test:** Build a paired-contrast set from P11 Stage-2 caches by matching correct-trace and incorrect-trace MATH-500 activations on prompt prefix; compute `u_19 = normalize(mean(h(correct) − h(incorrect)))` at the last input token; score `<h, u_19>` AUROC on held-out 100 problems (same fold split as F-2). 30min CPU.
**Requires:** existing P11 Stage-2 cached activations (1.5B, last-input-token position).
**Would change:** confirm → F-2's logistic probe is reframable as label-free DoM with explicit contrast set, and F-8's selective-prediction pipeline becomes label-free. Reject → supervised labels add real signal beyond contrast-set means; rules out the cheapest DoM derivation route.
**Blocks:** P11-FE330 outcome gates whether H-12 should include "Stolfo-style paired-contrast DoM" as a cheaper baseline before invoking SEPs.

### H-266: Multi-layer per-position injection beats single-layer summed injection for prefill-DoM steering on MATH-500
**Priority:** HIGH
**Motivated by:** 2410.12877 + H-1
**Test:** On a 50-problem MATH-500 subset, compare (a) summed `Σ_t α · w_t` injected at L19 only vs (b) each `w_t` injected at a different layer chosen by held-out perplexity gate, with `c` calibrated per P11-FE331. Score Δ-accuracy vs no-steering baseline. 1 H100-day.
**Requires:** H100, P11 Stage-2 per-token activations across all layers (some additional extraction needed for layers ≠ 19), dynamic-c computation from P11-FE331.
**Would change:** confirm → H-1 must be re-specified to multi-layer injection before any H100-day-scale sweep. Reject → summed single-layer injection is viable in our position-DoM setting (contrary to Stolfo's instruction-following finding), suggesting that correctness-direction superposition behaves differently from instruction-direction superposition.
**Blocks:** H-1 full sweep is gated on the outcome.

### H-267: Prefill_DoM and final_DoM are two redundant projections of one correctness feature, not two distinct mechanisms
**Priority:** MEDIUM
**Motivated by:** 2410.12877 + F-3
**Test:** Replicate Stolfo's instruct→base transfer protocol on Qwen-2.5-1.5B using both prefill_DoM and final_DoM as transferred directions; if both score similar held-out AUROC on the base model and steering with prefill_DoM at L19 affects final-token correctness as much as steering with final_DoM, the directions are functionally redundant. 1 H100-day.
**Requires:** Qwen-2.5-1.5B (base) extraction on MATH-500.
**Would change:** confirm → F-3's "two-circuit" framing weakens to "two redundant projections"; cos=0.046 becomes evidence of orthogonal-write rather than orthogonal-mechanism. Reject → F-3 strengthens; the two directions actually probe different circuits.
**Blocks:** nothing.

---


### H-268: Closed-form CoE-C beats single-layer L19 DoM on within-domain MATH-500 at 1024-tok labels
**Priority:** HIGH
**Motivated by:** 2410.13640 + F-9 + F-2
**Test:** Compute closed-form CoE-C from cached Qwen-2.5-1.5B P11 1024-tok activations on MATH-500. Report scalar OOF AUROC. Compare against L19 DoM 0.7731. ~20 min CPU.
**Requires:** CPU only; cached `pathway11_h100/.../*.npz`; no new GPU work. 7-layer-subsample approximation acceptable for first pass.
**Would change:** If CoE-C ≥ 0.80, F-9 weakens from "redundant" to "redundant only when reduced to 60 features"; H-6 effectively confirms with caveat. If CoE-C < 0.77, F-9 strengthens and the paper's headline claim is shown to be reduced when applied to our specific setup (single model, MATH-500, supervised baseline).
**Blocks:** H-269, P11-FE335, P11-FE336.

### H-269: Label-free CoE-C as the gating signal recovers F-8's 71.6% selective-prediction accuracy
**Priority:** HIGH
**Motivated by:** 2410.13640 + F-8
**Test:** Replace L19 DoM gating in `pathway11_h100/prefill_gated_compute/results.json` with closed-form CoE-C. Report acc@coverage 0.5. ~30 min CPU.
**Requires:** CPU only; cached features and the 500-problem K=2.5 schedule.
**Would change:** If matching or beating 71.6%, F-8's mechanism re-attributes from supervised prefill DoM to label-free trajectory geometry. If significantly below 71.6% (Δ ≥ 2 pp), F-8's supervised-DoM specificity holds and the paper's label-free claim is shown to undershoot in the selective-prediction regime even when matching in straight AUROC.
**Blocks:** nothing.

---


### H-270: Closed-form CoE-C beats single-layer L19 DoM on within-domain MATH-500 at 1024-tok labels
**Priority:** HIGH
**Motivated by:** 2410.13640 + F-9 + F-2
**Test:** Compute closed-form CoE-C from cached Qwen-2.5-1.5B P11 1024-tok activations on MATH-500. Report scalar OOF AUROC. Compare against L19 DoM 0.7731. ~20 min CPU.
**Requires:** CPU only; cached `pathway11_h100/.../*.npz`; no new GPU work. 7-layer-subsample approximation acceptable for first pass.
**Would change:** If CoE-C ≥ 0.80, F-9 weakens from "redundant" to "redundant only when reduced to 60 features"; H-6 effectively confirms with caveat. If CoE-C < 0.77, F-9 strengthens and the paper's headline claim is shown to be reduced when applied to our specific setup (single model, MATH-500, supervised baseline).
**Blocks:** H-271, P11-FE337, P11-FE338.

### H-271: Label-free CoE-C as the gating signal recovers F-8's 71.6% selective-prediction accuracy
**Priority:** HIGH
**Motivated by:** 2410.13640 + F-8
**Test:** Replace L19 DoM gating in `pathway11_h100/prefill_gated_compute/results.json` with closed-form CoE-C. Report acc@coverage 0.5. ~30 min CPU.
**Requires:** CPU only; cached features and the 500-problem K=2.5 schedule.
**Would change:** If matching or beating 71.6%, F-8's mechanism re-attributes from supervised prefill DoM to label-free trajectory geometry. If significantly below 71.6% (Δ ≥ 2 pp), F-8's supervised-DoM specificity holds and the paper's label-free claim is shown to undershoot in the selective-prediction regime even when matching in straight AUROC.
**Blocks:** nothing.

---


### H-272: σ-aware Linear-AcT probe beats mean-only DoM at L19 prefill
**Priority:** HIGH
**Motivated by:** 2410.23054 + F-2 (AUROC 0.7731 with mean-only DoM at L19
prefill) + AcT Figure 3 evidence that σ_a ≠ σ_b on real LLM activations.
**Test:** Fit closed-form Linear-AcT (ω_d, β_d) per coordinate d at L19
prefill on cached pathway11_h100 activations using correct (n=243) vs
incorrect (n=257) MATH-500 K=1 outcomes as source/target distributions.
Use the same 5-fold OOF split as EXP-037. Score `T(a) = ω·a + β` projected
onto the residual mean direction as the probe statistic; compare AUROC to
0.7731. Negative control: shuffle correctness labels.
**Requires:** ~20min CPU, no GPU, all data cached in
`pathway11_h100/prefill_gated_compute/`.
**Would change:** If AUROC > 0.7731 by ≥ 0.02 OOF, F-2's "L19 DoM is a
strong correctness predictor" framing must be updated to "L19 σ-aware
affine probe is a strong correctness predictor; mean-only DoM was a
floor." Updates `validate_claims.py` if the new headline number replaces
0.7731. If AUROC ≤ 0.7731, F-2 is robust to σ-aware extension and AcT's
σ-asymmetry argument doesn't apply at L19 for our task.
**Blocks:** H-273, H-274 (only worth running per-position / causal-stack
variants if single-layer σ-aware probe shows headroom first).

### H-273: Mean-pooling generation tokens raises cos(prefill_DoM, final_DoM)
**Priority:** MEDIUM
**Motivated by:** 2410.23054 (Appendix D: mean-pooling more robust than
last-token across models/tasks) + F-3 (cos = 0.046 was computed against
last-token-only final).
**Test:** Re-compute F-3's cosine using mean-pooled L19 activations across
all generated tokens per problem (in place of last-token-only). Use the
same 500 MATH-500 problems and the same DoM fit on correct/incorrect splits.
**Requires:** 30min CPU on cached pathway11_h100 + scratch/pathway10
activations.
**Would change:** If cos > 0.20, F-3's orthogonality narrative is partly
a pooling artifact and the H-1 design rationale weakens; project should
prefer Linear-AcT-style global maps over per-position-DoM. If cos stays
below 0.10, F-3 robust to pooling and the H-1 design rationale stands.
**Blocks:** nothing.

### H-274: Linear-AcT λ=1 steering moves MATH-500 K=1 accuracy ≥ 3pp
**Priority:** HIGH
**Motivated by:** 2410.23054 (consistent λ=1 peak on truthfulness,
concept induction, toxicity across Gemma2-2B and Llama3-8B; +5–8% MC1 on
TruthfulQA) + H-1 (per-position DoM steering proposal that this would
supersede).
**Test:** Fit Linear-AcT at L19 Post-LN-equivalent (Qwen-2.5-1.5B uses
RMSNorm) on mean-pooled prefill activations using correct/incorrect K=1
labels as source/target. Apply the fitted map at λ ∈ {0, 0.25, 0.5, 0.75, 1.0}
at every generation step on a held-out 100 MATH-500 problems. Score K=1
final-answer accuracy. Negative control: random-direction map with same
ω, β statistics; sanity control: λ=0 should reproduce baseline accuracy.
**Requires:** 1 H100-day for sweep + 30min CPU for fitting; data cached
except for fresh held-out generation pass.
**Would change:** If ΔAccuracy ≥ +3pp at λ=1, H-1's per-position-DoM
α-sweep is over-engineered; Linear-AcT λ=1 becomes the canonical steering
recipe. If ΔAccuracy < 1pp at all λ values, the project's "good
selective predictor, not a lever" closure clause fires with stronger
evidence than H-1 alone could have provided (because Linear-AcT is
literature's strongest principled steering primitive).
**Blocks:** the H-1 closure clause depends on this being run before H-1,
because failing AcT-style steering is a much stronger negative result
than failing per-position DoM α-sweeps.

---


### H-275: The L19 prefill DoM is out-of-distribution against natural Qwen-2.5-1.5B residual activations and naive scalar-multiplied addition (per H-1) is dominated by missing-default-component geometry rather than by the discriminative direction itself
**Priority:** HIGH
**Motivated by:** 2411.08790 + F-2 + H-1
**Test:** P11-FE343 (L₂-norm OOD diagnostic, 10 min CPU) followed by
P11-FE344 (post-injection OOD check across α-sweep, 15 min CPU). If the
DoM L₂ is below the 1st percentile of the natural-activation L₂
distribution and post-injection residual states drift outside the
pre-injection 99% L₂ interval at α ≥ 1, the hypothesis is confirmed and
H-1 must be rewritten with an in-distribution variant (gradient-pursuit
reconstruction inside the SAE basis, or default-component re-injection).
**Requires:** CPU only; cached pathway11_h100/prefill_gated_compute
NPZs (no GPU, no new data, no Qwen SAE).
**Would change:** *Confirm* → H-1's α-sweep should be expected to be
non-monotonic or to peak at very small α; H-1 needs an in-distribution
variant before consuming H100 budget. *Reject* → H-1 stays as specified
and Mayne et al.'s OOD critique is family-specific (Gemma 2 SAEs but
not Qwen residuals).
**Blocks:** H-1 (committing H100 budget to H-1's full sweep should wait
on this < 30-min CPU check).

### H-276: cos(prefill_DoM, final_DoM) = 0.046 reflects a default-component sign flip between prefill and generation contexts, not two mechanistically distinct correctness axes
**Priority:** MEDIUM
**Motivated by:** 2411.08790 (§3.2 + appendix D) + 2405.13967 (default
components) + F-3
**Test:** P11-FE345. Project prefill_DoM and final_token_DoM onto the
per-position K=1-correct-minus-incorrect centroid axis in L19 residual
space; if both DoMs have non-trivial signed projections onto the *same*
external correctness axis (just at different magnitudes/sign), the
"different mechanisms" reading of F-3 is undermined.
**Requires:** CPU only; cached per-token L19 activations from pathway
10 + pathway 11 Stage 2.
**Would change:** *Confirm* → F-3 should be reframed as "one correctness
axis with a context-dependent sign flip"; the project's narrative around
prefill vs final-token circuits collapses to a single circuit with two
context-dependent reads. *Reject* → F-3 stands as currently written;
prefill and final-token DoMs are mechanistically distinct.
**Blocks:** any "two-different-mechanisms" framing in writeups.

---


### H-277: Prefill DoM correctness signal is correlational, not causal — residual-stream patching at prefill does not flip K=1 outcome
**Priority:** CRITICAL
**Motivated by:** 2412.01113 + F-2 + F-7
**Test:** Replicate Kudo's residual-stream patching at L19 prefill on 50 MATH-500 problem pairs, length- and topic-matched, with one predicted correct and one predicted incorrect by the prefill DoM. Swap activations; record K=1 outcome flips. Threshold: ≥30% flip rate to call F-2 "causal"; <10% calls it "correlational." Already scoped as P11-FE3.
**Requires:** H100 pod for forward passes with hooks; cached prefill activations + matched-pair selection script
**Would change:** If correlational, F-2's narrative shifts from "model knows the answer" to "input is easy"; rewrites the F-2 mechanism section and downgrades F-8's "self-knowledge" framing to "input-difficulty gating."
**Blocks:** H-1 (steering position story), H-5 (familiarity-vs-decomposability) interpretation

### H-278: Prefill→final DoM rotation is continuous rather than abrupt — orthogonality is a long-drift artifact, not two-signal separation
**Priority:** HIGH
**Motivated by:** 2412.01113 + F-3 + EXP-relevant in pathway10
**Test:** Per-token, per-layer residuals from cached MATH-500 traces projected onto running DoM. Monotone angular drift (with bounded step-to-step jitter) vs sudden flip. Compute drift derivative distribution.
**Requires:** 4h CPU on cached `exp1_cross_model` NPZs
**Would change:** If continuous, F-3's "different signals" framing is weakened; merges with H-17 (RoPE-mechanical rotation). If abrupt, F-3 framing strengthens and we publish the angle-jump diagnostic.
**Blocks:** Nothing critical; informs F-3 narrative

### H-279: Step-boundary L19 DoM AUROC dominates prefill DoM AUROC at 7B but not 1.5B
**Priority:** HIGH
**Motivated by:** 2412.01113 + F-2 + F-8
**Test:** Step-boundary token probe (P11-FE346) on cached MATH-500 7B and 1.5B traces. AUROC at step boundaries vs at prefill, both 1.5B and 7B.
**Requires:** 1 day CPU on cached NPZs
**Would change:** If step-boundary > prefill on 7B but reversed on 1.5B, F-8 becomes "scale-conditioned" — selective prediction protocol must switch to step-boundary at ≥7B. Strong publication-shaping result.
**Blocks:** Cross-scale F-8 generalisation claim

### H-280: 7B logit-lens monotonicity is higher than 1.5B's on MATH-500 — faithfulness scales with size
**Priority:** MEDIUM
**Motivated by:** 2412.01113 + F-1 + F-2
**Test:** Per-layer top-1 probability of K=1 answer token at post-answer-newline; compute monotonicity fraction per problem; compare 1.5B vs 7B distributions (Mann-Whitney).
**Requires:** 1 day CPU on cached residuals + `lm_head` matrix from each model
**Would change:** Confirms or refutes that faithfulness = scale; either way, narrows the cross-scale interpretation of breathing (F-1) and prefill DoM (F-2).
**Blocks:** Nothing

---


### H-281: F-3's prefill/final-token orthogonality is partially a tied-embedding artifact
**Priority:** HIGH
**Motivated by:** 2412.15115 Table 1 (1.5B uses tied embeddings, 7B does not) + F-3 (cos = 0.046 on 1.5B).
**Test:** Compute cos(prefill_DoM, final_DoM) on Qwen2.5-7B using cached P11 H100 residuals. Compare to 0.046 baseline on 1.5B.
**Requires:** 2h CPU on cached prefill+final activations from `pathway11_h100/prefill_gated_compute/`. No new data.
**Would change:** Confirms F-3 mechanism if cos ≤ 0.10 on 7B; refutes the "two circuits" framing if cos ≥ 0.30 on 7B (in which case F-3 needs scoping to tied-embedding small models).
**Blocks:** any cross-scale generalization of F-3 until resolved.

### H-282: F-8 selective prediction is partly an artifact of GRPO variance-prioritization
**Priority:** HIGH
**Motivated by:** 2412.15115 §4.3 (GRPO prioritizes high-variance queries) + F-8 (71.6% selective accuracy at coverage 0.5 on Instruct).
**Test:** Re-run F-8 selective-prediction protocol on Qwen2.5-1.5B *base*. Compare base vs Instruct.
**Requires:** 1×H100 day for base activation extraction + 30min CPU. New activation cache for base model.
**Would change:** Confirms F-8 as a generic property if base ≥ 65%; reframes F-8 as a post-training artifact if base ≤ 55%.
**Blocks:** any "selective prediction is intrinsic to residual streams" generalization.

### H-283: L19 prefill DoM AUROC drops on execution-graded code
**Priority:** HIGH
**Motivated by:** 2412.15115 §3.1(3) (Qwen2-Math-RM-72B filtered pre-training synthetic data) + F-2 (AUROC 0.7731 on MATH-500).
**Test:** Evaluate the L19 prefill DoM probe on HumanEval/MBPP, scored by code execution rather than RM. Compare AUROC to the 0.7731 MATH-500 headline.
**Requires:** 1×H100 day for HumanEval/MBPP activation extraction + execution sandbox + 30min CPU for probe.
**Would change:** Confirms F-2 as a generic correctness direction if AUROC ≥ 0.70 on code; reframes F-2 as RM-policy-alignment if AUROC drops to ≤ 0.60.
**Blocks:** generalization of F-2 beyond math/RM-shaped distributions.

---


### H-284: Prefill DoM AUROC drops below 0.65 on non-math reasoning benchmarks
**Priority:** HIGH
**Motivated by:** 2412.15296 + F-2
**Test:** Generate K=1 Qwen-2.5-1.5B answers on BIG-Bench causal-judgment
(~200 q) and formal-fallacies (~250 q). Extract prefill L19 activations
through the existing P11 pipeline. Train 5-fold OOF DoM. Compare AUROC
to F-2's MATH-500 baseline 0.7731.
**Requires:** H100 (~1h generation) + CPU (~30min probe training).
Open-weights only — closed-source models in the source paper are not
required for this test.
**Would change:** If AUROC < 0.65 on either, F-2 should be re-scoped
from "prefill L19 DoM is a strong correctness predictor" to "...on
math reasoning specifically." If AUROC ≥ 0.7 on both, F-2's
generality is *strengthened* against the paper's pessimism — a
positive-result outcome that materially extends F-2.
**Blocks:** H-12 (cross-domain SEP comparison meaningful only after
this baseline lands).

### H-285: Prefill DoM residual-after-token-prob AUROC > 0.7 (i.e. token-prob does not subsume prefill DoM)
**Priority:** HIGH
**Motivated by:** 2412.15296 + F-2 + EXP that establishes 0.7731
**Test:** From cached K=1 MATH-500 greedy generations, compute mean
log-prob of the predicted answer span per problem. Fit single-feature
logreg on token-prob alone, AUROC. Then orthogonalize prefill-DoM
features against token-prob (residualize via OLS), retrain DoM,
recompute AUROC. Three numbers: token-prob-alone AUROC,
prefill-DoM-alone AUROC (= 0.7731), residualized-prefill-DoM AUROC.
**Requires:** CPU only (~20min). All inputs cached.
**Would change:** If residualized AUROC < 0.6, prefill DoM is largely
a re-skin of token-prob and F-2's significance is overstated — would
reframe the entire pathway-2-vs-pathway-11 story. If residualized
AUROC ≥ 0.7, prefill DoM carries genuinely new structural information
beyond logprobs, sharpening the case for residual-stream geometry as
an independent correctness channel.
**Blocks:** any future framing of prefill DoM as "more than logprobs."

### H-286: Reconsidered-answer accuracy on prefill-DoM-flagged low-confidence MATH-500 is ≤ first-pass accuracy
**Priority:** MEDIUM
**Motivated by:** 2412.15296 + F-8
**Test:** Run the persistence reprompt ("Are you sure? Please
reconsider.") on each of 500 MATH-500 K=1 problems for Qwen-2.5-1.5B.
Compute first-pass and second-pass accuracy on (a) bottom-50% prefill
DoM subset, (b) top-50% prefill DoM subset, (c) overall.
**Requires:** H100 (~30min) on the existing K=1 pipeline.
**Would change:** If second-pass accuracy on the bottom-50% subset is
≤ first-pass, F-8's recoverability premise is undermined and the
71.6%-at-coverage-0.5 gain must be re-attributed to K=2.5 sampling
variance rather than recovered answers. If second-pass accuracy is
*strictly higher* on the bottom-50% subset than on the top-50%
subset, this *strengthens* F-8 with a behavioral analogue.
**Blocks:** any selective-prediction writeup that claims
"recoverability" without this control.

---


### H-287: Step-boundary L19 probe with Bradley-Terry preference loss beats prefill-only DoM AUROC
**Priority:** HIGH
**Motivated by:** 2501.04519 + F-2
**Test:** Train a small linear+tanh head on cached K=8 trajectory L19 activations, segmented at `\n\n` step boundaries. Use top-2-Q vs bottom-2-Q pairs (Q from rStar-Math Eq. 2 back-prop on outcomes) as Bradley-Terry preference pairs. Compare OOF 5-fold AUROC against F-2's 0.7731 prefill baseline. ~4h CPU.
**Requires:** CPU only; cached L19 activations from `pathway11_h100/prefill_gated_compute/`.
**Would change:** If step-boundary AUROC ≥ 0.82, F-2 gets reframed as "best *prefill-time* signal"; if AUROC ≤ 0.75, F-2's claim that prefill carries the bulk of the correctness signal survives this denser-probe challenge.
**Blocks:** nothing.

### H-288: Prefill DoM correlates with mean-Q (decomposability) more strongly than with raw correct-rate (familiarity)
**Priority:** HIGH
**Motivated by:** 2501.04519 + H-5
**Test:** Compute back-propagated Q-values on our cached K=8 trajectories (Eq. 2). For each problem, compute (a) mean-Q over correct trajectories' steps and (b) raw correct-rate. Correlate prefill DoM percentile with each. ~2h CPU.
**Requires:** CPU only; cached outcomes + activations.
**Would change:** If r(DoM, mean-Q) > r(DoM, correct-rate), prefill DoM encodes step-level decomposability ≈ rStar-Math-style "is this problem reducible to verifiable atomic steps." If r(correct-rate) > r(mean-Q), prefill DoM is plain familiarity. Either result resolves H-5.
**Blocks:** H-5.

### H-289: rStar-Math's PPM-MCTS at K=8 closes the D-bucket gap (i.e., D-bucket fragility is policy-specific, not geometric)
**Priority:** MEDIUM
**Motivated by:** 2501.04519 + F-7
**Test:** Run rStar-Math 7B policy + PPM at K=8 on our 36 D-bucket problems. Report fraction solved at ≥7/8 trajectories correct.
**Requires:** rStar-Math weights (BLOCKED on github.com/microsoft/rStar release); 1 day H100 once available.
**Would change:** If ≥30/36 lifted, F-7's "distinctive collective signature" downgrades to "Qwen2.5-Math-Instruct-specific local minima." If ≤10/36 lifted, F-7's substrate-level claim is reinforced.
**Blocks:** F-7 deployment claims.

---


### H-290: SAE-feature L19 correctness probe beats raw L19 DoM AUROC on Qwen-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2501.11036 (LF-Steering Table 3, 16% accuracy gap between feature-level and neuron-level on RobustBOOLQ) + F-2 (raw L19 DoM AUROC 0.7731 ceiling).
**Test:** Load publicly-available Qwen-2.5-1.5B SAE; encode cached `pathway11_h100/prefill_gated_compute` L19 activations on the 500 MATH-500 problems; identify key features via the LF-Steering threshold-on-mean-difference recipe (`t=0.10`); fit logistic probe on the feature mask; report 5-fold OOF AUROC. ~1h CPU + 1h reconnaissance for the SAE checkpoint.
**Requires:** CPU, cached L19 NPZs (already on disk per DATA_MANIFEST.md), public Qwen-1.5B SAE.
**Would change:** On confirm (feature-AUROC ≥ 0.83) → F-2 is reframed as "best *raw-direction* L19 probe" and the narrative needs a polysemanticity qualifier; on reject (feature-AUROC < 0.79) → F-2's "single direction is sufficient" claim is strengthened.
**Blocks:** H-291 (feature-level steering), partial H-13 (head-level attribution becomes a weak comparison).

### H-291: SAE-feature DoM steering moves Qwen-1.5B MATH-500 K=1 accuracy by >5pp where raw DoM steering does not
**Priority:** HIGH
**Motivated by:** 2501.11036 (LF-Steering: feature-level α=20 yields +20pp on RobustBOOLQ vs +4pp for raw hidden-state steering) + H-1 (current DoM steering plan is α∈{-4,...,4}, raw direction).
**Test:** Apply the LF-Steering recipe (TopK SAE → key features via correct-vs-incorrect mean-difference threshold → add α·g_i → decode) to Qwen-1.5B at L19 during MATH-500 K=1 generation. Sweep α∈{5,10,20,30,50}. Compare to raw DoM-direction steering at matched α and matched α-magnitude. Run AG News + IMDB locality controls.
**Requires:** 1 H100-day, depends on H-290 producing a usable SAE encoder.
**Would change:** On confirm → H-1 retracted in favor of feature-level steering; raw DoM-direction steering joins ITI/CAA as polysemanticity-limited baselines; F-2 narrative tightens to "best *probe* under polysemantic constraints, weaker as a *steering substrate*". On reject → H-1's premise (steerable L19 correctness) is in serious trouble; either correctness is not single-layer-steerable, or our SAE checkpoint is wrong.
**Blocks:** H-1 promotion to F-status; F-9 reframing as "single-layer raw direction" not "single-layer".

---


### H-292: Last-decoded-token at the last layer matches or beats prefill L19 DoM on Qwen-2.5-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2501.12934 + F-2
**Test:** Train OPENIA-style 128/64 MLP probe on (a) last-decoded-token at last layer (L27), (b) prefill-last-token at L19 (F-2's setting), on cached `pathway11_h100/` MATH-500 NPZs. Report 5-fold AUROC for each. ~20min CPU.
**Requires:** CPU only, cached NPZs, Qwen-2.5-1.5B Stage 2 hidden states across all layers (or at minimum L19 and L27).
**Would change:** If (a) ≥ 0.7731 = (b) by >0.02, F-2's "prefill > final" is overturned for this model on this task — at minimum, "prefill > final" is a *layer-mismatched* comparison and the real question is "best layer × best token" jointly. Reframes Pathway 10's prefill/final orthogonality story (F-3) and the prefill-gated-compute pipeline (F-8). On reject (a < b by >0.02), F-2 strengthens.
**Blocks:** decisive resolution of refutation #1 above; gates H-293.

### H-293: At the last layer, the correctness direction is approximately token-invariant (cos ≥ 0.5 across token positions)
**Priority:** HIGH
**Motivated by:** 2501.12934 + F-3
**Test:** On Qwen-2.5-1.5B MATH-500, compute DoM at every layer L for first-prompt-token and final-decoded-token states. Plot cos(DoM_first(L), DoM_last(L)) vs L. ~20min CPU on cached layerwise_dom NPZs (or 1h fresh extraction if missing).
**Requires:** CPU only, cached layerwise_dom data.
**Would change:** If cos rises monotonically and exceeds 0.5 at L≥24, F-3's near-orthogonality is layer-local: at the readout layer, prefill and final are aligned, and the L19 cos=0.046 is a mid-depth disentanglement that gets re-mixed. F-3 narrative ("prefill ⊥ final") gets a "but only at intermediate depth" caveat. On reject (cos stays near 0 at all layers), F-3 is robust and OPENIA's stable-across-tokens result is code-domain-specific.
**Blocks:** nothing.

### H-294: OPENIA-style non-linear MLP probe outperforms linear DoM by >0.02 AUROC on Qwen-2.5-1.5B MATH-500 prefill L19
**Priority:** MEDIUM
**Motivated by:** 2501.12934
**Test:** Train 128/64 MLP probe on prefill L19 hidden states; 50 epochs, batch 32, lr 1e-3; 5-fold OOF AUROC. Compare to F-2's linear DoM 0.7731. ~20min CPU.
**Requires:** CPU only, cached prefill L19 NPZ.
**Would change:** If MLP > linear by >0.02 AUROC, the prefill correctness signal is non-linearly decodable — meaning any *linear* steering intervention (H-1 per-position DoM) is a lower bound on the manipulable signal. If MLP ≈ linear, linear DoM extracts all of the readable correctness information and our linear-only framing is justified.
**Blocks:** nothing; informs H-1 ceiling.

### H-295: Best-of-K selection guided by prefill DoM probe yields ≥ 5 % pass@1 improvement on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2501.12934 + F-8
**Test:** On cached K=8 MATH-500 samples (Qwen-2.5-1.5B and 7B), score each candidate by prefill-L19 DoM probe, select top-1, measure pass@1. Compare to plain K=1 sampling (48.6 % for 1.5B; 73.2 % for 7B). ~20min CPU.
**Requires:** CPU only; needs cached multi-sample MATH-500 data with per-sample prefill activations (verify availability).
**Would change:** OPENIA reports ~7 % average lift across settings. If we hit ≥ 5 %, our F-8 selective-prediction story extends to selection (not just abstention) and motivates packaging the probe as a decoder plugin. If <2 %, F-8's coverage-0.5-71.6 % framing is the binding ceiling and we should not advertise selection use cases.
**Blocks:** nothing; complements F-8.

---


### H-296: Prefill L19 DoM AUROC degrades on reasoning-distilled checkpoints
**Priority:** HIGH
**Motivated by:** 2501.12948 + F-2
**Test:** Extract L19 prefill activations from `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on the full MATH-500 set. Compute 5-fold OOF logistic-regression AUROC for K=1 correctness. Compare to the 0.7731 baseline measured on base Qwen-2.5-1.5B at 1024-tok labels. Decision threshold: AUROC drop of ≥ 0.06 (i.e. < 0.71) is the falsifier.
**Requires:** H100 (~30 min for extraction), CPU for fitting, no new data — MATH-500 prompts are reused.
**Would change:** On confirm (drop), F-2 is rewritten with a regime-of-validity clause restricting it to non-reflection-trained checkpoints; the project narrative shifts from "we found a universal correctness direction" to "we found a direction that decays under post-training that adds reflection." On reject (AUROC holds), F-2 is *more* general than currently claimed and the paper's emergent-reasoning framing becomes a non-event for our probe.
**Blocks:** H-297 (cos test on R1-distill), H-298 (breathing on R1-distill); these all require the same activation extraction pass.

### H-297: Prefill ⊥ Final orthogonality (F-3) breaks on reasoning-distilled checkpoints
**Priority:** HIGH
**Motivated by:** 2501.12948 + F-3
**Test:** Same activations as H-296. Compute prefill DoM and final-token DoM on R1-Distill-Qwen-1.5B; report cos. Decision threshold: cos > 0.25 (vs 0.046 baseline) refutes F-3's universality. Cos in [0.05, 0.25] is ambiguous — needs intermediate Qwen-2.5-Math-1.5B comparison.
**Requires:** Output of H-296 pipeline.
**Would change:** On confirm (cos rises), F-3's orthogonality stops being a deep finding about LLM geometry and becomes a marker for absence-of-learned-reflection. PERSPECTIVES.md narrative ("most surprising geometric finding") needs revision. On reject (cos stays near zero), the orthogonality is robust to post-training and the framing strengthens.
**Blocks:** nothing — independent test once H-296 activations exist.

### H-298: Dimensional breathing amplitude shifts after long-CoT distillation
**Priority:** MEDIUM
**Motivated by:** 2501.12948 + F-1 + H-10
**Test:** Compute peak PR and breathing amplitude on a matched 50-prompt MATH-500 subset for three checkpoints: base Qwen-2.5-1.5B, Qwen-2.5-Math-1.5B (R1-distill base), DeepSeek-R1-Distill-Qwen-1.5B. Two-sample test on amplitude distributions. Decision threshold: p < 0.01 with effect size > 1σ refutes "breathing is universal."
**Requires:** H100 for activations on all three checkpoints (~2h), CPU for PR (~1h).
**Would change:** On confirm (amplitude shifts), H-10 graduates to confirmed and F-1 takes a "modulo post-training" caveat; H-17 (RoPE-mechanical) is weakened. On reject (amplitudes match), F-1 is robust to a strong post-training perturbation, which is itself a publishable strengthening.
**Blocks:** nothing.

---


### H-299: Pairwise relative-confidence prompting matches or beats supervised prefill L19 DoM AUROC on Qwen-2.5-1.5B / MATH-500
**Priority:** HIGH
**Motivated by:** 2502.01126 + F-2
**Test:** Implement Algorithm 1 + TrueSkill aggregation, run on cached MATH-500 K=1 answers (n=15 pairwise comparisons per question, total 7500 forward passes on Qwen-1.5B). Compute AUROC against the 5-fold OOF split used for F-2. Compare to F-2's 0.7731.
**Requires:** 1h H100, Qwen-2.5-1.5B (already cached), pathway11 MATH-500 results.json
**Would change:** If relative-confidence AUROC ≥ 0.7731, F-2's headline must be reframed: "supervised internal probe matches free linguistic elicitation, no value-add." If < 0.7731 by ≥ 5 points, F-2 is strengthened — paper supplies the strongest extant label-free comparator and the DoM beats it.
**Blocks:** H-300 (only meaningful if the elicitation baseline is established)

### H-300: Question-only ("difficulty-only") relative-confidence on Qwen-1.5B correlates r ≥ 0.4 with prefill L19 DoM scalars
**Priority:** HIGH
**Motivated by:** 2502.01126 (Section 5: difficulty-only AUC 0.819 vs answer-conditioned 0.872) + F-3 + H-5
**Test:** Variant of H-299 prompt with answers stripped ("which question is harder"). TrueSkill-aggregate, then correlate the resulting scalar per question with the cached prefill L19 DoM scalar. Pearson r and Spearman ρ.
**Requires:** 1h H100, same setup as H-299
**Would change:** If r ≥ 0.4, prefill DoM is largely a difficulty proxy — F-3's orthogonality of prefill vs final has the simpler interpretation "question-difficulty axis vs answer-verification axis," and H-5's "familiarity / decomposability" framing is supported. If r < 0.2, prefill DoM is encoding something *beyond* difficulty (good for F-3's two-mechanism framing).
**Blocks:** nothing

### H-301: TrueSkill-aggregated relative confidence dominates Semantic Entropy Probes as a label-free correctness signal
**Priority:** MEDIUM
**Motivated by:** 2502.01126 + H-12
**Test:** Three-way bake-off: supervised DoM (F-2), SEP (Farquhar 2406.15927), relative-confidence (this paper). Same MATH-500 / Qwen-1.5B / 5-fold OOF split. Report AUROC and selective-classification AUC at coverage 0.25/0.50/0.75.
**Requires:** ~3h H100 total (FE126 + SEP sampling)
**Would change:** Forces a decision in H-12 between SEP and relative-confidence as the canonical label-free comparator. If relative > SEP > random, H-12 is reformulated to use relative-confidence as the SEP replacement candidate.
**Blocks:** nothing (but informs how H-12 is finalized)

---


### H-302: L19 is a "general legibility" layer, not a correctness-specific layer
**Priority:** HIGH
**Motivated by:** 2502.01657 + F-2
**Test:** Train a per-layer linear probe on Qwen-2.5-1.5B Stage-2 NPZs targeting MATH-500 problem-type labels (algebra/geometry/number-theory/probability/etc.). Plot accuracy vs layer. ~30min CPU.
**Requires:** CPU only, cached Pathway-11 H100 Stage-2 NPZs (already on disk per DATA_MANIFEST.md).
**Would change:** If problem-type classification peaks within ±2 layers of L19, F-2's framing shifts from "L19 is privileged for correctness" to "L19 is the layer where all problem-relevant linear features peak — correctness is one of many legible signals". If peak is ≥3 layers away, F-2 is correctness-specific (current framing holds).
**Blocks:** nothing.

### H-303: Per-prompt steering at L19 with predicted-DoM source vectors moves MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2502.01657 + F-3 + P10 v2 abandonment of fixed-α steering
**Test:** Train MLP `prefill_h_L19 → predicted_per_prompt_DoM` on K-fold split of MATH-500. Splice predicted vector at L19 with α ∈ {0.1, 0.3, 0.5, 0.7}. Measure accuracy on held-out fold. ~1h H100.
**Requires:** H100 (Qwen 1.5B inference), cached MATH-500 prefill activations.
**Would change:** If accuracy moves measurably (>+2pp), P10 v2's "fixed-vector steering is dead" was wrong-source not wrong-mechanism — un-parks H-1 and re-opens P10-FE1 under the per-prompt framing. If null, P10 v2 verdict stands and 2502.01657's success was VSA-target-specific.
**Blocks:** H-1 un-parking decision.

### H-304: Structured (HRR/VSA-target) linear probes at L19 beat scalar DoM on correctness AUROC
**Priority:** MEDIUM
**Motivated by:** 2502.01657 + F-2
**Test:** Train a multi-target linear probe at L19 with HRR-style structured targets (problem-type ⊛ difficulty-band ⊛ correctness-bit). Compare correctness AUROC against scalar DoM. ~2h CPU on cached NPZs.
**Requires:** CPU, cached Stage-2 NPZs, MATH-500 problem-type metadata.
**Would change:** If structured probe AUROC > 0.7731, scalar DoM is undercounting correctness signal — F-2 should be re-stated as a *structured* direction, and H-1 / P10 vectors should be derived from the structured probe rather than mass-mean. If ≤0.7731, scalar DoM is the right reduction (current F-2 framing optimal).
**Blocks:** nothing.

---


### H-305: L19 is not the globally optimal layer for correctness prediction in Qwen-2.5-1.5B

**Priority:** MEDIUM
**Motivated by:** 2502.03407 (layer sweep shows sharp adjacent-layer AUROC variation in analogous probing task) + F-2 (L19 chosen empirically without full sweep)
**Test:** Run 5-fold OOF logistic probe (C=0.1, l2, same protocol as F-2) on all 28 Qwen-2.5-1.5B layers using cached Stage 2 NPZs. Record AUROC at each layer. If max > 0.7731 at a different layer: update F-2 headline number and layer claim. If L19 remains max: F-2 is robustly confirmed.
**Requires:** CPU, cached Stage 2 NPZs. ~2h. Zero additional extraction.
**Would change:** If confirmed (non-L19 layer wins by ≥0.01): F-2 headline AUROC rises and "L19 is special" weakens to "L19 was the empirical winner in F-2's regime." If rejected (L19 stays max): adds robustness evidence to F-2 and partially answers whether 0.7731 is near the true ceiling.
**Blocks:** nothing.

---


### H-306: A constant-direction VSV-style intervention moves Qwen MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2502.03628 + F-3 + H-1
**Test:** Extract v = h_L19[last_prefill, X_p] − h_L19[last_prefill, X_n]
for X_p = system+question+CoT-marker vs X_n = system+question, inject
λ·v into all layer residual streams at every generation timestep with
norm rescaling (Eq. 6). Sweep λ ∈ {0.05, 0.10, 0.15, 0.20} on 500
MATH-500 problems. Report ΔK=1 accuracy and cosine of v against the
F-2 supervised L19 prefill DoM. Est: 30 min H100.
**Requires:** H100 pod (for fresh K=1 generation under intervention),
Qwen-2.5-1.5B, cached prefill activations for the cosine intermediate.
**Would change:** If accuracy moves > +1% at any λ, F-3's "orthogonal
directions" framing is undermined as load-bearing for steering design,
and H-1's per-position bank is shown unnecessary in at least the
constant-direction regime. If accuracy is flat or negative, F-3 + H-17
(rotation hypothesis) survive.
**Blocks:** P11-FE3 (activation patching) — both depend on H100 pod for
intervention runs.

### H-307: SLA-style window logit-lens ensemble outperforms single-layer L19 DoM for correctness prediction
**Priority:** HIGH
**Motivated by:** 2502.03628 + F-9 + EXP-029
**Test:** Apply Qwen LM head to cached Stage 2 final-token activations
at L24–L27, average to get o_aug, ensemble with L28 logits at γ ∈
{0.1, 0.2, 0.3, 0.4}, train a linear correctness probe on the gold
answer first-token rank in õ, compare AUROC to 0.7186 (final-token L28
baseline) and 0.7731 (prefill L19 baseline). Repeat for window L20–L27
(prefill side). Est: 1 h CPU.
**Requires:** CPU only, cached Stage 2 final-token + prefill residuals,
Qwen LM head weights (already on disk).
**Would change:** If window-AUROC > 0.7186 + 0.014 (F-9's redundancy
margin), F-9 is wrong and the CoE-60 redundancy claim must be retracted
or scoped. If window-AUROC ≤ 0.7186 + ε, F-9 survives a strong
out-of-domain methodology test.
**Blocks:** nothing.

---


### H-308: Mahalanobis distance to μ̂⁺ at L19 prefill beats DoM AUROC by ≥ 0.02 on MATH-500

**Priority:** HIGH
**Motivated by:** 2502.04043 + F-2
**Test:** P11-FE385 — fit Ledoit-Wolf Σ̂ on cached L19 prefill activations of correct + incorrect MATH-500 trajectories, compute Mahalanobis distance classifier OOF 5-fold, compare to F-2 anchor 0.7731. ~30 min CPU.
**Requires:** CPU only, cached `pathway11_h100/prefill_gated_compute/` NPZs.
**Would change:** Confirm → F-2 must be reframed as "first-order summary; second-order adds X pp"; Reject → F-2's linear/single-direction framing is sufficient; the FLORAIN-on-TruthfulQA gain over ITI is task-specific to truthfulness, not transferable.
**Blocks:** H-309 (we should not run a costly FLORAIN training before knowing if Σ̂ even matters at our regime).

### H-309: Input-dependent nonlinear low-rank steering (FLORAIN) beats fixed-vector DoM steering on MATH-500 K=1

**Priority:** MEDIUM
**Motivated by:** 2502.04043 + H-1
**Test:** P11-FE386 — train FLORAIN at L19 on Qwen-2.5-1.5B with MATH-500 correct/incorrect, measure Δ K=1 on held-out fold vs fixed-vector DoM steering at matched magnitude. ~1 day H100.
**Requires:** H100, Qwen-2.5-1.5B, fresh training (no cache reuse — FLORAIN parameters W, R, b, s).
**Would change:** Confirm → H-1 was failing because its functional form was too rigid, not because steering doesn't work; Reject → fixed-vector and input-dependent steering are equivalent in the small-Δ-K regime, evidence that no single-layer activation edit moves MATH-500 meaningfully.
**Blocks:** Any future P11 experiments on input-conditional intervention vs additive intervention.

### H-310: L19 is the global-DoM-optimal layer but not the Mahalanobis-optimal layer

**Priority:** MEDIUM
**Motivated by:** 2502.04043 + F-2
**Test:** P11-FE387 — sweep Mahalanobis AUROC at L4/L9/L14/L19/L24 on cached activations. ~1 h CPU.
**Requires:** CPU, cached multi-layer prefill NPZs (verify availability — P11 stage extractions).
**Would change:** Confirm → F-2's layer-specificity is a first-order phenomenon; Mahalanobis peaks elsewhere, suggesting different mechanisms at different layers. Reject → L19 is structurally special, not just first-order convenient.
**Blocks:** nothing.

---


### H-311: TELLME-style disentanglement at L19 raises Qwen2.5-1.5B prefill DoM AUROC above 0.85
**Priority:** HIGH
**Motivated by:** 2502.05242 + F-2
**Test:** P11-FE389 — apply TELLME LoRA to Qwen2.5-1.5B at L19 (and L22, the TELLME 80% rule) using K=1 correct/incorrect MATH-500 prefills as contrastive pairs and UltraChat as Retain Set; re-extract prefill activations on holdout fold; recompute OOF 5-fold prefill DoM AUROC. Est: 1d H100.
**Requires:** H100 pod; UltraChat split; LoRA hyperparameters from TELLME Appendix B.2.
**Would change:** Confirm → F-2 reframes from "discovery of the model's own correctness encoding" to "tip-of-iceberg lower bound on what disentanglement training exposes"; selective-prediction story (F-8) becomes a tunable training recipe rather than a passive readout. Reject (AUROC stays 0.77 ± 0.03) → F-2 hardens; the prefill DoM is genuinely architectural and not under-developed.
**Blocks:** H-312 informational (Self-Sim post-edit only meaningful if H-311 confirms)

### H-312: Self-Sim cosine probe on base 1.5B prefill matches supervised DoM AUROC within 0.05
**Priority:** HIGH
**Motivated by:** 2502.05242 + F-2
**Test:** P11-FE390 — on cached `pathway11_h100/prefill_gated_compute/` NPZs, compute mean correct-prefill / mean incorrect-prefill on training fold; classify holdout problems by argmax cosine; report AUROC. No training. Est: 20min CPU.
**Requires:** Cached prefill NPZs (already on disk).
**Would change:** Confirm → F-2's 0.7731 is largely linear-separability of mean shifts, not a sophisticated probe finding; H-12's case for label-free probes is half-made by mean-cosine alone. Reject (Self-Sim ≤ 0.60) → the supervised DoM is doing genuine work above mean-cosine, and TELLME's predicted headroom for FE73 is large.
**Blocks:** nothing — informational gate on H-311.

---


### H-313: Gradient-direction probes match or beat residual-stream DoM probes
**Priority:** HIGH
**Motivated by:** 2502.05911 + F-2
**Test:** LoRA-tune Qwen-2.5-1.5B on MATH-500 with idk=K=0/8 and ik=K≥7/8 (1 epoch, r=64, α=16, lr=2e-4). Extract per-sample LoRA gradients; project to low-dim via Johnson-Lindenstrauss; compute g_idk = mean over D_idk. Score each held-out problem by ⟨∇L(x), g_idk⟩; compute 5-fold OOF AUROC. Compare to F-2's 0.7731.
**Requires:** H100 (4h LoRA tune + grad extraction), Qwen-2.5-1.5B, EXP-033 K=8 cache for ik/idk labels.
**Would change:** If gradient probe AUROC > 0.80, F-2 should be reframed as "the correctness direction is a multi-layer gradient direction; L19 DoM is a downstream projection." If AUROC ≈ 0.77, the two probes are duals and DoM is the cheaper choice. If AUROC < 0.70, gradient-space adds noise and F-2's single-layer story is vindicated.
**Blocks:** H-314.

### H-314: Class-gradient orthogonality is a factual-recall artifact, not a reasoning property
**Priority:** HIGH
**Motivated by:** 2502.05911 + F-3
**Test:** Using LoRA gradients from H-313, compute cos(g_ik, g_idk) on MATH-500. GRAIT reports near-orthogonality on TriviaQA (Appendix A.3) — does it hold on multi-step reasoning?
**Requires:** Subset of H-313 (free additional measurement).
**Would change:** If cos > 0.3, F-3's near-orthogonality is regime-specific (factual recall only) and H-1 (per-position DoM steering) loses its theoretical foundation. If cos < 0.1, F-3 generalizes to reasoning and the orthogonality is a property of correctness gradients in general.
**Blocks:** H-1 reframing if cos > 0.3.

### H-315: Training-time refusal injection beats test-time DoM gating at matched compute
**Priority:** MEDIUM
**Motivated by:** 2502.05911 + F-8
**Test:** Apply full GRAIT pipeline (3 epochs, T_C=0.5, τ=0.05) to Qwen-2.5-1.5B on MATH-500 idk/ik split. Evaluate THS at the operating point matching F-8's 50% coverage. Compare to F-8's prefill-DoM-gated baseline (71.6% accuracy on answered, K=2.5 avg).
**Requires:** 8h H100 (full GRAIT training), MATH-500 idk/ik labels (cached).
**Would change:** If GRAIT THS > prefill-DoM THS at matched compute, the dominant abstention paradigm should shift from test-time probes (F-8) to training-time loss shaping. If GRAIT loses, F-8's "no fine-tuning required" advantage is preserved as a major finding.
**Blocks:** nothing.

---


### H-316: Optimization-based Δ at L19 outperforms heuristic L19 DoM as a MATH-500 steering vector
**Priority:** HIGH
**Motivated by:** 2502.06115 + EXP-037 (prefill AUROC 0.7731, the probe-direction H-1 plans to use as steering vector)
**Test:** Implement Eq. 2 of 2502.06115 at L19 last-token on Qwen-2.5-1.5B with N=10 MATH-500 pairs (5 correct + 5 incorrect) and NLL-of-gold-answer task loss + lasso (γ=0.01) + group-lasso (λ=0.01). Sweep injection α ∈ {0.1, 0.3, 1.0, 3.0}. Compare MATH-500 K=1 accuracy lift on the 490 held-out problems vs H-1's heuristic DoM steering at the same layer / α. Estimated time: 4h H100 once last-token-L19 re-extract is done.
**Requires:** H100 GPU, Qwen-2.5-1.5B-Instruct, last-token L19 activations from MATH-500 generations (re-extract if not cached), gold-answer tokenizations.
**Would change:** On confirm — H-1 rewritten to define steering vector via optimization, not heuristic DoM; opens the door to head-sparsity introspection (group lasso highlights the active heads). On reject — heuristic DoM is the right primitive after all and H-1 stands as written.
**Blocks:** Final form of H-1.

### H-317: Optimization-objective optimal layer for MATH-500 steering is shallower than L19
**Priority:** MEDIUM
**Motivated by:** 2502.06115 (ℓ=4 optimum on Llama3-8B) + F-2 (L19 optimum on Qwen-2.5-1.5B probe objective)
**Test:** Run unregularized Eq. 2 at layers ℓ ∈ {1, 4, 7, 10, 13, 16, 19, 22, 25, 28} on Qwen-2.5-1.5B with N=10 MATH-500 pairs; record Exact-Match of gold answer at each layer. Fit a smooth curve. If argmax ≪ 19, F-2 must be re-annotated as "probe-objective optimum, not generation-objective optimum". Estimated time: 1 day H100.
**Requires:** H100 GPU, all 28 Qwen-2.5-1.5B activation layers (cached for prefill, may need extraction for last-token), MATH-500 evaluation harness.
**Would change:** On confirm (ℓ_opt ≪ 19) — F-2 wording downgraded; the literature's "shallow layers carry steering signal" prevails over our probe finding; H-1 should switch to shallow layer. On reject (ℓ_opt ≈ 19) — F-2's L19 specialness extends to active steering objective; strong support for the L19 framing.
**Blocks:** H-1 layer choice.

---


### H-318: P(True) verbal probe matches or exceeds prefill L19 DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2502.06233 + F-2
**Test:** Run the P(True) confidence prompt over the 500×8 cached MATH-500 generations from `pathway11_h100/prefill_gated_compute/`. Score each path with P(token="Yes" | confidence prompt) and compute (a) AUROC against K=1 correctness, (b) Within-Question Discrimination per CISC §6.1, (c) head-to-head against prefill DoM (F-2) and final-token DoM. ~30 min H100 + 1h CPU.
**Requires:** Cached K=8 generations (have); a forward pass with the confidence-prompt suffix on Qwen-2.5-1.5B (~30 min on H100); WQD scoring code (new, ~50 LOC).
**Would change:** *Confirm (P(True) ≥ 0.77 AUROC)*: F-2 reframes — "geometric prefill signal is one of several signals at this AUROC level; P(True) verbal probe matches without needing supervised DoM training". *Reject (P(True) < 0.7186)*: F-2 strengthens — "supervised geometric probes outperform unsupervised verbal probes at correctness prediction even when verbal probes work for path-weighting".
**Blocks:** H-12 (semantic-entropy probes vs supervised DoM) — until we know how P(True) compares to prefill DoM, the SEP comparison is mis-specified.

### H-319: CISC-weighted majority vote on K=8 beats prefill-DoM selective prediction at matched compute
**Priority:** HIGH
**Motivated by:** 2502.06233 + F-8 + H-19
**Test:** Implement CISC (P(True) + softmax temperature scaling + confidence-weighted vote) on Qwen-2.5-1.5B/MATH-500 K=8. Compare accuracy at matched compute (K=2.5 avg, the F-8 operating point) against (a) F-8's prefill-DoM selective prediction at coverage 0.5, (b) H-19's PANL-routed K=1+verify. ~30 min H100 + 2h CPU.
**Requires:** Cached K=8 generations + P(True) scores from H-318; CISC implementation from `google-research/cisc` repo.
**Would change:** *Confirm (CISC > F-8 + H-19 at matched compute)*: weighted-vote at K=8 supersedes our hard-threshold abstention story; F-8 and H-19 demote to ablations. *Reject*: F-8/H-19 stand as the stronger inference-time scaling baseline.
**Blocks:** nothing.

### H-320: Within-Question Discrimination (WQD) replaces AUROC as Pathway 11's primary correctness-probe metric
**Priority:** MEDIUM
**Motivated by:** 2502.06233 + F-2 + F-8 + F-9
**Test:** For every correctness probe in Pathway 11 (prefill DoM, final-token DoM, CoE-60, P(True)), compute WQD on the 500×8 MATH set alongside the existing AUROC. Build a comparison table: AUROC ranking vs WQD ranking. If they disagree, WQD is the operationally meaningful metric (per CISC §6) and our framing should adopt it. ~2h CPU pure post-hoc.
**Requires:** Cached scored arrays for all probes; ~50 LOC for the WQD function.
**Would change:** *AUROC ↔ WQD agree*: F-2/F-8/F-9 stand as-is, WQD becomes a secondary cross-check. *AUROC ↔ WQD disagree*: F-2/F-8/F-9 numbers must be re-reported with WQD as the headline metric, since AUROC overweights between-question ordering that doesn't matter for selective prediction.
**Blocks:** H-12 reformulation (semantic-entropy probes need WQD evaluation, not just calibration).

---


### H-321: A 7B open-weight PRM beats prefill L19 DoM as a K=1-correctness predictor on Qwen2.5-1.5B/7B MATH-500 trajectories
**Priority:** HIGH
**Motivated by:** 2502.06703 + F-2 + F-9
**Test:** Score the 500 cached P11 K=1 trajectories (Qwen2.5-1.5B and 7B) with
Qwen2.5-Math-PRM-7B; aggregate by PRM-Min/Last/Avg; compute AUROC on
K=1-correctness; compare to F-2's 0.7731 / 0.876 and CoE-60 baseline. ~30 min
H100 + 10 min CPU.
**Requires:** H100 for one PRM forward pass over cached generations; no new
generation; Qwen2.5-Math-PRM-7B from HF (~15 GB).
**Would change:** If PRM AUROC ≥ 0.85 on 1.5B (Δ ≥ +0.08 over DoM), F-2 must
be re-framed as "free-lunch lower-bound" not "primary signal", and F-9's
"trajectory features are redundant" must be tightened to "low-capacity
trajectory features". If PRM AUROC ≤ 0.80, F-2 + F-9 stand and the prefill
DoM is genuinely the cheapest competitive correctness signal.
**Blocks:** H-322 (P11-FE404 motivated only if PRM beats DoM).

### H-322: PRM-guided BoN at matched compute beats prefill-DoM-gated selective prediction (F-8) on MATH-500
**Priority:** HIGH
**Motivated by:** 2502.06703 + F-8 + EXP-037
**Test:** On the same 500 MATH-500 problems used for F-8, run BoN with
Skywork-PRM-1.5B at N tuned to total compute matching F-8's avg K=2.5
(≈2–3 trajectories per problem). Report full-coverage accuracy. Compare to
F-8's 71.6% acc-on-answered at coverage 0.5 (utility-equivalent score 35.8%)
and against the natural baseline of K=1 unconditional accuracy 48.6%.
**Requires:** H100 (~2h for fresh BoN generation + PRM scoring) + cached F-8
correctness labels.
**Would change:** If full-coverage PRM-BoN ≥ 65% at matched compute, F-8's
"selective prediction is a useful operating regime on this model" claim must
be downgraded to "selective prediction is dominated by intra-problem
PRM-guided search at matched budget" — abstention loses to compute reallocation.
If PRM-BoN ≤ 50%, F-8 is robust and selective prediction remains a
defensible regime.
**Blocks:** H-19 reformulation (whether C_exact routing should use internal
or external verifiers).

---


### H-323: Learnable conformal thresholds substantially close the F-8 selective-prediction gap on prefill-DoM scores
**Priority:** HIGH
**Motivated by:** 2502.06884 + F-8
**Test:** Train CAP-style REINFORCE policy (πθ outputting Gaussian (α, β)) on cached prefill-DoM scores for Qwen-2.5-1.5B MATH-500. Compare AUARC at 90% coverage against F-8's 71.6% accuracy at 50% coverage. ~4h CPU on cached `pathway11_h100/prefill_gated_compute/results.json`.
**Requires:** Cached prefill-DoM scores (already on disk); CPU only; no new H100 time.
**Would change:** If CAP gains ≥10pp AUARC over fixed-threshold abstention on the same DoM scores, F-8's selective-prediction headline understates achievable performance and needs a learnable-threshold caveat. If gains are <2pp, F-8 is robust and CAP doesn't add value over the existing fixed-threshold operating point.
**Blocks:** nothing (P10-FE2b can proceed immediately on cached data).

### H-324: Softmax-confidence AUROC matches prefill-L19-DoM AUROC on Qwen-2.5-1.5B MATH-500
**Priority:** CRITICAL
**Motivated by:** 2502.06884 + F-2
**Test:** Extract `1 − p_predicted(answer)` from cached Qwen-2.5-1.5B MATH-500 logits (or re-run P11 H100 stage with logits saved). Compute AUROC under 1024-tok correctness labels. Compare to F-2's 0.7731. ~30min CPU if logits cached.
**Requires:** Cached or re-extractable answer logits from P11 H100 generations; CPU.
**Would change:** If softmax-AUROC ≥ 0.75, the residual-stream / DoM geometry story behind F-2 reduces to a softmax-equivalent claim and the project's headline narrative needs major revision. If softmax-AUROC ≤ 0.65, F-2's residual-stream advantage is real and quantified.
**Blocks:** F-2's "residual-stream geometry predicts correctness" framing depends on this test resolving in favour of DoM.

---


### H-325: Correctness in Qwen-2.5-1.5B is sparse and basis-aligned in L19 MLP neurons
**Priority:** HIGH
**Motivated by:** 2502.11096 + F-2
**Test:** Compute one-vs-rest mean-difference activation per MLP neuron
at L19 on cached `pathway11_h100/prefill` activations for Qwen-2.5-1.5B,
labeled by correct/incorrect. Score AUROC of `sum(top-K neurons)` for
K ∈ {1, 5, 10, 50, 200} against the L19 DoM AUROC of 0.7731.
Repeat on Qwen-2.5-7B prefill cache. ~4h CPU.
**Requires:** Cached `pathway11_h100/prefill_gated_compute/` MLP-intermediate
activations (already exist on disk per DATA_MANIFEST). CPU-only.
**Would change:** If AUROC > 0.78 for any K ≤ 50, F-2's "single direction"
framing is reframed as a continuous projection of a sparse mechanism;
selective-prediction (F-8) becomes basis-aligned thresholding (cheaper,
more interpretable, easier to ablate). If no K beats 0.7731, DoM is
genuinely the right continuous direction and MoTE-style sparse
localization does not transfer from MoE-routing to dense MLPs.
**Blocks:** H-326, H-327.

### H-326: Sparse L19-MLP-neuron ablation matches or exceeds per-position DoM steering on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2502.11096 + H-1 + F-2
**Test:** Identify top-10 differentially-active L19 MLP neurons for the
`incorrect` class via H-325. At inference on Qwen-2.5-1.5B, zero those
neurons via a forward-pass hook; re-run MATH-500 K=1. Pair with a
random-10-neuron control (per MoTE Figure 8). Run MT-Bench to verify
general-capability preservation. Compare ΔACC against H-1's per-position
DoM steering at matched compute. ~1d H100.
**Requires:** Qwen-2.5-1.5B inference, MATH-500 + MT-Bench. Output of H-325
to select neurons.
**Would change:** If sparse ablation ΔACC ≥ DoM-steering ΔACC at lower
compute, H-1's continuous bank-of-directions framing is operationally
inferior to discrete basis-aligned ablation; revisit H-1's design. If
DoM-steering wins, MoTE's sparse-localization pattern does not transfer
to *correctness* in dense models even if it transfers to *refusal* in MoE.
**Blocks:** depends on H-325.

### H-327: Prefill-DoM and final-token-DoM share basis-aligned MLP neuron support despite cos = 0.046
**Priority:** MEDIUM
**Motivated by:** 2502.11096 + F-3
**Test:** Compute differential-activation top-K MLP neuron lists at L19
for prefill and final-token activations separately; compute Jaccard
similarity for K ∈ {10, 50, 200}. ~3h CPU.
**Requires:** Cached prefill + final-token L19 MLP activations.
CPU-only.
**Would change:** If Jaccard > 0.5 for K = 50, F-3's interpretation of
prefill and final-token as "different subspaces" is reframed as a
continuous-projection artefact; the underlying sparse mechanism is
shared. If Jaccard < 0.2, F-3 strengthens — different sparse supports
back the orthogonality claim mechanistically.
**Blocks:** nothing.

---


### H-328: EKBM-aligned verbalized-confidence dominates prefill-DoM on Qwen2.5-1.5B-MATH-500 selective prediction
**Priority:** HIGH
**Motivated by:** 2503.02233 + F-8 + F-2
**Test:** Train Qwen2.5-1.5B-Instruct EKBM-style (1k self-sampled SFT + 2k DPO Post, α₁=0.25, α₂=0.75) on MATH-500 train split; evaluate sure-Accuracy at sure-Coverage 0.5; compare to F-8's 71.6%. Est time: 1 day H100.
**Requires:** H100 (SXM, ~24h budget); Qwen2.5-1.5B-Instruct weights; MATH-500 train/test split (~1k train, 500 test); GPT-4o or DeepSeek-R1-70B for refinement-stage CoT generation if extending to FE73.
**Would change:** Confirm → F-8 must be reframed as "internal-state-only baseline"; F-2's practical relevance drops. Reject (sure-Acc < 71.6% at coverage 0.5) → prefill-DoM selective prediction is genuinely additive over verbalized confidence; current P11 narrative survives.
**Blocks:** P11-FE414 (head-to-head gating), and any decision to deprioritize F-8 in publication.

### H-329: Self-resampling-consistency reproduces prefill-DoM AUROC without any training or hidden-state extraction
**Priority:** HIGH
**Motivated by:** 2503.02233 + F-4 + F-8
**Test:** On Qwen2.5-1.5B / MATH-500 K=1 outputs, sample i ∈ {2, 4, 8} additional generations per problem; tag sure/unsure by agreement rate; compute AUROC of consistency-flag against ground truth. Compare to prefill-DoM AUROC 0.7731 and F-4 collapse-strength metric. Est time: 30min H100 + 10min CPU.
**Requires:** Qwen2.5-1.5B inference (any GPU); ground-truth answers (already in MATH-500); existing K=1 cache.
**Would change:** Confirm (consistency-AUROC ≥ 0.74) → F-4's asymmetric collapse is a geometric shadow of behavioral consistency, not an independent signal; the geometric framing of P11 collapses to "consistency restated geometrically". Reject → geometric signal is genuinely additional and F-4 stands.
**Blocks:** any claim that F-2/F-4 are independent geometric findings rather than consistency-correlates.

### H-330: EKBM SFT/DPO alignment substantially degrades L19 prefill-DoM AUROC on Qwen2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2503.02233 + F-2 + H-10
**Test:** After H-328 / FE70, re-extract L19 prefill activations from EKBM-aligned weights on MATH-500; recompute prefill-DoM AUROC and compare to base 0.7731. Est time: 20min CPU.
**Requires:** CPU-only after FE70 produces weights; uses existing extraction pipeline.
**Would change:** Confirm (AUROC drops below 0.65) → F-2's signal is conditional on un-aligned base models, making the claim narrower than currently stated; supports H-10 (breathing changes after fine-tuning). Reject → F-2 is robust to behavioral alignment, strengthening the geometric-signal-is-fundamental story.
**Blocks:** any cross-model-checkpoint claim about F-2 stability without first running this control.

---


### H-331: A 3-layer MLP on prefill L19 activations beats linear DoM (F-2) on K=1 MATH-500 correctness
**Priority:** HIGH
**Motivated by:** 2503.10602 + F-2 + F-9
**Test:** Train MLP[256→128→1, BCE] 5-fold OOF on cached `pathway11_h100/prefill/L19_*.npz` against K=1 correctness labels; report AUROC and LR+ at FPR={0.01, 0.05, 0.10}. Compare against linear DoM 0.7731. ~30 min CPU.
**Requires:** CPU only, all data cached.
**Would change:** Confirm → F-2 reframes as "linear-probe lower bound on a non-linear correctness signal," and the F-9 "CoE-60 redundant with single-layer DoM" claim should be re-litigated against MLP-on-single-layer (CoE-60 may also be redundant with MLP). Reject (gap < 1.5 AUROC pts) → DoM is essentially optimal at this layer; the linear assumption holds and we can stop chasing non-linear probes.
**Blocks:** H-332 partial — if MLP wins, the position-shift question (H-332) should re-run with MLP rather than DoM.

### H-332: F-3 orthogonality (cos=0.046) is a positional artifact — pre-final-token DoM is *not* orthogonal to prefill DoM
**Priority:** HIGH
**Motivated by:** 2503.10602 (§3.1 previous-token framing) + F-3
**Test:** From cached per-token L19 activations, extract the activation at position N-1 (the token whose next-token prediction emits the answer). Recompute final-token DoM there via mass-mean. Report cos(prefill_DoM, pre-final_DoM). ~30 min CPU.
**Requires:** CPU only; needs cached per-token activations (verify in DATA_MANIFEST).
**Would change:** Confirm (cos > 0.3) → F-3 collapses to a one-position-off bug, F-3's "two orthogonal mechanisms" interpretation is wrong, E1 fixed-vector steering is revived (retires H-1 and H-17). Reject (cos still < 0.1) → F-3 is robust to position shift, the orthogonality is mechanistic rather than positional, H-1 and H-17 remain the right paths.
**Blocks:** H-1 (decision pivot), H-17 (also probes whether F-3 is mechanical vs. computed).

### H-333: ComnHallu-style subspace alignment recovers a "generic prefill DoM" between Qwen-1.5B and Qwen-7B
**Priority:** MEDIUM
**Motivated by:** 2503.10602 (§3.3 ComnHallu) + H-8
**Test:** Independent eigendecomposition of 1.5B and 7B prefill L19 activations (top-d'={16, 32, 64, 128} eigenvectors); compute alignment matrix M = K_S^T K_T; project 1.5B activations into aligned 7B subspace. Train DoM on aligned-1.5B, evaluate AUROC on aligned-7B. Compare against unaligned cross-scale transfer baseline. ~1 h CPU.
**Requires:** CPU only, both Qwen prefill caches.
**Would change:** Confirm (aligned cross-scale AUROC > 0.65, materially above unaligned) → "generic truthful direction" generalizes to cross-scale within a family; H-8 generalizes from cross-checkpoint to cross-scale; cheap small-model probes can substitute for expensive large-model probes if subspace-aligned. Reject → cross-scale truthfulness directions are *not* aligned by linear subspace projection alone, suggesting the F-2 signal is more architecture-/training-specific than the LVLM literature suggests.
**Blocks:** nothing immediate; informs H-11 (big-to-small distillation) framing.

---


### H-334: Math-correctness signal localizes earlier than L19 in Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2503.12730 (TinySQL §6 two-phase intent→grounding→assembly logit-lens result) + F-2 (L19 prefill DoM AUROC=0.7731).
**Test:** Layerwise logit-lens trajectory and per-layer prefill DoM AUROC sweep on cached Stage 2 NPZs for all 28 Qwen-2.5-1.5B layers. P11-FE419 + P11-FE420.
**Requires:** ~1.5h CPU on cached data. Zero dollar.
**Would change:** If AUROC peaks well before L19, F-2's narrative reframes from "L19 is THE probe site" to "L19 is one of several mid-stack grounding-phase layers" — and points to depth-fraction normalization as the cross-arch invariant for F-1. Confirm leaves F-2 as is.
**Blocks:** P11-FE421 framing — if correctness localizes at L8, the prefill-vs-final orthogonality story is layer-staged, not circuit-distinct.

### H-335: Prefill and final-token DoMs are projections of one fragmented signal, not two distinct circuits
**Priority:** HIGH
**Motivated by:** 2503.12730 §7.2 explicit framing that EAP yields fragmented mechanisms scattered across layers in larger models + F-3 cos(prefill, final)=0.046.
**Test:** Train a probe on the concatenation `[prefill_L19, final_L19]`; compare AUROC vs each component alone. If concat ≈ max → redundant → fragmented-signal story; if concat ≫ max → genuinely two circuits. P11-FE421.
**Requires:** ~20min CPU on cached prefill + final-token NPZs. Zero dollar.
**Would change:** If concat ≈ prefill alone, F-3's "orthogonal circuits" framing collapses to "fragmented projections of one signal" — F-3 needs rewriting. If concat ≫ both, F-3 strengthens to "demonstrably non-redundant signals" and H-13's two-circuit search becomes load-bearing.
**Blocks:** H-13 — if H-335 lands as redundant, H-13 returns null by construction.

### H-336: F-8 prefill selective-prediction is template-bound, not difficulty-bound
**Priority:** HIGH
**Motivated by:** 2503.12730 §4.2 (BM1 leverages surface-template shortcuts: schema for tables, instruction for fields) + F-8 (71.6% accuracy at 50% coverage).
**Test:** Generate 100 surface-paraphrased MATH-500 prompts (semantic-preserving, varied word order / synonyms). Re-extract prefill L19 activations. Recompute prefill DoM AUROC. If AUROC drops by >0.05 vs the canonical 0.7731, F-8's signal lives in template rather than difficulty. P11-FE423.
**Requires:** 2h H100 + 30min CPU. ~$4 in compute.
**Would change:** If AUROC is paraphrase-stable, F-8 becomes a defensible "model knows when it can't" claim. If AUROC collapses, F-8 reframes to "model knows the question shape" and selective-prediction becomes a classifier on prompt-style, not difficulty.
**Blocks:** any inference-time deployment of F-8-based gating — would need paraphrase-robust prefill features first.

### H-337: F-7 D-bucket signature is outlier-driven, not collective
**Priority:** MEDIUM
**Motivated by:** 2503.12730 §7.1 (significant within-class activation-patching variance even on same task type) + F-7 (36-problem D-bucket signature).
**Test:** Leave-one-out recompute of the F-7 collective metric across the 36 D-bucket problems. Plot metric distribution. If 2–3 problems dominate, F-7 is mean-shift not collective. P11-FE424.
**Requires:** ~30min CPU on cached F-7 metric arrays.
**Would change:** Confirm leaves F-7 as is. Reject reframes F-7 from "D-bucket has a distinctive collective signature" to "a small number of D-bucket problems have unusual geometry, but the bucket as a whole does not."
**Blocks:** any causal-mechanism story for F-7.

---


### H-338: Single-head L19 DoM beats single-layer L19 DoM at correctness prediction
**Priority:** HIGH
**Motivated by:** 2503.14130 + F-2
**Test:** Re-extract Qwen-2.5-1.5B prefill activations with per-head granularity on MATH-500, compute per-head DoM probe AUROC for each of L19's heads, compare best single-head AUROC against F-2's 0.7731. ~4h H100 + 1h CPU.
**Requires:** H100 access for re-extraction (1.5B prefill activations are cached at residual-stream level only); cached MATH-500 1024-tok problems.
**Would change:** Confirms → F-2 headline number is suboptimal; selective prediction (F-8) should be re-baselined on single-head DoM; CoE-60 redundancy claim (F-9) needs to compare against single-head, not single-layer. Rejects → layer-pooling captures the signal and head-level decomposition adds nothing, strengthening F-2 and F-3's "residual-stream level" framing.
**Blocks:** H-339, P10-FE26, P10-FE27.

### H-339: Per-head intervention on MATH-500 collapses recall before improving accuracy
**Priority:** MEDIUM
**Motivated by:** 2503.14130 (precision 1.00 at recall 0.07 with α=32.4, K=6) + H-1
**Test:** Apply Darm's per-head α tuning to the head identified in H-338 (best-AUROC L19 head), sweep α ∈ {1,5,10,20,40} × T ∈ {0.1, 0.5, 1.0}, measure MATH-500 precision/recall over 20 seeds per cell. ~1 day H100.
**Requires:** Outcome of H-338 (need the best single head); cached MATH-500.
**Would change:** Confirms → H-1 must be reformulated as "precision-mode gating" not "accuracy-improving steering"; per-position DoM steering on MATH-500 will hit the same recall wall. Rejects → MATH-500 accuracy responds monotonically to per-head α without recall collapse, validating H-1's original framing and suggesting requirement-verification recall collapse is a domain artefact.
**Blocks:** P11-FE425.

---


### H-340: Token-embedding-layer singular structure explains a substantial fraction of prefill DoM AUROC
**Priority:** HIGH
**Motivated by:** 2504.01002 + F-2
**Test:** Run paper's Algorithm 1 on Qwen-2.5-1.5B vocab (FE100), compute
per-prompt singular-token-density on MATH-500, correlate with prefill L19
DoM scores from `pathway11_h100/prefill_gated_compute/results.json`.
~1h CPU total.
**Requires:** CPU; Qwen-2.5-1.5B embedding matrix (already loaded);
cached prefill DoM scores (already on disk).
**Would change:** Confirm (r > 0.5): F-2's AUROC is largely explained by
input geometry; H-5's "familiarity vs decomposability" framing collapses
to "input-token irregularity"; reframes F-2/F-3 as input-inherited vs
computed signals. Reject (r < 0.2): the prefill direction is genuinely a
mid-network reasoning signal independent of input singularities, F-2
hardens.
**Blocks:** any future "what is the prefill direction reading?" mechanistic
work — this should run first.

### H-341: An L0 (embedding-lookup-only) DoM probe achieves AUROC ≥ 0.65 on MATH-500 1024-tok
**Priority:** HIGH
**Motivated by:** 2504.01002 + F-2 + F-3
**Test:** Mean-pool Qwen-2.5-1.5B input embeddings over each MATH-500
prompt stem; compute DoM between correct (243/500) and incorrect (257/500);
5-fold OOF AUROC. ~30min CPU.
**Requires:** CPU; embedding matrix; MATH-500 prompts; correctness labels
from `pathway11_h100/prefill_gated_compute/results.json`.
**Would change:** Confirm: F-2's L19 specificity is inflated; substantial
correctness signal exists pre-computation. Reject: L19 prefill genuinely
adds something that L0 lookup doesn't have, F-2 hardens. Either result
deconfounds F-2's interpretation.
**Blocks:** clean mechanistic interpretation of F-2 — this is the cheapest
deconfounding control available.

### H-342: D-bucket prompts (F-7) cluster in the high-singular-token-density tertile
**Priority:** MEDIUM
**Motivated by:** 2504.01002 + F-7
**Test:** Per-prompt singular-token density (from FE100), partition
MATH-500 into tertiles, count D-bucket vs non-D-bucket per tertile,
chi-square. ~10min CPU once FE100 lands.
**Requires:** CPU; FE100 outputs; D-bucket labels from cached results.
**Would change:** Confirm: F-7's "collective geometric signature" framing
is unnecessary — input-token irregularity at the embedding lookup
explains the pattern. Reject: F-7's trajectory-level framing survives
input-geometry control, hardens.
**Blocks:** F-7 publication framing.

---


### H-343: F-2's 0.7731 AUROC is substrate-limited, not method-limited
**Priority:** HIGH
**Motivated by:** 2504.05419 + F-2
**Test:** Run the F-2 prefill DoM extraction pipeline on R1-Distill-Qwen-1.5B and Qwen-2.5-1.5B base, on the same MATH-500 problems, with identical L19-prefill protocol and OOF 5-fold logistic. Report ΔAUROC.
**Requires:** H100 day for R1-Distill-Qwen-1.5B prefill extraction on MATH-500 (~$15); 30min CPU for probes.
**Would change:** If R1-Distill-Qwen-1.5B AUROC ≥ 0.85, the F-2 narrative shifts from "geometry of prefill encodes correctness" to "long-CoT supervised training installs the correctness direction in prefill geometry." If equivalent (Δ < 0.03), F-2's framing is robust to substrate.
**Blocks:** H-344, P11-FE432.

### H-344: Prefill→final DoM rotation is monotone-smooth, not orthogonal
**Priority:** HIGH
**Motivated by:** 2504.05419 §4.5 + F-3
**Test:** Project L19 activations from cached pathway11_h100 NPZs onto prefill_DoM and final_DoM at chunk positions {10,25,50,75,90,95,100}%. Plot cos(probe_at_t, prefill_DoM) and cos(probe_at_t, final_DoM) as a function of t. Fit a smooth rotation model.
**Requires:** 2hr CPU; cached all-token L19 activations (regenerate ~1 day H100 if not cached).
**Would change:** If the curve is monotone, F-3's "0.046 cosine = orthogonal information channels" interpretation is wrong; prefill and final are endpoints of one rotating direction. If discontinuous (sharp jump after the answer marker), F-3's orthogonality claim is structural, not artifact.
**Blocks:** nothing.

### H-345: Probe-confidence early-exit beats DoM-ranking selective prediction at matched compute
**Priority:** HIGH
**Motivated by:** 2504.05419 §5.2 + F-8
**Test:** Implement Zhang-style threshold early-exit using prefill_DoM as on-policy probe at chunk boundaries (keyword-segmented). Sweep Thr ∈ {0.7, 0.8, 0.85, 0.9, 0.95}. Report token reduction × accuracy frontier on MATH-500 × Qwen-2.5-1.5B. Compare to F-8 (71.6% at coverage 0.5, K=2.5 avg) and Zhang (24% token reduction at 88.2% accuracy on R1-Distill-Llama-8B).
**Requires:** H100 day for inference-time integration; existing prefill_DoM probe.
**Would change:** If matched-compute frontier dominates F-8, selective prediction should be reframed as on-policy early-exit. If matched accuracy needs Platt/temperature calibration, H-343 calibration audit is the gate.
**Blocks:** nothing.

### H-346: Linear probe (d=0) suffices for F-2 — no MLP gain
**Priority:** LOW
**Motivated by:** 2504.05419 Appendix A.3 (most grid-search optima at d=0)
**Test:** Re-fit F-2 logistic with hidden size grid {0, 16, 32, 64} on the existing 5-fold OOF setup. Compare val accuracy vs d.
**Requires:** 30 min CPU, existing F-2 OOF data.
**Would change:** Confirms or refutes F-9's "linear sufficient" sub-claim more directly than the CoE comparison did.
**Blocks:** nothing.

---


### H-347: F-2's 0.7731 AUROC is substrate-limited, not method-limited
**Priority:** HIGH
**Motivated by:** 2504.05419 + F-2
**Test:** Run the F-2 prefill DoM extraction pipeline on R1-Distill-Qwen-1.5B and Qwen-2.5-1.5B base, on the same MATH-500 problems, with identical L19-prefill protocol and OOF 5-fold logistic. Report ΔAUROC.
**Requires:** H100 day for R1-Distill-Qwen-1.5B prefill extraction on MATH-500 (~$15); 30min CPU for probes.
**Would change:** If R1-Distill-Qwen-1.5B AUROC ≥ 0.85, the F-2 narrative shifts from "geometry of prefill encodes correctness" to "long-CoT supervised training installs the correctness direction in prefill geometry." If equivalent (Δ < 0.03), F-2's framing is robust to substrate.
**Blocks:** H-348, P11-FE438.

### H-348: Prefill→final DoM rotation is monotone-smooth, not orthogonal
**Priority:** HIGH
**Motivated by:** 2504.05419 §4.5 + F-3
**Test:** Project L19 activations from cached pathway11_h100 NPZs onto prefill_DoM and final_DoM at chunk positions {10,25,50,75,90,95,100}%. Plot cos(probe_at_t, prefill_DoM) and cos(probe_at_t, final_DoM) as a function of t. Fit a smooth rotation model.
**Requires:** 2hr CPU; cached all-token L19 activations (regenerate ~1 day H100 if not cached).
**Would change:** If the curve is monotone, F-3's "0.046 cosine = orthogonal information channels" interpretation is wrong; prefill and final are endpoints of one rotating direction. If discontinuous (sharp jump after the answer marker), F-3's orthogonality claim is structural, not artifact.
**Blocks:** nothing.

### H-349: Probe-confidence early-exit beats DoM-ranking selective prediction at matched compute
**Priority:** HIGH
**Motivated by:** 2504.05419 §5.2 + F-8
**Test:** Implement Zhang-style threshold early-exit using prefill_DoM as on-policy probe at chunk boundaries (keyword-segmented). Sweep Thr ∈ {0.7, 0.8, 0.85, 0.9, 0.95}. Report token reduction × accuracy frontier on MATH-500 × Qwen-2.5-1.5B. Compare to F-8 (71.6% at coverage 0.5, K=2.5 avg) and Zhang (24% token reduction at 88.2% accuracy on R1-Distill-Llama-8B).
**Requires:** H100 day for inference-time integration; existing prefill_DoM probe.
**Would change:** If matched-compute frontier dominates F-8, selective prediction should be reframed as on-policy early-exit. If matched accuracy needs Platt/temperature calibration, H-347 calibration audit is the gate.
**Blocks:** nothing.

### H-350: Linear probe (d=0) suffices for F-2 — no MLP gain
**Priority:** LOW
**Motivated by:** 2504.05419 Appendix A.3 (most grid-search optima at d=0)
**Test:** Re-fit F-2 logistic with hidden size grid {0, 16, 32, 64} on the existing 5-fold OOF setup. Compare val accuracy vs d.
**Requires:** 30 min CPU, existing F-2 OOF data.
**Would change:** Confirms or refutes F-9's "linear sufficient" sub-claim more directly than the CoE comparison did.
**Blocks:** nothing.

---


### H-351: SEAL thought-type axis subsumes the DoM correctness axis at L19
**Priority:** HIGH
**Motivated by:** 2504.07986 + EXP-037 (F-2 prefill AUROC 0.7731)
**Test:** Apply SEAL's keyword classifier (Appendix B of 2504.07986) to cached pathway11_h100 K=1 MATH-500 generations on Qwen 2.5 1.5B. Extract residuals at "\n\n" boundary tokens at L19 from cached Stage 2 NPZs. Compute H_E − H_RT thought-type axis. Check (i) cos(thought-type axis, prefill DoM), (ii) cos drift across positions, (iii) AUROC of "fraction of R+T thoughts" alone for correctness, (iv) residual prefill-DoM AUROC after orthogonalizing the thought-type axis out of L19 features. ~2 h CPU.
**Requires:** CPU only; cached pathway11_h100 NPZs and MATH-500 generations. SEAL appendix B keyword list.
**Would change:** If thought-type AUROC ≥ 0.70 alone and residual DoM AUROC drops below 0.65, F-2's headline is a verbosity confound and the project's correctness axis interpretation needs revision. If thought-type AUROC is much lower or orthogonal to DoM, F-2 is a genuine independent signal and SEAL's axis can be added as a complementary feature for selective prediction (F-8).
**Blocks:** H-1 reframing — if thought-type axis dominates, H-1 should be redirected from DoM steering to thought-type steering before spending the H100 budget.

### H-352: SEAL-style steering closes the 1.5B → 7B accuracy gap on cached MATH-500
**Priority:** MEDIUM
**Motivated by:** 2504.07986 (+11pp on R1-Distill-Qwen-1.5B → 78.0%) + headline number (base Qwen 2.5 1.5B at 48.6%)
**Test:** Replicate SEAL on base Qwen 2.5 1.5B with default α=1.0 at L20 (and L19 ablation) on the same 500-problem MATH-500. Compare to base 48.6%. ~30 min H100 + 1h CPU for steering vector extraction. If SEAL closes ≥ 5pp of the 24.6pp gap to 7B (73.2%), latent-space calibration is a free 1.5B-class lift; project shifts focus from "predict correctness" to "calibrate at inference."
**Requires:** H100 pod, ~30 min generation; MATH train subset (~1000 samples) for vector extraction (Hendrycks 2021); SEAL keyword list.
**Would change:** Quantifies SEAL's portability from R1-Distill to base. If portability is confirmed, the project's ship narrative changes from "0.77 selective predictor" to "+Xpp accuracy via free latent-space calibration." If SEAL fails on base Qwen, R1-Distill specificity (long-CoT distilled training) is necessary for the thought-type axis to exist.
**Blocks:** H-1 if SEAL ≥ DoM-steering at matched compute.

### H-353: Asymmetric collapse (F-4) is a thought-pattern shadow, not a correctness shadow
**Priority:** MEDIUM
**Motivated by:** 2504.07986 (more reflection/transition in incorrect responses) + F-4 (correct trajectories collapse harder than incorrect at final token)
**Test:** On cached pathway11_h100 final-token L19 PR per problem, regress PR ∝ (count of execution thoughts) + (count of R+T thoughts) + correctness; check if correctness term retains significance after thought-count covariates are added. 1h CPU.
**Requires:** CPU only; cached final-token activations and SEAL-classified thought counts (depends on H-351 keyword extraction).
**Would change:** If correctness term loses significance, F-4's "asymmetric collapse" is mechanistically explained by "incorrect = more R+T thoughts = more residual variance" rather than a correct-specific collapse phenomenon. If correctness retains significance after partialling out, F-4 is independent of thought structure.
**Blocks:** H-15 (length-band PR control) becomes redundant with this if H-353 confirms.

---


### H-354: A vanilla "Are you sure?" linguistic-confidence baseline on a frozen Qwen-2.5-1.5B matches or beats prefill L19 DoM AUROC for MATH-500 K=1 selective prediction
**Priority:** HIGH
**Motivated by:** 2504.21773 + F-2 + F-8
**Test:** On cached MATH-500 K=1 generations from pathway11_h100, prompt base
Qwen-2.5-1.5B with "Are you sure you accurately answered the question?" after
each answer. Read confidence as P("sure") − P("unsure") at response token.
Compute AP and linear AUROC against K=1 correctness. Compare against F-2's
0.7731 prefill-DoM AUROC. Time: 4h H100 inference, no fine-tune.
**Requires:** H100 inference pod, cached MATH-500 K=1 generations from
pathway11_h100, base Qwen-2.5-1.5B.
**Would change:** If vanilla verbalized confidence ≥ 0.7731 AUROC, F-2's
"prefill-DoM is the strongest correctness signal" claim is downgraded —
the model already has access via verbalization. If << 0.7731, F-2 is
corroborated against a strong cheap prior.
**Blocks:** H-355.

### H-355: MAC-Tuned Qwen-2.5-1.5B beats prefill-DoM-based selective prediction at coverage=0.5 on MATH-500
**Priority:** HIGH
**Motivated by:** 2504.21773 + F-8 + H-19
**Test:** Two-step LoRA SFT on Qwen-2.5-1.5B per MAC-Tuning recipe (r=8,
alpha=32, lr=1e-5, 3 epochs): stage 1 P(A|Q) on MATH-500 train, stage 2
P(C|Q,A) using K=1 correctness from pathway11_h100 as confidence label.
Eval at coverage=0.5 against F-8's 71.6%. Time: 1 day H100.
**Requires:** H100 training pod, MATH-500 train/test split, K=1 correctness
labels from pathway11_h100.
**Would change:** If MAC-Tuning ≥ 73% accuracy@coverage=0.5, F-8 is
matched-or-beaten by an SFT alternative — and H-19's verification-routing
ROI drops sharply (we don't need a separate verification pass if the model
itself has been trained to verbalize uncertainty). If << 71.6%, F-8's
prefill-DoM gating is corroborated as harder to beat than expected.
**Blocks:** H-356.

### H-356: After MAC-Tuning, the post-answer-token L19 DoM AUROC exceeds the prefill L19 DoM AUROC, reversing F-2's ordering
**Priority:** MEDIUM
**Motivated by:** 2504.21773 + F-2
**Test:** Extract L19 activations at the confidence-token position from a
MAC-Tuned Qwen-2.5-1.5B; compute linear AUROC against K=1 correctness.
Compare against F-2's 0.7731 (prefill, base) and 0.7186 (final-token, base).
Time: 4h CPU on cached MAC-Tuned activations.
**Requires:** MAC-Tuned 1.5B (depends on H-355 first), MATH-500 K=1 holdout
correctness labels.
**Would change:** If post-answer L19 AUROC ≥ 0.78 on MAC-Tuned model, F-2's
"prefill > final-token" ordering is a property of the *frozen* model, not
of the residual stream itself — fine-tuning relocates the correctness signal
forward in the sequence. Strong methodological implication for H-13
(head-level attribution of prefill vs final-token circuits).
**Blocks:** nothing.

### H-357: Multi-problem prompts (n=3 concatenated MATH-500 problems) preserve prefill L19 DoM AUROC ≥ 0.70 per-problem
**Priority:** MEDIUM
**Motivated by:** 2504.21773 + F-2 + F-5
**Test:** Concatenate MATH-500 problems into n=3 batches, re-run Qwen-2.5-1.5B
forward pass, extract prefill L19 activations per sub-problem boundary,
project onto the frozen single-problem DoM direction, compute per-problem
AUROC against K=1 correctness. Time: 1 day H100.
**Requires:** H100 inference pod, MATH-500 problems, frozen DoM direction
from pathway11_h100/prefill_gated_compute.
**Would change:** If per-problem AUROC drops to ≤ 0.65, F-1/F-2/F-5 are
single-problem-regime-specific — a restriction we should explicitly flag.
If AUROC ≥ 0.70, F-2 generalizes across problem composition, strengthening
its claim against a new attack surface from MAC-Tuning's regime argument.
**Blocks:** nothing.

---


### H-358: Prefill L19 DoM AUROC is mostly explained by predicted output length
**Priority:** HIGH
**Motivated by:** 2505.00127 + F-2
**Test:** P11-FE447 + P11-FE448. (a) Length-only AUROC on MATH-500. (b) Length-prediction R² from L19 prefill activations. (c) Residual DoM AUROC after partialing out predicted length. ~1h CPU on cached NPZs.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/results.json` and L19 prefill NPZs.
**Would change:** Confirm → F-2 demoted from "geometric correctness direction" to "covert length-prior probe"; F-9 strengthened (length is the floor, not L19 DoM); rewrite headline narrative. Reject → F-2 survives the most threatening covariate; H-5 (familiarity vs decomposability) becomes the next test.
**Blocks:** Any publication framing F-2 as a hidden-state result.

### H-359: D-bucket overlaps with underthink-hard tier (length-failure, not geometric failure)
**Priority:** MEDIUM
**Motivated by:** 2505.00127 + F-7
**Test:** P11-FE449. Stratify MATH-500 into Easy/Medium/Hard via K=8 correctness rate. Cross-tabulate with D-bucket membership. Compute mean tokens per (tier × bucket) cell.
**Requires:** CPU only; cached K=8 outputs.
**Would change:** Confirm → F-7's D-bucket reframed as a length/reasoning-extension failure rather than a distinctive geometric signature; weakens F-7 novelty. Reject → D-bucket is orthogonal to length-tier difficulty, strengthening the geometric-signature claim.
**Blocks:** nothing (independent test).

---


### H-360: Decoder-regression LoRA fine-tune dominates prefill DoM as a MATH-500 correctness predictor
**Priority:** HIGH
**Motivated by:** 2505.01595 + F-2 + F-8
**Test:** Fine-tune Qwen2.5-1.5B with bin-token decoder-regression (Wang 2505.01595 recipe) on MATH-500 (problem, K=1 generation, correctness) using 5-fold CV. Compare OOF AUROC against prefill L19 DoM (0.7731) and final-token L19 DoM (0.7186). Add their rank-consistency loss with D-bucket vs A-bucket pairs.
**Requires:** 1d H100 LoRA + 1h CPU; uses cached MATH-500 1.5B generations in `pathway11_h100/prefill_gated_compute/`.
**Would change:** If AUROC > 0.7731 by ≥3pp, F-2 is reframed: prefill DoM is the strongest *probe* but loses to direct fine-tuning. If AUROC ≈ 0.7731, F-2 holds and we report parity (geometric content matches what fine-tuning extracts). If AUROC < 0.7731, F-2 is strengthened (geometric probing beats the strongest external supervised competitor).
**Blocks:** nothing.

### H-361: D-bucket is co-explained by K=8 sampling-disagreement (no geometric signature needed)
**Priority:** HIGH
**Motivated by:** 2505.01595 + F-7
**Test:** Compute per-problem mode-fraction and answer-set entropy across K=8 samples on MATH-500 1.5B. Build a label-free disagreement-filter classifier (mode-fraction < threshold ⇒ "unstable") and measure D-bucket recovery AUROC. Compare against F-7's geometric collective score AUROC for the same target.
**Requires:** 20 min CPU on cached K=8 outputs (assuming cached; otherwise 30 min H100 to regenerate).
**Would change:** If disagreement-filter AUROC ≥ geometric-signature AUROC, F-7's "distinctive geometric signature" claim is downgraded — D-bucket is a sampling-disagreement tail, not a structurally unique geometric subset. If geometric > disagreement by ≥5pp, F-7 is robustly distinct.
**Blocks:** nothing (but resolves the F-7 framing question).

### H-362: Verbalized-token-probability probe (Tian-style) matches prefill DoM AUROC on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2505.01595 + F-2 + H-12
**Test:** Run Tian et al. 2023's true/false-token probe on Qwen2.5-1.5B for MATH-500 K=1 correctness; calibrate via Thermometer-style temperature scaling on a held-out fold. Compare AUROC to prefill L19 DoM (0.7731).
**Requires:** 2h H100 (forward passes + held-out calibration).
**Would change:** If Probe AUROC ≈ DoM AUROC (within 2pp), the prefill direction adds no information beyond verbalized-token confidence — F-2's geometric framing becomes optional. If Probe AUROC < DoM by ≥5pp, F-2 holds: prefill geometry is genuinely richer than what the LM can verbalize.
**Blocks:** H-12 framing (labels-free vs supervised confidence).

---


### H-363: A label-free output-distribution probe (`"So, I'm" + sum-of-confidence-tokens`) matches or exceeds prefill L19 DoM as a correctness predictor on Qwen-2.5-1.5B MATH-500
**Priority:** CRITICAL
**Motivated by:** 2505.04881 + F-2
**Test:** Run FE181 — forward-pass through cached MATH-500 generations
appending probing prompt at PANL, read out four token probabilities,
compute AUROC. Compare against F-2's 0.7731.
**Requires:** Qwen-2.5-1.5B on H100 (or 4090 — small forward), cached
MATH-500 prompts + correctness labels in
`pathway11_h100/prefill_gated_compute/results.json`. ~30 min H100.
**Would change:** On confirm — F-2's "L19 DoM is the canonical correctness
direction" claim must be rewritten as "output-aware probes carry the
correctness signal, the residual-stream version at L19 prefill is one
instantiation". F-9's "single dimension is sufficient" extends to "the
dimension lives in the unembedding". On reject (AUROC < 0.7) — we have a
clean refutation of the most natural cheap-probe baseline, strengthening
the F-2 claim that the signal is not redundant with output-distribution
information.
**Blocks:** H-12 (semantic-entropy probes) — if H-363 confirms, H-12
should be reframed around `"So, I'm"`-style probes rather than SEPs.

### H-364: ConCISE-Decoding-style risk-compute curve Pareto-dominates F-8's prefill-DoM-gated risk-coverage curve at matched token budget on Qwen-2.5-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2505.04881 + F-8
**Test:** Run FE182 — sweep te ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8} on the
output-probe scores from H-363's run, plot accuracy-vs-tokens, overlay
against F-8's risk-coverage frontier.
**Requires:** Output of FE181 (per-problem c_hat). 5 min CPU.
**Would change:** On confirm — F-8's selective-prediction story refactors
around the trade-off between *output-distribution probing* and
*residual-stream probing*; the prefill-DoM gate becomes one point on a
broader frontier rather than the canonical operating point. On reject —
F-8 holds and prefill-DoM has a unique advantage we should characterize.
**Blocks:** nothing.

### H-365: Asymmetric collapse (F-4) is anchored at the wrong token; the gap grows or inverts at FAS+last-reflection vs FAS itself
**Priority:** HIGH
**Motivated by:** 2505.04881 + F-4
**Test:** On the subset of K=1 cached MATH-500 generations that contain
≥1 post-answer reflection (greedy decoding so reflection presence is
deterministic per problem), compute participation ratio at FAS, FAS+1
(start of first reflection), and FAS+last (end of last reflection),
separately for correct and incorrect groups. Compare PR(correct) -
PR(incorrect) at the three positions.
**Requires:** Cached residual streams from `pathway11_h100/` Stage 2 NPZs.
Need to verify those NPZs cover the FAS-to-end window. ~1h CPU if cache
exists; +H100 day to re-extract if not.
**Would change:** On confirm (gap grows post-reflection) — F-4's "final
token" framing is ill-anchored; the asymmetric-collapse signal lives in
the post-FAS reflection tokens, which means F-4 should be re-stated about
the *settled-confidence* token, not the answer-emission token. On reject
(gap is largest at FAS) — F-4's anchor is correct and the post-FAS
reflections are washing out the signal.
**Blocks:** H-15 (length-band PR control) — H-365's result determines
whether H-15's PR measurement should anchor on FAS or FAS+last.

---


### H-366: Distillation transfers correctness behavior but not the prefill DoM direction
**Priority:** HIGH
**Motivated by:** 2505.15442 + H-11 + F-2
**Test:** RevKD-distill Qwen-2.5-7B → Qwen-2.5-1.5B on Math10K (LoRA r=8, α=16, τ=2, max-length=1024), re-extract L19 prefill activations on MATH-500, fit DoM, compare against SFT-1.5B's L19 DoM. Three quantitative outcomes: (a) ΔMATH-500 accuracy from 48.6% baseline, (b) ΔAUROC from 0.7731 baseline, (c) cos(distilled_DoM, original_7B_DoM).
**Requires:** H100 SXM, ~1 day distillation + 4h re-extraction. Cached 7B and 1.5B prefill NPZs from pathway11_h100. Math10K from HF.
**Would change:** If accuracy rises ≥5% but cos(distilled_DoM, 7B_DoM) stays <0.3, F-2 is reframed as a *behavioral* signature (any model that's correct on these problems develops *some* prefill direction), not a *mechanistic* signature transferable across checkpoints. If both move together, H-11 is on track and the 7B prefill direction is a real distillation target.
**Blocks:** H-11 (KD-distillation experiment), H-8 (prefill stability across training)

### H-367: Prefill direction stability under fine-tuning is testable cheaply via SFT-vs-RevKD-vs-base on the same architecture
**Priority:** MEDIUM
**Motivated by:** 2505.15442 (self-fidelity differs between SFT and KD versions of same architecture) + F-2
**Test:** Extract L19 prefill activations on MATH-500 for three Qwen-2.5-1.5B checkpoints — the released base, our SFT-on-Math10K version, and our RevKD-distilled version (after H-366 is run). Compute pairwise cos(DoM_a, DoM_b) and AUROC of each checkpoint's DoM transferred onto each other checkpoint's activations.
**Requires:** ~30 min CPU once H-366 produces the third checkpoint.
**Would change:** Establishes whether the prefill direction is "trained in" by SFT and persists, or whether each fine-tuning regime produces its own. Generalizes H-8 (training-checkpoint stability) to fine-tuning regimes specifically.
**Blocks:** H-8

---


### H-368: σ_max(E_embed) / σ_max(W_layer) ratio predicts per-layer breathing amplitude in Qwen-2.5
**Priority:** MEDIUM
**Motivated by:** 2505.15624 + F-1
**Test:** Compute σ_max for embedding and each projection of every Qwen-2.5-1.5B layer (~28 layers × 7 projections) via SVD on HF checkpoint weights. Correlate row L_i against per-layer participation-ratio breathing amplitude. ~1h CPU.
**Requires:** CPU only; cached HF Qwen-2.5-1.5B weights; existing per-layer PR cache from pathway11_h100.
**Would change:** Confirm → adds an alternative mechanistic driver to H-4 that we must control for; F-1's universality claim becomes testable across model families. Reject → embedding-coupling story does not explain breathing, strengthens H-4 / EoS framing.
**Blocks:** nothing.

### H-369: Prefill L19 DoM AUROC carries a token-rarity confound
**Priority:** HIGH
**Motivated by:** 2505.15624 + F-2
**Test:** Compute per-prompt mean ‖E[token]‖₂ on MATH-500 inputs; refit prefill DoM correctness probe with rarity partialled out via residualization. Compare AUROC. 30min CPU on cached prefill activations.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/` activations + Qwen tokenizer.
**Would change:** Confirm (AUROC drops below ~0.74) → F-2 needs a rare-token control noted in narrative docs; selective-prediction (F-8) numbers may shift modestly. Reject → F-2 is robust against the stagnation-confound mechanism.
**Blocks:** nothing.

### H-370: Prefill DoM direction rotates across training checkpoints in concert with σ_max(E_embed)
**Priority:** MEDIUM
**Motivated by:** 2505.15624 + H-8
**Test:** On Pythia-1.4B or OLMo-1B intermediate checkpoints, refit prefill DoM at each checkpoint and report cos(DoM_t, DoM_t+k) and σ_max(E)_t curves. ~1 day H100.
**Requires:** H100; new activation extractions on Pythia/OLMo checkpoint releases (not Qwen — Qwen lacks public intermediate checkpoints).
**Would change:** Confirm → H-8 is wrong; prefill DoM stability is regime-dependent. Reject → embedding-coupling story is insufficient to drive prefill DoM rotation; H-8 strengthened.
**Blocks:** H-8.

---


### H-371: L19 prefill DoM is parallel across MATH-500 paraphrases / few-shot reorderings
**Priority:** HIGH
**Motivated by:** 2505.17306 + F-2
**Test:** generate 3 paraphrases per MATH-500 problem (or permute the few-shot ordering), extract L19 prefill DoM separately for each variant on the cached Stage-2 pipeline, compute pairwise cos similarity matrix. ~30min H100 to regenerate paraphrase activations + 10min CPU for cosines.
**Requires:** H100 for activation extraction on paraphrased prompts; existing Stage-2 extractor.
**Would change:** confirm → F-2 is a *geometry-claim* (a stable direction in residual space), not a benchmark-specific feature. Reject (low cosine, < 0.5) → F-2 is sensitive to surface form, weakens its claim of capturing "correctness" rather than "MATH-500 stylistic features".
**Blocks:** nothing.

### H-372: Mean-pool-prompt DoM beats prefill-final-token DoM at L19 for AUROC
**Priority:** MEDIUM
**Motivated by:** 2505.17306 + F-2
**Test:** re-extract DoM at L19 on cached `pathway11_h100/prefill_gated_compute` NPZs using attention-mask-weighted mean over prompt tokens. OOF 5-fold AUROC. ~20min CPU.
**Requires:** CPU only; cached Stage-2 NPZs (already exist).
**Would change:** confirm → F-2 should be re-anchored on the mean-pooled extraction. Reject → confirms the prefill-final-token specific position (counter-evidence for the universality-by-mean-pool hypothesis from Wang et al.).
**Blocks:** H-371 (cleaner extraction first before computing cross-distribution parallelism).

---


### H-373: Sinii's RL-trained per-layer steering vector at L17/L19 has high cosine with our prefill L19 DoM direction
**Priority:** HIGH
**Motivated by:** 2505.18706 + F-2 + F-3
**Test:** Load Sinii's public checkpoint for Qwen2.5-1.5B (`github.com/corl-team/steering-reasoning`); for each layer ℓ ∈ [0, 27], compute `cos(s_ℓ, prefill_L19_DoM)` and `cos(s_ℓ, final_L19_DoM)` using cached P11 prefill/final DoM vectors. ~2h CPU once checkpoint is downloaded.
**Requires:** CPU only; cached prefill_DoM vector from `pathway11_h100/prefill_gated_compute/`; Sinii checkpoint (HuggingFace or repo release).
**Would change:** If `cos(s_17, prefill_DoM) > 0.3` or `cos(s_19, prefill_DoM) > 0.3`, F-2 graduates from "correlational probe" to "causal direction RL would have learned" — strong corroboration. If both cosines are < 0.1, F-2 is a probe direction unrelated to actual reasoning steering, and the AUROC 0.7731 story needs reframing as an off-axis correlate of correctness rather than the correctness direction.
**Blocks:** Should run before P11-FE470 (FE73 only makes sense if there is *any* alignment between our DoM and the validated causal directions).

### H-374: Fixed (position-independent) bias steering on prefill L19 DoM lifts MATH500 K=1 accuracy by ≥ 5 points on Qwen2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2505.18706 + F-3 + H-1
**Test:** Add α · prefill_L19_DoM (cached, no retraining) as a fixed additive bias at L19 to all token positions during MATH500 K=1 greedy decoding on Qwen2.5-1.5B base. Sweep α ∈ {0.5, 1, 2, 5, 10}. Compare against base K=1 = 48.6%. ~30min H100.
**Requires:** H100 access (or 2060 if 1.5B weights fit), cached prefill_DoM vector, MATH500 dataset.
**Would change:** If best-α accuracy is ≥ 53.6%, the F-3 framing of "direction rotates so fixed steering fails" is at minimum incomplete — fixed-vector steering on the prefill-DoM direction has causal traction. If the result is null or negative across all α, H-1's per-position bank is the only path forward, and the prefill-DoM direction is genuinely not the right steering axis.
**Blocks:** H-1 should be re-baselined against H-374 — only the *delta* over fixed-vector steering is the H-1 contribution.

---


### H-375: Prefill L19 DoM is a confidence direction more than a correctness direction
**Priority:** HIGH
**Motivated by:** 2505.19184 (verbalized confidence is anti-Bayesian and miscalibrated by ~23pp on average) + F-2 (AUROC 0.7731 leaves headroom for non-correctness signal)
**Test:** Re-prompt Qwen-2.5-1.5B on the 500 cached MATH-500 problems, eliciting a 0-100 confidence after answer generation. Project cached prefill L19 activations onto DoM. Within each {correct, incorrect} class, compute Pearson correlation of DoM score with reported confidence vs Pearson correlation of DoM score with correctness across class boundary. If within-class DoM-confidence correlation > 0.3 *and* between-class DoM-correctness AUROC drops when controlling for reported confidence, the DoM is a confidence axis with a correctness side-effect, not a correctness probe.
**Requires:** CPU only, ~2h. Cached `pathway11_h100/prefill_gated_compute/prefill.npz` + answers + K=8 rollouts.
**Would change:** If confirmed, F-2 narrative shifts from "DoM predicts correctness" to "DoM is a calibration probe whose correctness AUROC reflects the static-QA confidence-correctness correlation." Validate-claims entry for AUROC 0.7731 stays valid as a number, but its semantic label changes. If rejected (DoM-correctness survives confidence control), F-2 is strengthened against this paper's challenge.
**Blocks:** Reframes H-12 (label-free abstention), since DoM-as-confidence-axis is closer to the SEP framing than to a "correctness probe"; affects how F-2 / F-8 are pitched in any writeup.

### H-376: Multi-turn dialogue contaminates the prefill DoM
**Priority:** MEDIUM
**Motivated by:** 2505.19184 (confidence escalates 72.9 → 83.3% across debate rounds with no ground-truth update)
**Test:** Construct 50 three-turn MATH-500 self-debates with Qwen-2.5-1.5B (turn 1: "solve this problem", turn 2: "your answer might be wrong; reconsider", turn 3: "give your final answer"). At each turn, extract L19 prefill activations on the dialogue-prefix-conditioned problem statement. Compute DoM score, DoM-correctness AUROC, and score-distribution mean per turn.
**Requires:** H100 4h re-extraction + 1h CPU analysis. Re-uses Qwen-2.5-1.5B at H100; new dialogue conditioning required (cached prefill is single-turn).
**Would change:** If AUROC drops ≥0.1 per turn and the score mean drifts upward, F-8's 71.6%@0.5 coverage number does not transfer to dynamic settings — narrative needs an explicit "static-QA only" caveat. If AUROC and mean are stable, F-2/F-8 generalize and the paper's behavioral overconfidence is decoupled from internal-state miscalibration (a strong, novel claim).
**Blocks:** Whether H-1 (per-position DoM steering) needs a multi-turn variant before any deployment claim.

---


### H-377: Concatenated upper-third MLP probe pushes correctness AUROC above 0.7731
**Priority:** HIGH
**Motivated by:** 2505.20309 + F-2 + F-9
**Test:** Train MLP (hidden=1024, ReLU) on `concat(h_L19, ..., h_L27)` from cached Stage 2 prefill activations, MSE-target = correctness. Report 5-fold OOF AUROC. Compare to F-2's 0.7731.
**Requires:** 4h CPU, no new data, cached Stage 2 NPZs from `pathway11_h100/prefill_gated_compute/`.
**Would change:** On confirm (gain ≥ 0.02): F-2 reframed as "low-dim manifold across upper-third layers mediates correctness," F-9 retracted in its strong "single-layer is sufficient" form. On reject: F-2 / F-9 strengthened — single L19 layer really does carry all the available signal.
**Blocks:** P11-FE7 (single-direction-vs-manifold SVD test) — concatenation probe gives a quicker, lower-bound version of the same question.

### H-378: Refusal direction (Eq. 7 of 2505.20309) carries non-trivial correctness AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2505.20309 + F-2 + H-12
**Test:** Compute `d_refuse` from Qwen-2.5-1.5B `W_U` using the WAS GitHub refusal/answer token lists. Project Stage 2 prefill L19 activations onto `d_refuse`, fit logistic, compute 5-fold OOF AUROC on MATH-500 correctness. Also compute `cos(d_refuse, prefill_DoM_L19)` and `cos(d_refuse, final_DoM_L19)`.
**Requires:** 30 min CPU, Qwen-2.5-1.5B unembedding matrix (already on disk), cached Stage 2 prefill activations.
**Would change:** On confirm (AUROC ≥ 0.6): label-free DoM proxy unlocked — semantic-entropy-style probes (H-12) can be skipped in favor of pure embedding arithmetic. On reject (AUROC ≈ 0.5): refusal-axis ⊥ correctness-axis in Qwen-2.5-1.5B, strengthening the F-3 orthogonality story by extending it to a third independent direction.
**Blocks:** H-12 (semantic entropy probes).

---


### H-379: A sparse atom-set probe at L19 beats single-direction DoM on MATH-500 correctness
**Priority:** HIGH
**Motivated by:** 2505.20322 + F-2
**Test:** On cached P11 L19 prefill activations (Qwen-2.5-1.5B, 500 MATH-500 problems), apply STA-style amplitude+frequency atom-selection treating each residual-stream coordinate as a candidate atom. Sweep thresholds (α, β); compare resulting sparse-probe OOF 5-fold AUROC to F-2's 0.7731 single-direction DoM. Est: 30min CPU.
**Requires:** CPU only; cached prefill NPZ (`pathway11_h100/.../prefill_gated_compute/`).
**Would change:** Confirm → F-2's "single direction" framing must be reframed as "low-dim sparse code." F-9's "CoE-60 redundant with single direction" also needs revision (CoE collapses to *DoM*, not necessarily to the optimal sparse code). Reject → F-2 stands as a true 1-D claim, robust to STA generalization.
**Blocks:** P11-FE478, P11-FE481

### H-380: Mean-pool-over-answer aggregation reconciles prefill / final-token orthogonality
**Priority:** MEDIUM
**Motivated by:** 2505.20322 + F-3
**Test:** From cached per-token L19 activations, compute answer-mean DoM. Measure cos to prefill DoM and final-token DoM; measure MATH-500 correctness AUROC. Predicts: if cos ≥ 0.5 to both AND AUROC ≥ 0.75, the 0.046 prefill/final orthogonality is a position-summary artifact, not a structural property of the residual stream. Est: 30min CPU.
**Requires:** CPU only; cached per-token NPZs.
**Would change:** Confirm → F-3 must be reframed as "per-position summaries are orthogonal but the underlying signal is unimodal." Reject → orthogonality is structural and STA-style mean-pool aggregation would underperform per-position probes on correctness.
**Blocks:** P11-FE479

### H-381: Steering at L19 changes generation length, not correctness
**Priority:** HIGH (gates H-1 interpretation)
**Motivated by:** 2505.20322 (reasoning-length-not-accuracy result on DeepSeek-R1-Distill-Qwen-7B/GSM8K) + H-1
**Test:** When H-1 (per-position DoM steering) runs, log per-problem output length alongside accuracy at each steering scale. Compute |Δlength|/baseline and |Δaccuracy| per scale. Predicts: |Δlength| dominates |Δaccuracy| — i.e., DoM is a length controller, not a correctness controller. Est: free piggyback on H-1's H100 budget.
**Requires:** Same as H-1 (H100 + intervention pipeline).
**Would change:** Confirm → H-1's "DoM steering moves accuracy" is reframed as "DoM steering moves length, accuracy follows length where length helps." Reject → DoM has an accuracy-axis component independent of length, which is the load-bearing claim for selective-prediction → steering generalization.
**Blocks:** H-1 interpretation; P11-FE480

---


### H-382: CCPS final-token features dominate single-direction prefill DoM at AUROC and selective accuracy on MATH-500
**Priority:** HIGH
**Motivated by:** 2505.21772 + F-2 + F-8
**Test:** Run P11-FE482. Compute CCPS feature bundle (Jacobian L2 norm, epsilon-to-flip, PEI, KL/JS perturbation-trajectory stats; ε_max=20.0, S=5) on cached P11 H100 final-token hidden states for Qwen-2.5-1.5B on MATH-500. Train 5-fold OOF MLP. Compare AUROC against F-2's 0.7731 (prefill DoM) and 0.7186 (final-token DoM); compare selective-prediction accuracy at coverage 0.5 against F-8's 71.6%. Estimated 1h CPU.
**Requires:** CPU, no new H100; uses existing P11 final-token hidden state NPZs and frozen Qwen-2.5-1.5B LM_Head.
**Would change:** Confirm → F-2 ordering ('prefill > final') is feature-engineering-bound, not geometric, and F-8 number is loose. Reject → CCPS does not transfer from MMLU to MATH-500 reasoning regime, confirming F-2's geometric reading.
**Blocks:** H-383, H-19 follow-up routing.

### H-383: Concatenating CCPS final-token features with prefill L19 DoM does not improve AUROC, refuting F-3's orthogonal-subspaces reading
**Priority:** HIGH
**Motivated by:** 2505.21772 + F-3
**Test:** After H-382, train [CCPS_final, prefill_DoM_projection] joint classifier vs CCPS-only and prefill-only. ΔAUROC < 0.005 → orthogonality is rotational redundancy, not complementary signal. Estimated 30min CPU.
**Requires:** CPU, depends on H-382 features.
**Would change:** Confirm → F-3's narrative ('different subspaces encode different correctness signal') retracted; cos 0.046 reframed as low-rank projection of one scalar. Reject → F-3 stands and prefill genuinely adds non-redundant information beyond saturated final-token features.
**Blocks:** nothing.

### H-384: CCPS epsilon-to-flip is a drop-in replacement for verification-prompt-based C_exact routing
**Priority:** MEDIUM
**Motivated by:** 2505.21772 + H-19
**Test:** Compute epsilon-to-flip on Qwen-2.5-1.5B MATH-500 generations, aggregate over answer-token span, threshold to route between K=1 and K=8 sampling. Compare matched-compute accuracy vs (a) H-19's prompt-based verification routing, (b) pure C_infer K=8. Estimated 30min H100 + 1h CPU.
**Requires:** H100 for the K=8 sampling lane; CPU for the gate.
**Would change:** Confirm → H-19 simplifies (no verification prompt needed) and saves one decode pass per problem. Reject → either epsilon-to-flip is too noisy at the answer span level, or prompt-verification carries information epsilon-to-flip misses.
**Blocks:** nothing.

---


### H-385: CCPS final-token features dominate single-direction prefill DoM at AUROC and selective accuracy on MATH-500
**Priority:** HIGH
**Motivated by:** 2505.21772 + F-2 + F-8
**Test:** Run P11-FE485. Compute CCPS feature bundle (Jacobian L2 norm, epsilon-to-flip, PEI, KL/JS perturbation-trajectory stats; ε_max=20.0, S=5) on cached P11 H100 final-token hidden states for Qwen-2.5-1.5B on MATH-500. Train 5-fold OOF MLP. Compare AUROC against F-2's 0.7731 (prefill DoM) and 0.7186 (final-token DoM); compare selective-prediction accuracy at coverage 0.5 against F-8's 71.6%. Estimated 1h CPU.
**Requires:** CPU, no new H100; uses existing P11 final-token hidden state NPZs and frozen Qwen-2.5-1.5B LM_Head.
**Would change:** Confirm → F-2 ordering ('prefill > final') is feature-engineering-bound, not geometric, and F-8 number is loose. Reject → CCPS does not transfer from MMLU to MATH-500 reasoning regime, confirming F-2's geometric reading.
**Blocks:** H-386, H-19 follow-up routing.

### H-386: Concatenating CCPS final-token features with prefill L19 DoM does not improve AUROC, refuting F-3's orthogonal-subspaces reading
**Priority:** HIGH
**Motivated by:** 2505.21772 + F-3
**Test:** After H-385, train [CCPS_final, prefill_DoM_projection] joint classifier vs CCPS-only and prefill-only. ΔAUROC < 0.005 → orthogonality is rotational redundancy, not complementary signal. Estimated 30min CPU.
**Requires:** CPU, depends on H-385 features.
**Would change:** Confirm → F-3's narrative ('different subspaces encode different correctness signal') retracted; cos 0.046 reframed as low-rank projection of one scalar. Reject → F-3 stands and prefill genuinely adds non-redundant information beyond saturated final-token features.
**Blocks:** nothing.

### H-387: CCPS epsilon-to-flip is a drop-in replacement for verification-prompt-based C_exact routing
**Priority:** MEDIUM
**Motivated by:** 2505.21772 + H-19
**Test:** Compute epsilon-to-flip on Qwen-2.5-1.5B MATH-500 generations, aggregate over answer-token span, threshold to route between K=1 and K=8 sampling. Compare matched-compute accuracy vs (a) H-19's prompt-based verification routing, (b) pure C_infer K=8. Estimated 30min H100 + 1h CPU.
**Requires:** H100 for the K=8 sampling lane; CPU for the gate.
**Would change:** Confirm → H-19 simplifies (no verification prompt needed) and saves one decode pass per problem. Reject → either epsilon-to-flip is too noisy at the answer span level, or prompt-verification carries information epsilon-to-flip misses.
**Blocks:** nothing.

---


### H-388: Label-free generation-time uncertainty (TokenSAR / MTE) matches prefill-DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2505.23224 + F-2
**Test:** Compute U_MTE and U_TSAR per-problem on cached pathway11_h100 K=1 generation logits (MATH-500, Qwen-2.5-1.5B). Train a logistic regressor on per-problem correctness vs. each signal. Compare AUROC to F-2's 0.7731 and to a multi-signal stack {U_MTE, U_TSAR, prefill-DoM}. Estimated time: 30min CPU.
**Requires:** CPU only. Cached pathway11_h100 NPZs + per-token logits + sentence-transformers/all-MiniLM-L6-v2.
**Would change:** Confirm → F-2's "*better than final-token*" claim must be re-stated with a baseline that includes label-free generation-time signals; the geometry frame is a *complement* to, not a *substitute* for, token statistics. Reject → strengthens F-2 as the strongest single correctness predictor on math reasoning, ruling out a major alternative.
**Blocks:** H-12 cannot be definitively answered until this is run.

### H-389: Per-sentence confidence variance carries information orthogonal to prefill-DoM
**Priority:** MEDIUM
**Motivated by:** 2505.23224 + F-3 + H-5
**Test:** On cached MATH-500 generations, sentence-segment outputs, compute U_MTE per sentence, take within-question variance as a feature. Regress per-problem correctness on (prefill-DoM, within-chain-variance, final-token-DoM) jointly. Test whether within-chain-variance has non-zero coefficient after partialling out the geometric features. Estimated time: 1h CPU.
**Requires:** CPU only. Cached pathway11_h100 generation strings + per-token logits.
**Would change:** Confirm → H-5's "decomposability" framing is wrong (or at least not captured by prefill); per-step calibration is informationally distinct. Reject → prefill-DoM subsumes within-chain dispersion, strengthening F-3's orthogonality and H-5's decomposability framing.
**Blocks:** nothing.

### H-390: Step-level (C_exact, MMBoundary-style) abstention beats response-level (F-8) abstention at matched compute
**Priority:** HIGH
**Motivated by:** 2505.23224 + F-8 + H-19
**Test:** On MATH-500 cached generations, identify the lowest-U_TSAR sentence per problem; regenerate from that mid-chain checkpoint; accept if local confidence improves. Match total regeneration tokens to F-8's K=2.5 baseline. Compare answered-set accuracy at 50% coverage. Estimated time: 2h H100 (mid-chain regeneration) + 30min CPU.
**Requires:** H100 pod for mid-chain regeneration; cached pathway11_h100 prefixes.
**Would change:** Confirm → F-8's per-problem framing is sub-optimal; selective prediction should be re-formulated at sentence granularity, with implications for the prefill-DoM headline. Reject → F-8's coarser-granularity framing is genuinely competitive, and per-step methods are not free wins.
**Blocks:** H-19 partially.

---


### H-391: Top-K\*≈20 SAE features at L19 prefill match or beat the F-2 single-vector DoM AUROC of 0.7731 on Qwen-2.5-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2505.23556 + F-2 + F-8
**Test:** P11-FE491 — load pretrained Qwen-2.5-1.5B residual-stream SAE at L19, derive V_R\* from MATH-500 prefill correct/incorrect difference-in-means, run CosSim-prefilter (K₀=10) + AP-on-SAE-features against an "I"/"Here" output proxy, take top K\*=20 features, fit a logistic probe over their activations, evaluate 5-fold OOF AUROC against the F-2 baseline 0.7731 on the same split. ~1 H100 day.
**Requires:** pretrained Qwen-2.5-1.5B residual-stream SAE (SAELens / community), cached Stage-2 NPZs at L19 prefill (already on disk), an H100 for the AP backward passes.
**Would change:** Confirm → F-2 is upgradable to a sparse-feature probe with strictly higher AUROC and (per H-392) much better OOD generalisation; F-9 (CoE-60 redundant with single-layer DoM) gets a stronger interpretation as "DoM captures F_common, CoE-60 captures F_common + a few F_specific". Reject → DoM is not just a coarse summary; the L19 single-direction story is mechanistically tight, strengthening F-2.
**Blocks:** H-392, H-393 (both reuse the same Qwen SAE setup).

### H-392: A sparse-feature probe over the K\* features from H-391 has a markedly smaller in-distribution-vs-OOD selective-prediction-accuracy gap than the F-8 dense L19 DoM probe
**Priority:** HIGH
**Motivated by:** 2505.23556 (Tab 4: dense gap 0.32–0.93 vs sparse gap 0.03–0.17) + F-8
**Test:** P11-FE492 — extract MATH-500 (in-distribution) and an OOD math set (AIME-2024, GSM-Hard, or held-out MATH-500 categories) prefill L19 activations on Qwen-2.5-1.5B; run both the F-8 dense DoM selective-prediction protocol and the H-391 sparse-feature variant on each; report selective-prediction accuracy at coverage 0.5 for both and compute the (in-dist − OOD) gap. ~2 H100 days, or ~1 day reusing FE71 features.
**Requires:** H-391 features, OOD math eval set with K=1 generations cached on Qwen-2.5-1.5B, an H100 for activation extraction.
**Would change:** Confirm → F-8's 71.6% becomes meaningful only at the in-distribution caveat; the sparse-feature variant becomes the canonical selective-prediction protocol; opens H-12 to SAE-feature-based label-free probes. Reject → F-8 is OOD-robust as-is, and the Yeo Tab 4 result is safety-domain-specific.
**Blocks:** nothing.

### H-393: SAE-decomposed L19 prefill and L19 final-answer-token activations on Qwen-2.5-1.5B share a substantial common feature set (mean Jaccard ≥ 0.5) despite cos(prefill_DoM, final_DoM) = 0.046
**Priority:** MEDIUM
**Motivated by:** 2505.23556 (F_common vs F_specific, Sect 4.2) + F-3
**Test:** P11-FE493 — for each MATH-500 problem, take top-K (K=20) active SAE features at L19 prefill and L19 final-answer-token; compute Jaccard. Mean across problems. ~30 min CPU after H-391's SAE setup.
**Requires:** H-391 SAE setup, cached Stage-2 NPZs at L19 prefill and final-token (already on disk).
**Would change:** Confirm (Jaccard ≥ 0.5) → F-3's "two unrelated streams" framing is incomplete; introduces an F_common backbone hypothesis and reframes F-9. Reject (Jaccard ≤ 0.2) → F-3 strengthens; the orthogonality is a real feature-level disconnect, not just a linear-summary artifact.
**Blocks:** nothing.

---


### H-394: Cross-scale affine mapping preserves the F-2 prefill DoM correctness signal within the Qwen-2.5 family
**Priority:** HIGH
**Motivated by:** 2506.00653 + F-2 + F-3
**Test:** P11-FE494: train affine map Qwen-2.5-1.5B L19 → 7B L21 on cached prefill activations, then evaluate (1) cos(mapped 1.5B DoM, 7B native DoM) and (2) mapped-DoM AUROC on 7B correctness vs 7B native AUROC.
**Requires:** H100 (~30 min train), 1.5B + 7B P11 cached prefill NPZs (already available), 7B correctness labels (already in `pathway11_h100/prefill_gated_compute/results.json`).
**Would change:** *Confirm* (mapped-DoM AUROC ≥ 0.70, cos ≥ 0.7) → F-2 generalises across scale via LRT mechanism, supports H-11 directly. *Reject* (mapped-DoM AUROC ≤ 0.55, cos ≤ 0.3) → F-2 is scale-coupled, contradicts LRT for correctness directions specifically (not just behavioral); promotes 2506.00653 to CONTRADICTED-on-correctness.
**Blocks:** H-11 cleanly (LRT *is* the H-11 mechanism if confirmed; H-11 needs a different mechanism if rejected)

### H-395: F-3 orthogonality (cos ≈ 0.046 prefill vs final DoM) is a 1.5B-specific projection artefact, not a universal feature-space property
**Priority:** MEDIUM
**Motivated by:** 2506.00653 + F-3
**Test:** P11-FE495: directly measure 7B-native cos(prefill_DoM, final_DoM) on cached 7B L21 activations. Three-way compare (1.5B-native, affine-projected, 7B-native).
**Requires:** No new compute beyond P11-FE494 + 5min CPU on cached 7B activations.
**Would change:** *Confirm* (7B native cos large, e.g. ≥ 0.3) → F-3 weakens to "1.5B-specific orthogonality"; H-13 (head-level attribution) becomes more attractive as the explanation. *Reject* (7B native cos ≤ 0.1) → F-3 strengthens to "scale-invariant orthogonality of correctness directions", which would be a stronger and more general finding.
**Blocks:** nothing

---


### H-396: Our prefill L19 DoM is a projection of Venhoff's uncertainty-estimation direction
**Priority:** HIGH
**Motivated by:** 2506.18167 + F-2
**Test:** Run Venhoff GPT-4o annotation on our 500 P11 K=1 CoTs.
Extract per-behavior DoM at L19 on Qwen-2.5-1.5B prefill activations
(uncertainty, backtracking, example-testing, knowledge-addition).
Compute cosine of each against our F-2 prefill L19 DoM. If
|cos(prefill_DoM, uncertainty_DoM)| > 0.5, F-2 is partially the
uncertainty axis. If a 4-stack outperforms our DoM AUROC by ≥0.02
OOF, F-2 framing under-decomposes.
**Requires:** ~$5 GPT-4o annotation; 1h CPU on cached prefill NPZs.
**Would change:** On confirm, F-2 narrative shifts from "single
correctness direction" to "uncertainty axis is the dominant
correctness predictor among 4 behavior axes". On reject, F-2's
single-direction framing is robust under behavior decomposition.
**Blocks:** H-1 reformulation (per-position steering should target
behavior-specific contrasts, not correctness contrast).

### H-397: F-3 prefill/final orthogonality reduces to behavior-decomposition orthogonality
**Priority:** MEDIUM
**Motivated by:** 2506.18167 + F-3
**Test:** After H-396 produces behavior DoMs at L19, reproject our
prefill DoM and final-token DoM onto each behavior axis. Compute
|cos(prefill_DoM, final_DoM)| as a function of which behavior axis
is removed. If projecting out the uncertainty axis from prefill and
the deduction axis from final brings cos > 0.3, F-3's "0.046
between time-points" reduces to "0.046 between behaviors" and the
temporal interpretation is wrong.
**Requires:** Cached prefill + final NPZs from P11; output of H-396.
**Would change:** On confirm, F-3 narrative shifts from "different
computations at different times" to "different behavior axes
dominate at different times". On reject, the temporal computation
story survives behavior decomposition.
**Blocks:** nothing.

### H-398: Behavior-decomposed steering improves over correctness-DoM steering on H-1
**Priority:** HIGH
**Motivated by:** 2506.18167 (public steering harness) + H-1
**Test:** On Qwen-2.5-1.5B at L19, +/- 1.0σ steering with
(a) F-2 correctness DoM vs (b) backtracking DoM + uncertainty DoM
(layer-mean-norm scaling). 500 MATH-500 problems, K=1 greedy.
Report accuracy delta vs 48.6% baseline and behavior-fraction
delta. If (b) produces accuracy delta > 0 while (a) produces 0,
H-1 must reformulate around behavior contrast.
**Requires:** 4h H100; output of H-396 for behavior DoMs.
**Would change:** On confirm, H-1 retired in favor of "behavior-
specific steering for accuracy" hypothesis. On reject, H-1's
correctness-DoM framing is correct and behavior decomposition is
unnecessary for accuracy steering.
**Blocks:** H-1 (would supersede if confirmed).

---


### H-399: L19/L20 hosts at least two near-orthogonal task-relevant directions (correctness axis ⊥ redundancy axis)
**Priority:** HIGH
**Motivated by:** 2506.18831 + F-2 + F-3
**Test:** Compute STU-PID-style redundancy DoM at L19 (and L20) on auto-labelled MATH-500 chunks; measure cos with F-2 prefill DoM; evaluate each direction's AUROC for the *other's* task. If both AUROCs are weak (each direction predicts only its own label) and cos < 0.2, H-399 holds.
**Requires:** CPU only; cached pathway11_h100 activations; small auto-label script ("after first \\boxed{}").
**Would change:** On confirm — F-2's "DoM" is one of several functional directions, not the correctness axis. Reframes any DoM-only paper claim. On reject — DoM is a single conflated axis that mixes correctness and redundancy.
**Blocks:** P11-FE505 should not run as a clean H-1 test until H-399 is settled.

### H-400: PID-gated steering on MATH-500 1024-tok yields ≥2pp accuracy gain at ≤−15% tokens
**Priority:** HIGH
**Motivated by:** 2506.18831 + H-1
**Test:** Run P11-FE505 STU-PID replication on MATH-500 with Qwen-2.5-1.5B (1024-tok, K=1). Pre-register threshold: ≥2pp accuracy gain with ≤−15% tokens vs unsteered baseline.
**Requires:** 1–2 H100-days; cached chunk classifier from H-399 work.
**Would change:** On confirm — we have a deployable inference-time efficiency lever and H-1 has a strong baseline protocol. On reject — STU-PID is GSM8K-specific (short-CoT artifact) and the H-1 design space stays open.
**Blocks:** Final design of any H-1 / H-19 protocol.

---


### H-401: Batch dispersion at L19 prefill matches or beats supervised DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2506.24106 + F-2 + F-8
**Test:** Bin 500 MATH-500 problems by K=8 majority-correctness in 10% strata; compute mean pairwise cosine distance of L19 prefill activations per stratum. Check monotonicity (target: non-overlapping SE bars at neighboring strata, like Tables 3–8) and compute bin-level AUC against the F-2 DoM AUROC of 0.7731. ~30 min CPU on cached `pathway11_h100/prefill_gated_compute` activations.
**Requires:** CPU; cached Stage 2 NPZ activations (already on disk).
**Would change:** Confirm → F-2's supervised DoM probe is unnecessary; correctness can be predicted from a 0-parameter scalar; H-12 (SEPs) becomes redundant; F-8 selective-prediction baseline gets a label-free competitor. Reject → DoM direction carries genuine information beyond scalar dispersion; F-2 mechanism stands.
**Blocks:** H-402.

### H-402: Per-problem prefill and final-token dispersion are scalar-correlated despite probe-direction orthogonality
**Priority:** HIGH
**Motivated by:** 2506.24106 + F-3
**Test:** From the L19 activations underlying `scratch/pathway10_temporal_and_verifier_results.json`, compute per-problem batch dispersion at the prefill window and at the final-token window. Check Pearson r across the 500 problems. ~30 min CPU.
**Requires:** CPU; cached pathway10 temporal NPZs.
**Would change:** Confirm (r > 0.5) → F-3's "two orthogonal signals" framing is a probe artifact; the underlying signal is a single scalar shared across deep-layer windows. Reject (r < 0.2) → F-3 stands; prefill and final-token activations encode genuinely independent correctness information.
**Blocks:** nothing.

### H-403: Output-embedding dispersion gap G predicts checkpoint accuracy without forward passes
**Priority:** MEDIUM
**Motivated by:** 2506.24106 + F-2 + H-8
**Test:** Compute G = within(digit-token rows of W_o) + between(digit, non-math reference) for Qwen2.5-{1.5B, 7B, 14B}, Qwen2.5-Math-{1.5B, 7B}, Distill-Qwen-{1.5B, 7B, 14B}. Check Spearman vs our P11 K=1 MATH-500 accuracies and the paper's reported MATH accuracies. ~5 min CPU.
**Requires:** CPU only; HF unembedding-matrix downloads.
**Would change:** Confirm (Spearman > 0.9) → G is a training-free screen that obviates expensive end-to-end checkpoint comparisons; H-8 (DoM stability across checkpoints) is partially solved by a much simpler statistic. Reject → checkpoint quality requires forward-pass-derived signal.
**Blocks:** nothing.

---


### H-404: F-3 prefill/final orthogonality is phase-change-localized, not circuit-distinction
**Priority:** HIGH
**Motivated by:** 2507.02199 + F-3
**Test:** Build the 28×28 cos similarity matrix of (DoM_prefill_L_i, DoM_final_L_j) on Qwen-2.5-1.5B using cached pathway11_h100 NPZs. Inspect whether the orthogonality (cos=0.046) is sharply concentrated at (L19, L19) or is a uniformly low-cos band across most (i,j). ~1h CPU.
**Requires:** Cached prefill + final-token activations for L0-L27 on Qwen-2.5-1.5B over MATH-500. (Currently we have L19 only — may require re-extraction for missing layers, +H100 time.)
**Would change:** If sharply localized, F-3 weakens to "phase change at L19" rather than "two circuits"; F-2's L19 specificity becomes an artifact of the residual stream passing through a transitional readout coordinate. If distributed (orthogonality across most i≠j), F-3's two-circuits reading is upgraded.
**Blocks:** nothing; supports re-interpretation of F-3 narrative in PERSPECTIVES.md.

### H-405: Linear DoM is not the optimal correctness probe at L19
**Priority:** MEDIUM
**Motivated by:** 2507.02199 + F-2
**Test:** Train 1-hidden-layer MLP (hidden_dim=64, ReLU) on L19 prefill activations, target MATH-500 K=1 correctness, 5-fold OOF. Compare to linear DoM AUROC=0.7731. ~30min CPU on cached NPZs.
**Requires:** Cached pathway11_h100 prefill NPZs (already available).
**Would change:** If MLP_AUROC > 0.7731 + 0.02, F-2's "linear DoM is the correctness direction" reading is incomplete and we need a richer probe family. If MLP_AUROC ≈ 0.7731, F-2 is robust and Huginn's lens-dependence is architecture-specific (tied recurrent weights).
**Blocks:** nothing.

### H-406: Latent reasoning does not emerge in current architectures (negative-result umbrella)
**Priority:** PARKED (umbrella negative-result hypothesis; affects framing not experiments)
**Motivated by:** 2507.02199 + H-22
**Test:** Triangulate against (a) Huginn rank-trajectory negative result (this paper), (b) Yang et al. 2402.16837 multi-hop latent reasoning negative results, (c) any future Coconut/Abstract-CoT replication. If three independent architectures (depth-recurrent, multi-hop QA, abstract-CoT) all fail to evidence latent reasoning, H-22's premise is undermined regardless of breathing magnitude.
**Requires:** No new compute; literature triangulation.
**Would change:** Demotes H-22 from MEDIUM to PARKED. Reframes F-1 / F-2 as signals over verbalized reasoning specifically, not "reasoning" generally. Narrows our project's scope claim in README.
**Blocks:** H-22 deprioritization.

---


### H-407: CoRE last-layer last-token-per-step trajectory dynamics carry correctness signal not captured by L19 prefill DoM
**Priority:** HIGH
**Motivated by:** 2507.06087 + F-2 + F-9
**Test:** Implement CoRE-Eval cycle detection (final layer, per-step, W=32, P_max=8, ρ*=0.7, M=8) on cached Qwen-2.5-1.5B MATH-500 1024-tok. Compute per-trial features (max ρ_t, fraction of steps in CYCLE) and measure (a) standalone AUROC, (b) residual AUROC after partialling out L19 prefill DoM score. Estimated time: 2h CPU.
**Requires:** CPU; cached `pathway11_h100/...` 1.5B MATH-500 trajectories. No new GPU run needed if last-layer hidden states were saved alongside L19 (DATA_MANIFEST check required).
**Would change:** If standalone AUROC ≥ 0.7731, F-2's "L19 privileged" framing weakens to "multiple layers carry independent correctness signal." If residual AUROC > 0.55, F-9's "CoE redundant with single-layer DoM" is refuted — trajectory dynamics carry independent signal. If both fail (AUROC ≤ 0.5 and residual ≈ 0), the paper's signal does not transfer to base (non-R1-distilled) Qwen, which itself constrains the universality claim of CoRE.
**Blocks:** H-409 (combined-gate selective prediction depends on this signal being non-trivial)

### H-408: CoRE cycle frequency separates F-7 D-bucket from A-bucket
**Priority:** MEDIUM
**Motivated by:** 2507.06087 + F-7
**Test:** Compute per-trial CoRE cycle-presence rate (fraction of steps in CYCLE state, max ρ_t) on the 36 D-bucket and 207 A-bucket trials in cached 1.5B MATH-500 1024-tok. Compare distributions; report AUC of cycle features for D-vs-A discrimination. Estimated time: 30 min CPU after H-407 infra.
**Requires:** CPU; cached trajectories; D/A bucket trial-id lists from `pathway11_h100/`.
**Would change:** If D-bucket trials show systematically more / longer cycles than A-bucket (e.g., AUC > 0.65 for the discrimination), F-7 is methodologically corroborated by a fully independent featurization. If no separation (AUC ≈ 0.5), F-7's "collective signature" claim must be reconciled with CoRE's null result on the same trials.
**Blocks:** nothing

### H-409: Label-free CoRE cycle gating matches or beats supervised L19 DoM at selective prediction
**Priority:** HIGH
**Motivated by:** 2507.06087 + F-8
**Test:** Build accuracy-vs-coverage curves on 1.5B MATH-500 1024-tok using three gating rules: (a) DoM-only (F-8 baseline, 71.6%/50%), (b) ¬cycle_detected only, (c) intersection (DoM ≥ τ AND ¬cycle). Compare AUC and the operating point at coverage 0.5. Estimated time: 30 min CPU after H-407 infra.
**Requires:** CPU; cached trajectories; F-8 reproduction script.
**Would change:** If (b) reaches ≥ 71.6% at coverage 0.5, F-8's implicit "supervision is necessary" is refuted — a label-free signal achieves parity. If (c) exceeds (a) by >2 pp, the combined gate becomes the new selective-prediction recommendation. If (b) collapses (e.g., < 65% at 0.5), CoRE's label-free advantage is overstated for our regime.
**Blocks:** nothing

---


### H-410: Latent-CoT models (Coconut / Recurrent-Depth) preserve or strengthen the L19 DoM correctness signal monotonically with vertical recurrence count
**Priority:** HIGH
**Motivated by:** 2507.06203 + F-2 + F-5
**Test:** Run Recurrent-Depth (2502.05171, open weights) on MATH-500 at iteration counts r ∈ {1, 4, 16, 64}; extract per-token residuals at the matching mid-depth layer; fit prefill DoM logistic at each r; plot AUROC vs r. Also fine-tune Qwen-2.5-1.5B with Coconut-style continuous-thought feedback on a small MATH-500 subset and re-run extraction. Est: 1–2 days H100 for Recurrent-Depth, plus 1 H100-day for Coconut fine-tune.
**Requires:** H100 SXM, public Recurrent-Depth checkpoint, MATH-500 (already cached), Coconut training script.
**Would change:** Confirm → DoM is a recurrence-amplified signal; our 0.7731 on Qwen-2.5-1.5B is a depth-floor, latent-CoT models should hit ≥0.85. Reject → DoM is an architectural fingerprint that does not transfer to recurrent-depth backbones; F-2 is Qwen-specific, broader claims weaken.
**Blocks:** P11-FE519, H-19 (D-bucket interpretation depends on this).

### H-411: F-4 asymmetric-collapse signal is downstream of OOD-stabilization, not correctness-specific
**Priority:** HIGH
**Motivated by:** 2507.06203 (citing Wang [112] embedding-trajectory volatility) + F-4
**Test:** Compute per-problem trajectory volatility (variance of consecutive-layer cosines) on cached 1024-tok Qwen-2.5-1.5B residuals; regress correctness ~ volatility_score + DoM_score on MATH-500. If volatility coefficient absorbs >50% of F-4's variance and DoM coefficient survives, F-4 is OOD detection in disguise. Est: 3h CPU.
**Requires:** Cached `pathway11_h100/prefill_gated_compute/` NPZs, scikit-learn.
**Would change:** Confirm → F-4 (asymmetric collapse) is reframed as OOD detection, not correctness. Reject → F-4 carries correctness-specific information beyond volatility, strengthening F-4.
**Blocks:** nothing (low downstream impact, but invalidating F-4 would shift the §1d graveyard).

### H-412: CoE-60 features residualized against L19 prefill DoM carry no incremental AUROC
**Priority:** MEDIUM
**Motivated by:** 2507.06203 (treating CoE [113] / 2410.13640 as a primary trajectory-geometry signal) + F-9
**Test:** Fit three logistics on cached 1024-tok activations: (a) all 60 CoE features, (b) CoE-60 with L19 prefill DoM projection removed, (c) prefill DoM alone; compare 5-fold OOF AUROC. Threshold: |AUROC_b − AUROC_c| < 0.01 confirms F-9. Est: 2h CPU.
**Requires:** Cached `pathway11_h100/prefill_gated_compute/` NPZs.
**Would change:** Confirm → F-9 generalizes; CoE field-of-view is just L19 DoM. Reject → there is residual trajectory-geometry signal beyond DoM; reopen P9 trajectory features.
**Blocks:** nothing.

### H-413: L19 correctness signal is rank-1 (DoM-only), not circuit-level
**Priority:** MEDIUM
**Motivated by:** 2507.06203 §4.2 circuit framing + F-2 + F-3
**Test:** At L19, project out (a) prefill DoM, (b) K random orthonormal directions of equal norm, (c) top-K PCA components by variance; re-run MATH-500 K=1 evaluation; compare accuracy delta. If random-direction ablation degrades by ≥50% of DoM-ablation delta, the survey's circuit framing is supported. Est: 1 day H100.
**Requires:** H100, frozen Qwen-2.5-1.5B, MATH-500 prompts.
**Would change:** Confirm → F-2's 1-D rank-1 framing holds; reject → mid-layer correctness depends on a circuit-scale subspace, not just DoM, which weakens the project's central headline.
**Blocks:** H-1 (per-position DoM steering) — direction-ablation is the linear analog.

---


### H-414: L19 DoM direction is stable across token positions during generation
**Priority:** HIGH
**Motivated by:** 2507.12428 Figure 3b (cross-position probe transfer on safety) + F-3
**Test:** Refit L19 DoM at 10 positions on `pathway11_h100/prefill_gated_compute/` cache, compute pairwise cosine matrix. Stable = most pairwise cos > 0.5.
**Requires:** CPU only, ~30 min, all data cached.
**Would change:** If stable, F-3 (cos=0.046 prefill vs final) is a prefill-only edge effect rather than a global property — the "prefill and final live in different subspaces" framing collapses. If unstable (cosines stay <0.2), F-3 is robust and the safety-vs-correctness asymmetry needs a domain-specific explanation.
**Blocks:** H-13 head-level attribution gets cleaner if direction is stable (single direction to attribute).

### H-415: Final-token-trained DoM achieves prefill-like AUROC when applied OOD to prefill
**Priority:** HIGH
**Motivated by:** 2507.12428 Section 5.2 "present-trained outperforms future-trained" + F-2
**Test:** Take final-token L19 DoM weights from existing JSON, score prefill activations from same fold, compute AUROC. Compare to 0.7731 prefill-trained AUROC and 0.7186 final-token-trained-and-tested AUROC.
**Requires:** CPU only, ~5 min, all data cached.
**Would change:** If AUROC ≈ 0.77 (matches prefill-trained), F-2's "prefill is a privileged probe site" is wrong — it's the *direction* that matters, and you can fit it anywhere. If AUROC ≈ 0.72 (matches final-token), F-2 holds.
**Blocks:** H-1 per-position steering interpretation depends on whether the direction is shared.

### H-416: PCA-50 + LR on prefill L19 matches or beats raw-DoM AUROC
**Priority:** MEDIUM
**Motivated by:** 2507.12428 Section 3.2 PCA-50 default pipeline
**Test:** Refit prefill-position probe with PCA-50 preprocessing, 5-fold OOF, compare AUROC to 0.7731.
**Requires:** CPU only, ~10 min, all data cached.
**Would change:** If PCA-50 changes AUROC by >0.01, our 91/91 invariant claim's robustness is questioned. Likely a no-op; cheap sanity check.
**Blocks:** nothing.

### H-417: Performative-math-CoTs are enriched in D-bucket
**Priority:** MEDIUM
**Motivated by:** 2507.12428 Section 6 performative CoT (~11% of safety data) + F-7 D-bucket
**Test:** Run the Chan et al. p=80 / q=50 filter on K=1 MATH-500 CoTs (using GPT-4.1-nano as the per-sentence correctness predictor). Compute overlap with the 36 D-bucket problems vs the 207 A-bucket problems.
**Requires:** GPT-4.1-nano API (~$15), 2h elapsed.
**Would change:** If D-bucket ⊃ performative-math-CoTs, F-7 D-bucket fragility gets a published-name mechanism (performative reasoning) and joins a broader literature. If no overlap, "performative" is a refusal-specific phenomenon and D-bucket fragility is something else (likely sampling stochasticity, supporting H-21).
**Blocks:** nothing; informs H-21.

---


### H-418: F-2 prefill L19 DoM AUROC drops within-MATH-subject vs pooled
**Priority:** HIGH
**Motivated by:** 2508.11290 + F-2 + EXP-024 (currently the source of 0.7731)
**Test:** Stratify Qwen-2.5-1.5B prefill L19 DoM 5-fold OOF AUROC by MATH-500 subject tag (7 subjects). Report per-subject AUROC + 95% CI. Pooled baseline is 0.7731. Compute the largest within-subject AUROC drop from pooled.
**Requires:** CPU, cached pathway11_h100/prefill_dom.npz + MATH-500 subject metadata. ~30 min.
**Would change:** If max within-subject AUROC drops below 0.70, F-2's "correctness signal" framing must be qualified to "correctness + which-subject confound." Updates would propagate to F-8 (selective prediction may be subject-routing in disguise).
**Blocks:** nothing (parallelizable to existing P11 work).

### H-419: Effectiveness-ratio peak layer ≠ AUROC peak layer
**Priority:** MEDIUM
**Motivated by:** 2508.11290 + F-2
**Test:** Compute Eff(ℓ) = ‖v(ℓ)‖/(σ_correct(ℓ) + σ_incorrect(ℓ) + ε) at every layer L0–L27 for Qwen-2.5-1.5B prefill correct vs incorrect groups. Argmax-Eff layer vs argmax-AUROC layer (currently L19). If they differ, the "best layer for AUROC prediction" is not the "best layer for steering" — relevant when deciding which layer P10-FE1 steers at.
**Requires:** CPU, cached prefill activation NPZs L0–L27 (regen if missing). ~30 min if cached, ~2h if regen needed.
**Would change:** Confirms or falsifies the implicit assumption that AUROC and steering-quality co-locate. If Eff peaks elsewhere (e.g., L23 like SafeConstellations' late-layer effects), P10-FE1 should steer at the Eff peak.
**Blocks:** P10-FE1 layer choice (not blocking — can run in parallel and revise).

### H-420: Within-subject cos(prefill_DoM, final_DoM) >> pooled 0.046
**Priority:** MEDIUM
**Motivated by:** 2508.11290 + F-3
**Test:** Compute c_subject_prefill_DoM and c_subject_final_DoM for each of 7 MATH-500 subjects. Report 7 within-subject cos values. Pooled baseline 0.046. SafeConstellations predicts within-subject |cos| ≫ pooled (their per-task silhouette is ~2× combined).
**Requires:** CPU, cached pathway11_h100 NPZs. ~15 min.
**Would change:** If max within-subject |cos| > 0.20, the F-3 orthogonality claim is partly a heterogeneity-pooling artifact. F-3's status as a "structural" finding gets downgraded; the framing shifts to "prefill and final DoMs are orthogonal *across* subjects but not necessarily *within* a subject."
**Blocks:** nothing.

---


### H-421: FASB-style top-k-head probe ensemble exceeds single-layer L19 DoM on Qwen-1.5B MATH-500
**Priority:** HIGH
**Motivated by:** 2508.17621 + F-2 + F-9
**Test:** Train per-head logistic probes on cached 1024-tok prefill
activations across all 28 layers × 12 heads of Qwen-1.5B; pick top-24 by
validation AUROC; evaluate ensemble AUROC against the F-2 single-direction
baseline of 0.7731. Sweep top-k ∈ {6, 12, 24, 48}. ~2–4h CPU.
**Requires:** Cached per-head prefill activations from `pathway11_h100/`;
if only residual-stream is cached, +1 day H100 to re-extract per-head.
**Would change:** Confirm: F-2 must be reframed as "single best of a
head-distributed family"; F-9 (CoE redundancy) must allow that the
redundancy is per-head-pooling rather than single-layer-DoM ceiling.
Reject: F-2 / F-9 stand as written.
**Blocks:** H-422 (downstream selective-prediction shootout).

### H-422: FASB-style steering recovers MATH-500 accuracy on the F-8-abstained subset
**Priority:** HIGH
**Motivated by:** 2508.17621 + F-8
**Test:** On the bottom-50%-confidence half of Qwen-1.5B MATH-500
(currently abstained per F-8 selective-prediction protocol), apply FASB
top-k-head steering with adaptive r = I(p>β)·p·α and s=10-token
backtrack. Measure post-intervention K=1 accuracy. ~2 days H100.
**Requires:** H100 with hooks for per-head MHSA output rewriting and
sequence-level rewind; FASB's α ∈ [20, 80] / β ∈ [0.3, 0.5] sweep.
**Would change:** Confirm: F-8 ("abstain is the ceiling") must be
softened to "abstain is the ceiling for *open-loop* use of L19 DoM";
H-1 gets a concrete recipe. Reject: F-8 stands; truthfulness-style
steering doesn't transfer to math correctness.
**Blocks:** nothing.

---


### H-423: L19 prefill DoM is partly a token-substitution direction at the unembedding lattice
**Priority:** HIGH
**Motivated by:** 2509.06608 + F-2 + EXP-018
**Test:** On Qwen-2.5-1.5B, project L19 prefill DoM through W_U and report top-10 nearest tokens by cosine. Run Qwen-2.5-1.5B and Qwen-2.5-7B K=1 greedy on MATH-500 with each prompt prefixed by `Step 1: ` (and `To ` control). Estimated time: 1 hr H100 + 30 min CPU.
**Requires:** Cached W_U for Qwen-2.5-1.5B/7B; cached MATH-500 baselines (`pathway11_h100/prefill_gated_compute/results.json`); H100 SXM for K=1 reruns.
**Would change:** On confirm (top-K is reasoning-process words AND prefix prompting recovers ≥75% of the predicted DoM-steering effect): F-2's FINDINGS.md entry gains a token-substitution mechanism note; H-1 priority drops; the project's "geometric correctness signal" framing must be qualified. On reject (top-K is non-process tokens AND prefix prompting recovers <25%): F-2 stays as a geometric finding, and Sinii et al.'s last-layer mechanism does not generalize to mid-layer prefill DoMs.
**Blocks:** H-1, P10-FE1.

### H-424: Multi-layer prefill DoM steering beats L19-only steering on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2509.06608 + F-9
**Test:** Train a 28-vector multi-layer prefill DoM on Qwen-2.5-1.5B; apply all 28 simultaneously at matched α; sweep α; compare K=1 MATH-500 acc to L19-only DoM at the same α grid. Estimated time: 4 hr H100.
**Requires:** H100 SXM, cached prefill activations from P11.
**Would change:** Confirm (multi ≥ L19-only + 5pp): F-9's "redundant" framing is qualified to probing only. Reject (multi ≤ L19-only + 1pp): F-9 generalizes to steering; the "single direction" story stays clean.
**Blocks:** Nothing.

---


### H-425: TriviaQA-trained DoM transfers to MATH-500 prefill correctness
**Priority:** HIGH
**Motivated by:** 2509.10625 + F-2
**Test:** Train Qwen-2.5-1.5B prefill-L19 DoM on TriviaQA (60K subsample, HF). Evaluate AUROC on cached MATH-500 prefill activations (`pathway11_h100/`). Compare to F-2's 0.7731 (in-distribution).
**Requires:** ~30 min H100 for TriviaQA prefill extraction; cached MATH-500 NPZs; HF dataset access.
**Would change:** If AUROC ≥ 0.65, F-2 is a *transferable* correctness direction (paper's universality claim wins on Qwen-1.5B as well). If AUROC < 0.55, F-2 is in-distribution-only and the paper's GSM8K-failure pattern replicates on our setup — major reframe of F-2 from "deeper signal" to "MATH-500-surface probe."
**Blocks:** H-426.

### H-426: MATH-500 prefill DoM AUROC drops monotonically with problem step count
**Priority:** HIGH
**Motivated by:** 2509.10625 (Math Ops AUROC 0.782–0.858 vs GSM8K 0.499–0.601 split) + F-2 + H-5
**Test:** GPT-4-judge step counts for all 500 MATH-500 problems. Stratify into single-step / 2–3 step / 4–5 step / 6+ step bins. Recompute prefill DoM AUROC per bin on cached Qwen-2.5-1.5B activations.
**Requires:** ~$5 GPT-4 API for labelling; CPU only for AUROC; cached `pathway11_h100/exp1_cross_model/qwen2.5-1.5b/prefill_dom_oof.json` per-problem scores.
**Would change:** Monotonic drop replicates paper's single/multi-step boundary, kills H-5's familiarity framing, and reframes F-2 as "single-step-recall" probe. Flat AUROC across bins refutes paper's mechanism on Qwen-1.5B/MATH and supports a Qwen-specific exception.
**Blocks:** nothing.

### H-427: Verbalised confidence on Qwen-2.5-1.5B underperforms prefill DoM on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2509.10625 baseline (verbalised confidence < DoM in-distribution AND OOD)
**Test:** Re-run K=1 MATH-500 generations on Qwen-2.5-1.5B with appended "How confident are you (0–100%)?" prompt, parse responses, compute AUROC for confidence-vs-correctness. Compare to F-2's 0.7731.
**Requires:** ~1h H100 + parsing CPU.
**Would change:** Replicates a baseline gap that strengthens F-8's selective-prediction value over verbalised methods. Confirms small-model verbalised confidence is uncalibrated.
**Blocks:** nothing.

### H-428: cos(prefill_DoM, final_DoM) shrinks with model scale
**Priority:** MEDIUM
**Motivated by:** 2509.10625 (intermediate-layer saturation across 7–70B) + F-3 (cos 0.046 on Qwen-1.5B)
**Test:** Compute cos(prefill_DoM, final_DoM) on Qwen-2.5-7B and Llama-3.1-8B at the saturating layer using cached MATH-500 prefill+final-token activations from `pathway11_h100/exp1_cross_model/`.
**Requires:** ~30min H100 for final-token DoM extraction on 7B/8B; CPU for cosine.
**Would change:** If cos rises above ~0.5 with scale, F-3's orthogonality is a 1.5B-specific quirk and "direction rotation" is not universal. If cos stays ≤ 0.1 in 7B+, F-3 strengthens.
**Blocks:** nothing.

---


### H-429: D²HScore (label-free all-layer dispersion+drift) matches or exceeds supervised L19 prefill DoM AUROC at 7B
**Priority:** HIGH
**Motivated by:** 2509.11569 + F-2, F-9
**Test:** Implement D²HScore (intra-layer attention-weighted centroid dispersion + inter-layer drift, all 28 layers, attention top-k at 0.4) on cached P11 7B Stage 2 NPZs for 500 MATH-500 problems. 1h CPU.
**Requires:** CPU only; cached 7B per-layer NPZs (already on disk per DATA_MANIFEST).
**Would change:** Confirm → F-2's "L19 prefill DoM is privileged" narrative needs a "supervised lift" subclaim and F-9's all-layer-collapses-to-single-layer no longer holds at 7B. Reject → F-2 / F-9 stand and D²HScore's headline doesn't transfer to MATH-500.
**Blocks:** Decision on whether to publish single-layer vs all-layer geometric framing.

### H-430: Intra-layer dispersion sign is opposite to PR-collapse sign
**Priority:** MEDIUM
**Motivated by:** 2509.11569 + F-4
**Test:** Compute attention-weighted final-layer centroid distance per problem on cached P11 1.5B NPZs. Bucket by K=1 correctness. If incorrect group has higher mean dispersion than correct, sign matches D²HScore — and F-4's "asymmetric collapse" framing flips to "asymmetric spread."
**Requires:** CPU only; cached final-token activations and K=1 labels.
**Would change:** Confirm → F-4 narrative re-frames; the geometric story is dispersion, not collapse, with PR being a flawed summary statistic. Reject → F-4 is genuinely PR-specific and dispersion isn't equivalent.
**Blocks:** Nothing.

### H-431: Attention-top-k token selection subsumes PANL token heuristic
**Priority:** MEDIUM
**Motivated by:** 2509.11569 + H-18 + F-7
**Test:** Compute attention-rollout-top-k tokens (k=0.4) per problem from P11 Stage 4a; compare set IoU against PANL token; train DoM on activations averaged over attention-top-k vs DoM on PANL alone for B/D-bucket discrimination.
**Requires:** CPU only; cached attention rollouts.
**Would change:** Confirm → PANL heuristic upgrades to a principled label-free token selector; downstream H-18/H-19 experiments simplify. Reject → PANL captures something attention-top-k misses (likely position-locked syntactic role).
**Blocks:** H-18 implementation choice.

---


### H-432: D²HScore (label-free all-layer dispersion+drift) matches or exceeds supervised L19 prefill DoM AUROC at 7B
**Priority:** HIGH
**Motivated by:** 2509.11569 + F-2, F-9
**Test:** Implement D²HScore (intra-layer attention-weighted centroid dispersion + inter-layer drift, all 28 layers, attention top-k at 0.4) on cached P11 7B Stage 2 NPZs for 500 MATH-500 problems. 1h CPU.
**Requires:** CPU only; cached 7B per-layer NPZs (already on disk per DATA_MANIFEST).
**Would change:** Confirm → F-2's "L19 prefill DoM is privileged" narrative needs a "supervised lift" subclaim and F-9's all-layer-collapses-to-single-layer no longer holds at 7B. Reject → F-2 / F-9 stand and D²HScore's headline doesn't transfer to MATH-500.
**Blocks:** Decision on whether to publish single-layer vs all-layer geometric framing.

### H-433: Intra-layer dispersion sign is opposite to PR-collapse sign
**Priority:** MEDIUM
**Motivated by:** 2509.11569 + F-4
**Test:** Compute attention-weighted final-layer centroid distance per problem on cached P11 1.5B NPZs. Bucket by K=1 correctness. If incorrect group has higher mean dispersion than correct, sign matches D²HScore — and F-4's "asymmetric collapse" framing flips to "asymmetric spread."
**Requires:** CPU only; cached final-token activations and K=1 labels.
**Would change:** Confirm → F-4 narrative re-frames; the geometric story is dispersion, not collapse, with PR being a flawed summary statistic. Reject → F-4 is genuinely PR-specific and dispersion isn't equivalent.
**Blocks:** Nothing.

### H-434: Attention-top-k token selection subsumes PANL token heuristic
**Priority:** MEDIUM
**Motivated by:** 2509.11569 + H-18 + F-7
**Test:** Compute attention-rollout-top-k tokens (k=0.4) per problem from P11 Stage 4a; compare set IoU against PANL token; train DoM on activations averaged over attention-top-k vs DoM on PANL alone for B/D-bucket discrimination.
**Requires:** CPU only; cached attention rollouts.
**Would change:** Confirm → PANL heuristic upgrades to a principled label-free token selector; downstream H-18/H-19 experiments simplify. Reject → PANL captures something attention-top-k misses (likely position-locked syntactic role).
**Blocks:** H-18 implementation choice.

---


### H-435: TD-learned value function on L19 prefill matches or beats static linear DoM
**Priority:** HIGH
**Motivated by:** 2509.12886 + F-2
**Test:** Train two-layer FC F̂_φ(s_t) via squared TD error on cached Qwen-2.5-1.5B MATH-500 L19 prefill + intermediate hidden-state NPZs (rollouts already cached in pathway11_h100). OOF 5-fold AUROC; compare to 0.7731 baseline. ~1h CPU.
**Requires:** CPU only; cached pathway11_h100 prefill NPZs; cached intermediate-state sequences for TD bootstrap.
**Would change:** Confirms — F-2's linear framing is the upper bound on prefill-correctness signal at L19. Rejects (V̂ AUROC > 0.7731 + 0.02) — F-2 needs a non-linear refinement; the prefill→correctness map has structure linear DoM is missing; F-9's "single-layer L19 DoM is sufficient" claim weakens.
**Blocks:** H-436 partially (if H-435 confirms, the layer comparison in H-436 is more interpretable).

### H-436: Last-layer (L27) prefill DoM matches L19 prefill DoM for correctness prediction
**Priority:** MEDIUM
**Motivated by:** 2509.12886 + F-2
**Test:** Extract last-layer prefill activations on cached MATH-500 problems (may need re-extract if NPZ only stores L19), fit linear DoM, OOF 5-fold AUROC. ~20min CPU + possibly 30min H100 for re-extract.
**Requires:** CPU; cached prompts + Qwen-2.5-1.5B model checkpoint if re-extraction needed.
**Would change:** Confirms (last-layer AUROC ≥ 0.75) — F-2's L19-specificity framing is overstated; rephrase as "L19 is *one of* several layers carrying prefill signal." Rejects (last-layer AUROC < 0.65) — L19 specificity is real; the mid-layer-specificity argument in F-2 holds.
**Blocks:** nothing.

### H-437: Within-Qwen-family L19 prefill DoM transfers from 7B to 1.5B
**Priority:** HIGH
**Motivated by:** 2509.12886 + H-11
**Test:** Project cached Qwen-2.5-1.5B L19 prefill activations onto cached Qwen-2.5-7B L19 prefill DoM direction (cosine projection or learned linear map), compute correctness AUROC on Qwen-1.5B MATH-500 K=1 labels. ~10min CPU.
**Requires:** CPU only; cached scratch/pathway10_temporal_and_verifier_results.json (has both 1.5B and 7B prefill DoM); cached pathway11_h100 1.5B prefill NPZ + labels.
**Would change:** Confirms (transfer AUROC ≥ 0.65) — H-11's distillation plan has a viable starting signal. Rejects (transfer AUROC < 0.60) — H-11 should be parked or redesigned; even within-family cross-scale prefill direction is not a usable teacher signal.
**Blocks:** H-11 (H-437 is the cheapest pre-flight check).

---


### H-438: TD-learned value function on L19 prefill matches or beats static linear DoM
**Priority:** HIGH
**Motivated by:** 2509.12886 + F-2
**Test:** Train two-layer FC F̂_φ(s_t) via squared TD error on cached Qwen-2.5-1.5B MATH-500 L19 prefill + intermediate hidden-state NPZs (rollouts already cached in pathway11_h100). OOF 5-fold AUROC; compare to 0.7731 baseline. ~1h CPU.
**Requires:** CPU only; cached pathway11_h100 prefill NPZs; cached intermediate-state sequences for TD bootstrap.
**Would change:** Confirms — F-2's linear framing is the upper bound on prefill-correctness signal at L19. Rejects (V̂ AUROC > 0.7731 + 0.02) — F-2 needs a non-linear refinement; the prefill→correctness map has structure linear DoM is missing; F-9's "single-layer L19 DoM is sufficient" claim weakens.
**Blocks:** H-439 partially (if H-438 confirms, the layer comparison in H-439 is more interpretable).

### H-439: Last-layer (L27) prefill DoM matches L19 prefill DoM for correctness prediction
**Priority:** MEDIUM
**Motivated by:** 2509.12886 + F-2
**Test:** Extract last-layer prefill activations on cached MATH-500 problems (may need re-extract if NPZ only stores L19), fit linear DoM, OOF 5-fold AUROC. ~20min CPU + possibly 30min H100 for re-extract.
**Requires:** CPU; cached prompts + Qwen-2.5-1.5B model checkpoint if re-extraction needed.
**Would change:** Confirms (last-layer AUROC ≥ 0.75) — F-2's L19-specificity framing is overstated; rephrase as "L19 is *one of* several layers carrying prefill signal." Rejects (last-layer AUROC < 0.65) — L19 specificity is real; the mid-layer-specificity argument in F-2 holds.
**Blocks:** nothing.

### H-440: Within-Qwen-family L19 prefill DoM transfers from 7B to 1.5B
**Priority:** HIGH
**Motivated by:** 2509.12886 + H-11
**Test:** Project cached Qwen-2.5-1.5B L19 prefill activations onto cached Qwen-2.5-7B L19 prefill DoM direction (cosine projection or learned linear map), compute correctness AUROC on Qwen-1.5B MATH-500 K=1 labels. ~10min CPU.
**Requires:** CPU only; cached scratch/pathway10_temporal_and_verifier_results.json (has both 1.5B and 7B prefill DoM); cached pathway11_h100 1.5B prefill NPZ + labels.
**Would change:** Confirms (transfer AUROC ≥ 0.65) — H-11's distillation plan has a viable starting signal. Rejects (transfer AUROC < 0.60) — H-11 should be parked or redesigned; even within-family cross-scale prefill direction is not a usable teacher signal.
**Blocks:** H-11 (H-440 is the cheapest pre-flight check).

---


### H-441: Prefill L19 DoM passes the Arditi/SteeringSafety KL<0.1 Alpaca filter
**Priority:** CRITICAL
**Motivated by:** 2509.13450 + 2406.11717 + F-2
**Test:** Apply directional ablation along prefill L19 DoM on Qwen-2.5-1.5B-Instruct
across a 256-prompt Alpaca eval set; measure final-token KL divergence between
ablated and clean models. Compare mean KL against 0.1 threshold. (~30min H100.)
**Requires:** Single H100, Qwen-2.5-1.5B-Instruct, cached prefill DoM at L19, 256
Alpaca prompts.
**Would change:** **Confirm**: F-2's direction is steering-eligible by 2025
state-of-the-art criteria; H-1 is methodologically defensible. **Reject**: F-2's
direction is probe-only, not causal at any steering scale that preserves fluency;
H-1 must search a different layer or downscale the intervention magnitude. The
finding controls every downstream steering experiment in P11.
**Blocks:** H-1 (cannot be properly tested without this), H-442, H-443.

### H-442: Per-layer DoM AUROC is broadly flat from L14 to L22 on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2509.13450 + F-2
**Test:** Sweep all 28 layers of cached prefill activations, fit per-layer OOF
5-fold logistic probe on DiffInMeans direction, plot AUROC curve. ~20min CPU.
**Requires:** CPU-only, cached prefill activations from
`pathway11_h100/prefill_gated_compute/`.
**Would change:** **Confirm flat**: F-2's "L19" is a single sample on a plateau;
the project's narrative emphasis on L19 specifically is over-strong.
**Confirm peaked**: F-2 stands and L19 deserves the emphasis.
**Reject** (AUROC peaks at a different layer): F-2 needs to be re-anchored.
**Blocks:** any future cross-model F-2 generalization claim.

### H-443: Conditional (CAST-style) DoM steering Pareto-dominates global DoM steering on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2509.13450 (Section 4.1.4) + Lee et al. 2024 CAST + F-3 + H-1
**Test:** Compare global directional ablation (apply to all tokens, all layers) vs.
conditional ablation (apply only when prefill activation similarity to a learned
trigger direction exceeds threshold) on MATH-500 K=1 accuracy. Measure both
effectiveness (Δ accuracy) and entanglement on {ARC-C, GPQA, TruthfulQA}. ~3h H100.
**Requires:** H100, Qwen-2.5-1.5B + 7B (optional), MATH-500, ARC-C, GPQA,
TruthfulQA. Calibrated threshold from a held-out trigger set.
**Would change:** **Confirm**: the project should adopt conditional steering as
default protocol for all P11 steering claims; F-3's "orthogonal directions" framing
should be supplemented by a gating story. **Reject**: global steering is not the
bottleneck and the F-3 framing is sufficient.
**Blocks:** any P11 steering paper draft.

---


### H-444: Prefill-L19 DoM AUROC is reachable by generic directions in the residual stream
**Priority:** HIGH
**Motivated by:** 2509.22067 + F-2
**Test:** P11-FE557 — sweep 1000 random unit vectors, compute OOF 5-fold AUROC against MATH-500 correctness on cached prefill activations, report null distribution and percentile of the 0.7731 DoM observation.
**Requires:** CPU only (~20 min); cached `pathway11_h100/prefill_l19_activations.npz`.
**Would change:** If the null 95th percentile exceeds 0.70, F-2's "privileged direction" framing weakens to "high-leverage subspace" and H-1's expected effect size shrinks (it becomes harder to distinguish DoM steering from random steering). If the null 95th percentile is ≤0.55, F-2 is robustly supported.
**Blocks:** H-1 (cannot interpret H-1 results without this null)

### H-445: F-3 cos=0.046 is at-or-below the high-dim baseline of two random unit vectors in d=1536
**Priority:** MEDIUM
**Motivated by:** 2509.22067 + F-3
**Test:** P11-FE558 — empirical CDF of |cos(prefill_DoM, v_i)| for 1000 i.i.d. random unit vectors; report rank of 0.046.
**Requires:** CPU (~5 min); cached DoM weights only.
**Would change:** If 0.046 is within ±1σ of the random-vector |cos| distribution, F-3's "two streams are orthogonal" finding reduces to "two streams are unrelated" — still a finding, but stops licensing two-stream architectural claims.
**Blocks:** nothing

### H-446: Steering at the Qwen MMLU-safe coefficient (c=0.5) using prefill_DoM moves MATH-500 accuracy beyond the random-vector null
**Priority:** HIGH
**Motivated by:** 2509.22067 + H-1 + F-2
**Test:** P11-FE559 — generate K=1 MATH-500 under DoM steering and 30 random-vector controls at c=0.5 at L19, compare accuracy distributions.
**Requires:** H100 day, Qwen-2.5-1.5B weights, MATH-500 prompts (cached).
**Would change:** If DoM-steered accuracy exceeds the random-vector 95th percentile, H-1 is confirmed and DoM is a steerable direction. If DoM-steered accuracy is within the random null (or below it — Rogue Scalpel suggests degradation is the default), H-1 is refuted *and* F-2's interpretation flips from "encodes correctness" to "predicts correctness via correlation, not causation."
**Blocks:** H-1 (this is the operationalization of H-1 with a proper null).

### H-447: Late-layer (L22+) prefill_DoM steering yields no MATH-500 lift (DoM is computed at L19, not RoPE-rotated)
**Priority:** MEDIUM
**Motivated by:** 2509.22067 + H-17 + F-2
**Test:** P11-FE560 — per-layer steering at {10,14,19,22,26} with c=0.5.
**Requires:** ~5 H100-hours, cached pipeline.
**Would change:** If lift is concentrated at L19 with sharp falloff matching Rogue Scalpel's late-layer null, H-17 (RoPE-mechanical rotation) is refuted in favor of "DoM is a layer-19-specific computed direction." If lift propagates roughly uniformly across layers, H-17 stands.
**Blocks:** H-17

---


### H-448: A text-only Qwen3-8B GCM trained on Qwen-2.5-1.5B's MATH-500 history matches the prefill L19 DoM AUROC of 0.7731
**Priority:** HIGH
**Motivated by:** 2509.24988 (RQ1 + Tables 2–3) + F-2 + EXP-relevant-prefill-runs in pathway11_h100
**Test:** Train Qwen3-8B with LoRA rank 32, BS 16, 1 epoch on (q, r, ĉ) tuples from MATH-500 K=1 (243/500 correct), evaluate AUROC on a 25% held-out slice. ~5h H100.
**Requires:** H100 pod, Qwen3-8B weights, cached pathway11_h100 K=1 generations. No new data.
**Would change:** If AUROC >= 0.77, F-2's hidden-state framing is dominated by a text-only CM and its narrative shifts from "L19 prefill direction encodes correctness" to "L19 prefill probe is a noisy readout of features visible from text". If AUROC <= 0.65, F-2's hidden-state framing is preserved as additive over text-only CMs.
**Blocks:** H-12 reframe, H-449.

### H-449: MATH-500 P(True) on Qwen-2.5-1.5B is matched by Qwen-2.5-7B P(True) reading 1.5B's responses, replicating 2509.24988's RQ1
**Priority:** HIGH
**Motivated by:** 2509.24988 RQ1 + F-2 + F-6
**Test:** Use the paper's verbatim P(True) prompt to elicit (a) Qwen-2.5-1.5B self-prediction, (b) Qwen-2.5-7B cross-prediction of 1.5B, on all 500 MATH-500 problems. Compute AUROC and ECE for each.
**Requires:** Existing H100 pod with Qwen-2.5-1.5B + 7B already loaded. ~2h.
**Would change:** Confirms (or refutes) that "self-knowledge" framings of F-2 / F-6 generalize to MATH-500 reasoning. If self - cross AUROC > 0.05, MATH-500 is *unlike* MMLU and self-knowledge is real on long-CoT tasks.
**Blocks:** H-448 interpretation.

### H-450: Spline calibration on prefill L19 DoM scores drops ECE below 0.03 without changing AUROC
**Priority:** MEDIUM
**Motivated by:** 2509.24988 Section 3.4 + Lucena 2018 + F-8
**Test:** Apply Lucena 2018 spline calibration to OOF prefill-DoM scores using 5% MATH-500 holdout. Recompute ECE and RMSCE on the remaining 95%.
**Requires:** Cached pathway11_h100 OOF scores. <10min CPU.
**Would change:** Cleanly improves F-8's calibration story without retraining. If ECE drops to <= 0.03, narrative for F-8 should add a calibrated-DoM variant alongside the raw probe.
**Blocks:** nothing.

### H-451: Answer-only `P(c | q, r̂)` CM matches the final-token L19 DoM AUROC of 0.7186
**Priority:** MEDIUM
**Motivated by:** 2509.24988 RQ2B + F-2 + F-9
**Test:** Train Qwen3-8B SCM on (q, r̂) inputs only (answer letter, not full trace) for MATH-500. Evaluate AUROC on 25% held-out, compare to 0.7186 final-token DoM.
**Requires:** H100 pod, ~2h.
**Would change:** If Answer-only AUROC >= 0.7186, the final-token DoM is doing nothing the model wouldn't know from just seeing the predicted answer choice. If far lower, the trace is doing real work.
**Blocks:** F-9 reframe.

---


### H-452: 7B prefill PR inversion (F-6) is a per-group sample-asymmetry artifact, not a geometric inversion
**Priority:** HIGH
**Motivated by:** 2509.26560 + F-6
**Test:** P11-FE566 — recompute peak prefill PR for the correct and incorrect groups inside Qwen-7B with (a) γ_row bias correction (full Q, sampled P), (b) bootstrap to matched P_correct = P_incorrect, with γ_naive. Two refutation paths in one experiment.
**Requires:** CPU only, ~1 h. Uses cached Stage 2 NPZs.
**Would change:** If the inversion vanishes under either treatment, F-6 was a within-model sample-asymmetry artifact; the "incorrect-group concentrates harder" story collapses. If it survives both, F-6 is robust and we can tighten the claim.
**Blocks:** nothing.

### H-453: Local dimensionality of L19 residual streams (γ_both^local) deviates from the Gaussian null even where PH (F-10) is at null
**Priority:** MEDIUM
**Motivated by:** 2509.26560 + F-10
**Test:** P11-FE567 — γ_both^local(r) sweep on L18–L20 of cached Stage 2 NPZs versus matched isotropic-Gaussian null with same (P, Q, per-feature variance). Compare across multiple r values.
**Requires:** CPU only, ~12 h total over three layers and a radius grid. Uses cached Stage 2 NPZs.
**Would change:** If local γ deviates from null at any radius, F-10's "PH = Gaussian null" finding does not extend to bias-corrected local dimensionality, and the topological-geometry framing is partially rescued at the local scale even though PH itself remains null. If γ_both^local also matches null, F-10 is strengthened across both global and local lenses.
**Blocks:** any future "local geometry differs from null" claim.

### H-454: Pathway-11 PR pipeline uses task-centering (column-center per-neuron across stimuli) consistently with Chun 2509.26560
**Priority:** MEDIUM
**Motivated by:** 2509.26560 + F-1 + F-4
**Test:** P11-FE569 — code audit of Stage 2 PR computation in pathway11_h100; if non-task-centered or mixed, recompute peak PR numbers under task-centering and report delta. Pure verification hypothesis.
**Requires:** CPU only, < 1 h. Uses cached Stage 2 NPZs.
**Would change:** If we currently use neuron-centering or mix, F-1 cross-architecture amplitudes and F-4 asymmetric-collapse ratios are not directly comparable to Chun's framework or to most of the literature. If we already task-center, no change.
**Blocks:** publication-grade write-up of F-1 and F-4.

---


### H-455: 7B prefill PR inversion (F-6) is a per-group sample-asymmetry artifact, not a geometric inversion
**Priority:** HIGH
**Motivated by:** 2509.26560 + F-6
**Test:** P11-FE570 — recompute peak prefill PR for the correct and incorrect groups inside Qwen-7B with (a) γ_row bias correction (full Q, sampled P), (b) bootstrap to matched P_correct = P_incorrect, with γ_naive. Two refutation paths in one experiment.
**Requires:** CPU only, ~1 h. Uses cached Stage 2 NPZs.
**Would change:** If the inversion vanishes under either treatment, F-6 was a within-model sample-asymmetry artifact; the "incorrect-group concentrates harder" story collapses. If it survives both, F-6 is robust and we can tighten the claim.
**Blocks:** nothing.

### H-456: Local dimensionality of L19 residual streams (γ_both^local) deviates from the Gaussian null even where PH (F-10) is at null
**Priority:** MEDIUM
**Motivated by:** 2509.26560 + F-10
**Test:** P11-FE571 — γ_both^local(r) sweep on L18–L20 of cached Stage 2 NPZs versus matched isotropic-Gaussian null with same (P, Q, per-feature variance). Compare across multiple r values.
**Requires:** CPU only, ~12 h total over three layers and a radius grid. Uses cached Stage 2 NPZs.
**Would change:** If local γ deviates from null at any radius, F-10's "PH = Gaussian null" finding does not extend to bias-corrected local dimensionality, and the topological-geometry framing is partially rescued at the local scale even though PH itself remains null. If γ_both^local also matches null, F-10 is strengthened across both global and local lenses.
**Blocks:** any future "local geometry differs from null" claim.

### H-457: Pathway-11 PR pipeline uses task-centering (column-center per-neuron across stimuli) consistently with Chun 2509.26560
**Priority:** MEDIUM
**Motivated by:** 2509.26560 + F-1 + F-4
**Test:** P11-FE573 — code audit of Stage 2 PR computation in pathway11_h100; if non-task-centered or mixed, recompute peak PR numbers under task-centering and report delta. Pure verification hypothesis.
**Requires:** CPU only, < 1 h. Uses cached Stage 2 NPZs.
**Would change:** If we currently use neuron-centering or mix, F-1 cross-architecture amplitudes and F-4 asymmetric-collapse ratios are not directly comparable to Chun's framework or to most of the literature. If we already task-center, no change.
**Blocks:** publication-grade write-up of F-1 and F-4.

---


### H-458: NRC1 inverts the F-4 asymmetric-collapse direction
**Priority:** HIGH
**Motivated by:** 2510.01105 + F-4
**Test:** Compute NRC1 (PCA-residual collapse metric, normalized features) on
cached L19 final-token residuals from pathway11_h100 MATH-500 sweep
(Qwen-2.5-1.5B). Define target subspace as top-n PCA of correct-only
answer-token activations. Per-problem NRC1, then 2-sample test on
NRC1(correct) vs NRC1(incorrect). Wall-clock ~1h CPU.
**Requires:** existing pathway11_h100 final-token L19 NPZs (no fresh data).
**Would change:** If NRC1 is *higher* (less collapse) for correct trajectories,
F-4's "collapse-as-correctness" framing aligns with 2510.01105's regression-
collapse-is-bad theory and F-4 needs re-interpretation as a regression
phenomenon not a confidence-collapse phenomenon. If NRC1 anti-correlates with
correctness, F-4 survives the import and the paper's framework does not
generalize cleanly to next-token prediction.
**Blocks:** nothing — independent test.

### H-459: Per-layer TwoNN-ID profile carries predictive signal beyond L19 DoM
**Priority:** MEDIUM
**Motivated by:** 2510.01105 + F-9
**Test:** For each MATH-500 problem on Qwen-2.5-1.5B, compute TwoNN-ID at
L0..L27 → 28-dim feature vector. Fit OOF 5-fold logistic on correctness,
compute AUROC. Compare to single-layer L19 DoM = 0.7731 baseline. ~2h CPU.
**Requires:** cached per-layer activations from pathway11_h100 (already on disk).
**Would change:** If the layer-ID profile lifts AUROC by >0.02 over L19 DoM
alone, F-9's "single layer suffices" claim weakens and the per-layer ID
trajectory becomes a candidate feature for selective prediction (F-8). If
no lift, F-9 strengthens against this orthogonal feature family.
**Blocks:** nothing.

### H-460: Breathing amplitude tracks per-problem K=8 majority entropy more than annotated difficulty
**Priority:** LOW
**Motivated by:** 2510.01105 + H-9
**Test:** Per-problem peak-PR amplitude on MATH-500. Regress amplitude on
(a) MATH-500 difficulty 1-5, (b) K=8 majority entropy (proxy for "noise"),
(c) generation length. Report partial-r² for each covariate. ~30min CPU.
**Requires:** K=8 sampling outputs (need to confirm we have these on disk for
the same 500 problems; otherwise BLOCKED on K=8 sweep).
**Would change:** If (b) dominates and (a) goes near-zero after partialing out,
H-9's "amplitude predicts difficulty" framing is wrong and amplitude is more
properly understood as tracking conditional answer-noise, matching the
2510.01105 noise-regime carve-up.
**Blocks:** nothing — diagnostic only.

---


### H-461: Full-layer activation-delta dominates single-layer L19 DoM for correctness prediction
**Priority:** HIGH
**Motivated by:** 2510.01591 + F-2 + F-9
**Test:** On cached P11 1.5B activations (MATH-500, all 28 layers, prefill + final-token positions), build CLUE-style centroids `V_succ`, `V_fail` using `Δh = h_prefill − h_final` over all layers, classify by layer-averaged Euclidean distance, OOF 5-fold. Compare top-1 accuracy and AUROC to single-layer L19 prefill DoM.
**Requires:** 30 min CPU on existing NPZ caches, no new generation.
**Would change:** If CLUE-AUROC ≥ 0.85 (vs. F-2: 0.7731), F-9's "CoE redundant with L19" must be revisited and F-2's L19-only framing should be replaced with a multi-layer-delta framing in narrative docs. If CLUE-AUROC ≤ 0.78, F-9 is reconfirmed and F-2's single-direction framing is robust.
**Blocks:** H-462.

### H-462: Layer-wise correctness separability is monotone in depth, not peaked at L19
**Priority:** HIGH
**Motivated by:** 2510.01591 (Figure 4) + F-2
**Test:** Compute prefill DoM AUROC at every layer 0-27 on 1.5B MATH-500 cached activations; plot alongside centroid-distance `d^(ℓ) = ||V_succ^(ℓ) − V_fail^(ℓ)||₂`. Test for monotone-increase vs. peaked-at-L19.
**Requires:** 20 min CPU on existing caches.
**Would change:** If monotone-up with peak at L25-L27, F-2's "L19 is peak" narrative claim is wrong; we should switch to a deeper-layer or multi-layer report. If genuinely peaked at L19, F-2 is reinforced *and* CLUE's "deeper-is-always-better" generalization is bounded.
**Blocks:** nothing.

### H-463: F-2's correctness signal disappears on pure-SFT (non-RL) models
**Priority:** MEDIUM
**Motivated by:** 2510.01591 (Section 4.5: RL → separable, SFT → entangled) + F-2
**Test:** Extract L19 prefill activations on Qwen2.5-Math-1.5B-Instruct (SFT-only, no RL stage) for MATH-500. Compute prefill DoM AUROC. Compare to Qwen-2.5-1.5B-Instruct (DPO-trained) F-2 result of 0.7731.
**Requires:** 4h H100 (new extraction on a second model), no new dataset.
**Would change:** If SFT-only AUROC drops to ≤ 0.55, F-2 / F-8 must be scoped to RL/DPO-trained models — a major narrative change. If SFT-only ≥ 0.70, CLUE's RL/SFT dichotomy is too strong, and F-2 is more general than CLUE predicts.
**Blocks:** any cross-paradigm steering experiment (H-1, H-10).

### H-464: The activation displacement Δh = h(prefill) − h(final) at L19 is a stronger correctness signal than either endpoint alone
**Priority:** HIGH
**Motivated by:** 2510.01591 + F-2 + F-3
**Test:** Compute the per-problem displacement vector Δh at L19 (prefill last-tok minus generated last-tok), fit DoM on this delta, OOF 5-fold AUROC. Compare to prefill-only (0.7731) and final-only (0.7186).
**Requires:** 30 min CPU on existing P11 caches.
**Would change:** If Δh-AUROC > 0.79 (above prefill), F-2's narrative should switch from "prefill is the signal" to "the reasoning displacement is the signal" — F-3's orthogonality finding becomes the *reason* the delta has signal. If Δh-AUROC ≤ prefill, F-2's prefill-privileged framing survives a direct CLUE-style test.
**Blocks:** nothing.

---


### H-465: Output-side NuclearNorm of K=N softmax matches L19 DoM AUROC for MATH-500 correctness
**Priority:** HIGH
**Motivated by:** 2510.02956 + F-2 (L19 DoM AUROC 0.7731) + F-9 (CoE-60 redundant with DoM)
**Test:** On cached P11 prefill_gated_compute K=8 MATH-500 results for Qwen-2.5-1.5B, compute per-problem NuclearNorm of the K=8 softmax answer-token matrix (also IM and ClassEntropy). Treat as a per-problem score, compute OOF 5-fold AUROC against K=1 correctness. ~30 min CPU.
**Requires:** CPU only. Cached K=8 softmax outputs (verify they include per-token logits, not just argmax answers; if not, ~3h H100 to re-extract).
**Would change:** If NuclearNorm AUROC ≥ 0.74, F-2's "supervised L19 DoM is the strong correctness predictor" reframes to "a strong predictor — but a label-free output aggregate captures most of it." H-12 promotes from speculative to partially confirmed (with NuclearNorm rather than semantic entropy as the winning family). If AUROC < 0.65, hidden-state geometry is genuinely necessary — F-2 gets stronger.
**Blocks:** H-466.

### H-466: Selective-prediction at coverage 0.5 holds with NuclearNorm replacing supervised DoM
**Priority:** HIGH
**Motivated by:** 2510.02956 + F-8 (71.6% answered acc at coverage 0.5)
**Test:** Re-run F-8's coverage-vs-answered-accuracy curve replacing supervised DoM scoring with per-problem NuclearNorm. Match the K=2.5 average compute budget. Compare answered accuracy at coverage [0.3, 0.4, 0.5, 0.6, 0.7].
**Requires:** CPU only, ~1h, depends on H-465 producing the score column.
**Would change:** If overlap within 2%, the program's selective-prediction headline is supervised-equivalent to a label-free aggregate — major positive update for H-12 and a story-level shift toward "the actionable inference recipe doesn't need probes." If gap > 5%, hidden-state probes are doing real work F-2 takes credit for.
**Blocks:** nothing.

### H-467: D-bucket fragility shows as dispersity collapse in K=8 softmax matrix
**Priority:** MEDIUM
**Motivated by:** 2510.02956 (dispersity-collapse-under-shift framing) + F-7 (D-bucket distinctive collective geometric signature)
**Test:** For the 36 D-bucket problems vs 207 A-bucket problems on Qwen-2.5-1.5B MATH-500, compute the K=8 answer-token NuclearNorm and ClassEntropy distributions. Two-sample test; effect size. If D-bucket median NuclearNorm is below A-bucket median by ≥ 1 SD, F-7 gains an output-side signature.
**Requires:** CPU only, ~30 min, depends on H-465 score column.
**Would change:** D-bucket gets a free output-side detector (no probe needed) and F-7's "geometric" framing softens to "visible at the output too." If no dispersity gap, F-7's residual-stream framing is genuinely necessary.
**Blocks:** nothing.

---


### H-468: PID-AcT closed-loop steering moves MATH-500 K=1 accuracy where ActAdd / DoM does not

**Priority:** HIGH
**Motivated by:** 2510.04309 + H-1 + F-2
**Test:** Implement PID-AcT at L19 on Qwen-2.5-1.5B with diff-in-means target = correct-class mean − incorrect-class mean. Sweep `K_p, K_i, K_d` from paper's best regime (`K_p ∈ {0.01, 0.05}`, `K_i ∈ {0.05, 0.10}`, `K_d ∈ {0.01, 0.05}`). Generate K=1 on MATH-500 and compare accuracy to base 48.6%. If any gain triple lifts accuracy by ≥3pp, PID-AcT confirms H-1's hope. If all gain triples leave accuracy ±1pp of base, the H-1 failure is in the disturbance structure, not the controller form (paper Sec. 4.2.1's `w⊥` argument).
**Requires:** H100 pod, ~4h, cached F-2 diff-in-means direction.
**Would change:** Confirm → H-1 promoted to active steering experiment with PID-AcT as default. Reject → close H-1 + downgrade Pathway 10 steering hopes; the prefill signal is read-only.
**Blocks:** H-1.

### H-469: Layer-wise PID feature direction does NOT beat single-layer L19 DoM as a correctness probe

**Priority:** MEDIUM
**Motivated by:** 2510.04309 + F-9 + F-2
**Test:** Compute the discrete PID feature `u(L=19) = K_p r(19) + K_i Σ_{j<19} r(j) + K_d (r(19)-r(18))` on cached Qwen-1.5B prefill activations across L0–L27. Fit logistic OOF readout against 1024-tok correctness labels. Compare AUROC against the F-2 anchor 0.7731. Test gain grid from FE69.
**Requires:** ~1h CPU, cached `pathway11_h100/prefill_gated_compute/` residuals.
**Would change:** Confirm (AUROC ≤ 0.7731) → re-confirm F-9; layer-wise dynamics matter for steering but not for probing — clean null. Reject (AUROC > 0.7731) → F-9 wrong, CoE-style multi-layer features have unrealised probe value, reopen Pathway 9.
**Blocks:** Nothing.

### H-470: The prefill→final-token rotation (F-3) is NOT bridged by layer-wise PID

**Priority:** MEDIUM
**Motivated by:** 2510.04309 + F-3 + H-17
**Test:** Compute `cos(u_PID(prefill_position, L=19), u_PID(final_token_position, L=19))` over the same gain grid as H-469. Compare against `cos(prefill_DoM, final_DoM) = 0.046` baseline.
**Requires:** ~30min CPU.
**Would change:** Confirm (cos remains <0.1) → admission note for 2510.04309 was wrong; the paper does not address our rotation problem. We need a *temporal* (across-token) PID variant, not the layer-wise PID the paper proposes. Reject (cos > 0.5) → unexpected — investigate why integrating across layers happens to align across token positions; could imply RoPE-like coupling between layer index and token index.
**Blocks:** Nothing.

---


### H-471: L19 prefill DoM peak is mechanically downstream of BOS massive activation
**Priority:** HIGH
**Motivated by:** 2510.06477 + F-2 + F-1
**Test:** Ablate MLP contribution to BOS at the emergence layer (located via FE77 entropy curve) on Qwen-2.5-1.5B and rerun MATH-500 K=1. Measure delta in L19 prefill DoM AUROC and in final-token DoM AUROC. Cached prefill NPZs sufficient for follow-up SVD analysis. ~1-2h H100.
**Requires:** H100 with hook capability, Qwen-2.5-1.5B weights, MATH-500 prompts
**Would change:** If AUROC collapses post-ablation, F-2 is a BOS-mediated readout (reframes selective-prediction story F-8 as a sink-direction probe). If AUROC survives, F-2 captures a content-direction orthogonal to massive activations and the L19 peak is mechanistically distinct from compression.
**Blocks:** H-13 (head-level circuit attribution becomes much sharper conditional on this result)

### H-472: Phase-2 dimensional breathing is BOS-norm modulation; deflation removes correct-vs-incorrect divergence
**Priority:** HIGH
**Motivated by:** 2510.06477 + F-1
**Test:** Per-layer SVD on cached Qwen-2.5-1.5B residual NPZs. For each layer, deflate the rank-1 σ₁ direction and recompute participation ratio. Compare correct vs incorrect PR curves before vs after deflation. ~10min CPU.
**Requires:** cached NPZs (Pathway 11), numpy SVD
**Would change:** Confirm → F-1 universality is BOS-scaffolding universality; reframe in PERSPECTIVES. Reject → F-1 captures content-driven geometry orthogonal to massive activations and is more interesting than the paper's mechanism allows.
**Blocks:** Nothing strictly, but resolves ambiguity behind H-9 (breathing-amplitude difficulty signal).

---


### H-473: Latent-Trajectory signals (Vilas 2025) outperform single-layer DoM at 1024-tok
**Priority:** HIGH
**Motivated by:** 2510.10494 + F-2, F-9
**Test:** Implement Net / Cumulative / Aligned Change on Qwen-2.5-1.5B 1024-tok cache (k=500 segments), compare layer-averaged AUROC against prefill L19 DoM AUROC 0.7731 and CoE-60 AUROC 0.811. Estimated 30min CPU.
**Requires:** CPU only, cached `pathway11_h100/*` NPZs.
**Would change:** If LT > 0.7731, F-9 needs a temporal-axis caveat ("redundant with single-layer DoM at fixed token, *but trajectory features can extract more*"). If LT ≤ 0.7731, F-9 generalizes cleanly to temporal axis and the prefill DoM remains the most parsimonious correctness signal.
**Blocks:** H-474 (drift-projection test depends on having LT signals computed first).

### H-474: Drift direction is a third correctness axis distinct from prefill DoM and final DoM
**Priority:** HIGH
**Motivated by:** 2510.10494 + F-3
**Test:** Compute drift vector per problem on 1024-tok cache; measure cos(drift, prefill_DoM), cos(drift, final_DoM), and AUROC of drift-direction projection. If both cosines small AND drift's AUROC > 0.7186, drift is a new axis. Estimated 1h CPU.
**Requires:** CPU only, cached residual streams; depends on segment-averaged hidden states from H-473 implementation.
**Would change:** F-3's two-axis framing extends to three axes. If drift is dominantly aligned with prefill or final, F-3 stays clean. If drift is its own subspace, P10's "five directions" inventory should add a sixth.
**Blocks:** any future steering experiment (P10-style) that wants to use drift as an intervention vector.

### H-475: Cumulative-Change negativity replicates on Qwen-2.5-1.5B (path geometry, not endpoint dimensionality)
**Priority:** MEDIUM
**Motivated by:** 2510.10494 + F-4
**Test:** Per-problem Cumulative Change at L19 and layer-averaged; Spearman vs K=1 correctness on 500 MATH-500 problems. Vilas r = -0.38 on 14B reasoning. If our r matches sign and is significant, F-4's asymmetric-collapse phenomenology unifies with their path-length framing. Estimated 20min CPU.
**Requires:** CPU only, cached NPZs.
**Would change:** F-4 reframed as "directed motion: correct traces move further with less wandering" instead of "correct traces collapse harder at final token." Same observation, different math; updates how we describe the finding in PROJECT_RECORD §1b.
**Blocks:** nothing.

---


### H-476: Prefill L19 DoM AUROC drops below 0.65 after controlling for prompt lexical diversity
**Priority:** HIGH
**Motivated by:** 2511.15210 + F-2
**Test:** Compute TAACO lemma-MATTR, function-word ratio, trigram-lemma-TTR, and token length per MATH-500 prompt. Regress prefill L19 DoM scalar on these four surface-feature predictors. Re-evaluate AUROC of residual DoM against correctness labels. ~30 min CPU on cached `pathway11_h100/prefill_gated_compute/results.json`.
**Requires:** CPU only. `pip install taaco`. Cached prefill DoM scalars and prompt strings already on disk.
**Would change:** If residual AUROC < 0.65, F-2 reframes as hybrid lexical+correctness signal — selective-prediction (F-8) coverage curve must be re-derived. If residual AUROC ≥ 0.70, F-2 is confirmed as a primarily-correctness signal robust to lexical confounds and the 0.7731 figure stands.
**Blocks:** H-9 interpretation (if F-2 is lexical-confounded, breathing-amplitude correlations with difficulty likely are too).

### H-477: D-bucket prompts are surface-form-distinctive, not reasoning-fragile
**Priority:** HIGH
**Motivated by:** 2511.15210 + F-7
**Test:** Compute TAACO features + token length + per-prompt PHDim on 36 D-bucket vs 207 A-bucket prompts. Logistic regression with surface features only. ~30 min CPU.
**Requires:** CPU only, cached `pathway11_h100/d_bucket_problems.json` + prompt strings.
**Would change:** If surface-feature classifier reaches AUROC > 0.85, F-7's "distinctive collective geometric signature" is plausibly an artifact of prompt surface form, not a fragile-reasoning attractor — H-21 (D-bucket attention narrowness) becomes lower-priority. If AUROC < 0.7, F-7's collective-geometry mechanism survives the surface-form challenge.
**Blocks:** H-21.

### H-478: PHDim discriminates genre on Qwen-2.5-1.5B residuals, refuting universal PH-null
**Priority:** HIGH
**Motivated by:** 2511.15210 + F-10 + EXP-026
**Test:** Build a 1000-text corpus mixing 500 MATH-500 prompts with 500 COLING wp/eli5/imdb stories (≥ 150 tokens). Extract Qwen-2.5-1.5B residual states at the same layer and protocol used in EXP-026. Run our PH pipeline (`pathway8_layer_pcb/persistence.py`) and Gaussian-null baseline. ~30 min CPU + ~1 h H100 for fresh extraction.
**Requires:** H100 for residual extraction on COLING subset; CPU for PH computation. Need to download a COLING subset (HF accessible).
**Would change:** Pedashenko predicts PHDim AUROC > 0.8 for genre discrimination across Gemma/Qwen/RoBERTa. F-10 predicts no signal vs null. Outcome resolves the scope of F-10: confirmed → F-10 holds within-genre only and must be re-stated; refuted → Pedashenko's external-corpus result fails to replicate on our extraction protocol (worth a methodological note).
**Blocks:** any future PH-based pathway-rebuild on heterogeneous corpora.

### H-479: Effective global linear embedding dimension at L19 prefill peaks near k≈60
**Priority:** MEDIUM
**Motivated by:** 2511.15210 + F-9
**Test:** Compute the explained-variance-vs-rank curve on cached prefill L19 activations across MATH-500. Compute PHDim per problem (random subsample). Compute Pearson r between PHDim and EV-k for k = 1..200. Locate peak. ~10 min CPU.
**Requires:** CPU, cached activations.
**Would change:** If peak r is at k near 60, F-9's CoE-60 ↔ global-linear-dim equivalence gains independent confirmation and the ~60-dim manifold becomes a portable framing for future write-ups. If peak is far from 60 (e.g. k < 20 or k > 120), F-9 is reframed as MATH-500-specific rather than reflecting a universal manifold dimension.
**Blocks:** nothing.

---


### H-480: Gaussian depth-schedule probe outperforms single-layer L19 for Qwen-2.5-1.5B correctness AUROC
**Priority:** HIGH
**Motivated by:** 2512.07667 + F-2 + F-9
**Test:** Compute per-layer contrastive DoM `d_l` from cached pathway-11 H100 last-non-pad activations on Qwen-2.5-1.5B; build Gaussian-weighted multi-layer score `s(x) = Σ_l α_l⟨a_l(x), d_l⟩` with `α_l = exp(−(l−μ)²/(2σ²))`; grid `(μ ∈ {0..27}, σ ∈ {1,2,4,8,14})` under 5-fold OOF; compare against single-layer L19 OOF AUROC = 0.7731. ≤ 1h CPU.
**Requires:** existing pathway-11 H100 NPZ caches (per-layer activations on the 500-problem MATH-500 split with K=1 correctness labels)
**Would change:** if multi-layer ≥ single-layer + 0.02, F-2 ('L19 is the privileged correctness layer') and F-9 ('CoE-60 redundant with single-layer L19 DoM') both need rewriting around a multi-layer ensemble framing; if not, F-2 strengthens against a serious challenger.
**Blocks:** depends-on for any future single-layer-vs-multi-layer probing question (would inform H-1, H-5, H-13, H-21).

### H-481: Per-layer correctness DoMs `d_l` form a smooth depth family on Qwen-2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2512.07667 + F-3
**Test:** Compute pairwise `cos(d_l, d_l')` for all 28×28 layers using same cached data as H-480. Smooth = monotone decay with `|l − l'|`, not two clusters. ≤ 20min CPU.
**Requires:** same cache as H-480.
**Would change:** confirms whether F-3's prefill/final cos = 0.046 is a single entry of a continuous family (paper-favoured interpretation) or evidence of two distinct geometries (F-3's current framing). If smooth, F-3 must be reframed.
**Blocks:** nothing directly; shapes interpretation of H-13 (head attribution) and H-17 (RoPE rotation).

---


### H-482: Diameter-normalized total persistence on prefill carries correctness signal independently of supervised DoM
**Priority:** HIGH
**Motivated by:** 2512.15285 + F-2 + F-10
**Test:** Compute Persistence_0 + Persistence_1 (eq. 1) on the P11 prefill point cloud (n=500, d=1536). For per-problem ranking, use Persistence on the kNN neighborhood (k=50). Report 5-fold OOF AUROC and compare against F-2's DoM AUROC 0.7731. Run on cached prefill activations — ~2h CPU.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/` activations; gudhi or ripser.
**Would change:** *Confirm* (AUROC ≥ 0.65) → H-12 (label-free probes) has a concrete winner; F-2 framing tightened from "DoM is the signal" to "DoM is one linear read of an underlying topological-richness signal." *Reject* (AUROC ≤ 0.55) → strengthens F-2's claim that the prefill correctness signal is intrinsically supervised/linear, and F-10's null verdict generalizes beyond the within-sequence setting.
**Blocks:** Nothing.

### H-483: Per-token total persistence reproduces dimensional breathing — or refutes its dimensional framing
**Priority:** MEDIUM
**Motivated by:** 2512.15285 (eq. 2) + F-1 + H-16
**Test:** On cached P11 NPZ for one canonical problem, compute (a) per-token Persistence_0 over the 28-layer point cloud, (b) per-layer Persistence_0 over a 50-token sliding window. Compare against the F-1 PR breathing curve via Spearman correlation across tokens.
**Requires:** CPU only; cached P11 single-problem NPZ; gudhi.
**Would change:** *Correlation r > 0.7* → F-1's breathing is dimension-real, not a covariance-eigenvalue artifact; H-16 (MP correction) gets independent corroboration from a non-spectral estimator. *Correlation r < 0.5* → F-1 reframed as a covariance-spectrum phenomenon, not a manifold one; the project's "dimensional" branding gets weakened.
**Blocks:** H-16 corroboration path.

---


### H-484: Prefill L19 DoM scores are well-calibrated probabilities (not just well-ranked)
**Priority:** HIGH
**Motivated by:** 2512.16030 + F-2 + F-8
**Test:** Sigmoid-calibrate the cached 5-fold OOF prefill-DoM scores on
Qwen-2.5-1.5B MATH-500 1024-tok set. Bin into 10 equal-width bins, compute
ECE, MCE, OCR@0.7, OCR@0.9, Brier Skill Score against MATH-500 base rate
(0.486). Compare against KalshiBench's frontier-model calibration table.
~30min CPU.
**Requires:** CPU only, cached
`pathway11_h100/prefill_gated_compute/results.json`.
**Would change:** If ECE > 0.10 (worse than Claude Opus 4.5's 0.120 on
KalshiBench), F-2's "correctness predictor" framing is downgraded to
"correctness ranker"; F-8 deployment recommendations require post-hoc
Platt scaling. If ECE < 0.05, F-2's calibration property is a new finding
beyond AUROC — strengthens the deployment case substantially.
**Blocks:** Any F-8-derived selective-prediction productization.

### H-485: Prefill L19 DoM transfers to temporally-OOD forecasting questions
**Priority:** MEDIUM
**Motivated by:** 2512.16030 + F-2
**Test:** Run Qwen-2.5-1.5B on the 300-question KalshiBench v2 sample with
verbalized-confidence prompting. Extract prefill L19 activations at end of
question + description. Fit DoM on a held-out 5-fold OOF, measure AUROC,
ECE, BSS against base rate (0.40 yes). Compare to MATH-500 prefill-DoM
AUROC=0.7731. ~4h H100.
**Requires:** H100 pod for inference; KalshiBench HF dataset
(2084Collective/kalshibench-v2).
**Would change:** If AUROC > 0.65 on temporally-clean OOD questions,
F-2 generalizes beyond MATH-500 memorization — the prefill signal is
genuinely about uncertainty rather than retrieval familiarity. If AUROC
near 0.5, F-2 collapses to a "MATH-500 memorization probe" and
Pathway-11 deployment claims must be restricted to in-training-
distribution problems.
**Blocks:** H-5 (familiarity vs decomposability framing depends on this).

---


### H-486: Ridge-regularized prefill L19 mass-mean (WRMD) lifts correctness AUROC above plain DoM
**Priority:** HIGH
**Motivated by:** 2512.16602 + F-2
**Test:** Refit WRMD = (Σ_N + λI)⁻¹(μ_correct − μ_incorrect), normalize, evaluate 5-fold OOF AUROC at L19 prefill. Sweep λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1e0}. Compare against F-2's 0.7731 baseline. ~20min CPU on cached `pathway11_h100/prefill_gated_compute/` NPZs.
**Requires:** CPU only, cached prefill L19 activations + binary correctness labels.
**Would change:** If WRMD AUROC > 0.78 at any λ, F-2's "plain DoM is the strong predictor" headline becomes "plain DoM was the wrong baseline" — and the F-9 CoE-redundancy claim is reopened (CoE may capture the multi-dimensional structure WRMD captures with one ridge knob). If WRMD ≤ DoM across all λ, F-2 is robust to a strict mathematical generalization and the headline is reinforced.
**Blocks:** H-487 (must establish WRMD baseline before testing top-k SVD as a *separate* improvement vs WRMD-only).

### H-487: Correctness at L19 prefill is rank-r > 1 in residual stream
**Priority:** HIGH
**Motivated by:** 2512.16602 + F-2 + F-3
**Test:** SVD of (X_correct − μ_incorrect) at L19 prefill. Evaluate cumulative OOF AUROC using top-k singular vectors as a probe basis, k ∈ {1, 2, 4, 8}. ~10min CPU.
**Requires:** CPU only, cached prefill L19 NPZs.
**Would change:** If top-4 lifts AUROC by ≥0.02 over top-1 (F-2 DoM), then correctness is a low-rank subspace, not a single direction — F-2 and F-3 (cos = 0.046 prefill ⊥ final) need to be rephrased as "leading-component" claims, and downstream interpretability work that assumes a rank-1 axis (probe-weight steering, single-direction ablation) is mis-specified. If k=1 within ±0.005 of k=8, F-2's rank-1 framing is robust.
**Blocks:** nothing.

### H-488: Aggressive DoM steering on MATH-500 inflicts collateral damage on MMLU/HumanEval
**Priority:** HIGH
**Motivated by:** 2512.16602 + H-1
**Test:** When P10-FE1 runs (probe-weight + mass-mean + per-layer bias steering arms), evaluate MMLU-Pro Engineering and HumanEval at each α value alongside MATH-500. Report (ΔMATH, ΔMMLU, ΔHumanEval) triples per α.
**Requires:** Same H100 day as P10-FE1 + ~+50% time for off-target evals.
**Would change:** If ΔMMLU < −2pp at the α that gives ΔMATH > +1pp, H-1 succeeds-but-not-cleanly: per-position DoM moves correctness *and* breaks unrelated capabilities, mirroring García-Ferrero's WRMD result (CCP refusal 92→24, but MMLU-Pro Eng 76.88→67.70). If MMLU is flat at α with material MATH lift, H-1 is a clean accuracy lever and the steering vector is genuinely correctness-specific (not generalized "trying harder").
**Blocks:** any future "DoM steering improves MATH-500 by Xpp" claim — those headlines are not interpretable without a collateral-damage column.

### H-489: Optimal probe layer is at fixed fractional depth, not absolute index L19
**Priority:** MEDIUM
**Motivated by:** 2512.16602 + F-2
**Test:** For Qwen-2.5-7B (also 28 layers, so same fraction; the test only fires on a different-depth architecture). Use cached Phi-3 (32 layers) or Llama (32 layers) Pathway 1 activations to compute per-layer DoM AUROC. Peak layer expected at ~67% depth (= ~L21 of 32) under fractional hypothesis; at L19 absolute under absolute hypothesis.
**Requires:** CPU only, cached per-layer activations from Pathway 1 / Pathway 8 if available for Phi-3/Llama. ~30min.
**Would change:** If peak ≈ 0.67 × depth across architectures, the F-2 anchor needs to be reframed as "the ~67%-depth direction" and any cross-model transfer must rescale layer indices. If peak is at varying fractional depths with no consistent pattern, F-2 is a Qwen-2.5-1.5B-specific empirical finding without principled cross-architecture extension.
**Blocks:** nothing.

---


### H-490: Step-level mpnet persistent homology of CoT carries correctness signal independent of L19 DoM
**Priority:** HIGH
**Motivated by:** 2512.19135 + F-9 + F-10
**Test:** On the 500 cached MATH-500 1.5B K=1 traces, parse generations by `\n\n` step delimiters; embed each step with `all-mpnet-base-v2`; compute Vietoris–Rips persistent homology and extract `|H₀|, |H₁|, persistent_entropy, max_lifetime`. Score AUROC for K=1 correctness. Refit logistic regression with `(prefill_DoM_score, |H₀|, |H₁|, persistent_entropy)` and report partial-correlation t-statistics. ~30min CPU.
**Requires:** CPU only; cached 1.5B 1024-tok MATH-500 K=1 generation logs (check that step-by-step text was logged, not just final answer); `sentence-transformers`, `ripser` or `gudhi`.
**Would change:** If topology coefficients are non-zero with DoM controlled, F-9's "CoE redundant with DoM" claim is substrate-specific (within hidden states) and we have a black-box-accessible third axis. F-10's PH-at-null result also gets re-scoped to per-token residuals only. If topology coefficients vanish, F-2/F-9 stand and Li et al.'s sentence-embedding signal is parallel to but redundant with our DoM finding.
**Blocks:** H-491.

### H-491: Black-box mpnet-PH predictor matches selective-prediction performance of prefill DoM at 50% coverage
**Priority:** MEDIUM
**Motivated by:** 2512.19135 + F-8
**Test:** Train a black-box-only logistic classifier on `(|H₀|, |H₁|, persistent_entropy, max_lifetime, token_count)` for 1.5B MATH-500 K=1 correctness. Apply selective-prediction: abstain on the lowest 50% confidence; compare answered-set accuracy against F-8's 71.6% benchmark. Threshold sweep at 30%, 50%, 70% coverage.
**Requires:** CPU; output of H-490 pipeline.
**Would change:** If black-box selective-prediction reaches ≥65% answered-accuracy at 50% coverage, F-8's framing as a hidden-state-required result needs softening and there is a deployable variant for closed-source models. If it underperforms substantially (<55%), confirms hidden-state access is materially necessary.
**Blocks:** nothing.

### H-492: Final-path simplicity replicates F-4 asymmetric collapse at step-embedding granularity
**Priority:** MEDIUM
**Motivated by:** 2512.19135 + F-4
**Test:** Generate or use cached K=8 samples on MATH-500 1.5B. For each problem compute (a) `|H₀|_full` over all 8 trajectories pooled, and (b) `|H₀|_path` over the majority-vote-winning trajectory only. Test correlation with majority-vote correctness. Per Li et al., (a) should correlate positively, (b) negatively or weaker.
**Requires:** ~1h CPU if K=8 generations exist; ~4h H100 if regenerating from `prefill_gated_compute`.
**Would change:** If reversal replicates, F-4's asymmetric collapse generalizes from final-token PR to step-level semantic PH — same shape of claim, different substrate, strengthens the "correctness ≈ collapse-of-options" framing. If reversal does not replicate, Li et al.'s finding may be GPT-4o-mini-specific (closed model, RL-tuned) and not a property of base Qwen.
**Blocks:** nothing.

---


### H-493: A mid-depth layer DoM has lower ECE than L19 DoM on Qwen-2.5-1.5B prefill at matched AUROC
**Priority:** HIGH
**Motivated by:** 2512.24560 (Gros & Devanbu, Section 6.1, Table 6) + F-2
**Test:** Refit centroid-difference DoM at every cached layer from Pathway 8 on 1.5B MATH-500 prefill; compare AUROC, ECE, Platt-scaled ECE, BSS. Identify the layer minimizing ECE at AUROC ≥ 0.77.
**Requires:** CPU only (cached Pathway-8 layer activations); 1h.
**Would change:** On confirm — F-2 split into "L19 = best AUROC" + "L<k> = best calibration"; selective prediction (F-8) retrained on the calibrated layer. On reject — F-2's L19 choice is robust under calibration metrics, strengthening the headline.
**Blocks:** P11-FE620 (calibration headline metrics) is more useful if H-493 confirms.

### H-494: Prefill L19 DoM AUROC degrades sharply when transferred to a different problem distribution (GSM8K or code)
**Priority:** HIGH
**Motivated by:** 2512.24560 (Table 7 leave-one-out RepoCod-s collapse) + F-8
**Test:** Train DoM on Qwen-2.5-1.5B MATH-500 prefill activations; evaluate AUROC on cached or freshly-extracted activations from GSM8K and a small code benchmark (e.g., HumanEval+) under the same model. Threshold of interest: does AUROC drop below ~0.6?
**Requires:** ~2h H100 for OOD activation extraction; cached MATH-500 DoM.
**Would change:** On confirm — F-8's selective-prediction claim becomes domain-conditional; project pivots toward "domain-of-validity" as a first-class claim. On reject — DoM transfers, materially strengthening generality of F-2 / F-8 and contradicting the paper's RepoCod-s analogue.
**Blocks:** All cross-domain extensions (H-3 breathing on code) are conditional on H-494 outcome.

---


### H-495: Prefill DoM confidence is fixed-distribution, not evidence-conditioned
**Priority:** HIGH
**Motivated by:** 2601.00138 + F-8 + F-2
**Test:** Construct MATH-500 perturbation set (last-30%-truncated problem, numeric-values-randomized, keyword-scrambled). Re-extract Qwen-1.5B L19 prefill activations; apply unchanged DoM probe; report mean DoM, IQR(DoM), and AUROC vs correctness on each perturbation. ~5h H100 + 1h CPU.
**Requires:** H100 pod; cached probe coefficients; perturbation script.
**Would change:** If DoM mean does not contract under perturbation (Δ < 0.1σ), F-8 must be qualified as a fixed-distribution gate, not an epistemic one — and H-1 (per-position steering) should be re-scoped to condition on evidence proxies. If DoM contracts proportionally, F-8 strengthens and Ortiz's representational-overconfidence critique does not transfer to math.
**Blocks:** Any future steering / verification-routing claim that assumes prefill DoM is evidence-conditioned (H-1, H-19).

### H-496: Selective-prediction at fixed coverage masks risk inflation under shift
**Priority:** MEDIUM
**Motivated by:** 2601.00138 + F-8
**Test:** Compute F-8's curve at *fixed threshold* rather than fixed coverage. (i) Find ε* on MATH-500 hitting risk = 9%; (ii) apply on AIME 2024 prefills with same probe; (iii) report Δrisk, Δcoverage. ~4h H100 (AIME activations) + 1h CPU.
**Requires:** AIME 2024 activation cache (does not exist yet).
**Would change:** If Δrisk on AIME ≤ 5pp at fixed ε*, F-8 has an epistemic component. If Δrisk > 10pp, F-8's headline number is fixed-coverage cosmetic; the validity envelope must be reported alongside.
**Blocks:** H-19 verification routing — ROI claim depends on cross-difficulty calibration.

### H-497: Logprob-derived confidence (p_max / margin / entropy) is at most as good as final-token DoM on MATH-500
**Priority:** MEDIUM
**Motivated by:** 2601.00138 + F-2
**Test:** From cached K=1 greedy MATH-500 generations on Qwen 1.5B, extract top_logprobs at final-answer token, renormalize over the answer-letter vocabulary, compute p_max / margin / entropy AUROC vs correctness. Compare to final-token DoM AUROC 0.7186 and prefill-L19 DoM AUROC 0.7731. ~30min CPU if logprobs cached, 2h H100 if re-decode required.
**Requires:** CPU + cached logprobs (or H100 re-decode).
**Would change:** If logprob p_max AUROC ≥ 0.75, prefill-DoM's lead is much smaller than F-2 implies, and the "trained probe" cost is hard to justify. If logprob AUROC ≤ 0.6, F-2's prefill-DoM lead is robust to internal-signal critique.
**Blocks:** nothing.

---


### H-498: Zigzag-PH on attention-graph evolution beats Gaussian null on MATH-500 correctness, narrowing F-10
**Priority:** HIGH
**Motivated by:** 2601.01552 + EXP-026
**Test:** Run P11-FE626 (HalluZig pipeline replication on Qwen-2.5-1.5B 1024-tok, 500 MATH-500 problems, attention-only) and P11-FE627 (rank-matched Gaussian null on the resulting zigzag features). Compare real-zigzag-PH AUROC vs null-zigzag-PH AUROC. ~1 H100-day + ~10 CPU-hours total.
**Requires:** P11 Stage-2 pipeline modified to dump attention matrices; FastZigzag + Gudhi locally; cached MATH-500 1024-tok labels.
**Would change:** If real-zigzag-PH > null-zigzag-PH by ≥0.05 AUROC, F-10 must be narrowed to "static PH on residual streams is at null" — leaving open that *zigzag PH on attention graphs* carries genuine signal. If gap ≤ 0.02, F-10 generalizes and the HalluZig hallucination signal must be re-explained (covariance? attention-spectral features?).
**Blocks:** H-499. Also unlocks revisiting the "Zigzag persistent homology across layers" graveyard entry in HYPOTHESES.md historical/abandoned section.

### H-499: HalluZig-style attention-graph zigzag-PH AUROC ≥ 0.7731 on MATH-500, refuting residual-stream privilege
**Priority:** MEDIUM
**Motivated by:** 2601.01552 + F-2
**Test:** Same pipeline as H-498. Compare AUROC against F-2's 0.7731 (Qwen-2.5-1.5B prefill L19 DoM). Stratify by depth-cutoff (10%, 30%, 50%, 70%, 100%) for direct comparison with HalluZig's Fig. 9.
**Requires:** Same as H-498; no extra data.
**Would change:** If attention-zigzag-PH AUROC ≥ 0.7731, the implicit framing in F-2 / F-7 / F-9 that residual-stream geometry is the privileged correctness substrate is wrong. Reframes the project: attention dynamics, not residual geometry, is the carrier. If AUROC < 0.65, the residual-stream framing stands and HalluZig's result is hallucination-specific (does not port to math reasoning).
**Blocks:** Decision on whether to re-run pathway 4–9 attention-side instead of residual-side. Significantly expands re-test surface area.

---


### H-500: Label-free P(SUFFICIENT) logit probe matches or beats L19 prefill DoM on MATH-500
**Priority:** HIGH
**Motivated by:** 2601.02179 + F-2
**Test:** Apply Zhang et al.'s P(SUFFICIENT) prompt template to Qwen-2.5-1.5B's K=1 greedy MATH-500 answers; one extra forward pass per problem with the binary probe template appended; take softmax(A) as confidence; compute AUROC against cached correctness labels; compare to F-2's 0.7731. ~30 min H100. Cached prompts and labels live in `pathway11_h100/prefill_gated_compute/`.
**Requires:** Qwen-2.5-1.5B loaded on H100 (~5GB), MATH-500 prompts + correctness cache (have).
**Would change:** **Confirm** (P(SUFFICIENT) AUROC ≥ 0.77): F-2 is overstated — the load-bearing correctness signal is recoverable by a zero-training logit probe; the L19-DoM apparatus is one route to the same signal, not the unique strong one. H-12 is largely settled. **Reject** (AUROC ≤ 0.65): F-2's supervised hidden-state probe is doing real work that logit probes can't reach, strengthening the F-2 framing and weakening H-12.
**Blocks:** H-12 (this is the clean test).

### H-501: Prefill L19 DoM AUROC has a length / prompt-structure confound analogous to P(TRUE)'s
**Priority:** HIGH
**Motivated by:** 2601.02179 §5.3 placebo experiment + F-2
**Test:** Construct a length-matched uninformative-filler prefix (Zhang et al.'s placebo idiom) for each MATH-500 prompt, prepend, recompute L19 prefill DoM AUROC on the same OOF 5-fold split. ~1 h H100. Compare to 0.7731. Also rerun with a "summary"-style reformulation of each prompt and check whether DoM stays constant (content) or shifts (structure).
**Requires:** Qwen-2.5-1.5B + cached MATH-500 prompts + extraction script from `pathway11_h100/`.
**Would change:** **Confirm** (AUROC drops by ≥ 0.04 under placebo prefix): F-2 has a length-position confound that mirrors P(TRUE)'s on GUESS; the headline AUROC is partly an artifact and the prefill-vs-final-token orthogonality (F-3) needs the structural-vs-content reinterpretation. **Reject** (AUROC stable within ± 0.01): F-2 is a content signal; F-3's geometric story is robust.
**Blocks:** nothing.

---


### H-502: L19 prefill DoM is one direction in a K≈6 input-conditional library, not the single correctness axis
**Priority:** HIGH
**Motivated by:** 2601.09269 + F-9 (CoE redundancy claim)
**Test:** K=6 K-means on cached P11-H100 1.5B prefill-L19 activations.
Fit a 6-component mixture-of-linear-probes (router MLP picks 1-of-6 per
input). Compute OOF 5-fold AUROC. Compare against single-direction L19
DoM (0.7731). Threshold for confirmation: AUROC > 0.7831 (≥ +0.01).
~30 min CPU on cached NPZs.
**Requires:** CPU, sklearn K-means + a small PyTorch gating MLP, no new
data.
**Would change:** Confirm → F-9 wrong-side; CoE-60 is *not* redundant
with single-direction L19 DoM, and the right framing is "L19 contains a
small library of correctness primitives." Also strengthens F-5
(content-dependent breathing) by giving it a vector-library mechanism.
Reject → F-9 stands; the L19 signal is genuinely one-dimensional.
**Blocks:** P11-FE633; if confirmed, gates a per-cluster steering
experiment (a follow-on H-N tied to H-1).

### H-503: Optimal prefill correctness probe layer is deeper than L19 on Qwen2.5
**Priority:** HIGH
**Motivated by:** 2601.09269 + F-2 (L19 chosen empirically from a
single early P11 sweep)
**Test:** Recompute prefill DoM and OOF 5-fold AUROC at L20, L22, L25
on Qwen2.5-1.5B from cached all-layer NPZs. Threshold: ΔAUROC ≥ +0.01
over L19 = layer-locality refuted. ~20 min CPU.
**Requires:** CPU, cached all-layer hidden_states from
`pathway11_h100/prefill_gated_compute/`.
**Would change:** Confirm → F-2 anchor moves from L19 to L25 (or
nearby); the 0.7731 number is a lower bound; PAPER_INDEX entries
referencing L19 specifically need an asterisk. Reject → L19 is
genuinely optimal locally and RISER's mid-layer finding is
architecture-specific to their 7B/14B/32B regime, not the 1.5B regime.
**Blocks:** nothing.

### H-504: Single-shot prefill DoM injection at L19 lifts MATH-500 K=1 accuracy by ≥1pp on Qwen2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2601.09269 + H-1 (per-position bank framing)
**Test:** Apply our cached L19 prefill DoM as a single intervention at
the prefill-last-token activation, layer 19, scale α ∈ {0.5, 1.0, 1.5,
2.0} (RISER's αmax = 2.0). Measure greedy K=1 MATH-500 accuracy delta
vs no-intervention baseline (48.6%). ~1 h H100.
**Requires:** Qwen2.5-1.5B loaded, MATH-500 prompts, intervention hook,
no labels needed.
**Would change:** Confirm at any α → simplest steering point on Pareto
frontier; H-1's per-position justification is undercut and a new
hypothesis "single-shot prefill is enough" supersedes it. Also evidence
against F-3 orthogonality (prefill direction influences final-token
output). Reject → prefill-only injection is insufficient; H-1's
per-position bank is the right scope; F-3 stands as influence-channel
independence.
**Blocks:** Any per-position H-1 work should not start before this
single-shot baseline lands.

---


### H-505: Per-head probe ensemble outperforms single-layer linear DoM at correctness prediction
**Priority:** HIGH
**Motivated by:** 2601.13015 + F-2
**Test:** Train LR + MLP + SVC-RBF probes per head at L18, L19, L20 on Qwen-2.5-1.5B
prefill activations from cached MATH-500 generations. Take top-15 heads by single-probe
OOF AUROC; compute majority-vote ensemble AUROC and best-single-probe AUROC. Compare
against F-2's linear residual-stream DoM (0.7731). 30min H100 re-extraction + 30min
CPU probing.
**Requires:** GPU briefly for per-head re-extraction (~30min H100 or ~2h on local
2060), then CPU. Cached prompts already in `pathway11_h100/`.
**Would change:** If ensemble AUROC ≥ 0.81, F-2 needs to be reframed as "linear DoM is
*a* strong predictor, but multi-head ensemble adds ~0.05 AUROC". If ensemble AUROC
≤ 0.78, F-2's linearity holds at residual level (head-level decomposition adds nothing
beyond residual aggregate). Either result is informative.
**Blocks:** H-506, H-75.

### H-506: Head-level fixed-vector steering moves MATH-500 K=1 accuracy
**Priority:** HIGH
**Motivated by:** 2601.13015 + H-1 + F-3
**Test:** With top-15 heads from H-505, install a forward hook on each head's
output-projection input that adds α · θ\_h to head outputs (θ\_h = unit-normalized
DoM at that head). Sweep α ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}. Run K=1 greedy
generation on MATH-500 with intervention enabled. Compare accuracy vs baseline 48.6%.
1 day H100.
**Requires:** H100 day; MeltRTL-style intervention pipeline in `pathway11_h100/`.
**Would change:** If accuracy increases to ≥ 50.5% at any α, fixed-vector steering
*can* work at the head level — H-1's per-position assumption is a residual-stream
artifact, not a fundamental constraint, and H-1 itself becomes lower-priority. If
accuracy is flat or worse at all α values, MeltRTL's claim does not transfer to the
1.5B + MATH-500 regime, and the rotation-blocks-fixed-steering story (H-1) is
strengthened.
**Blocks:** H-1 priority re-evaluation.

---


### H-507: Pooled-position contrastive activation mean (Eq 22 with no prefill/final split) recovers AUROC ≥ 0.77
**Priority:** HIGH
**Motivated by:** 2601.14004 §4.3 Eq 22 + F-3
**Test:** Compute v = mean(L19_correct@all_positions) − mean(L19_incorrect@all_positions) on cached `pathway11_h100/prefill_gated_compute` NPZs. Measure 5-fold OOF AUROC. Compare to F-2's 0.7731 (prefill-only) and 0.7186 (final-only). 30min CPU.
**Requires:** CPU only, cached NPZs.
**Would change:** Confirm → F-3's orthogonality is *not* a sign of "two computations"; it's that pooling across positions averages out a position-coherent direction. The "prefill is special" story collapses. Reject → F-3's orthogonality is robust to pooling, and the survey's single-direction LRH framing is wrong for correctness specifically.
**Blocks:** revision of F-2, F-3 phrasing.

### H-508: Per-token correctness AUROC has a "refusal-cliff" shape (high through CoT, drop near final)
**Priority:** HIGH
**Motivated by:** 2601.14004 §5.1.1 (Yin et al. 2025 refusal-cliff) + F-3
**Test:** Train L19 logistic probes on each cached token-position activation through generation. Plot AUROC vs token-position. Look for plateau through CoT followed by sharp drop near final tokens (a la Yin's refusal cliff). 1h CPU on cached per-token NPZs.
**Requires:** CPU only, cached per-token L19 NPZs from Stage 2.
**Would change:** Confirm → F-3's orthogonality should be re-framed as "final-stage circuit annihilates the prefill direction" rather than "two separate computations". H-13 (head-level attribution) becomes much sharper: which heads in the last ~10 tokens kill the signal? Reject → orthogonality is genuine throughout, F-3 stands as written.
**Blocks:** H-13 sharpening.

### H-509: SAE-feature steering vectors (Eqs 23-24) outperform single-layer DoM at predicting correctness
**Priority:** MEDIUM
**Motivated by:** 2601.14004 §3.1 + §5.2.3 (Galichin et al. 2025) + F-9
**Test:** Train (or use a released) Qwen-2.5-1.5B L19 SAE. Compute differential activation δⱼ on correct/incorrect, build steering vector v = Σⱼ δⱼ · fⱼ. Measure AUROC alone and stacked with our single-layer DoM. ~1 H100 day (SAE) + 1h CPU.
**Requires:** H100 day if no released Qwen SAE; otherwise CPU only.
**Would change:** Confirm → F-9's "CoE-60 redundant" claim weakens (a different feature basis is non-redundant). Probe-tier methodology gets richer. Reject → F-9 generalizes from CoE to SAE: single-layer DoM captures essentially all linearly-decodable correctness information at L19.
**Blocks:** nothing.

---


### H-510: Ridge-tuned logistic L19 prefill probe beats DoM AUROC at matched data
**Priority:** HIGH
**Motivated by:** 2602.00158 + F-2
**Test:** Refit RAPTOR (ℓ2-logistic, λ tuned on val split, normalized weights) on cached Qwen2.5-1.5B L19 prefill activations against K=1 MATH-500 correctness. 5-fold OOF AUROC. ~20 min CPU.
**Requires:** CPU only; cached prefill NPZ from Pathway 11.
**Would change:** Confirm ⇒ F-2's 0.7731 is a DoM-estimator artifact, headline AUROC revised upward, F-9's "CoE-60 redundant with single-layer DoM" must be re-checked against the stronger probe. Reject ⇒ DoM is somehow privileged for correctness signals — a surprising result that itself merits investigation (e.g. concept-as-mean is the right inductive bias for binary correctness).
**Blocks:** H-511, H-512.

### H-511: DoM directional stability under data ablation is below the F-3 orthogonality cushion
**Priority:** HIGH
**Motivated by:** 2602.00158 + F-3
**Test:** K=20 × 20%-drop ablations on cached prefill / final-token Qwen1.5B activations. Compute mean absolute pairwise cosine for DoM. Compare to |F-3's cos(prefill, final)| = 0.046. ~1 h CPU.
**Requires:** CPU only; cached activations.
**Would change:** Confirm (DoM ablation cosine well above 0.046, e.g. ≥ 0.5) ⇒ orthogonality claim survives. Reject (DoM ablation cosine ≈ or below 0.046) ⇒ F-3's orthogonality is dominated by estimator noise and must be re-stated as "non-aligned within DoM's own noise floor"; the qualitative finding "prefill and final encode different information" weakens.
**Blocks:** nothing.

### H-512: RAPTOR+GCAV steering at L19 prefill moves Qwen-1.5B MATH-500 K=1 accuracy
**Priority:** HIGH
**Motivated by:** 2602.00158 + H-1
**Test:** Inject RAPTOR-L19 prefill concept vector with GCAV per-sample α (target p₀ = 0.99 for "correct" probe) on Qwen2.5-1.5B MATH-500. Compare K=1 accuracy to baseline and to DoM+fixed-α. ~2 h H100.
**Requires:** H100; RAPTOR concept vector from H-510.
**Would change:** Confirm ⇒ probe-then-steer is operational; H-1's spec rewrites around RAPTOR+GCAV; F-2 elevates from "predictor" to "intervention direction." Reject (steering is null on MATH-500 even with the strongest probe-then-steer protocol) ⇒ H-1's prior of "the prefill direction is causally relevant" is false — the L19 prefill DoM/RAPTOR direction is correlational only, and the steering pathway is closed.
**Blocks:** H-1.

---


### H-513: Prefill-DoM AUROC at L19 is largely an attention-sink-mass surrogate
**Priority:** HIGH
**Motivated by:** 2602.01893 + F-2
**Test:** Run P11-FE651. Regress prefill_DoM_projection on sink_mass at L19 across the 500 MATH-500 prompts. Compute (i) R² of sink_mass alone, (ii) partial AUROC of prefill_DoM for K=1 correctness after partialling out sink_mass.
**Requires:** Per-head L19 attention weights at prefill last-token (depends on P11-FE649 re-extraction) + 1.5B Qwen + cached MATH-500 K=1 labels.
**Would change:** Confirm → F-2's mechanism story collapses; the prefill DoM signal should be re-described as a sink-attention surrogate, with implications for F-8 selective prediction. Reject → F-2 stands as a content-dependent computed-correctness signal.
**Blocks:** Promotion of any "prefill DoM = computed confidence" framing in the writeup.

### H-514: Cos(prefill_DoM, final_DoM) is a head-type-partition artifact, not a functional orthogonality
**Priority:** HIGH
**Motivated by:** 2602.01893 + F-3
**Test:** Run P11-FE652. Within each of {Retriever, Mixer, Reset} L19 head-type subspaces, compute cos(prefill_DoM, final_DoM). Compare against the unrestricted 0.046.
**Requires:** Head-type labels at L19 (from P11-FE650) + per-head value states at L19 prefill + final positions.
**Would change:** Confirm (any within-subspace cos > 0.5) → F-3 needs to be re-stated. Reject (all within-subspace cosines remain near 0) → F-3's orthogonality is a true functional-direction claim, not a head-type artifact.
**Blocks:** H-13 (the head-level attribution will bottom out at the same answer).

### H-515: Top-N=2 attention-restricted L19 aggregate beats the full residual stream at K=1 correctness AUROC
**Priority:** HIGH
**Motivated by:** 2602.01893 Theorem 1 + Theorem 2
**Test:** Run P11-FE653. Sweep N ∈ {1, 2, 3, 4, 8, 16, L}. At each N, re-fit DoM on the L19 prefill last-token aggregate restricted to top-N attention weights. Compute K=1 correctness AUROC.
**Requires:** Per-head L19 attention weights and value states at prefill last-token (P11-FE649).
**Would change:** Confirm → AUROC peaks at small N, replicates the paper's small-N prediction on Qwen-2.5-1.5B and gives us a free improvement on F-8 selective prediction. Reject → AUROC is monotone in N, suggesting that the value-state-space picture does not transfer to residual-stream-space DoM at L19.
**Blocks:** Any future selective-prediction work that wants to use the cheapest possible aggregate at L19.

### H-516: D-bucket geometric signature (F-7) sharpens at top-N=1–4 and dissolves at N=L
**Priority:** MEDIUM
**Motivated by:** 2602.01893 + F-7
**Test:** Run P11-FE654. Compute the D-bucket vs other-bucket separability under top-N restriction for N ∈ {1, 2, 4, L}.
**Requires:** P11-FE649 outputs + F-7 D-bucket assignments.
**Would change:** Confirm → F-7 re-derived in small-N regime, with margin-aware bounds. Reject → F-7 holds in the residual-stream / large-N regime, weakening 2602.01893's claim that intermediate-N is uniformly margin-noise.
**Blocks:** nothing.

---


### H-517: Prefill L19 DoM measures *answerability*, not latent planning
**Priority:** HIGH
**Motivated by:** 2602.02103 + F-2 + H-5
**Test:** Train Tele-Lens-style rank-256 low-rank bottleneck final-answer probe at
L19 prefill on cached Qwen-2.5-1.5B MATH-500 activations (FE73). Compare AUROC
against the 0.7731 linear-DoM number. If the trained nonlinear probe is also at
0.7731 or lower, the prefill subspace does not contain answer-encoding information
in the Tele-Lens sense, and F-2's signal must be reinterpreted (problem hardness,
familiarity, decomposability, length prior).
**Requires:** 30min H100 + 1h CPU; cached `prefill_gated_compute` activations + gold
answer extraction; rank-256 adapter implementation.
**Would change:** Confirms → F-2 framing must shift from "prefill predicts correctness"
to "prefill predicts answerability"; Refutes → our linear DoM is *already* recovering
nearly all the planning-relevant subspace, which is itself surprising given Tele-Lens's
myopic claim and a paper-worthy finding.
**Blocks:** H-5 (familiarity framing depends on resolution).

### H-518: Pivot-token aggregation beats single-layer L19 DoM (refutation of F-9)
**Priority:** HIGH
**Motivated by:** 2602.02103 + F-9 + F-8
**Test:** On cached MATH-500 trajectories, compute per-token final-answer entropy
from saved logprobs; pick 5 lowest-entropy positions per trajectory; aggregate L19
DoM at those positions; OOF-CV AUROC for correctness (FE74). Sweep K ∈ {1, 5, 10, 20}.
**Requires:** 20min CPU; cached generation logprobs from
`pathway11_h100/prefill_gated_compute`; existing L19 hidden states at all positions
(may require re-extraction for non-prefill positions).
**Would change:** Confirms (pivot AUROC > 0.7731) → F-9 collapses; CoE-60 was the
right neighborhood with the wrong aggregator; F-8 selective-prediction headroom rises.
Refutes (pivot ≤ 0.7731) → single-position DoM already captures the available
signal; the Tele-Lens pivot effect is specific to their multi-class final-answer head.
**Blocks:** H-19 (verification routing budget includes early-termination if FE76 lands).

---


### H-519: L19 prefill DoM steering on MATH-500 obeys the Xu et al. three-stage curve
**Priority:** HIGH
**Motivated by:** 2602.02343 + F-2 + H-1
**Test:** Sweep `m ∈ [-30, 30]` along the L19 prefill DoM on Qwen-2.5-1.5B, regenerate K=1 MATH-500 outputs, and fit Eq. 15 (preference log-odds) and Eq. 17 (utility log-odds). Operationally identical to P11-FE661. Estimated time: 8h H100 + 2h CPU.
**Requires:** H100 for K=1 regeneration at ~10 m-values × 500 problems; cached prefill activations exist already. Polarity-paired completions for MATH-500 (assemble from existing distractor pools or generate via a small LLM).
**Would change:** **Confirm** → MATH-500 accuracy is monotone-decreasing in |m| beyond a small linear regime; H-1's "moves accuracy" claim must be reformulated as "moves the preference axis with budgeted utility cost." **Reject** → if accuracy *peaks* at non-zero m, the paper's universal three-stage law fails on a reasoning task and F-2 lives in a different regime than attribute steering.
**Blocks:** definitively answers H-1.

### H-520: A SPLIT-trained L19 prefill direction exceeds F-8's 71.6% selective-prediction accuracy
**Priority:** MEDIUM
**Motivated by:** 2602.02343 + F-8 + F-9
**Test:** Train a steering-vector trainer with the SPLIT loss (Eq. 20: utility CE on both polarities + hinge-margin preference loss) on cached Qwen-2.5-1.5B L19 prefill activations and MATH-500 polarity pairs. Evaluate selective-prediction accuracy at 50% coverage. Operationally identical to P11-FE662. Estimated time: 1d H100.
**Requires:** H100 day for training; polarity-paired completions (shared with H-519).
**Would change:** **Confirm** → joint preference+utility training is the right way to extract a correctness probe, and F-8's 71.6% is sub-optimal under naive DoM. **Reject** → DoM is already at the achievable ceiling for selective prediction at this layer, and the paper's SPLIT advantage is task-specific to attribute control.
**Blocks:** future work on probe-training methodology.

### H-521: F-3's prefill/final orthogonality (cos = 0.046) is an artifact of per-position centering
**Priority:** HIGH
**Motivated by:** 2602.02343 + F-3 + F-2
**Test:** Fit Xu et al.'s Eq. 15 with one shared preference direction across prefill and final-token L19 activations on Qwen-2.5-1.5B for the "correctness" concept; report per-position R². Operationally identical to P11-FE663. Estimated time: 4h CPU.
**Requires:** cached L19 prefill + final-token activations; polarity-paired completions.
**Would change:** **Confirm** (R² > 0.95 at both positions) → F-3's orthogonality is metric-construction-dependent, not a geometric truth, and the dual-probe stack collapses to a single trained direction. **Reject** (R² drops at one position) → F-3 is robust and the paper's universal-direction assumption fails on reasoning tasks.
**Blocks:** the dual-probe selective-prediction story rests on F-3.

---


### H-522: Sparse top-k AU subset of L19 prefill matches or beats full L19 DoM AUROC (0.7731)

**Priority:** HIGH
**Motivated by:** 2602.04428 + F-2
**Test:** Compute activation momentum `s_i = max(r_pos, r_neg)` per residual dimension at L19 prefill on MATH-500 correct/incorrect pairs; refit logistic probe on top-k ∈ {5, 20, 50, 100} AUs; 5-fold OOF AUROC. Cached `pathway11_h100/extracted/qwen15b_*l19*.npz`. ~30 min CPU.
**Requires:** Existing P11 H100 cache only; no new generation.
**Would change:** Confirm → F-2 reframed from "the L19 DoM direction" to "a sparse L19 AU subset"; selective-prediction (F-8) ceiling shifts up; H-1 steering target changes from full-vector to top-k AUs. Reject (no k beats 0.7731) → AUSteer's heterogeneity claim does not transfer to long-form math correctness; F-2's coarse-direction framing survives.
**Blocks:** H-1 redesign (P11-FE667).

### H-523: Prefill / final-token "orthogonality" (cos = 0.046, F-3) is a mean-pooling artifact over partially-overlapping AU subsets

**Priority:** MEDIUM
**Motivated by:** 2602.04428 + F-3
**Test:** Compute top-100 momentum AUs for prefill→correctness and final-token→correctness contrasts at L19 on the same MATH-500 cache. Report Jaccard overlap and sign correlation of `s_i` scores. ~1h CPU.
**Requires:** Existing P11 H100 cache only.
**Would change:** Confirm (Jaccard ≥ 0.5, sign-correlated) → F-3 orthogonality is a magnitude-cancellation artifact, not evidence of two separate sub-circuits; downgrades the geometric story. Reject (Jaccard < 0.1) → F-3 reaffirmed at AU level (stronger).
**Blocks:** H-13 head-level attribution design.

### H-524: AU-level multiplicative steering moves MATH-500 accuracy where CAA-style block injection does not

**Priority:** HIGH
**Motivated by:** 2602.04428 + H-1 + F-2
**Test:** Three-arm intervention on MATH-500 K=1 (Qwen-2.5-1.5B): Arm A full-vector additive (CAA), Arm B AUSteer multiplicative on top-50 L19 AUs, Arm C Mixed Combination (B + 10 detrimental AUs). Sweep α. ~4h H100.
**Requires:** H100 pod for generation with interventions; cached top-k AU ranking from H-522.
**Would change:** Confirm (B > A and C < B) → AUSteer's "less is more" replicates on math correctness; H-1's full-vector plan was wrong unit. Reject (A ≈ B) → block-vs-AU granularity does not matter for our regime — supports CAA-style steering as the canonical intervention.
**Blocks:** H-1 (replaces it as the steering experiment of record if confirmed).

---


### H-525: The prefill L19 DoM is computed by a sparse (≤4-head) circuit at L19, not distributed across most heads
**Priority:** HIGH
**Motivated by:** 2602.04521 + F-2
**Test:** Output-side per-head ablation at L19 on Qwen-2.5-1.5B MATH-500 prefill activations; rank 32 L19 heads by attribution to prefill-DoM read-out; check whether top-4 heads account for ≥80% of the projection magnitude (sparse) vs. spreading across ≥16 heads (distributed). 2h H100, cached activations.
**Requires:** H100 (cached prefill activations from `pathway11_h100/`), Qwen-2.5-1.5B with TransformerLens or HF hooks.
**Would change:** Confirm → unlocks H-13 head-level attribution and a credible static weight-edit path (FE74). Reject → F-2's signal is distributed, head-level circuit story fails, and the L19 DoM framing should be replaced by a layer-band framing.
**Blocks:** H-526, P11-FE672.

### H-526: Prefill-DoM and final-token-DoM read-outs share their top-attributed L19 heads despite cos=0.046
**Priority:** MEDIUM
**Motivated by:** 2602.04521 + F-3
**Test:** Run H-525's per-head attribution twice — once for prefill-DoM read-out, once for final-token-DoM read-out — on the same 500 MATH-500 problems. Compute Jaccard over top-8 heads. ≥0.3 Jaccard = shared circuit; <0.1 = independent circuits.
**Requires:** H100, depends on H-525 infrastructure being in place (~1 additional H100 hour).
**Would change:** Confirm → F-3's "two-mechanism" interpretation is wrong; orthogonality is a coordinate artefact and prefill/final-token DoM share machinery. Reject → F-3 strengthened from "orthogonal directions" to "orthogonal circuits".
**Blocks:** nothing direct, but reshapes any further DoM framing.

---


### H-527: ESR re-orthogonalization explains the cos(prefill, final) = 0.046 baseline
**Priority:** HIGH
**Motivated by:** 2602.06941 + F-3
**Test:** On cached Pathway 10 L19 activations, compute cos(prefill_DoM, final_DoM) for correct and incorrect MATH-500 problems separately. Compare to the known 0.046 aggregate. ~30min CPU. If both classes show cos ≈ 0.046, the orthogonality is a generic property of the consistency circuit, not a correctness-channel signal. If they differ by ≥0.05, F-3's two-channel framing survives the refutation.
**Requires:** CPU; cached `scratch/pathway10_temporal_and_verifier_results.json` upstream activations.
**Would change:** On confirm (uniform cos), F-3's interpretation downgrades from "two-channel correctness encoding" to "consistency-circuit orthogonality + a separate prefill correctness signal." On reject (correctness-dependent cos), F-3 survives intact.
**Blocks:** P11-FE673 directly tests this.

### H-528: D-bucket signature (F-7) is partly ESR-circuit deficiency, not pure intrinsic difficulty
**Priority:** MEDIUM
**Motivated by:** 2602.06941 + F-7
**Test:** Partition cached MATH-500 prefill activations by F-7 bucket (A, B, C, D). Compute prefill-DoM projection magnitude and cos(prefill_DoM, final_DoM) per bucket. If D-bucket shows lower projection magnitude AND/OR systematically different cosine vs A-bucket, that's consistent with weaker consistency-circuit engagement on D-bucket problems. ~1h CPU.
**Requires:** CPU; F-7 bucket assignment file + Pathway 10 cached NPZs.
**Would change:** On confirm, F-7's "intrinsic difficulty" framing becomes "low-consistency-circuit-engagement signature" — different downstream story (interventions could target the circuit, not the problem). On reject, F-7's intrinsic interpretation survives.
**Blocks:** P11-FE674.

---


### H-529: Compositional steering over a low-dim concept basis beats single-direction DoM on MATH-500 correctness prediction
**Priority:** HIGH
**Motivated by:** 2602.07276 + F-2 + F-3
**Test:** (1) Build a 5-vector REP basis on Qwen-2.5-1.5B from Big-Five
contrastive prompts at L19 (~30min A6000). (2) Run Bayesian Optimization
(Matern-5/2 GP, EI acquisition, 50 Sobol + 350 BO iters, 5 seeds, search
bounds [-2,2]^5) over `{prefill_L19_DoM} ∪ {O,C,E,A,N}` with a
12-example balanced calibration split. (3) Score AUROC on the held-out
488 MATH-500 problems. Predicts: composed AUROC >= 0.79 in 5/5 seeds.
**Requires:** Cached `pathway11_h100/prefill_gated_compute/results.json`
+ Qwen-2.5-1.5B loaded (single A6000 sufficient, no H100). ~1 hour
total.
**Would change:** If composed AUROC >= 0.79, F-2's "single direction"
framing is incomplete — DoM is one useful basis vector, not *the*
correctness signal. If composed AUROC ≤ 0.7731 + noise, F-2 stands and
F-9-style "signal lives in a 1-D subspace" generalizes from CoE-60 to
the full Big-Five basis.
**Blocks:** nothing.

### H-530: Stability-aware objective J(α) = gain − λ·flip is a CPU-only kill-switch for naive H-1
**Priority:** CRITICAL
**Motivated by:** 2602.07276 + H-1
**Test:** Score `J(α) = Σ_err G_gain(x;α) − Σ_corr [λ_flip·I_flip(x) +
λ_drop·I_drop(x)]` with `λ_flip=20, λ_drop=10` for any candidate α
implied by H-1 (per-position DoM addition with sweep α ∈ {-1, -0.5, 0.5,
1, 2}). Compute log-prob shifts under activation addition offline using
cached residuals + logit lens — no new generation needed. Predicts: at
every α magnitude that produces nonzero accuracy gain on B-bucket
errors, `Σ_corr I_flip` exceeds `|B_err|` (more A-bucket flips than
B-bucket fixes).
**Requires:** Cached prefill activations for 500 MATH-500 problems
(have) + per-token log-probs (cached in P11 results). CPU only, ~30
min.
**Would change:** If J(α) < 0 for all naive sweep values, naive H-1
settles negative without ever needing an H100 pod, and H-1 reformulates
to "compositional steering with stability-aware BO" (i.e. H-529).
If J(α) > 0 for some α, naive H-1 survives offline screening and
deserves the H100 run.
**Blocks:** P10-FE1.

### H-531: A 12-example balanced calibration set recovers most of F-2's prefill DoM AUROC
**Priority:** MEDIUM
**Motivated by:** 2602.07276 (data-efficiency claim) + F-2
**Test:** Replace the OOF 5-fold (∼400-example folds) currently used to
report 0.7731 with a 12-example balanced (6 correct + 6 incorrect)
calibration set; report mean ± std AUROC across 50 random calibration
draws on the held-out 488. Predicts: 12-example mean AUROC >= 0.74 (i.e.
within 0.03 of 0.7731), consistent with Steer2Adapt's data-efficiency.
**Requires:** Cached results.json. CPU, ~10 min.
**Would change:** If 12-example AUROC >= 0.74, F-2 generalizes the
data-efficiency claim and we can drop the 5-fold burden in future
ablations. If 12-example AUROC ≤ 0.70, the BO refutations above need
larger calibration sets to be fair.
**Blocks:** nothing — but informs sample-size choice for H-529.

---


### H-532: Slerp-rotation toward μ\_prefill at L19 lifts MATH-500 K=1 accuracy more than additive DoM at matched intervention budget
**Priority:** HIGH
**Motivated by:** 2602.08169 + F-2
**Test:** Replicate Spherical Steering Eq. 9-15 on Qwen-2.5-1.5B / MATH-500 at L19 with α∈[0.2, 0.8], β∈[-0.05, 0.3]; compare K=1 accuracy delta against additive DoM at matched effective-rank drop. Estimated 4 h H100 + 1 h CPU.
**Requires:** H100 + cached prefill DoM + existing P11 forward-pass hooks
**Would change:** if confirmed, supersedes additive H-1 with norm-preserving variant; if rejected, evidence that the paper's recipe is TruthfulQA-specific.
**Blocks:** H-533 (the antipodal test feeds the gate calibration), H-172 (the durability test reuses the Slerp implementation)

### H-533: The contrastive correctness axis is *not* antipodal on MATH-500 (cos(μ\_T, -μ\_H) drifts below 0.95 even when ≈1.0 on TruthfulQA last-token)
**Priority:** HIGH
**Motivated by:** 2602.08169 + F-4
**Test:** Compute μ\_T and μ\_H separately on Qwen-2.5-1.5B / MATH-500 last-token L19; report cos(μ\_T, -μ\_H). Repeat at prefill and at decoding step 100. ~30 min CPU.
**Requires:** cached P11 NPZs only
**Would change:** if cos << 1 on math but ≈ 1 on TruthfulQA, the paper's vMF gate equations need replacement with a non-antipodal form before any reasoning-task transfer; if cos ≈ 1 holds, the paper's symmetry assumption is validated and we should adopt it.
**Blocks:** nothing (pure measurement)

---


### H-534: Inference-time residual-stream trajectory has non-null geometry detectable by random-basis projection control
**Priority:** HIGH
**Motivated by:** 2602.10496 + F-10
**Test:** P11-FE683. Treat per-token L19 sequence as a trajectory; compute PCA top-K basis on stacked step-to-step deltas; report ρ_exec/ρ_rand split by correct/incorrect and token-position band. 1h CPU on cached NPZs.
**Requires:** CPU; cached `pathway11_h100/prefill_gated_compute/extraction_*.npz`; numpy.
**Would change:** If ρ_exec/ρ_rand > 1.5 in any band, F-10's "geometry at the null" framing is invariant-specific (PH-only) and we have a complementary trajectory-geometry signal worth surfacing in the headline narrative. If ratio is at 1.0, F-10's null framing generalizes beyond PH.
**Blocks:** nothing.

### H-535: Qwen breathing magnitude depends on MLP residual contribution, not attention only
**Priority:** HIGH
**Motivated by:** 2602.10496 §4.5 + F-1
**Test:** P11-FE684. Forward-pass Qwen-2.5-1.5B on 50 MATH-500 problems with L19 MLP output zeroed; compute temporal PR; compare against PR=67 baseline and against L18/L20 MLP-ablation controls. ~4h H100.
**Requires:** H100 pod; 50-problem subset is sufficient given the magnitude of expected effect; existing extraction harness with hook on `model.layers[19].mlp` output.
**Would change:** If MLP ablation drops PR peak by >20%, F-1's cross-architecture universality is misscoped — breathing is partly MLP-driven. If unchanged, F-1's universality is robust and the paper's MLP-disruption framing is specific to attention-only training dynamics.
**Blocks:** scoping decisions for H-3 (code generation) and H-22 (Abstract-CoT).

### H-536: Prefill correctness signal lives in a 3-4 dim subspace, not a single direction
**Priority:** MEDIUM
**Motivated by:** 2602.10496 + F-2 + F-9
**Test:** P11-FE685. Top-K PCA logistic regression sweep on cached prefill L19 activations, K ∈ {1, 2, 3, 4, 5, 8, 16, 32}; 5-fold OOF AUROC at each K with random-basis control. 1h CPU.
**Requires:** CPU only; cached prefill activations.
**Would change:** If AUROC at K=4 is ≥+0.02 over K=1 (and >+0.01 over a random-K-basis control), F-2's single-direction framing is incomplete and DoM is one coordinate of a small execution manifold. If K=1 ≈ K=16, F-2 is vindicated and the paper's "distributed execution" picture does not transfer to inference.
**Blocks:** P10-FE5 SAE interpretation (multi-direction story would change which SAE features to inspect).

---


### H-537: Correctness lives in a small layer subset (L18–L20), not L19 specifically
**Priority:** HIGH
**Motivated by:** 2602.11910 + F-2
**Test:** Refit OOF 5-fold prefill DoM probe on cached 1.5B activations, concatenate layers across all (k choose layers) of {L17, L18, L19, L20, L21}, plus broader sweeps. Report best-k AUROC. ~2h CPU.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/*.npz`.
**Would change:** If best-k > 1 ensemble beats single-layer L19 by ≥0.02 AUROC, F-2 reframes as "small mid-layer subset" — joining TADA's universality claim. If single-layer L19 is genuinely best, F-2 holds and we can argue L19 is *more* concentrated than TADA's domain (which always finds 2-4).
**Blocks:** Reframing of F-9 (CoE redundancy) — if F-2 is a layer subset, F-9 may need re-examination.

### H-538: TADA-style ReNorm-CAA at prefill-only moves Qwen-1.5B MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2602.11910 + EXP-026 (P10 v1 fixed-vector kill)
**Test:** Compute v_c from L19 prefill DoM, apply at prefill only with ReNorm (h' = (h + α·v_c) · ||h||/||h + α·v_c||), sweep α ∈ {−100, …, 100}, measure K=1 accuracy on MATH-500.
**Requires:** H100 ~6h (model load + α sweep), one cached vector (already have).
**Would change:** If accuracy moves monotonically with α, the H-1 "per-position bank" framing is overkill — a single prefill ReNorm-CAA suffices, and the "DoM rotates" story reduces to a generation-time-only problem. If accuracy is flat, the rotation story holds and H-1's per-position bank stands.
**Blocks:** H-1 priority — if H-538 succeeds, H-1 demotes.

### H-539: TopK SAE on L19 prefill recovers a sparse correctness feature with AUROC > 0.7731
**Priority:** HIGH
**Motivated by:** 2602.11910 + F-2 + F-9
**Test:** Train TopK SAE (m=4, k=64) on 500 cached 1.5B L19 prefill activations from MATH-500. Compute TF-IDF feature score (TADA Eq. 6) with (correct, incorrect) as the contrast pair. Pick top-1 feature, compute correctness AUROC. Compare to L19 DoM 0.7731 and CoE-60 (~0.795).
**Requires:** CPU 6h SAE training; cached activations.
**Would change:** If top-1 SAE feature AUROC > 0.7731 + 0.02, the "single dense direction" frame in F-2 is wrong. If a 3-feature subset beats CoE-60, F-9's redundancy claim flips ("CoE is coarse, SAE is finer").
**Blocks:** H-12 (semantic-entropy probes) — H-539 is a stronger version of the SAE-replaces-DoM claim.

### H-540: A single SAE feature discriminates D-bucket from A-bucket
**Priority:** MEDIUM
**Motivated by:** 2602.11910 + F-7
**Test:** Reuse the TopK SAE from H-539. Re-score features with TF-IDF on (D-bucket prefill, A-bucket prefill). Pick top-1, threshold on out-of-fold, compute hit-rate / FPR.
**Requires:** Minutes CPU after H-539.
**Would change:** If hit-rate > 50% with FPR ≤ 20%, F-7's "no per-problem signature" is refuted — D-bucket has a sparse pre-generation marker we missed. If no feature exceeds chance, F-7 holds against the strongest tool we have.
**Blocks:** Nothing.

### H-541: Activation-patching at L19 produces larger correctness shifts than at any other layer
**Priority:** MEDIUM
**Motivated by:** 2602.11910 + F-2
**Test:** For 50 paired MATH-500 problems with divergent (correct vs incorrect) outcomes, cache K/V on correct-prefill, patch one layer at a time on incorrect-prefill (and vice versa), measure post-generation correctness flip rate per layer.
**Requires:** H100 4–6h; ~2800 generations.
**Would change:** If L19 patches dominate, F-2 gains causal evidence. If a different layer dominates (e.g. L23 attention), F-2's correlational story misidentified the bottleneck.
**Blocks:** Future planning of head-level attribution (H-13 should go to whichever layer wins H-541).

---


### H-542: Multi-layer interleaved DoM ensemble beats single-layer L19 DoM by ≥ 0.01 AUROC
**Priority:** HIGH
**Motivated by:** 2602.13567 (DistillLens Table 4: interleaved 5-layer beats single mid-layer by +0.5 R-L, ~14% of total multi-layer gain) + F-2
**Test:** Train 5-fold OOF DoM probes on Qwen2.5-1.5B cached prefill activations at single-L19 (baseline 0.7731) vs interleaved-{4,9,14,19,24}. ~30 min CPU.
**Requires:** CPU; cached pathway11_h100 prefill NPZs.
**Would change:** Confirm → F-2 narrative becomes "L19 is a strong probe site, but 5-layer interleaved is the actual SOTA"; the current selective-prediction headline at AUROC 0.7731 (and 71.6% acc at coverage 0.5 in F-8) becomes a soft floor. Reject (Δ < 0.01) → F-2's privileged-layer claim is reinforced and matches DistillLens's own observation that single mid-layer captures 86% of the gain.
**Blocks:** H-11 framing (if multi-layer is materially better, distillation target should be a layer-stack not a single layer).

### H-543: Prefill-L19 and final-L19 logit-lens distributions are NOT orthogonal in vocab space (mean per-problem JSD < 0.3)
**Priority:** HIGH
**Motivated by:** 2602.13567 (DistillLens succeeds aligning early/late layer distributions via vocab-space JSD; if these were orthogonal, alignment would be impossible) + F-3 (cos = 0.046)
**Test:** Compute logit-lens projection p^(L19)_prefill and p^(L19)_final using Qwen tied embedding as WU on cached Qwen2.5-1.5B activations. Compute per-problem JSD; report mean and distribution. ~1h CPU.
**Requires:** CPU; cached pathway11_h100 prefill + final-token NPZs; Qwen2.5-1.5B tokenizer + tied embedding.
**Would change:** Confirm (JSD < 0.3) → F-3's orthogonality is a per-class-direction artifact; broader distributional structure is preserved across the prefill→final transition. This refines the "orthogonal" narrative to "the correctness *direction* rotates, but the distributional support overlaps." Reject (JSD > 0.5) → F-3 strengthens to "orthogonality is fundamental, not just direction-level."
**Blocks:** H-11 (if even within-model prefill→final distributions diverge, cross-model 7B→1.5B distillation of the prefill signal is harder).

---


### H-544: Prefill and final-token DoM are aligned in dual (information-geometric) coordinates despite primal orthogonality
**Priority:** HIGH
**Motivated by:** Park et al. 2602.15293 + F-3 (cos = 0.046 in primal coords)
**Test:** Compute dual transform $\phi(\lambda) = \mathrm{softmax}(W_U \lambda) W_U$ at L19 using Qwen-2.5-1.5B `lm_head.weight` as the readout. For each cached MATH-500 problem, compute prefill_DoM_dual and final_DoM_dual as differences of class means in dual coords. Recompute cosine. Predicts cos_dual > 0.3 (i.e., directions are aligned in the natural geometry; the 0.046 was a coordinate artifact). Run on cached `pathway11_h100/prefill_gated_compute` NPZs. ~3h CPU.
**Requires:** CPU only, cached NPZs, `lm_head.weight` from Qwen-2.5-1.5B (already in HF cache from Pathway 11 extraction).
**Would change:** If cos_dual > 0.3, F-3's "orthogonal directions" headline becomes "orthogonal in primal, aligned in dual" — major reframing of the prefill-vs-final-token story (they encode similar things, but the residual stream stores them in different primal directions). Strengthens information-geometry as the right lens. If cos_dual ≤ 0.1, F-3 is geometry-invariant and the primal-coords critique doesn't apply to our setup; weakens motivation to pursue Park et al.'s framing further.
**Blocks:** P10-FE35 (no point dual-steering if dual coords don't even shift the basic geometry).

### H-545: Dual-coordinate steering causes < 5% off-target perplexity inflation at the alpha that flips MATH-500 K=1 correctness
**Priority:** HIGH
**Motivated by:** Park et al. 2602.15293 Figure 4 + H-1
**Test:** Run two-arm experiment per P10-FE35. For each arm (primal additive / dual), at the smallest alpha that flips ≥10% of cached "borderline" problems (those near the prefill DoM threshold), measure off-target perplexity on a held-out C4 slice. Predicts: primal arm shows 20–40% perplexity inflation (per their Figure 4); dual arm shows < 5%. If dual arm also shows >10% inflation, MATH correctness is probably not concept-factorizable (cross-ref H-71 / P11-FE697) and Park et al.'s framework is the wrong tool here. If primal arm shows < 5%, our DoM happens to coincide with a dual coordinate (interesting fact about residual streams; suggests a hidden information-geometric alignment in trained transformers worth a separate writeup).
**Requires:** H100 for steering generations (~1 day), CPU for off-target metrics (~4h). Same cache as H-1 / P10-FE1.
**Would change:** If primal degrades and dual is clean, H-1 in its current form is wrong-headed and we should prefer dual steering. If both degrade, Park et al.'s framework doesn't help us. If both are clean, the primal DoM was already a "lucky" coordinate.
**Blocks:** any H-1 follow-on that assumes naive primal steering is the right baseline.

---


### H-546: The L19 prefill DoM is the residual-stream image of an AdamW-induced parameter-space backbone
**Priority:** HIGH
**Motivated by:** 2602.23696 + F-2 + F-3
**Test:** P11-FE698 — uncentered SVD on per-token L19 residual trajectories. Confirm if PC1 ≥ 60%, cos(PC1, prefill_DoM) ≈ 1, and k*=1 (intra-signal gap) dominates across MATH-500 problems.
**Requires:** CPU-only, cached P11 H100 NPZs (already on disk).
**Would change:** On confirm, F-2 inherits an optimizer-grounded mechanism (single mode above spectral gap), and selective prediction (F-8) gains a label-free DoM identification path. On reject, F-2's L19 direction is something *other* than the residual-backbone — possibly supervised-probe-specific — and the parameter-space framing of this paper is not transferable.
**Blocks:** P11-FE698, P11-FE701.

### H-547: Breathing magnitude is reduced under non-AdamW pretraining
**Priority:** MEDIUM
**Motivated by:** 2602.23696 + F-1 + F-5
**Test:** Train a 51M GPT-2 (Xu's mini_gpt config, github.com/skydancerosel/mini_gpt) under (a) AdamW β2=0.95, (b) SGD+momentum, (c) AdamW β2=0.0. Compare per-token L19 PR breathing curves. Xu predicts dispersion across all directions and collapsed transverse dynamics under (b)/(c), so breathing should be quantitatively weaker or qualitatively absent.
**Requires:** Single H100 day for three matched 10k-step pretraining runs.
**Would change:** On confirm, F-1's "breathing universal across architectures and scales" is qualified to "universal across AdamW-pretrained models". F-5's content-dependence framing is enriched with an optimizer-dependence layer. On reject (breathing equally strong in SGD-trained models), Xu's optimizer-induction claim doesn't transfer to inference-time activation dynamics.
**Blocks:** nothing.

### H-548: prefill–final orthogonality (F-3) reflects a single-mechanism phase reversal, not two circuits
**Priority:** MEDIUM
**Motivated by:** 2602.23696 + F-3
**Test:** Re-extract prefill_DoM and final_DoM from a base-only Qwen-2.5-1.5B (pre-instruct) checkpoint if accessible via HF, and compute cos. If cos changes substantially (positive or strongly negative) vs the current 0.046 on the instruct model, prefill and final are the same direction observed in different post-training phases (Xu's λ-switch analog).
**Requires:** ~2h H100 for activation extraction on base checkpoint.
**Would change:** On confirm, F-3 stops being evidence for two-mechanism prefill-vs-final and becomes evidence for one mechanism with phase-dependent projection. Affects how H-13 (head-level attribution) is interpreted. On reject (cos stays near 0 across checkpoints), F-3's two-mechanism reading stands.
**Blocks:** H-13 framing.

### H-549: DoM stability across training checkpoints is bounded by Davis–Kahan, not unconditional
**Priority:** LOW
**Motivated by:** 2602.23696 + H-8
**Test:** P11-FE700 — measure cos(DoM_t, DoM_t') across Qwen-2.5-1.5B intermediate checkpoints. Test whether observed rotation respects α_1 ≈ 0.82 stability bound from Xu §9.3.
**Requires:** 4h H100 for DoM extraction × 3-4 checkpoints.
**Would change:** Reformulates H-8 from "stable" to "stable-within-phase, rotates at objective boundaries". Predicts the magnitude of rotation we should expect at SFT→DPO boundaries.
**Blocks:** H-8 promotion to a stronger claim.

---


### H-550: L19 prefill-DoM signal coincides with the dominant cross-layer attention concentration band on Qwen2.5-family models
**Priority:** HIGH
**Motivated by:** 2603.00437 + F-2
**Test:** Compute layer-pair cosine-similarity heatmap on Qwen-2.5-1.5B from cached pathway11_h100 prefill activations. Check if layers 19-21 form a high-similarity attended band (mirroring ICLA's Qwen2.5-VL-7B finding) and whether the band aligns with F-2's L19 peak. Cross-check on cached Qwen-2.5-7B activations. ~30min-1h CPU.
**Requires:** CPU; cached prefill_gated_compute hidden states for Qwen-2.5-1.5B and 7B.
**Would change:** Confirm = F-2 gains an architectural mechanism (Qwen-2.5 family specifically routes through L19-L21), upgrades H-13 priority. Reject = F-2 is a property of the math reasoning task independent of Qwen architectural routing, decouples L19 from the ICLA finding.
**Blocks:** P11-FE703 (do the cheap heatmap before the expensive CLA training).

### H-551: A trained diagonal cross-layer attention module collapses prefill/final DoM orthogonality
**Priority:** MEDIUM
**Motivated by:** 2603.00437 + F-3
**Test:** Train a CLA module (k0=16, r=128, alpha=0.02, ~100K params, parameter-shared) on Qwen-2.5-1.5B with frozen backbone using positive MATH-500 training samples. Measure cos(prefill_L19_DoM, final_L19_DoM) before vs after. ~3h H100 + 2h CPU.
**Requires:** H100 pod, positive math-training data, code path for forward-hooked residual injection.
**Would change:** Confirm (cos jumps from 0.046 to >0.3) = F-3 orthogonality is a routing artifact. Reject (cos stays near 0) = orthogonality is robust to the canonical layer-attention mitigation, strengthens F-3 as a fundamental geometric invariant.
**Blocks:** nothing.

---


### H-552: A multi-layer LoRA-gated verifier extracts non-redundant correctness signal beyond single-layer L19 DoM
**Priority:** HIGH
**Motivated by:** 2603.01025 + F-9 + F-2
**Test:** Train an OTV-style verifier on Qwen-1.5B (LoRA r=16, layers {15,17,19,21,23}, MetaMathQA 1k subset, linear-ramp labels). Compare to (a) single-layer L19 DoM (F-2), (b) CoE-60 (F-9), (c) OTV-paper's "probe" baseline (single-layer + 3-MLP, no LoRA). If multi-layer LoRA beats single-layer L19 DoM by >3 AUROC points on MATH-500, the F-9 redundancy claim is overturned. ~1 day H100.
**Requires:** H100 pod, Qwen-1.5B, MetaMathQA, LlamaFactory.
**Would change:** Confirm → F-9 inverted ("multi-layer matters when pooled correctly"); F-2's 0.7731 reframed as a single-probe floor, not a ceiling. Reject → F-9 strengthened; F-2 confirmed as architecture-independent ceiling.
**Blocks:** H-13 (head-level attribution), H-12 (label-free probes — OTV is decisively supervised so H-12 needs a different motivation).

### H-553: F-2's prefill DoM probe is fragile to surface-form perturbations
**Priority:** HIGH
**Motivated by:** 2603.01025 (Appendix E) + F-2 + the 256-tok truncation bug history
**Test:** Take 50 cached MATH-500 traces, generate semantics-preserving + logic-breaking perturbations (OTV Table 6 protocol), recompute prefill L19 DoM. Track AUROC delta per perturbation type. Pass = invariant under semantics-preserving (Δ<0.02) and drops under logic-breaking (Δ>0.05). ~4h CPU + 2h H100.
**Requires:** Cached Stage 4a traces, H100 for re-extraction on perturbed inputs.
**Would change:** Pass → F-2 elevated from "correlational signal" to "reasoning-grounded probe", strengthens H-19 routing case. Fail → F-2 demoted to surface-pattern detector; raises spectre that 1024-tok numbers may have hidden length/format confounds, requires controls before further claims.
**Blocks:** P11-FE3 activation patching (no point patching a surface-pattern signal).

### H-554: Late-token aggregation (mean/min over last 100 tokens) dominates prefill DoM
**Priority:** MEDIUM
**Motivated by:** 2603.01025 (Table 4) + F-2 + F-3
**Test:** On cached Qwen-1.5B + Qwen-7B per-token L19 DoM scores for MATH-500, compute AUROC at: prefill, final-token, mean/min/max over last 100 / 400 / 1600 tokens, all tokens. Trace-level filtering at 0/25/50/75%. ~1h CPU.
**Requires:** Cached Stage 4a NPZs only.
**Would change:** Late-token wins → F-2 contradicted; current "prefill is best probe location" headline becomes "best probe is late-token mean"; F-3's "orthogonal directions" framing requires reinterpretation as a single curved trajectory.
**Blocks:** Nothing — pure aggregation experiment, no downstream blocking.

---


### H-555: Non-linear MLP transport at L19 prefill outperforms linear DoM as a correctness probe and as a steering operator
**Priority:** HIGH
**Motivated by:** 2603.03163 + F-2 + H-1
**Test:** Train residual MLP `T(z)=z+MLP(z)` (single hidden layer, zero-init final, dual-objective loss with λ identity-preservation term) on cached `pathway11_h100/.../prefill_l19.npz`. Two readouts: (a) class separability AUROC OOF vs F-2's 0.7731 logistic baseline; (b) when used as a steering map at inference (push incorrect-class prefill activations through T and decode), measured impact on MATH-500 K=1 accuracy at matched fluency degradation vs ActAdd-DoM steering (H-1). ~30 min CPU for (a); ~4h H100 for (b).
**Requires:** CPU-only for (a) — cached NPZs sufficient. (b) requires H100 inference pod.
**Would change:** Confirm → F-2 should be reframed as "single direction is the linear projection of a low-dim non-linear correctness manifold"; H-1's null result (if observed) should be re-interpreted as ActAdd-construction failure, not a refutation of steerability. Reject → reinforces F-2 single-direction reading.
**Blocks:** H-1's interpretation depends on this — if MLP transport works, a null H-1 doesn't say "DoM doesn't steer correctness" but "ActAdd doesn't capture the right transport."

### H-556: Mahalanobis-OOD scoring on L19 prefill raises selective-prediction accuracy at coverage 0.5 above F-8's 71.6%
**Priority:** HIGH
**Motivated by:** 2603.03163 + F-8
**Test:** Compute regularized-shrinkage Σ̂⁻¹ (Eq. 4) per correctness class on cached L19 prefill features. Score each problem by squared Mahalanobis distance to incorrect-class centroid; gate at quantile threshold η to achieve 50% coverage; measure accuracy on answered subset. ~30 min CPU.
**Requires:** CPU-only; cached `prefill_l19.npz` and per-class centroids.
**Would change:** Confirm → F-8 numbers improve and the supervised-DoM/logistic recipe gets a non-parametric covariance-aware competitor that may transfer better cross-domain. Reject → strengthens the case that L19's correctness signal is genuinely linearly separable (Gaussian-equal-covariance regime).
**Blocks:** nothing.

### H-557: Per-layer Mahalanobis-OOD stack across 28 layers recovers a non-linear trajectory signal that single-layer L19 DoM misses, partially refuting F-9
**Priority:** MEDIUM
**Motivated by:** 2603.03163 + F-9
**Test:** For each of the 28 P11 stage-marker layers, compute class-conditional Mahalanobis-OOD log-likelihood (correct vs incorrect). Stack via simple sum across layers; compare 5-fold OOF AUROC against (a) single-L19 logistic 0.7731 and (b) CoE-60 0.811. ~2h CPU.
**Requires:** Cached per-layer activations from P11 stage markers.
**Would change:** Confirm → F-9 ("CoE redundant with single-layer DoM") should be qualified to "redundant *under linear readout*"; CoE's marginal lift becomes attributable to capturing per-layer non-linear shape, not just trajectory.
**Blocks:** nothing.

### H-558: The L19 prefill correct-vs-incorrect distribution exhibits CAT's "variance-mismatch" or "XOR" manifold class, not "isotropic-Gaussian"
**Priority:** MEDIUM
**Motivated by:** 2603.03163 (Section 4.3 synthetic taxonomy)
**Test:** Project class-conditional L19 prefill activations onto top-2 within-class PCs; visualize and apply diagnostic statistics (within-class covariance ratio, mode count via GMM-BIC). Classify which of {Gaussian, variance-mismatch, moon, XOR} best describes the geometry.
**Requires:** Cached `prefill_l19.npz`; ~1h CPU.
**Would change:** Confirm "variance-mismatch" or "XOR" → expectation is high that linear methods (DoM, ActAdd) underperform a CAT-style MLP, gating priority of H-555/74/75. Confirm "isotropic-Gaussian" → linear DoM is provably near-optimal, and we can de-prioritize the non-linear stack.
**Blocks:** Should run before H-555 / FE73.

---


### H-559: Confidence variance over W=2 per reasoning step is a non-redundant correctness predictor on top of L19 DoM
**Priority:** HIGH
**Motivated by:** 2603.12372 + F-9
**Test:** On cached MATH-500 Stage 2 traces (Qwen-2.5-1.5B-Instruct), compute
per-step c_s = exp(mean(log p_max)) and Var(c_s) over W=2. Aggregate per
problem (mean, max, fraction-of-steps-O / U / moderate). Fit logistic
regression for K=1 correctness; compute AUROC alone, AUROC of L19 prefill
DoM (anchor 0.7731), and AUROC of (DoM, variance) ensemble. ~30 min CPU.
**Requires:** Cached log-probs from P11 Stage 2; no GPU.
**Would change:** AUROC lift ≥ 0.02 → F-9 is parameterisation-specific to
CoE-60, not a general trajectory-feature null. AUROC lift < 0.005 → F-9
generalises to confidence-variance trajectory summaries too, and ReBalance's
benefit comes from the *steering* mechanism, not new information in the
variance signal.
**Blocks:** nothing.

### H-560: ReBalance's overthinking vs underthinking direction at L19 is approximately orthogonal to the prefill correctness DoM
**Priority:** HIGH
**Motivated by:** 2603.12372 + F-2 + F-3
**Test:** Compute mu^O, mu^U at L19 first-token-of-step on cached MATH-500
traces. Form v_OU = (mu^O - mu^U) / ||.||. Compute |cos(v_OU, prefill_DoM)|,
|cos(v_OU, final_token_DoM)|, |cos(v_OU, CoE_top_PC)|. ~30 min CPU.
**Requires:** Cached L19 activations (have).
**Would change:** |cos| > 0.3 → ReBalance is approximating supervised DoM
unsupervised; H-12 confirmed; F-2's correctness-label requirement weakened.
|cos| < 0.1 → over/under-thinking is a *second* behavioural axis distinct
from correctness, motivating a 2D steering bank over composed axes.
**Blocks:** H-1 (informs whether per-position DoM bank should compose
correctness DoM with confidence-variance DoM).

### H-561: ReBalance step-boundary steering reproduces the paper's MATH-500 lift on non-distilled Qwen-2.5-1.5B-Instruct
**Priority:** MEDIUM
**Motivated by:** 2603.12372 + H-1 + F-8
**Test:** Run MATH-500 K=1 generation on Qwen-2.5-1.5B-Instruct with
ReBalance step-boundary steering (FE74 prototype direction, Eq 10
modulation). Compare Pass@1 and token count to baseline (no steering).
Reference target: paper reports +3.4 Pass@1 / −23.1% tokens on
DeepSeek-R1-Distill-Qwen-1.5B (same base, different post-training).
**Requires:** H100; ~4h.
**Would change:** Lift ≥ +2 Pass@1 → method generalises beyond distilled
reasoning models; H-1 has a working recipe. Lift ≤ +0.5 → ReBalance is
specific to RL-distilled "long-CoT" base models, and Qwen-2.5-Instruct's
shorter CoT regime doesn't have the overthinking surface to exploit.
**Blocks:** nothing.

---


### H-562: Prefill L19 DoM is a decoder of pre-existing structure, not a distinct feature axis
**Priority:** HIGH
**Motivated by:** 2603.14923 (Taylor) §7-8 + F-2 (AUROC 0.7731) + F-3 (cos=0.046)
**Test:** Run paired Qwen-2.5-1.5B inference on MATH-500 with vs without per-step L19 DoM-projection-removal during generation (n=100). Compute (a) ΔK=1 accuracy, (b) layer-wise CKA between routed-off and baseline residual streams. CKA ≥ 0.95 + ΔAcc ≤ −5pp = decoder framing confirmed.
**Requires:** ~4h H100 (forward passes with hook) + 1h CPU (CKA computation). Cached P11 prefill activations needed for warm-start.
**Would change:** Confirm → F-2's signal is a readout of underlying knowledge; aligns with F-9 (CoE-60 redundant with single-layer DoM) and predicts F-8 (selective-prediction) will plateau on out-of-familiar-distribution tasks. Reject (CKA drops below 0.9) → DoM is a structural feature; supports the "distinct correctness axis" framing F-2 currently rests on.
**Blocks:** P10-FE36 (this is the formal hypothesis behind that FE).

### H-563: F-3 orthogonality dissolves under superposition baseline
**Priority:** MEDIUM
**Motivated by:** 2603.14923 (Taylor) §8 within-head 75.9° finding + F-3 cos(prefill_DoM, final_DoM)=0.046
**Test:** Project L19 DoM onto a per-head decomposition (12 heads × 128 d_head) for both prefill and final-token positions; compute pairwise angles between all 12 prefill-head DoMs and all 12 final-head DoMs. Compare to angle distribution of K=12 random unit vectors in 128D. Place cos=0.046 (87.4°) inside the empirical CDF.
**Requires:** ~3h CPU on cached P11 residuals + per-head DoM extraction (cache reorg).
**Would change:** If 87.4° sits between 25th-75th percentile of within-head learned-direction angles, F-3 collapses from "orthogonal directions" into "directions within the model's natural superposition tolerance." If it sits above 95th percentile, F-3 strengthens with a calibrated baseline.
**Blocks:** Any future paper-write-up that cites F-3 as evidence of independent prefill/final circuits.

### H-564: The F-7 D-bucket signature peaks at the lowest-variance, not highest-variance, Qwen layer
**Priority:** MEDIUM
**Motivated by:** 2603.14923 (Taylor) Table 4 (L9 most critical at lowest cross-domain variance) + F-7 (D-bucket distinctive signature)
**Test:** For each of Qwen-2.5-1.5B's 28 layers, fit a logistic probe distinguishing 36 D-bucket from 207 A-bucket problems on cached prefill residuals; record per-layer AUROC and per-layer cross-bucket activation variance. Plot AUROC vs variance.
**Requires:** ~1h CPU on cached P11 residuals.
**Would change:** Anti-correlation (AUROC peaks where variance is low) → F-7's "distinctive collective signature" should be sought at layers near LM head, reframing the D-bucket as a decoder-axis bucket. Positive correlation → conventional reading of F-7 holds. Flat → variance is independent of decision-relevance, weakening F-7 itself.
**Blocks:** P11-FE21 (D-bucket attention entropy at prefill — should be re-targeted to whichever layer wins this test).

---


### H-565: F-2's L19 prefill DoM is contaminated by subject and surface-form covariates
**Priority:** HIGH
**Motivated by:** 2603.18280 + F-2
**Test:** Run P11-FE719 (LOCO-CV across MATH-500's 7 subjects) and P11-FE720 (ridge-residualization against length/latex/subject/math-logit basis). If LOCO mean AUROC < 0.65 OR residualised AUROC < 0.65, hypothesis confirmed: F-2 is largely capability/topic, not correctness routing.
**Requires:** Cached `pathway11_h100/prefill_gated_compute` NPZs (already on disk); ~1.5h CPU total; MATH-500 subject metadata.
**Would change:** On confirm — F-2's status drops from STRONG to PARTIALLY CONFIRMED, F-8 inherits the downgrade, and any H-1 steering trial must use the residualised vector. On reject — F-2 graduates from "single-fold AUROC" to "category-out + capability-residualised AUROC" and becomes much harder to refute.
**Blocks:** H-1 (per-position DoM steering must use residualised direction), H-12 (any unsupervised replacement gets the same diagnostic test)

### H-566: F-2 is a single-direction projection of a multi-direction concept cone
**Priority:** MEDIUM
**Motivated by:** 2603.18280 + F-2 + F-3 (Wollschlager et al concept-cones reference in paper)
**Test:** P11-FE721 — SVD of per-example correctness-difference matrix at L19 prefill, rank sweep k ∈ {1,2,4,8,16}, compute max-projection AUROC. If AUROC rises monotonically to ≥0.83 by rank≥4, single-direction framing of F-2/F-3 is incomplete.
**Requires:** Cached prefill NPZs; ~1h CPU.
**Would change:** On confirm — F-2/F-3 are reframed as rank-k subspace findings; cos(prefill, final)=0.046 cosine becomes a subspace-angle question, not a vector-angle question. On reject — single-direction ansatz survives a multi-rank challenge, modest evidence boost.
**Blocks:** nothing critical

### H-567: Cross-model DoM transfer fails between Qwen-2.5-1.5B and Qwen-2.5-7B
**Priority:** MEDIUM
**Motivated by:** 2603.18280 + F-2 + H-11
**Test:** P11-FE723 — extract F-2 DoM on 1.5B, project/pad to 7B residual dim, apply to 7B prefill activations on same MATH-500 problems, measure AUROC vs native-7B DoM AUROC. Transfer-fails if Δ ≥ 0.10.
**Requires:** Both 1.5B and 7B prefill NPZs from `pathway11_h100`; ~2h CPU.
**Would change:** On confirm — H-11 (big-to-small distillation of 7B self-prediction into 1.5B) becomes much harder; routing geometry is scale-specific. On reject — paper's "model-specific routing" claim has a counterexample within the Qwen family, and H-11 is more attractive.
**Blocks:** H-11

---


### H-568: F-2's 0.7731 ceiling is probe-architecture-bound, not information-bound
**Priority:** HIGH
**Motivated by:** 2603.24787 + F-2
**Test:** Fit a 5-layer MLP (and an Attention Probe) on cached prefill L19 mean-pooled activations from `pathway11_h100/prefill_gated_compute/` for Qwen-2.5-1.5B MATH-500. Compare 5-fold OOF AUROC against the 0.7731 logistic baseline. ~30min CPU.
**Requires:** CPU only, cached NPZs (already gitignored but on disk per DATA_MANIFEST).
**Would change:** If MLP-AUROC ≥ 0.82, F-2 is downgraded from "L19 DoM *is* the signal" to "L19 *contains* the signal but linear DoM under-extracts." This re-opens H-13 (head-level attribution) and re-prioritizes nonlinear probes for F-8. If MLP-AUROC ≈ 0.78 or lower, F-2's linear-probe-is-fine narrative is reinforced.
**Blocks:** P11-FE725, P11-FE726.

### H-569: Prefill / final-token DoM orthogonality (F-3 cos = 0.046) is a single-token aliasing artifact
**Priority:** HIGH
**Motivated by:** 2603.24787 + F-3
**Test:** Train an attention-aggregation probe over L18–L20 across all prefill positions, separately across all generation positions, on cached MATH-500 NPZs. Compute cos between the aggregated probe weight vectors AND between each and the single-position L19 DoM. ~1h CPU.
**Requires:** CPU only, cached NPZs.
**Would change:** If the aggregated prefill and aggregated final probes have cos > 0.4 with each other, F-3's "two orthogonal correctness directions" framing is replaced by "one correctness direction with single-token aliasing." This collapses one of the central narrative claims of the project and forces re-write of F-3 + relevant PERSPECTIVES sections.
**Blocks:** P11-FE726.

### H-570: KL-regularized variational probe lifts F-8's 50% coverage ceiling above 75%
**Priority:** MEDIUM
**Motivated by:** 2603.24787 + F-8
**Test:** Train ReLope-style LoRA + variational bottleneck head on Qwen-2.5-1.5B L19 prefill states (4h H100). Measure 5-fold OOF AUROC and refuse-and-spend curve at coverages 0.25/0.5/0.75. Probe robustness with paraphrase + CoT prompt jitter.
**Requires:** H100 pod, MATH-500 prefill cache.
**Would change:** If coverage-0.5 acc-on-answered > 75%, F-8's headline is updated and the selective-prediction story shifts to "extraction matters as much as layer choice." If ≤ 72%, F-8's ceiling holds and the linear-probe-is-fine narrative is confirmed (and H-568 likely fails too).
**Blocks:** nothing (long-tail confirmation).

---


### H-571: Prefill DoM is the gold-calibration direction; the prefill/final-token orthogonality is task-elicited, not position-elicited
**Priority:** HIGH
**Motivated by:** 2603.25052 + F-3
**Test:** Re-extract prefill activations on MATH-500 under (a) no-CoT pure-correctness
prompt and (b) pure-confidence prompt. Fit DoMs at L19 for each. Measure cos(pure-correctness
DoM, joint-prompt prefill DoM) and cos(pure-correctness DoM, pure-confidence DoM).
~6h H100 + 1h CPU.
**Requires:** H100 pod, Qwen-2.5-1.5B, two new prompt templates, otherwise existing pipeline.
**Would change:** If pure-correctness DoM ≈ joint-prompt prefill DoM (cos > 0.5), F-3 is a
re-discovery of Miao-Ungar task-elicited orthogonality — re-frame F-3 in those terms.
If pure-correctness DoM and joint-prompt prefill DoM are themselves orthogonal, prefill
DoM is its own thing and F-3 is genuinely position-specific.
**Blocks:** nothing.

### H-572: Probe-only isotonic calibration of prefill L19 DoM achieves ECE < 10 on MATH-500 at 1024-tok
**Priority:** HIGH
**Motivated by:** 2603.25052 + F-8
**Test:** Apply isotonic regression to existing prefill L19 DoM probe scores on K-fold
held-out MATH-500. Compute ECE/Brier/MAE against per-question empirical accuracy from
existing K=8 generations. ~30 min CPU.
**Requires:** existing cached probe outputs and K=8 generations only.
**Would change:** Reframes F-8 from "selective accuracy at coverage 0.5" to "competitive
calibration without any steering" — directly comparable to Miao-Ungar Table 4. If ECE > 15,
the probe-only baseline is genuinely insufficient and steering would be needed for parity.
**Blocks:** must precede H-573 to gate whether full pipeline is worth running.

### H-573: Miao-Ungar two-stage adaptive steering pipeline reduces ECE on Qwen-2.5-1.5B + MATH-500
**Priority:** MEDIUM
**Motivated by:** 2603.25052 §3.8
**Test:** Full replication on Qwen-2.5-1.5B at 1024-tok. K=11 confidence framings (their
Appendix prompts), within-question CAA at L19 + L21, α-sweep on validation, PCHIP inversion,
adaptive generation. Report ECE/Brier/MAE on MATH-500 test fold. ~8h H100 + 1h CPU.
**Requires:** H100 pod, Qwen-2.5-1.5B, K=11 prompt templates, validation split.
**Would change:** If ECE drops by ≥3× over unsteered-verbal-baseline, calibration is the
right framing for our prefill direction — re-headline P11. If pipeline fails (ECE no
better than probe-only baseline), 1.5B may lack the dynamic range Miao-Ungar exploit at
7B; would suggest H-11 (big-to-small distillation) as the right path forward.
**Blocks:** nothing once H-572 establishes probe-only floor.

### H-574: Within-question CAA contrast vector outperforms mass-mean DoM at L19
**Priority:** MEDIUM
**Motivated by:** 2603.25052 Eq. 1
**Test:** Construct within-question CAA contrasts on Pathway 11 cached activations using
correctness as the within-question split (50 samples / question). Compare AUROC of the
resulting direction against existing prefill DoM (0.7731). ~2h CPU on cached NPZs.
**Requires:** existing K=50 (or K=8 → simulate by per-question correctness split) cached
activations.
**Would change:** If within-question CAA AUROC > 0.80, our mass-mean DoM is conflating
question-difficulty with correctness; replace with within-question contrast in F-2
narrative.
**Blocks:** nothing.

---


### H-575: Online TTT update of the prefill L19 DoM lifts MATH-500 selective-prediction AUROC over the static fit
**Priority:** HIGH
**Motivated by:** 2604.01170 + F-2 + F-9
**Test:** Initialize W_0 from our fitted prefill L19 DoM. For each MATH-500 problem, unroll a no-QK
TTT inner loop (η=0.01, Brier loss vs supervised correctness label) along the cached step-wise
residuals. Measure 5-fold OOF AUROC at the final step and report the trajectory `cos(W_0, W_t)`.
Compare against static AUROC 0.7731.
**Requires:** CPU only (~30min), cached `pathway11_h100/prefill_gated_compute` Stage-2 NPZs, our
existing fitted DoM as W_0.
**Would change:** *Confirm* (TTT AUROC > 0.79): F-2 must be restated as the *initialization* of an
adaptive direction; F-9's "trajectory features redundant" claim must be narrowed to closed-form
aggregations only; F-3's orthogonality interpretation is challenged. *Reject* (TTT AUROC ≤ 0.7731 ±
0.005): F-2's "static direction is sufficient" framing strengthens; ORCA's gains are
scale-/architecture-specific (32B → does not transfer to 1.5B).
**Blocks:** H-576 (LTT calibration is most informative if applied to the better-of-{static, TTT} probe).

### H-576: LTT-calibrated selective prediction on prefill DoM exceeds 71.6% accuracy at coverage 0.5 with finite-sample risk guarantee
**Priority:** HIGH
**Motivated by:** 2604.01170 + F-8
**Test:** Sweep threshold grid Λ on prefill DoM scores; compute binomial p-values
`p_j = P(Binom(n, δ) ≤ n·R̂_n(λ_j))`; apply fixed-sequence testing at ε=0.05 to control FWER; select
most-aggressive rejected λ*. Report (savings, error) for δ ∈ {0.05, 0.10, 0.15, 0.20} on MATH-500.
Compare against current F-8 number (71.6% acc at coverage 0.5, K=2.5).
**Requires:** CPU only (~20min), our existing prefill DoM scores on MATH-500 (pathway11 results
JSON).
**Would change:** *Confirm* (LTT-calibrated accuracy > 73% at coverage 0.5 with P(R ≤ 0.1) ≥ 0.95):
F-8 is upgraded with a finite-sample risk guarantee and a stronger headline number; we adopt LTT
as the canonical selective-prediction recipe. *Reject* (LTT-calibrated accuracy ≤ 71.6%): F-8 stands;
ORCA's gains are TTT-specific, not LTT-specific.
**Blocks:** nothing (cheap, parallelizable).

### H-577: Consistency-mode (C_t = I{ans(y_t) = ans(y_T)}) probes are competitive with supervised DoM as label-free correctness signals
**Priority:** MEDIUM
**Motivated by:** 2604.01170 + H-12
**Test:** Build an intermediate-answer parser for Qwen-1.5B MATH-500 trajectories. Construct
consistency labels at each step. Train (a) static probe on consistency labels, (b) TTT probe on
consistency labels, on cached L19 residuals. Compare AUROC and selective-prediction accuracy
against (i) supervised DoM, (ii) SEP-style entropy probes (2406.15927).
**Requires:** ~2h CPU for probes + ~2h one-time dev for the intermediate-answer parser.
**Would change:** *Confirm* (consistency probes within 1 AUROC point of supervised): H-12 redirects
from semantic-entropy probes specifically to consistency-vs-final-answer probes generally; we have
a label-free deployment path. *Reject* (consistency probes ≥ 3 AUROC points worse): supervised
labels remain load-bearing; ORCA's consistent-mode gains are 32B-specific.
**Blocks:** nothing.

---


### H-578: Step-wise trajectory features (concat of last-step difference + termination marker, PCA-128) outperform single-layer L19 prefill DoM by ≥0.04 AUROC on MATH-500

**Priority:** HIGH
**Motivated by:** 2604.05655 + F-9 (CoE-60 redundant with L19 DoM) + EXP-related to Pathway 11 prefill probe
**Test:** Implement Sun et al.'s late-step trajectory feature construction on Qwen-1.5B Stage 2 cached activations: locate Step boundaries, extract h(L19) at each, concatenate (h_termination_marker) ⊕ (h_step_(N−1) − h_step_N), PCA dim=128, OOF logistic. Compare to baseline 0.7731 prefill DoM. 2h CPU.
**Requires:** CPU. Cached Pathway 11 Stage 2 K=1 NPZs.
**Would change:** Confirmation refutes F-9 (single-layer DoM is sufficient) along the step-wise axis; rejection corroborates F-9's redundancy claim by extending it to step-wise trajectory features.
**Blocks:** Whether to prioritize trajectory-feature reformulation of F-2/F-8.

### H-579: Step-organized linear subspaces are universal across Qwen-1.5B / Qwen-7B / Llama-8B (cross-architecture step probes ≥0.85 best-layer accuracy for Steps 1–4)

**Priority:** MEDIUM
**Motivated by:** 2604.05655 + F-1 (dimensional breathing universal across transformer architectures)
**Test:** Train one-vs-all step-identity logistic regression on Llama-3.1-8B activations from cached generations, evaluate on Qwen-1.5B and Qwen-7B activations, vice versa. Report layer-wise transfer accuracy. 1h CPU.
**Requires:** CPU. Cached step-marker activations from Llama-3.1-8B (would need extraction) and Qwen-1.5B/7B (cached).
**Would change:** Confirmation extends F-1 from PR-curve universality to a second geometric universality (linear-subspace step organization). Rejection (Qwen fails to organize step subspaces) means F-1's PR-curve universality might be a measurement artifact.
**Blocks:** Cross-architecture confidence in any trajectory-based intervention.

### H-580: DoM(k) (step-indexed correctness direction) rotates smoothly through SO(d) — consecutive cosines >0.6 — rather than producing two orthogonal circuits at prefill and final-token

**Priority:** HIGH
**Motivated by:** 2604.05655 + F-3 (cos = 0.046, "directions are orthogonal") + H-17 (RoPE-mechanical rotation)
**Test:** Compute DoM(k) at every step boundary k for Qwen-1.5B MATH-500, report cos(DoM(k), DoM(k+1)) for k = 0..K_max. If consecutive cosines are 0.6–0.95, F-3 is reframed as smooth rotation. 2h CPU.
**Requires:** CPU. Cached Stage 2 per-token activations with Step-boundary indices located.
**Would change:** Confirmation refutes F-3 in favor of a single rotating axis through SO(d) — recasts H-17 as a confirmed mechanism. Rejection corroborates F-3's two-circuit framing.
**Blocks:** F-3 reformulation, H-17 priority.

### H-581: D-bucket members are detectable per-problem via cumulative trajectory deviation D_j against an A-bucket-derived ideal trajectory, with recall ≥80% at FPR ≤20%

**Priority:** MEDIUM
**Motivated by:** 2604.05655 + F-7 (D-bucket has *collective* not per-problem signature)
**Test:** Construct ideal-trajectory PCA-128 from A-bucket-only Qwen-1.5B trajectories, compute step-wise μ_j and σ_j, evaluate D_j on each bucket's members, report ROC for D-bucket vs A∪B∪C. 3h CPU.
**Requires:** CPU. Cached Stage 2 NPZs.
**Would change:** Confirmation refutes F-7's collective-only framing — D-bucket becomes per-problem-detectable along a different axis (cumulative trajectory deviation) than the PR/density measure.
**Blocks:** F-7 framing, prioritization of per-problem D-bucket interventions.

### H-582: In-sample termination steering vector s(ℓ) = E_k[h(ℓ, term) − h(ℓ, step_k)] is approximately collinear (cos > 0.7) with our final-token DoM at L19

**Priority:** MEDIUM
**Motivated by:** 2604.05655 + F-3 (prefill/final orthogonality) + H-1 (DoM steering moves accuracy)
**Test:** Compute s(ℓ) on cached Qwen-1.5B activations, report cos with prefill_DoM and final_DoM at L19, at MID 5 layers, and at LAST 5 layers. 1h CPU.
**Requires:** CPU. Cached Stage 2 NPZs.
**Would change:** Confirmation says Sun et al.'s steering target is the same as our final-token DoM — H-1 inherits their +1.80% MATH-500 result as a calibrated reference. Rejection (orthogonal) means termination steering is a third independent direction we should track separately.
**Blocks:** Pre-pod diagnostic for any H-1 implementation.

---


### H-583: Centroid-DoM at L19 outperforms latent-activation DoM for correctness prediction on Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2604.11962 + F-2 + F-3
**Test:** Extract L19 MLP centroids (J_L19^T · 1) for cached MATH-500 prefill positions on Qwen-2.5-1.5B (n=500). Train 5-fold OOF mass-mean probe on centroids; compare AUROC against F-2's latent-activation 0.7731. Secondary: test cross-task transfer to Marks & Tegmark `cities` / `sp_en_trans` to replicate Walker Fig 7. ~30min H100 + 15min CPU.
**Requires:** Qwen-2.5-1.5B model graph access, P11 H100 extraction harness, cached MATH-500 problem set, OOF 5-fold harness from EXP-018.
**Would change:** *On confirm*: F-2's 0.7731 is a partial / spurious signal under LCH — selective-prediction pipeline (F-8) should be rebuilt on centroids; PROJECT_RECORD §10 framing of "L19 prefill direction" needs LCH context. *On reject*: F-2 survives the substrate swap, the latent direction is a genuine functional signal, and our 91/91 invariant is reinforced.
**Blocks:** H-584 (centroid-orthogonality), H-585 (centroid-PH on F-10).

### H-584: Centroid prefill DoM and centroid final-token DoM are not orthogonal (cos > 0.5)
**Priority:** MEDIUM
**Motivated by:** 2604.11962 + F-3 (latent cos = 0.046)
**Test:** Compute centroid-DoMs at L19 for prefill last-token and final-answer-token across Qwen-2.5-1.5B MATH-500 (correct vs incorrect). Report cos(prefill_centroid_DoM, final_centroid_DoM); compare to F-3's 0.046. ~1h H100.
**Requires:** centroid extraction at both positions (depends on H-583 pipeline).
**Would change:** *On confirm*: F-3's "two distinct circuits" interpretation is a substrate artifact; the same correctness expert governs both positions. *On reject*: orthogonality is corroborated mechanistically and the two-circuits framing is strengthened.
**Blocks:** nothing.

### H-585: PH on L17–L21 centroid clouds deviates from the Gaussian null (refutation test for F-10)
**Priority:** MEDIUM
**Motivated by:** 2604.11962 + F-10 (PH on residuals = Gaussian null)
**Test:** Re-run EXP-029 PH pipeline (Gudhi VietorisRipsPersistence, alpha=0.001) on centroid clouds at L17–L21, n=500. Compare H_0/H_1 lifetimes and Betti curves to F-10's residual null and to a Gaussian-noise centroid null. ~1h H100 + 30min CPU.
**Requires:** centroid extraction at multiple layers (extends H-583 pipeline).
**Would change:** *On confirm*: F-10's "PH = Gaussian null" claim is substrate-bound and topological homology framing is partially recoverable on centroids. *On reject*: F-10 is robust to substrate; the topological-homology pivot stays in the graveyard.
**Blocks:** nothing.

### H-586: Centroid-perturbation attribution (Walker Eq 2) localizes correctness-relevant L19 MLP neurons
**Priority:** MEDIUM
**Motivated by:** 2604.11962 + H-13
**Test:** Apply Eq 2 to each L19 MLP neuron on Qwen-1.5B over correct vs incorrect prefill positions. Rank by attribution; check whether top-1% neurons concentrate near attention heads identified in any future H-13 analysis. ~2h H100.
**Requires:** model graph access for L19 MLP ablation, baseline centroids from H-583.
**Would change:** *On confirm*: H-13 has a concrete neuron-level method without depending on probe-direction crutches. *On reject*: centroid attribution is too noisy on small MLP widths and we fall back to ablation-based attribution.
**Blocks:** nothing.

---


### H-587: Mean per-token student entropy reaches AUROC ≥ 0.75 for MATH-500 correctness on 1.5B
**Priority:** HIGH
**Motivated by:** 2604.14084 + F-2 + F-8
**Test:** Re-extract L_final logits on the 500 cached 1.5B MATH-500 traces, compute softmax entropy per token, take per-trace mean, score AUROC against 1024-tok correctness label. Time: ~1h H100 + 30min CPU.
**Requires:** H100 pod for logit re-extraction (hidden states only currently cached); 500 cached generation transcripts.
**Would change:** If AUROC > 0.7731, F-2's "prefill DoM is best" leadership claim narrows to "best among static features"; F-8 mechanism shifts toward entropy-driven; H-12 ("SEP replaces DoM") gets direct support except on D-bucket.
**Blocks:** H-588 (needs entropy values to define Q3 axis).

### H-588: Q3-density (low-entropy + high teacher-student KL tokens) per trace correlates with D-bucket membership
**Priority:** HIGH
**Motivated by:** 2604.14084 + F-7
**Test:** Compute per-trace Q3-density on 500 MATH-500 problems (needs 7B teacher forced-decode on 1.5B rollouts to align tokens). Mann-Whitney across A/B/C vs D bucket. Pre-registered threshold: p < 0.01 for confirm.
**Requires:** H100 day for teacher 7B forced-decode pass; 1h CPU analysis.
**Would change:** If confirmed, F-7's "D-bucket signature is collective, not per-problem" framing is overturned at the per-token granularity — D-bucket becomes per-token diagnosable. Also gives a label-free D-bucket detector for H-19's verification routing.
**Blocks:** Refines H-19 (verification gating granularity).

---


### H-589: Spectral α at a single late layer matches or beats prefill L19 DoM for correctness prediction
**Priority:** HIGH
**Motivated by:** 2604.15350 + F-2
**Test:** Run P11-FE749 — compute α on cached pathway11 Stage-2 NPZs (Qwen-1.5B + 7B, 500 MATH-500 problems × all layers), train logistic regression with 5-fold stratified CV per layer, report best-layer α-AUROC and joint [α_L*, DoM_proj_L19] AUROC vs DoM-only baseline. ~30min CPU.
**Requires:** Cached Stage-2 prefill NPZs (already on disk from pathway11_h100).
**Would change:** If α-AUROC ≥ DoM-AUROC on Qwen-1.5B 1024-tok labels, F-2's "DoM is the strong single-feature predictor" framing must be revised to "DoM is one of several non-redundant single-feature predictors." If joint > DoM-only, F-9's CoE-redundancy parsimony claim must also be narrowed. If α-AUROC < DoM-AUROC, paper's AUC=1.000 result is benchmark-specific and we have a clean external null.
**Blocks:** Decision on whether to elevate H-12 (label-free probes) above current low-priority status.

### H-590: Trained residual stream has non-Gaussian spectral structure that α detects but PH cannot
**Priority:** HIGH
**Motivated by:** 2604.15350 + F-10
**Test:** Run P11-FE750 — fit power-law α on pathway11 residuals (500 × 28 layers × 2 models) AND on per-question-shuffled Marchenko-Pastur-matched Gaussian baselines. Per-layer Welch t-test on α distributions. ~1h CPU.
**Requires:** Cached residuals + simple Gaussian-baseline generator (numpy).
**Would change:** If real-α and Gaussian-baseline-α distributions differ at p<0.001 on most layers, F-10's null framing was a property of PH's blindness to non-Gaussian spectral structure, not a property of residual streams being Gaussian-equivalent. PH would become a strict subset of available geometric signal, and "non-Euclidean PH" (graveyard) might warrant revisiting with α-augmented features.
**Blocks:** Whether to revisit any of the killed graveyard experiments under the lens of α-augmented geometric features.

### H-591: AUC=1.000 spectral correctness prediction is a generation-length confound
**Priority:** HIGH
**Motivated by:** 2604.15350 + F-2 + H-15
**Test:** Run P11-FE751 — bin 500 MATH-500 generations into length terciles; recompute α-AUROC per bin (combine with FE80). ~30min CPU.
**Requires:** Cached Stage-2 NPZs + final-answer length metadata from `pathway11_h100/prefill_gated_compute/results.json`.
**Would change:** If α-AUROC drops below 0.7 within length-matched subsets, paper's AUC=1.000 result is a sequence-length artifact (analogous to our 256-tok bug, P6.5). Strengthens our cross-paper confidence that single-scalar single-layer "perfect prediction" claims should be length-controlled by default. Provides a methodological lesson reusable across the literature.
**Blocks:** Whether to adopt α as a default reporting metric in pathway11 results.

---


### H-592: Multi-layer LI score strictly improves over single-layer L19 DoM on MATH-500
**Priority:** HIGH
**Motivated by:** 2604.16217 + F-2 + F-9
**Test:** Implement Layerwise CP's LI score (eq 6–8 of the paper) on
cached Qwen-2.5-1.5B MATH-500 prefill residuals. Evaluate (a) LI
restricted to ℓ=19 only, (b) LI summed over all 28 layers. Score
correctness ranking via AUROC and via APSS at α∈{0.1, 0.2, 0.3} in a
split-conformal wrapper. Compare with the existing L19 DoM AUROC
0.7731. ~1 day CPU.
**Requires:** CPU only; cached `pathway11_h100/.../residuals_*.npz`
files already contain per-layer prefill hidden states; Qwen-2.5-1.5B
`lm_head` weights (loadable from HF).
**Would change:** If LI-all > LI-L19 by ≥3pp AUROC, F-9 ("CoE-60
redundant with L19") is contradicted and we'd reframe to "the right
multi-layer aggregation (logit-lens entropy) beats single-layer; the
wrong one (PCA-60 of raw activations) doesn't." If LI-all ≅ LI-L19,
F-9 is corroborated method-independently — much stronger evidence for
the single-layer story than we currently have.
**Blocks:** Conformal restatement of F-8 (P11-FE757) becomes more
informative once the right scoring rule is settled.

### H-593: L19 is not a probe-method artifact — LM-head-lens entropy reduction peaks at the same layer
**Priority:** HIGH
**Motivated by:** 2604.16217 + F-2
**Test:** Compute I_ℓ(x→y) = H_ℓ(y|∅) − H_ℓ(y|x) for ℓ ∈ {0..27} on
Qwen-2.5-1.5B MATH-500 prefill, where H_ℓ uses the LM-head-lens
projection of layer-ℓ hidden states. Report AUROC vs correctness per
layer. Compare argmax_ℓ I_ℓ with argmax_ℓ DoM-AUROC (currently L19).
30 min CPU.
**Requires:** CPU only; same cached residuals as H-592; null-context
forward pass (need to pin down what ∅ means — likely few-shot template
with question slot blanked; check Appendix B of paper before
implementing).
**Would change:** If argmax = 19, F-2 graduates from "the linear
probe likes L19" to "L19 is a probe-method-independent peak." If
argmax ≠ 19 (say L24), F-2's claim weakens to "late-mid block, with
L19 happening to be where DoM peaks but not the information peak."
**Blocks:** Nothing; descriptive layer-importance result.

---


### H-594: Mean token log-probability matches L19 DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2604.18805 (mean-log-prob domain separation, Sec 4.10)
**Test:** Compute per-problem mean log-prob over Qwen-2.5-1.5B K=1
generations from `pathway11_h100/prefill_gated_compute/`. AUROC vs
binary correctness. Compare to DoM AUROC = 0.7731. 20 min CPU if logs
cached, else 4 h H100.
**Requires:** Cached generation logs (or re-extraction); CPU.
**Would change:** Confirm → F-2 reframed as "DoM ≈ smoothed logit
confidence" and F-9 redundancy extends to logits; project narrative
shifts to "what does DoM see that logits don't?" Reject → F-2 / F-9
strengthen.
**Blocks:** H-1 prioritization (if DoM is just logits, steering is
unlikely to add value)

### H-595: F-8 selective-prediction does not survive Pass^5
**Priority:** HIGH
**Motivated by:** 2604.18805 (Pass^k decay framing)
**Test:** K=5 sampling on DoM-top-50% MATH-500 subset; compute Pass^5
vs bottom-50% and random-50% controls. 4 h H100.
**Requires:** H100 pod, Qwen-2.5-1.5B.
**Would change:** Confirm → F-8 is restricted to single-trial pipelines;
selective-prediction value proposition narrows. Reject → F-8 generalizes
to multi-trial reliability, strengthening it.
**Blocks:** Any deployment story that depends on selective prediction.

### H-596: F-3 prefill/final orthogonality is position-driven, not content-driven
**Priority:** MEDIUM
**Motivated by:** 2604.18805 (reasoning topology invariant across
domain groups)
**Test:** Extract L19 DoM at two prefill positions (original + 5
tokens earlier) per MATH-500 problem; compute cosine similarity. 30
min CPU on cached prefill activations.
**Requires:** Cached prefill NPZ (already in P11).
**Would change:** Confirm (cos near 0) → F-3 framing rewritten as
positional artifact; "two distinct decision processes" claim removed.
Reject (cos high) → F-3 strengthens.
**Blocks:** nothing.

---


### H-597: Mean token log-probability matches L19 DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2604.18805 (mean-log-prob domain separation, Sec 4.10)
**Test:** Compute per-problem mean log-prob over Qwen-2.5-1.5B K=1
generations from `pathway11_h100/prefill_gated_compute/`. AUROC vs
binary correctness. Compare to DoM AUROC = 0.7731. 20 min CPU if logs
cached, else 4 h H100.
**Requires:** Cached generation logs (or re-extraction); CPU.
**Would change:** Confirm → F-2 reframed as "DoM ≈ smoothed logit
confidence" and F-9 redundancy extends to logits; project narrative
shifts to "what does DoM see that logits don't?" Reject → F-2 / F-9
strengthen.
**Blocks:** H-1 prioritization (if DoM is just logits, steering is
unlikely to add value)

### H-598: F-8 selective-prediction does not survive Pass^5
**Priority:** HIGH
**Motivated by:** 2604.18805 (Pass^k decay framing)
**Test:** K=5 sampling on DoM-top-50% MATH-500 subset; compute Pass^5
vs bottom-50% and random-50% controls. 4 h H100.
**Requires:** H100 pod, Qwen-2.5-1.5B.
**Would change:** Confirm → F-8 is restricted to single-trial pipelines;
selective-prediction value proposition narrows. Reject → F-8 generalizes
to multi-trial reliability, strengthening it.
**Blocks:** Any deployment story that depends on selective prediction.

### H-599: F-3 prefill/final orthogonality is position-driven, not content-driven
**Priority:** MEDIUM
**Motivated by:** 2604.18805 (reasoning topology invariant across
domain groups)
**Test:** Extract L19 DoM at two prefill positions (original + 5
tokens earlier) per MATH-500 problem; compute cosine similarity. 30
min CPU on cached prefill activations.
**Requires:** Cached prefill NPZ (already in P11).
**Would change:** Confirm (cos near 0) → F-3 framing rewritten as
positional artifact; "two distinct decision processes" claim removed.
Reject (cos high) → F-3 strengthens.
**Blocks:** nothing.

---


### H-600: Closed-loop A-LQR steering on prefill L19 DoM moves MATH-500 K=1 accuracy
**Priority:** HIGH
**Motivated by:** 2604.19018 + F-2 + H-1
**Test:** Use the public lqr-activation-steering repo on Qwen-2.5-1.5B with v_k =
cached prefill L19 DoM. Run A-LQR over the 257 incorrect MATH-500 problems with
λ ∈ {0.5, 1, 2, 5}. Measure K=1 accuracy delta vs ActAdd open-loop and S-PID. ~1 day H100.
**Requires:** H100 pod, Qwen-2.5-1.5B, cached prefill L19 DoM, public A-LQR repo.
**Would change:** If A-LQR flips ≥10% of incorrect → correct without large PPL/MMLU loss,
F-2 promotes from "predictive direction" to "causal direction"; H-1 is confirmed as
operational. If A-LQR cannot flip any incorrect→correct despite hitting the β_k* setpoint
(Cor. 4.3 tracking guarantee), F-2 is exposed as correlational (problem-difficulty
confound) and H-1 is refuted in its current form.
**Blocks:** any future P10-FE that assumes prefill DoM is causally manipulable.

### H-601: Layer-19 Jacobian alignment is content-stable (B/D/A buckets indistinguishable)
**Priority:** HIGH
**Motivated by:** 2604.19018 §5.1 + F-3 + F-5 + H-17
**Test:** Extract ∂φ_19/∂z at 150 cached prefill activations (50 each B/D/A bucket) on
Qwen-2.5-1.5B. Compute pairwise top-m=64 subspace similarity per Eq. 24. Plot 150×150
grid grouped by bucket. ~5h total compute.
**Requires:** H100 pod for Jacobian extraction (autograd.functional.jacobian on a 1.5B
model is VRAM-tight but feasible at d=1536), CPU for similarity score.
**Would change:** If sim_m ≈ 0.6 with no visible bucket-block structure, F-3's
"orthogonal computations" interpretation is undermined and reduces to "composed layer
cascade"; H-17's "RoPE-mechanical" framing weakens (the rotation is just A_k composition,
not RoPE-specific). If bucket blocks are visible (B vs D vs A), then layer-19 dynamics
ARE content-sensitive at the bucket level — corroborates F-7 mechanistically and is a
substrate for F-5.
**Blocks:** H-21 (D-bucket prefill attention) and H-17 framing decisions.

### H-602: Linearized-cascade PR matches actual PR (refutes F-5 content-dependence)
**Priority:** MEDIUM
**Motivated by:** 2604.19018 §5.1 + F-5 + F-1
**Test:** With cached Jacobians from H-601, apply LTI cascade A_28 ∘ ... ∘ A_1 to z_0 for
50 high-peak-PR and 50 low-peak-PR Stage 2 prompts. Compute PR of linearized trajectory;
compare to actual PR. ~2h CPU.
**Requires:** Cached Jacobians from H-601 result, cached Stage 2 activations.
**Would change:** If linearized PR matches actual PR within ε<0.05, F-5 collapses to
amplitude scaling — breathing is dynamics-dictated, not content-dictated. If linearized PR
is qualitatively different (flat, monotonic, or shifted-peak), F-5 is corroborated and
breathing is genuine content-mediated nonlinear computation.
**Blocks:** H-9 (breathing amplitude predicts difficulty) interpretation.

---


### H-603: F-8's selective-prediction lift is dominated by uncertainty filtering, not correctness signal
**Priority:** HIGH
**Motivated by:** 2604.19974 + F-8 (71.6 % at 50 % coverage)
**Test:** Compute MaxProb / mean-token-probability over the 500 cached 1.5B MATH-500 completions; rank; take top-50 % most confident; measure accuracy on that subset. Also compute semantic-entropy approximation if logits saved. Compare both to F-8's 71.6 % at the same coverage. ~30 min CPU on cached logs.
**Requires:** Cached generation logits in `pathway11_h100/prefill_gated_compute/`. CPU only.
**Would change:** If MaxProb-selected accuracy at 50 % coverage ≈ 71.6 %, F-8's claim that DoM enables non-trivial selective prediction collapses into "DoM ≈ confidence filter." If MaxProb is materially worse (≤ 67 %), F-8 holds and the project's headline narrative is preserved.
**Blocks:** Nothing — the cleanest decisive cross-check we can run before any SAE work.

### H-604: F-2 DoM is causally inert in 1.5B generation despite AUROC 0.7731
**Priority:** HIGH
**Motivated by:** 2604.19974 (pure-incorrectness features functionally inert; only confounded features causal) + F-2 (AUROC 0.7731)
**Test:** Subtract a scaled DoM projection from L19 prefill residuals; re-run the 1.5B forward pass from L19 onward on MATH-500 (Stage-2 hot-restart from cached pre-L19 KV); measure delta accuracy and delta token entropy. Sweep the scaling coefficient over [-2σ, +2σ]. Compare against random-orthogonal-direction null at matched norm.
**Requires:** H100 for ~1 day; modified inference loop that surgically edits residuals at a single layer.
**Would change:** If accuracy and entropy both move significantly (and entropy inverse-tracks accuracy), F-2 DoM is causally upstream — a "correctness driver." If only entropy moves but accuracy is flat, F-2 is a confounded uncertainty feature in Patel's typology — its 0.7731 AUROC is downstream of confidence routing. If neither moves, F-2 is a pure-readout direction (statistical only) and H-1's steering program is in trouble.
**Blocks:** H-1 (per-position DoM steering) — if H-604 returns "inert," H-1 should be parked or reframed.

### H-605: Prefill DoM ⊥ final-token DoM (cos 0.046, F-3) reflects an uncertainty-axis vs correctness-axis decomposition rather than two independent computations
**Priority:** MEDIUM
**Motivated by:** 2604.19974 (uncertainty / correctness functional dissociation) + F-3 (cos 0.046)
**Test:** Once any SAE basis is available for Qwen-2.5-1.5B (P10-FE5 dependency, or a transferred basis), project both prefill and final-token DoMs into the SAE feature space. Classify each feature as entropy-correlated vs flat-entropy on MATH-500. Measure: how much of prefill DoM mass lives on entropy-correlated features? How much of final DoM mass? If they cleanly partition, F-3's "two independent computations" frame is replaced by "two basis vectors of the same uncertainty/correctness 2-D substrate."
**Requires:** SAE for Qwen-2.5-1.5B L19 (blocked on P10-FE5) + ~1 day CPU analysis.
**Would change:** If clean partition, F-3 becomes corroborative evidence for Patel's dissociation, not a free-standing finding. If both DoMs project onto the same feature subset (so the orthogonality is a basis rotation within one subspace), F-3's mechanism is "two probes of the same low-rank substrate" — consistent with P11-FE7's low-dim manifold hypothesis. If both project onto disjoint feature subsets that are not entropy-correlated, F-3 is mechanistically novel and Patel's frame doesn't apply.
**Blocks:** F-3 narrative claims; P10-FE5 readouts.

---


### H-606: Margin-tail exponential functional explains the prefill DoM signal
**Priority:** HIGH
**Motivated by:** 2604.20614 + F-2, F-8
**Test:** On cached pathway11_h100 prefill log-probs and correctness labels for 1.5B
MATH-500, compute ∑_i exp(-margin_i / τ) for τ ∈ {0.1, 0.5, 1.0, 2.0}. Bin problems
by margin-tail percentile and compute prefill DoM contribution per bin. If the
margin-tail functional rank-orders DoM contribution monotonically, the prefill DoM
signal collapses to a margin-tail readout. ~1h CPU.
**Requires:** No new data. `pathway11_h100/prefill_gated_compute/results.json` plus
the L19 prefill NPZ already on disk.
**Would change:** Confirm → F-2 is downgraded from "intrinsic geometric signal" to
"margin-calibration readout"; F-8's 71.6% becomes a calibration-floor result. Reject
→ prefill DoM is independent of curvature-style margin tails, strengthening F-2.
**Blocks:** H-4 reformulation (without this, we can't tell if H-4 is one axis or two).

### H-607: Training-time margin-aware fine-tune of Qwen-2.5-1.5B eliminates the prefill DoM AUROC lift
**Priority:** MEDIUM
**Motivated by:** 2604.20614 + F-2
**Test:** Brief instruction-fine-tune (1-2 epochs on GSM8K) of Qwen-2.5-1.5B with a
margin-tail + Hessian-Frobenius regulariser approximating Morosini et al.'s
objective. Re-extract L19 prefill DoM on MATH-500. If AUROC drops from 0.7731 to
≤0.55, the original signal was a calibration artefact of under-regularised
pretraining. ~1 day H100 + 1h CPU re-eval. Run only if H-606 confirms.
**Requires:** H100 pod, GSM8K train split, ability to back-prop through Qwen-2.5.
**Would change:** Strong refutation of F-2 if AUROC collapses; strong confirmation
of F-2 as model-independent if AUROC survives the fine-tune. Either result is
load-bearing.
**Blocks:** Any "DoM as universal LLM correctness probe" claim.

### H-608: ECE-curvature coupling holds at the residual-stream level
**Priority:** LOW
**Motivated by:** 2604.20614 + F-1, F-10
**Test:** For Qwen-2.5-1.5B on MATH-500, plot per-problem prefill ECE-proxy against
breathing amplitude (peak-PR per problem). If linear, residual-stream geometry
inherits the ECE-curvature coupling. ~30min CPU. Cheap exploratory plot, not a
priority signal.
**Requires:** No new data.
**Would change:** Confirm → unifies F-1 (breathing) with calibration literature;
provides a bridge to the broader theoretical framework. Reject → breathing is not
a curvature phenomenon at all.
**Blocks:** Nothing.

---


### H-609: Prefill DoM AUROC is supported by class-mean dispersion, not by a clean Fisher discriminant axis
**Priority:** HIGH
**Motivated by:** 2604.20817 + F-2 + EXP-029
**Test:** Compute λ_max(S_W^{-1} S_B), Tr(S_B)/N, cond(S_W) on cached L19 prefill activations (Qwen 1.5B, MATH-500). Compare to Theorem 1's bound: λ_max ≤ Φ_T / (N · λ_min(S_W)). If λ_max is two orders of magnitude smaller than the bound while AUROC stays at 0.7731, the prefill DoM probe is exploiting class-mean dispersion in a high-condition-number S_W rather than a stable separating direction. ~20min CPU.
**Requires:** CPU only, cached prefill NPZs.
**Would change:** Confirm — F-2 stays as a useful predictor but its mechanistic interpretation as "the correctness direction" is wrong; reframe as "the class-mean axis under noisy within-class scatter." Reject — F-2 has a clean discriminant and the DoM-as-direction framing is correct.
**Blocks:** H-1 (per-position DoM steering), H-13 (head-level attribution).

### H-610: Length-band-stratified label-shuffle preserves a substantial fraction of prefill DoM AUROC
**Priority:** HIGH
**Motivated by:** 2604.20817 + F-2 + F-8
**Test:** Shuffle correct/incorrect labels within length quartiles of MATH-500. Refit the logistic probe at L19 prefill across 30 runs (3 seeds × 10-fold CV). If the shuffled AUROC distribution's 95th percentile is above 0.65, the original 0.7731 has substantial length-confound contribution. Direct analog of Fu et al. Swap Numbers (Table 1, Fig. 4). ~1h CPU.
**Requires:** CPU only, cached prefill NPZs, length annotations.
**Would change:** Confirm — F-2 and F-8 are partially length-confounded and headline AUROC drops once length is regressed out. Reject — the correctness signal survives label-band shuffling and F-2 / F-8 are robust to the obvious confound.
**Blocks:** H-9 (breathing amplitude predicts difficulty — needs length deconfounding first).

### H-611: Cross-architecture matched-data test of F-1 dissociates spectral from geometric breathing
**Priority:** MEDIUM
**Motivated by:** 2604.20817 + F-1
**Test:** Apply the F-1 breathing extractor to Fu et al.'s released 300M Transformer / Gated DeltaNet / Mamba-2 / LSTM checkpoints (FineWeb-Edu, matched data). Measure (a) the breathing PR trace shape and (b) the correctness-predictive AUROC of the resulting per-layer features. If (a) is universal across architectures but (b) dissociates the way Cohen's κ does in Fu et al. (Transformer/Linear-RNN ≈ 0.7+; LSTM ≈ 0.5), F-1 is "spectral universal" not "functionally universal". 1 H100 day.
**Requires:** H100, MATH-500 adapter for 300M pretraining-only checkpoints (since these are not chat-tuned, breathing must be defined over next-token probability rather than answer correctness).
**Would change:** Confirm — F-1 universality scope narrows from "breathing is a functional signature" to "breathing is a training-data spectral artifact." Reject — F-1 holds and Fu et al.'s spectral/geometric dichotomy does not extend to inference-time breathing.
**Blocks:** F-1 generalization claim; H-22 abstract-CoT compressed breathing.

---


### H-612: PANL probe on Qwen 1.5B MATH-500 beats prefill DoM at L19
**Priority:** HIGH
**Motivated by:** 2604.22271 (PANL verification AUROC 0.986 on Gemma TriviaQA, A2-correctness 0.774 on Qwen 7B) + F-2
**Test:** Locate post-answer-newline token in cached P11 K=1 generations; extract L19 (and layer sweep) residuals from cached NPZs; train L₂-logistic (C=0.001, 5-fold stratified CV) against K=1 correctness; compare AUROC to prefill L19 DoM 0.7731.
**Requires:** CPU only, cached pathway11_h100 1024-tok residuals + results.json correctness labels, 20 min.
**Would change:** If PANL AUROC ≥ 0.85, F-2 must be revised — the headline correctness signal is post-answer, not prefill. If PANL AUROC ≈ prefill (0.75–0.80), prefill remains co-equal headline. If PANL AUROC < prefill, prefill DoM is the unambiguous winner for math reasoning (paper's TriviaQA result does not transfer).
**Blocks:** H-613, H-614.

### H-613: Prefill DoM is a problem-difficulty axis, not an evaluative-confidence axis
**Priority:** HIGH
**Motivated by:** 2604.22271 (orthogonality reframed as generative-vs-evaluative; PANL r=0.02 with answer logprob within incorrect) + F-3
**Test:** On the same 500 MATH problems, compute Pearson r and cosine between prefill-L19-DoM scores and PANL-L19-DoM scores (after H-612 produces PANL probe). Threshold: |r| < 0.2 ⇒ prefill is not evaluative.
**Requires:** CPU, after H-612, 30 min.
**Would change:** If |r| < 0.2, prefill is reframed as a *difficulty/familiarity* axis (collapses with H-5); F-3's "two confidence directions" reading is wrong and should be replaced with "one evaluative axis (PANL/post-answer) + one problem-difficulty axis (prefill)."
**Blocks:** nothing (refines F-3 / H-5).

### H-614: PANL is causally sufficient at L19 in Qwen 1.5B
**Priority:** MEDIUM
**Motivated by:** 2604.22271 (PANL alone at L30 restores 74% d′ in Gemma when answer corrupted)
**Test:** Mean-ablate answer-token activations on 1000 MATH-500 problems (100-trial calibration), restore PANL at L19, measure d′ recovery on a verification head. Threshold: ≥ 50% recovery ⇒ causally sufficient.
**Requires:** 1 H100 day, blocked on H100 pod, after H-612.
**Would change:** If sufficiency holds, P11-FE19 (verify-then-correct routing) is justified at the mechanism level, not just the probe-correlation level. If sufficiency fails, our verify-then-correct pipeline is at best correlational and cannot claim a Kumaran-style causal architecture.
**Blocks:** nothing.

### H-615: D-bucket collective signature generalizes to foil verification (not self-generation-unique)
**Priority:** MEDIUM
**Motivated by:** 2604.22271 (foil correctability AUROC 0.813–0.858 on non-self-generated answers) + F-7
**Test:** 100 MATH problems where 1.5B answered wrongly; paste 7B's (different) wrong answer into 1.5B's verification prompt; extract L19 PANL residuals; run pathway-9 collective-PR pipeline; compare to native D-bucket signature.
**Requires:** 0.5 H100 day, fresh verification-prompt generations.
**Would change:** Confirm ⇒ F-7 must broaden from "unique to self-generation dynamics" to "general property of question-answer-fit evaluation"; this would be a substantial reframing. Reject ⇒ F-7's self-generation specificity is preserved and strengthened.
**Blocks:** nothing.

### H-616: TF-IDF surface-feature baseline does not explain prefill DoM 0.7731
**Priority:** HIGH (cheap, pre-empts a strong refutation channel)
**Motivated by:** 2604.22271 (TF-IDF ceiling 0.564 vs PANL 0.986)
**Test:** Build 100 question + 100 answer TF-IDF dims + token-length features on cached MATH-500 generations; train logistic against K=1 correctness; compute AUROC. Threshold for safety: < 0.65.
**Requires:** CPU, 30 min, cached generation text already on disk.
**Would change:** AUROC < 0.6 ⇒ prefill DoM is genuine representation (control passes, F-2 strengthened). AUROC ≥ 0.65 ⇒ prefill DoM is partially shallow-statistics-explained; F-2 needs a TF-IDF-residualized re-statement.
**Blocks:** nothing.

---


### H-617: PANL probe on Qwen 1.5B MATH-500 beats prefill DoM at L19
**Priority:** HIGH
**Motivated by:** 2604.22271 (PANL verification AUROC 0.986 on Gemma TriviaQA, A2-correctness 0.774 on Qwen 7B) + F-2
**Test:** Locate post-answer-newline token in cached P11 K=1 generations; extract L19 (and layer sweep) residuals from cached NPZs; train L₂-logistic (C=0.001, 5-fold stratified CV) against K=1 correctness; compare AUROC to prefill L19 DoM 0.7731.
**Requires:** CPU only, cached pathway11_h100 1024-tok residuals + results.json correctness labels, 20 min.
**Would change:** If PANL AUROC ≥ 0.85, F-2 must be revised — the headline correctness signal is post-answer, not prefill. If PANL AUROC ≈ prefill (0.75–0.80), prefill remains co-equal headline. If PANL AUROC < prefill, prefill DoM is the unambiguous winner for math reasoning (paper's TriviaQA result does not transfer).
**Blocks:** H-618, H-619.

### H-618: Prefill DoM is a problem-difficulty axis, not an evaluative-confidence axis
**Priority:** HIGH
**Motivated by:** 2604.22271 (orthogonality reframed as generative-vs-evaluative; PANL r=0.02 with answer logprob within incorrect) + F-3
**Test:** On the same 500 MATH problems, compute Pearson r and cosine between prefill-L19-DoM scores and PANL-L19-DoM scores (after H-617 produces PANL probe). Threshold: |r| < 0.2 ⇒ prefill is not evaluative.
**Requires:** CPU, after H-617, 30 min.
**Would change:** If |r| < 0.2, prefill is reframed as a *difficulty/familiarity* axis (collapses with H-5); F-3's "two confidence directions" reading is wrong and should be replaced with "one evaluative axis (PANL/post-answer) + one problem-difficulty axis (prefill)."
**Blocks:** nothing (refines F-3 / H-5).

### H-619: PANL is causally sufficient at L19 in Qwen 1.5B
**Priority:** MEDIUM
**Motivated by:** 2604.22271 (PANL alone at L30 restores 74% d′ in Gemma when answer corrupted)
**Test:** Mean-ablate answer-token activations on 1000 MATH-500 problems (100-trial calibration), restore PANL at L19, measure d′ recovery on a verification head. Threshold: ≥ 50% recovery ⇒ causally sufficient.
**Requires:** 1 H100 day, blocked on H100 pod, after H-617.
**Would change:** If sufficiency holds, P11-FE19 (verify-then-correct routing) is justified at the mechanism level, not just the probe-correlation level. If sufficiency fails, our verify-then-correct pipeline is at best correlational and cannot claim a Kumaran-style causal architecture.
**Blocks:** nothing.

### H-620: D-bucket collective signature generalizes to foil verification (not self-generation-unique)
**Priority:** MEDIUM
**Motivated by:** 2604.22271 (foil correctability AUROC 0.813–0.858 on non-self-generated answers) + F-7
**Test:** 100 MATH problems where 1.5B answered wrongly; paste 7B's (different) wrong answer into 1.5B's verification prompt; extract L19 PANL residuals; run pathway-9 collective-PR pipeline; compare to native D-bucket signature.
**Requires:** 0.5 H100 day, fresh verification-prompt generations.
**Would change:** Confirm ⇒ F-7 must broaden from "unique to self-generation dynamics" to "general property of question-answer-fit evaluation"; this would be a substantial reframing. Reject ⇒ F-7's self-generation specificity is preserved and strengthened.
**Blocks:** nothing.

### H-621: TF-IDF surface-feature baseline does not explain prefill DoM 0.7731
**Priority:** HIGH (cheap, pre-empts a strong refutation channel)
**Motivated by:** 2604.22271 (TF-IDF ceiling 0.564 vs PANL 0.986)
**Test:** Build 100 question + 100 answer TF-IDF dims + token-length features on cached MATH-500 generations; train logistic against K=1 correctness; compute AUROC. Threshold for safety: < 0.65.
**Requires:** CPU, 30 min, cached generation text already on disk.
**Would change:** AUROC < 0.6 ⇒ prefill DoM is genuine representation (control passes, F-2 strengthened). AUROC ≥ 0.65 ⇒ prefill DoM is partially shallow-statistics-explained; F-2 needs a TF-IDF-residualized re-statement.
**Blocks:** nothing.

---


### H-622: Reasoning-step ordering is load-bearing for prefill DoM AUROC at our scale
**Priority:** HIGH
**Motivated by:** 2604.22709 (Ramji et al. Table 3a, verbal CoT lost 8-11 pts under turn-level permutation) + F-2
**Test:** Permute reasoning steps within cached 1.5B and 7B MATH-500 verbal CoTs, regenerate answer, recompute prefill-vs-final DoM AUROC. ~3h H100. P11-FE789.
**Requires:** cached pathway11_h100 generations + ~3h H100 for regen
**Would change:** confirms — F-2 is robust to permuted CoT, prefill genuinely "knows in advance" independent of reasoning coherence; rejects — F-2's 0.7731 is partially a property of orderly reasoning land
**Blocks:** nothing

### H-623: Prefill DoM AUROC at L19 is a prompt-encoding signal, not a verbal-CoT-anticipation signal
**Priority:** HIGH
**Motivated by:** 2604.22709 (Ramji et al. §3.2 Eq. 3, block-structured attention mask isolating discrete latent bottleneck) + F-2
**Test:** Re-run prefill DoM AUROC with verbal CoT context masked / PAD-replaced / neutral-prefix-replaced downstream. ~1h CPU on cached prefills. P11-FE790.
**Requires:** cached prefill NPZs only
**Would change:** confirms — F-2 framing as "prompt encoding tells you the answer" survives; rejects — prefill DoM encodes anticipated verbal trajectory, F-2 needs reframing as joint prompt+expected-trace direction
**Blocks:** H-22 (interpretation depends on what "prefill" means under Abstract-CoT bottleneck)

### H-624: Abstract-CoT collapses prefill-final-token orthogonality (cos > 0.3)
**Priority:** MEDIUM
**Motivated by:** 2604.22709 + F-3 (cos = 0.046 for verbal CoT)
**Test:** After fine-tuning Qwen2.5-1.5B with Abstract-CoT (P11-FE791 / H-22), compute cos(prefill_DoM_abs, final_DoM_abs). 1 day CPU on top of FE74.
**Requires:** completion of P11-FE791 (~1 week H100); then trivial CPU.
**Would change:** confirms — F-3 was specific to long verbal scratchpads; rejects — orthogonality is intrinsic to the L19 representation, survives compression
**Blocks:** nothing

### H-625: Prefill DoM AUROC saturates at top-64 PCA components, mirroring Abstract-CoT M=64 saturation
**Priority:** MEDIUM
**Motivated by:** 2604.22709 (Fig. 5-7 saturation at M=64) + F-2
**Test:** Re-fit prefill DoM probe on top-k PCA features of L19, k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}. Plot AUROC vs k. P11-FE793. ~2h CPU.
**Requires:** cached prefill NPZs only
**Would change:** convergent saturation supports a shared "effective rank" of reasoning at 1.5B scale; divergence (e.g. our probe saturates at 8 or 256) refutes the analogy
**Blocks:** nothing

### H-626: Continuous truncation degradation of prefill DoM AUROC follows accuracy curve
**Priority:** HIGH
**Motivated by:** 2604.22709 (Table 3b, k=32 truncation drops 7-11 pts for verbal CoT) + PROJECT_RECORD §1d (256-tok-vs-1024-tok confound) + F-2
**Test:** Truncate cached MATH-500 verbal CoTs at k ∈ {16, 32, 64, 128, 256, 512, 1024}, regenerate answer, measure accuracy and prefill-DoM AUROC at each k. P11-FE788. ~1h CPU + regen passes.
**Requires:** cached pathway11_h100 generations + ~3h H100 for K=1 regen at multiple truncation lengths
**Would change:** if AUROC tracks accuracy continuously, the 256-tok confound is a smooth gradient and our 1024-tok labels are stable; if AUROC stays flat while accuracy drops, prefill DoM is a robust signal that survives truncation collapses
**Blocks:** nothing

---


### H-627: Reasoning-step ordering is load-bearing for prefill DoM AUROC at our scale
**Priority:** HIGH
**Motivated by:** 2604.22709 (Ramji et al. Table 3a, verbal CoT lost 8-11 pts under turn-level permutation) + F-2
**Test:** Permute reasoning steps within cached 1.5B and 7B MATH-500 verbal CoTs, regenerate answer, recompute prefill-vs-final DoM AUROC. ~3h H100. P11-FE796.
**Requires:** cached pathway11_h100 generations + ~3h H100 for regen
**Would change:** confirms — F-2 is robust to permuted CoT, prefill genuinely "knows in advance" independent of reasoning coherence; rejects — F-2's 0.7731 is partially a property of orderly reasoning land
**Blocks:** nothing

### H-628: Prefill DoM AUROC at L19 is a prompt-encoding signal, not a verbal-CoT-anticipation signal
**Priority:** HIGH
**Motivated by:** 2604.22709 (Ramji et al. §3.2 Eq. 3, block-structured attention mask isolating discrete latent bottleneck) + F-2
**Test:** Re-run prefill DoM AUROC with verbal CoT context masked / PAD-replaced / neutral-prefix-replaced downstream. ~1h CPU on cached prefills. P11-FE797.
**Requires:** cached prefill NPZs only
**Would change:** confirms — F-2 framing as "prompt encoding tells you the answer" survives; rejects — prefill DoM encodes anticipated verbal trajectory, F-2 needs reframing as joint prompt+expected-trace direction
**Blocks:** H-22 (interpretation depends on what "prefill" means under Abstract-CoT bottleneck)

### H-629: Abstract-CoT collapses prefill-final-token orthogonality (cos > 0.3)
**Priority:** MEDIUM
**Motivated by:** 2604.22709 + F-3 (cos = 0.046 for verbal CoT)
**Test:** After fine-tuning Qwen2.5-1.5B with Abstract-CoT (P11-FE798 / H-22), compute cos(prefill_DoM_abs, final_DoM_abs). 1 day CPU on top of FE74.
**Requires:** completion of P11-FE798 (~1 week H100); then trivial CPU.
**Would change:** confirms — F-3 was specific to long verbal scratchpads; rejects — orthogonality is intrinsic to the L19 representation, survives compression
**Blocks:** nothing

### H-630: Prefill DoM AUROC saturates at top-64 PCA components, mirroring Abstract-CoT M=64 saturation
**Priority:** MEDIUM
**Motivated by:** 2604.22709 (Fig. 5-7 saturation at M=64) + F-2
**Test:** Re-fit prefill DoM probe on top-k PCA features of L19, k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}. Plot AUROC vs k. P11-FE800. ~2h CPU.
**Requires:** cached prefill NPZs only
**Would change:** convergent saturation supports a shared "effective rank" of reasoning at 1.5B scale; divergence (e.g. our probe saturates at 8 or 256) refutes the analogy
**Blocks:** nothing

### H-631: Continuous truncation degradation of prefill DoM AUROC follows accuracy curve
**Priority:** HIGH
**Motivated by:** 2604.22709 (Table 3b, k=32 truncation drops 7-11 pts for verbal CoT) + PROJECT_RECORD §1d (256-tok-vs-1024-tok confound) + F-2
**Test:** Truncate cached MATH-500 verbal CoTs at k ∈ {16, 32, 64, 128, 256, 512, 1024}, regenerate answer, measure accuracy and prefill-DoM AUROC at each k. P11-FE795. ~1h CPU + regen passes.
**Requires:** cached pathway11_h100 generations + ~3h H100 for K=1 regen at multiple truncation lengths
**Would change:** if AUROC tracks accuracy continuously, the 256-tok confound is a smooth gradient and our 1024-tok labels are stable; if AUROC stays flat while accuracy drops, prefill DoM is a robust signal that survives truncation collapses
**Blocks:** nothing

---


### H-632: An SAE-decomposed correctness probe at L19 prefill beats the single-direction DoM
**Priority:** HIGH
**Motivated by:** 2604.23829 + F-2 + EXP that produced AUROC 0.7731
**Test:** Cheap path first (P11-FE802): sparse-PCA proxy on cached prefill activations,
5-fold OOF logistic over components, compare AUROC to 0.7731. Heavy path
(P11-FE804): full Qwen L19 SAE training + multi-stage contrastive filter
(target=correct, contrast=incorrect) + probe over retained universe.
**Requires:** CPU only for proxy (~1h on cached NPZ); 1 week H100 for full SAE.
**Would change:** If proxy lift ≥0.04, F-2 should be reframed as "DoM is a low-rank
readout of a richer filtered feature basis." If proxy lift <0.02, F-2's single-direction
framing survives and the SAE investment is unjustified.
**Blocks:** P11-FE803, P11-FE804, P11-FE805.

### H-633: D-bucket problems individually localize to a bridge corridor in an SAE-feature co-occurrence graph
**Priority:** MEDIUM
**Motivated by:** 2604.23829 (bridge concepts §6.1) + F-7
**Test:** P11-FE803: top-50 sparse-PCA components as pseudo-features, problem-level
Jaccard co-occurrence graph, shared-coordinate density visualization for the 36
D-bucket vs 207 A-bucket problems. Per-problem rather than collective localization
to the bridge corridor refutes F-7.
**Requires:** CPU only after FE73, ~1h on cached data.
**Would change:** If D-bucket problems individually localize to a bridge region, F-7
is wrong about granularity and the right unit of analysis is per-problem in a
filtered feature basis. If they remain diffuse, F-7's collective framing survives.
**Blocks:** nothing.

### H-634: Transcoder mechanism graphs carry cross-layer correctness signal that CoE-60 misses
**Priority:** LOW (heavy experiment, blocked behind H-632)
**Motivated by:** 2604.23829 (transcoder mechanism graphs §4.2) + F-9 + F-3
**Test:** P11-FE805: train width-16k transcoder on L18→L19 (or L19→L20) for
Qwen-2.5-1.5B; build mechanism graph; (a) check feature-disjointness of prefill-DoM
vs final-DoM upstream features, (b) train probe over transcoder latents, compare to
0.7731. Lift ≥0.02 refutes F-9's broad "no cross-layer signal" framing.
**Requires:** GPU, ~1 week H100, ~$300.
**Would change:** Confirm reframes F-9 as "CoE-60 was the wrong cross-layer
summary, transcoder latents carry the right one." Reject leaves F-9 intact and
suggests cross-layer information is genuinely null for correctness in this regime.
**Blocks:** nothing.

---


### H-635: Aghajanyan d₉₀ for the MATH-500 correctness probe is in the hundreds, and a single L19 DoM direction lies at the floor of a much larger task-relevant subspace
**Priority:** HIGH
**Motivated by:** 2012.13255 + F-2 + F-9
**Test:** Project cached Qwen-1.5B L19 prefill activations into FastFood random subspaces of dimension k ∈ {1, 5, 10, 50, 200, 1000}, train OOF 5-fold logistic correctness classifier per k, plot AUROC(k). Aghajanyan-style threshold: smallest k achieving 0.9 × 0.7731 = 0.696. P11-FE806. ~30min CPU.
**Requires:** CPU only; cached `pathway11_h100/qwen_1_5b/stage_2_features.npz` L19 prefill activations + MATH-500 K=1 correctness labels.
**Would change:** Confirms → F-2's single-direction framing is empirically at-ceiling for the linear-probe family; reframe headline as "rank-1 sufficient" rather than "DoM is the only direction." Rejects → multi-direction probes climb significantly above 0.77 by k≈200, F-2's headline understates achievable selective-prediction accuracy and invalidates F-9's redundancy framing.
**Blocks:** Nothing.

### H-636: A SAID-style structure-aware multi-layer probe beats single-layer L19 DoM by ≥ 0.02 AUROC, refuting F-9
**Priority:** HIGH
**Motivated by:** 2012.13255 + F-9 + F-3
**Test:** Train probe with d_total = single-layer-DoM dim + 28 λᵢ scaling factors over all-layer cached activations, FastFood projection. Compare AUROC against L19 DoM (0.7731). P11-FE807. ~1h CPU.
**Requires:** CPU only; multi-layer (L0…L27) cached activations.
**Would change:** Confirms → F-9's redundancy claim is wrong; layer structure carries non-redundant correctness signal; CoE-60 result needs reinterpretation. Rejects → SAID's structure-aware advantage doesn't transfer to inference-time probing in autoregressive math models.
**Blocks:** P11-FE807's interpretation; partial overlap with H-13 (head-level attribution).

### H-637: d₉₀ of the correctness probe is monotonically lower for Qwen-7B than Qwen-1.5B at fixed labels
**Priority:** MEDIUM
**Motivated by:** 2012.13255 + F-1
**Test:** Run cross-model d₉₀ protocol on Qwen-1.5B and 7B (and any cached Phi-3/Llama). P11-FE809. ~2h CPU.
**Requires:** CPU only; cached cross-model activations from P11.
**Would change:** Confirms → F-1's "universality" framing is masking scale-dependent compression and the PR-magnitude metric is too coarse; the breathing magnitude story should be re-narrated as "amplitude is universal, but task-relevant subspace shrinks with scale." Rejects → universality claim survives a strong external prediction; strengthens F-1.
**Blocks:** Nothing.

---


### H-638: Final-token True-probe matches prefill L19 DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2205.14334 Table 1 (their Indirect logit final-token True-probe wins MAD on Multiply-divide) + F-2 prefill 0.7731 vs final 0.7186 ordering
**Test:** Append "True/False:" after each Qwen-2.5-1.5B MATH-500 K=1 answer, single forward pass, take logprob("True") at the final position. AUROC against K=1 correctness. Compare to prefill DoM 0.7731 and final-token DoM 0.7186. ~30min H100 (or CPU if final-token logits cached).
**Requires:** Cached Qwen-2.5-1.5B MATH-500 prompts + K=1 answers; one forward pass per problem with appended "True/False:". H100 access.
**Would change:** If True-probe AUROC ≥ 0.77, F-2's "prefill > final" ordering is a probe-design artifact (the Linear DoM probe direction at the final position is suboptimal vs. the True-token unembedding direction). If < 0.72, F-2 ordering survives a strong final-token alternative.
**Blocks:** H-639, H-640 (downstream correlation analyses depend on True-probe scores).

### H-639: 50-shot verbalized confidence achieves selective-prediction parity with supervised DoM
**Priority:** HIGH
**Motivated by:** 2205.14334 Sec. 3.3 + Fig. 6 (50-shot ≈ finetune calibration on GPT-3-175B without weight updates) + F-8 selective-prediction 71.6% at coverage 0.5
**Test:** For each Qwen-2.5-1.5B MATH-500 problem, sample 50 (problem, answer, accuracy-based-confidence) examples from held-out split, build in-context prompt, ask for "Confidence: X%" after K=1 answer, decode via top-5 weighted sum. AUROC + risk-coverage curve. Compare against F-8 E3 numbers. ~1h H100.
**Requires:** Held-out MATH-500 calibration split + accuracy-based confidence labels; H100 for 50-shot decoding × 500 problems.
**Would change:** If 50-shot risk-coverage matches F-8's 71.6% at coverage 0.5, supervised DoM (F-2 / E3) loses its key advantage and H-12 needs to broaden from "SEPs replace DoM" to "any label-free uncertainty signal replaces DoM". If 50-shot lags by >5% absolute, supervised labels carry meaningful extra signal.
**Blocks:** nothing.

### H-640: MATH-500 heuristic features explain prefill DoM AUROC
**Priority:** MEDIUM
**Motivated by:** 2205.14334 Table 2 row 2 (logreg-on-heuristics MSE 29.7 establishes a difficulty-features floor) + H-5 (prefill encodes familiarity vs decomposability)
**Test:** Hand-craft MATH-500 difficulty features (problem-statement length, integer-magnitude max, operator counts, has-LaTeX, level 1-5 tag, novel-token ratio). Logistic regression vs K=1 correctness, hold-out AUROC. Compare to prefill DoM 0.7731. ~30min CPU.
**Requires:** MATH-500 metadata + K=1 correctness labels (already cached).
**Would change:** If heuristic AUROC ≥ 0.72, prefill DoM is mostly difficulty-encoded — H-5's "epistemic decomposability" framing is too strong, and we should reframe F-2 as "in-distribution accuracy memorization." If < 0.65, prefill DoM exceeds surface heuristics and H-5 survives.
**Blocks:** Re-framing of H-5's narrative around "epistemic" vs "memorized" uncertainty.

### H-641: Per-MATH-500-level reporting preserves F-8's 71.6%-at-coverage-0.5
**Priority:** MEDIUM
**Motivated by:** 2205.14334 Fig. 8 (label-distribution shift makes pooled calibration metrics misleading)
**Test:** Stratify F-8 selective-prediction outputs by MATH-500 levels 1-5; compute per-level acc-at-coverage-0.5. ~20min CPU on cached outputs.
**Requires:** F-8 cached probabilities + per-problem level tags.
**Would change:** If per-level numbers all clear 65%, F-8 is a robust signal across difficulty. If one level dominates the answered set and another collapses below 50%, F-8's pooled number is partly an easy-question artifact and selective-prediction reports should be stratified.
**Blocks:** nothing.

---


### H-642: Per-point Euclidicity (TARDIS) on prefill L19 activations matches or beats the supervised DoM AUROC of 0.7731 *without labels*
**Priority:** HIGH
**Motivated by:** 2210.00069 (TARDIS) + F-2 + EXP-021
**Test:** Compute TARDIS Euclidicity for each MATH-500 prompt's L19 prefill activation (PCA-project to 128-dim first because Vietoris–Rips memory blows up otherwise; intrinsic dim from twoNN; k=50; 20-step (r,s) grid). Score Euclidicity AUROC for K=1 correctness, OOF 5-fold to match F-2's protocol. ~1 day CPU.
**Requires:** CPU; cached `pathway11_h100/prefill_gated_compute` L19 activations; TARDIS pip install; PCA pre-projection.
**Would change:** If Euclidicity AUROC ≥ 0.65, F-2's framing of L19-DoM as a *learned* probe is partly wrong — a fraction is unsupervised geometric singularity. If Euclidicity AUROC < 0.55, TARDIS-style geometry is uninformative for Qwen residual streams and we close the door on this whole TDA branch.
**Blocks:** H-7 (density-based outlier detection — Euclidicity is a strict generalisation; if Euclidicity wins, demote H-7).

### H-643: Per-token PID is flat across breathing peaks/troughs — i.e. breathing (F-1) is a covariance/sample-statistic phenomenon, not a local-intrinsic-dimension phenomenon
**Priority:** MEDIUM
**Motivated by:** 2210.00069 (TARDIS Theorem 2) + F-1 + F-5
**Test:** On cached pathway-10 temporal NPZs, compute TARDIS PID at every 50th token position for 100 correct + 100 incorrect MATH-500 problems on Qwen-1.5B. Plot mean PID vs token position alongside the participation-ratio temporal curve. ~1 day CPU.
**Requires:** CPU; cached pathway-10 temporal activation NPZs.
**Would change:** If PID oscillates with the same shape as PR, breathing is a real local-dim phenomenon and F-1 is a topological claim. If PID is flat (within ±0.5 of the mean) where PR oscillates by ≥10 ratio units, breathing is a 2nd-moment phenomenon (covariance flattening, not dimension change), and F-1's "universal dimensional breathing" framing should be reworded as "universal participation-ratio breathing."
**Blocks:** H-4 (Breathing ↔ Sharpness Dimension correspondence) — if PID is flat, the EoS-correspondence hypothesis weakens.

### H-644: Local 3-parameter PLH on residual streams escapes the F-10 Gaussian null for at least one layer
**Priority:** HIGH
**Motivated by:** 2210.00069 + F-10 + EXP-018 + EXP-026
**Test:** On 1000 token-residuals per layer ∈ {12, 19, 24, 27} from Pathway 8 cached activations, compute Euclidicity using TARDIS, compare to a covariance-matched Gaussian null of equal sample size, Tukey range test α=0.05. ~4h CPU per layer.
**Requires:** CPU; cached pathway-8 layer-wise NPZs; TARDIS install.
**Would change:** If at least one layer separates from the Gaussian null, F-10 should be re-scoped: "global one-parameter PH is at the Gaussian null; local multi-parameter PLH is not." If all layers match the null, F-10 generalises and we can permanently retire PH-on-residual-streams as a research direction.
**Blocks:** P7-FE1 (zigzag PH) and P7-FE2 (PH on attention graphs) — if local PLH already matches null, those alternative filtrations probably do too.

---


### H-645: Future-Lens linear probe on prefill L19 matches or beats DoM AUROC
**Priority:** HIGH
**Motivated by:** 2311.04897 + F-2
**Test:** Fit a per-layer linear map W: h_T_prefill → h_{T+k}_final on a small Pile slice, apply to cached Qwen-2.5-1.5B prefill L19 activations on MATH-500, decode top-1 token at k=1..10, build a per-problem decoded-confidence feature, compute AUROC against K=1 correctness. Compare directly to 0.7731. Est: 1h on cached NPZs.
**Requires:** CPU; cached `pathway11_h100/` Stage 2 NPZs; ~10MB Pile slice for probe fitting.
**Would change:** If Future-Lens AUROC ≥ 0.75, F-2 collapses into a degenerate look-ahead and the "abstract correctness" framing in PAPER_INDEX / FINDINGS narrative needs to be retracted. If Future-Lens AUROC ≤ 0.65, F-2 is strengthened against the strongest available competing null.
**Blocks:** H-5, H-13.

### H-646: Position-shift transform absorbs F-3 orthogonality
**Priority:** HIGH
**Motivated by:** 2311.04897 + F-3
**Test:** Fit linear T: h_prefill → h_final on cached residual-stream pairs across 500 MATH-500 problems (label-free regression). Compute cos(T·prefill_DoM, final_DoM). Compare to raw 0.046. Est: 30min on cached Stage 2 NPZs.
**Requires:** CPU; cached pairs already in `pathway11_h100/`.
**Would change:** If cos > 0.5, F-3's "directions are orthogonal" framing is reduced to "directions are orthogonal in raw coordinates only" — which materially changes how we describe the prefill / final-token relationship. If cos remains ≤ 0.1 even after the learned shift, F-3 is robust.
**Blocks:** H-13.

### H-647: Prefill L19 DoM lives in the MLP-out subspace, not attn-out
**Priority:** MEDIUM
**Motivated by:** 2311.04897 + F-2
**Test:** Single re-extract pass on Qwen-2.5-1.5B with hooks recording L19 attn_out and mlp_out separately. Fit DoM and compute AUROC on each component. ~1h H100.
**Requires:** H100 pod (small); cached MATH-500 prompts and labels.
**Would change:** If DoM AUROC drops near chance when restricted to attn_out and remains high on mlp_out, the DoM signal shares a substrate with the Future-Lens signal (mechanism-level support for Refutation 1). If DoM is attn-dominated, F-2 lives in a genuinely different circuit from Future Lens.
**Blocks:** nothing.

---


### H-648: Prefill L19 DoM AUROC has substantial spurious-template-factor share
**Priority:** HIGH
**Motivated by:** Tan et al. 2407.12404 (Sec 5, Fig 4) + F-2
**Test:** Decompose per-problem OOF scores from `pathway11_h100/prefill_gated_compute/oof_scores.json` against template features (answer length, boxed-token id, leading-digit of answer, $\boxed{}$ presence, prompt-template id). Report fraction of per-problem score variance explained by template features alone vs jointly with correctness label. ~30 min CPU.
**Requires:** CPU only, cached OOF scores + MATH-500 metadata.
**Would change:** If template features explain ≥40% of per-problem score variance, F-2's framing shifts from "L19 DoM = correctness representation" to "L19 DoM = template+correctness mixture", and downstream F-8 selective-prediction interpretation needs rephrasing. If <20%, F-2 strengthens significantly.
**Blocks:** H-1 framing — should not promote DoM steering as a "correctness intervention" until this decomposition is in.

### H-649: F-3 prefill⊥final-token orthogonality is prompt-distribution-driven, not two-circuit
**Priority:** HIGH
**Motivated by:** Tan et al. 2407.12404 (Sec 6 + App E.3) + F-3 + H-17
**Test:** Re-extract prefill and final-token L19 mean activations on 50 cached MATH-500 problems under three prompt variants (BASE, USER_POS, SYS_POS). Compute pairwise cos similarities. Predict: cos(prefill_BASE, prefill_SYS_POS) ≈ cos(prefill_BASE, final_BASE) ≈ 0.05 if prompt-distribution effect dominates; ≫ 0 if there really are two distinct circuits.
**Requires:** ~1 h H100, reuses Stage 2 extractor on 50 problems.
**Would change:** Confirm → F-3 demoted from "two-circuit" to "prompt-distribution-rotation" finding; H-17 (RoPE-mechanical) becomes the parsimonious explanation. Reject → F-3 strengthens substantially as a genuine two-circuit signal.
**Blocks:** nothing critical, but should run before any paper-write-up cites F-3 as a circuits result.

### H-650: H-1 per-position DoM steering on MATH-500 will produce ≥30% anti-steered problems
**Priority:** MEDIUM (pre-registration only — not a separate experiment)
**Motivated by:** Tan et al. 2407.12404 (Fig 1) + H-1
**Test:** Pre-registration. When H-1 runs, classify each MATH-500 problem as steered, neutral, or anti-steered based on sign of per-problem accuracy delta at the maximum-aggregate-effect λ. Predict ≥30% anti-steered (Tan-equivalent rate on MWE personas).
**Requires:** No additional GPU; piggy-backs on H-1 outputs.
**Would change:** Confirm (≥30% anti-steered) → H-1 cannot be claimed as a reliable intervention even if aggregate accuracy lifts; reframe as "DoM steering helps a subset, harms another subset". Reject (<10% anti-steered) → MATH-500 + correctness is a regime where SV reliability is much better than MWE personas, and Tan's critique doesn't transfer at full strength.
**Blocks:** premature claims of "DoM controls correctness".

---


### H-651: Multi-D matched-capacity calibration head closes the prefill > final asymmetry
**Priority:** HIGH
**Motivated by:** 2408.10764 + F-2
**Test:** Train an Otter-style head (~+128 FFN inner dim, +4 attention heads) on Qwen-2.5-1.5B with K=1 MATH-500 correctness as a regression target. Evaluate AUROC at prefill and final-token positions from the same trained head. Compare to F-2's 0.7731 / 0.7186 split.
**Requires:** H100 day, Qwen-2.5-1.5B base, MATH-500 K=1 correctness labels (already available in `pathway11_h100/prefill_gated_compute/results.json`).
**Would change:** If final-token AUROC ≥ prefill AUROC under the multi-D head, F-2's "prefill is uniquely strong" framing becomes a 1-D-probe artifact. If prefill remains better at matched capacity, F-2 strengthens.
**Blocks:** H-652, H-653.

### H-652: MHA insertion outperforms FFN insertion for correctness prediction
**Priority:** MEDIUM
**Motivated by:** 2408.10764 (Table 4) + H-13
**Test:** At matched parameter count, train FFN-only vs MHA-only Otter heads on Qwen-2.5-1.5B for MATH-500 K=1 correctness. Compare AUROC and selective-prediction-at-coverage-0.5.
**Requires:** H100 day, on top of H-651 codebase.
**Would change:** If MHA insertion wins, attention heads — not MLPs — are the primary site of correctness routing, supporting H-13's head-level attribution framing. If FFN wins, the De Cao et al. "FFN is knowledge" prior holds for correctness.
**Blocks:** nothing.

### H-653: Per-block correctness-prediction Otter beats single-layer Otter at matched capacity
**Priority:** MEDIUM
**Motivated by:** 2408.10764 + F-9
**Test:** Compare three Otter variants on MATH-500 K=1 correctness at matched param count: only-L19, L0..L19, all-32. Measure AUROC.
**Requires:** H100 day, on top of H-651 codebase.
**Would change:** If multi-layer beats single-layer by ≥ 2 AUROC points, F-9's "CoE-60 redundant with single-layer L19 DoM" interpretation needs revision: signal is spread across layers, single-layer probes are leaving capacity on the table.
**Blocks:** nothing.

---


### H-654: GCAV-style adaptive-ε steering on prefill L19 DoM moves MATH-500 K=1 accuracy
**Priority:** HIGH
**Motivated by:** 2501.05764 + H-1 + F-2
**Test:** Implement Eq 6 from GCAV using cached `w_DoM, b_DoM` from `phase2_prefill_dom.npz`. For each MATH-500 problem on Qwen-2.5-1.5B where the prefill DoM probability < 0.5 ("predicted incorrect"), compute `ε_i = (sigmoid⁻¹(0.5) − b − wᵀe_i) / ||w||` and add `ε_i · v_DoM` to the L19 residual stream at the prefill last-token. Regenerate K=1 with greedy decoding. Compare K=1 accuracy vs baseline 48.6% (243/500).
**Requires:** H100, Qwen-2.5-1.5B already loaded; cached prefill activations from P11; ~8h
**Would change:** On confirm (Δ ≥ +2pp): H-1 promoted from PARKED to active; F-2 reframed from "predictive only" to "predictive + causal"; plausible publication-quality finding. On reject: F-2 stays purely predictive, and H-1 should be downgraded with a methodological note that even the strongest known adaptive-steering recipe (GCAV) does not move correctness, suggesting correctness is not a single-direction concept in the GCAV sense.
**Blocks:** H-655

### H-655: Per-layer CAV probe AUROC and causal steering lift dissociate on Qwen-2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2501.05764 + F-9
**Test:** For L0..L27 on Qwen-2.5-1.5B, retrain the prefill DoM as a CAV; measure both held-out AUROC and adaptive-ε steering lift on a 50-problem MATH-500 subset. Cross-correlate the two layer-curves. GCAV claims they align; if they don't on a math task, F-9's "single-layer L19 sufficient" claim is overspecified for intervention.
**Requires:** H100 ~12h; cached prefill activations
**Would change:** On confirm dissociation: F-9 narrowed to predictive-only redundancy; multi-layer steering admitted as legitimately different. On confirm alignment: F-9 strengthened.
**Blocks:** nothing

### H-656: F-3's prefill/final-token DoM orthogonality is geometric, not semantic
**Priority:** MEDIUM
**Motivated by:** 2501.05764 + F-3
**Test:** Fit a 2D logistic regression on (prefill_DoM_score, final_DoM_score) and a 1D logistic on each individually. Compute correctness AUROC. If joint − max(individual) < 0.005 (within noise), F-3 stands as written. If joint − max(individual) ≥ 0.02, the two directions carry separable signal and "orthogonal directions encode different things" should be replaced with "orthogonal directions encode partially separable correctness components."
**Requires:** ~30min CPU on cached projections
**Would change:** F-3 narrative re-write.
**Blocks:** nothing

---


### H-657: BOS-token-position prefill activations dominate L19 DoM AUROC
**Priority:** HIGH
**Motivated by:** 2501.09929 + F-2
**Test:** Split MATH-500 prefill positions by token type (BOS, system header, user content). Compute DoM separately per token-type subset and evaluate correctness AUROC of each subset's DoM. If BOS-only subset achieves ≥ 0.70 AUROC and content-only subset is much lower, F-2's 0.7731 is a BOS-contamination artifact. ≈1h CPU on cached `pathway11_h100/prefill_gated_compute/` NPZs.
**Requires:** CPU only; cached NPZs.
**Would change:** On confirm, F-2 narrative needs "AUROC is partly carried by `<bos>` token activations, which is a known SAE-steering confound (FGAA 2501.09929)". On reject, F-2 strengthens (DoM is *not* a BOS artifact).
**Blocks:** P10-FE42, P10-FE44

### H-658: FGAA-filtered SAE features at L19 lift correctness AUROC above 0.7731
**Priority:** MEDIUM
**Motivated by:** 2501.09929 + P10-FE5 + F-2
**Test:** After P10-FE5 trains an SAE on L19 prefill, apply FGAA recipe: zero density>0.01, remove BOS-dominant features, retain top n₁∈[1,8] positive features, L1-normalize. Recompute correctness probe with filtered SAE-feature direction. AUROC > 0.79 confirms; AUROC ≤ 0.78 rejects.
**Requires:** SAE trained on L19 (P10-FE5 unblocker, ≈4h H100), then 1h CPU.
**Would change:** On confirm, F-2's 0.7731 is a floor; project should switch to FGAA-filtered direction as canonical correctness probe. On reject, raw DoM is genuinely the right operator (non-trivial result given FGAA's behavior-task gains).
**Blocks:** any future selective-prediction headline replaces 0.7731 with filtered AUROC.

---


### H-659: Prefill L19 DoM detects correctness but does not steer it
**Priority:** HIGH
**Motivated by:** 2501.17148 + F-2 + F-3
**Test:** Run AxBench DiffMean steering protocol (Eq. 4, two-halves α-selection, harmonic-mean LLM judge) on Qwen-2.5-1.5B at L19 prefill, sweeping 7 steering factors over 50 MATH-500 problems. Report K=1 accuracy delta from 48.6% baseline and judge-scored quality. (See P11-FE831.)
**Requires:** 4h H100 with `pyvene` hooks; cached L19 DoM direction.
**Would change:** Confirm → H-1 stays HIGH and we move to scale-up. Reject → H-1 becomes PARKED, and the project's H-1 rationale ("DoM is a causal handle, not just a probe") is dropped from the narrative; F-2 becomes a strict *detection* finding.
**Blocks:** H-1 (treats this as a precondition test before any further DoM-steering work).

### H-660: ReFT-r1-style joint training closes the H-1 gap when DiffMean fails
**Priority:** MEDIUM
**Motivated by:** 2501.17148 + H-659
**Test:** Train ReFT-r1 (TopK-gated detection + LM-loss steering with L1 regularizer on non-top-k latents) at Qwen-2.5-1.5B L19 with concept = "response-is-correct" on 200 MATH-500 problems; evaluate K=1 accuracy on the remaining 300 vs the prefill-DoM-gated baseline (F-8). (See P11-FE834.)
**Requires:** 1d H100 for training + 4h H100 evaluation; `pyvene` + ReFT codebase.
**Would change:** Confirm → restores H-1 as "DoM is causal *with* the right training procedure" and reframes F-2 as the detection half of a two-axis story. Reject → strong evidence that residual-stream steering is fundamentally hard for correctness, regardless of method.
**Blocks:** nothing (downstream of H-659).

### H-661: F-2 AUROC ceiling is label-induced, not DoM-induced
**Priority:** MEDIUM
**Motivated by:** 2501.17148 (AxBench DiffMean 0.942 mean AUROC) + F-2
**Test:** Synthesize a binary concept "response contains a calculation error" on the same MATH-500 generations using gpt-4o-mini per AxBench §3.1; refit DiffMean / Probe at L19 prefill; report AUROC. (See P11-FE833.)
**Requires:** $5 gpt-4o-mini + 30min CPU on cached activations.
**Would change:** Confirm (AUROC ≥ 0.92) → F-2's 0.7731 reflects correctness-label noise; the ceiling argument shifts to "improve labels, not methods". Reject (AUROC < 0.85) → F-2's 0.7731 is genuinely close to the DoM ceiling for math-style residual-stream encoding.
**Blocks:** nothing.

---


### H-662: Predict-control discrepancy holds for L19 prefill DoM
**Priority:** HIGH
**Motivated by:** 2502.18862 + F-2 (prefill DoM AUROC 0.7731) + EXP-037
**Test:** Compute ROC-AUC of `mean(prefill_activation · v)` on cached 500 MATH-500 prefill activations for v ∈ {prefill DoM, random unit, one-shot optimized SV from H-663}. <30min CPU on cached NPZs. Confirm: optimized SV with strong steering effect should have classifier ROC-AUC well below 0.7731 if the discrepancy holds.
**Requires:** cached `pathway11_h100/prefill_gated_compute/` activations + the optimized SV from H-663; CPU only.
**Would change:** confirm → F-2's mediation reading is downgraded to "predictive correlation only" and the 0.7731 number is reframed as a probe AUROC, not evidence for a causal direction. reject → F-2's causal reading survives a non-trivial challenge from the steering literature.
**Blocks:** H-1 framing — if confirmed, H-1 must be reformulated as "the *optimized* per-position SV moves accuracy" rather than "the *DoM* per-position SV moves accuracy."

### H-663: One-shot optimized L19 SV transfers correctness across MATH-500
**Priority:** HIGH
**Motivated by:** 2502.18862 §3 + §4 (96.9% Harmbench ASR from single example)
**Test:** Pick one B-bucket MATH-500 problem (K=1 wrong, K=8 right). Run Adam on additive L19 SV maximizing log P(correct_completion | prompt; v) at norms {2.5, 5, 7.5, 10, 15}. Evaluate accuracy lift on held-out 499. 2h on 4090 / 1h H100.
**Requires:** Qwen-2.5-1.5B loaded + autograd through L19 residual; 4090 sufficient; no new data.
**Would change:** confirm → H-1 has a working steering construction (not the DoM but a one-shot optimized vector); the project gains a per-pathway tool for "make this problem correct" interventions. reject → either L19 doesn't mediate correctness causally or the one-shot method doesn't generalize from refusal/safety to math correctness — both are interesting, the latter is consistent with Turner-et-al.-2025 scale-failure on Gemini 1.5v2.
**Blocks:** H-662, H-664; provides the optimized-SV input to both.

### H-664: L19 correctness signal is a low-dim subspace, not a single direction
**Priority:** MEDIUM
**Motivated by:** 2502.18862 §3.3 (high variance across training examples) + P11-FE7 (SVD of prefill activations)
**Test:** Optimize 5 one-shot correctness-steering SVs on 5 different B-bucket problems (1 per difficulty level if possible). Compute pairwise cos matrix + SVD spectrum. If pairwise cos < 0.5 *and* each SV transfers (≥+5pp on held-out 100), the underlying object is a subspace.
**Requires:** ~5× cost of H-663; same hardware.
**Would change:** confirm → F-2 reframed from "single direction" to "low-dim manifold" — consistent with P11-FE7's framing. reject → "single direction at L19" survives.
**Blocks:** F-2 and F-9 reframing.

---


### H-665: Attention-graph topology (MTop-Div) carries correctness signal orthogonal to L19 prefill DoM on MATH-500
**Priority:** HIGH
**Motivated by:** 2504.10063 + F-2 + F-10
**Test:** Compute MTop-Div_G(R, P) on cached Qwen2.5-1.5B MATH-500 prefills + generated reasoning. Rank heads by Δ_ij on a 50/50 correct/incorrect probe. Report 10-head-averaged AUROC (held-out 400 problems) vs F-2's 0.7731. Then stack [DoM logit, MTop-Div] in 2D logistic regression and report joint AUROC. ~1 day CPU if attention maps cached, +4h H100 if re-extraction needed.
**Requires:** Attention-map NPZs from pathway11_h100/data (verify availability for 1.5B; 7B likely needs re-extraction with capture_attention=True), gudhi or ripser.py, 50 labeled correct/incorrect items.
**Would change:** Confirm → F-10 scope must be restricted to residual point clouds; F-2 gets a near-free attention-side companion probe; potentially refutes F-9 for the attention pathway. Reject → F-10 stands as a topology-wide null and TOHA does not transfer to math reasoning.
**Blocks:** H-666 (induction-head reformulation) — meaningful only if MTop-Div carries signal.

### H-666: Hallucination/correctness-aware heads in Qwen2.5-1.5B are induction/copying heads, not "prefill" or "final-token" heads
**Priority:** MEDIUM
**Motivated by:** 2504.10063 + H-13 + F-3
**Test:** Compute copying-score (Feucht et al. 2025) for every Qwen2.5-1.5B head. Compare ranking against (a) Δ_ij-ranked heads from H-665/FE71, (b) heads whose attention entropy correlates with L19 prefill DoM probe scores. Hypothesis: top-25 copiers contain ≥ 3 of the top 10 hallucination-aware heads, with no preference for the prefill/final partition. ~4h CPU.
**Requires:** Qwen2.5-1.5B in HF, dual-route induction reference impl, attention-extraction pass.
**Would change:** Confirm → H-13 should be reformulated around induction-circuit structure; the prefill/final partition is the wrong axis. Reject → prefill/final remains the natural partition for circuit-level study.
**Blocks:** Reformulation of H-13.

---


### H-667: Prompt-conditional steering (HyperSteer-style) beats static DoM on MATH-500
**Priority:** HIGH
**Motivated by:** 2506.03292 + F-2
**Test:** Train a small hypernet text(problem) + L19 prefill activation → R^{d_model} on cached MATH-500 prefill NPZs (1.5B); regress on correctness. Compare OOF 5-fold AUROC and downstream selective-prediction accuracy at coverage 0.5 against the supervised L19 prefill DoM (AUROC 0.7731, sel-pred 71.6%).
**Requires:** CPU, ~4 h, all data cached in `pathway11_h100/prefill_gated_compute/`.
**Would change:** If hypernet AUROC > 0.80 *and* selective-prediction accuracy > 73%, F-2 is reframed: residual-stream correctness lives in a *prompt-conditional* direction, not a single global one. The DoM is a low-rank approximation. If hypernet ≤ DoM, the single-vector framing is on the Pareto frontier and hypernet is unnecessary complexity.
**Blocks:** Decision on whether H-1's steering bank should be a fixed per-position bank or a learned conditional emitter.

### H-668: Prefill ↔ final-token DoMs are reachable from a shared low-rank conditioning manifold
**Priority:** MEDIUM
**Motivated by:** 2506.03292 + F-3
**Test:** Train a single hypernet to emit two outputs (prefill-target, final-token-target) from the same problem-text encoding. If both outputs match their supervised DoM counterparts to AUROC within 1 pt, fit a low-rank linear map from prefill-emitter weights to final-emitter weights and report the singular-value spectrum.
**Requires:** CPU, ~6 h, cached prefill + final-token NPZs.
**Would change:** If a rank-≤ 8 map captures ≥ 95% of the cross-position variation, F-3's "orthogonal circuits" framing is replaced by "low-rank conditional rotation" — orthogonal in raw cosine but parametrically sibling. If no low-rank map fits, F-3's two-circuit interpretation strengthens.
**Blocks:** Interpretation of F-3 in the eventual writeup.

---


### H-669: RL-with-verifiable-rewards training degrades the prefill-L19 DoM correctness signal in proportion to ECE worsening
**Priority:** MEDIUM
**Motivated by:** 2507.16806 + F-2
**Test:** Take any publicly-available Qwen-2.5-7B-RLVR or Qwen-2.5-7B-RLCR checkpoint
(e.g. from Open-Reasoner-Zero / DeepSeek-R1-distilled releases that reproduce the RLVR
pipeline). Extract prefill-L19 activations on MATH-500. Re-fit the DoM probe with the
same 5-fold OOF protocol used for F-2. Predict: AUROC drops from 0.7731 (base) to ≤ 0.70
(post-RL) in lockstep with ECE worsening from ~0.39 to ~0.26. Estimated time: 4 h H100
to extract activations on a borrowed checkpoint + 30 min CPU to refit the probe.
**Requires:** H100 pod or rented checkpoint, MATH-500 cache (already have), DoM extraction
script (already have).
**Would change:** On confirm, F-2 must be retitled to "prefill L19 DoM is a strong
correctness predictor *on base / RLHF models* but degrades under RLVR" — restricts the
generality of the headline finding. On reject, F-2 holds across post-RL models too,
strengthening the persistence story.
**Blocks:** H-13 (head-level attribution depends on which model state we attribute), H-1
(steering interventions may need different vectors for base vs post-RL).

### H-670: Confidence-weighted majority vote with prefill-DoM-derived confidence beats threshold-based abstention at matched coverage on MATH-500
**Priority:** HIGH
**Motivated by:** 2507.16806 + F-8 + EXP-040
**Test:** On our K=8 MATH-500 cache (Qwen-2.5-1.5B and 7B), compute per-sample
sigmoid-calibrated prefill-L19 DoM as q_i. Compare three selection rules at coverage 0.5:
(a) F-8's current threshold abstention (drop low-DoM-mean problems), (b) max-confidence
(pick highest-q sample per problem and accept if max-q > τ), (c) confidence-weighted
majority vote (weighted vote per problem, accept if winning weight > τ). Predict: CWMV
matches or beats (a) by ≥ 1 pt answered accuracy at coverage 0.5. Estimated time: 1 h CPU.
**Requires:** Existing K=8 cache + per-sample DoM scores (already have).
**Would change:** On confirm, F-8 should be reframed around CWMV, not threshold-abstention,
giving a strictly cheaper test-time scaling story. On reject, threshold-abstention is the
better DoM consumption pattern, which itself is a useful negative result against the
paper's CWMV-superiority claim — possibly indicating DoM is poorly self-consistent across
samples.
**Blocks:** nothing.

---


### H-671: L19 prefill correctness lives in a multi-direction subspace, not a single rank-1 axis
**Priority:** HIGH
**Motivated by:** 2508.19505 + F-2, F-3, F-9
**Test:** Apply Iterative Nullspace Projection (Ravfogel 2004.07667) to cached Qwen-1.5B L19 prefill activations from `pathway11_h100/prefill_gated_compute/`. After fitting initial logistic-regression DoM probe (AUROC 0.7731), project out the separating hyperplane and refit on the residual; repeat 50 rounds. Record per-round AUROC and cumulative AUROC of the d-direction concat probe. ~2 h CPU.
**Requires:** CPU only; cached prefill NPZs already on disk.
**Would change:** If ≥5 INLP rounds yield non-chance AUROC, F-3 (cos = 0.046 between prefill and final DoM) downgrades from "two orthogonal correctness circuits" to "two random draws from a >5-dim correctness subspace." If 2-direction concat lifts AUROC above 0.79, F-9 (CoE-60 ≈ single-layer DoM) gains a clean escape: CoE-60 was matched by the *first* direction; subsequent directions were not represented in the rank-1 DoM. If INLP collapses to chance after 1-2 rounds, F-2/F-3/F-9 are robustly single-direction phenomena.
**Blocks:** nothing; informs H-1 (steering: which direction do we steer along?), H-13 (head attribution: per-direction or per-subspace?).

### H-672: Probe-accuracy capacity floor — 1.5B is below threshold for many behavioral signals while still above for correctness
**Priority:** MEDIUM
**Motivated by:** 2508.19505 + F-2
**Test:** Replicate Boxo et al.'s deception probe on cached Qwen-1.5B activations using their MMLU synthetic-argument pipeline (Appendix A.1). Compare layer-wise deception AUROC to our cached layer-wise correctness AUROC (P11). If deception probe stays at chance (~50%) at all layers for Qwen-1.5B while correctness peaks at 0.7731 at L19, we have a strict ordering: correctness < deception in capacity demand on this architecture. ~6 h H100 (argument generation) + ~1 h CPU (probing).
**Requires:** H100 for argument generation (model loaded), CPU for probing; new MMLU-derived synthetic dataset.
**Would change:** If Qwen-1.5B can host correctness probing but not deception probing, F-1's "universal across scales" claim restricts to *shape* of breathing, not to recoverability of behavioral content — the residual stream has structure but not capacity for deception-class abstractions at 1.5B. If both work, the capacity-threshold framing is wrong and Boxo et al.'s 1.5B-at-chance result is a methodology artifact (synthetic data quality, not residual-stream capacity).
**Blocks:** nothing; informs cross-scale findings (F-6) and H-11 (big-to-small distillation feasibility).

---


### H-673: LC+ prompt + DistilRoBERTa hedge mapper matches prefill-DoM AUROC on MATH-500
**Priority:** HIGH
**Motivated by:** 2509.24202 + F-2 (prefill L19 DoM AUROC 0.7731)
**Test:** Re-prompt 500 MATH-500 problems on Qwen-2.5-1.5B with LC+ prompt (Appendix C.3 of 2509.24202), apply released DistilRoBERTa mapper to convert hedge phrases to confidence scores, compute AUROC vs ground-truth correctness. Compare against F-2's 0.7731.
**Requires:** 1.5B model on H100 for re-prompted generation (~1hr), DistilRoBERTa mapper weights from anonymous repo, existing correctness labels for MATH-500.
**Would change:** If LC+ mapper-AUROC ≥0.77, F-2's framing as a *mechanistic* finding shrinks — the model's surface output already exposes the signal. F-2 becomes a calibration finding rather than a geometry finding. If LC+ AUROC ≤0.65, F-2's hidden-state signal genuinely exceeds what verbalization captures, strengthening the mechanistic claim.
**Blocks:** H-674 (no point distilling SU into SFT if simple LC+ prompting suffices).

### H-674: SU-distilled SFT beats prefill-DoM selective prediction on MATH-500 at matched coverage
**Priority:** HIGH
**Motivated by:** 2509.24202 (Table 4: LC SFT AUROC 0.7331 NQ-Open beats SU baseline 0.7252) + F-8 (selective acc 71.6% at coverage 0.5)
**Test:** K=10 sampling on MATH-500 with Qwen-2.5-1.5B → Farquhar-SU per question via DeBERTa entailment → discretize to 5 hedge levels → retrieve GPT-5-generated hedge sentences → LoRA SFT (r=32 α=32 dropout=0.05 3 epochs). Evaluate on held-out 100 problems: selective accuracy at coverage 0.5 vs F-8's 71.6%.
**Requires:** ~1 day H100, DeBERTa-v3 for entailment (small), GPT-5 API access for hedge sentence generation (~$50), held-out subset of MATH-500.
**Would change:** If SFT > 71.6% selective acc, the optimal F-8 pipeline becomes "amortize SU into training once" rather than "probe DoM at inference" — re-tiers F-8 as a feasibility result rather than a final answer. If SFT ≤ 71.6%, prefill-DoM probing remains the better tool; F-8 is robust.
**Blocks:** nothing.

### H-675: Prefill DoM AUROC tracks reasoning budget on Qwen-2.5-1.5B
**Priority:** MEDIUM
**Motivated by:** 2509.24202 (VNC AUROC moves +10pp with GPT-5 reasoning toggle) + F-2
**Test:** Extract prefill L19 activations on MATH-500 with `<think>` budgets {0, 256, 1024} tokens. Recompute DoM AUROC at each budget. Slope > 5pp would mark F-2 as budget-conditional.
**Requires:** 30min H100, existing extraction harness.
**Would change:** Confirm → F-2's "stable correctness signal" claim weakens; the prefill direction is a function of reasoning compute. Reject → strengthens F-2 as compute-invariant.
**Blocks:** nothing.

---


### H-676: Multi-target probe on prefill L19 outperforms single-DoM correctness probe
**Priority:** HIGH
**Motivated by:** 2509.24248 + F-9
**Test:** Fit a 3-output linear regression on cached prefill_L19 activations
predicting (a) K=1 correctness, (b) K=8 majority agreement, (c)
self-supervised minimum-sufficient-CoT-length. Report joint per-target AUROC
and compare to single-target prefill DoM. ~30 min CPU.
**Requires:** Cached `pathway11_h100/prefill_gated_compute/` NPZs; CPU only.
**Would change:** Confirm → F-9 needs a "redundant for correctness alone"
caveat; multi-axis probe replaces single DoM as headline. Reject → F-9
strengthens (single-axis collapse loses no information).
**Blocks:** H-677 (a label-free version of the same probe).

### H-677: Self-supervised minimum-sufficient-prefix probe matches supervised DoM at F-8 coverage
**Priority:** HIGH
**Motivated by:** 2509.24248 + F-8 + H-12
**Test:** Generate SpecExit-style self-supervised targets on cached MATH-500
K=1 generations (iterative paragraph truncation + completion match). Train
regression head on prefill L19 against this label. Compute risk-coverage
curve; report accuracy at coverage 0.5 vs supervised F-8 baseline (71.6%).
**Requires:** Cached K=1 generations + logits; ~2 h CPU for label generation,
20 min for probe fit.
**Would change:** Confirm → label-free probes operationalize H-12 without SEPs;
F-8 reframed as "selective prediction works, supervision optional." Reject →
self-supervised target carries less signal than ground-truth correctness;
H-12 stays parked.
**Blocks:** nothing (a precursor to a no-label H-1 steering variant).

---


### H-678: F-1 breathing is FFN-tail-first growth at inference time, not a reasoning-specific dimensionality change
**Priority:** HIGH
**Motivated by:** 2510.00537 + F-1 + F-5
**Test:** Decompose L19 residual into (attention-output, FFN-output, residual-carry) and compute PR on each at prefill, mid-CoT, final. If breathing curve is preserved in FFN-output-only and absent in attention-only, F-1 reduces to an FFN spectral phenomenon. Re-extract on H100 (cached NPZs hold summed residual only). ~4h H100 + 2h CPU. See P11-FE853.
**Requires:** H100 pod (per-component activation save), Qwen-2.5-1.5B, MATH-500.
**Would change:** On confirm, F-1 narrative shifts from "reasoning produces dimensional inflation" to "FFN tail-first growth modulates apparent residual-stream dimensionality across positions"; H-4 (Sharpness/EoS correspondence) becomes redundant with spectral-bias framing. On reject, F-1 is reaffirmed as a reasoning-specific signature.
**Blocks:** H-4 (reframes), H-22 (motivates re-running with FFN/attention decomposition rather than full residual).

### H-679: eDim and the (PR̃, eR̃, SC) triple are a more stable narrative substrate for F-1 than raw PR
**Priority:** MEDIUM
**Motivated by:** 2510.00537 + F-1
**Test:** Recompute the breathing trajectory on cached pathway11_h100 covariance matrices using eDim and the full triple at prefill / pos 30–70 / final. Report cross-model variability (Qwen-1.5B, Qwen-7B, Phi-3, Llama-3.2). If eDim variability across the four models is < PR variability while keeping the same monotone-then-collapse shape, eDim should replace PR in the F-1 narrative. ~1h CPU. See P11-FE852.
**Requires:** CPU only, cached covariance matrices.
**Would change:** Validate-claims invariant for F-1 expands to include eDim alongside PR; the headline "PR ~60–115 mid-CoT" gets re-stated with confidence intervals derived from a stable composite metric. Sharpens cross-model claims.
**Blocks:** Nothing.

---


### H-680: Difficulty steering (α=-3) outperforms supervised correctness DoM steering on Qwen2.5-Math-1.5B
**Priority:** HIGH
**Motivated by:** 2510.18147 + F-2, F-8
**Test:** Head-to-head MATH-500 K=1 Pass@1 with α∈{-3,-2,-1,0,+1,+2,+3} for (a) our prefill correctness DoM vs (b) Lugoloobi & Russell's AMC difficulty probe direction. Apply each at its probed (layer, pos). 1024-tok labels only. Estimate: 4h H100 + 2h CPU.
**Requires:** H100 pod, Qwen2.5-Math-1.5B, MATH-500 cached, Easy2HardBench AMC labels (HF download), our cached prefill NPZs.
**Would change:** If (b) beats (a) by >2pp K=1, H-1 reframes from "DoM steering" to "treat-as-easy steering" and our supervised correctness DoM is downgraded as the canonical steering direction. If (a) ≥ (b), F-2 / H-1 framing holds and we have a stronger external comparison point.
**Blocks:** nothing.

### H-681: F-5 breathing amplitude is dominated by static prefill-encoded difficulty
**Priority:** MEDIUM
**Motivated by:** 2510.18147 (ρ=0.88 static difficulty decoding from prefill) + F-5, F-1
**Test:** Per-problem regression of temporal PR amplitude (peak − floor) on AMC IRT difficulty (mapped via Easy2HardBench). 4h CPU on cached curves.
**Requires:** CPU, cached temporal PR curves from `pathway11_h100/prefill_gated_compute/results.json`, Easy2HardBench labels.
**Would change:** R² > 0.5 → F-5's "content-dependence" weakens to "difficulty-dependence" (a much weaker, less novel claim). R² < 0.2 → F-5 is robust to the difficulty-as-confounder critique and we should write that explicitly.
**Blocks:** nothing.

### H-682: F-8 selective-prediction accuracy is RL-fragile under GRPO post-training
**Priority:** MEDIUM
**Motivated by:** 2510.18147 (GRPO ablates LLM-derived probes by up to 50% in early/middle layers; β=-0.63 for LLM-difficulty / Pass@1 residualized correlation)
**Test:** Run Dr.GRPO on Qwen2.5-Math-1.5B (Verl, A100-80GB, batch 256, lr 1e-5, MATH-train ≥3 difficulty); checkpoint every step; re-fit L19 DoM at each checkpoint; re-evaluate F-8's coverage-0.5 accuracy at each step. 1d A100 + 10h CPU.
**Requires:** A100-80GB pod, Verl install, MATH-train preprocessing.
**Would change:** If F-8 degrades >5pp by peak-Pass@1 step, the project's deployable signal is RL-fragile and any production deployment needs to extract DoM *after* RL post-training, not from the base model. If stable, F-8 generalizes across post-training.
**Blocks:** H-680 (head-to-head steering should ideally be evaluated on both base and post-RL checkpoints).

---


### H-683: Prefill correctness signal at L19 is a rank-k manifold (k>1), not a single direction
**Priority:** HIGH
**Motivated by:** 2511.08379 + F-2
**Test:** Train a 4×4 SOM on cached L19 prefill activations from incorrect MATH-500 (Qwen-2.5-1.5B), compute correct-class centroid, derive 16 candidate directions, BO over k∈{2..5} via 5-fold OOF logistic-probe AUROC. Compare against 0.7731.
**Requires:** CPU only, ~1-2h, all cached in `pathway11_h100/prefill_gated_compute/`.
**Would change:** If multi-direction AUROC ≥ 0.80, F-2 reframes from "a single direction" to "a low-rank subspace" and selective-prediction (F-8) gains a free upgrade. If AUROC ≈ 0.7731, F-2's single-direction framing survives a non-trivial challenge — strengthens to STRONG.
**Blocks:** H-684 (subspace orthogonality) and the multi-direction version of H-1 (per-position multi-direction steering).

### H-684: Prefill and final-token L19 are non-orthogonal as subspaces, even though their DoMs are
**Priority:** MEDIUM
**Motivated by:** 2511.08379 + F-3
**Test:** Run SOM at both L19 prefill and L19 final-token, compute principal-angle CCA between direction sets, plus full pairwise cosine matrix.
**Requires:** CPU, ~1h, cached NPZs.
**Would change:** If max pairwise cos ≥ 0.4 or principal angle ≤ 30°, F-3 collapses to "two views of the same subspace". If still all ≤ 0.2, F-3 strengthens to "subspace-level orthogonality, not just direction-level".
**Blocks:** nothing (downstream of H-683).

---


### H-685: L19 is the AUROC peak inside a wider discriminative-layer band on the correctness contrast
**Priority:** HIGH
**Motivated by:** 2601.19375 + F-2
**Test:** Compute `μ̃correct(k) · μ̃incorrect(k)` for all 28 Qwen2.5-1.5B layers using cached Pathway 11 prefill activations. The discriminative set L_disc is {k : product < 0}. Compute prefill DoM AUROC (correct vs incorrect) per layer. Predict: L_disc is a contiguous band centred near L19 with |L_disc| ≥ 4, and L19 is the AUROC argmax inside the band. ~20 min CPU.
**Requires:** Cached `pathway11_h100/prefill_*qwen15b*.npz`, no new compute.
**Would change:** If confirmed, F-2's "L19 is canonical" framing softens to "L19 is the AUROC peak inside L_disc" — F-2 stays headline but gains nuance and a principled selection rule. If refuted (L_disc = {L19} alone), F-2's geometric privilege strengthens.
**Blocks:** P11-FE864 sweep (we want to know L_disc before deciding which layers to steer).

### H-686: Norm-preserving rotation at L19 produces controllable MATH-500 accuracy deltas without generation collapse, while naive activation-addition causes collapse
**Priority:** HIGH
**Motivated by:** 2601.19375 + H-1
**Test:** Implement both (a) the SS norm-preserving rotation R^P_θ at L19 on Qwen2.5-1.5B prefill, and (b) naive activation-addition at L19 with matched θ-equivalent coefficient. Sweep θ ∈ {0°, 30°, ..., 180°}. Generate K=1 MATH-500 answers. Predict: (a) gives smooth accuracy curves with PPL < 2× baseline at all angles; (b) hits PPL > 2× and gibberish output before accuracy moves measurably. 4h H100.
**Requires:** H100 generation pass, Qwen2.5-1.5B prefill hooks at L19.
**Would change:** Confirms or refutes H-1 *implementation*. If naive ActAdd collapses output before accuracy budges, then any past pilot we ran with ActAdd was mis-falsifying H-1; we should rerun with the norm-preserving variant before declaring H-1 dead. If norm-preserving rotation moves MATH-500 accuracy ≥ 5% absolute, H-1 is confirmed and SS's recipe becomes our default steering primitive.
**Blocks:** Nothing downstream — but supersedes the previous H-1 implementation plan.

### H-687: F-3's prefill/final DoM orthogonality is position-conditional, not a generic geometric phenomenon
**Priority:** MEDIUM
**Motivated by:** 2601.19375 + F-3
**Test:** Compute the 56×56 inter-layer cosine matrix of per-layer DoMs on Qwen2.5-1.5B for the correctness contrast, with rows/cols indexed by (position ∈ {prefill, final}, layer ∈ {0..27}). Predict: dense within-position blocks (mean off-diagonal > 0.6), sparse between-position blocks (mean ≤ 0.1). 30 min CPU.
**Requires:** Cached prefill + final-token activations for Qwen2.5-1.5B (already in P11 cache).
**Would change:** If confirmed, F-3 should be reworded from "prefill DoM and final-token DoM are orthogonal" to "prefill-position and final-token-position DoMs occupy near-orthogonal subspaces." This has implications for SS-style global-direction selection: it works *within* a position group but not *across* groups. If refuted (matrix is sparse everywhere or dense everywhere), F-3 generalises differently and we need to revisit its mechanism.
**Blocks:** Nothing.

---


### H-688: Prefill L19 DoM is template-stable but content-unstable
**Priority:** HIGH
**Motivated by:** 2604.24712 + F-2
**Test:** LV-paraphrase MATH-500 prompts (variable rename + verb broadening, keeping the instruction suffix and worked-example template fixed). Re-extract Qwen-2.5-1.5B L19 prefill activations. Re-fit 5-fold OOF DoM probe. Measure (a) AUROC on paraphrased prompts vs original 0.7731, (b) cosine similarity between original and paraphrased DoM directions. Estimated time: 2h H100 + 20min CPU.
**Requires:** H100 (re-extraction with new prompts), cached MATH-500 problem text, gpt-5-mini API for paraphrase generation.
**Would change:** If AUROC drops > 0.04 OR cos(orig DoM, paraphrased DoM) < 0.7, F-2's framing flips from "prefill encodes problem difficulty" to "prefill encodes prompt-surface features that correlate with difficulty under a fixed template." If both metrics are stable, F-2 is robustified.
**Blocks:** Decision on whether F-8 selective-prediction headline (71.6%) needs a paraphrase-stability disclaimer.

### H-689: D-bucket pathology is memorized-retrieval-cue pathology
**Priority:** HIGH
**Motivated by:** 2604.24712 + F-7
**Test:** Apply LV/US mutations to the 36 D-bucket problems and a matched-difficulty 36-problem A-bucket control on MATH-500. Re-sample K=8 on Qwen-2.5-1.5B. Count D→{A,B,C} and A→{B,C,D} bucket migrations. Manually classify each F→P (D→A or D→B) flip according to the Akli et al. Table 8 taxonomy (constraint-anchored wrong algorithm / I/O-format-primed parsing / bound-pushed buggy optimisation / no causal link). Estimated time: 3h H100 + 1h manual annotation.
**Requires:** H100 for re-sampling, LV mutation generator (gpt-5-mini batch), MATH-500 K=8 outputs from `pathway11_h100/prefill_gated_compute/`.
**Would change:** If D→{A,B} migration rate > 33% AND A→D rate < 11% AND > 50% of D→{A,B} flips classify cleanly into Akli's taxonomy, F-7 gets a mechanistic explanation (memorized-retrieval pathology) and predicts D-bucket vanishes for prompts that defeat retrieval. If migration rates are symmetric across buckets, F-7 is something else (sampling noise concentration, intrinsic-dimension floor).
**Blocks:** H-21 (D-bucket attention narrower than A-bucket — H-689 provides the causal upstream test that H-21 measures the consequence of).

### H-690: Familiarity-cue removal *improves* MATH-500 K=1 accuracy on a "named-trigger" subset
**Priority:** MEDIUM
**Motivated by:** 2604.24712 + H-5
**Test:** Manually annotate ~50 MATH-500 problems where the question contains a named-theorem cue (Pythagorean, Fermat's Little Theorem, AMC-style problem patterns). Apply LV neutralization replacing named cues with generic descriptions. Compare K=1 accuracy on Qwen-2.5-1.5B before vs after. Also extract the prefill DoM score per problem before/after. Estimated time: 2h H100 + 1h manual.
**Requires:** H100, manual annotation, MATH-500 source text.
**Would change:** If K=1 accuracy on the named-cue subset goes UP under neutralization while DoM score goes DOWN, H-5's "familiarity = good signal" framing is refuted in its current direction — DoM is reading a memorization-confidence signal that anticorrelates with correctness on this subset. Refines H-5 to "familiarity = good signal *when the memorized response is correct*; otherwise inverted."
**Blocks:** Whether H-5 should be split into "familiarity-correct" and "familiarity-trapped" sub-hypotheses before any work begins on it.

---


### H-691: RGTM in topology-sensitive zone matches Gaussian null on Qwen L19 prefill (consistent with F-10) OR exceeds it (refines F-10)
**Priority:** MEDIUM
**Motivated by:** 2309.11028 + F-10 + EXP-026
**Test:** Sweep (l, u) across 5 RGTM zones (TS/GS/LE/GE/I, 10 samples each) on cached Qwen2.5-1.5B L19 prefill NPZs. Train logistic regression on correctness with 5-fold OOF. Compare AUROC to F-2's 0.7731 (geometry baseline) AND to the rank-matched empirical-covariance Gaussian null from EXP-026. ~1h CPU.
**Requires:** CPU only, cached `pathway11_h100/prefill_gated_compute/` Stage-2 NPZs.
**Would change:** If TS-zone RGTM AUROC ≤ Gaussian null + 0.01 across all 10 samples, F-10 generalizes from PH to all topology-flavored summary statistics — *strong* statement worth highlighting in the headline. If TS-zone exceeds the null while PH did not, F-10 must be narrowed: PH features lose the signal; RGTM features retain it. Either way, F-10's reach gets clarified.
**Blocks:** H-7 framing (resolves whether topology has anything to add to L19 prefill prediction).

### H-692: D-bucket problems have elevated betweenness centrality on the RGTM graph
**Priority:** LOW
**Motivated by:** 2309.11028 (RGDM as graph-shortest-path) + F-7 + H-7
**Test:** Build the (l=0.40, u=0.65) RGTM weighted graph on 500-problem prefill activations. Compute betweenness centrality per node (problem). Compare distribution for D-bucket (36 problems, K=1-right-K=8-wrong) vs A-bucket (207, K=1-right-K=8-right) and B-bucket (K=1-wrong-K=8-right). 2-sided Mann-Whitney U test. ~30 min CPU.
**Requires:** CPU, cached prefill NPZs, bucket labels.
**Would change:** If D-bucket centrality > A-bucket + B-bucket median, then D-bucket sits on graph-bridges between confident-correct and confused-correct clusters — H-7's density-outlier framing is wrong, and we should pivot to graph-bridge detection (which has very different downstream tooling: betweenness, edge cuts). If no difference, H-7's density-outlier framing survives and Lin-Kriegeskorte's tooling is not the right import.
**Blocks:** nothing.

---


### H-693: Qwen2.5-1.5B residual stream at L19 sits in Liu-Liu-Gore "strong superposition" regime
**Priority:** HIGH
**Motivated by:** 2505.10465 + F-1, F-3, F-10
**Test:** Run P11-FE874 (ϕ_{1/2} on Qwen unembedding) and P11-FE875 (cos² histogram vs Beta(1/2, 1535/2) at L19). If both support strong-superposition, run P11-FE872 and P11-FE873 to ETF-correct F-3 and F-1/F-5. Total cost: ~2h CPU on cached data.
**Requires:** CPU only; cached `pathway11_h100/prefill_gated_compute/*.npz` and Qwen2.5-1.5B unembedding matrix.
**Would change:** On confirm, F-3's orthogonality reading becomes "geometric default, not circuit separation," F-10 reframes from null to ETF-convergence positive, F-1/F-5 need ETF-baseline subtraction in every figure. On reject, the four refutations above weaken and Liu et al. is downgraded to CITED ONLY.
**Blocks:** Re-running H-9 (breathing-amplitude vs difficulty) under both raw-PR and ETF-corrected PR — interpretation of H-9 depends on H-693 outcome.

### H-694: Cos(prefill_DoM, final_DoM) = 0.046 is statistically indistinguishable from a random-pair ETF null at m=1536
**Priority:** HIGH
**Motivated by:** 2505.10465 + F-3
**Test:** Compute Beta(1/2, 1535/2) z-score for cos² = 0.0021. Bootstrap with 1000 random label-shuffled probe directions on the same activations and check if 0.046 sits inside the 95% bootstrap CI of cos between two such "random" directions trained on the same stream.
**Requires:** CPU only, cached prefill activations, ~30min.
**Would change:** Confirm → F-3's "two distinct circuits" interpretation collapses; orthogonality is geometric default. Reject → F-3 stands and Liu et al.'s ETF prediction fails for our setting (which is itself a paper-worthy finding).
**Blocks:** H-13 (head-level circuit attribution) — only worth running if F-3's orthogonality survives the ETF null.

---


### H-695: Prefill-L19 DoM is collinear with bond-distribution divergence from a stable isomer reference
**Priority:** HIGH
**Motivated by:** 2601.06002 + F-2 + F-8
**Test:** Run FE20 (bond-edge labeler on cached P11 MATH-500 generations + per-problem π_C + KL divergence from R1-reference-π). Regress correctness against (a) prefill-DoM only, (b) bond-KL only, (c) both. Compare AUROC, ΔAUROC, and partial-R². ~4h CPU + $30 API.
**Requires:** Cached P11 H100 generations, LLM-as-judge labeler (Claude or Qwen-72B), no GPU.
**Would change:** *Confirm* (bond-KL ≈ prefill-DoM for AUROC, redundant features): demotes the "hidden-state correctness signal is novel" framing — F-2 becomes "geometric shadow of a textual statistic", and the goal-relevant takeaway is that for *small-model improvement* the cheaper bond-KL gate is preferable. *Reject* (prefill-DoM dominant, bond-KL adds little): hardens F-2 against the paper's competing claim and gives us a clean line in PAPER_INDEX explaining why a Long-CoT-topology paper is not the same probe as a residual-geometry probe.
**Blocks:** H-5 (prefill = familiarity vs decomposability) refinement; nothing else.

### H-696: Breathing peak coincides with metacognitive-oscillation onset in (entropy, Δ-entropy) phase space
**Priority:** HIGH
**Motivated by:** 2601.06002 + F-1 + F-5
**Test:** Run FE21 (Δ-entropy phase-space on cached P11 generations). For each problem, locate the token at which slope crosses 0.6 (paper's threshold) and align with the temporal-PR peak token. Mean offset, distribution. Also fraction-of-tokens-in-metacognitive-zone vs peak-PR amplitude per problem. 30min CPU.
**Requires:** Cached P11 H100 logprobs only.
**Would change:** *Confirm* (peaks within 5 tokens, correlation r > 0.5): unifies our breathing finding with the paper's information-flow finding under one mechanism — F-1 gets a label-free counterpart (cheaper alternative for H-12 SEP-style probes), and breathing universality (F-5) gains an information-theoretic explanation. *Reject* (no temporal alignment, low correlation): preserves breathing as a distinct geometric phenomenon orthogonal to entropy oscillation, and *strengthens* the case that hidden-state geometry encodes something the output distribution does not.
**Blocks:** H-12 (semantic-entropy probes) becomes more or less interesting depending on confirm/reject.

### H-697: Cached attention dumps from P11 show energy-level ordering Deep-Reasoning > Self-Reflection > Self-Exploration
**Priority:** MEDIUM
**Motivated by:** 2601.06002 (Boltzmann-attention reparameterization, Fig. 8)
**Test:** Re-run a 50-problem subset of P11 MATH-500 with `output_attentions=True` (or load existing dump if available). For each token-pair, classify edge type via heuristic (next-token = deep-reasoning, backward-look > N tokens to a semantically-similar earlier step = self-reflection, mid-range to a divergent step = self-exploration). Compute mean −q·k/√d per group. ~2h H100 if regenerating attentions.
**Requires:** H100 access OR cached attention from P11; bond-labeler (depends on FE20).
**Would change:** *Confirm* (paper's energy ordering replicates in our pipeline): gives us a third axis to project the prefill-DoM direction onto (covalent vs hydrogen vs van-der-Waals) and clean attribution candidate for H-13 (head-level circuit attribution). *Reject* (no energy ordering or different ordering on Qwen-1.5B/7B than on the paper's R1/OSS/QwQ models): bounds the bond-mechanism story to large reasoning models and reduces the threat to F-2/F-7.
**Blocks:** H-13 (head-level attribution) gets a cleaner target if confirmed.

---

## Synthesis-cascade follow-ups (2026-05-03)

Four explicit hypotheses derived from SYNTHESIS §4 ("What we still don't
know"). All four are within reach with cached caches; none has run.

### H-698: Cross-architecture replication of F-2 (prefill DoM AUROC on Phi-3-mini and Llama-3.2-1B)
**Priority:** HIGH
**Motivated by:** SYNTHESIS §4 Q2; F-2 pending_controls
**Test:** Run the F-2 OOF 5-fold prefill DoM probe on the cached Phi-3-mini
and Llama-3.2-1B 1024-tok L19 prefill activations in
`pathway11_h100/exp1_cross_model/`. Report AUROC + 95% CI per architecture
and the 0.05 effect-size band against Qwen-2.5-1.5B's 0.7731.
**Requires:** ~1h CPU per architecture; cached caches on disk; no new GPU.
**Would change:** *Confirm* (≥ 0.65 AUROC each, within 0.05 of Qwen's 0.7731):
F-2 graduates from one-model finding to cross-architecture pattern. Unblocks
APPLICATIONS App 1's non-Qwen production checklist. *Reject* (< 0.6 or large
gap): F-2 is Qwen-specific; APPLICATIONS Apps 1, 3, 4 lose their cross-arch
deployment pathway.
**Blocks:** APPLICATIONS App 1 production checklist; F-2 cross-arch pending
control.

### H-699: Penultimate-token PR-null test for F-1 / F-4 answer-vocabulary confound
**Priority:** HIGH
**Motivated by:** SYNTHESIS §4 Q5; PERSPECTIVES "strongest criticism"
**Test:** Compute participation ratio at the penultimate token (immediately
before `\boxed{}` answer) on cached MATH-500 1024-tok generations. Compare
PR distribution to (a) the final-token answer-position PR (F-4 collapse),
(b) random-token control. Three subgroups: large-answer (≥ 4 char),
small-answer (1 char), single-token answer. If penultimate PR shows the
F-1 breathing peak but final-token PR collapse is answer-vocabulary-driven,
F-4 is partially deflated.
**Requires:** ~1h CPU; cached generations on disk.
**Would change:** *Confirm* (penultimate PR ≈ peak, final PR explained by
answer-token unembed): F-4 weakens; the "correct trajectories collapse
harder" claim is partially answer-vocabulary. *Reject* (PR collapse pre-dates
the answer token): F-4 strengthens.
**Blocks:** F-4 strongest-criticism control; nothing else.

### H-700: PC9 is a length axis, not a difficulty axis
**Priority:** MEDIUM
**Motivated by:** SYNTHESIS §4 Q6; FE291 (PC9 single-feature AUROC 0.658
at variance share 2.5%)
**Test:** Project per-problem L19 prefill activations onto PC9. Spearman-
correlate the projection with (a) generation length in tokens
(`fe448-length-auroc` reference 0.7986), (b) MATH-500 official difficulty
level 1–5, (c) within-topic accuracy rate. PC9-as-length predicts |ρ| ≥ 0.6
on (a). PC9-as-difficulty predicts |ρ| ≥ 0.4 on (b) but weak on (a).
**Requires:** 30 min CPU; cached caches.
**Would change:** *PC9 is length:* the decomposition triangle's directional
ceiling collapses to "DoM + length feature," and the cov-spectrum lift
(0.7928) becomes the only F-2 result not explained by length confounding.
*PC9 is difficulty:* F-2 factors into "competence direction (PC1) + abstract
difficulty axis (PC9) + per-problem second-order shape," strengthening the
mechanistic story.
**Blocks:** F-2 mechanism; PERSPECTIVES "decomposition triangle" framing.

### H-701: Rank-truncate-cov-spectrum causal companion test
**Priority:** CRITICAL
**Motivated by:** SYNTHESIS §4 Q1 (load-bearing causal test for F-2)
**Test:** For each MATH-500 problem, truncate the L19 prefill covariance to
its top-r eigenvectors after PC1 residualization (sweep r ∈ {1, 5, 10, 20,
all}). Reconstruct activations, run continuation, measure correctness.
This is distinct from FE214 (which noises raw activations) — it surgically
removes the second-order *spectral* information that the cov-spectrum
probe (FE881, 0.7928) reads. If correctness drops monotonically with
truncation, the cov-spectrum is causally implicated, not merely correlational.
**Requires:** 1 H100-day; cached caches + custom forward hook at L19; not
yet a `:FutureExperiment` node — file alongside FE214/FE269/FE283.
**Would change:** *Confirm* (correctness drops with r): the cov-spectrum
0.7928 lift is causally load-bearing; APPLICATIONS App 3 ("cov-spectrum
probe") graduates from filter to mechanism. *Reject* (no drop): cov-spectrum
is a passive geometric correlate; F-2 remains directional + a per-problem
second-order shape readout that the model doesn't itself use.
**Blocks:** F-2 causal upgrade; APPLICATIONS App 3 production checklist.

---


### H-702: DoM AUROC is confounded by decoding-strategy-dependent correctness labels
**Priority:** MEDIUM
**Motivated by:** 2602.10346 + F-2
**Test:** Regenerate MATH-500 under Top-W decoding (λ=2.2, β=2.8, T=1.0) on Qwen-2.5-1.5B; recompute DoM AUROC under new labels. Also compute correlation between DoM score and per-problem accuracy gain from greedy→Top-W.
**Requires:** H100 (~4h) for regeneration; CPU for analysis
**Would change:** If confirmed (AUROC drops >0.03 under Top-W), F-2's interpretation shifts from "DoM predicts model knowledge" to "DoM predicts decode-tractability." Selective prediction (F-8) would still work operationally but the mechanism interpretation weakens.
**Blocks:** nothing

### H-703: Diagonal whitening of L19 activations improves DoM separability
**Priority:** HIGH
**Motivated by:** 2602.10346 + F-2
**Test:** Apply per-coordinate whitening (subtract mean, scale by inverse std) to cached L19 prefill activations. Recompute DoM (top singular vector of whitened correct-incorrect difference) and AUROC. Compare to 0.7731.
**Requires:** CPU only, 10min, cached NPZs
**Would change:** If confirmed (AUROC > 0.78), whitening becomes standard preprocessing. If rejected (AUROC drops), confirms that anisotropy is informative for our task.
**Blocks:** nothing

---


### H-704: L19 DoM direction is recoverable from weight-space Procrustes rotation axes alone

**Priority:** HIGH
**Motivated by:** 2602.05943 + F-2
**Test:** Procrustes decompose the L19 attention-out and MLP-down weight matrices (instruct vs base). Extract top-k eigenvectors of skew-symmetric Q. Project cached L19 prefill activations onto these axes and compute 5-fold AUROC. Compare with supervised DoM AUROC 0.7731.
**Requires:** CPU only. Qwen-2.5-1.5B base + instruct weights (HF download). Cached activations from pathway11.
**Would change:** If AUROC ≥ 0.75, F-2 is reframed: correctness signal is weight-geometric, not content-computed. Steering (H-1) would need to target rotation axes, not activation directions.
**Blocks:** H-1

### H-705: Per-layer orthogonality ratio ||ρ||/||R·W₀|| anti-correlates with per-layer DoM AUROC

**Priority:** MEDIUM
**Motivated by:** 2602.05943 + F-2
**Test:** Procrustes decompose each layer's MLP and attention weights. Correlate ||ρ||/||R·W₀|| with per-layer DoM AUROC from pathway11 layer sweep.
**Requires:** CPU only. Same weights as H-704.
**Would change:** If anti-correlation holds (more orthogonal = higher AUROC), this explains why L19 is special — its weight matrix is the most rotation-dominated. If no correlation, weight orthogonality is unrelated to correctness signal location.
**Blocks:** nothing

### H-706: F-3 prefill/final DoM orthogonality is mechanically explained by weight-space rotation at L19

**Priority:** HIGH
**Motivated by:** 2602.05943 + F-3
**Test:** Extract R from Procrustes at L19. Compute R · DoM_prefill and measure cos(R · DoM_prefill, DoM_final). If cos >> 0.046, a single weight rotation bridges the two directions. If cos ≈ 0.046, the orthogonality is not explained by weight rotation.
**Requires:** CPU only. Cached DoM directions from pathway11.
**Would change:** Confirms F-3 is structural if cos stays near zero (rotation doesn't bridge). Refutes F-3's "structural" label if cos is high (rotation mechanically explains it).
**Blocks:** nothing

---


### H-707: Prefill DoM direction at L19 rotates under MaxRL training vs GRPO training
**Priority:** HIGH
**Motivated by:** 2602.02710 §5 (weight function analysis) + F-2 (AUROC 0.7731 on GRPO-trained Qwen2.5-1.5B-Instruct)
**Test:** Train Qwen2.5-1.5B with MaxRL (single-line change to advantage normalization in verl), or use released Qwen3-1.7B-MaxRL checkpoint. Extract L19 prefill activations on MATH-500 at 1024 tok. Fit DoM direction. Report cos(DoM_GRPO, DoM_MaxRL). 1-2 days H100.
**Requires:** H100, MATH-train for RL training (or released checkpoint), extraction pipeline from P11
**Would change:** On confirm (cos < 0.5): F-2 and F-8 are partly GRPO artifacts; on reject (cos > 0.8): DoM is robust to RL objective, strengthening F-2 and effectively closing H-282.
**Blocks:** H-282

### H-708: D-bucket geometric signature (F-7) is caused by GRPO gradient void at low pass rates
**Priority:** HIGH
**Motivated by:** 2602.02710 Figure 6 (GRPO gradient norm → 0 at low pass rate, on our exact model+data) + F-7
**Test:** Compute GRPO w(p) for each MATH-500 problem using cached per-problem accuracy estimates. Check whether D-bucket problems cluster in the w(p) → 0 region. Cheap CPU analysis (FE110701) gives indirect evidence; direct confirmation requires MaxRL checkpoint (FE110704). 20min CPU + optional 2h H100.
**Requires:** Cached per-problem accuracy from K-sample generation, cached L19 activations
**Would change:** On confirm: F-7 is reframed as "GRPO gradient void signature" not "competence boundary signature." On reject: F-7 is robust and reflects genuine competence structure.
**Blocks:** nothing

---


### H-709: Correct/incorrect L19 prefill activations are separated by a one-parameter Lie subgroup action classifiable as elliptic, hyperbolic, or parabolic
**Priority:** HIGH
**Motivated by:** 2509.22219 + F-2 + F-3
**Test:** Decompose the inter-class covariance of cached L19 prefill activations into skew-symmetric, symmetric-traceless, and nilpotent components. Fit Hγ-Net–style invariant representations under each regime assumption. Compare classification AUROC to DoM 0.7731 and cov-spectrum 0.7928.
**Requires:** CPU, cached NPZ activations from pathway11_h100
**Would change:** On confirm (one regime clearly dominant + AUROC lift): F-2 reframed from "single direction" to "single one-parameter subgroup orbit," opening invariant-representation–based probes as the natural feature space. On reject (no clean subgroup, invariant rep ≤ 0.7731): correct/incorrect separation is genuinely linear, strengthening F-2's "single direction" framing.
**Blocks:** nothing

### H-710: The prefill→final-token DoM transformation is best described by a specific Lie-algebraic regime (elliptic/hyperbolic/parabolic), and the regime type determines whether H-17's RoPE-mechanical hypothesis holds
**Priority:** MEDIUM
**Motivated by:** 2509.22219 + F-3 + H-17
**Test:** Using cached prefill and final-token L19 activations, fit paired transformation models under each regime. If elliptic regime dominates and recovered rotation frequency matches RoPE frequency at L19's position, H-17 is supported. If hyperbolic or parabolic regime fits better, H-17 is refuted.
**Requires:** CPU, cached paired (prefill, final-token) activations from pathway11_h100
**Would change:** On confirm (elliptic + RoPE-matched): H-17 becomes CONFIRMED, F-3 gains precise geometric characterization. On reject (non-elliptic): H-17 is refuted, F-3's "structural" orthogonality gets a new mechanistic explanation.
**Blocks:** H-17

---


### H-711: Per-layer WeightWatcher alpha profile of Qwen-2.5-1.5B peaks (minimizes toward 2) at or near L19
**Priority:** HIGH
**Motivated by:** 2507.17912 + F-2 + H-489
**Test:** Run `weightwatcher` on all 28 transformer block weight matrices; plot alpha vs layer index; test whether the alpha minimum falls at L19 ± 2 layers.
**Requires:** CPU only, Qwen-2.5-1.5B weights from HuggingFace (~3GB download)
**Would change:** Confirms → L19 specialness has a weight-space explanation (SETOL predicts its correctness-prediction peak). Rejects → L19 is NOT spectrally special, correctness signal is representational, not "layer quality."
**Blocks:** nothing

### H-712: L19 DoM direction (cos=0.9216 with PC1) lies predominantly within the SETOL Effective Correlation Space of L19's weight matrix
**Priority:** HIGH
**Motivated by:** 2507.17912 + F-2 + F-10
**Test:** Compute ECS of L19 (PL-tail eigenvectors of W_L19^T W_L19). Project our DoM direction onto ECS basis. Measure fraction of DoM variance explained by ECS directions.
**Requires:** CPU only, Qwen-2.5-1.5B weights + cached DoM direction from P11
**Would change:** Confirms → our correctness signal is in the "generalizing subspace" per SETOL, supporting its theoretical interpretation. Rejects → our signal exploits "bulk" directions SETOL predicts should be noise, challenging HTSR completeness and suggesting hidden-state correctness prediction accesses information NOT captured by weight-space spectral theory.
**Blocks:** nothing

---


### H-713: mCCA between prefill and final-token L19 activations is high (≥0.9), implying F-3 orthogonality is within-equivalence-class
**Priority:** HIGH
**Motivated by:** 2602.15438 + F-3
**Test:** Compute mCCA(L19_prefill_500×1536, L19_final_500×1536) using canonical correlation analysis on cached NPZ activations. Compare against cos(DoM_prefill, DoM_final) = 0.046.
**Requires:** CPU, cached pathway11_h100 NPZs
**Would change:** If confirmed (mCCA ≥ 0.9): F-3 needs reinterpretation — prefill/final are linearly equivalent, the cos=0.046 orthogonality is a 1D projection artefact. If rejected (mCCA ≪ 0.9): F-3 strengthened — genuine structural separation, not just a rotated basis.
**Blocks:** nothing

### H-714: Logit distance between 1.5B and 7B on MATH-500 is small enough to guarantee high mCCA, providing theoretical backing for cross-scale DoM transfer
**Priority:** MEDIUM
**Motivated by:** 2602.15438 + F-1
**Test:** Forward-pass both models on 500 MATH-500 prefill prompts, compute d_logit (Def. 3.1), apply Thm. 3.4 bound using μ_m from logit covariance. Need both models' full vocab logits.
**Requires:** H100 (two forward passes), Qwen-2.5-{1.5B, 7B}
**Would change:** If confirmed (bound implies mCCA > 0.8): strong theoretical support for F-1 universality. If rejected (bound vacuous due to large d_logit): models are too distributionally different for this theory to apply cross-scale.
**Blocks:** H-713

---


### H-715: Hodge Laplacian spectral gap on VR complexes built from L19 prefill activations separates correct from incorrect groups (refining F-10)
**Priority:** LOW
**Motivated by:** 2604.27241 + F-10
**Test:** Build VR complexes from L19 prefill activations (correct group, incorrect group, matched Gaussian null). Compute the combinatorial Hodge Laplacian L_1 = B_1^T B_1 + B_2 B_2^T. Extract the spectrum. Compare the spectral gap λ_{min+1}(L_1) between groups via permutation test. Est. time: 90min CPU.
**Requires:** CPU, cached L19 prefill NPZs (500 × 1536), scipy/numpy for sparse eigendecomposition
**Would change:** If spectral gaps differ between groups: F-10's PH null is summary-statistic-specific, not topology-universal → H-35 confirmed. If spectral gaps are also null: F-10 is strengthened to cover Hodge spectral invariants.
**Blocks:** nothing

---


### H-716: Intrinsic dimension of L19 prefill activations is ≪ 1536 and explains the linear probe ceiling
**Priority:** HIGH
**Motivated by:** 2311.03757 + F-2 + F-10
**Test:** Run MLE intrinsic dimension estimator (Levina-Bickel) and correlation dimension on cached 500×1536 L19 prefill activations. If d_est < 20, manifold structure exists; if d_est ≈ 1–5, the linear probe ceiling (0.7731) is explained by the low-d manifold being nearly 1-d along the correctness axis.
**Requires:** CPU, cached NPZs
**Would change:** On confirm (d << 1536): F-10 reframed as "PH null despite manifold structure"; F-2 ceiling explained by manifold geometry. On reject (d ≈ 1536): manifold assumption is wrong; activations fill the space.
**Blocks:** H-683

### H-717: Nonlinear (Diffusion Maps) embedding of L19 activations achieves AUROC > 0.7731
**Priority:** HIGH
**Motivated by:** 2311.03757 + F-2 + F-9
**Test:** Compute Diffusion Maps embedding (m=2–10) with renormalized Laplacian on cached L19 prefill activations. Train 5-fold logistic regression on DM coordinates. Compare to F-2's 0.7731.
**Requires:** CPU, cached NPZs, scikit-learn or megaman
**Would change:** On confirm: F-2's "single linear direction" is incomplete; nonlinear structure carries extra signal, and F-9's CoE redundancy may reflect CoE partially capturing this. On reject: linear probe is near-optimal; manifold curvature does not help.
**Blocks:** nothing

---


### H-718: NAG-style sparse neuron decomposition at L19 FFN UP reveals correctness-discriminative functional backbone
**Priority:** HIGH
**Motivated by:** 2604.15706 + F-2 + H-64
**Test:** Compute neuron impact scores for L19 FFN UP projection across 500 MATH-500 problems using cached activations + model weights. Identify top-K neurons per problem, compute High-Δ neurons between correct and incorrect groups. Test overlap with top components of supervised DoM direction.
**Requires:** CPU, cached L19 activations, Qwen-2.5-1.5B L19 FFN weights (load from HuggingFace)
**Would change:** If High-Δ neurons align with DoM's top components: strengthens F-2's mechanistic interpretation (DoM direction ≈ sparse neuron backbone). If they diverge: identifies orthogonal signal the current probe misses, motivates a neuron-importance-based probe that may beat 0.7731 AUROC.
**Blocks:** nothing

### H-719: NAG Dice similarity in neuron-index space separates correct/incorrect MATH-500 groups where Euclidean distance in R^1536 does not
**Priority:** MEDIUM
**Motivated by:** 2604.15706 + F-10
**Test:** Construct NAG representations (top-K neuron indices per problem at L19), compute pairwise NAG Dice similarity, run permutation test for correct/incorrect group separation. Compare effect size to Euclidean-distance-based separation (which yielded F-10's PH null).
**Requires:** CPU, cached L19 activations, Qwen-2.5-1.5B weights
**Would change:** If NAG similarity separates groups: shows the correctness signal lives in sparse combinatorial neuron patterns, not in the continuous geometry that PH tested. Would reframe F-10 as geometry-null rather than signal-null.
**Blocks:** nothing

---


### H-720: L19 DoM direction is recoverable from MATH-500 completions via next-token loss alone (cos(v_r, DoM) > 0.5)
**Priority:** HIGH
**Motivated by:** 2604.25783 + F-2 (AUROC 0.7731)
**Test:** Apply paper's vector recovery protocol (Eq. 5) on Qwen-2.5-1.5B MATH-500 completions with soft-gated layer window, no correctness labels. Measure cos(v_r, DoM_L19).
**Requires:** H100, Qwen-2.5-1.5B, MATH-500 completions (cached or regenerated), ~3h
**Would change:** Confirm → DoM is self-supervisedly recoverable, strengthening the signal's robustness beyond labeled supervision; Reject → DoM requires explicit correctness labels to find, it's not encoded in generation patterns.
**Blocks:** nothing

### H-721: DoM verbalization reveals "mathematical rigor" or "problem difficulty" as the dominant semantic content, not "confidence"
**Priority:** HIGH
**Motivated by:** 2604.25783 + F-2 + F-5
**Test:** Alpha-sweep DoM injection on 20 neutral prompts (paper's protocol), LLM summarization. Does the model start producing math-related or confidence-related text?
**Requires:** H100, Qwen-2.5-1.5B, ~1h generation + API call for summarization
**Would change:** Confirm → DoM is a difficulty/rigor estimator, reframing F-2 from "correctness prediction" to "difficulty assessment"; Reject (confidence-related) → DoM is closer to a calibration signal; Reject (incoherent) → DoM may not have clean semantic content at all.
**Blocks:** H-5

### H-722: Correctness-signal alignment score s(ℓ) has a single sharp peak at L19 (width ≤ 4 layers), consistent with layer-localized training imprint
**Priority:** MEDIUM
**Motivated by:** 2604.25783 Figure 3 + F-2
**Test:** Compute per-layer s(ℓ) = cos(mean_correct - mean_incorrect, DoM_L19) across all 28 layers of Qwen-2.5-1.5B. Measure peak width at half-maximum.
**Requires:** CPU, per-layer activation caches from P8 or new extraction (~2h H100)
**Would change:** Confirm (sharp peak) → L19 specificity is real, possibly training-imprinted; Reject (broad peak) → correctness signal is a distributed computation, not layer-specific.
**Blocks:** nothing

---


### H-723: Graph-structural features of L19 activation k-NN neighborhoods predict correctness at AUROC > 0.7731
**Priority:** MEDIUM
**Motivated by:** 2312.04762 + F-2
**Test:** Build k-NN graph (k=5,10,20) from cached 500×1536 L19 prefill activations, compute per-node effective resistance, spectral gap contribution, and Forman curvature. Train 5-fold logistic regression for correct/incorrect. Compare AUROC to DoM baseline 0.7731.
**Requires:** CPU, cached NPZ activations, scipy + networkx
**Would change:** If confirmed: F-2 needs revision — correctness signal has a relational/graph-structural component beyond single-direction projection. If rejected: strengthens the "single direction suffices" claim.
**Blocks:** nothing

### H-724: PH on kTree-sparsified activation graphs shows non-null correct/incorrect separation (refining F-10)
**Priority:** LOW
**Motivated by:** 2312.04762 + F-10
**Test:** Build k-NN graph from cached L19 activations, apply kTree sparsification to avg degree 5, compute H_0/H_1 barcodes on clique complex for correct vs incorrect vs Gaussian null. Compare bottleneck/Wasserstein distances between groups.
**Requires:** CPU, cached NPZs, giotto-tda or ripser, networkx
**Would change:** If confirmed: F-10's null result is an artifact of dense VR complexes, not absence of topology. PH could be rehabilitated as a correctness signal. If rejected: further strengthens F-10 (topology is null even at optimal sparsity).
**Blocks:** nothing

---


### H-725: Prefill DoM direction is primarily detecting memorization proximity (low conditional entropy basin)
**Priority:** HIGH
**Motivated by:** 2604.26841 + F-2
**Test:** Compute per-problem logit-lens conditional entropy at L19 (unembedding applied to cached residuals), compute AUROC for correctness, compare to DoM AUROC 0.7731. Then partial-correlation: DoM→correctness controlling for entropy. If partial r ≈ 0, DoM is explained by memorization.
**Requires:** CPU, cached L19 activations (500×1536), Qwen unembedding matrix
**Would change:** If confirmed: F-2 is reframed from "correctness geometry" to "memorization proximity detection." If rejected (DoM retains signal after controlling for entropy): DoM captures something beyond memorization.
**Blocks:** H-5

### H-726: D-bucket collective signature (F-7) is explained by problems sitting in memorization regime
**Priority:** MEDIUM
**Motivated by:** 2604.26841 + F-7
**Test:** Stratify MATH-500 problems by D-bucket membership, compare conditional entropy distributions. If D-bucket problems show systematically lower entropy (sharper basins → memorization regime), the geometric signature is an artifact of memorization clustering.
**Requires:** CPU, cached L19 activations, problem-level D-bucket labels
**Would change:** If confirmed: F-7 is downgraded from "difficulty-specific geometry" to "memorization artifact." If rejected: F-7's collective signature is independent of memorization regime.
**Blocks:** nothing

---


### H-727: The AUROC ceiling (0.7928) for correctness prediction from L19 activations equals the signal-channel rank at layer 19
**Priority:** HIGH
**Motivated by:** 2605.01172 + F-2
**Test:** Compute Gram matrix eigenspectrum of L19 prefill activations (500×500 matrix); identify signal-channel rank via MP edge or elbow method; compare to effective dimensionality of best probe (20 features → 0.7928). If they match, the ceiling is a representational property.
**Requires:** CPU only, cached L19 activations
**Would change:** On confirm: F-2's ceiling becomes theoretically explained, shifts focus from probe engineering to representation engineering. On reject: signal-channel theory doesn't apply at single-layer level.
**Blocks:** nothing

### H-728: Leave-one-out self-influence computed from L19 activations achieves AUROC ≥ 0.75 for correctness prediction without supervised labels
**Priority:** HIGH
**Motivated by:** 2605.01172 Section 6 + H-12 + F-8
**Test:** For each of 500 problems, compute LOO influence (cosine distance to leave-one-out centroid, or Mahalanobis distance to LOO distribution). Compute AUROC for correct/incorrect classification. Target: match or exceed DoM (0.7731) without any labels.
**Requires:** CPU only, cached L19 activations
**Would change:** On confirm: label-free selective prediction is possible, resolving H-12. On reject: supervised probing remains necessary for deployment.
**Blocks:** H-12

---


### H-729: L19 prefill activation manifold has negligible curvature (geodesic ≈ Euclidean)
**Priority:** MEDIUM
**Motivated by:** 2605.02167 + F-2 + F-10
**Test:** Compute ISOMAP geodesic distance matrix on cached 500×1536 L19 prefill activations at k=10,15,20 nearest neighbors. Correlate (Spearman) with Euclidean distance matrix. Also compute intrinsic dimensionality via MLE estimator.
**Requires:** CPU only, cached NPZs, scikit-learn ISOMAP
**Would change:** If confirmed (r > 0.95): validates all Euclidean analyses (DoM, PCA, VR-PH) and closes the "manifold curvature" concern. If rejected (r < 0.85): motivates manifold-aware re-analysis of F-2, F-10, and potentially explains part of F-10's PH null.
**Blocks:** nothing

---


### H-730: L19 DoM AUROC partially depends on position-content cross-term coupling, not pure content signal
**Priority:** HIGH
**Motivated by:** 2006.15595 + F-2 + H-101
**Test:** P11-FE925 (length correlation, 15min CPU) then P11-FE926 (RoPE projection, 30min CPU) then P11-FE928 (attention decomposition, 4h H100 — only if Tier 1 shows signal)
**Requires:** Cached DoM scores + Qwen-2.5 config (Tier 1); H100 forward passes (Tier 2)
**Would change:** Confirm (|ρ_length| > 0.2 or RoPE-plane fraction > 20%) → F-2's headline AUROC 0.7731 needs a "position-residualized" companion number; ceiling analysis changes. Reject → F-2's content-signal interpretation strengthens; H-101 partially confirmed.
**Blocks:** H-101

### H-731: F-5's content-dependent breathing has a positional pathway component mediated by content→attention→position coupling
**Priority:** MEDIUM
**Motivated by:** 2006.15595 (Eq. 6 cross-terms) + F-5
**Test:** P11-FE927 (length-matched breathing comparison, 20min CPU)
**Requires:** Cached breathing amplitudes + prefill token counts
**Would change:** Confirm (breathing amplitude differences attenuate >30% after length-matching) → F-5 needs a "partially position-mediated" caveat. Reject (differences survive within 10%) → F-5's pure content interpretation strengthens.
**Blocks:** nothing

---


### H-732: Fisher divergence between correct/incorrect activations exceeds Gaussian-predicted value
**Priority:** HIGH
**Motivated by:** 2407.11094 + F-10
**Test:** Estimate Fisher divergence between correct/incorrect L19 prefill activations using (a) Gaussian assumption with shared covariance and (b) non-parametric score matching (kernel or neural). If (b) >> (a), non-Gaussian structure exists that PH missed. Run on cached 500×1536 NPZs.
**Requires:** CPU only, cached NPZ data, score-matching library (e.g., sliced score matching)
**Would change:** Confirm: F-10 is PH-specific, not geometry-general — motivates exploring non-PH non-parametric probes. Reject: F-10 extends to Fisher divergence — Gaussian really does suffice for this data.
**Blocks:** nothing

### H-733: L19 is a robustly detected change-point in the layer-wise correctness signal
**Priority:** HIGH
**Motivated by:** 2407.11094 + F-2 + F-1
**Test:** Apply CUSUM/RSCUSUM to the 28-point per-layer DoM AUROC (or Fisher divergence) sequence. Test whether L19 is the unique robustly detected change-point with controlled false-alarm rate.
**Requires:** CPU only, per-layer DoM AUROCs from P8 layer-wise analysis
**Would change:** Confirm: L19 is a robust phase boundary, not just an AUROC peak — strengthens F-2 and provides a statistical test for layer importance. Reject: change-point is elsewhere or no single change-point exists — F-2's framing needs revision toward a multi-layer or gradual-transition picture.
**Blocks:** nothing

---


### H-734: The L19 DoM direction aligns with the top Hessian eigenvector of the training loss at the converged checkpoint
**Priority:** MEDIUM
**Motivated by:** 2604.21016 + F-2
**Test:** Compute λ\_max(∇²L(θ)) and its eigenvector u at the Qwen-2.5-1.5B converged weights (on the MATH-500 training loss surface). Compare cos(u, DoM) and cos(u, PC1). Est: 4h H100.
**Requires:** H100, Qwen-2.5-1.5B weights, MATH-500 data, backward pass infrastructure
**Would change:** Confirm (cos > 0.8) → F-2's DoM is a fingerprint of training-time EoS dynamics, opening a mechanistic bridge between loss-landscape curvature and inference-time correctness signals. Reject (cos < 0.3) → DoM is not the top curvature direction; its origin must be sought elsewhere (possibly in the forward-pass computation, not the training trajectory).
**Blocks:** H-4

### H-735: Qwen-2.5-1.5B does not operate at Edge of Stability (sharpness far below 2/η due to CE loss)
**Priority:** HIGH
**Motivated by:** 2604.21016 + H-4
**Test:** Compute S(θ) = λ\_max(∇²L(θ)) at the converged weights and compare to 2/η (where η is Qwen's training learning rate). Check if the model is in the EoS regime. Est: 2h H100.
**Requires:** H100, Qwen-2.5-1.5B weights, training hyperparameters (learning rate, batch size)
**Would change:** Confirm (S ≪ 2/η) → H-4 should be deprioritized or reformulated; breathing is not an EoS phenomenon. Reject (S ≈ 2/η) → EoS framework applies, H-4 proceeds.
**Blocks:** H-4

---


### H-736: Correctness-confidence at L19 is a multi-dimensional manifold, not a single linear direction
**Priority:** HIGH
**Motivated by:** 2604.28119 + F-2
**Test:** Restricted-R² subspace capture (P11-FE934): measure how many PCA directions are needed to reconstruct correctness-class centroids in cached L19 prefill activations. If R² at 1 direction < 0.8 but R² at 5 directions > 0.95, correctness is manifold-structured.
**Requires:** CPU only, cached NPZs
**Would change:** On confirm: F-2's "single direction" framing becomes "single direction is best linear tile of a multi-dimensional correctness manifold" — the AUROC ceiling is set by tile coverage, not information limit. On reject: F-2 stands as stated, correctness is genuinely 1D at L19.
**Blocks:** H-26

### H-737: F-10's PH-null is an artifact of testing the superposed cloud rather than factor-projected sub-clouds
**Priority:** HIGH
**Motivated by:** 2604.28119 + F-10 + H-35
**Test:** Factor-manifold PH retest (P11-FE937): project L19 activations onto top-k PCA subspaces, recompute VR-PH on projected correct/incorrect sub-clouds. If non-trivial Betti numbers emerge in sub-clouds that were Gaussian-null in the full cloud, F-10's null was premature.
**Requires:** CPU only, cached NPZs + ripser/giotto-tda
**Would change:** On confirm: F-10 gets a caveat ("PH on the full superposed cloud is null; per-factor PH is non-trivial"), reopening topology as a signal source. On reject: F-10 is robust even under factor projection, strengthening the Gaussian-null claim.
**Blocks:** H-23, H-24, H-25, H-35

### H-738: Ising-community structure in L19 PCA components reveals correctness-aligned manifold groups
**Priority:** MEDIUM
**Motivated by:** 2604.28119 + F-2 + F-7
**Test:** Ising-coupling community detection (P11-FE935): binarize PC projections, fit Ising model, Louvain cluster. Check community alignment with correctness labels (A/B/C/D buckets) via adjusted Rand index.
**Requires:** CPU only, cached NPZs
**Would change:** On confirm: provides a principled grouping of L19 features by manifold membership, potentially improving D-bucket detection (F-7) and explaining why certain PCs matter more than others. On reject: L19 PCA components lack the co-activation structure needed for Ising discovery — SAE-level analysis (P11-FE938) may be needed instead.
**Blocks:** nothing

---


### H-739: Dimensional breathing is AdamW-specific, not architecture-universal

**Priority:** HIGH
**Motivated by:** 2605.04418 + F-1
**Test:** Train matched models with MACRO-spec vs AdamW, extract L19 prefill activations on MATH-500, measure breathing amplitude (PC1 variance ratio across sequence positions). If MACRO-trained model shows no breathing, F-1's universality claim is restricted to AdamW-family optimizers.
**Requires:** H100 training run (120M–330M scale), MACRO reimplementation
**Would change:** On confirm: F-1 is downgraded from "universal" to "AdamW-family-specific"; on reject (breathing persists under MACRO): F-1 is strengthened, breathing is architecture-intrinsic not optimizer-dependent.
**Blocks:** H-546, H-547

### H-740: DoM direction tracks the leading right singular vector v₁ of W_L19

**Priority:** HIGH
**Motivated by:** 2605.04418 + F-2 + EXP-FE291
**Test:** Extract W_L19 (attention output or value projection) from Qwen-2.5-1.5B checkpoint, compute SVD, measure cos(DoM, v₁). Compare to cos(DoM, PC1) = 0.9216. If cos(DoM, v₁) > 0.90, the correctness signal is primarily a weight-matrix spectral property.
**Requires:** CPU only, HuggingFace checkpoint, cached DoM vector from P11
**Would change:** On confirm: F-2 mechanism shifts from "activation-space covariance" to "weight-space spectral structure"; on reject: weight geometry and activation geometry are decoupled, DoM is emergent from data covariance not weight spectrum.
**Blocks:** nothing

---


### H-741: CITE anytime-valid mode certification on MATH-500 produces a selective-prediction curve that matches or exceeds F-8's 71.6% at coverage 0.5
**Priority:** HIGH
**Motivated by:** 2605.05873 + F-8 + F-2
**Test:** Generate K=16 samples per MATH-500 problem on Qwen-2.5-1.5B, run CITE Algorithm 1 at epsilon=0.05, define abstention set as non-certified problems at matched coverage, compare answered accuracy. ~2.5h total.
**Requires:** H100 GPU (2h for K=16 generation), CPU (30min for CITE)
**Would change:** Confirm: F-8's geometry-based abstention is redundant for deployment if CITE output-space signal dominates. Reject: F-8's DoM signal carries information beyond what output consistency captures, strengthening the case for internal-geometry methods.
**Blocks:** H-670

### H-742: W-CITE with prefill-DoM-derived confidence weights reduces expected stopping time relative to unweighted CITE on MATH-500
**Priority:** HIGH
**Motivated by:** 2605.05873 + F-2 + H-670
**Test:** Using K=16 samples from H-741, run W-CITE with DoM scores normalized to [0,1] as weights. Compare mean stopping time and certification rate against unweighted CITE. ~30min CPU.
**Requires:** CPU only (depends on H-741 for K=16 samples)
**Would change:** Confirm: DoM signal carries actionable information for output-space aggregation, validating H-670. Reject: DoM weights don't improve the effective weighted gap, suggesting DoM captures something orthogonal to answer-distribution structure.
**Blocks:** nothing

---


### H-743: L19's spectral gap is an outlier among transformer layers, explaining DoM ≈ PC1
**Priority:** HIGH
**Motivated by:** 2605.04418 + F-2 + H-711
**Test:** P11-FE946 — SVD all 28 layers' attention-out and MLP-down weight matrices in Qwen-2.5-1.5B. Compare (σ₁−σ₂)/σ₁ and stable rank κ profiles. Test whether L19 is >2σ outlier in spectral gap.
**Requires:** CPU only, HuggingFace model download.
**Would change:** Confirm → F-2's "single direction" has a weight-space mechanistic explanation via spectral dominance; H-711 (WeightWatcher alpha) gains a complementary spectral-gap metric. Reject → L19 is not spectrally special in weight space; DoM must arise from representational dynamics, not weight geometry.
**Blocks:** nothing

### H-744: The L19 DoM direction is the residual-stream image of the dominant right singular vector of L19's weight matrix
**Priority:** HIGH
**Motivated by:** 2605.04418 + F-2 + H-546
**Test:** P11-FE948 — project cached prefill activations onto weight SVD basis. Measure cos(v₁_weight, DoM) and cos(v₁_weight, PC1). If both > 0.8, the weight singular structure determines the activation geometry.
**Requires:** CPU only, cached P11 NPZs + HuggingFace weights.
**Would change:** Confirm → DoM is grounded in weight spectral structure, unifying F-2 with the manifold-constraint theory. H-546 is partially confirmed (backbone is weight-spectral, but mechanism is not AdamW-specific). Reject → DoM arises from input statistics or cross-layer interactions, not single-layer weight structure.
**Blocks:** H-546

### H-745: Disabling RMSNorm (identity substitution) at L19 preserves DoM AUROC within 0.02 of 0.7731
**Priority:** MEDIUM
**Motivated by:** 2605.04418 + F-2 + H-547
**Test:** P11-FE949 — freeze all RMSNorm layers to identity, re-extract L19 prefill activations on MATH-500, compute DoM AUROC.
**Requires:** H100, MATH-500 forward passes (~2h).
**Would change:** Confirm → correctness signal is in weight geometry, normalization is cosmetic for this purpose; strengthens manifold-constraint interpretation. Reject → normalization is load-bearing for correctness signal; the DoM depends on the norm-induced scale regulation, and manifold-constraint theory explains training stability but not inference-time probing.
**Blocks:** nothing

---


### H-746: SAM-pretrained LLMs show reduced or absent DoM correctness signal relative to AdamW-pretrained counterparts
**Priority:** HIGH
**Motivated by:** 2605.02105 + F-2, H-739
**Test:** Run P11 DoM extraction on the paper's SAM vs AdamW OLMo-2-1B checkpoints (both publicly available) on MATH-500 at 1024-tok. Compare L19 DoM AUROC. Threshold: if SAM AUROC < AdamW AUROC − 0.05, confirm.
**Requires:** H100, OLMo-2-1B SAM checkpoint, MATH-500 1024-tok pipeline
**Would change:** On confirm: F-2 must be qualified as optimizer-contingent; H-739 confirmed; deployment scope (APPLICATIONS.md) narrows to AdamW-trained models. On reject: F-2's universality strengthened; DoM is not a sharpness artifact.
**Blocks:** H-739

### H-747: The directional curvature of the AUROC surface around the DoM direction in L19 activation space is "sharp" (quadratic coefficient κ_act > 10), indicating fragility to direction perturbation
**Priority:** MEDIUM
**Motivated by:** 2605.02105 + F-2
**Test:** Compute AUROC(DoM + ε·v) for 100 random orthogonal directions v, fit quadratic. Cached L19 activations suffice.
**Requires:** CPU, cached NPZs
**Would change:** On confirm (sharp): deployment requires exact DoM direction, fragile to distribution shift. On reject (flat): DoM signal is robust, any nearby direction works, supporting deployment.
**Blocks:** nothing

---


### H-748: SAM-pretrained LLMs show reduced or absent DoM correctness signal relative to AdamW-pretrained counterparts
**Priority:** HIGH
**Motivated by:** 2605.02105 + F-2, H-739
**Test:** Run P11 DoM extraction on the paper's SAM vs AdamW OLMo-2-1B checkpoints (both publicly available) on MATH-500 at 1024-tok. Compare L19 DoM AUROC. Threshold: if SAM AUROC < AdamW AUROC − 0.05, confirm.
**Requires:** H100, OLMo-2-1B SAM checkpoint, MATH-500 1024-tok pipeline
**Would change:** On confirm: F-2 must be qualified as optimizer-contingent; H-739 confirmed; deployment scope (APPLICATIONS.md) narrows to AdamW-trained models. On reject: F-2's universality strengthened; DoM is not a sharpness artifact.
**Blocks:** H-739

### H-749: The directional curvature of the AUROC surface around the DoM direction in L19 activation space is "sharp" (quadratic coefficient κ_act > 10), indicating fragility to direction perturbation
**Priority:** MEDIUM
**Motivated by:** 2605.02105 + F-2
**Test:** Compute AUROC(DoM + ε·v) for 100 random orthogonal directions v, fit quadratic. Cached L19 activations suffice.
**Requires:** CPU, cached NPZs
**Would change:** On confirm (sharp): deployment requires exact DoM direction, fragile to distribution shift. On reject (flat): DoM signal is robust, any nearby direction works, supporting deployment.
**Blocks:** nothing

---


### H-750: Inter-layer curvature (angle between consecutive layer displacement vectors) separates correct from incorrect MATH-500 predictions at AUROC ≥ 0.73 at peak layer
**Priority:** HIGH
**Motivated by:** 2604.23985 + F-2
**Test:** Compute curvature from cached `prefill_all_layers` (29×1536), fit per-layer curvature AUROC, compare against DoM AUROC sweep.
**Requires:** CPU, cached NPZs (500 × 29 × 1536)
**Would change:** If confirmed, establishes curvature as a complementary geometric predictor to DoM; if refuted (AUROC < 0.55), curvature is an entropy feature but not a correctness feature.
**Blocks:** nothing

### H-751: The DoM direction has < 30% of its variance in the trajectory subspace (top-2 PCA of per-position displacement vectors at L19), predicting that DoM-direction steering (H-1) would fail under the curvature selectivity framework
**Priority:** HIGH
**Motivated by:** 2604.23985 + F-2 + H-1
**Test:** Compute per-sample trajectory PCA from `twothirds_positions`, project DoM, report variance fraction. Threshold: if in-trajectory variance < 30%, DoM is trajectory-misaligned.
**Requires:** CPU, cached NPZs
**Would change:** If confirmed, H-1 needs reformulation to use trajectory-aligned directions; if refuted (DoM is > 70% in-trajectory), the curvature selectivity result doesn't threaten H-1.
**Blocks:** H-1

### H-752: F-4 asymmetric collapse is better described as asymmetric straightening — correct samples have lower inter-layer curvature at final token than incorrect samples, and the curvature difference explains > 50% of the PR difference at the final token
**Priority:** MEDIUM
**Motivated by:** 2604.23985 + F-4
**Test:** Compute curvature from `final_all_layers`, compare correct vs incorrect group means, correlate with per-sample PR at final token.
**Requires:** CPU, cached NPZs
**Would change:** If confirmed, reframes F-4 from dimensional to trajectory-geometric terms, connecting collapse to the temporal straightening hypothesis. If refuted, collapse and curvature are independent phenomena.
**Blocks:** nothing

---


### H-753: FFN activation sparsity at L19 predicts correctness as well as DoM
**Priority:** HIGH
**Motivated by:** 2603.23198 + F-2
**Test:** Extract L19 FFN gate activations for Qwen-2.5-1.5B on MATH-500, compute per-problem mean L0 norm (count of active neurons across prefill positions). Train 5-fold OOF logistic classifier on L0 alone. Compare AUROC to DoM's 0.7731. Also test multivariate classifier on per-position L0 profile.
**Requires:** GPU (single forward pass with activation hook), ~1h H100.
**Would change:** If confirmed (L0 AUROC >= 0.75): DoM's geometric interpretation is weakened; F-2 needs reframing as "sparsity routing signal" rather than "geometric direction." If rejected (L0 AUROC << 0.7): confirms DoM captures genuine geometric structure beyond sparsity patterns.
**Blocks:** H-5 (prefill signal encodes problem familiarity vs decomposability)

### H-754: Asymmetric collapse (F-4) is driven by differential FFN sparsity between correct and incorrect final tokens
**Priority:** MEDIUM
**Motivated by:** 2603.23198 + F-4
**Test:** Extract final-token FFN L0 norms at L19 for correct vs incorrect MATH-500 problems. Mann-Whitney U test on L0 distributions. Effect size: Cohen's d and AUROC of L0 alone for correct/incorrect classification at the final token position.
**Requires:** GPU (single forward pass), ~1h H100.
**Would change:** If confirmed: F-4's asymmetric collapse gets a simple mechanistic explanation (fewer active neurons → lower-rank residual → lower PR). If rejected: the asymmetric collapse is a genuine residual-stream geometric phenomenon not reducible to FFN sparsity.
**Blocks:** nothing

---


### H-755: Correct vs incorrect L19 prefill groups have different activation covariance spectral decay (α_tail), and this spectral-shape feature predicts correctness at AUROC ≥ 0.75
**Priority:** HIGH
**Motivated by:** 2605.05683 + F-2
**Test:** Compute centered covariance of L19 activations separately for correct (n=243) and incorrect (n=257) MATH-500 samples. Trace-normalize eigenvalues, fit power-law α over rank windows [100,200] and [200,400]. Compare mean α_tail between groups; train a logistic classifier on per-sample spectral-shape features (variance ratio in different rank bands). 15min CPU.
**Requires:** CPU, cached L19 NPZs
**Would change:** If confirmed: F-2's "single direction" framing becomes "single direction within a richer spectral structure"; motivates spectral-shape features for selective prediction. If rejected: confirms DoM captures the relevant structure and spectral shape is uninformative.
**Blocks:** nothing

### H-756: RankMe (entropy effective rank) of per-sample L19 neighborhoods differs between correct and incorrect groups
**Priority:** MEDIUM
**Motivated by:** 2605.05683 + F-2 + F-10
**Test:** Compute RankMe of the covariance matrix restricted to correct-only vs incorrect-only L19 activations. Also compute a per-sample proxy: local effective rank in k-NN neighborhoods (k=10,20,50). Compare group means and test as a correctness predictor. 10min CPU.
**Requires:** CPU, cached L19 NPZs
**Would change:** If confirmed: adds a label-free spectral diagnostic to the selective-prediction toolkit. If rejected: spectral spread is not correctness-informative at the group level.
**Blocks:** nothing

---


### H-757: Correct vs incorrect L19 prefill groups have different activation covariance spectral decay (α_tail), and band-restricted α features predict correctness at AUROC ≥ 0.78
**Priority:** HIGH
**Motivated by:** 2605.05683 + F-2 + FE881
**Test:** Compute centered covariance of L19 activations separately for correct (n=243) and incorrect (n=257) MATH-500 samples. Trace-normalize eigenvalues, fit power-law α over adapted rank windows [50,150] and [150,350]. Compare mean α_tail between groups. Train logistic classifier on per-window α features. 15min CPU.
**Requires:** CPU, cached L19 NPZs
**Would change:** If confirmed: F-2's "single direction" framing becomes "single direction within a richer spectral structure"; motivates multi-band spectral features for selective prediction. If rejected: confirms DoM and top-20 cov-spectrum already capture the relevant spectral information.
**Blocks:** nothing

### H-758: Correct L19 prefill activations have higher spectral energy concentration in the DoM direction than incorrect activations
**Priority:** MEDIUM
**Motivated by:** 2605.05683 task-band concentration + F-2
**Test:** Compute H_DoM = (DoM^T Σ DoM) / tr(Σ) for correct-only and incorrect-only covariance matrices. Two-sample permutation test on the difference. 10min CPU.
**Requires:** CPU, cached L19 NPZs + cached DoM direction
**Would change:** If confirmed: provides a spectral-theoretic explanation for why DoM works — correct samples have more concentrated spectral energy along DoM. If rejected: suggests the DoM signal is about directional shift, not energy concentration.
**Blocks:** nothing

---


### H-759: DoM correctness signal is robust to synonym-level input perturbations
**Priority:** HIGH
**Motivated by:** 2605.04344 + F-2
**Test:** Apply random insertion/deletion/swap perturbations at α ∈ {0.025, 0.05, 0.10} to MATH-500 prefills, re-extract L19 activations, compute DoM AUROC. Compare to baseline 0.7731. N=500 problems × 3 perturbation levels × 10 perturbation samples = 15,000 forward passes.
**Requires:** H100 for forward passes (~1h), CPU for AUROC computation
**Would change:** If confirmed: strengthens F-5 (content-dependent, not token-identity-dependent). If rejected: implies DoM is fragile to paraphrasing, and selective prediction (F-8) needs robustification.
**Blocks:** nothing

### H-760: DoM AUROC degrades with distance from training support
**Priority:** MEDIUM
**Motivated by:** 2605.04344 Proposition 1 + F-2 + H-5
**Test:** Compute per-problem perplexity on MATH-500 under frozen Qwen-2.5-1.5B as proxy for Hamming distance from training support. Stratify DoM AUROC by perplexity quintile. Test for monotonic degradation via Spearman rank correlation. CPU only, ~30min.
**Requires:** CPU, cached L19 activations, frozen model for perplexity computation
**Would change:** If confirmed (ρ > 0.8 monotonic degradation): DoM is an interpolation signal, supports H-5 "familiarity" interpretation. If rejected (AUROC flat across quintiles): DoM encodes something beyond surface similarity, more consistent with H-5 "decomposability" interpretation.
**Blocks:** nothing

---


### H-761: L19 prefill activations lie on a curved submanifold of ℝ^1536, and Riemannian-aware distances outperform Euclidean distances for correct/incorrect separation
**Priority:** MEDIUM
**Motivated by:** 2605.04255 + F-2 + F-10
**Test:** Compute k-NN graph shortest-path distances (Isomap-style geodesic approximation) on cached 500×1536 L19 prefill activations. Compare Spearman ρ between Euclidean and geodesic pairwise distance matrices. If ρ < 0.95, activations have non-trivial curvature; then compare AUROC of geodesic-distance-based kNN classifier vs Euclidean DoM (0.7731). ~20min CPU.
**Requires:** CPU, cached L19 activations, scikit-learn Isomap or custom k-NN graph
**Would change:** If confirmed: F-2's linear DoM may be suboptimal; F-10's Euclidean PH null may be an artifact of wrong metric. If rejected: Euclidean assumptions hold and manifold methods are unnecessary for this data.
**Blocks:** nothing

### H-762: Entropic OT coupling weights between correct/incorrect L19 prefill distributions provide a per-sample correctness score that outperforms scalar DoM projection
**Priority:** MEDIUM
**Motivated by:** 2605.04255 + F-2
**Test:** Run Sinkhorn on cached 500×1536 L19 activations split by correctness label with ε = 0.05 × median(cost). Extract per-sample transport cost or marginal entropy from coupling. Train logistic regression on these features; compare 5-fold AUROC to DoM 0.7731. ~30min CPU.
**Requires:** CPU, cached L19 activations, POT or OTT-JAX library
**Would change:** If confirmed (AUROC > 0.79): correctness signal is distributional, not just directional — opens new feature family. If rejected: linear direction is sufficient and distributional methods add no value.
**Blocks:** nothing

---


### H-763: L19 DoM direction is the weight v1 of an L19 projection matrix (structural artifact, not learned correctness signal)
**Priority:** HIGH
**Motivated by:** 2605.04971 + F-2
**Test:** Extract weight v1 for each L19 projection (Q, K, V, O, Gate, Up, Down) in Qwen-2.5-1.5B. Compute cos(weight_v1, DoM_supervised). If cos > 0.85 for any projection, the DoM direction is plausibly a weight-structural artifact. Est: 20 min CPU.
**Requires:** CPU, Qwen-2.5-1.5B weights (HF download), cached DoM direction from P11 results.
**Would change:** If confirmed: F-2 reframed from "learned correctness signal" to "training-mechanics artifact that correlates with correctness." If rejected (all cos < 0.3): F-2's semantic interpretation strengthened.
**Blocks:** H-5 (if DoM is structural, the "familiarity vs. decomposability" interpretation needs revision)

### H-764: L19 is not special — adjacent layers (L17–L21) achieve comparable DoM AUROC
**Priority:** HIGH
**Motivated by:** 2605.04971 (geometric continuity predicts smooth v1 across layers) + F-2
**Test:** Compute per-layer DoM AUROC for layers L14–L24 on cached 1024-tok data (if available per-layer) or extract per-layer activations and train DoM probes. If AUROC is flat (±0.02) across L17–L21, L19 is not special. Est: depends on cached per-layer data availability.
**Requires:** CPU for SVD; may need per-layer cached activations (check DATA_MANIFEST).
**Would change:** If confirmed: F-2's "single L19 direction" becomes "broad mid-layer band." If rejected (sharp L19 peak): geometric continuity framework insufficient to explain L19 specialness.
**Blocks:** nothing

### H-765: F-3's prefill/final DoM orthogonality is explained by projection-specific continuity spaces (input-space v1 vs. output-space u1)
**Priority:** MEDIUM
**Motivated by:** 2605.04971 (projection-specific continuity) + F-3
**Test:** Decompose L19 residual stream contribution at prefill and final token into per-projection components. Compute alignment of each component with prefill DoM and final-token DoM. If prefill DoM aligns with Q/K/Gate input-space and final DoM aligns with O/Down output-space, orthogonality is architectural. Est: 30 min CPU (weight-only analysis) or 1–2h if per-token attribution needed.
**Requires:** CPU, Qwen-2.5-1.5B weights, cached DoM directions.
**Would change:** If confirmed: F-3 reframed from semantic (comprehension vs. generation) to architectural (read-space vs. write-space). If rejected: F-3's semantic interpretation survives.
**Blocks:** H-17 (if orthogonality is architectural, the RoPE-mechanical hypothesis becomes less relevant)

---


### H-766: L19 DoM direction is the weight v1 of an L19 projection matrix (structural artifact, not learned correctness signal)
**Priority:** HIGH
**Motivated by:** 2605.04971 + F-2
**Test:** Extract weight v1 for each L19 projection (Q, K, V, O, Gate, Up, Down) in Qwen-2.5-1.5B. Compute cos(weight_v1, DoM_supervised). If cos > 0.85 for any projection, the DoM direction is plausibly a weight-structural artifact. Est: 20 min CPU.
**Requires:** CPU, Qwen-2.5-1.5B weights (HF download), cached DoM direction from P11 results.
**Would change:** If confirmed: F-2 reframed from "learned correctness signal" to "training-mechanics artifact that correlates with correctness." If rejected (all cos < 0.3): F-2's semantic interpretation strengthened.
**Blocks:** H-5 (if DoM is structural, the "familiarity vs. decomposability" interpretation needs revision)

### H-767: L19 is not special — adjacent layers (L17–L21) achieve comparable DoM AUROC
**Priority:** HIGH
**Motivated by:** 2605.04971 (geometric continuity predicts smooth v1 across layers) + F-2
**Test:** Compute per-layer DoM AUROC for layers L14–L24 on cached 1024-tok data (if available per-layer) or extract per-layer activations and train DoM probes. If AUROC is flat (±0.02) across L17–L21, L19 is not special. Est: depends on cached per-layer data availability.
**Requires:** CPU for SVD; may need per-layer cached activations (check DATA_MANIFEST).
**Would change:** If confirmed: F-2's "single L19 direction" becomes "broad mid-layer band." If rejected (sharp L19 peak): geometric continuity framework insufficient to explain L19 specialness.
**Blocks:** nothing

### H-768: F-3's prefill/final DoM orthogonality is explained by projection-specific continuity spaces (input-space v1 vs. output-space u1)
**Priority:** MEDIUM
**Motivated by:** 2605.04971 (projection-specific continuity) + F-3
**Test:** Decompose L19 residual stream contribution at prefill and final token into per-projection components. Compute alignment of each component with prefill DoM and final-token DoM. If prefill DoM aligns with Q/K/Gate input-space and final DoM aligns with O/Down output-space, orthogonality is architectural. Est: 30 min CPU (weight-only analysis) or 1–2h if per-token attribution needed.
**Requires:** CPU, Qwen-2.5-1.5B weights, cached DoM directions.
**Would change:** If confirmed: F-3 reframed from semantic (comprehension vs. generation) to architectural (read-space vs. write-space). If rejected: F-3's semantic interpretation survives.
**Blocks:** H-17 (if orthogonality is architectural, the RoPE-mechanical hypothesis becomes less relevant)

---


### H-769: DoM direction at L19 is a semantic-Evaluation axis, not a dedicated correctness signal
**Priority:** HIGH
**Motivated by:** 2604.27169 + F-2
**Test:** Construct semantic axes (true-false, good-bad, certain-uncertain, etc.) in Qwen-2.5-1.5B at L19 via contrastive antonym pairs. Measure cos(DoM, axis) and per-axis correctness AUROC. If DoM aligns with evaluative axes and semantic projection matches DoM AUROC, the hypothesis is confirmed. Est: 1h CPU.
**Requires:** CPU, cached L19 activations, Qwen-2.5-1.5B embedding matrix (for token-level axis construction).
**Would change:** Confirm → F-2 interpretation shifts from "correctness computation" to "semantic evaluation signal." Reject → F-2 interpretation strengthened as a task-specific learned direction.
**Blocks:** H-1 (if confirmed, DoM steering is steering on semantics, not correctness)

### H-770: Prefill/final DoM orthogonality (F-3) is explained by semantic subspace rotation across token positions
**Priority:** MEDIUM
**Motivated by:** 2604.27169 + F-3
**Test:** Compute 3-d PCA semantic subspaces separately for prefill and final-token L19 activations. Measure subspace angle (principal angles via SVD). If both DoM directions lie within their respective 3-d subspaces but the subspaces are rotated, orthogonality is a rotation artifact. Est: 30min CPU.
**Requires:** CPU, cached L19 prefill and final-token activations.
**Would change:** Confirm → F-3 reframed from "distinct computational processes" to "same semantic structure, different orientation." Reject → F-3 strengthened.
**Blocks:** nothing

---


### H-771: Per-sample Gaussian curvature of the L19 activation manifold correlates with correctness
**Priority:** HIGH
**Motivated by:** 2409.05084 + F-2 + F-10
**Test:** Compute shape operator curvature K_i = det(S_i) at each of the 500 L19 prefill activation points (PCA-reduced to 50 dims, k=9 patches). Compute point-biserial correlation and AUROC of K_i vs correct/incorrect labels. Est 30min CPU.
**Requires:** CPU, cached NPZs (pathway11_h100), numpy/scipy
**Would change:** Confirm → narrows F-10 to "PH doesn't help" rather than "all geometry beyond covariance is useless." Reject → strengthens F-10's broader interpretation.
**Blocks:** nothing

### H-772: kK-NN curvature-adaptive classifier exceeds linear DoM AUROC 0.7731 on L19 prefill activations
**Priority:** HIGH
**Motivated by:** 2409.05084 + F-2 + H-26
**Test:** Run kK-NN (from github.com/alexandrelevada/kkNN) on PCA-reduced L19 prefill activations, binary classification, 5-fold CV. Compare AUROC and balanced accuracy to logistic regression DoM. Est 45min CPU.
**Requires:** CPU, cached NPZs, kK-NN Python package
**Would change:** Confirm → evidence that nonlinear decision boundaries improve correctness prediction (supports H-26). Reject → further evidence that the correctness signal is largely linear/directional.
**Blocks:** H-26

### H-773: D-bucket samples occupy high-curvature regions of the L19 activation manifold
**Priority:** MEDIUM
**Motivated by:** 2409.05084 + F-7 + H-7
**Test:** Compute per-sample curvature scores, stratify by ABCD-bucket. Test whether D-bucket curvature distribution differs from A-bucket (KS test). Est 20min CPU.
**Requires:** CPU, cached NPZs, ABCD bucket labels
**Would change:** Confirm → D-bucket geometric signature (F-7) is curvature-driven, reframes H-7. Reject → D-bucket signature is density/directional rather than curvature-based.
**Blocks:** H-7

---


### H-774: DGPO-trained Qwen-2.5-1.5B has a different L19 prefill DoM direction than GRPO-trained
**Priority:** HIGH
**Motivated by:** 2605.03327 + F-2 + H-707
**Test:** DGPO-train Qwen-2.5-1.5B-Base on DAPO-17K using the paper's recipe (adapted from 7B), extract L19 prefill activations on MATH-500, compute cos(DoM_DGPO, DoM_GRPO) and AUROC_DGPO. Threshold: cos < 0.5 → F-2 is RL-objective-specific.
**Requires:** H100, DGPO training code, DAPO-17K dataset
**Would change:** On confirm (cos < 0.5): F-2 and F-8 are partly GRPO artifacts, qualifying all downstream applications. On reject (cos > 0.8): DoM is robust to RL objective, strengthening F-2 universality. Subsumes H-707.
**Blocks:** H-282

### H-775: Per-sample Shannon entropy of L19 prefill activations predicts correctness independently of DoM
**Priority:** HIGH
**Motivated by:** 2605.03327 + F-2
**Test:** Softmax-normalize each L19 prefill activation vector (500 × 1536, cached), compute Shannon entropy per sample, evaluate AUROC for correctness prediction. Compare to 0.7731 (DoM) and test partial correlation controlling for DoM projection.
**Requires:** CPU only, cached NPZ
**Would change:** On confirm (entropy AUROC > 0.73 and partial corr with DoM < 0.5): entropy is an independent correctness feature, motivating a 2-feature (DoM + entropy) probe. On reject: entropy is redundant with DoM or uninformative.
**Blocks:** nothing

---


### H-776: Geodesic distance on fitted L19 activation manifold predicts correctness better than linear DoM
**Priority:** HIGH
**Motivated by:** 2605.05115 + F-2
**Test:** Fit cubic spline manifold to L19 prefill activations in PCA-64 subspace (correct/incorrect centroids). Compute geodesic distances from each sample to correct/incorrect centroids. AUROC of geodesic-distance classifier vs. 0.7731 linear DoM.
**Requires:** CPU, cached NPZs
**Would change:** If geodesic AUROC > 0.7731: F-2's ceiling is a linearization artifact, motivates manifold-aware probes. If geodesic AUROC ≤ 0.7731: linear approximation is sufficient for this task, validates current approach.
**Blocks:** nothing

### H-777: L19 prefill activation manifold has non-Gaussian curvature that PH missed
**Priority:** HIGH
**Motivated by:** 2605.05115 + F-10
**Test:** Fit spline manifold to L19 prefill activations. Compute Riemannian curvature statistics along the manifold. Compare against Gaussian null ensemble (same N, same covariance). If curvature rejects Gaussian null at p<0.01, F-10's PH approach was tool-limited.
**Requires:** CPU, cached NPZs
**Would change:** If curvature is non-Gaussian: F-10 is incomplete (PH misses manifold curvature); H-35 confirmed. If curvature is Gaussian: F-10's verdict extends to manifold geometry.
**Blocks:** H-35

### H-778: Manifold steering along L19 geodesic outperforms linear DoM addition for MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2605.05115 + H-1
**Test:** Fit L19 activation manifold. Steer along manifold geodesic (from incorrect-cluster centroid toward correct-cluster centroid) rather than linear DoM addition. Compare MATH-500 accuracy and fluency at matched intervention strength.
**Requires:** H100, model weights, manifold fitting
**Would change:** If manifold steering > linear: H-1's protocol needs redesign. If manifold steering ≈ linear: activation space is approximately flat for math correctness (linear approximation sufficient).
**Blocks:** H-1

### H-779: Prefill/final-token DoM orthogonality (F-3) is tangent-plane variation on a single activation manifold
**Priority:** MEDIUM
**Motivated by:** 2605.05115 + F-3
**Test:** Fit single manifold to L19 activations across token positions. Check whether prefill and final-token centroids are on the same manifold with orthogonal tangent planes. Measure on-manifold geodesic distance between the two.
**Requires:** CPU, cached NPZs for both prefill and final-token
**Would change:** If same manifold: F-3 is a manifestation of manifold curvature, not two circuits. If different manifolds: F-3's independent-circuit interpretation is supported.
**Blocks:** nothing

---


### H-780: Mean-pooled prefill DoM AUROC is substantially higher than last-token-only prefill DoM AUROC
**Priority:** HIGH
**Motivated by:** 2604.12016 + F-2
**Test:** Extract last prefill token L19 activation (single token, not mean-pooled) for all 500 MATH-500 problems. Compute DoM direction via logistic regression, measure 5-fold OOF AUROC. Compare to mean-pooled 0.7731. If last-token AUROC < 0.72, the mean-pooling aggregation mechanism contributes meaningfully to F-2's signal.
**Requires:** CPU if per-token activations cached; 2h GPU on 2060 Super if re-extraction needed.
**Would change:** On confirm: F-2 needs qualification ("mean-pooled" is load-bearing, not incidental). On reject (AUROCs comparable): strengthens F-2's "single direction" claim — the signal is genuinely per-token, not an aggregation artifact.
**Blocks:** nothing

### H-781: Correct-answer MATH-500 L19 activations form a tighter cluster (lower within-group cosine distance) than incorrect-answer activations
**Priority:** MEDIUM
**Motivated by:** 2604.12016 + F-4 + F-7
**Test:** Compute pairwise cosine distance matrices within correct (n=243) and within incorrect (n=257) groups at L19 prefill. Run Welch t-test on within-correct vs within-incorrect distances. Report Cohen's d. Prediction from F-4 (asymmetric collapse): within-correct < within-incorrect.
**Requires:** CPU, cached NPZs.
**Would change:** On confirm with d > 1: F-4 collapse reinterpretable as attractor convergence. On reject: asymmetric collapse is not geometric (challenges F-4).
**Blocks:** nothing

---


### H-782: Prefill DoM correctness signal is ICL-anchor-mediated rather than problem-intrinsic
**Priority:** HIGH
**Motivated by:** 2305.14160 + F-2
**Test:** Extract L19 prefill activations under zero-shot MATH-500 (no demonstrations). If DoM AUROC drops to <0.60, signal is anchor-mediated; if it stays >0.72, it's problem-intrinsic geometry.
**Requires:** CPU + one forward pass on Qwen-2.5-1.5B (zero-shot), cached DoM probe coefficients
**Would change:** On confirm (anchor-mediated): F-2 interpretation shifts from "the model encodes correctness in geometry" to "the model's ICL routing quality predicts correctness." On reject (problem-intrinsic): Strengthens F-2's claim as genuine correctness encoding.
**Blocks:** nothing

### H-783: L19 is a sharp phase-transition layer where correctness signal concentrates (>80% of cumulative R_l)
**Priority:** MEDIUM
**Motivated by:** 2305.14160 + F-2 + F-9
**Test:** Compute R_l cumulative ratio from existing layer-wise AUROC data. Threshold: >80% at L19 = sharp transition; <60% = distributed.
**Requires:** CPU only, existing layer-wise data from P8
**Would change:** On confirm: L19 is a privileged "anchor extraction" layer, motivating head-level decomposition (H-13). On reject: Signal distributes gradually, CoE-60 redundancy (F-9) is coincidental.
**Blocks:** nothing

---


### H-784: Prefill DoM correctness signal has a scale-invariant (TR-like) component and a scale-dependent (TL-like) component
**Priority:** LOW
**Motivated by:** 2305.09731 + F-2 + F-6
**Test:** Extract L19 DoM AUROC at Qwen-2.5-0.5B, 1.5B, 3B, 7B on MATH-500 with identical 1024-tok extraction. Fit a two-component model: constant (TR) + scale-dependent (TL). Check if the residuals from a constant-only model are significantly reduced.
**Requires:** GPU for fresh extraction at 0.5B and 3B; 1.5B and 7B already cached.
**Would change:** If confirmed, the DoM signal is a mixture and cross-scale comparisons must decompose it. If rejected (signal is purely scale-dependent or purely scale-invariant), the TR/TL framing adds nothing.
**Blocks:** nothing

---


### H-785: L19 DoM correctness signal is the principal axis of an implicit kernel Mahalanobis distance
**Priority:** HIGH
**Motivated by:** 2305.12766 + F-2
**Test:** Compute Mahalanobis distance d_M(hᵢ, μ) from cached L19 activations using the inverse sample covariance. Compare AUROC against DoM (0.7731). If d_M ≥ DoM, the kernel framing subsumes the directional one.
**Requires:** CPU only, cached NPZ activations + covariance matrix
**Would change:** On confirm, F-2 reinterprets DoM as a kernel proxy, motivating full Mahalanobis-based selective prediction. On reject, DoM is not explained by the kernel view.
**Blocks:** nothing

### H-786: Specific attention heads at L16-L21 of Qwen-2.5-1.5B implement a kernel regression whose confidence predicts MATH-500 correctness
**Priority:** HIGH
**Motivated by:** 2305.12766 + F-2 + H-13
**Test:** Extract per-head attention maps at L16-L21 during MATH-500 prefill. For each head, compute reconstructed output via attention-weighted label averaging. Identify head(s) where reconstruction accuracy > 80% (paper baseline on SST2).
**Requires:** H100 (attention extraction forward pass), ~3h
**Would change:** On confirm, provides mechanistic explanation for F-2 and validates H-13. On reject, kernel regression mechanism may not transfer from classification (SST2) to mathematical reasoning.
**Blocks:** H-13

---


### H-787: All-layer DoM steering (ICV-style) outperforms L19-only steering for MATH-500 accuracy
**Priority:** HIGH
**Motivated by:** 2311.06668 Table 3 + F-2 + H-1
**Test:** Compute per-layer DoM directions from cached per-layer activations (needs extraction at all 28 layers). Apply λ·DoM_l at each layer during generation with norm-preservation (Eq. 6 of 2311.06668). Sweep λ∈{0.05, 0.1, 0.15, 0.2}. Compare accuracy delta to L19-only steering. Est. 2–3h H100.
**Requires:** H100, Qwen-2.5-1.5B, per-layer cached activations (need extraction if not cached for all layers), MATH-500
**Would change:** On confirm: H-1 becomes "all-layer steering" not "L19 steering"; F-9 reopened for intervention. On reject: steering with contrastive directions is a dead end for math reasoning.
**Blocks:** H-1

### H-788: ICV's norm-preservation step is load-bearing for DoM-based steering
**Priority:** MEDIUM
**Motivated by:** 2311.06668 Eq. 6 + H-1
**Test:** Compare MATH-500 accuracy under DoM steering with vs without L2-norm renormalization after direction addition. 1h H100.
**Requires:** H100, Qwen-2.5-1.5B, MATH-500
**Would change:** On confirm: norm-preservation becomes a required component of any H-1 implementation. On reject: direction-only shift is sufficient.
**Blocks:** nothing

---


### H-789: Output-based semantic entropy dominates internal DoM for MATH-500 correctness prediction
**Priority:** HIGH
**Motivated by:** 2302.09664 + F-2
**Test:** Generate M=10 samples per MATH-500 problem, cluster by answer equivalence, compute SE AUROC. Compare to DoM AUROC 0.7731 and cov-spectrum 0.7928.
**Requires:** GPU for 10× generation, CPU for clustering. Or: CPU-only with cached K≥2 completions using simplified cluster-count proxy.
**Would change:** If confirmed, practical motivation for internal-geometry probing shifts from "best available" to "cheapest single-pass" — still useful for pre-generation abstention but no longer AUROC-optimal.
**Blocks:** nothing

### H-790: DoM and semantic entropy capture orthogonal uncertainty signals
**Priority:** HIGH
**Motivated by:** 2302.09664 + F-2 + F-3
**Test:** Rank-correlate DoM scores with answer-diversity metrics across MATH-500 problems. rho < 0.4 confirms orthogonality; rho > 0.8 confirms redundancy.
**Requires:** CPU only (cached activations + cached completions).
**Would change:** If orthogonal, combined model should push AUROC well above 0.7928; F-3's prefill/final distinction gets a concrete operational interpretation (knowledge vs expression). If redundant, F-2's practical value is reduced to "zero-cost approximation of SE."
**Blocks:** H-789

---


### H-791: L19 DoM is a linear proxy for logit-lens prediction entropy at L19
**Priority:** HIGH
**Motivated by:** 2310.04625 + F-2
**Test:** Decode cached L19 activations via W_U, compute per-sample entropy, correlate with DoM magnitude. If r² > 0.7, DoM ≈ entropy.
**Requires:** CPU, Qwen-2.5-1.5B unembedding matrix (W_U), cached L19 NPZs.
**Would change:** If confirmed, F-2's "correctness direction" interpretation becomes "calibration direction" — still useful for selective prediction but mechanistically less novel.
**Blocks:** H-1 (steering interpretation changes if DoM = calibration)

### H-792: Copy suppression self-repair limits DoM-based steering effectiveness (H-1)
**Priority:** MEDIUM
**Motivated by:** 2310.04625 Section 4 + H-1
**Test:** Attempt DoM steering on Qwen-2.5-1.5B; measure whether downstream heads (L20+) compensate by reducing their copy suppression, partially undoing the intervention.
**Requires:** H100, full inference with interventions.
**Would change:** If confirmed, H-1 steering experiments need anti-self-repair strategies (e.g., also ablating the compensating heads). If rejected, self-repair is weak in Qwen and steering is viable.
**Blocks:** nothing

---


### H-793: Correctness signal in L19 concentrates in high-kurtosis (non-Gaussian) neuron activations
**Priority:** HIGH
**Motivated by:** 2401.12181 + F-2 + F-10
**Test:** Partition L19 residual dimensions by kurtosis (threshold at 5). Train separate logistic regressors on high-kurtosis and low-kurtosis subsets. If high-kurtosis subset matches DoM AUROC (0.7731) while low-kurtosis subset is near-chance, correctness signal is sparse/non-Gaussian.
**Requires:** CPU, cached NPZ activations from P11.
**Would change:** Reframes F-10 (PH null is expected because PH sees Gaussian bulk). Strengthens F-2 if DoM aligns with high-kurtosis subspace.
**Blocks:** nothing

### H-794: Qwen-2.5-1.5B has entropy neurons at late layers analogous to GPT2's, and their activation correlates with DoM confidence score
**Priority:** HIGH
**Motivated by:** 2401.12181 + F-8
**Test:** Identify high-norm, low-logit-variance neurons in Qwen-2.5-1.5B layers 18–23. Measure Pearson correlation of their mean activation (over prefill positions) with per-problem DoM score. If |r| > 0.5, entropy neurons are a mechanistic substrate for DoM-based confidence.
**Requires:** Model weight access (CPU), cached activation data.
**Would change:** Provides causal grounding for F-8 selective prediction. If confirmed, suggests a simpler 1-neuron abstention rule.
**Blocks:** nothing

---


### H-795: SAE decomposition of L19 DoM reveals multiple distinct correctness-predictive latents
**Priority:** HIGH
**Motivated by:** 2406.04093 + F-2
**Test:** Train SAE (≥4096 latents, k=16–64) on Qwen-2.5-1.5B L19 activations; for each latent, train single-latent correctness probe; count latents with AUROC > 0.65 and test whether their weighted sum exceeds DoM AUROC 0.7731.
**Requires:** H100, ~500M tokens of math-adjacent text for SAE training, Qwen-2.5-1.5B forward passes.
**Would change:** If confirmed: F-2 needs reframing from "single direction" to "dominant feature axis in superposition." If rejected (DoM is irreducible): strengthens the geometric-signal interpretation.
**Blocks:** H-796

### H-796: The cov-spectrum ceiling (0.7928) is a basis limitation, not an information ceiling
**Priority:** MEDIUM
**Motivated by:** 2406.04093 + F-2 (cov-spectrum FE881)
**Test:** Compare SAE-latent-based correctness AUROC (using top-20 latents ordered by individual probe AUROC) against cov-spectrum 0.7928. If SAE probe > 0.82 on same data split, covariance basis leaves signal on the table.
**Requires:** Trained SAE from H-795; same eval protocol as FE881.
**Would change:** If confirmed: motivates switching from PCA to SAE features for all downstream selective-prediction work. If rejected: validates that covariance basis is near-optimal and superposition is mild at this layer.
**Blocks:** nothing

---


### H-797: Inference-time activation SD (Kaplan-Yorke analog) differs between correct and incorrect MATH-500 problems
**Priority:** HIGH
**Motivated by:** 2604.19740 + F-10 + F-2
**Test:** Compute per-problem covariance eigenspectra from L19 activations across token positions. Apply log-transform and Kaplan-Yorke formula to get activation-SD per problem. Test whether mean(SD|correct) ≠ mean(SD|incorrect) with p < 0.01.
**Requires:** CPU only, cached NPZ activations from P11 H100 extraction.
**Would change:** On confirm: H-4 gains strong evidence; F-10 gets nuanced (PH null but SD non-null). On reject: H-4 weakened — training-time SD doesn't transfer to inference activations.
**Blocks:** H-4

### H-798: The gap between DoM AUROC (0.7731) and cov-spectrum AUROC (0.7928) corresponds to information in Lyapunov directions 2 through j*
**Priority:** MEDIUM
**Motivated by:** 2604.19740 Theorem 4.5 + F-2 + FE881
**Test:** Compute activation-SD j* from cached eigenspectrum. Regress correctness on PC directions 1..j* with ridge regression. Compare AUROC to (a) PC1-only (≈ DoM at cos=0.9216) and (b) top-20 cov-spectrum. If the j*-direction probe matches cov-spectrum AUROC more closely than either PC1 or all-1536, SD theory predicts the information boundary.
**Requires:** CPU only, cached PCA results.
**Would change:** On confirm: SD provides principled feature-selection criterion for probe design. On reject: SD doesn't determine which covariance directions are informative for correctness.
**Blocks:** nothing

---


### H-799: Prefill DoM direction is invariant to prompt paraphrase (reworded MATH-500 problems yield cos > 0.9 with original DoM)
**Priority:** MEDIUM
**Motivated by:** 2307.13339 + F-2
**Test:** Reword 20+ MATH-500 problems preserving mathematical content, extract L19 prefill activations under original and reworded prompts, compare per-problem DoM projections and overall fitted DoM direction. Threshold: cos(DoM_orig, DoM_reworded) > 0.9 and AUROC delta < 0.02.
**Requires:** H100 for 20-problem re-extraction (~20min), CPU for analysis
**Would change:** Confirm: F-2 and F-8 are robust to deployment variation. Reject: AUROC 0.7731 is prompt-template-specific, selective prediction needs format-matched training.
**Blocks:** nothing

---


### H-800: The DoM direction's predictive power is geometrically inevitable (lives entirely in top-3 PCA directions)
**Priority:** HIGH
**Motivated by:** 2604.09780 Prop. 1 + F-2
**Test:** Compute data-aware Lipschitz bound ‖DoM·Π_r‖₂ for r=1..50 on cached L19 prefill activations. If r=1 captures >95% of DoM's discriminative power, the direction is "trivial geometry."
**Requires:** CPU only, cached NPZs
**Would change:** If confirmed, F-2 is demoted from "novel correctness signal" to "restatement of PCA structure." If rejected (significant power in r>3), DoM captures something beyond crude variance structure.
**Blocks:** nothing

### H-801: L19 prefill activations exhibit MoE-like "router collapse" — near-identical principal subspace across semantically different math problems
**Priority:** MEDIUM
**Motivated by:** 2604.09780 Section 5.3 (router collapse during prefilling) + F-3 + F-10
**Test:** Compute per-problem L19 activation SVD, measure cross-problem cosine similarity of first singular vector. If > 0.95 across all 500 problems, L19 shows the same collapse pattern documented for MoE routers at depth.
**Requires:** CPU only, cached NPZs
**Would change:** If confirmed, explains F-10 (PH null is because all problems share the same principal subspace → Gaussian-like), strengthens F-3 (prefill orthogonality is regime effect). If rejected, L19 activations are more diverse than MoE deep-layer states.
**Blocks:** nothing

### H-802: DoM-orthogonal residual carries independent correctness signal accounting for the 0.7731→0.7928 AUROC gap
**Priority:** HIGH
**Motivated by:** 2604.09780 Prop. 2 (direction suppression) + F-2 + H-798
**Test:** Project out DoM from L19 prefill activations, fit logistic regression on residualized activations. Compare residual AUROC to the 0.0197 gap between DoM (0.7731) and cov-spectrum ceiling (0.7928).
**Requires:** CPU only, cached NPZs
**Would change:** If confirmed, identifies the "missing" AUROC as DoM-orthogonal spectral information. If rejected, the gap is noise/overfitting in cov-spectrum probe.
**Blocks:** H-798

---


### H-803: L19 DoM detects convergence of an iterative computation (fixed-point interpretation) rather than dimensional sharpening
**Priority:** HIGH
**Motivated by:** 2602.16490 + F-2 (AUROC 0.7731 at L19)
**Test:** Compute inter-layer cosine similarity cos(h_l, h_{l+1}) stratified by correct/incorrect on cached per-layer activations. If correct problems show significantly higher layer-to-layer similarity at L17–L21 (indicating convergence), and DoM projection correlates with convergence speed, the fixed-point interpretation is confirmed.
**Requires:** CPU, cached per-layer activations from P11 H100 extraction
**Would change:** Reframes F-2's mechanism from "geometry predicts correctness" to "convergence of iterative computation predicts correctness" — more mechanistic, more actionable for steering (repeat computation rather than steer direction)
**Blocks:** H-1 (if confirmed, steering should repeat layers, not push toward DoM direction)

### H-804: Qwen-2.5-1.5B exhibits emergent 4-layer block periodicity despite having untied weights
**Priority:** MEDIUM
**Motivated by:** 2602.16490 + F-1 (dimensional breathing universality)
**Test:** Autocorrelation of per-layer activation statistics (norms, DoM projections, participation ratio) at lags 1–12 across all 24 layers. Significant peak at lag 4 confirms implicit block structure.
**Requires:** CPU, cached per-layer activations
**Would change:** If confirmed, explains why L19 is special (final aggregation layer of the last full 4-layer block before output layers 21–24). Strengthens H-489.
**Blocks:** nothing

### H-805: Inference-time middle-block looping improves MATH-500 accuracy on Qwen-2.5-1.5B without retraining
**Priority:** HIGH
**Motivated by:** 2602.16490 (up to 2× improvement on reasoning tasks) + F-2 + F-8
**Test:** Repeat layers 16–20 of Qwen-2.5-1.5B 2× at inference on full MATH-500. Compare K=1 accuracy to baseline 48.6%. Threshold: ≥2pp improvement (50.6%+) confirms; <1pp = null.
**Requires:** H100 GPU, ~2h inference time, Qwen-2.5-1.5B weights
**Would change:** If confirmed, provides a training-free intervention that improves accuracy — complements selective prediction (F-8) with compute scaling. If null, confirms L19's role is not iterative refinement.
**Blocks:** H-1 (alternative steering paradigm)

---


### H-806: The L19 DoM direction is effectively identifiable (unique up to sign within a < 10° cone) among all linear directions predicting correctness at AUROC ≥ 0.75
**Priority:** HIGH
**Motivated by:** 2502.20914 + F-2 + cos(DoM, PC1)=0.9216
**Test:** Enumerate all unit vectors achieving AUROC ≥ 0.75 on cached L19 activations via regularization sweep + random initialization + PCA direction scan. Measure angular diameter of the solution set.
**Requires:** CPU only, cached 500×1536 L19 prefill activations
**Would change:** If confirmed (cone < 10°), F-2's "single direction" narrative is validated against identifiability critique. If rejected (cone > 30°), F-2 must be reframed as "a direction from a family of near-equivalent solutions."
**Blocks:** H-1 (steering depends on which direction is chosen if non-identifiable)

### H-807: The prefill/final-token orthogonality (cos=0.046) is a property of the direction families, not an artifact of selecting one pair from non-identified solution sets
**Priority:** HIGH
**Motivated by:** 2502.20914 + F-3
**Test:** Bootstrap-resample DoM regression 100× for both prefill and final-token. Compute all 100×100 cross-family cosine values. If 95th percentile of |cos| < 0.15, orthogonality is a family-level property.
**Requires:** CPU only, cached activations for both prefill and final-token
**Would change:** If confirmed, F-3 is robustified. If rejected, F-3 becomes "one pair happens to be orthogonal" rather than "the modes are structurally orthogonal."
**Blocks:** nothing

---


### H-808: Mean-pooled generation-token hidden states outperform single-position prefill DoM for correctness prediction
**Priority:** HIGH
**Motivated by:** 2605.09969 + F-2 (AUROC 0.7731 from single L19 prefill direction)
**Test:** Extract L19 hidden states at all generation tokens (up to 1024) for MATH-500 on Qwen-2.5-1.5B, mean-pool, train logistic regression for correctness. Compare AUROC to 0.7731.
**Requires:** H100 GPU for forward pass extraction (~4h), CPU for classification
**Would change:** If confirmed: F-2's "single direction suffices" framing becomes "single direction is a cheap approximation of a richer trajectory-distributed signal." Selective-prediction (F-8) stack should switch to generation-mean features. If rejected: validates that single-position prefill is already optimal for binary correctness, and the paper's alignment improvements don't transfer to classification tasks.
**Blocks:** nothing

### H-809: Prefill/final DoM orthogonality (F-3) is a smooth trajectory rotation, not evidence of distinct computational circuits
**Priority:** HIGH
**Motivated by:** 2605.09969 (phase structure in generation) + F-3 (cos = 0.046)
**Test:** Compute DoM at intermediate generation positions (every 50 tokens). If cos(DoM_t, prefill_DoM) decreases monotonically while cos(DoM_t, final_DoM) increases monotonically, the orthogonality is a trajectory-length consequence.
**Requires:** H100 GPU for extraction at multiple generation positions (~3h)
**Would change:** If confirmed: F-3's interpretation weakens from "separate circuits" to "same circuit rotated over generation time." H-13 (head-level attribution) becomes less motivated. If rejected (discrete jump at some token position): strengthens F-3 and suggests a phase-transition in computation mid-generation.
**Blocks:** H-13

### H-810: Convex mixing of prefill and final-token activations yields AUROC above either endpoint
**Priority:** MEDIUM
**Motivated by:** 2605.09969 (simplex interior optima) + F-3 (orthogonality implies complementary information)
**Test:** For α ∈ {0, 0.1, ..., 1}, compute correctness AUROC from α*h_prefill + (1−α)*h_final. If max over interior α exceeds max(0.7731, 0.7186), prefill and final carry extractable complementary signal even without generation tokens.
**Requires:** CPU only, cached activations
**Would change:** If confirmed: motivates a two-position probe as cheap alternative to full generation extraction. If rejected: prefill and final information overlaps despite geometric orthogonality (F-3 is about direction, not about correctness-relevant variance).
**Blocks:** nothing

---


### H-811: Sliced Wasserstein distance between correct/incorrect L19 prefill activation clouds exceeds what a Gaussian distribution with matched mean and covariance would produce
**Priority:** HIGH
**Motivated by:** 2605.08424 + F-10 (PH null) + F-2 (DoM AUROC 0.7731)
**Test:** Compute SW₂(correct, incorrect) on real activations and on 1000 Gaussian draws with matched moments. Report p-value of real SW₂ under null distribution.
**Requires:** CPU, cached NPZ (L19 prefill, 500 samples)
**Would change:** If confirmed: F-10's null is PH-specific, not geometry-general. If rejected: strengthens the "Gaussian is sufficient" interpretation.
**Blocks:** nothing

### H-812: The optimal transport cost between correct and incorrect activation clouds is dominated by the mean-shift component (>90%), implying F-2's single-direction projection is near-optimal
**Priority:** HIGH
**Motivated by:** 2605.08424 (OT cost decomposition) + F-2 (cos(DoM, PC1)=0.9216)
**Test:** Compute W₂(correct, incorrect) and W₂(correct_centered, incorrect_centered). Report shape/total ratio.
**Requires:** CPU, cached NPZ
**Would change:** If confirmed (>90% from mean): validates DoM as sufficient, closes the "is there more?" question. If rejected (<90%): motivates multi-dimensional probe architectures.
**Blocks:** H-811

---


### H-813: L19 prefill DoM direction has specificity ratio < 0.20 (>80% variance shared with task-critical computation)
**Priority:** CRITICAL
**Motivated by:** 2605.05715 + F-2
**Test:** Compute specificity ratio from cached NPZs: project DoM onto mean(incorrect) − mean(correct) direction, measure residual. 10min CPU.
**Requires:** CPU, cached L19 prefill activations (existing NPZs)
**Would change:** On confirm: H-1 (steering) is structurally blocked; F-2 interpretation shifts from "causal direction" to "statistical signature co-occurring with task computation." On reject (ratio > 0.5): H-1 remains viable and paper's result is domain/architecture-specific.
**Blocks:** H-1

### H-814: Fixed linear steering of the L19 DoM direction produces Δ ≈ 0pp on MATH-500 accuracy (TOST-equivalent within ±2.5pp)
**Priority:** HIGH
**Motivated by:** 2605.05715 + F-2 + H-1
**Test:** Contrastive steering at L19 with α ∈ {0.5, 1.0, 1.5, 3.0}, TOST equivalence. 4h H100.
**Requires:** H100, Qwen-2.5-1.5B, MATH-500 500-sample set
**Would change:** On confirm: H-1 formally closed. On reject (Δ > 2.5pp): paper's result is domain-specific, DoM steering viable.
**Blocks:** H-1

### H-815: Learned non-linear adapter (bottleneck MLP on L19 residual stream) achieves Δ > +2pp on MATH-500 accuracy where linear steering fails
**Priority:** MEDIUM
**Motivated by:** 2605.05715 (their MLP achieves +2.8pp, the only non-null intervention) + F-2
**Test:** Train 1536→64→1536 residual adapter on correct/incorrect prefill activations, evaluate downstream accuracy. 6h H100.
**Requires:** H100, training pipeline for adapter
**Would change:** On confirm: establishes that correctness signal IS causally accessible via non-linear intervention (Billa tier-2 regime). On reject: signal may be purely epiphenomenal.
**Blocks:** nothing

---


### H-816: ICA decomposition of L19 prefill activations yields a correctness-predicting independent component with AUROC > 0.78 that is geometrically distinct from DoM (cos < 0.5)
**Priority:** HIGH
**Motivated by:** 2605.07407 + F-2
**Test:** FastICA on cached 500×1536 L19 prefill NPZ; project correct/incorrect labels; per-IC AUROC; cosine with DoM direction. 10min CPU.
**Requires:** CPU, cached L19 prefill NPZ, scikit-learn
**Would change:** If confirmed → F-2 insufficiency (single direction is not ceiling); suggests multi-symbol probe. If rejected → strengthens F-2's single-direction dominance.
**Blocks:** nothing

### H-817: CCA between prefill and final-token L19 activations recovers shared correctness-predictive subspace (top canonical correlation > 0.3) despite raw DoM orthogonality (cos = 0.046)
**Priority:** HIGH
**Motivated by:** 2605.07407 + F-3
**Test:** CCA on two 500×1536 matrices (prefill, final-token); top-10 canonical correlations; project shared variates onto correctness labels. 15min CPU.
**Requires:** CPU, cached prefill + final-token L19 NPZs
**Would change:** If confirmed → F-3 needs qualification: directions are orthogonal but representations share a rotated subspace. If rejected → F-3 is robust under CCA.
**Blocks:** nothing

### H-818: Wasserstein-1 distance between correct/incorrect L19 prefill distributions is dominated by the PC1 (DoM) component (>70% of total W1), confirming F-2's mean-shift dominance
**Priority:** MEDIUM
**Motivated by:** 2605.07407 + F-2
**Test:** W1 per-PC decomposition. If PC1 accounts for >70% of total W1, DoM captures the dominant distributional shift. If not, higher PCs carry non-trivial separability. 10min CPU.
**Requires:** CPU, cached L19 prefill NPZ
**Would change:** If confirmed → strengthens F-2. If rejected → higher PCs may have unexploited signal for selective prediction.
**Blocks:** nothing

---


### H-819: Inter-layer relation matrix SVD features predict correctness better than single-layer DoM
**Priority:** HIGH
**Motivated by:** 2605.07420 + F-2 (AUROC 0.7731) + F-9 (CoE-60 redundancy)
**Test:** Construct L×L relation matrices from multi-layer activations (L17–L21 or full 28-layer), extract singular value spectra as features, train logistic regression. AUROC > 0.7731 confirms; > 0.7928 (cov-spectrum ceiling) establishes inter-layer relations as dominant.
**Requires:** Multi-layer activations (L17–L21 from CoE cache or fresh H100 extraction for full 28 layers), CPU for SVD + LR.
**Would change:** On confirm: F-2's "single direction suffices" needs qualification — inter-layer relational structure carries additional signal. On reject: validates single-layer sufficiency, strengthens F-2.
**Blocks:** nothing

### H-820: D-bucket membership is better predicted by anomalous inter-layer relation spectra than by within-layer DoM
**Priority:** MEDIUM
**Motivated by:** 2605.07420 Theorem 2 (drift → margin degradation) + F-7 (D-bucket collective signature)
**Test:** Compute per-sample inter-layer relation matrices, compare SVD spectra of D-bucket vs non-D-bucket samples. If D-bucket samples show systematically different singular value distributions (e.g., higher effective rank, less dominant first SV), this characterizes F-7 in relational terms.
**Requires:** Multi-layer activations + D-bucket labels from existing MATH-500 pipeline, CPU.
**Would change:** On confirm: F-7's signature is inter-layer relational, not purely within-layer. On reject: F-7 is genuinely a single-layer phenomenon.
**Blocks:** nothing

---


### H-821: L19 prefill DoM direction is partially aligned with the token-identity (embedding) subspace — partialing out the top-k embedding PCs degrades AUROC by >0.03
**Priority:** HIGH
**Motivated by:** 2605.06216 + F-2
**Test:** Compute PCA of Qwen-2.5-1.5B embedding matrix restricted to MATH-500 vocabulary. Project L19 activations onto complement. Refit DoM. Compare AUROC to 0.7731 baseline.
**Requires:** CPU only, cached NPZs + model embedding weights (~200MB download)
**Would change:** If confirmed, F-2's DoM interpretation shifts from "correctness computation" to "partially token-training-quality proxy" — needs factor decomposition. If rejected (AUROC stable), strengthens F-2.
**Blocks:** H-5

### H-822: Per-problem mean token frequency (log-unigram probability) predicts MATH-500 correctness at AUROC >0.60 without any hidden-state access
**Priority:** HIGH
**Motivated by:** 2605.06216 + F-2 + F-7
**Test:** Tokenize MATH-500 with Qwen-2.5-1.5B tokenizer, compute mean log-frequency from WikiText-103 unigram counts, train logistic regression on mean-frequency → correct/incorrect.
**Requires:** CPU only, tokenizer + reference corpus frequency table
**Would change:** If confirmed (>0.60), establishes a shallow-feature baseline that DoM must beat. If the gap between this baseline and DoM is small, the "geometry predicts correctness" story weakens. If rejected (<0.55), token frequency alone is insufficient and the DoM signal is genuinely deeper.
**Blocks:** nothing

### H-823: D-bucket membership is predicted by prompt token rarity above chance (logistic AUROC >0.60)
**Priority:** MEDIUM
**Motivated by:** 2605.06216 + F-7
**Test:** Compute mean log-token-frequency per problem, predict D-bucket membership via logistic regression.
**Requires:** CPU only, tokenizer + reference corpus
**Would change:** If confirmed, F-7's "distinctive collective geometric signature" needs to be re-interpreted as partially driven by vocabulary statistics rather than problem-level difficulty encoding.
**Blocks:** nothing

---


### H-824: Lorem-prefix perturbation (100–300 random Latin tokens) does NOT degrade L19 prefill DoM AUROC by more than 0.02
**Priority:** HIGH
**Motivated by:** 2605.05566 + F-2 + H-553 + H-759
**Test:** Prepend 10 random Lorem prefixes to each MATH-500 problem, extract L19 prefill activations, compute DoM AUROC. Compare to baseline 0.7731.
**Requires:** CPU, Qwen-2.5-1.5B, cached pipeline
**Would change:** Confirm: DoM is a deep geometric signal, not surface-form dependent (strengthens F-2, resolves H-553 toward "not fragile"). Reject: DoM is shallow/surface-sensitive, selective-prediction (F-8) needs prompt normalization.
**Blocks:** H-759

### H-825: D-bucket membership (F-7) has >80% overlap with the zero-advantage set (pass@8=0 problems)
**Priority:** HIGH
**Motivated by:** 2605.05566 + F-7 + H-708
**Test:** Generate 8 rollouts per MATH-500 problem (or use cached if available), identify pass@8=0 set, compare to D-bucket membership labels.
**Requires:** CPU + 8 forward passes per problem (or cached generations)
**Would change:** Confirm: D-bucket is a GRPO training artifact, F-7's "geometric signature" is explained by gradient void. Reject: D-bucket exists independently of training dynamics, suggesting an intrinsic representational difficulty structure.
**Blocks:** nothing

### H-826: LoPE-style perturbation at inference time shifts dimensional breathing (participation ratio) trajectories for identical problems
**Priority:** MEDIUM
**Motivated by:** 2605.05566 + F-5
**Test:** Measure PR trajectories across token positions for 50 problems with and without Lorem prefix. If PR curves differ significantly, AR-mechanics (not just content) drive breathing.
**Requires:** CPU, generation pipeline (not just prefill)
**Would change:** Confirm: F-5's "content-dependent not AR-mechanics" is too strong — need to add "prefix-context AR effects also matter." Reject: F-5 stands as-is.
**Blocks:** nothing

---


### H-827: Per-cluster DoM probes outperform pooled single-direction DoM on MATH-500
**Priority:** HIGH
**Motivated by:** 2605.09129 + F-2
**Test:** Cluster MATH-500 L19 prefill activations (PCA-50, K-means k=2..8), fit per-cluster logistic probes, compare max per-cluster AUROC and best-of-K AUROC to pooled 0.7731
**Requires:** CPU only, cached L19 NPZ, sklearn
**Would change:** Confirm → F-2 needs "multi-mechanism" qualifier; model uses distinct computational strategies for different problem subsets. Reject → F-2 strengthened as genuinely unitary mechanism.
**Blocks:** H-828

### H-828: CoE-60/DoM redundancy (F-9) breaks for specific activation-defined subpopulations
**Priority:** MEDIUM
**Motivated by:** 2605.09129 + F-9
**Test:** Per DCD-style cluster (from H-827), compute CoE-60-only and DoM-only AUROC; test whether any cluster shows CoE-60 > DoM + 0.05
**Requires:** CPU only, cached CoE-60 features + L19 NPZ
**Would change:** Confirm → F-9's redundancy claim is dataset-specific; CoE-60 has complementary value for certain problem types. Reject → F-9's blanket redundancy holds uniformly.
**Blocks:** nothing

---


### H-829: The L19 prefill DoM direction is not unique — multiple orthogonal directions achieve comparable AUROC
**Priority:** HIGH
**Motivated by:** 2605.12671 + F-2
**Test:** Train 20 logistic probes with orthogonality constraints (sequential deflation) on cached L19 1536-d prefill activations; measure AUROC of each and pairwise cosines
**Requires:** CPU, cached NPZs (pathway11_h100)
**Would change:** If confirmed, F-2 must be reframed from "a single direction" to "a family of directions" — would motivate characterizing the correctness subspace dimensionality rather than a single vector
**Blocks:** H-1 (steering along DoM assumes DoM is canonical; if not, steering protocol needs rethinking)

### H-830: Correctness-prediction redundancy between CoE-60 and L19 DoM arises from mechanism plurality, not information equivalence
**Priority:** MEDIUM
**Motivated by:** 2605.12671 + F-9
**Test:** Identify samples where DoM and CoE-60 disagree on correctness prediction; train probe on disagreement residuals; check if union (DoM + CoE-60 ensemble) exceeds either alone
**Requires:** CPU, cached NPZs + CoE-60 features
**Would change:** If confirmed, F-9's "redundant" label becomes "complementary-but-overlapping" — changes architecture for selective prediction (H-8/F-8) to ensemble rather than pick-one
**Blocks:** nothing

---


### H-831: The L19 DoM direction is a Goldstone mode — a symmetry-protected information channel that resists gradient collapse
**Priority:** HIGH
**Motivated by:** 2605.14685 + F-2 + H-783
**Test:** Compute the "protected Jacobian component" (2605.14685 Eq. 7 analogue) at L19 by treating the 1536-d activation space as 768-d complex under a U(1) action. If the protected component's direction aligns with DoM (cos > 0.5), the Goldstone interpretation holds.
**Requires:** CPU only for cached activations; H100 for Jacobian computation
**Would change:** F-2 gets a theoretical mechanism; H-783 (why L19) gets a symmetry-breaking explanation
**Blocks:** nothing

### H-832: Prefill/final-token DoM orthogonality (F-3) is magnitude decay with conserved phase, not directional rotation
**Priority:** MEDIUM
**Motivated by:** 2605.14685 Eq. 11 + F-3
**Test:** Decompose prefill and final-token DoM vectors into radial + angular parts. Check if the angular (phase) component is approximately aligned (cos > 0.3) even though the full vectors have cos = 0.046. If yes, the orthogonality is a Δ-decay artifact.
**Requires:** CPU, cached activations at prefill and final token
**Would change:** F-3 reframed from "two independent signals" to "one signal that decays in magnitude"
**Blocks:** H-1 (steering)

### H-833: Correct/incorrect problem activations occupy different phases of a spontaneously broken symmetry at L19
**Priority:** HIGH
**Motivated by:** 2605.14685 Section 2.2 + F-2 + F-4
**Test:** Compute order parameter c (mean ||h||²) separately for correct and incorrect groups at L19. If they differ significantly (effect size > 0.5 Cohen's d), correctness prediction is a phase-identification problem.
**Requires:** CPU, cached L19 NPZs
**Would change:** F-2 and F-4 gain a unified theoretical mechanism via SSB
**Blocks:** nothing

---


### H-834: L19 DoM direction aligns with a Correlation Trap (MP spectral outlier) in the activation covariance, not the MP bulk
**Priority:** HIGH
**Motivated by:** 2605.12394 + F-2
**Test:** Compute MP fit for L19 activation covariance (Q≈0.326), extract eigenvectors of outliers beyond λ+ + Δ_TW, measure cos(DoM, v_outlier). High cosine confirms alignment.
**Requires:** CPU, cached NPZ activations (500×1536)
**Would change:** Confirms: DoM is a genuinely non-random spectral mode (strengthens F-2). Rejects: DoM is within bulk fluctuations (mechanism question opens).
**Blocks:** H-763, H-766

### H-835: Qwen-2.5-1.5B L19 weight matrices harbor Correlation Traps (entry-shuffle outliers beyond MP edge), marking L19 as spectrally distinctive among all layers
**Priority:** HIGH
**Motivated by:** 2605.12394 + H-711 + H-743
**Test:** Run WeightWatcher v0.7.5.5 shuffled-spectrum diagnostic on all 28 layers. L19 trap count > mean + 2σ across layers confirms.
**Requires:** CPU, Qwen-2.5-1.5B weights (local)
**Would change:** Confirms: L19 is spectrally special in weight space, not just activation space. Interpretation depends on whether traps are harmful or benign (FE974 follow-up). Rejects: L19 is unremarkable — would weaken the "L19 is special" narrative.
**Blocks:** nothing

### H-836: The top-20 cov-spectrum eigenvectors (AUROC 0.7928) are delocalized (IPR ≈ 1/d), not localized trap-like modes
**Priority:** MEDIUM
**Motivated by:** 2605.12394 + F-2
**Test:** Compute IPR = Σ v_i^4 for each of top-20 eigenvectors of L19 covariance. Compare to 1/1536 (delocalized null) vs O(1/k) for small k (localized).
**Requires:** CPU, cached covariance eigenvectors
**Would change:** Confirms (delocalized): signal is distributed and robust, not fragile trap-like concentration. Rejects (localized): cov-spectrum AUROC may be driven by a few coordinate outliers — fragility concern.
**Blocks:** nothing

---


### H-837: Standard transformers exhibit layer-wise attractor dynamics with a phase transition at ~2/3 depth
**Priority:** HIGH
**Motivated by:** 2605.12466 + F-2 (L19 = layer 19 of 28 ≈ 2/3 depth)
**Test:** Compute Jacobian spectral radius per layer from all-layer caches; check if ρ < 1 emerges consistently after L19
**Requires:** CPU, cached all-layer NPZs (500 samples × 29 layers × 1536-dim)
**Would change:** Reframes F-2 as detecting a dynamical phase boundary rather than a semantically meaningful direction
**Blocks:** H-4

### H-838: Equilibrium proximity (norm of residual at L19) is the underlying signal that DoM captures
**Priority:** HIGH
**Motivated by:** 2605.12466 equilibrium internalization + F-2 AUROC 0.7731
**Test:** Correlate per-sample ‖h_{L20} - h_{L19}‖ with DoM projection, and test residual-norm AUROC against DoM AUROC
**Requires:** CPU, cached all-layer NPZs
**Would change:** If confirmed, DoM is a proxy for convergence quality rather than a content-specific "correctness direction"
**Blocks:** nothing

---


### H-839: The L19 DoM direction is unstable under bootstrap resampling of MATH-500
**Priority:** HIGH
**Motivated by:** 2605.15154 + F-2
**Test:** 200-bootstrap logistic probe on cached 500×1536 L19 activations. Measure SD(cos(DoM_b, DoM_full)) and SD(AUROC). Instability threshold: SD(cos) > 0.10 or SD(AUROC) > 0.03.
**Requires:** CPU only, cached NPZ, sklearn + numpy
**Would change:** If confirmed (DoM is unstable): F-2 needs a confidence interval, selective prediction (F-8) needs recalibration, and the "single direction" framing becomes "mean direction ± stability cone." If rejected (DoM is stable): F-2 gains a formal robustness guarantee.
**Blocks:** nothing

### H-840: RoSHAP-ranked features yield a better-calibrated selective-prediction threshold than magnitude-ranked features
**Priority:** MEDIUM
**Motivated by:** 2605.15154 + F-8
**Test:** Compare selective-prediction accuracy@50% using top-k features ranked by RoSHAP vs top-k by absolute coefficient magnitude. k ∈ {1, 5, 20, 50}. Use 5-fold OOF on cached NPZ.
**Requires:** CPU only, cached NPZ, shap + sklearn
**Would change:** If confirmed: F-8's selective prediction could be improved by switching to stability-weighted features. If rejected: current magnitude-based approach is already optimal.
**Blocks:** nothing

---


### H-841: The token-level implicit reward (log π_teacher/π_ref) on MATH-500 is a better correctness predictor than L19 prefill DoM
**Priority:** MEDIUM
**Motivated by:** 2602.12125 + F-2
**Test:** Forward-pass Qwen-2.5-7B (teacher) and Qwen-2.5-1.5B (student=reference) on MATH-500 at 1024-tok, compute per-problem average implicit reward, fit AUROC for correct/incorrect. Compare against DoM AUROC 0.7731.
**Requires:** H100, ~4h for both forward passes. No cached data — need fresh logits.
**Would change:** If implicit-reward AUROC > 0.77, F-2's practical case weakens (logit ratios are cheaper to compute than residual-stream probing). If < 0.77, hidden-state geometry captures something logits miss.
**Blocks:** nothing

### H-842: ExOPD-distilled models develop a different prefill DoM direction than their teachers
**Priority:** HIGH
**Motivated by:** 2602.12125 + H-366 + F-2
**Test:** After ExOPD distillation (λ=1.25) of Qwen-2.5-7B → 1.5B, extract L19 prefill on MATH-500, fit DoM, measure cos(distilled_DoM, original_7B_DoM) and cos(distilled_DoM, original_1.5B_DoM).
**Requires:** H100 distillation (~1-2 days) + extraction (~4h). G-OPD codebase adaptation.
**Would change:** If cos < 0.3 with either original DoM but accuracy improves, DoM is a behavioral correlate (each training regime develops its own), not a fixed mechanistic feature. Directly updates H-366.
**Blocks:** H-366

---


### H-843: Label-weighted moment eigenvectors at L19 surpass unsupervised cov-spectrum AUROC (0.7928)
**Priority:** HIGH
**Motivated by:** 2605.13612 + F-2
**Test:** Compute C^(ℓ) = (1/n) Σ yᵢ zᵢ zᵢᵀ on cached L19 prefill activations, extract top-k eigenvectors, fit 5-fold logistic regression on projections. Compare AUROC to cov-spectrum (0.7928) and full-dim ceiling (0.7847).
**Requires:** CPU only, cached NPZs
**Would change:** On confirm: F-2 reframed — the correctness signal is multi-directional in label-weighted spectral space, and the right operator is C^(ℓ) not Σ. On reject: unsupervised covariance already captures the accessible label signal, validating cos(DoM, PC1) = 0.9216 as evidence of low effective dimension.
**Blocks:** nothing

### H-844: The correctness signal at L19 is shallow (low-degree), not hierarchically compositional — explaining F-9's CoE redundancy
**Priority:** MEDIUM
**Motivated by:** 2605.13612 + F-9
**Test:** Per-layer label-weighted moment leading eigenvectors across layers 1–28 (needs fresh extraction). If the 28-layer ensemble doesn't beat L19 DoM, the correctness signal has low compositional degree and is fully formed by layer 19.
**Requires:** H100 for per-layer extraction; CPU for analysis
**Would change:** On confirm: F-9 is explained theoretically (CoE redundancy follows from shallow signal), and multi-layer probes for correctness are fundamentally limited. On reject: the right multi-layer aggregation (Neural LoFi spectral, not CoE trajectory) exposes non-redundant per-layer features.
**Blocks:** nothing

### H-845: Our n=500 is near the Neural LoFi emergence threshold for the second eigenvector of C^(ℓ) at L19
**Priority:** MEDIUM
**Motivated by:** 2605.13612 + F-2 + H-839
**Test:** Estimate D^eff at L19 from the Gram matrix eigenvalue spectrum, compute the predicted sample-complexity threshold (Equation 18), check if n=500 is above/below threshold for eigenvectors k=1,2,3.
**Requires:** CPU only, cached NPZs
**Would change:** On confirm: explains the AUROC plateau and predicts how much data would be needed to detect additional features. On reject: 500 samples is well above threshold, and the AUROC ceiling is a genuine information ceiling.
**Blocks:** H-839

---


### H-846: DoM AUROC degrades to chance on tasks where model accuracy is ~0% (zero-competence regime)
**Priority:** MEDIUM
**Motivated by:** 2605.08605 + F-2
**Test:** Extract L19 prefill activations on a benchmark where Qwen-2.5-1.5B accuracy is <5% (e.g. hard Sudoku in NL format, or a competitive math benchmark well beyond 1.5B capability). Compute DoM AUROC. If AUROC ≈ 0.50, DoM detects familiarity, not reasoning capacity.
**Requires:** CPU + new forward passes on a zero-competence benchmark (H100 if using larger prompts)
**Would change:** Confirms or rules out the H-5 familiarity interpretation of DoM. If confirmed, F-2's signal is scope-limited to tasks within the model's competence range.
**Blocks:** nothing

---


### H-847: DoM AUROC 0.7731 is a sharp attractor ridge that degrades under isotropic Gaussian perturbation of L19 activations
**Priority:** HIGH
**Motivated by:** 2605.19376 + F-2
**Test:** Add Gaussian noise (σ ∈ {0.01, 0.05, 0.1, 0.5}) to cached L19 prefill activations, retrain DoM probe under 5-fold OOF, measure AUROC degradation curve. Sharp drop (>0.05 AUROC at σ=0.05) → confirm; gradual (<0.01 at σ=0.1) → reject.
**Requires:** CPU, cached L19 NPZs
**Would change:** If confirmed, suggests DoM is fragile / attractor-dependent, motivating multi-direction ensemble probes. If rejected, strengthens F-2's robustness claim.
**Blocks:** nothing

### H-848: A nonlinear (MLP) readout of L19 prefill activations significantly exceeds the linear DoM AUROC ceiling (0.7847)
**Priority:** HIGH
**Motivated by:** 2605.19376 (LPRM, Eq. 16) + F-2
**Test:** Train 2-layer MLP (hidden=128, ReLU, dropout=0.3) on 500×1536 cached L19 prefill activations predicting binary correctness, 5-fold CV. Compare to L2-regularized logistic ceiling 0.7847. Exceeds by >0.02 → confirm; within 0.01 → reject.
**Requires:** CPU, cached L19 NPZs
**Would change:** If confirmed, opens pathway to substantially better correctness prediction from same data. If rejected, confirms that the L19 correctness signal is essentially linear / rank-1.
**Blocks:** nothing

---


### H-849: Probe directions trained on correctness sub-objectives (arithmetic vs. reasoning) remain linearly connected in activation space
**Priority:** MEDIUM
**Motivated by:** 2306.04488 + F-2
**Test:** Stratify MATH-500 into arithmetic-heavy vs. reasoning-heavy subsets (~250 each), train separate DoM directions on each, interpolate, check if AUROC at λ=0.5 exceeds linear interpolation of individual AUROCs (concavity = LMC holds).
**Requires:** CPU only, cached NPZs, problem-type labels (derivable from MATH-500 categories)
**Would change:** If confirmed, motivates decomposing DoM into sub-objective directions. If refuted, single-direction probe is already near-optimal.
**Blocks:** nothing

### H-850: The near-orthogonality of prefill and final-token DoM (cos=0.046) reflects optimization on antagonistic objectives, analogous to Rewarded Soups' failure mode
**Priority:** LOW
**Motivated by:** 2306.04488 (Figure 15: LMC breaks for antagonistic rewards) + F-3
**Test:** Check if interpolated direction d(λ) = λ·DoM_prefill + (1-λ)·DoM_final has strictly lower AUROC than both endpoints for all λ — i.e., the interpolation curve is *convex* (anti-LMC). This would confirm they encode genuinely competing objectives.
**Requires:** CPU only, cached directions
**Would change:** If confirmed, provides theoretical grounding for F-3's orthogonality (they're not just unrelated — they're *antagonistic*). If refuted (concave curve), orthogonality is incidental, not structural.
**Blocks:** H-1

---


### H-851: Width-scaling (best-of-K via DoM selection) improves selective-prediction accuracy beyond single-pass DoM
**Priority:** HIGH
**Motivated by:** 2605.19943 + F-2 + F-8
**Test:** K ∈ {5, 10, 25} forward passes with temperature sampling, L19 DoM extraction each pass, best-DoM@K selection. Compare selective-prediction accuracy at 50% coverage against single-pass baseline (71.6%).
**Requires:** CPU (2h for K=25 × 500 problems), Qwen-2.5-1.5B, MATH-500
**Would change:** On confirm: F-8 updated with width-scaled numbers, DoM validated as practical verifier. On reject: confirms single-pass DoM is adequate, width-scaling adds noise not signal.
**Blocks:** nothing

### H-852: D-bucket problems correspond to "bad basin" latent-space structure that noise perturbation can disrupt
**Priority:** MEDIUM
**Motivated by:** 2605.19943 + F-7
**Test:** Add Gaussian noise to cached L19 prefill activations for D-bucket vs A-bucket problems, measure how DoM AUROC and D-bucket separability degrade as a function of σ. If D-bucket separability degrades faster, D-bucket occupies sharper/narrower basins.
**Requires:** CPU (15min), cached L19 prefill NPZs
**Would change:** On confirm: reframes F-7 as basin-structure claim rather than intrinsic-geometry claim, opens path to noise-based D-bucket escape. On reject: strengthens F-7 as robust geometric property.
**Blocks:** nothing

---


### H-853: Prefill DoM AUROC (0.7731) degrades under diversity-preserving RL (VPO) vs mode-collapsing RL (GRPO)
**Priority:** HIGH
**Motivated by:** 2605.22817 + F-2 + H-707
**Test:** Train Qwen-2.5-1.5B with VPO (drop-in GRPO replacement) on MATH-500 or equivalent math corpus. Extract L19 prefill activations, refit DoM direction, compute 5-fold OOF AUROC. Compare to GRPO baseline 0.7731.
**Requires:** H100 cluster (4×H100 ~48hrs), veRL framework, VPO advantage estimator code
**Would change:** If AUROC drops significantly (< 0.70), F-2 is RL-objective-specific, not architecture-intrinsic. If stable (≥ 0.75), F-2 is robust to training objective — strengthens the finding considerably.
**Blocks:** H-707, H-708

### H-854: Reward-space diversity (pairwise L1 of per-problem reward vectors) of our GRPO-trained Qwen is near-zero, confirming VPO's mode-collapse diagnosis applies to our setup
**Priority:** MEDIUM
**Motivated by:** 2605.22817 + F-2 + F-7
**Test:** Compute VPO-style reward-space diversity metric on K=8 MATH-500 completions from our model. Compare to VPO's reported GRPO diversity values (0.003–0.054 across domains).
**Requires:** CPU only, cached K-sample completions
**Would change:** If diversity is near-zero, confirms our model is mode-collapsed and DoM direction may be an artifact. If diversity is moderate, our model may not be as collapsed as VPO's GRPO baselines.
**Blocks:** nothing

### H-855: DoM-based best-of-K selection outperforms random selection more on GRPO-collapsed pools than on VPO-diverse pools
**Priority:** HIGH
**Motivated by:** 2605.22817 + F-8 + H-851
**Test:** Compare DoM-ranked selection vs random selection on K-sample pools from GRPO vs VPO-trained models. If VPO pools already contain near-optimal candidates at every position, DoM selection adds less value.
**Requires:** VPO checkpoint (blocked on P11-FE1136)
**Would change:** If DoM selection advantage vanishes on VPO pools, selective prediction (F-8) is a GRPO-specific workaround, not a general capability. If advantage persists, DoM captures genuine problem-difficulty signal beyond training-objective effects.
**Blocks:** H-851

---


### H-856: The L19 prefill DoM direction is suboptimal due to shared-pattern contamination
**Priority:** HIGH
**Motivated by:** 2605.21467 + F-2
**Test:** Apply DelTA's entropy-regularized α-reweighting (K=1 refinement) to cached 500×1536 L19 prefill activations with correct/incorrect labels. Extract refined DoM. Compare AUROC (5-fold OOF) against standard DoM 0.7731. Suboptimality confirmed if refined AUROC > 0.78.
**Requires:** CPU only, cached NPZs
**Would change:** If confirmed, F-2's DoM AUROC ceiling needs revision upward and the cov-spectrum probe surplus (0.7928) may be partially explained by its implicit shared-pattern suppression. If rejected, confirms DoM is near-optimal for centroid-based methods and cov-spectrum's surplus is genuinely non-centroid information.
**Blocks:** nothing

### H-857: Prefill/final DoM orthogonality survives discriminative refinement
**Priority:** MEDIUM
**Motivated by:** 2605.21467 + F-3
**Test:** Apply DelTA α-reweighting independently at prefill and final-token positions, compute cos between refined discriminative directions. Compare to raw 0.046. Orthogonality holds if |cos| < 0.3 post-refinement.
**Requires:** CPU only, cached NPZs for both prefill and final-token L19
**Would change:** If orthogonality holds post-refinement, strengthens F-3 (shared patterns are not the explanation). If cos increases substantially (>0.3), F-3's structural orthogonality is weakened — the two circuits may share discriminative structure masked by different shared patterns at each position.
**Blocks:** nothing

---


### H-858: L19 prefill DoM is concentrated in a sparse subset (~15%) of attention heads corresponding to "retrieval heads"
**Priority:** HIGH
**Motivated by:** 2605.16928 + F-2
**Test:** Reshape the cached 1536-d DoM vector into per-head blocks (12 × 128 for Qwen-2.5-1.5B). Compute L2-norm fraction per head. If top 2 heads carry >50% of DoM L2-norm, concentrated. Follow up with per-head-block AUROC decomposition. ~30min CPU.
**Requires:** CPU, cached DoM direction + L19 prefill activations (500 × 1536)
**Would change:** On confirm: F-2 gains a mechanistic explanation (retrieval-head gating). On reject: DoM is a distributed whole-layer property, more interesting geometrically.
**Blocks:** nothing

### H-859: Prefill/final DoM orthogonality (cos=0.046, F-3) arises from a phase transition between retrieval-head-dominated and local-head-dominated processing
**Priority:** MEDIUM
**Motivated by:** 2605.16928 + F-3
**Test:** Decompose both prefill and final DoM vectors into per-head blocks. Compute cosine within each head block. If retrieval-head blocks show positive cosine while local-head blocks show negative/zero, the orthogonality is a mixing artifact. ~20min CPU with cached DoM directions.
**Requires:** CPU, cached prefill and final-token DoM directions
**Would change:** On confirm: F-3's "structural orthogonality" becomes "compositional orthogonality" (different head types dominate at different positions). On reject: orthogonality is genuinely layer-wide and not reducible to head partitioning.
**Blocks:** nothing

---


### H-860: L19 prefill DoM direction is the leading Fourier mode of a co-occurrence kernel, not a learned correctness computation
**Priority:** HIGH
**Motivated by:** 2602.15029 + F-2 (AUROC 0.7731) + cos(DoM, PC1) = 0.9216
**Test:** Compute DFT of L19 activation Gram matrix ordered by DoM projection; measure alignment of DoM with k=1 Fourier mode. If alignment > 0.9, the co-occurrence framework explains F-2. Cross-check: does a Word2Vec-scale model on the same corpus produce a comparable direction?
**Requires:** CPU, cached L19 prefill NPZs (500 × 1536)
**Would change:** If confirmed, F-2's narrative shifts from "model learns correctness signal" to "model inherits correctness-correlated co-occurrence structure." AUROC ceiling becomes a spectral-gap prediction, not a probe-design question.
**Blocks:** H-114 (F-2's AUROC ceiling is probe-bound — moot if ceiling is spectral-gap-bound)

### H-861: F-3's prefill/final DoM orthogonality (cos=0.046) is Fourier harmonic separation, not circuit independence
**Priority:** MEDIUM
**Motivated by:** 2602.15029 + F-3 (cos=0.046)
**Test:** Project prefill and final DoM onto DFT basis of the L19 covariance eigenspectrum. If they localize to different harmonics with < 5% overlap, orthogonality is a Fourier artifact. If they spread across harmonics, it's not.
**Requires:** CPU, cached L19 prefill + final-token NPZs
**Would change:** If confirmed, F-3's interpretation shifts from "two independent circuits" to "one Fourier manifold, two sampling points." Weakens the argument for separate steering vectors.
**Blocks:** nothing

### H-862: Intrinsic dimensionality of the L19 correctness manifold determines the AUROC ceiling via ε² ~ r^{-1/D}
**Priority:** MEDIUM
**Motivated by:** 2602.15029 Proposition 3.3 + F-2 (0.7731) + FE291 (0.7856) + FE881 (0.7928)
**Test:** Rank-controlled ridge probe sweep (r=1..500), fit ε²(r) to extract D. If D=1, F-2 is near-optimal. If D≈2, the 2-feat AUROC 0.7856 is the natural ceiling. If D≈20, the cov-spectrum 0.7928 matches the theory.
**Requires:** CPU, cached L19 prefill NPZs
**Would change:** Gives a principled answer to "how many features should the correctness probe use?" and whether F-2's 0.7731 is improvable.
**Blocks:** nothing

---


### H-863: L19 DoM direction is the linear projection of an underlying fixed-point residual — DoM AUROC 0.7731 is a lower bound on attractor-convergence-based prediction
**Priority:** HIGH
**Motivated by:** 2605.21488 + F-2 (AUROC 0.7731)
**Test:** Extract all 28 layer activations, compute per-layer ‖h_{l+1} − h_l‖ trajectory. Fit logistic regression on the 27-dimensional residual vector. Compare AUROC to 0.7731 (DoM) and 0.7928 (cov-spectrum).
**Requires:** H100, 3h for full 28-layer extraction on 500 MATH-500 problems
**Would change:** If confirmed (residual AUROC > 0.7731): F-2 is reframed from "directional geometry" to "attractor convergence"; DoM is a linear approximation. If rejected (residual AUROC ≤ 0.7731): directional information is strictly richer than convergence information, attractor framing doesn't apply to frozen pretrained transformers.
**Blocks:** H-1 (steering depends on whether the signal is directional or convergence-based)

### H-864: Correct/incorrect MATH-500 problems occupy distinct attractor basins at L19, distinguishable by within-class activation covariance spectral gap
**Priority:** MEDIUM
**Motivated by:** 2605.21488 §4.2 landscape modes + F-7 (D-bucket collective signature)
**Test:** Compute activation covariance separately for correct and incorrect problems. Compare spectral gap (λ₁/λ₂), effective dimensionality, and Frobenius norm between the two classes. Test whether the correct-class covariance has a larger spectral gap (indicating a deeper/narrower attractor basin).
**Requires:** CPU, 20min on cached NPZs
**Would change:** If confirmed: F-7's "collective signature" is attractor basin geometry. If rejected: basin structure isn't linearly readable from single-layer covariance.
**Blocks:** nothing

---


### H-865: Effective latent rank of L19 prefill activations is lower for correct than incorrect MATH-500 problems
**Priority:** HIGH
**Motivated by:** 2511.05963 + F-4 + F-2
**Test:** SVD of cached 500×1536 L19 prefill matrix, split by correctness labels, exponentiated singular-value entropy per class. ~10min CPU.
**Requires:** CPU, cached NPZs (pathway11_h100/prefill_gated_compute/)
**Would change:** Confirm: F-4 asymmetric collapse gets a belief-state interpretation; effective rank becomes a new scalar correctness feature. Reject: collapse is not about representational compactness, must be directional.
**Blocks:** nothing

### H-866: Cross-layer predictability (L19→L20 linear R²) correlates with per-problem correctness
**Priority:** HIGH
**Motivated by:** 2511.05963 + H-803 + F-4
**Test:** Fit linear regression from L19 to L20 cached activations, compute per-problem R² residual, test rank-biserial correlation with correctness. Requires multi-layer cached NPZs. ~30min CPU.
**Requires:** CPU, cached L19 AND L20 activations (check DATA_MANIFEST.md for availability)
**Would change:** Confirm: validates H-803 fixed-point interpretation, provides mechanistic account of F-4. Reject: layer transitions are equally (un)predictable for correct and incorrect — belief-state convergence is not the mechanism.
**Blocks:** H-803

---


### H-867: Spectral-energy-weighted token aggregation lifts DoM AUROC above 0.7731
**Priority:** HIGH
**Motivated by:** 2605.21842 + F-2
**Test:** Compute per-token energy score on L19 prefill activations using PC1 projection, weight DoM aggregation by energy score, measure 5-fold OOF AUROC. Compare against uniform pooling baseline (0.7731).
**Requires:** CPU, cached NPZs (L19 prefill activations, 1024-tok)
**Would change:** If confirmed (AUROC > 0.78), establishes that token-importance weighting is a free lunch for single-direction probes; revises F-2 upward. If rejected (AUROC ≤ 0.7731), confirms uniform pooling is already optimal for the averaged-prefill regime.
**Blocks:** nothing

### H-868: Correct MATH-500 problems have lower effective spectral rank at L19 than incorrect ones
**Priority:** HIGH
**Motivated by:** 2605.21842 + F-2 + cov-spectrum 0.7928
**Test:** Compute per-problem covariance of L19 prefill activations (across token positions within each problem), extract eigenvalues, compute effective rank (90% energy threshold). Compare correct vs incorrect group distributions.
**Requires:** CPU, cached NPZs
**Would change:** If confirmed, provides a mechanistic interpretation of the cov-spectrum probe's success: correctness correlates with spectral energy concentration (coherent structure formation). If rejected, the cov-spectrum signal is about eigenvalue shape, not concentration.
**Blocks:** nothing

---


### H-869: L19 prefill DoM projects incoherently through Qwen's unembedding — it encodes a computational instruction, not an answer direction
**Priority:** HIGH
**Motivated by:** 2604.02608 + F-2
**Test:** Project DoM through LayerNorm + W_U, inspect top-50 tokens. Coherent = answer direction; incoherent = computational instruction. ~15min CPU on cached DoM + model weights.
**Requires:** CPU, cached DoM direction, Qwen-2.5-1.5B model weights
**Would change:** If confirmed, reframes F-2 interpretation: the 0.7731 AUROC reads a process signal, not content. If rejected (coherent projection), DoM is genuinely an answer-aligned direction — strengthening the current F-2 interpretation and diverging from the FV pattern.
**Blocks:** nothing

### H-870: A nonlinear (MLP) probe on L19 prefill activations exceeds the 0.7731 linear DoM AUROC ceiling
**Priority:** HIGH
**Motivated by:** 2604.02608 §5.4 + F-2 + cov-spectrum result (0.7928)
**Test:** 2-layer MLP (1536→1024→2, GELU, dropout 0.1) on cached L19 prefill activations, 5-fold CV. Compare to 0.7731 (linear DoM) and 0.7928 (cov-spectrum). ~30min CPU.
**Requires:** CPU, cached L19 NPZs
**Would change:** If MLP > 0.7928, there's accessible nonlinear signal beyond what cov-spectrum captures — motivates probe architecture search. If MLP ≈ 0.7731, the correctness signal at L19 is genuinely linear.
**Blocks:** nothing

### H-871: DoM steering on Qwen-2.5-1.5B operates modulatorily (Llama/Gemma-style), not representationally (Mistral-style)
**Priority:** MEDIUM
**Motivated by:** 2604.02608 §5.5 dual-mechanism hypothesis + H-1
**Test:** If/when H-1 steering experiments run, apply logit lens before and after DoM injection at L19. If post-steering logit-lens delta is near zero despite behavioral change, Qwen is modulatory; if delta is large, Qwen is representational. Requires H100 forward passes.
**Requires:** H100, MATH-500 prompts, Qwen-2.5-1.5B
**Would change:** Determines which mechanistic family Qwen belongs to, informing steering strategy for H-1 and safety-monitoring implications.
**Blocks:** H-1

---


### H-872: Per-layer Shannon capacity explains the L19 DoM AUROC peak
**Priority:** HIGH
**Motivated by:** 2605.23901 + F-2 (L19 DoM AUROC 0.7731)
**Test:** Compute per-layer SNR and Shannon capacity C_l = B_l·log₂(1 + S_l/N_l) for all 28 layers. If C(L19) is maximal and correlates with DoM AUROC, the information-theoretic explanation holds. FE131050 + FE131051.
**Requires:** CPU, cached L19 NPZs (Tier 1) or H100 extraction for all layers (Tier 2, FE131054)
**Would change:** On confirm: F-2 gains a theoretical explanation and a capacity ceiling. On reject: L19 peak is not about raw SNR — probe is exploiting non-Gaussian structure.
**Blocks:** nothing

### H-873: The cov-spectrum probe's 0.7928 AUROC reflects the Shannon signal subspace boundary
**Priority:** MEDIUM
**Motivated by:** 2605.23901 + F-2 (cov-spectrum AUROC 0.7928, full-1536d ceiling 0.7847)
**Test:** Decompose L19 covariance into signal/structured-noise/unstructured-noise subspaces per the paper's three-source taxonomy. If the signal subspace boundary coincides with the ~20 eigenvalues used by the cov-spectrum probe, Shannon noise decomposition explains the probe design. FE131052.
**Requires:** CPU, cached covariance matrices
**Would change:** On confirm: principled eigenvalue cutoff for the cov-spectrum probe. On reject: eigenvalue selection must use a different criterion.
**Blocks:** nothing

---


### H-874: Correct/incorrect L19 prefill activations exhibit functor-like geometric alignment (low within-group Dirichlet Energy relative to cross-group)
**Priority:** HIGH
**Motivated by:** 2602.01992 + F-2
**Test:** Compute Dirichlet Energy within correct group, within incorrect group, and cross-group on cached L19 prefill activations. Ratio test: DE(within) / DE(cross) < 0.5 indicates functor alignment.
**Requires:** CPU, cached NPZ data
**Would change:** On confirm: F-2's "direction" reframed as emergent category alignment (functor artifact). On reject: DoM remains best interpreted as a linear separator, not a relational mapping.
**Blocks:** nothing

### H-875: The prefill→final-token transformation is well-approximated by additive functor arithmetic (h_final ≈ h_prefill + f), explaining the cos≈0 orthogonality in F-3
**Priority:** HIGH
**Motivated by:** 2602.01992 + F-3
**Test:** Estimate f = mean(h_final − h_prefill) over 500 samples. Compute per-sample reconstruction error ‖h_final − (h_prefill + f)‖ / ‖h_final‖. If mean error < 0.3, functor model holds.
**Requires:** CPU, cached prefill + final-token NPZ data (both exist from P11)
**Would change:** On confirm: F-3's orthogonality explained by functor geometry — reduces from independent finding to consequence of F-2. On reject: prefill→final transformation is not additive — F-3 remains structurally surprising.
**Blocks:** nothing

### H-876: Geometric alignment of correct/incorrect representations shows non-monotonic scaling with model width, matching the paper's prediction that analogical structure degrades at large d_model
**Priority:** MEDIUM
**Motivated by:** 2602.01992 + F-1 + F-6
**Test:** Extract L19 prefill activations from Qwen-2.5-0.5B (d=896), 1.5B (d=1536), and 7B (d=3584). Compute Dirichlet Energy at each scale. Non-monotonic pattern (0.5B > 1.5B < 7B for alignment quality) supports the paper's scaling prediction and qualifies F-1.
**Requires:** H100 for 0.5B and 7B extractions; 1.5B cached
**Would change:** On confirm: F-1 needs "at intermediate scales" qualifier. On reject: breathing universality stands, non-monotonic scaling is synthetic-task-specific.
**Blocks:** H-874

---


### H-877: BM25-style nonlinear scoring of L19 cov-spectrum features beats the 0.7928 linear AUROC ceiling
**Priority:** HIGH
**Motivated by:** 2605.29384 + F-2
**Test:** Apply sqrt-tf × IDF reweighting to top-20 PC projections from FE881, 5-fold logistic regression on reweighted features. 20min CPU.
**Requires:** CPU, cached cov-spectrum features from FE881
**Would change:** If confirmed (AUROC > 0.82), the "ceiling" shifts from 0.7928 to a higher value and the bottleneck is scoring regime, not feature quality. If rejected (AUROC ≈ 0.79), linear scoring is near-optimal for these features.
**Blocks:** nothing

### H-878: Reconstruction-only SAE on L19 prefill activations recovers correctness-predictive features without supervision
**Priority:** HIGH
**Motivated by:** 2605.29384 + H-391 + H-539
**Test:** Train top-K SAE (K=16, 32K latents) on L19 activations with reconstruction loss only (no correctness labels). Post-hoc correlate latent activations with correctness. 2h H100.
**Requires:** H100, L19 prefill activations (cached), unlabeled text activations for training distribution
**Would change:** If confirmed, correctness information is structural (emerges from reconstruction), not supervisory. Reframes F-2 from "a direction we find with labels" to "a direction the model's own geometry exposes." If rejected, supervision is necessary and H-391/H-539 remain the path.
**Blocks:** H-795

---


### H-879: L19 is not uniquely specialized — multi-layer SAE-equivalent sweep reveals distributed math-correctness signal across Qwen-2.5-1.5B
**Priority:** HIGH
**Motivated by:** 2605.28649 + F-2 + F-9
**Test:** Extract prefill activations at all 28 Qwen layers (same 500 MATH-500 problems, 1024-tok regime). Compute per-layer DoM AUROC (5-fold OOF). Compare L19 AUROC (0.7731) against the full layer landscape. Test multi-layer logistic regression combining top-k layers.
**Requires:** H100 for fresh forward passes at all 28 layers (~4h). Analysis on CPU.
**Would change:** On confirm: F-2 becomes "L19 is strongest but not unique" — multi-layer probes could break the 0.7731 ceiling. On reject: strengthens F-2's L19 concentration claim.
**Blocks:** H-70 (which already questions L19 uniqueness from a different angle)

### H-880: Activation-space DoM steering (H-1) encounters the same geometric misalignment as SAE-based weight projection
**Priority:** MEDIUM
**Motivated by:** 2605.28649 + H-1 + H-42
**Test:** Compare activation-space DoM steering (add scaled DoM vector at L19 during generation) against a weight-space alternative (LoRA fine-tune on DoM-high vs DoM-low problems, inject raw task vector at L19 only). If the weight-space version substantially outperforms activation-space steering on MATH-500 accuracy, the geometric misalignment extends to our probing context.
**Requires:** H100 for LoRA fine-tuning + evaluation (~4h GPU).
**Would change:** On confirm: H-1 needs reformulation as weight-space editing rather than activation-space steering. On reject: the activation/weight-space misalignment is specific to SAE projection and doesn't apply to direct activation injection.
**Blocks:** H-1

---


### H-881: Hopfield energy on L19 prefill activations discriminates correct/incorrect above DoM AUROC 0.7731
**Priority:** HIGH
**Motivated by:** 2605.27975 + F-2
**Test:** Compute E(ξ|X;β) for each of 500 MATH-500 samples using cached L19 prefill activations as stored patterns; sweep β; compare AUROC to 0.7731.
**Requires:** CPU, cached NPZs from P11 H100 extraction
**Would change:** If confirmed, F-2's "single linear direction" framing needs revision — correctness signal has non-linear energy-landscape structure. If rejected, confirms that the linear DoM direction captures essentially all the available signal.
**Blocks:** nothing

### H-882: D-bucket problems occupy the high-energy tail of the Hopfield energy distribution on L19 prefill activations
**Priority:** HIGH
**Motivated by:** 2605.27975 Theorem 4.1 + F-7, H-7
**Test:** Compute per-sample Hopfield energy; test whether D-bucket membership is predicted by energy rank (top-k% by energy). Compare to existing density-based methods considered in H-7.
**Requires:** CPU, cached NPZs, D-bucket labels
**Would change:** If confirmed, H-7's density approach gets a principled energy-theoretic foundation. F-7's geometric signature is subsumed by MHN energy landscape. If rejected, the D-bucket signature is not an energy/density phenomenon.
**Blocks:** H-7

---


### H-883: Beta-regression direction localization recovers a direction distinct from DoM on L19 prefill activations
**Priority:** HIGH
**Motivated by:** 2605.29971 + F-2
**Test:** PCA(50) + OLS regression on 500×1536 cached L19 activations with binary correctness as target. Compare cos(β, DoM) — if < 0.95, β carries non-redundant signal. Evaluate β-direction AUROC vs DoM AUROC 0.7731.
**Requires:** CPU only, cached NPZs from P11
**Would change:** If β ≈ DoM (cos > 0.99), validates DoM's optimality. If β ≠ DoM with higher AUROC, motivates replacing DoM with regression-derived direction throughout.
**Blocks:** nothing

### H-884: The L19 DoM direction decomposes into a bias component and an error component with distinct causal roles
**Priority:** HIGH
**Motivated by:** 2605.29971 + F-2 + H-1
**Test:** Construct residual e_i = DoM_proj_i - y_i, localize error direction, test cos(error_dir, DoM). If near-orthogonal, the decomposition is real. Causal test (Tier 2): edit each component independently and measure downstream generation accuracy.
**Requires:** CPU for decomposition; H100 for causal test
**Would change:** On confirm: explains why DoM predicts (bias) but steering might fail (error not deployed). On reject: DoM is a unified signal without separable components.
**Blocks:** H-1

---


### H-885: L19 DoM AUROC 0.7731 is a convergence-rate indicator, not a learned correctness direction
**Priority:** HIGH
**Motivated by:** 2605.26106 (adaptive convergence stopping in LoopMDM) + F-2
**Test:** Compute scalar convergence ratio ||h_19 − h_18||/||h_19|| per problem from cached all-layer activations. AUROC ≥ 0.75 confirms convergence-rate interpretation; < 0.65 rejects it. Also sweep all layers to find best convergence-ratio layer.
**Requires:** CPU, pathway8_layerwise/data/math500/ NPZs, ~10min
**Would change:** If confirmed, F-2's mechanistic interpretation shifts from "directional correctness probe" to "convergence-rate indicator" — simpler, more parsimonious, and potentially architecture-general. If rejected, the directional interpretation gains support as non-trivially geometric.
**Blocks:** nothing

### H-886: F-4's asymmetric collapse is a layer-convergence-speed difference between correct and incorrect problems
**Priority:** MEDIUM
**Motivated by:** 2605.26106 (adaptive stopping shows differential convergence rates) + F-4
**Test:** Plot inter-layer delta norm ||h_l − h_{l-1}|| split by correct/incorrect across all 28 Qwen layers. Correct problems should show smaller deltas starting at mid-layers (L12–L18). Threshold: correct-group mean delta at L19 < 0.8× incorrect-group mean delta.
**Requires:** CPU, pathway8_layerwise/data/math500/ NPZs, ~10min
**Would change:** If confirmed, F-4 is reframed from a geometric observation to a convergence-rate phenomenon. If rejected, the geometric collapse interpretation survives.
**Blocks:** nothing

---


### H-887: Correctness-predictive geometry follows a transient emergence profile with peak before L19
**Priority:** HIGH
**Motivated by:** 2605.27970 + F-2
**Test:** Compute layer-wise RSA (correctness dissimilarity vs. cosine dissimilarity) at all 29 layers using cached `prefill_all_layers` from `pathway8_layerwise/data/math500/`. If RSA peaks at layers 10–16 and is significantly lower at L19, the hypothesis confirms.
**Requires:** CPU, cached NPZs (19GB, already on disk)
**Would change:** On confirm: F-2's interpretation shifts from "L19 is geometrically special" to "L19 captures a post-peak readout signal." On reject (RSA peaks at or near L19): strengthens F-2's geometric interpretation.
**Blocks:** nothing

### H-888: Per-problem RSA contribution is a correctness predictor competitive with DoM AUROC
**Priority:** MEDIUM
**Motivated by:** 2605.27970 + F-2 + F-9
**Test:** Compute leave-one-out RSA at L19. Each problem's RSA contribution = RSA_full − RSA_without_i. Evaluate as AUROC predictor against 0.7731 DoM baseline.
**Requires:** CPU, cached L19 prefill activations
**Would change:** On confirm (AUROC ≥ 0.75): DoM is a simplified projection of richer relational structure. On reject: single-direction probe is genuinely the right abstraction.
**Blocks:** nothing

---


### H-889: The L19 DoM direction is a bipolar feature axis where sign discriminates correct (positive) from incorrect (negative) activations
**Priority:** HIGH
**Motivated by:** 2605.28149 + F-2
**Test:** Project 500 MATH-500 L19 prefill activations onto DoM; compute correlation between sign(projection) and correctness label; test signed-threshold classifier AUROC vs unsigned DoM AUROC 0.7731 (P11-FE1202)
**Requires:** CPU only, cached NPZs
**Would change:** Confirm: F-2 reframed from "single direction" to "bipolar feature axis in superposition"; motivates SA-GSAE over standard SAE for H-391/H-539/H-795. Reject: DoM is genuinely one-sided, SA-GSAE offers no advantage for correctness probing.
**Blocks:** H-795

### H-890: The near-orthogonality of prefill and final-token DoM (cos=0.046) is a sign-cancellation artifact from bipolar features that flip polarity between computation stages
**Priority:** HIGH
**Motivated by:** 2605.28149 + F-3
**Test:** Decompose DoM projections by sign at prefill and final-token; compute within-sign-group cosine. If within-sign cosine >> 0.046, orthogonality is sign-cancellation (P11-FE1203)
**Requires:** CPU only, cached NPZs (L19 prefill + final-token)
**Would change:** Confirm: F-3 needs revision — "different mechanisms" becomes "same mechanism, flipped sign." Reject: orthogonality is genuine, supports distinct prefill vs generation circuits.
**Blocks:** nothing

### H-891: SA-GSAE at half-width (8192 latents) on L19 prefill achieves correctness-probe AUROC ≥ 0.7731 while Gated SAE at full-width (16384) does not, demonstrating bipolar efficiency for correctness features
**Priority:** HIGH
**Motivated by:** 2605.28149 + H-795 + H-539
**Test:** Train SA-GSAE (half-width 8192, L₀=64) and Gated SAE (full-width 16384, L₀=64) on Qwen-2.5-1.5B L19 activations; compare top-latent correctness AUROC and dead fraction (P11-FE1205)
**Requires:** H100, ~4h training per SAE variant, math-adjacent pretraining data
**Would change:** Confirm: SA-GSAE is the right SAE architecture for our pipeline; half-width sufficient. Reject: bipolar structure doesn't help correctness probing; stick with standard Gated SAE.
**Blocks:** H-391, H-539

---


### H-892: The L19 prefill DoM correctness signal is dominated by "stiff-mode" (high-eigenvalue) principal components, indicating it captures optimization structure rather than generalization geometry
**Priority:** HIGH
**Motivated by:** 2605.31175 + F-2
**Test:** Spectral elbow partitioning of PCA eigenvalues → separate correctness probes on stiff vs flat PCs. If stiff AUROC > 0.70 and flat AUROC < 0.60, the optimization-structure explanation dominates.
**Requires:** CPU, cached L19 prefill activations, cached PCA eigenvalues
**Would change:** If confirmed, F-2's interpretation shifts from "model encodes correctness in a geometric direction" to "dominant variance captures optimization rigidity, and correctness correlates with input difficulty." If rejected (flat modes carry signal), F-2's geometric interpretation strengthens.
**Blocks:** nothing

### H-893: Per-layer spectral concentration (cumulative energy at elbow-k) is uniform across Qwen-2.5-1.5B layers, and L19's spectral gap is not an outlier
**Priority:** MEDIUM
**Motivated by:** 2605.31175 + H-743
**Test:** Compute spectral elbow for each of 28 layers. If L19's gap ratio is within 1σ of the mean, H-743 is weakened.
**Requires:** CPU, per-layer cached activations (L19 available; others may need lightweight extraction)
**Would change:** If confirmed, need an alternative explanation for why L19 is special (not spectral gap). If rejected, H-743 holds and DiReCT's universal-concentration assumption doesn't apply to activation space.
**Blocks:** H-743

---


### H-894: The cov-spectrum probe's AUROC advantage (0.7928 vs DoM 0.7731) is explained by implicit multi-hyperplane encoding
**Priority:** MEDIUM
**Motivated by:** 2605.28501 + F-2 + FE881 (cov-spectrum result)
**Test:** Run P11-FE1210 (BIC sweep). If K_BIC > 1 and multi-hyperplane AUROC ≈ 0.79, the cov-spectrum advantage is explained by eigenvalue features capturing projections onto multiple hyperplane normals. If K_BIC = 1, reject this hypothesis.
**Requires:** CPU, cached L19 activations, PCA eigenvectors
**Would change:** Interpretation of F-2 (single vs multi-direction); mechanistic understanding of cov-spectrum probe
**Blocks:** nothing

---


### H-895: TrOPD trust-region filtering selectively erases the L19 DoM direction during distillation
**Priority:** HIGH
**Motivated by:** 2606.01249 + F-2 + H-366
**Test:** TrOPD-distill 7B→1.5B on MATH-500 prompts (200 steps, TrOPD FKL recipe from paper), re-extract L19 prefill activations, fit DoM, compute cos(new_DoM, original_DoM) and ΔAUROC vs 0.7731 baseline. Compare against vanilla OPD-distilled checkpoint as control.
**Requires:** H100 SXM, ~1.5 days (distillation + re-extraction). TrOPD code from GitHub. Cached original L19 prefill NPZs from pathway11_h100.
**Would change:** Confirm → F-2 is fragile to training recipe, DoM is checkpoint-specific. Reject → DoM direction is robust to trust-region-filtered distillation, strengthening F-2.
**Blocks:** H-366 (provides the TrOPD-variant distillation checkpoint)

### H-896: Cross-model activation agreement (cos(h_7B, h_1.5B) at L19 prefill) predicts 1.5B correctness independently of DoM
**Priority:** HIGH
**Motivated by:** 2606.01249 (trust-region = probability agreement predicts supervision quality) + F-2
**Test:** Compute per-problem cos(h_7B_L19_prefill, h_1.5B_L19_prefill) from cached NPZs. Fit 2-feature logistic (DoM + cross-model cos) on 1.5B correctness labels. If cross-model cos adds ≥0.01 AUROC over DoM alone, the trust-region concept has geometric substance.
**Requires:** CPU only, 20min. Cached 7B and 1.5B prefill NPZs from pathway11_h100.
**Would change:** Confirm → cross-model agreement encodes correctness information not captured by single-model DoM, motivating multi-model probe features. Reject → DoM already captures the relevant geometric structure, trust-region concept doesn't add signal at activation level.
**Blocks:** nothing

---


### H-897: The per-layer spectral radius of the residual-stream Jacobian in Qwen-2.5-1.5B follows a layer-dependent profile that correlates with the participation-ratio breathing curve (F-1), and this profile differs for correct vs. incorrect MATH-500 problems
**Priority:** MEDIUM
**Motivated by:** 2606.01495 + F-1 + F-2
**Test:** Compute ∂h_l/∂h_{l-1} Jacobian spectral radius at layers 1–28 for cached activations (or approximate via finite differences on cached layer-wise NPZs if available); correlate with PR curve. Est. 1–2h CPU if layer-wise activations cached, needs H100 forward pass if not.
**Requires:** Layer-wise cached activations (L1–L28 prefill, 500 samples). May need fresh extraction.
**Would change:** On confirm: F-1 reframed from "breathing" to "learned state-retention dynamics," connects to CART's LTI gate. On reject: breathing is genuinely a geometric (not optimization-dynamic) phenomenon.
**Blocks:** nothing

### H-898: The L19 DoM correctness signal measures "distance to fixed point" — problems with high DoM have representations that change little from L19 to L28, while low-DoM problems are still evolving
**Priority:** MEDIUM
**Motivated by:** 2606.01495 + F-2 + F-3
**Test:** Compute ||h_L19 - h_L28|| cosine distance for 500 MATH-500 samples, correlate with DoM and with correctness. 20min CPU on cached data.
**Requires:** L19 and L28 (or final-layer) cached prefill activations.
**Would change:** On confirm: provides a dynamical-systems interpretation of DoM, connects F-2 and F-3 to the fixed-point convergence framework. On reject: DoM is not about convergence rate, preserving its current "direction of maximum variance" interpretation.
**Blocks:** nothing

---


### H-899: Qwen-2.5-1.5B L19 latent space is Class 2 (ε_latent < 0.1) despite Class 1 embeddings (ε = 0.26)
**Priority:** HIGH
**Motivated by:** 2606.02765 + F-2, F-3
**Test:** Compute all-pairs cosine similarity of 500 L19 prefill activation vectors (cached NPZ), fit μ + 2σ = ε_latent. Class 2 if ε_latent < 0.1.
**Requires:** CPU, cached L19 prefill activations (500 × 1536)
**Would change:** Confirm: LRH interpretation of DoM is geometrically grounded at L19 even though embeddings are unstructured. Reject: L19 is also Class 1, weakening LRH-based interpretation of F-2 and requiring reframing of DoM as something other than a feature direction.
**Blocks:** H-693

### H-900: The Class 1 → Class 2 boundary (if it exists) in Qwen-2.5-1.5B's residual stream occurs at or before L19
**Priority:** HIGH
**Motivated by:** 2606.02765 + F-2, H-899
**Test:** Compute ε_latent per layer (0-27) from cached per-layer activations. Find the layer at which ε_latent drops below 0.1 (if it does).
**Requires:** CPU, per-layer cached activations (may need fresh extraction if only L19 is cached)
**Would change:** Confirm: identifies the depth at which structured feature representation emerges, potentially explaining why L19 probing works. Reject: no transition occurs, model never enters Class 2, LRH interpretation weakened across all layers.
**Blocks:** nothing

### H-901: The 1.5B/7B DoM AUROC inversion (F-6) is explained by the Class 1/Class 2 boundary
**Priority:** MEDIUM
**Motivated by:** 2606.02765 + F-6
**Test:** Compare ε_latent at L19 for 1.5B and 7B. If 1.5B is Class 1 and 7B is Class 2 at the latent level (mirroring embeddings), the AUROC inversion may be a geometric-regime effect: more features competing for inner-product channel at 7B.
**Requires:** CPU, cached L19 activations for both model scales
**Would change:** Confirm: mechanistic explanation for F-6 via representational capacity framework. Reject: both models in same class at L19, explanation must be elsewhere.
**Blocks:** nothing

---


### H-902: Text-level reasoning redundancy predicts correctness comparably to L19 prefill DoM
**Priority:** HIGH
**Motivated by:** 2606.03503 + F-2, F-8
**Test:** Compute n-gram repetition rate and/or consecutive-step semantic similarity from cached MATH-500 CoTs. Evaluate AUROC for correctness prediction; compare to DoM 0.7731. 30min CPU.
**Requires:** CPU, cached CoT text outputs
**Would change:** If confirmed (redundancy AUROC ≥ 0.75): DoM may be capturing anticipated redundancy rather than independent geometry. If rejected: DoM captures something beyond text-level redundancy patterns.
**Blocks:** nothing

### H-903: Mahalanobis distance from activation centroid outperforms Euclidean DoM projection for correctness prediction
**Priority:** HIGH
**Motivated by:** 2606.03503 Appendix A + F-2
**Test:** Compute Mahalanobis distance of each problem's L19 prefill activation from global mean, using full covariance. AUROC vs DoM (0.7731) and cov-spectrum (0.7928). 15min CPU.
**Requires:** CPU, cached L19 prefill activations + PCA covariance
**Would change:** If confirmed: motivates covariance-aware reformulation of correctness prediction, potentially closing the gap to the 0.7928 ceiling. If rejected: Euclidean projection onto DoM is already near-optimal given the covariance structure.
**Blocks:** nothing

---


### H-904: Text-level reasoning-graph efficiency η is correlated with L19 prefill DoM projection
**Priority:** HIGH
**Motivated by:** 2606.03883 + F-2
**Test:** Extract reasoning graphs from MATH-500 CoT traces using Berdoz pipeline, compute η, correlate with DoM. Partial correlation analysis: does η predict correctness beyond DoM?
**Requires:** CPU + LLM API for graph extraction (~$10). Cached L19 prefill activations.
**Would change:** If η ⊥ DoM: ceiling-breaker identified — combined η+DoM probe should exceed 0.7731. If η ≈ DoM: confirms DoM already captures reasoning structure, strengthening F-2.
**Blocks:** nothing

### H-905: D-bucket problems have distinctively low reasoning-flow efficiency η
**Priority:** MEDIUM
**Motivated by:** 2606.03883 + F-7
**Test:** Extract reasoning graphs from D-bucket MATH-500 problems, compute η, compare distribution against non-D-bucket failures and successes.
**Requires:** CPU + LLM API. Cached D-bucket labels.
**Would change:** If confirmed: F-7's geometric signature is downstream of reasoning structure. If rejected: geometry and reasoning structure are independent signals.
**Blocks:** nothing

---


### H-906: Third-order co-skewness of L19 prefill activations carries correctness-predictive information beyond what κ₂ (covariance) captures

**Priority:** HIGH
**Motivated by:** 2606.04010 + F-10 + F-2
**Test:** Tucker/HOSVD on L19 co-skewness tensor → compare AUROC of κ₂ features in Tucker basis vs PCA basis at matched rank R (P11-FE1231)
**Requires:** CPU, cached L19 NPZs, tensorly library
**Would change:** On confirm: F-10's "Gaussian null" weakened; new AUROC ceiling above 0.7928. On reject: F-10 strengthened — residual streams are genuinely Gaussian at κ₃.
**Blocks:** nothing

### H-907: The cov-spectrum probe's 0.7928 AUROC is not the linear ceiling — a κ₃-informed basis for the covariance features lifts it further

**Priority:** HIGH
**Motivated by:** 2606.04010 + F-2 (AUROC 0.7731) + FE881 (0.7928)
**Test:** Replace PCA eigenvectors with Tucker factor matrix in cov-spectrum pipeline. If Tucker-cov-spectrum AUROC > 0.7928, the current ceiling is basis-limited (P11-FE1231)
**Requires:** CPU, cached NPZs
**Would change:** On confirm: linear ceiling is higher than reported; motivates hybrid κ₂/κ₃ probes. On reject: PCA basis is near-optimal; κ₃ structure not informative for correctness.
**Blocks:** nothing

---


### H-908: SDPG-trained Qwen-2.5-1.5B has a different L19 prefill DoM direction than GRPO-trained (cos < 0.5)
**Priority:** HIGH
**Motivated by:** 2606.04036 + F-2 + H-707 + H-774
**Test:** Train Qwen-2.5-1.5B with SDPG using paper recipe (adapted from Qwen3-4B settings: α=1e-3, β_base=1e-3, T_warm=50, T_decay=350, DAPO-Math-17k). Extract L19 prefill activations on MATH-500 at 1024 tok. Fit DoM direction. Compute cos(DoM_SDPG, DoM_GRPO) and AUROC_SDPG. 2-3 days H100.
**Requires:** H100 cluster, verl framework, SDPG code (https://github.com/lauyikfung/SDPG), DAPO-Math-17k
**Would change:** On confirm (cos < 0.5): F-2 is RL-objective-specific, DoM direction is a training artifact, qualifying F-8 selective prediction. On reject (cos > 0.8): F-2 is robust across GRPO, MaxRL, DGPO, and SDPG — strong evidence for architecture-intrinsic correctness geometry.
**Blocks:** H-707

### H-909: SDPG's entropy-preserving training eliminates F-4's asymmetric collapse signature
**Priority:** MEDIUM
**Motivated by:** 2606.04036 Figure 3e (entropy dynamics) + F-4 + H-853
**Test:** On SDPG-trained checkpoint from H-908, extract per-token L19 participation ratios for correct vs incorrect MATH-500 trajectories. Compare the correct/incorrect PR gap to GRPO-trained baseline. Predict: PR gap < 50% of GRPO-trained gap. 4h H100 (extraction only, checkpoint from H-908).
**Requires:** SDPG-trained checkpoint (from H-908), existing PR extraction pipeline
**Would change:** On confirm: F-4 is a GRPO mode-collapse artifact, not an intrinsic feature of correct reasoning. On reject: asymmetric collapse is a robust feature of correct reasoning geometry, independent of training objective.
**Blocks:** nothing

---


### H-910: The L19 DoM direction is geometrically aligned with the top training-loss Hessian eigenvector (v1) — i.e., it is a training-time EoS directional imprint rather than an emergent inference-time confidence computation
**Priority:** MEDIUM
**Motivated by:** 2606.04212 + F-2
**Test:** Compare DoM directions across models pretrained with different learning rates or optimizers (different EoS regimes). If DoM rotates with EoS regime, it is training-shaped; if stable, it is an intrinsic model property. Requires access to multiple pretraining runs of similar-architecture models (e.g. Qwen-2.5-1.5B vs another 1.5B model with different pretraining).
**Requires:** Two or more 1.5B models with known different pretraining hyperparameters, ~2h H100 for extraction
**Would change:** On confirm: F-2 is reframed from "model confidence signal" to "training geometry imprint" — still useful for prediction but changes interpretation. On reject: DoM is robust to training regime, strengthening the inference-time confidence interpretation.
**Blocks:** nothing

### H-911: Centroid distance in L19 prefill activation space predicts per-example DoM projection magnitude (Spearman ρ ≥ 0.2), mirroring the EoS centroid-distance → v1-alignment correlation
**Priority:** HIGH
**Motivated by:** 2606.04212 (Figure 8, ρ=0.39) + F-2 + F-7
**Test:** P11-FE1238 — compute centroid distance and DoM projection from cached NPZs, compute Spearman ρ. 10min CPU.
**Requires:** CPU only, cached L19 prefill activations
**Would change:** On confirm (ρ ≥ 0.2): DoM signal has a training-geometry component, D-bucket may be geometric outliers. On reject (ρ < 0.1): inference-time DoM is not simply reflecting input-space typicality, strengthening the confidence interpretation.
**Blocks:** nothing

---


### H-912: The L19 prefill DoM direction is an artifact of SFT→RL distribution sharpening, not an intrinsic property of the model
**Priority:** HIGH
**Motivated by:** 2606.04272 + F-2
**Test:** Extract L19 prefill activations from Qwen-2.5-1.5B base checkpoint on MATH-500 (8-shot prompted), fit DoM, compare AUROC and direction cosine against instruct-model DoM (AUROC 0.7731). Also compare PC1 explained-variance ratio between base and instruct.
**Requires:** Qwen-2.5-1.5B base weights (HF), CPU or GPU for extraction, MATH-500 prompts
**Would change:** Confirm → F-2 is pipeline-specific, DoM is a training artifact, generality claims weaken significantly. Reject → F-2 is robust to training pipeline, DoM reflects pre-training geometry.
**Blocks:** H-8

### H-913: Distribution expansion (direct RL) produces higher-dimensional residual-stream geometry than distribution sharpening (SFT→RL), making single-direction DoM probes less effective
**Priority:** MEDIUM
**Motivated by:** 2606.04272 §4.1 + F-2 + cos(DoM,PC1)=0.9216
**Test:** On a direct-RL-trained model (e.g. OLMo2-1B-RL from Bansal et al. if released), extract L19 prefill residuals on MATH-500, compute PC1 explained-variance ratio and DoM AUROC. Compare both against Qwen instruct model.
**Requires:** Access to a direct-RL-only trained model (may need to train or wait for model release), H100 for extraction
**Would change:** Confirm → DoM-based selective prediction only works on sharpened models. Reject → DoM is robust to training pipeline, expansion/sharpening is an output-level phenomenon not reflected in residual-stream geometry.
**Blocks:** nothing

---


### H-914: REDI-style asymmetric negative-trace refinement reshapes residual-stream correctness separability, demonstrating DoM is recipe-dependent
**Priority:** MEDIUM
**Motivated by:** 2505.24850 + H-912 + F-2
**Test:** Re-extract L19 prefill activations on MATH-500 from a checkpoint before vs after REDI Stage-2 refinement (α≈0.8); recompute OOF DoM AUROC and final-token-DoM AUROC. Δ AUROC > 0.05 (or a prefill↔final crossover) confirms recipe-dependence. ~1 H100 day for refinement + minutes for the probe.
**Requires:** H100 for REDI refinement; 1.5B SFT checkpoint + Open-R1 DPref negatives; existing DoM probe.
**Would change:** Confirm → F-2's "intrinsic single direction" reading downgrades to "SFT→RL sharpening artifact" (H-912); reject → strengthens F-2 as recipe-invariant. NOTE: relies on the REFUTED geometry-portable-signal premise, so any FE form is born MOOTED.
**Blocks:** H-912, H-913

### H-915: Length-normalized sequence logprob (SimPO/REDI form) is a stronger free-baseline correctness readout than raw sequence logprob
**Priority:** MEDIUM
**Motivated by:** 2505.24850 + F-8 + free-baseline-strongest-readout
**Test:** On cached MATH-500 features compute logπ(y|x)/|y| and compare OOF AUROC against raw logprob and length alone; if the normalization lifts AUROC out-of-family it becomes the new free-baseline form. ~15min CPU (P11-FE1246).
**Requires:** CPU; cached per-example sequence logprob + token length.
**Would change:** Confirm → updates the canonical free-baseline definition used across the selective stack (F-8); reject → confirms raw length+logprob is already the right free readout.
**Blocks:** nothing

---


### H-916: The L19 prefill correctness axis (PC1≈DoM) is substantially a subject/domain-affinity axis
**Priority:** HIGH
**Motivated by:** 2508.09883 (PCA-shift latent analysis) + F-2 + FE291
**Test:** Subject-stratify MATH-500 (Hendrycks `subject`) and recompute prefill-DoM/PC1 AUROC within
each subject on the cached 500×1536 L19 cloud (P11-FE1249). Compare to the 0.7731 pooled number and to
the per-subject base accuracy.
**Requires:** CPU, cached NPZ + MATH-500 subject labels. ~20min.
**Would change:** Confirm ⇒ F-2's pooled AUROC is partly a domain artifact; reframe DoM as a
familiarity/affinity axis (supports H-5) and discount cross-domain claims. Reject (AUROC holds
within-subject) ⇒ strengthens F-2 as a genuine correctness direction.
**Blocks:** nothing

### H-917: Final-token "asymmetric collapse" (F-4) is mediated by per-example token entropy
**Priority:** MEDIUM
**Motivated by:** 2508.09883 (token-entropy analysis) + F-4 + EXP free-baseline work
**Test:** Partial out mean token entropy (−mean logprob) from the correct-vs-incorrect final-token
collapse gap on cached data (P11-FE1250); test whether the gap survives.
**Requires:** CPU, cached length+logprob+collapse features. ~15min.
**Would change:** Confirm (gap vanishes) ⇒ F-4 collapse is an entropy proxy, folds into the
free-baseline readout. Reject (gap survives) ⇒ collapse carries entropy-independent geometric signal.
**Blocks:** nothing

---


### H-918: A black-box answer-entropy trajectory-shape gate matches the internal prefill-DoM stack at 50% coverage on MATH-500
**Priority:** HIGH
**Motivated by:** 2603.18940 + F-8 (71.6% @ 50% coverage) / F-2 (AUROC 0.7731)
**Test:** Replicate the m=5/τ=0.7 per-step answer-distribution entropy trajectory on Qwen-2.5-7B MATH-500, score ε-monotonicity, and overlay its accuracy-vs-coverage curve against the F-8 prefill-DoM stack at matched coverage. ~1 H100 day.
**Requires:** H100, Qwen-2.5-7B-Instruct, fresh generations (no cache suffices).
**Would change:** Confirm → the internal prefill signal is *not necessary* for F-8's headline (a free black-box gate suffices), reframing F-8 as "internal-equals-black-box." Reject → internal signal retains a coverage advantage, strengthening F-8.
**Blocks:** H-919

### H-919: Trajectory SHAPE, not magnitude, is the load-bearing axis for correctness signals in this program
**Priority:** MEDIUM
**Motivated by:** 2603.18940 (scalar coherence −0.6 pp vs monotonicity +5.8 pp) + F-9 / H-9 / F-7
**Test:** Across our trajectory features (breathing PR curve, CoE-60 norms, per-token entropy), compare each feature's *magnitude* summary vs its *monotonicity/violation-count* shape summary as a correctness predictor on cached MATH-500; CPU-only where black-box features exist.
**Would change:** Confirm → magnitude-based findings (H-9 amplitude, cov-spectrum framing) are reformulated as shape phenomena; F-9's "trajectory redundant" verdict is scoped to magnitude only. Reject → magnitude remains the right summary and the paper's shape result is task-specific to answer entropy.
**Blocks:** nothing

---


### H-920: The prefill correctness direction is a property of the task manifold, not the model — a foreign encoder predicts a target's correctness as well as the target's own states
**Priority:** HIGH
**Motivated by:** 2603.20895 + F-2 / EXP-037
**Test:** Encoder-Target Decoupling on cached prefill (P11-FE1256): train an L2-LR probe on Qwen-1.5B L19 prefill, score Qwen-7B MATH-500 correctness labels, and the reverse; compare cross-model AUROC to within-model 0.7731/0.7186 and to the free length+logprob baseline. ~1hr CPU, no new forward passes.
**Requires:** CPU; cached 1.5B + 7B prefill states (`prefill_gated_compute/results.json`).
**Would change:** Confirm → F-2's single direction is a within-model shadow of a shared difficulty manifold, and the REFUTED `geometry-portable-signal` premise must be re-opened. Reject → reinforces that the signal is model-idiosyncratic and the EDGEGEN refutation generalizes.
**Blocks:** nothing

### H-921: Label-aware Fisher J, not label-free d_eff/anisotropy, picks the correctness-informative layer — and J's argmax coincides with our hand-chosen L19
**Priority:** MEDIUM
**Motivated by:** 2603.20895 + F-2 / F-10
**Test:** Compute Fisher J, d_eff, and anisotropy α across all cached prefill layers (P11-FE1257/20897); check whether J peaks at L19 while d_eff/α peak elsewhere, replicating the paper's "label-free peaks miss separability" finding. ~30min CPU.
**Requires:** CPU; cached per-layer prefill (or at least L19) activations + cov-spectrum NPZ (FE881).
**Would change:** Confirm → validates L19 on a principled supervised criterion and corroborates F-10 (unsupervised geometry summaries are weak). Reject (J picks a different layer) → our L19 choice is suboptimal and a J-driven re-pick could lift F-2.
**Blocks:** nothing

---


### H-922: The 7B-vs-1.5B correctness-geometry gap is a capacity/utility-tail effect, not a verification-strength effect
**Priority:** HIGH
**Motivated by:** 2605.29548 + F-6 + F-7
**Test:** Compute participation ratio / effective rank and the Stiefel-normalized signal-capture curve for L19 prefill covariance, 1.5B vs 7B, split by correctness (P11-FE1261, FE130604). Check whether the 7B carries a strictly fatter low-utility eigen-tail and whether D-bucket problems map to those low-utility directions.
**Requires:** CPU; cached `m7b_prefill.npz` + 1.5B prefill cache + `cov_spectrum/eigval_cache.npz`.
**Would change:** Confirm → F-6's "incorrect-group concentration" mechanism is replaced (or subsumed) by a capacity-allocation mechanism, and F-7's D-bucket gets a generative story (low-utility eigendirections the 1.5B cannot embed). Reject (spectra same shape) → F-6/F-7 mechanisms stand and the capacity account is ruled out at inference.
**Blocks:** nothing

### H-923: PC1/DoM encodes the high-utility common-task structure and systematically misses the rare/hard tail
**Priority:** HIGH
**Motivated by:** 2605.29548 (Theorem 3) + F-2 + F-9 + FE881
**Test:** Per-difficulty-stratum AUROC of PC1-only DoM vs PC1-residualized cov-spectrum probe on MATH-500 (P11-FE1262), joining HF MATH level labels. Predict PC1 AUROC collapses on the hardest stratum while residual directions recover it.
**Requires:** CPU; cached PC scores (FE291/FE881) + MATH-500 level join.
**Would change:** Confirm → F-9's "CoE/extra directions redundant" claim is bounded to the easy bulk; the hard tail needs low-utility directions, reframing the selective-prediction ceiling (F-8). Reject (PC1 uniform across strata) → F-2/F-9 sufficiency strengthened, utility-ranking prediction fails at inference.
**Blocks:** nothing

---


### H-924: A normalizing-flow log-likelihood-ratio on L19 prefill activations beats the 0.7928 linear cov-spectrum ceiling
**Priority:** HIGH
**Motivated by:** 2606.06447 + F-2, FE881 (cov-spectrum 0.7928), FE882 (full-1536 0.7847)
**Test:** Fit class-conditional MAF/RealNVP (nflows/normflows) on cached 500×1536 L19 prefill activations; 5-fold OOF; AUROC of per-example LLR. Est ~2h CPU.
**Requires:** CPU only; cached `pathway11_h100` activations + labels from `prefill_gated_compute/results.json`.
**Would change:** Confirm (>0.793 outside CI) → F-2's "signal is essentially one linear direction" is refuted; correctness signal is nonlinear/multimodal. Reject (≤0.793) → the linear ceiling is reinforced as a genuine ~1-D bound, strengthening F-2 and F-8.
**Blocks:** H-925, H-926.

### H-925: L19 prefill activations carry correctness-relevant non-Gaussian structure that a flow exploits beyond a single Gaussian (Mahalanobis)
**Priority:** HIGH
**Motivated by:** 2606.06447 + F-10, H-903, H-906
**Test:** Held-out NLL gap (flow vs single multivariate Gaussian) and AUROC gap (flow-LLR vs Gaussian-LDA) on L19 activations. Est ~90min CPU (reuses FE606447 flows).
**Requires:** CPU only; cached activations.
**Would change:** Confirm → F-10's "covariance captures it all / Gaussian-null adequate" interpretation is refuted (non-Gaussian, non-topological signal exists); supports H-906. Reject → covariance sufficiency reaffirmed; cov-spectrum 0.7928 stands as near-ceiling.
**Blocks:** nothing.

### H-926: The prefill→final-token transformation is better modeled by an invertible nonlinear flow than by additive arithmetic, recasting F-3's cos≈0 as a Jacobian-rotation artifact
**Priority:** MEDIUM
**Motivated by:** 2606.06447 + F-3, H-875
**Test:** Fit invertible flow g: prefill→final L19 activations; compare ‖h_final − g(h_prefill)‖ vs ‖h_final − (h_prefill + offset)‖. Est ~2h CPU.
**Requires:** CPU only; cached prefill and final-token L19 activations.
**Would change:** Confirm (flow residual ≪ additive) → F-3's structural-orthogonality reading and H-875's additive functor are undercut; the directions are nonlinearly coupled. Reject → additive approximation holds, F-3 strengthened.
**Blocks:** nothing.

---


### H-927: Correctness is a feature-split concept — no single SAE feature on L19 matches the DoM AUROC, but a sparse set does
**Priority:** HIGH
**Motivated by:** 2606.07007 + F-2 (FE291)
**Test:** Train a small Top-K SAE on the cached 500×1536 L19 prefill matrix; compute per-feature correctness AUROC and the best sparse k-feature subset AUROC. Est 45min CPU.
**Requires:** CPU, cached `pathway11_h100/prefill_gated_compute` activations, ~50-line torch Top-K SAE.
**Would change:** Confirm → F-2's "single direction" framing is reinterpreted as a superposition of split features; the (PC1,PC9)=0.7856 / cov-spectrum=0.7928 lifts become the predicted multi-neuron approximation gain. Reject (a single feature ≥0.77) → strengthens the atomic-direction reading of F-2.
**Blocks:** H-928, P11-FE1273

### H-928: The D-bucket is feature-absorbed, not single-feature separable
**Priority:** MEDIUM
**Motivated by:** 2606.07007 + F-7 + H-540
**Test:** Using SAE features from H-927, run subset-inclusion tests of D-bucket-selective feature active-sets against broader features. Est 20min CPU (after the SAE exists).
**Requires:** CPU, SAE features from H-927, cached D-bucket labels.
**Would change:** Confirm → refutes H-540 (no single SAE feature cleanly isolates the D-bucket; F-7's "collective signature" reading is reinforced by an absorption mechanism). Reject → single-feature D-bucket probe is viable, supporting H-540.
**Blocks:** nothing

---


### H-929: The L19 prefill DoM correctness direction is an off-principal weight read, fragile to SFT but stable under RLVR
**Priority:** HIGH
**Motivated by:** 2606.07082 + F-2 / H-8 / EXP-57
**Test:** Take a single base model, produce SFT-only and RLVR (GRPO) MATH
derivatives, re-extract L19 prefill activations, re-fit DoM per checkpoint.
Measure cos(DoM_base, DoM_regime), the principal-angle rotation of the L19
W_down/W_o top-k subspace (paper Eq.2), and transported-DoM AUROC drop. ~2 H100
days.
**Requires:** GPU fine-tuning + re-extraction; cached base-model DoM already in
hand.
**Would change:** Confirm → F-2's single-direction reading is partly
recipe-specific (DoM moves with SFT weight rotation), qualifying F-1
universality and H-8 stability; the correctness axis is off-principal. Reject →
DoM is recipe-invariant, strengthening F-2/H-8.
**Blocks:** H-8, H-10

### H-930: The L19 residual-stream covariance spectrum is heavy-tailed, making F-10's Gaussian null the wrong reference family
**Priority:** HIGH
**Motivated by:** 2606.07082 + F-10 / FE881 (EXP-57)
**Test:** Fit Hill tail exponent (paper Appendix E) to the eigenvalue spectrum
of the cached PC1-residualized L19 covariance; if finite/stable (heavy-tailed),
re-run the FE881 top-20 log-eigval probe against a heavy-tailed surrogate null
and recompute AUROC-over-null. ~45min CPU.
**Requires:** CPU; cached `pc1_resid_cov_spectrum_results.json` covariance.
**Would change:** Heavy-tailed → F-10's Gaussian-null comparison is uninformative
and FE881's 0.7928 must be re-benchmarked against a heavy-tailed null; the
PH-vs-Gaussian framing weakens. Gaussian → F-10's null choice is vindicated.
**Blocks:** nothing

### H-931: F-1 dimensional breathing amplitude is a post-training-regime fingerprint, not a universal transformer property
**Priority:** MEDIUM
**Motivated by:** 2606.07082 + F-1 / H-10
**Test:** On one base model's SFT-only / OPD / RLVR triple, extract the
prefill→mid-gen→final PR breathing curve on MATH for each. Compare peak position
and amplitude across regimes; also compute stable rank (paper Eq.8) per regime.
~1 H100 day (re-extraction) or free if checkpoints + activations available.
**Requires:** Three checkpoints of one base + activation re-extraction (GPU), or
public OPD/SFT/RLVR checkpoint triple.
**Would change:** Curves differ by regime → F-1's "universal across architectures
and scales" is regime-confounded (the four F-1 models were trained by different
pipelines); add a recipe-control caveat. Curves match → F-1 universality
survives a new and strong confound.
**Blocks:** nothing

---


### H-932: The L19 prefill DoM direction is partly aligned with the unembedding-matrix "average-token" anisotropy axis
**Priority:** HIGH
**Motivated by:** 2606.07502 + F-2 (FE291)
**Test:** Compute `h_avg = log(p̂) W_U⁺` for Qwen-2.5-1.5B (tied `W_U`); measure `cos(DoM, h_avg)`, `cos(PC1, h_avg)`. Run uniform-freq control + RedPajama freq. ~15min CPU.
**Requires:** CPU; Qwen-2.5-1.5B unembedding/embedding tensor (one-time ≈0.46 GB download); cached DoM/PC1 vectors (FE291).
**Would change:** If |cos| ≳ 0.5, F-2's correctness direction is reframed as partly a frequency/anisotropy confound; if |cos| ≈ 0, F-2 is reinforced as genuinely semantic.
**Blocks:** H-933

### H-933: Removing the W_U edge spectrum (EmbedFilter) does not destroy the L19 correctness signal
**Priority:** HIGH
**Motivated by:** 2606.07502 + FE881 (cov-spectrum 0.7928) + H-892
**Test:** Project cached L19 activations through `Φ_τ` (drop top-k + bottom-k singular dirs), recompute DoM and top-20 cov-spectrum AUROC for τ∈{2,4,8}. ~25min CPU.
**Requires:** CPU; `W_U` SVD; cached L19 prefill activations (500×1536).
**Would change:** A large AUROC drop refutes H-892 (signal lives in the non-semantic edge spectrum); stable/higher AUROC confirms the correctness signal is in the semantic bulk and survives anisotropy removal.
**Blocks:** nothing

---


### H-934: L19 prefill DoM AUROC is partly a slow length/topic-nuisance shortcut, not a pure correctness direction
**Priority:** HIGH
**Motivated by:** 2606.07770 + F-2 (FE291, cos(DoM,PC1)=0.9216)
**Test:** Residualize cached L19 prefill activations (and the PC1 projection) against per-problem prompt+CoT token-count and topic id; recompute OOF 5-fold correctness AUROC and report the drop from 0.7731. Cross-check by correlating the PC1 projection with token-count. ~25min CPU on existing P11 caches.
**Requires:** CPU; cached `pathway11_h100/prefill_gated_compute/results.json` + `pca_covariance/results.json`; per-problem token counts from P11 generation logs.
**Would change:** A material AUROC collapse downgrades F-2 from "correctness direction" to "partly length-confounded"; no drop hardens F-2 against the across-trajectory-contrast critique.
**Blocks:** H-935, H-936

### H-935: Prefill vs final-token DoM orthogonality is a slow-noise/dynamics timescale split
**Priority:** MEDIUM
**Motivated by:** 2606.07770 + F-3 (cos 0.046)
**Test:** Linear probes from prefill DoM and from final DoM onto problem-constant nuisance variables (length, topic); compare nuisance-decode R^2. If prefill ≫ final on nuisance decodability, the orthogonality reflects timescale, not two computations. ~25min CPU.
**Requires:** CPU; cached prefill + final-token L19 activations.
**Would change:** Reframes F-3 from "structurally distinct correctness computations" to "slow-nuisance axis ⟂ dynamics axis."
**Blocks:** nothing

### H-936: A within-trajectory contrastive probe on per-token L19 trajectories recovers a correctness direction distinct from static DoM
**Priority:** MEDIUM
**Motivated by:** 2606.07770 + F-2 + F-9
**Test:** Fresh per-token L19 extraction on MATH-500; train JEPA-style probe with within-trajectory negatives; compare direction (cosine) and AUROC to DoM. ~1 H100 day.
**Requires:** H100; new per-token forward passes (Qwen-2.5-1.5B, 1024-tok).
**Would change:** Divergent-but-comparable direction = DoM was a slow-noise shortcut (reframes F-2); convergent = DoM is genuinely the within-trajectory dynamics direction (hardens F-2/F-9).
**Blocks:** nothing

---


### H-937: A relational cross-model similarity signal predicts 1.5B correctness without a supervised direction
**Priority:** HIGH
**Motivated by:** 2606.07818 + F-2 + H-896
**Test:** Compute CKA and per-problem cross-model cosine between cached 1.5B and 7B L19-prefill activations; AUROC vs committed correctness labels; compare to DoM 0.7731 and to DoM+similarity stack. ~30min CPU (P11-FE1285).
**Requires:** CPU; cached 1.5B and 7B L19-prefill NPZs (verify 7B cache exists).
**Would change:** Confirm → DoM is one approximation of a relational signal, demoting "single direction" framing of F-2. Reject → DoM's directional sufficiency strengthened.
**Blocks:** nothing

### H-938: Cross-model representational similarity is most correctness-predictive at early layers, not L19
**Priority:** MEDIUM
**Motivated by:** 2606.07818 + F-6 + F-1
**Test:** Layer-swept CKA/RSA(1.5B^ℓ, 7B^ℓ) vs per-problem correctness across all layers; identify peak-predictive layer. Needs full-layer re-extraction for both scales (~1 H100 day, P11-FE1288).
**Requires:** GPU re-extraction (per-layer prefill activations, both scales); MATH-500.
**Would change:** Confirm → L19 specialness is partly a measurement-layer choice; reframes F-2/F-6. Reject → middle-layer L19 is genuinely the cross-scale information locus.
**Blocks:** nothing

---


### H-939: A FlowTracer throughput-weighted aggregate of L19 token activations beats the single-position prefill DoM (0.7731)
**Priority:** HIGH
**Motivated by:** 2606.10646 + F-2 + F-3
**Test:** Re-extract MATH-500 (Qwen-2.5-1.5B) with middle-layer attention maps + per-token L19 states; compute FlowTracer Doob-h answer-conditioned flow throughput; build a throughput-weighted DoM; 5-fold OOF AUROC vs 0.7731 / 0.7186 / 0.7928. ~1 H100 day.
**Requires:** H100 (fresh forward passes with attention output); model Qwen-2.5-1.5B; new attention + per-token cache (we have neither).
**Would change:** Confirm → F-2's fixed-position readout is suboptimal and a routing-aware readout is the better correctness signal; cos(flow_DoM, endpoints) would also test whether F-3's prefill⊥final is a missing-middle artifact. Reject → strengthens F-2 (last-token readout is already near-optimal; routing structure adds nothing for *correctness prediction*, even if it helps RL credit).
**Blocks:** nothing

### H-940: The L19 prefill DoM signal is carried disproportionately by structural-delimiter tokens, not semantic content
**Priority:** MEDIUM
**Motivated by:** 2606.10646 (§5.2 structural delimiters recover 38.8/39.4 of the gain) + F-5
**Test:** Partition prompt/generation tokens into delimiter vs content; compare per-token L19 DoM-projection separability and breathing amplitude across the two classes (needs per-token states — pair with the H-939 re-extract, or use a small CPU pilot on the no_cot per-token text where available). ~0.5 H100 day if bundled with H-939.
**Requires:** per-token L19 states + token-type tagging; H100 for the extract.
**Would change:** Confirm → qualifies F-5's "content-dependent, not AR-mechanics" — the signal is partly structural/positional. Reject → strengthens F-5.
**Blocks:** nothing

---

## Historical / abandoned

These were once candidates but have been explicitly killed or parked
further than LOW priority:

- **E1 fixed-vector steering** — direction rotates; revived only via H-1
  (per-position bank).
- **Zigzag persistent homology across layers** — parked under Pathway 10
  v1 Direction E; killed by EXP-026 (PH-under-null).
- **CoE + ABC-44 ensemble** — stacked +0.001 over CoE alone (EXP-025).
- **XGBoost over ABC-44** — signal is linear (EXP-029).
- **Non-Euclidean PH** — 0.774 < 0.7961 (EXP-018).
- **TwoNN intrinsic dimension** — 0.407 AUROC (EXP-021).

---

## Template for new hypotheses

```markdown
### H-{n}: {one-line hypothesis}
**Priority:** HIGH | MEDIUM | LOW | PARKED
**Motivated by:** {EXP-### result or arxiv ID}
**Test:** {what you'd run, what data, est time}
**Requires:** {GPU/CPU, model, new or cached data}
**Would change:** {what shifts on confirm vs reject}
**Blocks:** {H-### or "nothing"}
```
