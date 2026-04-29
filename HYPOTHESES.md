# HYPOTHESES.md — prioritized queue of untested hypotheses

Each hypothesis is actionable: a concrete test, an estimated cost, and a
resolution that would change our understanding. Ranked by how much they'd
change the story.

**Priority key.** HIGH | MEDIUM | LOW | PARKED.

**Numbering continues monotonically. Next ID: H-23.**

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
| **Cheap GPU** | H-19 | C_exact verification routing ★ | ~30 min H100 | ~$1 | Highest-leverage; converts F-8 from predictor to recipe. |
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
