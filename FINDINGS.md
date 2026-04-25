# FINDINGS.md — registry of findings that survived validation

Every finding is stated as a testable claim with its evidence. Strength
rating reflects how many models, controls, and independent replications it
has.

**Strength key.**
- **STRONG** — ≥ 3 models or benchmarks, controls run, negative null
  rejection ran.
- **MODERATE** — 1–2 models, at least one control passed, but not fully
  replicated.
- **PRELIMINARY** — observed but not yet controlled.

**Numbering continues monotonically. Next ID: F-11.**

---

### F-1: Dimensional breathing is universal across transformer architectures and scales

**Claim.** During chain-of-thought generation on reasoning tasks, residual-stream
covariance participation ratio rises sharply from prefill (~20-dim effective
subspace), peaks in a mid-generation region (positions 30–70, PR ~60-115
depending on model), and collapses at the final answer token (PR 2–18).
Correct trajectories collapse *harder* at the final token than incorrect.

**Strength:** STRONG (4 models, one null control).

**Evidence:** EXP-032 (Qwen-2.5-1.5B), EXP-031 (Qwen-2.5-7B), EXP-040
(Phi-3-mini-4k-instruct and Llama-3.2-1B-Instruct). EXP-041 (random-token
control) rejected the AR-mechanics null.

**Controls passed:**
- Random-token prompts give flat PR ≈ 10 across all positions (EXP-041).
- Stream-of-consciousness gives a different, muted curve (EXP-041).
- Reproduces across 4 distinct architectures (Qwen, Phi, Llama).
- Reproduces at two model scales (1.5B and 7B within Qwen).

**Controls not yet run:**
- Penultimate-token PR vs last-token (answer-vocabulary concentration null)
  — see PERSPECTIVES "strongest criticism" section.
- Code-generation benchmarks (HumanEval) — H-3.
- Short-CoT trajectory (20-40 tokens) — H-2.

**Strongest counterargument:** The final-token PR collapse may be driven
partly by the answer-token vocabulary being small (numbers, boxed LaTeX).
Mitigation: measure PR on penultimate token and on the residual minus
answer-token embedding — not yet done.

**Would be overturned by:** A short-CoT run (20-40 tokens) showing flat PR
across positions would indicate breathing is a long-generation artifact,
not a reasoning-trajectory property.

### F-2: Prefill L19 DoM is a strong correctness predictor, *better than final-token*

**Claim.** On Qwen-2.5-1.5B-Instruct, a 5-fold OOF logistic probe fit on
mean-pooled L19 activations at position 0 (last prompt token, pre-generation)
achieves **AUROC 0.7731** for K=1-correctness. Final-token DoM AUROC is
**0.7186**. The 7B shows the same pattern with a larger gap (~0.876 vs
~0.77).

**Strength:** MODERATE (2 models same family, clean methodology).

**Evidence:** EXP-037 (`pathway11_h100/prefill_gated_compute/results.json`
fields `prefill_DoM_auroc_oof` = 0.7731, `final_token_DoM_auroc_oof` =
0.7186).

**Controls passed:**
- Length-residualization: cos(L19 DoM, length direction) ≈ −0.09 at every
  layer; length-residualized probe AUROC matches raw within 0.002 (i.e., not
  a length artifact — in contrast to the ABC-44 pipeline).
- 5-fold stratified OOF (seed 9999) — not a test/train leak.

**Controls not yet run:**
- Replication on non-Qwen families (Phi-3, Llama) — Exp 1 captured the data
  but didn't measure prefill DoM AUROC for them; feasible post-hoc.
- Cross-benchmark: same prefill probe on BBH — partially done in Stage 5
  via transfer (AUROC 0.693 BBH→MATH) but not exactly the same quantity.

**Strongest counterargument:** The prefill signal may encode problem
*familiarity* (closeness to training data) rather than structural
decomposability. H-5 would distinguish these.

**Would be overturned by:** Replication on fresh held-out math problems from
a post-training-cutoff benchmark showing AUROC drop to ≤ 0.60 — would
indicate the signal is memorization, not structural prediction.

### F-3: Prefill and final-token DoM directions are orthogonal

**Claim.** `cos(prefill_DoM, final_DoM) ≈ 0.046` on Qwen-2.5-1.5B L19. At
every intermediate token position the cosine stays in [0.00, 0.20]; reaches
~0.37 only by position 200. The "can I solve this?" circuit and the "did I
solve this?" circuit live in geometrically unrelated subspaces.

**Strength:** MODERATE (one model, one benchmark, but replicated across
label schemes).

**Evidence:** `scratch/pathway10_temporal_and_verifier_results.json`
(`T1_temporal_dom.positions[0].cos_with_final_token_dom` = 0.046 at
256tok_NEW labels; qualitative result survives label correction).

**Controls passed:**
- Same cosine pattern at multiple intermediate positions (5, 9, 19, 29, 39,
  49, 59, 69 all in [0.00, 0.19]).
- Two-feature LR (prefill DoM + final DoM) achieves ≈ 0.794 AUROC, exceeding
  either alone — consistent with carrying complementary information.

**Controls not yet run:**
- Head-level attribution (H-13) to test the "different circuits"
  interpretation.
- Cross-architecture replication (compute from Exp 1 cache — feasible).

**Strongest counterargument:** Cosine is a crude metric for circuit
separation; attention heads could share with different weights. H-13 would
quantify.

**Would be overturned by:** A head-attribution study showing the same top-K
heads contribute to both probes.

### F-4: Correct trajectories collapse harder than incorrect at the final token ("asymmetric collapse")

**Claim.** Final-token participation ratio:
- Qwen-2.5-7B: correct 2.63 (n=366), incorrect 6.85 (n=134), ratio 0.384
- Qwen-2.5-1.5B: correct 4.33 (n=243), incorrect 8.51 (n=257), ratio 0.509
- Phi-3-mini: correct 12.2, incorrect 16.8
- Llama-3.2-1B: correct 4.4, incorrect 8.0

When the model is correct, it commits to a tighter low-dim answer geometry.

**Strength:** STRONG (4 models, bootstrap CIs on two, same qualitative
direction on all).

**Evidence:** EXP-031 / EXP-032 (`phase1_bootstrap.json` for Qwen), EXP-040
(`exp1_cross_model/results.json` for Phi-3, Llama).

**Controls passed:**
- Bootstrap CIs on Qwen: 7B P(ratio>1) = 0.000, 1.5B P(ratio>1) = 0.000 out
  of 1000 draws.
- Balance-controlled (n=134 vs 134): 7B ratio 0.381, 1.5B 0.511.

**Controls not yet run:**
- Answer-vocabulary null (see F-1 controls).
- Is "tighter final PR" predictive *within* the correct group of confidence
  (e.g., K=8 majority-correct vs K=1-only-correct)? — H-9 adjacent.

**Strongest counterargument:** Same as F-1 — answer-token vocabulary may
be smaller for the common correct answers.

**Would be overturned by:** Answer-vocabulary null showing the collapse is
fully vocabulary-driven.

### F-5: Breathing is content-dependent, not AR-mechanics

**Claim.** Feeding Qwen-2.5-1.5B random token IDs gives a flat PR ≈ 10
across all positions. Stream-of-consciousness produces a muted, early-peak
curve different from reasoning. MATH-500 produces the canonical three-phase
rise-to-30-collapse.

**Strength:** MODERATE (one model, clean null rejection but n=20 per
gibberish condition).

**Evidence:** EXP-041 (`gibberish_control/pr_curves.json` — conditions:
random_tokens, stream_of_consciousness, math500_baseline).

**Controls passed:**
- Random-token null clearly rejected (flat 9-12 vs MATH peak 30).
- Stream-of-consc comparison rules out "just fluent generation."

**Controls not yet run:**
- Cross-model replication (Phi-3, Llama on random tokens).
- Short-CoT (H-2) to distinguish trajectory-length vs reasoning-structure.
- Code generation (H-3) to test universality beyond math.

**Strongest counterargument:** n=20 per condition is small; possible that
a bigger sample would reveal muted-but-present breathing on random tokens.

**Would be overturned by:** A 100-prompt random-token rerun showing some
rise from pos 1 to pos 50.

### F-6: 7B prefill PR inversion is real but driven by incorrect-group concentration

**Claim.** Qwen-2.5-7B prefill correct-group PR (26.87, n=366) exceeds
incorrect-group PR (19.51, n=134); ratio 1.377 with bootstrap 95% CI
[1.226, 1.757], P(ratio>1) = 1.000. Balance-controlled (n=134 vs 134):
1.227, CI [1.131, 1.318]. **Not in 1.5B** (ratio 0.946, CI overlaps 1.0).
**Not in BBH** (all three subsets' ratios ≤ 1.01).

Mechanism: incorrect-group concentration at easy difficulty levels. Level 2
ratio = 3.98 (n_i = 5); Level 5 ratio = 1.11 (n_i = large). As difficulty
rises, both groups diversify and PR equalizes.

**Strength:** STRONG for the phenomenon (rigorous bootstrap, clean BBH
negative control), MODERATE for the level-driven mechanism.

**Evidence:** EXP-038 (`prefill_inversion/phase1_bootstrap.json`,
`phase2_cross_scale.json`, `phase3_mechanistic.json`).

**Controls passed:**
- Class-imbalance control (balance-controlled subsample).
- BBH three-subset negative control (no inversion even at balanced
  accuracy).
- Difficulty-level stratification shows the ratio decays from 3.98 to 1.11
  as level rises.

**Controls not yet run:**
- Inversion on other benchmarks (GSM8K, AIME) — H-OQ-9 in PROJECT_RECORD.
- Inversion on models between 1.5B and 7B (e.g., 3B) to see if there's a
  scale threshold.

**Strongest counterargument:** "Inversion" is aggregate — it's really
"incorrect-group concentration at easy-level failures." Not a deep
property of 7B residual streams as originally framed.

**Would be overturned by:** Finding inversion in a 1.5B model too, which
would undermine the 7B-specific story.

### F-7: D-bucket has a distinctive collective (not per-problem) geometric signature

**Claim.** 36 MATH-500 problems (7.2%) are K=1-right but K=8-majority-wrong
("pathological"). Their group prefill PR is **14.49**, lowest of any
bucket (A=18.72, B=16.65, C=20.04). But per-problem local PR (k=20 NN) is
**12.84**, indistinguishable from other buckets (A=12.66, B=12.80, C=12.97).
The cluster exists at group scale but dissolves per-problem.

**Strength:** MODERATE (one model, one benchmark; clear signal at group
level, clear failure at per-problem level).

**Evidence:** EXP-038 (`phase2_cross_scale.json.d_bucket`), EXP-039
(`phase1_features.json` + ablation showing `prefill_lpr` = +0.000 Δ F1).

**Controls passed:**
- Per-bucket means vs per-problem local PR are reproducibly different.
- `prefill_lpr` contributes exactly 0.000 to classifier macro-F1.

**Controls not yet run:**
- Density-based outlier detection (LOF, isolation forest) — H-7.
- Is D consistent across labels (would 1024-tok re-label change
  membership)? — partially addressed but not cleanly.

**Strongest counterargument:** 36/500 is small; with 1000 bootstrap draws
on the D-labels, boundary effects might shift the numbers.

**Would be overturned by:** D-recall > 60% from a density-based method,
which would indicate the cluster *is* detectable per-problem and EXP-039's
framing was wrong.

### F-8: Selective-prediction via prefill DoM works at 50% coverage

**Claim.** With prefill L19 DoM-gated refuse-and-spend on Qwen-2.5-1.5B at
coverage 0.5 (keep top-half by prefill score, answer Q4 at K=1, Q3 at K=4,
refuse bottom-half), accuracy on answered = **71.6%** at average K=2.5, vs
48.6% unconditional at K=1 and 49.4% random-refuse.

**Strength:** MODERATE (one model, replicated across multiple coverage
points).

**Evidence:** EXP-037
(`prefill_gated_compute/results.json.headline.refuse_and_spend_coverage_0.5`).

**Controls passed:**
- Random-refuse baseline (0.494 acc-on-answered).
- Neg-seq-len alternative (0.707, slightly worse than prefill DoM).
- Final-token DoM alternative (0.624, much worse — reinforces F-2).

**Controls not yet run:**
- Replication at different coverage levels (partial — 25% and 75% not
  reported at 1024tok labels, only at 256tok).
- Replication on another model.

**Strongest counterargument:** 50% coverage is arbitrary; at 25% coverage
the uplift might be smaller.

**Would be overturned by:** A replication on a different model where
prefill-driven refusal doesn't beat random — would suggest the Qwen-1.5B
signal is model-specific.

### F-9: CoE-60 trajectory features are redundant with single-layer L19 DoM

**Claim.** CoE symmetric MATH↔BBH transfer AUROC (Pathway 9 Exp 5) = 0.716.
Single-layer L19 DoM symmetric MATH↔BBH transfer at 1024-tok labels
(Stage 5) = 0.7199. Δ = +0.004 in favor of DoM. A 60-dim trajectory
classifier adds no cross-domain signal over a 1-dim direction at the right
layer.

**Strength:** MODERATE (cross-domain replication on 3 BBH subsets; only
directly contrasts CoE with DoM on transfer, not within-domain at 1024
tokens — that's H-6).

**Evidence:** EXP-030 (Pathway 9 CoE transfer), EXP-036 (Stage 5 L19 DoM
transfer, `verdict_vs_coe` = 0.0039).

**Controls passed:**
- Source-domain PCA applied to both domains (D11 fix from Pathway 9).
- BBH pooled across 3 subsets.

**Controls not yet run:**
- CoE within-domain at 1024-tok labels (H-6) — if CoE stays at 0.80+
  within-domain while DoM is 0.77, the within-domain claim tilts back
  toward CoE.

**Strongest counterargument:** Cross-domain and within-domain are different
tests; H-6 hasn't run. The redundancy claim is weaker within-domain.

**Would be overturned by:** H-6 showing CoE ≥ 0.80 at 1024-tok while
single-layer DoM stays at 0.77.

### F-10: Persistent homology on trained residual streams is at the Gaussian null

**Claim.** Five raw PH summary features (H0_max_lifetime, H0_entropy,
H1_pers_entropy, H1_n_features, H1_max_lifetime) on Qwen-2.5-1.5B L19
predict MATH-500 correctness at AUROC 0.690. A rank-matched
empirical-covariance Gaussian null on the same features gives 0.693. Gap
zero — PH captures no topology-specific signal beyond matched covariance
structure.

**Strength:** STRONG (explicit null, two replications in pathway 9, predicted by EoS framework).

**Evidence:** EXP-026 (`pathway9/results/exp3a_gaussian_null_v2.json`,
fields `classifiers.real_ph.auroc_holdout` = 0.690 vs `classifiers.null_ph.auroc_holdout` = 0.693).

**Controls passed:**
- SVD-based empirical covariance sampler (mathematically equivalent to
  `multivariate_normal`).
- 5 features (one dropped per D2 audit).
- CV std reduced from 0.245 (CV50) to 0.043 (CV5) after matching.
- Null prediction consistent with Tuci et al. Sharpness-Dimension
  framework.

**Controls not yet run:**
- Extension to other layers, not just L19 (PH across layers in Exp 1 is
  also weak at 0.646, consistent).
- Extension to other benchmarks.

**Strongest counterargument:** Null may be too strong — if PH does capture
something, it's explained by second moments. Still a valid framing change:
"topology equals covariance structure in this setting."

**Would be overturned by:** A per-layer sweep showing PH > null at a
specific layer / metric combination we didn't test. (Non-Euclidean PH was
tested in EXP-018 and also didn't lift — so this is unlikely.)

---

## Honorable mentions — findings we have evidence for but haven't fully validated

### F-candidate: Negative sequence length is the Pareto-dominant compute-allocation feature (beats prefill DoM)

Evidence: EXP-037 — neg-seq-len top-heavy K=4.5 → 0.542 accuracy; prefill
middle-heavy K=4.5 → 0.526; prefill τ-threshold K=4.5 → 0.492.

Not promoted to a full finding because neg-seq-len is post-hoc (requires
running generation to know it), so it's an *upper bound* on what any
pre-generation signal could achieve — not a deployable baseline.

### F-candidate: The 20.8% MATH-500 accuracy baseline was a truncation artifact

This is in PROJECT_RECORD §1d ND-10 as a killed direction rather than a
finding, because the "finding" is methodological: `max_new_tokens=256`
truncated two-thirds of correct trajectories. The replacement is the 48.6%
baseline at 1024 tokens (F-8 implicit reference).

---

## Template for new findings

```markdown
### F-{n}: {one-sentence claim}
**Strength:** STRONG | MODERATE | PRELIMINARY
**Evidence:** EXP-###
**Controls passed:** {list}
**Controls not yet run:** {list}
**Strongest counterargument:** {one sentence}
**Would be overturned by:** {what result would kill this}
```
