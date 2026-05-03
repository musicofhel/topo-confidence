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

### F-2: A single L19 prefill direction predicts correctness on Qwen-2.5-1.5B at AUROC 0.7731

**Claim.** Pathway 11 H100 stage 1 established F-2 via supervised
1536-d logistic regression at L19 prefill: 5-fold OOF AUROC 0.7731 on
500 MATH-500 problems (243 correct on Qwen-2.5-1.5B-Instruct K=1).
**EXP-55 reframes the direction as unsupervised-identifiable.** On the
same cached prefill (`pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`),
the dominant eigenvector of the centered prefill covariance
(unsupervised PCA) yields single-feature 5-fold OOF AUROC 0.7458 with
cosine 0.9216 against the supervised mass-mean DoM. Decomposing the
unit DoM in the PC basis: 85% of energy in PC1 alone, 97.6% in the
top-10. CAST (Lee 2409.05907) class-mean PCA-PC1 (μ = (μ⁺ + μ⁻)/2)
is numerically indistinguishable from the unsupervised PC1 (AUROC
0.7458, cosine with global-PC1 ≈ 1.000). Under matched 5-fold
single-feature protocol, supervised DoM AUROC is 0.7679. **The
correctness direction at L19 is the dominant variance direction.**
PC9 (variance rank 9, 2.5% of variance) carries a secondary spike
(single-feature AUROC 0.6575, DoM-coeff 0.225) — orthogonal-to-PC1
correctness signal that may be a topic / difficulty axis.

**Strength:** STRONG (now triangulated by supervised + unsupervised +
class-mean-supervised decompositions, all agreeing within 0.022 AUROC
and cosine ≥ 0.92 on the same direction).

**Evidence:** EXP-001 / `pathway11_h100/prefill_gated_compute/results.json`
(supervised 1536-d probe AUROC 0.7731); P11-E12 / FE110 /
`pathway11_h100/gpu_bundle/results.json` (DoM mass-mean replication
0.7711 on cached prefill); **EXP-55** /
`pathway11_h100/pca_covariance/results.json` (unsupervised PCA-PC1
AUROC 0.7458, CAST class-mean PCA-PC1 AUROC 0.7458, cos with DoM
0.9216, DoM-energy in PC1 0.849).

**Controls passed:**
- Supervised vs mass-mean DoM (FE110 EXP-53): 0.7711 ≈ 0.7731.
- Mean-shift on residualized clouds (FE115): residual DoM AUROC
  0.8016 (gap +0.029 over raw F-2).
- **Unsupervised PCA-PC1 (EXP-55, FE291)**: 0.7458 single-feature OOF,
  cosine 0.922 with DoM, 85% DoM-energy in PC1 — the direction is
  label-free recoverable.
- CAST class-mean PCA-PC1 (EXP-55, FE291): 0.7458, numerically same
  direction as unsupervised PC1.

**Strongest counterargument:** F-2 is correlational. EXP-55 strengthens
the *correlation* (the direction is unsupervised + supervised +
contrastive all agreeing) but does not address causality. The
load-bearing test of F-2 remains FE214 / FE269 / FE283 (noising,
ablation, layer-zero ablation). If those degrade accuracy, F-2 is
causal. If not, F-2 is "L19 is where correctness is *first readable*"
not "L19 is where correctness is *computed*".

**Would be overturned by:** Causal ablation/noising at L19 PC1
preserving MATH-500 accuracy (FE283/FE214/FE269); cross-checkpoint
PC1 rotation > cosine 0.5 across HF Qwen 1.5B checkpoints (FE700);
PC1 AUROC ≪ 0.6 on a non-MATH-500 benchmark via cached activations.

### F-3: Prefill and final-token DoM directions are orthogonal, *structurally — not positionally*

**Claim.** On Qwen-2.5-1.5B L19, the cosine between prefill DoM and
final-token DoM is **−0.0617 raw, 0.0008 after Song-Zhong residualization**
(μ + pos_t + ctx_i), and **−0.0315 under Park-Choe-Veitch causal inner
product whitening** (M = Cov(γ_unembed)^{-1}, EXP-51). The orthogonality
is structural — survives removal of global mean, per-position bias,
per-problem context vectors, AND unembed-row covariance whitening.
NC3-collapse alignment (FE244) shows max |cos| of prefill DoM with top-50
answer-token unembed rows = 0.067 (mean 0.022); final = 0.128 (mean
0.055) — both well below the 0.3 NC3 threshold. The "can I solve this?"
circuit and the "did I solve this?" circuit live in geometrically
unrelated subspaces of the residual stream.

**Strength:** STRONG (one model, but survives the strongest cheap-control
available; cosine numerically tighter on residuals than raw and on
causal-whitened than raw; NC3 alignment far below the alternative-
explanation threshold).

**Evidence:** EXP-037 / `scratch/pathway10_temporal_and_verifier_results.json`
(P10 raw cos = 0.046 across multiple positions, original anchor); **EXP-48**
/ `pathway11_h100/song_zhong/results.json` (raw cos = −0.0617 on per-problem
1024-tok cache, residualized cos = 0.0008); **EXP-51** /
`pathway11_h100/phase3_corroborators/results.json` (causal-whitened cos =
−0.0315; FE244 NC3 alignment max |cos| 0.067/0.128).

**Controls passed:**
- Same cosine pattern at multiple intermediate positions in P10 (0.00–0.19
  through token ~50, ~0.37 by position 200).
- Two-feature LR (prefill DoM + final DoM) ≈ 0.794 AUROC, exceeding either
  alone — consistent with carrying complementary information.
- **Song-Zhong residualization (EXP-48, FE115)** — cos drops from −0.0617
  to 0.0008 after subtracting μ + pos_t + ctx_i. Position is NOT the
  source of orthogonality.
- **Causal inner product whitening (EXP-51, FE136)** — cos shifts −0.062
  → −0.032 under Park-Choe-Veitch unembed-cov whitening. Unembed-row
  geometry is NOT the source of orthogonality.
- **NC3 alignment (EXP-51, FE244)** — max |cos| with top-50 answer-token
  unembed rows = 0.067 (prefill) / 0.128 (final), both below 0.3
  threshold. F-3 is NOT generic neural collapse.

**Controls not yet run:**
- Head-level attribution (H-13) to test the "different circuits"
  interpretation.
- Cross-architecture replication (compute from Exp 1 cache — feasible).

**Strongest counterargument:** Cosine is a crude metric for circuit
separation; attention heads could share with different weights but produce
decorrelated DoM directions. H-13 would quantify.

**Would be overturned by:** A head-attribution study showing the same top-K
heads contribute to both probes; OR a setting where post-Song-Zhong cos > 0.5;
OR FE244 max |cos| > 0.30 (NC3 collapse explanation).

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
layer. **Spectral α (HT-SR) head-to-head (EXP-50)**: best-layer α-AUROC =
0.7026 (1.5B L28) / 0.7128 (7B L28), both below DoM 0.7731 at L19; joint
[α_L28, prefill_DoM_proj_L19] = 0.7833, +1pp over DoM alone. α-AUROC is
U-shaped in depth, near-chance at mid-layers (L13–L19) where DoM peaks —
α and DoM are ~orthogonal probes that target different network regimes.
The parsimony framing now has two independent confirming counter-examples
(CoE trajectory features and spectral α single scalar).

**Strength:** MODERATE (cross-domain replication on 3 BBH subsets; only
directly contrasts CoE with DoM on transfer, not within-domain at 1024
tokens — that's H-6; spectral-α corroborator on both 1.5B and 7B
strengthens the broader "scalar competitors don't beat DoM" reading but
doesn't directly address the CoE within-domain question).

**Evidence:** EXP-030 (Pathway 9 CoE transfer), EXP-036 (Stage 5 L19 DoM
transfer, `verdict_vs_coe` = 0.0039); **EXP-50**
(`pathway11_h100/spectral_alpha/results.json`: best α-AUROC = 0.7026
1.5B / 0.7128 7B at L28, joint with DoM = 0.7833 = +1pp).

**Controls passed:**
- Source-domain PCA applied to both domains (D11 fix from Pathway 9).
- BBH pooled across 3 subsets.
- **Spectral α head-to-head (EXP-50)** — independent scalar competitor
  (HT-SR theory) also fails to beat L19 DoM; cross-scale (1.5B + 7B)
  agreement on U-shaped depth profile.

**Controls not yet run:**
- CoE within-domain at 1024-tok labels (H-6) — if CoE stays at 0.80+
  within-domain while DoM is 0.77, the within-domain claim tilts back
  toward CoE.

**Strongest counterargument:** Cross-domain and within-domain are different
tests; H-6 hasn't run. The redundancy claim is weaker within-domain.
EXP-50's α evidence is a *broadening* signal rather than a strengthening
one for the CoE-specific claim.

**Would be overturned by:** H-6 showing CoE ≥ 0.80 at 1024-tok while
single-layer DoM stays at 0.77.

### F-10: Topology summary statistics on residual streams are not distinguishable from a Gaussian null

**Claim.** F-10's narrow form (5 PH summary features ≈ matched-cov
Gaussian null on raw L19 clouds, FE026 gap −0.003; ≈ inverted on
Song-Zhong residualized clouds, FE116 gap −0.067) is *grounded* by
EXP-55: the supervised L19 prefill DoM is essentially the dominant
covariance principal component (PC1 cos 0.922, 85% DoM-energy in PC1).
The covariance signal F-10 says "PH cannot improve on" is a single
direction. PH features layered on the covariance ellipsoid add no
orthogonal signal because the correctness signal is concentrated in
14.7% of variance along PC1, and the PH descriptor family integrates
*all* of the cloud's H_0/H_1 structure without distinguishing PC1
from the rest.

**Strength:** STRONG (PH-null gap holds under raw, residualized, and
zigzag-trajectory cloud constructions; covariance pathway now grounded
in a single quantified direction with cos 0.922 to supervised DoM).

**Evidence:** EXP-026 (raw 5-feature PH gap −0.003); EXP-52 (FE321
zigzag gap −0.066 + 7-descriptor +0.097, no topology component);
EXP-54 (FE116 residualized gap −0.067); **EXP-55** (PCA-PC1 = DoM
direction, covariance pathway grounded as PC1).

**Controls passed:**
- Diagonal vs full covariance null (D1 fix in EXP-026 v2).
- Drop H0_n_features rank-tie (D2 fix).
- Zigzag descriptor pair on 28-layer trajectory (FE321).
- Song-Zhong residualization (FE116).
- **PCA-PC1 = supervised DoM (EXP-55, FE291)** — covariance pathway
  is a single direction with 14.7% var share and cos 0.922 to DoM.

**Strongest counterargument:** A covariance-controlled PH probe that
regresses out the per-problem PC1 projection before computing PH and
finds residual signal would falsify the new sharpened form. None
attempted yet.

**Would be overturned by:** PH features computed on the [PC2..PC1536]
subspace (PC1-residualized clouds) showing real_AUROC ≥ null + 0.05.

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
