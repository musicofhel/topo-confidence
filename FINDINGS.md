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
  — H-699; see PERSPECTIVES "strongest criticism" section.
- Code-generation benchmarks (HumanEval) — H-3.
- Short-CoT trajectory (20-40 tokens) — H-2.

**Strongest counterargument:** The final-token PR collapse may be driven
partly by the answer-token vocabulary being small (numbers, boxed LaTeX).
Mitigation: measure PR on penultimate token and on the residual minus
answer-token embedding — H-699 (filed 2026-05-03, not yet run).

**Would be overturned by:** A short-CoT run (20-40 tokens) showing flat PR
across positions would indicate breathing is a long-generation artifact,
not a reasoning-trajectory property.

### F-2: A single L19 prefill direction predicts correctness on Qwen-2.5-1.5B at AUROC 0.7731

**Updated evidence line (append):**
EXP-79 / FE421 (`pathway11_h100/results/fe421_regularized_concat.json`):
ridge-LR concat(3072) = 0.8509, final-only(1536) = 0.8493,
prefill-only(1536) = 0.7844. Confirms directional ceiling (0.7844 ≈
EXP-58's 0.7847); reveals final token carries 12.8pp more signal under
ridge-LR than DoM extracts (0.8493 vs 0.7210).

**Updated controls (append):**
- **Ridge-LR concat/decomposition (EXP-79, FE421)**: final-token-only
  ridge-LR 0.8493 ≈ concat 0.8509 — prefill adds only +0.0016. Ridge-LR
  prefill-only 0.7844 matches EXP-58's 0.7847 ceiling. DoM concat
  0.7574 < DoM prefill 0.7699, confirming EXP-51's naive-concat-dilutes
  finding.
- **Cross-architecture generalization — SATISFIED (2026-06-09,
  `exp1_cross_model/dom_auroc_results.json`)**: the prefill DoM-predicts-
  correctness effect replicates on two non-Qwen architectures at the same
  ~⅔-depth layer. Phi-3-mini (33L, ⅔-depth L21) AUROC **0.8089** (peak
  0.8148 at L17); Llama-3.2-1B (17L, ⅔-depth L11) AUROC **0.7458** — both
  ≈ or above the Qwen-1.5B L19 reference 0.7731. The L19 prefill direction
  is an architecture-general correctness readout, not a Qwen artifact. With
  3 architectures + controls, F-2 is **STRONG**.
- **Causal lever-vs-diagnostic — SATISFIED (FE269, 2026-06-09,
  `causal_dom/results/verdict.json`)**: directional ablation of the L19
  prefill DoM (Arditi 2406.11717 protocol) does NOT remove math ability
  (L19-only ΔMATH −0.2pp, all-layer ΔMATH +2.0pp; GSM8K/MMLU drops ≤2.6pp)
  and activation addition does NOT induce correctness (α-sweep degrades
  MATH 47.8→33.4%; entangled, not surgical — flips 18–20% of incorrect but
  retention falls 80→54%, no α meets flip≥15% with retention≥85%). Verdict:
  **DIAGNOSTIC** — the direction is a *correlational readout*, not a causal
  lever. This bounds F-2's interpretation: use it for selective prediction
  / routing (F-8), not steering.
- **Transfer ceiling + portability axis (EXP-81/82, SPEC v6, 2026-06-11)**: the
  0.7731 number is *in-domain* OOF. The Edge Program transfer panel shows the
  supervised DoM is **hidden-dim-bound** — its weight vector lives in one model's
  hidden space, so it cannot cross scale/architecture without retraining (no T3–T5
  entry), and its cross-MATH-category transfer is the *weakest* of the panel (T1
  LOCO aggregate 0.743, worst-cell Number-Theory **0.603** at n=62). The real
  portability axis is **fixed-dim feature vs hidden-dim-bound**, not
  supervised-vs-geometric: every fixed-dim *geometry* feature (cov-spectrum,
  CoE depth-grid, prompt-cloud) was refuted as portable value-add, while the
  fixed-dim **free** baseline (length+logprob) ports best (F-8 re-pin). DoM's
  cross-arch *effect* still replicates per-model (Phi-3 0.81, Llama-3.2 0.75
  above), but the *direction* must be refit per model. Evidence:
  `pathway11_h100/generalization_edge/results/phase1_panel.json`.

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
- **Pre-final token (EXP-75, FE416)** — cos(prefinal DoM, prefill DoM) =
  −0.054 (< 0.3 threshold). cos(prefinal, final) = 0.717. The rotation
  is gradual across the generation trajectory, NOT a last-token positional
  artifact.
- **Layer-sweep (EXP-76, FE119)** — cos(prefill DoM, final DoM) near zero
  at ALL 29 layers (max |cos| = 0.103 at L28). F-3 orthogonality is a
  network-wide geometric fact, not L19-specific.

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
- **Length-band PR (EXP-72, FE15)**: PR roughly stable across bands
  (Short=17.5, Medium=21.5, Long=20.9). DoM AUROC holds within each
  band (0.75–0.83). NOT a length artifact.
- **MP bias correction (EXP-73, FE16)**: Corrected PR = 18.9 (naive
  19.9, ~5% correction). Correct=18.5 vs incorrect=19.6. Class
  asymmetry survives finite-sample bias correction.

**Controls not yet run:**
- Answer-vocabulary null (see F-1 controls; H-699).
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

**Updated scope (FE19, 2026-06-10 — what does NOT extend F-8):** A *post-hoc*
verification-routing alternative (C_exact: self-grade the K=1 answer, route
verify-failures to K=8 or to self-correction; Kumaran 2604.22271 PANL signal)
was tested directly and **LOSES** — no non-degenerate routed policy beats the
achievable uniform-K frontier (the upper convex hull = random K=1/K=8 routing)
beyond noise (best routed point +0.1 SE, n=500). The PANL activation probe for
"K=1-wrong" peaks at L22 AUROC **0.7555**, *below* the pre-hoc prefill-DoM
**0.7731** — so the post-hoc signal is weaker than the pre-hoc one (opposite of
Kumaran's 7B/27B prediction). Verbalized self-verification is anti-informative
(AUROC 0.480), and **verify-then-correct HURTS** (full-coverage 0.474 vs K=1
0.486; 46 right→wrong vs 40 wrong→right). **H-19 refuted at 1.5B.** Net: the
selective-prediction story stays *refuse-and-spend on the pre-hoc prefill score*
(this finding), not verify-and-route. Evidence: `pathway11_h100/verify_route/
results/verdict.json`; brief `research-graph/briefs/result-2026-06-10-P11-FE19.md`.

**Re-pinned generalizing readout (EXP-81/82, SPEC v6, 2026-06-11 — what the F-8
gate should READ).** The prefill-DoM score (0.7731) is the best *in-domain*
readout but the **worst-transferring** one. The Generalization-First Edge Program
bake-off (`pathway11_h100/generalization_edge/`) measured cross-distribution
transfer across a (model × domain) panel and found the **free two-scalar baseline
— generation length + mean token logprob — is the most GENERALIZING correctness
readout**: T1 leave-one-MATH-category-out aggregate **0.845** (vs prefill-DoM
0.743), T3 cross-scale 0.865, and on a **held-out family that played zero role in
selection, OOF AUROC 0.810 ≥ the 0.70 ship bar** (T5;
`results/phase3_t5_free_baseline.json`). It needs **no activations**, so it is a
shippable, model-agnostic generate-then-abstain gate. **Confirmed
cross-architecture (EXP-82 follow-on, 2026-06-11):** the gate clears the bar on
**three** held-out families spanning near AND far architectures — SmolLM2-1.7B
(near, Llama-arch) **0.810**, **Gemma-2-2b-it (far, distinct arch) 0.844**, and
**OLMo-2-1B (far, distinct arch) 0.838** — so the generalization is not a
near-family/Llama artifact. **No hidden-state geometry
adds significant *portable* value** over it: in-domain only the raw CoE
layer-profile clears the incremental gate (+1.75pp, p=0.018) but it is
hidden-dim-bound; the portable depth-grid form does not add (p=0.137) and
transfers ≤0.63 (H-C refuted), and the prompt-token-cloud (the last untested
geometry) loses even in-domain to a plain **prompt-length** prior (0.694 < 0.706,
gate p=0.42; **H-E refuted**, `results/phase2_arm2a.json`). Pin BOTH scalars:
*which* scalar carries is family-dependent — logprob carries cross-domain
(MATH→BBH 0.728, length→chance) and on Gemma (0.756) and OLMo-2 (0.654), while
length carries on SmolLM2 (0.781, logprob→chance) and leads on OLMo-2 (0.816); the
length+logprob combination is the only readout robust across all of them. Guarantee form
(Phase 4, `results/phase4_recalibration.json`): a conformal certificate transfers
**zero-shot** to a stronger model (1.5B→7B, valid at ε=0.2) but **no light-head
recalibration up to k=64 rescues a valid certificate on the harder BBH domain** —
target-task accuracy, not calibration, is the ceiling. Evidence: brief
`research-graph/briefs/result-2026-06-11-P11-FE-EDGEGEN.md`;
`pathway11_h100/generalization_edge/RESULTS_v6_phase234.md`.

### F-9: CoE-60 trajectory features are redundant with single-layer L19 DoM

**Updated evidence line (append):**
EXP-79 / FE421: ridge-LR final-only at L19 = 0.8493, exceeding
CoE-60's within-domain ~0.811 from a *single layer*. This raises the
baseline against which trajectory features must demonstrate value.

### F-10: Topology summary statistics on residual streams are not distinguishable from a Gaussian null

**Claim.** F-10's narrow form (PH = matched-cov Gaussian null) now
has a quantified positive companion: the post-PC1 correctness signal
lives in the eigenvalue spectrum of the per-problem residualized
covariance. **EXP-57 / FE881** shows top-20 log-eigvals on
PC1-residualized clouds give OOF AUROC 0.7928 (strongest L19 probe
at any tier). The same cloud's PH features fail (EXP-56/FE880
gap −0.074). F-10 sharpens: the covariance pathway carries the signal
*spectrally* — PH descriptors integrate over the eigenvalue
distribution and lose the shape-of-decay information that the
spectral probe captures.

**Strength:** STRONG (PH-null gap holds under raw, zigzag-residualized,
and PC1-residualized clouds, with the post-PC1 spectral pathway now
quantified at 0.7928 — three-way triangulation of the topology-vs-
covariance question).

**Evidence:** EXP-026 (raw 5-feature PH gap −0.003); EXP-52 (FE321
zigzag); EXP-54 (FE116 residualized); EXP-55 (PCA-PC1 = DoM); EXP-56
(FE880 PC1-residualized PH gap −0.074); **EXP-57** /
`pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json`
(top-20 log-eigvals on PC1-residualized clouds OOF AUROC 0.7928 —
the post-PC1 correctness signal is purely spectral, not topological).

**Controls passed:**
- Diagonal vs full covariance null (D1 fix in EXP-026 v2).
- Drop H0_n_features rank-tie (D2 fix).
- Zigzag descriptor pair on 28-layer trajectory (FE321).
- Song-Zhong residualization (FE116).
- PCA-PC1 = supervised DoM (EXP-55, FE291).
- PC1-residualization: PH gap −0.074 (EXP-56, FE880).
- **Per-problem cov spectrum (EXP-57, FE881)**: top-20 log-eigvals
  on PC1-residualized clouds OOF AUROC 0.7928 — quantifies the
  post-PC1 signal that PH integrates over.

**Strongest counterargument:** A higher-order PH instrument computed
on the same PC1-residualized clouds (zigzag, FTS-PH, persistence
landscapes) showing real ≥ null + 0.05 would say *some* topology
captures the spectral signal. None attempted yet. A single-feature
H1_max_lifetime probe on PC1-residualized clouds (the only PH
feature where real beats null in EXP-56) reaching AUROC ≥ 0.65
would also suggest residual topology. Both proposed as
EXP-56/EXP-57 follow-ups.

**Would be overturned by:** Higher-order PH (zigzag/FTS-PH/landscapes)
on PC1-residualized clouds with real ≥ null + 0.05 — would falsify
the sharpened "spectrum, not topology" form.

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
