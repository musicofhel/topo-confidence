# NOVELTY_AUDIT.md — what's actually novel against the 220-paper graph

Each finding (F-1 … F-10 from FINDINGS.md) is rated against the
research-graph at `bolt://localhost:7688` using `query.py novelty
"<one-line claim>"` and the top neighbours it returns. Two new
end-of-program results are audited separately at the end.

**Three buckets.**
- **`novel`** — no graph paper expresses the same claim; closest
  neighbours are methodologically adjacent at best.
- **`refines-prior-work`** — the literature has the high-level claim;
  ours sharpens, controls, or quantifies it on a specific model/task.
- **`parallel-discovery`** — independent group has the same claim on a
  different model/task; ours adds confirmation rather than novelty.

**A note on graph IDs.** The Neo4j graph carries F-1..F-13 IDs that
have drifted from FINDINGS.md F-1..F-10 (graph F-7 = our F-10 PH-null;
graph F-11 = our F-8 selective; graph F-13 = our ND-10 truncation
artifact). This audit uses **FINDINGS.md numbering**. The drift is a
queue item — reseeding the graph is out of scope here.

**Procedure.** For each finding, the canonical novelty check is:

```bash
cd ~/topo-confidence/research-graph
python query.py novelty "<one-line claim>"
python query.py subgraph F-N --depth 2 | jq .   # context
```

All scores below come from the natural-language novelty index over the
220-paper corpus. Higher score = closer match.

---

## F-1 — Dimensional breathing universality

**Claim.** Residual-stream covariance participation ratio rises during
CoT and collapses at the final answer token; reproduces across 4
architectures (Qwen, Phi, Llama).

**Bucket: refines-prior-work.**

**Closest 3 prior-work papers.**
- `2012.13255` Aghajanyan et al. — *Intrinsic Dimensionality Explains
  Effectiveness of LM Fine-Tuning.* Pretrained LMs operate in a
  low-intrinsic-dim subspace. Score 7.86. **What we add:** the
  *temporal* dynamics during inference-time CoT (rise → peak → collapse),
  not just static intrinsic dim.
- `2511.15210` Lao et al. — *Unveiling Intrinsic Dimension of Texts.*
  Intrinsic dim of text representations as a quality measure. Score
  7.86. **What we add:** content-dependence rejection via random-token
  null (gibberish gives flat PR ≈ 10).
- `2509.26560` Levina-Bickel descendant — *Estimating Dimensionality of
  Neural Representations from Finite Samples.* Bias-corrected PR
  estimator. Score 5.55. **What we add:** the bias correction is a
  *threat* to F-1 (we span P/Q ratios 0.14–0.33, bias-significant
  regime); F-1 has not been re-run with Marchenko-Pastur correction.

**What's actually new in F-1:** the random-token / stream-of-consciousness
gibberish controls (F-5, separate finding) plus 4-architecture
universality. The PR-rises-then-collapses shape itself sits inside an
established intrinsic-dim line.

**Verification command.**
```bash
python query.py subgraph F-1 --depth 2 | jq '.connected_papers[]?.arxiv_id'
```

---

## F-2 — A single L19 prefill direction predicts correctness, AUROC 0.7731

**Claim.** Supervised 1536-d logistic on Qwen-2.5-1.5B L19 prefill
predicts MATH-500 K=1 correctness at OOF AUROC 0.7731. Direction is
unsupervised-identifiable (PCA-PC1 cosine 0.922 with DoM, 85% energy).

**Bucket: parallel-discovery (the AUROC) + novel (the decomposition triangle).**

**Closest 3 prior-work papers.**
- `2509.12886` Zhu et al. — *The LLM Already Knows.* Initial hidden
  state predicts correctness on Qwen-2.5-VL-7B. Score 6.49. **Direct
  parallel-discovery on a vision-language sibling.**
- `2604.15350` — *Spectral Geometry of Thought.* Spectral α of hidden
  activations predicts correctness pre-generation. Score 7.58.
  **Method-differs competitor at the same "pre-generation correctness"
  tier** — we tested it head-to-head (FE749), best-layer α =
  0.7026 (1.5B L28, `fe749-alpha-best-15b`), well below F-2's 0.7731;
  joint α + DoM = 0.7833 (`fe749-joint-alpha-dom-15b`, +1pp).
- `2311.04897` Hernandez et al. — *Future Lens.* A single hidden state
  predicts multiple future tokens. Score 7.58. **Method-differs
  ancestor** — frames the "single hidden state knows" question; F-2
  is the correctness analog.
- (Honourable mention: `2510.18147` — *LLMs Encode Problem Difficulty*
  — Qwen2.5-Math-1.5B prefill encodes difficulty ρ=0.88, score 6.14.
  Strong corroborator if difficulty ≈ correctness on MATH-500.)

**What's actually new in F-2:** the **decomposition triangle**
(directional ceiling at 2-feat, cov-spectrum lift to 0.7928 — see end
of audit) and the **CAST PC1 ≈ DoM** unsupervised-identifiability
result. The headline AUROC itself is parallel-discovery against
`2509.12886`.

---

## F-3 — Prefill and final-token DoM are structurally orthogonal

**Claim.** cos(prefill_DoM, final_DoM) = −0.0617 raw, ≈0 after
Song-Zhong residualization, −0.0315 after causal whitening. NC3
alignment max |cos| with answer-token unembed rows = 0.067/0.128 (both
below 0.30 threshold).

**Bucket: refines-prior-work.**

**Closest 3 prior-work papers.**
- `2604.22271` — *How LLMs Detect and Correct Their Own Errors.*
  Identifies PANL token as a second-order confidence signal orthogonal
  to logprobs (cos = 0.007). Score 8.95. **Closest analog** — they
  find a *post-hoc* orthogonal confidence signal; ours is the
  *pre-hoc* analog (prefill DoM ⊥ final-token DoM).
- `2602.01893` — *Geometric Analysis of Token Selection in MHA.*
  Attention head specialization (Retriever / Mixer / Reset) in
  residual/value-state space. Score 8.25. **Method-differs ancestor**
  — could mechanistically explain why prefill and final live in
  different subspaces (different heads).
- `2310.04861` — *Uncovering Hidden Geometry: Disentangling Position
  and Context.* Decomposes hidden states into position/context/residual.
  Score 5.64. **Method we used** (Song-Zhong residualization in FE115).

**What's actually new in F-3:** the *structural* orthogonality
(survives μ + pos_t + ctx_i removal, survives causal whitening,
survives NC3 control). Most steering literature assumes directions are
position-stable; F-3 gives the first quantitative measurement of how
fast a correctness direction rotates within a single trajectory
(cos stays in [0.00, 0.19] through token ~50 on MATH-500).

---

## F-4 — Correct trajectories collapse harder than incorrect at final token

**Claim.** Final-token PR ratio correct/incorrect: 7B = 0.384, 1.5B =
0.509, Phi-3 / Llama-3.2 same direction. Bootstrap rejects ratio > 1.

**Bucket: refines-prior-work.**

**Closest 3 prior-work papers.**
- `2405.17767` — *Linguistic Collapse.* Neural collapse at final layer
  scales with model size. Score 9.77. **Strongest match** — they have
  scale-monotonic collapse; F-4 sharpens it to *correct vs incorrect*
  asymmetry within each scale.
- `2505.00127` — *Between Underthinking and Overthinking.* Incorrect
  responses are systematically longer. Score 11.70. **Length confound
  parallel** — F-9 length result is downstream of this; F-4 is about
  PR not length.
- `2510.01105` — *Geometric Properties of Neural Multivariate
  Regression.* Intrinsic-dim and neural-collapse analysis. Score 5.18.
  Methodologically adjacent.

**What's actually new in F-4:** the *correct vs incorrect* asymmetry
on top of the established scale-monotonic collapse line. The bootstrap
test is rigorous (P(ratio > 1) = 0.000 on Qwen 7B and 1.5B).

---

## F-5 — Breathing is content-dependent, not AR-mechanics

**Claim.** Random tokens give flat PR ≈ 10 across all positions.
Stream-of-consciousness produces a muted, different-shape curve.
MATH-500 produces the canonical rise-to-30-collapse.

**Bucket: novel** (within the dim-breathing literature; the
null-rejection design is uncommon).

**Closest 3 prior-work papers.**
- `2511.15210` — *Intrinsic Dimension of Texts.* Score 7.98. Studies
  intrinsic dim across text genres but does not run a random-token null.
- `2509.26560` — *Estimating Dim from Finite Samples.* Score 5.55.
  Methodological threat (bias correction, see F-1) but no content-null.
- `2604.18805` — *AI Scientists Produce Results Without Reasoning
  Scientifically.* Score 4.41. Motivates the null-control framing
  (LLM agents ignore refutation evidence in 68% of traces; we make our
  null test explicit).

**What's actually new in F-5:** the random-token null rejection,
n=20 per condition. Modest sample but clean rejection (random PR 11.32
at pos1, 8.96 at final, vs MATH 13.74 at pos1, 30.55 at pos50, 22.19
at final).

---

## F-6 — 7B prefill PR inversion driven by incorrect-group concentration

**Claim.** Qwen-7B prefill correct-group PR > incorrect-group PR
(ratio 1.377, CI [1.226, 1.757]); not in 1.5B (0.946) or BBH (≤1.01).
Mechanism: incorrect-group concentration at easy difficulty levels
(Level 2 ratio 3.98, Level 5 ratio 1.11).

**Bucket: novel** (the inversion phenomenon and the difficulty-level
decomposition).

**Closest 3 prior-work papers.**
- `2510.18147` — *LLMs Encode Problem Difficulty.* Same model family
  (Qwen2.5-Math-1.5B), prefill encodes difficulty ρ=0.88. Score 6.14.
  **Possible mechanism for the level-driven decay** but on the 1.5B,
  not the 7B; F-6's 7B-specific story remains unexplained.
- (Note: novelty-index returned F-1 internal as second match;
  graph has no other tight neighbour for the inversion claim.)

**What's actually new in F-6:** the inversion itself (rigorous
bootstrap, BBH negative control) and the level-stratified decomposition.
Mechanism-strength MODERATE. Not in any prior paper.

---

## F-7 — D-bucket has collective geometric signature, dissolves per-problem

**Claim.** 36/500 MATH-500 problems are K=1-right K=8-majority-wrong.
Their *group* prefill PR is 14.49 (lowest of any bucket); their
*per-problem* local PR is 12.84 (indistinguishable from A/B/C). The
cluster exists at group scale but dissolves at k-NN scale.

**Bucket: novel** (the local-vs-global distinction is uncommon).

**Closest 3 prior-work papers.**
- `2210.00069` von Rohrscheidt — *Topological Singularity Detection at
  Multiple Scales.* Score 11.28. **Closest method match** — multi-scale
  local intrinsic dim and singularity detection; the right tool for
  H-7's density-based D-bucket follow-up.
- `2604.24712` — *When Prompt Under-Specification Improves Code
  Correctness.* Score 11.24. **Sampling-side analog** — over-specified
  prompts trigger memorized-but-wrong solutions; their Table 8
  taxonomy maps to D-bucket pathology types.
- `2401.10474` — *LDReg: Local Dimensionality Regularized SSL.* Score
  8.46. Local-vs-global dim framing is methodologically adjacent.

**What's actually new in F-7:** the K=1-right K=8-majority-wrong
operational definition of "pathological," and the group-vs-local
geometric mismatch. Density-based outlier methods (LOF, isolation
forest) on D's 36 activations were not tried — H-7 in HYPOTHESES.md.

---

## F-8 — Selective prediction via prefill DoM, 71.6% at 50% coverage

**Claim.** Refuse-and-spend gating on Qwen-1.5B MATH-500 at coverage
0.5 = 71.6% accuracy on answered (vs 48.6% unconditional, 49.4%
random-refuse). Average K=2.5.

**Bucket: refines-prior-work** (selective prediction is a heavy
subliterature; the prefill-DoM-as-gate specifics are a refinement).

**Closest 3 prior-work papers.**
- `2402.15610` — *Selective "Selective Prediction": Reducing
  Unnecessary Abstention in VL Reasoning.* Score 8.06. **Direct
  method-differs comparison.**
- `2502.06884` — *Learning Conformal Abstention Policies.* Score 7.17.
  RL-tuned conformal thresholds — alternative gating mechanism for
  the same selective-prediction problem.
- `2502.05911` — *GRAIT: Gradient-Driven Refusal-Aware Instruction
  Tuning.* Score 6.82. Training-time refusal tuning. Method-differs.

**What's actually new in F-8:** the prefill-only signal (no generation
required for the gate decision) at 1.5B scale on MATH. Most selective-
prediction papers gate post-generation. The +22pp lift over random at
matched coverage is concrete and reproducible.

---

## F-9 — Single-layer L19 DoM matches CoE-60 on cross-domain transfer

**Claim.** Symmetric MATH↔BBH transfer: CoE-60 = 0.716, single-layer
L19 DoM = 0.720. Δ = +0.004. Sixty trajectory features add no
cross-domain signal over a 1-d direction at the right layer.
Spectral α at L28 is also weaker (0.7026/0.7128 1.5B/7B vs DoM 0.7731).

**Bucket: novel** (this specific parsimony refutation).

**Closest 3 prior-work papers.**
- `2410.13640` Wang et al. — *Chain-of-Embedding (CoE).* Score 12.29.
  **The thing being refuted.** Original CoE paper claims CoE captures
  trajectory information; F-9 shows single-layer DoM is parsimony-
  equivalent on cross-domain transfer.
- `2402.03744` — *INSIDE / EigenScore.* Score 8.43. Methodological
  cousin (multi-sample covariance eigenvalues for hallucination); on
  our task, single-pass single-layer DoM is enough.
- `2604.15350` — *Spectral Geometry of Thought* (HT-SR α). Score 5.79.
  Direct head-to-head was FE749 — α ≈ chance at peak DoM layer
  (`fe749-alpha-l19-15b` = 0.5225); both probes are roughly orthogonal.

**What's actually new in F-9:** the explicit head-to-head refutation.
CoE within-domain at 1024-tok labels has not been re-tested (H-6) — if
CoE stays at 0.80 within-domain while DoM is 0.77, the within-domain
claim tilts back. Cross-domain claim is the strong form.

---

## F-10 — Topology summaries indistinguishable from Gaussian null

**Claim.** PH summaries on residual streams add no signal beyond
covariance, and on PC1-residualized clouds add **negative** signal
(real 0.6884 vs matched-cov null 0.7628, gap −0.074 — the
pre-registered overturning condition tested directly and failed).
Three orthogonal cloud constructions (raw, zigzag-residualized,
PC1-residualized) agree.

**Bucket: refines-prior-work** for the high-level "PH ≈ Gaussian null"
direction; **novel** for the F-10-strengthening-on-residualization
result.

**Closest 3 prior-work papers.**
- `2309.11028` — *The Topology and Geometry of Neural Representations
  (tRSA).* Score 9.62. **Closest method match** — proposes
  noise-robust geo-topological statistics; we'd recommend tRSA as the
  next instrument to test on residualized clouds (HYPOTHESES.md
  follow-up).
- `1802.04443` — *Characterizing NN Capacity using Algebraic Topology.*
  Score 9.11. Method ancestor; uses topology on data manifolds, not
  residual streams.
- `2205.09630` — *Acceptability Judgements via Attention-Map Topology.*
  Score 8.46. **Counterexample** — TDA on attention maps gives signal
  where TDA on residuals hits Gaussian null. Suggests F-10 is residual-
  stream-specific, not a general "TDA on transformers" failure.
- (Honourable mention: `2512.15285` — *Topological Metric for
  Unsupervised Embedding Quality.* Score 7.15. PH-based unsupervised
  quality; a candidate for H-12 label-free probes.)

**What's actually new in F-10:** the *strengthening on residualization*.
Most PH-vs-null papers test on raw point clouds; F-10 tests three
orthogonal cloud constructions and shows the gap goes more negative,
not less, after PC1 residualization. Pre-registered overturning
condition tested directly and failed by 12.4 pp in the wrong direction.

---

## End-of-program new results — outside the F-1..F-10 numbering

These are not yet promoted to F-N in FINDINGS.md. They sit inside the
F-2 / F-10 evidence sections as the FE291 cascade follow-ups (FE880,
FE881, FE882). They are the strongest novelty candidates of the
program and the load-bearing claims for any future paper.

### N1 — Per-problem cov-spectrum probe beats every directional probe

**Claim.** Top-20 log-eigvals of each problem's L19 PC1-residualized
covariance, fed to 5-fold OOF logistic, gives AUROC **0.7928**
(`pca-pc1resid-cov-spectrum-top20-auroc`). Beats supervised DoM
0.7679 (+2.49 pp), 2-feat directional ceiling 0.7856 (+0.7 pp), full
1536-d L2-reg ceiling 0.7847 (+0.8 pp), PH-on-same-clouds real probe
0.6884 (+10.4 pp), matched-cov Gaussian null on PH 0.7628 (+3.0 pp).

**Bucket: novel.**

**Closest 3 prior-work papers.**
- `2510.01591` — *CLUE: Non-parametric Verification from Experience
  via Hidden-State Clustering.* Score 8.27. **Closest method match** —
  uses hidden-state clustering for selective prediction; we use the
  per-problem covariance spectrum. Both go beyond a single DoM
  direction; CLUE is non-parametric, ours is supervised.
- `2603.24787` — *ReLope: KL-Regularized LoRA Probes for Multimodal
  LLM Routing.* Score 5.66. Probe-based routing alternative to single-
  layer DoM. Method-differs.
- `2402.03744` — *INSIDE / EigenScore.* Score 8.43 (from F-9). Method
  ancestor — multi-sample covariance eigenvalues for hallucination.
  **The closest framing** in the literature; we extend to per-problem
  *single-pass* eigval extraction after PC1 residualization. EigenScore
  uses multi-sample variance; ours uses single-sample shape-of-spectrum.

**Why this is the strongest novelty:** the per-problem residualized
covariance spectrum is, to our knowledge, not a probe in any existing
paper at the time of writing. The +0.7 pp lift over the directional
ceiling is correlationally orthogonal to direction (FE882 confirms via
full 1536-d L2-reg ceiling). The signal is in rank 2–20 of the
per-problem residualized cov, not in any single eigenvalue.

**Caveat.** Supervised. Beating the directional ceiling
supervised-vs-supervised says "there's information you can't capture
linearly," not "the model uses this information." The causal companion
test (rank-truncate per-problem PC1-residualized cov before
continuation) is unrun.

### N2 — F-10 strengthens after PC1 removal

Already covered in F-10 above. Listed here because it's a *new*
result, not a sharpening of an old one. The relevant graph neighbours
are the same as F-10.

### N3 — CAST PC1 ≈ supervised DoM (unsupervised-identifiable correctness direction)

**Claim.** Unsupervised PCA-PC1 of L19 prefill matches supervised DoM
at cosine 0.9216 (`pca-dom-pc1-cosine`); CAST class-mean PC1 matches
within 0.0001. 85% of unit-DoM energy in PC1; 97.6% in top-10 PCs.
Unsupervised PC1 single-feature OOF AUROC = 0.7458; supervised DoM =
0.7679. Labels barely matter for direction discovery.

**Bucket: refines-prior-work / parallel-discovery on Qwen-1.5B.**

**Closest 3 prior-work papers.**
- `2510.02956` — *Confidence and Dispersity as Signals: Unsupervised
  Model Evaluation.* Score 7.64. **Closest direction** — label-free
  correctness signal aggregation. We add: the dominant residual-stream
  variance direction *is* the supervised correctness direction at
  this layer.
- `2312.03813` — *Mean-Centring for Activation Steering.* Score 6.75.
  Methodological neighbour — mean-centred steering vectors are
  algebraically close to DoM.
- `2505.17306` — *Refusal Direction is Universal Across Languages.*
  Score 6.33. Single residual-stream direction controls behaviour
  across conditions; same shape of finding as F-2 + N3.

**What's actually new in N3:** the *equivalence*. Most steering /
probing work uses supervision. F-2 + N3 say: on this layer of this
model, you don't need labels to find the correctness direction.

### N4 — The 1024-tok rebuild itself

**Claim.** The 20.8% MATH-500 baseline was a `max_new_tokens=256`
truncation artifact. At 1024 tokens, 1.5B accuracy is 48.6%
(`baseline-1.5B-acc`). Every Pathway-1..10 success criterion was
calibrated against imaginary headroom.

**Bucket: novel** — not as a *result*, but as an explicit *retraction*
artifact. Public retractions of TDA-LLM numbers tied to specific
generation-budget caps are uncommon in the literature.

**Closest graph neighbours.** Graph F-13 records the retraction
internally; no external paper in the graph explicitly calls out
truncation-budget label confounds for residual-stream correctness
probes. This is a methodological novelty (the retraction template),
not a scientific one.

---

## Synthesis ranking

If asked "what's the most novel thing the program produced?":

1. **Cov-spectrum probe (N1).** Strongest correlational result; novel
   probe family; needs causal companion to land.
2. **F-10 strengthening on residualization (N2).** Three independent
   cloud constructions + pre-registered overturning condition tested
   and failed. Hard to overturn.
3. **CAST PC1 ≈ DoM equivalence (N3).** Reframes F-2 from "supervised
   probe finding" to "dominant residual-stream variance direction."
4. **F-7 D-bucket local-vs-global mismatch.** Genuinely surprising
   geometric phenomenon; needs density-based methods to act on.
5. **F-9 CoE-DoM parsimony refutation.** Concrete, narrow, but a clean
   "smaller probe is enough" result.
6. **F-5 content-dependent breathing null.** The random-token control
   is cleanly designed.
7. **F-3 prefill-final orthogonality** (refinement of well-trodden
   literature; the *measurement* of how fast directions rotate is the
   contribution).

Everything else is refinement or parallel-discovery against the
existing literature.

---

## Subgraph commands for context

```bash
cd ~/topo-confidence/research-graph
for fid in F-1 F-2 F-3 F-4 F-5 F-6 F-7 F-8 F-9 F-10; do
    echo "=== $fid ==="
    python query.py subgraph "$fid" --depth 2 | jq '.connected_papers[]?.arxiv_id' 2>/dev/null
done
```

Re-run `python query.py novelty "<one-line claim>"` to refresh any
finding's neighbour set after new papers land in the graph.
