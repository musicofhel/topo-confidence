# SYNTHESIS.md — what three weeks of experiments told us

A practitioner briefing. Read in 10 minutes, walk away knowing what's true,
what was overturned, and what's new. Not a paper.

Every quantitative claim below is registered in `validate_claims.py` — the
matching claim ID is shown in parentheses so you can grep for the regen.
Numbers are 1024-tok labels (Qwen-2.5-1.5B-Instruct on MATH-500 K=1) unless
explicitly marked `[256-tok]` (superseded; included only where the
contrast matters).

---

## 1. What survived

### 1.1 The L19 prefill direction predicts correctness — supervised AUROC 0.7731

A 1536-dim logistic regression on the L19 residual stream at the **last
prompt token** (before any generation) predicts MATH-500 K=1 correctness
at OOF AUROC = **0.7731** (`prefill-dom-auroc`). The mass-mean DoM
projection matches at **0.7711** (`fe110-supervised-dom-oof`). LEACE
linear erasure collapses the signal to chance (`fe101-auroc-erased` =
0.5000), so the signal is genuinely linear at L19.

The peak is at L21 (0.7718) but L19 sits within the ±0.005 band
(`fe145-l19-within-band`); the layer choice is not knife-edge.

**One-sentence mechanism.** The model "knows" before it generates whether
this prompt is in its competence — and that knowledge is one direction in
the residual stream.

### 1.2 Selective prediction — 71.6% accuracy at 50% coverage

DoM-gated refuse-and-spend: keep top-half of MATH-500 by prefill DoM
score, answer Q4 of those at K=1, Q3 at K=4, refuse the bottom-half.
Accuracy on the answered half = **0.716** (`refuse-prefill-acc`) at
average K=2.5, vs unconditional K=1 = **0.486** (`baseline-1.5B-acc`)
and random-refuse at the same coverage = **0.494** (`refuse-random-acc`).

**One-sentence mechanism.** Spend compute where the prefill says it will
land. Save it where the prefill says it won't.

### 1.3 Prefill and final-token directions are structurally orthogonal

cos(prefill_DoM, final_DoM) = **−0.0617** raw on the 1024-tok cache
(`fe115-cos-raw`), **0.0008** after Song-Zhong residualization
(`fe115-cos-resid`, μ + pos_t + ctx_i removed), **−0.0315** after
Park-Choe-Veitch causal whitening (`fe136-cos-whitened`). Max |cos|
with top-50 answer-token unembed rows is 0.067 (prefill,
`fe244-prefill-max-cos`) / 0.128 (final, `fe244-final-max-cos`) —
both well below the NC3 collapse threshold 0.30.

**One-sentence mechanism.** "Can I solve this?" and "did I solve this?"
live in geometrically unrelated subspaces — not just different magnitudes
of the same axis. This is what kills fixed-vector steering (§2.3).

### 1.4 Dimensional breathing is universal across architectures

Participation ratio of the L19 residual-stream covariance starts low at
prefill (~14, `gib-math-pos1`), rises sharply through CoT (peak ~30 by
position 50, `gib-math-pos50`), and collapses at the final answer token
(~22, `gib-math-final`; correct trajectories collapse harder than
incorrect — Qwen-1.5B final ratio correct/incorrect = **0.509**,
`inv-15B-final-ratio`; Qwen-7B = **0.384**, `inv-7B-final-ratio`).

Reproduces on Phi-3-mini (acc 0.448, `phi3-acc`) and Llama-3.2-1B (acc
0.252, `llama-acc`). Random-token gibberish gives flat PR ≈ 10 across
positions (`gib-random-pos1` = 11.32, `gib-random-final` = 8.96) —
breathing is content-dependent, not AR-mechanics.

**One-sentence mechanism.** Reasoning expands then commits. Random tokens
just diffuse.

### 1.5 The decomposition triangle — F-2's signal factors into three additive pieces

This is the cleanest result of the program. The L19 prefill correctness
signal decomposes as:

| Probe | OOF AUROC | Δ vs DoM 0.7679 | Claim ID |
|---|---|---|---|
| 1-d DoM (supervised, matched protocol) | 0.7679 | — | `pca-dom-oof-auroc` |
| 2-feat (PC1, PC9) | 0.7856 | +1.77 pp | `pca-pc1-pc9-oof-auroc` |
| Full 1536-d L2-reg (best C=0.001) | 0.7847 | +1.68 pp | `pca-full-lr-best-auroc` |
| Top-20 log-eigvals of PC1-residualized cov | **0.7928** | **+2.49 pp** | `pca-pc1resid-cov-spectrum-top20-auroc` |

The full 1536-d L2-reg ceiling sits at **0.7847 ≈ 2-feat 0.7856 within
fold noise** — so the directional ceiling at L19 saturates at two
features. The cov-spectrum lift to **0.7928** is therefore *genuinely
second-order*, not an under-regularized linear-directional artifact.

F-2 factors into three additive pieces:

- **(a) PC1 mean shift.** DoM ≈ 0.92·PC1 (`pca-dom-pc1-cosine` = 0.9216),
  85% of unit-DoM energy in PC1 (`pca-dom-energy-top10-share` = 0.976
  for top-10 PCs; PC1 alone carries the bulk). PC1 carries 14.7% of
  total L19 variance (`pca-pc1-var-share`) — a plurality.
- **(b) PC9 trim.** PC9 alone hits AUROC 0.658 at variance share 2.5%;
  the (PC1, PC9) re-weighting beats DoM by 1.8 pp and matches the full
  1536-d L2-reg ceiling. PC9 is the second-largest term in the DoM-PC
  basis decomposition (DoM coefficient 0.225, energy 5.0%).
- **(c) Per-problem residual second-order structure.** Top-20 log-eigvals
  of each problem's PC1-residualized covariance lifts to 0.7928. The
  signal is in the *shape* of the covariance spectrum tail (rate of
  eigenvalue decay), not in any single direction.

**Why this is load-bearing.** The unsupervised PCA-PC1 *recovers DoM*
(`pca-best-pc-index` = PC1, `pca-best-pc-auroc` = 0.7457; CAST class-mean
PC1 matches within 0.0001, `pca-cast-pc1-auroc` = 0.7458). Anyone running
PCA on Qwen-1.5B prefills finds the correctness direction without labels.
F-2 stops being "we trained a probe and it worked" and becomes "the
correctness signal occupies the dominant variance direction at L19, plus
a small directional trim, plus per-problem second-order shape that no
linear probe can recover."

---

## 2. What was overturned

### 2.1 The "topological homology" framing (F-10)

The repo is named topo-confidence. Persistent homology was the original
framing. It does not survive controls.

- **Raw L19 clouds.** 5 V-R PH features real OOF AUROC = 0.690 vs
  rank-matched Gaussian null 0.693 (`p9-gaussian-null-real`,
  `p9-gaussian-null-null`). Gap −0.003.
- **Song-Zhong residualized clouds.** Real PH 0.6958 vs null 0.7624
  (`fe116-real-ph-resid`, `fe116-null-ph-resid`). Gap **−0.067**.
- **PC1-residualized clouds (the pre-registered overturning condition
  for F-10).** Real PH 0.6884 vs matched-cov Gaussian null 0.7628
  (`pca-pc1resid-ph-real-auroc`, `pca-pc1resid-ph-null-auroc`). Gap
  **−0.074**. The pre-registered overturning condition was real ≥
  null + 0.05 — tested directly and failed by 12.4 pp in the wrong
  direction (`pca-pc1resid-ph-gap`).

PH summaries integrate over the eigenvalue distribution. The post-PC1
correctness signal lives in the *shape* of that distribution
(`pca-pc1resid-cov-spectrum-top20-auroc` = 0.7928). Gaussianizing the
covariance preserves the spectrum; PH summaries do not. F-10 sharpens
to: **topology adds no signal beyond covariance, and on residualized
clouds it adds negative signal.**

### 2.2 The 256-tok baselines (ABC-44 0.7961, MATH-500 20.8%)

`max_new_tokens=256` truncated two-thirds of correct CoT trajectories
before `\boxed{...}` was emitted. The 1.5B MATH-500 K=1 baseline jumped
from 20.8% to **48.6%** (`baseline-1.5B-acc` at 1024 tokens) — 2.3× off.
Every Pathway-1..10 success criterion calibrated against the 20.8% number
was calibrated against imaginary headroom. Track A's "+11 net gain" in
P4, the 0.818 / 0.754 cross-scale verifier asymmetry from P10v1, the
ABC-44 0.7961 headline — all 256-tok artifacts. The corrected 7B-as-
verifier transfer is **0.717** (no advantage over 1.5B self at 0.719,
PROJECT_RECORD §1d ND-6).

### 2.3 Fixed-vector steering (P10 v1 / v2 E1)

The original P10 v1 plan was ITI-style fixed-vector steering using a
final-token DoM. It would have produced nothing. cos(prefill, final) =
0.046 raw, ≈0 after Song-Zhong residualization. The DoM direction
**rotates continuously through generation** — cosine with final-token
DoM stays in [0.00, 0.19] through token ~50, only reaches 0.37 by
position 200. The pre-cached Pathway-2 steering vector has cos ≈ 0.05
with the current L19 DoM (PROJECT_RECORD §1d ND-5). A final-token-fit
steering vector injected at position 15 is energy in approximately the
wrong subspace.

Honest steering would require a per-position DoM bank — never built.

### 2.4 7B-as-verifier (P10 v1 Direction A)

Killed by the 256-tok correction. At 1024 tokens, 7B → 1.5B transfer is
0.717, 1.5B → 1.5B self is 0.719. No cross-scale verifier advantage.
E4 distillation deferred (PROJECT_RECORD §1d ND-6).

### 2.5 Softmax confidence at PANL is anti-calibrated

At the position-after-N-Lines (`\boxed{` neighbourhood in 1.5B-Instruct
generation), the top-1 softmax probability for predicting correctness
gives AUROC = **0.4395** (`fe23-softmax-conf`) — *below chance*. Lexical
hedging composite ConCISE c_hat lands at 0.589 (`fe455-c-hat`); P(' sure')
alone is 0.293 (`fe455-p-sure`, anti-predictive). DoM at L19 (0.7731) does
real work that no softmax-derived signal reaches on this stack.

### 2.6 CoE (60-dim trajectory features) doesn't beat single-layer DoM

Symmetric MATH↔BBH transfer: CoE-60 = 0.716, single-layer L19 DoM =
**0.720** (`stage5-symmetric`, `stage5-verdict-vs-coe` = +0.004). Sixty
trajectory features add no cross-domain lift over one direction at the
right layer. Spectral α (HT-SR) is also weaker — best-layer 1.5B α =
0.7026 (`fe749-alpha-best-15b`) at L28; joint with DoM = 0.7833
(`fe749-joint-alpha-dom-15b`, +1pp). DoM is the parsimony winner.

### 2.7 Monotonic compute gating at mid-confidence

P10v2 E2's plan was "low-confidence → K=8, high-confidence → K=1." It
fails because the recoverable B-bucket lives at *middle* prefill score,
not bottom. Middle-heavy at avg K=4.5 gets 0.526 (`middle-heavy-acc`),
neg-seq-len top-heavy gets 0.542 — but neg-seq-len is post-hoc (requires
running generation). Monotonic prefill-gating doesn't beat random refuse
on the answered half (PROJECT_RECORD §1d ND-7).

### 2.8 D-bucket detection from local features

3-feature logreg on {prefill DoM, per-problem local PR, seq-len} hits
D-recall **0.417** (`exp2b-logreg-Drecall`) — *below* the 0.486
class-prior baseline. Dropping `prefill_lpr` changes macro-F1 by 0.000.
The D-bucket has a *collective* group-PR signature (14.49 vs A/B/C at
18.7/16.6/20.0, F-7) but it does not survive per-problem k-NN
localization. Density-based methods (LOF, isolation forest) on D's 36
activations were not tried (PROJECT_RECORD §1d ND-8).

---

## 3. What is new at the end of this work

### 3.1 The cov-spectrum probe — strongest L19 correctness probe at any tier

Top-20 log-eigvals of each problem's PC1-residualized L19 trajectory
covariance, fed to 5-fold OOF logistic, gives AUROC **0.7928**
(`pca-pc1resid-cov-spectrum-top20-auroc`, FE881). Beats supervised DoM by
**+2.49 pp**, beats the 2-feat directional probe by +0.7 pp, beats the
full-1536-d L2-reg ceiling by +0.8 pp, beats the FE880 PH-on-same-clouds
real probe by +10.4 pp, beats the matched-cov Gaussian null on the same
PH features by +3.0 pp.

Stable across K ∈ [10, 50] within ±0.001 AUROC. CV5 std 0.027. The
single-feature baselines locate the signal: log_eigval_3 alone hits
0.7488, log_eigval_2 = 0.6925, log_eigval_1 = 0.5223 (mostly cloud-size
T_i). Signal is distributed across rank 2–20, not in any single
eigenvalue.

**Why this is novel.** It's a *per-problem* second-order structure
extracted *after* projecting out the dominant supervised direction. The
closest neighbours in the literature are (a) SAE feature-sparsity
papers, (b) Hewitt-style structural probes, (c) IRT-on-residual-streams.
None we've found use per-problem residualized covariance. Confirm
through `query.py novelty` — see NOVELTY_AUDIT.md.

### 3.2 F-10 strengthens, not weakens, on residualized covariance

The pre-registered overturning condition for F-10 was "PH features on
PC1-residualized clouds with real ≥ null + 0.05." The actual gap is
−0.074 — failed by 12.4 pp in the wrong direction. The matched-cov
Gaussian null *jumped* from 0.693 (raw F-10) to 0.7628 (residualized)
because PC1 was absorbing per-problem cov-spectrum variation
correctness-uniformly, and removing it exposed the residual covariance
spectrum that the Gaussian null preserves and PH does not.

This is a sharper, more falsifiable form of F-10: not "PH ≈ Gaussian null
on the raw cloud" but "PH adds *negative* signal beyond covariance, once
the dominant correctness direction is removed." Three orthogonal cloud
constructions (raw, zigzag-residualized, PC1-residualized) now agree.

### 3.3 CAST PC1 ≈ supervised DoM — the supervised probe is unsupervised-identifiable

Cosine of unsupervised PCA-PC1 with supervised DoM = **0.9216**
(`pca-dom-pc1-cosine`). 85% of unit-DoM energy lies in PC1; 97.6% in the
top-10 PCs (`pca-dom-energy-top10-share`). CAST class-mean PCA-PC1
matches the unsupervised PC1 within 0.0001 cosine (`pca-cast-pc1-auroc`
= 0.7458 vs unsupervised PC1 = 0.7457).

What this changes: F-2 was originally framed as a supervised probe
finding ("we trained a 1536-d logistic and it worked"). The supervised
structure is the dominant variance direction. *Anyone running PCA on
Qwen-1.5B prefills would find it without labels.* The supervised
single-feature DoM AUROC (0.7679) is only +0.022 over the unsupervised
PC1 (0.7458) — labels barely matter for direction discovery; they matter
for the PC9 re-weighting (lifts to 0.7856).

### 3.4 Non-Euclidean PH and layer-wise PH-168 are dead

Best non-Euclidean variant (diffusion, effective resistance, cosine,
DTM): 0.774 [256-tok] vs Euclidean baseline 0.7961 [256-tok]
(PROJECT_RECORD §1d ND-1). Layer-wise PH-168 holdout: 0.6463 [256-tok]
vs single-layer DoM 0.7705 (1024-tok) (`p8-layerwise-ph` =
0.646). Neither approach should be revisited unless a fundamentally
different metric or layer-aggregation scheme emerges.

### 3.5 LEACE linear erasure perfectly collapses F-2

`fe101-auroc-raw` = 0.7705, `fe101-auroc-erased` = 0.5000, collapse =
**0.2705 pp** (`fe101-collapse-pp`). The signal is genuinely linear at
L19 — a single 1-d affine subspace, removable by LEACE without
degrading any other linear function of the activations. This is a
strong constraint on causal-intervention designs: ablating PC1 alone
should be sufficient to remove the signal, no nonlinear surgery needed.

---

## 4. What we still don't know

Eight honest open questions, in rough order of decisiveness for the program.

1. **Is F-2 causal?** All probes are correlational. The cov-spectrum
   probe beats the directional ceiling supervised-vs-supervised — that
   says "there's information you can't capture with linear directions on
   raw activations," not "the model uses this spectral information." The
   load-bearing test is rank-truncating the PC1-residualized covariance
   per-problem before continuation; measure correctness drop. None of
   FE214 / FE269 / FE283 has run.

2. **Does F-2 cross architectures?** F-2 is one model (Qwen-2.5-1.5B-
   Instruct). Phi-3-mini and Llama-3.2-1B replicated dimensional
   breathing (F-1), but DoM AUROC was not extracted on either —
   `exp1_cross_model` cache exists; the analysis is a few-hour CPU run.
   Listed in `pending_controls` for F-2.

3. **Does F-2 cross benchmarks?** F-2 is MATH-500. Within-topic AUROC
   averages 0.7143 (`fe299-within-mean`), worst 0.6000 (Number Theory,
   `fe299-within-worst`). LOCO-CV by subject mean 0.7427
   (`loco-auroc-mean`), worst 0.6032 (`loco-auroc-worst`). Cross-domain
   MATH→BBH = 0.747 (`stage5-math-to-bbh-pooled`); BBH→MATH = 0.693
   (`stage5-bbh-to-math`). Suggests the signal is mathematical-reasoning-
   specific. GSM8K (0.615 [256-tok]) and AIME have not been re-run at
   1024 tok.

4. **Why does the 7B prefill PR ratio invert (1.377) but 1.5B doesn't
   (0.946)?** F-6's mechanism story (incorrect-group concentration at
   easy difficulty) is moderate-strength, not strong. The 3B intermediate-
   scale test is unrun.

5. **Is dimensional breathing answer-vocabulary-driven at the final
   token?** The strongest criticism of F-1/F-4 (PERSPECTIVES). Three
   tests would address it: penultimate-token PR, residual-minus-answer-
   embedding PR, large-answer vs small-answer subgroups. None has run.

6. **What is PC9?** It's the only mid-rank PC with real correctness
   signal (0.658 single-feature AUROC at variance share 2.5%). Could be
   a difficulty axis, a length axis (FE448 length-alone AUROC = 0.7986),
   or something else. The decomposition triangle uses it but doesn't
   name it.

7. **Does dimensional breathing need a long trajectory?** The no-CoT
   control (median 2 tokens) was inconclusive. Short-CoT (target 20–40
   tokens) is the next test (OQ-1).

8. **The 12 untriaged Discord-admitted papers in Neo4j.** 1 visible via
   `query.py pending` (2604.28119 "Do Sparse Autoencoders Capture
   Concept Manifolds?"), 11 with null status the CLI filter misses.
   `bash triage_pending.sh` clears the queue once status is reconciled.
   Defer until after NOVELTY_AUDIT.md and APPLICATIONS.md land.

---

## 5. The story in two sentences

Three weeks of experiments testing whether residual-stream geometry
predicts LLM correctness. The "topological homology" framing was
overturned (PH adds negative signal beyond covariance, F-10); what
survived is a single L19 prefill direction (DoM ≈ PC1) that predicts
correctness at AUROC 0.7731, factors cleanly into directional + spectral
pieces with the cov-spectrum probe lifting to 0.7928, and powers a
selective-prediction stack that hits 71.6% answered accuracy at 50%
coverage — open-ended for cross-architecture, cross-benchmark, and
causal validation.

---

## 6. Source map

- Headline numbers: `pathway11_h100/prefill_gated_compute/results.json`
- Decomposition triangle: `pathway11_h100/pca_covariance/results.json`
  (FE291, FE882), `pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json`
  (FE881), `pathway11_h100/ph_residuals/pc1_resid_results.json` (FE880)
- F-3 orthogonality: `pathway11_h100/song_zhong/results.json` (FE115),
  `pathway11_h100/phase3_corroborators/results.json` (FE136/FE244)
- LEACE erasure: `pathway11_h100/leace_erasure/results.json` (FE101)
- Breathing universality: `pathway11_h100/exp1_cross_model/`,
  `pathway11_h100/gibberish_control/`, `pathway11_h100/no_cot_control/`
- Selective prediction: `pathway11_h100/prefill_gated_compute/results.json`
  (refuse-and-spend block)
- Graveyard: `PROJECT_RECORD.md` §1d (ND-1 through ND-10)
- Reflective framing: `PERSPECTIVES.md`
- Per-claim regen: `python validate_claims.py` — all 172 internal claims
  back-checked against committed JSONs, 127/127 Tier-1 regen pass.
