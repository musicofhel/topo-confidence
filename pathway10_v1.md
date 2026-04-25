# Pathway 10: CoE-headline paper + new directions (v1)

## What this prompt is

Pathway 9 **CLOSED** with a decisive PH→CoE pivot (`pathway9/SUMMARY.md §7`):

- ABC-44 PH on MATH-500: 0.7961 headline → **0.7459** after raw-length deconfound.
- The 5 raw PH summary features are indistinguishable from an empirical-covariance Gaussian null (0.690 vs 0.693).
- Layer-wise PH-168 is **redundant** with CoE-60 (stacked ensemble +0.001 over CoE alone at 0.811).
- Cross-domain: **CoE is the only symmetric, domain-invariant signal** (MATH↔BBH 0.720 / 0.712). Layer-wise PH is inverse / chance (0.354 / 0.497). D2H-lite is one-way.

The paper narrative pivots to **Chain-of-Embedding (Wang ICLR 2025, [2410.13640](https://arxiv.org/abs/2410.13640))** as the headline, with PH as a within-domain baseline.

This document enumerates five candidate research directions that could follow, ranked by publishability gain per GPU-hour, with supporting literature from the pathway-9 sweep. Each direction maps to a concrete experiment the next agent can pick up without further scoping.

Supporting literature index: [`pathway10-papers.md`](./pathway10-papers.md) — 12 themed clusters of 80+ papers pulled from the link-forge Neo4j graph (560 new papers ingested since 2026-04-22).

---

## Context from pathway 9 (what's fixed)

| Claim | Value | Source |
|---|---|---|
| CoE-60 MATH-500 holdout AUROC | **0.811** | `results/exp2_orthogonality.json` |
| CoE MATH→BBH transfer | **0.720** | `results/exp5_cross_domain.json` |
| CoE BBH→MATH transfer | **0.712** | `results/exp5_cross_domain.json` |
| ABC-44 deconfounded holdout | 0.7459 | `results/exp1_raw_length_v2.json` |
| Matched CV5 (LR vs XGB) | 0.7012 vs 0.6960 | `results/exp4_xgboost_oinfo_v2.json` |
| PH 5-feature real vs Gaussian null | 0.690 vs 0.693 | `results/exp3a_gaussian_null_v2.json` |

**What's unexplored and publishable:**
1. Cross-scale transfer (1.5B → 7B, same domain).
2. CoE ⊕ orthogonal signals (attention, spectral, intrinsic dimension).
3. Calibration metrics (ECE / AURC / risk-coverage) — we only have AUROC.
4. Mechanistic localization — **why** does CoE transfer?
5. Confidence-gated generation (live abstention + steering correction).

---

## Direction A: Cross-scale CoE transfer

**Why this is the strongest candidate for a standalone paper.** Pathway 9 established domain-invariance on Qwen2.5-1.5B. Scale-invariance is the natural sequel and is directly addressable on the existing infrastructure.

**Hypothesis.** A CoE-60 predictor trained on Qwen2.5-1.5B hidden-state dynamics will retain ≥0.70 AUROC when applied to Qwen2.5-7B on the same problems (MATH-500), *without* retraining. If it holds, the claim "CoE is a scale-invariant proxy for correctness inside a model family" is publishable.

**Supporting lit** (see `pathway10-papers.md § Cross-domain / cross-model`):
- *Model-Agnostic Correctness Assessment via Dynamic Internal Representation Selection* ([2510.02934](https://arxiv.org/abs/2510.02934)) — AUTOPROBE does attention-based dynamic representation selection; establishes the cross-model problem framing.
- *On LLMs' Internal Representation of Code Correctness* ([2512.07404](https://arxiv.org/abs/2512.07404)) — ICSE'26, shows internal correctness signal beats verbalized confidence; relevant for HumanEval extension.
- *Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction* ([2604.17716](https://doi.org/10.48550/arxiv.2604.17716)) — validates confidence screens across **20 frontier models**; the multi-model protocol to copy.

**Experiment (one week, ~4h H100).**
1. Extract CoE-60 on Qwen2.5-7B MATH-500 (pathway-8 infra reused).
2. Three settings:
   - **Same-model baseline:** train CoE classifier on 7B-train, test on 7B-test. Confirms 7B's intrinsic signal.
   - **Frozen transfer:** train on 1.5B-all, test on 7B-all. Measures scale-invariance directly.
   - **Feature normalized:** z-score each CoE feature per-model before transferring. Controls for trivial magnitude differences.
3. Report: AUROC, ECE, AURC, risk-coverage curves.

**Success criterion.** Frozen transfer ≥ 0.70 AUROC (within 0.10 of same-model). Z-score ablation tells us whether CoE is *dynamic* or *magnitude* signal.

**Publishable if:** frozen transfer ≥ 0.70, **or** z-score variant ≥ 0.72 — either tells a clean story.

---

## Direction B: CoE ⊕ orthogonal signals (ensemble headline)

**Why.** Pathway 9 Exp 2 showed layer-wise PH is **redundant** with CoE (r=0.388, ensemble lift +0.001). But attention-based and spectral signals were never tested for orthogonality against CoE — and both literature clusters are crowded with 2025 papers.

**Hypothesis.** Attention-based features (ReDeEP / AUTOPROBE) and spectral features (EigenScore / EigenTrack / D²HScore) encode different failure modes than CoE. A stacked ensemble lifts over CoE-alone by ≥0.03 AUROC on MATH-500 holdout.

**Supporting lit.**

*Attention-based* (see `§ Attention-based correctness`):
- *ReDeEP* ([2410.11414](https://arxiv.org/pdf/2410.11414)) — decouples parametric (Knowledge FFNs) from external (Copying Heads) via mechanistic interp. Most principled attention-level hallucination detector.
- *Inference-Time Intervention (ITI)* ([2306.03341](https://arxiv.org/pdf/2306.03341)) — head-selected truthfulness direction; identifies *which* attention heads carry the signal.
- *SnapKV* ([2404.14469](https://arxiv.org/pdf/2404.14469)) — attention-head importance clustering, useful for feature selection.

*Spectral* (see `§ Spectral / eigenvalue`):
- *INSIDE / EigenScore* ([2402.03744](https://arxiv.org/pdf/2402.03744)) — covariance-eigenvalue-based self-consistency, ICLR 2024.
- *EigenTrack* ([2509.15735](https://arxiv.org/abs/2509.15735)) — 2025, spectral statistics of activation covariance, single-pass.
- *D²HScore* ([2509.11569](https://arxiv.org/pdf/2509.11569)) — intra-layer dispersion + inter-layer drift of attention-selected tokens, training-free. Already partially replicated (0.740 / 0.529 transfer in pathway-9 exp5).

**Experiment (~1 week).**
1. Extract three feature sets on cached pathway-8 MATH-500 activations:
   - **A:** attention-head top-k ratios (ReDeEP / AUTOPROBE style), 20 dims.
   - **S:** EigenScore + EigenTrack spectral summary (covariance eigenvalue spectrum, KL divergence across layers), 30 dims.
   - **D:** D²HScore full (58 dims, pathway-9 already has D2H-lite).
2. Compute prediction correlation r(CoE, X) for X ∈ {A, S, D}.
3. Stacked LR ensemble: CoE + best-orthogonal-set, 5-fold OOF.
4. Cross-domain transfer for each ensemble variant.

**Success.** ≥0.03 AUROC lift on MATH-500 holdout **and** ≥0.03 transfer lift BBH→MATH. Either alone is a weaker claim.

---

## Direction C: Calibration, not just AUROC

**Why.** Pathway 9 reports 6 AUROCs and zero calibration metrics. Referees will ask. This is a 2-day fix that multiplies the pathway's credibility.

**Hypothesis.** CoE's correctness predictions are well-ranked (AUROC 0.811) but poorly calibrated (ECE ≥ 0.15). A single temperature-scaling pass closes the gap without touching AUROC.

**Supporting lit** (see `§ Calibration, selective prediction, abstention`):
- *Entropy Alone is Insufficient for Safe Selective Prediction in LLMs* ([2603.21172](https://arxiv.org/pdf/2603.21172)) — 2026, explicitly shows risk-coverage matters more than AUROC.
- *Calibrating LLM Confidence by Probing Perturbed Representation Stability* (CCPS, [2505.21772](https://arxiv.org/pdf/2505.21772)) — reports −55% ECE reduction via adversarial perturbation of final hidden states; same methodological family as CoE.
- *MACE: Evaluating and Calibrating LLM Confidence on Questions with Multiple Correct Answers* ([2602.07842](https://arxiv.org/abs/2602.07842)) — 2026, fixes a blind spot: the 0.811 AUROC is measured on single-answer problems; MACE would catch multi-answer miscalibration.
- *Which models are innately best at uncertainty estimation?* ([2206.02152](https://arxiv.org/abs/2206.02152)) — benchmarks 484 models on selective prediction; provides the metric menu.

**Experiment (~2 days, CPU).**
1. On pathway-9 CoE-60 outputs, compute: ECE (15-bin), Brier score, AURC, risk-coverage @ 50/75/90% coverage.
2. Fit temperature scaling on train, apply to test. Report pre/post.
3. Histogram-binning and isotonic regression for comparison.
4. Cross-domain calibration: is MATH→BBH transfer miscalibrated by a constant shift, or something stranger?

**Success.** Report a calibration table alongside the AUROC table in `SUMMARY.md §6`. Post-temperature ECE < 0.05 is the baseline a CoE paper can cite.

---

## Direction D: Mechanistic localization — why does CoE transfer?

**Why.** The most interesting *scientific* question left open by pathway 9. We observe that ABC-44 PH fails to transfer and CoE succeeds, but we don't know why. If CoE's signal lives on a small number of attention heads or SAE features, that's a mechanistic interpretability finding that goes far beyond the confidence paper itself.

**Hypothesis.** CoE-60's predictive signal concentrates in 2–5 SAE features or attention heads that are stable across MATH and BBH. PH's signal is a larger, less stable basis that shifts under domain change.

**Supporting lit** (see `§ Mechanistic interp / probing / sparse autoencoders`):
- *Sparse Autoencoders Find Highly Interpretable Features in Language Models* ([2309.08600](https://arxiv.org/pdf/2309.08600)) — Anthropic/Bricken foundation for SAE extraction.
- *Mechanistic Interpretability for AI Safety — A Review* ([2404.14082](https://arxiv.org/pdf/2404.14082)) — survey, identifies the "truthfulness direction" as the canonical localization target.
- *How to use and interpret activation patching* ([2404.15255](https://arxiv.org/abs/2404.15255)) — the tool for confirming causal role of a located feature.
- *Weakly Supervised Distillation of Hallucination Signals into Transformer Representations* ([2604.06277](https://arxiv.org/abs/2604.06277)) — 2026, distills hallucination signals into internal representations for lightweight probing.

**Experiment (~2 weeks, 1 H100 day for SAE training).**
1. Compute CoE-60-vs-correctness feature importance (LR coefficients, permutation importance).
2. Train a lightweight SAE on Qwen2.5-1.5B layer-15 residual stream (peak semantic layer per [2302.00294](https://arxiv.org/pdf/2302.00294)).
3. Correlate SAE feature activations with CoE predictions; identify top-k CoE-aligned features.
4. Activation-patch top SAE features on BBH to test *causal* role in cross-domain transfer.

**Success.** Publishable as "CoE signal is mediated by N specific SAE features / heads, which are stable across MATH and BBH" — regardless of N. Even a negative result (*signal is diffuse, not localizable*) is informative and writeable.

---

## Direction E: Alternative TDA — one last honest shot at PH

**Why.** Pathway 7 (non-Euclidean PH) was NO-GO at 0.774. Pathway 8 (layer-wise PH) transfers at chance. Before writing PH off entirely for the paper, there's one TDA variant worth a focused 48h test: **zigzag persistence on the layer-wise trajectory**.

**Hypothesis.** Zigzag PH ([0812.0197](https://arxiv.org/abs/0812.0197), [2604.16502](https://arxiv.org/abs/2604.16502)) captures *transition-critical* layer dynamics that standard PH misses. If it gives ≥0.03 AUROC over layer-wise PH-168 and survives the empirical-covariance null, it's worth keeping as a secondary signal.

**Supporting lit** (see `§ Alternative topological tools`):
- *Zigzag Persistence* ([0812.0197](https://arxiv.org/abs/0812.0197)) — Carlsson & de Silva foundation.
- *Topology-Aware Layer Pruning for LVLMs* ([2604.16502](https://arxiv.org/abs/2604.16502)) — 2026 CVPR-style paper, uses zigzag PH on hidden-state simplicial complexes to preserve *transition-critical* layers during compression. Direct methodological match.
- *Discrete Morse Theory for Computing Zigzag Persistence* ([1807.05172](https://arxiv.org/abs/1807.05172)) — makes zigzag tractable on our scale.
- *DTM-based Filtrations* ([1811.04757](https://arxiv.org/abs/1811.04757)) — noise-robust filtration, the specific variant pathway-7 tested on a single metric family.

**Experiment (~3 days with Dionysus2 or zigzag-fast, existing pathway-7 zigzag scaffolding).**
1. Reuse pathway-7 `zigzag_features.py` (already written, was blocked on Dionysus2 install).
2. Run on the cached pathway-8 MATH-500 / BBH activations.
3. Test: zigzag-PH alone, zigzag-PH ⊕ CoE ensemble, zigzag-PH cross-domain.

**Success.** Zigzag-PH ⊕ CoE lifts ≥0.03 over CoE alone. If NO-GO, the paper can state confidently that **all** PH variants tested are redundant with or inferior to CoE on this benchmark family.

**Risk.** This is the lowest-EV direction — pathway 7 + 8 already make the null case. Only worth running if Directions A–D are all blocked or completed.

---

## Recommended sequencing

| Order | Direction | Cost | Publishable alone? | Gates next? |
|---|---|---|---|---|
| 1 | **C: Calibration** | 2 days, CPU | No (footnote to A) | Yes — needed before any submission |
| 2 | **A: Cross-scale transfer** | 1 week, ~4h H100 | **Yes** — standalone paper | Strengthens main claim |
| 3 | **B: CoE ⊕ orthogonal ensemble** | 1 week, ~8h H100 | Yes (if ≥0.03 lift) | Adds breadth to the method paper |
| 4 | **D: Mechanistic localization** | 2 weeks, 1 H100 day | **Yes** — workshop / MI venue | Opens a second line of work |
| 5 | **E: Zigzag PH** | 3 days | Only as negative result | Closes the book on PH |

Direction C first is non-negotiable — the calibration table is a blocker for any submission, regardless of which direction becomes the headline. A and D are the two strongest candidates for a standalone paper. B is a rounding-out result. E is low-EV and should wait.

---

## Artifact map (for the next agent)

| Artifact | Path |
|---|---|
| Pathway 9 results | `/home/musicofhel/topo-confidence/pathway9/results/` |
| Pathway 9 summary | `/home/musicofhel/topo-confidence/pathway9/SUMMARY.md` |
| Pathway 9 handoff | `/home/musicofhel/topo-confidence/pathway9/HANDOFF_v3.md` |
| Pathway 8 activations (MATH / BBH / HumanEval) | `/home/musicofhel/topo-confidence/pathway8_layerwise/` cache |
| Pathway 7 zigzag scaffolding | `/home/musicofhel/topo-confidence/pathway7/zigzag_features.py` |
| CoE feature extractor | `/home/musicofhel/topo-confidence/pathway8_layerwise/coe_features.py` |
| D2H feature extractor | `/home/musicofhel/topo-confidence/pathway8_layerwise/d2hscore_features.py` |
| TwoNN feature extractor | `/home/musicofhel/topo-confidence/pathway8_layerwise/twonn_features.py` |
| RunPod pod (stopped) | `runpodctl pod start 0agitikupjg259` |
| Literature index | `/home/musicofhel/topo-confidence/pathway10-papers.md` |
| Raw literature sweep | `/home/musicofhel/topo-confidence/pathway9/literature-sweep.md` |

Refresh the literature index: `cd ~/link-forge && node scripts/build-new-directions.mjs > ~/topo-confidence/pathway10-papers.md`.

---

## Open questions (not scoped into A–E)

- **Length confound in CoE.** Pathway 9 deconfounded ABC-44 against raw token length (−0.050 drop). Was CoE-60 tested against the same control? If not, the 0.811 headline needs the same footnote.
- **Label scheme reconciliation.** Pathway 8 manifest labels (MATH=231 correct) vs NEW-labels (104 correct). Any pathway-10 number should specify which. The cross-domain transfer used pathway-8 labels; the headline 0.811 used NEW.
- **BBH subset heterogeneity.** Pathway 9 pooled 3 BBH subsets × 250. Is CoE transfer uniform across subsets, or driven by one subset?
- **MATH subset difficulty.** MATH-500's 7 difficulty levels — does CoE AUROC degrade monotonically with difficulty, and does that carry over to BBH?
