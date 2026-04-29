# Pathway 10 (v2): use hidden-state signal to make Qwen2.5-1.5B better

## Goal (unambiguous)

**Raise Qwen2.5-1.5B's MATH-500 accuracy above its 20.8% baseline using inference-time interventions derived from its own hidden-state geometry, with zero retraining.** Secondary goal: well-calibrated refusal so the model declines to answer when its hidden states say it shouldn't.

This is a model-improvement program, not a measurement or publication program. Every experiment is judged by whether the model behaves better after it, not by AUROC or publishability.

v1 framed pathway 10 as "CoE headline paper + 5 candidate directions." That was the wrong target. v2 replaces v1.

---

## What pathway 9 is actually telling us

Pathway 9 is the *input* to pathway 10, not the output. Reading the numbers through the model-improvement lens:

| Pathway-9 finding | What it tells pathway 10 |
|---|---|
| CoE-60 MATH-500 AUROC **0.811** | An actionable correctness direction exists in the residual stream — probably a single direction or low-rank subspace, since a 60-dim linear classifier finds it. |
| CoE MATH↔BBH transfer **0.720 / 0.712** | That direction is approximately domain-invariant within reasoning benchmarks. A steering vector fit on MATH should help on BBH. |
| Layer-wise PH at Gaussian null (**0.690 vs 0.693**) | PH-derived steering vectors would be noise. Don't waste time. |
| ABC-44 length-deconfound drop **−0.050** | The geometry features partly index output length. Any probe-as-steering-vector must be length-controlled or steering will just make the model output shorter strings. |
| Qwen2.5-7B MATH-500 baseline accuracy **69.6%** vs 1.5B's **20.8%** | A ~49 pt accuracy gap. If hidden-state steering closes even 5 pt of it, that's a major small-model improvement. The headroom exists. |

The pathway-9 open questions (label scheme NEW vs manifest, BBH subset heterogeneity, MATH difficulty stratification) matter less here because steering is evaluated by Δaccuracy on held-out problems under whichever consistent label scheme we pick.

---

## The plan: four experiments, E1 first

### E1 — Localize-and-steer (the headline, 3–5 days)

**Hypothesis.** A linear probe fit on Qwen2.5-1.5B residual-stream activations to predict MATH-500 correctness yields a weight vector `w` that, when added to residual-stream activations during generation at inference time (ITI-style), raises accuracy on held-out MATH-500 problems.

**Why now.** Pathway-9 CoE-60 at 0.811 AUROC proves the signal exists and is linearly extractable. ITI (2306.03341) established that probe-derived directions can be used as steering vectors on TruthfulQA. Nobody has tried this with a reasoning-trajectory signal on a math benchmark. The infrastructure already exists: pathway-8 activations are cached (on the stopped RunPod volume) and a classifier exists in pathway 9 exp2. This is a one-week assembly job, not a research project.

**Steps.**
1. **Stage activations.** Resurrect RunPod pod `0agitikupjg259` OR rsync cached MATH-500 activations local. Confirm coverage of Qwen2.5-1.5B layers 0–27.
2. **Fit probes per-layer.** Train a logistic-regression probe on mean-pooled residual-stream activations, one per layer, predicting correctness (pick NEW labels, 104 correct / 396 incorrect, and stick with them for all of pathway 10). Note the per-layer AUROC curve — identify the peak layer(s).
3. **Length-regress the probe.** Residualize the probe weight vector against raw-token-length regressor before using it. Compare length-residualized vs raw probes on AUROC and on the steering experiments below — if the residualized version works, we know we have a correctness signal, not a length signal.
4. **Steering experiment.** Generate Qwen2.5-1.5B completions with residual-stream intervention at the peak layer (and a range around it). For each generated token's residual stream `h`, compute `h' = h + α·w_norm` where `w_norm` is the unit-norm probe direction. Sweep `α ∈ {−4, −2, −1, 0, 1, 2, 4}`. Generate full solutions, score correctness on MATH-500 holdout (100 problems).
5. **Negative controls.** Same sweep with (a) a random unit direction, (b) the negated probe (`−w_norm`). The real direction should produce an asymmetric α-vs-accuracy curve; random should be flat; negated should hurt.

**Success.** Accuracy at the best `α > 0` exceeds baseline α=0 by **≥ 3 absolute points** on MATH-500 holdout, negative controls are flat or negative, and the length-residualized probe retains most of the benefit. If all three hold, hidden-state steering is a real intervention for this model.

**Failure modes to recognize.**
- Accuracy rises but so does average output length — probe is encoding "be verbose", which sometimes helps on MATH. Length-residualized probe tells us.
- Random-direction accuracy rises too — probably an artifact of generation perturbation (anything that breaks greedy decoding helps on MATH). Decode with sampling + multiple seeds, report means.
- Only the peak-layer probe works — fine, report per-layer sensitivity. Might point at a specific circuit.
- Nothing works — that's an honest result about this model and benchmark. Pivot to E3 (refusal) directly; a probe that can't steer might still abstain.

---

### E2 — Confidence-gated compute allocation (contingent on E1 OR standalone, 3 days)

**Hypothesis.** A prefix-CoE probe (the probe from E1, applied to partial generations) predicts final-answer correctness accurately enough to gate self-consistency sampling: when probe says "high confidence," use K=1; when it says "low confidence," use K=10. Average K is much lower than 10, but holdout accuracy matches K=10 self-consistency.

**Why.** Standard self-consistency (Wang 2022) is the strongest no-finetuning accuracy boost for small reasoning models, at K× compute cost. Dynamic K driven by the hidden-state signal gives the accuracy of large K at the average cost of small K.

**Steps.**
1. Generate K=10 samples per MATH-500 holdout problem with Qwen2.5-1.5B. Record final answers and prefix-CoE signal at a fixed token position (e.g., after the first answer sentence).
2. Fit the prefix-CoE probe (same method as E1, but on prefix activations) against per-problem majority-correctness label.
3. Policy: at inference, generate one sample, evaluate prefix-CoE. If confidence > threshold τ, emit. If not, draw 9 more and majority-vote.
4. Sweep τ, report (accuracy, average K) pareto curve.
5. Compare against: fixed K=1 baseline, fixed K=10 self-consistency, random-gate self-consistency.

**Success.** At average K ≈ 3–4, accuracy matches fixed K=10 within 1 pt. That's ~3× compute savings at equivalent accuracy — a concrete small-model deployment win.

---

### E3 — Hidden-state refusal (2 days, runs on E1's probe)

**Hypothesis.** The E1 probe can drive a refusal policy: when the probe says "low confidence" on a partial or completed generation, the model emits "I don't know" instead of a likely-wrong answer. Trades coverage for precision in a calibrated way.

**Why.** Qwen2.5-1.5B is wrong 79% of the time on MATH-500. In deployment, that means it confidently emits wrong answers most of the time. A probe-driven refusal turns it into a model that answers the 30% it's likely right on and declines the rest — massively more useful, even with zero accuracy improvement.

**Steps.**
1. Use E1's fitted probe. Compute its output on each held-out completion.
2. For a sweep of thresholds τ, compute (accuracy on non-refused, refusal rate). This is the risk-coverage curve.
3. Compare against: baseline no-refusal, verbalized-confidence refusal ("ask the model how confident it is"), max-logit refusal.
4. Report: at 50% coverage (model answers half the problems), what's the accuracy on the answered half?

**Success.** At 50% coverage, probe-driven refusal yields ≥ 40% accuracy on the answered subset (vs 20.8% unconditional). At 25% coverage, ≥ 60%. Verbalized-confidence baseline should be much weaker. If the probe is a real correctness signal, this must work — it's the same math as the 0.811 AUROC, reformulated as a decision.

---

### E4 — Big-to-small distillation of hidden-state confidence (later, 2+ weeks)

**Hypothesis.** Qwen2.5-7B's residual stream encodes correctness more strongly than 1.5B's (consistent with its 69.6% vs 20.8% accuracy). Training Qwen2.5-1.5B with an auxiliary loss that matches its layer-15 activations to 7B's layer-15 activations on the same inputs — only for the correctness-predictive direction, not the whole activation — transfers some of 7B's self-knowledge into 1.5B.

**Why here and not now.** This requires fine-tuning, which pathway 10 was going to avoid. But if E1/E2/E3 max out what inference-time interventions can do, this is the natural next step — and it's a cleaner research contribution than the v1 doc's cross-scale-transfer framing (which went the wrong direction).

Deferred to after E1–E3 land. Not scoped further in v2.

---

## Ordering and gates

| # | Experiment | Cost | Gate |
|---|---|---|---|
| 0 | Resurrect pod or local-stage activations | 0.5 day | Prerequisite for everything |
| 1 | **E1: localize-and-steer** | 3–5 days, ~1 H100 day | If accuracy doesn't move, skip to E3 |
| 2 | **E3: refusal policy** | 2 days, CPU | Can run regardless of E1 outcome — only needs the probe |
| 3 | **E2: confidence-gated compute** | 3 days, ~4h H100 | Requires prefix-CoE working; test independently |
| 4 | E4: distillation | 2+ weeks, multi-day H100 | Only if E1–E3 motivate it |

**Start E1. The length-residualization step (E1.3) is the one check that must not be skipped — it's the only way to distinguish a real correctness signal from a length regressor.**

---

## What to abandon from v1

- **Direction A (cross-scale 1.5B → 7B transfer):** backwards. The goal is improving the small model, not showing signals transfer to the big one. Gone.
- **Direction B (CoE ⊕ orthogonal ensemble):** pure AUROC stacking. If E1 needs a stronger probe, try attention-head features and eigenvalue features as *additional probe inputs*, but don't ensemble for its own sake. Gone as a standalone direction.
- **Direction C (calibration metrics as standalone):** folded into E3. ECE/AURC/risk-coverage are *reports* on interventions, not experiments themselves.
- **Direction E (zigzag PH):** pure measurement. Pathway-9 already showed PH is at the Gaussian null — PH-derived steering vectors would be noise. Dead.
- **The "publishable if" framing everywhere:** irrelevant. Success = the model behaves better.

---

## Artifact map

| Artifact | Path |
|---|---|
| Pathway 9 CoE-60 classifier + weights | `~/topo-confidence/pathway9/results/exp2_orthogonality.json` (`stacked_coefs`) |
| Pathway 9 cross-domain probe | `~/topo-confidence/pathway9/results/exp5_cross_domain.json` |
| Cached MATH-500 activations (Qwen2.5-1.5B, 28 layers) | RunPod pod `0agitikupjg259` volume (currently stopped) |
| CoE feature extractor | `~/topo-confidence/pathway8_layerwise/coe_features.py` |
| Pathway 9 summary | `~/topo-confidence/pathway9/SUMMARY.md` |
| Pathway 7 zigzag code | `~/topo-confidence/pathway7/zigzag_features.py` — **not needed for pathway 10** |
| Pathway 10 papers (needs dedup + noise filter, see Open Questions) | `~/topo-confidence/pathway10-papers.md` |

**Preflight before any pathway-10 work:**
```bash
runpodctl pod status 0agitikupjg259          # confirm volume intact
docker ps --format '{{.Names}}' | grep neo4j  # link-forge running for lit queries
ls ~/topo-confidence/pathway9/results/*.json | wc -l   # ≥ 9
```

---

## Open questions (not gating E1, but watch for them)

- **Does the probe encode length?** E1.3 answers this. If length-residualized probe retains AUROC and steering lift, the signal is real. If it collapses, the steering direction is a length axis and we need to move to layer-specific or head-specific probes that don't pool length information.
- **Is the effect specific to MATH, or does it transfer to BBH / HumanEval?** Once E1 works on MATH, re-run on BBH with the MATH-fitted probe, no retraining. Pathway-9 showed CoE transfers; pathway-10 tests whether steering transfers. If yes, the intervention is a general-purpose small-model booster.
- **Per-layer or multi-layer steering?** E1 starts with peak-layer only. If that works marginally, try simultaneous intervention at 3 layers around the peak.
- **Does the intervention interact with sampling temperature?** At T=0.0 (greedy) the probe might just break decoding. At T=0.7 the probe might push the distribution toward a specific mode. Report both.
- **Corpus cleanup.** `pathway10-papers.md` has heavy duplication and keyword-match false positives; build-new-directions.mjs needs an arxiv-ID dedup pass and LLM-relevance AND clauses before the index is trustworthy as a reading list. Low priority for E1 but blocking for a serious literature review of steering/intervention work.

---

## Lit to actually read for E1 (not the full index)

- **Inference-Time Intervention** (Li et al., [2306.03341](https://arxiv.org/abs/2306.03341)) — the canonical probe-as-steering-vector paper. This is the blueprint for E1.
- **ReDeEP + AARF** ([2410.11414](https://arxiv.org/abs/2410.11414)) — head-level mechanistic interp with a mitigation method; relevant if E1 needs to go head-specific.
- **CCPS** ([2505.21772](https://arxiv.org/abs/2505.21772)) — perturbation-based calibration probe; relevant for making E3's refusal threshold robust.
- **Chain-of-Embedding** (Wang ICLR 2025, [2410.13640](https://arxiv.org/abs/2410.13640)) — the original CoE paper, for prefix-CoE design in E2.
- **Brain-Grounded Axes** ([2512.19399](https://arxiv.org/abs/2512.19399)) — external-coordinate steering, speculative reading for E4.

A supplementary corpus sweep on steering / representation-engineering / activation-editing should run in parallel with E1 — the pathway-9 corpus is optimized for confidence estimation, not intervention. Add queries to a text file and process via `cd ~/link-forge && npx tsx scripts/search-papers.ts --file <file> --max 10 --source arxiv,semantic-scholar,openalex`.

---

## Success criterion for pathway 10 as a whole

**A recipe that takes Qwen2.5-1.5B, an off-the-shelf 1.5B model with 20.8% MATH-500 accuracy, and — using only cached hidden-state signals from pathway 9 — produces:**

- A steered version with ≥ 24% MATH-500 accuracy, or
- A refusal policy that answers 50% of problems at ≥ 40% accuracy on the answered subset, or
- A confidence-gated self-consistency that matches K=10 accuracy at average K ≤ 4.

Any one of the three lands pathway 10. All three would be a toolkit. Zero of the three is an honest negative result that tells us Qwen2.5-1.5B's hidden-state signal, despite looking strong in AUROC, isn't actionable — a finding worth knowing.
