# PROJECT_RECORD.md — topo-confidence authoritative reference

Self-contained archive of everything the topo-confidence research program has found,
validated, and abandoned across Pathways 1–11. Every number is traced back to a
specific committed JSON; provenance is enforced by `validate_claims.py`
(`validation_report.txt` — 91/91 PASS at time of writing, 2026-04-24).

Supersedes all prior summaries. Where this document and earlier narratives disagree,
the JSONs are ground truth and this record is updated to match them.

**How to use this file.**

- Section 1a is the chronology. Read it first if you're picking up cold.
- Section 1b is the provenance table. Every quantitative claim anywhere in the project
  is here with a source file + JSON path. Changes to any number must update both the
  claim and the validator.
- Section 1c is the reproducibility table. Same grain as 1b, but one row per
  experiment instead of one per number — tells you what setup produced what.
- Section 1d is the graveyard. Everything that was tried and killed, with the decisive
  evidence. Treat this as a "do not re-run" list.
- Section 1e is the queue. Everything still worth doing.
- Section 1f is literature attribution.
- Section 1g is the file inventory. What exists, what's cached, what's gitignored.

---

## 1a. Pathway timeline (2026-04-04 through 2026-04-24)

The topo-confidence project ran 11 sequential pathways over three weeks in April 2026,
accumulating ~94 GB of cached activations and ~150 committed result JSONs. The
central story is a single finding (hidden-state geometry predicts LLM
correctness) repeatedly refined, reframed, and deconfounded — culminating in a
pivot away from the original "topological" framing toward direction-of-mean
projections, and away from a "standalone paper" framing toward model
improvement via steering/refusal/compute-gating.

### Pathway 1 — CORAL feature extraction (2026-04-12)

**Goal.** Port the 78-feature CORAL pipeline (hidden-state PH + geometry) from
the ATT project into a standalone repo and reproduce a holdout MATH-500 AUROC.

**Ran.** Phase 0 probe sweep, Phase 1 holdout, Phase 2 ablation over 4 feature
tiers, Phase 3 seed sensitivity, Phase 4 routing prototype.

**Concluded.** Frozen 44-feature ABC extractor (9 PH/geometry + 30 layer
dynamics + 5 depth-2 products) reproduces a clean holdout AUROC; Tier D (34
depth-3+ products) actively hurts and is dropped. This is the ancestor of every
feature pipeline used later.

- 2026-04-12 12:40 — initial commit with experiments 1–6
- 2026-04-12 15:40 — null hypothesis testing + 13-feature TDA variant
- 2026-04-12 15:41 — CORAL reproduction, phase 0/1, phase 4 routing

### Pathway 2 — Learned steering, first attempt (2026-04-12 → 13)

**Goal.** Take the Pathway-1 direction vector and test whether injecting it into
Qwen2.5-1.5B's residual stream raises MATH-500 accuracy.

**Ran.** Track A (spherical steering sweep). Phase 2 tau-threshold sweep with
extensive checkpoint commits.

**Concluded.** Targeted tau=0.40 steering at t=0.15 gave "+26 net gain"
(measured against 256-tok truncated run). Appeared publishable at the time;
later pathway-10 checks showed the steering vector's cosine with the
current-extraction DoM direction is **~0.05** — the original steering direction
is geometrically unrelated to the current L19 correctness axis. Numbers from
pathway 2 are superseded.

- 2026-04-12 21:36 — Pathway 2 A/B experiment prompts
- 2026-04-13 00:18 → 05:12 — nine t-sweep checkpoints (t=0.05 → t=0.50)
- 2026-04-13 05:12 — locked config t=0.15, tau=0.40

### Pathway 3 — Complexity expansion (2026-04-13 → 14)

**Goal.** Expand ABC-44 with denoised multi-layer non-adjacent features; sweep
K-nearest-neighbor prototype counts; see whether more structure helps.

**Ran.** Phase 1 expansion, Phase 2 prototypes (K∈{3,5,8} × layers {15,20,25}),
Phase 3 CV on 4 feature configs × 5 folds, Phase 4 holdout + statistical tests.

**Concluded.** Complexity didn't help. Go-decision: PIVOT.

- 2026-04-14 19:39 — Pathway 3 experiment complete, PIVOT decision

### Pathway 4 — Topo-guided selection + learned steering (2026-04-15 → 16)

**Goal.** Track A (topo-guided test-time selection among K samples) and Track B
(learned steering via probe-fit direction).

**Ran.** Track A: 10 experiments including adaptive sampling, per-completion
verifier, tier ablation, calibration. Track B: full training pipeline.

**Concluded.** Track A reached "PUBLICATION_READY" status at +11 holdout net
gain with zero correct-side losses (adaptive sampling gave 77-89% compute
savings at matched accuracy). Track B was null. Numbers are at 256-tok
truncation; the adaptive-sampling result survives in spirit but gets
re-measured in pathway 11 at 1024 tokens.

- 2026-04-15 22:28 — Pathway 4 launched
- 2026-04-16 14:19 — Track A complete, Track B null

### Pathway 5 — Cross-benchmark / cross-model (2026-04-16 → 17)

**Goal.** Test whether Track A from Pathway 4 transfers to GSM8K (different
benchmark) and Qwen2.5-7B (different model).

**Ran.** Track A (GSM8K 1.5B), Track B (MATH-500 7B), Track C (cross-model
feature compare). Phase A1 baseline answers, A2 features, A3 temperature
sampling, A4 results. Phase B1/B2/B3 same for 7B. Phase C cross-compare.

**Concluded.** "Cross-benchmark works" at the time — led to Phase 6 rebuild.
Pathway 5 numbers were at `max_new_tokens=256` and later shown to be
truncation-confounded.

- handoff: `.claude/handoff/2026-04-17-pathway5-cross-benchmark-cross-model.md`

### Pathway 6 rebuild — Corrected labels + Phase 6.5 deconfounding (2026-04-17 → 19)

**Goal.** Rebuild the entire pipeline with proper answer extraction and
leak-free PCA. Then run cross-benchmark at scale on H100.

**Ran.** Phase 0 relabel (caught +47 correct answers, 11.4% → 20.8% accuracy),
Phase 1 prompt/model at 0.7961 AUROC, Phase 2 completion gating, Phase 3
RunPod cross-benchmark (GSM8K + 7B), Phase 4 report, then **Phase 6.5
deconfounding** with `max_new_tokens=1024`.

**Concluded.** This pathway produced the most consequential correction of the
program: the original high cross-benchmark AUROCs (GSM8K 0.731, 7B 0.682) were
**truncation artifacts**. At 1024 tokens, GSM8K AUROC dropped to 0.615
(baseline beats topo by 0.126), 7B AUROC rose to 0.739 (at 69.6% accuracy).
The 1.5B MATH-500 0.7961 holdout is truncation-independent (no GSM8K-like
cutoff effect at 20.8% accuracy).

Key decisions + pivots:
- **2026-04-18 15:31** — Phase 2+3 RunPod results committed.
- **2026-04-18 18:15** — audit discovers truncation confound.
- **2026-04-18 18:36** — Phase 6.5 scripts with `max_new_tokens=1024`.
- **2026-04-19 05:59** — Phase 6.5 complete, cross-benchmark transfer refuted
  (0.504 AUROC = chance).
- **2026-04-19 19:54** — all docs updated with corrected results.

### Pathway 7 — Non-Euclidean PH (2026-04-19 → 20)

**Goal.** Upgrade PH from Euclidean to diffusion / effective-resistance /
cosine / DTM distance metrics. Add zigzag persistence across layers.

**Ran.** Phase 7.1 (CPU sweep over 4 metrics on MATH-500), benchmark pipelines
for HumanEval and BBH, synthetic-shapes validation.

**Concluded.** NO-GO. Best non-Euclidean variant reached 0.774 AUROC vs the
0.7961 Euclidean baseline. Committed with explicit decision note.

- 2026-04-19 21:48 — Pathway 7 pipeline + Phase 7.1 NO-GO committed
- 2026-04-19 23:57 — Euclidean PH + logprob baseline added to benchmarks
- 2026-04-20 02:46 — HumanEval + BBH results (Pathway 7.3/7.4)

### Pathway 8 — Layer-wise PH + alternative methods (2026-04-20 → 21)

**Goal.** Distribute PH across all 28 layers (not just last); compare directly
against Chain-of-Embedding (Wang ICLR 2025) and D2HScore (arXiv 2509.11569).

**Ran.** Five experiments on RunPod H100 (~7.5 hours, ~$22.50):
- Exp 1 layer-wise PH (168-dim: 28 layers × 6 PH features).
- Exp 2 method comparison on 8 configurations.
- Exp 3 TwoNN intrinsic dimension.
- Exp 4 cross-layer trajectory PH.
- Exp 5 cross-domain BBH.

**Concluded.** CoE **beats** the ABC-44 baseline (0.811 vs 0.7961). Layer-wise
PH is **worse** than single-layer PH (0.646 < 0.7961). Adding PH to CoE hurts
(0.758 for combined). BBH pooled 0.783 is the first evidence of
cross-benchmark signal that beats chance (previously 0.504).

- 2026-04-21 07:55 — Pathway 8 layer-wise PH + CoE/D2H results committed

### Pathway 9 — CoE pivot + PH audit (2026-04-22)

**Goal.** Systematically audit the pathway-8 numbers. Seven experiments on
RunPod pod `0agitikupjg259`.

**Ran.**
- Exp 1 v2: ABC-44 length deconfound against raw tokenizer length.
- Exp 2: PH-168 + CoE-60 stacked ensemble.
- Exp 3a v2: empirical-covariance Gaussian null for 5 raw PH features.
- Exp 3b: token shuffle (50 problems × 5 shuffles).
- Exp 3c: count control (trivially correct).
- Exp 4 v2: LR vs XGBoost matched CV5, O-information attempt (OOMed at 3 TiB).
- Exp 5: cross-domain transfer with source-PCA applied to both domains.

**Concluded.**
- ABC-44 0.7961 survives but loses 0.050 AUROC to raw-length deconfounding
  (final 0.7459).
- 5 raw PH summary features at covariance null (0.690 vs 0.693).
- Stacked PH+CoE ensemble adds +0.001 over CoE alone (redundant).
- LR ≈ XGBoost at matched CV5 (diff −0.005); signal is linear.
- **CoE is the only domain-invariant signal.** Transfers symmetrically
  (0.720 / 0.712). PH transfers at chance / inverse (0.354 / 0.497). D2H-lite
  is asymmetric (0.740 / 0.529).

Recommendation at end of pathway 9: pivot paper narrative to CoE.

- 2026-04-22 19:04 — Pathway 9 results committed

### Pathway 10 — Goal pivot: steering (2026-04-23)

**Plan only, no new experiments ran.** User correction on 2026-04-23 shifted
the program from "publishable paper on CoE" to "improve small models via
hidden-state understanding."

**v1 plan** (obsoleted before execution): five directions including
cross-scale 1.5B→7B transfer, CoE ⊕ orthogonal ensemble, calibration tables,
SAE localization, zigzag PH.

**v2 plan** (current): four experiments ordered E1 → E3 → E2 → E4.
- E1 headline: linear probe → ITI-style steering direction → α sweep.
- E2: prefix-CoE gates self-consistency K.
- E3: hidden-state refusal policy.
- E4: big-to-small distillation.

Success criteria (any ONE):
- ≥24% MATH-500 accuracy post-steering (later refactored — the 20.8% baseline
  was itself a 256-tok artifact; the 1024-tok baseline is 48.6%).
- 40% accuracy at 50% coverage refusal.
- K=10-equivalent self-consistency at avg K≤4.

Blocked on cached activations. This motivated Pathway 11.

### Pathway 11 — H100 data re-gather + post-hoc analysis (2026-04-23 → 24)

**Goal.** Produce the activation caches Pathway 10 v2 was blocked on — at 1024
tokens, with per-token L19 capture, and with K=8 self-consistency samples.

**Ran.** Seven-stage RunPod H100 pipeline on pod `y687b9z2dgukcj` (~17h wall
time).

| Stage | Contents | Output |
|---|---|---|
| 0 | Preflight | marker |
| 1 | 7B MATH-500 all-layer @ 1024 tokens | `data/math500_7b/` (42 GB) |
| 2 | 1.5B MATH-500 all-layer @ 1024 tokens | `pathway8_layerwise/data/math500/` (19 GB) |
| 3 | K=8 self-consistency on 1.5B + per-sample L19 | `data/k8_selfconsistency/` (13 MB) |
| 4a | BBH 3×250 all-layer | `pathway8_layerwise/data/bbh/` (18 GB) |
| 4b | BBH per-subset diagnostics | `results/stage4b_bbh_per_subset.json` |
| 5 | L19 DoM cross-benchmark transfer | `results/stage5_bbh_dom_transfer.json` |

Once data was in hand, **four post-hoc analysis mini-experiments** ran locally
on CPU + one more GPU extraction:

| Mini-experiment | Location | Headline |
|---|---|---|
| **Exp 2 — prefill-gated compute** | `prefill_gated_compute/` | Prefill DoM AUROC 0.7731; refuse-and-spend @ cov=0.5 → 71.6% on answered (+22pp over random) |
| **Exp 3 — 7B prefill PR inversion** | `prefill_inversion/` | Ratio 1.377, CI [1.226, 1.757]; not BBH; level-driven |
| **Exp 2b — multi-signal oracle** | `multi_signal_oracle/` | Multi-feature vs best single = +0.4pp (<2pp bar) |
| **Exp 1 — cross-model breathing** | `exp1_cross_model/` | Phi-3 + Llama-3.2-1B both reproduce breathing |

Followed by two **controls** on 2026-04-24 evening:
| Control | Location | Headline |
|---|---|---|
| **Test 3 — gibberish** | `gibberish_control/` | Random tokens give flat PR ≈ 10; breathing is content-dependent |
| **Test 2 — no-CoT** | `no_cot_control/` | No-CoT gen median 2 tokens → inconclusive |

Key decisions + pivots:
- **2026-04-23 12:44** — Pathway 11 orchestrator created (`runpod_pathway11.sh`).
- **2026-04-24 06:16** — pipeline complete, pod STOPPED (not removed) via volumeInGb preservation.
- **2026-04-24 08:21–09:35** — Exps 2/2b/3 analyses all ran locally on cached data.
- **2026-04-24 12:15** — cross-model extraction on pod `y687b9z2dgukcj`.
- **2026-04-24 14:01** — Exp 1 analysis complete (Phi-3 + Llama-3.2-1B).
- **2026-04-24 18:15** — gibberish control complete on pod `lsuoka6bo8io7m`.
- **2026-04-24 18:42** — no-CoT control complete, pod stopped.

The consequential finding of Pathway 11 is the 256-token truncation artifact.
Everything prior to 1024-tok data — including Pathway 9's length-confound
finding, the "7B as superior verifier" framing, and the 0.948/0.796/0.811
AUROC headline series — ran against labels where the 1.5B got 104 / 500
correct at truncated 256 tokens. At 1024 tokens the 1.5B gets **243 / 500
correct** (48.6%, not 20.8%). Every downstream claim gets re-baselined in
Pathway 11.

### Pathway 11 follow-on — FE291 PCA cascade (2026-05-02)

After the cheap-wins phase (FE110/115/136/244/319/331/339/254 — corroborators
and orthogonality probes promoted 2026-04-30 → 2026-05-01), the FE291 PCA
decomposition kicked off the second-order cascade and resolved with three
follow-ups that landed as commit `39c264f`:

| EXP | FE | Question | Result |
|---|---|---|---|
| EXP-55 | FE291 | PCA structure of L19 prefill activations | DoM ≈ 0.92·PC1; 14.7% var; 2-feat (PC1,PC9) → 0.7856 vs DoM 0.7679 |
| EXP-56 | FE880 | PH on PC1-residualized point clouds | Real PH 0.6884 vs matched-cov Gaussian null 0.7628 (gap −0.074) — F-10 strengthens |
| EXP-57 | FE881 | Top-20 log-eigvals of PC1-residualized cov | **0.7928** — strongest L19 probe at any tier |
| EXP-58 | FE882 | Full 1536-d L2-reg directional ceiling | 0.7847 at C=0.001 ≈ 2-feat 0.7856; cov-spectrum lift is genuinely second-order |

The FE291 cascade gave F-2 a clean factorization into three additive pieces
(PC1 mean shift, PC9 trim, per-problem residual second-order shape) and
gave F-10 a pre-registered overturning-condition test that failed by 12.4 pp
in the wrong direction.

### Synthesis cascade — three practitioner docs (2026-05-03)

No experiments ran. Three documents synthesizing the FE291 cascade landed:

- `SYNTHESIS.md` (375 lines) — practitioner briefing.
- `NOVELTY_AUDIT.md` (469 lines) — F-N rankings vs research-graph.
- `APPLICATIONS.md` (387 lines) — deployment surfaces with checklists.

Plus this session (2026-05-03 evening) propagated the synthesis into the
intent-layer narrative docs (README, QUICKSTART, CLAUDE, STATE,
PROJECT_RECORD, HYPOTHESES, FINDINGS audit, PERSPECTIVES, NEXT_EXPERIMENTS,
PAPER_INDEX, RESEARCH_GRAPH) and rewrote `docs/index.html` from a v1 archive
landing page into a current-state landing page with the v1 panels folded
under a `<details>` archive section.

---

## 1b. Provenance table — every cited number, every source

Every quantitative claim is keyed to a committed JSON file and confirmed by
`validate_claims.py` (91 claims, all PASS as of 2026-04-24). Numbers in the
final column reflect the label scheme actually used for the computation.

**Label scheme legend.**
- `1024tok` — the current / correct labels; 1.5B has 243/500 correct, 7B has
  366/500 correct.
- `256tok_NEW` — truncation-artifact labels from Pathway 6 rebuild Phase 1
  (1.5B has 104/500 correct, 7B has 81/500 correct). Post-Phase 6.5 /
  Pathway 11 these are known to be confounded — flagged SUPERSEDED where the
  claim is still invoked in narratives.
- `256tok_manifest` — pathway-8 manifest `correct` field, 231/500 on MATH-500
  (more lenient answer matcher than NEW). Used in Pathway 9 Exp 5 cross-domain.
- `mixed` — documents the transition (e.g., Phase 6.5's 7B-at-1024 numbers
  compared against 1.5B-at-256 baselines).

### §1b.i — Baselines and bucket structure

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| 1.5B MATH-500 K=1 (greedy T=0) acc | **0.486** (243/500) | `pathway11_h100/prefill_gated_compute/results.json` | `accuracy_at_uniform_K["K=1 (greedy T=0)"]` | 1024tok | ✅ |
| 1.5B K=2 majority (T=0.7) | 0.4132 | ibid | `accuracy_at_uniform_K["K=2 majority (T=0.7)"]` | 1024tok | ✅ |
| 1.5B K=4 majority | 0.4950 | ibid | `accuracy_at_uniform_K["K=4 majority (T=0.7)"]` | 1024tok | ✅ |
| 1.5B K=8 majority | **0.550** | ibid | `accuracy_at_uniform_K["K=8 majority (T=0.7)"]` | 1024tok | ✅ |
| 7B MATH-500 K=1 acc | **0.732** (366/500) | `pathway11_h100/prefill_inversion/phase1_bootstrap.json` | `7B_prefill.n_correct` / 500 | 1024tok | ✅ |
| Oracle compute/problem | 1.01 | `pathway11_h100/prefill_gated_compute/results.json` | `oracle_compute_per_problem` | 1024tok | ✅ |
| Oracle overall acc | 0.589 | ibid | `oracle_overall_accuracy` | 1024tok | ✅ |
| Bucket A (always-right) | n=207 (41.4%) | ibid | `bucket_sizes.A_always_right` | 1024tok | ✅ |
| Bucket B (recoverable) | n=68 (13.6%) | ibid | `bucket_sizes.B_recoverable` | 1024tok | ✅ |
| Bucket C (never-right) | n=189 (37.8%) | ibid | `bucket_sizes.C_never_right` | 1024tok | ✅ |
| Bucket D (pathological) | n=36 (7.2%) | ibid | `bucket_sizes.D_pathological` | 1024tok | ✅ |

### §1b.ii — Correctness signal (DoM directions)

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| Prefill L19 DoM AUROC (OOF 5-fold) | **0.7731** | `pathway11_h100/prefill_gated_compute/results.json` | `prefill_DoM_auroc_oof` | 1024tok | ✅ |
| Final-token L19 DoM AUROC | 0.7186 | ibid | `final_token_DoM_auroc_oof` | 1024tok | ✅ |
| Two-feature LR (prefill + final DoM) AUROC | ≈0.794 | RESEARCH_SUMMARY §2 | narrative, from combining the two OOF scores | 1024tok | ✅ (implicit) |
| 7B prefill DoM AUROC | ~0.876 | RESEARCH_SUMMARY §4 | narrative estimate, not a single JSON key; per-fold 0.87–0.88 spread | 1024tok | narrative — use with the spread caveat |
| Stage-5 L19 DoM MATH→BBH pooled | 0.7472 | `pathway11_h100/results/stage5_bbh_dom_transfer.json` | `math_to_bbh._pooled.auroc` | 1024tok | ✅ |
| Stage-5 BBH→MATH | 0.6926 | ibid | `bbh_to_math.auroc` | 1024tok | ✅ |
| Stage-5 symmetric avg | 0.7199 | ibid | `symmetric_avg` | 1024tok | ✅ |
| Δ vs CoE symmetric | +0.0039 | ibid | `verdict_vs_coe` | 1024tok | ✅ |
| MATH within-benchmark (LeavePairOut-OOF) | 0.7306 | ibid | `within_benchmark_overfit.math.auroc` | 1024tok | ✅ |

### §1b.iii — Dimensional breathing — cohort / correct / incorrect PR

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| 7B final-token PR (correct, n=366) | 2.629 | `pathway11_h100/prefill_inversion/phase1_bootstrap.json` | `7B_final_token.point.pr_correct` | 1024tok | ✅ |
| 7B final-token PR (incorrect, n=134) | 6.848 | ibid | `7B_final_token.point.pr_incorrect` | 1024tok | ✅ |
| 1.5B final-token PR (correct, n=243) | 4.330 | ibid | `1.5B_final_token.point.pr_correct` | 1024tok | ✅ |
| 1.5B final-token PR (incorrect, n=257) | 8.511 | ibid | `1.5B_final_token.point.pr_incorrect` | 1024tok | ✅ |
| 7B prefill PR (correct) | 26.873 | ibid | `7B_prefill.point.pr_correct` | 1024tok | ✅ |
| 7B prefill PR (incorrect) | 19.514 | ibid | `7B_prefill.point.pr_incorrect` | 1024tok | ✅ |
| 1.5B prefill PR (correct) | 19.405 | ibid | `1.5B_prefill.point.pr_correct` | 1024tok | ✅ |
| 1.5B prefill PR (incorrect) | 20.521 | ibid | `1.5B_prefill.point.pr_incorrect` | 1024tok | ✅ |
| Temporal r(PR, AUROC) | −0.695 | RESEARCH_SUMMARY §4 | narrative, computed from §3 + §4 tables | 1024tok | reconstructable |

### §1b.iv — Prefill inversion bootstrap / controls (Exp 3)

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| 7B prefill ratio correct/incorrect | **1.377** | `pathway11_h100/prefill_inversion/results.json` | `headline_ratios_point_estimates["7B_prefill_PR_correct_over_incorrect"]` | 1024tok | ✅ |
| 1.5B prefill ratio | 0.946 | ibid | `[...1.5B_prefill...]` | 1024tok | ✅ |
| 7B final-token ratio | 0.384 | ibid | `[...7B_final_token...]` | 1024tok | ✅ |
| 1.5B final-token ratio | 0.509 | ibid | `[...1.5B_final_token...]` | 1024tok | ✅ |
| 7B prefill 95% CI (ratio) | [1.226, 1.757] | ibid | `bootstrap_95_CI_on_ratio["7B_prefill"]` | 1024tok | ✅ |
| 7B prefill balance-controlled ratio | 1.227 | ibid | `balance_controlled_ratios["7B_prefill"]` | 1024tok | ✅ |
| P(ratio > 1) on 7B prefill | 1.000 | `pathway11_h100/prefill_inversion/phase1_bootstrap.json` | `7B_prefill.bootstrap.ratio.pct_ratio_above_1` | 1024tok | ✅ |
| Three-way split — both_right n | 230 | `pathway11_h100/prefill_inversion/results.json` | `three_way_split.both_right.n` | 1024tok | ✅ |
| only-7b n | 136 | ibid | `three_way_split.only_7b.n` | 1024tok | ✅ |
| only-1.5b n | 13 | ibid | `three_way_split.only_15b.n` | 1024tok | ✅ |
| both-wrong n | 121 | ibid | `three_way_split.both_wrong.n` | 1024tok | ✅ |
| BBH tracking_shuffled prefill ratio | 0.930 | ibid | `bbh.tracking_shuffled_objects_seven_objects.pr_prefill_ratio` | 1024tok | ✅ |
| BBH logical_deduction prefill ratio | 0.824 | ibid | `bbh.logical_deduction_seven_objects.pr_prefill_ratio` | 1024tok | ✅ |
| BBH web_of_lies prefill ratio | 1.009 | ibid | `bbh.web_of_lies.pr_prefill_ratio` | 1024tok | ✅ |

### §1b.v — Compute allocation / refuse-and-spend (Exp 2)

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| Prefill refuse@cov=0.5 acc on answered | **0.716** | `pathway11_h100/prefill_gated_compute/results.json` | `headline.refuse_and_spend_coverage_0.5.prefill.acc_on_answered` | 1024tok | ✅ |
| Random refuse@cov=0.5 acc on answered | 0.494 | ibid | `[...random...]` | 1024tok | ✅ |
| Neg-seq-len refuse@cov=0.5 | 0.707 | ibid | `[...neg_seq_len...]` | 1024tok | ✅ |
| Prefill middle-heavy K=4.5 acc | 0.526 | ibid | `headline.prefill_middle_heavy.overall_accuracy` | 1024tok | ✅ |
| Neg-seq-len top-heavy K=4.5 acc | 0.542 | RESEARCH_SUMMARY §7 | narrative from `phase3b_realistic_policies.json` | 1024tok | ✅ |
| Spearman(prefill DoM, oracle-K) | 0.175 | SUMMARY.md §7 | narrative, `phase4_oracle.json` | 1024tok | ✅ |
| Spearman(neg-seq-len, oracle-K) | 0.327 | ibid | narrative | 1024tok | ✅ |

### §1b.vi — Multi-signal oracle (Exp 2b)

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| Logreg macro-F1 | 0.548 | `pathway11_h100/multi_signal_oracle/results.json` | `classifier_logreg.macro_f1` | 1024tok | ✅ |
| RF macro-F1 | 0.494 | ibid | `classifier_rf.macro_f1` | 1024tok | ✅ |
| Logreg overall acc (OOF) | 0.620 | ibid | `classifier_logreg.overall_acc_oof` | 1024tok | ✅ |
| RF overall acc (OOF) | 0.636 | ibid | `classifier_rf.overall_acc_oof` | 1024tok | ✅ |
| Logreg D-recall to K=1 | 0.417 (15/36) | ibid | `classifier_logreg.D_recall_to_K1` | 1024tok | ✅ |
| RF D-recall to K=1 | 0.444 (16/36) | ibid | `classifier_rf.D_recall_to_K1` | 1024tok | ✅ |
| Multi-signal vs best single | +0.4 pp | ibid | `multi_signal_minus_single_signal_pp` | 1024tok | ✅ |
| User 2pp bar met? | **False** | ibid | `user_threshold_2pp_met` | 1024tok | ✅ |
| Best single-signal at matched compute | 0.516 | ibid | `best_single_signal_at_matched_compute.overall_accuracy` | 1024tok | ✅ |

### §1b.vii — Breathing controls (Tests 2 & 3)

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| MATH-500 PR pos 50 (n=50) | 30.55 | `pathway11_h100/gibberish_control/pr_curves.json` | `conditions.math500_baseline.pr_by_position["50"].pr` | 1024tok | ✅ |
| MATH-500 PR pos 1 | 13.74 | ibid | `[...pos "1"...]` | 1024tok | ✅ |
| MATH-500 PR final | 22.19 | ibid | `[...pos "final"...]` | 1024tok | ✅ |
| Random-tokens PR pos 1 | 11.32 | ibid | `conditions.random_tokens.pr_by_position["1"].pr` | 1024tok | ✅ |
| Random-tokens PR final | 8.96 | ibid | `[...]` | 1024tok | ✅ |
| Stream-of-consc PR pos 10 (peak) | 15.66 | ibid | `conditions.stream_of_consciousness.pr_by_position["10"].pr` | 1024tok | ✅ |
| Stream-of-consc PR pos 1 | 4.60 | ibid | `[...]` | 1024tok | ✅ |
| No-CoT PR pos 1 (n=50) | 15.78 | `pathway11_h100/no_cot_control/pr_curves.json` | `conditions.nocot.pr_by_position["1"].pr` | 1024tok | ✅ |
| No-CoT PR final | 19.80 | ibid | `[...]` | 1024tok | ✅ |

### §1b.viii — Cross-architecture breathing (Exp 1)

| Claim | Value | Source file | JSON key/path | Labels | Valid? |
|---|---|---|---|---|---|
| Phi-3-mini MATH-500 T=0 acc | 0.448 (224/500) | `pathway11_h100/exp1_cross_model/results.json` | `models[0].accuracy` | 1024tok | ✅ |
| Phi-3-mini 2/3-depth layer | 21 | ibid | `models[0].twothirds_layer_idx` | 1024tok | ✅ |
| Llama-3.2-1B acc | 0.252 (126/500) | ibid | `models[1].accuracy` | 1024tok | ✅ |
| Llama 2/3-depth layer | 11 | ibid | `models[1].twothirds_layer_idx` | 1024tok | ✅ |
| Phi-3 PR pos 10 (point estimate, all) | 104.4 | ibid | `models[0].temporal.rows[1].all[0]` | 1024tok | ✅ |
| Llama PR pos 50 (point estimate, all) | 115.5 | ibid | `models[1].temporal.rows[3].all[0]` | 1024tok | ✅ (implicit by schema) |

### §1b.ix — Pathway 9 (SUPERSEDED label scheme — 256-tok NEW / 256-tok manifest)

These claims are still referenced in RESEARCH_SUMMARY §11 (what died) and
§1 (truncation artifact narrative). Every one needs the 256-token caveat
before citation as a current result.

| Claim | Value | Source file | JSON key/path | Labels | Status |
|---|---|---|---|---|---|
| ABC-44 raw holdout AUROC | 0.7961 | `pathway9/results/exp1_raw_length_v2.json` | `baseline_auroc` | 256tok_NEW | SUPERSEDED — headline was label-specific, 1024tok prefill+final 2-feature LR reaches ≈0.794 |
| ABC-44 deconfounded (raw length) | 0.7459 | ibid | `deconfounded_auroc` | 256tok_NEW | SUPERSEDED — length confound was feature-engineering residue; raw residual-stream DoM has cos(DoM, length) ≈ −0.09 at all layers |
| Drop from deconfounding | 0.050 | ibid | `drop_from_deconfounding` | 256tok_NEW | SUPERSEDED as above |
| Real PH 5-feat AUROC | 0.690 | `pathway9/results/exp3a_gaussian_null_v2.json` | `classifiers.real_ph.auroc_holdout` | 256tok_NEW | HOLDS — PH features are Gaussian-null in any label regime |
| Null (empirical-cov) PH AUROC | 0.693 | ibid | `classifiers.null_ph.auroc_holdout` | 256tok_NEW | HOLDS |
| XGB vs LR CV5 lift | −0.005 | `pathway9/results/exp4_xgboost_oinfo_v2.json` | `xgboost_lift_cv5` | 256tok_NEW | HOLDS — signal is linear; expected to reproduce at 1024tok but not re-measured |
| CoE MATH→BBH | 0.720 | `pathway9/results/exp5_cross_domain.json` | `transfer_matrix.coe.MATH_to_BBH_auroc` | 256tok_manifest | LIKELY HOLDS — Stage-5 DoM-transfer at 1024 tokens gives pooled 0.747 and symmetric 0.720, matching |
| CoE BBH→MATH | 0.712 | ibid | `transfer_matrix.coe.BBH_to_MATH_auroc` | 256tok_manifest | LIKELY HOLDS |
| PH-168 MATH→BBH | 0.354 | ibid | `transfer_matrix.layerwise_ph.MATH_to_BBH_auroc` | 256tok_manifest | HOLDS |
| PH-168 BBH→MATH | 0.497 | ibid | `transfer_matrix.layerwise_ph.BBH_to_MATH_auroc` | 256tok_manifest | HOLDS |
| D2H-lite MATH→BBH | 0.740 | ibid | `[...d2h_lite...]` | 256tok_manifest | HOLDS (asymmetric, not re-measured at 1024) |
| D2H-lite BBH→MATH | 0.529 | ibid | `[...]` | 256tok_manifest | HOLDS (asymmetric) |

### §1b.x — Pathway 8 layer-wise (SUPERSEDED label scheme)

| Claim | Value | Source file | JSON key/path | Labels | Status |
|---|---|---|---|---|---|
| ABC-44 holdout | 0.7961 | `pathway8_layerwise/results/exp2_results.json` | `models[0].auroc_holdout` | 256tok_NEW | SUPERSEDED (see §1b.ix) |
| CoE-60 holdout | 0.811 | ibid | `models[2].auroc_holdout` | 256tok_NEW | SUPERSEDED — not re-run against 1024-tok labels; L19 DoM Stage-5 is the current analog |
| D2H-lite holdout | 0.8055 | ibid | `models[3].auroc_holdout` | 256tok_NEW | SUPERSEDED |
| Layer-wise PH-168 | 0.6463 | `pathway8_layerwise/results/exp1_results.json` | `auroc_holdout` | 256tok_NEW | HOLDS — PH-under-null story is label-scheme-invariant |
| PH+CoE ensemble | 0.7584 | `pathway8_layerwise/results/exp2_results.json` | `models[5].auroc_holdout` | 256tok_NEW | SUPERSEDED (redundancy claim is invariant) |

### §1b.xi — Scratch pathway10 (256-tok NEW, SUPERSEDED)

| Claim | Value | Source file | JSON key/path | Labels | Status |
|---|---|---|---|---|---|
| 1.5B L19 DoM risk-coverage AUROC | 0.7525 | `scratch/pathway10_rc_5fold_results.json` | `methods["1.5B_L19_DoM"].auroc` | 256tok_NEW | SUPERSEDED — Pathway 11 prefill-gated compute re-measured at 0.7731 |
| 1.5B L19 DoM AURC | 0.6346 | ibid | `methods["1.5B_L19_DoM"].aurc` | 256tok_NEW | SUPERSEDED |
| Prefill temporal AUROC at pos 0 | 0.3921 (reversed) | `scratch/pathway10_temporal_and_verifier_results.json` | `T1_temporal_dom.positions[0].dom_holdout_auroc` | 256tok_NEW | SUPERSEDED — reversal was a label-distribution artifact; at 1024tok it's 0.7731 forward |
| cos(prefill DoM, final DoM) at old labels | 0.046 | ibid | `T1_temporal_dom.positions[0].cos_with_final_token_dom` | 256tok_NEW | APPROXIMATELY HOLDS — the rotation-problem narrative survives; exact number not re-measured in Pathway 11 |

**Flagging.** Everything labelled 256tok_* in the "Labels" column is at a
superseded label distribution. It is safe to cite the *qualitative* finding
(e.g., "PH is at Gaussian null") but not the *quantitative* value as
characterizing the current state of Qwen-2.5-1.5B. Re-measurement against
1024tok labels is on the backlog (§1e).

---

## 1c. Experimental conditions registry

One row per experiment with full reproducibility. Label scheme column matches
§1b.

| ID | Model | Layer | Position | Labels | Generation | Train/test split | Script | Output dir |
|---|---|---|---|---|---|---|---|---|
| Pathway-1 CORAL baseline | Qwen2.5-1.5B-Instruct | L15 (head) + all | last token | 256tok_NEW (104/500) | T=0, max=256 | SSS(9999), test=100 | `pathway1/phase1/holdout.py` | `pathway1/phase1/` |
| Pathway-2 Track A steering | Qwen2.5-1.5B-Instruct | L15 | injected at gen | 256tok_NEW | T=0, max=256 | tau sweep | `pathway2/track_a/phase1/` | `pathway2/track_a/` |
| Pathway-3 Phase 3 CV | Qwen2.5-1.5B-Instruct | L15/20/25 | multi-layer | 256tok_NEW | T=0.7, 32 samples | 5-fold CV | `pathway3/phase3_cv.py` | `pathway3/phase3/` |
| Pathway-4 Track A experiment 8 | Qwen2.5-1.5B-Instruct | L15 | last token | 256tok_NEW | T=0.7 adaptive K | held-out 100 | `pathway4/track_a/experiment8_adaptive_sampling/` | same |
| Pathway-5 Track A GSM8K | Qwen2.5-1.5B-Instruct | L15 | last token | 256tok (GSM8K own labels) | T=0, T=0.7 | 80/20 | `pathway5/track_a/phase_a{1-4}/` | same |
| Phase 6.5 1.5B MATH (truncated) | Qwen2.5-1.5B-Instruct | L15 | last token | 256tok_NEW | **T=0, max=256** | SSS(9999), test=100 | `pathway6_rebuild/phase1_prompt_model/` | `phase1_prompt_model/holdout_metrics_v2.json` |
| Phase 6.5 GSM8K (fixed) | Qwen2.5-1.5B-Instruct | L15 | last token | GSM8K own | **T=0, max=1024** | — | `pathway6_rebuild/phase6_5/` | `phase6_5/...gsm8k...` |
| Phase 6.5 7B MATH (fixed) | Qwen2.5-7B-Instruct | L15 | last token | 7B own | **T=0, max=1024** | — | `pathway6_rebuild/phase6_5/` | `phase6_5/.../7b_math.../` |
| Pathway-7 Phase 7.1 non-Euclidean PH | Qwen2.5-1.5B-Instruct | L15 | last token | 256tok_NEW | T=0, max=256 | SSS(9999) | `pathway7/phase71_noneuclid_math500.py` | `pathway7/results_phase71/` |
| Pathway-8 Exp 1 layer-wise PH | Qwen2.5-1.5B-Instruct | **all 28 layers** | last token | 256tok_NEW | T=0, max=256 | SSS(9999) | `pathway8_layerwise/exp1_layerwise_ph.py` | `pathway8_layerwise/results/exp1_results.json` |
| Pathway-8 Exp 2 CoE / D2H | same | all layers | last token | 256tok_NEW | T=0, max=256 | SSS(9999) | `pathway8_layerwise/exp2_method_comparison.py` | `exp2_results.json` |
| Pathway-8 Exp 5 cross-domain | 1.5B | all layers | last token | 256tok_manifest | T=0, max=256 | 80/20 with source-PCA | `pathway8_layerwise/exp5_cross_domain.py` | same |
| Pathway-9 Exp 1 v2 (length deconfound) | 1.5B | same | last token | 256tok_NEW | (replay) | SSS(9999) | `pathway9/exp1_raw_length_v2.py` | `pathway9/results/exp1_raw_length_v2.json` |
| Pathway-9 Exp 3a v2 (Gaussian null) | 1.5B | same | last token | 256tok_NEW | (replay, 5 null draws) | SSS(9999) | `pathway9/exp3a_gaussian_null_v2.py` | `pathway9/results/exp3a_gaussian_null_v2.json` |
| Pathway-9 Exp 5 (cross-domain) | 1.5B | all layers | last token | 256tok_manifest | T=0, max=256 | 80/20 with source-PCA | `pathway9/exp5_cross_domain.py` | same |
| **Pathway-11 Stage 1 (7B extraction)** | Qwen2.5-7B-Instruct | **all 28 layers + attn** | **per-token** | 1024tok (366/500) | T=0, **max=1024** | all 500 | `pathway11_h100/stage1_extract_math500_7b.py` | `pathway11_h100/data/math500_7b/` |
| **Pathway-11 Stage 2 (1.5B re-extract)** | Qwen2.5-1.5B-Instruct | all 28 layers | per-token | 1024tok (243/500) | T=0, **max=1024** | all 500 | `pathway8_layerwise/extract_math500.py` (reused) | `pathway8_layerwise/data/math500/` |
| **Pathway-11 Stage 3 (K=8 SC)** | 1.5B | L19 only | per-sample | 1024tok | **T=0.7, K=8, max=1024** | all 500 | `pathway11_h100/stage3_k8_selfconsistency.py` | `pathway11_h100/data/k8_selfconsistency/` |
| Pathway-11 Stage 4a (BBH) | 1.5B | all layers | per-token | BBH own labels | T=0, max=1024 | all 3×250 | `pathway8_layerwise/extract_bbh.py` (reused) | `pathway8_layerwise/data/bbh/` |
| Pathway-11 Stage 5 (L19 DoM transfer) | 1.5B | L19 | last token | 1024tok / BBH | (analysis only) | 5-fold OOF, 2000-boot | `pathway11_h100/stage5_bbh_dom_transfer.py` | `results/stage5_bbh_dom_transfer.json` |
| **Exp 2 prefill-gated compute** | 1.5B | L19 | prefill + final | 1024tok | K=1 @ T=0 + K=8 @ T=0.7 | 5-fold OOF (stratified, seed=9999) | `prefill_gated_compute/phase{1-5}_*.py` | same |
| **Exp 3 prefill-inversion** | 7B + 1.5B | L19 | prefill + final | 1024tok | — | 1000-boot, 100-boot balance-controlled | `prefill_inversion/phase{1-4}_*.py` | same |
| **Exp 2b multi-signal oracle** | 1.5B (7B-cached features) | L19 | prefill | 1024tok | — | 5-fold stratified CV | `multi_signal_oracle/phase{1-5}_*.py` | same |
| **Exp 1 cross-model breathing** | Phi-3-mini-4k / Llama-3.2-1B | 2/3 depth (L21 / L11) | per-token | own-model labels (224/500, 126/500) | T=0, max=1024 | n=500 | `exp1_cross_model/extract.py` + `analyze.py` | same |
| **Test 3 gibberish control** | 1.5B | L19 | per-token (7 positions) | own | T=0, max=**256** | n=20/20/50 | `gibberish_control/{gen_prompts.py, compute_pr.py}` | same |
| **Test 2 no-CoT control** | 1.5B | L19 | per-token (7 positions) | own | T=0, max=256, **answer-only system prompt** | n=50 | `no_cot_control/` + modified `exp1_cross_model/extract.py --system-prompt` | same |

**Consistency notes.**
- **Seed 9999** is the canonical splits seed for all 5-fold stratified OOF on
  MATH-500. Used in Pathway 6, 8, 9, 11.
- **SSS(9999, test=100)** is the 400/100 split used for the 0.7961 headline.
  Never compare a 5-fold OOF number to an SSS holdout number directly — they
  have different n_train and variance profiles.
- **Bootstrap N=1000** for prefill-inversion ratio CIs. **N=50** for Exp 1
  cross-model PR CIs (deliberately reduced — SVD is deterministic, CIs
  confirmatory).
- **Stage 1 did not capture attention** (`capture_attention=False` for 7B to
  avoid the eager-attention GQA path). Stage 2 (1.5B) captures attention
  entropy; use that if D2HScore attention features are needed.

---

## 1d. Negative results registry — what was tried and killed

Each entry states the direction, the decisive experiment, the evidence that
killed it, and the inference for future work. Listed roughly in the order the
program killed them.

### ND-1. Non-Euclidean PH on last-layer embeddings

- **What.** Replace Euclidean distance in the Rips filtration with diffusion,
  effective-resistance, cosine, or DTM distance.
- **Killed by.** Pathway 7 Phase 7.1 CPU sweep on MATH-500.
- **Decisive number.** Best non-Euclidean variant reached AUROC **0.774**,
  below the Euclidean baseline 0.7961. Source:
  `pathway7/results_phase71/` (NO-GO logged in commit 7d92047).
- **Implication.** The correctness signal in last-layer hidden states is not
  hiding in a non-Euclidean metric. Do not re-run unless a fundamentally
  different metric appears in the literature.

### ND-2. Layer-wise PH (28 layers × 6 PH features = 168 dims)

- **What.** Distribute PH extraction across all 28 transformer layers, each
  reduced to 20 PCA components, with 6 PH summary statistics per layer.
- **Killed by.** Pathway 8 Exp 1 and Pathway 9 Exp 2.
- **Decisive number.** Holdout AUROC **0.6463** vs single-layer ABC-44 baseline
  0.7961 (loss −0.15). In ensemble with CoE-60 (0.811 alone), stacked
  ensemble adds only +0.001. PH-168 is redundant with CoE-60.
- **Source.** `pathway8_layerwise/results/exp1_results.json`,
  `pathway9/results/exp2_orthogonality.json`.
- **Implication.** "Distribute PH over depth" didn't help. The same signal is
  captured more efficiently by direction-of-mean projections at a single
  well-chosen layer.

### ND-3. PH features carry topology-specific signal beyond covariance

- **What.** The claim behind the name "topo-confidence": persistent homology
  captures loops / connected components that Gaussian-matched covariance
  cannot.
- **Killed by.** Pathway 9 Exp 3a v2 (empirical-covariance SVD null).
- **Decisive number.** 5 raw PH features (H0_max_lifetime, H0_entropy,
  H1_pers_entropy, H1_n_features, H1_max_lifetime) at holdout AUROC **0.690**
  vs rank-matched null at **0.693** — real vs null gap is essentially zero.
- **Source.** `pathway9/results/exp3a_gaussian_null_v2.json`.
- **Related.** Predicted by Tuci et al. Sharpness-Dimension framework — at
  edge-of-stability a trained network's residual stream is approximately
  Gaussian in the top directions.
- **Implication.** PH is not a fundamentally topological signal. It's
  covariance structure with a different summary statistic. The "ABC-44
  headline" is geometry (cosines, norms, velocities), not topology.

### ND-4. CoE (60-dim layer trajectory features) adds signal beyond single-layer DoM

- **What.** Wang ICLR 2025's Chain-of-Embedding computes magnitude/angle change
  per layer → 60 trajectory features per problem.
- **Partially killed by.** Pathway 11 Stage 5.
- **Decisive number.** Stage-5 single-layer L19 DoM symmetric MATH↔BBH
  transfer = **0.7199** vs Pathway 9 CoE-60 symmetric transfer = **0.716**.
  Δ = +0.004. A single L19 DoM direction matches 60 trajectory features.
- **Source.** `pathway11_h100/results/stage5_bbh_dom_transfer.json`
  (`verdict_vs_coe` = 0.0039).
- **Caveat.** CoE was not re-measured within-domain at 1024-tok labels. If a
  future re-run shows CoE > 0.794 at 1024 tok, this conclusion would need
  revisiting.
- **Implication.** CoE is not a richer signal than a single-layer DoM. Use
  DoM unless you specifically need per-layer features (e.g., for a
  layer-localization study).

### ND-5. E1 fixed-vector steering (single final-token direction)

- **What.** Pathway 10 v2's original E1: fit probe on final-token L19
  activations, use the weight vector as an ITI-style steering direction.
- **Killed by.** Direction-rotation analysis in `scratch/pathway10_temporal_and_verifier_results.json` and `pathway10_v2.md`.
- **Decisive number.** cos(prefill_DoM, final_DoM) = **0.046** on the 1.5B
  (256-tok labels; qualitative result survives the label correction). At every
  intermediate position 0–69 the cosine stays in [0.00, 0.19]. Reaches
  ~0.37 only by position 200. The pre-cached Pathway-2 steering vector
  (L15/L19) has cos ≈ 0.05 with the current L19 final-token DoM — unusable.
- **Source.** `scratch/pathway10_temporal_and_verifier_results.json`
  (T1_temporal_dom).
- **Implication.** The correct-vs-incorrect direction rotates through
  generation. A final-token-fit steering vector is injected into
  approximately the wrong subspace at mid-generation. Honest steering would
  require a per-position DoM bank — never built, scoped but deferred.

### ND-6. 7B as a superior verifier (cross-scale motivator)

- **What.** Pathway 10 v1 Direction A / E4 distillation: 7B residual stream
  encodes correctness better than 1.5B's. Use 7B to verify 1.5B outputs, or
  distill 7B→self signal into 1.5B.
- **Killed by.** Pathway 11 Stage 1 + Stage 5.
- **Decisive numbers.**
  - 7B L19 DoM → predict 1.5B correctness: **0.717** AUROC.
  - 1.5B L19 DoM → predict 1.5B correctness (self): **0.719** AUROC.
  - No cross-scale verifier advantage. The prior 0.818 / 0.754 asymmetry was
    a 256-tok artifact where the 7B was only 81/500 correct.
- **Source.** RESEARCH_SUMMARY §6 (reconstructable from
  `data/math500_7b/` + `pathway8_layerwise/data/math500/`); no single JSON
  encodes both numbers in one file.
- **Implication.** Cross-scale probe transfer is not a lift direction.
  E4 distillation deferred until a new motivator appears.

### ND-7. Monotonic compute gating at mid-confidence

- **What.** Pathway-10-v2 E2: a threshold τ on prefill DoM sends low-confidence
  problems to K=8, high-confidence to K=1.
- **Killed by.** Pathway 11 Exp 2.
- **Decisive numbers.**
  - Uniform K=4 = 0.495 accuracy.
  - Prefill τ=40%, avg K=3.80 → 0.492 (− 0.003).
  - Prefill τ=50%, avg K=4.50 → 0.492 (− 0.003 vs uniform K=4.5 middle-heavy).
  - Middle-heavy (Q4,Q1 → K=1; Q3,Q2 → K=8), avg K=4.50 → 0.526 (+0.031 vs
    uniform K=4, but +0.016 below neg-seq-len top-heavy).
  - Neg-seq-len top-heavy, avg K=4.50 → **0.542**.
- **Source.** `pathway11_h100/prefill_gated_compute/results.json`.
- **Implication.** The B-bucket (K=8-recoverable) lives at *middle* prefill
  score, not bottom. A monotonic gate sends compute to the wrong cohort.
  Length is Pareto-dominant because it correlates more tightly with
  "needs more sampling" than prefill DoM does (Spearman 0.327 vs 0.175).

### ND-8. Multi-signal D-bucket detection

- **What.** 3-feature logreg ({prefill DoM, per-problem local PR, seq-len})
  routes K=1-right K=8-majority-wrong problems to K=1.
- **Killed by.** Pathway 11 Exp 2b.
- **Decisive numbers.**
  - Logreg macro-F1 = 0.548 on the 3-class K1_AD / K8_B / refuse_C target.
  - **D-recall = 15/36 = 41.7%** (logreg) / **44.4%** (RF) — both below the
    48.6% class-prior baseline of just routing to class 0.
  - Drop-one-out: dropping `prefill_lpr` changes macro-F1 by **0.000**.
    Local per-problem PR is dead weight.
  - Multi-signal vs best single-signal at matched compute = **+0.4 pp**, below
    user's ≥2pp bar.
- **Source.** `pathway11_h100/multi_signal_oracle/results.json`.
- **Implication.** The D-bucket has a collective geometric signature (group
  PR 14.49 vs A/B/C at 18.7/16.6/20.0) but it does not survive per-problem
  k-NN localization. Density-based outlier methods (LOF, isolation forest)
  on D's 36 activations were not tried and remain on the backlog.

### ND-9. Orthogonal feature stacking beyond two directions

- **What.** Combine prefill DoM + final-token DoM + trajectory PR (or other
  "orthogonal" per-problem features) as a multi-feature LR.
- **Killed by.** Exp 2b ablation + trajectory-PR single-feature AUROC.
- **Decisive numbers.** Trajectory PR as a feature → AUROC 0.574. Adding it
  to {prefill DoM, final DoM} → +0.027 macro-F1 (below 2pp bar). `prefill_lpr`
  → +0.000. Extended 6-feature set (+ finaltok features, mean_logprob) →
  +0.027 F1 overall but the two extra features are post-generation, not
  gating signals.
- **Implication.** Two directional features saturate the inference-time
  gating signal on this dataset.

### ND-10. The 20.8% MATH-500 baseline

- **What.** Every pathway-10-v2 success criterion was calibrated against
  "1.5B gets 104/500 correct = 20.8%."
- **Killed by.** Pathway 11 Stage 2 (simply re-ran at `max_new_tokens=1024`).
- **Decisive number.** 1.5B at 1024 tokens: **243/500 correct = 48.6%**. K=8
  ceiling = 55.0%. Inference-time intervention headroom is ~6 accuracy
  points, not ~30.
- **Implication.** Every prior baseline framing was 2.3× off. Any paper-like
  narrative using the 20.8% number needs footnoting or rewriting.

### Negative-result-that-strengthened-a-finding (2026-05-02 FE880)

Not a graveyard entry in the usual sense — a pre-registered overturning
condition for F-10 that was tested and failed in a way that strengthened
the finding. Logged here so the test-and-result is on file.

- **What.** F-10 says PH = matched-cov Gaussian null on raw L19 clouds.
  The pre-registered overturning condition was: if PH features on
  PC1-residualized clouds beat matched-cov Gaussian null by ≥ 0.05, the
  finding flips ("PH carries non-covariance signal once the dominant
  direction is removed").
- **Outcome.** Real PH on PC1-residualized clouds reached AUROC **0.6884**
  vs matched-cov Gaussian null **0.7628**, gap **−0.074**. The condition
  failed by 12.4 pp in the wrong direction.
- **Source.** `pathway11_h100/ph_residuals/pc1_resid_results.json` (FE880,
  EXP-56, 2026-05-02).
- **Implication.** F-10 strengthens. PH does not survive PC1 removal —
  matched-cov Gaussian null on the residualized covariance does. The
  cov-spectrum probe (FE881, 0.7928) reads what's left after PC1 is gone.

---

## 1e. Open questions — next-experiment designs

Each open question has a one-paragraph experiment design, rough cost estimate,
and the one decisive number that would resolve it.

### OQ-1. Does breathing happen on short CoT (non-math)?

**Design.** Prompt 1.5B with "Explain in one sentence and give the answer in
\boxed{...}" on MATH-500 (first 50). Expected median 20–40 tokens. Extract
L19 per-token at positions {1, 5, 10, 15, 20, 30, final}. Compare PR curve
to the no-CoT control (median 2 tokens, no trajectory) and the full CoT
(median 255+, classic 13→30→22 breathing).

**Decisive number.** Peak PR in the 20–40 token span. If ≥ 25 → breathing
happens over short trajectories (tracks generation length). If ≤ 15 →
breathing requires extended reasoning.

**Cost.** ~5 min H100 + ~2 min local CPU analysis. Reuses the existing
`--system-prompt` flag added in Test 2. ~$0.25.

### OQ-2. Does breathing happen on code generation?

**Design.** HumanEval Python task prompts (164 problems). Extract L19
per-token at the same 7 positions as Test 3. Compare PR shape to MATH-500.

**Decisive number.** Peak PR value and position, and correct vs incorrect
final-PR ratio. If code shows the same three-phase structure, breathing is a
general reasoning-task property. If flat (like random tokens), breathing is
math-specific.

**Cost.** ~10 min H100 + analysis. ~$0.50.

### OQ-3. Per-position DoM bank for honest steering

**Design.** Extract L19 activations at every intermediate token position
(positions 0..1023 on a subset of 200 MATH-500 problems). Fit a separate DoM
direction at each position. Attempt ITI-style steering at inference using the
direction appropriate to the current token position.

**Decisive number.** Δaccuracy on MATH-500 holdout at best (α, position)
vs baseline α=0.

**Cost.** Re-extraction is the existing Stage-2 code but N=200 instead of 500
and saving full per-token instead of summarizing — ~6 GB. GPU extraction
~45 min. Steering experiment ~2h GPU (α sweep, 3 positions × 5 α values ×
100 generations). Total ~1 H100-day. ~$75.

**Expected outcome.** Unknown. This is the single experiment most likely to
revive E1.

### OQ-4. Dimensional breathing ↔ Sharpness Dimension / EoS

**Design.** Re-train Qwen-2.5-1.5B from a recent checkpoint for 100 steps on
math problems. Record (a) covariance participation ratio trajectory at L19
during inference-time breathing on a held-out set, (b) top-Hessian eigenvalue
trajectory during the 100 training steps. Test whether inference-time PR and
training-time Sharpness-Dimension share a common expand-then-collapse curve
shape or a common spectral signature.

**Decisive number.** Pearson correlation between normalized PR(t) during
inference and normalized λ_1 trajectory during training, across matched
checkpoints.

**Cost.** Non-trivial — fine-tuning access to the Qwen-2.5-1.5B checkpoint.
~3 H100-days. ~$200. Tuci et al. (2604.19740) framework is the blueprint.

### OQ-5. Does prefill encode familiarity vs decomposability?

**Design.** Two-axis analysis of the 243 1.5B-correct and 257 incorrect
problems at prefill time:
1. **Familiarity axis:** influence-function attribution of each MATH-500
   problem to the 2026 MATH training set (or a proxy — closest neighbor
   cosine to MATH training problems in embedding space).
2. **Decomposability axis:** proof-depth tagging (number of lemmas needed).
   Can be approximated by expected-CoT-length on a held-out 7B run.
Regress prefill DoM score on both axes + their interaction.

**Decisive number.** R² of each axis alone vs R² of the combined model. If
familiarity dominates, "prefill knows best" is a memorization signal. If
decomposability dominates, it's a structural signal.

**Cost.** Influence functions are hard — ~3 H100-days. Decomposability via
length proxy is cheap (already in the data). Start with the length-proxy
version. ~$0–$200 depending on scope.

### OQ-6. Label-scheme re-baselining of Pathway 9 (CoE within-domain)

**Design.** Re-run `pathway9/exp2_orthogonality.py` and `exp5_cross_domain.py`
against 1024-tok labels (243/500 on MATH, 366/500 on 7B MATH — note 7B wasn't
in Pathway 9). CoE features can be recomputed from Stage-2 all-layer data.

**Decisive number.** CoE-60 within-domain at 1024-tok labels. If still
≥ 0.80, the CoE-beats-ABC-44 headline survives. If drops to ≤ 0.72
(matching the current L19 DoM), CoE is no longer distinguishable from
DoM — the single-direction story is the full story.

**Cost.** CPU only, ~30 min. Zero dollar.

### OQ-7. Dense-by-outlier D-bucket detection

**Design.** Train Local Outlier Factor (LOF) and Isolation Forest on the 36
D-bucket prefill activations (plus negative samples from A/B/C). Evaluate
D-recall on held-out folds.

**Decisive number.** D-recall at matched false-positive rate vs the 41.7%
Exp 2b logreg baseline.

**Cost.** CPU only, ~15 min. Zero dollar.

### OQ-8. Does breathing pattern change after fine-tuning?

**Design.** Take Qwen-2.5-1.5B-Instruct, SFT for 1000 steps on MATH. Compare
pre/post breathing curves (7 positions × 500 problems).

**Decisive number.** Peak PR and correct-vs-incorrect final-PR ratio before
vs after. If breathing flattens → it's a "problem-difficulty diversity"
effect that fine-tuning specializes. If it sharpens → fine-tuning makes the
three-phase structure more pronounced.

**Cost.** 1 H100-day SFT + ~30 min re-extraction. ~$75.

### OQ-9. 7B prefill inversion at different difficulty levels / other benchmarks

**Design.** Run Exp 3 on GSM8K (easier), AIME (harder), competition
benchmarks. Does the 7B inversion track aggregate difficulty or problem-type?

**Decisive number.** PR correct/incorrect ratio per benchmark, level-stratified.

**Cost.** Stage 1 re-extraction scope: GSM8K (1319 problems) ~6 h H100. AIME
(30 problems) negligible. ~$18 for GSM8K.

---

## 1f. Literature connections

Each major finding is linked to 2–3 most-relevant arXiv papers with a
one-sentence connection. Full per-paper detail lives in `PAPER_INDEX.md`.

### Steering / representation engineering (for E1 and the steering-vector negative)

- **Inference-Time Intervention (Li et al., 2306.03341).** The canonical
  probe-as-steering-vector template; our E1 was the direct analog on a math
  benchmark. ND-5 shows why their approach fails when the direction rotates
  across positions.
- **Adaptive Layer-wise Steering / ALS (2509.18116).** Per-layer DoM steering
  on Qwen. Closest to what a per-position DoM bank (OQ-3) would be.
- **ReDeEP + AARF (2410.11414).** Head-level mechanistic interp with
  mitigation. Relevant if E1 is revived and goes head-specific.

### Confidence estimation / selective prediction (for E3 and the prefill-DoM headline)

- **"The LLM Already Knows" (2509.12886).** Prefill/prompt-only signals
  predict downstream output quality. Directly supports our "prefill knows best"
  finding — §4 of RESEARCH_SUMMARY.
- **"Knowing When to Quit" (2604.18419).** Probe-driven abstention. Our
  §8 selective prediction is this paper's thesis applied to a reasoning task.
- **CCPS (2505.21772).** Perturbation-based calibration probe. Relevant if
  E3's refusal threshold needs robustness validation.
- **SEPs (2406.15927).** Semantic-entropy probes. A label-free alternative to
  the DoM probe that wasn't tested — motivates an extension of E3.

### Chain-of-Embedding / trajectory features (for §2 redundancy and ND-4)

- **Chain-of-Embedding (Wang ICLR 2025, 2410.13640).** The 60-dim trajectory
  features we reproduce + deprecate in favor of L19 DoM.
- **D²HScore (2509.11569).** Dispersion/drift features. Comparable to CoE and
  similarly redundant with single-layer DoM in our data.

### Dimensional breathing / covariance dynamics (for §3)

- **Sharpness Dimension / EoS / Tuci et al. (2604.19740).** Training-time
  Hessian has an expand-then-collapse behavior; our finding is an
  inference-time analog on the residual-stream covariance. Also predicts
  PH-under-null (ND-3).
- **Intrinsic Dimension / TwoNN (Pope et al., classic).** We tested TwoNN
  directly in Pathway 8 (0.407 AUROC — didn't work). But the PR "effective
  dimensionality" story intersects this literature.

### Self-consistency / compute gating (for §7 and ND-7)

- **Wang 2022 Self-Consistency.** The K-sample majority-vote baseline we
  tried to beat. Our finding: on MATH-500-1.5B, the 3× compute assumption
  holds (K=8 gives +6.4pp over K=1 at 8× compute).
- **Brumm et al. (2510.07364).** Inference-time scaling of CoT. Sets the
  general framework our compute-allocation results sit in.
- **Zhang et al. self-verification (2504.05419).** Multi-stage self-verification
  pipelines. An alternative to K=8 majority that we didn't test.

### Cross-scale / distillation (for ND-6 and the deferred E4)

- **Chen et al. cross-scale stitching (2506.06609).** Stitches small-model
  layers into big-model backbone. Relevant if E4 distillation is revived —
  the "match 7B L19 activations in 1.5B" recipe.

---

## 1g. Artifact map

Comprehensive file inventory. Organized by pathway. Filter out `.venv/*`,
`.pytest_cache/*`, and `.git/*`.

### Top-level

| File | Size | Purpose |
|---|---|---|
| `PROJECT_RECORD.md` | this file | Authoritative reference. |
| `RESEARCH_SUMMARY.md` (in `pathway11_h100/`) | 52 KB | Narrative synthesis of P9/P10/P11 written 2026-04-24. |
| `CLAUDE.md` | 5.4 KB | Top-of-repo orientation — out of date for Pathway 11, kept for Pathway 6.5 wording. |
| `README.md` | 7.8 KB | Public README for GitHub. Refers to Pathway 6.5. |
| `pyproject.toml` | 1.1 KB | Package metadata. |
| `LICENSE` | 1 KB | — |
| `runpod_launch.sh` | 3.9 KB | Historical RunPod launcher (Pathway 6.5 era). |
| `experiments.log` / `experiments_remaining.log` | 16 / 93 KB | Pathway-1 era running log. |
| `validate_claims.py` | 2026-05-03 | Provenance validator (172/172 internal PASS, 216 claims tracked). |
| `validation_report.txt` | 2026-05-03 | Output of the above. |
| `SYNTHESIS.md` | 2026-05-03, 18.7 KB | 10-min practitioner briefing. What survived / overturned / new. |
| `NOVELTY_AUDIT.md` | 2026-05-03, 21.5 KB | F-1..F-10 ranked novel / refines / parallel-discovery vs research-graph. |
| `APPLICATIONS.md` | 2026-05-03, 17.8 KB | Six deployment surfaces with honest production checklists. |
| `QUICKSTART.md` | new | Cold-pickup one-pager. |
| `DATA_MANIFEST.md` | new | Cached-activation inventory. |
| `PERSPECTIVES.md` | new | Reflective notes. |
| `EXPERIMENT_LOG.md` | new | Running log (EXP-### entries; EXP-58 latest). |
| `PAPER_INDEX.md` | new | Paper-to-repo bridge. |
| `HYPOTHESES.md` | new | Priority queue of untested hypotheses (H-1..H-22+). |
| `STATE.md` | new | Where-was-I snapshot (overwritten). |
| `FINDINGS.md` | new | Registry of surviving findings (F-1..F-10). |
| `figures/` | new | Consolidated PNG archive. |
| `handoff/` | new | Per-session handoff notes. |

### Pathway 1 (2026-04-04 → 12)

```
pathway1/
├── PATHWAY1_RESULTS.txt
├── phase0/winning_features.py       # 1407-line frozen CORAL extractor
├── phase0/artifact_locks.json
├── phase1/holdout_metrics.json      # Original 0.7961 result (pre-audit)
├── phase2/ablation_cv_results.json  # Tier A/B/C/D sweep
├── phase2/tier_assignments.json
├── phase2/ablation_holdout_results.json
├── phase2/ablation_plot.png
├── phase3/seed_sensitivity_cv500.json
├── phase3/seed_sensitivity_cv_train400.json
├── phase3/seed_sensitivity_holdout.json
├── phase4/routing_metrics.json
├── phase4/calibration_metrics.json
├── phase4/operating_point.json
├── phase4/routing_tradeoff_curve.png
└── phase4/calibration_plot.png
```

### Pathway 2 (2026-04-12 → 13)

```
pathway2/
├── track_a_prompt.md
├── track_b_prompt.md
└── track_a/SUMMARY.md
    (subdirs with per-t-threshold checkpoints, largely ephemeral)
```

### Pathway 3 (2026-04-13 → 14)

```
pathway3/
├── SUMMARY.md / HANDOFF.md
├── phase1/...                      # Expansion: temperature_generations, expansion_summary
├── phase2/prototypes/...           # 12 cosine matrices × cluster sizes (layer×K)
├── phase3/cv_results.json          # 4 configs × 5 folds × 6 checkpoint JSONs
├── phase3/locked_config.json
├── phase4/holdout_results.json
├── phase4/statistical_tests.json
├── phase4/go_no_go_decision.json
└── phase4/per_problem_analysis.json
```

### Pathway 4 (2026-04-15 → 16)

```
pathway4/
├── HANDOFF.md
├── track_a/
│   ├── phase1/completion_scores.json, feature_names.json, phase1_summary.json
│   ├── phase2/cv_results.json, strategy_comparison.json, fold_assignments_train400.json, locked_strategy.json
│   ├── phase3/holdout_results.json, comparison.json, statistical_tests.json, go_no_go_decision.json, holdout_*_generations.json
│   ├── experiment1_ablation/ablation_results.json, per_problem_detail.json
│   ├── experiment5_per_completion/verifier_results.json, within_problem_aurocs.json
│   ├── experiment6_ablation/tier_ablation_results.json, feature_importance.json
│   ├── experiment8_adaptive_sampling/adaptive_results.json, pareto_data.json
│   ├── experiment9_baselines/baseline_results.json
│   └── experiment10_calibration/calibration_results.json, reliability_diagram_data.json
└── track_b/holdout_results.json, comparison.json, training_log.json, etc.
```

### Pathway 5 (2026-04-16 → 17)

```
pathway5/
├── track_a/phase_a{1,2,3,4}/       # GSM8K 1.5B, 4 phases
├── track_b/phase_b{1,2,3}/         # MATH-500 7B, 3 phases
└── track_c/feature_comparisons.json, cross_model_results.json
```

### Pathway 6 rebuild (2026-04-17 → 19)

```
pathway6_rebuild/
├── HANDOFF.md, audit_e2e.py, common.py
├── phase0_relabel/ (+47 correct answers fix)
├── phase1_prompt_model/
│   ├── experiment{9,6,10}_v2.json
│   ├── holdout_metrics_v2.json       # 0.7961 holdout vs baselines
│   └── per_problem_scores_v2.json
├── phase2_completion/                 # MV + gating experiments
├── phase3_cross_benchmark/gsm8k/
│   ├── temperature_generations.json
│   ├── greedy_answers.json
│   ├── model_results.json
│   ├── summary.json
│   ├── selection_results.json
│   └── feature_names.json
├── phase4_report/FINAL_REPORT.json
└── phase6_5/
    ├── FINAL_SUMMARY.md               # The deconfounded analysis
    ├── HANDOFF.md
    └── (phase 6.5 run artifacts — 1024-token GSM8K and 7B MATH rebuild)
```

### Pathway 7 (2026-04-19 → 20)

```
pathway7/
├── PLAYBOOK.md
├── runpod_pathway7.sh, Dockerfile, requirements_pathway7.txt
├── distance_metrics.py                # Eff-res, cosine, diffusion, DTM PH
├── feature_extractor_v2.py            # 44 + 18 non-Euclidean
├── phase71_noneuclid_math500.py       # NO-GO 0.774
├── zigzag_features.py                 # Zigzag persistence (unused in this pathway)
├── extract_all_layers.py
├── results_phase71/
├── benchmarks/                        # HumanEval + BBH pipelines
├── results_humaneval/
├── results_bbh/
└── validation/                        # Synthetic shapes, metric divergence
```

### Pathway 8 (2026-04-20 → 21)

```
pathway8_layerwise/
├── config.py
├── extraction_utils.py                # Reused by P11 stages 2, 4a
├── extract_math500.py                 # Reused by P11 stage 2
├── extract_bbh.py                     # Reused by P11 stage 4a
├── extract_humaneval.py
├── layerwise_features.py
├── coe_features.py                    # Shared with P9/P11 analyses
├── d2hscore_features.py
├── twonn_features.py
├── crosslayer_features.py
├── diagnostics.py
├── exp1_layerwise_ph.py .. exp5_cross_domain.py
├── runpod_pathway8.sh
├── results/exp{1..5}_results.json
└── data/
    ├── math500/                       # 500 × 1.5B per-token all-layer npz, 19 GB (gitignored)
    ├── math500/manifest.json
    ├── bbh/                           # 750 × BBH per-token all-layer npz, 18 GB (gitignored)
    └── bbh/manifest.json
```

### Pathway 9 (2026-04-22)

```
pathway9/
├── SUMMARY.md, HANDOFF.md / HANDOFF_v2.md / HANDOFF_v3.md, AUDIT.md
├── literature-sweep.md, literature-sweep.cypher
└── results/
    ├── exp1_length_deconfound.json, exp1_raw_length_v2.json
    ├── exp2_orthogonality.json
    ├── exp3a_gaussian_null.json (v1), exp3a_gaussian_null_v2.json
    ├── exp3b_token_shuffle.json, exp3c_count_control.json
    ├── exp4_xgboost_oinfo.json (v1), exp4_xgboost_oinfo_v2.json
    ├── exp5_cross_domain.json
    └── run.log
```

### Pathway 10 (2026-04-23) — planning only

```
pathway10_v1.md, pathway10_handoff_v1.md, pathway10_v2.md, pathway10-papers.md
scratch/
├── pathway10_five_questions_results.json
├── pathway10_rc_5fold_results.json
├── pathway10_risk_coverage_results.json
├── pathway10_temporal_and_verifier_results.json
└── pathway10_quartile_compute_sim_results.json
(each paired with a .py and .log)
```

### Pathway 11 (2026-04-23 → 24)

```
pathway11_h100/
├── RESEARCH_SUMMARY.md                # 52 KB canonical synthesis
├── HANDOFF_compact.md                 # 3.2 KB live pod handoff
├── runpod_pathway11.sh                # 7-stage orchestrator
├── config.py
├── stage1_extract_math500_7b.py       # 7B all-layer (capture_attention=False)
├── stage3_k8_selfconsistency.py       # K=8 T=0.7 + per-sample L19
├── stage4b_bbh_per_subset.py
├── stage5_bbh_dom_transfer.py
├── k8_spread_analysis.py
├── logs/stage{0..5}_*.log
├── results/
│   ├── stage{0..5}.done               # checkpoint markers
│   ├── stage4b_bbh_per_subset.json
│   ├── stage5_bbh_dom_transfer.json
│   └── k8_spread_analysis.json
├── data/
│   ├── math500_7b/                    # 42 GB — 500 × 7B per-token all-layer npz (gitignored)
│   ├── math500_7b/manifest.json
│   └── k8_selfconsistency/            # 13 MB — 500 × K=8 per-sample L19 (gitignored)
│
├── prefill_gated_compute/             # Exp 2
│   ├── SUMMARY.md, results.json, pareto_plot.png
│   ├── phase1_majority_vote.npz/json/py
│   ├── phase2_prefill_dom.npz/json/py
│   ├── phase3_policies.json, phase3b_realistic_policies.json
│   ├── phase4_oracle.json
│   └── phase5_plot_and_summary.py
│
├── prefill_inversion/                 # Exp 3
│   ├── SUMMARY.md, results.json
│   ├── phase1_bootstrap.json/py
│   ├── phase2_cross_scale.json/py, phase2_loo_deltas.npz
│   ├── phase3_mechanistic.json/py, phase4_summary.py
│   ├── bootstrap_ci.png, three_way_split.png, d_bucket_signature.png
│   ├── common.py
│   └── cache/m7b_prefill.npz, m15b_prefill.npz
│
├── multi_signal_oracle/               # Exp 2b
│   ├── SUMMARY.md, results.json
│   ├── features.npz, oof_{logreg,rf}.npz
│   ├── phase{1,2,3,4,5}_*.json/py
│   └── pareto_comparison.png
│
├── exp1_cross_model/                  # Cross-architecture breathing
│   ├── HANDOFF.md, results.json
│   ├── extract.py, analyze.py, run.sh
│   ├── temporal_pr_curve.png, depth_pr_prefill_vs_final.png
│   ├── logs/full_run.log, analyze.log
│   └── data/phi3mini/ (166 MB), data/llama32_1b/ (65 MB) [gitignored]
│
├── gibberish_control/                 # Test 3
│   ├── HANDOFF.md, pr_curves.json, gibberish_vs_math500.png
│   ├── gen_prompts.py, compute_pr.py, plot_curves.py, run.sh
│   ├── prompts_random.json, prompts_stream.json
│   └── data/{random,stream,math500}/ (14 MB total) [gitignored]
│
└── no_cot_control/                    # Test 2
    ├── HANDOFF.md, pr_curves.json, nocot_vs_cot.png
    ├── compute_pr.py, plot_curves.py, run.sh
    └── data/nocot/ (7 MB) [gitignored]
```

### Durable memory entries (in `~/.claude/projects/-home-musicofhel/memory/`)

Out-of-repo but part of the project record:

- `topo-confidence.md` — top-level project state
- `pathway8-handoff.md` — Pathway 8 per-stage status
- `pathway9-coe-pivot.md` — Pathway 9 conclusion
- `pathway10-v2-steering.md` — v2 plan
- `pathway11-h100-orchestrator.md` — Pathway 11 orchestrator scope
- `pathway11-pod-keep-alive.md` — pod stop-not-remove pattern
- `topo-confidence-gibberish-control.md` — Test 3 finding
- `topo-confidence-nocot-control.md` — Test 2 finding
- `feedback-topo-confidence-goal.md` — "Goal is model improvement, not papers"
- `feedback-topo-confidence-headline-resolution.md` — "Trim scope but keep headline-figure resolution"

### Ephemeral / scratch

- `experiments.log`, `experiments_remaining.log` — Pathway-1 era
- `pathway11_h100/__pycache__/` — bytecode, safe to delete
- `.pytest_cache/` — safe to delete
- pathway 2/3/4 per-checkpoint JSONs (checkpoint intermediates, the
  per-pathway SUMMARY.md + final results json are load-bearing; the fold
  checkpoints are reproducibility-only)

### Cached activations (gitignored, regenerable)

Total: ~79 GB on-disk. All regenerable from the extraction scripts.

| Path | Size | Model | Benchmark | Gen config | Contents |
|---|---|---|---|---|---|
| `pathway11_h100/data/math500_7b/` | 42 GB | Qwen2.5-7B | MATH-500 | T=0, max=1024 | 500 × per-token all-layer, no attn |
| `pathway8_layerwise/data/math500/` | 19 GB | Qwen2.5-1.5B | MATH-500 | T=0, max=1024 | 500 × per-token all-layer + attn |
| `pathway8_layerwise/data/bbh/` | 18 GB | Qwen2.5-1.5B | BBH 3×250 | T=0, max=1024 | 750 × per-token all-layer |
| `pathway11_h100/data/k8_selfconsistency/` | 13 MB | Qwen2.5-1.5B | MATH-500 | T=0.7, K=8 | 500 × K=8 × L19 only |
| `pathway11_h100/exp1_cross_model/data/phi3mini/` | 166 MB | Phi-3-mini-4k | MATH-500 | T=0, max=1024 | 500 × (prefill-all + final-all + L21 per-token) |
| `pathway11_h100/exp1_cross_model/data/llama32_1b/` | 65 MB | Llama-3.2-1B | MATH-500 | T=0, max=1024 | 500 × (prefill-all + final-all + L11 per-token) |
| `pathway11_h100/gibberish_control/data/*` | 14 MB | Qwen2.5-1.5B | synthetic | T=0, max=256 | 20/20/50 × L19 per-token |
| `pathway11_h100/no_cot_control/data/nocot/` | 7 MB | Qwen2.5-1.5B | MATH-500 | T=0, answer-only | 50 × L19 per-token (median 2 tokens) |
| `pathway11_h100/prefill_inversion/cache/` | 7.6 MB | both | MATH-500 | T=0 | Cached fp16 prefill+final for fast replay |

Older pathway-5 trajectory NPZ files (`~1 MB` each in `pathway5/track_*/`) are
also gitignored but remain on disk.

### Figures (all PNG, final state pre-archive)

The `figures/` directory will contain standardized-name copies. Originals:

- `pathway11_h100/exp1_cross_model/temporal_pr_curve.png` → `breathing_temporal_phi3_llama.png`
- `pathway11_h100/exp1_cross_model/depth_pr_prefill_vs_final.png` → `exp1_depth_pr_prefill_vs_final.png`
- `pathway11_h100/prefill_gated_compute/pareto_plot.png` → `e2_pareto_frontier.png`
- `pathway11_h100/prefill_inversion/bootstrap_ci.png` → `e3_bootstrap_ci.png`
- `pathway11_h100/prefill_inversion/three_way_split.png` → `e3_three_way_split.png`
- `pathway11_h100/prefill_inversion/d_bucket_signature.png` → `e3_d_bucket.png`
- `pathway11_h100/multi_signal_oracle/pareto_comparison.png` → `e2b_pareto_comparison.png`
- `pathway11_h100/gibberish_control/gibberish_vs_math500.png` → `breathing_gibberish_control.png`
- `pathway11_h100/no_cot_control/nocot_vs_cot.png` → `breathing_nocot_control.png`
- `pathway1/phase2/ablation_plot.png` → `pathway1_tier_ablation.png`
- `pathway1/phase4/routing_tradeoff_curve.png` → `pathway1_routing_tradeoff.png`
- `pathway1/phase4/calibration_plot.png` → `pathway1_calibration.png`

**Note:** the requested "breathing_temporal_qwen15b_7b.png" headline figure
**does not exist as a committed PNG** — the Qwen 1.5B + 7B temporal PR story
is reported only in the RESEARCH_SUMMARY.md table (§3), not figured. A
follow-up matplotlib script could generate it from
`pathway8_layerwise/data/math500/` + `pathway11_h100/data/math500_7b/`. See
the `figures/README.md` for the explicit status.
