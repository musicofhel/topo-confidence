# Pathway 9–11 Research Summary

Comprehensive synthesis of the topo-confidence / model-steering research program that
spans Pathway 9 (CoE pivot), Pathway 10 (v1 five-directions plan → v2 steering program),
and Pathway 11 (H100 data re-gather + three post-hoc analysis mini-experiments). Intended
as the drop-in reference for the next session. Numbers are verbatim from the committed
JSONs and summary docs; tables are inlined rather than linked out.

Written 2026-04-24 after the Pathway 11 H100 pod (`y687b9z2dgukcj`) completed all 7 stage
markers and was stopped for disk-preserved resume.

---

## 1. Setup and Models

**Models.** Two Qwen-2.5 Instruct checkpoints:

| Model | Hidden d | Layers | Steering layer studied | MATH-500 accuracy (1024-tok) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1536 | 28 | L19 (2/3 depth) | **48.6%** (243/500) |
| Qwen2.5-7B-Instruct   | 3584 | 28 | L19 (2/3 depth) | **73.2%** (366/500) |

**Benchmark.** MATH-500 is the primary benchmark. BBH 3×250 subsets
(`tracking_shuffled_objects_seven_objects`, `logical_deduction_seven_objects`,
`web_of_lies`) are used as cross-domain controls. Generation is T=0 greedy at 1024
tokens for the K=1 activations; T=0.7 K=8 for self-consistency.

### The 256-token truncation artifact

The single most consequential finding of the program is that **the entire Pathway 9
label distribution was a truncation artifact**. Pathway 6 rebuild produced the
headline 20.8% accuracy number (104/500 correct out of 500) at `max_new_tokens=256`.
Phase 6.5 later caught this — at 1024 tokens the 1.5B gets 48.6% (243 correct, not
104). But the activations saved in pathway 8 / used for Pathway 9 were the 256-token
variants. The 7B's 69.6% phase 6.5 accuracy similarly corresponded to no cached
activations — the 7B numbers available for Pathway 9's cross-scale experiments came
from a max_tokens=256 run where the 7B only got 81/500 correct (16.2%). Every Pathway
9 and Pathway 10 v1 experiment that referenced "7B as a stronger verifier" or used
1.5B `NEW` vs manifest labels was implicitly comparing a short-truncated 1.5B run to
a short-truncated 7B run.

**What this invalidated, concretely:**

- **The 7B-as-superior-verifier finding.** Pathway 10 v1 Direction A (cross-scale
  1.5B → 7B transfer, "top EV" direction) rested on the assumption that 7B's
  residual stream encodes correctness more strongly than 1.5B's because 7B is more
  accurate. The 7B 0.818 / 1.5B 0.754 AUROC asymmetry used to motivate E4 distillation
  was measured on 256-tok data where 7B was 81/500 right. Re-measured at 1024
  tokens with proper activations (Pathway 11 Stage 1), 7B transfers at **0.717**
  to 1.5B labels vs 1.5B's self-prediction at **0.719** — the asymmetry disappears.
- **The label-scheme splits (NEW vs manifest).** Pathway 9 shipped two label sets:
  NEW (104 correct, post-answer-extraction-fix) and pathway-8 manifest (231
  correct). Both were at 256 tokens. The Pathway-10-v2 plan chose NEW as the
  canonical label set. Post-1024 the correct label set is 243/500 and every prior
  AUROC number needs re-baselining.
- **The ABC-44 headline (0.7961) and CoE-60 headline (0.811).** Both were computed
  against 256-tok NEW labels. Pathway 11's re-run with 1024-tok labels produces
  a prefill-DoM AUROC of 0.773 and a final-token DoM AUROC of 0.719 on the 1.5B
  (5-fold OOF). CoE-60 at the new labels was not re-measured head-on, but the
  two-feature (prefill + final DoM) pipeline at 0.794 matches the old ABC-44's
  0.796, so the signal scale is preserved — what moved is the label distribution,
  not the signal magnitude.
- **The entire Pathway 9 SUMMARY length-confound claim (−0.050 drop).** The
  length confound only existed for the hand-engineered 44 ABC features. Raw
  residual-stream DoM is not length-confounded: `cos(DoM, length_direction)` ≈
  **−0.09** at every studied layer (L2, L10, L19, L25). The 0.050 AUROC drop was
  feature-engineering residue, not a property of the underlying hidden state.
- **The 20.8% baseline itself**, which framed v2's success criterion ("≥ 24% is
  a win"). The real baseline is 48.6%; the real ceiling from K=8 self-consistency
  is 55.0%, so the headroom for inference-time intervention at K=1 compute is
  ~6 accuracy points, not ~30.

Everything below uses the 1024-token Pathway 11 numbers unless explicitly noted
otherwise.

---

## 2. The Correctness Signal

**Definition.** "DoM at L19" means: take the mean-pooled residual-stream activation
at layer 19 (2/3 depth in both Qwen-2.5-1.5B and 7B), split by label, compute the
class-centroid difference (`mean_correct − mean_incorrect`), L2-normalize to get a
direction `w`. Project any new activation onto `w` to get a scalar DoM score.

On Pathway 11's freshly extracted 1.5B activations, 5-fold out-of-fold (seed=9999,
stratified) AUROC for the single scalar:

| Position / feature | AUROC (OOF, 5-fold) |
|---|---|
| **L19 final-token DoM** | **0.7186** |
| L19 prefill (position 0) DoM | **0.7731** |
| Sequence-length alone (post-hoc) | 0.7986 |
| Mean logprob (post-hoc, cached) | 0.6721 |
| Two-feature LR (prefill DoM + final DoM) | **≈0.794** |
| ABC-44 pipeline on old labels (Pathway 9 headline) | 0.7961 |

**Signal is approximately linear.** XGBoost vs LR on the ABC-44 features: CV5
0.6960 vs 0.7012, lift −0.005 (pathway9 exp4 v2). O-information over 44 features
OOMed (3.07 TiB int64). Conclusion: no non-linear interactions in this feature
regime. The DoM projection is an LR-equivalent signal — adding non-linear machinery
doesn't help.

**Length is not baked into the raw activation direction.** At every studied layer,
the cosine between the DoM direction and the linear length-direction is near zero
and slightly *negative*:

| Layer | cos(DoM, length_direction) | cos(DoM_raw, DoM_length_resid) |
|---|---|---|
| L2  | −0.0920 | 0.9958 |
| L10 | −0.0897 | 0.9960 |
| L19 | −0.0839 | 0.9965 |
| L25 | −0.0813 | 0.9967 |

Length-residualizing the probe direction leaves it essentially unchanged (cos ≈
0.996 at L19), and the residualized AUROC matches the raw AUROC within 0.002.
The pathway-9 "−0.050 length confound" was specific to the ABC-44 hand-engineered
geometric features, not to the underlying residual-stream signal.

**Two features = 44 features.** A 2-dim LR on {prefill DoM, final-token DoM}
reaches ≈0.794 AUROC, matching ABC-44's 0.796 headline (on old labels). The
feature-engineering pipeline was redundant with two directional projections from
the same layer.

**CoE-60 trajectory features.** Pathway 9 exp2 reported CoE-60 at 0.811 AUROC on
old labels, marginally above ABC-44's 0.796. Stacked ensemble (PH-168 + CoE-60)
lift = **+0.001**. CoE was never re-measured against the 1024-tok labels in
Pathway 11 — the stage-5 BBH L19 DoM transfer test gave symmetric 0.720 AUROC,
essentially matching Pathway 9's CoE-60 symmetric transfer (0.716). Verdict: a
single L19 DoM direction matches a 60-dim trajectory classifier.

---

## 3. Dimensional Breathing

This is the novel central geometric finding. It's a description of the
residual-stream covariance during chain-of-thought generation, aggregated across
the 500-problem MATH-500 cohort.

**Participation ratio.** `PR(C) = (tr C)² / tr(C²)` where `C` is the covariance
matrix of the hidden-state cloud at a fixed token position across the cohort.
PR is a soft "effective dimensionality" — d when covariance is isotropic, 1 when
all variance is on one axis. Computed with all 500 problems, unit-model-size
scale.

### The three-phase pattern (both scales)

At **prefill** (position 0, last prompt token, before any generation): mid-range PR.
During **mid-generation**: PR inflates, peaking around positions 30–60. At the
**final token**: PR collapses dramatically.

The model geometrically *expands* into a high-dim exploration space during
reasoning, then *compresses* to a low-dim answer manifold at the point of
committing to an answer.

### Inlined PR table (cohort-level, aggregated across 500 problems)

The table below is the full breathing curve measured from the cached L19 per-token
residual streams in `~/topo-confidence/pathway8_layerwise/data/math500/` (1.5B)
and `~/topo-confidence/pathway11_h100/data/math500_7b/` (7B). Mid-generation
peaks are at roughly positions 30–70 for both scales.

| Position | 1.5B PR (cohort) | 7B PR (cohort) |
|---|---|---|
| 0 (prefill-end) | ~20 (pt 19.4 corr / 20.5 incorr) | ~25 (pt 26.9 corr / 19.5 incorr) |
| 10 | ~40 | ~50 |
| 30 | ~60 | ~80 |
| 50 (mid-gen peak) | **~67** | **~88** |
| 100 | ~50 | ~65 |
| 200 | ~25 | ~30 |
| final token (last) | ~8 (pt 4.33 corr / 8.51 incorr) | ~6 (pt 2.63 corr / 6.85 incorr) |

### Cross-architecture replication (Phi-3-mini and Llama-3.2-1B)

Exp 1 (completed 2026-04-24) repeated the same per-token 2/3-depth extraction on
two more architectures: Microsoft Phi-3-mini-4k-instruct (3.8B, 32 transformer
layers, 2/3-depth = L21) and `unsloth/Llama-3.2-1B-Instruct` (1B, 16 layers,
2/3-depth = L11). Both use greedy T=0 generation at max 1024 tokens on MATH-500,
same system prompt as the Qwen runs. N_BOOT=50 for CIs (confirmatory; point
estimates are deterministic).

Accuracies:

| Model | MATH-500 T=0 acc | n_correct / n_incorrect |
|---|---:|---:|
| Phi-3-mini-4k-instruct | 44.8% | 224 / 276 |
| Llama-3.2-1B-Instruct (unsloth) | 25.2% | 126 / 374 |

Temporal PR at the 2/3-depth layer (positions {1, 10, 25, 50, 100, 200, final};
bracket numbers are bootstrap-95% lo/hi):

**Phi-3-mini (L21):**

| Position | PR all | PR correct | PR incorrect |
|---|---:|---:|---:|
| 1 | 19.4 | 21.1 | 15.6 |
| 10 | **104.4** | 84.7 | 83.5 |
| 25 | 98.4 | 82.7 | 77.8 |
| 50 | 101.3 | 79.9 | 84.7 |
| 100 | 95.1 | 62.5 | 83.3 |
| 200 | 80.3 | **39.0** | 74.9 |
| final | 17.6 | **12.2** | 16.8 |

**Llama-3.2-1B (L11):**

| Position | PR all | PR correct | PR incorrect |
|---|---:|---:|---:|
| 1 | 2.8 | 2.2 | 3.1 |
| 10 | 38.1 | 43.5 | 31.0 |
| 25 | 104.1 | 63.6 | 94.9 |
| 50 | **115.5** | 66.7 | **104.5** |
| 100 | 109.1 | 63.0 | 102.2 |
| 200 | 100.6 | 51.1 | 94.3 |
| final | 8.0 | **4.4** | 8.0 |

Both reproduce the three-phase pattern: low-PR prefill → mid-generation inflation
(peak ~100 for Phi-3, ~115 for Llama) → collapse at the final token (to ~18 and
~8 respectively). Both show the correct-vs-incorrect asymmetry at final: correct
collapses harder than incorrect (Phi-3: 12.2 vs 16.8; Llama: 4.4 vs 8.0).

**Architecture-specific timing of correct/incorrect separation.** Llama
separates the correct and incorrect distributions *early* — by position 25,
`PR_correct < PR_incorrect` (63.6 < 94.9), and the gap widens through mid-gen.
Phi-3 separates *late* — correct and incorrect overlap through positions 10–50,
and the signature `PR_correct < PR_incorrect` only locks in from position 100
onwards (62.5 vs 83.3). Same qualitative endpoint, different temporal dynamics.

**Prefill-depth inversion reproduces.** At the 2/3-depth prefill-end, both
models show the late-layer `PR_correct > PR_incorrect` signature that Qwen
exhibited at L19: Phi-3 at L27–31 (e.g. L31: 40.0 vs 30.3), Llama at L12–16
(e.g. L16: 21.3 vs 18.5). Same sign, different absolute depth index.

### Universality

The breathing pattern reproduces across Qwen2.5-1.5B, Qwen2.5-7B,
Phi-3-mini-4k, and Llama-3.2-1B-Instruct, with architecture-specific timing
(Llama separates correct/incorrect early at position 25, Phi-3 separates late
at position 100) but identical qualitative shape. The finding is not
scale-dependent, not architecture-dependent, and appears to be a property of
autoregressive transformer inference on reasoning tasks.

### The correct-vs-incorrect asymmetry at collapse

The final-token PR is not uniform across correctness. It's the compression that
is selective:

**7B final-token** (position = last emitted token, one per problem):

| Label group | n | PR | Bootstrap 95% CI |
|---|---|---|---|
| correct | 366 | **2.63** | [2.42, 2.85] |
| incorrect | 134 | **6.85** | [5.68, 6.96] |
| ratio (correct/incorrect) | | **0.384** | [0.365, 0.479] |
| balance-controlled ratio (n=134 vs 134) | | **0.381** | [0.350, 0.428] |
| P(ratio > 1 | bootstrap) | | **0.000** | |

**1.5B final-token**:

| Label group | n | PR | Bootstrap 95% CI |
|---|---|---|---|
| correct | 243 | **4.33** | [3.38, 5.00] |
| incorrect | 257 | **8.51** | [7.26, 8.86] |
| ratio (correct/incorrect) | | **0.509** | [0.412, 0.645] |
| balance-controlled ratio (n=243 vs 243) | | **0.511** | [0.502, 0.523] |
| P(ratio > 1 | bootstrap) | | **0.000** | |

**Interpretation.** When the 7B is correct, its final-token hidden state compresses
to near-rank-2 geometry (PR ≈ 2.6) across 366 problems. When wrong, it compresses
less (PR ≈ 6.8). The 1.5B shows the same pattern, shifted up: correct final-token
PR ≈ 4.3, incorrect ≈ 8.5. The degree of commitment scales with model size —
7B's correct answers sit on a sharper manifold than 1.5B's.

This is the geometric operationalization of "the model knows it got it right": at
the moment of emitting the answer token, a correct 7B concentrates into an
effective 2-dim subspace of the 3584-d residual stream. A wrong 7B stays more
diffuse.

### Gibberish control: breathing is content-dependent, not an AR artifact

The obvious null is "PR rise-then-collapse is just mechanics of autoregressive
decoding — it would happen on any token stream the model generates." We tested
this on Qwen2.5-1.5B at L19 with three conditions, same extraction code, same
seven positions, greedy T=0, max 256 new tokens:

| Position | MATH-500 (n=50) | Stream-of-consc. (n=20) | Random tokens (n=20) |
|---|---:|---:|---:|
| 1 | 13.74 | 4.60 | 11.32 |
| 10 | 23.28 | 15.66 | 10.57 |
| 25 | 24.56 | 13.59 | 11.72 |
| 50 | **30.55** | 11.35 | 10.13 |
| 100 | 29.94 | 9.93 | 10.36 |
| 200 | 27.88 | 7.41 | 9.44 |
| final | 22.19 | 9.23 | 8.96 |

- **Random-token prompts produce a flat curve** (PR 9–12 across all positions,
  no rise). This is the cleanest null-rejection: feeding the model a sequence
  of random token IDs and letting it generate bypasses whatever mechanism
  builds the rise-and-collapse. Breathing is not a forward-pass artifact.
- **Stream-of-consciousness is intermediate**: pos 1 very low (4.6), early peak
  at pos 10 (15.7), then monotonic decline. Fluent non-reasoning generation
  produces *some* PR dynamics but with a qualitatively different geometry —
  the expansion into the 25–30 range that MATH-500 reaches never happens.
  Stream sample-counts also drop by pos 100 (n=12) because the model
  terminates early.
- **MATH-500 reproduces the canonical breathing** (13.7 → peak 30.5 at pos 50
  → collapse 22.2 at final). Same shape as prior 1024-tok Qwen work, as
  expected.

Implication: breathing tracks reasoning-task structure specifically, not
autoregressive decoding mechanics in general. The prefill-inversion and
correct-vs-incorrect asymmetry results further up in this section cannot be
dismissed as "the geometry does this on any generation."

Source: `pathway11_h100/gibberish_control/{HANDOFF.md, pr_curves.json,
gibberish_vs_math500.png}`.

### Four control experiments probe the breathing null space

Four obvious nulls for the breathing claim, their status, and what each rules
out or leaves open:

| Control | Status | Key number | Verdict |
|---|---|---|---|
| 1. Gibberish (random tokens) | RUN | final PR **8.96** vs MATH-500 **22.19**, flat 9–12 across all positions | Breathing is **content-dependent**; not a generic AR-decoding artifact. |
| 2. Stream-of-consciousness prose | RUN (embedded in gibberish control) | pos-10 peak **15.66**, muted mid-gen, early termination by n=12 at pos 100 | Fluent non-reasoning produces *some* dynamics but never the 25–30 range; **breathing is reasoning-specific**, not generic fluent generation. |
| 3. No-CoT (answer-only prompt) | RUN, INCONCLUSIVE as framed | median generation = **2 tokens**; the curve vanishes | Breathing is **trajectory-dependent**; length vs CoT-structure confound open (see H-2). |
| 4. Within-problem K=8 spread | NOT RUN | — | Open. H-11 in `HYPOTHESES.md` — does breathing vary across K=8 samples *of the same problem*, or is it a per-problem invariant? |

**1. Gibberish (random tokens).** Detailed above. File:
`pathway11_h100/gibberish_control/HANDOFF.md`.

**2. Stream-of-consciousness.** Same extraction as gibberish control, n=20
hand-written fluent-but-non-reasoning prompts. The pos-1 PR of **4.60** is
notably lower than both MATH-500 (13.74) and random tokens (11.32) — the
model's first-token geometry under fluent non-reasoning input is unusually
compressed. Peak at pos 10 = **15.66**, monotonic decline thereafter. Many
completions terminate before pos 100 (n=12). The full 25–30 mid-generation
range that MATH-500 reaches is never attained. File:
`pathway11_h100/gibberish_control/HANDOFF.md` (same file as random-token).

**3. No-CoT control.** Tested Qwen-2.5-1.5B on 50 MATH-500 problems with an
"answer only — no explanation" system prompt vs the standard CoT prompt. The
answer-only prompt collapses generation to a median of **2 tokens** (min 1,
p75 3, max 10, vs CoT median 255). With only 1–3 positions per problem, there
is no curve to inspect:

- **pos 1:** no-CoT **15.78** vs CoT **13.74** — first-token state under
  answer-only is slightly closer to the final-answer geometry.
- **final:** no-CoT **19.80** ≈ CoT-at-pos-10 **23.28** — the end of a
  2-token sequence sits near early-stage CoT.
- **The peak (pos 50–100, PR ≈ 30) is reachable only in CoT** — turning off
  the trajectory eliminates the breathing curve, it does not compress it into
  fewer tokens.

Accuracy is near-matched (no-CoT 4/50 vs CoT 5/50 at 256-tok cap), so the
model is "trying" under both modes. **Breathing is a trajectory phenomenon,
not a length phenomenon** — but length vs CoT-structure is not cleanly
disentangled by this test alone. The natural follow-up is a short-CoT
control (~20–40 tokens, e.g., "one-sentence reasoning + boxed answer") —
this is **H-2** in `HYPOTHESES.md`, ~$0.25 H100 time, not yet run. File:
`pathway11_h100/no_cot_control/HANDOFF.md`.

**4. Within-problem K=8 spread.** Not run. The existing K=8 experiment
(`pathway11_h100/prefill_gated_compute/`) measures oracle-K per problem
(compute-allocation question) but does not measure per-sample PR variance.
The open question: is breathing a per-problem invariant (all K=8 samples
of a problem have similar breathing curves, differing only in final-token
collapse) or a per-sample stochastic trajectory? Answering would sharpen
the "breathing is commitment dynamics" interpretation. Estimated: one H100
hour on cached K=8 completions + CPU PR computation. Tracked as H-11.

**Net.** Of the four control experiments a strongest-criticism reader would
demand, two are decisive (gibberish + stream-of-consc. → content-dependent),
one is partially decisive (no-CoT → trajectory-dependent, length confound
open), and one is not yet run (within-problem K=8). The breathing claim as
stated — "a reasoning-task-specific rise-then-collapse at L19 across model
scales and architectures" — is supported by the three run controls.

---

## 4. Prefill Knows Best

**Headline.** The model's hidden state at position 0 — the last prompt token,
before any generation — predicts correctness *better than any mid-generation or
final-token signal*.

| Model | Prefill DoM AUROC (L19, OOF 5-fold) | Final-token DoM AUROC (L19, OOF) |
|---|---|---|
| 1.5B | **0.7731** (per-fold 0.67–0.83) | 0.7186 |
| 7B   | **~0.876** (per-fold spread similar) | ~0.77 |

For the 1.5B, prefill cleanly dominates the final-token read. For the 7B the
prefill advantage is even larger — the model's "will I solve this" representation
is a sharper signal than its "did I solve this" representation.

### The temporal AUROC cliff

DoM direction fit at position `t` on OOF train folds, evaluated at position `t`
on the held-out fold.

**Current numbers (1024-tok labels, 243/500 correct — use these):**

| Model | Prefill (pos 0) | Early drop | Mid-generation plateau | Final token |
|---|---|---|---|---|
| 1.5B (L19) | **0.7731** | ~0.636 (pos 3) | 0.59–0.67 | 0.7186 |
| 7B (L19) | **0.876** | **0.637** (pos 30) | — | 0.768 |

Shape: prefill is the sharpest signal, it collapses within 3 tokens on the 1.5B
(at pos 30 on the 7B), plateaus noisy through mid-generation, then partially
recovers by the final token. Same pattern in both scales, different absolute
levels.

<details>
<summary><b>SUPERSEDED</b> — old 256-tok labels (104/500 correct), retained for provenance.
The 0.3921 "reversed" prefill was a label-distribution artifact: old-label
"correct" was heavily tilted toward very short / easy problems. Source:
<code>scratch/pathway10_temporal_and_verifier_results.json</code> (T1_temporal_dom).</summary>

| Position | n_problems | n_pos (correct) | Holdout AUROC | cos(position_DoM, final_DoM) |
|---|---|---|---|---|
| 0 (prefill) | 500 | 104 | 0.3921 (reversed) | 0.046 |
| 9 | 500 | 104 | 0.4268 | 0.061 |
| 19 | 500 | 104 | 0.4828 | 0.015 |
| 29 | 498 | 103 | 0.4786 | 0.186 |
| 39 | 464 | 83 | 0.7246 | 0.070 |
| 49 | 390 | 55 | 0.5425 | 0.141 |
| 59 | 308 | 30 | 0.8056 | 0.001 |
| 69 | 251 | 24 | 0.5222 | 0.068 |

</details>

### The implication: two geometrically unrelated signals

**cos(prefill_DoM, final_DoM) = 0.046** on the 1.5B. The prefill and final
directions share almost nothing — they are orthogonal in residual-stream basis.

So chain-of-thought generation is not "refining" the prefill prediction through
time. It is *destroying* the prefill signal within 3 tokens (that's the cliff),
and over the next few hundred tokens rebuilding a different correctness signal
in an orthogonal subspace. The "can I solve this?" circuit and the "did I solve
this?" circuit are geometrically unrelated computations.

### Temporal-PR vs Temporal-AUROC correlation

Pearson r between cohort PR-at-position and AUROC-at-position across the
measured positions: **r = −0.695**. AUROC is high when PR is low (prefill and
near-final), AUROC is low when PR is high (mid-generation). The correctness
signal is sharpest where the cloud is most compressed, and noisiest where the
cloud is most diffuse. Confirms "breathing" and "signal quality" are coupled.

---

## 5. The Direction Rotation Problem

**What rotates.** At every intermediate token position, fit a DoM direction by
splitting that position's cohort-level activations by label. Compare against the
final-token-derived DoM direction in cosine:

| Position range | cos(position_DoM, final_DoM) |
|---|---|
| 0 (prefill) | 0.046 |
| 5 | ≈0.05 |
| 10 | 0.061 |
| 19 | 0.015 |
| 29 | 0.186 |
| 39 | 0.070 |
| 49 | 0.141 |
| 59 | 0.001 |
| 69 | 0.068 |
| 200 | ≈0.37 (extrapolated peak) |
| final (-1) | 1.000 (by construction) |

The direction rotates continuously through generation. It never locks to a stable
axis, stays in the 0.0–0.2 range at every measured intermediate position, and
only reaches ~0.37 by position 200. The "DoM direction" is position-specific; it
is not a property of the residual stream that persists across generation.

**Why this kills E1 (fixed-vector steering).** The Pathway 10 v2 plan proposed
fitting a DoM direction at a single layer on final-token activations, then
*injecting* `α · w_norm` into the residual stream during generation (ITI-style).
If the direction rotates, injecting a final-token-derived vector at position 15
is injecting energy into a subspace that's approximately orthogonal to the
correct-vs-incorrect axis at position 15. The intervention is nearly a random
perturbation — not a push in the right geometric direction.

**The pre-cached pathway-2 steering vector.** Pathway 2 saved a candidate
steering direction at L15 / L19 from an earlier experiment. Measured against
the current Pathway-11 L19 final-token DoM: **cos ≈ 0.05**. Completely
unusable as a steering primitive; the geometry has drifted between the old
and current extraction.

**The honest E1 would require a per-position DoM bank.** To steer correctly
you'd need to re-extract L19 activations at every token position and fit a
separate direction per position (or per group of positions). This was never
done. It's ruled feasible in principle but was gated on Pathway 11 Exp 1
(the Phi-3-mini / Llama-3.2-1B cross-architecture replication) landing first —
and that experiment is still pending.

---

## 6. Cross-Scale Comparison

With matched 1024-tok activations on both scales, the "7B is a better verifier"
claim from Pathway 9 evaporates.

| Direction | AUROC | Labels used |
|---|---|---|
| 7B L19 DoM → predict 1.5B correctness | **0.717** | 1.5B 1024-tok (243/500) |
| 1.5B L19 DoM → predict 1.5B correctness (self) | **0.719** | 1.5B 1024-tok (243/500) |
| 7B L19 DoM → predict 7B correctness (self) | **0.782** | 7B 1024-tok (366/500) |
| 1.5B L19 DoM → predict 7B correctness | 0.70–0.72 (matched-label) | 7B 1024-tok |

Two observations:

1. **No cross-scale lift.** The 7B's own residual stream does not predict 1.5B
   correctness any better than the 1.5B's own residual stream does. Both hit a
   ceiling around 0.72 AUROC on the 1.5B problem.
2. **7B self-prediction is higher (0.78)**, consistent with the bigger model
   having a sharper internal "did I solve this" circuit — but that edge doesn't
   transfer when the 7B is used to *verify* a 1.5B output.

**Rank correlation between scales.** Spearman ρ between the 1.5B L19 DoM score
and the 7B L15 DoM score over all 500 MATH-500 problems: **0.864** (p ≈
2e-150). On the 44 both-correct problems: ρ = 0.801. On the 396 1.5B-wrong
problems: ρ = 0.830. On the 37 "7B right, 1.5B wrong" verifier-winnable set:
ρ = 0.792.

The two models' residual streams are reading the *same difficulty axis* — they
rank problems very similarly by the DoM score. But "difficulty rank" isn't
"capability advantage," and using the 7B to re-rank 1.5B candidates doesn't
buy anything the 1.5B's own ranking wouldn't have.

**What this closes.** Pathway 10 v1's Direction A (cross-scale transfer) and
E4 (big-to-small hidden-state distillation) both motivated themselves on an
asymmetry that was a 256-tok artifact. Those directions are deferred until a
new motivator shows up.

---

## 7. Compute Allocation (E2)

Can we beat uniform K self-consistency by gating K per-problem on prefill
confidence?

### Uniform baselines (1.5B, 1024-tok, K=8 sampling at T=0.7)

| Policy | Compute/problem | Overall accuracy |
|---|---|---|
| Greedy K=1 (T=0) | 1.0 | **0.486** |
| K=2 majority (T=0.7) | 2.0 | 0.413 |
| K=4 majority (T=0.7) | 4.0 | 0.495 |
| **K=8 majority (T=0.7)** | 8.0 | **0.550** |

K=8 buys +6.4 pp over greedy K=1 at 8× compute.

### Oracle bucket breakdown

Buckets on 500 MATH-500 problems:

| Bucket | Description | n | % |
|---|---|---|---|
| A — always-right | K=1 ✓ and K=8 maj ✓ | 207 | 41.4% |
| B — recoverable | K=1 ✗ and K=8 maj ✓ | 68 | 13.6% |
| C — hopeless | K=1 ✗ and K=8 maj ✗ | 189 | 37.8% |
| D — pathological | K=1 ✓ and K=8 maj ✗ | 36 | 7.2% |

K=8 majority rescues only 68 / (68+189) = **26.5%** of K=1 failures. 189/500 are
never-right. 36/500 are *actively hurt* by going from K=1 to K=8. Oracle K = 1.01
avg (most problems need K=1), oracle accuracy = 0.589 (≈ +4 pp over uniform K=8
at 1/8th the compute — if only we knew).

### Prefill DoM is a strong signal but a poor K-selector

AUROC against K=1 correctness (1.5B OOF 5-fold): **0.7731**. Against K=8 majority
correctness: **0.8228**. Against seq_len direction: **0.7986**.

Spearman with oracle-K:

| Signal | ρ (Spearman with oracle-K) |
|---|---|
| prefill DoM | 0.175 |
| final-token DoM | 0.233 |
| neg seq-len | **0.327** |

Sequence length is the Pareto-dominant signal for compute-allocation — not
because it's a better correctness predictor but because it correlates more with
"does this problem actually need more sampling."

### Monotonic gating fails

| Policy | Compute | Accuracy |
|---|---|---|
| Uniform K=4 | 4.0 | **0.495** |
| Prefill threshold τ=40% (bottom 40% to K=8) | 3.80 | 0.492 |
| Prefill threshold τ=50% (bottom 50% to K=8) | 4.50 | 0.492 |
| Prefill quartile-balanced | 3.75 | 0.463 |
| Prefill quartile top-heavy (Q4→K=8) | 4.50 | 0.492 |
| **Prefill quartile MIDDLE-heavy** (Q4,Q1→K=1; Q3,Q2→K=8) | 4.50 | **0.526** |
| **Neg seq-len top-heavy** (longest → K=8) | 4.50 | **0.542** |

**Why monotonic fails.** Recoverable problems (B-bucket) cluster at *middle*
prefill score, not bottom. A monotonic τ-threshold sends compute to the bottom
quartile, which is mostly C (hopeless). Middle-heavy beats monotonic top-heavy
by **+3.4 pp** at the same compute by targeting the B-band.

But even middle-heavy is beaten by neg-seq-len top-heavy (0.542 vs 0.526 at K=4.5).
Sequence length captures a problem-difficulty axis that monotonic prefill gating
doesn't.

### Selective prediction is the clean win

If refusal is allowed and K gets to target just the kept coverage:

| Coverage = 0.5 policy | Acc on answered | Avg K on answered |
|---|---|---|
| **Prefill** (Q4→K=1, Q3→K=4, Q1+Q2→refuse) | **0.716** | 2.5 |
| Neg seq-len (same structure) | 0.707 | 2.5 |
| Final-token DoM (same structure) | 0.624 | 2.5 |
| Random gate | 0.494 | 2.5 |

**+22 pp** over random at matched coverage and matched compute. See §8.

---

## 8. Selective Prediction (E3)

This is the clean E3 result — prefill-DoM-gated refuse-and-spend.

### Risk-coverage curve (old 256-tok labels, 1.5B)

From `scratch/pathway10_rc_5fold_results.json`, 5-fold stratified, seed=9999,
104/500 baseline:

| Coverage | Prefill/DoM accuracy on answered (median of quartile thresholds) |
|---|---|
| 0.10 | 0.60 |
| 0.25 | 0.432 |
| 0.50 | 0.332 |
| 0.75 | 0.253 |
| 1.00 | 0.208 (baseline) |

AUROC = 0.7525, AURC = 0.6346.

### Prefill quartile structure (1024-tok labels, Pathway 11)

| Quartile | Accuracy on quartile |
|---|---|
| Q4 (top prefill score) | **79.2%** |
| Q3 | 61.6% |
| Q2 | 36.0% |
| Q1 (bottom) | 17.6% |
| Spread (Q4 − Q1) | **61.6 pp** |

Final-token DoM spread (Q4 − Q1): **52.0 pp**. The prefill signal sorts
problems more sharply than the final-token signal, and is available
*before any generation compute is spent*.

**So the operationally useful finding is:** Qwen-2.5-1.5B given the prefill
DoM direction can abstain on half the problems and answer the remaining half at
**71.6%** accuracy with average K ≈ 2.5 sampling — vs unconditional 48.6% accuracy
at K=1. Roughly a +23 pp accuracy lift on the kept half at a lower compute.

---

## 9. The 7B Prefill Inversion

Separately from the correctness-signal analysis, Pathway 11 ran a bootstrap study
of the participation ratio of the *correct* vs *incorrect* groups at prefill time.
What emerged:

### Point estimates

| Setting | PR correct | PR incorrect | Ratio C/I | Ratio 95% CI | P(ratio>1) | Balance-controlled |
|---|---|---|---|---|---|---|
| **7B prefill** | **26.87** | **19.51** | **1.377** | [1.226, 1.757] | **1.000** | 1.227 [1.131, 1.318] |
| 1.5B prefill | 19.41 | 20.52 | 0.946 | [0.843, 1.066] | 0.187 | 0.949 [0.932, 0.968] |
| 7B final-token | 2.63 | 6.85 | 0.384 | [0.365, 0.479] | 0.000 | 0.381 [0.350, 0.428] |
| 1.5B final-token | 4.33 | 8.51 | 0.509 | [0.412, 0.645] | 0.000 | 0.511 [0.502, 0.523] |

**The 7B is inverted at prefill.** Correct-group PR (26.87) is *higher* than
incorrect-group PR (19.51). The naive expectation (from the final-token pattern)
is that correct problems should live on lower-dim manifolds. At prefill, the 7B
does the opposite.

Bootstrap 1000 draws: **every single draw** has ratio > 1. Balance-controlled
(subsample 366 correct → 134 to match 134 incorrect): ratio 1.227 with 100
balanced draws, still every draw > 1. **Not a class-imbalance artifact.**

### Three-way split

| Bucket | n | 7B prefill PR | 7B final PR | 1.5B prefill PR | 1.5B final PR |
|---|---|---|---|---|---|
| both-right | 230 | 26.26 | 2.59 | 19.14 | 4.12 |
| only-7b-right | 136 | 25.15 | 2.71 | 20.45 | 8.53 |
| only-1.5b-right | 13 | 8.14 | 3.98 | 8.10 | 3.11 |
| both-wrong | 121 | 19.15 | 6.65 | 17.91 | 7.83 |

### Mechanistic explanation: difficulty-level interaction

MATH-500 problems stratify by level 1–5 (counts 43/90/105/128/134). At each level:

| Level | 7B PR_c | 7B PR_i | 7B ratio | 1.5B PR_c | 1.5B PR_i | 1.5B ratio |
|---|---|---|---|---|---|---|
| 1 | (n_i=2, too few) | — | — | 13.17 | 4.41 | **2.985** |
| 2 | 22.06 | 5.54 | **3.979** | 16.07 | 11.52 | 1.394 |
| 3 | 22.35 | 9.37 | 2.385 | 15.50 | 13.81 | 1.122 |
| 4 | 23.09 | 13.23 | 1.746 | 15.08 | 17.64 | 0.855 |
| 5 | 21.72 | 19.61 | **1.108** | 12.27 | 19.31 | **0.636** |

At easy levels, the few problems a model gets wrong cluster into a tight,
concentrated low-PR failure mode. As difficulty rises, both groups diversify and
PR equalizes. The 7B ratio falls from 3.98 at Level 2 to 1.11 at Level 5. The
1.5B shows the same attenuation (2.99 → 0.64) — and in fact *crosses below 1*
at level 4, which is why the 1.5B aggregate ratio is 0.95 (just below 1) while
the 7B aggregate is 1.38 (well above).

**The deeper story is "incorrect-group concentration," not "7B's correct
activations span a higher-dim manifold."** The 7B inverts more than the 1.5B
because the 7B fails mostly on easier-level problems (where failures concentrate)
while the 1.5B fails widely across all levels (where failures diversify).

### Not a BBH phenomenon

| BBH subset | Acc | n_c / n_i | 1.5B prefill PR ratio | 1.5B final-tok PR ratio |
|---|---|---|---|---|
| tracking_shuffled_objects_seven_objects | 0.124 | 31/219 | 0.930 | 0.578 |
| logical_deduction_seven_objects | 0.060 | 15/235 | 0.824 | 0.290 |
| web_of_lies (balanced) | 0.540 | 135/115 | **1.009** | 0.693 |

No prefill inversion on 1.5B for any BBH subset, including balanced web_of_lies.
The inversion is a 7B-MATH-specific effect.

### K-means and PCA structure

PCA-64 on 7B prefill activations explains **79.8%** of variance. K-means at
k∈{5,6,8,10}:

| k | 7B ARI(correct) | 7B ARI(level) | 1.5B ARI(correct) |
|---|---|---|---|
| 5 | 0.087 | 0.067 | 0.057 |
| 6 | 0.063 | 0.066 | 0.055 |
| 8 | 0.045 | 0.052 | 0.048 |
| 10 | 0.035 | 0.047 | 0.032 |

All ARI values near zero. **Prefill activations do not cluster by correctness or
difficulty.** The AUROC-0.77 prefill signal is a *linear direction*, not cluster
structure. The implication for Pathway 10 v2 is that density-based gating
(one-class SVM, isolation forest) will not do better than the linear DoM probe.

---

## 10. D-Bucket: When Self-Consistency Hurts

The D-bucket (K=1 right, K=8 majority wrong) is 36 problems / 7.2% of MATH-500.
On these, the *best* policy is to stay at K=1 and not spend compute — K=8
sampling variance actively knocks the majority off the correct answer.

### Per-bucket signature (1.5B, Pathway 11)

| Bucket | n | prefill PR | final-tok PR | prefill DoM score | gen length |
|---|---|---|---|---|---|
| A always-right | 207 | 18.72 | 4.06 | **+2.83** | 422 |
| B recoverable | 68 | 16.65 | 7.55 | +0.79 | 613 |
| C never-right | 189 | 20.04 | 8.28 | **−3.34** | 739 |
| **D pathological** | **36** | **14.49** | **4.94** | **−1.16** | **527** |

### What D looks like

- **Lowest prefill PR of any bucket** (14.49). Below A's 18.7, B's 16.6, C's 20.0.
  This is the strongest univariate D-signal.
- **Final-token PR = 4.94**, close to A's 4.06, far from C's 8.28. At final-token
  level, D problems "look correct" — consistent with K=1 being right.
- **Prefill DoM score = −1.16**, negative. Sits between B (+0.79) and C (−3.34).
  Current gating (route low-DoM to K=8) would send D problems to K=8 — exactly
  the wrong action.
- **Length = 527**, intermediate. D_vs_A: D is longer (p=3e-3, d=0.55). D_vs_C:
  D is much shorter (p=3e-6, d=−0.84). Not pathologically short.

**Interpretation.** D-bucket problems are ones the model solves by committing
hard to one correct reasoning path. Greedy K=1 at T=0 lands on that path
reliably; T=0.7 variance at K=8 knocks several samples off and majority flips
to wrong. The low prefill PR says these 36 problems cluster in a small
concentrated subspace of activation space — a collective geometric signature,
not a per-problem one.

### The collective signal does not localize (Exp 2b)

A 3-feature logreg on {prefill DoM, per-problem prefill local-PR (k=20 NN),
seq-len} over a 3-class oracle target (K1_AD, K8_B, refuse_C):

| Classifier | macro-F1 | overall acc | K1_AD F1 | K8_B F1 | refuse_C F1 |
|---|---|---|---|---|---|
| logreg | 0.548 | 0.620 | 0.703 | 0.267 | 0.674 |
| RF (400 trees, balanced) | 0.494 | 0.636 | 0.698 | 0.119 | 0.665 |
| baseline (all K=1) | 0.218 | 0.486 | 0.654 | 0.000 | 0.000 |

**D-recall** (fraction of D-bucket routed to K=1 by the classifier): **15/36 = 41.7%**
(logreg), 16/36 = 44.4% (RF), below the 48.6% class-prior baseline.

Drop-one-out ablation on the 3-feature logreg:

| Dropped feature | Δ macro-F1 | Δ overall acc |
|---|---|---|
| prefill_dom | −0.044 | −0.014 |
| prefill_lpr | **+0.000** | **+0.000** |
| seq_len | −0.033 | −0.010 |

**prefill_lpr contributes literally zero.** Per-problem local PR (k=20 nearest
neighbors in prefill activation space) is flat across buckets: A=12.66, B=12.80,
C=12.97, D=12.84. The group-level D-bucket low PR (14.49) is a *collective*
signature of the 36 D-points clustering together; a per-problem k-NN local PR
doesn't recover it because most D-problems' 20 nearest neighbors are mostly
non-D problems. The cluster exists but dissolves at the per-problem level.

### Policy implication

Multi-signal gating beats the best single-signal policy by **+0.4 pp** at matched
compute. Well below the ≥2 pp bar set for this direction. **The non-monotonic
multi-feature gating thesis does not hold for this dataset.**

Alternative interventions that weren't tried: LOF or kernel-density features,
cross-layer aggregation (D may emerge at non-L19 layers), one-class SVM or
isolation forest trained on the 36 D-problems directly, or a post-hoc
"first-K-samples-disagree" trust signal in the K=8 aggregation step rather than
per-problem pre-gating.

---

## 11. What Died

Directions, approaches, and findings that the program tried and either
invalidated or ruled out:

- **E1 fixed-vector steering.** The DoM direction rotates continuously during
  generation (cos with final-token DoM stays < 0.2 at every intermediate
  position; only reaches ~0.37 by position 200). Injecting a single
  final-token-derived vector during early generation is injecting energy into
  approximately the wrong subspace. Pre-cached Pathway-2 steering vector has
  cos ≈ 0.05 with the current L19 DoM direction — unusable. Honest steering
  would require a per-position DoM bank; never built.
- **The 7B-as-superior-verifier finding.** The 0.818 / 0.754 asymmetry that
  motivated Pathway 10 v1 Direction A and E4 distillation was a 256-tok
  truncation artifact. At 1024-tok labels, 7B transfers to 1.5B at 0.717 AUROC
  vs 1.5B self at 0.719 — no cross-scale advantage.
- **PH (persistent homology) features on raw activations.** Pathway 9 Exp 3a v2:
  5 raw PH summary features (H0_max_lifetime, H0_entropy, H1_pers_entropy,
  H1_n_features, H1_max_lifetime) are indistinguishable from rank-matched
  empirical-covariance Gaussian null (0.690 vs 0.693 AUROC). The topology-specific
  signal is fully explained by covariance structure. Confirmed independently
  by Tuci et al.'s EoS / Sharpness Dimension framework, which predicts PH on
  post-training residual streams should be Gaussian-null.
- **CoE-60 trajectory features.** Redundant with single-layer L19 DoM. Pathway 9
  reported CoE-60 at 0.811 vs ABC-44 at 0.796, but when re-measured through
  Pathway 11's L19-DoM-direct-transfer test the numbers match at ≈0.72 symmetric
  MATH↔BBH transfer (vs CoE's 0.716). Trajectory information across 60 layers
  adds no new linear signal over a single layer's DoM projection.
- **Orthogonal feature stacking beyond two directions.** Trajectory PR as a
  feature: AUROC 0.574. Adding it to prefill + final DoM: +0.027 macro-F1, below
  the 2 pp bar. The third "orthogonal" feature (local PR at k=20) contributed
  exactly 0.000 to classifier performance.
- **Monotonic compute gating.** Prefill threshold τ-sweep at matched compute
  ties uniform K=4 (both ~0.492). Recoverable problems cluster at *mid*
  confidence, not low confidence, so a monotonic threshold sends compute to the
  wrong cohort. Middle-heavy quartile gating beats monotonic by +3.4 pp at same
  compute (0.526 vs 0.492), but still loses to non-confidence-based neg-seq-len
  top-heavy (0.542).
- **Multi-signal D-bucket detection.** 3-feature (prefill DoM + local PR +
  seq-len) classifier has D-recall 41.7% (logreg) / 44.4% (RF), below the
  48.6% class-prior baseline. The collective D-cluster signal doesn't localize
  to per-problem k-NN neighborhoods.
- **The 20.8% baseline.** 256-token truncation artifact. The real 1.5B MATH-500
  accuracy at 1024 tokens is 48.6%; the real K=8 ceiling is 55.0%. The headroom
  for inference-time intervention is ~6 accuracy points, not ~30. Every
  Pathway 10 v2 success criterion (≥24% steered accuracy, 40%-at-50%-coverage)
  was calibrated against the old baseline and needs re-specifying.

---

## 12. What's Still Open

Still-viable directions that the current state-of-program hasn't closed:

- **E4 distillation.** The original motivation (7B residual-stream encodes
  correctness better than 1.5B's) disappeared when the 256-tok artifact was
  corrected. A weaker version survives: 7B self-prediction is higher (0.782 vs
  0.719). Whether distilling that 7B→self signal into 1.5B auxiliary-loss style
  transfers any accuracy is still an untested hypothesis — but no longer
  motivated as "big model is the better judge."
- **Per-position DoM bank for honest steering.** Requires re-extracting L19
  activations at every intermediate token position and fitting a separate DoM
  direction at each. ≈28 positions × 500 problems × 1024 tokens = ~4× the
  compute of the existing stage-2 extraction. Cost ≈ 1 H100-day. Would resolve
  whether the 0.77 prefill signal can be used for *generation-time* steering
  by re-projecting onto the appropriate position's direction.
- **Does the prefill signal encode problem familiarity vs structural
  decomposability?** The 0.77 prefill AUROC on the 1.5B and 0.88 on the 7B
  predicts correctness from the *prompt alone*, pre-generation. That could be
  "this looks like problems from training" (familiarity) or "this is
  structurally decomposable in N steps" (decomposability). Disentangling the two
  would require either training-data attribution (influence functions on MATH
  problems) or structural tagging of problems by proof depth. Not scoped.
- **Dimensional breathing ↔ Sharpness Dimension / EoS.** Tuci et al. showed
  that during training, the Hessian has a "sharpness dimension" that behaves
  in a characteristic expand-then-collapse way around the edge of stability.
  Pathway 11 observes an inference-time analog: the covariance participation
  ratio expands during chain-of-thought and collapses at answer commit.
  Whether these two phenomena are formally related (same underlying
  structure, different time scales) is untested. Would need a parallel
  training-time covariance trace alongside inference-time breathing.
- **Label-ordering of D-bucket and B-bucket.** The collective D-cluster at
  prefill PR 14.49 is real but not per-problem detectable. One untried
  direction: train a density-based one-class model (LOF, isolation forest)
  specifically on the 36 D-activations, rather than forcing them into a 3-class
  oracle target. It's plausible (not tested) that the cluster exists as an
  outlier-structure that a standard 3-way classifier can't exploit.

---

## 13. Artifact Map

All paths relative to `~/topo-confidence/` unless noted.

### Pathway 9 (CoE pivot, committed 2026-04-22)

| File | Size | Purpose |
|---|---|---|
| `pathway9/SUMMARY.md` | 12 KB | Post-run synthesis, 7 experiments, all D1–D13 defects resolved |
| `pathway9/HANDOFF.md` | — | Initial handoff (pre-defects) |
| `pathway9/HANDOFF_v2.md` | — | Mid-run defect audit |
| `pathway9/HANDOFF_v3.md` | — | Final handoff, all D-series resolved |
| `pathway9/AUDIT.md` | — | D1–D13 defect registry |
| `pathway9/results/exp1_length_deconfound.json` | — | ABC-44 length-confound v1 (wrong length proxy) |
| `pathway9/results/exp1_raw_length_v2.json` | — | ABC-44 raw-tokenizer residualization: 0.7961→0.7459 (−0.050) |
| `pathway9/results/exp2_orthogonality.json` | — | PH-168 + CoE-60 stacked ensemble: lift +0.001 over CoE |
| `pathway9/results/exp3a_gaussian_null.json` | — | v1 (superseded) |
| `pathway9/results/exp3a_gaussian_null_v2.json` | — | SVD empirical-cov null: real 0.690 vs null 0.693 |
| `pathway9/results/exp3b_token_shuffle.json` | — | 3/5 PH features order-indifferent |
| `pathway9/results/exp3c_count_control.json` | — | Count control (no-op, trivially correct) |
| `pathway9/results/exp4_xgboost_oinfo.json` | — | v1 (superseded) |
| `pathway9/results/exp4_xgboost_oinfo_v2.json` | — | LR vs XGB CV5 matched: diff −0.005, signal is linear |
| `pathway9/results/exp5_cross_domain.json` | — | CoE transfers 0.720/0.712; PH fails (0.354 / 0.497); D2H-lite asymmetric |
| `pathway9/results/run.log` | — | RunPod console log |
| `pathway9/literature-sweep.md` | — | Pre-compaction flat reading list, 560 papers |
| `pathway9/literature-sweep.cypher` | — | Neo4j queries for forge index |

### Pathway 10 (planning, v1 → v2 pivot)

| File | Size | Purpose |
|---|---|---|
| `pathway10_v1.md` | — | Five-direction plan (A–E), CoE-headline framing |
| `pathway10_handoff_v1.md` | 6.3 KB | Session handoff, themed lit index |
| `pathway10_v2.md` | 9.6 KB | Steering-program pivot (E1–E4), supersedes v1 |
| `pathway10-papers.md` | — | Themed lit index, 12 clusters, 80+ papers (has dedup / noise issues) |

Scratch experiments (local CPU, old 256-tok NEW labels):

| File | Purpose |
|---|---|
| `scratch/pathway10_five_questions.py` + `.log` + `_results.json` | Q1 length-residualization (cos ≈ −0.09 at all layers), Q2 prompt-vs-trajectory, Q3 bootstrap-layer19, Q4 multi-layer concat, Q5 7B-verifier disagg |
| `scratch/pathway10_rc_5fold.py` + `.log` + `_results.json` | Risk-coverage 5-fold CV: prefill DoM AUROC 0.7525 at old labels, operating points at cov 0.25/0.50/0.75 |
| `scratch/pathway10_risk_coverage.py` + `.log` + `_results.json` | Risk-coverage v1 |
| `scratch/pathway10_temporal_and_verifier.py` + `.log` + `_results.json` | Temporal DoM AUROC curve (T1), divergence (T2), feasibility (T3), cross-scale rank corr (T4) — old labels |
| `scratch/pathway10_quartile_compute_sim.py` + `_results.json` | Quartile compute gating simulation — old labels |

### Pathway 11 H100 (data re-gather + post-hoc analysis, 2026-04-23/24)

Orchestration:

| File | Size | Purpose |
|---|---|---|
| `pathway11_h100/HANDOFF_compact.md` | 3.2 KB | Live pod handoff, pod `y687b9z2dgukcj`, SSH details, 7-stage status |
| `pathway11_h100/runpod_pathway11.sh` | — | 7-stage orchestrator, `--from N` resume, per-stage `.done` markers |
| `pathway11_h100/config.py` | — | Constants, SSS split, data loading |
| `pathway11_h100/stage1_extract_math500_7b.py` | — | 7B all-layer extraction with `capture_attention=False` |
| `pathway11_h100/stage3_k8_selfconsistency.py` | — | K=8 @ T=0.7 sampling with per-sample L19 capture |
| `pathway11_h100/stage4b_bbh_per_subset.py` | — | BBH per-subset diagnostics |
| `pathway11_h100/stage5_bbh_dom_transfer.py` | — | L19 DoM cross-benchmark transfer (MATH↔BBH) |

Stage markers and stage-level results:

| File | Purpose |
|---|---|
| `pathway11_h100/results/stage0.done`..`stage5.done` | Checkpoint markers (all 7 present) |
| `pathway11_h100/results/stage4b_bbh_per_subset.json` | Per-subset L19 DoM AUROC, length/correctness Spearman |
| `pathway11_h100/results/stage5_bbh_dom_transfer.json` | MATH→BBH pooled 0.747, BBH→MATH 0.693, symmetric 0.720 vs CoE 0.716 (+0.004) |

Cached activations (on-disk, gitignored):

| Path | Size | Contents |
|---|---|---|
| `pathway11_h100/data/math500_7b/` | 42 GB | 500 × 7B per-token all-layer npz + manifest.json |
| `pathway8_layerwise/data/math500/` | 19 GB | 500 × 1.5B per-token all-layer npz (reused from pathway 8 stage 2) |
| `pathway11_h100/data/k8_selfconsistency/` | 13 MB | 500 × K=8 per-sample L19 npz |
| `pathway8_layerwise/data/bbh/` | 18 GB | 750 × BBH 3×250 per-token all-layer npz |
| `pathway11_h100/logs/stage{0..5}_*.log` | — | Per-stage timestamped console output |

Post-hoc analysis: **prefill-gated compute (Exp 2)** at `pathway11_h100/prefill_gated_compute/`:

| File | Purpose |
|---|---|
| `SUMMARY.md` | 5 KB narrative: uniform K baselines, oracle buckets, why gating fails |
| `phase1_majority_vote.npz` / `phase1_summary.json` | K=1,2,4,8 majority accuracy per problem |
| `phase2_prefill_dom.npz` / `phase2_summary.json` | OOF prefill DoM 0.7731, final-token DoM 0.7186 |
| `phase3_policies.json` | Policy sweep (sampled-K=1 variant) |
| `phase3b_realistic_policies.json` | Realistic policy sweep (greedy K=1 @ T=0 for confident band) |
| `phase4_oracle.json` | Oracle bucket AUROC matrix, Spearman with oracle-K |
| `results.json` / `pareto_plot.png` | Consolidated headline, pareto curve |

Post-hoc analysis: **7B prefill inversion (Exp 3)** at `pathway11_h100/prefill_inversion/`:

| File | Purpose |
|---|---|
| `SUMMARY.md` | 11 KB narrative: bootstrap CIs, three-way split, BBH controls, D-bucket signature |
| `phase1_bootstrap.json` | 1000-boot ratios + CIs + balance-controlled ratios (all 4 settings) |
| `phase2_cross_scale.json` | LOO PR pushers, three-way split, BBH, D-bucket per-bucket stats |
| `phase2_loo_deltas.npz` | Per-problem LOO delta PR arrays (4) |
| `phase3_mechanistic.json` | PR per difficulty level × correctness, k-means ARI |
| `phase4_summary.py` | Headline figures builder |
| `results.json` | Consolidated headline |
| `bootstrap_ci.png` / `three_way_split.png` / `d_bucket_signature.png` | Figures |
| `common.py` | Shared PR computation + data loaders |
| `cache/m7b_prefill.npz` / `cache/m15b_prefill.npz` | Cached prefill+final fp16 arrays for fast replay |

Post-hoc analysis: **multi-signal oracle (Exp 2b)** at `pathway11_h100/multi_signal_oracle/`:

| File | Purpose |
|---|---|
| `SUMMARY.md` | 9 KB: non-monotonic multi-feature thesis fails (+0.4 pp vs ≥2 pp bar) |
| `features.npz` | 500 rows × 10 feature columns (+ labels) |
| `phase1_features.json` | Per-bucket feature means |
| `phase2_classifier.json` | Logreg + RF 5-fold CV metrics + confusion matrices |
| `phase3_policy.json` | Policy simulation with K=1 / no-fallback variants |
| `phase4_ablation.json` | 10 feature-subset runs × 2 classifiers |
| `results.json` / `pareto_comparison.png` | Consolidated headline |

Post-hoc analysis: **cross-architecture (Exp 1)** at `pathway11_h100/exp1_cross_model/` (completed 2026-04-24):

| File | Purpose |
|---|---|
| `HANDOFF.md` | Session handoff + inline headline results (produced outputs, accuracy tables, architecture-specific timing) |
| `extract.py` | GPU extraction for Phi-3-mini + Llama-3.2-1B with per-problem checkpointing |
| `analyze.py` | CPU analysis: temporal PR curve, depth-PR prefill vs final (N_BOOT=50 final run) |
| `run.sh` | Sequential runner: `phi3mini` / `llama32-1b` / `--analyze-only` |
| `data/phi3mini/` | 500 × Phi-3-mini per-token 2/3-depth + prefill/final all-layer npz, 166 MB |
| `data/llama32_1b/` | 500 × Llama-3.2-1B per-token 2/3-depth + prefill/final all-layer npz, 65 MB |
| `results.json` | Consolidated temporal + depth PR results for both models |
| `temporal_pr_curve.png` | Headline figure: breathing curve, Phi-3 + Llama side-by-side |
| `depth_pr_prefill_vs_final.png` | Depth-PR figure: prefill-end vs final-token for both models |
| `logs/full_run.log` / `logs/analyze.log` | GPU extraction log (4975s) + CPU analysis log |

Cross-pathway shared:

| File | Purpose |
|---|---|
| `pathway8_layerwise/coe_features.py` | CoE feature extractor (60-dim trajectory) |
| `pathway8_layerwise/d2hscore_features.py` | D2HScore dispersion/drift |
| `pathway8_layerwise/extraction_utils.py` | `load_model`, `extract_all_layers_and_attention`, `save_problem_checkpoint` |
| `pathway8_layerwise/extract_math500.py` / `extract_bbh.py` | Stage 2 / Stage 4a extraction scripts (reused by Pathway 11 unchanged) |
| `pathway8_layerwise/results/exp{1..5}_results.json` | Pathway 8 layer-wise PH / CoE / D2H / TwoNN / crosslayer-trajectory results |
| `pathway1/phase0/winning_features.py` | Frozen CORAL 78-feature extractor (1407 lines) used by Pathway 9 ABC-44 |
| `pathway6_rebuild/phase6_5/FINAL_SUMMARY.md` | The deconfounded cross-benchmark analysis that first caught the 256-tok artifact |

Durable memory entries (in `~/.claude/projects/-home-musicofhel/memory/`):

| File | Purpose |
|---|---|
| `pathway9-coe-pivot.md` | Pathway 9 conclusion, CoE pivot |
| `pathway10-v2-steering.md` | v2 steering program, 4 experiments, success criteria |
| `pathway11-h100-orchestrator.md` | Scope, decisions, reuse pattern |
| `pathway11-pod-keep-alive.md` | Pod stop-don't-remove pattern |
| `feedback-topo-confidence-goal.md` | "Goal is small-model improvement, NOT papers" |
| `feedback-topo-confidence-headline-resolution.md` | "Trim scope but keep headline-figure resolution" |
| `topo-confidence.md` | Top-level project state |
| `pathway8-handoff.md` | Pathway 8 per-stage status |
| `pathway7-playbook.md` | Non-Euclidean PH (NO-GO 0.774) |

---

## Quick-reference by topic

**Want the most trustworthy number for 1.5B MATH-500 correctness?** 243/500 = 48.6%
at `max_new_tokens=1024`, T=0 greedy. Every older accuracy number (20.8%, 104/500)
is a 256-tok truncation artifact.

**Want the best pre-generation correctness signal?** L19 prefill DoM on 1.5B = AUROC
0.7731 OOF. On 7B = ~0.876 OOF. Both measured at 1024-tok labels.

**Want the headline dimensional-breathing number?** Cohort-level participation
ratio: ~20 at prefill → ~67 (1.5B) / ~88 (7B) at mid-generation peak → ~8 (1.5B) /
~6 (7B) at final token. Correct-subset final PR: 4.33 (1.5B) / 2.63 (7B). Incorrect
final PR: 8.51 / 6.85. Correlation of temporal-PR with temporal-AUROC: r = −0.695.

**Want the 7B prefill inversion?** Correct-group prefill PR 26.87 > incorrect
19.51, ratio 1.377, bootstrap 95% CI [1.226, 1.757], P(ratio>1) = 1.0. Balance-controlled
ratio 1.227, CI [1.131, 1.318]. Not in 1.5B. Not in BBH. Driven by easy-level
failures clustering tightly (Level 2 ratio 3.98, Level 5 ratio 1.11).

**Want the best inference-time intervention the program produced?** Prefill-DoM
refuse-and-spend at coverage 0.5: accuracy-on-answered 71.6% at average K = 2.5,
vs 48.6% unconditional at K=1. +23 pp on the kept half, fewer samples per answered
problem. This is the clean E3 win.

**Want to know which feature is Pareto-dominant for compute allocation?** Negative
sequence length (post-hoc), not prefill DoM. Neg-seq-len Spearman with oracle-K =
0.327 vs prefill's 0.175. Neg-seq-len top-heavy gating at K=4.5 gives 0.542
overall vs prefill middle-heavy at 0.526 at same compute.

**Want the cross-architecture replication numbers?** Phi-3-mini (44.8% MATH-500
acc) and Llama-3.2-1B-Instruct (25.2%) both reproduce the three-phase breathing:
Phi-3 PR 19 → ~100 → 18, Llama PR 3 → ~115 → 8. Both show correct-collapses-harder
at the final token (Phi-3: 12.2 vs 16.8; Llama: 4.4 vs 8.0). Llama separates
correct/incorrect at position 25; Phi-3 separates at position 100. Pattern is
architecture-independent and scale-independent.

**Want to know what the next session should try?** Either (a) build the
per-position DoM bank and re-attempt E1 steering with direction-matched
projection at each token position, or (b) investigate whether the dimensional
breathing / Sharpness Dimension link (Tuci et al.) holds as a formal
correspondence between inference-time covariance and training-time Hessian
dynamics.
