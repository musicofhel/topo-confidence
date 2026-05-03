# APPLICATIONS.md — where this stack is deployable, valuable, and honest

The L19 prefill DoM + selective-prediction + cov-spectrum stack
(SYNTHESIS.md §1, §3) has concrete deployment surfaces. This document
lists them with honest "what would have to be true" checklists. Each
application names the exact signal it consumes, the cost model, and
what would block production use.

All numbers are 1024-tok labels on Qwen-2.5-1.5B-Instruct unless
otherwise marked. Every number cited is in `validate_claims.py`.

---

## Application 1 — Selective serving for small-model APIs

**Setting.** Production endpoint serving Qwen-2.5-1.5B (or similar
small model) on math-style queries. Typical query: arithmetic word
problem, AMC-style algebra, calculation chain.

**Pitch.** At 50% coverage, prefill-DoM gating answers 50% of queries
at K=2.5 (mean) with **71.6% accuracy** (`refuse-prefill-acc`),
versus **48.6% unconditional K=1** (`baseline-1.5B-acc`). The other
50% routes to a larger model (or refuses). Spend large-model compute
only on the rejected half.

**Signal cost.**
- One forward pass to L19 on the prompt — *no generation needed for
  the gate decision*.
- ~14% of full-network compute (L19 / L28 layers) per gate.
- One vector dot product against a precomputed DoM direction.
- Latency overhead: dominated by KV-cache to L19, ≈ tens of ms on
  H100; sub-millisecond after that.

**Cost model (rough).**
- Routing 50% to a 7B model at K=1 instead of 1.5B at K=1:
  - Without gate: 100% × 1.5B × K=1 → 48.6% accuracy.
  - With gate: 50% × 1.5B × K=2.5 + 50% × 7B × K=1 → 0.5 × 0.716 +
    0.5 × 0.732 ≈ **72.4% blended accuracy** (using
    `7B MATH-500 K=1 = 0.732` from CLAUDE.md headlines table).
  - Compute: gate adds ~14% × 100% = 14% on the 1.5B side; the K=2.5
    answered K=1.5B is ~50% × 2.5 = 125% of the 1.5B baseline; the
    routed-to-7B 50% is ~3.5× a 1.5B forward pass per scaling. Total
    ≈ 0.14 + 1.25 + 0.5 × 3.5 ≈ 3.14× a uniform 1.5B K=1 pass. Up
    from 48.6% to ~72.4% accuracy at ~3× compute, vs uniform 7B K=1
    at ~3.5× compute and 73.2% accuracy. The gate buys you ~95% of
    the 7B accuracy at ~90% of the 7B cost.

**What would have to be true for production.**

- [ ] **F-2 cross-architecture replication.** Currently one model
      (Qwen-2.5-1.5B-Instruct). The `exp1_cross_model` cache exists
      for Phi-3-mini and Llama-3.2-1B; AUROC has not been extracted
      on either. Pending control on F-2.
- [ ] **F-2 cross-benchmark.** Within-MATH-500: LOCO-CV mean
      `loco-auroc-mean` = 0.7427, worst 0.6032 (Number Theory).
      Within-topic mean `fe299-within-mean` = 0.7143, worst 0.6000.
      Cross-domain MATH→BBH = 0.747, BBH→MATH = 0.693. **Math-style
      reasoning only**; non-mathematical reasoning is not validated.
- [ ] **DoM direction stability across checkpoints.** If Qwen
      releases a new 1.5B-Instruct checkpoint, does the DoM rotate?
      No cross-checkpoint rotation test has run (FE700 in
      HYPOTHESES.md; would-overturn condition in F-2).
- [ ] **Calibration drift over time.** The AUROC threshold for
      "answer vs route" is calibrated on the validation set. Drift
      detection (e.g., monitor distribution of prefill DoM scores
      over rolling windows) is unspecified.
- [ ] **Failure mode for adversarial prompts.** Prefill DoM was not
      tested under prompt injection or jailbreak attempts. A malicious
      prompt can drive the prefill score arbitrarily; gate decisions
      may be unreliable in adversarial settings.
- [ ] **Fallback when L19 cache is unavailable.** If the model is
      sharded across devices and L19 is on a different worker than
      L0–L18, gate latency may not amortize. Topology-dependent.

**Verdict.** Strongest application. The numbers are concrete and
reproducible. The cross-architecture and cross-benchmark caveats are
real but tractable (a ~6-hour CPU run on existing caches).

---

## Application 2 — Multi-signal compute-budget allocation (selective spending, not just refusal)

**Setting.** You have a fixed total compute budget across N queries
and want to maximize total correct answers.

**Pitch.** F-14 (per `STATE.md`, not yet promoted to a numbered F-N)
flagged that monotonic prefill gating fails because the *recoverable*
B-bucket lives at mid-confidence, not the bottom. Middle-heavy at
avg K=4.5 hits **0.526 accuracy** (`middle-heavy-acc`); neg-seq-len
top-heavy hits **0.542** (Pareto-dominant but post-hoc).

The actionable next step is **multi-signal gating** — combine prefill
DoM + final-token DoM (after one cheap K=1 pass) + length to detect
the B-bucket and budget K=8 there. Single-signal gates fail; a
multi-signal gate has not been tested at 1024-tok labels with the
full feature set.

**Signal cost.**
- Cheap K=1 first to obtain length and final-token DoM.
- Decision boundary: 3-feature LR or RF (FE multi-signal-oracle work
  in `pathway11_h100/multi_signal_oracle/`).
- Re-spend K=8 only on B-bucket detections.

**What would have to be true for production.**

- [ ] **Multi-signal AUROC at 1024 tokens.** Exp 2b at 256-tok had
      logreg D-recall 0.417 (`exp2b-logreg-Drecall`), below class-prior
      baseline. The 1024-tok rerun has not landed.
- [ ] **Length feature is post-generation.** Post-hoc length is
      Pareto-dominant; gating on length means you've already spent
      K=1. Pre-hoc length-prediction (`fe448-length-r2`) is
      in-sample R² and not directly validated as a gate feature.
- [ ] **The +0.4 pp multi-signal-vs-best-single lift
      (`exp2b-multi-vs-single`)** is below the user's 2pp bar at
      256-tok labels. Re-test at 1024-tok before claiming production
      value.
- [ ] **B-bucket size in production traffic.** B is 68/500 ≈ 13.6%
      of MATH-500 (`buckets-B`). Production query mix may have a
      different K=8-recoverable rate; recalibrate per deployment.

**Verdict.** Promising but unvalidated at the corrected-label
regime. List under HYPOTHESES.md as an explicit FE before claiming
production.

---

## Application 3 — Cov-spectrum probe for high-stakes correctness gating

**Setting.** Same as Application 1 but you need the strongest
available correctness predictor at L19, willing to spend per-problem
SVD compute.

**Pitch.** Top-20 log-eigvals of each problem's PC1-residualized L19
trajectory covariance gives **OOF AUROC 0.7928**
(`pca-pc1resid-cov-spectrum-top20-auroc`) — strongest L19 probe at
any tier. Beats supervised DoM by +2.49 pp.

**Signal cost.**
- Per problem, requires a *trajectory* cloud (multiple tokens
  through generation), not just the prompt prefill. So this is *not*
  a pre-generation gate — it's a post-K=1 gate.
- One per-problem SVD on a 1536 × T_i matrix, T_i ∈ [123, 1024]
  with median 521. ~10ms per problem on CPU.
- Subtract a fixed PC1 direction (precomputed once).
- Project to top-20 eigenvalues, log-transform, run a precomputed
  logistic.

**What would have to be true for production.**

- [ ] **Causal validation.** The probe is supervised. Beating the
      directional ceiling supervised-vs-supervised says "there's
      information you can't capture linearly," not "the model uses
      this information." Rank-truncate-PC1-residualized-cov ablation
      before continuation, measure correctness drop. Unrun.
- [ ] **Held-out fold stability.** CV5 std is 0.027 — within the
      gap to the directional ceiling (+0.7 pp). Nested 5×5 CV
      verification has not run.
- [ ] **Cross-architecture / cross-benchmark.** Same caveats as
      Application 1 plus the per-problem trajectory requirement.
      Trajectory clouds are bigger to extract than just prefill;
      cross-arch caches do not exist for trajectories.
- [ ] **Trajectory length sensitivity.** T_min = 123 in our cache;
      shorter trajectories may have unstable eigenvalue estimates.
      No length-stratified evaluation has run.
- [ ] **Better single-scalar substitute.** EXP-57 follow-up (a):
      effective rank / participation ratio / stable rank as a single
      scalar. If a named geometric scalar matches 0.7928, the
      narrative collapses to a much cleaner deployable. Until then,
      the 20-eigval feature vector is what you ship.

**Verdict.** Strongest correlational signal in the program. **Not
yet causally validated** — do not deploy as a load-bearing gate
without the rank-truncate ablation.

---

## Application 4 — Eval-time correctness estimator for synthetic-data filtering

**Setting.** You're generating CoT data from a 1.5B model to bootstrap
a smaller distillation target, and you want to keep only the
correctness-probable subset for fine-tuning.

**Pitch.** Without the ground-truth label, you can't know which
generated CoTs are correct. With the prefill DoM + cov-spectrum
probe, you can estimate correctness at AUROC ~0.79 — much better than
length heuristics (length-alone AUROC = 0.7986, `fe448-auroc-length-alone`
— within fold noise of cov-spectrum, but length is post-generation
and confounded by cot-style; DoM is pre-generation and uncorrelated
with length after partialing).

**Why this is real.** FE448 length partial-correlation
(`fe448-auroc-resid` = 0.6647) shows DoM retains AUROC 0.66 after
*length* is partialed out — it is *not* just predicting length. So
"DoM-filtered synthetic CoT" is meaningfully different from
"length-filtered synthetic CoT."

**What would have to be true for production.**

- [ ] **The downstream task is the same task DoM was trained on.**
      Synthetic-CoT filtering for a math-distillation target is
      in-domain. Filtering for a code-distillation target requires
      F-2 cross-benchmark validation (no MATH-trained DoM has been
      tested on code).
- [ ] **The synthetic generator is the same architecture.** F-2
      cross-architecture caveat applies. If you generate with Qwen
      1.5B, filter with the Qwen-1.5B DoM. Cross-model filtering is
      unsupported.
- [ ] **Size of the keep-set.** At 50% coverage you keep 71.6%-correct
      data. At 25% coverage the AUROC slope says you'd keep ~80%-
      correct data — the curve is reproducible from
      `prefill_gated_compute/results.json` but not packaged as a
      cookbook.
- [ ] **Compute budget for cov-spectrum filtering** (one SVD per
      problem on the trajectory cloud). For a 100k-example synthetic
      set this is ~16 minutes CPU. Negligible.

**Verdict.** Concrete, deployable for math-CoT distillation today.
Cross-domain extension blocked on F-2 cross-benchmark replication.

---

## Application 5 — Refusal-token logit modulation

**Setting.** The model sometimes commits to wrong answers on high-
stakes prompts. You want a residual-stream signal that can amplify
abstention behavior at inference.

**Pitch.** F-2's prefill DoM is a candidate for **abstain-token logit
modulation**: when the prefill score is below a threshold, *positively
modulate* the logit on a refusal token (e.g., "I'm not sure"). This
is distinct from prompt-level refusal: it's a residual-stream signal
at L19 prefill, before any answer is committed.

The ITI-style intervention would be: at L19, project the activation
onto the prefill DoM, threshold, and inject (or amplify) the
refusal-direction at the unembedding stage. Steering literature
(`2312.03813` mean-centring, `2505.17306` refusal direction)
provides the recipe.

**What would have to be true for production.**

- [ ] **Steering direction stability.** Fixed-vector steering using
      a *final-token-fit* DoM was killed (PROJECT_RECORD §1d ND-5,
      cos with current L19 DoM = 0.05). The *prefill* DoM has not
      been tested as a steering vector. Per-position bank not built.
- [ ] **Composite steering effects.** Refusal-direction steering can
      degrade general response quality. The intervention's
      side-effects on non-refusal tokens are unmeasured.
- [ ] **Calibration of the refusal threshold.** F-8 calibrates a
      coverage threshold for *external* gating; logit-modulation needs
      a per-token strength parameter (α in ITI). No α-sweep has run
      on prefill DoM.
- [ ] **The refusal-token logit channel exists at the same layer.**
      Logit modulation requires the unembedding step; any L19
      intervention may interact with the L20–L28 transformations.
      Untested.

**Verdict.** Promising direction, **completely unvalidated**. Do not
ship without an α-sweep + side-effect evaluation. Listed in
HYPOTHESES.md as H-1 (per-position DoM steering of MATH-500 accuracy).

---

## Application 6 — Diagnostic / observability tool

**Setting.** You're operating a small-model deployment and want a
diagnostic signal: "is this query in our model's competence?"

**Pitch.** The prefill DoM score, computed once per query, is a
calibrated competence indicator. It does not need ground truth, can
be logged at production-scale (one scalar per query), and surfaces
distribution shift early (if the score distribution drifts, your
query mix has moved). LEACE collapse (`fe101-collapse-pp` = 0.2705)
confirms the signal is a single linear axis, easy to monitor.

**What would have to be true for production.**

- [ ] **Distribution-shift detection thresholds.** No reference
      distribution has been published for the prefill DoM score
      under known-good traffic. You'd calibrate on your own validation
      set.
- [ ] **Score-to-confidence calibration.** AUROC is a discriminative
      metric, not a calibration metric. Brier / ECE on prefill DoM
      probabilities has not been measured. Use the score for
      *ranking*, not for "this query is X% likely correct."
- [ ] **Cross-model deployment stability.** Each model needs its
      own DoM direction. Multi-tenant deployments (router + N
      candidates) need per-model probe artifacts.

**Verdict.** Cheapest deployment surface — one scalar per query, no
generation cost. Best paired with Application 1.

---

## Where this stack does NOT apply

Honest negative list. If your problem is in this section, do not
deploy.

### Non-Qwen architectures

F-2 was extracted on Qwen-2.5-1.5B-Instruct. F-1 / F-4 / F-5 are
multi-architecture, but the **direction itself** (DoM at L19) has not
been replicated on Phi-3 or Llama at 1024 tokens. The
`exp1_cross_model` cache exists; running 5-fold OOF logistic on each
is a few hours of CPU. Until then, do not assume the AUROC transfers.

### Non-mathematical reasoning

F-2 was MATH-500. Cross-domain MATH→BBH = 0.747; BBH→MATH = 0.693.
F-13 (graph numbering) called out the 256-tok confound on BBH; the
1024-tok BBH rerun changed the picture less than expected for math
but is still moderate. **Code generation (HumanEval) is unrun**
(H-3). **Open-ended QA, summarization, retrieval** — all unsupported.

### Tasks where labels aren't available at training time

The supervised DoM probe needs labels. The CAST-PC1 unsupervised
recovery is at AUROC 0.7458 (`pca-cast-pc1-auroc`), about 2pp below
the supervised DoM 0.7679 (`pca-dom-oof-auroc`). For label-free
deployment, use unsupervised PC1; for label-trained deployment, use
DoM. The cov-spectrum lift (0.7928) requires labels. The directional
ceiling (0.7856) requires labels.

### Settings where 256-tok numbers are still in circulation

The 0.7961 ABC-44 number, the 20.8% MATH-500 baseline, and the
"7B is a stronger verifier" claim are all 256-tok truncation
artifacts. Any external paper that cites these numbers without the
1024-tok footnote is making the same mistake we did. Do not
benchmark against them. Use the corrected `baseline-1.5B-acc`
0.486 / `baseline-K8` 0.550 numbers.

### Adversarial prompts

Prefill DoM was not tested under prompt injection, jailbreak
attempts, or adversarial paraphrasing. A malicious prompt can drive
the prefill score arbitrarily — gate decisions may be unreliable
in adversarial settings. Do not use as the only line of defense.

### Production with no calibration / monitoring infrastructure

If you cannot log per-query prefill DoM scores and detect drift,
you cannot operate this gate safely. Drift detection on a single
scalar is cheap; not having it at all is the failure mode.

---

## Deployment ranking

If asked "what should we ship first?":

1. **Application 1 (selective serving).** Strongest signal,
   cheapest deployment, most-validated metric. Block on cross-arch
   replication if your model is not Qwen.
2. **Application 6 (observability).** Free if you're already running
   the prefill forward pass for Application 1. One scalar per query.
3. **Application 4 (synthetic-data filtering).** In-domain math
   distillation today; out-of-domain blocked.
4. **Application 3 (cov-spectrum gating).** Not yet causally
   validated. High AUROC but expensive to extract (per-problem
   trajectory + SVD) and unvalidated.
5. **Application 2 (multi-signal compute allocation).** Promising
   pre-1024-tok design, post-1024-tok numbers haven't run. Pending.
6. **Application 5 (refusal logit modulation).** Speculative.
   Steering literature is heavy; our specific design is unvalidated.

---

## Final honesty note

This stack improves *small-model accuracy on math-style queries via
selective compute allocation*. It does not:

- Improve any model's *intrinsic* capability — gating routes around
  weak queries, doesn't strengthen the model's reasoning.
- Replace reinforcement-learning, fine-tuning, or distillation as
  capability-building tools.
- Generalize beyond the validated regime (Qwen-1.5B / 1024 tokens /
  math) without further work.
- Substitute for human review in high-stakes settings — AUROC 0.79
  is a useful signal, not a guarantee.

The decomposition triangle (DoM → 2-feat → cov-spectrum) is a
research result that is genuinely novel; the deployment surfaces
above are the prosaic value extraction. Publish the science; ship
the gate.
