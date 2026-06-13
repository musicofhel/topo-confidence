# Generalization-First Edge Program — SPEC v6

_Pushing the correctness-prediction signal that feeds F-8 selective prediction — measured by
**cross-distribution transfer**, not in-domain fit — AND the applied surfaces that consume it
(selection, cascade, guarantees). All compute LOCAL (2060 Super + CPU)._

> **v6 changelog (POST-EXECUTION re-scope, 2026-06-11).** This is the first changelog written
> *after* running the spec. SPEC v5's Phase 0 + Phase 1 + Phase 1B were **EXECUTED end-to-end on
> local CPU** (no generation) as **EXP-81**. The harness lives at
> `pathway11_h100/generalization_edge/` (`activation_loader.py`, `probes.py`, `transfer.py`,
> `metrics.py`, `k8_lib.py`, `verify_anchors.py`, `exp_1b1_reranking.py`, `exp_1b2_earliest.py`,
> `exp_1b3_conformal.py`, `exp_1b4_overlap.py`, `phase1_bakeoff.py`), built **on**
> `nocompute/lib.py`; results in `pathway11_h100/generalization_edge/results/`
> (`phase1_panel.json`, `1b1_reranking.json`, `1b2_earliest.json`, `1b3_conformal.json`,
> `1b4_overlap.json`, `1b4_overlap_table.csv`); full result brief at
> `research-graph/briefs/result-2026-06-11-P11-FE-EDGEGEN.md` (fe_id `P11-FE-EDGEGEN`).
> **15/15 verification anchors reproduced exactly.** The execution resolved the program's central
> tension empirically and the **forward plan (Phases 2–4) is now re-scoped around what actually
> generalized.** Five updates, all grounded in the saved artifacts:
>
> **(V5-1) RE-SCOPE Phases 2/3 — the layer-profile re-extraction is now low-value, drop it.** The
> Phase-1 generalization winner is **only the free baseline (length + mean_logprob)**: T1 LOCO
> aggregate **0.845** (worst 0.735 number_theory), T3 cross-scale **0.865** (worst 0.837), near-zero
> overfit gap — beating the incumbent prefill-DoM (T1 0.743 / worst 0.603). The geometry bet **lost**:
> raw CoE layer-profile is the *only* feature that clears the in-domain incremental gate (+0.0175,
> p=0.018, OOF 0.854 — see `results/phase1_panel.json`) but it is **hidden-dim-bound** (cannot cross
> scale/arch), and its **portable depth-grid form neither adds in-domain (+0.0081, p=0.137) NOR
> transfers** (T2 0.588, T3 0.627 — the *worst* cross-scale of any candidate). **H-C REFUTED.** So
> Phase 2's "re-extract Phi-3 / Llama-3.2 layer-profile trajectories" is dropped — there is no
> portable geometry feature to carry into the cross-arch cells. Phase 2 keeps **only** the
> **prefill-only pass for arm-2A (prompt-token-cloud)** + the free length/logprob columns.
>
> **(V5-2) REFRAME the T5 STRETCH around the FREE BASELINE, not geometry.** Since the free
> length+logprob readout is the generalizing winner and **needs no activations**, the live T5
> question becomes: *does length+logprob hit ≥0.70 on the two held-out families (SmolLM2-1.7B,
> Gemma-2-2B-it)?* If yes, that is a **shippable, model-agnostic, generate-then-abstain F-8 gate** —
> the program's deliverable, full stop (the V3-4 "ship the simplest winner" fork, now the headline
> path). This only requires **generation** on the held-outs (`causal_dom/common.py`), not the
> expensive trajectory extraction. arm-2A prompt-cloud is demoted to a *secondary* geometry probe.
>
> **(V5-3) arm-2A (prompt-token-cloud) is now the ONLY untested geometry** and the single remaining
> shot at a **portable pre-flight (Tier-A) gate**. It was logged as `NOT_IMPLEMENTED` in
> `phase1_bakeoff.py` (it needs the prefill-only pass that Phase 1 never ran). Every other geometry
> arm is now resolved: token-cloud cov-spectrum REFUTED (dry run + closure), layer-profile depth-grid
> REFUTED (above), arm-3 GRIDE/PHD never beat the free baseline in-domain. So arm-2A is elevated from
> one-of-three-arms to **the one open geometry question** — run it in Phase 2's prefill pass and gate
> it against **prompt-length** (still unmeasured anywhere).
>
> **(V5-4) ELEVATE the hybrid 1.5B→7B cascade to the headline ROUTING deliverable.** 1B.4
> (`results/1b4_overlap.json`, `1b4_overlap_table.csv`) found the 7B rescues **52.9%** of the 257
> K=1 fails (number_theory 74%, algebra/prealgebra ~75%; unrescuable core =
> intermediate_algebra / precalculus / geometry), while K=8 self-consistency rescues only 27.2%.
> Against the FE19 random-mix hull (the honest comparator, line (0.2,0.486)→(1.0,0.732)), the
> **post-gen HYBRID cascade gains +3.38 ± 0.96pp — SIGNIFICANT** — the *one* routing result in the
> whole program that beats its honest baseline (the pre-gen-only G1 gate is +1.7pp, n.s., exactly the
> hull-marginal outcome V4-1 predicted). This is the live routing product; Phase 4 calibrates it.
>
> **(V5-5) Make Phase 4 conformal recalibration LOAD-BEARING, and record the hypothesis
> resolutions.** 1B.3 (`results/1b3_conformal.json`) CONFIRMED **H-H**: split-conformal / LTT
> certificates are valid in-domain (δ=0.1, ε=0.2 → free_baseline 0.855 validity @ 28% cov / 85% acc)
> but **collapse to 0 coverage under the MATH→BBH shift** (infeasible at ε=0.1/0.2, coverage 0.0 at
> ε=0.3). So a deployable guarantee **requires** target-domain recalibration — Phase 4 is no longer a
> LOSE-clause consolation, it is the only path to a valid cross-domain certificate. **Hypothesis
> resolutions to carry forward:** **H-A** pre-confirmed (token-cloud null); **H-C / H-F / H-G
> REFUTED**; **H-H CONFIRMED**; **H-40 / FE41 EXECUTED** (1B.1). And **re-pin the F-8 readout from
> prefill-DoM (0.7731) to length + mean_logprob** — it dominates on transfer (T1 0.845 vs 0.743) at
> zero activation cost.
>
> _Phase 1B sub-results (for the record): **1B.1 / H-F REFUTED** — no reranking arm beats K=8
> majority 0.554 (probe-argmax 0.480, −7.4pp, McNemar p<0.001 WORSE; weighted-maj 0.538 n.s.;
> confidence-fallback 0.552 n.s., no D-bucket regression); `L19_samples` confirmed final-token.
> **1B.2 / H-G REFUTED** — L19 mean-pool DoM stays below the prefill gate until ~m200 (m25 0.704,
> m200 0.783, full 0.819), the ≥0.82-by-m≤128 trigger never fired, abort-escalate costs 0.96× for
> +3.7pp (useless) — skip early-abort drafting. **CORRECTION (anchor):** the v5-prose anchor
> "MATH→BBH 0.747" is **STALE** — it matches no recipe incl. the saved canonical
> `nocompute/results/C1_math_to_bbh.json` (pooled mean-DoM = 0.4158, which anti-transfers); the real
> finding is **prefill-DoM ports to BBH at 0.796 while mean-DoM INVERTS to 0.416** — `verify_anchors.py`
> was corrected to assert against the artifact. **Gate-bug fix:** `transfer.py::incremental_gate`
> originally concatenated raw 1536-d activations with the 2 free scalars (overfit → spurious negative
> deltas); fixed to gate on the probe's **1-D OOF readout score** + `[n_gen_tokens, mean_logprob]` in
> a logistic vs free-only, paired DeLong. **Graph triage done (no novelty claims):** 2402.10978
> (conformal-factual-lm), 2406.15927 (Semantic Entropy Probes), 2205.14334 (verbalized uncertainty,
> already-rejected H-12 dup) — all graphed/triaged before 1B.3._

> **v5 changelog (10,000-ft whole-repo audit, 2026-06-11).** A repo-wide sweep
> (FINDINGS/STATE/EXPERIMENT_LOG/nocompute results/caches + research-graph queries) re-examined the
> *applied* surface — the user's goal is "confidence in correctness + routing" — and found the spec
> optimizes the readout but under-exploits its **uses**. Five findings, all grounded in existing
> artifacts; net change = new **Phase 1B (applied quick wins, CPU-only, runs alongside Phase 1)**
> plus a conformal guarantee layer in Phase 4:
> **(V4-1) The routing verdict was over-generalized.** FE19 killed *same-model K-routing*, but
> `nocompute/results/FU2_cascade_combined.json` shows the **1.5B→7B cascade already works**: the
> hybrid (post-gen) signal hits AUROC **0.852**; at a 90%-of-7B accuracy target it keeps 46% of
> problems on the 1.5B (cost 0.632×) vs 33% for the pre-gen probe (0.736×) vs **oracle 64%
> (0.488×)** — an **11.2pp savings gap to oracle** that every +1pp of readout AUROC converts into.
> Cascade conversion is now an explicit deliverable surface. The pre-gen-only cascade (G1) is
> hull-marginal (+~1pp ≈ 0.5 SE vs random 1.5B/7B mixing at matched cost) and must be
> re-adjudicated against the random-mix hull with paired tests — the FE19 lesson applied to G1.
> **(V4-2) A registered 1-hour experiment was never run.** H-40 / P11-FE41 probe-reranking: the
> K=8 cache (`pathway11_h100/data/k8_selfconsistency/`, 500 problems) holds **per-sample L19 states
> `(8,1536)` + per-sample correctness** (rows verified distinct per sample). Reranking is
> *selection at fixed compute*, not routing — the FE19 hull baseline structurally cannot apply; the
> comparator is K=8 plain majority (55.0%). Verify first that the cached position is the final token.
> **(V4-3) Nobody measured WHEN the signal arrives.** Prefill = 0.77, full-gen = 0.85, and the full
> `(29,T,1536)` caches permit a pure-simulation **earliest-decision curve**: probe AUROC vs prefix
> length m. This unifies Tier-A/Tier-B into one time axis and underwrites the only routing shape
> never tested or refuted: **draft → abort early → escalate**.
> **(V4-4) "Confidence in correctness" lacks the guarantee form.** F-8 is a risk-coverage *point*;
> calibration is partial (histogram ECE 0.041). Add **split-conformal / Learn-then-Test risk
> control** on the existing OOF scores → a (τ, ε, δ) certificate. Graph-checked: conformal-adjacent
> entries 2205.14334, 2402.10978, 2406.15927 already in the research-graph — triage before any
> novelty claim.
> **(V4-5) The per-problem 1.5B×7B overlap table does not exist** (which 7B rescues are real, by
> category) — a 30-min CPU job that is the prerequisite for V4-1's re-adjudication and V4-3's
> escalation sim.

> **v4 changelog (third fresh-eyes audit + 200-problem DRY RUN, 2026-06-10).** This audit ran v3's
> own pre-registered gates on 200 cached MATH problems before committing compute. Eight findings:
> **(V3-1, severe, empirical) The token-cloud bet is already dead at its own gate.** On the dry run
> (L19, first 200 problems): token-cloud spectrum **+ length 0.761 < length-only 0.786** (the
> spectrum adds *negative* value over length); logT-residualized spectrum **0.554 ≈ chance**; and at
> **fixed m=100 subsampled tokens** the spectrum still correlates −0.58 with T — the length coupling
> is genuine geometry-covaries-with-length, NOT a sample-size artifact, so no eigenvalue correction
> rescues it. H-A's null is effectively pre-confirmed. cov-spectrum is demoted to one Phase-1
> closure row.
> **(V3-2, severe, methodological) v3's gate (ii) was mis-specified.** Residualizing features on
> logT destroys *legitimately shared* variance because the **label** is length-correlated too.
> Proof: CoE-60 fails naive residualization (0.595) yet **adds +4pp over length** (0.826 vs 0.786).
> Gate (ii) is now an **incremental-value test** — paired (feature+length) vs length-only — with the
> residualized AUROC reported as a *decomposition* (size of the length-orthogonal channel), not a
> kill criterion.
> **(V3-3) Missed candidate family, now the arm-2 core (fork: re-center).** The spec omitted the
> already-implemented **layer-axis profile features**: CoE-60 (`pathway8_layerwise/coe_features.py`,
> exp2 holdout **0.811 > cov-spectrum 0.7928**; dry-run 0.825), D2H-dispersion, cached
> `d2h_attn_entropy`. The layer axis doesn't grow with generation; a per-layer profile resampled to
> a **fixed depth-fraction grid** is fixed-dim across architectures and angles are scale-invariant.
> **(V3-4) The free two-scalar baseline is the real bar.** **length + mean_logprob = 0.834** on the
> dry run — no activations, model-agnostic by construction, enters every tier including T5. New
> pinned **Tier-B free baseline**. (fork: if it's the best portable T5 signal, **ship it** — goal is
> model improvement, not geometry.)
> **(V3-5) Length portability was asserted, never measured.** Length-only and length+logprob now run
> on ALL tiers as first-class candidates; their transfer is a question, not a nuisance. A
> **prompt-length** baseline is also missing everywhere (`m15b_prefill.npz:seq_len` is GENERATION
> length — identical 123/521/1024 distribution, same 0.7986).
> **(V3-6) Tier-A starvation (fork: add prompt-cloud arm).** Every trajectory feature is Tier-B and
> arm 1 can't cross models → the deployable pre-flight surface had ZERO portable candidates. New
> **arm 2A: prompt-token-cloud features** (covariance/profile over PROMPT tokens at 2/3 depth) —
> pre-generation, fixed-dim, immune to generation-length confound by construction; needs a cheap
> prefill-only re-extraction pass.
> **(V3-7) T3–T5 had no incumbent** (DoM can't enter) → "≥ incumbent" was undefined exactly where
> the claim lives. T3–T5 adjudication = vs the **length+logprob free baseline** + ≥0.70 absolute.
> **(V3-8) Phase-2/3 extraction over-specified.** Full all-layer trajectories for Phi-3 = tens of
> GB; the layer-profile family needs only **per-layer token-means `(L, H)`** + scalars — ~1000×
> smaller, computable online. That is now the extraction default.
> Literature (graph-checked): INSIDE/EigenScore 2402.03744 and CoE 2410.13640 in graph; **SIVR
> 2604.15741 (token-wise layer-wise variance, ACL 2026) is in link-forge but NOT in the
> research-graph — triage before any novelty claim on dispersion features.**

> **v3 changelog (second fresh-eyes mythos audit of v2, 2026-06-10).** The v2 audit fixed the
> framing but missed a confound that would have manufactured a false WIN. Six v2 findings, grounded
> in `eigval_cache.npz`:
> **(V2-1, severe) Length confound.** The cov-spectrum cloud is built over the *token axis* —
> `states[19]` is `(T_i, 1536)`, covariance over T_i points (`recompute_pc1_resid_cov_spectrum.py:106,71-77`),
> T_i ranging 123→1024. The top-20 log-eigvals are ~collinear with generation length
> (**corr −0.78**), and generation length *alone* scores **AUROC 0.7986 — above cov-spectrum
> (0.7928) and the incumbent (0.7731).** The "portable geometric signal" is largely a token-counter:
> (a) length is the *least* portable feature (tokenizer/verbosity-specific) → poisons the cross-model
> bet; (b) it requires the generated trajectory → cov-spectrum is **Tier-B (post-gen)**, cannot feed
> the pre-flight F-8 gate; (c) it's an FE19-style false WIN (beats DoM but loses to a free counter).
> Fix: a **length-only baseline + length-residualized form are mandatory gates ALL trajectory-cloud
> features (arm 2 AND arm 3) must clear** before any transfer claim; cov-spectrum reclassified Tier-B.
> **(V2-2) Only T5 is a true holdout.** The winning arm/feature-form is *selected* on T1–T4 numbers,
> so reporting those tiers as transfer evidence is selection-on-the-eval-set. T1–T4 are now **model-
> development tiers**; T5 (held-out families) is the **sole confirmatory tier**.
> **(V2-3) Phase-4 power ghost.** "Recover the gap by k≤32" is unmeasurable on the worst cell (n=62 →
> ~30 left after k=32 → SE≈0.10). Recalibration recovery now measured **pooled across tier cells / on
> the large cells (BBH 250, Qwen-7B)**, with the scoring n stated.
> **(V2-4) "Two families" is ~1.5.** SmolLM2 is Llama-architecture (Llama-3.2 is in the selection
> pool); honest relabel: **one near-family (SmolLM2≈Llama) + one far-family (Gemma)** holdout.
> **(V2-5) Paired DeLong needs a frozen fold map.** Incumbent and challenger OOF use different
> partitions; Phase 0 now mandates **one frozen fold assignment scored by every method**.
> **(V2-6) Arm-3 relabel.** GRIDE/PHD give a scalar → a 1-bit (sign) fit on labels; "label-free
> *feature*, 1-bit-supervised *readout*," not "label-free readout."

> **v2 changelog (fresh-eyes mythos audit, 2026-06-10).** v1 carried four load-bearing cracks,
> now corrected: **(M1–M3)** the "geometric = label-free = overfit-immune" framing was false for the
> strongest geometric candidate — cov-spectrum (0.7928) is a *supervised* LogisticRegression on
> top-20 log-eigvals, anchored to a Qwen-specific prefill-PC1, with *scale-dependent* eigval
> magnitudes. The real portability axis is **fixed-dim feature (ports across hidden sizes) vs
> hidden-dim-bound**, and the arms are now split into genuinely-label-free scalars (GRIDE/PHD) vs
> supervised-on-geometric (cov-spectrum), with a **pre-registered scale-invariant** cov-spectrum.
> **(M4–M5)** panel-wide `min` compared methods over different cell sets (supervised can't produce
> cross-model cells) and was under-powered (worst LOCO cell n=62 → 1.96·SE≈0.14); replaced by
> **per-tier worst-cell + paired DeLong over the full cell vector + robust aggregate**, per-cell n
> tabulated. **(M7)** n=1 unseen model → **two held-out families** (SmolLM2 + Gemma). **(M11)** the
> per-domain recalibration recipe is now a **first-class Phase 4**, not a LOSE-clause consolation.
> Minor: pinned single DoM incumbent (M6), resolved fixed-layer↔H-D collision (M8), dropped the
> in-domain-motivates-transfer contradiction (M9) and the result-presupposing combiner line (M10).

## Context — why this, why now

topo-confidence's one deployable result is **F-8**: a hidden-state read of Qwen-2.5-1.5B
predicts whether it will answer a MATH problem correctly, and refuse-and-spend on that score
answers 50% of queries at **71.6%** vs 48.6% blind. The *quality of that product is exactly the
quality of the correctness-prediction signal.* What is dead and out of scope (v5 nuance): **steering**
(F-2 is DIAGNOSTIC, a thermometer not a thermostat — FE269, 2026-06-09) and **same-model K-routing**
(C_infer prefill-gating and C_exact verification both failed to beat the uniform-K hull — FE19,
2026-06-10). What is NOT dead (V4-1): **cross-model cascade** — FU2 shows the 1.5B→7B hybrid cascade
already beats the pre-gen frontier with an 11.2pp cost gap to oracle remaining; that gap is consumed
directly by readout improvements. So the frontier is **making the readout better and more general
(Phases 1–3) AND using it better at fixed readout quality (Phase 1B: selection, timing, guarantees).**

The infrastructure sweep surfaced the real opportunity. Nearly every number we quote is
**in-domain** (train + test on the same distribution, OOF folds): prefill-DoM 0.7731, prefill
ridge-LR 0.7844, final-token ridge-LR 0.8493, concat 0.8509, cov-spectrum 0.7928. The moment we go
cross-distribution it collapses: cross-domain MATH↔BBH ~0.72, leave-one-MATH-category-out worst
**0.603** (Number Theory). **That ~0.15–0.25 AUROC gap between in-domain and transfer is the
unexploited juice.** Squeezing the in-domain number harder *is* the overfitting the user wants to
avoid; closing the transfer gap is the "real solution that generalizes to other models and domains."

**Decisions taken this session:** explore pre- and post-generation signals **together** (report both
tiers per method, let the conclusions emerge); run the **full program** including generation on
**two genuinely unseen models** as the generalization floor.

**(M12, flagged not resolved.)** F-8 = selective prediction ("refuse half the questions"), which is an
odd *educational* product. A better transfer AUROC is necessary for any deployment surface, so the
experiment design is unaffected — but before shipping, re-examine the surface (abstain vs.
confidence-display vs. hint-routing) against the stated "for the sake of education" goal. Out of scope
for the bake-off; in scope for the eventual product framing.

## North-star metric & pre-registered decision rules

The success metric is **transfer AUROC across a (model × domain) panel**, never in-domain. v1 used
a single panel-wide `min`; that is statistically fragile (a downward-biased, high-variance order
statistic) **and ill-posed** — supervised methods cannot produce cross-model cells (dim mismatch),
so their `min` is taken over a smaller, easier cell set than a geometric method's. v2 adjudicates
**per transfer tier, like-for-like**, on the full paired cell vector.

**Transfer tiers** (a method is only ever compared to another method on the *same* tier's cells):

| Tier | Cells | Hardest? | Who can enter |
|---|---|---|---|
| T1 cross-category | LOCO leave-one-MATH-subject-out (7) | mild | all methods |
| T2 cross-domain | MATH↔BBH (same model) | medium | all methods |
| T3 cross-scale | Qwen-1.5B↔7B | medium | fixed-dim / geometric only |
| T4 cross-arch | Qwen↔Phi↔Llama | hard | fixed-dim / geometric only |
| T5 cross-model (held out) | →SmolLM2, →Gemma | hardest | fixed-dim / geometric only |

**Selection firewall (V2-2).** Method/feature-form selection (which arm, which scale-invariant form,
which layer) is done on **T1–T4 only — these are model-development tiers.** **T5 is the sole
confirmatory tier**, touched once with the already-frozen winner; its number is the only one
presented as clean cross-model generalization evidence. T1–T4 numbers are reported but flagged as
selection-contaminated (optimistically biased), never cited as the generalization proof.

**Free-baseline control & incremental gate (V2-1 revised by V3-2/V3-4).** Generation length alone
scores AUROC **0.7986** in-domain, and **length + mean_logprob = 0.834 (dry run, n=200)** — two free
scalars, no activations, model-agnostic, enter every tier. So the **pinned Tier-B free baseline =
length+logprob**, a first-class entry in every tier, and **every trajectory/activation feature must
clear the incremental gate:** the paired per-tier test of **(feature + length + logprob) vs
(length + logprob)** must show the feature *adds* signal. The naive logT-residualized AUROC is
reported alongside as a **decomposition** — the size of the feature's length-orthogonal channel —
but is NOT a kill criterion (residualizing on a label-correlated covariate destroys legitimately
shared variance; V3-2). The length-*shared* component of a feature ports iff the length→correctness
relation itself ports — which is measured directly, because **length-only and length+logprob run on
ALL tiers as candidates (V3-5)**: their transfer is an empirical question, not an assumed failure.
A feature that adds nothing over the free baseline is reported as "= free baseline" — a non-result.

Per tier, report: **(a) worst cell** with its bootstrap CI, **(b) robust aggregate** = mean
off-diagonal AUROC with a paired CI, **(c) the full per-cell vector with each cell's n** (so power
is visible, not hidden — see the panel table), **(d) the free baseline (length+logprob), the
incremental-gate paired test, and the length-orthogonal decomposition** alongside each feature
method. Adjudicate method-vs-incumbent
with a **paired test across the shared cells** (DeLong where the same test set is scored by both;
bootstrap-paired where not), not by eyeballing two independent minima. **All methods are scored on
one frozen fold/test partition per cell (V2-5)** so the paired test compares identical test items.

- **Overfit gap** (the alarm): `mean(in-domain OOF) − mean(transfer)` *within each tier*. Lower =
  more robust. A method that raises in-domain but widens the gap is *rejected* regardless of headline.
- **Two signal tiers reported for every method** (orthogonal to the transfer tiers above): Tier A =
  pre-generation only (prefill activations — the deployable pre-flight gate; in-domain ceiling ~0.79).
  Tier B = post-generation allowed (final-token / full-trajectory; in-domain ceiling ~0.85). Compare
  the two to learn how much edge is purchased by waiting for the generation. **Every trajectory /
  generation-derived feature (incl. length, logprob, layer-profiles over generated tokens) is Tier-B
  by construction.** Tier-B is still a real product — **generate-then-abstain** selective prediction
  (the answer is hidden when the gate fires; compute is spent but quality holds) — just not the
  pre-flight compute-saving gate. The only *portable Tier-A* candidates are the new **arm-2A
  prompt-token-cloud features (V3-6)**: profiles/covariance over the PROMPT tokens at 2/3 depth,
  pre-generation, immune to generation-length confound by construction (their confound is
  prompt-length, which is exogenous, shared across models, and gets its own baseline). Without
  arm 2A, the pre-flight surface's honest ceiling stays at per-model prefill ~0.79 (arm-1).

**Pinned incumbent** = **prefill-DoM, OOF 5-fold, the 0.7731 protocol** (`prefill_gated_compute/
phase2_prefill_dom.py::oof_dom_scores`, the CLAUDE.md headline — *not* the FE291 SEED=0 0.7679
anchor; reproduce *this exact* number through the harness before any comparison). Its T1 worst cell
is **0.603 (Number Theory, n_test=62: 27 pos / 35 neg)**; T1 in-domain 0.7834 → gap ≈0.18.

**Power note (load-bearing).** At n_test=62, Hanley–McNeil SE ≈ 0.07, so 1.96·SE ≈ 0.14 — a
single-cell "beat by 1.96·SE" bar is nearly unmeetable and noise-dominated. That is *why* v2 judges
on the **paired full-vector test + robust aggregate**, and where a tier's cells share structure,
**pools test items within the tier** to shrink the CI before computing worst/aggregate.

Pre-registered, per tier, before running any new method:

- **WIN:** the method's **per-tier robust aggregate ≥ the tier's reference (T1/T2: the pinned
  prefill-DoM incumbent; T3–T5: the length+logprob free baseline, V3-7), AND its paired full-vector
  test is significant (no worse on the worst cell beyond noise), AND the overfit gap is not wider,
  AND it clears the incremental gate — (feature+length+logprob) > (length+logprob) on the paired
  test (V3-2).** Genuine robustness, not in-domain inflation, a lucky single cell, or a free-signal
  artifact. WIN on T1–T4 is *development* evidence (selection-contaminated, V2-2), not the
  generalization proof.
- **STRETCH (the only clean generalization claim):** clears WIN on T2 **AND** posts a non-degenerate
  number on **T5** — ≥0.70 mean on **both** held-out families (with the far-family Gemma number as the
  load-bearing one, V2-4), clearing the incremental gate there, with zero held-out data in any
  fit/selection. T5 is touched once with the pre-frozen winner. **Separately: if the free baseline
  itself posts ≥0.70 on T5, that is a shippable deliverable in its own right (V3-4 fork: ship the
  simplest winner)** — a free, universal generate-then-abstain gate is model improvement, full stop.
- **LOSE:** no feature method clears its tier reference / the incremental gate anywhere → activation
  geometry adds nothing portable over free signals. Still the **modal, pre-acknowledged outcome**
  for the *geometry* arms (the dry run already pre-confirmed it for the token-cloud spectrum, V3-1).
  The deliverable then is the documented ceiling, **whatever the free baseline delivers on T2–T5**,
  **plus the Phase-4 recalibration recipe** — see Phase 4.

## The transfer panel

Rows = train distribution, cols = test distribution; fit on A, score on B, AUROC + 2000-boot 95% CI
**and n_test on every cell** (verified small: LOCO subjects 38–124; BBH subsets 250; so CIs are wide
— the metric section accounts for this).

**Layer policy (resolves the v1 fixed-layer ↔ H-D collision).** The *panel* fixes the operating
point at **~2/3 depth** per model (Qwen-1.5B L19/29, Qwen-7B ~L19/29, Phi-3 L21/33, Llama-3.2-1B
L11/17, held-out models their own 2/3 fraction) — itself a transferred assumption we keep fixed for
the headline so it is honest and cheap. H-D's layer/position *search* is a **separate, train-only
sub-study**: the (layer, position) is selected on the train distribution via nested CV and the
chosen point is applied blind to the test cells — never picking the transfer-optimal layer on the
test model. Cache status (from the infra sweep):

| Cell | Activations | Cache-ready? |
|---|---|---|
| Qwen-1.5B MATH-500 — full `(29,T,1536)` | `pathway8_layerwise/data/math500/` | ✅ all-layer all-position |
| Qwen-1.5B BBH ×3 (tracking / logical-deduction / web-of-lies) | `pathway8_layerwise/data/bbh/<subset>/` | ✅ all-layer all-position |
| Qwen-7B MATH-500 — full `(29,T,3584)` (cross-**scale**) | `pathway11_h100/data/math500_7b/` | ✅ all-layer all-position |
| Qwen-1.5B MATH by 7 categories (LOCO within-domain transfer) | derived from MATH cache | ✅ |
| Phi-3-mini MATH (cross-**arch**) | `pathway11_h100/exp1_cross_model/data/phi3mini/` | ⚠️ prefill+final+7mid only |
| Llama-3.2-1B MATH (cross-**arch**) | `pathway11_h100/exp1_cross_model/data/llama32_1b/` | ⚠️ prefill+final+7mid only |
| Phi-3 / Llama on BBH (cross-arch × cross-domain) | — | ❌ Phase 2 re-extraction |
| **Held-out near-family (SmolLM2-1.7B, Llama-arch)** MATH + ≥1 BBH subset | — | ❌ Phase 3 generation |
| **Held-out far-family (Gemma-2-2B-it)** MATH + ≥1 BBH subset | — | ❌ Phase 3 generation |

## The central structural insight: what actually ports across hidden spaces

v1 framed this as "supervised (bad, overfits) vs label-free geometric (good, immune)." Reading the
artifacts, that dichotomy is wrong. The real axis is **fixed-dimensional feature vs
hidden-dim-bound**, and there are *three* arms, not two:

1. **Hidden-dim-bound supervised** (DoM vector, ridge-LR weights, concat). The weight vector lives in
   one model's hidden space — a 1536-d Qwen vector cannot be applied to Phi's 3072-d space. **Can do
   T1/T2 (cross-category, cross-domain) only**; cross-model requires retraining per model. Honest
   annotation, not a defect.

2. **Supervised-on-fixed-dim-geometric-features — re-centered on the DEPTH axis (V3-1/V3-3).**
   The v3 core (token-cloud cov-spectrum) is demoted: the dry run showed spectrum+length 0.761 <
   length 0.786 (negative incremental value), residualized 0.554, and the length coupling survives
   fixed-m subsampling (corr −0.58 at m=100) — the token-axis geometry genuinely co-varies with
   length and carries nothing portable beyond it. It gets **one Phase-1 closure row** to formally
   close the H-A null at full n, nothing more. The new arm-2 core is the **layer-profile family**:
   per-layer aggregates over the generation — **CoE mag/ang profiles**
   (`pathway8_layerwise/coe_features.py`, exp2 holdout **0.811**, dry-run 0.825 and **+4pp over
   length**), **D2H dispersion** (SIVR-like token-wise layer variance), **attn-entropy** (cached
   `d2h_attn_entropy`). Portability constructions, pre-registered:
   - **Depth grid:** per-layer profiles are resampled to a **fixed depth-fraction grid** (e.g. 16
     points from layer-fraction 0→1), so 29-, 33-, and 17-layer models map to the same feature
     space — the depth analog of "fixed-K eigvals."
   - **Scale (M2):** angle features are scale-invariant by construction; magnitude profiles are
     normalized (per-model trace/total-displacement normalization, chosen on train data only).
   - **Incremental gate (V3-2):** every layer-profile form must add over length+logprob on the
     paired test — the dry run says it does in-domain (+4pp); whether the *added* part transfers is
     the program's central question.

2A. **Prompt-token-cloud features (the only portable Tier-A arm, V3-6).** Same constructions
   (spectrum / profile / dispersion) but over the **prompt tokens** before any generation: fixed-dim,
   scale-normalized, **pre-flight by construction** and immune to generation-length confound. Its
   own mandatory baseline is **prompt-length** (exogenous difficulty prior, currently unmeasured
   anywhere — V3-5). Requires a cheap prefill-only re-extraction pass (no generation; minutes–hours
   on the 2060 across the panel). The PC1-anchor rule (M3) carries over: any per-model PC1/whitening
   is derived unsupervised on that model's own prompts, logged, never tuned on outcomes.

3. **Genuinely label-free *features*, 1-bit-supervised readouts (V2-6)** — **GRIDE intrinsic-dim**
   (`P11-FE238`) and **persistent-homology dim** (`P11-FE55`). A single dimensionless number per
   cloud; the *feature* uses no labels, but turning the scalar into an AUROC needs a fitted sign
   (1 bit on labels) — so this is "label-free feature, 1-bit-supervised readout," not a fully
   label-free readout. Nearly DOF-free, directly comparable across architectures, the **strongest
   overfit-immune control** — but the program's *weakest* signals, so the bet does not rest on them;
   they isolate "is transfer coming from label-free geometry, or from a fitted-but-fixed-dim
   classifier?" **The same incremental gate (V3-2) applies** — intrinsic-dim estimates drift with
   sample count, so each must add over length+logprob on the paired test before trusting.

**The core bet, restated honestly (v4):** the free **length+logprob** baseline is portable by
construction and sets the bar everywhere (V3-4). The geometry bet is now specifically: **depth-axis
profile features (arm 2) and prompt-cloud features (arm 2A) carry correctness signal that (a) adds
over the free baseline and (b) ports across models via the fixed depth-grid / fixed-K
constructions.** Supervised hidden-dim-bound readouts cannot enter T3–T5; the token-cloud spectrum
is already refuted on (a) by the dry run. If the geometry arms fail their gates, the program's
deliverable is whatever the free baseline achieves on T2–T5 plus Phase 4 — explicitly acceptable
(ship the simplest winner). In-domain superiority is never cited as evidence for transfer (M9).

## Phase 0 — shared harness (the durable infrastructure)

Every existing script re-implements NPZ loading and AUROC-with-CI inline. Build one module under
new dir **`pathway11_h100/generalization_edge/`** so the bake-off is uniform and the spec is
reproducible:

- `activation_loader.py` — `load(model, dataset, layers, positions) -> (X, y, meta)`; handles the
  full-trajectory caches and the partial Phi/Llama caches behind one interface; knows each model's
  2/3-depth layer.
- `probes.py` — register each candidate as `fit(X_tr, y_tr) -> state` / `score(state, X_te) -> s`.
  Wrap the existing recipes: DoM (`prefill_gated_compute/phase2_prefill_dom.py::oof_dom_scores`),
  ridge-LR (`regularized_concat/recompute_fe421.py`), **layer-profile family
  (`pathway8_layerwise/coe_features.py::compute_coe_single`, D2H dispersion, attn-entropy) + a
  `depth_grid(profile, n=16)` resampler (V3-3)**, cov-spectrum (closure row), per-layer sweep
  (`verify_route/fit_panl_probe.py`). **Plus: `free_baseline` probe (length+logprob), `length_only`,
  `prompt_length`, and an `incremental_gate(feature, free)` paired-test wrapper (V3-2)** every
  feature probe composes with (residualized AUROC computed alongside as the decomposition).
- `transfer.py` — `run_panel(method, panel) -> {cell: auroc±ci}`; computes worst-cell, mean,
  and gap, **and runs the free baseline + incremental gate + decomposition for every feature
  method** so the V3-2 gate is automatic, not optional.
- `metrics.py` — `auroc_with_ci(scores, y, n_boot=2000, seed)`; nested-CV helper; **DeLong paired
  test; and a `frozen_folds(cell, seed)` helper that returns one fixed StratifiedKFold partition per
  cell, reused by every method (V2-5)** so paired DeLong always compares identical test items.

## Phase 1 — CPU bake-off on cache-ready cells (zero GPU)

Run every method through `run_panel` on the cache-ready cells (Qwen-1.5B MATH, BBH×3, Qwen-7B,
LOCO-7). Report Tier-A and Tier-B numbers, worst-cell, mean, gap. Candidates, grouped, mapped to
their filed FE nodes:

**Baselines (the things to beat):** prefill-DoM (pinned incumbent, T1/T2 reference),
**length+logprob free baseline (dry-run 0.834 — the pinned Tier-B reference for all tiers, V3-4)**,
length-only (0.7986), mean-logprob (`P10-FE23`, 0.6721), **prompt-length (V3-5, new — the
Tier-A/arm-2A exogenous-difficulty prior, currently unmeasured)**. The free baselines run on **every
tier** as candidates — if length+logprob ports to T5 at ≥0.70 it is itself a shippable deliverable.

**Arm 1 — hidden-dim-bound supervised (T1/T2 only):** prefill-DoM (pinned incumbent), final-token
DoM, concat ridge-LR (`P11-FE254`/FE421). These get the honest "no T3–T5 entry, retrain per model"
annotation.

**Arm 2 — depth-axis layer-profile family (the portability bet; enters all tiers, Tier-B):**
CoE mag/ang profiles (exp2 0.811 / dry-run 0.825, +4pp over length), D2H dispersion, attn-entropy —
each in (i) raw per-layer form (in-domain parity with exp2), (ii) the **depth-grid resampled +
scale-normalized** form (the only form eligible for T3–T5), (iii) with the incremental gate +
decomposition reported. **cov-spectrum gets ONE closure row** — full-n confirmation of the dry-run
refutation (spectrum+length vs length paired test); its prior recompute-parity check (cached MATH
0.7928 ±0.005) still applies to that row.

**Arm 2A — prompt-token-cloud features (the only portable Tier-A arm; needs the Phase-2 prefill
pass for non-Qwen cells, but Qwen MATH/BBH prompt clouds may be re-derivable immediately if prompt
tokens can be cheaply re-prefilled):** prompt-cloud spectrum/profile/dispersion at 2/3 depth,
gated against **prompt-length**.

**Arm 3 — label-free features / 1-bit readouts (overfit-immune control; enters all tiers):**
**GRIDE intrinsic-dim** per layer (`P11-FE238`), **persistent-homology dim** per cloud (`P11-FE55`).
Weakest signals; their job is to isolate label-free-geometry from fitted-fixed-dim. **Subject to the
same incremental gate (V3-2)** — intrinsic-dim drifts with sample count, so each must add over
length+logprob on the paired test.

**Richer supervised readouts (test the gap-inflation hypothesis, arm 1):** rStar-Math Bradley-Terry
process probe (`P11-FE357`), Linear-AcT closed-form (`P11-FE339`), LEAT post-`\boxed` token
position probe (`P11-FE301`).

**Systematic search (H-D sub-study):** Arditi position×layer sweep selected on **train only** for the
best *transfer* point, applied blind to test cells (`P11-FE268`) — run on Qwen full-trajectory caches.

**Combiners:** multi-signal stack (DoM + layer-profile + free baseline) and multi-layer concat — fit on
train dist only, scored on transfer. **Pre-registered prediction:** combiners maximize in-domain but
widen the overfit gap (more free parameters → more distribution-specific fit); **decision criterion:**
a combiner is kept only if its per-tier robust aggregate beats its best single component on the paired
test. (No presupposing the result — the harness adjudicates.)

**Overfit-diagnostic control:** task-recognition vs task-learning (`P11-FE70`) — does the signal
survive rephrased / format-stripped MATH? A large drop means the signal is partly format-memorized.
(Needs small generation; fold into Phase 2.)

## Phase 1B — applied quick wins (selection, timing, guarantees; CPU-only, parallel to Phase 1)

Four pre-registered experiments on existing caches; none requires generation; none blocks or is
blocked by Phase 1. They attack the **use** of the readout while Phases 1–3 attack its generality.
Order by dependency: 1B.4 → 1B.1 / 1B.2 → 1B.3.

**1B.1 — Probe-reranking on the K=8 cache (executes H-40 / P11-FE41; ~1h CPU).**
- Data: `pathway11_h100/data/k8_selfconsistency/problem_*.npz` (500 problems: `L19_samples (8,1536)
  float16`, `texts (8,)`, `correct (8,)`). **Step 0: confirm from the extraction script that
  `L19_samples` is the final-token state of each sample**; if it is another position, refit the
  probe at that position on the main cache first (still trivial).
- Arms: (a) **probe-argmax** — answer of the highest-scoring sample; (b) **probe-weighted majority**
  — votes weighted by probe score; (c) **confidence-fallback** — keep the K=1 greedy answer when its
  own probe score clears a threshold (D-bucket protection; F-7 showed K=8 majority *hurts* the
  D-bucket).
- Comparators at **identical compute**: K=8 plain majority (**55.0%**); context rows K=1 greedy
  (48.6%) and the **oracle best-of-8 ceiling** (any-correct, computable from the cache — report it;
  it bounds all selection methods).
- Adjudication: paired **McNemar** vs plain majority, n=500, frozen folds for the probe fit;
  stratified by F-7 bucket. **WIN = any arm beats plain majority significantly without a D-bucket
  regression.** This is *selection, not routing* — fixed compute, so the FE19 hull critique
  structurally cannot apply.

**1B.2 — Earliest-decision curve (prefix-truncated probes; ~hours CPU).**
- On the full Qwen-1.5B `(29,T,1536)` MATH cache (Qwen-7B cache for the cross-scale check): for
  each prefix length m ∈ {25, 50, 100, 200, 400, full}, compute each readout **restricted to the
  first m generated tokens** — DoM/probe on the token-mean at L19, the free-baseline-so-far
  (m itself is constant at fixed m, so this is mean_logprob≤m + still-going indicator), the arm-2
  layer-profile over the prefix — and its frozen-fold OOF AUROC.
- Deliverable: **AUROC(m) curve per readout** — when does 0.77 (prefill) become 0.85 (full gen)?
  F-1's mid-gen PR peak (30–70 tokens) predicts "early."
- Pre-registered decision: **if some readout reaches ≥0.82 by m ≤ 128**, register and simulate
  **abort-and-escalate**: draft on 1.5B, stop at m, escalate flagged problems to 7B; cost model =
  (m/1024)·draft + escalation; adjudicate vs the **random-mix hull** AND vs the **FU2 hybrid
  frontier** (keep-46% @ 0.632× at the 90% target), paired. This is the bridge between Tier-A and
  Tier-B: how much of the post-gen edge is available at a fraction of the draft cost.

**1B.3 — Conformal risk control on F-8 (the guarantee layer; ~1 day CPU).**
- Split-conformal / Learn-then-Test on existing OOF scores (incumbent prefill-DoM, free baseline,
  and the Phase-1 winner when available): thresholds **τ(ε, δ)** such that answered-set error ≤ ε
  with probability ≥ 1−δ; validate empirical coverage over ≥1000 repeated calibration/test splits.
- Deliverable: (ε, δ, τ, achieved coverage, answered-set accuracy, coverage fraction) table at
  ε ∈ {0.1, 0.2, 0.3}; the same machinery applied to the cascade escalation decision (1B.4 table).
- **Graph triage first (V4-4):** 2205.14334 / 2402.10978 / 2406.15927 are already in the
  research-graph — read them before any novelty claim.
- Phase-4 link: the recalibration head re-fit (k ∈ {8..64}) gets a conformal variant — does a k=32
  target-domain calibration set yield *valid* coverage there? (Guarantee validity under shift is
  the conformal restatement of the program's overfit-gap metric.)

**1B.4 — 1.5B×7B overlap table + hull-honest cascade re-adjudication (~30 min CPU; do first).**
- Build `[problem_id, correct_1.5B_K1, correct_1.5B_K8maj, correct_7B]` for MATH-500 from existing
  caches/results (the table does not exist anywhere — V4-5).
- Re-score G1/FU2 against the **random-mix hull** (the FE19 comparator) with paired SEs at matched
  cost; report which problem categories 7B actually rescues.
- Substrate for 1B.2's escalation sim and Phase 4's cascade calibration.

## Phase 2 — cross-architecture, winners only (small local 2060)

> **v6 RE-SCOPE (V5-1/V5-3).** Phase 1 produced **no portable geometry winner** — the only Phase-1
> "winner" carried into the cross-model cells is the **free length+logprob baseline** (needs no
> activations), and the layer-profile depth-grid was REFUTED (in-domain p=0.137, T3 0.627). So
> **step 1 below (re-extract Phi-3 / Llama-3.2 layer-profile trajectories) is DROPPED** — there is
> nothing portable to carry. Phase 2 now reduces to **the prefill-only pass (step 2)** for the one
> open geometry question (arm-2A prompt-token-cloud, V5-3) plus the free length/logprob columns,
> and the rephrased-MATH control (step 4). The expensive trajectory extraction does not run.

Take the Phase-1 winners (the **free length+logprob baseline** + the supervised incumbent for the
T1/T2 cells; **no portable layer-profile geometry survived** — V5-1) and:

1. ~~Re-extract Phi-3 and Llama-3.2 on MATH with the lean extraction default (per-layer token-means
   `(L, H)` + per-layer norms + scalars).~~ **DROPPED (V5-1)** — no portable geometry feature
   survived Phase 1, so there is nothing to extract these trajectories *for*. (Kept here struck
   through so the audit trail shows it was a deliberate cut, not an omission.)
2. **Prefill-only pass (arm 2A, V3-6/V5-3 — now the SOLE geometry probe):** capture prompt-token
   states at 2/3 depth for ALL panel models on MATH + BBH (no generation — cheap; minutes–hours on
   the 2060) + tokenize-only prompt-length for every cell. This is the last untested portable Tier-A
   candidate and the only remaining shot at a pre-flight gate that crosses models — gate it against
   **prompt-length** (H-E). It was logged `NOT_IMPLEMENTED` in `phase1_bakeoff.py` because Phase 1
   ran on cached generation states only; this pass is what unblocks it.
3. Generate + extract Phi-3 / Llama on one BBH subset (cross-arch × cross-domain cells) — needed for
   the **free baseline's** T4 cells (length/logprob require generation), not for geometry.
4. Run the rephrased-MATH overfit control (`P11-FE70`).

Supervised directions get the honest "must retrain per model" annotation. Reuse
`causal_dom/common.py` (`load_model` bf16/sdpa, `batched_generate`, `check_correct`) for generation;
mirror the `exp1_cross_model` extraction.

## Phase 3 — the unseen-model proof: TWO held-out families (local 2060)

> **v6 REFRAME (V5-2).** The T5 holdout is now a test of **the free length+logprob readout**, not
> geometry. Phase 1 made length+logprob the generalizing winner (T1 0.845, T3 0.865) and it needs
> **only generation** on the held-outs — no activation extraction. The load-bearing T5 question is:
> **does length+logprob hit ≥0.70 on SmolLM2-1.7B and Gemma-2-2B-it?** If yes, ship it — a free,
> universal, generate-then-abstain F-8 gate is the deliverable (V3-4 fork, now the headline). The
> arm-2A prompt-cloud features (if Phase 2 finds them portable) are a *secondary* T5 probe, gated
> against prompt-length; geometry is no longer the thing on trial here.

n=1 unseen model is an existence check, not a generalization claim. Generate + extract MATH-500 and
≥1 BBH subset on **two** models that played **zero role** in method selection — **one near-family,
one far-family (V2-4 honest relabel):**
- **SmolLM2-1.7B-Instruct** — **near-family holdout**: it is a *Llama-architecture* model, and
  Llama-3.2-1B is in the selection pool (Phase 2). So this tests "does it port to an unseen *model*
  within a *seen architecture family*" — a real but weaker holdout than v2's prose implied. (~3.4GB bf16.)
- **Gemma-2-2B-it** — **far-family holdout**: genuinely distinct mechanism (alternating local/global
  attention, logit soft-capping, own tokenizer); ~5GB bf16, fits 8GB. This is the only *novel-arch*
  cross-model test → the truly-novel-architecture holdout is effectively **n=1 (Gemma)**.

Apply the winning arm-2/arm-3 method with **all fitting and selection done on Qwen/Phi/Llama only**.
The one allowed pass on a held-out model is the unsupervised, label-free PC1/normalization
derivation on its own MATH prompts (logged; see arm 2A / M3 rule) plus scoring. T5 result = the **paired** number across
both families, read with the near/far asymmetry in mind; STRETCH requires both ≥0.70, with the Gemma
(far-family) number weighted as the real generalization signal. If a clean non-Llama small model that
fits 8GB surfaces (OLMo-2-1B, Falcon3-1B), prefer swapping it in for SmolLM2 to make both holdouts
far-family.

## Phase 4 — per-domain recalibration recipe (first-class deliverable)

> **v6 ELEVATION (V5-5).** 1B.3 CONFIRMED **H-H**: conformal/LTT certificates are valid in-domain
> but **collapse to 0 coverage under the MATH→BBH shift** (`results/1b3_conformal.json`). That
> turns this phase from a LOSE-clause consolation into **the only path to a deployable cross-domain
> guarantee** — target-domain recalibration is *required*, not optional. The "best surviving
> extractor" to freeze and recalibrate is now pinned to the **free length+logprob readout** (it
> generalizes best and is model-agnostic), and the conformal certificate is the headline output, not
> an aside. Additionally Phase 4 calibrates the **1.5B→7B hybrid cascade** (V5-4) — the one routing
> surface that beat its honest baseline (+3.38pp, 1B.4) — re-fitting its escalation threshold per
> target domain.

The modal honest outcome is that no fixed readout transfers cleanly (LOSE on the strict tiers) — and
Phase 1 confirmed it for geometry. That does **not** end the program — it makes **cheap
recalibration** the product. Pre-register, measured with the same per-tier paired machinery:

- **Question:** how little target-domain labeled data re-fits a deployable F-8 gate? Take the
  best surviving extractor (arm 2/2A layer-profile, or the free baseline if geometry fails its
  gates; frozen, fit on source), and on each target cell **re-fit only the light head** — threshold
  + scaler (and, if needed, a 1-D Platt/isotonic calibration) — on `k ∈ {8, 16, 32, 64}` labeled
  target examples, scoring on the held-out remainder.
- **Power (V2-3, mandatory).** Spending k on the head shrinks the scoring set, so the recovery curve
  must be measured **only where it is measurable**: pool across a tier's cells, or use the **large
  cells (BBH subsets n=250, Qwen-7B MATH n=500)**, never the n=62 worst LOCO cell where k=32 leaves
  ~30 to score (SE≈0.10). State the scoring n at every k; a recovery claim on a cell with scoring
  n<100 is reported as underpowered, not as a win.
- **Curve:** target-cell AUROC and selective-prediction accuracy@50%-coverage vs k, with bootstrap
  CIs, averaged over resampled support sets. **Win = the recipe recovers most of the in-domain gap by
  k≤32 on the adequately-powered cells** (a few dozen labels is a realistic deployment budget; report
  the exact recovered fraction and the scoring n).
- **Guarantee form (V4-4, from 1B.3):** alongside the AUROC / selective-accuracy curve, report the
  conformal certificate achievable with the same k labels — "k=32 target labels buy a valid
  (ε=0.2, δ=0.1) answered-set guarantee at X% coverage" is the deployable claim format; invalid
  coverage at small k is itself a reportable limit.
- **Why first-class:** "freeze a fixed-dim geometric extractor, re-fit a 32-example head per new
  domain/model" is an actually-shippable F-8 hardening result regardless of whether zero-shot transfer
  works — it calibrates the "real solution" expectation honestly up front.

## Pre-registered hypotheses (state before running, to avoid HARKing)

> **v6 resolution scoreboard (EXP-81, 2026-06-11).** H-A pre-confirmed; **H-C REFUTED**, **H-F
> REFUTED**, **H-G REFUTED**, **H-H CONFIRMED**; **H-40/FE41 EXECUTED** (1B.1). H-A′ partially
> resolved (raw form adds in-domain, portable form does not — folded into H-C). Still OPEN: H-B
> (not adjudicated as a transfer claim), H-D (layer/position sub-study not run), **H-E (arm-2A
> prompt-cloud — the one live geometry hypothesis, needs the Phase-2 prefill pass)**. Per-hypothesis
> verdicts inline below.

- **H-A (token-cloud null — pre-confirmed by the dry run, V3-1, close at full n):** the cov-spectrum's
  edge over DoM is entirely the length confound. Dry-run evidence: spectrum+length 0.761 <
  length-only 0.786; residualized 0.554; corr −0.58 survives fixed-m=100 subsampling. Phase 1 runs
  one closure row at n=500 to make this formal. **→ PRE-CONFIRMED; closure row logged.**
- **H-A′ (the new arm-2 bet):** a **depth-grid, scale-normalized layer-profile** readout adds over
  the length+logprob free baseline on the paired test (dry-run in-domain: +4pp / CoE 0.825 vs free
  0.834 — note the free baseline is *higher*; the gate is incremental value in combination, and
  transfer of the added part), and has a smaller per-tier gap than arm-1 DoM. **→ SPLIT VERDICT:**
  the **raw** CoE layer-profile DOES add in-domain (+0.0175, p=0.018, OOF 0.854) but is
  hidden-dim-bound; the **portable depth-grid** form adds neither in-domain (+0.0081, p=0.137) nor on
  transfer (T2 0.588, T3 0.627). The added part does **not** port — refuted as a *portable* win.
- **H-B:** richer supervised readouts (concat ridge-LR 0.85) buy in-domain at the cost of transfer —
  higher gap, no robust-aggregate gain on the paired test. **→ OPEN (in-domain concat 0.8509
  reproduced; transfer-gap claim not formally adjudicated — supervised arms are T1/T2-only).**
- **H-C:** at least one arm-2/2A feature delivers cross-**model** T5 transfer (≥0.70 both families)
  **that adds over the free baseline there** — something no arm-1 readout can produce structurally.
  Arm-3 scalars are the control: if they also clear T5, the transfer is label-free geometry; if only
  arm 2 does, it's the fitted fixed-dim head. *(Competing null, now front-and-center: the free
  baseline ports as well as anything — both Qwen and the held-outs share the difficulty↔length↔
  logprob relation; if so, ship the free gate (V3-4) and report geometry as "= free baseline".)*
  **→ REFUTED for arm-2 (depth-grid transfers ≤0.63, below the free baseline everywhere). The
  competing null WON: the free baseline ports as well as anything. arm-2A (H-E) is the only
  remaining way H-C could be salvaged; geometry otherwise reports "= or < free baseline."**
- **H-D:** a (layer, position) **selected on the train distribution** transfers better than fixed
  2/3-depth prefill — tested in the train-only sub-study, applied blind to test cells (never the
  test-optimal layer). **→ OPEN (sub-study not run in EXP-81).**
- **H-E (Tier-A, arm 2A):** prompt-cloud features add over the prompt-length prior and approach the
  per-model prefill ceiling (~0.79) while remaining portable — the only path to a deployable
  pre-flight gate that crosses models. **→ OPEN — the single live geometry hypothesis; needs the
  Phase-2 prefill-only pass (V5-3). It was `NOT_IMPLEMENTED` in `phase1_bakeoff.py`.**
- **H-F (selection, 1B.1):** probe-reranking beats K=8 plain majority (55.0%) at identical compute
  on the paired McNemar test, without a D-bucket regression — the probe carries *cross-sibling
  comparative* signal that unweighted voting discards. **→ REFUTED:** no arm beats K=8 majority
  (0.554); probe-argmax 0.480 is −7.4pp WORSE (McNemar p<0.001); weighted-maj 0.538 and
  confidence-fallback 0.552 both n.s. (`results/1b1_reranking.json`). `L19_samples` confirmed
  final-token. (H-40/FE41 EXECUTED.)
- **H-G (timing, 1B.2):** ≥80% of the prefill→full-gen AUROC gain (0.77→0.85) is available by
  m=128 generated tokens — the decision signal arrives early (F-1's mid-gen PR peak predicts this)
  — and the resulting abort-and-escalate policy beats the FU2 hybrid frontier at matched accuracy.
  **→ REFUTED:** L19 mean-pool DoM stays *below* the prefill gate until ~m200 (m25 0.704, m100
  0.758, m200 0.783, m400 0.809, full 0.819); the ≥0.82-by-m≤128 trigger never fired;
  abort-escalate costs 0.96× for +3.7pp (useless). Skip early-abort drafting
  (`results/1b2_earliest.json`).
- **H-H (guarantees, 1B.3):** split-conformal thresholds on OOF scores achieve nominal coverage on
  held-out splits in-domain; coverage degrades on transfer cells in proportion to the overfit gap —
  linking guarantee validity to the program's central metric. **→ CONFIRMED:** valid in-domain
  (δ=0.1, ε=0.2 → free_baseline 0.855 validity @ 28% cov / 85% acc) but coverage collapses to **0**
  under MATH→BBH shift (infeasible at ε=0.1/0.2). The guarantee is intrinsically domain-local →
  Phase 4 recalibration is load-bearing (`results/1b3_conformal.json`).

## Anti-overfitting guards (the methodological spine)

- Optimize **per-tier robust aggregate + worst cell on the paired test**, never in-domain; report the
  gap alongside every level. Never compare two independent panel-wide minima.
- **The free baseline (length+logprob) is the mandatory incremental gate (V3-2/V3-4).** Every
  feature must add over it on the paired (feature+free) vs free test; the logT-residualized AUROC is
  reported as the decomposition, never as the kill criterion. Free-baseline transfer is *measured* on
  every tier, not assumed away (V3-5). All generation-derived features are **Tier-B** — only arm 2A
  (prompt-cloud) can claim the pre-flight surface.
- **Selection firewall (V2-2).** T1–T4 are model-development tiers (selection-contaminated); **T5 is
  the sole confirmatory tier**, touched once. Never present a T1–T4 number as the generalization proof.
- **One frozen fold/test partition per cell (V2-5)**, reused by every method, so the paired DeLong
  test always compares identical test items.
- **Test distribution touched once.** All hyperparameters (direction, probe C, layer, K-eigvals,
  scale-invariant feature form, thresholds) selected on the train distribution via nested CV. The
  *only* allowed pass on held-out models is the unsupervised label-free PC1 derivation (arm 2 / M3
  rule), logged.
- **Arm 3 (GRIDE/PHD) is the genuine overfit-immune control** — if arm-1/arm-2 win in-domain but lose
  on transfer while arm 3 holds, that gap *is* the finding. cov-spectrum is **not** a label-free
  control (it is supervised); do not present it as one.
- **Both held-out families (SmolLM2 + Gemma) are reserved** end-to-end; no held-out data in any
  label-using fit or selection step.
- Bootstrap CIs (n=2000) on every cell, with **n_test printed per cell**; "wins" require the paired
  per-tier test to clear noise, not a single lucky cell (worst LOCO cell n=62 → 1.96·SE≈0.14).

## Reuse map (do not reinvent)

| Need | Reuse |
|---|---|
| OOF DoM scorer | `prefill_gated_compute/phase2_prefill_dom.py::oof_dom_scores` |
| Direction + transfer + bootstrap CI | `stage5_bbh_dom_transfer.py` (`dom_direction`, `score_with_ci`) |
| Ridge-LR concat | `regularized_concat/recompute_fe421.py` |
| **Layer-profile features (arm-2 core): CoE-60 mag/ang (exp2 holdout 0.811), D2H dispersion, attn-entropy** | `pathway8_layerwise/coe_features.py`, `d2hscore_features.py`, `results/exp2_results.json` |
| cov-spectrum (closure row only — **refuted on the dry run: spectrum+length 0.761 < length 0.786, resid 0.554, corr −0.58 at fixed m=100**) | `cov_spectrum/recompute_pc1_resid_cov_spectrum.py` + `eigval_cache.npz` |
| Per-layer probe sweep | `verify_route/fit_panl_probe.py` |
| Generation / model load (Phase 2/3) | `causal_dom/common.py` |
| Fast prefill/final replay | `prefill_inversion/cache/m{15b,7b}_prefill.npz` |
| **K=8 per-sample L19 states + per-sample correctness (1B.1)** | `pathway11_h100/data/k8_selfconsistency/problem_*.npz` |
| **Cascade frontiers + oracle anchors (1B.4)** | `nocompute/results/{G1_cascade,FU2_cascade_combined,Q6_cascade_oracle}.json`, produced by `nocompute/run_experiments.py` |

## Verification

> **v6 status: all 15 anchors REPRODUCED EXACTLY in EXP-81** (`verify_anchors.py`). The checks below
> stand as the regression suite; one anchor was CORRECTED — see step 1.

1. **Harness sanity:** reproduce the pinned in-domain anchors through the new harness — prefill-DoM
   **0.7731** (±0.005, the pinned incumbent protocol, *not* 0.7679), CoE-60 exp2 holdout 0.811,
   concat ridge-LR **0.8509** (LogisticRegressionCV L2 on `[L19 prefill, final]`, NOT RidgeClassifier
   — `regularized_concat/recompute_fe421.py`). **CORRECTED ANCHOR (v6):** the v5-prose
   "cross-domain MATH→BBH 0.747" is **STALE** and matches no recipe — assert instead against the
   saved canonical `nocompute/results/C1_math_to_bbh.json` (pooled **mean-DoM = 0.4158**, which
   anti-transfers). The real transfer finding is **prefill-DoM ports to BBH at 0.796 while mean-DoM
   INVERTS to 0.416**. If any drifts, the loader/probe wrapper is wrong.
2. **Dry-run anchors (V3-1/V3-4, first-200-problems subset, must reproduce ±0.01):** length-only
   0.786, **length+logprob 0.834**, CoE-60 0.825 (and +4pp over length on the paired test),
   spectrum+length 0.761 (< length), residualized spectrum 0.554. These pin the harness to the
   audit's evidence before any new number is trusted.
3. **Transfer sanity:** LOCO Number-Theory worst cell reproduces ≈0.603 (n_test=62) for prefill-DoM.
4. **Depth-grid check (V3-3):** the depth-grid resampled layer-profile, fit on Qwen-1.5B (29
   layers), scores non-degenerately on Qwen-7B / Phi-3 / Llama (different layer counts) — proving
   the grid construction ports; the raw per-layer-index form should fail this by construction.
5. **H-A closure row (full n=500):** spectrum+length vs length paired test, expected NOT
   significant; report and close the token-cloud line formally.
5B. **Phase-1B anchors (V4-1/V4-2 — REPRODUCED in EXP-81):**
   K=1 greedy 0.486, **K=8 plain majority 0.554** (TRUE mode-vote over extracted `\boxed{}` answers
   via `k8_lib.py`, cluster correct iff any member correct — the ≥5-of-8 proxy gives a wrong 0.370),
   oracle best-of-8 0.704, all-7B 0.732, FU2 hybrid AUROC 0.852 and savings-gap 11.2pp, G1 pre-gen
   point (keep 5% → 0.724 @ 0.96×). 1B.1 position check: `L19_samples` confirmed final-token
   (stage3 `hidden_states[-1][19][0,-1,:]`).
6. **End-to-end:** per-tier table — method × tier × {in-domain, robust-aggregate, worst-cell+n,
   gap, paired-test p, incremental-over-free-baseline, length-orthogonal decomposition}; the free
   baseline's own row on every tier incl. T5; T5 shows both held-out families (near/far flagged);
   plus the Phase-4 recalibration curve with scoring n at each k.

## Bookkeeping after the verdict (mirror the FE19/FE269 close-out)

> **v6 status.** EXP-81 (Phase 0/1/1B) is DONE; the brief is written; the items below are the
> **remaining** close-out, all PENDING explicit user go (graph promotion + FINDINGS/STATE edits not
> yet run). Items already done are marked ✅.

- File a new umbrella `:FutureExperiment` for the program (and mark the constituent nodes —
  FE238/FE55/FE357/FE339/FE301/FE268/FE254/FE70, **plus FE41 — H-40 EXECUTED in 1B.1** —
  COMPLETED/SUBSUMED) via `research-graph/update_status.py`; regenerate `NEXT_EXPERIMENTS.md`.
  **PENDING.**
- ✅ **DONE — Triaged the conformal-adjacent graph entries** (2402.10978 conformal-factual-lm,
  2406.15927 Semantic Entropy Probes, 2205.14334 verbalized-uncertainty = already-rejected H-12 dup)
  before 1B.3; no novelty claim made (method prior cited).
- **SIVR (arXiv 2604.15741, token-wise layer-wise variance) — triage now MOOT for novelty** (the
  dispersion/geometry arm LOST), but still in link-forge only, NOT in the research-graph; triage if
  any dispersion claim is ever revived. INSIDE/EigenScore (2402.03744) already graphed.
- ✅ **DONE — Result brief written** at `research-graph/briefs/result-2026-06-11-P11-FE-EDGEGEN.md`
  (has the `## FE` / `## EXPERIMENT_LOG entry` / `## STATE.md last-experiment update` sections;
  fe_id `P11-FE-EDGEGEN`). **PENDING:** promote via `research-graph/promote_result.py --skip-git-check`.
- FINDINGS.md (**PENDING**): extend **F-2** (transfer ceiling + the supervised-vs-geometric
  portability result: geometry is hidden-dim-bound, portable depth-grid does not transfer) and
  **re-pin F-8** from prefill-DoM (0.7731) to **length+mean_logprob** (T1 0.845 vs 0.743 — the best
  generalizing gate, V5-5); add the hybrid-cascade routing finding (V5-4, +3.38pp).
- `validate_claims.py` (**PENDING**): add claims for the free-baseline worst-cell transfer
  (T1 0.735), the depth-grid refutation (T3 0.627), the cascade +3.38pp, and (when generated) the
  Gemma/SmolLM2 T5 numbers; run full suite.
- STATE.md, EXPERIMENT_LOG.md (EXP-81), memory companion (✅ `topo-confidence-spec-v4.md` slug +
  MEMORY.md pointer already say EXECUTED 2026-06-11). Commit on `max-depth-retriage-2026-04-28`;
  `git checkout --` any spurious `elapsed_seconds` regen diffs before committing.
- ✅ **DONE — spec written into the repo.** This file is durably saved as **SPEC v6** at
  `pathway11_h100/generalization_edge/SPEC_v6.md` (the v5 copy `SPEC_v5.md` remains alongside as the
  pre-execution record). The plan file `~/.claude/plans/reactive-waddling-toast.md` is the source.

## Compute & constraints

- **LOCAL ONLY** (2060 Super 8GB + CPU), per CLAUDE.md — NOT RunPod. Phase 1 is pure CPU on cached
  data. Phase 2/3 generation runs fit the 2060 (1.5–2.6B models, bf16). Mirrors the FE19/FE269
  local precedent.
- Goal = **model improvement (a generalizing readout), not papers.** Keep headline-figure resolution.
- All trigger papers (CAP 2502.06884, rStar-Math, GRIDE, Arditi, Linear-AcT, etc.) are confirmed in
  the research-graph, not training data.
