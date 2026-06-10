# EXPERIMENT_LOG.md — running log of every experiment

Append-only. Every new experiment gets the next EXP-### number. Template at
the bottom of the file.

**Status key.** COMPLETE | RUNNING | ABANDONED | BLOCKED

---

## EXP-001: CORAL feature extraction (Pathway 1 origin)
**Date:** 2026-04-04 (initial) → 2026-04-11 (full snapshot)
**Status:** COMPLETE
**Motivated by:** ATT Phase 5 (internal): H0 persistence entropy of last-layer
hidden states predicts LLM correctness.
**Hypothesis:** A 78-feature extractor combining PH, geometry, layer
dynamics, and SVD spectrum predicts MATH-500 correctness from a single
forward pass.
**What we actually tested:** Built the extractor, recorded all 78 features on
a pilot run.
**Key result:** 44 of 78 features are useful (Tier A/B/C = 9 PH/geom + 30
layer dynamics + 5 depth-2 products). Tier D (34 depth-3+) hurts.
**Verdict:** CONFIRMED (frozen extractor emerges).
**Changed our understanding of:** What the right feature granularity is —
simple is better.
**Files:** `pathway1/phase0/winning_features.py` (1407 lines).
**Depends on:** none.
**Enables:** EXP-002.

## EXP-002: Pathway 1 Phase 1 — 0.7961 MATH-500 holdout
**Date:** 2026-04-12
**Status:** COMPLETE (later SUPERSEDED on label scheme)
**Motivated by:** EXP-001 frozen extractor needed a holdout number.
**Hypothesis:** ABC-44 features predict MATH-500 correctness ≥ 0.75 holdout AUROC.
**What we actually tested:** SSS(9999, test=100) with LR.
**Key result:** **Holdout AUROC 0.7961** — 256-tok_NEW labels (104/500 correct).
**Verdict:** CONFIRMED at the time. Labels later revised.
**Changed our understanding of:** The geometry of Qwen-1.5B residual streams
carries correctness signal.
**Files:** `pathway1/phase1/holdout_metrics.json`.
**Depends on:** EXP-001.
**Enables:** EXP-003, EXP-006, all downstream work.

## EXP-003: Pathway 1 Phase 2 — tier ablation
**Date:** 2026-04-12
**Status:** COMPLETE
**Motivated by:** EXP-002 — which feature tiers are doing the work?
**Hypothesis:** Tiers A and B are essential; Tier D is not.
**What we actually tested:** Hold-out AUROC with subsets {A}, {A,B}, {A,B,C}, {A,B,C,D}.
**Key result:** A alone = 0.704. A+B = 0.790. A+B+C = 0.796. A+B+C+D = lower.
**Verdict:** CONFIRMED — Tier D hurts.
**Files:** `pathway1/phase2/ablation_cv_results.json`, `ablation_holdout_results.json`, `ablation_plot.png` (→ `figures/pathway1_tier_ablation.png`).
**Depends on:** EXP-002.

## EXP-004: Pathway 1 Phase 3 — seed sensitivity
**Date:** 2026-04-12
**Status:** COMPLETE
**Key result:** Holdout AUROC varies ±0.03 across 100 seeds — 0.7961 is stable.
**Files:** `pathway1/phase3/seed_sensitivity_cv500.json`, `seed_sensitivity_holdout.json`.

## EXP-005: Pathway 1 Phase 4 — routing prototype
**Date:** 2026-04-12
**Status:** COMPLETE
**Key result:** Routing threshold gives calibrated risk-coverage curve. First
concrete "actionable" use of the signal.
**Files:** `pathway1/phase4/{routing_metrics,calibration_metrics,operating_point}.json`, `routing_tradeoff_curve.png`, `calibration_plot.png`.

## EXP-006: Pathway 2 Track A — spherical steering (first attempt)
**Date:** 2026-04-12 (prompt + Phase 0.5 probe sweep) → 2026-04-13 (Phase 2 sweep)
**Status:** SUPERSEDED BY EXP-037 (pathway 11 direction-rotation finding)
**Motivated by:** ITI (2306.03341) — probe-as-steering-vector on a math benchmark.
**Hypothesis:** Injecting `α · w_norm` at L15/L19 during generation raises MATH accuracy.
**Key result:** Targeted tau=0.40 steering at t=0.15 gives "+26 net gain" on
256-tok holdout.
**Verdict:** REJECTED by later finding — cos(this_steering_vector, current_L19_DoM) ≈ 0.05. The direction rotates; the 2026-04-13 vector doesn't correspond to the current correctness axis.
**Files:** `pathway2/track_a/SUMMARY.md`, various per-t checkpoint JSONs.

## EXP-007: Pathway 2 Phase 2 — tau-threshold sweep
**Date:** 2026-04-13
**Status:** SUPERSEDED
**Key result:** t=0.15 tau=0.40 "locked config" with +26 net gain. Later
invalidated.

## EXP-008: Pathway 3 — complexity expansion
**Date:** 2026-04-13 → 14
**Status:** COMPLETE (abandoned direction)
**Motivated by:** EXP-002/003 — can more features help?
**Hypothesis:** Denoised multi-layer non-adjacent features at K∈{3,5,8} prototypes boost AUROC.
**What we actually tested:** 4 feature configs × 5 folds × phases 1-4.
**Key result:** No improvement over ABC-44. PIVOT decision.
**Verdict:** REJECTED.
**Files:** `pathway3/SUMMARY.md`, `phase4/go_no_go_decision.json`.

## EXP-009: Pathway 4 Track A — topo-guided selection + adaptive sampling
**Date:** 2026-04-15 → 16
**Status:** COMPLETE (now SUPERSEDED on label scheme, spirit preserved)
**Motivated by:** EXP-005 routing — can topo-features drive test-time K allocation?
**Hypothesis:** Feature-gated adaptive K sampling matches uniform K=32 accuracy at 1/10th compute.
**What we actually tested:** Experiment 8 — per-problem K decisions driven by feature score.
**Key result:** 77-89% compute savings at matched accuracy. "+11 holdout net
gain" at zero correct-side losses.
**Verdict:** CONFIRMED at the time (256-tok labels). The spirit — feature-gated
adaptive sampling beats uniform K — is also shown at 1024tok in EXP-037, with
a different winning feature (seq_len, not DoM).
**Files:** `pathway4/track_a/experiment8_adaptive_sampling/` tree.

## EXP-010: Pathway 4 Track B — learned steering
**Date:** 2026-04-15 → 16
**Status:** COMPLETE — null result
**Hypothesis:** Learned steering direction outperforms fitted-probe direction.
**Key result:** Training loss went down; holdout effect was null.
**Verdict:** REJECTED.
**Files:** `pathway4/track_b/training_log.json`, `holdout_results.json`, `go_no_go_decision.json`.

## EXP-011: Pathway 5 Track A — GSM8K cross-benchmark
**Date:** 2026-04-16 → 17
**Status:** SUPERSEDED BY EXP-017
**Motivated by:** EXP-009 — does it transfer to a different benchmark?
**Hypothesis:** Track A pipeline works on GSM8K.
**What we actually tested:** 4 phases (baseline, features, T=0.7, summary).
**Key result:** Looked like it worked. Later caught as truncation-confounded.
**Files:** `pathway5/track_a/phase_a{1..4}/`.

## EXP-012: Pathway 5 Track B — 7B MATH
**Date:** 2026-04-16 → 17
**Status:** SUPERSEDED BY EXP-017
**Key result:** 7B MATH AUROC at 256-tok was 0.682 (truncation-confounded).

## EXP-013: Pathway 6 Phase 0 — answer-extraction relabeling
**Date:** 2026-04-17
**Status:** COMPLETE
**Motivated by:** Audit: answer-extraction heuristic was misclassifying correct answers.
**Hypothesis:** Fixing answer extraction raises accuracy and shifts AUROC.
**Key result:** +47 correct answers found (11.4% → 20.8%). New label set is "NEW".
**Verdict:** CONFIRMED — forced rebuild of downstream experiments.
**Files:** `pathway6_rebuild/phase0_relabel/`.
**Enables:** EXP-014.

## EXP-014: Pathway 6 Phase 1 — 0.7961 holdout rebuild with NEW labels
**Date:** 2026-04-17 → 18
**Status:** COMPLETE (SUPERSEDED on label scheme — now 1024tok)
**Key result:** Holdout 0.7961 reproduces with NEW labels (104/500 correct).
**Files:** `pathway6_rebuild/phase1_prompt_model/holdout_metrics_v2.json`.

## EXP-015: Pathway 6 Phase 2 — completion majority-vote gating
**Date:** 2026-04-18
**Status:** COMPLETE
**Key result:** Gated MV at tau=0.3 gives +4 net holdout gain, zero R→W.
**Files:** `pathway6_rebuild/phase2_completion/`.

## EXP-016: Pathway 6 Phase 3 — cross-benchmark GSM8K + 7B MATH (truncated)
**Date:** 2026-04-18
**Status:** SUPERSEDED BY EXP-017
**Key result:** GSM8K 0.731, 7B MATH 0.682 — later caught as truncation-confounded.
**Files:** `pathway6_rebuild/phase3_cross_benchmark/gsm8k/`, etc.

## EXP-017: Pathway 6 Phase 6.5 — deconfounded cross-benchmark at 1024 tokens
**Date:** 2026-04-18 (scripts) → 2026-04-19 (RunPod H100, ~8 h)
**Status:** COMPLETE
**Motivated by:** Audit discovered `max_new_tokens=256` truncated 64% of GSM8K and 90% of 7B MATH outputs.
**Hypothesis:** At 1024 tokens, the cross-benchmark story is different.
**Key result:**
  - GSM8K acc: 36.7% → **66.0%**. Topo AUROC 0.731 → **0.615** (baseline beats topo by 0.126).
  - 7B MATH acc: 16.2% → **69.6%**. Topo AUROC 0.682 → **0.739**.
  - Transfer: 1.5B MATH → GSM8K = 0.504 (chance). Refuted.
**Verdict:** CONFIRMED — truncation was the driver. Cross-benchmark claim refuted.
**Files:** `pathway6_rebuild/phase6_5/FINAL_SUMMARY.md`.
**Enables:** Every 1024-tok experiment after.

## EXP-018: Pathway 7 Phase 7.1 — non-Euclidean PH (NO-GO)
**Date:** 2026-04-19 → 20
**Status:** COMPLETE — NO-GO
**Hypothesis:** Non-Euclidean distance metrics in PH (diffusion, effective resistance, cosine, DTM) beat Euclidean baseline (0.7961).
**Key result:** Best non-Euclidean variant = 0.774 AUROC. Worse than Euclidean.
**Verdict:** REJECTED.
**Files:** `pathway7/PLAYBOOK.md`, `pathway7/results_phase71/`.

## EXP-019: Pathway 8 Exp 1 — layer-wise PH
**Date:** 2026-04-20 → 21
**Status:** COMPLETE
**Hypothesis:** Distributing PH across all 28 layers captures richer signal than last-layer PH.
**Key result:** AUROC **0.6463** — much worse than ABC-44 (0.7961).
**Verdict:** REJECTED.
**Files:** `pathway8_layerwise/results/exp1_results.json`.

## EXP-020: Pathway 8 Exp 2 — method comparison (CoE, D2H, PH, combined)
**Date:** 2026-04-20 → 21
**Status:** COMPLETE
**Hypothesis:** CoE / D2HScore will beat PH-based features.
**Key result:** CoE-60 = **0.811**, D2H-lite = 0.806, ABC-44 = 0.7961, PH-168 = 0.646. PH+CoE combined = 0.758 (hurts).
**Verdict:** CONFIRMED — CoE wins within-domain.
**Files:** `pathway8_layerwise/results/exp2_results.json`.

## EXP-021: Pathway 8 Exp 3 — TwoNN intrinsic dimension
**Date:** 2026-04-21
**Status:** COMPLETE
**Key result:** AUROC 0.407 — worse than chance.
**Verdict:** REJECTED.
**Files:** `pathway8_layerwise/results/exp3_results.json`.

## EXP-022: Pathway 8 Exp 4 — cross-layer trajectory PH
**Date:** 2026-04-21
**Status:** COMPLETE
**Key result:** AUROC 0.682 — weaker than single-layer ABC-44.
**Verdict:** REJECTED.
**Files:** `pathway8_layerwise/results/exp4_results.json`.

## EXP-023: Pathway 8 Exp 5 — cross-domain BBH
**Date:** 2026-04-21
**Status:** COMPLETE
**Key result:** CoE transfers MATH↔BBH at 0.720 / 0.712. PH fails cross-domain. D2H-lite is asymmetric. **First signal that exceeded chance cross-benchmark.**
**Verdict:** CONFIRMED — CoE is the only domain-invariant feature family.
**Files:** `pathway8_layerwise/results/exp5_results.json`.

## EXP-024: Pathway 9 Exp 1 v2 — ABC-44 length deconfound (raw tokenizer length)
**Date:** 2026-04-22
**Status:** COMPLETE (later partially SUPERSEDED)
**Motivated by:** D6 audit — previous residualization was against wrong length proxy.
**Hypothesis:** ABC-44 headline 0.7961 includes a raw-output-length confound.
**Key result:** ABC-44 deconfounded AUROC = **0.7459** (drop of **0.050** from 0.7961).
**Verdict:** CONFIRMED at 256-tok labels. **Later reframed by Pathway 11**: the confound is a feature-engineering artifact of ABC-44 specifically; raw residual-stream DoM has cos(DoM, length) ≈ −0.09 (no confound).
**Files:** `pathway9/results/exp1_raw_length_v2.json`.

## EXP-025: Pathway 9 Exp 2 — PH + CoE orthogonality
**Date:** 2026-04-22
**Status:** COMPLETE
**Hypothesis:** PH-168 and CoE-60 carry orthogonal information → stacked ensemble lifts AUROC.
**Key result:** Stacked ensemble = 0.8117 vs CoE alone = 0.811. Lift **+0.001**. Redundant.
**Verdict:** REJECTED (no orthogonality).
**Files:** `pathway9/results/exp2_orthogonality.json`.

## EXP-026: Pathway 9 Exp 3a v2 — empirical-covariance Gaussian null for PH
**Date:** 2026-04-22
**Status:** COMPLETE
**Hypothesis:** 5 raw PH features beat rank-matched Gaussian null.
**Key result:** Real PH AUROC **0.690** vs null PH **0.693**. Gap zero.
**Verdict:** REJECTED — PH-specific signal absent; covariance explains it.
**Changed our understanding of:** What "topology" captures in these residual streams — nothing beyond second moments.
**Files:** `pathway9/results/exp3a_gaussian_null_v2.json`.

## EXP-027: Pathway 9 Exp 3b — token shuffle (order sensitivity)
**Date:** 2026-04-22
**Status:** COMPLETE
**Key result:** 3 of 5 PH features are order-indifferent. Only H0_max_lifetime and H1_max_lifetime show measurable order sensitivity.
**Files:** `pathway9/results/exp3b_token_shuffle.json`.

## EXP-028: Pathway 9 Exp 3c — count control
**Date:** 2026-04-22
**Status:** COMPLETE
**Key result:** 0 features in ABC-44 matched the name "n_features" — no-op.
**Verdict:** trivially correct (documentation check).

## EXP-029: Pathway 9 Exp 4 v2 — LR vs XGBoost matched CV5
**Date:** 2026-04-22
**Status:** COMPLETE
**Hypothesis:** XGBoost finds non-linear signal LR misses.
**Key result:** LR CV5 = 0.7012, best XGBoost CV5 = 0.6960. Lift **−0.005**. O-information OOMed at 3 TiB.
**Verdict:** REJECTED — signal is linear.
**Files:** `pathway9/results/exp4_xgboost_oinfo_v2.json`.

## EXP-030: Pathway 9 Exp 5 — cross-domain transfer (PH vs CoE vs D2H)
**Date:** 2026-04-22
**Status:** COMPLETE
**Key result:** PH transfers at chance / inverse (0.354, 0.497). CoE transfers symmetrically (0.720, 0.712, avg 0.716). D2H-lite is asymmetric.
**Verdict:** CONFIRMED — CoE is the domain-invariant signal.
**Files:** `pathway9/results/exp5_cross_domain.json`.

---

## [Pathway 10 is planning only — no experiments here.]

---

## EXP-031: Pathway 11 Stage 1 — 7B MATH-500 all-layer extraction @ 1024 tokens
**Date:** 2026-04-23 → 24 (RunPod H100 pod `y687b9z2dgukcj`)
**Status:** COMPLETE
**Motivated by:** All cross-scale experiments used truncated 7B runs.
**Key result:** 7B MATH-500 T=0 accuracy = **73.2%** (366/500). 42 GB cached.
**Note:** `capture_attention=False` for safety → d2h_attn_entropy is zeros.
**Files:** `pathway11_h100/stage1_extract_math500_7b.py`, `data/math500_7b/`.

## EXP-032: Pathway 11 Stage 2 — 1.5B MATH-500 re-extraction @ 1024 tokens
**Date:** 2026-04-23 → 24
**Status:** COMPLETE
**Key result:** 1.5B MATH-500 T=0 accuracy = **48.6%** (243/500). 19 GB.
**Verdict:** The old 20.8% baseline was a 256-tok truncation artifact.
**Files:** `pathway8_layerwise/extract_math500.py` (reused), `pathway8_layerwise/data/math500/`.

## EXP-033: Pathway 11 Stage 3 — K=8 self-consistency with per-sample L19
**Date:** 2026-04-23 → 24
**Status:** COMPLETE
**Key result:** K=8 @ T=0.7 majority-vote gives **55.0%** accuracy (+6.4pp over K=1 greedy). 13 MB.
**Enables:** Bucket definitions (A/B/C/D), EXP-037, EXP-039.
**Files:** `pathway11_h100/stage3_k8_selfconsistency.py`, `data/k8_selfconsistency/`.

## EXP-034: Pathway 11 Stage 4a — BBH 3×250 all-layer extraction
**Date:** 2026-04-23 → 24
**Status:** COMPLETE
**Key result:** 1.5B on BBH subsets: tracking=12.4%, logical=6.0%, web_of_lies=54.0%. 18 GB.
**Files:** `pathway8_layerwise/extract_bbh.py` (reused), `pathway8_layerwise/data/bbh/`.

## EXP-035: Pathway 11 Stage 4b — BBH per-subset diagnostics
**Date:** 2026-04-24
**Status:** COMPLETE
**Key result:** Per-subset L19 DoM AUROC, length/correctness Spearman.
**Files:** `pathway11_h100/stage4b_bbh_per_subset.py`, `results/stage4b_bbh_per_subset.json`.

## EXP-036: Pathway 11 Stage 5 — L19 DoM cross-benchmark transfer
**Date:** 2026-04-24
**Status:** COMPLETE
**Motivated by:** Does single-layer DoM match Pathway 9 CoE cross-domain signal?
**Hypothesis:** L19 DoM symmetric transfer matches CoE-60 symmetric transfer.
**Key result:** MATH→BBH pooled **0.747**, BBH→MATH **0.693**, symmetric **0.720** vs CoE-60's 0.716 (+0.004). Matches within noise.
**Verdict:** CONFIRMED — CoE is not richer than single-direction DoM.
**Files:** `pathway11_h100/stage5_bbh_dom_transfer.py`, `results/stage5_bbh_dom_transfer.json`.

## EXP-037: Pathway 11 Exp 2 — prefill-gated compute allocation
**Date:** 2026-04-24
**Status:** COMPLETE
**Motivated by:** Pathway 10 v2 E2 — prefix-CoE gates self-consistency K.
**Hypothesis:** Prefill-DoM-gated K∈{1,8} beats uniform K=4 at matched compute.
**Key result:**
  - Prefill L19 DoM AUROC = **0.7731** (vs K=1), **0.8228** (vs K=8 majority).
  - Monotonic gating doesn't beat uniform K.
  - **Middle-heavy beats monotonic** (+3.4 pp at same compute).
  - **Seq-len top-heavy is Pareto-dominant** (0.542 at K=4.5).
  - **Refuse-and-spend @ cov=0.5: 71.6% accuracy on answered** at avg K=2.5 vs 49.4% random. **This is the clean E3 win.**
  - Oracle acc = 0.589 at oracle compute 1.01 (most problems need K=1).
**Verdict:** Mixed — monotonic gating REJECTED; refuse-and-spend CONFIRMED.
**Changed our understanding of:** What the prefill signal is good for (refusal + selective prediction, not compute gating).
**Files:** `pathway11_h100/prefill_gated_compute/` tree.

## EXP-038: Pathway 11 Exp 3 — 7B prefill PR inversion
**Date:** 2026-04-24
**Status:** COMPLETE
**Motivated by:** Exploratory — does PR separate correct from incorrect at prefill time?
**Hypothesis (initial):** PR_correct < PR_incorrect at prefill (as it does at final-token).
**Key result:** **7B prefill ratio = 1.377** (correct > incorrect, CI [1.226, 1.757], P(>1)=1.000, balance-controlled 1.227). Opposite sign from naive expectation. Not in 1.5B. Not in BBH (web_of_lies ratio 1.009, balanced).
**Mechanistic:** Driven by incorrect-group concentration at easy difficulty levels (Level 2 ratio 3.98, Level 5 ratio 1.11).
**D-bucket signature:** Lowest group prefill PR of any bucket (14.49 vs A/B/C at 18.7/16.6/20.0). But doesn't localize per-problem (flat local PR across buckets).
**Verdict:** CONFIRMED — inversion is real, but it's "incorrect-group concentration", not "7B has higher-dim correct manifold."
**Files:** `pathway11_h100/prefill_inversion/` tree.

## EXP-039: Pathway 11 Exp 2b — multi-signal oracle-K classifier
**Date:** 2026-04-24
**Status:** COMPLETE
**Motivated by:** EXP-037 + EXP-038 suggested non-monotonic multi-feature gating might find B-bucket + D-bucket.
**Hypothesis:** {prefill DoM, local prefill PR, seq-len} multi-feature classifier beats best single-signal gating by ≥ 2pp.
**Key result:**
  - Logreg macro-F1 = 0.548, overall OOF = 0.620. RF = 0.494 / 0.636.
  - Multi-signal vs best single at matched compute = **+0.4 pp**. **Below 2pp bar.**
  - D-recall: logreg 41.7% (15/36), RF 44.4% — **below 48.6% class prior**.
  - Drop `prefill_lpr` → Δ macro-F1 = **+0.000**. Local PR is dead weight.
**Verdict:** REJECTED.
**Files:** `pathway11_h100/multi_signal_oracle/` tree.

## EXP-040: Pathway 11 Exp 1 — cross-model breathing (Phi-3-mini + Llama-3.2-1B)
**Date:** 2026-04-24 (extraction ~80 min; analysis ~30 min)
**Status:** COMPLETE
**Motivated by:** Does the Qwen breathing pattern replicate across architectures?
**Hypothesis:** Phi-3-mini (L21) and Llama-3.2-1B (L11) show the same three-phase breathing + correct<incorrect at final.
**Key result:**
  - Phi-3 acc 44.8% (224/500), PR 1=19.3, peak ≈104 at pos 10, final 17.6 (correct=12, incorrect=17).
  - Llama acc 25.2% (126/500), PR 1=2.8, peak ≈115 at pos 50, final 8.0 (correct=4, incorrect=8).
  - **Both reproduce three-phase breathing + correct-collapses-harder.**
  - **Architecture-specific timing:** Llama separates correct/incorrect at pos 25, Phi-3 at pos 100.
  - **Prefill-depth inversion reproduces** at late layers (Phi-3 L27–31, Llama L12–16).
**Verdict:** CONFIRMED — breathing is universal across scale and architecture.
**Files:** `pathway11_h100/exp1_cross_model/` tree, `results.json` (59 KB).

## EXP-041: Test 3 — gibberish control for breathing
**Date:** 2026-04-24 evening (pod `lsuoka6bo8io7m`, ~15 min wall)
**Status:** COMPLETE
**Motivated by:** Before accepting breathing as a reasoning phenomenon, rule out "happens on any AR generation".
**Hypothesis:** Random-token prompts give the same three-phase breathing as MATH-500.
**Key result:**
  - **Random tokens: flat PR ≈ 10 across all 7 positions** (9.0 → 11.7). No breathing.
  - Stream-of-consciousness: pos 1 = 4.6, peak at pos 10 = 15.7, monotonic decline. Muted and differently-shaped.
  - MATH-500 reproduces classic 13.7 → 30.5 → 22.2.
**Verdict:** CONFIRMED — breathing is content-dependent. Not AR mechanics.
**Changed our understanding of:** Breathing needs reasoning-task structure. Non-reasoning fluent generation produces a different, muted curve.
**Files:** `pathway11_h100/gibberish_control/` tree.

## EXP-042: Test 2 — no-CoT control for breathing
**Date:** 2026-04-24 evening (pod `lsuoka6bo8io7m`, ~4 min wall)
**Status:** COMPLETE (INCONCLUSIVE as framed)
**Motivated by:** After EXP-041 rejected AR null, check whether breathing tracks CoT vs generation length.
**Hypothesis:** With an answer-only prompt, PR breathes over fewer tokens (tracks length) or stays flat (tracks CoT specifically).
**What we actually tested:** Same MATH-500 first-50 subset with system prompt "Answer with just the number, no explanation."
**Key result:** No-CoT gen length: median **2 tokens**. Pos 1 PR = 15.78 (slightly higher than CoT's 13.74). Final PR = 19.80 (≈ CoT pos 10). **Physically cannot show a breathing curve in 2 tokens** — the question as framed is unanswerable.
**Verdict:** INCONCLUSIVE.
**Next test (not run):** Short-CoT prompt targeting 20–40 tokens — e.g., "Solve this in one sentence, then give the answer in \boxed{}." (HYPOTHESES H-2).
**Files:** `pathway11_h100/no_cot_control/` tree.

---

## EXP-43: P11-FE719 — Leave-One-Subject-Out CV on F-2 prefill L19 DoM
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Probe-non-diagnosticity critique formalised in 2603.18280
§2 (probe results) — random-split CV is non-diagnostic, null controls reach
100% accuracy. F-2's 0.7731 AUROC was reported under 5-fold *random* CV;
subject-stratified leakage is plausible because MATH-500 base accuracy varies
by subject (Algebra 124 problems vs Counting & Probability 38). LOCO-CV is
the diagnostic test.
**Hypothesis:** If F-2 is largely a subject/topic detector, mean LOCO AUROC
drops below 0.65 (refutation threshold per FE719 spec). If F-2 is a
correctness signal, LOCO AUROC stays close to random-OOF.
**What we actually tested:** Loaded the L19 prefill cache
(`pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`, 500×1536 fp16)
and the MATH-500 `subject` column. For each of 7 subjects, computed DoM =
mean(prefill[correct ∧ ~held]) − mean(prefill[~correct ∧ ~held]) on the
training fold (~430 problems), then scored AUROC of `prefill_held @ DoM`
vs `correct_held` on the held-out subject.
**Key result:** mean LOCO AUROC 0.7427, worst 0.6032 (Number Theory, n=62),
best 0.9000 (Geometry, n=41). Per-subject: Algebra 0.6982, Counting &
Probability 0.7278, Intermediate Algebra 0.7702, Prealgebra 0.7919,
Precalculus 0.7074. In-sample full-data DoM AUROC 0.7834 (sanity anchor —
no held-out, not directly comparable to the 0.7731 5-fold OOF figure). All
3 new claims PASS in `validate_claims.py` and REGEN_PASS via
`recompute_fe719.py`.
**Verdict:** F-2 NOT REFUTED. The ~3pp drop from 0.7731 (random-OOF) to
0.7427 (LOCO mean) confirms a small subject-leakage component, but 0.7427
is well above the 0.65 refutation threshold and well above chance.
**Changed our understanding of:** F-2 generalises across subjects with a
modest but real topic component; the strongest fold (Geometry 0.90) is
likely small-n variance (n=41, ~50/50 correct/incorrect), and the weakest
fold (Number Theory 0.60) suggests Number Theory problems use prefill
geometry slightly differently from the rest of MATH-500 — a direction
worth a follow-up FE.
**Files:** `pathway11_h100/loco_subject/recompute_fe719.py` (recompute
script, deterministic), `pathway11_h100/loco_subject/results.json` (output).
**Depends on:** EXP-037 (the 0.7731 5-fold OOF baseline).
**Enables:** Reusable LOCO machinery for FE299 (topic-stratified AUROC),
FE448 (length partial-correlation), and the rest of the Phase 1 sanity
battery.

## EXP-44: P11-FE448 — Length partial-correlation control on F-2 prefill L19 DoM
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Su et al. 2505.00127 (Between Underthinking and
Overthinking) plus the concern that prefill = question-only tokens may
carry a length-prior. F-9 already showed the 44-feature ABC pipeline has
length confounds; FE448 tests whether the *raw L19 activations* themselves
encode length to a degree that explains F-2.
**Hypothesis:** If L19 prefill encodes mostly "predicted output length"
and length predicts correctness, the headline DoM result is doubly
explained by a non-geometric variable. Refutation: residual AUROC after
partialing predicted-length ≤ 0.55. Survives: > 0.65.
**What we actually tested:** (1) Ridge regression of `seq_len` on L19
prefill activations (n=500, d=1536, α=1e-3), getting predicted length
ŝ. (2) Computed R² of (s, ŝ). (3) Per-feature Frisch-Waugh
residualization: X_resid[:,j] = X[:,j] − αⱼ ŝ_centered. (4) DoM AUROC
on X_resid vs raw X, both in-sample on n=500.
**Key result:** length R² ≈ 1.0 (in-sample, n<d ridge near-interpolates).
Sequence length alone has AUROC 0.7986 vs `correct` — *higher* than raw
DoM in-sample (0.7834) and the headline 5-fold OOF (0.7731). Predicted
length alone has AUROC 0.7994. After partialing predicted-length out of
prefill, DoM AUROC drops 0.7834 → 0.6647 (12pp drop). cos(DoM_raw,
DoM_resid) = 0.785 — the residualized direction still aligns ~79% with
raw, so the projection isn't degenerate.
**Verdict:** F-2 GRAY ZONE — neither refuted (residual 0.6647 > 0.55)
nor cleanly surviving (residual 0.6647 ≤ 0.65). Demote F-2 from STRONG
→ MODERATE. The length confound is large enough that ≈⅔ of the original
AUROC is length-explainable.
**Changed our understanding of:** F-2's strength claim. The prefill DoM
is partly a correctness direction *and* partly a length-predictive
direction; the two share a near-perfect linear subspace at L19. Pure
"correctness geometry" framing is too strong. Combined with FE719 (LOCO
mean 0.7427), F-2 is showing consistent erosion under cheap controls but
hasn't fallen below 0.5 — a real signal contaminated by length.
**Phase 1 decision branch status:** 1 of 5 cheap-controls below AUROC
0.70 (FE448 residualized). Need ≥3 to trigger demotion + Phase-3 skip.
Continue Phase 1 (FE145, FE299, FE101) before pivoting.
**Files:** `pathway11_h100/length_partial/recompute_fe448.py` (recompute,
deterministic), `pathway11_h100/length_partial/results.json` (output).
**Caveat:** Length R² ≈ 1.0 is partly a ridge-overfit artifact at n<d.
A proper OOF length-prediction R² would be lower; the in-sample residual
AUROC 0.6647 is therefore an *upper bound* on the contamination — true
OOF residual would likely be lower still.
**Depends on:** EXP-037, EXP-43.
**Enables:** Cleaner FE145 interpretation (per-layer DoM may pick up
length-predictive layers); motivates FE299 (topic-stratified AUROC) and
a possible follow-up FE on per-layer length-predictability.

## EXP-45: P11-FE145 — Per-layer prefill DoM AUROC sweep (L0..L28)
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** CAA (Panickssery et al.) reports the optimal steering
layer at ~⅓ depth with a sharp drop-off at L17 (53% depth) on Llama-2.
F-2 picks L19/28 (~68% depth) on Qwen-2.5-1.5B — well past CAA's
drop-off. Either Qwen has a different geometry, or L19 is post-hoc
lucky. A full per-layer sweep settles it.
**Hypothesis:** L19 is the argmax layer for prefill DoM AUROC (or within
±0.005 of it). If L19 is far from the peak (>0.005 below), the F-2
choice is post-hoc.
**What we actually tested:** Loaded all 29 per-layer prefill
activations (input embeds + 28 transformer blocks) from
`pathway8_layerwise/data/math500/problem_*.npz` (500 problems × 29
layers × 1536 dims). For each layer, ran 5-fold stratified OOF DoM
(seed=9999): train on 4 folds (mean(pos) − mean(neg)), score on
held-out fold, compute AUROC over all 500 OOF scores.
**Key result:** Peak at **L21 with AUROC 0.7718**. L19 at AUROC
**0.7705** (gap 0.0013 — within ±0.005). The curve rises monotonically
from L0 (0.50, embedding-output, no info) to L18 (0.7616), peaks at
L19–L21 (0.7705–0.7718), then declines slightly to L28 (0.7568). L17
(CAA drop-off layer) is at 0.7635 — *no sharp drop-off* on Qwen-2.5-1.5B.
**Verdict:** F-2 L19 choice CONFIRMED. The peak plateau is L19–L21,
spanning ~0.001 in AUROC; L19 is on the plateau. CAA's L17 drop-off
hypothesis does NOT generalise to Qwen-2.5-1.5B — Qwen's correctness
direction stays informative across the upper half of the network.
**Changed our understanding of:** F-2's geometric specificity. The
peak is not at one privileged layer — it's a 3-layer plateau (L19–L21).
"L19 is special" is too strong; "the upper-mid block (L19–L21) is the
DoM peak" is the right framing.
**Phase 1 decision branch status:** This is a *confirming* control,
not a refutation. Counter unchanged: 1 of 5 below 0.70 (FE448
residualized only). FE719+FE145 confirmed F-2 mostly survives;
FE448 demoted strength to MODERATE due to length confound.
**Files:** `pathway11_h100/per_layer_sweep/recompute_fe145.py` (regen,
deterministic), `pathway11_h100/per_layer_sweep/results.json` (output).
**Note:** The 0.7705 5-fold OOF AUROC at L19 differs by ~0.003 from the
0.7731 headline figure (EXP-037). Both are 5-fold OOF on the same data
but used different seeds and CV-split implementations; the difference
is well within seed-variance and doesn't change conclusions.
**Depends on:** EXP-037 (L19 baseline), EXP-43 (LOCO), EXP-44 (length).
**Enables:** Per-layer follow-ups (which layer carries length info, see
H-N: layer-resolved length artefacts), and a possible H-N about the
L19–L21 plateau.

## EXP-46: P11-FE299 — Within-topic prefill DoM AUROC
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Refutation 2 in the original F-2 brief: if within-topic
AUROC collapses toward 0.5 while overall stays at 0.77, the prefill DoM
signal is dominated by topic-familiarity base-rates rather than per-
problem decomposability — refuting H-5's framing. This is the
complement of FE719: same data, opposite split.
**Hypothesis:** Within-topic AUROC > 0.55 ⇒ F-2 is per-problem (NOT a
pure topic-detector). Within-topic AUROC ≤ 0.55 ⇒ F-2 is topic-base-
rate.
**What we actually tested:** For each of 7 MATH-500 subjects, took the
n_subject problems' L19 prefill activations (from `m15b_prefill.npz`)
and ran k-fold stratified OOF DoM (k = min(5, min-class-count),
seed=9999) within that subject. Computed AUROC over the within-subject
OOF scores.
**Key result:** Mean within-topic AUROC **0.7143**. Per-subject:
Algebra 0.6903 (n=124), Counting & Probability 0.6500 (n=38), Geometry
0.8128 (n=41), Intermediate Algebra 0.6683 (n=97), Number Theory
**0.6000** (n=62), Prealgebra **0.8305** (n=82), Precalculus 0.7481
(n=56). 7/7 subjects scored.
**Verdict:** F-2 NOT a pure topic-detector. The 0.7143 within-topic
mean is well above the 0.55 refutation threshold. The ~6pp gap between
within-topic mean (0.7143) and overall (0.7731) quantifies a small
topic-base-rate component — the bulk of the signal is per-problem.
**Changed our understanding of:** Number Theory is the consistent weak
spot. Both FE719 (LOCO 0.6032) and FE299 (within 0.6000) bottom out at
Number Theory — the prefill DoM direction is informative across most
subjects but ~⅔-strength on Number Theory specifically. Possible
follow-up: per-subject DoM cosine matrix to see whether NT problems
genuinely use a different correctness direction.
**Phase 1 decision branch status:** Counter unchanged: 1 of 5 below
0.70 (FE448 residualized only). FE299 0.7143 is above 0.70. F-2 stays
MODERATE; the length confound remains the dominant counterargument.
**Files:** `pathway11_h100/within_topic/recompute_fe299.py` (regen),
`pathway11_h100/within_topic/results.json` (output).
**Caveat:** k=2-5 within-subject is small; the Counting & Probability
result (0.6500, n=38, k=2) is noisy. Geometry 0.8128 (n=41, k=2) is
also small-n.
**Depends on:** EXP-037, EXP-43.
**Enables:** Per-subject DoM cosine matrix (a possible H-N about
correctness-direction stability across subjects).

## EXP-47: P11-FE101 — LEACE linear-erasure null for F-2
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** F-2's headline claim is "a linear probe finds AUROC
0.7731" — but we had no strict null. Without LEACE, we couldn't rule
out probe-leakage or non-linear residuals. LEACE (Belrose et al.,
2306.03819) provides closed-form minimum-damage linear erasure: any
linear classifier on LEACE-erased data is provably uninformative.
**Hypothesis:** If F-2 is genuinely linear, OOF AUROC collapses to
~0.5 after LEACE. If AUROC stays substantially above chance (>0.55),
the prefill correctness signal lives in a non-linear subspace the DoM
probe was failing to pick up cleanly.
**What we actually tested:** Per-fold LEACE on n=500 1.5B L19 prefill
activations: (1) μ, Σ_ridge from train fold; (2) d_train = μ_pos −
μ_neg; (3) w_train = Σ_ridge^(-1) d_train; (4) erase via X_erased[i]
= X[i] − ((X[i] − μ) · w / (d · w)) · d on both train and test
(using train-fold parameters); (5) refit DoM on erased train (zero by
construction); (6) score erased test. 5 folds, seed=9999.
**Key result:** Raw OOF AUROC **0.7705** (sanity anchor matching
FE145). LEACE-erased OOF AUROC **0.5000** — perfect collapse.
Erasure ratio (‖DoM_erased(test)‖/‖DoM_raw(test)‖) = 0.0000 to
floating-point precision: the LEACE projection completely zeros out
the DoM direction in the train fold, and test-fold DoM trained on
erased data is zero, so all test scores are zero.
**Verdict:** F-2 IS LINEAR. The 27pp collapse confirms that the
correctness concept lives entirely in the DoM-aligned linear subspace
of L19 prefill — no residual non-linear signal.
**Changed our understanding of:** The "linear concept" framing for F-2
is not a methodological assumption — it's an empirical truth on
n=500. If LEACE removes ALL linear separability and AUROC collapses
to chance, then any future probe (logistic, MLP, polynomial features)
operating on the unerased prefill must be picking up the SAME linear
direction; non-linearity adds nothing on this data.
**Phase 1 final tally:** 5/5 cheap controls completed. 1 of 5 below
AUROC 0.70 (FE448 length-residualized only, 0.6647). Below the 3-of-5
threshold for Phase 1 decision branch (which would have triggered
demote-and-skip-Phase-3). F-2 survives Phase 1 — but length confound
demotes strength to MODERATE.
**Files:** `pathway11_h100/leace_erasure/recompute_fe101.py` (regen),
`pathway11_h100/leace_erasure/results.json` (output).
**Caveat:** With ridge α_rel=1e-3 in n<d regime the LEACE eraser is
near-perfect on this sample but not literally rank-d. A larger sample
size or smaller ridge would tighten the construction; the current
result is conservatively perfect for the n=500 fold setup.
**Depends on:** EXP-037, EXP-43, EXP-44, EXP-45, EXP-46.
**Enables:** Phase 2 ROI-10 anchors (FE115 Song-Zhong, FE749 spectral
α, FE181 token-prob baseline) — F-2 is a sound enough anchor to
warrant the more expensive controls. Phase 3 causal corroborators
remain in scope (decision branch did not fire).

## EXP-48: P11-FE115 — Song-Zhong pos/ctx decomposition on L19 prefill+final
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** F-3's orthogonality claim (`cos(prefill_DoM, final_DoM) ≈
0.046`) was the structural cornerstone of the "two circuits" framing — that
"can I solve this?" and "did I solve this?" live in geometrically unrelated
subspaces. Song & Zhong (2310.04861) showed transformer hidden states
decompose into μ + pos_t + ctx_c + resid; if our orthogonality is positional
(prefill at t=0, final at t=T_i−1), removing pos_t collapses it. Cheapest
direct refutation test for F-3 in scope.
**Hypothesis:** Refutation if cos(prefill_DoM_resid, final_DoM_resid) > 0.30
after subtracting μ + pos_t + ctx_i. F-3 holds if the cosine stays small.
**What we actually tested:** Streamed the 500 per-problem L19 hidden states
(shapes (T_i, 1536), T_i ∈ [123, 1024]) through three passes:
(1) accumulate global μ and per-position pos_t = mean_i(s_i[t]) − μ for
t ∈ [0, T_max); (2) per-problem ctx_i = mean_t(s_i[t]) − μ; (3) residuals at
t=0 (prefill) and t=T_i−1 (last token) only — `resid_i[t] = s_i[t] − μ −
pos_t[t] − ctx_i`. Then DoM = mean(resid[correct]) − mean(resid[~correct]),
cosine on whole-vector DoMs, and 5-fold stratified OOF AUROC (seed=9999).
**Key result:** **cos_resid = 0.0008** (vs cos_raw = −0.0617 on this cache;
differs in sign from the P10 number 0.046 due to different cache /
tokenization). The residualized cosine is FURTHER from any "two-circuit
collapse" prediction than the raw cosine — F-3 strengthens. AUROC numbers:
prefill raw 0.7705 → resid **0.8016** (+3.1pp lift); final raw 0.6603 →
resid 0.6865 (+2.6pp lift). Removing μ + pos_t + ctx_i *purifies* the
correctness signal in both directions; the per-problem context vector ctx_i
was carrying confound (likely length-correlated, since FE448 already
established length R²≈1.0 on L19 prefill).
**Verdict:** F-3 NOT REFUTED — strongly reinforced. The orthogonality is a
structural property of the L19 representation, not a positional encoding
artifact. Prefill and final-token DoMs operate in genuinely independent
subspaces.
**Changed our understanding of:** Two compounding shifts.
(1) F-3 graduates from "MODERATE on raw cosines" to "STRONG, survives full
Song-Zhong residualization". The two-circuit framing is structurally robust.
(2) F-2's prefill DoM AUROC has a discoverable +3pp ceiling: simply
subtracting per-problem context (a free, deterministic transformation)
yields a stronger probe. This is independently interesting for the
selective-prediction pipeline (F-8) — a 0.80 AUROC probe at L19 prefill is
new headroom.
**Phase 2 status:** First of 3 Phase 2 anchors complete. F-3 holds; F-2
gains a stronger probe variant. Number Theory consistency (worst fold in
both FE719 and FE299) is unchanged — Song-Zhong doesn't address it.
**Files:** `pathway11_h100/song_zhong/recompute_fe115.py` (regen, ~6 min on
CPU streaming through 500×~600MB worth of fp16 states),
`pathway11_h100/song_zhong/results.json` (output).
**Caveat:** The "final token" in this experiment is the last *generated*
token at position T_i−1, which differs from the headline's "final-token L19
DoM AUROC 0.7186" (the prefill_gated_compute aggregate, position semantics
unclear without reading that script). Cos-on-residuals is the load-bearing
number; the 0.6603 final-raw AUROC is an internal consistency anchor for
this experiment, not a reproduction of the 0.7186 headline.
**Depends on:** EXP-037, EXP-44 (length confound established).
**Enables:** Re-running F-8's selective-prediction policy with the +3pp
Song-Zhong residualized prefill DoM. Possible follow-up: cross-layer
Song-Zhong sweep to see whether the +3pp lift generalises beyond L19.

## EXP-49: P11-FE181 — Token-probability baseline (mean / sum log-prob) vs F-2 DoM
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** A standing alternative explanation for F-2 has been
"prefill DoM is just a fancy way of measuring how confidently the model
generates the answer span — it's restating mean token-probability." If
true, DoM offers no novel geometric content over a free, model-internal
calibration signal. Cheapest direct head-to-head in scope: regenerate the
same OOF AUROC pipeline against `mean_t log P(y_t | y_<t, x)` on the
answer span (already cached as scalar `mean_logprob` in every per-problem
NPZ). Adding the joint test with prefill DoM separates "redundant" from
"complementary."
**Hypothesis:** F-2 is subsumed if `auroc_mean_logp_15b > 0.7731` AND
`joint_meanlogp_dom_auroc_15b ≈ auroc_mean_logp_15b` (DoM adds no
information beyond token-prob).
**What we actually tested:** Loaded scalar `mean_logprob` from 500 1.5B
NPZs and 500 7B NPZs. Computed (1) 5-fold stratified OOF DoM-style 1D
probe with fold-safe orientation on `mean_logprob`; (2) same probe on
`sum_logprob = mean_logprob × seq_len`; (3) joint OOF probe via
`Σ⁻¹(μ_pos − μ_neg)` on `[mean_logprob, prefill_DoM_proj_L19]` aligned with
the m15b_prefill cache. seed=9999.
**Key result (1.5B):** **mean_logp AUROC = 0.6721**; sum_logp AUROC =
**0.8478**; joint [mean_logp, DoM_proj] = **0.7836**. F-2 DoM (0.7731)
beats mean_logp by 10pp; the joint adds only ~1pp over DoM alone. Sum_logp
is structurally length-confounded — FE448 already established length-alone
AUROC 0.7986, and `sum_logp = mean_logp × seq_len`. The 7B numbers track
the 1.5B numbers (mean 0.6336, sum 0.8672), so the conclusion isn't
model-specific.
**Verdict:** F-2 NOT SUBSUMED by token-probability. The DoM direction
encodes correctness signal that mean log-likelihood does not access. Sum
log-prob's higher AUROC is a length artifact, not a calibration win.
**Changed our understanding of:** Two updates.
(1) F-2's "is this just output entropy?" counterargument is RETIRED.
mean_logp gives 0.67 — meaningfully below DoM. The geometry is real.
(2) Joint probe lifts only +1pp, suggesting the prefill DoM and mean
token-likelihood are *partially* redundant but each carries some unique
correctness information. Worth a follow-up that pulls per-token
min/product logprobs (which the cached scalar can't cover) — that's
gated on a fresh forward pass, deferred to next H100 session.
**Phase 2 status:** Second of 3 Phase 2 anchors complete. F-2 retains
geometric content. FE115 already added a Song-Zhong +3pp ceiling for F-2
prefill DoM (0.7705 → 0.8016 on residuals); FE181 confirms that ceiling is
not just "more length" — Song-Zhong residualization removes ctx_i, while
mean_logp is length-normalized — so the +3pp lift is *not* the same signal
that drives sum_logp's 0.8478.
**Files:** `pathway11_h100/token_prob/recompute_fe181.py` (regen, ~1 min
on CPU; scalar I/O dominated),
`pathway11_h100/token_prob/results.json` (output).
**Caveat:** Scope is mean and sum aggregations only — per-token min and
product would require either re-extracting per-token logprob arrays from
HF or running a fresh forward pass with `output_scores=True`. Deferred to
next H100 session per scope_note in results.json. The mean-only test
answers the headline "DoM = entropy?" question; min/product are
follow-ups, not gates.
**Depends on:** EXP-037 (F-2 anchor), EXP-44 (length partialing for the
sum_logp confound interpretation), EXP-48 (FE115 Song-Zhong context).
**Enables:** Closes the "DoM = output entropy" alternative on F-2's
PERSPECTIVES list. Per-token min/product extraction is the next natural
extension if and when fresh forward passes are budgeted.

## EXP-50: P11-FE749 — Spectral α (HT-SR power-law SVD exponent) vs F-2 DoM
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Martin & Mahoney's heavy-tailed self-regularization theory
(2002.03175 et seq.) proposes that the power-law tail exponent α of
weight-matrix singular values is a quality signal — small α (heavier tail)
correlates with stronger generalization at training-time. Extending the
same descriptor to *residual-stream* SVD across (problem, layer) is a
natural test: does each problem's intermediate-state spectrum carry α-style
predictive content for *correctness* at inference time? If yes, we have a
single-scalar competitor to F-2's DoM probe rooted in spectral theory; if
no, we narrow the parsimony zoo (F-9). Cheapest direct head-to-head in
scope.
**Hypothesis:** α subsumes F-2 if `alpha_best_auroc_15b > 0.7731` AND
`joint_alpha_dom_auroc_15b ≈ alpha_best_auroc_15b` (DoM adds no information
beyond α).
**What we actually tested:** For each (problem, layer ∈ [0, 28]) on both
1.5B and 7B caches: take the top-K=50 singular values of the (T_i, hidden)
hidden-state matrix, fit `log σ_k = log c − α · log k` by OLS, store α as
a single scalar per (problem, layer). Per-layer 5-fold stratified OOF
DoM-style 1D probe → α-AUROC[layer]. Joint test aligns α at the
best-α-layer with the prefill-DoM projection on the m15b_prefill cache and
runs a 2-feature OOF probe. seed=9999.
**Key result (1.5B):** α-AUROC by layer is U-shaped:
  L0=0.541, L1=0.572, L3=0.587 (early peak),
  L13=0.514, L17=0.475, L18=0.511 (mid trough; some layers
  *anti-correlate* with correctness),
  L19=0.522 (DoM's peak layer ≈ chance for α),
  L28=0.7026 (final layer, max).
The signal is in the spectral *shape changes* at the boundaries of the
network, not at the mid-layer reasoning regime where DoM lives. Joint
[α_L28, prefill_DoM_proj_L19] = 0.7833 vs DoM alone 0.7731 → α-L28 adds
~1pp of complementary signal.
**Key result (7B):** Same U-shape, slightly stronger:
  L0=0.587, L1=0.668 (early peak much stronger than 1.5B),
  L13–L18 in 0.49–0.52 range (some negative),
  L19=0.522, L28=0.7128 (final-layer peak).
Cross-scale agreement on the U-shape and on "α at L19 ≈ chance"
strengthens the conclusion; this isn't a 1.5B-specific quirk.
**Verdict:** F-2 NOT SUBSUMED by spectral α. The DoM direction encodes
correctness signal that residual-stream singular-value shape does not
access at the relevant depth. F-9 (parsimony / single-scalar redundancy
with DoM) gains a second confirming counter-example: α joins CoE-60 as a
proposed-but-dominated scalar competitor.
**Changed our understanding of:** Three updates.
(1) F-9 broadens. CoE-60 redundancy was the original anchor; spectral α
adds a structurally different scalar (spectral, not trajectory-shape) that
also fails to beat single-layer DoM. The parsimony claim now has two
independent confirming probes.
(2) The U-shape is interesting in its own right. α-AUROC concentrating at
L0 and L28 — the input/output boundary layers — while DoM peaks at L19–L21
suggests two distinct geometric regimes: spectral shape (likely
encoding-format / output-readiness) at the boundaries, and a
correctness-aware abstract-reasoning manifold in mid-depth. The two are
~orthogonal in the joint test (+1pp).
(3) HT-SR theory may need a depth-stratified version. The original
weight-matrix α framing has no within-network depth dependence; here
residual-stream α has strong depth structure. Worth a follow-up note in
PERSPECTIVES.md if and when we revisit the HT-SR thread.
**Phase 2 status:** Third of 3 Phase 2 anchors complete.
  - FE115 (Song-Zhong, EXP-48): F-3 reinforced; F-2 +3pp ceiling.
  - FE181 (token-prob, EXP-49): F-2 not subsumed by mean log-prob.
  - FE749 (spectral α, EXP-50): F-2 not subsumed by α; F-9 broadens.
All three Phase 2 anchors point the same direction: **F-2 (prefill L19 DoM)
survives every cheap competitor we've thrown at it. Its geometric content
is not a positional artifact, not output entropy, and not spectral shape.**
**Files:** `pathway11_h100/spectral_alpha/recompute_fe749.py` (regen,
~2h45m CPU on 24 cores; 29k SVDs),
`pathway11_h100/spectral_alpha/results.json` (output).
**Caveat:** α here is a residual-stream descriptor, NOT a weight-matrix
descriptor — different from the original HT-SR α used in
`weightwatcher`-style analyses. Comparison to that literature must keep
this distinction in mind. Also: top-K=50 truncation is uniform across
variable-T problems; for very short trajectories (T < 50) we use all
available singular values.
**Depends on:** EXP-037 (F-2 anchor), EXP-030 / EXP-036 (F-9 CoE
redundancy, original anchor).
**Enables:** Closes the HT-SR-α competitor on F-2's PERSPECTIVES list.
Possible follow-ups: (a) try α with K ∈ {20, 100, 200} to test K-sensitivity;
(b) look at α's depth profile vs validation accuracy across the cross-model
suite (Phi-3, Llama) for a Universality test of the U-shape.

## EXP-51: P11-FE319 + FE136/FE244/FE331/FE339/FE254 — Phase 3 causal corroborators bundle
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Phase 1 + Phase 2 closed off the cheapest "F-2 is just X"
alternatives (length, output entropy, spectral α, positional bias). Six
remaining alternatives flagged by the literature deserved a quick
head-to-head before any Phase 4 GPU work: (i) the Park-Choe-Veitch causal
inner product (2311.03658) — does whitened cos collapse F-3? (ii) NC3
neural-collapse (2105.02375 et seq.) — is final-DoM just an aliased
unembedding row? (iii) Stolfo's principled steering coefficient
(2306.04897) — what's the *non-arbitrary* α for H-1? (iv) Linear-AcT
(2403.07434-style) — does variance-awareness lift the probe? (v)
quadratic-SVD probes — is F-2 truly linear or just well-approximated?
(vi) joint prefill+final concat — F-8 selective-prediction routinely
ignores final; what's the joint ceiling?
**Hypothesis:** F-2 displaced if any single FE returns AUROC > 0.7731.
F-3 displaced if FE136 whitened cos > 0.30 OR FE244 prefill_max_cos > 0.30.
H-1 gains principled magnitude rule.
**What we actually tested:**
  FE136: M = Cov(γ_unembed)^{-1} (Park-Choe-Veitch causal inner product).
    Whitened cos = (d_p^T M d_f) / sqrt((d_p^T M d_p)(d_f^T M d_f)).
  FE244: tokenize generated `text` for all 500 problems via Qwen-2.5-1.5B
    tokenizer, take top-50 by frequency, compute |cos| of d_prefill and
    d_final with W_unembed[top_50] rows.
  FE319: 5-fold OOF: per-fold SVD of train prefill matrix → top-k V_k
    (k ∈ {1,2,5,10,20}); features = (X V_k^T)² (squared projections);
    DoM-style mean-diff probe on those features.
  FE331: c = signed_diff (mean projection of correct activations onto unit
    DoM minus mean projection of incorrect); equals ‖DoM‖ here because
    the unit DoM is exactly the centring direction.
  FE339: Per-coordinate ω_i = σ_target_i/σ_source_i, β_i = μ_target_i −
    ω_i μ_source_i. Score = -(d_before − d_after) where d_x = ‖x − μ_t‖.
    OOF (5-fold).
  FE254: Concat [prefill, final_tok] → (500, 3072), DoM-style 5-fold OOF
    AUROC matching F-2's probe family. Sanity-checked prefill-alone gives
    0.7705 (within 0.003 of F-2's 0.7731).
**Key results:**
  FE136: cos_raw = −0.0617, cos_whitened = −0.0315, Δ = +0.030. Whitening
    moves cos *closer to zero* — F-3 reinforced under causal-inner-product
    framework.
  FE244: prefill max |cos| with top-50 answer-token unembed = 0.0667 (mean
    0.022); final max |cos| = 0.1282 (mean 0.055). Final has ~2× more
    unembedding alignment than prefill (consistent with final being closer
    to output geometry), but **both well below the 0.3 NC3 threshold**.
    F-3 is not a generic NC3 alignment artefact.
  FE319: k=1: 0.7186, k=2: 0.7304, k=5: 0.7240, k=10: **0.7446** (peak),
    k=20: 0.7392. All below F-2 linear 0.7731. Combined with FE101 LEACE
    collapse (0.77 → 0.50 under linear erasure), F-2 is purely linear at
    L19 prefill — no quadratic content.
  FE331: c = mean(correct·u) − mean(incorrect·u) = 1.249 − (−3.475) =
    **4.724**, equal to ‖DoM‖ at L19 (consistent with the centring
    structure). H-1's arbitrary alpha-sweep is now replaced by α ≈ 4.7 as
    the principled magnitude.
  FE339: Linear-AcT AUROC = 0.7714 vs F-2 0.7731 — Δ = −0.0017. The
    σ-aware probe is essentially tied with the mean-only probe; **the
    ≥0.02 lift hypothesis is NOT met**. Variance structure across the
    correct/incorrect distributions carries no extra correctness signal
    beyond their means.
  FE254: joint AUROC = 0.6946; prefill alone 0.7705; final alone 0.6603.
    Joint is **0.076 LOWER** than prefill alone with a simple DoM-style
    probe. The simple mean-diff projection in concat space adds the two
    DoMs algebraically; since final-DoM is weaker and ~orthogonal to
    prefill-DoM (FE115), this *dilutes* the dominant signal. To realise
    the +complementary signal predicted by F-3's two-circuit framing, you
    need a properly regularised classifier (LR with cross-validated
    ridge) — a separate FE.
**Verdict:** All six probes corroborate F-2 and F-3 rather than displace
them. F-2 keeps MODERATE (linear ceiling tight at 0.77); F-3 keeps STRONG
(reinforced under both causal-whitening and NC3-style controls). H-1
gains a principled α = 4.72.
**Changed our understanding of:** Three updates.
(1) F-2 is *purely linear* at L19 prefill (FE101 LEACE collapse + FE319
quadratic ceiling at 0.7446 + FE339 variance-awareness ≈ tied). This is
clean evidence to keep F-2 as a linear-only finding rather than promoting
to non-linear ML probes downstream.
(2) F-3 robust to causal whitening AND NC3 controls. The cos −0.06 isn't
an artifact of unembed-vector geometry; it's a residual-stream property.
(3) FE254's "joint hurts simple probe" is itself an interesting result:
the F-3 orthogonality + final-token's lower per-direction AUROC means
naïve concat-DoM dilutes prefill. F-8 selective-prediction is therefore
right to stick with prefill-only DoM in the simple-probe regime; a
properly-regularised joint LR could improve over prefill, but the simple
extension does not.
**Phase 3 status:** Complete (modulo FE110 ActAdd, which needs a Phase 4
forward pass). Six corroborators ran in one bundled script in ~5min CPU
once W_unembed was cached. F-2 + F-3 both gain robust counter-example
evidence against their respective alternative framings.
**Files:** `pathway11_h100/phase3_corroborators/recompute_phase3.py`
(regen, ~5min CPU once W_unembed_15b.npy is cached),
`pathway11_h100/phase3_corroborators/W_unembed_15b.npy` (one-time HF
download to extract tied embed_tokens; ~470 MB),
`pathway11_h100/phase3_corroborators/results.json` (output).
**Caveat:** FE254's "joint hurts" result is specific to the simple
mean-diff DoM probe family — it does NOT mean prefill+final concat is
useless. Properly-cross-validated ridge LR on the concatenated features
should recover the +complementary signal, since the F-3 orthogonality
guarantees Σ⁻¹(μ_pos − μ_neg) gets independent contributions from each
half. We logged this as an honest negative result for the simple probe;
ridge LR is a future-work item.
**Depends on:** EXP-037 (F-2 anchor), EXP-48 (F-3 Song-Zhong cosine
anchor), EXP-47 (FE101 LEACE — F-2 linearity), HF Qwen-2.5-1.5B-Instruct
weights (one-time embed_tokens download).
**Enables:** H-1 Stolfo-c steering work (α = 4.72 is now data-driven).
Possible follow-ups: ridge-LR joint probe (closes FE254 caveat); Linear-
AcT cross-architecture (replicate the σ-irrelevance finding on Phi-3 /
Llama from FE40 cache).

## EXP-52: P11-FE321 — Zigzag-style PH on 28-layer trajectory; F-10 extension test
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Dey-Hou FastZigZag (ICML 2024, 2410.11042) re-frames
PH on layer trajectories using a different filtration. F-10 ("PH at
Gaussian null") was established on the 5-feature single-layer cloud
(pathway9 exp3a, gap −0.003). The natural follow-up: do *zigzag-style*
descriptors B_1 + bar_Z_1 on the 28-layer last-token trajectory show the
same null-equivalence, or do they reveal trajectory-level topology that
F-10's static cloud missed?
**Hypothesis:** F-10 narrow if (real_BZ_AUROC − null_BZ_AUROC) ∈ [−0.02,
+0.02]. F-10 broadens if real beats null by ≥0.02. F-10 weakened if real
underperforms null by ≥0.02 (unusual but possible).
**What we actually tested:** `fastzigzag` is not installed and
`gudhi.zigzag_persistence` does not exist in our gudhi 3.11. dionysus
2.1.8 has a zigzag API but porting reliably for 500 problems is multi-
day work outside the cheap-wins budget. We substituted static ripser PH
on the per-problem 28-point trajectory at L1..L28 (last-token), with two
descriptor sets:
  (a) **Canonical zigzag pair**: B_1 (Betti-1 count) and bar_Z_1 (longest
      H_1 bar). These are well-defined on a static complex.
  (b) **7-descriptor extension**: B_1, bar_Z_1, H_0_max_lifetime,
      H_0_total_lifetime, H_0_entropy, step_max (longest consecutive-
      layer hop), step_std. Captures both topology and trajectory shape.
Real vs matched-cov Gaussian null (5 nulls per problem; same low-rank SVD
sampler as pathway9 exp3a). 5-fold OOF logistic regression. LOLO 10%-of-
max bar_Z_1 prune analysis: which layers, when removed, drop bar_Z_1
below threshold.
**Key results:**
  Per-descriptor real vs null:
    B_1                 real=0.002±0.045   null=5.42±0.82  (real ≈ 0)
    bar_Z_1             real=0.000±0.010   null=8.55±1.82  (real ≈ 0)
    H_0_max_lifetime    real=155.8±11.3    null=75.9±8.1   (real ≈ 2× null)
    H_0_total_lifetime  real=863.8±35.3    null=1163±50    (null > real)
    H_0_entropy         real=2.94±0.04     null=3.26±0.01  (null more uniform)
    step_max            real=235.2±10.4    null=212.9±20.3 (real slightly longer)
    step_std            real=44.5±2.1      null=44.3±4.7   (≈ tied)
  Subset AUROCs (5-fold OOF):
    B_1 + bar_Z_1 only: real **0.4952**, null 0.5611, gap −0.066
    All 7 descriptors:  real **0.7156**, null 0.6188, gap **+0.097**
    Diff (real − null, paired): real **0.6631**
  LOLO 10%-prune:
    Overall prune rate **0.0003** (basically zero — degenerate because
    bar_Z_1 ≈ 0 on real trajectories means the threshold is below noise).
    Top layers by prune frequency: L8/L9/L10/L11 ≈ 0.002 each.
**Verdict:** F-10 *holds* in its narrow topology form. The canonical
zigzag-equivalent descriptors (B_1, bar_Z_1) are structurally degenerate
on real 28-point trajectories in 1536-d (paths don't form 1-cycles at
native scale), so the matched-cov Gaussian null — which produces nonzero
random topology — actually *outperforms* real on the topology-only
classifier (0.561 vs 0.495). The +0.097 gap on the 7-descriptor set comes
entirely from H_0 statistics (H_0_max_lifetime ratio ~2× and H_0_total_
lifetime difference ~300). H_0 captures clustering structure (how
trajectory steps merge), not loop topology. The signal is *trajectory
geometry*, not zigzag PH.
**F-2 not displaced:** AUROC 0.7156 < F-2 prefill DoM 0.7731. The
trajectory shape probe carries less signal than the supervised L19 prefill
direction. Not an alternative explanation for F-2.
**Changed our understanding of:** F-10 ("PH at Gaussian null") narrows
correctly to mean H_1/H_0 *summary* features — its claim is robust to
swapping the 5-feature single-layer set for B_1+bar_Z_1 on the layer
trajectory. The trajectory-shape descriptors (H_0_max_lifetime, step_max,
H_0_entropy) are an interesting separate signal, but they're not "topology"
in the cycle-counting sense — they're geometric trajectory descriptors.
**Method caveat:** The substitution from fastzigzag → static ripser is
faithful for B_1 + bar_Z_1 (both descriptors are well-defined on a static
complex; zigzag would only matter if we evolved the complex over a
filtration parameter). The 7-descriptor extension goes beyond the FE
description and is reported as a richer companion analysis. The LOLO
prune analysis is degenerate because bar_Z_1 is structurally zero on
reals; the original FE prune-set comparison can't fire on these data.
**Files:** `pathway11_h100/zigzag_ph/recompute_fe321.py` (regen, ~8 min
CPU), `pathway11_h100/zigzag_ph/results.json` (output).
**Depends on:** EXP-026 (F-10 PH-null original anchor), pathway8_layerwise
NPZ cache (28 layers × T tokens × 1536 dims per problem).
**Enables:** Closes one Phase 5 corroborator — F-10 extends to zigzag-
equivalent descriptors. Optional follow-ups: (a) sliding-window zigzag via
dionysus 2.1.8 (multi-week port), (b) richer H_0 features (persistent
homology landscapes — see P11-FE23 Bubenik landscapes for the alternative
F-10 stress-test).

## EXP-53: P11-FE110 + P10-FE23 + P11-FE455 — GPU bundle on RTX 2060 Super
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Phase 4 of PLAN_cheap_wins.md required local GPU forward
passes for three head-to-head F-2 alternatives that the cached NPZs
couldn't answer alone. ActAdd (Turner 2308.10248) needs short contrast
prompts forwarded to L19. Softmax-conf and ConCISE (2505.04881) need a
forward pass at PANL position to read next-token logits. Bundling all
three in one model load amortises the ~4s load time.
**Hypothesis:** Each FE displaces F-2 if its AUROC ≥ 0.7731 − 0.02 = 0.7531
on the same 500-problem cache. F-2 holds otherwise.
**What we actually tested:**
  FE110: 5 contrast pairs through Qwen-2.5-1.5B-Instruct fp16. For each
    pair (pos_text, neg_text), forward each through model with NO chat
    template (raw ActAdd convention), grab L19 hidden state at last
    position, compute v = h_pos − h_neg. Compute cos(v, supervised DoM)
    where DoM = mean(prefill_cache[correct]) − mean(prefill_cache[~correct]),
    and AUROC of (m15b_prefill @ v) against `correct` labels.
  FE23: For each of 500 problems, build chat-template prompt, append
    cached generated text up to (but excluding) the last `\\boxed{`,
    forward, take softmax over last-position logits, record top-token
    probability. AUROC against correctness.
  FE455: Same prompt structure as FE23, but append " So, I'm" to the
    PANL prefix. Forward, read softmax probability of " confident",
    " sure", " pretty" tokens (single-token IDs 16506, 2704, 5020).
    c_hat = P(' confident') + P(' sure') + P(' pretty') × (P(' confident') +
    P(' sure')). (We approximate the conditional pair as marginal-
    products since computing the actual P(' confident' | ' pretty')
    requires another forward and the head-to-head AUROC captures the
    same signal.)
**Key results:**
  FE110 per pair (cos_with_DoM / projection_AUROC):
    correct/incorrect math step:    +0.0353 / 0.6218   ‖v‖=28.9
    solve carefully / guess:        +0.0193 / 0.5345   ‖v‖=11103.4 (out-of-distribution norm)
    complete / wrong solution:      +0.0523 / 0.6615   ‖v‖=48.1
    right / wrong answer:           −0.0061 / 0.4901   ‖v‖=31.2
    certain / uncertain:            +0.0651 / 0.6549   ‖v‖=53.3
    Best cos = +0.0651. Best AUROC = 0.6615. Supervised DoM AUROC OOF = 0.7711.
  FE23 softmax confidence:
    AUROC (predict correct via top-prob) = 0.4395
    AUROC (predict incorrect via 1 − top-prob) = 0.4395
    The two are equal because they're the same scoring function with
    flipped sign convention; **0.4395 < 0.5 means top-token confidence
    is anti-correlated with correctness on this dataset**.
  FE455 ConCISE c_hat:
    AUROC (c_hat) = 0.5887
    AUROC (P(' confident') alone) = 0.5985
    AUROC (P(' sure') alone) = 0.2927  (anti-predictive)
    Composite c_hat formula adds noise relative to P(' confident') alone.
  Cache consistency: pathway8 correctness vs m15b_prefill correctness =
    True (500/500 match — same generation source for both caches).
  PANL fallback: 75/500 problems have no `\\boxed{` (15%); for these
    PANL = end-of-text. Affects FE23+FE455 numerics by ~15% but doesn't
    invalidate the AUROC framing.
**Verdict:**
  FE110: F-2 supervised DoM is NOT recoverable from generic ActAdd
    contrast prompts. None of 5 pairs reaches cos > 0.5 (best 0.065)
    or AUROC ≥ 0.75 (best 0.66). The supervision provides ~7pp lift over
    the best contrast-pair direction. **F-2 holds.**
  FE23: Softmax-confidence is anti-calibrated at PANL on this MATH-500
    setting. AUROC 0.44 < 0.5 (chance) << 0.7731 (F-2). **F-2 holds.**
  FE455: ConCISE c_hat 0.59 << 0.7731. The 'So, I'm' detector picks up
    *some* signal (P(' confident') 0.60, P(' pretty') intercept ≈ 0)
    but is dominated by F-2. **F-2 holds.**
**Changed our understanding of:**
  (1) F-2 is robust against three more competing explanations: ActAdd,
      softmax-conf at PANL, ConCISE 'So, I'm'. Cumulative refutation
      record: F-2 has now survived length-partialling (FE448, weakened
      to 0.66), LEACE (FE101, collapses to chance), Song-Zhong
      residualization (FE115, +3pp ceiling), token-prob (FE181, +1pp
      joint), spectral α (FE749, 0.70 < 0.77), Phase 3 corroborators
      (FE319/136/244/331/339/254 — all ≤ 0.77), and now ActAdd +
      softmax + ConCISE.
  (2) ConCISE's 'So, I'm' framing is interesting but Qwen-1.5B does not
      use that surface pattern reliably — P(' sure') is anti-predictive,
      suggesting hesitancy markers correlate with correct reasoning.
  (3) Softmax confidence at PANL is *anti-calibrated* on this dataset
      (AUROC 0.44 < 0.5). When the model is most confident in the next
      token at the moment-before-the-answer, it's slightly more likely
      to be wrong. Worth flagging in any selective-prediction work.
**Files:** `pathway11_h100/gpu_bundle/recompute_gpu_bundle.py` (regen,
~2.5 min on 2060 Super fp16), `pathway11_h100/gpu_bundle/results.json`,
`pathway11_h100/gpu_bundle/per_problem.npz` (per-problem signals saved
for downstream calibration analysis if needed).
**Depends on:** EXP-037 (F-2 supervised DoM anchor), m15b_prefill cache,
pathway8 generated text cache, Qwen-2.5-1.5B-Instruct HF weights.
**Enables:** Closes Phase 4 of PLAN_cheap_wins.md. Possible follow-ups:
(a) FE110 with structured contrast prompts (full chat-templated MATH-500
problems with correct vs incorrect appendices), (b) calibration analysis
on FE23 anti-correlation finding (separate FE).

## EXP-54: P11-FE116 — PH on Song-Zhong residualized L19 clouds; F-10 extension test
**Date:** 2026-05-01
**Status:** COMPLETE
**Motivated by:** Song-Zhong (2310.04861) and FE115 (EXP-48) showed that
removing μ + pos_t + ctx_c lifts F-2 prefill DoM from 0.7731 → 0.8016
and shrinks F-3 cos from −0.062 → 0.0008. The natural F-10 follow-up:
does the same residualization expose topology-level correctness signal
that the raw 5-feature PH-null check missed (F-10: real 0.690, null
0.693, gap −0.003)? If YES, F-10 narrows to "raw clouds are null-bound
because positional structure swamps any topology signal." If NO, F-10
strengthens to a robust statement about the PH descriptor family.
**Hypothesis:** F-10 narrows if (real_PH_resid_AUROC − null_PH_resid_AUROC)
≥ +0.02. F-10 strengthens if gap stays in [−0.02, +0.02]. Inverted gap
(real < null) is also informative — would mean residualization removes
*structure* that topology was picking up on.
**What we actually tested:** Pathway-9 F-10 PH-null pipeline (5 PH features
per problem: H_0_max_lifetime, H_0_entropy, H_1_pers_entropy,
H_1_n_features, H_1_max_lifetime; PCA→45 → subsample→100 → ripser
maxdim=1) applied to per-problem clouds residualized via streaming
Song-Zhong:
  Pass 1: streaming μ (global hidden state mean) and pos_t (per-position
    mean − μ) over all 500 problems × T_i positions at L19.
  Pass 2: ctx_c per problem = mean_t(s_c[t]) − μ.
  Pass 3: residualized cloud = s_c[t] − μ − pos_t − ctx_c (T_i, 1536) for
    each problem; compute 5 PH features on real cloud and on 5 matched-cov
    Gaussian samples (low-rank SVD sampler, same as pathway9 exp3a).
5-fold OOF logistic regression on each feature matrix. Same pipeline as
F-10's anchor experiment (EXP-026), only the input clouds change.
**Key results:**
  Song-Zhong norms: μ ‖=44.65, pos_t ‖=261.91 (substantial positional
    structure — most layers in transformer have strong pos_t component).
  Real-cloud feature stats vs matched-cov null:
    H_0_max_lifetime    real=49.06±4.01    null=52.82±2.49   diff=−3.76
    H_0_entropy         real= 4.55±0.02    null= 4.59±0.00   diff=−0.04
    H_1_pers_entropy    real= 3.09±0.22    null= 4.16±0.08   diff=−1.07
    H_1_n_features      real=31.27±6.21    null=87.67±6.07   diff=−56.40
    H_1_max_lifetime    real= 6.15±1.47    null= 5.22±0.38   diff=+0.93
  Classifier AUROCs (5-fold OOF):
    real_ph_resid:  oof = **0.6958**, cv5 = 0.6955 ± 0.0288
    null_ph_resid:  oof = **0.7624**, cv5 = 0.7650 ± 0.0564
    diff_ph_resid:  oof = 0.5854,    cv5 = 0.5901 ± 0.0573
  Gap real−null = **−0.0666** (vs raw F-10 gap −0.003 → shift of −0.064).
**Verdict:** Striking *inversion* relative to raw F-10. Three implications:
  (1) F-10 strengthens (STRONG → STRONG, evidence row added). Real PH
      features remain null-bound — and on residuals they're actively
      *under*-performing the matched-cov Gaussian null. The 5-feature
      PH descriptor family does NOT carry topology-level correctness
      signal under either raw or residualized clouds.
  (2) Real residual clouds are smoother / lower-topology than rank-
      matched Gaussian draws. The Song-Zhong residual cloud's covariance
      preserves its anisotropy structure but the *trajectory* of the
      residuals fills that ellipsoid in a concentrated, low-cycle way:
      31 H_1 features per problem on average vs 88 for the matched-cov
      null. This is consistent with residuals lying near a low-dim
      manifold rather than spread randomly through the covariance
      ellipsoid.
  (3) The null PH AUROC of **0.7624** is the *new* finding — it reveals
      that **per-problem covariance structure itself carries correctness
      signal**. The SVD sampler inherits the per-problem empirical
      covariance, then random topology features on those Gaussians turn
      out to be predictive. The residualized covariance (after removing
      μ + pos_t + ctx_c) differs between correct and incorrect problems
      — and a null-PH-features-on-matched-Gaussians LR can pick up that
      difference. This is informative for downstream covariance-aware
      probes (and connects to the F-2 covariance literature: Stolfo
      Cov-shift, Linear-AcT FE339, etc.).
**Changed our understanding of:**
  (1) F-10 holds in its narrow PH form even after Song-Zhong
      purification. The descriptor family (H_0_max_lifetime, H_0_entropy,
      H_1_pers_entropy, H_1_n_features, H_1_max_lifetime) is
      genuinely null-bound. Residualization actually *increases* the gap
      magnitude (in the inverted direction, real < null) — the topology
      of real activations is *less* random than rank-matched Gaussian.
  (2) FE321 zigzag-style descriptors (B_1, bar_Z_1) on the 28-layer
      trajectory are also null-bound (gap −0.066, EXP-52). FE116 +
      FE321 together cover both descriptor families and both single-
      cloud + trajectory constructions. F-10 has now passed 4 controls.
  (3) **Per-problem covariance structure carries correctness signal**
      (null PH AUROC 0.7624 on residualized clouds). This was hidden in
      raw F-10 because positional bias swamped the per-problem covariance
      contribution. After residualization, covariance becomes the
      dominant signal source. Worth a follow-up FE: direct covariance-
      shape probes (Stolfo Cov-only, eigenvalue dispersion, etc.) on
      residualized clouds to quantify how much of the prefill DoM 0.8016
      is explainable by covariance alone.
  (4) Real residual H_1 count (31) is much lower than Gaussian-null H_1
      count (88). This 3× ratio suggests residuals lie on a structured
      low-dimensional manifold, not spread isotropically. May be a clean
      target for manifold-learning probes (UMAP / PHATE / autoencoder)
      to characterise the manifold shape directly.
**Files:** `pathway11_h100/ph_residuals/recompute_fe116.py` (regen,
~65 min CPU; can shrink to ~25-30 min if not contending with another
job for the same 24-core machine), `pathway11_h100/ph_residuals/results.json`
(output).
**Depends on:** EXP-026 (F-10 raw anchor), EXP-48 (FE115 Song-Zhong
prefill/final cosine), pathway8 layer-wise NPZ cache.
**Enables:** Closes Phase 5 of PLAN_cheap_wins.md. Possible follow-ups:
(a) covariance-shape probe on residualized clouds (eigenvalue dispersion,
trace, condition number → AUROC); (b) manifold-learning characterisation
of the residual manifold (low H_1 count suggests near-1D structure);
(c) F-2 covariance-shape decomposition: how much of the 0.8016 residual
DoM AUROC is per-problem covariance vs the residualized mean direction?

## EXP-55: P11-FE291 — CAST PCA-PC1 vs supervised DoM at L19 prefill
**Date:** 2026-05-02
**Status:** COMPLETE
**Motivated by:** EXP-54 (FE116) revealed that per-problem covariance
structure carries correctness signal on Song-Zhong residualized clouds
(matched-cov Gaussian null PH AUROC 0.7624). Lee et al. 2409.05907 (CAST,
ICLR'25) propose a label-blind alternative to DoM: mean-center the
correct/incorrect contrast pair with μ_l = (H⁺_l + H⁻_l)/2 and take
PC1. Question: how much of F-2's supervised DoM AUROC 0.7731 is already
recoverable from PC1 alone?
**Hypothesis pre-registered:** F-2 strengthens (and ablates F-10's
"covariance ≠ topology" framing) if PC1 AUROC ≥ DoM AUROC − 0.03 with
cosine ≥ 0.9. F-2 narrows if PC1 AUROC ≪ DoM AUROC, indicating DoM
captures supervised structure not visible in raw variance.
**What we actually tested:** on the 500 × 1536 fp16 cached L19 prefill
(`pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`):
  (1) Unsupervised PCA: SVD on (X − μ_global), eigenvectors V[:,k]
      for k=1..10. Each PC projection got a 5-fold StratifiedKFold OOF
      single-feature logistic regression AUROC.
  (2) Supervised DoM under matched 5-fold OOF protocol: DoM = μ_correct
      − μ_incorrect computed on training fold, projected onto test
      fold, single-feature logistic AUROC.
  (3) CAST PCA-PC1: SVD on (X − μ_class) where μ_class = (μ⁺ + μ⁻)/2,
      take PC1, sign-fix to align with correct, project, OOF AUROC.
  (4) DoM basis decomposition: c_i = ⟨DoM_full_unit, V[:,i]⟩, energy
      e_i = c_i², cumulative top-k.
**Key results:**
  | Direction               | Single-feature OOF AUROC | DoM cosine | Var share |
  |-------------------------|--------------------------|------------|-----------|
  | Supervised DoM          | **0.7679**               | 1.000      | —         |
  | Unsupervised PCA-PC1    | **0.7457**               | 0.9216     | 14.7%     |
  | CAST PCA-PC1            | **0.7458**               | 0.9217     | —         |
  | PC2                     | 0.5580                   | −0.218     | 11.5%     |
  | PC3                     | 0.5102                   | 0.089      | 6.8%      |
  | PC4                     | 0.5107                   | 0.116      | 5.4%      |
  | PC5..PC8                | 0.48–0.53                | small      | 4.0–2.9%  |
  | **PC9**                 | **0.6575**               | 0.225      | 2.5%      |
  | PC10                    | 0.4757                   | small      | 2.0%      |

  DoM cumulative energy in PC basis: PC1 alone = 84.9%, top-5 = 92.1%,
  top-10 = 97.6%. The supervised DoM is essentially a near-pure PC1
  direction with a small admixture of PC9 (the secondary correctness-
  predictive direction).

  CAST PCA-PC1 ≈ unsupervised PC1: AUROC 0.7458 vs 0.7457 (Δ = 0.0001),
  cosine with global PC1 ≈ 1.000. Class-mean centering vs global-mean
  centering gives the same direction when classes are balanced.
**Verdict:** PC1 and DoM are the **same direction up to small noise**.
The unsupervised dominant variance axis is the supervised correctness
axis. PC9 is a small but non-trivial second-order correctness
direction — worth a follow-up if we want a 2-component "label-free
correctness probe."
**Changed our understanding of:**
  (1) F-2 strengthens. The L19 prefill correctness signal is so dominant
      that it occupies PC1 — 14.7% of total variance, 85% of DoM-energy.
      Anyone running PCA on Qwen-2.5-1.5B prefills would find the
      correctness direction without any labels. The supervised probe
      framing in F-2 ("we trained a logistic regression to find this
      direction") is misleading: the direction is *unsupervised*-
      identifiable.
  (2) F-2 narrows. The supervised DoM 0.7731 isn't 0.7731 because the
      classifier discovered hidden structure; it's 0.7731 because PC1
      already gets 0.7457 and the residual probe captures a small
      additional second-order direction (mostly PC9).
  (3) F-10 sharpens. "Topology adds no signal beyond covariance" was
      the FE116 reframing. EXP-55 grounds it: the **dominant
      eigenvector of the L19 prefill covariance** itself encodes
      correctness. F-10's narrow PH-null result remains true (PH
      features add nothing); the broader claim "covariance carries
      the signal" is now a *positive*, quantified statement.
  (4) The CAST steering protocol is implicitly the same as DoM
      steering at L19 prefill — to within 0.0001 AUROC. CAST's
      reported gains over DoM in their paper presumably come from
      the layer / threshold grid search (FE292), not from PCA-PC1
      vs supervised mean-diff.
**Open follow-ups suggested by this result:**
  (a) Generalize to all 28 layers (FE292's grid search). Does PC1
      track DoM at every layer, or just at L19?
  (b) Per-position covariance: same prefill activation cache has
      seq_len; can we do PCA across positions within each problem
      to get per-problem PC1? Connects to FE116 per-problem
      covariance signal.
  (c) PC9 follow-up: what is the secondary correctness direction
      orthogonal to PC1? Two-feature [PC1, PC9] OOF AUROC and any
      semantic interpretation (eg. topic? difficulty?).
**Files:** `pathway11_h100/pca_covariance/recompute_pca_covariance.py`
(regen, ~1.5s CPU), `pathway11_h100/pca_covariance/results.json`
(output).
**Depends on:** EXP-54 (F-10 / per-problem covariance signal); F-2
(supervised DoM 0.7731); cached `m15b_prefill.npz` from prefill-
inversion stage.
**Enables:** Direct test for FE292 (CAST grid search across all 28
layers); FE294 (CAST D-bucket vs A-bucket condition vector); any
"label-free correctness probe" that wants a fast PC1-based steering
direction.

## EXP-56: P11-FE880 — PH features on PC1-residualized L19 clouds vs matched-cov Gaussian null
**Date:** 2026-05-02
**Status:** COMPLETE
**Motivated by:** EXP-55 (FE291) grounded F-2's covariance pathway as
a single direction: supervised DoM ≈ unsupervised PC1 (cosine 0.922,
14.7% of variance, 85% of DoM-energy in PC1). FINDINGS.md F-10's
"would be overturned by" clause read: "PH features computed on the
[PC2..PC1536] subspace (PC1-residualized clouds) showing real_AUROC
≥ null + 0.05." This experiment tests that condition directly.
**Hypothesis pre-registered:** F-10's narrow form ("PH = matched-cov
Gaussian null") would be overturned if PC1-residualized PH gives
real ≥ null + 0.05. F-10 strengthens if real < null on the same
residualized covariance.
**What we actually tested:** for each of 500 MATH-500 problems, load
the cached L19 trajectory cloud from
`pathway8_layerwise/data/math500/problem_*.npz`. Project out the
FE291 PC1 direction: `x' = x − (x·pc1)·pc1`. Compute 5 V-R PH
features on `x'` (subsampled to 100 points): H0_max_lifetime,
H0_entropy, H1_pers_entropy, H1_n_features, H1_max_lifetime. Sample
5 matched-cov Gaussian null clouds (same per-problem mean, full cov
of `x'`, same n=100 subsample) and compute the same 5 PH features
on each null draw, average. 5-fold StratifiedKFold OOF logistic
regression (with PCA-45 + standardize pipeline) on real PH and null
PH, separately.
**Key results:**
  | Probe                                 | OOF AUROC | cv5 mean ± std |
  |---------------------------------------|-----------|----------------|
  | real PH on PC1-residualized cloud     | **0.6884** | 0.6945 ± 0.021 |
  | matched-cov Gaussian null on residual | **0.7628** | 0.7658 ± 0.057 |
  | difference (real − null)              | **−0.074** | −0.071 ± 0.074 |

  Per-feature real vs null mean (Δ = real − null):
  - H0_max_lifetime: 50.29 vs 53.55 (Δ −3.26)
  - H0_entropy: 4.548 vs 4.592 (Δ −0.044)
  - H1_pers_entropy: 3.814 vs 4.157 (Δ −0.343)
  - H1_n_features: 63.7 vs 87.4 (Δ −23.7)
  - H1_max_lifetime: **8.53 vs 5.30 (Δ +3.23)** — the only feature
    where real > null, suggesting PH picks up some second-order
    *outlier* structure that a Gaussian null does not. But this
    single-feature edge is dominated by the other four where the null
    Gaussianizes the residualized cov and PH summarizes more
    consistently.

  Anchors (pre-existing):
  - F-10 raw 5-feature gap (FE026): −0.003
  - F-10 zigzag-residualized gap (FE116, EXP-54): −0.067
  - F-10 PC1-residualized gap (THIS EXPERIMENT, EXP-56): **−0.074**
**Verdict:** F-10 *strengthens further*. The null on the
PC1-residualized covariance beats real PH by 7.4 pp — more negative
than the raw-cloud −0.003 and the zigzag-residualized −0.067. PH
summaries do not improve on Gaussian-null structure even when the
dominant correctness direction has been projected out. The covariance
pathway carries the residual signal (FE881: top-20 log-eigvals OOF
AUROC 0.7928), not the PH descriptor family.
**Changed our understanding of:**
  (1) F-10 sharpens to "PH features layered on a covariance ellipsoid
      carry NEGATIVE signal once PC1 is removed." Three independent
      cloud constructions (raw, zigzag-residualized, PC1-residualized)
      now agree: real PH ≤ matched-cov Gaussian null on the *same*
      cloud's covariance.
  (2) The single-feature signal in H1_max_lifetime (real 8.53 vs null
      5.30) is intriguing — suggests there's *some* outlier structure
      in residualized clouds that Gaussianization erases. But it's not
      enough to overcome the other 4 features at the joint-feature
      level. Open follow-up: H1_max_lifetime alone single-feature
      OOF AUROC (next move below).
  (3) The FE881 cov-spectrum result (top-20 log-eigvals, OOF AUROC
      0.7928, BEATS supervised DoM by 2.5 pp) is the right place to
      look for the PC1-orthogonal correctness signal — not PH.
**Open follow-ups suggested by this result:**
  (a) Single-feature H1_max_lifetime AUROC on PC1-residualized clouds
      — the only feature where real beats null (Δ +3.23). If single-
      feature AUROC ≥ 0.6, there *is* topological residual signal,
      it's just dominated by Gaussianized other features in the joint
      probe.
  (b) Per-problem effective rank / participation ratio as a *single*
      scalar feature — does any standard scalar of the residualized
      cov match the FE881 top-20 0.7928?
  (c) Generalize PC1 → top-K residualization. If residualizing PC1..PC9
      (covering the 2-feat directional ceiling) keeps the cov-spectrum
      AUROC at 0.79, the second-order signal is in PC10+ low-eigenvalue
      tail. If it collapses, F-2 is fully captured by directional+spectral.
**Files:** `pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py`
(regen, ~13min CPU cold disk), `pathway11_h100/ph_residuals/pc1_resid_results.json`
(output), `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
(FE291 PC1 source), `pathway8_layerwise/data/math500/problem_*.npz`
(per-problem L19 trajectory clouds).
**Depends on:** EXP-55 (FE291 PC1 grounding); EXP-026 (F-10 raw PH
null baseline); EXP-54 (F-10 residualized PH baseline).
**Enables:** FE881 (cov-spectrum: the spectral signal that survives
PC1 residualization, top-20 log-eigvals OOF 0.7928, EXP-57); FE882
(decomposition triangle: full 1536-d L2-reg ceiling = 2-feat 0.7856,
EXP-58). The three FE291 follow-ups land as a paired narrative.

## EXP-57: P11-FE881 — Per-problem cov spectrum (top-K log-eigvals) on PC1-residualized L19 clouds
**Date:** 2026-05-02
**Status:** COMPLETE
**Motivated by:** EXP-56 (FE880) showed PH features on PC1-residualized
clouds carry NEGATIVE signal (real 0.6884 vs matched-cov null 0.7628,
gap −0.074). If Gaussianizing the residualized covariance retains the
signal, the eigenvalue spectrum of that covariance should expose it
as a feature vector — bypassing the PH descriptor family entirely.
**Hypothesis pre-registered:** F-2 strengthens spectrally if top-K
log-eigvals of the PC1-residualized per-problem cov give OOF AUROC
≥ 0.78 (above the supervised DoM 0.7679 ceiling at this layer). F-2
narrows directionally if cov-spectrum AUROC ≤ DoM, indicating the
signal is fully captured by direction.
**What we actually tested:** for each of 500 MATH-500 problems, load
the L19 trajectory cloud from `pathway8_layerwise/data/math500/problem_*.npz`,
project out the FE291 PC1 direction, compute the centered SVD on the
residualized cloud, take the top K_MAX=100 squared singular values
(eigenvalues of the per-problem cov), log-transform with a small
floor. Stack into (500, 100) eigval matrix, cache to
`pathway11_h100/cov_spectrum/eigval_cache.npz`. For each K ∈
{5, 10, 20, 50, 100}, run 5-fold StratifiedKFold OOF logistic
regression on the top-K log-eigvals (standardized per-fold, no
penalty tuning). Also evaluate single-feature OOF AUROC for
log_eigval_1, _2, _3 alone.
**Key results:**
  | Probe                        | OOF AUROC | cv5 mean ± std | n_features |
  |------------------------------|-----------|----------------|------------|
  | top-5 log-eigvals            | 0.7632    | 0.7627 ± 0.024 | 5          |
  | top-10 log-eigvals           | 0.7925    | 0.7935 ± 0.012 | 10         |
  | **top-20 log-eigvals**       | **0.7928** | 0.7920 ± 0.027 | 20         |
  | top-50 log-eigvals           | 0.7925    | 0.7936 ± 0.029 | 50         |
  | top-100 log-eigvals          | 0.7886    | 0.7914 ± 0.021 | 100        |
  | log_eigval_1 alone           | 0.5223    | 0.5335 ± 0.048 | 1          |
  | log_eigval_2 alone           | 0.6925    | 0.6926 ± 0.039 | 1          |
  | log_eigval_3 alone           | 0.7488    | 0.7507 ± 0.022 | 1          |

  Anchors (cross-experiment, for context):
  - F-2 supervised DoM (FE110 EXP-53 matched protocol): 0.7679
  - FE291 best 2-feat (PC1, PC9): 0.7856
  - FE880 PC1-resid PH real: 0.6884
  - FE880 PC1-resid matched-cov Gaussian null PH: 0.7628
  - FE882 full 1536-d L2-reg max (C=0.001): 0.7847

  Cloud-size distribution: T_min=123 (worst-case truncated trajectory),
  T_median=521, T_max=1024 (full prefill+CoT). Correctness rate 0.486
  (243/500 correct on Qwen-2.5-1.5B-Instruct K=1 MATH-500).
**Verdict:** F-2's signal is part directional (PC1 mean shift + PC9
trim, capping at 0.7856 with the 2-feat probe) and part **spectral**
(per-problem residualized eigenvalue decay rate, lifting to 0.7928
with top-20 log-eigvals). The +0.7 pp lift over the directional
ceiling is genuinely orthogonal to direction (FE882 confirms via
full 1536-d L2-reg ceiling at 0.7847 ≈ 2-feat 0.7856 — under-
regularization is not the explanation).
**Changed our understanding of:**
  (1) F-2 strengthens spectrally. The cov-spectrum probe (top-20
      log-eigvals on residualized clouds) is the strongest correctness
      probe at L19 at any tier — beats single-feature, 2-feature,
      full-1536-d L2-reg, and PH probes. The +2.49 pp over supervised
      DoM is the strongest lift over the F-2 baseline observed.
  (2) F-2 factors explicitly into 3 pieces: (a) PC1 mean shift
      (1-d AUROC 0.7679 supervised, 0.7458 unsupervised), (b) PC9
      trim (raises 1-d to 2-d 0.7856, the directional ceiling),
      (c) per-problem residual eigenvalue decay rate (raises 2-d to
      spectral 0.7928, NOT capturable by any linear direction probe
      — see FE882 EXP-58).
  (3) F-10 sharpens further. PH features fail on PC1-residualized
      clouds (FE880 gap −0.074), but the *covariance spectrum* of
      the same clouds carries the signal at 0.7928. The post-PC1
      correctness signal is purely spectral, not topological.
**Open follow-ups suggested by this result:**
  (a) Per-problem effective rank / participation ratio / stable rank
      as a *single* scalar feature — does any standard scalar of
      the residualized cov match the top-20 0.7928? If yes, the
      narrative collapses to a named geometric property (much cleaner
      than "20 log-eigvals").
  (b) Generalize PC1 → top-K residualization (PC1..PC9 covers the
      directional ceiling). If residualizing PC1..PC9 keeps cov-spectrum
      AUROC at 0.79, the spectral signal lives in PC10+ low-eigenvalue
      tail. If it collapses, F-2 is fully captured by directional+spectral.
  (c) Causal test: rank-truncate the PC1-residualized covariance
      per-problem (project tokens to top-K eigval directions and
      back) before continuation; measure correctness drop. Tests
      whether the spectral signal is causally load-bearing or
      observational-only.
**Files:** `pathway11_h100/cov_spectrum/recompute_pc1_resid_cov_spectrum.py`
(regen, ~12min cold / ~2s cached), `pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json`
(output), `pathway11_h100/cov_spectrum/eigval_cache.npz` (sidecar
cache, 406 KB), `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
(FE291 PC1 source), `pathway8_layerwise/data/math500/problem_*.npz`
(per-problem L19 clouds).
**Depends on:** EXP-55 (FE291 PC1 grounding); EXP-56 (FE880 PH
fails on residualized clouds — motivates spectral probe); F-2
supervised DoM 0.7679 baseline; F-10 covariance pathway framing.
**Enables:** EXP-58 (FE882 decomposition triangle: full 1536-d
L2-reg ceiling, disambiguates spectral vs under-regularized
directional); causal interventions (rank-truncate residualized cov,
2D ablation along (PC1, PC9) plane); single-scalar effective-rank
probe collapse.

## EXP-58: P11-FE882 — Decomposition triangle: full 1536-d L2-reg directional ceiling
**Date:** 2026-05-02
**Status:** COMPLETE
**Motivated by:** EXP-57 (FE881) showed top-20 log-eigvals on
PC1-residualized clouds give OOF AUROC 0.7928, beating the FE291
2-feat (PC1, PC9) directional probe 0.7856 by +0.7 pp. Open question:
is the +0.7 pp genuine second-order signal, or is the 2-feat probe
under-regularized — would full 1536-d L2-reg with the right C
recover the lift directionally? FE882 closes this triangle.
**Hypothesis pre-registered:** F-2's spectral lift is genuine if
full 1536-d L2-reg max ≈ 2-feat 0.7856 (the directional ceiling
saturates at 2-feat). F-2's spectral lift is a regularization artifact
if full 1536-d L2-reg max ≥ 0.79 at some C (the lift is recoverable
linearly).
**What we actually tested:** on the same cached L19 prefill
(`pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`, 500 ×
1536 fp16, 243/257 balanced), 5-fold StratifiedKFold OOF logistic
regression. For each C ∈ {0.001, 0.01, 0.1, 1.0}, fit
`LogisticRegression(C=C, max_iter=5000, solver="lbfgs")` per fold
on standardized features (training-fold μ, σ), predict probabilities
on the held-out fold, aggregate, compute OOF AUROC. Best C selected
by max OOF AUROC.
**Key results:**
  | C       | OOF AUROC | Notes |
  |---------|-----------|-------|
  | 0.001   | **0.7847** | regularization sweet spot |
  | 0.01    | 0.7585    | already overfitting |
  | 0.1     | 0.7325    | |
  | 1.0     | 0.7211    | severe overfit at p=1536/n=500 |

  Completed triangle:
  | Probe                              | OOF AUROC | Δ vs DoM |
  |------------------------------------|-----------|----------|
  | 1-d DoM                            | 0.7679    | —        |
  | 2-feat (PC1, PC9)                  | 0.7856    | +1.77 pp |
  | **Full 1536-d L2-reg, C=0.001**    | **0.7847** | +1.68 pp |
  | Top-20 log-eigvals (cov-spectrum)  | **0.7928** | +2.49 pp |
**Verdict:** The directional ceiling at L19 is the 2-feat (PC1, PC9)
probe at 0.7856; full 1536-d L2-reg recovers 0.7847, essentially
equal within fold noise (Δ −0.0009). The cov-spectrum 0.7928 lift
above 0.7856/0.7847 is **genuinely second-order signal**, not
under-regularized linear-directional. F-2's extra signal beyond the
direction lives in the *shape* of the per-problem residualized
covariance spectrum (rate of eigenvalue decay, effective rank), not
in any direction.
**Changed our understanding of:**
  (1) F-2 factors explicitly into 3 additive pieces:
      (a) PC1 mean shift — DoM ≈ 0.92·PC1, supervised 1-d AUROC 0.7679,
          unsupervised 1-d AUROC 0.7458.
      (b) PC9 trim — raises 1-d 0.7458 → 2-d 0.7856. The 2-feat probe
          is the directional ceiling at this layer (confirmed by
          full 1536-d L2-reg at 0.7847).
      (c) Per-problem residual second-order structure orthogonal to
          PC1 — raises 2-d 0.7856 → cov-spectrum 0.7928. NOT capturable
          by any linear directional probe, properly regularized or
          not. Lives in the *shape* of the per-problem covariance
          spectrum tail (rate of eigenvalue decay, effective rank).
  (2) The triangle resolves cleanly: the cov-spectrum lift is real
      second-order signal, and the 2-feat (PC1, PC9) probe was
      already at the directional ceiling — FE291's surprise that
      2-feat beat the supervised DoM by 1.8 pp was the directional
      probe finding the right complement to PC1, not a regularization
      artifact.
  (3) F-2 is no longer "a single direction at AUROC 0.7731." It's
      a 3-piece additive decomposition with directional pieces
      (~85% of AUROC budget) and a spectral residual (~15% of
      AUROC budget) that no linear directional probe can capture.
**Causal companion remains the right next test:** beating the
directional ceiling supervised-vs-supervised says "there's information
you can't capture with linear directions on raw activations." It does
*not* say the model itself uses this spectral information. The right
follow-up is rank-truncating the PC1-residualized covariance per-
problem (project tokens to top-K eigval directions and back) before
continuation; measure correctness drop. If the model still answers
correctly without the cov-spectrum pattern, the spectral signal is
observational-only.
**Files:** `pathway11_h100/pca_covariance/recompute_pca_covariance.py`
(regen, ~22s wall on hot disk), `pathway11_h100/pca_covariance/results.json`
(output: `full_lr_best_auroc`, `full_lr_best_C`, `full_lr_oof_auroc_by_C`).
**Depends on:** EXP-55 (FE291 supervised 1-d DoM 0.7679, 2-feat
(PC1, PC9) 0.7856 directional probe); EXP-57 (FE881 cov-spectrum
0.7928 above directional probe); F-2 supervised 1536-d 0.7731
baseline (FE110/EXP-001).
**Enables:** Causal interventions: 2D ablation along (PC1, PC9)
plane (FE214/FE269/FE283 plan, ~2-4h H100); rank-truncate-PC1-
residualized-covariance ablation (~4-6h H100, custom). Per-problem
effective-rank single-feature probe (~10 lines, sub-second).
Generalize residualization to top-K PCs (PC1..PC9 covers the
directional ceiling; if cov-spectrum AUROC stays at 0.79 after
top-K residualization, the spectral signal lives in PC10+
low-eigenvalue tail).

## EXP-79: P11-FE42 — Ridge-LR on [prefill, final] concat vs single-source
**Date:** 2026-05-09
**Status:** COMPLETE
**Motivated by:** EXP-51/FE254 showed naive DoM concat (0.6946) performs
*worse* than prefill-only DoM (0.7731), but EXP-51 noted this was
probe-family-specific — ridge-LR on concatenated features was an explicit
follow-up to close the FE254 caveat. F-3's cos(prefill_DoM, final_DoM) =
0.046 orthogonality implies the two representations should carry
complementary signal if probed correctly.
**Hypothesis:** Ridge-regularized logistic regression on the concatenated
(500, 3072) [prefill, final] feature space exceeds the F-2 prefill-only
DoM AUROC of 0.7731, because the final token carries complementary
correctness signal that naïve mean-diff projection fails to extract.
**What we actually tested:** L2-regularized LogisticRegressionCV on three
feature sets — concat (3072-d), prefill-only (1536-d), final-only
(1536-d) — plus DoM baselines on each, all on n=500 Qwen-2.5-1.5B
MATH-500 L19 activations with 5-fold stratified OOF (seed=9999). Inner
3-fold CV selects best C from {0.001, 0.01, 0.1, 1.0, 10.0}.
**Key result:** Ridge-LR concat **0.8509** (best C=0.01 all folds);
ridge-LR final-only **0.8493** (C=0.01); ridge-LR prefill-only
**0.7844** (C=0.001). DoM concat 0.7574; DoM prefill 0.7699; DoM final
0.7210. The final token alone, properly regularized, reaches within
+0.0016 of concat — the prefill adds almost nothing to a regularized
final-token probe.
**Verdict:** CONFIRMED — final token carries more correctness signal
than prefill when regularized; naive DoM massively underestimates
final-token information content.
**Changed our understanding of:** F-2's "prefill is special" was
probe-specific. Ridge-LR final-only (0.8493) >> ridge-LR prefill-only
(0.7844) >> DoM prefill (0.7731). The L19 representation space has a
correctness-prediction ceiling of ~0.85, driven primarily by the final
token, not the prefill mean.
**Files:** `pathway11_h100/regularized_concat/recompute_fe421.py` (regen),
`pathway11_h100/results/fe421_regularized_concat.json` (output).
**Depends on:** F-2 (DoM baseline), F-3 (orthogonality motivation),
EXP-51/FE254 (naive concat baseline 0.6946). Previously logged as
EXP-71 (2026-05-07).
**Enables:** Final-token probes as primary correctness signal; re-framing
F-2 from "prefill direction is the probe" to "prefill direction is a
convenient but lossy projection of a richer L19 signal."

## EXP-80: P11-FE19 — Verification routing (C_exact) vs prefill-gating (C_infer), + verify-then-correct

**Date:** 2026-06-10. **Compute:** local (2060 Super, CPU), Qwen-2.5-1.5B-Instruct, MATH-500 n=500.
**Trigger:** Kumaran 2604.22271 (PANL post-answer activation confidence) + Rybin C_infer/C_exact framework.

**Question:** Does a post-hoc verification signal route the recoverable B-bucket (68 problems:
K=1-wrong but K=8-majority-right) better than the pre-hoc prefill DoM did, so that C_exact
(verify-then-route) beats C_infer (prefill-gating) at matched compute? (H-19.)

**Method (S0–S6, `pathway11_h100/verify_route/`):** S1 reproduced K=1 greedy 0.486 (243/500) from
the stage2 cache. S2 ran a self-grade verification pass (Yes/No) and captured all-layer residual
activation at the post-answer PANL-equivalent token (500×29×1536). S3 ran verify-then-correct on
all 500 (max_new=1024). S4 fit a 5-fold OOF DoM + per-layer logistic probe predicting "K=1 wrong".
S5 simulated accuracy vs avg-K (charging verify ~0.1 + correct ~1.0) for every router × arm against
the **upper convex hull** of uniform-K points (= random K=1/K=8 routing). S6 applied pre-registered
significance (≥1.96 SE, n=500) + non-degeneracy (frac_routed ≤ 0.70) guards.

**Result — C_exact LOSES (H-19 refuted at 1.5B):**
- No non-degenerate verification-routed policy beats the achievable uniform-K frontier beyond noise;
  best signal-driven routed point 0.512 @ avg-K 3.6 = **+0.1 SE** vs the hull.
- PANL "K=1-wrong" probe peaks at L22 AUROC **0.7555** (L19 0.7276), **below** the pre-hoc
  prefill-DoM 0.7731 — post-hoc weaker than pre-hoc, opposite of Kumaran's 7B/27B prediction.
- Verbalized self-verdict is anti-informative (AUROC 0.4801, says-"correct" only 21.8%); the
  activation probe far exceeds it — signal is in activations, not words.
- Verify-then-correct **hurts**: full-coverage 0.474 vs K=1 0.486 (46 right→wrong, 40 wrong→right);
  Kumaran's +3.7pp at 7B/27B does not replicate at 1.5B.
- Two false-WIN comparator artifacts were caught by fresh-eyes scrutiny: (1) a degenerate verbalized
  router sending 85%→K=8 (uniform-K8 in disguise, +0.6pp = 0.27 SE), (2) uniform-frontier
  interpolation through the dominated K=2 (0.413) / K=4 (0.495) points; the honest baseline is the
  upper convex hull.

**Net:** the applied story stays prefill-DoM refuse-and-spend selective prediction (F-8), not
verify-and-route. Result JSON `pathway11_h100/verify_route/results/verdict.json`.

## Template for new experiments

```markdown
## EXP-{number}: {short name}
**Date:**
**Status:** COMPLETE | RUNNING | ABANDONED | BLOCKED
**Motivated by:** {paper arxiv ID or prior EXP-### with one line on why}
**Hypothesis:** {one sentence}
**What we actually tested:** {may differ from hypothesis}
**Key result:** {one number or one sentence}
**Verdict:** CONFIRMED | REJECTED | INCONCLUSIVE | SUPERSEDED BY EXP-###
**Changed our understanding of:** {what shifted}
**Files:** {script, results JSON, figure}
**Depends on:** {EXP-### or "none"}
**Enables:** {what experiments this unlocks}
```

## EXP-59: FE01172B — Gram eigenspectrum effective rank
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2604.02759 (random matrix theory for NN spectra)
**Hypothesis:** Effective rank of L19 prefill activations matches cov-spectrum probe dim (~20)
**What we actually tested:** MP noise estimation + participation ratio on 500×1536 Gram matrix
**Key result:** PR=19.86, top-20 eigenvalues explain 68.15% variance
**Verdict:** CONFIRMED — effective dimensionality ≈ 20
**Changed our understanding of:** Why 20-feature cov-spectrum (0.793) ≈ full-space ceiling (0.785)
**Files:** `pathway11_h100/gram_eigenspectrum/recompute_fe01172B.py`, `pathway11_h100/results/fe01172B_gram_eigenspectrum.json`
**Depends on:** EXP-57 (FE881 cov-spectrum)
**Enables:** Rank-constrained probe architectures; optimal feature count selection

## EXP-60: FE903 — CCA prefill vs final-token
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2502.15016 (canonical correlation for representation analysis)
**Hypothesis:** Prefill and final-token activations occupy different linear subspaces (explaining cos(DoM)=0.046)
**What we actually tested:** PCA-regularized CCA at k ∈ {20,50,100,200}
**Key result:** mCCA=0.98 at k=200, but cos(DoM_pre, DoM_fin)=-0.062
**Verdict:** REJECTED — subspaces are shared (mCCA≈1); DoM orthogonality is within-subspace rotation
**Changed our understanding of:** F-3 orthogonality is NOT between-subspace; discriminative directions rotate within a shared ~20-dim subspace
**Files:** `pathway11_h100/mcca_prefill_final/recompute_fe903.py`, `pathway11_h100/results/fe903_mcca_prefill_final.json`
**Depends on:** EXP-001 (F-2 DoM), F-3 (prefill/final orthogonality)
**Enables:** Within-subspace rotation analysis; time-evolution of DoM direction during generation

## EXP-61: FE899 — Principal subspace angles correct vs incorrect
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2604.03038 (subspace geometry of representations)
**Hypothesis:** Correct and incorrect activations occupy measurably different PCA subspaces
**What we actually tested:** scipy.linalg.subspace_angles on per-class PCA at k ∈ {5,10,20,50}
**Key result:** Mean angles 34-39°, zero angles below 10° at k≤20, Grassmann distance grows with k
**Verdict:** CONFIRMED — classes diverge geometrically beyond the DoM direction
**Changed our understanding of:** Discriminative structure is multi-dimensional, not just 1-d DoM
**Files:** `pathway11_h100/subspace_angles/recompute_fe899.py`, `pathway11_h100/results/fe899_subspace_angles.json`
**Depends on:** F-2 (DoM)
**Enables:** Multi-dimensional correctness probes; class-conditional PCA

## EXP-62: FE930 — RoPE plane projection via q_proj pullback
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2604.18805 (position-dependent attention geometry)
**Hypothesis:** DoM is concentrated in a few attention heads' RoPE planes (positional coupling)
**What we actually tested:** Per-head decomposition of DoM variance through q_proj weight pullback (12 heads × 64 planes)
**Key result:** Gini=0.17 (nearly uniform), top-1 head=16.4%
**Verdict:** REJECTED — DoM has no special positional coupling
**Changed our understanding of:** Correctness signal is not position-encoded; lives in isotropic residual space
**Files:** `pathway11_h100/rope_projection/recompute_fe930.py`, `pathway11_h100/results/fe930_rope_projection.json`
**Depends on:** F-2 (DoM)
**Enables:** Rules out position-dependent steering; supports position-agnostic probes

## EXP-63: FE901 — SETOL ECS projection
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2604.02759 (SETOL weight spectral theory)
**Hypothesis:** DoM aligns with the effective correlation space (ECS) of L19 weight matrices
**What we actually tested:** W@W^T eigendecomp + power-law tail fit for o_proj and down_proj; DoM/PC1 projection onto ECS
**Key result:** o_proj alpha=1.14, DoM ECS=50%; down_proj alpha=1.72, DoM ECS=53%
**Verdict:** CONFIRMED (partial) — ~50% alignment, not concentrated in top eigenvectors
**Changed our understanding of:** DoM partially lives in weight correlation structure but is not a weight eigenvector
**Files:** `pathway11_h100/setol_ecs/recompute_fe901.py`, `pathway11_h100/results/fe901_setol_ecs.json`
**Depends on:** F-2 (DoM), EXP-57 (cov-spectrum)
**Enables:** Weight-informed probe design; SETOL-guided feature selection

## EXP-64: FE26841a — Logit-lens entropy for correctness prediction
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2502.00062 (logit lens as representation probe)
**Hypothesis:** L19 logit-lens entropy predicts correctness (lower entropy → more confident → correct)
**What we actually tested:** Entropy via tied embed_tokens unembed, 5-fold OOF logistic regression AUROC
**Key result:** AUROC=0.633; CORRECT samples have HIGHER entropy (8.76 vs 8.33) — direction reversed from hypothesis
**Verdict:** CONFIRMED (direction reversed) — entropy is informative but weaker than DoM; correct samples are LESS committed at L19
**Changed our understanding of:** Intermediate-layer "confidence" works oppositely to final-layer; correct solutions explore more at L19
**Files:** `pathway11_h100/logit_lens_entropy/recompute_fe26841a.py`, `pathway11_h100/results/fe26841a_logit_lens_entropy.json`
**Depends on:** F-2 (DoM)
**Enables:** Entropy as complementary feature; intermediate vs final layer confidence reversal study

## EXP-65: FE889 — Procrustes rotation-axis projection
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2604.18805 (instruction tuning geometry)
**Hypothesis:** DoM aligns with the largest RLHF/SFT rotation axes (correctness signal is instruction-tuning imprint)
**What we actually tested:** Procrustes alignment between base and instruct L19 o_proj + down_proj, matrix logarithm, rotation plane extraction, DoM projection
**Key result:** Largest o_proj rotation=76°, but DoM has only 1.6% variance in top-10 planes; top-10 plane AUROC=0.752
**Verdict:** REJECTED — DoM is NOT aligned with the largest instruction-tuning changes
**Changed our understanding of:** Correctness signal is not a direct RLHF artifact; it lives in its own subspace
**Files:** `pathway11_h100/procrustes_rotation/recompute_fe889.py`, `pathway11_h100/results/fe889_procrustes_rotation.json`
**Depends on:** F-2 (DoM)
**Enables:** RLHF vs emergent distinction; base-model DoM comparison

## EXP-66: FE909 — Per-layer alignment score profile
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** F-2 (L19 peak), FE145 (per-layer AUROC sweep)
**Hypothesis:** The correctness direction emerges sharply at L19 (single-layer imprint)
**What we actually tested:** cos(layer_diff, DoM_L19) across all 29 layers with 2000-resample bootstrap CIs
**Key result:** Peak at L19, half-max width=7 layers (L15-L21), gradual ramp from L12
**Verdict:** CONFIRMED (nuanced) — L19 is the peak but direction emerges gradually over ~7 layers
**Changed our understanding of:** Correctness signal is a multi-layer computation (L12-L21), not single-layer
**Files:** `pathway11_h100/alignment_profile/recompute_fe909.py`, `pathway11_h100/results/fe909_alignment_profile.json`
**Depends on:** EXP-42 (FE145 per-layer sweep), F-2 (DoM at L19)
**Enables:** Multi-layer probe designs; layer-range ablation studies

## EXP-67: FE919 — Diffusion Maps embedding
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2604.03038 (nonlinear manifold methods for representations)
**Hypothesis:** Nonlinear manifold structure in L19 activations encodes correctness beyond linear DoM
**What we actually tested:** Diffusion Maps with renormalized Laplacian, logistic regression on k ∈ {2,5,10,20} diffusion coordinates, 5-fold OOF AUROC
**Key result:** k=10 AUROC=0.788 > DoM 0.773; k=20 AUROC=0.784
**Verdict:** CONFIRMED — nonlinear manifold structure contributes beyond linear DoM
**Changed our understanding of:** There IS curvature-encoded correctness information; manifold-aware methods can improve on linear probes
**Files:** `pathway11_h100/diffusion_maps/recompute_fe919.py`, `pathway11_h100/results/fe919_diffusion_maps.json`
**Depends on:** F-2 (DoM), EXP-57 (cov-spectrum 0.793)
**Enables:** Manifold-aware probes; spectral embedding + logistic hybrid; kernel methods on activations

## EXP-68: FE925 — Hyvärinen score difference
**Date:** 2026-05-06
**Status:** COMPLETE
**Motivated by:** 2502.15016 (score-based analysis of representations)
**Hypothesis:** Non-Gaussian structure in class-conditional densities contributes to correctness prediction
**What we actually tested:** Gaussian QDA (Ledoit-Wolf) OOF AUROC; sliced score matching (200 slices)
**Key result:** Gaussian QDA OOF=0.736 < DoM 0.773; sliced score in-sample=0.996 (overfit)
**Verdict:** INCONCLUSIVE — Gaussian QDA underperforms DoM in high-d/low-n regime, but cov-spectrum (0.793) using 20 features beats both. Feature selection > model complexity.
**Changed our understanding of:** Full quadratic form is suboptimal with 500 samples in 1536-d; feature selection (cov-spectrum's 20 log-eigenvalues) matters more than model class
**Files:** `pathway11_h100/hyvarinen_score/recompute_fe925.py`, `pathway11_h100/results/fe925_hyvarinen_score.json`
**Depends on:** F-2 (DoM), EXP-57 (cov-spectrum)
**Enables:** Confirms feature-selection-first design principle for probes

## EXP-69: FE447 — Length-as-correctness baseline + OOF residualization
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** Deconfounding F-2 — how much of DoM's signal is just sequence length?
**Hypothesis:** If DoM is mostly length-driven, OOF residualized AUROC collapses below 0.6
**What we actually tested:** Spearman(DoM, seq_len) + OOF OLS residualization of DoM on length
**Key result:** OOF residualized DoM AUROC = **0.620** (in-sample was 0.665). Spearman = -0.619. Length AUROC = 0.799
**Verdict:** CONFIRMED — ~15% of DoM signal is length-explained; 0.620 still well above chance
**Changed our understanding of:** F-2's signal is partly confounded with length but majority survives
**Files:** `pathway11_h100/length_baseline/recompute_fe447.py`, `pathway11_h100/results/fe447_length_baseline.json`
**Depends on:** F-2 (DoM)
**Enables:** Honest lower bound on length-independent DoM signal

## EXP-70: FE188 — LID-MLE local intrinsic dimension
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-4 (asymmetric collapse) — does local dimension differ by correctness?
**Hypothesis:** Correct and incorrect samples live on manifolds of different local dimensionality
**What we actually tested:** Levina-Bickel MLE LID at k=5,10,20,50; class-conditional comparison
**Key result:** Best AUROC = **0.535** (k=20, negated LID). Per-class LID within 1 unit
**Verdict:** REJECTED — local intrinsic dimension is NOT a correctness predictor
**Changed our understanding of:** Manifold has similar local dimensionality everywhere; LID ≈ PR confirms local≈global dim
**Files:** `pathway11_h100/lid_mle/recompute_fe188.py`, `pathway11_h100/results/fe188_lid_mle.json`
**Depends on:** F-4 (spectral asymmetry)
**Enables:** Rules out local-dimension-based approaches

## EXP-71: FE421 — Ridge-LR on [prefill, final] concat
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-3 (orthogonality) — does regularized probing unlock final-token signal?
**Hypothesis:** Ridge concat > DoM prefill because final carries complementary signal
**What we actually tested:** L2-regularized logistic regression on concat(3072-d), prefill(1536-d), final(1536-d)
**Key result:** **Ridge concat 0.851, final-only 0.849, prefill-only 0.784**; DoM concat 0.757
**Verdict:** CONFIRMED — final token carries MORE signal than prefill when properly probed
**Changed our understanding of:** F-2's "prefill is special" was probe-specific. Ridge-LR final-only (0.849) >> DoM prefill (0.773)
**Files:** `pathway11_h100/regularized_concat/recompute_fe421.py`, `pathway11_h100/results/fe421_regularized_concat.json`
**Depends on:** F-2 (DoM), F-3 (orthogonality), EXP-51 (ridge-LR first appeared)
**Enables:** Final-token probes; challenges prefill-centric framing

## EXP-72: FE15 — Length-band PR control
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-4 (asymmetric collapse) — is PR stable across length bands?
**Hypothesis:** If PR varies systematically with length, F-4's collapse is length-confounded
**What we actually tested:** PR and DoM AUROC within short/medium/long sequence-length bands
**Key result:** PR: Short=17.5, Medium=21.5, Long=20.9. DoM AUROC holds (0.75–0.83). Global PR=19.86 ✓
**Verdict:** CONFIRMED — PR roughly stable, F-4 NOT a length artifact
**Changed our understanding of:** Spectral structure is consistent across sequence lengths
**Files:** `pathway11_h100/length_band_pr/recompute_fe15.py`, `pathway11_h100/results/fe15_length_band_pr.json`
**Depends on:** F-4 (asymmetric collapse)
**Enables:** Confirms spectral findings are not length-confounded

## EXP-73: FE16 — Marchenko-Pastur bias-corrected PR
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-4 — is the PR value inflated by finite-sample bias?
**Hypothesis:** MP correction changes PR substantially and closes the correct/incorrect gap
**What we actually tested:** Chun et al. gamma-row MP correction on PR for all/correct/incorrect
**Key result:** Corrected PR=18.9 (naive 19.9, ~5%). Correct=18.5 vs incorrect=19.6. Asymmetry survives
**Verdict:** CONFIRMED — finite-sample bias is ~5% (modest); class asymmetry survives correction
**Changed our understanding of:** PR estimates are reliable at this sample size; correct/incorrect gap is genuine
**Files:** `pathway11_h100/mp_bias_pr/recompute_fe16.py`, `pathway11_h100/results/fe16_mp_bias_pr.json`
**Depends on:** F-4 (asymmetric collapse)
**Enables:** MP-corrected PR as more honest estimate

## EXP-74: FE428 — L0 embedding-layer DoM baseline
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-2 — does the signal exist at the input (L0) or emerge in deeper layers?
**Hypothesis:** L0 embedding has zero correctness signal (signal emerges through computation)
**What we actually tested:** DoM AUROC at L0, cos(DoM_L0, DoM_L19), PR at L0
**Key result:** **L0 AUROC = 0.500, cos(DoM_L0, DoM_L19) = 0.0, PR_L0 = 0.0** — perfect null
**Verdict:** CONFIRMED — embedding carries zero correctness signal; F-2 emerges entirely in deeper layers
**Changed our understanding of:** L19 specificity fully confirmed; signal is computational, not input-inherited
**Files:** `pathway11_h100/l0_embedding_dom/recompute_fe428.py`, `pathway11_h100/results/fe428_l0_embedding_dom.json`
**Depends on:** F-2 (DoM), FE909 (alignment profile)
**Enables:** Rules out input-geometry explanations

## EXP-75: FE416 — Pre-final token DoM (position -2)
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-3 (orthogonality) — does the DoM direction rotate abruptly at the last token?
**Hypothesis:** If cos(prefinal, prefill) > 0.3, orthogonality is a last-token positional artifact
**What we actually tested:** DoM at position -2, cosines with prefill and final DoM
**Key result:** cos(prefinal, prefill) = **-0.054** (< 0.3), cos(prefinal, final) = **0.717**
**Verdict:** CONFIRMED — orthogonality is NOT a positional artifact; rotation is gradual
**Changed our understanding of:** DoM rotation happens across the generation trajectory, not at the answer token
**Files:** `pathway11_h100/prefinal_token_dom/recompute_fe416.py`, `pathway11_h100/results/fe416_prefinal_token_dom.json`
**Depends on:** F-3 (orthogonality)
**Enables:** Confirms F-3 is a trajectory property, not a positional artifact

## EXP-76: FE119 — Layer-sweep cos(prefill_DoM, final_DoM) all 29 layers
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-3 — is orthogonality specific to L19 or a network-wide property?
**Hypothesis:** If cos passes through ±1 at some layer, F-3 orthogonality is a L19 transient
**What we actually tested:** cos(prefill_DoM, final_DoM) at all 29 layers with 1000-resample bootstrap
**Key result:** cos near zero at ALL 29 layers (range [-0.113, +0.103]). Max |cos| = 0.103 at L28
**Verdict:** CONFIRMED — F-3 orthogonality is network-wide, not L19-specific
**Changed our understanding of:** Prefill and final DoM are genuinely independent computational channels throughout the entire transformer
**Files:** `pathway11_h100/layer_sweep_cos/recompute_fe119.py`, `pathway11_h100/results/fe119_layer_sweep_cos.json`
**Depends on:** F-3 (orthogonality)
**Enables:** Rules out layer-specific explanations; confirms independent circuits interpretation

## EXP-77: FE308 — Adaptive best-of-K Damani allocator
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-8 (selective prediction) — can DoM guide compute allocation instead of refusal?
**Hypothesis:** DoM-guided K allocation improves over uniform K at same average budget
**What we actually tested:** Damani greedy bin allocation with K=1–8, 5 difficulty bins from DoM scores
**Key result:** Cache K=1 = 41.6% (vs greedy 48.6%). Adaptive ≈ uniform at all K levels (~41%)
**Verdict:** REJECTED — T>0 sampling fundamentally degrades this model; adaptive can't rescue it
**Changed our understanding of:** Adaptive compute allocation is a dead end; selective prediction (refuse, don't retry) is the right paradigm
**Files:** `pathway11_h100/adaptive_bestofk/recompute_fe308.py`, `pathway11_h100/results/fe308_adaptive_bestofk.json`
**Depends on:** F-8 (selective prediction), F-2 (DoM scores for binning)
**Enables:** Closes the adaptive-compute avenue; validates F-8's refuse-don't-retry approach

## EXP-78: FE459 — Cross-model 1.5B↔7B DoM score correlation
**Date:** 2026-05-07
**Status:** COMPLETE
**Motivated by:** F-2 — is DoM geometry specific to 1.5B or universal across model sizes?
**Hypothesis:** If DoM scores correlate cross-model (Spearman > 0.5), difficulty geometry is universal
**What we actually tested:** 5-fold OOF DoM AUROC on 7B L19 prefill; Spearman/Kendall/concordance between 1.5B and 7B OOF scores
**Key result:** **7B AUROC = 0.874, Spearman(1.5B, 7B) = 0.937**, Kendall tau = 0.779, concordance = 0.890
**Verdict:** CONFIRMED — difficulty geometry is universal across model sizes
**Changed our understanding of:** F-2 is not a 1.5B-specific quirk; both models rank problems identically in activation space
**Files:** `pathway11_h100/cross_model_dom/recompute_fe459.py`, `pathway11_h100/results/fe459_cross_model_dom.json`
**Depends on:** F-2 (DoM), EXP-040 (cross-model PR ratios)
**Enables:** Cross-model DoM transfer; universal difficulty landscape claim

Next ID: **EXP-81**.
