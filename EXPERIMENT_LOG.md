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

Next ID: **EXP-043**.
