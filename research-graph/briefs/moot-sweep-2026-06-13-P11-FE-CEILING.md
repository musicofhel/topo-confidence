# Moot sweep — P11-FE-CEILING (2026-06-13)

**Verdict:** SPEC v8 'Raise the Ceiling' executed end-to-end (EXP-90/91, ~6 H100 hrs): four non-probe levers tested against the free-gate cascade's matched-cost MATH-500 ceiling (0.648 @ budget 2220.7). H-N CONFIRMED: escalating to Qwen2.5-Math-7B-Instruct instead of generic 7B lifts the cascade +4.20pp (0.648->0.690, p=0.0025), rescues 41/121 previously-unrescuable base failures, oracle 0.758->0.800 -> new pinned escalation target. H-O INCONCLUSIVE: Mathstral-7B adds 15/121 disjoint rescues (below the 25 confirm bar). H-P CONFIRMED: off-the-shelf Qwen2.5-Math-1.5B-Instruct scores 0.740 standalone at ~1/4 budget (cost 529), beating the whole budget-matched cascade by +9.2pp (p~0) -> the real product lever is the base, not introspection; REFUTED for DeepSeek-R1-Distill-1.5B (greedy 0.634 / T=0.6 0.678 @ cost 2742, over budget, reasoning length intrinsic). H-Q CONFIRMED: the free gate (length+logprob) generalizes to all new bases (OOF AUROC 0.863 Math-1.5B, 0.950/0.936 R1-Distill). H-R REFUTED (the centerpiece): a zero-label confidence filter curating 4000 Math-7B distillation traces into the generic 1.5B does NOT beat unfiltered at matched N=3065 (gate arm 0.494 < unfiltered 0.500 < perfect-label skyline 0.504; gate arm even < length-matched control 0.502 -> gate is a noisy step-length proxy at selection, not a quality signal). Even perfect label filtering buys only +0.4pp over unfiltered, so there is no curation increment to capture; and MATH-only SFT catastrophically forgets BBH on every arm (best 0.208 < base 0.267), failing the no-forgetting guardrail. GENERAL CLOSURE: the matched-cost ceiling is raised by swapping the base/target (free, off-the-shelf), not by curating data with the confidence stack; confidence-curated distillation is a free-rider. Honest recipe: use the off-the-shelf domain base + free-gate cascade on top, escalation target = Math-7B.

**Shortlisted:** 150 | **MOOT:** 3 | **ANSWERED:** 0 | **KEEP:** 147

| FE | sim | verdict | reason |
|---|---|---|---|
| P11-FE548 | 0.487 | MOOT | Identical to FE551; H-11 premise (7B as distillation teacher improves 1.5B) is refuted by verdict. |
| P11-FE551 | 0.487 | MOOT | H-11 (using 7B as teacher for 1.5B) is refuted by verdict showing off-the-shelf Qwen2.5-Math-1.5B beats cascade and distillation-based approaches fail. |
| P11-FE846 | 0.487 | MOOT | Would use MATH-500 supervision for SFT, but verdict explicitly documented that MATH-only SFT catastrophically forgets BBH (best 0.208 < base 0.267), failing the no-forgetting guardrail. |
| P10-FE22 | 0.480 | KEEP | Verdict does not address threshold-optimization strategies (CAP, learnable policies); selective-prediction ceiling improvement is orthogonal. |
| P10-FE26 | 0.460 | KEEP | Per-head L19 DoM probe to test whether layer-mean is suboptimal; verdict confirms layer-mean baseline (F-2: 0.7731) but does not preclude better per-head decomposition. |
| P10-FE35 | 0.435 | KEEP | Verdict does not test dual-coordinate steering or Park et al. control protocols. |
| P10-FE46 | 0.457 | KEEP | Verdict does not address learned-conditional steering; it refutes data-curation approaches, not the principle that position-conditional vectors could help vs. static vectors. |
| P11-FE1092 | 0.454 | KEEP | Verdict does not address circuit discovery via the full DCD pipeline; it focuses on the cascade/gate system and does not characterize whether multiple circuits explain MATH-500 correctness. |
| P11-FE110625A | 0.424 | KEEP | Tests cross-domain DoM transfer (TriviaQA→MATH); verdict shows domain-specific bases help but doesn't address whether DoM itself transfers across domains. |
| P11-FE110625B | 0.507 | KEEP | Step-count stratification of DoM AUROC not tested by the verdict. |
| P11-FE114 | 0.428 | KEEP | Verdict confirms free gate works but doesn't isolate length's contribution to F-8's 71.6% selective-prediction metric at coverage 0.5. |
| P11-FE1205 | 0.470 | KEEP | Tests SA-GSAE bipolar architecture efficiency for correctness detection; the verdict does not address SAE architectural variants. |
| P11-FE1236 | 0.478 | KEEP | Verdict does not test SDPG training or whether dense RL supervision reshapes DoM direction; training-method effects are orthogonal. |
| P11-FE1244 | 0.427 | KEEP | Verdict uses instruct checkpoint but doesn't test base vs. instruct; pipeline-specificity of DoM remains an open question. |
| P11-FE129 | 0.423 | KEEP | Verdict does not address whether discrete-code structure or codebooks beat the DoM probe; tests an orthogonal mechanistic question (Tamkin-style sparsity). |
| P11-FE131 | 0.427 | KEEP | Verdict doesn't address k-means discretization or whether discrete codes recover predictive lift without fine-tuning. |
| P11-FE140 | 0.439 | KEEP | Sanity check for BATS implementation on Qwen; verdict does not provide BATS results, so prerequisite validation remains necessary. |
| P11-FE168 | 0.463 | KEEP | Compares PDS (NLI-based) to prefill DoM at matched K=5; verdict does not address black-box baseline comparisons or single-head decomposition. |
| P11-FE180 | 0.512 | KEEP | Toy circuit reproduction remains independent of the verdict's production-lever findings. |
| P11-FE187 | 0.428 | KEEP | Verdict doesn't address refusal-detection probes or label-free abstention policies; orthogonal to confidence-curation and gate findings. |
| P11-FE195 | 0.452 | KEEP | Verdict does not examine layer-wise DoM signs; it does not refute the hypothesis that L0 has a sign-flip or that early-layer dynamics amplify to L19. |
| P11-FE199 | 0.421 | KEEP | Verdict does not address head-level attribution or per-head MLP accuracy; orthogonal mechanistic investigation. |
| P11-FE201 | 0.423 | KEEP | Verdict confirms gate generalization across models (H-Q) but does not test cross-task DoM transfer (MMLU/ARC); orthogonal probe-generalization question. |
| P11-FE233 | 0.422 | KEEP | Verdict does not address kernel-score layer selection; tests an orthogonal label-free mechanistic alternative. |
| P11-FE237 | 0.441 | KEEP | Verdict confirms DoM but does not address matching score as a mechanistic signal; this is an orthogonal interpretability question. |
| P11-FE248 | 0.499 | KEEP | Measures verbalized-confidence as a DoM alternative; orthogonal signal baseline not addressed by verdict. |
| P11-FE249 | 0.426 | KEEP | Tests whether RL changes residual-stream geometry; verdict shows introspection doesn't help product but doesn't answer whether RL rotates the prefill/final DoM relationship. |
| P11-FE260 | 0.467 | KEEP | Compares black-box engineered-feature LR vs white-box L19 DoM; the verdict does not evaluate black-box approaches and this comparison is still live. |
| P11-FE261 | 0.449 | KEEP | Tests external linguistic NLI feature on cached CoT; orthogonal to verdict's refutation of confidence-curated distillation (H-R) since NLI is a separate signal. |
| P11-FE263 | 0.441 | KEEP | Verdict validates DoM but does not settle whether it reduces to sparse vocab-projectable features; mechanistic decomposition remains open. |
| P11-FE289 | 0.496 | KEEP | Tests representation stability across fine-tuning; orthogonal to verdict's findings on base selection and distillation curation. |
| P11-FE303 | 0.422 | KEEP | Verdict does not settle whether prefill DoM is MATH-specific; tests cross-task probe transfer (TriviaQA/Winogrande→MATH), not tested in verdict. |
| P11-FE304 | 0.423 | KEEP | Tests probe-as-selector on K=8 resamples; verdict shows introspection broadly doesn't beat base-swapping but doesn't directly measure whether probe-based sample selection improves over baselines. |
| P11-FE307 | 0.476 | KEEP | Verdict does not address probe architecture (linear vs MLP) or layer-specificity of correctness signal. |
| P11-FE333 | 0.446 | KEEP | Tests cross-model transfer of prefill DoM from Instruct to base; verdict does not address transfer properties or mechanism separation (F-3). |
| P11-FE334 | 0.440 | KEEP | Verdict tests Qwen family but does not address cross-architecture generalization to Phi-3 or Gemma; generalization scope remains open. |
| P11-FE341 | 0.497 | KEEP | Compares steering methods (Linear-AcT vs α-sweep); verdict confirms steering works, just not as the product lever. |
| P11-FE343 | 0.433 | KEEP | Verdict does not test L2-norm OOD properties or Mayne et al. histogram comparisons. |
| P11-FE351 | 0.452 | KEEP | Verdict does not test cross-task generalization on execution-graded code (HumanEval/MBPP); it focuses on MATH-500 and the cascade system. |
| P11-FE352 | 0.494 | KEEP | F-1 cross-scale extension to 3B; verdict didn't refute universal breathing pattern, only showed base selection is the product lever. |
| P11-FE359 | 0.422 | KEEP | Verdict confirms gate generalizes to new bases (H-Q) but does not measure whether the prefill DoM itself transfers to AIME/AMC; cross-benchmark transfer untested. |
| P11-FE361 | 0.473 | KEEP | Tests SAE polysemanticity as a limit on L19 single-direction probes; the verdict does not address feature-level disentanglement. |
| P11-FE364 | 0.469 | KEEP | Tests prefill L19 vs final-token L27 probe strength; the verdict focuses on base selection and confidence-stacking, not on this layer/token comparison. |
| P11-FE366 | 0.436 | KEEP | Verdict does not test non-linear vs linear probe trade-off; whether subspace is non-linearly decodable remains unanswered. |
| P11-FE369 | 0.483 | KEEP | Verdict tested R1-Distill as a base model but did not measure DoM probe AUROC generalization; specific question remains unsettled. |
| P11-FE373 | 0.452 | KEEP | Tests Shrivastava's pairwise relative confidence method on Qwen-1.5B; verdict shows 0.740 standalone but does not measure this specific elicitation technique or its AUROC on MATH-500. |
| P11-FE382 | 0.420 | KEEP | Verdict doesn't address mechanistic steering or single-direction uniformity across layers; base-model conclusions don't refute the steering premise. |
| P11-FE383 | 0.480 | KEEP | Verdict does not test multi-layer logit-lens or window-probe strategies; single-layer privileging claim remains untested. |
| P11-FE386 | 0.437 | KEEP | Verdict refutes data curation but does not test steering interventions; FLORAIN as an H-1 method upgrade remains live. |
| P11-FE389 | 0.454 | KEEP | Verdict refutes confidence-based data curation, not disentanglement as a way to improve the direction itself; whether TELLME can raise AUROC > 0.85 remains untested. |
| P11-FE391 | 0.420 | KEEP | Verdict confirms prefil-DoM exists and generalizes (H-Q) but doesn't provide the 5-metric disentanglement audit P11-FE391 proposes to quantify. |
| P11-FE396 | 0.457 | KEEP | Verdict does not test optimization-based steering deltas; it refutes confidence-filtered data curation, which is orthogonal to whether optimization improves heuristic steering. |
| P11-FE397 | 0.462 | KEEP | Tests layer-optimization locus on Qwen-2.5-1.5B; verdict confirms F-2 baseline but does not address layer-wise optimization or replication of mechanistic findings from 2502.06115. |
| P11-FE414 | 0.478 | KEEP | Verdict tested confidence-filtered distillation (H-R) but not verbalized-confidence routing gates vs DoM gates; direct comparison remains unsettled. |
| P11-FE419 | 0.425 | KEEP | Tests layer-specificity of correctness signal; verdict doesn't address whether earlier layers have comparable signal to L19. |
| P11-FE422 | 0.551 | KEEP | Gradient-based edge attribution methodology remains untested by the verdict. |
| P11-FE425 | 0.459 | KEEP | Per-head DoM steering precision-recall at matched tuning; verdict says introspection is not the system-level lever but does not directly test whether steering improves accuracy in isolation. |
| P11-FE429 | 0.513 | KEEP | DoM probe AUROC on R1-Distill not directly measured; verdict reports gate AUROC (0.950) but not probe AUROC. |
| P11-FE431 | 0.431 | KEEP | Verdict found DoM logit is 'not a quality signal' for data curation, but does not directly test inference-time early-exit, which may have different dynamics. |
| P11-FE435 | 0.513 | KEEP | Identical to FE429; DoM probe measurement on R1-Distill not provided by verdict. |
| P11-FE437 | 0.431 | KEEP | Verdict confirms the free gate (length+logprob) generalizes, but doesn't address whether prefill DoM early-exit outperforms Zhang's approach—still a valid question. |
| P11-FE445 | 0.419 | KEEP | Verdict refuted data-curation (H-R) but not SFT-based selective prediction; MAC-Tuning's label use in SFT is mechanistically distinct from the distillation-curation tested. |
| P11-FE446 | 0.427 | KEEP | Tests regime-specificity of F-2 (multi-problem vs single-problem); verdict doesn't address whether DoM AUROC depends on problem concatenation. |
| P11-FE449 | 0.428 | KEEP | Verdict doesn't address difficulty-tier stratification or Su et al.'s underthink-hard frame; orthogonal question. |
| P11-FE451 | 0.505 | KEEP | Wang et al. fine-tuning AUROC not measured; verdict shows base-lever priority but doesn't test this method. |
| P11-FE458 | 0.423 | KEEP | Tests whether DoM survives RevKD distillation; verdict refutes curated distillation (H-R) but doesn't address whether unfiltered RevKD preserves the direction geometrically. |
| P11-FE462 | 0.420 | KEEP | Verdict confirms task-aware base models win and off-the-shelf dominates refinement, but doesn't directly test the RevKD mechanism with specific teacher quality variants—premise isn't refuted. |
| P11-FE463 | 0.458 | KEEP | Singular-asymmetry σ_max(E)/σ_max(W) as predictor of breathing amplitude; verdict does not address embedding-coupling dynamics or mechanistic drivers of F-1 amplitude. |
| P11-FE477 | 0.492 | KEEP | Tests refusal direction as label-free correctness proxy; orthogonal geometry type not addressed by verdict. |
| P11-FE480 | 0.436 | KEEP | Verdict tests base selection and data curation, not DoM steering mechanisms or length confounds. |
| P11-FE481 | 0.458 | KEEP | Verdict does not test cross-architecture generalization on Gemma; the free-gate generalization result (H-Q) applies to the gate itself, not to whether DoM generalizes to new architectures. |
| P11-FE491 | 0.441 | KEEP | Verdict validates prefill DoM baseline but does not compare to SAE sparse features; alternative representation remains untested. |
| P11-FE501 | 0.436 | KEEP | Verdict does not address F-7's per-problem geometric signature claims or backtracking-direction detection. |
| P11-FE505 | 0.433 | KEEP | Verdict does not test STU-PID or other inference-time steering schedules. |
| P11-FE510 | 0.474 | KEEP | Tests dispersion-gap as a checkpoint-selection metric, which remains useful regardless of the verdict's finding that domain-optimized bases are superior. |
| P11-FE513 | 0.476 | KEEP | Verdict does not localize correctness signal via rank trajectories across layers; layer-wise signal distribution remains unaddressed. |
| P11-FE517 | 0.420 | KEEP | Verdict establishes DoM gating works (F-8 at 71.6%/50%) but doesn't test cycle-detection gating, leaving the competitive premise unrefuted. |
| P11-FE534 | 0.490 | KEEP | Tests steering recovery on rejected subset; verdict showed base selection is the product lever but didn't directly compare steering vs selective-prediction on hard examples. |
| P11-FE536 | 0.528 | KEEP | Token-substitution mechanism testing not addressed; verdict de-prioritizes but doesn't measure token effects. |
| P11-FE547 | 0.424 | KEEP | Tests layer-specificity by comparing L27 to L19 DoM; verdict doesn't address whether final-layer probes match middle-layer performance. |
| P11-FE550 | 0.424 | KEEP | Tests layer-specificity by comparing L27 to L19 DoM; verdict doesn't address whether final-layer probes match middle-layer performance. |
| P11-FE553 | 0.423 | KEEP | Tests steering-validity KL divergence filter; verdict doesn't address whether prefill DoM passes the SteeringSafety constraint. |
| P11-FE560 | 0.490 | KEEP | adjudicator returned no/invalid verdict — kept |
| P11-FE561 | 0.448 | KEEP | Tests whether text-only GCM can match DoM AUROC; verdict does not measure the necessity of hidden states for correctness prediction. |
| P11-FE579 | 0.454 | KEEP | Verdict tests domain-specialized models (Qwen2.5-Math-1.5B-Instruct, etc.) but does not explicitly report whether any are pure-SFT without DPO/RL, so the RL-vs-SFT scoping question is not directly answered. |
| P11-FE59 | 0.420 | KEEP | Verdict does not measure log-prob calibration across K=8 paths; orthogonal baseline-mechanism test. |
| P11-FE590 | 0.430 | KEEP | Verdict doesn't address norm equalization or F-4's asymmetric-collapse mechanism—orthogonal to base/curation findings. |
| P11-FE594 | 0.455 | KEEP | Verdict does not test path-geometry via cumulative L2 changes; it refutes data-curation strategies, not the hypothesis that path wandering correlates with correctness. |
| P11-FE602 | 0.475 | KEEP | Verdict does not test multi-layer Gaussian depth schedules; claim that single-layer L19 is degenerate remains untested. |
| P11-FE611 | 0.421 | KEEP | Verdict does not test covariance whitening (WRMD) variants; orthogonal probe improvement question. |
| P11-FE612 | 0.430 | KEEP | Verdict focuses on base swapping and data curation; doesn't address whether DoM layer is absolute or fractional across architectures. |
| P11-FE62 | 0.420 | KEEP | Identical to FE59 (log-prob calibration audit); orthogonal to verdict. |
| P11-FE626 | 0.479 | KEEP | Verdict does not test attention-graph PH features; substrate privileging (residual vs attention) is orthogonal to base-model levers. |
| P11-FE630 | 0.451 | KEEP | Tests label-free P(SUFFICIENT) logit probe; verdict does not measure label-free elicitation methods or their AUROC. |
| P11-FE632 | 0.464 | KEEP | Tests monotonicity of confidence during step-wise CoT reasoning; verdict refutes confidence filtering (H-R) but not whether confidence increases monotonically during generation. |
| P11-FE633 | 0.467 | KEEP | Tests whether single-direction L19 vs mixture-of-6-directions better captures input-conditional correctness signal; the verdict does not settle expressiveness of single vs multi-probe. |
| P11-FE634 | 0.446 | KEEP | Tests layer sensitivity across L20/L22/L25; verdict does not measure whether adjacent layers achieve comparable AUROC to L19. |
| P11-FE635 | 0.489 | KEEP | adjudicator returned no/invalid verdict — kept |
| P11-FE647 | 0.448 | KEEP | Tests RAPTOR+GCAV steering interventions; verdict refutes introspection as the *product lever* but does not test whether steering can causally move accuracy post-deployment. |
| P11-FE649 | 0.510 | KEEP | Per-head activation extraction is prerequisite data collection for downstream refutation tests. |
| P11-FE65 | 0.434 | KEEP | Verdict confirms prefill DoM AUROC (0.7731) but does not test circuit decomposition via path patching. |
| P11-FE661 | 0.455 | KEEP | Verdict does not characterize DoM steerability mechanics; it confirms the free gate works (H-Q) but does not resolve whether DoM obeys the three-stage curve or what the steerability budget L± is. |
| P11-FE667 | 0.467 | KEEP | Tests AUSteer multiplicative steering design vs CAA additive steering; verdict refutes confidence-filtering specifically but does not directly eliminate alternative steering mechanisms. |
| P11-FE670 | 0.441 | KEEP | Verdict confirms DoM but does not test per-head attribution or circuit sparsity; mechanistic circuit questions remain open. |
| P11-FE677 | 0.453 | KEEP | Verdict refutes confidence-filtered data curation and favors base-swapping over introspection; however, it does not directly test multi-basis representation engineering, so whether REP-Big-Five composition beats single DoM remains open. |
| P11-FE679 | 0.469 | KEEP | Tests Spherical Steering as an alternative introspection method; verdict refutes confidence-curated distillation specifically but does not directly rule out other steering approaches. |
| P11-FE680 | 0.427 | KEEP | Verdict doesn't address antipodal symmetry or F-4's asymmetric-collapse mechanism; orthogonal to base/gate findings. |
| P11-FE697 | 0.425 | KEEP | Tests concept-factorizability as prerequisite for dual-steering; verdict doesn't address whether MATH correctness factors cleanly on the hyperplane. |
| P11-FE705 | 0.422 | KEEP | Verdict refutes H-R (confidence filtering for data curation) but does not test whether multi-layer LoRA with confidence supervision beats single-layer DoM; uses confidence differently. |
| P11-FE712 | 0.461 | KEEP | Aligns ReBalance's O/U behavioral taxonomy with F-7 buckets; verdict refutes confidence filtering (H-R) but not label-free behavioral decomposition or signal alignment. |
| P11-FE716 | 0.622 | KEEP | Verdict does not test steering vector transfer; orthogonal to base-model lever findings. |
| P11-FE723 | 0.471 | KEEP | Tests cross-model DoM transfer (1.5B→7B geometric transfer), which the verdict does not address; comparing domain-optimized checkpoints ≠ testing direction transfer. |
| P11-FE724 | 0.510 | KEEP | MLP vs linear probe architecture comparison not measured by the verdict. |
| P11-FE727 | 0.451 | KEEP | Tests whether ridge probes with ℓ2 sweep outperform mass-mean DoM on cached activations; verdict does not measure alternative probe methods. |
| P11-FE731 | 0.502 | KEEP | Tests steering calibration/transfer at 1.5B scale; verdict found the product lever is base selection (Math-7B) but didn't refute that steering signal quality can be characterized at smaller scales. |
| P11-FE732 | 0.512 | KEEP | TTT probe updates not tested; verdict de-prioritizes probe refinement but doesn't refute TTT's mechanistic questions. |
| P11-FE734 | 0.460 | KEEP | Label-free consistency-based TTT probe (ORCA's consistent mode); verdict refutes label-based confidence filtering (H-R) but does not address label-free alternatives. |
| P11-FE736 | 0.468 | KEEP | Tests whether step-organized linear subspaces are architecture-universal (Qwen vs Llama); the verdict does not address step-marker organization. |
| P11-FE741 | 0.442 | KEEP | Verdict refutes data curation but does not test steering interventions; characterizing steering-direction landscape remains live. |
| P11-FE770 | 0.423 | KEEP | Tests whether findings reduce to margin-tail artifacts; verdict shows base-swapping is the main lever but doesn't address whether F-2/F-7/F-8 are calibration epiphenomena. |
| P11-FE778 | 0.443 | KEEP | Verdict validates prefill DoM on new Qwen bases but does not address PANL probe comparison; this tests a distinct methodological question. |
| P11-FE783 | 0.443 | KEEP | Identical to FE778; PANL vs DoM comparison is orthogonal to the verdict's focus on bases and data curation. |
| P11-FE788 | 0.422 | KEEP | Verdict does not address truncation sensitivity or continuous CoT-length degradation curve; orthogonal infrastructure analysis. |
| P11-FE791 | 0.460 | KEEP | Unblocks FE22 to test F-1/2/3/5 on Abstract-CoT dataset; verdict concerns system-level levers for MATH-500, not cross-benchmark generalization of geometric properties. |
| P11-FE795 | 0.422 | KEEP | Identical to FE788 (truncation-at-32 sweep); orthogonal to verdict's conclusions about base vs introspection. |
| P11-FE798 | 0.460 | KEEP | Identical to FE791 (unblock and reframe FE22 for Abstract-CoT); verdict does not address generalization of F-1/2/3/5 to different reasoning datasets. |
| P11-FE8 | 0.451 | KEEP | Tests Mostafazadeh et al.'s question-only protocol as a replication study; verdict does not measure this specific elicitation protocol. |
| P11-FE804 | 0.428 | KEEP | Verdict says 'base is the real lever, not introspection' but doesn't refute whether multi-direction extraction could still provide gains on top of that. |
| P11-FE81 | 0.445 | KEEP | Tests circuit localization via ACDC (H-13 hypothesis); verdict refutes H-R (curation) but does not refute or settle H-13's premise about DoM causality. |
| P11-FE810 | 0.433 | KEEP | Verdict does not test final-token indirect-logit probes. |
| P11-FE811 | 0.427 | KEEP | Verdict refutes confidence-curated distillation (training-data filtering), not in-context verbalized confidence (inference-time labeling); different mechanisms. |
| P11-FE817 | 0.426 | KEEP | Tests competing null (Future Lens) against DoM; verdict doesn't address whether linear future-state probe matches DoM AUROC. |
| P11-FE836 | 0.523 | KEEP | Optimized steering vector transfer not measured by the verdict. |
| P11-FE840 | 0.475 | KEEP | Tests head-level circuit structure (induction/copying heads) which is orthogonal to the verdict's finding about base-model selection and confidence-stack failure. |
| P11-FE841 | 0.443 | KEEP | Verdict confirms DoM works on Qwen bases but does not provide Brier/ECE calibration metrics; this adds an orthogonal measurement dimension. |
| P11-FE843 | 0.484 | KEEP | Verdict does not address whether multiple correctness directions exist; INLP discovery is orthogonal to ceiling-raising levers tested. |
| P11-FE845 | 0.419 | KEEP | Verdict's confirmation that F-2 generalizes (H-Q) actually supports mechanistic framing; doesn't test whether simpler non-mechanistic methods could recover the same AUROC. |
| P11-FE847 | 0.461 | KEEP | Tests DoM AUROC stability across reasoning-token budgets {0, 256, 1024}; verdict does not address robustness of F-2 across varying compute regimes. |
| P11-FE856 | 0.492 | KEEP | Head-to-head steering comparison with external method; verdict didn't settle competing steering approaches. |
| P11-FE858 | 0.431 | KEEP | Verdict does not test F-8 stability during GRPO training or durability under RL fine-tuning. |
| P11-FE86 | 0.460 | KEEP | Circuit decomposition via path-patching at L19 prefill; verdict refutes confidence-curation system leverage but not the existence or mechanistic structure of the DoM sub-circuit. |
| P11-FE864 | 0.428 | KEEP | Verdict doesn't address steering implementations or norm preservation; it focuses on cascades, gates, and base selection. |
| P11-FE894 | 0.466 | KEEP | Tests whether DoM signal is GRPO-specific or RL-objective-agnostic; the verdict does not address MaxRL and leaves this orthogonal question open. |
| P11-FE928 | 0.453 | KEEP | Verdict does not decompose attention mechanisms or test word-position cross-terms; it refutes data-curation approaches, not the hypothesis that attention positional confounds exist. |
| P11-FE94 | 0.434 | KEEP | Verdict does not test cross-task transfer of DoM from TriviaQA to MATH-500. |
| P11-FE976 | 0.449 | KEEP | Identical to P11-FE980; tests layer-wise geometric properties not addressed by the verdict. |
| P11-FE98 | 0.434 | KEEP | Verdict does not test cross-task transfer of DoM from TriviaQA to MATH-500. |
| P11-FE980 | 0.449 | KEEP | Tests whether L19's predictive power is explained by geometric continuity structure; verdict focuses on base-swapping and curation, not layer geometry. |
| P11-FE995 | 0.436 | KEEP | Verdict does not test whether prefill DoM is stable across RL objectives (GRPO vs DGPO). |
| P7-FE4 | 0.489 | KEEP | Sanity-check for TDA/attention methods; verdict didn't refute topological approaches, only showed they weren't the winning product lever. |

Review: wrongly-closed FEs can be revived with
`python update_status.py <id> READY`.
