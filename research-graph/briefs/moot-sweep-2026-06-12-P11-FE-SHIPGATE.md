# Moot sweep — P11-FE-SHIPGATE (2026-06-12)

**Verdict:** SPEC v7 executed end-to-end (EXP-84..89): confgate/ pip package shipped FIRST from pinned v6 results, then every challenger ran at matched cost and LOST. H-I holds on dev AND confirmatory T5 (SmolLM2 0.808 / Gemma 0.842 / OLMo-2 0.836 free-gate OOF; no arm adds >=+0.01 at <=1.05x cost anywhere; entropy no-harm passes all three families). H-K refuted with prejudice (spectral-alpha SUBTRACTS, -0.0206 p=0.017, closed permanently). H-J confirmed (K=8 majority 0.554 vs cascade 0.732 at matched cost). H-M refuted (nested-OOF router -0.40pp vs plain cascade at the 11.2pp-gap operating point, p=0.68; rescue density — 47% of 1.5B failures unrescuable by 7B — is the binding constraint, not ranking). PRM block (resolves P11-FE403): Qwen2.5-Math-PRM-7B is an excellent verifier (0.9396 standalone, +0.1018 over gate p<1e-4) but costs 1.042x a full escalation; even a hypothetical 1.5B-cost PRM with 7B AUROC loses to plain escalation. GENERAL CLOSURE: at matched cost, no introspection probe beats spending the same tokens on escalation. P2 (executes P11-FE412): H-L1/H-L2 refuted (pseudo-label noise 0.388 on web_of_lies; k=32 TRUE labels is the honest recipe), H-L3 confirmed (7B-agreement pseudo-labels precision 0.951 give valid cross-scale conformal certs, zero human labels). Cache defects found+corrected: D-3 BBH n_gen_tokens all-zero (honest BBH gate 0.782->0.806, MATH->BBH transfer anchor 0.785->0.828 — gate was UNDERRATED); D-4 SmolLM2/OLMo-2 tokenizer round-trip failure (genscore gen-time capture, corr 0.9998 on matched subset). Deliverable: confgate cascade-as-is ships as final.

**Shortlisted:** 150 | **MOOT:** 3 | **ANSWERED:** 0 | **KEEP:** 147

| FE | sim | verdict | reason |
|---|---|---|---|
| P11-FE405 | 0.419 | MOOT | The verdict's general closure—'at matched cost, no introspection probe beats escalation'—directly refutes hybrid routing combining PRM and DoM, as the verdict already tested PRM alone and found it loses, and tested nested-OOF routing which also failed. |
| P11-FE678 | 0.462 | MOOT | Verdict closes introspection methods (no probe beats escalation), refuting premise that H-1 stability is worth pre-checking. |
| P11-FE822 | 0.454 | MOOT | Verdict closes H-1 (introspection loses to escalation), refuting premise that H-1 will be tested and needs per-problem metrics. |
| P10-FE26 | 0.417 | KEEP | The verdict uses layer-mean L19 DoM as the baseline for confgate but doesn't evaluate per-head decomposition or whether a single attention head carries the signal more cleanly. |
| P10-FE30 | 0.465 | KEEP | Tests adaptive vs fixed steering coefficients in P10 (older pathway); P11-SPEC v7 verdict does not explicitly cover P10-era steering parameter optimization. |
| P10-FE33 | 0.453 | KEEP | WRMD steering is not addressed by the verdict; the verdict tested other steering arms but not WRMD's performance on MATH tasks. |
| P11-FE105 | 0.439 | KEEP | Verdict addresses matched-cost gating comparisons, not per-layer geometric analysis or F-2's 'knowledge axis' framing. |
| P11-FE110625B | 0.438 | KEEP | Verdict does not address step-count stratification or H-5 familiarity framing; orthogonal to the matched-cost comparisons. |
| P11-FE111 | 0.430 | KEEP | Tests layer-wise ActAdd AUROC to validate L19 as steering-optimal; verdict does not address layer selection for steering. |
| P11-FE114 | 0.429 | KEEP | Verdict confirms F-8 at 71.6% coverage but does not test whether length alone achieves equivalent performance. |
| P11-FE1205 | 0.416 | KEEP | Verdict establishes DoM 0.7731 as the baseline but does not test whether SA-GSAE's bipolar architecture matches or exceeds this AUROC. |
| P11-FE1214 | 0.438 | KEEP | Verdict does not address distillation robustness or direction stability across checkpoints. |
| P11-FE1244 | 0.415 | KEEP | Verdict validates DoM on Instruct checkpoint but does not test whether the direction generalizes to base-model checkpoints or differs across pipeline variants. |
| P11-FE131 | 0.438 | KEEP | Verdict does not address discretization or k-means clustering as an alternative to fine-tuning. |
| P11-FE168 | 0.468 | KEEP | Compares PDS (behavioral) to prefill-DoM (geometric) at matched K; verdict shows escalation wins overall but does not directly compare these two signal types. |
| P11-FE18 | 0.460 | KEEP | Verdict doesn't address mid-generation (post-answer-newline) probes; temporal question remains orthogonal. |
| P11-FE182 | 0.416 | KEEP | Verdict confirms cascade routing works well but does not directly compare SOFT-SC's adaptive logit-based stopping against H-19's verification routing. |
| P11-FE187 | 0.450 | KEEP | The verdict does not test ReCoVERR-style clarification wrappers as an abstention policy; this is orthogonal to the selective-prediction stack comparison. |
| P11-FE199 | 0.444 | KEEP | Head-level MLP probe heatmap for H-13 is orthogonal to verdict's focus on routing strategies and gate confirmation. |
| P11-FE217 | 0.498 | KEEP | Causal intervention test of F-9 (CoE-60 redundancy) via L19 noising; verdict does not use causal methods to test F-9. |
| P11-FE248 | 0.466 | KEEP | Tests verbalized confidence vs geometric probes; verdict refutes geometry-as-the-right-substrate but does not directly test raw verbalized confidence at this scale/task. |
| P11-FE249 | 0.500 | KEEP | Tests whether RL training rotates residual-stream geometry post-training; verdict does not test post-RL geometry changes. |
| P11-FE252 | 0.421 | KEEP | Verdict confirms DoM approach but does not test causal steering or compare against BiPO and Wang & Shu vectors. |
| P11-FE254 | 0.439 | KEEP | Joint prefil+final probe versus prefil-only (trajectory-contrastive signal question) is not tested in the verdict. |
| P11-FE260 | 0.469 | KEEP | Head-to-head comparison of black-box vs white-box probes for selective prediction; verdict establishes escalation > introspection but does not directly settle which introspection approach is better. |
| P11-FE261 | 0.448 | KEEP | The verdict does not test SRC contradiction-probability features; it closes escalation-vs-probes but not alternative feature families. |
| P11-FE263 | 0.427 | KEEP | Verdict confirms F-2 L19 DoM works but does not test mechanistic reduction to sparse vocab-projectable MLP columns. |
| P11-FE288 | 0.429 | KEEP | Verdict confirms F-2 overall but does not test whether L19 is specifically special vs uniformly useful across mid-layers. |
| P11-FE289 | 0.521 | KEEP | Tests H-8 (prefill direction stability) and H-10 (breathing pattern changes) via fine-tuning; verdict does not address these hypotheses. |
| P11-FE298 | 0.421 | KEEP | Verdict confirms F-8 but does not test counterfactual prompting as label-free alternative baseline. |
| P11-FE307 | 0.456 | KEEP | Verdict doesn't address MLP architectures or probe optimality; orthogonal to escalation and layer-choice questions. |
| P11-FE313 | 0.448 | KEEP | The verdict confirms robustness of prefill DoM across model families but does not measure direction stability versus norm stability across re-fits. |
| P11-FE335 | 0.435 | KEEP | Verdict confirms F-8's 71.6% as shipped but does not test whether label-free CoE-C can achieve equivalent selective-prediction performance. |
| P11-FE337 | 0.435 | KEEP | Verdict confirms DoM-gated selective prediction (71.6%) and ships it, but does not test whether CoE-C as a label-free replacement can match this threshold. |
| P11-FE343 | 0.493 | KEEP | L2-norm OOD diagnostic for L19 DoM against Mayne et al. baseline; verdict does not address OOD properties of the DoM. |
| P11-FE344 | 0.435 | KEEP | Verdict does not mention H-1 or its alpha-sweep; OOD-drift concern during per-position steering is orthogonal to the matched-cost escalation closure. |
| P11-FE346 | 0.439 | KEEP | Verdict does not address step-boundary signal analysis or scale-dependent behavior; orthogonal to matched-cost closure. |
| P11-FE357 | 0.469 | KEEP | Compares two probe designs (step-boundary vs prefill), not whether probes beat escalation; verdict settles escalation vs introspection broadly but not which probe design is better. |
| P11-FE364 | 0.429 | KEEP | Verdict does not compare OPENIA last-decoded-token AUROC against prefill L19 DoM. |
| P11-FE369 | 0.431 | KEEP | Tests F-2 generalization to reasoning-distilled models; verdict does not address reasoning-checkpoint effects. |
| P11-FE371 | 0.412 | KEEP | Verdict does not address whether base-model generations contain reflection-marker tokens or whether post-answer-newline activations co-occur with reflection. |
| P11-FE373 | 0.418 | KEEP | The verdict doesn't evaluate whether Shrivastava's linguistic-elicitation baseline (without internal access) matches F-2's 0.7731 AUROC; this is an orthogonal competitive baseline. |
| P11-FE38 | 0.437 | KEEP | Verdict does not address topological invariants or rotation-invariant distance metrics for F-3's mechanism claim. |
| P11-FE386 | 0.429 | KEEP | Verdict does not mention FLORAIN or test nonlinear steering methods; general closure about probes does not specify which probe methods were examined. |
| P11-FE389 | 0.453 | KEEP | The verdict confirms prefill DoM works (H-I holds) but does not address whether fine-tuning via TELLME can improve the direction itself. |
| P11-FE390 | 0.433 | KEEP | Verdict does not establish the Self-Sim label-free baseline (mean-cosine on correct vs. incorrect) against which to measure supervised DoM headroom. |
| P11-FE391 | 0.413 | KEEP | Verdict does not measure or compare disentanglement metrics (Coding Rate, eRank, l2 distance, angle, Hausdorff) against TELLME's baseline values. |
| P11-FE394 | 0.424 | KEEP | Verdict confirms F-8 is final but does not compute THS; adding a secondary metric to confirmed F-8 is independent work. |
| P11-FE396 | 0.420 | KEEP | The verdict doesn't directly address whether optimized steering (Eq. 2 of 2502.06115) beats heuristic DoM steering; it focuses on escalation outperforming introspection routing, which is orthogonal to steering-strategy comparison. |
| P11-FE399 | 0.487 | KEEP | Compares verbal-probe signal against geometric probes on same K=8 paths; verdict does not test whether verbal probes carry orthogonal information vs geometric signals. |
| P11-FE401 | 0.441 | KEEP | Verdict tested K=8 majority (loses) and general routing strategies, but CISC, F-8 selective prediction, and H-19 routing specifics were not directly measured. |
| P11-FE404 | 0.449 | KEEP | Although the verdict rules that PRM-based approaches lose to escalation at matched cost, the specific comparison of BoN trajectory search versus DoM-gated selective-prediction is not directly tested. |
| P11-FE41 | 0.440 | KEEP | Best-of-8 reranking by DoM probe to recover D-bucket errors is not tested in the verdict. |
| P11-FE414 | 0.465 | KEEP | Compares two gates for escalation routing (behavioral vs geometric); verdict establishes pure escalation > gated escalation, but does not settle which gate signal is better for routing decisions when gating is employed. |
| P11-FE422 | 0.439 | KEEP | EAP attribution for head vs MLP dominance (H-13 premise) is orthogonal to verdict's routing-strategy closures. |
| P11-FE429 | 0.430 | KEEP | Tests F-2 on reasoning models to evaluate substrate-limiting hypothesis; verdict does not address substrate effects. |
| P11-FE431 | 0.478 | KEEP | Identical to P11-FE437; token-efficiency comparison against external baselines remains live. |
| P11-FE433 | 0.431 | KEEP | Tests calibration metrics (ECE/Brier) of F-2; verdict does not evaluate or refute calibration properties. |
| P11-FE435 | 0.430 | KEEP | Tests F-2 on reasoning models to evaluate substrate-limiting hypothesis; verdict does not address substrate effects. |
| P11-FE437 | 0.478 | KEEP | The verdict gives the F-8 operating point (71.6% at 50% coverage) but does not provide token-reduction metrics or comparison to Zhang's 24% token reduction baseline, leaving the apples-to-apples efficiency comparison unresolved. |
| P11-FE439 | 0.431 | KEEP | Tests calibration metrics (ECE/Brier) of F-2; verdict does not evaluate or refute calibration properties. |
| P11-FE444 | 0.468 | KEEP | Tests verbalized confidence as a baseline alternative; verdict refutes introspection-as-infrastructure but does not directly test or settle raw verbalized confidence on this exact task. |
| P11-FE451 | 0.451 | KEEP | The verdict does not test whether decoder-regression tokens can beat the 0.7731 prefill DoM baseline. |
| P11-FE454 | 0.412 | KEEP | The verdict tests PRMs (a related introspection probe) against escalation, but does not directly compare verbalized confidence to DoM on the same split, so the specific question remains open. |
| P11-FE456 | 0.424 | KEEP | General closure about probes does not specifically name So-Im; early-stopping threshold comparison against F-8's risk-coverage curve is not addressed. |
| P11-FE461 | 0.440 | KEEP | Knowledge distillation's effect on L19 DoM peak layer (F-2 rescoping question) is not addressed in verdict. |
| P11-FE463 | 0.432 | KEEP | Tests singular asymmetry's correlation with F-1 breathing amplitude; verdict does not address weight-matrix asymmetry mechanisms. |
| P11-FE475 | 0.433 | KEEP | Verdict confirms single-layer L19 DoM (0.7731) and ships it, but does not test multi-layer weighted MLPs on concatenated L19–L27 to determine if they lift ≥0.02. |
| P11-FE476 | 0.436 | KEEP | Verdict establishes gating works (improved D-3 AUROC post-correction) but does not optimize or test alternative layers. |
| P11-FE477 | 0.448 | KEEP | The verdict does not address whether refusal geometry overlaps with correctness geometry or carries useful signal for MATH correctness. |
| P11-FE482 | 0.432 | KEEP | Verdict confirms DoM baselines (prefill 0.7731, final-token 0.7186, selective 71.6%) but does not test CCPS final-token feature bundle on P11 MATH-500 for direct comparison. |
| P11-FE485 | 0.432 | KEEP | Identical to P11-FE482; verdict does not test CCPS final-token features on P11 cached activations. |
| P11-FE49 | 0.412 | KEEP | Verdict reports in-distribution F-2/F-8 numbers (0.7731 AUROC, 71.6% at 50% coverage) but does not evaluate OOD min-max robustness across GSM-Hard, numeric perturbations, paraphrases, and holdouts. |
| P11-FE491 | 0.479 | KEEP | The verdict establishes the dense L19 DoM baseline (0.7731 AUROC) but does not test whether K*=20 SAE features can match or exceed it, a mechanistic interpretability question orthogonal to the escalation closure. |
| P11-FE492 | 0.419 | KEEP | The verdict confirms F-8's 71.6% in-distribution performance but doesn't evaluate OOD robustness against held-out categories or other distributions. |
| P11-FE50 | 0.496 | KEEP | Tests H-7 mechanism (density) and F-7 ('collective only') via PH-outlierness; verdict does not address PH-outlierness as a signal mechanism. |
| P11-FE501 | 0.447 | KEEP | The verdict does not address per-token backtracking-direction geometry or whether D-bucket has detectable geometric signatures. |
| P11-FE507 | 0.476 | KEEP | The verdict does not test whether batch-dispersion alone (a label-free scalar) can replace the supervised DoM as the explanatory mechanism, leaving this refutation test unresolved. |
| P11-FE513 | 0.435 | KEEP | Rank-trajectory analysis across layers for latent-reasoning signatures is a mechanistic probe orthogonal to the matched-cost routing comparisons in the verdict. |
| P11-FE529 | 0.481 | KEEP | The verdict confirms L19 utility for selectivity but does not measure SafeConstellations' steering-effectiveness ratio Eff(ℓ), leaving the layer-optimality criterion under that framework unresolved. |
| P11-FE534 | 0.466 | KEEP | Tests steering on a specific abstained subset; verdict's broad 'no arm adds value' may cover steering overall, but this experiment targets a fine-grained regime (bottom-50% failures) not explicitly ruled out in the verdict summary. |
| P11-FE536 | 0.463 | KEEP | Core token-substitution mechanism question survives; only the H-1 comparison baseline is lost, but prefix-steering test itself is still relevant. |
| P11-FE539 | 0.431 | KEEP | Tests D²HScore vs DoM at 7B scale; verdict does not directly test this specific probe or model scale. |
| P11-FE543 | 0.444 | KEEP | D²HScore vs L19 DoM comparison at 7B scale tests F-9 collapse claim, which the verdict does not address. |
| P11-FE547 | 0.411 | KEEP | Layer-specificity (L27 vs L19) is not directly addressed in the verdict; confirming DoM works across model families does not settle whether final-layer would match L19 on the same dataset. |
| P11-FE550 | 0.411 | KEEP | Layer-specificity (L27 vs L19) is not directly addressed in the verdict; confirming DoM works across model families does not settle whether final-layer would match L19 on the same dataset. |
| P11-FE553 | 0.486 | KEEP | The verdict validates the prefill DoM's selectivity performance but does not test the SteeringSafety/Arditi KL<0.1 steering-validity filter, which remains a live methodological gate. |
| P11-FE561 | 0.419 | KEEP | The verdict doesn't address whether text-only models can replicate F-2's 0.7731 AUROC; it focuses on the final routing choice (escalation) rather than interpretation of the signal. |
| P11-FE563 | 0.472 | KEEP | The verdict provides F-8's single operating point (50% coverage) but does not provide the full selective-prediction AURC curve or comparisons to baseline methods, leaving the standard scalar metric unresolved. |
| P11-FE564 | 0.417 | KEEP | Spline calibration is a post-hoc improvement to F-8's ECE and doesn't challenge the foundational signal; the verdict doesn't address calibration quality. |
| P11-FE568 | 0.485 | KEEP | The verdict does not address F-7's D-bucket geometric signature or whether it is a small-N artifact; this matched-/P/ bootstrap test remains orthogonal to the escalation-closure findings. |
| P11-FE57 | 0.444 | KEEP | Identical to FE60; K=20,40 SC ceiling numbers remain unmeasured. |
| P11-FE572 | 0.485 | KEEP | Identical to P11-FE568; the verdict does not test whether the distinctive D-bucket signature survives size-matching, leaving this confound check live. |
| P11-FE593 | 0.412 | KEEP | Latent-Trajectory signals are not directly measured in the verdict; the verdict shows DoM is strong but does not refute that LT signals could also be competitive with DoM. |
| P11-FE60 | 0.444 | KEEP | Verdict provides K=8 data (H-J: 0.554 accuracy) but not K=20,40; SC ceiling at higher K is not yet established. |
| P11-FE602 | 0.460 | KEEP | Verdict doesn't address multi-layer Gaussian schedules; orthogonal depth-schedule optimization question survives. |
| P11-FE612 | 0.426 | KEEP | Verdict does not test whether peak DoM layer is absolute (L19) or fractional, or whether it shifts across model depths. |
| P11-FE614 | 0.439 | KEEP | Black-box-only mpnet PH predictor benchmark is orthogonal; verdict does not settle whether F-2 requires white-box access. |
| P11-FE626 | 0.462 | KEEP | Verdict doesn't address attention-based PH; substrate question (residual vs attention) remains orthogonal. |
| P11-FE633 | 0.494 | KEEP | Tests whether L19 signal is single-direction or multi-cluster via mixture-of-probes; verdict addresses routing superiority of escalation but not underlying signal structure. |
| P11-FE634 | 0.464 | KEEP | Verdict doesn't address layer-specific optimization; layer-choice question remains orthogonal to escalation conclusions. |
| P11-FE635 | 0.417 | KEEP | The verdict doesn't directly address single-shot DoM steering improving K=1 accuracy within a fixed budget; its general closure about matched-cost comparisons targets routing strategies relative to escalation, not inference-time steering interventions. |
| P11-FE641 | 0.414 | KEEP | Verdict establishes DoM AUROC 0.7731 at prefill/final-token positions but does not probe intermediate trajectory positions or test refusal-cliff/mid-CoT analogs. |
| P11-FE645 | 0.429 | KEEP | Tests RAPTOR ridge-logistic vs DoM; verdict does not test RAPTOR-specific architecture or regularization scheme. |
| P11-FE649 | 0.500 | KEEP | Foundational re-extraction of per-head states needed for Refutation tests of 2602.01893; verdict does not address this prerequisite data. |
| P11-FE65 | 0.426 | KEEP | Verdict confirms L19 DoM but does not perform circuit decomposition or minimize the head set required for the direction. |
| P11-FE651 | 0.470 | KEEP | The verdict validates the F-2 AUROC (0.7731) but does not test whether sink-attention-mass is a confound or whether the signal survives partialling, leaving the mechanistic validity unresolved. |
| P11-FE661 | 0.434 | KEEP | Verdict does not execute H-1's steerability sweep or measure the collapse budget L+; this three-stage curve fit on steering magnitude is unpaired by any outcome in SPEC v7. |
| P11-FE665 | 0.413 | KEEP | Verdict confirms full-residual DoM AUROC 0.7731 but does not test whether sparse AU subsets match or exceed this performance. |
| P11-FE670 | 0.434 | KEEP | Verdict does not test per-head attribution or sparsity of the prefill-DoM circuit; this mechanistic ablation is orthogonal to the cost-matched hypothesis tests. |
| P11-FE673 | 0.437 | KEEP | Verdict does not address ESR-artifact interpretation or correctness-split analysis of direction orthogonality. |
| P11-FE674 | 0.458 | KEEP | Verdict doesn't address F-7 bucket interpretation or consistency-circuit framing; mechanism question is orthogonal. |
| P11-FE705 | 0.439 | KEEP | Verdict's general closure about introspection probes is broad, but this experiment tests a specific hypothesis about multi-layer LoRA's properties relative to single-layer DoM, which remains open. |
| P11-FE712 | 0.458 | KEEP | Verdict doesn't address ReBalance's overthinking/underthinking axis; label-alignment question is orthogonal. |
| P11-FE713 | 0.456 | KEEP | Verdict doesn't address confidence-variance feature redundancy; orthogonal to both F-9 redundancy and F-2 baseline. |
| P11-FE715 | 0.416 | KEEP | Verdict confirms prefill DoM 0.7731 and final-token 0.7186 but does not probe intermediate step-boundary token positions. |
| P11-FE716 | 0.488 | KEEP | Tests ReBalance steering method on Qwen-2.5-1.5B; verdict addresses verifier/probe methods but not steering interventions. |
| P11-FE723 | 0.431 | KEEP | Tests cross-model DoM transfer (1.5B→7B); verdict is agnostic to transfer mechanisms across model scales. |
| P11-FE724 | 0.532 | KEEP | Tests whether nonlinear probe architecture extracts more signal than linear on L19; verdict addresses routing strategies but not probe capacity/signal extraction. |
| P11-FE727 | 0.434 | KEEP | Verdict confirms prefill DoM (0.7731) but does not test Miao-Ungar ridge probes on cached P11 activations to determine if regularized fitting beats mass-mean. |
| P11-FE731 | 0.452 | KEEP | The verdict closes introspection-vs-escalation at matched cost but does not directly test Miao-Ungar's two-stage calibration pipeline. |
| P11-FE732 | 0.470 | KEEP | Tests online adaptation of a single probe (DoM), not whether introspection beats escalation; orthogonal to verdict's comparison of competing arms. |
| P11-FE737 | 0.420 | KEEP | Verdict does not test Sun et al. trajectory feature construction or compare step-wise trajectory probes against single-layer L19 DoM. |
| P11-FE76 | 0.433 | KEEP | Verdict does not test tuned-lens translators with KL distillation or iForest/LOF on prediction trajectories for detecting incorrectness. |
| P11-FE778 | 0.451 | KEEP | Duplicate of FE783; the verdict does not test PANL-equivalent probes on Qwen 1.5B MATH-500. |
| P11-FE783 | 0.451 | KEEP | The verdict does not address PANL probes; the verdict confirms escalation beats introspection probes generically but does not test PANL specifically. |
| P11-FE788 | 0.418 | KEEP | The verdict doesn't address truncation-sensitivity degradation curves; while cache defects were corrected, a continuous sweep of CoT truncation is orthogonal to the main routing findings. |
| P11-FE791 | 0.521 | KEEP | Identical to FE798; domain-specific validation of F-1/F-2/F-3/F-5 is not answered by the verdict. |
| P11-FE795 | 0.418 | KEEP | Identical to FE788—the verdict doesn't address truncation-sensitivity curves as a separate investigation. |
| P11-FE798 | 0.521 | KEEP | Unblocks FE22 and tests F-1/F-2/F-3/F-5 on Abstract-CoT domain; verdict validates these findings on MATH-500/BBH but not on Abstract-CoT. |
| P11-FE804 | 0.434 | KEEP | Verdict confirms prefill DoM works but does not test SAE + contrastive filtering on P11 activations to determine if they lift ≥0.04; general matched-cost closure does not refute this premise. |
| P11-FE807 | 0.413 | KEEP | Verdict refutes H-M (nested routing loses to plain cascade) but does not directly address whether multi-layer SAID-style probes achieve higher AUROC than single-layer DoM. |
| P11-FE81 | 0.466 | KEEP | adjudicator returned no/invalid verdict — kept |
| P11-FE817 | 0.439 | KEEP | Future Lens linear future-state probe as a null hypothesis for F-2 is not addressed by the verdict. |
| P11-FE840 | 0.419 | KEEP | The verdict doesn't evaluate head-level structure or the relationship between prefill/final circuits and induction/copying heads; this is orthogonal to the routing findings. |
| P11-FE843 | 0.462 | KEEP | Verdict doesn't address direction multiplicity or INLP; orthogonal to escalation and redundancy claims. |
| P11-FE846 | 0.465 | KEEP | Training-time SU→SFT distillation is a separate pipeline (amortizing supervision at train time vs inference-time probing); verdict addresses inference-time introspection, not training-time approaches. |
| P11-FE852 | 0.475 | KEEP | The verdict refutes spectral-alpha for escalation (H-K) but does not test whether the F-1 breathing trajectory narrative survives re-narration under eDim instead of PR, a distinct mechanistic question about trajectory stability. |
| P11-FE856 | 0.414 | KEEP | Verdict confirms DoM as a working correctness direction but does not run the head-to-head steering comparison against Lugoloobi & Russell's difficulty probe. |
| P11-FE859 | 0.420 | KEEP | The verdict uses L19 as the baseline for confgate but doesn't evaluate whether L19 is the globally optimal layer/position; this pressure-test is orthogonal to the routing-strategy verdict. |
| P11-FE86 | 0.466 | KEEP | Circuit identification (H-13) tests structural understanding, not whether DoM improves accuracy; orthogonal to verdict's comparison of competing performance-boosting arms. |
| P11-FE874 | 0.437 | KEEP | Verdict establishes that certain claims hold (H-I, J confirmed; H-K, M refuted) but does not establish whether Qwen is in strong-superposition regime. |
| P11-FE938 | 0.429 | KEEP | Tests SAE+Ising manifold discovery; verdict refutes spectral-alpha but does not directly test this specific methodology. |
| P11-FE939 | 0.429 | KEEP | Tests L19 spectral gap / stable rank as predictor of DoM stability; verdict does not address weight-matrix spectral properties. |
| P11-FE953 | 0.411 | KEEP | The verdict shows DoM works across Qwen, SmolLM2, Gemma, and OLMo-2, but does not test optimizer-contingency (SAM vs AdamW); architecture robustness does not directly refute the possibility of optimizer-dependence. |
| P11-FE958 | 0.411 | KEEP | The verdict shows DoM works across Qwen, SmolLM2, Gemma, and OLMo-2, but does not test optimizer-contingency (SAM vs AdamW); architecture robustness does not directly refute the possibility of optimizer-dependence. |
| P7-FE4 | 0.437 | KEEP | Verdict does not address topological data analysis methods or CoLA pipeline replication. |
| P8-FE7 | 0.441 | KEEP | PH reimplementation validation via Stolz sphere-cube is methodological gatekeeping, not addressed by verdict. |
| P8-FE8 | 0.416 | KEEP | Verdict confirms F-10's finding (spectral-alpha subtracts signal) but does not address whether the PH-Gaussian null survives re-summarization with diameter-normalized persistence. |
| P8-FE9 | 0.478 | KEEP | The verdict does not test whether persistent homology on per-step embeddings adds signal orthogonal to the DoM (a different substrate from the per-token residuals in F-10), leaving the topology-plus-DoM hypothesis live. |

Review: wrongly-closed FEs can be revived with
`python update_status.py <id> READY`.
