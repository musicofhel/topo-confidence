# Moot sweep — FE19 (2026-06-11)

**Verdict:** FE19 verification-routing test (2026-06-10): H-19 REFUTED. Routing low-confidence problems to self-verification / re-generation (C_exact) LOSES at 1.5B — self-correction hurts; the verification pass degrades net accuracy versus simply abstaining or escalating. Same-model K-routing was already killed: gating K=8 self-consistency by prefill confidence is hull-dominated (recoverable problems live at mid-confidence, FE19/F-14). What survives is selective prediction (abstain/escalate on the readout) and the cross-model 1.5B-to-7B hybrid cascade (+3.38pp significant vs random-mix hull). Experiments premised on self-verification, self-correction loops, or same-model verify-then-retry routing at small scale are dead; cross-model escalation and abstention remain live.

**Shortlisted:** 250 | **MOOT:** 14 | **ANSWERED:** 0 | **KEEP:** 236

| FE | sim | verdict | reason |
|---|---|---|---|
| P11-FE185 | 0.423 | MOOT | Tests routing low-confidence problems to self-clarification/re-generation; verdict directly refutes 'routing low-confidence problems to self-verification / re-generation (C_exact) LOSES'. |
| P11-FE312 | 0.657 | MOOT | Compares two same-model K-routing methods; verdict explicitly states 'Same-model K-routing was already killed' and both Ada-BoK and C_exact are inferior to selective prediction. |
| P11-FE33 | 0.606 | MOOT | Cluster-conditioned verification routing is still verification-based routing, which the verdict refutes. |
| P11-FE353 | 0.492 | MOOT | Tests persistence reprompt to see if reconsidered answers are better; directly contradicted by FE19's finding that 'self-correction hurts' and verification passes degrade accuracy. |
| P11-FE407 | 0.623 | MOOT | Same-model K-routing (routing between K=1 and K=8); verdict says this approach was already killed because recoverable problems don't align with any single prefill signal. |
| P11-FE484 | 0.618 | MOOT | Variant of verification routing using epsilon-to-flip instead of prompt verification; verdict refutes all verification-based routing. |
| P11-FE487 | 0.618 | MOOT | Duplicate of FE484; variant of verification routing refuted by verdict. |
| P11-FE490 | 0.419 | MOOT | Verdict refutes self-correction entirely ('self-correction hurts'); per-sentence regeneration is a self-correction mechanism that the verdict has already closed. |
| P11-FE535 | 0.416 | MOOT | Tests whether FASB backtracking (self-correction) improves selective prediction; verdict refuted self-correction at small scale. |
| P11-FE58 | 0.423 | MOOT | Duplicate of P11-FE61; tests K=8 consistency-rate which verdict refutes as dead same-model routing. |
| P11-FE61 | 0.423 | MOOT | Tests K=8 consistency-rate as selector; verdict categorically states 'same-model routing at small scale are dead' and this applies to K-routing. |
| P11-FE610 | 0.688 | MOOT | Experiment depends on H-19 (verification routing) being implemented; verdict refutes that verification routing improves accuracy. |
| P11-FE780 | 0.498 | MOOT | Explicitly framed as validation 'before committing to verify-then-correct as canonical'; FE19 refuted verify-then-correct (C_exact loses), mooting the premise. |
| P11-FE785 | 0.498 | MOOT | Identical to FE780: premise depends on verify-then-correct viability, which FE19 refuted. |
| P10-FE12 | 0.433 | KEEP | Tests steering magnitude calibration via closed-form formula; steering optimization is orthogonal to verification-routing failure. |
| P10-FE18 | 0.404 | KEEP | Tests dynamic-alpha steering intervention; orthogonal to verdict which refutes self-correction routing, not steering itself. |
| P10-FE25 | 0.423 | KEEP | Tests mechanistic grounding of L19 DoM via SAE-EAP overlap; verdict refutes routing strategy, not directional validity. |
| P10-FE29 | 0.517 | KEEP | Dynamic-layer selection for selective prediction optimizes a surviving strategy; orthogonal to routing verdict. |
| P10-FE30 | 0.455 | KEEP | Instruments steering with adaptive confidence gating; the verdict refutes routing to self-verification, not steering as a technique. |
| P10-FE34 | 0.401 | KEEP | The verdict refutes same-model routing/verification, not steering optimization; the verdict itself suggests low-dim composition search is the fix P10-FE34 proposes. |
| P10-FE42 | 0.443 | KEEP | Tests SAE-feature filtering for signal improvement; orthogonal to routing-to-verification and self-correction strategies. |
| P10-FE44 | 0.437 | KEEP | Tests CAA vs FGAA steering arms; FE19 refutes self-verification, not steering efficacy in general. |
| P10-FE45 | 0.446 | KEEP | Tests prompt-conditional steering vectors; orthogonal to FE19's same-model routing refutation. |
| P10-FE46 | 0.437 | KEEP | Tests learned conditional steering vs static vectors; FE19 refutes same-model routing, not steering method design. |
| P10-FE5 | 0.426 | KEEP | Mechanistic investigation of uncertainty vs correctness confounding in DoM; orthogonal to routing/verification refutation. |
| P10-FE8 | 0.429 | KEEP | Tests per-token gradient steering (PPLM) vs static directions; steering optimization is independent of same-model verification-routing failure. |
| P11-FE100 | 0.401 | KEEP | Tests data-efficiency saturation of prefill DoM; orthogonal to verdict which refutes self-verification routing. |
| P11-FE103 | 0.448 | KEEP | LEACE-based refutation of F-9; tests layer composition independence, not routing or self-correction. |
| P11-FE1047 | 0.441 | KEEP | Tests cross-model transfer between 1.5B variants; verdict addresses 1.5B-to-7B cascade, not 1.5B-to-1.5B transfer. |
| P11-FE105 | 0.430 | KEEP | Tests layer-wise FFN/MHSA architecture to localize L19; independent characterization, orthogonal to routing verdict. |
| P11-FE106 | 0.415 | KEEP | Component decomposition of DoM signal (MHSA vs FFN); orthogonal to verdict on self-correction. |
| P11-FE1072 | 0.404 | KEEP | Tests ICA vs PCA for direction discovery; orthogonal to verdict which addresses routing, not decomposition methods. |
| P11-FE108 | 0.471 | KEEP | Tests MHSA-specific CoE-60 decomposition against F-9 redundancy claim; orthogonal to verification routing. |
| P11-FE1096 | 0.443 | KEEP | Tests complementarity of CoE-60 and DoM error residuals; orthogonal to routing and self-verification verdict. |
| P11-FE1106 | 0.422 | KEEP | Tests layer-wise convergence residual as alternative correctness signal; orthogonal to verdict's routing refutation. |
| P11-FE110625A | 0.441 | KEEP | Tests cross-domain transfer (TriviaQA→MATH-500); verdict does not address domain-transfer generalization. |
| P11-FE110625B | 0.451 | KEEP | Tests whether DoM AUROC degrades monotonically with step count; FE19 doesn't address single-step vs multi-step signal quality. |
| P11-FE110625C | 0.399 | KEEP | The verdict doesn't refute layer-wise saturation properties; characterizing whether L19 is a peak or plateau remains orthogonal to routing verdict. |
| P11-FE1108 | 0.413 | KEEP | Tests whether convergence speed through layer trajectory predicts correctness; orthogonal to verdict about self-verification. |
| P11-FE111 | 0.411 | KEEP | Tests whether L19 is the correctness-optimal layer; verdict doesn't address layer selection. |
| P11-FE1125 | 0.406 | KEEP | Tests whether nonlinear value heads extract more signal than linear DoM; orthogonal to self-verification routing mechanisms. |
| P11-FE1137 | 0.435 | KEEP | Tests if F-8 selective prediction survives VPO/GRPO; FE19 confirmed it works on base model but did not test diversity-preserving variants. |
| P11-FE1158 | 0.489 | KEEP | Tests cross-layer predictability as a mechanistic account of correctness; orthogonal to FE19's findings about routing and selective prediction. |
| P11-FE1189 | 0.400 | KEEP | The verdict doesn't refute F-9's redundancy findings; testing whether CoE-60 differs under non-linear scoring is orthogonal to routing. |
| P11-FE1195 | 0.477 | KEEP | Tests convergence-ratio feature vs DoM (F-2); orthogonal to routing verdict. |
| P11-FE1196 | 0.470 | KEEP | Tests layer-wise convergence as interpretation of F-4 asymmetric collapse; independent of H-19. |
| P11-FE1198 | 0.404 | KEEP | Tests layer-wise RSA geometry across all 28 layers; orthogonal to verification routing verdict. |
| P11-FE1214 | 0.429 | KEEP | Tests whether DoM survives TrOPD distillation; independent of same-model routing verdict, orthogonal mechanistic question. |
| P11-FE1217 | 0.425 | KEEP | Tests whether per-layer ensemble recovers signal against F-9's redundancy claim; orthogonal to H-19 refutation. |
| P11-FE123 | 0.411 | KEEP | Tests whether paired contrast training improves DoM NIE; verdict doesn't address this methodological question. |
| P11-FE1237 | 0.420 | KEEP | F-4 asymmetric-collapse asymmetry is not addressed by verification-routing verdict. |
| P11-FE1244 | 0.465 | KEEP | Tests pipeline-specificity of DoM (base vs instruct); orthogonal to verification routing verdict. |
| P11-FE126 | 0.419 | KEEP | Answer-length stratified CoE vs single-layer comparison is not addressed by routing verdict. |
| P11-FE129 | 0.455 | KEEP | Tests discrete codebook structure in L19; orthogonal to verdict's refutation of inference-time routing strategies. |
| P11-FE131 | 0.475 | KEEP | Tests k-means discretization sanity check; orthogonal to routing verdict. |
| P11-FE134 | 0.425 | KEEP | Tests whether multi-layer probes lift signal; orthogonal to verdict's refutation of self-correction routing. |
| P11-FE163 | 0.408 | KEEP | Layer-selection ablation for EigenScore; unrelated to self-verification or routing mechanisms the verdict refutes. |
| P11-FE166 | 0.408 | KEEP | Layer-selection ablation for EigenScore; unrelated to self-verification or routing mechanisms the verdict refutes. |
| P11-FE168 | 0.402 | KEEP | Compares NLI-based PDS to prefill DoM at matched compute; orthogonal to verdict which addresses routing, not signal comparison. |
| P11-FE18 | 0.429 | KEEP | Tests post-answer-newline DoM timing for bucket prediction; FE19 refutes prefill routing but doesn't explicitly rule out post-answer-token discrimination. |
| P11-FE182 | 0.569 | KEEP | Adaptive-stopping based on token log-probability is orthogonal to the verdict; uses logits already available, not verification routing. |
| P11-FE187 | 0.427 | KEEP | Tests label-free abstention policies (which survive per the verdict); comparing clarification vs entropy is orthogonal to routing refutation. |
| P11-FE193 | 0.421 | KEEP | Multi-layer vs single-layer signal distribution is not addressed by verdict about self-correction routing. |
| P11-FE201 | 0.406 | KEEP | Tests OOD transfer of prefill DoM across domains; orthogonal to verification routing, which refutes self-correction routing. |
| P11-FE204 | 0.444 | KEEP | Tests directional stationarity of prefill DoM across generation timesteps; orthogonal to routing and verification strategies. |
| P11-FE207 | 0.444 | KEEP | Identical to P11-FE204; tests directional stationarity during generation, orthogonal to routing verdict. |
| P11-FE212 | 0.413 | KEEP | Tests layer selection for correctness probing; verdict doesn't address whether L19 is optimal vs other layers. |
| P11-FE213 | 0.477 | KEEP | Tests multi-layer probe ensemble (F-9 redundancy); orthogonal to routing verdict. |
| P11-FE216 | 0.449 | KEEP | Tests whether F-2's AUROC is a discrete-metric artifact; FE19 validates selective prediction but doesn't address metric choice. |
| P11-FE217 | 0.502 | KEEP | Testing CoE-60 backup vs redundancy is orthogonal to FE19's findings about verification routing and doesn't depend on verify-then-retry viability. |
| P11-FE236 | 0.418 | KEEP | Investigates F-9 redundancy across generation-length partitions; orthogonal to verdict on self-verification routing. |
| P11-FE240 | 0.427 | KEEP | Tests CoE-60 redundancy via layer-regional decomposition; orthogonal to H-19 verification-routing refutation. |
| P11-FE246 | 0.401 | KEEP | Measuring neural collapse properties via canonical tools is independent of the routing/verification verdict. |
| P11-FE247 | 0.439 | KEEP | Tests cluster-size consensus as a confidence signal for selective prediction; verdict killed routing-to-verification, not raw self-consistency statistics as a signal. |
| P11-FE248 | 0.444 | KEEP | Tests verbalized confidence as an alternative signal to DoM; verdict killed routing-to-verification but does not directly compare alternative confidence signals. |
| P11-FE249 | 0.461 | KEEP | FE249 tests whether RL training changes geometry relationships; the verdict refutes inference-time routing to self-verification, not whether training procedures affect geometry. |
| P11-FE254 | 0.453 | KEEP | Tests whether concatenating prefill and final tokens beats prefill-only probe; orthogonal to verdict about routing strategies. |
| P11-FE260 | 0.418 | KEEP | Black-box engineered-feature comparison against prefill DoM is independent of self-correction routing conclusions. |
| P11-FE262 | 0.517 | KEEP | Cross-model transferability is orthogonal; tests whether DoM represents dataset structure vs model-specific geometry. |
| P11-FE263 | 0.430 | KEEP | Tests whether F-2 reduces to sparse vocab-projectable columns; mechanistic characterization independent of routing strategy. |
| P11-FE267 | 0.404 | KEEP | Tests sparse SAE features across layers; orthogonal to verdict which concerns routing strategy, not feature sparsity. |
| P11-FE27 | 0.425 | KEEP | Tests whether linear DoM ceiling is topological vs Bayes-noise; mechanistic investigation orthogonal to routing/verification refutation. |
| P11-FE283 | 0.482 | KEEP | Tests L19 causal role (F-2); orthogonal to routing verdict. |
| P11-FE286 | 0.436 | KEEP | Tests L19 signal specificity via layer reordering, orthogonal to FE19's routing/verification verdict. |
| P11-FE287 | 0.409 | KEEP | Tests whether WiC and DoM peak at different layers; verdict doesn't address feature-type comparisons. |
| P11-FE288 | 0.453 | KEEP | Tests whether L19 is special for DoM across layer sweep; orthogonal to verdict about routing and self-verification. |
| P11-FE289 | 0.485 | KEEP | Tests directional stability across checkpoints (H-8, H-10); orthogonal to verification-routing verdict. |
| P11-FE298 | 0.425 | KEEP | Tests label-free prompting baseline (CF) against supervised probe; verdict does not address whether prompting-based selectors can be viable. |
| P11-FE30 | 0.401 | KEEP | The verdict mentions cross-model cascade survives; this experiment tests whether the breathing phenomenon correlates across models, orthogonal to routing efficacy. |
| P11-FE300 | 0.466 | KEEP | Tests CCA redundancy across three confidence signals; independent of verification routing. |
| P11-FE303 | 0.405 | KEEP | Tests task-specificity of prefill DoM (TriviaQA→MATH transfer); orthogonal to verification routing verdict. |
| P11-FE307 | 0.476 | KEEP | Tests Damani MLP probe design (F-2 probe optimality); orthogonal to routing verdict. |
| P11-FE313 | 0.445 | KEEP | Tests direction vs norm stability of DoM probe; FE19 validates selective prediction but doesn't address probe-level stability properties. |
| P11-FE315 | 0.403 | KEEP | Tests jointly-trained multi-layer dynamic probe; orthogonal to verdict which refutes same-model routing, not multi-layer probes. |
| P11-FE322 | 0.419 | KEEP | Pruning analysis and layer-necessity testing are orthogonal to self-correction conclusions. |
| P11-FE329 | 0.406 | KEEP | Tests multi-layer DoM ensemble vs single-layer L19; orthogonal to verdict which addresses routing strategies, not aggregation. |
| P11-FE334 | 0.425 | KEEP | Tests cross-model layer specificity; verdict does not address whether L19 generalizes across architectures. |
| P11-FE335 | 0.489 | KEEP | Tests CoE-C as an alternative gating signal for selective prediction; FE19 validated selective prediction, so testing different gate signals remains viable. |
| P11-FE337 | 0.489 | KEEP | Identical to FE335; CoE-C gating for selective prediction is unaffected by FE19's verdict against verify-then-retry. |
| P11-FE343 | 0.422 | KEEP | Tests L2-norm OOD properties of DoM; orthogonal mechanistic question about direction structure. |
| P11-FE346 | 0.525 | KEEP | Step-boundary DoM probe is orthogonal; tests where in the sequence the signal concentrates, not routing strategy. |
| P11-FE354 | 0.421 | KEEP | Cross-domain generalization of prefill DoM is not addressed by the verdict. |
| P11-FE357 | 0.456 | KEEP | Tests step-boundary probe design against prefill DoM; the verdict does not address probe architecture comparisons. |
| P11-FE361 | 0.449 | KEEP | Tests L19 polysemanticity ceiling; FE19 doesn't address whether F-2's single-direction signal is over-constrained. |
| P11-FE38 | 0.485 | KEEP | Tests topological invariance (F-3 mechanism revision); orthogonal to verification-routing verdict. |
| P11-FE380 | 0.450 | KEEP | Tests alternative OOD detection mechanism; FE19 validates selective prediction but doesn't refute prototype-based alternatives. |
| P11-FE382 | 0.407 | KEEP | Tests steering via directional injection during generation; distinct from self-verification routing and not addressed by the verdict's refutation of verification-based routing. |
| P11-FE389 | 0.453 | KEEP | Tests TELLME editing on model geometry; the verdict does not address representation learning approaches. |
| P11-FE399 | 0.487 | KEEP | adjudicator returned no/invalid verdict — kept |
| P11-FE40 | 0.459 | KEEP | Tests gradient steering on low-DoM problems; the verdict does not address steering as a technique, only routing to self-verification. |
| P11-FE401 | 0.553 | KEEP | CISC confidence-weighted majority vote is a different confidence aggregation method orthogonal to the routing strategies tested. |
| P11-FE403 | 0.449 | KEEP | Compares PRM vs DoM as standalone verifiers; FE19 refutes routing/gating strategies, not signal dominance comparisons. |
| P11-FE404 | 0.450 | KEEP | Tests PRM-guided search (external verifier), not same-model self-correction; orthogonal to FE19. |
| P11-FE405 | 0.480 | KEEP | PRM-based routing is cross-model escalation, which verdict explicitly says survives; same-model routing (C_exact) is what was killed. |
| P11-FE408 | 0.434 | KEEP | Tests sparse MLP neuron mechanisms for correctness, independent of FE19's routing/verification refutation. |
| P11-FE41 | 0.427 | KEEP | Passive reranking from K=8 cached samples is distinct from self-verification routing, which the verdict refutes. |
| P11-FE411 | 0.418 | KEEP | Tests whether self-sampling confidence estimation beats prefill-DoM selective prediction; orthogonal to verdict on self-correction routing. |
| P11-FE412 | 0.445 | KEEP | Tests whether raw self-consistency statistics suffice for selective prediction; verdict killed routing-to-verification and same-model verify-then-retry, not the use of self-consistency as a direct confidence signal. |
| P11-FE413 | 0.434 | KEEP | Tests L19 DoM robustness under EKBM alignment, orthogonal to FE19's self-verification verdict. |
| P11-FE414 | 0.466 | KEEP | Tests which gate signal (verbalized vs geometric) better routes to 7B; cross-model escalation survives, so comparison remains live. |
| P11-FE415 | 0.428 | KEEP | Tests whether 3-layer MLP beats linear F-2 probe; mechanistic characterization independent of routing strategy verdict. |
| P11-FE431 | 0.427 | KEEP | Confidence-based early-exit truncates generation; distinct from self-verification routing, which is refuted. |
| P11-FE432 | 0.413 | KEEP | Per-layer DoM sweep on R1-Distill-Qwen (duplicate FE438); orthogonal to verdict on self-correction routing. |
| P11-FE433 | 0.420 | KEEP | Calibration metrics (Brier/ECE) are independent of routing/self-correction findings. |
| P11-FE434 | 0.408 | KEEP | Tests temporal DoM features across CoT positions; unrelated to self-verification or same-model routing strategies the verdict refutes. |
| P11-FE437 | 0.427 | KEEP | Identical to FE431; confidence-based early-exit is not the self-verification routing refuted by FE19. |
| P11-FE438 | 0.413 | KEEP | Per-layer DoM sweep on R1-Distill-Qwen; orthogonal to verdict on self-correction routing. |
| P11-FE439 | 0.420 | KEEP | Calibration metrics (Brier/ECE) are independent of routing/self-correction findings. |
| P11-FE440 | 0.408 | KEEP | Tests temporal DoM features across CoT positions; unrelated to self-verification or same-model routing strategies the verdict refutes. |
| P11-FE444 | 0.496 | KEEP | Tests whether verbal confidence matches geometric substrates; orthogonal to FE19's verdict about verification routing, only challenges the choice of signal substrate. |
| P11-FE445 | 0.500 | KEEP | Explores SFT-based selective prediction as an alternative method; FE19 killed verify-then-correct but left selective prediction intact, so testing different substrates for selective prediction remains viable. |
| P11-FE45 | 0.403 | KEEP | Tests prefill/final orthogonality via sparse-shift-stability; orthogonal to verdict about self-correction routing. |
| P11-FE454 | 0.426 | KEEP | Compares verbalized confidence probes to residual-stream DoM; the verdict doesn't settle which probe type is richer. |
| P11-FE456 | 0.478 | KEEP | Tests alternative within selective-prediction family; verdict validates DoM works but does not compare against competing methods. |
| P11-FE461 | 0.448 | KEEP | Tests distillation fragility of L19-canonicality; FE19 doesn't address KD robustness. |
| P11-FE465 | 0.413 | KEEP | Tests H-8 (direction stability during training); verdict doesn't address training-time drift. |
| P11-FE47 | 0.417 | KEEP | Diagnostic of causal vs anticausal direction for F-2 using SSL; orthogonal to routing strategy verdict. |
| P11-FE472 | 0.421 | KEEP | Verdict refutes K-routing with confidence gating but doesn't directly answer whether DoM correlates with verbalized confidence; the mechanistic question remains open. |
| P11-FE473 | 0.433 | KEEP | Tests multi-turn DoM contamination, orthogonal to same-model routing verdict; selective prediction survives per FE19, but multi-turn transfer is untested. |
| P11-FE475 | 0.494 | KEEP | Tests multi-layer vs single-layer probe architecture; FE19 preserved selective prediction (F-8) so architectural refinements to the DoM signal remain viable. |
| P11-FE489 | 0.445 | KEEP | Tests H-5's decomposability hypothesis via within-chain entropy variance; orthogonal to the routing-to-verification verdict. |
| P11-FE491 | 0.429 | KEEP | Replicates F-2 via SAE sparse features; FE19 affirms selective prediction but doesn't settle whether sparse features suffice. |
| P11-FE492 | 0.440 | KEEP | Tests OOD robustness of dense vs sparse-feature probes; verdict confirms selective prediction survives but does not address OOD generalization. |
| P11-FE50 | 0.421 | KEEP | PH-outlierness as a mechanism for anomaly detection is orthogonal to the self-correction routing verdict. |
| P11-FE501 | 0.449 | KEEP | Tests F-7's claim that D-bucket lacks geometric signature via backtracking direction; orthogonal to FE19's routing verdict. |
| P11-FE507 | 0.489 | KEEP | Tests label-free dispersion as an alternative to supervised DoM; FE19 preserved the selective prediction mechanism so testing alternatives to supervised probes remains live. |
| P11-FE517 | 0.409 | KEEP | Tests whether cycle-detection gating rivals DoM for selective prediction; verdict confirms selective prediction survives but doesn't compare CoRE vs DoM mechanisms. |
| P11-FE521 | 0.459 | KEEP | Tests CoE-60 redundancy, orthogonal to the verdict's refutation of self-verification routing. |
| P11-FE524 | 0.418 | KEEP | Cosine-stability of DoM across token positions in F-3 orthogonality testing is orthogonal to routing findings. |
| P11-FE525 | 0.418 | KEEP | Cross-position DoM transferability is not addressed by verdict about self-correction routing. |
| P11-FE529 | 0.444 | KEEP | Tests layer selection by SafeConstellations' steering-effectiveness criterion; orthogonal to routing strategy and verification-based routing. |
| P11-FE532 | 0.443 | KEEP | Tests head-distribution hypothesis via per-head probing; orthogonal to routing-to-verification verdict about single-direction dominance. |
| P11-FE533 | 0.406 | KEEP | Tests cross-position probe transfer to falsify F-3's orthogonality; unrelated to the verdict's findings about self-verification routing. |
| P11-FE534 | 0.488 | KEEP | Tests external steering intervention on low-confidence problems; distinct from verify-then-retry (which FE19 refuted) and not explicitly covered by FE19's findings about routing or same-model recovery. |
| P11-FE55 | 0.456 | KEEP | Tests unsupervised persistent homology dimension as correctness predictor; orthogonal to verdict about routing. |
| P11-FE553 | 0.421 | KEEP | Tests steering-safety validity of DoM directional ablation; orthogonal to verdict's same-model routing refutation. |
| P11-FE557 | 0.400 | KEEP | The verdict doesn't address whether prefill DoM's 0.7731 AUROC is privileged or reachable by random directions; this is a fundamental characterization question about the signal itself. |
| P11-FE561 | 0.439 | KEEP | Tests text-only GCM prediction capability, independent of FE19's refutation of self-verification loops. |
| P11-FE563 | 0.462 | KEEP | Computes full AURC curve for F-8 selective-prediction; F-8 mechanism survives, full metric not answered by H-19. |
| P11-FE57 | 0.469 | KEEP | Identical to FE60; establishes SC ceiling for multiple findings, not fully answered by H-19 routing comparison alone. |
| P11-FE576 | 0.459 | KEEP | Tests per-layer intrinsic dimensionality signal; orthogonal to the verdict's findings about routing strategies. |
| P11-FE579 | 0.452 | KEEP | Tests whether geometric separability requires RL-tuning; the verdict refutes routing to self-verification, not the RL/SFT distinction. |
| P11-FE584 | 0.414 | KEEP | Layer-wise PID feature direction test; orthogonal to verdict on verification routing. |
| P11-FE593 | 0.446 | KEEP | Tests Latent-Trajectory signals vs F-2 single-direction; FE19 validates DoM for selective prediction but doesn't settle signal-family comparisons. |
| P11-FE594 | 0.424 | KEEP | Tests path-geometry cumulative-change signal; orthogonal to verdict's routing refutation. |
| P11-FE60 | 0.469 | KEEP | Establishes full SC ceiling across K=1..40; while H-19 routing headroom is addressed, broader baseline for F-7/F-8/F-9 is not fully settled by the verdict. |
| P11-FE607 | 0.466 | KEEP | Tests calibration properties of F-2 DoM scores; independent of H-19 routing outcome. |
| P11-FE609 | 0.403 | KEEP | Compares DoM calibration to verbalized confidence; orthogonal to verdict which addresses routing strategy, not calibration comparison. |
| P11-FE613 | 0.402 | KEEP | Tests dimensionality of correctness signal via SVD; orthogonal to verdict about same-model routing. |
| P11-FE614 | 0.437 | KEEP | Tests black-box PH features for prediction, independent of FE19's verdict on same-model verification. |
| P11-FE615 | 0.400 | KEEP | The verdict refutes that K=8 routing helps, but doesn't refute the premise that K=8 samples can be analyzed for view-dependence replication. |
| P11-FE616 | 0.451 | KEEP | Tests methodological robustness of PH (FTS vs V-R), orthogonal to FE19's refutation of self-correction routing. |
| P11-FE618 | 0.438 | KEEP | Tests OOD generalization of selective prediction (which FE19 confirmed survives); FE19 settles in-domain performance only. |
| P11-FE622 | 0.411 | KEEP | Tests whether DoM is truly epistemic (evidence-conditioned) vs distribution-specific; verdict confirms selective prediction works but doesn't test this robustness property. |
| P11-FE624 | 0.429 | KEEP | Tests F-8 threshold transfer across MATH-500 to AIME; FE19 affirms selective prediction survives but doesn't test OOD generalization. |
| P11-FE632 | 0.410 | KEEP | Tests step-wise monotonicity for selective prediction during generation; verdict confirms final-readout selective prediction works but doesn't address per-CoT-step dynamics. |
| P11-FE633 | 0.468 | KEEP | Tests single-direction vs multi-cluster structure of L19 signal; orthogonal to verification routing verdict. |
| P11-FE634 | 0.421 | KEEP | Verdict does not address layer-locality questions; testing whether L19 is locally optimal is orthogonal to self-correction routing. |
| P11-FE641 | 0.408 | KEEP | Measures per-token probe AUROC along CoT trajectory; orthogonal to the verdict's findings about self-verification routing. |
| P11-FE656 | 0.401 | KEEP | The verdict doesn't refute the existence of answer-content information at prefill; it only refutes that routing by confidence gates helps. |
| P11-FE661 | 0.465 | KEEP | Tests steerability of L19 DoM and H-1 budget (not H-19); H-19 verdict does not address whether steering obeys the three-stage law. |
| P11-FE662 | 0.422 | KEEP | Enhances selective prediction with SPLIT-style joint training; verdict confirms selective prediction survives as viable strategy. |
| P11-FE67 | 0.409 | KEEP | Tests sparsity of the correctness circuit; verdict doesn't address whether high-AUROC heads exist or signal is distributed. |
| P11-FE673 | 0.494 | KEEP | Tests structure of prefil/final orthogonality under correct/incorrect split; orthogonal to FE19's verdict about routing and self-correction. |
| P11-FE674 | 0.521 | KEEP | Tests mechanism of F-7 (selective prediction), which survives; orthogonal to the routing verdict. |
| P11-FE677 | 0.402 | KEEP | Tests REP multi-basis composition vs DoM alone; orthogonal to verdict which refutes self-correction, not basis composition. |
| P11-FE685 | 0.482 | KEEP | Tests single vs multi-direction probe architecture (F-2, F-9); orthogonal to routing verdict. |
| P11-FE689 | 0.408 | KEEP | Tests whether TopK SAE recovers multi-dimensional correctness structure; unrelated to self-verification or routing strategies. |
| P11-FE694 | 0.438 | KEEP | Tests layer-pattern ablation for DoM extraction, orthogonal to FE19's self-verification refutation. |
| P11-FE698 | 0.447 | KEEP | Tests parameter-space backbone hypothesis via SVD; orthogonal to FE19's routing/correction verdict. |
| P11-FE7 | 0.431 | KEEP | Tests whether F-2 is 1-D or low-dim manifold; dimensionality is orthogonal to routing-strategy verdict, and selective prediction survives. |
| P11-FE703 | 0.505 | KEEP | Tests if learned cross-layer attention can collapse prefill/final orthogonality; orthogonal to verdict on routing strategy. |
| P11-FE705 | 0.418 | KEEP | Tests multi-layer LoRA-gated verifier training; different from self-verification routing tested by FE19. |
| P11-FE706 | 0.409 | KEEP | Tests robustness of DoM under perturbations; verdict confirms selective prediction works but doesn't test invariance under semantics-preserving edits. |
| P11-FE707 | 0.409 | KEEP | Tests whether nonlinear MLP transport recovers signal linear probes miss; orthogonal to the verdict's refutation of self-verification routing strategies. |
| P11-FE711 | 0.413 | KEEP | Tests whether prefill/final DoM relationship is non-linear; verdict is silent on this geometric question. |
| P11-FE712 | 0.428 | KEEP | Tests alignment of ReBalance's O/U partition with F-7 buckets; orthogonal to the routing hypothesis refutation. |
| P11-FE713 | 0.529 | KEEP | Variance feature redundancy test is orthogonal; tests feature engineering, not routing strategy. |
| P11-FE715 | 0.426 | KEEP | Tests whether step boundaries are privileged like prefill; orthogonal investigation of generality. |
| P11-FE716 | 0.439 | KEEP | Tests ReBalance steering efficacy, orthogonal to FE19's verdict on self-verification routing. |
| P11-FE720 | 0.422 | KEEP | Tests ridge-residualization to rule out capability confounds; orthogonal to verdict's routing refutation. |
| P11-FE722 | 0.430 | KEEP | Tests whether DoM or surface features drive F-8's 71.6%; FE19 affirms selective prediction works but doesn't settle the mechanism. |
| P11-FE723 | 0.533 | KEEP | Cross-model DoM transfer test is orthogonal; tests model-specificity of the representation, not routing strategy. |
| P11-FE724 | 0.476 | KEEP | Tests MLP vs linear probe architecture (F-2 ceiling); orthogonal to routing verdict. |
| P11-FE725 | 0.406 | KEEP | Tests attention-probe aggregation against F-3's orthogonality claim; unrelated to self-verification or routing strategies the verdict refutes. |
| P11-FE726 | 0.428 | KEEP | Tests robustness improvement to selective prediction (which survives per the verdict); orthogonal to routing/verification refutation. |
| P11-FE727 | 0.436 | KEEP | Tests ridge-probe methodology for DoM extraction, independent of FE19's self-verification failure. |
| P11-FE729 | 0.400 | KEEP | The verdict doesn't address whether prefill/final-token orthogonality is task-elicited vs. position-elicited; F-3's directional findings survive. |
| P11-FE732 | 0.484 | KEEP | Tests test-time online probe adaptation; distinct from self-correction loops that the verdict refuted. |
| P11-FE733 | 0.424 | KEEP | Enhances selective prediction (which verdict confirms survives) with finite-sample risk guarantees via LTT calibration. |
| P11-FE736 | 0.437 | KEEP | Tests step-marker subspace universality across architectures, orthogonal to FE19's routing verdict. |
| P11-FE737 | 0.483 | KEEP | Tests trajectory-feature family vs static DoM (F-9); orthogonal to routing strategy verdict. |
| P11-FE751 | 0.414 | KEEP | Length-band confound check on spectral α; orthogonal to verdict on self-correction. |
| P11-FE752 | 0.424 | KEEP | Tests spectral α temporal behavior in cached data; orthogonal methodological comparison to prior work. |
| P11-FE757 | 0.403 | KEEP | Formalizes selective prediction with conformal guarantees; verdict confirms approach survives but doesn't address formal validity packaging. |
| P11-FE771 | 0.470 | KEEP | Tests calibration mechanism underlying F-8 selective-prediction lift; survives since selective prediction remains live. |
| P11-FE775 | 0.428 | KEEP | Cohen's kappa is a reporting metric change for existing findings; the verdict doesn't refute the underlying F-2/F-9/F-7 measurements. |
| P11-FE776 | 0.409 | KEEP | Tests whether RFM kernel probe outperforms linear at L19; verdict doesn't address probe-family comparisons. |
| P11-FE78 | 0.436 | KEEP | Tests causal dimensionality of L19 DoM via CBE, independent of FE19's self-correction refutation. |
| P11-FE789 | 0.457 | KEEP | Tests whether step ordering matters for reasoning correctness; orthogonal to verdict about inference-time routing. |
| P11-FE796 | 0.457 | KEEP | Identical to FE789; tests step ordering robustness, not routing or self-verification strategies. |
| P11-FE805 | 0.454 | KEEP | Tests cross-layer transcoder circuits and mechanistic disjointness; orthogonal to verdict about routing. |
| P11-FE806 | 0.417 | KEEP | Dimensionality test (d_90) for correctness signal; orthogonal to verdict on self-correction routing. |
| P11-FE807 | 0.477 | KEEP | Tests SAID-style multi-layer probe architecture (F-9 refutation); orthogonal to routing verdict. |
| P11-FE808 | 0.430 | KEEP | Tests per-difficulty d_90 characterization of the L19 probe; independent mechanistic question, orthogonal to routing verdict. |
| P11-FE81 | 0.444 | KEEP | Tests H-13 circuit localization via ACDC; orthogonal to routing-to-verification verdict. |
| P11-FE817 | 0.418 | KEEP | Refutation test for F-2 using future-state probe; orthogonal to verdict on self-correction. |
| P11-FE821 | 0.427 | KEEP | Investigates template bias confounding in the prefill DoM signal; the verdict doesn't settle whether the signal is template-contaminated. |
| P11-FE827 | 0.419 | KEEP | Multi-layer signal screening at frozen-LLM level is not addressed by routing verdict. |
| P11-FE829 | 0.438 | KEEP | Tests per-layer probe-to-steering alignment, distinct from FE19's routing/verification verdict. |
| P11-FE832 | 0.406 | KEEP | Layer-band ablation for DoM; orthogonal to the verdict's refutation of self-verification routing strategies. |
| P11-FE84 | 0.408 | KEEP | Tests ACDC circuit robustness across baselines; orthogonal to the verdict's refutation of self-verification routing. |
| P11-FE840 | 0.406 | KEEP | Tests head-level circuit structure (copying vs hallucination-aware heads); unrelated to self-verification or routing strategies the verdict refutes. |
| P11-FE841 | 0.408 | KEEP | Adds calibration metrics (Brier, ECE) to F-2's existing AUROC result; the verdict does not address calibration measurement. |
| P11-FE842 | 0.446 | KEEP | Tests confidence-weighted voting on K=8; FE19 refutes gating/routing by confidence but doesn't explicitly address intra-sample vote weighting vs unweighted majority. |
| P11-FE843 | 0.442 | KEEP | Tests dimensionality of correctness signal via INLP; orthogonal to routing-to-verification verdict. |
| P11-FE849 | 0.464 | KEEP | Tests whether single L19 DoM loses orthogonal signal axes; independent of verification routing. |
| P11-FE858 | 0.426 | KEEP | Tests GRPO fragility of F-8; the verdict confirms selective prediction survives but doesn't answer robustness under GRPO. |
| P11-FE859 | 0.463 | KEEP | Tests whether other (layer, position) pairs beat L19; pressure-tests F-2 framing independent of H-19. |
| P11-FE86 | 0.410 | KEEP | Tests circuit structure via path-patching; verdict doesn't address whether a minimal sparse circuit exists. |
| P11-FE879 | 0.433 | KEEP | Tests per-problem isomer-instability mechanism via K=8 bond distributions; independent of routing strategy verdict. |
| P11-FE88 | 0.416 | KEEP | Robustness test of selective prediction (which verdict confirms survives); strengthens survivor mechanism. |
| P11-FE895 | 0.472 | KEEP | Tests whether GRPO-specific gradient effects cause F-4/F-7 phenomena; orthogonal to H-19's verdict on verification routing. |
| P11-FE938 | 0.417 | KEEP | SAE manifold discovery for correctness; orthogonal to verdict on self-correction loops. |
| P11-FE96 | 0.401 | KEEP | The verdict refutes routing/verification strategies, not whether the prefill DoM signal exists or its data-efficiency saturation properties. |
| P6-FE1 | 0.412 | KEEP | Tests whether affine-aligned transfer improves cross-scale DoM transfer; verdict shows cross-model cascade works but doesn't compare affine vs unaligned mechanisms. |
| P7-FE3 | 0.415 | KEEP | Re-tests F-10 nullity with alternate PH estimator; orthogonal to verdict on verification routing. |
| P8-FE2 | 0.432 | KEEP | Tests step-level CoE feature extraction for temporal signal; independent of whether same-model routing works. |
| P8-FE9 | 0.488 | KEEP | adjudicator returned no/invalid verdict — kept |
| P9-FE4 | 0.477 | KEEP | Tests PDS vs DoM signal quality for selective prediction; verdict validates DoM but does not rule out competing signals. |

Review: wrongly-closed FEs can be revived with
`python update_status.py <id> READY`.
