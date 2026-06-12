# Moot sweep — P11-FE-EDGEGEN (2026-06-11)

**Verdict:** SPEC v6 generalization-edge program (EXP-81/82/83, P11-FE-EDGEGEN, 2026-06-11): geometry FULLY CLOSED. No activation-geometry feature adds portable correctness signal over free baselines: token-cloud covariance spectrum has NEGATIVE incremental value over length (0.761 < 0.786) and is length-confounded (r=-0.58, no eigenvalue fix); the CoE/layer-depth profile's portable depth-grid form does not add (p=0.137) and transfers at chance-ish (<=0.63 cross-domain/cross-scale); prompt-token-cloud spectra lose to raw prompt-length (0.694 < 0.706, p=0.42, r=-0.89 collinear). H-C and H-E refuted; PH was already null (F-7/F-10). The free length+mean-logprob baseline is the most generalizing readout (LOCO 0.845, cross-scale 0.865) and is CONFIRMED cross-architecture on three held-out families (SmolLM2 0.810, Gemma-2 0.844, OLMo-2 0.838) — any proposed readout must beat it out-of-family to matter. Also: probe-reranking of K=8 samples LOSES to majority vote (H-F refuted); mean-pool early-abort signal stays below the prefill gate until ~200 tokens (H-G refuted); conformal certificates transfer zero-shot cross-scale but NO light-head recalibration (k<=64) yields a valid certificate cross-domain (MATH->BBH) — target-task accuracy is the ceiling. New free Tier-A pre-generation signal: prompt-length 0.71. Experiments premised on activation-geometry features (spectra, PH, layer profiles, token clouds) beating free signals, on probe-based sample reranking, or on early-abort from partial-generation pooling are dead unless they target IN-DOMAIN hidden-dim-bound use (raw CoE layer-profile +1.75pp in-domain survives).

**Shortlisted:** 250 | **MOOT:** 35 | **ANSWERED:** 5 | **KEEP:** 210

| FE | sim | verdict | reason |
|---|---|---|---|
| P10-FE38 | 0.438 | MOOT | Tests whether F-10's PH null is a substrate artifact. Verdict concluded 'Experiments premised on...PH...are dead unless they target IN-DOMAIN hidden-dim-bound use'—substrate testing on PH falls under the dead experiments clause. |
| P10-FE7 | 0.387 | MOOT | The premise that non-linear topological structure in residuals beats linear DoM is refuted by the verdict's finding that PH was null on residuals and no activation-geometry feature adds portable correctness signal. |
| P11-FE1039 | 0.393 | MOOT | Verdict refutes premise that activation-geometry spectral features beat free length+logprob baseline (tested cov-spectrum 0.7928 vs 0.845); SD-PS is another spectral metric in that category. |
| P11-FE108 | 0.401 | MOOT | Verdict explicitly refutes the premise: 'CoE/layer-depth profile... does not add (p=0.137)'; decomposed CoE variants are still layer-trajectory features. |
| P11-FE1189 | 0.389 | MOOT | The premise that non-linear trajectory energy beats single-layer energy is refuted by the verdict's finding that multi-layer geometric features (CoE/layer profiles) do not add value and no activation-geometry feature generalizes. |
| P11-FE123985A | 0.419 | MOOT | Premised on inter-layer curvature (a trajectory-geometric activation feature) as complementary to DoM; verdict closes all layer-profile and trajectory-based activation-geometry features. |
| P11-FE126 | 0.412 | MOOT | Premise that CoE is useful on long-answer subsets is refuted; verdict shows activation-geometry features do not add value portably. |
| P11-FE162 | 0.443 | MOOT | Identical premise to FE165; EigenScore is a spectral feature of token-cloud covariance, which the verdict found has negative incremental value. |
| P11-FE165 | 0.443 | MOOT | Premise: unsupervised EigenScore (K=10 covariance spectral) beats DoM. Verdict refuted token-cloud covariance spectra as a feature class (0.761 < 0.786, length-confounded). |
| P11-FE189 | 0.429 | MOOT | Tests LID (local intrinsic dimensionality, an activation-geometry feature) as correctness predictor; verdict refuted all activation-geometry features beating free baselines. |
| P11-FE23 | 0.412 | MOOT | Premise that landscape statistics would reveal PH signal is refuted; verdict states experiments premised on PH beating free signals are dead. |
| P11-FE24 | 0.497 | MOOT | Tests whether vectorized PH landscape features beat 0.7731 baseline; verdict explicitly refutes the premise that PH features add value to correctness prediction. |
| P11-FE26 | 0.509 | MOOT | Tests PH on per-class labeled supports; verdict establishes that PH activation-geometry features are dead regardless of whether applied to aggregate or labeled-subset clouds. |
| P11-FE37 | 0.423 | MOOT | Verdict declares 'PH was already null' and closes all PH-based features; testing alternative PH measurements is covered by the blanket closure on activation-geometry features. |
| P11-FE385 | 0.390 | MOOT | The premise that covariance structure carries signal beyond DoM is directly refuted by the verdict's finding that token-cloud covariance spectrum has negative incremental value and no activation-geometry feature works. |
| P11-FE55 | 0.438 | MOOT | Premise: unsupervised PHD matches F-2 in-domain. Verdict explicitly states 'PH was already null (F-7/F-10)' and 'Experiments premised on...PH...are dead'; PHD is a variant of the dead class. |
| P11-FE628 | 0.412 | MOOT | Premise that PH variants (zigzag vs static) might succeed is refuted; verdict establishes PH-based experiments are dead. |
| P11-FE668 | 0.422 | MOOT | Premised on multi-layer AU sets beating single-layer baselines; verdict closes all activation-geometry features (spectra, PH, layer profiles, token clouds) unless in-domain. |
| P11-FE685 | 0.427 | MOOT | Tests whether multi-directional manifold (K PCs) beats single direction; verdict found activation-geometry features don't add value over free baselines. |
| P11-FE694 | 0.405 | MOOT | Verdict explicitly refutes the premise: 'CoE/layer-depth profile's portable depth-grid form does not add (p=0.137).' |
| P11-FE709 | 0.422 | MOOT | Premised on per-layer Mahalanobis-OOD trajectory stack recovering signal from activation geometry; verdict closes layer-profile and trajectory-based activation-geometry features. |
| P11-FE71 | 0.427 | MOOT | Tests whether PCA reveals low-dimensional concept axis; verdict found no activation-geometry feature (including PCA manifold structure) adds value. |
| P11-FE718 | 0.419 | MOOT | Verdict closes F-7 ('PH was already null'); attempting to resurrect F-7 with alternative layer-profile signatures falls under the blanket closure on activation-geometry features. |
| P11-FE802 | 0.391 | MOOT | The premise that complex geometric features (sparse-PCA components) beat the simple DoM direction is directly refuted by the verdict's finding that no activation-geometry feature adds portable correctness signal over free baselines. |
| P11-FE838 | 0.390 | MOOT | The premise that PH might work on attention graphs (even though null on residuals) is refuted by the verdict's finding that PH was already null and that no activation-geometry feature beats free baselines. |
| P11-FE869 | 0.478 | MOOT | Tests whether RGTM topology-flavored statistics exceed Gaussian null; verdict refutes the premise that activation-geometry features (explicitly including topology-based ones) beat free baselines. |
| P11-FE93 | 0.419 | MOOT | Identical to FE97; premised on per-head aggregations beating residual-stream baseline, which verdict closes. |
| P11-FE937 | 0.430 | MOOT | Tests PH on factor-projected sub-clouds; verdict explicitly states 'PH was already null' and mutes experiments premised on activation-geometry features beating free signals. |
| P11-FE968 | 0.419 | MOOT | Premised on band-restricted spectral power-law refining covariance-spectrum features; verdict declares covariance spectrum has negative incremental value. |
| P11-FE97 | 0.419 | MOOT | Premised on per-head aggregations beating full residual-stream L19 DoM baseline; verdict closes activation-geometry features beating free signals. |
| P11-FE991 | 0.392 | MOOT | Verdict refutes premise that spectral features beat free baseline; local covariance spectra are still spectral features (0.7928 global < 0.845 free baseline, so local variants unlikely to escape). |
| P7-FE1 | 0.383 | MOOT | Proposes zigzag PH variant when verdict fully closed geometry work ('Geometry FULLY CLOSED') and explicitly killed PH approaches. |
| P7-FE3 | 0.412 | MOOT | Premise that PHD estimator would change null conclusion is refuted; verdict establishes PH was already null. |
| P8-FE2 | 0.399 | MOOT | Verdict explicitly refuted H-G (early-abort from partial-generation pooling) and states this class is dead unless targeting IN-DOMAIN hidden-dim-bound use; P8-FE2 lacks this claim. |
| P8-FE6 | 0.414 | MOOT | Premise that PH null is an estimator-choice artifact is refuted; verdict establishes PH was already conclusively null. |
| P11-FE1162 | 0.416 | ANSWERED | Verdict established cov-spectrum is length-confounded; testing mechanistic explanation of a confounded feature is now moot. |
| P11-FE193 | 0.410 | ANSWERED | Verdict confirms F-9 (single-layer L19 subsumes multi-layer), directly settling whether multi-layer DoM beats L19 by 1.5 points. |
| P11-FE211 | 0.394 | ANSWERED | Verdict establishes layer-depth profiles don't add portable value over L19 (CoE p=0.137, transfers at chance); tuned-lens multi-layer ensemble falls into this refuted category. |
| P11-FE336 | 0.468 | ANSWERED | Tests whether CoE-C's universality range is tighter than DoM range across models; verdict already measured CoE's cross-domain/cross-scale transfer performance and found it poor, directly answering whether CoE's universality is tight. |
| P11-FE751 | 0.443 | ANSWERED | Tests whether spectral α is length-confounded; verdict already established that token-cloud covariance spectra are length-confounded (r=-0.58, no eigenvalue fix). |
| P10-FE28 | 0.418 | KEEP | Tests orthogonal method comparison (local-dim hypothesis test vs PH); verdict doesn't settle which estimator applies. |
| P10-FE29 | 0.384 | KEEP | Tests dynamic layer selection within geometry approach in-domain on MATH-500; verdict explicitly allows in-domain geometry improvements (CoE survives in-domain). |
| P11-FE01172A | 0.382 | KEEP | Tests feature-selection criterion comparison (SNR vs. PCA top-k); orthogonal to verdict's findings on activation-geometry vs. free-baseline portability. |
| P11-FE01172E | 0.458 | KEEP | Orthogonal theoretical test (NTK eigenspectrum vs signal-channel ceiling); verdict settles generalization but not whether theory explains F-2's AUROC ceiling. |
| P11-FE1037 | 0.389 | KEEP | Testing whether Sharpness Dimension j* differs between correct and incorrect problems is a mechanistic question orthogonal to the verdict's scope. |
| P11-FE1043 | 0.465 | KEEP | Mechanistic SVD explanation for DoM≈PC1 and PH-null is orthogonal to generalization findings. |
| P11-FE1045 | 0.396 | KEEP | Verdict provides last-token DoM AUROC (0.7731) but not mean-pooled representation AUROC; these are different features. |
| P11-FE105 | 0.420 | KEEP | Replicates PMET's FFN/MHSA analysis for mechanistic understanding of where L19 sits in the stability spectrum, not a baseline-beating feature. |
| P11-FE1058 | 0.390 | KEEP | Testing local neighborhood robustness of DoM is orthogonal to the verdict's assessment of out-of-family generalization; it addresses fragility to distribution shift rather than portable signal. |
| P11-FE1062 | 0.403 | KEEP | OT-NNA is a distinct geometric test from PH; verdict tested persistance homology on residuals, not optimal transport nearest-neighbor accuracy. |
| P11-FE1078 | 0.403 | KEEP | Tests inter-layer relation matrices as a geometry feature; while verdict refutes geometry broadly, this specific feature type was not evaluated and conservative approach warrants keeping. |
| P11-FE1085 | 0.385 | KEEP | Tests surface-form sensitivity of L19 activations under Lorem prefixing, orthogonal to whether geometry beats free signals. |
| P11-FE1101 | 0.447 | KEEP | Tests whether spectrum eigenvalues are beyond Marchenko-Pastur edge; verdict refutes spectrum's portability but does not test MP boundary, relevant to distinguishing signal from noise. |
| P11-FE1103 | 0.385 | KEEP | Tests mechanistic property (DoM alignment with MP outliers) that doesn't directly contradict verdict; verdict shows geometry loses but doesn't refute this spectral mechanism. |
| P11-FE110625A | 0.390 | KEEP | The verdict does not directly test cross-domain DoM transfer (TriviaQA→MATH); it only confirms length transfers well cross-architecture and rules out geometric features out-of-family. |
| P11-FE110625B | 0.422 | KEEP | Step-count stratification of DoM AUROC tests a property of the existing signal (complexity dependence), not proposing a new feature to beat free baselines. |
| P11-FE110625C | 0.411 | KEEP | Measuring layer-wise saturation of prefill DoM; verdict does not directly answer whether L19 is a peak or plateau. |
| P11-FE111 | 0.401 | KEEP | Tests whether L19 is correctness-optimal or steering-optimal; verdict does not address ActAdd vector layer sweep or comparison to steering literature. |
| P11-FE112 | 0.390 | KEEP | The verdict shows length is the best out-of-family signal but does not characterize confounding within MATH-500; testing whether DoM's in-domain 0.7731 is confounded by length is orthogonal to the verdict's scope. |
| P11-FE1127 | 0.381 | KEEP | Tests linear mode connectivity between DoM and PC1; verdict does not settle whether interpolations exceed single-direction AUROC. |
| P11-FE1128 | 0.463 | KEEP | Pareto hypervolume measurement; while spectral is negative, the measurement itself could characterize dominated regions. |
| P11-FE1141 | 0.391 | KEEP | Diagnostic analysis of what drives cov-spectrum AUROC via reweighting; tests composition of existing result, not proposing new feature competing with free baseline. |
| P11-FE1147 | 0.448 | KEEP | Tests theoretical eigenvalue-decay shape (exponential-kernel fit); verdict shows spectrum is confounded but does not test goodness-of-fit to MP or Fourier-kernel predictions. |
| P11-FE1149 | 0.383 | KEEP | Tests intrinsic dimensionality of correctness manifold; independent theoretical question not addressed by verdict. |
| P11-FE1163 | 0.416 | KEEP | Tests technical refinement of DoM extraction via energy-weighting; verdict doesn't directly address this variant. |
| P11-FE1167 | 0.391 | KEEP | Tests nonlinear MLP readout on L19; verdict refutes activation-geometry spectral/profile features but does not specifically test whether nonlinear decoding of L19 activations beats free baseline. |
| P11-FE1169 | 0.382 | KEEP | Tests information-theoretic explanation (per-layer SNR bounds on capacity); orthogonal to verdict's architectural findings. |
| P11-FE117 | 0.409 | KEEP | F-1 dimensional breathing is orthogonal to verdict's focus on activation-geometry correctness signals. |
| P11-FE1173 | 0.388 | KEEP | Testing whether Shannon capacity law explains the L19 peak is a mechanistic/theoretical question not directly addressed by the verdict's empirical findings on generalization. |
| P11-FE1184 | 0.418 | KEEP | Tests cross-paper SAE methodology port, not directly addressed by verdict about generalization. |
| P11-FE1185 | 0.452 | KEEP | Tests layer-sensitivity landscape within architecture; verdict establishes L19 DoM works but does not test whether other layers or multi-layer combination improve in-domain AUROC. |
| P11-FE1195 | 0.410 | KEEP | Inter-layer convergence ratio as label-free predictor is not directly addressed; verdict refutes H-G (early-abort) which is related but distinct. |
| P11-FE1196 | 0.409 | KEEP | Layer-wise convergence profile is related to but distinct from H-G refutation (early-abort); verdict does not directly address convergence speed splits. |
| P11-FE120476C | 0.384 | KEEP | Compares graph-theoretic and covariance geometric framings within-domain as theoretical unification, not testing portability. |
| P11-FE1206 | 0.392 | KEEP | Tests whether PH-null depends on coordinate system (SA-GSAE basis); verdict confirms null on raw residuals but does not address whether coordinate transform reveals topology. |
| P11-FE1207 | 0.393 | KEEP | Diagnostic decomposition of DoM into stiff/flat modes (mechanism analysis), not proposing a new feature that must beat free baseline. |
| P11-FE1208 | 0.400 | KEEP | Tests spectral concentration structure within confirmed cov-spectrum signal (0.7928); orthogonal granularity question about confirmed feature. |
| P11-FE1209 | 0.500 | KEEP | Tests whether L19 spectral gap is an outlier; verdict measured spectrum as a predictive feature (not gap magnitude) and found it negative-valued, leaving open whether gap ratios are special. |
| P11-FE1210 | 0.442 | KEEP | Tests decision boundary dimensionality (rank-1 vs multi-hyperplane); the verdict doesn't address this structural question, only cross-domain generalization failure. |
| P11-FE1215 | 0.410 | KEEP | Spectral radius and dynamical-systems interpretation are orthogonal to the verdict's focus on activation-geometry portability. |
| P11-FE1221 | 0.390 | KEEP | This mechanistic question about layer-wise ε emergence is orthogonal to the verdict's finding about generalization of geometric features; it does not test any premise the verdict refutes. |
| P11-FE1222 | 0.384 | KEEP | Tests whether L19 capacity constrains AUROC ceiling; orthogonal mechanistic question about why geometry loses, not directly settled by verdict. |
| P11-FE123985C | 0.397 | KEEP | Verdict does not test layer-wise curvature profiles or alternative mechanistic interpretations of F-4. |
| P11-FE123985E | 0.412 | KEEP | Testing curvature extraction from different paper (2604.23985); verdict does not address curvature-based approaches. |
| P11-FE132 | 0.411 | KEEP | Testing diagonal-extraction probe on cached activations; verdict does not address this alternative layer-trajectory feature. |
| P11-FE135 | 0.396 | KEEP | Verdict does not test implicit CoT as a control condition for breathing across training paradigms. |
| P11-FE148 | 0.383 | KEEP | Tests inter-layer direction stability and convergence; mechanistic analysis orthogonal to portability verdict. |
| P11-FE163 | 0.445 | KEEP | Tests layer optimality for EigenScore; verdict refutes that spectral features generalize, but doesn't settle layer-specific structural properties. |
| P11-FE166 | 0.446 | KEEP | Tests per-layer EigenScore optimum vs DoM optimum; verdict does not evaluate EigenScore metric or test whether layer-best for EigenScore matches layer-best for DoM. |
| P11-FE177 | 0.421 | KEEP | Tests mechanistic hypothesis about register-token information localization, not proposing a token-position feature to beat the length+logprob baseline. |
| P11-FE178 | 0.417 | KEEP | Tests depth-confounding of F-2; verdict doesn't address stratified analysis by solution depth. |
| P11-FE190 | 0.461 | KEEP | LID-GeoMLE on final-token is a specific method; verdict doesn't test this particular approach. |
| P11-FE195 | 0.383 | KEEP | Tests layer emergence via L0 sign-flip mechanism; orthogonal mechanistic question not addressed by verdict. |
| P11-FE197 | 0.433 | KEEP | Tests whether non-linear probes exceed the linear DoM ceiling, a methodological question orthogonal to the verdict's generalization findings. |
| P11-FE208 | 0.449 | KEEP | Tests whether L19 is locally optimal or artifact; verdict confirms L19 DoM but does not test per-layer AUROC landscape or whether Paulo et al.'s middle-layer universality holds here. |
| P11-FE212 | 0.396 | KEEP | Verdict confirms L19 prefill AUROC 0.7731 but does not test per-layer sweep within Qwen to verify layer optimality. |
| P11-FE213 | 0.407 | KEEP | Multi-layer probe ensemble is not tested specifically; verdict confirms CoE redundancy with L19 but ensemble may behave differently. |
| P11-FE216 | 0.401 | KEEP | Tests whether continuous logit-difference target reveals hidden signal in L19 beyond binary AUROC 0.7731; measurement methodology check on confirmed signal. |
| P11-FE217 | 0.391 | KEEP | Tests in-domain causal mechanism (CoE-L19 redundancy vs OR-gate); verdict permits in-domain hidden-dim-bound use and does not directly refute in-domain causal interaction hypothesis. |
| P11-FE236 | 0.415 | KEEP | Tests whether F-9 redundancy is length-confounded via VTS/HTS stratification; verdict doesn't stratify by generation length. |
| P11-FE238 | 0.482 | KEEP | Tests GRIDE intrinsic dimension per layer and its predictive value; verdict does not address ID-based analysis or whether L19 sits in a depth-ID peak. |
| P11-FE240 | 0.459 | KEEP | Verdict refutes portable CoE signal (H-C) but explicitly confirms in-domain CoE survives (+1.75pp); this tests whether in-domain signal has layer-region structure, which verdict does not settle. |
| P11-FE245 | 0.381 | KEEP | Tests whether F-10's PH-null result is a basis-choice artifact; verdict established PH is null, but this tests the mechanism and robustness of that null. |
| P11-FE267 | 0.382 | KEEP | Tests SAE features vs DoM within-domain; verdict forbids geometry beating free signals, not in-domain geometric feature comparisons. |
| P11-FE268 | 0.384 | KEEP | Tests whether optimal (layer, position) exists beyond L19 prefill; verdict tests L19 but doesn't settle whether other locations yield higher AUROC. |
| P11-FE27 | 0.431 | KEEP | Tests the topological-expressivity phase-transition ceiling prediction, a theoretical question about in-domain probe expressivity not addressed by the verdict. |
| P11-FE284 | 0.384 | KEEP | Tests theoretical mechanism (prediction/suppression neuron bases) for F-3's orthogonality, orthogonal to verdict's portability claims. |
| P11-FE285 | 0.381 | KEEP | Tests whether F-1 breathing universality is explained by CKA block structure; orthogonal to verdict's activation-geometry vs. free-baseline comparison. |
| P11-FE288 | 0.444 | KEEP | Tests whether L19 is structurally special; orthogonal to whether DoM features generalize across architectures. |
| P11-FE290 | 0.407 | KEEP | Trace-of-covariance metric is not tested or addressed by the verdict. |
| P11-FE297 | 0.393 | KEEP | Tests whether PH-null is substrate-universal vs LLM-specific (mechanism refinement); verdict confirms PH null on LLMs but does not address whether nullness transfers to Tracr's symbolic substrate. |
| P11-FE306 | 0.429 | KEEP | Tests probe position (where signal peaks), orthogonal to verdict's finding that activation-geometry features don't generalize; probe location is separate from feature utility. |
| P11-FE307 | 0.493 | KEEP | Tests whether 2-layer MLP probe on prefill activations beats linear baseline; verdict does not address probe architecture comparisons, only whether activation-geometry features beat free signals. |
| P11-FE310 | 0.433 | KEEP | Tests whether prefill-final orthogonality (F-3) is specific to the K=1 correctness target, a mechanistic question not addressed by the verdict. |
| P11-FE315 | 0.452 | KEEP | Tests within-domain multi-layer refinement not beating free baseline; verdict confirms activation-geometry doesn't generalize but doesn't settle multi-layer vs single-layer within-domain improvement. |
| P11-FE325 | 0.415 | KEEP | Tests whether per-cluster DoM heterogeneity explains shrinkage in global cos(prefill_DoM, final_DoM) = 0.046; verdict doesn't address. |
| P11-FE334 | 0.400 | KEEP | Verdict tested free baseline cross-architecture but not whether L19 is optimal layer within each model (Phi-3, Gemma-2). |
| P11-FE338 | 0.468 | KEEP | Verdict shows CoE transfers poorly but doesn't directly answer whether CoE's universality range across models is tighter than DoM's. |
| P11-FE34 | 0.393 | KEEP | Tests weight-space PH topology; verdict refutes activation-space PH but does not directly test whether weight matrices show distinctive persistence separating trained from null. |
| P11-FE340 | 0.382 | KEEP | Tests pooling robustness of F-3 (mean-pooled vs last-token); verdict does not address pooling artifacts. |
| P11-FE347 | 0.392 | KEEP | Tests DoM rotation curve mechanism across generation (prefill→final angle drift); verdict does not directly address whether rotation pattern is monotonic vs abrupt. |
| P11-FE352 | 0.407 | KEEP | F-1 universality across scales is orthogonal to verdict's findings on activation-geometry correctness signals. |
| P11-FE355 | 0.383 | KEEP | Tests whether DoM carries information orthogonal to token-prob within-domain; verdict forbids portability, not in-domain orthogonality analysis. |
| P11-FE36 | 0.465 | KEEP | Tests PH robustness under Rieck filtration; verdict confirms standard PH is null but doesn't test alternative filtrations. |
| P11-FE365 | 0.515 | KEEP | Tests token-invariance of DoM across layers; verdict does not address whether orthogonality at L19 persists or is layer-local. |
| P11-FE38 | 0.452 | KEEP | Tests topological equivalence (persistence diagrams) between prefill/final despite coordinate orthogonality; not directly addressed by verdict's generalization findings. |
| P11-FE381 | 0.444 | KEEP | Tests layer-choice robustness and domain contamination of F-2; the verdict doesn't refute F-2's in-domain existence, only its generalization. |
| P11-FE383 | 0.464 | KEEP | In-domain SLA-window probe; verdict states multi-layer methods survive in-domain, so KEEP. |
| P11-FE384 | 0.417 | KEEP | Tests whether L27 ≥ L19 under VISTA's framing; verdict doesn't address cross-layer comparisons. |
| P11-FE387 | 0.417 | KEEP | Tests Mahalanobis (second-order) layer specificity; verdict doesn't address higher-order signal distribution. |
| P11-FE389 | 0.445 | KEEP | Tests whether F-2 is a learnable architectural property vs probe ceiling; the verdict doesn't address whether the underlying model capacity exists. |
| P11-FE390 | 0.460 | KEEP | Self-Sim baseline measurement is independent baseline; verdict doesn't test label-free mean-cosine approaches. |
| P11-FE391 | 0.392 | KEEP | Quantifies disentanglement metrics on existing cached data (diagnostic analysis); not proposing new readouts that must beat free baseline out-of-family. |
| P11-FE399 | 0.436 | KEEP | Tests confidence-probe informativeness for correctness, distinct from H-F's refutation of probe-based sample reranking for selection. |
| P11-FE415 | 0.447 | KEEP | Tests whether non-linear probe beats linear DoM in-domain; verdict does not evaluate MLP-based refinement, only confirms linear L19 DoM baseline. |
| P11-FE419 | 0.421 | KEEP | Tests layerwise signal emergence via logit-lens to understand depth localization, not proposing a layer-profile feature to beat baselines. |
| P11-FE420 | 0.421 | KEEP | Tests whether L19 is specifically powerful or if signal emerges across a grounding plateau—mechanistic refinement of F-2, not proposing a layer-profile feature. |
| P11-FE427 | 0.432 | KEEP | Tests whether singular-token density in prompts confounds the prefill-DoM signal, a mechanistic isolation question not addressed by the verdict. |
| P11-FE429 | 0.384 | KEEP | Duplicate of FE435; tests substrate dependence of F-2 on reasoning models, not directly settled by verdict. |
| P11-FE432 | 0.428 | KEEP | Tests layer selection on reasoning model (duplicate of FE438); orthogonal to verdict about activation-geometry feature utility. |
| P11-FE435 | 0.384 | KEEP | Tests whether F-2's 0.7731 is substrate-limited on reasoning models; verdict doesn't test R1-Distill variant and this is orthogonal to free-signal comparison. |
| P11-FE438 | 0.428 | KEEP | Tests whether L19 optimal-layer claim holds in long-CoT-trained model; layer selection is orthogonal to verdict about activation-geometry features. |
| P11-FE45 | 0.443 | KEEP | Tests prompt-sensitivity via SMS, which is a different angle on independence than the verdict's cross-domain generalization claim. |
| P11-FE454 | 0.417 | KEEP | Tests verbalized-confidence probe as alternative modality; verdict doesn't address this feature family. |
| P11-FE461 | 0.411 | KEEP | Testing whether knowledge distillation shifts DoM layer peaks; verdict does not address KD effects on layer canonicality. |
| P11-FE463 | 0.390 | KEEP | Testing whether singular-asymmetry predicts breathing amplitude is a mechanistic question orthogonal to the verdict's scope on generalization of geometric readouts. |
| P11-FE467 | 0.389 | KEEP | Testing cross-benchmark DoM parallelism (MATH/GSM8K/AIME) is a distinct universality test not directly settled by the verdict's findings. |
| P11-FE476 | 0.442 | KEEP | Tests which layer has best geometric separation for D/A-bucket; the verdict refutes that PH generalizes, but doesn't settle layer-dependent geometric structure. |
| P11-FE482 | 0.409 | KEEP | CCPS final-token method is not tested by verdict; in-domain AUROC comparison is not directly answered. |
| P11-FE485 | 0.409 | KEEP | Identical to FE482; CCPS final-token method is not tested by verdict. |
| P11-FE491 | 0.449 | KEEP | Tests sparse-feature refinement within activation-geometry; verdict carves out in-domain use as permissible, and this targets SAE-based within-domain signal, not beating free baseline out-of-family. |
| P11-FE492 | 0.400 | KEEP | Verdict tested cross-scale/cross-domain for free baselines, not OOD robustness gap for DoM probe specifically. |
| P11-FE494 | 0.412 | KEEP | Testing LRT cross-scale affine transfer of DoM; verdict shows free baselines transfer well (0.865) but does not directly measure DoM cross-scale AUROC. |
| P11-FE504 | 0.381 | KEEP | Tests H-17 token-position trajectory behavior (redundancy vs. DoM oscillation); verdict does not settle the correctness-axis-specific scope of H-17. |
| P11-FE507 | 0.453 | KEEP | Tests label-free dispersion as readout; verdict does not evaluate this specific scalar metric, only establishes free baseline (length+logprob) and L19 DoM performance. |
| P11-FE51 | 0.431 | KEEP | Verdict states PH was null but does not specify whether giotto-tda Heat/Silhouette vectorizations were tested; the methodological gap remains uncertain. |
| P11-FE513 | 0.410 | KEEP | Rank trajectory analysis is not addressed by the verdict; mechanism question orthogonal to portability conclusions. |
| P11-FE514 | 0.425 | KEEP | Tests non-linear probe method vs linear; probe method choice is orthogonal to verdict about whether activation-geometry features add value. |
| P11-FE521 | 0.437 | KEEP | Tests local algebraic redundancy of CoE with DoM, not whether CoE generalizes out-of-family (which the verdict already refuted). |
| P11-FE532 | 0.426 | KEEP | Tests per-head attention distribution (structural property, not activation-geometry feature); orthogonal to verdict about geometric features. |
| P11-FE547 | 0.401 | KEEP | Tests whether last-layer L27 matches L19 at 0.7731; verdict confirms L19 layer-specificity but does not address final-layer performance. |
| P11-FE550 | 0.401 | KEEP | Tests whether last-layer L27 matches L19 at 0.7731; verdict confirms L19 layer-specificity but does not address final-layer performance. |
| P11-FE552 | 0.431 | KEEP | Tests whether L19 is a sharp layer-specific peak, not addressed by the verdict's focus on portable generalization across models. |
| P11-FE56 | 0.471 | KEEP | Tests correlational relationship between PHD and breathing across architectures; verdict does not address this correlation-discovery task, which is orthogonal to baseline-beating claims. |
| P11-FE563 | 0.397 | KEEP | Verdict reports 50% coverage point (71.6%) but not AURC (full curve metric). |
| P11-FE567 | 0.466 | KEEP | Same as FE571; tests local dimensionality robustness of F-10's null with a different method. |
| P11-FE571 | 0.466 | KEEP | Testing a different dimensionality estimator (γ_both^local) for F-10's PH-null robustness, not directly addressed by verdict. |
| P11-FE576 | 0.482 | KEEP | Tests whether TwoNN-ID per-layer trajectory carries signal; verdict tested CoE but not TwoNN-ID specifically, and different per-layer feature families may have different outcomes. |
| P11-FE577 | 0.410 | KEEP | Verdict confirms multi-layer approaches don't generalize, but CLUE specifically is not tested; in-domain performance is not directly answered. |
| P11-FE584 | 0.438 | KEEP | PID feature direction is not among the refuted activation-geometry families (spectra, PH, layer profiles, token clouds) and tests an in-domain comparison against F-2. |
| P11-FE585 | 0.385 | KEEP | Tests whether per-layer PID method bridges token-position rotation; verdict doesn't test PID itself and this is mechanistic, not about beating free signals. |
| P11-FE589 | 0.467 | KEEP | Per-layer DoM AUROC profile is orthogonal to the generalization-edge verdict focused on cross-model transfer. |
| P11-FE593 | 0.435 | KEEP | In-domain test of Latent-Trajectory signals; the verdict's refutation of activation-geometry generalization does not settle in-domain signal strength. |
| P11-FE600 | 0.460 | KEEP | Tests mechanistic EV-k dimensionality relationship and CoE-60 equivalence; verdict doesn't address this relationship directly. |
| P11-FE602 | 0.539 | KEEP | Tests whether multi-layer Gaussian depth schedule beats single-layer L19; verdict establishes that activation-geometry features don't beat free baselines but doesn't directly address multi-layer vs single-layer probe architecture comparisons. |
| P11-FE603 | 0.414 | KEEP | Measuring DoM geometric structure across layers—whether cosine profile is continuous or clustered—orthogonal to whether DoM beats free baselines. |
| P11-FE606 | 0.418 | KEEP | Tests F-1 breathing measurement via independent intrinsic-dim estimator; orthogonal to verdict about generalization. |
| P11-FE607 | 0.384 | KEEP | Tests calibration (ECE, Brier) of DoM scores, orthogonal to verdict's claim about geometry vs free baselines. |
| P11-FE612 | 0.415 | KEEP | Tests whether optimal DoM layer is absolute or fractional-depth; verdict doesn't address layer-scaling question. |
| P11-FE616 | 0.406 | KEEP | Tests methodological robustness of F-10's PH null to choice of PH instrument; verdict does not address whether F-10's null result holds across different PH tools. |
| P11-FE617 | 0.428 | KEEP | Tests calibration metrics (ECE, BSS) of existing DoM across layers, orthogonal to verdict about feature utility. |
| P11-FE619 | 0.465 | KEEP | Tests whether encoder size affects redundancy findings; verdict doesn't evaluate cross-encoder effects. |
| P11-FE623 | 0.395 | KEEP | Verdict mentions free baseline includes logprob but does not provide in-distribution logprob AUROC in isolation for comparison. |
| P11-FE626 | 0.406 | KEEP | Tests whether attention-graph PH can match residual-stream DoM signal; verdict does not evaluate attention-geometry substrate, only residual-space features. |
| P11-FE627 | 0.405 | KEEP | Tests whether attention-graph zigzag PH distinguishes from Gaussian null; extends F-10's null to attention substrate, which verdict does not address. |
| P11-FE633 | 0.455 | KEEP | Tests within-domain refinement (multi-cluster vs single-direction L19) not generalization; verdict confirms L19 DoM but doesn't settle whether multi-cluster improves in-domain. |
| P11-FE634 | 0.455 | KEEP | Tests layer sensitivity within architecture; verdict confirms L19 DoM beat baseline but does not test whether other layers might beat L19 more. |
| P11-FE645 | 0.468 | KEEP | Tests whether ridge-tuned RAPTOR probe outperforms linear DoM baseline; verdict does not address probe architecture or regularization improvements on top of activations. |
| P11-FE649 | 0.409 | KEEP | Infrastructure task; verdict does not settle whether this re-extraction is necessary or helpful. |
| P11-FE665 | 0.437 | KEEP | AU-level sparsity is not addressed by the verdict, which tested full token-cloud spectra; this is an orthogonal refinement. |
| P11-FE683 | 0.393 | KEEP | Diagnostic validation of trajectory geometry structure (ρ_exec/ρ_rand control); not proposing a feature-engineering approach that requires beating free baseline. |
| P11-FE687 | 0.382 | KEEP | Tests multi-layer ensemble localization within-domain (TADA-motivated); verdict allows in-domain optimization of geometry features. |
| P11-FE692 | 0.382 | KEEP | Tests multi-layer ensemble within-domain (in-domain hidden-dim-bound use); verdict allows such optimization beyond single-layer. |
| P11-FE693 | 0.390 | KEEP | Testing whether orthogonality (cos=0.046) persists under logit-lens projection is orthogonal to the verdict's assessment of portable correctness signals. |
| P11-FE696 | 0.386 | KEEP | Tests coordinate-invariance of F-2 and F-3 measurements, orthogonal to whether the verdict's geometry-vs-free-baseline comparison holds across coordinate systems. |
| P11-FE698 | 0.421 | KEEP | Tests mechanistic hypothesis linking prefill DoM to Xu's parameter-space backbone; not proposing a new activation-geometry feature. |
| P11-FE702 | 0.433 | KEEP | Tests structural basis for F-2's L19 selection via layer-interaction patterns, not about generalization. |
| P11-FE703 | 0.398 | KEEP | Verdict does not test learned cross-layer routing modules, only free baselines and static geometry. |
| P11-FE705 | 0.428 | KEEP | In-domain LoRA-gated verifier targeting hidden-dim-bound use on MATH-500 (excepted by verdict); not comparing activation-geometry features to free out-of-family baseline. |
| P11-FE707 | 0.417 | KEEP | Tests whether non-linear MLP uncovers hidden manifold signal in-domain; orthogonal to verdict about cross-domain generalization. |
| P11-FE720 | 0.439 | KEEP | Tests confound robustness of F-2 via ridge-residualization; the verdict doesn't address whether DoM is confounded with capability/topic axes. |
| P11-FE723 | 0.386 | KEEP | Tests DoM transfer within Qwen scales, which the verdict doesn't directly test; verdict confirms geometry loses to free signals but doesn't settle whether within-family transfer is feasible. |
| P11-FE724 | 0.491 | KEEP | Tests 5-layer MLP probe against linear baseline; verdict focuses on feature-level contribution, not probe architecture capacity or information-extraction efficiency. |
| P11-FE725 | 0.430 | KEEP | Tests whether attention-probe aggregation method lifts signal, not whether activation-geometry features add value; orthogonal to verdict. |
| P11-FE726 | 0.385 | KEEP | Tests in-domain robustness of a geometry-based probe via KL-regularization; verdict explicitly allows in-domain geometry improvements (CoE survives in-domain). |
| P11-FE727 | 0.429 | KEEP | Tests ridge-probe calibration method on same cached data, not whether activation-geometry features add value. |
| P11-FE736 | 0.430 | KEEP | Tests whether step-organized subspaces are universal across architectures (F-1), orthogonal to verdict's conclusions about activation-geometry features beating free baselines. |
| P11-FE753 | 0.411 | KEEP | Testing Layerwise CP layer-aggregation method; verdict does not engage with this alternative information decomposition. |
| P11-FE77 | 0.393 | KEEP | Prediction-depth is a trajectory prediction-space feature orthogonal to covariance/layer-profile geometry; verdict does not directly test whether prediction-depth predicts correctness. |
| P11-FE770 | 0.383 | KEEP | Tests alternative explanation (margin-tail curvature) for existing findings; verdict does not settle this mechanism question. |
| P11-FE777 | 0.462 | KEEP | Non-Transformer architecture test is orthogonal; verdict addresses dense Transformers only. |
| P11-FE790 | 0.414 | KEEP | Testing what information DoM depends on via CoT masking; verdict does not directly measure whether this ablation changes DoM AUROC. |
| P11-FE797 | 0.414 | KEEP | Identical to FE790—testing CoT dependence of DoM; verdict does not address this information-source question. |
| P11-FE805 | 0.381 | KEEP | Targets in-domain hidden-dim-bound use (L18→L19 circuit analysis on MATH-500), which verdict explicitly carved as a non-dead exception. |
| P11-FE807 | 0.462 | KEEP | In-domain SAID test on Qwen-1.5B; verdict confirms multi-layer adds in-domain (+1.75pp for CoE), so SAID could also contribute. |
| P11-FE827 | 0.402 | KEEP | Tests whether signal concentrates at L19 or spreads across layers; verdict confirms L19 at 0.7731 but does not evaluate multi-layer performance. |
| P11-FE829 | 0.410 | KEEP | Verdict refutes H-C and H-E but does not explicitly address whether probe accuracy aligns with steering success. |
| P11-FE832 | 0.436 | KEEP | Tests whether L19 is a layer-specific peak vs plateau, orthogonal to the verdict's focus on generalization and free-baseline comparison. |
| P11-FE841 | 0.384 | KEEP | Adds calibration metrics (Brier, ECE) to F-2's existing AUROC; orthogonal to whether geometry beats free signals. |
| P11-FE843 | 0.383 | KEEP | Tests INLP-discoverable directions and dimensionality of orthogonality; verdict does not forbid within-domain mechanistic analysis. |
| P11-FE849 | 0.414 | KEEP | Tests multi-target joint probe (correctness + agreement + CoT-length) in-domain; verdict about cross-domain generalization. |
| P11-FE851 | 0.398 | KEEP | Verdict does not test external model (SpecExit) steering vector alignment. |
| P11-FE852 | 0.424 | KEEP | Re-narrating F-1's breathing trajectory with alternative metrics (eDim vs PR) is a measurement question about an existing phenomenon, not testing whether a new feature beats free baselines. |
| P11-FE859 | 0.426 | KEEP | Tests probe layer/position selection, orthogonal to verdict's conclusions about activation-geometry feature utility. |
| P11-FE875 | 0.387 | KEEP | Tests whether L19 satisfies ETF representation structure, orthogonal to whether activation geometry predicts correctness better than free signals. |
| P11-FE884 | 0.432 | KEEP | Tests whether Wasserstein-1 distance captures geometric separability differently than AUROC, a metric-choice question orthogonal to signal existence. |
| P11-FE893 | 0.403 | KEEP | Post-hoc confound analysis of a confirmed signal (cov-spectrum 0.7928); verdict does not address whether the spectral advantage stems from GRPO dependency or task structure. |
| P11-FE897 | 0.381 | KEEP | Tests DoM symmetry-invariance under one-parameter subgroups; orthogonal to verdict's signal-value findings. |
| P11-FE898 | 0.416 | KEEP | Tests whether orbit-separating representation captures in-domain non-linearity; orthogonal to generalization verdict. |
| P11-FE90 | 0.393 | KEEP | Token-attribution analysis of DoM structure (interpretation/mechanism of validated signal); does not propose new feature competing with free baseline. |
| P11-FE908 | 0.398 | KEEP | Verdict confirms PH is null (F-10) but Hodge Laplacian spectrum is a different topological invariant not tested. |
| P11-FE939 | 0.383 | KEEP | Tests weight matrix spectral properties and perturbation stability of DoM direction, orthogonal to portability verdict. |
| P11-FE941 | 0.387 | KEEP | Testing whether L19 has an anomalous spectral profile is a mechanistic question orthogonal to the verdict's assessment of whether geometric features beat free baselines. |
| P11-FE942 | 0.399 | KEEP | Verdict tested cross-architecture (all AdamW-trained); SAM vs AdamW comparison on same model is orthogonal. |
| P11-FE948 | 0.413 | KEEP | Testing weight-space origins of DoM via attention-out SVD; orthogonal to whether activation-geometry features beat free baselines. |
| P11-FE953 | 0.397 | KEEP | Verdict tested cross-architecture (all AdamW); SAM vs AdamW optimizer comparison is not addressed. |
| P11-FE958 | 0.397 | KEEP | Verdict tested cross-architecture (all AdamW); SAM vs AdamW optimizer comparison is not addressed. |
| P11-FE960 | 0.409 | KEEP | Stable rank and FFN sparsity relationship not addressed by verdict. |
| P11-FE961 | 0.485 | KEEP | Tests mechanistic explanation of spectrum via FFN sparsity rank model; verdict shows spectrum lacks predictive value but does not refute the mechanistic claim itself. |
| P11-FE966 | 0.537 | KEEP | Tests whether L19 is spectrally special or a generic depth effect; verdict did not measure per-layer α_tail divergence or test this specific claim about L19's specialness. |
| P11-FE976 | 0.384 | KEEP | Tests whether L19 is geometrically special (weight continuity peak); mechanistic property orthogonal to verdict's conclusion that geometry doesn't beat free signals. |
| P11-FE978 | 0.446 | KEEP | Tests rank structure of activations, which is orthogonal to whether that structure generalizes as a feature—the verdict refutes generalization, not structural existence. |
| P11-FE980 | 0.384 | KEEP | Duplicate of FE976; tests layer-wise weight continuity structure, orthogonal to geometry-vs-free-signals comparison. |
| P11-FE982 | 0.446 | KEEP | Identical to FE978; tests intrinsic dimensionality, not predictive power. |
| P8-FE10 | 0.462 | KEEP | Local PLH methodology differs from global PH tested in F-10; verdict doesn't directly address local vs. global. |
| P8-FE3 | 0.401 | KEEP | Tests whether simple dispersion can replace D2H-lite's 58-dim features; orthogonal efficiency question about existing features, not about geometry vs baselines. |
| P8-FE8 | 0.430 | KEEP | Tests whether F-10's PH-null verdict survives diameter-normalized persistence, a methodological robustness question; the verdict does not specify which summaries were ruled out. |
| P8-FE9 | 0.454 | KEEP | adjudicator returned no/invalid verdict — kept |
| P9-FE5 | 0.382 | KEEP | Identical to P9-FE6; in-domain CoE testing not directly refuted by verdict's portable-setting closure. |
| P9-FE6 | 0.382 | KEEP | Verdict carved out exception for in-domain hidden-dim-bound use (raw CoE +1.75pp survives in-domain), and this experiment targets MATH-500 in-domain testing. |

Review: wrongly-closed FEs can be revived with
`python update_status.py <id> READY`.
