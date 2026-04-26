# Geometry of reasoning in language models: a literature map for an 11-pathway research program

This map organizes ~250 papers around the project's empirical findings across pathways 7–11. **The project's three most novel contributions relative to the 2023–2026 literature are: (1) the prefill-vs-final-token *orthogonality* finding (cos = -0.06), (2) *asymmetric dimensional collapse* tied to correctness with magnitude scaling in model size, and (3) the explicit demonstration of *direction rotation* across generation positions that breaks fixed-vector steering.** The strongest practical result — selective prediction lifting a 48.6% baseline to 71.6% at 50% coverage on MATH-500 — sits squarely in a converging consensus that simple linear DoM probes on prefill activations are competitive with far more elaborate methods.

The map is organized into the three categories the user requested: (A) *Past* — papers that explain or validate the project's findings; (B) *Present* — parallel work reaching similar or divergent conclusions; (C) *Future* — papers whose experiments could extend the work.

---

## Category A — Past: Explanations and theoretical grounding

### A.1 Why persistent homology failed (NO-GO at 0.774, Gaussian-null indistinguishable, layer-wise PH-168 at 0.646)

The 2024–2026 TDA literature has converged on the explanation that **standard Vietoris–Rips PH on high-dimensional point clouds measures dispersion/covariance rather than topology**, exactly the project's empirical observation.

- **Damrich et al., "Persistent Homology for High-dimensional Data Based on Spectral Methods"** (NeurIPS 2024, arXiv:2311.03087, repo: berenslab/persistent-homology-spectral). Proves Rips PH fails to detect even a noisy circle's topology once ambient dim exceeds ~30; the Qwen2.5-1.5B residual stream at d=1536 lives well inside this regime.
- **Hiraoka, Imoto, Lacombe, Shirai, Suzaki, "Curse of Dimensionality on Persistence Diagrams"** (arXiv:2404.18194, 2024). Companion theorem: Hausdorff distance between observed and noise persistence diagrams is unbounded in probability under high-dim noise.
- **Bobrowski & Skraba, "A Universal Null-distribution for Topological Data Analysis"** (Sci. Reports 13:12274, 2023; arXiv:2207.03926). Establishes that "noisy" features in random-cloud persistence diagrams obey a universal log-logistic limit — supplying the canonical null the project's experiment 3 implicitly uses.
- **Turkeš, Montúfar, Otter, "On the Effectiveness of Persistent Homology"** (arXiv:2206.10551, NeurIPS 2022). Shows PH detects holes only on clean low-dim clouds; performance collapses with noise — **directly explains** why layer-wise PH-168 only reached 0.646.
- **"How much of persistent homology is topology? A quantitative decomposition for spin model phase transitions"** (arXiv:2603.29072, 2026). Introduces **density-matched shuffled null** (methodologically identical to the project's token-shuffle test) and finds 94–100% of standard H₀ statistics are density-driven. **This is the strongest external validation of the project's Gaussian-null finding.**
- **Naitzat, Zhitnikov, Lim, "Topology of Deep Neural Networks"** (JMLR 21:184, 2020; arXiv:2004.06093). Trained ReLU networks drive class-manifold Betti numbers toward 0/1 — a successful generative LM has *already* simplified its activation topology, leaving little for PH to find.

### A.2 Why the correctness signal concentrates in late layers (signal at L19/28 ≈ 68% depth)

- **Tenney, Das, Pavlick, "BERT Rediscovers the Classical NLP Pipeline"** (ACL 2019, arXiv:1905.05950). Foundational: low-level features early, high-level semantic features (truth, factuality) in upper-middle layers.
- **Marks & Tegmark, "The Geometry of Truth"** (arXiv:2310.06824, repo: saprmarks/geometry-of-truth). Truth direction emerges in mid-layers (~8–14 of 32–40); mass-mean (DoM) probes generalize as well as logistic regression — **the methodological precedent for finding 4 (DoM matches engineered features)**.
- **Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens"** (arXiv:2303.08112, repo: AlignmentResearch/tuned-lens). Shows refinement concentrates in late layers; latent trajectories carry diagnostic signal.
- **Lad, Gurnee, Tegmark, "The Remarkable Robustness of LLMs: Stages of Inference?"** (arXiv:2406.19384). Posits 4 stages: detokenization → feature engineering → prediction ensembling → residual sharpening. **Directly explains the temporal AUROC cliff** (prefill detokenization → early-decoding feature engineering hand-off).
- **Csordás et al., "Do Language Models Use Their Depth Efficiently?"** (ICLR 2025). Later layers refine output distribution rather than perform new computation — explains why a probe at L19 saturates near final-layer accuracy.
- **"Calibration Across Layers"** (arXiv:2511.00280, 2025). Accuracy plateaus at a sudden mid-layer peak; calibration adjusts later without changing the answer.
- **"Detecting LLM Hallucination Through Layer-wise Information Deficiency"** (arXiv:2412.10246, EMNLP 2025). V-usable information across layers concentrates the hallucination signal in middle-to-late layers.

### A.3 Why dimensional breathing occurs during reasoning (PR ~20 → ~67/88 → ~8/6)

The participation-ratio expand-then-compress signature has clear neuroscience and ML antecedents — but **the project's contribution is moving the analysis from *layer depth* to *generation time* and tying compression magnitude to correctness**.

- **Gao et al., "A theory of multineuronal dimensionality, dynamics and measurement"** (bioRxiv 2017). Defines PR = (Σλᵢ)²/Σλᵢ² as upper-bounded by Neural Task Complexity — methodological foundation.
- **Stringer et al., "High-dimensional geometry of population responses in visual cortex"** (Nature 571, 2019). 1/n eigenspectrum trade-off: high-dim supports flexible coding, low-dim supports robust readout — exactly mirrors the project's expand→compress arc.
- **Cohen, Chung, Lee, Sompolinsky, "Separability and geometry of object manifolds in DNNs"** (Nature Communications 11:746, 2020). Manifold capacity theory: lower manifold dimension → higher classification capacity. Theoretical grounding for why correct-answer compression (PR ≈ 4) supports separability.
- **Valeriani et al., "The Geometry of Hidden Representations of Large Transformer Models"** (NeurIPS 2023, arXiv:2302.00294, repo: diegodoimo/geometry_representations). Documents the canonical hump-shaped ID profile (expansion → compression) across protein and image transformers — the closest layer-depth precedent for the project's generation-time breathing.
- **Cheng, Doimo, Kervadec et al., "Emergence of a High-Dimensional Abstraction Phase in Language Transformers"** (ICLR 2025, arXiv:2405.15471, repo: chengemily1/id-llm-abstraction). Across 5 LMs and 3 corpora: a central high-ID abstraction phase whose peak-height predicts LM performance — **the single most analogous prior result, but along depth, not time**.
- **Ansuini, Laio et al., "Intrinsic Dimension of Data Representations in Deep Neural Networks"** (NeurIPS 2019, arXiv:1905.12784). Original observation: ID hump shape across layers; final-layer ID predicts test accuracy *across models*, not per-sample.
- **Wu & Papyan, "Linguistic Collapse: Neural Collapse in (Large) Language Models"** (NeurIPS 2024, arXiv:2405.17767, repo: rhubarbwu/linguistic-collapse). Top-layer features in CLMs converge toward simplex-ETF geometry as scale and regularization grow. **Strongest precedent for finding 1's "commitment scales with model size".**
- **Hierarchical Reasoning Model (HRM)** (arXiv:2506.21734, 2025). Trained HRMs spontaneously develop a dimensionality hierarchy with high-level modules ~3× higher PR — direct ML precedent that high-dim mid-reasoning representations matter.
- **Chun, Canatar, Chung, Lee, "Estimating Dimensionality of Neural Representations from Finite Samples"** (arXiv:2509.26560, 2025). Bias-corrected PR estimator — **important methodological caveat**: PR estimates from few tokens (especially for the PR ≈ 4–6 endpoints) need bias correction.

### A.4 Why prefill is orthogonal to final-token (cos = -0.06 between "can I solve" and "did I solve")

This is the project's most novel single finding; the literature establishes that the two computations *exist*, but no surveyed work measures their geometric relationship.

- **Kadavath et al., "Language Models (Mostly) Know What They Know"** (Anthropic, arXiv:2207.05221). Introduces **P(IK)** (predict pre-generation correctness) and **P(True)** (post-hoc evaluation) as separate probes — but never compares them as directions.
- **Geva, Bastings, Filippova, Globerson, "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"** (EMNLP 2023, arXiv:2304.14767). Three-stage recall: subject-token enrichment in early MLPs, propagation, attention-based extraction. Mechanistically explains why the prefill last-prompt-token is information-dense.
- **Slobodkin et al., "The Curious Case of Hallucinatory (Un)answerability"** (EMNLP 2023, arXiv:2310.11877). Earliest demonstration that prompt-position activations encode (un)answerability.
- **Orgad et al., "LLMs Know More Than They Show"** (ICLR 2025, arXiv:2410.02707, repo: technion-cs-nlp/LLMsKnow). Truthfulness is concentrated in *exact-answer tokens* and **encoding is multifaceted, not universal** — closest analogue to the project's orthogonality claim.
- **Song et al., "Mind the Gap: Examining the Self-Improvement Capabilities of LLMs"** (ICLR 2025, arXiv:2412.02674). Formalizes the **generation-verification gap (GV-Gap)**: models verify better than they generate. Conceptual precedent but non-geometric.
- **Lugoloobi & Russell, "LLMs Encode How Difficult Problems Are"** (arXiv:2510.18147, 2025). Linear probes on Qwen2.5-Math-1.5B (the project's exact model) on Easy2HardBench (math) decode human-labeled difficulty with Spearman ρ ≈ 0.88.
- **"LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations"** (arXiv:2602.09924, 2026, ICLR-LIT 2026). Pre-generation probes get AUROC > 0.7; **probe AUROC drops monotonically as reasoning budget grows** — important caveat if extending to long-CoT thinking models.

### A.5 Why the Difference-of-Means direction rotates during generation (cos < 0.2 across positions)

- **Cherepanov et al., "Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors"** (arXiv:2509.06608, repo: corl-team/steering-reasoning). Last-layer SV acts as a token-substitution bias on the **first generated token only**; penultimate-layer SV affects MLP outputs differently. Direct mechanistic evidence for position-dependent SV effects.
- **Tan, Chanin, Lynch et al., "Analyzing the Generalization and Reliability of Steering Vectors"** (NeurIPS 2024, arXiv:2407.12404). Steerability is highly variable; cached SVs are unreliable across distributions — **explains the cos = 0.05 finding for the cached vector**.
- **Pres, Ruis, Lubana, Krueger, "Towards Reliable Evaluation of Behavior Steering Interventions"** (MINT@NeurIPS 2024, arXiv:2410.17245). CAA's reported effectiveness is over-stated under stricter evaluation.
- **Braun et al., "Understanding (Un)Reliability of Steering Vectors in LMs"** (ICLR 2025, arXiv:2505.22637). Higher cosine separation between training contrast pairs predicts more effective steering; cos = 0.05 = unsteerable. **Directly explains finding 3's cached-vector failure.**
- **Wollschläger et al., "The Geometry of Refusal: Concept Cones and Representational Independence"** (ICML 2025, arXiv:2502.17420). Refusal lives in multi-dimensional cones, not a single direction — formal proof that single-direction steering is insufficient.
- **Zhang & Nanda, "Towards Best Practices of Activation Patching"** (ICLR 2024, arXiv:2309.16042). Fixed interventions push activations OOD — exactly the failure mode of finding 3's E1 attempt.

### A.6 Why representation dispersion (D2H) is predictive of correctness

- **Ethayarajh, "How Contextual are Contextualized Word Representations?"** (EMNLP-IJCNLP 2019, arXiv:1909.00512). Documents that representations are anisotropic in all layers — sets the baseline against which D2H's intra-layer dispersion is measured.
- **Godey, de la Clergerie, Sagot, "Anisotropy Is Inherent to Self-Attention in Transformers"** (EACL 2024, arXiv:2401.12143). Anisotropy is intrinsic to self-attention, not an artifact of training objectives.
- **Cai, Huang, Bian, Church, "Isotropy in the Contextual Embedding Space: Clusters and Manifolds"** (ICLR 2021, arXiv:2012.05864). Anisotropy is mostly cluster-driven; within clusters, representations can be isotropic.
- **Chen et al., "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection" (EigenScore)** (ICLR 2024, arXiv:2402.03744). EigenScore = differential entropy of multi-sample-embedding covariance — direct precursor and conceptual cousin of D2H-score.
- **D²HScore** (arXiv:2509.11569, 2025). **This is the same construction as the project's D2H feature** — intra-layer dispersion + inter-layer drift. The project's 0.806 AUROC essentially replicates this preprint.
- **Du, Xiao, Li, "HaloScope"** (NeurIPS 2024 spotlight, arXiv:2409.17504, repo: deeplearning-wisc/haloscope). Hallucination subspace identified via SVD on unlabeled generations.

### A.7 Why output length confounds hidden-state features (deconfound drops AUROC by 0.050; absent in raw activations)

- **Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying" (SAPLMA)** (Findings of EMNLP 2023, arXiv:2304.13734). **Earliest explicit warning** that LLM-assigned probability is confounded by sentence length and word frequency.
- **Singhal, Goyal, Xu, Durrett, "A Long Way to Go: Investigating Length Correlations in RLHF"** (COLM 2024, arXiv:2310.03716). Length-only reward reproduces most RLHF gains.
- **Chen, Zhu et al., "ODIN: Disentangled Reward Mitigates Hacking in RLHF"** (arXiv:2402.07319, 2024). Two-head decorrelation — **direct precedent for the residualization-deconfounding** the project used in Experiment 1.
- **"Bias Fitting to Mitigate Length Bias of Reward Model in RLHF (FiMi-RM)"** (arXiv:2505.12843, 2025). Length-reward bias is non-linear; ResNet head fits-and-subtracts.
- **"Mitigating Length Bias in RLHF through a Causal Lens"** (arXiv:2511.12573, 2025). Causal counterfactual augmentation.
- **"Between Underthinking and Overthinking"** (arXiv:2505.00127, 2025). Incorrect responses are systematically much longer than correct ones — **direct empirical support for the length-correctness anti-correlation**.
- **"More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models"** (arXiv:2505.21523, 2025). Longer reasoning chains → more hallucination.
- **"Are Reasoning Models More Prone to Hallucination?"** (arXiv:2505.23646, 2025). Long-CoT-trained LRMs hallucinate more; flaw-repetition and think-answer mismatch.

### A.8 Why the correctness signal is approximately linear (XGBoost ≈ LR by -0.005)

- **Park, Choe, Veitch, "The Linear Representation Hypothesis and the Geometry of Large Language Models"** (ICML 2024, arXiv:2311.03658, repo: KihoPark/linear_rep_geometry). Formal counterfactual definitions of linear representation and equivalence under causal inner product. **Theoretical justification for finding 4.**
- **Elhage, Nanda et al., "A Mathematical Framework for Transformer Circuits"** and **"Toy Models of Superposition"** (Anthropic, transformer-circuits.pub). Residual stream as linear additive workspace; high-level features encoded as approximate directions even under superposition.
- **Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks"** (EMNLP 2019, arXiv:1909.03368). Linear probes have higher selectivity than MLPs — XGBoost ≈ LR is the methodologically conservative outcome.
- **Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances"** (Computational Linguistics 2022, arXiv:2102.12452). Survey grounding the linear-probing-sufficiency methodology.
- **Jiang, Aragam, Veitch et al., "On the Origins of Linear Representations in LLMs"** (arXiv:2403.03867, 2024). Proves softmax + cross-entropy + GD implicit bias produces linear concept representations.
- **"Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation"** (arXiv:2512.22245, 2025). Brier-loss-trained linear probe beats prompted self-confidence and complex methods — **independent confirmation of finding 4**.

### A.9 Why correct answers compress more than incorrect (asymmetric collapse PR ≈ 4.3 vs 8.5)

- **Wu & Papyan, "Linguistic Collapse"** (NeurIPS 2024, arXiv:2405.17767). Already cited — neural collapse in CLMs scales with model size, providing the geometric vocabulary for "commitment".
- **Kothapalli et al., "Generalized Neural Collapse for many classes"** (NeurIPS 2024). Theoretical extension to vocabularies ≫ embedding dim.
- **Hamilton, "Detecting Mode Collapse in Language Models via Narration"** (arXiv:2402.04477). Behavioral analog of geometric commitment.
- **Kudo et al., "LLMs Faithfully and Iteratively Compute Answers During CoT"** (arXiv:2412.01113, 2024). Probes show models obtain sub-answers during generation; causal interventions confirm a recency-biased commitment graph. **Direct evidence that commitment occurs during reasoning, not before.**
- **Afzal et al., "Knowing Before Saying: LLM Representations Encode Information About CoT Success Before Completion"** (arXiv:2505.24362, 2025). Probing predicts CoT correctness from representations early in generation — supports asymmetric-collapse interpretation.

### A.10 Cross-domain transfer of correctness signals — who else has shown this?

- **Orgad et al., "LLMs Know More Than They Show"** (ICLR 2025, arXiv:2410.02707). **Probes do NOT generalize across datasets** — strongly resonates with the project's finding that PH transfers at chance (0.354/0.497) and D2H asymmetrically.
- **Cencerrado et al., "No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes"** (arXiv:2509.10625, 2025). Probes trained on TriviaQA generalize across factual datasets but **fail on math/arithmetic (GSM8K)** — sets up the project's MATH-500 result as a notable exception (likely Qwen2.5-Math specialization).
- **Bao et al., "Probing the Geometry of Truth: Consistency and Generalization Across Logical Transformations and QA"** (ACL Findings 2025, arXiv:2506.00823). Capable models show more consistent truth directions across logical transformations.
- **Bürger, Hamprecht, Nadler, "Truth is Universal"** (NeurIPS 2024, arXiv:2407.12831, repo: sciai-lab/Truth_is_Universal). 2-D truth subspace appears in Gemma, Llama2, Mistral, Llama3 — universality counterpoint.
- **"LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance"** (arXiv:2510.11905, 2025). Probe-based detectors degrade sharply OOD.

---

## Category B — Present: Parallel work reaching similar or divergent conclusions

### B.1 Hidden-state correctness prediction and the "LLM already knows" phenomenon

This is the densest current research area. **The user's reference to "Zhu et al. 'The LLM Already Knows'" maps to Zhu, Liu, Lin, Tong, Zhong, Shao, "The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations" (arXiv:2509.12886, 2025).** Models token-level generation as a Markov chain and estimates difficulty from the *initial* hidden state alone, validated on Qwen2.5-VL-7B and InternVL3-8B.

Other directly comparable contemporaries:

- **Cencerrado et al.** (arXiv:2509.10625) — closest sibling methodology; reports failure on math.
- **Zhang et al., "Reasoning Models Know When They're Right"** (COLM 2025, arXiv:2504.05419, repo: AngelaZZZ-611/reasoning_models_probing). Lightweight 2-layer probes on hidden states at end of CoT chunks; ROC-AUC > 0.9 on AIME with R1-Distill-Qwen-32B. Used for early-exit (24% token reduction). **Directly supports the project's premise but uses mid-trajectory rather than prefill — so contradicts only the *prefill is best* claim, not the existence of the signal.**
- **Manvi, Singh, Ermon, "Adaptive Inference-Time Compute"** (arXiv:2410.02725, 2024). Generative reward-model formulation: LLMs predict mid-generation if restarting will yield better output.
- **Ji et al., "LLM Internal States Reveal Hallucination Risk Faced With a Query"** (BlackboxNLP 2024). Query-stage hallucination prediction.
- **Su et al., "MIND: Unsupervised Real-Time Hallucination Detection"** (Findings ACL 2024, arXiv:2403.06448, repo: oneal2000/MIND).
- **"ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection"** (ACL 2025). Cross-layer residual stream probe (peak AUC at layer ~11/32).
- **Wang et al. (CoE)** "Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation" (ICLR 2025, arXiv:2410.13640, repo: Alsace08/Chain-of-Embedding) — **the canonical CoE paper the project's pathway 8 implements**.
- **"Tracing the Traces: Latent Temporal Signals for Efficient and Accurate Reasoning"** (arXiv:2510.10494, 2025). Latent-Trajectory signals (net change, accumulated change, progress) — successor to CoE.
- **Gottesman & Geva, "Estimating Knowledge in LLMs Without Generating a Single Token (KEEN)"** (arXiv:2406.12673, 2024). Subject-representation probe predicts QA accuracy and FActScore before generation.
- **Kapoor et al., "Large Language Models Must Be Taught to Know What They Don't Know"** (NeurIPS 2024, arXiv:2406.08391, repo: activatedgeek/calibration-tuning). 1k examples + LoRA on features yields cross-model uncertainty.
- **Sriramanan et al., "LLM-Check"** (NeurIPS 2024).
- **Beigi et al., "Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs"** (ACL 2025, arXiv:2505.16520). SAPLMA generalization to LLM-generated facts is poor — partial contradiction of the strong-signal claim.

### B.2 Representation dynamics during chain-of-thought

- **"LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals"** (arXiv:2604.05655, 2026). Models CoT as a trajectory; correctness decoded with AUROC up to 0.87 from late-step deviations from ideal trajectory. **Achieves higher AUROC than the project's PH ceiling — strongly suggests step-wise temporal structure is the right representation, not static topology.**
- **"REMA: A Unified Reasoning Manifold Framework"** (arXiv:2509.22518, 2025). Defines correct-trajectory manifold; errors as geometric deviation.
- **"The Geometry of Reasoning: Flowing Logics in Representation Space"** (arXiv:2510.09782, 2025). Menger curvature of cumulative-prefix embeddings.
- **"Emergent Manifold Separability during Reasoning in Large Language Models"** (arXiv:2602.20338, 2026). Manifold Capacity Theory: separability is a *transient pulse*, then re-collapses — **explains why static PH summaries miss the signal**.
- **"The Geometry of Thought: How Scale Restructures Reasoning"** (arXiv:2601.13358, 2026; v2 withdrawn — treat preliminarily). Math reasoning trajectories remain "Liquid"; law trajectories crystallize. Scale-dependent geometric reorganization independently observed.
- **CoT Vectors** (arXiv:2510.00579, 2025). U-shaped three-stage structure with high-dim middle phase, no dominant direction — **directly mirrors expand→compress along layers within CoT-bearing forward passes**.
- **Joshi et al., "Geometry of Decision Making in LLMs"** (NeurIPS 2025). MLE/TwoNN/GRIDE on MLP outputs reveals hump shape coinciding with decision formation.

### B.3 Activation steering for accuracy/reasoning (not just truthfulness/safety)

The 2024–2026 literature has produced many direct competitors to the project's E1 goal:

- **Højer, Jarvis, Heinrich, "Improving Reasoning Performance in LLMs via Representation Engineering"** (ICLR 2025, arXiv:2504.19483, repo: bertramhojer/improve-reasoning-iclr-2025). Most direct precedent: control vectors on Mistral-7B residuals for bAbI/GSM8K/IOI.
- **Chen et al., "SEAL: Steerable Reasoning Calibration of LLMs for Free"** (COLM 2025, arXiv:2504.07986, repo: VITA-Group/SEAL). Reduces tokens 11.8–50.4% AND improves accuracy up to 11% on MATH500/GSM8K **with DeepSeek-R1-Distill 1.5B/7B — same model class as the project**.
- **Tang et al., "GLoRE: Unlocking General Long Chain-of-Thought Reasoning Capabilities via Representation Engineering"** (ACL 2025, arXiv:2503.11314). Question-aware retrieval of fine-grained representations.
- **Sun et al., "Feature Extraction and Steering for Enhanced CoT Reasoning"** (arXiv:2505.15634, 2025). SAE-based decomposition; SAE-free variant.
- **Sinii et al., "Steering LLM Reasoning Through Bias-Only Adaptation"** (arXiv:2505.18706, 2025). One d-dim steering vector per layer trained with RL matches full RL fine-tuning at 0.0016% extra parameters.
- **Cherepanov et al., "Small Vectors, Big Effects"** (already cited). Mechanistic analysis showing SV effects concentrate on first generated token.
- **Venhoff et al., "Understanding Reasoning in Thinking LMs via Steering Vectors"** (arXiv:2506.18167, 2025, repo: cvenhoff/steering-thinking-llms). Linear directions for backtracking, uncertainty, hypothesis-generation.
- **CREST** (arXiv:2512.24574, 2026). +17.50% AMC23, 37.60% token reduction with R1-1.5B.
- **Bharadwaj, "STU-PID: Steering Token Usage via PID Controller"** (arXiv:2506.18831, 2025, repo: arambharadwaj/pid_steering). +3.9pp accuracy and 23% token reduction on GSM8K with **DeepSeek-R1-Distill-Qwen-1.5B (the project's exact model)**.
- **Nguyen et al., "Activation Steering with a Feedback Controller (PID Steering)"** (arXiv:2510.04309, 2025). Full PID frame for steering — most relevant to the rotation problem.
- **Scalena, Sarti, Nissim, "Multi-Property Steering with Dynamic Activation Composition (DAC)"** (BlackboxNLP 2024). KL-modulated per-token steering — most-cited adaptive method.
- **Lee et al., "Conditional Activation Steering (CAST)"** (ICLR 2025, arXiv:2409.05907, repo: IBM/activation-steering).
- **Wang, Yang, Peng, "Semantics-Adaptive Dynamic Intervention (SADI)"** (ICLR 2025, arXiv:2410.12299, repo: weixuan-wang123/SADI).
- **"Steering Vector Fields"** (arXiv:2602.01654, 2026). Explicitly motivates and addresses the rotation problem.
- **"ODESteer: ODE-Based Steering Framework"** (OpenReview 2025). Reformulates steering as ODE integration; conventional ActAdd is first-order Euler.
- **AxBench** (arXiv:2501.17148, 2025) and **"Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts"** (arXiv:2506.23845, 2025). SAE steering underperforms simpler baselines.

### B.4 Selective prediction via internal signals (the project's strongest result, finding 7)

- **Cencerrado et al.** (already cited). Same recipe; fails on math — making the project's MATH-500 result a notable contrasting data point.
- **Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees"** (ICML 2024, arXiv:2402.10978, repo: tatsu-lab/conformal-factual-lm). Reports MATH numbers under conformal back-off — **direct numerical comparison**.
- **Quach et al., "Conformal Language Modeling"** (ICLR 2024, arXiv:2306.10193, repo: Varal7/conformal-language-modeling).
- **Yadkori et al., "Mitigating LLM Hallucinations via Conformal Abstention"** (arXiv:2405.01563, 2024, DeepMind).
- **Kossen et al., "Semantic Entropy Probes (SEPs)"** (arXiv:2406.15927, 2024, repo: OATML/semantic-entropy-probes). **Single-pass linear probe approximating semantic entropy** — closest methodological cousin; notes prefill probes are nearly as informative as post-answer.
- **Wen et al., "Know Your Limits: A Survey of Abstention in LLMs"** (TACL 2025, arXiv:2407.18418).
- **Kirichenko et al., "AbstentionBench"** (arXiv:2506.09038, 2025). Reasoning fine-tuning degrades abstention by 24%.
- **Tomani et al., "Uncertainty-Based Abstention in LLMs Improves Safety"** (arXiv:2404.10960, 2024).
- **Traub et al., "Overcoming Common Flaws in the Evaluation of Selective Classification Systems"** (NeurIPS 2024, arXiv:2407.01032). Proposes AUGRC — **the project should report this metric**.
- **"When Silence Is Golden: Can LLMs Learn to Abstain?"** (OpenReview 2025). Uses Qwen2.5-1.5B-Instruct (the project's base) and finds RL-abstention beats GPT-4o.
- **Vazhentsev et al., "LM-Polygraph"** (TACL 2025). UQ benchmark.
- **"Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations"** (arXiv:2604.16217, 2026). Combines conformal prediction with hidden-state probes.

### B.5 Verbalized confidence vs internal probes

- **Lin, Hilton, Evans, "Teaching Models to Express Their Uncertainty in Words"** (TMLR 2022, arXiv:2205.14334, repo: sylinrl/CalibratedMath).
- **Tian et al., "Just Ask for Calibration"** (EMNLP 2023, arXiv:2305.14975). Verbalized > token-logprob for RLHF models.
- **Xiong et al., "Can LLMs Express Their Uncertainty?"** (ICLR 2024, arXiv:2306.13063, repo: MiaoXiong2320/llm-uncertainty). Black-box vs white-box gap is narrow (0.522 → 0.605 AUROC).
- **Yoon et al., "Reasoning Models Better Express Their Confidence"** (arXiv:2505.14489, 2025).
- **Damani et al., "Uncertainty Distillation"** (arXiv:2503.14749, 2025).
- **Detommaso et al., "Multicalibration for Confidence Scoring in LLMs"** (ICML 2024, arXiv:2404.04689).

### B.6 Self-consistency failure modes (the D-bucket, finding 8)

- **Wang et al., "Self-Consistency Improves CoT Reasoning"** (ICLR 2023, arXiv:2203.11171). Foundational SC.
- **Wang, Prasad, Stengel-Eskin, Bansal, "Soft Self-Consistency"** (ACL 2024, arXiv:2402.13212, repo: HanNight/soft_self_consistency). **Documents SC failure when valid answers are diverse** — closest precedent for D-bucket.
- **Stroebl et al., "Inference Scaling FLaws"** (arXiv:2411.17501, 2024). Weaker models produce false-positive agreements; majority voting plateaus and degrades.
- **Brown et al., "Large Language Monkeys"** (arXiv:2407.21787, 2024, repo: ScalingIntelligence/large_language_monkeys). Coverage scales log-linearly with K but **selection methods plateau** — same gap as D-bucket.
- **"How Effective Is Self-Consistency for Long-Context Problems?"** (arXiv:2411.01101, 2024). SC fails when errors are correlated.
- **"Rethinking Fine-Tuning when Scaling Test-Time Compute"** (arXiv:2502.07154, 2025). CE-trained models become overconfident, hurting pass@N — mechanism for D-bucket.
- **Taubenfeld et al., "CISC: Confidence Improves Self-Consistency"** (arXiv:2502.06233, 2025). Confidence-weighted majority cuts cost ~46%.
- **Aggarwal et al., "Adaptive-Consistency"** (EMNLP 2023, arXiv:2305.11860, repo: Pranjal2041/AdaptiveConsistency). Beta-Dirichlet adaptive K.
- **Li et al., "Early-Stopping Self-Consistency (ESC)"** (ICLR 2024, arXiv:2401.10480). 33–84% cost reduction.
- **Wang et al., "Difficulty-Adaptive Self-Consistency (DSC)"** (NAACL Findings 2025, arXiv:2408.13457, repo: WangXinglin/DSC).
- **"Reliability-Aware Adaptive Self-Consistency (ReASC)"** (arXiv:2601.02970, 2026).
- **"ACTSC: Activation-Informed Difficulty-Aware SC"** (arXiv:2602.09438, 2026). **Probe on FFN activations to predict difficulty without pre-sampling** — most directly aligned with the project's aspirations for finding 6.
- **Fu et al., "Deep Think with Confidence (DeepConf)"** (arXiv:2508.15260, 2025, repo: facebookresearch/deepconf). AIME 2025 → 99.9% with 84.7% fewer tokens via local-confidence filtering — **state-of-the-art validation that sequence-position confidence dominates global signal (finding 6e)**.
- **"Maximizing Prefix-Confidence at Test-Time"** (arXiv:2507.18122, 2025).
- **Z. Zhou et al., "Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning (RPC)"** (NeurIPS 2025, arXiv:2502.00511; theoretical follow-up arXiv:2510.15444). Theory: SC has high estimation error; perplexity has high model error; combining gives exponential convergence.

### B.7 Test-time compute scaling

- **Snell et al., "Scaling LLM Test-Time Compute Optimally"** (ICLR 2025, arXiv:2408.03314). **Compute-optimal allocation depends on prompt difficulty bin.**
- **Wu et al., "Inference Scaling Laws (REBASE)"** (ICLR 2025, arXiv:2408.00724, repo: thu-wyz/inference-scaling).
- **Singhi et al., "When To Solve, When To Verify"** (arXiv:2504.01005, 2025, repo: nishadsinghi/sc-genrm-scaling). SC > GenRM up to 8× budget.
- **Liu et al., "Can 1B LLM Surpass 405B LLM?"** (arXiv:2502.06703, 2025).
- **Muennighoff et al., "s1: Simple Test-Time Scaling"** (arXiv:2501.19393, repo: simplescaling/s1).
- **Guan et al., "rStar-Math: Small LLMs Can Master Math Reasoning"** (ICML 2025, arXiv:2501.04519, repo: microsoft/rStar). **Pushes Qwen2.5-Math-1.5B from 51.2% → 87.8% on MATH-500** — the most ambitious comparable target.
- **Wang et al., "OpenR"** (arXiv:2410.09671, 2024, repo: openreasoner/openr). Process search on Qwen2.5-Math-1.5B/7B.
- **"Towards Thinking-Optimal Scaling"** (arXiv:2502.18080, 2025). Longer CoTs hurt in some domains.
- **Zuo & Zhu, "Strategic Scaling of Test-Time Compute"** (arXiv:2506.12721, 2026). Bandit framework; +11.1% MATH-500 with Llama-3.2-1B.

### B.8 Process reward models (relation to hidden-state probes)

- **Lightman et al., "Let's Verify Step by Step"** (ICLR 2024, arXiv:2305.20050, repo: openai/prm800k). **Defines MATH-500.**
- **Wang et al., "Math-Shepherd"** (ACL 2024, arXiv:2312.08935).
- **Luo et al., "OmegaPRM"** (arXiv:2406.06592, 2024).
- **Zhang et al., "Generative Verifiers (GenRM)"** (ICLR 2025, arXiv:2408.15240).
- **Setlur et al., "Rewarding Progress: Scaling Automated Process Verifiers (PAV)"** (arXiv:2410.08146, 2024). Step-level *advantage* signal.
- **Khalifa et al., "ThinkPRM"** (arXiv:2504.16828, 2025). 1K-example PRM matches massive PRMs.
- **Zhang et al., "Lessons of Developing Process Reward Models in Mathematical Reasoning"** (arXiv:2501.07301, 2025). MC-estimation pitfalls.
- **Cui, Yuan et al., "Free Process Rewards Without Process Labels (PRIME)"** (2025). Implicit PRM from outcome.

### B.9 Cross-model and cross-domain probe transfer

- **Bello et al., "Linear Representation Transferability Hypothesis (LRT)"** (arXiv:2506.00653, 2025). **Affine maps between hidden states of different-sized LMs allow steering vector transfer** — the strongest direct precedent for the project's E4 distillation goal.
- **Mallen & Belrose, "Eliciting Latent Knowledge from Quirky Language Models"** (arXiv:2312.01037, 2023). Probes trained in easy contexts generalize OOD.
- **"Layer by Layer: Uncovering Hidden Representations in Language Models"** (arXiv:2502.02013, 2025). Cross-scale probing — intermediate layers often beat final by up to 16%.
- **"Linear Probe Accuracy Scales with Model Size"** (arXiv:2604.13386, 2026). Log-linear deception-probe AUROC scaling — explains the 7B's higher AUROC (0.876) vs 1.5B's (0.771).
- **Wendler et al., "Do Llamas Work in English?"** (ACL 2024, arXiv:2402.10588, repo: epfl-dlab/llm-latent-language).
- **Lan et al., "Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models"** (arXiv:2410.06981, 2024/2025).
- **Huh, Cheung, Wang, Isola, "The Platonic Representation Hypothesis"** (ICML 2024, arXiv:2405.07987).
- **Schaeffer, Miranda, Koyejo, "Are Emergent Abilities of LLMs a Mirage?"** (NeurIPS 2023, arXiv:2304.15004). With continuous metrics, scaling is smooth — supports the **ρ = 0.864 cross-scale difficulty correlation finding**.
- **Easy2Hard-Bench / PSN-IRT** (arXiv:2409.18433; arXiv:2505.15055). Item-difficulty parameters stable across models.

---

## Category C — Future: Experiments that could extend this work

### C.1 Steering methods that handle direction rotation

- **Steering Vector Fields** (arXiv:2602.01654, 2026). Differentiable concept-scoring whose local gradient defines context-dependent direction.
- **ODESteer** (OpenReview 2025). Multi-step adaptive steering via barrier function.
- **PID Steering** (Nguyen et al. arXiv:2510.04309; STU-PID arXiv:2506.18831). **STU-PID is the most actionable: directly applicable to Qwen2.5 1.5B.**
- **Activation State Machine (ASM)** (OpenReview 2025/2026). Kalman-filter-style stateful steering.
- **DAC** (Scalena et al., BlackboxNLP 2024). KL-modulated per-token.
- **SADI** (Wang et al., ICLR 2025).
- **WAS** (Hegazy et al., arXiv:2505.20309, 2025).
- **Flexible Activation Steering with Backtracking (FASB)** (arXiv:2508.17126, 2025).
- **Divergence Steering** (arXiv:2512.22944, 2025). KL-based decoding-time steering on math reasoning.
- **HyperSteer** (Sun et al., 2025). Hypernetwork-generated steering vectors.

### C.2 Improving small model reasoning at inference time without retraining

- **rStar-Math** (microsoft/rStar) and **OpenR** (openreasoner/openr) — both open-source and target Qwen2.5-Math-1.5B/7B specifically.
- **DeepConf** (facebookresearch/deepconf).
- **LATTS: Locally Adaptive Test-Time Scaling** (arXiv:2509.20368, 2025). Per-step verifier-driven allocation.
- **LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning** (arXiv:2512.05325, 2025). **Probe at "wait/hmm" cue tokens + conformal calibration for distribution-free early-exit guarantees** — combines the project's probe direction with conformal guarantees.
- **"Adaptive Test-Time Compute Allocation with Evolving In-Context Demonstrations"** (arXiv:2604.21018, 2026).

### C.3 Confidence-aware decoding strategies

- **"ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States"** (arXiv:2511.06209, 2025). Transformer-encoder probe for step-level correctness.
- **"SPAE: Step Potential Advantage Estimation"** (arXiv:2601.03823, 2026).
- **Wang et al., "Confidence-Aware Reasoning"** (EMNLP 2025 industry track). Bayesian-EIG threshold to insert `</think>` early.
- **"CGES: Confidence-Guided Early Stopping"** (arXiv:2511.02603, 2025). Bayesian aggregation with theoretical guarantees.
- **"Seer Self-Consistency"** (arXiv:2511.09345, 2025). System-1 entropy estimation pre-allocates System-2 budget.

### C.4 Distillation of internal confidence/correctness signals from large to small

- **Hager et al., "Uncertainty Distillation"** (arXiv:2503.14749, 2025). Self-distills semantic confidence into output tokens.
- **"Efficient Uncertainty Estimation via Distillation of Bayesian Large Language Models (EUD)"** (arXiv:2505.11731, 2025).
- **"Efficient Uncertainty in LLMs through Evidential Knowledge Distillation"** (arXiv:2507.18366, 2025).
- **"CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?"** (arXiv:2510.24505, 2025). Distilled student exceeds teacher calibration on MATH-Perturb.
- **Burns et al., "Weak-to-Strong Generalization"** (ICML 2024, arXiv:2312.09390, repo: openai/weak-to-strong). The inverse direction — strong-to-weak (the project's E4 setup) is less studied.
- **"Understanding the Capabilities and Limitations of Weak-to-Strong Generalization"** (arXiv:2502.01458, 2025). KL theory and calibration analysis.

### C.5 Topological methods that actually work on neural activations (beyond raw PH)

- **"Persistent Topological Features in Large Language Models"** (Gardinazzi et al., arXiv:2410.11042, repo: RitaSciencePark/topo_llm). **Zigzag-persistence pipeline that explicitly avoids Dionysus2** — direct unblocking path for the project's pathway 7 limitation.
- **Dey & Hou, "Computing Zigzag Persistence on Graphs in Near-Linear Time"** (SoCG 2021, arXiv:2103.07353, repo: taohou01/fzz). The fast-zigzag implementation.
- **Dey & Samaga, "Quasi Zigzag Persistence"** (ICML 2025, arXiv:2502.16049). Modern multiparameter PH + zigzag tooling.
- **"Zigzag Persistence of Neural Responses to Time-Varying Stimuli"** (arXiv:2603.03037, 2026). Frame-by-frame cubical zigzag template.
- **Yin, Srinivasa, Chang, "Characterizing Truthfulness with Local Intrinsic Dimension (LID)"** (ICML 2024, arXiv:2402.18048, repo: fanyin3639/Truthfulness-LID). **LID has demonstrated per-sample predictive value where global TwoNN failed.**
- **Rigoni et al., "Less is More: Local Intrinsic Dimensions of Contextual Language Models"** (NeurIPS 2025, arXiv:2506.01034, repo: aidos-lab/Topo_LLM_public).
- **HalluZig** (arXiv:2601.01552, 2026). Zigzag PH on attention graphs.
- **Bazarova et al., "TOHA: Topological Divergence on Attention Graphs"** (arXiv:2504.10063, 2025). PH **on attention graphs, not residuals** — the only regime where PH has shown reliable predictive value on LLM correctness.

### C.6 Process reward models and their relation to hidden-state probes

- **A. Zhang et al., "Reasoning Models Know When They're Right"** (already cited). Most directly relevant: hidden-state probes can serve as per-step verifiers cheaper than PRMs.
- **ThinkPRM** (Khalifa et al., arXiv:2504.16828). 1K examples → matches massive PRMs — shows PRM data efficiency potential.
- **"Hidden States as Early Signals: Step-Level Trace Evaluation"** (arXiv:2601.09093, 2026). Step-scorer on hidden-state averages at first 25/50/75% of trace.
- **PAV (Setlur et al.)**. Step-advantage formulation.

### C.7 Test-time compute scaling × internal confidence interaction

- **Damani, Shenfeld, Peng, Bobu, Andreas, "Learning How Hard to Think"** (arXiv:2410.04707, 2024, MIT). **Trains lightweight predictor on input embeddings to estimate marginal reward of more samples** — direct framework for the project's E2 dynamic-K goal.
- **Manvi, Singh, Ermon "Adaptive Inference-Time Compute"** (arXiv:2410.02725; follow-up arXiv:2512.01457). Mid-generation prediction of restart benefit.
- **Kang et al., "Scalable Best-of-N Selection via Self-Certainty"** (arXiv:2502.18581, 2025).
- **Huang et al., "Efficient Test-Time Scaling via Self-Calibration"** (arXiv:2503.00031, 2025).
- **"LLM Router: Rethinking Routing with Prefill Activations"** (arXiv:2603.20895, 2026). Prefill-activation-based routing — natural extension of the project's prefill DoM finding to the multi-model regime.

### C.8 Activation editing / representation patching for accuracy

- **Yin et al., "LoFiT: Localized Fine-Tuning on LLM Representations"** (NeurIPS 2024).
- **"Steer2Edit: From Activation Steering to Component-Level Editing"** (arXiv:2602.09870, 2026). Treats steering vectors as **diagnostic** signals revealing which heads/MLP-neurons govern a behavior; converts to selective weight edits — bypasses rotation entirely.
- **CorrSteer** (arXiv:2508.12535, 2025). Generation-time SAE steering via correlation with sample correctness.
- **"Probing the Difficulty Perception Mechanism of LLMs"** (arXiv:2510.05969, 2025). Locates final-layer attention heads with opposite easy/hard activation patterns.

### C.9 Per-token / adaptive direction methods

(See C.1 — heavily overlapping with the steering-rotation list. The top three actionable references are STU-PID, PID Steering, and DAC, all of which provide working code or precise protocols on small math models.)

### C.10 Cross-cutting frameworks worth tracking

- **Lugoloobi & Russell, "LLMs Encode How Difficult Problems Are"** (arXiv:2510.18147, 2025). **Uses Qwen2.5-Math-1.5B + MATH-500** (project's exact setup); steering toward "easier" representations reduces hallucination; during GRPO the human-difficulty probe strengthens.
- **"Mind the Gap"** (Song et al., ICLR 2025). GV-gap formalization.
- **"Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations"** (arXiv:2604.16217, 2026). Combines conformal prediction with hidden-state probes — provides the natural follow-on to obtain risk guarantees on the project's selective-prediction operating point.

---

## Synthesis — what this map reveals about the project's contributions

The project sits at the intersection of three rapidly converging 2024–2026 research lines: **hidden-state correctness probing** (CoE, D2H, P(IK), KEEN, SEPs, Cencerrado), **representation dynamics during reasoning** (Cheng abstraction phase, CoT Vectors, REMA, Trajectory Geometry, Linguistic Collapse), and **test-time compute scaling for small math models** (rStar-Math, OpenR, DeepConf, RPC, Damani). Pathway 9's experimental rigor — Gaussian-null testing, length-deconfounding, cross-domain transfer, XGBoost-vs-LR — anticipates and validates concerns that the broader field has only recently begun raising systematically (e.g., the 2026 spin-model paper independently proposing density-matched shuffles; the "LLM Knowledge is Brittle" critique; AbstentionBench's reasoning-degrades-abstention).

**Three findings appear genuinely novel** against the surveyed literature: **(i) the explicit cosine measurement showing prefill-correctness and final-token-correctness directions are orthogonal** (cos = -0.06) — Orgad's "multifaceted" claim and the GV-gap are the closest analogues but neither is geometric; **(ii) asymmetric collapse with magnitude scaling in model size** (PR ≈ 4.3/2.6 correct vs 8.5/6.9 incorrect; 1.5B → 7B) — Linguistic Collapse and the Cheng abstraction phase establish the depth-axis precedent, but the project's generation-time + correctness-asymmetry combination is, as far as the survey reveals, original; **(iii) explicit demonstration of direction rotation** (cos < 0.2 across 200 positions) breaking fixed-vector steering on math accuracy — Cherepanov's "first generated token" mechanism and Steering Vector Fields' setup motivate the problem, but the project's quantitative cliff is sharper than published numbers.

**The most consequential gaps the project reveals** are: (a) the field still lacks a *causal* test of whether prefill geometry drives downstream commitment or merely correlates with it (Kudo's recency-causal-graph work suggests the latter); (b) D-bucket pathologies (~7% on MATH-500) are not individually detectable from prefill features and remain the cleanest open challenge in self-consistency literature, with Soft-SC and CISC offering only partial remedies; (c) cross-scale probe transfer with proper labels appears flat, contradicting the optimistic LRT hypothesis (Bello 2025) and aligning with Orgad's multifaceted-encoding claim — suggesting strong-to-weak distillation of the correctness *direction* may be fundamentally harder than weak-to-strong of the correctness *behavior*.

The most actionable next experiments, mapped to existing tooling: replace global TwoNN with **local LID** (Yin et al., repo available); replace static Rips PH with **fast zigzag** on attention graphs (Dey-Hou repo, RitaSciencePark/topo_llm pipeline); test steering with **STU-PID** or **PID Steering** on the project's exact Qwen2.5-Math-1.5B base; compose the prefill DoM probe with **conformal abstention** (Mohri-Hashimoto on MATH provides the direct numerical comparison) for risk-guaranteed selective prediction; and benchmark dynamic K with **DSC, Damani's "Learning How Hard to Think", or DeepConf** as the contemporary baselines on MATH-500.