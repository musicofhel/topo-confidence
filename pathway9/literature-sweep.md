# topo-confidence P9 — Literature sweep reading list

_Generated 2026-04-23T13:31:36.502Z from link-forge Neo4j graph._

**Ingest:** 560 papers since 2026-04-22 · 385 arxiv · 312 tagged research-paper

Refresh: `cd ~/link-forge && node scripts/build-topo-confidence-reading-list.mjs > ~/topo-confidence/pathway9/literature-sweep.md`

## 2025–2026 (35)

### [Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling](https://arxiv.org/abs/2505.12225)
_2025 · forgeScore 0.78 · research-paper_

Introduces SWIFT, a lightweight method that learns a reward function from LLM hidden states using simple linear layers to enable efficient Best-of-N sampling without heavy text-based reward models.

### [Topology-Aware Layer Pruning for Large Vision-Language Models](https://arxiv.org/abs/2604.16502)
_2026 · forgeScore 0.72 · research-paper_

A topology-aware layer pruning framework for Large Vision-Language Models that uses zigzag persistent homology on simplicial complexes of hidden states to preserve transition-critical layers during compression.

### [Eliminating Hallucination-Induced Errors in LLM Code Generation with Functional Clustering](https://arxiv.org/abs/2506.11021)
_2025 · forgeScore 0.72 · analysis_

A black-box wrapper that eliminates hallucination-induced errors in LLM code generation by sampling candidate programs, executing them on self-generated tests, and clustering by I/O behavior to produce a tunable confidence score.

### [An Axiomatic Assessment of Entropy- and Variance-based Uncertainty Quantification in Regression](https://arxiv.org/abs/2504.18433)
_2025 · forgeScore 0.72 · research-paper_

Axiomatic analysis of entropy- and variance-based uncertainty measures for supervised regression, generalized via predictive exponential families.

### [ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks](https://arxiv.org/abs/2508.16889)
_2025 · forgeScore 0.70 · research-paper_

A benchmark (ObjexMT) testing whether LLM judges can recover hidden conversational objectives from multi-turn jailbreak transcripts and accurately self-report confidence, evaluated across six frontier models.

### [An Introduction to Topological Data Analysis Ball Mapper in Python](https://arxiv.org/abs/2505.03022)
_2025 · forgeScore 0.70 · tutorial_

An introductory guide with Python code for using Topological Data Analysis Ball Mapper (TDABM) to visualize multivariate datasets without information loss.

### [Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](https://arxiv.org/abs/2604.15741)
_2026 · forgeScore 0.70 · research-paper_

ACL 2026 paper introducing SIVR, a supervised hallucination detection framework for LLMs that uses token-wise, layer-wise variance of hidden states to estimate uncertainty.

### [Grammars of Formal Uncertainty: When to Trust LLMs in Automated Reasoning Tasks](https://arxiv.org/pdf/2505.20047)
_2025 · forgeScore 0.70 · research-paper_

Research paper investigating uncertainty quantification in LLM-generated formal specifications, introducing a probabilistic context-free grammar framework for selective verification that reduces errors by 14-100%.

### [Certifying Robustness via Topological Representations](https://arxiv.org/abs/2501.10876)
_2025 · forgeScore 0.70 · research-paper_

An arXiv paper proposing a neural network architecture that learns discriminative geometric representations from persistence diagrams with controllable Lipschitz stability to certify epsilon-robustness against adversarial perturbations.

### [Detecting AI Hallucinations in Finance: An Information-Theoretic Method Cuts Hallucination Rate by 92%](https://arxiv.org/abs/2512.03107)
_2025 · forgeScore 0.70 · research-paper_

ECLIPSE is an information-theoretic framework that detects LLM hallucinations in financial QA by measuring the mismatch between semantic entropy and evidence capacity, achieving 0.89 ROC AUC versus 0.50 for entropy-only baselines.

### [Are Hallucinations Bad Estimations?](https://arxiv.org/abs/2509.21473)
_2025 · forgeScore 0.70 · research-paper_

Theoretical paper formalizing hallucinations in generative models as estimation failures, proving that even loss-minimizing optimal estimators hallucinate due to structural misalignment between loss minimization and human-acceptable outputs.

### [EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs](https://arxiv.org/abs/2509.15735)
_2025 · forgeScore 0.70 · research-paper_

EigenTrack is a white-box real-time detector that uses spectral statistics (entropy, eigenvalue gaps, KL divergence) of hidden activation covariances to flag hallucination and OOD drift in LLMs/VLMs in a single forward pass.

### [HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification](https://arxiv.org/pdf/2504.07069)
_2025 · forgeScore 0.70 · research-paper_

Paper introducing HDM-2, a hallucination detection model that validates LLM outputs against both context and common knowledge, with a new HDMBench dataset.

### [Calibrating LLM Confidence by Probing Perturbed Representation Stability](https://arxiv.org/pdf/2505.21772)
_2025 · forgeScore 0.70 · research-paper_

Introduces CCPS, a method for calibrating LLM confidence by applying adversarial perturbations to final hidden states and training a lightweight classifier to predict answer correctness, achieving ~55% reduction in Expected Calibration Error.

### [Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](https://arxiv.org/abs/2604.04863)
_2026 · forgeScore 0.70 · research-paper_

A CVPR 2026 paper introducing a patch-level hallucination detection framework for large vision-language models that analyzes fine-grained token-level attention patterns to achieve up to 90% accuracy in detecting hallucinated tokens.

### [Towards Trustworthy Breast Tumor Segmentation in Ultrasound using Monte Carlo Dropout and Deep Ensembles for Epistemic Uncertainty Estimation](https://arxiv.org/abs/2508.17768)
_2025 · forgeScore 0.70 · research-paper_

Research paper evaluating Monte Carlo dropout and deep ensembles for epistemic uncertainty estimation in breast ultrasound tumor segmentation using a modified Residual Encoder U-Net.

### [Probabilistic distances-based hallucination detection in LLMs with RAG](https://arxiv.org/abs/2506.09886)
_2025 · forgeScore 0.70 · research-paper_

An arXiv paper proposing an unsupervised hallucination detection method for RAG systems based on probabilistic distances between prompt and response token embedding distributions.

### [Hallucination Detection with Small Language Models](https://arxiv.org/abs/2506.22486)
_2025 · forgeScore 0.70 · research-paper_

A research paper proposing a framework that uses multiple small language models to detect hallucinations in LLM responses by verifying sentences against retrieved context via 'Yes' token probabilities.

### [Large Language Models Hallucination: A Comprehensive Survey](https://arxiv.org/pdf/2510.06265)
_2025 · forgeScore 0.70 · reference_

Comprehensive survey of LLM hallucination covering taxonomy of types, root causes across the development lifecycle, detection approaches, mitigation strategies, and evaluation benchmarks.

### [An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs](https://arxiv.org/abs/2505.18332)
_2025 · forgeScore 0.70 · analysis_

Research paper demonstrating a reconstruction attack that breaks permutation-based private inference schemes for LLMs, recovering original prompts from permuted hidden states with near-perfect accuracy.

### [Evaluating and Calibrating LLM Confidence on Questions with Multiple Correct Answers](https://arxiv.org/abs/2602.07842)
_2026 · forgeScore 0.70 · research-paper_

Introduces MACE benchmark and Semantic Confidence Aggregation (SCA) to fix LLM confidence calibration failures on questions with multiple valid answers.

### [On LLMs' Internal Representation of Code Correctness](https://arxiv.org/abs/2512.07404)
_2025 · forgeScore 0.70 · research-paper_

ICSE'26 paper showing LLMs internally encode a code-correctness signal in hidden states that outperforms log-likelihood ranking and verbalized confidence for selecting correct code samples without test execution.

### [Latent Semantic Manifolds in Large Language Models](https://arxiv.org/abs/2603.22301)
_2026 · forgeScore 0.62 · analysis_

A mathematical framework treating LLM hidden states as points on a Riemannian latent semantic manifold, proving rate-distortion and linear volume scaling laws for vocabulary-induced expressibility gaps and validating them across six transformer models.

### [First Hallucination Tokens Are Different from Conditional Ones](https://arxiv.org/abs/2507.20836)
_2025 · forgeScore 0.60 · analysis_

Research paper showing that the first hallucinated token in an LLM output is far more detectable than subsequent ones, enabling more effective token-level hallucination detection.

### [Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration](https://arxiv.org/abs/2603.22812)
_2026 · forgeScore 0.60 · research-paper_

Proposes an adaptive Bayesian framework for semantic entropy that dynamically adjusts LLM sampling budgets to detect hallucinations more efficiently, achieving ~50% fewer samples and +12.6% AUROC over fixed-budget baselines.

### [D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/pdf/2509.11569)
_2025 · forgeScore 0.60 · research-paper_

A training-free, label-free framework (D²HScore) that detects LLM hallucinations by measuring intra-layer semantic dispersion and inter-layer representational drift of attention-selected tokens.

### [Mind the Generation Process: Fine-Grained Confidence Estimation During LLM Generation](https://arxiv.org/abs/2508.12040)
_2025 · forgeScore 0.60 · research-paper_

Introduces FineCE, a fine-grained confidence estimation method that predicts continuous confidence scores during LLM text generation, using a Backward Confidence Integration strategy to improve calibration.

### [Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation](https://arxiv.org/abs/2502.11620)
_2025 · forgeScore 0.60 · analysis_

Research paper adapting entropy and mutual-information uncertainty estimation techniques from NLG to LLM code generation, using symbolic-execution-based semantic equivalence checks to predict correctness and drive an abstention policy.

### [Dist2ill: Distributional Distillation for One-Pass Uncertainty Estimation in Large Language Models](https://arxiv.org/abs/2505.11731)
_2025 · forgeScore 0.60 · analysis_

Proposes Dist2ill, a distributional distillation framework that trains LLMs to produce diverse reasoning paths and calibrated confidence scores in a single forward pass.

### [Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection](https://arxiv.org/abs/2511.07364)
_2025 · forgeScore 0.60 · research-paper_

Research paper showing that stepwise confidence scoring outperforms holistic scoring (up to 15% AUC-ROC gain) for detecting failures in multi-step LLM reasoning tasks.

### [Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief](https://arxiv.org/abs/2509.01564)
_2025 · forgeScore 0.55 · research-paper_

Proposes EAGLE, a calibration method that aggregates internal hidden-state beliefs across multiple LLM layers during self-evaluation to produce more accurate confidence scores.

### [Mitigating Multimodal Hallucination via Phase-wise Self-reward](https://arxiv.org/abs/2604.17982)
_2026 · forgeScore 0.55 · research-paper_

Introduces PSRD (Phase-wise Self-Reward Decoding), a self-rewarding framework that mitigates vision hallucination in LVLMs at inference time by distilling guidance signals into a lightweight reward model.

### [Fact-Controlled Diagnosis of Hallucinations in Medical Text Summarization](https://arxiv.org/abs/2506.00448)
_2025 · forgeScore 0.55 · research-paper_

Research paper introducing fact-controlled and natural hallucination datasets for evaluating hallucination detection methods in medical dialogue summarization, showing general-domain detectors fail on clinical content.

### [Semantic Energy: Detecting LLM Hallucination Beyond Entropy](https://arxiv.org/abs/2508.14496)
_2025 · forgeScore 0.55 · analysis_

Introduces Semantic Energy, an uncertainty estimation framework that operates on penultimate-layer logits and combines semantic clustering with a Boltzmann-inspired energy distribution to detect LLM hallucinations more reliably than semantic entropy.

### [Model-Agnostic Correctness Assessment for LLM-Generated Code via Dynamic Internal Representation Selection](https://arxiv.org/abs/2510.02934)
_2025 · forgeScore 0.55 · research-paper_

Introduces AUTOPROBE, a model-agnostic method that uses attention-based dynamic selection of LLM internal representations to assess correctness (compilability, functionality, security) of generated code.

## 2024 (18)

### [HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection](https://arxiv.org/pdf/2409.17504)
_2024 · forgeScore 0.72 · research-paper_

NeurIPS 2024 Spotlight paper introducing HaloScope, a framework that leverages unlabeled LLM generations to train a binary truthfulness classifier for hallucination detection without requiring human annotations.

### [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/pdf/2406.15927)
_2024 · forgeScore 0.72 · research-paper_

Introduces Semantic Entropy Probes (SEPs), a cheap hallucination detection method that approximates semantic entropy from a single generation's hidden states, eliminating the need for multiple samples.

### [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/pdf/2402.03744)
_2024 · forgeScore 0.70 · research-paper_

Proposes INSIDE, a method using LLMs' internal hidden states and an EigenScore metric over response covariance eigenvalues to detect hallucinations via semantic self-consistency.

### [Your Mixture-of-Experts LLM Is Secretly an Embedding Model For Free](https://arxiv.org/abs/2410.10814)
_2024 · forgeScore 0.70 · analysis_

Research paper showing that Mixture-of-Experts LLM router weights can serve as off-the-shelf embeddings, and combining them with hidden states (MoEE) outperforms either alone on MTEB benchmarks.

### [Intrinsic Dimension Correlation: uncovering nonlinear connections in multimodal representations](https://arxiv.org/pdf/2406.15812)
_2024 · forgeScore 0.70 · research-paper_

Research paper introducing Intrinsic Dimension Correlation, a metric that quantifies nonlinear correlations between high-dimensional manifolds, demonstrated on multimodal vision-language embeddings.

### [Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation](https://arxiv.org/abs/2410.13640)
_2024 · forgeScore 0.70 · research-paper_

Proposes Chain-of-Embedding (CoE), using progressive hidden states in LLM latent space to perform label-free, output-free self-evaluation of response correctness in real time.

### [ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](https://arxiv.org/pdf/2410.11414)
_2024 · forgeScore 0.70 · research-paper_

Proposes ReDeEP, a mechanistic-interpretability method that detects RAG hallucinations by decoupling LLMs' use of parametric knowledge (Knowledge FFNs) from external context (Copying Heads), plus AARF for mitigation.

### [Cycles of Thought: Measuring LLM Confidence through Stable Explanations](https://arxiv.org/pdf/2406.03441)
_2024 · forgeScore 0.70 · analysis_

Research paper proposing a framework to measure LLM uncertainty by treating model+explanation pairs as test-time classifiers and computing a posterior answer distribution via explanation entailment.

### [Hallucination Index: An Image Quality Metric for Generative Reconstruction Models](https://arxiv.org/abs/2407.12780)
_2024 · forgeScore 0.70 · research-paper_

Proposes a Hallucination Index metric based on Hellinger distance between reconstructed and reference distributions to quantify hallucinations in diffusion-based medical image reconstruction.

### [Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models](https://arxiv.org/pdf/2401.01301)
_2024 · forgeScore 0.70 · research-paper_

Empirical study documenting pervasive legal hallucinations (58-88%) in major LLMs when answering verifiable questions about federal court cases, with a typology and warnings against unsupervised legal use.

### [Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States](https://arxiv.org/abs/2402.09733)
_2024 · forgeScore 0.70 · research-paper_

An empirical study examining whether LLMs' hidden states differ when producing hallucinated versus genuine answers, using the LLaMA family, and showing that hidden representations can guide hallucination mitigation.

### [States Hidden in Hidden States: LLMs Emerge Discrete State Representations Implicitly](https://arxiv.org/abs/2407.11421)
_2024 · forgeScore 0.70 · analysis_

Research paper uncovering that LLMs form Implicit Discrete State Representations (IDSRs) in hidden states to perform multi-step arithmetic internally without chain-of-thought.

### [LoRA-Ensemble: Efficient Uncertainty Modelling for Self-Attention Networks](https://arxiv.org/pdf/2405.14438)
_2024 · forgeScore 0.70 · research-paper_

Introduces LoRA-Ensemble, a parameter-efficient implicit ensembling method for self-attention networks that uses shared pre-trained weights with individual low-rank attention projections to achieve calibrated uncertainty estimates.

### [CSS: Contrastive Semantic Similarity for Uncertainty Quantification of LLMs](https://arxiv.org/abs/2406.03158)
_2024 · forgeScore 0.70 · research-paper_

Proposes Contrastive Semantic Similarity (CSS), a CLIP-based feature extraction method for measuring semantic dispersion to quantify uncertainty in LLM generations and reject unreliable outputs.

### [Position: Topological Deep Learning is the New Frontier for Relational Learning](https://arxiv.org/pdf/2402.08871)
_2024 · forgeScore 0.62 · analysis_

Position paper arguing that topological deep learning (TDL) is the new frontier for relational learning, surveying open problems, practical benefits, and theoretical foundations.

### [LLMs Will Always Hallucinate, and We Need to Live With This](https://arxiv.org/pdf/2409.05746)
_2024 · forgeScore 0.60 · research-paper_

An arXiv paper arguing that hallucinations in LLMs are mathematically inevitable, grounded in Gödel's incompleteness and the undecidability of problems like halting and acceptance.

### [Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs](https://arxiv.org/pdf/2402.10612)
_2024 · forgeScore 0.60 · research-paper_

Rowen is an adaptive RAG framework that detects LLM hallucinations via cross-lingual/cross-model semantic consistency and triggers external retrieval only when uncertainty is high.

### [Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation](https://arxiv.org/abs/2402.09267)
_2024 · forgeScore 0.60 · analysis_

Paper proposing Self-Alignment for Factuality, using an LLM's self-evaluation to generate training signals (via SK-Tuning and DPO) that reduce hallucinations without human factuality annotations.

## 2023 and earlier (27)

### [Zigzag Persistence](https://arxiv.org/abs/0812.0197)
_2008 · forgeScore 0.78 · research-paper_

Foundational paper introducing zigzag persistence, a generalization of persistent homology for studying topological features across families of spaces or point clouds.

### [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/pdf/2311.05232)
_2023 · forgeScore 0.72 · analysis_

A comprehensive survey on hallucination in large language models covering taxonomy, contributing factors, detection methods, benchmarks, and mitigation strategies.

### [A Survey on Uncertainty Quantification Methods for Deep Learning](https://arxiv.org/pdf/2302.13425)
_2023 · forgeScore 0.72 · research-paper_

A survey paper taxonomizing uncertainty quantification methods for deep neural networks by uncertainty source (data vs model), covering applications to active learning, OOD robustness, and deep RL.

### [On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology](https://arxiv.org/pdf/2302.02941)
_2023 · forgeScore 0.72 · research-paper_

ICML 2023 theoretical paper proving how width, depth, and graph topology each affect over-squashing in Message Passing Neural Networks, justifying graph rewiring methods.

### [www.cambridge.org](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/BB0DA0F0EBD79809C563AF80B555A23C/S0962492914000051a.pdf/div-class-title-topological-pattern-recognition-for-point-cloud-data-a-href-fn01-ref-type-fn-a-div.pdf)
_? · forgeScore 0.72 · research-paper_

Survey paper on topological pattern recognition methods for point cloud data, covering persistent homology, Mapper, and related techniques for extracting shape from data.

### [Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction](https://doi.org/10.48550/arxiv.2604.17716)
_? · forgeScore 0.72 · research-paper_

An arXiv paper validating a three-tier screen (Valid/Indeterminate/Invalid) that classifies LLM confidence signals and predicts selective prediction performance across 20 frontier models.

### [Calibrate and Debias Layer-wise Sampling for Graph Convolutional Networks](https://arxiv.org/abs/2206.00583)
_2022 · forgeScore 0.70 · research-paper_

An arXiv paper that revisits layer-wise sampling in GCN training from a matrix approximation view, proposing calibrated sampling probabilities and a debiasing algorithm to reduce estimation variance.

### [Learning Topology-Preserving Data Representations](https://arxiv.org/abs/2302.00136)
_2023 · forgeScore 0.70 · research-paper_

ICLR 2023 paper introducing RTD-AE, an autoencoder that preserves topological features of data manifolds by minimizing Representation Topology Divergence as a differentiable loss.

### [A Geometric Method for Improved Uncertainty Estimation in Real-time](https://arxiv.org/abs/2206.11562)
_2022 · forgeScore 0.70 · research-paper_

A 2022 arXiv paper proposing a geometric-distance-based approach for post-hoc uncertainty calibration of ML classifiers, usable in near real-time.

### [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063)
_2023 · forgeScore 0.70 · research-paper_

Empirical study benchmarking black-box methods (prompting, sampling, aggregation) for eliciting verbalized confidence from LLMs, finding they tend to be overconfident but can be partially mitigated.

### [Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts](https://arxiv.org/pdf/2306.04723)
_2023 · forgeScore 0.70 · research-paper_

Proposes using the intrinsic dimensionality of the embedding manifold (~9 for human text, ~1.5 lower for AI text) as a robust, model-agnostic detector of AI-generated text.

### [A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation](https://arxiv.org/pdf/2307.03987)
_2023 · forgeScore 0.70 · analysis_

Research paper proposing an active detection and mitigation approach for LLM hallucinations by validating low-confidence generations using logit values, reducing GPT-3.5 hallucination rates from 47.5% to 14.5%.

### [Characterizing the Shape of Activation Space in Deep Neural Networks](https://arxiv.org/pdf/1901.09496)
_2019 · forgeScore 0.70 · research-paper_

Research paper introducing persistent homology over neural network activation graphs to characterize learned representations and offer a topological explanation for adversarial examples.

### [Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models](https://arxiv.org/pdf/2309.01219)
_2023 · forgeScore 0.70 · research-paper_

A survey paper on hallucination in large language models, presenting taxonomies, evaluation benchmarks, and mitigation approaches.

### [Reduced-order modeling of advection-dominated systems with recurrent neural networks and convolutional autoencoders](https://arxiv.org/pdf/2002.00470)
_2020 · forgeScore 0.70 · research-paper_

Research paper showing that convolutional autoencoders combined with RNNs outperform POD-Galerkin for reduced-order modeling of advection-dominated PDEs, achieving accurate shock profile reproduction in very low-dimensional latent spaces.

### [LUMÁWIG: An Efficient Algorithm for Dimension Zero Bottleneck Distance Computation in Topological Data Analysis](https://arxiv.org/abs/2010.00371)
_2020 · forgeScore 0.70 · research-paper_

Introduces LUMÁWIG, a near-linear-time algorithm for computing dimension-zero bottleneck distance between persistence diagrams in topological data analysis, with an application to classification features.

### [DTM-based Filtrations](https://arxiv.org/abs/1811.04757)
_2018 · forgeScore 0.70 · research-paper_

Introduces DTM-filtrations, a noise- and outlier-robust family of filtrations for persistent homology built on distance-to-measure functions over Euclidean point clouds.

### [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187)
_2023 · forgeScore 0.70 · analysis_

Research paper proposing confidence and uncertainty measures for black-box LLMs, showing that semantic dispersion reliably predicts response quality in natural language generation.

### [Discrete Morse Theory for Computing Zigzag Persistence](https://arxiv.org/abs/1807.05172)
_2018 · forgeScore 0.70 · analysis_

Research paper introducing a discrete Morse theory framework as preprocessing to efficiently compute zigzag persistent homology via Morse reductions of filtration complexes.

### [Which models are innately best at uncertainty estimation?](https://arxiv.org/abs/2206.02152)
_2022 · forgeScore 0.60 · analysis_

Empirical study of selective prediction and uncertainty estimation across 484 pretrained ImageNet classifiers, finding that ViT and distillation-based training yield the best uncertainty estimation performance.

### [Calibration of Machine Learning Classifiers for Probability of Default Modelling](https://arxiv.org/abs/1710.08901)
_2017 · forgeScore 0.60 · analysis_

Benchmarks calibration techniques (Platt Scaling, Isotonic Regression) across Logistic Regression, Random Forest, and Gradient Boosting on 18 credit scoring datasets, showing Isotonic Regression re-calibration improves long-term calibration on time-series data.

### [Managing Uncertainty in Rule Based Cognitive Models](https://arxiv.org/abs/1304.1083)
_2013 · forgeScore 0.55 · research-paper_

An arXiv paper on modeling propagation of uncertainty in rule-based cognitive reasoning using certainty factors combined via min/max and Heckerman's modified technique.

### [Computing Zigzag Persistent Cohomology](https://arxiv.org/abs/1608.06039)
_2016 · forgeScore 0.55 · analysis_

Research paper presenting an efficient zigzag persistent cohomology algorithm with experimental comparisons to standard persistent homology and sparse filtration methods.

### [Efficient Planning of Multi-Robot Collective Transport using Graph Reinforcement Learning with Higher Order Topological Abstraction](https://doi.org/10.1109/icra48891.2023.10161517)
_? · forgeScore 0.55 · research-paper_

Research paper on multi-robot collective transport task allocation using graph reinforcement learning augmented with topological descriptors from persistent homology for better scaling to larger problem instances.

### [Connectivity in fMRI: Blind Spots and Breakthroughs](https://doi.org/10.1109/tmi.2018.2831261)
_? · forgeScore 0.55 · analysis_

A review paper surveying limitations of existing functional brain network analysis methods in fMRI and introducing emerging approaches like stochastic block models, exponential random graph models, persistent homology, and time-varying connectivity analysis.

### [Topology and geometry cannot be measured by an operator measurement in quantum gravity](https://arxiv.org/pdf/1605.06166)
_2016 · forgeScore 0.55 · research-paper_

A 2016 theoretical physics paper arguing that in quantum gravity, superpositions of coherent states with trivial topology can yield classical limits with different spacetime topologies, implying topology and geometry are not operator-measurable observables.

### [Topological phenomena at topological defects](https://arxiv.org/abs/2208.05082)
_2022 · forgeScore 0.55 · analysis_

A perspective review surveying the interaction between band topology and topological lattice defects in materials, covering phenomena like topological pumping, embedded topological phases, synthetic dimensions, and non-Hermitian skin effects.

