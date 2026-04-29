# topo-confidence — pathway-10 new directions · supporting literature

_Generated 2026-04-23T13:58:52.016Z from link-forge Neo4j graph (ingested since 2026-04-22)._

This is a themed cut of the pathway-9 literature sweep, organized around concrete
research directions left open by `pathway9/SUMMARY.md` and `HANDOFF_v3.md`. Each theme
lists the 5–8 strongest hits; use as anchors when drafting the pathway-10 plan doc.

Refresh: `cd ~/link-forge && node scripts/build-new-directions.mjs > ~/topo-confidence/pathway10-papers.md`

## Chain-of-Embedding & layer-wise trajectories (2)

_CoE is the pathway-10 headline. Look for variants, extensions, and diagnostics of layer-trajectory methods._

- **[Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation](https://arxiv.org/abs/2410.13640)** — _2024 · forgeScore 0.70 · research-paper_
  Proposes Chain-of-Embedding (CoE), using progressive hidden states in LLM latent space to perform label-free, output-free self-evaluation of response correctness in real time.
- **[Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](https://arxiv.org/abs/2604.15741)** — _2026 · forgeScore 0.70 · research-paper_
  ACL 2026 paper introducing SIVR, a supervised hallucination detection framework for LLMs that uses token-wise, layer-wise variance of hidden states to estimate uncertainty.

## Spectral / eigenvalue hidden-state methods (5)

_EigenScore / EigenTrack / D²HScore show a second family of cheap, white-box confidence signals — compare head-to-head with CoE._

- **[INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/pdf/2402.03744)** — _2024 · forgeScore 0.70 · research-paper_
  Proposes INSIDE, a method using LLMs' internal hidden states and an EigenScore metric over response covariance eigenvalues to detect hallucinations via semantic self-consistency.
- **[EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs](https://arxiv.org/abs/2509.15735)** — _2025 · forgeScore 0.70 · research-paper_
  EigenTrack is a white-box real-time detector that uses spectral statistics (entropy, eigenvalue gaps, KL divergence) of hidden activation covariances to flag hallucination and OOD drift in LLMs/VLMs in a single forward pass.
- **[D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/pdf/2509.11569)** — _2025 · forgeScore 0.60 · research-paper_
  A training-free, label-free framework (D²HScore) that detects LLM hallucinations by measuring intra-layer semantic dispersion and inter-layer representational drift of attention-selected tokens.
- **[EigenScore: OOD Detection using Covariance in Diffusion Models](https://arxiv.org/abs/2510.07206)** — _2025 · forgeScore 0.60 · analysis_
  Proposes EigenScore, an OOD detection method that uses the eigenvalue spectrum of posterior covariance from diffusion models, achieving SOTA AUROC with a Jacobian-free subspace iteration estimator.
- **[D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/abs/2509.11569)** — _2025 · forgeScore 0.55 · analysis_
  Proposes D²HScore, a training-free hallucination detection method for LLMs that combines intra-layer token dispersion and inter-layer semantic drift of attention-selected tokens.

## Attention-based correctness & hallucination detection (6)

_Lookback lens / AUTOPROBE / ReDeEP etc. — attention maps are orthogonal to residual-stream signals. Candidate for ensembling with CoE._

- **[SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/pdf/2404.14469)** — _2024 · forgeScore 0.72 · analysis_
  Research paper introducing SnapKV, a fine-tuning-free method that compresses LLM KV caches by selecting clustered important KV positions per attention head, achieving 3.6x faster generation and 8.2x better memory efficiency on 16K-token inputs.
- **[Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/pdf/2306.03341)** — _2023 · forgeScore 0.72 · research-paper_
  Research paper introducing Inference-Time Intervention (ITI), a technique that shifts LLM activations across select attention heads during inference to improve truthfulness, nearly doubling Alpaca's TruthfulQA score with only a few hundred examples.
- **[Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](https://arxiv.org/abs/2604.04863)** — _2026 · forgeScore 0.70 · research-paper_
  A CVPR 2026 paper introducing a patch-level hallucination detection framework for large vision-language models that analyzes fine-grained token-level attention patterns to achieve up to 90% accuracy in detecting hallucinated tokens.
- **[ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](https://arxiv.org/pdf/2410.11414)** — _2024 · forgeScore 0.70 · research-paper_
  Proposes ReDeEP, a mechanistic-interpretability method that detects RAG hallucinations by decoupling LLMs' use of parametric knowledge (Knowledge FFNs) from external context (Copying Heads), plus AARF for mitigation.
- **[D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/pdf/2509.11569)** — _2025 · forgeScore 0.60 · research-paper_
  A training-free, label-free framework (D²HScore) that detects LLM hallucinations by measuring intra-layer semantic dispersion and inter-layer representational drift of attention-selected tokens.
- **[D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/abs/2509.11569)** — _2025 · forgeScore 0.55 · analysis_
  Proposes D²HScore, a training-free hallucination detection method for LLMs that combines intra-layer token dispersion and inter-layer semantic drift of attention-selected tokens.

## Mechanistic interp / probing / sparse autoencoders (5)

_If CoE works because certain directions encode correctness, SAE features and probe targets should localize it._

- **[Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/pdf/2309.08600)** — _2023 · forgeScore 0.78 · research-paper_
  Foundational paper showing that sparse autoencoders trained on language model activations recover interpretable, monosemantic features that resolve superposition.
- **[Mechanistic Interpretability for AI Safety -- A Review](https://arxiv.org/pdf/2404.14082)** — _2024 · forgeScore 0.72 · research-paper_
  A review paper surveying mechanistic interpretability methods for reverse-engineering neural networks into human-understandable algorithms to advance AI safety and alignment.
- **[How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)** — _2024 · forgeScore 0.72 · tutorial_
  A tutorial paper on activation patching, a mechanistic interpretability technique, covering application methods, result interpretation, metric choices, and common pitfalls when investigating neural network circuits.
- **[ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](https://arxiv.org/pdf/2410.11414)** — _2024 · forgeScore 0.70 · research-paper_
  Proposes ReDeEP, a mechanistic-interpretability method that detects RAG hallucinations by decoupling LLMs' use of parametric knowledge (Knowledge FFNs) from external context (Copying Heads), plus AARF for mitigation.
- **[Weakly Supervised Distillation of Hallucination Signals into Transformer Representations](https://arxiv.org/abs/2604.06277)** — _2026 · forgeScore 0.55 · analysis_
  A weakly supervised framework that distills hallucination detection signals into transformer hidden states, enabling internal detection via lightweight probing classifiers without external verification at inference.

## Semantic entropy & information-theoretic uncertainty (5)

_These are the current public-facing benchmarks to beat. Need head-to-head comparisons on MATH-500 / BBH / GSM8K._

- **[Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/pdf/2406.15927)** — _2024 · forgeScore 0.72 · research-paper_
  Introduces Semantic Entropy Probes (SEPs), a cheap hallucination detection method that approximates semantic entropy from a single generation's hidden states, eliminating the need for multiple samples.
- **[www.nature.com](https://www.nature.com/articles/s41586-024-07421-0.pdf)** — _? · forgeScore 0.70 · research-paper_
  Nature paper by Farquhar, Kossen, Kuhn, and Gal introducing semantic entropy as a method for detecting hallucinations (confabulations) in large language models.
- **[Detecting AI Hallucinations in Finance: An Information-Theoretic Method Cuts Hallucination Rate by 92%](https://arxiv.org/abs/2512.03107)** — _2025 · forgeScore 0.70 · research-paper_
  ECLIPSE is an information-theoretic framework that detects LLM hallucinations in financial QA by measuring the mismatch between semantic entropy and evidence capacity, achieving 0.89 ROC AUC versus 0.50 for entropy-only baselines.
- **[Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration](https://arxiv.org/abs/2603.22812)** — _2026 · forgeScore 0.60 · research-paper_
  Proposes an adaptive Bayesian framework for semantic entropy that dynamically adjusts LLM sampling budgets to detect hallucinations more efficiently, achieving ~50% fewer samples and +12.6% AUROC over fixed-budget baselines.
- **[Semantic Energy: Detecting LLM Hallucination Beyond Entropy](https://arxiv.org/abs/2508.14496)** — _2025 · forgeScore 0.55 · analysis_
  Introduces Semantic Energy, an uncertainty estimation framework that operates on penultimate-layer logits and combines semantic clustering with a Boltzmann-inspired energy distribution to detect LLM hallucinations more reliably than semantic entropy.

## Calibration, selective prediction, abstention (8)

_AUROC alone isn't publishable. ECE / risk-coverage / AURC are the standard metrics for a confidence paper._

- **[Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction](https://doi.org/10.48550/arxiv.2604.17716)** — _? · forgeScore 0.72 · research-paper_
  An arXiv paper validating a three-tier screen (Valid/Indeterminate/Invalid) that classifies LLM confidence signals and predicts selective prediction performance across 20 frontier models.
- **[Entropy Alone is Insufficient for Safe Selective Prediction in LLMs](https://arxiv.org/pdf/2603.21172)** — _2026 · forgeScore 0.70 · research-paper_
  Research paper showing entropy-based uncertainty alone fails for safe LLM selective prediction, and that combining entropy with a correctness probe improves risk-coverage trade-offs across QA benchmarks.
- **[Calibrate and Debias Layer-wise Sampling for Graph Convolutional Networks](https://arxiv.org/abs/2206.00583)** — _2022 · forgeScore 0.70 · research-paper_
  An arXiv paper that revisits layer-wise sampling in GCN training from a matrix approximation view, proposing calibrated sampling probabilities and a debiasing algorithm to reduce estimation variance.
- **[ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks](https://arxiv.org/abs/2508.16889)** — _2025 · forgeScore 0.70 · research-paper_
  A benchmark (ObjexMT) testing whether LLM judges can recover hidden conversational objectives from multi-turn jailbreak transcripts and accurately self-report confidence, evaluated across six frontier models.
- **[Calibrating LLM Confidence by Probing Perturbed Representation Stability](https://arxiv.org/pdf/2505.21772)** — _2025 · forgeScore 0.70 · research-paper_
  Introduces CCPS, a method for calibrating LLM confidence by applying adversarial perturbations to final hidden states and training a lightweight classifier to predict answer correctness, achieving ~55% reduction in Expected Calibration Error.
- **[Evaluating and Calibrating LLM Confidence on Questions with Multiple Correct Answers](https://arxiv.org/abs/2602.07842)** — _2026 · forgeScore 0.70 · research-paper_
  Introduces MACE benchmark and Semantic Confidence Aggregation (SCA) to fix LLM confidence calibration failures on questions with multiple valid answers.
- **[Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation](https://arxiv.org/abs/2502.11620)** — _2025 · forgeScore 0.60 · analysis_
  Research paper adapting entropy and mutual-information uncertainty estimation techniques from NLG to LLM code generation, using symbolic-execution-based semantic equivalence checks to predict correctness and drive an abstention policy.
- **[Which models are innately best at uncertainty estimation?](https://arxiv.org/abs/2206.02152)** — _2022 · forgeScore 0.60 · analysis_
  Empirical study of selective prediction and uncertainty estimation across 484 pretrained ImageNet classifiers, finding that ViT and distillation-based training yield the best uncertainty estimation performance.

## Cross-domain / cross-model / cross-scale transfer (7)

_The whole pathway-10 pitch is domain-invariance. Need to survey what currently transfers and how authors measure it._

- **[Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts](https://arxiv.org/pdf/2306.04723)** — _2023 · forgeScore 0.70 · research-paper_
  Proposes using the intrinsic dimensionality of the embedding manifold (~9 for human text, ~1.5 lower for AI text) as a robust, model-agnostic detector of AI-generated text.
- **[EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs](https://arxiv.org/abs/2509.15735)** — _2025 · forgeScore 0.70 · research-paper_
  EigenTrack is a white-box real-time detector that uses spectral statistics (entropy, eigenvalue gaps, KL divergence) of hidden activation covariances to flag hallucination and OOD drift in LLMs/VLMs in a single forward pass.
- **[Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs](https://arxiv.org/pdf/2402.10612)** — _2024 · forgeScore 0.60 · research-paper_
  Rowen is an adaptive RAG framework that detects LLM hallucinations via cross-lingual/cross-model semantic consistency and triggers external retrieval only when uncertainty is high.
- **[Predicting concentration levels of air pollutants by transfer learning and recurrent neural network](https://arxiv.org/abs/2502.01654)** — _2025 · forgeScore 0.55 · research-paper_
  Research paper applying LSTM RNNs with transfer learning to predict air pollutant concentrations in Macau, using 12+ years of meteorological and air quality station data.
- **[Roweis Discriminant Analysis: A Generalized Subspace Learning Method](https://arxiv.org/abs/1910.05437)** — _2019 · forgeScore 0.55 · research-paper_
  Research paper introducing Roweis Discriminant Analysis (RDA), a generalized subspace learning framework that unifies PCA, Supervised PCA, and Fisher Discriminant Analysis as special cases, including kernel and dual variants.
- **[Model-Agnostic Correctness Assessment for LLM-Generated Code via Dynamic Internal Representation Selection](https://arxiv.org/abs/2510.02934)** — _2025 · forgeScore 0.55 · research-paper_
  Introduces AUTOPROBE, a model-agnostic method that uses attention-based dynamic selection of LLM internal representations to assess correctness (compilability, functionality, security) of generated code.
- **[Bigraded Betti numbers and generalized persistence diagrams](https://arxiv.org/abs/2111.02551)** — _2021 · forgeScore 0.55 · research-paper_
  Research paper showing that bigraded Betti numbers of 2-parameter persistence modules can be recovered by counting corner points of subsets derived from Möbius inversion of the generalized rank invariant.

## Alternative topological tools (zigzag / DTM / Ricci / Mapper) (8)

_If pathway 7 non-Euclidean PH was NO-GO and pathway 8 layer-wise PH fails to transfer, a targeted tool swap is the last honest PH shot._

- **[Zigzag Persistence](https://arxiv.org/abs/0812.0197)** — _2008 · forgeScore 0.78 · research-paper_
  Foundational paper introducing zigzag persistence, a generalization of persistent homology for studying topological features across families of spaces or point clouds.
- **[Topology-Aware Layer Pruning for Large Vision-Language Models](https://arxiv.org/abs/2604.16502)** — _2026 · forgeScore 0.72 · research-paper_
  A topology-aware layer pruning framework for Large Vision-Language Models that uses zigzag persistent homology on simplicial complexes of hidden states to preserve transition-critical layers during compression.
- **[DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature](https://arxiv.org/pdf/2301.11305)** — _2023 · forgeScore 0.70 · research-paper_
  ICML 2023 paper introducing DetectGPT, a zero-shot method for detecting LLM-generated text using log probability curvature without training a classifier.
- **[Discrete Morse Theory for Computing Zigzag Persistence](https://arxiv.org/abs/1807.05172)** — _2018 · forgeScore 0.70 · analysis_
  Research paper introducing a discrete Morse theory framework as preprocessing to efficiently compute zigzag persistent homology via Morse reductions of filtration complexes.
- **[An Introduction to Topological Data Analysis Ball Mapper in Python](https://arxiv.org/abs/2505.03022)** — _2025 · forgeScore 0.70 · tutorial_
  An introductory guide with Python code for using Topological Data Analysis Ball Mapper (TDABM) to visualize multivariate datasets without information loss.
- **[DTM-based Filtrations](https://arxiv.org/abs/1811.04757)** — _2018 · forgeScore 0.70 · research-paper_
  Introduces DTM-filtrations, a noise- and outlier-robust family of filtrations for persistent homology built on distance-to-measure functions over Euclidean point clouds.
- **[Statistical detection of format dialects using the weighted Dowker complex](https://arxiv.org/abs/2201.08267)** — _2022 · forgeScore 0.70 · research-paper_
  An arXiv paper presenting a probabilistic model using weighted Dowker complexes to statistically detect file format dialects based on parser behavior messages.
- **[Self-supervised edge features for improved Graph Neural Network training](https://arxiv.org/pdf/2007.04777)** — _2020 · forgeScore 0.70 · research-paper_
  Research paper presenting a framework for generating edge features via self-supervised and unsupervised learning, combined with Forman-Ricci curvature, to improve GNN node classification on biological datasets.

## Intrinsic dimension & manifold geometry (8)

_TwoNN / intrinsic dimension is cheap, interpretable, and already lives in pathway 8. Good supporting axis for a CoE paper's geometry section._

- **[The geometry of hidden representations of large transformer models](https://arxiv.org/pdf/2302.00294)** — _2023 · forgeScore 0.72 · analysis_
  Analyzes how hidden representations in large transformers evolve across layers by measuring intrinsic dimension, showing semantic content peaks at intermediate layers where the ID profile hits a relative minimum.
- **[Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts](https://arxiv.org/pdf/2306.04723)** — _2023 · forgeScore 0.70 · research-paper_
  Proposes using the intrinsic dimensionality of the embedding manifold (~9 for human text, ~1.5 lower for AI text) as a robust, model-agnostic detector of AI-generated text.
- **[Intrinsic Dimension Correlation: uncovering nonlinear connections in multimodal representations](https://arxiv.org/pdf/2406.15812)** — _2024 · forgeScore 0.70 · research-paper_
  Research paper introducing Intrinsic Dimension Correlation, a metric that quantifies nonlinear correlations between high-dimensional manifolds, demonstrated on multimodal vision-language embeddings.
- **[Latent Semantic Manifolds in Large Language Models](https://arxiv.org/abs/2603.22301)** — _2026 · forgeScore 0.62 · analysis_
  A mathematical framework treating LLM hidden states as points on a Riemannian latent semantic manifold, proving rate-distortion and linear volume scaling laws for vocabulary-induced expressibility gaps and validating them across six transformer models.
- **[Dilatation structures in sub-riemannian geometry](https://arxiv.org/abs/0708.4298)** — _2007 · forgeScore 0.55 · analysis_
  Mathematical paper proving that regular sub-riemannian manifolds admit dilatation structures, giving an intrinsic treatment to sub-riemannian geometry.
- **[Deep neural networks architectures from the perspective of manifold learning](https://arxiv.org/pdf/2306.03406)** — _2023 · forgeScore 0.55 · analysis_
  Research paper analyzing deep neural network architectures (CNNs and Transformers) through the lens of manifold learning, topological data analysis, and persistent homological fractal dimension to understand internal representations across layers.
- **[Non-Euclidean elasticity for rods and almost isometric embeddings of geodesic tubes](https://arxiv.org/abs/2512.00643)** — _2025 · forgeScore 0.55 · analysis_
  Research paper proving a Γ-convergence result for elastic energy of maps between Riemannian manifolds on thin geodesic tubes, generalizing Mora-Müller's Euclidean thin rod theory to the non-Euclidean setting.
- **[Betti and Tachibana numbers](https://arxiv.org/abs/1306.6800)** — _2013 · forgeScore 0.55 · analysis_
  An arXiv paper presenting a classification of differential forms on Riemannian manifolds and defining Tachibana numbers as an analog of Betti numbers, showing connections between them.

## Length / dispersion / drift confounds (8)

_Pathway 9 lost 0.05 AUROC to a raw length confound. Other papers hitting hidden-state dispersion must face the same issue — learn from them._

- **[CSS: Contrastive Semantic Similarity for Uncertainty Quantification of LLMs](https://arxiv.org/abs/2406.03158)** — _2024 · forgeScore 0.70 · research-paper_
  Proposes Contrastive Semantic Similarity (CSS), a CLIP-based feature extraction method for measuring semantic dispersion to quantify uncertainty in LLM generations and reject unreliable outputs.
- **[Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187)** — _2023 · forgeScore 0.70 · analysis_
  Research paper proposing confidence and uncertainty measures for black-box LLMs, showing that semantic dispersion reliably predicts response quality in natural language generation.
- **[EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs](https://arxiv.org/abs/2509.15735)** — _2025 · forgeScore 0.70 · research-paper_
  EigenTrack is a white-box real-time detector that uses spectral statistics (entropy, eigenvalue gaps, KL divergence) of hidden activation covariances to flag hallucination and OOD drift in LLMs/VLMs in a single forward pass.
- **[Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](https://arxiv.org/abs/2604.15741)** — _2026 · forgeScore 0.70 · research-paper_
  ACL 2026 paper introducing SIVR, a supervised hallucination detection framework for LLMs that uses token-wise, layer-wise variance of hidden states to estimate uncertainty.
- **[Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](https://arxiv.org/abs/2601.12359)** — _2026 · forgeScore 0.60 · research-paper_
  Proposes ZEDD, a zero-shot framework that detects prompt injection attacks in LLMs by measuring cosine-similarity drift between benign and suspect input embeddings, achieving >93% accuracy across multiple model architectures.
- **[D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/pdf/2509.11569)** — _2025 · forgeScore 0.60 · research-paper_
  A training-free, label-free framework (D²HScore) that detects LLM hallucinations by measuring intra-layer semantic dispersion and inter-layer representational drift of attention-selected tokens.
- **[D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](https://arxiv.org/abs/2509.11569)** — _2025 · forgeScore 0.55 · analysis_
  Proposes D²HScore, a training-free hallucination detection method for LLMs that combines intra-layer token dispersion and inter-layer semantic drift of attention-selected tokens.
- **[Pointer: An Energy-Efficient ReRAM-based Point Cloud Recognition Accelerator with Inter-layer and Intra-layer Optimizations](https://arxiv.org/abs/2410.17782)** — _2024 · forgeScore 0.55 · research-paper_
  Proposes Pointer, a ReRAM-based accelerator for point cloud recognition with inter-layer coordination and intra-layer reordering, achieving 40-393x speedup and 22-163x energy efficiency over prior accelerators.

## Hidden-state steering / representation engineering (2)

_Pathway 3 used steering for accuracy; CoE could be inverted for on-the-fly correction when a low-confidence prefix is detected._

- **[Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/pdf/2306.03341)** — _2023 · forgeScore 0.72 · research-paper_
  Research paper introducing Inference-Time Intervention (ITI), a technique that shifts LLM activations across select attention heads during inference to improve truthfulness, nearly doubling Alpaca's TruthfulQA score with only a few hundred examples.
- **[Brain-Grounded Axes for Reading and Steering LLM States](https://arxiv.org/abs/2512.19399)** — _2025 · forgeScore 0.70 · research-paper_
  Research paper proposing the use of human brain activity (MEG data) as a coordinate system to read and steer LLM internal states via lightweight adapters, without fine-tuning the underlying model.

## Code / math / reasoning-domain correctness signals (3)

_pathway-10 target benchmarks are MATH-500, GSM8K, BBH, HumanEval. Look at what's been reported domain-by-domain._

- **[Eliminating Hallucination-Induced Errors in LLM Code Generation with Functional Clustering](https://arxiv.org/abs/2506.11021)** — _2025 · forgeScore 0.72 · analysis_
  A black-box wrapper that eliminates hallucination-induced errors in LLM code generation by sampling candidate programs, executing them on self-generated tests, and clustering by I/O behavior to produce a tunable confidence score.
- **[Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation](https://arxiv.org/abs/2502.11620)** — _2025 · forgeScore 0.60 · analysis_
  Research paper adapting entropy and mutual-information uncertainty estimation techniques from NLG to LLM code generation, using symbolic-execution-based semantic equivalence checks to predict correctness and drive an abstention policy.
- **[Multicalibration for LLM-based Code Generation](https://arxiv.org/abs/2512.08810)** — _2025 · forgeScore 0.55 · analysis_
  Research paper investigating multicalibration techniques to improve confidence score reliability in LLM-based code generation across multiple function synthesis benchmarks.

