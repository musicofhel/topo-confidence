# Baseline Notes — Topological AUROC Task

## Goal

Predict whether Qwen2.5-1.5B-Instruct answered a MATH-500 problem correctly,
using topological features extracted from cached hidden states.

**Baseline AUROC**: 0.713 (7 features, 50-fold StratifiedKFold, LogisticRegression balanced)
**Target**: 0.75+

## Dataset

- **500 MATH-500 problems**, 57 correct (11.4%). Heavily imbalanced.
- **Token trajectories**: Per-token last-layer hidden states. Each problem has
  (n_tokens, 1536) where n_tokens ranges 30-350. Cached at:
  `~/topo-confidence/data/experiment1_v2/trajectories.npz`
- **Layer states**: Mean hidden state per layer, shape (500, 29, 1536). 29 layers
  including embedding layer. Cached at:
  `~/att-docs/data/transformer/math500_hidden_states_aligned.npz`

## Current 7-Feature Baseline

All features are extracted from token trajectories (last layer only):

1. **H0_persistence_entropy** — Shannon entropy of H0 (connected component) lifetimes.
   Measures complexity of cluster structure.
2. **H0_total_persistence** — Sum of all H0 lifetimes. Measures total geometric spread.
3. **H0_n_features** — Number of H0 features. Number of clusters at birth.
4. **H1_max_lifetime** — Longest H1 (loop) lifetime. Measures largest topological hole.
5. **H1_persistence_entropy** — Shannon entropy of H1 lifetimes. Loop complexity.
6. **H1_n_features** — Number of H1 features. Number of loops detected.
7. **bridge_silhouette** — Silhouette coefficient of position-0 token in k=2 KMeans
   clustering of the PCA-reduced point cloud. Best single structural feature.

Pipeline: PCA(30 components) → subsample(100 points) → ripser(max_dim=1) → features.

## What Was Tried and FAILED

### Shuffled-token null hypothesis (z-scores)
**DEGENERATE**. Permuting rows of the (n_tokens × 1536) matrix preserves the overall
point cloud geometry because token embeddings are not spatially ordered — they're
unordered points. Both H0 and H1 z-scores were identically 0.0 for all 500 problems.
**Lesson**: Need a geometry-destroying null, not a row-shuffling null. Consider:
- Gaussian null (sample from N(mu, Sigma) fitted to the point cloud)
- Toroidal bootstrap (resample birth-death pairs from a fitted distribution)
- Dimension-permutation null (independently permute each column)

### H2 features (cavities/voids)
**SPARSE AND HURT**. At subsample=100 in 30D PCA space, H2 features are mostly zero
(45% of problems have H2_n_features=0). Adding H2 features dropped AUROC from 0.713
to 0.691. H2 computation at max_dim=2 is also ~3x slower.
**Lesson**: H2 needs more points or lower PCA dimensions to be informative.

### Topological sensitivity (perturbation slope)
**NO PREDICTIVE POWER**. Univariate AUROC = 0.511 (chance). The OLS slope of H1
total persistence vs. Gaussian noise epsilon showed no correlation with correctness.
**Lesson**: Sensitivity to random noise is not the right stability measure here.

## Key Unexploited Data: Layer States

`layer_states` has shape (500, 29, 1536) — 29 points per problem in 1536D,
representing how the hidden state evolves through the transformer layers.
**This data has never been used for correctness prediction.**

Ideas for exploiting it:
- **Cross-layer PH**: Treat the 29 layer means as a point cloud, compute PH.
  Topology of the layer-space trajectory.
- **Crystallization**: Measure when the representation "stabilizes" — e.g.,
  pairwise distance matrix between consecutive layers, find the layer where
  distances drop below a threshold.
- **Layer-wise entropy**: Compute persistence entropy at each layer (using
  subsets of token trajectories), track how it changes.
- **Dissolution**: Inverse of crystallization — when does structure break down?
  H0/H1 features of early vs. late layer subsets.
- **Wasserstein distance**: Between persistence diagrams of adjacent layers.
  Measures topological change rate through the network.

## Research Directions

### Feature engineering
- Persistence landscapes (functional summaries of persistence diagrams)
- Wasserstein distances between diagrams
- Bottleneck distance
- Betti curves (number of features alive at each filtration value)
- Persistence images (vectorized persistence diagrams)

### Parameter tuning
- PCA dimensions: try 10, 20, 50
- Subsample size: try 50, 200
- Different distance metrics for ripser

### Feature selection
- Forward/backward feature selection
- L1 regularization to find optimal sparse subset
- Remove features that hurt (ablation study)

### Better classifiers
- Note: the grader uses LogisticRegression with class_weight="balanced".
  You cannot change the classifier. Focus on better features.

## Key Papers

- Vejdemo-Johansson & Mukherjee 2018 — Statistical testing for PH significance
- Cohen-Steiner et al. 2007 — Stability of persistence diagrams
- Adams et al. 2017 — Persistence images
- Bubenik 2015 — Persistence landscapes
