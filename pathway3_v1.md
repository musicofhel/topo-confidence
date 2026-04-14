# Pathway 3: Closing the Steering Generalization Gap (v1)

## What this prompt is

Track A proved that confidence-gated spherical steering works: +26 net gain on train-400, +3 on holdout-100. The mechanism is validated. The problem is generalization. Pathway 3 closes that gap through four stacked interventions:

1. **Expand the contrastive set** — from 46 correct examples to 150+ via temperature sampling.
2. **Extract multi-prototype vectors** — cluster activation differences into K reasoning prototypes instead of collapsing to one mean.
3. **Denoise and multi-layer** — bootstrap averaging, PCA projection, and non-adjacent multi-layer injection.
4. **Soft confidence gating** — replace the brittle hard threshold with sigmoid scaling.

Each intervention addresses a diagnosed root cause of the gap. They are applied cumulatively, evaluated via 5-fold cross-validation (replacing the underpowered single-holdout design), and the final configuration gets a single holdout touch.

**Estimated compute:** ~2 days. Day 1: data expansion + vector extraction + prototypes. Day 2: gating improvements + full evaluation + holdout.

---

## Context from Track A

Track A is **COMPLETE with SKIP_TRACK_B → PROCEED_PATHWAY3**:

- **Train-400:** Targeted steering at τ=0.40, t=0.15, layer 24 → **+26 net gain** (28 wrong→right, 2 right→wrong).
- **Holdout-100:** Same config → **+3 net gain** (14% vs 11%). McNemar p=0.505 (underpowered at n=100).
- **Confidence gating is the key multiplier:** Uniform steering was +1 on holdout. Targeted was +3. The gating concentrates intervention on predicted failures and protects correct answers.
- **Steering vector:** 5-seed averaged mean-difference at layer 24, cosine stability 0.77–0.87. Near-orthogonal to topo-confidence (r=0.031).
- **Bug fixed:** Hook needed dynamic dim handling for 2D/3D hidden states + non-tuple decoder layer output.
- **Model:** Qwen2.5-1.5B-Instruct, 28 layers, hidden_dim=1536, greedy decoding, max_new_tokens=256.
- **GPU:** RTX 2060 Super (8GB VRAM).

### Diagnosed root causes of the gap

1. **Data scarcity (biggest factor):** Only 46 correct examples. Cosine stability 0.77–0.87 means 13–23% of the vector direction is noise. The vector captures "average correct reasoning" from a small, potentially non-representative subset.

2. **Single-vector collapse (second):** Mean-difference collapses all the ways a problem can be "correct" into one direction. A holdout problem that fails for a different reason than the train failures won't respond to this vector.

3. **Brittle gating (third):** The confidence model's holdout predictions differed by 0.9 from Pathway 1's predictions. On train, steered_subset_baseline_accuracy was 0–2% (near-perfect separation). On holdout, 3 correct problems leaked into the steered set and got flipped wrong.

4. **Possible layer artifact (fourth):** Layer 24 (~86% depth) is deeper than SEAL's recommendation of layer 20 (~71% depth). "Small Vectors, Big Effects" documents LayerNorm placement effects at layers 23–24 in Qwen models that can reduce generalization.

---

## Artifact locations

### From Track A (verify all exist)

| Artifact | Path | Verify |
|----------|------|--------|
| Track A summary | `pathway2/track_a/SUMMARY.md` | Full writeup |
| Baseline answers | `pathway2/track_a/phase0/baseline_answers.json` | 500 problems |
| Baseline correct | `pathway2/track_a/phase0/baseline_correct.npy` | sum=57 |
| Multi-layer activations | `pathway2/track_a/phase0/activations/layer_*.npy` | Layers {14,17,19,20,22,24}, each (500,1536) |
| Confidence scores | `pathway2/track_a/phase0/confidence_scores.npy` | (500,) |
| Confidence model | `pathway2/track_a/phase0/confidence_model.pkl` | Fitted LR |
| Partition | `pathway2/track_a/phase0/partition.json` | 2×2 table |
| Probe AUROCs | `pathway2/track_a/phase0/probe_aurocs.json` | Per-layer |
| Selected layer | `pathway2/track_a/phase0/selected_layer.txt` | "24" |
| Steering vector | `pathway2/track_a/phase1/steering_vector.npy` | (1536,) |
| Sampling cosines | `pathway2/track_a/phase1/sampling_cosines.json` | 5-seed matrix |
| Locked config | `pathway2/track_a/phase2/locked_config.json` | t, τ, layer |
| Holdout results | `pathway2/track_a/phase3/holdout_results.json` | All conditions |
| Per-problem analysis | `pathway2/track_a/phase3/per_problem_analysis.json` | 100 holdout problems |

### From Pathway 1

| Artifact | Path |
|----------|------|
| Holdout mask | `pathway1/phase1/holdout_mask_seed9999.npy` |
| Winning features | `pathway1/phase0/winning_features.py` |
| Tier assignments | `pathway1/phase2/tier_assignments.json` |
| Youden threshold | `pathway1/phase1/holdout_metrics.json` → `youden_threshold_from_train` |
| Trajectory metadata | `data/experiment1_v2/trajectory_meta.json` |
| Token trajectories | `data/experiment1_v2/trajectories.npz` |

### Output directory

All Pathway 3 artifacts go to `~/topo-confidence/pathway3/`. Create subdirectories `phase1/`, `phase2/`, `phase3/`, `phase4/`.

---

## Phase 1: Contrastive Set Expansion (~8 hours GPU, run overnight)

### Goal
Expand from 46 correct examples to 150+ by generating multiple attempts per problem at non-zero temperature.

### Steps

1. **Generate 32 completions per train-400 problem at T=0.8.**
   
   For each of the 400 train problems:
   - Generate 32 completions with `do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=256`.
   - Check correctness of each completion via `check_correct()`.
   - Record: problem index, completion text, correct (Y/N).
   
   **VRAM note:** Generate one problem at a time with batch_size=1 across 32 sequential generations. At ~1.5s per generation, this is 32 × 400 × 1.5s ≈ 5.3 hours. If time-constrained, reduce to 16 generations per problem (~2.7 hours).
   
   Save to `pathway3/phase1/temperature_generations.json`:
   ```json
   {
     "problem_idx": {
       "completions": ["text1", "text2", ...],
       "correct": [true, false, ...],
       "n_correct": K
     }
   }
   ```

2. **Tally the expanded contrastive set.**
   
   - Count total unique correct solutions across all problems. Target: 150+.
   - Count how many NEW problems now have at least one correct solution (problems that were incorrect under greedy but correct under sampling).
   - If total correct solutions < 100, increase to T=1.0 or 48 generations per problem.
   
   Print: "Expanded contrastive set: X correct solutions across Y problems (was 46 across 46 problems)."

3. **Extract activations for ALL correct completions.**
   
   For each correct completion:
   - Tokenize the full prompt + completion.
   - Run a forward pass.
   - Cache the residual stream at the LAST token of the PROMPT (before generation begins) at layers {15, 20, 24, 25}.
   
   **Why layers 15, 20, 24, 25:** Layer 24 was Track A's selection. Layer 20 is SEAL's recommendation. Layers 15 and 25 provide non-adjacent partners for multi-layer composition ("Small Vectors, Big Effects" showed non-adjacent pairs compose constructively).
   
   **Important:** The activations are extracted from the PROMPT forward pass, not during generation. Different completions of the same prompt produce identical prompt activations. So you only need one forward pass per unique prompt — just extract at the 4 layers.
   
   For the NEGATIVE set: use the greedy (baseline) incorrect answers. Their prompt activations are already in Track A Phase 0.
   
   Save to `pathway3/phase1/expanded_activations/`:
   - `correct_activations_layer15.npy`, etc. — shape `(N_correct, 1536)`
   - `correct_problem_indices.npy` — which problem each activation belongs to
   - `expansion_summary.json` — counts, per-problem breakdown

4. **Extract activations at layers 15 and 25 for ALL 500 problems.**
   
   Track A already has layers {14, 17, 19, 20, 22, 24}. You need 15 and 25 for multi-layer experiments.
   
   Save to `pathway3/phase1/additional_activations/`:
   - `layer_15.npy`, `layer_25.npy` — each (500, 1536)

---

## Phase 2: Multi-Prototype Vector Extraction & Denoising (~1 hour CPU)

### Goal
Extract denoised, multi-prototype steering vectors at multiple layers that capture diverse failure modes.

### Steps

1. **Construct the expanded contrastive dataset.**
   
   - Positive set: all correct activations from Phase 1 (150+ samples).
   - Negative set: all incorrect train problems under greedy decoding (~354 samples).
   - For each layer in {15, 20, 25}: compute per-sample activation differences `d_i = h_correct_i - mean(h_incorrect)`.

2. **Extract denoised single vectors (baseline improvement).**
   
   For each layer in {15, 20, 25}:
   
   **(a) Bootstrap averaging:** Draw B=20 bootstrap samples from the expanded positive set, compute mean-difference for each, average the 20 vectors. Normalize to unit length.
   
   **(b) PCA projection:** Compute SVD on the per-sample difference matrix (N_correct × 1536). Project the bootstrap-averaged vector onto the subspace spanned by the top-k principal components. Sweep k ∈ {5, 10, 20, 50}. Select k by 5-fold CV on train-400 (see Phase 3 for CV protocol).
   
   Alternative (if extraction-based denoising is insufficient): YaPO (Bounhar et al. 2026, arXiv:2601.08441) learns sparse steering vectors in SAE latent space via DPO-style preference optimization, rather than extracting them from mean differences. It addresses the multi-semanticity entanglement problem that causes dense vectors to conflate unrelated features. A pretrained SAE for Qwen2.5-1.5B exists on HuggingFace (Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k). YaPO requires a training loop (~1-2 hours) using your correct/incorrect pairs as preference data, but converges faster and generalizes better than dense baselines. Code: https://github.com/MBZUAI-Paris/YaPO. Consider this if the bootstrap+PCA+sparsification pipeline from Steps 2a-2c doesn't close the generalization gap in Phase 3 CV results.
   
   **(c) Top-d sparsification:** After PCA projection, zero out all but the top-d largest-magnitude dimensions. Sweep d ∈ {50, 100, 200, 500}. Select by CV.
   
   Save the denoised vectors to `pathway3/phase2/denoised_vectors/`:
   - `layer15_bootstrap.npy`, `layer15_pca_k10.npy`, `layer15_sparse_d100.npy`, etc.

3. **Extract multi-prototype vectors via clustering.**
   
   For each layer in {15, 20, 25}:
   
   **(a) K-means clustering:** Cluster the per-sample activation differences into K groups. Sweep K ∈ {3, 5, 8}. Each cluster centroid is a "reasoning prototype" — a direction that captures a specific mode of correct reasoning.
   
   **(b) For each K, evaluate the cluster quality:**
   - Silhouette score (internal quality).
   - Per-cluster size (reject if any cluster has < 5 samples — unstable centroid).
   - Cross-cluster cosine similarity (low = good — clusters capture distinct directions).
   
   **(c) Prototype application at inference:** For each input problem, project its activation onto all K prototypes and compose a weighted steering vector:
   ```python
   projections = [dot(h_input, prototype_k) for k in range(K)]
   weights = softmax(projections / temperature)
   steering_vector = sum(w_k * prototype_k for w_k, prototype_k in zip(weights, prototypes))
   steering_vector /= norm(steering_vector)
   ```
   
   The softmax temperature controls how sharply the model selects a single prototype vs. blending. Sweep temperature ∈ {0.1, 0.5, 1.0}.
   
   Save to `pathway3/phase2/prototypes/`:
   - `layer20_K5_centroids.npy` — shape `(5, 1536)`
   - `layer20_K5_cluster_sizes.json`
   - `layer20_K5_cosine_matrix.json`

4. **Configure multi-layer injection.**
   
   For multi-layer experiments, extract vectors (either single or prototyped) at each target layer independently. At inference, register hooks at multiple layers simultaneously.
   
   Multi-layer configurations to test:
   
   | Config | Layers | Rationale |
   |--------|--------|-----------|
   | Track A baseline | 24 | Replication with expanded data |
   | SEAL | 20 | Literature recommendation for 1.5B |
   | Non-adjacent pair A | 15, 25 | Wide span, avoids 23-24 LayerNorm artifact |
   | Non-adjacent pair B | 20, 25 | SEAL + penultimate |
   | Triple | 15, 20, 25 | Full span |
   
   For multi-layer, each layer gets its OWN independently extracted vector (or prototype set). Steering magnitudes are initially set to `t=0.15` (Track A's optimal) for each layer, tuned in Phase 3.

   **SPREAD context (Yue et al. 2026):** "Test-time Diverse Reasoning by Riemannian Activation Steering" formulates steering as Riemannian optimization on a product of spheres and has been evaluated on **Qwen2.5-Math-1.5B-Instruct** on **MATH500** — our exact model family and benchmark. SPREAD targets diversity (pass@N) via log-determinant maximization of intervened activations, while we target accuracy on predicted failures via confidence-gated rotation. The approaches are complementary: SPREAD validates that geometry-aware steering works at this model scale for math reasoning. Their multi-trajectory formulation could inform future work on combining topo-confidence gating with diverse sampling.

5. **Save all Phase 2 artifacts.**

---

## Phase 3: Cross-Validated Evaluation (~10 hours GPU across Day 1 afternoon + Day 2 morning)

### Goal
Evaluate all candidate configurations via 5-fold stratified cross-validation, replacing the underpowered single-holdout design. Select the best configuration BEFORE touching the holdout.

### Why 5-fold CV instead of single holdout

Track A's McNemar p=0.505 showed n=100 is underpowered for detecting the true effect. 5-fold CV:
- Uses ALL 500 problems for both training and evaluation (each problem is in exactly one test fold).
- Produces 5 independent effect estimates → mean ± standard deviation.
- Each fold has 400 train / 100 test — same split size as Track A, so compute per fold is comparable.
- Stratified splits preserve the 11.4% base rate in each fold.

### Steps

1. **Create 5 stratified folds.**
   
   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   folds = list(skf.split(range(500), correct_labels))
   ```
   
   Save fold assignments to `pathway3/phase3/fold_assignments.json`.
   
   **Important:** The Pathway 1 holdout (seed 9999) is NOT used here. These are new folds for CV. The original holdout is reserved for the final single touch in Phase 4.

2. **For each fold (k=1..5):**
   
   **(a) Extract the steering vector(s) from the train fold (400 problems).**
   
   Using the expanded contrastive set from Phase 1:
   - Select correct completions from problems in this fold's train set.
   - Select incorrect completions from problems in this fold's train set.
   - Apply the denoising pipeline from Phase 2 (bootstrap average → PCA project → sparsify).
   - If using prototypes: cluster the train-fold differences and extract centroids.
   
   **(b) Fit the confidence model from the train fold.**
   
   Using the CORAL pipeline (PCA fit on train-fold trajectories, rank-bin on train-fold, StandardScaler fit on train-fold, LR on A+B+C features):
   - Predict P(correct) for the 100 test-fold problems.
   
   **(c) Apply steering to the test fold.**
   
   For each test-fold problem:
   - Compute topo-confidence.
   - Apply the gating rule (hard threshold at τ=0.40, or soft sigmoid — see Step 3).
   - If steered: generate with the steering hook active. If not: generate normally.
   - Record: generated text, correctness, whether steered, flip type.
   
   **(d) Record fold-level metrics:**
   - Net gain (wrong→right minus right→wrong).
   - Accuracy (test fold).
   - Number steered, steered subset accuracy, unsteered subset accuracy.

3. **Configurations to evaluate (in priority order):**
   
   Run each configuration across all 5 folds. Each fold costs ~2 hours (generate 100 problems × possibly multiple configs). Prioritize:
   
   **Config 1 (baseline replication):** Track A's exact setup — single vector at layer 24, hard gating at τ=0.40, t=0.15. This establishes the CV baseline and checks whether Track A's +3 holdout was representative or an outlier.
   
   **Config 2 (expanded data only):** Same as Config 1, but using the expanded contrastive set (150+ correct examples) for vector extraction. This isolates the effect of more data.
   
   **Config 3 (expanded + denoised):** Config 2 with bootstrap averaging + PCA projection at the best k from Phase 2. Isolates the denoising effect.
   
   **Config 4 (expanded + denoised + multi-layer):** Config 3 with the best multi-layer pair from Phase 2 Step 4. Isolates the multi-layer effect.
   
   **Config 5 (expanded + denoised + prototypes):** Config 3 with K-prototype clustering instead of single vector. Isolates the multi-prototype effect.
   
   **Config 6 (expanded + denoised + multi-layer + prototypes + soft gating):** The full stack. Uses:
   - Expanded contrastive data.
   - Denoised via bootstrap + PCA + sparsification.
   - Multi-prototype vectors at the best K.
   - Non-adjacent multi-layer injection.
   - Soft sigmoid gating: `t_per_problem = t_base * sigmoid(k * (threshold - confidence))`.
   
   Sweep the sigmoid sharpness k ∈ {5, 10, 20} on fold 1 only, then lock for folds 2–5.
   
   **Compute budget:** 6 configs × 5 folds × ~40 min each ≈ 20 hours. This exceeds the 2-day budget. **Prioritize Configs 1, 2, 4, and 6.** Configs 3 and 5 can be inferred from the others — if Config 4 (expanded + denoised + multi-layer) beats Config 2 (expanded only), the denoising and multi-layer each contributed. Run Configs 3 and 5 only if time permits.

4. **Aggregate results across folds.**
   
   For each configuration:
   - Mean net gain ± standard deviation across 5 folds.
   - Per-fold net gain (to check variance).
   - BCa bootstrap 95% CI on the mean net gain (10,000 resamples).
   - Bayesian P(improvement): model beneficial flip probability with Beta prior. If b total wrong→right and c total right→wrong across all folds, posterior = Beta(b+1, c+1). P(improvement) = 1 - BetaCDF(0.5 | b+1, c+1).
   - Mid-p McNemar's test per fold (more appropriate than asymptotic for small discordant pair counts).
   
   Save to `pathway3/phase3/cv_results.json`:
   ```json
   {
     "config_1": {
       "per_fold_net_gain": [X, X, X, X, X],
       "mean_net_gain": X.X,
       "std_net_gain": X.X,
       "bca_ci_95": [X.X, X.X],
       "bayesian_p_improvement": X.XX,
       "total_wrong_to_right": X,
       "total_right_to_wrong": X
     },
     ...
   }
   ```

5. **Select the best configuration BEFORE Phase 4.**
   
   Best = highest mean net gain across 5 folds, with tiebreaker going to the simpler configuration (fewer components).
   
   **Lock the configuration.** Save to `pathway3/phase3/locked_config.json`. Do not modify after seeing holdout results.

6. **Save all Phase 3 artifacts.**

---

## Phase 4: Holdout Evaluation & Mechanistic Analysis (~3 hours GPU)

### Goal
Test the locked configuration on the original Pathway 1 holdout-100. This is the honest answer to whether the generalization gap has closed.

### Steps

1. **Verify the holdout mask is the Pathway 1 original.**
   Load `pathway1/phase1/holdout_mask_seed9999.npy`. Assert sum=100. These 100 problems were NOT in any CV fold's train set (the CV used different splits), but they WERE in some CV folds' test sets — so the CV results on these problems are available for comparison.

2. **Apply the locked configuration to holdout-100.**
   
   Extract the steering vector(s) using ALL 400 train problems (the original Pathway 1 train set, not a CV fold). This gives the vector its maximum data. Fit the confidence model on the same 400. Score and generate holdout-100 under the locked config.
   
   Record:
   - Net gain vs. no-steering baseline.
   - Net gain vs. Track A's holdout result (+3).
   - Per-problem analysis (same format as Track A Phase 3).
   - Bootstrap 95% CI on net gain.
   - Mid-p McNemar's test.
   - Bayesian P(improvement) with flat prior.

3. **Compare against Track A holdout results.**
   
   The key comparison table:
   
   | Metric | Track A (holdout) | Pathway 3 (holdout) |
   |--------|-------------------|---------------------|
   | Net gain | +3 | ? |
   | Accuracy | 14% | ? |
   | Wrong→right | ? | ? |
   | Right→wrong | ? | ? |
   | Problems steered | ? | ? |
   | Steered subset accuracy | ? | ? |
   
   Also report: which specific holdout problems flipped differently between Track A and Pathway 3?

4. **Mechanistic analysis (if net gain improved).**
   
   **(a) Logit-lens decomposition** of the best vector(s). Compare against Track A's single vector — do the expanded/denoised/prototyped vectors promote different tokens?
   
   **(b) Prototype analysis** (if using K>1): which prototype is most frequently selected? Do different MATH subject areas (algebra, geometry, number theory) map to different prototypes? This would confirm the "single vector collapses failure modes" hypothesis.
   
   **(c) Multi-layer contribution** (if using multi-layer): ablate each layer individually. Does removing one layer from the pair/triple degrade performance? This confirms whether multi-layer composition is load-bearing or redundant.
   
   **(d) Confidence shift**: re-compute topo-confidence on steered activations using the pre-steering snapshot pattern. Does the new configuration push confidence scores higher on steered problems than Track A's single vector did?

   **(e) Connection to "The Shape of Reasoning" (Tan et al. 2026):** This paper independently showed that topological features (Vietoris-Rips persistence) of reasoning traces predict answer quality and proposed them as future reward signals for RL. Compare your topo-confidence features (PH on hidden-state point clouds) against their approach (PH on text-embedded reasoning steps). If both predict correctness but from different representations, the combination could be a stronger signal than either alone. This comparison strengthens the paper narrative — two independent lines of evidence that topology encodes reasoning quality.

5. **Save all Phase 4 artifacts** to `pathway3/phase4/`:
   - `holdout_results.json`
   - `per_problem_analysis.json` and `.txt`
   - `comparison_with_track_a.txt`
   - `statistical_tests.json`
   - `mechanistic/` subdirectory for logit-lens, prototype analysis, etc.

---

## Go/No-Go Decision

### For Pathway 4 (topologically-informed distillation):

- **FULL GO:** Pathway 3 CV mean net gain ≥ 5 across folds AND holdout net gain ≥ 5. The generalization gap is closed. The steering direction can serve as privileged information for distillation — it captures what "correct reasoning" looks like in activation space.

- **CONDITIONAL GO:** Pathway 3 holdout net gain ≥ 3 AND improves on Track A's +3 by any amount. The gap narrowed but didn't fully close. Proceed to Pathway 4 using the topo-confidence routing signal (still has AUROC 0.948) as privileged information, but don't rely on the steering direction itself.

- **PUBLICATION-READY STOP:** Pathway 3 CV mean net gain ≥ 3 AND holdout net gain ≥ 3 AND Bayesian P(improvement) > 0.90. The system works. Write up the full Pathway 1–3 arc as a paper: topological confidence predicts failure (Pathway 1), and confidence-gated steering improves accuracy on predicted failures (Pathways 2–3). The contribution is the combination, not the individual components.

- **PIVOT:** Pathway 3 doesn't improve over Track A. The expanded data and denoised vectors don't help. Consider:
  - **(a) PRA-style step-wise scoring** (Sohn et al. 2026, arxiv:2604.09482): use topo-confidence as an online process reward during beam search instead of pre-generation activation steering.
  - **(b) Bias-Only Adaptation** (Sinii et al. 2025, EMNLP): train steering vectors via RL rather than extracting from contrastive pairs. Requires more compute (~1 day of GRPO training) but produces vectors optimized for the objective rather than extracted from correlations.
  - **(c) Accept the routing-only result**: topo-confidence is diagnostic, not interventional. Pathway 1's routing result (2.5–4.5× lift) is the main contribution. Publish that.
  - **(d) YaPO sparse learned vectors** (Bounhar et al. 2026, arXiv:2601.08441): instead of extracting vectors from activation differences, learn sparse steering vectors in SAE latent space via DPO-style preference optimization using correct/incorrect math pairs. Disentangles multi-semantic features that dense mean-difference vectors conflate. Pretrained Qwen2.5-1.5B SAE available. Requires ~1-2 hours of training but produces vectors optimized for the objective rather than extracted from correlations.

---

## Technical notes

- **Temperature sampling config:** `do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=256`. This differs from Track A's greedy config. The temperature generations are ONLY for expanding the contrastive set — all evaluation is still greedy.
- **Greedy evaluation is mandatory:** All steering evaluation (Phase 3 CV folds, Phase 4 holdout) uses `do_sample=False, max_new_tokens=256`. Same as Track A and Pathway 1.
- **Prompt activations are identical across completions:** Different temperature completions of the same prompt produce identical prompt-end activations (the activations depend only on the prompt, not the generation). You need only one forward pass per prompt to get the contrastive activations.
- **Hook architecture:** Reuse Track A's steering hook with dynamic dim handling for 2D/3D hidden states and non-tuple decoder output. The bug fix from Track A is essential.
- **Multi-layer hooks:** Register one hook per layer. Each hook has its own vector and magnitude. Remove all hooks between configurations.
- **CV fold statefulness:** For each fold, the ENTIRE pipeline (PCA, rank-binning, StandardScaler, LR, vector extraction, prototype clustering) must be fit on the fold's train set only. No leakage from the test fold. This is the same statefulness warning from Pathway 1 — it applies per-fold now.
- **The Pathway 1 holdout is still sacred.** The 5-fold CV in Phase 3 uses DIFFERENT splits (random_state=42). The original holdout (seed 9999) is touched exactly once in Phase 4.
- **Qwen2.5-1.5B architecture:** 28 layers (0–27), hidden_dim=1536, 12 attention heads. Residual stream: `model.model.layers[L]`. See Track A for hook implementation details.
- **VRAM budget:** Model ~3GB. KV cache ~1–2GB. Activations for 4 layers ~25MB. Prototypes ~50KB. Total ~5GB with headroom.

---

## Key references

- **PDS (Prototype-Based Dynamic Steering):** Kayan & Zhang, arXiv:2510.05498. Clusters activation differences into reasoning prototypes.
- **CHaRS (Concept Heterogeneity-aware Representation Steering):** arXiv:2603.02237. GMM+optimal transport for multi-modal steering. Tested on Qwen2.5-3B.
- **Small Vectors, Big Effects:** Sinii et al., arXiv:2509.06608. Non-adjacent multi-layer composition on Qwen2.5-Math-7B.
- **SEAL:** Chen et al., COLM 2025. Layer 20 for 1.5B models.
- **SAE-RSV:** arXiv:2509.23799. SAE-based vector denoising. Matches 1000-sample CAA with 10 samples.
- **Steering Vector Generalization:** Tan et al., NeurIPS 2024, arXiv:2407.12404. Diminishing returns after ~80 samples.
- **Mean-Difference Optimality:** Im & Li, 2025, arXiv:2502.02716. Mean-diff is optimal under squared error but not per-sample.
- **PRA:** Sohn et al. 2026, arXiv:2604.09482. Step-wise reward for frozen policies (pivot option).
- **Bias-Only Adaptation:** Sinii et al., EMNLP 2025. RL-trained steering for Qwen2.5-1.5B.
- **Spherical Steering:** You et al. 2026, arXiv:2602.08169. Norm-preserving rotation, vMF gating.
- **What Drives Steering:** Cheng et al. 2026, arXiv:2604.08524. OV circuit mechanism.
- **YaPO:** Bounhar et al. 2026, arXiv:2601.08441. Learnable sparse steering vectors in SAE space via DPO. Code: https://github.com/MBZUAI-Paris/YaPO
- **SPREAD (Test-time Diverse Reasoning by Riemannian Activation Steering):** Yue et al. 2026. Geometry-aware steering on Qwen2.5-Math-1.5B for MATH500. Validates spherical steering at our exact model scale and benchmark.
- **The Shape of Reasoning:** Tan et al. 2026. Vietoris-Rips persistence on reasoning traces predicts answer quality. Closest existing work to topo-confidence — independently validates that topological features predict reasoning correctness, though computed on text-embedded reasoning steps rather than hidden-state point clouds.
