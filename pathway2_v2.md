# Pathway 2: Topo-Confidence-Guided Activation Steering (v2)

## Context you need to know

This prompt is part of a 4-pathway research arc. Pathway 1 (validation, ablation, routing prototype) is **COMPLETE with FULL GO**:

- Holdout AUROC = **0.948** [0.898, 0.986] — firmly in the >=0.90 band.
- Topo-confidence **dominates** output-entropy at all meaningful coverage levels (e.g., 21% coverage: topo=0.476 vs entropy=0.095).
- Output-entropy routing is essentially random — the model's token-level uncertainty does NOT predict problem-level correctness.
- Best surviving feature tier: **A+B+C (44 features)**.
- Decision: **FULL GO** for Pathways 2-4. Pathway 4 has positive prior (topo-confidence captures something output-entropy doesn't).

Pathway 2 answers the question: **can we use topo-confidence to identify where the model will fail, then apply activation steering to the model's hidden states during generation to improve correctness on those predicted-failure problems?**

This is the causal test. Pathway 1 showed topo-confidence *predicts* failure. Pathway 2 tests whether we can *intervene* on that prediction. If steering improves accuracy specifically on the low-confidence subset without degrading the high-confidence subset, we have evidence that the topological features are capturing something causally relevant to the model's reasoning process — not just a correlate.

**Important context on the model:** You are steering **Qwen2.5-1.5B-Instruct** (NOT a reasoning model like DeepSeek-R1-Distill). This model does not produce extended chain-of-thought with reflection/transition thoughts. It generates relatively short, direct answers. This means SEAL-style thought-type decomposition does NOT apply directly — you need CAA-style contrastive steering instead.

### Key findings from ATT Phase 7 (transformer hidden-state topology)

These findings were established in the ATT research project (10 directions on Qwen2.5-1.5B MATH-500) and are directly relevant to steering decisions:

1. **Terminal-layer concentration (z = 8.12):** The topological correctness signal is overwhelmingly concentrated at layer 28 (the terminal layer). All 28 internal layers are significant (z > 2), but layer 28 dominates. The CORAL Tier B features that drive the +0.105 AUROC gain mostly involve layers 26-28 (`cos_l27_l28`, `vel_l26_l27`, `cos_l9_l28`, `cos_l16_l28_bin`, `cos_l14_l28_bin`). **Layer 28 is the a priori best candidate for steering.**

2. **Token-position topology:** Problem-region tokens carry 24.7x stronger topology-difficulty signal than instruction tokens (diff = 0.815 vs 0.033). Steering focused on problem-region token positions may be more effective than full-sequence steering.

3. **Terminal-layer H1 compression:** All difficulty levels show 0 H1 features at the terminal layer — the model squeezes loop structure out completely by layer 28. The steering vector may fight this natural compression at high alpha, which could explain degeneration.

4. **Attention-hidden binding decouples with difficulty:** Easy problems have tight coupling (0.683), hard problems decouple (0.465). Increasing binding score after steering would suggest the intervention is restoring internal alignment.

5. **Intrinsic dimension scales with difficulty:** Terminal-layer intrinsic dimension ranges from 6.67 (easy) to 12.01 (hard). Steering should not collapse this dimensionality.

6. **Terminal-layer effect is Qwen-specific:** Only 1/4 tested models shows the terminal-layer concentration. Transfer to non-Qwen models is not expected.

7. **Topology and output entropy are orthogonal:** Pearson r = 0.062. They measure genuinely different aspects of the model's computation.

---

## Hardware constraints

- **GPU:** NVIDIA RTX 2060 Super (8GB VRAM)
- **Model:** Qwen2.5-1.5B-Instruct in fp16 ~ 3GB VRAM
- **Headroom:** ~5GB for activations, KV cache, and steering vector extraction
- **Steering vectors:** Per-layer vectors with hidden_dim=1536 across 28 layers = ~170KB total. Negligible.
- **Key limitation:** You cannot run inference on larger models (7B+) for contrastive pair generation. All contrastive data comes from Qwen2.5-1.5B-Instruct itself.
- **Time budget:** Feature extraction took ~55 min for 500 problems in Pathway 1. Expect similar per-run times for steered inference. Plan accordingly — each full MATH-500 evaluation run costs ~1 hour.

---

## Artifact locations (from Pathway 1)

All Pathway 1 artifacts are in `~/topo-confidence/pathway1/`. Verify each exists before proceeding. **Additionally, verify SHA-256 hashes match `pathway1/phase0/artifact_locks.json` for `winning_features.py`, `trajectories.npz`, and `trajectory_meta.json`.**

| Artifact | Path | Description |
|----------|------|-------------|
| Per-sample confidence scores | `pathway1/phase1/holdout_predictions.npy` | (100,) predicted P(correct) for holdout-100 |
| Holdout mask | `pathway1/phase1/holdout_mask_seed9999.npy` | Boolean (500), True=holdout |
| Train features | `pathway1/phase1/features_train400.npy` | (400, 78) feature matrix |
| Holdout features | `pathway1/phase1/features_holdout100.npy` | (100, 78) feature matrix |
| Holdout metrics | `pathway1/phase1/holdout_metrics.json` | AUROC, CI, classification thresholds |
| Best tier | `pathway1/phase2/best_surviving_tier.txt` | A+B+C (44 features) |
| Tier assignments | `pathway1/phase2/tier_assignments.json` | Feature-to-tier mapping |
| Calibration metrics | `pathway1/phase4/calibration_metrics.json` | ECE, Brier |
| Routing metrics | `pathway1/phase4/routing_metrics.json` | Coverage/accuracy tradeoffs |
| Operating point | `pathway1/phase4/operating_point.json` | Best tau threshold |
| Winning features code | `pathway1/phase0/winning_features.py` | Frozen CORAL feature extractor (1407 lines) |
| Artifact locks | `pathway1/phase0/artifact_locks.json` | SHA-256 hashes for key artifacts |
| Token trajectories | `data/experiment1_v2/trajectories.npz` | 500 arrays, each (n_tokens, 1536) |
| Trajectory metadata | `data/experiment1_v2/trajectory_meta.json` | Correctness labels, generated texts |
| Hidden state extractor | `topo_confidence/extractor.py` | `HiddenStateExtractor` class |
| Baselines | `topo_confidence/baselines.py` | Output entropy, max token prob |

**Output directory:** All Pathway 2 artifacts go to `~/topo-confidence/pathway2/`. Create subdirectories `phase0/`, `phase1/`, `phase2/`, `phase3/`, `phase4/`.

---

## The steering approach

### Why CAA (Contrastive Activation Addition), not SEAL

SEAL (Chen et al., COLM 2025) decomposes chain-of-thought into execution/reflection/transition thoughts and steers to suppress wasteful reflection. It was tested on **DeepSeek-R1-Distill-Qwen-1.5B** — a reasoning model that produces long CoT traces with explicit reflection. **Qwen2.5-1.5B-Instruct is NOT a reasoning model.** It produces short, direct answers without extended CoT. SEAL's thought-type decomposition is inapplicable.

Instead, use **CAA** (Rimsky et al., ACL 2024): extract a steering vector from the mean activation difference between correct and incorrect reasoning traces, then add it to the residual stream during generation. This is the right tool for a non-CoT model where the intervention target is "reason more like you do when you get things right."

### The novel element: confidence-gated steering

The literature consistently warns that uniform steering degrades performance on inputs the model already handles well. IBM's CAST paper (ICLR 2025): "simple activation steering indiscriminately affects all inputs, rendering the steered model much less useful." ITI documents a truthfulness-helpfulness tradeoff. Spherical Steering shows ungated steering degrades generation quality at high strengths.

**Your model already solves ~57/500 = 11.4% of MATH-500 correctly.** Uniform steering risks degrading those correct solutions to potentially help the remaining 88.6%. Your AUROC of 0.948 means the confidence system can route interventions with high precision — concentrate steering on predicted failures while leaving predicted successes alone.

The confidence-gated approach:
1. Compute topo-confidence score for each problem (using the Pathway 1 pipeline — no generation needed, just a forward pass).
2. If P(correct) < tau_steer: apply the steering vector during generation.
3. If P(correct) >= tau_steer: generate normally (no intervention).

This is the key experiment. The comparison is: **targeted steering (confidence-gated) vs. uniform steering vs. no steering**.

---

## Phase 0: Environment Setup & Baseline Generation

### Goal

Establish the unsteered baseline on MATH-500 and verify you can extract activations at arbitrary layers during generation. This phase also collects the contrastive activation data needed for Phase 1.

### Steps

1. **Verify all Pathway 1 artifacts exist and match locked hashes.**
   - Check every path in the artifact table above.
   - For `winning_features.py`, `trajectories.npz`, and `trajectory_meta.json`: compute SHA-256 and verify against `pathway1/phase0/artifact_locks.json`.
   - If any are missing or hashes mismatch, halt. You depend on the holdout mask, per-sample confidence scores, and correctness labels.

2. **Install dependencies.**
   ```
   pip install repeng  # For steering vector extraction/application
   ```
   If `repeng` doesn't support Qwen2.5 out of the box, fall back to manual hook-based steering (see Technical Notes). The `repeng` library wraps HuggingFace models in a `ControlModel` and provides `ControlVector.train()` — but it may not handle all architectures. Test with a single forward pass before committing.

   **Alternative libraries (if repeng fails):**
   - `steering-vectors` (pip install steering-vectors) — cleaner API, may have better arch support.
   - Manual PyTorch hooks — always works but requires more code. See Technical Notes for the hook pattern.

3. **Regenerate all 500 MATH-500 answers AND extract multi-layer prompt-end activations.**

   > **NOTE: The existing `HiddenStateExtractor` only extracts from the LAST specified layer** (see `extractor.py` line 103). You must modify it — or write a new extraction function — to collect hidden states from ALL candidate layers simultaneously. The modification is straightforward: inside the batch loop, iterate over `self.layers` and collect each layer's hidden states from `outputs.hidden_states[layer_idx + 1]`.

   > **Architecture note:** The extractor uses a two-pass design. Pass 1: batched forward pass on prompts to extract hidden states (no generation). Pass 2: one-at-a-time `model.generate()` for text output. The "prompt-end activations" come from Pass 1, extracted at the **last real token position** of each prompt. These are NOT collected during generation.

   This is your unsteered baseline:
   - Use the same generation config as Pathway 1: **`do_sample=False` (greedy decoding), `max_new_tokens=256`**. Both are hardcoded/locked from Pathway 1. Generation is fully deterministic — no need for multiple runs per condition.
   - For EACH problem, during the forward pass (Pass 1):
     - **Cache the residual stream activations at layers 14, 17, 19, 20, 22, 24, 26, 27, 28** at the LAST token position of the prompt (before generation begins). These are the "prompt-end activations" you'll use for contrastive vector extraction.
     - Note: layers 26-28 are included because ATT Phase 7 Direction 1 found the terminal layer (28) has z-score = 8.12 — the strongest topological signal by far. The CORAL Tier B features driving the +0.105 AUROC gain primarily involve these layers.
   - In Pass 2, generate the output text.
   - Verify correctness using `check_correct()` from `scripts/experiment1_v2.py`.
   - **Assert that the baseline accuracy matches Pathway 1's recorded accuracy** (57/500 = 11.4%). If it doesn't match within +/-2 correct answers, something has changed (model weights, generation config, correctness checker). Investigate before proceeding.
   - Save to `pathway2/phase0/`:
     - `baseline_answers.json` — {problem_idx: generated_text} for all 500
     - `baseline_correct.npy` — boolean array (500,)
     - `baseline_accuracy.txt` — e.g., "57/500 = 11.4%"
     - `prompt_end_activations.npz` — keyed by layer: `layer_14`, `layer_17`, ..., `layer_28`, each shape (500, 1536)

4. **Compute topo-confidence scores for all 500 problems.**

   > **STATEFULNESS WARNING:** The CORAL feature pipeline has massive leakage if PCA/rank-binning/scaling are fit on all-500 instead of train-400 only. Pathway 1 Phase 0 measured up to 5M mean absolute drift in derived features (worst: `jerk_l7_x_h0tot_sq`). The `winning_features.py` code's `fit()` and `transform()` methods must be called separately: `fit(train_400_trajectories)`, then `transform(all_500_trajectories)`. Verify by comparing confidence scores against `pathway1/phase1/holdout_predictions.npy` — they should match exactly for the holdout-100 subset.

   - Load the Pathway 1 pipeline: extract features using `winning_features.py` (PCA fit on train-400, rank-bin on train-400, StandardScaler fit on train-400).
   - Fit LogisticRegression on train-400 with the best surviving tier (A+B+C).
   - Predict P(correct) for all 500 problems.
   - Save to `pathway2/phase0/confidence_scores.npy` — array of (500,) floats.
   - Also save the fitted model: `pathway2/phase0/confidence_model.pkl`.

5. **Partition problems by confidence and correctness.**
   Create the 2x2 partition using the Youden threshold from Pathway 1 (or use tau=0.32 from the results):
   - **True Positives (TP):** high confidence AND correct. These problems should NOT be steered.
   - **True Negatives (TN):** low confidence AND incorrect. These are the primary steering targets.
   - **False Positives (FP):** high confidence AND incorrect. The confidence system failed — steering would skip these.
   - **False Negatives (FN):** low confidence AND correct. Steering would be applied unnecessarily — must not degrade these.
   
   Save counts and indices to `pathway2/phase0/partition.json`. Print the 2x2 table.

6. **Save artifacts** to `pathway2/phase0/`.

---

## Phase 1: Steering Vector Extraction

### Goal

Extract a "correct reasoning" steering vector from contrastive pairs (correct vs. incorrect math solutions). Identify the optimal layer and validate that the vector captures a meaningful direction.

### Steps

1. **Construct contrastive pairs from the model's own behavior.**
   
   Using the baseline results from Phase 0:
   - **Positive set (P+):** All problems the model answered correctly within train-400 (~46 samples). These are instances where the model's internal computation succeeded.
   - **Negative set (P-):** A randomly sampled subset of problems the model answered incorrectly within train-400, matched in size to P+ (~46 samples, random_state=42).
   
   **Important:** Use ONLY the **train-400** subset for vector extraction. The holdout-100 is reserved for final evaluation in Phase 3. Within train-400, expect ~46 correct and ~354 incorrect. Sample 46 from the incorrect set.
   
   The contrastive pairs are NOT paired 1-to-1 by problem difficulty or topic. We compute the mean activation difference across each set (CAA-style), not per-pair differences. This is more robust with limited data.

2. **Compute the steering vector via mean activation difference.**

   For each candidate layer L in {14, 17, 19, 20, 22, 24, 26, 27, 28}:
   ```
   H_correct = mean(activations[L][correct_train_indices], axis=0)  # shape (1536,)
   H_incorrect = mean(activations[L][incorrect_train_sample_indices], axis=0)  # shape (1536,)
   steering_vector[L] = H_correct - H_incorrect  # shape (1536,)
   ```
   
   Normalize each vector to unit length: `steering_vector[L] /= np.linalg.norm(steering_vector[L])`.
   
   **Sampling stability check:** Repeat the mean-difference computation with 5 different random seeds (42, 43, 44, 45, 46) for the negative set sampling. Compute pairwise cosine similarity between the 5 vectors at each layer. If all cosines > 0.9, the seed-42 vector is stable — use it. If any cosine < 0.7, average the 5 vectors and re-normalize — the direction is sensitive to the specific sample, and averaging reduces this noise.
   
   Save all steering vectors to `pathway2/phase1/steering_vectors.npz` — keyed by layer.
   Save sampling stability results to `pathway2/phase1/sampling_stability.json`.

3. **Validate the steering direction with a linear probe.**
   
   For each candidate layer, train a logistic regression probe on the prompt-end activations (train-400 only):
   - Input: activation at layer L, shape (400, 1536)
   - Target: correctness label (0/1)
   - Use `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
   - Report AUROC via 10-fold stratified CV on train-400.
   
   This tells you which layers carry the most correctness-predictive information. If a layer's probe AUROC is near 0.5, steering at that layer is unlikely to help — the layer doesn't encode the correct/incorrect distinction.
   
   Save to `pathway2/phase1/probe_aurocs.json` — {layer: auroc}.

4. **Select the best layer.**
   
   Primary criterion: highest probe AUROC on train-400 CV. 
   
   Secondary criterion: if multiple layers are within 0.01 AUROC of each other, prefer layers 26-28. ATT Phase 7 Direction 1 found the terminal layer (28) has the strongest topological correctness signal (z = 8.12). This is also consistent with the general finding that mid-to-late layers are most steerable.
   
   **Lock the layer choice before Phase 2.** Save to `pathway2/phase1/best_layer.txt`.

5. **Compute PCA refinement (optional but recommended).**
   
   Instead of using the raw mean-difference vector, extract the first principal component of the per-sample contrastive differences:
   ```
   diffs = activations[best_layer][correct_indices] - mean(activations[best_layer][incorrect_sample], axis=0)
   # This gives (n_correct, 1536) differences
   pca = PCA(n_components=1, svd_solver='full')
   pca.fit(diffs)
   refined_vector = pca.components_[0]  # shape (1536,)
   ```
   
   Compare the cosine similarity between the raw mean-difference vector and the PCA-refined vector. If cosine > 0.8, they're capturing the same direction and the mean-difference is fine. If cosine < 0.5, the PCA direction may be more robust — test both in Phase 2.
   
   Save to `pathway2/phase1/steering_vector_pca.npy` and `pathway2/phase1/cosine_raw_vs_pca.txt`.

6. **Sanity check: project confidence scores onto the steering direction.**
   
   For each of the 400 train samples, compute:
   ```
   projection = dot(activation[best_layer][i], steering_vector) 
   ```
   
   Compute the Spearman correlation between these projections and the topo-confidence scores from Phase 0. If the correlation is strong (|r| > 0.3), the steering vector and topo-confidence are aligned — they're measuring related aspects of the model's internal state. If the correlation is weak (|r| < 0.1), they're capturing orthogonal signals, which is actually *good* for the targeted steering story (topo-confidence identifies WHERE to steer, the steering vector provides WHAT direction to push).
   
   Save to `pathway2/phase1/projection_confidence_correlation.json`.

7. **Save artifacts** to `pathway2/phase1/`.

---

## Phase 2: Steering Strength Calibration (Train-400 Only)

### Goal

Find the optimal steering coefficient alpha that maximizes accuracy improvement on predicted-failure problems without degrading predicted-success problems. All calibration uses train-400 only. The holdout-100 is NOT touched until Phase 3.

### Steps

1. **Set up the steering inference pipeline.**
   
   The steering mechanism adds `alpha * steering_vector` to the residual stream at the best layer during generation. Implementation options:
   
   **(a) Using repeng:**
   ```python
   from repeng import ControlModel, ControlVector
   model = ControlModel(base_model, layer_range=range(best_layer, best_layer+1))
   cv = ControlVector(directions={best_layer: steering_vector})
   model.set_control(cv, coeff=alpha)
   output = model.generate(...)
   ```
   
   **(b) Using PyTorch hooks (if repeng doesn't support Qwen2.5):**
   ```python
   def steering_hook(module, input, output):
       # output[0] is the hidden states tensor, shape (batch, seq_len, 1536)
       # IMPORTANT: Do NOT modify in-place — return a new tuple.
       # In-place modification can corrupt KV cache tensors during generate().
       modified = output[0] + alpha * torch.tensor(steering_vector, device=output[0].device, dtype=output[0].dtype)
       return (modified,) + output[1:]
   
   handle = model.model.layers[best_layer].register_forward_hook(steering_hook)
   output = model.generate(...)
   handle.remove()
   ```
   
   **Critical:** Verify the hook injection point. For Qwen2 architecture, the residual stream after each transformer block is `model.model.layers[L]`. Check the model's `named_modules()` output to confirm the exact path. A wrong hook point will silently produce garbage.
   
   Test with a single problem first: generate with alpha=0 (should match baseline exactly — generation is deterministic with `do_sample=False`) and alpha=1.0 (should produce different output). If alpha=0 doesn't match baseline character-for-character, the hook is corrupting something.

2. **Sweep steering strength on train-400.**
   
   For each alpha in {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}:
   - Run inference on ALL 400 train problems with uniform steering (every problem gets steered at strength alpha).
   - Record the generated answer and correctness for each problem.
   - Compute:
     - **Overall accuracy** (fraction correct out of 400)
     - **Accuracy on originally-correct** subset (~46 problems): does steering degrade these?
     - **Accuracy on originally-incorrect** subset (~354 problems): does steering fix any of these?
     - **Net gain** = (newly correct) - (newly incorrect) — the number of problems that flipped from wrong->right minus those that flipped right->wrong.
     - **Coherence check**: for 10 random samples at each alpha, manually inspect the generated text. At what alpha does output become incoherent or degenerate? Record this as alpha_max_coherent.
   
   **This is computationally expensive.** 9 alpha values x ~1 hour each = ~9 hours. Consider:
   - Running alpha=0 first (sanity check — must match Phase 0 baseline character-for-character since generation is greedy/deterministic).
   - Then alpha=1.0 (standard CAA coefficient). If this produces zero change, the steering vector may be too weak or at the wrong layer — investigate before running the full sweep.
   - Then sweep the rest if alpha=1.0 shows any effect.
   
   Save to `pathway2/phase2/uniform_sweep_results.json`:
   ```json
   {
     "alpha_values": [0.0, 0.25, "..."],
     "overall_accuracy": ["..."],
     "correct_subset_accuracy": ["..."],
     "incorrect_subset_accuracy": ["..."],
     "net_gain": ["..."],
     "alpha_max_coherent": "X.X",
     "per_problem_results": {
       "0": {"alpha_0.0": "correct", "alpha_0.5": "correct"},
       "..."
     }
   }
   ```

3. **Identify the best uniform alpha.**
   
   The best alpha maximizes net gain subject to:
   - alpha <= alpha_max_coherent (outputs must remain coherent)
   - Accuracy on originally-correct subset drops by <= 2 problems (tolerable degradation)
   
   If NO alpha value produces positive net gain, the steering vector is ineffective. See the Failure Modes section for next steps.
   
   **Lock the best uniform alpha before proceeding to targeted steering.** Save to `pathway2/phase2/best_uniform_alpha.txt`.

4. **Test confidence-gated (targeted) steering on train-400.**
   
   Using the best uniform alpha from Step 3:
   - Compute topo-confidence for each train-400 problem (from Phase 0).
   - Sweep tau_steer (the confidence threshold below which steering is applied) from 0.1 to 0.5 in steps of 0.05:
     - For each problem: if P(correct) < tau_steer, generate with steering at strength alpha. Otherwise, generate normally.
     - Record: overall accuracy, accuracy on each quadrant of the 2x2 partition (TP/TN/FP/FN from Phase 0 Step 5), and the number of problems steered.
   
   Also test **strength-modulated steering** at the best tau_steer:
   - Instead of binary steer/don't-steer, set alpha_per_problem = alpha_base * (1 - P(correct)) / (1 - tau_steer).
   - This scales steering strength proportionally to predicted failure probability — problems the model is "almost certainly wrong about" get full strength, borderline cases get light steering.

5. **Ablation: multi-layer steering (optional but recommended).**
   
   If single-layer steering at the best layer shows any effect (net gain >= 1 at any alpha):
   - Apply the steering vector simultaneously at the top-3 layers by probe AUROC from Phase 1.
   - Use the same best alpha at each layer (or alpha/3 at each to keep total perturbation magnitude constant).
   - This tests whether multi-site intervention captures more of the layer-dynamics signal — since Tier B's +0.105 AUROC comes from layer-to-layer cosine similarities, velocity, and curvature, single-layer steering cannot directly optimize these multi-layer dynamics.
   
   Save to `pathway2/phase2/multilayer_results.json`.

6. **Ablation: token-position-selective steering (optional but recommended).**
   
   ATT Phase 7 Direction 8 found that problem-region tokens carry 24.7x stronger topology-difficulty signal than instruction tokens. Test whether focusing steering on problem-token positions is more effective:
   - Identify the token positions corresponding to the problem statement (between `"Problem: "` and `"\n\nSolution:"` in the prompt template from `scripts/experiment1_v2.py`).
   - Apply the steering vector ONLY at those token positions during generation, leaving instruction tokens unsteered.
   - Compare against full-sequence steering at the same alpha.
   
   Save to `pathway2/phase2/position_selective_results.json`.

7. **Compare: targeted vs. uniform vs. no steering.**
   
   Produce a comparison table:
   
   | Condition | Overall Acc | Correct-subset Acc | Incorrect-subset Acc | Net Gain |
   |-----------|-------------|--------------------|-----------------------|----------|
   | No steering (baseline) | X | Y | Z | 0 |
   | Uniform alpha=best | ... | ... | ... | ... |
   | Targeted tau=best | ... | ... | ... | ... |
   | Strength-modulated | ... | ... | ... | ... |
   | Multi-layer (if run) | ... | ... | ... | ... |
   | Position-selective (if run) | ... | ... | ... | ... |
   
   The key comparison is: **does targeted steering achieve similar or better net gain than uniform steering, while preserving more of the correct-subset accuracy?** This is the entire thesis of the project.

8. **Lock the steering configuration before touching holdout.**
   
   Save to `pathway2/phase2/locked_config.json`:
   ```json
   {
     "best_layer": "L",
     "best_alpha": "X.X",
     "best_tau_steer": "X.X",
     "steering_mode": "targeted | uniform | strength_modulated | multilayer | position_selective",
     "vector_type": "mean_diff | pca_refined",
     "multi_layer_config": "null or {layers: [...], alphas: [...]}"
   }
   ```
   
   **This configuration is frozen.** Do not modify it after seeing holdout results.

9. **Save artifacts** to `pathway2/phase2/`.

---

## Phase 3: Holdout Evaluation

### Goal

Test the locked steering configuration on the holdout-100 that was never used during any calibration. This is the honest answer to "does confidence-gated steering work?"

### Steps

1. **Verify the holdout mask is the same one from Pathway 1.**
   Load `pathway1/phase1/holdout_mask_seed9999.npy`. Verify it has sum = 100. These 100 problems were NOT used for vector extraction, alpha calibration, or tau selection.

2. **Run all conditions on holdout-100:**
   
   For each holdout problem:
   
   **(a) No steering:** Generate normally. Record answer and correctness.
   
   **(b) Uniform steering:** Apply steering at alpha=best_alpha to ALL holdout problems. Record answer and correctness.
   
   **(c) Targeted steering (locked config):** Compute topo-confidence for each holdout problem. Apply steering only when P(correct) < tau_steer. Record answer, correctness, and whether steering was applied.
   
   **(d) Strength-modulated (if it won in Phase 2):** Apply proportional steering based on confidence. Record all details.
   
   **(e) Any ablation winners (multi-layer, position-selective):** Run the locked config from Phase 2.

3. **Compute metrics for each condition:**
   
   For each condition:
   - **Accuracy:** fraction correct out of 100.
   - **Accuracy on holdout-correct subset** (~11 problems): degradation from baseline?
   - **Accuracy on holdout-incorrect subset** (~89 problems): improvement from baseline?
   - **Net gain:** newly correct minus newly incorrect.
   - **Bootstrap 95% CI** on accuracy and net gain (1000 resamples).
   
   For targeted steering specifically:
   - **Steered subset accuracy** (problems where steering was applied): before vs. after.
   - **Unsteered subset accuracy** (problems where steering was NOT applied): should match baseline exactly (verify this — if it doesn't, something is leaking).
   - **Precision of steering application:** what fraction of steered problems actually needed help (were originally incorrect)?
   - **Recall of steering application:** what fraction of originally-incorrect problems received steering?

4. **Per-problem analysis.**
   
   Create a table of all 100 holdout problems with columns:
   - Problem index
   - Topo-confidence score
   - Steered? (Y/N)
   - Baseline correct? (Y/N)
   - Uniform-steered correct? (Y/N)
   - Targeted-steered correct? (Y/N)
   - Flip type: none / wrong->right / right->wrong
   
   Identify specific problems where:
   - Targeted steering flipped wrong->right (success cases — analyze the generated text)
   - Targeted steering flipped right->wrong (failure cases — analyze what went wrong)
   - Uniform steering hurt but targeted steering preserved correctness (the value of gating)
   
   Save to `pathway2/phase3/per_problem_analysis.json` and a human-readable `per_problem_analysis.txt`.

5. **Statistical tests.**
   
   - **McNemar's test:** Compare paired correctness between baseline and targeted steering on holdout-100. This tests whether the number of flips (wrong->right vs right->wrong) is significantly different from chance.
   - **Bootstrap difference test:** Compute the bootstrap distribution of (targeted_accuracy - baseline_accuracy). Report the 95% CI of this difference. If the CI includes 0, the improvement is not statistically significant at n=100.
   - Report the **exact p-value** and whether it passes the alpha=0.05 threshold. Be honest — n=100 is small, and we expect wide CIs.

6. **Save artifacts** to `pathway2/phase3/`:
   - `holdout_results.json` (all conditions, all metrics)
   - `per_problem_analysis.json` and `.txt`
   - `statistical_tests.json` (McNemar p-value, bootstrap CI)
   - `holdout_comparison_table.txt`

---

## Phase 4: Mechanistic Analysis & Transfer Test

### Goal

Understand WHY steering works (or doesn't) and test whether the steering vector transfers to unseen problems.

### Steps

1. **Analyze the steering vector.**
   
   - Compute the top-20 logit-lens projections: multiply the steering vector by the model's unembedding matrix to see which tokens are most promoted/suppressed by the vector.
     ```python
     unembed = model.lm_head.weight  # shape (151936, 1536)
     logit_lens = unembed @ steering_vector  # shape (151936,)
     top_promoted = logit_lens.topk(20)
     top_suppressed = logit_lens.topk(20, largest=False)
     ```
   - Report the top-20 promoted and suppressed tokens. If the promoted tokens include mathematical operators, "Step", "=", digits, or structured reasoning markers, the vector is steering toward productive computation. If it's promoting random tokens, the direction may be noise.
   - Save to `pathway2/phase4/logit_lens_analysis.json`.

2. **Visualize the intervention effect.**
   
   For 5 representative problems (mix of successful flips, failed flips, and unchanged):
   - Record the token-by-token log-probability differences between steered and unsteered generation.
   - Identify WHERE in the generation sequence the steering vector has the most impact (early vs. late tokens).
   - Save generation traces to `pathway2/phase4/generation_traces/`.

3. **Transfer test (if time permits).**
   
   If you have access to a second math dataset (e.g., GSM8K subset, or a different split of MATH), run the locked steering configuration on it WITHOUT any recalibration:
   - Extract topo-confidence features using the same pipeline (Pathway 1's PCA, scaler, LR model).
   - Apply the same steering vector at the same layer and alpha.
   - Apply the same tau_steer threshold.
   - Report accuracy: baseline vs. uniform steering vs. targeted steering.
   
   **Caveat:** The terminal-layer topological concentration is Qwen-specific (ATT Phase 7 Direction 6: 1/4 models replicate it). Transfer to non-Qwen models is not expected. A fair same-architecture transfer test would use Qwen2.5-0.5B or a different math dataset on the same Qwen2.5-1.5B model.
   
   Save to `pathway2/phase4/transfer_results.json` (or `transfer_skipped.txt` if not run).

4. **Revisit the topo-confidence features post-steering.**
   
   For the steered outputs (using the locked config):
   - Re-extract hidden states and compute topo-confidence features.
   - Compare topo-confidence scores before vs. after steering for each problem.
   - If steering INCREASES topo-confidence on the steered subset, the intervention is moving the model's internal state toward the "correct answer" geometry — strong evidence that topology is causally relevant.
   - If steering DECREASES topo-confidence but improves accuracy, the features are correlative but not causally upstream.
   
   Save to `pathway2/phase4/confidence_shift.json`.

5. **Monitor intrinsic dimension and attention-hidden binding (optional).**
   
   - **Intrinsic dimension:** Compute TwoNN intrinsic dimension of terminal-layer representations before and after steering. Terminal-layer ID ranges from 6.67 (easy) to 12.01 (hard) in unsteered problems (ATT Phase 7 Direction 7). A large decrease after steering suggests the vector is simplifying representations rather than enriching them.
   
   - **Attention-hidden binding:** Compute the attention-hidden binding score (see `att/llm/attention_binding.py`) before and after steering. Easy problems have binding = 0.683, hard problems = 0.465 (ATT Phase 7 Direction 10). If steering increases binding toward 0.68 on hard problems, it's restoring the internal alignment that difficulty disrupts.
   
   Save to `pathway2/phase4/mechanistic_metrics.json`.

6. **Save artifacts** to `pathway2/phase4/`.

---

## Deliverables

By the end of Pathway 2, you should have produced:

1. **Baseline verification:** regenerated MATH-500 answers, confirmed accuracy matches Pathway 1.
2. **Steering vector:** extracted from train-400 correct/incorrect contrastive pairs, at the best layer, with linear probe validation and sampling stability check.
3. **Strength calibration:** uniform alpha sweep results, showing the effect curve and identifying the best alpha and coherence limit.
4. **Targeted vs. uniform comparison:** on train-400, demonstrating whether confidence-gating improves the steering outcome.
5. **Ablations:** multi-layer steering and position-selective steering results (if run).
6. **Holdout evaluation:** all conditions (no steering, uniform, targeted) on holdout-100 with bootstrap CIs and statistical tests.
7. **Per-problem analysis:** identifying specific success and failure cases, with generated text for qualitative review.
8. **Mechanistic analysis:** logit-lens decomposition, generation traces, confidence shift analysis, optional ID/binding metrics.
9. **Go/no-go decision for Pathway 3** (topology-aware fine-tuning):

### Go/No-Go Criteria

- **FULL GO:** Targeted steering achieves net gain >= 3 on holdout-100 (i.e., at least 3 more problems flipped wrong->right than right->wrong), AND targeted steering outperforms uniform steering on correct-subset preservation. Proceed to Pathway 3 with the validated steering vector as a training signal.

- **CONDITIONAL GO:** Targeted steering achieves net gain >= 1 on holdout-100, OR uniform steering achieves net gain >= 3. The steering vector captures real signal, but confidence-gating may not add value. Proceed to Pathway 3 with uniform steering as baseline.

- **PIVOT:** No steering condition achieves net gain >= 1 on holdout-100, BUT the linear probe at the best layer has AUROC >= 0.70. The layer encodes the correct/incorrect distinction, but the mean-difference vector doesn't capture a useful intervention direction. Pivot to: (a) try SEAL-style decomposition if Qwen2.5 produces any multi-step reasoning, (b) try per-problem adaptive alpha, (c) try multi-layer steering (if not already tested in Phase 2 Step 5).

- **NO-GO for activation steering:** No steering condition achieves net gain >= 1, AND linear probes at all layers have AUROC < 0.60. The model's internal representations don't encode correctness in a linearly separable way that's amenable to additive steering. Skip Pathway 3, proceed directly to Pathway 4 (distillation) if the topo-confidence routing signal is still valuable.

---

## Failure modes and contingency plans

### Failure Mode 1: No effect at any alpha

If alpha=1.0 produces outputs identical or near-identical to baseline:
- Check that the hook is actually firing (add a print statement inside the hook, verify it runs during generation).
- Check that you're hooking the residual stream AFTER the layer, not the attention output.
- Try applying the vector at ALL token positions during generation (not just the prompt-end). CAA typically adds to all generated tokens.
- Try un-normalized vectors (remove the unit-length normalization). The natural scale of the mean difference may be important.

### Failure Mode 2: Steering produces gibberish at all alpha

If even alpha=0.25 produces incoherent output:
- The steering vector may be too large relative to the activation scale. Compute the ratio `||steering_vector|| / mean(||activations||)` — if it's > 0.1, the vector is a large perturbation.
- **Consider the terminal-layer H1 compression pattern** (ATT Phase 7 Direction 9): the model naturally squeezes all loop structure out by the terminal layer. If the steering vector opposes this compression (pushing toward richer topology at a layer that expects zero loops), the model's downstream computation may not handle it. Try steering at an earlier layer (e.g., layer 22 or 24) where some topological structure still exists.
- Try applying only at the first generated token position, not all positions.
- Try the PCA-refined vector instead — it may be cleaner.

### Failure Mode 3: Steering helps on incorrect problems but equally hurts correct ones (net gain ~ 0)

This is the exact problem confidence-gating is designed to solve. If uniform steering has net gain ~ 0 but flips >5 problems in each direction, then targeted steering (applying only to predicted-failure cases) should achieve positive net gain by avoiding the right->wrong flips. Proceed to Phase 2 Step 4 with optimism.

### Failure Mode 4: Topo-confidence scores are uncorrelated with steering benefit

If the problems that benefit from steering are NOT the low-confidence ones, but random:
- The topo-confidence signal predicts failure from a different cause than what steering fixes.
- Try using the linear probe's predicted probability (from Phase 1 Step 3) as the gating signal instead of topo-confidence. This uses the layer's own representation of correctness rather than the external topological signal.

---

## Technical notes

- **Hook application during generation:** The steering vector must be added at EVERY forward pass during autoregressive generation, not just the initial prompt processing. Each new token's forward pass should see the steering vector. If using `model.generate()`, register the hook before calling generate and remove it after. If using a manual generation loop, apply the hook at each step.

- **Token position for activation extraction (Phase 0):** Use the LAST token position of the input prompt. For Qwen2.5's chat template, this is typically the last token before generation begins. Verify by checking that `model.generate()` produces the first new token immediately after this position.

- **Generation config is locked:** `max_new_tokens=256`, `do_sample=False` (greedy decoding). Both are from Pathway 1 (`extractor.py:166`, `configs/default.yaml`). Generation is fully deterministic — the same input always produces the same output. No need for multiple runs per condition or random seeds during generation. All conditions must use identical generation config for valid comparison.

- **Batch size during steered generation:** With 3GB model + 5GB headroom, you can likely run batch_size=1 for generation with long outputs. If VRAM is tight, generate one problem at a time. Steering vectors add negligible VRAM.

- **The holdout is sacred.** Do not use holdout-100 for anything in Phase 0, 1, or 2. All vector extraction, alpha tuning, and tau selection happen on train-400 only.

- **Save EVERYTHING.** Generated texts, not just correctness labels. You need the texts for qualitative analysis and for re-checking correctness if the checker has edge cases.

- **n=100 is noisy.** With 11.4% baseline accuracy on holdout, expect ~11 correct out of 100. Even a perfect steering vector can only flip at most 89 wrong->right. A net gain of 3 means going from 11/100 to 14/100 — a 27% relative improvement but only 3 percentage points absolute. Bootstrap CIs will be wide. This is expected. The value of this pathway is the proof of concept, not a production-ready system.

- **Qwen2.5-1.5B architecture reference:**
  - 28 transformer layers (indexed 0-27)
  - Hidden dim: 1536
  - 12 attention heads, head dim 128
  - Residual stream: `model.model.layers[L]` outputs a tuple; the first element is the hidden states.
  - Unembedding: `model.lm_head.weight`, shape (151936, 1536) — Qwen2.5 has a large vocabulary.

- **Prompt template** (from `scripts/experiment1_v2.py`):
  ```
  Solve the following math problem. Give your final answer after '####'.
  
  Problem: {problem}
  
  Solution:
  ```
  The problem-region tokens lie between "Problem: " and "\n\nSolution:" — this boundary is needed for position-selective steering in Phase 2 Step 6.

---

## Key references

- **CAA:** Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (ACL 2024). The core method. https://aclanthology.org/2024.acl-long.828/
- **SEAL:** Chen et al., "Steerable Reasoning Calibration of Large Language Models for Free" (COLM 2025). Proved steering works for math reasoning on the same architecture family. Code: https://github.com/VITA-Group/SEAL
- **RepE:** Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (2023). The broader framework. https://arxiv.org/abs/2310.01405
- **ITI:** Li et al., "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (NeurIPS 2023). Head-targeted intervention. https://arxiv.org/abs/2306.03341
- **CAST:** IBM, "Programming Refusal with Conditional Activation Steering" (ICLR 2025). Conditional steering — closest to the confidence-gating approach. https://github.com/IBM/activation-steering
- **repeng library:** https://github.com/vgel/repeng — simplest implementation path for CAA.
- **steering-vectors library:** https://github.com/steering-vectors/steering-vectors — alternative with better docs.
- **Bias-Only Adaptation:** Sinii et al. (EMNLP 2025). RL-trained steering for math. https://aclanthology.org/2025.emnlp-main.467/
- **Small Vectors, Big Effects:** Mechanistic study of RL-induced reasoning via steering vectors (2026). https://arxiv.org/html/2509.06608
