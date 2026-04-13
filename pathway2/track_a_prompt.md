# Pathway 2 — Track A: Minimum Viable Steering Experiment (v2)

## What this prompt is

This is the focused, must-run experiment that answers one question: **does activation steering improve Qwen2.5-1.5B accuracy on MATH-500 problems that topo-confidence predicts will fail?**

Track A produces a clean yes/no answer in ~10–12 hours of GPU compute. It uses ONE steering method (spherical), ONE layer (probe-selected from a quick sweep), ONE gating approach (topo-confidence threshold), and ONE primary metric (net gain on holdout-100). No ablations, no alternatives, no design-space exploration. Those belong in Track B, which runs only if Track A shows signal.

---

## Context

Pathway 1 is **COMPLETE with FULL GO**:

- Holdout AUROC = **0.948** [0.898, 0.986]
- Topo-confidence **dominates** output-entropy at all meaningful coverage levels
- Best surviving feature tier: **A+B+C (44 features)**
- Model: **Qwen2.5-1.5B-Instruct** (28 layers, hidden_dim=1536)
- GPU: **RTX 2060 Super** (8GB VRAM). Model in fp16 ≈ 3GB, leaving ~5GB headroom.
- Generation: **greedy decoding** (`do_sample=False`, hardcoded in `extractor.py:166`). Fully deterministic. No multi-run averaging needed.
- Generation length: **`max_new_tokens=256`** (from `default.yaml` and `experiment1_v2.py:199`).

### Why spherical steering, not additive CAA

Additive CAA (`h' = h + α·v`) changes activation norms in an input-dependent way, causing effective-rank collapse and generation degradation — especially in models below 7B (Spherical Steering, You et al. 2026; Selective Steering, Dang & Ngo 2026). Your att-docs findings show that hard problems already compress H1 topology at the terminal layer. Additive steering would compound this compression.

Spherical steering rotates activations via slerp while preserving norms: `h' = ||h|| · slerp(h/||h||, v/||v||, t)` where `t` controls rotation strength. This guarantees `||h'|| = ||h||` — no magnitude distortion, no effective-rank collapse. You et al. report +10–13% over additive CAA on TruthfulQA with Qwen2.5-7B.

### Why probe-selected layer, not layer 28

Your topological signal peaks at layer 28 (z=8.12), but that is where to DETECT failure, not where to INTERVENE. Multiple independent findings converge:
- SEAL uses layer 20 for DeepSeek-R1-Distill-Qwen-1.5B (same architecture family, same parameter count).
- ROME (Meng et al. 2022): factual edits generalize best at ~40–60% depth; terminal layers cause "regurgitation."
- Amnesic Probing (Elazar et al. 2021): probe accuracy has Spearman r=0.085 with actual causal importance.
- CAA Scaling Laws (Ali et al. 2025): CAA most effective at early-mid layers, not terminal. For a 28-layer model, this predicts layers ~10–16.
- Your att-docs: terminal-layer H1 compresses to 0 for hard problems. Steering against natural compression at that layer fights the model's computation.

Rather than hard-coding layer 20, Phase 0.5 runs a quick probe sweep (~30 min) to find where correctness is most linearly separable, then commits to that layer. The sweep covers {14, 17, 19, 20, 22, 24} — spanning the 50–86% depth range where the literature suggests interventions are most effective, while avoiding terminal layers.

---

## Artifact locations

### From Pathway 1 (verify all exist before proceeding)

| Artifact | Path | Verify |
|----------|------|--------|
| Holdout mask | `pathway1/phase1/holdout_mask_seed9999.npy` | `sum = 100` |
| Holdout metrics | `pathway1/phase1/holdout_metrics.json` | Contains AUROC, CI |
| Train features | `pathway1/phase1/features_train400.npy` | Shape `(400, 78)` |
| Holdout features | `pathway1/phase1/features_holdout100.npy` | Shape `(100, 78)` |
| Best tier | `pathway1/phase2/best_surviving_tier.txt` | "A+B+C" |
| Tier assignments | `pathway1/phase2/tier_assignments.json` | Feature→tier |
| Artifact hashes | `pathway1/phase0/artifact_locks.json` | SHA-256 hashes |
| Winning features | `pathway1/phase0/winning_features.py` | 1407 lines, verify SHA-256 |
| Token trajectories | `data/experiment1_v2/trajectories.npz` | 500 keys, verify SHA-256 |
| Trajectory metadata | `data/experiment1_v2/trajectory_meta.json` | `correct` array, sum=57 |
| Calibration metrics | `pathway1/phase4/calibration_metrics.json` | ECE, Brier |
| Youden threshold | `pathway1/phase1/holdout_metrics.json` | `youden_threshold_from_train: 0.3203` |

**Verify SHA-256 hashes** against `pathway1/phase0/artifact_locks.json` for: `winning_features.py`, `trajectories.npz`, `trajectory_meta.json`.

### Output directory

All Track A artifacts go to `~/topo-confidence/pathway2/track_a/`. Create subdirectories `phase0/`, `phase1/`, `phase2/`, `phase3/`.

---

## Phase 0: Baseline & Confidence Scores (~1.5 hours)

### Goal
Regenerate the unsteered MATH-500 baseline, extract activations at candidate layers, and compute topo-confidence scores for all 500 problems.

### Steps

1. **Verify Pathway 1 artifacts.** Check every path in the table above. Verify SHA-256 hashes match `artifact_locks.json`. If any mismatch, halt.

2. **Create output directories.**
   ```
   mkdir -p ~/topo-confidence/pathway2/track_a/{phase0,phase1,phase2,phase3}
   ```

3. **Extend the extractor for multi-layer activation capture.**
   
   `HiddenStateExtractor.extract()` currently collects hidden states from `self.layers[-1]` only (line 103). You need activations at multiple candidate layers for the Phase 0.5 probe sweep.
   
   Modify the extractor (or write a wrapper) to:
   - Run a forward pass on each prompt (no generation).
   - Cache the residual stream at the LAST token position for layers {14, 17, 19, 20, 22, 24}.
   
   These are two independent passes: (1) batched forward pass → hidden states, (2) one-at-a-time generation → text. The existing `extract_with_output()` already implements this two-pass pattern. Do not confuse "activations during generation" with "activations from the prompt forward pass."

4. **Generate all 500 MATH-500 answers (unsteered baseline).**
   - Use identical generation config to Pathway 1: `do_sample=False`, `max_new_tokens=256`.
   - For each problem, save the generated text.
   - Check correctness via `check_correct()` from `scripts/experiment1_math500.py`.
   - **Assert baseline accuracy matches Pathway 1** (57/500 = 11.4%, tolerance ±2). The 57 is across all 500 problems; expect ~46 correct in train-400, ~11 in holdout-100.
   - If it doesn't match, halt and investigate. Do not proceed with a drifted baseline.

5. **Extract activations at candidate layers for all 500 problems.**
   - Forward pass each prompt.
   - Cache `activations[layer][last_token_position]` → shape `(1536,)` per problem, for each layer in {14, 17, 19, 20, 22, 24}.
   - Save to `track_a/phase0/activations/layer_14.npy` through `layer_24.npy` — each shape `(500, 1536)`.

6. **Compute topo-confidence scores for all 500 problems.**
   
   **STATEFULNESS WARNING:** The CORAL feature pipeline has massive leakage if PCA/rank-binning/scaling are fit on all-500 instead of train-400 only. Pathway 1 Phase 0 measured up to 5M mean absolute drift in derived features. You MUST:
   - Fit PCA on train-400 trajectories only.
   - Fit rank-binning on train-400 only (holdout samples: percentile-rank within train distribution via `np.searchsorted`).
   - Fit StandardScaler on train-400 features only.
   - Transform all 500 with the train-fitted pipeline.
   
   **Feature subsetting:** `features_train400.npy` has shape `(400, 78)` — all tiers A+B+C+D. You need only the 44 A+B+C features. Load `pathway1/phase2/tier_assignments.json`, select feature columns where `tier ∈ {A, B, C}`. This produces a `(400, 44)` matrix. Apply the same column selection to holdout and all-500 features.
   
   Fit LogisticRegression on the 44-column train-400 matrix. Predict P(correct) for all 500.
   
   **Verification:** Compare holdout-100 confidence scores against `pathway1/phase1/holdout_predictions.npy`. They MUST match exactly (to float precision). If they don't, the pipeline has drifted.
   
   Save to `track_a/phase0/confidence_scores.npy` — shape `(500,)`.

7. **Create the 2×2 partition** using the Youden threshold τ=0.3203 (from `pathway1/phase1/holdout_metrics.json`, key `youden_threshold_from_train`):
   
   | | Predicted Success (P≥τ) | Predicted Failure (P<τ) |
   |---|---|---|
   | Actually Correct | TP | FN |
   | Actually Incorrect | FP | TN |
   
   Print the counts. Save partition indices and counts to `track_a/phase0/partition.json`.

8. **Save all Phase 0 artifacts** to `track_a/phase0/`:
   - `baseline_answers.json`, `baseline_correct.npy`, `baseline_accuracy.txt`
   - `activations/layer_14.npy` through `activations/layer_24.npy`
   - `confidence_scores.npy`, `confidence_model.pkl`
   - `partition.json`

---

## Phase 0.5: Probe Sweep & Layer Selection (~30 minutes)

### Goal
Select the best intervention layer by probing where correctness is most linearly separable. This replaces the hard-coded layer 20 from v1.

### Steps

1. **Train linear probes at all candidate layers.**
   
   For each layer in {14, 17, 19, 20, 22, 24}:
   - Load `track_a/phase0/activations/layer_XX.npy`.
   - Select train-400 rows using the holdout mask.
   - Train `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
   - Report AUROC via 10-fold stratified CV on train-400.
   
   Save to `track_a/phase0/probe_aurocs.json` — `{layer: auroc}`.

2. **Select the intervention layer.**
   
   Primary criterion: highest probe AUROC.
   
   Tiebreaker (within 0.01 AUROC): prefer the layer at 50–71% depth (layers 14–20). The CAA scaling laws suggest earlier layers respond better to additive/rotational intervention, even if later layers encode the distinction more strongly.
   
   **Lock the layer.** Save to `track_a/phase0/selected_layer.txt`. Print: "Selected layer: XX (probe AUROC: X.XXX)".
   
   All subsequent phases use this layer. References to "layer 20" below should be read as "the selected layer."

3. **If the best probe AUROC < 0.55:** Halt. No candidate layer encodes correctness well enough for steering. Skip to the NO-GO outcome directly.

---

## Phase 1: Steering Vector Extraction (~15 minutes)

### Goal
Extract a "correct reasoning" steering vector from train-400 contrastive pairs at the selected layer (from Phase 0.5).

In the steps below, `L` refers to the selected layer from `track_a/phase0/selected_layer.txt`.

### Steps

1. **Partition train-400 by correctness.**
   - Load the holdout mask. Select the 400 train indices.
   - Identify correct train problems (~46) and incorrect train problems (~354).

2. **Construct contrastive sets.**
   - Positive set: all correct train problems (~46).
   - Negative set: randomly sample 46 from incorrect train problems (`random_state=42`).
   
   **Sampling stability check:** Repeat with seeds 43, 44, 45, 46. Compute pairwise cosine similarity between the 5 resulting mean-difference vectors. If all cosines > 0.9, use seed-42 vector. If any cosine < 0.7, average all 5 vectors and re-normalize.

3. **Compute the steering vector.**
   ```
   activations = np.load(f'track_a/phase0/activations/layer_{L}.npy')
   H_correct = mean(activations[correct_train_indices], axis=0)  # (1536,)
   H_incorrect = mean(activations[sampled_incorrect_indices], axis=0)  # (1536,)
   steering_vector = H_correct - H_incorrect  # (1536,)
   steering_vector /= np.linalg.norm(steering_vector)  # unit normalize
   ```

4. **Validate with a linear probe.**
   - Already computed in Phase 0.5. Load `track_a/phase0/probe_aurocs.json` and report the selected layer's AUROC.
   - If AUROC < 0.55, this was already caught in Phase 0.5 Step 3.

5. **Compute projection–confidence correlation.**
   - For each train-400 problem: `projection = dot(layer_L_activation, steering_vector)`.
   - Compute Spearman r between projections and topo-confidence scores.
   - If |r| > 0.3: steering direction and topo-confidence are aligned (measuring related things).
   - If |r| < 0.1: orthogonal signals (topo-confidence identifies WHERE to steer, vector provides WHAT direction — this is actually ideal for targeted steering).

6. **Save artifacts** to `track_a/phase1/`:
   - `steering_vector.npy` — shape `(1536,)`
   - `sampling_cosines.json` — 5-seed cosine matrix
   - `probe_auroc.txt`
   - `projection_confidence_correlation.json`

---

## Phase 2: Steering Strength Calibration on Train-400 (~6 hours)

### Goal
Find the optimal rotation strength `t` for spherical steering that maximizes net gain on predicted-failure problems without degrading predicted-success problems. Train-400 only. Holdout is NOT touched.

`L` refers to the selected layer from Phase 0.5 throughout.

### Steps

1. **Implement spherical steering.**
   
   The steering hook rotates the residual stream at layer L toward the steering direction:
   ```python
   import torch
   import torch.nn.functional as F
   
   def slerp(v0, v1, t):
       """Spherical linear interpolation."""
       v0_norm = F.normalize(v0, dim=-1)
       v1_norm = F.normalize(v1, dim=-1)
       dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
       omega = torch.acos(dot)
       sin_omega = torch.sin(omega)
       # Handle near-parallel case
       if sin_omega.abs().min() < 1e-6:
           return (1 - t) * v0_norm + t * v1_norm
       s0 = torch.sin((1 - t) * omega) / sin_omega
       s1 = torch.sin(t * omega) / sin_omega
       return s0 * v0_norm + s1 * v1_norm
   
   steering_vec_tensor = torch.tensor(steering_vector, device=device, dtype=torch.float16)
   
   def steering_hook(module, input, output):
       h = output[0]  # shape varies: (batch, seq_len, 1536) during prompt, (1, 1, 1536) during cached generation
       norms = h.norm(dim=-1, keepdim=True)  # preserve original norms
       rotated = slerp(h, steering_vec_tensor.unsqueeze(0).unsqueeze(0).expand_as(h), t)
       modified = rotated * norms  # restore original norms
       return (modified,) + output[1:]
   ```
   
   **Critical implementation notes:**
   - Return a NEW tuple. Do NOT modify `output[0]` in-place — in-place operations corrupt KV cache during `model.generate()`.
   - The hook fires at EVERY forward pass during autoregressive generation, not just prompt processing.
   - **KV cache behavior:** During the initial prompt pass, `output[0]` has shape `(1, seq_len, 1536)` — all token positions are steered. During cached autoregressive generation, `output[0]` has shape `(1, 1, 1536)` — only the CURRENT token is steered, because KV cache means previous positions aren't recomputed. This is actually ideal: the prompt gets a full-sequence rotation, and each new generated token gets individually steered. No special handling needed.
   - Register on `model.model.layers[L]`. Verify the module path via `dict(model.named_modules()).keys()`.

2. **Sanity check: t=0 must match baseline.**
   - Generate answers for 10 random train problems with the hook active at `t=0`.
   - Outputs MUST be identical to Phase 0 baseline. If not, the hook is corrupting something. Debug before proceeding.

3. **Sweep rotation strength on train-400.**
   
   For each `t` in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}:
   - Apply **uniform** steering to ALL 400 train problems at strength `t`.
   - Record generated answer and correctness for each problem.
   - Compute:
     - **Overall accuracy** (fraction correct out of 400)
     - **Originally-correct accuracy** (~46 problems): degradation from baseline?
     - **Originally-incorrect accuracy** (~354 problems): any new correct answers?
     - **Net gain** = (newly correct) − (newly incorrect)
     - **KL divergence** between steered and unsteered output token distributions for 20 random samples (as a coherence diagnostic — if KL explodes, the model is generating from a distorted distribution).
   
   **Compute budget:** 7 values × ~50 min each ≈ 6 hours. If time-constrained:
   - Run t=0 (sanity), t=0.10, t=0.20 first.
   - If t=0.10 shows zero effect AND t=0.20 shows zero effect, the steering vector is ineffective at the selected layer. See "If steering has no effect" below.
   - If t=0.10 shows some effect, fill in the rest of the sweep.

4. **Identify the best uniform `t`.**
   
   Best `t` maximizes net gain subject to:
   - Originally-correct subset loses ≤ 2 problems.
   - KL divergence remains below the threshold observed at the point where outputs start degrading qualitatively.
   
   **Lock this value.** Save to `track_a/phase2/best_t.txt`.

5. **Test confidence-gated steering on train-400.**
   
   Using the best `t` from Step 4:
   - For τ_steer in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}:
     - For each train problem: if `confidence_score < τ_steer`, generate with steering at strength `t`. Otherwise, generate normally.
     - Record overall accuracy, per-quadrant accuracy, number of problems steered.
   
   **Coverage note:** At τ=0.30, Pathway 1 routing metrics show only ~10% coverage (few problems fall below threshold). At τ=0.10, coverage jumps to ~39%. Lower thresholds steer more problems, giving better statistical power to measure net gain. The sweep extends down to 0.10 to capture this range.
   
   The comparison that matters: **does targeted steering achieve higher net gain than uniform steering?** This happens if targeted steering avoids the right→wrong flips that uniform steering causes on already-correct problems.

6. **Lock the full configuration before holdout.**
   
   Save to `track_a/phase2/locked_config.json`:
   ```json
   {
     "method": "spherical",
     "layer": L,
     "layer_probe_auroc": X.XXX,
     "best_t": X.XX,
     "best_tau_steer": X.XX,
     "vector_type": "mean_diff_seed42" | "averaged_5seeds"
   }
   ```
   **This is frozen.** Do not modify after seeing holdout results.

7. **Save artifacts** to `track_a/phase2/`:
   - `uniform_sweep.json` — all t values, all metrics, KL divergences
   - `targeted_sweep.json` — all τ_steer values, all metrics
   - `locked_config.json`
   - `comparison_table.txt` — no-steering vs. uniform vs. targeted

---

## Phase 3: Holdout Evaluation (~2 hours)

### Goal
Test the locked configuration on holdout-100. This is the honest answer.

### Steps

1. **Verify holdout mask matches Pathway 1.**
   Load `pathway1/phase1/holdout_mask_seed9999.npy`. Assert sum=100.

2. **Run three conditions on holdout-100:**
   
   **(a) No steering:** Generate normally. Record answer + correctness.
   
   **(b) Uniform steering:** Apply spherical steering at `t=best_t` to ALL holdout problems.
   
   **(c) Targeted steering (locked config):** Compute topo-confidence for each holdout problem. Apply steering only when `P(correct) < τ_steer`. Otherwise generate normally.
   
   For all conditions: `do_sample=False`, `max_new_tokens=256`. Same config, always.

3. **Compute metrics for each condition:**
   - **Accuracy** out of 100
   - **Net gain** vs. no-steering baseline
   - **Originally-correct preservation** (how many of the ~11 holdout-correct problems stayed correct?)
   - **Originally-incorrect recovery** (how many of the ~89 holdout-incorrect problems became correct?)
   - **Bootstrap 95% CI** on accuracy and net gain (1000 stratified resamples)
   
   For targeted steering:
   - How many problems were steered?
   - Steered subset: how many flipped wrong→right? How many stayed wrong?
   - Unsteered subset: accuracy must match no-steering baseline exactly (verify — if it doesn't, something is leaking).

4. **McNemar's test.**
   - Compare paired correctness between no-steering and targeted steering.
   - Report exact p-value.
   - n=100 is small. Report honestly. The value of Track A is the proof of concept.

5. **Per-problem analysis table.**
   
   For all 100 holdout problems, record:
   - Problem index, topo-confidence score, steered? (Y/N)
   - Baseline correct, uniform correct, targeted correct
   - Flip type: none / wrong→right / right→wrong
   
   Save as `track_a/phase3/per_problem_analysis.json` and human-readable `.txt`.

6. **Save artifacts** to `track_a/phase3/`:
   - `holdout_results.json` — all conditions, all metrics, bootstrap CIs
   - `statistical_tests.json` — McNemar p-value, bootstrap CI of accuracy difference
   - `per_problem_analysis.json` and `.txt`
   - `holdout_comparison_table.txt`

---

## Go/No-Go Decision

Based on holdout-100 results:

- **GO to Track B:** Net gain ≥ 1 on holdout-100 under ANY condition (uniform or targeted). There is signal to optimize. Run Track B to explore the design space.

- **GO to Track B (no-effect variant):** Net gain = 0 on holdout BUT the Phase 0.5 linear probe at the selected layer has AUROC ≥ 0.65. The layer encodes correctness but the mean-difference vector doesn't capture a useful intervention direction. Track B should explore: different layers, neuron-level steering, additive vs. spherical comparison.

- **SKIP Track B, proceed to Pathway 3:** Targeted steering achieves net gain ≥ 3 on holdout-100 AND outperforms uniform steering. The mechanism works well enough to build on without further optimization.

- **NO-GO for activation steering:** Net gain ≤ 0 under all conditions AND linear probes at all candidate layers have AUROC < 0.55 (from Phase 0.5). The model's representations don't encode correctness in a way amenable to steering. Consider: (a) the signal may only be in the terminal layer and steering there is fundamentally limited, (b) proceed directly to Pathway 4 (distillation) where the topo-confidence signal serves as privileged information.

---

## If steering has no effect

If `t=0.10` and `t=0.20` both produce outputs identical to baseline:

1. **Verify the hook fires.** Add a print inside the hook. Confirm it runs during `model.generate()`.
2. **Check the hook target.** Print `model.model.layers[L]` and confirm it's a transformer block, not an attention submodule.
3. **Check vector magnitude.** Compute `||steering_vector|| / mean(||layer_L_activations||)`. If the ratio is < 0.001, the rotation is imperceptible. Try `t=0.5` or `t=1.0`.
4. **Try a different layer.** If the selected layer has zero effect, try the next-best probe layer from Phase 0.5.
5. **If nothing works at any reasonable `t`:** the spherical steering formulation may not produce enough perturbation for this model size. Switch to additive CAA with small α (0.5–2.0) as a diagnostic. If additive works but spherical doesn't, the issue is with slerp damping at small angles. Increase `t` aggressively.

---

## Technical notes

- **Qwen2.5-1.5B architecture:** 28 layers (0–27), hidden_dim=1536, 12 attention heads, vocab=151936. Residual stream after block L: `model.model.layers[L]`. Unembedding: `model.lm_head.weight`.
- **Hook registration:** `handle = model.model.layers[L].register_forward_hook(steering_hook)` where L is the selected layer. Remove after generation: `handle.remove()`. If running a sweep, re-register per condition.
- **VRAM:** Model ~3GB + KV cache ~1–2GB + activations ~0.5GB = ~5GB. Steering vectors add ~6KB. Generate one problem at a time (`batch_size=1`).
- **Determinism:** `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`. With greedy decoding, all conditions are fully deterministic. Same prompt + same model + same steering config = same output, every time.
- **Two-pass architecture:** Activations come from a forward pass on the prompt. Generation is a separate call. The existing `extract_with_output()` implements this. Do not confuse "activations during generation" with "activations from the prompt forward pass."
- **The holdout is sacred.** Do not use holdout-100 for anything in Phase 0, 1, or 2.
- **Save generated texts, not just correctness.** You need the texts for qualitative analysis and correctness re-checking.
- **n=100 is noisy.** Bootstrap CIs will be wide (±4–6%). This is expected. Report honestly.
