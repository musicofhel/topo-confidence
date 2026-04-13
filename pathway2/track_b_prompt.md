# Pathway 2 — Track B: Ablation Battery (v2)

## Prerequisites

**Do NOT run Track B until Track A is complete.** Track B requires:

1. Track A Phase 0 artifacts (baseline answers, multi-layer activations, confidence scores, partition).
2. Track A Phase 0.5 artifacts (probe AUROCs, selected layer).
3. Track A Phase 1 artifacts (steering vector, sampling cosines).
4. Track A Phase 2 artifacts (best `t`, best `τ_steer`, locked config, uniform and targeted sweep results).
5. Track A Phase 3 artifacts (holdout results, per-problem analysis, statistical tests).

**Entry condition:** Track A Go/No-Go decision is either "GO to Track B" or "GO to Track B (no-effect variant)." If Track A says "NO-GO for activation steering," do not run Track B.

---

## What this prompt is

Track A answers: "does it work at all?" Track B answers: "what works best and why?"

Track B is a battery of six ablations, each testing ONE design decision against the Track A baseline. Each ablation runs on train-400 only unless it outperforms the Track A locked config, in which case it gets a single holdout touch for reporting. This controls holdout usage — we do not run all ablations on holdout.

**Estimated compute:** ~30 hours total across all ablations. Each can be run independently.

---

## Shared infrastructure

All ablations share Track A's Phase 0 artifacts. Additionally, Track B Phase 0 extends activation extraction:

### Track B Phase 0: Extended Activation Extraction (~1 hour)

Track A v2 already extracted activations at {14, 17, 19, 20, 22, 24} and ran probes at all of them. Track B Phase 0 extends this with two additional layers and neuron-level analysis.

1. **Extract activations at layers {26, 27}.**
   
   These are near-terminal layers where your Tier B features (cos_l27_l28, vel_l26_l27) operate. NOT for single-layer steering — for multi-layer pairs in Ablation 2.
   
   Save to `pathway2/track_b/phase0/activations/`:
   - `layer_26.npy`, `layer_27.npy` — each shape `(500, 1536)`

2. **Train linear probes at layers 26 and 27.**
   
   Same protocol as Track A Phase 0.5. Append results to `track_a/phase0/probe_aurocs.json` or save separately to `track_b/phase0/probe_aurocs_extended.json`.

3. **Extract per-neuron contrastive statistics at best-probe layer.**
   
   For the layer with the highest probe AUROC:
   - Compute mean activation per neuron dimension for correct vs. incorrect train-400 problems.
   - Compute the signed difference: `neuron_diff[i] = mean_correct[i] - mean_incorrect[i]`.
   - Identify polarity-flip neurons: dimensions where the sign of the mean activation flips between correct and incorrect sets.
   - Rank neurons by `|neuron_diff[i]|`.
   - Save top-200 neuron indices, their signed differences, and polarity-flip flags.
   
   Save to `track_b/phase0/neuron_analysis.json`.

---

## Ablation 1: Layer Selection (~6 hours)

### Question
Is layer 20 actually the best single layer? How much does the optimal layer vary across problems?

### Protocol

**Part 1: Fixed-layer sweep.**
For each layer in {14, 17, 19, 20, 22, 24}:
- Extract steering vector at that layer (same contrastive pair construction as Track A Phase 1).
- Apply spherical steering at Track A's `best_t`.
- Run on ALL 400 train problems (uniform steering).
- Record overall accuracy, net gain, originally-correct preservation.

**Part 2: Oracle per-problem best layer.**
For each train-400 problem:
- Run steering at all 6 layers independently.
- Record which layer (if any) flips the problem from wrong→right without flipping any previously-correct answer.
- Compute the **per-problem optimal layer distribution**: how much variance is there?

**Analysis:**
- If one layer dominates across problems (>70% of flips come from one layer): fixed-layer is fine. Track A's choice is validated (or should be updated to this layer).
- If optimal layer varies substantially: input-dependent layer selection (W2S-style) is justified. Report the oracle upper bound (what accuracy you'd get if you always picked the best layer per-problem).
- If no layer produces meaningful flips: steering at this strength/method is ineffective. The issue is not layer choice.

**Save to** `track_b/ablation1/`:
- `fixed_layer_comparison.json` — per-layer metrics
- `oracle_per_problem.json` — best layer per problem
- `layer_variance_summary.txt`

---

## Ablation 2: Multi-Layer Steering (~4 hours)

### Question
Do non-adjacent layer pairs compose constructively, as "Small Vectors, Big Effects" predicts?

### Protocol

Extract steering vectors at layers 14, 20, and 24 (if not already done in Ablation 1).

Test the following multi-layer configurations, each with `t_per_layer = best_t / N_layers` (1/N scaling rule from Venhoff et al.):

| Config | Layers | Rationale |
|--------|--------|-----------|
| Single (Track A baseline) | 20 | Baseline |
| Non-adjacent pair A | 14, 20 | Spans early-mid to mid, non-adjacent |
| Non-adjacent pair B | 14, 24 | Wide span |
| Non-adjacent pair C | 20, 26 | Mid to near-terminal |
| Triple | 14, 20, 24 | Full span, non-adjacent |

For each config:
- Apply spherical steering simultaneously at all listed layers, each with its own layer-specific steering vector and `t/N`.
- Run on train-400, record overall accuracy, net gain, originally-correct preservation.

**Key comparison:** Does any multi-layer config beat single-layer (Track A) in net gain?

**Implementation:**
```python
handles = []
for layer_idx, vec in zip(layers, vectors):
    t_scaled = best_t / len(layers)
    hook = make_steering_hook(vec, t_scaled)
    h = model.model.layers[layer_idx].register_forward_hook(hook)
    handles.append(h)
# generate
for h in handles:
    h.remove()
```

**Save to** `track_b/ablation2/`:
- `multilayer_comparison.json`
- `best_multilayer_config.txt`

---

## Ablation 3: Additive vs. Spherical Steering (~3 hours)

### Question
Is spherical steering actually better than additive CAA for this specific model and task?

### Protocol

Using the Track A steering vector at layer 20:

**Additive sweep:** `h' = h + α·v` with α in {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}.

**Spherical sweep:** Already done in Track A. Use those results.

For each additive α:
- Run on train-400 (uniform steering).
- Record overall accuracy, net gain, originally-correct preservation.
- Compute: `||h'||/||h||` averaged across problems (magnitude distortion metric).

**Hook for additive:**
```python
def additive_hook(module, input, output):
    h = output[0]
    modified = h + alpha * steering_vec_tensor.unsqueeze(0).unsqueeze(0)
    return (modified,) + output[1:]
```

**Comparison metrics:**
- Net gain at each method's best strength.
- Magnitude distortion: spherical should be 1.0 by construction; additive will vary.
- Originally-correct preservation: does spherical protect correct answers better?

**Save to** `track_b/ablation3/`:
- `additive_sweep.json`
- `method_comparison.json` — additive vs. spherical head-to-head

---

## Ablation 4: Sparse Neuron-Level Steering (~4 hours)

### Question
Does steering only the top-K most discriminative neurons beat full-layer steering?

### Protocol

Using the neuron analysis from Track B Phase 0:

1. **Construct sparse steering vectors.**
   - For K in {50, 100, 200, 500}:
     - Zero out all dimensions except the top-K by `|neuron_diff|`.
     - Apply polarity-aware filtering: keep the sign of the neuron difference (correct - incorrect), not just magnitude.
     - Normalize to unit length.

2. **Run each sparse vector** at the Track A `best_t`, on train-400.
   Record: overall accuracy, net gain, originally-correct preservation.

3. **Compare against dense (full-1536-dim) Track A vector.**

**Why this matters:** AdaRAS (Dong et al. 2026) found that top-50 neurons with polarity filtering achieve +13% on AIME, and full-layer steering DEGRADES performance. If sparse beats dense here, it means the mean-difference vector contains noise dimensions that actively hurt.

**Save to** `track_b/ablation4/`:
- `sparse_vectors.npz` — K=50, 100, 200, 500 vectors
- `sparse_comparison.json`

---

## Ablation 5: Position-Selective Steering (~3 hours)

### Question
Does restricting steering to problem-region token positions improve results?

### Protocol

Your att-docs Direction 8 found problem-region tokens carry 24.7× stronger topology-difficulty signal than instruction tokens.

1. **Identify position boundary.**
   For Qwen2.5-1.5B-Instruct's chat template, the prompt structure is:
   ```
   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n[PROBLEM TEXT]<|im_end|>\n<|im_start|>assistant\n
   ```
   The problem-region is the tokens between `user\n` and the final `<|im_end|>`. Identify these token position indices for each problem.

2. **Implement position-selective hook.**
   
   **KV cache note:** Position masking only applies during the initial (non-cached) prompt forward pass, where `output[0]` has shape `(1, seq_len, 1536)`. During cached autoregressive generation, `output[0]` is `(1, 1, 1536)` — there's only one position (the current token), so the mask is irrelevant. The hook must detect which pass it's in by checking `h.shape[1]`.
   
   ```python
   def position_selective_hook(module, input, output):
       h = output[0]
       norms = h.norm(dim=-1, keepdim=True)
       h_rotated = slerp(h, steering_vec_expanded, t)
       
       if h.shape[1] > 1:
           # Prompt pass: apply only at problem-region positions
           mask = torch.zeros(h.shape[1], device=h.device, dtype=torch.bool)
           mask[problem_start:problem_end] = True
           modified = torch.where(mask.unsqueeze(0).unsqueeze(-1), h_rotated * norms, h)
       else:
           # Cached generation pass: always steer the current token
           modified = h_rotated * norms
       
       return (modified,) + output[1:]
   ```

3. **Run on train-400** with the Track A `best_t`. Compare:
   - Full-sequence steering (Track A baseline)
   - Problem-region-only steering (this ablation)
   - Generation-only steering (apply only to newly generated token positions, not prompt)

**Save to** `track_b/ablation5/`:
- `position_selective_results.json`
- `position_comparison.txt`

---

## Ablation 6: Mechanistic Analysis (~2 hours, runs on Track A holdout results)

### Question
WHY does steering work (or not)? What happens to the model's internal representations?

### Protocol

This ablation uses Track A's holdout results. It does NOT generate new answers — it analyzes existing ones.

1. **Logit-lens decomposition of the steering vector.**
   ```python
   unembed = model.lm_head.weight  # (151936, 1536)
   logit_lens = unembed @ steering_vector  # (151936,)
   top_promoted = logit_lens.topk(20)
   top_suppressed = logit_lens.topk(20, largest=False)
   ```
   Report the top-20 promoted and suppressed tokens. Mathematical operators, reasoning markers ("Step," "=," digits) = good sign. Random tokens = noise direction.

2. **Topo-confidence shift analysis.**
   
   For holdout problems that received steering in Track A Phase 3 (targeted condition):
   - Re-run the forward pass WITH steering active.
   - Extract hidden states.
   - Re-compute topo-confidence features from the steered hidden states.
   - Compare confidence scores: before steering vs. after steering.
   
   **Circularity check:** Use the **pre-steering snapshot** pattern — extract hidden states from the forward pass BEFORE the steering hook fires. If you're computing confidence from post-steering states, the scores are contaminated (the steering vector modifies the same representations the confidence system reads).
   
   If steering INCREASES topo-confidence on steered problems: the intervention moves the model toward "correct topology" — evidence for causal relevance.
   If steering DECREASES topo-confidence but improves accuracy: topology is correlative, not causal.

3. **Attention-hidden binding score (if `att.llm.attention_binding` is available).**
   
   For 10 representative holdout problems (mix of successful flips, failed flips, unchanged):
   - Compute binding score between attention patterns and hidden states, before and after steering.
   - If steering pushes binding toward 0.68 (easy-problem level, from att-docs Direction 10), the intervention is restoring internal alignment.
   
   This is optional — only run if the ATT binding infrastructure is available and tested.

4. **Effective rank analysis.**
   
   For the same 10 problems:
   - Compute the effective rank (exponential of spectral entropy) of the hidden state matrix at the selected layer L, before and after steering.
   - If steering reduces effective rank: the intervention is collapsing representation diversity (bad sign, especially for a 1.5B model).
   - If steering preserves or increases effective rank: norm-preserving rotation is working as designed.

5. **Intrinsic dimension check (optional).**
   
   Compute intrinsic dimension (TwoNN estimator) of terminal-layer representations (layer 27) for:
   - Unsteered correct problems
   - Unsteered incorrect problems  
   - Steered incorrect problems (that were targeted)
   
   att-docs Direction 9: easy=6.67, hard=12.01. If steering reduces ID of hard problems toward easy-problem levels, it's simplifying the representation in the "right" direction.

6. **Topological compression analysis (Fay et al. 2025).**
   
   Compute PH features (H0 and H1 persistence entropy, total persistence, feature count) on the hidden-state point cloud at the terminal layer, before and after steering:
   - For steered holdout problems: do H1 features increase (steering reverses compression) or stay near zero (steering doesn't affect terminal-layer topology)?
   - If steering at layer L reverses H1 compression at layer 27: the intervention propagates downstream and changes the model's representation structure — strong causal evidence.
   - If H1 stays compressed: the steering effect is local to layer L and doesn't reshape the global computation.
   
   This directly tests the connection between your att-docs finding (H1→0 for hard problems) and the steering intervention.

7. **Anchor verification (STIR concept).**
   
   For TP problems (correct + high confidence):
   - Compute: `projection = dot(layer_L_activation, steering_vector)` for each.
   - These problems should naturally have HIGH projection onto the steering direction (they're already in the "correct" region of activation space).
   - Verify the steering vector has near-zero MARGINAL effect on them — they should already be aligned and rotation should barely change anything.
   - If TP problems have LOW projection: the steering vector may not be capturing "correctness" — it's capturing something else.

8. **vMF strength modulation (optional extension).**
   
   Instead of binary steer/don't-steer, test von Mises-Fisher concentration-based modulation from Spherical Steering (You et al. 2026):
   - For each problem, compute κ = the concentration of the activation around the steering direction.
   - Set `t_per_problem = t_base * (1 - κ/κ_max)` — problems already aligned get light steering, misaligned problems get full strength.
   - Run on train-400 and compare net gain against the binary gating from Track A.
   - This adds one hyperparameter (κ_max) but may improve precision by avoiding over-steering borderline cases.

**Save to** `track_b/ablation6/`:
- `logit_lens_analysis.json` — top-20 promoted/suppressed tokens
- `confidence_shift.json` — before/after scores for steered problems
- `binding_scores.json` (if computed)
- `effective_rank.json`
- `intrinsic_dimension.json` (if computed)
- `topological_compression.json` — H0/H1 features before/after steering
- `anchor_verification.json` — TP projection stats
- `vmf_modulation.json` (if computed)

---

## Synthesis: Choosing the Best Configuration

After all ablations complete, produce a synthesis document:

### `track_b/SYNTHESIS.md`

1. **Layer selection recommendation:**
   - Does any layer beat Track A's probe-selected layer? (Ablation 1)
   - Does multi-layer beat single-layer? By how much? (Ablation 2)
   - How much per-problem layer variance exists? Is input-adaptive selection worth pursuing? (Ablation 1 Part 2)

2. **Steering method recommendation:**
   - Spherical vs. additive: which produces higher net gain with lower magnitude distortion? (Ablation 3)
   - Sparse vs. dense: does neuron selection improve results? At what K? (Ablation 4)
   - Position-selective: does restricting to problem tokens help? (Ablation 5)

3. **Best overall configuration:**
   - Combine the winners from each ablation into a single config.
   - If the best config differs from Track A's locked config, run it on holdout-100 as a SINGLE additional holdout touch.
   - Report: net gain, bootstrap CI, McNemar p-value.

4. **Mechanistic story:**
   - What does the logit lens say about what the steering vector does? (Ablation 6)
   - Does steering restore topological structure or just shift outputs? (Ablation 6)
   - Is there evidence for causal relevance of topology? (Ablation 6 Step 2)

5. **Go/no-go for Pathway 3 (topology-aware fine-tuning):**
   
   - **FULL GO:** Best Track B config achieves net gain ≥ 5 on holdout-100, AND mechanistic analysis shows steering moves representations toward "correct topology." Pathway 3 can use the steering direction as a training signal.
   
   - **CONDITIONAL GO:** Best Track B config achieves net gain ≥ 2, OR Track A alone achieved net gain ≥ 3. Steering works but the effect is modest. Pathway 3 should explore fine-tuning approaches that go beyond additive/rotational perturbation.
   
   - **PIVOT to Pathway 4:** No steering config achieves net gain ≥ 1 on holdout, BUT topo-confidence routing (Pathway 1) still provides 2.5–4.5× lift. Activation steering doesn't work, but the confidence signal is valuable for routing/distillation. Skip Pathway 3, proceed to Pathway 4.
   
   - **NO-GO for steering-based Pathways:** No steering config works AND mechanistic analysis shows the steering vector captures noise, not correctness structure. Re-evaluate the entire arc. The topological signal may be diagnostic (useful for routing) but not interventional (useful for steering/training).

---

## Reference: Papers informing Track B ablations

| Ablation | Key Paper | Finding |
|----------|-----------|---------|
| 1 (Layer selection) | W2S (Gadgil et al. 2026) | Optimal layer varies 3.8–6.5 layers across inputs |
| 1 (Layer selection) | CAA Scaling Laws (Ali et al. 2025) | CAA most effective at early-mid, not terminal layers |
| 2 (Multi-layer) | Small Vectors, Big Effects (Sinii et al. 2025) | Non-adjacent pairs compose constructively |
| 2 (Multi-layer) | Venhoff et al. (ICLR 2025 Workshop) | 1/N scaling rule for multi-layer |
| 3 (Additive vs. spherical) | Spherical Steering (You et al. 2026) | Rotation preserves norms, +10–13% over additive |
| 3 (Additive vs. spherical) | Selective Steering (Dang & Ngo 2026) | Additive causes collapse in <7B models |
| 4 (Sparse neurons) | AdaRAS (Dong et al. 2026) | Top-50 neurons beat full-layer; polarity-aware filtering |
| 5 (Position-selective) | Steering Vector Fields (Li et al. 2026) | Last-token steering most effective; all-token can degrade |
| 5 (Position-selective) | att-docs Direction 8 | Problem tokens 24.7× stronger topology signal |
| 6 (Mechanistic) | Amnesic Probing (Elazar et al. 2021) | Probe accuracy ≠ causal importance |
| 6 (Mechanistic) | Topological Compression (Fay et al. 2025) | H1 compression as failure detection signal |
| 6 (Mechanistic) | Steering Effectiveness (Jafari et al. 2026) | NBF + KL predict steering success |
| 6 (Mechanistic) | STIR (Shi et al. 2026) | Anchor-based gating prevents over-intervention |
