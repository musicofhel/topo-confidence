# Pathway 4 Handoff — 2026-04-14

## What was done

All 8 implementation scripts for Pathway 4 are **written and validated** (imports resolve, syntax clean, core logic tested). No GPU runs have been started yet.

### Files created (2,994 lines total)

```
pathway4/
  track_a/
    common.py          (377 lines) — shared utils, trajectory extraction, PCA monkey-patching, 4 selection strategies, CV folds
    phase1_score.py    (349 lines) — score 354×32 temperature completions via topo features
    phase2_cv.py       (371 lines) — 5-fold CV evaluation of selection strategies
    phase3_holdout.py  (433 lines) — holdout-100 generation + scoring + go/no-go
    phase1/            (empty, artifacts go here)
    phase2/            (empty)
    phase3/            (empty)
  track_b/
    common.py          (397 lines) — differentiable bias hook, log prob, KL penalty, GRPO utils
    phase0_pilot.py    (353 lines) — 50-step pilot validation (gradient flow, reward signal)
    phase1_train.py    (351 lines) — full 200-step GRPO training loop
    phase2_evaluate.py (363 lines) — evaluate trained bias, comparison table, go/no-go
    checkpoints/       (empty)
```

### Plan file

Full plan at `~/.claude/plans/prancy-waddling-zephyr.md` — includes context, design decisions, artifact paths, verification checklist.

## What to run next

### Step 1: Track A Phase A1 (~2hr GPU)

```bash
cd ~/topo-confidence/pathway4/track_a && python3 phase1_score.py
```

This does 11,328 forward passes (354 wrong-greedy problems × 32 completions), extracts per-token trajectories, computes 78 topo features via ripser, and scores with a completion-level confidence model. Checkpoints every 50 problems to `phase1/trajectories_checkpoint.json` — safe to interrupt and resume.

**Produces**: `phase1/completion_scores.json`, `phase1/completion_features.npz`, `phase1/feature_names.json`, `phase1/phase1_summary.json`

**Key metric to check**: `phase1_summary.json` → `max_confidence_accuracy`. If the max-confidence completion is correct significantly more than random (~12% = 4.4/32), topo-confidence CAN differentiate between completions. If it's near random, the prompt+completion trajectory doesn't carry enough signal and Track A won't work.

### Step 2: Track A Phase A2 (~10min CPU)

```bash
cd ~/topo-confidence/pathway4/track_a && python3 phase2_cv.py
```

Evaluates 4 strategies: random, max-confidence, majority vote, confidence-weighted vote. Creates new 5-fold splits over train-400 (NOT reusing Pathway 3 folds which span all 500).

**Produces**: `phase2/cv_results.json`, `phase2/strategy_comparison.json`, `phase2/locked_strategy.json`

**Key metric**: `locked_strategy.json` → `mean_net_gain`. If >> 3, proceed to Phase A3. If ≤ 3, skip Phase A3 and go to Track B.

### Step 3: Track A Phase A3 (~6hr GPU)

```bash
cd ~/topo-confidence/pathway4/track_a && python3 phase3_holdout.py
```

Generates 32 temperature completions for holdout-100 (~5hr), extracts trajectories, scores, applies locked strategy. This is the **single holdout touch** for Track A.

**Produces**: `phase3/holdout_results.json`, `phase3/go_no_go_decision.json`

**Go/no-go**: `holdout_results.json` → `net_gain ≥ 5 AND bayesian_p_improvement > 0.90` = PUBLICATION_READY. Note: R→W is always 0 by design (selection never overrides correct greedy answers).

### Step 4: Track B Phase B0 (~2.5hr GPU)

```bash
cd ~/topo-confidence/pathway4/track_b && python3 phase0_pilot.py
```

Validates: (1) Qwen2DecoderLayer returns plain tensor, (2) zero bias = identity, (3) gradient flows to bias param. Then runs 50 training steps.

**Produces**: `pilot_report.json`

### Step 5: Track B Phase B1 (~10hr GPU)

```bash
cd ~/topo-confidence/pathway4/track_b && python3 phase1_train.py
```

Full GRPO: 200 steps, batch=4 problems, G=8 rollouts, lr=5e-4, KL penalty β=0.01. Checkpoints every 25 steps, quick eval every 50 steps. Resumes from latest checkpoint if interrupted.

**Produces**: `trained_bias.pt`, `training_log.json`, `checkpoints/step_*.pt`

### Step 6: Track B Phase B2 (~45min GPU)

```bash
cd ~/topo-confidence/pathway4/track_b && python3 phase2_evaluate.py
```

Evaluates trained bias on train-400 + holdout-100, both ungated and confidence-gated (τ=0.40). Produces comparison table across all methods.

**Produces**: `holdout_results.json`, `comparison.json`, `go_no_go_decision.json`

## Critical design decisions baked in

1. **Completion-level features, not prompt-level**: The existing topo-confidence model scores PROMPTS (identical for all 32 completions). Track A runs forward passes on `prompt + completion_text` to get trajectory-dependent features. This is the key insight that makes selection possible.

2. **Pre-fitted PCA from greedy trajectories**: Uses the `PreFittedPCA` monkey-patching pattern from `pathway2/track_a/phase0_baseline.py:270-295`. Ensures stable PCA subspace. Without this, PCA refitting on 32 same-problem completions would be noisy.

3. **New CV folds over train-400 only**: The Pathway 3 folds at `pathway3/phase3/fold_assignments.json` span all 500 problems including holdout-100. Track A creates new StratifiedKFold(5, seed=42) over just the 400 train problems to avoid holdout contamination.

4. **Additive bias hook (Track B)**: Uses simple `output + bias` instead of slerp. Slerp is non-differentiable at zero initialization. Additive bias is the Bias-Only Adaptation standard (Sinii et al. 2025). Confirmed: `Qwen2DecoderLayer.forward()` returns plain tensor in transformers 4.57.6.

5. **KL approximation via bias norm**: Track B uses `β * ||bias||²` as a KL proxy instead of double forward passes. This saves ~50% compute per step and correlates well with actual KL for small biases.

## Key numbers

- Train: 46 correct, 354 wrong (out of 400)
- Holdout: 11 correct, 89 wrong (out of 100)
- Oracle ceiling (train): +153 (43% of wrong problems have ≥1 correct completion in 32 samples)
- Mean correct completions per solvable problem: 4.4/32
- Track A contrastive steering holdout: +3 net gain (6 W→R, 3 R→W)
- Publication threshold: holdout net gain ≥ 5 AND P(improvement) > 0.90

## If something breaks

- **Import errors**: Scripts use `sys.path.insert(0, '.')` pattern. Must `cd` into the script's directory before running.
- **VRAM OOM in Phase A1**: Reduce batch processing. The script processes one completion at a time (no batching), so OOM is unlikely. If ripser OOM on 11,328 trajectories, process in chunks by modifying `extract_features_with_prefitted_pca()` call.
- **VRAM OOM in Track B**: Reduce `GRPO_GROUP_SIZE` from 8 to 4 in `track_b/common.py`. Or reduce `MAX_LENGTH` from 512 to 384.
- **Phase A1 checkpoint resume**: If interrupted, it resumes from `phase1/trajectories_checkpoint.json`. If resume fails (corrupted checkpoint), delete all `*_partial.*` and `*_checkpoint.*` files in `phase1/` and restart.
- **Phase B1 checkpoint resume**: Automatically finds latest `checkpoints/step_*.pt` and resumes. Delete `checkpoints/` to restart from scratch.
