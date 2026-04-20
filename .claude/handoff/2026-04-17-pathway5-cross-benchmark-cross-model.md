# Handoff: Pathway 5 — Cross-Benchmark and Cross-Model Generalization

**Date**: 2026-04-17
**Session scope**: Implement all scripts for Pathway 5 (3 tracks, 10 scripts)

## What was done

Created the complete implementation for Pathway 5: 10 Python scripts across 3 tracks, plus shared utilities. All scripts are ready for RunPod execution. No GPU was available locally, so no results yet.

## File structure

```
pathway5/
├── common.py                              (410 lines) — shared utilities
├── track_a/                               (GSM8K replication, 1.5B)
│   ├── phase_a1_extraction.py             (GPU: greedy + trajectories + logprobs)
│   ├── phase_a2_features_and_model.py     (CPU: features + confidence model + baselines)
│   ├── phase_a3_sampling.py               (GPU: temperature sampling + completion trajectories)
│   └── phase_a4_selection.py              (CPU: per-completion scoring + all strategies)
├── track_b/                               (7B cross-model, MATH-500)
│   ├── phase_b1_extraction.py             (GPU: greedy + trajectories + logprobs)
│   ├── phase_b2_features_and_model.py     (CPU: features + confidence model + baselines)
│   ├── phase_b3_sampling.py               (GPU: temperature sampling + completion trajectories)
│   └── phase_b4_selection.py              (CPU: per-completion scoring + all strategies)
└── track_c/                               (CPU-only cross-model comparison)
    └── cross_model_analysis.py            (feature distribution + transfer + distillation)
```

## Run order

### Track A (GSM8K, ~$10-20, ~6-10 H100-hours)
```bash
# On RunPod H100:
cd ~/topo-confidence
python pathway5/track_a/phase_a1_extraction.py    # ~1-2 hrs GPU
python pathway5/track_a/phase_a2_features_and_model.py  # ~1-4 hrs CPU (can run locally)
python pathway5/track_a/phase_a3_sampling.py       # ~2-4 hrs GPU
python pathway5/track_a/phase_a4_selection.py      # ~30 min-2 hrs CPU (can run locally)
```

### Track B (7B, ~$20-35, ~8-12 H100-hours)
```bash
python pathway5/track_b/phase_b1_extraction.py    # ~2-3 hrs GPU
python pathway5/track_b/phase_b2_features_and_model.py  # ~1-4 hrs CPU
python pathway5/track_b/phase_b3_sampling.py       # ~4-6 hrs GPU
python pathway5/track_b/phase_b4_selection.py      # ~30 min-2 hrs CPU
```

### Track C (CPU only, requires Track B completed)
```bash
python pathway5/track_c/cross_model_analysis.py    # ~10 min CPU
```

## Key design decisions

### 1. Same feature pipeline, different PCA
The 78-feature `winning_features.py` pipeline works for any hidden_dim because PCA projects to 45-D regardless of input dimension. 7B (3584-D) and 1.5B (1536-D) both get projected to the same 45-D PCA space. PCA is fit fresh per model/dataset (on train split only) to prevent data leakage.

### 2. Transfer PCA experiment (Track A only)
Phase A2 runs features twice: once with fresh PCA (fit on GSM8K train), once with transfer PCA (fit on MATH-500 train). If transfer AUROC ≈ fresh AUROC, the PCA projection is benchmark-universal. Only possible for same-model transfers (1.5B→1.5B) since hidden_dim must match.

### 3. GSM8K answer format
GSM8K uses `#### [number]` format. The existing `extract_answer()` already handles `####` as its first pattern. `extract_gsm8k_ground_truth()` in common.py handles comma-separated numbers in ground truths. `normalize_answer()` strips commas during comparison.

### 4. 7B uses bf16 (not fp16)
Loaded with `torch.bfloat16` for better numerical stability at the 7B scale. 1.5B uses fp16 for consistency with existing experiments.

### 5. Checkpoint strategy
All GPU scripts checkpoint every 50-200 problems. Trajectory extraction checkpoints to .npz files. Temperature generations checkpoint to .json. All scripts resume transparently from checkpoints.

### 6. Holdout-only sampling for 7B
Track B Phase B3 only generates temperature samples for wrong-greedy holdout problems (not the full 400 train). This saves ~70% compute since 7B gets ~75-77% correct on MATH-500.

### 7. Track C transfer experiment
Trains LR on 1.5B features → evaluates on 7B features (and vice versa). Uses StandardScaler fit on training model's data, applied to test model's features. This tests whether the "shape of correctness" in topo-feature space transfers across model scales.

## Dependencies

| Script | Requires | GPU? | ~Time |
|--------|----------|------|-------|
| A1 | GSM8K dataset | Yes | 1-2 hrs |
| A2 | A1 output | No | 1-4 hrs |
| A3 | A1, A2 output | Yes | 2-4 hrs |
| A4 | A1-A3 output | No | 0.5-2 hrs |
| B1 | MATH-500 dataset | Yes | 2-3 hrs |
| B2 | B1 output | No | 1-4 hrs |
| B3 | B1 output | Yes | 4-6 hrs |
| B4 | B1-B3 output | No | 0.5-2 hrs |
| C | Pathway 1 + B1, B2 | No | 10 min |

## Expected outputs per phase

### Phase A1
- `gsm8k_baseline_answers.json` — per-problem greedy answers
- `gsm8k_baseline_correct.npy` — (1319,) boolean correctness
- `gsm8k_trajectories.npz` — 1319 token trajectories (variable length × 1536)
- `gsm8k_layer_states.npy` — (1319, 29, 1536)
- `gsm8k_entropy_scores.json` — per-problem logprob features

### Phase A2
- `gsm8k_features_fresh_pca.npy` — (1319, 78)
- `gsm8k_features_transfer_pca.npy` — (1319, 78) using MATH-500 PCA
- `gsm8k_topo_scores_fresh.npy` — (1319,) P(correct)
- `phase_a2_results.json` — AUROC, baselines, comparison

### Phase A3
- `temperature_generations.json` — test-split × 32 completions
- `completion_trajectories.npz` — per-completion trajectories
- `gsm8k_sampling_baselines.json` — vote_margin, p_majority, etc.

### Phase A4
- `completion_features.npy`, `completion_correct.npy`, `completion_problem_ids.npy`
- `phase_a4_results.json` — all strategies, gating, final comparison table

### Phase B1-B4
Same pattern as A1-A4 but with `math500_7b_` prefix, hidden_dim=3584.

### Track C
- `cross_model_results.json` — distribution comparison, transfer AUROC, distillation verdict
- `feature_comparisons.json` — per-feature cross-model analysis

## Go/no-go framework

| Outcome | Condition | Paper target |
|---------|-----------|-------------|
| STRONG | GSM8K AUROC > 0.85 AND 7B AUROC > 0.85 | NeurIPS main track |
| PARTIAL | One > 0.85, other 0.70-0.85 | EMNLP or NeurIPS workshop |
| NARROW | Both < 0.85 or one < 0.70 | Workshop or TMLR |

## Budget estimate

| Track | H100-hours | Cost |
|-------|-----------|------|
| A (GPU phases) | 3-6 | $10-20 |
| B (GPU phases) | 6-10 | $20-35 |
| C (CPU only) | 0 | $0 |
| **Total** | **9-16** | **$30-55** |

## What's next

1. **Start with Track A** — cheaper, faster, GSM8K is the standard self-consistency benchmark
2. **Phase A2 go/no-go**: If GSM8K AUROC < 0.70, reconsider Track B priority
3. **Track B**: Run after Track A results inform expectations
4. **Track C**: Run immediately after Track B Phase B2 completes (needs both models' features)
5. **Final synthesis**: Build 3-column comparison table (1.5B×MATH, 1.5B×GSM8K, 7B×MATH)
