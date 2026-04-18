# Pathway 6 Rebuild: Handoff (2026-04-17, updated)

## What This Is

Full pipeline rebuild fixing two critical bugs found in the topo-confidence audit:
1. **Answer extraction**: 47 false negatives in greedy baseline (57/500 → 104/500)
2. **PCA leakage**: Per-completion features used PCA fit on ALL 500 instead of train-400

## What's DONE (runs locally, no GPU)

### Phase 0: Relabel Everything
- `common.py` — Corrected answer checker with balanced-brace `\boxed{}`, sympy-based LaTeX comparison, corrected selection strategies
- `phase0_relabel/step0_validate_checker.py` — 33/33 tests pass
- `phase0_relabel/step1_relabel_all.py` — Relabeled greedy + train temp (12,800) + holdout temp (3,200)
- `phase0_relabel/step2_summary.py` — Foundational counts

**Key results:**
- Greedy: 57/500 → **104/500** (+47 false negatives, 0 false positives)
- Train: 46→89 correct, Holdout: 11→15 correct
- Oracle ceiling holdout: 28→30, Train solvable: 153→137

### Phase 1: Refit Prompt-Level Model
- `phase1_prompt_model/step1_refit_and_experiments.py` — Refit LR + Exps 9, 6, 10

**Key results:**
- Topo AUROC: 0.935 → **0.796**, gap +0.208 → **+0.057** (vs vote_margin)
- neg_entropy: 0.457 → 0.508, ECE: 0.118 → 0.257, Brier: 0.087 → 0.185
- Feature ablation: Tier A still anchor (0.704 alone), A+B+C = 0.796

### Phase 2 CPU Parts
- `phase2_completion/step2_features_and_experiments.py` — Selection strategies with corrected labels
- **Now updated** to auto-detect step1 features and use real per-completion scores when available
- When step1 features are missing, falls back to uniform scores (= MV) as placeholder

**Key results (MV only, weighted vote placeholder until step1 runs):**
- Ungated MV: +12 (14 W→R, 2 R→W)
- Gated MV (tau=0.3): +4 (4 W→R, 0 R→W)
- Optimal tau ≈ 0.5-0.6

### Phase 4: Comparison Report
- `phase4_report/generate_report.py` — Full comparison table + claim assessment

## What's WRITTEN (needs RunPod H100)

### Phase 2 Step 1: Per-Completion Trajectory Extraction
- Script: `phase2_completion/step1_extract_trajectories.py`
- Extracts trajectories for ~311 train + ~85 holdout wrong-greedy × 32 completions
- Fits PCA on train-400 only, computes topo features
- Checkpoints every 50 problems
- **Estimated: ~30-60 min on H100**
- After this, re-run step2 to get real weighted vote + Experiment 5

### Phase 3 Step 1: GSM8K × 1.5B Full Pipeline
- Script: `phase3_cross_benchmark/step1_gsm8k_full.py`
- Complete pipeline: greedy (chat template) → trajectories → features → model → temp sampling → selection
- Uses `format_prompt_chat(q, tokenizer, template="gsm8k")` for chat template
- Uses `check_correct_v2()` and `select_*_v2()` throughout
- PCA on train-80% only
- Checkpoints at each section (greedy/100, traj/200, temp/50, comp_traj/50)
- **Estimated: ~2-3 hours on H100**
- Expected accuracy: ~73% (was 40.9% without chat template)
- Output: `phase3_cross_benchmark/gsm8k/`

### Phase 3 Step 2: MATH-500 × 7B Full Pipeline
- Script: `phase3_cross_benchmark/step2_math7b_full.py`
- Same structure as step1 but with Qwen2.5-7B-Instruct (bf16, device_map="auto")
- hidden_dim = 3584, same 400/100 train/holdout split as 1.5B
- **Estimated: ~2-3 hours on H100**
- Expected accuracy: ~75-77% (was 20.6% without chat template)
- Output: `phase3_cross_benchmark/math7b/`

### Phase 3 Step 3: Cross-Benchmark Analysis (CPU)
- Script: `phase3_cross_benchmark/step3_cross_analysis.py`
- Loads results from step1, step2, and Phase 1 rebuild
- Feature distribution comparison (Cohen's d, rank correlations)
- Transfer experiments (train on one model/benchmark, test on another)
- Final comparison table: 1.5B×MATH vs 1.5B×GSM8K vs 7B×MATH
- Output: `phase3_cross_benchmark/cross_analysis.json`

## RunPod Execution Order

```
1. cd ~/topo-confidence/pathway6_rebuild

2. Phase 2 GPU (30-60 min):
   python phase2_completion/step1_extract_trajectories.py

3. Phase 2 CPU update (re-run with real scores):
   python phase2_completion/step2_features_and_experiments.py

4. Phase 3 GSM8K (2-3 hours):
   python phase3_cross_benchmark/step1_gsm8k_full.py

5. Phase 3 MATH 7B (2-3 hours):
   python phase3_cross_benchmark/step2_math7b_full.py

6. Phase 3 cross-analysis (CPU, 5 min):
   python phase3_cross_benchmark/step3_cross_analysis.py

7. Phase 4 update (regenerate report):
   python phase4_report/generate_report.py
```

Steps 4+5 are independent of step 3 (only need Phase 0 outputs + common.py).
Steps 4 and 5 can run sequentially on same GPU.

## Artifacts Produced

```
pathway6_rebuild/
  common.py                                          # Corrected utilities
  HANDOFF.md                                         # This file
  phase0_relabel/
    baseline_correct_v2.npy                          # (500,) bool
    temperature_generations_v2.json                  # 400×32 relabeled
    holdout_temperature_generations_v2.json          # 100×32 relabeled
    label_diff_report.json
    foundational_counts.json
    checker_validation.txt
  phase1_prompt_model/
    holdout_metrics_v2.json                          # AUROC=0.796
    model_v2.pkl
    per_problem_scores_v2.json
    experiment9_v2.json, experiment6_v2.json, experiment10_v2.json
  phase2_completion/
    holdout_results_v2.json                          # MV selection
    experiment1_v2.json                              # Tau sweep
    experiment8_v2.json                              # Adaptive sampling
    step1_extract_trajectories.py                    # READY for RunPod
    step2_features_and_experiments.py                # UPDATED: auto-detects step1 features
    --- After RunPod step1 ---
    train_completion_features_v2.npz                 # From step1
    holdout_completion_features_v2.npz               # From step1
    greedy_pca_train400.pkl                          # From step1
    feature_names_v2.json                            # From step1
    experiment5_v2.json                              # From step2 re-run
    completion_scorer_v2.pkl                         # From step2 re-run
  phase3_cross_benchmark/
    step1_gsm8k_full.py                              # READY for RunPod
    step2_math7b_full.py                             # READY for RunPod
    step3_cross_analysis.py                          # READY (CPU, after step1+2)
    --- After RunPod ---
    gsm8k/
      greedy_answers.json, baseline_correct.npy
      trajectories.npz, layer_states.npy
      features.npy, feature_names.json, topo_scores.npy
      model.pkl, model_results.json
      temperature_generations.json
      completion_features.npz
      selection_results.json, summary.json
    math7b/
      (same structure as gsm8k/)
    cross_analysis.json
  phase4_report/
    generate_report.py
    FINAL_REPORT.json
```

## Critical Design Decisions

1. **common.py** re-exports non-buggy functions from old code via importlib. NEVER import `extract_answer`, `normalize_answer`, `check_correct` from old code.
2. **SSS ordering fix** is encapsulated in `load_features_with_sss_fix()`. Uses OLD labels for SSS split reconstruction.
3. **Selection strategies** (`select_*_v2` functions) in common.py use corrected checker.
4. **Phase 2 step2 auto-detects step1 features** — runs with placeholder (uniform scores) if step1 hasn't run yet, automatically uses real per-completion topo scores if step1 artifacts exist.
5. **Greedy MATH-500 1.5B trajectories NOT regenerated** with chat template. Chat template only for GSM8K/7B in Phase 3.
6. **Phase 3 scripts use `format_prompt_chat()`** which calls `tokenizer.apply_chat_template()` — this is what fixes the implausible accuracies.
7. **Phase 3 scripts use variable-dim trajectory extraction** from pathway5 (`extract_trajectory`, `extract_completion_trajectory`) — handles both 1536 (1.5B) and 3584 (7B) hidden dims.
8. **Phase 3 PCA fitted on train split only** for each benchmark (80% for GSM8K, 400/500 for 7B MATH).

## Corrected Numbers (1.5B × MATH-500)

| Metric | Old | New |
|--------|:---:|:---:|
| Greedy accuracy | 57/500 (11.4%) | 104/500 (20.8%) |
| Holdout correct | 11/100 | 15/100 |
| Topo AUROC (holdout) | 0.935 | **0.796** |
| Topo vs best baseline | +0.208 | **+0.057** (vs vote_margin) |
| Ungated MV | +8 | **+12** |
| Gated MV (tau=0.3) | +10 (0 R→W) | **+4** (0 R→W) |
| ECE | 0.118 | 0.257 |

## Narrative Summary

The headline AUROC drops from 0.935 to **0.796** — still above chance and above the best baseline (vote_margin 0.767), but the gap shrinks from +0.208 to **+0.057**. The logprob anti-correlation narrative is gone. However:

- **Topo still wins** over all baselines
- **0 R→W at tau=0.3** survives
- **Adaptive sampling** still works: 89% savings at +3
- **Ungated MV improved**: +12 vs old +8
- **Paper needs reframing** from "dominant signal" to "complementary, safe signal"
- **GSM8K/7B results pending** — expected to show chat template fixes accuracy
