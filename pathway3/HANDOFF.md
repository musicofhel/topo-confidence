# Pathway 3 Handoff — 2026-04-13

## Status: IMPLEMENTATION COMPLETE, READY TO RUN

All 4 phase scripts are written, imports verified, utility functions smoke-tested. No GPU execution has happened yet.

## Run Order

```bash
cd ~/topo-confidence/pathway3
python phase1_expansion.py   # ~5-8h GPU (overnight)
python phase2_prototypes.py  # ~1h CPU
python phase3_cv.py          # ~10h GPU
python phase4_holdout.py     # ~3h GPU
```

## What Was Built

| File | Lines | Purpose |
|------|-------|---------|
| `common.py` | 675 | Shared utilities extending Track A: multi-layer hooks, prototype steering, soft sigmoid gating, bootstrap/PCA/sparsify, K-means clustering, confidence pipeline for arbitrary CV folds, statistical tests (mid-p McNemar, Bayesian P(improvement), BCa bootstrap CI) |
| `phase1_expansion.py` | 371 | Temperature sampling (32 completions/problem at T=0.8), activation extraction at layers {15,20,24,25}, checkpoint/resume every 50 problems |
| `phase2_prototypes.py` | 421 | Bootstrap averaging (B=20), PCA projection (k∈{5,10,20,50}), sparsification (d∈{50,100,200,500}), K-means prototypes (K∈{3,5,8}), multi-layer config table |
| `phase3_cv.py` | 829 | 5-fold stratified CV, 4 config classes (Config1=Track A baseline, Config2=expanded data, Config4=denoised+multi-layer, Config6=full stack with prototypes+soft gating), sigmoid sharpness sweep on fold 0, per-fold checkpointing |
| `phase4_holdout.py` | 737 | Holdout-100 evaluation with locked config, Track A comparison, per-problem flip analysis, mechanistic analysis (prototype selection, multi-layer ablation), go/no-go decision |

## Prompt Document

`~/topo-confidence/pathway3_v1.md` — full Pathway 3 specification. Updated with YaPO (arXiv:2601.08441) as fallback in Phase 2 Step 2, PIVOT option (d), and Key references.

## Key Design Decisions

- **importlib** loads Track A's `common.py` to avoid circular import (pathway3/common.py would shadow it via sys.path)
- **Checkpoint/resume**: Phase 1 saves every 50 problems; Phase 3 saves per config×fold
- **4 config classes** in Phase 3 as OOP hierarchy, each implementing `prepare_fold()` and `should_steer()`
- **Full pipeline per fold** — PCA, scaler, LR, vectors, prototypes all fit on fold's train set only (no leakage)
- **Priority configs**: 1, 2, 4, 6 (configs 3 and 5 skipped per prompt's compute budget guidance)
- Default multi-layer: `non_adjacent_a` = layers [15, 25]; default prototype K=5, PCA k=10, sparse d=100

## Artifacts From Track A (consumed by Pathway 3)

- `pathway2/track_a/phase0/baseline_correct.npy` — (500,) correctness
- `pathway2/track_a/phase0/baseline_answers.json` — 500 generated answers
- `pathway2/track_a/phase0/activations/layer_{14,17,19,20,22,24}.npy` — (500, 1536) each
- `pathway2/track_a/phase0/confidence_scores.npy` — (500,) P(correct)
- `pathway2/track_a/phase1/steering_vector.npy` — (1536,) Track A vector
- `pathway2/track_a/phase2/locked_config.json` — t=0.15, τ=0.40, layer=24
- `pathway2/track_a/phase3/holdout_results.json` — +3 net gain on holdout
- `pathway1/phase1/holdout_mask_seed9999.npy` — sacred holdout (100 problems)
- `pathway1/phase1/features_train400.npy`, `features_holdout100.npy` — cached CORAL features

## Expected Outputs (after running)

```
pathway3/
├── phase1/
│   ├── temperature_generations.json      # 32 completions × 400 problems
│   ├── expansion_summary.json            # tallied contrastive set
│   ├── expanded_activations/             # correct problem activations at 4 layers
│   └── additional_activations/           # layers 15, 25 for all 500
├── phase2/
│   ├── phase2_summary.json
│   ├── denoised_vectors/                 # bootstrap, PCA, sparse variants per layer
│   └── prototypes/                       # K-means centroids per layer×K
├── phase3/
│   ├── fold_assignments.json             # 5-fold split
│   ├── cv_results.json                   # aggregated results per config
│   ├── locked_config.json                # selected best config
│   └── checkpoint_*.json                 # per config×fold
└── phase4/
    ├── holdout_results.json
    ├── per_problem_analysis.json + .txt
    ├── comparison_with_track_a.txt
    ├── statistical_tests.json
    ├── go_no_go_decision.json
    └── mechanistic/                      # if improved over Track A
```

## Go/No-Go Thresholds

- **FULL GO** → Pathway 4: CV mean ≥ 5 AND holdout ≥ 5
- **CONDITIONAL GO**: holdout ≥ 3 and improves over Track A's +3
- **PUBLICATION STOP**: CV ≥ 3, holdout ≥ 3, Bayesian P > 0.90
- **PIVOT**: no improvement → consider PRA, Bias-Only Adaptation, YaPO, or routing-only publication
