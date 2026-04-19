# topo-confidence

Topological uncertainty estimation for LLMs. Uses persistent homology on hidden-state geometry to predict whether an LLM's output is correct from a single forward pass.

## Current state (April 2026)

All experiments complete through Pathway 6 rebuild + Phase 6.5 deconfounding.

### Key results

| Config | Accuracy | Topo AUROC | Best Baseline | Gap |
|--------|----------|------------|---------------|-----|
| Qwen2.5-1.5B x MATH-500 | 104/500 (20.8%) | **0.796** | vote_margin 0.767 | +0.057 |
| Qwen2.5-1.5B x GSM8K | 871/1319 (66.0%) | 0.615 | neg_entropy 0.741 | -0.126 |
| Qwen2.5-7B x MATH-500 | 348/500 (69.6%) | **0.739** | first_token 0.637 | +0.102 |

**What works:** MATH-500 across model scales. **What doesn't:** GSM8K (baseline wins), cross-benchmark transfer (chance level).

### History of corrections

The initial AUROC was 0.948, later corrected to 0.796 after fixing:
1. Answer extraction false negatives (+47 problems, accuracy 11.4% -> 20.8%)
2. PCA holdout leakage (PCA was fit on all 500, now train-only)
3. Truncation confound (max_new_tokens=256 inflated GSM8K/7B AUROCs)

## Code structure

```
topo_confidence/           # Pip-installable package
pathway1/                  # CORAL feature extraction (78 features, 4 tiers)
  phase0/winning_features.py  # Frozen feature extractor (1407 lines)
  phase1/                  # Original holdout experiment
pathway6_rebuild/          # Corrected pipeline (use this, not pathway1-5)
  common.py                # Shared utilities (answer checking, PCA, generation)
  phase0_relabel/          # Answer extraction fix
  phase1_prompt_model/     # AUROC 0.796 experiment
  phase2_completion/       # Majority vote + gating experiments
  phase3_cross_benchmark/  # GSM8K + 7B (confounded by truncation)
  phase6_5/                # Deconfounded results (max_tokens=1024)
    FINAL_SUMMARY.md       # Complete deconfounded analysis
```

## Running experiments

- **Local (CPU-only analysis):** `python pathway6_rebuild/audit_e2e.py`
- **GPU (RunPod H100):** `bash pathway6_rebuild/phase6_5/runpod_phase6_5.sh`
- **Setup on RunPod:** `bash pathway6_rebuild/runpod_setup.sh`

## Binary data policy

- `.npy`, `.npz`, `.pkl`, `.pt` files are gitignored (too large for git)
- All numeric results committed as JSON
- To regenerate binaries: run the pipeline scripts on GPU

## Key files for understanding results

- `pathway6_rebuild/phase4_report/FINAL_REPORT.json` — corrected vs old comparison
- `pathway6_rebuild/phase6_5/FINAL_SUMMARY.md` — deconfounded cross-benchmark analysis
- `pathway6_rebuild/phase1_prompt_model/experiment9_v2.json` — all baseline comparisons
- `pathway6_rebuild/phase2_completion/experiment1_v2.json` — gating tau sweep

## Dependencies

Python >= 3.10, torch >= 2.0, transformers >= 4.36, ripser >= 0.6, persim >= 0.3, scikit-learn >= 1.3

## Feature pipeline

44 ABC-tier features from the CORAL pipeline:
- **Tier A (9):** Persistent homology (H0/H1 entropy, lifetimes) + geometry (centroid distances, token norms)
- **Tier B (30):** Layer dynamics (inter-layer cosines, PCA spectrum, SVD ratios)
- **Tier C (5):** Depth-2 cross-tier products

Feature extractor frozen at `pathway1/phase0/winning_features.py`. Adding Tier D (34 depth-3+ features) hurts performance.
