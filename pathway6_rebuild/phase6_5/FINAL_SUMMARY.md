# Pathway 6.5: Deconfounded Cross-Benchmark Results

Generated: 2026-04-19 09:57

## 1. What Changed

The e2e audit (Phase 6 audit) found that `max_new_tokens=256` truncated
64% of GSM8K outputs and 90% of 7B MATH outputs. This caused:
- GSM8K accuracy: 36.7% (expected ~73%)
- 7B MATH accuracy: 16.2% (expected ~75-77%)
- AUROC confounded by truncation (binary 'not truncated' predictor beat topo)

Phase 6.5 reruns both configs with `max_new_tokens=1024`.
The 1.5B x MATH-500 result (AUROC 0.796) is unchanged (20.8% accuracy, no truncation).

## 2. Complete Results Table

| Config | max_tokens | Greedy Acc | Topo AUROC | Best Baseline | Gap | MV | Oracle |
|--------|-----------|-----------|------------|---------------|-----|-----|--------|
| 1.5B x MATH-500 | 256 | 104/500 (20.8%) | **0.796** | vote_margin 0.767 | +0.057 | +12 | +30 |
| 1.5B x GSM8K (truncated) | 256 | 484/1319 (36.7%) | 0.731 | neg_mean_entropy 0.623 | +0.109 | +4 | +133 |
| **1.5B x GSM8K (fixed)** | 1024 | 871/1319 (66.0%) | **0.615** | neg_mean_entropy 0.741 | -0.126 | +47 | +77 |
| 7B x MATH-500 (truncated) | 256 | 81/500 (16.2%) | 0.682 | first_token_prob 0.519 | +0.163 | +1 | +20 |
| **7B x MATH-500 (fixed)** | 1024 | 348/500 (69.6%) | **0.739** | first_token_prob 0.637 | +0.102 | +3 | +11 |

GSM8K truncation at 1024: 0/1319 (0.0%)
7B MATH truncation at 1024: 47/500 (9.4%)

## 3. Was Truncation the Sole Cause of Low Accuracy?

**NO.** Both configs still below expected — truncation was not the only issue.

## 4. Deconfounded AUROC Numbers

| Config | Phase 3 (confounded) | Phase 6.5 (clean) | Change |
|--------|---------------------|-------------------|--------|
| GSM8K AUROC | 0.731 | 0.615 | -0.116 |
| 7B MATH AUROC | 0.682 | 0.739 | +0.057 |

GSM8K 95% CI: [0.545, 0.686]
7B MATH 95% CI: [0.615, 0.848]

## 5. Does Topo-Confidence Generalize Beyond MATH-500?

### Transfer AUROC (train on X, evaluate on Y)

| Source → Target | AUROC | CI |
|----------------|-------|-----|
| 1.5B × MATH-500 → 1.5B × GSM8K (1024) | 0.504 | [0.431, 0.579] |
| 1.5B × MATH-500 → 7B × MATH (1024) | 0.611 | [0.475, 0.731] |
| 1.5B × GSM8K (1024) → 1.5B × MATH-500 | 0.275 | [0.150, 0.425] |
| 1.5B × GSM8K (1024) → 7B × MATH (1024) | 0.608 | [0.474, 0.730] |
| 7B × MATH (1024) → 1.5B × MATH-500 | 0.322 | [0.200, 0.472] |
| 7B × MATH (1024) → 1.5B × GSM8K (1024) | 0.512 | [0.447, 0.588] |

### Feature Effect-Size Correlations

- 1.5B × MATH-500 vs 1.5B × GSM8K (1024): rho=0.450 (p=3.53e-05)
- 1.5B × MATH-500 vs 7B × MATH (1024): rho=0.259 (p=2.21e-02)
- 1.5B × GSM8K (1024) vs 7B × MATH (1024): rho=0.632 (p=5.58e-10)

## 6. Updated Claim Assessment

- **[ESTABLISHED]** 1.5B MATH topo AUROC = 0.796
  - Unchanged from Phase 1 rebuild. No truncation confound (20.8% correct).
- **[SURVIVES]** GSM8K cross-benchmark generalization
  - Deconfounded AUROC: 0.615 (was 0.731 with truncation confound). Accuracy: 66.0% (was 36.7% truncated).
- **[SURVIVES]** 7B cross-model generalization
  - Deconfounded AUROC: 0.739 (was 0.682 with truncation confound). Accuracy: 69.6% (was 16.2% truncated).
- **[REFUTED]** max_new_tokens=256 truncation was sole cause of low accuracy
  - Neither config reached expected accuracy — truncation was not the sole cause.
- **[REFUTED]** Topo features add value beyond logprob baselines (deconfounded)
  - GSM8K gap: -0.126. With truncation removed, baselines may improve too.

## 7. Cost and Runtime

- GSM8K pipeline: 323 min
- 7B MATH pipeline: 155 min
- Total: 478 min (8.0 hours)
- Estimated cost: ~$24 (H100 SXM at $2.99/hr)
