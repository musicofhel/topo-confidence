# Phase 6.5: Truncation Fix — RunPod Handoff

## What This Is

The e2e audit found that `max_new_tokens=256` (inherited from `pathway5/common.py`) truncates 64% of GSM8K outputs and 90% of 7B MATH outputs. Non-truncated accuracy is 76.7% (GSM8K) and 92.2% (7B) — the models work fine, they just run out of tokens.

The cross-benchmark AUROC numbers (0.731 GSM8K, 0.682 7B) are **confounded** — simply predicting "not truncated" gives higher AUROC than topo features. Phase 6.5 reruns both with `max_new_tokens=1024` to get clean numbers.

The 1.5B x MATH-500 result (**AUROC 0.796**) is unaffected and does NOT need rerunning.

## Files Created

```
pathway6_rebuild/phase6_5/
├── step1_gsm8k_1024.py      — GSM8K × 1.5B, max_new_tokens=1024 (GPU)
├── step2_math7b_1024.py      — MATH-500 × 7B, max_new_tokens=1024 (GPU)
├── step3_cross_analysis.py   — Cross-benchmark comparison (CPU)
├── step4_summary.py          — FINAL_SUMMARY.md generation (CPU)
├── runpod_phase6_5.sh        — Master execution script
└── HANDOFF.md                — This file
```

## How to Run on RunPod

1. Start an H100 SXM pod ($2.99/hr). Estimated 4-6 hours total.

2. Setup (if not already done):
   ```bash
   cd ~/topo-confidence
   git pull
   bash pathway6_rebuild/runpod_setup.sh
   ```

3. Run:
   ```bash
   bash pathway6_rebuild/phase6_5/runpod_phase6_5.sh
   ```

4. Resume from a specific step if interrupted:
   ```bash
   bash pathway6_rebuild/phase6_5/runpod_phase6_5.sh --from 2  # skip GSM8K, start at 7B
   ```

## Key Design Decisions

1. **Trajectory reuse**: Greedy trajectories are prompt-based (forward pass on the question, not the answer). They're identical regardless of `max_new_tokens`. Scripts attempt to load Phase 3 trajectories first, falling back to extraction if not found. This saves ~1 hour per benchmark.

2. **Train/test split changes**: GSM8K uses StratifiedShuffleSplit with labels as the stratification variable. Different labels (from 1024 vs 256) = different indices. This is correct — we want proper stratification. PCA and features are recomputed for the new split.

3. **7B MATH split is fixed**: Uses the same 400/100 holdout mask as 1.5B (seed 9999). This is deterministic and does NOT change with labels.

4. **Temperature completions**: Regenerated entirely because (a) completions change with 1024 tokens, (b) fewer wrong-greedy problems need sampling (higher accuracy → fewer wrong), (c) per-completion trajectories depend on completion text.

5. **Checkpointing**: Every section checkpoints progress (same pattern as Phase 3). Safe to kill and restart.

## Expected Results

| Metric | Phase 3 (truncated) | Phase 6.5 (expected) |
|--------|--------------------|--------------------|
| GSM8K accuracy | 36.7% | ~73-77% |
| GSM8K topo AUROC | 0.731 (confounded) | TBD (deconfounded) |
| 7B MATH accuracy | 16.2% | ~75-77% |
| 7B MATH topo AUROC | 0.682 (unreliable) | TBD (deconfounded) |

The key question is whether topo AUROC holds up when truncation can't inflate it. The audit's non-truncated subset analysis showed GSM8K dropping from 0.731 to 0.627 — but that was on a small, non-representative subset. Phase 6.5 gives the definitive answer.

## After RunPod

1. Copy results locally:
   ```bash
   rsync -avz runpod:~/topo-confidence/pathway6_rebuild/phase6_5/ \
     ~/topo-confidence/pathway6_rebuild/phase6_5/
   ```

2. Check FINAL_SUMMARY.md for the complete analysis.

3. Git commit the JSON results (binary .npy/.npz are gitignored):
   ```bash
   cd ~/topo-confidence
   git add pathway6_rebuild/phase6_5/
   git commit -m "Phase 6.5: deconfounded cross-benchmark results (max_new_tokens=1024)"
   git push
   ```

## Cost Estimate

- GSM8K × 1.5B: ~3 hours (1319 problems, greedy + temp + trajectories)
- MATH × 7B: ~2.5 hours (500 problems, 7B model slower but fewer problems)
- Cross-analysis + summary: ~5 min (CPU only)
- **Total: ~5.5 hours, ~$16 at $2.99/hr**
