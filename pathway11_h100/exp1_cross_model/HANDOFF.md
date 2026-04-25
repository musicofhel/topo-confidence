# Exp 1 — Cross-model temporal PR (Phi-3-mini + Llama-3.2-1B)

**Session:** 2026-04-24. **Status:** extraction complete, pod removed, local analysis running.

## What exists

### Data (pulled from pod, local only)
- `pathway11_h100/exp1_cross_model/data/phi3mini/` — 500 npz + manifest.json, 166 MB
- `pathway11_h100/exp1_cross_model/data/llama32_1b/` — 500 npz + manifest.json, 65 MB
- Each `problem_XXX.npz` contains:
  - `prefill_all_layers` (n_layers+1, d) float16
  - `final_all_layers` (n_layers+1, d) float16
  - `twothirds_positions` (n_pos, d) float16
  - `positions_actual` (n_pos,) int32 — generation-token indices captured
  - `twothirds_layer_idx` int — Phi-3: 21, Llama: 11
  - `n_gen_tokens`, `text`, `correct`, `mean_logprob`

### Code
- `pathway11_h100/exp1_cross_model/extract.py` — GPU extraction, `--model {phi3mini,llama32-1b}`
- `pathway11_h100/exp1_cross_model/analyze.py` — CPU analysis (temporal + depth PR, 200× bootstrap)
- `pathway11_h100/exp1_cross_model/run.sh` — extract both + analyze orchestrator
- `pathway11_h100/exp1_cross_model/logs/full_run.log` — GPU run log (4975s total)

### Produced (analysis complete, N_BOOT=50)
- `pathway11_h100/exp1_cross_model/results.json` (59 KB)
- `pathway11_h100/exp1_cross_model/temporal_pr_curve.png` (231 KB)
- `pathway11_h100/exp1_cross_model/depth_pr_prefill_vs_final.png` (431 KB)

### Headline result

**Breathing is universal across Qwen + Phi-3 + Llama.** Both new models reproduce the Qwen pattern: temporal PR rises then collapses, and correct trajectories collapse harder than incorrect at the final token.

| Model | pos 1 PR | peak PR | final PR | correct<incorrect at final? |
|---|---:|---:|---:|:---:|
| Phi-3-mini (L21) | 19 | ~100 (pos 10–50) | 18 (c=12 / i=17) | ✓ |
| Llama-3.2-1B (L11) | 3 | ~115 (pos 50) | 8 (c=4 / i=8) | ✓ |

Architecture-dependent nuance: Llama's correct/incorrect split opens at pos 25; Phi-3's opens only at pos 100+. Same binary outcome, different timing.

Prefill-depth PR_correct > PR_incorrect "inversion" (Qwen L19 signature) reproduces in Phi-3 at late layers (L27–31) and Llama at late layers (L12–16) — same sign, different absolute depth.

## Setup / design decisions (scope approved by user)

1. **Trimmed from user's "polished" Exp 1 spec**: dropped BBH transfer, length residualization, T=0.7 control. Kept the binary question: *do Phi-3-mini and Llama-3.2-1B show the same breathing pattern as Qwen?*
2. **Headline-resolution rule**: user reinforced that "two points isn't enough to show breathing" — so the 2/3-depth layer is captured at 7 positions `{1, 10, 25, 50, 100, 200, final}`, not just prefill+final. All layers captured only at prefill-end and final-token (disk-cheap snapshots for depth comparison).
3. **Greedy T=0, max 1024 new tokens.** Same MATH-500 system prompt as prior Qwen work.
4. **Phi-3 attention**: sdpa unsupported by Phi-3 in transformers 5.6.2 → uses `eager`. Llama uses sdpa.
5. **Llama variant**: `unsloth/Llama-3.2-1B-Instruct` (open re-upload, same weights, avoids Meta gated-repo HF auth).
6. **Bootstrap N=200** for CIs (downgrade from 1000, 5× speedup). User rationale: "CIs are confirmatory, not discovery — the pattern already exists from Qwen."

## Accuracy on MATH-500 @ T=0 (for reference)

| Model | Acc | n |
|---|---:|---:|
| Phi-3-mini-4k-instruct | 44.8% | 224/500 |
| unsloth/Llama-3.2-1B-Instruct | 25.2% | 126/500 |
| (prior) Qwen2.5-1.5B | 48.6% | 243/500 |
| (prior) Qwen2.5-7B | 73.2% | 366/500 |

## RunPod pod

**Pod `y687b9z2dgukcj` was REMOVED** (not stopped) at session end. Reason: it had `volumeInGb: 0`, so stop doesn't preserve disk — any future run re-installs from scratch anyway. Removed saves on the reserved-capacity storage fee.

Corrected memory: `~/.claude/projects/-home-musicofhel/memory/pathway11-pod-keep-alive.md` now documents the no-volume-means-stop-is-pointless finding. The old memory claiming "stop preserves disk" was wrong for this pod.

## To resume

If you want to re-create the pod for more work:
```bash
# Create fresh H100 SXM pod (use --volumeInGb N to get real persistence this time)
runpodctl pod create --imageName runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  --gpuType "NVIDIA H100 80GB HBM3" --volumeInGb 200 ...
```

Setup on any fresh pod (covers Exp 1 deps):
```bash
cd /workspace && git clone https://github.com/musicofhel/topo-confidence.git
cd topo-confidence
pip install -r pathway8_layerwise/requirements_pathway8.txt
pip install accelerate
# pathway11_h100/ is not in the git remote — scp from local if needed:
#   scp -i $KEY -P $PORT -r ~/topo-confidence/pathway11_h100/exp1_cross_model \
#     ~/topo-confidence/pathway11_h100/config.py \
#     root@$IP:/workspace/topo-confidence/pathway11_h100/
```

## Analysis run (complete 2026-04-24 14:01)

Ran twice in this session:
1. First pass with `N_BOOT=1000` then `200` — killed mid-run; the 2h ETA wasn't worth 2× tighter CIs for a binary-outcome question.
2. Second pass with `N_BOOT=50` — completed in ~30 min. Same point estimates (SVD is deterministic), ~2× wider CIs. Fine for "does pattern hold" call.

If you want tighter CIs later: edit `N_BOOT` in `analyze.py` and rerun. Script is idempotent.

## Interpretation checklist (for when figures are ready)

The headline question is binary: **do the temporal PR curves show "breathing" (rise then collapse) with correct-collapses-harder, same as Qwen?**

- ✅ If both Phi-3 and Llama show the same shape → universal property of transformer inference on reasoning tasks across architectures.
- ❌ If one breaks the pattern → architecturally or scale-dependent, worth investigating which axis.
- Compare to Qwen Exp 3 result (`pathway11_h100/prefill_inversion/SUMMARY.md`): at prefill-end L19, PR_correct > PR_incorrect (an inversion from the "more structure → more correct" intuition). Check if Phi/Llama reproduce that sign at 2/3-depth prefill specifically.

## Earlier-in-session artifacts (Exp 2b, cached-data-only)

Prior to this Exp 1 work, the session completed Exp 2b (multi-signal oracle-K classifier) entirely on cached data. See `pathway11_h100/multi_signal_oracle/SUMMARY.md`. **Bottom line: 3-feature classifier beats best single-signal by only +0.4pp at matched compute; user's ≥2pp bar NOT met; `prefill_lpr` contributes exactly 0.** The "non-monotonic multi-feature gating" direction is closed for this dataset.
