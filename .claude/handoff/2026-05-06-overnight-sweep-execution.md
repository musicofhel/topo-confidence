# Handoff — 2026-05-06 — Overnight Sweep Execution

## What to do

Run 10 CPU experiments from today's triage, save results, write briefs, update graph.

**Full plan**: `/home/musicofhel/.claude/plans/dapper-sprouting-rose.md`

## 10 experiments, 3 batches (~85 min total)

**Batch 1** (~10 min, 3 parallel, no model weights):
- FE01172B: Gram eigenspectrum → effective rank of L19 activations
- FE903: Regularized CCA between prefill and final-token activations
- FE899: Principal subspace angles between correct/incorrect groups

**Batch 2** (~30 min, 4 parallel, safetensors weight loading):
- FE930: RoPE plane projection via q_proj pullback (per-head decomposition)
- FE901: SETOL ECS projection (ROI 9, highest priority)
- FE26841a: Logit-lens entropy for correctness prediction
- FE889: Procrustes rotation axes via matrix logarithm

**Batch 3** (~45 min, 3 parallel, compute-heavy):
- FE909: Per-layer alignment score profile (cos with L19 DoM across all 28 layers)
- FE925: Hyvärinen score (Gaussian baseline + sliced score matching primary)
- FE919: Diffusion Maps embedding (k ∈ {2,5,10,20})

## Critical rules (from 2 audit rounds)

1. Cast float16→float32 after NPZ load
2. Center before any Gram/covariance computation
3. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (OpenBLAS ignores OMP)
4. Folds: `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)` — NOT `stratified_kfold_indices()` from recompute_fe145.py (different folds, same seed)
5. `mark_done("feXXX", results_dir=RESULTS_DIR)` — default goes to pathway8, not pathway11
6. Load weights via `safetensors.safe_open` + `.float()`, never `torch_dtype=torch.float16` (model is bfloat16)
7. FE901: `W @ W^T` for ALL matrices (residual-space eigenvectors), not `W^T @ W`
8. FE899: `subspace_angles` expects shape (1536, k), so PCA components need `.T`
9. FE889: `scipy.linalg.logm(R)` not Cayley transform; extract real 2D planes from complex conjugate pairs
10. FE909: Copy `load_per_layer_prefill()` into script, don't import from recompute_fe145.py
11. Runner: PID capture + exit code checking + per-experiment log files

## Key data paths

- Convenience cache: `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz` (500×1536, float16, keys: prefill, final_tok, correct, seq_len)
- Full NPZs: `pathway8_layerwise/data/math500/problem_*.npz` (500 files, states shape (29, T, 1536))
- DoM scores: `pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz` (key: correct_k1, NOT correct)
- Instruct safetensors: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors`
- Base safetensors: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*/model.safetensors`

## Post-experiment

1. Write result brief per experiment: `research-graph/briefs/result-2026-05-06-P11-FE<ID>.md`
2. `python research-graph/update_status.py P11-FE<ID> COMPLETED --outcome "..."`
3. `python research-graph/generate_next_experiments.py`
4. Add claims to `validate_claims.py`, verify pass
5. Update EXPERIMENT_LOG.md (next ID: EXP-59), STATE.md
6. Commit + push on branch `max-depth-retriage-2026-04-28`
