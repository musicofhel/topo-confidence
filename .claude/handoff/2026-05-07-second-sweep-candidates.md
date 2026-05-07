# Handoff — 2026-05-07 — Second Local-Compute Sweep

## Previous session

Ran 10 CPU experiments overnight (EXP-59–68). All passed. Committed + pushed
on `max-depth-retriage-2026-04-28` (6f50f78). Key findings:
- FE919 Diffusion Maps k=10 AUROC **0.788** > DoM 0.773 (nonlinear helps)
- FE903 mCCA=0.98 (shared subspace, orthogonal DoM directions)
- FE909 alignment profile peaks L19, half-max L15-L21 (7 layers)
- FE889 DoM NOT aligned with RLHF rotation axes
- FE01172B participation ratio 19.86 (matches cov-spectrum dim=20)
- FE26841a correct samples have HIGHER entropy at L19 (counterintuitive)
- FE925 Gaussian QDA OOF 0.736 < DoM 0.773

**Full brief**: `research-graph/briefs/result-2026-05-06-P11-overnight-sweep.md`

## What to do

Run 10 more CPU experiments from the research-graph queue, then write briefs and update graph. Create a plan first (user wants plan mode).

**Full plan**: to be created in plan mode.

## 10 experiments, 3 batches (~4h sequential, ~1.5h parallel)

### Batch 1 — Convenience cache only (~30 min, 5 parallel)

All use `m15b_prefill.npz` (500×1536 float16). Light compute.

| FE ID | ROI | Est. | Description |
|---|---|---|---|
| FE136 | 9 | 20min | Park-Choe-Veitch causal inner product: whitened cos(prefill_DoM, final_DoM) |
| FE254 | 9 | 30min | Joint [prefill, final] concat probe AUROC vs DoM-only 0.7731 |
| FE331 | 9 | 10min | Dynamic-c steering coefficient for H-1 (Stolfo Eq. 2) |
| FE339 | 9 | 20min | Linear-AcT variance-aware probe: σ-scaled DoM beats mean-only? |
| FE421 | 9 | 20min | Prefill+Final concat DoM AUROC (two-circuits test for F-3) |

**Why this batch**: All follow up on FE903's finding (shared subspace, orthogonal DoMs). Tests whether combining prefill+final helps, whether orthogonality survives whitening, and whether variance-aware probing beats DoM.

### Batch 2 — Full NPZs + safetensors (~45 min, 4 parallel)

Need per-problem NPZs (500 files, 19GB) and/or safetensors for W_unembed.

| FE ID | ROI | Est. | Description |
|---|---|---|---|
| FE244 | 9 | 10min | NC3 violation test: DoM vs W_unembed row alignment |
| FE428 | 9 | 30min | L0 embedding-layer DoM baseline (deconfounding control) |
| FE15 | 9 | 20min | Length-band PR control: does PR=20 survive length stratification? |
| FE188 | 9 | 30min | LID-MLE + GeoMLE intrinsic dimension (complements PR=20) |

**Why this batch**: Deconfounding controls (FE428 tests L0, FE15 tests length, FE244 tests structural NC3) and an alternative dimensionality estimate (FE188 LID vs FE01172B PR).

### Batch 3 — K=8 + per-token data (~1h, 2 parallel)

| FE ID | ROI | Est. | Description |
|---|---|---|---|
| FE416 | 9 | 30min | Pre-final token DoM: does shifting 1 position recover cos>0.3? |
| FE308 | 9 | 1h | Adaptive best-of-k (Damani): continuous K-allocation vs F-8 binary |

**Why this batch**: FE416 tests the positional-artifact hypothesis for F-3 orthogonality. FE308 directly compares DoM-based compute allocation against the selective-prediction frontier.

## Key data paths (same as previous sweep)

- Convenience cache: `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
- Full NPZs: `pathway8_layerwise/data/math500/problem_*.npz` (500 files, states shape (29, T, 1536))
- DoM scores: `pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz` (key: `correct_k1`)
- K=8 cache: `pathway11_h100/data/k8_selfconsistency/problem_*.npz` (keys: L19_samples (8,1536), texts (8,), correct (8,))
- Instruct safetensors: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors`

## Critical rules (carried from previous sweep)

1. Cast float16→float32 after NPZ load
2. Center before any Gram/covariance computation
3. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`
4. Folds: `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)`
5. `mark_done("feXXX", results_dir=RESULTS_DIR)` — default goes to pathway8
6. Load weights via `safetensors.safe_open` + `.float()`, never `torch_dtype=torch.float16`

## Method notes for tricky experiments

### FE136 — Causal inner product
Park-Choe-Veitch Theorem 3.2: whitened inner product M = Cov(γ)^{-1}. Compute
full activation covariance (all 500 samples), invert (or pseudoinverse since
rank ≤ 499), compute cos in whitened coords: `cos_M(u, v) = u^T M v / sqrt(u^T M u * v^T M v)`.
Compare raw cos(prefill_DoM, final_DoM) = 0.046 vs whitened cos.

### FE254 / FE421 — Joint probe
Concatenate prefill and final-token into (500, 3072). LogisticRegression(C=1.0) OOF AUROC.
FE254: l2-regularized. FE421: DoM on the concat space.

### FE428 — L0 embedding baseline
Use states[0, 0, :] from per-problem NPZs (layer 0, position 0 = prefill-end embedding).
DoM at L0 → OOF AUROC. If substantial AUROC at L0, F-2's L19 specificity is partially
inherited from input geometry.

### FE308 — Adaptive best-of-k
Use DoM scores from `phase2_prefill_dom.npz` as λ̂ per problem. Damani's greedy allocator:
sort by ascending λ̂, give low-confidence problems more K. Offline binning: split into
B=5 bins by λ̂, assign K_b proportional to (1-λ̂_median_b). Simulate on cached K=8
data, measure accuracy at average K ∈ {1, 2, 2.5, 4, 8}. Compare to F-8's 71.6%
at coverage 0.5 (avg K=2.5).

### FE416 — Pre-final token DoM
From per-problem NPZ states[19, -2, :] (L19, second-to-last token). Compute DoM on this
position. Report cos(prefinal_DoM, prefill_DoM) — if >0.3, F-3 orthogonality is a
positional artifact of the answer token itself.

### FE188 — LID-MLE
Levina-Bickel MLE for local intrinsic dimension at each sample. Use k=20 nearest
neighbors. Report mean LID, LID per class, LID as correctness predictor (AUROC).
Compare to FE01172B PR=19.86.

## Post-experiment (same workflow)

1. Write result brief: `research-graph/briefs/result-2026-05-07-P11-FE<ID>.md`
2. `python research-graph/update_status.py P11-FE<ID> COMPLETED --outcome "..."`
3. `python research-graph/generate_next_experiments.py`
4. Update EXPERIMENT_LOG.md (next ID: EXP-69), STATE.md
5. Commit + push on `max-depth-retriage-2026-04-28`
