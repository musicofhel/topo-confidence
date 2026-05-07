# Handoff — 2026-05-07 — Second Sweep Plan Ready

## What happened this session

Re-entered plan mode for the second CPU sweep (10 experiments). **Found that 6 of the 11 handoff candidates were already done** in EXP-51 (2026-05-01, "Phase 3 causal corroborators bundle"): FE136, FE244, FE331, FE339, FE254, FE319. Script: `pathway11_h100/phase3_corroborators/recompute_phase3.py`.

Replaced them with 5 fresh picks from the queue. Verified all data caches. Wrote detailed plan.

## Plan location

**`.claude/plans/dapper-sprouting-rose.md`** — complete implementation plan for 10 experiments.

## What to do next

**Exit plan mode, then implement all 10 experiments + runner script from the plan.** The plan has:
- Per-experiment method descriptions with pitfalls
- Directory structure and file names
- Batching (3 batches, ~1.5h total)
- Global implementation rules
- Verification checklist
- Post-experiment workflow

## The 10 experiments (EXP-69 through EXP-78)

| EXP | FE ID | Time | Description | Data source |
|---|---|---|---|---|
| 69 | FE447 | 10min | Length-as-correctness baseline + OOF residualization | convenience cache |
| 70 | FE188 | 30min | LID-MLE local intrinsic dimension | convenience cache |
| 71 | FE421 | 20min | Ridge-LR on [prefill, final] concat | convenience cache |
| 72 | FE15 | 20min | Length-band PR control | convenience cache |
| 73 | FE16 | 30min | Marchenko-Pastur bias-corrected PR | convenience cache |
| 74 | FE428 | 15min | L0 embedding-layer DoM baseline | per-problem NPZs |
| 75 | FE416 | 15min | Pre-final token DoM (position -2) | per-problem NPZs |
| 76 | FE119 | 10min | Layer-sweep cos(prefill_DoM, final_DoM) | per-problem NPZs |
| 77 | FE308 | 1h | Adaptive best-of-k Damani allocator | DoM scores + K=8 cache |
| 78 | FE459 | 30min | Cross-model 1.5B↔7B DoM score correlation | 7B NPZs |

## Key discoveries during planning

1. **Existing length AUROC = 0.7986** in `pathway11_h100/length_partial/results.json` — HIGHER than DoM 0.7731. Residualized DoM = 0.6647 (in-sample). FE447 adds OOF version.
2. **7B NPZ schema:** states (29, T, 3584) float16, correct (scalar bool), text (scalar str). L19 position 0 is extractable.
3. **K=8 cache schema:** L19_samples (8, 1536), texts (8,), correct (8,). Majority vote K=8 = 43.8% (LOWER than greedy K=1 = 48.6%).
4. **Phase 3 corroborators used custom folds** (NOT sklearn) — all new experiments must use `StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)` for comparability with DoM 0.7731.

## Critical data paths

- Convenience cache: `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
- DoM scores: `pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz` (key: `prefill_score`)
- K=8 cache: `pathway11_h100/data/k8_selfconsistency/problem_*.npz` (500 files)
- Per-problem 1.5B: `pathway8_layerwise/data/math500/problem_*.npz` (500 files)
- Per-problem 7B: `pathway11_h100/data/math500_7b/problem_*.npz` (500 files)

## Branch / commit state

Branch: `max-depth-retriage-2026-04-28`, last commit: 6f50f78, pushed. Next EXP ID: EXP-69.
validate_claims.py passes. All infrastructure (Neo4j, research-graph) is live.
