# Pathway 9 — CLOSED (2026-04-22)

> **Supersedes** `HANDOFF.md` (v1) and `HANDOFF_v2.md` (v2, pre-run).
> Read `SUMMARY.md` first for results. This file is the short orienting doc.

---

## Status

**All 7 experiments complete. SUMMARY written. All 13 defects (D1–D13) resolved or qualified.**

| Stage | Where | Verdict file | Status |
|---|---|---|---|
| Exp 1 v1 — traj_rows deconfound | local | `results/exp1_length_deconfound.json` | ✅ |
| Exp 1 v2 — raw tokenizer (D6) | local | `results/exp1_raw_length_v2.json` | ✅ −0.050 drop |
| Exp 2 — PH vs CoE orthogonality | RunPod | `results/exp2_orthogonality.json` | ✅ CoE dominates |
| Exp 3a v2 — empirical-cov null (D1+D2) | RunPod | `results/exp3a_gaussian_null_v2.json` | ✅ real ≈ null |
| Exp 3b — token-shuffle null (GPU) | RunPod | `results/exp3b_token_shuffle.json` | ✅ 60% indistinguishable |
| Exp 3c — count control | local | `results/exp3c_count_control.json` | ✅ trivial |
| Exp 4 v2 — matched CV5 (D4+D5) | local | `results/exp4_xgboost_oinfo_v2.json` | ✅ signal linear |
| Exp 5 — MATH↔BBH transfer (D11+D12) | RunPod | `results/exp5_cross_domain.json` | ✅ CoE wins |

---

## Headline (see SUMMARY.md §7)

**Pivot topo-confidence's paper narrative from persistent-homology to Chain-of-Embedding (CoE).**

- CoE transfers MATH↔BBH at 0.720 / 0.712 (symmetric, avg 0.716).
- Layer-wise PH transfers at 0.354 / 0.497 (inverse / chance — fails).
- CoE-60 alone = 0.811 MATH-500 holdout, higher than ABC-44 PH at 0.7961 headline and 0.7459 deconfounded.
- 5 raw PH summary features are indistinguishable from a rank-matched empirical-covariance Gaussian null (0.690 vs 0.693).
- Signal is linear (matched CV5: LR 0.701 vs best XGB 0.696).

---

## What changed from HANDOFF_v2 plan

| Plan step | Actual |
|---|---|
| exp5 edits (D11 + D12) | Done pre-flight. |
| Data on pod | Missing `data/experiment1_v2/trajectories.npz` (162 MB). Uploaded from laptop. |
| BBH manifest layout | Script expected per-subset manifests; reality is ONE combined `bbh/manifest.json`. Patched `load_benchmark` mid-run to filter by `subset` field. |
| Deps on pod | `ripser`, `persim`, `transformers`, `accelerate` not present on the runpod/pytorch:2.4.0 image. Installed via pip. |
| Runtimes (new estimate 90-150 min) | Actual: ~55 min wall (exp3a 5.6 min, exp2 <1 min, exp5 ~45 min, exp3b <1 min). H100 crushed exp3b on GPU; PH's 6 compute passes in exp5 dominated wall time. |

All patches kept local *and* on the pod; no divergence.

---

## Files (absolute paths)

Results (all retrieved to laptop):
- `/home/musicofhel/topo-confidence/pathway9/results/exp1_length_deconfound.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp1_raw_length_v2.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp2_orthogonality.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp3a_gaussian_null_v2.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp3b_token_shuffle.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp3c_count_control.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp4_xgboost_oinfo_v2.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp5_cross_domain.json`

Synthesis:
- `/home/musicofhel/topo-confidence/pathway9/SUMMARY.md` — 7-section, publication-ready.

Code:
- `/home/musicofhel/topo-confidence/pathway9/run_local_experiments.py`
- `/home/musicofhel/topo-confidence/pathway9/run_fixes.py`
- `/home/musicofhel/topo-confidence/pathway9/exp3a_full_cov_null.py`
- `/home/musicofhel/topo-confidence/pathway9/exp2_orthogonality.py`
- `/home/musicofhel/topo-confidence/pathway9/exp3b_token_shuffle.py`
- `/home/musicofhel/topo-confidence/pathway9/exp5_cross_domain_transfer.py` — edited with D11+D12+BBH-manifest fixes
- `/home/musicofhel/topo-confidence/pathway9/run_runpod.sh`

Prior handoffs / audit:
- `/home/musicofhel/topo-confidence/pathway9/HANDOFF.md` (v1, superseded)
- `/home/musicofhel/topo-confidence/pathway9/HANDOFF_v2.md` (v2, pre-run)
- `/home/musicofhel/topo-confidence/pathway9/AUDIT.md` (D1–D8 register)

Pod + billing:
- RunPod pod `0agitikupjg259` (H100 SXM) is **STOPPED** (billing halted, volume intact). Resume via `runpodctl pod start 0agitikupjg259` if needed.

Memory:
- `~/.claude/projects/-home-musicofhel/memory/pathway9-coe-pivot.md` — durable project memory noting the PH→CoE pivot decision.

---

## Suggested next steps (for whoever picks this up)

1. **Pathway 10: CoE-headline paper draft.** Frame MATH-500 0.811 + cross-domain 0.716 as the publishable number. PH is a within-domain baseline.
2. **Pathway 10 alt: pure-CoE cross-model transfer.** Test whether CoE-60 trained on Qwen2.5-1.5B transfers to Qwen2.5-7B and GSM8K. If yes, the domain-invariance claim strengthens substantially.
3. **D5 follow-up if someone wants it (low priority):** replace O-information with a feasible alternative (mutual-information via k-NN estimator on a reduced feature set) to characterize redundancy within ABC-44. Not blocking.

---

## Guard rails for any rerun

- Pod `0agitikupjg259` already has all PH cache files, CoE cache, and (now) the trajectories.npz. Deps installed.
- `run_runpod.sh` is idempotent — `.done` markers gate each step.
- If the pod is recreated, must re-upload `trajectories.npz` (laptop has it at `/home/musicofhel/topo-confidence/data/experiment1_v2/trajectories.npz`, 162 MB) and reinstall deps (ripser, persim, transformers, accelerate).
