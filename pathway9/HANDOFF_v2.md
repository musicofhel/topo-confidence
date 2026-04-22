# Pathway 9 Handoff v2 — 2026-04-21 (post fresh-eyes audit)

> **Supersedes** `/home/musicofhel/topo-confidence/pathway9/HANDOFF.md` (kept for history).
> Written pre-compact. Read this first after reloading.

---

## TL;DR

6/7 experiments done locally. 4 remaining RunPod scripts are written. A fresh-eyes audit (3 parallel Explore agents, no pre-loaded context) confirmed all 8 prior defects (D1–D8) and found **one new 🔴 blocking bug (D11) in exp5** plus a small 🟡 leak (D12) and a 🟢 naming nit (D13). **Fix D11 + D12 before launching RunPod.**

HANDOFF v1 numeric claims all verified clean against their JSON files (baseline 0.7960784 to 8 decimals across exp1/exp3c/exp4 — no silent drift).

---

## Fresh audit findings (new)

### 🔴 D11 — exp5 transfer uses mismatched PCA bases (MUST FIX)

**File:** `/home/musicofhel/topo-confidence/pathway9/exp5_cross_domain_transfer.py:139-141, 146-155`

**What's wrong:** For the `layerwise_ph` method, features are computed with each domain fitting its own PCA over its own tokens:
```python
math_feats = compute_features(math_states, method, train_idx=math_idx)  # line 139, PCA fit on MATH
bbh_feats  = compute_features(bbh_states,  method, train_idx=bbh_idx)   # line 140, PCA fit on BBH
# ...
math2bbh = transfer_test(math_feats, math_labels, bbh_feats, bbh_labels, ...)  # line 154
```

Even though both arrays are shape `(n, 168)`, the 168 columns are PH summary stats over *different PCA projections*. The LR trained on MATH's feature space is evaluated on BBH's feature space — silently invalid.

The dim-mismatch guard at :146–152 never fires because `compute_layerwise_ph_batch` always returns 168 dims regardless of PCA rank.

CoE and D2H-lite do NOT fit a PCA (features are raw hidden-state statistics), so they are unaffected.

**Fix:** For PH only, fit PCA on the source domain and apply to both source and target:
```python
if method == "layerwise_ph":
    # MATH → BBH: fit PCA on MATH, apply to both
    math_transforms = fit_per_layer_pca(math_states, math_idx)
    mf_m = compute_layerwise_ph_batch(math_states, math_transforms, n_jobs=-1)
    bf_m = compute_layerwise_ph_batch(bbh_states,  math_transforms, n_jobs=-1)
    math2bbh = transfer_test(mf_m, math_labels, bf_m, bbh_labels, f"MATH→BBH ({method})")

    # BBH → MATH: fit PCA on BBH, apply to both
    bbh_transforms = fit_per_layer_pca(bbh_states, bbh_idx)
    mf_b = compute_layerwise_ph_batch(math_states, bbh_transforms, n_jobs=-1)
    bf_b = compute_layerwise_ph_batch(bbh_states,  bbh_transforms, n_jobs=-1)
    bbh2math = transfer_test(bf_b, bbh_labels, mf_b, math_labels, f"BBH→MATH ({method})")
else:
    # existing path for coe / d2h_lite
    math_feats = compute_features(math_states, method, train_idx=math_idx)
    bbh_feats  = compute_features(bbh_states,  method, train_idx=bbh_idx)
    math2bbh = transfer_test(math_feats, math_labels, bbh_feats, bbh_labels, ...)
    bbh2math = transfer_test(bbh_feats,  bbh_labels,  math_feats, math_labels, ...)
```

Delete the dim-mismatch guard (dead code after fix).

### 🟡 D12 — exp5 within-domain baseline leaks PCA into holdout

**File:** `/home/musicofhel/topo-confidence/pathway9/exp5_cross_domain_transfer.py:170-178`

**What's wrong:** PCA is fit on the entire domain, THEN the 80/20 split happens. The 20% holdout's tokens were seen by PCA fitting.

**Fix:** For PH within-domain baselines, fit PCA on the 80% train split only:
```python
mtr_transforms = fit_per_layer_pca(math_states, mtr)
mf_w = compute_layerwise_ph_batch(math_states, mtr_transforms, n_jobs=-1)
within_math = transfer_test(mf_w[mtr], math_labels[mtr], mf_w[mho], math_labels[mho], ...)
# same idea for BBH with btr
```
CoE/D2H-lite paths unchanged.

### 🟢 D13 — SVD-null naming nit

**File:** `/home/musicofhel/topo-confidence/pathway9/exp3a_full_cov_null.py` (docstring + verdict strings)

The sampler draws from the empirical cov (rank ≤ n_tokens − 1). This is equivalent to `np.random.multivariate_normal(mean, cov_empirical)` and is the correct test. Just reword "full-covariance" → "empirical-covariance (rank-matched)" when writing SUMMARY.md. No code change.

---

## Prior defects (confirmed, from AUDIT.md)

Full detail at `/home/musicofhel/topo-confidence/pathway9/AUDIT.md`.

| ID | Severity | Experiment | Status |
|----|----------|------------|--------|
| D1 | Medium | exp3a: diagonal variance, not full cov | Fixed in `exp3a_full_cov_null.py` (RunPod) |
| D2 | Medium | exp3a: H0_n_features structurally identical | Fixed — dropped, now 5 features |
| D3 | Medium | exp3a: CV std 0.245, no power | Inherent design; narrower verdict only |
| D4 | Medium | exp4: LR 50-fold vs XGB 5-fold unfair | Fixed in v2 (`exp4_xgboost_oinfo_v2.json`) |
| D5 | Low | exp4: O-info dtype error → then 3 TiB OOM | Abandoned O-info; XGB vs LR answers Q |
| D6 | Low→High | exp1: wrong length proxy (traj_rows, not tokens) | Fixed in v2 (`exp1_raw_length_v2.json`) — **drops AUROC by 0.050** |
| D7 | Medium | exp3a verdict overreach | Reword pending v2 run |
| D8 | Low | exp3c: count-control was a no-op | Trivially correct; no rerun needed |

---

## Status table

| Stage | Where | Verdict file | Status |
|---|---|---|---|
| Exp 1 v1 — traj_rows deconfound | local | `exp1_length_deconfound.json` | ✅ done |
| Exp 1 v2 — raw tokenizer (D6) | local | `exp1_raw_length_v2.json` | ✅ done — **0.050 AUROC drop** |
| Exp 3a v1 — diag-variance null | local | `exp3a_gaussian_null.json` | ✅ done (superseded) |
| Exp 3a v2 — empirical-cov null, 5 feats (D1+D2) | **RunPod** | `exp3a_gaussian_null_v2.json` | 🟡 script ready |
| Exp 3c — count control | local | `exp3c_count_control.json` | ✅ done (trivial) |
| Exp 4 v1 — mismatched CV | local | `exp4_xgboost_oinfo.json` | ✅ done (superseded) |
| Exp 4 v2 — matched CV5, O-info abandoned (D4+D5) | local | `exp4_xgboost_oinfo_v2.json` | ✅ done |
| Exp 2 — PH vs CoE orthogonality | **RunPod** | `exp2_orthogonality.json` | 🟡 script ready |
| Exp 3b — token-shuffle null (GPU) | **RunPod** | `exp3b_token_shuffle.json` | 🟡 script ready |
| Exp 5 — MATH↔BBH transfer | **RunPod** | `exp5_cross_domain.json` | 🔴 **FIX D11+D12 FIRST** |

---

## Honest numbers (post-audit)

| Config | AUROC |
|---|---|
| ABC-44 uncorrected (holdout 100) | **0.7961** |
| ABC-44 length-deconfounded (raw tokenizer) | **0.7459** (drop 0.050) |
| ABC-44 CV5 (matched) | 0.7012 |
| Best XGBoost (regularized, CV5) | 0.6960 (lift −0.005 vs LR → LR appropriate) |

**Publication wording:** report 0.7961 with explicit footnote "0.7459 after deconfounding against raw tokenizer length (0.050 drop); LR signal is linear per matched-CV XGBoost comparison."

---

## Data substrate (verified 2026-04-21)

Laptop:
```
/home/musicofhel/topo-confidence/pathway2/track_a/phase0/baseline_correct.npy            ✅ 500, sum=57 (OLD labels)
/home/musicofhel/topo-confidence/pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy ✅ 500, sum=104 (NEW labels)
/home/musicofhel/topo-confidence/pathway1/phase1/features_train400.npy                   ✅ (400, 78)
/home/musicofhel/topo-confidence/pathway1/phase1/features_holdout100.npy                 ✅ (100, 78)
/home/musicofhel/topo-confidence/data/experiment1_v2/trajectories.npz                    ✅ 500 keys
/home/musicofhel/topo-confidence/data/experiment1_v2/trajectory_meta.json                ✅ has generated_texts (500)
/home/musicofhel/topo-confidence/pathway8_layerwise/results/exp1_features_all.npy        ❌ NOT on laptop
/home/musicofhel/topo-confidence/pathway8_layerwise/results/exp2_coe_features.npy        ❌ NOT on laptop
/home/musicofhel/topo-confidence/pathway8_layerwise/data/math500/problem_*.npz           ❌ 0 files on laptop
/home/musicofhel/topo-confidence/pathway8_layerwise/data/bbh/<subset>/problem_*.npz      ❌ dirs missing on laptop
```

Missing files live on **RunPod volume `0agitikupjg259`** (per HANDOFF v1). Must verify they're still there before running. Baseline reproduces at **0.7961** with StandardScaler + LR pipeline.

`runpodctl` is installed (2.1.9-673143d). **No pod currently running — must provision H100 SXM.**

---

## Execution plan

### Step 0 — Make the exp5 fix (local edit, ~15 min)

Edit `/home/musicofhel/topo-confidence/pathway9/exp5_cross_domain_transfer.py` per D11 and D12 above. The imports needed (`fit_per_layer_pca`, `compute_layerwise_ph_batch`) are already at line 41-44.

Verify by re-reading the diff and confirming:
- `layerwise_ph` transfer fits PCA on source domain once, applies to both.
- `layerwise_ph` within-domain fits PCA on the 80% train split, applies to both halves.
- `coe` and `d2h_lite` paths unchanged.
- Dim-mismatch guard at :146-152 removed.

### Step 1 — Provision RunPod H100 SXM

Per `~/.claude/projects/-home-musicofhel/memory/runpod-preferences.md`: always H100 SXM, never downgrade. Mount volume `0agitikupjg259` at `/workspace`.

### Step 2 — Sanity-check volume before full run

```bash
ssh root@$POD "cd /workspace/topo-confidence && \
  ls pathway8_layerwise/results/exp1_features_all.npy \
     pathway8_layerwise/results/exp2_coe_features.npy \
     data/experiment1_v2/trajectories.npz \
     data/experiment1_v2/trajectory_meta.json && \
  ls pathway8_layerwise/data/math500/ | head -3 && \
  for d in tracking_shuffled_objects_seven_objects logical_deduction_seven_objects web_of_lies; do
    echo \"\$d: \$(ls pathway8_layerwise/data/bbh/\$d 2>/dev/null | wc -l)\"
  done"
```

If any missing → stop, escalate to user (re-running pathway 8 extraction is out of scope).

### Step 3 — Ship scripts

```bash
scp /home/musicofhel/topo-confidence/pathway9/exp3a_full_cov_null.py \
    /home/musicofhel/topo-confidence/pathway9/exp2_orthogonality.py \
    /home/musicofhel/topo-confidence/pathway9/exp3b_token_shuffle.py \
    /home/musicofhel/topo-confidence/pathway9/exp5_cross_domain_transfer.py \
    /home/musicofhel/topo-confidence/pathway9/run_runpod.sh \
    root@$POD:/workspace/topo-confidence/pathway9/
```

### Step 4 — Execute

```bash
ssh root@$POD "cd /workspace/topo-confidence && bash pathway9/run_runpod.sh"
```

Order (cheap CPU → expensive CPU → GPU):
1. `exp3a_v2` (~30–60 min CPU) — SVD-sampled empirical-cov null, 5 PH features
2. `exp2`    (~2 min CPU) — uses cached pathway 8 PH (168) + CoE (60) arrays
3. `exp5`    (~30–60 min CPU) — cross-domain transfer (with D11 + D12 fixes; PH recomputes 6× total: 2 transfer dirs × 2 domains + 2 within-domain train-split PCAs)
4. `exp3b`   (~15 min GPU) — token-shuffle null on Qwen2.5-1.5B-Instruct

Total ~90–150 min on H100 SXM. `.done` markers gate resume on rerun.

CPU-only fallback: `bash pathway9/run_runpod.sh --skip-exp3b` and run exp3b on a separate GPU pod later.

### Step 5 — Retrieve results

```bash
scp root@$POD:/workspace/topo-confidence/pathway9/results/\*.json \
    /home/musicofhel/topo-confidence/pathway9/results/
scp root@$POD:/workspace/topo-confidence/pathway9/results/\*.done \
    /home/musicofhel/topo-confidence/pathway9/results/
```

Expected new files:
- `/home/musicofhel/topo-confidence/pathway9/results/exp3a_gaussian_null_v2.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp2_orthogonality.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp3b_token_shuffle.json`
- `/home/musicofhel/topo-confidence/pathway9/results/exp5_cross_domain.json`

### Step 6 — Synthesize SUMMARY.md

Write `/home/musicofhel/topo-confidence/pathway9/SUMMARY.md` with 7 sections:
1. Length-confound verdict (Exp 1 v2) — significant, 0.050 drop
2. PH vs CoE redundancy or complementarity (Exp 2) — gate: r>0.8 redundant, lift>0.01 complementary
3. Null-test verdict: topology, covariance, or length? (Exp 3a v2, 3b, 3c) — reword "full-cov" → "empirical-cov (rank-matched)"
4. Classifier verdict: LR vs XGBoost (Exp 4 v2) — LR fine, signal linear
5. Cross-domain verdict (Exp 5) — note whether D11-fixed PH still chance vs CoE/D2H
6. Honest final numbers table (all deconfounded AUROCs)
7. Recommendation: continue PH, pivot to CoE, or publish with caveats

---

## Key files (absolute paths)

Scripts:
- `/home/musicofhel/topo-confidence/pathway9/exp3a_full_cov_null.py` — empirical-cov null via SVD, 5 features
- `/home/musicofhel/topo-confidence/pathway9/exp2_orthogonality.py` — PH vs CoE stacked ensemble
- `/home/musicofhel/topo-confidence/pathway9/exp3b_token_shuffle.py` — GPU, Qwen2.5-1.5B, 50 × 5 shuffles
- `/home/musicofhel/topo-confidence/pathway9/exp5_cross_domain_transfer.py` — **EDIT D11+D12 FIRST**
- `/home/musicofhel/topo-confidence/pathway9/run_runpod.sh` — idempotent driver
- `/home/musicofhel/topo-confidence/pathway9/run_local_experiments.py` — v1 local (exp1, 3a, 3c, 4)
- `/home/musicofhel/topo-confidence/pathway9/run_fixes.py` — v2 fix reruns (exp1_v2, exp4_v2)

Reference:
- `/home/musicofhel/topo-confidence/pathway9/AUDIT.md` — 8 defect register (D1–D8)
- `/home/musicofhel/topo-confidence/pathway9/HANDOFF.md` — v1 handoff (superseded by this)
- `/home/musicofhel/topo-confidence/pathway8_layerwise/config.py` — SSS (9999, test=100), LR_PARAMS, load_manifest, load_all_layer_states
- `/home/musicofhel/topo-confidence/pathway8_layerwise/layerwise_features.py` — `fit_per_layer_pca`, `compute_layerwise_ph_batch`
- `/home/musicofhel/topo-confidence/pathway8_layerwise/coe_features.py`, `d2hscore_features.py`

Results directory:
- `/home/musicofhel/topo-confidence/pathway9/results/`

Plan file (parallel to this handoff, produced during plan mode):
- `/home/musicofhel/.claude/plans/elegant-booping-map.md`

---

## Guard rails for next session

- **Do NOT re-read pathway 8 code in full.** Prior two sessions died of context overflow doing this. Reference this handoff + AUDIT.md only.
- **Do NOT attempt O-information.** `hoi.metrics.Oinfo` OOMs with a 3.07 TiB allocation by construction on 44 features at order 3-5. Abandoned.
- **Do NOT re-run pathway 8 extraction locally.** If the RunPod volume lost the caches, escalate to user.
- **Do NOT skip Step 0 (exp5 fix).** The transfer AUROCs will be silently invalid for PH if you do.
- `/home/musicofhel/.claude/projects/-home-musicofhel/memory/MEMORY.md` is at 210 lines (limit 200) — flag for a separate condense pass.

---

## Quick resume checklist

```
[ ] Read this file (/home/musicofhel/topo-confidence/pathway9/HANDOFF_v2.md)
[ ] Read /home/musicofhel/topo-confidence/pathway9/AUDIT.md (for D1–D8 context)
[ ] Edit exp5_cross_domain_transfer.py per D11 + D12 (Step 0)
[ ] Ask user for RunPod POD_ID; confirm volume 0agitikupjg259 mounted
[ ] Sanity-check volume (Step 2)
[ ] scp scripts (Step 3)
[ ] Run bash pathway9/run_runpod.sh (Step 4)
[ ] scp results back (Step 5)
[ ] Write /home/musicofhel/topo-confidence/pathway9/SUMMARY.md (Step 6)
```
