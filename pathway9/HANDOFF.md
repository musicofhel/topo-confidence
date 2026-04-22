# Pathway 9 Handoff — 2026-04-21 (v3, post-audit)

## Status

**6/7 experiments done locally.** 4 remaining need RunPod (GPU+CPU). Scripts
and runner are written and ready.

| Stage | Where done | Verdict file | Status |
|---|---|---|---|
| Exp 1 — length deconfound (traj_rows)   | local | `exp1_length_deconfound.json` | ✅ done (verdict revised, see below) |
| Exp 1 v2 — raw tokenizer length (audit D6) | local | `exp1_raw_length_v2.json`     | ✅ done |
| Exp 3a — Gaussian null (diag variance)  | local | `exp3a_gaussian_null.json`    | ✅ done (superseded by v2) |
| Exp 3a v2 — full-cov null, 5 feats (D1+D2) | **RunPod** | `exp3a_gaussian_null_v2.json` | 🟡 pending — local attempt was too slow |
| Exp 3c — count control                  | local | `exp3c_count_control.json`    | ✅ done |
| Exp 4 — XGBoost vs LR (50-fold vs 5-fold) | local | `exp4_xgboost_oinfo.json`     | ✅ done (superseded by v2) |
| Exp 4 v2 — matched CV5, O-info dtype fix (D4+D5) | local | `exp4_xgboost_oinfo_v2.json`  | ✅ done (O-info still OOMs — 3 TiB alloc) |
| Exp 2 — PH vs CoE orthogonality         | **RunPod** | `exp2_orthogonality.json`     | 🟡 script ready |
| Exp 3b — token-shuffle null (GPU)       | **RunPod** | `exp3b_token_shuffle.json`    | 🟡 script ready |
| Exp 5 — MATH↔BBH transfer               | **RunPod** | `exp5_cross_domain.json`      | 🟡 script ready |

---

## What changed in this session — the audit

See `pathway9/AUDIT.md` for the full defect register. Two verdicts were
revised, one discovery was made.

### 🔴 DISCOVERY: the original Exp 1 used the wrong "length"

Exp 1 residualized against `trajectory_shapes[:, 0]` (rows of the saved
hidden-state matrices), which correlates only **r=+0.096** with the actual
tokenizer-encoded completion length. They are different quantities entirely.

When I reran Exp 1 with the true tokenizer length (Qwen2.5 tokenizer, no
special tokens, on `generated_texts`):

| Measure | trajectory_rows (orig Exp 1) | raw tokenizer (Exp 1 v2) |
|---|---|---|
| length-only AUROC | 0.704 | 0.705 |
| mean/median length | 89 / 70 | 169 / 172 |
| max length | 512 | 256 (capped by `max_new_tokens`) |
| **Deconfounded AUROC** | **0.803** (+0.007 vs 0.796) | **0.746** (−0.050 vs 0.796) |

**Revised Exp 1 verdict:** A real length confound exists. Raw-length
residualization drops AUROC by 0.050 (0.796 → 0.746). The original Exp 1
did not catch this because it residualized against a different quantity.

Recommend: when reporting MATH-500 AUROC for topo-confidence, report both
0.796 (uncorrected) and 0.746 (length-deconfounded) figures.

### 🟡 REVISED: Exp 3a verdict was overreach

Original verdict: "PH measures COVARIANCE, not topology."

Problems with this claim:
1. Code used **diagonal variance**, not full covariance (D1)
2. One of the 6 features (`H0_n_features`) is **structurally n_points−1**,
   identical in real vs null by construction (D2)
3. CV std 0.245 on n=100 holdout → **no statistical power** to separate
   real (0.678) from null (0.700) (D3)
4. Only 6 raw PH features tested — **does not speak for ABC-44**

Revised wording (pending v2 rerun): "These 5 raw PH features (H0_n_features
excluded) do not outperform a full-covariance Gaussian null on this data.
The predictive content is explained by matched covariance structure. This
finding is **narrow** — it applies only to the summary-statistic PH
features, not to ABC-44 which includes geometric terms (cosines, norms,
velocities, cross-products) that were not tested here."

### 🟢 CONFIRMED: Exp 4 verdict holds under matched CV

Original Exp 4 compared LR (50-fold CV) against XGBoost (5-fold CV) —
unfair. Matched-CV rerun (both cv=5):

| Model | CV5 AUROC | CV std |
|---|---|---|
| LR (baseline) | 0.701 | 0.029 |
| XGBoost (best, regularized) | 0.696 | 0.032 |

Lift = **−0.005** (within 1σ noise). Verdict holds: **LR is appropriate,
signal is linear**.

O-information: `hoi.metrics.Oinfo` on 44 features at order 3-5 tries to
allocate 3.07 TiB — structural blowup, not a dtype bug. Abandon O-info,
XGBoost vs LR comparison already answers the redundancy/synergy question.

---

## Honest numbers table (post-audit)

| Config | AUROC (uncorrected) | AUROC (raw-length-deconfounded) | CV5 |
|---|---|---|---|
| ABC-44 | 0.7961 | **0.7459** | 0.7012 |
| Length alone (raw tokenizer) | 0.7055 | — | — |
| Best XGBoost (regularized) | 0.695 | — | 0.696 |

**The gap vs baselines (vote_margin 0.767, first_token 0.637) is tightened
by the length-deconfound finding but not eliminated.** ABC-44 still beats
vote_margin when both are computed fairly on the same split — but the
"+0.057 gap" needs a caveat about the shared length correlate.

---

## RunPod execution plan

### 0. Prep

```bash
# On your laptop — push new scripts to RunPod volume:
POD=<your-pod-id>   # e.g. 0agitikupjg259
scp pathway9/exp3a_full_cov_null.py pathway9/exp2_orthogonality.py \
    pathway9/exp3b_token_shuffle.py pathway9/exp5_cross_domain_transfer.py \
    pathway9/run_runpod.sh \
    root@$POD:/workspace/topo-confidence/pathway9/
```

### 1. Run everything

```bash
# On the pod (GPU pod — H100 recommended):
cd /workspace/topo-confidence
bash pathway9/run_runpod.sh
```

Order is cheap-CPU → expensive-CPU → GPU:
1. `exp3a_v2` (~30-60 min CPU) — full-cov null via SVD low-rank sampler (~200× faster than multivariate_normal)
2. `exp2` (~2 min CPU) — loads cached PH + CoE feature arrays
3. `exp5` (~15 min CPU) — BBH feature compute + transfer LRs
4. `exp3b` (~15 min GPU) — 50 problems × 5 shuffles on Qwen2.5-1.5B

Each script writes `pathway9/results/<exp>.done` when successful. Rerunning
the bash runner skips completed stages.

### 2. CPU-only alternative

If you restart as CPU-only pod (cheaper, saves ~$0.90/hr for H100):
```bash
bash pathway9/run_runpod.sh --skip-exp3b
```
Run `exp3b` separately later on a GPU pod.

### 3. Retrieve results

```bash
# Back on laptop:
scp root@$POD:/workspace/topo-confidence/pathway9/results/*.json \
    ~/topo-confidence/pathway9/results/
scp root@$POD:/workspace/topo-confidence/pathway9/results/*.done \
    ~/topo-confidence/pathway9/results/
```

### 4. Write SUMMARY.md locally

Once all 4 RunPod results arrive, synthesize into `pathway9/SUMMARY.md`
with these sections:
1. Length confound verdict (Exp 1 v2) — significant, 0.050 drop
2. PH vs CoE: redundant or complementary? (Exp 2)
3. Null test verdict: topology, covariance, or length? (Exp 3a v2, 3b, 3c)
4. Classifier verdict: LR vs XGBoost (Exp 4 v2) — LR fine
5. Cross-domain verdict (Exp 5)
6. Honest final numbers table (all deconfounded AUROCs)
7. Recommendation: continue, pivot to CoE, or publish what you have

---

## Required data on RunPod

Already present on volume from pathway 8 runs:

```
pathway1/phase1/features_train400.npy          # ABC-44 train
pathway1/phase1/features_holdout100.npy        # ABC-44 holdout
pathway2/track_a/phase0/baseline_correct.npy   # OLD labels (for SSS split)
pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy   # NEW labels
data/experiment1_v2/trajectories.npz           # raw clouds (for exp3a v2)
data/experiment1_v2/trajectory_meta.json       # generated_texts
pathway8_layerwise/results/exp1_features_all.npy   # (500, 168) layer-wise PH
pathway8_layerwise/results/exp2_coe_features.npy   # (500, 60) CoE
pathway8_layerwise/data/math500/problem_XXX.npz    # all-layer states (exp5)
pathway8_layerwise/data/bbh/<subset>/problem_XXX.npz  # 3 subsets × 250 (exp5)
```

If any of these are missing on the volume, the failing script will crash
with `FileNotFoundError`. The runner will stop and print the missing path.

---

## Key files

```
pathway9/
├── AUDIT.md                           # Soup-to-nuts audit (8 defects)
├── HANDOFF.md                         # This file
├── run_local_experiments.py           # v1 (Exp 1, 3a, 3c, 4) — done
├── run_fixes.py                       # audit fix reruns — done (exp1_v2, exp4_v2)
├── exp3a_full_cov_null.py             # ← run on RunPod (audit fix D1+D2)
├── exp2_orthogonality.py              # ← run on RunPod
├── exp3b_token_shuffle.py             # ← run on RunPod (GPU)
├── exp5_cross_domain_transfer.py      # ← run on RunPod
├── run_runpod.sh                      # ← driver with .done-based resume
└── results/
    ├── exp1_length_deconfound.json    # orig Exp 1 (traj_rows)
    ├── exp1_raw_length_v2.json        # Exp 1 v2 (raw tokenizer) — REVISED VERDICT
    ├── exp3a_gaussian_null.json       # orig Exp 3a (diag variance, 6 feats)
    ├── exp3c_count_control.json
    ├── exp4_xgboost_oinfo.json        # orig Exp 4 (mismatched CV)
    ├── exp4_xgboost_oinfo_v2.json     # Exp 4 v2 (matched cv=5) — CONFIRMED
    └── (pending: exp3a_gaussian_null_v2.json, exp2_*, exp3b_*, exp5_*)
```

---

## Quick reference — what each remaining script does

**exp3a_full_cov_null.py** (CPU, ~30-60 min, 500 problems)
- Drops `H0_n_features` (was structurally identical in real vs null).
- Uses SVD low-rank sampler for full-covariance Gaussian draws (cov has
  rank ≤ n_tokens-1 in a (n_tokens, 1536) cloud, so we sample in that
  subspace — ~200× faster than np.random.multivariate_normal).
- Tests: does real PH beat a proper full-cov null?

**exp2_orthogonality.py** (CPU, ~2 min)
- Loads cached `exp1_features_all.npy` (PH, 168) and
  `exp2_coe_features.npy` (CoE, 60) from pathway 8.
- Trains LR on each, measures prediction correlation on holdout.
- Computes simple average ensemble and nested-CV stacked ensemble.
- Decision: complementary if lift >0.01, redundant if r>0.8.

**exp3b_token_shuffle.py** (GPU, ~15 min, 50 problems × 5 shuffles)
- Loads Qwen2.5-1.5B-Instruct bf16 sdpa.
- For each sampled problem: forward pass on original token ids; forward
  pass on 5 independent permutations; PH features on last-layer hidden
  states.
- Reports per-feature MAE between real and shuffled vs inter-problem std.
- If MAE/std < 1 for ≥80% of features → PH is order-independent.
- Note: forward passes completion text alone (not prompt+completion) as
  an ablation; full test would rewrite extraction pipeline.

**exp5_cross_domain_transfer.py** (CPU, ~15 min)
- Computes PH / CoE / D2H-lite features on MATH-500 and BBH-pooled (750).
- Cross-domain: train on one, test on the other, with TRAIN-only scaler.
- Also within-domain baseline (80/20 split) for each method.
- Decision: transfer signal if best avg AUROC >0.65.
- Note: layer-wise PH has per-layer PCA fit on train — if dims don't
  match between domains, transfer is skipped for PH with a warning.

---

## Context budget notes (for the next session)

- Last two Claude Code sessions died on context overflow after reading
  pathway 8 scripts plus the pathway 8 summary (~400 lines each).
- Don't re-read pathway 8 code in the next session. Reference this handoff
  + AUDIT.md only.
- `hoi.metrics.Oinfo` loads JAX, which OOMs in WSL2 during import. Don't
  bother with O-information. XGBoost vs LR already answers the question.
- MEMORY.md is at 210 lines (over 200 limit). Consider condensing next
  session.
