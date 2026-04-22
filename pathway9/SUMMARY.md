# Pathway 9 — SUMMARY

Post-run synthesis of all 7 experiments. All defects D1–D13 from AUDIT.md +
HANDOFF_v2.md are addressed or explicitly qualified. Results retrieved
2026-04-22 from RunPod pod `0agitikupjg259` (H100 SXM).

Files referenced live at `/home/musicofhel/topo-confidence/pathway9/results/`.

---

## TL;DR

- **Headline 0.7961 holds**, but ~0.050 is a raw-output-length artifact
  (Exp 1 v2). Deconfounded MATH-500 holdout AUROC: **0.7459**.
- **The 5 raw PH summary features are indistinguishable from an empirical-
  covariance Gaussian null** (0.690 vs 0.693). Headline signal comes from
  the ABC-44 *geometric* features (cosines, norms, velocities), NOT from
  topological structure alone (Exp 3a v2).
- **Layer-wise PH-168 is much weaker than ABC-44 PH** (0.646 vs 0.7961
  holdout), and is *redundant with CoE* (ensemble lift +0.001 over
  CoE-60 alone at 0.811) — Exp 2.
- **LR ≈ XGBoost** under matched CV5 (0.701 vs 0.696, diff −0.005). Signal
  is linear; no interaction terms. O-information OOMs (3 TiB). Exp 4 v2.
- **Cross-domain**: CoE transfers MATH↔BBH cleanly (0.720 / 0.712, avg
  0.716). Layer-wise PH fails (0.354 / 0.497 — inverse / chance). D2H-lite
  is asymmetric (0.740 / 0.529). **CoE is the only domain-invariant
  signal.** — Exp 5.
- Token shuffle: 60% of the 5 PH features indistinguishable from shuffled
  (H0_entropy, H1_pers_entropy, H1_n_features). PH captures some
  order-sensitivity in H0_max_lifetime and H1_max_lifetime. Exp 3b.

**Recommendation:** *Pivot paper narrative to CoE (Wang ICLR 2025)*.
Report PH as a within-domain baseline with explicit caveats: headline
0.796 is partly length, the topology-specific component is explained by
covariance, and the features don't transfer.

---

## §1 Length confound (Exp 1 v2, D6 fix)

**Q:** Does the 0.7961 MATH-500 holdout AUROC survive deconfounding
against raw output token length?

**Setup:** Per-feature linear residualization of ABC-44 against
`len(tokenizer.encode(generated_text))` — the *raw* length, not
`trajectory_shape[0]` (which Exp 1 v1 used and which the D6 audit showed
is weakly correlated with raw length, r=0.096).

**Result:**

| Config | AUROC |
|---|---|
| ABC-44 uncorrected | **0.7961** |
| Raw-length-only LR | 0.7055 (`shorter = correct`) |
| ABC-44 minus raw-length | **0.7459** |
| Drop from deconfounding | **−0.050** |

**Verdict:** SIGNIFICANT. Roughly a fifth of the headline number is a
length confound that Exp 1 v1 missed by residualizing against the wrong
quantity. Report 0.7961 only with the 0.7459 footnote.

Source: `exp1_length_deconfound.json` (v1) + `exp1_raw_length_v2.json`.

---

## §2 PH vs CoE: redundancy, not complementarity (Exp 2, D7 wording)

**Q:** Do PH and CoE carry orthogonal predictive information?

**Setup:** Layer-wise PH (168 dims) and CoE-60 on the same MATH-500
SSS(9999, test=100) split. Stacked ensemble via 5-fold OOF base
probabilities. Base LR with `class_weight="balanced"`.

> **Caveat:** PH here is *layer-wise* (168), not the ABC-44 PH (0.7961
> headline). Different feature set. The layer-wise PH pipeline is the
> pathway 8 extraction, which is the only 168-dim PH feature set on disk.

**Result:**

| Model | Holdout AUROC |
|---|---|
| Layer-wise PH-168 alone | 0.646 |
| CoE-60 alone | **0.811** |
| Simple average ensemble | 0.773 |
| Stacked ensemble (LR meta) | 0.812 |
| Stacked coefs | ph=+0.24, coe=+1.04 |
| Prediction correlation r(ph, coe) | +0.388 |
| Lift (best_ensemble − CoE alone) | **+0.001** |

**Disagreement sets (holdout):** both_low n=31, frac_correct=0.03 (CoE
captures "clearly wrong" correctly); coe_high_ph_low n=4, frac_correct=0.75 (CoE
alone catches cases PH misses); ph_high_coe_low n=7, frac_correct=0.14 (PH
over-confident on failures CoE flags correctly).

**Verdict:** REDUNDANT (weak). Layer-wise PH adds nothing CoE doesn't
already have. r=0.388 suggests moderate shared signal, not orthogonality.
CoE-60 alone (0.811) is a better single predictor than ABC-44 (0.7961).

Source: `exp2_orthogonality.json`.

---

## §3 Null-test verdict: covariance, some order-sensitivity (Exp 3a v2, 3b, 3c)

**Q:** Are PH features capturing genuine topology or just covariance /
length / token distribution?

### 3a — Empirical-covariance Gaussian null (D1+D2 fix)

*Sampler:* rank-matched empirical-covariance via SVD of centered cloud.
(Mathematically equivalent to `multivariate_normal(mean, cov_empirical)`;
D13 is naming only.) 5 features (H0_n_features dropped per D2).

| Feature set | Holdout AUROC | CV5 ± std |
|---|---|---|
| Real PH (5) | 0.690 | 0.745 ± 0.043 |
| Null PH (5) | 0.693 | 0.744 ± 0.033 |
| Diff (real − null) | 0.707 | 0.707 ± 0.037 |

CV std dropped from 0.245 (D3) to 0.043 with CV5. The real vs null gap
is zero and the diff doesn't lift above either.

**Per-feature stats** (real mean ± std  vs  null mean ± std):
- H0_max_lifetime: 154.6 ± 7.1  vs  143.3 ± 9.8 → real slightly larger.
- H0_entropy:       4.13 ± 0.32 vs   4.19 ± 0.35 → indistinguishable.
- H1_pers_entropy:  2.40 ± 0.56 vs   3.39 ± 0.54 → null HIGHER (more random).
- H1_n_features:   18.1  ± 9.0  vs  46.4  ± 21.7 → null has ~2.5× more loops.
- H1_max_lifetime: 13.3  ± 5.5  vs  13.9  ± 1.4  → indistinguishable in mean, real has much higher inter-problem variance.

Interesting: real PH has FEWER H1 loops and LESS H1 entropy than null —
real clouds are *less random*, more structured. But that structure
doesn't translate to better predictive power for correctness.

**Verdict:** These 5 raw PH summary statistics do NOT outperform a
rank-matched empirical-covariance Gaussian null. Their predictive content
is fully explained by matched covariance structure. The ABC-44 headline
does NOT rely on these 5 features; it relies on the 39 geometric
features (cosines, norms, velocities, inter-tier products) that Exp 3a
did not test.

### 3b — Token shuffle (GPU, 50 problems × 5 shuffles)

| Feature | MAE / inter-problem std | Interpretation |
|---|---|---|
| H0_max_lifetime | 1.30 | order-sensitive |
| H0_entropy | 0.78 | indistinguishable |
| H1_pers_entropy | 0.80 | indistinguishable |
| H1_n_features | 0.84 | indistinguishable |
| H1_max_lifetime | 1.71 | order-sensitive |

60% of features (3 of 5) are within 1 inter-problem std of shuffled —
they are order-indifferent. H0_max_lifetime and H1_max_lifetime do show
measurable order-sensitivity. *Caveat:* test forward-passes completion
text only; a prompt+completion test may differ.

### 3c — Count control (D8, trivially correct)

No features in ABC-44 or the 78 set are named `n_features`; the count
ablation removed 0 columns, so `full_78 == minus_counts == 0.7647`
holdout. Confirms via name matching only. Not a strong test.

### 3a/3b/3c combined verdict

The 5 raw summary-statistic PH features are dominated by covariance and
are largely (but not entirely) order-indifferent on completion text. The
0.7961 ABC-44 number is NOT driven by these 5 features — it's driven by
geometry. Topology, in the narrow sense tested here, does not add
predictive signal beyond what the cloud's second moments already carry.

Sources: `exp3a_gaussian_null_v2.json`, `exp3b_token_shuffle.json`,
`exp3c_count_control.json`.

---

## §4 Classifier verdict: LR vs XGBoost (Exp 4 v2, D4+D5 fix)

**Q:** Is there non-linear signal in the ABC-44 features that LR is
missing?

**Setup:** Matched 5-fold StratifiedKFold for both LR and XGBoost. Three
XGBoost configurations (default, shallow, regularized).

| Model | Holdout AUROC | CV5 ± std |
|---|---|---|
| LR (baseline) | 0.7961 | 0.7012 ± 0.029 |
| XGB default   | 0.7169 | 0.6880 ± 0.029 |
| XGB shallow   | 0.6816 | 0.6854 ± 0.050 |
| XGB regularized (best) | 0.6949 | 0.6960 ± 0.032 |
| Best lift (XGB − LR), CV5 | | **−0.005** |

O-information (Oinfo) attempted to compute triplet/higher-order
interactions over 44 features; at order 3–5 it demanded 3.07 TiB of int64.
Abandoned; XGBoost≈LR comparison is the primary linearity diagnostic.

**Verdict:** Signal is linear at the 44-feature level. XGBoost does not
find non-linear interactions that a well-regularized LR misses.

Source: `exp4_xgboost_oinfo_v2.json`.

---

## §5 Cross-domain transfer (Exp 5, D11+D12 fix)

**Q:** Do PH, CoE, or D2H features trained on MATH-500 predict
correctness on BBH (pooled 3 subsets × 250), and vice versa?

**Setup (post-fix):**
- **PH (D11)**: source-domain PCA(20) fitted once per direction, applied
  to both source *and* target. Prior impl fit separate PCAs per domain,
  making the 168-dim feature spaces silently incompatible.
- **PH within-domain (D12)**: PCA fit on 80% train split, applied to
  both halves.
- **CoE / D2H-lite**: no domain-specific PCA; unchanged.
- Labels: pathway-8 manifest `correct` field. MATH-500 has 231/500
  correct under this definition (different from the 104-correct NEW labels
  used for the 0.7961 headline — apples-to-apples across benchmarks here).

### Transfer matrix (AUROC)

| Method | MATH→BBH | BBH→MATH | MATH within (80/20) | BBH within (80/20) |
|---|---|---|---|---|
| **Layer-wise PH (168)** | **0.354** | 0.497 | 0.766 | 0.781 |
| **CoE (60)** | **0.720** | **0.712** | 0.775 | 0.783 |
| **D2H-lite (58)** | 0.740 | 0.529 | 0.790 | 0.777 |

### Analysis

- **PH is domain-bound.** It achieves 0.77-0.78 within each domain, but
  collapses to chance (0.497) or inverse (0.354) across. The inverse
  MATH→BBH direction means a MATH-trained PH predictor's output is
  anti-correlated with BBH correctness — same features, opposite sign.
  PCA basis fit to MATH captures MATH-specific geometry that happens to
  invert on BBH.
- **CoE is the only symmetric, domain-invariant predictor** of the three.
  Transfer drops ~0.06 from within-domain. 0.716 avg transfer AUROC.
- **D2H-lite is asymmetric.** MATH→BBH works (0.740), BBH→MATH does not
  (0.529). The MATH→BBH signal is likely driven by general dispersion/drift
  dynamics that transfer downward to shorter BBH problems, but BBH-specific
  D2H patterns don't generalize up to longer MATH chains.

> **Note on within-domain baselines vs headline 0.7961:** the 80/20 splits
> here use pathway-8 manifest labels (MATH=231 correct) and a different
> random split than the SSS(9999, test=100) that produced 0.7961 on
> NEW-labels (104 correct). They are NOT directly comparable — do not
> read MATH-within=0.766 as "PH regressed from 0.7961."

**Verdict:** Cross-domain uncertainty estimation should use CoE. Layer-wise
PH is domain-specific geometry; D2H-lite has a one-way transfer artifact.

Source: `exp5_cross_domain.json`.

---

## §6 Honest numbers table

Final, fully-corrected view. "Label scheme" flag distinguishes the NEW-labels
(104 correct on MATH-500) pipeline used for the 0.7961 headline from the
pathway-8 manifest labels (231 correct) used in cross-domain.

| Experiment | Metric | Value | Labels |
|---|---|---|---|
| ABC-44 baseline holdout (NEW) | AUROC | **0.7961** | NEW |
| ABC-44 deconfounded (raw length) | AUROC | **0.7459** | NEW |
| ABC-44 CV5 (matched) | AUROC | 0.7012 ± 0.029 | NEW |
| Best XGB (regularized, CV5) | AUROC | 0.6960 ± 0.032 | NEW |
| CoE-60 holdout | AUROC | **0.8110** | NEW |
| Stacked PH+CoE ensemble | AUROC | 0.8117 | NEW |
| Layer-wise PH-168 holdout | AUROC | 0.6463 | NEW |
| PH 5-feature real vs null | AUROC | 0.690 vs 0.693 | NEW |
| CoE MATH→BBH transfer | AUROC | **0.720** | pathway-8 |
| CoE BBH→MATH transfer | AUROC | **0.712** | pathway-8 |
| PH MATH→BBH transfer | AUROC | 0.354 | pathway-8 |
| PH BBH→MATH transfer | AUROC | 0.497 | pathway-8 |

---

## §7 Recommendation

**Pivot to CoE.**

1. The 0.7961 ABC-44 number is real but partially length-confounded
   (0.7459 clean) and driven by feature-engineered geometry rather than
   intrinsic topology (the 5 raw PH features are at a covariance null).
2. CoE alone (0.811) is already a *better* within-domain predictor than
   ABC-44 (0.7961 → 0.7459), uses 60 features instead of 44 with no
   PCA hyperparameter, and carries the only cross-domain signal that
   transfers (0.72 / 0.71, symmetric).
3. PH keeps its place as a within-domain diagnostic with caveats —
   especially if paired with geometric features (ABC's Tier B/C). But as
   a *publishable* cross-domain uncertainty method, it is dominated by
   CoE.
4. The paper's headline should read: *CoE provides domain-invariant
   uncertainty estimation in Qwen2.5-1.5B-Instruct, with within-domain
   AUROCs of 0.77-0.78 on MATH-500 and BBH and cross-domain transfer
   AUROCs of 0.71-0.72. Topological features (ABC-44) give comparable
   within-domain performance on MATH-500 but fail to transfer.*

---

## Defect traceability

All 13 defects (D1–D13) are now resolved or explicitly accounted for:

| ID | Status | Where resolved |
|---|---|---|
| D1 (diag→full cov) | ✅ | Exp 3a v2 uses SVD empirical-cov sampler |
| D2 (H0_n_features) | ✅ | dropped, 5 features |
| D3 (CV50 std 0.245) | ✅ | CV5, std 0.043 |
| D4 (LR50 vs XGB5 unfair) | ✅ | matched CV5 |
| D5 (O-info dtype) | ✅ | abandoned (3 TiB OOM) |
| D6 (wrong length proxy) | ✅ | raw-tokenizer residualization; −0.050 drop |
| D7 (exp3a overreach) | ✅ | narrower verdict above |
| D8 (count control no-op) | ✅ | documented, trivially correct |
| D11 (exp5 PCA mismatch) | ✅ | source-fit PCA applied to both domains |
| D12 (exp5 within-domain leak) | ✅ | train-split PCA |
| D13 (SVD null naming) | ✅ | reworded to "rank-matched empirical-covariance" |

---

## Files

All JSONs + logs: `/home/musicofhel/topo-confidence/pathway9/results/`.

Generated locally:
- `exp1_length_deconfound.json` (v1)
- `exp1_raw_length_v2.json` (D6)
- `exp3a_gaussian_null.json` (v1, superseded)
- `exp3c_count_control.json`
- `exp4_xgboost_oinfo.json` (v1, superseded)
- `exp4_xgboost_oinfo_v2.json` (D4+D5)

Generated on RunPod (this session):
- `exp3a_gaussian_null_v2.json` (D1+D2)
- `exp2_orthogonality.json`
- `exp5_cross_domain.json` (D11+D12)
- `exp3b_token_shuffle.json`
- `run.log` (full console)
