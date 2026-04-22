# Pathway 9 Audit — 2026-04-21

Soup-to-nuts review of the 4 local experiments completed before the two
Claude sessions died. Audit covers `run_local_experiments.py`, the four
JSON result files, and the claims in `HANDOFF.md`.

## TL;DR

- **Exp 1 (length deconfound)**: Methodologically sound. Verdict stands, but
  the "token count" being residualized is `trajectory_shape[0]`, which may
  not equal raw completion length. The claim needs a tighter phrasing.
- **Exp 3a (Gaussian null)**: Results are real, but the HANDOFF's verdict
  ("PH measures COVARIANCE, not topology") is **overreach**. Three issues:
  (1) null uses diagonal *variance*, not full covariance; (2) H0_n_features
  is structurally identical in real and null (point count, not topology),
  biasing the comparison; (3) CV std is 0.25 — the test has essentially no
  power to separate real (0.678) from null (0.700). Restate as: "these 6
  raw PH features do not distinguish real hidden states from variance-matched
  Gaussian noise as predictors of correctness".
- **Exp 3c (count control)**: Correct but trivially so. Both ABC-44 and the
  full 78 contain zero count-by-name features, so the "remove counts" test
  was a no-op (full_78_auroc == minus_counts_auroc == 0.7647 exactly).
  Conclusion "0.796 is count-free" is fine — just not a test of anything.
- **Exp 4 (XGBoost vs LR)**: **CV comparison is not apples-to-apples.** LR
  uses 50-fold CV, XGBoost uses 5-fold CV. Different train-fold sizes bias
  the mean AUROC. The -0.009 lift is within noise. "Signal is linear"
  plausible but unproved. O-information crashed (dtype bug), no info there.

**Pipeline integrity**: Baseline reproduces at 0.7960784 across all four
experiments — no silent drift. Feature/label alignment OK in all code paths
inspected.

**Ready for RunPod?** Yes, with caveats. Before writing SUMMARY.md, decide
whether to (a) accept the HANDOFF's current verdict wording, or (b) rerun
exp3a with full covariance + matched classifier methodology for a cleaner
claim. I'd recommend (b) — it's ~30 min of local compute.

---

## Defect register

### D1 — Exp 3a diagonal-variance, not covariance-matched null
**Severity:** Medium. Affects interpretation, not direction of result.
**File:** `pathway9/run_local_experiments.py:267-277`.
**What the code does:** Generates synthetic clouds as
`randn(n, d) * sqrt(var) + mean` — independent Gaussians with matched
per-dimension variance.
**What the HANDOFF claims:** "Gaussian-matched null" using full covariance.
**Why it matters:** Full-covariance null preserves cross-dim correlations;
variance-only null strips them. The observed result (null matches real)
therefore says "PH features don't leverage cross-dimensional structure" —
not the stronger "PH features are covariance proxies".
**Fix:** Either rename the verdict (s/covariance/variance/) or rerun with
full `np.random.multivariate_normal(mean, cov+eps*I, n)`.

### D2 — Exp 3a: H0_n_features is structurally identical in real and null
**Severity:** Medium. Contaminates the 6-feature comparison.
**File:** JSON `exp3a_gaussian_null.json` line 32-36: real_mean=70.09,
null_mean=70.09, real_std=22.39, null_std=22.39, diff_mean=0.0.
**Why:** H0_n_features counts finite H0 bars, which equals `n_points − 1`
(modulo degeneracies). Both clouds share the same `n_points`, so the
feature is a deterministic function of cloud size, not topology. Including
it in the 6-feature set makes the null classifier trivially match real on
this column.
**Fix:** Rerun the null test with this feature excluded (5 features), or
note the caveat and decouple the count from the topology claim.

### D3 — Exp 3a: CV std 0.245 → no statistical power
**Severity:** Medium. Invalidates the CV-based comparison.
**File:** `exp3a_gaussian_null.json` lines 9, 15, 19: cv_std=0.245-0.262.
**Why:** 50-fold CV on 400 samples with 104 positives → ~2 positives per
fold → many folds have 0 or 1 positive → AUROC unstable or undefined.
Mean AUROC comparable, but confidence intervals overlap heavily. DeLong
on n=100 holdout gives ±0.13 CI at AUROC 0.7 — real (0.678) and null
(0.700) are indistinguishable.
**Fix:** Use cv=5 or cv=10 (StratifiedKFold). Report DeLong p-value for
holdout difference.

### D4 — Exp 4: unfair CV scheme (LR 50-fold vs XGBoost 5-fold)
**Severity:** Medium. Makes the -0.009 lift uninterpretable.
**File:** `run_local_experiments.py:90` (`n_cv=50` for LR) vs `506-509`
(`cv=5` for XGBoost).
**Why:** 50-fold trains on 98% of the data per fold; 5-fold trains on 80%.
Larger train folds produce slightly higher CV AUROC for any reasonable
classifier. LR's CV 0.705 is likely 0.01-0.02 inflated relative to a
5-fold run; XGBoost's 0.696 is on the pessimistic side.
**Fix:** Rerun both with cv=5. If LR still ≥ XGBoost by >0.01, "signal is
linear" holds. Otherwise it's inconclusive.

### D5 — Exp 4: O-information crashed with dtype error
**Severity:** Low. Not critical (XGBoost vs LR is the primary diagnostic)
but leaves the redundancy/synergy question unanswered.
**File:** `exp4_xgboost_oinfo.json` line 67:
`"error": "data dtype should be float, not int64"`.
**Cause:** `np.digitize` returns int64; `hoi.metrics.Oinfo` wants float.
**Fix:** `X_disc = X_disc.astype(float)` before calling Oinfo, or feed raw
standardized continuous values directly (hoi supports both modes).

### D6 — Exp 1: "token count" source is trajectory_shape[0], not raw output length
**Severity:** Low. Confound test is still valid — just of a different quantity.
**File:** `run_local_experiments.py:62`: `s[0] for s in meta["trajectory_shapes"]`.
**What this is:** Number of token-position rows in the saved hidden-state
trajectory. Spot check: problem 0 has shape (67, 1536); its displayed
generated text looks much longer than 67 tokens. The trajectory may have
been captured at a specific layer/stride/window, not 1-to-1 with
completion tokens.
**Impact:** The test shows "residualizing against trajectory-row count
doesn't lose AUROC", which is slightly narrower than "residualizing
against raw output length". If the two quantities differ materially, a
stronger length confound could still exist and be untested.
**Fix:** Add a second residualization against `len(tokenizer.encode(generated_text))`
as cross-check. Cheap: ~30 sec with a tokenizer.

### D7 — Exp 3a scope creep in HANDOFF verdict
**Severity:** Medium. The one-line takeaway in HANDOFF.md §"Key Takeaways"
item 2 is broader than what the experiment tested.
**What HANDOFF says:** "PH features measure covariance, not topology."
**What exp3a tested:** The 6 raw PH features (H0/H1 max_lifetime, n_features,
entropy) from a specific PCA(45)+subsample(100)+ripser pipeline, against a
diagonal-variance null, on 500 problems, with underpowered 50-fold CV.
**What it did NOT test:** The 44 ABC-tier production features, the layer-wise
PH features (168, pathway8), or any full-covariance null.
**Fix:** Rephrase takeaway as "The 6 raw summary-statistic PH features do
not outperform variance-matched random clouds as predictors of correctness,
suggesting these particular features rely on marginal spread rather than
genuine topological structure. The ABC-44 result is likely driven by
non-PH geometric features (cosines, norms, velocities, interactions)."

### D8 — Exp 3c: count-control test was a no-op
**Severity:** Low. Conclusion is correct ("0.796 is count-free") but it
was confirmed by name-matching, not by ablation.
**File:** `exp3c_count_control.json` lines 3-4: `count_features_in_abc44: []`,
`count_features_in_78: []`. Then `full_78_auroc == minus_counts_auroc ==
0.7647` because the column filter removed zero columns.
**Why:** The name regex `"n_features" in n.lower()` missed any count-adjacent
features renamed differently (e.g., something encoding point count under
another name). The HumanEval audit claim "n_features columns at r=0.82 with
token count" is about RAW PH outputs, not the pathway1 78 set.
**Fix:** Instead of string matching, compute `|corr(feat, n_points)|` for
all 78 features, flag those > 0.5, and ablation-test those. This catches
count-proxies regardless of their name.

---

## What's fine

- Baseline AUROC 0.7960784 reproduces exactly across exp1, exp3c, exp4.
  No silent pipeline drift.
- SSS split reconstruction matches pathway8's config (`random_state=9999`,
  `test_size=100`, sorted indices, OLD labels for split then NEW labels
  for training).
- ABC-44 column selection matches (`tiers.get(name) in ("A", "B", "C")`,
  44 columns from 78).
- StandardScaler is fit on train only, applied to holdout (exp1 deconfound
  loop: `LinearRegression().fit(tc_tr, X_tr)` then `.predict(tc_ho)` for
  holdout).
- Length residualization is per-feature linear and uses train statistics
  only — no label leakage.
- Checkpoint-based resumption via `.done` files works; main() reloads
  completed experiments' JSON rather than re-running.

---

## Cross-experiment consistency checks

| Metric | Exp 1 | Exp 3c | Exp 4 | Match? |
|---|---|---|---|---|
| ABC-44 holdout AUROC | 0.79608 | 0.79608 | 0.79608 | exact |
| ABC-44 CV AUROC | 0.70476 | 0.70476 | 0.70476 | exact |

All three experiments used `fit_lr` with the same params. Values match to
7 decimals → pipeline is deterministic and consistent. ✓

---

## Before SUMMARY.md — recommended fixes

**Must-fix (blocks trust in HANDOFF claims):**

1. **D4**: Rerun exp4 with `cv=5` for both LR and XGBoost. 5 min local.
2. **D7**: Rewrite exp3a verdict in HANDOFF.md + JSON to match narrower
   experimental scope. No rerun needed, just text edit.

**Should-fix (strengthens claims):**

3. **D1 + D2**: Rerun exp3a with full-covariance null and without
   H0_n_features. ~30 min local.
4. **D6**: Add a raw-output-length cross-check in exp1. ~2 min.
5. **D5**: Fix the O-info dtype and rerun just that subsection. ~5 min.

**Can-skip (cosmetic):**

6. **D8**: Replace string-match with correlation-based count detection.
   Only matters if we're suspicious the 78 set has unnamed count proxies.
   Low prior.

---

## Audit of the 3 remaining experiments — not yet run

None of exp2 (orthogonality), exp3b (token-shuffle), exp5 (cross-domain)
have code yet. HANDOFF has stubs. Flagging now what to watch for when
they're written so audit isn't needed a third time:

- **Exp 2**: Use *nested* CV for stacking — out-of-fold PH probs and CoE
  probs from `cross_val_predict(method="predict_proba")` on train, then
  fit the meta-LR on those, evaluate on holdout via refitted base models.
  Naive stacking leaks labels.
- **Exp 2**: Report DeLong p-value for ensemble vs each individual model.
  A +0.01 holdout AUROC lift on n=100 is well within noise.
- **Exp 3b**: Seed the shuffle RNG for reproducibility. Use the SAME 6-PH
  pipeline as exp3a (or be explicit if you change it). Also worth: compare
  shuffle vs Gaussian null — if shuffle PH matches real PH more closely
  than Gaussian null does, that's diagnostic.
- **Exp 5**: Critical — fit StandardScaler on the TRAIN domain only, then
  apply to target domain. Easy to accidentally refit on target. Also: the
  ABC-44 features have MATH-500-specific PCA; they will NOT transfer to
  BBH without regenerating features in the target domain's feature space.
  This may be why "PH cross-domain" is poor — the features are not
  domain-invariant, not because the signal isn't.

---

## Files referenced

- `pathway9/run_local_experiments.py` (645 lines) — the 4 experiments
- `pathway9/results/exp1_length_deconfound.json` (175 lines)
- `pathway9/results/exp3a_gaussian_null.json` (67 lines)
- `pathway9/results/exp3c_count_control.json` (13 lines)
- `pathway9/results/exp4_xgboost_oinfo.json` (70 lines)
- `pathway9/HANDOFF.md` — prior session's claims
