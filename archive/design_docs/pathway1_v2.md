# Pathway 1: Validation, Ablation & Routing Prototype (v2)

## Context you need to know

This prompt is part of a 4-pathway research arc investigating whether topological features of LLM hidden states can predict — and eventually improve — model correctness. A prior CORAL run used 3 Claude-Opus agents to engineer features from persistent homology (PH) descriptors of Qwen2.5-1.5B hidden states on MATH-500, reaching 0.9779 AUROC on a fixed 50-fold CV protocol. That number is almost certainly inflated by evaluation overfitting (676 attempts against the same CV split, seed=42). Pathway 1 answers the question: **what's the honest signal, and is it enough to build a router on?**

Everything in Pathways 2–4 (activation steering, topology-aware fine-tuning, topologically-informed distillation) is contingent on the results here. Do not skip steps or cut corners on statistical rigor — a wrong answer here propagates downstream for weeks.

---

## Artifact locations

All paths are absolute. Do not guess — verify each exists before proceeding.

| Artifact | Path | Description |
|----------|------|-------------|
| Token trajectories | `~/topo-confidence/data/experiment1_v2/trajectories.npz` | 500 arrays (`traj_0`..`traj_499`), each `(n_tokens, 1536)`. Variable `n_tokens` (30–350). |
| Trajectory metadata | `~/topo-confidence/data/experiment1_v2/trajectory_meta.json` | `correct` key: binary 0/1 array (500 elements). Also: `ground_truths`, `generated_texts`, `trajectory_shapes`. |
| Layer states | `~/att-docs/data/transformer/math500_hidden_states_aligned.npz` | Key `layer_hidden_states`, shape `(500, 29, 1536)`. Mean hidden state per layer. |
| CORAL winning features | `coral-topo-auroc-run` branch: `coral_run_2026_04_08/winning_features.py` | 1407 lines. Function: `extract_features(trajectories, layer_states) -> (features, feature_names)`. |
| CORAL task definition | `coral-topo-auroc-run` branch: `coral_run_2026_04_08/task.yaml` | Grader config, baseline AUROC (0.713), tips, data paths. |
| Baseline feature extractor | `~/topo-confidence/topo_confidence/features.py` | `TopologicalFeatureExtractor` class, 13 features, used before CORAL. |
| Baselines | `~/topo-confidence/topo_confidence/baselines.py` | `output_entropy(logits_list)`: mean per-token Shannon entropy over generated tokens. Also `max_token_probability()`, `first_token_probability()`. |
| Hidden state extractor | `~/topo-confidence/topo_confidence/extractor.py` | `HiddenStateExtractor.extract_with_output(collect_logits=True)` — needed if logits aren't cached. |
| Evaluation script | `~/topo-confidence/scripts/experiment1_math500.py` | Contains `check_correct()`, CV protocol, LR config. |
| Feature validation | `~/topo-confidence/scripts/validate_features.py` | Per-feature AUROC, ablation, forward/backward selection. |
| Default config | `~/topo-confidence/configs/default.yaml` | Model, feature, calibration, generation, evaluation params. |

**Output directory:** All Pathway 1 artifacts go to `~/topo-confidence/pathway1/`. Create subdirectories `phase0/`, `phase1/`, `phase2/`, `phase3/`, `phase4/`.

---

## Labels

Correctness labels are binary 0/1, stored in `trajectory_meta.json` under the `"correct"` key. Ground truth is extracted from `\boxed{...}` in the MATH-500 solution field. Comparison uses normalized string matching via `check_correct()` in `experiment1_math500.py` (lines 99–104): strip whitespace, lowercase, remove `$` and `,`, attempt float parsing and integer rounding. The dataset is imbalanced: **57/500 correct (11.4%)**.

---

## Statefulness warning

The CORAL `extract_features()` pipeline has **three stateful operations** that create train/test leakage if not handled:

1. **PCA** (lines 308–312 of `winning_features.py`): `PCA(n_components=45, svd_solver="full")` is fit on ALL pooled token trajectories concatenated. If you fit on train-only tokens, the principal components change, and every downstream feature shifts.

2. **Rank-binning** (throughout the post-transform block): The `rb11` transform computes `np.argsort(np.argsort(raw_product)) / n` then digitizes into 11 bins. Ranks are batch-dependent — a sample's rank changes depending on which other samples are in the batch.

3. **StandardScaler** (in the evaluation pipeline, not in `winning_features.py`): `StandardScaler().fit_transform(features)` — zero-mean, unit-variance normalization. Must be fit on train only.

**Correct holdout procedure:**
- **PCA:** Fit on train-400 token trajectories only. Transform all trajectories (train and holdout) with the train-fitted PCA.
- **Rank-binning:** For each `rb11` product, compute raw product values for all samples using the train-fitted PCA features. Compute the holdout sample's percentile rank within the train distribution: `np.searchsorted(np.sort(train_raw), holdout_raw) / len(train)`. Digitize using the same `edges292` bins.
- **StandardScaler:** Fit on train-400 feature matrix. Transform holdout-100 with the train-fitted scaler.

Phase 0 measures the magnitude of leakage from these sources before any evaluation begins.

---

## Phase 0: Reproduce & Lock

### Goal
Verify you are measuring the same system CORAL measured. Every subsequent phase depends on this. If Phase 0 fails, stop and investigate — do not proceed with a drifted pipeline.

### Steps

1. **Verify all artifacts exist.** Check every path in the artifact table above. For branch-resident files, run `git show coral-topo-auroc-run:coral_run_2026_04_08/winning_features.py | wc -l` and confirm 1407 lines. For data files, verify shapes:
   - `trajectories.npz`: 500 keys, each `(n_tokens, 1536)`
   - `trajectory_meta.json`: `correct` array of length 500, sum = 57
   - `math500_hidden_states_aligned.npz`: `layer_hidden_states` shape `(500, 29, 1536)`
   
   If any artifact is missing, halt with an error message. Do not improvise substitutes.

2. **Extract the CORAL winning feature code.** Check out `winning_features.py` from the CORAL branch into `pathway1/phase0/`:
   ```bash
   git show coral-topo-auroc-run:coral_run_2026_04_08/winning_features.py > pathway1/phase0/winning_features.py
   ```
   This is the copy you will use for all subsequent phases. Do not modify it.

3. **Reproduce CORAL's 0.9779 AUROC.**
   - Load trajectories and layer states.
   - Call `extract_features(trajectories, layer_states)` on ALL 500 samples (matching CORAL's pipeline — PCA fit on all, ranks computed on all).
   - Apply `StandardScaler().fit_transform(features)`.
   - Run `StratifiedKFold(n_splits=50, shuffle=True, random_state=42)`.
   - Fit `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
   - Use `cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]`.
   - Compute `roc_auc_score(y, proba)`.
   - **This number must match 0.9779 to within ±0.001.** If it doesn't, something in the pipeline has drifted. Print the exact AUROC and halt if outside tolerance.

4. **Measure statefulness leakage magnitude.**
   - Create a temporary stratified split: `StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=12345)`. This is NOT the real holdout — it's a throwaway for quantifying leakage.
   - Run `extract_features` two ways:
     - (a) **All-500:** Fit PCA on all 500 trajectories, compute all features, split afterward. (This is what CORAL did.)
     - (b) **Train-only:** Fit PCA on train-400 trajectories only, transform all 500, recompute rank-bin features using train-400 ranks for holdout percentile-ranking.
   - Compare the feature matrices for the 400 train samples between (a) and (b). Report the mean absolute difference per feature column. If any column differs by more than 0.1 in mean absolute terms, flag it — those features are heavily PCA-path-dependent.
   - Delete the throwaway split. It is not used again.

5. **Lock artifacts by hash.**
   Save to `pathway1/phase0/artifact_locks.json`:
   ```json
   {
     "winning_features_sha256": "<SHA-256 of winning_features.py>",
     "trajectories_sha256": "<SHA-256 of trajectories.npz>",
     "trajectory_meta_sha256": "<SHA-256 of trajectory_meta.json>",
     "layer_states_sha256": "<SHA-256 of math500_hidden_states_aligned.npz>",
     "reproduced_auroc": 0.XXXX,
     "lr_params": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42, "solver": "lbfgs"},
     "cv_params": {"n_splits": 50, "shuffle": true, "random_state": 42},
     "pca_params": {"n_components": 45, "svd_solver": "full"},
     "scaler": "StandardScaler",
     "leakage_magnitude": {"mean_abs_diff_per_feature": [...], "max_feature_drift": X.XX}
   }
   ```

6. **Freeze policy.** From this point forward:
   - `winning_features.py` must not be modified. Any feature engineering happens in Pathway 2+, not here.
   - The holdout mask (created in Phase 1) must not be regenerated after seeing results.
   - LR hyperparameters are locked to the values above. If you need to tune C, do it via nested CV on train-400 in a clearly separated step and document the chosen value before touching holdout.

---

## Phase 1: Three-Number Comparison

### Goal
Quantify honest signal by comparing three AUROC numbers measured at increasing remove from the CORAL optimization loop:
- **CV-500**: CORAL's own protocol on all data (reproduced in Phase 0).
- **CV-train-400**: Same protocol, but on the 400-sample train subset only.
- **Holdout-100**: Single LR fit on train-400, scored on the 100 samples CORAL never touched.

The meaningful gap is **CV-train-400 minus holdout-100** — this isolates evaluation overfitting from the effect of a smaller training set.

### Steps

1. **Create the holdout mask.**
   - Use `StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)` to select indices. Stratification ensures proportional class balance in both splits.
   - Save to `pathway1/phase1/holdout_mask_seed9999.npy` (boolean array of length 500, `True` = holdout).
   - Print: first 10 holdout indices, class balance in train (n_correct / 400) and holdout (n_correct / 100). With 57/500 correct and stratified split, expect ~11 correct in holdout, ~46 in train.

2. **Number 1: CV-500.** Already computed in Phase 0. Record the value.

3. **Number 2: CV-train-400.**
   - Extract features using the **train-only PCA** path (fit PCA on train-400 trajectories, transform train-400, compute rank-bins on train-400 only).
   - Apply `StandardScaler` fit on train-400 features.
   - Run `StratifiedKFold(n_splits=50, shuffle=True, random_state=42)` on the 400 train samples.
   - `cross_val_predict` with `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
   - Record AUROC.

4. **Number 3: Holdout-100.**
   - Using the train-400-fitted PCA, transform holdout-100 trajectories.
   - Compute holdout feature products. For rank-binned features: percentile-rank each holdout sample's raw product within the sorted train-400 raw values using `np.searchsorted`, then digitize.
   - Apply the train-400-fitted StandardScaler to holdout features.
   - Fit LR on scaled train-400 features. Score on scaled holdout-100 features.
   - Compute: `roc_auc_score` + **bootstrap 95% CI** (1000 resamples of the holdout set, stratified by class).
   - Also compute at the default 0.5 threshold: accuracy, precision, recall, F1.
   - Also compute at the Youden's J optimal threshold (found on train-400 CV predictions, NOT holdout): accuracy, precision, recall, F1.
   - Assert `np.isfinite(features).all()` before scoring. If any inf/nan, identify the source feature and report.

5. **Interpret the three numbers.**
   - `CV-500 − CV-train-400` = effect of smaller training set + loss of holdout samples from PCA fit. Expected to be small (0.005–0.02).
   - `CV-train-400 − holdout-100` = evaluation overfitting. This is the number that matters.
   - Diagnostic bands for the holdout-100 AUROC point estimate:
     - ≥ 0.90 → the CORAL result is largely real. Proceed with high confidence.
     - 0.80–0.90 → the signal is real but late-stage recursive products were overfitting. The methodology is validated; the depth-8 features are not.
     - 0.75–0.80 → marginal. Core PH + layer features carry signal but the product engineering amplified noise more than signal.
     - ≤ 0.75 → almost entirely overfit. Only the methodology generalizes, not the features.

6. **Save artifacts.** To `pathway1/phase1/`:
   - `holdout_mask_seed9999.npy`
   - `features_train400.npy`, `features_holdout100.npy`
   - `cv500_auroc.txt`, `cv_train400_auroc.txt`
   - `holdout_predictions.npy` (predicted probabilities for all 100 holdout samples)
   - `holdout_metrics.json` (all metrics + bootstrap CI)
   - `three_number_comparison.txt` (human-readable summary)

---

## Phase 2: Feature Ablation by Tier

### Goal
Determine what fraction of the holdout AUROC comes from each era of feature engineering. Tier selection uses nested CV on train-400 only. The holdout is touched once at the end, for reporting only.

### Steps

1. **Define feature tiers.** Trace through `winning_features.py` and categorize every output feature column by depth. Use this rule: **a feature's tier = max depth of any operand + 1.**

   - **Tier A — Baseline (pre-CORAL):** The 7 original PH descriptors (`H0_persistence_entropy`, `H0_total_persistence`, `H0_max_lifetime`, `H1_persistence_entropy`, `H1_total_persistence`, `bridge_silhouette`) plus the 6 geometric features (`first_token_norm`, `pairwise_dist_mean`, `norm_mean`, `last_token_centroid_dist`, `last5_centroid_dist`, `last_token_layer_alignment`). These are the raw measurements that existed before any CORAL engineering. Note: some baseline features were subsequently dropped by CORAL's drop-add cycles — Tier A uses only the baseline features that survive in the final output.
   
   - **Tier B — Depth-1 (single transforms/products + layer features):** Features that are a single transform of baseline features (exp, log, sq, rank-bin) OR pairwise products of two baseline features OR novel layer-state features added during CORAL's first phase (cosine similarities like `cos_l9_l28`, SVD spectrum features like `layer_pca_var_ratio`, curvature features like `layer_angle_min`). These are features where all operands are Tier A or raw data.
   
   - **Tier C — Depth-2 products (ADD #287–#288):** Products that combine Tier B features with each other or with Tier A features. Two levels of multiplication. Includes the `nr_x_h0ent35` family, `interact_tk_x_s21_rb`, etc.
   
   - **Tier D — Depth-3+ recursive products (ADD #289–#299):** Everything with a `p3_`, `p4_`, `p5_`, `p6_`, `p7_`, `p8_` prefix, plus the monster multi-way products. These are the features that reference other winning features as operands.

   Save the tier assignment as `pathway1/phase2/tier_assignments.json` mapping each output feature name to its tier (A/B/C/D). Review the assignment manually — ambiguous cases should be assigned to the higher (deeper) tier.

2. **Tier selection via nested CV on train-400.**
   Run four cumulative tiers (A, A+B, A+B+C, A+B+C+D) through the same CV protocol on train-400:
   - Train-only PCA, train-only rank-binning, train-only StandardScaler (all within the train-400).
   - `StratifiedKFold(n_splits=50, shuffle=True, random_state=42)`.
   - `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
   - Record AUROC for each cumulative tier.
   
   Also run each tier in isolation (A alone, B alone, C alone, D alone) to check if any single tier dominates.
   
   **Lock the "best-surviving tier"** = smallest cumulative tier that is within 0.01 AUROC of the full set (A+B+C+D) on train-400 CV. This tier is used in Phase 4. Lock this decision BEFORE looking at holdout results.

3. **Single holdout touch for reporting.**
   For each cumulative tier, train on train-400, score on holdout-100. Report AUROC per tier. This is for the ablation table only — the tier used in Phase 4 was already locked in Step 2.

4. **Produce a table and a line plot.**
   - Table: columns = Tier, N_features, CV-train-400 AUROC, Holdout AUROC.
   - Plot: x-axis = cumulative tier, y-axis = AUROC (two lines: CV-train-400 and holdout). If the curves diverge between C and D (CV rises but holdout flattens or drops), that's direct evidence the deep products are overfitting.

5. **Save artifacts.** To `pathway1/phase2/`:
   - `tier_assignments.json`
   - `ablation_cv_results.json` (cumulative + isolated, CV-train-400 AUROCs)
   - `ablation_holdout_results.json` (cumulative holdout AUROCs)
   - `ablation_table.txt`, `ablation_plot.png`
   - `best_surviving_tier.txt` (locked tier name + rationale)

---

## Phase 3: Fold Seed Sensitivity

### Goal
Determine if the CORAL result is brittle to the specific CV fold assignment (random_state=42) that was held fixed across all 676 optimization attempts.

### Steps

1. **Vary CV fold seed on CV-500.** Re-run the full winning feature set (all tiers, all-500 PCA as in CORAL's own protocol) with `StratifiedKFold` `random_state` values of 42, 43, 44, 45, 46. Everything else unchanged: same features, same LR hyperparameters. Record AUROC for each seed.

2. **Vary CV fold seed on CV-train-400.** Same 5 seeds, but now using the proper train-only pipeline (train-400 PCA, train-400 rank-bins, train-400 scaler). Record AUROC for each seed.

3. **Vary holdout split seed.** Run holdout evaluation with split seeds 9999, 9998, 9997, 9996, 9995 (each produces a different 400/100 stratified split). For each:
   - Fit PCA on that seed's train-400, extract features, fit scaler, fit LR.
   - Score on that seed's holdout-100.
   - Record AUROC.
   
   This tests whether the holdout result is an artifact of which 100 samples happened to land in the holdout.

4. **Compute summary statistics.** For each of the three groups (CV-500, CV-train-400, holdout):
   - Mean, std, min, max across 5 seeds.

5. **Interpret:**
   - CV-500 std < 0.005 → the CORAL AUROC is stable across fold assignments. The optimization overfitting (if any) is feature-level, not fold-luck.
   - CV-500 std 0.005–0.02 → moderate fold sensitivity. The ranking of features might shuffle, meaning CORAL's "winning" set is one of several near-equivalent sets.
   - CV-500 std > 0.02 → the CORAL AUROC is fold-fragile. The specific number is noise; only the broad regime matters.
   - Holdout std across split seeds: if > 0.03, the n=100 holdout is too noisy for fine-grained go/no-go decisions. Report this and factor it into the Phase 4 prerequisite check.

6. **Save artifacts.** To `pathway1/phase3/`:
   - `seed_sensitivity_cv500.json` (5 seeds, 5 AUROCs)
   - `seed_sensitivity_cv_train400.json`
   - `seed_sensitivity_holdout.json` (5 split seeds, 5 AUROCs)
   - `seed_sensitivity_summary.txt`

---

## Phase 4: Routing Prototype

### Goal
Build a minimal inference-time router that uses the topo-confidence score to decide whether to trust the small model's answer or escalate to a larger model.

### Prerequisite
Phase 1 holdout AUROC **bootstrap 95% CI lower bound ≥ 0.78**. If the CI straddles 0.78 (lower < 0.78 < upper), run a tiebreaker holdout with seed 9998. If the tiebreaker CI also straddles, **stop — proceed to Pathway 2 with Tier A+B features only**. If lower bound < 0.75, stop entirely (NO-GO).

### Steps

1. **Calibrate first.**
   - Fit LR on train-400 with the best-surviving feature tier (locked in Phase 2 Step 2).
   - Produce a calibration plot (reliability diagram) on holdout-100: bin predicted probabilities into 10 equally-spaced bins, plot mean predicted probability vs. observed fraction correct.
   - Compute Expected Calibration Error (ECE) and Brier score.
   - If ECE > 0.05: apply Platt scaling or isotonic regression via nested 5-fold CV on train-400. Refit the calibrated model. Report both calibrated and uncalibrated metrics.
   - The routing threshold sweep below uses the (possibly calibrated) model's `predict_proba`.

2. **Define the routing policy.**
   - For each sample, `predict_proba` gives P(correct). This is the confidence score.
   - Samples with P(correct) ≥ τ are "trusted" (use small model's answer). Samples with P(correct) < τ are "escalated" (route to a larger model or flag as uncertain).
   - Sweep τ from 0.1 to 0.9 in steps of 0.05.

3. **Compute routing metrics on the holdout set.** For each τ:
   - **Coverage:** fraction of samples trusted (not escalated). Higher = more efficient (using the small model more).
   - **Trusted accuracy:** accuracy of the small model on the trusted subset. Higher = router correctly identifies easy problems.
   - **Escalation PPV:** fraction of escalated samples that the small model actually got wrong. Higher = not wasting the big model's compute.
   - **Oracle gain:** assuming the big model gets 100% of escalated samples correct, what's the overall accuracy vs. no routing? This is the ceiling.
   - **Bootstrap 95% CI** on coverage and trusted accuracy (1000 resamples). n=100 means noisy curves — CIs are mandatory.

4. **Produce a coverage-vs-accuracy tradeoff curve.**
   X-axis = coverage (fraction trusted), y-axis = trusted accuracy. A good router stays high (>90% accuracy) at high coverage (>70%), then drops sharply.

5. **Find the operating point.** The practical question: "at what threshold do I get coverage ≥ 0.90 with trusted accuracy ≥ 0.95?" If such a threshold exists in the holdout data, report it and its bootstrap CI. If not, report the coverage at which trusted accuracy first exceeds 95%.

6. **Sanity-check with baselines.** Compare the topo-confidence router against:

   **(a) Random routing.** For each τ in the sweep, escalate `(1 - coverage_at_τ)` fraction of samples uniformly at random. This gives the expected routing metrics if the confidence score were uncorrelated with correctness. Average over 100 random draws.

   **(b) Output-entropy routing.** Use `baselines.output_entropy(logits_list)` from `topo_confidence/baselines.py`. This function computes mean per-token Shannon entropy over all generated tokens using softmax of output logits:
   - **Logit source:** Check if `data/experiment1_v2/` contains stored logits (e.g., in the NPZ or a separate file). If not, re-extract: instantiate `HiddenStateExtractor("Qwen/Qwen2.5-1.5B-Instruct")`, call `extract_with_output(prompts, collect_logits=True)`, and cache the logits to `pathway1/phase4/output_logits.pkl`.
   - **Routing direction:** Low entropy = high confidence = trust. Threshold the negative entropy to match the same coverage levels as the topo-confidence router.
   - Compute the same routing metrics (coverage, trusted accuracy, escalation PPV, oracle gain) at the same coverage levels for a fair comparison.

   **(c) Trivial baseline (optional).** Route by problem text length (number of tokens in the prompt). Longer problems are "harder" — escalate those above a length threshold calibrated to match coverage levels.

7. **Interpret the baseline comparison.**
   - If topo-confidence routing **beats** output-entropy routing at most operating points: topology captures something the model's own uncertainty doesn't. Pathway 4 (distillation) has a positive prior.
   - If topo-confidence routing **matches** output-entropy: topology is an expensive proxy for information already available in the forward pass. Pathway 2 (steering) may still add value, but Pathway 4 probably won't.
   - If topo-confidence routing **loses** to output-entropy: the topology is not capturing anything useful for routing. Rethink before proceeding.

8. **Save artifacts.** To `pathway1/phase4/`:
   - `calibration_plot.png`, `calibration_metrics.json` (ECE, Brier)
   - `routing_metrics.json` (all τ values, all metrics, all baselines)
   - `routing_tradeoff_curve.png`
   - `operating_point.json`
   - `output_logits.pkl` (if extracted)
   - `baseline_comparison.txt`

---

## Deliverables

By the end of Pathway 1, you should have produced:

1. **Phase 0 reproducibility proof:** reproduced AUROC (must match 0.9779 ±0.001), artifact hashes, leakage magnitude per feature.
2. **Three-number comparison:** CV-500, CV-train-400, holdout-100 (point estimate + bootstrap 95% CI).
3. **Ablation table:** holdout AUROC per feature tier (cumulative and isolated), with the best-surviving tier locked before holdout was consulted.
4. **Seed sensitivity tables:** 5 seeds × {CV-500, CV-train-400} + 5 holdout split seeds.
5. **Routing tradeoff curve** with calibration check, operating point annotated, bootstrap CIs.
6. **Baseline comparison:** topo-confidence vs. random vs. output-entropy routing at matched coverage levels.
7. **Go/no-go decision** for Pathway 2, using the **lower bound of the holdout 95% bootstrap CI**:
   - **GO:** CI lower bound ≥ 0.78 AND topo-confidence routing beats output-entropy at ≥ 1 operating point.
   - **CONDITIONAL GO:** CI lower bound ≥ 0.78 AND topo-confidence ≈ output-entropy (proceed to Pathway 2 to test causality, but expect Pathway 4 may not add value beyond standard methods).
   - **NARROW GO:** CI lower bound in [0.75, 0.78). Proceed to Pathway 2 with Tier A+B features only. Deep products are not validated.
   - **NO-GO:** CI lower bound < 0.75. Revert to baseline features, rethink PH feature engineering, possibly re-run CORAL with a proper train/holdout split baked into the grader.
   - **Tiebreaker:** If the CI straddles 0.78 (lower < 0.78 < point estimate > 0.78), run holdout with split seed 9998. If the tiebreaker CI lower bound ≥ 0.78, upgrade to GO/CONDITIONAL GO. If tiebreaker also straddles, default to NARROW GO.

---

## Technical notes

- **Determinism matters.** Use `svd_solver='full'` in every PCA step. The CORAL run's Era 1→2 breakthrough was discovering that `'auto'` picks `'randomized'` for this matrix size, introducing ±0.003 AUROC noise. Don't reintroduce it.
- **Don't retrain on holdout, ever.** The holdout-100 is sacred. If you need to tune hyperparameters (e.g., LR regularization strength C, calibration method), do it via nested CV on train-400 only.
- **Match CORAL's LR hyperparameters exactly:** `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs")`. The original prompt incorrectly specified `max_iter=10000` and omitted `class_weight="balanced"` — the repo uses `max_iter=1000` and `class_weight="balanced"` throughout.
- **StandardScaler is mandatory.** LR regularization is scale-dependent. CORAL's pipeline always applies `StandardScaler` before fitting. Fit on train, transform holdout.
- **Feature extraction is NOT stateless.** This is the single most important difference from v1 of this prompt. The CORAL `extract_features()` function fits PCA on pooled input and computes rank-bins across all input samples. The statefulness warning section above specifies the exact procedure for proper train/holdout separation.
- **Assert finite features.** After extraction, verify `np.isfinite(features).all()`. Depth-3+ products with rank-binning should be bounded, but if any inf/nan appears, identify the source column and report before continuing.
- **Save intermediate artifacts after each phase.** You'll need Phase 1's per-sample confidence scores for Pathway 2 (activation steering identifies the "predicted-failure" subset).
- **The code is frozen.** Do not modify `winning_features.py` during Pathway 1. Do not re-engineer features after seeing holdout results. Feature improvement happens in Pathway 2+.
- **n=100 is noisy.** Bootstrap CIs on the holdout will be wide (expect ±0.04–0.06). This is fine — the solution is honesty about width, not avoiding the test. All go/no-go decisions use CI lower bounds, not point estimates.
- **Two data sources.** The CORAL winning features use BOTH `trajectories.npz` (per-token hidden states) AND `math500_hidden_states_aligned.npz` (per-layer mean states). Do not forget the second file — many features (cosines, SVD spectrum, curvature) require `layer_states`.
- **Class imbalance.** 57/500 correct (11.4%). `class_weight="balanced"` in LR handles this. Stratified splits and stratified CV preserve the ratio in all subsets.
