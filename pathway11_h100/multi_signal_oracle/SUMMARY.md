# Exp 2b — Multi-signal oracle-K classifier

**Bottom line: the non-monotonic multi-feature gating thesis does not hold.** A 3-feature classifier using `{prefill DoM, prefill local PR, sequence length}` beats the best single-signal policy by **+0.4pp** at matched compute — far below the user's ≥2pp bar. And the "extra" feature (prefill local PR) contributes **literally zero**: drop-one-out gives Δmacro-F1 = +0.000, Δoverall = +0.000.

## Setup

- 500 MATH-500 problems, Qwen-2.5-7B cached activations from Exp 3 (`prefill_inversion/cache/m7b_prefill.npz`).
- Per-problem features:
  - `prefill_dom` — OOF DoM score at L19 prefill-end (from Exp 2 Phase 2).
  - `prefill_lpr` — per-problem local participation ratio = PR of 20 nearest neighbors in prefill activation space.
  - `seq_len` — length of greedy K=1 @ T=0.7 generation.
  - (extended) `finaltok_dom`, `finaltok_lpr`, `mean_logprob`.
- 3-class oracle target from Exp 2 Phase 4 buckets:
  - `K1_AD` (A + D): 243 problems (48.6%) — should run K=1 only.
  - `K8_B` (B): 68 problems (13.6%) — K=8 majority vote recovers correctness.
  - `refuse_C` (C): 189 problems (37.8%) — never right, don't waste compute.

Classifiers: logistic regression (balanced, StandardScaler) and random forest (400 trees, balanced). 5-fold stratified CV.

## 1. Classifier quality (5-fold CV)

| Classifier | macro-F1 | overall acc | K1_AD F1 | K8_B F1 | refuse_C F1 |
|---|---:|---:|---:|---:|---:|
| logreg | 0.548 | 0.620 | 0.703 | 0.267 | 0.674 |
| random forest | 0.494 | 0.636 | 0.698 | 0.119 | 0.665 |
| baseline (all K=1) | 0.218 | 0.486 | 0.654 | 0.000 | 0.000 |

**Logreg confusion matrix (3-feature):**
```
              pred
true        K1   K8  refuse
K1_AD      163   44    36
K8_B        25   22    21
refuse_C    33   31   125
```
- K=8 class (B) is hardest because its prefill-DoM signature sits between A (strongly positive) and C (strongly negative). Only 32% B-recall.
- Random forest collapses K=8 further (7% recall) — too-high class threshold.

## 2. Policy simulation (at coverage=1.0, K=1 fallback on refused)

| Policy | coverage | compute/prob | overall acc |
|---|---:|---:|---:|
| Oracle (upper bound) | 1.00 | 1.006 | 0.589 |
| uniform K=1 | 1.00 | 1.000 | 0.486 |
| uniform K=2 | 1.00 | 2.000 | 0.413 |
| uniform K=4 | 1.00 | 4.000 | 0.495 |
| uniform K=8 | 1.00 | 8.000 | 0.550 |
| threshold_prefill @ 20pct | 1.00 | 2.40 | 0.486 |
| threshold_neg_seq_len @ 25pct | 1.00 | 2.74 | 0.516 |
| quartile_neg_seq_len_top_heavy | 1.00 | 4.50 | 0.542 |
| **multi_signal_logreg (K=1 fb)** | **1.00** | **2.36** | **0.520** |
| multi_signal_rf (K=1 fb) | 1.00 | 1.22 | 0.492 |

**Multi-signal logreg vs best single-signal at matched compute:** +0.4pp (0.520 vs 0.516 at K≈2.5). Below the ≥2pp threshold.

Multi-signal RF (K=1.22, 0.492) does beat uniform K=1 by +0.6pp at +22% compute — a very small efficient step but still well under 2pp.

No fallback (refuse = no answer) operating points:
- logreg: coverage=0.64, answered_acc=0.704, overall=0.448
- rf:     coverage=0.57, answered_acc=0.657, overall=0.376

## 3. Ablation — which features matter?

Drop-one-out from the full `{prefill_dom, prefill_lpr, seq_len}` set (logreg):

| Dropped feature | Δ macro-F1 | Δ overall acc | Δ D-recall to K=1 |
|---|---:|---:|---:|
| prefill_dom | **−0.044** | −0.014 | **+0.194** |
| prefill_lpr | **+0.000** | **+0.000** | +0.000 |
| seq_len | −0.033 | −0.010 | −0.111 |

Single-feature classifiers (logreg):
- `prefill_dom` alone: F1=0.525, overall=0.512
- `seq_len` alone: F1=0.514, overall=0.508
- `prefill_lpr` alone: F1=0.363, overall=0.494 (barely above baseline 0.486)

Extended 6-feature set (adds finaltok_dom, finaltok_lpr, mean_logprob):
- F1=0.577 (+0.027), overall=0.528 (+0.008), D-recall=0.444
- Helpful, but `finaltok_*` and `mean_logprob` require actually running K=1 first — they're post-generation features, not gating features.

**Conclusion on features:**
- `prefill_lpr` (local PR) is dead weight. Per-problem local PR is nearly flat across all buckets (A=12.66, B=12.80, C=12.97, D=12.84). The Exp 3 group-level "D-bucket has low prefill PR = 14.49" is a *collective* signature of the 36 D-points clustering together; a k=20 nearest-neighbor local PR doesn't recover it because most D-problems' neighbors are mostly non-D problems.
- `prefill_dom` and `seq_len` carry all the useful variance. They produce classifiers nearly indistinguishable from the full set.
- `finaltok_dom` would help modestly (+0.8pp overall) but is post-hoc.

## 4. D-bucket protection — does the classifier help?

The explicit motivation: 36 D-bucket problems where K=1 is right but K=8 majority vote corrupts the answer. Can we route them to K=1 before running K=8?

| Classifier | D → K=1 | D → K=8 | D refused |
|---|---:|---:|---:|
| logreg | **15/36 (41.7%)** | 6 | 15 |
| random forest | **16/36 (44.4%)** | 0 | 20 |
| random assignment to class 0 (prior) | 48.6% | — | — |

**D-bucket recall is below class prior.** The classifier is worse than randomly assigning every problem to class 0 (48.6% baseline). D-bucket features are genuinely ambiguous:
- `prefill_dom` mean for D = −1.16 (between B=+0.79 and C=−3.34, classifier tends to refuse).
- `seq_len` for D = 527 (intermediate, between A=422 and B=613).
- `prefill_lpr` for D = 12.84 (indistinguishable from other buckets).

Dropping `prefill_dom` bumps D-recall to 0.611 — because `prefill_dom` is actively pulling D's into the "refuse" class. But this comes at the cost of overall policy performance (−0.014 overall). The D-pathology and the general discrimination task are fighting each other in the feature weights.

## 5. What didn't work and why

The thesis was: "combining prefill DoM (discriminates correctness) with prefill PR (discriminates D-pathology) gives non-monotonic gating that single-signal policies miss". 

Two reasons it doesn't work:
1. **The D-bucket geometric signal doesn't survive per-problem localization.** Group PR over 36 D-problems = 14.49 (anomalously low). Per-problem local PR computed from the same 36 problems' k-NN neighborhoods = 12.84 (same as everyone else). Each D-problem's neighbors are mostly *non-D*, so per-problem locality doesn't probe the collective cluster.
2. **D is a small, ambiguous minority.** 36 out of 500 = 7.2%. Its feature signature overlaps with B (K=8-recoverable) and partly with C (never-right). No feature combination in the available set cleanly isolates D without collateral damage to the main K=1-vs-K=8-vs-refuse discrimination.

## 6. Connection to Exp 2/3 findings

- Exp 2: prefill DoM AUROC = 0.77 for K=1 correctness, but gating on it doesn't Pareto-beat uniform K=8. Top-heavy `neg_seq_len` (K=4.5, 0.542) was best single-signal policy.
- Exp 3: 7B prefill PR inversion ("PR_correct > PR_incorrect") is real and bootstrap-robust. D-bucket has distinctive low *group* PR.
- Exp 2b (this): the group-level D-bucket PR signature does **not** translate to a per-problem feature. `prefill_lpr` contributes zero. Multi-feature gating improves single-feature gating by ≤0.4pp — within noise. Not worth the complexity.

## 7. What *would* help (open)

- Density-based per-problem features: e.g., LOF (local outlier factor) or kernel density on prefill activations. These might detect "tight cluster membership" better than k-NN local PR.
- Features from intermediate layers (not just L19). D-bucket might emerge at earlier/later layers.
- Learning the D-bucket directly: one-class SVM or isolation forest on the 36 D-problems' prefill activations, rather than forcing it into a multi-class target.
- Final-token features (`finaltok_dom` helped +0.8pp) — deploy as a post-hoc "should we trust K=8's majority?" check that runs K=1 first, then rescales K=8 aggregation based on final-token confidence.

## 8. Outputs

- `features.npz` — all 10 feature/label arrays (500 rows).
- `phase1_features.json` — per-bucket feature means.
- `phase2_classifier.json` — logreg + RF CV metrics, confusion matrices.
- `phase3_policy.json` — policy simulation (both fallback variants) + Exp 2 comparison.
- `phase4_ablation.json` — 10 feature-subset runs × 2 classifiers.
- `pareto_comparison.png` — compute-vs-accuracy scatter with Exp 2 + Exp 2b + oracle.
- `results.json` — consolidated summary.

## 9. Decision implication

The "non-monotonic multi-feature gating" direction is closed for this dataset. Three practical next steps for Pathway 10 v2 steering:
1. **Don't spend complexity budget on local PR features** — they don't carry the signal.
2. **Prefill DoM + seq_len is the efficient frontier** for pre-generation gating. Any further gains require either (a) intermediate-layer features (breadth), (b) final-token rescoring (post-hoc), or (c) training-set-scale feature engineering.
3. **The D-bucket pathology needs a different intervention.** Since D is intermediate in every available feature, routing it at prefill is ~chance. Better to address K=8 aggregation itself: e.g., weight samples by final-token DoM, or use "first-K-samples-disagree" as a trust signal rather than raw majority vote.
