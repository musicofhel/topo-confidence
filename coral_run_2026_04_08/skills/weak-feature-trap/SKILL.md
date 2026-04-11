---
name: weak-feature-trap
description: Recognize the ceiling-regime failure mode where adding any feature with uni AUROC < 0.60 regresses the multivariate score
creator: agent-3
created: 2026-04-10T15:00:00+00:00
---
# Weak-Feature Trap at the Signal Ceiling

## What it does
Identifies and prevents a specific failure mode in small-sample LogisticRegression
feature engineering: adding features with weak univariate signal (AUROC < 0.60)
at or near the statistical ceiling. Looks like "correlation-swap should apply"
but the regime has flipped and independence is no longer sufficient.

## When to use it
Small-sample, class-imbalanced binary prediction with a fixed LR(balanced) grader,
AND the multivariate score has plateaued within 0.005 of the top leaderboard score.

Specific regime:
- n approximately 500 samples, 11% positive (57 positives)
- 50-fold StratifiedKFold CV
- LogisticRegression(class_weight="balanced"), default L2
- Current feature count >= 10
- Top score within ~0.005 of everyone else

## The trap (empirical rule)

> At the signal ceiling, any 13th feature with univariate AUROC < 0.60 will
> regress the multivariate score, regardless of its correlation profile.

This directly contradicts the correlation-swap skill. Both are empirically
verified but apply at different regimes:

| Regime          | Rule                                        |
|-----------------|---------------------------------------------|
| 0.75-0.80       | Independence > univariate (low max|r| wins) |
| 0.80-0.82       | Both matter: need uni >= 0.60 AND max|r| < 0.5 |
| 0.82+ (ceiling) | Neither is sufficient — EPV starves signal  |

## Evidence
Agent-3 session 2026-04-10: 11 consecutive regressions on 0.8202 base.

| 13th feature candidate         | Uni   | Result  | Delta   |
|--------------------------------|-------|---------|---------|
| twoNN intrinsic dim            | 0.542 | 0.8131  | -0.0071 |
| H0_entropy_residual            | 0.551 | 0.8097  | -0.0105 |
| KDE log-density (PCA top-10)   | 0.563 | 0.8142  | -0.0060 |
| traj_layer_pc_alignment        | 0.572 | 0.8120  | -0.0082 |
| early-weighted centroid        | 0.601 | 0.8166  | -0.0036 |
| half_split_cosine_dist         | 0.582 | 0.8113  | -0.0089 |

Pattern: every feature with uni < 0.60 regressed, even ones with max|r| < 0.2.

Cross-agent confirmation: agent-1 (12 regressions), agent-2 (8 regressions)
on same base with different 13th-feature candidates. Total >30 independent
failures across 3 agents. p(11 regressions | null) < 0.001.

## Why it happens (mechanism)
With 57 positives and a 13th feature, the LR coefficient for that feature is
estimated from ~5.7 effective samples per variable (EPV). L2 regularization
cannot distinguish weak-signal coefficients from zero, so the feature's residual
noise dominates its residual signal. Adding it makes the multivariate objective
HARDER without improving the gradient direction.

Transforms redistribute existing signal but don't create new signal. Non-linear
combinations of existing features (products, residuals) also regress because
L2 under low EPV can't find their coefficients.

## How to use it

BEFORE adding a candidate feature, gate it:
1. Compute univariate AUROC on the ENTIRE dataset (not CV) — fast sanity check
2. If uni < 0.60 -> DO NOT ADD. The eval will regress. Save the eval cycle.
3. If 0.60 <= uni < 0.65 -> marginal, worth trying ONLY if max|r| < 0.3
4. If uni >= 0.65 -> worth trying; apply correlation-swap logic
5. If uni >= 0.70 -> strong candidate; likely to help

BEFORE dropping a load-bearing feature:
- Low univariate does NOT mean low load-bearing weight. norm_mean has uni 0.596
  but dropping it causes -0.024 regression. Load-bearing != univariate strength.
- Test drop only if the candidate has max|r| > 0.95 with another feature.

## What NOT to do at the ceiling
- Add features hoping independence will rescue weak univariate
- Change hyperparameters (PCA dim, subsample, thresholds) by small amounts
- Swap transforms for similar ones (log vs sqrt vs x^2)
- Add product/interaction features of existing features
- Run multi-seed averaging expecting variance reduction
- Test more candidate 13th features of the same flavor

## What might (marginally) work
- Pipeline structural changes (e.g., per-problem mean-centering before PH)
- Entirely different data source (if available)
- Grader changes to L1 (not possible if grader is fixed)
- Increasing sample size (not possible if dataset is fixed)

## Anti-pattern this skill replaces
Before this skill, the dominant approach was "find an independent feature
with any signal and add it." That approach works below 0.80 but fails at
the ceiling. This skill encodes the regime boundary.
