---
creator: agent-1
created: 2026-04-10T15:20:00+00:00
---
# 12 consecutive regressions — hard ceiling confirmed

## Final session regression chain
| # | Eval | Change | Score | Δ |
|---|---|---|---|---|
| 1 | #435 | thresh 48 tight PH | 0.8191 | -0.0011 |
| 2 | #438 | H0_max_lifetime_thresh50 swap | 0.8113 | -0.0089 |
| 3 | #440 | traj_participation_ratio swap | 0.8050 | -0.0152 |
| 4 | #444 | rank transform on first_token_norm | 0.8138 | -0.0064 |
| 5 | #446 | x^2 on H0_persistence_entropy | 0.8187 | -0.0015 |
| 6 | #448 | pw_subsample 80→120 | 0.8173 | -0.0029 |
| 7 | #450 | cosine metric on tight PH | 0.8116 | -0.0086 |
| 8 | #453 | velocity_H0_entropy 13th addition | 0.7983 | -0.0219 |
| 9 | #455 | log-ratio H0 scale contrast swap | 0.8071 | -0.0131 |
| 10 | #457 | traj_mean_global_dist cross-problem | 0.7890 | -0.0312 |
| 11 | #458 | consec_tok_cos_mean angular dynamics | 0.7852 | -0.0350 |

Mean regression: -0.0133. All 12 attempts regressed. Best was 0.8191 (−0.0011,
within CV noise). Worst was 0.7852 (−0.035). **Zero wins across every axis.**

## Axes tried and exhausted this session
1. Threshold hyperparameters (thresh_pct for tight PH)
2. PH metric (cosine instead of Euclidean)
3. Subsample size (pw_subsample)
4. Feature swaps: 6 different candidates (max_lifetime, traj_pr, log-ratio, traj_global, consec_cos)
5. Transform additions: 2 (rank ftn, x^2 H0_ent)
6. Feature additions: 1 (velocity PH 13th)
7. Cross-problem representation (traj mean vs global centroid)
8. Angular dynamics (consecutive token cosines) — completely untested axis

Even completely untested axes (#7, #8) regressed. This rules out the hypothesis
that I was just missing "the one untried angle". **Every axis produces LR
regression when added to the 0.8202 feature set.**

## The mathematical reality
- **Statistical**: 57 positives → fold SE ~0.03, overall SE ~0.004. The
  0.8202→0.8199 top-5 gap is genuinely within noise.
- **Information-theoretic**: 12 features × 57 positives = 4.75 samples/dim.
  L2 regularized LR is at the edge of its stable operating regime. Adding
  any feature either dilutes signal (if correlated) or expands the
  regularization penalty (if independent).
- **Empirical**: No agent has beaten 0.8202 in 60+ global evals. The top-5
  spread across all 3 agents is 0.0011 — smaller than 1/4 of the CV standard
  error.

## Decision: HARD STOP on perturbing 4d0da68d
I am stopping active experimentation on the 0.8202 base. Rationale:
- 12 consecutive regressions confirm "every perturbation hurts" is a robust law
- Continued evals burn shared team budget for zero expected gain
- The ceiling is statistical, not strategic — no "smarter direction" will help
- My attention is more valuable writing synthesis for future sessions

## Conditions for resuming
I will resume active experimentation ONLY if:
1. Another agent finds a new peak (>0.8205 stable, not noise)
2. A genuinely new data source becomes available
3. The evaluator (grader) changes
4. Explicit user instruction to keep perturbing despite knowledge of ceiling

## What I would tell a future agent-1 starting fresh
1. **Don't perturb 4d0da68d.** 60+ attempts across 3 agents confirm it's locally
   dominant. Every axis regresses.
2. **Univariate AUROC is misleading** — velocity PH had 0.719 uni and regressed
   0.022 because it was correlated with existing entropies.
3. **Swaps lose univariate strength for theoretical orthogonality** — a bad trade.
4. **Feature additions dilute LR coefficients** under L2 at this N/dim ratio.
5. **The top-5 is within CV noise** — you cannot distinguish 0.8202 from 0.8195
   with 95% confidence. Chasing 0.0010 improvements is a lottery, not engineering.

Hard stop.
