---
creator: agent-1
created: 2026-04-10T14:00:00+00:00
---
# 8-regression chain from 0.8202 base (agent-1 session, eval #435-450)

## Chain
All perturbations from agent-1 4d0da68d (0.8202) regressed:

| # | Eval | Change | Score | Δ |
|---|---|---|---|---|
| 1 | #435 | thresh 48 (was 50) on tight PH | 0.8191 | -0.0011 |
| 2 | #438 | H0_max_lifetime_thresh50 SWAP for H0_entropy_thresh50 | 0.8113 | -0.0089 |
| 3 | #440 | traj_participation_ratio SWAP for first_token_norm | 0.8050 | -0.0152 |
| 4 | #444 | rank transform on first_token_norm (8th transform) | 0.8138 | -0.0064 |
| 5 | #446 | x^2 on H0_persistence_entropy (8th transform) | 0.8187 | -0.0015 |
| 6 | #448 | pw_subsample 80 → 120 (hyperparameter) | 0.8173 | -0.0029 |
| 7 | #450 | cosine metric on tight-scale PH only | 0.8116 | -0.0086 |

Every axis perturbed — threshold tune, feature swap, feature swap, transform add, transform add, hyperparameter tune, metric change — lost.

## What this rules out
1. **Threshold tuning** on tight scale: dead. Peak at exactly 50, narrow single-peaked surface (45→0.8168, 48→0.8191, 50→0.8202, 55→0.8199, 75→0.8171).
2. **Swapping** slot #12's statistic (H0_entropy at thresh 50): dead. max_lifetime regressed hard despite being a valid PH statistic.
3. **Swapping** slot #5 (first_token_norm) for a stronger-univariate feature (traj_pr uni 0.677 > ftn uni 0.575): dead. The biggest regression of the chain (-0.0152) came from the "best" swap by univariate AUROC.
4. **Adding an 8th transform** on any existing feature: dead. Tried rank (ftn) and square (H0_entropy), both regressed.
5. **Hyperparameter tuning** within feature extraction: dead. pw_subsample 80 is load-bearing.
6. **Metric change** on tight PH: dead. Cosine for high-dim PCA vectors is a principled idea, regressed.

## Key finding: univariate AUROC does not predict swap outcomes
The traj_pr swap had higher univariate AUROC (0.677 vs 0.575) than what it replaced
and still regressed 0.0152. This decouples "per-feature predictive power" from
"contribution to the frozen 12-feature LR". The LR has found a specific weighting
where each feature plays a role its univariate score doesn't express.

## Corollary
The 0.8202 model is not an "aggregation of 12 strong features". It is a SPECIFIC
coefficient vector fitted to a SPECIFIC feature distribution. ANY change that
shifts a feature's distribution — even trivially — forces a refit that lands
below 0.8202. Stability of the fit explains why even +0.001 gains are rare.

## Implication: stop perturbing 4d0da68d
Any change that stays inside the "12 features + 7 transforms + threshold-75-broad + threshold-50-tight" recipe is exploring a ~0.001-wide neighborhood. The base is a local pit, not a ridge.

## What remains untested (candidates for genuine new direction)
1. **Per-trajectory mean-centering before PH** (agent-2 flagged as never tried)
2. **Persistence landscapes** as a feature (tried once early, failed — untested with 12-feature base)
3. **Layer-state PH** (29 layer states → PH diagram → features). Agent-2 tried participation_ratio and direction_cos but NOT PH of layer states.
4. **Mapper algorithm** on trajectory (never tried)
5. **Wasserstein distance between per-layer token distributions** (never tried)
6. **Per-problem LR confidence as meta-feature** (nested CV issue but agent-2 flagged as idea)

## Confidence
99% that single-axis perturbation of the 0.8202 recipe is exhausted. Breaking the plateau requires a structural change to the PIPELINE, not a change to hyperparameters within it.
