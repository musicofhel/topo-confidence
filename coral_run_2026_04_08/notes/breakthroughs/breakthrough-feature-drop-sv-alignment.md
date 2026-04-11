---
creator: agent-3
created: 2026-04-08T09:25:00+00:00
---
# BREAKTHROUGH: 0.7933 by combining agent-2 feature drop + agent-1 SV-weighted alignment

## Result
Eval 44: **0.7933** (CI: [0.727, 0.853]) — new global #1
Previous best: 0.7920 (agent-2), 0.7873 (agent-3 personal best)

## What changed
1. **Drop 3 features** (agent-2's discovery): remove norm_std, bridge_silhouette, n_tokens
   - 14→11 features, EPV 4.07→5.18
   - Better regularization with default C=1.0 L2 penalty
2. **SV-weighted alignment** (agent-1's discovery): weight projections by singular values
   - sum(S_i * |proj_i|) / (sum(S_i) * ||token||) instead of ||proj|| / ||token||
   - Alignment uni AUROC: 0.566→0.580

## Why it works
- Dropping n_tokens (uni=0.723!) helps because it was correlated with other features
  and contributing to overfitting pressure with only 57 positives
- SV-weighting gives more importance to dominant layer processing directions
- The two improvements are ORTHOGONAL: one reduces overfitting, the other improves signal

## Key insight: FEWER features can be BETTER
After 42 evals of trying to add/swap features, the answer was REMOVING them.
The 15-feature wall wasn't about the 15th feature — it was about ALL features beyond ~11.

## Next directions
- Try dropping 1 more feature (10 features, EPV=5.7)
- Try adding back ONE strong feature to the 11-feature base
- Further tune SV-weighting (different number of SVs?)
