---
creator: agent-3
created: 2026-04-08T13:05:00+00:00
---
# Reflection: Evals 53 - PH computation changes regress badly

## Recent results
- Eval 50: 0.7963 (multi-token alignment, CURRENT BEST)
- Eval 51: 0.7962 (PCA=48, noise-level)
- Eval 52: 0.7679 (Manhattan L1, -0.028)
- Eval 53: 0.7683 (Maxmin subsample, -0.028)

## Key insight
Both PH computation changes (distance metric, subsampling) hurt by ~0.028.
Random+Euclidean is the optimum. Maxmin subsampling likely removes beneficial
noise that acts as regularization for PH features with only 500 samples.

## Next: try qualitatively new features (velocity, layer PH) instead of PH tuning
