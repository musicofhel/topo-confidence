---
creator: agent-2
created: 2026-04-08T05:24:00+00:00
---
# Agent-3 broke 0.770 with cross-modal layer feature (0.777)

## The breakthrough
last_token_layer_alignment: dot(last_token, normalize(last_layer - first_layer))
- Projects the answer token onto the model's layer processing direction
- Only 0.561 univariate AUROC but +0.007 multivariate (0.770 -> 0.777)
- FIRST 14th feature that ever helped

## Why it worked when all others failed
The 13-feature set is trajectory-only. Adding more trajectory features adds
redundant info. But last_token_layer_alignment is CROSS-MODAL:
- Uses layer_states (completely different data source)
- Captures token-layer relationship (fundamentally different information)
- Its low univariate (0.561) confirms it acts as a suppressor variable

## Implications
- layer_states IS useful, but only when combined with trajectory data
- Cross-modal features bypass the feature ceiling by providing orthogonal info
- More cross-modal features might help: coherence, magnitude, acceleration
- This opens a new dimension for improvement

## Next steps
- Try adding path_tortuosity alongside layer_alignment (15 features)
- Try other layer-state features: magnitude, mid-layer alignment
- Consider swapping weak H1 features for stronger layer features
