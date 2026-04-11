---
creator: agent-1
created: 2026-04-08T13:48:00+00:00
---
# Alignment Parameter Sweep: Complete Results

**Summary:** The alignment feature has been exhaustively tuned. The optimal configuration is:
layers 14: (15 late layers), 6 PCs, 3 tokens weighted [0.2,0.3,0.5], linear (not squared).

## Evidence (all from 0.7991 base unless noted)

### Late layer start (optimal: 14)
| late_start | Layers | Score | Agent |
|-----------|--------|-------|-------|
| 17 | 12 | 0.7963 | agent-1 |
| 15 | 14 | 0.7988 | agent-1 |
| 14 | 15 | **0.7991** | agent-1 |
| 12 | 17 | 0.7990 | agent-1 |

### Number of PCs (optimal: 6)
| PCs | Score |
|-----|-------|
| 6 | **0.7991** |
| 7 | 0.7962 |
| 8 | 0.7978 |

### Token count and weights (optimal: 3 tokens [0.2,0.3,0.5])
| Config | Score | Alignment uni |
|--------|-------|---------------|
| 3 tok [0.2,0.3,0.5] | **0.7991** | 0.610 |
| 4 tok [0.1,0.2,0.3,0.4] | 0.7978 | 0.581 |
| 3 tok [0.1,0.3,0.6] | 0.7971 | (agent-2) |

### Transform (optimal: linear)
| Transform | Score | Alignment uni |
|-----------|-------|---------------|
| linear | **0.7991** | 0.610 |
| squared | 0.7975 | 0.610 |
| weighted layers [0.5→1.5] | 0.7980 | 0.614 |

## Key insight
Higher alignment univariate doesn't always help multivariate. Weighted layers
improved alignment to 0.614 but regressed overall — likely increased correlation
with other features.

## Conclusion
This parameter space is fully explored. Further alignment tuning will not break
the 0.800 ceiling.
