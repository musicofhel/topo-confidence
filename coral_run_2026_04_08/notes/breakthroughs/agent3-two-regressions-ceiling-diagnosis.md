---
creator: agent-3
created: 2026-04-11
---
# Two consecutive regressions on the 26-feat base: ceiling diagnosis

## Events
- Eval #266: normratio_l14l28_x_toklayer (27th), local +0.00141 (t=8.37), grader -0.00115. REGRESSED.
- Eval #267: tok_count (27th), local +0.00084 (t=11.98, pos=0.94), grader -0.00146. REGRESSED.

## Root cause
Grader-CV gap at 0.90+ (~2-5mi over-delivery) now exceeds marginal improvements (+0.0005-0.0015).
Candidate deltas are indistinguishable from noise on the grader side. Add-only recipe exhausted.

## New strategies to try
1. SWAPs not ADDs: find features with near-zero LR coef and replace them
2. Different source: per-layer persistence homology (H0/H1 on layer points directly)
3. Leave-one-out local CV to identify weakest current feature

## Current state
0.9061 leader restored at commit 0a94ff9c (26 features). Next: LOO-CV to find SWAP targets.
