# Experiment 2 — Prefill-Gated Compute Allocation — Summary

**Model:** Qwen-2.5-1.5B-Instruct. **Benchmark:** MATH-500 (n=500).
**Data:** cached from Pathway 11 pod session (Stage 2 greedy K=1 + all-layer per-token; Stage 3 K=8 @ T=0.7 + per-sample L19).

## Uniform baselines

| Policy | Compute/prob | Accuracy |
|---|---|---|
| Greedy K=1 (T=0) | 1.0 | 0.486 |
| K=2 majority (T=0.7) | 2.0 | 0.413  *(majority tie-break ≈ K=1-from-sample-pool)* |
| K=4 majority (T=0.7) | 4.0 | 0.495 |
| K=8 majority (T=0.7) | 8.0 | 0.550 |

K=8 buys +6.4pp over greedy K=1 at 8× compute. That narrow headroom bounds everything below.

## Prefill DoM signal quality

- OOF 5-fold AUROC on K=1 greedy correctness: **0.7731** (per-fold 0.66–0.83).
- OOF AUROC on K=8 majority correctness: **0.8228**.
- Final-token DoM AUROC (K=1): 0.719. Seq-len alone (Spearman with correctness = −0.52) → AUROC 0.80 for K=1.

Prefill is a strong correctness classifier. But the question the allocation experiment asks isn't "is it right?" — it's "does it need extra compute?"

## Oracle buckets

| Bucket | Description | n | % |
|---|---|---|---|
| A | K=1 ✓ and K=8 ✓ | 207 | 41.4% |
| B | K=1 ✗ and K=8 maj ✓ — **recoverable** | 68 | 13.6% |
| C | K=1 ✗ and K=8 maj ✗ — **hopeless** | 189 | 37.8% |
| D | K=1 ✓ and K=8 maj ✗ — **pathological** | 36 | 7.2% |

K=8 majority rescues only 68 / 257 = **26.5%** of K=1 failures. And K=8 actively hurts 36 problems (7.2%).

Oracle-K (smallest K s.t. majority-vote correct): 186 problems are "never" right; 243 need K=1; only 71 (14%) benefit from K > 1.
Oracle compute-per-problem = 1.01. Oracle accuracy = 0.589 (+3.9pp over uniform K=8 at 1/8th compute).

## Does prefill gating work?

**No — at fixed compute, prefill gating does not Pareto-dominate uniform K.**

| Policy | Compute | Accuracy |
|---|---|---|
| Uniform K=4 | 4.0 | **0.495** |
| Prefill threshold τ=40% | 3.80 | 0.492 |
| Prefill threshold τ=50% | 4.50 | 0.492 |
| Prefill quartile-balanced | 3.75 | 0.463 |
| Prefill quartile top-heavy | 4.50 | 0.492 |
| **Prefill quartile middle-heavy** (Q4,Q1→K=1; Q3,Q2→K=8) | 4.50 | **0.526** |
| **Neg seq-len top-heavy** | 4.50 | **0.542** |

Middle-heavy beats the monotonic prefill policies because **recoverable (B) problems cluster at mid-prefill-score**, not at the bottom.

But the real winner on the Pareto frontier is **sequence-length gating** — length correlates more directly with which K a problem needs (Spearman with oracle-K = 0.33 vs prefill's 0.18). Neg seq-len threshold dominates every other curve (see `pareto_plot.png`).

**Caveat:** seq-len is post-hoc. It's an upper bound on what any pre-generation signal could achieve. Prefill is pre-generation and cheap, but its signal is mis-aligned for this task.

## Clean win: refuse-and-spend (selective prediction)

If refusal is allowed (coverage < 100%), prefill gating is genuinely useful.

| Coverage = 0.50 policy | Acc on answered | Avg K on answered |
|---|---|---|
| Prefill (Q4→K=1, Q3→K=4, Q1+Q2→refuse) | **0.716** | 2.5 |
| Neg seq-len (same structure) | 0.707 | 2.5 |
| Random | 0.494 | 2.5 |

Prefill selective prediction gives **+22pp over random at the same coverage and same compute**, with accuracy-on-answered at 71.6% — well above uniform K=8's 55% overall accuracy.

## Why prefill gating fails at non-selective compute allocation

1. K=8 majority's recovery rate on K=1 failures is only 26.5% — so the "upside" of spending K=8 on uncertain problems is small to begin with.
2. 37.8% of problems are in the "hopeless" bucket C — prefill low-score mostly selects for C, not B. Compute sent to C is burned.
3. Prefill AUROC(B vs C | K=1 wrong) = 0.77 — within-K=1-wrong, prefill DOES separate recoverable from hopeless. But a monotonic gate can't exploit this because the bottom quartile is still mostly C.
4. 7.2% of problems are pathological (K=8 < K=1) — gating K=8 to low-confidence problems lands on some of these and loses accuracy.

**Implication for Pathway 10 v2:** compute gating using DoM-style confidence signals requires careful policy design beyond the monotonic τ-threshold. The middle-heavy finding suggests training a **two-sided gate** (confidence too high OR too low → K=1; middle band → K=8) is worth trying. Length-residualization of the DoM direction (Exp 1 Phase 2.6) is also motivated — seq-len is doing a lot of the real work here, and prefill DoM may become more useful once that signal is orthogonalized out.

## Files

- `phase1_majority_vote.npz` / `phase1_summary.json` — K=1,2,4,8 majority vote per problem
- `phase2_prefill_dom.npz` / `phase2_summary.json` — OOF prefill/final-token DoM scores + seq_len
- `phase3_policies.json` — sampled-K=1 variant (41% baseline)
- `phase3b_realistic_policies.json` — realistic variant (greedy K=1 @ T=0 for confident)
- `phase4_oracle.json` — oracle bucket analysis
- `results.json` — consolidated headline
- `pareto_plot.png` — compute-vs-accuracy Pareto with all policies
