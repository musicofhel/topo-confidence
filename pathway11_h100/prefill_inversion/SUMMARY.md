# Experiment 3 — 7B Prefill PR Inversion — Summary

**Models:** Qwen-2.5-7B-Instruct (d=3584, 73.2% acc on MATH-500) vs Qwen-2.5-1.5B-Instruct (d=1536, 48.6% acc).
**Layer:** L19 (2/3 depth; same used in Exp 2). **Position:** prefill-end (last prompt token, first row of `states` in the extraction npz).
**Data:** Pathway 11 cached extractions — 500 MATH-500 × 2 models, plus BBH 3×250 × 1.5B.

## TL;DR

1. **7B prefill PR inversion is real and robust.** Correct-group participation ratio (26.87) > incorrect-group (19.51). Ratio point = 1.377; bootstrap mean 1.485, 95% CI [1.226, 1.757], P(ratio>1) = **1.000** in 1000 bootstraps. Balance-controlled (subsample 366→134 correct): ratio 1.227, CI [1.131, 1.318], P(>1) = 1.000.
2. **1.5B shows no inversion.** Ratio 0.946, CI [0.843, 1.066] overlaps 1.0.
3. **Both models show normal (correct << incorrect) pattern at final token** — 7B ratio 0.38, 1.5B ratio 0.51.
4. **Root cause is NOT pure "capability breadth"** — it's more subtle. By difficulty level: 7B Level 2 ratio = 3.98, Level 5 ratio = 1.11. The inversion is strongest when the *incorrect group is small and concentrated* (few easy problems that fail in similar ways). The 1.5B inverts on Levels 1-3 but normalizes on 4-5 where incorrect groups are large and diverse.
5. **D-bucket has a distinctive geometric signature** — lowest prefill PR of any bucket (14.49 vs A=18.7, B=16.6, C=20.0). But its prefill DoM score is in the "low confidence" zone where current gating would route it to K=8 (wrong action). Additional signal like prefill-PR (local/neighborhood) could rescue these.

## Phase 1 — Bootstrap validation

| Setting | Point ratio | Bootstrap mean | 95% CI | P(ratio>1) | Balance-controlled |
|---|---|---|---|---|---|
| **7B prefill** | **1.377** | **1.485** | **[1.226, 1.757]** | **1.000** | **1.227 [1.131, 1.318]** |
| 1.5B prefill | 0.946 | 0.951 | [0.843, 1.066] | 0.187 | 0.949 [0.932, 0.968] |
| 7B final-token | 0.384 | 0.416 | [0.365, 0.479] | 0.000 | 0.381 [0.350, 0.428] |
| 1.5B final-token | 0.509 | 0.521 | [0.412, 0.645] | 0.000 | 0.511 [0.502, 0.523] |

The 7B prefill inversion is **not an artifact of class imbalance** — even when the 366-problem correct group is subsampled to 134 (matching incorrect), the ratio stays 1.227 and every one of 100 balanced draws showed ratio > 1.

Final-token direction is "normal" in both models: correct answers converge to a low-PR manifold (PR~2.6 for 7B, ~4.3 for 1.5B); wrong answers spray (~6.8 / ~8.5). The 7B correct final-token PR of 2.6 on 366 problems is striking — nearly rank-2 geometry at scale.

See: `bootstrap_ci.png`.

## Phase 2 — Cross-scale analysis

### (a) LOO PR contribution

7B correct-group PR = 26.87. The 10 problems whose removal drops PR most (biggest individual PR contributors):
`[255, 296, 344, 368, 119, 34, 257, 135, 461, 277]`.

Saved to `phase2_loo_deltas.npz` (delta_correct_7b, delta_incorrect_7b, delta_correct_15b, delta_incorrect_15b).

### (b) Three-way split — the capability-breadth story, partially

| Bucket | n | 7B prefill PR | 7B final PR | 1.5B prefill PR | 1.5B final PR |
|---|---|---|---|---|---|
| both_right | 230 | 26.26 | 2.59 | 19.14 | 4.12 |
| **only_7b** | **136** | **25.15** | **2.71** | **20.45** | **8.53** |
| only_15b | 13 | 8.14 | 3.98 | 8.10 | 3.11 |
| both_wrong | 121 | 19.15 | 6.65 | 17.91 | 7.83 |

The 7B correct group (both_right + only_7b = 366 problems) spans a high-dim manifold at prefill (PR ≈ 25-26). Its incorrect group (only_15b + both_wrong = 134) has lower PR because it's dominated by 121 both-wrong problems (PR 19.15) — hard problems where both models fail, likely concentrated on similar failure modes.

The **only_7b bucket has 7B-prefill PR 25.15** — supporting the "capability breadth" element: problems the 7B solves that 1.5B can't still live on a diverse high-PR manifold at 7B prefill time. But Phase 3's level-breakdown complicates this interpretation.

See: `three_way_split.png`.

### (c) BBH check

| Subset | Acc | n_c / n_i | Prefill PR ratio | Final-tok PR ratio |
|---|---|---|---|---|
| tracking_shuffled_objects_seven_objects | 0.124 | 31 / 219 | 0.930 | 0.578 |
| logical_deduction_seven_objects | 0.060 | 15 / 235 | 0.824 | 0.290 |
| web_of_lies (balanced) | 0.540 | 135 / 115 | **1.009** | 0.693 |

**No prefill inversion on 1.5B for any BBH subset.** Even on web_of_lies at 54% balanced accuracy, the prefill PR ratio is 1.009 — essentially tied. So the 7B prefill inversion is NOT a generic "balanced task produces it" phenomenon — it's 7B-specific.

### (d) D-bucket signature (new sub-phase)

Buckets defined on 1.5B: A = K=1 ✓ ∩ K=8 maj ✓ (207), B = K=1 ✗ ∩ K=8 maj ✓ recoverable (68), C = K=1 ✗ ∩ K=8 maj ✗ hopeless (189), D = K=1 ✓ ∩ K=8 maj ✗ pathological (36).

| Bucket | n | prefill PR | final-tok PR | prefill DoM | gen length |
|---|---|---|---|---|---|
| A always-right | 207 | 18.72 | 4.06 | +2.83 | 422 |
| B recoverable | 68 | 16.65 | 7.55 | +0.79 | 613 |
| C never-right | 189 | 20.04 | 8.28 | −3.34 | 739 |
| **D pathological** | **36** | **14.49** | **4.94** | **−1.16** | **527** |

**D HAS a distinctive geometric signature:**

- **Lowest prefill PR** of any bucket (14.49, clearly below A=18.7, B=16.6, C=20.0). This is the strongest distinguishing signal.
- **Final-token PR = 4.94** — close to A's 4.06, NOT like C's 8.28. D problems "look correct" at final-token PR — consistent with K=1 being right.
- **Prefill DoM score = −1.16** (negative, between B's +0.79 and C's −3.34). Current gating would route D to K=8 — the wrong action, since K=8 majority actively flips these to wrong.
- **Length = 527** — intermediate. D_vs_A: D is *longer* (p=3e-3, d=0.55). D_vs_C: D is *much shorter* (p=3e-6, d=-0.84). **Not "short-and-correct-at-K=1"**. D problems have moderate-length reasoning that greedy K=1 happens to land on the right answer for, but T=0.7 variance knocks K=8 majority off.

**Interpretation:** D-bucket pathology is partly sampling-induced (T=0.7 variance), but the reasoning lengths aren't pathologically short — these aren't "obvious problems solved in 100 tokens." The low prefill PR suggests D problems cluster in a small, concentrated subspace of activation space that's geometrically distinct from both A (confident) and C (hopeless).

**Policy implication for Pathway 10 v2:** a gating policy that uses both *prefill DoM score* AND *neighborhood prefill PR* (or local intrinsic dimension) could detect D and force K=1 even when the score alone says "low confidence → K=8." Combined with Exp 2's finding that length-residualization should help, the next-gen gate should be multi-signal.

See: `d_bucket_signature.png`.

## Phase 3 — Mechanistic analysis

### PR per difficulty level

MATH-500 has levels 1-5 with counts {43, 90, 105, 128, 134}.

| Level | 7B PR_c | 7B PR_i | 7B ratio | 1.5B PR_c | 1.5B PR_i | 1.5B ratio |
|---|---|---|---|---|---|---|
| 1 | (n_i=7, too few) | — | — | 13.17 | 4.41 | 2.985 |
| 2 | 22.06 | 5.54 | **3.979** | 16.07 | 11.53 | **1.394** |
| 3 | 22.35 | 9.37 | 2.385 | 15.50 | 13.81 | 1.122 |
| 4 | 23.09 | 13.23 | 1.746 | 15.08 | 17.64 | 0.855 |
| 5 | 21.72 | 19.61 | 1.108 | 12.27 | 19.31 | 0.636 |

**The inversion is a level-dependent phenomenon.** At Level 2 the 7B ratio is 3.98 (correct group 22-D, incorrect group 5.5-D!). At Level 5 it's 1.11 (barely). Same pattern for 1.5B but weaker.

**Root cause:** on easier levels, the few problems a model gets wrong cluster into a *tight, low-dim failure mode* (small n_incorrect, similar error patterns → low PR_incorrect). As difficulty rises, both correct and incorrect groups contain diverse problem types → PR equalizes.

This recasts the "inversion" as **incorrect-group concentration**, not as a deep property of 7B representations. The 7B inverts more than 1.5B aggregate-wise primarily because its incorrect group is heavier-weighted toward the concentrated easy/medium failures (where the 1.5B fails on so many levels that its incorrect group is diverse).

### K-means clustering

PCA-64 explains 79.8% of 7B prefill variance. K-means on PCA'd activations at k∈{5,6,8,10}:

| k | 7B ARI(correct) | 7B ARI(level) | 1.5B ARI(correct) |
|---|---|---|---|
| 5 | 0.087 | 0.067 | 0.057 |
| 6 | 0.063 | 0.066 | 0.055 |
| 8 | 0.045 | 0.052 | 0.048 |
| 10 | 0.035 | 0.047 | 0.032 |

All ARI values near zero. Prefill activations **do not cluster by correctness or difficulty** — the correctness signal from Exp 2 (AUROC 0.77) is a *linear direction*, not cluster structure. Top-accuracy k=5 clusters have mean level ~2-3 (easier problems), bottom-acc clusters ~4-5 (harder). That's consistent with "difficulty predicts accuracy" but gives no additional clustering information.

## Takeaways

1. **The 7B prefill inversion is real, quantifiable (ratio 1.38, CI [1.23, 1.76]), and not a class-imbalance artifact.**
2. **But it's not a clean "capability breadth" effect.** The deeper explanation is "incorrect-group concentration": on easier difficulty levels, the few problems that fail do so in similar ways, producing a concentrated low-PR incorrect manifold. This effect is stronger in the 7B because its failures skew easier-level than the 1.5B's.
3. **D-bucket has a distinctive fingerprint:** lowest prefill PR (14.49) of any bucket, final-token PR that "looks correct", intermediate length. The signature exists — but current DoM-score gating can't exploit it because score puts D in the low-confidence zone where it would be wrongly routed to K=8.
4. **Practical policy implication:** a multi-signal gate (prefill DoM + local prefill PR + length) could detect D and force K=1. Combined with the middle-heavy gating finding from Exp 2, this suggests non-monotonic multi-feature policies are the next direction for compute gating, not single-threshold DoM.
5. **Clean negative control:** no prefill inversion in 1.5B on BBH, including balanced web_of_lies. The inversion is a 7B-MATH feature.
6. **For Pathway 10 v2:** avoid claiming "7B activations separate correct from incorrect by dimension". A more defensible claim: "7B incorrect problems on easy levels form a small concentrated subspace in activation space." That framing survives all the controls.

## Files

All in `pathway11_h100/prefill_inversion/`:
- `phase1_bootstrap.json` — bootstrap ratios, CIs, balance controls (all 4 settings)
- `phase2_cross_scale.json` — LOO pushers, three-way split, BBH, D-bucket
- `phase2_loo_deltas.npz` — per-problem LOO delta PR arrays
- `phase3_mechanistic.json` — PR per level × correctness, k-means ARI
- `results.json` — consolidated
- `bootstrap_ci.png`, `three_way_split.png`, `d_bucket_signature.png`
- `common.py` — PR computation and data loaders (shared)
- `cache/m7b_prefill.npz`, `cache/m15b_prefill.npz` — cached prefill+final arrays (fp16)
