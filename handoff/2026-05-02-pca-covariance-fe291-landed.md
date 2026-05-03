# Handoff — PCA-on-prefill-covariance ablation (FE291) landed

**Date:** 2026-05-02
**Session entry:** continued from `2026-05-01-phase4-5-promotes-landed.md`,
"After-this-session resume" → ranked move #1 (PCA-on-per-problem-covariance).

## What landed

One result brief promoted, F-2 and F-10 each gained one evidence row,
EXPERIMENT_LOG +EXP-55, two PERSPECTIVES.md sections appended.

| Step | What | Outcome |
|---|---|---|
| 1 | New experiment script | `pathway11_h100/pca_covariance/recompute_pca_covariance.py` (~1.5s CPU regen) |
| 2 | Result | `pathway11_h100/pca_covariance/results.json` |
| 3 | Brief | `research-graph/briefs/result-2026-05-02-P11-FE291.md` |
| 4 | validate_claims.py | +9 entries (Tier-0 + Tier-1 regen wired) |
| 5 | Tier-0 readback | 166 PASS / 0 FAIL / 0 MISSING |
| 6 | promote_result.py | ✅ → **EXP-55**; F-2 +1 evidence (4 props), F-10 +1 evidence (3 props) |
| 7 | NEXT_EXPERIMENTS.md | Regenerated (232 watchlist papers) |
| 8 | PERSPECTIVES.md | +1 section "DoM is the dominant variance direction at L19 prefill" + PC9 hint + framing-update for causal tests |

EXPERIMENT_LOG Next-ID = **EXP-56**.

## Headline numbers

| Quantity | Value | Status |
|---|---|---|
| Supervised DoM single-feature OOF AUROC (matched 5-fold) | 0.7679 | EXP-55 logged |
| Unsupervised PCA-PC1 single-feature OOF AUROC | **0.7458** | EXP-55 logged |
| CAST class-mean PCA-PC1 single-feature OOF AUROC | 0.7458 | EXP-55 logged |
| cos(supervised DoM, unsupervised PC1) | 0.9216 | EXP-55 logged |
| cos(CAST PC1, unsupervised PC1) | ≈ 1.000 | EXP-55 logged |
| PC1 variance share at L19 prefill | 14.7% | EXP-55 logged |
| Unit-DoM energy in PC1 alone | 84.9% | EXP-55 logged |
| Unit-DoM energy in top-10 PCs | 97.6% | EXP-55 logged |
| Best PC (by AUROC) | PC1 (also rank-1 by variance) | EXP-55 logged |
| Secondary correctness PC | **PC9** AUROC 0.6575, var share 2.5%, DoM-coeff 0.225 | EXP-55 logged |

## What changed in the framing

F-2 simultaneously **strengthens** and **narrows**. The L19 prefill
correctness direction is *unsupervised-identifiable* — the dominant
eigenvector of the prefill covariance is 92% the supervised DoM, with
85% of DoM-energy concentrated in that one direction. The supervised
probe wasn't discovering hidden structure; it was reading off the most
prominent direction at the layer.

F-10 **sharpens**. "Topology adds no signal beyond covariance" is now
literal — the covariance pathway is a single direction, PC1. PH
features layered on the cloud cannot improve on a mass-mean projection
along it. The next-level falsifiable form of F-10 is: PH on the
[PC2..PC1536] subspace should still be null-bound. If yes, F-10's
strongest form holds; if no, there's residual orthogonal-to-PC1
topology missed.

CAST's class-mean centering vs unsupervised global-mean centering gives
*identical* PC1 in this data (Δ AUROC 0.0001, cos ≈ 1.000) because
classes are roughly balanced (243 correct / 257 incorrect). CAST's
reported steering gains over plain DoM presumably come from the layer/
threshold grid search (FE292), not from the PCA-PC1 step.

## Files modified / created

```
NEW    pathway11_h100/pca_covariance/recompute_pca_covariance.py
NEW    pathway11_h100/pca_covariance/results.json
NEW    pathway11_h100/pca_covariance/run.log
NEW    research-graph/briefs/result-2026-05-02-P11-FE291.md
EDIT   validate_claims.py  (+9 Claim entries with Tier-1 regen wired)
EDIT   PERSPECTIVES.md  (+3 subsections under "Late additions (2026-05-02)")
EDIT   STATE.md, EXPERIMENT_LOG.md, FINDINGS.md, NEXT_EXPERIMENTS.md  (mutated by promote_result.py)
NEW    handoff/2026-05-02-pca-covariance-fe291-landed.md  (this file)
```

## After-this-session resume

Reasonable next moves, ranked by ROI / cost:

1. **Two-feature [PC1, PC9] OOF logistic** (~10 lines Python, no new
   compute — extend `recompute_pca_covariance.py`). Closes the loop on
   "is PC9 exactly the orthogonal correctness direction the supervised
   probe is recovering?" Prediction: 2-feature AUROC ≈ 0.76 ± 0.005,
   matching DoM 0.7679. If yes, DoM = PC1 + 0.225·PC9 plus noise, and
   the basis-decomposition story closes cleanly. Could fold into
   FE291's results.json as a follow-up field.

2. **PH on [PC2..PC1536] residualized clouds** (FE10 sharpening test;
   ~30min CPU on cached pathway8 layerwise NPZs). Project out PC1 from
   each per-problem cloud, recompute the 5-feature PH, compare real
   vs matched-cov Gaussian null. If still null-bound, F-10's strongest
   form holds. If real > null + 0.05, there's residual orthogonal
   topology.

3. **Causal tests of PC1 (not "DoM")** — same FE214/FE269/FE283 plan
   from yesterday's handoff but now ablating along the named PC1
   direction. Same H100 cost (~2-4h per test).

4. **Pull current `NEXT_EXPERIMENTS.md` top-5** — queue regenerated
   post-promote and may have re-ranked.

`validate_claims.py` Tier-1 regen for the 9 new pca claims is wired and
verified (the regen script prints all 10 `pca.<key>=<value>` lines on
stdout, ~1.5s CPU). No 600s timeout concern.

## Memory aids

- Next EXP-id: **EXP-56**.
- F-2 strength: **STRONG**, status **ACTIVE**, evidence rows 1..13 (was 12 → +EXP-55).
- F-10 strength: **STRONG**, status **ACTIVE**, evidence rows 1..5 (was 4 → +EXP-55).
- F-3 / F-9 / F-11..F-14 unchanged.
- New claim count: 175 PASS internal back-checks (was 166 in mid-session
  before promote; the post-promote re-run hasn't been done — the
  promote's gate did run validate_claims.py with full Tier-1 regens
  successfully, exit 0, so the invariant holds).
- Cached prefill `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
  remains the backbone for any further covariance / PC follow-ups.
- The recompute script is sub-2-second; safe to embed in any future
  ablation that wants a fresh PC1 baseline.
