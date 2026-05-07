# Handoff — 2026-05-07 — Second Sweep Complete (10/10 PASS)

## What happened

Implemented and ran 10 CPU experiments (EXP-69–78) from plan `.claude/plans/dapper-sprouting-rose.md`. All 10 passed. FE119 had a numpy bool JSON serialization bug (fixed inline, re-ran). Total wall time ~9 minutes.

## Result JSONs

All at `pathway11_h100/results/`:

| EXP | FE | JSON | Headline |
|---|---|---|---|
| 69 | FE447 | `fe447_length_baseline.json` | OOF residualized DoM AUROC = **0.620** (in-sample was 0.665). Spearman(dom,len) = -0.619. Length AUROC cross-check = 0.7986 ✓ |
| 70 | FE188 | `fe188_lid_mle.json` | Mean LID = 15.7–26 (k-dep). Best AUROC = **0.535**. LID ≈ PR confirms local≈global dim. NOT a correctness predictor |
| 71 | FE421 | `fe421_regularized_concat.json` | **Ridge concat 0.851, ridge final-only 0.849, ridge prefill 0.784**. DoM concat 0.757. Huge untapped signal in final token via regularized probe |
| 72 | FE15 | `fe15_length_band_pr.json` | PR: Short=17.5, Med=21.5, Long=20.9. Per-band DoM AUROC holds (0.75–0.83). Global PR = 19.86 ✓. F-4 NOT a length artifact |
| 73 | FE16 | `fe16_mp_bias_pr.json` | Corrected PR = 18.9 (naive 19.9, ~5% correction). Correct=18.5 vs incorrect=19.6. Class asymmetry survives MP correction |
| 74 | FE428 | `fe428_l0_embedding_dom.json` | **L0 AUROC = 0.500, cos(DoM_L0, DoM_L19) = 0.0, PR_L0 = 0.0**. Zero signal at embedding layer — F-2 L19 specificity fully confirmed |
| 75 | FE416 | `fe416_prefinal_token_dom.json` | cos(prefinal, prefill) = **-0.054**, cos(prefinal, final) = **0.717**. Prefinal AUROC = 0.736. NOT a positional artifact — orthogonality is gradual |
| 76 | FE119 | `fe119_layer_sweep_cos.json` | cos(prefill_DoM, final_DoM) near zero at ALL 29 layers (max |cos| = 0.11 at L28). F-3 orthogonality is network-wide, not L19-specific |
| 77 | FE308 | `fe308_adaptive_bestofk.json` | Adaptive ≈ uniform at all K levels (~41%). Cache K=1 = 41.6% (vs greedy 48.6%). T>0 sampling degrades this model; adaptive allocation can't rescue it |
| 78 | FE459 | `fe459_cross_model_dom.json` | **7B AUROC = 0.874, Spearman(1.5B, 7B) = 0.937**, Kendall tau = 0.779, concordance = 0.890. 7B correct = 366 ✓. Difficulty geometry is universal across model sizes |

## Key surprises

1. **FE421**: Ridge-LR on final-token alone (0.849) matches concat (0.851) and vastly exceeds prefill DoM (0.773). The final token carries MORE correctness signal than prefill when probed properly. This challenges F-2's "prefill is special" framing.
2. **FE428**: Perfect zero — L0 has literally no correctness geometry. The signal emerges entirely in deeper layers.
3. **FE459**: Spearman 0.937 cross-model — the DoM scores rank problems nearly identically across 1.5B and 7B despite very different accuracy (48.6% vs 73.2%).
4. **FE119**: Orthogonality is everywhere, not just L19. Prefill and final DoM directions are independent at every layer.
5. **FE308**: Adaptive best-of-k is a dead end — T>0 sampling is fundamentally degraded for this model size.

## What to do next

**Write findings down.** Full post-experiment workflow:

1. **Write consolidated result brief**: `research-graph/briefs/result-2026-05-07-P11-second-sweep.md` — structured brief with all 10 results, YAML FE blocks, interpretations
2. **Update EXPERIMENT_LOG.md**: Append EXP-69 through EXP-78 entries. Next ID becomes EXP-79
3. **Update STATE.md**: Replace "Last experiment completed" section
4. **Neo4j updates**: `python research-graph/update_status.py P11-FE<ID> COMPLETED --outcome "..."` for each of the 10 FEs
5. **Regenerate queue**: `python research-graph/generate_next_experiments.py`
6. **Update validate_claims.py**: Add claims for new headline numbers (especially FE421 0.851, FE459 0.874/0.937, FE428 0.500)
7. **Consider FINDINGS.md updates**: FE421's ridge-final 0.849 may warrant updating F-3 or creating a new finding. FE459's cross-model correlation may strengthen F-2
8. **Commit + push** on `max-depth-retriage-2026-04-28`

## Scripts created this session

```
pathway11_h100/length_baseline/recompute_fe447.py
pathway11_h100/lid_mle/recompute_fe188.py
pathway11_h100/regularized_concat/recompute_fe421.py
pathway11_h100/length_band_pr/recompute_fe15.py
pathway11_h100/mp_bias_pr/recompute_fe16.py
pathway11_h100/l0_embedding_dom/recompute_fe428.py
pathway11_h100/prefinal_token_dom/recompute_fe416.py
pathway11_h100/layer_sweep_cos/recompute_fe119.py
pathway11_h100/adaptive_bestofk/recompute_fe308.py
pathway11_h100/cross_model_dom/recompute_fe459.py
pathway11_h100/run_second_sweep.sh
```

## Branch / commit state

Branch: `max-depth-retriage-2026-04-28`. Scripts + results are uncommitted. No new commits yet.
