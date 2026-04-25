# figures/ — consolidated PNG archive

Every figure produced across Pathways 1–11 that carries a load-bearing claim
in the project record. Originals remain in their pathway directories; these
are renamed copies.

One-sentence captions + cross-reference to PROJECT_RECORD.md section.

## Breathing / dimensional PR curves

- **`breathing_temporal_phi3_llama.png`** — Temporal PR curve at 2/3-depth
  layer for Phi-3-mini (L21) and Llama-3.2-1B (L11) on MATH-500.
  Shows the three-phase breathing (rise → peak ~100/~115 → collapse)
  replicating across architectures. Source:
  `pathway11_h100/exp1_cross_model/temporal_pr_curve.png`. Supports
  PROJECT_RECORD §1a (Pathway 11) and RESEARCH_SUMMARY §3 (cross-architecture
  replication).

- **`breathing_gibberish_control.png`** — Same PR axes, three conditions:
  MATH-500 (n=50, classic breathing), stream-of-consciousness (n=20, muted
  early peak), and random tokens (n=20, flat ≈ 10 across all positions).
  Rejects the AR-decoding null. Source:
  `pathway11_h100/gibberish_control/gibberish_vs_math500.png`. Supports
  PROJECT_RECORD §1b.vii and RESEARCH_SUMMARY §3 (gibberish control).

- **`breathing_nocot_control.png`** — CoT (median 255 gen tokens) vs no-CoT
  (median 2). Shows PR curve physically cannot exist without a trajectory;
  pos 1 no-CoT (15.78) is slightly higher than pos 1 CoT (13.74). Source:
  `pathway11_h100/no_cot_control/nocot_vs_cot.png`. Supports PROJECT_RECORD
  §1b.vii and the no-CoT caveat in the negative-results section.

## Exp 1 cross-model depth

- **`exp1_depth_pr_prefill_vs_final.png`** — Depth-PR for Phi-3-mini and
  Llama-3.2-1B at prefill-end and final-token across all layers. Shows the
  late-layer `PR_correct > PR_incorrect` signature reproducing at different
  absolute depths (Phi-3 L27–31, Llama L12–16). Source:
  `pathway11_h100/exp1_cross_model/depth_pr_prefill_vs_final.png`. Supports
  PROJECT_RECORD §1a and RESEARCH_SUMMARY §3 (prefill-depth inversion
  reproduces).

## Prefill-inversion (Exp 3)

- **`e3_bootstrap_ci.png`** — Four-panel bootstrap distribution of
  PR_correct / PR_incorrect ratio: 7B prefill (mean 1.485, CI [1.226, 1.757]),
  1.5B prefill (0.951, [0.843, 1.066]), 7B final (0.416), 1.5B final (0.521).
  Shows 7B prefill inversion is robust and not class-imbalance. Source:
  `pathway11_h100/prefill_inversion/bootstrap_ci.png`. Supports
  PROJECT_RECORD §1b.iv.

- **`e3_three_way_split.png`** — Per-bucket PR (both-right, only-7B,
  only-1.5B, both-wrong) at prefill and final for both scales. Shows the
  capability-breadth partial story and the 121 both-wrong group's lower PR.
  Source: `pathway11_h100/prefill_inversion/three_way_split.png`. Supports
  PROJECT_RECORD §1b.iv.

- **`e3_d_bucket.png`** — D-bucket per-bucket signature chart: D has lowest
  prefill PR (14.49), final-token PR that looks correct (4.94), intermediate
  length (527), negative prefill DoM score (−1.16). Source:
  `pathway11_h100/prefill_inversion/d_bucket_signature.png`. Supports
  PROJECT_RECORD §1b.v + §1d.ND-8.

## Compute allocation (Exp 2 + 2b)

- **`e2_pareto_frontier.png`** — Compute-vs-accuracy Pareto curve for all
  gating policies tested in Exp 2: uniform K=1/2/4/8, prefill τ-thresholds,
  prefill quartile policies (top / bottom / middle-heavy / balanced),
  neg-seq-len top-heavy, oracle. Neg-seq-len dominates the frontier.
  Source: `pathway11_h100/prefill_gated_compute/pareto_plot.png`. Supports
  PROJECT_RECORD §1b.v.

- **`e2b_pareto_comparison.png`** — Same axes, adds multi-signal logreg and
  RF policies. Shows multi-signal is not Pareto-dominant over single-signal
  gating. Source: `pathway11_h100/multi_signal_oracle/pareto_comparison.png`.
  Supports PROJECT_RECORD §1b.vi.

## Pathway 1 (historical)

- **`pathway1_tier_ablation.png`** — Feature-tier A/B/C/D ablation plot from
  the original CORAL pipeline. A+B+C peaks at 0.796 holdout; adding D hurts.
  Source: `pathway1/phase2/ablation_plot.png`. Supports PROJECT_RECORD §1a
  (Pathway 1) — anchors why ABC-44 is the frozen extractor.

- **`pathway1_routing_tradeoff.png`** — Routing compute-vs-accuracy curve
  from Pathway 1 Phase 4. Supports §1a (Pathway 1).

- **`pathway1_calibration.png`** — Calibration plot from Pathway 1 Phase 4.
  Referenced only as historical context. Supports §1a (Pathway 1).

## Figures intentionally not archived

- **`breathing_temporal_qwen15b_7b.png`** — **DOES NOT EXIST.** The Qwen-1.5B
  + 7B temporal PR headline is reported only in RESEARCH_SUMMARY.md §3 as an
  inlined table, not a figure. The underlying data is at
  `pathway8_layerwise/data/math500/` (1.5B) + `pathway11_h100/data/math500_7b/`
  (7B). A 40-line matplotlib script could generate it; open in HYPOTHESES.md
  as a polishing task.

- `pathway11_h100/prefill_inversion/common.py`-cached NPZ plots — not
  archived because they're intermediate analysis artifacts, not headline
  figures.
