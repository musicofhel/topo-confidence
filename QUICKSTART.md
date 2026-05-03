# QUICKSTART.md

**Read me first if you've walked into this project cold or are a future
version of me who forgot everything.**

## What this project is

topo-confidence started as "predict LLM correctness from hidden-state
persistent homology" and ended, after three weeks of experiments, as
**"use direction-of-mean projections in a small model's prefill activations
to steer compute and refusal."** Everything topology-specific turned out to
be Gaussian-matched covariance; the geometry is real but it's linear, not
topological.

## What we found (that still holds after controls)

For the full 10-min briefing see [SYNTHESIS.md](SYNTHESIS.md). The bullets below are the working summary.

1. **Dimensional breathing.** Residual-stream covariance rises during CoT
   generation and collapses at the final token. Universal across Qwen-1.5B,
   Qwen-7B, Phi-3-mini, Llama-3.2-1B. Rejected as AR artifact by random-token
   control (flat PR ≈ 10).
2. **Prefill knows best.** Probe fit on the **prefill** (position 0) L19
   activation predicts correctness at AUROC **0.7731** OOF on 1.5B (~0.876
   on 7B), *better* than any mid-generation or final-token signal.
3. **The decomposition triangle.** F-2's signal factors into three additive
   pieces: 1-d DoM 0.7679 → 2-feat (PC1, PC9) 0.7856 → top-20 log-eigvals of
   the PC1-residualized covariance **0.7928** (the strongest L19 probe at any
   tier). The directional ceiling saturates at 2 features; the cov-spectrum
   lift is genuinely second-order. CAST PC1 ≈ supervised DoM (cosine 0.9216)
   — the dominant correctness direction is unsupervised-identifiable.
4. **Orthogonal signals.** cos(prefill_DoM, final_DoM) ≈ 0.05. "Can I solve
   this?" and "did I solve this?" live in unrelated subspaces.
5. **Selective-prediction win.** Prefill-DoM refuse-and-spend at coverage 0.5:
   **71.6%** accuracy on answered at avg K=2.5, vs **48.6%** unconditional
   at K=1. +22 pp on the kept half at lower compute.
6. **7B prefill inversion.** Correct-group PR > incorrect-group PR at 7B
   prefill (ratio 1.38, CI [1.23, 1.76]), driven by easy-level failures
   clustering. Not in 1.5B, not in BBH — 7B-MATH-specific.
7. **D-bucket exists but doesn't localize.** 36/500 problems are pathological
   (K=1 right, K=8 majority wrong). Has a collective geometric signature
   (low group prefill PR 14.49) but per-problem detection failed.

## What we found that was wrong

- **The 20.8% baseline** (and every "cross-benchmark AUROC" number before
  Phase 6.5) was a truncation artifact. At `max_new_tokens=1024` the 1.5B
  gets 48.6%, not 20.8%.
- **The 7B-superior-verifier claim** was the same truncation talking — at
  1024 tokens there's no cross-scale advantage (0.717 / 0.719).
- **PH is topology** — no. PH at rank-matched empirical-covariance Gaussian
  null (0.690 vs 0.693). It's covariance dressed differently. F-10 *strengthens*
  on PC1-residualized clouds: real PH 0.6884 vs matched-cov Gaussian null
  0.7628, gap −0.074 (pre-registered overturning condition tested directly
  and failed by 12.4 pp in the wrong direction).
- **CoE is a richer signal than single-layer DoM** — no. A single L19 DoM
  direction matches CoE-60 on cross-domain transfer (0.720 vs 0.716).
- **Fixed-vector steering (E1)** — direction rotates through generation,
  injecting a final-token-derived vector at position 15 is ~random.

## Where the data lives

- **Canonical 1024-tok caches:** `pathway8_layerwise/data/math500/` (19 GB,
  1.5B), `pathway11_h100/data/math500_7b/` (42 GB, 7B),
  `pathway8_layerwise/data/bbh/` (18 GB). All gitignored. See
  `DATA_MANIFEST.md`.
- **Current results JSONs:** `pathway11_h100/prefill_gated_compute/`,
  `prefill_inversion/`, `multi_signal_oracle/`, `exp1_cross_model/`,
  `gibberish_control/`, `no_cot_control/`.
- **Validated provenance:** run `python validate_claims.py` — 216 claims
  tracked (172 internal back-checked / 41 external REGISTERED / 3 PENDING_FE).
  Should print 172/172 internal PASS.

## What to read first

1. `SYNTHESIS.md` — the practitioner briefing. Read in 10 min.
2. `NOVELTY_AUDIT.md` — what's actually new vs the 220-paper research-graph.
3. `APPLICATIONS.md` — where this is deployable, with honest checklists.
4. `PROJECT_RECORD.md` — authoritative archive. Start with §1a (timeline)
   and §1d (negative results).
5. `PERSPECTIVES.md` — reflective notes: what surprised, what we got wrong.
6. `FINDINGS.md` and `HYPOTHESES.md` — what's alive and what to try next.
7. `STATE.md` — where the session left off.

## What to try next (ranked)

1. **Per-position DoM bank.** ~1 H100-day. Would resolve whether the 0.77
   prefill signal is usable for generation-time steering (HYPOTHESES H-1).
2. **Short-CoT breathing test.** ~$0.25. Disentangles length vs reasoning
   (H-2). The no-CoT control collapsed gen to 2 tokens, so it was
   inconclusive.
3. **CoE re-baseline at 1024 tokens.** CPU-only. Checks whether CoE-60 still
   beats single-layer DoM once labels are corrected (H-6).
4. **Breathing ↔ Sharpness Dimension.** Multi-day. The single experiment
   that would most change theoretical understanding.

~480 words.
