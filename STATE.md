# STATE.md — where was I

*Overwritten at the end of every session. Not appended. For append-only history see `EXPERIMENT_LOG.md`.*

**Date:** 2026-04-24

**Last experiment completed:** EXP-042 (Test 2 — no-CoT control, INCONCLUSIVE
as framed because the answer-only prompt collapses generation to 2 tokens).

**Top 3 hypotheses to test next:**
1. **H-2 short-CoT breathing test** — ~5 min H100, ~$0.25. Decisive on
   whether breathing tracks length or CoT-reasoning specifically. Blocks
   the natural follow-up to EXP-042.
2. **H-1 per-position DoM steering** — ~1 H100-day, ~$200. The single
   experiment that would most reshape what the project is about (diagnostic
   vs lever). Mentioned in both PROJECT_RECORD §1e and PERSPECTIVES.
3. **H-6 CoE-60 re-baseline at 1024 tok** — CPU only, ~30 min, free.
   Validates whether the CoE > ABC-44 headline survives label correction.

**Blocked experiments:**
- H-4 (breathing ↔ EoS) — needs Qwen-2.5-1.5B training checkpoints; ~3
  H100-days. Unblocked by HF hub access + budget.
- H-8 (prefill direction across training checkpoints) — same checkpoint
  access requirement.

**Pod status:**
- `y687b9z2dgukcj` (Pathway 11 main pipeline): **REMOVED** on 2026-04-24 at
  exp1_cross_model end-of-session. Had volumeInGb=0 so stop wouldn't have
  preserved disk. Re-create from scratch if needed; ~15 min setup.
- `lsuoka6bo8io7m` (gibberish + no-CoT): **STOPPED, volumeInGb=50, disk
  preserved**. Resume with `runpodctl pod start lsuoka6bo8io7m`. Note: pip
  packages live on container disk (not volume) and need reinstalling after
  stop/start.

**Disk usage summary:**
- `pathway11_h100/data/math500_7b/` — 42 GB (7B MATH-500, 1024 tok)
- `pathway8_layerwise/data/math500/` — 19 GB (1.5B MATH-500, 1024 tok)
- `pathway8_layerwise/data/bbh/` — 18 GB (BBH 3×250, 1024 tok)
- All other caches combined — < 1 GB.
- **Total project** — ~94 GB on `~/topo-confidence/`.

**Open threads for next session:**
- Run `python validate_claims.py` as a smoke test — should PASS 91/91.
- If pursuing H-1, first audit whether the Stage-2 NPZs preserved full
  per-token states or only last-token (the one-line answer is in
  `pathway8_layerwise/extract_math500.py`; schema probe confirms 29-layer ×
  T-gen × 1536 shape).
- `pathway11_h100/` is NOT in the GitHub remote; back up if the local disk
  is at risk.
