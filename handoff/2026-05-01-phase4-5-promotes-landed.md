# Handoff — Phase 4 + 5 promotes landed (post-compact)

**Date:** 2026-05-01 (late evening, post-compact)
**Session entry:** continued from `2026-05-01-phase4-5-mid-promote.md`;
auto-mode active.

## What landed in this session (post-compact)

Three result briefs promoted, three sibling FE statuses flipped, two
bugs in validate_claims/regen surfaced and fixed, PERSPECTIVES.md
extended with two new "what surprised me" sections.

| Step | FE | Outcome |
|---|---|---|
| 1 | P11-FE321 | ✅ Promoted → **EXP-52**; F-10 +1 evidence (4 props updated) |
| 2 | P11-FE110 (+FE23, +FE455 siblings) | ✅ Promoted → **EXP-53**; F-2 +1 evidence (4 props updated) |
| 3 | P10-FE23 | ✅ `update_status.py … COMPLETED` ("anti-calibrated 0.4395") |
| 4 | P11-FE455 | ✅ `update_status.py … COMPLETED` ("ConCISE c_hat 0.5887") |
| 5 | P11-FE116 | ✅ Promoted → **EXP-54**; F-10 +1 evidence (4 props updated) |
| 6 | NEXT_EXPERIMENTS.md | ✅ Regenerated (954 experiments, 232 watchlist papers) |
| 7 | PERSPECTIVES.md | ✅ Two new sections appended |

`validate_claims.py --no-regen`:
**157 PASS / 0 FAIL / 0 MISSING / 41 REGISTERED / 3 PENDING_FE = 201 tracked.**
EXPERIMENT_LOG Next-ID = **EXP-55**.

## Two bugs fixed mid-session

### Bug 1: FE116 H_1_n_features key path typo (Tier-0 readback FAIL)

The 9 FE116 claims were authored with path
`["feature_comparison", "H_1_n_features", …]` but the JSON key is
`H1_n_features` (no underscore between H and 1). Both H_1_n_features
claims came up `KEY_MISSING` on the very next `validate_claims` run.

**Fix:** Edit at validate_claims.py:467,470 — drop the underscore.
PR-ready, no JSON re-write needed.

### Bug 2: recompute_gpu_bundle.py didn't emit Tier-1 regen-readback lines

The 7 GPU bundle claims declared
`regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py"`
with `regen_key="fe110.max_cos_with_dom"` etc. The script wrote a JSON
file but **never printed** parseable `key=value` lines on stdout, so
validate_claims's `_exec_regen` parser found no values and reported all
7 as `REGEN_KEY_MISSING`. That's a hard failure under `regen_hard_fail`
counting → exit 1 → promote gate refused.

**Fix:** Append 7 `print(f"fe110.max_cos_with_dom={fe110_max_cos:.10f}")`-
style lines at the end of `recompute_gpu_bundle.py` (just before the
final `return 0`). The variable names in scope at that point are
`fe110_max_cos`, `fe110_max_auroc`, `auroc_sup`, `auroc_fe23_conf`,
`auroc_fe455`, `auroc_fe455_just_conf`, `auroc_fe455_just_sure` — these
match the regen_key values declared in validate_claims.py:412–445.

The pattern to remember when adding Tier-1 regen for any future script:
**the regen script must print `key=value` to stdout at the end** (one
line per claim), where `key` exactly matches `regen_key`. Tier-1 regen
is parsed from stdout, not from the JSON. The recompute_fe145.py and
recompute_phase3.py scripts are the canonical references — both end
with a block of `print(f"{name}={value:.10f}")` calls.

### Brief filename mismatch (third hiccup, not really a bug)

`promote_result.py` enforces that the YAML `fe_id` in the brief
matches the filename FE-id segment. The GPU bundle brief was named
`result-2026-05-01-P11-FE110+FE23+FE455.md` but its primary YAML block
declared just `fe_id: P11-FE110`. Promote refused with
`Filename declares fe_id=P11-FE110+FE23+FE455 but ## FE block declares
fe_id=P11-FE110.`

**Fix:** Renamed the brief to `result-2026-05-01-P11-FE110.md`.
Sibling FEs (P10-FE23, P11-FE455) handled out-of-band via
`update_status.py … COMPLETED`. Going forward: name multi-FE bundle
briefs after the primary YAML id; document siblings in the body.

## Headline numbers (recap)

| Quantity | Value | Status |
|---|---|---|
| FE321 7-descriptor real AUROC | 0.7156 vs null 0.6188 (gap +0.097) | EXP-52 logged |
| FE321 B_1+bar_Z_1 alone | 0.4952 vs 0.5611 (gap −0.066, degenerate) | EXP-52 logged |
| FE116 real PH AUROC on residuals | 0.6958 | EXP-54 logged |
| FE116 null PH AUROC on residuals | 0.7624 (**gap −0.067, INVERTED**) | EXP-54 logged |
| FE116 H_1 cycle count real vs null | 31.3 vs 87.7 | EXP-54 logged |
| FE110 best ActAdd cos with DoM | +0.065 (max AUROC 0.6615) | EXP-53 logged |
| FE23 softmax-conf AUROC at PANL | 0.4395 (BELOW chance) | EXP-53 logged |
| FE455 ConCISE c_hat AUROC | 0.5887 | EXP-53 logged |

## F-2 / F-10 status after this session

- **F-2 [STRONG, ACTIVE]**: now 12 evidence rows. ActAdd contrast cosine
  +0.065 (≪ 0.5 recovery threshold), softmax-conf 0.4395 (anti-
  calibrated), ConCISE 0.5887 — none subsume the L19 prefill DoM
  AUROC 0.7731. F-2 cleanly survives three more challenger probes.
- **F-10 [STRONG, ACTIVE]**: now 4 evidence rows in this session's
  branch. The framing strengthens to *"topology adds no signal beyond
  covariance"* — FE116 inversion shows matched-cov Gaussian beats real
  PH on residualized clouds. FE321 zigzag-style descriptors confirm
  trajectory geometry signal exists (gap +0.097) but isn't topological.

## PERSPECTIVES.md additions

Two sections appended to the end:

1. **"Per-problem covariance carries the signal F-10 was supposed to
   refute"** — frames the FE116 inversion: real residuals are smoother
   than rank-matched Gaussians, and the SVD null inherits empirical
   covariance which itself encodes problem difficulty. Suggests PCA on
   per-problem covariance as the natural next ablation.
2. **"Softmax confidence is anti-calibrated on 1.5B at PANL"** — flags
   that AUROC 0.4395 < 0.5 means high softmax confidence at the
   pre-`\boxed{` position correlates *with* error. Worth pre-empting in
   any selective-prediction discussion. ConCISE c_hat at 0.589 is
   weakly positive — lexical hedging informative, raw softmax peaks
   anti-informative.

## Files modified / created this session (post-compact)

```
EDIT  validate_claims.py
        - fe116-h1-n-features-real path: H_1_n_features → H1_n_features
        - fe116-h1-n-features-null path: same
EDIT  pathway11_h100/gpu_bundle/recompute_gpu_bundle.py
        - +7 trailing `print(f"key=value")` lines for Tier-1 regen-readback
RENAME  research-graph/briefs/result-2026-05-01-P11-FE110+FE23+FE455.md
        → research-graph/briefs/result-2026-05-01-P11-FE110.md
EDIT  STATE.md, EXPERIMENT_LOG.md, FINDINGS.md, NEXT_EXPERIMENTS.md
        (all mutated by promote_result.py runs for FE321, FE110, FE116)
EDIT  PERSPECTIVES.md
        +1 section "Per-problem covariance carries the signal F-10 was
        supposed to refute"
        +1 section "Softmax confidence is anti-calibrated on 1.5B at PANL"
NEW   handoff/2026-05-01-phase4-5-promotes-landed.md  (this file)
```

Three result briefs are now part of the project record:

```
research-graph/briefs/result-2026-05-01-P11-FE321.md  (Phase 5 zigzag PH)
research-graph/briefs/result-2026-05-01-P11-FE110.md  (Phase 4 GPU bundle)
research-graph/briefs/result-2026-05-01-P11-FE116.md  (Phase 5 PH on residuals)
```

## After-this-session resume

The Phase 4 + 5 work plan from `2026-05-01-phase2-3-cheap-wins.md` is
**fully complete**. Five completed experiments (FE282 cache-completion,
FE321, FE110, FE116, plus FE23/FE455 sibling-promoted), three briefs
landed, two regen bugs caught and patched.

Reasonable next moves, ranked:

1. **PCA-on-per-problem-covariance ablation** (cheap, ~1 H100 hour,
   directly motivated by FE116 inversion). Test whether top-k principal
   components of the per-problem 1536-d covariance match DoM AUROC.
   If yes, DoM is a covariance principal component → the directions
   framing factors through second-order statistics. If no, DoM is
   recovering a higher-order moment.
2. **Per-position DoM bank + steering sweep** (the "if I had unlimited
   compute" experiment from PERSPECTIVES.md, ~3 H100-days, ~$200).
   Resolves whether the prefill signal is *actionable* (steering moves
   accuracy) or merely *diagnostic* (good selective predictor).
3. **Pull current `NEXT_EXPERIMENTS.md` top-5** and pick whatever the
   ROI scoring now floats — the queue regenerated post-promote and
   incorporates the three new EXPs as evidence.

`validate_claims` has 7 Tier-0-only entries (FE321) and 9 Tier-0-only
entries (FE116) — neither has automated Tier-1 regen because the
underlying recompute scripts take >600s. The manual regen commands
remain commented in validate_claims.py preceding each block. This
mirrors the FE749 (~2h45m) precedent.

## Memory aids

- Next EXP-id: **EXP-55** in EXPERIMENT_LOG.md.
- F-2 strength: **STRONG**, status **ACTIVE**, evidence rows 1..12.
- F-3 strength: **STRONG**, status **ACTIVE**, evidence rows 1..5.
- F-9 strength: **MODERATE**, status **ACTIVE**, evidence rows 1..3.
- F-10 strength: **STRONG**, status **ACTIVE**, evidence rows 1..4.
- F-11..F-14: see FINDINGS.md (auto-updated by promotes).
- Cached prefill at `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
  (500 × 1536 fp16 prefill at L19 + 500 × 1536 final-tok + correct + seq_len)
  — usable for the PCA-on-covariance follow-up without re-loading the model.
- Cached W_unembed at `pathway11_h100/phase3_corroborators/W_unembed_15b.npy`
  (151936 × 1536 fp32, 891 MB).
- 2060 Super has 8 GB; Qwen-2.5-1.5B-Instruct fp16 fits with ~5 GB headroom.
- `promote_result.py` enforces filename ↔ YAML fe_id match — name
  multi-FE bundle briefs after the primary YAML id.
- For Tier-1 regen: the recompute script **must print `key=value` lines**
  to stdout — JSON-only output won't be parsed. See
  `recompute_fe145.py` for the canonical pattern.
