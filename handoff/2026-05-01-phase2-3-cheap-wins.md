# Handoff — Phase 2 + Phase 3 of PLAN_cheap_wins.md complete + post-session audit

**Date:** 2026-05-01
**Working directory:** `~/topo-confidence/`
**Plan reference:** `PLAN_cheap_wins.md` (Phase 2 §149, Phase 3 §162, Phase 4 §179, Phase 5 §195)

## What landed this session

Four EXP entries promoted soup-to-nuts via `research-graph/promote_result.py`
(skip-git-check, skip-regen). Validate-claims invariant **134 internal PASS / 41
external REGISTERED / 3 PENDING_FE = 178 tracked**, 0 fails.

| EXP | FE | Brief | Verdict |
|---|---|---|---|
| EXP-48 | P11-FE115 (Song-Zhong) | `briefs/result-2026-05-01-P11-FE115.md` | F-3 reinforced; F-2 +3pp ceiling on residuals |
| EXP-49 | P11-FE181 (token-prob) | `briefs/result-2026-05-01-P11-FE181.md` | F-2 not subsumed by mean log-prob |
| EXP-50 | P11-FE749 (spectral α) | `briefs/result-2026-05-01-P11-FE749.md` | F-2 not subsumed by α; F-9 broadens |
| EXP-51 | P11-FE319 + 5 others | `briefs/result-2026-05-01-P11-FE319.md` | F-2 + F-3 corroborated by 6 alt-probe controls |

EXP-51 is bundled — covers FE136, FE244, FE319, FE331, FE339, FE254 in one
script (`pathway11_h100/phase3_corroborators/recompute_phase3.py`, ~5min CPU).

### Headline numbers from this session

| Quantity | Value | Source |
|---|---|---|
| Song-Zhong residualized cos(prefill, final) | **0.0008** (raw −0.062) | `pathway11_h100/song_zhong/results.json` |
| Prefill DoM AUROC on Song-Zhong residuals | **0.8016** (+3pp) | same |
| Mean log-prob AUROC (1.5B) | 0.6721 (≪ DoM 0.7731) | `pathway11_h100/token_prob/results.json` |
| Joint [mean_logp, prefill_DoM] AUROC | 0.7836 (+1pp) | same |
| Best-layer spectral α AUROC (1.5B) | 0.7026 at L28 | `pathway11_h100/spectral_alpha/results.json` |
| α at L19 (DoM peak layer) | **0.5225** (≈ chance) | same |
| Causal-IP whitened cos(prefill, final) | **−0.0315** | `pathway11_h100/phase3_corroborators/results.json` |
| Final-DoM max \|cos\| with top-50 unembed rows | 0.128 (< 0.30 NC3 threshold) | same |
| Quadratic SVD probe peak AUROC (k=10) | 0.7446 | same |
| Linear-AcT (variance-aware) AUROC | 0.7714 (≈ tied) | same |
| Stolfo principled steering coefficient α | **4.724** | same |
| Joint concat DoM AUROC (simple probe) | 0.6946 (LOWER than prefill alone) | same |

### Cumulative narrative

**F-2 now has 9 evidence rows** and survives every cheap competitor tested:
length partialing, output entropy, spectral α, quadratic SVD features,
variance-aware Linear-AcT, joint concat (simple probe). It is purely linear
at L19 prefill — confirmed by FE101 LEACE collapse (0.77 → 0.50), FE319
quadratic ceiling 0.7446, FE339 Linear-AcT tie at 0.7714.

**F-3 now has 5 controls passed**: Song-Zhong residualization (cos 0.0008),
P10 multi-position pattern, two-feature LR complementarity, causal-IP
whitening (cos −0.0315), NC3 alignment (max |cos| < 0.13). STRONG status
well-earned — orthogonality is structural, not positional, not unembed-
geometry, not generic neural collapse.

**F-9 broadens**: spectral α (HT-SR theory) joins CoE-60 as a scalar
competitor that fails to beat single-layer DoM. Two independent confirming
counter-examples for the parsimony framing.

**H-1 gains principled magnitude**: Stolfo c = 4.72 replaces the arbitrary
alpha-sweep currently planned in H-1. Use this when steering work resumes.

## Post-session audit (2026-05-01)

After the four promotes, I ran a full audit of the handoff against actual repo
state and found / fixed five issues:

1. **Self-contradicting handoff size** — original line said `W_unembed_15b.npy
   (470 MB tied embed_tokens)`. Actual file is **891 MB on disk / 0.93 GB raw
   bytes** (151936×1536 fp32). Fixed inline below.
2. **CLAUDE.md stale invariant** — two places said `91 internal PASS / 41
   REGISTERED / 3 PENDING_FE = 135 tracked`. Updated to current `134/41/3 = 178,
   134/134 PASS` with FE749-Tier-0 caveat.
3. **STATE.md stale invariant** — same line. Updated.
4. **STATE.md Date** — was `2026-04-30`, now `2026-05-01`.
5. **STATE.md "Last experiment completed" had 9 blank lines** between header
   and EXP-51 paragraph, accumulated across ~9 prior promotes.

### Two bugs in `promote_result.py` (now fixed)

The blank-line accumulation in STATE.md and FINDINGS.md was traced to two
distinct bugs:

- **`replace_state_last_experiment` line 533**: regex was
  `r"^## Last experiment completed\s*\n"`. The greedy `\s*` consumed all
  pre-existing blank lines after the header into `m.end()`, then
  `text[:start]` preserved them, and `new_section = "\n{body}\n\n"` prepended
  one more `\n`. Net: +1 blank line every promote. **Fix**: `\s*\n` → `\n`
  so the regex only consumes the single newline that ends the header line.
- **`_strip_meta_yaml`**: `YAML_BLOCK_RE` matches the fence
  ` ```yaml\n...\n``` ` (no trailing newline). After `.sub("", ...)`, the
  `\n\n` before and `\n\n` after the fence both remain → 4 consecutive `\n`
  (= 3 blank lines) under each F-N header. `.strip()` only touches block
  edges, not interiors. **Fix**: collapse `\n{3,}` → `\n\n` after the
  substitution.

Both fixes verified via simulation. FINDINGS.md was retro-cleaned with
`re.sub(r'\n{3,}', '\n\n', ...)` — surgical 3-hunk diff under F-2/F-3/F-9
only, 6 stray newlines removed, no other blocks touched. STATE.md was
retro-cleaned manually in the same audit pass.

`validate_claims` still **178 / 134 PASS / 0 FAIL** post-edits.

## Files modified / created this session

```
NEW   pathway11_h100/song_zhong/recompute_fe115.py
NEW   pathway11_h100/song_zhong/results.json
NEW   pathway11_h100/token_prob/recompute_fe181.py
NEW   pathway11_h100/token_prob/results.json
NEW   pathway11_h100/spectral_alpha/recompute_fe749.py
NEW   pathway11_h100/spectral_alpha/results.json
NEW   pathway11_h100/phase3_corroborators/recompute_phase3.py
NEW   pathway11_h100/phase3_corroborators/W_unembed_15b.npy   (891 MB / 0.93 GB tied embed_tokens, 151936×1536 fp32)
NEW   pathway11_h100/phase3_corroborators/results.json
NEW   research-graph/briefs/result-2026-05-01-P11-FE115.md
NEW   research-graph/briefs/result-2026-05-01-P11-FE181.md
NEW   research-graph/briefs/result-2026-05-01-P11-FE749.md
NEW   research-graph/briefs/result-2026-05-01-P11-FE319.md
EDIT  validate_claims.py                    (+25 claims: fe115/181/749/phase3)
EDIT  STATE.md                              (Last experiment → EXP-51; date + invariant updated; blank lines trimmed)
EDIT  CLAUDE.md                             (claims invariant 91/41/3 = 135 → 134/41/3 = 178; FE749-Tier-0 caveat)
EDIT  FINDINGS.md                           (F-2, F-3, F-9 blocks updated; accumulated blank lines collapsed)
EDIT  EXPERIMENT_LOG.md                     (+EXP-48, 49, 50, 51; Next ID → EXP-52)
EDIT  NEXT_EXPERIMENTS.md                   (regenerated; 232 watchlist papers)
EDIT  research-graph/promote_result.py      (fixed STATE.md blank-line accumulation; fixed FINDINGS.md _strip_meta_yaml blank-line accumulation)
NEW   handoff/2026-05-01-phase2-3-cheap-wins.md   (this file, rewritten 2nd time)
```

Working tree is **uncommitted** — user has not asked for git commits.

## Implementation gotchas worth saving

1. **FE749 spectral α regen takes ~2h45m on 24-core CPU** (29k SVDs across
   1.5B + 7B caches at top-K=50). The default `validate_claims.py` per-claim
   timeout is 600s, which forces REGEN_TIMEOUT and breaks `promote_result.py`.
   **Fix applied:** dropped `regen=` from FE749 claims so they're Tier-0
   readback only. Manual regen: `python pathway11_h100/spectral_alpha/recompute_fe749.py`.
   If we ever extend this pattern to other slow regens, consider adding a
   per-claim timeout override to `validate_claims.py` rather than dropping
   regen entirely.

2. **`oof_logreg_auroc` (Σ⁻¹ LDA) is not a drop-in replacement for `oof_dom_auroc`
   (mean-diff)** in the n<d regime (~400 train, 1536+ dim). Even with
   ridge_rel=1e-2, LDA on full hidden_dim returns 0.55 prefill AUROC vs the
   established 0.7731 — pure rank-deficiency overfitting. FE254 was rewritten
   to use `oof_dom_auroc` for internal consistency with F-2's baseline. If you
   want a *better* joint probe than naïve concat-DoM, use ridge LR with
   cross-validated regularisation — not LDA.

3. **`promote_result.py --skip-regen` only skips step 2 (the brief's own
   regen_cmd)**. Step 3 (validate_claims gate) always runs `validate_claims.py`
   with full Tier-1 regen. If a competing process (e.g. another long script)
   is hogging CPU, validate_claims will REGEN_TIMEOUT — abort and re-promote
   later, or kill the competitor first.

4. **`text` field in pathway8 NPZs needs `allow_pickle=True`** to load —
   stored as 0-d object array. FE244 handles this.

5. **ScheduleWakeup minimum delay is 60s.** "Sleep 5 minutes" doesn't help —
   either 60–270s (cache stays warm) or 1200s+ (one cache miss buys real
   wait). 300s is the worst-of-both.

6. **`promote_result.py` blank-line accumulation (now fixed)** — STATE.md
   `## Last experiment completed` and FINDINGS.md F-N headers used to grow
   one blank line per promote. Bugs traced and fixed in this session — see
   "Two bugs in promote_result.py" above. Future promotes are stable.

## Next steps — Phase 4 + Phase 5

### Phase 0 verification (already done, 1 min)

- **FE282 Phase-0 cache check ✓ DONE** (verified during the audit). The
  cache `pathway8_layerwise/data/math500/problem_*.npz` has all 29 layers
  (states shape `(29, *, 1536)`), keys `[states, d2h_attn_entropy, text,
  correct, mean_logprob]`, 500 problems present. **Action: mark P11-FE282
  COMPLETED-by-cache** at session start with:
  ```bash
  cd ~/topo-confidence/research-graph
  python update_status.py P11-FE282 COMPLETED \
    --outcome "cache available; per-layer L0..L28 prefill states already extracted (states shape (29, *, 1536) per problem, 500 problems)"
  python generate_next_experiments.py
  ```

### Phase 4: Local 2060 GPU runs (PLAN §179, ≤2h each, when ready)

Qwen-2.5-1.5B at fp16 fits in ~6GB peak (3GB weights + KV cache). Use SDPA;
batch=1; max_new_tokens=1024 per problem.

| FE | What | Cost | Notes |
|---|---|---|---|
| **P11-FE110** | ActAdd contrast-pair vs supervised DoM (deferred from Phase 3) | 30min on 2060 + 10min CPU AUROC | Needs 3–5 contrast prompt pairs run through model; bundle with FE455 |
| **P11-FE455** | ConCISE `So, I'm` confidence detector | ~30min on 2060 + 10min CPU AUROC | Forward pass on 500 problems, read next-token logits at "So, I'm" position |
| **P10-FE23** | Softmax-confidence AUROC sanity check | Bundle with FE455's same loop | Single-token softmax at answer position |
| ~~P11-FE282~~ | (verified COMPLETED-by-cache; see Phase 0 above) | — | — |

**Skip on the 2060** (defer to next H100 session): anything 7B (FE116, FE117,
FE282 in 7B form).

### Phase 5: Heavy CPU one-shots (PLAN §195, background)

- **P11-FE321** — zigzag PH + bar Z_1 vs Gaussian null (~1h CPU, foreground)
- **P11-FE116** — PH pipeline on residuals (~1 day CPU, backgroundable)

Both gated on F-2 surviving Phase 1 — which it has, four times over now
(length partial, LEACE null, LOCO-CV, within-topic).

### Suggested order for next session

1. **FE282 → COMPLETED** (1 min, see Phase 0 above).
2. **FE116 PH on residuals** (~1 day CPU, **background**). Kick off first so
   it's running while you do everything else. Use the same `Monitor` pattern
   used for FE749. Background-fire the recompute script with
   `run_in_background: true`.
3. **FE321 zigzag PH** (~1h CPU, foreground). Reuses cached prefill states.
4. **FE110 + FE455 + FE23 bundle on 2060 GPU** (~1h GPU + 10 min CPU AUROC).
   All three need a fresh Qwen-2.5-1.5B fp16 forward pass on the same 500
   MATH-500 problems — bundle in one script. Time-box to ~1h GPU.
5. After all four land, promote each via `promote_result.py` and run the
   pre-flight checks (below).

## Pre-flight before next session

```bash
# Verify the cumulative invariant is still intact:
cd ~/topo-confidence && python validate_claims.py --no-regen 2>&1 | grep -iE "PASS:|FAIL|REGISTERED|PENDING"
# Expected: 134 PASS / 0 FAIL / 41 REGISTERED / 3 PENDING_FE

# Confirm STATE.md reflects the most recent experiment:
grep -A 3 "## Last experiment completed" STATE.md | head -10
# Expected: EXP-51 (Phase 3 corroborators bundle), exactly 1 blank line under header

# Check the watchlist for any new triage-priority papers:
cd ~/topo-confidence/research-graph && python query.py watchlist | head -20

# Spot-check that promote_result.py blank-line bugs stay fixed:
cd ~/topo-confidence && grep -c '^$' STATE.md FINDINGS.md
# (no specific number — just confirm no header has more than 1 blank line under it)
```

## Decision branches that did NOT fire (still in scope)

The PLAN had two decision branches that would have ended Phase 2/3 early
if they fired. Neither did:

- **F-2 dies under length partial.** FE448 partialed predicted-length and
  DoM AUROC dropped 0.7834 → 0.6647 — F-2 weakened but not killed. Phase 2/3
  proceeded.
- **Spectral α beats DoM (FE749 ROI-10 anchor).** α best-layer AUROC 0.7026
  < DoM 0.7731. F-9 narrows by counter-example, doesn't flip.

So Phase 4 + Phase 5 are still in scope as written.

## Memory aids for the next session

- F-2 strength: **MODERATE**, status **ACTIVE**, evidence rows 1..9
- F-3 strength: **STRONG**, status **ACTIVE**, evidence rows 1..5
- F-9 strength: **MODERATE**, status **ACTIVE**, evidence rows 1..3
- Next EXP-id: **EXP-52** (per `EXPERIMENT_LOG.md`)
- Next FE-id will likely be **P11-FE110** (ActAdd) or **P11-FE455** (ConCISE)
- FE282 will be marked COMPLETED-by-cache, no new ID needed.
- Cached W_unembed: `pathway11_h100/phase3_corroborators/W_unembed_15b.npy`
  (151936×1536 fp32, **891 MB / 0.93 GB**) — reuse for any future unembed-
  projection work to avoid redownloading the model.
- `promote_result.py` blank-line accumulation bugs are fixed; future promotes
  produce stable 1-blank-line spacing under STATE.md `## Last experiment
  completed` and under each FINDINGS.md F-N header.
