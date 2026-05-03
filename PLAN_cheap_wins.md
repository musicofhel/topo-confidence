# PLAN — Cheap Wins, Cached-Data Phase

_Drafted 2026-04-29. Status: NOT YET EXECUTED. Do Phase -1 before anything else._

## Goal

Cash in on the 220-paper triage by running every CPU-feasible / sub-2h / cached-data
experiment that bears on F-2 (prefill L19 DoM, AUROC 0.7731 — load-bearing) and
its neighbours. No fresh H100 generation. Surface refutations early, integrate
each result through the existing claims/graph rails (extending where needed).

## Constraints

- **Pure CPU** (any duration), or **<2h on the local RTX 2060 Super 8GB** at fp16.
- Qwen-2.5-1.5B fits at fp16 (~3GB + KV cache). **Qwen-2.5-7B does NOT** — skip
  any 7B-required FE; defer to next H100 session.
- Must reuse cached NPZs / generations from the RunPod sessions. No fresh forward
  passes over MATH-500 unless the cache is provably missing AND fits the 2060.
- Each landed result must update `validate_claims.py` (Tier-0 + Tier-1 regen)
  and the Neo4j graph (`update_status.py`). The 91-internal-PASS invariant
  stays load-bearing.

---

## Phase -1: Re-validate architecture (RUN BEFORE EXECUTION)

The pipeline pieces have all been touched in the last week (validate_claims `kind`
field, claims gate, promote-brief auto-promote, status reset migration). Confirm
each is in a known-good state before we start producing new results that depend
on them.

```bash
# 1. Claims registry — 135 tracked, 91 PASS / 41 REGISTERED / 3 PENDING_FE
cd ~/topo-confidence
python validate_claims.py > /tmp/validate_baseline.txt
tail -20 /tmp/validate_baseline.txt
# EXPECT: exit 0, "91/91 internal PASS, 41 REGISTERED, 3 PENDING_FE", REGEN section "68/68 REGEN_PASS"

# 2. Tier-1 regen scripts still work (sanity sample)
python pathway11_h100/multi_signal_oracle/recompute_extras.py
python pathway11_h100/prefill_inversion/recompute_pr_ratios.py
# EXPECT: key=value lines, no MISSING_REGEN_INPUT, no Python errors

# 3. Neo4j up + graph in expected state
cd ~/topo-confidence/research-graph && docker compose ps
python query.py status-report
# EXPECT: Neo4j healthy on bolt://localhost:7688
# EXPECT: 221 papers / 221 graphed / 0 pending_triage / 0 candidate / 0 rejected

# 4. Queue regen is deterministic
python generate_next_experiments.py
git diff --stat ../NEXT_EXPERIMENTS.md
# EXPECT: zero-line diff (queue already current) OR small re-ordering only

# 5. promote_brief.py imports clean (no syntax errors after recent edits)
python -c "import promote_brief; print('ok')"

# 6. update_status.py argparse OK
python update_status.py --help > /dev/null && echo "ok"

# 7. Working tree clean — no stray edits before kickoff
cd ~/topo-confidence && git status --short
# EXPECT: empty OR only this PLAN_cheap_wins.md
```

**Stop conditions**:
- Any of (1)–(6) fails → fix root cause before any FE work. Don't paper over.
- (7) shows unexpected modifications → triage them; could be a teammate's WIP.
- Tier-1 regen fails → `pathway11_h100/...` cache may have been mutated;
  investigate before trusting any new results.

If all 7 pass, write `Phase -1 PASS at <date>` to the bottom of this plan and
proceed.

---

## Phase 0: Cache verification (15 min)

Two unverified assumptions from the queue scan. Don't write any new scripts
until these resolve.

```bash
cd ~/topo-confidence

# (a) Are sequence-likelihoods cached for FE181?
python -c "
import numpy as np
from pathlib import Path
gens = Path('pathway11_h100/prefill_gated_compute/generations.npz')
if gens.exists():
    d = np.load(gens)
    print('keys:', list(d.keys()))
else:
    print('MISSING:', gens)
"
# Looking for: 'logprobs', 'token_logp', 'mean_logp', 'sum_logp', or similar.
# If absent: FE181 needs a fresh forward pass on the 2060 (~1h, fits at fp16).

# (b) Per-layer prefill cache — L19 only, or all layers?
ls pathway11_h100/prefill_inversion/cache/
python -c "
import numpy as np
d = np.load('pathway11_h100/prefill_inversion/cache/m15b_prefill.npz')
for k in d.files:
    print(k, d[k].shape if hasattr(d[k], 'shape') else type(d[k]))
"
# If only 'prefill' / 'final_tok' (L19): FE145 (per-layer sweep) and FE282 (layer specificity)
# need fresh extraction. Move them to Phase 4 (GPU) or skip on the 2060.
```

Outcome captured in this file as `## Phase 0 results` block before continuing.

---

## Phase 1: Sub-1h CPU sanity battery (one evening, ~3h total)

Goal: confirm or shake F-2 with the cheapest probes available. Each FE produces
a single result JSON + a `Claim(...)` entry + an updated graph status.

| Order | FE | Cost | Claim it tests |
|---|---|---|---|
| 1 | **P11-FE719** | 20min CPU | LOCO-CV by MATH-500 subject — leakage check on F-2 |
| 2 | **P11-FE448** | 1h CPU | Length-partial-correlation control on F-2 |
| 3 | **P11-FE145** | 20min CPU | Per-layer AUROC sweep — confirms L19 is peak (skip if cache is L19-only) |
| 4 | **P11-FE299** | 20min CPU | Topic-stratified AUROC — sister to FE719 |
| 5 | **P11-FE101** | 20min CPU | LEACE linear erasure null |

Per-FE protocol:
1. Write `pathway11_h100/<area>/recompute_<feid>.py` (deterministic, prints
   `key=value` lines, soft-skips with `MISSING_REGEN_INPUT` if cache absent).
2. Add `Claim(...)` entries to `validate_claims.py` (kind="internal", regen_cmd
   set).
3. Run the recompute → save result JSON.
4. `python validate_claims.py` → confirm new entries PASS + REGEN_PASS.
5. `promote_result.py P11-FE<id> --result <path> --outcome "..." [--finding-update F-2:WEAKENED]`
   (see "Bookkeeping" below — this script doesn't exist yet, build in Phase -0.5).
6. Commit (one commit per FE, or one per phase — see decision branch).

**Phase-1 decision branch**: if ≥3 of these soften F-2 below AUROC 0.70:
- Demote F-2 in `STATE.md` and `FINDINGS.md` (status REVISED).
- Skip Phase 3 (causal corroborators are uninteresting if F-2 is already a
  partial artifact).
- Jump to Phase 2 + Phase 4 to look for a stronger signal.

Otherwise: F-2 confirmed under cheap controls → proceed to Phase 2.

---

## Phase 2: ROI-10 anchors (one weekend day, ~3.5h CPU)

| FE | Cost | What lands |
|---|---|---|
| **P11-FE115** | 2h CPU | Song-Zhong pos/ctx mean decomposition. Big update if `cos(prefill_DoM_resid, final_DoM_resid)` jumps from 0.046 → >0.30 — F-3's "two circuits" framing collapses to a positional artifact. |
| **P11-FE749** | 30min CPU | Spectral-α single-scalar AUROC vs DoM. Paper claims 1.000 on Qwen-7B. If α matches/beats DoM on 1.5B, F-9 parsimony narrows. |
| **P11-FE181** | 1h CPU (or ~1h on 2060 if logprobs absent) | Mean/min/product token-prob AUROC head-to-head with DoM. The "is DoM just restating output entropy?" test. |

Run serially, not parallel — share CPU cores. Each goes through the same per-FE
protocol as Phase 1.

---

## Phase 3: Causal corroborators (filler day, ~2.5h CPU total bundle)

Run only if F-2 survived Phase 1. Bundle into one script run:

- **P11-FE110** (20min) — ActAdd contrast-pair vs supervised DoM
- **P11-FE136** (20min) — causal-inner-product whitening for F-3 orthogonality
- **P11-FE244** (10min, weights-only — no NPZ needed) — W_unembed alignment
- **P11-FE319** (10min) — quadratic probe on top-k SVD directions
- **P11-FE331** (10min) — Stolfo principled steering coefficient (gates H-1)
- **P11-FE339** (20min) — Linear-AcT vs Mean-AcT
- **P11-FE254** (30min) — joint prefill+final concat probe

Output: one comparison table appended to PERSPECTIVES.md or a new section in
FINDINGS.md.

---

## Phase 4: Local 2060 GPU runs (≤2h each, when ready)

Qwen-2.5-1.5B at fp16 fits with ~6GB peak (3GB weights + KV cache). Use SDPA
or flash-attn 2; batch=1; max_new_tokens=1024 per problem.

| FE | What | Notes |
|---|---|---|
| **P11-FE455** | ConCISE `So, I'm` confidence detector — forward pass on 500 problems, read next-token logits | ~30min on 2060 + 10min CPU AUROC |
| **P10-FE23** | Softmax-confidence AUROC sanity check | Bundle the forward pass with FE455 — same loop |
| **P11-FE282** *(partial)* | Per-layer feature extraction (L0…L27, prefill only). ~1.5h on 2060 at bf16 batch=1 | Skip if Phase 0 (b) shows all layers already cached |

**Skip on the 2060**: anything 7B (FE116, FE117, FE282 in 7B form). Defer to
next H100 session.

---

## Phase 5: Heavy CPU one-shots (background)

Run only after Phase 1 has set direction. If F-2 dies, FE116 becomes moot.

- **P11-FE116** — PH pipeline on residuals (1 day CPU, backgroundable)
- **P11-FE321** — zigzag PH + bar Z_1 vs Gaussian null (1h CPU)

---

## Bookkeeping — soup-to-nuts integration

`research-graph/promote_result.py` is **BUILT** (2026-04-30). It mirrors
`promote_brief.py`'s "human writes a structured brief, machine places blocks
deterministically" pattern. The author writes a brief at
`research-graph/briefs/result-YYYY-MM-DD-<fe-id>.md`, and the script:

1. Parses required sections (`## FE`, `## EXPERIMENT_LOG entry`,
   `## STATE.md last-experiment update`, `## FINDINGS.md updates`,
   `## New claims`, `## Sources`); refuses on missing/filler.
2. Verifies the result JSON exists and contains the keys declared in the
   `## FE` YAML's `result_json_keys`.
3. Optionally re-runs `regen_cmd`; aborts on FAIL.
4. Claims gate: `validate_claims.py.mtime > brief.mtime` AND PASS
   (same gate as `promote_brief.py:322`).
5. Refuses if `git status` dirty outside the expected file set
   (validate_claims.py, FINDINGS.md, STATE.md, EXPERIMENT_LOG.md,
   NEXT_EXPERIMENTS.md, the result_json, the regen script, the brief).
6. Neo4j: `update_status.py <fe-id> COMPLETED --outcome ...`; for each F-N
   meta-yaml block, MERGE/SET strength/status/counterargument/overturned_by
   on the `:Finding`, plus `(:Experiment)-[:PRODUCED]->(:Finding)` edges
   for declared `add_evidence`.
7. **Auto-writes** EXPERIMENT_LOG.md (append + bump Next ID), STATE.md
   ("## Last experiment completed" body replace), FINDINGS.md (in-place
   block replace via `_f_block_bounds`, or append before
   `## Honorable mentions` for new F-N + bump Next ID).
8. Runs `generate_next_experiments.py`.

PERSPECTIVES.md is intentionally never touched.

Smoke-tested 2026-04-30 with
`briefs/result-2026-04-30-P11-FE115.md --dry-run --skip-git-check`: all 10
steps parsed and dry-run cleanly. Brief file is marked DO NOT PROMOTE in its
title.

Usage:
```bash
python promote_result.py briefs/result-<date>-<fe-id>.md --dry-run
python promote_result.py briefs/result-<date>-<fe-id>.md
python promote_result.py briefs/result-<date>-<fe-id>.md --skip-regen
python promote_result.py briefs/result-<date>-<fe-id>.md --skip-git-check
```

Brief template (copy from `briefs/result-2026-04-30-P11-FE115.md`):
```yaml
# in ## FE section
fe_id: P11-FE<N>          # must match FE id in NEXT_EXPERIMENTS.md
status: COMPLETED         # or ABANDONED
outcome: "<one sentence>"
result_json: pathway11_h100/.../results.json
result_json_keys: [auroc_oof, ...]
regen_cmd: "python pathway11_h100/.../recompute.py"  # optional
```

For each `### F-N:` block under `## FINDINGS.md updates`, an optional
leading ```yaml fence carries the meta:
```yaml
strength: WEAKENED        # STRONG | MODERATE | PRELIMINARY
status: ACTIVE            # ACTIVE | WEAKENED | INVALIDATED | SUPERSEDED
add_evidence: [P11-E43]
counterargument: "..."
overturned_by: "..."
```
The yaml is consumed for graph updates; the prose body of the F-N block is
what lands in FINDINGS.md.

---

## Proposed kickoff sequence

When the user says go:

1. Phase -1 re-validation (15 min). Block if anything fails.
2. Phase 0 cache verification (15 min). Capture results in this file.
3. ~~Build `promote_result.py`~~ — DONE 2026-04-30; smoke-tested dry-run.
4. Phase 1 FE719 — first end-to-end test of the new pipeline (~30 min:
   recompute script + Claim entry + run + promote).
5. Continue Phase 1 (FE448, optionally FE145, FE299, FE101).
6. Phase 1 decision branch.
7. Phase 2 / Phase 3 / Phase 4 as direction warrants.
8. Phase 5 backgrounded.

Estimated total wall-clock: 2 evenings + 1 weekend day if F-2 holds; 1 evening
+ pivot-to-something-else if Phase 1 nukes F-2.

---

## Status

- 2026-04-29: Plan drafted. NOT EXECUTED.
- 2026-04-30: `promote_result.py` built and dry-run smoke-tested. Spec
  rewritten to match the soup-to-nuts pattern (auto-writes FINDINGS.md /
  STATE.md / EXPERIMENT_LOG.md from a structured brief), mirroring
  `promote_brief.py`. Smoke-test brief at
  `research-graph/briefs/result-2026-04-30-P11-FE115.md` (DO NOT PROMOTE).
- **Phase -1 PASS at 2026-05-01 01:25 UTC.** All 7 checks green:
  (1) `validate_claims.py` exit 0 — `91 PASS / 0 FAIL / 41 REGISTERED /
  3 PENDING_FE / 68 REGEN_PASS / 0 REGEN_FAIL`. (2) Tier-1 regen scripts
  `multi_signal_oracle/recompute_extras.py` and
  `prefill_inversion/recompute_pr_ratios.py` both produced expected
  key=value output, no errors. (3) Neo4j healthy on bolt://localhost:7688
  (Up 4 days); 221 papers `status='graphed'`, 0 `pending_triage`, 0
  `candidate`, 0 `rejected`. (4) `generate_next_experiments.py` is
  deterministic — diff against committed `NEXT_EXPERIMENTS.md` is the
  generation-timestamp line only. (5) `import promote_brief` clean. (6)
  `update_status.py --help` returns 0. (7) Working tree shows only the
  expected files-in-flight from the previous session: `STATE.md` (M),
  `PLAN_cheap_wins.md`, `research-graph/briefs/result-2026-04-30-P11-FE115.md`,
  `research-graph/promote_result.py` (all `??`). No teammate WIP.
  **Notes (non-blocking):** (a) `:Finding` count is 14 ACTIVE, not 10 —
  F-11..F-14 landed during the max-depth re-triage; CLAUDE.md headline is
  stale. (b) 11 stub `:Paper` nodes have `status=NULL`; each is the
  target of exactly one `TRIGGERED_BY` edge from a `:FutureExperiment` —
  these are reference stubs, not triage-queue items, and are excluded
  from `query.py pending`.
- **Phase 0 results at 2026-05-01 01:35 UTC.** Both assumptions resolved
  via the per-problem NPZ cache (not the aggregate `generations.npz`,
  which does **not** exist):
  - **(a) sequence-likelihoods cached as `mean_logprob`.** Every
    `pathway8_layerwise/data/math500/problem_*.npz` (1.5B) and
    `pathway11_h100/data/math500_7b/problem_*.npz` (7B) carries a scalar
    `mean_logprob` field per problem. **FE181 (token-prob baseline)
    needs no fresh forward pass** — just aggregate the 500 scalars and
    compute AUROC vs `correct`.
  - **(b) per-layer cache: ALL 29 layers, both models.** Each per-problem
    NPZ has `states` with shape `(29, T, hidden_dim)` (T = sequence
    length, varies per problem; hidden_dim = 1536 / 3584). The
    `prefill_inversion/cache/m15b_prefill.npz` aggregate is L19-only,
    but the per-problem source files contain L0..L28 inclusive.
    **FE145 (per-layer DoM sweep) and FE282 (layer specificity) are
    CPU-feasible** — Phase 4 GPU is no longer needed for these.
- **Phase 1 — COMPLETE 2026-05-01.** All 5 cheap controls done end-to-end
  through `promote_result.py`. Claims invariant grew from 91/41/3 →
  **109/41/3 = 153 tracked, 109/109 internal PASS, 86/86 REGEN_PASS**.
  EXP-43..EXP-47 logged. F-2 graph state: `strength=MODERATE,
  status=ACTIVE, evidence=['P11-E3','P11-E43','P11-E44','P11-E45',
  'P11-E46','P11-E47']`.

  | # | FE | Result | F-2 verdict |
  |---|---|---|---|
  | 1 | FE719 LOCO-CV | mean 0.7427 (worst 0.6032 NT, best 0.9000 Geo) | NOT refuted (≥ 0.65) |
  | 2 | FE448 length partial-corr | residual 0.6647 (length R²≈1.0, length-alone AUROC 0.7986) | gray zone — STRONG → MODERATE |
  | 3 | FE145 per-layer sweep | peak L21 0.7718, L19 0.7705 (gap 0.0013) | layer choice CONFIRMED |
  | 4 | FE299 within-topic | mean 0.7143 (worst 0.6000 NT) | NOT a topic-detector |
  | 5 | FE101 LEACE null | raw 0.7705 → erased 0.5000 (perfect collapse) | F-2 IS purely linear |

  **Phase 1 decision-branch counter: 1 of 5 below AUROC 0.70** (FE448
  residualized only). Below the 3-of-5 demotion threshold — Phase 3
  (causal corroborators) stays in scope. Cumulative narrative: F-2 is a
  real, purely linear correctness signal, but ~⅔ of its AUROC is
  length-explainable. Number Theory is the consistent weak fold across
  both topic controls. The dominant counterargument is now FE448's
  length-confound, not topic-leakage or non-linearity.

  **Per-FE artefacts:**
  - `pathway11_h100/loco_subject/{recompute_fe719.py, results.json}`
  - `pathway11_h100/length_partial/{recompute_fe448.py, results.json}`
  - `pathway11_h100/per_layer_sweep/{recompute_fe145.py, results.json}`
  - `pathway11_h100/within_topic/{recompute_fe299.py, results.json}`
  - `pathway11_h100/leace_erasure/{recompute_fe101.py, results.json}`
  - `research-graph/briefs/result-2026-05-01-P11-FE{719,448,145,299,101}.md`

- **Phase 2 — pending (3 ROI-10 anchors).** FE115 (Song-Zhong pos/ctx
  decomp), FE749 (spectral α), FE181 (token-prob baseline; Phase 0
  confirmed `mean_logprob` cached in per-problem NPZs).
- **Phase 3 — pending** (causal corroborators). Decision-branch did NOT
  fire, so this stays in scope.
- _(Phase 4/5 status)_ — pending
