# SPEC v8 — Raise the Ceiling

**Status: PRE-REGISTERED 2026-06-12 (not yet executed).**
Successor to SPEC_v7.md (executed; logged EXP-84, spec-internal labels
EXP-84..89). Self-contained: a fresh agent can execute this spec without the
authoring conversation.

---

## Context & north-star

v7 shipped `confgate` and then established the **general closure**: at matched
token-FLOP cost, NO probe (token-level features, K-consistency, P(True),
verbalized, spectral-α, PRM-7B, cost-aware router) beats spending the same
tokens on escalation. H-I held on dev and confirmatory T5; H-M refuted the
router (−0.40pp, p=0.68). The binding constraint is **rescue density**: of the
257 base-model failures on MATH-500, only 136 are rescuable by the current
escalation target — 121 (47%) are unrescuable by Qwen2.5-7B-Instruct at any
gate quality. Ranking is solved; the ceiling is not.

**v8's inversion of the question:** stop trying to beat the gate; raise what
the gate gates into. Three levers remain, none of which v6/v7 touched:

1. **The escalation target.** The 4.7× cost is fixed by parameter count, but
   the *model occupying that slot* is a free choice. A math-specialized 7B at
   identical cost moves the oracle ceiling directly.
2. **The base model.** The 1.5B slot is equally free. Reasoning-distilled and
   math-specialized 1.5Bs report 75–84% MATH-500 — nominally above the entire
   current cascade's oracle ceiling (0.758) — but at unmeasured token cost
   (long CoT) and with unknown gate validity. Honest matched-cost placement
   on the frontier has never been done here.
3. **The base model's weights.** The durable version of lever 2: use the
   confidence stack itself (7B free-gate scores, the v7 H-L3 zero-label
   machinery, precision 0.951) to *curate distillation data* and SFT the
   generic base. This is the project goal — improve small models — executed
   with the tools this project actually validated.

This aligns with the regenerated queue: P11-FE1214 (ROI 9, distill strong-7B →
1.5B) and P11-FE5 (ROI 9, R1-Distill-Qwen-1.5B comparison). v8 executes the
**behavioral core** of both; their geometric sub-questions (DoM direction
preservation, breathing) sit under refuted premises (`geometry-portable-signal`,
`dom-causal-lever`) and are NOT revived here — the moot sweep adjudicates them
at promote time.

**North-star metric:** matched-cost accuracy on MATH-500 at the v7 operating
point (budget = c0 + 0.5·c_esc = 673.37 + 1547.34 = **2220.71 token-FLOPs per
problem**; 1 unit = one 1.5B token, 7B token = 4.7). The number to beat is the
**free-gate cascade 0.648** (oracle 0.758, random-mix hull 0.609). Deliverable
is updated `confgate` pins + a documented data-engine recipe, not a paper.

**Execution parameters (standing policy; pod budget needs user sign-off
before any pod spins up):**
- ALL generation and training on RunPod H100 SXM (no local GPU ever, even
  forward-only — feedback-no-local-gpu-even-rescoring). Local = CPU analysis.
- Pods have no network volume: pull result JSONs after EACH task; REMOVE pod.
- Token features captured at **generation time** (v7 D-4 house method,
  `genscore`-style: output_scores → n_gen_tokens, mean_logprob), never by
  decode→re-encode rescoring.
- Estimated total: **~8–10 H100 hrs** across two pod sessions (v7 used ~2.3
  of its 8–10 authorized).

---

## Negative priors (pre-registered, so wins are surprising and nulls are cheap)

| Arm | Prior | Source |
|---|---|---|
| Math-7B target swap (H-N) | **strong FOR** | Qwen2.5-Math-7B-Instruct reports ~0.83 MATH-500 vs current target's measured 0.732; identical 4.7× cost. The interesting outcome is the SIZE of the gain and what fraction of the 121 unrescuables it claims. |
| family-diverse second target (H-O) | weak | Same-benchmark-trained models correlate errors; Mathstral-7B reports ~0.56 MATH — weaker overall, so its value is only in DISJOINT rescues. |
| reasoning/specialist base swap (H-P) | FOR on accuracy, **open on cost** | R1-Distill-Qwen-1.5B reports 83.9% MATH-500 but via long CoT — if mean CoT > ~2,221 tokens it exceeds the matched budget by itself. Qwen2.5-Math-1.5B-Instruct (~0.75 reported) is short-form, likely cheap. Possible benchmark contamination in both; treated as descriptive caveat, comparison is still honest (same eval for all systems). |
| gate validity on reasoning base (H-Q) | genuinely open | Length–correctness coupling may invert or sharpen under long-CoT (overthinking literature both ways); never measured in this harness. This is the load-bearing unknown for re-anchoring confgate. |
| confidence-curated distillation (H-R) | distillation works (trivially); **curation increment open** | SFT on teacher traces improving a 1.5B is not in question. The pre-registered question is whether a ZERO-LABEL confidence filter (7B free-gate score, per v7 H-L3) matches ground-truth filtering and beats unfiltered at matched trace count. Small-N curated sets sufficing is literature-supported (DED, 2508.09883: 0.8k curated examples reach SOTA). Known confound from the data-selection literature ("Step Length Confounding in LLM Reasoning Data Selection", link-forge local PDF): selection signals built on token statistics systematically prefer trace LENGTH over trace quality — and our gate's length feature makes this a first-order risk, hence arm (d) below. |

---

## Verified artifact facts the spec depends on (Phase 0 re-pins these)

1. `results/v7_phase4_router.json`: cascade 0.648 / oracle 0.758 / hull 0.609
   at esc-0.5; c0 = 673.37, c_esc = 3094.6868, budget_extra = 1547.3434.
2. Rescue density recomputable from pinned caches: y_1.5B has 257 failures,
   y_7B rescues 136, leaves 121 (47.1%) unrescuable. Phase 0 recomputes and
   pins these three integers as anchors.
3. Models in harness: base `Qwen/Qwen2.5-1.5B-Instruct` (0.486 greedy
   MATH-500), target `Qwen/Qwen2.5-7B-Instruct` (0.732). Grader = the pinned
   harness grader (k8_lib extraction + equivalence); **no new grading logic
   anywhere in v8** — any answer-format issue is handled by pre-registered
   parse rules (below), not grader edits.
4. Free gate recipe frozen since v6: StandardScaler + LogisticRegression
   (C=1.0, max_iter=2000) on [n_gen_tokens, mean_logprob], frozen 5-fold OOF
   (`metrics.py` folds). Dev AUROC 0.849 (MATH 1.5B), T5 0.8083/0.8422/0.8357.
5. v7 H-L3: 7B-agreement pseudo-labels precision 0.951 → valid cross-scale
   conformal certs with zero human labels. v8's H-R filter is the gate-score
   analogue of this machinery.
6. MATH-500 is the PRM800K test subset of Hendrycks MATH; the MATH **train**
   split (7.5k) is disjoint by construction → legal SFT pool.
7. Claims invariant at v8 start: **245/245 internal PASS** (`validate_claims.py`),
   Tier-1 regen 123/123. Never regress. Footguns: full regen mutates
   `pathway11_h100/gpu_bundle/results.json` elapsed_seconds (git checkout
   before commit); EXPERIMENT_LOG Next ID is **EXP-85** — the result brief logs
   as EXP-85 regardless of spec-internal labels; hand-created FE nodes need
   `roi_score` set.

---

## Selection firewall (v8)

- **Dev surface** (free to iterate): MATH-500 with existing pinned caches and
  frozen 5-fold splits; all Phase-1 frontier arithmetic.
- **Frozen before generation:** every Phase-1 model choice, prompt format,
  max_new_tokens, and parse rule is fixed in this spec — Phase 1 is
  single-shot per arm (one greedy run per model, no re-runs after seeing
  accuracy).
- **Gate G1 (before any Phase-2 GPU minute):** the full SFT recipe — teacher
  identity (a pre-registered *conditional* on H-N, not a post-hoc choice),
  train pool, filter operating point, LoRA hyperparameters, seed — is frozen
  in the G1 freeze block appended to this file.
- **Gate G2 (Phase-2 evaluation):** exactly ONE evaluation per checkpoint on
  MATH-500 and ONE on BBH-750. No checkpoint selection against eval sets:
  fixed recipe, final checkpoint, single seed. If a run crashes mid-training
  it may be restarted from scratch (restart logged); a COMPLETED checkpoint
  is never re-trained.
- All hypothesis thresholds below are pre-registered; deviations get D-n
  blocks (v7 protocol) written BEFORE the affected cell is adjudicated.

---

## Hypotheses & decision rules

Cost convention everywhere: token-FLOP units, 1 = one 1.5B token; 7B = 4.7;
Mathstral-7B = 4.7 (7.2B ≈ same bucket); 1.5B-class base swaps (R1-Distill,
Math-1.5B) = 1.0/token; measured token counts, not assumed.
Statistics: paired bootstrap B=2000 over problems for accuracy deltas;
DeLong for AUROC.

**H-N (target swap raises the ceiling).** Replace the escalation target with
`Qwen/Qwen2.5-Math-7B-Instruct` (identical 4.7× cost; same gate, same
escalation set, same esc-0.5 budget).
- CONFIRM: matched-cost cascade accuracy ≥ 0.648 + 3pp with p<0.05.
- REFUTE: < +1pp.
- Always report: new oracle ceiling, rescues claimed among the 121
  unrescuables, and overlap matrix vs the incumbent target.
- Teacher rule for Phase 2: Math-7B **iff CONFIRMED**; any other H-N outcome
  (refuted OR the +1pp..+3pp inconclusive zone) → incumbent Qwen2.5-7B. No
  judgment call at G1.
- If CONFIRMED → Math-7B also becomes the pinned confgate target.

**H-O (family diversity buys disjoint rescues).** `mistralai/Mathstral-7B-v0.1`
(ungated, Mistral family) greedy on MATH-500.
- CONFIRM: rescues ≥ 20% (≥25) of the 121 Qwen-7B-unrescuable problems.
- REFUTE: < 10%.
- Confirmation does NOT add a router (H-M closed routers over probes; a
  router over TARGETS would be a new FE filed at promote, not executed in v8).
  Refutation closes portfolio escalation and strengthens the "ceiling is
  problem difficulty, not family quirks" reading.

**H-P (better bases dominate the frontier at honest cost).** Two base-swap
arms, greedy on MATH-500 with gen-time feature capture:
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (max_new_tokens 4096; answer =
last \boxed after the think block, else final-number fallback; unparseable =
wrong, parse-failure rate is its own column) and
`Qwen/Qwen2.5-Math-1.5B-Instruct` (max_new_tokens 1024, standard parse).
Prompts: the harness MATH instruction through each model's own chat template.
**Pre-registered fallback for a KNOWN failure mode:** R1-Distill's model card
warns greedy decoding causes repetition loops; primary run is still greedy
(house determinism convention), but if >20% of outputs hit max_new_tokens
without a parseable answer, run ONE fallback at T=0.6 seed 9999, adjudicate
H-P/H-Q on the fallback, report both, and log it as a D-n deviation — decided
here, not mid-run.
- CONFIRM (per arm): accuracy > 0.648 (paired bootstrap p<0.05) at measured
  mean cost ≤ 2220.71 token-FLOPs/problem (the matched budget).
- REFUTE (per arm): accuracy ≤ 0.648 at that budget, or cost above budget
  with accuracy below the cost-equivalent cascade point (read off the v7
  frontier).
- Report reported-vs-measured accuracy gap as a contamination caveat, not an
  adjudication input.

**H-Q (the free gate survives better bases).** Fit the frozen free-gate
recipe per H-P arm (5-fold OOF on that arm's own outputs).
- CONFIRM: OOF AUROC ≥ 0.70 on BOTH arms.
- PARTIAL: ≥ 0.70 on one.
- REFUTE: < 0.70 on both.
- Also report: sign of the length coefficient (does long-CoT invert it?),
  risk–coverage at 0.9/0.8/0.7, and (descriptive) whether a cascade ON TOP of
  the improved base still beats the improved base alone at its own esc-0.5
  point. H-Q refuted = a documented confgate product limit ("gate not
  validated for reasoning-style bases"), not a v8 failure.

**H-R (confidence-curated distillation — the centerpiece).** LoRA-SFT the
generic base on teacher traces from MATH-train, four filter arms at
**matched trace count N** (N = the size of the smallest filter's keep-set;
larger pools are random-subsampled, seed 0):
- (a) **GT-filter** (skyline): keep traces whose final answer is correct
  (train labels exist; deployment doesn't get them).
- (b) **Gate-filter** (zero-label): keep traces whose teacher free-gate score
  clears the precision-0.90 operating point. The gate is fit once on the
  pinned 7B MATH-500 dev cell (true labels live in DEV, never in train) and
  applied frozen to train traces — the H-L3 deployment pattern.
- (c) **Unfiltered control**: random N traces.
- (d) **Length-matched control**: random N traces whose length distribution
  is matched to arm (b)'s keep-set (quantile binning, 10 bins, seed 0).
  Pre-registered because the gate's length feature makes "gate selects
  short" vs "gate selects correct" a first-order confound (the step-length
  confounding result in reasoning-data selection): if (b) beats (c) but not
  (d), the win is a length artifact, not a confidence signal.
Decision rules (greedy MATH-500, single-shot per G2):
- CONFIRM H-R: arm (b) ≥ arm (c) + 1.5pp AND arm (b) ≥ arm (d) + 1pp AND
  arm (b) ≥ arm (a) − 1pp AND arm (b) ≥ base 0.486 + 3pp.
- PARTIAL (length-artifact verdict): (b) ≥ (c) + 1.5pp but (b) < (d) + 1pp →
  the honest recipe is "filter by length", documented as such.
- REFUTE: arm (b) < arm (c) + 0.5pp (the filter adds nothing over volume).
- Guardrails (any failure = the failing arm cannot re-pin anything):
  BBH-750 greedy ≥ base − 1pp (no catastrophic forgetting); post-SFT free-gate
  OOF AUROC on the SFT'd model ≥ 0.849 − 0.02 (improvement must not destroy
  the confidence signal that gates it).
- Always report: filter confusion matrix vs train labels (the honest
  precision/recall of the zero-label filter), trace-count N, per-arm
  trace-length distributions (the confound made visible), and arm (a) vs
  base as the plain-distillation reference.
- Pre-registered follow-up if CONFIRMED (filed at promote, not executed):
  REDI-style use of gate-REJECTED traces as negatives (2505.24850,
  pending_triage) — v8 discards them; the literature says they carry signal.

---

## Phases

### Phase 0 — Hygiene + anchors (EXP-90-pre; CPU only, ~1h)
- `verify_anchors.py`: add `step5_v8()` pinning: cascade 0.648 / oracle 0.758
  / hull 0.609 / budget 2220.71 / c_esc 3094.6868; rescue triple (257, 136,
  121) recomputed from cached y-vectors; base 0.486 / target 0.732; dev gate
  0.849; T5 trio. **Gate G0: all reproduce.**
- `validate_claims.py` full run: 245/245 + regen 123/123 (then
  `git checkout -- pathway11_h100/gpu_bundle/results.json`).
- Premise pass: seed `escalation-beats-introspection` (CONFIRMED by
  P11-FE-SHIPGATE 2026-06-12: "at matched cost, no probe beats spending the
  same tokens on escalation") via `premises.py`; link reliant FEs; regen
  queue. v8's own FEs RELY_ON it.
- File the v8 umbrella FE node (P11-FE-CEILING) with `roi_score` set
  explicitly (footgun #3).

### Phase 1 — Frontier raisers (EXP-90; pod session 1, ~2.5–3 H100 hrs)
New `v8_runpod_frontier.py` (reuses v7 pod harness conventions: script files
not heredocs, `pgrep -f '[r]un_x.sh'` guards, ssh bg with `< /dev/null`,
gen-time feature capture, gzip JSON per task, pull after each task):
1. `Qwen/Qwen2.5-Math-7B-Instruct` greedy MATH-500 (1024 max tokens).
2. `mistralai/Mathstral-7B-v0.1` greedy MATH-500 (1024).
3. `Qwen/Qwen2.5-Math-1.5B-Instruct` greedy MATH-500 (1024).
4. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` greedy MATH-500 (4096 — the
   long pole, run last so 1–3 survive an interruption; pre-registered T=0.6
   repetition-loop fallback per H-P).
**Banked side-captures (capture-only, no v8 adjudication; ~free since the
forwards happen anyway):** for all four models, (i) per-token logprob AND
per-token entropy ARRAYS (gzipped), not just scalars — unblocks
entropy-trajectory-shape work (2603.18940, pending_triage) on cached data;
(ii) mean-pooled prefill hidden states: L19 + last layer for the two
1.5B-class arms (same 28-layer architecture as base, so L19 is meaningful —
this is the FE429 comparison cell), last layer only for the 7B-class arms —
collapses P11-FE429..440 ("Reasoning Models Know When They're Right",
2504.05419; FE429 alone is budgeted "H100 day") to CPU-only follow-ups.
Geometry stays closed unless H-Q fails; banking is insurance, not a
re-opening.
Local CPU: `v8_phase1_frontier.py` → grading via pinned grader, frontier
placement at measured costs, H-N/H-O/H-P/H-Q adjudication, rescue-overlap
matrices → `results/v8_phase1_frontier.json`.
**Gate G1: append the Phase-2 freeze block** (teacher = Math-7B if H-N
confirmed else incumbent 7B; everything in the H-R recipe below) **before any
Phase-2 generation.**

### Phase 2 — Confidence-curated distillation (EXP-91; pod session 2, ~5–6 H100 hrs)
Pre-registered recipe (frozen now; G1 only fills in the teacher identity):
- Train pool: Hendrycks MATH **train** split, uniform random n=4000, seed 0
  (`EleutherAI/hendrycks_math`; pool list committed as
  `pod_payload/v8_math_train_4000.jsonl.gz` before the pod starts).
- Teacher traces: greedy, 1024 max tokens, harness MATH prompt, gen-time
  feature capture (for the gate filter).
- Filters applied locally (CPU) → four trace sets at matched N → back to pod.
- LoRA SFT ×4 on `Qwen/Qwen2.5-1.5B-Instruct`: rank 16, α 32, dropout 0.05,
  LR 1e-4 cosine, 2 epochs, effective batch 32, max_seq 1024, bf16, seed 0,
  loss on completion tokens only, final checkpoint (no selection).
- Eval per checkpoint (G2 single-shot): greedy MATH-500 (1024) + BBH-750
  greedy + gen-time features for the post-SFT gate refit.
Local: `v8_phase2_distill.py` → H-R adjudication →
`results/v8_phase2_distill.json`.

### Phase 3 — Integration + bookkeeping (EXP-92; CPU, ~half day)
- confgate: pin whatever won — new target coefficients (H-N), reasoning-base
  validity section or documented limit (H-P/H-Q), the zero-label data-engine
  recipe + SFT'd-model gate pins (H-R). README verdict table extended.
  8/8 pytest stays green; new pins get tests.
- `edge-v8-*` claims into `validate_claims.py` (every headline number above,
  from JSONs not narrative); invariant 245→245+k, zero regressions.
- Result brief `result-2026-06-12-P11-FE-CEILING.md` (or dated at execution):
  ## FE yaml (executes the behavioral core of P11-FE5; answers the premise of
  P11-FE1214; relies_on escalation-beats-introspection;
  would_update F-2/F-8), ## EXPERIMENT_LOG entry titled **EXP-85** (Next-ID
  gate), ## STATE.md update. `promote_result.py --skip-git-check --sweep`;
  if step 10 crashes, run `moot_sweep.py --from-trigger` manually.
- Commit (results/, pod_results/, harness .py stay untracked per convention;
  spec + scripts + claims + brief committed).

---

## Kill criteria / what re-anchors what

- H-N confirmed → confgate target re-pinned; v6/v7 cascade numbers remain
  pinned history (new pins are ADDITIONS, headline-figure resolution kept).
- H-P confirmed + H-Q confirmed on the same arm → the product frontier
  re-anchors on that base; the cascade story is then "gate + escalation on
  top of the best small model," and a follow-up FE (not v8) re-runs the v7
  matched-cost closure there.
- H-P confirmed + H-Q refuted → ship the documented limit; the generic-base
  cascade remains the validated product. The pre-existing contingency FE is
  P11-FE429 (mid-trajectory/prefill probes on R1-Distill, ROI 8, READY) —
  v8's banked prefill captures make it CPU-only; it runs as a follow-up, not
  inside v8.
- H-O confirmed → the target-portfolio routing FE filed at promote should
  triage 2603.20895 (prefill-activation routing, pending_triage) first, noting
  its tension with the H-M closure and the refuted geometry premise.
- H-R confirmed → the durable result: the confidence stack is a zero-label
  data engine that improves small models — the project goal, demonstrated.
- H-R refuted with (a) ≫ (c) → distillation works but curation is free-rider;
  document "use all traces" as the honest recipe.
- Everything refuted → v8 is still product documentation: the ceiling is
  problem difficulty; cascade-as-is stays final; program pivots to
  deployment/packaging (confgate release).

## Budget

~1 day CPU/eng + **~8–10 H100 SXM hrs** in two pod sessions (Phase 1 ≈ 2.5–3,
Phase 2 ≈ 5–6; the fourth SFT arm (d) is the length-confound control and is
worth its ~1 hr). Pod spend requires user sign-off at execution time. Pods
removed after verified result mirror, every session.

---

## Gate G1 — Phase-2 recipe FROZEN (appended 2026-06-13, before any Phase-2 GPU minute)

Phase-1 single-shot results (EXP-90, `results/v8_phase1_frontier.json`), each
arm graded once with the pinned grader per the selection firewall:

| Hyp | Arm | Result | Verdict |
|---|---|---|---|
| H-N | Qwen2.5-Math-7B-Instruct as escalation target | cascade 0.648→0.690 (+4.20pp, boot p=0.0025), 41/121 prior-unrescuable rescued, oracle 0.758→0.800 | **CONFIRMED** |
| H-O | Mathstral-7B-v0.1 as 2nd target | standalone 0.550, 15/121 disjoint rescues (< confirm thresh 25, > refute thresh 12.1) | INCONCLUSIVE |
| H-P | Qwen2.5-Math-1.5B-Instruct as base | 0.740 @ cost 529 (+9.2pp over budget-2221 cascade, p≈0, within budget) | **CONFIRMED** |
| H-P | DeepSeek-R1-Distill-Qwen-1.5B as base | greedy 0.634 @ cost 2671 (> budget; greedy-degenerate, fallback T=0.6 pending) | REFUTED (greedy) |
| H-Q | free gate on Math-1.5B | OOF AUROC 0.863, len-coef −1.354 | PASS (≥0.70) |
| H-Q | free gate on R1-Distill | OOF AUROC 0.951, len-coef −3.007 | PASS (≥0.70) |

**Teacher frozen for Phase 2: `Qwen/Qwen2.5-Math-7B-Instruct`** (H-N confirmed →
the strongest available escalation target is the distillation teacher).

**H-P reframing (recorded, does NOT alter the frozen recipe):** the off-the-shelf
Qwen2.5-Math-1.5B-Instruct already beats the v7 budget-matched cascade by +9.2pp
at ~¼ the cost. This re-anchors the *product* frontier (per the kill-criteria
block above) but is orthogonal to the H-R question, which asks whether a
**zero-label confidence filter curates distillation data into the GENERIC base**.
Phase-2 SFT base therefore stays `Qwen/Qwen2.5-1.5B-Instruct` exactly as
pre-registered — swapping it to the math base would confound the curation
increment with the base-identity lift already measured by H-P. The math-base
product re-anchoring is a follow-up FE, not part of v8 Phase 2.

**Frozen H-R recipe** (no parameter below may change after this line):
- Teacher: Qwen2.5-Math-7B-Instruct, greedy, 1024 max tokens, harness MATH
  prompt, gen-time feature capture.
- Train pool: `pod_payload/v8_math_train_4000.jsonl.gz` (Hendrycks MATH-train,
  n=4000, seed 0).
- Gate filter: free gate (n_gen_tokens, mean_logprob) fit once on the Math-7B
  MATH-500 DEV cell at the precision-0.90 operating point, applied frozen to the
  4000 train traces (H-L3 deployment pattern, zero train labels touched).
- Four arms at matched N = min(|GT-correct|, |gate-keep|), seed 0:
  (a) GT-filter skyline, (b) gate-filter zero-label, (c) unfiltered random,
  (d) length-matched control (10 quantile bins on n_gen, anti-confound for the
  step-length selection artifact).
- SFT base: `Qwen/Qwen2.5-1.5B-Instruct` (generic). LoRA rank 16, α 32,
  dropout 0.05, LR 1e-4 cosine, 2 epochs, eff. batch 32, max_seq 1024, bf16,
  seed 0, loss on completion tokens only, FINAL checkpoint (no selection).
- Eval (G2 single-shot per checkpoint): greedy MATH-500 + BBH-750 + gen-time
  features for the post-SFT gate refit; plus the un-adapted base reference
  (`evalbase`: BBH-750 forgetting baseline + MATH-500 0.486 reconfirm).
- Decision rules (frozen): CONFIRM b≥c+1.5pp ∧ b≥d+1pp ∧ b≥a−1pp ∧ b≥base+3pp;
  PARTIAL-length-artifact b≥c+1.5pp ∧ b<d+1pp; REFUTE b<c+0.5pp; else
  INCONCLUSIVE. Guardrails per arm: BBH≥base_bbh−1pp, post-SFT gate OOF
  AUROC≥0.829; winning-arm guardrail failure vetoes any re-pin.
