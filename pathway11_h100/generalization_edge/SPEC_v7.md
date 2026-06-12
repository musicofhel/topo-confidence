# SPEC v7 — Ship the Gate, Then Try to Beat It

**Status: PRE-REGISTERED 2026-06-11 (not yet executed).**
Successor to SPEC_v6.md (executed; EXP-81/82/83). Self-contained: a fresh agent
can execute this spec without the authoring conversation.

---

## Context & north-star

v6 closed the geometry program and pinned the **free gate** — logistic
regression on (response length, mean logprob) — as the most generalizing
correctness readout: T1 LOCO 0.845, T3 cross-scale 0.865, T5 held-out
SmolLM2 0.8097 / Gemma-2-2b 0.8439 / OLMo-2-1B 0.8382. FE269 closed DoM
causality (diagnostic readout, not a lever). FE19 closed verification routing.
The 2026-06-11 moot sweeps closed 110 stale FEs.

The project goal is **improving small models, not papers**. The only surface
with proven headroom is the 1.5B→7B hybrid cascade's **11.2pp savings gap to
oracle** (cascade itself beats the random-mix hull +3.38±0.96pp — the only
routing surface that beat its honest baseline).

**v7's design inversion:** ship the deliverable FIRST from already-pinned v6
results; run the bake-off / adaptation experiments as *upgrade slots* into a
released artifact, not prerequisites for one. Every challenger arm enters with
a documented negative prior (see priors table); a null is still product
documentation ("we tested X at matched cost"), a win lands as a release.

**North-star metric:** the end-to-end accuracy/cost/coverage frontier of the
shipped gate+cascade, measured through the installed package. Transfer AUROC
remains the per-arm currency; in-domain-only wins do not re-pin anything.

**User-decided execution parameters (2026-06-11):**
- ALL new generation runs on a RunPod H100 SXM (pre-authorized for this spec;
  always H100 SXM per standing preference). Analysis/probes/conformal on local
  CPU; local 2060 Super 8GB available for 1.5B rescoring.
- Deliverable is an installable pip package (topo-features precedent).
- RunPod pods without a network volume lose data on stop — pull result JSONs
  before stopping the pod.

---

## Negative priors (pre-registered, so wins are surprising and nulls are cheap)

| Arm | Prior | Source |
|---|---|---|
| verbalized confidence | weak | 2205.14334 was an already-rejected H-12 dup |
| P(True) | weak at 1.5B | Kadavath scaling: P(True) emerges with scale |
| spectral-α | three strikes | FE749 per-layer AUROC ≈0.47–0.59 (1.5B, full-trajectory; best 0.703 at last layer ≈ length proxy); arm-2A eigenspectrum failed the incremental gate (−0.0096, p=0.42, r=−0.89 with prompt-length); premise `geometry-portable-signal` REFUTED |
| K-consistency (system view) | loses to cascade | 8× 1.5B cost > 4.7× = one 7B escalation, and the cascade already beats the hull |
| token-level logprob family | genuinely open | never tested here; literature-standard (max-softmax-prob, sequence entropy, min-token-logprob); free at deployment |

---

## Verified artifact facts the spec depends on (Phase 0 re-pins these)

1. `pathway11_h100/spectral_alpha/results.json` (FE749): α already executed and
   weak. α is therefore a **CPU closure row**, not a headline arm.
2. `generalization_edge/results/phase4_recalibration.json`: **cross-domain
   (MATH→BBH) conformal validity = 0.0 at ε=0.2 for free_baseline at k=0 AND
   with true target labels at every k in {8,16,32,64}** (eps0.3 partial at
   best). The remembered claim "k=32 labels restore validity" is TRUE ONLY for
   the cross-scale cell (math1.5b_to_7b: k0 validity 1.0 @ 60% coverage,
   ε=0.2). Phase 2's conformal hypothesis is scoped accordingly; cross-domain
   certificates are declared infeasible-with-any-light-head — a documented
   product limit, not a failure of pseudo-labels.
3. Per-token logprobs are cached **nowhere** (caches carry scalar
   `mean_logprob` only), but every cache stores generated **texts** → one
   teacher-forced rescoring pass (deterministic, no generation) recovers
   token-level features on every cell.
4. K=8 sampled cache exists ONLY for 1.5B MATH
   (`pathway11_h100/data/k8_selfconsistency/`, per-sample texts + correctness).
   Cross-domain/T5 consistency arms need new K=8 generation.
5. Held-out caches `results/phase3_{smollm2,gemma,olmo2}.npz` contain
   `{y, mean_logprob, n_gen_tokens, subjects, texts}` — **no activations**.
   Any activation-based arm on T5 requires new pod forwards.

---

## Selection firewall (V7)

- **Dev tiers (selection allowed):** Qwen-1.5B MATH (frozen LOCO-7 folds via
  `metrics.py::frozen_folds`), BBH×3 @1.5B, Qwen-7B MATH. All already
  selection-contaminated from v6; honest to keep as dev.
- **Confirmatory (one shot per arm):** SmolLM2-1.7B / Gemma-2-2b-it /
  OLMo-2-1B. Touched once in v6 *for the pinned incumbent only*. Every v7 arm
  gets exactly ONE T5 evaluation, with the full recipe (prompt formats, K,
  aggregation, parse rules) frozen at gate G2 before any T5 generation.
- **Pre-pinned now, before any 1b result is seen:**
  - K=8 T5 family: **Gemma-2-2b only** (far-family, mid-cost).
  - Phase 2 confirmatory BBH subset: **web_of_lies** (dev = the other two).
- All claims land in `validate_claims.py` as PENDING_FE at spec time, flipped
  on landing. Existing claims invariant: currently 223/223 internal PASS —
  never break existing claims; numbers come from result JSONs, not narrative.

---

## Phases

Execution order: **P0 → (P3a ∥ P1a) → G1 → P1b (pod session 1) → G2 → P1c
(pod session 2) → P2 → P3b → P4.**
Budget: ~3 days CPU/eng + ~8–10 H100 SXM hrs across two pod sessions.

### Phase 0 — Hygiene + anchor pinning (EXP-84-pre; ~1–2h CPU, 0 H100)

1. Premise-link pass over the top-30 READY queue items
   (`research-graph/premises.py link <fe> <premise>` — auto-moots when premise
   already REFUTED). Known survivors that contradict verdicts:
   - P11-FE283 (zero-ablation causality) → `dom-causal-lever`
   - P10-FE9, P10-FE18 (steering arms) → `dom-steering`
   - P8-FE1, P9-FE1 (CoE label re-validations, portability-framed) →
     `geometry-portable-signal`
   Judgement calls allowed; anything kept gets a one-line keep-reason in the
   Phase-0 note. Regenerate NEXT_EXPERIMENTS.md.
2. Resolve the k=32-validity discrepancy: identify which protocol produced the
   "k=32 restores validity" memory (likely exp_1b3's split-conformal-on-target
   vs phase4's CP-upper light-head, or ε difference); record the verdict in
   this file under "Phase 0 resolutions".
3. Pin v7 anchors in `verify_anchors.py` (extend the 15 v6 anchors):
   T5 free trio 0.8097/0.8439/0.8382; prompt-length 0.7056; arm-2A Δ −0.0096
   p=0.42; cascade +3.38±0.96pp; K=8 majority 0.554 (TRUE mode-vote — the
   ≥5-of-8 proxy is a known bug, do not reuse); FE749 α (L19 0.522, best
   last-layer 0.703); MATH→BBH frozen free-gate 0.785; cross-scale conformal
   k0 validity 1.0 @ 0.60 coverage ε=0.2; cross-domain conformal validity 0.0
   at ε=0.2 ∀k.
4. **Gate G0:** all anchors reproduce → proceed. Any failure: stop, diagnose
   cache drift before anything else.

### Phase 3a — Package scaffold (parallel track after G0; ~2 days eng, 0 H100)

New top-level dir **`confgate/`** (pip name `confgate`); do NOT rev the legacy
`topo_confidence/` package (dead v1 framing).

- `confgate/gate.py` — fit/score API; ships pinned per-family free-gate
  coefficients (Qwen-1.5B, Qwen-7B, SmolLM2-1.7B, Gemma-2-2b, OLMo-2-1B).
  Family-dependent scalar note from v6: SmolLM2 length-only, Gemma both,
  OLMo-2 length-led — pin BOTH scalars, per-family weights.
- `confgate/route.py` — 1.5B→7B cascade (port `exp_1b4_overlap.py` logic).
- `confgate/certify.py` — split-conformal LTT (port `exp_1b3_conformal.py`);
  cross-scale zero-shot certificate + k-label recalibration; refuses to emit
  cross-domain certificates (documented limit per Phase 0 fact 2).
- `confgate/preflight.py` — prompt-length Tier-A pre-generation signal (0.71).
- `confgate/cli.py` — demo: score a JSONL of (prompt, response, token
  logprobs) → gate scores, route decisions, certificate.
- `pyproject.toml` mirroring the topo-features package precedent.
- Built ONLY on pinned v6 results — zero dependency on Phases 1–2.

### Phase 1a — Zero-cost arms, local (EXP-84; ~1 day CPU + ~2h 2060, 0 H100)

New `v7_rescore_local.py`: teacher-forced rescoring of cached generated texts
(Qwen-1.5B on MATH-500 + BBH×3; fits on the 2060) → per-token logprobs,
entropies, top-2 margins, saved as a sidecar NPZ per cell.

Arms (register each in `probes.py`; evaluate standalone AUROC + DeLong
incremental gate over the free gate, frozen folds):
- min token logprob; bottom-decile mean logprob
- mean token entropy; mean top-2 logit margin
- **answer-span logprob** — mean logprob over the `\boxed{...}` span (locate
  via `k8_lib.py` answer-extraction regex)
- perplexity: documented-and-skipped (monotone transform of mean_logprob,
  AUROC-identical — state this in the result brief to preempt the reviewer)
- K-consistency from the existing K=8 cache, K∈{2,4,8} agreement rate +
  cluster entropy (exact-match clustering = semantic entropy for math;
  2406.15927 already triaged, no novelty claim). **Both label semantics**
  (§cost-matching below).
- spectral-α closure row: prefill-α fit on
  `cache/arm2a_prompt_cloud_qwen1.5b_math.npz` eigenspectrum vs prompt-length.

**H-K:** prefill-α adds over prompt-length in-domain (paired Δ>0, p<0.05).
Refutation condition: gate failure → α CLOSED permanently, zero further
compute, no T5 pass. (Expected outcome: refuted.)

**Gate G1 (promotion to pod time):** an arm proceeds to 1b iff
incremental Δ>0 at p<0.05 over the free gate, OR standalone AUROC ≥
free gate − 0.01, on in-domain dev. Free arms that pass G1 are folded into a
**candidate gate** (free gate + winners, refit) carried forward alongside the
incumbent.

### Phase 1b — Behavioral arms, dev tiers (EXP-85; pod session 1, ~4–6 H100 hrs)

New `v7_runpod_arms.py` (single pod script) + `v7_phase1b_costmatched.py`
(local CPU analysis). Pod workload:
1. Teacher-forced rescores: 7B MATH, (re-check BBH if not done locally), and
   the three T5 families' cached texts (~minutes each).
2. **P(True)**: second forward pass asking "Is the proposed answer correct?
   (A) yes (B) no", read P(A). Exactly TWO frozen prompt formats, selected on
   1.5B MATH dev only; both reported (= the sensitivity measurement); the
   better one frozen for all other cells.
3. **Verbalized confidence**: binary sure/unsure + 0–100 score, short decode.
   Pre-registered parse protocol: unparseable → impute median confidence;
   parse-failure rate reported as its own column.
4. **K=8 T=0.7 sampling on BBH×3 @1.5B** (750×8 generations) — dual-use:
   Phase-1 cross-domain consistency arm + Phase-2 pseudo-labels.
5. Optional, does not gate anything: **PRM arm** — Qwen2.5-Math-PRM-7B scores
   cached 1.5B generations (~30 min; resolves queue item P11-FE403). Its
   honest cost comparator is 7B *escalation* (≈ same FLOPs), i.e. the cascade.

**Cost axis:** token-FLOP units = generated (or scored) tokens × per-token
model cost, 7B ≈ 4.7× 1.5B. Every arm reported as (cost, signal-view AUROC,
system-view risk-coverage curve).

**Cost-matching rules (pre-registered):**
- *Signal view*: all arms predict the SAME greedy answer's correctness
  (K-agreement-around-the-greedy-answer included). Clean AUROC comparison.
- *System view (adjudicating)*: each arm is a (cost, selective-accuracy
  frontier) pair. The honest comparator for K=8-majority+agreement is NOT the
  1× free gate — it is the best use of the same budget, i.e. the 1.5B→7B
  cascade.

**H-I:** no arm adds ≥ +0.01 AUROC over the free gate at ≤1.05× cost on dev
tiers (paired DeLong, p<0.05). Refutation: any does → it enters the T5 shot
and the candidate gate.
**H-J:** K=8-majority+agreement loses to the cascade on selective accuracy at
every shared cost point (paired bootstrap). Refutation: beats it anywhere by
>1pp p<0.05 → consistency joins the Phase-4 router menu.

**Gate G2:** freeze the complete T5 recipe (which arms, formats, K,
aggregations, parse rules) by editing this file BEFORE any T5 generation.

Output artifact: `results/v7_frontier_table.json` — the matched-cost frontier
= the router's signal-cost menu (Phase 4's direct input).

### Phase 1c — Confirmatory T5, one shot (EXP-86; pod session 2, ~3–4 H100 hrs)

- Rescore-derived arms + P(True) + verbalized on all three families.
- K=8 sampling on **Gemma-2-2b only** (pre-pinned).
- α prefill passes ONLY if H-K survived 1a (expected: skipped).
- Adjudicate H-I / H-J / H-K at T5; register `edge-v7-*` claims in
  `validate_claims.py`; file result brief via `promote_result.py --sweep`
  (premise tags: H-I confirmation strengthens
  `free-baseline-strongest-readout`; new closures tag their premise).

### Phase 2 — Label-free domain adaptation (EXP-87; ~½ day CPU, 0 new H100)

Elevation of P11-FE412. Consumes 1b's BBH K=8 cache. New
`v7_phase2_pseudolabel.py`.

Pseudo-labels = K-sample agreement (K∈{4,8}) on BBH @1.5B. Dev = two BBH
subsets; confirmatory = `web_of_lies` (pre-pinned). Use cases tested: (a)
operating-point selection — choose gate threshold τ on a new domain with zero
true labels; (b) light-head refit; (c) conformal recalibration where
certificates are feasible at all.

- **H-L1 (threshold transfer — the live one):** τ from pseudo-labels achieves
  target coverage within ±5pp and answered-accuracy within 2pp of
  τ-from-32-TRUE-labels, scored on true labels.
- **H-L2 (refit no-harm):** pseudo-label-refit gate's true-label AUROC ≥
  frozen zero-shot 0.785 − 0.01.
- **H-L3 (conformal, cross-scale cell ONLY per Phase-0 resolution):**
  pseudo-label calibration achieves validity ≥0.9 at the (ε, k) where
  true-label calibration achieves it.

**Circularity guards (pre-registered — pseudo-labels correlate with
mean_logprob, errors concentrate where the gate is confident-wrong):**
1. ALL adjudicating metrics on TRUE labels (BBH has 750); pseudo-labels touch
   only calibration/refit sets → circularity surfaces as a measurable
   validity gap, never a silent pass.
2. Report pseudo-label confusion matrix vs truth (noise rate, precision/
   recall) and corr(pseudo-label-error, gate score); if errors concentrate
   above the gate threshold, report the induced coverage bias explicitly.
3. Monotonicity ablation: K=4 vs K=8 — true-label benefit must be monotone in
   K if the mechanism is real.

Refutation of H-L1/L2 → `confgate` documents "k=32 true labels per new
domain" as the honest deployment recipe. Still a shippable answer.

### Phase 3b — Package finalization + release (EXP-88; ~1–2 days eng + pod smoke minutes)

- Fold Phase 1/2 winners (if any) into `confgate`; re-pin coefficients.
- Measure the end-to-end accuracy/cost/coverage frontier on MATH + BBH
  **through the installed package** (pre-flight → 1.5B → gate → escalate →
  certify); pin frontier anchors in `verify_anchors.py`.
- Demo CLI + README; release (pip-installable; GitHub).

### Phase 4 — Cost-aware router (EXP-89; CPU only; promoted from stretch)

Stack the surviving signal menu (free gate + prompt-length + 1a/1b winners +
consistency-when-budgeted) into a router over actions
{abstain, answer-1.5B, escalate-7B, sample-more}; input =
`results/v7_frontier_table.json`.

- **H-M:** router shrinks the 11.2pp oracle savings gap by ≥3pp vs the hybrid
  cascade at the 90%-of-7B-accuracy target (paired). Comparators: random-mix
  hull (the FE19 lesson) AND best single component (v6 combiner rule).
  Refutation (<1pp): the cascade-as-is ships as final; H-M result documented
  in `confgate`.
- **H-M operating-point disambiguation (recorded 2026-06-12, BEFORE any
  router code ran):** re-deriving the gap from cached data shows the spec's
  two anchors describe different axes. The 11.2pp reproduces exactly as the
  **accuracy gap to oracle at the matched-cost escalation-fraction-0.5
  point** (free-gate cascade 0.6460 vs oracle ceiling 0.7580; oracle =
  escalate exactly the 136/500 problems where 7B rescues, capped at 0.758),
  NOT as a savings gap at the 90%-of-7B target (that gap is 36.4pp of
  escalation fraction). **Primary H-M adjudication: paired accuracy delta
  router-vs-cascade at matched cost = the esc-0.5 cascade point (cost
  673.37 + 0.5×3094.69 token-FLOPs/problem), router policy strictly OOF on
  the frozen 5-fold split, paired bootstrap B=2000; confirm iff Δ ≥ +3pp
  with p<0.05, refute iff Δ < +1pp.** Secondary (reported, not
  adjudicating): escalation-cost savings at the 90%-of-7B accuracy target.
  All learned router components fit on train folds only; the action menu
  and cost accounting come from `results/v7_frontier_table.json` (D-2
  semantics: token-feature arms cost 0 extra; K-sample and P(True) arms
  pay their measured token costs; escalation pays the 7B 4.7×/token cost).

---

## Pre-registered hypotheses (summary)

| ID | Statement | Threshold | Refutation consequence |
|---|---|---|---|
| H-I | No bake-off arm beats free gate at ≤1.05× cost (dev) | Δ ≥ +0.01 AUROC, p<0.05 | winner → T5 shot + candidate gate |
| H-J | K-consistency loses to cascade at every matched cost point | >1pp anywhere, p<0.05 | consistency → Phase-4 router menu |
| H-K | Prefill spectral-α adds over prompt-length in-domain | Δ>0, p<0.05 | (expected) α CLOSED permanently |
| H-L1 | Pseudo-label τ ≈ 32-true-label τ (coverage ±5pp, acc ±2pp) | as stated, true-label-scored | package documents k=32 requirement |
| H-L2 | Pseudo-refit no-harm vs frozen 0.785 | ≥ −0.01 | same as H-L1 |
| H-L3 | Pseudo-conformal valid where true-label is (cross-scale only) | validity ≥0.9 at same (ε,k) | certificate requires true labels |
| H-M | Router beats cascade ≥3pp of the 11.2pp oracle gap | ≥3pp at 90%-of-7B target, paired | cascade-as-is ships |

## Feasibility matrix (arm × tier; "cached"=no new compute)

| Arm | 1.5B MATH | BBH 1.5B | 7B MATH | T5 ×3 |
|---|---|---|---|---|
| free gate (incumbent) | cached | cached | cached | cached |
| prompt-length | CPU | CPU | CPU | CPU |
| token-logprob family | 2060 rescore | 2060 rescore | pod rescore | pod rescore |
| K-consistency / cluster entropy | cached (k8 npz) | pod K=8 gen | skipped (not load-bearing) | pod K=8, Gemma only |
| P(True) | pod 1 pass | pod | pod | pod ×3 |
| verbalized | pod decode | pod | pod | pod ×3 |
| spectral-α prefill | CPU (arm2a cache) | only if H-K survives | CPU (7B all-layer cache) | impossible from cache; pod prefill only on survival |
| prefill DoM (reference) | cached | cached | cached | excluded (hidden-dim-bound) |
| PRM-7B (optional) | pod ~30min | pod | — | skip |

## Reuse map

- `generalization_edge/probes.py` — probe registry (extend; `free_baseline` is
  the incumbent), `transfer.py` — panel runner + DeLong incremental gate,
  `metrics.py` — AUROC+CI, frozen folds, `k8_lib.py` — K-sample utilities +
  answer extraction, `exp_1b3_conformal.py` — split-conformal LTT,
  `exp_1b4_overlap.py` — cascade, `verify_anchors.py` — anchor regression.
- `causal_dom/common.py` — model loading (bf16/sdpa), batched generation,
  boxed-answer checking (pod scripts).
- Caches: `pathway8_layerwise/data/math500/` (1.5B all-layer),
  `pathway11_h100/data/math500_7b/`, `pathway8_layerwise/data/bbh/`,
  `pathway11_h100/data/k8_selfconsistency/`,
  `generalization_edge/results/phase3_{smollm2,gemma,olmo2}.npz`,
  `generalization_edge/cache/arm2a_prompt_cloud_qwen1.5b_math.npz`.
- New files: `v7_rescore_local.py`, `v7_phase1a.py`, `v7_runpod_arms.py`,
  `v7_phase1b_costmatched.py`, `v7_phase2_pseudolabel.py`, `confgate/` (top
  level).

## Verification

- G0 anchor regression before any new work; `validate_claims.py` after every
  phase (existing 223/223 internal PASS must never regress).
- Phase 3b end-to-end: `pip install -e confgate && confgate demo --dataset
  math500` reproduces pinned frontier numbers from cached data.
- Every phase files a result brief via `promote_result.py --sweep` so the
  moot architecture auto-propagates verdicts (premise tags in FE YAML).

## Phase 0 resolutions

*(filled in 2026-06-12, EXP-84-pre)*

- **k=32 conformal validity discrepancy: RESOLVED — it was a feasibility/validity
  conflation, single artifact, no protocol conflict.** Both claims trace to
  `results/phase4_recalibration.json` (CP-upper light-head recalibration,
  R=200). Cross-scale (math1.5b_to_7b, free_baseline, ε=0.2): k0 zero-shot is
  feasible_frac 1.0 / validity 1.0 / coverage 0.60; k=8/16 are INFEASIBLE
  (CP-upper small-sample penalty); k=32 restores partial *feasibility* (0.21;
  ε=0.3: 0.72) and k=64 more (0.655), with validity 1.0 conditional-on-feasible
  but coverage always below zero-shot (k64 ε=0.2: 0.29). So "recalibrate only
  past k=32" = feasibility returns at k≥32; zero-shot remains the best
  cross-scale certificate. Cross-domain (math_to_bbh): validity 0.0 at ε=0.2 at
  EVERY k∈{0,32,64} where feasible; best anywhere = 0.41 at (k32, ε=0.3) —
  never near the 0.9 target. `1b3_conformal.json::transfer_HH` (calib MATH →
  deploy BBH, prefill-DoM) agrees: infeasible at ε≤0.2, coverage 0.0 at ε=0.3.
  **Consequences confirmed:** H-L3 scoped to the cross-scale cell only;
  `certify.py` refuses cross-domain certificates and prefers k=0 zero-shot
  cross-scale (k-label recal documented as feasibility-restoring, not
  coverage-improving).

- **Premise-link pass (2026-06-12): 9 FEs linked → auto-MOOTED** (premises
  already REFUTED): P11-FE283→dom-causal-lever; P10-FE9, P10-FE18, P11-FE443,
  P10-FE1, P3-FE1, P10-FE16→dom-steering (FE443/FE1/P3-FE1/FE16 are judgment
  calls — all four are steering-for-accuracy designs or exist only to feed
  P10-FE1's steering arms); P8-FE1, P9-FE1→geometry-portable-signal.
  **Kept (one-line reasons):** P10-FE20, P10-FE48, P11-FE5, P11-FE1214,
  P11-FE1003, P11-FE1047, P11-FE110625A and the P11-FE10xx mechanistic cluster
  — understanding/controls for F-2/F-3, not premise-reliant; P10-FE49 — uses
  injection for *verbalization/interpretation*, not accuracy steering;
  P11-FE1021/FE1022 — internal-vs-output-signal complementarity, directly
  feeds v7 Phase 1a's K-consistency arms. Queue regenerated; top is now free
  of pre-verdict steering/causality items.

## Execution deviations + Phase 1a verdicts

*(recorded 2026-06-12, mid-execution)*

- **DEVIATION D-1 (compute placement):** Phase 1a's BBH rescore and all other
  GPU passes moved from the local 2060 to the pod (user directive 2026-06-12:
  no local GPU work, even forward-only — the spec's "local 2060 available"
  carve-out is void). Consequence: G1 could not be adjudicated before pod
  session 1, so **G1 merges into the pre-G2 analysis step**; pod session 1
  runs the full dev-tier menu unconditionally (all of it was session-1 budget
  anyway). The selection firewall is unaffected: G1+G2 still gate everything
  that reaches confirmatory T5, and no T5 generation happens before the G2
  freeze. MATH rescore (the one cell that did run locally, before the
  directive) was kept: alignment corr 0.9985.
- **DEVIATION D-2 (cost accounting, pre-registered before 1b adjudication):**
  the token-logprob family is scored at marginal deployment cost 0 (rel_cost
  1.0) because min/p10/entropy/margin/answer-span read off the decoder's own
  logits during generation; the rescore pass exists only to reconstruct what
  the v6 caches didn't store. H-I's ≤1.05× bracket therefore contains the
  free gate + token arms; P(True)/verbalized/K-sample/PRM pay measured extra
  forwards (pod token counts).
- **Phase 1a verdicts (EXP-84, `results/v7_phase1a.json`):** free gate OOF
  math 0.8490 / bbh 0.7823 (LOCO-free in-domain protocol, not the 0.845 LOCO
  anchor). **H-K REFUTED with prejudice** — α+length 0.6850 vs length-only
  0.7056, Δ=−0.0206 p=0.017 (α *subtracts*); spectral-α CLOSED permanently,
  no T5 pass, `geometry-portable-signal` closure deepens. MATH token arms: no
  G1 promotion (free gate unbeaten in-domain, v6 story holds). BBH:
  **mean_token_entropy promotes with a significant incremental win**
  (+0.0151, p=0.001); bottom_decile_logprob and token-family-joint promote
  via the standalone route. K-consistency (MATH, signal view): all agreement
  arms promote decisively (k8_agreement combined 0.9279, Δ+0.0789, p<0.001);
  cost adjudication deferred to 1b (H-J). Sidecar alignment corrs: bbh-pod
  0.9929, 7b-math-pod 0.9787, math-local 0.9985 — all pass the 0.90 gate.

- **DEVIATION D-3 (discovered during Phase 2, after the G2 freeze below):**
  `nocompute/cache/bbh_1p5b.npz` carries `n_gen_tokens` ALL-ZERO (v6 cache
  artifact) — every v6/v7 BBH "free gate (length+logprob)" number was
  logprob-only de facto, including the pinned 0.785 MATH→BBH anchor and the
  1b BBH free gate 0.7823. Pinned artifacts stay untouched (anchors are
  artifact numbers); the correction lives in
  `results/v7_bbh_length_sensitivity.json`: honest gate with real (sidecar)
  lengths = **0.8061** (+0.0238, p=0.076), length alone 0.8006, and the H-I
  "winner" mean_token_entropy adds **−0.0004** over the honest gate — the
  +0.0151 was a missing-length proxy. **Corrected adjudication: H-I HOLDS in
  all dev cells under the honestly-computed free gate; the BBH-cell
  refutation is reclassified as artifact-of-degenerate-gate.** The
  free-gate thesis strengthens. Phase-2 H-L1/L2 keep the protocol-frozen
  (degenerate) gate because their threshold (0.785) is the pinned anchor;
  noted in their JSON's interpretation. T5 cells are unaffected (real
  lengths verified by nonzero pinned length coefficients). Two honest-gate
  ripples (same JSON): honest MATH→BBH zero-shot transfer = **0.8277** (the
  pinned 0.785 understated cross-domain transfer by ~4pp), and the honest
  H-L2 re-check **REFUTES refit no-harm** (pseudo-refit 0.6214 < frozen
  0.6986 − 0.01; the protocol-frozen "pass" was vacuous because degenerate
  2-feature refits collapse to the same logprob ranking).

- **Phase 2 verdicts (EXP-87, `results/v7_phase2_pseudolabel.json`):**
  **H-L1 REFUTED** on confirmatory web_of_lies (pseudo-τ coverage 0.284 vs
  true-32 0.535 at +10pp target — pseudo-label noise 0.388 there; guards
  caught it exactly as designed). **H-L2 vacuous under the pinned gate,
  REFUTED under the honest gate** (D-3). **H-L3 CONFIRMED** — cross-model
  pseudo-labels (7B greedy == 1.5B K=8 modal, precision 0.951) give
  conformal validity ≥0.9 in all four live (ε,k) cells; label-free
  cross-scale certificates are real. Deployment recipe for confgate: free
  gate transfers zero-shot (honest 0.8277 MATH→BBH); for thresholds/refits
  on a shifted domain use k=32 TRUE labels (pseudo-labels refuted); for
  cross-scale certificates pseudo-labels suffice.

- **DEVIATION D-4 (discovered during P1c pre-validation, before any T5
  adjudication):** the SmolLM2 and OLMo-2 tokenizers do not round-trip
  decode→encode on their own cached generations (exact re-encode match 66%
  and 0%; OLMo-2 re-encodes ~7% longer), so teacher-forced rescoring from
  cached texts is invalid for those families — their pod rescore sidecars
  fail the pre-registered 0.90 alignment gate (corr 0.7865 / 0.0367) for a
  *mechanical* reason, not signal absence. Gemma round-trips cleanly (corr
  0.9998) and keeps the rescore path. Fix: a new pod task `genscore`
  regenerates the T5 cells greedily and captures the token features
  directly from the decoder's logits at generation time (no
  re-tokenization). Faithfulness is validated by **text equality vs the
  pinned cache** per row: non-matching rows are NaN'd (median-imputed
  downstream, finite_fraction reported), and the alignment gate becomes
  corr(gen-time mean logprob, cached mean_logprob) ≥ 0.90 on the matched
  subset (`v7_pod_to_sidecar.convert_genscore`). The G2 freeze is
  unaffected: same arms, same adjudication criteria, only the
  feature-extraction mechanism for two families changed, recorded here
  before seeing any T5 arm number.

## Gate G2 — frozen T5 recipe (recorded 2026-06-12, BEFORE any T5 generation)

Phase 1b verdicts driving the freeze (`results/v7_phase1b_costmatched.json`):
**H-I REFUTED** in the qwen1.5b_bbh cell by token_mean_entropy (Δ+0.0151,
p=0.0014, rel_cost 1.0); holds in both MATH cells. **H-J CONFIRMED** — K-
majority loses to the cascade at every shared cost point, every coverage
(Δ −0.15…−0.18, paired bootstrap p(K wins >1pp)=1.0); consistency joins the
Phase-4 router as a *signal* only. P(True) format frozen by the pre-pinned
1.5B-MATH-dev rule: **fmtB** ("True or False", 0.5381 > fmtA 0.5021); fmtA
stays recorded as sensitivity. Verbalized: null on all dev cells (parse
failure 0.00 everywhere).

The one-shot confirmatory recipe (families SmolLM2-1.7B / Gemma-2-2b-it /
OLMo-2-1B, MATH-500 cells, exactly one evaluation per arm):
1. **Rescore pass** (one teacher-forced forward per problem): adjudicating
   arm = free+mean_token_entropy combined. Because the H-I win is
   BBH-domain-specific (entropy adds nothing on MATH dev), the T5 criterion
   is **no-harm**: combined ≥ free − 0.01 per family. The BBH entropy win
   itself is registered as a dev-cell claim; cross-family×cross-domain
   confirmation is out of scope/budget. Other token arms reported
   descriptively only. Alignment corr ≥ 0.90 required per family.
2. **P(True)**: fmtB adjudicates (standalone + combined-with-free); fmtA
   recorded (sensitivity).
3. **Verbalized** binary + 0–100, frozen parse protocol (NaN→median, failure
   rate reported); descriptive (dev nulls), adjudication only if some family
   shows combined Δ≥+0.01 p<0.05 — which would *contradict* dev and be
   flagged as such, not promoted.
4. **K=8 T=0.7 seed 9999 on Gemma-2-2b ONLY** (pre-pinned in the spec before
   any 1b result): k{2,4,8} agreement + cluster entropy, signal view vs
   greedy correctness + majority-accuracy system view; grading LOCAL via the
   canonical math grader. (`v7_runpod_arms.py --task k8t5 --model gemma`.)
5. **Spectral-α: SKIPPED** (H-K refuted at 1a).
6. Protocol per family: frozen 5-fold OOF, StandardScaler+LR(C=1.0,
   max_iter=2000), DeLong for increments — identical to dev. No iteration,
   no format/arm/threshold changes after this point.

## Phase 1c verdict (EXP-86, `results/v7_phase1c_t5.json`, 2026-06-12)

One shot, exactly as frozen at G2. **H-I HOLDS at T5 in all three held-out
families**: no token arm, P(True) (either format), or verbalized arm adds
≥+0.01 AUROC with p<0.05 anywhere. **The adjudicating entropy no-harm
criterion passes all three families** (combined ≥ free − 0.01: smollm2
+0.0003, gemma +0.0025, olmo2 −0.0020). Free gate OOF AUROC: SmolLM2 0.8083,
Gemma 0.8422, OLMo-2 0.8357 — the v6 cross-architecture pins reproduce
through the v7 harness. Token-feature faithfulness: Gemma rescore corr
0.9998; SmolLM2/OLMo-2 via D-4 genscore, matched-subset corr 0.9998 both,
text-match (finite) fraction 0.250/0.272 — greedy regeneration under bs=8
batching diverges from the pinned bs-differing cache on most 1024-token
generations, so the SmolLM2/OLMo-2 token rows carry reduced power
(75% median-imputed); the adjudication is unaffected (no-harm passes, and
the *high-power* Gemma cell agrees with the dev nulls). Gemma K=8
(pre-pinned family): the consistency *signal* view replicates the dev
pattern out-of-family (k8_agreement Δ+0.0545, p<0.001 over the free gate)
while the *system* view replicates the H-J economics — K=8 majority 0.264
vs greedy 0.256, +0.8pp for 8× the tokens vs the cascade's much larger
matched-cost gain. Verdict: the v7 bake-off ends with the free gate
unbeaten at ≤1.05× cost on dev AND confirmatory tiers; consistency remains
a strong *signal* whose honest *use* is dominated by escalation.

## Phase 4 verdict (EXP-89, `results/v7_phase4_router.json`, 2026-06-12)

**H-M REFUTED** (Δ −0.40pp < +1pp): at the matched-cost esc-0.5 point the
nested-OOF router (inner 5-fold policy selection over rich-gate /
rescue-ranker / two-model / K=2-band) lands at 0.6440 vs the free-gate
cascade's 0.6480 (paired bootstrap p(Δ≤0)=0.68; oracle 0.7580, random-mix
hull 0.6090). Every K=2-band allocation is strictly worse (probe tokens eat
escalation budget; acc falls monotonically in probe fraction f). Secondary
90%-of-7B view: rescue-ranking does not beat free-gate ordering on
escalation cost either (0.552 vs 0.538; oracle 0.174). **The cascade-as-is
ships as final** (pre-registered consequence).

**PRM descriptive row (resolves P11-FE403's comparator question):**
Qwen2.5-Math-PRM-7B is an excellent *verifier* — prm_mean standalone AUROC
0.9396, combined-with-free 0.9508 (Δ+0.1018, p<1e-4) — but costs 1.042× a
full 7B escalation per scored transcript, so it is cost-dominated as a
*routing* signal under the frozen menu. Two exploratory blocks (labeled
NOT-H-M in the JSON) close the loophole: (a) PRM-banded cascade at true
cost is strictly worse at every probe fraction (0.518–0.606 < 0.648);
(b) even re-costed hypothetically at 1.5B scale (0.223× escalation) while
keeping the 7B AUROC — an upper bound no real small PRM reaches — the band
never beats the plain cascade (best 0.630). **General closure: at matched
cost, no probe in the menu (K-sample, P(True), PRM) beats spending the
same tokens on escalation itself; the binding constraint is rescue
density, not ranking quality** — the gate cannot know which 1.5B failures
the 7B will also fail (121/257 failures are unrescuable), and probes only
sharpen P(1.5B wrong), not P(7B rescues). No cheap-PRM follow-up FE is
warranted on this evidence.

## Single biggest risk (named so execution doesn't drift)

Phase 1 is a bake-off the incumbent is pre-ordained to win; the proven
headroom is the 11.2pp cascade-oracle gap. Mitigations baked in: package-first
parallel track (deliverable can't be hostaged by nulls), G1 gating (pod money
only follows in-domain survival), Phase 4 promoted with a real win condition
(H-M) and the frontier table as its contracted input. Do not let confirmatory
budget migrate from Phases 3b/4 into extra bake-off cells.
