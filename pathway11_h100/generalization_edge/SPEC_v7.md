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

*(filled in at execution time)*

- k=32 conformal validity discrepancy: __pending__
- Premise-link pass keep/moot decisions: __pending__

## Single biggest risk (named so execution doesn't drift)

Phase 1 is a bake-off the incumbent is pre-ordained to win; the proven
headroom is the 11.2pp cascade-oracle gap. Mitigations baked in: package-first
parallel track (deliverable can't be hostaged by nulls), G1 gating (pod money
only follows in-domain survival), Phase 4 promoted with a real win condition
(H-M) and the frontier table as its contracted input. Do not let confirmatory
budget migrate from Phases 3b/4 into extra bake-off cells.
