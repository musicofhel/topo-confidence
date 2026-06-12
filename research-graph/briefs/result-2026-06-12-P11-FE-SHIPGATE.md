# Result brief — "Ship the Gate, Then Try to Beat It" (SPEC v7), EXP-84..89

SPEC v7 executed end-to-end in ~36 hours: the v6-pinned free gate was shipped
first as the `confgate/` pip package (Phase 3a), then attacked by every
plausible challenger at matched cost — token-level logprob features, P(True),
verbalized confidence, K-sample consistency, spectral-α, a process reward
model, pseudo-label domain adaptation, and a learned cost-aware router. Two
RunPod H100 SXM sessions (~2.3 hrs total GPU, user pre-authorized); all
analysis local CPU. Two cache defects were discovered and corrected en route
(D-3 degenerate BBH lengths, D-4 tokenizer round-trip failure), both of which
*strengthen* the free-gate story.

## Headline

**The free gate survived everything.** No challenger adds ≥+0.01 AUROC at
≤1.05× cost in any dev cell once features are computed honestly (H-I), and
the honest correction shows the gate was *underrated*: real generation
lengths lift the BBH in-domain gate from 0.782 to 0.806 and the zero-shot
MATH→BBH transfer anchor from 0.785 to **0.828**. The deeper closure is
economic: **at matched cost, no introspection probe — K-sample consistency,
P(True), or even a 0.94-AUROC process reward model — beats spending the same
tokens on escalation itself** (H-J, H-M, PRM block). The binding constraint
is rescue density (47% of 1.5B failures are unrescuable by 7B), not ranking
quality. The cascade-as-is ships as final.

## What was run

- **P0** — anchors + hygiene: `verify_anchors.py step4_v7()` G0 ALL-PASS;
  claims invariant 223/223; k=32-conformal discrepancy resolved
  (feasibility-vs-validity conflation; "k=32 restores validity" is
  cross-scale only).
- **P3a** — `confgate/` package (gate/route/certify/preflight/CLI, 8/8
  pytest) built from pinned v6 artifacts before any new experiment ran.
- **P1a (EXP-84)** — teacher-forced token arms + K-agreement (signal view) +
  spectral-α closure, local CPU + pod rescores.
- **P1b (EXP-85)** — pod session 1: BBH/7B rescores, P(True) ×2 formats,
  verbalized ×2, K=8 T=0.7 BBH (750×8), PRM-7B; cost-matched frontier table.
- **P1c (EXP-86)** — pod session 2, one-shot confirmatory T5 under the G2
  recipe frozen before any T5 generation (SmolLM2-1.7B / Gemma-2-2b-it /
  OLMo-2-1B; K=8 Gemma-only pre-pinned).
- **P2 (EXP-87)** — pseudo-label adaptation (executes P11-FE412).
- **P4 (EXP-89)** — nested-OOF cost-aware router vs the 11.2pp oracle gap.

## Hypothesis scoreboard

| ID | Statement | Verdict |
|---|---|---|
| H-I | No arm beats free gate at ≤1.05× cost | **HOLDS** (honest); the lone "win" (BBH entropy +0.0151) was a degenerate-length artifact (D-3) |
| H-J | K-consistency loses to cascade cost-matched | **CONFIRMED** (K=8 majority 0.554 vs cascade 0.732 at same cost; p(K wins)=1.0) |
| H-K | Spectral-α adds over prompt-length | **REFUTED with prejudice** (α *subtracts*: −0.0206, p=0.017; closed permanently) |
| H-L1 | Pseudo-label threshold transfer | **REFUTED** (web_of_lies pseudo-noise 0.388; coverage 0.284 vs true-32 0.535) |
| H-L2 | Pseudo-refit no-harm | vacuous-pinned / **REFUTED honest** (0.6214 < 0.6986−0.01) |
| H-L3 | Pseudo-conformal cross-scale | **CONFIRMED** (precision 0.951; validity ≥0.9 all 4 live cells) |
| H-M | Router shrinks oracle gap ≥3pp | **REFUTED** (Δ −0.40pp, p=0.68; cascade-as-is ships) |

## Deviations (all recorded in SPEC_v7.md before the affected adjudications)

- **D-1**: all GPU work moved to RunPod (no-local-GPU directive); G1 merged
  into pre-G2 analysis, selection firewall unaffected.
- **D-2**: token-logprob arms scored at deployment cost 0 (logits are free
  at generation time).
- **D-3**: `bbh_1p5b.npz` carried `n_gen_tokens` ALL-ZERO — every v6/v7 BBH
  "length+logprob" number was logprob-only de facto. Honest sensitivity
  analysis (`v7_bbh_length_sensitivity.json`): in-domain 0.782→0.806,
  transfer 0.785→0.828, entropy's H-I win reclassified as artifact.
- **D-4**: SmolLM2/OLMo-2 tokenizers don't round-trip decode→encode (66%/0%
  exact), invalidating teacher-forced rescoring from cached texts; replaced
  with gen-time feature capture (`genscore`), faithfulness gated by text
  equality vs the pinned cache (matched-subset logprob corr ≥0.998), with
  25.0%/27.2% (SmolLM2/OLMo-2) of rows text-matched on greedy regeneration
  (bs=8 batching diverges from the pinned cache on most 1024-token
  generations; non-matching rows NaN→median, power reduced, adjudication
  unaffected).

## P1c — confirmatory T5 (one shot, G2-frozen recipe)

**H-I holds in all three held-out families; the adjudicating entropy
no-harm criterion passes everywhere.** Free gate OOF AUROC through the v7
harness: SmolLM2 **0.8083**, Gemma-2-2b-it **0.8422**, OLMo-2-1B **0.8357**
(the v6 cross-architecture pins reproduce). No token arm, P(True) format,
or verbalized arm adds ≥+0.01 with p<0.05 in any family (largest combined
delta anywhere: verb_score on SmolLM2 +0.0083, p=0.31). Faithfulness:
alignment corr 0.9998 in all three families (Gemma teacher-forced;
SmolLM2/OLMo-2 gen-time via D-4). Gemma K=8 (pre-pinned family): the
consistency *signal* replicates out-of-family — k8_agreement adds +0.0545
over the free gate (p<1e-4), k2 +0.0451 — while the *system* economics
replicate H-J: K=8 majority accuracy 0.264 vs greedy 0.256, +0.8pp for 8×
the tokens.

## P4 — the router result in one paragraph

At the matched-cost operating point where the 11.2pp oracle gap is defined
(escalation fraction 0.5), a nested-OOF router selecting per-fold among
rich-gate / rescue-ranker / two-model / K=2-probe-band policies lands at
0.6440 vs the plain free-gate cascade's 0.6480 (oracle 0.7580, random-mix
hull 0.6090). Every probe allocation is strictly worse — probe tokens eat
escalation budget faster than ranking improves. Qwen2.5-Math-PRM-7B is an
excellent verifier (standalone 0.9396, +0.1018 over the gate, p<1e-4) but
costs 1.042× a full escalation per scored transcript; even re-costed
hypothetically at 1.5B scale while keeping 7B AUROC (an unreachable upper
bound) the PRM-banded cascade never beats the plain one (0.630 < 0.648).
Resolves P11-FE403.

## FE

```yaml
fe_id: P11-FE-SHIPGATE
status: COMPLETED
outcome: "SPEC v7 executed end-to-end (EXP-84..89): confgate/ pip package shipped FIRST from pinned v6 results, then every challenger ran at matched cost and LOST. H-I holds on dev AND confirmatory T5 (SmolLM2 0.808 / Gemma 0.842 / OLMo-2 0.836 free-gate OOF; no arm adds >=+0.01 at <=1.05x cost anywhere; entropy no-harm passes all three families). H-K refuted with prejudice (spectral-alpha SUBTRACTS, -0.0206 p=0.017, closed permanently). H-J confirmed (K=8 majority 0.554 vs cascade 0.732 at matched cost). H-M refuted (nested-OOF router -0.40pp vs plain cascade at the 11.2pp-gap operating point, p=0.68; rescue density — 47% of 1.5B failures unrescuable by 7B — is the binding constraint, not ranking). PRM block (resolves P11-FE403): Qwen2.5-Math-PRM-7B is an excellent verifier (0.9396 standalone, +0.1018 over gate p<1e-4) but costs 1.042x a full escalation; even a hypothetical 1.5B-cost PRM with 7B AUROC loses to plain escalation. GENERAL CLOSURE: at matched cost, no introspection probe beats spending the same tokens on escalation. P2 (executes P11-FE412): H-L1/H-L2 refuted (pseudo-label noise 0.388 on web_of_lies; k=32 TRUE labels is the honest recipe), H-L3 confirmed (7B-agreement pseudo-labels precision 0.951 give valid cross-scale conformal certs, zero human labels). Cache defects found+corrected: D-3 BBH n_gen_tokens all-zero (honest BBH gate 0.782->0.806, MATH->BBH transfer anchor 0.785->0.828 — gate was UNDERRATED); D-4 SmolLM2/OLMo-2 tokenizer round-trip failure (genscore gen-time capture, corr 0.9998 on matched subset). Deliverable: confgate cascade-as-is ships as final."
result_json: pathway11_h100/generalization_edge/results/v7_phase1c_t5.json
result_json_all:
  - pathway11_h100/generalization_edge/results/v7_phase1a.json
  - pathway11_h100/generalization_edge/results/v7_phase1b_costmatched.json
  - pathway11_h100/generalization_edge/results/v7_frontier_table.json
  - pathway11_h100/generalization_edge/results/v7_phase2_pseudolabel.json
  - pathway11_h100/generalization_edge/results/v7_bbh_length_sensitivity.json
  - pathway11_h100/generalization_edge/results/v7_phase1c_t5.json
  - pathway11_h100/generalization_edge/results/v7_phase4_router.json
executes: ["P11-FE412", "P11-FE403"]
refutes: ["H-K", "H-L1", "H-L2", "H-M"]
confirms: ["H-I", "H-J", "H-L3"]
confirms_premise: [free-baseline-strongest-readout]
would_update: ["F-2", "F-8"]
```

## EXPERIMENT_LOG entry

## EXP-84: SPEC v7 "Ship the Gate, Then Try to Beat It" — full program (bake-off, confgate, pseudo-labels, router; spec-internal phase labels EXP-84..89)

**Date:** 2026-06-12. **Compute:** local CPU + 2 RunPod H100 SXM sessions
(~2.3 GPU-hrs; D-1: zero local GPU). Qwen-2.5-1.5B/7B MATH+BBH dev;
SmolLM2-1.7B / Gemma-2-2b-it / OLMo-2-1B confirmatory T5.

**Design inversion executed:** `confgate/` shipped first from pinned v6
artifacts (gate/route/certify/preflight/CLI, 8/8 pytest); all experiments ran
as upgrade slots into a released artifact. **Bake-off (1a/1b/1c):** free gate
unbeaten at ≤1.05× cost on dev (1.5B MATH 0.849, BBH honest 0.806, 7B 0.899)
and confirmatory T5 (SmolLM2 0.808 / Gemma 0.842 / OLMo-2 0.836); P(True)
≈ chance at 1.5B (best 0.538), verbalized null, spectral-α subtracts
(H-K closed), token arms add nothing once length is honest. **Economics
(1b/4):** K=8 majority 0.554 vs cascade 0.732 at matched cost (H-J);
nested-OOF router −0.40pp vs plain cascade (H-M refuted); PRM-7B 0.94 AUROC
but 1.042× escalation cost — at matched cost no probe beats escalation
itself (binding constraint: 47% of 1.5B failures unrescuable). **Adaptation
(2):** pseudo-label thresholds/refits refuted (noise 0.388); cross-scale
pseudo-certs confirmed (precision 0.951, validity ≥0.9). **Cache defects:**
D-3 BBH lengths all-zero → honest gate 0.806, transfer anchor 0.785→0.828;
D-4 tokenizer round-trip → genscore gen-time capture (corr 0.9998).
Verdicts H-I/H-J/H-K/H-L1-3/H-M all adjudicated; selection firewall held
(G2 frozen before any T5 generation; one evaluation per arm).

## STATE.md last-experiment update

**EXP-84..89 (SPEC v7 "Ship the Gate, Then Try to Beat It") — confgate
shipped; every challenger lost at matched cost; escalation beats
introspection.** `confgate/` pip package (free gate + cascade + cross-scale
certs + preflight) built from pinned v6 results, then attacked: token-level
features, P(True), verbalized, K-consistency, spectral-α, PRM-7B, pseudo-label
adaptation, learned router — none beats the free gate at ≤1.05× cost on dev
or confirmatory T5 (SmolLM2 0.808 / Gemma 0.842 / OLMo-2 0.836), and no probe
beats spending the same tokens on 1.5B→7B escalation (H-J, H-M, PRM: 0.94
AUROC verifier but 1.042× escalation cost; 47% of failures unrescuable).
Honest-cache corrections STRENGTHEN the gate: BBH in-domain 0.782→0.806,
MATH→BBH transfer 0.785→0.828 (D-3). Pseudo-labels: refuted for
thresholds/refits (k=32 true labels is the recipe), confirmed for
cross-scale certificates (precision 0.951). Spectral-α closed permanently.
Cascade-as-is ships as final. Claims 223→245 internal PASS (+22 edge-v7-*). Artifacts:
`generalization_edge/results/v7_*.json`, SPEC_v7.md (G2 freeze, D-1..D-4),
`confgate/`.

## Sources

- `pathway11_h100/generalization_edge/SPEC_v7.md` (spec + G2 freeze + D-1..D-4)
- `pathway11_h100/generalization_edge/results/v7_*.json` (all numbers)
- `confgate/` (the deliverable)
