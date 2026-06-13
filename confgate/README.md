# confgate

**The free confidence gate for LLM correctness.** A logistic regression on two
scalars that every greedy generation gives you for free — `n_gen_tokens`
(response length) and `mean_logprob` (mean token logprob) — is the most
*generalizing* correctness readout found across the topo-confidence project's
geometry program (SPEC v6, 2026-06). No activations, no extra forward passes,
no model surgery: a zero-cost, model-agnostic generate-then-abstain gate.

## Pinned evidence (all numbers read from committed artifacts; see `confgate/data/pinned_meta.json`)

| Cell | Protocol | AUROC |
|---|---|---|
| Qwen2.5-1.5B MATH-500 | LOCO-7 (leave-category-out) | 0.845 |
| 1.5B→7B cross-scale | frozen transfer | 0.865 |
| SmolLM2-1.7B (held-out, near-family) | OOF 5-fold | 0.810 |
| Gemma-2-2b-it (held-out, far-family) | OOF 5-fold | 0.844 |
| OLMo-2-1B (held-out, far-family) | OOF 5-fold | 0.838 |

Which scalar carries is **family-dependent** (SmolLM2 length-only, Gemma both,
OLMo-2 length-led), so both are always pinned, with per-family weights.

## Install

```bash
pip install topo-confgate     # distribution name on PyPI
# or, from a clone:
pip install -e .
```

The import name is `confgate` (`import confgate`) regardless of how it was
installed. Python >= 3.10; deps: numpy, scikit-learn.

## Use

```python
from confgate import FreeGate, Cascade, certify, PreflightGate

gate = FreeGate.from_pinned("qwen2.5-1.5b")   # 5 families pinned
p = gate.score(n_gen_tokens, mean_logprob)    # P(correct)

# Route between a small and a large model (keep=1, escalate=5 cost model)
rows = Cascade.frontier(p, y_small, y_large)
op = Cascade.pick_operating_point(rows, target=0.69)
casc = Cascade(gate=gate, tau=op["tau"])

# Risk certificate (split-conformal LTT, Clopper-Pearson)
cert = certify(cal_scores, cal_y, eps=0.2, delta=0.1)

# In-domain group-conditional certificate (Mondrian): per-group threshold,
# each group's answered-set error certified <= eps (valid but low coverage)
from confgate import certify_grouped, apply_grouped
gcert = certify_grouped(cal_scores, cal_y, cal_categories, eps=0.2)
deployed = apply_grouped(gcert, test_scores, test_categories)   # per-group taus

# Before generating anything: prompt-length preflight (AUROC ~0.71)
pf = PreflightGate.from_pinned()
```

```bash
confgate demo --dataset math500        # reproduce pinned anchors (needs repo caches)
confgate score generations.jsonl       # {"gen_tokens":..,"mean_logprob":..} per line
```

## Certificates: what is and is not supported

- **Supported — cross-scale zero-shot (k=0):** calibrate on the small model,
  deploy the same threshold on the larger one. Pinned: validity 1.0 at 0.60
  coverage (ε=0.2, Qwen 1.5B→7B). k-label recalibration only restores
  *feasibility* past k≥32 and never beats zero-shot coverage.
- **Supported (new in 0.1.1) — in-domain group-conditional (Mondrian):**
  `certify_grouped(scores, y, groups, eps)` gives a separate threshold per
  group so *each* group's answered-set error is certified ≤ ε, and surfaces
  the cross-group coverage gap a marginal certificate hides. Pinned in-domain
  (MATH-500 categories, EXP-86/C3): marginal CP hides a **0.383** coverage
  gap with 2/7 categories silently over budget; per-group Mondrian restores
  **valid** certs for high-accuracy categories (algebra/prealgebra validity
  1.0) at a per-group budget of **k≈16–32** true labels. Honest caveat: valid
  but **low coverage** (algebra 0.5%, prealgebra 5.4% @ ε=0.2). Apply with
  `apply_grouped`.
- **Refused — cross-domain:** `certify_cross_domain()` raises
  `NotImplementedError`. The cross-domain wall is **fundamental — the model's
  task accuracy, not the calibration method**: the full distribution-shift CP
  toolkit (weighted/Tibshirani, non-exchangeable/Barber, online-ACI, Mondrian)
  all give validity 0.0 at ε=0.2 (`results/c1_cross_domain_adaptive_cp.json`,
  EXP-86/C1), and the Barber non-exchangeable TV coverage-gap 0.470 > ε makes
  a distribution-free cross-domain cert *formally impossible* from MATH alone
  (BBH base accuracy 0.241). For a new domain, collect ≥32 true labels and
  certify in-domain.

## What we tested against it — the v7 bake-off (EXP-84..89, 2026-06-12)

Every challenger ran at matched cost against this gate (SPEC v7; artifacts
in `pathway11_h100/generalization_edge/results/v7_*.json`):

- **No arm beats the gate at ≤1.05× cost (H-I holds).** Token-level
  features (min/bottom-decile logprob, entropy, top-2 margin, answer-span
  logprob) add nothing once length is honestly present. The one apparent
  win (entropy on BBH, +0.015) was an artifact of a degenerate cached
  length feature; with real lengths it adds −0.0004 (deviation D-3).
- **Honest cross-domain transfer is better than advertised:** MATH→BBH
  zero-shot AUROC **0.828** (the earlier 0.785 pin was computed with the
  degenerate length).
- **P(True) ≈ chance at 1.5B** (0.538 best format), verbalized confidence
  null, spectral-α *subtracts* (−0.021, p=0.017; closed permanently).
- **Escalation dominates every probe (H-J + H-M refuted-router).** K=8
  self-consistency majority loses to the 1.5B→7B cascade at every shared
  cost point (0.554 vs 0.732 at full-escalation cost). A learned router
  (rescue-ranking, two-model, K=2 probe bands) cannot beat the plain
  gate-ordered cascade at matched cost (Δ −0.4pp). Even Qwen2.5-Math-PRM-7B
  — a genuinely excellent verifier, AUROC 0.94 standalone — costs 1.04× a
  full escalation to run, and a hypothetical 1.5B-cost PRM with the same
  AUROC still loses. The binding constraint is rescue density (47% of
  small-model failures are unrescuable by the big model), not ranking
  quality. **Marginal compute should buy escalation, not introspection.**
- **Domain adaptation needs true labels (H-L1/L2 refuted):** K-sample
  pseudo-labels mis-set thresholds on the confirmatory BBH subset (38.8%
  pseudo-label noise). Deployment recipe: **k=32 true labels per new
  domain.** The exception is **cross-scale certificates (H-L3 confirmed):**
  big-model-agreement pseudo-labels (precision 0.95) give valid conformal
  certificates with zero human labels when moving 1.5B→7B on the same
  domain.

## Raising the ceiling — the v8 frontier raisers (EXP-90/91, 2026-06-13)

v7 showed no *probe* beats the gate; v8 asked whether anything else moves the
matched-cost ceiling (SPEC v8; artifacts `results/v8_*.json`). The product
answer: **swap the base/target, don't curate data.**

- **Pin the escalation target = `Qwen2.5-Math-7B-Instruct` (H-N).** Routed
  through the *same* gate-ordered cascade, the math-specialized 7B lifts
  matched-cost MATH-500 from 0.648 to **0.690** (+4.20pp, p=0.0025) and rescues
  41 of the 121 base failures the generic 7B cannot (oracle 0.758→**0.800**).
  Drop-in: the gate, the escalation mask, and the cost model are unchanged —
  only the large model identity changes.
- **The cheapest win is a base swap (H-P).** Off-the-shelf
  `Qwen2.5-Math-1.5B-Instruct` scores **0.740 standalone at ~¼ the budget**
  (529 vs 2221 token-FLOPs), beating the whole budget-matched cascade by
  +9.2pp. The free gate still applies on top (H-Q: OOF AUROC **0.863**).
  Recommended product shape: **off-the-shelf domain base + free-gate cascade,
  escalation target = Math-7B.** (A reasoning-distilled base — R1-Distill-1.5B —
  is *refused*: its traces run 5× longer, blowing the budget for 0.634/0.678.)
- **Confidence-curated distillation is a free-rider (H-R, REFUTED).** Using the
  gate as a zero-label filter to curate 4000 Math-7B distillation traces does
  **not** beat training on all of them at matched count (gate-filtered 0.494 <
  unfiltered 0.500 < perfect-label skyline 0.504; the gate arm even trails its
  own length-matched control). At this scale even *perfect* label filtering buys
  +0.4pp, so there is no curation increment for any filter to capture, and
  MATH-only SFT catastrophically forgets BBH (every arm < the un-adapted base).
  **Deployment recipe: don't curate distillation data with the gate — use all
  traces, or skip distillation and swap the base.** The gate's value is at
  *inference-time routing*, not *training-data selection*.

## Regenerating the pins

```bash
~/topo-confidence/.venv/bin/python scripts/fit_pinned.py
```

Refits deployment coefficients (full-data, exact pinned recipe:
`StandardScaler -> LogisticRegression(max_iter=2000, C=1.0)`) from the repo
caches and re-reads every anchor from the v6 result JSONs.

Part of [topo-confidence](https://github.com/musicofhel/topo-confidence);
SPEC v7 Phase 3a deliverable.
