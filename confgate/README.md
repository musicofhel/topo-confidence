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
pip install -e .          # from this directory
```

Python >= 3.10; deps: numpy, scikit-learn.

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
- **Refused — cross-domain:** `certify_cross_domain()` raises
  `NotImplementedError`. Validity is 0.0 at ε=0.2 for every feasible k
  (`results/phase4_recalibration.json`). For a new domain, collect ≥32 true
  labels and certify in-domain.

## Regenerating the pins

```bash
~/topo-confidence/.venv/bin/python scripts/fit_pinned.py
```

Refits deployment coefficients (full-data, exact pinned recipe:
`StandardScaler -> LogisticRegression(max_iter=2000, C=1.0)`) from the repo
caches and re-reads every anchor from the v6 result JSONs.

Part of [topo-confidence](https://github.com/musicofhel/topo-confidence);
SPEC v7 Phase 3a deliverable.
