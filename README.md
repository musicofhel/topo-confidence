# topo-confidence

**Hidden-state geometry as a window into LLM reasoning.**

Three weeks of experiments (April 2026, Pathways 1–11) trying to predict whether a language model's chain-of-thought answer is correct from its residual-stream activations alone. The project began as "persistent homology of token clouds predicts correctness," and ended somewhere very different.

## Where the project actually landed

The original "topology" framing was overturned. Persistent-homology features on trained-model residual streams sit at the rank-matched Gaussian null (AUROC 0.690 vs null 0.693, [F-10](FINDINGS.md)). What remained, after correcting label and truncation bugs, is a much simpler signal:

| Quantity | Value | Source |
|---|---|---|
| **Prefill L19 DoM AUROC** (Qwen-2.5-1.5B, K=1 correctness, OOF 5-fold) | **0.7731** | [F-2](FINDINGS.md) |
| Final-token L19 DoM AUROC (same setup) | 0.7186 | [F-2](FINDINGS.md) |
| cos(prefill_DoM, final_DoM) | **0.046** | [F-3](FINDINGS.md) |
| Prefill-gated selective prediction at coverage 0.5 | **71.6% acc on answered**, K=2.5 avg | [F-8](FINDINGS.md) |
| Unconditional baseline at K=1 | 48.6% | [F-8](FINDINGS.md) |

**Headline.** A *single direction* in the prefill-position L19 residual stream — fit by logistic regression on K=1 correctness labels — predicts whether the model will get the answer right *better than the final-token activations after the model has reasoned*. Refusing the bottom half by this score and spending the saved compute on the top half gives 71.6% accuracy on what's answered, vs 48.6% unconditional.

## Findings that survived controls

Full registry in [FINDINGS.md](FINDINGS.md). Strength rating reflects model count + control coverage.

- **F-1 (STRONG):** Dimensional breathing — residual-stream covariance participation ratio rises during CoT generation and collapses at the answer token. Universal across **Qwen-1.5B, Qwen-7B, Phi-3-mini, Llama-3.2-1B**. Random-token control is flat (PR ≈ 10).
- **F-2 (MODERATE):** Prefill DoM is a stronger correctness predictor than final-token DoM. ~0.876 on 7B, 0.7731 on 1.5B.
- **F-3 (MODERATE):** Prefill and final-token DoM directions are geometrically orthogonal (cos ≈ 0.046). "Can I solve this?" and "did I solve this?" live in unrelated subspaces.
- **F-4 (STRONG):** Correct trajectories collapse harder than incorrect at the final token (4 models, bootstrap CIs).
- **F-5 (MODERATE):** Breathing is content-dependent, not AR-mechanics. Random tokens give flat PR.
- **F-6 (STRONG):** 7B prefill PR inversion (correct-group PR > incorrect-group). Driven by easy-level failures clustering. Not in 1.5B, not in BBH.
- **F-7 (MODERATE):** D-bucket (K=1-right but K=8-majority-wrong) has a group-level signature but doesn't localize per-problem.
- **F-8 (MODERATE):** Selective prediction via prefill DoM works at 50% coverage. +22 pp over unconditional at lower compute.
- **F-9 (MODERATE):** CoE-60 trajectory features are redundant with single-layer L19 DoM (Δ = +0.004 in favor of DoM on cross-domain transfer).
- **F-10 (STRONG):** Persistent homology on trained residual streams sits at the Gaussian null. PH = covariance dressed differently.

## What was overturned

Full graveyard at [PROJECT_RECORD §1d](PROJECT_RECORD.md). The big ones:

- **The 20.8% MATH-500 accuracy baseline** was a `max_new_tokens=256` truncation artifact. At 1024 tokens the 1.5B gets 48.6%, not 20.8%.
- **The 0.796 ABC-44 / topo-AUROC headline** was computed against the truncated label distribution. At 1024 tokens, single-direction DoM gets 0.7731 — and the topology-specific signal in the 44-feature pipeline is fully explained by covariance structure (F-10).
- **Cross-scale "7B is a stronger verifier."** At 1024 tokens, 7B → 1.5B transfer is 0.717 vs 1.5B's self-prediction at 0.719 — no asymmetry.
- **CoE-60 > ABC-44.** Both numbers (0.811 vs 0.7961) were on truncated labels. At 1024 tok, single-direction DoM matches CoE on transfer (Δ = +0.004 in favor of DoM).
- **Fixed-vector steering (Pathway 2, Pathway 10 E1).** The "correctness axis" rotates through generation; injecting a final-token-fit vector at position 15 is geometrically random.

## Reading order

If you've never seen this project before, read in this order:

1. **[QUICKSTART.md](QUICKSTART.md)** (~480 words) — the orientation document. What we found, what we were wrong about, where the data lives.
2. **[STATE.md](STATE.md)** — where the most recent session left off. Overwritten each session.
3. **[PROJECT_RECORD.md](PROJECT_RECORD.md)** — authoritative archive. §1a chronology, §1b provenance table (every claim → JSON), §1d graveyard, §1e queue.
4. **[FINDINGS.md](FINDINGS.md)** — F-1…F-10 registry with controls.
5. **[HYPOTHESES.md](HYPOTHESES.md)** — H-1…H-14 prioritized queue with cost estimates.
6. **[PERSPECTIVES.md](PERSPECTIVES.md)** — reflective notes on what surprised, what was wrong.
7. **[DATA_MANIFEST.md](DATA_MANIFEST.md)** — cached activation inventory (~79 GB, gitignored).
8. **[PAPER_INDEX.md](PAPER_INDEX.md)** — external papers that informed the work, with our REPLICATED / CONTRADICTED status.

Smoke test: `python validate_claims.py` — 91 quantitative claims back-checked against committed JSONs. Should print 91/91 PASS.

## What's next

Top of [HYPOTHESES.md](HYPOTHESES.md):

| H | Cost | Decides |
|---|---|---|
| H-2 short-CoT breathing | ~$0.25 | Whether breathing tracks CoT length or reasoning structure |
| H-6 CoE re-baseline at 1024 tok | $0 / 30 min CPU | Whether CoE-beats-DoM headline survives label correction |
| H-14 Qwen breathing headline figure | $0 / 30 min CPU | Generates the missing PNG that should anchor the project |
| H-1 per-position DoM steering | ~$200 / 3 H100-days | The most decisive: diagnostic vs lever |

H-1 is the question that decides the program: if a *position-aware* DoM bank steers MATH-500 accuracy by ≥ 3 pp, the prefill signal is an actionable lever. If not, the program closes at "good selective predictor."

## Code

The `topo_confidence/` Python package (v0.2.0) is the *original-framing* reference implementation — 44-feature ABC pipeline + logistic regression + CLI. The headline result it computes (AUROC 0.796 on MATH-500) is the now-overturned 256-tok number, kept for reproducibility of the historical claim.

```bash
pip install -e .
```

```python
from topo_confidence import TopoConfidence
tc = TopoConfidence("Qwen/Qwen2.5-1.5B-Instruct")
tc.calibrate(calibration_prompts, calibration_labels)
confidences = tc.predict_confidence(["What is 2+2?", "Prove the Riemann Hypothesis"])
```

The current findings (prefill DoM, breathing, gated compute) live in the pathway directories — there's no packaged API for them yet. See [`pathway11_h100/`](pathway11_h100/) for the most recent extraction + analysis pipelines.

## Repo layout

```
QUICKSTART.md, STATE.md          # Read first
PROJECT_RECORD.md                # Authoritative archive
FINDINGS.md, HYPOTHESES.md       # Live claims and queue
PERSPECTIVES.md                  # Reflective notes
DATA_MANIFEST.md                 # ~79 GB cached NPZ inventory
EXPERIMENT_LOG.md                # EXP-001…EXP-042 append-only log
PAPER_INDEX.md                   # External-paper bridge
validate_claims.py               # Provenance smoke test (91/91 PASS)

topo_confidence/                 # Pip-installable package (v1 framing)
pathway1/ … pathway11_h100/      # Per-pathway code, results JSONs, NPZ caches
figures/                         # Headline figures (see figures/README.md)
scratch/                         # Pathway 10 sanity-check JSONs (cited as evidence)
archive/                         # Superseded design docs and pre-rebuild scripts
data/, configs/, tests/          # Working data and harness
```

## License

MIT
