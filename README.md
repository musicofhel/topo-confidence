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

## Research path & literature

Each pathway below is one chronological step of the program, paired with the external papers that informed or were tested by it. Status tags follow [PAPER_INDEX.md](PAPER_INDEX.md): ✅ replicated on our data, ❌ contradicted, ≈ partially confirmed, • cited / motivating only. Authoritative finding↔paper edges live in `research-graph/seed.py` (`PAPER_EDGES`).

```mermaid
flowchart LR
    P1[P1 · CORAL] --> P2[P2 · Steering v1]
    P2 --> P3[P3 · Complexity]
    P3 --> P4[P4 · ABC-44 + Track-A]
    P4 --> P5[P5 · Cross-bench]
    P5 --> P6[P6 · Rebuild + 6.5]
    P6 --> P7[P7 · Non-Eucl PH]
    P7 --> P8[P8 · Layer-wise]
    P8 --> P9[P9 · CoE pivot + audit]
    P9 --> P10[P10 · Steering plan v2]
    P10 --> P11[P11 · H100 + headline]
```

### P1 — CORAL feature extraction *(closed)*
Ported the 78-feature CORAL pipeline from ATT and froze it as the 44-feature ABC extractor — ancestor of every later feature pipeline. **Papers:** [1].

### P2 — Steering v1 *(closed, superseded)*
Track A spherical steering hit "+26 net gain" at 256-tok; Pathway 11 later showed the cached vector has cos ≈ 0.05 with the current L19 correctness axis. **Papers:** [2].

### P3 — Complexity expansion *(closed, NO-GO)*
Multi-layer non-adjacent features + K-NN prototypes didn't help. PIVOT. **Papers:** [4][5].

### P4 — ABC-44 + Track-A selection *(closed)*
Track A topo-guided test-time selection at +11 net gain (256-tok); Track B learned steering null. **Papers:** [6][7][8][9][10].

### P5 — Cross-benchmark / cross-model *(closed, superseded)*
GSM8K + 7B transfer "worked" at 256-tok and was later refuted by Phase 6.5; no external paper directly informed this test. **Papers:** —

### P6 — Rebuild + Phase 6.5 deconfounding *(closed)*
Caught the 256-tok truncation bug. At 1024 tokens cross-benchmark transfer collapses to chance (0.504). **Papers:** [13].

### P7 — Non-Euclidean PH *(closed, NO-GO)*
Diffusion / effective-resistance / cosine / DTM metrics hit 0.774 vs the 0.7961 Euclidean baseline. **Papers:** [10][11][12].

### P8 — Layer-wise PH + alternatives *(closed)*
Layer-wise PH (168-dim) = 0.646 < single-layer; CoE-60 = 0.811 beats ABC-44 = 0.7961 (256-tok). **Papers:** [6][7][9][10].

### P9 — CoE pivot + PH audit *(closed)*
PH at Gaussian null (0.690 vs 0.693). CoE-60 the only domain-invariant signal. LR ≈ XGBoost — signal is linear. **Papers:** [6][8][9][10][11][12][14][15][16].

### P10 — Goal pivot to steering *(active, plan-only)*
v2 plan: E1 ITI-style steering, E2 prefix-CoE gating, E3 hidden-state refusal, E4 big→small distillation. **Papers:** [2][3][13][17][19][23][24][25][26][27][37][38].

### P11 — H100 re-extract + headline *(active)*
Truncation discovery (48.6%, not 20.8%). Prefill DoM 0.7731. Direction rotates with position. Selective prediction 71.6% accuracy at 50% coverage. Dimensional breathing universal across Qwen-1.5B, Qwen-7B, Phi-3-mini, Llama-3.2-1B. **Papers:** [4][5][14][17][18][19][20][21][22][23][24][25][27][28][29][30][31][32][33][34][35][36].

### References

[1] *ATT Phase 5* — internal predecessor (att-docs repo, seed for Pathway 1 CORAL). ≈

[2] Li et al. 2023, *Inference-Time Intervention* — [arXiv:2306.03341](https://arxiv.org/abs/2306.03341). ❌

[3] *Adaptive Layer-wise Steering (ALS)* — [arXiv:2509.18116](https://arxiv.org/abs/2509.18116). •

[4] Tan, *Generalization of Steering Vectors* — [arXiv:2407.12404](https://arxiv.org/abs/2407.12404). •

[5] *Small Vectors, Big Effects* — [arXiv:2509.06608](https://arxiv.org/abs/2509.06608). ✅

[6] Wang et al. ICLR 2025, *Chain-of-Embedding (CoE)* — [arXiv:2410.13640](https://arxiv.org/abs/2410.13640). ≈

[7] Pope et al., *Intrinsic Dimension via TwoNN* (classic). ❌

[8] Chen et al. ICLR 2024, *INSIDE / EigenScore* — [arXiv:2402.03744](https://arxiv.org/abs/2402.03744). •

[9] *D²HScore* — [arXiv:2509.11569](https://arxiv.org/abs/2509.11569). ≈

[10] *Persistent Topological Features in LLMs* — [arXiv:2410.11042](https://arxiv.org/abs/2410.11042). •

[11] Birdal, *ID, PH and Generalization* — [arXiv:2111.13171](https://arxiv.org/abs/2111.13171). •

[12] *Truthfulness via Local ID* — [arXiv:2402.18048](https://arxiv.org/abs/2402.18048). •

[13] Chen et al., *Cross-scale stitching* — [arXiv:2506.06609](https://arxiv.org/abs/2506.06609). •

[14] Tuci et al., *Sharpness Dimension / EoS* — [arXiv:2604.19740](https://arxiv.org/abs/2604.19740). ≈

[15] *A Long Way to Go (length in RLHF)* — [arXiv:2310.03716](https://arxiv.org/abs/2310.03716). ✅

[16] *Between Underthinking and Overthinking* — [arXiv:2505.00127](https://arxiv.org/abs/2505.00127). ✅

[17] Zhu, *The LLM Already Knows* — [arXiv:2509.12886](https://arxiv.org/abs/2509.12886). ✅

[18] *LLMs Encode Problem Difficulty* — [arXiv:2510.18147](https://arxiv.org/abs/2510.18147). ✅

[19] Zhang et al., *Reasoning Models Know When They're Right* — [arXiv:2504.05419](https://arxiv.org/abs/2504.05419). •

[20] *LLMs Know More Than They Show* — [arXiv:2410.02707](https://arxiv.org/abs/2410.02707). •

[21] Marks & Tegmark, *Geometry of Truth* — [arXiv:2310.06824](https://arxiv.org/abs/2310.06824). ✅

[22] Park et al., *Linear Representation Hypothesis* — [arXiv:2311.03658](https://arxiv.org/abs/2311.03658). •

[23] *PID Steering* — [arXiv:2510.04309](https://arxiv.org/abs/2510.04309). •

[24] *STU-PID Steering* — [arXiv:2506.18831](https://arxiv.org/abs/2506.18831). •

[25] *Knowing When to Quit* — [arXiv:2604.18419](https://arxiv.org/abs/2604.18419). ✅

[26] *CCPS Calibration* — [arXiv:2505.21772](https://arxiv.org/abs/2505.21772). •

[27] OATML, *Semantic Entropy Probes (SEPs)* — [arXiv:2406.15927](https://arxiv.org/abs/2406.15927). •

[28] Mohri & Hashimoto, *Conformal Factuality* — [arXiv:2402.10978](https://arxiv.org/abs/2402.10978). •

[29] Brumm et al., *Inference-time scaling of CoT* — [arXiv:2510.07364](https://arxiv.org/abs/2510.07364). ≈

[30] Wang et al. ICLR 2023, *Self-Consistency* — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171). ✅

[31] *High-Dim Abstraction Phase* — [arXiv:2405.15471](https://arxiv.org/abs/2405.15471). ✅

[32] *Linguistic Collapse* — [arXiv:2405.17767](https://arxiv.org/abs/2405.17767). ✅

[33] *Geometry of Hidden Representations* — [arXiv:2302.00294](https://arxiv.org/abs/2302.00294). •

[34] *LLM Reasoning as Trajectories* — [arXiv:2604.05655](https://arxiv.org/abs/2604.05655). •

[35] *EigenTrack* — [arXiv:2509.15735](https://arxiv.org/abs/2509.15735). •

[36] *Attention Sinks = Compression Valleys* — [arXiv:2510.06477](https://arxiv.org/abs/2510.06477). •

[37] *ReDeEP + AARF* — [arXiv:2410.11414](https://arxiv.org/abs/2410.11414). •

[38] *Brain-Grounded Axes* — [arXiv:2512.19399](https://arxiv.org/abs/2512.19399). •

## License

MIT
