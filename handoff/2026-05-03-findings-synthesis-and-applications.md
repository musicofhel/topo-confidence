# 2026-05-03 — Findings synthesis, novelty audit, applications

Written at the end of session 5 (FE291 follow-up cascade lands as commit 39c264f
on `max-depth-retriage-2026-04-28`). The user's ask: stop adding findings for a
beat and instead **synthesize what we have, what difference it made, what is
genuinely novel, and where it is most usefully applied.** This handoff is the
launching point for that synthesis after compaction.

## Where things stand at 2026-05-03

- **Active findings**: 14 (`F-1 … F-14`). FINDINGS.md is the registry, F-2 and
  F-10 were just updated by FE881/FE882/FE880.
- **Headline number, unchanged**: prefill L19 DoM AUROC = 0.7731 on
  Qwen-2.5-1.5B MATH-500 (1024-tok, 5-fold OOF). Selective prediction at 50%
  coverage = 71.6% answered accuracy.
- **FE291 triangle, freshly resolved**:

  | Probe | OOF AUROC | Δ vs DoM 0.7679 |
  |---|---|---|
  | 1-d DoM | 0.7679 | — |
  | 2-feat (PC1, PC9) | 0.7856 | +1.77 pp |
  | full 1536-d L2-reg (best C=0.001) | 0.7847 | +1.68 pp (≈ 2-feat) |
  | top-20 log-eigvals on PC1-residualized cov | **0.7928** | +2.49 pp |

  The cov-spectrum probe **beats every directional probe** including full
  1536-d L2-reg. Per-problem second-order structure is a real, separable
  correctness channel. **Genuinely novel candidate** — needs the novelty
  audit below to confirm.

- **F-10 strengthens, didn't get overturned**: PH features on PC1-residualized
  L19 clouds give real OOF 0.6907 vs matched-cov Gaussian null 0.7627
  (gap −0.072). The pre-registered F-10 overturning condition (real ≥ null +
  0.05) tested directly and failed by 12pt in the wrong direction. PH does
  not survive after PC1 removal; matched-Gaussian on the residualized cov
  *does* — which is what FE881 is reading.
- **Validation**: 172/172 internal PASS, 127/127 Tier-1 regen PASS, 0 FAIL,
  0 MISSING. 216 claims tracked total (172 internal + 41 external + 3
  PENDING_FE).
- **Discord/link-forge ingest is alive but the deep-pass loop has not run
  since 2026-04-29.** 12 admitted papers sit untriaged in Neo4j (1 visible
  via `query.py pending` as fresh admission `2604.28119`, 11 with null
  status that the CLI filter misses — see "Open queue" below).

## Three questions to answer (in order)

### Q1. What did we actually find, in one practitioner-readable narrative?

Goal: a single document — preliminary title `SYNTHESIS.md` — that a working ML
engineer could read in 10 minutes and walk away knowing what's true, what was
overturned, and what's new. Not a paper. A briefing.

**Inputs to read**:

- `FINDINGS.md` (F-1 … F-14)
- `STATE.md` "Last experiment completed" (will reference EXP-58 / FE882)
- `EXPERIMENT_LOG.md` EXP-50 onwards (FE291 cascade) and EXP-1..30 spot-check
- `PROJECT_RECORD.md` §1a chronology (one-line per pathway) and §1d graveyard
  (what got buried and why — 256-tok artifacts, fixed-vector steering refutation,
  PH vs Gaussian null)
- `PERSPECTIVES.md` for narrative framing
- Briefs in `research-graph/briefs/result-2026-05-02-P11-FE880|881|882.md`
  (most-recent results, well-formatted YAML headers)

**Structure to write**:

1. **What survived** — DoM at L19 prefill, selective prediction, the
   decomposition triangle (DoM → 2-feat → cov-spectrum). Numbers, label-scheme
   tags, and one sentence of mechanism each.
2. **What was overturned** — the topological-homology framing (F-10:
   PH = Gaussian null), the 256-tok ABC-44 0.796 / 20.8% baseline,
   fixed-vector steering (P10 v1).
3. **What is new at the end of this work** — the cov-spectrum probe. The PH-
   after-PC1-removal *strengthening* of F-10 (PH worse than null after
   residualization). The CAST PC1 ≈ supervised DoM unsupervised equivalence
   (cosine 0.922 from FE291).
4. **What we still don't know** — list 5–8 honest open questions. F-2 cross-
   architecture is in pending_controls. 7B vs 1.5B prefill PR ratios diverge
   (F-1) but mechanism unclear. Etc.

**Bar to clear**: every quantitative claim in `SYNTHESIS.md` must already be
in `validate_claims.py`. If you state a number, grep for its claim ID. No new
narrative numbers — synthesis only.

### Q2. What is genuinely novel against the 220-paper research-graph?

Goal: a `NOVELTY_AUDIT.md` that ranks each of our 14 findings on a 3-bucket
scale: `novel / refines-prior-work / parallel-discovery`. The research-graph
exists exactly for this purpose.

**Procedure**:

```bash
cd ~/topo-confidence/research-graph

# For each finding, run the canonical novelty check:
python query.py novelty "F-2: prefill hidden-state DoM at L19 predicts MATH correctness AUROC 0.7731"
python query.py novelty "decomposition triangle: 1-d direction → 2-feat → per-problem residualized cov spectrum, each tier strictly improves"
python query.py novelty "PH features fail to beat matched-cov Gaussian null on PC1-residualized clouds (gap -0.07)"
# … one per finding + one per FE291 triangle leg

# Also pull the neighborhoods for context:
python query.py subgraph F-2 --depth 2  | jq .   # what it's connected to
python query.py subgraph F-10 --depth 2 | jq .
```

**Strong candidates for "genuinely novel"** (predicted, to be confirmed):

- **Cov-spectrum probe (FE881)**: per-problem L19 covariance eigenvalue
  spectrum as a supervised correctness signal, after projecting out the
  dominant supervised direction. Closest neighbors: SAE feature sparsity
  papers, Hewitt-style probes, IRT-on-residual-streams. None we've seen
  use *per-problem* residualized covariance.
- **F-10 strengthening on residualized covariance**: PH summaries on a
  Gaussianized covariance test directly — most PH-vs-null papers test on
  the raw point cloud.
- **The 1024-tok rebuild itself** (F-13): publishing the truncation-artifact
  retraction openly is unusual; lots of TDA-LLM papers cite the 256-tok-era
  numbers without flagging.

**Bar to clear**: each "novel" claim cites a research-graph subgraph query
and lists the closest 3 prior-work papers it refines/replaces.

### Q3. Where is this most usefully applied?

Goal: a `APPLICATIONS.md` enumerating concrete settings where the L19 prefill
DoM + selective-prediction stack is *deployable*, *valuable*, and *honest about
its limits*.

**Suggested application axes** (start with these, expand from there):

1. **Selective serving for small-model APIs** — the immediate win. 71.6% acc
   at 50% coverage on a 1.5B model means: route 50% of MATH-style queries
   to the small model with confidence; spend 7B/large-model compute only
   on the rejected half. Cost model: 1024-tok prefill is one forward pass
   at L19 — trivial latency overhead.
2. **Self-consistency budget allocation** — F-14 says monotonic prefill
   gating fails because B-bucket lives at mid-confidence, but multi-signal
   gating (prefill + finaltok + length) is on the table and not yet tested.
   Worth a HYPOTHESIS entry.
3. **Refusal training** — F-2's prefill direction is a candidate for
   abstain-token logit modulation. Distinct from prompt-level refusal:
   this is a residual-stream signal at L19 prefill, before any answer is
   committed.
4. **Eval-time correctness estimator for synthetic data filtering** — if
   you're generating CoT data with a 1.5B model and want to keep the
   correctness-probable subset, the cov-spectrum probe at 0.7928 AUROC is
   a much better filter than a length heuristic.
5. **What it does NOT apply to (be honest)**:
   - Non-Qwen architectures — pending control on F-2.
   - Non-mathematical reasoning — F-2 was MATH-500, F-13 specifically
     called out the 256-tok confound on BBH.
   - Tasks where labels aren't available — DoM is supervised. CAST PC1 is
     unsupervised but its accuracy is the supervised DoM (0.7679), not the
     cov-spectrum (0.7928).

**Bar to clear**: each application has an honest "what would have to be true
for this to work in production" checklist. No hype.

## Open queue items (do NOT lose track of after compact)

1. **12 untriaged papers** in research-graph Neo4j:
   - 1 visible via `query.py pending`: `2604.28119` "Do Sparse Autoencoders
     Capture Concept Manifolds?" (relevant to H-13 + DoM framing)
   - 11 with null status that the CLI filter misses. Reconcile with:
     ```bash
     cd ~/topo-confidence/research-graph
     python -c "
     from neo4j import GraphDatabase
     d = GraphDatabase.driver('bolt://localhost:7688', auth=('neo4j', 'topo_graph_dev'))
     with d.session() as s:
         r = s.run('MATCH (p:Paper) WHERE p.status IS NULL RETURN p.arxiv_id, p.suggested_at ORDER BY p.suggested_at DESC')
         for rec in r: print(rec['p.arxiv_id'], rec['p.suggested_at'])
     "
     ```
   Then either fix link-forge ingest to set `status='pending_triage'`, or
   bulk-update existing nulls to `'pending_triage'`. Then `bash
   triage_pending.sh` clears the backlog.

2. **No FE-id on the synthesis / novelty / applications work.** This is
   meta-work, not a new experiment. Don't add a `:FutureExperiment` for it.
   The artifacts are documents, not result JSONs.

3. **Last triage loop**: 2026-04-29 (newest brief: `triage-2026-04-29-2604.24712.md`).
   ~4-day gap. Discord ingest still works.

## Resume points for post-compact session

If the next session is **cold** (compact happened):

1. Read this handoff first.
2. Read `STATE.md` for the latest "Last experiment completed" block (will
   reference EXP-58 / FE882).
3. Read `FINDINGS.md` quickly. F-2, F-10 just updated.
4. Pick a question (Q1, Q2, or Q3 above) and start writing the corresponding
   document. Q1 first is recommended — Q2 and Q3 lean on it.
5. Avoid kicking off new experiments before the synthesis lands. The user
   was explicit: *"I honestly want to see what we have found, what difference
   it made and if anything new and novel came out."*

If the next session is **warm** (compact didn't happen, this is being read
mid-session):

- Same plan, just skip step 1.

## Memory aids

- Branch: `max-depth-retriage-2026-04-28` (head: 39c264f)
- Latest commit: "FE291 follow-up cascade: promote FE880/FE881/FE882 (EXP-56,57,58)"
- Next EXP-id: **59** (do not regress)
- Next FE-id: **883** (FE880/881/882 are taken)
- validate_claims target: **172/172 internal PASS** — do not break
- Tier-1 regen target: **127/127 PASS** — do not break
- The PH cache (`pathway11_h100/ph_residuals/ph_cache.npz`, ~40 KB) is
  gitignored. Cold rebuild = ~38 min on this machine; warm load = ~5 s.
  If a fresh clone has no cache and validate_claims is invoked, the FE880
  regens will take 38 min on first pass — heads-up for CI / fresh boxes.

## What this handoff does NOT cover

- Pathway 12+ planning. Out of scope for the synthesis.
- Cross-architecture replication of F-2. Out of scope for the synthesis.
  Listed as an open question in Q1 step 4.
- Migration of the 12 null-status papers. Listed as queue item 1; pick up
  after the synthesis docs land.
