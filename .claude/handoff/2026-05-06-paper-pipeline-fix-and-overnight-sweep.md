# Handoff — 2026-05-06 — Paper Pipeline Fix + Overnight Experiment Sweep

## What happened this session

### Problem
35 arxiv papers posted to link-forge Discord over the last 4 days produced only 2 research-graph candidates and 0 experiments. Two bottlenecks:

1. **Content-type gate** (`link-forge/src/processor/index.ts:282`): only `content_type === "research-paper"` triggered the research-graph bridge. The Claude classifier labeled 24/35 papers as `analysis` and 3 as `reference`, so they never reached the relevance check.

2. **Relevance prompt** (`link-forge/src/processor/claude-cli.ts`): focused on citation relationships ("could this be cited under CORROBORATED_BY / CONTRADICTED_BY?"), missing methodology transfers and abstract experiment potential.

### Fixes applied (3 files)

1. **`link-forge/src/processor/index.ts`** — Widened gate: any URL matching `\d{4}\.\d{4,5}` (arxiv pattern) now goes through the relevance check regardless of content_type. `suggestToResearchGraph` still gates on config + arxiv ID internally.

2. **`link-forge/src/processor/claude-cli.ts`** — Rewrote relevance prompt from "could this be cited?" to "could this generate an experiment?" Added explicit two-tier thinking (cheap local 5min-2hr on cached NPZs, high-ROI H100), widened CLEAR FITS to include manifold learning, information geometry, spectral methods, geometric statistics. False positive cost reframed as $0.05 triage pass vs permanent loss.

3. **`topo-confidence/research-graph/_paper_triage_prompt.template`** — Added two-tier FutureExperiment guidance: Tier 1 (cheap local, CPU, every paper must produce at least one), Tier 2 (H100, only if ROI ≥ 7). Describes the cached 500×1536 activation matrix explicitly.

4. **`link-forge/scripts/backfill-research-graph-suggest.ts`** — Added `--since YYYY-MM-DD` flag for date-filtered backfills.

### Results

- Backfill re-evaluated 37 papers with new prompt: **17 new admissions** (old prompt: 2/35 = 5.7%, new prompt: 17/~25 freshly evaluated = 68%)
- `triage_pending.sh` ran all 20 pending papers (3 parallel workers): **20/20 triaged and promoted, zero failures**
- **63 new FutureExperiment nodes** merged into research-graph: 52 Tier 1 (CPU), 11 Tier 2 (H100)
- Total READY experiments now: **887** (was 819)
- Link-forge bot restarted with new code, running as PID 74036

## Next session: overnight experiment sweep

**User wants to pick ~10 cheap CPU experiments from today's 52 Tier 1 and run them overnight.**

**Hardware constraint: NO H100 access. Only a GTX 2060 Super (8GB VRAM). All experiments must be CPU-only or light enough for 2060S.**

### Top 10 candidates by ROI (all CPU, all READY)

| ID | ROI | Cost | Description |
|---|---|---|---|
| P11-FE901 | 9 | 30min CPU | Project DoM + PC1 onto SETOL ECS basis of L19 weights |
| P11-FE930 | 8 | 45min CPU | Hyvärinen score difference as per-sample correctness feature |
| P11-FE925 | 8 | 15min CPU | Spearman correlation: DoM scores vs prefill sequence length |
| P11-FE26841a | 8 | 30min CPU | Logit-lens entropy from cached L19 prefill activations |
| P11-FE01172B | 8 | 10min CPU | Signal-channel rank at L19 via Gram matrix eigenspectrum |
| P11-FE919 | 8 | 20min CPU | Alignment-score layer profile: cos(mean_diff, DoM) per layer |
| P11-FE909 | 8 | 15min CPU | Intrinsic dimension estimation via MLE (Levina-Bickel) |
| P11-FE903 | 8 | 10min CPU | mCCA between L19 prefill and final-token activations |
| P11-FE899 | 8 | 20min CPU | WeightWatcher per-layer alpha profile vs DoM AUROC |
| P11-FE889 | 8 | 20min CPU | Project DoM onto weight-Procrustes rotation axes |

Total estimated runtime: ~3.5 hours sequential. Could parallelize some.

### How to run

1. Read STATE.md for current pod/data status
2. Verify cached NPZs exist per DATA_MANIFEST.md (these experiments all use cached L19 prefill activations)
3. For each FE, read its full brief in `research-graph/briefs/triage-2026-05-06-*.md` (search for the FE ID)
4. Implement the experiment, save results JSON to `pathway11_h100/` (or appropriate pathway dir)
5. Write a result brief: `research-graph/briefs/result-<FE-ID>.md`
6. Update status: `python research-graph/update_status.py <FE-ID> COMPLETED --outcome "..."`
7. Regenerate queue: `python research-graph/generate_next_experiments.py`

### Note on FE26841a (logit-lens entropy)
This one needs the unembedding matrix from the model weights — may need a quick `transformers` load on CPU or 2060S. Should be fine for Qwen-2.5-1.5B (3GB weights).

### Pipeline health
- Link-forge bot: running (PID 74036), new code active
- Research-graph Neo4j: running (`bolt://localhost:7688`)
- Link-forge Neo4j: running (`bolt://localhost:7687`)
- `validate_claims.py`: not re-run after today's promotes — run before starting experiments
