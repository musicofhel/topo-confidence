# CLAUDE.md — agent orientation

This is the agent-facing guide. Humans should read [README.md](README.md) first, then [QUICKSTART.md](QUICKSTART.md).

## Source of truth

Always read these before answering questions about project state — they are kept current and supersede anything you might infer from older files:

| File | What it has | When to read |
|---|---|---|
| [STATE.md](STATE.md) | Where the most recent session left off, pod status, top 3 next experiments | First, every session |
| [QUICKSTART.md](QUICKSTART.md) | ~480-word orientation: what we found, what was wrong, where data lives | If you're cold on the project |
| [PROJECT_RECORD.md](PROJECT_RECORD.md) | Authoritative archive. §1a chronology, §1b provenance table, §1c reproducibility, §1d graveyard, §1e queue, §1f literature, §1g file inventory | When any claim needs verification |
| [FINDINGS.md](FINDINGS.md) | F-1…F-10 registry with controls and counterarguments | When discussing what's been established |
| [HYPOTHESES.md](HYPOTHESES.md) | H-1…H-14 prioritized queue with cost estimates | When proposing next experiments |
| [PERSPECTIVES.md](PERSPECTIVES.md) | Reflective notes — what surprised, what was wrong | For framing/narrative |
| [DATA_MANIFEST.md](DATA_MANIFEST.md) | NPZ schema, sizes, regeneration commands | Before claiming a cache exists |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | EXP-001…EXP-042 append-only log | When tracing where a number came from |
| [PAPER_INDEX.md](PAPER_INDEX.md) | External papers, REPLICATED / CONTRADICTED / TO TEST | Before citing literature |
| [RESEARCH_GRAPH.md](RESEARCH_GRAPH.md) | Neo4j knowledge graph linking findings to literature | Before claiming novelty / planning experiments / citing prior work |
| [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) | Auto-generated priority queue from `:FutureExperiment` nodes | When picking what to work on next |

## Research graph

A Neo4j knowledge graph at `bolt://localhost:7688` (auth `neo4j / topo_graph_dev`)
links this project's findings (F-1…F-14) to ~40 external papers via typed edges
(CORROBORATED_BY, CONTRADICTED_BY, EXTENDED_BY, METHOD_DIFFERS, EXPLAINS). It is
separate from link-forge (`bolt://localhost:7687`); Paper nodes here are arxiv
stubs that resolve to link-forge for full metadata.

```bash
# Before claiming a finding is novel:
cd ~/topo-confidence/research-graph && python query.py novelty "<one-line claim>"

# Before planning a new experiment:
cd ~/topo-confidence/research-graph && python query.py extensions <F-id>

# What's validated, what's pending:
cd ~/topo-confidence/research-graph && python query.py status-report

# Full neighborhood of a finding (LLM context-friendly JSON):
cd ~/topo-confidence/research-graph && python query.py subgraph <F-id> --depth 2

# Resolve an arxiv ID against link-forge for forgeScore + title:
cd ~/topo-confidence/research-graph && python bridge.py resolve <arxiv-id>
```

Key files: `research-graph/seed.py` (authoritative seed data),
`research-graph/query.py` (CLI), `research-graph/bridge.py` (link-forge resolver).
Start with: `cd research-graph && docker compose up -d`.

### Next experiments

`:FutureExperiment` nodes track planned work across all 11 pathways with ROI scores,
trigger papers, and depends-on/would-update edges to findings. The priority queue
is rendered to `NEXT_EXPERIMENTS.md` at the repo root.

```bash
# Before starting any new experiment, check the priority queue:
cat ~/topo-confidence/NEXT_EXPERIMENTS.md

# Before planning a new experiment, check if it already exists:
cd ~/topo-confidence/research-graph && python query.py future <pathway-id>

# After completing an experiment, update its status and refresh the queue:
cd ~/topo-confidence/research-graph
python update_status.py <fe-id> COMPLETED --outcome "..."
python generate_next_experiments.py    # rewrites NEXT_EXPERIMENTS.md

# After reading a new paper, check if it triggers anything:
python query.py watchlist
```

## Smoke test

```bash
python validate_claims.py > validation_report.txt
```

91 quantitative claims back-checked against committed JSONs. **91/91 PASS as of 2026-04-24.** If you've changed any number in the narrative docs, you must update the matching `Claim` entry in `validate_claims.py` and re-run.

## What the project is, in two sentences

Three weeks of experiments testing whether residual-stream geometry predicts LLM correctness. The "topological homology" framing was overturned (PH = Gaussian null, F-10); what survived is a single L19 prefill direction (DoM) that predicts correctness at AUROC 0.7731 on Qwen-2.5-1.5B and enables 71.6% selective-prediction accuracy at 50% coverage.

## Headline numbers (1024-tok labels, the only ones that aren't truncation-confounded)

| Quantity | Value | Source |
|---|---|---|
| 1.5B MATH-500 K=1 accuracy | 48.6% (243/500) | `pathway11_h100/prefill_gated_compute/results.json` |
| 7B MATH-500 K=1 accuracy | 73.2% (366/500) | `pathway11_h100/prefill_gated_compute/results.json` |
| Prefill L19 DoM AUROC (1.5B, OOF 5-fold) | 0.7731 | same |
| Final-token L19 DoM AUROC (1.5B) | 0.7186 | same |
| cos(prefill_DoM, final_DoM) | 0.046 | `scratch/pathway10_temporal_and_verifier_results.json` |
| Selective-prediction acc at coverage 0.5 | 71.6% on answered, K=2.5 avg | same |

The 0.796 ABC-44 number, the 20.8% baseline, and any cross-scale "7B is a stronger verifier" claim are all 256-tok truncation artifacts — superseded.

## Repo layout

```
README.md, QUICKSTART.md, STATE.md, CLAUDE.md
PROJECT_RECORD.md, FINDINGS.md, HYPOTHESES.md, PERSPECTIVES.md
DATA_MANIFEST.md, EXPERIMENT_LOG.md, PAPER_INDEX.md
validate_claims.py + validation_report.txt

topo_confidence/                 # Pip-installable package (v0.2.0, v1-framing reference impl)
pathway1/ … pathway11_h100/      # Per-pathway code + result JSONs (NPZ binaries gitignored)
figures/                         # Headline PNGs, see figures/README.md for captions
scratch/                         # Pathway 10 sanity-check JSONs — cited as evidence in PROJECT_RECORD §1b
archive/                         # Superseded design docs and pre-rebuild scripts
configs/, data/, tests/          # Harness + working data
```

## Pathway timeline (very short)

Full chronology in PROJECT_RECORD §1a. One-liner per pathway:

- **P1** CORAL feature extraction → frozen ABC-44 extractor.
- **P2** Spherical steering → Pathway 10 v1 later showed the steering vector was geometrically unrelated to the current DoM direction.
- **P3** Complexity expansion → didn't help.
- **P4** Track A topo-guided selection → +11 net gain (PUBLICATION_READY at the time).
- **P5** Cross-benchmark / cross-model → "transfer works" (later truncation-confounded).
- **P6 rebuild** Corrected labels + Phase 6.5 deconfounding → caught the 256-tok bug.
- **P7** Non-Euclidean PH → NO-GO (0.774 < 0.796).
- **P8 layer-wise** Per-layer PH across all 28 layers → modest lift, then null vs Gaussian.
- **P9** CoE pivot → CoE-60 = 0.811 (later shown matched by single-layer DoM).
- **P10** v1 five directions, v2 steering program → direction-rotation refutes fixed-vector steering; prefill/final orthogonality emerges.
- **P11** H100 re-extract at 1024 tok → current findings, 7 stage markers, exp1_cross_model + gibberish + no-CoT controls.

## Working norms

- **Numbers come from JSONs, not narrative docs.** If you see a discrepancy, the JSON wins and the narrative doc gets updated.
- **Label scheme matters.** Tag every number you cite with `1024tok` (current canonical), `256tok_NEW`, or `256tok_manifest`. The 256-tok numbers are not directly comparable to 1024-tok — see PROJECT_RECORD §1d.
- **Don't add a new claim without updating `validate_claims.py`.** The 91/91 invariant is load-bearing.
- **NPZ caches are gitignored.** Regeneration commands are in DATA_MANIFEST.md. Don't assume a cache exists on a fresh machine.

## RunPod

Always use H100 SXM. Pod state is in STATE.md (currently: `lsuoka6bo8io7m` stopped with volume preserved; `y687b9z2dgukcj` removed). See `~/.claude/projects/-home-musicofhel/memory/runpod-preferences.md` for general defaults.
