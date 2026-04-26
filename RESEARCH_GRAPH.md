# Research Graph

This repo ships a Neo4j knowledge graph linking the project's experimental
findings to the external literature. It runs in its own container alongside
(but separate from) the `link-forge` paper database.

- **Research graph** — `bolt://localhost:7688`, auth `neo4j / topo_graph_dev`
- **link-forge** (read-only sibling, ~700 arxiv links) — `bolt://localhost:7687`

The two databases never write to each other; the bridge resolves Paper stubs
in this graph (just an arxiv_id + relevance note) to full metadata in
link-forge on demand.

## Quick start

```bash
cd research-graph
docker compose up -d
pip install -r requirements.txt
python seed.py                    # first time only
python query.py status-report
```

Re-running `seed.py` is idempotent (uses MERGE). To wipe and reseed:

```bash
python seed.py --reset
```

## Key queries

```bash
# What corroborates the breathing finding?
python query.py corroborators F-1

# Is a claim already in the graph?
python query.py novelty "prefill direction is orthogonal to final-token"

# Full neighborhood of a finding (LLM-friendly JSON)
python query.py subgraph F-2 --depth 2

# Which papers propose experiments that extend a finding?
python query.py extensions F-3

# Where does a finding sit in the dependency web?
python query.py related F-6

# Pathway timeline
python query.py timeline

# All findings grouped by status, with pending controls
python query.py status-report

# Pull a paper with its incoming edges
python query.py paper 2410.13640
```

## Bridge to link-forge

```bash
# Resolve a single arxiv ID
python bridge.py resolve 2410.13640

# Copy title + forgeScore for every Paper stub
python bridge.py enrich

# Discover papers in link-forge tagged with X that are NOT in this graph
python bridge.py ungraphed steering
```

## Schema

See `schema.cypher` for the full constraint + index list. The relationship
vocabulary documented at the bottom of that file is the source of truth.

Node labels:
- `:Pathway`  (P1..P11) — chronological backbone
- `:Experiment`  (P9-E1, P11-E3, …) — concrete tests
- `:Finding`  (F-1..F-14) — testable claims with strength + controls
- `:Paper`  — arxiv stubs; full metadata lives in link-forge
- `:Artifact`  — result JSONs and other files referenced by experiments
- `:Tag`  — topical groupings (e.g. `steering`, `breathing`, `prefill`)

The five edges that carry the most value:
- `(Finding)-[:CORROBORATED_BY]->(Paper)` — independent confirmation
- `(Finding)-[:CONTRADICTED_BY]->(Paper)` — published disagreement, with `why` + `resolution`
- `(Finding)-[:EXTENDED_BY]->(Paper)` — experiment ideas, often `actionable=true`
- `(Finding)-[:METHOD_DIFFERS]->(Paper)` — same conclusion, different method
- `(Finding)-[:EXPLAINS]->(Paper)` — paper supplies the mechanism

## Adding new things

```bash
# 1. Record the finding in FINDINGS.md, pick the next F-number
# 2. Add it to the graph
python add_finding.py --id F-15 --claim "..." \
  --strength MODERATE --evidence P11-E10 \
  --counterargument "..." --overturned-by "..." --tags steering prefill

# 3. Add a paper stub if it's not in the graph yet
python add_paper.py --arxiv 2501.12345 --title "..." --year 2025 \
  --repo https://github.com/... --relevance "..." --tags steering

# 4. Wire them together
python add_edge.py F-15 CORROBORATED_BY 2501.12345 \
  --note "Replicates on Llama-3 8B."

# 5. (Optional) pull title + forgeScore from link-forge
python bridge.py resolve 2501.12345
python enrich.py
```

## Connection details

| Service           | Bolt                  | HTTP                  | Auth                       |
|-------------------|-----------------------|-----------------------|----------------------------|
| Research graph    | bolt://localhost:7688 | http://localhost:7475 | neo4j / topo_graph_dev     |
| link-forge        | bolt://localhost:7687 | http://localhost:7474 | neo4j / link_forge_dev     |

The `.env` file in `research-graph/` is loaded by every script — override
ports/auth there if needed.

## Future experiments

The graph also tracks planned experiments across all 11 pathways. Each
`:FutureExperiment` node has a priority score (1–10), trigger conditions, and
edges to the findings it depends on / would update and the papers that triggered
it. Status flow: `READY` → `TRIGGERED` (paper appeared) → `COMPLETED` /
`ABANDONED`. Completed experiments stay in the graph with an `outcome` field —
the graph preserves the full lifecycle, never deletes.

```bash
# What should I work on next?
cat ~/topo-confidence/NEXT_EXPERIMENTS.md

# Or query the graph directly:
cd research-graph
python query.py highest-roi 10          # top 10 by ROI
python query.py triggered                # papers have made these actionable
python query.py blocked                  # waiting on resources / prior experiments
python query.py future P11               # all future experiments for one pathway
python query.py impact P3-FE1            # full impact JSON for one experiment
python query.py watchlist                # papers to monitor for new triggers
```

### Adding a future experiment

```bash
python add_future_experiment.py \
  --id P11-FE6 --pathway P11 \
  --description "..." --rationale "..." --trigger "..." \
  --status READY --priority HIGH --roi 7 --cost "4h CPU" \
  --depends-on F-1 F-2 \
  --would-update F-1 \
  --triggered-by 2501.12345
```

### Updating status

```bash
# Promote READY → TRIGGERED when a relevant paper appears
python update_status.py P5-FE1 TRIGGERED

# Mark as completed with an outcome
python update_status.py P3-FE1 COMPLETED \
  --outcome "PID steering hit 51.2% acc, +2.6 over base."
```

### Regenerating the priority queue

```bash
cd research-graph && python generate_next_experiments.py
```

This overwrites `~/topo-confidence/NEXT_EXPERIMENTS.md` (a snapshot, not a live
view). Commit the result so the queue is reviewable in PR diffs.

## What this graph is *not*

- Not a copy of link-forge — Paper nodes are stubs (arxiv_id, title,
  relevance_note). Abstracts and citations stay in link-forge.
- Not a research log — `EXPERIMENT_LOG.md` and the per-pathway handoff
  files remain authoritative for the chronological narrative.
- `HYPOTHESES.md` and `:FutureExperiment` overlap by design: HYPOTHESES.md is
  the human-readable narrative queue, FutureExperiment is the queryable
  structured version. When they disagree, update both.
