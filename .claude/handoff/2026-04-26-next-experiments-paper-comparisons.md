# Handoff — Add "What they did vs What we did" to NEXT_EXPERIMENTS.md

**Date:** 2026-04-26
**Status:** Plan approved, no code/data changes yet. Ready to execute.
**Approved plan file:** `~/.claude/plans/https-github-com-musicofhel-topo-confide-crystalline-feigenbaum.md` (read this first)

## What the user asked for

Per-experiment "what they did" vs "what we did" prose tied to source papers, with same/differs lines. The reason each experiment is queued should be sourced, not implicit.

## Key constraint that shapes everything

`NEXT_EXPERIMENTS.md` at the repo root is **auto-generated** from the Neo4j research graph by `research-graph/generate_next_experiments.py`. Direct edits to the markdown are wiped on regeneration. All changes must land in the graph + generator.

## Two Neo4j graphs to keep straight

| Port | Graph | Auth | Use |
|---|---|---|---|
| 7687 | link-forge (~219 papers, ~/link-forge) | `neo4j / link_forge_dev` | Source-of-truth for paper title/abstract/model/result |
| 7688 | topo-confidence research-graph | `neo4j / topo_graph_dev` | Findings F-1…F-14, FutureExperiments, Papers (stubs) |

User picked: **re-read each paper fresh from link-forge** (not mine PAPER_INDEX.md) — so query port 7687 first, write the comparison, then commit edge properties on port 7688.

## Three files to modify (in order)

1. `research-graph/schema.cypher` — add comment block documenting new `:TRIGGERED_BY` properties: `their_method`, `their_result`, `our_method`, `same`, `differs`.
2. `research-graph/generate_next_experiments.py` — fetch new edge props in Cypher (lines 38-58); add `render_source_paper()` helper; insert call inside `render_experiment()` after `**Why:**` and before `**Cost:**`. If `triggered_by` is empty, render `**Source:** Internal re-validation — no external trigger paper.` instead.
3. `research-graph/seed_future_experiments.py` — replace flat `"triggered_by_papers": ["2510.04309", ...]` with structured dicts holding the 5 comparison fields. Lines 1-120 read; **120-588 still need to be read** to enumerate all 24 experiment dicts.

The generator stays the only writer of NEXT_EXPERIMENTS.md.

## Trigger-paper inventory (~30 edges across 24 FEs)

- **CRITICAL (6):** P10-FE1 (4 papers: 2501.17148, 2504.07986, 2505.18706, 2306.03341), P11-FE5 (2501.12948), P3-FE1 (2510.04309, 2506.18831), P11-FE3 (2404.15255, 2412.01113). P8-FE1 + P9-FE1 = no triggers (re-validation).
- **HIGH (10):** P10-FE2 (2604.16217, 2402.10978), P10-FE3 (2410.04707), P11-FE2 (2510.18147), P4-FE2 (2402.18048), P8-FE2 (2504.05419), P7-FE1 (2410.11042, 2103.07353), P11-FE4 (2402.13212), P2-FE1 (2306.03341). P11-FE1 + P9-FE3 = no triggers.
- **MEDIUM (7):** P10-FE4 (2604.14084), P6-FE1 (2506.00653), P7-FE2 (2601.01552, 2504.10063), P8-FE3 (2506.24106), P4-FE1 (2306.03819). P5-FE1 + P1-FE1 = no triggers.
- **LOW (1):** P9-FE2 (1207.6437).

Re-validation FEs (P8-FE1, P9-FE1, P11-FE1, P9-FE3, P5-FE1, P1-FE1) get the "Internal re-validation" note.

## Sourcing workflow per paper

For each arxiv_id (recommended via link-forge MCP if running, else direct Cypher on bolt://localhost:7687):

```cypher
MATCH (l:Link) WHERE l.url CONTAINS '<arxiv_id>'
RETURN l.title, l.description, l.content, l.authors, l.forgeScore
```

Write 5 fields, each one short sentence:
- `their_method` — neutral 1-2 sentence summary of the paper's method
- `their_result` — concrete numbers + model + benchmark
- `our_method` — from the FE's existing `description` + `rationale`
- `same` — what overlaps
- `differs` — what's different (target signal, benchmark, metric, model, etc.)

If link-forge has nothing for a paper: enqueue via `~/link-forge/scripts/search-papers.ts`, run the processor, then proceed. Last-resort fallback: PAPER_INDEX.md notes.

## Existing pattern to mirror

`Finding→Paper` edges already use this style (e.g., `seed.py:570-640`):
- `CORROBORATED_BY {their_model, note}`
- `METHOD_DIFFERS {theirs, ours}`
- `EXTENDED_BY {experiment_idea, actionable}`
- `CONTRADICTED_BY {why, resolution}`
- `EXPLAINS {mechanism}`

Use the same vocabulary on the new `TRIGGERED_BY` properties — but renamed to be FE-facing (`our_method` not `ours`, etc.) so the generator's "what we'll do" framing reads naturally.

## Recommended execution order

1. Read `seed_future_experiments.py:120-588` to confirm all 24 FE dicts.
2. Update `schema.cypher` (smallest change, no risk).
3. Update `generate_next_experiments.py` — write graceful-empty handling first. Run once to confirm output is unchanged when no edges have new props.
4. Seed CRITICAL tier first (6 FEs, ~7 paper edges). Re-seed + regenerate + spot-check.
5. Continue HIGH → MEDIUM → LOW.
6. Final regenerate; diff against original NEXT_EXPERIMENTS.md.

## Verification gates

- `python research-graph/seed_future_experiments.py` clean (idempotent MERGE).
- `python research-graph/generate_next_experiments.py` reports `experiments: 24, watchlist papers: 25`.
- `python validate_claims.py` still 91/91 PASS (no quantitative claims touched).
- Spot-check P3-FE1 (CRITICAL) names DSR1-Distill-Qwen-1.5B in `their_result`; P10-FE2 (HIGH) cites Mohri & Hashimoto; P9-FE2 (LOW) shows Bubenik landscapes contrast.
- Watchlist table at end of file unchanged.

## Out of scope (do not do)

- Editing PAPER_INDEX.md — different audience.
- Adding new findings or experiments to the graph.
- Migrating existing `Finding→Paper` edges — they serve a different audience and already work.
- Hand-editing NEXT_EXPERIMENTS.md.

## Bootstrap prompt for next session

> Continue the approved plan at `~/.claude/plans/https-github-com-musicofhel-topo-confide-crystalline-feigenbaum.md`. Handoff at `~/topo-confidence/.claude/handoff/2026-04-26-next-experiments-paper-comparisons.md`. Start by reading `seed_future_experiments.py:120-588`, then update `schema.cypher` and the generator, then seed the CRITICAL tier first.
