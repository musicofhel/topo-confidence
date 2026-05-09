# Handoff: Autopilot Daemon — Plan Approved, Implementation Starting

**Date**: 2026-05-09
**Branch**: `max-depth-retriage-2026-04-28`
**Plan file**: `.claude/plans/dapper-sprouting-rose.md`

## What happened this session

1. User dropped arxiv paper `2605.05873` (CITE: Anytime-Valid Statistical Inference in LLM Self-Consistency) in Discord
2. Watched it flow through link-forge admission → research-graph `pending_triage`
3. Ran `/paper-triage 2605.05873` — subagent produced brief at `research-graph/briefs/triage-2026-05-08-2605.05873.md`
4. Dry-run `promote_brief.py` succeeded — would create 3 FEs (P11-FE939/940/941), 2 hypotheses (H-739/740), PAPER_INDEX entry
5. User asked to automate the full pipeline: Discord drop → experiment runs automatically
6. Ran fresh-eyes audit of initial plan — found critical issues (review gate is structural not a flag, 96% FEs have no scripts, claims gate blocks automation, tiny local-runnable set)
7. Explored link-forge codebase for robustness ideas — found trigger file hook point, Discord notification patterns, SQLite queue pattern
8. Designed and got approval for 6-step implementation plan

## The approved plan (6 steps)

**Goal**: Drop arxiv link in Discord → experiments run with zero human intervention.

| Step | File | What |
|------|------|------|
| 1 | `pipeline/graph.py` | `no_review` param — conditionally wire `interpret_findings → write_brief` bypassing `review_gate` node. The `interrupt()` call is structural, can't be skipped with a state flag. |
| 2 | `pipeline/generate_recompute.py` + `pipeline/prompts/recompute_system.py` | **New files.** LLM-based script generator. Queries Neo4j for scriptless READY FEs, calls `claude -p` with 3 exemplar scripts + FE description + NPZ schema, validates output (ast.parse, banned imports, path containment). |
| 3 | `pipeline/autopilot.py` | **New file.** Single-threaded poll loop: triage pending papers → generate scripts → run experiments → classify results. Budget cap (15 calls/day), failure quarantine (3 retries), graceful shutdown. |
| 4 | `~/link-forge/src/processor/research-graph-suggest.ts` | 3-line trigger hook: after Neo4j MERGE, `appendFile` arxiv ID to `~/topo-confidence/.autopilot/trigger`. Add `appendFile, mkdir` to imports at line 19. |
| 5 | `pipeline/result_triage.py` | **New file.** AUROC-based classifier (HIT/NEAR_MISS/NULL/INCONCLUSIVE). Logs results, writes follow-up proposals to `.autopilot/followups/` for human review. |
| 6 | `pipeline/cli.py` | `--no-review` flag on `run` subcommand, `autopilot` subcommand with `--budget`, `--poll`, `--dry-run` |

## Implementation status

**NOT STARTED.** Plan was approved moments before compaction. Begin with Step 1 (review gate bypass in `graph.py`).

## Critical code locations

- `pipeline/graph.py:29-77` — `build_graph()`, the function to modify. Line 69: `builder.add_edge("interpret_findings", "review_gate")` is the edge to conditionally replace.
- `pipeline/nodes.py:488` — `interrupt(payload)` call in `review_gate()` that halts the state machine
- `pipeline/nodes.py:53-60` — `_find_recompute_script()` — glob search for recompute scripts
- `pipeline/nodes.py:67-72` — `_is_local_runnable()` — cost string check
- `pipeline/nodes.py:24-50` — `_query_neo4j_ready_fes()` — Neo4j query for READY FEs
- `pipeline/cli.py:447-452` — run subparser where `--no-review` goes
- `research-graph/triage_one.sh` — existing per-paper triage (subprocess target for daemon)
- `link-forge/src/processor/research-graph-suggest.ts:271` — insertion point for trigger hook
- `link-forge/src/processor/research-graph-suggest.ts:19` — import line to expand
- Exemplar scripts for prompt template: `pathway11_h100/leace_erasure/recompute_fe101.py`, `pathway11_h100/adaptive_bestofk/recompute_fe308.py`, `pathway11_h100/length_baseline/recompute_fe447.py`

## Key design decisions already made

- **Graph topology change, not state flag**: `no_review=True` removes `review_gate` and `revise_artifacts` nodes entirely from the compiled graph
- **Daemon over cron**: User explicitly wanted polling loop for extensibility
- **Script generation via `claude -p`**: Too varied for rigid templates, but consistent enough for LLM with exemplars
- **One item per phase per loop iteration**: Predictable resource usage on 2060 Super
- **Follow-up FE proposals are the one human gate**: Result classification is automatic, but follow-up experiment design writes to staging file for human review
- **Daily LLM budget cap**: Default 15 calls (~$15-20/day)
- **Trigger file from link-forge**: `appendFile` to `.autopilot/trigger`, atomic for <4KB writes

## Brief from this session (not yet promoted)

`research-graph/briefs/triage-2026-05-08-2605.05873.md` — dry-run passed, can promote with:
```bash
cd ~/topo-confidence/research-graph && python promote_brief.py briefs/triage-2026-05-08-2605.05873.md
```
