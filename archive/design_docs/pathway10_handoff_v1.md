# Pathway 10 — Handoff (v1, 2026-04-23)

Session goal: find papers for a "v1 document that explores new directions" after pathway 9 closed with the PH→CoE pivot. **Done.** Plan doc + themed lit index delivered.

---

## Current state

**Pathway 9: CLOSED** (`pathway9/HANDOFF_v3.md`, `pathway9/SUMMARY.md`).
Pivot: paper narrative goes to CoE. CoE-60 MATH-500 holdout **0.811**, cross-domain MATH↔BBH **0.720 / 0.712**.

**Pathway 10: planning** (this session's output).

---

## What this session produced

| Artifact | Path | Purpose |
|---|---|---|
| Plan doc | `~/topo-confidence/pathway10_v1.md` | 5 candidate directions (A–E), each with hypothesis, supporting lit, experiment, success criterion. Recommended sequencing table. |
| Themed lit index | `~/topo-confidence/pathway10-papers.md` | 12 themed clusters, 80+ papers from link-forge Neo4j. |
| Regenerator | `~/link-forge/scripts/build-new-directions.mjs` | `node scripts/build-new-directions.mjs > ~/topo-confidence/pathway10-papers.md` |
| (pre-compaction) Raw lit sweep | `~/topo-confidence/pathway9/literature-sweep.md` | Flat forgeScore-ranked reading list, 560 papers ingested since 2026-04-22. |
| (pre-compaction) Flat regenerator | `~/link-forge/scripts/build-topo-confidence-reading-list.mjs` | Refreshes the flat list. |

---

## Five directions (see `pathway10_v1.md` for full spec)

| # | Direction | Cost | Standalone paper? |
|---|---|---|---|
| A | Cross-scale CoE transfer (1.5B → 7B) | 1 wk, ~4h H100 | **Yes** — top EV |
| B | CoE ⊕ attention ⊕ spectral ensemble | 1 wk, ~8h H100 | Yes (if ≥0.03 lift) |
| C | Calibration (ECE / AURC / risk-coverage) | 2 days, CPU | No — blocker for any submission |
| D | Mechanistic localization (SAE + patching) | 2 wks, 1 H100 day | **Yes** — MI venue |
| E | Zigzag PH | 3 days | Negative-result close-out |

**Recommended sequencing: C → A → B → D → E.** C first is non-negotiable — we have zero calibration numbers today.

---

## What the next agent should do first

1. **Read** `pathway10_v1.md` and `pathway9/SUMMARY.md` (in that order).
2. **Start Direction C** — it's 2 days of CPU work on cached pathway-9 CoE outputs. Gates everything downstream because a submission needs an ECE / AURC table alongside the AUROC table.
3. **Only after C**, decide A vs D based on whether the headline paper leans empirical (A) or mechanistic (D).

**Do NOT:**
- Start Direction E before A–D. Pathway 7 + 8 already make the PH null case; zigzag is low-EV.
- Refresh the lit sweep before using it. The current index covers the corpus we need. Refresh only if a new direction opens up.
- Spin up the RunPod pod (`0agitikupjg259`, stopped) for Direction C — it's CPU work. Directions A / B / D need it.

---

## Known gaps / open questions (copied from `pathway10_v1.md § Open questions`)

- **CoE length confound.** Pathway 9 deconfounded ABC-44 against raw token length (−0.050). CoE-60 was *not* tested against the same control. The 0.811 headline needs the same footnote before the paper ships.
- **Label scheme reconciliation.** Pathway 8 manifest labels (MATH=231 correct) vs NEW-labels (104 correct). Any pathway-10 number must specify which. 0.811 used NEW; cross-domain 0.720 / 0.712 used manifest.
- **BBH subset heterogeneity.** Transfer was measured on pooled 3 subsets × 250. Is CoE transfer uniform across subsets, or driven by one?
- **MATH difficulty stratification.** Does CoE AUROC degrade monotonically across MATH's 7 levels, and does that carry over to BBH?

---

## Lit coverage — where the corpus is thin

From the themed build, 12 clusters, counts of relevant hits since 2026-04-22:

| Theme | Hits | Note |
|---|---|---|
| Calibration / selective / abstention | 19 | Strong — direction C is well-supported |
| Alternative TDA (zigzag/DTM/Dowker/Ricci/Mapper) | 16 | Strong |
| Manifold / intrinsic dimension | 9 | Strong |
| Attention-based | 6 | OK |
| Spectral / eigenvalue | 6 | OK |
| Info-theoretic / semantic entropy | 5 | OK |
| Mech-interp / SAE / probing | 5 | Thin for direction D — **supplementary sweep recommended before starting D** |
| Cross-domain / cross-model | 4 | Thin but existing hits are strong (AUTOPROBE, 20-model validation screen) |
| Code / math / reasoning-domain correctness | 3 | Thin |
| CoE variants / extensions | 2 | **Thin — supplementary sweep strongly recommended for A and B** |
| Steering / repengineering | 2 | Thin |

**Supplementary sweep queries to run if Direction D becomes active:** SAE correctness probing, circuit-level hallucination, truthfulness direction probing, attention-head truthfulness.

**Supplementary sweep queries if A / B become active:** chain-of-embedding extensions, hidden-state trajectory variants, Wang ICLR 2025 follow-ups, cross-model probe transfer.

To run supplementary queries: add them as a text file and `cd ~/link-forge && npx tsx scripts/search-papers.ts --file <file> --max 10 --source arxiv,semantic-scholar,openalex`. Processor (PROCESSOR_WORKERS=4) picks them up automatically.

---

## Research DB state

- **Neo4j**: bolt://localhost:7687 (container `link-forge-neo4j`, user `neo4j`, pass `link_forge_dev`). 560 new papers since 2026-04-22 on top of the existing corpus.
- **Processor**: `PROCESSOR_WORKERS=4` (reduced from 8 after EPIPE crash mid-session). Still running in the background from the pre-compaction session; if `docker ps | grep link-forge` shows it healthy, queue will drain automatically when new queries are added.
- **MCP tools**: `forge_search`, `forge_concepts`, `forge_authors`, `forge_related`, `forge_recent` available from any Claude Code session.
- **Discord RAG**: `/forge ask <question>` in the link-forge channel.

Memory: `~/.claude/projects/-home-musicofhel/memory/research-db-link-forge.md` (durable note that link-forge is the research DB).
