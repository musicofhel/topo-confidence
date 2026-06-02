# Research Graph RAG — Spec v2

## Problem Statement

The research graph (Neo4j, port 7688) has 308 papers, 14 findings, 1305
future experiments, 327 triage briefs, and 370 tags. Its current search
surface — four Lucene fulltext indexes plus dedicated `query.py` subcommands —
has two failure modes:

**1. FutureExperiment drowning.** The `future_experiment_search` fulltext index
has 1305 entries with 3 indexed text fields (description, trigger, rationale)
averaging 720 chars each. Any cross-type query returns FEs exclusively because
they dominate the Lucene score space. Tested: 12/12 sample queries returned
only FEs in the top-3 when all indexes are queried together.

**2. Paper vocabulary mismatch.** The `paper_relevance` fulltext index covers
`title` + `relevance_note`, but paper titles use specific naming conventions
that don't match natural-language queries. Example: "activation steering
intervention truthfulness" returns zero hits for the ITI paper (2306.03341)
because its title is "Inference-Time Intervention" and its relevance_note
doesn't contain "activation steering." Vector search would catch this via
semantic similarity.

**What already works well:**
- Per-type finding search via `finding_claims` index: all 8 test queries
  returned the expected finding at rank 1
- Structured FE queries via `query.py future`, `query.py highest-roi`,
  `query.py blocked`, `query.py triggered` — these use exact graph filters,
  not fuzzy search
- Graph traversal queries via `query.py corroborators`, `query.py extensions`,
  `query.py subgraph` — these are single-hop Cypher, inherently precise

**Conclusion:** The graph's typed edges ARE the primary retrieval mechanism
for structured queries. Vector search fills a specific gap — semantic paper
discovery and cross-type "tell me everything about X" queries — without
replacing what works.

## What Exists Today

| Capability | Status |
|---|---|
| Fulltext indexes | 4: finding_claims, paper_relevance, experiment_hypotheses, future_experiment_search |
| Graph traversal queries | 18 subcommands in `query.py` (860 lines) |
| Vector indexes | None |
| Semantic search | None |
| Triage briefs | 327 files, avg 21KB, unindexed |

## Design Principles (lessons from link-forge)

1. **Measure before building.** Link-forge's eval harness (50 cases) was built
   first, then used to accept/reject each enhancement. This spec follows the
   same pattern: Phase 0 establishes the baseline, each subsequent phase is
   gated on measured improvement.

2. **FEs are not a search surface.** With 1305 entries of homogeneous technical
   jargon, FEs produce high intra-class similarity and drown cross-type results.
   FE retrieval stays structured (`query.py future --status READY --min-roi 8`).

3. **Extend `query.py`, don't fork.** Adding a separate MCP server or `rag.py`
   creates a second interface that must track schema changes. Instead, add a
   `semantic` subcommand and an `ask` subcommand to the existing CLI.

4. **Graph edges beat vector similarity.** "What papers contradict F-10?" is
   a single Cypher hop returning the exact answer (2402.18048). No retrieval
   pipeline can improve on this. The RAG layer handles only the queries that
   graph traversal can't: semantic discovery and open-ended synthesis.

---

## Phase 0: Eval Harness + Baseline [~2 hours]

Build the eval harness FIRST with concrete expected results, then measure
what fulltext already achieves. This determines whether Phases 1-3 are needed.

### 0a. Test cases

New file: `research-graph/eval_rag.py`

Each case has `expected: list[tuple[str, str]]` — concrete `(label, id)` pairs.

**Finding retrieval (6 cases):**

| # | Query | Expected |
|---|---|---|
| 1 | "What predicts LLM correctness from hidden states?" | `[("Finding", "F-2"), ("Finding", "F-4")]` |
| 2 | "Does persistent homology add signal beyond covariance?" | `[("Finding", "F-7"), ("Finding", "F-10")]` |
| 3 | "What happens to participation ratio during chain of thought?" | `[("Finding", "F-1")]` |
| 4 | "How does the DoM direction change during generation?" | `[("Finding", "F-3")]` |
| 5 | "What accuracy can selective prediction achieve?" | `[("Finding", "F-11")]` |
| 6 | "Is output length a confound for correctness prediction?" | `[("Finding", "F-9")]` |

**Paper retrieval (8 cases):**

| # | Query | Expected |
|---|---|---|
| 7 | "Papers about activation steering for truthfulness" | `[("Paper", "2306.03341")]` |
| 8 | "Linear probes on LLM representations for truth" | `[("Paper", "2310.06824")]` |
| 9 | "Chain of embedding for correctness" | `[("Paper", "2410.13640")]` |
| 10 | "Papers showing LLMs encode problem difficulty" | `[("Paper", "2510.18147")]` |
| 11 | "Persistent topological features in language models" | `[("Paper", "2410.11042")]` |
| 12 | "Length bias in RLHF reward models" | `[("Paper", "2310.03716")]` |
| 13 | "Intrinsic dimension and truthfulness" | `[("Paper", "2402.18048")]` |
| 14 | "Small steering vectors with large effects" | `[("Paper", "2509.06608")]` |

**Cross-type semantic queries (6 cases):**

| # | Query | Expected |
|---|---|---|
| 15 | "What's the full picture on steering in this project?" | `[("Finding", "F-3"), ("Paper", "2306.03341")]` |
| 16 | "Evidence for and against topological features" | `[("Finding", "F-7"), ("Paper", "2410.11042")]` |
| 17 | "What corroborates the prefill direction finding?" | `[("Finding", "F-2"), ("Paper", "2509.12886")]` |
| 18 | "Calibration and confidence in LLM predictions" | `[("Finding", "F-11"), ("Paper", "2510.18147")]` |
| 19 | "Papers evaluated on MATH-500" | Papers in MATH-500 dataset subset (discover via Cypher) |
| 20 | "Papers evaluated on TruthfulQA" | Papers in TruthfulQA dataset subset |

**Queries that should NOT need RAG (handled by `query.py`):**

| # | Query | Existing command |
|---|---|---|
| 21 | "What high-ROI experiments should we run?" | `query.py highest-roi` |
| 22 | "What experiments are blocked?" | `query.py blocked` |
| 23 | "What papers contradict F-10?" | `query.py contradictors F-10` |
| 24 | "FEs triggered by the ITI paper" | graph traversal |
| 25 | "Full neighborhood of F-2" | `query.py subgraph F-2` |

Cases 21-25 are control cases: they verify that the RAG pipeline doesn't
regress on queries already handled well by structured commands. The RAG
pipeline should either route to the existing command or return equivalent
results.

### 0b. Metrics

- **Precision**: fraction of expected `(label, id)` pairs found in top-10
- **MRR**: `1 / (rank of first expected hit + 1)`, 0 if no hit
- Per-category aggregation (finding / paper / cross-type / control)
- Per-case detail in JSON output at `research-graph/eval/`

### 0c. Baseline measurement

Run all 25 cases against existing fulltext indexes (per-type, not combined):

```bash
python eval_rag.py --mode fulltext-only --out baseline
```

Expected outcome based on manual testing:
- Finding cases: ~100% precision (all tested queries hit at rank 1)
- Paper cases: ~60-70% precision (vocabulary mismatch for several papers)
- Cross-type cases: ~40-50% precision (no mechanism to combine results)
- Control cases: 100% precision (routed to existing commands)

**Gate:** If baseline precision is ≥90% across all categories, Phases 1-3 are
not justified. Proceed only if paper or cross-type precision is below 80%.

### 0d. Acceptance criteria for RAG additions (Phases 1-3)

Each phase must demonstrate:
1. ≥5 percentage point precision improvement over baseline
2. Zero regression on finding cases or control cases
3. Two consecutive cached runs produce identical output

---

## Phase 1: Paper Vector Search [~3 hours]

Add vector embeddings to Papers and Findings ONLY. FutureExperiments are
excluded from the semantic search surface (see Design Principles §2).

### 1a. Embedding model

Python `sentence-transformers` with `all-MiniLM-L6-v2` (384-dim, cosine).
Install into the repo-root `.venv`:

```bash
cd ~/topo-confidence && .venv/bin/pip install sentence-transformers
```

The root `.venv` (Python 3.12) already has PyTorch 2.11 — `sentence-transformers`
installs cleanly with no conflicts.

MiniLM truncates silently at 512 tokens (~2000 chars). All Paper and Finding
texts are well within this limit (Paper avg ~370 chars, Finding avg ~327 chars).

### 1b. What gets embedded

| Node | Text formula | Null handling | Count |
|---|---|---|---|
| **Paper** | `title + " " + relevance_note + " " + tags_joined` | 11 papers missing title (use `""`) ; 24 missing relevance_note (use `""`) ; 98 (32%) have no tags | 308 |
| **Finding** | `claim + " " + strongest_counterargument` | All 14 have both fields | 14 |

Tag text for Papers: query `MATCH (p)-[:TAGGED]->(t:Tag) RETURN collect(t.name)`
and join with spaces, normalizing underscores to spaces
(`selective_prediction` → `selective prediction`).

**FutureExperiments** are NOT embedded. At 1305 entries with high lexical
overlap, they would dominate vector results the same way they dominate
fulltext. FE retrieval uses structured filters (`query.py future`).

**Experiments** (33 total, 17 missing hypothesis+result) are deferred. Too
sparse and low-value for initial implementation.

### 1c. Schema additions

Add to `schema.cypher`:

```cypher
// ---- Vector indexes (384-dim all-MiniLM-L6-v2) -------------------------

CREATE VECTOR INDEX paper_embedding_idx IF NOT EXISTS
  FOR (p:Paper) ON (p.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX finding_embedding_idx IF NOT EXISTS
  FOR (f:Finding) ON (f.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};
```

Neo4j 5.26 Community supports vector indexes (added in 5.18 for Community
edition). Confirmed working on the running instance.

### 1d. Backfill script

`research-graph/backfill_embeddings.py`:

```bash
python backfill_embeddings.py [--dry-run] [--node-type Paper|Finding|all] [--force]
```

- Load model once, batch-embed all nodes of each type
- Skip nodes where `embedding IS NOT NULL` unless `--force`
- Handle null title/relevance_note gracefully (empty string)
- Normalize tag underscores to spaces before concatenation
- Progress bar via `tqdm`
- Expected: 308 Paper + 14 Finding = 322 nodes. ~3 seconds on CPU.
- Idempotent, safe to re-run after new papers are promoted

### 1e. Embedding at promotion time

Modify `promote_brief.py` to call `embed_node()` after Paper MERGE and
after FutureExperiment MERGE — but only embed the Paper, not the FEs.

**Caveat:** At promotion time, the Paper may not yet have TAGGED edges
(tags come from seed.py or other processes). The embedding will be
title + relevance_note only. Run `backfill_embeddings.py --force
--node-type Paper` periodically to refresh with tags, or accept the
partial embedding as good enough (title + relevance_note is ~350 chars,
tags add ~50 chars — marginal impact).

**Coupling risk:** If `sentence-transformers` fails to import, paper
promotion breaks. Mitigate with a try/except that logs a warning and
skips embedding — promotion is more important than embedding.

```python
try:
    from backfill_embeddings import embed_node
    embed_node("Paper", paper_arxiv_id, text)
except Exception as e:
    logger.warning(f"Embedding skipped for {paper_arxiv_id}: {e}")
```

---

## Phase 2: Typed Retrieval + RRF [~4 hours]

### 2a. Retrieval paths

Add retrieval functions to `query.py` (not a separate file). Import the
existing `_driver()`, `_run()`, and `_escape_lucene()` helpers.

| # | Path ID | Method | Target | Weight | Limit |
|---|---|---|---|---|---|
| 1 | `paper-vec` | Vector similarity on Paper.embedding | Paper | 2.0 | 20 |
| 2 | `finding-vec` | Vector similarity on Finding.embedding | Finding | 2.0 | 10 |
| 3 | `finding-ft` | Fulltext `finding_claims` index | Finding | 1.5 | 10 |
| 4 | `paper-ft` | Fulltext `paper_relevance` index | Paper | 1.5 | 20 |
| 5 | `tag-match` | Normalize query terms → match Tag.name → TAGGED → Paper/Finding | Paper + Finding | 2.5 | 15 |
| 6 | `dataset-match` | Substring match on Dataset.display_name → USED_IN → Paper | Paper | 2.0 | 10 |
| 7 | `finding-neighborhood` | Seed findings from paths 2-3 → traverse CORROBORATED_BY / CONTRADICTED_BY / EXTENDED_BY / METHOD_DIFFERS / EXPLAINS → Paper | Paper | 1.0 | 15 |

**7 paths, not 8.** No FE paths. No experiment paths.

**Tag normalization (path 5):** Tags use underscores (`selective_prediction`).
Query terms use spaces. Normalize both to lowercase with underscores before
matching. Match via `CONTAINS` substring, not exact equality.

**32% of papers have no tags** — they are invisible to path 5. This is
acceptable because paths 1, 4, and 6 cover untagged papers.

**Dataset path (path 6):** 85 datasets appear in 3+ papers. Queries like
"papers evaluated on MATH-500" or "TruthfulQA benchmarks" benefit from
`query → Dataset.display_name → USED_IN → Paper`. Dataset `display_name`
averages 25 chars (e.g., "MATH-500", "TruthfulQA", "GSM8K").

**Finding-neighborhood (path 7):** Only 25 typed edges exist across 14
findings. The path is cheap but low-yield. Weight 1.0 reflects this — it's
a precision booster, not a recall path.

### 2b. RRF merge

```python
def rrf_merge(
    path_results: dict[str, list[tuple[str, str, int]]],
    weights: dict[str, float],
    rescue_paths: set[str],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """Reciprocal Rank Fusion with rescue mechanism.
    
    Each path_results entry: path_id -> [(label, node_id, rank), ...]
    Returns top_n merged results.
    """
    scores: dict[str, float] = {}
    appearances: dict[str, set[str]] = {}

    for path_id, results in path_results.items():
        w = weights.get(path_id, 1.0)
        for label, node_id, rank in results:
            key = f"{label}||{node_id}"
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            appearances.setdefault(key, set()).add(path_id)

    # Rescue: items found by exactly 1 path AND that path is a rescue path
    # AND the item ranked in top-3 of that path get a 0.04 boost.
    # This prevents high-confidence single-path hits from being buried
    # by multi-path items with mediocre scores.
    RESCUE_BOOST = 0.04
    RESCUE_MAX_RANK = 3
    for path_id, results in path_results.items():
        if path_id not in rescue_paths:
            continue
        for label, node_id, rank in results:
            key = f"{label}||{node_id}"
            if len(appearances.get(key, set())) == 1 and rank <= RESCUE_MAX_RANK:
                scores[key] += RESCUE_BOOST

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    out = []
    for key, score in ranked[:top_n]:
        label, node_id = key.split("||", 1)
        out.append({"label": label, "id": node_id, "score": score})
    return out
```

Key differences from v1 pseudocode:
- **`||` separator** instead of `:` — no collision risk with arxiv IDs
- **Single-path-only guard** on rescue (`len(appearances[key]) == 1`)
- **Stable sort tiebreaker** — `(-score, key)` ensures deterministic ordering
  when scores are equal
- **Rescue paths**: `tag-match` and `dataset-match` (high-confidence structured
  matches that might only appear in one path)

### 2c. Context assembly

Per-type formatting for LLM synthesis:

**Paper:**
```
[1] Paper: "Inference-Time Intervention" (2306.03341, 2023)
    Relevance: ITI applies activation shifts at attention heads using directions
    derived from labeled probes. Alpaca TruthfulQA doubles.
    Tags: activation-steering, intervention
    ForgeScore: 7.2
    Edges: CORROBORATED_BY ← F-2, F-4 | TAGGED → steering, truthfulness
```

**Finding:**
```
[3] Finding F-2 [ACTIVE/STRONG]:
    Claim: Prefill hidden state (position 0) predicts correctness at 0.771 AUROC
    (1.5B) and 0.876 AUROC (7B)...
    Strongest counterargument: A single direction may be a shallow shortcut...
    Evidence: [list of evidence items]
    Corroborated by: 2509.12886, 2510.18147
```

**Relationship context is mandatory.** For each Paper, include its typed edges
to Findings (CORROBORATED_BY, CONTRADICTED_BY, etc.). For each Finding, include
its top corroborating/contradicting papers. This is the main value of graph
RAG over flat vector search.

### 2d. System prompt

```
You are answering questions about the topo-confidence research project, which
investigates whether residual-stream geometry predicts LLM correctness.

The context below contains Findings (F-N: established results with evidence
and counterarguments) and Papers (arxiv papers linked to findings via typed
edges like CORROBORATED_BY, CONTRADICTED_BY, EXTENDED_BY).

When answering:
- Cite specific finding IDs (F-2, F-7, etc.) and arxiv IDs
- Distinguish between ACTIVE, INVALIDATED, and SUPERSEDED findings
- Note the strength level (STRONG, MODERATE, PRELIMINARY)
- If a finding has a strongest_counterargument, mention it
- Reference specific AUROC values, coverage numbers, and benchmarks
```

### 2e. New `query.py` subcommands

```bash
# Semantic search — vector + fulltext + graph, RRF merged
python query.py semantic "activation steering for truthfulness"

# Ask — semantic search + LLM synthesis
python query.py ask "What's the evidence for and against topological features?"

# Both support --skip-llm (retrieval only) and --json (machine-readable)
python query.py semantic "steering vectors" --json
python query.py ask "..." --skip-llm
```

These extend the existing CLI rather than creating a separate MCP server.
The `_driver()` and `_run()` helpers are already available. An MCP server
can be added later if demand materializes — it would be a thin wrapper
around these same functions.

---

## Phase 3: LLM Enhancements [~3 hours, gated on Phase 2 eval]

**Proceed only if Phase 2 eval shows precision <85% or MRR <0.5.**

### 3a. HyDE (Hypothetical Document Embedding)

Domain-specific prompt:

```
Write a short factual paragraph (3-4 sentences) as if you are a research note
in the topo-confidence project about residual-stream geometry and LLM
correctness prediction. Reference specific concepts: hidden-state geometry,
DoM (difference-of-means direction), AUROC, participation ratio, covariance
spectrum, persistent homology, steering vectors, selective prediction,
MATH-500 benchmark. Be concrete about methods and results. Do not restate
the question.
```

Implementation:
- Embed the synthetic text with `sentence-transformers`
- Average element-wise with the query embedding, L2-renormalize
- Use the combined embedding for paths 1 (`paper-vec`) and 2 (`finding-vec`)
- Cache key: `hyde:{query_text}`
- Cache value: the raw synthetic text (re-embed on read to allow model changes)

### 3b. Concept expansion

Generate 8-10 related technical terms for additional tag-match and fulltext
queries. Weight 0.25 (very low — link-forge learned this prevents noise from
LLM-generated terms swamping real matches).

Prompt:
```
Given the question below about the topo-confidence research project (LLM
correctness prediction via residual-stream geometry), generate 8-10 related
technical terms that might appear in paper titles, finding claims, or tag
names. Output one term per line, lowercase, hyphenated. No bullets or numbers.
```

Cache key: `concept-expand:{query_text}`

### 3c. Reranking

After RRF merge, send top-10 context snippets to Claude. Claude returns a
relevance-ordered ranking.

Cache key: `rerank:{query_text}|||{sorted(f"{r['label']}||{r['id']}" for r in top_results)}`

The key includes node labels to avoid collisions across types.

### 3d. RAG-Fusion

**Deferred to Phase 3b (optional).** At 322 embeddable nodes, running 7 paths
4x (28 total) is overkill. If HyDE + concept expansion + reranking don't
close the gap, add fusion as a last resort.

If implemented:
- Generate 3 query variants (domain-agnostic prompt is fine here)
- Run paths 1-5 only (not graph traversal paths 6-7) with reduced limit (8 per path)
- Base query weight 3.0 in fusion-level RRF; variants weight 1.0
- Reranking on base query only, not variants
- Cache key: `fusion:{query_text}`, value: JSON-serialized list

### 3e. LLM cache

```python
# research-graph/llm_cache.py
CACHE_PATH = Path(__file__).parent / "cache" / "llm-cache.json"
```

**NOT in `data/`** — that directory is the Neo4j Docker volume mount
(`./data:/data` in docker-compose.yml). Writing there risks deletion on
`docker compose down -v`. Use `cache/` with a `.gitignore` entry.

API:
- `cache_get(type: str, key: str) -> str | None`
- `cache_set(type: str, key: str, value: str)`
- `cache_size() -> int`

Storage format: `{"type:key": "value", ...}` — same as link-forge.

When `--no-cache` flag is set, skip both reads AND writes (link-forge skips
both). Thread the flag through every function that touches the cache:
`semantic_search(query, no_cache=False)` → `_hyde(query, no_cache)` →
`cache_get(...)` / `cache_set(...)`.

### 3f. LLM backend

Use `claude -p` (Claude Code CLI) for HyDE, concept expansion, and reranking.
Subprocess call with 30-second timeout. Delete `CLAUDECODE` from the
subprocess environment to prevent nested-session detection:

```python
env = os.environ.copy()
env.pop("CLAUDECODE", None)
result = subprocess.run(
    ["claude", "-p", prompt],
    input=user_msg, capture_output=True, text=True,
    timeout=30, env=env,
)
```

---

## Phase 4: Tuning + Shipping [~2 hours]

### 4a. Weight tuning protocol

1. Run eval with default weights → record precision and MRR per category
2. For each path, test weight ×0.5 and ×2.0 independently
3. Keep the weight that maximizes overall MRR without regressing any category's
   precision below the baseline
4. Repeat until no single-weight change improves MRR by >0.02

Optimize for MRR (ranking quality) as the primary metric; precision as a
hard constraint (no regressions).

### 4b. Determinism verification

Two consecutive runs with fully-seeded cache must produce byte-identical
JSON output. Gotchas to address:

1. **Cache seeding must be single-threaded** — parallel seeding can produce
   different cache entries for the same key if LLM calls are non-deterministic
2. **Neo4j tiebreakers** — add `ORDER BY score DESC, node.arxiv_id ASC` (or
   equivalent) to all vector and fulltext queries
3. **Python dict insertion order** — use `sorted()` on path results before
   merging to ensure consistent RRF scoring regardless of execution order
4. **Float precision** — RRF scores are floats; use `round(score, 10)` before
   sorting to avoid epsilon-level ordering instability

### 4c. Final acceptance criteria

1. Finding cases: precision ≥ baseline (expected ~100%)
2. Paper cases: precision ≥ baseline + 10pp (the main improvement target)
3. Cross-type cases: precision ≥ baseline + 5pp
4. Control cases: 100% (no regressions on structured queries)
5. Overall MRR ≥ 0.4
6. Determinism verified
7. `python -m py_compile query.py` passes (no syntax errors)

---

## Phase 5: Brief Embeddings [~2 hours, optional]

The 327 triage briefs (avg 21KB each, ~7MB total) are the richest text in
the system — multi-page structured analyses with refutations, methodology
comparisons, and cross-paper signals. They are currently unindexed.

### 5a. Brief embedding strategy

Each brief is too long for a single MiniLM embedding (512-token limit). Split
into sections (methodology, refutations, cross-paper signals) and embed each
section as a separate vector linked to the Paper node:

```cypher
CREATE CONSTRAINT brief_section_id IF NOT EXISTS
  FOR (bs:BriefSection) REQUIRE bs.id IS UNIQUE;

CREATE VECTOR INDEX brief_section_embedding_idx IF NOT EXISTS
  FOR (bs:BriefSection) ON (bs.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};
```

Node: `(:BriefSection {id: "<arxiv_id>#<section>", text: "...", embedding: [...]})`
Edge: `(:BriefSection)-[:FROM_BRIEF]->(:Paper)`

### 5b. New retrieval path

Add path 8: `brief-vec` — vector similarity on BriefSection.embedding,
weight 1.5, limit 15. Deduplicate by Paper (multiple sections from the same
brief collapse to the highest-ranked one).

### 5c. Gate

Only proceed if Phase 2 eval shows paper retrieval precision is still below
85% after vector + fulltext + graph paths. Briefs are a recall booster for
papers where title + relevance_note are too terse to match.

---

## Implementation Order

| Phase | Work | Gate | Est. time |
|---|---|---|---|
| **0** | Eval harness + baseline measurement | — | 2h |
| **1** | sentence-transformers, backfill, vector indexes | Baseline precision <80% on paper/cross-type | 3h |
| **2** | 7 retrieval paths, RRF merge, `query.py semantic/ask` | Phase 1 backfill complete | 4h |
| **3** | HyDE + concept-expand + rerank (cached) | Phase 2 precision <85% or MRR <0.5 | 3h |
| **4** | Weight tuning, determinism, ship | Phase 2 or 3 complete | 2h |
| **5** | Brief section embeddings + path 8 | Paper precision still <85% after Phase 3 | 2h |
| | **Minimum (0-2 + 4)** | | **~11 hours** |
| | **Maximum (all)** | | **~16 hours** |

## Critical Path

```
Phase 0 (eval harness + baseline)
    │
    ├── baseline ≥90%? → STOP (current system is sufficient)
    │
    ▼
Phase 1 (embeddings + vector indexes)
    │
    ▼
Phase 2 (typed retrieval + RRF) → eval
    │
    ├── precision ≥85% & MRR ≥0.5? → Phase 4 (ship)
    │
    ▼
Phase 3 (HyDE + concept-expand + rerank) → eval
    │
    ├── paper precision ≥85%? → Phase 4 (ship)
    │
    ▼
Phase 5 (brief embeddings) → eval → Phase 4 (ship)
```

Each phase has an explicit gate. No phase is assumed necessary until the
previous phase's eval justifies it.

## Files Created/Modified

| File | Action | Description |
|---|---|---|
| `research-graph/eval_rag.py` | **New** | 25-case eval harness with precision + MRR |
| `research-graph/backfill_embeddings.py` | **New** | Batch embed Paper + Finding nodes |
| `research-graph/llm_cache.py` | **New** | JSON KV cache for LLM calls |
| `research-graph/query.py` | **Modify** | Add `semantic` and `ask` subcommands, 7 retrieval paths, RRF merge |
| `research-graph/schema.cypher` | **Modify** | Add 2 vector index definitions (Paper, Finding) |
| `research-graph/promote_brief.py` | **Modify** | Embed Paper at promotion time (try/except guarded) |
| `research-graph/cache/llm-cache.json` | **New** (gitignored) | LLM cache entries |
| `research-graph/cache/.gitignore` | **New** | `*.json` |
| `research-graph/eval/` | **New** (dir) | Eval result JSON snapshots |

## Data Inventory

| Entity | Count | Embeddable | Notes |
|---|---|---|---|
| Paper | 308 | 308 (11 missing title, 24 missing relevance_note, 98 no tags) | Primary search target |
| Finding | 14 | 14 (all have claim + counterargument) | All ACTIVE, none invalidated |
| FutureExperiment | 1305 | **Excluded** | Handled by structured `query.py` commands |
| Experiment | 33 | **Deferred** | 17 missing hypothesis+result |
| Triage brief | 327 | Phase 5 (section-level) | Avg 21KB, richest text in system |
| Tag | 370 | N/A (used for graph traversal path) | Underscore-separated, avg 16 chars |
| Dataset | 862 (85 shared) | N/A (used for graph traversal path) | display_name may be null — use name fallback |
| Method | 1794 (1 shared) | **Excluded** | Too granular — nearly all paper-specific |

## Risks

| Risk | Mitigation |
|---|---|
| FE drowning via any retrieval path | FEs excluded from embedding + retrieval surface entirely |
| Baseline is already good enough (≥90%) | Phase 0 gate — stop if fulltext suffices |
| sentence-transformers breaks promote_brief.py | try/except guard; embedding is optional, promotion is not |
| Vector search doesn't improve over fulltext for 308 papers | Phase 2 eval gate — if precision gain <5pp, revert to fulltext-only |
| Brief sections add noise (avg 21KB per brief, many sections) | Phase 5 gated on demonstrated need; deduplicate by Paper |
| Neo4j 5.26 Community vector index limitations | Confirmed working via live test on running instance |
| Cache in `data/` would be inside Neo4j volume mount | Fixed: use `cache/` directory instead |

## Non-Goals

- **No FE retrieval via RAG**: FEs are too numerous (1305) with too-similar
  text. Structured filters (`query.py future/highest-roi/blocked/triggered`)
  are strictly better for FE queries.
- **No separate MCP server**: Extend `query.py` instead. MCP can be added
  later as a thin wrapper if needed.
- **No cross-database retrieval**: link-forge and research-graph stay separate.
- **No RAG-Fusion in initial phases**: At 322 embeddable nodes, running 7
  paths 4x is disproportionate. Deferred to Phase 3b if simpler enhancements
  don't close the gap.
- **No changes to autopilot daemon**: The triage pipeline continues as-is.
