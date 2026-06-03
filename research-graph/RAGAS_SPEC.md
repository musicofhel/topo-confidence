# RAGAS Eval Generator — v2.1 Spec (audited for RAGAS 0.4.3)

## Goal

Auto-generate 150–300 eval cases from the Neo4j research graph, covering
vocabulary diversity and multi-hop reasoning that hand-written cases miss.
Output in `eval_rag.py`'s `EvalCase` format so both hand-written and synthetic
cases run through the same harness.

**v2.1 additions** (audit pass for RAGAS 0.4.3): Node.id UUID mapping
layer, forward-reference reordering, freshness check logic fix,
`LangchainLLMWrapper` → `llm_factory`, `cosine_similarity` →
`summary_similarity`, pinned `ragas==0.4.3`, dropped langchain deps.

**v2 additions**: Pydantic schema validation at every boundary, freshness
hashing to detect ground-truth rot, semantic validation pass to catch
polarity errors (CONTRADICTED_BY presented as "supports"), difficulty
bucketing for meaningful per-tier reporting, and an incremental regeneration
model instead of wholesale replacement.

## Design Principles

1. **Golden set is immovable.** The 41 hand-written cases are the primary
   signal. Synthetic cases are supplementary — a regression on golden cases
   is a real problem regardless of synthetic results.
2. **Generate once, curate, commit.** Synthetic cases are frozen artifacts.
   When the graph changes, a freshness check flags stale cases; only those
   get regenerated, not the full set.
3. **Validate at every boundary.** Pydantic models enforce structure from
   Neo4j extraction through final JSON output. Runtime gates verify
   ground-truth nodes still exist before eval.
4. **Report per difficulty tier.** Aggregate precision across 300 cases
   is meaningless. Per-tier reporting (single-hop / multi-hop / adversarial)
   surfaces real regressions that softballs would hide.

## Architecture

```
Neo4j (308 papers, 14 findings, 862 datasets, 1794 methods, 370 tags)
  │
  ├─ Pydantic: Neo4jPaper, Neo4jFinding ← schema gate 1
  ▼
neo4j_to_ragas.py          ← adapter: Cypher → RAGAS KnowledgeGraph
  │
  ├─ Pydantic: RagasNodePayload ← schema gate 2
  ▼
RAGAS TestsetGenerator     ← synthesizers produce (query, reference_contexts, answer)
  │
  ├─ Pydantic: RawSyntheticCase ← schema gate 3
  ▼
postprocess.py             ← dedup, semantic validation, freshness hash, difficulty tag
  │
  ├─ Pydantic: SyntheticEvalCase ← schema gate 4
  ▼
eval/synthetic-v1.json     ← frozen, version-controlled
  │
  ├─ Runtime: freshness check + node existence ← gate 5
  ▼
eval_rag.py --synthetic    ← runs golden + synthetic through same harness
```

---

## Part 1: Pydantic Schema Models

All models live in `ragas_gen/schemas.py`. Single source of truth for
every data structure in the pipeline.

**RAGAS 0.4.x compat note:** RAGAS `Node.id` requires `uuid.UUID`, not
arbitrary strings. The adapter maintains a bidirectional mapping
(`neo4j_id ↔ UUID`) so synthesizer output can be traced back to Neo4j
node IDs during post-processing. The mapping is stored in
`RagasNodePayload.ragas_uuid` and a module-level `UUID_TO_NEO4J` dict.

```python
from __future__ import annotations
import uuid
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum

# ── Neo4j extraction models ───────────────────────────────────────────
# (ordered so forward references resolve without model_rebuild)

class FindingEdge(BaseModel):
    finding_id: str
    rel_type: str  # CORROBORATED_BY | CONTRADICTED_BY | EXTENDED_BY | EXPLAINS | METHOD_DIFFERS

    @field_validator("rel_type")
    @classmethod
    def valid_rel(cls, v: str) -> str:
        allowed = {"CORROBORATED_BY", "CONTRADICTED_BY", "EXTENDED_BY", "EXPLAINS", "METHOD_DIFFERS"}
        if v not in allowed:
            raise ValueError(f"Unknown rel_type: {v}")
        return v

class Neo4jPaper(BaseModel):
    arxiv_id: str
    title: str
    relevance_note: str = ""
    status: str
    tags: list[str] = []
    embedding: list[float] = Field(min_length=384, max_length=384)
    finding_edges: list[FindingEdge] = []

class Neo4jFinding(BaseModel):
    id: str  # F-1, F-2, ...
    claim: str
    status: str
    strength: str
    embedding: list[float] = Field(min_length=384, max_length=384)

# ── RAGAS adapter models ──────────────────────────────────────────────

class RagasNodePayload(BaseModel):
    """Validated before creating a RAGAS Node."""
    page_content: str = Field(min_length=10)
    entities: list[str] = Field(min_length=1)
    themes: list[str] = Field(min_length=1)
    embedding: list[float] = Field(min_length=384, max_length=384)
    summary: str = Field(min_length=5)
    neo4j_label: str  # Paper | Finding
    neo4j_id: str
    ragas_uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    content_hash: str  # SHA-256 of page_content — freshness anchor

# Module-level reverse lookup, populated during KG construction:
#   UUID_TO_NEO4J: dict[uuid.UUID, tuple[str, str]]  # uuid → (label, neo4j_id)

# ── Post-processing models ────────────────────────────────────────────

class Difficulty(str, Enum):
    SINGLE_HOP_EXACT = "single-hop-exact"          # 1 expected node, direct vocabulary match
    SINGLE_HOP_PARAPHRASE = "single-hop-paraphrase" # 1 expected node, vocabulary varies
    MULTI_HOP = "multi-hop"                          # 2+ expected nodes, bridging required
    ADVERSARIAL = "adversarial"                      # misspelled, poor grammar, keyword-style

class RawSyntheticCase(BaseModel):
    """Output from RAGAS before post-processing."""
    query: str
    reference_contexts: list[str]
    reference_answer: str
    synthesizer_name: str
    query_style: str

class SemanticValidation(BaseModel):
    """Result of LLM judge checking ground-truth correctness."""
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str  # one-line explanation from the judge

class SyntheticEvalCase(BaseModel):
    """Final validated case ready for eval."""
    id: int = Field(ge=1000)
    query: str = Field(min_length=5)
    expected: list[tuple[str, str]]  # [(label, node_id), ...]
    category: str
    difficulty: Difficulty
    synthesizer: str
    style: str
    reference_answer: str
    freshness_hash: str  # SHA-256 of concat(expected node content_hashes)
    generated_at: datetime
    semantic_validation: SemanticValidation

    @field_validator("expected")
    @classmethod
    def non_empty_expected(cls, v):
        if not v:
            raise ValueError("expected must have at least one entry")
        return v

# ── Output container ──────────────────────────────────────────────────

class GeneratorConfig(BaseModel):
    synthesizers: list[str]
    style_distribution: dict[str, float]
    personas: list[str]
    dedup_threshold: float = 0.85
    semantic_validation_model: str = "anthropic/claude-sonnet-4-6"
    openrouter_model: str = "anthropic/claude-sonnet-4-6"

class GenerationStats(BaseModel):
    total_generated: int      # before filtering
    dedup_removed: int
    semantic_invalid_removed: int
    freshness_valid: int      # should equal len(cases)
    by_difficulty: dict[str, int]
    by_synthesizer: dict[str, int]

class SyntheticEvalSet(BaseModel):
    """Top-level schema for synthetic-v1.json."""
    generated_at: datetime
    ragas_version: str
    graph_snapshot_hash: str  # hash of all node IDs + content_hashes at generation time
    generator_config: GeneratorConfig
    cases: list[SyntheticEvalCase]
    stats: GenerationStats
```

---

## Part 2: Neo4j → RAGAS KnowledgeGraph Adapter

### Node mapping

| Neo4j label | RAGAS NodeType | `page_content` source | `entities` | `themes` | `embedding` |
|---|---|---|---|---|---|
| Paper | CHUNK | `"{title}. {relevance_note}"` | Tags from `:TAGGED` edges | Tags (same) | `p.embedding` (384-d) |
| Finding | CHUNK | `"{id}: {claim}"` | `[id, status, strength]` | `["finding", status]` | `f.embedding` (384-d) |

**Skip** FutureExperiment, Dataset, Method, Tag, Pathway, Experiment, Artifact
as standalone nodes — too sparse for question generation. They appear as
properties on Paper/Finding nodes instead (via `entities`/`themes`).

### Relationship mapping — with polarity preservation

v1 mapped all Finding→Paper edges to generic `entities_overlap`. This
loses the semantic difference between CORROBORATED_BY and CONTRADICTED_BY,
causing the synthesizer to generate questions with wrong polarity.

v2 preserves polarity in the `overlapped_items` property:

| Neo4j edge | RAGAS rel type | `overlapped_items` value |
|---|---|---|
| `CORROBORATED_BY` | `entities_overlap` | `["{f_id}:corroborates"]` |
| `CONTRADICTED_BY` | `entities_overlap` | `["{f_id}:contradicts"]` |
| `EXTENDED_BY` | `entities_overlap` | `["{f_id}:extends"]` |
| `EXPLAINS` | `entities_overlap` | `["{f_id}:explains"]` |
| `METHOD_DIFFERS` | `entities_overlap` | `["{f_id}:method_differs"]` |
| Co-tag (≥2 shared) | `entities_overlap` | `[shared_tag_names...]` |
| Cosine sim > 0.7 | `summary_similarity` | score in properties |

**Note:** RAGAS 0.4.x `MultiHopAbstractQuerySynthesizer` defaults to
`relation_property="summary_similarity"`. We use that name (not
`"cosine_similarity"`) so the synthesizer finds our similarity edges
without needing param overrides.

The `neo4j_rel` property on each RAGAS relationship stores the original
edge type. The semantic validation pass (Part 5) reads this to verify
generated questions don't invert polarity.

### What to skip

RAGAS's default transform pipeline: SummaryExtractor, NERExtractor,
ThemesExtractor, EmbeddingExtractor, CosineSimilarityBuilder,
OverlapScoreBuilder. We skip **all of them** — our Neo4j graph already has
embeddings, entities (tags), themes, and relationships. Zero LLM cost for
graph construction.

### Implementation: `neo4j_to_ragas.py`

```python
import hashlib
import uuid
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas_gen.schemas import Neo4jPaper, Neo4jFinding, RagasNodePayload, FindingEdge

# Bidirectional UUID ↔ Neo4j ID mapping, populated during build_knowledge_graph()
UUID_TO_NEO4J: dict[uuid.UUID, tuple[str, str]] = {}   # uuid → (label, neo4j_id)
NEO4J_TO_UUID: dict[tuple[str, str], uuid.UUID] = {}   # (label, neo4j_id) → uuid

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def extract_papers(session) -> list[Neo4jPaper]:
    """Cypher → validated Pydantic models. Fails fast on schema violations."""
    rows = session.run("""
        MATCH (p:Paper)
        WHERE p.status = 'graphed' AND p.embedding IS NOT NULL
        OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
        WITH p, collect(DISTINCT t.name) AS tags
        OPTIONAL MATCH (p)<-[r]-(f:Finding)
        WHERE type(r) IN ['CORROBORATED_BY','CONTRADICTED_BY','EXTENDED_BY','EXPLAINS','METHOD_DIFFERS']
        WITH p, tags, collect({finding_id: f.id, rel_type: type(r)}) AS edges
        RETURN p, tags, edges
    """).data()

    papers = []
    for r in rows:
        p = r["p"]
        papers.append(Neo4jPaper(
            arxiv_id=p["arxiv_id"],
            title=p.get("title", ""),
            relevance_note=p.get("relevance_note", ""),
            status=p["status"],
            tags=r["tags"],
            embedding=p["embedding"],
            finding_edges=[FindingEdge(**e) for e in r["edges"] if e["finding_id"]],
        ))
    return papers

def payload_to_ragas_node(payload: RagasNodePayload) -> Node:
    """Convert validated payload → RAGAS Node with UUID, register in mapping."""
    node = Node(
        id=payload.ragas_uuid,
        type=NodeType.CHUNK,
        properties={
            "page_content": payload.page_content,
            "entities": payload.entities,
            "themes": payload.themes,
            "embedding": payload.embedding,
            "summary": payload.summary,
        },
    )
    UUID_TO_NEO4J[payload.ragas_uuid] = (payload.neo4j_label, payload.neo4j_id)
    NEO4J_TO_UUID[(payload.neo4j_label, payload.neo4j_id)] = payload.ragas_uuid
    return node

def paper_to_ragas_payload(p: Neo4jPaper) -> RagasNodePayload:
    content = f"{p.title}. {p.relevance_note}".strip()
    return RagasNodePayload(
        page_content=content,
        entities=p.tags if p.tags else [p.arxiv_id],
        themes=p.tags if p.tags else [p.title.split()[0]],
        embedding=p.embedding,
        summary=p.relevance_note or p.title,
        neo4j_label="Paper",
        neo4j_id=p.arxiv_id,
        content_hash=_content_hash(content),
    )
```

---

## Part 3: RAGAS Synthesizers

### Synthesizer selection

| Synthesizer | What it generates | Expected yield | Our value |
|---|---|---|---|
| `SingleHopSpecificQuerySynthesizer` | "What does paper X say about Y?" | ~150 cases | Vocabulary robustness — many ways to ask about one node |
| `MultiHopSpecificQuerySynthesizer` | "How does finding F relate to paper P?" | ~80 cases | Tests cross-type retrieval via `entities_overlap` edges |
| `MultiHopAbstractQuerySynthesizer` | "What's the landscape of X in this research?" | ~30 cases | Tests broad/thematic retrieval via theme clusters |

### Query style distribution

Target distribution:
- 50% PERFECT_GRAMMAR
- 30% WEB_SEARCH_LIKE (keyword-style: "steering vectors LLM truthfulness")
- 10% MISSPELLED (robustness: "persistant homology language model")
- 10% POOR_GRAMMAR

### Difficulty assignment

Difficulty is assigned deterministically from synthesizer + style + expected count:

```python
def assign_difficulty(synth: str, style: str, expected_count: int) -> Difficulty:
    if style in ("misspelled", "poor_grammar"):
        return Difficulty.ADVERSARIAL
    if synth == "single_hop_specific" and expected_count == 1:
        if style == "web_search_like":
            return Difficulty.SINGLE_HOP_PARAPHRASE
        return Difficulty.SINGLE_HOP_EXACT
    return Difficulty.MULTI_HOP
```

### Persona list

1. **ML researcher** — precise technical vocabulary
2. **Practitioner** — "how do I use X" framing
3. **Reviewer** — "what evidence supports/contradicts X" framing

### LLM for generation

OpenRouter as the API gateway, routing to Claude Sonnet. No direct
Anthropic or OpenAI API key needed — just `OPENROUTER_API_KEY`.

```python
import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory

openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
llm = llm_factory(
    model="anthropic/claude-sonnet-4-6",
    client=openrouter_client,
)
```

**RAGAS 0.4.x note:** `LangchainLLMWrapper` is deprecated and marked for
removal. `llm_factory` is the replacement. It accepts an `openai.AsyncOpenAI`
client directly — works with OpenRouter since OpenRouter exposes an
OpenAI-compatible API. This also removes the `langchain-openai` dependency.

Embedding model: local `sentence-transformers/all-MiniLM-L6-v2` (same as
the graph's 384-d vectors). No API call for embeddings.

### Environment

```bash
# .env (gitignored)
OPENROUTER_API_KEY=sk-or-...
```

`config.py` reads `OPENROUTER_API_KEY` from env. The key is never
hardcoded or committed.

### LLM call budget

| Phase | Calls | Cost |
|---|---|---|
| Graph construction | 0 | $0 |
| Synthesizers (~260 cases × 2) | ~520 | ~$3-4 |
| Semantic validation (~260 × 1) | ~260 | ~$1-2 |
| **Total** | ~780 | **~$5-6** |

---

## Part 4: Freshness System

The #1 long-term failure mode is ground-truth rot: synthetic cases reference
node IDs that no longer exist or whose content has changed enough to
invalidate the question. The freshness system catches this without
requiring full regeneration.

### Content hash

Every RAGAS node gets a `content_hash`: SHA-256 of its `page_content`
(truncated to 16 hex chars). Every synthetic case gets a `freshness_hash`:
SHA-256 of the sorted concatenation of its expected nodes' content hashes.

```python
def compute_freshness_hash(expected_nodes: list[RagasNodePayload]) -> str:
    combined = "|".join(sorted(n.content_hash for n in expected_nodes))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
```

### Graph snapshot hash

The `SyntheticEvalSet` stores a `graph_snapshot_hash`: hash of all
node IDs + content_hashes at generation time. This detects bulk graph
changes (new paper batch ingested, finding renumbered).

### Pre-eval freshness check

Before running eval with `--synthetic`, the harness:

1. Loads `synthetic-v1.json`
2. For each case, queries Neo4j for the expected nodes
3. Recomputes content_hash from current node properties
4. Compares against stored `freshness_hash`

Results:
- **FRESH**: hash matches → run the case
- **STALE**: hash differs → skip, log warning, flag for regeneration
- **MISSING**: node not found in Neo4j → skip, log error

```python
class FreshnessResult(BaseModel):
    case_id: int
    status: str  # FRESH | STALE | MISSING
    details: str = ""

def check_freshness(cases: list[SyntheticEvalCase], session) -> list[FreshnessResult]:
    results = []
    for case in cases:
        missing = []
        current_hashes = []
        for label, node_id in case.expected:
            current = fetch_node_content(session, label, node_id)
            if current is None:
                missing.append(f"{label}:{node_id}")
            else:
                current_hashes.append(_content_hash(current))

        if missing:
            results.append(FreshnessResult(case_id=case.id, status="MISSING",
                           details=f"Nodes gone: {missing}"))
            continue

        # Recompute freshness hash from current content and compare
        combined = "|".join(sorted(current_hashes))
        current_freshness = hashlib.sha256(combined.encode()).hexdigest()[:16]
        if current_freshness != case.freshness_hash:
            results.append(FreshnessResult(case_id=case.id, status="STALE",
                           details=f"Freshness hash mismatch: {case.freshness_hash} → {current_freshness}"))
        else:
            results.append(FreshnessResult(case_id=case.id, status="FRESH"))
    return results
```

### Incremental regeneration

When stale/missing cases are detected:

```bash
# Show what's stale
python -m ragas_gen.freshness --check

# Regenerate only stale cases (keeps fresh ones intact)
python -m ragas_gen.generate --refresh-stale

# Full regeneration (rare — only after major graph restructure)
python -m ragas_gen.generate --full
```

`--refresh-stale` removes stale cases from the JSON, generates
replacements for just those slots (same synthesizer + difficulty tier),
re-validates, and writes back. Case IDs are stable — a refreshed case
keeps its original ID.

---

## Part 5: Semantic Validation

After RAGAS generates a case, a cheap LLM call checks whether the
ground truth is actually correct for the generated question. This catches
the polarity inversion problem (CONTRADICTED_BY → "What supports...?")
and unanswerable questions.

### Judge prompt

```
Given this question and the expected retrieval results, assess whether
the ground truth is correct.

Question: {query}

Expected results:
{for each expected node: label, id, page_content, and the Neo4j
relationship type that connects them}

Is each expected result genuinely relevant to the question? Would a
domain expert agree these are correct answers?

Respond with:
- valid: true/false
- confidence: 0.0-1.0
- reason: one sentence
```

### Rejection criteria

- `is_valid == false` → drop the case
- `confidence < 0.7` → drop the case
- Remaining cases get the `SemanticValidation` object stored in JSON

### Cost

~260 cases × ~500 input tokens + ~50 output tokens = ~$1-2 with Sonnet.

---

## Part 6: Eval Harness Integration

### New flags on `eval_rag.py`

```
--synthetic PATH       Load synthetic cases from JSON sidecar
--validate-synthetic   Run freshness check before eval, skip stale cases
--difficulty TIER      Filter to a specific difficulty tier
```

### Reporting changes

Per-difficulty-tier precision and MRR, reported alongside per-category:

```
  Difficulty              Precision        MRR   Cases
  ---------------------- ---------- ---------- -------
  single-hop-exact            1.000      0.850      95
  single-hop-paraphrase       0.960      0.720      48
  multi-hop                   0.880      0.550      62
  adversarial                 0.750      0.480      30
  ---------------------- ---------- ---------- -------
  SYNTHETIC OVERALL           0.940      0.680     235

  GOLDEN (41 hand-written)    0.972      0.663      41
```

Golden and synthetic results are always reported separately. No blended
headline number — that would let easy synthetic cases mask golden
regressions.

### Deduplication against golden set

Synthetic queries with >0.85 cosine similarity to any golden case get
dropped during generation. This prevents synthetic cases from inflating
precision on already-covered retrieval paths.

```python
from sentence_transformers import SentenceTransformer

def dedup_against_golden(synthetic: list[RawSyntheticCase],
                          golden_queries: list[str],
                          threshold: float = 0.85) -> list[RawSyntheticCase]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    golden_embs = model.encode(golden_queries, normalize_embeddings=True)
    kept = []
    for case in synthetic:
        case_emb = model.encode(case.query, normalize_embeddings=True)
        max_sim = (golden_embs @ case_emb).max()
        if max_sim < threshold:
            kept.append(case)
    return kept
```

---

## Part 7: Dependencies & Installation

```bash
cd ~/topo-confidence/research-graph
pip install "ragas==0.4.3" openai sentence-transformers
```

`ragas==0.4.3` pinned — the API changed significantly across 0.2→0.3→0.4.
`openai` provides the `AsyncOpenAI` client that `llm_factory` wraps for
OpenRouter. No langchain deps needed — `llm_factory` accepts the openai
client directly. RAGAS pulls in pydantic, numpy, datasets (HuggingFace).
Already have neo4j driver and sentence-transformers in the venv.

```bash
# .env (gitignored)
OPENROUTER_API_KEY=sk-or-...
```

---

## Part 8: File Plan

```
research-graph/
  ragas_gen/
    __init__.py
    schemas.py             # All Pydantic models (Part 1)
    neo4j_to_ragas.py      # Neo4j → RAGAS KG adapter (Part 2)
    generate.py            # Orchestrator: build KG → synthesize → postprocess → write JSON
    postprocess.py         # Dedup, semantic validation, difficulty tagging, freshness hash
    freshness.py           # Pre-eval freshness checker (Part 4)
    config.py              # Synthesizer weights, style distribution, personas, OpenRouter setup
  eval/
    synthetic-v1.json      # Frozen generated output, version-controlled
  eval_rag.py              # Add --synthetic, --validate-synthetic, --difficulty flags
```

---

## Part 9: Quality Gates Summary

| Gate | When | What it catches | Tool |
|---|---|---|---|
| **Schema gate 1** | Neo4j extraction | Missing embeddings, empty claims, bad rel types | `Neo4jPaper`, `Neo4jFinding` Pydantic models |
| **Schema gate 2** | RAGAS adapter | Empty page_content, wrong embedding dim, missing provenance | `RagasNodePayload` model |
| **Schema gate 3** | RAGAS output | Malformed synthesizer output, missing fields | `RawSyntheticCase` model |
| **Dedup gate** | Post-generation | Overlap with golden set or between synthetic cases | Cosine similarity, threshold 0.85 |
| **Semantic gate** | Post-generation | Wrong polarity, unanswerable questions, bad ground truth | LLM judge, ~$1-2 |
| **Schema gate 4** | Final output | Invalid IDs, empty expected, malformed JSON | `SyntheticEvalCase` model |
| **Freshness gate** | Pre-eval | Stale node IDs, changed content, deleted nodes | Content hash comparison |
| **Existence gate** | Pre-eval | Node exists in Neo4j but embedding missing | Runtime Neo4j query |

### Difficulty calibration (post-first-run)

After the first eval run with synthetic cases:
- If any tier has >95% precision → too easy for that tier, tighten
- If any tier has <50% precision → likely bad ground truth, audit that tier
- Target: 80-95% precision per tier (hard enough to be useful, not so
  hard that failures are noise)

---

## Part 10: Estimated Timeline

| Step | Time |
|---|---|
| `schemas.py` — all Pydantic models | 30 min |
| `neo4j_to_ragas.py` — adapter with validation | 45 min |
| `generate.py` — orchestrator | 30 min |
| `postprocess.py` — dedup + semantic validation + difficulty | 45 min |
| `freshness.py` — pre-eval checker | 20 min |
| First generation run (~$6 LLM) | 15 min |
| Wire into `eval_rag.py` | 20 min |
| First eval run + difficulty calibration | 20 min |
| Prune bad cases, adjust ratios | 30 min |
| **Total** | **~4 hours** |

---

## Part 11: Resolved Questions (from v1)

1. **FutureExperiment nodes**: Skipped. They test planning, not retrieval.
2. **Multi-hop depth**: 2-hop only, capped. RAGAS `find_indirect_clusters`
   with `depth_limit=2`.
3. **Answer validation**: Reference answers stored in JSON. Answer-quality
   metrics (RAGAS faithfulness/relevance) deferred to v3.
4. **Schema validation**: Pydantic throughout (not Zod — we're Python-side).
   See Part 1 + Part 8 for the full gate map.
5. **Ground-truth rot**: Freshness hash system (Part 4) with incremental
   regeneration. No wholesale replacement.

## Part 12: v2.1 Audit Fixes (RAGAS 0.4.3 compat)

Issues caught during fresh-eyes audit before implementation:

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | `Node.id` must be `uuid.UUID` in RAGAS 0.4.x, not string | Critical | UUID mapping layer in adapter + `RagasNodePayload.ragas_uuid` field |
| 2 | Forward-reference ordering: `Neo4jPaper` referenced `FindingEdge` before definition | Critical | Reordered all classes — deps defined before dependents |
| 3 | Freshness check compared individual hash `not in` combined hash string | Critical | Recompute full freshness hash from current content, compare `==` |
| 4 | `LangchainLLMWrapper` deprecated in 0.4.x, marked for removal | Medium | Replaced with `llm_factory` + `openai.AsyncOpenAI` client |
| 5 | Relationship type `"cosine_similarity"` doesn't match synthesizer default | Medium | Changed to `"summary_similarity"` (matches `MultiHopAbstractQuerySynthesizer`) |
| 6 | No RAGAS version pin — API changed across 0.2→0.3→0.4 | Medium | Pinned `ragas==0.4.3`, dropped `langchain-openai`/`langchain-community` deps |
