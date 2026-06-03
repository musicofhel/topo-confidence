from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Neo4j extraction models ──────────────────────────────────────────────


class FindingEdge(BaseModel):
    finding_id: str
    rel_type: str

    @field_validator("rel_type")
    @classmethod
    def valid_rel(cls, v: str) -> str:
        allowed = {
            "CORROBORATED_BY",
            "CONTRADICTED_BY",
            "EXTENDED_BY",
            "EXPLAINS",
            "METHOD_DIFFERS",
        }
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
    id: str
    claim: str
    status: str
    strength: str
    embedding: list[float] = Field(min_length=384, max_length=384)


# ── RAGAS adapter models ─────────────────────────────────────────────────


class RagasNodePayload(BaseModel):
    page_content: str = Field(min_length=10)
    entities: list[str] = Field(min_length=1)
    themes: list[str] = Field(min_length=1)
    embedding: list[float] = Field(min_length=384, max_length=384)
    summary: str = Field(min_length=5)
    neo4j_label: str
    neo4j_id: str
    ragas_uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    content_hash: str


# ── Post-processing models ───────────────────────────────────────────────


class Difficulty(str, Enum):
    SINGLE_HOP_EXACT = "single-hop-exact"
    SINGLE_HOP_PARAPHRASE = "single-hop-paraphrase"
    MULTI_HOP = "multi-hop"
    ADVERSARIAL = "adversarial"


class RawSyntheticCase(BaseModel):
    query: str
    reference_contexts: list[str]
    reference_answer: str
    synthesizer_name: str
    query_style: str | None = None
    node_ids: list[tuple[str, str]] = []


class SemanticValidation(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SyntheticEvalCase(BaseModel):
    id: int = Field(ge=1000)
    query: str = Field(min_length=5)
    expected: list[tuple[str, str]]
    category: str
    difficulty: Difficulty
    synthesizer: str
    style: str
    reference_answer: str
    freshness_hash: str
    generated_at: datetime
    semantic_validation: SemanticValidation

    @field_validator("expected")
    @classmethod
    def non_empty_expected(cls, v):
        if not v:
            raise ValueError("expected must have at least one entry")
        return v


# ── Output container ─────────────────────────────────────────────────────


class GeneratorConfig(BaseModel):
    synthesizers: list[str]
    style_distribution: dict[str, float]
    personas: list[str]
    dedup_threshold: float = 0.85
    semantic_validation_model: str = "anthropic/claude-sonnet-4-6"
    openrouter_model: str = "anthropic/claude-sonnet-4-6"


class GenerationStats(BaseModel):
    total_generated: int
    dedup_removed: int
    semantic_invalid_removed: int
    freshness_valid: int
    by_difficulty: dict[str, int]
    by_synthesizer: dict[str, int]


class SyntheticEvalSet(BaseModel):
    generated_at: datetime
    ragas_version: str
    graph_snapshot_hash: str
    generator_config: GeneratorConfig
    cases: list[SyntheticEvalCase]
    stats: GenerationStats


# ── Freshness ────────────────────────────────────────────────────────────


class FreshnessResult(BaseModel):
    case_id: int
    status: str  # FRESH | STALE | MISSING
    details: str = ""
