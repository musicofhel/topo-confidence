from __future__ import annotations

import hashlib
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from ragas_gen.schemas import (
    Difficulty,
    RagasNodePayload,
    RawSyntheticCase,
    SemanticValidation,
    SyntheticEvalCase,
)


def assign_difficulty(synth: str, style: str | None, expected_count: int) -> Difficulty:
    if style and style in ("misspelled", "poor_grammar"):
        return Difficulty.ADVERSARIAL
    if "single_hop" in synth and expected_count == 1:
        if style == "web_search_like":
            return Difficulty.SINGLE_HOP_PARAPHRASE
        return Difficulty.SINGLE_HOP_EXACT
    return Difficulty.MULTI_HOP


def compute_freshness_hash(payloads: list[RagasNodePayload]) -> str:
    combined = "|".join(sorted(p.content_hash for p in payloads))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def dedup_against_golden(
    synthetic: list[RawSyntheticCase],
    golden_queries: list[str],
    threshold: float = 0.85,
) -> tuple[list[RawSyntheticCase], int]:
    if not golden_queries or not synthetic:
        return synthetic, 0

    model = SentenceTransformer("all-MiniLM-L6-v2")
    golden_embs = model.encode(golden_queries, normalize_embeddings=True)

    kept = []
    removed = 0
    for case in synthetic:
        case_emb = model.encode(case.query, normalize_embeddings=True)
        max_sim = float((golden_embs @ case_emb).max())
        if max_sim < threshold:
            kept.append(case)
        else:
            removed += 1

    return kept, removed


def dedup_inter_synthetic(
    cases: list[RawSyntheticCase],
    threshold: float = 0.85,
) -> tuple[list[RawSyntheticCase], int]:
    if len(cases) <= 1:
        return cases, 0

    model = SentenceTransformer("all-MiniLM-L6-v2")
    queries = [c.query for c in cases]
    embs = model.encode(queries, normalize_embeddings=True)

    keep_mask = [True] * len(cases)
    for i in range(len(cases)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(cases)):
            if not keep_mask[j]:
                continue
            sim = float(embs[i] @ embs[j])
            if sim >= threshold:
                keep_mask[j] = False

    kept = [c for c, k in zip(cases, keep_mask) if k]
    removed = len(cases) - len(kept)
    return kept, removed


async def validate_semantic(
    case: RawSyntheticCase,
    client,
    model: str,
) -> SemanticValidation:
    prompt = f"""Given this question and the expected retrieval results, assess whether
the ground truth is correct.

Question: {case.query}

Expected results:
{chr(10).join(f"- {ctx[:200]}" for ctx in case.reference_contexts)}

Is each expected result genuinely relevant to the question? Would a
domain expert agree these are correct answers?

Respond with ONLY valid JSON (no markdown):
{{"valid": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        data = json.loads(text)
        return SemanticValidation(
            is_valid=data.get("valid", False),
            confidence=float(data.get("confidence", 0.0)),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        return SemanticValidation(
            is_valid=False,
            confidence=0.0,
            reason=f"Validation failed: {e}",
        )


def assign_category(synth: str, expected_count: int, node_labels: list[str]) -> str:
    labels = set(node_labels)
    if len(labels) > 1:
        return "cross-type"
    if "multi_hop" in synth:
        return "cross-type"
    if "Finding" in labels:
        return "finding"
    return "paper"
