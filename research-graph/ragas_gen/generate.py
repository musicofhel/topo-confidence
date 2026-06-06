"""Orchestrator: build KG → synthesize → postprocess → write JSON."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ragas
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)

from ragas_gen.config import (
    DEDUP_THRESHOLD,
    PERSONAS,
    SYNTHETIC_OUTPUT,
    SYNTHESIZER_WEIGHTS,
    get_embeddings,
    get_llm,
)
from ragas_gen.neo4j_to_ragas import (
    UUID_TO_NEO4J,
    build_knowledge_graph,
    compute_graph_snapshot_hash,
)
from ragas_gen.postprocess import (
    assign_category,
    assign_difficulty,
    compute_freshness_hash,
    dedup_against_golden,
    dedup_inter_synthetic,
    validate_semantic,
)
from ragas_gen.schemas import (
    GenerationStats,
    GeneratorConfig,
    RawSyntheticCase,
    SyntheticEvalCase,
    SyntheticEvalSet,
)


def _load_golden_queries() -> list[str]:
    eval_rag = Path(__file__).resolve().parent.parent / "eval_rag.py"
    source = eval_rag.read_text()

    import re
    queries = []
    for m in re.finditer(r'EvalCase\(\s*\d+\s*,\s*"([^"]+)"', source):
        queries.append(m.group(1))
    return queries


def _testset_to_raw_cases(testset) -> list[RawSyntheticCase]:
    cases = []
    for sample in testset.samples:
        eval_sample = sample.eval_sample
        synth_name = sample.synthesizer_name

        # Map reference_contexts back to node IDs via UUID mapping
        node_ids = []
        if hasattr(eval_sample, "reference_context_ids") and eval_sample.reference_context_ids:
            import uuid
            for ctx_id in eval_sample.reference_context_ids:
                try:
                    uid = uuid.UUID(str(ctx_id)) if not isinstance(ctx_id, uuid.UUID) else ctx_id
                    if uid in UUID_TO_NEO4J:
                        node_ids.append(UUID_TO_NEO4J[uid])
                except (ValueError, AttributeError):
                    pass

        # Fallback: match reference_contexts to node page_content
        if not node_ids and hasattr(eval_sample, "reference_contexts") and eval_sample.reference_contexts:
            for ctx in eval_sample.reference_contexts:
                ctx_text = ctx[:200].strip()
                for uid, (label, nid) in UUID_TO_NEO4J.items():
                    # Fuzzy match: check if the context starts with the node content
                    pass  # filled in _match_contexts_to_nodes

        cases.append(RawSyntheticCase(
            query=eval_sample.user_input,
            reference_contexts=eval_sample.reference_contexts or [],
            reference_answer=eval_sample.reference or "",
            synthesizer_name=synth_name,
            query_style=getattr(eval_sample, "query_style", None),
            node_ids=node_ids,
        ))
    return cases


def _match_contexts_to_nodes(
    cases: list[RawSyntheticCase],
    payloads: dict,
) -> list[RawSyntheticCase]:
    """For cases with no node_ids, try to match reference_contexts to payloads."""
    content_to_key: dict[str, tuple[str, str]] = {}
    for (label, nid), payload in payloads.items():
        content_to_key[payload.page_content] = (label, nid)

    for case in cases:
        if case.node_ids:
            continue
        matched = []
        for ctx in case.reference_contexts:
            ctx_clean = ctx.strip()
            for content, key in content_to_key.items():
                if content in ctx_clean or ctx_clean in content:
                    if key not in matched:
                        matched.append(key)
                    break
        case.node_ids = matched
    return cases


def run_generation(
    testset_size: int = 260,
    output: Path | None = None,
    dry_run: bool = False,
):
    output = output or SYNTHETIC_OUTPUT
    t0 = time.time()

    print("\n=== RAGAS Synthetic Eval Generator ===\n")

    # 1. Build knowledge graph
    print("[1/6] Building knowledge graph from Neo4j...")
    kg, payloads = build_knowledge_graph()
    snapshot_hash = compute_graph_snapshot_hash(payloads)
    print(f"  Snapshot hash: {snapshot_hash}")

    if dry_run:
        print("\n  DRY RUN — skipping synthesis. KG built successfully.")
        return

    # 2. Set up RAGAS
    print("\n[2/6] Configuring RAGAS synthesizers...")
    llm = get_llm()
    embeddings = get_embeddings()

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=PERSONAS,
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), SYNTHESIZER_WEIGHTS["single_hop_specific"]),
        (MultiHopSpecificQuerySynthesizer(llm=llm), SYNTHESIZER_WEIGHTS["multi_hop_specific"]),
        (MultiHopAbstractQuerySynthesizer(llm=llm), SYNTHESIZER_WEIGHTS["multi_hop_abstract"]),
    ]
    print(f"  Synthesizers: {', '.join(s.name for s, _ in query_distribution)}")
    print(f"  Target: {testset_size} cases")

    # 3. Generate
    print(f"\n[3/6] Generating {testset_size} cases (this calls OpenRouter)...")
    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
    )
    raw_cases = _testset_to_raw_cases(testset)
    raw_cases = _match_contexts_to_nodes(raw_cases, payloads)
    total_generated = len(raw_cases)
    print(f"  Generated {total_generated} raw cases")

    # Drop cases with no matched nodes
    raw_cases = [c for c in raw_cases if c.node_ids]
    print(f"  {len(raw_cases)} cases with matched node IDs ({total_generated - len(raw_cases)} unmatched dropped)")

    # 4. Dedup
    print("\n[4/6] Deduplicating...")
    golden_queries = _load_golden_queries()
    raw_cases, golden_dedup_removed = dedup_against_golden(raw_cases, golden_queries, DEDUP_THRESHOLD)
    raw_cases, inter_dedup_removed = dedup_inter_synthetic(raw_cases, DEDUP_THRESHOLD)
    dedup_removed = golden_dedup_removed + inter_dedup_removed
    print(f"  Removed {golden_dedup_removed} golden duplicates, {inter_dedup_removed} inter-synthetic duplicates")

    # 5. Semantic validation
    print(f"\n[5/6] Semantic validation ({len(raw_cases)} cases)...")
    validated_cases: list[tuple[RawSyntheticCase, ...]] = []
    sem_invalid = 0

    from openai import AsyncOpenAI
    from ragas_gen.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
    validation_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    import asyncio as _asyncio
    sem = _asyncio.Semaphore(10)

    async def _validate_one(case):
        async with sem:
            return await validate_semantic(case, validation_client, OPENROUTER_MODEL)

    async def _validate_all():
        nonlocal sem_invalid
        tasks = [_validate_one(c) for c in raw_cases]
        validations = await asyncio.gather(*tasks, return_exceptions=True)
        for case, val in zip(raw_cases, validations):
            if isinstance(val, Exception):
                sem_invalid += 1
                continue
            if not val.is_valid or val.confidence < 0.7:
                sem_invalid += 1
                continue
            validated_cases.append((case, val))

    asyncio.run(_validate_all())
    print(f"  Valid: {len(validated_cases)}, Invalid/low-confidence: {sem_invalid}")

    # 6. Build final cases
    print(f"\n[6/6] Building final eval set...")
    final_cases: list[SyntheticEvalCase] = []
    now = datetime.now(timezone.utc)
    by_difficulty: dict[str, int] = {}
    by_synthesizer: dict[str, int] = {}

    for idx, (raw, validation) in enumerate(validated_cases):
        case_id = 1000 + idx
        node_labels = [label for label, _ in raw.node_ids]
        expected_payloads = [payloads[key] for key in raw.node_ids if key in payloads]
        difficulty = assign_difficulty(raw.synthesizer_name, raw.query_style, len(raw.node_ids))
        category = assign_category(raw.synthesizer_name, len(raw.node_ids), node_labels)
        freshness = compute_freshness_hash(expected_payloads) if expected_payloads else "unknown"

        final = SyntheticEvalCase(
            id=case_id,
            query=raw.query,
            expected=list(raw.node_ids),
            category=category,
            difficulty=difficulty,
            synthesizer=raw.synthesizer_name,
            style=raw.query_style or "perfect_grammar",
            reference_answer=raw.reference_answer,
            freshness_hash=freshness,
            generated_at=now,
            semantic_validation=validation,
        )
        final_cases.append(final)
        by_difficulty[difficulty.value] = by_difficulty.get(difficulty.value, 0) + 1
        by_synthesizer[raw.synthesizer_name] = by_synthesizer.get(raw.synthesizer_name, 0) + 1

    eval_set = SyntheticEvalSet(
        generated_at=now,
        ragas_version=ragas.__version__,
        graph_snapshot_hash=snapshot_hash,
        generator_config=GeneratorConfig(
            synthesizers=[s.name for s, _ in query_distribution],
            style_distribution={"perfect_grammar": 0.50, "web_search_like": 0.30, "misspelled": 0.10, "poor_grammar": 0.10},
            personas=[p.name for p in PERSONAS],
        ),
        cases=final_cases,
        stats=GenerationStats(
            total_generated=total_generated,
            dedup_removed=dedup_removed,
            semantic_invalid_removed=sem_invalid,
            freshness_valid=len(final_cases),
            by_difficulty=by_difficulty,
            by_synthesizer=by_synthesizer,
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(eval_set.model_dump_json(indent=2))

    elapsed = time.time() - t0
    print(f"\n  Wrote {len(final_cases)} cases to {output}")
    print(f"  Stats: {total_generated} generated → {dedup_removed} deduped → {sem_invalid} invalid → {len(final_cases)} final")
    print(f"  Difficulty: {by_difficulty}")
    print(f"  Synthesizer: {by_synthesizer}")
    print(f"  Total time: {elapsed:.1f}s")


def refresh_stale(input_path: Path | None = None) -> None:
    """Remove stale/missing cases from an existing synthetic eval set."""
    input_path = input_path or SYNTHETIC_OUTPUT
    if not input_path.exists():
        print(f"No synthetic eval set at {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    eval_set = SyntheticEvalSet(**data)
    total_before = len(eval_set.cases)
    print(f"\n=== Refresh Stale: {input_path} ===")
    print(f"  Cases before: {total_before}")

    from ragas_gen.freshness import check_freshness
    results = check_freshness(eval_set)

    fresh_ids = {r.case_id for r in results if r.status == "FRESH"}
    stale = [r for r in results if r.status != "FRESH"]

    if not stale:
        print("  All cases are fresh — nothing to remove.")
        return

    print(f"  Fresh: {len(fresh_ids)}, removing {len(stale)}:")
    for r in stale:
        print(f"    [{r.status:7s}] case {r.case_id}: {r.details}")

    eval_set.cases = [c for c in eval_set.cases if c.id in fresh_ids]
    eval_set.stats.freshness_valid = len(eval_set.cases)

    by_difficulty: dict[str, int] = {}
    for c in eval_set.cases:
        key = c.difficulty.value
        by_difficulty[key] = by_difficulty.get(key, 0) + 1
    eval_set.stats.by_difficulty = by_difficulty

    with open(input_path, "w") as f:
        f.write(eval_set.model_dump_json(indent=2))

    print(f"  Cases after: {len(eval_set.cases)} (removed {total_before - len(eval_set.cases)})")
    print(f"  Wrote: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eval cases")
    parser.add_argument("--size", type=int, default=260,
                        help="Number of cases to generate (default: 260)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: eval/synthetic-v1.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build KG only, don't call LLM")
    parser.add_argument("--full", action="store_true",
                        help="Full regeneration (ignore existing)")
    parser.add_argument("--refresh-stale", action="store_true",
                        help="Remove stale/missing cases from existing eval set")
    args = parser.parse_args()

    if args.refresh_stale:
        refresh_stale(args.output)
        return

    run_generation(
        testset_size=args.size,
        output=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
