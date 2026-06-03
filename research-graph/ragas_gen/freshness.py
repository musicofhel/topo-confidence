"""Pre-eval freshness checker for synthetic cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

from ragas_gen.config import NEO4J_BOLT, NEO4J_USER, NEO4J_PASSWORD, SYNTHETIC_OUTPUT
from ragas_gen.schemas import FreshnessResult, SyntheticEvalSet


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fetch_node_content(session, label: str, node_id: str) -> str | None:
    if label == "Paper":
        rows = session.run(
            "MATCH (p:Paper {arxiv_id: $id}) RETURN p.title AS title, p.relevance_note AS note",
            id=node_id,
        ).data()
        if rows:
            return f"{rows[0]['title']}. {rows[0].get('note', '')}".strip()
    elif label == "Finding":
        rows = session.run(
            "MATCH (f:Finding {id: $id}) RETURN f.claim AS claim",
            id=node_id,
        ).data()
        if rows:
            return f"{node_id}: {rows[0]['claim']}"
    return None


def check_freshness(eval_set: SyntheticEvalSet) -> list[FreshnessResult]:
    driver = GraphDatabase.driver(NEO4J_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []

    with driver.session() as session:
        for case in eval_set.cases:
            missing = []
            current_hashes = []
            for label, node_id in case.expected:
                content = _fetch_node_content(session, label, node_id)
                if content is None:
                    missing.append(f"{label}:{node_id}")
                else:
                    current_hashes.append(_content_hash(content))

            if missing:
                results.append(FreshnessResult(
                    case_id=case.id,
                    status="MISSING",
                    details=f"Nodes gone: {missing}",
                ))
                continue

            combined = "|".join(sorted(current_hashes))
            current_freshness = hashlib.sha256(combined.encode()).hexdigest()[:16]
            if current_freshness != case.freshness_hash:
                results.append(FreshnessResult(
                    case_id=case.id,
                    status="STALE",
                    details=f"Hash mismatch: {case.freshness_hash} → {current_freshness}",
                ))
            else:
                results.append(FreshnessResult(
                    case_id=case.id,
                    status="FRESH",
                ))

    driver.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Check synthetic case freshness")
    parser.add_argument("--input", type=Path, default=SYNTHETIC_OUTPUT,
                        help="Path to synthetic eval set JSON")
    parser.add_argument("--check", action="store_true", default=True,
                        help="Run freshness check (default)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"No synthetic eval set at {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)
    eval_set = SyntheticEvalSet(**data)

    print(f"Checking {len(eval_set.cases)} cases...")
    results = check_freshness(eval_set)

    fresh = sum(1 for r in results if r.status == "FRESH")
    stale = sum(1 for r in results if r.status == "STALE")
    missing = sum(1 for r in results if r.status == "MISSING")

    print(f"\n  FRESH: {fresh}  STALE: {stale}  MISSING: {missing}")
    for r in results:
        if r.status != "FRESH":
            print(f"  [{r.status:7s}] case {r.case_id}: {r.details}")


if __name__ == "__main__":
    main()
