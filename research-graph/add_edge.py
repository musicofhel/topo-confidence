"""Add an edge between a Finding and a Paper, or between two Findings.

Examples:
    # Finding -> Paper
    python add_edge.py F-15 CORROBORATED_BY 2410.13640 \\
        --note "Same effect on Llama-3 8B."

    python add_edge.py F-3 EXTENDED_BY 2510.04309 \\
        --experiment-idea "Try PID controller for direction rotation" --actionable

    python add_edge.py F-7 METHOD_DIFFERS 2402.03744 \\
        --theirs "EigenScore" --ours "PH summary stats"

    # Finding -> Finding
    python add_edge.py F-15 DEPENDS_ON F-2 --reason "Uses prefill DoM"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

PAPER_RELS = {"CORROBORATED_BY", "CONTRADICTED_BY", "EXTENDED_BY",
              "METHOD_DIFFERS", "EXPLAINS", "CITED"}
FINDING_RELS = {"DEPENDS_ON", "INVALIDATED_BY", "SUPERSEDED_BY", "CONTRADICTS"}


def looks_like_finding(token: str) -> bool:
    return token.startswith("F-") and token[2:].isdigit()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="F-id of the source finding")
    p.add_argument("relation", help="Edge type (e.g. CORROBORATED_BY, DEPENDS_ON)")
    p.add_argument("target", help="arxiv_id (paper) or F-id (finding)")
    # Common props (kept generic — extra ones go via --prop k=v)
    p.add_argument("--note", default=None)
    p.add_argument("--reason", default=None)
    p.add_argument("--why", default=None)
    p.add_argument("--resolution", default=None)
    p.add_argument("--mechanism", default=None)
    p.add_argument("--theirs", default=None)
    p.add_argument("--ours", default=None)
    p.add_argument("--outcome-comparison", default=None)
    p.add_argument("--their-model", default=None)
    p.add_argument("--their-benchmark", default=None)
    p.add_argument("--their-metric", default=None)
    p.add_argument("--our-metric", default=None)
    p.add_argument("--method-comparison", default=None)
    p.add_argument("--experiment-idea", default=None)
    p.add_argument("--actionable", action="store_true")
    p.add_argument(
        "--prop",
        action="append",
        default=[],
        help="Extra property as key=value; can be repeated.",
    )
    args = p.parse_args()

    target_is_finding = looks_like_finding(args.target)
    rel = args.relation.upper()

    if target_is_finding and rel not in FINDING_RELS:
        print(f"Error: {rel} is not a valid finding-to-finding relation. "
              f"Use one of {sorted(FINDING_RELS)}.")
        sys.exit(2)
    if not target_is_finding and rel not in PAPER_RELS:
        print(f"Error: {rel} is not a valid finding-to-paper relation. "
              f"Use one of {sorted(PAPER_RELS)}.")
        sys.exit(2)

    props: dict[str, object] = {
        k: v
        for k, v in {
            "note": args.note,
            "reason": args.reason,
            "why": args.why,
            "resolution": args.resolution,
            "mechanism": args.mechanism,
            "theirs": args.theirs,
            "ours": args.ours,
            "outcome_comparison": args.outcome_comparison,
            "their_model": args.their_model,
            "their_benchmark": args.their_benchmark,
            "their_metric": args.their_metric,
            "our_metric": args.our_metric,
            "method_comparison": args.method_comparison,
            "experiment_idea": args.experiment_idea,
        }.items()
        if v is not None
    }
    if args.actionable:
        props["actionable"] = True
    for kv in args.prop:
        if "=" not in kv:
            print(f"--prop must be key=value (got {kv!r})")
            sys.exit(2)
        k, v = kv.split("=", 1)
        props[k.strip()] = v.strip()

    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        if target_is_finding:
            cypher = (
                "MATCH (a:Finding {id: $src}), (b:Finding {id: $dst}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += $props"
            )
        else:
            cypher = (
                "MATCH (a:Finding {id: $src}), (b:Paper {arxiv_id: $dst}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += $props"
            )
        s.run(cypher, src=args.source, dst=args.target, props=props)
    drv.close()
    print(f"Edge {args.source} -[{rel}]-> {args.target} written.")


if __name__ == "__main__":
    main()
