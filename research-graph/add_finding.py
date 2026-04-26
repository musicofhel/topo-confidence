"""Add a new Finding node to the research graph.

Example:
    python add_finding.py --id F-15 --claim "..." \\
        --evidence P11-E10 --strength MODERATE \\
        --counterargument "..." --overturned-by "..."
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="e.g. F-15")
    p.add_argument("--claim", required=True)
    p.add_argument("--strength", required=True, choices=["STRONG", "MODERATE", "PRELIMINARY"])
    p.add_argument("--status", default="ACTIVE",
                   choices=["ACTIVE", "INVALIDATED", "SUPERSEDED"])
    p.add_argument("--evidence", nargs="+", default=[],
                   help="Experiment IDs (e.g. P11-E10 P11-E11)")
    p.add_argument("--controls-passed", nargs="*", default=[])
    p.add_argument("--controls-pending", nargs="*", default=[])
    p.add_argument("--counterargument", default="")
    p.add_argument("--overturned-by", default="")
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args()

    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        s.run(
            """
            MERGE (f:Finding {id: $id})
            SET f.claim = $claim,
                f.strength = $strength,
                f.status = $status,
                f.evidence = $evidence,
                f.controls_passed = $controls_passed,
                f.controls_pending = $controls_pending,
                f.strongest_counterargument = $counterarg,
                f.would_be_overturned_by = $overturn
            """,
            id=args.id,
            claim=args.claim,
            strength=args.strength,
            status=args.status,
            evidence=args.evidence,
            controls_passed=args.controls_passed,
            controls_pending=args.controls_pending,
            counterarg=args.counterargument,
            overturn=args.overturned_by,
        )
        for exp_id in args.evidence:
            s.run(
                """
                MATCH (f:Finding {id: $f}), (x:Experiment {id: $x})
                MERGE (x)-[:PRODUCED]->(f)
                """,
                f=args.id,
                x=exp_id,
            )
        for tag in args.tags:
            s.run("MERGE (:Tag {name: $t})", t=tag)
            s.run(
                """
                MATCH (f:Finding {id: $f}), (t:Tag {name: $tag})
                MERGE (f)-[:TAGGED]->(t)
                """,
                f=args.id,
                tag=tag,
            )
    drv.close()
    print(f"Finding {args.id} written.")


if __name__ == "__main__":
    main()
