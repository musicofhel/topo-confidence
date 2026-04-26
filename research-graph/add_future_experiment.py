"""Add a new FutureExperiment node to the research graph.

Example:
    python add_future_experiment.py \\
        --id P11-FE6 --pathway P11 \\
        --description "..." \\
        --rationale "..." \\
        --trigger "..." \\
        --status READY --priority HIGH --roi 7 \\
        --cost "4h CPU" \\
        --depends-on F-1 F-2 \\
        --would-update F-1 \\
        --triggered-by 2501.12345
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

VALID_STATUS = ["READY", "TRIGGERED", "BLOCKED", "COMPLETED", "ABANDONED"]
VALID_PRIORITY = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="e.g. P11-FE6")
    p.add_argument("--pathway", required=True, help="e.g. P11")
    p.add_argument("--description", required=True)
    p.add_argument("--rationale", default="")
    p.add_argument("--trigger", default="")
    p.add_argument("--status", default="READY", choices=VALID_STATUS)
    p.add_argument("--blocked-by", default=None,
                   help="What's preventing execution (resource, paper, prior experiment)")
    p.add_argument("--priority", default="MEDIUM", choices=VALID_PRIORITY)
    p.add_argument("--roi", type=int, required=True, help="Integer 1-10")
    p.add_argument("--cost", default="", dest="estimated_cost")
    p.add_argument("--depends-on", nargs="*", default=[],
                   help="Finding IDs (e.g. F-1 F-2) this experiment depends on")
    p.add_argument("--would-update", nargs="*", default=[],
                   help="Finding IDs whose status would change")
    p.add_argument("--triggered-by", nargs="*", default=[],
                   help="arxiv IDs of trigger papers")
    p.add_argument("--blocked-by-experiment", nargs="*", default=[],
                   help="FutureExperiment IDs that must complete first")
    args = p.parse_args()

    if not (1 <= args.roi <= 10):
        raise SystemExit("--roi must be in [1, 10]")

    today = date.today().isoformat()
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        s.run(
            """
            MERGE (fe:FutureExperiment {id: $id})
            SET fe.pathway_id = $pathway_id,
                fe.description = $description,
                fe.rationale = $rationale,
                fe.trigger = $trigger,
                fe.status = $status,
                fe.blocked_by = $blocked_by,
                fe.priority = $priority,
                fe.estimated_cost = $cost,
                fe.roi_score = $roi,
                fe.created_date = coalesce(fe.created_date, $today)
            """,
            id=args.id,
            pathway_id=args.pathway,
            description=args.description,
            rationale=args.rationale,
            trigger=args.trigger,
            status=args.status,
            blocked_by=args.blocked_by,
            priority=args.priority,
            cost=args.estimated_cost,
            roi=args.roi,
            today=today,
        )
        s.run(
            """
            MATCH (p:Pathway {id: $pid}), (fe:FutureExperiment {id: $fid})
            MERGE (p)-[:HAS_FUTURE_EXPERIMENT]->(fe)
            """,
            pid=args.pathway, fid=args.id,
        )
        for f in args.depends_on:
            s.run(
                """
                MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $fnd})
                MERGE (fe)-[:DEPENDS_ON_FINDING]->(f)
                """,
                fid=args.id, fnd=f,
            )
        for f in args.would_update:
            s.run(
                """
                MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $fnd})
                MERGE (fe)-[:WOULD_UPDATE]->(f)
                """,
                fid=args.id, fnd=f,
            )
        for arxiv_id in args.triggered_by:
            s.run("MERGE (:Paper {arxiv_id: $a})", a=arxiv_id)
            s.run(
                """
                MATCH (fe:FutureExperiment {id: $fid}), (p:Paper {arxiv_id: $a})
                MERGE (fe)-[:TRIGGERED_BY]->(p)
                """,
                fid=args.id, a=arxiv_id,
            )
        for other_id in args.blocked_by_experiment:
            s.run(
                """
                MATCH (a:FutureExperiment {id: $a}), (b:FutureExperiment {id: $b})
                MERGE (a)-[:BLOCKED_BY_EXPERIMENT]->(b)
                """,
                a=args.id, b=other_id,
            )
    drv.close()
    print(f"FutureExperiment {args.id} written.")
    print("Run `python generate_next_experiments.py` to refresh NEXT_EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
