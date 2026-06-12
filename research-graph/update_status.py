"""Update the status of an existing FutureExperiment.

Examples:
    # Mark as completed with an outcome
    python update_status.py P3-FE1 COMPLETED \\
        --outcome "PID steering hit 51.2% acc, +2.6 over base; replicated STU-PID delta"

    # Re-classify (e.g. paper appears that triggers a READY experiment)
    python update_status.py P5-FE1 TRIGGERED

    # Abandon
    python update_status.py P9-FE2 ABANDONED \\
        --outcome "Persistence landscapes also pass Gaussian null — PH definitively dead."

    # Moot (premise refuted by an adjacent experiment) / Answer (question already
    # settled by an adjacent result). --by records provenance and, when the id
    # matches a graph node, a MOOTED_BY / ANSWERED_BY edge:
    python update_status.py P11-FE44 MOOTED \\
        --by FE269 --outcome "DoM is a diagnostic readout, not a causal lever."
    python update_status.py P11-FE58 ANSWERED \\
        --by P11-FE-EDGEGEN --outcome "v6 bake-off already measured this comparison."
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

VALID_STATUS = ["READY", "TRIGGERED", "BLOCKED", "COMPLETED", "ABANDONED",
                "MOOTED", "ANSWERED"]
TERMINAL_STATUS = ["COMPLETED", "ABANDONED", "MOOTED", "ANSWERED"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fe_id", help="e.g. P3-FE1")
    p.add_argument("status", choices=VALID_STATUS)
    p.add_argument("--outcome", default=None,
                   help="What happened. Required for COMPLETED/ABANDONED.")
    p.add_argument("--blocked-by", default=None,
                   help="If status=BLOCKED, what's blocking it.")
    p.add_argument("--by", default=None,
                   help="For MOOTED/ANSWERED: id of the experiment/premise that "
                        "closed this FE (e.g. FE269, P11-FE-EDGEGEN). Stored as "
                        "fe.closed_by and, when a matching node exists, as a "
                        "MOOTED_BY/ANSWERED_BY edge.")
    args = p.parse_args()

    if args.status in TERMINAL_STATUS and not args.outcome:
        print(f"--outcome is required when setting status to {args.status}.")
        sys.exit(2)
    if args.status in ("MOOTED", "ANSWERED") and not args.by:
        print(f"--by is required when setting status to {args.status} "
              "(what closed it?).")
        sys.exit(2)

    today = date.today().isoformat()
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        rec = s.run(
            """
            MATCH (fe:FutureExperiment {id: $id})
            SET fe.status = $status,
                fe.outcome = coalesce($outcome, fe.outcome),
                fe.closed_by = CASE
                    WHEN $status IN ['MOOTED', 'ANSWERED'] THEN $by
                    ELSE fe.closed_by
                END,
                fe.blocked_by = CASE
                    WHEN $status = 'BLOCKED' THEN coalesce($blocked, fe.blocked_by)
                    WHEN $status IN $terminal THEN null
                    ELSE fe.blocked_by
                END,
                fe.completed_date = CASE
                    WHEN $status IN $terminal THEN $today
                    ELSE fe.completed_date
                END
            RETURN fe.id AS id, fe.status AS status
            """,
            id=args.fe_id,
            status=args.status,
            outcome=args.outcome,
            blocked=args.blocked_by,
            by=args.by,
            today=today,
            terminal=TERMINAL_STATUS,
        ).single()

        if rec and args.status in ("MOOTED", "ANSWERED") and args.by:
            edge = "MOOTED_BY" if args.status == "MOOTED" else "ANSWERED_BY"
            linked = s.run(
                f"""
                MATCH (fe:FutureExperiment {{id: $id}})
                MATCH (src) WHERE (src:FutureExperiment OR src:Experiment
                                   OR src:Premise OR src:Finding)
                            AND src.id = $by
                MERGE (fe)-[r:{edge}]->(src)
                SET r.reason = $outcome, r.date = $today
                RETURN count(src) AS n
                """,
                id=args.fe_id, by=args.by, outcome=args.outcome, today=today,
            ).single()
            if linked and linked["n"] == 0:
                print(f"  (no graph node with id {args.by!r} — provenance kept "
                      "as fe.closed_by property only)")

    drv.close()
    if not rec:
        print(f"No FutureExperiment {args.fe_id}.")
        sys.exit(1)
    print(f"FutureExperiment {rec['id']} -> {rec['status']}")
    print("Run `python generate_next_experiments.py` to refresh NEXT_EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
