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

VALID_STATUS = ["READY", "TRIGGERED", "BLOCKED", "COMPLETED", "ABANDONED"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fe_id", help="e.g. P3-FE1")
    p.add_argument("status", choices=VALID_STATUS)
    p.add_argument("--outcome", default=None,
                   help="What happened. Required for COMPLETED/ABANDONED.")
    p.add_argument("--blocked-by", default=None,
                   help="If status=BLOCKED, what's blocking it.")
    args = p.parse_args()

    if args.status in ("COMPLETED", "ABANDONED") and not args.outcome:
        print(f"--outcome is required when setting status to {args.status}.")
        sys.exit(2)

    today = date.today().isoformat()
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        rec = s.run(
            """
            MATCH (fe:FutureExperiment {id: $id})
            SET fe.status = $status,
                fe.outcome = coalesce($outcome, fe.outcome),
                fe.blocked_by = CASE
                    WHEN $status = 'BLOCKED' THEN coalesce($blocked, fe.blocked_by)
                    WHEN $status IN ['COMPLETED', 'ABANDONED'] THEN null
                    ELSE fe.blocked_by
                END,
                fe.completed_date = CASE
                    WHEN $status IN ['COMPLETED', 'ABANDONED'] THEN $today
                    ELSE fe.completed_date
                END
            RETURN fe.id AS id, fe.status AS status
            """,
            id=args.fe_id,
            status=args.status,
            outcome=args.outcome,
            blocked=args.blocked_by,
            today=today,
        ).single()

    drv.close()
    if not rec:
        print(f"No FutureExperiment {args.fe_id}.")
        sys.exit(1)
    print(f"FutureExperiment {rec['id']} -> {rec['status']}")
    print("Run `python generate_next_experiments.py` to refresh NEXT_EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
