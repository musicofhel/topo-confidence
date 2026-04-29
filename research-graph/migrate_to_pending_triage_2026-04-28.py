"""Migrate every :Paper node to status='pending_triage' for max-depth re-triage.

The pre-existing graph has 220 :Paper nodes in three flavors:

  - 137 'rejected' papers — admitted by the LLM admission filter but later
    status-flipped by an operator/script. Their relevance_notes are accept-
    quality (avg 295 chars, cite F-N/H-N).
  - 1 'candidate' paper — most-recent admission, never triaged.
  - ~82 'graphed' papers — but only 6 of those have actual deep briefs
    (today's Desktop backfill: 2604.18805, 2509.26560, 2604.22271,
    2405.07987, 2604.24712, 2604.22709). The other ~76 are either seed.py
    one-paragraph entries or `query.py promote` status-flips with no brief.

This migration:
  1. Snapshots every :Paper to migration_2026-04-28_papers.json (full
     reversibility from JSON).
  2. Sets status='pending_triage' on every paper.
  3. Preserves the previous state on each node:
       - prior_status_at_migration  (rejected / candidate / graphed)
       - prior_admission_note       (old relevance_note)
       - prior_brief_kind           (seed_paragraph | desktop_chat |
                                     admission_only | null)
       - prior_rejection_reason     (old rejection_reason if any)
       - prior_triaged_at           (old triaged_at if any)

Untouched: Findings, Pathways, FutureExperiments (37, including TRIGGERED_BY
edges from P11-FE15..FE22 to the 6 backfill papers), Experiments, Artifacts.
HYPOTHESES.md and PAPER_INDEX.md are not modified — deep briefs append /
update via promote_brief.py downstream.

Idempotent: papers that already have prior_status_at_migration set are
skipped on re-run.

Usage:
    python migrate_to_pending_triage_2026-04-28.py --dry-run    # preview
    python migrate_to_pending_triage_2026-04-28.py --apply      # commit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

SNAPSHOT_PATH = ROOT / "migration_2026-04-28_papers.json"

# 6 papers backfilled from today's Desktop session into FE15..FE22 / H-15..H-22.
BACKFILL_IDS: set[str] = {
    "2604.18805",
    "2509.26560",
    "2604.22271",
    "2405.07987",
    "2604.24712",
    "2604.22709",
}


def _seed_arxiv_ids() -> set[str]:
    """Import seed.py's PAPERS list to get the canonical seed arxiv IDs.

    Avoids hardcoding — if seed.py grows a new paper we still classify it.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from seed import PAPERS  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return {p["arxiv_id"] for p in PAPERS}


def _classify_brief_kind(
    arxiv_id: str, current_status: str | None, seed_ids: set[str]
) -> str | None:
    if arxiv_id in BACKFILL_IDS:
        return "desktop_chat"
    if arxiv_id in seed_ids:
        return "seed_paragraph"
    if current_status == "graphed":
        return "admission_only"
    # rejected / candidate / unknown — admission filter wrote relevance_note,
    # but no brief of any kind exists.
    return None


def _snapshot_papers(session) -> list[dict[str, Any]]:
    """Pull every :Paper with all properties and adjacent edges."""
    return list(
        session.run(
            """
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[r_out]->(out_node)
            OPTIONAL MATCH (in_node)-[r_in]->(p)
            WITH p,
                 collect(DISTINCT {
                     type: type(r_out),
                     props: properties(r_out),
                     to_label: head(labels(out_node)),
                     to_id: coalesce(out_node.id, out_node.arxiv_id, out_node.name)
                 }) AS out_edges,
                 collect(DISTINCT {
                     type: type(r_in),
                     props: properties(r_in),
                     from_label: head(labels(in_node)),
                     from_id: coalesce(in_node.id, in_node.arxiv_id, in_node.name)
                 }) AS in_edges
            RETURN p.arxiv_id AS arxiv_id,
                   properties(p) AS props,
                   [e IN out_edges WHERE e.type IS NOT NULL] AS out_edges,
                   [e IN in_edges WHERE e.type IS NOT NULL] AS in_edges
            ORDER BY p.arxiv_id
            """
        ).data()
    )


def _serialize(obj: Any) -> Any:
    """Make Neo4j datetimes / dates JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "iso_format"):  # neo4j.time.DateTime
        return obj.iso_format()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _summarize(papers: list[dict[str, Any]], seed_ids: set[str]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_brief_kind: dict[str, int] = {}
    by_target_tier: dict[int, int] = {}
    already_migrated = 0

    for row in papers:
        props = row["props"]
        status = props.get("status")
        by_status[status or "<null>"] = by_status.get(status or "<null>", 0) + 1

        if props.get("prior_status_at_migration"):
            already_migrated += 1
            continue

        brief_kind = _classify_brief_kind(row["arxiv_id"], status, seed_ids)
        key = brief_kind or "<no_brief>"
        by_brief_kind[key] = by_brief_kind.get(key, 0) + 1

        # Replicate query.py's tier ordering for the report.
        if status == "rejected":
            tier = 1
        elif status == "candidate":
            tier = 2
        elif brief_kind == "admission_only":
            tier = 3
        elif brief_kind == "seed_paragraph":
            tier = 4
        elif brief_kind == "desktop_chat":
            tier = 5
        else:
            tier = 6
        by_target_tier[tier] = by_target_tier.get(tier, 0) + 1

    return {
        "total_papers": len(papers),
        "already_migrated_skip": already_migrated,
        "by_current_status": by_status,
        "by_assigned_brief_kind": by_brief_kind,
        "by_target_tier": dict(sorted(by_target_tier.items())),
    }


def _apply_migration(session, papers: list[dict[str, Any]], seed_ids: set[str]) -> int:
    n = 0
    for row in papers:
        props = row["props"]
        if props.get("prior_status_at_migration"):
            continue  # already migrated — idempotent

        old_status = props.get("status")
        brief_kind = _classify_brief_kind(row["arxiv_id"], old_status, seed_ids)

        session.run(
            """
            MATCH (p:Paper {arxiv_id: $a})
            SET p.status = 'pending_triage',
                p.prior_status_at_migration = $old_status,
                p.prior_admission_note = $old_note,
                p.prior_brief_kind = $brief_kind,
                p.prior_rejection_reason = $old_reason,
                p.prior_triaged_at = $old_triaged_at,
                p.migrated_at = datetime()
            """,
            a=row["arxiv_id"],
            old_status=old_status,
            old_note=props.get("relevance_note"),
            brief_kind=brief_kind,
            old_reason=props.get("rejection_reason"),
            old_triaged_at=props.get("triaged_at"),
        )
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true",
                     help="Snapshot to JSON and print summary, no writes.")
    grp.add_argument("--apply", action="store_true",
                     help="Snapshot then commit status flips.")
    p.add_argument("--snapshot-path", default=str(SNAPSHOT_PATH),
                   help="Where to write the JSON snapshot.")
    args = p.parse_args()

    seed_ids = _seed_arxiv_ids()
    print(f"Loaded {len(seed_ids)} seed arxiv IDs from seed.py")
    print(f"Backfill (Desktop) IDs: {sorted(BACKFILL_IDS)}")

    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    try:
        with drv.session() as session:
            papers = _snapshot_papers(session)
            print(f"Pulled {len(papers)} :Paper nodes from {BOLT}")

            snapshot_path = Path(args.snapshot_path)
            with snapshot_path.open("w") as fh:
                json.dump([_serialize(row) for row in papers], fh, indent=2,
                          default=str)
            print(f"Snapshot written: {snapshot_path}")

            summary = _summarize(papers, seed_ids)
            print("\nMigration summary:")
            print(json.dumps(summary, indent=2))

            if args.dry_run:
                print("\n[dry-run] No writes. Re-run with --apply to commit.")
                return

            print("\n[apply] Committing status flips...")
            n = _apply_migration(session, papers, seed_ids)
            print(f"[apply] Updated {n} papers to status='pending_triage'.")
            print(f"\nNext: python query.py pending --ids-only | head")
            print("      bash triage_pending.sh   # once briefs/ exists and "
                  "loop is calibrated")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
