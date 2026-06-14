"""Reconcile final FE status/roi/priority from a preserved NEXT_EXPERIMENTS snapshot.

Part of the 2026-06-14 high-fidelity rebuild (SPEC_REBUILD_2026-06-14.md §3.2).

Briefs carry the *initial* FE status; the *final* status (MOOTED/ANSWERED/
TRIGGERED, plus any hand edits to roi/priority) accreted in the live graph and is
not reproducible edge-for-edge by replaying briefs + premise cascades. This script
closes that residue: it parses the golden snapshot and, for any FE whose final
status / roi_score / priority still differs from the rebuilt graph, sets it to the
snapshot value and stamps provenance ``reconciled_from``.

The snapshot is the generator's own output, so two record shapes carry the truth:

  * tier / completed FEs   -> ``### <id> (<pathway>) — [ROI: <roi>, <status>, <prio>]``
  * closed-by-adjacency FEs -> ``- **<id>** [<MOOTED|ANSWERED> by <closer>, <date>] — …``
    (these never appear as ``###`` headers, so we must parse the bullets too).

Usage:
    python reconcile_from_snapshot.py golden/NEXT_EXPERIMENTS.golden-2026-06-13.md
    python reconcile_from_snapshot.py <file> --dry-run     # report only, no writes
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

PROVENANCE = "NEXT_EXPERIMENTS@2026-06-13"
RECONCILE_DATE = "2026-06-14"

# `### P11-FE1214 (P11) — [ROI: 9, TRIGGERED, HIGH]`
HEADER_RE = re.compile(
    r"^### (?P<id>\S+) \((?P<pathway>[^)]+)\) — "
    r"\[ROI: (?P<roi>[^,]+), (?P<status>[^,]+), (?P<priority>[^\]]+)\]\s*$"
)
# `- **P11-FE846** [MOOTED by P11-FE-CEILING, 2026-06-13] — …`
BULLET_RE = re.compile(
    r"^- \*\*(?P<id>[^*]+)\*\* "
    r"\[(?P<status>MOOTED|ANSWERED) by (?P<by>[^,]+), (?P<date>[^\]]+)\]"
)


def _driver():
    return GraphDatabase.driver(
        BOLT, auth=(USER, PASSWORD), notifications_min_severity="OFF",
    )


def _parse_roi(token: str) -> float | int:
    token = token.strip()
    return float(token) if "." in token else int(token)


def parse_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Return {fe_id: {status, roi, priority, pathway, closed_by, kind}}."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        m = HEADER_RE.match(line)
        if m:
            out[m["id"]] = {
                "kind": "header",
                "pathway": m["pathway"].strip(),
                "roi": _parse_roi(m["roi"]),
                "status": m["status"].strip(),
                "priority": m["priority"].strip(),
                "closed_by": None,
            }
            continue
        m = BULLET_RE.match(line)
        if m:
            out[m["id"].strip()] = {
                "kind": "bullet",
                "pathway": None,
                "roi": None,          # bullets carry no ROI; keep graph value
                "status": m["status"].strip(),
                "priority": None,     # bullets carry no priority; keep graph value
                "closed_by": m["by"].strip(),
            }
    return out


def fetch_graph_fes(session) -> dict[str, dict[str, Any]]:
    rows = session.run(
        "MATCH (fe:FutureExperiment) "
        "RETURN fe.id AS id, fe.status AS status, fe.roi_score AS roi, "
        "fe.priority AS priority, fe.closed_by AS closed_by, "
        "fe.description AS description"
    )
    return {r["id"]: dict(r) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="path to golden NEXT_EXPERIMENTS snapshot")
    ap.add_argument("--dry-run", action="store_true",
                    help="report drift but make no writes")
    args = ap.parse_args()

    snap_path = Path(args.snapshot)
    if not snap_path.exists():
        sys.exit(f"snapshot not found: {snap_path}")

    snap = parse_snapshot(snap_path)
    print(f"Parsed {len(snap)} FEs from snapshot "
          f"({sum(1 for v in snap.values() if v['kind']=='header')} headers, "
          f"{sum(1 for v in snap.values() if v['kind']=='bullet')} closed bullets)")

    unchanged = 0
    drift: list[str] = []
    missing: list[str] = []          # in snapshot, absent from rebuilt graph
    extra: list[str] = []            # in graph, absent from snapshot (post-snapshot)

    with _driver() as drv, drv.session() as s:
        graph = fetch_graph_fes(s)

        for fe_id, want in snap.items():
            have = graph.get(fe_id)
            if have is None:
                missing.append(fe_id)
                if not args.dry_run:
                    # Stub the node so the graph is structurally complete and the
                    # gap is visible (flagged), rather than silently absent.
                    s.run(
                        """
                        MERGE (fe:FutureExperiment {id: $id})
                        SET fe.status = $status,
                            fe.roi_score = $roi,
                            fe.priority = coalesce($priority, fe.priority),
                            fe.pathway_id = coalesce($pathway, fe.pathway_id),
                            fe.closed_by = coalesce($closed_by, fe.closed_by),
                            fe.description = coalesce(fe.description,
                                '(body not recovered from briefs — snapshot-only stub)'),
                            fe.snapshot_only = true,
                            fe.reconciled_from = $prov,
                            fe.reconciled_date = $date
                        """,
                        id=fe_id, status=want["status"], roi=want["roi"],
                        priority=want["priority"], pathway=want["pathway"],
                        closed_by=want["closed_by"], prov=PROVENANCE, date=RECONCILE_DATE,
                    )
                continue

            # Compare the fields the snapshot is authoritative for.
            changes: dict[str, Any] = {}
            if have["status"] != want["status"]:
                changes["status"] = want["status"]
            if want["roi"] is not None and have["roi"] != want["roi"]:
                changes["roi_score"] = want["roi"]
            if want["priority"] is not None and have["priority"] != want["priority"]:
                changes["priority"] = want["priority"]
            if want["closed_by"] is not None and have["closed_by"] != want["closed_by"]:
                changes["closed_by"] = want["closed_by"]

            if not changes:
                unchanged += 1
                continue

            drift.append(f"{fe_id}: {have['status']}->{want['status']} "
                         f"roi {have['roi']}->{want['roi']} "
                         f"prio {have['priority']}->{want['priority']}")
            if not args.dry_run:
                s.run(
                    """
                    MATCH (fe:FutureExperiment {id: $id})
                    SET fe += $changes,
                        fe.reconciled_from = $prov,
                        fe.reconciled_date = $date
                    """,
                    id=fe_id, changes=changes, prov=PROVENANCE, date=RECONCILE_DATE,
                )

        extra = sorted(set(graph) - set(snap))

    print("\n=== reconciliation summary ===")
    print(f"  unchanged (already matched snapshot): {unchanged}")
    print(f"  reconciled (status/roi/priority drift fixed): {len(drift)}")
    print(f"  MISSING from rebuilt graph (stubbed, INVESTIGATE): {len(missing)}")
    print(f"  extra in graph, not in snapshot (post-snapshot, left as-is): {len(extra)}")

    if drift:
        print("\n--- reconciled drift (first 40) ---")
        for line in drift[:40]:
            print(f"  {line}")
    if missing:
        print("\n--- MISSING (snapshot FEs not reproduced by brief/seed replay) ---")
        for fe_id in missing:
            print(f"  {fe_id}  [{snap[fe_id]['status']}]")
    if extra:
        print("\n--- extra graph FEs (not in snapshot) ---")
        for fe_id in extra[:40]:
            print(f"  {fe_id}")

    if args.dry_run:
        print("\n(dry-run — no writes made)")


if __name__ == "__main__":
    main()
