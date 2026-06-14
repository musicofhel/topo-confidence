"""Semantic dedup of FutureExperiment nodes (SPEC_REBUILD_2026-06-14.md §8).

326 triage briefs were promoted independently, so several propose near-identical
experiments (the canonical case is FE335/FE337). No dedup key exists today. This
pass, run *after* embeddings are backfilled:

  1. Loads every open FE's 384-d MiniLM embedding (normalized -> dot == cosine).
  2. Finds pairs with cosine >= --threshold (default 0.88, conservative).
  3. Keeps the higher roi_score (tie -> lexicographically smaller id); closes the
     loser as ANSWERED with closed_by=<keeper>, and unions the loser's
     TRIGGERED_BY edges onto the keeper so no trigger paper is orphaned.

--dry-run (default behaviour is dry unless --apply) writes a review list to
briefs/dedup-<date>.md and makes NO graph writes. Review, then re-run with
--apply.

    python dedup_future_experiments.py --threshold 0.88            # dry-run
    python dedup_future_experiments.py --threshold 0.88 --apply
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

OPEN_STATUSES = ["READY", "TRIGGERED", "BLOCKED"]
REPORT = ROOT / "briefs" / "dedup-2026-06-14.md"


def _driver():
    return GraphDatabase.driver(
        BOLT, auth=(USER, PASSWORD), notifications_min_severity="OFF",
    )


def fetch_open_fes(session):
    rows = session.run(
        """
        MATCH (fe:FutureExperiment)
        WHERE fe.status IN $open AND fe.embedding IS NOT NULL
        RETURN fe.id AS id, fe.roi_score AS roi, fe.description AS description,
               fe.embedding AS embedding
        ORDER BY fe.id
        """,
        open=OPEN_STATUSES,
    )
    return [dict(r) for r in rows]


def find_pairs(fes, threshold):
    """Return [(sim, keeper_idx, loser_idx)] sorted by descending sim."""
    if not fes:
        return []
    mat = np.array([f["embedding"] for f in fes], dtype=np.float64)
    # embeddings are already L2-normalized at backfill; renormalize defensively
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    sims = mat @ mat.T
    pairs = []
    n = len(fes)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s >= threshold:
                # keeper = higher roi, tie -> lexicographically smaller id
                a, b = fes[i], fes[j]
                roi_a = a["roi"] if a["roi"] is not None else -1
                roi_b = b["roi"] if b["roi"] is not None else -1
                if (roi_a, ) > (roi_b, ) or (roi_a == roi_b and a["id"] < b["id"]):
                    keeper, loser = i, j
                else:
                    keeper, loser = j, i
                pairs.append((s, keeper, loser))
    pairs.sort(key=lambda t: -t[0])
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.88)
    ap.add_argument("--apply", action="store_true",
                    help="perform closures (default: dry-run review only)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="if >0, only consider each FE's top-K neighbours (unused "
                         "for the full O(n^2) pass; reserved)")
    args = ap.parse_args()

    with _driver() as drv, drv.session() as s:
        fes = fetch_open_fes(s)
        print(f"open FEs with embeddings: {len(fes)}")
        pairs = find_pairs(fes, args.threshold)
        print(f"pairs at cosine >= {args.threshold}: {len(pairs)}")

        # Greedy: walk pairs by descending sim; a node already closed as a loser
        # can no longer be a keeper or get re-closed.
        closed: dict[str, str] = {}       # loser_id -> keeper_id
        report_rows = []
        for sim, ki, li in pairs:
            keeper_id, loser_id = fes[ki]["id"], fes[li]["id"]
            if loser_id in closed or keeper_id in closed:
                continue
            closed[loser_id] = keeper_id
            report_rows.append((sim, keeper_id, loser_id,
                                fes[ki]["description"], fes[li]["description"]))

        # ---- write review report (always) ----
        lines = [
            "# FE semantic dedup review — 2026-06-14",
            "",
            f"_Threshold: cosine >= {args.threshold}. "
            f"{len(report_rows)} loser FEs would close as ANSWERED._",
            f"_Mode: {'APPLIED' if args.apply else 'DRY-RUN (no writes)'}._",
            "",
            "| cosine | keeper | loser (closed ANSWERED) | keeper desc | loser desc |",
            "|---|---|---|---|---|",
        ]
        for sim, k, l, kd, ld in report_rows:
            kd_s = (kd or "")[:90].replace("|", "\\|").replace("\n", " ")
            ld_s = (ld or "")[:90].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {sim:.3f} | {k} | {l} | {kd_s} | {ld_s} |")
        REPORT.write_text("\n".join(lines) + "\n")
        print(f"wrote review report -> {REPORT}")

        # FE335/FE337 sanity check
        fe335_337 = [r for r in report_rows
                     if {r[1], r[2]} & {"P11-FE335", "P11-FE337", "FE335", "FE337"}]
        if fe335_337:
            print("  FE335/FE337 collapse CONFIRMED:",
                  [(r[1], r[2], round(r[0], 3)) for r in fe335_337])
        else:
            print("  note: FE335/FE337 not collapsed at this threshold "
                  "(check ids/threshold)")

        if not args.apply:
            print("\n(dry-run — no graph writes. Re-run with --apply to close losers.)")
            return

        # ---- apply closures ----
        for sim, keeper_id, loser_id, _, _ in report_rows:
            s.run(
                """
                MATCH (loser:FutureExperiment {id: $loser})
                MATCH (keeper:FutureExperiment {id: $keeper})
                // union trigger edges onto the keeper
                WITH loser, keeper
                OPTIONAL MATCH (loser)-[t:TRIGGERED_BY]->(p:Paper)
                FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
                    MERGE (keeper)-[:TRIGGERED_BY]->(p))
                WITH DISTINCT loser, keeper
                SET loser.status = 'ANSWERED',
                    loser.closed_by = keeper.id,
                    loser.completed_date = '2026-06-14',
                    loser.outcome = coalesce(loser.outcome, '') +
                        ' [ANSWERED by dedup: near-duplicate of ' + keeper.id +
                        ' (cosine ' + toString($sim) + ')]'
                MERGE (loser)-[r:ANSWERED_BY]->(keeper)
                SET r.reason = 'semantic dedup cosine ' + toString($sim),
                    r.date = '2026-06-14'
                """,
                loser=loser_id, keeper=keeper_id, sim=round(sim, 3),
            )
        print(f"\nAPPLIED: closed {len(report_rows)} loser FEs as ANSWERED.")


if __name__ == "__main__":
    main()
