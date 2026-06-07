"""CLI query interface for the topo-confidence research graph.

Examples:
    python query.py corroborators F-1
    python query.py contradictors F-10
    python query.py extensions F-3
    python query.py related F-6
    python query.py novelty "prefill direction is orthogonal"
    python query.py timeline
    python query.py status-report
    python query.py subgraph F-2 --depth 2
    python query.py paper 2410.13640
    python query.py pending                # tier-ordered list of pending_triage papers
    python query.py pending --ids-only     # arxiv IDs only — feeds triage_pending.sh
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _driver():
    # Suppress "property key does not exist" warnings — these fire for nullable
    # fields (completed_date, outcome) that don't get written until an experiment
    # runs. Functionally harmless, visually noisy.
    return GraphDatabase.driver(
        BOLT,
        auth=(USER, PASSWORD),
        notifications_min_severity="OFF",
    )


def _run(cypher: str, **params) -> list[dict[str, Any]]:
    with _driver() as drv, drv.session() as session:
        return [dict(record) for record in session.run(cypher, **params)]


# ---------------------------------------------------------------------------
# Library functions (importable)
# ---------------------------------------------------------------------------

def corroborators(finding_id: str) -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (f:Finding {id: $id})-[r:CORROBORATED_BY]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, properties(r) AS edge
        ORDER BY p.year DESC
        """,
        id=finding_id,
    )


def contradictors(finding_id: str) -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (f:Finding {id: $id})-[r:CONTRADICTED_BY]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, properties(r) AS edge
        ORDER BY p.year DESC
        """,
        id=finding_id,
    )


def extensions(finding_id: str) -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (f:Finding {id: $id})-[r:EXTENDED_BY]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, properties(r) AS edge
        ORDER BY coalesce(r.actionable, false) DESC, p.year DESC
        """,
        id=finding_id,
    )


def method_differs(finding_id: str) -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (f:Finding {id: $id})-[r:METHOD_DIFFERS]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, properties(r) AS edge
        """,
        id=finding_id,
    )


def explains(finding_id: str) -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (f:Finding {id: $id})-[r:EXPLAINS]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, properties(r) AS edge
        """,
        id=finding_id,
    )


def related_findings(finding_id: str) -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (f:Finding {id: $id})-[r]-(g:Finding)
        WHERE type(r) IN ['DEPENDS_ON', 'INVALIDATED_BY', 'SUPERSEDED_BY', 'CONTRADICTS']
        WITH f, g, r,
             CASE WHEN startNode(r) = f THEN 'outgoing' ELSE 'incoming' END AS direction
        RETURN g.id AS other_id, g.claim AS other_claim, g.status AS other_status,
               type(r) AS relation, direction, properties(r) AS edge
        ORDER BY g.id
        """,
        id=finding_id,
    )


def novelty_check(claim: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fulltext search across findings + papers. Lucene query syntax allowed.

    For free-text safety we wrap the query so reserved characters don't blow up.
    """
    safe = _escape_lucene(claim)
    finding_hits = _run(
        """
        CALL db.index.fulltext.queryNodes('finding_claims', $q)
        YIELD node, score
        RETURN 'Finding' AS kind, node.id AS id, node.claim AS text,
               node.status AS status, score
        ORDER BY score DESC LIMIT $limit
        """,
        q=safe,
        limit=limit,
    )
    paper_hits = _run(
        """
        CALL db.index.fulltext.queryNodes('paper_relevance', $q)
        YIELD node, score
        RETURN 'Paper' AS kind, node.arxiv_id AS id,
               coalesce(node.title, '') + ' — ' + coalesce(node.relevance_note, '') AS text,
               node.year AS status, score
        ORDER BY score DESC LIMIT $limit
        """,
        q=safe,
        limit=limit,
    )
    return finding_hits + paper_hits


def pathway_timeline() -> list[dict[str, Any]]:
    return _run(
        """
        MATCH (p:Pathway)
        OPTIONAL MATCH (p)-[:HAS_EXPERIMENT]->(e:Experiment)
        WITH p, count(e) AS exp_count
        RETURN p.id AS id, p.name AS name, p.status AS status,
               p.summary AS summary, exp_count
        ORDER BY p.id
        """
    )


def finding_status_report() -> dict[str, list[dict[str, Any]]]:
    rows = _run(
        """
        MATCH (f:Finding)
        OPTIONAL MATCH (f)-[:DEPENDS_ON]->(d:Finding)
        WITH f, collect(d.id) AS depends_on
        RETURN f.id AS id, f.claim AS claim, f.strength AS strength,
               f.status AS status, f.controls_pending AS pending,
               depends_on
        ORDER BY f.id
        """
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["status"], []).append(row)
    return grouped


def subgraph_for_finding(finding_id: str, depth: int = 2) -> dict[str, Any]:
    """Return the full neighborhood as a dict suitable for LLM context."""
    finding = _run(
        "MATCH (f:Finding {id: $id}) RETURN properties(f) AS props",
        id=finding_id,
    )
    if not finding:
        return {"error": f"No finding {finding_id}"}
    payload: dict[str, Any] = {"finding": finding[0]["props"]}

    payload["produced_by_experiments"] = _run(
        """
        MATCH (e:Experiment)-[:PRODUCED]->(f:Finding {id: $id})
        RETURN e.id AS id, e.name AS name, e.verdict AS verdict,
               e.hypothesis AS hypothesis, e.result AS result,
               e.result_json AS result_json
        ORDER BY e.id
        """,
        id=finding_id,
    )
    payload["corroborated_by"] = corroborators(finding_id)
    payload["contradicted_by"] = contradictors(finding_id)
    payload["extended_by"] = extensions(finding_id)
    payload["method_differs"] = method_differs(finding_id)
    payload["explains"] = explains(finding_id)
    payload["related_findings"] = related_findings(finding_id)
    payload["tags"] = [
        r["name"]
        for r in _run(
            "MATCH (:Finding {id: $id})-[:TAGGED]->(t:Tag) RETURN t.name AS name",
            id=finding_id,
        )
    ]

    if depth >= 2:
        # Pull adjacent findings' immediate neighbors (papers only — not full recursion)
        payload["second_order_papers"] = _run(
            """
            MATCH (f:Finding {id: $id})-[:DEPENDS_ON|INVALIDATED_BY|SUPERSEDED_BY|CONTRADICTS]-(g:Finding)
            MATCH (g)-[r]->(p:Paper)
            RETURN DISTINCT g.id AS via_finding, p.arxiv_id AS arxiv_id,
                   p.title AS title, type(r) AS relation, properties(r) AS edge
            ORDER BY via_finding, arxiv_id
            """,
            id=finding_id,
        )
    return payload


def future_experiments(
    pathway_id: str | None = None,
    status: str | None = None,
    min_roi: int | None = None,
) -> list[dict[str, Any]]:
    """Return future experiments, optionally filtered. Sorted by roi_score desc."""
    where = []
    params: dict[str, Any] = {}
    if pathway_id:
        where.append("fe.pathway_id = $pid")
        params["pid"] = pathway_id
    if status:
        where.append("fe.status = $status")
        params["status"] = status
    if min_roi is not None:
        where.append("fe.roi_score >= $min_roi")
        params["min_roi"] = min_roi
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    return _run(
        f"""
        MATCH (fe:FutureExperiment)
        {where_clause}
        OPTIONAL MATCH (fe)-[:TRIGGERED_BY]->(p:Paper)
        OPTIONAL MATCH (fe)-[:DEPENDS_ON_FINDING]->(d:Finding)
        OPTIONAL MATCH (fe)-[:WOULD_UPDATE]->(u:Finding)
        WITH fe,
             collect(DISTINCT p.arxiv_id) AS triggered_by,
             collect(DISTINCT d.id) AS depends_on,
             collect(DISTINCT u.id) AS would_update
        RETURN fe.id AS id, fe.pathway_id AS pathway_id,
               fe.description AS description, fe.rationale AS rationale,
               fe.trigger AS trigger, fe.status AS status,
               fe.blocked_by AS blocked_by, fe.priority AS priority,
               fe.estimated_cost AS estimated_cost, fe.roi_score AS roi_score,
               fe.created_date AS created_date, fe.completed_date AS completed_date,
               fe.outcome AS outcome,
               [x IN triggered_by WHERE x IS NOT NULL] AS triggered_by,
               [x IN depends_on WHERE x IS NOT NULL] AS depends_on,
               [x IN would_update WHERE x IS NOT NULL] AS would_update
        ORDER BY fe.roi_score DESC, fe.id
        """,
        **params,
    )


def triggered_experiments() -> list[dict[str, Any]]:
    """Return all future experiments with status=TRIGGERED."""
    return future_experiments(status="TRIGGERED")


def blocked_experiments() -> list[dict[str, Any]]:
    """Return all BLOCKED future experiments with their blocked_by reason."""
    rows = future_experiments(status="BLOCKED")
    if rows:
        return rows
    # Many "READY" rows have a non-null blocked_by (blocked on a resource, not a status).
    # Surface those too.
    return _run(
        """
        MATCH (fe:FutureExperiment)
        WHERE fe.blocked_by IS NOT NULL AND fe.status <> 'COMPLETED'
        RETURN fe.id AS id, fe.pathway_id AS pathway_id,
               fe.description AS description, fe.status AS status,
               fe.blocked_by AS blocked_by, fe.priority AS priority,
               fe.roi_score AS roi_score, fe.estimated_cost AS estimated_cost
        ORDER BY fe.roi_score DESC, fe.id
        """
    )


def experiment_impact(fe_id: str) -> dict[str, Any]:
    """Full impact analysis for a single future experiment."""
    base = _run(
        """
        MATCH (fe:FutureExperiment {id: $id})
        RETURN properties(fe) AS props
        """,
        id=fe_id,
    )
    if not base:
        return {"error": f"No FutureExperiment {fe_id}"}
    payload: dict[str, Any] = {"future_experiment": base[0]["props"]}

    payload["triggered_by_papers"] = _run(
        """
        MATCH (:FutureExperiment {id: $id})-[r:TRIGGERED_BY]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, properties(r) AS edge
        """,
        id=fe_id,
    )
    payload["depends_on_findings"] = _run(
        """
        MATCH (:FutureExperiment {id: $id})-[r:DEPENDS_ON_FINDING]->(f:Finding)
        RETURN f.id AS id, f.claim AS claim, f.status AS status,
               f.strength AS strength, properties(r) AS edge
        ORDER BY f.id
        """,
        id=fe_id,
    )
    payload["would_update_findings"] = _run(
        """
        MATCH (:FutureExperiment {id: $id})-[r:WOULD_UPDATE]->(f:Finding)
        RETURN f.id AS id, f.claim AS claim, f.status AS status,
               f.strength AS strength, properties(r) AS edge
        ORDER BY f.id
        """,
        id=fe_id,
    )
    payload["blocked_by_experiments"] = _run(
        """
        MATCH (:FutureExperiment {id: $id})-[:BLOCKED_BY_EXPERIMENT]->(other:FutureExperiment)
        RETURN other.id AS id, other.description AS description, other.status AS status,
               other.priority AS priority
        ORDER BY other.id
        """,
        id=fe_id,
    )
    return payload


def highest_roi(n: int = 10) -> list[dict[str, Any]]:
    """Top N future experiments by roi_score (active only — excludes COMPLETED/ABANDONED)."""
    return _run(
        """
        MATCH (fe:FutureExperiment)
        WHERE fe.status IN ['READY', 'TRIGGERED', 'BLOCKED']
        OPTIONAL MATCH (fe)-[:TRIGGERED_BY]->(p:Paper)
        OPTIONAL MATCH (fe)-[:DEPENDS_ON_FINDING]->(d:Finding)
        OPTIONAL MATCH (fe)-[:WOULD_UPDATE]->(u:Finding)
        WITH fe,
             collect(DISTINCT p.arxiv_id) AS triggered_by,
             collect(DISTINCT d.id) AS depends_on,
             collect(DISTINCT u.id) AS would_update
        RETURN fe.id AS id, fe.pathway_id AS pathway_id,
               fe.description AS description, fe.status AS status,
               fe.priority AS priority, fe.roi_score AS roi_score,
               fe.estimated_cost AS estimated_cost,
               fe.blocked_by AS blocked_by,
               [x IN triggered_by WHERE x IS NOT NULL] AS triggered_by,
               [x IN depends_on WHERE x IS NOT NULL] AS depends_on,
               [x IN would_update WHERE x IS NOT NULL] AS would_update
        ORDER BY fe.roi_score DESC, fe.id
        LIMIT $n
        """,
        n=n,
    )


def watchlist() -> list[dict[str, Any]]:
    """Papers referenced as TRIGGERED_BY targets — monitor for follow-ups."""
    return _run(
        """
        MATCH (fe:FutureExperiment)-[:TRIGGERED_BY]->(p:Paper)
        WHERE fe.status <> 'COMPLETED'
        WITH p, collect(DISTINCT {fe_id: fe.id, status: fe.status}) AS triggers
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.repo_url AS repo_url, triggers
        ORDER BY p.year DESC, p.arxiv_id
        """
    )


def pending_triage_papers() -> list[dict[str, Any]]:
    """Papers awaiting deep triage (status='pending_triage'), tier-ordered.

    Tier priority (most signal-dense first), driven by migration-preserved fields:
      1. prior_status_at_migration='rejected'   — admitted by LLM, then operator-flipped
      2. prior_status_at_migration='candidate'  — most recent admission, never triaged
      3. prior_brief_kind='admission_only'      — graphed via triage-promote, no brief
      4. prior_brief_kind='seed_paragraph'      — seed.py one-paragraph notes
      5. prior_brief_kind='desktop_chat'        — Desktop session backfill briefs
      6. (no prior_* fields)                    — freshly admitted post-migration
    """
    return _run(
        """
        MATCH (p:Paper)
        WHERE p.status IN ['pending_triage', 'candidate']
        OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
        WITH p, collect(DISTINCT t.name) AS tags,
             CASE p.prior_status_at_migration
               WHEN 'rejected' THEN 1
               WHEN 'candidate' THEN 2
               ELSE
                 CASE p.prior_brief_kind
                   WHEN 'admission_only' THEN 3
                   WHEN 'seed_paragraph' THEN 4
                   WHEN 'desktop_chat' THEN 5
                   ELSE 6
                 END
             END AS tier
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.relevance_note AS relevance_note,
               p.prior_admission_note AS prior_admission_note,
               p.prior_brief_kind AS prior_brief_kind,
               p.prior_status_at_migration AS prior_status_at_migration,
               p.linkforge_url AS linkforge_url,
               p.suggested_at AS suggested_at,
               tags, tier
        ORDER BY tier, p.suggested_at DESC, p.arxiv_id
        """
    )


# Backward-compat alias — `cmd_triage` historically called this.
candidate_papers = pending_triage_papers


def promote_paper(arxiv_id: str, relevance_note: str | None = None) -> bool:
    """Move a pending_triage (or legacy 'candidate') paper to status='graphed'."""
    rows = _run(
        """
        MATCH (p:Paper {arxiv_id: $a})
        WHERE p.status IN ['pending_triage', 'candidate']
        SET p.status = 'graphed',
            p.relevance_note = coalesce($rel, p.relevance_note),
            p.triaged_at = datetime()
        RETURN p.arxiv_id AS arxiv_id
        """,
        a=arxiv_id,
        rel=relevance_note,
    )
    return bool(rows)


def reject_paper(arxiv_id: str, reason: str | None = None) -> bool:
    """Tombstone a pending_triage (or legacy 'candidate') paper as status='rejected'."""
    rows = _run(
        """
        MATCH (p:Paper {arxiv_id: $a})
        WHERE p.status IN ['pending_triage', 'candidate']
        SET p.status = 'rejected',
            p.rejection_reason = $reason,
            p.triaged_at = datetime()
        RETURN p.arxiv_id AS arxiv_id
        """,
        a=arxiv_id,
        reason=reason,
    )
    return bool(rows)


def resolve_paper(arxiv_id: str) -> dict[str, Any] | None:
    rows = _run(
        """
        MATCH (p:Paper {arxiv_id: $a})
        OPTIONAL MATCH (p)<-[r]-(f:Finding)
        WITH p, collect({finding: f.id, relation: type(r), props: properties(r)}) AS edges
        RETURN properties(p) AS paper, edges
        """,
        a=arxiv_id,
    )
    if not rows:
        return None
    out = rows[0]["paper"]
    out["incoming_edges"] = [e for e in rows[0]["edges"] if e["finding"]]
    return out


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

LUCENE_RESERVED = r'+-&|!(){}[]^"~*?:\\/'


def _escape_lucene(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if ch in LUCENE_RESERVED:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def _print_paper_edges(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("  (none)")
        return
    for r in rows:
        title = r.get("title") or "(untitled)"
        year = r.get("year")
        print(f"  - {r['arxiv_id']} ({year}) {title}")
        if r.get("repo_url"):
            print(f"      repo: {r['repo_url']}")
        edge = r.get("edge") or {}
        for k, v in edge.items():
            wrapped = textwrap.fill(
                f"{k}: {v}", width=88, subsequent_indent=" " * 8, initial_indent="      "
            )
            print(wrapped)


def cmd_corroborators(args) -> None:
    rows = corroborators(args.finding_id)
    print(f"\nCorroborators for {args.finding_id} ({len(rows)}):")
    _print_paper_edges(rows)


def cmd_contradictors(args) -> None:
    rows = contradictors(args.finding_id)
    print(f"\nContradictors for {args.finding_id} ({len(rows)}):")
    _print_paper_edges(rows)


def cmd_extensions(args) -> None:
    rows = extensions(args.finding_id)
    print(f"\nExperiment ideas extending {args.finding_id} ({len(rows)}):")
    _print_paper_edges(rows)


def cmd_method_differs(args) -> None:
    rows = method_differs(args.finding_id)
    print(f"\nMethod differs from {args.finding_id} ({len(rows)}):")
    _print_paper_edges(rows)


def cmd_explains(args) -> None:
    rows = explains(args.finding_id)
    print(f"\nMechanisms explaining {args.finding_id} ({len(rows)}):")
    _print_paper_edges(rows)


def cmd_related(args) -> None:
    rows = related_findings(args.finding_id)
    print(f"\nRelated findings for {args.finding_id} ({len(rows)}):")
    if not rows:
        print("  (none)")
        return
    for r in rows:
        arrow = "->" if r["direction"] == "outgoing" else "<-"
        print(f"  {arrow} {r['relation']:18} {r['other_id']} ({r['other_status']})")
        print(textwrap.fill(f"     {r['other_claim']}", width=92, subsequent_indent="     "))
        edge = r.get("edge") or {}
        if edge:
            for k, v in edge.items():
                print(textwrap.fill(f"     [{k}: {v}]", width=92, subsequent_indent="     "))


def cmd_novelty(args) -> None:
    rows = novelty_check(args.claim, limit=args.limit)
    print(f"\nNovelty check for: {args.claim!r}")
    if not rows:
        print("  (no fulltext matches — likely novel, but verify against external lit too)")
        return
    rows.sort(key=lambda r: r["score"], reverse=True)
    for r in rows[: args.limit]:
        kind = r["kind"]
        print(f"\n  [{kind} {r['id']}] score={r['score']:.2f}")
        print(textwrap.fill(f"    {r['text']}", width=92, subsequent_indent="    "))


def cmd_timeline(_args) -> None:
    rows = pathway_timeline()
    print("\nPathway timeline:")
    for r in rows:
        print(f"\n  {r['id']:4} [{r['status']:8}] {r['name']}  (experiments: {r['exp_count']})")
        print(textwrap.fill(f"    {r['summary']}", width=92, subsequent_indent="    "))


def cmd_status_report(_args) -> None:
    grouped = finding_status_report()
    print("\nFinding status report:")
    for status in ("ACTIVE", "INVALIDATED", "SUPERSEDED"):
        items = grouped.get(status, [])
        print(f"\n  === {status} ({len(items)}) ===")
        for it in items:
            pending = ", ".join(it["pending"]) if it["pending"] else "—"
            deps = ", ".join(it["depends_on"]) if it["depends_on"] else "—"
            print(f"    {it['id']} [{it['strength']}]")
            print(textwrap.fill(f"      claim: {it['claim']}", width=92, subsequent_indent="             "))
            print(f"      depends_on: {deps}")
            print(f"      pending controls: {pending}")


def cmd_subgraph(args) -> None:
    payload = subgraph_for_finding(args.finding_id, depth=args.depth)
    print(json.dumps(payload, indent=2, default=str))


def cmd_paper(args) -> None:
    paper = resolve_paper(args.arxiv_id)
    if not paper:
        print(f"No paper {args.arxiv_id} in research graph.")
        sys.exit(1)
    print(json.dumps(paper, indent=2, default=str))


# ---- Future experiments ----

def _print_future_experiment(fe: dict[str, Any], indent: str = "  ") -> None:
    cost = fe.get("estimated_cost") or "—"
    blocked = fe.get("blocked_by")
    blocked_str = f"  blocked: {blocked}" if blocked else ""
    print(f"\n{indent}{fe['id']}  [ROI={fe['roi_score']}, {fe['status']}, {fe['priority']}]  {cost}{blocked_str}")
    desc = fe.get("description") or ""
    print(textwrap.fill(desc, width=92, initial_indent=indent + "  ", subsequent_indent=indent + "  "))
    if fe.get("triggered_by"):
        print(f"{indent}  triggered by: {', '.join(fe['triggered_by'])}")
    if fe.get("depends_on"):
        print(f"{indent}  depends on:   {', '.join(fe['depends_on'])}")
    if fe.get("would_update"):
        print(f"{indent}  would update: {', '.join(fe['would_update'])}")


def cmd_future(args) -> None:
    rows = future_experiments(
        pathway_id=args.pathway_id,
        status=args.status,
        min_roi=args.min_roi,
    )
    label = args.pathway_id or "ALL"
    print(f"\nFuture experiments ({label}, {len(rows)}):")
    for r in rows:
        _print_future_experiment(r)


def cmd_triggered(_args) -> None:
    rows = triggered_experiments()
    print(f"\nTRIGGERED future experiments ({len(rows)}) — papers make these actionable now:")
    for r in rows:
        _print_future_experiment(r)


def cmd_blocked(_args) -> None:
    rows = blocked_experiments()
    print(f"\nBLOCKED future experiments ({len(rows)}):")
    for r in rows:
        _print_future_experiment(r)


def cmd_highest_roi(args) -> None:
    rows = highest_roi(n=args.n)
    print(f"\nTop {args.n} future experiments by ROI:")
    for r in rows:
        _print_future_experiment(r)


def cmd_impact(args) -> None:
    payload = experiment_impact(args.fe_id)
    print(json.dumps(payload, indent=2, default=str))


def cmd_pending(args) -> None:
    rows = pending_triage_papers()
    if getattr(args, "ids_only", False):
        # Loop runner mode — just emit arxiv IDs in tier-priority order.
        for r in rows:
            print(r["arxiv_id"])
        return
    print(f"\nPending-triage papers ({len(rows)}) — admitted, awaiting deep `/paper-triage`:")
    if not rows:
        print("  (none)")
        return
    tier_labels = {
        1: "T1 admitted-then-rejected",
        2: "T2 prior candidate",
        3: "T3 graphed-no-brief",
        4: "T4 seed paragraph",
        5: "T5 Desktop backfill",
        6: "T6 fresh admission",
    }
    current_tier: int | None = None
    for r in rows:
        tier = r.get("tier")
        if tier != current_tier:
            current_tier = tier
            label = tier_labels.get(tier or 6, f"T{tier}")
            print(f"\n  --- {label} ---")
        title = r.get("title") or "(untitled)"
        suggested_at = r.get("suggested_at") or "—"
        print(f"\n  {r['arxiv_id']}  suggested: {suggested_at}")
        print(textwrap.fill(f"    title: {title}", width=92, subsequent_indent="           "))
        note = r.get("relevance_note") or r.get("prior_admission_note") or "(none)"
        print(textwrap.fill(f"    note:  {note}", width=92, subsequent_indent="           "))
        url = r.get("linkforge_url")
        if url:
            print(f"    link:  {url}")
        tags = r.get("tags") or []
        if tags:
            print(f"    tags:  {', '.join(tags)}")
    print("\nDeep pass: claude -p '/paper-triage <arxiv-id>'")
    print("Promote brief: python promote_brief.py <brief-path>")
    print("Reject:        python query.py reject <arxiv-id> [--reason '...']")


# Backward-compat alias — older docs reference cmd_triage / `triage` subcommand.
cmd_triage = cmd_pending


def cmd_promote(args) -> None:
    ok = promote_paper(args.arxiv_id, relevance_note=args.note)
    if ok:
        print(f"Promoted {args.arxiv_id} -> status='graphed'.")
    else:
        print(f"No candidate {args.arxiv_id} found (already triaged or doesn't exist).")
        sys.exit(1)


def cmd_reject(args) -> None:
    ok = reject_paper(args.arxiv_id, reason=args.reason)
    if ok:
        print(f"Rejected {args.arxiv_id} -> status='rejected' (kept as tombstone).")
    else:
        print(f"No candidate {args.arxiv_id} found (already triaged or doesn't exist).")
        sys.exit(1)


def cmd_watchlist(_args) -> None:
    rows = watchlist()
    print(f"\nWatchlist — papers triggering future experiments ({len(rows)}):")
    for r in rows:
        triggers = r.get("triggers") or []
        trigger_strs = [f"{t['fe_id']}({t['status']})" for t in triggers]
        title = r.get("title") or "(untitled)"
        year = r.get("year") or "—"
        print(f"  - {r['arxiv_id']} ({year}) {title}")
        print(f"      triggers: {', '.join(trigger_strs)}")
        if r.get("repo_url"):
            print(f"      repo: {r['repo_url']}")


# ---------------------------------------------------------------------------
# RAG: Semantic retrieval (Phase 2)
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _embed_query(query: str) -> list[float]:
    model = _get_embedding_model()
    return model.encode(query, normalize_embeddings=True).tolist()


def _path_paper_vec(query_vec: list[float], limit: int = 20) -> list[tuple[str, str, int]]:
    rows = _run(
        """
        CALL db.index.vector.queryNodes('paper_embedding_idx', $limit, $vec)
        YIELD node, score
        RETURN 'Paper' AS label, node.arxiv_id AS id, score
        ORDER BY score DESC, node.arxiv_id ASC
        """,
        vec=query_vec, limit=limit,
    )
    return [("Paper", r["id"], i) for i, r in enumerate(rows)]


def _path_finding_vec(query_vec: list[float], limit: int = 10) -> list[tuple[str, str, int]]:
    rows = _run(
        """
        CALL db.index.vector.queryNodes('finding_embedding_idx', $limit, $vec)
        YIELD node, score
        RETURN 'Finding' AS label, node.id AS id, score
        ORDER BY score DESC, node.id ASC
        """,
        vec=query_vec, limit=limit,
    )
    return [("Finding", r["id"], i) for i, r in enumerate(rows)]


def _path_finding_ft(query: str, limit: int = 10) -> list[tuple[str, str, int]]:
    safe = _escape_lucene(query)
    rows = _run(
        """
        CALL db.index.fulltext.queryNodes('finding_claims', $q)
        YIELD node, score
        RETURN 'Finding' AS label, node.id AS id, score
        ORDER BY score DESC, node.id ASC
        LIMIT $limit
        """,
        q=safe, limit=limit,
    )
    return [("Finding", r["id"], i) for i, r in enumerate(rows)]


def _path_paper_ft(query: str, limit: int = 20) -> list[tuple[str, str, int]]:
    safe = _escape_lucene(query)
    rows = _run(
        """
        CALL db.index.fulltext.queryNodes('paper_relevance', $q)
        YIELD node, score
        RETURN 'Paper' AS label, node.arxiv_id AS id, score
        ORDER BY score DESC, node.arxiv_id ASC
        LIMIT $limit
        """,
        q=safe, limit=limit,
    )
    return [("Paper", r["id"], i) for i, r in enumerate(rows)]


def _path_tag_match(query: str, limit: int = 15) -> list[tuple[str, str, int]]:
    terms = [t.lower().replace(" ", "_") for t in query.split() if len(t) > 2]
    if not terms:
        return []
    results: list[tuple[str, str, float]] = []
    for term in terms:
        rows = _run(
            """
            MATCH (t:Tag) WHERE toLower(t.name) CONTAINS $term
            MATCH (n)-[:TAGGED]->(t)
            WHERE (n:Paper OR n:Finding)
            WITH CASE
                WHEN n:Paper THEN 'Paper'
                WHEN n:Finding THEN 'Finding'
            END AS label,
            CASE
                WHEN n:Paper THEN n.arxiv_id
                WHEN n:Finding THEN n.id
            END AS id,
            1.0 AS score
            RETURN DISTINCT label, id, score
            """,
            term=term,
        )
        for r in rows:
            results.append((r["label"], r["id"], r["score"]))

    seen: dict[str, int] = {}
    for label, nid, _ in results:
        key = f"{label}||{nid}"
        seen[key] = seen.get(key, 0) + 1

    ranked = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
    out = []
    for key, _ in ranked[:limit]:
        label, nid = key.split("||", 1)
        out.append((label, nid, len(out)))
    return out


def _path_dataset_match(query: str, limit: int = 25) -> list[tuple[str, str, int]]:
    terms = [t for t in query.split() if len(t) > 2]
    if not terms:
        return []
    results = []
    for term in terms:
        rows = _run(
            """
            MATCH (d:Dataset)-[:USED_IN]->(p:Paper)
            WHERE toLower(coalesce(d.display_name, d.name)) CONTAINS toLower($term)
            RETURN DISTINCT 'Paper' AS label, p.arxiv_id AS id
            ORDER BY p.arxiv_id
            """,
            term=term,
        )
        for r in rows:
            results.append((r["label"], r["id"]))

    seen: dict[str, int] = {}
    for label, nid in results:
        key = f"{label}||{nid}"
        seen[key] = seen.get(key, 0) + 1

    ranked = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
    out = []
    for key, _ in ranked[:limit]:
        label, nid = key.split("||", 1)
        out.append((label, nid, len(out)))
    return out


_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


def _extract_arxiv_ids(query: str) -> list[str]:
    """Extract arxiv ID patterns from a query string."""
    return list(dict.fromkeys(_ARXIV_ID_RE.findall(query)))


def _path_arxiv_id_match(arxiv_ids: list[str]) -> list[tuple[str, str, int]]:
    """Exact-match retrieval for queries containing arxiv IDs."""
    if not arxiv_ids:
        return []
    rows = _run(
        """
        UNWIND $ids AS aid
        MATCH (p:Paper {arxiv_id: aid})
        RETURN 'Paper' AS label, p.arxiv_id AS id
        """,
        ids=arxiv_ids,
    )
    return [("Paper", r["id"], i) for i, r in enumerate(rows)]


def _path_finding_neighborhood(
    finding_results: list[tuple[str, str, int]], limit: int = 25,
) -> list[tuple[str, str, int]]:
    ranked = [(nid, rank) for label, nid, rank in finding_results if label == "Finding"]
    if not ranked:
        return []
    fid_rank = {}
    for nid, rank in ranked:
        if nid not in fid_rank:
            fid_rank[nid] = rank
    rows = _run(
        """
        UNWIND $fids AS fid
        MATCH (f:Finding {id: fid})-[r]->(p:Paper)
        WHERE type(r) IN ['CORROBORATED_BY','CONTRADICTED_BY','EXTENDED_BY','METHOD_DIFFERS','EXPLAINS']
        RETURN DISTINCT fid AS finding, 'Paper' AS label, p.arxiv_id AS id
        """,
        fids=list(fid_rank.keys()),
    )
    scored = []
    seen = set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        scored.append((fid_rank.get(r["finding"], 99), r["id"]))
    scored.sort()
    out = []
    for _, pid in scored[:limit]:
        out.append(("Paper", pid, len(out)))
    return out


DEFAULT_WEIGHTS = {
    "paper-vec": 2.0,
    "finding-vec": 2.0,
    "finding-ft": 1.5,
    "paper-ft": 2.5,
    "tag-match": 1.5,
    "dataset-match": 2.0,
    "finding-neighborhood": 2.0,
    "arxiv-id-match": 5.0,
}

RESCUE_PATHS = {"tag-match", "dataset-match", "paper-ft", "finding-ft"}

FINDING_RESERVE = 4


CORE_PATHS = set(DEFAULT_WEIGHTS.keys())


def rrf_merge(
    path_results: dict[str, list[tuple[str, str, int]]],
    weights: dict[str, float] | None = None,
    rescue_paths: set[str] | None = None,
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if rescue_paths is None:
        rescue_paths = RESCUE_PATHS

    scores: dict[str, float] = {}
    appearances: dict[str, set[str]] = {}

    for path_id, results in sorted(path_results.items()):
        w = weights.get(path_id, 1.0)
        for label, node_id, rank in results:
            key = f"{label}||{node_id}"
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            appearances.setdefault(key, set()).add(path_id)

    RESCUE_BOOST = 0.04
    RESCUE_MAX_RANK = 3
    for path_id, results in sorted(path_results.items()):
        if path_id not in rescue_paths:
            continue
        for label, node_id, rank in results:
            key = f"{label}||{node_id}"
            core_count = len(appearances.get(key, set()) & CORE_PATHS)
            if core_count <= 1 and rank <= RESCUE_MAX_RANK:
                scores[key] += RESCUE_BOOST

    ranked = sorted(scores.items(), key=lambda x: (-round(x[1], 10), x[0]))

    findings = []
    papers = []
    for key, score in ranked:
        label, node_id = key.split("||", 1)
        entry = {"label": label, "id": node_id, "score": round(score, 10)}
        if label == "Finding" and len(findings) < FINDING_RESERVE:
            findings.append(entry)
        elif label != "Finding" or len(findings) >= FINDING_RESERVE:
            papers.append(entry)

    paper_slots = top_n - min(len(findings), FINDING_RESERVE)
    out = papers[:paper_slots] + findings
    out.sort(key=lambda x: (-x["score"], x["label"], x["id"]))
    return out[:top_n]


# ── LLM enhancements (Phase 3) ──────────────────────────────────────────────

HYDE_PROMPT = """\
Write a short factual paragraph (3-4 sentences) as if you are a research note \
in the topo-confidence project about residual-stream geometry and LLM \
correctness prediction. Reference specific concepts: hidden-state geometry, \
DoM (difference-of-means direction), AUROC, participation ratio, covariance \
spectrum, persistent homology, steering vectors, selective prediction, \
MATH-500 benchmark. Be concrete about methods and results. Do not restate \
the question."""

CONCEPT_EXPAND_PROMPT = """\
Given the question below about the topo-confidence research project (LLM \
correctness prediction via residual-stream geometry), generate 8-10 related \
technical terms that might appear in paper titles, finding claims, or tag \
names. Output one term per line, lowercase, hyphenated. No bullets or numbers."""


def _call_claude(system: str, user: str, timeout: int = 30) -> str | None:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    prompt = f"{system}\n\n{user}"
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _hyde_embed(query: str, no_cache: bool = False) -> list[float] | None:
    """Generate hypothetical document and average its embedding with query."""
    from llm_cache import cache_get, cache_set

    cache_key = query
    if not no_cache:
        cached = cache_get("hyde", cache_key)
        if cached is not None:
            model = _get_embedding_model()
            import numpy as np
            q_vec = model.encode(query, normalize_embeddings=True)
            h_vec = model.encode(cached, normalize_embeddings=True)
            combined = q_vec + h_vec
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            return combined.tolist()

    text = _call_claude(HYDE_PROMPT, query)
    if not text:
        return None

    if not no_cache:
        cache_set("hyde", cache_key, text)

    model = _get_embedding_model()
    import numpy as np
    q_vec = model.encode(query, normalize_embeddings=True)
    h_vec = model.encode(text, normalize_embeddings=True)
    combined = q_vec + h_vec
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined.tolist()


def _concept_expand(query: str, no_cache: bool = False) -> list[str]:
    """Generate related technical terms for broader tag/fulltext search."""
    from llm_cache import cache_get, cache_set

    cache_key = query
    if not no_cache:
        cached = cache_get("concept-expand", cache_key)
        if cached is not None:
            return [t.strip() for t in cached.split("\n") if t.strip()]

    text = _call_claude(CONCEPT_EXPAND_PROMPT, query)
    if not text:
        return []

    if not no_cache:
        cache_set("concept-expand", cache_key, text)

    return [t.strip() for t in text.split("\n") if t.strip()]


def _fetch_rerank_context(results: list[dict]) -> dict[str, str]:
    """Fetch titles/claims for reranker context."""
    paper_ids = [r["id"] for r in results if r["label"] == "Paper"]
    finding_ids = [r["id"] for r in results if r["label"] == "Finding"]
    context: dict[str, str] = {}
    if paper_ids:
        rows = _run(
            "UNWIND $ids AS aid "
            "MATCH (p:Paper {arxiv_id: aid}) "
            "RETURN p.arxiv_id AS id, p.title AS title, p.relevance_note AS note",
            ids=paper_ids,
        )
        for r in rows:
            parts = [r["title"] or ""]
            if r["note"]:
                parts.append(r["note"])
            context[f"Paper||{r['id']}"] = " — ".join(p for p in parts if p)
    if finding_ids:
        rows = _run(
            "UNWIND $ids AS fid "
            "MATCH (f:Finding {id: fid}) "
            "RETURN f.id AS id, f.claim AS claim",
            ids=finding_ids,
        )
        for r in rows:
            context[f"Finding||{r['id']}"] = r["claim"] or ""
    return context


def _rerank(query: str, results: list[dict], no_cache: bool = False,
            final_n: int | None = None) -> list[dict]:
    """Rerank top results using Claude, optionally trimming to final_n."""
    from llm_cache import cache_get, cache_set

    if len(results) <= 1:
        return results[:final_n] if final_n else results

    sorted_keys = sorted(f"{r['label']}||{r['id']}" for r in results)
    cache_key = f"{query}|||{','.join(sorted_keys)}"

    if not no_cache:
        cached = cache_get("rerank", cache_key)
        if cached is not None:
            import json as _json
            try:
                order = _json.loads(cached)
                id_map = {f"{r['label']}||{r['id']}": r for r in results}
                reranked = [id_map[k] for k in order if k in id_map]
                remaining = [r for r in results if f"{r['label']}||{r['id']}" not in set(order)]
                out = reranked + remaining
                return out[:final_n] if final_n else out
            except (ValueError, KeyError):
                pass

    ctx = _fetch_rerank_context(results)
    snippets = []
    for i, r in enumerate(results):
        key = f"{r['label']}||{r['id']}"
        desc = ctx.get(key, "")
        line = f"{i+1}. [{r['label']}] {r['id']}"
        if desc:
            line += f": {desc}"
        snippets.append(line)

    system = (
        "You are reranking search results for the topo-confidence research project "
        "about residual-stream geometry and LLM correctness prediction. "
        "Given the query and numbered results with descriptions, return ONLY the "
        "numbers in order of relevance, one per line. Most relevant first. No explanation."
    )
    user = f"Query: {query}\n\nResults:\n" + "\n".join(snippets)
    text = _call_claude(system, user)
    if not text:
        return results

    try:
        indices = []
        for line in text.strip().split("\n"):
            line = line.strip().rstrip(".")
            if line.isdigit():
                idx = int(line) - 1
                if 0 <= idx < len(results) and idx not in indices:
                    indices.append(idx)
        if not indices:
            return results
        reranked = [results[i] for i in indices]
        remaining = [results[i] for i in range(len(results)) if i not in set(indices)]
        reranked.extend(remaining)

        if not no_cache:
            order = [f"{r['label']}||{r['id']}" for r in reranked]
            import json as _json
            cache_set("rerank", cache_key, _json.dumps(order))

        return reranked[:final_n] if final_n else reranked
    except (ValueError, IndexError):
        return results[:final_n] if final_n else results


RERANK_OVER_RETRIEVE = 3

DECOMPOSE_PROMPT = """\
You are analyzing a search query for the topo-confidence research project \
(LLM correctness prediction via residual-stream geometry). Determine if the \
query compares, contrasts, or asks about multiple specific methods, papers, \
or techniques. If so, decompose it into focused sub-queries, one per entity.

Rules:
- If the query mentions 2+ specific named methods/techniques/papers, output \
one focused sub-query per entity (2-3 queries max)
- Each sub-query should be a concise retrieval query for that single entity
- If the query is about a single topic, output just: SINGLE
- Output only the sub-queries, one per line, no numbering or bullets"""


def _decompose_query(query: str, no_cache: bool = False) -> list[str] | None:
    """Decompose a multi-hop query into focused sub-queries."""
    from llm_cache import cache_get, cache_set

    cache_key = query
    if not no_cache:
        cached = cache_get("decompose", cache_key)
        if cached is not None:
            if cached.strip().upper() == "SINGLE":
                return None
            return [t.strip() for t in cached.split("\n") if t.strip()]

    text = _call_claude(DECOMPOSE_PROMPT, query)
    if not text:
        return None

    if not no_cache:
        cache_set("decompose", cache_key, text)

    if text.strip().upper() == "SINGLE":
        return None
    subs = [t.strip() for t in text.split("\n") if t.strip()]
    return subs if len(subs) >= 2 else None


def semantic_search(
    query: str,
    weights: dict[str, float] | None = None,
    top_n: int = 10,
    hyde: bool = False,
    concept_expand: bool = False,
    rerank: bool = False,
    no_cache: bool = False,
    decompose: bool = False,
) -> list[dict]:
    """Run retrieval paths and RRF-merge results."""
    sub_queries = _decompose_query(query, no_cache=no_cache) if decompose else None

    query_vec = _embed_query(query)

    path_results: dict[str, list[tuple[str, str, int]]] = {}

    arxiv_ids = _extract_arxiv_ids(query)
    if arxiv_ids:
        path_results["arxiv-id-match"] = _path_arxiv_id_match(arxiv_ids)

    path_results["paper-vec"] = _path_paper_vec(query_vec)
    path_results["finding-vec"] = _path_finding_vec(query_vec)

    if hyde:
        hyde_vec = _hyde_embed(query, no_cache=no_cache)
        if hyde_vec:
            path_results["paper-vec-hyde"] = _path_paper_vec(hyde_vec)
            path_results["finding-vec-hyde"] = _path_finding_vec(hyde_vec)
            if weights is None:
                weights = dict(DEFAULT_WEIGHTS)
            weights.setdefault("paper-vec-hyde", 1.0)
            weights.setdefault("finding-vec-hyde", 1.0)
    path_results["finding-ft"] = _path_finding_ft(query)
    path_results["paper-ft"] = _path_paper_ft(query)
    path_results["tag-match"] = _path_tag_match(query)
    path_results["dataset-match"] = _path_dataset_match(query)

    if concept_expand:
        terms = _concept_expand(query, no_cache=no_cache)
        if terms:
            expanded_query = query + " " + " ".join(terms)
            expand_tag = _path_tag_match(expanded_query, limit=8)
            expand_ft_paper = _path_paper_ft(expanded_query, limit=10)
            expand_ft_finding = _path_finding_ft(expanded_query, limit=5)
            path_results["expand-tag"] = expand_tag
            path_results["expand-ft-paper"] = expand_ft_paper
            path_results["expand-ft-finding"] = expand_ft_finding
            if weights is None:
                weights = dict(DEFAULT_WEIGHTS)
            weights.setdefault("expand-tag", 0.25)
            weights.setdefault("expand-ft-paper", 0.25)
            weights.setdefault("expand-ft-finding", 0.25)

    finding_seed = path_results["finding-vec"] + path_results["finding-ft"]
    path_results["finding-neighborhood"] = _path_finding_neighborhood(finding_seed)

    merge_n = top_n + RERANK_OVER_RETRIEVE if rerank else top_n
    results = rrf_merge(path_results, weights=weights, top_n=merge_n)

    if rerank:
        pre_rerank_findings = [r for r in results if r["label"] == "Finding"]
        results = _rerank(query, results, no_cache=no_cache, final_n=top_n)
        post_findings = {r["id"] for r in results if r["label"] == "Finding"}
        dropped = [f for f in pre_rerank_findings[:FINDING_RESERVE]
                   if f["id"] not in post_findings]
        if dropped:
            papers_in = [r for r in results if r["label"] != "Finding"]
            findings_in = [r for r in results if r["label"] == "Finding"]
            findings_in.extend(dropped)
            results = (papers_in[:top_n - len(findings_in)] + findings_in)
            results.sort(key=lambda x: (-x["score"], x["label"], x["id"]))
            results = results[:top_n]

    if sub_queries:
        results = _augment_with_decomposed(
            query, sub_queries, results, weights=weights, top_n=top_n,
            hyde=hyde, concept_expand=concept_expand,
            rerank=rerank, no_cache=no_cache,
        )

    return results


def _augment_with_decomposed(
    original_query: str,
    sub_queries: list[str],
    monolithic: list[dict],
    weights: dict[str, float] | None = None,
    top_n: int = 10,
    hyde: bool = False,
    concept_expand: bool = False,
    rerank: bool = False,
    no_cache: bool = False,
) -> list[dict]:
    """Augment monolithic results with sub-query hits (union strategy)."""
    per_sub = max(top_n // len(sub_queries) + 2, 5)
    existing = {f"{r['label']}||{r['id']}" for r in monolithic}

    novel: list[dict] = []
    for sq in sub_queries[:3]:
        r = semantic_search(
            sq, weights=weights, top_n=per_sub,
            hyde=hyde, concept_expand=concept_expand,
            rerank=rerank, no_cache=no_cache,
            decompose=False,
        )
        for item in r:
            key = f"{item['label']}||{item['id']}"
            if key not in existing:
                existing.add(key)
                novel.append(item)

    if not novel:
        return monolithic

    combined = monolithic + novel
    if rerank and len(combined) > top_n:
        combined = _rerank(original_query, combined[:top_n + RERANK_OVER_RETRIEVE],
                           no_cache=no_cache, final_n=top_n)
    return combined[:top_n]


# ── Context assembly ────────────────────────────────────────────────────────

def _fetch_paper_context(arxiv_id: str) -> dict[str, Any]:
    rows = _run(
        """
        MATCH (p:Paper {arxiv_id: $a})
        OPTIONAL MATCH (p)<-[r]-(f:Finding)
        WITH p, collect({finding: f.id, rel: type(r)}) AS edges
        OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
        WITH p, edges, collect(t.name) AS tags
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
               p.relevance_note AS relevance_note, p.forge_score AS forge_score,
               edges, tags
        """,
        a=arxiv_id,
    )
    return dict(rows[0]) if rows else {}


def _fetch_finding_context(finding_id: str) -> dict[str, Any]:
    rows = _run(
        """
        MATCH (f:Finding {id: $id})
        OPTIONAL MATCH (f)-[r]->(p:Paper)
        WHERE type(r) IN ['CORROBORATED_BY','CONTRADICTED_BY','EXTENDED_BY','METHOD_DIFFERS','EXPLAINS']
        WITH f, collect({arxiv_id: p.arxiv_id, rel: type(r)}) AS paper_edges
        RETURN f.id AS id, f.claim AS claim, f.status AS status,
               f.strength AS strength,
               f.strongest_counterargument AS counterargument,
               paper_edges
        """,
        id=finding_id,
    )
    return dict(rows[0]) if rows else {}


def format_context(results: list[dict]) -> str:
    """Build context string for LLM synthesis."""
    parts = []
    for i, r in enumerate(results):
        rank = i + 1
        if r["label"] == "Paper":
            ctx = _fetch_paper_context(r["id"])
            if not ctx:
                continue
            title = ctx.get("title") or "(untitled)"
            year = ctx.get("year") or "?"
            note = ctx.get("relevance_note") or ""
            tags = ", ".join(ctx.get("tags") or [])
            fs = ctx.get("forge_score") or "?"
            edges = ctx.get("edges") or []
            edge_strs = [f"{e['rel']} ← {e['finding']}" for e in edges if e.get("finding")]
            edge_line = " | ".join(edge_strs) if edge_strs else "none"
            parts.append(
                f"[{rank}] Paper: \"{title}\" ({ctx['arxiv_id']}, {year})\n"
                f"    Relevance: {note}\n"
                f"    Tags: {tags}\n"
                f"    ForgeScore: {fs}\n"
                f"    Edges: {edge_line}"
            )
        elif r["label"] == "Finding":
            ctx = _fetch_finding_context(r["id"])
            if not ctx:
                continue
            status = ctx.get("status") or "?"
            strength = ctx.get("strength") or "?"
            claim = ctx.get("claim") or ""
            counter = ctx.get("counterargument") or ""
            edges = ctx.get("paper_edges") or []
            edge_strs = [f"{e['rel']} → {e['arxiv_id']}" for e in edges if e.get("arxiv_id")]
            edge_line = ", ".join(edge_strs) if edge_strs else "none"
            parts.append(
                f"[{rank}] Finding {ctx['id']} [{status}/{strength}]:\n"
                f"    Claim: {claim}\n"
                f"    Strongest counterargument: {counter}\n"
                f"    Paper edges: {edge_line}"
            )
    return "\n\n".join(parts)


SYSTEM_PROMPT = """\
You are answering questions about the topo-confidence research project, which \
investigates whether residual-stream geometry predicts LLM correctness.

The context below contains Findings (F-N: established results with evidence \
and counterarguments) and Papers (arxiv papers linked to findings via typed \
edges like CORROBORATED_BY, CONTRADICTED_BY, EXTENDED_BY).

When answering:
- Cite specific finding IDs (F-2, F-7, etc.) and arxiv IDs
- Distinguish between ACTIVE, INVALIDATED, and SUPERSEDED findings
- Note the strength level (STRONG, MODERATE, PRELIMINARY)
- If a finding has a strongest_counterargument, mention it
- Reference specific AUROC values, coverage numbers, and benchmarks"""


def llm_synthesize(query: str, context: str) -> str:
    """Send context + query to Claude for synthesis."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    prompt = f"{SYSTEM_PROMPT}\n\n---\nContext:\n{context}\n---\nQuestion: {query}"
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=60, env=env,
    )
    if result.returncode != 0:
        return f"LLM synthesis failed: {result.stderr[:200]}"
    return result.stdout.strip()


def cmd_semantic(args) -> None:
    results = semantic_search(
        args.query, top_n=args.top_n,
        hyde=args.hyde, concept_expand=args.expand,
        rerank=args.rerank, no_cache=args.no_cache,
    )
    if args.json:
        print(json.dumps(results, indent=2))
        return
    print(f"\nSemantic search: {args.query!r}\n")
    for r in results:
        print(f"  [{r['label']:8}] {r['id']:<16} score={r['score']:.6f}")
    print(f"\n  {len(results)} results")


def cmd_ask(args) -> None:
    results = semantic_search(
        args.query, top_n=args.top_n,
        hyde=args.hyde, concept_expand=args.expand,
        rerank=args.rerank, no_cache=args.no_cache,
    )
    context = format_context(results)
    if args.skip_llm:
        print(context)
        return
    if args.json:
        answer = llm_synthesize(args.query, context)
        print(json.dumps({"query": args.query, "answer": answer, "sources": results}, indent=2))
        return
    print(f"\nQuestion: {args.query}\n")
    print("Sources:")
    for r in results:
        print(f"  [{r['label']:8}] {r['id']}")
    answer = llm_synthesize(args.query, context)
    print(f"\n{answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="topo-confidence research graph CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in [
        ("corroborators", cmd_corroborators),
        ("contradictors", cmd_contradictors),
        ("extensions", cmd_extensions),
        ("method-differs", cmd_method_differs),
        ("explains", cmd_explains),
        ("related", cmd_related),
    ]:
        s = sub.add_parser(name)
        s.add_argument("finding_id")
        s.set_defaults(func=fn)

    s = sub.add_parser("novelty")
    s.add_argument("claim")
    s.add_argument("--limit", type=int, default=8)
    s.set_defaults(func=cmd_novelty)

    s = sub.add_parser("timeline")
    s.set_defaults(func=cmd_timeline)

    s = sub.add_parser("status-report")
    s.set_defaults(func=cmd_status_report)

    s = sub.add_parser("subgraph")
    s.add_argument("finding_id")
    s.add_argument("--depth", type=int, default=2)
    s.set_defaults(func=cmd_subgraph)

    s = sub.add_parser("paper")
    s.add_argument("arxiv_id")
    s.set_defaults(func=cmd_paper)

    s = sub.add_parser("future", help="List future experiments (optionally filter by pathway/status/roi)")
    s.add_argument("pathway_id", nargs="?", default=None,
                   help="e.g. P7  — omit for all pathways")
    s.add_argument("--status", default=None,
                   choices=["READY", "TRIGGERED", "BLOCKED", "COMPLETED", "ABANDONED"])
    s.add_argument("--min-roi", type=int, default=None)
    s.set_defaults(func=cmd_future)

    s = sub.add_parser("triggered", help="All TRIGGERED future experiments")
    s.set_defaults(func=cmd_triggered)

    s = sub.add_parser("blocked", help="All BLOCKED future experiments (or those with a blocked_by reason)")
    s.set_defaults(func=cmd_blocked)

    s = sub.add_parser("highest-roi", help="Top N future experiments by ROI")
    s.add_argument("n", nargs="?", type=int, default=10)
    s.set_defaults(func=cmd_highest_roi)

    s = sub.add_parser("impact", help="Full impact analysis for a single future experiment")
    s.add_argument("fe_id")
    s.set_defaults(func=cmd_impact)

    s = sub.add_parser("watchlist", help="Papers triggering future experiments — monitor for follow-ups")
    s.set_defaults(func=cmd_watchlist)

    s = sub.add_parser(
        "pending",
        aliases=["triage"],
        help="Papers awaiting deep `/paper-triage` (status='pending_triage'), tier-ordered",
    )
    s.add_argument(
        "--ids-only",
        action="store_true",
        dest="ids_only",
        help="Emit one arxiv ID per line, tier-priority order. Used by triage_pending.sh.",
    )
    s.set_defaults(func=cmd_pending)

    s = sub.add_parser("promote", help="Promote a candidate to graphed (rejection: see 'reject')")
    s.add_argument("arxiv_id")
    s.add_argument("--note", default=None,
                   help="Override the auto-generated relevance_note while promoting.")
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("reject", help="Tombstone a candidate so it isn't re-suggested")
    s.add_argument("arxiv_id")
    s.add_argument("--reason", default=None,
                   help="Optional rejection reason recorded on the node.")
    s.set_defaults(func=cmd_reject)

    s = sub.add_parser("semantic", help="Semantic search — vector + fulltext + graph, RRF merged")
    s.add_argument("query")
    s.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    s.add_argument("--top-n", type=int, default=10)
    s.add_argument("--hyde", action="store_true", help="Enable HyDE (hypothetical document embedding)")
    s.add_argument("--expand", action="store_true", help="Enable concept expansion")
    s.add_argument("--rerank", action="store_true", help="Enable LLM reranking")
    s.add_argument("--no-cache", action="store_true", help="Skip LLM cache reads and writes")
    s.set_defaults(func=cmd_semantic)

    s = sub.add_parser("ask", help="Semantic search + LLM synthesis")
    s.add_argument("query")
    s.add_argument("--skip-llm", action="store_true", help="Retrieval only, no LLM")
    s.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    s.add_argument("--top-n", type=int, default=10)
    s.add_argument("--hyde", action="store_true", help="Enable HyDE (hypothetical document embedding)")
    s.add_argument("--expand", action="store_true", help="Enable concept expansion")
    s.add_argument("--rerank", action="store_true", help="Enable LLM reranking")
    s.add_argument("--no-cache", action="store_true", help="Skip LLM cache reads and writes")
    s.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
