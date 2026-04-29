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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
