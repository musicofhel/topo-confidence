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
    return GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
