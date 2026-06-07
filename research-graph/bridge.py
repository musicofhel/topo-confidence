"""Bridge from the topo-confidence research graph to link-forge.

link-forge stores academic papers as (:Link) nodes whose URL contains the
arxiv ID (e.g. http://arxiv.org/abs/2410.13640). This module maps an arxiv
ID to that node and enriches the research graph's Paper stubs with the
fields we want to keep here (title and forgeScore — abstracts stay in
link-forge to avoid duplication).

Connections:
    research graph -> bolt://localhost:7688 (read+write)
    link-forge     -> bolt://localhost:7687 (READ ONLY — never written)

If link-forge is unreachable, every function returns gracefully (None or
[]) and prints a one-line warning.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

RG_BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
RG_USER = os.environ.get("NEO4J_USER", "neo4j")
RG_PASS = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

LF_BOLT = os.environ.get("LINK_FORGE_BOLT_URL", "bolt://localhost:7687")
LF_USER = os.environ.get("LINK_FORGE_USER", "neo4j")
LF_PASS = os.environ.get("LINK_FORGE_PASSWORD", "link_forge_dev")


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

@contextmanager
def _link_forge_session() -> Iterator[Any]:
    """Yield a link-forge session, or None if unreachable."""
    try:
        drv = GraphDatabase.driver(LF_BOLT, auth=(LF_USER, LF_PASS))
        with drv.session(default_access_mode="READ") as s:
            # quick handshake
            s.run("RETURN 1").consume()
            yield s
        drv.close()
    except (ServiceUnavailable, AuthError, OSError) as exc:
        print(f"[bridge] link-forge unreachable at {LF_BOLT}: {exc}")
        yield None


@contextmanager
def _research_session() -> Iterator[Any]:
    drv = GraphDatabase.driver(RG_BOLT, auth=(RG_USER, RG_PASS))
    with drv.session() as s:
        yield s
    drv.close()


# ---------------------------------------------------------------------------
# Core queries
# ---------------------------------------------------------------------------

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _arxiv_to_url_substr(arxiv_id: str) -> str:
    """Return a substring guaranteed to appear in any canonical arxiv URL."""
    return f"/{arxiv_id}"


def resolve_from_linkforge(arxiv_id: str) -> dict[str, Any] | None:
    """Look up a paper in link-forge by arxiv ID.

    Returns dict with title/forgeScore/url/quality/contentType/concepts (top 8),
    or None if not found / link-forge unreachable.
    """
    needle = _arxiv_to_url_substr(arxiv_id)
    with _link_forge_session() as session:
        if session is None:
            return None
        rows = list(
            session.run(
                """
                MATCH (l:Link)
                WHERE l.url CONTAINS $needle
                OPTIONAL MATCH (l)-[:RELATES_TO_CONCEPT]->(c:Concept)
                OPTIONAL MATCH (l)-[:TAGGED_WITH]->(t:Tag)
                WITH l,
                     collect(DISTINCT c.name)[0..8] AS concepts,
                     collect(DISTINCT t.name)[0..8] AS tags
                RETURN l.title       AS title,
                       l.url         AS url,
                       l.forgeScore  AS forgeScore,
                       l.quality     AS quality,
                       l.contentType AS contentType,
                       l.purpose     AS purpose,
                       concepts,
                       tags
                LIMIT 1
                """,
                needle=needle,
            )
        )
    if not rows:
        return None
    rec = dict(rows[0])
    rec["arxiv_id"] = arxiv_id
    return rec


def enrich_all_papers(verbose: bool = True) -> dict[str, Any]:
    """For every Paper in the research graph, copy title + forgeScore from link-forge.

    Existing title/repo_url/relevance_note are preserved; only fills in
    missing title and adds forgeScore. Returns a small summary dict.
    """
    with _research_session() as rg:
        papers = list(
            rg.run(
                "MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id, p.title AS title"
            )
        )

    found, missing = [], []
    for r in papers:
        arxiv_id = r["arxiv_id"]
        meta = resolve_from_linkforge(arxiv_id)
        if meta is None:
            missing.append(arxiv_id)
            if verbose:
                print(f"  miss   {arxiv_id}")
            continue
        found.append(arxiv_id)
        with _research_session() as rg:
            rg.run(
                """
                MATCH (p:Paper {arxiv_id: $a})
                SET p.linkforge_title = $title,
                    p.forgeScore = $score,
                    p.linkforge_url = $url
                """,
                a=arxiv_id,
                title=meta.get("title"),
                score=meta.get("forgeScore"),
                url=meta.get("url"),
            )
        if verbose:
            score = meta.get("forgeScore")
            print(f"  hit    {arxiv_id}  forgeScore={score}  {meta.get('title') or ''}")

    return {"enriched": len(found), "missing": len(missing), "missing_ids": missing}


def enrich_descriptions(verbose: bool = True) -> dict[str, Any]:
    """Copy description text from link-forge to Paper nodes in the research graph.

    Only fills in missing descriptions (won't overwrite existing ones).
    """
    with _research_session() as rg:
        papers = list(
            rg.run(
                "MATCH (p:Paper) WHERE p.description IS NULL "
                "RETURN p.arxiv_id AS arxiv_id"
            )
        )

    if verbose:
        print(f"  Papers missing descriptions: {len(papers)}")

    found, missing = [], []
    with _link_forge_session() as lf:
        if lf is None:
            return {"enriched": 0, "missing": len(papers), "missing_ids": [r["arxiv_id"] for r in papers]}
        for r in papers:
            arxiv_id = r["arxiv_id"]
            needle = _arxiv_to_url_substr(arxiv_id)
            rows = list(lf.run(
                "MATCH (l:Link) WHERE l.url CONTAINS $needle "
                "AND l.description IS NOT NULL "
                "RETURN l.description AS description LIMIT 1",
                needle=needle,
            ))
            if not rows:
                missing.append(arxiv_id)
                if verbose:
                    print(f"  miss   {arxiv_id}")
                continue
            desc = rows[0]["description"]
            found.append(arxiv_id)
            with _research_session() as rg:
                rg.run(
                    "MATCH (p:Paper {arxiv_id: $a}) SET p.description = $desc",
                    a=arxiv_id, desc=desc,
                )
            if verbose:
                print(f"  hit    {arxiv_id}  desc={desc[:60]}...")

    return {"enriched": len(found), "missing": len(missing), "missing_ids": missing}


def find_ungraphed_papers(tag: str, limit: int = 25) -> list[dict[str, Any]]:
    """Find arxiv links in link-forge whose URL/title/concepts match a tag and
    that are NOT yet present in the research graph.

    Used as a discovery aid: "what's in link-forge tagged with `steering`
    that I haven't pulled into the research graph yet?"
    """
    with _research_session() as rg:
        graphed = {
            r["arxiv_id"]
            for r in rg.run("MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id")
        }

    candidates: list[dict[str, Any]] = []
    with _link_forge_session() as session:
        if session is None:
            return []
        rows = session.run(
            """
            MATCH (l:Link)
            WHERE l.url CONTAINS 'arxiv'
              AND (toLower(coalesce(l.title, '')) CONTAINS toLower($tag)
                   OR toLower(coalesce(l.description, '')) CONTAINS toLower($tag)
                   OR toLower(coalesce(l.purpose, '')) CONTAINS toLower($tag))
            RETURN l.url AS url, l.title AS title,
                   l.forgeScore AS forgeScore, l.contentType AS contentType
            ORDER BY coalesce(l.forgeScore, 0) DESC
            LIMIT $limit
            """,
            tag=tag,
            limit=limit * 3,
        )
        for r in rows:
            url = r["url"] or ""
            m = ARXIV_RE.search(url)
            if not m:
                continue
            arxiv_id = m.group(1)
            if arxiv_id in graphed:
                continue
            candidates.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": r["title"],
                    "url": url,
                    "forgeScore": r["forgeScore"],
                    "contentType": r["contentType"],
                }
            )
            if len(candidates) >= limit:
                break
    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("resolve")
    s.add_argument("arxiv_id")

    s = sub.add_parser("enrich")
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("ungraphed")
    s.add_argument("tag")
    s.add_argument("--limit", type=int, default=25)

    args = p.parse_args()

    if args.cmd == "resolve":
        out = resolve_from_linkforge(args.arxiv_id)
        if out is None:
            print(json.dumps({"arxiv_id": args.arxiv_id, "found": False}))
        else:
            print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "enrich":
        summary = enrich_all_papers(verbose=not args.quiet)
        print(json.dumps(summary, indent=2))
    elif args.cmd == "ungraphed":
        rows = find_ungraphed_papers(args.tag, limit=args.limit)
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
