"""Add a new Paper stub to the research graph.

Example:
    python add_paper.py --arxiv 2501.12345 --title "..." \\
        --year 2025 --repo https://github.com/... \\
        --relevance "..." --tags steering calibration
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--repo", default=None)
    p.add_argument("--relevance", default=None,
                   help="One or two sentences on why this paper is in the graph.")
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args()

    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        s.run(
            """
            MERGE (p:Paper {arxiv_id: $a})
            SET p.title = coalesce($title, p.title),
                p.year = coalesce($year, p.year),
                p.repo_url = coalesce($repo, p.repo_url),
                p.relevance_note = coalesce($rel, p.relevance_note)
            """,
            a=args.arxiv,
            title=args.title,
            year=args.year,
            repo=args.repo,
            rel=args.relevance,
        )
        for tag in args.tags:
            s.run("MERGE (:Tag {name: $t})", t=tag)
            s.run(
                """
                MATCH (p:Paper {arxiv_id: $a}), (t:Tag {name: $tag})
                MERGE (p)-[:TAGGED]->(t)
                """,
                a=args.arxiv,
                tag=tag,
            )
    drv.close()
    print(f"Paper {args.arxiv} written.")


if __name__ == "__main__":
    main()
