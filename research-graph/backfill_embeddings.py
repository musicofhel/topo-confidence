"""Backfill vector embeddings for Paper, Finding, and FutureExperiment nodes.

Uses all-MiniLM-L6-v2 (384-dim, cosine) from sentence-transformers.

Usage:
    python backfill_embeddings.py                      # embed all missing
    python backfill_embeddings.py --node-type Paper     # papers only
    python backfill_embeddings.py --force               # re-embed even if exists
    python backfill_embeddings.py --dry-run             # preview without writing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_driver():
    return GraphDatabase.driver(
        BOLT, auth=(USER, PASSWORD), notifications_min_severity="OFF",
    )


def _build_paper_text(rec: dict) -> str:
    title = rec.get("title") or ""
    note = rec.get("relevance_note") or ""
    tags_raw = rec.get("tags") or []
    tags = " ".join(t.replace("_", " ") for t in tags_raw)
    parts = [p for p in [title, note, tags] if p]
    return " ".join(parts)


def _build_finding_text(rec: dict) -> str:
    claim = rec.get("claim") or ""
    counter = rec.get("strongest_counterargument") or ""
    parts = [p for p in [claim, counter] if p]
    return " ".join(parts)


def _build_fe_text(rec: dict) -> str:
    desc = rec.get("description") or ""
    rationale = rec.get("rationale") or ""
    parts = [p for p in [desc, rationale] if p]
    return " ".join(parts)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_node(node_type: str, node_id: str, text: str):
    """Embed a single node and write to Neo4j. Used by promote_brief.py."""
    vecs = embed_texts([text])
    if node_type == "Paper":
        id_field, id_prop = "arxiv_id", node_id
    elif node_type in ("Finding", "FutureExperiment"):
        id_field, id_prop = "id", node_id
    else:
        raise ValueError(f"Unknown node type: {node_type}")

    cypher = f"""
        MATCH (n:{node_type} {{{id_field}: $nid}})
        SET n.embedding = $vec
    """
    with _get_driver() as drv, drv.session() as s:
        s.run(cypher, nid=id_prop, vec=vecs[0])


def backfill_papers(force: bool = False, dry_run: bool = False) -> int:
    where = "" if force else "WHERE p.embedding IS NULL"
    with _get_driver() as drv, drv.session() as s:
        rows = list(s.run(f"""
            MATCH (p:Paper)
            {where}
            OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
            WITH p, collect(t.name) AS tags
            RETURN p.arxiv_id AS arxiv_id, p.title AS title,
                   p.relevance_note AS relevance_note, tags
            ORDER BY p.arxiv_id
        """))

    if not rows:
        print("  Papers: nothing to embed")
        return 0

    texts = [_build_paper_text(dict(r)) for r in rows]
    ids = [r["arxiv_id"] for r in rows]

    if dry_run:
        print(f"  Papers: would embed {len(texts)} nodes")
        for i, (aid, t) in enumerate(zip(ids, texts)):
            if i < 3:
                print(f"    {aid}: {t[:80]}...")
        return len(texts)

    vecs = embed_texts(texts)

    with _get_driver() as drv, drv.session() as s:
        for aid, vec in tqdm(zip(ids, vecs), total=len(ids), desc="  Papers"):
            s.run(
                "MATCH (p:Paper {arxiv_id: $aid}) SET p.embedding = $vec",
                aid=aid, vec=vec,
            )

    print(f"  Papers: embedded {len(ids)} nodes")
    return len(ids)


def backfill_findings(force: bool = False, dry_run: bool = False) -> int:
    where = "" if force else "WHERE f.embedding IS NULL"
    with _get_driver() as drv, drv.session() as s:
        rows = list(s.run(f"""
            MATCH (f:Finding)
            {where}
            RETURN f.id AS id, f.claim AS claim,
                   f.strongest_counterargument AS strongest_counterargument
            ORDER BY f.id
        """))

    if not rows:
        print("  Findings: nothing to embed")
        return 0

    texts = [_build_finding_text(dict(r)) for r in rows]
    ids = [r["id"] for r in rows]

    if dry_run:
        print(f"  Findings: would embed {len(texts)} nodes")
        for i, (fid, t) in enumerate(zip(ids, texts)):
            if i < 3:
                print(f"    {fid}: {t[:80]}...")
        return len(texts)

    vecs = embed_texts(texts)

    with _get_driver() as drv, drv.session() as s:
        for fid, vec in tqdm(zip(ids, vecs), total=len(ids), desc="  Findings"):
            s.run(
                "MATCH (f:Finding {id: $fid}) SET f.embedding = $vec",
                fid=fid, vec=vec,
            )

    print(f"  Findings: embedded {len(ids)} nodes")
    return len(ids)


def backfill_future_experiments(force: bool = False, dry_run: bool = False) -> int:
    where = "" if force else "WHERE fe.embedding IS NULL"
    with _get_driver() as drv, drv.session() as s:
        rows = list(s.run(f"""
            MATCH (fe:FutureExperiment)
            {where}
            RETURN fe.id AS id, fe.description AS description,
                   fe.rationale AS rationale
            ORDER BY fe.id
        """))

    if not rows:
        print("  FutureExperiments: nothing to embed")
        return 0

    texts = [_build_fe_text(dict(r)) for r in rows]
    ids = [r["id"] for r in rows]

    if dry_run:
        print(f"  FutureExperiments: would embed {len(texts)} nodes")
        for i, (fid, t) in enumerate(zip(ids, texts)):
            if i < 3:
                print(f"    {fid}: {t[:80]}...")
        return len(texts)

    vecs = embed_texts(texts)

    with _get_driver() as drv, drv.session() as s:
        for fid, vec in tqdm(zip(ids, vecs), total=len(ids),
                             desc="  FutureExperiments"):
            s.run(
                "MATCH (fe:FutureExperiment {id: $fid}) SET fe.embedding = $vec",
                fid=fid, vec=vec,
            )

    print(f"  FutureExperiments: embedded {len(ids)} nodes")
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description="Backfill vector embeddings")
    parser.add_argument("--node-type",
                        choices=["Paper", "Finding", "FutureExperiment", "all"],
                        default="all")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed even if embedding exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    args = parser.parse_args()

    total = 0
    if args.node_type in ("Paper", "all"):
        total += backfill_papers(force=args.force, dry_run=args.dry_run)
    if args.node_type in ("Finding", "all"):
        total += backfill_findings(force=args.force, dry_run=args.dry_run)
    if args.node_type in ("FutureExperiment", "all"):
        total += backfill_future_experiments(force=args.force, dry_run=args.dry_run)

    print(f"\n  Total: {total} nodes {'would be ' if args.dry_run else ''}embedded")


if __name__ == "__main__":
    main()
