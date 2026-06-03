from __future__ import annotations

import hashlib
import uuid

from neo4j import GraphDatabase
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship

from ragas_gen.config import NEO4J_BOLT, NEO4J_USER, NEO4J_PASSWORD
from ragas_gen.schemas import (
    FindingEdge,
    Neo4jFinding,
    Neo4jPaper,
    RagasNodePayload,
)

UUID_TO_NEO4J: dict[uuid.UUID, tuple[str, str]] = {}
NEO4J_TO_UUID: dict[tuple[str, str], uuid.UUID] = {}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _get_driver():
    return GraphDatabase.driver(NEO4J_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))


def extract_papers(session) -> list[Neo4jPaper]:
    rows = session.run("""
        MATCH (p:Paper)
        WHERE p.status = 'graphed' AND p.embedding IS NOT NULL
        OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
        WITH p, collect(DISTINCT t.name) AS tags
        OPTIONAL MATCH (p)<-[r]-(f:Finding)
        WHERE type(r) IN ['CORROBORATED_BY','CONTRADICTED_BY','EXTENDED_BY','EXPLAINS','METHOD_DIFFERS']
        WITH p, tags, collect({finding_id: f.id, rel_type: type(r)}) AS edges
        RETURN p, tags, edges
    """).data()

    papers = []
    for r in rows:
        p = r["p"]
        edges = [FindingEdge(**e) for e in r["edges"] if e.get("finding_id")]
        papers.append(Neo4jPaper(
            arxiv_id=p["arxiv_id"],
            title=p.get("title", ""),
            relevance_note=p.get("relevance_note", ""),
            status=p["status"],
            tags=r["tags"],
            embedding=list(p["embedding"]),
            finding_edges=edges,
        ))
    return papers


def extract_findings(session) -> list[Neo4jFinding]:
    rows = session.run("""
        MATCH (f:Finding)
        WHERE f.embedding IS NOT NULL
        RETURN f
    """).data()

    findings = []
    for r in rows:
        f = r["f"]
        findings.append(Neo4jFinding(
            id=f["id"],
            claim=f["claim"],
            status=f.get("status", "active"),
            strength=f.get("strength", "unknown"),
            embedding=list(f["embedding"]),
        ))
    return findings


def paper_to_payload(p: Neo4jPaper) -> RagasNodePayload | None:
    content = f"{p.title}. {p.relevance_note}".strip()
    if len(content) < 10:
        return None
    summary = p.relevance_note or p.title or p.arxiv_id
    if len(summary) < 5:
        summary = f"Paper {p.arxiv_id}"
    return RagasNodePayload(
        page_content=content,
        entities=p.tags if p.tags else [p.arxiv_id],
        themes=p.tags if p.tags else [p.arxiv_id],
        embedding=p.embedding,
        summary=summary,
        neo4j_label="Paper",
        neo4j_id=p.arxiv_id,
        content_hash=_content_hash(content),
    )


def finding_to_payload(f: Neo4jFinding) -> RagasNodePayload:
    content = f"{f.id}: {f.claim}"
    return RagasNodePayload(
        page_content=content,
        entities=[f.id, f.status, f.strength],
        themes=["finding", f.status],
        embedding=f.embedding,
        summary=f.claim,
        neo4j_label="Finding",
        neo4j_id=f.id,
        content_hash=_content_hash(content),
    )


def payload_to_ragas_node(payload: RagasNodePayload) -> Node:
    node = Node(
        id=payload.ragas_uuid,
        type=NodeType.CHUNK,
        properties={
            "page_content": payload.page_content,
            "entities": payload.entities,
            "themes": payload.themes,
            "embedding": payload.embedding,
            "summary": payload.summary,
        },
    )
    UUID_TO_NEO4J[payload.ragas_uuid] = (payload.neo4j_label, payload.neo4j_id)
    NEO4J_TO_UUID[(payload.neo4j_label, payload.neo4j_id)] = payload.ragas_uuid
    return node


def _build_edges(
    papers: list[Neo4jPaper],
    findings: list[Neo4jFinding],
    node_map: dict[tuple[str, str], Node],
    session,
) -> list[Relationship]:
    rels: list[Relationship] = []

    # Finding→Paper typed edges (polarity-preserving)
    for p in papers:
        paper_key = ("Paper", p.arxiv_id)
        if paper_key not in node_map:
            continue
        for edge in p.finding_edges:
            finding_key = ("Finding", edge.finding_id)
            if finding_key not in node_map:
                continue
            polarity = edge.rel_type.lower().replace("_by", "").replace("_", "")
            rels.append(Relationship(
                source=node_map[finding_key],
                target=node_map[paper_key],
                type="entities_overlap",
                properties={
                    "overlapped_items": [f"{edge.finding_id}:{polarity}"],
                    "neo4j_rel": edge.rel_type,
                },
            ))

    # Co-tag edges: papers sharing ≥2 tags
    paper_tags: dict[str, set[str]] = {}
    for p in papers:
        if p.tags:
            paper_tags[p.arxiv_id] = set(p.tags)

    arxiv_ids = list(paper_tags.keys())
    for i, a1 in enumerate(arxiv_ids):
        for a2 in arxiv_ids[i + 1:]:
            shared = paper_tags[a1] & paper_tags[a2]
            if len(shared) >= 2:
                k1 = ("Paper", a1)
                k2 = ("Paper", a2)
                if k1 in node_map and k2 in node_map:
                    rels.append(Relationship(
                        source=node_map[k1],
                        target=node_map[k2],
                        type="entities_overlap",
                        properties={"overlapped_items": sorted(shared)},
                    ))

    # Cosine similarity edges (> 0.7) between findings and papers
    import numpy as np
    all_nodes = []
    all_keys = []
    for p in papers:
        key = ("Paper", p.arxiv_id)
        if key in node_map:
            all_nodes.append(np.array(p.embedding))
            all_keys.append(key)
    for f in findings:
        key = ("Finding", f.id)
        if key in node_map:
            all_nodes.append(np.array(f.embedding))
            all_keys.append(key)

    if len(all_nodes) > 1:
        embs = np.stack(all_nodes)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embs_norm = embs / norms
        sims = embs_norm @ embs_norm.T

        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                if sims[i, j] > 0.7:
                    k1, k2 = all_keys[i], all_keys[j]
                    if k1 in node_map and k2 in node_map:
                        rels.append(Relationship(
                            source=node_map[k1],
                            target=node_map[k2],
                            type="summary_similarity",
                            properties={"summary_similarity": float(sims[i, j])},
                        ))

    return rels


def build_knowledge_graph() -> tuple[KnowledgeGraph, dict[tuple[str, str], RagasNodePayload]]:
    UUID_TO_NEO4J.clear()
    NEO4J_TO_UUID.clear()

    with _get_driver() as driver, driver.session() as session:
        papers = extract_papers(session)
        findings = extract_findings(session)

    print(f"  Extracted {len(papers)} papers, {len(findings)} findings from Neo4j")

    payloads: dict[tuple[str, str], RagasNodePayload] = {}
    node_map: dict[tuple[str, str], Node] = {}

    skipped = 0
    for p in papers:
        payload = paper_to_payload(p)
        if payload is None:
            skipped += 1
            continue
        key = ("Paper", p.arxiv_id)
        payloads[key] = payload
        node_map[key] = payload_to_ragas_node(payload)

    if skipped:
        print(f"  Skipped {skipped} papers with insufficient content")

    for f in findings:
        payload = finding_to_payload(f)
        key = ("Finding", f.id)
        payloads[key] = payload
        node_map[key] = payload_to_ragas_node(payload)

    with _get_driver() as driver, driver.session() as session:
        rels = _build_edges(papers, findings, node_map, session)

    kg = KnowledgeGraph()
    kg.nodes.extend(node_map.values())
    kg.relationships.extend(rels)

    print(f"  Built KG: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
    return kg, payloads


def compute_graph_snapshot_hash(payloads: dict[tuple[str, str], RagasNodePayload]) -> str:
    import hashlib
    parts = []
    for (label, nid), payload in sorted(payloads.items()):
        parts.append(f"{label}:{nid}:{payload.content_hash}")
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
