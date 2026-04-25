#!/usr/bin/env bash
# Build a markdown reading list from the link-forge Neo4j graph after the
# topo-confidence literature sweep ingestion finishes.
#
# Usage: ./build-reading-list.sh > literature-sweep.md

set -euo pipefail

NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-link_forge_dev}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"

cypher() {
  docker exec -i link-forge-neo4j cypher-shell \
    -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
    --format plain "$@"
}

echo "# topo-confidence — P9 literature sweep reading list"
echo ""
echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) from link-forge Neo4j graph."
echo ""

echo "## Ingest overview"
echo ""
echo '```'
cypher <<'CYPHER'
MATCH (l:Link)
WHERE l.savedAt >= date() - duration('P2D')
RETURN count(l) AS papersIngested,
       count(CASE WHEN l.contentType = 'research-paper' THEN 1 END) AS researchPapers,
       count(CASE WHEN l.url STARTS WITH 'https://arxiv.org/' THEN 1 END) AS arxivPapers;
CYPHER
echo '```'
echo ""

echo "## Top concepts in this ingest"
echo ""
cypher <<'CYPHER'
MATCH (l:Link)-[:RELATES_TO_CONCEPT]->(c:Concept)
WHERE l.savedAt >= date() - duration('P2D')
RETURN c.name AS concept, count(l) AS mentions
ORDER BY mentions DESC
LIMIT 25;
CYPHER
echo ""

echo "## Priority papers (title/desc matches target keywords)"
echo ""
cypher <<'CYPHER'
MATCH (l:Link)
WHERE l.savedAt >= date() - duration('P2D')
  AND (
    toLower(coalesce(l.title, '')) CONTAINS 'hidden state'
    OR toLower(coalesce(l.title, '')) CONTAINS 'internal representation'
    OR toLower(coalesce(l.title, '')) CONTAINS 'persistent homolog'
    OR toLower(coalesce(l.title, '')) CONTAINS 'topolog'
    OR toLower(coalesce(l.title, '')) CONTAINS 'hallucinat'
    OR toLower(coalesce(l.title, '')) CONTAINS 'chain of embedding'
    OR toLower(coalesce(l.title, '')) CONTAINS 'semantic entropy'
    OR toLower(coalesce(l.title, '')) CONTAINS 'intrinsic dimension'
    OR toLower(coalesce(l.description, '')) CONTAINS 'output-free'
    OR toLower(coalesce(l.description, '')) CONTAINS 'single forward pass'
  )
RETURN l.title AS title, l.url AS url, l.forgeScore AS score
ORDER BY l.forgeScore DESC
LIMIT 40;
CYPHER
echo ""

echo "## Top authors"
echo ""
cypher <<'CYPHER'
MATCH (l:Link)-[:AUTHORED_BY]->(a:Author)
WHERE l.savedAt >= date() - duration('P2D')
RETURN a.name AS author, count(l) AS papers
ORDER BY papers DESC
LIMIT 25;
CYPHER
