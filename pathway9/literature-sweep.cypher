// topo-confidence P9 literature sweep — Cypher queries against link-forge Neo4j
// Run with: cypher-shell -u neo4j -p link_forge_dev -f literature-sweep.cypher
// or paste blocks into Neo4j Browser at http://localhost:7474

// =====================================================================
// 1. Overview — papers ingested from the topo-confidence search pass
// =====================================================================
// These are Links enqueued with discordAuthorName='paper-search' (see search-papers.ts).
// Note: that metadata lives in the SQLite queue, not Neo4j. We identify recent
// ingest by savedAt + URL pattern instead.

MATCH (l:Link)
WHERE l.savedAt >= date() - duration('P2D')
  AND (l.url STARTS WITH 'https://arxiv.org/'
    OR l.url STARTS WITH 'https://doi.org/'
    OR l.contentType IN ['research-paper', 'whitepaper'])
RETURN count(l) AS recentPapers;

// =====================================================================
// 2. Priority papers — title or description mentions hidden states / internal
//    representations / correctness-from-internals
// =====================================================================
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
    OR toLower(coalesce(l.title, '')) CONTAINS 'eigen'
    OR toLower(coalesce(l.title, '')) CONTAINS 'intrinsic dimension'
    OR toLower(coalesce(l.description, '')) CONTAINS 'hidden state'
    OR toLower(coalesce(l.description, '')) CONTAINS 'internal representation'
    OR toLower(coalesce(l.description, '')) CONTAINS 'persistent homolog'
    OR toLower(coalesce(l.description, '')) CONTAINS 'output-free'
    OR toLower(coalesce(l.description, '')) CONTAINS 'single forward pass'
  )
RETURN l.title AS title, l.url AS url, l.forgeScore AS score,
       l.contentType AS type, substring(coalesce(l.description, ''), 0, 300) AS summary
ORDER BY l.forgeScore DESC
LIMIT 40;

// =====================================================================
// 3. Concept network — what concepts dominate this ingest?
// =====================================================================
MATCH (l:Link)-[:RELATES_TO_CONCEPT]->(c:Concept)
WHERE l.savedAt >= date() - duration('P2D')
RETURN c.name AS concept, count(l) AS mentions
ORDER BY mentions DESC
LIMIT 30;

// =====================================================================
// 4. Top authors in the ingest (2025–2026 recent papers especially)
// =====================================================================
MATCH (l:Link)-[:AUTHORED_BY]->(a:Author)
WHERE l.savedAt >= date() - duration('P2D')
RETURN a.name AS author, count(l) AS papers,
       collect(l.title)[0..3] AS sampleTitles
ORDER BY papers DESC
LIMIT 25;

// =====================================================================
// 5. Recent (2024+) papers — arxiv IDs starting with 24, 25, 26
// =====================================================================
MATCH (l:Link)
WHERE l.savedAt >= date() - duration('P2D')
  AND l.url =~ 'https://arxiv\\.org/abs/(24|25|26)[0-9]{2}\\..*'
RETURN l.title AS title, l.url AS url, l.forgeScore AS score,
       l.keyConcepts AS concepts
ORDER BY l.forgeScore DESC
LIMIT 50;

// =====================================================================
// 6. Cluster detection via concept co-occurrence in the new ingest
// =====================================================================
MATCH (c1:Concept)<-[:RELATES_TO_CONCEPT]-(l:Link)-[:RELATES_TO_CONCEPT]->(c2:Concept)
WHERE l.savedAt >= date() - duration('P2D')
  AND c1.name < c2.name
RETURN c1.name AS conceptA, c2.name AS conceptB, count(l) AS coMentions
ORDER BY coMentions DESC
LIMIT 25;

// =====================================================================
// 7. Vector-similarity neighborhood — find papers closest to a seed query
//    (requires passing a 384-d embedding; use via forge_search MCP tool instead)
// =====================================================================
// See mcp tool: forge_search(query="hidden state confidence LLM correctness", ...)
