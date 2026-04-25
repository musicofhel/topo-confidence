#!/usr/bin/env node
// Build a markdown reading list from link-forge's Neo4j graph.
// Pulls priority hits for topo-confidence pathway 9 (hidden-state / CoE / hallucination / topology).
// Usage: node build-reading-list.mjs > literature-sweep.md

import neo4j from "neo4j-driver";
import { readFileSync } from "fs";

const envRaw = readFileSync("/home/musicofhel/link-forge/.env", "utf-8");
const env = Object.fromEntries(
  envRaw
    .split("\n")
    .filter((l) => l && !l.startsWith("#") && l.includes("="))
    .map((l) => {
      const [k, ...rest] = l.split("=");
      return [k.trim(), rest.join("=").trim()];
    }),
);

const driver = neo4j.driver(
  env.NEO4J_URI ?? "bolt://localhost:7687",
  neo4j.auth.basic(env.NEO4J_USER ?? "neo4j", env.NEO4J_PASSWORD ?? "link_forge_dev"),
);

const SAVED_AT_SINCE = "2026-04-22";

const KEYWORD_FILTER = `(
  toLower(coalesce(l.title, '')) CONTAINS 'hidden state'
  OR toLower(coalesce(l.title, '')) CONTAINS 'internal representation'
  OR toLower(coalesce(l.title, '')) CONTAINS 'persistent homolog'
  OR toLower(coalesce(l.title, '')) CONTAINS 'topolog'
  OR toLower(coalesce(l.title, '')) CONTAINS 'hallucinat'
  OR toLower(coalesce(l.title, '')) CONTAINS 'chain of embedding'
  OR toLower(coalesce(l.title, '')) CONTAINS 'semantic entropy'
  OR toLower(coalesce(l.title, '')) CONTAINS 'intrinsic dimension'
  OR toLower(coalesce(l.title, '')) CONTAINS 'eigenvalue'
  OR toLower(coalesce(l.title, '')) CONTAINS 'confidence'
  OR toLower(coalesce(l.title, '')) CONTAINS 'uncertainty'
  OR toLower(coalesce(l.title, '')) CONTAINS 'truthfulness'
  OR toLower(coalesce(l.title, '')) CONTAINS 'calibrat'
  OR toLower(coalesce(l.title, '')) CONTAINS 'self-evaluat'
  OR toLower(coalesce(l.description, '')) CONTAINS 'hidden state'
  OR toLower(coalesce(l.description, '')) CONTAINS 'output-free'
  OR toLower(coalesce(l.description, '')) CONTAINS 'single forward pass'
  OR toLower(coalesce(l.description, '')) CONTAINS 'chain-of-embedding'
  OR toLower(coalesce(l.description, '')) CONTAINS 'persistent homolog'
  OR toLower(coalesce(l.description, '')) CONTAINS 'latent space'
)`;

function arxivYear(url) {
  const m = url.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4})\./);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  const yy = Math.floor(n / 100);
  return yy >= 91 ? 1900 + yy : 2000 + yy;
}

async function runQuery(cypher, params = {}) {
  const session = driver.session();
  try {
    const result = await session.run(cypher, params);
    return result.records.map((r) => Object.fromEntries(r.keys.map((k) => [k, r.get(k)])));
  } finally {
    await session.close();
  }
}

async function main() {
  const overview = await runQuery(
    `MATCH (l:Link) WHERE l.savedAt >= $since
     RETURN count(l) AS total,
            count(CASE WHEN l.url STARTS WITH 'https://arxiv.org/' THEN 1 END) AS arxiv,
            count(CASE WHEN l.contentType = 'research-paper' THEN 1 END) AS papers`,
    { since: SAVED_AT_SINCE },
  );
  const ov = overview[0];

  const priority = await runQuery(
    `MATCH (l:Link)
     WHERE l.savedAt >= $since AND ${KEYWORD_FILTER}
     RETURN l.title AS title, l.url AS url, l.forgeScore AS score,
            l.description AS description, l.contentType AS type
     ORDER BY l.forgeScore DESC LIMIT 80`,
    { since: SAVED_AT_SINCE },
  );

  console.log("# topo-confidence P9 — Literature sweep reading list");
  console.log("");
  console.log(`_Generated ${new Date().toISOString()} from link-forge Neo4j graph._`);
  console.log("");
  console.log(`**Ingest:** ${ov.total.toNumber()} papers since ${SAVED_AT_SINCE} · ${ov.arxiv.toNumber()} arxiv · ${ov.papers.toNumber()} tagged research-paper`);
  console.log("");
  console.log("Refresh: `cd ~/link-forge && node scripts/build-topo-confidence-reading-list.mjs > ~/topo-confidence/pathway9/literature-sweep.md`");
  console.log("");

  const bins = { "2025–2026": [], "2024": [], "2023 and earlier": [] };
  for (const p of priority) {
    const yr = arxivYear(p.url);
    if (yr !== null && yr >= 2025) bins["2025–2026"].push({ ...p, year: yr });
    else if (yr === 2024) bins["2024"].push({ ...p, year: yr });
    else bins["2023 and earlier"].push({ ...p, year: yr ?? "?" });
  }

  for (const label of ["2025–2026", "2024", "2023 and earlier"]) {
    if (bins[label].length === 0) continue;
    console.log(`## ${label} (${bins[label].length})`);
    console.log("");
    for (const p of bins[label]) {
      const score = typeof p.score?.toNumber === "function" ? p.score.toNumber() : p.score;
      const desc = (p.description ?? "").replace(/\s+/g, " ").trim();
      console.log(`### [${p.title}](${p.url})`);
      console.log(`_${p.year} · forgeScore ${score?.toFixed?.(2) ?? score} · ${p.type ?? "?"}_`);
      console.log("");
      console.log(desc);
      console.log("");
    }
  }

  await driver.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
