// =====================================================================
// topo-confidence research graph schema
// =====================================================================
// Run: cypher-shell -u neo4j -p topo_graph_dev -a bolt://localhost:7688 -f schema.cypher
// =====================================================================

// ---- Node uniqueness constraints ------------------------------------

CREATE CONSTRAINT pathway_id IF NOT EXISTS
  FOR (p:Pathway) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT experiment_id IF NOT EXISTS
  FOR (e:Experiment) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT finding_id IF NOT EXISTS
  FOR (f:Finding) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT artifact_path IF NOT EXISTS
  FOR (a:Artifact) REQUIRE a.path IS UNIQUE;

CREATE CONSTRAINT paper_arxiv IF NOT EXISTS
  FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE;

CREATE CONSTRAINT tag_name IF NOT EXISTS
  FOR (t:Tag) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT future_experiment_id IF NOT EXISTS
  FOR (fe:FutureExperiment) REQUIRE fe.id IS UNIQUE;

// Premise / Method / Dataset uniqueness — prevent silent duplicates on any
// re-run of premises.py / promote_brief.py (added 2026-06-14, rebuild §5).
CREATE CONSTRAINT premise_id IF NOT EXISTS
  FOR (p:Premise) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT method_key IF NOT EXISTS
  FOR (m:Method) REQUIRE m.key IS UNIQUE;

CREATE CONSTRAINT dataset_key IF NOT EXISTS
  FOR (d:Dataset) REQUIRE d.key IS UNIQUE;

// ---- Lookup indexes -------------------------------------------------

// FE status/roi are filtered/sorted on every moot-sweep and queue regen across
// ~1,300 nodes — unindexed before the 2026-06-14 rebuild (§5).
CREATE INDEX future_experiment_status IF NOT EXISTS
  FOR (fe:FutureExperiment) ON (fe.status);

CREATE INDEX future_experiment_roi IF NOT EXISTS
  FOR (fe:FutureExperiment) ON (fe.roi_score);

CREATE INDEX premise_status IF NOT EXISTS
  FOR (p:Premise) ON (p.status);


CREATE INDEX experiment_pathway IF NOT EXISTS
  FOR (e:Experiment) ON (e.pathway_id);

CREATE INDEX finding_status IF NOT EXISTS
  FOR (f:Finding) ON (f.status);

CREATE INDEX finding_strength IF NOT EXISTS
  FOR (f:Finding) ON (f.strength);

CREATE INDEX paper_year IF NOT EXISTS
  FOR (p:Paper) ON (p.year);

CREATE INDEX paper_status IF NOT EXISTS
  FOR (p:Paper) ON (p.status);

// ---- Vector indexes (384-dim all-MiniLM-L6-v2) ---------------------

CREATE VECTOR INDEX paper_embedding_idx IF NOT EXISTS
  FOR (p:Paper) ON (p.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX finding_embedding_idx IF NOT EXISTS
  FOR (f:Finding) ON (f.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};

// ---- Fulltext indexes for RAG --------------------------------------

CREATE FULLTEXT INDEX finding_claims IF NOT EXISTS
  FOR (f:Finding) ON EACH [f.claim, f.strongest_counterargument];

CREATE FULLTEXT INDEX paper_relevance IF NOT EXISTS
  FOR (p:Paper) ON EACH [p.relevance_note, p.title];

CREATE FULLTEXT INDEX experiment_hypotheses IF NOT EXISTS
  FOR (e:Experiment) ON EACH [e.hypothesis, e.result];

CREATE FULLTEXT INDEX future_experiment_search IF NOT EXISTS
  FOR (fe:FutureExperiment) ON EACH [fe.description, fe.trigger, fe.rationale];

// =====================================================================
// Relationship vocabulary (documentation only — Neo4j is schema-free
// for relationships, so these are MERGE'd at seed time).
// =====================================================================
//
// Timeline / structure:
//   (Pathway)-[:NEXT]->(Pathway)
//   (Pathway)-[:HAS_EXPERIMENT]->(Experiment)
//   (Experiment)-[:PRODUCED]->(Finding)
//   (Experiment)-[:USED_ARTIFACT]->(Artifact)
//   (Experiment)-[:PRODUCED_ARTIFACT]->(Artifact)
//   (Experiment)-[:CITED {reason}]->(Paper)
//
// Internal finding relations:
//   (Finding)-[:INVALIDATED_BY {reason}]->(Finding)
//   (Finding)-[:SUPERSEDED_BY {reason}]->(Finding)
//   (Finding)-[:DEPENDS_ON {reason}]->(Finding)
//   (Finding)-[:CONTRADICTS {reason}]->(Finding)
//
// External paper-to-finding relations (the key value of the graph):
//   (Finding)-[:CORROBORATED_BY {their_model, their_benchmark, their_metric,
//     our_metric, method_comparison, note}]->(Paper)
//   (Finding)-[:CONTRADICTED_BY {why, resolution}]->(Paper)
//   (Finding)-[:EXTENDED_BY {experiment_idea, actionable}]->(Paper)
//   (Finding)-[:METHOD_DIFFERS {theirs, ours, outcome_comparison}]->(Paper)
//   (Finding)-[:EXPLAINS {mechanism}]->(Paper)
//
// Tagging:
//   (Paper)-[:TAGGED]->(Tag)
//   (Finding)-[:TAGGED]->(Tag)
//
// Future experiments (forward-looking, distinct from completed Experiment):
//   (Pathway)-[:HAS_FUTURE_EXPERIMENT]->(FutureExperiment)
//   (FutureExperiment)-[:TRIGGERED_BY {reason, their_method, their_result,
//     our_method, same, differs}]->(Paper)
//     - reason         legacy free-form reason (kept for backward compat)
//     - their_method   1-2 sentence neutral summary of the trigger paper's method
//     - their_result   concrete numbers + model + benchmark from the paper
//     - our_method     what we'll do, derived from FE.description + FE.rationale
//     - same           single sentence: what overlaps between their work and ours
//     - differs        single sentence: target signal/benchmark/metric/model deltas
//   (FutureExperiment)-[:DEPENDS_ON_FINDING {why}]->(Finding)
//   (FutureExperiment)-[:WOULD_UPDATE {if_positive, if_negative}]->(Finding)
//   (FutureExperiment)-[:WOULD_CREATE_FINDING {claim}]->(Tag)
//   (FutureExperiment)-[:BLOCKED_BY_EXPERIMENT]->(FutureExperiment)
//
// Premises + closure-by-adjacency (premises.py, moot_sweep.py):
//   (:Premise {id, statement, status, refuted_by, reason, created_date,
//              status_date})
//     - status: LIVE | REFUTED | CONFIRMED. Controlled vocabulary seeded in
//       premises.py — a premise is a falsifiable assumption shared by many FEs.
//   (FutureExperiment)-[:RELIES_ON]->(Premise)
//     - written by promote_brief.py from the FE YAML `relies-on:` field, by
//       premises.py link, or by moot_sweep.py when it moots an FE under a
//       named premise. Refuting a premise cascade-MOOTs every open reliant FE.
//   (FutureExperiment)-[:MOOTED_BY {reason, date}]->(FutureExperiment|Experiment|Premise|Finding)
//   (FutureExperiment)-[:ANSWERED_BY {reason, date}]->(FutureExperiment|Experiment|Premise|Finding)
//     - provenance for FE statuses MOOTED (premise refuted by an adjacent
//       result) and ANSWERED (question settled by an adjacent result).
//       FE.closed_by mirrors the target id as a property. Both statuses are
//       reversible: update_status.py <id> READY.
// =====================================================================
