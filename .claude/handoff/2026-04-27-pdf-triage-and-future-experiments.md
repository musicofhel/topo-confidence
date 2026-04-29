# 2026-04-27 — PDF triage + FutureExperiment creation handoff

## What this is

The 2026-04-27 edu-fetch session ingested **51 publisher PDFs** into link-forge. The research-graph hook in link-forge silently skipped all of them because `:Paper {arxiv_id}` is the schema's natural key and PDF uploads carry `file://` URLs with no embedded arxiv ID. Two code fixes have been landed to close that gap going forward (link-forge bridge + OpenAlex publisher blocklist). What remains for the next agent is **post-hoc triage of today's PDFs** + **FutureExperiment node creation** for any genuinely-new triggers those PDFs introduce.

User's stated goal: *"have all the different experiments ready and then run compute on the high roi ones."* Treat NEXT_EXPERIMENTS.md as the destination — every action below should be evaluated by whether it changes that file.

## State as of handoff

- Queue drained. `research-sweep:topo-confidence`: 1008 completed (237 retries succeeded), 215 structural publisher 403s/timeouts left in `failed` (deferred — not retryable without edu-fetch).
- `129` candidates in research-graph `:Paper {status:'candidate'}`, all from the sweep, none from today's PDFs. Awaiting manual promote/reject.
- `NEXT_EXPERIMENTS.md` regenerated 2026-04-27 23:28 UTC: **24 FutureExperiments**, **25 watchlist papers**. Top tier (ROI 9-10): P8-FE1, P9-FE1, P10-FE1, P11-FE5, P3-FE1, P11-FE3.
- Triage report written: `~/topo-confidence/research-graph/edu-fetch-triage-2026-04-27.md` — 51 PDFs in Tier A (15) / Tier B (10) / Tier C (26).
- **Bot restart pending** (see Fix 1 below). Sandbox declined the restart while queue was draining; queue is now drained, user can restart per `~/link-forge/.claude/CLAUDE.md` "Restarting the Bot" section.

## Fixes landed (no further code work needed for these)

### Fix 1: link-forge bridge handles PDF uploads via Semantic Scholar

`~/link-forge/src/processor/research-graph-suggest.ts` — the suggest hook now tries to resolve an arxiv ID for `file://` URLs by hitting Semantic Scholar's `/graph/v1/paper/search` endpoint with the categorizer-extracted title, then verifying token-jaccard ≥ 0.70 and first-author-lastname overlap before accepting the match. On match → existing relevance check + candidate write fires normally. On no-match → silent skip, same as before.

**Bot restart required to load.** From `~/link-forge`:
```bash
pkill -f "tsx src/index.ts"; sleep 2
nohup npx tsx src/index.ts >> logs/bot.log 2>&1 &
pgrep -f "tsx src/index.ts" -a | grep -v pgrep   # verify single instance
```

The patch is defensive: filename-junk titles (e.g. `s41598 023 28985 3`) are short-circuited before hitting SS by requiring ≥3 word tokens of length ≥4 in the title. Real titles (e.g. `Toward Causal Representation Learning`) flow through.

### Fix 2: OpenAlex publisher domain blocklist

`~/link-forge/scripts/lib/paper-apis.ts` — `searchOpenAlex` now drops results whose only URL is a known paywall host. Top blockers from today's failure dig (doi.org, academic.oup.com, MDPI, Wiley, APS, BMJ, Springer, ScienceDirect, IEEE Xplore, ACM, Cell, Tandfonline, JSTOR, IOPscience, SSRN, PubMed, dx.doi.org) plus `www.science.org`. If a result has a non-blocked `pdfUrl`, it stays — open-access PDFs from those hosts (rare) are still kept.

Effect: future `/research-sweep` runs will enqueue ~200 fewer guaranteed-fail OpenAlex hits per sweep. **Genuinely-relevant paywalled papers will still get through if `oa_url` exists** (the blocklist applies to the `landing_page_url` fallback, which is what was reliably 403ing).

No code work needed; takes effect on the next sweep.

## Triage workflow — direct paths

### Where the artifacts live

| Artifact | Path |
|---|---|
| Triage report (Tier A/B/C of today's 51 PDFs) | `~/topo-confidence/research-graph/edu-fetch-triage-2026-04-27.md` |
| Today's PDFs in link-forge (raw query) | see "Re-pull today's PDFs" below |
| Existing candidates in research-graph | `python query.py triage` |
| Promote helper | `python query.py promote <arxiv-id> [--note '...']` |
| Reject helper | `python query.py reject <arxiv-id> [--reason '...']` |
| Manual paper add (skips candidate state) | `python add_paper.py --arxiv <id> --title '...' --status graphed --tags ...` |
| FutureExperiment helper | `python add_future_experiment.py --id PN-FEm --pathway PN ...` (full signature in the file's docstring) |
| Schema (Paper, FutureExperiment, edges) | `~/topo-confidence/research-graph/schema.cypher` |
| Regenerate priority queue | `python generate_next_experiments.py` |

All Python invocations from `~/topo-confidence/research-graph/`.

### Step-by-step plan the next agent should follow

**1. Restart the bot** so the bridge fix is live (see Fix 1 above). Verify pid is single. Tail `logs/bot.log` for any startup errors.

**2. Decide whether to back-fill today's 51 PDFs or move on.** Two options:

- **Option A (recommended): leave the 51 PDFs alone** — open the triage report at `edu-fetch-triage-2026-04-27.md`, eyeball Tier A, and only manually `add_paper.py --status candidate` for the 4-6 papers that look most likely to trigger a NEW experiment vs. just being methodologically adjacent to F-10. Most Tier A entries are PH-on-other-objects (brain connectivity, molecules, 3D shape) — these reinforce F-10's framing but don't introduce new experiment ideas.
- **Option B: re-trigger the bridge.** Once the bot is up with the patch, the bridge only fires on *new* ingests. Re-uploading the 51 PDFs would re-categorize them and is wasteful. To retroactively run the bridge on existing `file://` Links, write a one-off script that iterates `:Link {url STARTS WITH 'file:///'} WHERE savedAt >= '2026-04-27T00:00:00Z'` and calls `suggestToResearchGraph()` directly for each. Roughly:
  ```ts
  // ~/link-forge/scripts/backfill-research-graph-pdfs.ts (NEW, ~50 lines)
  // Connect to link-forge Neo4j (bolt://localhost:7687).
  // For each Link with file:// URL today, call the same suggest function the
  // processor calls. Sequential — Semantic Scholar's free tier is 100/5min.
  ```

**3. Triage current 129 sweep candidates.** The bigger and more impactful pile. `python query.py triage` lists them in DESC order of suggested_at. For each, the `note` field already cites which F-N or H-N the paper bears on. Decision rule:

- If the note cites a finding/hypothesis the project actively cares about (look up `~/topo-confidence/FINDINGS.md` and `~/topo-confidence/HYPOTHESES.md`) AND the paper proposes a method that hasn't been replicated → **promote**, write a 1-2 sentence relevance note that names the F-N/H-N.
- If the note cites a finding/hypothesis but the paper just confirms what's already known → **promote** with a "corroborates F-N" note.
- If the paper is keyword-noise (OpenAlex tokenizer hit on a domain we don't care about — bariatric surgery, Swedish 401(k) econometrics, STEM education research) → **reject** with reason.
- If unsure → **reject** is cheap; the curator gate is supposed to be ruthless.

**4. After promoting, decide which promoted papers should trigger NEW FutureExperiment nodes.** Schema reference (from `schema.cypher`):
```
(Pathway)-[:HAS_FUTURE_EXPERIMENT]->(FutureExperiment)
(FutureExperiment)-[:TRIGGERED_BY {their_method, their_result, our_method, same, differs}]->(Paper)
(FutureExperiment)-[:DEPENDS_ON_FINDING {why}]->(Finding)
(FutureExperiment)-[:WOULD_UPDATE {if_positive, if_negative}]->(Finding)
(FutureExperiment)-[:WOULD_CREATE_FINDING {claim}]->(Tag)
(FutureExperiment)-[:BLOCKED_BY_EXPERIMENT]->(FutureExperiment)
```

ROI scoring rubric (1-10) — derived from current NEXT_EXPERIMENTS.md tiers:
- **9-10 CRITICAL**: re-validation work that updates a foundational finding (e.g., `P8-FE1`, `P9-FE1` rebuilding 256-tok numbers); or a high-leverage method comparison the literature predicts a clear ranking on (e.g., `P10-FE1`).
- **7-8 HIGH**: methodologically novel test that updates an open-question finding (e.g., `P11-FE5`, `P11-FE3` causal patching, `P4-FE2` local LID).
- **5-6 MEDIUM**: confirmation runs (universality across architectures), tractable extensions of partially-explored directions.
- **1-4 LOW**: speculative; only run if compute is free and other priorities are exhausted.

The full create command:
```bash
cd ~/topo-confidence/research-graph
python add_future_experiment.py \
    --id P<N>-FE<M> --pathway P<N> \
    --description "What we'll do, in 1-2 sentences. Concrete enough to check off." \
    --rationale "Why this matters now. Cite the F-N/H-N being tested." \
    --trigger "What in the literature or internal state made this become priority. Brief." \
    --status TRIGGERED --priority HIGH --roi 8 \
    --cost "4h H100" \
    --depends-on F-2 \
    --would-update F-2 \
    --triggered-by 2502.12345 2503.45678
```

After every batch of FutureExperiment additions, **run `python generate_next_experiments.py`** to rewrite `~/topo-confidence/NEXT_EXPERIMENTS.md`. Sanity-check the diff to confirm new entries appear at the expected ROI tier.

**5. Add per-paper trigger metadata.** `add_future_experiment.py --triggered-by` creates `:TRIGGERED_BY` edges with the legacy `reason` field empty. The fields `their_method`, `their_result`, `our_method`, `same`, `differs` (used in NEXT_EXPERIMENTS.md "Source paper" sections) need a separate Cypher write. Pattern from the existing graph:
```cypher
MATCH (fe:FutureExperiment {id: $feId})-[r:TRIGGERED_BY]->(p:Paper {arxiv_id: $arxivId})
SET r.their_method = $theirMethod,
    r.their_result = $theirResult,
    r.our_method = $ourMethod,
    r.same = $same,
    r.differs = $differs
```
Without these fields the FE renders without a source-paper block in NEXT_EXPERIMENTS.md (less polished, still functional).

## Direct queries the next agent will want

### Re-pull today's 51 PDFs from link-forge

```bash
cd ~/link-forge && node -e "
const neo4j = require('neo4j-driver');
const d = neo4j.driver('bolt://localhost:7687', neo4j.auth.basic('neo4j', 'link_forge_dev'));
(async () => {
  const s = d.session();
  const r = await s.run(\`
    MATCH (l:Link)
    WHERE l.url STARTS WITH 'file:///'
      AND l.savedAt >= '2026-04-27T00:00:00Z'
    RETURN l.title AS title, l.authors AS authors, l.keyConcepts AS concepts,
           l.url AS url, l.contentType AS contentType
    ORDER BY l.savedAt DESC
  \`);
  for (const rec of r.records) {
    console.log(JSON.stringify({
      title: rec.get('title'),
      authors: rec.get('authors'),
      concepts: rec.get('concepts'),
      url: rec.get('url'),
    }));
  }
  await s.close(); await d.close();
})();
"
```

### List existing :Paper IDs (to avoid duplicate adds)

```bash
cd ~/topo-confidence/research-graph && python -c "
from query import _run
ids = sorted(r['id'] for r in _run('MATCH (p:Paper) RETURN p.arxiv_id AS id'))
print(f'{len(ids)} papers in graph'); print(ids)
"
```
Last count: 207 papers.

### Inspect a candidate's full record before deciding

```bash
cd ~/topo-confidence/research-graph && python -c "
from query import _run
import json
rows = _run('''
  MATCH (p:Paper {arxiv_id: \$id})
  OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
  RETURN p.arxiv_id AS id, p.title AS title, p.year AS year,
         p.relevance_note AS note, p.status AS status,
         p.linkforge_url AS lf_url, p.suggested_at AS at,
         collect(DISTINCT t.name) AS tags
''', id='2205.14334')
print(json.dumps(rows, indent=2, default=str))
"
```

### Check what's in NEXT_EXPERIMENTS.md without rebuilding

```bash
cd ~/topo-confidence/research-graph && python -c "
from query import _run
rows = _run('''
  MATCH (fe:FutureExperiment)
  OPTIONAL MATCH (fe)-[:TRIGGERED_BY]->(p:Paper)
  RETURN fe.id AS id, fe.pathway_id AS pathway, fe.priority AS priority,
         fe.roi AS roi, fe.status AS status, fe.estimated_cost AS cost,
         collect(DISTINCT p.arxiv_id) AS triggers
  ORDER BY fe.roi DESC, fe.id
''')
for r in rows: print(r)
"
```

## Recommended Tier A → arxiv_id mapping

These are my best guesses for arxiv IDs of the strong-relevance Tier A papers — **the next agent should verify each before adding** by hitting Semantic Scholar (now possible via the patched bridge) or by checking the abs page directly. Not every one will resolve.

| PDF title (link-forge) | Likely arxiv_id | Why it's in Tier A | Probable trigger? |
|---|---|---|---|
| Toward Causal Representation Learning | 2102.11107 | Schölkopf et al. — frames hidden-state representations as causal mechanisms | Could trigger an FE around "is L19 prefill DoM causal vs correlate?" — note this overlaps existing P11-FE3 (activation patching) |
| A Survey on In-context Learning (Dong, EMNLP 2024) | 2301.00234 | ICL framing relevant to F-2 prefill DoM and induction-head hypotheses | Maybe — depends on whether it surveys hidden-state probes for ICL specifically |
| Networks beyond pairwise interactions (Battiston Physics Reports 2020) | 2006.01764 | Higher-order networks / simplicial complex methodology | Methodological context for F-10; unlikely to trigger new FE |
| Persistent Topological Features in LLMs (zigzag) | 2410.11042 | **Already in graph** as trigger for P7-FE1 — verify before adding | Already wired |
| Adversarially Trained PH-GCN (Bian, brain connectivity) | unknown | PH applied to brain graphs — methodologically adjacent | Probably not — wrong domain |
| de Silva-Carlsson "witness-complex" | likely no arxiv | Foundational PH method | Reference only |
| PHG-Net medical image classification | unknown | PH guidance for medical imaging | Wrong domain for trigger |
| PH-GCN 3D Shape Segmentation (Wong) | unknown | PH on 3D meshes | Wrong domain for trigger |
| SynthID-Text (Dathathri Nature 2024) | unknown | LLM watermarking via inference-time intervention | Methodologically related to steering work — could trigger an FE if interesting |
| Causal Deconfounding for Spurious Correlation in Domain Generalization | unknown | Backdoor adjustment / SCM relevant to P5/P6 deconfounding | Maybe — if it offers a method we haven't tried |

**Honest read**: only Schölkopf, Dong-ICL-survey, and possibly SynthID-Text are likely to trigger genuinely-new FutureExperiments. The PH-on-other-objects cluster (8 papers) is reinforcing context for F-10's null result, not triggering material. Don't manufacture FEs to "use" the papers — only add an FE when the paper's method genuinely opens a question we haven't planned for.

## What "done" looks like

After this handoff is executed, the user should be able to:

1. Run `cat ~/topo-confidence/NEXT_EXPERIMENTS.md` and see ≥24 (current count) or more FutureExperiments, none of which are obviously stale.
2. Run `python query.py triage` and see ≤ 30 candidates remaining (the rest promoted or rejected).
3. Pick a top-of-queue FutureExperiment (currently P8-FE1, P9-FE1, P10-FE1) and run it on H100. Pod state in `~/topo-confidence/STATE.md`.

If the next agent is uncertain about a triage decision, **default to reject** — false-rejects cost one paper, false-promotes pollute the graph for everyone.

## References

- Project orientation: `~/topo-confidence/CLAUDE.md`
- Findings registry: `~/topo-confidence/FINDINGS.md` (F-1…F-14)
- Hypotheses queue: `~/topo-confidence/HYPOTHESES.md` (H-1…H-14)
- Project record §1f literature triage notes: `~/topo-confidence/PROJECT_RECORD.md`
- Pathway 11 latest state: `~/topo-confidence/pathway11-h100/`
- Edu-fetch pipeline reference: `~/.claude/projects/-home-musicofhel/memory/link-forge-edu-fetch-pipeline.md`
- Original 2026-04-27 dig handoff: `~/link-forge/.claude/handoff/2026-04-27-research-sweep-failure-dig-edu-pipeline.md`
- Bridge code (after Fix 1): `~/link-forge/src/processor/research-graph-suggest.ts:121-216`
- OpenAlex blocklist (after Fix 2): `~/link-forge/scripts/lib/paper-apis.ts:174-208`
