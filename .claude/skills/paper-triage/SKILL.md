# /paper-triage

Deep triage of a single arxiv paper for the topo-confidence research program.
Produces a structured brief at `research-graph/briefs/triage-YYYY-MM-DD-<arxiv-id>.md`
that can be fed to `promote_brief.py` for auto-promotion into the research graph.

## Usage

`/paper-triage <arxiv-id>` — e.g. `/paper-triage 2605.05873`

## How it works

### Step 0: Check for existing brief

```bash
ls ~/topo-confidence/research-graph/briefs/triage-*-<arxiv-id>.md 2>/dev/null
```

If a brief already exists, report it and ask whether to overwrite or skip.

### Step 1: Load default context (NOTHING ELSE until Refutations is written)

Load these files in this order. Do not load anything else until after the
Refutations section is written. This ordering is a confirmation-bias defense
per Rios-Garcia 2604.18805 (68% in agent traces).

1. Headlines table from `~/topo-confidence/CLAUDE.md` — read only the
   `## Headline numbers` section, not the whole file.
2. F-N + H-N one-liners:
   ```bash
   grep -E "^### (F|H)-[0-9]+:" ~/topo-confidence/FINDINGS.md ~/topo-confidence/HYPOTHESES.md
   ```
3. Existing PAPER_INDEX.md status flags:
   ```bash
   grep -E "^## [0-9]+\.|^\*\*Status:\*\*" ~/topo-confidence/PAPER_INDEX.md
   ```
4. The paper's prior admission note:
   ```bash
   cd ~/topo-confidence/research-graph && python query.py paper <arxiv-id>
   ```

Do **not** load the full FINDINGS.md, HYPOTHESES.md, or NEXT_EXPERIMENTS.md by
default. Loading them causes the model to fit the paper into existing frames
instead of finding refutations.

### Step 2: Fetch the paper

Use `mcp__paper-search__read_arxiv_paper` or `mcp__arxiv__read_paper`. If
those fail, fall back to `WebFetch` of `https://arxiv.org/abs/<arxiv-id>`.

### Step 3: Write the brief

Save to `~/topo-confidence/research-graph/briefs/triage-{today}-{arxiv-id}.md`.

The brief must follow this exact structure, in this exact order:

```markdown
# Triage brief — <arxiv-id> — <paper title>

Source: https://arxiv.org/abs/<arxiv-id>
Triaged: <today>

## Refutations

List >=3 distinct ways this paper would *refute* current F-N findings or H-N
hypotheses. Each refutation must:
  - cite a specific F-N or H-N (not "their work in general"),
  - name a mechanism by which the refutation would fire,
  - be falsifiable by a concrete experiment.

If you genuinely cannot find >=3 substantive refutations after a careful read,
write "No genuine refutations identified after [N] minutes — paper is likely
off-topic or purely confirming. Recommend rejection or weak CITED ONLY status."
Do not invent thin refutations just to fill the section.

## Direct connections to F-N / H-N

Each connection must cite a specific F-N or H-N and the relevant numerical
anchor from the headlines table (AUROC 0.7731, cos 0.046, peak PR 67/88, etc.).
On-demand Read of ~/topo-confidence/FINDINGS.md for specific F-N sections is
allowed — log reads in "Sources consulted".

## Methodologies extracted

Every technique / method / probe / loss / training trick the paper uses that
touches hidden states, residual streams, probing, steering, geometry, or
correctness signals. For each:

- **<method name>** — one-line description.
  *Replication cost*: e.g. "20min CPU", "H100 day", "needs new dataset".
  *We'd plausibly run this*: yes | no — and one phrase on why or why not.

Leave empty (write "none extracted") only if the paper is pure theory with no
method we could borrow.

## Approaches & framings

Theoretical lens shifts, reformulations, and conceptual moves the paper
introduces. One sentence each. How does each intersect our F-N / H-N framings?

Leave empty (write "none extracted") if the paper makes only narrow incremental
claims with no reusable framing.

## Datasets & benchmarks

Datasets and benchmarks used or introduced, with applicability to MATH-500 /
1024-tok regime / hidden-state pipeline. For each:

- **<name>** — size, license, accessibility (HF | gated | proprietary | not
  released). Applicable? yes | no — one phrase.

Leave empty (write "none") if purely theoretical or trivial synthetic data.

## Implementation details worth capturing

Architectures, hyperparameters, training schedules, gotchas, code links. Bullet
list. Leave empty (write "none") if no concrete implementation guidance.

## Replicable intermediates

The smallest cross-check we could run *right now* against the paper's headline
claim with cached NPZs / P11 H100 data / frozen extractors. Bullet list naming
the cached artifact / script. Leave empty (write "none — paper claims need
fresh data we don't have") if no sanity check is possible.

## Cross-paper signals

Papers this paper cites that are already in our research-graph, OR papers it
cites that should be (and aren't yet). Format:

- <arxiv-id> — already in graph (status: graphed | pending_triage). Connection.
- <arxiv-id> — NOT in graph; recommend admission. Connection.

Run `python query.py paper <arxiv-id>` against the graph for any cited paper
you suspect is relevant. Leave empty (write "none worth flagging") otherwise.

## Proposed FutureExperiments

Generate experiments in two tiers. Every paper should produce AT LEAST one
Tier 1 experiment if at all possible.

### Tier 1 — Cheap local (5min-2hr, CPU only, ROI 5-10)

Can ANY technique, metric, decomposition, statistical test, distance measure,
or analytical framework from this paper be applied to our cached NPZ
activations (L19 prefill, 1024-tok, Qwen-2.5-1.5B, MATH-500, 500 samples x
1536-dim hidden states)? Think abstractly — the paper doesn't need to be about
LLMs for its method to apply to our cached activation matrices.

Mark these with `cost: "<N>min CPU"` and `roi:` reflecting how much the result
would move a finding or hypothesis.

### Tier 2 — H100 required (hours-days, high ROI only)

Does the paper suggest a probe architecture, steering intervention, new
extraction regime, or training-time analysis that requires fresh forward passes
on GPU? Only propose these if ROI >= 7.

One or more YAML blocks with EXACTLY these fields (every field required):

    ```yaml
    - id: P11-FE<next>
      pathway: P11
      description: "..."
      rationale: "..."
      trigger: "..."
      status: READY
      priority: HIGH
      roi: 8
      cost: "20min CPU"
      depends-on: [F-1, F-7]
      would-update: [F-7]
      triggered-by: ["<arxiv-id>"]
      blocked-by: null
      blocked-by-experiment: []
    ```

Find the next available ID via:
```bash
cd ~/topo-confidence/research-graph && python query.py future P11
```

Pick numbers well above the highest currently-graphed ID (add 50+ to avoid
collisions with concurrent briefs).

## Proposed HYPOTHESES.md additions

H-N blocks using the template from the end of HYPOTHESES.md. Pick numbers well
above the current Next ID to avoid collisions.

    ### H-<next>: <one-line hypothesis>
    **Priority:** HIGH | MEDIUM | LOW | PARKED
    **Motivated by:** <arxiv-id> + <relevant F-N or EXP-N>
    **Test:** <what you'd run, what data, est time>
    **Requires:** <CPU/GPU, model, cached or new data>
    **Would change:** <what shifts on confirm vs reject>
    **Blocks:** <H-N or "nothing">

Skip if paper motivates only methodology FEs without new hypotheses.

## Proposed PAPER_INDEX.md classification

    ## <arxiv-id> — <paper title> (<authors>, <year>)

    **Relevance:** <one paragraph>

    **Key claim we tested:** <their claim, in our terms>

    **Our result:** **<STATUS>** <one paragraph>

    **Related experiments:** <FE-N, H-N, EXP-N>

    **Status:** REPLICATED | CONTRADICTED | PARTIALLY CONFIRMED | CITED ONLY | TO TEST

## New claims

List quantitative claims this brief introduces into narrative docs that need a
validate_claims.py entry. Empty if the brief only quotes existing numbers.

## Sources consulted

List every file you read on-demand beyond the default context. Format:
    - <path> — <reason>
```

### Step 4: Report

Print the absolute path of the brief on completion. If auto-promote is desired,
tell the user:

```bash
cd ~/topo-confidence/research-graph && python promote_brief.py briefs/triage-{today}-{arxiv-id}.md --dry-run
```

## Hard rules

1. **Refutations first.** Write the Refutations section before any on-demand
   Read of F-N bodies. This is the confirmation-bias defense.
2. **Use only F-N IDs that exist** (F-1 through F-10 currently). Verify with
   the grep from Step 1.
3. **Use only high H-N numbering** to avoid collisions with concurrent briefs.
4. **Brief filename**: exactly `briefs/triage-{today}-{arxiv-id}.md` relative
   to `~/topo-confidence/research-graph/`.
5. **Leave a section empty** (with "none" or "none extracted") rather than
   inventing filler.
6. **Local compute only.** 2060 Super + CPU. Tier 2 FEs requiring H100 should
   be flagged but are not actionable until budget allows.

## Batch mode

For batch triage of all pending papers, use the existing shell dispatcher:

```bash
cd ~/topo-confidence/research-graph && bash triage_pending.sh
```

This spawns parallel `claude -p` workers using `_paper_triage_prompt.template`
(same content as this skill). Each worker auto-promotes its brief via
`promote_brief.py` with flock serialization.
