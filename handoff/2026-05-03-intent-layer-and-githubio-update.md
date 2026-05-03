# 2026-05-03 — Intent-layer + github.io update from synthesis results

Written at the end of the session in which `SYNTHESIS.md`,
`NOVELTY_AUDIT.md`, and `APPLICATIONS.md` landed (uncommitted; see
"Working tree" below). The user's ask going into the next session:
*"make sure the intent layers of topo-confidence are all updated
systematically from our results, as well as the github.io page."*

The synthesis docs are the **source-of-current-truth**. Every intent
layer below should be brought into alignment with them.

---

## Working tree at compaction (commit, then update)

```
Branch: max-depth-retriage-2026-04-28 (head 39c264f, pushed)
Untracked:
  SYNTHESIS.md
  NOVELTY_AUDIT.md
  APPLICATIONS.md
  handoff/2026-05-03-findings-synthesis-and-applications.md
  handoff/2026-05-03-intent-layer-and-githubio-update.md   ← this file
```

**First action next session:** `git add` the four new docs + this
handoff and commit as a single "synthesis cascade lands" commit before
starting the propagation work. Keeps the diffs to follow scoped to
intent-layer edits, not synthesis-doc creation.

---

## Intent layers — what each needs, in priority order

The "intent layers" are the canonical narrative docs at the repo root.
Numbers in `validate_claims.py` are authoritative (172/172 internal
PASS, 127/127 Tier-1 regen PASS); narrative docs cite them. Goal:
every narrative doc points at the synthesis docs and uses claim IDs
that grep clean.

### 1. `README.md` — top-of-repo, what visitors see first

**Status:** Untouched since FE291 cascade lands.

**What to update.**
- Add a "Read these in order" pointer at the top: `SYNTHESIS.md`
  (10-min briefing) → `NOVELTY_AUDIT.md` (what's actually new) →
  `APPLICATIONS.md` (where this is deployable).
- Update the headline-numbers paragraph to mention the **decomposition
  triangle** — DoM 0.7679 → 2-feat (PC1, PC9) 0.7856 → cov-spectrum
  0.7928 — and the F-10 strengthening on PC1-residualized clouds
  (gap −0.074, pre-registered overturning condition tested directly
  and failed).
- Reference the github.io page only as the *v1 archive*. Visitors
  arriving from the github.io page should bounce to README which
  bounces to SYNTHESIS.

**Bar to clear.** Anyone landing on the README in 60 seconds knows
(a) the headline AUROC, (b) the decomposition triangle, (c) where
the synthesis docs live, (d) that github.io is archive-only.

### 2. `QUICKSTART.md` — ~480-word orientation

**Status:** Pre-synthesis framing.

**What to update.**
- Replace the "what we found" section with a 4-bullet pull from
  `SYNTHESIS.md` §5 (the two-sentence story).
- Mention the cov-spectrum probe by name (it's the strongest L19
  probe; QUICKSTART should not omit it).
- Drop any 256-tok references that don't carry the explicit
  "[256-tok, superseded]" marker.

**Bar to clear.** A new contributor reads QUICKSTART, knows enough to
pick up SYNTHESIS, and can reproduce `validate_claims.py` PASS in
under an hour.

### 3. `CLAUDE.md` — agent orientation

**Status:** Source-of-truth table at top is current except for two
known discrepancies.

**What to update.**
- **Add three rows** to the "Source of truth" table:
  - `SYNTHESIS.md` — practitioner briefing — "When you need the
    one-page summary"
  - `NOVELTY_AUDIT.md` — F-N novelty rankings vs the 220-paper
    research-graph — "When asked what's new"
  - `APPLICATIONS.md` — deployment surfaces with honest checklists
    — "When asked where to ship this"
- **Fix the F-N count typo.** `CLAUDE.md` "Research graph" section
  references `F-1…F-14` in the Neo4j description, but FINDINGS.md
  has F-1..F-10 (next ID F-11). The graph itself has IDs that have
  drifted from FINDINGS.md — see NOVELTY_AUDIT §header. Correct
  CLAUDE.md to F-1..F-10 and add a one-liner about the drift.
- Update "Headline numbers" table to add the cov-spectrum 0.7928
  row.
- Update the "What the project is, in two sentences" block to match
  SYNTHESIS.md §5 verbatim (so the framing doesn't drift between
  files).

### 4. `STATE.md` — where the most recent session left off

**Status:** Last update 2026-05-01. Says "EXP-58 (P11-FE882)" is the
last experiment. **This session did not run an experiment** — it
wrote synthesis docs.

**What to update.**
- Replace "Last experiment completed" with a *Last documents
  completed* block listing the three synthesis docs and the two
  handoffs.
- Update "Queued — next session" to point at this handoff (intent-
  layer + github.io update), with PLAN_cheap_wins.md as the
  follow-on.
- Update "Open threads" to remove the stale "first cheap-win lands
  through promote_result.py" item — that already happened during
  the FE291 cascade.

### 5. `FINDINGS.md` — F-N registry

**Status:** F-2 and F-10 already fully updated through the FE291
cascade. F-1..F-9 (other than F-2) are at their pre-cascade state.

**What to update (low-impact, audit pass).**
- Skim each F-N for any 256-tok contamination that snuck through.
  Flagged candidates: F-4, F-7 mention 256-tok numbers in the
  evidence section.
- F-1 / F-4 — strongest-criticism section should mention the
  unrun penultimate-token PR null (PERSPECTIVES "strongest criticism"
  block). Currently mentioned but not as a structured "Controls
  not yet run" entry on F-1.
- The "Honorable mentions" candidate "Negative sequence length is
  Pareto-dominant" — fine to keep as a candidate but cross-link to
  APPLICATIONS Application 2.

**Do not promote F-11 yet.** The cov-spectrum / CAST PC1 / F-10
strengthening results live as evidence inside F-2 and F-10 today;
they have not been promoted to standalone F-N entries. Decision for
the next session: leave inside F-2/F-10 (current state), or promote
the cov-spectrum result to a standalone F-11. Recommendation:
**leave embedded** until cross-architecture replication or the
causal companion test runs — promotion needs new evidence, not
re-shuffling.

### 6. `HYPOTHESES.md` — H-N priority queue

**Status:** Last refresh through the FE291 cascade (H-1..H-22 per
the prior backfill).

**What to update.**
- Mark any H-N that the FE291 cascade resolved as `RESOLVED` with
  a one-line outcome and pointer to the EXP / FE.
- Add explicit H-N entries for the unrun follow-ups called out in
  SYNTHESIS §4:
  - Cross-architecture F-2 replication (Phi-3, Llama caches exist)
  - Penultimate-token PR null for F-1 / F-4
  - PC9 mechanism (length axis vs difficulty axis)
  - The 4 unrun causal tests on PC1 (FE214/FE269/FE283 + the
    rank-truncate-cov-spectrum ablation)

**Do not delete H-N entries** — mark resolved and keep them. The
graveyard structure (PROJECT_RECORD §1d) is for killed directions;
HYPOTHESES is a queue with status.

### 7. `PROJECT_RECORD.md` — authoritative archive

**Status:** §1a chronology stops at the May-1 phase 2/3 cheap-wins
landing.

**What to update.**
- §1a: add 1–3 line entries for the 2026-05-02 FE291 PCA cascade
  (FE291 EXP-55, FE880 EXP-56, FE881 EXP-57, FE882 EXP-58) and the
  2026-05-03 synthesis cascade (the three docs).
- §1d (graveyard): one new entry for the F-10 *strengthening* —
  "PH-on-residualized as a falsification of F-10 was tested
  pre-registered and failed by 12.4 pp in the wrong direction." It's
  not a graveyard entry in the usual sense (it didn't kill a
  direction; it strengthened a finding) but the registry should
  have the test-and-result on file. Could also live as a separate
  "negative results that strengthened a finding" sub-section.
- §1e queue: roll forward to point at PLAN_cheap_wins.md.
- §1g file inventory: add the three synthesis docs.

### 8. `PERSPECTIVES.md` — reflective notes

**Status:** Already has "Late additions (2026-05-01)" and "Late
additions (2026-05-02)" sections covering the cov-spectrum + triangle
results.

**What to update.**
- Add a "Late additions (2026-05-03): the synthesis cascade" stub
  noting the three docs landed and what surprised in the synthesis
  process — e.g., NOVELTY_AUDIT identified `2509.12886` as a
  parallel-discovery on F-2 that we already had in the graph but
  hadn't framed as parallel-discovery; the synthesis surfaced it.
- Resist re-stating SYNTHESIS content verbatim — PERSPECTIVES is
  the *thinking-out-loud* register, not the briefing register.

### 9. `PAPER_INDEX.md` — external paper anchors

**Status:** 220+ entries, ~41 external claims registered in
validate_claims (`REGISTERED`).

**What to update (low-priority).**
- For each of the closest-3-prior-work papers in NOVELTY_AUDIT,
  audit the PAPER_INDEX entry to ensure the "What we add / refute"
  paragraph reflects the synthesis-era framing. Big offenders to
  check: `2410.13640` (CoE — F-9 framing), `2509.12886` (LLM Already
  Knows — F-2 parallel-discovery), `2604.22271` (PANL — F-3 pre-hoc
  analog), `2510.18147` (LLMs Encode Difficulty — F-2 / F-6).

### 10. `NEXT_EXPERIMENTS.md` — auto-generated

**Status:** Regenerated through the FE291 cascade promotes.

**What to update.**
- Re-run `python research-graph/generate_next_experiments.py` after
  HYPOTHESES.md updates land. It pulls from `:FutureExperiment`
  Neo4j nodes; HYPOTHESES.md changes don't propagate automatically.
- Verify the top-3 priority queue makes sense given the synthesis —
  the rank-truncate-cov-spectrum causal test should now be a top
  priority (not yet a `:FutureExperiment` node, file as one
  manually).

### 11. `RESEARCH_GRAPH.md` — Neo4j knowledge graph

**Status:** Per CLAUDE.md, this is the schema/usage doc for the graph.

**What to update.**
- Add a "Known ID drift" note: graph F-1..F-13 vs FINDINGS F-1..F-10.
  Document the mapping (graph F-7 = FINDINGS F-10 PH-null; graph
  F-11 = FINDINGS F-8 selective; graph F-13 = ND-10 truncation
  artifact).
- Re-seeding the graph to match FINDINGS is **out of scope for this
  intent-layer pass** — list as a future cleanup queue item.

### 12. `validate_claims.py` + `validation_report.txt`

**Status:** 172/172 internal PASS, 127/127 Tier-1 regen PASS, 216
total claims tracked. No changes needed; just re-run after any
narrative-doc update touches a number, to confirm the invariant
holds.

---

## github.io page — `docs/`

**Status (audit results).**

```
docs/
  index.html                       (Apr 24, 31KB)
  hidden-state-control-map.md      (Apr 26, 20KB, +x)
  literature-map.md                (Apr 26, 47KB, +x)
```

`docs/index.html` is currently the **v1 archive** — title:
*"topo-confidence (v1 archive) — see README for current findings"*.
Meta-description acknowledges the 0.796 MATH-500 AUROC headline was
overturned and points to the GitHub README.

So the page is *deliberately* a redirect/archive surface. It is not
out of date *as an archive*. The decision is whether to keep it that
way or rewrite it.

**Recommendation: rewrite to a "current state, with archive
clearly labeled" page.** The synthesis docs are the right thing to
hand a visitor; an archive-only page invites them to read the wrong
thing first.

### Page rewrite plan

Single-page, dark-theme (existing CSS is good), three vertical
sections:

1. **Headline + 30-second pitch.**
   - Title: "topo-confidence — selective prediction via prefill
     residual-stream geometry on Qwen-2.5-1.5B"
   - Headline numbers card: prefill DoM AUROC 0.7731, cov-spectrum
     0.7928, selective acc 71.6% at 50% coverage. All with
     1024-tok labels and the matching `validate_claims.py` claim ID
     in tooltips.
   - Decomposition triangle table (the four-row table from
     SYNTHESIS §1.5).
   - Three buttons: "Read SYNTHESIS" / "Read NOVELTY_AUDIT" /
     "Read APPLICATIONS" (each linking to the GitHub raw-rendered
     markdown).

2. **What was overturned (the truncation-artifact retraction
   section).**
   - Keep the existing paragraph saying the 0.796 ABC-44 number,
     20.8% MATH-500 baseline, and 7B-as-verifier framing were all
     `max_new_tokens=256` truncation artifacts.
   - Link to PROJECT_RECORD §1d (graveyard) for the full list.

3. **Archive section, collapsed by default.**
   - The current v1 page content (the "old framing" panels) folded
     under a `<details>` block with summary "v1 archive (April
     2026)". Visitors can expand to see what the project looked
     like before the truncation correction; default view is
     current state.

### What does NOT change

- `docs/literature-map.md` and `docs/hidden-state-control-map.md`
  are auxiliary maps from late April. Skim once to ensure no
  256-tok numbers leak; otherwise leave. They're not the landing
  page; they're deep links.

### Deployment

The repo serves `docs/` via GitHub Pages (origin is
`https://github.com/musicofhel/topo-confidence.git`). Push to the
main branch's `docs/` folder updates the page. Confirm the Pages
source-folder setting before assuming push deploys.

**Pre-deploy check:**
- Validate all internal links (SYNTHESIS / NOVELTY / APPLICATIONS
  paths) using `bin/check-md-links.sh` or equivalent. Currently
  no such script exists; raw `grep -rn "\[.*\](.*\.md)"` works.
- Make sure the page doesn't introduce any 256-tok numbers
  without the `[256-tok, superseded]` annotation.

---

## Order of operations next session

1. **Commit the four new docs and this handoff** as one
   "synthesis cascade lands" commit. Push to
   `max-depth-retriage-2026-04-28`.
2. **Run `validate_claims.py`**, confirm 172/172 PASS still holds.
   No narrative-doc changes have happened yet, so it should.
3. **README.md → QUICKSTART.md → CLAUDE.md** — top-of-repo updates
   in order. Each is a small targeted edit; don't rewrite, just
   add the SYNTHESIS pointer + headline-number alignment.
4. **STATE.md** — replace last-experiment block with the synthesis
   cascade.
5. **PROJECT_RECORD §1a / §1d / §1g** — chronology + graveyard +
   file inventory updates.
6. **HYPOTHESES.md** — mark resolved, add the unrun-follow-ups
   from SYNTHESIS §4.
7. **FINDINGS.md audit pass** — 256-tok cleanup, structured
   "Controls not yet run" alignment.
8. **PERSPECTIVES.md** — append the synthesis-cascade reflection.
9. **NEXT_EXPERIMENTS.md** — regen after HYPOTHESES.md edits land.
10. **PAPER_INDEX.md / RESEARCH_GRAPH.md** — low-priority audit
    pass, can defer.
11. **`validate_claims.py`** — re-run, expect 172/172. Commit doc
    updates as a second "intent-layer alignment with synthesis"
    commit.
12. **github.io rewrite (`docs/index.html`)** — single-page rewrite
    with the structure above. Keep existing CSS; replace content.
    Commit and push; confirm Pages deploys.
13. **Final `validate_claims.py`** — third confirmation pass, 172/172.

Total estimated time: 2–3 hours for steps 1–11 (intent layers),
~1 hour for step 12 (github.io rewrite). Single session if
focused.

---

## What this handoff does NOT cover

- The 12 untriaged Discord-admitted papers in Neo4j. Listed as
  "Open queue item 1" in the prior handoff
  (`2026-05-03-findings-synthesis-and-applications.md`); still
  deferred. Pick up after intent-layer + github.io land.
- Pathway 12+ planning. Out of scope.
- The causal companion tests for F-2 (rank-truncate-cov-spectrum,
  PC1 ablation, FE214/FE269/FE283). These are the *next experiments*
  after intent-layer alignment finishes; not part of the
  intent-layer pass itself.
- Re-seeding the research-graph to align F-N IDs with FINDINGS.md.
  Documented as a future cleanup queue item under RESEARCH_GRAPH.md.

---

## Memory aids

- Branch: `max-depth-retriage-2026-04-28` (head 39c264f, pushed)
- Next EXP-id: **59** (no experiment ran this session)
- Next FE-id: **883** (no FE created this session)
- validate_claims target: **172/172 internal PASS** — must hold
  through every doc update
- Tier-1 regen target: **127/127 PASS** — unchanged
- Synthesis docs at:
  - `/home/musicofhel/topo-confidence/SYNTHESIS.md` (375 lines)
  - `/home/musicofhel/topo-confidence/NOVELTY_AUDIT.md` (469 lines)
  - `/home/musicofhel/topo-confidence/APPLICATIONS.md` (387 lines)
- Prior session handoff:
  `handoff/2026-05-03-findings-synthesis-and-applications.md`
  (the queued tasks that *were* completed this session)
- Github.io archive page: `docs/index.html` (v1 framing,
  31KB, Apr 24)
- Github remote: `https://github.com/musicofhel/topo-confidence.git`
- Pages source: presumed `docs/` on `main` (verify in repo settings)
