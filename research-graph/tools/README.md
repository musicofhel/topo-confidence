# research-graph/tools/

Robustness tooling for the topo-confidence research pipeline. Each tool here
addresses a concrete failure mode hit during the synthesis-cascade work
(2026-05-03). Tools are deliberately small (~200 LOC each), dependency-free
where possible, and CI-friendly (exit 0 / 1).

| Tool | Script | Pain it addresses |
|---|---|---|
| tool-01 | `headline_linter.py` | Headline numbers drift across SYNTHESIS / README / CLAUDE / docs |
| tool-02 | `finding_drift.py` | Graph `(:Finding)` IDs drift from FINDINGS.md F-N IDs |
| tool-05 | `artifact_meta.py` | NPZ/JSON artifacts lack provenance; 256-tok/1024-tok mixed silently |
| tool-11 | `brief_score.py` | `/paper-triage` briefs produce filler in expanded sections |

Build order, conventions, and v2 backlog below.

---

## tool-01 — `headline_linter.py`

**What it does.** Reads `validate_claims.py`'s `CLAIMS` list (216 entries, the
single source of truth for every quantitative claim in the project). For
each canonical narrative doc, scans for backtick-tagged claim IDs
(``` `prefill-dom-auroc` ```), extracts the nearest numeric value, and
flags any disagreement with the registry.

**Conventions.**
- Tag a claim by writing the cid in backticks adjacent to the value:
  ``the AUROC is **0.7731** (`prefill-dom-auroc`).``
- For a name-only reference (the cid is mentioned as evidence but the
  doc doesn't display its value), opt out with an inline HTML comment:
  ``(`fe145-l19-within-band` <!-- noclaim:fe145-l19-within-band — name-only ref --> )``
- Use `~` to mark deliberate approximations: ``the PR is ~30 by position 50
  (`gib-math-pos50`)``. The linter relaxes tolerance to ±1 unit at the shown
  precision when `~` precedes a value.

**Run.**
```bash
python research-graph/tools/headline_linter.py            # report
python research-graph/tools/headline_linter.py --verbose  # also list clean docs
python research-graph/tools/headline_linter.py --json out.json
```

**Coverage as of 2026-05-03.** 75 cids actively drift-checked across
SYNTHESIS.md (58), APPLICATIONS.md (12), NOVELTY_AUDIT.md (5).
README / QUICKSTART / CLAUDE / STATE / FINDINGS / docs/index.html have
zero tagged cids — that's the v2 tagging-pass.

---

## tool-02 — `finding_drift.py`

**What it does.** Diffs the Neo4j research-graph's `(:Finding {id})` set
against `FINDINGS.md`'s F-N entries. Reports three drift modes: extras
(graph nodes with no FINDINGS.md counterpart), missing (FINDINGS.md F-N
absent from graph), and mismatched-claims (same ID, divergent claim text by
Jaccard threshold).

**Why now.** The graph carries F-1..F-14; FINDINGS.md is F-1..F-10. Beyond
the count drift, every shared ID is misaligned: graph F-6 ("CoE-60 0.811
AUROC") vs FINDINGS F-6 ("7B prefill PR inversion"), graph F-11 (selective
prediction) vs FINDINGS F-8 (selective prediction), etc. Operator-led
re-seeding has been queued cleanup; this tool is the diff that drives it.

**Heuristic remap.** For graph extras, the tool computes Jaccard token
overlap (with stopwords + length filter) against every FINDINGS.md
(title + claim). Above `JACCARD_HINT=0.05` becomes a candidate;
above `JACCARD_OK=0.30` is treated as aligned. Heuristic — operator
must review every proposed remap before applying.

**Run.**
```bash
python research-graph/tools/finding_drift.py
python research-graph/tools/finding_drift.py --verbose
python research-graph/tools/finding_drift.py --json drift.json
python research-graph/tools/finding_drift.py --cypher migration.cypher
```

**Coverage as of 2026-05-03.** Graph F-1..F-14 vs FINDINGS F-1..F-10:
4 extras (graph F-11..F-14), 0 missing, 10 ID-shared with claim drift.
Two extras get high-confidence remaps (graph F-11→FINDINGS F-8 selective
prediction, graph F-12→FINDINGS F-7 D-bucket); the other two (F-13
truncation artifact = ND-10, F-14 monotonic gating) have no FINDINGS.md
counterpart and need either archival or new F-N seeding.

**Cypher migration policy.** Output is commented-out `MATCH … SET f.id`
statements so nothing executes accidentally — operator hand-applies after
review. Apply order matters: collapse extras first (frees IDs), then re-id
shared-but-divergent claims, then seed missing.

---

## tool-05 — `artifact_meta.py`

**What it does.** Defines a stamping convention for hidden-state NPZ / JSON
artifacts so cross-`seq_len` mixing (the failure mode that produced the v1
docs/index.html retraction) becomes a loud refusal instead of a silent bug.

**Convention.** Every artifact carries a `meta` block with:

```python
{
  "seq_len":   int,
  "tokenizer": str,    # HF tokenizer ID
  "model_id":  str,    # HF model ID
  "dataset":   str,
  "git_sha":   str,    # 7-char commit
  "timestamp": str,    # ISO-8601 UTC
  "script":    str,    # producing script path
  "schema":    "topo-confidence/v1",
}
```

NPZ files: stored under reserved key `__meta__`, JSON-serialized.
JSON files: stored at top-level key `meta`.

**Library API.**
```python
from artifact_meta import pack_meta, write_npz, write_json, read_meta, assert_compatible

meta = pack_meta(seq_len=1024, tokenizer="Qwen/Qwen2.5-1.5B-Instruct",
                 model_id="Qwen/Qwen2.5-1.5B-Instruct", dataset="MATH-500")
write_npz("results.npz", meta, prefill_score=arr1, correct=arr2)

# Cross-artifact safety:
assert_compatible(["a.npz", "b.npz", "c.npz"])  # raises CompatibilityError on seq_len drift
```

**CLI.**
```bash
python research-graph/tools/artifact_meta.py inspect path/to/file.npz
python research-graph/tools/artifact_meta.py scan pathway11_h100/      # report unstamped
python research-graph/tools/artifact_meta.py compat a.npz b.npz        # check compat
python research-graph/tools/artifact_meta.py demo                      # write a sample
```

**v2 backlog.** Backfill existing NPZs (~8+ in `pathway11_h100/prefill_gated_compute/`
alone) with stamps. Scripts that write artifacts should switch to
`write_npz` / `write_json` from this module.

---

## tool-11 — `brief_score.py`

**What it does.** Scores each of the 6 expanded sections of a `/paper-triage`
brief on a 0–3 heuristic scale, plus the Refutations section's bullet count.
Refuses promotion (exit 1) when sections are filler.

**Scoring.**
- 0: section absent OR explicit "N/A" placeholder (legitimate for some papers)
- 1: present but superficial (<80 words or <2 bullets)
- 2: substantive (≥80 words, ≥2 bullets)
- 3: rich (≥200 words, ≥3 bullets, with cost/code-link/impl-detail keywords)

**Promote-block policy.**
- Refutations section: ≥3 bullets (or explicit "no genuine refutations identified" escape)
- Each of the 6 expanded sections: ≥1 (allowing legitimate explicit-N/A)
- ≥4 of the 6 expanded sections must be ≥2

**Run.**
```bash
python research-graph/tools/brief_score.py research-graph/briefs/triage-X.md
python research-graph/tools/brief_score.py --all-in research-graph/briefs/
python research-graph/tools/brief_score.py briefs/X.md --json
```

**Recognized item-marker styles** (each counts as one bullet):
- `- foo` / `* foo` / `+ foo`
- `1. foo` / `**1. foo**`
- `R1 — foo` / `**H3 — foo**` / `R1. foo`
- `### 1. foo` / `### Foo`

**As of 2026-05-03.** 221 existing briefs scored: 219 PASS, 2 BLOCK
(both with 3/6 substantive sections — borderline content the operator
should review).

---

## Conventions across all tools

- **No external deps beyond `numpy`** (used only by `artifact_meta.py`).
- **Importable as a library**, not just a CLI.
- **Exit 0 on success, 1 on failure** so they slot into pre-commit / CI.
- **Lives in `research-graph/tools/`** so the existing scripts can
  `from tools.foo import bar` by adjusting `sys.path`.

## Build order (recommended next, per the plan)

The plan in `~/.claude/plans/2026-05-03-pipeline-robustness-tools.md`
ranks 11 more tools. Highest-leverage queued items:

- **tool-09** pre-registration ledger (machine-readable H-N thresholds).
- **tool-10** overturn-condition tracker (auto-flag when an experiment
  satisfies a Finding's pre-registered overturn clause; F-10 caught manually).
- **tool-13** cross-arch replication harness (unblocks H-698, currently
  unscoped).
- **tool-14** validate-claims pre-commit hook (the 134/134 internal-PASS
  invariant currently relies on operator memory).
