"""Promote a reviewed paper-triage brief into the research repo.

Parses a brief at ~/topo-confidence/research-graph/briefs/triage-<date>-<arxiv-id>.md
(or the legacy flat path triage-<date>-<arxiv-id>.md) and:

  1. Extracts YAML blocks from "## Proposed FutureExperiments" → MERGE
     :FutureExperiment nodes (with DEPENDS_ON_FINDING / WOULD_UPDATE /
     TRIGGERED_BY edges).
  2. Inserts H-N blocks from "## Proposed HYPOTHESES.md additions" into
     HYPOTHESES.md before the "Historical / abandoned" section. Replaces
     existing H-N blocks in --update-existing mode.
  3. Appends or updates "## Proposed PAPER_INDEX.md classification" content
     in PAPER_INDEX.md, with the 5 deep-extraction subsections folded in as
     `### <subsection>` subheadings (Methodologies / Approaches / Datasets /
     Implementation / Replicable intermediates / Cross-paper signals).
  4. Writes (:Method)-[:USED_IN]->(:Paper) and
     (:Dataset)-[:USED_IN]->(:Paper) graph edges so cross-paper method/data
     queries become possible.
  5. Calls bridge.py resolve for the source arxiv id (and any extra paper
     stubs declared via triggered-by).
  6. Refuses if "## New claims" is non-empty AND validate_claims.py hasn't
     been edited since the brief was last modified — protects the claims
     invariant.
  7. Calls generate_next_experiments.py to refresh NEXT_EXPERIMENTS.md.
  8. Sets :Paper {status:'graphed'} for the arxiv id.

Usage:
    python promote_brief.py briefs/triage-2026-04-29-2604.22271.md
    python promote_brief.py briefs/triage-2026-04-29-2604.22271.md --dry-run
    python promote_brief.py briefs/triage-2026-04-29-2604.22271.md --update-existing

`--update-existing` is for re-promoting briefs where HYPOTHESES / FE / PAPER_INDEX
entries already exist (e.g., the 6 backfill papers from the 2026-04-28 Desktop
session). It replaces existing H-N blocks and PAPER_INDEX entries in place;
without it, conflicts raise.

Brief filename convention (mandatory): triage-YYYY-MM-DD-<arxiv-id>.md
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

BRIEF_FILENAME_RE = re.compile(r"^triage-(\d{4}-\d{2}-\d{2})-(.+)\.md$")
YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)
H_N_HEADER_RE = re.compile(r"^### H-\d+:", re.MULTILINE)

# Top-level section headings the brief is *required* to use.
# Order matters — parser uses these as anchors so embedded `##` lines inside
# section bodies (e.g. PAPER_INDEX entries that start with `## arxiv-id —`)
# are captured as content, not split as new sections.
KNOWN_SECTIONS = [
    "## Refutations",
    "## Direct connections to F-N / H-N",
    "## Methodologies extracted",
    "## Approaches & framings",
    "## Datasets & benchmarks",
    "## Implementation details worth capturing",
    "## Replicable intermediates",
    "## Cross-paper signals",
    "## Proposed FutureExperiments",
    "## Proposed HYPOTHESES.md additions",
    "## Proposed PAPER_INDEX.md classification",
    "## New claims",
    "## Sources consulted",
]

# Subheading order for folding deep-extraction sections into the PAPER_INDEX entry.
DEEP_SUBSECTIONS = [
    ("## Methodologies extracted", "### Methodologies extracted"),
    ("## Approaches & framings", "### Approaches & framings"),
    ("## Datasets & benchmarks", "### Datasets & benchmarks"),
    ("## Implementation details worth capturing",
     "### Implementation details worth capturing"),
    ("## Replicable intermediates", "### Replicable intermediates"),
    ("## Cross-paper signals", "### Cross-paper signals"),
]

# Pattern: `- **<name>** — ...` for parsing :Method / :Dataset names.
NAMED_BULLET_RE = re.compile(r"^\s*-\s*\*\*([^*]+)\*\*", re.MULTILINE)


def _is_empty_section(body: str) -> bool:
    """A section is considered 'empty' if it's blank or starts with a 'none' marker.

    Briefs are instructed to write 'none', 'none extracted', or 'none worth flagging'
    rather than inventing filler. We detect those so empty sections don't leak into
    PAPER_INDEX or graph nodes.
    """
    stripped = body.strip().lower()
    if not stripped:
        return True
    first_line = stripped.splitlines()[0].strip("-* ").strip()
    return first_line.startswith("none")


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------

def parse_brief(path: Path) -> dict[str, Any]:
    text = path.read_text()

    section_offsets: list[tuple[str, int]] = []
    for heading in KNOWN_SECTIONS:
        idx = text.find("\n" + heading + "\n")
        if idx == -1 and text.startswith(heading + "\n"):
            idx = 0
        elif idx != -1:
            idx += 1
        if idx != -1:
            section_offsets.append((heading, idx))
    section_offsets.sort(key=lambda x: x[1])

    sections: dict[str, str] = {}
    for i, (heading, start) in enumerate(section_offsets):
        body_start = start + len(heading) + 1
        body_end = section_offsets[i + 1][1] if i + 1 < len(section_offsets) else len(text)
        sections[heading] = text[body_start:body_end].strip()

    parsed: dict[str, Any] = {
        "raw": text,
        "sections": sections,
        "future_experiments": [],
        "hypotheses_blocks": [],
        "paper_index_entry": "",
        "new_claims_lines": [],
        "method_names": [],
        "dataset_names": [],
        "deep_subsections": {},  # map subheading -> body (only non-empty)
    }

    for section_key, sub_heading in DEEP_SUBSECTIONS:
        body = sections.get(section_key, "")
        if _is_empty_section(body):
            continue
        parsed["deep_subsections"][sub_heading] = body.strip()
        if section_key == "## Methodologies extracted":
            parsed["method_names"] = [
                m.group(1).strip()
                for m in NAMED_BULLET_RE.finditer(body)
            ]
        elif section_key == "## Datasets & benchmarks":
            parsed["dataset_names"] = [
                m.group(1).strip()
                for m in NAMED_BULLET_RE.finditer(body)
            ]

    fe_section = sections.get("## Proposed FutureExperiments", "")
    for yaml_match in YAML_BLOCK_RE.finditer(fe_section):
        try:
            data = yaml.safe_load(yaml_match.group(1))
        except yaml.YAMLError as e:
            raise SystemExit(f"YAML parse error in brief {path}: {e}")
        if isinstance(data, list):
            parsed["future_experiments"].extend(data)
        elif isinstance(data, dict):
            parsed["future_experiments"].append(data)

    hyp_section = sections.get("## Proposed HYPOTHESES.md additions", "")
    h_starts = [m.start() for m in H_N_HEADER_RE.finditer(hyp_section)]
    for i, start in enumerate(h_starts):
        end = h_starts[i + 1] if i + 1 < len(h_starts) else len(hyp_section)
        block = hyp_section[start:end].strip()
        if block:
            parsed["hypotheses_blocks"].append(block)

    parsed["paper_index_entry"] = sections.get(
        "## Proposed PAPER_INDEX.md classification", ""
    ).strip()

    claims_section = sections.get("## New claims", "")
    parsed["new_claims_lines"] = [
        line.strip("- ").strip()
        for line in claims_section.splitlines()
        if line.strip() and line.strip().startswith("-")
    ]

    return parsed


def parse_filename(path: Path) -> tuple[str, str]:
    m = BRIEF_FILENAME_RE.match(path.name)
    if not m:
        raise SystemExit(
            f"Brief filename '{path.name}' does not match required pattern "
            "triage-YYYY-MM-DD-<arxiv-id>.md"
        )
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# ID renumbering (collision-safe contiguous allocation at promote time)
# ---------------------------------------------------------------------------

# A "promote-time ID" is any H-N or P{N}(_h100)?-FE{N} the brief picked.
# Bulk-triage briefs are written in parallel, so they all jump to high
# numbers ("collision-safe") to avoid stomping each other. The graph wants
# contiguous numbering from current Next IDs, so we remap everything at
# promote time.
ID_TOKEN_RE = re.compile(r"\b(?:H-\d+|P\d+(?:_h100)?-FE\d+)\b")
FE_ID_RE = re.compile(r"^(P\d+(?:_h100)?)-FE(\d+)$")


def _max_fe_for_pathway(driver, pathway: str) -> int:
    """Return the largest existing FE-N for `pathway` (0 if none)."""
    with driver.session() as s:
        res = s.run(
            """
            MATCH (fe:FutureExperiment)
            WHERE fe.id STARTS WITH $prefix
            WITH fe.id AS id, $prefix AS prefix
            WITH toInteger(substring(id, size(prefix))) AS n
            RETURN max(n) AS max_n
            """,
            prefix=f"{pathway}-FE",
        ).single()
    return int(res["max_n"]) if res and res["max_n"] is not None else 0


def renumber(parsed: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Remap brief-declared H-N / P{X}-FE-N to contiguous IDs from current
    Next-ID state. Mutates `parsed` in place; returns the remap dicts."""
    # H-N remap: contiguous from HYPOTHESES.md "Next ID: H-N".
    hyp_text = (REPO / "HYPOTHESES.md").read_text()
    m = re.search(r"Next ID: H-(\d+)", hyp_text)
    if not m:
        raise SystemExit('No "Next ID: H-N" line in HYPOTHESES.md.')
    next_h = int(m.group(1))

    h_remap: dict[str, str] = {}
    for i, block in enumerate(parsed["hypotheses_blocks"]):
        hm = re.match(r"### H-(\d+):", block)
        if not hm:
            continue
        old = f"H-{hm.group(1)}"
        new = f"H-{next_h + i}"
        if old != new:
            h_remap[old] = new

    # FE-N remap: per-pathway contiguous from current graph max+1.
    fe_remap: dict[str, str] = {}
    pathways: dict[str, int] = {}
    if parsed["future_experiments"]:
        drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
        try:
            seen_pathways = sorted({fe["pathway"] for fe in parsed["future_experiments"]})
            for pw in seen_pathways:
                pathways[pw] = _max_fe_for_pathway(drv, pw) + 1
        finally:
            drv.close()
        for fe in parsed["future_experiments"]:
            old = fe["id"]
            fm = FE_ID_RE.match(old)
            if not fm:
                continue
            pw = fm.group(1)
            new = f"{pw}-FE{pathways[pw]}"
            pathways[pw] += 1
            if old != new:
                fe_remap[old] = new

    full = {**h_remap, **fe_remap}
    if not full:
        return {"h_remap": {}, "fe_remap": {}}

    # Substitute longest keys first so "H-72" doesn't shadow "H-723".
    sorted_keys = sorted(full.keys(), key=len, reverse=True)

    def remap_text(text: str) -> str:
        for k in sorted_keys:
            v = full[k]
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![0-9])", v, text)
        return text

    # Mutate FE blocks: id, prose fields, blocked-by-experiment cross-refs.
    for fe in parsed["future_experiments"]:
        if fe["id"] in fe_remap:
            fe["id"] = fe_remap[fe["id"]]
        for field in ("description", "rationale", "trigger"):
            v = fe.get(field)
            if isinstance(v, str):
                fe[field] = remap_text(v)
        bbe = fe.get("blocked-by-experiment")
        if isinstance(bbe, list):
            fe["blocked-by-experiment"] = [fe_remap.get(x, x) for x in bbe]

    # Mutate H-N blocks (header + body), PAPER_INDEX entry, deep subsections,
    # and new-claim lines so cross-references stay aligned.
    parsed["hypotheses_blocks"] = [remap_text(b) for b in parsed["hypotheses_blocks"]]
    parsed["paper_index_entry"] = remap_text(parsed["paper_index_entry"])
    parsed["deep_subsections"] = {
        k: remap_text(v) for k, v in parsed["deep_subsections"].items()
    }
    parsed["new_claims_lines"] = [remap_text(c) for c in parsed["new_claims_lines"]]

    return {"h_remap": h_remap, "fe_remap": fe_remap}


# ---------------------------------------------------------------------------
# Validate-claims gate (protects the claims invariant: 91 internal PASS,
# 41 external REGISTERED, 3 PENDING_FE — see CLAUDE.md)
# ---------------------------------------------------------------------------

def assert_claims_gate(parsed: dict[str, Any], brief_path: Path) -> None:
    if not parsed["new_claims_lines"]:
        print("  no new claims declared — gate skipped")
        return
    validate_path = REPO / "validate_claims.py"
    if not validate_path.exists():
        raise SystemExit(f"validate_claims.py not found at {validate_path} — abort.")
    if validate_path.stat().st_mtime <= brief_path.stat().st_mtime:
        print("  CLAIMS GATE: brief declares new quantitative claims:")
        for c in parsed["new_claims_lines"]:
            print(f"    - {c}")
        raise SystemExit(
            "validate_claims.py has NOT been modified since the brief — refusing to "
            "promote. Add a Claim entry per declared new claim, run "
            "`python validate_claims.py`, then re-run promote_brief.py."
        )
    print(f"  ok: {len(parsed['new_claims_lines'])} new claims, validate_claims.py updated since brief")


# ---------------------------------------------------------------------------
# FutureExperiment writes (direct Neo4j, mirrors add_future_experiment.py)
# ---------------------------------------------------------------------------

VALID_STATUS = {"READY", "TRIGGERED", "BLOCKED", "COMPLETED", "ABANDONED",
                "MOOTED", "ANSWERED"}
VALID_PRIORITY = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def write_future_experiments(fes: list[dict[str, Any]], dry_run: bool) -> list[str]:
    if not fes:
        print("  no FutureExperiment YAML blocks in brief")
        return []

    arxiv_ids: set[str] = set()
    for fe in fes:
        for required in ("id", "pathway", "description", "roi"):
            if required not in fe:
                raise SystemExit(f"FE block missing required field '{required}': {fe}")
        if fe.get("status", "READY") not in VALID_STATUS:
            raise SystemExit(f"Invalid status {fe['status']} in FE {fe['id']}")
        if fe.get("priority", "MEDIUM") not in VALID_PRIORITY:
            raise SystemExit(f"Invalid priority {fe['priority']} in FE {fe['id']}")
        roi = int(fe["roi"])
        if not (1 <= roi <= 10):
            raise SystemExit(f"roi out of [1,10] in FE {fe['id']}: {roi}")
        for arx in fe.get("triggered-by") or []:
            arxiv_ids.add(str(arx))

    if dry_run:
        for fe in fes:
            print(f"  Would MERGE FutureExperiment {fe['id']} (priority={fe.get('priority','MEDIUM')}, ROI={fe['roi']})")
        return sorted(arxiv_ids)

    from datetime import date as _date
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    today = _date.today().isoformat()
    try:
        with drv.session() as s:
            for fe in fes:
                s.run(
                    """
                    MERGE (fe:FutureExperiment {id: $id})
                    SET fe.pathway_id = $pathway_id,
                        fe.description = $description,
                        fe.rationale = $rationale,
                        fe.trigger = $trigger,
                        fe.status = $status,
                        fe.blocked_by = $blocked_by,
                        fe.priority = $priority,
                        fe.estimated_cost = $cost,
                        fe.roi_score = $roi,
                        fe.created_date = coalesce(fe.created_date, $today)
                    """,
                    id=fe["id"],
                    pathway_id=fe["pathway"],
                    description=fe["description"],
                    rationale=fe.get("rationale", ""),
                    trigger=fe.get("trigger", ""),
                    status=fe.get("status", "READY"),
                    blocked_by=fe.get("blocked-by"),
                    priority=fe.get("priority", "MEDIUM"),
                    cost=fe.get("cost", ""),
                    roi=int(fe["roi"]),
                    today=today,
                )
                s.run(
                    "MATCH (p:Pathway {id: $pid}), (fe:FutureExperiment {id: $fid}) MERGE (p)-[:HAS_FUTURE_EXPERIMENT]->(fe)",
                    pid=fe["pathway"], fid=fe["id"],
                )
                for fnd in (fe.get("depends-on") or []):
                    s.run(
                        "MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $fnd}) MERGE (fe)-[:DEPENDS_ON_FINDING]->(f)",
                        fid=fe["id"], fnd=fnd,
                    )
                for fnd in (fe.get("would-update") or []):
                    s.run(
                        "MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $fnd}) MERGE (fe)-[:WOULD_UPDATE]->(f)",
                        fid=fe["id"], fnd=fnd,
                    )
                for arx in (fe.get("triggered-by") or []):
                    s.run("MERGE (:Paper {arxiv_id: $a})", a=arx)
                    s.run(
                        "MATCH (fe:FutureExperiment {id: $fid}), (p:Paper {arxiv_id: $a}) MERGE (fe)-[:TRIGGERED_BY]->(p)",
                        fid=fe["id"], a=arx,
                    )
                for other in (fe.get("blocked-by-experiment") or []):
                    s.run(
                        "MATCH (a:FutureExperiment {id: $a}), (b:FutureExperiment {id: $b}) MERGE (a)-[:BLOCKED_BY_EXPERIMENT]->(b)",
                        a=fe["id"], b=other,
                    )
                born_mooted = None
                for premise in (fe.get("relies-on") or []):
                    rec = s.run(
                        """
                        MATCH (fe:FutureExperiment {id: $fid}), (pr:Premise {id: $pid})
                        MERGE (fe)-[:RELIES_ON]->(pr)
                        RETURN pr.status AS status, pr.refuted_by AS refuted_by,
                               pr.reason AS reason
                        """,
                        fid=fe["id"], pid=premise,
                    ).single()
                    if rec is None:
                        print(f"  WARNING: FE {fe['id']} relies-on unknown "
                              f"premise {premise!r} — edge skipped. Known ids: "
                              "python premises.py list")
                    elif rec["status"] == "REFUTED":
                        born_mooted = (premise, rec["refuted_by"], rec["reason"])
                if born_mooted and fe.get("status", "READY") not in ("COMPLETED", "ABANDONED"):
                    premise, refuted_by, reason = born_mooted
                    s.run(
                        """
                        MATCH (fe:FutureExperiment {id: $fid})
                        SET fe.status = 'MOOTED',
                            fe.outcome = $outcome,
                            fe.closed_by = $by,
                            fe.completed_date = $today
                        WITH fe
                        MATCH (pr:Premise {id: $pid})
                        MERGE (fe)-[r:MOOTED_BY]->(pr)
                        SET r.reason = $outcome, r.date = $today
                        """,
                        fid=fe["id"], pid=premise, by=refuted_by or premise,
                        outcome=f"Born MOOTED — relies on refuted premise "
                                f"'{premise}' ({reason})",
                        today=today,
                    )
                    print(f"  ok FutureExperiment {fe['id']} — born MOOTED "
                          f"(premise '{premise}' already refuted)")
                    continue
                print(f"  ok FutureExperiment {fe['id']}")
    finally:
        drv.close()
    return sorted(arxiv_ids)


# ---------------------------------------------------------------------------
# Markdown insertions
# ---------------------------------------------------------------------------

def _h_block_bounds(text: str, h_n: int) -> tuple[int, int] | None:
    """Find [start, end) of an existing `### H-<n>:` block in HYPOTHESES.md.

    A block ends at the next `### H-` header, the `## Historical` section, or EOF.
    """
    m = re.search(rf"^### H-{h_n}:", text, re.MULTILINE)
    if not m:
        return None
    start = m.start()
    next_h = re.search(r"^### H-\d+:", text[m.end():], re.MULTILINE)
    next_section = re.search(r"^## ", text[m.end():], re.MULTILINE)
    candidates = [c.start() + m.end() for c in [next_h, next_section] if c]
    end = min(candidates) if candidates else len(text)
    return (start, end)


def insert_hypotheses(blocks: list[str], dry_run: bool, update_existing: bool) -> None:
    if not blocks:
        print("  no H-N blocks in brief")
        return
    path = REPO / "HYPOTHESES.md"
    text = path.read_text()

    next_id_match = re.search(r"Next ID: H-(\d+)", text)
    if not next_id_match:
        raise SystemExit('No "Next ID: H-N" line found in HYPOTHESES.md.')
    next_id = int(next_id_match.group(1))

    declared_ids = []
    for block in blocks:
        m = re.match(r"### H-(\d+):", block)
        if not m:
            raise SystemExit(f"Block lacks ### H-N: header:\n{block[:200]}")
        declared_ids.append(int(m.group(1)))

    new_blocks: list[str] = []
    update_blocks: list[tuple[int, str]] = []
    for h_n, block in zip(declared_ids, blocks):
        if h_n < next_id:
            if not update_existing:
                raise SystemExit(
                    f"H-{h_n} already exists in HYPOTHESES.md and --update-existing "
                    f"was not passed. Either drop the block from the brief or re-run "
                    f"with --update-existing to replace it in place."
                )
            update_blocks.append((h_n, block))
        else:
            new_blocks.append(block)

    expected_new = list(range(next_id, next_id + len(new_blocks)))
    actual_new = [int(re.match(r"### H-(\d+):", b).group(1)) for b in new_blocks]  # type: ignore[union-attr]
    if actual_new != expected_new:
        raise SystemExit(
            f"New H-N IDs {actual_new} do not match contiguous sequence starting "
            f"from current Next ID H-{next_id}: expected {expected_new}. "
            f"Renumber brief or sync HYPOTHESES.md."
        )

    new_text = text

    # 1. Replace existing blocks in place (--update-existing only).
    for h_n, block in update_blocks:
        bounds = _h_block_bounds(new_text, h_n)
        if not bounds:
            raise SystemExit(f"Could not locate H-{h_n} block to update.")
        start, end = bounds
        # Keep the trailing whitespace/newlines of the original block.
        trailing = ""
        tail = new_text[start:end]
        if tail.endswith("\n\n"):
            trailing = "\n\n"
        elif tail.endswith("\n"):
            trailing = "\n"
        new_text = new_text[:start] + block.strip() + trailing + new_text[end:]

    # 2. Insert new blocks before the Historical anchor.
    if new_blocks:
        new_next = next_id + len(new_blocks)
        hist_anchor = "## Historical / abandoned"
        if hist_anchor not in new_text:
            raise SystemExit("Historical-section anchor not found.")
        insertion = "\n" + "\n\n".join(new_blocks) + "\n\n---\n\n"
        new_text = new_text.replace(
            f"Next ID: H-{next_id}.",
            f"Next ID: H-{new_next}.",
        )
        new_text = new_text.replace(hist_anchor, insertion + hist_anchor)
    else:
        new_next = next_id

    if dry_run:
        if update_blocks:
            print(f"  Would update {len(update_blocks)} existing H-N block(s) in place: "
                  f"{[h for h, _ in update_blocks]}")
        if new_blocks:
            print(f"  Would insert {len(new_blocks)} new H-N block(s): {actual_new}")
            print(f"  Would update Next ID: H-{next_id} -> H-{new_next}")
        if not update_blocks and not new_blocks:
            print("  (nothing to do)")
    else:
        path.write_text(new_text)
        if update_blocks:
            print(f"  ok HYPOTHESES.md replaced {len(update_blocks)} block(s) "
                  f"in place: {[h for h, _ in update_blocks]}")
        if new_blocks:
            print(f"  ok HYPOTHESES.md appended {len(new_blocks)} new block(s), "
                  f"Next ID -> H-{new_next}")


def _paper_index_entry_bounds(text: str, arxiv_id: str) -> tuple[int, int] | None:
    """Find [start, end) of an existing PAPER_INDEX entry for arxiv_id.

    PAPER_INDEX entries start with `## <arxiv_id> — ...`. The entry ends at the
    next `## ` heading or EOF. Falls back to a `## ` heading whose body mentions
    the arxiv id within the first line if the canonical pattern doesn't match.
    """
    pattern = rf"^## {re.escape(arxiv_id)}(\s|—|-)"
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        # Fallback — match any heading that contains the arxiv id.
        m = re.search(rf"^## .*{re.escape(arxiv_id)}", text, re.MULTILINE)
        if not m:
            return None
    start = m.start()
    next_h = re.search(r"^## ", text[m.end():], re.MULTILINE)
    end = next_h.start() + m.end() if next_h else len(text)
    return (start, end)


def _build_paper_index_block(entry: str, deep_subsections: dict[str, str]) -> str:
    """Combine the brief's PAPER_INDEX entry with the deep extraction subheadings."""
    parts = [entry.strip()]
    for _section_key, sub_heading in DEEP_SUBSECTIONS:
        body = deep_subsections.get(sub_heading)
        if not body:
            continue
        parts.append(f"\n{sub_heading}\n\n{body.strip()}")
    return "\n".join(parts).rstrip() + "\n"


def append_paper_index(
    entry: str,
    deep_subsections: dict[str, str],
    arxiv_id: str,
    dry_run: bool,
    update_existing: bool,
) -> None:
    if not entry:
        print("  no PAPER_INDEX.md entry in brief")
        return
    path = REPO / "PAPER_INDEX.md"
    text = path.read_text()
    block = _build_paper_index_block(entry, deep_subsections)

    bounds = _paper_index_entry_bounds(text, arxiv_id)
    extra_subs = list(deep_subsections.keys())

    if bounds:
        if not update_existing:
            raise SystemExit(
                f"PAPER_INDEX.md already has an entry for {arxiv_id}. "
                f"Re-run with --update-existing to replace it in place."
            )
        if dry_run:
            print(f"  Would replace existing PAPER_INDEX.md entry for {arxiv_id} "
                  f"({len(extra_subs)} deep subsection(s) folded in)")
            return
        start, end = bounds
        new_text = text[:start] + block.rstrip() + "\n\n" + text[end:]
        path.write_text(new_text.rstrip() + "\n")
        print(f"  ok PAPER_INDEX.md entry for {arxiv_id} replaced in place "
              f"(+{len(extra_subs)} deep subsection(s))")
        return

    if dry_run:
        print(f"  Would append PAPER_INDEX.md entry for {arxiv_id} "
              f"({block.count(chr(10))+1} lines, {len(extra_subs)} deep subsection(s))")
        return
    path.write_text(text.rstrip() + "\n\n" + block)
    print(f"  ok PAPER_INDEX.md appended (+{len(extra_subs)} deep subsection(s))")


# ---------------------------------------------------------------------------
# Bridge.py resolve and status='graphed'
# ---------------------------------------------------------------------------

def resolve_papers(arxiv_ids: list[str], primary_arxiv: str, dry_run: bool) -> None:
    all_ids = sorted(set(arxiv_ids + [primary_arxiv]))
    for arx in all_ids:
        if dry_run:
            print(f"  Would call bridge.py resolve {arx}")
            continue
        res = subprocess.run(
            [sys.executable, str(ROOT / "bridge.py"), "resolve", arx],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            print(f"  bridge.py resolve {arx} non-zero (continuing): {res.stderr.strip()}")
        else:
            print(f"  ok bridge.py resolve {arx}")


def set_paper_graphed(arxiv_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  Would set Paper {arxiv_id} status='graphed'")
        return
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        s.run(
            "MERGE (p:Paper {arxiv_id: $a}) SET p.status = 'graphed'",
            a=arxiv_id,
        )
    drv.close()
    print(f"  ok Paper {arxiv_id} status='graphed'")


def write_method_dataset_edges(
    arxiv_id: str,
    method_names: list[str],
    dataset_names: list[str],
    dry_run: bool,
) -> None:
    """MERGE :Method / :Dataset nodes and connect to the source paper.

    Empty lists are no-ops. Names are normalized to lowercase-trimmed for the
    MERGE key, but the original casing is kept in `display_name` for the UI.
    """
    methods = [m for m in (n.strip() for n in method_names) if m]
    datasets = [d for d in (n.strip() for n in dataset_names) if d]
    if not methods and not datasets:
        print("  no methods/datasets to write")
        return
    if dry_run:
        if methods:
            print(f"  Would MERGE {len(methods)} :Method node(s): {methods}")
        if datasets:
            print(f"  Would MERGE {len(datasets)} :Dataset node(s): {datasets}")
        return
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        for name in methods:
            key = name.lower()
            s.run(
                """
                MERGE (m:Method {key: $key})
                  ON CREATE SET m.display_name = $name
                  ON MATCH SET m.display_name = coalesce(m.display_name, $name)
                """,
                key=key, name=name,
            )
            s.run(
                """
                MATCH (m:Method {key: $key}), (p:Paper {arxiv_id: $a})
                MERGE (m)-[:USED_IN]->(p)
                """,
                key=key, a=arxiv_id,
            )
        for name in datasets:
            key = name.lower()
            s.run(
                """
                MERGE (d:Dataset {key: $key})
                  ON CREATE SET d.display_name = $name
                  ON MATCH SET d.display_name = coalesce(d.display_name, $name)
                """,
                key=key, name=name,
            )
            s.run(
                """
                MATCH (d:Dataset {key: $key}), (p:Paper {arxiv_id: $a})
                MERGE (d)-[:USED_IN]->(p)
                """,
                key=key, a=arxiv_id,
            )
    drv.close()
    print(f"  ok wrote {len(methods)} :Method / {len(datasets)} :Dataset edge(s)")


def regen_next_experiments(dry_run: bool) -> None:
    if dry_run:
        print("  Would call generate_next_experiments.py")
        return
    res = subprocess.run(
        [sys.executable, str(ROOT / "generate_next_experiments.py")],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("  generate_next_experiments.py FAILED:")
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(res.returncode)
    print(f"  ok {res.stdout.strip().splitlines()[-1] if res.stdout.strip() else 'NEXT_EXPERIMENTS.md regenerated'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("brief", help="Path to triage brief markdown file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing")
    p.add_argument("--update-existing", action="store_true",
                   help="Re-promote a brief whose H-N / FE / PAPER_INDEX entries "
                        "already exist. Replaces them in place idempotently.")
    p.add_argument("--no-renumber", action="store_true",
                   help="Skip the renumber-on-promote step (use brief's exact "
                        "H-N / FE-N IDs verbatim). Default: renumber to "
                        "contiguous IDs from current Next-ID state.")
    args = p.parse_args()

    brief_path = Path(args.brief).expanduser().resolve()
    if not brief_path.exists():
        raise SystemExit(f"Brief not found: {brief_path}")

    date_str, arxiv_id = parse_filename(brief_path)
    mode_tags = ["DRY-RUN" if args.dry_run else "APPLY"]
    if args.update_existing:
        mode_tags.append("UPDATE-EXISTING")
    print(f"=== promote_brief: {brief_path.name} ===")
    print(f"date={date_str}  arxiv_id={arxiv_id}  mode={' / '.join(mode_tags)}")

    parsed = parse_brief(brief_path)
    print(f"\nparsed: {len(parsed['future_experiments'])} FE blocks, "
          f"{len(parsed['hypotheses_blocks'])} H-N blocks, "
          f"paper_index_entry={'yes' if parsed['paper_index_entry'] else 'no'}, "
          f"new_claims={len(parsed['new_claims_lines'])}, "
          f"deep subsections={len(parsed['deep_subsections'])}, "
          f"methods={len(parsed['method_names'])}, "
          f"datasets={len(parsed['dataset_names'])}")

    if args.no_renumber:
        print("\nstep 0: renumber (skipped — --no-renumber)")
    else:
        print("\nstep 0: renumber to contiguous IDs from current Next-ID state")
        remaps = renumber(parsed)
        if remaps["h_remap"]:
            for old, new in sorted(remaps["h_remap"].items(),
                                   key=lambda kv: int(kv[0].split("-")[1])):
                print(f"  H remap: {old} -> {new}")
        if remaps["fe_remap"]:
            for old, new in sorted(remaps["fe_remap"].items()):
                print(f"  FE remap: {old} -> {new}")
        if not remaps["h_remap"] and not remaps["fe_remap"]:
            print("  (no IDs needed renumbering)")

    print("\nstep 1: validate-claims gate")
    assert_claims_gate(parsed, brief_path)

    print("\nstep 2: FutureExperiment writes")
    arxiv_from_fes = write_future_experiments(parsed["future_experiments"], args.dry_run)

    print("\nstep 3: HYPOTHESES.md")
    insert_hypotheses(parsed["hypotheses_blocks"], args.dry_run, args.update_existing)

    print("\nstep 4: PAPER_INDEX.md (with deep-extraction subheadings)")
    append_paper_index(
        parsed["paper_index_entry"],
        parsed["deep_subsections"],
        arxiv_id,
        args.dry_run,
        args.update_existing,
    )

    print("\nstep 5: :Method / :Dataset graph edges")
    write_method_dataset_edges(
        arxiv_id,
        parsed["method_names"],
        parsed["dataset_names"],
        args.dry_run,
    )

    print("\nstep 6: bridge.py resolve (link-forge enrichment)")
    resolve_papers(arxiv_from_fes, arxiv_id, args.dry_run)

    print("\nstep 7: regenerate NEXT_EXPERIMENTS.md")
    regen_next_experiments(args.dry_run)

    print("\nstep 8: set Paper status='graphed'")
    set_paper_graphed(arxiv_id, args.dry_run)

    print("\nstep 8b: embed Paper for vector search")
    if not args.dry_run:
        try:
            from backfill_embeddings import embed_node
            drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
            with drv.session() as s:
                row = s.run(
                    "MATCH (p:Paper {arxiv_id: $a}) "
                    "RETURN p.title AS title, p.relevance_note AS note",
                    a=arxiv_id,
                ).single()
            drv.close()
            title = (row["title"] or "") if row else ""
            note = (row["note"] or "") if row else ""
            text = f"{title} {note}".strip()
            if text:
                embed_node("Paper", arxiv_id, text)
                print(f"  ok embedded {arxiv_id} ({len(text)} chars)")
            else:
                print(f"  skipped — no text for {arxiv_id}")
        except Exception as e:
            print(f"  WARNING: embedding skipped for {arxiv_id}: {e}")
    else:
        print(f"  Would embed Paper {arxiv_id}")

    if not args.dry_run:
        try:
            from datetime import datetime as _dt
            from pipeline.publisher import publish, set_research_field, init_redis_publisher, close_redis_publisher
            init_redis_publisher()
            publish("topoconf:research:triaged", {
                "arxiv_id": arxiv_id,
                "fe_count": len(parsed["future_experiments"]),
                "hypothesis_count": len(parsed["hypotheses_blocks"]),
            })
            set_research_field(arxiv_id, {
                "status": "graphed",
                "brief_path": str(brief_path),
                "fe_count": str(len(parsed["future_experiments"])),
                "hypothesis_count": str(len(parsed["hypotheses_blocks"])),
                "triaged_at": _dt.now().isoformat(),
            })
            close_redis_publisher()
        except Exception:
            pass

    print()
    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"DONE. Brief is now promoted. Consider:")
        try:
            rel = brief_path.relative_to(REPO)
            print(f"  git add {rel}")
        except ValueError:
            print(f"  git add {brief_path}")


if __name__ == "__main__":
    main()
