"""Promote a completed-experiment result brief into the research repo.

Soup-to-nuts mirror of promote_brief.py for the *internal-experiment* side.
Author writes a structured brief at
~/topo-confidence/research-graph/briefs/result-YYYY-MM-DD-<fe-id>.md
(fe-id is the FutureExperiment id, e.g. P11_h100-FE719). promote_result.py:

  1. Parses required sections (refuses if any are missing or filler-only).
  2. Verifies result JSON exists and contains the keys the brief promises.
  3. Optionally re-runs the regen command and aborts on FAIL.
  4. Claims gate: validate_claims.py mtime > brief mtime AND PASSes.
  5. Refuses if `git status` is dirty outside expected files.
  6. Updates Neo4j: FutureExperiment status + outcome (via update_status.py),
     Finding status / strength / evidence in place, PRODUCED edges from
     :Experiment to :Finding.
  7. Appends EXPERIMENT_LOG.md entry, bumps "Next ID: EXP-N".
  8. Replaces "## Last experiment completed" section in STATE.md.
  9. Replaces existing F-N blocks in FINDINGS.md in place; appends new
     F-N blocks before the "## Honorable mentions" anchor and bumps
     "Next ID: F-N".
 10. Regenerates NEXT_EXPERIMENTS.md via generate_next_experiments.py.

PERSPECTIVES.md is intentionally never touched — that's judgment-only prose.

Usage:
    python promote_result.py briefs/result-2026-04-30-P11_h100-FE719.md
    python promote_result.py briefs/result-... --dry-run
    python promote_result.py briefs/result-... --skip-regen
    python promote_result.py briefs/result-... --skip-git-check

Brief filename convention (mandatory): result-YYYY-MM-DD-<fe-id>.md
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

BRIEF_FILENAME_RE = re.compile(r"^result-(\d{4}-\d{2}-\d{2})-(.+)\.md$")
YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)
F_N_HEADER_RE = re.compile(r"^### F-\d+:", re.MULTILINE)
EXP_HEADER_RE = re.compile(r"^## EXP-(\d+):", re.MULTILINE)

# Required top-level sections — the parser uses these as anchors.
KNOWN_SECTIONS = [
    "## FE",
    "## Result JSON verification",
    "## EXPERIMENT_LOG entry",
    "## STATE.md last-experiment update",
    "## FINDINGS.md updates",
    "## New claims",
    "## Sources",
]

REQUIRED_SECTIONS = [
    "## FE",
    "## EXPERIMENT_LOG entry",
    "## STATE.md last-experiment update",
]

VALID_FE_STATUS = {"COMPLETED", "ABANDONED"}
VALID_F_STRENGTH = {"STRONG", "MODERATE", "PRELIMINARY"}
VALID_F_STATUS = {"ACTIVE", "WEAKENED", "INVALIDATED", "SUPERSEDED"}


def _is_empty_section(body: str) -> bool:
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

    for required in REQUIRED_SECTIONS:
        body = sections.get(required, "")
        if _is_empty_section(body):
            raise SystemExit(
                f"Brief is missing required section '{required}' (or it is empty / "
                f"'none'). Cannot promote a result with no FE / EXPERIMENT_LOG / "
                f"STATE.md content."
            )

    fe_yaml = _parse_single_yaml(sections.get("## FE", ""), "## FE")
    for required in ("fe_id", "status", "outcome"):
        if required not in fe_yaml:
            raise SystemExit(f"## FE block missing required key '{required}'")
    if fe_yaml["status"] not in VALID_FE_STATUS:
        raise SystemExit(
            f"## FE status must be one of {sorted(VALID_FE_STATUS)}, got "
            f"{fe_yaml['status']!r}"
        )

    findings_blocks = _split_findings_blocks(
        sections.get("## FINDINGS.md updates", "")
    )
    new_finding_yaml = _maybe_parse_yaml_meta(findings_blocks)

    new_claims_lines = [
        line.strip("- ").strip()
        for line in sections.get("## New claims", "").splitlines()
        if line.strip().startswith("-")
    ]

    return {
        "raw": text,
        "sections": sections,
        "fe": fe_yaml,
        "experiment_log": sections.get("## EXPERIMENT_LOG entry", "").strip(),
        "state_last_experiment": sections.get(
            "## STATE.md last-experiment update", ""
        ).strip(),
        "findings_blocks": findings_blocks,
        "findings_meta": new_finding_yaml,
        "new_claims_lines": new_claims_lines,
        "result_verification": sections.get(
            "## Result JSON verification", ""
        ).strip(),
    }


def _parse_single_yaml(body: str, label: str) -> dict[str, Any]:
    matches = list(YAML_BLOCK_RE.finditer(body))
    if len(matches) != 1:
        raise SystemExit(
            f"{label} requires exactly one ```yaml block; found {len(matches)}."
        )
    try:
        data = yaml.safe_load(matches[0].group(1))
    except yaml.YAMLError as e:
        raise SystemExit(f"YAML parse error in {label}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"{label} YAML must be a mapping, got {type(data).__name__}")
    return data


def _split_findings_blocks(body: str) -> list[str]:
    """Split the FINDINGS.md updates section into individual `### F-N:` blocks."""
    if _is_empty_section(body):
        return []
    starts = [m.start() for m in F_N_HEADER_RE.finditer(body)]
    blocks: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(body)
        block = body[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def _maybe_parse_yaml_meta(blocks: list[str]) -> list[dict[str, Any]]:
    """Each finding block may begin with a fenced yaml that declares
    strength/status/add_evidence — extract it as structured metadata.
    """
    metas: list[dict[str, Any]] = []
    for block in blocks:
        m = re.match(r"^### F-(\d+):", block)
        if not m:
            raise SystemExit(f"FINDINGS block lacks ### F-N: header:\n{block[:200]}")
        f_n = int(m.group(1))
        meta: dict[str, Any] = {"f_n": f_n}
        ymatch = YAML_BLOCK_RE.search(block)
        if ymatch:
            try:
                data = yaml.safe_load(ymatch.group(1)) or {}
            except yaml.YAMLError as e:
                raise SystemExit(f"YAML parse error in F-{f_n} block: {e}")
            if data.get("strength") and data["strength"] not in VALID_F_STRENGTH:
                raise SystemExit(
                    f"F-{f_n} strength must be one of {sorted(VALID_F_STRENGTH)}"
                )
            if data.get("status") and data["status"] not in VALID_F_STATUS:
                raise SystemExit(
                    f"F-{f_n} status must be one of {sorted(VALID_F_STATUS)}"
                )
            meta.update(data)
        metas.append(meta)
    return metas


def parse_filename(path: Path) -> tuple[str, str]:
    m = BRIEF_FILENAME_RE.match(path.name)
    if not m:
        raise SystemExit(
            f"Brief filename {path.name!r} does not match required pattern "
            "result-YYYY-MM-DD-<fe-id>.md"
        )
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Result-JSON verification + regen
# ---------------------------------------------------------------------------

def verify_result_json(parsed: dict[str, Any], dry_run: bool) -> None:
    fe = parsed["fe"]
    rel = fe.get("result_json")
    if not rel:
        if fe["status"] == "COMPLETED":
            raise SystemExit(
                "## FE.result_json is required when status=COMPLETED. "
                "Set it to a path under the repo (e.g. pathway11_h100/.../results.json)."
            )
        print("  result_json not declared (status=ABANDONED) — skipped")
        return
    path = (REPO / rel).resolve()
    if not path.exists():
        raise SystemExit(f"result_json missing on disk: {path}")
    expected_keys = fe.get("result_json_keys") or []
    if expected_keys:
        try:
            import json
            data = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"result_json {path} is not valid JSON: {e}")
        flat_keys = _flatten_keys(data)
        missing = [k for k in expected_keys if k not in flat_keys]
        if missing:
            raise SystemExit(
                f"result_json {rel} is missing declared keys: {missing}. "
                f"Top-level keys present: {sorted(flat_keys)[:20]}{'…' if len(flat_keys) > 20 else ''}"
            )
        print(f"  ok result_json {rel} has all {len(expected_keys)} declared key(s)")
    else:
        print(f"  ok result_json {rel} exists ({path.stat().st_size:,} bytes)")


def _flatten_keys(data: Any, prefix: str = "") -> set[str]:
    """Return dotted-key set so brief can declare nested keys like 'cv.auroc_mean'."""
    keys: set[str] = set()
    if isinstance(data, dict):
        for k, v in data.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            keys.add(full)
            keys.add(str(k))
            keys |= _flatten_keys(v, full)
    return keys


def run_regen(parsed: dict[str, Any], dry_run: bool) -> None:
    cmd = parsed["fe"].get("regen_cmd")
    if not cmd:
        print("  regen_cmd not declared — skipped")
        return
    if dry_run:
        print(f"  Would run regen: {cmd}")
        return
    print(f"  running regen: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    tail = (res.stdout + res.stderr).strip().splitlines()[-5:]
    if res.returncode != 0:
        for line in tail:
            print(f"    {line}")
        raise SystemExit(
            f"regen_cmd failed (exit {res.returncode}). "
            f"Fix the regen and re-run promote_result."
        )
    for line in tail:
        print(f"    {line}")
    print("  ok regen passed")


# ---------------------------------------------------------------------------
# Claims gate (mirror promote_brief.py:322)
# ---------------------------------------------------------------------------

def assert_claims_gate(parsed: dict[str, Any], brief_path: Path, dry_run: bool) -> None:
    new_claims = parsed["new_claims_lines"]
    validate_path = REPO / "validate_claims.py"
    if not validate_path.exists():
        raise SystemExit(f"validate_claims.py not found at {validate_path} — abort.")

    if new_claims:
        if validate_path.stat().st_mtime <= brief_path.stat().st_mtime:
            print("  CLAIMS GATE: brief declares new quantitative claims:")
            for c in new_claims:
                print(f"    - {c}")
            raise SystemExit(
                "validate_claims.py has NOT been modified since the brief — refusing "
                "to promote. Add a Claim entry per declared new claim, run "
                "`python validate_claims.py`, then re-run promote_result.py."
            )
        print(f"  ok: {len(new_claims)} new claims, validate_claims.py updated since brief")

    if dry_run:
        print("  Would run validate_claims.py")
        return
    print("  running validate_claims.py …")
    res = subprocess.run(
        [sys.executable, str(validate_path), "--no-regen"],
        cwd=REPO, capture_output=True, text=True,
    )
    last_line = (res.stdout + res.stderr).strip().splitlines()[-1:]
    if res.returncode != 0:
        for line in last_line:
            print(f"    {line}")
        raise SystemExit(
            f"validate_claims.py failed (exit {res.returncode}). Fix the failing "
            f"Claim entry before promoting."
        )
    for line in last_line:
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Git cleanliness
# ---------------------------------------------------------------------------

def assert_git_clean(brief_path: Path, parsed: dict[str, Any], skip: bool) -> None:
    if skip:
        print("  --skip-git-check passed; not enforcing")
        return
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"  git status failed (returncode={res.returncode}); skipping check")
        return
    expected = {
        "validate_claims.py",
        "validation_report.txt",
        "FINDINGS.md",
        "STATE.md",
        "EXPERIMENT_LOG.md",
        "NEXT_EXPERIMENTS.md",
    }
    fe = parsed["fe"]
    if fe.get("result_json"):
        expected.add(fe["result_json"])
    if fe.get("regen_cmd"):
        # Best-effort: the regen script path, if a literal `python <path>` form.
        m = re.match(r"^\s*python(?:3)?\s+([^\s]+)", fe["regen_cmd"])
        if m:
            expected.add(m.group(1))
    try:
        rel_brief = brief_path.relative_to(REPO).as_posix()
        expected.add(rel_brief)
    except ValueError:
        pass

    unexpected: list[str] = []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        # porcelain: "XY <path>" (or "XY <path> -> <new>")
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if path in expected:
            continue
        if any(path.startswith(e + "/") for e in expected):
            continue
        unexpected.append(line.rstrip())

    if unexpected:
        print("  GIT CHECK: working tree dirty outside expected files:")
        for line in unexpected[:20]:
            print(f"    {line}")
        raise SystemExit(
            "Refusing to promote with unrelated uncommitted changes. Commit / stash "
            "them, or pass --skip-git-check if you know what you're doing."
        )
    print("  ok working tree clean (or only expected files dirty)")


# ---------------------------------------------------------------------------
# Neo4j: FE status + Finding updates
# ---------------------------------------------------------------------------

def update_fe_status(parsed: dict[str, Any], dry_run: bool) -> None:
    fe = parsed["fe"]
    if dry_run:
        print(f"  Would call update_status.py {fe['fe_id']} {fe['status']} "
              f"--outcome {fe['outcome']!r}")
        return
    res = subprocess.run(
        [sys.executable, str(ROOT / "update_status.py"),
         fe["fe_id"], fe["status"], "--outcome", fe["outcome"]],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(f"update_status.py failed (exit {res.returncode})")
    last = (res.stdout.strip().splitlines() or [""])[-1]
    print(f"  ok {last}")


def update_premises_and_queue_sweep(parsed: dict[str, Any], dry_run: bool) -> None:
    """Flip :Premise nodes declared in the ## FE block and queue the semantic
    moot sweep.

    Optional ## FE YAML keys:
        refutes_premise: [dom-causal-lever]   # REFUTED + deterministic cascade
        confirms_premise: [free-baseline-strongest-readout]

    Every promoted result also appends `<fe_id>\\t<outcome>` to
    .autopilot/moot-trigger so `moot_sweep.py --from-trigger` can adjudicate
    the semantic long tail asynchronously.
    """
    fe = parsed["fe"]
    refutes = fe.get("refutes_premise") or []
    confirms = fe.get("confirms_premise") or []

    if refutes or confirms:
        from premises import cascade_moot
        from datetime import date as _date
        today = _date.today().isoformat()
        drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
        try:
            with drv.session() as s:
                for pid, new_status in (
                    [(p, "REFUTED") for p in refutes]
                    + [(p, "CONFIRMED") for p in confirms]
                ):
                    if dry_run:
                        print(f"  Would set Premise {pid} -> {new_status}"
                              + (" + cascade-moot reliant FEs"
                                 if new_status == "REFUTED" else ""))
                        continue
                    rec = s.run(
                        """
                        MATCH (pr:Premise {id: $pid})
                        SET pr.status = $status, pr.refuted_by = $by,
                            pr.reason = $reason, pr.status_date = $today
                        RETURN pr.id AS id
                        """,
                        pid=pid, status=new_status, by=fe["fe_id"],
                        reason=fe["outcome"], today=today,
                    ).single()
                    if not rec:
                        print(f"  WARNING: unknown premise {pid!r} — skipped "
                              "(python premises.py list)")
                        continue
                    print(f"  ok Premise {pid} -> {new_status}")
                    if new_status == "REFUTED":
                        mooted = cascade_moot(s, pid, fe["fe_id"], fe["outcome"])
                        print(f"     cascade mooted {len(mooted)} FE(s)"
                              + (f": {', '.join(mooted)}" if mooted else ""))
        finally:
            drv.close()
    else:
        print("  no premise declarations in ## FE block")

    one_line = " ".join(str(fe["outcome"]).split())
    if dry_run:
        print(f"  Would queue moot sweep: {fe['fe_id']}\\t{one_line[:80]}…")
        return
    trigger = REPO / ".autopilot" / "moot-trigger"
    trigger.parent.mkdir(exist_ok=True)
    with trigger.open("a") as fh:
        fh.write(f"{fe['fe_id']}\t{one_line}\n")
    print("  ok queued for moot_sweep.py --from-trigger "
          f"({trigger.relative_to(REPO)})")


def update_findings_in_graph(parsed: dict[str, Any], dry_run: bool) -> None:
    metas = parsed["findings_meta"]
    if not metas:
        print("  no F-N updates declared")
        return
    if dry_run:
        for m in metas:
            print(f"  Would MERGE/SET Finding F-{m['f_n']} "
                  f"(strength={m.get('strength','-')}, status={m.get('status','-')}, "
                  f"add_evidence={m.get('add_evidence', [])})")
        return
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    try:
        with drv.session() as s:
            for m in metas:
                f_id = f"F-{m['f_n']}"
                sets: list[str] = []
                params: dict[str, Any] = {"id": f_id}
                if m.get("strength"):
                    sets.append("f.strength = $strength")
                    params["strength"] = m["strength"]
                if m.get("status"):
                    sets.append("f.status = $status")
                    params["status"] = m["status"]
                if m.get("counterargument"):
                    sets.append("f.strongest_counterargument = $counterarg")
                    params["counterarg"] = m["counterargument"]
                if m.get("overturned_by"):
                    sets.append("f.would_be_overturned_by = $overturn")
                    params["overturn"] = m["overturned_by"]
                if sets:
                    s.run(
                        f"MERGE (f:Finding {{id: $id}}) SET {', '.join(sets)}",
                        **params,
                    )
                for exp_id in (m.get("add_evidence") or []):
                    s.run(
                        """
                        MERGE (f:Finding {id: $f})
                        MERGE (x:Experiment {id: $x})
                        MERGE (x)-[:PRODUCED]->(f)
                        SET f.evidence = CASE
                            WHEN f.evidence IS NULL THEN [$x]
                            WHEN $x IN f.evidence THEN f.evidence
                            ELSE f.evidence + [$x]
                        END
                        """,
                        f=f_id, x=exp_id,
                    )
                print(f"  ok Finding {f_id} updated "
                      f"({len(sets)} props, +{len(m.get('add_evidence') or [])} evidence)")
    finally:
        drv.close()


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------

def append_experiment_log(parsed: dict[str, Any], dry_run: bool) -> int:
    path = REPO / "EXPERIMENT_LOG.md"
    text = path.read_text()
    next_match = re.search(r"Next ID:\s*\*\*EXP-(\d+)\*\*", text)
    if not next_match:
        raise SystemExit('No "Next ID: **EXP-N**" line in EXPERIMENT_LOG.md.')
    next_id = int(next_match.group(1))

    block = parsed["experiment_log"].strip()
    decl_match = re.match(r"^## EXP-(\d+):", block)
    if not decl_match:
        raise SystemExit(
            "## EXPERIMENT_LOG entry must start with `## EXP-N: ...` header."
        )
    decl_id = int(decl_match.group(1))
    if decl_id != next_id:
        raise SystemExit(
            f"EXPERIMENT_LOG entry declares EXP-{decl_id} but Next ID is "
            f"EXP-{next_id}. Renumber the brief."
        )

    if dry_run:
        print(f"  Would append EXP-{decl_id} to EXPERIMENT_LOG.md, bump Next ID -> EXP-{decl_id + 1}")
        return decl_id

    template_anchor = "## Template for new experiments"
    insertion = block.rstrip() + "\n\n"
    if template_anchor in text:
        text = text.replace(template_anchor, insertion + template_anchor, 1)
    else:
        text = text.rstrip() + "\n\n" + insertion
    text = re.sub(
        r"Next ID:\s*\*\*EXP-\d+\*\*",
        f"Next ID: **EXP-{decl_id + 1}**",
        text,
        count=1,
    )
    path.write_text(text)
    print(f"  ok EXPERIMENT_LOG.md +EXP-{decl_id}, Next ID -> EXP-{decl_id + 1}")
    return decl_id


def replace_state_last_experiment(parsed: dict[str, Any], dry_run: bool) -> None:
    path = REPO / "STATE.md"
    text = path.read_text()
    body = parsed["state_last_experiment"].strip()
    anchor = "## Last experiment completed"

    m = re.search(r"^## Last experiment completed\n", text, re.MULTILINE)
    if not m:
        raise SystemExit('No "## Last experiment completed" section in STATE.md.')
    start = m.end()
    next_h = re.search(r"^## ", text[start:], re.MULTILINE)
    end = next_h.start() + start if next_h else len(text)

    new_section = f"\n{body.strip()}\n\n"
    if dry_run:
        print(f"  Would replace STATE.md '{anchor}' section "
              f"({end - start} chars -> {len(new_section)} chars)")
        return
    new_text = text[:start] + new_section + text[end:]
    path.write_text(new_text)
    print(f"  ok STATE.md '{anchor}' replaced")


def _f_block_bounds(text: str, f_n: int) -> tuple[int, int] | None:
    """Find [start, end) of an existing `### F-<n>:` block in FINDINGS.md.

    Block ends at the next `### F-` header, the `## ` next-section anchor,
    or EOF.
    """
    m = re.search(rf"^### F-{f_n}:", text, re.MULTILINE)
    if not m:
        return None
    start = m.start()
    next_f = re.search(r"^### F-\d+:", text[m.end():], re.MULTILINE)
    next_section = re.search(r"^## ", text[m.end():], re.MULTILINE)
    candidates = [c.start() + m.end() for c in [next_f, next_section] if c]
    end = min(candidates) if candidates else len(text)
    return (start, end)


def update_findings_md(parsed: dict[str, Any], dry_run: bool) -> None:
    blocks = parsed["findings_blocks"]
    if not blocks:
        print("  no F-N blocks in brief")
        return
    path = REPO / "FINDINGS.md"
    text = path.read_text()

    next_match = re.search(r"Next ID:\s*F-(\d+)", text)
    if not next_match:
        raise SystemExit('No "Next ID: F-N" line in FINDINGS.md.')
    next_id = int(next_match.group(1))

    update_blocks: list[tuple[int, str]] = []
    new_blocks: list[tuple[int, str]] = []
    for block in blocks:
        m = re.match(r"### F-(\d+):", block)
        f_n = int(m.group(1))
        clean = _strip_meta_yaml(block)
        if f_n < next_id:
            update_blocks.append((f_n, clean))
        else:
            new_blocks.append((f_n, clean))

    expected_new = list(range(next_id, next_id + len(new_blocks)))
    actual_new = [n for n, _ in new_blocks]
    if actual_new != expected_new:
        raise SystemExit(
            f"New F-N IDs {actual_new} do not match contiguous sequence from "
            f"Next ID F-{next_id} (expected {expected_new}). Renumber the brief."
        )

    new_text = text
    for f_n, block in update_blocks:
        bounds = _f_block_bounds(new_text, f_n)
        if not bounds:
            raise SystemExit(f"Could not locate F-{f_n} block to update.")
        start, end = bounds
        tail = new_text[start:end]
        trailing = "\n\n" if tail.endswith("\n\n") else "\n" if tail.endswith("\n") else ""
        new_text = new_text[:start] + block.strip() + trailing + new_text[end:]

    if new_blocks:
        new_next = next_id + len(new_blocks)
        anchor = "## Honorable mentions"
        if anchor not in new_text:
            raise SystemExit('FINDINGS.md missing "## Honorable mentions" anchor.')
        insertion = "\n" + "\n\n".join(b for _, b in new_blocks) + "\n\n---\n\n"
        new_text = new_text.replace(
            f"Next ID: F-{next_id}.",
            f"Next ID: F-{new_next}.",
            1,
        )
        new_text = new_text.replace(anchor, insertion + anchor, 1)
    else:
        new_next = next_id

    if dry_run:
        if update_blocks:
            print(f"  Would replace {len(update_blocks)} F-N block(s) in place: "
                  f"{[n for n, _ in update_blocks]}")
        if new_blocks:
            print(f"  Would append {len(new_blocks)} new F-N block(s): {actual_new}, "
                  f"Next ID -> F-{new_next}")
        return
    path.write_text(new_text)
    if update_blocks:
        print(f"  ok FINDINGS.md replaced {len(update_blocks)} block(s) in place: "
              f"{[n for n, _ in update_blocks]}")
    if new_blocks:
        print(f"  ok FINDINGS.md +{len(new_blocks)} block(s), Next ID -> F-{new_next}")


def _strip_meta_yaml(block: str) -> str:
    """Remove a leading ```yaml fence from a finding block (the meta-yaml is
    consumed for graph updates; the prose body is what lands in FINDINGS.md).

    Collapses runs of >2 newlines that the fence removal can leave behind so
    successive promotes don't accumulate blank lines under each F-N header.
    """
    stripped = YAML_BLOCK_RE.sub("", block, count=1)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


# ---------------------------------------------------------------------------
# Regenerate NEXT_EXPERIMENTS.md
# ---------------------------------------------------------------------------

def regen_next_experiments(dry_run: bool) -> None:
    if dry_run:
        print("  Would call generate_next_experiments.py")
        return
    res = subprocess.run(
        [sys.executable, str(ROOT / "generate_next_experiments.py")],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(res.returncode)
    last = (res.stdout.strip().splitlines() or ["NEXT_EXPERIMENTS.md regenerated"])[-1]
    print(f"  ok {last}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("brief", help="Path to result brief markdown file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing")
    p.add_argument("--skip-regen", action="store_true",
                   help="Don't run the regen_cmd declared in the brief")
    p.add_argument("--skip-git-check", action="store_true",
                   help="Skip the git-cleanliness gate (use only if you understand "
                        "the risk of bundling unrelated changes)")
    p.add_argument("--sweep", action="store_true",
                   help="After promotion, run moot_sweep.py --from-trigger "
                        "synchronously (LLM adjudication of adjacent open FEs). "
                        "Without this flag the verdict just queues in "
                        ".autopilot/moot-trigger.")
    args = p.parse_args()

    brief_path = Path(args.brief).expanduser().resolve()
    if not brief_path.exists():
        raise SystemExit(f"Brief not found: {brief_path}")

    date_str, fe_id = parse_filename(brief_path)
    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"=== promote_result: {brief_path.name} ===")
    print(f"date={date_str}  fe_id={fe_id}  mode={mode}")

    parsed = parse_brief(brief_path)
    if parsed["fe"]["fe_id"] != fe_id:
        raise SystemExit(
            f"Filename declares fe_id={fe_id} but ## FE block declares "
            f"fe_id={parsed['fe']['fe_id']}. They must match."
        )

    print(f"\nparsed: status={parsed['fe']['status']}, "
          f"findings={len(parsed['findings_blocks'])}, "
          f"new_claims={len(parsed['new_claims_lines'])}, "
          f"experiment_log={'yes' if parsed['experiment_log'] else 'no'}, "
          f"state_update={'yes' if parsed['state_last_experiment'] else 'no'}")

    print("\nstep 1: result_json verification")
    verify_result_json(parsed, args.dry_run)

    print("\nstep 2: regen")
    if args.skip_regen:
        print("  --skip-regen passed; not running")
    else:
        run_regen(parsed, args.dry_run)

    print("\nstep 3: validate-claims gate")
    assert_claims_gate(parsed, brief_path, args.dry_run)

    print("\nstep 4: git cleanliness")
    assert_git_clean(brief_path, parsed, args.skip_git_check)

    print("\nstep 5: FE status update (Neo4j)")
    update_fe_status(parsed, args.dry_run)

    print("\nstep 5b: Premise updates + moot-sweep queue")
    update_premises_and_queue_sweep(parsed, args.dry_run)

    print("\nstep 6: Finding updates (Neo4j)")
    update_findings_in_graph(parsed, args.dry_run)

    print("\nstep 7: EXPERIMENT_LOG.md")
    append_experiment_log(parsed, args.dry_run)

    print("\nstep 8: STATE.md (Last experiment completed)")
    replace_state_last_experiment(parsed, args.dry_run)

    print("\nstep 9: FINDINGS.md")
    update_findings_md(parsed, args.dry_run)

    print("\nstep 10: regenerate NEXT_EXPERIMENTS.md")
    regen_next_experiments(args.dry_run)

    if args.sweep and not args.dry_run:
        print("\nstep 11: moot sweep (--sweep)")
        subprocess.run(
            [sys.executable, str(ROOT / "moot_sweep.py"), "--from-trigger"],
            check=False,
        )

    if not args.dry_run:
        try:
            from datetime import datetime as _dt
            from pipeline.publisher import publish, set_research_field, init_redis_publisher, close_redis_publisher
            init_redis_publisher()
            publish("topoconf:research:promoted", {
                "fe_id": fe_id,
                "status": parsed["fe"]["status"],
                "findings_count": len(parsed["findings_blocks"]),
            })
            arxiv_id = parsed["fe"].get("triggered_by_arxiv", "")
            if arxiv_id:
                set_research_field(arxiv_id, {
                    f"fe_{fe_id}_findings_updated": _dt.now().isoformat(),
                })
            close_redis_publisher()
        except Exception:
            pass

    print()
    if args.dry_run:
        print("Dry run complete.")
    else:
        try:
            rel = brief_path.relative_to(REPO)
            print(f"DONE. Consider:\n  git add {rel} FINDINGS.md STATE.md "
                  f"EXPERIMENT_LOG.md NEXT_EXPERIMENTS.md validate_claims.py")
        except ValueError:
            print("DONE.")


if __name__ == "__main__":
    main()
