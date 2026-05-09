"""File-based session ledger for the outer orchestration loop.

The session state lives in `.claude/session-state.md` as a YAML frontmatter
block. Skills read it to know what stage they're in and write it back after
each stage transition. Compaction between stages is intentional — each new
context window reads this file cold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_STATE_PATH = REPO_ROOT / ".claude" / "session-state.md"

STAGES = [
    "BATCH_GATE",
    "EXECUTE",
    "TRIAGE",
    "PLAN_V1",
    "PLAN_V2",
    "PLAN_V3",
    "IMPLEMENT",
    "SWEEP",
    "COMPLETE",
]


@dataclass
class SessionState:
    stage: str = "BATCH_GATE"
    plan_file: str = ""
    plan_version: int = 0
    current_phase: int = 0
    total_phases: int = 0
    completed_phases: list[str] = field(default_factory=list)
    branch: str = ""
    batch_id: str = ""
    started: str = ""
    last_updated: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.started:
            self.started = date.today().isoformat()
        if not self.last_updated:
            self.last_updated = date.today().isoformat()

    def advance_to(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}. Must be one of {STAGES}")
        self.stage = stage
        self.last_updated = date.today().isoformat()

    def next_stage(self) -> str | None:
        if self.stage not in STAGES:
            return None
        idx = STAGES.index(self.stage)
        if idx + 1 >= len(STAGES):
            return None
        return STAGES[idx + 1]

    def complete_phase(self, phase_name: str) -> None:
        if phase_name not in self.completed_phases:
            self.completed_phases.append(phase_name)
        self.current_phase += 1
        self.last_updated = date.today().isoformat()


def load(path: Path | None = None) -> SessionState | None:
    """Load session state from the ledger file. Returns None if no file exists."""
    p = path or SESSION_STATE_PATH
    if not p.exists():
        return None
    text = p.read_text()
    front = _parse_frontmatter(text)
    if front is None:
        return None
    return SessionState(
        stage=front.get("stage", "BATCH_GATE"),
        plan_file=front.get("plan_file", ""),
        plan_version=int(front.get("plan_version", 0)),
        current_phase=int(front.get("current_phase", 0)),
        total_phases=int(front.get("total_phases", 0)),
        completed_phases=front.get("completed_phases", []),
        branch=front.get("branch", ""),
        batch_id=front.get("batch_id", ""),
        started=front.get("started", ""),
        last_updated=front.get("last_updated", ""),
        notes=front.get("notes", ""),
    )


def save(state: SessionState, path: Path | None = None) -> Path:
    """Write session state to the ledger file."""
    p = path or SESSION_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = date.today().isoformat()
    content = _render(state)
    p.write_text(content)
    return p


def create_new(branch: str = "", batch_id: str = "") -> SessionState:
    """Create a fresh session state with defaults."""
    today = date.today().isoformat()
    if not branch:
        branch = f"experiment/session-{today}"
    if not batch_id:
        batch_id = f"batch-{today}"
    return SessionState(
        stage="BATCH_GATE",
        branch=branch,
        batch_id=batch_id,
        started=today,
        last_updated=today,
    )


def _parse_frontmatter(text: str) -> dict[str, Any] | None:
    """Extract YAML-like frontmatter between --- delimiters."""
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    data: dict[str, Any] = {}
    lines = m.group(1).strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "|":
            # Multi-line scalar: collect indented continuation lines
            parts = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or lines[i] == ""):
                parts.append(lines[i][2:] if lines[i].startswith("  ") else "")
                i += 1
            data[key] = "\n".join(parts).strip()
            continue
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if inner:
                data[key] = [x.strip().strip("'\"") for x in inner.split(",")]
            else:
                data[key] = []
        elif val.isdigit():
            data[key] = int(val)
        else:
            data[key] = val
        i += 1
    return data


def _render(state: SessionState) -> str:
    """Render session state as a markdown file with YAML frontmatter."""
    phases_str = ", ".join(state.completed_phases) if state.completed_phases else ""
    notes_block = ""
    if state.notes:
        indented = "\n".join(f"  {line}" for line in state.notes.split("\n"))
        notes_block = f"notes: |\n{indented}"
    else:
        notes_block = "notes:"
    return f"""---
stage: {state.stage}
plan_file: {state.plan_file}
plan_version: {state.plan_version}
current_phase: {state.current_phase}
total_phases: {state.total_phases}
completed_phases: [{phases_str}]
branch: {state.branch}
batch_id: {state.batch_id}
started: {state.started}
last_updated: {state.last_updated}
{notes_block}
---

# Session: {state.batch_id}

**Stage:** {state.stage}
**Branch:** {state.branch}
**Progress:** phase {state.current_phase}/{state.total_phases} ({len(state.completed_phases)} completed)

{state.notes}
"""
