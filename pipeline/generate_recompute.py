"""Generate recompute scripts for FutureExperiments that lack them.

Queries Neo4j for READY FEs that are local-runnable and have no script,
then uses claude -p to generate a recompute script from exemplars + FE metadata.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pipeline.nodes import (
    REPO_ROOT,
    _find_recompute_script,
    _is_local_runnable,
    _query_neo4j_ready_fes,
)
from pipeline.prompts.recompute_system import (
    EXEMPLAR_FE101,
    EXEMPLAR_FE308,
    EXEMPLAR_FE447,
    RECOMPUTE_SYSTEM,
)

log = logging.getLogger(__name__)

BANNED_IMPORTS = {"requests", "urllib", "httpx", "torch", "tensorflow", "jax", "transformers", "accelerate"}
PATH_PREFIX = "/home/musicofhel/topo-confidence"


def scriptless_local_fes() -> list[dict[str, Any]]:
    """Return READY/TRIGGERED FEs that are local-runnable and have no script."""
    fes = _query_neo4j_ready_fes()
    return [
        fe for fe in fes
        if _is_local_runnable(fe) and _find_recompute_script(fe["id"]) is None
    ]


def _fe_to_slug(description: str) -> str:
    slug = re.sub(r"[^a-z0-9 ]", "", description.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:40]


def _extract_python(text: str) -> str:
    fenced = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    fenced = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _validate_script(code: str, script_path: Path) -> list[str]:
    """Validate generated script. Returns list of errors (empty = OK)."""
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    banned_found = imports & BANNED_IMPORTS
    if banned_found:
        errors.append(f"Banned imports: {banned_found}")

    has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )
    if not has_main:
        errors.append("Missing def main()")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("/") and not node.value.startswith(PATH_PREFIX):
                if node.value.startswith("/home/") or node.value.startswith("/tmp/"):
                    errors.append(f"Path outside repo: {node.value}")

    return errors


def _build_prompt(fe: dict[str, Any]) -> str:
    fe_id = fe["id"]
    desc = fe.get("description", "")
    rationale = fe.get("rationale", "")
    depends = fe.get("depends_on", []) or fe.get("depends-on", [])
    cost = fe.get("estimated_cost", "")

    lines = [
        f"Generate a recompute script for experiment {fe_id}.",
        f"",
        f"Description: {desc}",
        f"Rationale: {rationale}",
        f"Depends on findings: {depends}",
        f"Estimated cost: {cost}",
        f"",
        f"The script must write its results JSON to:",
        f"  pathway11_h100/<appropriate_subdir>/results.json",
        f"",
        f"Here are 3 exemplar scripts that show the exact pattern to follow:",
        f"",
        f"=== EXEMPLAR 1: FE101 (LEACE erasure) ===",
        EXEMPLAR_FE101,
        f"",
        f"=== EXEMPLAR 2: FE308 (adaptive best-of-k) ===",
        EXEMPLAR_FE308,
        f"",
        f"=== EXEMPLAR 3: FE447 (length baseline) ===",
        EXEMPLAR_FE447,
    ]
    return "\n".join(lines)


def generate_for_fe(fe: dict[str, Any], *, dry_run: bool = False) -> Path | None:
    """Generate a recompute script for a single FE. Returns path or None on failure."""
    fe_id = fe["id"]
    fe_num = fe_id.split("-")[-1].lower()
    slug = _fe_to_slug(fe.get("description", fe_id))
    script_dir = REPO_ROOT / f"pathway11_h100/{slug}"
    script_path = script_dir / f"recompute_{fe_num}.py"

    if script_path.exists():
        log.info("%s: script already exists at %s", fe_id, script_path)
        return script_path

    log.info("%s: generating recompute script → %s", fe_id, script_path)

    system = RECOMPUTE_SYSTEM
    user_msg = _build_prompt(fe)

    if dry_run:
        log.info("%s: dry run — would call claude -p", fe_id)
        return None

    claude_bin = shutil.which("claude")
    if not claude_bin:
        log.error("claude CLI not found on PATH")
        return None

    prompt = f"{system}\n\n---\n\n{user_msg}"
    try:
        result = subprocess.run(
            [claude_bin, "-p", "--output-format", "json",
             "--bare", "--tools", ""],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        log.error("%s: claude -p timed out", fe_id)
        return None

    if result.returncode != 0:
        log.error("%s: claude -p exited %d: %s", fe_id, result.returncode, result.stderr[:500])
        return None

    try:
        events = json.loads(result.stdout)
        if not isinstance(events, list):
            events = [events]
        tool_use_events = [ev for ev in events if ev.get("type") == "tool_use"]
        if tool_use_events:
            log.critical(
                "BREAKPOINT: %s claude -p entered tool-use loop despite --bare --tools ''! "
                "%d tool_use events. First tool: %s",
                fe_id, len(tool_use_events),
                tool_use_events[0].get("name", "unknown"),
            )
            return None
        log.info(
            "AUDIT scriptgen %s: %d bytes, %d events",
            fe_id, len(result.stdout), len(events),
        )
        text = ""
        for ev in events:
            if ev.get("type") == "result":
                text = ev.get("result", "")
                break
        if not text:
            log.error("%s: no result event in claude output", fe_id)
            return None
    except json.JSONDecodeError:
        log.error("%s: failed to parse claude output", fe_id)
        return None

    code = _extract_python(text)
    errors = _validate_script(code, script_path)
    if errors:
        log.error("%s: validation failed: %s", fe_id, "; ".join(errors))
        return None

    script_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code)
    log.info("%s: wrote %d bytes to %s", fe_id, len(code), script_path)

    try:
        subprocess.run(
            [str(REPO_ROOT / ".venv/bin/python"), "-c", f"import ast; ast.parse(open('{script_path}').read())"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        log.warning("%s: post-write parse check failed: %s", fe_id, e)

    return script_path
