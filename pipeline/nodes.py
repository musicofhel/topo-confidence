"""Node implementations for the experiment pipeline graph."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from pipeline._match import (
    find_recompute_script as _find_recompute_script,
    match_result_dir,
    parse_fe_num,
)
from pipeline.observability import get_tracer, score_experiment, traced_subprocess
from pipeline.state import ExperimentState

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_GRAPH = REPO_ROOT / "research-graph"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _query_neo4j_ready_fes() -> list[dict[str, Any]]:
    """Query Neo4j for FEs with status READY or TRIGGERED, sorted by ROI desc."""
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv

        load_dotenv(RESEARCH_GRAPH / ".env")
        uri = os.getenv("NEO4J_BOLT_URL", "bolt://localhost:7688")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "topo_graph_dev")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (fe:FutureExperiment)
                    WHERE fe.status IN ['READY', 'TRIGGERED']
                    OPTIONAL MATCH (fe)-[:WOULD_UPDATE]->(f:Finding)
                    WITH fe, collect(DISTINCT f.id) AS would_update
                    RETURN fe {.*, would_update: would_update} AS fe
                    ORDER BY fe.roi_score DESC
                    """
                )
                return [dict(record["fe"]) for record in result]
        finally:
            driver.close()
    except Exception as e:
        log.warning("Neo4j query failed: %s", e)
        return []



# Cost tags that require cloud GPU (H100, A100, etc.)
_CLOUD_GPU_KEYWORDS = {"h100", "a100", "a6000", "cloud", "runpod", "pod"}


def _is_local_runnable(fe: dict[str, Any]) -> bool:
    """Check if an FE can run on local hardware (2060 Super + CPU)."""
    cost = str(fe.get("estimated_cost", "")).lower()
    if not cost:
        return True
    return not any(kw in cost for kw in _CLOUD_GPU_KEYWORDS)


# --- Phase 2: Deterministic nodes ---


def select_experiment(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.select")
    with tracer.start_as_current_span("select_experiment") as span:
        max_runs = state.get("max_runs", 0)
        completed = state.get("completed_this_session", [])
        if max_runs > 0 and len(completed) >= max_runs:
            span.set_attribute("selected_fe", "none")
            span.set_attribute("reason", f"max_runs cap reached ({len(completed)}/{max_runs})")
            return {"current_fe": None}

        local_only = state.get("local_only", False)
        fes = _query_neo4j_ready_fes()
        span.set_attribute("candidate_count", len(fes))

        for fe in fes:
            if local_only and not _is_local_runnable(fe):
                continue
            fe_id = fe.get("id", "")
            script = _find_recompute_script(fe_id)
            if script is not None:
                expected = parse_fe_num(fe_id)
                stem_num = script.stem.replace("recompute_", "").lower()
                if stem_num != expected and stem_num != expected[2:]:
                    log.critical(
                        "BREAKPOINT: select_experiment FE-script mismatch! "
                        "fe_id=%s but script=%s (stem=%s)",
                        fe_id, script, stem_num,
                    )
                    continue
                span.set_attribute("selected_fe", fe_id)
                span.set_attribute("script_path", str(script))
                return {
                    "current_fe": fe,
                    "script_path": str(script),
                    "result_json_path": "",
                    "result_json": None,
                    "brief_markdown": "",
                    "claims": [],
                    "findings_updates": [],
                    "human_verdict": "",
                    "human_edits": None,
                }

        span.set_attribute("selected_fe", "none")
        return {"current_fe": None}


def run_experiment(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.run")
    script_path = state["script_path"]
    fe_id = state["current_fe"]["id"]

    try:
        from pipeline.publisher import publish, set_research_field
        arxiv_id = state["current_fe"].get("triggered_by_arxiv", "")
        publish("topoconf:research:experiment_started", {
            "arxiv_id": arxiv_id,
            "fe_id": fe_id,
            "script_path": script_path,
        })
        if arxiv_id:
            set_research_field(arxiv_id, {
                f"fe_{fe_id}_status": "running",
                f"fe_{fe_id}_started_at": datetime.now().isoformat(),
            })
    except Exception:
        pass

    with traced_subprocess(
        tracer, "experiment.run", fe_id=fe_id, script_path=script_path
    ) as span:
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
        result = subprocess.run(
            [str(VENV_PYTHON), script_path],
            capture_output=True,
            text=True,
            timeout=14400,
            env=env,
            cwd=str(REPO_ROOT),
        )
        span.set_attribute("exit_code", result.returncode)
        if result.returncode != 0:
            span.set_attribute("stderr", result.stderr[:2000])

        return {"script_exit_code": result.returncode}


def parse_results(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.parse")
    with tracer.start_as_current_span("parse_results") as span:
        script_dir = Path(state["script_path"]).parent
        result_json_path = None

        for candidate in [
            script_dir / "results.json",
            *script_dir.glob("results_*.json"),
        ]:
            if candidate.exists():
                result_json_path = candidate
                break

        fe_id = state["current_fe"]["id"]
        if result_json_path is None:
            results_dir = REPO_ROOT / "pathway11_h100" / "results"
            if results_dir.is_dir():
                for candidate in results_dir.glob("*.json"):
                    if match_result_dir(fe_id, candidate.stem):
                        log.info("Result fallback matched: %s -> %s", fe_id, candidate)
                        result_json_path = candidate
                        break

        if result_json_path is None:
            span.set_attribute("error", "no result JSON found")
            return {
                "result_json_path": "",
                "result_json": None,
                "script_exit_code": 1,
            }

        with open(result_json_path) as f:
            data = json.load(f)

        span.set_attribute("result_json_path", str(result_json_path))
        span.set_attribute("key_count", len(data))

        try:
            from pipeline.publisher import publish, set_research_field
            arxiv_id = state["current_fe"].get("triggered_by_arxiv", "")
            auroc = data.get("auroc", data.get("test_auroc", ""))
            verdict = data.get("verdict", "")
            publish("topoconf:research:experiment_completed", {
                "arxiv_id": arxiv_id,
                "fe_id": fe_id,
                "auroc": auroc,
                "verdict": verdict,
            })
            if arxiv_id:
                set_research_field(arxiv_id, {
                    f"fe_{fe_id}_status": "completed",
                    f"fe_{fe_id}_auroc": str(auroc),
                    f"fe_{fe_id}_verdict": str(verdict),
                })
        except Exception:
            pass

        return {
            "result_json_path": str(result_json_path),
            "result_json": data,
        }


def handle_failure(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.failure")
    fe = state.get("current_fe", {})
    fe_id = fe.get("id", "unknown")
    with tracer.start_as_current_span(
        "handle_failure", attributes={"fe_id": fe_id}
    ) as span:
        error = {
            "fe_id": fe_id,
            "step": "run_experiment",
            "exit_code": state.get("script_exit_code", -1),
            "script_path": state.get("script_path", ""),
            "timestamp": time.time(),
        }
        span.set_attribute("error_summary", json.dumps(error))
        return {"errors": [error]}


# --- Phase 3: LLM nodes ---


def _parse_next_exp_id() -> int:
    log_path = REPO_ROOT / "EXPERIMENT_LOG.md"
    if not log_path.exists():
        log_path = RESEARCH_GRAPH / "EXPERIMENT_LOG.md"
    if not log_path.exists():
        return 1
    text = log_path.read_text()
    ids = [int(x) for x in re.findall(r"^## EXP-(\d+):", text, re.MULTILINE)]
    return max(ids) + 1 if ids else 1


def _read_findings_blocks(
    finding_ids: list[str],
) -> tuple[str, dict[str, str]]:
    findings_path = REPO_ROOT / "FINDINGS.md"
    if not findings_path.exists():
        return "", {}
    text = findings_path.read_text()

    header_end = text.find("\n### F-")
    header = text[:header_end].strip() if header_end != -1 else ""

    blocks: dict[str, str] = {}
    for m in re.finditer(r"(### (F-\d+):.*?)(?=\n### F-|\Z)", text, re.DOTALL):
        fid = m.group(2)
        if fid in finding_ids:
            blocks[fid] = m.group(1).strip()
    return header, blocks


def _call_claude(
    *,
    system: str,
    user_msg: str,
    tracer,
    span_name: str,
    fe_id: str,
) -> str:
    """Call claude CLI in print mode. Uses the user's existing Claude Code session."""
    import shutil

    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise EnvironmentError("'claude' CLI not found on PATH.")

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("fe_id", fe_id)
        span.set_attribute("backend", "claude-code-cli")
        try:
            prompt = f"{system}\n\n---\n\n{user_msg}"
            result = subprocess.run(
                [claude_bin, "-p", "--output-format", "json",
                 "--bare", "--tools", ""],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(REPO_ROOT),
            )
            if result.returncode != 0:
                span.set_attribute("error", result.stderr[:2000])
                raise RuntimeError(
                    f"claude -p exited {result.returncode}: {result.stderr[:500]}"
                )
            events = json.loads(result.stdout)
            if not isinstance(events, list):
                events = [events]
            tool_use_events = [ev for ev in events if ev.get("type") == "tool_use"]
            if tool_use_events:
                span.set_attribute("error", f"tool_use_loop: {len(tool_use_events)} events")
                raise RuntimeError(
                    f"claude -p entered tool-use loop ({len(tool_use_events)} tool_use events). "
                    f"First tool: {tool_use_events[0].get('name', 'unknown')}"
                )
            span.set_attribute("output_bytes", len(result.stdout))
            span.set_attribute("event_count", len(events))
            text = ""
            for ev in events:
                if ev.get("type") == "result":
                    text = ev.get("result", "")
                    if "total_cost_usd" in ev:
                        span.set_attribute("cost_usd", ev["total_cost_usd"])
                    break
            if not text:
                span.set_attribute("error", "no result event in claude output")
                raise RuntimeError("claude -p returned no result event")
            return text
        except subprocess.TimeoutExpired:
            span.set_attribute("error", "timeout")
            raise
        except Exception as e:
            span.set_attribute("error", str(e))
            raise


def _drill(data: Any, path: list) -> Any:
    cur = data
    for p in path:
        if isinstance(cur, dict):
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {p!r}")
    return cur


def generate_brief(state: ExperimentState) -> dict[str, Any]:
    from pipeline.prompts import brief_template

    tracer = get_tracer("pipeline.llm")
    fe = state["current_fe"]
    fe_id = fe.get("id", "unknown")

    with tracer.start_as_current_span("generate_brief", attributes={"fe_id": fe_id}):
        exp_id = _parse_next_exp_id()
        would_update = fe.get("would_update", [])
        findings_header, findings_blocks = _read_findings_blocks(would_update)

        user_msg = brief_template.build_user_message(
            fe=fe,
            result_json=state["result_json"],
            result_json_path=state["result_json_path"],
            script_path=state["script_path"],
            exp_id=exp_id,
            findings_header=findings_header,
            findings_blocks=findings_blocks,
        )

        text = _call_claude(
            system=brief_template.SYSTEM_PROMPT,
            user_msg=user_msg,
            tracer=tracer,
            span_name="llm.call.brief",
            fe_id=fe_id,
        )

        return {"brief_markdown": text}


def extract_claims(state: ExperimentState) -> dict[str, Any]:
    from pipeline.prompts import claims_template

    tracer = get_tracer("pipeline.llm")
    fe = state["current_fe"]
    fe_id = fe.get("id", "unknown")
    result_json = state.get("result_json") or {}

    with tracer.start_as_current_span("extract_claims", attributes={"fe_id": fe_id}) as node_span:
        brief = state.get("brief_markdown", "")
        claims_section = ""
        m = re.search(
            r"## New claims\n(.*?)(?=\n## |\Z)", brief, re.DOTALL
        )
        if m:
            claims_section = m.group(1).strip()

        user_msg = claims_template.build_user_message(
            result_json=result_json,
            result_json_path=state.get("result_json_path", ""),
            script_path=state.get("script_path", ""),
            fe_id=fe_id,
            brief_claims_section=claims_section,
        )

        text = _call_claude(
            system=claims_template.SYSTEM_PROMPT,
            user_msg=user_msg,
            tracer=tracer,
            span_name="llm.call.claims",
            fe_id=fe_id,
        )

        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        try:
            proposed = json.loads(raw)
        except json.JSONDecodeError:
            node_span.set_attribute("parse_error", "invalid JSON from LLM")
            return {"claims": []}

        validated = []
        for claim in proposed:
            cid = claim.get("cid", "")
            path = claim.get("path", [])
            expected = claim.get("expected")
            tol = claim.get("tol", 0.005)
            file_path = claim.get("file", state.get("result_json_path", ""))

            try:
                actual = _drill(result_json, path)
            except (KeyError, IndexError, TypeError):
                node_span.set_attribute(f"dropped.{cid}", "invalid path")
                continue

            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if abs(actual - expected) > tol:
                    claim["expected"] = actual

            claim["file"] = file_path
            desc = claim.get("description", "").replace('"', '\\"')
            code = (
                f'    Claim("{cid}", "{desc}",\n'
                f'          "{file_path}",\n'
                f"          {path!r}, {actual}, {tol}, "
                f'"{claim.get("labels", "1024tok")}"),'
            )
            claim["code"] = code
            claim["expected"] = actual
            validated.append(claim)

        node_span.set_attribute("claims_proposed", len(proposed))
        node_span.set_attribute("claims_validated", len(validated))
        return {"claims": validated}


def interpret_findings(state: ExperimentState) -> dict[str, Any]:
    from pipeline.prompts import findings_template

    tracer = get_tracer("pipeline.llm")
    fe = state["current_fe"]
    fe_id = fe.get("id", "unknown")
    would_update = fe.get("would_update", [])

    with tracer.start_as_current_span("interpret_findings", attributes={"fe_id": fe_id}) as node_span:
        if not would_update:
            node_span.set_attribute("skipped", "no would_update edges")
            return {"findings_updates": []}

        findings_header, findings_blocks = _read_findings_blocks(would_update)
        if not findings_blocks:
            node_span.set_attribute("skipped", "no matching findings blocks")
            return {"findings_updates": []}

        exp_id = _parse_next_exp_id()
        user_msg = findings_template.build_user_message(
            brief=state.get("brief_markdown", ""),
            result_json=state.get("result_json") or {},
            exp_id=exp_id,
            would_update=would_update,
            findings_header=findings_header,
            findings_blocks=findings_blocks,
        )

        text = _call_claude(
            system=findings_template.SYSTEM_PROMPT,
            user_msg=user_msg,
            tracer=tracer,
            span_name="llm.call.findings",
            fe_id=fe_id,
        )

        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        try:
            updates = json.loads(raw)
        except json.JSONDecodeError:
            node_span.set_attribute("parse_error", "invalid JSON from LLM")
            return {"findings_updates": []}

        node_span.set_attribute("findings_updated", len(updates))
        return {"findings_updates": updates}


# --- Phase 4: Human review ---


def review_gate(state: ExperimentState) -> dict[str, Any]:
    """Phase 4 human review gate — interrupts pipeline for approve/reject/edit."""
    tracer = get_tracer("pipeline.review")
    fe = state.get("current_fe", {})
    fe_id = fe.get("id", "unknown")
    fe_desc = fe.get("description", "")

    with tracer.start_as_current_span("review_gate", attributes={"fe_id": fe_id}) as span:
        payload = {
            "fe_id": fe_id,
            "description": fe_desc,
            "result_json_path": state.get("result_json_path", ""),
            "brief_markdown": state.get("brief_markdown", ""),
            "claims": state.get("claims", []),
            "findings_updates": state.get("findings_updates", []),
            "dry_run": state.get("dry_run", False),
        }

        response = interrupt(payload)

        verdict = response.get("verdict", "reject") if isinstance(response, dict) else str(response)
        edits = response.get("edits") if isinstance(response, dict) else None

        span.set_attribute("verdict", verdict)

        try:
            trace_hex = format(span.get_span_context().trace_id, "032x")
            score_experiment(trace_hex, human_approved=(verdict == "approve"))
        except Exception:
            pass

        return {
            "review_payload": payload,
            "human_verdict": verdict,
            "human_edits": edits,
        }


def revise_artifacts(state: ExperimentState) -> dict[str, Any]:
    edits = state.get("human_edits") or {}
    updates: dict[str, Any] = {}

    if "brief_patch" in edits:
        updates["brief_markdown"] = edits["brief_patch"]
    if "claims_patch" in edits:
        updates["claims"] = edits["claims_patch"]
    if "findings_patch" in edits:
        updates["findings_updates"] = edits["findings_patch"]

    return updates


# --- Phase 2: File write + promote nodes ---


def write_brief(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.write")
    with tracer.start_as_current_span("write_brief") as span:
        fe_id = state["current_fe"]["id"]
        today = date.today().isoformat()
        brief_filename = f"result-{today}-{fe_id}.md"
        brief_path = RESEARCH_GRAPH / "briefs" / brief_filename

        brief_path.write_text(state["brief_markdown"])
        span.set_attribute("brief_path", str(brief_path))
        return {"result_json_path": state.get("result_json_path", "")}


def write_claims(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.write")
    with tracer.start_as_current_span("write_claims") as span:
        claims = state.get("claims", [])
        if not claims:
            span.set_attribute("skipped", True)
            return {"claims_validation_result": "no claims to write"}

        validate_path = REPO_ROOT / "validate_claims.py"
        content = validate_path.read_text()

        insert_marker = "# ----- §7"
        if insert_marker not in content:
            insert_marker = "# -----"

        claims_code = "\n".join(c["code"] for c in claims if "code" in c)
        if claims_code and insert_marker in content:
            content = content.replace(
                insert_marker, f"{claims_code}\n\n{insert_marker}"
            )
            validate_path.write_text(content)

        brief_path = _latest_brief_path(state)
        if brief_path and brief_path.exists():
            assert validate_path.stat().st_mtime > brief_path.stat().st_mtime, (
                "Claims gate: validate_claims.py must be newer than the brief"
            )

        result = subprocess.run(
            [str(VENV_PYTHON), str(validate_path), "--no-regen"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        span.set_attribute("exit_code", result.returncode)
        span.set_attribute("stdout_tail", result.stdout[-500:])

        try:
            trace_hex = format(span.get_span_context().trace_id, "032x")
            score_experiment(trace_hex, claims_pass_rate=1.0 if result.returncode == 0 else 0.0)
        except Exception:
            pass

        if result.returncode != 0:
            validate_path.write_text(
                validate_path.read_text().replace(claims_code + "\n\n", "")
            )
            return {
                "claims_validation_result": f"FAIL: {result.stdout[-500:]}",
                "script_exit_code": 1,
            }

        return {"claims_validation_result": result.stdout[-500:]}


def _latest_brief_path(state: ExperimentState) -> Path | None:
    fe_id = state.get("current_fe", {}).get("id", "")
    today = date.today().isoformat()
    brief_path = RESEARCH_GRAPH / "briefs" / f"result-{today}-{fe_id}.md"
    return brief_path if brief_path.exists() else None


def promote(state: ExperimentState) -> dict[str, Any]:
    tracer = get_tracer("pipeline.promote")
    fe_id = state["current_fe"]["id"]

    if state.get("dry_run", False):
        return {
            "promotion_stdout": f"DRY RUN: skipped promotion for {fe_id}",
            "completed_this_session": [f"{fe_id} (dry-run)"],
        }

    with traced_subprocess(tracer, "promote.run", fe_id=fe_id) as span:
        brief_path = _latest_brief_path(state)
        if brief_path is None:
            return {
                "promotion_stdout": "ERROR: no brief found",
                "errors": [
                    {"fe_id": fe_id, "step": "promote", "exit_code": 1}
                ],
            }

        result = subprocess.run(
            [
                str(VENV_PYTHON),
                str(RESEARCH_GRAPH / "promote_result.py"),
                str(brief_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(RESEARCH_GRAPH),
        )
        span.set_attribute("exit_code", result.returncode)

        try:
            trace_hex = format(span.get_span_context().trace_id, "032x")
            score_experiment(trace_hex, promotion_success=(result.returncode == 0))
        except Exception:
            pass

        if result.returncode != 0:
            return {
                "promotion_stdout": result.stderr[-1000:],
                "errors": [
                    {
                        "fe_id": fe_id,
                        "step": "promote",
                        "exit_code": result.returncode,
                        "stderr": result.stderr[-500:],
                    }
                ],
            }

        return {
            "promotion_stdout": result.stdout[-500:],
            "completed_this_session": [fe_id],
        }
