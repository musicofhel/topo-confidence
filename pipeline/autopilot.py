"""Autopilot daemon: triage papers → generate scripts → run experiments → classify results.

Supports multi-worker deployment via --phase flag. Each phase runs independently
with its own budget file and lock, so three workers can churn concurrently.

Usage:
    python -m pipeline autopilot                          # all phases, sequential (legacy)
    python -m pipeline autopilot --phase triage --budget 20
    python -m pipeline autopilot --phase scriptgen --budget 20
    python -m pipeline autopilot --phase experiment --budget 15
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

from pipeline.generate_recompute import generate_for_fe, scriptless_local_fes
from pipeline.nodes import (
    REPO_ROOT,
    _find_recompute_script,
    _is_local_runnable,
    _query_neo4j_ready_fes,
)
from pipeline.result_triage import classify

log = logging.getLogger("autopilot")

AUTOPILOT_DIR = REPO_ROOT / ".autopilot"
TRIGGER_FILE = AUTOPILOT_DIR / "trigger"
BUDGET_FILE = AUTOPILOT_DIR / "budget.json"
FAILURES_FILE = AUTOPILOT_DIR / "failures.json"
FOLLOWUPS_DIR = AUTOPILOT_DIR / "followups"
EXPERIMENT_LOCK = AUTOPILOT_DIR / "experiment.lock"

TRIAGE_TIMEOUT = 1800
EXPERIMENT_TIMEOUT = 14400
MAX_RETRIES = 2


def _ensure_dirs() -> None:
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    FOLLOWUPS_DIR.mkdir(parents=True, exist_ok=True)


def _budget_path(phase: str | None) -> Path:
    if phase:
        return AUTOPILOT_DIR / f"budget-{phase}.json"
    return BUDGET_FILE


def _read_budget(daily_cap: int, phase: str | None = None) -> tuple[int, str]:
    """Returns (remaining_calls, budget_date)."""
    today = str(date.today())
    path = _budget_path(phase)
    if path.exists():
        try:
            with open(path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.loads(f.read())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                if data.get("date") == today:
                    return max(0, daily_cap - data.get("calls", 0)), today
        except (json.JSONDecodeError, KeyError):
            pass
    path.write_text(json.dumps({"date": today, "calls": 0}))
    return daily_cap, today


def _increment_budget(phase: str | None = None) -> None:
    today = str(date.today())
    path = _budget_path(phase)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            data = {"date": today, "calls": 0}
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    if data.get("date") != today:
                        data = {"date": today, "calls": 0}
                except (json.JSONDecodeError, KeyError):
                    data = {"date": today, "calls": 0}
            data["calls"] = data.get("calls", 0) + 1
            path.write_text(json.dumps(data))
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _read_failures() -> dict[str, int]:
    if FAILURES_FILE.exists():
        try:
            return json.loads(FAILURES_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _record_failure(item_id: str) -> int:
    failures = _read_failures()
    failures[item_id] = failures.get(item_id, 0) + 1
    FAILURES_FILE.write_text(json.dumps(failures, indent=2))
    return failures[item_id]


def _is_quarantined(item_id: str) -> bool:
    return _read_failures().get(item_id, 0) > MAX_RETRIES


def _check_trigger() -> list[str]:
    """Read and clear the trigger file. Returns list of arxiv IDs."""
    if not TRIGGER_FILE.exists():
        return []
    try:
        content = TRIGGER_FILE.read_text().strip()
        if not content:
            return []
        ids = [line.strip() for line in content.splitlines() if line.strip()]
        TRIGGER_FILE.write_text("")
        return ids
    except Exception:
        return []


def _pending_papers() -> list[str]:
    """Query Neo4j for pending_triage papers."""
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / "research-graph" / ".env")
        uri = os.getenv("NEO4J_BOLT_URL", "bolt://localhost:7688")
        driver = GraphDatabase.driver(uri, auth=("neo4j", "topo_graph_dev"))
        with driver.session() as session:
            result = session.run(
                "MATCH (p:Paper) WHERE p.status = 'pending_triage' "
                "RETURN p.arxiv_id AS aid ORDER BY p.suggested_at DESC"
            )
            return [r["aid"] for r in result if r["aid"]]
    except Exception as e:
        log.warning("Neo4j pending query failed: %s", e)
        return []


def _triage_one_paper(arxiv_id: str, *, dry_run: bool = False) -> bool:
    """Run triage_one.sh for a single paper. Returns True on success."""
    log.info("Triaging %s", arxiv_id)
    if dry_run:
        log.info("[dry-run] would triage %s", arxiv_id)
        return True

    triage_script = REPO_ROOT / "research-graph" / "triage_one.sh"
    try:
        result = subprocess.run(
            ["bash", str(triage_script), arxiv_id],
            capture_output=True,
            text=True,
            timeout=TRIAGE_TIMEOUT,
            cwd=str(REPO_ROOT / "research-graph"),
        )
        log.info("triage_one.sh %s exited %d: %s", arxiv_id, result.returncode, result.stdout.strip())
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.error("triage_one.sh %s timed out after %ds", arxiv_id, TRIAGE_TIMEOUT)
        return False


def _runnable_fes() -> list[dict[str, Any]]:
    """FEs that are local-runnable and have a recompute script."""
    fes = _query_neo4j_ready_fes()
    return [
        fe for fe in fes
        if _is_local_runnable(fe) and _find_recompute_script(fe["id"]) is not None
    ]


_dirty_tree_streak = 0


def _git_is_clean() -> bool:
    global _dirty_tree_streak
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "pathway11_h100/"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            _dirty_tree_streak = 0
            return True
    except Exception:
        return False

    _dirty_tree_streak += 1
    if _dirty_tree_streak >= 5:
        dirty = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", "pathway11_h100/"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        dirty_files = dirty.stdout.strip().splitlines() if dirty.stdout else []
        log.warning("Dirty-tree streak hit %d, auto-committing: %s",
                     _dirty_tree_streak, dirty_files)

        add_result = subprocess.run(
            ["git", "add", "--", "pathway11_h100/"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=30,
        )
        if add_result.returncode != 0:
            log.critical(
                "BREAKPOINT: git add failed (rc=%d): %s",
                add_result.returncode, add_result.stderr[:500],
            )
            return False

        commit_result = subprocess.run(
            ["git", "commit", "-m",
             f"autopilot: auto-commit dirty experiment outputs\n\n"
             f"Files: {', '.join(dirty_files[:10])}"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=30,
        )
        if commit_result.returncode != 0:
            log.critical(
                "BREAKPOINT: git commit failed (rc=%d): %s",
                commit_result.returncode, commit_result.stderr[:500],
            )
            return False

        verify = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        log.info("Auto-commit verified: %s", verify.stdout.strip()[:12])

        _dirty_tree_streak = 0
        return True
    return False


def _acquire_experiment_lock() -> int | None:
    """Try to acquire experiment lock. Returns fd on success, None if busy."""
    try:
        fd = os.open(str(EXPERIMENT_LOCK), os.O_WRONLY | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (BlockingIOError, OSError):
        return None


def _release_experiment_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def _auto_commit_results(fe_id: str) -> bool:
    """Commit result files in pathway11_h100/. Returns True on success."""
    try:
        diff = subprocess.run(
            ["git", "status", "--porcelain", "--", "pathway11_h100/"],
            capture_output=True, text=True, timeout=10, cwd=str(REPO_ROOT),
        )
        if not diff.stdout.strip():
            return True

        add_result = subprocess.run(
            ["git", "add", "--", "pathway11_h100/"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=30,
        )
        if add_result.returncode != 0:
            log.critical(
                "BREAKPOINT: git add failed for %s (rc=%d): %s",
                fe_id, add_result.returncode, add_result.stderr[:500],
            )
            return False

        commit_result = subprocess.run(
            ["git", "commit", "-m", f"autopilot: {fe_id} experiment results"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=30,
        )
        if commit_result.returncode != 0:
            log.critical(
                "BREAKPOINT: git commit failed for %s (rc=%d): %s",
                fe_id, commit_result.returncode, commit_result.stderr[:500],
            )
            return False

        verify = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        log.info("Auto-committed results for %s (hash: %s)",
                 fe_id, verify.stdout.strip()[:12])
        return True
    except Exception as e:
        log.critical("BREAKPOINT: auto-commit exception for %s: %s", fe_id, e)
        return False


def _run_one_experiment(fe_id: str, *, dry_run: bool = False) -> dict[str, Any] | None:
    """Run pipeline for a single FE. Returns result JSON or None."""
    log.info("Running experiment %s", fe_id)
    if dry_run:
        log.info("[dry-run] would run experiment %s", fe_id)
        return None

    venv_python = str(REPO_ROOT / ".venv" / "bin" / "python")
    try:
        result = subprocess.run(
            [venv_python, "-m", "pipeline", "run",
             "--fe", fe_id, "--local", "--no-review", "--max-runs", "1",
             "--no-langfuse", "--no-jaeger"],
            capture_output=True,
            text=True,
            timeout=EXPERIMENT_TIMEOUT,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
        log.info("pipeline run %s exited %d", fe_id, result.returncode)
        if result.stdout:
            log.debug("stdout: %s", result.stdout[-1000:])
        if result.returncode != 0:
            log.error("stderr: %s", result.stderr[-1000:])
            return None
        return {"exit_code": 0, "fe_id": fe_id}
    except subprocess.TimeoutExpired:
        log.error("pipeline run %s timed out after %ds", fe_id, EXPERIMENT_TIMEOUT)
        return None


def _classify_latest_result(fe_id: str) -> str | None:
    """Find and classify the most recent result JSON for an FE."""
    from pipeline._match import match_result_dir

    candidates = []
    for results_json in REPO_ROOT.glob("pathway11_h100/**/results.json"):
        parent = results_json.parent.name
        if match_result_dir(fe_id, parent):
            candidates.append(results_json)

    if len(candidates) > 1:
        log.critical(
            "BREAKPOINT: multiple result dirs match %s: %s — returning None",
            fe_id, [str(c) for c in candidates],
        )
        return None
    if not candidates:
        return None

    try:
        data = json.loads(candidates[0].read_text())
        classification = classify(data)
        log.info("%s classified as %s (from %s)", fe_id, classification, candidates[0])
        return classification
    except Exception as e:
        log.warning("Failed to classify %s: %s", candidates[0], e)
    return None


def _write_followup(fe_id: str, classification: str, result: dict[str, Any] | None) -> None:
    """Write a follow-up proposal for human review."""
    followup_path = FOLLOWUPS_DIR / f"{fe_id}.md"
    lines = [
        f"# Follow-up: {fe_id}",
        f"",
        f"Classification: **{classification}**",
        f"Date: {date.today()}",
        f"",
        f"## Result summary",
        f"",
        json.dumps(result, indent=2, default=str) if result else "No result data captured.",
        f"",
        f"## Recommended next steps",
        f"",
        f"_Human review required — autopilot does not auto-generate follow-up experiments._",
    ]
    followup_path.write_text("\n".join(lines))
    log.info("Wrote follow-up to %s", followup_path)


def _check_stale_pid(phase: str | None) -> bool:
    """Returns True if ok to start, False if another instance is running.

    Uses O_CREAT|O_EXCL for atomic creation after stale-PID cleanup.
    Treats PID files older than 24h as stale even if the PID is still alive.
    """
    pid_name = f"daemon-{phase}.pid" if phase else "daemon.pid"
    pid_path = AUTOPILOT_DIR / pid_name

    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            os.kill(old_pid, 0)
            age_hours = (time.time() - pid_path.stat().st_mtime) / 3600
            if age_hours > 24:
                log.warning(
                    "PID file for %s is %.1fh old (PID %d). Treating as stale.",
                    phase or "all", age_hours, old_pid,
                )
            else:
                log.error(
                    "Phase %s already running (PID %d, age %.1fh), exiting",
                    phase or "all", old_pid, age_hours,
                )
                return False
        except (OSError, ValueError):
            log.info("Removing stale PID file for %s", phase or "all")

    try:
        pid_path.unlink(missing_ok=True)
        fd = os.open(str(pid_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        log.error(
            "Race condition: another %s worker started between check and write",
            phase or "all",
        )
        return False


def run_loop(
    *,
    poll_interval: int = 60,
    daily_cap: int = 15,
    dry_run: bool = False,
    phase: str | None = None,
) -> None:
    """Main autopilot loop.

    When *phase* is set, only that phase runs — suitable for multi-worker
    deployment where each terminal runs one phase with its own budget.
    When *phase* is None, all phases run with legacy sequential gating.
    """
    _ensure_dirs()

    if not _check_stale_pid(phase):
        return

    pid_name = f"daemon-{phase}.pid" if phase else "daemon.pid"
    pid_path = AUTOPILOT_DIR / pid_name

    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        log.info("Received signal %d, shutting down after current item", signum)
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Autopilot starting (phase=%s, poll=%ds, budget=%d/day, dry_run=%s)",
             phase or "all", poll_interval, daily_cap, dry_run)

    skipped_this_cycle: set[str] = set()

    try:
        while not shutdown:
            triggered_ids = _check_trigger()
            if triggered_ids:
                log.info("Trigger file contained: %s", triggered_ids)

            remaining, _ = _read_budget(daily_cap, phase)
            if remaining <= 0:
                log.info("Daily budget exhausted, sleeping")
                time.sleep(poll_interval)
                skipped_this_cycle.clear()
                continue

            did_work = False

            # Phase 1: Triage one pending paper
            if phase in (None, "triage"):
                pending = _pending_papers()
                pending = [p for p in pending
                           if not _is_quarantined(f"triage-{p}") and p not in skipped_this_cycle]
                if pending and remaining >= 1:
                    arxiv_id = pending[0]
                    ok = _triage_one_paper(arxiv_id, dry_run=dry_run)
                    # Rate-limit: consume budget slot regardless of outcome.
                    # Quarantine mechanism handles retries.
                    _increment_budget(phase)
                    remaining -= 1
                    did_work = True
                    if not ok:
                        count = _record_failure(f"triage-{arxiv_id}")
                        log.warning("Triage failed for %s (attempt %d/%d)", arxiv_id, count, MAX_RETRIES + 1)
                    skipped_this_cycle.add(arxiv_id)

            # Phase 2: Generate script for one scriptless FE
            if phase in (None, "scriptgen") and (phase is not None or not did_work):
                scriptless = scriptless_local_fes()
                scriptless = [fe for fe in scriptless
                              if not _is_quarantined(f"gen-{fe['id']}") and fe['id'] not in skipped_this_cycle]
                if scriptless and remaining >= 1:
                    fe = scriptless[0]
                    path = generate_for_fe(fe, dry_run=dry_run)
                    # Rate-limit: consume budget slot regardless of outcome.
                    _increment_budget(phase)
                    remaining -= 1
                    did_work = True
                    if path is None and not dry_run:
                        count = _record_failure(f"gen-{fe['id']}")
                        log.warning("Script gen failed for %s (attempt %d/%d)", fe["id"], count, MAX_RETRIES + 1)
                    skipped_this_cycle.add(fe['id'])

            # Phase 3: Run one experiment
            if phase in (None, "experiment") and (phase is not None or not did_work):
                runnable = _runnable_fes()
                runnable = [fe for fe in runnable
                            if not _is_quarantined(f"run-{fe['id']}") and fe['id'] not in skipped_this_cycle]
                if runnable and remaining >= 3:
                    if not _git_is_clean():
                        log.warning("Experiment output tree is dirty, skipping")
                    else:
                        lock_fd = _acquire_experiment_lock()
                        if lock_fd is None:
                            log.info("Experiment lock held by another worker, skipping")
                        else:
                            try:
                                fe = runnable[0]
                                result = _run_one_experiment(fe["id"], dry_run=dry_run)
                                # Rate-limit: consume 3 budget slots regardless of outcome.
                                for _ in range(3):
                                    _increment_budget(phase)
                                remaining -= 3
                                did_work = True
                                if result is None and not dry_run:
                                    count = _record_failure(f"run-{fe['id']}")
                                    log.warning("Experiment failed for %s (attempt %d/%d)", fe["id"], count, MAX_RETRIES + 1)
                                elif result is not None:
                                    committed = _auto_commit_results(fe["id"])
                                    if not committed:
                                        log.error("Auto-commit failed for %s, skipping classification", fe["id"])
                                    else:
                                        classification = _classify_latest_result(fe["id"])
                                        if classification and classification in ("HIT", "NEAR_MISS"):
                                            _write_followup(fe["id"], classification, result)
                                skipped_this_cycle.add(fe['id'])
                            finally:
                                _release_experiment_lock(lock_fd)

            if not did_work:
                log.debug("Nothing to do, sleeping %ds", poll_interval)
                time.sleep(poll_interval)
                skipped_this_cycle.clear()
    finally:
        pid_path.unlink(missing_ok=True)

    log.info("Autopilot shut down gracefully")
