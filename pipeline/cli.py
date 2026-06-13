"""CLI entry point for the experiment pipeline.

Usage:
    python -m pipeline run                     # run next experiment from queue
    python -m pipeline run --fe P11-FE101      # run specific FE
    python -m pipeline run --dry-run            # generate artifacts without promoting
    python -m pipeline resume                   # resume from last checkpoint
    python -m pipeline status                   # show current pipeline state
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import date

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pipeline.observability import init_tracing

_console = Console()


# ---------------------------------------------------------------------------
# Shared interrupt loop
# ---------------------------------------------------------------------------


def _run_interrupt_loop(graph, config, result) -> dict:
    """Handle the review-gate interrupt loop. Used by both run and resume."""
    from langgraph.types import Command

    while "__interrupt__" in str(result):
        state = graph.get_state(config)
        if not state.tasks:
            break

        _console.rule("[bold cyan]Human Review Gate[/bold cyan]")

        payload_data = None
        for task in state.tasks:
            if hasattr(task, "interrupts"):
                for intr in task.interrupts:
                    payload_data = intr.value
                    _display_review(payload_data)

        if payload_data is None:
            break

        verdict = _prompt_verdict(payload_data)
        if verdict == "q":
            _console.print("[yellow]Pipeline stopped by user.[/yellow]")
            return {"__quit__": True}

        resume_value: dict = {"verdict": verdict}
        if verdict == "edit":
            resume_value["edits"] = _prompt_edits(payload_data)

        result = graph.invoke(Command(resume=resume_value), config)

    return result


def _print_summary(result: dict) -> None:
    completed = result.get("completed_this_session", [])
    errors = result.get("errors", [])
    _console.rule()
    _console.print(f"Pipeline complete. [bold]{len(completed)}[/bold] experiments promoted.")
    if completed:
        _console.print(f"  Completed: {', '.join(completed)}")
    if errors:
        _console.print(f"  [red]Errors ({len(errors)}):[/red]")
        for e in errors:
            _console.print(
                f"    - {e.get('fe_id', '?')}: {e.get('step', '?')} "
                f"(exit {e.get('exit_code', '?')})"
            )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_review(payload: dict) -> None:
    fe_id = payload.get("fe_id", "?")
    desc = payload.get("description", "")
    dry_run = payload.get("dry_run", False)

    title = f"REVIEW: {fe_id}"
    if dry_run:
        title += " [DRY RUN]"
    _console.print(Panel(desc or "No description", title=title, border_style="cyan"))

    brief = payload.get("brief_markdown", "")
    if brief:
        lines = brief.split("\n")
        preview = "\n".join(lines[:40])
        _console.print(Panel(preview, title="Brief Preview (first 40 lines)", border_style="green"))
        if len(lines) > 40:
            _console.print(
                f"  [dim]... {len(lines)} total lines. "
                f"Press [bold]v[/bold] to view full brief in pager.[/dim]"
            )
    _console.print()

    claims = payload.get("claims", [])
    if claims:
        table = Table(title=f"Claims ({len(claims)})")
        table.add_column("cid", style="cyan", no_wrap=True)
        table.add_column("description", style="white")
        table.add_column("expected", justify="right", style="green")
        table.add_column("tol", justify="right", style="dim")
        for c in claims:
            table.add_row(
                c.get("cid", "?"),
                c.get("description", "")[:80],
                str(c.get("expected", "?")),
                str(c.get("tol", "?")),
            )
        _console.print(table)
    else:
        _console.print("  [dim]No claims extracted.[/dim]")

    updates = payload.get("findings_updates", [])
    if updates:
        table = Table(title=f"Findings Updates ({len(updates)})")
        table.add_column("finding_id", style="cyan", no_wrap=True)
        table.add_column("strength", style="yellow")
        table.add_column("status", style="yellow")
        table.add_column("summary", style="white")
        for u in updates:
            table.add_row(
                u.get("finding_id", "?"),
                u.get("strength", "?"),
                u.get("status", "?"),
                u.get("summary", "")[:100],
            )
        _console.print(table)
    else:
        _console.print("  [dim]No findings updates proposed.[/dim]")
    _console.print()


def _view_full_brief(payload: dict) -> None:
    brief = payload.get("brief_markdown", "")
    if not brief:
        _console.print("  [dim]No brief content available.[/dim]")
        return
    with _console.pager(styles=True):
        _console.print(brief)


# ---------------------------------------------------------------------------
# Verdict + edit prompts
# ---------------------------------------------------------------------------


def _prompt_verdict(payload: dict) -> str:
    while True:
        choice = input("  [a]pprove  [r]eject  [e]dit  [v]iew full  [q]uit > ").strip().lower()
        if choice in ("a", "approve"):
            return "approve"
        if choice in ("r", "reject"):
            return "reject"
        if choice in ("e", "edit"):
            return "edit"
        if choice in ("v", "view"):
            _view_full_brief(payload)
            continue
        if choice in ("q", "quit"):
            return "q"
        _console.print("  [red]Invalid choice. Try again.[/red]")


def _prompt_edits(payload: dict) -> dict:
    """Open $EDITOR with current artifacts for human editing."""
    brief = payload.get("brief_markdown", "")
    claims = payload.get("claims", [])
    updates = payload.get("findings_updates", [])

    doc = _build_edit_document(brief, claims, updates)
    edited = _open_in_editor(doc)
    if edited is None:
        _console.print("  [dim]No edits made.[/dim]")
        return {}

    return _parse_edit_document(edited, brief, claims, updates)


def _build_edit_document(brief: str, claims: list, updates: list) -> str:
    lines = [
        "# Review Edits",
        "#",
        "# Edit the sections below. Save and close to apply changes.",
        "# Leave a section unchanged to keep the original.",
        "",
        "## BRIEF",
        "",
        brief,
        "",
        "## CLAIMS",
        "",
        "```json",
        json.dumps(claims, indent=2, default=str),
        "```",
        "",
        "## FINDINGS_UPDATES",
        "",
        "```json",
        json.dumps(updates, indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def _open_in_editor(content: str) -> str | None:
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="pipeline-review-", delete=False
    ) as f:
        f.write(content)
        tmppath = f.name

    try:
        subprocess.run([editor, tmppath], check=True)
        with open(tmppath) as f:
            edited = f.read()
        if edited.strip() == content.strip():
            return None
        return edited
    except subprocess.CalledProcessError:
        _console.print("  [red]Editor exited with error.[/red]")
        return None
    finally:
        os.unlink(tmppath)


def _parse_edit_document(
    edited: str, original_brief: str, original_claims: list, original_updates: list
) -> dict:
    patches: dict = {}

    brief_match = re.search(
        r"## BRIEF\n\n?(.*?)(?=\n## CLAIMS)", edited, re.DOTALL
    )
    if brief_match:
        new_brief = brief_match.group(1).strip()
        if new_brief and new_brief != original_brief.strip():
            patches["brief_patch"] = new_brief

    claims_match = re.search(r"## CLAIMS.*?```json\n(.*?)```", edited, re.DOTALL)
    if claims_match:
        try:
            new_claims = json.loads(claims_match.group(1).strip())
            if new_claims != original_claims:
                patches["claims_patch"] = new_claims
        except json.JSONDecodeError:
            _console.print(
                "  [yellow]WARNING: Could not parse claims JSON, keeping original.[/yellow]"
            )

    findings_match = re.search(
        r"## FINDINGS_UPDATES.*?```json\n(.*?)```", edited, re.DOTALL
    )
    if findings_match:
        try:
            new_updates = json.loads(findings_match.group(1).strip())
            if new_updates != original_updates:
                patches["findings_patch"] = new_updates
        except json.JSONDecodeError:
            _console.print(
                "  [yellow]WARNING: Could not parse findings JSON, keeping original.[/yellow]"
            )

    return patches


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    init_tracing(
        enable_langfuse=not args.no_langfuse,
        enable_jaeger=not args.no_jaeger,
    )

    from langgraph.checkpoint.sqlite import SqliteSaver

    from pipeline.graph import build_graph

    with SqliteSaver.from_conn_string("pipeline_state.db") as checkpointer:
        no_review = getattr(args, "no_review", False)
        graph = build_graph(checkpointer=checkpointer, no_review=no_review)

        thread_id = f"session-{date.today()}"
        if args.fe:
            thread_id = f"single-{args.fe}-{date.today()}"

        config = {"configurable": {"thread_id": thread_id}}

        max_runs = getattr(args, "max_runs", 0) or 0
        local_only = getattr(args, "local", False)

        initial_state = {
            "current_fe": None,
            "completed_this_session": [],
            "errors": [],
            "dry_run": args.dry_run,
            "local_only": local_only,
            "max_runs": max_runs,
        }

        _console.print(f"Starting pipeline (thread: [cyan]{thread_id}[/cyan])")
        if no_review:
            _console.print("[yellow]NO REVIEW — skipping human review gate[/yellow]")
        if args.dry_run:
            _console.print("[yellow]DRY RUN — will generate artifacts but skip promotion[/yellow]")
        if local_only:
            _console.print("[yellow]LOCAL ONLY — skipping FEs that require cloud GPU[/yellow]")
        if max_runs > 0:
            _console.print(f"[yellow]MAX RUNS: {max_runs}[/yellow]")

        result = graph.invoke(initial_state, config)

        if not no_review:
            result = _run_interrupt_loop(graph, config, result)
            if result.get("__quit__"):
                return

        _print_summary(result)


def cmd_resume(args: argparse.Namespace) -> None:
    init_tracing(
        enable_langfuse=not args.no_langfuse,
        enable_jaeger=not args.no_jaeger,
    )

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    from pipeline.graph import build_graph

    with SqliteSaver.from_conn_string("pipeline_state.db") as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        thread_id = args.thread or f"session-{date.today()}"
        config = {"configurable": {"thread_id": thread_id}}

        state = graph.get_state(config)
        if state is None or state.values is None:
            _console.print(f"[red]No checkpoint found for thread '{thread_id}'[/red]")
            return

        _console.print(f"Resuming thread '[cyan]{thread_id}[/cyan]'")
        _console.print(
            f"  Completed so far: {state.values.get('completed_this_session', [])}"
        )

        has_interrupt = False
        payload_data = None
        if state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    has_interrupt = True
                    for intr in task.interrupts:
                        payload_data = intr.value

        if has_interrupt and payload_data is not None:
            _console.rule("[bold cyan]Pending Review[/bold cyan]")
            _display_review(payload_data)

            verdict = _prompt_verdict(payload_data)
            if verdict == "q":
                _console.print("[yellow]Resume cancelled.[/yellow]")
                return

            resume_value: dict = {"verdict": verdict}
            if verdict == "edit":
                resume_value["edits"] = _prompt_edits(payload_data)

            result = graph.invoke(Command(resume=resume_value), config)
            result = _run_interrupt_loop(graph, config, result)

            if result.get("__quit__"):
                return
            _print_summary(result)
        else:
            _console.print("  No pending review. Continuing from checkpoint...")
            result = graph.invoke(None, config)
            result = _run_interrupt_loop(graph, config, result)
            if result.get("__quit__"):
                return
            _print_summary(result)


def cmd_status(args: argparse.Namespace) -> None:
    _console.print("Pipeline status:")
    _console.print(f"  Checkpoint DB: pipeline_state.db")

    try:
        from pipeline.nodes import _find_recompute_script, _is_local_runnable, _query_neo4j_ready_fes

        fes = _query_neo4j_ready_fes()
        with_scripts = [fe for fe in fes if _find_recompute_script(fe["id"])]
        local_runnable = [fe for fe in with_scripts if _is_local_runnable(fe)]

        _console.print(f"  Ready/Triggered FEs in Neo4j: {len(fes)}")
        _console.print(f"  With recompute scripts: {len(with_scripts)}")
        _console.print(f"  Local-runnable (2060S + CPU): [bold]{len(local_runnable)}[/bold]")

        if len(local_runnable) >= 5:
            _console.print(f"  [green]Batch gate: PASS ({len(local_runnable)} >= 5)[/green]")
        else:
            _console.print(f"  [yellow]Batch gate: WAITING ({len(local_runnable)} < 5 needed)[/yellow]")

        if local_runnable:
            _console.print("  Top 5 local by ROI:")
            for fe in local_runnable[:5]:
                _console.print(
                    f"    {fe['id']} (ROI: {fe.get('roi_score', '?')}, "
                    f"cost: {fe.get('estimated_cost', '?')}, "
                    f"status: {fe.get('status', '?')})"
                )
    except Exception as e:
        _console.print(f"  Neo4j unavailable: {e}")


def cmd_autopilot_status(args: argparse.Namespace) -> None:
    import json
    import os
    from pathlib import Path

    ap_dir = Path(".autopilot")
    if not ap_dir.exists():
        _console.print("[red]No .autopilot directory found[/red]")
        return

    _console.print("[bold]Autopilot worker status[/bold]\n")

    for phase in ("triage", "scriptgen", "experiment"):
        pid_file = ap_dir / f"daemon-{phase}.pid"
        budget_file = ap_dir / f"budget-{phase}.json"
        log_file = ap_dir / f"daemon-{phase}.log"

        alive = False
        pid = None
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                alive = True
            except (OSError, ValueError):
                pass
        if not alive:
            import subprocess as _sp
            try:
                out = _sp.run(["pgrep", "-f", f"autopilot --phase {phase}"],
                              capture_output=True, text=True, timeout=5)
                if out.stdout.strip():
                    pid = int(out.stdout.strip().splitlines()[0])
                    alive = True
            except Exception:
                pass

        status = f"[green]RUNNING (PID {pid})[/green]" if alive else "[red]STOPPED[/red]"

        budget_str = "no data"
        if budget_file.exists():
            try:
                data = json.loads(budget_file.read_text())
                budget_str = f"{data.get('calls', 0)} calls used ({data.get('date', '?')})"
            except Exception:
                pass

        last_log = "no log"
        if log_file.exists():
            try:
                lines = log_file.read_text().strip().splitlines()
                interesting = [l for l in lines if "INFO" in l and "sleeping" not in l.lower() and "budget exhausted" not in l.lower()]
                if interesting:
                    last_log = interesting[-1].split(" autopilot ")[1] if " autopilot " in interesting[-1] else interesting[-1]
            except Exception:
                pass

        _console.print(f"  [bold]{phase:12s}[/bold] {status}")
        _console.print(f"               Budget: {budget_str}")
        _console.print(f"               Last: {last_log}")
        _console.print()

    _console.print("[bold]Queue depths[/bold]\n")
    try:
        from pipeline.autopilot import _pending_papers, _runnable_fes
        from pipeline.generate_recompute import scriptless_local_fes

        pending = _pending_papers()
        scriptless = scriptless_local_fes()
        runnable = _runnable_fes()
        _console.print(f"  Pending triage:  {len(pending)} papers")
        _console.print(f"  Scriptless FEs:  {len(scriptless)}")
        _console.print(f"  Runnable FEs:    {len(runnable)}")
    except Exception as e:
        _console.print(f"  [red]Neo4j query failed: {e}[/red]")

    failures_file = ap_dir / "failures.json"
    if failures_file.exists():
        try:
            failures = json.loads(failures_file.read_text())
            quarantined = {k: v for k, v in failures.items() if v > 2}
            if quarantined:
                _console.print(f"\n  [yellow]Quarantined ({len(quarantined)}):[/yellow]")
                for k, v in list(quarantined.items())[:10]:
                    _console.print(f"    {k}: {v} failures")
        except Exception:
            pass


def cmd_audit_fes(args: argparse.Namespace) -> None:
    """Cross-check every FE's attributed script against actual script paths."""
    from pipeline._match import find_recompute_script, parse_fe_num
    from pipeline.nodes import _query_neo4j_ready_fes

    fes = _query_neo4j_ready_fes()
    _console.print(f"Auditing {len(fes)} FEs...")
    mismatches = []
    found = 0

    for fe in fes:
        fe_id = fe["id"]
        script = find_recompute_script(fe_id)
        if script is None:
            continue
        found += 1
        fe_num = parse_fe_num(fe_id)
        num = fe_num[2:] if fe_num.startswith("fe") else fe_num
        stem_num = script.stem.replace("recompute_", "").lower()
        if stem_num != num and stem_num != fe_num:
            mismatches.append((fe_id, str(script), fe_num, stem_num))

    _console.print(f"  Scripts found: {found}/{len(fes)}")
    if mismatches:
        _console.print(f"\n[red]MISMATCHES FOUND ({len(mismatches)}):[/red]")
        for fe_id, path, expected, got in mismatches:
            _console.print(f"  {fe_id}: expected {expected}, script has {got} ({path})")
    else:
        _console.print("[green]All FE-to-script mappings verified.[/green]")


def cmd_autopilot(args: argparse.Namespace) -> None:
    import logging
    from pathlib import Path

    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    phase = getattr(args, "phase", None)
    if phase:
        log_dir = Path(".autopilot")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_dir / f"daemon-{phase}.log"))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        # The daemon is launched with `>> daemon-{phase}.log 2>&1`, so the root
        # StreamHandler's stderr output already lands in this file. Without
        # propagate=False every record would be written twice (once by fh, once
        # via root->stderr->redirect) — which doubled the log's growth.
        for name in ("pipeline", "autopilot"):
            lg = logging.getLogger(name)
            lg.addHandler(fh)
            lg.propagate = False

    from pipeline.autopilot import run_loop

    _console.print(f"[bold]Autopilot daemon starting[/bold]")
    _console.print(f"  Phase: {phase or 'all (sequential)'}")
    _console.print(f"  Budget: {args.budget} calls/day")
    _console.print(f"  Poll interval: {args.poll}s")
    if args.dry_run:
        _console.print("[yellow]DRY RUN — logging actions only[/yellow]")

    run_loop(
        poll_interval=args.poll,
        daily_cap=args.budget,
        dry_run=args.dry_run,
        phase=phase,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline", description="Experiment pipeline orchestrator"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-langfuse", action="store_true", help="Disable Langfuse tracing")
        p.add_argument("--no-jaeger", action="store_true", help="Disable Jaeger tracing")

    run_parser = sub.add_parser("run", help="Run experiments from queue")
    run_parser.add_argument("--fe", help="Run specific FE ID instead of top-of-queue")
    run_parser.add_argument("--dry-run", action="store_true", help="Skip promotion")
    run_parser.add_argument("--local", action="store_true", help="Only run FEs that fit on local GPU (2060 Super + CPU)")
    run_parser.add_argument("--max-runs", type=int, default=0, help="Stop after N experiments (0 = unlimited)")
    run_parser.add_argument("--no-review", action="store_true", help="Skip human review gate (autopilot mode)")
    _add_common(run_parser)
    run_parser.set_defaults(func=cmd_run)

    resume_parser = sub.add_parser("resume", help="Resume from checkpoint")
    resume_parser.add_argument("--thread", help="Thread ID to resume")
    _add_common(resume_parser)
    resume_parser.set_defaults(func=cmd_resume)

    status_parser = sub.add_parser("status", help="Show pipeline status")
    _add_common(status_parser)
    status_parser.set_defaults(func=cmd_status)

    autopilot_parser = sub.add_parser("autopilot", help="Run autopilot daemon")
    autopilot_parser.add_argument("--phase", choices=["triage", "scriptgen", "experiment"],
                                  default=None, help="Run only this phase (for multi-worker deployment)")
    autopilot_parser.add_argument("--budget", type=int, default=1000000000, help="Daily LLM call cap (default: effectively uncapped)")
    autopilot_parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    autopilot_parser.add_argument("--dry-run", action="store_true", help="Log actions, don't execute")
    autopilot_parser.add_argument("--verbose", action="store_true", help="Debug logging")
    _add_common(autopilot_parser)
    autopilot_parser.set_defaults(func=cmd_autopilot)

    ap_status_parser = sub.add_parser("autopilot-status", help="Show autopilot worker status")
    ap_status_parser.set_defaults(func=cmd_autopilot_status)

    audit_parser = sub.add_parser("audit-fes", help="Cross-check FE-to-script mappings")
    audit_parser.set_defaults(func=cmd_audit_fes)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
