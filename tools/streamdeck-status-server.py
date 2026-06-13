#!/usr/bin/env python3
"""HTTP status server for topo-confidence autopilot daemons and pipeline monitoring.

Reads .autopilot/ files and serves JSON on GET /status and GET /pipeline.
Designed for Stream Deck plugin polling. Zero external dependencies.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

AUTOPILOT_DIR = Path(__file__).resolve().parent.parent / ".autopilot"
LINK_FORGE_QUEUE_DB = Path.home() / "link-forge" / "data" / "queue.db"
TRIGGER_FILE = AUTOPILOT_DIR / "trigger"
PHASES = ("triage", "scriptgen", "experiment")
DEFAULT_PORT = 9876
DEFAULT_BUDGET_CAP = 15

DOCKER_CONTAINERS = [
    ("link_forge_neo4j", "link-forge-neo4j"),
    ("research_graph_neo4j", "topo-research-graph"),
    ("ngs_postgres", "node-graph-substrate-postgres-1"),
    ("ngs_redis", "node-graph-substrate-redis-1"),
    ("ngs_server", "node-graph-substrate-server-1"),
]

_neo4j_cache: dict = {"ts": 0, "val": 0}
_NEO4J_CACHE_TTL = 5

_bot_cache: dict = {"ts": 0.0, "alive": False}
_BOT_CACHE_TTL = 5


def _is_bot_alive() -> bool:
    now = time.time()
    if now - _bot_cache["ts"] < _BOT_CACHE_TTL:
        return _bot_cache["alive"]
    try:
        out = subprocess.run(
            ["pgrep", "-f", "tsx src/index"],
            capture_output=True, text=True, timeout=5,
        )
        alive = bool(out.stdout.strip())
    except Exception:
        alive = False
    _bot_cache["ts"] = now
    _bot_cache["alive"] = alive
    return alive

ACTIVITY_PATTERNS = {
    "triage": re.compile(r"Triaging (\S+)"),
    "scriptgen": re.compile(r"(P\d+-FE\d+): generating recompute script"),
    "experiment": re.compile(r"Running experiment (\S+)"),
}


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_phase(phase: str, budget_cap: int) -> dict:
    pid_file = AUTOPILOT_DIR / f"daemon-{phase}.pid"
    budget_file = AUTOPILOT_DIR / f"budget-{phase}.json"
    cap_file = AUTOPILOT_DIR / f"cap-{phase}.json"
    log_file = AUTOPILOT_DIR / f"daemon-{phase}.log"

    # The daemon records its real per-day cap; prefer it over the client-supplied
    # value so an uncapped (or re-capped) daemon is reported honestly.
    if cap_file.exists():
        try:
            budget_cap = int(json.loads(cap_file.read_text()).get("cap", budget_cap))
        except Exception:
            pass

    alive = False
    pid = None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_alive(pid):
                alive = True
            else:
                pid = None
        except (OSError, ValueError):
            pass

    if not alive:
        try:
            out = subprocess.run(
                ["pgrep", "-f", f"autopilot --phase {phase}"],
                capture_output=True, text=True, timeout=5,
            )
            if out.stdout.strip():
                pid = int(out.stdout.strip().splitlines()[0])
                alive = True
        except Exception:
            pass

    budget_used = 0
    budget_date = None
    if budget_file.exists():
        try:
            data = json.loads(budget_file.read_text())
            budget_used = data.get("calls", 0)
            budget_date = data.get("date")
        except Exception:
            pass

    last_activity = None
    current_item = None
    if log_file.exists():
        try:
            lines = _tail(log_file, 100)
            seen = set()
            deduped = []
            for line in lines:
                msg = line.split(" ", 3)[-1] if len(line.split(" ", 3)) >= 4 else line
                if msg not in seen:
                    seen.add(msg)
                    deduped.append(line)

            interesting = [
                l for l in deduped
                if "INFO" in l
                and "sleeping" not in l.lower()
                and "budget exhausted" not in l.lower()
                and "no recompute script found" not in l.lower()
            ]

            if interesting:
                raw = interesting[-1]
                for sep in (" autopilot ", " pipeline.generate_recompute ", " pipeline._match "):
                    if sep in raw:
                        raw = raw.split(sep, 1)[1]
                        break
                raw = re.sub(r"^(INFO|WARNING|DEBUG|ERROR)\s+", "", raw.strip())
                last_activity = raw

                pat = ACTIVITY_PATTERNS.get(phase)
                if pat:
                    m = pat.search(last_activity)
                    if m:
                        current_item = m.group(1)
        except Exception:
            pass

    if not alive:
        state = "stopped"
    elif budget_date == str(date.today()) and budget_used >= budget_cap:
        state = "exhausted"
    elif last_activity and current_item:
        all_lines = []
        if log_file.exists():
            try:
                all_lines = _tail(log_file, 10)
            except Exception:
                pass
        recent_sleep = any(
            "sleeping" in l.lower() or "budget exhausted" in l.lower()
            for l in all_lines if "INFO" in l
        )
        state = "idle" if recent_sleep else "active"
    else:
        state = "idle"

    return {
        "state": state,
        "pid": pid,
        "budget_used": budget_used,
        "budget_cap": budget_cap,
        "current_item": current_item,
        "last_activity": last_activity,
    }


def _tail(path: Path, n: int) -> list[str]:
    """Read last n lines of a file efficiently via seek-from-end."""
    with open(path, "rb") as f:
        try:
            f.seek(0, 2)
            size = f.tell()
        except OSError:
            return []

        chunk = min(size, n * 300)
        f.seek(max(0, size - chunk))
        data = f.read().decode("utf-8", errors="replace")
        return data.strip().splitlines()[-n:]


def _check_link_forge_queue() -> dict:
    stats = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
    if not LINK_FORGE_QUEUE_DB.exists():
        return stats
    try:
        conn = sqlite3.connect(str(LINK_FORGE_QUEUE_DB), timeout=2)
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM queue GROUP BY status"
        ).fetchall()
        conn.close()
        for status, count in rows:
            if status in stats:
                stats[status] = count
    except Exception:
        pass
    return stats


def _check_research_graph_pending() -> int:
    now = time.time()
    if now - _neo4j_cache["ts"] < _NEO4J_CACHE_TTL:
        return _neo4j_cache["val"]
    try:
        out = subprocess.run(
            ["docker", "exec", "topo-research-graph", "cypher-shell",
             "-u", "neo4j", "-p", "topo_graph_dev",
             "MATCH (p:Paper {status: 'pending_triage'}) RETURN count(p)"],
            capture_output=True, text=True, timeout=10,
        )
        for line in reversed(out.stdout.strip().splitlines()):
            line = line.strip()
            if line.isdigit():
                _neo4j_cache["val"] = int(line)
                _neo4j_cache["ts"] = now
                return _neo4j_cache["val"]
    except Exception:
        pass
    return _neo4j_cache["val"]


def _check_trigger_count() -> int:
    if not TRIGGER_FILE.exists():
        return 0
    try:
        text = TRIGGER_FILE.read_text()
        return len([l for l in text.splitlines() if l.strip()])
    except Exception:
        return 0


def _check_services() -> dict:
    services = {}
    for key, container in DOCKER_CONTAINERS:
        try:
            out = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container],
                capture_output=True, text=True, timeout=5,
            )
            services[key] = out.stdout.strip() == "true"
        except Exception:
            services[key] = False

    try:
        out = subprocess.run(
            ["pgrep", "-f", "tsx src/index"],
            capture_output=True, text=True, timeout=5,
        )
        services["link_forge_bot"] = bool(out.stdout.strip())
    except Exception:
        services["link_forge_bot"] = False

    try:
        # Match the NGS Vite by its project dir, not by port: 42-macro-dashboard's
        # frontend squats on 5173, so NGS runs on 5174. A dir-based match detects
        # the NGS frontend on whatever port it lands.
        out = subprocess.run(
            ["pgrep", "-f", "node-graph-substrate/frontend"],
            capture_output=True, text=True, timeout=5,
        )
        services["ngs_frontend"] = bool(out.stdout.strip())
    except Exception:
        services["ngs_frontend"] = False

    return services


def _check_forge_stage() -> dict:
    """Check what stage the currently-processing link-forge item is at via Redis."""
    result = {"stage": "idle", "title": None, "queue_id": None}
    try:
        queue_stats = _check_link_forge_queue()
        if queue_stats["processing"] == 0:
            return result

        conn = sqlite3.connect(str(LINK_FORGE_QUEUE_DB), timeout=2)
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT id, url FROM queue WHERE status = 'processing' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            return result

        qid = str(row[0])
        result["queue_id"] = qid

        out = subprocess.run(
            ["docker", "exec", "node-graph-substrate-redis-1",
             "redis-cli", "hgetall", f"linkforge:paper:{qid}"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            result["stage"] = "processing"
            return result

        lines = out.stdout.strip().splitlines()
        fields = set()
        for i in range(0, len(lines) - 1, 2):
            fields.add(lines[i])

        if "title" in fields:
            idx = lines.index("title")
            if idx + 1 < len(lines):
                result["title"] = lines[idx + 1]

        if "success" in fields or "completed_at" in fields:
            result["stage"] = "done"
        elif "chunk_count" in fields:
            result["stage"] = "bridge"
        elif "stored_done" in fields:
            result["stage"] = "chunk"
        elif "embedding_dim" in fields:
            result["stage"] = "store"
        elif "category" in fields:
            result["stage"] = "embed"
        elif "content_length" in fields:
            result["stage"] = "claude"
        elif "url" in fields:
            result["stage"] = "scrape"
        else:
            result["stage"] = "processing"

    except Exception:
        pass
    return result


def _pipeline_status() -> dict:
    return {
        "link_forge_queue": _check_link_forge_queue(),
        "forge_stage": _check_forge_stage(),
        "research_graph_pending": _check_research_graph_pending(),
        "trigger_count": _check_trigger_count(),
        "services": _check_services(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/ping":
            self._json_response({"ok": True})
            return

        if parsed.path == "/health":
            bot_up = _is_bot_alive()
            self._json_response(
                {"ok": bot_up, "bot": bot_up},
                status=200 if bot_up else 503,
            )
            return

        if parsed.path == "/status":
            qs = parse_qs(parsed.query)
            budget_cap = int(qs.get("budget_cap", [str(DEFAULT_BUDGET_CAP)])[0])
            daemons = {}
            for phase in PHASES:
                daemons[phase] = _check_phase(phase, budget_cap)
            self._json_response({"daemons": daemons})
            return

        if parsed.path == "/pipeline":
            self._json_response(_pipeline_status())
            return

        self.send_error(404)

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
            elif arg == "--port" and sys.argv.index(arg) + 1 < len(sys.argv):
                port = int(sys.argv[sys.argv.index(arg) + 1])

    if not AUTOPILOT_DIR.exists():
        print(f"Error: {AUTOPILOT_DIR} does not exist", file=sys.stderr)
        sys.exit(1)

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Daemon status server on 0.0.0.0:{port} (reading {AUTOPILOT_DIR})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
