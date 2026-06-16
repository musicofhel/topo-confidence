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
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

AUTOPILOT_DIR = Path(__file__).resolve().parent.parent / ".autopilot"
LINK_FORGE_QUEUE_DB = Path.home() / "link-forge" / "data" / "queue.db"
TRIGGER_FILE = AUTOPILOT_DIR / "trigger"
PHASES = ("triage", "scriptgen", "experiment")
DEFAULT_PORT = 9876

# The Stream Deck "pause script gen" button holds the token-spending phases
# (scriptgen + experiment) while leaving triage running, so papers keep getting
# digested into the FE queue. Pause = write a `paused-<phase>` flag the daemon
# checks each cycle; resume = remove it. No systemctl, no mid-item interruption.
PAUSE_PHASES = ("scriptgen", "experiment")


def _pause_flag(phase: str) -> Path:
    return AUTOPILOT_DIR / f"paused-{phase}"


def _scriptgen_paused() -> bool:
    return any(_pause_flag(p).exists() for p in PAUSE_PHASES)


def _set_scriptgen_paused(paused: bool) -> bool:
    """Write/remove the pause flags for the token-spending phases. Returns the
    resulting paused state. Best-effort, idempotent."""
    for p in PAUSE_PHASES:
        flag = _pause_flag(p)
        try:
            if paused:
                flag.write_text("")
            elif flag.exists():
                flag.unlink()
        except OSError:
            pass
    return _scriptgen_paused()


_ready_fe_cache: dict = {"ts": 0.0, "val": None}
_READY_FE_CACHE_TTL = 30


def _ready_fe_count() -> int | None:
    """READY/TRIGGERED FutureExperiment backlog — the queue that grows while
    script gen is paused. Cached 30s; None if Neo4j is unreachable."""
    now = time.time()
    if now - _ready_fe_cache["ts"] < _READY_FE_CACHE_TTL:
        return _ready_fe_cache["val"]
    val = None
    try:
        out = subprocess.run(
            ["docker", "exec", "topo-research-graph", "cypher-shell",
             "-u", "neo4j", "-p", "topo_graph_dev",
             "MATCH (fe:FutureExperiment) WHERE fe.status IN ['READY','TRIGGERED'] RETURN count(fe)"],
            capture_output=True, text=True, timeout=10,
        )
        for line in reversed(out.stdout.strip().splitlines()):
            line = line.strip()
            if line.isdigit():
                val = int(line)
                break
    except Exception:
        pass
    _ready_fe_cache["ts"] = now
    _ready_fe_cache["val"] = val
    return val


def _button_scriptgen() -> str:
    """Plain-text title for the pause button. Reflects current state + backlog."""
    paused = _scriptgen_paused()
    q = _ready_fe_count()
    qline = f"q:{q}" if q is not None else "q:?"
    glyph = "⏸ PAUSED" if paused else "▶ RUNNING"
    return f"SCRIPT GEN\n{glyph}\n{qline}"

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

# /pipeline shells out to docker inspect/exec serially (~2.4s). The Stream Deck
# disconnects after ~1-2s, which surfaced as a BrokenPipeError mid-response and a
# frozen button. A background thread keeps this snapshot warm so client polls are
# served instantly from cache and never pay the docker cost on the request path.
_pipeline_cache: dict = {"ts": 0.0, "val": None}
_pipeline_lock = threading.Lock()
_PIPELINE_REFRESH_INTERVAL = 2.0


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


def _check_phase(phase: str) -> dict:
    pid_file = AUTOPILOT_DIR / f"daemon-{phase}.pid"
    log_file = AUTOPILOT_DIR / f"daemon-{phase}.log"

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
    elif last_activity and current_item:
        all_lines = []
        if log_file.exists():
            try:
                all_lines = _tail(log_file, 10)
            except Exception:
                pass
        recent_sleep = any(
            "sleeping" in l.lower()
            for l in all_lines if "INFO" in l
        )
        state = "idle" if recent_sleep else "active"
    else:
        state = "idle"

    return {
        "state": state,
        "pid": pid,
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


COMPLETED_FILE = AUTOPILOT_DIR / "completed_experiments.json"


def _completed_count_and_age() -> tuple[int, float | None]:
    """How many FEs the experiment phase has marked done, and how long ago the
    last one landed (seconds). The completed-marker file is rewritten on every
    successful run, so its mtime is a precise "is it still finishing FEs?" signal."""
    if not COMPLETED_FILE.exists():
        return 0, None
    try:
        count = len(json.loads(COMPLETED_FILE.read_text()))
    except Exception:
        count = 0
    try:
        age = time.time() - COMPLETED_FILE.stat().st_mtime
    except OSError:
        age = None
    return count, age


def _button_experiment() -> str:
    """Compact plain-text summary of the experiment phase for a Stream Deck title.

    Three short lines:  <glyph + phase> / <current-or-last FE> / done <count>
      ✅ a run finished in the last 2 min (actively working through the queue)
      💤 alive but no recent completion (queue drained / between polls)
      ⛔ daemon stopped or Neo4j unreachable (can't pick FEs)
    """
    phase = _check_phase("experiment")
    count, age = _completed_count_and_age()

    neo4j_up = False
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", "topo-research-graph"],
            capture_output=True, text=True, timeout=5,
        )
        neo4j_up = out.stdout.strip() == "true"
    except Exception:
        pass

    if phase["state"] == "stopped" or not neo4j_up:
        glyph = "⛔"
    elif age is not None and age < 120:
        glyph = "✅"
    else:
        glyph = "💤"

    fe = phase.get("current_item")
    if not fe:
        # fall back to the FE named in the last activity line ("Auto-committed ... P11-FE155")
        la = phase.get("last_activity") or ""
        m = re.search(r"(P\d+-FE\d+)", la)
        fe = m.group(1) if m else "—"

    return f"EXP {glyph}\n{fe}\ndone {count}"


def _pipeline_status() -> dict:
    return {
        "link_forge_queue": _check_link_forge_queue(),
        "forge_stage": _check_forge_stage(),
        "research_graph_pending": _check_research_graph_pending(),
        "trigger_count": _check_trigger_count(),
        "services": _check_services(),
    }


def _refresh_pipeline_loop(interval: float = _PIPELINE_REFRESH_INTERVAL):
    """Recompute the (slow, docker-bound) pipeline snapshot on a fixed interval so
    GET /pipeline can be served instantly from cache — no client poll ever waits on
    docker, so the Stream Deck never times out and BrokenPipes mid-response."""
    while True:
        try:
            snap = _pipeline_status()
            with _pipeline_lock:
                _pipeline_cache["val"] = snap
                _pipeline_cache["ts"] = time.time()
        except Exception:
            pass
        time.sleep(interval)


def _get_pipeline_cached() -> dict:
    with _pipeline_lock:
        val = _pipeline_cache["val"]
    if val is None:
        # cold start before the refresher's first pass completes — compute once
        val = _pipeline_status()
        with _pipeline_lock:
            if _pipeline_cache["val"] is None:
                _pipeline_cache["val"] = val
                _pipeline_cache["ts"] = time.time()
    return val


# ---------------------------------------------------------------------------
# confgate paper-triage button
#
# A separate Stream Deck key for the confgate repo's research-graph (a STANDALONE
# Neo4j on bolt:7689, not the topo docker graph). "Pending" = papers admitted as
# :Paper{status:'pending_triage'} that DON'T YET HAVE A BRIEF — i.e. the queue of
# papers still to be deep-triaged. Pressing the key launches triage_pending.sh,
# which spawns one `claude -p` worker per paper and writes a brief each; as briefs
# land the queue drains to 0. The button polls /confgate-state and triggers
# /confgate-digest.
# ---------------------------------------------------------------------------
CONFGATE_DIR = Path.home() / "confgate" / "research-graph"
CONFGATE_BRIEFS = CONFGATE_DIR / "briefs"

_confgate_cache: dict = {"ts": 0.0, "val": None}
_CONFGATE_CACHE_TTL = 20


def _confgate_python() -> str:
    """query.py needs the neo4j driver. Prefer the confgate venv, else the topo
    venv, else system python3 (verified to have neo4j on this box)."""
    for cand in (
        Path.home() / "confgate" / ".venv" / "bin" / "python",
        Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    return "python3"


def _confgate_pending_count() -> int | None:
    """Count :Paper{status:'pending_triage'} in the confgate graph that don't yet
    have a brief on disk (briefs/triage-*-<id>.md). That's the to-digest queue —
    it drops to 0 as triage writes briefs. Cached 20s; None if unreachable."""
    now = time.time()
    if now - _confgate_cache["ts"] < _CONFGATE_CACHE_TTL:
        return _confgate_cache["val"]
    val = None
    try:
        out = subprocess.run(
            [_confgate_python(), "query.py", "pending", "--ids-only"],
            cwd=str(CONFGATE_DIR), capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            ids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            briefed = set()
            if CONFGATE_BRIEFS.is_dir():
                for b in CONFGATE_BRIEFS.glob("triage-*-*.md"):
                    # filename: triage-YYYY-MM-DD-<arxiv-id>.md
                    stem = b.name[len("triage-"):-len(".md")]
                    parts = stem.split("-", 3)
                    if len(parts) == 4:
                        briefed.add(parts[3])
            val = sum(1 for i in ids if i not in briefed)
    except Exception:
        pass
    _confgate_cache["ts"] = now
    _confgate_cache["val"] = val
    return val


def _confgate_digesting() -> bool:
    """True while a triage dispatcher or any per-paper worker is running."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", r"triage_(pending|one)\.sh"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _confgate_launch_env() -> dict:
    """Env for the spawned triage. This server runs under a systemd user unit
    with the minimal default PATH (no ~/.nvm bin), so a bare `claude` in
    triage_one.sh would be command-not-found (exit 127). Prepend any nvm node
    bin dir that actually contains `claude` so the workers can find it."""
    env = dict(os.environ)
    # `claude -p` refuses to start inside an existing Claude session; the script
    # unsets this itself, but strip it here too so a server launched from a
    # session can't leak it in.
    env.pop("CLAUDECODE", None)
    extra = []
    for cand in sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin"), reverse=True):
        if (cand / "claude").exists():
            extra.append(str(cand))
            break
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.is_dir():
        extra.append(str(local_bin))
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _confgate_start_digest() -> bool:
    """Launch triage_pending.sh detached if not already digesting. Returns the
    resulting digesting state. Idempotent — a second press while running is a
    no-op (the dispatcher + triage_one skip papers that already have a brief)."""
    if _confgate_digesting():
        return True
    script = CONFGATE_DIR / "triage_pending.sh"
    if not script.exists():
        return False
    env = _confgate_launch_env()
    try:
        CONFGATE_BRIEFS.mkdir(parents=True, exist_ok=True)
        logf = open(CONFGATE_DIR / "triage_pending.log", "a")
        subprocess.Popen(
            ["bash", str(script)],
            cwd=str(CONFGATE_DIR), env=env,
            stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
            start_new_session=True,  # detach: survives this request/handler
        )
    except Exception:
        return False
    _confgate_cache["ts"] = 0.0  # force the next /confgate-state to re-poll
    return True


def _confgate_state() -> dict:
    digesting = _confgate_digesting()
    return {
        "pending": _confgate_pending_count(),
        "digesting": digesting,
        "reachable": True,
    }


class Handler(BaseHTTPRequestHandler):
    def _handle_action(self, path: str) -> bool:
        """Pause-button actions. Accept on GET and POST so any Stream Deck
        web-request plugin works. Each returns the button title text so the key
        updates immediately on press. Returns True if it handled the path."""
        if path == "/pause-scriptgen":
            _set_scriptgen_paused(True)
            self._text_response(_button_scriptgen())
            return True
        if path == "/resume-scriptgen":
            _set_scriptgen_paused(False)
            self._text_response(_button_scriptgen())
            return True
        if path == "/toggle-scriptgen":
            _set_scriptgen_paused(not _scriptgen_paused())
            self._text_response(_button_scriptgen())
            return True
        if path == "/button-scriptgen":
            self._text_response(_button_scriptgen())
            return True
        if path == "/confgate-digest":
            _confgate_start_digest()
            self._json_response(_confgate_state())
            return True
        return False

    def do_POST(self):
        parsed = urlparse(self.path)
        if self._handle_action(parsed.path):
            return
        self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/ping":
            self._json_response({"ok": True})
            return

        if self._handle_action(parsed.path):
            return

        if parsed.path == "/health":
            bot_up = _is_bot_alive()
            self._json_response(
                {"ok": bot_up, "bot": bot_up},
                status=200 if bot_up else 503,
            )
            return

        if parsed.path == "/status":
            daemons = {}
            for phase in PHASES:
                daemons[phase] = _check_phase(phase)
            # The experiment phase has no Neo4j "done" state; expose how many FEs
            # it has marked complete and how long ago the last one landed so the
            # Stream Deck can show live FE progress (count ticking up == working).
            if "experiment" in daemons:
                cnt, age = _completed_count_and_age()
                daemons["experiment"]["completed_count"] = cnt
                daemons["experiment"]["completed_age"] = age
            self._json_response({"daemons": daemons})
            return

        if parsed.path == "/pipeline":
            self._json_response(_get_pipeline_cached())
            return

        if parsed.path == "/button":
            # Plain-text, ready to drop straight into a Stream Deck button title.
            self._text_response(_button_experiment())
            return

        if parsed.path == "/scriptgen-state":
            # JSON the Stream Deck "Pause Script Gen" plugin action polls.
            self._json_response({
                "paused": _scriptgen_paused(),
                "ready_fes": _ready_fe_count(),
            })
            return

        if parsed.path == "/confgate-state":
            # JSON the Stream Deck "confgate" plugin action polls.
            self._json_response(_confgate_state())
            return

        self.send_error(404)

    def _text_response(self, text: str, status: int = 200):
        body = text.encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionError):
            pass

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionError):
            # client (Stream Deck) hung up before we finished writing — harmless
            pass

    def log_message(self, fmt, *args):
        # Lightweight request log so we can SEE whether the Stream Deck (or any
        # client) actually reaches the server, and on which path. Writes one line
        # per request to a dedicated file; failures are swallowed (never block a
        # response on logging).
        try:
            line = "%s %s %s\n" % (
                time.strftime("%H:%M:%S"),
                self.client_address[0] if self.client_address else "?",
                (fmt % args) if args else fmt,
            )
            with open(AUTOPILOT_DIR / "status-requests.log", "a") as f:
                f.write(line)
        except Exception:
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

    # Keep the slow /pipeline snapshot warm off the request path.
    threading.Thread(target=_refresh_pipeline_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Daemon status server on 0.0.0.0:{port} (reading {AUTOPILOT_DIR})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
