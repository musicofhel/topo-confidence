#!/usr/bin/env bash
# Watchdog for triage_pending.sh: kills any direct `claude -p` child older than
# 30 minutes. The parent loop sees the non-zero exit and continues. Per-paper
# crashes were already handled by `if claude -p ...; then ... else fi`; this
# adds protection against hangs.
#
# Exits cleanly when the parent triage_pending.sh is no longer running.
#
# Usage:
#   bash triage_watchdog.sh > triage_watchdog.log 2>&1 &

set -u

MAX_SECONDS="${MAX_SECONDS:-1800}"   # 30 min
POLL_SECONDS="${POLL_SECONDS:-60}"

log() { echo "$(date -Iseconds) $*"; }

log "watchdog start  MAX_SECONDS=$MAX_SECONDS  POLL_SECONDS=$POLL_SECONDS"

while :; do
  parent_pid="$(pgrep -f 'bash triage_pending.sh' | head -1 || true)"
  if [[ -z "$parent_pid" ]]; then
    log "watchdog stop — parent triage_pending.sh not found"
    exit 0
  fi

  # Direct children of the loop named "claude" with elapsed time > MAX_SECONDS.
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    etime="$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    log "KILL stale claude pid=$pid etime=${etime}s — sending SIGTERM"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
      log "KILL pid=$pid still alive — SIGKILL"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done < <(ps -eo pid,ppid,etimes,comm | awk -v pp="$parent_pid" -v max="$MAX_SECONDS" \
            '$2==pp && $3>max && $4=="claude" {print $1}')

  sleep "$POLL_SECONDS"
done
