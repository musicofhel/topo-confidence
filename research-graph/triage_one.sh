#!/usr/bin/env bash
# Worker: deep-triage a single paper. Called by triage_pending.sh, one process
# per paper, runnable in parallel.
#
# Usage:
#   bash triage_one.sh <arxiv-id>
#
# Reads the prompt template from _paper_triage_prompt.template, substitutes
# arxiv id + today's date, calls `claude -p --dangerously-skip-permissions` with
# stdin closed (so it can't consume the dispatcher's input stream), and wraps
# claude in `timeout 1800` so a hung agent can't pin a worker slot indefinitely.
#
# Skips work if a brief for this arxiv_id already exists from any date.
# Per-paper log lands at briefs/triage-YYYY-MM-DD-<arxiv-id>.md.log.

set -uo pipefail

cd "$(dirname "$0")"

arxiv_id="${1:?usage: triage_one.sh <arxiv-id>}"

TODAY="$(date +%Y-%m-%d)"
TEMPLATE_PATH="./_paper_triage_prompt.template"
TIMEOUT_SECONDS="${TRIAGE_TIMEOUT:-1800}"   # 30 min per paper

brief="briefs/triage-${TODAY}-${arxiv_id}.md"
log="${brief}.log"

# Skip if a brief for this arxiv_id exists from any date.
existing="$(ls briefs/triage-*-"${arxiv_id}".md 2>/dev/null | head -1 || true)"
if [[ -n "$existing" ]]; then
  echo "skip  $arxiv_id (brief exists at $existing)"
  exit 0
fi

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "FAIL  $arxiv_id — template missing at $TEMPLATE_PATH"
  exit 1
fi

prompt="$(sed -e "s|__ARXIV_ID__|${arxiv_id}|g" -e "s|__TODAY__|${TODAY}|g" "$TEMPLATE_PATH")"

echo "===== $arxiv_id ====="
if timeout "$TIMEOUT_SECONDS" claude -p --dangerously-skip-permissions "$prompt" < /dev/null > "$log" 2>&1; then
  if [[ -f "$brief" ]]; then
    echo "  ok  $brief"
    exit 0
  else
    echo "  WARN $arxiv_id — claude exited 0 but no brief written; see $log"
    exit 2
  fi
else
  rc=$?
  if [[ "$rc" -eq 124 ]]; then
    echo "  TIMEOUT $arxiv_id — exceeded ${TIMEOUT_SECONDS}s; see $log"
  else
    echo "  FAIL $arxiv_id — exit $rc; see $log"
  fi
  exit "$rc"
fi
