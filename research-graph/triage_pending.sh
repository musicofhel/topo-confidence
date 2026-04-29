#!/usr/bin/env bash
# Parallel dispatcher for max-depth paper triage over every
# :Paper {status:'pending_triage'}.
#
# Spawns N worker processes (default 3), each running triage_one.sh on one
# paper at a time via xargs -P. Each worker spawns its own `claude -p` process,
# preserving per-paper isolation (the SKILL.md hard rule per Ríos-García
# 2604.18805 confirmation-bias defense).
#
# Why 3? Conservative parallelism — each `claude -p` makes upstream API + MCP
# calls. 3 keeps load well below any rate limit and keeps local memory usage
# under ~3GB. Bump via `PARALLEL=5 bash triage_pending.sh` if the API holds up.
#
# Each worker wraps its claude call in `timeout 1800` (30 min ceiling), so a
# hung paper releases its slot rather than pinning one worker. A separate
# triage_watchdog.sh is no longer required — kill stale claude children
# upstream by raising PARALLEL or `pkill -f "claude -p"`.
#
# Tier-priority ordering (admitted-then-rejected first, etc.) is enforced by
# `query.py pending --ids-only`. With parallelism, ordering is best-effort:
# faster workers may take Tier-2 papers before slower workers finish Tier-1,
# but every paper still gets processed.
#
# Usage:
#   bash triage_pending.sh                 # 3 workers, all pending
#   PARALLEL=5 bash triage_pending.sh      # 5 workers
#   LIMIT=10 bash triage_pending.sh        # process at most 10 IDs from queue

set -euo pipefail

cd "$(dirname "$0")"

# `claude -p` refuses to start cleanly inside an existing Claude Code session
# (the parent CLAUDECODE env var triggers a nested-session guard). Strip it so
# the spawned per-paper sessions are truly fresh.
unset CLAUDECODE

mkdir -p briefs

PARALLEL="${PARALLEL:-3}"
LIMIT="${LIMIT:-0}"

ids="$(python query.py pending --ids-only)"
total="$(printf '%s\n' "$ids" | grep -c '^[0-9]' || true)"

if [[ "$LIMIT" -ne 0 ]]; then
  ids="$(printf '%s\n' "$ids" | head -n "$LIMIT")"
  echo "==> $total papers in queue, processing first $LIMIT with PARALLEL=$PARALLEL"
else
  echo "==> $total papers in queue, processing all with PARALLEL=$PARALLEL"
fi

# xargs -P PARALLEL spawns up to PARALLEL workers; -n 1 means one arxiv_id per
# worker invocation; --no-run-if-empty avoids running with no input.
# Worker exit status is captured but does NOT abort siblings (xargs default).
printf '%s\n' "$ids" \
  | grep '^[0-9]' \
  | xargs --no-run-if-empty -P "$PARALLEL" -n 1 -I {} bash triage_one.sh {}

echo
echo "==> dispatcher exited."
echo "    Brief count: $(ls briefs/triage-*.md 2>/dev/null | wc -l)"
echo "    Review briefs in $(pwd)/briefs/, then promote keepers:"
echo "      python promote_brief.py briefs/triage-YYYY-MM-DD-<arxiv-id>.md --dry-run"
echo "      python promote_brief.py briefs/triage-YYYY-MM-DD-<arxiv-id>.md"
