#!/usr/bin/env bash
# Loop runner for max-depth paper triage over every :Paper {status:'pending_triage'}.
#
# Calls `claude -p "<full agent prompt>"` per paper. Each invocation is a separate
# `claude` process — the per-paper isolation the SKILL.md hard-rule depends on
# (Ríos-García 2604.18805 confirmation-bias defense). Briefs land in
# ./briefs/triage-YYYY-MM-DD-<arxiv-id>.md.
#
# We DON'T use `claude -p "/paper-triage <id>"` because the spawned assistant
# tends to second-guess slash-command invocations in headless mode. Inlining
# the full prompt template removes that ambiguity.
#
# Skips a paper if its brief already exists. Logs failures to <brief>.log so you
# can re-run without losing progress.
#
# Tier-priority ordering is enforced by `query.py pending --ids-only`:
#   T1 admitted-then-rejected → T2 prior candidate → T3 graphed-no-brief
#   → T4 seed paragraph → T5 Desktop backfill → T6 fresh admission
#
# Usage:
#   bash triage_pending.sh             # process all pending papers
#   LIMIT=10 bash triage_pending.sh    # stop after 10 successful briefs

set -euo pipefail

cd "$(dirname "$0")"

# `claude -p` refuses to start cleanly inside an existing Claude Code session
# (the parent CLAUDECODE env var triggers a nested-session guard). Strip it so
# the spawned per-paper sessions are truly fresh.
unset CLAUDECODE

mkdir -p briefs

TEMPLATE_PATH="./_paper_triage_prompt.template"
if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "FATAL: prompt template missing at $TEMPLATE_PATH" >&2
  exit 1
fi
TEMPLATE_RAW="$(cat "$TEMPLATE_PATH")"

LIMIT="${LIMIT:-0}"
TODAY="$(date +%Y-%m-%d)"
processed=0
skipped=0
failed=0

ids="$(python query.py pending --ids-only)"
total="$(printf '%s\n' "$ids" | grep -c '^[0-9]' || true)"
echo "==> $total papers in pending_triage queue (LIMIT=${LIMIT:-unlimited})"

build_prompt() {
  local arxiv_id="$1"
  printf '%s\n' "$TEMPLATE_RAW" \
    | sed -e "s|__ARXIV_ID__|${arxiv_id}|g" \
          -e "s|__TODAY__|${TODAY}|g"
}

while IFS= read -r arxiv_id; do
  [[ -z "$arxiv_id" ]] && continue

  brief="briefs/triage-${TODAY}-${arxiv_id}.md"
  log="${brief}.log"

  # Skip if a brief for this arxiv_id exists from any date — don't redo work.
  existing="$(ls briefs/triage-*-"${arxiv_id}".md 2>/dev/null | head -1 || true)"
  if [[ -n "$existing" ]]; then
    echo "skip  $arxiv_id (brief exists at $existing)"
    skipped=$((skipped + 1))
    continue
  fi

  echo "===== $arxiv_id ====="
  prompt="$(build_prompt "$arxiv_id")"
  if claude -p --dangerously-skip-permissions "$prompt" > "$log" 2>&1; then
    if [[ -f "$brief" ]]; then
      processed=$((processed + 1))
      echo "  ok  $brief"
    else
      failed=$((failed + 1))
      echo "  WARN $arxiv_id — claude exited 0 but no brief written; see $log"
    fi
  else
    failed=$((failed + 1))
    echo "  FAIL $arxiv_id — see $log"
  fi

  if [[ "$LIMIT" -ne 0 && "$processed" -ge "$LIMIT" ]]; then
    echo "==> reached LIMIT=$LIMIT, stopping"
    break
  fi
done <<< "$ids"

echo
echo "==> done.  processed=$processed  skipped=$skipped  failed=$failed"
echo "Review briefs in $(pwd)/briefs/, then promote keepers:"
echo "    python promote_brief.py briefs/triage-${TODAY}-<arxiv-id>.md --dry-run"
echo "    python promote_brief.py briefs/triage-${TODAY}-<arxiv-id>.md"
