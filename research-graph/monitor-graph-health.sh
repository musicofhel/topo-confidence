#!/usr/bin/env bash
# Fail-loud graph monitor (rebuild §7). Runs on a 10-min systemd timer.
# If the graph is empty/near-empty (a wipe), write an alert file and POST to the
# link-forge Discord webhook so a wipe is caught in minutes, not days.
set -uo pipefail

CONTAINER="topo-research-graph"
USER_NEO="neo4j"
PASS_NEO="topo_graph_dev"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALERT_FILE="$HERE/.graph-wipe-alert"
MIN_NODES=100
WEBHOOK="${RESEARCH_GRAPH_DISCORD_WEBHOOK:-}"   # set in env/.env if Discord alerting wanted

NODES="$(docker exec "$CONTAINER" cypher-shell -u "$USER_NEO" -p "$PASS_NEO" \
  --format plain 'MATCH (n) RETURN count(n)' 2>/dev/null | tail -1 | tr -d '[:space:]')"
FINDINGS="$(docker exec "$CONTAINER" cypher-shell -u "$USER_NEO" -p "$PASS_NEO" \
  --format plain 'MATCH (f:Finding) RETURN count(f)' 2>/dev/null | tail -1 | tr -d '[:space:]')"

if [ -z "$NODES" ]; then
  MSG="🚨 topo-research-graph UNREACHABLE (cypher-shell failed) at $(date -u +%FT%TZ)"
  STATE=down
elif [ "$NODES" -le "$MIN_NODES" ] || [ "${FINDINGS:-0}" -lt 1 ]; then
  MSG="🚨 topo-research-graph WIPE SUSPECTED: nodes=$NODES findings=${FINDINGS:-0} at $(date -u +%FT%TZ)"
  STATE=wiped
else
  # healthy: clear any stale alert and exit quiet
  [ -f "$ALERT_FILE" ] && { echo "recovered: nodes=$NODES findings=$FINDINGS" ; rm -f "$ALERT_FILE"; }
  exit 0
fi

echo "$MSG"
# de-dupe alerts: only fire once until recovery clears the file
if [ ! -f "$ALERT_FILE" ]; then
  echo "$MSG" > "$ALERT_FILE"
  if [ -n "$WEBHOOK" ]; then
    curl -fsS -H 'Content-Type: application/json' \
      -d "{\"content\": \"$MSG\"}" "$WEBHOOK" >/dev/null 2>&1 || true
  fi
fi
exit 1
