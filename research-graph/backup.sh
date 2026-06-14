#!/usr/bin/env bash
# Nightly online APOC cypher export of the topo research graph (rebuild §6).
# Zero-downtime: APOC exports while the DB stays up. Writes into the container's
# /backups (host ./backups), gzips, keeps the most recent 30.
set -euo pipefail

CONTAINER="topo-research-graph"
USER_NEO="neo4j"
PASS_NEO="topo_graph_dev"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="$HERE/backups"
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
INNER="/backups/graph-${STAMP}.cypher"
OUTER="$BACKUP_DIR/graph-${STAMP}.cypher"

mkdir -p "$BACKUP_DIR"

# Refuse to back up an empty/wiped graph (don't overwrite good history with junk).
NODES="$(docker exec "$CONTAINER" cypher-shell -u "$USER_NEO" -p "$PASS_NEO" \
  --format plain 'MATCH (n) RETURN count(n)' | tail -1 | tr -d '[:space:]')"
if [ -z "$NODES" ] || [ "$NODES" -lt 100 ]; then
  echo "ABORT: graph has $NODES nodes (<100) — refusing to snapshot a wiped DB." >&2
  exit 1
fi

docker exec "$CONTAINER" cypher-shell -u "$USER_NEO" -p "$PASS_NEO" \
  "CALL apoc.export.cypher.all('${INNER}', {format:'cypher-shell'})"

gzip -f "$OUTER"
echo "backup: ${OUTER}.gz ($NODES nodes)"

# rotate: keep newest 30
ls -1t "$BACKUP_DIR"/graph-*.cypher.gz 2>/dev/null | tail -n +31 | xargs -r rm -f
echo "rotation: $(ls -1 "$BACKUP_DIR"/graph-*.cypher.gz 2>/dev/null | wc -l) backups retained"
