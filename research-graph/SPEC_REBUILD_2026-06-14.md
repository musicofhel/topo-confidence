# SPEC — Research-Graph High-Fidelity Rebuild + Hardening

_Status: NOT YET EXECUTED. Authored 2026-06-14. Owner: graph rebuild after the 2026-06-14 wipe._
_Run nothing in this spec until the rebuild section is approved to execute._

## 0. Why this exists (root cause of the wipe)

The `topo-research-graph` Neo4j (bolt://localhost:7688) was found holding **1 node** (a single
paper inserted by the link-forge bridge at 04:06 UTC). Forensics from neo4j's own `debug.log`:

| Startup (UTC)   | `neo4j` store STORE_ID    | last txId | state                         |
|-----------------|---------------------------|-----------|-------------------------------|
| Jun 12 03:31    | Apr-26 (`…149249`)        | 19,791    | real graph intact             |
| **Jun 13 17:23**| Apr-26 (`…149249`)        | **20,342**| real graph **still intact**   |
| **Jun 14 03:31**| **Jun-14** (`…860074`)    | 3 → fresh | **brand-new empty database**  |

Both `neo4j` and `system` databases were re-created from scratch at the 03:30 UTC startup
(`NodeStore used=0`, tx-log reset to `version=0`). That is **not** a `DETACH DELETE` (that keeps
txId high) — it is the signature of the bind-mounted `./data` directory being **empty when the
container started**. Root cause: an **unclean WSL2 shutdown** lost unsynced store files on the
9p/virtiofs bind mount; the `restart: unless-stopped` container then **silently re-initialized an
empty DB** at next boot and served it for ~half a day before discovery.

**Ruled out:** Tailscale (no path to this data; link-forge sync targets link-forge's own DB and is
disabled), the startup scripts (no destructive ops), and `git clean -fdx` (other git-ignored files
like `triage_run_*.log` survived — a `-fdx` would have deleted them too).

**No `.dump` backup existed.** This spec fixes both the loss and the silence.

## 1. Goal

1. Rebuild the graph so it **provably matches** the lost state (verified against the preserved
   `NEXT_EXPERIMENTS.md` snapshot), not just "approximately reseeds."
2. Harden the schema, add the missing backup, make a future wipe **fail loud**, and de-duplicate
   the FE backlog — all folded into the one-time rebuild.

## 2. Source-of-truth inventory (what reconstructs from what)

Verified 2026-06-14:

| Component                     | Source of truth                                              | Recoverable? |
|-------------------------------|-------------------------------------------------------------|--------------|
| Pathways, completed Experiments, seed Findings, seed Papers | `seed.py` (hardcoded)                  | 100% |
| Early FEs (P11 low-numbered)  | `seed_future_experiments.py` `FUTURE_EXPERIMENTS` list      | 100% |
| Bulk FE nodes (~1,100)        | 326 triage briefs `briefs/triage-*.md` (~847 FE blocks in the Apr-29 batch alone) | 100% structure |
| Result/COMPLETED state        | 25 `briefs/result-*.md`                                     | 100% |
| Migration papers (220)        | `migration_2026-04-28_papers.json`                          | 100% |
| Premises (7) + refutations    | `premises.py` `SEED_PREMISES` (hardcoded)                   | 100% |
| Embeddings (384-d MiniLM)     | `backfill_embeddings.py` (deterministic)                    | 100% |
| Paper metadata enrichment     | `bridge.py` (link-forge, read-only)                         | 100% (graceful if down) |
| **Final FE status/roi/priority** | **`golden/NEXT_EXPERIMENTS.golden-2026-06-13.md`** (1,207-FE snapshot, preserved 2026-06-14) | see §3 |
| **Accreted MOOTED/ANSWERED closures** (~110 bootstrap + later) | `briefs/moot-sweep-*.md` + premise cascades | see §3 |

> **CORRECTION to an earlier audit pass:** there is **no pile of ~880 hand-added FEs**. The FEs
> come from brief YAML blocks + the `seed_future_experiments.py` list. The only real drift is FE
> **status**, addressed in §3.

**Genuinely unrecoverable:** any change made *directly in the graph DB* that was never written to a
brief/script/snapshot (ad-hoc `priority`/`roi_score` edits without a brief, prose edits to
narrative docs outside the promote pipeline). The 2026-06-13 snapshot caps this loss at "whatever
changed in the graph between 16:30 Jun 13 and the wipe."

## 3. The fidelity strategy (status reconciliation + verification gate)

Briefs carry the **initial** `status:`; the **final** status accreted later. Three-layer recovery:

1. **Replay deterministic closures** (have provenance, reproducible):
   - `premises.py seed` then replay each refutation → cascade-moots every FE with a `RELIES_ON`
     edge to a REFUTED premise (born-MOOTED).
   - `promote_result.py` over the 25 result briefs → marks COMPLETED, updates findings.
2. **Reconcile the residue from the snapshot:** a new importer parses the preserved
   `golden/NEXT_EXPERIMENTS.golden-2026-06-13.md` and, for any FE whose final
   `status`/`roi_score`/`priority` still differs from the rebuilt graph, sets it to the snapshot
   value and stamps provenance `reconciled_from = "NEXT_EXPERIMENTS@2026-06-13"`. This closes the
   MOOTED/ANSWERED/TRIGGERED residue that replay can't reproduce edge-for-edge.
3. **Verification gate (the check the old pipeline never had):** run
   `generate_next_experiments.py` to emit a fresh `NEXT_EXPERIMENTS.md`, then **diff it against the
   golden snapshot**. A clean diff (modulo the regeneration timestamp line) ⇒ the rebuild matches.
   Any residual diff is an explicit, reviewable list of what did not reproduce.

> Snapshot is used to *set* residue status AND to *verify* — note the mild circularity: the diff
> proves structure + reconciliation succeeded, not that the snapshot itself was correct. That is
> acceptable; the snapshot is the best record of the lost state.

## 4. Canonical rebuild sequence (corrected, ordered)

> Prereq already done: golden snapshot preserved at
> `research-graph/golden/NEXT_EXPERIMENTS.golden-2026-06-13.md` (1,207 FEs). **Do not let any step
> overwrite it** — `generate_next_experiments.py` writes the live `NEXT_EXPERIMENTS.md`, never the
> golden copy.

```bash
cd ~/topo-confidence/research-graph

# 1. Apply schema (now includes §5 hardening) + core seed (pathways/exps/findings/papers).
#    seed.py runs `MATCH (n) DETACH DELETE n` first — safe, graph is already empty.
python seed.py --reset

# 2. Early hardcoded FEs.
python seed_future_experiments.py

# 3. (Optional) reload the 220 migration papers as pending_triage.
python migrate_to_pending_triage_2026-04-28.py --apply

# 4. Promote all triage briefs → FE nodes (initial status/roi/priority, methods, datasets, premise links).
for b in briefs/triage-*.md; do python promote_brief.py "$b" --update-existing; done

# 5. Seed premises + replay refutations → cascade-moot reliant FEs (deterministic closures).
python premises.py seed
#   replay each historical refutation, e.g.:
#   python premises.py refute dom-causal-lever --by FE269 --reason "diagnostic readout, not causal lever"
#   (enumerate from SEED_PREMISES status==REFUTED; these are idempotent.)

# 6. Promote result briefs → COMPLETED + finding updates.
for r in briefs/result-*.md; do python promote_result.py "$r"; done

# 7. Reconcile residual final status/roi/priority from the golden snapshot (NEW script, §3.2).
python reconcile_from_snapshot.py golden/NEXT_EXPERIMENTS.golden-2026-06-13.md

# 8. Embeddings (needed by dedup + moot cosine ranking).
python backfill_embeddings.py

# 9. Semantic FE dedup pass (NEW script, §7).
python dedup_future_experiments.py --threshold 0.88 --apply

# 10. Enrich paper metadata from link-forge (graceful if down).
python bridge.py enrich

# 11. VERIFICATION GATE: regenerate and diff against golden.
python generate_next_experiments.py
diff <(grep -E '^### P11-FE|status|ROI' NEXT_EXPERIMENTS.md) \
     <(grep -E '^### P11-FE|status|ROI' golden/NEXT_EXPERIMENTS.golden-2026-06-13.md) \
     | tee golden/rebuild-diff-2026-06-14.txt
#   Expect: only the generation-timestamp header differs. Investigate any FE-level diff.
```

## 5. Schema hardening (fold into `schema.cypher` before step 1)

All `IF NOT EXISTS`, all safe to add to a fresh DB:

```cypher
-- missing index: every moot-sweep/query filters fe.status across ~1,200 nodes unindexed
CREATE INDEX future_experiment_status IF NOT EXISTS
  FOR (fe:FutureExperiment) ON (fe.status);

-- null-safe priority ordering already enforced in code; also index roi for tiering
CREATE INDEX future_experiment_roi IF NOT EXISTS
  FOR (fe:FutureExperiment) ON (fe.roi_score);

-- prevent silent duplicate premises / methods / datasets on any re-run
CREATE CONSTRAINT premise_id IF NOT EXISTS FOR (p:Premise)  REQUIRE p.id  IS UNIQUE;
CREATE INDEX     premise_status IF NOT EXISTS FOR (p:Premise) ON (p.status);
CREATE CONSTRAINT method_key  IF NOT EXISTS FOR (m:Method)  REQUIRE m.key IS UNIQUE;
CREATE CONSTRAINT dataset_key IF NOT EXISTS FOR (d:Dataset) REQUIRE d.key IS UNIQUE;
```

Code companions:
- `generate_next_experiments.py` queue sort → `ORDER BY coalesce(fe.roi_score, 0) DESC, fe.id`
  (a missing score must sort to the bottom, never scramble the queue).
- `promote_brief.py` `VALID_STATUS` set → include `MOOTED`, `ANSWERED` so a born-MOOTED FE is never
  validated against an incomplete enum.

## 6. Durability: APOC online export (replaces the offline-dump-with-downtime plan)

**Decision:** community edition has no online `neo4j-admin backup`, and offline `neo4j-admin
database dump` requires stopping the container — which collides with the autopilot/bridge writers
and is fragile on WSL2. **APOC is already a loaded plugin**, so use **online cypher export, zero
downtime**, written to a host directory that can be git-tracked.

Compose additions (`docker-compose.yml`):
```yaml
environment:
  # ... existing ...
  NEO4J_apoc_export_file_enabled: "true"
  NEO4J_dbms_checkpoint_interval_time: "15m"        # shrink the unsynced-loss window
volumes:
  - ./data:/data
  - ./backups:/backups                               # writable export target
```

Nightly export (online — DB stays up):
```bash
docker exec topo-research-graph cypher-shell -u neo4j -p topo_graph_dev \
  "CALL apoc.export.cypher.all('/backups/graph-'+date()+'.cypher', {format:'cypher-shell'})"
gzip -f research-graph/backups/graph-*.cypher        # rotate: keep 30
```

Wrap in `research-graph/backup.sh` + a **systemd user timer** (`neo4j-backup.timer`, `OnCalendar=*-*-* 02:00`, `Persistent=true`).
Restore path: `cat graph-DATE.cypher | cypher-shell -u neo4j -p topo_graph_dev` into a fresh DB.
Keep one **monthly** offline `neo4j-admin database dump` (accepts brief downtime) as a belt-and-suspenders binary snapshot.

> Implementation-time checks: confirm `apoc.export.cypher.all` is permitted under the loaded APOC
> build (needs `apoc.export.file.enabled=true`, set above); confirm `date()` interpolation in the
> shell call or compute the filename in `backup.sh` instead.

## 7. Fail-loud (replace silent empty re-init)

Two layers:
1. **Data-aware healthcheck** in compose (so an empty graph reports *unhealthy*, not "RETURN 1 OK"):
   ```yaml
   healthcheck:
     test: ["CMD-SHELL", "test \"$(cypher-shell -u neo4j -p topo_graph_dev --format plain 'MATCH (f:Finding) RETURN count(f)' | tail -1)\" -ge 1 || exit 1"]
     interval: 30s
     timeout: 10s
     retries: 5
     start_period: 40s
   ```
2. **Monitor + alert** systemd timer (`research-graph-monitor`, every 10 min): if total nodes ≤ 1 or
   Finding count = 0, write an alert file and post to the link-forge Discord webhook. Catches a wipe
   in minutes, not days.

> Tradeoff: a data-aware healthcheck means the container is "unhealthy" during a *legitimate* empty
> rebuild — run the rebuild with the healthcheck temporarily relaxed, or gate it on a
> `REBUILD_IN_PROGRESS` flag file.

## 8. Semantic FE dedup (`dedup_future_experiments.py`, NEW)

Problem: 326 briefs independently propose overlapping experiments (e.g. FE335/FE337). No dedup key
exists today. Pass (post-embeddings, step 9):
1. For each open FE, find top-K nearest by cosine on the MiniLM embedding.
2. For pairs with sim ≥ 0.88: keep the higher `roi_score`; close the loser as `ANSWERED` with
   `closed_by = <keeper id>`; union the `TRIGGERED_BY` edges onto the keeper.
3. `--dry-run` first; emit a review list to `briefs/dedup-2026-06-14.md`; only `--apply` after review.
Threshold starts at 0.88 (conservative); tune against the known FE335/FE337 case.

## 9. Acceptance criteria

- [ ] Node/edge counts within tolerance of the snapshot: **1,207 FE nodes** (pre-dedup), 7 premises,
      findings F-1…F-14+, papers ≥ 277.
- [ ] `reconcile_from_snapshot.py` reports 0 FEs left with status diverging from golden.
- [ ] `golden/rebuild-diff-2026-06-14.txt` shows only the timestamp header differing (FE-level diffs
      triaged to zero or explicitly documented).
- [ ] `validate_claims.py` still passes its internal-PASS invariant (graph-independent, but run it).
- [ ] One APOC export file exists in `backups/` and the restore path round-trips on a scratch DB.
- [ ] Healthcheck flips to **unhealthy** when pointed at an empty DB (test once, then revert).
- [ ] `dedup_future_experiments.py --dry-run` review file generated; FE335/FE337 collapse confirmed.

## 10. New artifacts this spec introduces

- `reconcile_from_snapshot.py` — parse golden `NEXT_EXPERIMENTS.md`, set residual FE status/roi/priority + provenance.
- `dedup_future_experiments.py` — embedding-cosine FE dedup (`--dry-run`/`--apply`).
- `backup.sh` + `~/.config/systemd/user/neo4j-backup.{service,timer}` — nightly APOC export.
- `monitor-graph-health.sh` + `~/.config/systemd/user/research-graph-monitor.{service,timer}` — fail-loud monitor.
- `schema.cypher` edits (§5); `docker-compose.yml` edits (§6/§7); `generate_next_experiments.py` + `promote_brief.py` companions (§5).
- Preserved: `research-graph/golden/NEXT_EXPERIMENTS.golden-2026-06-13.md` (done 2026-06-14).

## 11. Open risks

- The snapshot predates the wipe by ~half a day; any graph-only change in that window is lost (cap on loss).
- `reconcile_from_snapshot.py` depends on the golden file's format being stable — pin the parser to the 2026-06-13 generator output.
- APOC export permission/config must be confirmed against the loaded APOC build at implementation time.
- WSL2 remains the underlying fragility; durability here is mitigation (backup + fail-loud + checkpoint tuning), not prevention. The only true prevention is moving the store off the 9p bind mount (named volume) — deferred, noted.
