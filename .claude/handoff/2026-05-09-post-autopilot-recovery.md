# Handoff — 2026-05-09 — Post-Autopilot Recovery

## What happened this session

The previous session ("post-experiment-write-up-sweep") ran a three-phase
autopilot daemon (01:17–08:14) that partially succeeded and partially broke.
This session recovered the state, promoted results, and diagnosed all failures.

### Commits made (branch `max-depth-retriage-2026-04-28`)

1. **e0051c5** — Autopilot run: 13 paper triages, FE42 result brief, 2 new claims
   - 13 triage briefs (all auto-promoted to Neo4j by the daemon)
   - H-741..H-749 added to HYPOTHESES.md
   - NEXT_EXPERIMENTS.md + PAPER_INDEX.md regenerated
   - validate_claims.py: +2 claims (fe421-dom-prefill, fe421-dom-final) → 202/202 PASS
   - Autopilot code changes (autopilot.py, cli.py)

2. **9298b53** — EXP-79: promote FE42 ridge-LR result
   - EXPERIMENT_LOG.md: +EXP-79, next ID → EXP-80
   - FINDINGS.md: F-2 and F-9 updated with FE421 ridge-LR evidence
   - STATE.md: last experiment → EXP-79
   - Neo4j: P11-FE42 COMPLETED, F-2/F-9 evidence edges
   - promote_result.py: patched to pass `--no-regen` to validate_claims

### Current state

- **validate_claims.py**: 202/202 internal PASS, 41 REGISTERED, 3 PENDING_FE (246 total)
- **EXPERIMENT_LOG.md**: 80 experiments (EXP-001..EXP-079), next ID EXP-80
- **Neo4j**: 267 papers, 14 active findings (F-1..F-14), 935 READY + 19 TRIGGERED + 105 BLOCKED + 40 COMPLETED FEs
- **Autopilot daemons**: both killed (PID 8973 triage, 9137 experiment)
- **Branch**: 5 commits ahead of remote

### Untracked files still on disk (not committed)

- `.autopilot/` — daemon logs, budget JSONs, failures.json, lockfiles, trigger file
- `pathway11_h100/apply_parkchoeveitch_causal_inner_produc/` — empty experiment scaffold
- `pathway11_h100/compute_pds_pairwise_process_supervision/` — empty experiment scaffold
- `pathway11_h100/logs/sweep_20260506_215829/` — sweep 1 runner logs
- `pathway11_h100/logs/sweep2_20260507_173430/` — sweep 2 runner logs
- `pathway11_h100/logs/sweep_runner.log` — runner log
- `pipeline_state.db` — langgraph checkpoint database

---

## Fix list — ordered by priority

### FIX-1: Push to remote

Branch is 5 commits ahead. `pathway11_h100/` is NOT in the GitHub remote
(gitignored due to size). Back up this directory if the local disk is at risk.

```bash
cd ~/topo-confidence && git push
```

### FIX-2: Update STATE.md header

STATE.md still says "Date: 2026-05-07" and the "Where the project actually is"
section says 200/200 internal PASS. Should say:

- **Date:** 2026-05-09
- **Claims**: 202/202 internal PASS, 246 total
- The "Queued — next session" section should note that Phase 1 of PLAN_cheap_wins
  is already complete (FE719, FE448, FE145, FE299, FE101 — all run in the
  EXP-59..78 sweeps). Phase 2/3 are next.

### FIX-3: Bump script-gen timeout (300s → 600s)

**File**: `pipeline/generate_recompute.py:164`

All 11 script-gen failures were `claude -p` timeouts at 300s. The exemplar-heavy
prompt (3 full scripts + FE metadata) takes longer than 5 minutes to generate.

```python
# Line 164: change from
timeout=300,
# to
timeout=600,
```

**Validation**: After fixing, run a single FE generation to confirm:
```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.generate_recompute import scriptless_local_fes, generate_for_fe
fes = scriptless_local_fes()
print(f'{len(fes)} scriptless FEs')
if fes:
    result = generate_for_fe(fes[0])
    print(f'Result: {result}')
"
```

### FIX-4: Dirty-tree check improvements in autopilot

**Problem**: The experiment daemon hit a modify→restart→dirty-loop sequence:
1. FE42 ran successfully and wrote `pathway11_h100/results/fe421_regularized_concat.json`
2. Daemon was killed/restarted (SIGTERM during budget bump)
3. Modified result JSON triggered `_git_is_clean()` failure forever
4. Daemon polled "output tree is dirty" every 120s for 6 hours with no recovery path

**File**: `pipeline/autopilot.py:188-199`

Two fixes needed:

**Fix 4a**: After a successful experiment run, auto-commit the result JSON
before returning to the poll loop. The experiment phase already writes the
result — it should also commit it.

**Fix 4b**: Add a self-healing escape hatch. If the dirty-tree warning fires
N consecutive times (e.g., 5 = 10 minutes), log the dirty files, attempt to
commit them with a descriptive message, and continue. The current behavior is
an infinite loop.

### FIX-5: SIGTERM handling in experiment pipeline

**Problem**: The `generate_brief` node calls `claude -p` which got SIGTERM (exit
143) when the parent daemon was killed for budget reconfig. The brief was already
written to disk but the pipeline didn't know — it reported failure.

**Files**: `pipeline/nodes.py:267` (`_call_claude`), `pipeline/autopilot.py`

**Fix**: The daemon's SIGTERM handler (which does `shutting_down = True`) should
wait for the current pipeline run to finish before exiting, or at minimum check
whether the brief was written on disk before recording a failure.

### FIX-6: promote_result.py should not run full regen by default

**Already partially fixed** this session (added `--no-regen` to the
validate_claims subprocess call). But the fix is in the runtime behavior —
the `--skip-regen` CLI flag for promote_result.py itself is separate from
the validate_claims regen.

Consider making `--no-regen` the default for validate_claims within
promote_result.py, since the full regen (including FE749 at ~2h45m) is
inappropriate as a gate for promotion.

### FIX-7: Commit or gitignore operational artifacts

Decide what to do with untracked files:

```bash
# Option A: Gitignore operational artifacts
echo '.autopilot/' >> .gitignore
echo 'pipeline_state.db' >> .gitignore
echo 'pathway11_h100/logs/' >> .gitignore

# Option B: Commit them for traceability
git add .autopilot/daemon-*.log .autopilot/failures.json
git add pathway11_h100/logs/
git commit -m "Autopilot operational logs from 2026-05-09 run"
```

The empty experiment scaffolds (`apply_parkchoeveitch_*`, `compute_pds_*`) were
generated by the script-gen phase but left empty when generation failed. They can
be deleted:

```bash
rm -rf pathway11_h100/apply_parkchoeveitch_causal_inner_produc/
rm -rf pathway11_h100/compute_pds_pairwise_process_supervision/
```

### FIX-8: Clear stale autopilot state before next run

```bash
# Reset budget files (they track daily counts from May 9)
rm ~/topo-confidence/.autopilot/budget-*.json
rm ~/topo-confidence/.autopilot/budget-*.lock

# Reset failure counters (so failed FEs get fresh attempts)
rm ~/topo-confidence/.autopilot/failures.json

# Clear experiment lock
rm ~/topo-confidence/.autopilot/experiment.lock

# Verify no stale daemon PIDs
cat ~/topo-confidence/.autopilot/daemon.pid  # should not match any running process
```

### FIX-9: Validate the 13 triage briefs were correctly promoted to Neo4j

The triage daemon reported auto-promotion but we should verify:

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7688', auth=('neo4j','topo_graph_dev'))
ids = [
    '2409.05084', '2603.23198', '2604.12016', '2604.23985', '2604.27169',
    '2605.02105', '2605.03327', '2605.04230', '2605.04255', '2605.04344',
    '2605.04418', '2605.04971', '2605.05115', '2605.05683',
]
with d.session() as s:
    for aid in ids:
        r = s.run('MATCH (p:Paper) WHERE p.arxiv_id = \$id RETURN p.status as st', id=aid)
        rec = r.single()
        st = rec['st'] if rec else 'NOT_FOUND'
        flag = '' if st == 'graphed' else '  *** PROBLEM ***'
        print(f'  {aid}: {st}{flag}')
d.close()
"
```

Expected: all 14 should show `graphed`. If any show `pending_triage` or
`NOT_FOUND`, re-run promotion manually:

```bash
cd ~/topo-confidence/research-graph
python promote_brief.py briefs/triage-2026-05-09-<arxiv-id>.md --update-existing
```

### FIX-10: The 12 status-NULL Discord papers (pre-existing issue)

STATE.md §Open threads #3 mentions 12 Discord-admitted papers with status NULL
that are missed by `query.py pending`. These pre-date the autopilot run and were
never reconciled.

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7688', auth=('neo4j','topo_graph_dev'))
with d.session() as s:
    r = s.run('MATCH (p:Paper) WHERE p.status IS NULL RETURN p.arxiv_id, p.title LIMIT 20')
    for rec in r:
        print(f'  {rec[\"arxiv_id\"]}: {rec[\"title\"][:60]}')
d.close()
"
```

For each, either:
- Set status to `pending_triage` → they'll be picked up by the triage daemon
- Or set status to `rejected` if they're off-topic

---

## Validation checklist (run all before next experiment session)

```bash
cd ~/topo-confidence

# 1. Claims invariant
.venv/bin/python validate_claims.py --no-regen 2>&1 | grep -E "PASS|FAIL"
# EXPECT: 202 PASS, 0 FAIL

# 2. Neo4j health
.venv/bin/python research-graph/query.py status-report 2>&1 | head -5
# EXPECT: 14 active findings, no INVALIDATED/SUPERSEDED

# 3. EXP-79 appears in log
grep "EXP-79" EXPERIMENT_LOG.md
# EXPECT: ## EXP-79: FE42 — Ridge-LR ...

# 4. Next ID is EXP-80
tail -1 EXPERIMENT_LOG.md
# EXPECT: Next ID: **EXP-80**

# 5. F-2 in FINDINGS.md has FE421 evidence
grep "FE421" FINDINGS.md
# EXPECT: ridge-LR reference in F-2 controls section

# 6. No stale daemons
ps aux | grep autopilot | grep -v grep
# EXPECT: nothing

# 7. Git clean (except untracked operational artifacts)
git status --short
# EXPECT: only ?? lines for .autopilot/, logs/, pipeline_state.db

# 8. PLAN_cheap_wins.md Phase 1 status
# Confirm these FEs are all COMPLETED in Neo4j:
# FE719, FE448, FE145, FE299, FE101 (Phase 1)
# FE447, FE188, FE421, FE15, FE16, FE428, FE416, FE119, FE308, FE459 (sweeps)

# 9. Branch push status
git log --oneline origin/max-depth-retriage-2026-04-28..HEAD
# EXPECT: 5 commits ahead (push when ready)
```

---

## Next experiment work

### Immediate: PLAN_cheap_wins Phase 2 (ROI-10 anchors)

Phase 1 is done. Phase 2 targets:
- **FE115** — Song-Zhong position/context decomposition
- **FE749** — Spectral α head-to-head (NOTE: regen takes ~2h45m)
- **FE181** — Token-probability baseline

Check current FE status before running:
```bash
cd ~/topo-confidence && .venv/bin/python research-graph/query.py future P11
```

### Medium-term: Fix autopilot and restart

After FIX-3 through FIX-8 are done:

```bash
cd ~/topo-confidence
.venv/bin/python -m pipeline autopilot --phase triage --budget 300 --poll 60 --verbose --no-langfuse --no-jaeger &
.venv/bin/python -m pipeline autopilot --phase scriptgen --budget 300 --poll 120 --verbose --no-langfuse --no-jaeger &
.venv/bin/python -m pipeline autopilot --phase experiment --budget 300 --poll 120 --verbose --no-langfuse --no-jaeger &
```

### Open threads (carried from STATE.md)

1. **Causal companion test for F-2** — rank-truncate-PC1-residualized-cov
   ablation. Not yet a `:FutureExperiment` node. File as one.
2. **Cross-architecture replication** — Phi-3-mini and Llama-3.2-1B caches
   exist in `pathway11_h100/exp1_cross_model/`. ~1h CPU each.
3. **12 status-NULL Discord papers** — see FIX-10 above.
4. **pathway11_h100/ backup** — not in GitHub remote due to gitignore/size.

---

## Key numbers (all 1024tok, full precision in validate_claims.py)

| Metric | Value | Source |
|--------|-------|--------|
| F-2 prefill DoM AUROC | 0.7731 | EXP-040 |
| Ridge-LR final-only | 0.8493 | EXP-79/FE421 |
| Ridge-LR concat | 0.8509 | EXP-79/FE421 |
| Ridge-LR prefill-only | 0.7844 | EXP-79/FE421 |
| Cov-spectrum top-20 | 0.7928 | FE881 |
| 7B prefill DoM AUROC | 0.874 | EXP-78/FE459 |
| Cross-model Spearman | 0.937 | EXP-78/FE459 |
| Length-residualized DoM | 0.620 | EXP-74/FE447 |
| Selective acc@50% cov | 71.6% | EXP-040 |
