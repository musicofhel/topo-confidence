# /session-loop

Session orchestrator entry point. Reads the session ledger, reports current
stage, and dispatches to the appropriate action.

## How it works

The session state lives in `.claude/session-state.md`. This skill reads that
file to determine where the session is and what to do next. If no session
exists, it creates one.

## Stages

| Stage | What happens | Next |
|-------|-------------|------|
| BATCH_GATE | Check Neo4j for >= 5 local-runnable FEs. If not enough, stop. | EXECUTE |
| EXECUTE | Run `python -m pipeline run --local --max-runs 5 --dry-run` | TRIAGE |
| TRIAGE | Review results. Which experiments succeeded? Generate follow-up FEs. | PLAN_V1 |
| PLAN_V1 | Write implementation plan based on triage results. Compact after. | PLAN_V2 |
| PLAN_V2 | Fresh-eyes audit of plan v1. "Please use fresh eyes, audit this plan. I don't want to spend any money." Compact after. | PLAN_V3 |
| PLAN_V3 | Second fresh-eyes audit. Write final plan + handoff. Compact after. | IMPLEMENT |
| IMPLEMENT | Implement current phase from plan v3. Write next-steps. Compact. Repeat until all phases done. | SWEEP |
| SWEEP | Final sweep of all changes. Commit and push to branch. | COMPLETE |
| COMPLETE | Done. Start a new session or stop. | — |

## Instructions

When this skill is invoked:

1. Read `.claude/session-state.md` using the Read tool
2. If the file doesn't exist, ask the user if they want to start a new session
3. Report the current stage, branch, and progress clearly
4. Based on the current stage, do the following:

### BATCH_GATE
- Run: `cd ~/topo-confidence && .venv/bin/python -c "from pipeline.nodes import _query_neo4j_ready_fes, _find_recompute_script; fes = _query_neo4j_ready_fes(); local = [fe for fe in fes if _find_recompute_script(fe['id'])]; print(f'{len(local)} local-runnable FEs ready')"` 
- If >= 5: advance to EXECUTE, update ledger, tell user to proceed
- If < 5: report how many are ready, suggest waiting or running `/triage-results` to generate more

### EXECUTE
- Tell the user: "Run the batch now with: `python -m pipeline run --local --max-runs 5`"
- After execution completes, advance to TRIAGE and update ledger
- Remind user to compact after this stage

### TRIAGE
- Tell user to invoke `/triage-results` to analyze completed experiments
- After triage, advance to PLAN_V1 and update ledger

### PLAN_V1
- Read triage results and write an implementation plan
- Save plan to the path in `plan_file` (default: `.claude/plans/session-plan-v1.md`)
- Update ledger: plan_version=1, advance to PLAN_V2
- Tell user: "Plan v1 written. Compact now, then re-invoke /session-loop for fresh-eyes audit."

### PLAN_V2
- Read plan v1 with completely fresh eyes
- Audit it soup-to-nuts: "I don't want to spend any money. Is this plan sound?"
- Write revised plan to `.claude/plans/session-plan-v2.md`
- Update ledger: plan_version=2, advance to PLAN_V3
- Tell user: "Plan v2 written. Compact now, then re-invoke /session-loop for final audit."

### PLAN_V3
- Read plan v2 with fresh eyes again
- Final audit pass. Write plan v3 + determine number of implementation phases
- Update ledger: plan_version=3, total_phases=N, advance to IMPLEMENT
- Tell user: "Plan v3 finalized with N phases. Compact now, then re-invoke /session-loop to begin phase 1."

### IMPLEMENT
- Read plan v3 and the session ledger to find current_phase
- Implement phase N (the current phase)
- Write next-steps for phase N+1 as notes in the ledger
- Update ledger: complete_phase("phase-N"), if current_phase < total_phases stay in IMPLEMENT, else advance to SWEEP
- Tell user: "Phase N complete. Compact now, then re-invoke /session-loop for phase N+1." (or "All phases done, invoke /session-loop for final sweep.")

### SWEEP
- Read all changes made during this session (git diff against main)
- Commit to the branch specified in the ledger
- Write a handoff file to `.claude/handoff/`
- Advance to COMPLETE
- Tell user: "Session complete. Branch ready for manual merge."

### COMPLETE
- Report: session is done. Show summary of what was accomplished.
- Ask if user wants to start a new session.

## Updating the ledger

After each stage transition, update the session state file by running:

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load, save
s = load()
s.advance_to('NEXT_STAGE')
save(s)
"
```

Replace `NEXT_STAGE` with the actual next stage name.

## Creating a new session

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import create_new, save
s = create_new()
save(s)
print(f'Session created: {s.batch_id} on branch {s.branch}')
"
```

## Key constraint

**Local compute only.** 2060 Super + CPU. No RunPod, no cloud GPU.
All experiments must be runnable on local hardware.
