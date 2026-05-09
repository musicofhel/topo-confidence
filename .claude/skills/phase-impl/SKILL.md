# /phase-impl

Implement the current phase from the finalized plan (v3). Reads the plan and
session ledger, implements the phase, writes next-steps, and updates the ledger.

## When to invoke

Called from `/session-loop` during the IMPLEMENT stage. Each invocation
implements one phase, then the user compacts and re-invokes for the next phase.

## Inputs

No arguments. Reads everything from the session ledger and plan file.

## How it works

### Step 1: Read current state

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load
s = load()
print(f'stage={s.stage}')
print(f'plan_file={s.plan_file}')
print(f'current_phase={s.current_phase}')
print(f'total_phases={s.total_phases}')
print(f'completed_phases={s.completed_phases}')
print(f'branch={s.branch}')
print(f'notes={s.notes}')
"
```

Then read the plan file (should be `.claude/plans/session-plan-v3.md`).

### Step 2: Find the current phase

Look up `Phase {current_phase + 1}` in the plan (phases are 1-indexed in the
plan, 0-indexed in the ledger's `current_phase`). Read:
- Goal
- Experiments to run
- Success criteria
- Estimated time

If `notes` in the ledger contains next-steps from the previous phase, read
those too — they may contain adjustments based on previous results.

### Step 3: Implement the phase

Do the work described in the phase:

1. **Write experiment scripts** if needed (recompute scripts, analysis code)
2. **Run experiments** using `python -m pipeline run --local` or direct script
   execution
3. **Collect results** and verify against success criteria
4. **Update docs** if results change established findings

All work must be on the session branch (from the ledger). If not already on
that branch:

```bash
cd ~/topo-confidence && git checkout -b {branch} 2>/dev/null || git checkout {branch}
```

### Step 4: Write next-steps

After completing the phase, write notes for the next phase. These go into the
session ledger's `notes` field so the next context window (after compaction)
knows what happened and what to do next.

Include:
- What was accomplished in this phase
- Any results that affect later phases
- Specific adjustments for the next phase (if the plan needs tweaking based
  on what was learned)
- Whether success criteria were met

### Step 5: Update session ledger

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load, save
s = load()
s.complete_phase('phase-{N}')
s.notes = '''Phase {N} complete.

Results: {summary}

Next-steps for phase {N+1}:
- {step 1}
- {step 2}
'''
# If all phases are done, advance to SWEEP
if s.current_phase >= s.total_phases:
    s.advance_to('SWEEP')
save(s)
"
```

After updating, tell the user:
- If more phases remain: "Phase {N} complete. Compact now, then re-invoke
  `/session-loop` for phase {N+1}."
- If all phases done: "All {total_phases} phases complete. Compact now, then
  re-invoke `/session-loop` for final sweep."

## Constraints

- **One phase per invocation.** Compact between phases to get fresh context.
- **Local compute only.** 2060 Super + CPU. If a phase requires cloud GPU,
  skip it and note in the ledger why.
- **Stay on the session branch.** Never commit to main.
- **Don't modify the plan.** If you discover the plan is wrong, note the issue
  in the ledger's `notes` field. The next invocation can decide whether to
  adjust.
- **Verify success criteria.** Don't mark a phase complete unless the criteria
  from the plan are met (or documented as not met with an explanation).
