# Session Orchestrator — Vertical Slices

## Goal
Close the experiment loop: papers get digested, cheap experiments batch and run
locally (2060 Super + CPU only), results get triaged, follow-ups get planned
through a 3-pass audit cycle, then implemented in phases with compaction between
each. Everything file-driven, no cloud compute.

## Architecture
- **Inner loop**: Existing LangGraph pipeline (select → run → brief → claims → review → promote)
- **Outer loop**: Skill-based session orchestrator with file-based ledger state
- **Compaction = feature**: Each `/compact` gives a fresh-eyes context window

## Vertical Slices

### Slice 1: Ledger + Session State
Create `session-state.md` convention and the `/session-loop` entry skill that
reads it, determines current stage, and dispatches to the right action.

Files: `.claude/skills/session-loop/SKILL.md`, `pipeline/session_state.py`

Testable: Write a session-state file, invoke the skill, verify it reads
and reports the correct stage.

---

### Slice 2: Batch Gate
Add `--max-runs N` and `--local` flags to the pipeline CLI. Add cost filtering
to `select_experiment` (skip FEs requiring GPU > 2060 Super). Add minimum batch
check (need >= 5 runnable FEs before proceeding).

Files: `pipeline/cli.py`, `pipeline/nodes.py`, `pipeline/state.py`

Testable: `python -m pipeline status` shows local-only FE count.
`python -m pipeline run --local --max-runs 5 --dry-run` respects the cap.

---

### Slice 3: Result Triage Skill
`/triage-results` skill reads completed experiment results, identifies which
came back positive, and generates follow-up FE proposals (iterate in different
directions). Writes candidates to a staging file for human review.

Files: `.claude/skills/triage-results/SKILL.md`

Testable: Point it at a completed experiment's brief + results.json, verify
it proposes follow-up directions.

---

### Slice 4: Plan Audit Skill (3-pass cycle)
`/plan-audit` skill implements the fresh-eyes review cycle:
- Reads a plan file
- Audits it soup-to-nuts with zero prior context
- Writes a revised plan (v1 → v2 → v3)
- Updates the session ledger with the plan version

Files: `.claude/skills/plan-audit/SKILL.md`

Testable: Write a deliberately flawed plan, run the skill, verify it catches
issues and writes a revised version.

---

### Slice 5: Phase Implementation Skill
`/phase-impl` skill reads the plan v3, identifies the current phase from the
session ledger, implements it, writes next-steps for phase N+1, and updates
the ledger.

Files: `.claude/skills/phase-impl/SKILL.md`

Testable: Create a 2-phase plan, run the skill for phase 1, verify it updates
the ledger to phase 2 and writes next-steps.

---

### Slice 6: Sweep + Handoff Skill
`/sweep` skill does the final session wrap: reads all changes, commits to
branch, writes a handoff file, updates session-state to COMPLETE.

Files: `.claude/skills/sweep/SKILL.md`

Testable: Make some changes, run the skill, verify it commits to a branch
(not main) and writes a handoff.

---

### Slice 7: Strip RunPod References
Remove RunPod from pipeline code and docs. Replace with local-only compute
references. Update STATE.md, CLAUDE.md.

Files: Various docs, `runpod_launch.sh` (delete or archive)

Testable: `grep -ri runpod pipeline/` returns nothing.

---

## Current Status
- [x] Slice 1: Ledger + Session State
- [x] Slice 2: Batch Gate
- [x] Slice 3: Result Triage Skill
- [x] Slice 4: Plan Audit Skill
- [x] Slice 5: Phase Implementation Skill
- [x] Slice 6: Sweep + Handoff Skill
- [x] Slice 7: Strip RunPod References

## Principles
- Local only: 2060 Super + CPU. No RunPod, no cloud.
- Jeffrey Emanuel style: spend planning tokens to save implementation tokens
- Each skill is stateless — reads ledger, does one thing, writes ledger
- Compaction between skills is intentional (fresh-eyes audit)
- Test each slice before moving to the next
