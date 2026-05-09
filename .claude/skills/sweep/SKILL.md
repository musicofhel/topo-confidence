# /sweep

Final session wrap. Reads all changes made during the session, commits to the
session branch, writes a handoff file, and marks the session COMPLETE.

## When to invoke

Called from `/session-loop` during the SWEEP stage, after all implementation
phases are done.

## How it works

### Step 1: Read session state

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load
s = load()
print(f'branch={s.branch}')
print(f'batch_id={s.batch_id}')
print(f'completed_phases={s.completed_phases}')
print(f'started={s.started}')
print(f'notes={s.notes}')
"
```

### Step 2: Review all changes

```bash
cd ~/topo-confidence
git checkout {branch}
git diff main --stat
git diff main --name-only
git log main..HEAD --oneline
```

Read through the diff to understand what changed. Categorize changes:
- New experiment scripts
- Result files
- Doc updates (FINDINGS.md, EXPERIMENT_LOG.md, etc.)
- Pipeline code changes
- Config changes

### Step 3: Stage and commit

If there are uncommitted changes:

```bash
cd ~/topo-confidence
git add -A
git commit -m "session {batch_id}: {summary of what was accomplished}

Phases completed: {list}
Key results: {1-2 sentence summary}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Step 4: Write handoff file

Write a handoff file to `.claude/handoff/` following the existing convention:

```bash
ls ~/topo-confidence/.claude/handoff/ 2>/dev/null
```

Create `.claude/handoff/{date}-session-{batch_id}.md`:

```markdown
# Session Handoff — {batch_id}

**Date**: {date}
**Branch**: {branch}
**Status**: Complete

## What was accomplished
{Summary of all phases and their outcomes}

## Key results
{Most important findings from this session's experiments}

## Changes made
{List of significant file changes}

## What's next
{Recommendations for the next session — what to explore, what to avoid}

## Open questions
{Anything unresolved that the next session should address}
```

### Step 5: Update session ledger and validate claims

Before marking complete, run the claims validator:

```bash
cd ~/topo-confidence && python validate_claims.py 2>&1 | tail -5
```

If any claims fail, fix them before completing. The claims invariant is
load-bearing.

Then update the ledger:

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load, save
s = load()
s.advance_to('COMPLETE')
s.notes = 'Session complete. Branch {branch} ready for manual merge.'
save(s)
"
```

### Step 6: Report to user

Tell the user:
- Branch name and how to merge: `git checkout main && git merge {branch}`
- Summary of what was accomplished
- Handoff file location
- Any open questions or recommendations for next session

Do NOT push to remote. The user will handle that manually.

## Constraints

- **Commit to session branch only.** Never commit to main.
- **Do NOT push.** The user merges and pushes manually.
- **Run validate_claims.py.** Don't skip this — the 134/134 internal-PASS
  invariant must hold.
- **Write the handoff.** Future sessions depend on these for continuity.
