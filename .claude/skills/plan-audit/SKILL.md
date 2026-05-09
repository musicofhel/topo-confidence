# /plan-audit

Fresh-eyes audit of an implementation plan. Reads the plan with zero prior
context, audits it soup-to-nuts, and writes a revised version. This is the
mechanism behind the 3-pass planning cycle (v1 -> v2 -> v3).

## When to invoke

Called from `/session-loop` during PLAN_V1, PLAN_V2, and PLAN_V3 stages. Can
also be invoked standalone on any plan file.

## Inputs

`/plan-audit` with no arguments reads the plan path from the session ledger.
`/plan-audit path/to/plan.md` audits a specific file.

## How it works

### Step 1: Determine which plan version to audit

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load
s = load()
print(f'stage={s.stage} plan_file={s.plan_file} plan_version={s.plan_version}')
"
```

- If `stage == PLAN_V1`: Write the initial plan from triage results. Read
  `.claude/triage-staging.md` and create `.claude/plans/session-plan-v1.md`.
- If `stage == PLAN_V2`: Audit plan v1, write v2.
- If `stage == PLAN_V3`: Audit plan v2, write v3.

### Step 2: Read the plan (or triage results for v1)

For v1 (initial plan creation):
- Read `.claude/triage-staging.md`
- Read `NEXT_EXPERIMENTS.md` for existing queue context
- Read `FINDINGS.md` for what's already established
- Write a plan that covers: what to implement, why, estimated phases, risks

For v2 and v3 (audit passes):
- Read ONLY the previous version of the plan
- Do NOT read the triage staging or session history
- You are a fresh reviewer with no prior context on this plan
- The whole point is that you haven't seen this before

### Step 3: Audit checklist

When auditing (v2, v3), evaluate the plan against these criteria:

1. **Feasibility**: Can every step run on a 2060 Super + CPU? Any hidden cloud
   GPU requirements?
2. **Completeness**: Does the plan cover all hits and near-misses from triage?
   Are there obvious experiments it missed?
3. **Order**: Are phases sequenced correctly? Do later phases depend on earlier
   results?
4. **Cost**: Is the total compute time reasonable? Flag anything over 4 hours
   per phase.
5. **Redundancy**: Are any proposed experiments duplicating existing findings
   (F-1..F-10) or queued FEs?
6. **Controls**: Does each experiment have a clear success criterion and a
   null-result interpretation?
7. **Risk**: What's the biggest thing that could go wrong? Is there a
   mitigation?

### Step 4: Write the revised plan

Write the revised plan to the next version file:
- v1 -> `.claude/plans/session-plan-v1.md`
- v2 -> `.claude/plans/session-plan-v2.md`
- v3 -> `.claude/plans/session-plan-v3.md`

Every plan file must have this structure:

```markdown
# Session Plan v{N} — {date}

## Session: {batch_id}

## Audit notes (v2+ only)
- What changed from v{N-1} and why
- Issues found and how they were addressed

## Objective
{One paragraph: what are we trying to accomplish this session?}

## Phases

### Phase 1: {title}
**Goal**: {what this phase produces}
**Experiments**: {list of experiments to run}
**Success criteria**: {how we know it worked}
**Null interpretation**: {what it means if it doesn't work}
**Est. time**: {local compute time}
**Local-runnable**: yes

### Phase 2: {title}
...

## Risks
- {risk 1}: {mitigation}
- {risk 2}: {mitigation}

## Total estimated compute time
{sum of phase times}
```

For v3 (final version), also add:
```markdown
## Implementation order
{Explicit sequence: do phase 1 first, compact, then phase 2, etc.}

## Phase count
{N phases total. This number gets written to the session ledger.}
```

### Step 5: Update session ledger

```bash
cd ~/topo-confidence && .venv/bin/python -c "
from pipeline.session_state import load, save
s = load()
s.plan_version = {N}
s.plan_file = '.claude/plans/session-plan-v{N}.md'
# For v3, also set total_phases:
# s.total_phases = {phase_count}
s.advance_to('{NEXT_STAGE}')
save(s)
"
```

Stage transitions:
- After v1: advance to PLAN_V2
- After v2: advance to PLAN_V3
- After v3: advance to IMPLEMENT, set `total_phases`

After updating, tell the user: "Plan v{N} written. Compact now, then
re-invoke `/session-loop`."

## The fresh-eyes principle

The entire reason for the 3-pass cycle is that a fresh context window catches
things the original author missed. To preserve this:

- v2 and v3 auditors must NOT read the triage staging file
- v2 and v3 auditors must NOT read prior conversation context
- The plan file must be self-contained — if the auditor can't understand the
  plan without external context, the plan needs more detail
- Each pass should genuinely challenge the plan, not rubber-stamp it

"I don't want to spend any money." — every audit pass should verify that
nothing in the plan requires cloud GPU or paid compute.

## Constraints

- **Local compute only.** 2060 Super + CPU. Flag and remove anything else.
- **Self-contained plans.** The plan must be readable cold, with no assumed
  context from this conversation.
- **Spend planning tokens to save implementation tokens.** It's better to
  discover a flaw in the plan now than during phase 3 of implementation.
