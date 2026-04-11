# CORAL run: Topological AUROC — 2026-04-08 to 2026-04-11

This directory is a snapshot of a 4-day CORAL (multi-agent autonomous
evolution) run that optimized topological features for predicting
Qwen2.5-1.5B correctness on MATH-500. All files here come from a frozen run
at `~/coral-tasks/topo-auroc/results/topological-auroc/2026-04-08_000744/`
(not in this repo — only the distilled outputs are).

## TL;DR

| Metric | Value |
|---|---|
| Task | Predict Qwen2.5-1.5B correctness on MATH-500 from PH features of hidden states |
| Baseline (prior 7-feature hand-crafted) | **AUROC 0.713** |
| CORAL target | AUROC 0.75 |
| **CORAL final best** | **AUROC 0.9779** (+0.265, +37% relative) |
| Winning commit | `4eeb2a28` (agent-3) |
| Winning title | "ADD #299 DROP+ADD on 76-base (0.9755, eval #298)" |
| Winning time | 2026-04-11 22:21:21 |
| Total attempts | 676 (674 scored, 1 crashed) |
| Run duration | Apr 8 00:07 → Apr 11 22:42 (≈4 days) |
| Model | `claude-opus` × 3 agents, max_turns=200 per session |
| Agents active at end | agent-2, agent-3 (both converging) |
| Agent-1 | idle 33+ hours before stop — effectively dead |

The CORAL run **massively overshot** the 0.75 target. Best AUROC 0.9779 is
approaching the Bayes limit for this classifier (50-fold CV, balanced
LogisticRegression on 500 examples where 57 are positive — 11.4% imbalance).

## What was actually learned (the real finding)

The task asked "can we beat 0.75". The actual answer the CORAL agents
produced is different and more interesting: **a drop-add cycle with
super-additive stacks of depth-6 to depth-8+ recursive product features can
push a 7-feature topological baseline to 0.9779 AUROC**. The winning
features are not cleaner PH statistics — they are deeply nested products of
PH statistics with layer-state geometry with derivatives with k-way
interaction terms.

In plain English: the signal that predicts Qwen2.5-1.5B correctness on
MATH-500 is not any single topological feature. It is the orchestrated
interaction of many weak signals through recursive multiplication and
selective feature dropping. No single feature in the winning set would pass
a significance test. The whole set does.

This has three concrete implications for `topo-confidence` as a research
project:

1. **Feature engineering is a lost cause for this dataset**. Adding more
   hand-crafted PH features will not move the needle. The search space is
   so high-dimensional that automated search with iterative drop/add is the
   only practical approach to find the right basis.
2. **The 0.713 baseline was undersold**. It was not a ceiling of what PH
   can do — it was the ceiling of what PH can do with 7 hand-crafted
   features. Automated search crossed 0.90 in about 96 hours.
3. **Verify the result is not overfitting 50-fold CV**. 676 attempts
   across 4 days is enough to overfit a 500-problem CV protocol. Before
   reading anything deep into the 0.9779 number, re-run the winning
   feature set on a held-out MATH split that the agents never saw.

## Directory layout

```
coral_run_2026_04_08/
├── README.md                      ← you are here
├── task.yaml                      ← the CORAL task spec (read-only)
├── baseline_notes.md              ← the original 7-feature baseline doc
├── winning_features.py            ← the 0.9779 AUROC feature extractor (1407 lines)
├── leaderboard_top50.md           ← top 50 attempts ranked by score
├── final_status.txt               ← coral status output at stop-time
├── notes/
│   ├── _connections.md            ← cross-category pattern map (agent-authored)
│   ├── _open-questions.md         ← unresolved contradictions and gaps
│   ├── _synthesis/                ← 22 synthesis notes distilling 284 raw ones
│   └── breakthroughs/             ← 69 notes tagged breakthrough/ceiling/plateau
└── skills/                        ← 6 reusable procedures the agents packaged
    ├── correlation-swap/
    ├── drop-add-cycle/
    ├── leader-ceiling-diagnosis/
    ├── local-cv-harness/
    ├── skill-creator/
    └── weak-feature-trap/
```

## How to read this

If you only have 5 minutes:
1. `leaderboard_top50.md` — see the score progression
2. `notes/_synthesis/product-interactions-break-plateau.md` — the single
   biggest methodological finding
3. `notes/_synthesis/self-product-composition-depth-era.md` — the era of
   depth-4+ recursive products, what actually drives the late-run gains
4. `notes/_open-questions.md` — what the agents flagged as unresolved

If you have 30 minutes:
5. `notes/_synthesis/drop-add-cycle-0.91-to-0.912.md` — the recipe that
   replaces feature engineering with search
6. `notes/_synthesis/agent2-drop-add-cycle-0.95715-to-0.96622.md` —
   mid-run push past 0.96
7. `notes/_synthesis/plateau-0.82-is-statistical-ceiling.md` — why the
   agents kept hitting walls and how they broke through
8. `skills/drop-add-cycle/SKILL.md` — the packaged procedure
9. Skim `winning_features.py` to get a sense of the 100+ internal features
   and the drop-to-output mapping

If you are trying to reproduce or build on this:
10. `task.yaml` + `baseline_notes.md` + `winning_features.py` + the
    unchanged data files under `~/topo-confidence/data/experiment1_v2/`
    are sufficient to re-run
11. `skills/local-cv-harness/SKILL.md` — how the agents got deterministic
    local grader-exact matches (23 consecutive exact matches toward the end)
12. `skills/leader-ceiling-diagnosis/SKILL.md` — how to tell when you have
    plateaued vs when another agent has simply pulled ahead

## The score progression in one paragraph

Agents started near the 0.713 baseline. agent-1 (active early, dead for the
last 33 hours) worked through single-feature swaps, stalled at 0.8202,
identified it as a statistical ceiling, and later sessions stopped
improving. agent-2 broke into the 0.80s via product interactions, crossed
0.85 with cross-modal features, hit 0.90 with the self-product composition
depth era, and crossed 0.95 with super-additive quadruples of 6-way and
7-way recursive products plus deliberate DROPs. agent-3 paralleled agent-2's
trajectory, was briefly at 0.8682 (leader), fell to second, then reclaimed
#1 at 0.9779 on eval #299 via a DROP(11,49)+ADD(4-stack super-additive
7-way product) joint move. See `leaderboard_top50.md` for exact timestamps.

## The recipe that ended up working (after ~300 evals per lead agent)

From `notes/_synthesis/drop-add-cycle-0.91-to-0.912.md` and the agent-2 /
agent-3 session handoffs:

1. Maintain a base feature set (`N-base`, e.g. 76-base, 84-base).
2. Run a scan script (`scan_<N>.py`) that enumerates candidate new features
   — products of existing "hot" features with new PH statistics, new
   layer-state coordinates, or new monster anchors.
3. Verify top candidates in a multi-seed harness (`verify_<N>.py`) to find
   the ones with `pos >= 0.6` across 10 seeds and `ms_mean > 0`.
4. Run a combinatorial joint DROP+ADD search over all (drop-combo,
   k-stack) pairs. Look for **super-additive** joints: cases where the
   joint move delta exceeds the sum of the individual drop and stack deltas.
5. The winning joint is usually a DROP of 2 weak/parasitic features plus an
   ADD of a 3-stack or 4-stack of verified candidates. The 4-stack
   multiplies 2.5–2.7× the maximum solo delta in that scan.
6. Verify the move reproduces under the local grader-exact harness. The
   winning moves in the last 25 evals reproduced **exactly** with the
   remote CORAL grader — see `skills/local-cv-harness/SKILL.md`.
7. Document as a note and repeat.

The drop is essential. Late-run gains came not from adding more features
but from dropping weak/parasitic features whose signal had been absorbed
by the new recursive products. `skills/weak-feature-trap/SKILL.md`
diagnoses this pattern: features that were hot in round N become parasitic
in round N+5 because their information is now contained in downstream
products.

## The Künneth / independence link (worth thinking about)

The multi-agent domain sweep on `att-docs` flagged the **Künneth
cross-term insight**: binding score measures topological *independence*,
not coupling. In this run, the depth-6+ recursive products the agents
settled on are effectively computing high-order topological interactions
— and the LR classifier is learning to weigh them. The fact that the
winning features are products of products of products implies the
predictive signal lives in how topological features *combine*, not in any
of them alone.

This would be a publishable finding *if* it survives a held-out split.
See "Open questions" below.

## Open questions (from `notes/_open-questions.md` and my own read)

1. **Does 0.9779 hold on a held-out MATH split the agents never saw?**
   This is the blocking question. 676 attempts against a 50-fold CV
   protocol is enough to overfit. Run the winning features on a
   randomly-selected 100-problem subset withheld before CORAL started —
   if AUROC drops below 0.85, the number is inflated.
2. **Does the recursive product structure generalize to other models?**
   The task trained on Qwen2.5-1.5B-Instruct hidden states only. Run
   `winning_features.py` on Qwen2.5-7B or Llama-3-8B hidden states and
   compare. If only the first 2–3 features transfer, the depth of the
   nesting is model-specific.
3. **Why did agent-1 die at 0.8202?** agent-1 reached its statistical
   ceiling, detected it, and ran out of moves. agent-2 and agent-3 broke
   through via product interactions. Whether agent-1's session hit a
   Claude Code turn limit, a bug, or simply gave up is not clear from the
   notes. See `agent1-session-hard-stop-0.8202-ceiling.md`.
4. **Is the local grader-exact match count (23 consecutive) a diagnostic
   or a trap?** Late-run moves reproduced to `1e-6` under the local
   harness. The harness is itself deterministic, so this is more a
   reproducibility check than a generalization signal. Don't mistake it
   for test-set performance.
5. **What's in `winning_features.py` lines 280-1407?** The file has
   `extract_features()` at line 281 and ~222 `features[i] = ...` lines
   after that. Skimming it shows recursive product expressions (`rb11`
   layers, `p3/p4/p5/p6/p7/p8` axes, "hot" features, "monster" anchors)
   whose semantics are documented in the synthesis notes. A careful
   read-through with the synthesis notes open is the only way to
   understand what each output feature means. This is not a
   human-authored file and will not be easily refactored.

## How the agents worked — infrastructure notes

- **Model**: `claude-opus`, 3 agents, 200 max_turns per Claude Code session.
  agent-1 ran 16,985 "sessions" (likely byte-rate counter, not literal
  sessions). agent-2 and agent-3 ran 18,401 and 18,328. Total CPU:
  manager+agents consumed 506 CPU-minutes over 4 days.
- **Scoring**: 50-fold StratifiedKFold, LogisticRegression with
  `class_weight="balanced"` (imbalance ~11.4%). Timeout 300s per grader run.
- **Shared state**: `sharing.attempts=true, notes=true, skills=true`.
  Agents could read each other's attempts, notes, and skill packages. They
  also shared a "heartbeat" `consolidate` action that fired every 10
  attempts telling them to synthesize the notes dir. This is why there
  are 22 `_synthesis/` notes.
- **Data**: read-only absolute paths in the task. Data never moved.
- **Killed by**: `coral stop` at 2026-04-11 ~late evening. 2 agents were
  still ACTIVE at stop-time, still producing new attempts roughly every
  20–40 minutes, delta per attempt shrinking into `<0.001 AUROC` territory.

## What is NOT in this directory

For size and cleanliness, the following are deliberately omitted:

- **The 3 agent git worktrees** (`repo/.git/worktrees/agent-{1,2,3}/`),
  which contain 676 commits of intermediate feature.py states, `.venv`s,
  and scan script outputs. Lives at
  `~/coral-tasks/topo-auroc/results/topological-auroc/2026-04-08_000744/`
  and totals **2.7 GB**. If you need the full history, `git log` the
  agent-3 worktree for the winning lineage.
- **262 non-synthesis raw notes**. The 22 synthesis notes and the 69
  breakthrough-tagged notes capture the signal. The rest are iteration
  logs.
- **The grader code**. The CORAL-internal grader calls `features.py` in a
  subprocess; the task.yaml does not include grader source. The grader is
  a 50-fold CV LogisticRegression run, which is reproducible from the task
  description alone.
- **Scan scripts** (`scan_*.py`) used by agents. These were one-off
  search harnesses — dozens of them, each tied to a specific N-base.
  `agents/agent-3/` contains ~20 of them if you want to mine them.

## Resurrection instructions

To continue the CORAL run:

```bash
cd ~/coral-tasks/topo-auroc
uv run --project ~/CORAL coral resume
```

To inspect a specific attempt from the frozen run:

```bash
cd ~/coral-tasks/topo-auroc
uv run --project ~/CORAL coral show 4eeb2a28 --diff   # winner
uv run --project ~/CORAL coral show c40ef4ab          # #2
```

To reproduce the winning score against the CORAL grader:

```bash
cp coral_run_2026_04_08/winning_features.py \
   ~/coral-tasks/topo-auroc/seed/features.py
cd ~/coral-tasks/topo-auroc
uv run --project ~/CORAL coral validate .
```

To reproduce against a held-out split (recommended — see open questions):

```python
# in a fresh script, use the same features.py but split data differently
import numpy as np
from coral_run_2026_04_08.winning_features import extract_features
# ... load trajectories, layer_states, labels ...
# held_idx = np.random.RandomState(9999).choice(500, 100, replace=False)
# train on not-held, test on held, compute AUROC
```

## Provenance

This directory was created on `coral-topo-auroc-run` branch after the
CORAL run was manually stopped. Nothing has been merged into `main`.

- CORAL run dir: `~/coral-tasks/topo-auroc/results/topological-auroc/2026-04-08_000744/`
- CORAL version: 0.2.0
- Task spec: `~/coral-tasks/topo-auroc/task.yaml` (copied here as
  `task.yaml`)
- Winning commit: `4eeb2a28` in the agent-3 git worktree
- Branch owner: `musicofhel`
