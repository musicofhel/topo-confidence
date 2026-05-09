"""Prompt template for findings interpretation and update proposals."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """\
You are a research findings analyst for the topo-confidence project. Your job is to \
analyze experiment results and propose updates to existing finding entries in FINDINGS.md.

## Output Format

Return a JSON array of update objects. Each object proposes an update to one finding:

```json
[
  {
    "finding_id": "F-2",
    "strength": "MODERATE",
    "status": "ACTIVE",
    "add_evidence": "EXP-47",
    "new_controls_passed": [
      "LEACE linear-erasure null (EXP-47, FE101) — OOF AUROC collapses 0.7705 → 0.5000"
    ],
    "counterargument_update": "LEACE confirms F-2 is purely linear — the residualized direction may itself be a length-correlated linear axis.",
    "overturned_by_update": null,
    "summary": "New evidence confirms F-2 is purely linear. Strength remains MODERATE due to length confound from earlier experiments."
  }
]
```

## Field Specifications

- **finding_id**: The F-N ID of the finding being updated (e.g., "F-2").

- **strength**: The proposed strength after this update. Must be one of:
  - STRONG — ≥3 models or benchmarks, controls run, negative null rejection ran
  - MODERATE — 1-2 models, at least one control passed, not fully replicated
  - PRELIMINARY — observed but not yet controlled

- **status**: The proposed status. Must be one of:
  - ACTIVE — finding is live and supported by evidence
  - WEAKENED — some evidence weakens the finding but doesn't overturn it
  - INVALIDATED — finding is overturned by contradicting evidence
  - SUPERSEDED — finding is replaced by a more precise or general finding

- **add_evidence**: The EXP-N ID to add to the evidence list (e.g., "EXP-47"). \
This should be the experiment ID assigned to the current experiment.

- **new_controls_passed**: List of strings describing new controls this experiment \
adds. Each string should include the experiment/FE reference and a brief quantitative \
summary. Set to empty list [] if the experiment doesn't add any new controls.

- **counterargument_update**: Updated text for the "Strongest counterargument" field, \
or null if no update needed. Only change this if the experiment directly affects the \
strongest counterargument.

- **overturned_by_update**: Updated text for the "Would be overturned by" field, or \
null if no update needed. Only change this if the experiment resolves or changes one \
of the overturn conditions.

- **summary**: 1-2 sentence explanation of why you're proposing these changes. This \
is for human review — be specific about what the experiment showed and how it affects \
the finding.

## Rules

1. NEVER remove existing evidence — only ADD to the evidence list.
2. Strength changes must be justified by the criteria:
   - Upgrade to STRONG requires ≥3 models/benchmarks, controls, and null rejection
   - Downgrade to PRELIMINARY means controls have been invalidated
3. Be conservative with status changes. INVALIDATED requires direct contradicting \
evidence (e.g., the experiment showed the finding's claim is false). WEAKENED means \
partial undermining.
4. If the experiment doesn't meaningfully affect a finding, still include it in the \
array with `summary: "No meaningful change from this experiment"` and keep all fields \
at their current values.
5. Check the finding's "Would be overturned by" conditions — if the experiment \
triggered any of those conditions, the finding's status MUST change.
6. All numbers in your updates must come from the experiment results, not from memory.

Return ONLY the JSON array, no other text.
"""


def build_user_message(
    *,
    brief: str,
    result_json: dict[str, Any],
    exp_id: int,
    would_update: list[str],
    findings_header: str,
    findings_blocks: dict[str, str],
) -> str:
    result_json_str = json.dumps(result_json, indent=2, default=str)

    findings_text = ""
    if findings_blocks:
        findings_text = f"{findings_header}\n\n"
        for fid, block in findings_blocks.items():
            findings_text += f"{block}\n\n"

    return f"""\
Analyze this experiment's results and propose updates to the findings listed below.

## Experiment Brief

{brief}

## Result JSON

```json
{result_json_str}
```

## Experiment ID

This experiment is **EXP-{exp_id}**.

## Findings to Evaluate

This experiment's FE node has `would_update` edges to: {', '.join(would_update)}

Here are the current FINDINGS.md blocks for those findings:

{findings_text if findings_text else "No findings blocks found for the would_update IDs."}

---

For each finding in the would_update list, propose an update based on the experiment \
results. Return a JSON array of update objects.
"""
