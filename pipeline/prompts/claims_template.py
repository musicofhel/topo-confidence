"""Prompt template for claims extraction from experiment results."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """\
You are a claims extractor for the topo-confidence research project. Your job is to \
identify quantitative values from an experiment's result JSON and produce structured \
claim entries that can be back-checked against the source data.

## Output Format

Return a JSON array of claim objects. Each object has these fields:

```json
[
  {
    "cid": "fe101-auroc-raw",
    "description": "FE101 raw DoM 5-fold OOF AUROC, sanity anchor",
    "file": "pathway11_h100/leace_erasure/results.json",
    "path": ["auroc_raw_oof"],
    "expected": 0.7705,
    "tol": 0.005,
    "labels": "1024tok"
  }
]
```

## Field Specifications

- **cid**: Short identifier. Pattern: `fe{num}-{metric-slug}` where `num` is the \
lowercase FE number extracted from the FE ID (e.g., FE101 → "101", FE719 → "719"). \
The metric-slug should be a concise kebab-case name (e.g., "auroc-raw", "spearman-rho", \
"collapse-pp").

- **description**: Human-readable description of the claim. Include the FE ID, the \
metric name, and the value. Format: "{FE_ID_short} {metric description} = {value}".

- **file**: Path to the result JSON file, relative to the repo root. Use exactly the \
path provided in the input.

- **path**: A list of string or integer keys to drill into the JSON to reach the value. \
For nested objects use string keys. For arrays use integer indices (0-based). This must \
be a valid path that reaches the exact value in the JSON.

- **expected**: The exact numeric value from the JSON. For floats, use the full precision \
from the JSON (don't round). For integers, use the integer value.

- **tol**: Absolute tolerance for float comparison. Guidelines:
  - AUROC, accuracy, proportions: 0.005
  - Cosine similarities: 0.003
  - Counts (integers): 0.5
  - Ratios with high precision: 1e-5
  - Norms, large values: 0.05 or 0.5
  - Default: 0.005

- **labels**: Always "1024tok" unless explicitly stated otherwise in the experiment context.

## Rules

1. Only extract values that exist in the provided result JSON. Never hallucinate paths or values.
2. Focus on headline metrics — the 3-8 most important quantitative results. Don't extract \
every single nested value.
3. The `path` must be valid: each element must match an actual key or index in the JSON.
4. For nested paths, list each key in order: `["headline", "refuse_and_spend_coverage_0.5", \
"prefill", "acc_on_answered"]` → drills `json["headline"]["refuse_and_spend_coverage_0.5"]\
["prefill"]["acc_on_answered"]`.
5. If the brief's "New claims" section lists specific claims, use those as your primary guide \
for what to extract. Match their cid naming and values.

## Few-Shot Examples

Here are real Claim entries from validate_claims.py to show the expected style:

```python
Claim("fe101-auroc-raw", "FE101 raw DoM 5-fold OOF AUROC, sanity anchor",
      "pathway11_h100/leace_erasure/results.json",
      ["auroc_raw_oof"], 0.7705, 0.002, "1024tok"),

Claim("pca-dom-pc1-cosine", "FE291 cosine between supervised DoM and unsupervised PC1 \
= 0.9216 (near-identical directions)",
      "pathway11_h100/pca_covariance/results.json",
      ["dom_pc1_cosine"], 0.9216, 0.003, "1024tok"),

Claim("fe459-7b-auroc", "FE459 7B L19 prefill DoM AUROC (OOF) = 0.874",
      "pathway11_h100/results/fe459_cross_model_dom.json",
      ["model_7b", "auroc_oof"], 0.8738071935404943, 0.005, "1024tok"),
```

Return ONLY the JSON array, no other text.
"""


def build_user_message(
    *,
    result_json: dict[str, Any],
    result_json_path: str,
    script_path: str,
    fe_id: str,
    brief_claims_section: str,
) -> str:
    fe_num = fe_id.split("-")[-1].lower()
    result_json_str = json.dumps(result_json, indent=2, default=str)

    return f"""\
Extract quantitative claims from this experiment's result JSON.

## Experiment

- **FE ID:** {fe_id}
- **FE number for cid prefix:** fe{fe_num}
- **Result JSON path:** `{result_json_path}`
- **Script path:** `{script_path}`

## Result JSON

```json
{result_json_str}
```

## Claims Suggested by Brief

The experiment brief suggested these claims (use as guidance for what to extract):

{brief_claims_section if brief_claims_section else "No claims section found in brief."}

Extract the headline claims from the result JSON. Return a JSON array of claim objects.
"""
