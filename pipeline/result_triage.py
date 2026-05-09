"""AUROC-based result classifier for autopilot."""

from __future__ import annotations

from typing import Any


def classify(result_json: dict[str, Any]) -> str:
    """Classify an experiment result by AUROC.

    Returns one of: HIT, NEAR_MISS, NULL, INCONCLUSIVE.
    """
    auroc = (
        result_json.get("auroc_oof")
        or result_json.get("auroc_raw_oof")
        or result_json.get("auroc")
    )
    if auroc is None:
        return "INCONCLUSIVE"
    try:
        auroc = float(auroc)
    except (TypeError, ValueError):
        return "INCONCLUSIVE"
    if auroc >= 0.75:
        return "HIT"
    if auroc >= 0.60:
        return "NEAR_MISS"
    if auroc >= 0.52:
        return "NULL"
    return "INCONCLUSIVE"
