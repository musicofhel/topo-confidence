from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class ExperimentState(TypedDict, total=False):
    current_fe: dict[str, Any] | None

    script_path: str
    script_exit_code: int
    result_json_path: str
    result_json: dict[str, Any] | None

    brief_markdown: str
    claims: list[dict[str, Any]]
    findings_updates: list[dict[str, Any]]
    experiment_log_entry: str

    review_payload: dict[str, Any]
    human_verdict: str
    human_edits: dict[str, Any] | None

    dry_run: bool
    local_only: bool
    max_runs: int

    promotion_stdout: str
    claims_validation_result: str

    completed_this_session: Annotated[list[str], add]
    errors: Annotated[list[dict[str, Any]], add]
