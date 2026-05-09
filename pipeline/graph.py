"""LangGraph state machine for the experiment pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from pipeline.state import ExperimentState


def _route_after_run(state: ExperimentState) -> str:
    if state.get("script_exit_code", 1) != 0:
        return "handle_failure"
    return "parse_results"


def _route_after_review(state: ExperimentState) -> str:
    verdict = state.get("human_verdict", "reject")
    if verdict == "reject":
        return "select_experiment"
    if verdict == "edit":
        return "revise_artifacts"
    return "write_brief"


def _route_after_promote(state: ExperimentState) -> str:
    return "select_experiment"


def build_graph(checkpointer=None, *, no_review: bool = False):
    from pipeline.nodes import (
        extract_claims,
        generate_brief,
        handle_failure,
        interpret_findings,
        parse_results,
        promote,
        run_experiment,
        select_experiment,
        write_brief,
        write_claims,
    )

    builder = StateGraph(ExperimentState)

    builder.add_node("select_experiment", select_experiment)
    builder.add_node("run_experiment", run_experiment)
    builder.add_node("parse_results", parse_results)
    builder.add_node("generate_brief", generate_brief)
    builder.add_node("extract_claims", extract_claims)
    builder.add_node("interpret_findings", interpret_findings)
    builder.add_node("write_brief", write_brief)
    builder.add_node("write_claims", write_claims)
    builder.add_node("promote", promote)
    builder.add_node("handle_failure", handle_failure)

    if not no_review:
        from pipeline.nodes import review_gate, revise_artifacts

        builder.add_node("review_gate", review_gate)
        builder.add_node("revise_artifacts", revise_artifacts)

    builder.add_edge(START, "select_experiment")
    builder.add_conditional_edges(
        "select_experiment",
        lambda s: END if s.get("current_fe") is None else "run_experiment",
    )
    builder.add_conditional_edges("run_experiment", _route_after_run)
    builder.add_edge("parse_results", "generate_brief")
    builder.add_edge("generate_brief", "extract_claims")
    builder.add_edge("extract_claims", "interpret_findings")

    if no_review:
        builder.add_edge("interpret_findings", "write_brief")
    else:
        builder.add_edge("interpret_findings", "review_gate")
        builder.add_conditional_edges("review_gate", _route_after_review)
        builder.add_edge("revise_artifacts", "review_gate")

    builder.add_edge("write_brief", "write_claims")
    builder.add_edge("write_claims", "promote")
    builder.add_conditional_edges("promote", _route_after_promote)
    builder.add_edge("handle_failure", "select_experiment")

    return builder.compile(checkpointer=checkpointer)
