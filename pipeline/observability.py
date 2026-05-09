"""
Dual-backend OTEL observability: Langfuse (LLM traces) + Jaeger (everything).

Usage:
    from pipeline.observability import init_tracing, get_tracer, score_experiment

    init_tracing()  # call once at startup
    tracer = get_tracer("pipeline.nodes")

    with tracer.start_as_current_span("experiment.run", attributes={"fe_id": "P11-FE101"}) as span:
        ...

    score_experiment(trace_id, auroc=0.77, claims_pass=True)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

_initialized = False


def init_tracing(
    *,
    jaeger_endpoint: str = "http://localhost:4318/v1/traces",
    service_name: str = "topo-pipeline",
    enable_langfuse: bool = True,
    enable_jaeger: bool = True,
) -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if enable_jaeger:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=jaeger_endpoint))
        )

    if enable_langfuse:
        try:
            from langfuse.opentelemetry import LangfuseSpanProcessor

            _blocked_scopes = {"neo4j", "subprocess", "fileio", "urllib3", "urllib"}
            provider.add_span_processor(
                LangfuseSpanProcessor(
                    should_export_span=lambda span: (
                        span.instrumentation_scope.name not in _blocked_scopes
                    ),
                )
            )
        except ImportError:
            print(
                "WARNING: langfuse not installed — LLM traces will not be sent to Langfuse"
            )

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "pipeline") -> trace.Tracer:
    return trace.get_tracer(name)


def score_experiment(
    trace_id: str,
    *,
    auroc: float | None = None,
    claims_pass_rate: float | None = None,
    human_approved: bool | None = None,
    promotion_success: bool | None = None,
) -> None:
    """Attach numeric/boolean scores to a Langfuse trace for trend tracking."""
    try:
        from langfuse import get_client

        lf = get_client()
    except (ImportError, Exception):
        return

    scores: list[tuple[str, Any, str]] = []
    if auroc is not None:
        scores.append(("auroc", auroc, "NUMERIC"))
    if claims_pass_rate is not None:
        scores.append(("claims_pass_rate", claims_pass_rate, "NUMERIC"))
    if human_approved is not None:
        scores.append(("human_approved", human_approved, "BOOLEAN"))
    if promotion_success is not None:
        scores.append(("promotion_success", promotion_success, "BOOLEAN"))

    for name, value, data_type in scores:
        lf.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,
        )


@contextmanager
def traced_subprocess(tracer: trace.Tracer, name: str, **attributes: Any):
    """Context manager that wraps a subprocess call in an OTEL span."""
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
