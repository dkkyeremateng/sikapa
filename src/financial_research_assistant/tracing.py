"""Optional OpenTelemetry tracing, built from the AgentEvent stream.

No-op unless BOTH hold: opentelemetry is installed (``pip install
'langgraph-agent[tracing]'``) AND tracing is enabled via env
(``AGENT_TRACING=1`` or any ``OTEL_EXPORTER_OTLP_ENDPOINT`` set). Fake runs
are never traced.

Because it consumes only the generic ``AgentEvent`` stream (never framework
internals), this module is identical across every scaffold — the adapter
contract is the single integration point. It emits spans following the
OpenTelemetry GenAI semantic conventions (``invoke_agent`` parent,
``execute_tool`` children, token usage + cost on the parent).

Vendor routing uses standard OTLP env vars — point them at Langfuse,
LangSmith, Arize Phoenix, Grafana Tempo, or any OTLP collector:

    OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 pk:sk>

The GenAI conventions are still experimental; attribute names may shift as
they stabilize. Verify against your backend's expected schema.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from .auth import redact
from .events import AgentEvent
from .pricing import cost_usd

SERVICE_NAME = "langgraph-agent"

_tracer = None
_configured = False


def _enabled() -> bool:
    if os.environ.get("AGENT_TRACING", "").lower() in ("1", "true", "yes"):
        return True
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def _get_tracer():
    """Lazily configure a tracer; return None if opentelemetry is absent."""
    global _tracer, _configured
    if _configured:
        return _tracer
    _configured = True
    try:
        from opentelemetry import trace  # pyright: ignore[reportMissingImports]  (optional extra)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # pyright: ignore[reportMissingImports]  (optional extra)
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports]  (optional extra)
        from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports]  (optional extra)
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # pyright: ignore[reportMissingImports]  (optional extra)
    except Exception:  # opentelemetry not installed → stay a no-op
        _tracer = None
        return None
    provider = trace.get_tracer_provider()
    # Only install our own provider if the host app hasn't configured one.
    if not isinstance(provider, TracerProvider):
        service = os.environ.get("OTEL_SERVICE_NAME", SERVICE_NAME)
        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(SERVICE_NAME)
    return _tracer


async def traced(
    events: AsyncIterator[AgentEvent],
    *,
    user_msg: str,
    session_id: str,
    model: str,
    fake: bool,
) -> AsyncIterator[AgentEvent]:
    """Pass ``events`` through unchanged, emitting OTel spans as a side effect.

    Transparent when tracing is disabled/unavailable or ``fake`` is set, so
    callers can always wrap ``run_turn`` with it.

    Everything written onto a span goes through ``auth.redact`` first — the
    prompt and the final answer included. Spans leave the machine for a
    third-party backend and are retained there, and a user pasting a key into
    the chat ("here's my token, check the balance") is the ordinary case rather
    than the exotic one.
    """
    tracer = None if (fake or not _enabled()) else _get_tracer()
    if tracer is None:
        async for ev in events:
            yield ev
        return

    from opentelemetry.trace import Status, StatusCode  # pyright: ignore[reportMissingImports]  (optional extra)

    with tracer.start_as_current_span(f"invoke_agent {SERVICE_NAME}") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", SERVICE_NAME)
        span.set_attribute("gen_ai.conversation.id", session_id)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.input.messages", redact(user_msg[:2000]))
        tool_spans: dict[str, Any] = {}
        final_text = ""
        # usage arrives as per-call deltas; sum them
        tok_in = tok_out = tok_cache = tok_write = 0
        try:
            async for ev in events:
                if ev.kind == "tool_start":
                    ts = tracer.start_span(f"execute_tool {ev.tool}")
                    ts.set_attribute("gen_ai.operation.name", "execute_tool")
                    ts.set_attribute("gen_ai.tool.name", ev.tool)
                    if ev.call_id:
                        ts.set_attribute("gen_ai.tool.call.id", ev.call_id)
                    if ev.detail:
                        ts.set_attribute("gen_ai.tool.call.arguments", redact(ev.detail[:2000]))
                    tool_spans[ev.call_id or ev.tool] = ts
                elif ev.kind == "tool_end":
                    ts = tool_spans.pop(ev.call_id or ev.tool, None)
                    if ts is not None:
                        if ev.detail:
                            ts.set_attribute("gen_ai.tool.call.result", redact(ev.detail[:2000]))
                        ts.set_status(Status(StatusCode.OK if ev.ok else StatusCode.ERROR))
                        ts.end()
                elif ev.kind == "usage":
                    tok_in += ev.tokens_in
                    tok_out += ev.tokens_out
                    tok_cache += ev.tokens_cache
                    tok_write += ev.tokens_cache_write
                    span.set_attribute("gen_ai.usage.input_tokens", tok_in)
                    span.set_attribute("gen_ai.usage.output_tokens", tok_out)
                    if tok_cache:
                        span.set_attribute("gen_ai.usage.cached_input_tokens", tok_cache)
                    if tok_write:
                        span.set_attribute("gen_ai.usage.cache_creation_input_tokens", tok_write)
                    cost = cost_usd(model, tok_in, tok_out, tok_cache, tok_write)
                    if cost is not None:
                        span.set_attribute("gen_ai.usage.cost_usd", round(cost, 6))
                elif ev.kind == "final":
                    final_text = ev.text
                elif ev.kind == "error":
                    span.set_status(Status(StatusCode.ERROR, redact(ev.text[:200])))
                yield ev
            if final_text:
                span.set_attribute("gen_ai.output.messages", redact(final_text[:2000]))
        finally:
            for ts in tool_spans.values():
                ts.end()  # close any tool span left open by an error mid-turn
