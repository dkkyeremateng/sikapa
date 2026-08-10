"""What tracing writes onto a span leaves the machine.

Spans are shipped to whatever OTLP backend the operator points at — Langfuse,
LangSmith, a shared Tempo — and retained there. So every attribute has to go
through ``auth.redact`` first, including the two that carry free text the user
typed and the model wrote back.

The opentelemetry SDK is an optional extra, so these tests stand up a recording
tracer and a stub ``opentelemetry.trace`` rather than requiring it to be
installed.
"""

import sys
import types

import pytest

from financial_research_assistant import tracing
from financial_research_assistant.events import AgentEvent

# Shaped like a real Anthropic key so `redact`'s pattern matches it without the
# process having to be holding the credential.
SECRET = "sk-ant-api03-0123456789abcdefghijKLMNOP"


class _RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, object] = {}
        self.status = None
        self.ended = False

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.status = status

    def end(self) -> None:
        self.ended = True

    def __enter__(self) -> "_RecordingSpan":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.ended = True
        return False


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_span(self, name: str) -> _RecordingSpan:
        span = _RecordingSpan(name)
        self.spans.append(span)
        return span

    def start_span(self, name: str) -> _RecordingSpan:
        span = _RecordingSpan(name)
        self.spans.append(span)
        return span


@pytest.fixture
def tracer(monkeypatch):
    """A tracer that records instead of exporting, with the optional SDK stubbed."""
    otel = types.ModuleType("opentelemetry")
    trace_mod = types.ModuleType("opentelemetry.trace")

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    trace_mod.StatusCode = StatusCode
    trace_mod.Status = lambda code, description="": (code, description)
    otel.trace = trace_mod
    monkeypatch.setitem(sys.modules, "opentelemetry", otel)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)

    recorder = _RecordingTracer()
    monkeypatch.setenv("AGENT_TRACING", "1")
    monkeypatch.setattr(tracing, "_get_tracer", lambda: recorder)
    return recorder


async def _run(tracer, *, user_msg, events):
    async def stream():
        for ev in events:
            yield ev

    out = []
    async for ev in tracing.traced(
        stream(), user_msg=user_msg, session_id="s1", model="gpt-4o-mini", fake=False
    ):
        out.append(ev)
    return out, tracer.spans[0]


async def test_the_prompt_is_redacted_before_it_reaches_the_span(tracer):
    """A user pasting a key into the chat is ordinary, not exotic — 'here's my
    token, check the balance'."""
    _out, span = await _run(
        tracer,
        user_msg=f"use my key {SECRET} and check AAPL",
        events=[AgentEvent(kind="final", text="AAPL is fine")],
    )
    attr = span.attributes["gen_ai.input.messages"]
    assert SECRET not in attr
    assert "***" in attr
    assert "check AAPL" in attr, "redaction must mask the key, not eat the prompt"


async def test_the_answer_is_redacted_before_it_reaches_the_span(tracer):
    """The model echoes the prompt back constantly — confirming a setting, quoting
    an error — so masking only the input leaks the same secret out the other side."""
    _out, span = await _run(
        tracer,
        user_msg="what key am I using?",
        events=[AgentEvent(kind="final", text=f"You're configured with {SECRET}.")],
    )
    attr = span.attributes["gen_ai.output.messages"]
    assert SECRET not in attr
    assert "***" in attr


async def test_events_pass_through_unchanged(tracer):
    """Redaction is for the span only: the user's own terminal must still show
    what they typed and what the model said."""
    answer = f"You're configured with {SECRET}."
    out, _span = await _run(
        tracer,
        user_msg="what key am I using?",
        events=[AgentEvent(kind="status", text="thinking"), AgentEvent(kind="final", text=answer)],
    )
    assert [ev.kind for ev in out] == ["status", "final"]
    assert out[-1].text == answer
