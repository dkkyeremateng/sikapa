"""Adapter, checkpointer, compaction, and real graph-session tests."""

from financial_research_assistant.adapter import run_turn

from .helpers.fakes import fake_ibkr_tools_session as _fake_ibkr_tools_session
from .helpers.graphs import (
    scripted_think_graph as _scripted_think_graph,
    scripted_tool_graph as _scripted_tool_graph,
)


async def test_stream_events_pair_tool_start_and_end():
    """The real-mode streaming loop turns AIMessage.tool_calls into a
    tool_start and the matching ToolMessage into the paired tool_end."""
    from financial_research_assistant.adapter import _stream_events

    graph = _scripted_tool_graph()
    inputs = {"messages": [{"role": "user", "content": "count these"}]}
    events = [ev async for ev in _stream_events(graph, inputs, {})]

    starts = [ev for ev in events if ev.kind == "tool_start"]
    ends = [ev for ev in events if ev.kind == "tool_end"]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].tool == ends[0].tool == "word_count"
    assert starts[0].call_id == ends[0].call_id == "call_1"
    assert "hi there" in starts[0].detail  # args JSON snippet
    assert ends[0].detail == "2"  # result snippet
    assert ends[0].ok
    assert ends[0].duration >= 0.0
    assert events.index(starts[0]) < events.index(ends[0])


async def test_checkpointer_ctx_defaults_to_memory(monkeypatch):
    """With no FINANCIAL_RESEARCH_CHECKPOINT_DB, checkpointer_ctx yields the cached
    in-process MemorySaver (same instance across turns) — unchanged default."""
    from langgraph.checkpoint.memory import MemorySaver

    from financial_research_assistant import adapter

    monkeypatch.delenv("FINANCIAL_RESEARCH_CHECKPOINT_DB", raising=False)
    assert adapter._checkpoint_db_path() is None
    async with adapter.checkpointer_ctx("s1", None, True) as a:
        async with adapter.checkpointer_ctx("s1", None, True) as b:
            assert isinstance(a, MemorySaver) and a is b  # cached, reused


async def test_adapter_emits_single_zero_usage_before_final():
    """In fake mode run_turn emits exactly one all-zero usage event, and it is
    the event immediately before the single final."""
    events = [ev async for ev in run_turn("hi", "usage-sess", fake=True)]
    usage = [ev for ev in events if ev.kind == "usage"]
    assert len(usage) == 1
    assert (usage[0].tokens_in, usage[0].tokens_out, usage[0].tokens_cache) == (0, 0, 0)
    assert events[-1].kind == "final"
    assert events[-2].kind == "usage"


async def test_run_turn_accepts_model_kwarg():
    """run_turn accepts a model= override and still produces the fake answer."""
    events = [
        ev async for ev in run_turn("hi", "model-kwarg", fake=True, model="gpt-4o")
    ]
    finals = [ev for ev in events if ev.kind == "final"]
    errors = [ev for ev in events if ev.kind == "error"]
    assert len(finals) == 1
    assert "FAKE-OK" in finals[0].text
    assert not errors


async def test_compact_session_shrinks_and_reseeds():
    """compact_session summarizes older turns and rewrites the thread: the message
    count drops, a summary seed leads the history, and the kept tail starts on a
    user message (never a dangling tool result)."""
    from langchain_core.messages import HumanMessage

    from financial_research_assistant.adapter import _fake_graph_for, compact_session

    sid = "compact-e2e"
    for msg in ("hi", "what is AAPL", "and MSFT", "compare them"):
        async for _ in run_turn(msg, sid, fake=True):
            pass
    graph = _fake_graph_for(sid, True)
    cfg = {"configurable": {"thread_id": sid}}
    before = len((await graph.aget_state(cfg)).values["messages"])

    res = await compact_session(sid, fake=True)
    assert res["removed"] > 0
    after_msgs = (await graph.aget_state(cfg)).values["messages"]
    assert len(after_msgs) < before
    assert res["summary"] in after_msgs[0].content   # summary seeds the thread
    assert isinstance(after_msgs[1], HumanMessage)    # tail begins on a user msg


async def test_compact_session_noop_on_short_history():
    """With too little history to be worth compacting, compact_session is a no-op
    (removed == 0) and leaves the thread untouched."""
    from financial_research_assistant.adapter import compact_session

    sid = "compact-noop"
    async for _ in run_turn("hi", sid, fake=True):
        pass
    res = await compact_session(sid, fake=True)
    assert res["removed"] == 0


async def test_run_turn_auto_compacts_over_threshold(monkeypatch):
    """AGENT_AUTO_COMPACT makes run_turn compact *before* a turn once the prior
    turn's input reached the configured fraction of the context window — so any
    non-TUI caller (headless, eval, service) gets bounded context. Off by default."""
    from financial_research_assistant import adapter
    from financial_research_assistant.pricing import context_cap

    sid = "auto-compact"
    for m in ("hi", "one", "two", "three"):
        async for _ in run_turn(m, sid, fake=True):
            pass

    # Default (env unset): no auto-compaction even with a huge recorded prior input.
    monkeypatch.delenv("AGENT_AUTO_COMPACT", raising=False)
    adapter._last_input[sid] = 10_000_000
    off = [ev async for ev in run_turn("x", sid, fake=True) if ev.kind == "status"]
    assert not any("auto-compacted" in ev.text for ev in off)

    # Enabled and over threshold: compacts before the turn and says so.
    monkeypatch.setenv("AGENT_AUTO_COMPACT", "0.85")
    adapter._last_input[sid] = int(context_cap("gpt-4.1-mini") * 0.9)
    on = [ev async for ev in run_turn("y", sid, fake=True) if ev.kind == "status"]
    compacted = [ev for ev in on if "auto-compacted" in ev.text]
    assert compacted
    # The auto-compaction status snapshots context as 0 so the footer's ctx% gauge
    # drops the instant the context shrinks — not only on the next turn.
    assert compacted[0].context_tokens == 0


def test_quick_llm_tier_resolution(monkeypatch):
    """quick_llm uses QUICK_MODEL (+ QUICK_* endpoint overrides) when set — the
    cheap tier for summarization — and falls back to the passed/primary model when
    unset, so behavior is unchanged by default."""
    from financial_research_assistant import graph

    captured = {}

    def fake_make_llm(model=None, **kw):
        captured.clear()
        captured["model"] = model
        captured.update(kw)
        return object()

    monkeypatch.setattr(graph, "_make_llm", fake_make_llm)

    # Unset: falls back to the passed model, no endpoint overrides.
    for v in ("QUICK_MODEL", "QUICK_API_BASE", "QUICK_API_KEY", "QUICK_MODEL_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    graph.quick_llm("primary-model")
    assert captured == {"model": "primary-model", "provider": None,
                        "base_url": None, "api_key": None}

    # Set: QUICK_MODEL wins over the passed model, overrides are forwarded.
    monkeypatch.setenv("QUICK_MODEL", "cheap-mini")
    monkeypatch.setenv("QUICK_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("QUICK_MODEL_PROVIDER", "openai")
    graph.quick_llm("primary-model")
    assert captured["model"] == "cheap-mini"                 # quick tier wins
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["provider"] == "openai"


async def test_summarize_messages_uses_quick_tier(monkeypatch):
    """compaction summarization routes through the quick tier (quick_llm), so
    QUICK_MODEL offloads the summary to the cheap model."""
    from financial_research_assistant import graph
    from langchain_core.messages import AIMessage, HumanMessage

    seen = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            seen["called"] = True
            return AIMessage(content="recap")

    monkeypatch.setattr(graph, "quick_llm", lambda model=None: seen.update(arg=model) or _FakeLLM())
    out = await graph.summarize_messages(
        [HumanMessage(content="hi"), AIMessage(content="hello")], model="primary"
    )
    assert out == "recap"
    assert seen["called"] and seen["arg"] == "primary"  # went through quick_llm, not _make_llm


async def test_final_usage_event_carries_context_snapshot():
    """The final (reconciling) usage event carries a context_tokens snapshot — the
    turn's total input — so a UI can size ctx% to the CURRENT context rather than
    cumulative session spend. Ordinary live-delta usage events carry no snapshot."""
    events = [ev async for ev in run_turn("hi", "ctx-snap", fake=True)]
    usage = [ev for ev in events if ev.kind == "usage"]
    assert usage, "a turn always emits a final usage event"
    # The final usage event (last one) is the reconciling one and carries a snapshot.
    assert usage[-1].context_tokens >= 0


async def test_run_turn_think_false_emits_no_reasoning():
    """think=False suppresses reasoning events end-to-end (the fake graph omits
    reasoning_content and the adapter drops any that slip through)."""
    on = [ev async for ev in run_turn("hi", "think-on", fake=True, think=True)]
    off = [ev async for ev in run_turn("hi", "think-off", fake=True, think=False)]
    assert any(ev.kind == "reasoning" for ev in on)
    assert not any(ev.kind == "reasoning" for ev in off)
    assert any(ev.kind == "final" for ev in off)  # still completes


def test_empty_model_env_falls_back_to_default(monkeypatch):
    """An empty OPENAI_MODEL (exactly what .env.example ships) must still
    resolve to the default, not "". Regression: .get(key, default) only fills
    the default when the key is absent, but dotenv loads a blank line as ""."""
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    captured: dict = {}

    class _FakeChat:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChat)
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: kw)
    from financial_research_assistant.graph import _build_real_graph

    _build_real_graph()
    assert captured["model"] == "gpt-4.1-mini"


def test_usage_delta_parses_flat_message_metadata():
    """_usage_delta reads a single message's flat usage_metadata (one call),
    including cache-read details, and tolerates None."""
    from financial_research_assistant.adapter import _usage_delta

    assert _usage_delta(None) == (0, 0, 0)
    assert _usage_delta({"input_tokens": 10, "output_tokens": 3}) == (10, 3, 0)
    assert _usage_delta(
        {"input_tokens": 10, "output_tokens": 3, "input_token_details": {"cache_read": 4}}
    ) == (10, 3, 4)


async def test_think_tool_calls_surface_as_reasoning_not_tool_panels():
    from financial_research_assistant.adapter import _stream_events

    graph = _scripted_think_graph()
    inputs = {"messages": [{"role": "user", "content": "review a.py"}]}
    events = [ev async for ev in _stream_events(graph, inputs, {})]

    reasoning = [ev for ev in events if ev.kind == "reasoning"]
    starts = [ev for ev in events if ev.kind == "tool_start"]
    ends = [ev for ev in events if ev.kind == "tool_end"]
    # the think call became a reasoning event carrying its thought text
    assert any("read a.py first" in ev.text for ev in reasoning)
    # and produced NO 🛠 panel: only the real read_source tool paired start/end
    assert [ev.tool for ev in starts] == ["read_source"]
    assert [ev.tool for ev in ends] == ["read_source"]
    assert all(ev.tool != "think" for ev in starts + ends)


def test_think_text_extracts_thought():
    from financial_research_assistant.adapter import _think_text

    assert _think_text('{"thought": "plan the review"}') == "plan the review"
    assert _think_text("not json") == "not json"
    assert _think_text("") == "(thinking)"


def test_context_window_env_override(monkeypatch):
    """OPENAI_CONTEXT_WINDOW overrides the built-in per-model window; a bad or
    unset value falls back to the table / 128k default."""
    from financial_research_assistant.pricing import context_cap, context_pct

    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)
    assert context_cap("unknown-local-model") == 128_000  # default fallback

    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "32768")
    assert context_cap("unknown-local-model") == 32_768   # override wins
    assert context_cap("gpt-4o") == 32_768                # even over the table
    assert context_pct("unknown-local-model", 16_384) == 50

    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "not-a-number")
    assert context_cap("gpt-4o") == 128_000               # bad value ignored


async def test_think_reasoning_event_carries_duration():
    """The think tool's reasoning event is stamped with the step's duration
    (measured like a tool's), so the 💭 panel can show (Xs) like tool panels."""
    from financial_research_assistant.adapter import _stream_events

    graph = _scripted_think_graph()
    inputs = {"messages": [{"role": "user", "content": "go"}]}
    events = [ev async for ev in _stream_events(graph, inputs, {})]
    reasoning = [ev for ev in events if ev.kind == "reasoning"]
    assert reasoning, "think call should surface a reasoning event"
    assert "read a.py first" in reasoning[0].text
    assert reasoning[0].duration >= 0.0  # a real, tracked duration


async def test_real_graph_session_composes_local_and_mcp_tools(monkeypatch):
    """real_graph_session opens the IBKR session and hands the combined local +
    market-data tool list (plus the persistent checkpointer) to create_agent."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    class _FakeChat:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChat)
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: kw)
    monkeypatch.setattr(
        "financial_research_assistant.graph.ibkr_tools_session",
        _fake_ibkr_tools_session("get_price_snapshot"),
    )
    from financial_research_assistant.graph import real_graph_session

    sentinel = object()  # stands in for the persistent checkpointer
    async with real_graph_session(think=False, checkpointer=sentinel) as result:
        names = [getattr(t, "name", getattr(t, "__name__", "")) for t in result["tools"]]
        assert "get_price_snapshot" in names   # MCP market-data tool wired in
        assert "current_date" in names          # local calculator wired in
        assert "think" not in names             # think=False omits the scratchpad
        assert result["checkpointer"] is sentinel  # persistent checkpointer used
        # No authenticate tool present -> no auth self-heal guidance in the prompt.
        assert "authenticate" not in result["system_prompt"]


async def test_auth_guidance_added_only_when_authenticate_tool_present(monkeypatch):
    """When `authenticate` is among the loaded tools (IBKR_ALLOW_AUTHENTICATE on),
    the system prompt gains the self-heal guidance so the model re-auths and
    retries instead of relaying the server's 'authentication required' error."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    class _FakeChat:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChat)
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: kw)
    monkeypatch.setattr(
        "financial_research_assistant.graph.ibkr_tools_session",
        _fake_ibkr_tools_session("get_positions", "authenticate"),
    )
    from financial_research_assistant.graph import real_graph_session

    async with real_graph_session(think=False) as result:
        prompt = result["system_prompt"]
        assert "authenticate" in prompt and "retry the original call" in prompt


async def test_pa_guidance_added_only_when_pa_tool_present(monkeypatch):
    """The Portfolio Analyst guidance appears only when
    `get_pa_performance_all_periods` is loaded (a live session); otherwise the
    prompt stays free of it and relies on the statement-based charts."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    class _FakeChat:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChat)
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: kw)

    monkeypatch.setattr(
        "financial_research_assistant.graph.ibkr_tools_session",
        _fake_ibkr_tools_session("get_pa_performance_all_periods"),
    )
    from financial_research_assistant.graph import real_graph_session

    async with real_graph_session(think=False) as result:
        assert "get_pa_performance_all_periods" in result["system_prompt"]
        assert "Portfolio Analyst" in result["system_prompt"]

    # Without the PA tool, the guidance is absent.
    monkeypatch.setattr(
        "financial_research_assistant.graph.ibkr_tools_session",
        _fake_ibkr_tools_session("get_price_snapshot"),
    )
    async with real_graph_session(think=False) as result:
        assert "Portfolio Analyst" not in result["system_prompt"]
