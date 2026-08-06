"""Adapter, checkpointer, compaction, and real graph-session tests."""

from financial_research_assistant.adapter import run_turn

from .helpers.fakes import fake_ibkr_tools_session as _fake_ibkr_tools_session
from .helpers.graphs import (
    alerting_tool_graph as _alerting_tool_graph,
    nested_model_graph as _nested_model_graph,
    scripted_think_graph as _scripted_think_graph,
    scripted_tool_graph as _scripted_tool_graph,
)


async def test_fired_alerts_surface_as_events_after_their_tool_call():
    """A rule that fires inside a tool reaches the UI as its own `alert` event
    right after that tool's tool_end, so an interface can notify on it instead of
    hoping the user spots it in the digest prose."""
    from financial_research_assistant import alerts
    from financial_research_assistant.adapter import _stream_events

    alerts.drain_triggered()  # start from a clean buffer
    graph = _alerting_tool_graph(["AAPL down 6.2% — now 180.10", "TSLA at 195.00"])
    inputs = {"messages": [{"role": "user", "content": "any alerts?"}]}
    events = [ev async for ev in _stream_events(graph, inputs, {})]

    kinds = [ev.kind for ev in events]
    fired = [ev.text for ev in events if ev.kind == "alert"]
    assert fired == ["AAPL down 6.2% — now 180.10", "TSLA at 195.00"]
    assert kinds.index("tool_end") < kinds.index("alert")
    # Drained, not merely copied — a second turn must not replay stale alerts.
    assert alerts.drain_triggered() == []


async def test_models_running_inside_tools_do_not_stream_into_the_answer():
    """A model invoked inside a tool (what dispatch_subagents does, three at a
    time) shares the parent run's callbacks, so LangGraph streams its tokens into
    the same messages stream. Unfiltered they interleave word-by-word and the
    reply renders as a mash; only the agent's own model node is the answer."""
    from financial_research_assistant.adapter import _stream_events

    graph = _nested_model_graph()
    inputs = {"messages": [{"role": "user", "content": "go"}]}
    events = [ev async for ev in _stream_events(graph, inputs, {})]

    streamed = "".join(ev.text for ev in events if ev.kind == "token")
    assert "NESTED" not in streamed
    assert "ANSWER" in streamed
    # The tool itself still pairs normally — filtering answer text must not
    # suppress the 🛠 panel for the call the nested models ran under.
    assert [ev.tool for ev in events if ev.kind == "tool_start"] == ["dispatch"]
    assert [ev.tool for ev in events if ev.kind == "tool_end"] == ["dispatch"]


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
    from financial_research_assistant import llm

    captured = {}

    def fake_make_llm(model=None, **kw):
        captured.clear()
        captured["model"] = model
        captured.update(kw)
        return object()

    monkeypatch.setattr(llm, "_make_llm", fake_make_llm)

    # Unset: falls back to the passed model, no endpoint overrides.
    for v in ("QUICK_MODEL", "QUICK_API_BASE", "QUICK_API_KEY", "QUICK_MODEL_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    llm.quick_llm("primary-model")
    assert captured == {"model": "primary-model", "provider": None,
                        "base_url": None, "api_key": None, "scope": "quick"}

    # Set: QUICK_MODEL wins over the passed model, overrides are forwarded.
    monkeypatch.setenv("QUICK_MODEL", "cheap-mini")
    monkeypatch.setenv("QUICK_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("QUICK_MODEL_PROVIDER", "openai")
    llm.quick_llm("primary-model")
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

    # Patch the name `graph` resolves, not `llm`'s: graph binds quick_llm at
    # import, so patching the source module would not reach this call.
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
    including both cache details, and tolerates None."""
    from financial_research_assistant.adapter import _usage_delta

    assert _usage_delta(None) == (0, 0, 0, 0)
    assert _usage_delta({"input_tokens": 10, "output_tokens": 3}) == (10, 3, 0, 0)
    assert _usage_delta(
        {"input_tokens": 10, "output_tokens": 3, "input_token_details": {"cache_read": 4}}
    ) == (10, 3, 4, 0)
    # cache_creation is what a cache WRITE costs; it bills at a different rate
    # from both fresh input and a cache read, so it is counted separately.
    assert _usage_delta({
        "input_tokens": 10, "output_tokens": 3,
        "input_token_details": {"cache_read": 4, "cache_creation": 5},
    }) == (10, 3, 4, 5)


def test_sum_usage_aggregates_both_cache_counters():
    from financial_research_assistant.adapter import _sum_usage

    assert _sum_usage(None) == (0, 0, 0, 0)
    assert _sum_usage({
        "modelA": {"input_tokens": 10, "output_tokens": 2,
                   "input_token_details": {"cache_read": 3, "cache_creation": 4}},
        "modelB": {"input_tokens": 20, "output_tokens": 5,
                   "input_token_details": {"cache_creation": 6}},
    }) == (30, 7, 3, 10)


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


def test_pricing_data_file_override(monkeypatch, tmp_path):
    """A pricing.json data file adds/updates model prices and context windows
    without editing code; the per-model env vars still win over it."""
    import json as _json

    from financial_research_assistant.pricing import context_cap, rates

    for var in ("OPENAI_CONTEXT_WINDOW", "OPENAI_INPUT_COST_PER_1M", "OPENAI_OUTPUT_COST_PER_1M"):
        monkeypatch.delenv(var, raising=False)
    f = tmp_path / "pricing.json"
    f.write_text(_json.dumps({
        "pricing": {"my-local-model": [1.5, 4.5], "gpt-4o": [9.9, 9.9]},  # add + override
        "context": {"my-local-model": 262144},
    }))
    monkeypatch.setenv("FINANCIAL_RESEARCH_PRICING_FILE", str(f))

    assert rates("my-local-model-v2") == (1.5, 4.5)     # new model added (prefix match)
    assert rates("gpt-4o") == (9.9, 9.9)                # file overrides the built-in
    assert context_cap("my-local-model-v2") == 262144  # new context window from the file
    assert rates("totally-unknown-xyz") is None         # still None for the truly unknown

    # the per-model env override still wins over the data file
    monkeypatch.setenv("OPENAI_INPUT_COST_PER_1M", "2.0")
    monkeypatch.setenv("OPENAI_OUTPUT_COST_PER_1M", "6.0")
    assert rates("gpt-4o") == (2.0, 6.0)


def test_malformed_pricing_data_file_degrades(monkeypatch, tmp_path):
    """pricing.json is hand-edited, so a wrong shape must fall back to the
    built-in tables rather than raise. A section holding a string used to reach
    ``.items()`` and throw AttributeError out of a helper every cost lookup calls
    — including the status bar's, mid-turn."""
    import json as _json

    from financial_research_assistant.pricing import _load_overrides, context_cap, rates

    for var in ("OPENAI_CONTEXT_WINDOW", "OPENAI_INPUT_COST_PER_1M", "OPENAI_OUTPUT_COST_PER_1M"):
        monkeypatch.delenv(var, raising=False)
    f = tmp_path / "pricing.json"
    monkeypatch.setenv("FINANCIAL_RESEARCH_PRICING_FILE", str(f))

    for payload in (
        {"pricing": "oops", "context": [1, 2]},   # sections of the wrong type
        {"pricing": None, "context": None},        # explicit nulls
        {},                                        # sections absent entirely
        {"pricing": {"m": "not-a-pair"}, "context": {"m": "not-a-number"}},  # bad values
    ):
        f.write_text(_json.dumps(payload))
        assert _load_overrides() == ({}, {}), payload
        assert rates("gpt-4o") == (2.50, 10.00)    # built-in table, untouched
        assert context_cap("gpt-4o") == 128_000


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


# --- unbacked scheduling claims -------------------------------------------------


def test_a_scheduling_claim_with_no_tool_call_is_corrected():
    """Live testing found claude-haiku-4-5 calling schedule_task on one of three
    identical requests — and saying "✓ Scheduled" on the other two. A user who
    believes work is queued when nothing is loses the thing they asked for."""
    from financial_research_assistant.adapter import verify_schedule_claim

    for claim in (
        "✓ **Scheduled.** NVDA analysis will run tomorrow at 09:00.",
        "Got it. I've scheduled a task to analyse NVDA's earnings tomorrow.",
        "I have now queued the analysis for you.",
        "Done — the task is scheduled and will be delivered to Telegram.",
    ):
        out = verify_schedule_claim(claim, set())
        assert "nothing was actually scheduled" in out.lower(), claim
        assert "--schedule" in out, "the correction must say how to fix it"


def test_a_real_schedule_is_left_alone():
    from financial_research_assistant.adapter import verify_schedule_claim

    claim = "✓ Scheduled [s1] for tomorrow 09:00 — delivered to Telegram."
    assert verify_schedule_claim(claim, {"schedule_task"}) == claim


def test_listing_existing_tasks_is_not_mistaken_for_a_claim():
    """`list_scheduled_tasks` answers are full of the word "scheduled"; warning on
    them would train the user to ignore the warning."""
    from financial_research_assistant.adapter import verify_schedule_claim

    listing = "You have 2 scheduled tasks:\n  [s1] tomorrow 09:00 — NVDA earnings"
    assert verify_schedule_claim(listing, {"list_scheduled_tasks"}) == listing
    # even with no tool recorded, a third-person listing is not a creation claim
    assert verify_schedule_claim(listing, set()) == listing


def test_ordinary_answers_are_untouched():
    from financial_research_assistant.adapter import verify_schedule_claim

    for text in (
        "AAPL closed at $214.30, up 1.2% on the day.",
        "The earnings call is scheduled for August 13 — that is the company's date.",
        "",
    ):
        assert verify_schedule_claim(text, set()) == text


def test_the_exact_phrasings_seen_in_live_runs_are_caught():
    """Verbatim openings from three live claude-haiku-4-5 runs that claimed a
    schedule without calling the tool. The middle one defeated the first version of
    the pattern, which required a determiner after the verb."""
    from financial_research_assistant.adapter import verify_schedule_claim

    for opening in (
        "✓ **Scheduled.** NVDA typically reports pre-market (before 9:30am ET). "
        "I've queued analysis to run at 10:30am tomorrow.",
        "Done. I've scheduled an earnings analysis for **tomorrow at 4:30 PM "
        "(after market close)**, when NVDA's results will be available.",
        "Got it. I've scheduled a task to analyze NVDA's earnings tomorrow at 9am.",
    ):
        assert "nothing was actually scheduled" in verify_schedule_claim(opening, set()).lower()


# --- repairing an unbacked claim ------------------------------------------------


def _fake_quick(monkeypatch, content):
    """Stand in for the cheap tier used by the repair pass."""
    class _Resp:
        def __init__(self, c): self.content = c

    class _LLM:
        async def ainvoke(self, _prompt): return _Resp(content)

    from financial_research_assistant import llm
    monkeypatch.setattr(llm, "quick_llm", lambda *a, **k: _LLM())


async def test_an_unbacked_claim_is_made_true(monkeypatch):
    """The turn already told the user it scheduled something. Creating the task is
    the honest resolution — the alternative is a retraction for work they asked for
    and were told they had."""
    from financial_research_assistant import tasks
    from financial_research_assistant.adapter import settle_schedule_claim

    _fake_quick(monkeypatch, '{"prompt": "Analyse NVDA Q3 vs consensus", '
                             '"when": "tomorrow 9am", "repeat": "once"}')
    out = await settle_schedule_claim(
        "Monitor NVDA's earnings tomorrow.",
        "Done. I've scheduled an earnings analysis for tomorrow.",
        set(),
    )
    assert "Task created" in out and "[s1]" in out
    stored = tasks.pending_tasks()
    assert len(stored) == 1 and stored[0]["prompt"] == "Analyse NVDA Q3 vs consensus"


async def test_a_failed_repair_retracts_rather_than_inventing(monkeypatch):
    from financial_research_assistant import tasks
    from financial_research_assistant.adapter import settle_schedule_claim

    _fake_quick(monkeypatch, "sorry, I can't do that")  # unparseable
    out = await settle_schedule_claim(
        "Monitor NVDA.", "I've scheduled it for tomorrow.", set()
    )
    assert "nothing was actually scheduled" in out.lower()
    assert tasks.load_tasks() == []


async def test_a_real_tool_call_skips_the_repair_entirely(monkeypatch):
    from financial_research_assistant import tasks
    from financial_research_assistant.adapter import settle_schedule_claim

    _fake_quick(monkeypatch, '{"prompt": "x", "when": "+1h"}')
    answer = "✓ Scheduled [s1] for tomorrow 09:00."
    assert await settle_schedule_claim("q", answer, {"schedule_task"}) == answer
    assert tasks.load_tasks() == [], "the repair must not double-create"


async def test_fake_mode_never_calls_a_model(monkeypatch):
    """Offline tests and --fake runs must stay model-free."""
    from financial_research_assistant.adapter import settle_schedule_claim

    def explode(*_a, **_k):
        raise AssertionError("the repair pass called a model in fake mode")

    from financial_research_assistant import llm
    monkeypatch.setattr(llm, "quick_llm", explode)
    answer = "I've scheduled it for tomorrow."
    assert await settle_schedule_claim("q", answer, set(), fake=True) == answer


async def test_an_ordinary_answer_never_reaches_the_repair(monkeypatch):
    from financial_research_assistant.adapter import settle_schedule_claim

    def explode(*_a, **_k):
        raise AssertionError("the repair pass ran on a turn with no claim")

    from financial_research_assistant import llm
    monkeypatch.setattr(llm, "quick_llm", explode)
    answer = "AAPL closed at $214.30, up 1.2%."
    assert await settle_schedule_claim("q", answer, set()) == answer


# --- interrupted tool calls ------------------------------------------------------


def _ai_with_calls(*ids):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="working on it",
        tool_calls=[
            {"id": i, "name": f"tool_{n}", "args": {}} for n, i in enumerate(ids)
        ],
    )


def test_an_unanswered_tool_call_gets_a_synthetic_result():
    """Every provider rejects a tool call with no result, so a turn killed between
    the model node and the tool node would brick the thread for good."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from financial_research_assistant.adapter import repair_tool_call_pairs

    history = [HumanMessage(content="analyse my holdings"), _ai_with_calls("a", "b", "c")]
    fixed, made, dropped = repair_tool_call_pairs(history)
    assert (made, dropped) == (3, 0)
    results = [m for m in fixed if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == ["a", "b", "c"]
    assert all(m.status == "error" for m in results)
    # The tool may well have worked — only its result was lost — so the message
    # reports on the run, and must not tell the model the tool itself failed.
    body = results[0].content.lower()
    assert "no result was recorded" in body
    assert "tool failed" not in body


def test_results_that_did_arrive_are_left_alone():
    """A partially-answered batch keeps its real results and gains only the gaps,
    in the order the calls were made."""
    from langchain_core.messages import ToolMessage

    from financial_research_assistant.adapter import repair_tool_call_pairs

    history = [
        _ai_with_calls("a", "b", "c"),
        ToolMessage(content="real answer", tool_call_id="b", name="tool_1"),
    ]
    fixed, made, dropped = repair_tool_call_pairs(history)
    assert (made, dropped) == (2, 0)
    results = [m for m in fixed if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == ["b", "a", "c"]
    assert results[0].content == "real answer"


def test_a_result_whose_call_is_gone_is_dropped():
    """The mirror-image break — a tool_result with no tool_use — which compaction
    or a hand-edited thread can produce, and which the API rejects just as hard."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from financial_research_assistant.adapter import repair_tool_call_pairs

    history = [
        HumanMessage(content="hi"),
        ToolMessage(content="orphan", tool_call_id="ghost", name="t"),
    ]
    fixed, made, dropped = repair_tool_call_pairs(history)
    assert (made, dropped) == (0, 1)
    assert not [m for m in fixed if isinstance(m, ToolMessage)]


def test_a_healthy_history_is_returned_unchanged():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from financial_research_assistant.adapter import repair_tool_call_pairs

    history = [
        HumanMessage(content="hi"),
        _ai_with_calls("a"),
        ToolMessage(content="ok", tool_call_id="a", name="tool_0"),
        AIMessage(content="done"),
    ]
    fixed, made, dropped = repair_tool_call_pairs(history)
    assert (made, dropped) == (0, 0)
    assert fixed == history


def test_an_invalid_tool_call_still_needs_a_result():
    """A call whose args failed to parse is still serialized as a tool_use block,
    so it still has to be paired."""
    from langchain_core.messages import AIMessage

    from financial_research_assistant.adapter import repair_tool_call_pairs

    broken = AIMessage(
        content="",
        invalid_tool_calls=[
            {"id": "x", "name": "screen_stocks", "args": "{not json", "error": "bad args"}
        ],
    )
    _fixed, made, dropped = repair_tool_call_pairs([broken])
    assert (made, dropped) == (1, 0)


async def test_a_broken_thread_heals_before_the_next_turn():
    """End to end: seed a thread with an interrupted tool call, then take a turn —
    it must answer rather than error, and say what it repaired."""
    from financial_research_assistant.adapter import (
        _fake_graph_for,
        repair_tool_call_pairs,
    )

    sid = "heal-e2e"
    async for _ in run_turn("hi", sid, fake=True):
        pass
    graph = _fake_graph_for(sid, True)
    cfg = {"configurable": {"thread_id": sid}}
    await graph.aupdate_state(cfg, {"messages": [_ai_with_calls("a", "b")]},
                              as_node="respond")

    events = [ev async for ev in run_turn("still there?", sid, fake=True)]
    assert not [ev for ev in events if ev.kind == "error"]
    assert len([ev for ev in events if ev.kind == "final"]) == 1
    assert any("unfinished tool call" in ev.text for ev in events if ev.kind == "status")

    messages = (await graph.aget_state(cfg)).values["messages"]
    _fixed, made, dropped = repair_tool_call_pairs(messages)
    assert (made, dropped) == (0, 0), "the thread must be left healthy"


async def test_a_healthy_thread_is_not_rewritten():
    """The repair runs on every turn, so a clean thread must pay a read and
    nothing else — no status line, no checkpoint churn."""
    sid = "heal-noop"
    async for _ in run_turn("hi", sid, fake=True):
        pass
    events = [ev async for ev in run_turn("again", sid, fake=True)]
    assert not any("unfinished tool call" in ev.text
                   for ev in events if ev.kind == "status")
