"""Offline smoke tests — no network, no API keys required."""

from typing import Any, cast

from financial_research_assistant.adapter import run_turn
from financial_research_assistant.events import AgentEvent
from financial_research_assistant.graph import build_graph
from financial_research_assistant.tui import AgentApp, AgentMessageWidget, UserMessageWidget
from textual.widgets import Input, Static


async def test_adapter_round_trip():
    """run_turn in fake mode yields exactly one final (FAKE-OK), no errors."""
    events = [ev async for ev in run_turn("hi", "t1", fake=True)]
    finals = [ev for ev in events if ev.kind == "final"]
    errors = [ev for ev in events if ev.kind == "error"]
    assert len(finals) == 1
    assert "FAKE-OK" in finals[0].text
    assert not errors


async def test_tui_round_trip(monkeypatch, tmp_path):
    """Typing a message into the TUI mounts the user's chat bubble and the
    agent's reply bubble in #log, and the config bar reports FAKE mode. (The
    fake graph is a single node with no tools, so no ToolPanel appears — the
    bubble widgets are the contract here; the tool pipeline is covered above.)"""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="tui-test")
    async with app.run_test() as pilot:
        assert "FAKE" in str(app.query_one("#config", Static).render())
        app.query_one(Input).value = "hello tui"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query(UserMessageWidget).last()._query == "hello tui"
        assert "FAKE-OK" in app.query(AgentMessageWidget).last().text


async def test_checkpointer_accumulates_history():
    """Framework wiring: MemorySaver keeps per-thread state across invokes."""
    graph = build_graph(fake=True)
    config = {"configurable": {"thread_id": "wiring"}}
    await cast(Any, graph).ainvoke(
        {"messages": [{"role": "user", "content": "first"}]}, config
    )
    await cast(Any, graph).ainvoke(
        {"messages": [{"role": "user", "content": "second"}]}, config
    )
    messages = cast(Any, graph).get_state(config).values["messages"]
    assert len(messages) > 2 # second invoke saw accumulated history
    assert len(messages) == 4 # 2 user + 2 ai messages checkpointed


async def test_durable_checkpointer_survives_restart_and_reset(monkeypatch, tmp_path):
    """Opt-in durable mode persists thread state to SQLite across separate
    checkpointer_ctx opens (i.e. process restarts), and reset_session erases just
    that thread's rows."""
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langchain_core.messages import AIMessage

    from financial_research_assistant import adapter

    db = tmp_path / "sub" / "ckpt.db" # parent dir created lazily
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHECKPOINT_DB", str(db))
    assert adapter._checkpoint_db_path() == db

    def _graph(saver):
        def respond(state):
            return {"messages": [AIMessage(content="ok")]}
        g = StateGraph(MessagesState)
        g.add_node("respond", respond)
        g.add_edge(START, "respond")
        g.add_edge("respond", END)
        return g.compile(checkpointer=saver)

    cfg = {"configurable": {"thread_id": "work"}}
    # "Process 1": two turns on session "work".
    async with adapter.checkpointer_ctx("work", None, True) as saver:
        graph = _graph(saver)
        await graph.ainvoke({"messages": [{"role": "user", "content": "a"}]}, cfg)
        await graph.ainvoke({"messages": [{"role": "user", "content": "b"}]}, cfg)
    # "Process 2": a fresh connection to the same file restores the history.
    async with adapter.checkpointer_ctx("work", None, True) as saver:
        state = await _graph(saver).aget_state(cfg)
        assert len(state.values["messages"]) == 4 # 2 user + 2 ai, survived restart

    # reset_session erases the durable thread.
    adapter.reset_session("work")
    async with adapter.checkpointer_ctx("work", None, True) as saver:
        state = await _graph(saver).aget_state(cfg)
        assert not state.values.get("messages")


async def test_once_mode_error_propagates_nonzero_return_code(monkeypatch, tmp_path):
    """A --once run whose turn errors must exit non-zero. Regression: the exit
    code was passed positionally to App.exit(result, return_code=0), so it set
    result and left return_code=0, and main.py's `app.return_code` stayed 0."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def boom(*args, **kwargs):
        yield AgentEvent("error", "kaboom")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", boom)
    app = AgentApp(fake=True, once="trigger", session_id="once-err")
    async with app.run_test():
        await app.workers.wait_for_complete()
        assert app.return_code == 1


# --- C: data/workflow improvements -----------------------------------------


def test_eval_llm_judge_uses_configured_model_no_bare_key(monkeypatch):
    """Regression: the eval llm_judge scores via llm._make_llm (the app's
    configured model) rather than a bare OpenAI() client, so a local
    OPENAI_API_BASE / MODEL_PROVIDER setup works without OPENAI_API_KEY — and it
    parses the score out of a prose reply."""
    import importlib.util
    from pathlib import Path

    import financial_research_assistant.llm as g

    # Load eval/evaluate.py (not a package module) by path.
    eval_path = Path(__file__).resolve().parent.parent / "eval" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", eval_path)
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)

    # No OPENAI_API_KEY in the environment; a stubbed model returns a prose score.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    captured = {}

    class _Resp:
        content = "0.9 — the answer meets all the criteria"

    class _LLM:
        def invoke(self, messages):
            captured["called"] = True
            return _Resp()

    monkeypatch.setattr(g, "_make_llm", lambda model=None: _LLM())
    score = evaluate._llm_judge("q", "agent output", [{"answer": "cites a price"}], fake=False)
    assert captured.get("called") is True
    assert score == 0.9 # parsed from prose, clamped to [0, 1]

    # A model/endpoint error is caught and scored 0.0, never raised into the run.
    monkeypatch.setattr(g, "_make_llm", lambda model=None: (_ for _ in ()).throw(RuntimeError("boom")))
    assert evaluate._llm_judge("q", "out", [{"answer": "x"}], fake=False) == 0.0


# -- eval-driven self-improvement (self-learning layer #4) ------------------


def test_prompt_addendum_env_and_file(monkeypatch, tmp_path):
    """prompt_addendum resolves inline env (empty string wins as 'none') over the
    file, and returns '' when neither is set."""
    from financial_research_assistant import graph

    monkeypatch.setenv("FINANCIAL_RESEARCH_PROMPT_ADDENDUM_FILE", str(tmp_path / "add.txt"))
    monkeypatch.delenv("FINANCIAL_RESEARCH_PROMPT_ADDENDUM", raising=False)
    assert graph.prompt_addendum() == "" # nothing set
    (tmp_path / "add.txt").write_text("Prefer tool X for Y.")
    assert graph.prompt_addendum() == "Prefer tool X for Y." # file used
    monkeypatch.setenv("FINANCIAL_RESEARCH_PROMPT_ADDENDUM", "")
    assert graph.prompt_addendum() == "" # inline empty overrides the file
    monkeypatch.setenv("FINANCIAL_RESEARCH_PROMPT_ADDENDUM", "Inline wins.")
    assert graph.prompt_addendum() == "Inline wins."


def test_build_real_graph_appends_prompt_addendum(monkeypatch):
    """A configured addendum is appended to the system prompt the agent is built
    with; absent one leaves the prompt unchanged."""
    captured: dict = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: object())
    monkeypatch.setattr(
        "langchain.agents.create_agent", lambda **kw: captured.update(kw) or kw
    )
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    from financial_research_assistant.graph import _build_real_graph

    monkeypatch.setenv("FINANCIAL_RESEARCH_PROMPT_ADDENDUM", "LEARNED: use foo_tool for bar.")
    _build_real_graph(think=False)
    assert captured["system_prompt"].endswith("LEARNED: use foo_tool for bar.")
    monkeypatch.setenv("FINANCIAL_RESEARCH_PROMPT_ADDENDUM", "")
    _build_real_graph(think=False)
    assert "LEARNED:" not in captured["system_prompt"]


def test_improve_diagnose_routing_and_content():
    """diagnose flags a trajectory item that missed its expected tool, passes a
    hit, and reports a content miss."""
    from financial_research_assistant import improve

    traj = {"query": "my realized gains?", "eval_type": "trajectory",
    "criteria": [{"tool": "realized_gains"}]}
    # Missed the expected tool → routing diagnosis.
    d = improve.diagnose(traj, 0.0, ["allocation"], "", floor=1.0)
    assert d["kind"] == "routing" and d["expected"] == ["realized_gains"]
    # Called it → no diagnosis.
    assert improve.diagnose(traj, 1.0, ["realized_gains"], "", floor=1.0) is None
    # Content miss.
    cont = {"query": "explain P/E", "eval_type": "llm_judge",
    "criteria": [{"answer": "price over earnings"}]}
    dc = improve.diagnose(cont, 0.3, [], "P/E is vibes", floor=0.8)
    assert dc["kind"] == "content" and "price over earnings" in dc["criteria"]


def test_improve_propose_addendum_and_ab_decision():
    """propose_addendum makes one deduped hint per missed tool; ab_decision only
    approves a real improvement."""
    from financial_research_assistant import improve

    diagnoses = [
        {"kind": "routing", "query": "my realized gains?", "expected": ["realized_gains"]},
        {"kind": "routing", "query": "gains again?", "expected": ["realized_gains"]}, # dup tool
        {"kind": "routing", "query": "beating the market?", "expected": ["portfolio_vs_benchmark"]},
        {"kind": "content", "query": "explain P/E", "criteria": ["x"]},
    ]
    addendum = improve.propose_addendum(diagnoses)
    assert addendum.count("realized_gains") == 1 # deduped
    assert "portfolio_vs_benchmark" in addendum
    assert improve.propose_addendum([{"kind": "content"}]) == "" # nothing actionable
    assert improve.ab_decision(0.80, 0.90) is True
    assert improve.ab_decision(0.90, 0.90) is False # tie never applies
    assert improve.ab_decision(0.90, 0.85) is False # regression never applies


def test_improve_render_report_has_sections():
    """render_report surfaces the A/B verdict, routing misses, and the addendum."""
    from financial_research_assistant import improve

    diagnoses = [{"kind": "routing", "query": "gains?", "score": 0.0,
    "expected": ["realized_gains"], "called": ["allocation"]}]
    report = improve.render_report(
        diagnoses, "hint here", "2026-07-15T00:00:00Z", mean=0.7, ab=(0.7, 0.85)
    )
    assert "delta: **+0.150** → **APPLY**" in report
    assert "Routing misses" in report and "realized_gains" in report
    assert "hint here" in report


def test_ci_gate_emits_diagnosis_on_failure(monkeypatch, tmp_path):
    """On a gate failure, _emit_diagnosis writes a report with the routing miss and
    a proposed addendum — reusing the already-run results (no re-run)."""
    import importlib.util
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    monkeypatch.syspath_prepend(str(eval_dir)) # so ci_gate's `import evaluate` works
    spec = importlib.util.spec_from_file_location("ci_gate", eval_dir / "ci_gate.py")
    ci_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_gate)

    monkeypatch.setenv("FINANCIAL_RESEARCH_EVAL_DIR", str(tmp_path))
    items = [{"query": "my realized gains?", "eval_type": "trajectory",
    "criteria": [{"tool": "realized_gains"}]}]
    results = [{"score": 0.0, "tools": ["allocation"], "_answer": "..."}]
    ci_gate._emit_diagnosis(items, results, mean=0.0)
    report = (tmp_path / "ci-diagnosis-report.md").read_text()
    assert "realized_gains" in report and "Routing misses" in report
