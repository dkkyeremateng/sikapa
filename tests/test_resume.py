"""A scheduled run that stops partway resumes from its last checkpoint.

Driven through the real runner, ``_answer`` and ``run_turn``, on a scripted graph
that checkpoints to a real SQLite file (durable mode, as the server runs). The
graph does what the ReAct agent does — model, tools, model — and its tool step
calls the real ``schedule_task``, so the resume is checked together with the
once-per-request guard it relies on.

Offline: no model, no network.
"""

import asyncio
import contextlib
from typing import cast

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from financial_research_assistant import adapter, scheduler, tasks

ANSWER = "NVDA follow-ups are in place — figures, sources and a verdict here."


@pytest.fixture
def durable(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(adapter, "get_memory", lambda: None)
    from financial_research_assistant import channels

    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (["test"], []))


def _graph_over(monkeypatch, build, sessions=None):
    """Run every real-mode turn on ``build(checkpointer)``, with the same
    checkpointer the real graph gets: a fresh connection to the durable file.
    ``sessions`` collects the conversation each turn ran in."""
    @contextlib.asynccontextmanager
    async def ctx(session_id, fake, model, think):
        if sessions is not None:
            sessions.append(session_id)
        async with adapter.checkpointer_ctx(session_id, model, think) as saver:
            yield build(saver)

    monkeypatch.setattr(adapter, "_graph_ctx", ctx)


def _two_step_graph(log, crash_once):
    """Step 1 schedules A and reads a price; step 2 schedules B. The first time
    the tool step reaches B it dies before running it, as a killed process or a
    provider error would leave the thread."""
    def model(state: MessagesState):
        log.append("model")
        done = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if not done:
            calls = [
                {"name": "schedule_task", "args": {"prompt": "A", "when": "+1d"}, "id": "c1"},
                {"name": "price", "args": {}, "id": "c2"},
            ]
        elif len(done) == 2:
            calls = [{"name": "schedule_task", "args": {"prompt": "B", "when": "+2d"}, "id": "c3"}]
        else:
            return {"messages": [AIMessage(content=ANSWER)]}
        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    def tools(state: MessagesState):
        calls = cast(AIMessage, state["messages"][-1]).tool_calls
        if crash_once and any(c["args"].get("prompt") == "B" for c in calls):
            crash_once.pop()
            raise RuntimeError("killed mid-step")
        out = []
        for c in calls:
            log.append(f"tool:{c['name']}")
            text = tasks.schedule_task(**c["args"]) if c["name"] == "schedule_task" else "101.5"
            out.append(ToolMessage(content=text, tool_call_id=c["id"], name=c["name"]))
        return {"messages": out}

    def build(saver):
        g = StateGraph(MessagesState)
        g.add_node("model", model)
        g.add_node("tools", tools)
        g.add_edge(START, "model")
        g.add_conditional_edges(
            "model", lambda s: "tools" if cast(AIMessage, s["messages"][-1]).tool_calls else END)
        g.add_edge("tools", "model")
        return g.compile(checkpointer=saver)

    return build


def _prompts():
    return sorted(t["prompt"] for t in tasks.load_tasks() if t["prompt"] in ("A", "B"))


def test_a_run_that_died_partway_resumes_instead_of_starting_over(durable, monkeypatch):
    log: list[str] = []
    _graph_over(monkeypatch, _two_step_graph(log, crash_once=[True]))
    parent = tasks.add_task("plan NVDA follow-ups", "+0m")

    first = asyncio.run(scheduler.run_due())
    assert [r["ok"] for r in first] == [False]
    assert log == ["model", "tool:schedule_task", "tool:price", "model"]
    session = tasks.get_task(parent["id"])["run"]["session"]
    assert asyncio.run(adapter.interrupted(session))

    second = asyncio.run(scheduler.run_due())
    assert [r["ok"] for r in second] == [True]
    assert ANSWER in second[0]["answer"]
    # Only the step that never ran is run: A and the price are not fetched again,
    # and B — the second schedule_task of the occurrence — is not mistaken for A.
    assert log[4:] == ["tool:schedule_task", "model"]
    assert _prompts() == ["A", "B"]
    done = tasks.get_task(parent["id"])
    assert done["status"] == "done" and "run" not in done
    assert not asyncio.run(adapter.interrupted(session))


def test_a_run_that_finished_badly_starts_a_fresh_conversation(durable, monkeypatch):
    """A complete answer the gate rejected has nothing left to resume: the retry
    asks again in a new conversation, as before."""
    answers = iter(["ok", ANSWER])

    def build(saver):
        def model(state: MessagesState):
            return {"messages": [AIMessage(content=next(answers))]}

        g = StateGraph(MessagesState)
        g.add_node("model", model)
        g.add_edge(START, "model")
        g.add_edge("model", END)
        return g.compile(checkpointer=saver)

    seen: list[str] = []
    _graph_over(monkeypatch, build, seen)
    parent = tasks.add_task("check NVDA", "+0m")
    assert [r["ok"] for r in asyncio.run(scheduler.run_due())] == [False]
    assert not asyncio.run(adapter.interrupted(tasks.get_task(parent["id"])["run"]["session"]))

    import time

    time.sleep(1.1)  # run ids are stamped to the second
    second = asyncio.run(scheduler.run_due())
    assert [r["ok"] for r in second] == [True] and ANSWER in second[0]["answer"]
    assert len(seen) == 2 and seen[0] != seen[1]


def test_without_durable_checkpoints_nothing_is_resumed(tmp_path, monkeypatch):
    monkeypatch.delenv("FINANCIAL_RESEARCH_CHECKPOINT_DB", raising=False)
    assert not asyncio.run(adapter.interrupted("task-s1-20261028T090000"))


def test_a_new_occurrence_never_inherits_the_last_ones_conversation():
    t = tasks.add_task("daily", "+0m", repeat="daily")
    tasks.note_run(t["id"], t["due"], "task-s1-x")
    assert tasks.unfinished_run(tasks.get_task(t["id"])) == "task-s1-x"
    moved = dict(tasks.get_task(t["id"]), due="2099-01-01T09:00:00+00:00")
    assert tasks.unfinished_run(moved) == ""


def test_a_thread_is_mid_run_only_while_the_graph_owes_a_step():
    from langchain_core.messages import HumanMessage

    q = HumanMessage(content="q")
    call = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
    result = ToolMessage(content="r", tool_call_id="1", name="t")
    stop = adapter._stopped_mid_run
    assert stop([q]) and stop([q, call]) and stop([q, call, result])
    assert not stop([q, call, result, AIMessage(content="answer")])
    assert not stop([])
