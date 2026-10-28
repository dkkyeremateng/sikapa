"""search_history: earlier turns stay findable after compaction summarizes them.

Offline. Transcripts are written with the real ``sessions.log_turn`` into the
throwaway home the conftest sets up; the tool is run through LangGraph's tool
node, which is what hands it the conversation's ``thread_id``.
"""

import asyncio

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from financial_research_assistant import catalog, history, sessions


def _ask(query, session="telegram-999", **args):
    """Call the tool the way the agent's tool step does: a tool node in a graph
    running ``session``'s thread."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    call = {"name": "search_history", "args": {"query": query, **args}, "id": "c1"}
    g = StateGraph(MessagesState)
    g.add_node("tools", ToolNode([history.search_history]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    out = asyncio.run(g.compile(checkpointer=MemorySaver()).ainvoke(
        {"messages": [AIMessage(content="", tool_calls=[call])]},
        config={"configurable": {"thread_id": session}},
    ))
    return out["messages"][-1].content


def _chat():
    sessions.log_turn("telegram-999", "set a stop on AAPL at 172.50",
                      "Noted: AAPL stop at 172.50, about 6% under today's close.")
    sessions.log_turn("telegram-999", "how did NVDA do this week?",
                      "NVDA rose 4.1% on the week on datacenter guidance.")
    sessions.log_turn("tui-1", "what is my AAPL cost basis?",
                      "Your AAPL cost basis is 151.20 across two lots.")


def test_the_model_cannot_see_or_set_the_session():
    schema = history.search_history  # what the agent binds
    from langchain_core.tools import StructuredTool

    args = StructuredTool.from_function(schema).args
    assert set(args) == {"query", "all_sessions"}


def test_a_detail_from_earlier_in_this_chat_is_found():
    _chat()
    out = _ask("what stop did I set for AAPL?")
    assert "172.50" in out and "[" in out
    assert "cost basis" not in out, "another conversation is not searched by default"


def test_all_sessions_searches_every_conversation():
    _chat()
    out = _ask("AAPL cost basis", all_sessions=True)
    assert "151.20" in out and "tui-1" in out


def test_more_matching_words_rank_first_then_the_newest():
    sessions.log_turn("s", "NVDA earnings", "old: NVDA beat")
    sessions.log_turn("s", "NVDA guidance and earnings", "both words")
    sessions.log_turn("s", "NVDA again", "new: NVDA flat")
    found = history.search(["s"], "NVDA earnings guidance")
    assert [f["answer"] for f in found][:2] == ["both words", "old: NVDA beat"]
    assert history.search(["s"], "NVDA")[0]["answer"] == "new: NVDA flat"


def test_a_scheduled_runs_framing_is_not_what_is_searched():
    from financial_research_assistant import scheduler

    sessions.log_turn("task-s1-x", f"{scheduler._TASK_PREAMBLE}\n\ncheck MSFT", "MSFT up 1%.")
    found = history.search(["task-s1-x"], "report analysis scratch")
    assert found == [], "the preamble's words are not the user's"
    assert history.search(["task-s1-x"], "MSFT")[0]["query"] == "check MSFT"


def test_nothing_found_and_nothing_to_search_say_so():
    _chat()
    assert "Nothing in this conversation" in _ask("TSLA")
    assert "Give a few words" in _ask("what did I say")


def test_the_agent_has_the_tool():
    assert "search_history" in {catalog.tool_name(t) for t in catalog.active_tools()}


def test_a_compacted_conversation_points_at_the_search():
    """After /compact the model sees only a summary; the seed says where the rest is."""
    from financial_research_assistant.adapter import _fake_graph_for, compact_session, run_turn

    sid = "compact-history"

    async def go():
        for q in ("set a stop on AAPL at 172.50", "and NVDA?", "and AMD?", "and TSLA?"):
            async for _ in run_turn(q, sid, fake=True):
                pass
        await compact_session(sid, fake=True, keep_last=2)
        state = await _fake_graph_for(sid, True).aget_state({"configurable": {"thread_id": sid}})
        return str(state.values["messages"][0].content)

    seed = asyncio.run(go())
    assert seed.startswith("Summary of the earlier conversation") and "search_history" in seed
