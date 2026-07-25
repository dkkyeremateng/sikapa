import asyncio
from typing import cast

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph


def scripted_tool_graph():
    """A deterministic tool-calling graph shaped like the real ReAct agent:
    model announces a tool call, the tool node answers with a ToolMessage,
    the model replies. Lets us verify the tool-event pipeline offline."""

    def model(state: MessagesState):
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="counted 2 words")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "word_count",
                            "args": {"text": "hi there"},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        }

    def tools(state: MessagesState):
        return {
            "messages": [
                ToolMessage(content="2", tool_call_id=tc["id"], name=tc["name"])
                for tc in cast(AIMessage, state["messages"][-1]).tool_calls
            ]
        }

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model",
        lambda state: "tools" if cast(AIMessage, state["messages"][-1]).tool_calls else END,
    )
    g.add_edge("tools", "model")
    return g.compile()


def nested_model_graph():
    """A graph whose tool node runs THREE models concurrently — the shape
    ``dispatch_subagents`` creates. A model invoked inside a tool inherits the
    parent run's callbacks, so LangGraph streams its tokens into the same
    ``messages`` stream; unfiltered they interleave word-by-word with each other
    and with the real reply. Only the ``model`` node's own tokens are the answer."""

    def fake(text: str):
        return GenericFakeChatModel(messages=iter([text] * 9))

    async def tools(state: MessagesState):
        await asyncio.gather(
            *(fake(f"NESTED{c} one two three").ainvoke([HumanMessage(content="q")])
              for c in "ABC")
        )
        last = cast(AIMessage, state["messages"][-1])
        return {
            "messages": [
                ToolMessage(content="ok", tool_call_id=tc["id"], name=tc["name"])
                for tc in last.tool_calls
            ]
        }

    async def model(state: MessagesState):
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            parts = [
                str(c.content)
                async for c in fake("ANSWER alpha beta").astream([HumanMessage(content="q")])
            ]
            return {"messages": [AIMessage(content="".join(parts))]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "dispatch", "args": {"tasks": "a"}, "id": "d1"}],
                )
            ]
        }

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model",
        lambda s: "tools" if cast(AIMessage, s["messages"][-1]).tool_calls else END,
    )
    g.add_edge("tools", "model")
    return g.compile()


def scripted_think_graph():
    """Model calls `think`, then `read_source`, then answers — to verify think
    calls surface as reasoning (no 🛠 panel) while real tools still pair up."""

    def model(state: MessagesState):
        msgs = state["messages"]
        did_read = any(isinstance(m, ToolMessage) and m.name == "read_source" for m in msgs)
        did_think = any(isinstance(m, ToolMessage) and m.name == "think" for m in msgs)
        if did_read:
            return {"messages": [AIMessage(content="review done")]}
        nxt = (
            {"name": "read_source", "args": {"path": "a.py"}, "id": "r1"}
            if did_think
            else {"name": "think", "args": {"thought": "read a.py first"}, "id": "t1"}
        )
        return {"messages": [AIMessage(content="", tool_calls=[nxt])]}

    def tools(state: MessagesState):
        last = cast(AIMessage, state["messages"][-1])
        return {
            "messages": [
                ToolMessage(content="ok", tool_call_id=tc["id"], name=tc["name"])
                for tc in last.tool_calls
            ]
        }

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model",
        lambda s: "tools" if cast(AIMessage, s["messages"][-1]).tool_calls else END,
    )
    g.add_edge("tools", "model")
    return g.compile()
