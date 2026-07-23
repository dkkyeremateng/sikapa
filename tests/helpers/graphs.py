from typing import cast

from langchain_core.messages import AIMessage, ToolMessage
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
