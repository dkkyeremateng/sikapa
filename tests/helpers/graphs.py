import asyncio
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send


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


def alerting_tool_graph(messages: list[str]):
    """A tool-calling graph whose tool fires user alerts while it runs — the
    shape build_digest has when a rule triggers deep inside it."""

    def model(state: MessagesState):
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="digest ready")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "digest", "args": {}, "id": "call_1"}],
                )
            ]
        }

    def tools(state: MessagesState):
        from financial_research_assistant import alerts

        alerts._fired.extend(messages)  # what evaluate_alerts does when a rule fires
        return {
            "messages": [
                ToolMessage(content="digest text", tool_call_id=tc["id"], name=tc["name"])
                for tc in cast(AIMessage, state["messages"][-1]).tool_calls
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


class ScriptedChunkModel(BaseChatModel):
    """A chat model that streams a fixed list of ``AIMessageChunk``s.

    The real providers announce a tool call INCREMENTALLY — a first chunk with the
    id and the opening of the arguments, then continuation chunks carrying only an
    index and more argument text, with the name itself sometimes split across
    them. No fake model in langchain-core emits that shape, so the adapter's
    chunk-pairing path (``by_index`` routing, fragmented names, the
    ``_args_complete`` gate) had no offline coverage. This scripts it.
    """

    chunks: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-chunks"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self.chunks:
            await asyncio.sleep(0)  # yield the loop so concurrent models interleave
            yield ChatGenerationChunk(message=chunk)


def tool_call_chunk(index: int, name: str | None, args: str, call_id: str | None = None):
    """One streamed tool-call fragment, in the shape a provider sends it."""
    return AIMessageChunk(content="", tool_call_chunks=[{
        "index": index, "id": call_id, "name": name, "args": args,
        "type": "tool_call_chunk",
    }])


def interleaved_chunk_graph():
    """Two models streaming a tool call AT THE SAME TIME, in separate tasks of the
    same node — the shape a fan-out produces. Both number their call index 0 and
    send id-less continuation fragments, so the index alone cannot say which call a
    fragment belongs to, and the two argument streams are distinguishable
    (``NVDA`` vs ``AAPL``) precisely so a mix-up is visible."""

    scripts = {
        "price": [
            tool_call_chunk(0, "price_", '{"symbol": '),
            tool_call_chunk(0, "history", '"NVDA"}'),  # the name arrives in pieces too
        ],
        "news": [
            tool_call_chunk(0, "news_", '{"query": '),
            tool_call_chunk(0, "search", '"AAPL"}'),
        ],
    }

    async def model(state: MessagesState):
        which = str(state["messages"][-1].content)
        llm = ScriptedChunkModel(chunks=scripts[which])
        async for _ in llm.astream([HumanMessage(content="go")]):
            pass
        return {"messages": [AIMessage(content=f"done {which}")]}

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_conditional_edges(
        START,
        lambda _s: [
            Send("model", {"messages": [HumanMessage(content=w)]}) for w in scripts
        ],
        ["model"],
    )
    g.add_edge("model", END)
    return g.compile()


def subagent_chunk_graph():
    """The agent calls a tool, and a model running INSIDE that tool announces a
    tool call of its own — what a subagent (or report synthesis) does, streaming
    into the parent's callbacks. Its own tool node is in another graph, so no
    matching result ever reaches this stream."""

    async def tools(state: MessagesState):
        llm = ScriptedChunkModel(chunks=[
            tool_call_chunk(0, "schedule_task", '{"when": "daily"}', call_id="sub1"),
        ])
        async for _ in llm.astream([HumanMessage(content="delegated")]):
            pass
        last = cast(AIMessage, state["messages"][-1])
        return {
            "messages": [
                ToolMessage(content="findings", tool_call_id=tc["id"], name=tc["name"])
                for tc in last.tool_calls
            ]
        }

    def model(state: MessagesState):
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "dispatch_subagent", "args": {"task": "x"},
                                 "id": "d1"}],
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


def usage_reporting_graph(checkpointer=None):
    """A two-step ReAct turn that reports token usage the way a provider does: one
    figure per MODEL CALL, each re-sending the conversation so far, plus a model
    inside the tool reporting its own (separate, much larger) context. Compiled
    with a checkpointer because a turn that streams no token chunks reads its
    answer back from state."""
    from langgraph.checkpoint.memory import MemorySaver

    def usage(tokens_in: int, tokens_out: int) -> dict[str, int]:
        return {"input_tokens": tokens_in, "output_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out}

    async def tools(state: MessagesState):
        llm = ScriptedChunkModel(chunks=[
            AIMessageChunk(content="sub", usage_metadata=usage(90_000, 500)),
        ])
        async for _ in llm.astream([HumanMessage(content="delegated")]):
            pass
        last = cast(AIMessage, state["messages"][-1])
        return {
            "messages": [
                ToolMessage(content="findings", tool_call_id=tc["id"], name=tc["name"])
                for tc in last.tool_calls
            ]
        }

    def model(state: MessagesState):
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [
                AIMessage(content="the answer", usage_metadata=usage(30_000, 200))
            ]}
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": "dispatch_subagent", "args": {"task": "x"}, "id": "d1"}],
            usage_metadata=usage(20_000, 100),
        )]}

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model",
        lambda s: "tools" if cast(AIMessage, s["messages"][-1]).tool_calls else END,
    )
    g.add_edge("tools", "model")
    return g.compile(checkpointer=checkpointer or MemorySaver())


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
