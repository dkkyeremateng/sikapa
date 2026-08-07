"""Framework adapter: bridges the compiled graph to the AgentEvent stream.

Interfaces (TUI, headless CLI, eval harness) depend only on ``run_turn``;
swapping agent frameworks means reimplementing this module and nothing else.

Tool-call events (real mode): ``astream(stream_mode="messages")`` yields
every message produced inside the graph. Tool calls announced on streamed
``AIMessageChunk.tool_call_chunks`` (or complete ``AIMessage.tool_calls``
from non-streaming nodes) become ``tool_start`` events; the ``ToolMessage``
the tool node produces on completion becomes the paired ``tool_end`` with
wall-clock duration and a result snippet. The fake graph has no tools, so
fake runs emit no tool events.

Reasoning events: any streamed chunk carrying
``additional_kwargs["reasoning_content"]`` becomes a ``reasoning`` event
(``text`` = the thought, ``agent`` = AGENT_NAME). One code path serves both
modes — the fake graph attaches a deterministic ``reasoning_content`` to its
answer (so the 💭 panel is exercised offline), and a real reasoning model's
``reasoning_content`` deltas surface the same way.
"""

from typing import Any
import contextlib
import datetime
import itertools
import json
import os
import re
import time

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

try:  # aggregates token usage across the model invocation(s) in the turn
    from langchain_core.callbacks import get_usage_metadata_callback
except Exception:  # pragma: no cover - older langchain-core
    get_usage_metadata_callback = None

from .events import AgentEvent, format_duration
from .graph import build_graph, real_graph_session, summarize_messages
from .graph import _build_real_graph  # no-MCP graph for state read/rewrite
from .pricing import context_cap
from .memory import get_memory, inject, recall_facts

AGENT_NAME = "Agent"  # single-agent scaffold: one name, depth stays 0


def describe_error(exc: BaseException) -> str:
    """Flatten an exception to something a user can act on.

    The graph runs inside anyio task groups (the MCP session), so a failure
    anywhere arrives wrapped: ``str()`` gives "unhandled errors in a TaskGroup
    (1 sub-exception)", which is true and useless — the actual 404 or auth error
    is a leaf several levels down. Walk to the leaves and report those, keeping
    the type name because "NotFoundError" vs "AuthenticationError" is usually the
    whole diagnosis. Credentials are masked: an httpx error can carry the request
    headers.
    """
    from .auth import redact

    leaves: list[str] = []

    def walk(e: BaseException, depth: int = 0) -> None:
        subs = getattr(e, "exceptions", None)  # ExceptionGroup / anyio group
        if subs and depth < 5:
            for sub in subs:
                walk(sub, depth + 1)
            return
        text = str(e).strip()
        leaves.append(f"{type(e).__name__}: {text}" if text else type(e).__name__)

    walk(exc)
    # dict.fromkeys dedupes while keeping order — a fan-out often fails identically
    # in several tasks at once.
    return redact("; ".join(dict.fromkeys(leaves)) or type(exc).__name__)

# Fake graphs are cached (cheap, and their MemorySaver keeps fake-mode history).
# Real turns build the graph PER TURN inside a per-turn IBKR MCP session (see
# _graph_ctx) — the session is opened and closed within the turn's own task, so
# its stdio connection stays warm for every tool call in the turn yet never
# outlives it (task-safe; nothing dangles at shutdown). A persistent checkpointer
# per (session_id, model, think) carries conversation state across the rebuilds.
_fake_graphs: dict[tuple[str, bool], Any] = {}
_checkpointers: dict[tuple[str, str | None, bool], Any] = {}

_call_ids = itertools.count(1)  # fallback ids when the provider omits them


def _fake_graph_for(session_id: str, think: bool):
    key = (session_id, think)
    if key not in _fake_graphs:
        _fake_graphs[key] = build_graph(fake=True, think=think)
    return _fake_graphs[key]


def _checkpointer_for(session_id: str, model: str | None, think: bool):
    from langgraph.checkpoint.memory import MemorySaver

    key = (session_id, model, think)
    if key not in _checkpointers:
        _checkpointers[key] = MemorySaver()
    return _checkpointers[key]


_CHECKPOINT_KEYWORDS = {"1", "true", "yes", "on", "default"}


def _checkpoint_db_path():
    """Path to the durable checkpoint SQLite DB, or None for the default
    in-memory checkpointer.

    Enabled via ``FINANCIAL_RESEARCH_CHECKPOINT_DB``: a filesystem path stores
    conversation state there (surviving restarts, so ``--resume`` restores the
    model's actual memory, not just the transcript); the keywords ``1``/``true``/
    ``on``/``default`` select the standard location
    (``~/.financial-research-assistant/checkpoints.db``). Unset (the default)
    keeps the in-process ``MemorySaver`` — so tests and ``--fake`` stay hermetic
    and behavior is unchanged unless you opt in."""
    from pathlib import Path

    raw = (os.environ.get("FINANCIAL_RESEARCH_CHECKPOINT_DB") or "").strip()
    if not raw:
        return None
    if raw.lower() in _CHECKPOINT_KEYWORDS:
        return Path.home() / ".financial-research-assistant" / "checkpoints.db"
    return Path(os.path.expandvars(raw)).expanduser()


def durable_checkpoints_enabled() -> bool:
    """True when conversation state is persisted to disk (``FINANCIAL_RESEARCH_
    CHECKPOINT_DB`` set), so a restart restores the model's actual memory — not just
    the replayed transcript. False (the default) keeps the in-process ``MemorySaver``,
    which is why nothing sensitive is written to disk unless the user opts in."""
    return _checkpoint_db_path() is not None


@contextlib.asynccontextmanager
async def checkpointer_ctx(session_id: str, model: str | None, think: bool):
    """Yield the checkpointer for one turn.

    Default (no ``FINANCIAL_RESEARCH_CHECKPOINT_DB``): the cached in-process
    ``MemorySaver``, whose state must persist across the per-turn graph rebuilds
    within a process — so it stays cached, exactly as before.

    Durable: a per-turn ``AsyncSqliteSaver`` opened on the configured DB file and
    closed on exit — opened and closed inside the turn's own task, mirroring the
    MCP session's task-affinity rule (an aiosqlite connection cached across turns
    would be finalized in a different task). State lives in the file, so there is
    nothing to cache between turns; a restart reopens the same file and the
    thread's history is still there."""
    db = _checkpoint_db_path()
    if db is None:
        yield _checkpointer_for(session_id, model, think)
        return
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        yield saver


@contextlib.asynccontextmanager
async def _graph_ctx(session_id: str, fake: bool, model: str | None, think: bool):
    """Yield the graph for one turn. Fake mode reuses a cached graph; real mode
    opens a per-turn MCP session (real_graph_session) whose warm connection
    serves every tool call in the turn and is closed on exit, with a persistent
    checkpointer (in-memory or durable SQLite) so history carries across turns."""
    if fake:
        yield _fake_graph_for(session_id, think)
    else:
        async with checkpointer_ctx(session_id, model, think) as checkpointer:
            async with real_graph_session(
                model=model, think=think, checkpointer=checkpointer
            ) as graph:
                yield graph


def _delete_durable_thread(session_id: str) -> None:
    """Erase a session's rows from the durable checkpoint DB (no-op when durable
    mode is off or the file is absent). Introspects the schema and deletes from
    every table carrying a ``thread_id`` column, so it survives langgraph schema
    changes rather than hardcoding table names."""
    import sqlite3

    db = _checkpoint_db_path()
    if db is None or not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            for t in tables:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
                if "thread_id" in cols:
                    conn.execute(f'DELETE FROM "{t}" WHERE thread_id = ?', (session_id,))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass  # a locked/corrupt DB must not break /new


def _forget_cached(store: dict[tuple[Any, ...], Any], session_id: str) -> None:
    """Drop every entry of ``store`` whose key starts with ``session_id``.

    The key list is materialized first: popping while iterating a dict raises.
    """
    for key in [k for k in store if k and k[0] == session_id]:
        store.pop(key, None)


def reset_session(session_id: str) -> None:
    """Drop any cached fake graph and checkpointer for ``session_id`` so its next
    turn starts with fresh conversation memory, and erase its durable checkpoint
    rows when durable mode is on. ``/new`` calls this."""
    # Each store is keyed by a differently-shaped tuple (fake graphs by
    # (session, think); checkpointers by (session, model, think)), so iterating
    # them together unions the key types and a key from one is not a valid key for
    # the other. Both share the session id in slot 0, which is all this needs.
    _forget_cached(_fake_graphs, session_id)
    _forget_cached(_checkpointers, session_id)
    _last_input.pop(session_id, None)
    _delete_durable_thread(session_id)


# Auto-compaction (interface-agnostic): the last turn's input-token count per
# session, used to decide whether to compact *before* the next turn. This makes
# /compact's benefit available to every run_turn caller — headless services, the
# eval harness, bots, REPLs — not just the TUI command.
_last_input: dict[str, int] = {}


def _autocompact_fraction() -> float | None:
    """Fraction of the context window at which run_turn compacts before a turn,
    read from ``AGENT_AUTO_COMPACT`` (e.g. ``"0.85"``). Unset, non-numeric, or
    outside ``(0, 1]`` disables it — so default behavior is unchanged."""
    raw = os.environ.get("AGENT_AUTO_COMPACT")
    if not raw:
        return None
    try:
        frac = float(raw)
    except ValueError:
        return None
    return frac if 0 < frac <= 1 else None


def _resolved_model(model: str | None) -> str:
    """The model name used for pricing/context display, matching what _make_llm
    actually builds — so a non-OpenAI provider's default is reflected, not a
    hardcoded gpt-4.1-mini. Delegates to graph.resolved_model, which the context
    middleware's trigger threshold also uses."""
    from .llm import resolved_model

    return resolved_model(model)


#: What a synthesized result says. Phrased as a fact about the RUN rather than
#: about the tool, because the tool may well have worked — the turn just died
#: before its answer was recorded, and a model told "the tool failed" would
#: report a failure that never happened.
INTERRUPTED_TOOL_RESULT = (
    "No result was recorded — the previous turn ended before this tool call "
    "completed (cancelled, or the run failed). Call the tool again if you still "
    "need it; do not report its outcome from memory."
)


def _tool_call_ids(message: BaseMessage) -> list[tuple[str, str]]:
    """``(id, name)`` for every tool call an AI message will serialize.

    Includes ``invalid_tool_calls`` — a call whose arguments failed to parse is
    still sent to the provider as a ``tool_use`` block, so it still needs a
    result to pair with.
    """
    out: list[tuple[str, str]] = []
    for group in ("tool_calls", "invalid_tool_calls"):
        for call in getattr(message, group, None) or []:
            cid = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if cid:
                out.append((cid, name or "tool"))
    return out


def repair_tool_call_pairs(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], int, int]:
    """Make a message history satisfy "every tool call is answered, exactly once".

    Anthropic (and OpenAI) reject a conversation where a ``tool_use`` block has no
    ``tool_result`` in the next message. A turn that dies between the model node
    and the tool node — cancelled with Esc, killed mid-run, or failed inside the
    tool node — checkpoints exactly that shape, and then EVERY later turn on the
    thread is rejected before it starts. The session is bricked until ``/clear``,
    which throws away the conversation to fix a bookkeeping artifact.

    So the history is repaired on the way in rather than trusted: an unanswered
    call gets a synthetic error result, and a result whose call is not in the
    history at all is dropped. Returns ``(messages, synthesized, dropped)``.
    """
    declared = {cid for m in messages for cid, _n in _tool_call_ids(m)}
    out: list[BaseMessage] = []
    pending: list[tuple[str, str]] = []
    synthesized = dropped = 0

    def flush() -> None:
        nonlocal synthesized
        for cid, name in pending:
            out.append(ToolMessage(
                content=INTERRUPTED_TOOL_RESULT,
                tool_call_id=cid,
                name=name,
                status="error",
            ))
            synthesized += 1
        pending.clear()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in declared:
                dropped += 1  # a result for a call no longer in the history
                continue
            # Results arrive as a run directly after their AI message, so this
            # only clears what that message is still waiting on.
            pending[:] = [p for p in pending if p[0] != msg.tool_call_id]
            out.append(msg)
            continue
        flush()  # any other message ends the run of results
        out.append(msg)
        pending.extend(_tool_call_ids(msg))
    flush()
    return out, synthesized, dropped


async def heal_thread(graph: Any, config: RunnableConfig, as_node: str) -> int:
    """Repair a thread's checkpoint in place. Returns how many calls it answered.

    Runs before every turn, so a thread broken by a cancel or a crash recovers on
    the next message instead of erroring forever. A clean thread costs one state
    read and is left untouched.
    """
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    state = await graph.aget_state(config)
    messages = list(state.values.get("messages", []) or [])
    if not messages:
        return 0
    fixed, synthesized, dropped = repair_tool_call_pairs(messages)
    if not (synthesized or dropped):
        return 0
    await graph.aupdate_state(
        config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *fixed]},
        as_node=as_node,
    )
    return synthesized


COMPACT_KEEP_LAST = 4  # recent messages left verbatim after the summary seed


async def compact_session(
    session_id: str,
    fake: bool = False,
    model: str | None = None,
    think: bool = True,
    keep_last: int = COMPACT_KEEP_LAST,
) -> dict[str, Any]:
    """Summarize the older messages in this thread and rewrite its checkpoint so
    the running context (and ctx%) shrinks, while recent turns stay verbatim and
    the conversation continues coherently.

    Reads the thread's message history from the checkpointer, summarizes all but
    the last ``keep_last`` messages (rounded forward to a user-message boundary so
    the kept tail never starts mid tool-call), and replaces the history with a
    single summary message followed by that tail. Returns
    ``{"removed", "kept", "summary"}``; ``removed == 0`` means there was too little
    history to be worth compacting (a no-op).
    """
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    if fake:
        # Fake mode has no model to summarize with; operate on the cached fake
        # graph's own checkpointer and stub the summary so /compact is exercised
        # offline (tests, --fake).
        graph = _fake_graph_for(session_id, think)
        return await _rewrite_thread(graph, "respond", config, fake, model, keep_last)
    # A no-MCP ReAct graph sharing the turn's checkpointer (in-memory or durable
    # SQLite): same AgentState schema as the turn graph, so it reads/rewrites the
    # same checkpoint without opening an IBKR MCP session just to edit state. The
    # durable saver is opened for the read+rewrite and closed here, in this task.
    async with checkpointer_ctx(session_id, model, think) as checkpointer:
        graph = _build_real_graph(model, think, extra_tools=[], checkpointer=checkpointer)
        return await _rewrite_thread(graph, "model", config, fake, model, keep_last)


async def _rewrite_thread(graph: Any, as_node: str, config: RunnableConfig, fake: bool,
                          model: str | None, keep_last: int) -> dict[str, Any]:
    """Read the thread's messages from ``graph``'s checkpoint, summarize all but
    the last ``keep_last`` (rounded to a user-message boundary), and rewrite the
    checkpoint to a single summary seed plus that verbatim tail."""
    from langchain_core.messages import HumanMessage, RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    state = await graph.aget_state(config)
    messages = list(state.values.get("messages", []))
    if len(messages) <= keep_last + 1:
        return {"removed": 0, "kept": len(messages), "summary": ""}

    # Cut at or after the target so the kept tail begins on a user message (or is
    # empty) — never a bare ToolMessage that would dangle without its AI tool-call.
    cut = len(messages) - keep_last
    while cut < len(messages) and not isinstance(messages[cut], HumanMessage):
        cut += 1
    head, tail = messages[:cut], messages[cut:]
    if not head:  # nothing old enough to summarize after the boundary shift
        return {"removed": 0, "kept": len(messages), "summary": ""}

    if fake:
        summary = f"[compacted {len(head)} earlier message(s)]"
    else:
        summary = await summarize_messages(head, model=model)

    seed = HumanMessage(content=f"Summary of the earlier conversation:\n{summary}")
    await graph.aupdate_state(
        config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), seed, *tail]},
        as_node=as_node,
    )
    return {"removed": len(head), "kept": len(tail) + 1, "summary": summary}


def _sum_usage(usage_metadata: dict[str, Any] | None) -> tuple[int, int, int, int]:
    """Sum input/output/cache-read/cache-write tokens across every model in a
    usage dict. Both cache figures are subsets of ``input_tokens``."""
    tin = tout = tcache = twrite = 0
    for v in (usage_metadata or {}).values():
        tin += int(v.get("input_tokens", 0) or 0)
        tout += int(v.get("output_tokens", 0) or 0)
        details = v.get("input_token_details") or {}
        tcache += int(details.get("cache_read", 0) or 0)
        twrite += int(details.get("cache_creation", 0) or 0)
    return tin, tout, tcache, twrite


def _usage_delta(usage_metadata: dict[str, Any] | None) -> tuple[int, int, int, int]:
    """Tokens from a single message's flat ``usage_metadata`` (one model call),
    as opposed to ``_sum_usage`` which reads the callback's by-model aggregate.
    Used to emit a live ``usage`` event as each model call in a turn completes."""
    um = usage_metadata or {}
    details = um.get("input_token_details") or {}
    return (
        int(um.get("input_tokens", 0) or 0),
        int(um.get("output_tokens", 0) or 0),
        int(details.get("cache_read", 0) or 0),
        int(details.get("cache_creation", 0) or 0),
    )


def _chunk_text(chunk: BaseMessage) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    # Content-block format: a list of dicts carrying "text" entries.
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def _final_answer(messages: list[BaseMessage]) -> str:
    """The turn's answer, read back from checkpointed state.

    Only reached when the run produced no streamed token chunks, which is exactly
    where two shapes the streaming path never sees turn up:

    * A provider that doesn't stream reports its reply as content BLOCKS (a list
      of dicts), not a string. Taking ``.content`` raw handed a list to everything
      downstream — ``settle_schedule_claim`` raised ``TypeError`` on it and the
      outer handler reported a COMPLETED turn as an error. ``_chunk_text`` already
      flattens both shapes, so it does the reading here too.
    * A run stopped at the recursion limit ends on a ``ToolMessage``, whose
      content is a tool result rather than an answer.

    So this walks back to the last AI message with text, mirroring how
    ``subagents._final_text`` reads a subagent's reply, and returns "" when there
    is genuinely no answer instead of passing off some other message as one.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _chunk_text(msg).strip()
            if text:
                return text
    return ""


def _snippet(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _args_complete(args: str) -> bool:
    """True once an incrementally streamed args string is whole JSON."""
    if not args:
        return False
    try:
        json.loads(args)
    except ValueError:
        return False
    return True


def _tool_start(cid: str, entry: dict[str, Any]) -> AgentEvent:
    entry["announced"] = True
    return AgentEvent(
        "tool_start",
        f"{AGENT_NAME} · {entry['name']} …",
        agent=AGENT_NAME,
        tool=entry["name"],
        detail=_snippet(entry["args"], 400),
        call_id=cid,
    )


def _tool_end(cid: str, entry: dict[str, Any], msg: ToolMessage) -> AgentEvent:
    ok = getattr(msg, "status", "success") != "error"
    dt = time.monotonic() - entry["started"]
    # Chart tools split their result: `content` is the summary the model sees,
    # `artifact` is the full text including the chart art, which is display-only
    # (see tools.ChartText). Prefer the artifact so the panel still shows the
    # whole chart while the model's context carries only the stats.
    display = getattr(msg, "artifact", None) or msg.content
    return AgentEvent(
        "tool_end",
        f"{AGENT_NAME} · {entry['name']} {'✓' if ok else '✗'} ({format_duration(dt)})",
        agent=AGENT_NAME,
        tool=entry["name"],
        # Roomy cap so a rendered chart shows in full in the tool-result panel;
        # other results are far shorter than this.
        detail=_snippet(display, 4000),
        duration=dt,
        ok=ok,
        call_id=cid,
    )


THINK_TOOL = "think"  # the scratchpad tool; its calls render as 💭 reasoning


def _think_text(args: str) -> str:
    """Pull the ``thought`` out of a think-tool call's args JSON."""
    try:
        return (json.loads(args) or {}).get("thought", "").strip() or "(thinking)"
    except (ValueError, AttributeError):
        return (args or "").strip() or "(thinking)"


def _announce(cid: str, entry: dict[str, Any]) -> AgentEvent | None:
    """The event to emit when a tool call is first seen: a ``tool_start`` for a
    real tool. For the think tool, return ``None`` and defer — its ``reasoning``
    event is emitted when the call's result returns, stamped with the step's
    duration (measured the same way a tool's duration is)."""
    if entry["name"] == THINK_TOOL:
        entry["announced"] = True
        entry["is_think"] = True
        return None
    return _tool_start(cid, entry)


def _fired_alerts() -> list[AgentEvent]:
    """One ``alert`` event per user rule that fired during the tool call that
    just finished. Alerts are evaluated deep inside ``build_digest``, so they'd
    otherwise only reach the user as prose in that tool's result — easy to miss
    in a long digest. Draining at tool_end lets an interface surface them the
    moment they trigger, while keeping the digest text the source of truth."""
    from .alerts import drain_triggered

    return [AgentEvent("alert", msg, agent=AGENT_NAME) for msg in drain_triggered()]


TOOLS_NODE = "tools"  # create_agent's tool-executing node


def _from_own_node(meta: dict[str, Any] | None) -> bool:
    """True when a streamed chunk is the agent's OWN reply, rather than output
    from a model running inside one of its tools.

    A chat model invoked inside a tool inherits the parent run's callbacks, so
    LangGraph streams its tokens into this same ``messages`` stream. That happens
    in two shapes, which report different metadata:

    * a bare model called in a tool (report synthesis, self-critique) — reports
      ``langgraph_node="tools"``;
    * a whole subagent graph — reports its inner ``langgraph_node="model"``, but
      stays namespaced under the parent's ``tools:<id>`` task.

    Either way the work sits under the tool task, so that is what we test. It
    matters most for ``dispatch_subagents``, which runs several subagents at
    once: unfiltered, their tokens interleave word-by-word with each other and
    with the real answer, and the reply renders as an unreadable mash. Nothing is
    lost by dropping them — a subagent's answer comes back as its tool result and
    renders in that tool's panel.

    Everything else is kept by comparing the task namespace to the node, so this
    holds for any graph shape without naming the answering node (the real agent
    answers from ``model``, the fake graph from ``respond``).
    """
    node = (meta or {}).get("langgraph_node") or ""
    ns = (meta or {}).get("checkpoint_ns") or ""
    # A completed message carries no task namespace (reasoning models that don't
    # stream, and the fake graph, report their reply that way), so it stands in
    # as its own root.
    root = ns.split(":", 1)[0] if ns else node
    if TOOLS_NODE in (root, node):
        return False
    return root == node


async def _stream_events(graph: Any, inputs: dict[str, Any], config: RunnableConfig):
    """Translate ``stream_mode="messages"`` output into AgentEvents.

    Yields "reasoning" for streamed model thinking, "token" for streamed
    answer text, plus paired "tool_start"/"tool_end" for every tool call.

    **Fragility note:** The tool-call chunk pairing below relies on
    ``AIMessageChunk.tool_call_chunks`` matching LangGraph's streaming
    format. If a future provider adapter emits complete tool calls in a
    single chunk (instead of incrementally), the ``by_index`` / ``calls``
    pairing may break. Inspect raw chunk shape first when debugging
    orphaned tool_start or mismatched tool names.
    """
    calls: dict[str, dict[str, Any]] = {}   # call_id -> {name, args, started, announced}
    by_index: dict[int, str] = {}  # streaming chunk index -> call_id

    def register(cid: str | None, name: str, args: str) -> tuple[str, dict[str, Any]]:
        cid = cid or f"call_{next(_call_ids)}"
        entry = calls.get(cid)
        if entry is None:
            entry = {
                "name": name,
                "args": args,
                "started": time.monotonic(),
                "announced": False,
            }
            calls[cid] = entry
        return cid, entry

    async for chunk, meta in graph.astream(
        inputs, config, stream_mode="messages"
    ):
        own = _from_own_node(meta)
        # Any message type may carry the model's chain-of-thought in
        # additional_kwargs["reasoning_content"] (reasoning models put it
        # there; the fake graph mirrors that). Surface it before tool/token
        # handling so the 💭 panel can render alongside the answer.
        rc = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
        if rc and own:
            yield AgentEvent("reasoning", rc, agent=AGENT_NAME)
        # Live token usage: a model call reports its usage on its final chunk, so
        # emit a per-call delta as each completes (ReAct turns have several). The
        # counts tick up mid-run instead of jumping at the end; run_turn reconciles
        # any provider that doesn't stream per-call usage. Deliberately NOT filtered
        # to the agent's own node: a subagent's tokens are billed too, so they must
        # count toward the turn's cost even though its text isn't shown.
        din, dout, dcache, dwrite = _usage_delta(getattr(chunk, "usage_metadata", None))
        if din or dout or dcache or dwrite:
            yield AgentEvent(
                "usage", "", tokens_in=din, tokens_out=dout,
                tokens_cache=dcache, tokens_cache_write=dwrite,
            )
        if isinstance(chunk, ToolMessage):
            cid, entry = register(chunk.tool_call_id, chunk.name or "tool", "")
            if entry["name"] == THINK_TOOL:
                # The think tool's result ends a reasoning step: emit the thought
                # now (no 🛠 panel), stamped with how long the step took.
                entry["is_think"] = True
                dt = time.monotonic() - entry["started"]
                yield AgentEvent(
                    "reasoning", _think_text(entry["args"]),
                    agent=AGENT_NAME, duration=dt,
                )
            else:
                if not entry["announced"]:
                    # Args never became parseable: announce late so start/end pair.
                    yield _tool_start(cid, entry)
                yield _tool_end(cid, entry, chunk)
                for ev in _fired_alerts():
                    yield ev
        elif isinstance(chunk, AIMessageChunk):
            for tc in chunk.tool_call_chunks:
                idx, cid = tc.get("index"), tc.get("id")
                if cid and cid in calls:  # provider repeats ids every chunk
                    entry = calls[cid]
                elif cid:  # first chunk of a new call
                    cid, entry = register(cid, tc.get("name") or "", "")
                    if idx is not None:
                        by_index[idx] = cid
                else:  # continuation chunks carry only the index
                    cid = by_index.get(idx) if idx is not None else None
                    if cid is None:
                        cid, entry = register(None, tc.get("name") or "", "")
                        if idx is not None:
                            by_index[idx] = cid
                    else:
                        entry = calls[cid]
                name = tc.get("name")
                if name and not entry["announced"] and name != entry["name"]:
                    entry["name"] += name  # fragmented names concatenate
                entry["args"] += tc.get("args") or ""
                if (
                    not entry["announced"]
                    and entry["name"]
                    and _args_complete(entry["args"])
                ):
                    ev = _announce(cid, entry)  # None for the think tool (deferred)
                    if ev is not None:
                        yield ev
            text = _chunk_text(chunk)
            if text and own:
                yield AgentEvent("token", text)
        elif isinstance(chunk, AIMessage):
            # Complete message from a non-streaming node: whole tool calls.
            for tc in chunk.tool_calls:
                cid = tc.get("id")
                if cid and cid in calls:
                    continue  # already tracked via streamed chunks
                cid, entry = register(
                    cid,
                    tc.get("name") or "tool",
                    json.dumps(tc.get("args") or {}, ensure_ascii=False),
                )
                ev = _announce(cid, entry)  # None for the think tool (deferred)
                if ev is not None:
                    yield ev


# First-person claims that a task was created. Deliberately narrow: "you have 3
# scheduled tasks" (a `list_scheduled_tasks` answer) must not match, so every
# pattern needs the model asserting it did the thing.
_SCHEDULE_CLAIM = re.compile(
    # No object required after the verb. The first version demanded one of
    # "a/the/this/it" and a live run slipped straight past it with "I've scheduled
    # AN earnings analysis" — the determiner is exactly the wrong thing to hinge on.
    r"(?:\bi(?:'ve| have| ’ve)?\s+(?:now\s+|just\s+)?(?:scheduled|queued|set\s+up)\b"
    r"|\bi'?ll\s+schedule\b"
    r"|\b(?:task|analysis|it)\s+(?:is|has been)\s+(?:now\s+)?(?:scheduled|queued)\b"
    r"|^\s*(?:✓|✅)?\s*\**scheduled\b)",
    re.IGNORECASE | re.MULTILINE,
)

#: Any of these having run means the answer's scheduling talk is grounded in a real
#: call, so it is left alone.
_TASK_TOOLS = frozenset({"schedule_task", "list_scheduled_tasks", "cancel_scheduled_task"})

_UNBACKED_SCHEDULE_NOTE = (
    "\n\n---\n"
    "⚠️ **Correction — nothing was actually scheduled.** The answer above claims a "
    "task was created, but the `schedule_task` tool was never called, so no task "
    "exists and nothing will run. Ask again, or create it directly with:\n"
    "`financial-research-assistant --schedule 'WHEN|PROMPT'`  ·  check with `--tasks`."
)


#: Asked of the cheap tier when a claim needs repairing. Extraction only — it does
#: not decide WHETHER to schedule (the main turn already told the user it had), just
#: what the task should say.
_REPAIR_PROMPT = (
    "Extract a scheduled task from this exchange. The assistant told the user it "
    "scheduled work but failed to create it, so you are recovering the details.\n\n"
    "USER ASKED:\n{user}\n\nASSISTANT REPLIED:\n{answer}\n\n"
    'Reply with ONLY a JSON object: {{"prompt": "...", "when": "...", "repeat": "once"}}\n'
    "- prompt: the instruction for a FRESH assistant that cannot see this exchange — "
    "name the ticker, the event, and exactly what to produce.\n"
    "- when: the time the assistant said it would run, as '2026-08-14 09:00', "
    "'tomorrow 9am', 'friday' or '+2h'. If no time was stated, use 'tomorrow 9am'.\n"
    "- repeat: once, hourly, daily, weekdays or weekly.\n"
    "No prose, no code fence."
)


async def _repair_schedule_claim(user_msg: str, answer: str) -> str | None:
    """Create the task the answer claims exists. Returns a confirmation, or None.

    The main turn already told the user it scheduled something; the honest options
    are to make that true or to retract it. This makes it true, using the cheap tier
    for what is a pure extraction (which ticker, what to produce, when) rather than
    a judgement — the decision to schedule was the user's and has already been
    acted on in the reply they will read.

    Only ever reached from an unbacked claim, so it costs nothing on a normal turn.
    Any failure returns None and the caller falls back to retracting.
    """
    from . import tasks
    from .llm import quick_llm

    try:
        resp = await quick_llm().ainvoke(
            _REPAIR_PROMPT.format(user=user_msg[:2000], answer=answer[:2000])
        )
        raw = getattr(resp, "content", "")
        if not isinstance(raw, str):
            return None
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        spec = json.loads(match.group(0))
        task = tasks.add_task(
            str(spec.get("prompt") or "").strip(),
            str(spec.get("when") or "tomorrow 9am").strip(),
            str(spec.get("repeat") or "once").strip().lower(),
        )
    except Exception:  # noqa: BLE001 - extraction, parsing, bad time, unwritable store
        return None
    due = datetime.datetime.fromisoformat(task["due"]).astimezone()
    return (
        "\n\n---\n"
        f"✅ **Task created — `[{task['id']}]`, running {due:%Y-%m-%d %H:%M} local.** "
        "(The scheduling tool was not called on the first pass, so the task was "
        "recovered from this answer. Check it with `--tasks`; cancel with "
        f"`--unschedule {task['id']}`.)"
    )


async def settle_schedule_claim(
    user_msg: str, answer: str, called_tools: set[str], fake: bool = False
) -> str:
    """Make an answer's scheduling claim true, or visibly retract it.

    Live testing put ``claude-haiku-4-5`` at roughly one real ``schedule_task`` call
    per four identical requests, while it claimed success on most of the misses.
    Prompt and description changes moved that a little and no further: on one miss
    the model looked up the earnings calendar and then simply never scheduled. So
    the guarantee is enforced here instead of hoped for upstream.
    """
    if fake or not answer or _TASK_TOOLS & called_tools:
        return answer
    if not _SCHEDULE_CLAIM.search(answer):
        return answer
    repaired = await _repair_schedule_claim(user_msg, answer)
    return answer + (repaired if repaired is not None else _UNBACKED_SCHEDULE_NOTE)


def verify_schedule_claim(answer: str, called_tools: set[str]) -> str:
    """Append a correction when the answer claims a schedule that never happened.

    Live testing found ``claude-haiku-4-5`` calling ``schedule_task`` on only one of
    three identical requests — and on the other two it *said* "✓ Scheduled" anyway.
    That is worse than the promise-to-check-back this feature replaced: the user
    walks away believing work is queued when nothing is.

    A prompt rule cannot fix it (the rule is what produces the confident phrasing),
    so the claim is checked against what the turn actually did. Conservative by
    construction: it fires only on a first-person creation claim with no task tool
    called at all, so an answer *listing* existing tasks, or one that genuinely
    scheduled, is untouched. A false positive prints a correction the user can
    disprove with ``--tasks``; a false negative is a lie they cannot.
    """
    if not answer or _TASK_TOOLS & called_tools:
        return answer
    return answer + _UNBACKED_SCHEDULE_NOTE if _SCHEDULE_CLAIM.search(answer) else answer


def _recall_and_frame(user_msg: str) -> tuple[str, str | None]:
    """Build a turn's injected user content — durable-fact memory plus a few-shot
    preamble from earlier feedback — and an optional "recalled …" status line.
    When long-term memory is disabled this passes the message through unchanged
    and returns no status, so default behavior is untouched."""
    if get_memory() is None:
        return user_msg, None
    from .feedback import format_fewshot, recall_feedback

    facts = recall_facts(user_msg)
    exemplars, avoids = recall_feedback(user_msg)
    content = format_fewshot(exemplars, avoids) + inject(facts, user_msg)
    n = len(facts) + len(exemplars) + len(avoids)
    if not n:
        return content, None
    detail = f"recalled {n} memory item(s)"
    if exemplars or avoids:
        detail += f" (incl. {len(exemplars) + len(avoids)} feedback)"
    return content, detail


async def run_turn(
    user_msg: str,
    session_id: str,
    fake: bool = False,
    model: str | None = None,
    think: bool = True,
):
    """Run one conversational turn.

    Async generator yielding zero or more "reasoning"/"token"/"status"/
    "tool_start"/"tool_end" AgentEvents, then exactly one "usage" event, then
    exactly one "final" (or "error"). Conversation state is checkpointed per
    ``session_id`` via the graph's thread_id config; ``model`` overrides the
    model factory for this session. ``think=False`` suppresses reasoning
    events (and the fake graph's reasoning), so no 💭 trace is produced.
    """
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    # Long-term memory (opt-in via MEMORY_BACKEND): recall relevant durable facts
    # AND few-shot guidance from earlier feedback (👍/👎 on similar questions),
    # inject both into the prompt; remember the exchange after the answer.
    mem = get_memory()
    content, recall_status = _recall_and_frame(user_msg)
    inputs = {"messages": [{"role": "user", "content": content}]}
    try:
        if recall_status:
            yield AgentEvent("status", recall_status)
        # Auto-compaction: if the previous turn's input reached the configured
        # fraction of the context window, compact before running this turn so the
        # context stays bounded. Works for any interface (headless, eval, service),
        # not just the TUI's /compact command.
        frac = _autocompact_fraction()
        if frac is not None:
            prev = _last_input.get(session_id, 0)
            cap = context_cap(_resolved_model(model))
            if prev and prev >= cap * frac:
                res = await compact_session(
                    session_id, fake=fake, model=model, think=think
                )
                if res.get("removed"):
                    _last_input[session_id] = 0
                    yield AgentEvent(
                        "status",
                        f"auto-compacted {res['removed']} older message(s) — context "
                        f"was ~{round(prev / cap * 100)}% of the window",
                        # Context just shrank: tell the UI to drop the ctx% gauge now,
                        # before this (compacted) turn even runs.
                        context_tokens=0,
                    )
        # Built inside the try: real mode raises here when no model is
        # configured, and that must surface as an "error" event too.
        # Enter the per-turn graph context: for real mode this opens the IBKR MCP
        # session (reused for every tool call this turn) and closes it on exit,
        # all within this task. It also raises here when no model is configured,
        # which must surface as an "error" event too.
        async with _graph_ctx(session_id, fake, model, think) as graph:
            # A turn that died between the model node and the tool node leaves a
            # tool call with no result in the checkpoint, and every provider
            # rejects that history — so the session would answer nothing until
            # it was cleared. Heal it here instead, and say so rather than
            # editing the conversation behind the user's back.
            try:
                healed = await heal_thread(
                    graph, config, "respond" if fake else "model"
                )
            except Exception:  # noqa: BLE001 - a failed repair must not block the turn
                healed = 0
            if healed:
                yield AgentEvent(
                    "status",
                    f"recovered {healed} unfinished tool call(s) from an "
                    "interrupted turn — their results are gone, so ask again if "
                    "an answer below looks thin",
                )
            # One code path for both modes: the fake graph streams its answer as a
            # complete AIMessage (no token chunks) carrying reasoning_content, so
            # the reasoning panel is exercised and the answer is read back below.
            parts: list[str] = []
            called_tools: set[str] = set()  # for the unbacked-claim check below
            emitted_in = emitted_out = emitted_cache = emitted_write = 0  # live usage already yielded
            cb_ctx = (
                get_usage_metadata_callback()
                if get_usage_metadata_callback is not None
                else contextlib.nullcontext()
            )
            with cb_ctx as cb:
                async for ev in _stream_events(graph, inputs, config):
                    if ev.kind == "reasoning" and not think:
                        continue  # thinking disabled: drop the reasoning trace
                    if ev.kind == "tool_end":
                        called_tools.add(ev.tool)
                    if ev.kind == "token":
                        parts.append(ev.text)
                    elif ev.kind == "usage":  # live per-call delta from _stream_events
                        emitted_in += ev.tokens_in
                        emitted_out += ev.tokens_out
                        emitted_cache += ev.tokens_cache
                        emitted_write += ev.tokens_cache_write
                    yield ev
                if parts:
                    answer = "".join(parts)
                else:
                    # Model/endpoint did not stream token chunks (fake graph, or a
                    # non-streaming endpoint); the run still executed and
                    # checkpointed, so read the answer back from graph state.
                    state = await graph.aget_state(config)
                    answer = _final_answer(list(state.values.get("messages") or []))
                tokens_in, tokens_out, tokens_cache, tokens_write = _sum_usage(
                    getattr(cb, "usage_metadata", None)
                )
            if mem:
                mem.remember(user_msg, answer)  # persist the exchange for future turns
            # Remember this turn's input size so the next turn can decide whether to
            # auto-compact (see the pre-turn check above).
            _last_input[session_id] = tokens_in
            # Reconcile: emit only the remainder beyond what streamed live, so the
            # per-call deltas plus this event sum to the authoritative callback
            # total. A provider that never streams per-call usage emits nothing
            # above, so this carries the full total (the original single-event
            # behavior); fake mode has no usage at all, so this is the one all-zero
            # event before final.
            yield AgentEvent(
                "usage", "",
                tokens_in=max(0, tokens_in - emitted_in),
                tokens_out=max(0, tokens_out - emitted_out),
                tokens_cache=max(0, tokens_cache - emitted_cache),
                tokens_cache_write=max(0, tokens_write - emitted_write),
                # Authoritative context size for this turn (same figure the
                # auto-compaction threshold uses), so the footer's ctx% reflects
                # how full the window is now — not the cumulative session input.
                context_tokens=tokens_in,
            )
            yield AgentEvent(
                "final",
                await settle_schedule_claim(user_msg, answer, called_tools, fake),
            )
    except Exception as e:  # surface as an event, never raise into the UI
        yield AgentEvent("error", describe_error(e))
