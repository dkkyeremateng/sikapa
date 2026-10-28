"""``search_history``: the full text of earlier turns, after compaction drops it.

Compaction (``/compact``, or ``AGENT_AUTO_COMPACT`` before a turn) replaces the
older messages of a conversation with a summary, so a detail the summary left
out is gone from the model's context — "what stop did I say for AAPL last
month?" on a phone chat that has run for weeks. Every interface also writes
each turn to the session's transcript (``sessions.log_turn``), which compaction
never touches, so the detail is still on disk. This tool lets the model read it
back.

The session is the graph's own ``thread_id``, which LangChain passes to a tool
that takes a ``RunnableConfig`` — so a tool call cannot name, or be steered
into, another conversation except through ``all_sessions``.
"""

from __future__ import annotations

from typing import Any
import re

from langchain_core.runnables import RunnableConfig

from . import sessions

#: Turns returned per search: enough to find the one meant, small enough not to
#: refill the context compaction just emptied.
LIMIT = 5

#: How many recent conversations ``all_sessions`` reads: each scheduled run is
#: its own transcript, so the store grows by one a day per recurring task.
MAX_SESSIONS = 200

_QUERY_CHARS = 240
_ANSWER_CHARS = 420

_WORD = re.compile(r"[a-z0-9][a-z0-9$%.\-]*")
_STOP = frozenset(
    "a an and are about at be did do does for from had has have i in is it me my "
    "of on or our said say the that this to was we were what when where which who "
    "with you your".split()
)


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(
        w.strip(".-") for w in _WORD.findall((query or "").lower())
        if w.strip(".-") and w.strip(".-") not in _STOP
    ))


def _found(term: str, text: str) -> int:
    """Where ``term`` starts a word in ``text`` (lowercased), or -1."""
    m = re.search(rf"(?<![a-z0-9]){re.escape(term)}", text)
    return m.start() if m else -1


def _snippet(text: str, at: int, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    start = max(0, min(at - width // 3, len(text) - width))
    out = text[start:start + width]
    return ("…" if start else "") + out + ("…" if start + width < len(text) else "")


def search(session_ids: list[str], query: str, limit: int = LIMIT) -> list[dict[str, Any]]:
    """The turns of ``session_ids`` that match ``query`` best: most query words
    first, then most recent. Each ``{session, ts, query, answer}``, trimmed to a
    snippet around the first match."""
    from .adapter import instruction

    terms = _terms(query)
    if not terms:
        return []
    # (score, time, position): a transcript is stamped to the second, so turns
    # logged together tie on time and their order in the file decides.
    hits: list[tuple[int, str, int, dict[str, Any]]] = []
    for sid in session_ids:
        for n, turn in enumerate(sessions.read_transcript(sid)):
            asked = instruction(str(turn.get("query") or ""))
            answer = str(turn.get("answer") or "")
            low_q, low_a = asked.lower(), answer.lower()
            score = sum(1 for t in terms if _found(t, low_q) >= 0 or _found(t, low_a) >= 0)
            if not score:
                continue
            at = next((p for t in terms if (p := _found(t, " ".join(low_a.split()))) >= 0), 0)
            hits.append((score, str(turn.get("ts") or ""), n, {
                "session": sid, "ts": str(turn.get("ts") or ""),
                "query": _snippet(asked, 0, _QUERY_CHARS),
                "answer": _snippet(answer, at, _ANSWER_CHARS),
            }))
    hits.sort(key=lambda h: (h[0], h[1], h[2]), reverse=True)
    return [h[3] for h in hits[:limit]]


def _recent_sessions() -> list[str]:
    base = sessions.store_dir()
    if not base.exists():
        return []
    files = sorted((p for p in base.glob("*.jsonl") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files[:MAX_SESSIONS]]


def search_history(query: str, all_sessions: bool = False, *, config: RunnableConfig) -> str:
    """Search the full text of earlier turns: what the user asked and what you
    answered. Use it when the user refers to something said before that is not
    in your context — older turns of a long conversation are replaced by a
    summary, so details the summary left out (a price level, a ticker, a date, a
    decision) are only found here. E.g. 'what stop did I set for AAPL?', 'what
    did you say about NVDA last week?'. Searches this conversation; set
    ``all_sessions`` to search every saved conversation (other chats, the
    terminal, scheduled runs). Returns the best-matching turns with their dates;
    quote them as what was said then, not as current figures."""
    session = str((config.get("configurable") or {}).get("thread_id") or "")
    scope = _recent_sessions() if all_sessions else ([session] if session else [])
    if not scope:
        return "No conversation to search: this turn has no session."
    if not _terms(query):
        return "Give a few words to search for — a ticker, a level, a topic."
    found = search(scope, query)
    if not found:
        where = "any saved conversation" if all_sessions else "this conversation's earlier turns"
        return f"Nothing in {where} matches {query!r}."
    lines = [f"{len(found)} earlier turn(s) matching {query!r}, best first:"]
    for hit in found:
        tag = f" · {hit['session']}" if all_sessions else ""
        lines.append(f"\n[{hit['ts']}{tag}]\nUser: {hit['query']}\nYou: {hit['answer']}")
    return "\n".join(lines)


HISTORY_TOOLS = [search_history]
