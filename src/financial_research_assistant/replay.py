"""Side effects that survive a rerun: a tool that acts does so once per request.

An unattended request can run more than once. A Telegram message is committed
only after its reply goes out, so a restart mid-answer answers it again; a
scheduled task whose runner dies is reclaimed when its claim expires, and a
failed run is retried. Each rerun is a fresh model turn that knows nothing of
what the earlier attempt already did, so a tool that acts — schedule a task,
log a call, send a report — would do it twice: a recurring task that delivers
every message twice from then on, a call the track record counts twice.

The runner names the request (``request``): ``telegram:<update_id>`` for a phone
message, ``task:<id>:<due>`` for one occurrence of a scheduled task (``due``
only moves on success, so every attempt at one occurrence shares it). A tool
that acts wraps the act in ``once``, keyed by what the act is ABOUT — not its
exact arguments, since the model words them differently on the second attempt.
A key already in the ledger returns the earlier result instead of acting again.

Outside a request (the TUI, the CLI) there is no key and ``once`` just acts:
typing the same thing twice means it twice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
import contextvars
import json
import threading

from .storage import locked, read_json, state_file, write_private

#: How long an entry is kept. Reruns land within the claim timeout and retry
#: limit — hours, not days — so a week is generous and keeps the file small.
KEEP_DAYS = 7

_REQUEST: contextvars.ContextVar[str] = contextvars.ContextVar("fra_request", default="")

#: Per-attempt call counts, for acts with no stable identity (see ``nth``). A
#: mutable dict in the context var, so every thread the turn's tools run in
#: shares the one counter; the lock covers parallel tool calls.
_COUNTS: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "fra_request_counts", default=None
)
_COUNT_LOCK = threading.Lock()

_NOTE = ("(Already done by an earlier attempt at this same request, which was "
         "interrupted and rerun, so it was not repeated. The earlier result:)\n")


def ledger_file():
    return state_file("replay-ledger.json", "FRA_REPLAY_LEDGER")


@contextmanager
def request(key: str) -> Iterator[None]:
    """Run the block as one attempt at the request ``key``. A context variable,
    so it follows the turn into its tools and nowhere else."""
    token = _REQUEST.set(key)
    counts = _COUNTS.set({})
    try:
        yield
    finally:
        _COUNTS.reset(counts)
        _REQUEST.reset(token)


def current() -> str:
    return _REQUEST.get()


def nth(tool: str) -> str:
    """The identity for an act that has no stable one: its position among this
    attempt's calls to ``tool`` ("1", "2"…). ``schedule_task`` needs it — the
    model rewords the prompt on a rerun, and "in two hours" resolves to a later
    time — so the first call of the rerun is matched to the first call of the
    earlier attempt."""
    counts = _COUNTS.get()
    if counts is None:
        return ""
    with _COUNT_LOCK:
        counts[tool] = counts.get(tool, 0) + 1
        return str(counts[tool])


def already_called(tools: list[str]) -> None:
    """Count calls this attempt inherits rather than makes: a resumed turn keeps
    the earlier attempt's finished tool calls, so its next ``schedule_task`` is
    the one after them, not the first."""
    counts = _COUNTS.get()
    if counts is None:
        return
    with _COUNT_LOCK:
        for tool in tools:
            counts[tool] = counts.get(tool, 0) + 1


def _entry_key(tool: str, identity: str) -> str:
    return f"{_REQUEST.get()}|{tool}|{identity}"


def _done(key: str) -> str | None:
    data = read_json(ledger_file(), {})
    entry = data.get(key) if isinstance(data, dict) else None
    return str(entry.get("result", "")) if isinstance(entry, dict) else None


def _record(key: str, result: str, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=KEEP_DAYS)
    path = ledger_file()
    with locked(path):
        data = read_json(path, {})
        if not isinstance(data, dict):
            data = {}
        kept: dict[str, Any] = {}
        for k, v in data.items():
            try:
                at = datetime.fromisoformat(str(v.get("at")))
            except (AttributeError, TypeError, ValueError):
                continue  # unreadable: dropped, as a corrupt store reads as empty
            if at >= cutoff:
                kept[k] = v
        kept[key] = {"at": now.isoformat(timespec="seconds"), "result": result}
        write_private(path, json.dumps(kept, indent=2))


def once(tool: str, identity: str, act: Callable[[], tuple[bool, str]]) -> str:
    """Run ``act`` once per request for ``(tool, identity)``.

    ``act`` returns ``(ok, text)``. Only a success is recorded: an act that was
    refused ("no price data", a bad time) did nothing, so the rerun may try it.
    """
    if not _REQUEST.get():
        return act()[1]
    key = _entry_key(tool, identity)
    earlier = _done(key)
    if earlier is not None:
        return _NOTE + earlier
    ok, text = act()
    if ok:
        _record(key, text)
    return text


async def once_async(
    tool: str, identity: str, act: Callable[[], Awaitable[tuple[bool, str]]]
) -> str:
    """``once`` for an async act."""
    if not _REQUEST.get():
        return (await act())[1]
    key = _entry_key(tool, identity)
    earlier = _done(key)
    if earlier is not None:
        return _NOTE + earlier
    ok, text = await act()
    if ok:
        _record(key, text)
    return text
