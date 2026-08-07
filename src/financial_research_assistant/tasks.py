"""Scheduled tasks — work the agent runs later, on its own, and delivers to you.

The assistant is otherwise strictly request/response: it exists only while a turn
is running, so "monitor NOMD earnings tomorrow and analyse the results" was a
promise nothing could keep. This module is the missing half — a durable record of
*what to run, when, and where to send the answer* — and ``scheduler.run_due``
executes whatever is due.

A task is one JSON object in ``~/.financial-research-assistant/tasks.json``
(override ``FINANCIAL_RESEARCH_TASKS_FILE``), written ``0600`` because a prompt can
quote position sizes and account names::

    {"id": "s1",
     "prompt": "NOMD reported. Pull actuals vs consensus and give a call.",
     "due": "2026-08-14T13:30:00+00:00",   # always UTC in the file
     "repeat": "once",                      # once | hourly | daily | weekdays | weekly
     "session": "task-s1", "channel": "",   # "" = every configured channel
     "status": "pending",                   # pending | done | error
     "runs": 0, "last_run": null, "last_ok": null, "last_result": ""}

Three rules the rest of the module exists to enforce:

**Times are stored in UTC and shown in local time.** A task due "tomorrow 9am" is
scheduled against the wall clock the user is looking at, but a laptop that crosses
a timezone (or a cron job inheriting a different TZ) must not silently shift it.

**Claiming is atomic.** ``claim_due`` marks a task as running inside the same lock
that selects it, so two overlapping ticks — a cron run and a ``--watch`` loop, or a
tick that outlives its interval — cannot both execute the same task and bill two
model runs for one job.

**A recurring task reschedules from its due time, not from now.** Rescheduling
from completion drifts: a daily 09:00 task that runs at 09:04 would creep to 09:08
and eventually into the afternoon.

Nothing here reaches the network or the model. ``scheduler`` runs the turn,
``channels`` delivers the answer, and this module only remembers.
"""

from __future__ import annotations

from typing import Any
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .storage import write_private

#: How a finished task picks its next due time. "once" retires it.
REPEATS = ("once", "hourly", "daily", "weekdays", "weekly")

#: Cap on stored result text. The store is read on every tick and re-serialised on
#: every write; keeping whole reports here would turn a 2 KB file into megabytes.
#: The answer's home is the channel it was delivered to — this is only a receipt.
_RESULT_SNIPPET = 400

#: A task that errors is retried on the next tick until this many attempts, then
#: parked as "error" so a permanently-broken prompt stops burning model calls on
#: every tick forever.
MAX_ATTEMPTS = 3

#: How many undeliverable answers to re-attempt before giving up on the channel.
#: Higher than MAX_ATTEMPTS because a redelivery is FREE — the model call is
#: already spent — so the only cost of trying again is one HTTP request.
MAX_DELIVERY_ATTEMPTS = 8

#: Tasks run per tick. Bounded so a backlog built up while the machine was off
#: drains over consecutive ticks instead of firing twenty model runs at once.
DEFAULT_BATCH = 10


def batch_limit() -> int:
    raw = (os.environ.get("FINANCIAL_RESEARCH_TASK_BATCH") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_BATCH


def tasks_file() -> Path:
    raw = os.environ.get("FINANCIAL_RESEARCH_TASKS_FILE")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "tasks.json"


@contextmanager
def _locked():
    """Serialize read-modify-write across processes.

    A cron tick and a ``--watch`` loop can run at the same moment; without this
    they interleave read/modify/write and one silently drops the other's status
    update — which in practice means a task runs twice. POSIX-only; without
    ``fcntl`` this degrades to no locking rather than failing, the same trade
    ``auth.py`` makes for its store.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    path = tasks_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_tasks() -> list[dict[str, Any]]:
    """Every stored task. A missing, unreadable or corrupt file reads as "none"
    rather than raising — a bad hand-edit must not take down the tick that would
    have run the other tasks."""
    path = tasks_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict) and t.get("id") and t.get("prompt")]


def save_tasks(items: list[dict[str, Any]]) -> None:
    """Write the store ``0600``, atomically — a prompt naming holdings is never
    briefly world-readable, and a tick reading a half-written file would see "no
    tasks" and skip everything due. See ``storage.write_private``."""
    write_private(tasks_file(), json.dumps(items, indent=2) + "\n", prefix=".tasks-")


# --- runner heartbeat ----------------------------------------------------------
#
# Scheduling something and having nothing run it is the failure this whole feature
# exists to prevent, and it is SILENT: the task sits pending, the store looks fine,
# and the user waits for a message that is never coming. So the runner stamps every
# pass — including empty ones, which is the point — and anything that promises a
# future delivery checks the stamp first.


def _heartbeat_file() -> Path:
    return tasks_file().with_name("scheduler-heartbeat.json")


def record_tick(now: datetime | None = None) -> None:
    """Stamp that a runner just made a pass. Best-effort; never raises."""
    try:
        path = _heartbeat_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_tick": (now or now_utc()).isoformat()}), encoding="utf-8"
        )
    except OSError:
        pass  # a missing heartbeat costs a warning, not a run


def last_tick() -> datetime | None:
    try:
        raw = json.loads(_heartbeat_file().read_text(encoding="utf-8")).get("last_tick")
    except (OSError, ValueError, AttributeError):
        return None
    return _parse_due(raw)


def _max_silence_minutes() -> int:
    raw = (os.environ.get("FINANCIAL_RESEARCH_RUNNER_MAX_SILENCE") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 60


def runner_is_live() -> bool:
    """Whether a runner has made a pass recently enough to be trusted.

    Tolerant by default (60 minutes, ``FINANCIAL_RESEARCH_RUNNER_MAX_SILENCE``): a
    15-minute cron may miss a couple of ticks to a sleeping laptop without the agent
    crying wolf, but a machine with no runner at all is caught immediately.
    """
    tick = last_tick()
    if tick is None:
        return False
    return (now_utc() - tick) <= timedelta(minutes=_max_silence_minutes())


def runner_warning() -> str:
    """The line to append when nothing is going to run what was just scheduled."""
    if runner_is_live():
        return ""
    tick = last_tick()
    when = (
        f"not since {tick.astimezone():%Y-%m-%d %H:%M} local"
        if tick
        else "never — no runner has ever run here"
    )
    return (
        f" ⚠ WARNING: no task runner is active ({when}), so this will NOT run and the "
        "user will receive nothing. Tell them to start one: `--watch` in a terminal, "
        "or install the launchd/cron entry in the README ('Scheduled tasks')."
    )


def _next_id(items: list[dict[str, Any]]) -> str:
    """A short stable id (``s1``, ``s2``, …), one past the current max — matching
    the ``a1``/``a2`` scheme alert rules already use."""
    n = 0
    for t in items:
        tid = str(t.get("id", ""))
        if tid.startswith("s") and tid[1:].isdigit():
            n = max(n, int(tid[1:]))
    return f"s{n + 1}"


# --- time ---------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(dt: datetime) -> datetime:
    """Attach the local zone to a naive datetime, then convert to UTC.

    Naive means "what the user typed", which is local wall-clock; treating it as
    UTC instead would fire a 9am task at 9am UTC — up to a working day off.
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(timezone.utc)


def parse_when(text: str, now: datetime | None = None) -> datetime:
    """Turn a human time expression into a UTC datetime. Raises ValueError.

    Accepts what a person actually types at a chat prompt or a shell:
    ``2026-08-14T13:30``, ``2026-08-14 09:00``, ``2026-08-14`` (09:00 local),
    ``tomorrow 9am``, ``today 16:00``, ``monday 8:30``, ``+2h``, ``+30m``, ``+3d``.

    Bare dates default to 09:00 local rather than midnight: "monitor X on Friday"
    means during the day, and a midnight run would report on a market that has been
    shut for hours.
    """
    now = now or now_utc()
    local_now = now.astimezone()
    raw = (text or "").strip().lower()
    if not raw:
        raise ValueError("no time given")

    # Relative: +90m, +2h, +3d, +1w (also accepted without the leading '+')
    m = re.fullmatch(r"\+?(\d+)\s*(m|min|mins|minutes?|h|hr|hrs|hours?|d|days?|w|weeks?)", raw)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("m") and unit != "mo":
            return now + timedelta(minutes=n)
        if unit.startswith("h"):
            return now + timedelta(hours=n)
        if unit.startswith("d"):
            return now + timedelta(days=n)
        return now + timedelta(weeks=n)

    # Optional leading day word, with the rest treated as a time-of-day.
    day_offset: int | None = None
    rest = raw
    for word, offset in (("today", 0), ("tomorrow", 1), ("tonight", 0)):
        if raw == word or raw.startswith(word + " "):
            day_offset, rest = offset, raw[len(word):].strip()
            break
    weekdays = ("monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday")
    if day_offset is None:
        for i, name in enumerate(weekdays):
            if raw == name or raw.startswith(name + " ") or raw.startswith(name[:3] + " "):
                ahead = (i - local_now.weekday()) % 7 or 7  # always the NEXT one
                day_offset = ahead
                rest = raw.split(" ", 1)[1].strip() if " " in raw else ""
                break

    if day_offset is not None:
        hour, minute = _parse_clock(rest) if rest else (9, 0)
        target = (local_now + timedelta(days=day_offset)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        return _to_utc(target)

    # Absolute: ISO-ish date, optionally with a time.
    iso = raw.replace("/", "-")
    for fmt in ("%Y-%m-%dt%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dt%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(iso, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            dt = dt.replace(hour=9)
        return _to_utc(dt)

    # A bare time of day: the next occurrence of it.
    try:
        hour, minute = _parse_clock(raw)
    except ValueError:
        raise ValueError(
            f"could not read a time from {text!r} — try '2026-08-14 09:00', "
            "'tomorrow 9am', 'friday', or '+2h'"
        ) from None
    target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= local_now:
        target += timedelta(days=1)
    return _to_utc(target)


def _parse_clock(text: str) -> tuple[int, int]:
    """``9``, ``9am``, ``09:30``, ``9:30 pm``, ``1630`` -> (hour, minute)."""
    t = (text or "").strip().replace(".", ":")
    m = re.fullmatch(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        suffix = m.group(3)
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    m = re.fullmatch(r"(\d{2})(\d{2})", t)  # 1630
    if m and int(m.group(1)) <= 23 and int(m.group(2)) <= 59:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"not a time of day: {text!r}")


def _parse_due(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def next_due(due: datetime, repeat: str) -> datetime | None:
    """The following occurrence after ``due``, or None when the task is done.

    Advanced from the scheduled time and rolled forward until it is in the future:
    a machine that was asleep for three days resumes on the next real occurrence
    rather than firing three catch-up runs of a daily task.
    """
    repeat = (repeat or "once").strip().lower()
    if repeat not in REPEATS or repeat == "once":
        return None
    step = {
        "hourly": timedelta(hours=1),
        "daily": timedelta(days=1),
        "weekdays": timedelta(days=1),
        "weekly": timedelta(weeks=1),
    }[repeat]
    nxt = due + step
    now = now_utc()
    while nxt <= now:
        nxt += step
    if repeat == "weekdays":
        while nxt.astimezone().weekday() >= 5:  # local Sat/Sun -> next Monday
            nxt += step
    return nxt


def describe(task: dict[str, Any]) -> str:
    """One line for a listing — local time, since that's the clock the user set it
    against, with the repeat and last outcome when there is one."""
    due = _parse_due(task.get("due"))
    when = due.astimezone().strftime("%Y-%m-%d %H:%M") if due else "?"
    bits = [f"[{task.get('id')}] {when}"]
    repeat = (task.get("repeat") or "once").lower()
    if repeat != "once":
        bits.append(repeat)
    status = task.get("status", "pending")
    if status != "pending":
        bits.append(status)
    if task.get("channel"):
        bits.append(f"→{task['channel']}")
    prompt = str(task.get("prompt", ""))
    head = "  ".join(bits)
    line = f"{head} — {prompt[:120]}{'…' if len(prompt) > 120 else ''}"
    if task.get("last_run") and task.get("last_ok") is False:
        line += f"\n      last run failed: {str(task.get('last_result', ''))[:160]}"
    if task.get("pending_delivery"):
        # Visible, because otherwise a finished answer nobody received looks
        # identical to one that was delivered.
        line += (f"\n      answer waiting to be delivered "
                 f"({task.get('delivery_attempts', 0)} attempt(s) so far)")
    return line


# --- reading / writing tasks ---------------------------------------------------


def add_task(
    prompt: str,
    when: str,
    repeat: str = "once",
    channel: str = "",
    session: str = "",
) -> dict[str, Any]:
    """Store a task and return it. Raises ValueError on an unreadable time.

    Each task gets its own session id by default, so a scheduled run starts from a
    clean conversation instead of inheriting whatever the interactive session was
    talking about — and so two tasks can never compact each other's history.
    """
    due = parse_when(when)
    rep = (repeat or "once").strip().lower()
    if rep not in REPEATS:
        raise ValueError(f"repeat must be one of: {', '.join(REPEATS)}")
    if not (prompt or "").strip():
        raise ValueError("a task needs a prompt")
    with _locked():
        items = load_tasks()
        task = {
            "id": _next_id(items),
            "prompt": prompt.strip(),
            "due": due.isoformat(),
            "repeat": rep,
            "channel": (channel or "").strip().lower(),
            "session": "",
            "status": "pending",
            "created": now_utc().isoformat(),
            "runs": 0,
            "attempts": 0,
            "last_run": None,
            "last_ok": None,
            "last_result": "",
        }
        task["session"] = (session or "").strip() or f"task-{task['id']}"
        items.append(task)
        save_tasks(items)
    return task


def remove_task(task_id: str) -> bool:
    """Delete one task by id, or every task when given ``all``."""
    tid = (task_id or "").strip().lower()
    with _locked():
        items = load_tasks()
        if not items:
            return False
        if tid in ("all", "*"):
            save_tasks([])
            return True
        kept = [t for t in items if str(t.get("id", "")).lower() != tid]
        if len(kept) == len(items):
            return False
        save_tasks(kept)
        return True


def get_task(task_id: str) -> dict[str, Any] | None:
    tid = (task_id or "").strip().lower()
    for t in load_tasks():
        if str(t.get("id", "")).lower() == tid:
            return t
    return None


def pending_tasks() -> list[dict[str, Any]]:
    """Tasks still waiting to run, soonest first."""
    items = [t for t in load_tasks() if t.get("status") == "pending"]
    return sorted(items, key=lambda t: t.get("due") or "")


def due_count(now: datetime | None = None) -> int:
    """How many tasks are due right now, ignoring the per-tick batch limit.

    So a truncated tick can SAY what it deferred. A cap that silently drops work
    reads as "everything ran" when it didn't.
    """
    now = now or now_utc()
    return sum(
        1 for t in load_tasks()
        if t.get("status") == "pending"
        and (due := _parse_due(t.get("due"))) is not None
        and due <= now
    )


def claim_due(now: datetime | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Mark every task due at ``now`` as running and return them.

    Selection and marking happen inside ONE lock — that is the whole point. Two
    ticks overlapping (a slow run still going when cron fires again) would
    otherwise each select the same pending task and run the model twice for it.

    ``limit`` bounds one tick: a store that accumulated a backlog while the machine
    was off should drain over several ticks rather than firing twenty model runs at
    once. Defaults to ``batch_limit()`` (``FINANCIAL_RESEARCH_TASK_BATCH``); pair it
    with ``due_count`` to report what a truncated tick deferred.
    """
    now = now or now_utc()
    limit = batch_limit() if limit is None else limit
    claimed: list[dict[str, Any]] = []
    with _locked():
        items = load_tasks()
        for t in sorted(items, key=lambda t: t.get("due") or ""):
            if len(claimed) >= limit:
                break
            if t.get("status") != "pending":
                continue
            due = _parse_due(t.get("due"))
            if due is None or due > now:
                continue
            t["status"] = "running"
            t["claimed_at"] = now.isoformat()
            claimed.append(dict(t))
        if claimed:
            save_tasks(items)
    return claimed


def release(task_id: str) -> None:
    """Put a claimed task back to pending, unchanged.

    For the interrupted case — Ctrl-C, a killed watcher — where the run never
    reached a verdict. Without it the task would sit in "running" forever and never
    be claimed again.
    """
    with _locked():
        items = load_tasks()
        for t in items:
            if str(t.get("id")) == str(task_id) and t.get("status") == "running":
                t["status"] = "pending"
                t.pop("claimed_at", None)
                save_tasks(items)
                return


def record_result(task_id: str, ok: bool, result: str, now: datetime | None = None) -> dict[str, Any] | None:
    """Write back the outcome of a run and schedule (or retire) the task.

    Success: a recurring task moves to its next occurrence, a one-shot is done.
    Failure: retried on the next tick until ``MAX_ATTEMPTS``, then parked as
    ``error`` — a prompt that fails every time (a deleted document, a bad ticker)
    otherwise bills a model run on every tick for the rest of time.
    """
    now = now or now_utc()
    with _locked():
        items = load_tasks()
        for t in items:
            if str(t.get("id")) != str(task_id):
                continue
            t.pop("claimed_at", None)
            t["runs"] = int(t.get("runs") or 0) + 1
            t["last_run"] = now.isoformat()
            t["last_ok"] = bool(ok)
            t["last_result"] = (result or "")[:_RESULT_SNIPPET]
            if ok:
                t["attempts"] = 0
                due = _parse_due(t.get("due")) or now
                nxt = next_due(due, str(t.get("repeat") or "once"))
                if nxt is None:
                    t["status"] = "done"
                else:
                    t["status"] = "pending"
                    t["due"] = nxt.isoformat()
            else:
                t["attempts"] = int(t.get("attempts") or 0) + 1
                t["status"] = "pending" if t["attempts"] < MAX_ATTEMPTS else "error"
            save_tasks(items)
            return t
    return None


# --- undelivered answers -------------------------------------------------------
#
# Delivery is best-effort and must never fail a completed run (see channels.py),
# but "best-effort" used to mean the answer was gone: a Telegram blip at the wrong
# moment cost a run that had already been paid for in tokens. So an answer that
# reached NO channel is parked on its task and re-sent on later ticks. Redelivery
# is free — the model call is already spent — which is why it retries far more
# patiently than a failed run does.
#
# Kept on the task record rather than in a second file: it needs the same lock and
# the same atomic write, and the set is bounded by "answers currently undeliverable",
# which is normally empty. `last_result` stays truncated; this holds the full text
# precisely because it still has to be delivered.


def queue_delivery(task_id: str, text: str) -> None:
    """Park an answer that reached no channel, for a later tick to re-send."""
    with _locked():
        items = load_tasks()
        for t in items:
            if str(t.get("id")) == str(task_id):
                t["pending_delivery"] = text
                t["delivery_attempts"] = int(t.get("delivery_attempts") or 0) + 1
                save_tasks(items)
                return


def undelivered() -> list[dict[str, Any]]:
    """Tasks holding an answer that still has to be delivered, oldest first."""
    return [t for t in load_tasks() if t.get("pending_delivery")]


def delivery_done(task_id: str) -> None:
    """Clear a parked answer once it has gone out."""
    with _locked():
        items = load_tasks()
        for t in items:
            if str(t.get("id")) == str(task_id) and t.get("pending_delivery"):
                t.pop("pending_delivery", None)
                t.pop("delivery_attempts", None)
                save_tasks(items)
                return


def delivery_exhausted(task: dict[str, Any]) -> bool:
    return int(task.get("delivery_attempts") or 0) >= MAX_DELIVERY_ATTEMPTS


def purge(keep_errors: bool = True) -> int:
    """Drop finished tasks; returns how many went. Errors are kept by default so a
    failure isn't silently swept away before it's been read."""
    with _locked():
        items = load_tasks()
        kept = [
            t for t in items
            if t.get("status") not in ("done",) and not (t.get("status") == "error" and not keep_errors)
        ]
        removed = len(items) - len(kept)
        if removed:
            save_tasks(kept)
        return removed


# --- Model-facing tools --------------------------------------------------------


def schedule_task(prompt: str, when: str, repeat: str = "once", channel: str = "") -> str:
    """Create a scheduled task that runs later and pushes its answer to the user.

    Call this for ANY request about a future moment: 'monitor NOMD earnings
    tomorrow', 'watch AAPL this week', 'let me know when the 10-Q lands', 'remind me
    Friday', 'every morning before the open'. Calling it is what creates the task —
    describing a schedule in your reply does NOT create one, and a reply that claims
    a task exists when this tool was not called is shown to the user as a false
    confirmation. Report only what this tool returns.

    ``prompt`` is the instruction the future run receives — write it standalone, as
    if to a fresh assistant that cannot see this conversation ("NOMD reported Q3 on
    Aug 13; pull actual EPS/revenue vs consensus and give a buy/hold/sell").
    ``when`` accepts '2026-08-14 09:00', 'tomorrow 9am', 'friday', '+2h'.
    ``repeat`` is once (default), hourly, daily, weekdays or weekly.
    ``channel`` is a delivery channel name (e.g. 'telegram'); blank sends to every
    configured one. Manage with `list_scheduled_tasks` / `cancel_scheduled_task`.
    """
    try:
        task = add_task(prompt, when, repeat, channel)
    except ValueError as exc:
        return f"Could not schedule that: {exc}"
    due_local = (_parse_due(task["due"]) or now_utc()).astimezone()
    from . import channels

    where = channels.describe_targets(task["channel"])
    return (
        f"Scheduled [{task['id']}] for {due_local:%Y-%m-%d %H:%M} local"
        f"{'' if task['repeat'] == 'once' else ', repeating ' + task['repeat']}. "
        f"The answer will be sent to {where}."
        # Never promise a delivery nothing will make: a queued task with no runner
        # is silent — it just sits pending while the user waits for a message.
        + runner_warning()
    )


def list_scheduled_tasks() -> str:
    """List the scheduled tasks (id, when they run, what they will do), so the user
    can see what is queued. Use for 'what have you got scheduled / what are you
    watching / what did you set up for me'."""
    items = load_tasks()
    if not items:
        return ("Nothing scheduled. Use schedule_task(prompt, when) to queue work — "
                "e.g. schedule_task('Analyse NOMD Q3 results vs consensus', 'tomorrow 9am').")
    live = [t for t in items if t.get("status") in ("pending", "running")]
    finished = [t for t in items if t.get("status") not in ("pending", "running")]
    out = ["Scheduled tasks:"] + [f"  {describe(t)}" for t in live]
    if not live:
        out = ["No tasks are waiting to run."]
    warning = runner_warning()
    if live and warning:
        out.append(warning.strip())
    if finished:
        out.append("Finished:")
        out += [f"  {describe(t)}" for t in finished[-5:]]
    return "\n".join(out)


def cancel_scheduled_task(task_id: str) -> str:
    """Cancel a scheduled task by its id (from `list_scheduled_tasks`), or pass
    ``all`` to clear every one. Use for 'stop watching X / cancel that / never
    mind about the Friday one'."""
    if not (task_id or "").strip():
        return "Which one? `list_scheduled_tasks` shows the ids."
    if remove_task(task_id):
        return f"Cancelled {task_id}."
    return f"No scheduled task with id {task_id!r}. Use `list_scheduled_tasks` to see the ids."


TASK_TOOLS = [schedule_task, list_scheduled_tasks, cancel_scheduled_task]
