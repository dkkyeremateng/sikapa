"""The runner — executes due tasks and inbound messages, and pushes the answers.

``tasks.py`` remembers what to run, ``channels.py`` knows where to send it; this is
the part that actually runs a turn while nobody is watching. Two entry points, both
wired to CLI flags:

- ``run_due()`` — run everything due right now, then return. Model-free until it
  finds work, so it costs nothing on an empty tick and is safe to call from cron
  or launchd every few minutes (the ``--digest`` / ``--flex-sync`` pattern).
- ``watch()`` — the same thing on a loop, for a machine where adding a cron entry
  is more trouble than leaving a process running. It is deliberately a thin wrapper:
  the scheduling logic must not fork into a "cron version" and a "daemon version".
  It also long-polls the Telegram inbox between passes.
- ``serve()`` — the always-on service. The job pass, the Telegram inbox and any
  registered watchers run as independent loops, so a ten-minute report never
  leaves a phone message waiting, and a loop that crashes is restarted without
  taking the others down.

``run_due`` does NOT read the inbox (it used to claim it did): a cron pass has no
long-poll window, and answering chat is the service's job.

**A scheduled run is a normal turn.** It goes through ``adapter.run_turn`` with its
own session id, so it gets the same tools, the same read-only broker boundary, the
same tracing and the same long-term memory as anything typed into the TUI. The only
differences are that nothing streams to a terminal and the final answer is
delivered rather than printed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
import asyncio
import contextlib
import contextvars
import os
import signal
import sys
import threading
import time
from datetime import datetime

from . import channels, tasks

#: Prefix on every delivered message, so a phone notification is self-identifying
#: rather than an anonymous wall of analysis.
_HEADER = "🤖 {label}"

#: Inbound messages run under one session per chat, so a conversation on the phone
#: keeps its context across messages the way the TUI does.
_CHAT_SESSION = "telegram-{chat_id}"

#: Which concurrency lane a model run takes: "chat" (a phone message) or
#: "background" (scheduled work, event analyses). Only enforced while ``serve``
#: runs; see ``_lane``.
_CURRENT_LANE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fra_lane", default="background"
)

#: The lanes' semaphores while ``serve`` is running, else None (no limits — the
#: one-shot ``--run-due`` and the ``--watch`` loop are sequential anyway).
_lanes: dict[str, asyncio.Semaphore] | None = None


def background_runs() -> int:
    """How many background model runs may overlap (``FRA_BACKGROUND_RUNS``, 1).

    One by default: a scheduled report and an event analysis both hit the same
    rate-limited market-data endpoints and the same token budget, and neither is
    in a hurry. Chat has its own lane and never waits for them."""
    raw = (os.environ.get("FRA_BACKGROUND_RUNS") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 1


#: Lanes this context already holds. A job holds the background lane for its
#: whole run; if it then starts a model turn, that turn must not wait for the
#: very slot its own job is sitting in.
_HELD: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "fra_lanes_held", default=frozenset()
)


@contextlib.asynccontextmanager
async def _lane(name: str) -> AsyncIterator[None]:
    lanes = _lanes
    key = "chat" if name == "chat" else "background"
    if lanes is None or key in _HELD.get():
        yield
        return
    async with lanes[key]:
        token = _HELD.set(_HELD.get() | {key})
        try:
            yield
        finally:
            _HELD.reset(token)


#: Prepended to every scheduled prompt.
#:
#: A task runs in its own fresh session, so it cannot see the conversation that
#: created it — and a prompt written as a back-reference ("send a summary of the
#: analysis we did") arrives as an instruction about something that, from the run's
#: point of view, does not exist. Observed in the wild: the run replied "I don't
#: have a record of that research" and that non-answer was delivered to the user's
#: phone, counted as a success.
#:
#: `schedule_task`'s docstring already asks for standalone prompts; this states it
#: as a fact of the environment instead of advice, and — the load-bearing part —
#: says what to do when the instruction refers to something missing: go and produce
#: it, rather than report that you can't find it.
_TASK_PREAMBLE = (
    "[SCHEDULED TASK] You are running work the user queued earlier. This is a fresh "
    "session: you CANNOT see the conversation that created this task, and no prior "
    "research from it is available to you. Treat the instruction below as "
    "self-contained and gather everything it needs with your tools NOW. If it refers "
    "to a report, analysis or figure you have no record of, PRODUCE that work from "
    "scratch — never reply that you cannot find it or ask a follow-up question, "
    "because nobody is at a terminal to answer. Your reply is delivered to the user "
    "as a message, so make it complete and self-explanatory on its own."
)


#: Openings that mean "I did not do the work", not "here is the work".
#:
#: A task used to count as successful whenever the model returned ANY text, so a
#: run that replied "I don't have a record of that research" was marked done and
#: pushed to the user's phone as if it were the analysis they asked for. There is
#: no cheap way to judge an answer's QUALITY without a second model call, but this
#: class of non-answer is recognisable: it is what the run says instead of starting.
#:
#: Matched only in the opening of the answer, and only for short-to-middling ones.
#: A real report opens with results; one that mentions "no record of a prior filing"
#: in its third paragraph is doing its job, and must not be thrown away.
_NON_ANSWER_PATTERNS = (
    "i don't have a record", "i do not have a record", "no record of",
    "i wasn't able to find", "i was unable to find", "i'm unable to find",
    "could you clarify", "can you clarify", "please clarify",
    "could you provide", "please provide more", "what would you like",
    "i need more information", "i need you to",
)
_NON_ANSWER_WINDOW = 300   # only the opening counts
_NON_ANSWER_MAX_LEN = 1500  # a long answer did the work, whatever its first line
_MIN_ANSWER_LEN = 40       # shorter than this is not an analysis of anything


def _non_answer_reason(text: str) -> str:
    """Why this reply isn't an answer, or "" if it looks like one.

    Deliberately conservative: a false positive costs a retry (and, on the last
    attempt, still delivers), while a false negative is what already happened —
    a confused non-answer delivered as the real thing.
    """
    body = (text or "").strip()
    if len(body) < _MIN_ANSWER_LEN:
        return f"the run returned {len(body)} characters, which is not an answer"
    if len(body) > _NON_ANSWER_MAX_LEN:
        return ""
    opening = body[:_NON_ANSWER_WINDOW].lower()
    for pattern in _NON_ANSWER_PATTERNS:
        if pattern in opening:
            return (
                f"the run opened with {pattern!r} instead of doing the work — it "
                "either could not find what the prompt referred to, or asked a "
                "question nobody is there to answer"
            )
    return ""


def _log(msg: str) -> None:
    """Progress goes to stderr: stdout is the answer, which a cron entry may pipe
    into mail or a file."""
    print(msg, file=sys.stderr, flush=True)


async def _answer(prompt: str, session_id: str, fake: bool = False) -> tuple[bool, str]:
    """Run one turn and return ``(ok, answer_or_error)``.

    Consumes the whole event stream rather than the final event alone: ``run_turn``
    ends in exactly one ``final`` or ``error``, and an error carries the only
    explanation of what went wrong — which is what gets stored and delivered.

    Nobody is at a terminal for any turn run from here, so it runs with the
    unattended toolset (see ``catalog.unattended``), and — under ``serve`` — in
    the lane the caller set (``_CURRENT_LANE``), so a phone message is never
    queued behind a scheduled report.
    """
    from . import catalog

    async with _lane(_CURRENT_LANE.get()):
        with catalog.unattended():
            return await _answer_unlocked(prompt, session_id, fake)


async def _answer_unlocked(prompt: str, session_id: str, fake: bool) -> tuple[bool, str]:
    from . import sessions
    from .adapter import run_turn
    from .tracing import traced

    from .adapter import _resolved_model

    model = "scripted-fake" if fake else _resolved_model(None)
    final = ""
    error = ""
    async for ev in traced(
        run_turn(prompt, session_id, fake=fake),
        user_msg=prompt, session_id=session_id, model=model, fake=fake,
    ):
        if ev.kind == "final":
            final = ev.text
        elif ev.kind == "error":
            error = ev.text
        elif ev.kind == "alert":
            # A rule that fires inside a scheduled run still deserves a push — it
            # is exactly the "tell me when" the user asked for, and there is no
            # terminal here to show the toast.
            #
            # Off the loop thread: `deliver` is urllib, and a blocking socket
            # inside a coroutine stalls EVERYTHING sharing that loop — here, the
            # event stream of the very turn that raised the alert, which stops
            # mid-answer while a notification API takes its time.
            await asyncio.to_thread(channels.deliver, f"🔔 {ev.text}")
    # Persist the turn like the headless CLI does. A background run is the case
    # where a transcript matters MOST — nobody watched it, so without this a task
    # that fails leaves only a one-line reason and there is no way to see what it
    # actually did. `--resume task-s1` replays it.
    try:
        sessions.log_turn(session_id, prompt, error or final or "(no answer)")
    except OSError:
        pass  # a lost transcript must not fail a run that produced an answer
    if error:
        return False, error
    return bool(final), final or "the run produced no answer"


def run_session(task: dict[str, Any], now: datetime | None = None) -> str:
    """The conversation id for ONE run of a task: its stored session plus the run's
    time, e.g. ``task-s1-20260930T0900``.

    One id per run, not per task, for two reasons found the expensive way round.
    A recurring task reused its thread, so a daily brief carried every previous
    day's conversation into the next — a context, and a bill, that grew daily for
    as long as the task lived. And ids restart at ``s1`` once the list is cleared,
    so a brand-new task could open inside an old one's conversation, directly
    contradicting the preamble that tells it this is a fresh session.
    """
    base = str(task.get("session") or f"task-{task.get('id')}")
    return f"{base}-{(now or datetime.now()).strftime('%Y%m%dT%H%M%S')}"


# --- jobs ---------------------------------------------------------------------------
#
# A task of kind "job" runs a registered function instead of a model turn on its
# prompt. The reports are jobs because their figures must be COMPUTED, not typed
# by a model working from a paragraph of instructions (see periodic.py); the Flex
# sync is a job because it needs no model at all.


def register_job(name: str, handler: Any) -> None:
    """Kept for callers that registered through the scheduler; the registry now
    lives in ``jobs`` so that module need not import this one."""
    from . import jobs

    jobs.register_job(name, handler)


def registered_jobs() -> list[str]:
    from . import jobs

    return jobs.registered_jobs()


async def _run_job(task: dict[str, Any], fake: bool) -> tuple[bool, str, list[str], bool]:
    """Run a job task in the background lane; ``(ok, text, files, notify)``."""
    from . import jobs

    name = str(task.get("job") or "")
    handler = jobs.handler_for(name)
    if handler is None:
        known = ", ".join(jobs.registered_jobs()) or "none"
        return False, f"unknown job {name!r} (known: {known})", [], True
    async with _lane("background"):
        res = await handler(task, fake)
    return res.ok, res.text, list(res.files), res.notify


async def run_task(task: dict[str, Any], fake: bool = False) -> dict[str, Any]:
    """Run one claimed task, deliver its answer, and record the outcome.

    Delivery failure does NOT fail the task. The work is done and the model call is
    spent; marking it failed would re-run the whole analysis on the next tick
    because a notification API was down. The failure is logged and the answer stays
    in the task's stored result.
    """
    tid = str(task.get("id"))
    prompt = str(task.get("prompt", ""))
    is_job = task.get("kind") == "job"
    session = run_session(task)
    _log(f"▶ task {tid}: {prompt[:80]}" + ("" if is_job else f" [{session}]"))
    files: list[str] = []
    notify = True
    try:
        if is_job:
            ok, answer, files, notify = await _run_job(task, fake)
        else:
            # The stored prompt stays clean (it's what `--tasks` shows and what the
            # user wrote); the framing is added only on the way into the model.
            ok, answer = await _answer(f"{_TASK_PREAMBLE}\n\n{prompt}", session, fake=fake)
    except asyncio.CancelledError:
        # Ctrl-C or a killed watcher: hand the task back rather than leaving it
        # stuck in "running", which nothing would ever claim again.
        tasks.release(tid)
        raise
    except Exception as exc:  # noqa: BLE001 - a task must not take down the tick
        from .adapter import describe_error

        ok, answer = False, describe_error(exc)
    finally:
        from .adapter import release_session

        release_session(session)

    # A reply that isn't an answer counts as a failure, so the task retries with
    # the framing preamble rather than delivering a shrug as the finished work.
    # Jobs build their own text, so the gate is for model turns only.
    reason = _non_answer_reason(answer) if ok and not is_job else ""
    if reason:
        ok = False
        _log(f"  not an answer: {reason}")

    # Record BEFORE delivering: whether this is the last attempt decides whether
    # the user hears about the failure now or after the retries are exhausted.
    record = tasks.record_result(tid, ok, reason or answer)
    if (record is not None and not ok and record.get("status") == "pending"
            and not record.get("exhausted")):
        # Silent on purpose: three phone notifications for one task that is still
        # being retried is noise, and the outcome is not known yet.
        _log(f"  ✗ task {tid}; retrying next tick "
             f"({record.get('attempts')}/{tasks.MAX_ATTEMPTS})")
        return {"id": tid, "ok": False, "delivered": [], "failed": [], "answer": answer}

    if ok and not notify:
        _log(f"  ✓ task {tid} (nothing to send)"
             + (f"; next {record['due']}" if record and record.get("status") == "pending" else ""))
        return {"id": tid, "ok": True, "delivered": [], "failed": [], "answer": answer}

    label = f"task {tid}" + ("" if ok else " (failed)")
    if record is not None and record.get("exhausted") and record.get("status") == "pending":
        label += " — skipped this run; it stays scheduled"
    body = f"{_HEADER.format(label=label)}\n\n{answer}"
    delivered, failed = await asyncio.to_thread(
        channels.deliver, body, str(task.get("channel") or "")
    )
    if failed:
        _log(f"  delivery failed on: {', '.join(failed)}")
    if not channels.carried_full_text(delivered):
        # The work is done and paid for, so the answer is parked and re-sent on
        # later ticks instead of being lost to a channel that was briefly down.
        #
        # A banner channel does not settle the debt: the desktop notifier is on by
        # default and shows the first 200 characters, so "Telegram was down but the
        # toast went up" used to count as delivered and threw the analysis away.
        tasks.queue_delivery(tid, body)
        if delivered:
            _log(f"  only {', '.join(delivered)} took it, and only as a banner — the "
                 "full answer is parked for redelivery (see --tasks)")
        else:
            # stdout too, so a cron entry redirecting output still captures it.
            print(body, flush=True)
            _log("  nowhere to deliver — parked for redelivery (see --tasks)")
    for path in files:
        sent, lost = await asyncio.to_thread(
            channels.deliver_file, path, "", str(task.get("channel") or ""), True
        )
        if lost or not sent:
            # The message above names the file's path, and it stays on disk.
            _log(f"  file not delivered ({', '.join(lost) or 'no file channel'}): {path}")

    tail = f"; next {record['due']}" if record and record.get("status") == "pending" else ""
    _log(f"  {'✓' if ok else '✗'} task {tid}"
         + (f" → {', '.join(delivered)}" if delivered else "") + tail)
    return {"id": tid, "ok": ok, "delivered": delivered, "failed": failed, "answer": answer}


async def retry_deliveries() -> list[str]:
    """Re-send answers that reached no channel. Returns the task ids that went out.

    Runs at the top of every tick, before any model call: an answer already paid
    for should reach the user before new work is started.
    """
    out: list[str] = []
    for task in tasks.undelivered():
        tid = str(task.get("id"))
        body = str(task.get("pending_delivery") or "")
        if tasks.delivery_exhausted(task):
            _log(f"  giving up on delivering task {tid} after "
                 f"{tasks.MAX_DELIVERY_ATTEMPTS} attempts; the answer is in the log")
            tasks.delivery_done(tid)
            continue
        delivered, _failed = await asyncio.to_thread(
            channels.deliver, body, str(task.get("channel") or "")
        )
        if channels.carried_full_text(delivered):
            tasks.delivery_done(tid)
            out.append(tid)
            _log(f"  redelivered task {tid} → {', '.join(delivered)}")
        else:
            # A banner is not the answer, so the parked text stays parked and waits
            # for a channel that carries the whole thing.
            tasks.queue_delivery(tid, body)  # bumps the attempt counter
    return out


def _score_theses() -> list[dict[str, Any]]:
    """Score the directional calls that have come due, and log each outcome.

    Wrapped because it is housekeeping, not the tick's job: a price source being
    down must not stop tasks from running, and an unscoreable call is left open
    for a later tick rather than burned as a miss (see ``journal.score_entry``).
    """
    from . import journal

    try:
        scored = journal.score_due()
    except Exception as exc:  # noqa: BLE001 - scoring must never sink a tick
        from .adapter import describe_error

        _log(f"couldn't score recorded calls: {describe_error(exc)}")
        return []
    for entry in scored:
        _log(f"  {'✓' if entry.get('hit') else '✗'} {journal.describe_outcome(entry)}")
    return scored


def job_timeout_seconds() -> float:
    """Wall-clock ceiling on one task (``FRA_JOB_TIMEOUT_MINUTES``, default 45).

    Every network call here has its own timeout, but a turn is dozens of them plus
    the model's, and one that never returns would otherwise hold the job loop —
    and every task behind it — until the claim expired and a second copy started.
    """
    raw = (os.environ.get("FRA_JOB_TIMEOUT_MINUTES") or "").strip()
    try:
        minutes = float(raw) if raw else 45.0
    except ValueError:
        minutes = 45.0
    return max(1.0, minutes) * 60.0


async def run_due(
    now: datetime | None = None, fake: bool = False, limit: int | None = None,
    stop: asyncio.Event | None = None,
) -> list[dict[str, Any]]:
    """Claim and run everything due. Returns one result per task run.

    Sequential, not concurrent: each task is a full tool-using turn, and running
    several at once multiplies both the token spend and the load on the same rate-
    limited market-data endpoints. A backlog drains over consecutive ticks.

    ``stop`` (set by ``serve`` on SIGTERM) hands back the claimed tasks that have
    not started yet, so a restart doesn't leave them "running" until their claims
    expire half an hour later.
    """
    # Stamp EVERY pass, including empty ones — an idle runner still proves a runner
    # exists, which is what `schedule_task` checks before promising a delivery.
    tasks.record_tick()
    # Answers already paid for go out before any new work is started.
    await retry_deliveries()
    # Score any directional calls whose horizon has passed. Model-free (two price
    # lookups and a subtraction), so it costs nothing on a tick with none due and
    # cannot itself hallucinate a result.
    _score_theses()
    waiting = tasks.due_count(now)
    claimed = tasks.claim_due(now, limit=limit)
    if not claimed:
        return []
    deferred = waiting - len(claimed)
    # Name what the batch cap dropped: a silent truncation reads as "everything
    # ran" when it didn't.
    _log(f"{len(claimed)} task(s) due"
         + (f" (+{deferred} deferred to the next tick; "
            f"raise FINANCIAL_RESEARCH_TASK_BATCH to widen)" if deferred > 0 else ""))
    results = []
    for task in claimed:
        if stop is not None and stop.is_set():
            tasks.release(str(task.get("id")))
            continue
        try:
            results.append(
                await asyncio.wait_for(run_task(task, fake=fake), job_timeout_seconds())
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            # `run_task` released the claim when it was cancelled, which on its own
            # would re-run the same stuck task on every tick for ever. Counting it
            # as a failed attempt lets the retry limit park it.
            tid = str(task.get("id"))
            reason = f"timed out after {job_timeout_seconds() / 60:.0f} minutes"
            _log(f"  ✗ task {tid}: {reason}")
            try:
                tasks.record_result(tid, False, reason)
            except Exception:  # noqa: BLE001 - the store may be what's stuck
                pass
            results.append(
                {"id": tid, "ok": False, "delivered": [], "failed": [], "answer": reason}
            )
        except Exception as exc:  # noqa: BLE001 - one task must not strand the batch
            # `run_task` already absorbs a failing turn; what reaches here is its
            # bookkeeping — a store write, a channel call — blowing up. Without
            # this the whole rest of the CLAIMED batch is abandoned mid-loop, and
            # every one of those tasks sits in "running" until its claim expires.
            from .adapter import describe_error

            tid = str(task.get("id"))
            reason = describe_error(exc)
            _log(f"  ✗ task {tid}: {reason}")
            try:
                # Only if the run never reached a verdict. `run_task` records
                # BEFORE it delivers, so a failure on the delivery side is already
                # written down, and recording it again would flip a finished task
                # back to pending and pay for the same analysis a second time.
                stored = tasks.get_task(tid)
                if stored is not None and stored.get("status") == "running":
                    tasks.record_result(tid, False, reason)
            except Exception:  # noqa: BLE001 - the store itself may be what failed
                _log(f"  could not record the failure for task {tid}; its claim "
                     "will expire and the task will be retried")
            results.append(
                {"id": tid, "ok": False, "delivered": [], "failed": [], "answer": reason}
            )
    return results


async def poll_inbox(fake: bool = False, timeout: int = 0) -> list[dict[str, Any]]:
    """Answer allowlisted Telegram messages. Returns one result per message.

    ``timeout`` is the long-poll window: 0 to take what is waiting and return, or
    a few seconds in a loop.

    Slash commands (``/tasks``, ``/status``…) are answered without the model, so
    they work with no credential configured and never cost a turn. Anything else
    is a prompt, run in the chat lane.

    At-least-once: each message is committed only after its reply went out (see
    ``telegram.get_updates``), so a restart mid-answer answers it again rather
    than dropping it.
    """
    from . import telegram

    if not telegram.inbound_enabled():
        return []
    try:
        # A long poll parks on a socket for up to POLL_TIMEOUT seconds. Run on the
        # loop thread it would block every other coroutine on it — including the
        # `--watch` tick and any in-flight turn — for the whole window, which is
        # the entire point of long-polling in the first place.
        messages = await asyncio.to_thread(
            telegram.get_updates, timeout=timeout, commit=False
        )
    except RuntimeError as exc:
        _log(f"telegram poll failed: {exc}")
        return []

    out: list[dict[str, Any]] = []
    for msg in messages:
        chat_id, text = msg["chat_id"], msg["text"]
        _log(f"✉ {msg['name']}: {text[:80]}")
        ok = True
        try:
            reply = await run_command(text, fake=fake)
        except Exception as exc:  # noqa: BLE001 - a broken command must not eat the message
            from .adapter import describe_error

            ok, reply = False, f"That command failed: {describe_error(exc)}"
        if reply is None:
            token = _CURRENT_LANE.set("chat")
            try:
                ok, reply = await _answer(
                    text, _CHAT_SESSION.format(chat_id=chat_id), fake=fake
                )
            finally:
                _CURRENT_LANE.reset(token)
        try:
            await asyncio.to_thread(telegram.send_message, reply, chat_id=chat_id)
        except RuntimeError as exc:
            _log(f"  reply failed: {exc}")
        # Committed whether or not the reply went out: the answer is logged in the
        # session transcript, and re-running a paid turn because Telegram blipped
        # would bill it twice. What at-least-once protects against is the process
        # dying BEFORE this line.
        telegram.commit_update(msg.get("update_id"))
        out.append({"chat_id": chat_id, "ok": ok})
    return out


# --- chat commands ---------------------------------------------------------------

#: A model-free chat command: ``(argument, fake) -> reply``.
CommandHandler = Callable[[str, bool], Awaitable[str]]

#: Registered commands: name -> (handler, one-line help). Modules that own a
#: command register it (reports, ideas, the autonomy switch), so this file never
#: needs to learn their internals.
_COMMANDS: dict[str, tuple[CommandHandler, str]] = {}


def register_command(name: str, handler: CommandHandler, help_line: str) -> None:
    _COMMANDS[name.lower().lstrip("/")] = (handler, help_line)


def _parse_command(text: str) -> tuple[str, str] | None:
    """``"/cancel s3"`` -> ``("cancel", "s3")``; None when it isn't a command.
    Telegram appends the bot's name in groups (``/status@my_bot``), so that goes."""
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    head, _, arg = raw.partition(" ")
    name = head[1:].split("@", 1)[0].lower()
    return (name, arg.strip()) if name else None


def help_text() -> str:
    lines = ["Send me anything and I'll research it. Commands:"]
    lines += [f"  /{name} — {line}" for name, (_h, line) in sorted(_COMMANDS.items())
              if name not in ("start", "help")]
    lines.append("Ask me to 'analyse NVDA earnings tomorrow 9am' and I'll schedule it.")
    return "\n".join(lines)


async def run_command(text: str, fake: bool = False) -> str | None:
    """Answer a slash command, or None when ``text`` is a prompt for the model.

    An unknown command gets the help text, not a model turn: "/stauts" is a typo,
    and paying for the model to guess what it meant is the wrong trade.
    """
    parsed = _parse_command(text)
    if parsed is None:
        return None
    name, arg = parsed
    entry = _COMMANDS.get(name)
    if entry is None:
        return f"Unknown command /{name}.\n\n{help_text()}"
    return await entry[0](arg, fake)


async def _cmd_help(_arg: str, _fake: bool) -> str:
    return help_text()


async def _cmd_tasks(_arg: str, _fake: bool) -> str:
    return tasks.list_scheduled_tasks()


async def _cmd_cancel(arg: str, _fake: bool) -> str:
    return tasks.cancel_scheduled_task(arg) if arg else "Usage: /cancel <id> (see /tasks)"


async def _cmd_status(_arg: str, _fake: bool) -> str:
    return format_status(service_status())


register_command("start", _cmd_help, "what I can do")
register_command("help", _cmd_help, "what I can do")
register_command("tasks", _cmd_tasks, "what's scheduled")
register_command("schedule", _cmd_tasks, "what's scheduled")
register_command("cancel", _cmd_cancel, "drop a scheduled task: /cancel <id>")
register_command("status", _cmd_status, "is the service healthy, what's next")


async def watch(
    interval: float = 60.0, fake: bool = False, iterations: int | None = None
) -> None:
    """Run due tasks (and drain the inbox) forever, every ``interval`` seconds.

    A thin loop over ``run_due`` on purpose — the moment the daemon path grows its
    own scheduling logic, the cron path and the loop path start disagreeing about
    what "due" means. ``iterations`` bounds the loop for tests.
    """
    _log(f"watching for due tasks every {interval:g}s (Ctrl-C to stop)")
    count = 0
    while iterations is None or count < iterations:
        count += 1
        try:
            await run_due(fake=fake)
            # Long-poll the inbox for most of the interval: it costs one idle HTTP
            # request and makes a phone reply feel immediate instead of waiting out
            # the tick.
            waited = 0.0
            from . import telegram

            if telegram.inbound_enabled():
                poll = max(1, min(int(interval), telegram.POLL_TIMEOUT))
                await poll_inbox(fake=fake, timeout=poll)
                waited = float(poll)
            if interval > waited:
                await asyncio.sleep(interval - waited)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad tick must not stop the watch
            from .adapter import describe_error

            _log(f"tick failed: {describe_error(exc)}")
            await asyncio.sleep(interval)


# --- the always-on service ---------------------------------------------------------
#
# `serve` is `watch` grown up: the same `run_due` pass, but next to it — not in
# series with it — a loop answering the phone and a heartbeat proving the process
# is responsive. `watch` ran the pass and then the long poll one after the other,
# so a ten-minute monthly review left a phone message unanswered for ten minutes.

#: Extra loops other modules run inside the service (the event watchers): name ->
#: ``async (stop, fake) -> None``, returning when ``stop`` is set.
ServiceLoop = Callable[[asyncio.Event, bool], Awaitable[None]]
_SERVICE_LOOPS: dict[str, ServiceLoop] = {}


def register_service_loop(name: str, loop: ServiceLoop) -> None:
    _SERVICE_LOOPS[name] = loop


#: In-process view of the running service, for `/status`. Mirrored to disk
#: (``service-state.json``) so `--status` from another process — `docker exec`,
#: the container health check — sees the same thing.
_service: dict[str, Any] = {}


def _state_file():
    from .storage import state_file

    return state_file("service-state.json")


def _save_state() -> None:
    from .storage import write_private
    import json

    try:
        write_private(_state_file(), json.dumps(_service, default=str), prefix=".svc-")
    except OSError:
        pass  # a missing status file costs a stale /status, not the service


def _mark(loop: str, **fields: Any) -> None:
    entry = _service.setdefault("loops", {}).setdefault(loop, {})
    entry.update(fields)


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    """Sleep, but wake the moment ``stop`` is set, so shutdown isn't held up by a
    minute-long idle wait."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=max(0.0, seconds))
    except asyncio.TimeoutError:
        pass


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


async def _jobs_loop(stop: asyncio.Event, fake: bool, interval: float) -> None:
    while not stop.is_set():
        _service["jobs_started"] = time.time()
        try:
            await run_due(fake=fake, stop=stop)
            _mark("jobs", last_ok=datetime.now().isoformat(timespec="seconds"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad tick must not stop the service
            from .adapter import describe_error

            _log(f"job tick failed: {describe_error(exc)}")
            _mark("jobs", last_error=describe_error(exc))
        await _sleep_or_stop(stop, interval)


async def _inbox_loop(stop: asyncio.Event, fake: bool) -> None:
    from . import telegram

    while not stop.is_set():
        if not telegram.inbound_enabled():
            # Checked each round rather than once: nothing to poll is a state, not
            # an error, and it costs one env read a minute.
            await _sleep_or_stop(stop, 60.0)
            continue
        try:
            await poll_inbox(fake=fake, timeout=telegram.POLL_TIMEOUT)
            _mark("inbox", last_ok=datetime.now().isoformat(timespec="seconds"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            from .adapter import describe_error

            _log(f"inbox poll failed: {describe_error(exc)}")
            _mark("inbox", last_error=describe_error(exc))
            await _sleep_or_stop(stop, 10.0)


def _ping_healthcheck() -> None:
    """GET the dead-man URL (``FRA_HEALTHCHECK_URL``). Best-effort: an outage at the
    check service must not look like — or cause — an outage here."""
    import urllib.request

    url = (os.environ.get("FRA_HEALTHCHECK_URL") or "").strip()
    if not url:
        return
    try:
        with urllib.request.urlopen(url, timeout=10):  # noqa: S310 - operator-set URL
            pass
    except Exception:  # noqa: BLE001
        pass


async def _heartbeat_loop(stop: asyncio.Event, fake: bool) -> None:
    """Prove the service is alive, to three audiences: the watchdog thread (the
    event loop is responsive), `--status --check` (the state file is fresh), and
    the outside dead-man check (pinged only while the job loop is healthy too)."""
    last_ping = 0.0
    every = _env_float("FRA_HEALTHCHECK_EVERY", 300.0)
    while not stop.is_set():
        now = time.time()
        _service["alive"] = now
        _save_state()
        if now - last_ping >= every and _jobs_fresh(now):
            last_ping = now
            await asyncio.to_thread(_ping_healthcheck)
        await _sleep_or_stop(stop, 15.0)


def _jobs_fresh(now: float | None = None) -> bool:
    """Whether the job loop has started a pass recently enough. A pass may
    legitimately run for as long as its longest job, so the allowance is the job
    timeout plus a few intervals — past that, the loop is stuck, not busy."""
    started = _service.get("jobs_started")
    if started is None:
        return True  # not yet ticked: starting up, not stuck
    allowance = job_timeout_seconds() + 5 * 60.0
    return ((now or time.time()) - float(started)) <= allowance


def watchdog_minutes() -> float:
    """``FRA_WATCHDOG_MINUTES`` (default 10; 0 disables)."""
    return _env_float("FRA_WATCHDOG_MINUTES", 10.0)


class _Watchdog(threading.Thread):
    """Exit the process when the service has hung, so the supervisor restarts it.

    A THREAD, not a coroutine, because the failure it exists for is the event loop
    itself being stuck — a synchronous call that never returns blocks every
    coroutine, including one that would have noticed. systemd (or launchd) sees
    the exit and starts a fresh process; `os._exit` because a hung loop would never
    finish a clean shutdown anyway.
    """

    def __init__(self, limit_seconds: float) -> None:
        super().__init__(daemon=True, name="fra-watchdog")
        self.limit = limit_seconds
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(15.0):
            why = self.check(time.time())
            if why:
                self._die(why)

    def check(self, now: float) -> str:
        """Why the service should be restarted now, or "" if it is fine."""
        alive = float(_service.get("alive") or now)
        if now - alive > self.limit:
            return f"the event loop has not responded for {now - alive:.0f}s"
        if not _jobs_fresh(now):
            return "the job loop is stuck"
        return ""

    def _die(self, why: str) -> None:
        _log(f"watchdog: {why} — exiting so the supervisor restarts the service")
        os._exit(70)


async def _supervise(name: str, loop: Callable[[], Awaitable[None]], stop: asyncio.Event) -> None:
    """Run one loop for the life of the service, restarting it if it raises.

    Each loop already survives a bad iteration; this is for the one that escapes —
    a bug in the loop itself. Backoff doubles to five minutes, so a loop that fails
    on every start costs a log line every five minutes, not a hot spin.
    """
    backoff = _env_float("FRA_LOOP_BACKOFF", 5.0)
    while not stop.is_set():
        try:
            await loop()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            from .adapter import describe_error

            entry = _service.setdefault("loops", {}).setdefault(name, {})
            entry["restarts"] = int(entry.get("restarts") or 0) + 1
            entry["last_error"] = describe_error(exc)
            _log(f"loop {name} crashed ({describe_error(exc)}); restarting in {backoff:.0f}s")
            await _sleep_or_stop(stop, backoff)
            backoff = min(backoff * 2, 300.0)


def stop_grace_seconds() -> float:
    """How long a shutdown waits for in-flight work (``FRA_STOP_GRACE``, 60s)."""
    return _env_float("FRA_STOP_GRACE", 60.0)


async def serve(
    fake: bool = False,
    job_interval: float = 60.0,
    stop: asyncio.Event | None = None,
    watchdog: bool = True,
) -> None:
    """Run the always-on service until ``stop`` is set (SIGTERM/SIGINT set it).

    Loops: ``jobs`` (``run_due`` every ``job_interval``), ``inbox`` (the Telegram
    long poll), ``heartbeat``, plus whatever is in ``_SERVICE_LOOPS``. Model runs
    share two lanes — chat and background — so the phone never waits on a report.

    Shutdown is graceful: stop claiming work, give in-flight turns
    ``stop_grace_seconds()`` to finish, then cancel what's left (a cancelled task
    hands its claim back; an unanswered message was never committed, so both are
    picked up again after the restart).
    """
    global _lanes
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    handled: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
            handled.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass  # not the main thread, or no signal support
    _lanes = {"chat": asyncio.Semaphore(1), "background": asyncio.Semaphore(background_runs())}
    _service.clear()
    _service.update({"pid": os.getpid(), "started": time.time(), "alive": time.time(),
                     "loops": {}})
    _save_state()
    loops: dict[str, Callable[[], Awaitable[None]]] = {
        "jobs": lambda: _jobs_loop(stop, fake, job_interval),
        "inbox": lambda: _inbox_loop(stop, fake),
        "heartbeat": lambda: _heartbeat_loop(stop, fake),
    }
    for name, extra in _SERVICE_LOOPS.items():
        loops[name] = (lambda e=extra: e(stop, fake))
    dog = _Watchdog(watchdog_minutes() * 60.0) if watchdog and watchdog_minutes() > 0 else None
    if dog is not None:
        dog.start()
    _log(f"serving: {', '.join(loops)} (Ctrl-C or SIGTERM to stop)")
    running = [asyncio.create_task(_supervise(n, fn, stop), name=f"fra-{n}")
               for n, fn in loops.items()]
    try:
        await stop.wait()
        _log(f"stopping: letting in-flight work finish (up to {stop_grace_seconds():.0f}s)")
        _done, pending = await asyncio.wait(running, timeout=stop_grace_seconds())
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for t in running:
            t.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        if dog is not None:
            dog.stop()
        for sig in handled:
            loop.remove_signal_handler(sig)
        _lanes = None
        _service["stopped"] = time.time()
        _save_state()
        _log("stopped")


# --- status ------------------------------------------------------------------------


def service_status() -> dict[str, Any]:
    """What `/status` and `--status` report, from whichever process asks.

    The serving process answers from memory; any other one (`docker exec`, the
    container health check) reads the state file the service keeps fresh.
    """
    from .storage import read_json
    from . import telegram

    state = dict(_service) if _service else read_json(_state_file(), {})
    now = time.time()
    tick = tasks.last_tick()
    alive = state.get("alive")
    stopped = state.get("stopped")
    serving = bool(alive) and not (stopped and float(stopped) >= float(alive))
    fresh_s = _env_float("FRA_HEALTH_MAX_SECONDS", 180.0)
    return {
        "serving": serving,
        "alive_age": (now - float(alive)) if alive else None,
        "healthy": serving and alive is not None and (now - float(alive)) <= fresh_s,
        "started": state.get("started"),
        "loops": state.get("loops") or {},
        "last_tick": tick.isoformat() if tick else None,
        "runner_live": tasks.runner_is_live(),
        "pending": tasks.pending_tasks()[:5],
        "inbound": telegram.inbound_enabled(),
        "channels": [c.key for c in channels.active_channels()],
        "extras": {name: line for name, fn in _STATUS_EXTRAS.items()
                   if (line := _safe_call(fn))},
    }


#: Lines other modules add to the status (spend today, autonomy paused…):
#: name -> ``() -> str``.
_STATUS_EXTRAS: dict[str, Callable[[], str]] = {}


def register_status_line(name: str, fn: Callable[[], str]) -> None:
    _STATUS_EXTRAS[name] = fn


def _safe_call(fn: Callable[[], str]) -> str:
    try:
        return fn() or ""
    except Exception:  # noqa: BLE001 - a status line must not break /status
        return ""


def format_status(st: dict[str, Any]) -> str:
    """The human version of ``service_status``."""
    lines: list[str] = []
    if st.get("serving"):
        up = ""
        if st.get("started"):
            secs = int(time.time() - float(st["started"]))
            up = f" · up {secs // 86400}d {secs % 86400 // 3600}h {secs % 3600 // 60}m"
        health = "healthy" if st.get("healthy") else (
            f"NOT RESPONDING (last heard {st['alive_age']:.0f}s ago)"
            if st.get("alive_age") is not None else "unknown")
        lines.append(f"🟢 service {health}{up}")
    else:
        lines.append("⚪ the always-on service is not running here")
    for name, info in sorted((st.get("loops") or {}).items()):
        bits = []
        if info.get("last_ok"):
            bits.append(f"ok {info['last_ok']}")
        if info.get("restarts"):
            bits.append(f"{info['restarts']} restart(s)")
        if info.get("last_error"):
            bits.append(f"last error: {str(info['last_error'])[:80]}")
        if bits:
            lines.append(f"  {name}: " + " · ".join(bits))
    tick = st.get("last_tick")
    lines.append(f"last job pass: {tick[:16].replace('T', ' ') + ' UTC' if tick else 'never'}")
    pending = st.get("pending") or []
    if pending:
        lines.append("next up:")
        lines += [f"  {tasks.describe(t)}" for t in pending]
    else:
        lines.append("nothing scheduled")
    lines.append("phone chat: " + ("on" if st.get("inbound") else
                                   "off (set TELEGRAM_ALLOWED_CHAT_IDS)"))
    lines.append("delivery: " + (", ".join(st.get("channels") or []) or "none configured"))
    for _name, line in sorted((st.get("extras") or {}).items()):
        lines.append(line)
    return "\n".join(lines)
