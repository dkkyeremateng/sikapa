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

Both also drain the Telegram inbox when inbound is enabled, so the phone path needs
no separate process.

**A scheduled run is a normal turn.** It goes through ``adapter.run_turn`` with its
own session id, so it gets the same tools, the same read-only broker boundary, the
same tracing and the same long-term memory as anything typed into the TUI. The only
differences are that nothing streams to a terminal and the final answer is
delivered rather than printed.
"""

from __future__ import annotations

from typing import Any
import asyncio
import sys
from datetime import datetime

from . import channels, tasks

#: Prefix on every delivered message, so a phone notification is self-identifying
#: rather than an anonymous wall of analysis.
_HEADER = "🤖 {label}"

#: Inbound messages run under one session per chat, so a conversation on the phone
#: keeps its context across messages the way the TUI does.
_CHAT_SESSION = "telegram-{chat_id}"

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
    """
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


async def run_task(task: dict[str, Any], fake: bool = False) -> dict[str, Any]:
    """Run one claimed task, deliver its answer, and record the outcome.

    Delivery failure does NOT fail the task. The work is done and the model call is
    spent; marking it failed would re-run the whole analysis on the next tick
    because a notification API was down. The failure is logged and the answer stays
    in the task's stored result.
    """
    tid = str(task.get("id"))
    prompt = str(task.get("prompt", ""))
    session = str(task.get("session") or f"task-{tid}")
    _log(f"▶ task {tid}: {prompt[:80]}")
    try:
        # The stored prompt stays clean (it's what `--tasks` shows and what the user
        # wrote); the framing is added only on the way into the model.
        ok, answer = await _answer(f"{_TASK_PREAMBLE}\n\n{prompt}", session, fake=fake)
    except asyncio.CancelledError:
        # Ctrl-C or a killed watcher: hand the task back rather than leaving it
        # stuck in "running", which nothing would ever claim again.
        tasks.release(tid)
        raise
    except Exception as exc:  # noqa: BLE001 - a task must not take down the tick
        from .adapter import describe_error

        ok, answer = False, describe_error(exc)

    # A reply that isn't an answer counts as a failure, so the task retries with
    # the framing preamble rather than delivering a shrug as the finished work.
    reason = _non_answer_reason(answer) if ok else ""
    if reason:
        ok = False
        _log(f"  not an answer: {reason}")

    # Record BEFORE delivering: whether this is the last attempt decides whether
    # the user hears about the failure now or after the retries are exhausted.
    record = tasks.record_result(tid, ok, reason or answer)
    if record is not None and not ok and record.get("status") == "pending":
        # Silent on purpose: three phone notifications for one task that is still
        # being retried is noise, and the outcome is not known yet.
        _log(f"  ✗ task {tid}; retrying next tick "
             f"({record.get('attempts')}/{tasks.MAX_ATTEMPTS})")
        return {"id": tid, "ok": False, "delivered": [], "failed": [], "answer": answer}

    label = f"task {tid}" + ("" if ok else " (failed)")
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


async def run_due(
    now: datetime | None = None, fake: bool = False, limit: int | None = None
) -> list[dict[str, Any]]:
    """Claim and run everything due. Returns one result per task run.

    Sequential, not concurrent: each task is a full tool-using turn, and running
    several at once multiplies both the token spend and the load on the same rate-
    limited market-data endpoints. A backlog drains over consecutive ticks.
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
        try:
            results.append(await run_task(task, fake=fake))
        except asyncio.CancelledError:
            raise
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

    ``timeout`` is the long-poll window: 0 for a cron tick (take what is waiting
    and return), or a few seconds in a watch loop.

    Two commands are handled without the model, because they must work even when
    no credential is configured and must never cost a model call: ``/tasks`` lists
    what is scheduled, ``/cancel ID`` removes one. Anything else is a prompt.
    """
    from . import telegram

    if not telegram.inbound_enabled():
        return []
    try:
        # A long poll parks on a socket for up to POLL_TIMEOUT seconds. Run on the
        # loop thread it would block every other coroutine on it — including the
        # `--watch` tick and any in-flight turn — for the whole window, which is
        # the entire point of long-polling in the first place.
        messages = await asyncio.to_thread(telegram.get_updates, timeout=timeout)
    except RuntimeError as exc:
        _log(f"telegram poll failed: {exc}")
        return []

    out: list[dict[str, Any]] = []
    for msg in messages:
        chat_id, text = msg["chat_id"], msg["text"]
        _log(f"✉ {msg['name']}: {text[:80]}")
        reply = _builtin_command(text)
        ok = True
        if reply is None:
            ok, reply = await _answer(
                text, _CHAT_SESSION.format(chat_id=chat_id), fake=fake
            )
        try:
            await asyncio.to_thread(telegram.send_message, reply, chat_id=chat_id)
        except RuntimeError as exc:
            _log(f"  reply failed: {exc}")
        out.append({"chat_id": chat_id, "ok": ok})
    return out


def _builtin_command(text: str) -> str | None:
    """Handle the model-free chat commands, or None to treat the text as a prompt."""
    raw = (text or "").strip()
    low = raw.lower()
    if low in ("/tasks", "/schedule", "/scheduled"):
        return tasks.list_scheduled_tasks()
    if low.startswith("/cancel"):
        arg = raw[len("/cancel"):].strip()
        return tasks.cancel_scheduled_task(arg) if arg else "Usage: /cancel <id> (see /tasks)"
    if low in ("/help", "/start"):
        return (
            "Send me anything and I'll research it. Commands:\n"
            "  /tasks — what's scheduled\n"
            "  /cancel <id> — drop a scheduled task\n"
            "Ask me to 'analyse NVDA earnings tomorrow 9am' and I'll schedule it."
        )
    return None


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
