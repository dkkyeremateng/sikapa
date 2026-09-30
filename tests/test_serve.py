"""The always-on service: loops side by side, crash isolation, graceful stop.

Offline and fast: the model turn is replaced below `_answer` (so the lanes it
enforces are still exercised), Telegram is a scripted inbox, and every interval
is shrunk to milliseconds.
"""

import asyncio
import json
import os
import signal
import time

import pytest

from financial_research_assistant import catalog, channels, scheduler, tasks, telegram

_LONG = " — a complete answer with figures, sources and a verdict."


@pytest.fixture
def phone(monkeypatch):
    """An allowlisted bot whose inbox yields the scripted messages once each."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "999")
    inbox: list[dict] = []
    replies: list[tuple[str, str]] = []

    def get_updates(timeout=0, commit=True):
        time.sleep(0.005)  # a real long poll blocks; never hot-spin the loop
        out, inbox[:] = list(inbox), []
        return out

    monkeypatch.setattr(telegram, "get_updates", get_updates)
    monkeypatch.setattr(
        telegram, "send_message",
        lambda text, chat_id="": bool(replies.append((chat_id, text)) or True),
    )
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (["test"], []))
    return {"inbox": inbox, "replies": replies}


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setenv("FRA_LOOP_BACKOFF", "0.01")
    monkeypatch.setenv("FRA_STOP_GRACE", "2")
    yield
    scheduler._lanes = None


async def _until(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def _run(coro):
    return asyncio.run(coro)


def test_a_phone_message_is_answered_while_a_long_job_is_running(monkeypatch, phone):
    """The job below cannot finish until the chat reply has gone out. If chat and
    background work shared one lane — or one loop, as `watch` does — this would
    deadlock instead of passing."""
    tasks.add_task("Write the monthly review", "+0m")
    chat_done = asyncio.Event()

    async def answer(prompt, session_id, fake):
        if session_id.startswith("telegram-"):
            chat_done.set()
            return True, "NVDA is up" + _LONG
        await chat_done.wait()
        return True, "The review" + _LONG

    monkeypatch.setattr(scheduler, "_answer_unlocked", answer)

    async def main():
        stop = asyncio.Event()
        server = asyncio.create_task(
            scheduler.serve(job_interval=0.02, stop=stop, watchdog=False)
        )
        await _until(lambda: tasks.load_tasks()[0]["status"] == "running")
        phone["inbox"].append(
            {"chat_id": "999", "text": "how is NVDA?", "name": "me", "update_id": 7}
        )
        await _until(lambda: tasks.load_tasks()[0]["status"] == "done")
        stop.set()
        await server

    _run(main())
    assert phone["replies"] == [("999", "NVDA is up" + _LONG)]


def test_every_unattended_turn_runs_without_the_path_reading_tools(monkeypatch, phone):
    seen: list[bool] = []

    async def answer(prompt, session_id, fake):
        seen.append(catalog.is_unattended())
        return True, "fine" + _LONG

    monkeypatch.setattr(scheduler, "_answer_unlocked", answer)
    tasks.add_task("brief me", "+0m")
    _run(scheduler.run_due())
    phone["inbox"].append({"chat_id": "999", "text": "hi", "name": "me", "update_id": 1})
    _run(scheduler.poll_inbox())
    assert seen == [True, True]
    assert catalog.is_unattended() is False, "and only inside the turn"


def test_the_unattended_toolset_drops_exactly_the_path_readers():
    attended = {catalog.tool_name(t) for t in catalog.active_tools(frozenset())}
    with catalog.unattended():
        unattended = {catalog.tool_name(t) for t in catalog.active_tools(frozenset())}
    assert attended - unattended == set(catalog.UNATTENDED_DENY)
    assert catalog.UNATTENDED_DENY <= attended, "the deny list names real tools"


def test_a_loop_that_crashes_is_restarted_and_the_others_keep_going(monkeypatch):
    starts: list[float] = []
    passes: list[float] = []

    async def boom(stop, fake):
        starts.append(time.time())
        raise RuntimeError("bug in a watcher")

    async def counted_run_due(**_kw):
        passes.append(time.time())
        return []

    monkeypatch.setitem(scheduler._SERVICE_LOOPS, "boom", boom)
    monkeypatch.setattr(scheduler, "run_due", counted_run_due)

    async def main():
        stop = asyncio.Event()
        server = asyncio.create_task(
            scheduler.serve(job_interval=0.01, stop=stop, watchdog=False)
        )
        await _until(lambda: len(starts) >= 3 and len(passes) >= 3)
        stop.set()
        await server

    _run(main())
    assert scheduler._service["loops"]["boom"]["restarts"] >= 2
    assert "bug in a watcher" in scheduler._service["loops"]["boom"]["last_error"]


def test_sigterm_stops_the_service_cleanly(monkeypatch):
    monkeypatch.setattr(scheduler, "run_due", lambda **_kw: asyncio.sleep(0, []))

    async def main():
        asyncio.get_running_loop().call_later(0.1, os.kill, os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(scheduler.serve(job_interval=0.01, watchdog=False), 5)

    _run(main())
    assert scheduler._service.get("stopped")
    assert scheduler.service_status()["serving"] is False


def test_a_stop_hands_back_claimed_tasks_that_had_not_started(monkeypatch):
    ran: list[str] = []

    async def answer(prompt, session_id, fake):
        ran.append(session_id)
        return True, "x" + _LONG

    monkeypatch.setattr(scheduler, "_answer_unlocked", answer)
    tasks.add_task("one", "+0m")
    tasks.add_task("two", "+0m")
    stop = asyncio.Event()
    stop.set()
    _run(scheduler.run_due(stop=stop))
    assert ran == []
    assert [t["status"] for t in tasks.load_tasks()] == ["pending", "pending"]
    assert all(not t.get("attempts") for t in tasks.load_tasks()), "not a failure"


def test_a_turn_still_running_when_the_grace_runs_out_is_handed_back(monkeypatch):
    monkeypatch.setenv("FRA_STOP_GRACE", "0.05")

    async def forever(prompt, session_id, fake):
        await asyncio.sleep(3600)
        return True, "never"

    monkeypatch.setattr(scheduler, "_answer_unlocked", forever)
    tasks.add_task("slow", "+0m")

    async def main():
        stop = asyncio.Event()
        server = asyncio.create_task(
            scheduler.serve(job_interval=0.01, stop=stop, watchdog=False)
        )
        await _until(lambda: tasks.load_tasks()[0]["status"] == "running")
        stop.set()
        await asyncio.wait_for(server, 5)

    _run(main())
    t = tasks.load_tasks()[0]
    assert t["status"] == "pending" and not t.get("attempts")


def test_a_job_that_never_returns_is_timed_out_and_counted(monkeypatch):
    monkeypatch.setattr(scheduler, "job_timeout_seconds", lambda: 0.05)

    async def forever(prompt, session_id, fake):
        await asyncio.sleep(3600)
        return True, "never"

    monkeypatch.setattr(scheduler, "_answer_unlocked", forever)
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (["test"], []))
    tasks.add_task("stuck", "+0m")
    results = _run(scheduler.run_due())
    assert results[0]["ok"] is False and "timed out" in results[0]["answer"]
    t = tasks.load_tasks()[0]
    # Counted, so the retry limit parks it instead of re-running it every tick.
    assert t["attempts"] == 1 and t["status"] == "pending"


# --- commands ---------------------------------------------------------------------


def test_commands_never_reach_the_model(monkeypatch):
    async def no_model(*a, **k):
        raise AssertionError("a command must not start a turn")

    monkeypatch.setattr(scheduler, "_answer_unlocked", no_model)
    status = _run(scheduler.run_command("/status"))
    assert "service" in status and "phone chat" in status
    assert "/status" in _run(scheduler.run_command("/help"))
    assert _run(scheduler.run_command("/status@my_bot")) == status
    typo = _run(scheduler.run_command("/stauts"))
    assert typo.startswith("Unknown command /stauts") and "/tasks" in typo
    assert _run(scheduler.run_command("how is NVDA?")) is None


# --- liveness ---------------------------------------------------------------------


def test_the_watchdog_fires_on_a_frozen_loop_and_a_stuck_job(monkeypatch):
    dog = scheduler._Watchdog(limit_seconds=600)
    now = time.time()
    scheduler._service.clear()
    scheduler._service.update({"alive": now - 30, "jobs_started": now - 60})
    assert dog.check(now) == ""
    scheduler._service["alive"] = now - 900
    assert "not responded" in dog.check(now)
    scheduler._service["alive"] = now
    monkeypatch.setattr(scheduler, "job_timeout_seconds", lambda: 60.0)
    scheduler._service["jobs_started"] = now - 3600
    assert dog.check(now) == "the job loop is stuck"


def test_status_check_reflects_the_state_file_from_another_process(monkeypatch):
    """`--status --check` runs in a different process from the service (the
    container health check), so it can only go by the file the service keeps."""
    from financial_research_assistant.storage import write_private

    scheduler._service.clear()
    assert scheduler.service_status()["healthy"] is False  # never started here
    write_private(scheduler._state_file(), json.dumps({"alive": time.time(), "started": time.time()}))
    assert scheduler.service_status()["healthy"] is True
    write_private(scheduler._state_file(), json.dumps({"alive": time.time() - 3600}))
    st = scheduler.service_status()
    assert st["serving"] is True and st["healthy"] is False
    assert "NOT RESPONDING" in scheduler.format_status(st)


def test_the_dead_man_ping_is_sent_only_while_the_jobs_are_healthy(monkeypatch):
    pings: list[str] = []
    monkeypatch.setenv("FRA_HEALTHCHECK_URL", "https://hc.example/ping/abc")
    monkeypatch.setenv("FRA_HEALTHCHECK_EVERY", "0")
    monkeypatch.setattr(scheduler, "_ping_healthcheck", lambda: pings.append("ping"))
    monkeypatch.setattr(scheduler, "job_timeout_seconds", lambda: 60.0)

    async def one_beat(jobs_started):
        scheduler._service.clear()
        scheduler._service["jobs_started"] = jobs_started
        stop = asyncio.Event()
        beat = asyncio.create_task(scheduler._heartbeat_loop(stop, False))
        await asyncio.sleep(0.05)
        stop.set()
        await beat

    _run(one_beat(time.time()))
    assert pings == ["ping"]
    _run(one_beat(time.time() - 3600))  # the job loop is stuck: stay silent
    assert pings == ["ping"], "the outside check must see the silence and alert"
