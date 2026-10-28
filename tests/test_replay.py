"""A request that runs twice acts once.

A Telegram message is answered again after a restart mid-answer, and a scheduled
task is retried after a failure or reclaimed after a crash. Each rerun is a fresh
model turn, so these tests drive the real runner with a stand-in turn that calls
the REAL tools, through the same LangChain tool objects the graph binds — a sync
tool runs in a worker thread there, and the request key has to survive that hop.

Offline: no model, no network; prices and report builds are stubbed.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from financial_research_assistant import (
    alerts, catalog, journal, periodic, recommend, replay, scheduler, tasks,
)


def _tool(name):
    """The catalog entry as the agent runs it: a plain function is wrapped as a
    StructuredTool (what ``create_agent`` does), whose async call runs the sync
    function in a worker thread."""
    from langchain_core.tools import BaseTool, StructuredTool

    t = next(t for t in catalog.TOOLS if catalog.tool_name(t) == name)
    return t if isinstance(t, BaseTool) else StructuredTool.from_function(t)


async def _call(name, **args):
    """Invoke a tool the way the graph's tool node does."""
    return await _tool(name).ainvoke(args)


@pytest.fixture(autouse=True)
def _no_delivery(monkeypatch):
    from financial_research_assistant import channels

    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (["test"], []))


@pytest.fixture
def priced(monkeypatch):
    """A close for every recent day, so a call can be recorded and priced."""
    import financial_research_assistant.tools as t
    from financial_research_assistant.pointintime import as_of_series

    today = date.today()
    series = [((today - timedelta(days=i)).isoformat(), 100.0) for i in range(30, -1, -1)]

    def fake(sym, days, strict=False, as_of=None, **_kw):
        return as_of_series(series, as_of, days)

    monkeypatch.setattr(t, "_fetch_daily", fake)


# --- through the runner ----------------------------------------------------------


def test_a_retried_task_does_not_schedule_its_follow_up_twice(monkeypatch):
    """The first attempt schedules a follow-up and then fails; the retry words the
    follow-up differently and schedules it again. One follow-up, not two."""
    attempts = []

    async def turn(prompt, session_id, fake=False):
        n = len(attempts) + 1
        out = await _call("schedule_task", prompt=f"check NVDA earnings (attempt {n})",
                          when="+1d", repeat="daily")
        attempts.append(out)
        if n == 1:
            return False, "provider error mid-turn"
        return True, "Done — the follow-up is booked, with figures and a verdict here."

    monkeypatch.setattr(scheduler, "_answer", turn)
    tasks.add_task("plan my NVDA follow-ups", "+0m")
    asyncio.run(scheduler.run_due())  # fails, stays due for a retry
    asyncio.run(scheduler.run_due())  # the retry: same occurrence, same key

    assert len(attempts) == 2
    follow_ups = [t for t in tasks.load_tasks() if "NVDA earnings" in t["prompt"]]
    assert [t["prompt"] for t in follow_ups] == ["check NVDA earnings (attempt 1)"]
    assert "Already done" in attempts[1] and "Scheduled [" in attempts[1]


def test_the_next_occurrence_of_a_recurring_task_acts_again(monkeypatch):
    """The key is per occurrence, not per task: tomorrow's run of a daily task is a
    new request, and what it schedules is new work."""
    async def turn(prompt, session_id, fake=False):
        await _call("schedule_task", prompt="one-off reminder", when="+3h")
        return True, "Booked the reminder — figures, sources and a verdict follow."

    monkeypatch.setattr(scheduler, "_answer", turn)
    parent = tasks.add_task("daily planner", "+0m", repeat="daily")
    asyncio.run(scheduler.run_due())
    # Bring the next occurrence forward to now, as tomorrow would.
    items = tasks.load_tasks()
    for t in items:
        if t["id"] == parent["id"]:
            assert t["due"] != parent["due"], "success moves the task on"
            t["due"] = datetime.now(timezone.utc).isoformat()
    tasks.save_tasks(items)
    asyncio.run(scheduler.run_due())

    assert len([t for t in tasks.load_tasks() if t["prompt"] == "one-off reminder"]) == 2


@pytest.fixture
def inbox(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "999")
    from financial_research_assistant import telegram

    monkeypatch.setattr(telegram, "send_message", lambda text, chat_id="": True)
    monkeypatch.setattr(telegram, "commit_update", lambda update_id: None)
    pending = []
    monkeypatch.setattr(telegram, "get_updates", lambda timeout=0, **_: list(pending))
    return pending


def test_a_phone_message_answered_twice_logs_its_call_once(monkeypatch, inbox, priced):
    """At-least-once delivery answers an update again after a restart. The second
    answer rewords the thesis; the journal still holds one call. A DIFFERENT
    message making the same call is new work and is logged."""
    async def turn(prompt, session_id, fake=False):
        await _call("record_thesis", symbol="nvda", verdict="bullish",
                    thesis=f"datacenter demand ({len(journal.load_entries())})")
        return True, "Logged the call — reasoning, figures and the horizon above."

    monkeypatch.setattr(scheduler, "_answer", turn)
    inbox.append({"chat_id": "999", "text": "log NVDA bullish", "name": "me", "update_id": 41})
    asyncio.run(scheduler.poll_inbox())
    asyncio.run(scheduler.poll_inbox())  # same update again: the restart case
    assert len(journal.load_entries()) == 1

    inbox[0] = {**inbox[0], "update_id": 42}
    asyncio.run(scheduler.poll_inbox())
    assert len(journal.load_entries()) == 2


# --- the tools ---------------------------------------------------------------------


def test_outside_a_request_every_call_acts():
    """The TUI has no request: typing the same thing twice means it twice."""
    tasks.schedule_task("remind me", "+1h")
    tasks.schedule_task("remind me", "+1h")
    assert len(tasks.load_tasks()) == 2


def test_a_refused_act_is_not_recorded_so_the_rerun_can_try(monkeypatch, priced):
    """No price data means no call was logged. The rerun, with prices, must log it
    rather than be told it already happened."""
    import financial_research_assistant.tools as t

    working = t._fetch_daily
    monkeypatch.setattr(t, "_fetch_daily", lambda *a, **k: [])
    with replay.request("telegram:7"):
        first = journal.record_thesis("AMD", "bearish", "margins")
    assert journal.load_entries() == [] and "Already done" not in first

    monkeypatch.setattr(t, "_fetch_daily", working)
    with replay.request("telegram:7"):
        second = journal.record_thesis("AMD", "bearish", "margins, reworded")
    assert "Already done" not in second
    assert len(journal.load_entries()) == 1


def test_one_request_may_log_both_directions_on_different_tickers(priced):
    with replay.request("task:s1:2026-10-28T09:00:00+00:00"):
        journal.record_thesis("NVDA", "bullish", "a")
        journal.record_thesis("AMD", "bearish", "b")
        journal.record_thesis("NVDA", "bullish", "a again")
    assert sorted((e["symbol"], e["verdict"]) for e in journal.load_entries()) == [
        ("AMD", "bearish"), ("NVDA", "bullish")]


def test_two_follow_ups_in_one_attempt_are_both_scheduled():
    """Matched by position: the second call of an attempt is not the first."""
    with replay.request("telegram:9"):
        tasks.schedule_task("first", "+1h")
        tasks.schedule_task("second", "+2h")
    assert [t["prompt"] for t in tasks.load_tasks()] == ["first", "second"]


def test_an_identical_alert_is_refused_any_time(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERTS_FILE", str(tmp_path / "alerts.json"))
    alerts.add_alert("NVDA", "drop", 5)
    again = alerts.add_alert("nvda", "drop", 5.0)
    assert "already does that" in again
    alerts.add_alert("NVDA", "drop", 8)
    assert [(r["symbol"], r["value"]) for r in alerts.load_alerts()] == [("NVDA", 5.0), ("NVDA", 8.0)]


def test_a_report_is_sent_once_per_request(monkeypatch):
    sent = []

    async def report_now(kind, deliver=True, fake=False):
        sent.append((kind, deliver))
        return f"{kind} report"

    monkeypatch.setattr(periodic, "report_now", report_now)

    async def go():
        with replay.request("telegram:11"):
            await _call("periodic_report", kind="weekly")
            again = await _call("periodic_report", kind="weekly")
            await _call("periodic_report", kind="weekly", deliver=False)  # sends nothing
        return again

    again = asyncio.run(go())
    assert sent == [("weekly", True), ("weekly", False)]
    assert "Already done" in again


def test_ideas_run_once_per_sleeve_per_request(monkeypatch):
    runs = []

    async def run(mode, focus=""):
        runs.append(focus)
        return {"focus": focus}

    monkeypatch.setattr(recommend, "run", run)
    monkeypatch.setattr(recommend, "sheet",
                        lambda r: {"message": f"ideas for {r['focus']}", "markdown": ""})

    async def go():
        with replay.request("task:s4:2026-10-28T09:00:00+00:00"):
            await _call("recommend_ideas", focus="semis")
            await _call("recommend_ideas", focus="semiconductors, reworded")
            await _call("recommend_ideas", sleeve="bonds")

    asyncio.run(go())
    assert runs == ["semis", "bonds"]


def test_old_entries_are_dropped_from_the_ledger():
    old = datetime.now(timezone.utc) - timedelta(days=replay.KEEP_DAYS + 1)
    with replay.request("telegram:1"):
        replay._record(replay._entry_key("t", "x"), "old", now=old)
        replay._record(replay._entry_key("t", "y"), "new")
    data = replay.read_json(replay.ledger_file(), {})
    assert list(data) == ["telegram:1|t|y"]
