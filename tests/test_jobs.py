"""Scheduling for the always-on agent: zones, monthly repeats, jobs, the ledger.

Every date below is real and chosen for its edge: a US daylight-saving change, a
31st followed by a short month, a leap day, a month that starts on a weekend.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from financial_research_assistant import channels, jobs, market, scheduler, tasks

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@pytest.fixture
def host_zone():
    """Set the HOST's zone — the one a task without `tz` follows."""
    previous = os.environ.get("TZ")

    def use(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield use
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


def _freeze(monkeypatch, moment: datetime) -> None:
    monkeypatch.setattr(tasks, "now_utc", lambda: moment.astimezone(UTC))


# --- zones --------------------------------------------------------------------------


def test_a_task_pinned_to_new_york_follows_the_us_clock_from_a_zone_without_dst(
    host_zone, monkeypatch
):
    """Nairobi has no daylight saving. A daily job at 17:15 host-time would run at
    16:15 New York in the winter — before the close it exists to report on."""
    host_zone("Africa/Nairobi")
    due = datetime(2026, 10, 30, 17, 15, tzinfo=NY)  # Friday, EDT
    _freeze(monkeypatch, due + timedelta(minutes=1))
    nxt = tasks.next_due(due.astimezone(UTC), "daily", tz="America/New_York")
    assert nxt.astimezone(NY).hour == 17 and nxt.astimezone(NY).minute == 15
    assert (due.astimezone(UTC).hour, nxt.astimezone(UTC).hour) == (21, 21)
    # Two days on, across the 1 Nov change, it is still 17:15 in New York —
    # which is now 22:15 UTC.
    later = tasks.next_due(nxt, "daily", tz="America/New_York")
    _freeze(monkeypatch, later + timedelta(minutes=1))
    after = tasks.next_due(later, "daily", tz="America/New_York")
    assert after.astimezone(NY).date().isoformat() == "2026-11-02"
    assert after.astimezone(NY).hour == 17 and after.astimezone(UTC).hour == 22


def test_a_weekday_named_before_a_clock_change_lands_at_that_days_hour(host_zone):
    """`parse_when` added days to an aware 'now', which keeps TODAY's offset: on
    Friday 6 March, 'monday 9am' came out as 10:00 EDT — an hour late."""
    host_zone("America/New_York")
    friday = datetime(2026, 3, 6, 12, 0, tzinfo=NY)
    got = tasks.parse_when("monday 9am", now=friday.astimezone(UTC)).astimezone(NY)
    assert (got.month, got.day, got.hour) == (3, 9, 9)


def test_parse_when_reads_the_clock_of_the_zone_it_is_given():
    now = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)  # 08:00 in New York
    got = tasks.parse_when("16:45", now=now, tz="America/New_York")
    assert got.astimezone(NY).strftime("%Y-%m-%d %H:%M") == "2026-09-29 16:45"


def test_an_unknown_zone_is_refused_not_ignored():
    with pytest.raises(ValueError, match="unknown time zone"):
        tasks.add_task("x", "+1h", tz="Mars/Olympus_Mons")


def test_a_weekday_job_first_runs_on_a_weekday(monkeypatch):
    friday_evening = datetime(2026, 10, 2, 18, 0, tzinfo=NY)
    _freeze(monkeypatch, friday_evening)
    t = tasks.add_task("brief", "16:45", "weekdays", tz="America/New_York")
    first = datetime.fromisoformat(t["due"]).astimezone(NY)
    assert first.strftime("%a %H:%M") == "Mon 16:45"


# --- monthly ---------------------------------------------------------------------------


def test_monthly_keeps_its_day_through_a_short_month():
    """Jan 31 → Feb 28 → Mar 31. Stepping from the previous run instead would stay
    on the 28th for the rest of time after the first February."""
    t = {"repeat": "monthly", "tz": "America/New_York", "anchor_day": 31}
    jan = datetime(2030, 1, 31, 8, 0, tzinfo=NY)
    feb = tasks._next_for(t, jan.astimezone(UTC))
    mar = tasks._next_for(t, feb)
    assert feb.astimezone(NY).date().isoformat() == "2030-02-28"
    assert mar.astimezone(NY).date().isoformat() == "2030-03-31"
    assert mar.astimezone(NY).hour == 8


def test_monthly_uses_a_leap_day_when_there_is_one():
    t = {"repeat": "monthly", "tz": "America/New_York", "anchor_day": 30}
    jan = datetime(2028, 1, 30, 8, 0, tzinfo=NY)
    assert tasks._next_for(t, jan.astimezone(UTC)).astimezone(NY).day == 29


def test_first_weekday_of_the_month_skips_a_weekend_start(monkeypatch):
    """1 August 2026 is a Saturday."""
    t = {"repeat": "monthly-first-weekday", "tz": "America/New_York"}
    july = datetime(2026, 7, 1, 8, 0, tzinfo=NY)
    _freeze(monkeypatch, july + timedelta(minutes=1))
    aug = tasks._next_for(t, july.astimezone(UTC)).astimezone(NY)
    assert aug.strftime("%Y-%m-%d %a %H:%M") == "2026-08-03 Mon 08:00"


def test_monthly_expressions_parse_to_the_next_occurrence():
    now = datetime(2026, 9, 15, 12, 0, tzinfo=NY)
    first = tasks.parse_when("1st of the month 8am", now=now, tz="America/New_York")
    assert first.astimezone(NY).strftime("%Y-%m-%d %H:%M") == "2026-10-01 08:00"
    fw = tasks.parse_when("first weekday of the month 08:00", now=now, tz="America/New_York")
    assert fw.astimezone(NY).strftime("%Y-%m-%d") == "2026-10-01"  # a Thursday
    assert tasks.parse_when("15th of the month 13:00", now=now,
                            tz="America/New_York").astimezone(NY).day == 15


def test_add_task_records_the_monthly_anchor(monkeypatch):
    _freeze(monkeypatch, datetime(2030, 1, 20, 12, 0, tzinfo=NY))
    t = tasks.add_task("review", "31st of the month 8am", "monthly", tz="America/New_York")
    assert t["anchor_day"] == 31


# --- a recurring task survives a bad day ----------------------------------------------


def test_a_recurring_task_that_fails_out_skips_the_run_but_stays_scheduled():
    """One-shots park after three failures. A daily report used to do the same —
    so a provider outage on one day ended the daily report for good."""
    t = tasks.add_task("daily brief", "+0m", repeat="daily")
    for _ in range(tasks.MAX_ATTEMPTS - 1):
        tasks.claim_due()
        assert tasks.record_result(t["id"], False, "boom").get("exhausted") is None
    tasks.claim_due()
    rec = tasks.record_result(t["id"], False, "boom")
    assert rec["exhausted"] is True
    stored = tasks.get_task(t["id"])
    assert stored["status"] == "pending" and stored["attempts"] == 0
    assert stored["missed_runs"] == 1 and "exhausted" not in stored
    assert datetime.fromisoformat(stored["due"]) > tasks.now_utc()


def test_an_expired_claim_that_exhausts_a_recurring_task_also_moves_it_on():
    t = tasks.add_task("daily brief", "+0m", repeat="daily")
    later = tasks.now_utc()
    for _ in range(tasks.MAX_ATTEMPTS):
        tasks.claim_due(now=later)
        later += timedelta(minutes=tasks.DEFAULT_CLAIM_TIMEOUT_MINUTES + 1)
    tasks.claim_due(now=later)
    stored = tasks.get_task(t["id"])
    assert stored["status"] == "pending" and stored["missed_runs"] == 1


def test_the_failed_run_of_a_recurring_task_is_reported_once(monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (sent.append(text), (["t"], []))[1])

    async def fails(prompt, session_id, fake=False):
        return False, "the model provider is down"

    monkeypatch.setattr(scheduler, "_answer", fails)
    t = tasks.add_task("daily brief", "+0m", repeat="daily")
    for _ in range(tasks.MAX_ATTEMPTS):
        tasks.claim_due()
        asyncio.run(scheduler.run_task(tasks.get_task(t["id"])))
    assert len(sent) == 1, "silent while retrying, one message when it gives up"
    assert "skipped this run; it stays scheduled" in sent[0]


# --- jobs -------------------------------------------------------------------------------


@pytest.fixture
def job_runs(monkeypatch):
    runs: list[dict] = []
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        channels, "deliver", lambda text, prefer="": (sent.append(("text", text)), (["t"], []))[1]
    )
    monkeypatch.setattr(
        channels, "deliver_file",
        lambda path, caption="", prefer="", full_quality=False: (sent.append(("file", path)), (["t"], []))[1],
    )
    return runs, sent


def test_a_job_task_runs_its_handler_with_the_occurrence_it_is_for(monkeypatch, job_runs):
    runs, sent = job_runs

    async def handler(task, fake):
        runs.append(task)
        return jobs.JobResult(True, "the weekly report", files=["/tmp/w.pdf"])

    monkeypatch.setitem(jobs._JOBS, "test-report", handler)
    t = tasks.add_task("[job] test", "+0m", kind="job", job="test-report")
    asyncio.run(scheduler.run_due())
    assert runs[0]["due"] == t["due"], "the period comes from the due time"
    assert sent[0][0] == "text" and "the weekly report" in sent[0][1]
    assert sent[1] == ("file", "/tmp/w.pdf")
    assert tasks.get_task(t["id"])["status"] == "done"


def test_a_quiet_job_records_success_and_sends_nothing(monkeypatch, job_runs):
    _runs, sent = job_runs

    async def handler(task, fake):
        return jobs.JobResult(True, "market closed — no daily report", notify=False)

    monkeypatch.setitem(jobs._JOBS, "quiet", handler)
    tasks.add_task("[job] quiet", "+0m", kind="job", job="quiet")
    results = asyncio.run(scheduler.run_due())
    assert results[0]["ok"] is True and sent == []


def test_a_job_is_not_held_to_the_model_answer_gate(monkeypatch, job_runs):
    """"ok" is a complete report from a job; from a model it would be a non-answer."""
    async def handler(task, fake):
        return jobs.JobResult(True, "ok")

    monkeypatch.setitem(jobs._JOBS, "terse", handler)
    tasks.add_task("[job] terse", "+0m", kind="job", job="terse")
    assert asyncio.run(scheduler.run_due())[0]["ok"] is True


def test_an_unknown_job_fails_visibly(job_runs):
    t = tasks.add_task("[job] ghost", "+0m", kind="job", job="no-such-job")
    res = asyncio.run(scheduler.run_due())
    assert res[0]["ok"] is False and "unknown job" in res[0]["answer"]
    assert tasks.get_task(t["id"])["attempts"] == 1


def test_a_job_that_starts_a_turn_does_not_deadlock_on_its_own_lane(monkeypatch, job_runs):
    """The job holds the one background slot; the turn it starts must not wait
    for that same slot."""
    async def inner(prompt, session_id, fake):
        return True, "inner turn" + " with enough words to count as an answer."

    monkeypatch.setattr(scheduler, "_answer_unlocked", inner)

    async def handler(task, fake):
        ok, text = await scheduler._answer("p", "s", fake)
        return jobs.JobResult(ok, text)

    monkeypatch.setitem(jobs._JOBS, "nested", handler)
    tasks.add_task("[job] nested", "+0m", kind="job", job="nested")

    async def main():
        scheduler._lanes = {"chat": asyncio.Semaphore(1), "background": asyncio.Semaphore(1)}
        try:
            return await asyncio.wait_for(scheduler.run_due(), 5)
        finally:
            scheduler._lanes = None

    assert asyncio.run(main())[0]["ok"] is True


def test_the_flex_job_is_quiet_on_success_and_loud_on_failure(monkeypatch):
    from financial_research_assistant import flex

    monkeypatch.setattr(flex, "flex_sync", lambda: "Fetched Flex statement … Imported 2026-09-29")
    ok = asyncio.run(jobs._flex_sync_job({}, False))
    assert ok.ok and not ok.notify
    monkeypatch.setattr(flex, "flex_sync", lambda: "Flex sync failed: token expired")
    bad = asyncio.run(jobs._flex_sync_job({}, False))
    assert not bad.ok and bad.notify


# --- the ledger and the default schedule ---------------------------------------------------


def test_the_ledger_remembers_what_went_out():
    assert jobs.already_sent("weekly", "2026-W39") is None
    jobs.record_sent("weekly", "2026-W39", {"files": ["/r/w.pdf"]})
    entry = jobs.already_sent("weekly", "2026-W39")
    assert entry["files"] == ["/r/w.pdf"] and entry["period"] == "2026-W39"
    assert jobs.already_sent("weekly", "2026-W40") is None
    assert jobs.sent_reports()[0]["period"] == "2026-W39"


def test_default_jobs_are_created_once_and_only_where_they_apply(monkeypatch):
    monkeypatch.delenv("IBKR_FLEX_TOKEN", raising=False)
    rows = dict((j, note) for j, _t, note in jobs.ensure_default_jobs())
    assert rows["flex-sync"].startswith("not applicable")
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "t")
    first = {j: t for j, t, _n in jobs.ensure_default_jobs()}
    assert first["flex-sync"]["tz"] == "America/New_York"
    again = {j: note for j, _t, note in jobs.ensure_default_jobs()}
    assert again["flex-sync"] == "already scheduled"
    assert sum(t.get("job") == "flex-sync" for t in tasks.load_tasks()) == 1


def test_market_hours_are_new_york_hours():
    assert market.regular_hours(datetime(2026, 9, 29, 10, 0, tzinfo=NY)) is True
    assert market.regular_hours(datetime(2026, 9, 29, 16, 0, tzinfo=NY)) is False
    assert market.regular_hours(datetime(2026, 10, 3, 11, 0, tzinfo=NY)) is False  # Sat
    assert market.regular_hours(datetime(2026, 9, 29, 13, 0, tzinfo=UTC)) is False  # 9:00 NY
