"""Scheduled-task store: time parsing, claiming, recurrence, and bookkeeping.

Offline — no model, no network. The autouse fixture in conftest.py points every
test at a throwaway tasks.json, so nothing here can queue work in the real store.
"""

import stat
from datetime import datetime, timedelta, timezone

import pytest

from financial_research_assistant import tasks


def _local(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm).astimezone()


# --- time parsing --------------------------------------------------------------


def test_absolute_dates_and_times():
    assert tasks.parse_when("2026-08-14 09:00") == _local(2026, 8, 14, 9, 0).astimezone(timezone.utc)
    assert tasks.parse_when("2026-08-14T13:30") == _local(2026, 8, 14, 13, 30).astimezone(timezone.utc)
    # A bare date means during the working day, not midnight: a task set for
    # "friday" that ran at 00:00 would report on a market shut for eight hours.
    assert tasks.parse_when("2026-08-14").astimezone().hour == 9


def test_times_are_stored_utc_but_read_as_local_wall_clock():
    """The user types the clock on their wall. Treating it as UTC instead would
    move a 9am task by the whole offset — most of a working day in Asia."""
    got = tasks.parse_when("2026-08-14 09:00")
    assert got.tzinfo is timezone.utc
    assert got.astimezone().hour == 9


def test_relative_and_day_words():
    now = tasks.now_utc()
    assert abs((tasks.parse_when("+2h", now) - now) - timedelta(hours=2)) < timedelta(seconds=2)
    assert abs((tasks.parse_when("+30m", now) - now) - timedelta(minutes=30)) < timedelta(seconds=2)
    assert abs((tasks.parse_when("3d", now) - now) - timedelta(days=3)) < timedelta(seconds=2)

    tomorrow = tasks.parse_when("tomorrow 9am", now).astimezone()
    assert tomorrow.hour == 9 and tomorrow.date() == (now.astimezone() + timedelta(days=1)).date()

    assert tasks.parse_when("tomorrow", now).astimezone().hour == 9  # default hour


def test_a_weekday_name_means_the_next_one():
    now = tasks.now_utc()
    got = tasks.parse_when("monday 8:30", now).astimezone()
    assert got.weekday() == 0 and got.hour == 8 and got.minute == 30
    assert got > now.astimezone(), "a weekday must resolve forward, never into the past"


def test_a_bare_time_rolls_forward_to_the_next_occurrence():
    now = tasks.now_utc()
    got = tasks.parse_when("00:01", now)
    assert got > now, "a time already past today must mean tomorrow"


def _at(hour, minute=0):
    """A fixed 'now', given as local wall-clock and returned in UTC — the shape
    parse_when takes. Pinned rather than relative to the real clock so these cases
    ('it is already past 09:00') hold whenever the suite runs.

    The hour is set on the NAIVE local time before `astimezone` attaches an offset:
    replacing the hour on an already-aware value keeps the offset belonging to the
    real current time, which is the wrong one for that wall clock on the two DST
    transition days a year.
    """
    naive = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return naive.astimezone().astimezone(timezone.utc)


def test_tonight_means_this_evening_not_this_morning():
    """"tonight" defaulted to the same 09:00 as every other bare day word, so past
    mid-morning it produced a time in the PAST — the task was immediately due and
    ran on the next tick, seconds after being scheduled."""
    got = tasks.parse_when("tonight", _at(14)).astimezone()
    assert got.hour == 20, "tonight is an evening hour"
    assert got.date() == datetime.now().astimezone().date(), "and it is still tonight"


def test_a_day_word_never_resolves_into_the_past():
    """A task dated earlier than now is immediately due, so it fires on the very
    next tick — the failure this guards is 'I scheduled it and it ran at once'."""
    for text, now in (
        ("today", _at(15)),        # default hour already gone by
        ("tonight", _at(23)),      # evening default already gone by
        ("today 16:00", _at(17)),  # an explicit time the user got wrong
        ("tomorrow", _at(23)),
    ):
        # `>=`, not `>`: a bare "today" past its default hour resolves to exactly
        # now (see below), which is the earliest a task may legitimately be due.
        assert tasks.parse_when(text, now) >= now, f"{text!r} at {now}"


def test_today_with_no_time_stays_today():
    """When the default hour has passed, "today" resolves to now — the task runs on
    the next tick, which is still today. Rolling it to tomorrow would contradict
    the one word the user actually said."""
    now = _at(15)
    got = tasks.parse_when("today", now)
    assert got - now < timedelta(seconds=2)


def test_an_explicit_past_time_rolls_to_the_next_occurrence():
    """"today 16:00" at 17:00 is a mistake about the clock, and the next 16:00 is
    what the bare-time branch already means by "16:00" — so the two agree."""
    now = _at(17)
    got = tasks.parse_when("today 16:00", now).astimezone()
    assert got.hour == 16
    assert got.date() == (now.astimezone() + timedelta(days=1)).date()


def test_unparseable_time_is_reported_not_guessed():
    """Silently defaulting to 'now' would run an expensive task immediately; the
    error names the formats that work."""
    with pytest.raises(ValueError, match="could not read a time"):
        tasks.parse_when("whenever it feels right")


# --- the store -----------------------------------------------------------------


def test_add_list_remove_roundtrip():
    t = tasks.add_task("Analyse NOMD", "+1h")
    assert t["id"] == "s1" and t["status"] == "pending"
    assert t["session"] == "task-s1", "each task gets a clean session of its own"
    assert [x["id"] for x in tasks.pending_tasks()] == ["s1"]
    assert tasks.remove_task("s1") is True
    assert tasks.pending_tasks() == []
    assert tasks.remove_task("s1") is False


def test_store_file_is_0600():
    """A prompt can quote positions and account names, so the file is no more
    readable than the credential store beside it."""
    tasks.add_task("check something", "+1h")
    mode = stat.S_IMODE(tasks.tasks_file().stat().st_mode)
    assert mode == 0o600, f"tasks file is {oct(mode)}"


def test_a_corrupt_store_reads_as_empty_rather_than_raising():
    """A bad hand-edit must not take down the tick that would have run the others."""
    tasks.tasks_file().parent.mkdir(parents=True, exist_ok=True)
    tasks.tasks_file().write_text("{not json", encoding="utf-8")
    assert tasks.load_tasks() == []


def test_invalid_repeat_is_rejected():
    with pytest.raises(ValueError, match="repeat must be one of"):
        tasks.add_task("x", "+1h", repeat="fortnightly")


# --- claiming ------------------------------------------------------------------


def test_claim_takes_only_what_is_due():
    tasks.add_task("due", "+0m")
    tasks.add_task("later", "+2h")
    claimed = tasks.claim_due()
    assert [c["prompt"] for c in claimed] == ["due"]


def test_a_claimed_task_cannot_be_claimed_again():
    """The double-run guard: a cron tick firing while a slow --watch run is still
    going must not start the same task (and bill the same model run) twice."""
    tasks.add_task("due", "+0m")
    assert len(tasks.claim_due()) == 1
    assert tasks.claim_due() == []


def test_release_puts_an_interrupted_task_back():
    """Ctrl-C mid-run: without this the task sits in 'running' forever and is never
    picked up again."""
    tasks.add_task("due", "+0m")
    claimed = tasks.claim_due()
    tasks.release(claimed[0]["id"])
    assert len(tasks.claim_due()) == 1


def test_claim_is_bounded_so_a_backlog_drains_over_several_ticks():
    for i in range(5):
        tasks.add_task(f"job {i}", "+0m")
    assert len(tasks.claim_due(limit=2)) == 2


# --- recurrence and bookkeeping ------------------------------------------------


def test_a_one_shot_task_is_done_after_it_runs():
    t = tasks.add_task("once only", "+0m")
    tasks.claim_due()
    rec = tasks.record_result(t["id"], True, "the answer")
    assert rec["status"] == "done" and rec["runs"] == 1
    assert tasks.claim_due() == []


def test_a_recurring_task_reschedules_from_its_due_time_not_from_now():
    """Rescheduling from completion drifts: a 09:00 daily that takes four minutes
    would creep to 09:04, then 09:08, and eventually into the afternoon."""
    due = tasks.now_utc() - timedelta(minutes=5)
    t = tasks.add_task("daily brief", due.astimezone().strftime("%Y-%m-%d %H:%M"), repeat="daily")
    tasks.claim_due()
    rec = tasks.record_result(t["id"], True, "done")
    assert rec["status"] == "pending"
    nxt = datetime.fromisoformat(rec["due"])
    original = datetime.fromisoformat(t["due"])
    assert (nxt - original) == timedelta(days=1)


def test_a_long_sleep_does_not_fire_a_burst_of_catch_up_runs():
    """A laptop shut for three days should resume tomorrow, not run three dailies."""
    old = tasks.now_utc() - timedelta(days=3)
    nxt = tasks.next_due(old, "daily")
    assert nxt > tasks.now_utc()
    assert nxt - tasks.now_utc() < timedelta(days=1)


def test_weekdays_repeat_skips_the_weekend():
    saturday = datetime(2026, 8, 7, 9, 0).astimezone()  # Friday
    while saturday.weekday() != 4:
        saturday += timedelta(days=1)
    nxt = tasks.next_due(saturday.astimezone(timezone.utc), "weekdays")
    assert nxt.astimezone().weekday() < 5


def test_a_failing_task_retries_then_parks_so_it_stops_burning_model_calls():
    t = tasks.add_task("broken", "+0m")
    for attempt in range(1, tasks.MAX_ATTEMPTS):
        tasks.claim_due()
        rec = tasks.record_result(t["id"], False, "boom")
        assert rec["status"] == "pending", f"attempt {attempt} should retry"
    tasks.claim_due()
    rec = tasks.record_result(t["id"], False, "boom")
    assert rec["status"] == "error"
    assert tasks.claim_due() == [], "a parked task must not keep running forever"


def test_a_successful_run_clears_the_failure_count():
    t = tasks.add_task("flaky", "+0m", repeat="hourly")
    tasks.claim_due()
    tasks.record_result(t["id"], False, "transient")
    rec = tasks.record_result(t["id"], True, "fine now")
    assert rec["attempts"] == 0


def test_stored_results_are_capped():
    """The store is re-read and rewritten on every tick; whole reports would turn a
    2 KB file into megabytes. The answer's home is the channel it went to."""
    t = tasks.add_task("verbose", "+0m")
    tasks.claim_due()
    rec = tasks.record_result(t["id"], True, "x" * 10_000)
    assert len(rec["last_result"]) <= 500


def test_purge_keeps_errors_by_default():
    done = tasks.add_task("a", "+0m")
    bad = tasks.add_task("b", "+0m")
    tasks.claim_due()
    tasks.record_result(done["id"], True, "ok")
    for _ in range(tasks.MAX_ATTEMPTS):
        tasks.record_result(bad["id"], False, "no")
    assert tasks.purge() == 1
    assert [t["id"] for t in tasks.load_tasks()] == [bad["id"]]


# --- model-facing tools --------------------------------------------------------


def test_schedule_task_tool_confirms_when_and_where():
    out = tasks.schedule_task("Analyse NOMD Q3 vs consensus", "tomorrow 9am")
    assert "Scheduled [s1]" in out and "09:00" in out
    assert tasks.pending_tasks()[0]["prompt"] == "Analyse NOMD Q3 vs consensus"


def test_schedule_task_tool_reports_a_bad_time_instead_of_scheduling_junk():
    out = tasks.schedule_task("do a thing", "sometime soon")
    assert "Could not schedule" in out
    assert tasks.load_tasks() == []


def test_list_and_cancel_tools():
    tasks.schedule_task("watch AAPL", "+2h")
    listing = tasks.list_scheduled_tasks()
    assert "watch AAPL" in listing and "[s1]" in listing
    assert "Cancelled s1" in tasks.cancel_scheduled_task("s1")
    assert "Nothing scheduled" in tasks.list_scheduled_tasks()
    assert "No scheduled task with id" in tasks.cancel_scheduled_task("s9")


# --- the runner heartbeat ------------------------------------------------------


def test_scheduling_with_no_runner_warns_instead_of_promising_delivery():
    """The silent failure this exists to catch: a task was queued, nothing was
    running --run-due, and the user waited for a message that was never coming."""
    out = tasks.schedule_task("Analyse FISV", "+5m")
    assert "WARNING: no task runner is active" in out
    assert "--watch" in out


def test_a_recent_tick_means_no_warning():
    tasks.record_tick()
    out = tasks.schedule_task("Analyse FISV", "+5m")
    assert "WARNING" not in out
    assert tasks.runner_is_live() is True


def test_a_stale_tick_counts_as_no_runner(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_RUNNER_MAX_SILENCE", "30")
    tasks.record_tick(tasks.now_utc() - timedelta(hours=2))
    assert tasks.runner_is_live() is False
    assert "not since" in tasks.runner_warning()


def test_the_listing_warns_too_when_tasks_are_waiting():
    tasks.add_task("queued", "+1h")
    assert "no task runner is active" in tasks.list_scheduled_tasks()
