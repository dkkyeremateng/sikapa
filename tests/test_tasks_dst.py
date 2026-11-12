"""Repeat rules keep the time and day they were set for."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from financial_research_assistant import tasks

NY = ZoneInfo("America/New_York")


def test_a_daily_task_in_the_spring_forward_gap_comes_back(monkeypatch):
    """02:30 does not exist on 8 March 2026 in New York. The run that day is at
    03:30, but the next must be 02:30 again; it drifted to 03:30 for good."""
    monkeypatch.setattr(tasks, "now_utc", lambda: datetime(2026, 3, 7, 12, tzinfo=timezone.utc))
    t = tasks.add_task("night job", "2026-03-07 02:30", "daily", tz="America/New_York")
    assert t["clock"] == "02:30"
    due = datetime.fromisoformat(t["due"])
    seen = []
    for _ in range(4):
        monkeypatch.setattr(tasks, "now_utc", lambda d=due: d)
        due = tasks._next_for(t, due)
        seen.append(due.astimezone(NY).strftime("%m-%d %H:%M"))
    # The gap day itself can only be 03:30; every day after is 02:30 again.
    assert seen == ["03-08 03:30", "03-09 02:30", "03-10 02:30", "03-11 02:30"], seen


def test_the_first_run_of_a_first_weekday_task_is_a_weekday(monkeypatch):
    """Set up in July, "1st of the month 8am" first ran on Saturday 1 August."""
    monkeypatch.setattr(tasks, "now_utc", lambda: datetime(2026, 7, 15, 12, tzinfo=timezone.utc))
    t = tasks.add_task("monthly review", "2026-08-01 08:00", "monthly-first-weekday",
                       tz="America/New_York")
    first = datetime.fromisoformat(t["due"]).astimezone(NY)
    assert first.strftime("%a %Y-%m-%d %H:%M") == "Mon 2026-08-03 08:00"
