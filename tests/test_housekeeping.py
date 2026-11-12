"""Housekeeping: each store trimmed to what anything still reads, nothing else."""

import asyncio
import os
from datetime import datetime, timedelta, timezone

from financial_research_assistant import autonomy, housekeeping, jobs, scheduler, sessions, tasks, watchers
from financial_research_assistant.storage import append_jsonl, read_jsonl

NOW = datetime(2026, 11, 12, 3, 30, tzinfo=timezone.utc)


def _stamp(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).astimezone().replace(tzinfo=None).isoformat(
        timespec="seconds")


def test_logs_keep_what_is_still_read(monkeypatch):
    for days in (200, 100, 70, 10):
        append_jsonl(scheduler.runs_log(), {"at": _stamp(days), "id": f"r{days}"})
        append_jsonl(autonomy.usage_log(), {"at": _stamp(days), "tokens": days})
        append_jsonl(watchers.events_log(), {"at": _stamp(days), "key": f"e{days}"})
    removed = housekeeping.run(NOW)
    assert [r["id"] for r in read_jsonl(scheduler.runs_log())] == ["r70", "r10"]
    assert [r["tokens"] for r in read_jsonl(autonomy.usage_log())] == [10]
    assert [r["key"] for r in read_jsonl(watchers.events_log())] == ["e100", "e70", "e10"]
    assert removed["runs"] == 2 and removed["usage"] == 3 and removed["events"] == 1


def test_finished_tasks_go_after_a_month_errors_and_pending_stay():
    old = tasks.add_task("old one-shot", "+0m")
    recent = tasks.add_task("recent one-shot", "+0m")
    broken = tasks.add_task("broken", "+0m")
    pending = tasks.add_task("later", "+2h")
    tasks.claim_due()
    tasks.record_result(old["id"], True, "done", now=NOW - timedelta(days=40))
    tasks.record_result(recent["id"], True, "done", now=NOW - timedelta(days=2))
    for _ in range(tasks.MAX_ATTEMPTS):
        tasks.record_result(broken["id"], False, "boom", now=NOW - timedelta(days=40))
    housekeeping.run(NOW)
    assert {t["id"] for t in tasks.load_tasks()} == {recent["id"], broken["id"], pending["id"]}


def test_old_scheduled_run_transcripts_go_chats_stay():
    sessions.log_turn("task-s1-20260101", "q", "a")
    sessions.log_turn("telegram-999", "q", "a")
    sessions.log_turn("task-s2-20261110", "q", "a")
    old = (NOW - timedelta(days=housekeeping.TRANSCRIPT_DAYS + 1)).timestamp()
    for name in ("task-s1-20260101", "telegram-999"):
        os.utime(sessions.store_dir() / f"{name}.jsonl", (old, old))
    housekeeping.run(NOW)
    left = sorted(p.stem for p in sessions.store_dir().glob("*.jsonl"))
    assert left == ["task-s2-20261110", "telegram-999"]


def test_it_runs_nightly_and_quietly():
    assert any(j.job == "housekeeping" and j.repeat == "daily" for j in jobs.DEFAULT_JOBS)
    res = asyncio.run(jobs.handler_for("housekeeping")({"id": "s1"}, False))
    assert res.ok and not res.notify
