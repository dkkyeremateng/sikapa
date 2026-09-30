"""Two simulated days of the always-on scheduler, on an accelerated clock.

Every tick is a real `run_due` pass and every task a real fake-mode turn through
`adapter.run_turn` — only the clock and the delivery channel are faked. What
this catches is what only shows up over time: a claim that is never released, a
task delivered twice for one occurrence, a recurring task that drifts or stops,
and per-run state that is never freed (a daily task used to keep every earlier
day's conversation; a long-lived process kept every run's graph).
"""

import asyncio
import tracemalloc
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from financial_research_assistant import adapter, channels, jobs, scheduler, tasks

NY = ZoneInfo("America/New_York")
START = datetime(2026, 9, 28, 0, 0, tzinfo=NY)  # a Monday
TICK = timedelta(minutes=30)
TICKS = 96  # 48 hours


def test_two_days_of_ticks_leave_nothing_stranded_duplicated_or_leaking(monkeypatch):
    clock = [START.astimezone(timezone.utc)]
    monkeypatch.setattr(tasks, "now_utc", lambda: clock[0])
    monkeypatch.setattr(tasks, "record_tick", lambda now=None: None)
    sent: list[tuple[str, str]] = []

    def deliver(text, prefer=""):
        sent.append((text.split("\n", 1)[0], clock[0].isoformat()))
        return ["telegram"], []

    monkeypatch.setattr(channels, "deliver", deliver)
    quiet_runs: list[str] = []

    async def quiet_job(task, fake):
        quiet_runs.append(task["due"])
        return jobs.JobResult(True, "synced", notify=False)

    monkeypatch.setitem(jobs._JOBS, "soak-quiet", quiet_job)

    flaky_calls = Counter()
    real_answer = scheduler._answer_unlocked

    async def answer(prompt, session_id, fake):
        if "flaky" in prompt:
            flaky_calls[session_id.rsplit("-", 1)[0]] += 1
            if flaky_calls[session_id.rsplit("-", 1)[0]] % 2:
                return False, "provider blip"
        return await real_answer(prompt, session_id, fake)

    monkeypatch.setattr(scheduler, "_answer_unlocked", answer)

    hourly = tasks.add_task("pulse check", "+0m", "hourly")
    daily = tasks.add_task("morning brief", "09:00", "daily", tz="America/New_York")
    weekday_job = tasks.add_task("[job] sync", "16:45", "weekdays", tz="America/New_York",
                                 kind="job", job="soak-quiet")
    once = tasks.add_task("one-off question", "+5h")
    flaky = tasks.add_task("flaky daily", "10:00", "daily", tz="America/New_York")

    # Other tests leave their own sessions in these module-level caches; what
    # matters here is that none of THESE runs stay behind.
    graphs_before, checkpointers_before = set(adapter._fake_graphs), set(adapter._checkpointers)
    tracemalloc.start()
    baseline = None
    for n in range(TICKS):
        clock[0] += TICK
        asyncio.run(scheduler.run_due(fake=True))
        if n == 12:
            baseline = tracemalloc.take_snapshot()
    growth = sum(s.size_diff for s in tracemalloc.take_snapshot().compare_to(baseline, "filename"))
    tracemalloc.stop()

    by_task = Counter(head for head, _at in sent)
    # hourly: due at 00:00 and every hour through hour 48 inclusive — 49 runs,
    # none skipped and none doubled by the half-hour ticks.
    assert by_task["🤖 task " + hourly["id"]] == 49
    assert by_task["🤖 task " + daily["id"]] == 2, "09:00 New York, two mornings"
    assert by_task["🤖 task " + once["id"]] == 1
    # Flaky: fails every other attempt, so each morning takes a retry — and is
    # still delivered exactly once per day.
    assert by_task["🤖 task " + flaky["id"]] == 2
    assert len(quiet_runs) == 2 and not any("sync" in h for h, _ in sent), "Mon + Tue, silent"

    # One delivery per occurrence: no hour sees the same task twice.
    per_slot = Counter((head, at[:13]) for head, at in sent)
    assert max(per_slot.values()) == 1

    # Nothing left claimed, nothing parked, nothing in error.
    final = {t["id"]: t for t in tasks.load_tasks()}
    assert all(t["status"] in ("pending", "done") for t in final.values())
    assert not any(t.get("pending_delivery") for t in final.values())
    assert final[once["id"]]["status"] == "done"
    assert datetime.fromisoformat(final[daily["id"]]["due"]).astimezone(NY).hour == 9, "no drift"
    assert final[weekday_job["id"]]["status"] == "pending"

    # Per-run state is freed, and memory is flat after warm-up.
    assert set(adapter._fake_graphs) - graphs_before == set()
    assert set(adapter._checkpointers) - checkpointers_before == set()
    assert growth < 8 * 1024 * 1024, f"grew {growth / 1e6:.1f} MB over 42 simulated hours"
