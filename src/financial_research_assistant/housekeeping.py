"""Housekeeping: the always-on server's state, kept from growing for ever.

Every store here is written on each tick, pass or turn, and nothing removed
anything: finished one-shot tasks piled up in ``tasks.json`` (rewritten on every
claim), and the run, usage and event logs grew for the life of the server, with
``autonomy.spent`` re-reading the whole usage log on every budget check. A daily
job trims each to what anything still reads. The thesis journal, the ideas runs
and your own conversations are records, not logs, and are kept.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import asyncio
import time

from . import jobs

#: Days each log is kept, and why that long.
KEEP_DAYS = {
    "runs": 90,      # /status and the run history look back days, not months
    "usage": 62,     # the monthly budget needs this month and the last
    "events": 180,   # the weekly report reads a week; /events the latest few
}
#: Finished one-shot tasks stay listed this long after they ran.
TASK_DAYS = 30
#: Transcripts of scheduled runs (``task-*``), and answers given up undelivered.
TRANSCRIPT_DAYS = 180


def run(now: datetime | None = None) -> dict[str, int]:
    """Trim every store; returns what went from each."""
    from . import autonomy, scheduler, sessions, tasks, watchers
    from .storage import prune_jsonl

    now = now or datetime.now(timezone.utc)

    def since(days: int) -> str:
        return (now - timedelta(days=days)).astimezone().replace(tzinfo=None).isoformat()

    out = {
        "runs": prune_jsonl(scheduler.runs_log(), since(KEEP_DAYS["runs"])),
        "usage": prune_jsonl(autonomy.usage_log(), since(KEEP_DAYS["usage"])),
        "events": prune_jsonl(watchers.events_log(), since(KEEP_DAYS["events"])),
        "tasks": tasks.purge(finished_before=now - timedelta(days=TASK_DAYS)),
    }
    cutoff = (now - timedelta(days=TRANSCRIPT_DAYS)).timestamp()
    gone = 0
    folders = [(sessions.store_dir(), "task-*.jsonl"),
               (tasks.tasks_file().with_name("undelivered"), "*.txt")]
    for folder, pattern in folders:
        for f in folder.glob(pattern) if folder.exists() else ():
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                gone += 1
    out["files"] = gone
    return out


async def _job(task: dict[str, Any], fake: bool) -> jobs.JobResult:
    started = time.monotonic()
    removed = await asyncio.to_thread(run)
    summary = ", ".join(f"{k} {v}" for k, v in removed.items() if v) or "nothing to remove"
    return jobs.JobResult(True, f"housekeeping ({time.monotonic() - started:.1f}s): {summary}",
                          notify=False)


jobs.register_job("housekeeping", _job)
jobs.register_default_job(jobs.DefaultJob(
    "housekeeping", "[job] Housekeeping", "03:30", "daily"))
