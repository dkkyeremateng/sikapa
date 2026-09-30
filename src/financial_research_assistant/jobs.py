"""Built-in jobs, the report ledger, and the default schedule.

A *job* is a task of kind ``job``: the scheduler runs a registered function
instead of a model turn (see ``scheduler.register_job``). This module registers
the built-in ones, and knows which of them a fresh install should have
scheduled (``ensure_default_jobs``, behind ``--reports-setup``).

**The ledger** (``report-ledger.json``) is what makes a report idempotent. A
report is keyed by its kind and the period it covers (``daily:2026-09-29``,
``weekly:2026-W39``), and one whose key is already there does nothing — no model
call, no second message. Without it, a claim that expired while a report was
still rendering, or a restart between rendering and recording, would send the
same week twice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
import asyncio
import json
import os
from datetime import datetime, timezone

from . import tasks
from .storage import locked, read_json, state_file, write_private


# --- the registry ----------------------------------------------------------------
#
# Here rather than in `scheduler` so job modules never import the scheduler: it
# imports them (lazily, when a job runs), and the dependency only goes one way.


@dataclass
class JobResult:
    """What a job hands back. ``text`` is the message; ``files`` are delivered
    after it (a rendered PDF, its cover). ``notify=False`` records the outcome
    and pushes nothing — a quiet success, a holiday with no report to send."""

    ok: bool
    text: str = ""
    files: list[str] = field(default_factory=list)
    notify: bool = True


#: ``(task, fake) -> JobResult``. The task is the claimed copy, so ``task["due"]``
#: is the occurrence being run — which is what a report's period is derived from.
JobHandler = Callable[[dict[str, Any], bool], Awaitable[JobResult]]
_JOBS: dict[str, JobHandler] = {}


def register_job(name: str, handler: JobHandler) -> None:
    _JOBS[name] = handler


def handler_for(name: str) -> JobHandler | None:
    _load_job_modules()
    return _JOBS.get(name)


def registered_jobs() -> list[str]:
    _load_job_modules()
    return sorted(_JOBS)


# --- the ledger ------------------------------------------------------------------


def ledger_file():
    return state_file("report-ledger.json", "FRA_REPORT_LEDGER")


def _key(report: str, period: str) -> str:
    return f"{report}:{period}"


def already_sent(report: str, period: str) -> dict[str, Any] | None:
    """The ledger entry for this report and period, if it has gone out."""
    data = read_json(ledger_file(), {})
    entry = data.get(_key(report, period)) if isinstance(data, dict) else None
    return entry if isinstance(entry, dict) else None


def record_sent(report: str, period: str, info: dict[str, Any] | None = None) -> None:
    """Enter a report in the ledger. Written under a lock and atomically, like
    every other store here."""
    path = ledger_file()
    with locked(path):
        data = read_json(path, {})
        if not isinstance(data, dict):
            data = {}
        data[_key(report, period)] = {
            "report": report, "period": period,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(info or {}),
        }
        write_private(path, json.dumps(data, indent=2), prefix=".ledger-")


def sent_reports(limit: int = 20) -> list[dict[str, Any]]:
    """The most recent ledger entries, newest first."""
    data = read_json(ledger_file(), {})
    items = [v for v in data.values() if isinstance(v, dict)] if isinstance(data, dict) else []
    items.sort(key=lambda e: str(e.get("at") or ""), reverse=True)
    return items[:limit]


def scheduled_for(task: dict[str, Any]) -> datetime:
    """The occurrence a job run is FOR — its due time, not the time it happens to
    run. A weekly report run late on Tuesday after an outage still covers the
    week it was scheduled for, rather than the one that has just started."""
    due = tasks._parse_due(task.get("due"))  # pyright: ignore[reportPrivateUsage]
    return due or datetime.now(timezone.utc)


# --- built-in jobs -------------------------------------------------------------------


async def _flex_sync_job(task: dict[str, Any], fake: bool) -> JobResult:
    """Pull the IBKR Flex statement into the store. Model-free plumbing, so a
    success is silent; a failure goes through the normal retry-then-report path."""
    if fake:
        return JobResult(True, "flex sync skipped in fake mode", notify=False)
    from .flex import flex_sync

    out = await asyncio.to_thread(flex_sync)
    ok = out.startswith("Fetched") and "could not be imported" not in out
    return JobResult(ok, out, notify=not ok)


register_job("flex-sync", _flex_sync_job)


# --- the default schedule --------------------------------------------------------------


@dataclass(frozen=True)
class DefaultJob:
    """One entry in the schedule ``--reports-setup`` creates."""

    job: str
    description: str
    when: str
    repeat: str
    tz: str = ""
    #: Only created when this says the job can work here (e.g. a Flex token).
    applies: Callable[[], bool] = lambda: True
    options: tuple[tuple[str, Any], ...] = ()


def _market_tz() -> str:
    return "America/New_York"


def _user_tz() -> str:
    """The zone for weekly/monthly jobs: ``TZ`` when it names an IANA zone (the
    server env sets it), else the host's local zone."""
    raw = (os.environ.get("TZ") or "").strip().lstrip(":")
    if raw and "/" in raw:
        try:
            tasks.zone(raw)
            return raw
        except ValueError:
            pass
    return ""


#: Filled by the modules that own each job, so this list never has to import them.
DEFAULT_JOBS: list[DefaultJob] = [
    DefaultJob(
        "flex-sync", "[job] Sync the IBKR Flex statement", "16:45", "weekdays",
        tz=_market_tz(), applies=lambda: bool(os.environ.get("IBKR_FLEX_TOKEN")),
    ),
]


def register_default_job(job: DefaultJob) -> None:
    if all(j.job != job.job for j in DEFAULT_JOBS):
        DEFAULT_JOBS.append(job)


def _load_job_modules() -> None:
    """Import the modules that register jobs, so the registry and the default
    schedule are complete before either is read."""
    import importlib

    for name in ("periodic", "recommend"):
        try:
            importlib.import_module(f"{__package__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.{name}":
                raise


def ensure_default_jobs() -> list[tuple[str, dict[str, Any] | None, str]]:
    """Create each default job that applies and isn't already scheduled.

    Idempotent: a job already pending or running is left alone, so running this
    on every deploy is safe. Returns ``(job, task_or_None, note)`` per entry.
    """
    _load_job_modules()
    out: list[tuple[str, dict[str, Any] | None, str]] = []
    live = {
        str(t.get("job")) for t in tasks.load_tasks()
        if t.get("kind") == "job" and t.get("status") in ("pending", "running")
    }
    for spec in DEFAULT_JOBS:
        if spec.job in live:
            out.append((spec.job, None, "already scheduled"))
            continue
        if not spec.applies():
            out.append((spec.job, None, "not applicable here (not configured)"))
            continue
        tz = spec.tz or _user_tz()
        task = tasks.add_task(
            spec.description, spec.when, spec.repeat, tz=tz, kind="job", job=spec.job,
            options=dict(spec.options) or None,
        )
        out.append((spec.job, task, "scheduled"))
    return out


def describe_setup(rows: list[tuple[str, dict[str, Any] | None, str]]) -> str:
    lines = []
    for job, task, note in rows:
        if task is not None:
            lines.append(f"  + {job:<14} {tasks.describe(task)}")
        else:
            lines.append(f"  · {job:<14} {note}")
    return "\n".join(lines)
