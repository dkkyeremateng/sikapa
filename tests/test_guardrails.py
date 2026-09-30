"""Guardrails on autonomous work: the switch, quiet hours, the budget, the record."""

import asyncio
from datetime import date, datetime

import pytest

from financial_research_assistant import (
    autonomy, channels, guardrails, periodic, scheduler, statements, tasks,
)


# --- the switch ---------------------------------------------------------------------


def test_a_pause_survives_a_restart_and_resume_lifts_it():
    """Stored in a file, not in memory: a deploy that silently un-paused the agent
    would make the switch worthless."""
    assert guardrails.paused_reason() == ""
    guardrails.pause("travelling")
    assert "paused" in guardrails.paused_reason()
    assert guardrails._read()["reason"] == "travelling"  # what a new process reads
    guardrails.resume()
    assert guardrails.paused_reason() == ""


def test_the_environment_switch_outranks_resume(monkeypatch):
    monkeypatch.setenv("FRA_AUTONOMY", "off")
    reply = asyncio.run(scheduler.run_command("/resume"))
    assert "FRA_AUTONOMY" in reply
    assert autonomy.blocked()


def test_pause_and_resume_from_the_phone():
    assert "Paused" in asyncio.run(scheduler.run_command("/pause"))
    assert autonomy.blocked()
    assert "Resumed" in asyncio.run(scheduler.run_command("/resume"))
    assert not autonomy.blocked()


# --- quiet ---------------------------------------------------------------------------


def test_quiet_hours_wrap_midnight(monkeypatch):
    monkeypatch.setenv("FRA_QUIET_HOURS", "22:00-07:00")
    assert guardrails.quiet_now(datetime(2026, 9, 30, 23, 30))
    assert guardrails.quiet_now(datetime(2026, 9, 30, 6, 59))
    assert guardrails.quiet_now(datetime(2026, 9, 30, 7, 0)) == ""
    assert guardrails.quiet_now(datetime(2026, 9, 30, 12, 0)) == ""
    monkeypatch.setenv("FRA_QUIET_HOURS", "nonsense")
    assert guardrails.quiet_now(datetime(2026, 9, 30, 23, 30)) == ""


def test_quiet_for_a_while_then_off():
    assert "Quiet until" in asyncio.run(scheduler.run_command("/quiet 2h"))
    assert guardrails.quiet_now().startswith("quiet until")
    assert asyncio.run(scheduler.run_command("/quiet off")) == "Quiet mode off."
    assert guardrails.quiet_now() == ""
    assert "Usage" in asyncio.run(scheduler.run_command("/quiet soon"))


# --- the budget ------------------------------------------------------------------------


def test_spend_is_summed_by_day_and_month():
    autonomy.record_usage("report:daily", 1000, 200)
    autonomy.record_usage("task", 50, 50)
    used = autonomy.spent()
    assert used == {"today": 1300, "month": 1300}
    assert autonomy.spent(datetime(2030, 1, 1)) == {"today": 0, "month": 0}


def test_an_exhausted_budget_stops_autonomous_model_calls(monkeypatch):
    monkeypatch.setenv("FRA_AUTONOMY_DAILY_TOKENS", "1000")
    autonomy.record_usage("report:weekly", 900, 150)
    assert "today's autonomous token budget is spent" in autonomy.over_budget()

    def no_model(*a, **k):
        raise AssertionError("over budget: the model must not be called")

    import financial_research_assistant.llm as llm

    monkeypatch.setattr(llm, "quick_llm", no_model)
    monkeypatch.setattr(llm, "_make_llm", no_model)
    assert asyncio.run(autonomy.ask("s", "u")) is None
    brief = periodic.Brief(kind="daily", period="d", label="l", title="t", subtitle="",
                           highlights="A | +1.00%", markdown="", message="")
    text, note = asyncio.run(periodic.write_commentary(brief))
    assert text == "" and "budget is spent" in note


def test_the_status_shows_the_spend_and_the_switch(monkeypatch):
    monkeypatch.setenv("FRA_AUTONOMY_MONTHLY_TOKENS", "5000")
    autonomy.record_usage("task", 100, 20)
    line = autonomy.budget_line()
    assert line.startswith("autonomy: on") and "this month 120/5,000" in line
    guardrails.pause()
    assert "paused" in scheduler.format_status(scheduler.service_status())


# --- what scheduled turns cost, and the record of every run --------------------------------


class _Ev:
    def __init__(self, kind, text="", tool="", tokens_in=0, tokens_out=0):
        self.kind, self.text, self.tool = kind, text, tool
        self.tokens_in, self.tokens_out = tokens_in, tokens_out


@pytest.fixture
def scripted_turn(monkeypatch):
    """A real `_answer` path over a scripted event stream: usage and tool events
    included, so the accounting is exercised, not stubbed away."""
    from financial_research_assistant import adapter, tracing

    async def run_turn(prompt, session_id, fake=False):
        yield _Ev("tool_end", tool="risk_metrics")
        yield _Ev("usage", tokens_in=1200, tokens_out=300)
        yield _Ev("final", "A full answer with figures, sources and a verdict to act on.")

    monkeypatch.setattr(adapter, "run_turn", run_turn)
    monkeypatch.setattr(tracing, "traced", lambda gen, **_kw: gen)
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (["t"], []))


def test_a_scheduled_turn_counts_as_autonomous_spend_and_is_recorded(scripted_turn):
    tasks.add_task("brief me", "+0m")
    asyncio.run(scheduler.run_due())
    assert autonomy.spent()["today"] == 1500
    run = scheduler.recent_runs()[-1]
    assert run["kind"] == "task" and run["ok"] is True
    assert (run["tokens_in"], run["tokens_out"]) == (1200, 300)
    assert run["tools"] == ["risk_metrics"]
    assert "1,500 tok" in scheduler.describe_run(run)


def test_a_phone_turn_is_recorded_but_is_not_autonomous_spend(scripted_turn, monkeypatch):
    from financial_research_assistant import telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:a")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "9")
    monkeypatch.setattr(telegram, "get_updates", lambda timeout=0, commit=True: [
        {"chat_id": "9", "text": "how is NVDA?", "name": "me", "update_id": 4}])
    monkeypatch.setattr(telegram, "send_message", lambda text, chat_id="": True)
    asyncio.run(scheduler.poll_inbox())
    assert autonomy.spent()["today"] == 0, "the user asked: chat is never autonomous spend"
    run = scheduler.recent_runs()[-1]
    assert run["kind"] == "chat" and run["tokens_in"] == 1200


def test_a_job_run_is_recorded_with_its_files(monkeypatch):
    from financial_research_assistant import jobs

    async def handler(task, fake):
        return jobs.JobResult(True, "report", files=["/r/w.pdf"])

    monkeypatch.setitem(jobs._JOBS, "rec", handler)
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": (["t"], []))
    monkeypatch.setattr(channels, "deliver_file",
                        lambda path, caption="", prefer="", full_quality=False: (["t"], []))
    tasks.add_task("[job] rec", "+0m", kind="job", job="rec")
    asyncio.run(scheduler.run_due())
    run = scheduler.recent_runs()[-1]
    assert run["kind"] == "job" and run["name"] == "rec" and run["files"] == ["/r/w.pdf"]
    assert "rec" in asyncio.run(scheduler.run_command("/runs"))


# --- stale positions -------------------------------------------------------------------------


def test_a_report_on_an_old_book_says_so():
    book = {"as_of": "2026-08-07", "holdings": []}
    warning = periodic.staleness(book, date(2026, 9, 29))
    assert "2026-08-07 (53 days old)" in warning and "Flex sync" in warning
    assert periodic.staleness(book, date(2026, 8, 10)) == ""
    assert periodic.staleness({"as_of": ""}, date(2026, 9, 29)) == ""


def test_the_cli_switch(monkeypatch, capsys):
    import sys

    from financial_research_assistant import main

    # `cli()` loads .env first; in a test that would pull the developer's real
    # settings (memory dir, tokens) into the process for every test after it.
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["fra", "--pause", "on holiday"])
    with pytest.raises(SystemExit):
        main.cli()
    assert "paused" in guardrails.paused_reason()
    monkeypatch.setattr(sys, "argv", ["fra", "--resume-autonomy"])
    with pytest.raises(SystemExit):
        main.cli()
    assert guardrails.paused_reason() == ""
