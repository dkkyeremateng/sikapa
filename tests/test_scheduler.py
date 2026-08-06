"""The runner: due tasks run, answers get delivered, outcomes get recorded.

Offline — the model is the scripted fake (or a stubbed turn) and every channel is
a recorder, so no test here spends a token or sends a message.
"""

import asyncio

import pytest

from financial_research_assistant import channels, scheduler, tasks


@pytest.fixture
def delivered(monkeypatch):
    """Capture what would have been pushed, per channel preference."""
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        channels, "deliver", lambda text, prefer="": (sent.append((prefer, text)), (["test"], []))[1]
    )
    return sent


@pytest.fixture
def answers(monkeypatch):
    """Replace the model turn with a scripted (ok, text) reply per prompt."""
    seen: list[tuple[str, str]] = []
    script: dict[str, tuple[bool, str]] = {}

    async def fake_answer(prompt, session_id, fake=False):
        seen.append((prompt, session_id))
        # A task run is framed with _TASK_PREAMBLE; script and answer off the task's
        # own text so the tests below read as what the user actually scheduled.
        bare = prompt.rsplit("\n\n", 1)[-1]
        return script.get(bare, (True, f"answer to {bare}"))

    monkeypatch.setattr(scheduler, "_answer", fake_answer)
    return {"seen": seen, "script": script}


def test_a_due_task_runs_and_its_answer_is_delivered(answers, delivered):
    tasks.add_task("Analyse NOMD", "+0m")
    results = asyncio.run(scheduler.run_due())
    assert [r["ok"] for r in results] == [True]
    assert answers["seen"][0][0].endswith("Analyse NOMD")
    assert "answer to Analyse NOMD" in delivered[0][1]
    assert tasks.load_tasks()[0]["status"] == "done"


def test_a_scheduled_run_uses_the_tasks_own_session(answers, delivered):
    """Its own thread, so a scheduled run starts clean instead of inheriting
    whatever the interactive session was mid-conversation about."""
    t = tasks.add_task("brief me", "+0m")
    asyncio.run(scheduler.run_due())
    assert answers["seen"][0][1] == t["session"] == "task-s1"


def test_the_delivered_message_identifies_itself(answers, delivered):
    """A push arriving hours later on a phone must say what it is; an anonymous
    wall of analysis is unreadable in a notification list."""
    tasks.add_task("x", "+0m")
    asyncio.run(scheduler.run_due())
    assert delivered[0][1].startswith("🤖 task s1")


def test_a_tasks_channel_preference_is_passed_through(answers, delivered):
    tasks.add_task("x", "+0m", channel="telegram")
    asyncio.run(scheduler.run_due())
    assert delivered[0][0] == "telegram"


def test_nothing_due_costs_nothing(answers, delivered):
    tasks.add_task("later", "+2h")
    assert asyncio.run(scheduler.run_due()) == []
    assert answers["seen"] == [], "an empty tick must not touch the model"


def test_a_failed_run_is_recorded_and_retried(answers, delivered):
    answers["script"]["broken"] = (False, "no such ticker")
    tasks.add_task("broken", "+0m")
    results = asyncio.run(scheduler.run_due())
    assert results[0]["ok"] is False
    stored = tasks.load_tasks()[0]
    assert stored["status"] == "pending" and stored["attempts"] == 1
    assert "failed" in delivered[0][1], "a failure must be reported, not swallowed"


def test_an_exception_in_the_turn_does_not_take_down_the_tick(monkeypatch, delivered):
    """One bad task must not stop the others in the same tick from running."""
    async def explode(prompt, session_id, fake=False):
        if prompt.endswith("bad"):
            raise RuntimeError("model exploded")
        return True, "fine"

    monkeypatch.setattr(scheduler, "_answer", explode)
    tasks.add_task("bad", "+0m")
    tasks.add_task("good", "+0m")
    results = asyncio.run(scheduler.run_due())
    assert [r["ok"] for r in results] == [False, True]


def test_delivery_failure_does_not_fail_the_task(monkeypatch, answers):
    """The work is done and the model call is spent. Marking it failed would re-run
    the whole analysis on the next tick because a notification API was down."""
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": ([], ["telegram"]))
    tasks.add_task("x", "+0m")
    results = asyncio.run(scheduler.run_due())
    assert results[0]["ok"] is True
    assert tasks.load_tasks()[0]["status"] == "done"


def test_an_undeliverable_answer_still_reaches_stdout(monkeypatch, answers, capsys):
    """Last resort: a run whose answer went nowhere would otherwise be lost, and a
    cron entry redirecting stdout still captures it."""
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": ([], []))
    tasks.add_task("x", "+0m")
    asyncio.run(scheduler.run_due())
    assert "answer to x" in capsys.readouterr().out


def test_a_recurring_task_is_queued_again_after_running(answers, delivered):
    tasks.add_task("daily brief", "+0m", repeat="daily")
    asyncio.run(scheduler.run_due())
    stored = tasks.load_tasks()[0]
    assert stored["status"] == "pending" and stored["runs"] == 1
    assert asyncio.run(scheduler.run_due()) == [], "it must not run twice in one day"


def test_tasks_run_one_at_a_time(monkeypatch, delivered):
    """Concurrent runs would multiply the token spend and hammer the same rate-
    limited market-data endpoints."""
    live = 0
    peak = 0

    async def track(prompt, session_id, fake=False):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return True, "ok"

    monkeypatch.setattr(scheduler, "_answer", track)
    for i in range(3):
        tasks.add_task(f"job {i}", "+0m")
    asyncio.run(scheduler.run_due())
    assert peak == 1


def test_watch_runs_the_same_pass_as_run_due(answers, delivered):
    """A thin loop on purpose: the moment the daemon grows its own scheduling
    logic, cron and --watch start disagreeing about what 'due' means."""
    tasks.add_task("x", "+0m")
    asyncio.run(scheduler.watch(interval=0, iterations=1))
    assert tasks.load_tasks()[0]["status"] == "done"


def test_watch_survives_a_failing_tick(monkeypatch):
    calls = {"n": 0}

    async def flaky(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("transient")

    monkeypatch.setattr(scheduler, "run_due", flaky)
    asyncio.run(scheduler.watch(interval=0, iterations=2))
    assert calls["n"] == 2, "a bad tick must not stop the watch"


# --- inbound -------------------------------------------------------------------


@pytest.fixture
def inbox(monkeypatch):
    """A configured, allowlisted bot with a scripted inbox and recorded replies."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "999")
    from financial_research_assistant import telegram

    replies: list[tuple[str, str]] = []
    monkeypatch.setattr(
        telegram, "send_message",
        lambda text, chat_id="": bool(replies.append((chat_id, text)) or True),
    )
    return {"telegram": telegram, "replies": replies, "monkeypatch": monkeypatch}


def test_an_inbound_message_is_answered_in_a_per_chat_session(inbox, answers):
    inbox["monkeypatch"].setattr(
        inbox["telegram"], "get_updates",
        lambda timeout=0: [{"chat_id": "999", "text": "how is NVDA?", "name": "me"}],
    )
    asyncio.run(scheduler.poll_inbox())
    assert answers["seen"][0] == ("how is NVDA?", "telegram-999")
    assert inbox["replies"][0] == ("999", "answer to how is NVDA?")


def test_chat_commands_are_answered_without_a_model_call(inbox, answers):
    """They must work with no credential configured, and must never cost a turn."""
    tasks.add_task("watch AAPL", "+2h")
    inbox["monkeypatch"].setattr(
        inbox["telegram"], "get_updates",
        lambda timeout=0: [{"chat_id": "999", "text": "/tasks", "name": "me"}],
    )
    asyncio.run(scheduler.poll_inbox())
    assert answers["seen"] == []
    assert "watch AAPL" in inbox["replies"][0][1]


def test_cancel_command_removes_a_task(inbox, answers):
    tasks.add_task("watch AAPL", "+2h")
    inbox["monkeypatch"].setattr(
        inbox["telegram"], "get_updates",
        lambda timeout=0: [{"chat_id": "999", "text": "/cancel s1", "name": "me"}],
    )
    asyncio.run(scheduler.poll_inbox())
    assert tasks.load_tasks() == []


def test_polling_is_skipped_entirely_when_inbound_is_off(monkeypatch, answers):
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    assert asyncio.run(scheduler.poll_inbox()) == []
    assert answers["seen"] == []


# --- the real seam -------------------------------------------------------------


def test_a_task_runs_through_the_real_turn_machinery(delivered):
    """Everything above stubs `_answer`, which would hide a break in the one join
    that matters: a scheduled task is an ordinary `adapter.run_turn` with its own
    session. This runs the whole path against the scripted fake model."""
    tasks.add_task("what is the price of AAPL?", "+0m")
    results = asyncio.run(scheduler.run_due(fake=True))
    assert results[0]["ok"] is True
    assert results[0]["answer"].strip(), "the fake model produced no answer"
    assert tasks.load_tasks()[0]["status"] == "done"
    assert delivered, "the answer was never handed to a channel"


# --- tool wiring ---------------------------------------------------------------


def test_schedule_task_is_always_available_but_the_rest_is_gated():
    """`schedule_task` is the bootstrap tool: gating it on 'a task already exists'
    would mean the agent could never create its first one. The management tools
    only appear once there is something to manage."""
    from financial_research_assistant import catalog

    names = {catalog.tool_name(t) for t in catalog.active_tools()}
    assert "schedule_task" in names
    assert "list_scheduled_tasks" not in names

    tasks.add_task("something", "+2h")
    names = {catalog.tool_name(t) for t in catalog.active_tools()}
    assert {"schedule_task", "list_scheduled_tasks", "cancel_scheduled_task"} <= names


def test_a_scheduled_run_is_told_it_cannot_see_the_conversation(answers, delivered):
    """Observed in the wild: a task written as a back-reference ('summarise the
    analysis we did') ran in its fresh session, replied 'I have no record of that
    research', and that non-answer was delivered to the user's phone as a success."""
    tasks.add_task("Send the user a summary of the FISV analysis report", "+0m")
    asyncio.run(scheduler.run_due())
    sent_prompt = answers["seen"][0][0]
    assert "SCHEDULED TASK" in sent_prompt
    assert "PRODUCE that work from scratch" in sent_prompt
    assert sent_prompt.endswith("Send the user a summary of the FISV analysis report")


def test_the_framing_is_not_written_into_the_stored_task(answers, delivered):
    """`--tasks` must show what the user asked for, not our preamble."""
    tasks.add_task("Analyse FISV", "+0m")
    asyncio.run(scheduler.run_due())
    assert tasks.load_tasks()[0]["prompt"] == "Analyse FISV"


def test_an_inbound_chat_message_gets_no_task_framing(inbox, answers):
    """A person is on the other end there — telling it nobody can answer a follow-up
    would be false, and would suppress a reasonable clarifying question."""
    inbox["monkeypatch"].setattr(
        inbox["telegram"], "get_updates",
        lambda timeout=0: [{"chat_id": "999", "text": "how is NVDA?", "name": "me"}],
    )
    asyncio.run(scheduler.poll_inbox())
    assert answers["seen"][0][0] == "how is NVDA?"
