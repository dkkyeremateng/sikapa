"""Suite-wide fixtures."""

import pytest

from financial_research_assistant import alerts


@pytest.fixture(autouse=True)
def _never_read_the_real_credential_store(tmp_path_factory, monkeypatch):
    """Point every test at a throwaway auth.json.

    Without this a test that builds an LLM picks up whatever the developer is
    actually signed in with — which is how a live OAuth token ended up printed in
    a pytest assertion diff. Suite-wide and autouse because the leak happens in
    tests that have nothing to do with auth (any `_make_llm` call reaches the
    store now), so per-file fixtures cannot be relied on to cover it.
    """
    store = tmp_path_factory.mktemp("auth") / "auth.json"
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(store))


@pytest.fixture(autouse=True)
def _never_actually_play_audio(monkeypatch):
    """Stop any test from making noise or raising a desktop notification.

    Alert delivery ends in real subprocesses — an audio player and an OS
    notifier — so a test that fires a rule would otherwise play a sound and pop
    a banner on whoever's running the suite, and on CI spawn processes that
    aren't going anywhere. Stubbing the one spawn point keeps the surrounding
    logic (enabled? which file? which command?) under test while the only thing
    lost is the subprocess itself. Tests that care assert on the recorded
    commands instead.
    """
    spawned: list[list[str]] = []
    monkeypatch.setattr(alerts, "_spawn", spawned.append)
    return spawned


@pytest.fixture(autouse=True)
def _never_touch_the_real_task_store(tmp_path_factory, monkeypatch):
    """Point every test at a throwaway tasks.json, and unset the delivery channels.

    Two separate hazards, one fixture. A test that schedules something would
    otherwise queue work in the developer's real store — which a later `--run-due`
    would faithfully execute, spending real tokens on a test fixture's prompt. And
    a test that runs the scheduler would deliver to whatever channel the developer
    has configured, i.e. send a Telegram message from the suite.
    """
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_TASKS_FILE",
        str(tmp_path_factory.mktemp("tasks") / "tasks.json"),
    )
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_TELEGRAM_STATE",
        str(tmp_path_factory.mktemp("tg") / "offset.json"),
    )
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_CHAT_IDS"):
        monkeypatch.delenv(var, raising=False)
    # An unrecognised key selects no channel, so the default for the suite is
    # "deliver nowhere". Tests that exercise delivery set this themselves.
    monkeypatch.setenv("NOTIFY_CHANNELS", "none")
