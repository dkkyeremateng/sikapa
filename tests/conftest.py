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
