"""Suite-wide fixtures."""

import pytest

from financial_research_assistant import alerts


@pytest.fixture(autouse=True)
def _never_actually_play_audio(monkeypatch):
    """Stop any test from making noise.

    Alert delivery ends in a real audio player, so a test that fires a rule
    would otherwise play a sound on whoever's running the suite — and on CI,
    spawn a process that isn't going anywhere. Stubbing the spawn keeps the
    surrounding logic (enabled? which file? which player?) under test while the
    only thing lost is the subprocess itself. Tests that care about it assert on
    the recorded commands instead.
    """
    played: list[list[str]] = []
    monkeypatch.setattr(alerts, "_spawn", played.append)
    return played
