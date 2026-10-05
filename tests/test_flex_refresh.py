"""A portfolio question pulls a fresh Flex statement when the one on file is old.

Live IBKR timed out and the agent answered "account overview" from a statement
five days old, without saying how old: `query_portfolio` gave no date, and the
model had no way to fetch Flex itself. Only the Flex download is faked here; the
statement store, the lock and the throttle are real.
"""

import threading
import time
from datetime import datetime, timezone

import pytest

from financial_research_assistant import flex, statements, tools

from .fixtures.statements import mini_statement

# A Monday: the newest statement Flex can have is Friday's.
MONDAY = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)
FRIDAY_PERIOD = "October 1, 2025 - October 2, 2026"
OLD_PERIOD = "September 30, 2025 - September 29, 2026"


@pytest.fixture
def flex_env(monkeypatch):
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "test-token")
    monkeypatch.setenv("IBKR_FLEX_QUERY_ID", "123")


@pytest.fixture
def pulls(monkeypatch):
    """Fake `flex_sync`: each call imports a statement ending on Friday."""
    calls: list[float] = []

    def fake_sync():
        calls.append(time.monotonic())
        time.sleep(0.2)  # long enough for a concurrent caller to arrive
        statements.import_statement(mini_statement("U1", FRIDAY_PERIOD, "1%", 1000))
        return "Fetched Flex statement for query 123 (1 bytes) → saved x.\nImported …"

    monkeypatch.setattr(flex, "flex_sync", fake_sync)
    return calls


def _on_file(period):
    statements.import_statement(mini_statement("U1", period, "1%", 900))


def test_an_old_statement_is_refreshed_and_the_answer_says_so(flex_env, pulls):
    _on_file(OLD_PERIOD)
    note = flex.refresh_if_stale(now=MONDAY)
    assert len(pulls) == 1
    assert note == ("Refreshed from IBKR Flex just now: positions as of 2026-10-02 "
                    "(the statement on file was from 2026-09-29).")
    assert statements.positions_as_of() == "2026-10-02"


def test_a_current_statement_is_not_fetched_again(flex_env, pulls):
    _on_file(FRIDAY_PERIOD)
    assert flex.refresh_if_stale(now=MONDAY) == ""
    assert pulls == []


def test_without_a_flex_token_nothing_is_attempted(pulls):
    _on_file(OLD_PERIOD)
    assert flex.refresh_if_stale(now=MONDAY) == ""
    assert pulls == []


def test_a_failed_pull_is_reported_and_not_retried_for_a_while(flex_env, monkeypatch):
    _on_file(OLD_PERIOD)
    calls = []
    monkeypatch.setattr(flex, "flex_sync", lambda: calls.append(1) or
                        "Flex sync error: URLError: <urlopen error [Errno -3] Temporary failure>")
    note = flex.refresh_if_stale(now=MONDAY)
    assert note.startswith("Couldn't refresh from IBKR Flex (Flex sync error: URLError")
    assert note.endswith("using the statement on file.")
    # IBKR throttles Flex: a question every minute must not become a pull a minute.
    assert flex.refresh_if_stale(now=MONDAY.replace(minute=20)) == ""
    assert flex.refresh_if_stale(now=MONDAY.replace(hour=16)) != ""
    assert len(calls) == 2


def test_two_questions_at_once_share_one_pull(flex_env, pulls, monkeypatch):
    """The agent can call query_portfolio and allocation in the same step. With
    the throttle off, only the lock stops the second caller from pulling too:
    it waits, then finds the statement fresh."""
    monkeypatch.setattr(flex, "REFRESH_EVERY_MINUTES", 0)
    _on_file(OLD_PERIOD)
    notes = []
    threads = [threading.Thread(target=lambda: notes.append(flex.refresh_if_stale(now=MONDAY)))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(pulls) == 1
    assert sorted(bool(n) for n in notes) == [False, True]


def test_query_portfolio_names_the_day_its_figures_describe(flex_env, monkeypatch):
    _on_file(OLD_PERIOD)
    monkeypatch.setattr(flex, "flex_sync", lambda: "Flex sync error: URLError: timed out")
    out = tools.query_portfolio()
    assert out.splitlines()[0].startswith("Couldn't refresh from IBKR Flex")
    assert "Positions as of 2026-09-29 (latest IBKR statement on file; not a live balance)." in out
    assert "NET ASSET VALUE" in out
