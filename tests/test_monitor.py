"""The monitoring digest's view of the holdings it scans.

Offline — positions come from a stubbed store and both market-data fetches are
counters, so nothing here touches the network.
"""

import datetime as dt

import pytest

from financial_research_assistant import alerts, fundamentals, monitor, statements, tools


@pytest.fixture
def held(monkeypatch, tmp_path):
    """AAPL held in two accounts (plus MSFT), with every fetch counted."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERTS_FILE", str(tmp_path / "alerts.json"))
    monkeypatch.setattr(statements, "query_positions", lambda account=None: [
        {"account": "U1", "symbol": "AAPL", "quantity": 10},
        {"account": "U2", "symbol": "aapl", "quantity": 5},
        {"account": "U1", "symbol": "MSFT", "quantity": 3},
    ])
    fetched: list[str] = []

    def fake_daily(sym, days, **_kw):
        fetched.append(sym.upper())
        return [("2026-01-01", 100.0), ("2026-01-05", 90.0)]

    monkeypatch.setattr(tools, "_fetch_daily", fake_daily)
    monkeypatch.setattr(fundamentals, "_fetch_calendar", lambda sym: {})
    return fetched


def test_a_symbol_held_in_two_accounts_is_scanned_once(held):
    """The same holding in two accounts is one thing to look at. Scanning it twice
    doubles the price lookups against a rate-limited endpoint and lists the move
    twice, as if two different positions had moved."""
    out = monitor.build_digest(today=dt.date(2026, 1, 5), move_threshold=5.0)
    assert sorted(held) == ["AAPL", "MSFT"]
    assert out.count("AAPL") == 1
    assert "2 holding(s)" in out


def test_a_wildcard_alert_rule_fires_once_per_symbol_not_once_per_lot(held):
    """A duplicate symbol used to fire every ``*`` rule again — the same 10% drop
    reported twice, which reads as two separate events."""
    alerts.add_alert("*", "drop", 5)
    out = monitor.build_digest(today=dt.date(2026, 1, 5), move_threshold=5.0)
    fired = [ln for ln in out.splitlines() if ln.startswith("- ") and "AAPL down" in ln]
    assert len(fired) == 1, fired


def test_the_scanned_list_keeps_first_seen_order(held, monkeypatch):
    """The digest's ordering is the statement's, so de-duplicating must not quietly
    reorder the holdings (a set would)."""
    captured: list[list[str]] = []

    def spy(symbols, *_args):
        captured.append(list(symbols))
        return [], [], []

    monkeypatch.setattr(monitor, "_scan_holdings", spy)
    monitor.build_digest(today=dt.date(2026, 1, 5))
    assert captured[0] == ["AAPL", "MSFT"]
