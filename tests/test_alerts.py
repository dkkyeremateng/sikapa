"""Alert-rule store, evaluation, and digest integration — fully offline."""

import datetime as dt

import pytest

from financial_research_assistant import alerts


@pytest.fixture(autouse=True)
def _tmp_alerts(monkeypatch, tmp_path):
    """Isolate the alert store to a temp file so tests never touch the real one."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERTS_FILE", str(tmp_path / "alerts.json"))


def test_add_list_remove_and_validation():
    assert "Added alert a1" in alerts.add_alert("AAPL", "drop", 5)
    assert "Added alert a2" in alerts.add_alert("*", "move", 8)
    assert "Added alert a3" in alerts.add_alert("TSLA", "below", 200)
    # synonyms map to canonical kinds ("up" -> rise)
    assert "rises" in alerts.add_alert("MSFT", "up", 3)

    out = alerts.list_alerts()
    assert "AAPL drops ≥ 5%" in out
    assert "any holding moves ≥ 8%" in out
    assert "TSLA price at or below 200" in out
    assert "MSFT rises ≥ 3%" in out

    # validation
    assert "Unknown alert kind" in alerts.add_alert("X", "frobnicate", 1)
    assert "needs a specific symbol" in alerts.add_alert("*", "below", 100)
    assert "positive threshold" in alerts.add_alert("X", "drop", 0)
    # earnings defaults to 7 days when no value is given
    assert "within 7 day(s)" in alerts.add_alert("NVDA", "earnings", 0)

    # remove by id, unknown id, and clear-all
    assert "Removed alert a1" in alerts.remove_alert("a1")
    assert "No alert with id" in alerts.remove_alert("zzz")
    assert "Removed all" in alerts.remove_alert("all")
    assert "No alert rules set" in alerts.list_alerts()


def test_evaluate_alerts_fires_the_right_rules(monkeypatch):
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools

    def fake_daily(sym, days, **kw):
        return {
            "AAPL": [("2026-01-01", 100.0), ("2026-01-05", 90.0)],   # -10%
            "MSFT": [("2026-01-01", 100.0), ("2026-01-05", 106.0)],  # +6%
            "TSLA": [("2026-01-05", 180.0)],
        }.get(sym, [])

    monkeypatch.setattr(tools, "_fetch_daily", fake_daily)
    monkeypatch.setattr(fund, "_fetch_calendar",
                        lambda sym: {"Earnings Date": [dt.date(2026, 1, 8)]} if sym == "NVDA" else {})

    alerts.add_alert("AAPL", "drop", 5)       # a1 fires (down 10%)
    alerts.add_alert("MSFT", "rise", 5)       # a2 fires (up 6%)
    alerts.add_alert("TSLA", "below", 200)    # a3 fires (180 <= 200), even if not held
    alerts.add_alert("NVDA", "earnings", 7)   # a4 fires (earnings in 3d)
    alerts.add_alert("AAPL", "rise", 5)       # a5 does NOT fire (AAPL is down)

    fired = "\n".join(alerts.evaluate_alerts(["AAPL", "MSFT"], 5, dt.date(2026, 1, 5)))
    assert "[a1]" in fired and "AAPL down -10.0%" in fired
    assert "[a2]" in fired and "MSFT up +6.0%" in fired
    assert "[a3]" in fired and "TSLA at 180.00" in fired
    assert "[a4]" in fired and "NVDA reports earnings 2026-01-08" in fired
    assert "[a5]" not in fired


def test_evaluating_rules_buffers_them_for_notification(monkeypatch):
    """Firing a rule also records it for out-of-band delivery, so an interface
    can raise a notification the moment it triggers. The buffer drains once."""
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **kw: [("2026-01-01", 100.0), ("2026-01-05", 88.0)])
    monkeypatch.setattr(fund, "_fetch_calendar", lambda sym: {})
    alerts.drain_triggered()

    alerts.add_alert("AAPL", "drop", 5)
    alerts.evaluate_alerts(["AAPL"], 5, dt.date(2026, 1, 5))

    fired = alerts.drain_triggered()
    assert len(fired) == 1
    assert "AAPL down -12.0%" in fired[0]
    assert alerts.drain_triggered() == []  # drained, so it won't replay next turn


def test_notification_buffer_is_bounded(monkeypatch):
    """Nothing guarantees a drain (the eval harness ignores alert events), so the
    buffer caps rather than growing for the life of the process."""
    alerts.drain_triggered()
    for i in range(alerts._fired.maxlen + 25):
        alerts._fired.append(f"alert {i}")

    fired = alerts.drain_triggered()
    assert len(fired) == alerts._fired.maxlen
    assert fired[-1] == f"alert {alerts._fired.maxlen + 24}"  # newest kept


def test_evaluate_star_applies_to_each_holding(monkeypatch):
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **kw: [("2026-01-01", 100.0), ("2026-01-05", 88.0)])  # -12%
    monkeypatch.setattr(fund, "_fetch_calendar", lambda sym: {})
    alerts.add_alert("*", "move", 8)  # any holding moving ≥ 8%
    fired = alerts.evaluate_alerts(["AAPL", "MSFT"], 5, dt.date(2026, 1, 5))
    assert len(fired) == 2 and any("AAPL" in f for f in fired) and any("MSFT" in f for f in fired)


def test_digest_surfaces_triggered_alerts(monkeypatch, tmp_path):
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools
    from financial_research_assistant import monitor
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(
        "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,"
        "Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n"
        "Open Positions,Data,Summary,Stocks,USD,AAPL,1,1,100,100,100,100,0,\n"
    )
    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **kw: [("2026-01-01", 100.0), ("2026-01-05", 90.0)])
    monkeypatch.setattr(fund, "_fetch_calendar", lambda sym: {})

    # No rules yet → no alerts section at all.
    out0 = monitor.build_digest(today=dt.date(2026, 1, 5))
    assert "🔔" not in out0

    alerts.add_alert("AAPL", "drop", 5)
    out = monitor.build_digest(today=dt.date(2026, 1, 5))
    assert "## 🔔 Alerts triggered" in out
    assert "AAPL down -10.0%" in out


def test_alert_tools_registered():
    from financial_research_assistant.tools import TOOLS

    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in TOOLS}
    assert {"add_alert", "list_alerts", "remove_alert"} <= names
