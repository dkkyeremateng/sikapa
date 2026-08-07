from contextlib import asynccontextmanager
from datetime import date, timedelta


class RecordingWorker:
    """Stand-in for a running turn worker that records .cancel() calls."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeTool:
    """Minimal stand-in for a loaded MCP tool (only ``.name`` matters here)."""

    def __init__(self, name: str) -> None:
        self.name = name


def fake_ibkr_tools_session(*tool_names):
    """A stand-in ibkr_tools_session async CM yielding the named fake tools."""

    @asynccontextmanager
    async def _cm():
        yield [FakeTool(n) for n in tool_names]

    return _cm


#: The fake price source's "today". Fixed rather than the real clock, so fixtures
#: can name dates literally and read the same in any year.
FAKE_TODAY = date(2025, 3, 4)


def install_fake_price_fetch(monkeypatch):
    """Stub the Yahoo daily-close fetch with a deterministic series.

    It models three properties of the real fetch, and each earns its place:

    **Length follows ``days``.** The original returned a fixed 60 rows whatever was
    asked for, which made every window-sizing bug invisible — including a live one
    where ``risk_metrics(days=365, as_of=…)`` computed over 976 days because the
    widened request pulled a larger range bucket. A stub that ignores the argument
    controlling the window cannot test the window.

    **``as_of`` widens then trims**, via the same helpers the real fetch uses, so
    the trim is exercised rather than assumed.

    **Sessions are weekdays.** Rows and calendar days therefore diverge as they do
    in real data, which is what makes a by-row trim distinguishable from a by-date
    one.

    Prices vary gently so volatility, drawdown and beta are all well-defined.
    """
    import financial_research_assistant.tools as t
    from financial_research_assistant.pointintime import as_of_series, lookback_days

    def fake(sym, days, strict=False, as_of=None, **_kw):
        base = {"AAPL": 100.0, "MSFT": 50.0, "SPY": 400.0}.get(sym.upper(), 100.0)
        span = lookback_days(days, as_of)
        sessions = [
            d for i in range(span)
            if (d := FAKE_TODAY - timedelta(days=span - 1 - i)).weekday() < 5
        ]
        series = [
            (d.isoformat(), base * (1 + 0.02 * ((i % 7) - 3) / 100 + 0.0005 * i))
            for i, d in enumerate(sessions)
        ]
        return as_of_series(series, as_of, days)

    monkeypatch.setattr(t, "_fetch_daily", fake)


def install_stub_fx(monkeypatch, rates):
    """Stub the Yahoo FX fetch: rates maps currency -> USD-per-unit."""
    import financial_research_assistant.tools as t

    def fake(sym, days, strict=False, **_kw):
        # sym like "EURUSD=X" -> currency EUR
        ccy = sym.replace("USD=X", "")
        r = rates.get(ccy)
        return [("2025-01-01", r), ("2025-12-31", r)] if r else []

    monkeypatch.setattr(t, "_fetch_daily", fake)
