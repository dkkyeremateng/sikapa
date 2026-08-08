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


#: Yahoo answers in coarse ``range`` buckets, so a request never comes back the
#: length it was made: 400 days arrives as two years, and an all-time-high request
#: as everything there is. These are ``tools._yahoo_range``'s buckets expressed in
#: days, so the fake overshoots exactly where the real source does.
_RANGE_BUCKETS = (5, 30, 90, 180, 365, 730, 1825, 3650)


def _bucket_days(days: int) -> int:
    """The smallest range bucket covering ``days`` (36500 ≈ Yahoo's ``max``)."""
    return next((b for b in _RANGE_BUCKETS if days <= b), 36500)


def install_fake_price_fetch(monkeypatch):
    """Stub the Yahoo daily-close fetch with a deterministic series.

    It models four properties of the real fetch, and each earns its place:

    **The reply overshoots the request**, rounded up to a range bucket. This is the
    property that makes window bugs visible at all: a caller that consumes the
    series whole computes over the surplus, which was seen live as a 365-day
    regression covering 976 days and as a "52-week high" that was really a
    two-year high.

    **Length otherwise follows ``days``.** The original returned a fixed 60 rows
    whatever was asked for. A stub that ignores the argument controlling the
    window cannot test the window.

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
        span = _bucket_days(lookback_days(days, as_of))
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
    """Stub the Yahoo FX fetch: ``rates`` maps currency -> USD-per-unit.

    The rate itself is constant — these tests assert on converted amounts, and a
    drifting rate would only make the expected numbers unreadable. What is NOT
    constant is the series it lives in: the dates are generated from ``days``
    (and ``as_of``) back from ``FAKE_TODAY``, exactly as ``install_fake_price_fetch``
    does, because that is the argument ``_fx_series_days`` computes in order to
    reach a historical date. A two-point fixed series answers every lookup no
    matter how far back it asks, which is precisely how a window that falls short
    of the requested date stays invisible.
    """
    import financial_research_assistant.tools as t
    from financial_research_assistant.pointintime import as_of_series, lookback_days

    def fake(sym, days, strict=False, as_of=None, **_kw):
        # sym like "EURUSD=X" -> currency EUR
        ccy = sym.replace("USD=X", "")
        r = rates.get(ccy)
        if not r:
            return []
        span = lookback_days(days, as_of)
        series = [
            (d.isoformat(), r) for i in range(span)
            if (d := FAKE_TODAY - timedelta(days=span - 1 - i)).weekday() < 5
        ]
        return as_of_series(series, as_of, days)

    monkeypatch.setattr(t, "_fetch_daily", fake)
