"""Stock-screener tests.

The screener's every network access goes through the mockable ``_fetch_*``
helpers (``tools._fetch_daily``, ``fundamentals._fetch_info`` /
``_fetch_earnings_history``), so the whole multi-criteria screen is exercised
offline here by monkeypatching those on their source modules — the screener
imports them lazily at call time, so patching the module attribute takes effect.
"""

from financial_research_assistant import fundamentals, screener, tools


# Ten aligned trading sessions shared by the stock and the market index, so a
# near-high day lines up with a market return on the same date.
_DATES = [f"2026-07-{d:02d}" for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]

# WINNER: high of 100 (set early), and on the LAST session it closes 99.5 —
# 0.5% from the high, i.e. within 1%.
_WINNER = list(zip(_DATES, [100, 92, 95, 94, 96, 97, 98, 97, 98, 99.5]))
# LAGGARD: same 100 high early on, but the recent window sits ~15% below it.
_LAGGARD = list(zip(_DATES, [100, 96, 90, 88, 86, 85, 85, 84, 85, 85]))
# SPY: engineered so the last session's close-to-close return is −0.8% (a
# down-market day), and the prior sessions are flat/up.
_SPY = list(zip(_DATES, [500, 501, 502, 501, 503, 504, 505, 504, 500, 496.0]))


def _install(monkeypatch, series_by_symbol, info_by_symbol, earnings_by_symbol=None):
    earnings_by_symbol = earnings_by_symbol or {}

    def fake_daily(symbol, days):
        return list(series_by_symbol.get(symbol.upper(), []))

    def fake_info(symbol):
        return dict(info_by_symbol.get(symbol.upper(), {}))

    def fake_earnings(symbol, limit=6):
        return list(earnings_by_symbol.get(symbol.upper(), []))

    monkeypatch.setattr(tools, "_fetch_daily", fake_daily)
    monkeypatch.setattr(fundamentals, "_fetch_info", fake_info)
    monkeypatch.setattr(fundamentals, "_fetch_earnings_history", fake_earnings)


def test_no_criteria_prompts_for_at_least_one(monkeypatch):
    _install(monkeypatch, {}, {})
    out = screener.screen_stocks(symbols="AAPL")
    assert "No screening criteria set" in out


def test_market_cap_bound_filters_universe(monkeypatch):
    info = {
        "BIG": {"longName": "Big Co", "marketCap": 2_000e9, "sector": "Technology"},
        "SMALL": {"longName": "Small Co", "marketCap": 5e9, "sector": "Technology"},
    }
    _install(monkeypatch, {}, info)
    out = screener.screen_stocks(symbols="BIG, SMALL", min_market_cap_b=10)
    assert "BIG" in out and "Big Co" in out
    assert "SMALL" not in out.split("QUALITATIVE")[0]  # excluded from the results
    assert "1 passed" in out


def test_example_query_near_ath_down_market_and_beats(monkeypatch):
    """The example screen: within 1% of the (all-time) high in the last 2 weeks,
    on a day the market fell ≥0.5%, ≥2 consecutive EPS beats, cap > $10B."""
    info = {
        "WIN": {"longName": "Winner Inc", "marketCap": 800e9, "sector": "Technology",
                "currentPrice": 99.5},
        "LAG": {"longName": "Laggard Inc", "marketCap": 500e9, "sector": "Technology",
                "currentPrice": 85.0},
    }
    beats = [  # newest first; two straight beats then a miss
        {"reported": 2.1, "estimate": 1.8, "surprise": 16.0},
        {"reported": 1.9, "estimate": 1.7, "surprise": 11.0},
        {"reported": 1.0, "estimate": 1.2, "surprise": -16.0},
    ]
    _install(
        monkeypatch,
        {"WIN": _WINNER, "LAG": _LAGGARD, "SPY": _SPY},
        info,
        {"WIN": beats, "LAG": beats},
    )
    out = screener.screen_stocks(
        symbols="WIN, LAG",
        min_market_cap_b=10,
        near_high_pct=1.0,
        near_high_within_days=10,
        high_lookback_days=9000,
        market_down_pct=0.5,
        min_earnings_beats=2,
    )
    # WIN qualifies on every leg; LAG is ~15% off its high so it fails near-high.
    body = out.split("QUALITATIVE")[0]
    assert "WIN" in body and "Winner Inc" in body
    assert "LAG" not in body
    assert "1 passed" in out
    # The measured near-high figure and the down-market cross-reference show up.
    assert "0.50% from high on 2026-07-17" in out
    assert "mkt -0.80%" in out
    assert "2 EPS beat(s)" in out


def test_down_market_condition_excludes_up_market_day(monkeypatch):
    """Same near-high stock, but if the market was UP that day the relative-
    strength (down-market) condition rejects it."""
    up_spy = list(zip(_DATES, [500, 501, 502, 503, 504, 505, 506, 507, 508, 512.0]))
    info = {"WIN": {"longName": "Winner Inc", "marketCap": 800e9, "sector": "Tech"}}
    _install(monkeypatch, {"WIN": _WINNER, "SPY": up_spy}, info)
    out = screener.screen_stocks(
        symbols="WIN", near_high_pct=1.0, market_down_pct=0.5,
    )
    assert "0 passed" in out
    assert "No stocks in this universe met all the criteria" in out


def test_near_high_window_restricts_recency(monkeypatch):
    """A stock that was near its high early but has since fallen fails when the
    near-high window is short."""
    early = list(zip(_DATES, [99.5, 98, 97, 90, 88, 86, 85, 84, 85, 85]))  # near high on day 1 only
    info = {"OLD": {"longName": "Old High", "marketCap": 300e9}}
    _install(monkeypatch, {"OLD": early}, info)
    out = screener.screen_stocks(symbols="OLD", near_high_pct=1.0, near_high_within_days=3)
    assert "0 passed" in out


def test_beat_streak_stops_at_first_miss_and_skips_upcoming(monkeypatch):
    history = {
        "X": [
            {"reported": None, "estimate": 2.0},   # upcoming — skipped, not a break
            {"reported": 2.2, "estimate": 2.0},    # beat
            {"reported": 1.9, "estimate": 2.0},    # miss -> streak stops at 1
            {"reported": 3.0, "estimate": 1.0},    # (never counted)
        ],
    }
    _install(monkeypatch, {}, {}, history)
    assert screener._beat_streak("X", need=3) == 1


def test_default_universe_and_truncation_note(monkeypatch):
    """With no `symbols`, the built-in universe is used; max_symbols truncates it
    with a note. Everything Yahoo-less is skipped gracefully."""
    _install(monkeypatch, {}, {})  # no info for anyone -> all skipped
    out = screener.screen_stocks(min_market_cap_b=10, max_symbols=5)
    assert "built-in large-cap universe" in out
    assert "truncated to 5" in out
    assert "had no usable Yahoo data" in out


def test_qualitative_reminder_always_present(monkeypatch):
    info = {"BIG": {"longName": "Big Co", "marketCap": 2_000e9}}
    _install(monkeypatch, {}, info)
    out = screener.screen_stocks(symbols="BIG", min_market_cap_b=10)
    assert "QUALITATIVE criteria are NOT screened" in out
    assert "guidance" in out
    assert "not investment advice" in out.lower()


def test_named_universe_sp500_is_used(monkeypatch):
    """`universe='sp500'` resolves the constituents via the (mocked) keyless fetch
    and screens those tickers; the source label reflects the S&P 500."""
    monkeypatch.setattr(screener, "_UNIVERSE_CACHE", {}, raising=False)
    monkeypatch.setattr(screener, "_fetch_sp500", lambda: ["BIG", "SMALL"])
    # Re-point the named-universe table at the patched fetch.
    monkeypatch.setitem(screener._NAMED_UNIVERSES, "sp500", ("S&P 500", screener._fetch_sp500))
    info = {
        "BIG": {"longName": "Big Co", "marketCap": 2_000e9},
        "SMALL": {"longName": "Small Co", "marketCap": 5e9},
    }
    _install(monkeypatch, {}, info)
    out = screener.screen_stocks(universe="sp500", min_market_cap_b=10)
    assert "from the S&P 500" in out
    assert "BIG" in out and "SMALL" not in out.split("QUALITATIVE")[0]


def test_universe_name_normalization_matches_variants():
    for variant in ("sp500", "S&P 500", "s and p 500", "SP-500", "s&p500"):
        assert screener._norm_universe_name(variant) == "sp500"


def test_unknown_universe_returns_friendly_error(monkeypatch):
    _install(monkeypatch, {}, {})
    out = screener.screen_stocks(universe="russell5000", min_market_cap_b=10)
    assert "Unknown universe" in out and "sp500" in out


def test_named_universe_fetch_failure_is_graceful(monkeypatch):
    """When the S&P 500 source can't be fetched, the tool explains and points at
    fallbacks rather than screening an empty set."""
    monkeypatch.setitem(screener._NAMED_UNIVERSES, "sp500", ("S&P 500", lambda: []))
    _install(monkeypatch, {}, {})
    out = screener.screen_stocks(universe="sp500", min_market_cap_b=10)
    assert "Couldn't fetch the S&P 500" in out


def test_fetch_sp500_parses_ticker_column_and_normalizes(monkeypatch):
    """_fetch_sp500 reads the first CSV column and maps dotted tickers (BRK.B) to
    Yahoo's dash form (BRK-B)."""
    csv_text = (
        "Symbol,Security,GICS Sector\n"
        "MMM,3M,Industrials\n"
        "BRK.B,Berkshire Hathaway,Financials\n"
        "AAPL,Apple,Technology\n"
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return csv_text.encode()

    monkeypatch.setattr(screener, "_UNIVERSE_CACHE", {}, raising=False)
    monkeypatch.setattr(screener.urllib.request, "urlopen", lambda *a, **k: _Resp())
    syms = screener._fetch_sp500()
    assert syms == ["MMM", "BRK-B", "AAPL"]


def test_screen_stocks_registered_as_tool():
    names = {getattr(t, "__name__", "") for t in tools.TOOLS}
    assert "screen_stocks" in names
