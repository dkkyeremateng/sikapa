"""Point-in-time answers: exact where the data allows, refused where it doesn't.

The bug these guard is silent, which is what makes it worth this many tests: a
tool asked about January 2025 that returns January 2026's figures produces a
confident, well-formatted, wrong answer with nothing in it to signal the problem.
So the assertions are mostly about what does NOT appear in the output.
"""

from datetime import date, timedelta

import pytest

from financial_research_assistant import pointintime as pit

from .helpers.fakes import install_fake_price_fetch as _fake_fetch


def _sessions(out: str) -> int:
    """The session count a tool reports in its header — the window it actually
    computed over, which is what an as_of has to change."""
    import re

    found = re.search(r"\((\d+) sessions\)", out)
    assert found, f"no session count in: {out[:120]}"
    return int(found.group(1))


#: The regression needs ~30 overlapping days, and an as_of cut has to leave enough
#: on BOTH sides to be visible — so the factor fixtures run over a long, dense,
#: real-calendar window rather than the short shared price fake.
_FF_START = date(2025, 1, 1)
_FF_DAYS = 200
_FF_CUT = "2025-03-01"


def _ff_dates() -> list[str]:
    return [(_FF_START + timedelta(days=i)).isoformat() for i in range(_FF_DAYS)]


def _fake_factor_prices(monkeypatch):
    """A long daily series, so cutting it still leaves a runnable regression."""
    import financial_research_assistant.tools as t

    series = [(d, 100.0 + i * 0.4 + (i % 7)) for i, d in enumerate(_ff_dates())]

    def fake(sym, days, strict=False, as_of=None, **_kw):
        return pit.as_of_series(series, as_of)

    monkeypatch.setattr(t, "_fetch_daily", fake)


def _fake_factors(monkeypatch):
    """Stub the Ken French download with factors covering the same window.

    Only the DATES matter here: the regression intersects the return series with
    these, so the day count in the output is what the as_of cut produced.
    """
    from financial_research_assistant import factors

    names = ["Mkt-RF", "SMB", "HML"]
    ff = {
        d: {"Mkt-RF": 0.001 * ((i % 5) - 2), "SMB": 0.0005, "HML": 0.0, "RF": 0.0}
        for i, d in enumerate(_ff_dates())
    }
    monkeypatch.setattr(
        factors, "_fetch_ff_factors", lambda five_factor=False: (names, ff)
    )


def _days(out: str) -> int:
    """The number of days the regression reports running over."""
    import re

    found = re.search(r"· (\d+) days", out)
    assert found, f"no day count in: {out[:120]}"
    return int(found.group(1))


def _window(out: str) -> tuple[str, str]:
    """The ``(start, end)`` dates of the window a tool reports covering."""
    import re

    found = re.findall(r"\d{4}-\d{2}-\d{2}", out.split("(")[0])
    assert len(found) >= 2, f"no window in: {out[:120]}"
    return found[0], found[-1]


def _window_end(out: str) -> str:
    return _window(out)[1]


def _assert_window_moved_not_shrunk(dated: str, current: str, cut: str) -> None:
    """The shared check for a tool that honours ``as_of``.

    Three properties, and the third is the one that took a live run to get right.
    The window must END at or before the cut; it must also START earlier, proving
    the window MOVED rather than being truncated at the front; and it must be the
    SAME LENGTH, because `days` means the same span whether or not a date was
    passed. An earlier version of these tests asserted the dated window was
    SHORTER — which passed only because the fetch was over-fetching and the tools
    were silently computing on more history than they asked for.
    """
    d_start, d_end = _window(dated)
    c_start, c_end = _window(current)
    assert c_end > cut, "the fixture must span past the cut, or this proves nothing"
    assert d_end <= cut, f"window ends {d_end}, after the cut {cut}"
    assert d_start < c_start, "the window must move back, not just lose its tail"
    assert abs(_sessions(dated) - _sessions(current)) <= 2, (
        f"`days` must mean the same span either way: "
        f"{_sessions(dated)} vs {_sessions(current)} sessions"
    )


# --- parsing --------------------------------------------------------------------


def test_an_empty_as_of_means_now():
    assert pit.parse_as_of("") is None
    assert pit.parse_as_of(None) is None
    assert pit.parse_as_of("   ") is None


def test_a_date_parses():
    assert pit.parse_as_of("2025-01-31") == date(2025, 1, 31)
    assert pit.parse_as_of("2025-01-31T00:00:00") == date(2025, 1, 31)


def test_an_unreadable_date_raises_rather_than_falling_back_to_today():
    """Falling back is the whole bug: the caller asked about the past, and
    answering about the present with no error is how a wrong figure reaches a
    report untraceably."""
    with pytest.raises(pit.AsOfError, match="YYYY-MM-DD"):
        pit.parse_as_of("January 2025")
    with pytest.raises(pit.AsOfError):
        pit.parse_as_of("2025-13-45")


def test_a_future_as_of_is_refused():
    ahead = (date.today() + timedelta(days=30)).isoformat()
    with pytest.raises(pit.AsOfError, match="future"):
        pit.parse_as_of(ahead)


def test_today_itself_is_allowed():
    assert pit.parse_as_of(date.today().isoformat()) == date.today()


# --- cutting the series ---------------------------------------------------------


_SERIES = [("2025-01-02", 10.0), ("2025-01-03", 11.0), ("2025-06-01", 12.0)]


def test_no_as_of_leaves_the_series_alone():
    assert pit.as_of_series(_SERIES, None) == _SERIES


def test_rows_after_the_date_are_dropped():
    assert pit.as_of_series(_SERIES, date(2025, 1, 3)) == _SERIES[:2]


def test_a_date_with_no_session_falls_back_to_the_one_before():
    """A weekend, holiday or halt has no row. The last session that did trade is
    the honest answer — and the caller reports which one it was."""
    cut = pit.as_of_series(_SERIES, date(2025, 3, 15))
    assert cut == _SERIES[:2]
    assert pit.window_note(date(2025, 3, 15), "2025-01-03").startswith(
        " · AS OF 2025-03-15 (last session on or before it: 2025-01-03)"
    )


def test_a_date_before_the_series_yields_nothing():
    assert pit.as_of_series(_SERIES, date(2024, 1, 1)) == []


def test_the_fetch_window_widens_by_the_gap():
    """Yahoo's window always ends today, so 180 days as of two years ago would cut
    to nothing without widening the request first."""
    assert pit.lookback_days(180, None) == 180
    gap = 400
    widened = pit.lookback_days(180, date.today() - timedelta(days=gap))
    assert widened >= 180 + gap


def test_the_widening_is_undone_by_the_trim():
    """The widened request reaches a bigger range bucket, so without trimming, a
    caller that uses the series WHOLE computes over far more history than it asked
    for — seen live as a 365-day regression covering 976 days."""
    long_series = [
        ((date(2024, 1, 1) + timedelta(days=i)).isoformat(), 100.0 + i)
        for i in range(900)
    ]
    cut = pit.as_of_series(long_series, date(2025, 6, 30), days=365)
    assert cut[-1][0] == "2025-06-30"
    assert cut[0][0] == "2024-07-01", "the window starts `days` calendar days back"
    # Without `days`, the whole cut history comes back — the old behaviour, still
    # correct for callers that slice it themselves.
    assert len(pit.as_of_series(long_series, date(2025, 6, 30))) > len(cut)


def test_the_trim_is_by_date_so_days_means_the_same_thing_either_way():
    """`days` picks a Yahoo range bucket when there is no as_of, so it is CALENDAR
    days — `days=365` is the ~252 sessions in a year. Trimming to the last 365 ROWS
    would make the same argument mean ~17 months as soon as a date was passed."""
    # A five-day trading week, so rows and calendar days diverge the way they do
    # in real data.
    sessions = [
        d for i in range(900)
        if (d := date(2024, 1, 1) + timedelta(days=i)).weekday() < 5
    ]
    series = [(d.isoformat(), 100.0 + i) for i, d in enumerate(sessions)]
    cut = pit.as_of_series(series, date(2025, 6, 30), days=365)
    assert 240 <= len(cut) <= 265, f"a year of weekdays, got {len(cut)} rows"


def test_the_trim_never_touches_a_current_request():
    """No as_of means no widening happened, so nothing may be trimmed — callers
    without a date must behave exactly as before."""
    series = [(f"2025-01-{i + 1:02d}", 100.0 + i) for i in range(28)]
    assert pit.as_of_series(series, None, days=5) == series


def test_a_dated_fetch_returns_at_most_the_days_requested(monkeypatch):
    """End to end through `_fetch_daily`, with a fake that models the real thing:
    it returns a whole range bucket, widening as the request widens. The shared
    price fake returns a fixed-length series regardless of `days`, which is why it
    could not have caught this."""
    import financial_research_assistant.tools as t

    def bucketed(sym, days, strict=False, as_of=None, **_kw):
        # A bucket at least as long as asked for, ending today — as Yahoo does.
        span = pit.lookback_days(days, as_of)
        series = [
            ((date.today() - timedelta(days=span - i)).isoformat(), 100.0 + i)
            for i in range(span)
        ]
        return pit.as_of_series(series, as_of, days)

    monkeypatch.setattr(t, "_fetch_daily", bucketed)
    as_of = date.today() - timedelta(days=400)
    cut = t._fetch_daily("AAPL", 365, as_of=as_of)
    assert cut[-1][0] <= as_of.isoformat()
    assert cut[0][0] > (as_of - timedelta(days=365)).isoformat()
    assert len(cut) <= 366, f"a 365-day window returned {len(cut)} rows"


# --- the note -------------------------------------------------------------------


def test_a_current_answer_carries_no_note():
    assert pit.window_note(None) == ""


def test_a_dated_answer_always_says_so():
    """The figures look identical either way. If the output doesn't say it is
    as-of, nothing does."""
    note = pit.window_note(date(2025, 1, 3), "2025-01-03")
    assert "AS OF 2025-01-03" in note
    assert "later data deliberately excluded" in note


# --- price-derived tools honour it ----------------------------------------------


def test_price_history_chart_excludes_later_sessions(monkeypatch):
    import financial_research_assistant.tools as t

    _fake_fetch(monkeypatch)
    dated = t.price_history_chart("AAPL", days=60, as_of="2025-01-20")
    current = t.price_history_chart("AAPL", days=60)
    assert "AS OF 2025-01-20" in dated
    _assert_window_moved_not_shrunk(dated, current, "2025-01-20")


def test_risk_metrics_excludes_later_sessions(monkeypatch):
    import financial_research_assistant.tools as t

    _fake_fetch(monkeypatch)
    dated = t.risk_metrics("AAPL", days=60, as_of="2025-01-28")
    current = t.risk_metrics("AAPL", days=60)
    assert "AS OF 2025-01-28" in dated
    assert "AS OF" not in current
    # Assert on the WINDOW, not on the strings differing: the note alone makes them
    # differ, so `dated != current` would pass even if as_of never reached the
    # fetch.
    _assert_window_moved_not_shrunk(dated, current, "2025-01-28")
    # And the metrics really are computed over it, not merely labelled with it.
    assert "Sharpe" in dated and dated != current


def test_compare_prices_excludes_later_sessions(monkeypatch):
    import financial_research_assistant.tools as t

    _fake_fetch(monkeypatch)
    dated = t.compare_prices("AAPL, SPY", days=60, as_of="2025-01-20")
    current = t.compare_prices("AAPL, SPY", days=60)
    assert "AS OF 2025-01-20" in dated
    _assert_window_moved_not_shrunk(dated, current, "2025-01-20")


def test_correlation_matrix_excludes_later_sessions(monkeypatch):
    from financial_research_assistant import analytics

    _fake_fetch(monkeypatch)
    dated = analytics.correlation_matrix("AAPL, SPY", days=60, as_of="2025-01-20")
    current = analytics.correlation_matrix("AAPL, SPY", days=60)
    assert "AS OF 2025-01-20" in dated
    _assert_window_moved_not_shrunk(dated, current, "2025-01-20")


def test_factor_exposure_regresses_only_over_the_window(monkeypatch):
    """The regression is over daily returns, so cutting the series cuts the window
    the loadings are estimated from."""
    from financial_research_assistant import factors

    _fake_factor_prices(monkeypatch)
    _fake_factors(monkeypatch)
    dated = factors.factor_exposure("AAPL", days=_FF_DAYS, as_of=_FF_CUT)
    current = factors.factor_exposure("AAPL", days=_FF_DAYS)
    assert f"AS OF {_FF_CUT}" in dated
    assert "AS OF" not in current
    assert _days(dated) < _days(current), "the regression must use fewer days"


def test_a_dated_portfolio_regression_says_the_weights_are_current(monkeypatch):
    """The returns are point-in-time; the weights are whatever is held NOW, because
    the statement store has no position history. An unlabelled answer would read as
    'this is what you held then', which it is not."""
    from financial_research_assistant import factors

    _fake_factor_prices(monkeypatch)
    _fake_factors(monkeypatch)
    # The portfolio branch reads the statement store; stub it to the same series
    # shape so this test is about the caveat, not about holdings plumbing. It
    # still honours as_of, so a regression really does run over the cut window.
    def portfolio_returns(days, account=None, as_of=None):
        rows = pit.as_of_series([(d, 0.0) for d in _ff_dates()], as_of)
        return [(d, 0.001 * ((i % 5) - 2)) for i, (d, _) in enumerate(rows)]

    monkeypatch.setattr(factors, "_portfolio_returns", portfolio_returns)
    out = factors.factor_exposure("", days=_FF_DAYS, as_of=_FF_CUT)
    assert "CURRENT holdings" in out
    # A ticker regression has no weights, so it must not carry the caveat.
    dated_ticker = factors.factor_exposure("AAPL", days=_FF_DAYS, as_of=_FF_CUT)
    assert "CURRENT holdings" not in dated_ticker
    # Nor does an undated portfolio one — the caveat is about MIXING eras.
    assert "CURRENT holdings" not in factors.factor_exposure("", days=_FF_DAYS)


def test_a_bad_as_of_is_reported_by_the_tool_not_raised(monkeypatch):
    """Tools return strings; a parse failure has to reach the model as a result it
    can act on, not as an exception that kills the turn."""
    import financial_research_assistant.tools as t

    _fake_fetch(monkeypatch)
    out = t.price_history_chart("AAPL", days=60, as_of="last January")
    assert "YYYY-MM-DD" in out
    assert "AS OF" not in out


# --- snapshot tools refuse ------------------------------------------------------


def test_a_snapshot_tool_refuses_and_names_an_alternative():
    """`stock_fundamentals` reads a current Yahoo record. There is no 2025 version
    of it to return, so the only honest reply is a redirect."""
    out = pit.snapshot_guard("stock_fundamentals", "2025-01-31")
    assert out is not None
    assert "cannot answer as of 2025-01-31" in out
    assert "sec_financials" in out, "the refusal must name a source that CAN answer"


def test_a_snapshot_tool_with_no_as_of_just_runs():
    assert pit.snapshot_guard("stock_fundamentals", "") is None
    assert pit.snapshot_guard("stock_fundamentals", None) is None


def test_every_guarded_tool_has_a_redirect():
    """A refusal that says only "no" sends the model round the loop again. Each
    guarded tool must name where to go instead."""
    for tool in pit.REDIRECTS:
        out = pit.snapshot_guard(tool, "2025-01-31")
        assert out and "For a point-in-time answer use:" in out, tool


def test_the_snapshot_tools_are_wired_to_the_guard(monkeypatch):
    """The guard has to run BEFORE the network call, so a refusal costs nothing and
    a stubbed-out network can't mask a missing wire-up."""
    from financial_research_assistant import fundamentals, screener, valuation

    def explode(*_a, **_k):
        raise AssertionError("fetched data for a request that should have been refused")

    monkeypatch.setattr(fundamentals, "_fetch_info", explode)
    monkeypatch.setattr(fundamentals, "_fetch_fund_data", explode)

    assert "cannot answer as of" in fundamentals.stock_fundamentals("AAPL", as_of="2025-01-31")
    assert "cannot answer as of" in fundamentals.analyst_ratings("AAPL", as_of="2025-01-31")
    assert "cannot answer as of" in fundamentals.compare_stocks("AAPL, MSFT", as_of="2025-01-31")
    assert "cannot answer as of" in fundamentals.etf_exposure("VOO", as_of="2025-01-31")
    assert "cannot answer as of" in valuation.dcf_valuation("AAPL", as_of="2025-01-31")
    assert "cannot answer as of" in screener.screen_stocks(symbols="AAPL", as_of="2025-01-31")
    assert "cannot answer as of" in fundamentals.dividend_projection(as_of="2025-01-31")


def test_a_forward_projection_redirects_to_what_was_actually_received():
    """`dividend_projection` looks FORWARD from today's holdings and today's rates,
    so there is no past version of it. The historical question is what was actually
    paid, which a different tool answers."""
    out = pit.snapshot_guard("dividend_projection", "2025-01-31")
    assert out and "income_summary" in out


def test_a_screen_says_it_is_not_a_backtest():
    out = pit.snapshot_guard("screen_stocks", "2025-01-31")
    assert out and "not a backtest" in out


def test_a_bad_as_of_reaches_a_snapshot_tool_as_a_message_too(monkeypatch):
    from financial_research_assistant import fundamentals

    def explode(*_a, **_k):
        raise AssertionError("fetched data despite an unreadable as_of")

    monkeypatch.setattr(fundamentals, "_fetch_info", explode)
    assert "YYYY-MM-DD" in fundamentals.stock_fundamentals("AAPL", as_of="early 2025")
