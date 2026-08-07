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


def _window_end(out: str) -> str:
    """The last date in the window a tool reports covering."""
    import re

    found = re.findall(r"\d{4}-\d{2}-\d{2}", out.split("(")[0])
    assert found, f"no window dates in: {out[:120]}"
    return found[-1]


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
    assert _window_end(current) > "2025-01-20", "fixture must span past the cut"
    assert _window_end(dated) <= "2025-01-20"
    assert _sessions(dated) < _sessions(current)


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
    assert _window_end(current) > "2025-01-28", (
        "the fixture must span past the cut, or this test proves nothing"
    )
    assert _window_end(dated) <= "2025-01-28"
    # And the metrics are computed over the shorter window, not merely labelled.
    assert _sessions(dated) < _sessions(current)


def test_compare_prices_excludes_later_sessions(monkeypatch):
    import financial_research_assistant.tools as t

    _fake_fetch(monkeypatch)
    dated = t.compare_prices("AAPL, SPY", days=60, as_of="2025-01-20")
    current = t.compare_prices("AAPL, SPY", days=60)
    assert "AS OF 2025-01-20" in dated
    assert _window_end(current) > "2025-01-20", "fixture must span past the cut"
    assert _window_end(dated) <= "2025-01-20"
    assert _sessions(dated) < _sessions(current)


def test_correlation_matrix_excludes_later_sessions(monkeypatch):
    from financial_research_assistant import analytics

    _fake_fetch(monkeypatch)
    dated = analytics.correlation_matrix("AAPL, SPY", days=60, as_of="2025-01-20")
    current = analytics.correlation_matrix("AAPL, SPY", days=60)
    assert "AS OF 2025-01-20" in dated
    assert _window_end(current) > "2025-01-20", "fixture must span past the cut"
    assert _window_end(dated) <= "2025-01-20"
    assert _sessions(dated) < _sessions(current)


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


def test_a_screen_says_it_is_not_a_backtest():
    out = pit.snapshot_guard("screen_stocks", "2025-01-31")
    assert out and "not a backtest" in out


def test_a_bad_as_of_reaches_a_snapshot_tool_as_a_message_too(monkeypatch):
    from financial_research_assistant import fundamentals

    def explode(*_a, **_k):
        raise AssertionError("fetched data despite an unreadable as_of")

    monkeypatch.setattr(fundamentals, "_fetch_info", explode)
    assert "YYYY-MM-DD" in fundamentals.stock_fundamentals("AAPL", as_of="early 2025")
