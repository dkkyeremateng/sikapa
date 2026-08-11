"""Single-stock report briefs — fully offline.

EDGAR access goes through ``edgar._fetch_json`` and price access through
``tools._fetch_daily``; both are monkeypatched here, so filings, prices and every
derived figure are exercised without a network call.

The fixture deliberately has the shape real 10-Q data has: no fiscal-Q4 quarters,
because the 10-K reports that period as the full year. Several of the bugs below
only appear against that gap.
"""

from datetime import date, timedelta

import pytest

from financial_research_assistant import edgar, stocks, tools


_TICKERS = {"0": {"cik_str": 798354, "ticker": "FISV", "title": "Fiserv, Inc."}}

#: (quarter end, revenue, net income, diluted EPS). Fiserv's actual as-reported
#: figures, Q4s absent exactly as EDGAR has them.
_QUARTERS = [
    ("2026-06-30", 5_290e6, 627e6, 1.17),
    ("2026-03-31", 5_030e6, 571e6, 1.07),
    ("2025-09-30", 5_260e6, 792e6, 1.46),
    ("2025-06-30", 5_520e6, 1_030e6, 1.86),
    ("2025-03-31", 5_130e6, 851e6, 1.51),
    ("2024-09-30", 5_210e6, 564e6, 0.98),
]


def _entry(end, val):
    start = date.fromisoformat(end) - timedelta(days=91)
    return {"form": "10-Q", "start": start.isoformat(), "end": end,
            "val": val, "filed": end}


def _facts(quarters=_QUARTERS):
    def unit(idx, key):
        return {"units": {key: [_entry(q[0], q[idx]) for q in quarters]}}

    return {
        "entityName": "FISERV, INC.",
        "facts": {"us-gaap": {
            "Revenues": unit(1, "USD"),
            "NetIncomeLoss": unit(2, "USD"),
            "EarningsPerShareDiluted": unit(3, edgar._PER_SHARE),
        }},
    }


#: A collapse: $237.79 on 2025-03-03 down to $52.22, with the trailing-year low
#: well above the multi-year high. The two windows disagree on purpose.
def _prices():
    rows, price = [], 120.0
    day = date(2023, 5, 1)
    path = [(date(2025, 3, 3), 237.79), (date(2026, 6, 22), 47.18),
            (date(2026, 8, 10), 52.22)]
    prev_day, prev_price = day, price
    for target_day, target_price in path:
        span = (target_day - prev_day).days
        for i in range(1, span + 1):
            rows.append(((prev_day + timedelta(days=i)).isoformat(),
                         prev_price + (target_price - prev_price) * i / span))
        prev_day, prev_price = target_day, target_price
    return [(day.isoformat(), price)] + rows


@pytest.fixture
def brief(monkeypatch):
    edgar._TICKER_CIK.clear()
    payloads = {"company_tickers.json": _TICKERS, "companyfacts": _facts()}
    monkeypatch.setattr(
        edgar, "_fetch_json",
        lambda url, timeout=20.0: next(
            (v for k, v in payloads.items() if k in url), {}
        ),
    )
    monkeypatch.setattr(tools, "_fetch_daily", lambda sym, days, **kw: _prices())
    return stocks.build_stock_brief("FISV")


def test_a_margin_is_a_level_not_a_change(brief):
    """Net income over revenue, 627/5290 = 11.8%. Computed with the CHANGE formula
    instead it reads -88.2%, which is what this module was written to stop and what
    it nonetheless shipped onto a generated sheet before a live run caught it."""
    assert brief["facts"]["net_margin_pct"] == pytest.approx(11.85, abs=0.05)
    assert "Net margin | 11.9% |" in brief["highlights"]
    # The change formula over the same two numbers. Nothing on the sheet may hold it.
    assert "-88" not in brief["highlights"]


def test_the_margin_tile_says_it_is_a_level(brief):
    """Written bare, a margin gets restated as a move and charted among changes —
    "net margin compressed +11.8%" beside a -25.5% EPS change."""
    line = next(l for l in brief["highlights"].splitlines() if "Net margin" in l)
    assert "level, not a change" in line


def test_the_year_ago_quarter_is_matched_by_date_not_position(brief):
    """No fiscal-Q4 sits in a 10-Q series, so counting four back through the list
    lands on Q1 2025 and labels a fifteen-month change as year-over-year."""
    assert brief["facts"]["year_ago_end"] == "2025-06-30"
    # 5290 vs 5520, not 5290 vs 5130 (which would read +3.1%).
    assert brief["facts"]["revenue_yoy_pct"] == pytest.approx(-4.17, abs=0.05)


def test_every_price_figure_carries_the_window_it_covers(brief):
    """-66.3% and -78.0% are both true of the same stock on the same day. A sheet
    printing one without its span invites the other to be written beside it."""
    tiles = brief["highlights"]
    assert "trailing 12 months" in tiles
    assert "high since 2023-05-01" in tiles
    assert "trough 2026-06-22" in tiles
    assert brief["facts"]["from_peak_pct"] == pytest.approx(-78.0, abs=0.2)
    # The two are DIFFERENT measurements, and the trailing year excludes the
    # 2025-03-03 peak entirely — so the drawdown must come out strictly less
    # severe. Asserting the relationship rather than a magnitude keeps this
    # about the windows and not about the fixture's synthetic price path.
    assert brief["facts"]["drawdown_pct"] > brief["facts"]["from_peak_pct"] + 3.0
    assert brief["facts"]["trough_day"] == "2026-06-22"


def test_the_high_is_never_called_all_time(brief):
    """Nothing here can see before the fetched window, so it does not claim to."""
    assert "all-time" not in brief["highlights"].lower()
    assert "all time" not in brief["markdown"].lower()


def test_the_basis_is_stated_on_the_revenue_tile(brief):
    """$4.96B against an as-reported $5.29B was plausibly the adjusted, non-GAAP
    figure — and the sheet said neither."""
    line = next(l for l in brief["highlights"].splitlines() if "Revenue" in l)
    assert "GAAP as-reported" in line
    assert "$5.29B" in line


def test_the_body_says_which_numbers_are_not_carried(brief):
    assert "non-GAAP" in brief["markdown"]


def test_a_missing_year_ago_quarter_is_said_rather_than_faked(monkeypatch):
    """Two quarters of history cannot support a YoY, and inventing one from the
    oldest row available is how a fifteen-month change gets a one-year label."""
    edgar._TICKER_CIK.clear()
    payloads = {"company_tickers.json": _TICKERS, "companyfacts": _facts(_QUARTERS[:3])}
    monkeypatch.setattr(
        edgar, "_fetch_json",
        lambda url, timeout=20.0: next(
            (v for k, v in payloads.items() if k in url), {}
        ),
    )
    monkeypatch.setattr(tools, "_fetch_daily", lambda sym, days, **kw: _prices())
    got = stocks.build_stock_brief("FISV")
    assert got["facts"]["year_ago_end"] is None
    assert got["facts"]["revenue_yoy_pct"] is None
    assert "n/a" in got["highlights"]


def test_an_unknown_ticker_is_refused_not_rendered_empty(monkeypatch):
    edgar._TICKER_CIK.clear()
    monkeypatch.setattr(
        edgar, "_fetch_json",
        lambda url, timeout=20.0: _TICKERS if "company_tickers" in url else {},
    )
    with pytest.raises(ValueError):
        stocks.build_stock_brief("ZZZZ")


def test_no_price_history_is_refused_not_rendered_empty(monkeypatch):
    edgar._TICKER_CIK.clear()
    payloads = {"company_tickers.json": _TICKERS, "companyfacts": _facts()}
    monkeypatch.setattr(
        edgar, "_fetch_json",
        lambda url, timeout=20.0: next(
            (v for k, v in payloads.items() if k in url), {}
        ),
    )
    monkeypatch.setattr(tools, "_fetch_daily", lambda sym, days, **kw: [])
    with pytest.raises(ValueError):
        stocks.build_stock_brief("FISV")


def test_the_brief_charts_as_the_renderer_reads_it(brief):
    """The tables must actually become charts — a brief whose body renders flat is
    the failure that produced a cover of tiles and no charts at all."""
    from financial_research_assistant import reports

    titles = [s["title"] for s in reports.extract_series(brief["markdown"])]
    assert any("Revenue by Quarter" in t for t in titles)
    assert len(titles) >= 2


def test_the_margin_chart_is_not_read_as_a_run_of_changes(brief):
    """Levels are unsigned, so the margin table charts as magnitudes rather than
    diverging around zero — and the sign-coherence guard leaves it alone."""
    from financial_research_assistant import reports

    series = [s for s in reports.extract_series(brief["markdown"])
              if "Net Margin" in s["title"]]
    assert series and not series[0]["signed"]
