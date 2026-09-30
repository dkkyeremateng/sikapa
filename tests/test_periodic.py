"""The daily, weekly and monthly reports: every figure computed, every window named.

The price fake is date-aware on purpose (see `test-fakes-must-model-the-knob`):
each symbol's close depends on the DATE, not on the row's position, and a series
only reaches back as far as `days` asks. So a report that measures the wrong
window — Monday's close instead of the Friday before, today instead of the
scheduled session — produces a different number, and the tests see it.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from financial_research_assistant import (
    autonomy, fundamentals, guardrails, jobs, journal, market, periodic, reports,
    statements, tools,
)

NY = ZoneInfo("America/New_York")
TODAY = date(2026, 10, 2)  # a Friday
BASE = {"SPY": 600.0, "AAPL": 200.0, "MSFT": 400.0, "QQQ": 500.0}
HOLIDAYS: set[date] = set()


def close(sym: str, day: date) -> float:
    """A close that depends on the calendar date, differently per symbol."""
    n = (day - date(2026, 1, 1)).days
    drift = {"AAPL": 0.0011, "MSFT": -0.0004, "SPY": 0.0005}.get(sym, 0.0002)
    wobble = ((n * 7 + len(sym) * 3) % 9 - 4) / 400
    return round(BASE.get(sym, 100.0) * (1 + drift * n + wobble), 4)


def sessions(until: date = TODAY):
    d = date(2026, 1, 2)
    while d <= until:
        if d.weekday() < 5 and d not in HOLIDAYS:
            yield d
        d += timedelta(days=1)


@pytest.fixture
def market_data(monkeypatch):
    fetched: list[tuple[str, int]] = []

    def fake(sym, days, strict=False, as_of=None, **_kw):
        fetched.append((sym, days))
        start = TODAY - timedelta(days=days)
        return [(d.isoformat(), close(sym, d)) for d in sessions() if d >= start]

    monkeypatch.setattr(tools, "_fetch_daily", fake)
    monkeypatch.setattr(fundamentals, "_fetch_calendar", lambda sym: {})
    HOLIDAYS.clear()
    return fetched


@pytest.fixture
def book(monkeypatch):
    """10 AAPL and 5 MSFT, valued at the statement's close on 25 September."""
    as_of = date(2026, 9, 25)
    rows = [
        {"symbol": "AAPL", "asset_category": "Stocks", "currency": "USD",
         "close_price": close("AAPL", as_of), "value": 10 * close("AAPL", as_of),
         "description": "APPLE INC"},
        {"symbol": "MSFT", "asset_category": "Stocks", "currency": "USD",
         "close_price": close("MSFT", as_of), "value": 5 * close("MSFT", as_of),
         "description": "MICROSOFT"},
        {"symbol": "AAPL  261218C00250000", "asset_category": "Equity and Index Options",
         "currency": "USD", "close_price": 3.0, "value": 300.0},
    ]
    monkeypatch.setattr(statements, "query_positions", lambda account=None: rows)
    monkeypatch.setattr(statements, "default_account", lambda: "U1")
    monkeypatch.setattr(statements, "list_imports", lambda: [
        {"account": "U1", "period": "August 26, 2026 - September 25, 2026"}])
    return rows


def pct(sym: str, a: date, b: date) -> float:
    return (close(sym, b) / close(sym, a) - 1) * 100


# --- the book ----------------------------------------------------------------------------


def test_a_flex_imported_book_is_priced_too(monkeypatch):
    """Flex spells the category STK, the Activity CSV "Stocks". Matching only the
    second priced nothing for a Flex-imported book — found running against the
    real store, not by a fixture written in the CSV's spelling."""
    monkeypatch.setattr(statements, "query_positions", lambda account=None: [
        {"symbol": "VOO", "asset_category": "STK", "currency": "USD",
         "close_price": 500.0, "value": 5000.0},
        {"symbol": "BTC.USD-PAXOS", "asset_category": "CRYPTO", "currency": "USD",
         "close_price": 60000.0, "value": 6000.0},
        {"symbol": "ES", "asset_category": "FUT", "currency": "USD",
         "close_price": 5000.0, "value": 1.0},
    ])
    monkeypatch.setattr(statements, "default_account", lambda: "U1")
    monkeypatch.setattr(statements, "list_imports", lambda: [])
    b = periodic.load_book()
    assert {h["symbol"] for h in b["holdings"]} == {"VOO", "BTC-USD"}
    assert b["unpriced_value"] == 1.0


def test_a_mid_week_weekly_says_it_is_the_week_so_far(market_data, book):
    brief = periodic.build_weekly_brief(date(2026, 9, 29))  # a Tuesday
    assert brief.label == "week of 28 September 2026 (to Tuesday 29 September)"


def test_the_book_prices_stocks_only_and_knows_its_date(book):
    b = periodic.load_book()
    assert [h["symbol"] for h in b["holdings"]] == ["AAPL", "MSFT"] or \
           [h["symbol"] for h in b["holdings"]] == ["MSFT", "AAPL"]
    assert b["unpriced_value"] == 300.0, "an option's Yahoo series is not the option"
    assert b["as_of"] == "2026-09-25"
    aapl = next(h for h in b["holdings"] if h["symbol"] == "AAPL")
    assert aapl["units"] == pytest.approx(10.0)


# --- daily -----------------------------------------------------------------------------------


def test_the_daily_brief_measures_the_session_it_is_for(market_data, book):
    session, prev = date(2026, 9, 29), date(2026, 9, 28)
    brief = periodic.build_daily_brief(session)
    assert brief.facts["session"] == "2026-09-29" and brief.facts["prev_session"] == "2026-09-28"
    assert brief.facts["moves"]["AAPL"] == round(pct("AAPL", prev, session), 2)
    assert brief.facts["spy_pct"] == pytest.approx(pct("SPY", prev, session))
    dollars = 10 * (close("AAPL", session) - close("AAPL", prev)) + \
        5 * (close("MSFT", session) - close("MSFT", prev))
    assert brief.facts["holdings_dollars"] == pytest.approx(dollars)
    # Every tile names its window.
    assert "2026-09-28 → 2026-09-29" in brief.highlights
    assert "positions as of 2026-09-25" in brief.highlights
    assert brief.message.startswith("📊 Daily close · Tuesday 29 September 2026")


def test_a_session_with_no_prices_yet_is_not_reported_as_yesterdays(market_data, book):
    """The source hasn't printed today's bar: the report must not quietly use the
    last one it has and call it today."""
    brief = periodic.build_daily_brief(TODAY + timedelta(days=3))  # Monday, no data
    assert brief.facts["holdings_pct"] is None and brief.facts["moves"] == {}
    assert "No prices for this session yet" in brief.markdown


# --- weekly ------------------------------------------------------------------------------------


def test_the_weekly_brief_runs_from_the_close_before_monday(market_data, book):
    """Measuring from Monday's close would drop Monday's move from the week."""
    fri, prior_fri = date(2026, 9, 25), date(2026, 9, 18)
    brief = periodic.build_weekly_brief(fri)
    assert brief.period == "2026-W39"
    assert (brief.facts["from"], brief.facts["to"]) == ("2026-09-18", "2026-09-25")
    assert brief.facts["moves"]["MSFT"] == round(pct("MSFT", prior_fri, fri), 2)
    assert brief.facts["spy_pct"] == pytest.approx(pct("SPY", prior_fri, fri))
    assert "What Carried the Week" in brief.markdown and "vs S&P 500" in brief.highlights


def test_a_holiday_friday_ends_the_week_on_thursday(market_data, book):
    HOLIDAYS.add(date(2026, 9, 25))
    brief = periodic.build_weekly_brief(date(2026, 9, 25))
    assert brief.facts["to"] == "2026-09-24"
    assert brief.facts["moves"]["AAPL"] == round(pct("AAPL", date(2026, 9, 18), date(2026, 9, 24)), 2)


# --- monthly -----------------------------------------------------------------------------------


def test_the_monthly_brief_without_a_flex_file_says_so_and_uses_the_holdings(market_data, book):
    brief = periodic.build_monthly_brief(2026, 9)
    assert brief.period == "2026-09"
    assert any("not available for September 2026" in n for n in brief.notes)
    assert (brief.facts["from"], brief.facts["to"]) == ("2026-08-31", "2026-09-30")
    assert brief.facts["holdings_pct"] is not None
    assert "Holdings This Month" in brief.markdown


def test_a_month_is_labelled_from_the_session_it_really_starts_from(market_data, book):
    """With 31 August closed, September's baseline is the 28th's close — and the
    label must say the 28th, not a date that had no close."""
    HOLIDAYS.add(date(2026, 8, 31))
    brief = periodic.build_monthly_brief(2026, 9)
    assert brief.facts["from"] == "2026-08-28"
    assert brief.facts["holdings_pct"] is not None


def test_the_monthly_brief_carries_the_ideas_track_record(market_data, book):
    journal.save_entries([
        {"id": "t1", "symbol": "AAPL", "source": "recommender", "status": "scored",
         "scored_on": "2026-09-10", "hit": True, "alpha_pct": 4.0, "conviction": 4,
         "verdict": "bullish"},
        {"id": "t2", "symbol": "MSFT", "source": "recommender", "status": "scored",
         "scored_on": "2026-09-12", "hit": False, "alpha_pct": -2.0, "conviction": 2,
         "verdict": "bullish"},
        {"id": "t3", "symbol": "NVDA", "status": "scored", "scored_on": "2026-09-12",
         "hit": True, "alpha_pct": 9.0, "verdict": "bullish"},  # a chat call, not an idea
    ])
    brief = periodic.build_monthly_brief(2026, 9)
    rec = brief.facts["track_record"]["all"]
    assert (rec["n"], rec["hits"]) == (2, 1) and rec["avg_alpha"] == pytest.approx(1.0)
    assert "Track Record of the Agent's Ideas" in brief.markdown


# --- which period a scheduled run covers -------------------------------------------------------


def test_each_report_covers_the_period_it_was_scheduled_for():
    daily = datetime(2026, 9, 29, 17, 15, tzinfo=NY)
    assert periodic._period_for("daily", daily, "America/New_York") == (date(2026, 9, 29), "2026-09-29")
    saturday = datetime(2026, 10, 3, 9, 0, tzinfo=NY)
    assert periodic._period_for("weekly", saturday, "America/New_York") == (date(2026, 10, 2), "2026-W40")
    first = datetime(2026, 10, 1, 8, 0, tzinfo=NY)
    assert periodic._period_for("monthly", first, "America/New_York") == ((2026, 9), "2026-09")
    january = datetime(2027, 1, 4, 8, 0, tzinfo=NY)
    assert periodic._period_for("monthly", january, "America/New_York")[1] == "2026-12"


# --- the commentary check ------------------------------------------------------------------------


def _brief(**kw):
    base = dict(kind="weekly", period="2026-W39", label="w", title="Weekly Report",
                subtitle="", highlights="Your holdings | +0.84% | x\nS&P 500 | -1.27% | y",
                markdown="| AAPL | +3.41% | +$1,234 |", message="", facts={"x": 18.456})
    base.update(kw)
    return periodic.Brief(**base)


def test_commentary_may_round_the_sheets_figures_but_not_invent_one():
    b = _brief()
    assert periodic.unsupported_figures("Holdings rose 0.8% while the S&P 500 fell 1.3%.", b) == []
    assert periodic.unsupported_figures("AAPL added $1,234 and 3.4%; 18.5 too.", b) == []
    assert periodic.unsupported_figures("In 2026 Q3, 3 of 5 holdings rose over 10-year highs.", b) == []
    assert periodic.unsupported_figures("You beat the market by 2.11 points.", b) == ["2.11"]


def test_commentary_that_invents_figures_is_retried_then_withheld(monkeypatch):
    replies = iter(["Beat the market by 2.11 points.", "Holdings rose 0.84%."])

    async def ask(system, user, **kw):
        return next(replies)

    monkeypatch.setattr(autonomy, "ask", ask)
    text, note = asyncio.run(periodic.write_commentary(_brief()))
    assert text == "Holdings rose 0.84%." and note == ""

    replies = iter(["Beat by 2.11 points.", "Still 9.99 better."])
    text, note = asyncio.run(periodic.write_commentary(_brief()))
    assert text == "" and "withheld" in note and "2.11" in note


def test_no_commentary_while_paused():
    guardrails.pause()
    text, note = asyncio.run(periodic.write_commentary(_brief()))
    assert text == "" and "paused" in note


# --- the job ------------------------------------------------------------------------------------


@pytest.fixture
def quiet_render(monkeypatch):
    monkeypatch.setattr(periodic, "_render", lambda brief, commentary, note: ["/r/x.png", "/r/x.pdf"])


def test_a_report_goes_out_once_per_period(market_data, book, quiet_render):
    task = {"id": "s1", "due": datetime(2026, 10, 3, 13, 0, tzinfo=timezone.utc).isoformat(),
            "tz": "America/New_York", "kind": "job", "job": "report-weekly"}
    run = jobs.handler_for("report-weekly")
    first = asyncio.run(run(task, True))
    assert first.ok and first.notify and first.files == ["/r/x.png", "/r/x.pdf"]
    assert first.text.startswith("🗓 Weekly report · week of 28 September 2026")
    assert jobs.already_sent("weekly", "2026-W40")
    again = asyncio.run(run(task, True))
    assert again.ok and not again.notify and "already sent" in again.text


def test_a_daily_report_on_a_holiday_sends_nothing_and_enters_nothing(market_data, book, monkeypatch):
    monkeypatch.setattr(market, "latest_session", lambda symbol="SPY": date(2026, 9, 25))
    task = {"id": "s1", "due": datetime(2026, 9, 28, 21, 15, tzinfo=timezone.utc).isoformat(),
            "tz": "America/New_York"}
    res = asyncio.run(jobs.handler_for("report-daily")(task, False))
    assert res.ok and not res.notify and "no session" in res.text
    assert jobs.already_sent("daily", "2026-09-28") is None


def test_a_daily_report_waits_when_the_price_source_is_down(market_data, book, monkeypatch):
    monkeypatch.setattr(market, "latest_session", lambda symbol="SPY": None)
    task = {"id": "s1", "due": datetime(2026, 9, 28, 21, 15, tzinfo=timezone.utc).isoformat(),
            "tz": "America/New_York"}
    res = asyncio.run(jobs.handler_for("report-daily")(task, False))
    assert res.ok is False, "fail, so the retry path tries again later"


def test_a_paused_agent_skips_its_reports(market_data, book, quiet_render):
    guardrails.pause()
    task = {"id": "s1", "due": datetime(2026, 10, 3, 13, 0, tzinfo=timezone.utc).isoformat(),
            "tz": "America/New_York"}
    res = asyncio.run(jobs.handler_for("report-weekly")(task, True))
    assert res.ok and not res.notify and "paused" in res.text
    assert jobs.already_sent("weekly", "2026-W40") is None


def test_the_default_schedule_includes_the_three_reports(monkeypatch):
    rows = {j: t for j, t, _n in jobs.ensure_default_jobs()}
    assert rows["report-daily"]["repeat"] == "weekdays"
    assert rows["report-daily"]["tz"] == "America/New_York"
    due = datetime.fromisoformat(rows["report-daily"]["due"]).astimezone(NY)
    assert (due.hour, due.minute) == (17, 15) and due.weekday() < 5
    assert rows["report-weekly"]["repeat"] == "weekly"
    assert datetime.fromisoformat(rows["report-weekly"]["due"]).astimezone().weekday() == 5
    assert rows["report-monthly"]["repeat"] == "monthly-first-weekday"


# --- by hand ------------------------------------------------------------------------------------


def test_a_hand_built_periodic_report_is_refused():
    out = reports.render_report("Weekly Report — week of 21 September", "- a: 1%\n- b: 2%\n- c: 3%",
                                highlights="Return | +1.2%", deliver=False)
    assert out.startswith("NOT RENDERED") and "periodic_report" in out
    ok = reports.render_report("NVDA vs AMD: weekly momentum compared", "x", deliver=False,
                               allow_prose=True)
    assert not ok.startswith("NOT RENDERED — this is one of the scheduled reports")


def test_the_report_command_builds_one_now(market_data, book, quiet_render, monkeypatch):
    monkeypatch.setattr(market, "latest_session", lambda symbol="SPY": date(2026, 9, 25))
    from financial_research_assistant import channels, scheduler

    sent: list[str] = []
    monkeypatch.setattr(channels, "deliver_file",
                        lambda path, caption="", prefer="", full_quality=False: (sent.append(path), (["t"], []))[1])
    out = asyncio.run(scheduler.run_command("/report weekly", fake=True))
    assert out.startswith("🗓 Weekly report") and "(files sent)" in out
    assert sent == ["/r/x.png", "/r/x.pdf"]
    assert "Which report?" in asyncio.run(scheduler.run_command("/report yearly", fake=True))


def test_a_partial_month_compares_the_market_over_the_same_days(market_data, book, monkeypatch):
    """The Flex file stopped on 7 August: the account return covers 3-7 August.
    The S&P beside it covered the whole month, and the title said "August 2026" —
    two windows on one sheet, found running against the real file."""
    from financial_research_assistant import reviews

    def partial(period="", account=""):
        return {"period_label": "August 2026 (data covers 2026-08-03 to 2026-08-07)",
                "title": "t", "subtitle": "2026-08-03 to 2026-08-07",
                "highlights": "Return | +2.29% | time-weighted, 5 sessions",
                "markdown": "## Holdings by Weight\n- **AAPL:** 50%",
                "facts": {"start": "2026-08-03", "end": "2026-08-07", "truncated": True,
                          "return_pct": 2.29}}

    monkeypatch.setattr(reviews, "build_review", partial)
    brief = periodic.build_monthly_brief(2026, 8)
    assert brief.facts["spy_pct"] == pytest.approx(pct("SPY", date(2026, 7, 31), date(2026, 8, 7)))
    assert "data covers 2026-08-03 to 2026-08-07" in brief.title
    assert "2026-08-03 → 2026-08-07" in brief.message
    assert brief.message.startswith("📅 Monthly report · August 2026 (data covers")
    assert any("same days" in n for n in brief.notes)
