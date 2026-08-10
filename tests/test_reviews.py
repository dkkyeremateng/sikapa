"""Report templates: period phrases in, computed briefs out."""

import pytest

from financial_research_assistant import flex as _flex, reviews


# The data's own extent, which every window is anchored on.
_LAST = "2026-08-07"
_FIRST = "2025-08-08"


def _window(phrase):
    start, end, label = reviews.resolve_period(phrase, _LAST, _FIRST)
    return start, end, label


def test_periods_are_anchored_on_the_data_not_the_clock():
    """A statement is a snapshot. Pulled in one month and read in another, "this
    month" still means the month the DATA is in — anchoring on today would shift
    every window by the age of the file, silently.

    The stale date matters: with a last session in the same month as today, an
    anchored-on-today implementation passes this by coincidence and the test
    proves nothing."""
    stale = "2025-11-14"
    assert reviews.resolve_period("this month", stale, _FIRST)[:2] == ("2025-11-01", stale)
    assert reviews.resolve_period("last month", stale, _FIRST)[:2] == (
        "2025-10-01", "2025-10-31"
    )
    assert reviews.resolve_period("ytd", stale, _FIRST)[:2] == ("2025-01-01", stale)
    assert reviews.resolve_period("ytd", stale, _FIRST)[2] == "2025 year to date"


def test_the_default_period_is_year_to_date():
    assert _window("")[:2] == ("2026-01-01", _LAST)
    assert _window("  ")[:2] == ("2026-01-01", _LAST)


@pytest.mark.parametrize("phrase, expected", [
    ("year to date", ("2026-01-01", _LAST)),
    ("Year-to-Date", ("2026-01-01", _LAST)),
    ("previous month", ("2026-07-01", "2026-07-31")),
    ("last quarter", ("2026-04-01", "2026-06-30")),
    ("this quarter", ("2026-07-01", _LAST)),
    ("Q2 2026", ("2026-04-01", "2026-06-30")),
    ("2026 q1", ("2026-01-01", "2026-03-31")),
    ("2025", ("2025-01-01", "2025-12-31")),
    ("last year", ("2025-01-01", "2025-12-31")),
    ("last 90 days", ("2026-05-09", _LAST)),
    ("last 3 months", ("2026-05-01", _LAST)),
    ("2026-02-01..2026-04-30", ("2026-02-01", "2026-04-30")),
    ("since inception", (_FIRST, _LAST)),
])
def test_the_phrases_a_person_actually_types(phrase, expected):
    assert _window(phrase)[:2] == expected


def test_a_quarter_at_a_year_boundary_walks_back_a_year():
    assert reviews.resolve_period("last quarter", "2026-02-15", _FIRST)[:2] == (
        "2025-10-01", "2025-12-31"
    )


def test_an_unrecognised_period_says_what_it_accepts():
    """Guessing a window is the failure mode this whole module exists to stop, so
    an unparseable phrase must fail loudly rather than default to something."""
    with pytest.raises(ValueError, match="last month"):
        reviews.resolve_period("whenever", _LAST, _FIRST)


def test_money_puts_the_sign_outside_the_currency_mark():
    """`${x:+,.2f}` renders "$-63.03", which reads as a typo."""
    assert reviews._money(-63.03) == "-$63.03"
    assert reviews._money(7412.61, signed=True) == "+$7,412.61"
    assert reviews._money(41265.4) == "$41,265.40"


_DAILY_XML = """<FlexQueryResponse queryName="r" type="AF"><FlexStatements count="3">
<FlexStatement accountId="U1" fromDate="2026-01-02" toDate="2026-01-02">
<ChangeInNAV startingValue="1000" endingValue="1100" twr="10.0" /></FlexStatement>
<FlexStatement accountId="U1" fromDate="2026-02-02" toDate="2026-02-02">
<ChangeInNAV startingValue="1100" endingValue="1290" twr="9.0909091"
 depositsWithdrawals="90" /></FlexStatement>
<FlexStatement accountId="U1" fromDate="2026-03-02" toDate="2026-03-02">
<ChangeInNAV startingValue="1290" endingValue="1419" twr="10.0" /></FlexStatement>
</FlexStatements></FlexQueryResponse>"""


#: Bound BEFORE any patching, so the stand-in cannot call the mock it replaces.
_REAL_PERIOD_RETURN = _flex.period_return


def _fake_period(start="", end="", xml=""):
    return _REAL_PERIOD_RETURN(start=start, end=end, xml=_DAILY_XML)


def _stub_store(monkeypatch):
    from financial_research_assistant import statements
    monkeypatch.setattr(_flex, "period_return", _fake_period)
    monkeypatch.setattr(statements, "default_account", lambda: "U1")
    monkeypatch.setattr(statements, "allocation", lambda account=None: {
        "positions": [{"symbol": "VOO", "weight_pct": 100.0, "value": 1419.0,
                       "unrealized_pl": 219.0}],
        "top5_concentration_pct": 100.0, "largest_weight_pct": 100.0,
    })
    monkeypatch.setattr(statements, "realized_gains",
                        lambda account=None, year=None: {"total_realized": 0.0, "rows": []})
    monkeypatch.setattr(reviews, "_window_income", lambda a, s, e: {
        "gross_dividends": 0.0, "withholding_tax": 0.0, "fees": 0.0, "net": 0.0,
        "_by_symbol": {},
    })


def test_a_truncated_window_says_so_in_its_own_title(monkeypatch):
    """The statement clamps the window to what it holds. Asking for a full year
    against a file that starts mid-year returns the part it has — a real figure
    under a label claiming the whole year, which is the same mislabelling that put
    a trailing twelve months on a sheet as year-to-date. The label becomes the
    title, so it has to carry the truth."""
    _stub_store(monkeypatch)
    # The fixture holds January-March only; "2026" asks for the whole year.
    assert "data covers 2026-01-02 to 2026-03-02" in reviews.build_review("2026")["title"]


def test_an_exactly_covered_window_is_not_labelled_truncated(monkeypatch):
    """The caveat must not fire on every report, or it stops meaning anything."""
    _stub_store(monkeypatch)
    title = reviews.build_review("2026-01-02..2026-03-02")["title"]
    assert "data covers" not in title


def test_the_brief_reports_the_money_side_it_was_given(monkeypatch):
    """The figures must come from the source, not be retyped: a delivered sheet
    printed a total two dollars off the line items directly above it."""
    _stub_store(monkeypatch)
    built = reviews.build_review("ytd")
    # NAV 1000 -> 1419 across the window, 90 of it deposited, so 329 was earned.
    assert round(built["facts"]["investment_gain"], 2) == 329.0
    assert built["facts"]["deposits"] == 90.0
    assert "+$329.00" in built["highlights"]
    assert "$-" not in built["highlights"], "the sign belongs outside the $"
    assert "$+" not in built["highlights"], "and never inside it"
