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


# --- render_review: the figures are not the model's to write ---------------------


def _capture(monkeypatch):
    """Stand in for `render_report`, recording exactly what it was handed."""
    from financial_research_assistant import reports, tools
    seen = {}

    def fake(title, markdown, highlights="", subtitle="", deliver=True,
             allow_prose=False, theme="", output="", stance=""):
        seen.update(title=title, markdown=markdown, highlights=highlights,
                    subtitle=subtitle, deliver=deliver, stance=stance)
        return "Rendered 1 page(s)."

    monkeypatch.setattr(reports, "render_report", fake)
    monkeypatch.setattr(tools, "render_review", tools.render_review)  # keep the real one
    return seen


def _brief(monkeypatch):
    from financial_research_assistant import reviews as rv
    monkeypatch.setattr(rv, "build_review", lambda period="", account="": {
        "period_label": "2026 year to date",
        "title": "Portfolio Performance — 2026 year to date",
        "subtitle": "2026-01-01 to 2026-08-07 · time-weighted, deposit-independent",
        "highlights": "Return | +7.64% | time-weighted\nUnrealised P/L | +$7,412.61 | since purchase",
        "markdown": "## Holdings by Weight\n- **VOO:** 27.2% — $12,530.40\n",
        "facts": {},
    })


def test_the_figures_come_from_the_brief_not_the_caller(monkeypatch):
    """Three delivered sheets in a row put $7,410.61 where the source said
    $7,412.61 — each time because the model rewrote a figure by hand. Here it has
    no opportunity to: the tiles and body are generated, and `observations` is the
    only text it supplies."""
    from financial_research_assistant import tools
    seen = _capture(monkeypatch)
    _brief(monkeypatch)

    tools.render_review(period="ytd", observations="April carried the year.")
    assert "+$7,412.61" in seen["highlights"]
    assert seen["title"] == "Portfolio Performance — 2026 year to date"
    assert "## Holdings by Weight" in seen["markdown"]
    # The signature has no parameter through which a figure could arrive.
    import inspect
    params = set(inspect.signature(tools.render_review).parameters)
    assert params == {"period", "observations", "stance", "deliver", "theme", "account"}


def test_observations_are_appended_as_their_own_section(monkeypatch):
    from financial_research_assistant import tools
    seen = _capture(monkeypatch)
    _brief(monkeypatch)

    tools.render_review(observations="- April carried the year\n- Deposits dominated")
    assert "## Observations" in seen["markdown"]
    assert "- April carried the year" in seen["markdown"]
    assert "- Deposits dominated" in seen["markdown"]


def test_a_paragraph_of_observations_becomes_bullets(monkeypatch):
    """Prose sinks into a paragraph the cover cannot use; bullets reach the
    observations block."""
    from financial_research_assistant import tools
    seen = _capture(monkeypatch)
    _brief(monkeypatch)

    tools.render_review(observations="April carried the year.\nDeposits dominated.")
    assert "- April carried the year." in seen["markdown"]
    assert "- Deposits dominated." in seen["markdown"]


def test_a_review_with_no_observations_says_so(monkeypatch):
    """Figures alone are a table, not a review — the omission is reported rather
    than shipped silently."""
    from financial_research_assistant import tools
    _capture(monkeypatch)
    _brief(monkeypatch)

    assert "No observations were supplied" in tools.render_review()


def test_an_unknown_period_is_reported_not_rendered(monkeypatch):
    from financial_research_assistant import reports, tools
    called = []
    monkeypatch.setattr(reports, "render_report",
                        lambda *a, **k: called.append(1) or "rendered")
    out = tools.render_review(period="whenever")
    assert "Could not build the review" in out and not called
