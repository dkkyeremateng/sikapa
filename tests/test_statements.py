"""Statement parser, persistence, and accounting behavior tests."""

from .fixtures.statements import BUYSELL_STATEMENT as _BUYSELL_STATEMENT, MULTI_CURRENCY_STATEMENT as _MULTI_CURRENCY_STATEMENT, SAMPLE_STATEMENT as _SAMPLE_STATEMENT, mini_statement as _mini_statement
from .helpers.fakes import install_stub_fx as _stub_fx


def test_a_blended_total_row_is_not_a_currency():
    """A consolidated statement closes each cash section with 'Total in USD' (and
    'Total Dividends in USD'), which restate the rows above in one currency. They
    are aggregates, not dividends: ingesting them invents a currency and counts
    every figure twice."""
    from financial_research_assistant import statements

    parsed = statements.parse_statement(_MULTI_CURRENCY_STATEMENT)
    currencies = {row["currency"] for row in parsed["cash"]}
    assert currencies == {"USD", "EUR"}

    dividends = [r for r in parsed["cash"] if r["kind"] == "dividend"]
    assert [r["amount"] for r in dividends] == [12.0, 8.0]
    assert sum(r["amount"] for r in parsed["cash"]) == 18.0  # 12 + 8 - 2


def test_parse_statement_extracts_trades_and_cash_skipping_totals():
    """The parser keeps Data/Order trades and the four cash-flow sections while
    dropping every SubTotal/Total aggregate row (so nothing double-counts)."""
    from financial_research_assistant import statements

    parsed = statements.parse_statement(_SAMPLE_STATEMENT)
    assert parsed["account"] == "U1111111"
    assert parsed["period"] == "April 1, 2026 - May 19, 2026"

    assert len(parsed["trades"]) == 2  # AMZN + MSFT, no subtotal/total rows
    amzn = parsed["trades"][0]
    assert amzn["symbol"] == "AMZN"
    assert amzn["quantity"] == 5.25
    assert amzn["proceeds"] == -1050
    assert amzn["code"] == "FPA;O;P"

    kinds = sorted(c["kind"] for c in parsed["cash"])
    assert kinds == ["deposit_withdrawal", "dividend", "dividend", "fee", "withholding_tax"]
    dep = next(c for c in parsed["cash"] if c["kind"] == "deposit_withdrawal")
    assert dep["amount"] == 1500 and dep["date"] == "2026-04-30"  # Settle Date used


def test_parse_statement_extracts_positions_instruments_and_nav():
    """The parser also captures Open Positions (DataDiscriminator=Summary),
    Financial Instrument Information (ISIN/description), and the NAV breakdown +
    time-weighted return — skipping the Total rows in each."""
    from financial_research_assistant import statements

    parsed = statements.parse_statement(_SAMPLE_STATEMENT)

    assert len(parsed["positions"]) == 2  # AMZN + MSFT, no Total row
    amzn = next(p for p in parsed["positions"] if p["symbol"] == "AMZN")
    assert amzn["quantity"] == 12 and amzn["value"] == 2760
    assert amzn["unrealized_pl"] == 600

    assert len(parsed["instruments"]) == 2
    inst = next(i for i in parsed["instruments"] if i["symbol"] == "AMZN")
    assert inst["description"] == "AMAZON.COM INC"
    assert inst["security_id"] == "US0231351067"  # ISIN

    # NAV has two Data rows (Cash, Stock) — the Total row is skipped — plus TWRR.
    assert [n["asset_class"] for n in parsed["nav"]] == ["Cash", "Stock"]
    assert parsed["twrr"] == "12.5%"

    # Corporate Actions: the split row is kept, its Total row skipped.
    assert len(parsed["corporate_actions"]) == 1
    ca = parsed["corporate_actions"][0]
    assert ca["quantity"] == 30 and "Split 3 for 1" in ca["description"]
    assert ca["report_date"] == "2026-04-11"


def test_import_statement_persists_and_is_idempotent(monkeypatch, tmp_path):
    """Importing stores trades + cash + positions/instruments/NAV; re-importing
    the same (account, period) replaces rather than duplicates."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    summary = statements.import_statement(_SAMPLE_STATEMENT)
    assert summary["replaced"] is False
    assert summary["trades"] == 2 and summary["cash"] == 5
    assert summary["positions"] == 2 and summary["instruments"] == 2
    assert summary["nav"] == 2 and summary["twrr"] == "12.5%"
    assert summary["corporate_actions"] == 1
    assert summary["cash_by_kind"]["dividend"]["count"] == 2
    assert round(summary["cash_by_kind"]["dividend"]["amounts"]["USD"], 2) == 50.2

    again = statements.import_statement(_SAMPLE_STATEMENT)
    assert again["replaced"] is True
    # Only one import survives — no doubling.
    imports = statements.list_imports()
    assert len(imports) == 1
    assert imports[0]["positions"] == 2 and imports[0]["nav"] == 2
    assert imports[0]["corporate_actions"] == 1
    assert len(statements.query_transactions()) == 8  # 2 trades + 5 cash + 1 corp action
    assert len(statements.query_positions()) == 2      # not doubled either


def test_query_transactions_filters_by_kind_and_symbol(monkeypatch, tmp_path):
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    statements.import_statement(_SAMPLE_STATEMENT)

    trades = statements.query_transactions(kind="trade")
    assert len(trades) == 2 and all(t["kind"] == "trade" for t in trades)

    amzn = statements.query_transactions(symbol="AMZN")
    assert len(amzn) == 1 and amzn[0]["symbol"] == "AMZN"

    divs = statements.query_transactions(kind="dividend")
    assert len(divs) == 2 and all(d["kind"] == "dividend" for d in divs)

    corp = statements.query_transactions(kind="corporate_action")
    assert len(corp) == 1 and corp[0]["kind"] == "corporate_action"
    assert "Split 3 for 1" in corp[0]["description"]
    # symbol filter matches the ticker inside a corporate-action description
    assert len(statements.query_transactions(symbol="SCHD")) == 1


def test_query_transactions_symbol_like_wildcards_are_literal(monkeypatch, tmp_path):
    """SQL wildcards in a symbol filter match literally: '%' must not act as
    match-everything against cash/corporate-action descriptions."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    statements.import_statement(_SAMPLE_STATEMENT)

    # Unescaped, "%" would match every dividend and "_VDA" would match "NVDA".
    assert statements.query_transactions(kind="dividend", symbol="%") == []
    assert statements.query_transactions(kind="dividend", symbol="_VDA") == []
    assert len(statements.query_transactions(kind="dividend", symbol="NVDA")) == 1
    # And the escape helper itself round-trips the specials.
    assert statements._like_contains("A%B_C\\D") == "%A\\%B\\_C\\\\D%"


def test_query_transactions_empty_store_returns_empty(monkeypatch, tmp_path):
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    assert statements.query_transactions() == []
    assert statements.list_imports() == []


def test_query_positions_joins_instrument_name_and_isin(monkeypatch, tmp_path):
    """Positions come back enriched with the instrument description and ISIN, and
    can be filtered to one symbol."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    statements.import_statement(_SAMPLE_STATEMENT)

    positions = statements.query_positions()
    assert len(positions) == 2
    amzn = next(p for p in positions if p["symbol"] == "AMZN")
    assert amzn["description"] == "AMAZON.COM INC"
    assert amzn["security_id"] == "US0231351067"

    just = statements.query_positions(symbol="MSFT")
    assert len(just) == 1 and just[0]["symbol"] == "MSFT"


def test_query_nav_returns_breakdown_and_twrr(monkeypatch, tmp_path):
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    statements.import_statement(_SAMPLE_STATEMENT)

    nav = statements.query_nav()
    assert nav["twrr"] == "12.5%"
    assert [r["asset_class"] for r in nav["rows"]] == ["Cash", "Stock"]
    stock = next(r for r in nav["rows"] if r["asset_class"] == "Stock")
    assert stock["current_total"] == 24000


def test_period_bounds_parses_ibkr_period_string():
    from financial_research_assistant import statements

    assert statements._period_bounds("January 1, 2024 - December 31, 2024") == (
        "2024-01-01", "2024-12-31")
    assert statements._period_bounds("April 1, 2026 - May 19, 2026") == (
        "2026-04-01", "2026-05-19")
    assert statements._period_bounds("nonsense") is None


def test_query_nav_history_stitches_snapshots_across_imports(monkeypatch, tmp_path):
    """Two statements whose periods chain produce a merged, date-sorted NAV
    series: each contributes its period-start (prior) and period-end (current)
    total NAV, and the shared boundary value lines up."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    y2024 = _SAMPLE_STATEMENT  # period Apr 1 2026 - May 19 2026, NAV prior 0 -> 21200 total? use as-is
    statements.import_statement(y2024)
    # A second statement in a later, non-overlapping period.
    later = y2024.replace(
        "April 1, 2026 - May 19, 2026", "June 1, 2026 - June 30, 2026"
    ).replace(
        "Net Asset Value,Data,Cash ,1200,800,0,800,-400",
        "Net Asset Value,Data,Cash ,800,700,0,700,-100",
    ).replace(
        "Net Asset Value,Data,Stock,20000,24000,0,24000,4000",
        "Net Asset Value,Data,Stock,24000,40000,0,40000,16000",
    )
    statements.import_statement(later)

    hist = statements.query_nav_history()
    dates = [p["date"] for p in hist]
    assert dates == sorted(dates)  # oldest -> newest
    assert dates[0] == "2026-04-01" and dates[-1] == "2026-06-30"
    navs = {p["date"]: p["nav"] for p in hist}
    # period-end of the first statement = 800 +24000
    assert round(navs["2026-05-19"], 2) == round(800 + 24000, 2)
    # period-end of the second statement = 700 + 40000
    assert round(navs["2026-06-30"], 2) == 40700.0


def test_query_performance_history_chains_twrr(monkeypatch, tmp_path):
    """The performance index starts at 100 and compounds each statement's TWRR in
    chronological order, independent of deposits."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    # First statement: period Apr 1 - May 19 2026, TWRR 12.5%.
    statements.import_statement(_SAMPLE_STATEMENT)
    # A later, day-adjacent statement (May 20 picks up where May 19 ends, so no
    # gap) with a clean 10% TWRR.
    later = _SAMPLE_STATEMENT.replace(
        "April 1, 2026 - May 19, 2026", "May 20, 2026 - June 30, 2026"
    ).replace("12.5%", "10%")
    statements.import_statement(later)

    hist = statements.query_performance_history()
    pts = hist["points"]
    assert pts[0]["index"] == 100.0 and pts[0]["date"] == "2026-04-01"
    # After 12.5% then 10%: 100 * 1.125 * 1.10 = 123.75
    assert round(pts[-1]["index"], 2) == 123.75
    assert round(pts[-1]["cumulative_return_pct"], 2) == 23.75
    assert [p["date"] for p in pts] == sorted(p["date"] for p in pts)
    assert hist["dropped"] == [] and hist["gaps"] == []


def test_performance_history_drops_overlaps_and_flags_gaps(monkeypatch, tmp_path):
    """Overlapping statements are reconciled to a non-overlapping set (finer
    periods win), and a gap between kept periods is detected — not silently
    chained as if contiguous."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    def stmt(period, twrr):
        return _SAMPLE_STATEMENT.replace(
            "April 1, 2026 - May 19, 2026", period
        ).replace("12.5%", twrr)

    # A full-year 2024 statement, and a monthly one it fully contains (overlap):
    statements.import_statement(stmt("January 1, 2024 - December 31, 2024", "12%"))
    statements.import_statement(stmt("March 1, 2024 - March 31, 2024", "3%"))
    # A 2026 statement, leaving all of 2025 as a gap:
    statements.import_statement(stmt("January 1, 2026 - June 30, 2026", "8%"))

    hist = statements.query_performance_history()
    kept_dates = [p["date"] for p in hist["points"]]
    # The annual 2024 overlaps the March 2024 slice, so one of them is dropped.
    assert len(hist["dropped"]) == 1
    # Activity-selection keeps the finer (shorter) March slice over the full year.
    assert "January 1, 2024 - December 31, 2024" in hist["dropped"]
    # 2025 is missing entirely -> a multi-hundred-day gap is reported.
    assert hist["gaps"] and hist["gaps"][0]["days"] > 300
    assert "2024-03-31" in kept_dates and "2026-06-30" in kept_dates

    # The gap is tagged with the point index the break falls after.
    assert "after_point" in hist["gaps"][0]

    # The tool surfaces both caveats inline for the user.
    import financial_research_assistant.tools as tools
    out = tools.portfolio_performance_chart()
    assert "dropped 1 overlapping statement" in out
    assert "gap in coverage" in out and "break in the line" in out


def test_parse_pct_handles_percent_strings():
    from financial_research_assistant import statements

    assert statements._parse_pct("10.98645721%") == 10.98645721
    assert statements._parse_pct("  -3.5 % ".replace(" %", "%")) == -3.5
    assert statements._parse_pct("") is None
    assert statements._parse_pct("n/a") is None


def test_to_float_is_tolerant_of_bad_cells():
    from financial_research_assistant import statements as s
    assert s._to_float("1,234.5") == 1234.5
    assert s._to_float("") == 0.0
    assert s._to_float(None) == 0.0
    assert s._to_float("n/a") == 0.0      # non-numeric -> 0.0, never raises
    assert s._to_float("12.5%") == 12.5


def test_malformed_numeric_cell_does_not_abort_import(monkeypatch, tmp_path):
    """A single non-numeric value in a numeric column drops to 0.0; the rest of
    the statement still imports (no all-or-nothing failure)."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    bad = (
        "Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code\n"
        'Trades,Data,Order,Stocks,USD,U1,AMZN,"2026-04-01, 12:00:00",oops,200,198.5,-1050,-0.35,1050,0,-8,\n'
    )
    summary = s.import_statement(bad)
    assert summary["trades"] == 1
    row = s.query_transactions(kind="trade")[0]
    assert row["quantity"] == 0.0 and row["symbol"] == "AMZN"  # bad qty -> 0, rest intact


def test_parse_detects_multiple_accounts(monkeypatch, tmp_path):
    from financial_research_assistant import statements as s
    multi = (
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U111,2024-06-01,A Cash Dividend,5\n"
        "Dividends,Data,USD,U222,2024-06-02,B Cash Dividend,7\n"
    )
    parsed = s.parse_statement(multi)
    assert parsed["accounts"] == ["U111", "U222"]
    assert parsed["account"] == "U111+U222"           # combined, not just the first
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    summary = s.store_statement(parsed)
    assert summary["multi_account"] is True and len(summary["accounts"]) == 2


def test_cash_sums_grouped_by_currency(monkeypatch, tmp_path):
    """USD and EUR dividends are summed separately, never blended into one $."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    mixed = (
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U1,2024-06-01,A Cash Dividend,10\n"
        "Dividends,Data,EUR,U1,2024-06-02,B Cash Dividend,20\n"
    )
    summary = s.import_statement(mixed)
    amounts = summary["cash_by_kind"]["dividend"]["amounts"]
    assert amounts == {"USD": 10.0, "EUR": 20.0}


def test_history_queries_are_account_scoped(monkeypatch, tmp_path):
    """Two accounts' statements don't blend: each account's NAV/performance series
    is independent, and the default is the newest import's account."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    # Account A: two years; Account B: one year with a very different return.
    s.import_statement(_mini_statement("AAA", "January 1, 2024 - December 31, 2024", "10%", 1000))
    s.import_statement(_mini_statement("AAA", "January 1, 2025 - December 31, 2025", "20%", 1300))
    s.import_statement(_mini_statement("BBB", "January 1, 2024 - December 31, 2024", "-50%", 500))

    assert set(s.list_accounts()) == {"AAA", "BBB"}
    # AAA performance chains 10% then 20% -> index 132; BBB is just -50% -> 50.
    a = s.query_performance_history(account="AAA")["points"][-1]["index"]
    b = s.query_performance_history(account="BBB")["points"][-1]["index"]
    assert round(a, 1) == 132.0
    assert round(b, 1) == 50.0
    # NAV history for AAA has its own points and doesn't include BBB's 500.
    navs_a = [p["nav"] for p in s.query_nav_history(account="AAA")]
    assert 500.0 not in navs_a and 1300.0 in navs_a
    # Default account = newest import's account (BBB was imported last).
    assert s.query_performance_history()["points"][-1]["index"] == b


def test_query_transactions_limit_is_recency_across_types(monkeypatch, tmp_path):
    """A small limit returns the newest rows across ALL types, not just trades —
    an older trade is dropped in favor of a newer cash row."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    stmt = (
        "Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code\n"
        'Trades,Data,Order,Stocks,USD,U1,OLD,"2020-01-01, 10:00:00",1,1,1,-1,0,1,0,0,\n'
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U1,2026-06-01,RECENT Cash Dividend,5\n"
    )
    s.import_statement(stmt)
    top = s.query_transactions(limit=1)
    assert len(top) == 1
    # The 2026 dividend is newer than the 2020 trade, so it wins the single slot.
    assert top[0]["kind"] == "dividend"


def test_schema_migration_adds_missing_column(monkeypatch, tmp_path):
    """An older DB whose `imports` table predates the `twrr` column is migrated in
    place on connect, so importing into it doesn't raise 'no such column'."""
    import sqlite3
    from financial_research_assistant import statements as s
    db = tmp_path / "old.db"
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(db))
    # Simulate a pre-twrr store: create just the imports table without `twrr`.
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE imports (id INTEGER PRIMARY KEY, account TEXT NOT NULL, "
        "period TEXT NOT NULL, imported TEXT, UNIQUE(account, period))"
    )
    conn.commit(); conn.close()
    # Import should migrate (ALTER ADD twrr) and succeed.
    summary = s.import_statement(_mini_statement("U1", "January 1, 2024 - December 31, 2024", "10%", 1000))
    assert summary["twrr"] == "10%"
    assert s.query_nav(account="U1")["twrr"] == "10%"


def test_realized_gains_fifo_short_long_split(monkeypatch, tmp_path):
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    r = s.realized_gains(year=2025)
    # AMZN: sold 6 @150, cost 100, held >1yr -> +300 long-term.
    assert round(r["by_symbol"]["AMZN"]["long_term"], 2) == 300.0
    assert round(r["by_symbol"]["AMZN"]["short_term"], 2) == 0.0
    # MSFT: sold 4 @40, cost 50, held 30 days -> -40 short-term.
    assert round(r["by_symbol"]["MSFT"]["short_term"], 2) == -40.0
    assert round(r["total_realized"], 2) == 260.0
    assert round(r["long_term"], 2) == 300.0 and round(r["short_term"], 2) == -40.0
    assert r["unmatched_proceeds"] == 0.0


def test_the_long_term_boundary_is_more_than_a_year_not_exactly_one():
    """The holding period starts the DAY AFTER acquisition, so a position bought on
    1 Jan and sold the following 1 Jan is SHORT term. The boundary was `>= 365`
    days, which called that trade long-term and understated the tax owed — an error
    in the direction that costs the taxpayer, not the IRS."""
    from financial_research_assistant import statements as s

    assert not s.is_long_term("2025-01-01", "2026-01-01"), "one year exactly"
    assert s.is_long_term("2025-01-01", "2026-01-02"), "a year and a day"


def test_a_leap_day_in_the_span_does_not_shorten_the_year():
    """Why this can't be a day count: 2023-06-01 → 2024-06-01 crosses 29 February,
    so it is 366 days — but it is still exactly one calendar year, and still short
    term. Any `> N days` rule gets one of these two cases wrong."""
    from financial_research_assistant import statements as s

    assert s._days_between("2023-06-01", "2024-06-01") == 366
    assert not s.is_long_term("2023-06-01", "2024-06-01")
    assert s.is_long_term("2023-06-01", "2024-06-02")


def test_a_leap_day_purchase_has_a_holding_period():
    """29 February has no anniversary in a common year. The holding period begins
    1 March, so a sale on 1 March 2026 is exactly one year (short) and 2 March is
    long — rather than raising, or silently bucketing as short forever."""
    from financial_research_assistant import statements as s

    assert not s.is_long_term("2024-02-29", "2025-03-01")
    assert s.is_long_term("2024-02-29", "2025-03-02")


def test_an_unreadable_date_buckets_short_rather_than_raising():
    """A malformed lot date must not sink the whole realized-gains run; short term
    is the conservative bucket (it never understates the tax)."""
    from financial_research_assistant import statements as s

    assert not s.is_long_term("not-a-date", "2026-01-02")
    assert not s.is_long_term("2020-01-01", "")


def test_realized_gains_reports_unmatched_sell(monkeypatch, tmp_path):
    """A sell with no imported opening lot surfaces as unmatched proceeds, not a
    silently wrong gain."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    stmt = (
        "Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code\n"
        'Trades,Data,Order,Stocks,USD,U1,ZZZ,"2025-05-01, 10:00:00",-5,20,20,100,0,-100,0,0,\n'
    )
    s.import_statement(stmt)
    r = s.realized_gains()
    assert r["unmatched_proceeds"] == 100.0 and r["by_symbol"] == {}


def test_open_lots_leftover_after_fifo(monkeypatch, tmp_path):
    """open_lots returns the shares still held after FIFO-matching sells against
    buys — the input tax-loss harvesting works from."""
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)  # AMZN buy10/sell6, MSFT buy4/sell4
    ol = s.open_lots()
    assert set(ol) == {"AMZN"}                       # MSFT fully closed, dropped
    assert len(ol["AMZN"]) == 1
    lot = ol["AMZN"][0]
    assert lot["qty"] == 4.0 and lot["cost_per_share"] == 100.0
    assert lot["open_date"] == "2024-01-01"


_TRADE_HEADER = (
    "Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,"
    "Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,"
    "MTM P/L,Code\n"
)


def _trades_statement(period, *rows):
    """A statement carrying just a period and some trade rows."""
    return (
        "Statement,Header,Field Name,Field Value\n"
        f'Statement,Data,Period,"{period}"\n' + _TRADE_HEADER + "".join(rows)
    )


#: One buy and one sell of AMZN in March, as they appear in both the statement
#: covering the whole year and the one covering just that month.
_MARCH_BUY = 'Trades,Data,Order,Stocks,USD,U1,AMZN,"2025-03-03, 10:00:00",10,100,100,-1000,0,1000,0,0,\n'
_MARCH_SELL = 'Trades,Data,Order,Stocks,USD,U1,AMZN,"2025-03-20, 10:00:00",-4,150,150,600,0,-400,200,0,\n'


def test_overlapping_imports_count_each_trade_once(monkeypatch, tmp_path):
    """Importing an annual statement and a monthly one inside it stores every trade
    in the covered month twice. Walked as-is, the FIFO stream opens the buy lot
    twice and matches the sell twice — doubling the open lots, the realized gain
    and every tax figure derived from them, with nothing in the output to say so."""
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_trades_statement(
        "January 1, 2025 - December 31, 2025", _MARCH_BUY, _MARCH_SELL))
    s.import_statement(_trades_statement(
        "March 1, 2025 - March 31, 2025", _MARCH_BUY, _MARCH_SELL))

    assert round(s.realized_gains()["by_symbol"]["AMZN"]["realized"], 2) == 200.0
    lots = s.open_lots()["AMZN"]
    assert sum(lot["qty"] for lot in lots) == 6.0


def test_a_genuinely_repeated_fill_is_kept(monkeypatch, tmp_path):
    """Two identical fills within ONE statement are two fills, not a duplicate —
    the same order can fill twice at the same price in the same second. Only
    repetition ACROSS imports is an overlap."""
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_trades_statement(
        "March 1, 2025 - March 31, 2025", _MARCH_BUY, _MARCH_BUY))
    assert sum(lot["qty"] for lot in s.open_lots()["AMZN"]) == 20.0


def test_an_overlapping_split_is_applied_once(monkeypatch, tmp_path):
    """A corporate action in two overlapping statements would rescale every open
    lot by the SQUARE of its factor — 9× the shares at a ninth of the cost."""
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    split = (
        "Corporate Actions,Header,Asset Category,Currency,Account,Report Date,"
        "Date/Time,Description,Quantity,Proceeds,Value,Realized P/L,Code\n"
        'Corporate Actions,Data,Stocks,USD,U1,2025-06-11,"2025-06-10, 20:25:00",'
        '"AMZN(US8) Split 3 for 1 (AMZN, ETF, US8)",20,0,0,0,\n'
    )
    body = _trades_statement("January 1, 2025 - December 31, 2025", _MARCH_BUY) + split
    s.import_statement(body)
    s.import_statement(
        _trades_statement("June 1, 2025 - June 30, 2025") + split
    )
    lot = s.open_lots()["AMZN"][0]
    assert lot["qty"] == 30.0
    assert round(lot["cost_per_share"], 4) == round(100.0 / 3.0, 4)


def test_lots_are_not_pooled_across_accounts(monkeypatch, tmp_path):
    """A sell in one account cannot consume a lot opened in another: the shares are
    still held, and the basis and holding period belong to a different book. Pooled,
    the U2 sell matched U1's cheap lot and reported a gain against shares that were
    never sold."""
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_trades_statement(
        "January 1, 2025 - December 31, 2025",
        'Trades,Data,Order,Stocks,USD,U1,AMZN,"2025-01-05, 10:00:00",10,50,50,-500,0,500,0,0,\n',
    ))
    s.import_statement(_trades_statement(
        "January 1, 2026 - December 31, 2026",
        'Trades,Data,Order,Stocks,USD,U2,AMZN,"2026-02-01, 10:00:00",-3,150,150,450,0,-450,0,0,\n',
    ))
    # U2 is the newest import's account, so it is the default scope.
    newest = s.realized_gains()
    assert newest["by_symbol"] == {}
    assert newest["unmatched_proceeds"] == 450.0
    # U1's lot is untouched and still open.
    assert s.open_lots(account="U1")["AMZN"][0]["qty"] == 10.0
    assert s.open_lots() == {}
    # And the escape hatch still pools, for whoever explicitly asks.
    assert round(s.realized_gains(account="all")["by_symbol"]["AMZN"]["realized"], 2) == 300.0


def test_a_pooled_lot_view_says_that_is_what_it_is(monkeypatch, tmp_path):
    """`account="all"` is a real view of nothing — it has to label itself, or the
    figure reads as one account's."""
    from financial_research_assistant import statements as s
    from financial_research_assistant import tools as t

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    assert "account U1" in t.realized_gains()
    assert "ALL accounts pooled" in t.realized_gains(account="all")


def test_income_summary_by_currency_and_symbol(monkeypatch, tmp_path):
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    inc = s.income_summary(year=2025)
    assert inc["by_currency"]["USD"]["gross_dividends"] == 12.0
    assert inc["by_currency"]["USD"]["withholding_tax"] == -2.0
    assert inc["by_currency"]["USD"]["net"] == 10.0
    assert inc["by_currency"]["EUR"]["gross_dividends"] == 8.0   # not blended with USD
    assert inc["dividends_by_symbol"]["AMZN"] == 12.0


def test_income_is_not_doubled_by_overlapping_statements(monkeypatch, tmp_path):
    """Re-importing an overlapping period must not double the income.

    Only trades were reconciled across imports; cash never was. A trailing-twelve-
    month Flex pull alongside a year-to-date CSV of the same months reported every
    dividend, tax and fee twice, and `income_summary` returned exactly double.
    Nothing looked wrong — a doubled dividend is still a plausible dividend —
    which is why it surfaced only against the broker's own XML.
    """
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    once = s.income_summary(year=2025)
    assert once["by_currency"]["USD"]["gross_dividends"] == 12.0

    # A DIFFERENT period covering the same months — the real shape: a trailing
    # window pulled from Flex over a year-to-date CSV. Re-importing the identical
    # period would not reproduce it, since the store is keyed (account, period)
    # and simply replaces.
    s.import_statement(
        'Statement,Data,Period,"April 1, 2025 - April 30, 2025"\n'
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U1,2025-04-01,AMZN(US1) Cash Dividend,12\n"
        "Withholding Tax,Header,Currency,Account,Date,Description,Amount,Code\n"
        "Withholding Tax,Data,USD,U1,2025-04-01,AMZN(US1) Cash Dividend - US Tax,-2,\n"
    )
    assert s.income_summary(year=2025) == once


def test_two_holdings_paying_the_same_amount_are_both_counted():
    """The guard against over-correcting: identity includes the description, so
    two securities paying the same amount on the same day stay two dividends —
    even when they arrive in separate imports, where a description-blind key
    would collapse them into one and UNDER-count."""
    from financial_research_assistant import statements as s

    def row(import_id, desc):
        return {"import_id": import_id, "account": "U1", "date": "2025-03-01",
                "kind": "dividend", "currency": "USD", "description": desc,
                "amount": 5.0}

    kept = s._dedupe_across_imports(
        [row(1, "AMZN cash dividend"), row(2, "MSFT cash dividend")],
        key=s._cash_key,
    )
    assert [r["description"] for r in kept] == ["AMZN cash dividend", "MSFT cash dividend"]


def test_allocation_weights_and_concentration(monkeypatch, tmp_path):
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    a = s.allocation()
    assert a["total_value"] == 1240.0
    amzn = next(p for p in a["positions"] if p["symbol"] == "AMZN")
    assert round(amzn["weight_pct"], 1) == 51.6
    assert round(a["top5_concentration_pct"], 1) == 100.0
    assert set(a["by_category"]) == {"Stocks", "ETFs"}


def test_realized_gains_adjusts_for_stock_split(monkeypatch, tmp_path):
    """A 3-for-1 split rescales open lots (qty ×3, cost/sh ÷3) so a later sale of
    the post-split share count matches correctly instead of leaving unmatched
    proceeds or a wrong basis."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    stmt = (
        "Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code\n"
        'Trades,Data,Order,Stocks,USD,U1,SCHD,"2024-01-01, 10:00:00",10,30,30,-300,0,300,0,0,\n'
        'Trades,Data,Order,Stocks,USD,U1,SCHD,"2024-11-01, 10:00:00",-30,12,12,360,0,-300,60,0,\n'
        "Corporate Actions,Header,Asset Category,Currency,Account,Report Date,Date/Time,Description,Quantity,Proceeds,Value,Realized P/L,Code\n"
        'Corporate Actions,Data,Stocks,USD,U1,2024-10-11,"2024-10-10, 20:25:00","SCHD(US8) Split 3 for 1 (SCHD, ETF, US8)",20,0,0,0,\n'
    )
    s.import_statement(stmt)
    r = s.realized_gains()
    assert round(r["by_symbol"]["SCHD"]["realized"], 2) == 60.0
    assert r["unmatched_proceeds"] == 0.0   # the whole 30 shares matched


def test_split_factor_parsing():
    from financial_research_assistant import statements as s
    assert s._split_factor("SCHD(x) Split 3 for 1 (…)") == 3.0
    assert s._split_factor("X Split 1 for 10 reverse") == 0.1
    assert s._split_factor("NVDA cash dividend") is None


def test_allocation_converts_multi_currency_to_usd(monkeypatch, tmp_path):
    """A EUR position is converted to USD before weighting, so the book total and
    weights are in one currency."""
    from financial_research_assistant import statements as s
    import financial_research_assistant.tools as t
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    stmt = (
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U1,2025-01-01,seed,0\n"  # sets the account
        "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n"
        "Open Positions,Data,Summary,Stocks,USD,AMZN,1,1,100,100,100,100,0,\n"
        "Open Positions,Data,Summary,Stocks,EUR,SAP,1,1,100,100,100,100,0,\n"
    )
    s.import_statement(stmt)
    # Offline (statements) with an fx map: EUR 100 -> 200 USD, total 300.
    res = s.allocation(fx={"USD": 1.0, "EUR": 2.0})
    assert res["total_value"] == 300.0
    sap = next(p for p in res["positions"] if p["symbol"] == "SAP")
    assert sap["value"] == 200.0 and round(sap["weight_pct"], 1) == 66.7
    assert res["currencies_converted"] == ["EUR"]
    # Tool path fetches rates; stub EUR=2.0.
    _stub_fx(monkeypatch, {"EUR": 2.0})
    out = t.allocation()
    assert "total 300.00 USD" in out and "converted to USD: EUR" in out



def test_transfers_dedupe_despite_differing_broker_wording(monkeypatch, tmp_path):
    """The two exports word the same transfer differently — the CSV writes
    "Electronic Fund Transfer", the Flex XML writes "CASH RECEIPTS / ELECTRONIC
    FUND TRANSFERS". Keying deposits on description let every one through twice,
    so YTD deposits read $35,000 against a true $17,500. Dividends survived only
    because their two spellings differ by case alone, which `.upper()` closed —
    so checking dividends alone made a partial fix look complete."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    def row(import_id, desc):
        return {"import_id": import_id, "account": "U1", "date": "2026-01-06",
                "kind": "deposit_withdrawal", "currency": "USD",
                "description": desc, "amount": 2500.0}

    kept = s._dedupe_across_imports(
        [row(1, "Electronic Fund Transfer"),
         row(2, "CASH RECEIPTS / ELECTRONIC FUND TRANSFERS")],
        key=s._cash_key,
    )
    assert len(kept) == 1, "the same transfer, worded two ways"


def test_two_transfers_on_one_day_in_one_statement_are_both_kept():
    """Multiplicity WITHIN an import is what keeps the description-blind key from
    merging genuinely separate transfers."""
    from financial_research_assistant import statements as s

    def row(import_id):
        return {"import_id": import_id, "account": "U1", "date": "2026-01-06",
                "kind": "deposit_withdrawal", "currency": "USD",
                "description": "Electronic Fund Transfer", "amount": 2500.0}

    assert len(s._dedupe_across_imports([row(1), row(1)], key=s._cash_key)) == 2


def test_listing_transactions_does_not_repeat_overlapping_rows(monkeypatch, tmp_path):
    """`query_transactions` read cash raw, so anything summing that view counted
    each overlapping movement twice."""
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    once = s.query_transactions(kind="dividend", account="U1", limit=500)
    s.import_statement(
        'Statement,Data,Period,"April 1, 2025 - April 30, 2025"\n'
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U1,2025-04-01,AMZN(US1) Cash Dividend,12\n"
    )
    assert len(s.query_transactions(kind="dividend", account="U1", limit=500)) == len(once)


def test_the_portfolio_total_is_summed_before_it_is_rounded(monkeypatch, tmp_path):
    """Adding the DISPLAYED per-position figures compounds one rounding per row
    into the total. On the real book the raw values sum to 7,412.6016 — a true
    $7,412.60 — while summing the rounded rows gives $7,412.61, a cent that is not
    in the data and that a second tool reporting the same quantity disagrees with.
    """
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr(s, "query_positions", lambda symbol=None, account=None: [
        # Each rounds UP a fraction of a cent; three rows turn that into a whole one.
        {"symbol": "A", "value": 100.0, "unrealized_pl": 1.004, "currency": "USD",
         "description": "", "asset_category": "STK"},
        {"symbol": "B", "value": 100.0, "unrealized_pl": 1.004, "currency": "USD",
         "description": "", "asset_category": "STK"},
        {"symbol": "C", "value": 100.0, "unrealized_pl": 1.004, "currency": "USD",
         "description": "", "asset_category": "STK"},
    ])
    res = s.allocation()
    assert res["unrealized_pl"] == 3.01, "raw 3.012 rounds to 3.01"
    assert sum(p["unrealized_pl"] for p in res["positions"]) == 3.00, (
        "the displayed rows sum to something else — which is the trap"
    )
