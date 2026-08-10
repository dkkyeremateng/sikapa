"""Flex Web Service XML -> the normalized statement dict.

Flex names its fields differently from the Activity Statement CSV the manual
importer reads, and with "Breakout by Day" it splits one pull into a statement
per business day that repeats every position and security. These tests pin both:
that the field map is right, and that the daily repetition collapses to one
import instead of multiplying the portfolio by the number of trading days.
"""

import pytest

from .fixtures.statements import (
    FLEX_LOT_LEVEL_XML as _FLEX_LOTS,
    FLEX_SEGREGATED_ACCOUNT_XML as _FLEX_SEGREGATED,
    FLEX_UNREPORTED_CLASS_XML as _FLEX_GAP,
    SAMPLE_FLEX_XML as _FLEX,
    SAMPLE_STATEMENT as _CSV,
)


@pytest.fixture
def parsed():
    from financial_research_assistant import flex

    return flex.parse_flex_xml(_FLEX)


# --- the daily-statement merge ---------------------------------------------

def test_activity_accumulates_across_the_daily_statements(parsed):
    """Each day carries only its own trades and cash, so the whole window is the
    concatenation. Taking just the last statement — the obvious reading of a file
    whose final block looks like a complete statement — would import one day of a
    year's activity."""
    assert len(parsed["trades"]) == 2
    assert {t["symbol"] for t in parsed["trades"]} == {"AMZN"}
    assert len(parsed["corporate_actions"]) == 1


def test_holdings_come_from_the_closing_day_not_every_day(parsed):
    """Positions are a snapshot, not activity: they are restated in full every
    day. Concatenating them the way trades are concatenated would report the same
    shares once per trading day — a portfolio 261x its real size over a year."""
    assert len(parsed["positions"]) == 2
    by_symbol = {p["symbol"]: p for p in parsed["positions"]}
    assert by_symbol["AMZN"]["quantity"] == 6.0  # the closing figure, not 10 + 6
    assert by_symbol["MSFT"]["quantity"] == 4.0


def test_an_instrument_is_listed_once_however_many_days_repeat_it(parsed):
    """Instruments join to positions on symbol, so a duplicate row would fan the
    positions query out into one row per copy."""
    symbols = [i["symbol"] for i in parsed["instruments"]]
    assert sorted(symbols) == ["AMZN", "MSFT"]


def test_the_period_spans_the_whole_file_in_the_stores_own_format(parsed):
    """The period is the import key and the x-axis of the NAV curve, which
    `statements._period_bounds` reads back with "%B %d, %Y". A period covering
    only the last day would let each pull overwrite the previous one."""
    from financial_research_assistant import statements

    assert parsed["period"] == "April 01, 2026 - April 02, 2026"
    assert statements._period_bounds(parsed["period"]) == ("2026-04-01", "2026-04-02")


# --- field mapping ----------------------------------------------------------

def test_a_buy_maps_onto_the_fields_fifo_matching_reads(parsed):
    """Cost basis and commission drive every gain figure, and Flex spells them
    `cost` and `ibCommission` — neither name appears in the CSV the importer was
    written against."""
    buy = next(t for t in parsed["trades"] if t["quantity"] > 0)
    assert buy["quantity"] == 10.0
    assert buy["basis"] == 1801.05  # Flex `cost`
    assert buy["comm_fee"] == -1.05  # Flex `ibCommission`
    assert buy["proceeds"] == -1800.0
    assert buy["close_price"] == 181.0  # Flex `closePrice`
    assert buy["code"] == "P"  # Flex `notes`


def test_a_sell_keeps_the_negative_quantity_fifo_matches_on(parsed):
    """The sign is what tells a lot-opening buy from a lot-closing sell. An
    unsigned sell would open a lot instead of closing one, and the shares would
    never leave the book."""
    sell = next(t for t in parsed["trades"] if t["quantity"] < 0)
    assert sell["quantity"] == -4.0
    assert sell["proceeds"] == 760.0
    assert sell["realized_pl"] == 38.58


def test_a_sell_reported_unsigned_is_still_a_sell():
    """Flex signs quantity itself, but the sign is re-derived from buySell rather
    than trusted — the failure it guards against (a sale opening a lot) is silent
    and corrupts every subsequent gain for that symbol."""
    from financial_research_assistant import flex

    unsigned = _FLEX.replace('quantity="-4"', 'quantity="4"')
    trades = flex.parse_flex_xml(unsigned)["trades"]
    assert next(t for t in trades if t["symbol"] == "AMZN" and t["proceeds"] > 0)["quantity"] == -4.0


def test_the_datetime_is_normalized_to_the_shape_every_query_slices(parsed):
    """Flex joins date and time with the query's configured separator (a
    semicolon by default). Everything downstream takes the first ten characters,
    and date filters compare these strings, so the date half has to lead."""
    buy = next(t for t in parsed["trades"] if t["quantity"] > 0)
    assert buy["datetime"] == "2026-04-01, 12:14:23"
    assert buy["datetime"][:10] == "2026-04-01"


def test_cash_types_map_onto_the_kinds_the_income_report_groups_by(parsed):
    """`income_summary` reports gross dividends, withholding tax and fees
    separately. Flex distinguishes them by a `type` attribute where the CSV used
    separate sections, so the mapping is this parser's job."""
    kinds = sorted(c["kind"] for c in parsed["cash"])
    assert kinds == ["deposit_withdrawal", "dividend", "fee", "withholding_tax"]
    tax = next(c for c in parsed["cash"] if c["kind"] == "withholding_tax")
    assert tax["amount"] == -0.55 and tax["date"] == "2026-04-02"


def test_an_unmodelled_cash_type_is_dropped_rather_than_binned_somewhere(parsed):
    """The fixture carries broker interest, which the store has no kind for.
    Filing it under the nearest neighbour would quietly inflate that line of the
    income report; the row is dropped instead."""
    assert not any(c["description"] == "Credit Interest" for c in parsed["cash"])
    assert sum(1 for c in parsed["cash"] if c["kind"] == "fee") == 1


def test_a_position_maps_onto_the_fields_allocation_reads(parsed):
    """Flex's markPrice/positionValue/costBasisMoney are the CSV's Close Price/
    Value/Cost Basis under different names."""
    amzn = next(p for p in parsed["positions"] if p["symbol"] == "AMZN")
    assert amzn["close_price"] == 189.0
    assert amzn["value"] == 1134.0
    assert amzn["cost_basis"] == 1080.63
    assert amzn["cost_price"] == 180.105
    assert amzn["unrealized_pl"] == 53.37


def test_a_corporate_action_keeps_the_description_split_detection_parses(parsed):
    """The split factor and its ticker are read out of the description text, so
    that field surviving intact is what makes lots split-adjust."""
    from financial_research_assistant import statements

    action = parsed["corporate_actions"][0]
    assert statements._split_factor(action["description"]) == 3.0
    assert statements._symbol_from_description(action["description"]) == "AMZN"
    assert action["report_date"] == "2026-04-02"


# --- NAV and return ---------------------------------------------------------

def test_nav_spans_the_file_so_the_curve_has_two_real_ends(parsed):
    """The store keeps one prior and one current total per asset class, and dates
    them from the period. Prior therefore has to be the file's *earliest*
    snapshot, not the last day's opening balance."""
    by_class = {n["asset_class"]: n for n in parsed["nav"]}
    assert by_class["Cash"]["prior_total"] == 1000.0
    assert by_class["Cash"]["current_total"] == 1200.0
    assert by_class["Stocks"]["change"] == 500.0


def test_the_brokers_own_total_is_not_carried_as_a_row(parsed):
    """`query_nav_history` sums these rows to get account NAV. A Total row would
    be summed alongside its own components and double the whole curve."""
    assert "Total" not in {n["asset_class"] for n in parsed["nav"]}
    assert sum(n["current_total"] for n in parsed["nav"]) == 10700.0


def test_an_asset_class_the_query_cannot_report_still_counts_toward_nav():
    """IBKR folds crypto into the NAV total but the section offers no field for
    it, so the reported classes under-sum. Trusting them alone shows an account
    smaller than the broker says it is — and the shortfall is invisible, because
    every line that *is* shown is correct."""
    from financial_research_assistant import flex

    rows = flex.parse_flex_xml(_FLEX_GAP)["nav"]
    other = next(r for r in rows if r["asset_class"] == "Other")
    assert other["prior_total"] == 1000.0
    assert other["current_total"] == 1300.0
    assert sum(r["current_total"] for r in rows) == 12000.0  # the broker's total


def test_no_other_row_appears_when_the_classes_already_reconcile(parsed):
    """The remainder is a repair, not a fixture: an account whose reported
    classes add up must not grow a phantom zero line."""
    assert "Other" not in {n["asset_class"] for n in parsed["nav"]}


def test_daily_returns_are_chain_linked_not_added(parsed):
    """Two days of +1% and +2% compound to +3.02%, not +3%. Adding them
    understates a rising account, and over a year of daily figures the gap is not
    a rounding difference."""
    assert parsed["twrr"].endswith("%")
    assert round(float(parsed["twrr"].rstrip("%")), 4) == 3.02


def test_a_statement_with_no_return_reported_says_so_rather_than_zero():
    """An absent TWR is unknown, not 0% — a fabricated zero would show up as a
    flat year in the performance history."""
    from financial_research_assistant import flex

    stripped = _FLEX.replace('twr="1.0"', 'twr=""').replace('twr="2.0"', 'twr=""')
    assert flex.parse_flex_xml(stripped)["twrr"] == ""


# --- level of detail --------------------------------------------------------

def test_lot_rows_do_not_double_the_position_they_belong_to():
    """Selecting Open Positions at both Summary and Lot restates the same shares
    twice: once as a holding, once split across its tax lots. Keeping both would
    report double the portfolio."""
    from financial_research_assistant import flex

    positions = flex.parse_flex_xml(_FLEX_LOTS)["positions"]
    assert len(positions) == 1
    assert positions[0]["quantity"] == 6.0


# --- segregated sub-accounts ------------------------------------------------

def test_a_crypto_lot_is_not_stranded_in_its_paxos_segment(monkeypatch, tmp_path):
    """Crypto executes in a "<account>-P" segment that never appears as a
    statement account. Positions are scoped by import so the coins show up in the
    portfolio either way — but the trade that bought them is filtered out of lot
    matching by account, leaving a holding whose cost basis silently doesn't
    exist. Both belong to the account the statement is for."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    statements.import_statement(_FLEX_SEGREGATED)

    held = {p["symbol"]: p["quantity"] for p in statements.query_positions()}
    lots = statements.open_lots()
    assert held["BTC.USD-PAXOS"] == 0.05
    assert sum(lot["qty"] for lot in lots["BTC.USD-PAXOS"]) == 0.05


def test_a_genuinely_different_account_keeps_its_own_identity():
    """The fold is prefix-scoped on purpose. A consolidated statement covering a
    second account must not have its trades absorbed into the first — that would
    pool two books into one and quietly corrupt both sides' FIFO."""
    from financial_research_assistant import flex

    other = _FLEX_SEGREGATED.replace('accountId="U1111111-P"', 'accountId="U77770000"')
    accounts = {t["account"] for t in flex.parse_flex_xml(other)["trades"]}
    assert accounts == {"U1111111", "U77770000"}


# --- routing and failure ----------------------------------------------------

def test_flex_xml_is_detected_by_its_root_tag_not_its_extension(tmp_path):
    """A Flex pull is saved as .xml, but so is OFX v2 — and the file may be
    renamed. Routing on content is the same rule the OFX path learned."""
    from financial_research_assistant import statements

    assert statements._detect_format(_FLEX) == "flex_xml"
    assert statements._detect_format(_CSV) == "ibkr_csv"

    saved = tmp_path / "pull.txt"  # extension says nothing
    saved.write_text(_FLEX)
    assert statements._detect_format(str(saved)) == "flex_xml"


def test_importing_a_flex_file_populates_the_same_store_a_csv_would(monkeypatch, tmp_path):
    """The point of the format registry: Flex data is queryable through the very
    same tools, with no separate read path."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    saved = tmp_path / "flex.xml"
    saved.write_text(_FLEX)

    summary = statements.import_statement(saved)
    assert summary["account"] == "U1111111"
    assert summary["trades"] == 2 and summary["positions"] == 2

    assert len(statements.query_positions()) == 2
    divs = statements.query_transactions(kind="dividend")
    assert len(divs) == 1 and divs[0]["amount"] == 5.5

    # End to end through the split the fixture also carries: the 10-share lot at
    # 180.105 becomes 30 at 60.035 on the 2nd, so the four shares sold at 190
    # realize (190 - 60.035) x 4. Getting 38.58 here — the broker's own
    # fifoPnlRealized, which predates the split — would mean the corporate action
    # never reached the lot book.
    gains = statements.realized_gains()
    assert gains["total_realized"] == pytest.approx(519.86, abs=0.01)
    assert gains["unmatched_proceeds"] == 0.0


def test_a_response_that_is_not_a_statement_fails_loudly(tmp_path):
    """IBKR answers an expired token with a <FlexStatementResponse> error, which
    parses as XML perfectly well. Accepting it would store an empty import and
    report success over a statement nobody fetched."""
    from financial_research_assistant import flex

    error_page = (
        '<FlexStatementResponse timestamp="02 April, 2026 03:00 AM EDT">'
        "<Status>Fail</Status><ErrorCode>1020</ErrorCode>"
        "<ErrorMessage>Invalid request or unable to validate request.</ErrorMessage>"
        "</FlexStatementResponse>"
    )
    with pytest.raises(flex.FlexError, match="FlexQueryResponse"):
        flex.parse_flex_xml(error_page)

    with pytest.raises(flex.FlexError, match="unparseable"):
        flex.parse_flex_xml("<FlexQueryResponse><oops")
