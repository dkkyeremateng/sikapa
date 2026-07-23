"""Cross-broker OFX/QFX statement importer tests.

Covers the format dispatcher (statements._detect_format / import_statement) and
parse_ofx, which map a non-IBKR broker's OFX export onto the same normalized dict
and SQLite store the IBKR CSV path uses. The parse_ofx tests need the optional
`ofxtools` extra; the dispatcher and missing-dependency tests do not.
"""

import sys

import pytest

from .fixtures.statements import SAMPLE_OFX as _SAMPLE_OFX, SAMPLE_STATEMENT as _SAMPLE_STATEMENT


# --- format detection (no ofxtools needed) ---------------------------------

def test_detect_format_routes_ofx_and_csv(tmp_path):
    """OFX header text, .ofx/.qfx files route to 'ofx'; IBKR CSV (and anything
    else) defaults to 'ibkr_csv' so existing imports are unaffected."""
    from financial_research_assistant import statements

    assert statements._detect_format(_SAMPLE_OFX) == "ofx"
    assert statements._detect_format(_SAMPLE_STATEMENT) == "ibkr_csv"

    qfx = tmp_path / "acct.qfx"
    qfx.write_text("garbage that is not really ofx")  # extension wins
    assert statements._detect_format(qfx) == "ofx"

    csv = tmp_path / "activity.csv"
    csv.write_text(_SAMPLE_STATEMENT)
    assert statements._detect_format(csv) == "ibkr_csv"


def test_parse_ofx_missing_dependency_raises_support_error(monkeypatch):
    """When ofxtools isn't installed, parse_ofx raises OfxSupportError (not a bare
    ImportError) so callers can render a clean install hint. Forced by masking the
    module, so this test runs whether or not the [ofx] extra is present."""
    from financial_research_assistant import statements

    monkeypatch.setitem(sys.modules, "ofxtools", None)
    monkeypatch.setitem(sys.modules, "ofxtools.Parser", None)
    with pytest.raises(statements.OfxSupportError):
        statements.parse_ofx(_SAMPLE_OFX)


def test_import_tool_reports_missing_dependency_friendly(monkeypatch, tmp_path):
    """The model-facing import tool returns the install-hint string (never raises)
    when a real .ofx file is given but ofxtools is absent."""
    import financial_research_assistant.tools as tools

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    monkeypatch.setitem(sys.modules, "ofxtools", None)
    monkeypatch.setitem(sys.modules, "ofxtools.Parser", None)
    f = tmp_path / "acct.ofx"
    f.write_text(_SAMPLE_OFX)

    out = tools.import_ibkr_statement(str(f))
    assert "ofxtools" in out and "[ofx]" in out


# --- parsing + dispatch (needs the optional ofxtools extra) ----------------


def test_parse_ofx_maps_trades_dividend_position_instrument():
    """parse_ofx maps an OFX investment statement onto the normalized dict:
    BUY/SELL trades with type-derived signs, an INCOME dividend, an INVBANKTRAN
    deposit, an INVPOSLIST position, and SECLIST instruments with CUSIP->ticker."""
    pytest.importorskip("ofxtools")
    from financial_research_assistant import statements

    parsed = statements.parse_ofx(_SAMPLE_OFX)
    assert parsed["account"] == "1234567"
    assert parsed["period"] == "April 01, 2026 - May 15, 2026"
    assert parsed["twrr"] == "" and parsed["nav"] == []  # OFX carries neither

    # Two trades; symbols resolved from the CUSIP via SECLIST.
    assert len(parsed["trades"]) == 2
    buy = next(t for t in parsed["trades"] if t["quantity"] > 0)
    sell = next(t for t in parsed["trades"] if t["quantity"] < 0)
    assert buy["symbol"] == "AMZN" and buy["quantity"] == 10.0
    assert buy["proceeds"] == -1801.05 and buy["comm_fee"] == -1.05  # buy spends cash
    # Sell is sign-normalized to negative shares / positive proceeds regardless of
    # the raw units sign the broker reported.
    assert sell["quantity"] == -4.0 and sell["proceeds"] == 759.0

    kinds = sorted(c["kind"] for c in parsed["cash"])
    assert kinds == ["deposit_withdrawal", "dividend"]
    div = next(c for c in parsed["cash"] if c["kind"] == "dividend")
    assert div["amount"] == 5.5 and div["description"] == "AMZN DIVIDEND"

    assert len(parsed["positions"]) == 1
    pos = parsed["positions"][0]
    assert pos["symbol"] == "AMZN" and pos["quantity"] == 6.0
    assert pos["value"] == 1110.0 and pos["close_price"] == 185.0

    assert len(parsed["instruments"]) == 1
    inst = parsed["instruments"][0]
    assert inst["symbol"] == "AMZN" and inst["security_id"] == "023135106"
    assert inst["description"] == "AMAZON COM INC"


def test_import_statement_dispatches_ofx_into_the_store(monkeypatch, tmp_path):
    """import_statement auto-routes an OFX document to parse_ofx and persists it in
    the same store the IBKR CSV path uses — queryable via the normal read tools."""
    pytest.importorskip("ofxtools")
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    summary = statements.import_statement(_SAMPLE_OFX)
    assert summary["account"] == "1234567"
    assert summary["trades"] == 2 and summary["positions"] == 1
    assert summary["cash_by_kind"]["dividend"]["count"] == 1

    # Rows are readable through the same query surface as IBKR imports.
    assert len(statements.query_positions()) == 1
    divs = statements.query_transactions(kind="dividend")
    assert len(divs) == 1 and divs[0]["amount"] == 5.5


def test_import_statement_still_routes_ibkr_csv(monkeypatch, tmp_path):
    """The dispatcher must not regress the IBKR CSV path: a CSV source still parses
    via parse_statement and stores its full section set."""
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    summary = statements.import_statement(_SAMPLE_STATEMENT)
    assert summary["trades"] == 2 and summary["nav"] == 2
    assert summary["twrr"] == "12.5%"  # NAV/TWRR only the CSV path carries
