"""SEC EDGAR filing-intelligence tests — fully offline.

Every network access in edgar.py goes through ``_fetch_json``; these tests
monkeypatch it with a small URL router returning canned SEC payloads, so CIK
lookup, the recent-filings list, XBRL financials, and full-text search are all
exercised without a network call. The module-level ticker→CIK cache is cleared
per test so the fake map is re-read."""

from financial_research_assistant import edgar, tools


def _router(payloads):
    """A fake _fetch_json: returns the payload whose key is a substring of the URL."""
    def fake(url, timeout=20.0):
        for key, val in payloads.items():
            if key in url:
                return val
        return {}
    return fake


_TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def _with_tickers(monkeypatch, extra):
    edgar._TICKER_CIK.clear()
    payloads = {"company_tickers.json": _TICKERS}
    payloads.update(extra)
    monkeypatch.setattr(edgar, "_fetch_json", _router(payloads))


def test_cik_lookup_pads_and_misses(monkeypatch):
    _with_tickers(monkeypatch, {})
    assert edgar._cik_for("aapl") == "0000320193"   # zero-padded to 10
    assert edgar._cik_for("ZZZZ") is None


def test_filing_url_builds_archive_path():
    assert edgar._filing_url("0000320193", "0000320193-25-000100", "aapl.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000100/aapl.htm"
    )


def test_sec_filings_lists_filters_and_links(monkeypatch):
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-K", "8-K", "10-Q"],
        "filingDate": ["2025-11-01", "2025-10-15", "2025-08-01"],
        "accessionNumber": ["0000320193-25-000100", "0000320193-25-000090",
                            "0000320193-25-000080"],
        "primaryDocument": ["aapl-20250930.htm", "ex99.htm", "aapl-q3.htm"],
        "primaryDocDescription": ["10-K", "8-K", "10-Q"],
    }}}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})

    out = edgar.sec_filings("AAPL")
    assert "Apple Inc." in out and "10-K" in out and "8-K" in out
    assert ("https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019325000100/aapl-20250930.htm") in out

    only8k = edgar.sec_filings("AAPL", form_type="8-K")
    assert "8-K" in only8k and "10-Q" not in only8k     # form filter applied

    assert "No SEC CIK" in edgar.sec_filings("ZZZZ")     # unknown ticker


_FACTS = {"entityName": "Apple Inc.", "facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"fy": 2024, "fp": "FY", "form": "10-K", "end": "2024-09-28", "val": 391035000000},
        {"fy": 2023, "fp": "FY", "form": "10-K", "end": "2023-09-30", "val": 383285000000},
    ]}},
    "GrossProfit": {"units": {"USD": [
        {"fy": 2024, "fp": "FY", "form": "10-K", "end": "2024-09-28", "val": 180683000000},
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        {"fy": 2024, "fp": "FY", "form": "10-K", "end": "2024-09-28", "val": 93736000000},
        {"fy": 2023, "fp": "FY", "form": "10-K", "end": "2023-09-30", "val": 96995000000},
        # a 10-Q quarterly row must be IGNORED by the annual selector
        {"fy": 2024, "fp": "Q3", "form": "10-Q", "end": "2024-06-29", "val": 21448000000},
    ]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        {"fy": 2024, "fp": "FY", "form": "10-K", "end": "2024-09-28", "val": 6.08},
    ]}},
}}}


def test_sec_financials_summary_computes_margins_and_growth(monkeypatch):
    _with_tickers(monkeypatch, {"companyfacts/CIK0000320193": _FACTS})
    out = edgar.sec_financials("AAPL")
    assert "FY2024" in out and "FY2023" in out
    assert "391.04B" in out              # revenue, compact money
    assert "6.08" in out                 # diluted EPS (USD/shares)
    assert "net 24.0%" in out            # 93736/391035
    assert "gross 46.2%" in out          # 180683/391035
    assert "revenue growth FY2023→FY2024: +2.0%" in out  # (391035-383285)/383285


def test_sec_financials_single_concept_and_bad_tag(monkeypatch):
    _with_tickers(monkeypatch, {"companyfacts/CIK0000320193": _FACTS})
    out = edgar.sec_financials("AAPL", concept="NetIncomeLoss")
    assert "NetIncomeLoss" in out and "93.74B" in out
    assert "FY2024" in out and "FY2023" in out
    # quarterly row excluded -> only two annual points
    assert "21.45B" not in out
    bad = edgar.sec_financials("AAPL", concept="BogusTag")
    assert "No us-gaap concept" in bad


def test_sec_filing_search_returns_cited_hits(monkeypatch):
    fts = {"hits": {"total": {"value": 42}, "hits": [
        {"_id": "0000320193-24-000123:aapl-20240928.htm", "_source": {
            "ciks": ["0000320193"], "display_names": ["Apple Inc. (AAPL)"],
            "form": "10-K", "file_date": "2024-11-01"}},
    ]}}
    _with_tickers(monkeypatch, {"search-index": fts})
    out = edgar.sec_filing_search("supply chain risk", symbol="AAPL", forms="10-K")
    assert "Apple Inc." in out and "10-K" in out and "2024-11-01" in out
    assert "42 total hits" in out
    assert ("https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000123/aapl-20240928.htm") in out

    assert "Give a keyword" in edgar.sec_filing_search("")

    _with_tickers(monkeypatch, {"search-index": {"hits": {"hits": []}}})
    assert "matched" in edgar.sec_filing_search("nonexistent phrase", symbol="AAPL")


def test_sec_material_events_decodes_item_codes(monkeypatch):
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["8-K", "10-Q", "8-K"],
        "filingDate": ["2026-04-30", "2026-02-01", "2026-01-15"],
        "accessionNumber": ["0000320193-26-000011", "xx", "0000320193-26-000002"],
        "primaryDocument": ["a8k.htm", "q.htm", "b8k.htm"],
        "primaryDocDescription": ["8-K", "10-Q", "8-K"],
        "items": ["2.02,9.01", "", "5.02"],
    }}}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    out = edgar.sec_material_events("AAPL")
    assert "Results of Operations" in out and "9.01" in out   # 2.02 decoded
    assert "Departure/Election of Directors" in out           # 5.02 decoded
    assert "10-Q" not in out                                  # only 8-Ks listed
    assert ("https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019326000011/a8k.htm") in out


def test_edgar_text_helpers():
    assert edgar._decode_items("2.02,9.01").startswith("2.02 Results")
    assert edgar._decode_items("99.9") == "99.9"             # unknown code passes through
    assert edgar._html_to_text("<p>Hello <b>x</b></p><div>y &amp; z</div>") == "Hello x\ny & z"
    long_para = ("A substantive paragraph about revenue growth and margin expansion "
                 "that easily exceeds the minimum length threshold for a passage here.")
    ps = edgar._passages(long_para + "\ntiny", "revenue margin", 3)
    assert ps and "revenue growth" in ps[0]
    assert edgar._passages("some text", "", 3) == []          # no query terms


def test_sec_filing_excerpt_extracts_matching_passages(monkeypatch):
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-Q", "10-K"],           # newest-first; the 10-K is the target
        "filingDate": ["2026-02-01", "2025-11-01"],
        "accessionNumber": ["yy", "0000320193-25-000100"],
        "primaryDocument": ["q.htm", "aapl-10k.htm"],
        "primaryDocDescription": ["10-Q", "10-K"],
        "items": ["", ""],
    }}}
    doc = (
        "<html><body>"
        "<p>Item 1A. Risk Factors</p>"
        "<p>The Company depends on a concentrated supply chain in Asia and a limited "
        "number of component suppliers, and any prolonged disruption to that supply "
        "chain could materially and adversely affect the Company's results of "
        "operations and financial condition in a given period.</p>"
        "<p>Unrelated boilerplate about forward-looking statements and safe harbor.</p>"
        "</body></html>"
    )
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    monkeypatch.setattr(edgar, "_fetch_text", lambda url, **kw: doc)

    out = edgar.sec_filing_excerpt("AAPL", "supply chain disruption", form_type="10-K")
    assert "10-K filed 2025-11-01" in out          # picked the latest 10-K, not the 10-Q
    assert "concentrated supply chain" in out       # the matching passage text
    assert "aapl-10k.htm" in out                     # source filing link

    assert "No passage" in edgar.sec_filing_excerpt("AAPL", "zzznonexistentterm", form_type="10-K")
    assert "topic/phrase" in edgar.sec_filing_excerpt("AAPL", "")   # empty query


def test_edgar_tools_registered():
    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools.TOOLS}
    assert {"sec_filings", "sec_material_events", "sec_financials",
            "sec_filing_search", "sec_filing_excerpt"} <= names
