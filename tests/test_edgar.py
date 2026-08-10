"""SEC EDGAR filing-intelligence tests — fully offline.

Every network access in edgar.py goes through ``_fetch_json``; these tests
monkeypatch it with a small URL router returning canned SEC payloads, so CIK
lookup, the recent-filings list, XBRL financials, and full-text search are all
exercised without a network call. The module-level ticker→CIK cache is cleared
per test so the fake map is re-read."""

from financial_research_assistant import catalog, edgar, tools


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


_QFACTS = {"entityName": "Apple Inc.", "facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        # 10-Q single quarters (~90 days) — newest to oldest, 5 for a YoY compare
        {"fp": "Q2", "form": "10-Q", "start": "2025-01-01", "end": "2025-03-31", "val": 100_000_000_000, "filed": "2025-05-01"},
        {"fp": "Q1", "form": "10-Q", "start": "2024-10-01", "end": "2024-12-31", "val": 95_000_000_000, "filed": "2025-02-01"},
        {"fp": "Q4", "form": "10-Q", "start": "2024-07-01", "end": "2024-09-30", "val": 90_000_000_000, "filed": "2024-11-01"},
        {"fp": "Q3", "form": "10-Q", "start": "2024-04-01", "end": "2024-06-30", "val": 88_000_000_000, "filed": "2024-08-01"},
        {"fp": "Q2", "form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "val": 80_000_000_000, "filed": "2024-05-01"},
        # a 6-month YTD span in a 10-Q (~181 days) must be EXCLUDED (not a quarter)
        {"fp": "H1", "form": "10-Q", "start": "2025-01-01", "end": "2025-06-30", "val": 210_000_000_000, "filed": "2025-08-01"},
        # an annual 10-K row must be EXCLUDED from the quarterly series
        {"fp": "FY", "form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "val": 383_000_000_000, "filed": "2025-02-01"},
    ]}},
    "GrossProfit": {"units": {"USD": [
        {"fp": "Q2", "form": "10-Q", "start": "2025-01-01", "end": "2025-03-31", "val": 46_000_000_000, "filed": "2025-05-01"},
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        {"fp": "Q2", "form": "10-Q", "start": "2025-01-01", "end": "2025-03-31", "val": 24_000_000_000, "filed": "2025-05-01"},
    ]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        {"fp": "Q2", "form": "10-Q", "start": "2025-01-01", "end": "2025-03-31", "val": 1.55, "filed": "2025-05-01"},
    ]}},
}}}


def test_sec_quarterly_financials_summary(monkeypatch):
    """The quarterly view selects only 10-Q single quarters (excluding YTD spans and
    the annual 10-K), labels columns by period-end date, and computes per-quarter
    margins plus revenue QoQ and YoY growth."""
    _with_tickers(monkeypatch, {"companyfacts/CIK0000320193": _QFACTS})
    out = edgar.sec_quarterly_financials("AAPL")
    assert "quarterly" in out.lower()
    assert "2025-03-31" in out and "2024-03-31" in out    # period-end column labels
    assert "100.00B" in out and "1.55" in out             # newest revenue + EPS
    assert "gross 46.0%" in out and "net 24.0%" in out    # 46/100, 24/100
    assert "revenue QoQ 2024-12-31→2025-03-31: +5.3%" in out   # (100-95)/95
    assert "revenue YoY 2024-03-31→2025-03-31: +25.0%" in out  # (100-80)/80
    assert "210.00B" not in out                           # 6-month YTD span excluded
    assert "383.00B" not in out                           # annual 10-K excluded


def test_sec_quarterly_financials_single_concept(monkeypatch):
    _with_tickers(monkeypatch, {"companyfacts/CIK0000320193": _QFACTS})
    out = edgar.sec_quarterly_financials("AAPL", concept="RevenueFromContractWithCustomerExcludingAssessedTax")
    assert "quarter ended 2025-03-31" in out and "100.00B" in out
    assert "210.00B" not in out                           # YTD span still excluded
    assert "No SEC CIK" in edgar.sec_quarterly_financials("ZZZZ")   # unknown ticker


def _form4_xml(owner, rel_xml, txns_xml):
    return (f'<?xml version="1.0"?><ownershipDocument>'
            f'<reportingOwner><reportingOwnerId><rptOwnerName>{owner}</rptOwnerName>'
            f'</reportingOwnerId><reportingOwnerRelationship>{rel_xml}'
            f'</reportingOwnerRelationship></reportingOwner>'
            f'<nonDerivativeTable>{txns_xml}</nonDerivativeTable></ownershipDocument>')


def _txn(date, code, shares, ad, price=None):
    px = f'<transactionPricePerShare><value>{price}</value></transactionPricePerShare>' if price else ''
    return (f'<nonDerivativeTransaction><transactionDate><value>{date}</value></transactionDate>'
            f'<transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>'
            f'<transactionAmounts><transactionShares><value>{shares}</value></transactionShares>'
            f'{px}<transactionAcquiredDisposedCode><value>{ad}</value>'
            f'</transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction>')


def test_insider_transactions_summarizes_open_market_vs_routine(monkeypatch):
    """Form 4 parsing separates open-market buys (P) and sales (S) — the conviction
    signals — from routine grants, computes the net, and lists recent transactions
    with each insider's role."""
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["4", "4"],
        "filingDate": ["2025-04-30", "2025-04-15"],
        "accessionNumber": ["0000320193-25-000201", "0000320193-25-000200"],
        "primaryDocument": ["form4a.xml", "form4b.xml"],
        "primaryDocDescription": ["", ""], "items": ["", ""],
    }}}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    xml_a = _form4_xml("Cook Timothy D", "<isOfficer>1</isOfficer><officerTitle>CEO</officerTitle>",
                       _txn("2025-04-30", "S", 50000, "D", "170.00"))
    xml_b = _form4_xml("Levinson Arthur D", "<isDirector>1</isDirector>",
                       _txn("2025-04-15", "P", 5000, "A", "165.00") + _txn("2025-04-10", "A", 1000, "A"))
    monkeypatch.setattr(edgar, "_fetch_text",
                        lambda url, **kw: xml_a if "form4a" in url else xml_b)

    out = edgar.insider_transactions("AAPL")
    assert "INSIDER TRANSACTIONS" in out and "AAPL" in out
    assert "1 buy(s) 5,000 sh" in out and "1 sale(s) 50,000 sh" in out
    assert "net -45,000 sh (net selling)" in out
    assert "1 routine transaction" in out                       # the grant (code A)
    assert "Cook Timothy D (CEO)" in out and "open-market sale" in out
    assert "Levinson Arthur D (Director)" in out and "open-market buy" in out
    assert "8.50M" in out                                       # 50000 * 170


def test_insider_transactions_no_filings_and_unknown(monkeypatch):
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-K"], "filingDate": ["2025-11-01"],
        "accessionNumber": ["x"], "primaryDocument": ["a.htm"],
        "primaryDocDescription": [""], "items": [""],
    }}}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    assert "No recent Form 4" in edgar.insider_transactions("AAPL")   # none present
    assert "No SEC CIK" in edgar.insider_transactions("ZZZZ")         # unknown ticker


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
    # Filing text is third-party content: the model is reminded not to obey any
    # instructions embedded in it (indirect prompt injection).
    assert "NOT instructions" in out

    assert "No passage" in edgar.sec_filing_excerpt("AAPL", "zzznonexistentterm", form_type="10-K")
    assert "topic/phrase" in edgar.sec_filing_excerpt("AAPL", "")   # empty query


def test_filing_summary_builds_fixed_slot_tearsheet(monkeypatch):
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-K"], "filingDate": ["2025-11-01"],
        "accessionNumber": ["0000320193-25-000100"], "primaryDocument": ["aapl-10k.htm"],
        "primaryDocDescription": ["10-K"], "items": [""],
    }}}
    doc = (
        "<html><body>"
        "<p>The Company operates in one business segment and sells products and "
        "services across its markets and operations worldwide to a broad base of "
        "customers through several distribution channels and partners globally.</p>"
        "<p>Net sales revenue increased in the period, driven by higher demand and "
        "growth in services, with continued momentum across the product lineup and "
        "installed base expansion over the prior fiscal year comparison window.</p>"
        "<p>The Company repurchased shares and paid a dividend, returning capital to "
        "shareholders alongside ongoing capital expenditures for its facilities, "
        "reflecting its capital allocation priorities for the fiscal year.</p>"
        "</body></html>"
    )
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    monkeypatch.setattr(edgar, "_fetch_text", lambda url, **kw: doc)
    out = edgar.filing_summary("AAPL")
    assert "Structured summary" in out and "10-K filed 2025-11-01" in out
    # fixed slots present
    for slot in ("## Business & segments", "## Revenue & growth drivers",
                 "## Capital allocation", "## Key risks"):
        assert slot in out
    assert "one business segment" in out            # business passage matched
    assert "Net sales revenue increased" in out     # revenue passage matched
    assert "returning capital to" in out            # capital allocation passage matched
    # a slot with no evidence renders the placeholder, not a fabricated summary
    assert "no matching passage found" in out


def _facts(entity, rev, ni, gp, fy):
    return {"entityName": entity, "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"fy": fy, "fp": "FY", "form": "10-K", "end": f"{fy}-12-31", "val": rev}]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"fy": fy, "fp": "FY", "form": "10-K", "end": f"{fy}-12-31", "val": ni}]}},
        "GrossProfit": {"units": {"USD": [
            {"fy": fy, "fp": "FY", "form": "10-K", "end": f"{fy}-12-31", "val": gp}]}},
    }}}


def test_compare_sec_financials_builds_matrix(monkeypatch):
    edgar._TICKER_CIK.clear()
    tickers = {"0": {"cik_str": 1, "ticker": "AAA", "title": "Aaa"},
               "1": {"cik_str": 2, "ticker": "BBB", "title": "Bbb"}}
    payloads = {
        "company_tickers.json": tickers,
        "companyfacts/CIK0000000001": _facts("Aaa Corp", 1000e9, 250e9, 500e9, 2024),
        "companyfacts/CIK0000000002": _facts("Bbb Corp", 400e9, 40e9, 120e9, 2024),
    }
    monkeypatch.setattr(edgar, "_fetch_json", _router(payloads))

    out = edgar.compare_sec_financials("AAA, BBB")
    assert "AAA vs BBB" in out and "AAA FY2024" in out and "BBB FY2024" in out
    assert "1.00T" in out and "400.00B" in out          # revenues
    assert "Net margin %" in out and "25.0%" in out      # AAA 250/1000
    assert "10.0%" in out                                # BBB 40/400

    # concept mode: one metric across companies
    conc = edgar.compare_sec_financials("AAA, BBB", concept="Net income")
    assert "Net income" in conc and "FY2024" in conc
    assert "250.00B" in conc and "40.00B" in conc

    # fewer than two resolvable -> guidance
    assert "2–6 tickers" in edgar.compare_sec_financials("AAA")


def test_annual_facts_keys_by_end_year_merges_tags_excludes_quarters():
    """Regression: the XBRL 'fy' field is the FILING's year (shared by every
    comparative period in a 10-K), so annual values must be keyed by the period-END
    year; fallback tags must be merged (a company can switch revenue tags); and a
    same-year quarter must never displace the full-year value."""
    facts = {"facts": {"us-gaap": {
        # old tag: data only through FY2022
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"fy": 2022, "fp": "FY", "form": "10-K", "start": "2021-01-31",
             "end": "2022-01-30", "val": 26_914e6, "filed": "2022-02-18"},
        ]}},
        # new tag: FY2025 + FY2026 comparatives, all carrying filing-year fy=2026
        "Revenues": {"units": {"USD": [
            {"fy": 2026, "fp": "FY", "form": "10-K", "start": "2025-01-27",
             "end": "2026-01-25", "val": 215_938e6, "filed": "2026-02-20"},
            {"fy": 2026, "fp": "FY", "form": "10-K", "start": "2024-01-29",
             "end": "2025-01-26", "val": 130_497e6, "filed": "2026-02-20"},
            # a same-end-year QUARTER (~90 days) — must be excluded
            {"fy": 2026, "fp": "FY", "form": "10-K", "start": "2025-10-27",
             "end": "2026-01-25", "val": 57_000e6, "filed": "2026-02-20"},
        ]}},
    }}}
    rows = edgar._annual_facts(
        facts, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"], "USD")
    by_fy = {r["fy"]: r["val"] for r in rows}
    assert by_fy[2026] == 215_938e6   # full-year, NOT the 90-day quarter (57B)
    assert by_fy[2025] == 130_497e6
    assert by_fy[2022] == 26_914e6    # merged in from the old tag
    assert [r["fy"] for r in rows] == [2026, 2025, 2022]   # newest first


def test_tone_score_negative_density():
    assert edgar._tone_score("")[0] is None
    # 10 words; 2 are LM-negative (risk, adverse) -> 20%
    score, n = edgar._tone_score("the risk of adverse events among the many good things")
    assert n == 10 and round(score, 1) == 20.0


def test_filing_tone_trend_scores_each_10k(monkeypatch):
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-K", "10-Q", "10-K"],
        "filingDate": ["2025-11-01", "2025-08-01", "2024-11-01"],
        "accessionNumber": ["0000320193-25-000100", "q", "0000320193-24-000090"],
        "primaryDocument": ["k25.htm", "q.htm", "k24.htm"],
        "primaryDocDescription": ["10-K", "10-Q", "10-K"], "items": ["", "", ""],
    }}}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    # newest 10-K reads more negative than the older one
    docs = {"k25.htm": "<p>" + ("risk adverse litigation " * 5 + "growth ") + "</p>",
            "k24.htm": "<p>" + ("growth strong revenue " * 5 + "risk ") + "</p>"}
    monkeypatch.setattr(edgar, "_fetch_text",
                        lambda url, **kw: next(v for k, v in docs.items() if k in url))
    out = edgar.filing_tone_trend("AAPL", years=3)
    assert "Filing tone trend" in out
    assert "2025-11-01" in out and "2024-11-01" in out
    assert "10-Q" not in out                      # only 10-Ks scored
    assert "MORE negative/cautious" in out        # newest > oldest density


def test_sec_metric_rank_from_frame(monkeypatch):
    edgar._TICKER_CIK.clear()
    tickers = {"0": {"cik_str": 1, "ticker": "MID", "title": "Mid Co"}}
    frame = {"data": [
        {"entityName": "Mid Co", "cik": 1, "val": 400e9, "end": "2024-12-31"},
        {"entityName": "Huge Co", "cik": 2, "val": 900e9, "end": "2024-12-31"},
        {"entityName": "Small Co", "cik": 3, "val": 50e9, "end": "2024-12-31"},
    ]}

    def fetch(url, **kw):
        if "company_tickers.json" in url:
            return tickers
        if "frames/us-gaap" in url:
            return frame
        return {}

    monkeypatch.setattr(edgar, "_fetch_json", fetch)
    out = edgar.sec_metric_rank("MID", concept="Revenue", year=2024)
    assert "Mid Co" in out and "CY2024" in out
    assert "400.00B" in out                       # the company's own value
    assert "#2 of 3" in out                       # one filer (900B) ranks higher
    assert "top 66.7%" in out                     # 2/3
    assert "peer median" in out and "400.00B" in out

    # a company absent from the frame -> friendly note, not a crash
    tickers["1"] = {"cik_str": 99, "ticker": "GONE", "title": "Gone Co"}
    edgar._TICKER_CIK.clear()
    assert "didn't report" in edgar.sec_metric_rank("GONE", concept="Revenue", year=2024)


def test_passages_semantic_optin(monkeypatch):
    """Default keyword ranking is unchanged; when SEC_EDGAR_SEMANTIC is on, the
    keyword shortlist is re-ranked by embedding similarity (mocked here). Both
    paragraphs mention the query, so keyword-matching ties and insertion order
    holds — the embedding rerank is what flips them."""
    text = "\n".join([
        "Alpha paragraph on supply chain logistics and distribution networks worldwide "
        "that clearly exceeds the minimum passage length threshold used by the ranker.",
        "Beta paragraph on supply chain resilience and supplier risk that also clearly "
        "exceeds the minimum passage length threshold used by the ranker for inclusion.",
    ])
    monkeypatch.delenv("SEC_EDGAR_SEMANTIC", raising=False)
    kw = edgar._passages(text, "supply chain", 2)
    assert "logistics" in kw[0].lower()          # keyword tie -> insertion order (Alpha first)

    # Semantic on: mock embeddings so Beta ("resilience") is closest to the query.
    monkeypatch.setenv("SEC_EDGAR_SEMANTIC", "1")
    import financial_research_assistant.embeddings as emb

    def fake_embed(t):
        tl = t.lower()
        if "resilience" in tl:
            return [0.9, 0.1]      # Beta — closest to the query vector
        if "logistics" in tl:
            return [0.3, 0.7]      # Alpha — farther
        return [1.0, 0.0]          # the query itself

    monkeypatch.setattr(emb, "embed_query", fake_embed)
    monkeypatch.setattr(emb, "cosine", lambda a, b: a[0] * b[0] + a[1] * b[1])
    ranked = edgar._passages(text, "supply chain", 2)
    assert len(ranked) == 2 and "resilience" in ranked[0].lower()   # rerank flipped order


def test_edgar_tools_registered():
    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in catalog.TOOLS}
    assert {"sec_filings", "sec_material_events", "sec_financials",
            "sec_filing_search", "sec_filing_excerpt", "filing_summary",
            "compare_sec_financials", "filing_tone_trend", "sec_metric_rank"} <= names


def test_a_non_dict_payload_is_never_cached(monkeypatch):
    """SEC answering with a JSON list (an error envelope, a changed endpoint) used
    to be cached while the call reported ``{}``, so the SECOND lookup of the same
    URL handed the raw list back and the next ``.get()`` raised AttributeError
    mid-turn — a failure that only appeared on the retry."""
    edgar._JSON_CACHE.pop("https://example.test/list.json", None)

    class _Resp:
        def read(self):
            return b'[{"not": "a dict"}]'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(edgar.urllib.request, "urlopen", lambda *a, **kw: _Resp())
    url = "https://example.test/list.json"
    assert edgar._fetch_json(url) == {}
    assert url not in edgar._JSON_CACHE
    assert edgar._fetch_json(url) == {}          # and the second call agrees
    # A cache poisoned by hand is still survivable: the hit path returns dicts only.
    edgar._JSON_CACHE[url] = ["poison"]          # pyright: ignore[reportArgumentType]
    try:
        assert edgar._fetch_json(url) == {}
    finally:
        edgar._JSON_CACHE.pop(url, None)


def test_the_original_filing_is_preferred_over_its_amendment(monkeypatch):
    """An amendment is often a cover page and one restated exhibit, so when a
    10-K/A is the newest matching filing the tearsheet reported it "couldn't locate
    the usual sections" while the real 10-K sat one row below it."""
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-K/A", "10-K"],          # the amendment is newest
        "filingDate": ["2025-12-02", "2025-11-01"],
        "accessionNumber": ["0000320193-25-000110", "0000320193-25-000100"],
        "primaryDocument": ["aapl-10ka.htm", "aapl-10k.htm"],
        "primaryDocDescription": ["10-K/A", "10-K"],
        "items": ["", ""],
    }}}
    body = (
        "<html><body><p>The Company depends on a concentrated supply chain in Asia "
        "and any prolonged disruption to that supply chain could materially and "
        "adversely affect its results of operations in a given period.</p>"
        "</body></html>"
    )
    docs = {"aapl-10ka.htm": "<html><body><p>Amendment No. 1 cover page.</p></body></html>",
            "aapl-10k.htm": body}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    monkeypatch.setattr(
        edgar, "_fetch_text", lambda url, **kw: next(v for k, v in docs.items() if k in url)
    )

    out = edgar.sec_filing_excerpt("AAPL", "supply chain disruption", form_type="10-K")
    assert "10-K filed 2025-11-01" in out and "aapl-10k.htm" in out
    assert "concentrated supply chain" in out

    # Asked for the amendment explicitly, that is what comes back.
    amended = edgar.sec_filing_excerpt("AAPL", "cover page", form_type="10-K/A")
    assert "aapl-10ka.htm" in amended


def test_an_amendment_is_still_used_when_it_is_the_only_filing(monkeypatch):
    """Preferring the original must not mean returning nothing: a filer whose only
    10-K on file is an amendment still resolves to it."""
    subs = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["10-K/A"], "filingDate": ["2025-12-02"],
        "accessionNumber": ["0000320193-25-000110"], "primaryDocument": ["aapl-10ka.htm"],
        "primaryDocDescription": ["10-K/A"], "items": [""],
    }}}
    _with_tickers(monkeypatch, {"submissions/CIK0000320193": subs})
    monkeypatch.setattr(edgar, "_fetch_text", lambda url, **kw: (
        "<html><body><p>The Company restated its concentrated supply chain "
        "disclosure for the period after identifying an error in the original "
        "filing, which could adversely affect comparability.</p></body></html>"
    ))
    out = edgar.sec_filing_excerpt("AAPL", "supply chain", form_type="10-K")
    assert "aapl-10ka.htm" in out and "restated" in out
