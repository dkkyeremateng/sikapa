"""Offline analytics, research, monitoring, and factor tests."""

from .fixtures.statements import BUYSELL_STATEMENT as _BUYSELL_STATEMENT


def test_fundamentals_num_helper_handles_nan():
    from financial_research_assistant.fundamentals import _num

    assert _num(3.5) == 3.5 and _num("4") == 4.0
    assert _num(None) is None and _num("x") is None
    assert _num(float("nan")) is None # NaN (from a missing DataFrame cell) -> None


def test_stock_fundamentals_formats_and_empty(monkeypatch):
    import financial_research_assistant.fundamentals as f

    info = {
        "longName": "Apple Inc.", "sector": "Technology",
        "industry": "Consumer Electronics", "currency": "USD",
        "marketCap": 4_624_460_808_192, "trailingPE": 38.44, "forwardPE": 32.7,
        "trailingEps": 8.19, "currentPrice": 314.86, "fiftyTwoWeekHigh": 323.45,
        "fiftyTwoWeekLow": 201.5, "dividendRate": 1.08, "beta": 1.10,
        "recommendationKey": "buy", "numberOfAnalystOpinions": 42,
        "targetMeanPrice": 316.76,
    }
    monkeypatch.setattr(f, "_fetch_info", lambda s: info)
    out = f.stock_fundamentals("AAPL")
    assert "Apple Inc." in out and "Technology" in out
    assert "4.62T" in out # market cap compacted
    assert "yield 0.34%" in out # computed from rate/price
    assert "consensus buy" in out and "42 analysts" in out
    assert "+0.6% vs price" in out # target upside vs current

    monkeypatch.setattr(f, "_fetch_info", lambda s: {})
    assert "No fundamental data" in f.stock_fundamentals("NOPE")


def test_parse_symbols_dedupes_and_caps():
    from financial_research_assistant.fundamentals import _parse_symbols

    assert _parse_symbols("aapl, msft nvda|goog") == ["AAPL", "MSFT", "NVDA", "GOOG"]
    assert _parse_symbols("AAPL AAPL aapl") == ["AAPL"]        # de-duped
    assert _parse_symbols("a b c d e f", limit=4) == ["A", "B", "C", "D"]  # capped


def test_compare_stocks_builds_normalized_table(monkeypatch):
    import financial_research_assistant.fundamentals as f

    data = {
        "AAPL": {
            "longName": "Apple Inc.", "currency": "USD", "currentPrice": 314.86,
            "marketCap": 4_624_460_808_192, "trailingPE": 38.4, "forwardPE": 32.7,
            "priceToSalesTrailing12Months": 9.1, "revenueGrowth": 0.096,
            "profitMargins": 0.243, "trailingEps": 8.19, "beta": 1.10,
            "recommendationKey": "buy", "numberOfAnalystOpinions": 42,
            "targetMeanPrice": 346.35,  # +10% vs 314.86
        },
        "MSFT": {
            "shortName": "Microsoft", "currency": "USD", "currentPrice": 500.0,
            "marketCap": 3_700_000_000_000, "trailingPE": 34.0, "forwardPE": 29.0,
            "priceToSalesTrailing12Months": 12.0, "revenueGrowth": 0.15,
            "profitMargins": 0.36, "trailingEps": 12.0, "beta": 0.9,
            "dividendRate": 3.0,  # 0.60% yield at 500
            "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 50,
            "targetMeanPrice": 550.0,
        },
    }
    monkeypatch.setattr(f, "_fetch_info", lambda s: data.get(s.upper(), {}))
    out = f.compare_stocks("aapl, msft")
    assert "COMPARE · AAPL vs MSFT" in out
    assert "4.62T" in out and "3.70T" in out          # market caps compacted
    assert "9.6" in out and "15.0" in out             # revenue growth % (0.096 -> 9.6)
    assert "0.60" in out                              # MSFT div yield 3/500
    assert "strong buy" in out                        # consensus key humanized
    assert "+10.0% upside" in out                     # AAPL target vs price


def test_compare_stocks_needs_two_and_reports_missing(monkeypatch):
    import financial_research_assistant.fundamentals as f

    # Fewer than two resolvable tickers -> guidance, not a crash.
    monkeypatch.setattr(f, "_fetch_info", lambda s: {})
    assert "2–4 tickers" in f.compare_stocks("AAPL")
    assert "Couldn't find fundamentals" in f.compare_stocks("NOPE, ALSONOPE")

    # One good, one bad: still compares the good ones AND names the missing.
    good = {"longName": "Apple Inc.", "currency": "USD", "currentPrice": 100.0,
            "marketCap": 1e12, "trailingEps": 5.0}
    monkeypatch.setattr(f, "_fetch_info", lambda s: good if s.upper() == "AAPL" else {})
    # need >=2 found for a table, so give two good + one missing
    monkeypatch.setattr(f, "_fetch_info",
                        lambda s: good if s.upper() in ("AAPL", "MSFT") else {})
    out = f.compare_stocks("AAPL, MSFT, BOGUS")
    assert "AAPL vs MSFT" in out
    assert "No data for: BOGUS" in out


def test_analyst_ratings_formats(monkeypatch):
    import financial_research_assistant.fundamentals as f

    monkeypatch.setattr(f, "_fetch_info", lambda s: {
        "currency": "USD", "recommendationKey": "buy",
        "recommendationMean": 2.0, "numberOfAnalystOpinions": 40,
    })
    monkeypatch.setattr(f, "_fetch_price_targets", lambda s: {
        "current": 300.0, "low": 215.0, "mean": 330.0, "median": 320.0, "high": 400.0,
    })
    monkeypatch.setattr(f, "_fetch_rating_changes", lambda s: [
        {"date": "2026-07-01", "firm": "Big Bank", "from": "Hold",
         "to": "Buy", "action": "up"},
    ])
    out = f.analyst_ratings("AAPL")
    assert "consensus: buy from 40 analysts" in out
    assert "mean 330.00" in out
    assert "+10.0% to mean" in out # (330-300)/300
    assert "Big Bank" in out and "Hold → Buy" in out


def test_earnings_calendar_formats(monkeypatch):
    import financial_research_assistant.fundamentals as f

    monkeypatch.setattr(f, "_fetch_calendar", lambda s: {
        "Earnings Date": [__import__("datetime").date(2026, 7, 30)],
        "Earnings Average": 1.89, "Earnings Low": 1.83, "Earnings High": 1.99,
        "Ex-Dividend Date": __import__("datetime").date(2026, 5, 11),
        "Dividend Date": __import__("datetime").date(2026, 5, 14),
    })
    monkeypatch.setattr(f, "_fetch_earnings_history", lambda s: [
        {"date": "2026-04-30", "estimate": 1.94, "reported": 2.01, "surprise": 3.46},
        {"date": "2026-01-29", "estimate": 2.67, "reported": None, "surprise": None},
    ])
    out = f.earnings_calendar("AAPL")
    assert "next earnings: 2026-07-30" in out and "consensus EPS 1.89" in out
    assert "ex-dividend: 2026-05-11" in out
    assert "2026-04-30" in out and "1.94 → 2.01" in out and "+3.5%" in out
    assert "2.67 → —" in out # missing reported renders as a dash


def test_dividend_projection(monkeypatch, tmp_path):
    import financial_research_assistant.fundamentals as f
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT) # positions: AMZN qty 4 basis 400, VOO qty 10 basis 500

    fake = {
        "AMZN": {"currency": "USD", "dividendRate": 1.0, "currentPrice": 160.0},
        "VOO": {"currency": "USD", "dividendRate": 2.0, "currentPrice": 60.0},
    }
    monkeypatch.setattr(f, "_fetch_info", lambda sym: fake.get(sym.upper(), {}))
    out = f.dividend_projection()
    # VOO: 10 * 2.0 = 20/yr; AMZN: 4 * 1.0 = 4/yr; total 24 USD.
    assert "2 paying holding(s)" in out
    assert "VOO" in out and "20.00 USD" in out
    assert "AMZN" in out and "yield-on-cost 1.00%" in out # 4/400
    assert "TOTAL ≈ 24.00 USD/yr" in out


def test_dividend_projection_empty_and_no_payers(monkeypatch, tmp_path):
    import financial_research_assistant.fundamentals as f
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    assert "No positions found" in f.dividend_projection()

    s.import_statement(_BUYSELL_STATEMENT)
    monkeypatch.setattr(f, "_fetch_info", lambda sym: {}) # no dividend data
    assert "None of your imported positions pay a dividend" in f.dividend_projection()


async def test_research_gather_runs_all_sections(monkeypatch):
    """gather_sections fans out over every section function and returns them labeled;
    a section that raises degrades to an 'unavailable' note, not a failure."""
    from financial_research_assistant import research

    def fake_fns(symbol):
        return {
            "Fundamentals": lambda: f"fundamentals {symbol}",
            "Recent news": lambda: (_ for _ in ()).throw(RuntimeError("rate limited")),
        }

    monkeypatch.setattr(research, "_section_fns", fake_fns)
    out = await research.gather_sections("AAPL")
    labels = {label for label, _ in out}
    assert labels == {"Fundamentals", "Recent news"}
    text = dict(out)
    assert text["Fundamentals"] == "fundamentals AAPL"
    assert "unavailable" in text["Recent news"] and "rate limited" in text["Recent news"]


async def test_research_ticker_fake_end_to_end(monkeypatch, tmp_path):
    """research_ticker gathers (fake stubs), synthesizes (fake), and saves the
    markdown report — fully offline."""
    from financial_research_assistant import research

    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path / "reports"))
    res = await research.research_ticker("aapl", fake=True)
    assert res["symbol"] == "AAPL"
    assert res["report"].startswith("# Research report: AAPL")
    assert "FAKE-OK" in res["report"]
    saved = list((tmp_path / "reports").glob("research-AAPL-*.md"))
    assert len(saved) == 1 and saved[0].read_text() == res["report"]
    # all seven sections were gathered
    assert len(res["sections"]) == 7


def test_research_report_tool_returns_labeled_findings(monkeypatch):
    """The research_report tool gathers findings (sequentially) and frames them for
    the agent to synthesize — without a nested model call."""
    from financial_research_assistant import research

    monkeypatch.setattr(research, "_section_fns", lambda sym: {
        "Fundamentals": lambda: f"{sym} P/E 30",
        "Recent news": lambda: f"{sym} up on earnings",
    })
    out = research.research_report("MSFT")
    assert "synthesize" in out.lower() and "MSFT" in out
    assert "## Fundamentals" in out and "MSFT P/E 30" in out
    assert "## Recent news" in out


def test_bull_bear_debate_frames_findings(monkeypatch):
    from financial_research_assistant import research

    monkeypatch.setattr(research, "_section_fns", lambda sym: {
        "Fundamentals": lambda: f"{sym} forward P/E 25",
        "Recent news": lambda: f"{sym} guidance raised",
    })
    # recall_lessons returns [] with long-term memory off (default), so no monkeypatch.
    out = research.bull_bear_debate("NVDA")
    assert "Bull case" in out and "Bear case" in out and "Verdict" in out
    assert "NVDA" in out
    assert "## Fundamentals" in out and "NVDA forward P/E 25" in out
    assert "## Recent news" in out


def test_recent_move_summarizes_window_and_last_session(monkeypatch):
    from financial_research_assistant import research
    import financial_research_assistant.tools as tools

    # oldest -> newest; last session +2% (100->102), window (95->102) +7.37%
    series = [("2026-07-14", 95.0), ("2026-07-15", 97.0), ("2026-07-16", 99.0),
              ("2026-07-17", 100.0), ("2026-07-18", 102.0)]
    monkeypatch.setattr(tools, "_fetch_daily", lambda s, d: series)
    out = research._recent_move("AAPL", days=4)
    assert "AAPL last close 102.00 on 2026-07-18" in out
    assert "+2.00% vs prior session" in out
    assert "+7.37%" in out                       # 95 -> 102 over the 4-session window

    monkeypatch.setattr(tools, "_fetch_daily", lambda s, d: [])
    assert research._recent_move("NOPE", days=4) is None


def test_explain_stock_move_gathers_move_ratings_news(monkeypatch):
    from financial_research_assistant import research
    import financial_research_assistant.tools as tools
    import financial_research_assistant.fundamentals as f

    series = [("2026-07-16", 99.0), ("2026-07-17", 100.0), ("2026-07-18", 110.0)]
    monkeypatch.setattr(tools, "_fetch_daily", lambda s, d: series)
    monkeypatch.setattr(tools, "web_search",
                        lambda q, max_results=5: "1. Big beat (Reuters · 2026-07-18)\n   url")
    monkeypatch.setattr(f, "_fetch_rating_changes", lambda s, limit=5: [
        {"date": "2026-07-18", "firm": "Big Bank", "from": "Hold", "to": "Buy", "action": "up"},
    ])
    out = research.explain_stock_move("AAPL", days=5)
    # directive to attribute + cite by number, and the three labeled sections
    assert "cite news items by their number" in out or "cite" in out.lower()
    assert "## Price move" in out and "+10.00% vs prior session" in out
    assert "## Recent analyst rating changes" in out and "Big Bank" in out
    assert "## Recent news" in out and "Big beat" in out


def test_explain_stock_move_no_price_data(monkeypatch):
    from financial_research_assistant import research
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(tools, "_fetch_daily", lambda s, d: [])
    out = research.explain_stock_move("NOPE")
    assert "No recent price data" in out


def test_digest_empty_store(monkeypatch, tmp_path):
    from financial_research_assistant.monitor import build_digest

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    assert "No holdings to monitor" in build_digest()


def test_digest_flags_movers_earnings_exdivs(monkeypatch, tmp_path):
    """The digest reads imported holdings and surfaces price movers beyond the
    threshold, upcoming earnings in the window, and upcoming ex-dividend dates —
    all offline via the mockable fetch helpers, with an injected 'today'."""
    import datetime as _dt
    import re

    import financial_research_assistant.tools as tools
    import financial_research_assistant.fundamentals as fund
    from financial_research_assistant import statements as s
    from financial_research_assistant.monitor import build_digest

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT) # holdings: AMZN, VOO

    # AMZN jumps +8% over the window; VOO is flat (not a mover).
    def fake_daily(sym, days):
        if sym.upper() == "AMZN":
            return [("2026-07-10", 100.0), ("2026-07-15", 108.0)]
        return [("2026-07-10", 60.0), ("2026-07-15", 60.1)]

    def fake_calendar(sym):
        if sym.upper() == "AMZN":
            return {"Earnings Date": [_dt.date(2026, 7, 20)], "Earnings Average": 1.25}
        return {"Ex-Dividend Date": _dt.date(2026, 7, 18)} # VOO ex-div soon

    monkeypatch.setattr(tools, "_fetch_daily", fake_daily)
    monkeypatch.setattr(fund, "_fetch_calendar", fake_calendar)

    out = build_digest(today=_dt.date(2026, 7, 15), move_threshold=5.0, earnings_within=14)
    assert re.search(r"AMZN\s+\+8\.00%", out) # mover symbol and value
    assert "VOO" not in out.split("Movers")[1].split("Upcoming earnings")[0] # VOO not a mover
    assert re.search(r"AMZN\s+2026-07-20\s+· consensus EPS 1\.25", out)
    assert re.search(r"VOO\s+2026-07-18", out) # ex-dividend symbol and date


def test_digest_window_excludes_far_events(monkeypatch, tmp_path):
    """Earnings/ex-div dates outside the window are not listed, and a sub-threshold
    move is reported as 'none'."""
    import datetime as _dt

    import financial_research_assistant.tools as tools
    import financial_research_assistant.fundamentals as fund
    from financial_research_assistant import statements as s
    from financial_research_assistant.monitor import build_digest

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)

    monkeypatch.setattr(tools, "_fetch_daily",
        lambda sym, days: [("2026-07-10", 100.0), ("2026-07-15", 101.0)])
    monkeypatch.setattr(fund, "_fetch_calendar",
        lambda sym: {"Earnings Date": [_dt.date(2026, 9, 1)]}) # far off

    out = build_digest(today=_dt.date(2026, 7, 15), move_threshold=5.0, earnings_within=14)
    assert "none beyond the threshold" in out
    assert "none scheduled in the window" in out
    assert "2026-09-01" not in out


def test_tax_loss_harvest_flags_loss_and_wash(monkeypatch, tmp_path):
    import datetime as _dt

    import financial_research_assistant.analytics as a
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT) # open lot: AMZN 4 @ 100 (2024-01-01)

    monkeypatch.setattr(a, "_current_price", lambda sym, fallback=None: 80.0)
    # today 19 days after the buy -> short-term loss AND wash-sale window.
    out = a.tax_loss_harvest(today=_dt.date(2024, 1, 20))
    assert "AMZN" in out
    assert "loss -80.00" in out and "ST -80.00" in out
    assert "est. tax benefit 28.00" in out # 80 * 0.35 short-term rate
    assert "wash-sale risk" in out # bought within 30d


def test_tax_loss_harvest_no_candidates(monkeypatch, tmp_path):
    import datetime as _dt

    import financial_research_assistant.analytics as a
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_BUYSELL_STATEMENT)
    monkeypatch.setattr(a, "_current_price", lambda sym, fallback=None: 150.0) # a gain
    out = a.tax_loss_harvest(today=_dt.date(2026, 7, 15))
    assert "No tax-loss-harvesting candidates" in out


def test_tax_loss_harvest_empty_store(monkeypatch, tmp_path):
    import financial_research_assistant.analytics as a

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    assert "No open lots" in a.tax_loss_harvest()


def test_portfolio_lookthrough_aggregates_and_finds_hidden_concentration(monkeypatch, tmp_path):
    """Look-through expands ETF holdings: sector exposure is complete, and a stock
    held directly AND inside an ETF shows combined true exposure with its sources."""
    import financial_research_assistant.analytics as a
    import financial_research_assistant.fundamentals as f
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    stmt = (
        "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n"
        "Open Positions,Data,Summary,ETFs,USD,VOO,1,1,1000,1000,1000,1000,0,\n" # ETF, value 1000
        "Open Positions,Data,Summary,Stocks,USD,AAPL,1,1,500,500,500,500,0,\n" # stock, value 500
    )
    s.import_statement(stmt)

    monkeypatch.setattr(f, "_fetch_fund_data", lambda sym: {
        "sectors": {"technology": 0.4, "healthcare": 0.3, "financial_services": 0.3},
        "holdings": [("AAPL", "Apple Inc", 0.07), ("MSFT", "Microsoft", 0.05)],
    } if sym.upper() == "VOO" else {})
    monkeypatch.setattr(f, "_fetch_info", lambda sym: {"sector": "Technology"})

    out = a.portfolio_lookthrough()
    # Sector: VOO tech 400 + AAPL direct 500 = 900 of 1500 -> 60.0%.
    assert "Technology" in out and "60.0%" in out
    # AAPL true exposure: 500 direct + 70 via VOO = 570 of 1500 -> 38.0%, both sources.
    assert "AAPL" in out and "38.0%" in out
    assert "held directly" in out and "via VOO" in out
    # Covered fraction: (500 + 70 + 50) / 1500 = 41%.
    assert "covers 41% of assets" in out


def test_portfolio_lookthrough_empty(monkeypatch, tmp_path):
    import financial_research_assistant.analytics as a

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    assert "No positions found" in a.portfolio_lookthrough()


async def test_research_portfolio_fake_end_to_end(monkeypatch, tmp_path):
    """research_portfolio gathers the 7 portfolio sections (fake), synthesizes, and
    saves a PORTFOLIO report — fully offline."""
    from financial_research_assistant import research

    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path / "reports"))
    res = await research.research_portfolio(fake=True)
    assert res["symbol"] == "PORTFOLIO"
    assert res["report"].startswith("# Research report: Portfolio")
    assert len(res["sections"]) == 7
    saved = list((tmp_path / "reports").glob("research-PORTFOLIO-*.md"))
    assert len(saved) == 1 and saved[0].read_text() == res["report"]


def test_ff_csv_parser():
    """The Fama-French daily CSV parser reads the header, converts percent→decimal,
    and keys rows by ISO date; RF is excluded from the factor-name list."""
    from financial_research_assistant.factors import _parse_ff_csv

    text = (
        "This file was created ...\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "20260102, 1.00, -0.50, 0.25, 0.01\n"
        "20260103, -0.20, 0.10, -0.05, 0.01\n"
        "Copyright 2026\n"
    )
    names, by_date = _parse_ff_csv(text)
    assert names == ["Mkt-RF", "SMB", "HML"] # RF excluded from factors
    assert by_date["2026-01-02"]["Mkt-RF"] == 0.01 # 1.00% -> 0.01
    assert by_date["2026-01-02"]["SMB"] == -0.005
    assert by_date["2026-01-03"]["RF"] == 0.0001
    assert len(by_date) == 2


def test_factor_exposure_recovers_known_loadings(monkeypatch):
    """End-to-end regression check: build a ticker whose excess return is exactly
    1.5·Mkt + 0.5·SMB (no value tilt, no alpha), and confirm factor_exposure
    recovers those loadings, ~0 HML, ~0 alpha, and R²≈1."""
    import re

    from financial_research_assistant import factors

    # 60 synthetic days with non-collinear factor variation.
    names = ["Mkt-RF", "SMB", "HML"]
    ff = {}
    returns = []
    for i in range(60):
        d = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
        mkt, smb, hml, rf = (i % 7 - 3) / 100, (i % 5 - 2) / 100, (i % 3 - 1) / 100, 0.0001
        ff[d] = {"Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": rf}
        returns.append((d, 1.5 * mkt + 0.5 * smb + rf)) # excess = 1.5mkt + 0.5smb

    monkeypatch.setattr(factors, "_fetch_ff_factors", lambda five_factor=False: (names, ff))
    monkeypatch.setattr(factors, "_ticker_returns", lambda sym, days: returns)

    out = factors.factor_exposure("AAPL", days=90)
    assert re.search(r"market\s+\(Mkt-RF\)\s+\+1\.50", out)
    assert re.search(r"size\s+\(SMB\)\s+\+0\.50", out) and "small-cap tilt" in out
    assert re.search(r"value\s+\(HML\)", out) and "(neutral)" in out # no value/growth tilt
    assert "alpha " in out and "0.0%/yr" in out # ~zero alpha
    assert re.search(r"R²\s+1\.00", out) # factors fully explain it


def test_factor_exposure_no_data(monkeypatch, tmp_path):
    from financial_research_assistant import factors

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    # empty symbol -> portfolio path -> no holdings
    assert "Import a portfolio" in factors.factor_exposure("")


def test_correlation_matrix(monkeypatch):
    import financial_research_assistant.analytics as a
    import financial_research_assistant.tools as tools

    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    closes = {
        "AAPL": [100.0, 110.0, 105.0, 115.0],
        "MSFT": [50.0, 55.0, 52.5, 57.5], # proportional to AAPL -> corr +1.00
    }
    monkeypatch.setattr(tools, "_aligned_closes", lambda syms, days: (dates, closes))
    out = a.correlation_matrix("AAPL, MSFT", days=90)
    assert "AAPL" in out and "MSFT" in out
    assert "1.00" in out # diagonal + perfect corr
    assert "correlation" in out.lower()
    # fewer than two tickers is rejected
    assert "at least two" in a.correlation_matrix("AAPL")


def _risk_series(base: float, n: int = 30) -> list[float]:
    """A gently trending, oscillating close series so returns have non-zero
    variance and a real drawdown — deterministic, offline."""
    return [base * (1 + 0.03 * ((i % 5) - 2) / 100 + 0.002 * i) for i in range(n)]


def _two_position_stmt(a_sym="AAPL", a_val=600, b_sym="MSFT", b_val=400) -> str:
    cols = ("Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,"
            "Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n")
    row = "Open Positions,Data,Summary,Stocks,USD,{s},1,1,{v},{v},{v},{v},0,\n"
    return cols + row.format(s=a_sym, v=a_val) + row.format(s=b_sym, v=b_val)


def test_portfolio_risk_computes_metrics(monkeypatch, tmp_path):
    """portfolio_risk value-weights the current holdings into one return series and
    reports volatility, drawdown, Sharpe, and beta vs the benchmark."""
    import financial_research_assistant.analytics as a
    import financial_research_assistant.tools as tools
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_two_position_stmt())  # AAPL 600 (60%), MSFT 400 (40%)

    dates = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(30)]
    closes = {"AAPL": _risk_series(100.0), "MSFT": _risk_series(50.0),
              "SPY": _risk_series(400.0)}
    monkeypatch.setattr(tools, "_aligned_closes",
                        lambda syms, days, **kw: (dates, {s2: closes[s2] for s2 in syms if s2 in closes}))

    out = a.portfolio_risk()
    assert "PORTFOLIO RISK" in out
    assert "annualized volatility" in out and "max drawdown" in out
    assert "Sharpe" in out and "beta vs SPY" in out
    assert "AAPL 60%" in out and "MSFT 40%" in out  # value weights
    assert "covers 100% of portfolio value" in out


def test_portfolio_risk_excludes_unpriced_and_reports_coverage(monkeypatch, tmp_path):
    """A holding with no price history is excluded (not treated as risk-free), the
    covered share of portfolio value is reported, and the excluded name is named."""
    import financial_research_assistant.analytics as a
    import financial_research_assistant.tools as tools
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_two_position_stmt("AAPL", 600, "FOO", 400))  # FOO has no data

    dates = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(30)]
    closes = {"AAPL": _risk_series(100.0), "SPY": _risk_series(400.0)}  # no FOO
    monkeypatch.setattr(tools, "_aligned_closes",
                        lambda syms, days, **kw: (dates, {s2: closes[s2] for s2 in syms if s2 in closes}))

    out = a.portfolio_risk()
    assert "covers 60% of portfolio value" in out  # 600 of 1000
    assert "no price history for FOO" in out


def test_portfolio_risk_empty_store(monkeypatch, tmp_path):
    import financial_research_assistant.analytics as a
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    out = a.portfolio_risk()
    assert "No positions found" in out and "risk_metrics" in out  # points at per-ticker tool


def test_etf_exposure(monkeypatch):
    import financial_research_assistant.fundamentals as f

    monkeypatch.setattr(f, "_fetch_fund_data", lambda sym: {
        "sectors": {"technology": 0.30, "healthcare": 0.15},
        "holdings": [("AAPL", "Apple Inc", 0.07), ("MSFT", "Microsoft Corp", 0.05)],
    })
    out = f.etf_exposure("VOO")
    assert "technology" in out and "30.0%" in out
    assert "AAPL" in out and "Apple Inc" in out and "7.00%" in out

    monkeypatch.setattr(f, "_fetch_fund_data", lambda sym: {})
    assert "No fund look-through" in f.etf_exposure("AAPL")
