"""The recommender: model-free candidates, a typed verdict, fit checks, the ledger.

Every data source is faked with something that varies by symbol and by date, so
a window or a filter that is wrong changes the result.
"""

import asyncio
import json
from datetime import date, datetime, timedelta

import pytest

from financial_research_assistant import (
    autonomy, catalog, channels, etfs, fundamentals, journal, periodic, profile, recommend,
    screener, statements, tools, universes,
)

TODAY = date(2026, 9, 29)


def _trend(daily: float, n: int = 420, base: float = 100.0, wobble: float = 0.004):
    """A weekday series ending TODAY: a steady trend with a small alternating wobble."""
    days, d = [], TODAY
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    price, out = base, []
    for i, day in enumerate(days):
        price *= 1 + daily + (wobble if i % 2 else -wobble)
        out.append((day.isoformat(), round(price, 4)))
    return out


PRICES = {
    "WIN": _trend(0.0020),   # strong, orderly uptrend
    "MEH": _trend(0.0002),
    "DOWN": _trend(-0.0015),
    "AAPL": _trend(0.0008),
    "SPY": _trend(0.0004),
    "VEA": _trend(0.0003), "IEFA": _trend(0.0003), "BND": _trend(0.0001),
    "AGG": _trend(0.0001), "EFA": _trend(0.0002),
}
INFO = {
    "WIN": {"sector": "Technology", "longName": "Winner Inc"},
    "QUAL1": {"forwardPE": 15.0, "profitMargins": 0.25, "returnOnEquity": 0.30,
              "revenueGrowth": 0.08, "sector": "Industrials"},
    "DIV1": {"dividendRate": 4.0, "currentPrice": 100.0, "payoutRatio": 0.5,
             "sector": "Utilities"},
    "AAPL": {"sector": "Technology"},
    "VEA": {"netExpenseRatio": 0.05, "totalAssets": 2e11},
    "IEFA": {"netExpenseRatio": 0.07, "totalAssets": 1e11},
    "BND": {"netExpenseRatio": 0.03, "totalAssets": 3e11},
    "AGG": {"netExpenseRatio": 0.03, "totalAssets": 1e11},
}


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **_kw: PRICES.get(sym.upper(), [])[-max(days // 7 * 5, 10):])
    monkeypatch.setattr(fundamentals, "_fetch_info", lambda sym: dict(INFO.get(sym.upper(), {})))
    monkeypatch.setattr(fundamentals, "_fetch_fund_data", lambda sym: {})
    monkeypatch.setattr(fundamentals, "stock_fundamentals", lambda sym: f"{sym}: forward P/E 20.0")
    monkeypatch.setattr(fundamentals, "analyst_ratings", lambda sym: f"{sym}: 12 buy, 3 hold")
    monkeypatch.setattr(fundamentals, "etf_exposure", lambda sym: f"{sym}: diversified")
    monkeypatch.setattr(tools, "risk_metrics", lambda sym, days=365: f"{sym}: volatility 22.0%")
    monkeypatch.setattr(tools, "web_search", lambda q, max_results=3: f"news for {q}")
    monkeypatch.setattr(screener, "_fetch_sp500", lambda: ["WIN", "MEH", "DOWN"])
    monkeypatch.setattr(screener, "_DEFAULT_UNIVERSE", ["QUAL1", "DIV1"])
    monkeypatch.setattr(statements, "query_positions", lambda account=None: [
        {"symbol": "AAPL", "asset_category": "STK", "currency": "USD",
         "close_price": PRICES["AAPL"][-1][1], "value": 9000.0},
        {"symbol": "BND", "asset_category": "STK", "currency": "USD",
         "close_price": PRICES["BND"][-1][1], "value": 1000.0},
    ])
    monkeypatch.setattr(statements, "default_account", lambda: "U1")
    monkeypatch.setattr(statements, "list_imports", lambda: [
        {"account": "U1", "period": "August 30, 2026 - September 28, 2026"}])
    from financial_research_assistant import stocks

    def no_filings(symbol, quarters=8, **_kw):
        raise RuntimeError("no SEC data in tests")

    monkeypatch.setattr(stocks, "build_stock_brief", no_filings)
    monkeypatch.setattr(journal, "_spot", lambda sym: (123.45, "2026-09-29"))
    monkeypatch.delenv("FRA_IDEAS_SHADOW", raising=False)


# --- the profile --------------------------------------------------------------------------


def test_the_profile_starts_generic_and_is_validated():
    assert not profile.is_set()
    p = profile.load()
    assert sum(p["targets"].values()) == 100 and p["risk"] == "moderate"
    with pytest.raises(ValueError, match="adds up to 90%"):
        profile.save({"targets": {"us_equity": 60, "bonds": 30}})
    with pytest.raises(ValueError, match="unknown asset class"):
        profile.save({"targets": {"stonks": 100}})
    out = profile.update_investor_profile("max_position_pct", "8%")
    assert out.startswith("Profile updated") and profile.load()["max_position_pct"] == 8.0
    assert profile.is_set()
    profile.update_investor_profile("targets", "us_equity=60, intl_equity=20, bonds=20")
    assert profile.load()["targets"] == {"us_equity": 60.0, "intl_equity": 20.0, "bonds": 20.0}
    assert "Unknown setting" in profile.update_investor_profile("colour", "blue")
    profile.add_to_watchlist("nvda, amd")
    profile.add_to_watchlist("NVDA")
    assert profile.load()["watchlist"] == ["NVDA", "AMD"]


# --- universes and funds ---------------------------------------------------------------------


def test_the_catalog_is_sound():
    funds = universes.catalog()
    assert len(funds) >= 50
    assert len({f["symbol"] for f in funds}) == len(funds), "no duplicate tickers"
    assert {f["asset_class"] for f in funds} <= set(profile.ASSET_CLASSES)
    assert all(f["benchmark"] != f["symbol"] for f in funds), "an idea scored against itself"


def test_gaps_are_measured_in_the_profiles_asset_classes(world):
    book = periodic.load_book()
    assert universes.allocation(book) == pytest.approx({"us_equity": 90.0, "bonds": 10.0})
    gaps = universes.gaps(book, {"us_equity": 60, "intl_equity": 20, "bonds": 20})
    assert [(g[0], g[1], round(g[2])) for g in gaps] == [("intl_equity", 20.0, 0), ("bonds", 20.0, 10)]
    assert universes.asset_class_of("IBIT") == "crypto"
    assert universes.asset_class_of("BTC-USD") == "crypto"


def test_expense_ratios_are_read_in_the_unit_each_field_uses():
    assert etfs.expense_ratio_pct({"netExpenseRatio": 0.03}) == 0.03
    assert etfs.expense_ratio_pct({"annualReportExpenseRatio": 0.0003}) == pytest.approx(0.03)
    assert etfs.expense_ratio_pct({}) is None


def test_trailing_returns_name_their_window():
    series = _trend(0.0004, n=800)
    one = etfs._annualized(series, 1)
    assert one["to"] == "2026-09-29" and one["from"] <= "2025-09-29"
    assert etfs._annualized(series, 5) is None, "the series doesn't reach back five years"


# --- candidates -------------------------------------------------------------------------------


def test_candidates_come_from_every_source_and_are_filtered(world):
    profile.save({"targets": {"us_equity": 60, "intl_equity": 20, "bonds": 20},
                  "exclude": ["DIV1"], "watchlist": ["MEH"]})
    journal.save_entries([{"id": "t1", "symbol": "QUAL1", "source": "recommender",
                           "status": "open", "opened": datetime.now().isoformat()}])
    kept, dropped, label = recommend.candidates(periodic.load_book(), profile.load(), "deep")
    by = {c.symbol: c for c in kept}
    assert label == "S&P 500"
    assert by["VEA"].source == "gap" and by["VEA"].benchmark == "EFA"
    assert by["WIN"].source == "momentum"
    assert "over 6 months (" in by["WIN"].reason, "the momentum figure carries its window"
    assert "DOWN" not in by, "a downtrend below its 200-day average is not momentum"
    assert by["AAPL"].held and by["AAPL"].source == "holding"
    assert by["MEH"].source in ("watchlist", "momentum")
    reasons = dict(dropped)
    assert reasons["DIV1"] == "excluded in your profile"
    assert reasons["QUAL1"].startswith("recommended within the last 30 days")


def test_light_mode_reviews_only_holdings_with_something_to_say(world):
    profile.save({"max_position_pct": 95})
    kept, _dropped, _l = recommend.candidates(periodic.load_book(), profile.load(), "light")
    assert "AAPL" not in {c.symbol for c in kept}
    profile.save({"max_position_pct": 50})
    kept, _dropped, _l = recommend.candidates(periodic.load_book(), profile.load(), "light")
    aapl = next(c for c in kept if c.symbol == "AAPL")
    assert "above your 50% position cap" in aapl.reason


def test_crypto_needs_the_profiles_permission(world):
    profile.save({"watchlist": ["IBIT"]})
    _kept, dropped, _l = recommend.candidates(periodic.load_book(), profile.load(), "deep")
    assert dict(dropped)["IBIT"] == "crypto is excluded in your profile"
    book = {**periodic.load_book()}
    book["holdings"] = book["holdings"] + [{"symbol": "BTC-USD", "value": 500.0, "units": 0.01,
                                            "currency": "USD", "weight_pct": 5.0, "name": ""}]
    import financial_research_assistant.recommend as r
    kept, dropped, _l = r.candidates(book, {**profile.load(), "max_position_pct": 1}, "deep")
    assert "BTC-USD" not in dict(dropped), "a coin you hold is still reviewed"


# --- the verdict ---------------------------------------------------------------------------------


def _cand(**kw):
    base = dict(symbol="WIN", kind="stock", source="momentum", score=70, reason="r",
                facts={"price": {"price": 100.0}})
    base.update(kw)
    return recommend.Candidate(**base)


def test_a_verdict_is_typed_and_its_one_number_is_checked():
    good = {"verdict": "buy-candidate", "conviction": 4, "thesis": "A durable trend with support.",
            "key_risks": ["valuation"], "horizon_days": 120, "evidence_ids": ["E1", "E9"],
            "invalidation": {"price": 90.0, "event": "a break of the trend"},
            "price_target": 150}
    v, why = recommend.validate_verdict(good, _cand(), {"E1", "E2"})
    assert why == "" and v["evidence_ids"] == ["E1"] and v["invalidation_price"] == 90.0
    assert "price_target" not in v, "no field for a target survives"
    wrong_side = dict(good, invalidation={"price": 110.0, "event": ""})
    assert recommend.validate_verdict(wrong_side, _cand(), {"E1"})[0]["invalidation_price"] is None
    assert recommend.validate_verdict(dict(good, verdict="add"), _cand(), {"E1"})[0] is None
    held_ok = recommend.validate_verdict(dict(good, verdict="trim"), _cand(held=True), {"E1"})[0]
    assert held_ok["verdict"] == "trim"
    assert recommend.validate_verdict(dict(good, conviction=9), _cand(), {"E1"})[1] == "conviction must be 1-5"
    assert recommend.validate_verdict(dict(good, evidence_ids=["E7"]), _cand(), {"E1"})[1] == "cites no evidence"


def test_a_verdict_citing_figures_not_in_the_evidence_is_retried(monkeypatch):
    replies = iter([
        json.dumps({"verdict": "buy-candidate", "conviction": 4, "horizon_days": 90,
                    "thesis": "Revenue will grow 37.5% next year on new products.",
                    "key_risks": ["x"], "evidence_ids": ["E1"], "invalidation": {}}),
        json.dumps({"verdict": "buy-candidate", "conviction": 3, "horizon_days": 90,
                    "thesis": "The trend is strong and volatility is 22.0% a year.",
                    "key_risks": ["x"], "evidence_ids": ["E1"], "invalidation": {}}),
    ])

    async def ask(system, user, **kw):
        return next(replies)

    monkeypatch.setattr(autonomy, "ask", ask)
    ev = [("E1", "Risk", "volatility 22.0%")]
    v, why = asyncio.run(recommend.judge(_cand(), ev))
    assert why == "" and v["conviction"] == 3


# --- fit -----------------------------------------------------------------------------------------


def test_fit_checks_drop_what_breaks_the_book(world):
    prof = {**profile.load(), "max_sector_pct": 35, "max_expense_ratio_pct": 0.2}
    book = periodic.load_book()
    idea = recommend.Idea(candidate=_cand(), verdict="buy-candidate", conviction=4, thesis="t",
                          key_risks=[], horizon_days=90, evidence_ids=["E1"])
    assert "Technology is already 90.0%" in recommend.fit_check(idea, book, prof, {"Technology": 90.0})
    assert recommend.fit_check(idea, book, prof, {"Technology": 10.0}) == ""
    trim = recommend.Idea(candidate=_cand(held=True), verdict="trim", conviction=4, thesis="t",
                          key_risks=[], horizon_days=90, evidence_ids=["E1"])
    assert recommend.fit_check(trim, book, prof, {"Technology": 90.0}) == "", "a trim never breaks a cap"
    fund = recommend.Idea(candidate=_cand(symbol="VEA", kind="etf", facts={
        "etf": {"expense_ratio_pct": 0.5, "overlap": {"direct_pct": 0, "via_funds": {}}}}),
        verdict="buy-candidate", conviction=3, thesis="t", key_risks=[], horizon_days=90,
        evidence_ids=["E1"])
    assert "expense ratio 0.50%" in recommend.fit_check(fund, book, prof, {})


# --- a run, end to end --------------------------------------------------------------------------


def test_a_run_records_every_call_against_its_asset_classes_benchmark(world):
    profile.save({"targets": {"us_equity": 60, "intl_equity": 20, "bonds": 20}})
    run_ = asyncio.run(recommend.run("deep", fake=True))
    assert run_["ideas"], run_["dropped"]
    entries = {e["symbol"]: e for e in journal.load_entries()}
    for row in run_["ideas"]:
        e = entries[row["symbol"]]
        assert e["source"] == "recommender" and e["run_id"] == run_["run_id"]
        assert e["benchmark"] == row["benchmark"] and e["conviction"] == row["conviction"]
        assert row["journal_id"] == e["id"] and row["due"] == e["due"]
    symbols = [r["symbol"] for r in run_["ideas"]]
    assert "VEA" in symbols and entries["VEA"]["benchmark"] == "EFA"
    assert "IEFA" not in symbols, "one fund per gap"
    assert ("IEFA", "same gap as VEA, which ranks higher") in run_["dropped"]
    # Short of bonds while holding BND: the idea is to add to BND, not buy AGG.
    assert "AGG" not in symbols
    bnd = next(r for r in run_["ideas"] if r["symbol"] == "BND")
    assert bnd["verdict"] == "add" and bnd["source"] == "gap"
    # And the tech names break the 35% sector cap on a book that is 90% tech.
    assert any(s == "WIN" and "sector cap" in why for s, why in run_["dropped"])
    assert recommend.latest_run()["run_id"] == run_["run_id"]
    s = recommend.sheet(run_)
    assert "Considered and Dropped" in s["markdown"] or not run_["dropped"]
    assert "scored in the journal on" in s["markdown"]
    assert s["message"].startswith("💡 Deep ideas")


def test_generic_mode_is_said_out_loud(world):
    run_ = asyncio.run(recommend.run("light", fake=True))
    assert run_["generic"] is True
    assert "GENERIC MODE" in recommend.sheet(run_)["subtitle"]


def test_a_shadow_run_is_recorded_and_scored_but_not_sent(world, monkeypatch):
    monkeypatch.setenv("FRA_IDEAS_SHADOW", "1")
    monkeypatch.setattr(recommend, "render_run", lambda run_: [])
    from financial_research_assistant import jobs

    res = asyncio.run(jobs.handler_for("ideas")({"id": "s1"}, True))
    assert res.ok and res.notify is False
    assert all(e.get("shadow") for e in journal.load_entries() if e.get("source") == "recommender")


def test_without_a_model_the_run_says_so_and_records_nothing(world, monkeypatch):
    from financial_research_assistant import guardrails

    guardrails.pause()
    run_ = asyncio.run(recommend.run("light", fake=False))
    assert run_["ideas"] == [] and "paused" in run_["blocked"]
    assert [e for e in journal.load_entries() if e.get("source") == "recommender"] == []
    assert "Model judgement was skipped" in recommend.sheet(run_)["markdown"]


# --- nothing here can trade --------------------------------------------------------------------------


#: The mutating tools of the IBKR connector this user has, as that connector names them.
_IBKR_MUTATORS = [
    "create_order_instruction", "delete_order_instruction", "create_alert", "update_alert",
    "delete_alert", "set_alert_status", "create_watchlist", "edit_watchlist",
    "delete_watchlist", "provide_customer_feedback", "place_order", "confirm_order",
    "cancel_order", "modify_order",
]


def test_no_unattended_path_can_reach_a_broker_write():
    """The recommender opens no broker session at all, and every tool an unattended
    turn (a scheduled task, a phone message, an event analysis) can bind is local
    and read-only toward the broker: the broker's own tools pass the read-only
    filter before any turn sees them, and none of its writes get through it."""
    import inspect

    from financial_research_assistant import recommend as rec, watchers

    for module in (rec, watchers):
        src = inspect.getsource(module)
        assert "broker_tools_session" not in src and "ibkr_tools_session" not in src
    with catalog.unattended():
        names = {catalog.tool_name(t) for t in catalog.active_tools(
            frozenset({"statements", "documents", "alerts", "tasks", "journal"}))}
    assert not names & set(_IBKR_MUTATORS)

    class _T:
        def __init__(self, name):
            self.name = name

    kept = {t.name for t in tools.filter_readonly([_T(n) for n in _IBKR_MUTATORS])}
    assert kept == set(), f"the read-only filter let through: {kept}"
    reads = [_T("get_account_positions"), _T("get_price_snapshot"), _T("search_contracts")]
    assert {t.name for t in tools.filter_readonly(reads)} == {t.name for t in reads}


def test_the_ideas_jobs_and_command_are_registered(world):
    from financial_research_assistant import jobs, scheduler

    rows = {j: t for j, t, _n in jobs.ensure_default_jobs()}
    assert rows["ideas"]["repeat"] == "weekly"
    assert rows["ideas-deep"]["repeat"] == "monthly-first-weekday"
    assert asyncio.run(scheduler.run_command("/ideas")).startswith("No ideas yet")
    assert "Investor profile (generic" in asyncio.run(scheduler.run_command("/profile"))
