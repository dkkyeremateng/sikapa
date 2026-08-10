"""Options-explainer tests — fully offline.

The payoff math (``_mid`` / ``_moneyness`` / ``_intrinsic`` / ``_long_economics`` /
``_implied_move`` / ``_dte``) is pure and tested directly. The tool's two network
reaches — ``_fetch_expiries`` and ``_fetch_chain`` — plus the ``_spot`` price are
monkeypatched, so ``explain_option`` runs end-to-end without a network call.
"""

from datetime import date

import pytest

from financial_research_assistant import catalog, options, tools


# --- pure math ---------------------------------------------------------------
def test_mid_prefers_bidask_then_last():
    assert options._mid(2.0, 2.4, 5.0) == pytest.approx(2.2)   # midpoint wins
    assert options._mid(0, 0, 3.1) == pytest.approx(3.1)       # fall back to last
    assert options._mid(None, None, 0) is None


def test_moneyness_call_and_put():
    label, pct = options._moneyness(110.0, 100.0, is_call=True)
    assert label == "in the money" and pct == pytest.approx(0.0909, abs=1e-3)
    label, _ = options._moneyness(90.0, 100.0, is_call=True)
    assert label == "out of the money"
    label, _ = options._moneyness(100.0, 100.0, is_call=True)
    assert label == "at the money"
    label, _ = options._moneyness(90.0, 100.0, is_call=False)  # put, spot<strike
    assert label == "in the money"


def test_intrinsic_value():
    assert options._intrinsic(110, 100, True) == 10
    assert options._intrinsic(90, 100, True) == 0
    assert options._intrinsic(90, 100, False) == 10


def test_long_call_economics_breakeven_and_unlimited_upside():
    e = options._long_economics(is_call=True, spot=100.0, strike=100.0, premium=5.0)
    assert e["breakeven"] == 105.0
    assert e["breakeven_move_pct"] == pytest.approx(0.05)
    assert e["cost"] == 500.0 and e["max_loss"] == 500.0
    assert e["max_profit"] is None            # unbounded
    assert e["extrinsic"] == 5.0 and e["intrinsic"] == 0.0


def test_long_put_economics_capped_profit_and_time_value():
    # ITM put: spot 90, strike 100, premium 12 → intrinsic 10, extrinsic 2.
    e = options._long_economics(is_call=False, spot=90.0, strike=100.0, premium=12.0)
    assert e["breakeven"] == 88.0
    assert e["intrinsic"] == 10.0 and e["extrinsic"] == 2.0
    assert e["max_profit"] == pytest.approx((100.0 - 12.0) * 100)  # strike→0


def test_implied_move_and_dte():
    mv = options._implied_move(100.0, 0.365, 365)  # ~1 year, 36.5% IV
    assert mv == pytest.approx(36.5, abs=0.5)
    assert options._implied_move(100.0, 0, 30) is None
    assert options._dte("2026-08-22", today=date(2026, 7, 23)) == 30
    assert options._dte("not-a-date") is None


# --- tool (mocked chain) -----------------------------------------------------
def _chain():
    def mk(strike, last, bid, ask, iv, itm):
        return {"contractSymbol": f"X{strike}", "strike": strike, "lastPrice": last,
                "bid": bid, "ask": ask, "impliedVolatility": iv, "volume": 100,
                "openInterest": 500, "inTheMoney": itm}
    calls = [mk(90, 12.0, 11.8, 12.2, 0.30, True),
             mk(100, 5.0, 4.8, 5.2, 0.32, False),
             mk(110, 1.5, 1.4, 1.6, 0.35, False)]
    puts = [mk(90, 1.2, 1.1, 1.3, 0.31, False),
            mk(100, 4.5, 4.3, 4.7, 0.33, True)]
    return {"calls": calls, "puts": puts}


def _install(monkeypatch, expiries=("2026-08-21",), spot=100.0):
    monkeypatch.setattr(options, "_fetch_expiries", lambda s: list(expiries))
    monkeypatch.setattr(options, "_fetch_chain", lambda s, e: _chain())
    monkeypatch.setattr(options, "_spot", lambda s: spot)


def test_explain_option_single_contract(monkeypatch):
    _install(monkeypatch)
    out = options.explain_option("AAPL", expiry="2026-08-21", strike=100, option_type="call")
    assert "100.00 CALL" in out
    assert "Breakeven at expiry : $105.00" in out   # strike 100 + mid premium 5.0
    assert "Max profit" in out and "unlimited" in out
    assert "Time (extrinsic)" in out
    assert "per contract" in out


def test_explain_option_chain_summary_when_no_strike(monkeypatch):
    _install(monkeypatch)
    out = options.explain_option("AAPL", expiry="2026-08-21", option_type="put")
    assert "puts expiring 2026-08-21" in out
    assert "strike" in out and "IV" in out
    assert "Pass a `strike`" in out


def test_explain_option_nearest_expiry_noted(monkeypatch):
    _install(monkeypatch, expiries=("2026-08-21", "2026-09-18"))
    out = options.explain_option("AAPL", expiry="2026-08-01", strike=100)
    assert "nearest available" in out            # 08-01 → 08-21


def test_nearest_expiry_looks_both_ways():
    """"The first one on/after, else the soonest of all" is not nearest. It walks
    past a Friday two days before the requested date to reach one nineteen days
    after it, and for a date past the last listed expiry it falls back to the FRONT
    month — the furthest available contract from the one asked about."""
    monthlies = ["2026-08-21", "2026-09-18", "2026-10-16"]
    assert options._nearest_expiry(monthlies, "2026-08-23") == ("2026-08-21", False)
    assert options._nearest_expiry(monthlies, "2026-12-01") == ("2026-10-16", False)
    assert options._nearest_expiry(monthlies, "2026-08-21") == ("2026-08-21", True)
    # Equidistant: the later expiry, which has more time on it, is the safer stand-in.
    assert options._nearest_expiry(["2026-08-01", "2026-08-11"], "2026-08-06") == (
        "2026-08-11", False)


def test_explain_option_no_options(monkeypatch):
    monkeypatch.setattr(options, "_fetch_expiries", lambda s: [])
    out = options.explain_option("PRIVATECO")
    assert "No options data" in out


def test_explain_option_registered():
    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in catalog.TOOLS}
    assert "explain_option" in names
    assert options.explain_option in options.OPTIONS_TOOLS
