"""DCF valuation tests — fully offline.

The arithmetic (``_project_fcf`` / ``_dcf`` / ``_terminal_value`` / ``_cagr`` /
``_sensitivity`` / ``_as_rate``) is pure and tested directly. The ``dcf_valuation``
tool's two network reaches — ``edgar._fetch_company_facts`` (+ ``edgar._cik_for``)
for the FCF history and ``fundamentals._fetch_info`` for the net-debt/shares/price
snapshot — are monkeypatched, so the end-to-end tool runs without a network call.
"""

import math

import pytest

from financial_research_assistant import catalog, edgar, fundamentals, tools, valuation


# --- pure math ---------------------------------------------------------------
def test_as_rate_reads_percent_or_decimal():
    assert valuation._as_rate(10) == pytest.approx(0.10)
    assert valuation._as_rate(0.10) == pytest.approx(0.10)
    assert valuation._as_rate(2.5) == pytest.approx(0.025)
    assert valuation._as_rate(0) == 0.0


def test_as_rate_reads_one_as_one_percent_not_a_doubling():
    """The percent/fraction boundary has to include 1 itself. Reading
    `growth_rate=1` as 100% a year is a 20× valuation from a plausible keystroke,
    while reading it as 1% is wrong by four percentage points at worst."""
    assert valuation._as_rate(1) == pytest.approx(0.01)
    assert valuation._as_rate(-1) == pytest.approx(-0.01)
    assert valuation._as_rate(0.99) == pytest.approx(0.99)


def test_as_rate_keeps_an_unset_knob_distinct_from_zero():
    """`0` is an assumption — "no growth" — and `None` is the absence of one. A
    float return type cannot carry that difference, and collapsing it replaced the
    caller's zero with a 6% default under a label that said "user-supplied"."""
    assert valuation._as_rate(None) is None
    assert valuation._as_rate("") is None
    assert valuation._as_rate(0) == 0.0


def test_cagr_and_none_on_nonpositive():
    assert valuation._cagr(100.0, 200.0, 2) == pytest.approx(math.sqrt(2) - 1)
    assert valuation._cagr(-1.0, 200.0, 2) is None
    assert valuation._cagr(100.0, 200.0, 0) is None


def test_project_and_present_value():
    flows = valuation._project_fcf(100.0, 0.10, 3)
    assert flows == pytest.approx([110.0, 121.0, 133.1])
    pv = valuation._present_values([110.0], 0.10)
    assert pv[0] == pytest.approx(100.0)


def test_terminal_value_diverges_when_discount_not_above_growth():
    assert valuation._terminal_value(100.0, 0.02, 0.09) == pytest.approx(
        100.0 * 1.02 / (0.09 - 0.02))
    assert valuation._terminal_value(100.0, 0.09, 0.09) is None
    assert valuation._terminal_value(100.0, 0.10, 0.09) is None


def test_dcf_breakdown_and_per_share():
    res = valuation._dcf(100.0, 0.05, 5, 0.09, 0.025)
    assert res is not None
    # EV = PV(stage-1 flows) + PV(terminal); all positive and terminal dominates
    assert res["enterprise_value"] > sum(res["pv_flows"]) > 0
    assert res["pv_terminal"] > 0
    ps = valuation._per_share(res["enterprise_value"], net_debt=0.0, shares=10.0)
    assert ps == pytest.approx(res["enterprise_value"] / 10.0)
    assert valuation._per_share(res["enterprise_value"], 0.0, 0) is None


def test_sensitivity_grid_shape_and_monotonicity():
    discounts, tgs, grid = valuation._sensitivity(
        100.0, 0.05, 5, net_debt=0.0, shares=10.0,
        discount=0.09, terminal_growth=0.025)
    assert len(discounts) == 5 and len(tgs) == 5
    assert all(len(row) == 5 for row in grid)
    # A higher discount rate lowers intrinsic value (rows are increasing discount).
    mid_col = [row[2] for row in grid]
    assert mid_col[0] > mid_col[-1]


def test_resolve_growth_prefers_user_then_cagr_then_default():
    hist = [{"fy": 2025, "fcf": 121.0}, {"fy": 2024, "fcf": 110.0},
            {"fy": 2023, "fcf": 100.0}]
    g, src = valuation._resolve_growth(15, hist)
    assert g == pytest.approx(0.15) and "user" in src
    g, src = valuation._resolve_growth(None, hist)
    assert g == pytest.approx(0.10, abs=1e-6) and "CAGR" in src  # 100→121 over 2y
    g, src = valuation._resolve_growth(None, [{"fy": 2025, "fcf": -5.0}])
    assert g == valuation._DEFAULT_STAGE1_GROWTH and "default" in src


def test_resolve_growth_honours_an_explicit_zero():
    """"Assume no growth" is a conservative assumption a caller is entitled to
    make. Treating it as "not set" swapped in a 6% default and reported the result
    as the caller's own — a higher valuation than the one that was asked for."""
    hist = [{"fy": 2025, "fcf": 121.0}, {"fy": 2023, "fcf": 100.0}]
    g, src = valuation._resolve_growth(0, hist)
    assert g == 0.0 and "user" in src


def test_resolve_growth_spans_the_fiscal_years_not_the_rows():
    """`_fcf_history` drops years with no reported operating cash flow, so FY2025
    and FY2022 sit in adjacent rows three years apart. Counting rows compounds the
    growth over two years instead of three, overstating it — and the error then
    runs through five projected years and the terminal value."""
    gapped = [{"fy": 2025, "fcf": 133.1}, {"fy": 2024, "fcf": 121.0},
              {"fy": 2022, "fcf": 100.0}]  # 10%/yr over three years, in two rows
    g, src = valuation._resolve_growth(None, gapped)
    assert g == pytest.approx(0.10, abs=1e-6)
    assert "3y" in src
    # Counting rows would compound the same 33% over two years, not three.
    assert g < 1.331 ** (1 / 2) - 1


def test_resolve_growth_clamps_extreme_cagr():
    hist = [{"fy": 2025, "fcf": 1000.0}, {"fy": 2024, "fcf": 10.0}]  # 100x in 1y
    g, src = valuation._resolve_growth(None, hist)
    assert g == valuation._GROWTH_CAP and "clamped" in src


# --- tool (mocked network) ---------------------------------------------------
def _facts(ocf_by_year, capex_by_year):
    """Minimal companyfacts payload: annual 10-K OCF + capex per fiscal year."""
    def units(by_year):
        return [{"form": "10-K", "fy": fy, "fp": "FY",
                 "start": f"{fy}-01-01", "end": f"{fy}-12-31",
                 "filed": f"{fy + 1}-02-01", "val": val}
                for fy, val in by_year.items()]
    return {"entityName": "Testco Inc.", "facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": units(ocf_by_year)}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": units(capex_by_year)}},
    }}}


def _install(monkeypatch, facts, info):
    monkeypatch.setattr(edgar, "_cik_for", lambda s: "0000000001")
    monkeypatch.setattr(edgar, "_fetch_company_facts", lambda cik: facts)
    monkeypatch.setattr(fundamentals, "_fetch_info", lambda s: info)


def test_dcf_valuation_end_to_end(monkeypatch):
    facts = _facts({2025: 120e9, 2024: 110e9, 2023: 100e9, 2022: 90e9},
                   {2025: 20e9, 2024: 18e9, 2023: 16e9, 2022: 15e9})
    info = {"longName": "Testco Inc.", "currentPrice": 50.0,
            "sharesOutstanding": 1e9, "totalDebt": 30e9, "totalCash": 10e9,
            "currency": "USD"}
    _install(monkeypatch, facts, info)

    out = valuation.dcf_valuation("TSTC")
    assert "DCF intrinsic valuation" in out and "Testco" in out
    assert "Intrinsic value / share" in out
    assert "Sensitivity" in out
    assert "Net debt" in out
    # FY2025 FCF = 120 − 20 = 100B is the base, labeled as the FCF base year.
    assert "FY2025" in out
    assert "Upside" in out  # price present → upside line rendered


def test_dcf_rejects_negative_base_fcf(monkeypatch):
    facts = _facts({2025: 10e9}, {2025: 25e9})  # capex > ocf → negative FCF
    _install(monkeypatch, facts, {"sharesOutstanding": 1e9})
    out = valuation.dcf_valuation("TSTC")
    assert "negative" in out.lower()


def test_dcf_rejects_discount_below_terminal_growth(monkeypatch):
    facts = _facts({2025: 120e9, 2024: 110e9}, {2025: 20e9, 2024: 18e9})
    _install(monkeypatch, facts, {"sharesOutstanding": 1e9})
    out = valuation.dcf_valuation("TSTC", discount_rate=3, terminal_growth=5)
    assert "must exceed terminal growth" in out


def test_dcf_honours_a_zero_growth_assumption_end_to_end(monkeypatch):
    """The knob the caller set is the knob the report prints. A flat-FCF DCF is a
    perfectly ordinary conservative case, and it must not come back describing a
    derived or default rate."""
    facts = _facts({2025: 120e9, 2024: 110e9}, {2025: 20e9, 2024: 18e9})
    _install(monkeypatch, facts, {"sharesOutstanding": 1e9, "currency": "USD"})
    out = valuation.dcf_valuation("TSTC", growth_rate=0)
    assert "Stage-1 FCF growth :    0.0%  (user-supplied)" in out


def test_dcf_refuses_an_absurd_supplied_growth(monkeypatch):
    """A supplied rate is never clamped — overriding the caller's own assumption is
    the failure this module exists to avoid — so an impossible one has to be
    refused rather than modelled."""
    facts = _facts({2025: 120e9, 2024: 110e9}, {2025: 20e9, 2024: 18e9})
    _install(monkeypatch, facts, {"sharesOutstanding": 1e9})
    out = valuation.dcf_valuation("TSTC", growth_rate=500)
    assert "isn't a modelling assumption" in out
    assert "Intrinsic value" not in out


def test_dcf_no_cashflow_data_message(monkeypatch):
    monkeypatch.setattr(edgar, "_cik_for", lambda s: "0000000001")
    monkeypatch.setattr(edgar, "_fetch_company_facts", lambda cik: {"facts": {"us-gaap": {}}})
    monkeypatch.setattr(fundamentals, "_fetch_info", lambda s: {})
    out = valuation.dcf_valuation("BANK")
    assert "No as-reported cash-flow data" in out


def test_dcf_unknown_ticker(monkeypatch):
    monkeypatch.setattr(edgar, "_cik_for", lambda s: None)
    out = valuation.dcf_valuation("ZZZZ")
    assert "ZZZZ" in out


def test_dcf_valuation_registered():
    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in catalog.TOOLS}
    assert "dcf_valuation" in names
    assert valuation.dcf_valuation in valuation.VALUATION_TOOLS
