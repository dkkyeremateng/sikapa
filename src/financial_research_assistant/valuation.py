"""Deterministic discounted-cash-flow (DCF) intrinsic valuation.

A DCF here is pure arithmetic — *the tool* does the math, never the model — over
AS-REPORTED inputs. The free-cash-flow history comes from SEC EDGAR XBRL
(operating cash flow minus capital expenditures, audited 10-K figures, reusing
``edgar._annual_facts``); the net-debt / shares / current-price snapshot comes
from Yahoo (``fundamentals._fetch_info``). Every input and assumption is shown
with its source so the intrinsic value is fully traceable and reproducible, and
nothing is fabricated.

This mirrors the grounded-figures ethos of the SEC tools: a valuation is only as
trustworthy as its stated assumptions, so the tool surfaces the assumptions and a
sensitivity grid rather than a single false-precise "price target". The two-stage
model is a stage-1 explicit projection (FCF grown at a rate) discounted to
present value, plus a Gordon-growth terminal value.

All the arithmetic lives in small pure functions (``_project_fcf``, ``_dcf``,
``_terminal_value``, ``_sensitivity``, ``_cagr``) that need no network, so the
math is exhaustively unit-tested offline; only the two ``_fetch``-style gather
helpers touch the network and they lazy-import the source modules so tests
monkeypatch the same attributes the rest of the suite does.
"""

from __future__ import annotations

import math

# --- Cash-flow-statement XBRL tags -------------------------------------------
# FCF = operating cash flow − capital expenditures. Each is a fallback list of
# us-gaap tags (filers tag the same concept differently across years/companies);
# ``edgar._annual_facts`` merges them and keys by period-end year.
_OPERATING_CASH_FLOW = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
_CAPEX = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsForCapitalImprovements",
]

# --- Assumption defaults (used when the caller passes 0 for that knob) --------
_DEFAULT_YEARS = 5
_MIN_YEARS, _MAX_YEARS = 3, 10
_DEFAULT_DISCOUNT = 0.09          # equity discount rate / WACC proxy
_DEFAULT_TERMINAL_GROWTH = 0.025  # ~long-run nominal GDP
_DEFAULT_STAGE1_GROWTH = 0.06     # fallback when FCF history can't yield a CAGR
_GROWTH_FLOOR, _GROWTH_CAP = -0.05, 0.20  # clamp a *derived* stage-1 growth


# --- Small formatting/parse helpers (self-contained, no network) -------------
def _num(v):
    """A float, or None for missing / NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _money(v) -> str:
    """Compact money: 4.62T / 12.3B / 45.6M / 1,234."""
    n = _num(v)
    if n is None:
        return "n/a"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n:,.0f}"


def _pct(v) -> str:
    n = _num(v)
    return f"{n * 100:.1f}%" if n is not None else "n/a"


def _as_rate(v) -> float:
    """Interpret a rate knob leniently: 12 or 0.12 both mean 12%. A magnitude
    above 1 is read as a percentage (the model often passes whole numbers)."""
    n = _num(v) or 0.0
    return n / 100.0 if abs(n) > 1.0 else n


# --- Pure DCF math (no network; exhaustively unit-tested) --------------------
def _cagr(first: float, last: float, periods: int):
    """Compound annual growth rate first→last over ``periods`` years, or None when
    it isn't defined (non-positive endpoints or no span)."""
    if periods <= 0 or first is None or last is None or first <= 0 or last <= 0:
        return None
    return (last / first) ** (1.0 / periods) - 1.0


def _project_fcf(base: float, growth: float, years: int) -> list[float]:
    """Stage-1 projected FCF for each of the next ``years`` years."""
    out, val = [], base
    for _ in range(years):
        val *= 1.0 + growth
        out.append(val)
    return out


def _present_values(flows: list[float], discount: float) -> list[float]:
    return [cf / (1.0 + discount) ** (i + 1) for i, cf in enumerate(flows)]


def _terminal_value(last_fcf: float, terminal_growth: float, discount: float):
    """Gordon-growth terminal value at the end of the horizon, or None when the
    discount rate doesn't exceed terminal growth (the perpetuity diverges)."""
    if discount <= terminal_growth:
        return None
    return last_fcf * (1.0 + terminal_growth) / (discount - terminal_growth)


def _dcf(base_fcf: float, growth: float, years: int,
         discount: float, terminal_growth: float):
    """Enterprise value from a two-stage DCF, or None if terminal value diverges.
    Returns the full breakdown so the caller can show every intermediate figure."""
    flows = _project_fcf(base_fcf, growth, years)
    pv_flows = _present_values(flows, discount)
    tv = _terminal_value(flows[-1], terminal_growth, discount)
    if tv is None:
        return None
    pv_terminal = tv / (1.0 + discount) ** years
    return {
        "flows": flows,
        "pv_flows": pv_flows,
        "terminal_value": tv,
        "pv_terminal": pv_terminal,
        "enterprise_value": sum(pv_flows) + pv_terminal,
    }


def _per_share(ev: float, net_debt: float, shares):
    """Equity value per share, or None when shares are unknown."""
    if not shares or shares <= 0:
        return None
    return (ev - net_debt) / shares


def _sensitivity(base_fcf, growth, years, net_debt, shares,
                 discount, terminal_growth):
    """Intrinsic-per-share grid over ±discount (rows) × ±terminal-growth (cols)."""
    discounts = [round(discount + d, 4) for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    tgs = [round(terminal_growth + d, 4) for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    grid = []
    for r in discounts:
        row = []
        for g in tgs:
            res = _dcf(base_fcf, growth, years, r, g)
            row.append(_per_share(res["enterprise_value"], net_debt, shares)
                       if res else None)
        grid.append(row)
    return discounts, tgs, grid


# --- Data gather (network; lazy-imported so tests monkeypatch the source) -----
def _fcf_history(cik: str, years: int) -> list[dict]:
    """As-reported FCF per fiscal year, newest-first: operating cash flow − capex.
    Keeps one extra year beyond ``years`` as a CAGR base. Rows lacking operating
    cash flow are skipped (capex missing counts as 0 outflow)."""
    from . import edgar

    facts = edgar._fetch_company_facts(cik)
    if not facts or not facts.get("facts"):
        return []
    ocf = {r["fy"]: r["val"] for r in edgar._annual_facts(facts, _OPERATING_CASH_FLOW, "USD")}
    capex = {r["fy"]: r["val"] for r in edgar._annual_facts(facts, _CAPEX, "USD")}
    rows = []
    for fy in sorted(ocf, reverse=True):
        o = _num(ocf.get(fy))
        if o is None:
            continue
        c = _num(capex.get(fy)) or 0.0
        rows.append({"fy": fy, "ocf": o, "capex": c, "fcf": o - c})
    return rows[: years + 1]


def _snapshot(symbol: str) -> dict:
    """Live net-debt / shares / price snapshot from Yahoo (keyless)."""
    from . import fundamentals

    info = fundamentals._fetch_info(symbol) or {}
    return {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "price": fundamentals._price(info),
        "shares": _num(info.get("sharesOutstanding")),
        "total_debt": _num(info.get("totalDebt")) or 0.0,
        "total_cash": _num(info.get("totalCash")) or 0.0,
        "currency": info.get("currency") or "USD",
    }


# --- Assumption resolution ---------------------------------------------------
def _resolve_growth(growth_rate, history: list[dict]) -> tuple[float, str]:
    """Stage-1 growth: an explicit knob wins; else derive a CAGR from the FCF
    history (clamped to a sane band); else a conservative default."""
    g = _as_rate(growth_rate)
    if g:
        return g, "user-supplied"
    if len(history) >= 2:
        base, oldest = history[0]["fcf"], history[-1]["fcf"]
        cagr = _cagr(oldest, base, len(history) - 1)
        if cagr is not None:
            clamped = max(_GROWTH_FLOOR, min(cagr, _GROWTH_CAP))
            note = f"derived from {len(history)}y FCF CAGR"
            if clamped != cagr:
                note += f" ({_pct(cagr)} clamped)"
            return clamped, note
    return _DEFAULT_STAGE1_GROWTH, "default (history not usable for CAGR)"


# --- Report rendering --------------------------------------------------------
def _assumptions_block(growth, growth_src, discount, terminal_growth, years) -> list[str]:
    return [
        "Assumptions:",
        f"  Stage-1 FCF growth : {_pct(growth):>7}  ({growth_src})",
        f"  Projection horizon : {years:>4} years",
        f"  Discount rate      : {_pct(discount):>7}",
        f"  Terminal growth    : {_pct(terminal_growth):>7}",
    ]


def _projection_block(res: dict, base_fy: int) -> list[str]:
    lines = ["", f"{'Year':<8}{'Projected FCF':>16}{'PV of FCF':>16}"]
    for i, (cf, pv) in enumerate(zip(res["flows"], res["pv_flows"]), start=1):
        lines.append(f"{('+' + str(i)):<8}{_money(cf):>16}{_money(pv):>16}")
    lines.append(f"{'Term.':<8}{_money(res['terminal_value']):>16}{_money(res['pv_terminal']):>16}")
    lines.append(f"(FCF base: FY{base_fy} as-reported operating cash flow − capex.)")
    return lines


def _valuation_block(res, net_debt, snap) -> tuple[list[str], float | None]:
    ev = res["enterprise_value"]
    equity = ev - net_debt
    intrinsic = _per_share(ev, net_debt, snap["shares"])
    ccy = snap["currency"]
    lines = [
        "",
        f"  Enterprise value        : {_money(ev)} {ccy}",
        f"  Less net debt           : {_money(net_debt)} {ccy}",
        f"  Equity value            : {_money(equity)} {ccy}",
        f"  Shares outstanding      : {_money(snap['shares'])}",
    ]
    if intrinsic is None:
        lines.append("  Intrinsic value / share : n/a (shares outstanding unknown)")
        return lines, None
    lines.append(f"  Intrinsic value / share : {intrinsic:,.2f} {ccy}")
    price = snap["price"]
    if price:
        upside = (intrinsic - price) / price
        verdict = "undervalued" if upside > 0 else "overvalued"
        lines.append(f"  Current price           : {price:,.2f} {ccy}")
        lines.append(f"  Upside / (downside)     : {_pct(upside)}  → {verdict} on these assumptions")
    return lines, intrinsic


def _sensitivity_block(base_fcf, growth, years, net_debt, snap, discount, tg) -> list[str]:
    if not snap["shares"] or snap["shares"] <= 0:
        return []
    discounts, tgs, grid = _sensitivity(
        base_fcf, growth, years, net_debt, snap["shares"], discount, tg)
    lines = ["", "Sensitivity — intrinsic value / share (rows: discount, cols: terminal growth):",
             f"{'disc\\tg':<9}" + "".join(f"{_pct(g):>9}" for g in tgs)]
    for r, row in zip(discounts, grid):
        cells = "".join(f"{(f'{v:,.0f}' if v is not None else '—'):>9}" for v in row)
        lines.append(f"{_pct(r):<9}{cells}")
    return lines


def _render(sym, snap, history, growth, growth_src, discount, tg, years, res) -> str:
    net_debt = snap["total_debt"] - snap["total_cash"]
    base_fy = history[0]["fy"]
    lines = [f"DCF intrinsic valuation · {snap['name']} ({sym}) — two-stage FCF model", ""]
    lines += _assumptions_block(growth, growth_src, discount, tg, years)
    lines.append("")
    lines.append("As-reported FCF history (SEC EDGAR XBRL, operating cash flow − capex):")
    for r in history[: years + 1]:
        lines.append(f"  FY{r['fy']}: OCF {_money(r['ocf'])} − capex {_money(r['capex'])} "
                     f"= FCF {_money(r['fcf'])}")
    lines += _projection_block(res, base_fy)
    val_lines, _ = _valuation_block(res, net_debt, snap)
    lines += val_lines
    lines.append(f"  Net debt = total debt {_money(snap['total_debt'])} − cash "
                 f"{_money(snap['total_cash'])} (Yahoo snapshot).")
    lines += _sensitivity_block(history[0]["fcf"], growth, years, net_debt, snap, discount, tg)
    lines += [
        "",
        "This is a MODEL, not a price target: the output is only as good as the "
        "assumptions above. A DCF suits stable free-cash-flow generators; it is a "
        "poor fit for banks/insurers (no meaningful capex) and pre-FCF growth names. "
        "Vary the assumptions (or read the sensitivity grid) before drawing a "
        "conclusion. Inputs: FCF from SEC 10-K XBRL; net debt/shares/price from Yahoo.",
    ]
    return "\n".join(lines)


# --- The tool ----------------------------------------------------------------
def dcf_valuation(symbol: str, growth_rate: float = 0.0, discount_rate: float = 0.0,
                  terminal_growth: float = 0.0, years: int = 0) -> str:
    """Estimate a stock's intrinsic value with a deterministic two-stage
    discounted-cash-flow (DCF) model. Free-cash-flow history is pulled AS-REPORTED
    from the company's SEC 10-K filings (XBRL: operating cash flow − capital
    expenditures); net debt, shares outstanding and current price come from Yahoo.
    The tool does all the arithmetic and shows every input, assumption, the
    projected cash flows and their present values, the terminal value, the implied
    intrinsic value per share, upside/(downside) vs the current price, and a
    sensitivity grid over discount rate × terminal growth.

    Assumption knobs (pass 0 to use a sensible default / derived value; rates
    accept either 0.10 or 10 for 10%): ``growth_rate`` stage-1 annual FCF growth
    (default: derived from the historical FCF CAGR, clamped to −5%..20%);
    ``discount_rate`` (default 9%); ``terminal_growth`` (default 2.5%, must be
    below the discount rate); ``years`` projection horizon (default 5, 3–10).

    Use for 'what's X worth / intrinsic value / fair value / is X over- or
    undervalued / DCF / run a valuation'. It's a model, not a recommendation —
    present it with its assumptions and note it doesn't fit banks or pre-FCF
    companies. For as-reported line items alone use ``sec_financials``; for a
    quick market-multiple snapshot use ``stock_fundamentals``."""
    sym = symbol.strip().upper()
    from . import edgar

    cik = edgar._cik_for(sym)
    if not cik:
        return edgar._no_cik(sym)

    years = min(max(int(years or _DEFAULT_YEARS), _MIN_YEARS), _MAX_YEARS)
    history = _fcf_history(cik, years)
    if not history:
        return (f"No as-reported cash-flow data found for {sym} in SEC XBRL — a DCF "
                f"needs operating cash flow and capex from the 10-K. (Financial "
                f"firms often don't report capex; try stock_fundamentals instead.)")

    base_fcf = history[0]["fcf"]
    if base_fcf <= 0:
        return (f"{sym}'s most recent free cash flow is negative "
                f"(FY{history[0]['fy']}: {_money(base_fcf)}). A growth-DCF can't be "
                f"anchored on negative FCF; use a multiples approach (stock_fundamentals) "
                f"or a longer normalized-FCF view instead.")

    growth, growth_src = _resolve_growth(growth_rate, history)
    discount = _as_rate(discount_rate) or _DEFAULT_DISCOUNT
    tg = _as_rate(terminal_growth) or _DEFAULT_TERMINAL_GROWTH
    if discount <= tg:
        return (f"Discount rate ({_pct(discount)}) must exceed terminal growth "
                f"({_pct(tg)}) for the DCF to converge. Raise the discount rate or "
                f"lower terminal growth and retry.")

    res = _dcf(base_fcf, growth, years, discount, tg)
    if res is None:  # defensive; guarded above
        return "DCF did not converge with these assumptions — adjust and retry."

    snap = _snapshot(sym)
    return _render(sym, snap, history, growth, growth_src, discount, tg, years, res)


VALUATION_TOOLS = [dcf_valuation]
