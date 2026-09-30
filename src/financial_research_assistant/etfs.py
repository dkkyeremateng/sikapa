"""The single-fund brief — ``stock_brief``'s counterpart for an ETF.

Every figure is computed here with its window beside it: cost (expense ratio),
size, trailing total returns against the fund's benchmark over the SAME closes,
how much of it the book already owns, and how closely it moves with the book.
The last two are what a fund idea turns on — a cheap, good fund that is 60% the
names you already hold adds cost, not diversification.
"""

from __future__ import annotations

from typing import Any
import statistics
from datetime import date


def _num(v: Any) -> float | None:
    from .fundamentals import _num as num  # pyright: ignore[reportPrivateUsage]

    return num(v)


def expense_ratio_pct(info: dict[str, Any]) -> float | None:
    """The expense ratio in percent. Yahoo reports ``netExpenseRatio`` already in
    percent (0.03 = 0.03%) and the older ``annualReportExpenseRatio`` as a fraction
    (0.0003); reading one as the other is off by 100x."""
    net = _num(info.get("netExpenseRatio"))
    if net is not None:
        return net
    old = _num(info.get("annualReportExpenseRatio"))
    return old * 100 if old is not None else None


def _annualized(series: list[tuple[str, float]], years: int) -> dict[str, Any] | None:
    """Annualized return over the last ``years`` years of ``series``, with the
    window it covers; None when the series doesn't reach back that far."""
    if not series:
        return None
    end_day = date.fromisoformat(series[-1][0])
    start_day = end_day.replace(year=end_day.year - years)
    earlier = [(d, c) for d, c in series if d <= start_day.isoformat()]
    if not earlier or (start_day - date.fromisoformat(earlier[-1][0])).days > 7:
        return None
    (d0, p0), (d1, p1) = earlier[-1], series[-1]
    if not p0:
        return None
    total = p1 / p0 - 1
    return {"from": d0, "to": d1, "pct": ((1 + total) ** (1 / years) - 1) * 100}


def _overlap(symbol: str, book: dict[str, Any]) -> dict[str, Any]:
    """How much of the fund the book already owns: its top-holding weights in names
    held directly, and its overlap with each ETF held (sum of the smaller weight
    across the two funds' top holdings)."""
    from .fundamentals import _fetch_fund_data  # pyright: ignore[reportPrivateUsage]

    data = _fetch_fund_data(symbol) or {}
    top = {str(s).upper(): float(w or 0) for s, _n, w in data.get("holdings") or []}
    held = {h["symbol"] for h in book.get("holdings", [])}
    direct = sum(w for s, w in top.items() if s in held) * 100
    via_funds: dict[str, float] = {}
    for h in book.get("holdings", []):
        other = _fetch_fund_data(h["symbol"]) if h["symbol"] != symbol else {}
        theirs = {str(s).upper(): float(w or 0) for s, _n, w in (other or {}).get("holdings") or []}
        if theirs:
            shared = sum(min(w, theirs[s]) for s, w in top.items() if s in theirs) * 100
            if shared:
                via_funds[h["symbol"]] = round(shared, 1)
    return {"top_holdings": len(top), "direct_pct": round(direct, 1), "via_funds": via_funds}


def _correlation(symbol: str, book: dict[str, Any]) -> dict[str, Any] | None:
    """Correlation of daily returns between the fund and the book (value-weighted
    holdings) over the trailing year of common sessions."""
    from .periodic import _fetch_many  # pyright: ignore[reportPrivateUsage]

    holdings = [h for h in book.get("holdings", []) if h["symbol"] != symbol]
    if not holdings:
        return None
    series = _fetch_many([symbol] + [h["symbol"] for h in holdings], 370)
    fund_rows = dict(series.get(symbol) or [])
    common = sorted(set(fund_rows).intersection(*[dict(series.get(h["symbol"]) or []).keys()
                                                   for h in holdings]))
    if len(common) < 60:
        return None
    total = sum(h["value"] for h in holdings)
    weights = {h["symbol"]: h["value"] / total for h in holdings}
    closes = {h["symbol"]: dict(series[h["symbol"]]) for h in holdings}
    f_ret, b_ret = [], []
    for a, b in zip(common, common[1:]):
        f_ret.append(fund_rows[b] / fund_rows[a] - 1)
        b_ret.append(sum(weights[s] * (closes[s][b] / closes[s][a] - 1) for s in weights))
    try:
        corr = statistics.correlation(f_ret, b_ret)
    except (statistics.StatisticsError, AttributeError):  # constant series; py<3.10
        return None
    return {"corr": round(corr, 2), "from": common[0], "to": common[-1], "sessions": len(common)}


def build_etf_brief(symbol: str, book: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every figure a fund idea needs, labelled. Returns ``{symbol, name, facts,
    lines}`` — ``lines`` is the human text, each figure with its window."""
    from . import universes
    from .fundamentals import _fetch_info  # pyright: ignore[reportPrivateUsage]
    from .tools import _fetch_daily  # pyright: ignore[reportPrivateUsage]

    sym = symbol.strip().upper()
    entry = universes.fund(sym) or {}
    bench = str(entry.get("benchmark") or "SPY")
    info = _fetch_info(sym)
    er = expense_ratio_pct(info)
    aum = _num(info.get("totalAssets"))
    yld = _num(info.get("yield"))
    prices = _fetch_daily(sym, 1830, adjusted=True)
    bench_prices = _fetch_daily(bench, 1830, adjusted=True)
    returns = {}
    for years in (1, 3, 5):
        mine = _annualized(prices, years)
        theirs = _annualized([(d, c) for d, c in bench_prices if d <= (mine or {}).get("to", "")]
                             if mine else [], years)
        if mine:
            returns[f"{years}y"] = {**mine, "benchmark_pct": theirs["pct"] if theirs else None}
    overlap = _overlap(sym, book) if book else None
    corr = _correlation(sym, book) if book else None
    lines = [f"{sym} — {entry.get('name') or info.get('longName') or sym} "
             f"({entry.get('sleeve', 'fund')})"]
    lines.append("  expense ratio " + (f"{er:.2f}%" if er is not None else "n/a")
                 + (f" · assets ${aum / 1e9:,.1f}B" if aum else "")
                 + (f" · trailing yield {yld * 100:.2f}% (as reported by Yahoo)" if yld else ""))
    for key, r in returns.items():
        vs = (f" vs {bench} {r['benchmark_pct']:+.2f}%/yr" if r["benchmark_pct"] is not None
              else "")
        lines.append(f"  {key} total return {r['pct']:+.2f}%/yr, {r['from']} → {r['to']}{vs}")
    if overlap:
        lines.append(f"  you already own {overlap['direct_pct']:.1f}% of its top "
                     f"{overlap['top_holdings']} holdings directly"
                     + ("; overlap with your funds: "
                        + ", ".join(f"{k} {v:.0f}%" for k, v in overlap["via_funds"].items())
                        if overlap["via_funds"] else ""))
    if corr:
        lines.append(f"  correlation with your holdings {corr['corr']:+.2f} "
                     f"(daily returns, {corr['from']} → {corr['to']})")
    return {
        "symbol": sym, "name": entry.get("name") or info.get("longName") or sym,
        "asset_class": entry.get("asset_class"), "sleeve": entry.get("sleeve"),
        "benchmark": bench,
        "facts": {"expense_ratio_pct": er, "assets": aum, "yield_pct": yld * 100 if yld else None,
                  "returns": returns, "overlap": overlap, "correlation": corr,
                  "price": prices[-1][1] if prices else None,
                  "price_date": prices[-1][0] if prices else None},
        "lines": lines,
    }
