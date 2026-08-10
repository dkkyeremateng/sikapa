"""Fundamentals, analyst, earnings, and dividend-projection tools (Yahoo, keyless).

These complement the real-time IBKR quotes and the local price-history charts with
the reference data an analyst reaches for: valuation/profile snapshots, analyst
ratings and price targets, the earnings calendar, and a forward dividend-income
projection over the imported portfolio.

Data source: the ``yfinance`` library (free, no API key) — the same Yahoo backend
the price charts use, but via yfinance's ``Ticker`` accessors for the
``quoteSummary`` modules (which need Yahoo's crumb/cookie dance yfinance handles).
Every network access goes through a small ``_fetch_*`` helper that returns plain
Python structures and swallows failures (returning ``{}``/``[]``), so a Yahoo
outage degrades to a friendly "no data" message instead of aborting the turn — and
the helpers are trivially monkeypatched in tests, keeping them offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

# Same-process cache of ``symbol -> (info, fetched_at)``: yfinance's ``.info`` is
# a slow quoteSummary round-trip, so a tool that touches several holdings
# (dividend projection) doesn't refetch the same ticker.
#
# Entries expire on the same clock as the price-series cache, because ``.info``
# carries the spot price that `explain_option`'s breakeven, `dcf_valuation`'s
# upside and `compare_stocks` are all quoted against. Held for the life of the
# process, a session left open across a trading day would price an option off
# this morning's quote while the chart beside it showed this afternoon's close —
# two tools disagreeing about "now" within one conversation.
_INFO_CACHE: dict[str, tuple[dict[str, Any], float]] = {}


def _ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol.strip().upper())


def _fetch_info(symbol: str) -> dict[str, Any]:
    """yfinance ``.info`` (valuation/profile/analyst summary) as a dict, cached
    for ``FINANCIAL_RESEARCH_PRICE_TTL`` seconds; ``{}`` on any failure or
    unknown ticker."""
    from .tools import _price_ttl

    sym = symbol.strip().upper()
    ttl = _price_ttl()
    cached = _INFO_CACHE.get(sym)
    if cached is not None and ttl > 0 and (time.time() - cached[1]) < ttl:
        return cached[0]
    try:
        info = _ticker(sym).info or {}
    except Exception:  # noqa: BLE001 — network/parse failure degrades to no-data
        info = {}
    if info:
        _INFO_CACHE[sym] = (info, time.time())
    return info


def _fetch_calendar(symbol: str) -> dict[str, Any]:
    """yfinance ``.calendar`` — next earnings date + dividend/ex-dividend dates;
    ``{}`` on failure."""
    try:
        return _ticker(symbol).calendar or {}
    except Exception:  # noqa: BLE001
        return {}


#: Scheduled-but-unreported quarters yfinance lists ahead of the reported ones —
#: it publishes roughly a year of forward dates. A caller that wants N *reported*
#: quarters has to reach past them, or the upcoming rows eat the whole limit.
_UPCOMING_QUARTERS = 4


def _fetch_earnings_history(
    symbol: str, limit: int = 6, reported_only: bool = False
) -> list[dict[str, Any]]:
    """Recent + upcoming earnings as a list of ``{date, estimate, reported,
    surprise}`` dicts (newest first), converting yfinance's DataFrame to plain
    data and NaN to None; ``[]`` on failure.

    ``reported_only`` drops the scheduled quarters that have no figure yet, and
    does so BEFORE ``limit`` is applied. The order matters: yfinance returns
    upcoming dates first, so ``limit=4`` on a company with four scheduled dates
    yields four rows with nothing reported in them — and a caller counting a beat
    streak reads that as "no streak" rather than as "wrong rows"."""
    want = limit + _UPCOMING_QUARTERS if reported_only else limit
    try:
        df = _ticker(symbol).get_earnings_dates(limit=want)
    except Exception:  # noqa: BLE001
        return []
    if df is None or getattr(df, "empty", True):
        return []
    # yfinance's own ``limit`` is unreliable (can return many more), so cap the
    # rows here to keep the tool result bounded. The frame is newest-date-first.
    try:
        df = df.head(want)
    except Exception:  # noqa: BLE001
        pass
    out: list[dict[str, Any]] = []
    # yfinance is untyped, so `df` widens to dict; the `.empty` guard above
    # already returned for anything that isn't a DataFrame.
    for idx, row in df.iterrows():
        out.append({
            "date": _row_date(idx),
            "estimate": _num(row.get("EPS Estimate")),
            "reported": _num(row.get("Reported EPS")),
            "surprise": _num(row.get("Surprise(%)")),
        })
    if reported_only:
        out = [r for r in out if r["reported"] is not None]
    return out[:limit]


def _fetch_price_targets(symbol: str) -> dict[str, Any]:
    """yfinance ``.analyst_price_targets`` (current/low/mean/median/high);
    ``{}`` on failure."""
    try:
        return _ticker(symbol).analyst_price_targets or {}
    except Exception:  # noqa: BLE001
        return {}


def _fetch_rating_changes(symbol: str, limit: int = 6) -> list[dict[str, Any]]:
    """Recent analyst upgrades/downgrades as ``{date, firm, from, to, action}``
    dicts (newest first); ``[]`` on failure."""
    try:
        df = _ticker(symbol).upgrades_downgrades
    except Exception:  # noqa: BLE001
        return []
    # yfinance's stubs say this is always a DataFrame; in practice it returns None
    # for tickers with no coverage, so the guard is load-bearing at runtime.
    if df is None or getattr(df, "empty", True):  # pyright: ignore[reportUnnecessaryComparison]
        return []
    # yfinance is untyped, so `df` widens to Unknown/dict for a checker. The guard
    # above already returned for anything without `.empty`, i.e. anything that
    # isn't a DataFrame — name that once here instead of at each use below.
    frame: Any = df
    try:
        frame = frame.sort_index(ascending=False).head(limit)
    except Exception:  # noqa: BLE001
        pass
    out: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        out.append({
            "date": _row_date(idx)[:10],
            "firm": str(row.get("Firm") or ""),
            "from": str(row.get("FromGrade") or ""),
            "to": str(row.get("ToGrade") or ""),
            "action": str(row.get("Action") or ""),
        })
    return out


def _row_date(idx: Any) -> str:
    """ISO date string from a pandas row index (Timestamp) or any label."""
    d = getattr(idx, "date", None)
    return str(d() if callable(d) else idx)


def _num(v: Any) -> float | None:
    """A float, or None for missing / NaN."""
    import math

    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _money(v: Any) -> str:
    """Compact money formatting: 4.62T / 12.3B / 45.6M / 1,234."""
    n = _num(v)
    if n is None:
        return "n/a"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n:,.0f}"


def _fmt(v: Any, spec: str = ".2f", suffix: str = "") -> str:
    n = _num(v)
    return f"{n:{spec}}{suffix}" if n is not None else "n/a"


def _price(info: dict[str, Any]):
    """Best available current price from an info dict."""
    return _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))


def stock_fundamentals(symbol: str, as_of: str = "") -> str:
    """Fundamental snapshot for a stock ``symbol``: company profile (name, sector,
    industry), valuation (market cap, trailing/forward P/E, EPS), current price and
    52-week range, dividend rate & yield, beta, and a one-line analyst summary.
    Data source: Yahoo Finance (yfinance, keyless) — reference data, not real-time
    IBKR quotes. Use for 'fundamentals / valuation / P/E / market cap / what does
    this company do / is it a dividend stock' questions about a ticker.
    These figures are CURRENT ONLY. For a past date pass ``as_of`` (YYYY-MM-DD) and
    this tool will redirect you to a source that has history, rather than returning
    today's numbers as though they were then's."""
    from .pointintime import snapshot_guard

    refusal = snapshot_guard("stock_fundamentals", as_of)
    if refusal:
        return refusal
    sym = symbol.strip().upper()
    info = _fetch_info(sym)
    if not info or not (info.get("longName") or info.get("shortName")):
        return (
            f"No fundamental data found for {sym!r}. Check the ticker (US symbols, "
            f"or Yahoo suffixes like VOD.L, SAP.DE)."
        )
    name = info.get("longName") or info.get("shortName")
    ccy = info.get("currency") or ""
    price = _price(info)
    rate = _num(info.get("dividendRate"))
    dyield = (rate / price * 100.0) if (rate and price) else None
    lines = [
        f"{sym} · {name}",
        f"  sector: {info.get('sector') or 'n/a'} · industry: {info.get('industry') or 'n/a'}",
        f"  market cap: {_money(info.get('marketCap'))} {ccy}",
        f"  price: {_fmt(price)} {ccy} · 52w range {_fmt(info.get('fiftyTwoWeekLow'))} – "
        f"{_fmt(info.get('fiftyTwoWeekHigh'))}",
        f"  P/E: trailing {_fmt(info.get('trailingPE'))} · forward "
        f"{_fmt(info.get('forwardPE'))} · EPS (ttm) {_fmt(info.get('trailingEps'))}",
        f"  dividend: {_fmt(rate)} {ccy}/yr" + (
            f" · yield {dyield:.2f}%" if dyield is not None else " (none)"
        ),
        f"  beta: {_fmt(info.get('beta'))}",
    ]
    rec = info.get("recommendationKey")
    n_an = _num(info.get("numberOfAnalystOpinions"))
    tgt = _num(info.get("targetMeanPrice"))
    if rec or tgt:
        upside = ((tgt - price) / price * 100.0) if (tgt and price) else None
        parts = []
        if rec:
            parts.append(f"consensus {rec.replace('_', ' ')}")
        if n_an:
            parts.append(f"{int(n_an)} analysts")
        if tgt:
            up = f" ({upside:+.1f}% vs price)" if upside is not None else ""
            parts.append(f"mean target {tgt:.2f} {ccy}{up}")
        lines.append("  analysts: " + " · ".join(parts))
    lines.append("(Yahoo reference data — can be delayed; verify before acting.)")
    return "\n".join(lines)


def analyst_ratings(symbol: str, as_of: str = "") -> str:
    """Analyst ratings for a stock ``symbol``: the buy/hold/sell consensus and
    number of analysts, price targets (low / mean / median / high) with implied
    upside vs the current price, and recent upgrades/downgrades (firm, grade
    change, date). Data source: Yahoo Finance (yfinance, keyless). Use for 'what
    do analysts think / price target / upgrades / downgrades / is it a buy'
    questions. The consensus is CURRENT ONLY — pass ``as_of`` (YYYY-MM-DD) for a
    past date and this tool will say so rather than presenting today's view as
    the view back then."""
    from .pointintime import snapshot_guard

    refusal = snapshot_guard("analyst_ratings", as_of)
    if refusal:
        return refusal
    sym = symbol.strip().upper()
    info = _fetch_info(sym)
    targets = _fetch_price_targets(sym)
    current = _num(targets.get("current")) or _price(info)
    mean = _num(targets.get("mean")) or _num(info.get("targetMeanPrice"))
    if not info and not targets:
        return f"No analyst data found for {sym!r}. Check the ticker."
    ccy = info.get("currency") or ""
    lines = [f"Analyst ratings · {sym}"]
    rec = info.get("recommendationKey")
    n_an = _num(info.get("numberOfAnalystOpinions"))
    recmean = _num(info.get("recommendationMean"))
    if rec or n_an:
        extra = f" (mean score {recmean:.2f}, 1=strong buy…5=strong sell)" if recmean else ""
        who = f" from {int(n_an)} analysts" if n_an else ""
        lines.append(f"  consensus: {(rec or 'n/a').replace('_', ' ')}{who}{extra}")
    if any(_num(targets.get(k)) for k in ("low", "mean", "median", "high")) or mean:
        upside = ((mean - current) / current * 100.0) if (mean and current) else None
        lines.append(
            f"  price targets {ccy}: low {_fmt(targets.get('low'))} · mean "
            f"{_fmt(mean)} · median {_fmt(targets.get('median'))} · high "
            f"{_fmt(targets.get('high'))}"
        )
        if current:
            up = f" ({upside:+.1f}% to mean)" if upside is not None else ""
            lines.append(f"  current price {current:.2f} {ccy}{up}")
    changes = _fetch_rating_changes(sym)
    if changes:
        lines.append("  recent rating changes:")
        for c in changes:
            grade = f"{c['from']} → {c['to']}" if c["from"] or c["to"] else c["action"]
            lines.append(f"    {c['date']}  {c['firm']:<22} {grade}  ({c['action']})")
    if len(lines) == 1:
        return f"No analyst ratings found for {sym!r}."
    lines.append("(Yahoo reference data — verify before acting; not advice.)")
    return "\n".join(lines)


def earnings_calendar(symbol: str) -> str:
    """Earnings calendar for a stock ``symbol``: the next earnings date and
    consensus EPS estimate, the upcoming ex-dividend / dividend pay dates, and a
    short history of recent quarters (estimate vs reported EPS and the surprise
    %). Data source: Yahoo Finance (yfinance, keyless). Use for 'when does X
    report / next earnings / earnings history / ex-dividend date' questions."""
    sym = symbol.strip().upper()
    cal = _fetch_calendar(sym)
    history = _fetch_earnings_history(sym)
    if not cal and not history:
        return f"No earnings data found for {sym!r}. Check the ticker."
    lines = [f"Earnings calendar · {sym}"]
    ed = cal.get("Earnings Date")
    if isinstance(ed, (list, tuple)):
        ed = ed[0] if ed else None
    if ed:
        est = _num(cal.get("Earnings Average"))
        lo, hi = _num(cal.get("Earnings Low")), _num(cal.get("Earnings High"))
        est_txt = ""
        if est is not None:
            rng = f" (range {lo:.2f}–{hi:.2f})" if (lo is not None and hi is not None) else ""
            est_txt = f" · consensus EPS {est:.2f}{rng}"
        lines.append(f"  next earnings: {ed}{est_txt}")
    if cal.get("Ex-Dividend Date") or cal.get("Dividend Date"):
        lines.append(
            f"  ex-dividend: {cal.get('Ex-Dividend Date') or 'n/a'} · "
            f"dividend pay: {cal.get('Dividend Date') or 'n/a'}"
        )
    if history:
        lines.append("  recent quarters (date · est → reported · surprise):")
        for h in history:
            est = _fmt(h["estimate"])
            rep = _fmt(h["reported"]) if h["reported"] is not None else "—"
            sur = f"{h['surprise']:+.1f}%" if h["surprise"] is not None else "—"
            lines.append(f"    {h['date'][:10]}  {est} → {rep}  {sur}")
    return "\n".join(lines)


def dividend_projection(account: str = "", as_of: str = "") -> str:
    """Project forward 12-month dividend income across your imported portfolio.
    For each open position it multiplies the share count by the stock's forward
    annual dividend rate (Yahoo, keyless), reports per-holding income, yield-on-
    cost and current yield, and totals them in the base currency (non-base
    holdings converted at market FX). Use for 'dividend income / projected
    dividends / passive income / yield on cost / which holdings pay dividends'
    questions. ``account`` scopes to one account (default: the newest import's).
    Needs positions imported via `import_ibkr_statement` first.

    This is a FORWARD projection from today's holdings and today's declared rates.
    For dividends actually received in a past period use `income_summary(year=…)`;
    passing ``as_of`` here says so rather than dating a projection that has no past
    version."""
    from .pointintime import snapshot_guard

    refusal = snapshot_guard("dividend_projection", as_of)
    if refusal:
        return refusal
    from . import statements
    from .tools import BASE_CURRENCY, _fx_rate

    positions = statements.query_positions(account=account or None)
    if not positions:
        return (
            "No positions found. Import a statement with open positions using "
            "`import_ibkr_statement`, then try again."
        )
    rows, total_base, payers = [], 0.0, 0
    fx_missing: set[str] = set()
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        qty = _num(p.get("quantity")) or 0.0
        if not sym or qty <= 0:
            continue
        info = _fetch_info(sym)
        rate = _num(info.get("dividendRate"))
        if not rate:
            continue  # non-payer (or no data) — skip in the income roll-up
        payers += 1
        price = _price(info)
        pos_ccy = (info.get("currency") or p.get("currency") or BASE_CURRENCY).upper()
        annual = qty * rate
        basis = _num(p.get("cost_basis")) or 0.0
        yoc = (annual / basis * 100.0) if basis else None
        cyield = (rate / price * 100.0) if price else None
        rate_base = _fx_rate(pos_ccy)
        if rate_base is None:
            fx_missing.add(pos_ccy)
            rate_base = 1.0
        annual_base = annual * rate_base
        total_base += annual_base
        rows.append((annual_base, sym, annual, pos_ccy, yoc, cyield))
    if not rows:
        return (
            "None of your imported positions pay a dividend (or Yahoo has no "
            "dividend data for them). Nothing to project."
        )
    rows.sort(reverse=True)
    lines = [
        f"FORWARD DIVIDEND PROJECTION ({payers} paying holding(s), "
        f"next 12 months):"
    ]
    for annual_base, sym, annual, ccy, yoc, cyield in rows:
        detail = []
        if cyield is not None:
            detail.append(f"yield {cyield:.2f}%")
        if yoc is not None:
            detail.append(f"yield-on-cost {yoc:.2f}%")
        native = f"{annual:,.2f} {ccy}"
        base = f" ≈ {annual_base:,.2f} {BASE_CURRENCY}" if ccy != BASE_CURRENCY else ""
        extra = ("  · " + " · ".join(detail)) if detail else ""
        lines.append(f"  {sym:<6} {native}{base}{extra}")
    note = f" (missing FX for {', '.join(sorted(fx_missing))})" if fx_missing else ""
    lines.append(f"TOTAL ≈ {total_base:,.2f} {BASE_CURRENCY}/yr{note}")
    lines.append(
        "(Forward estimate from the current declared rate × shares; actual "
        "dividends can be cut, raised, or suspended. Yahoo data, verify before "
        "relying on it.)"
    )
    return "\n".join(lines)


def _parse_symbols(symbols: str, limit: int = 4) -> list[str]:
    """Split a user-supplied ticker string (comma/space/pipe-separated) into an
    upper-cased, de-duplicated, order-preserving list capped at ``limit``."""
    raw = symbols.replace(",", " ").replace("|", " ").split()
    seen: list[str] = []
    for s in raw:
        t = s.strip().upper()
        if t and t not in seen:
            seen.append(t)
    return seen[:limit]


# The metric rows of the comparison table, each: (label, extractor(info)->value,
# format spec). The extractor pulls from a ticker's yfinance ``.info`` dict; a
# missing field renders "n/a" via ``_fmt``/``_money``. Kept declarative so the
# row set is easy to extend without touching the render loop.
def _pe_fwd(info: dict[str, Any]):
    return _num(info.get("forwardPE"))


def _peg(info: dict[str, Any]):
    # yfinance moved PEG under different keys across versions; try both.
    return _num(info.get("trailingPegRatio")) or _num(info.get("pegRatio"))


def _rev_growth(info: dict[str, Any]):
    g = _num(info.get("revenueGrowth"))
    return g * 100.0 if g is not None else None


def _margin(info: dict[str, Any]):
    m = _num(info.get("profitMargins"))
    return m * 100.0 if m is not None else None


def _div_yield(info: dict[str, Any]):
    rate, price = _num(info.get("dividendRate")), _price(info)
    return (rate / price * 100.0) if (rate and price) else None


_COMPARE_ROWS = [
    ("Price", lambda i: _price(i), ".2f"),
    ("Market cap", lambda i: i.get("marketCap"), "money"),
    ("P/E (ttm)", lambda i: _num(i.get("trailingPE")), ".1f"),
    ("P/E (fwd)", _pe_fwd, ".1f"),
    ("PEG", _peg, ".2f"),
    ("P/S", lambda i: _num(i.get("priceToSalesTrailing12Months")), ".2f"),
    ("Rev growth %", _rev_growth, ".1f"),
    ("Profit margin %", _margin, ".1f"),
    ("EPS (ttm)", lambda i: _num(i.get("trailingEps")), ".2f"),
    ("Div yield %", _div_yield, ".2f"),
    ("Beta", lambda i: _num(i.get("beta")), ".2f"),
]


def _compare_cell(info: dict[str, Any], extract: Callable[[dict[str, Any]], Any],
                  spec: str) -> str:
    """Render one table cell: the extracted value formatted per ``spec`` (``money``
    uses the compact T/B/M scale), or 'n/a' when missing."""
    val = extract(info)
    if spec == "money":
        return _money(val)
    return _fmt(val, spec)


def compare_stocks(symbols: str, as_of: str = "") -> str:
    """Compare 2–4 stocks side by side in one normalized metric table: price,
    market cap, trailing/forward P/E, PEG, P/S, revenue growth, profit margin,
    EPS, dividend yield, beta, plus the analyst consensus and mean price target
    with implied upside. Pass the tickers in one string (e.g. ``"AAPL, MSFT,
    NVDA"`` — comma/space separated). Data source: Yahoo Finance (yfinance,
    keyless), reference data (not real-time IBKR quotes). Use for 'compare X vs
    Y', 'which is cheaper/growing faster', 'X or Y' questions across a few names.
    CURRENT ONLY — pass ``as_of`` (YYYY-MM-DD) for a past date and this tool will
    point you at the as-reported source instead of comparing today's multiples."""
    from .pointintime import snapshot_guard

    refusal = snapshot_guard("compare_stocks", as_of)
    if refusal:
        return refusal
    syms = _parse_symbols(symbols)
    if len(syms) < 2:
        return (
            "Give 2–4 tickers to compare, e.g. `compare_stocks(\"AAPL, MSFT, NVDA\")`."
        )
    infos = {s: _fetch_info(s) for s in syms}
    found = [s for s in syms if infos[s] and (infos[s].get("longName") or infos[s].get("shortName"))]
    missing = [s for s in syms if s not in found]
    if len(found) < 2:
        got = f" (only found {', '.join(found)})" if found else ""
        return (
            f"Couldn't find fundamentals for enough of {', '.join(syms)}{got}. "
            f"Check the tickers (US symbols, or Yahoo suffixes like VOD.L)."
        )
    col = max(12, max(len(s) for s in found) + 2)
    label_w = max(len(lbl) for lbl, _, _ in _COMPARE_ROWS) + 2
    header = " " * label_w + "".join(f"{s:>{col}}" for s in found)
    lines = [f"COMPARE · {' vs '.join(found)}", header]
    for label, extract, spec in _COMPARE_ROWS:
        row = f"{label:<{label_w}}" + "".join(
            f"{_compare_cell(infos[s], extract, spec):>{col}}" for s in found
        )
        lines.append(row)
    # Analyst consensus + mean target with implied upside — a separate block since
    # the "upside" needs both the target and the current price.
    lines.append("")
    lines.append("Analyst view:")
    for s in found:
        info = infos[s]
        price = _price(info)
        rec = (info.get("recommendationKey") or "n/a").replace("_", " ")
        tgt = _num(info.get("targetMeanPrice"))
        up = f", {(tgt - price) / price * 100.0:+.1f}% upside" if (tgt and price) else ""
        tgt_txt = f"mean target {tgt:.2f}{up}" if tgt else "no target"
        n_an = _num(info.get("numberOfAnalystOpinions"))
        who = f" ({int(n_an)} analysts)" if n_an else ""
        lines.append(f"  {s:<6} {rec}{who} · {tgt_txt}")
    if missing:
        lines.append("")
        lines.append(f"(No data for: {', '.join(missing)}.)")
    lines.append("(Yahoo reference data — can be delayed; verify before acting. Not advice.)")
    return "\n".join(lines)


def _fetch_fund_data(symbol: str) -> dict[str, Any]:
    """ETF/fund sector weights + top holdings via yfinance ``funds_data``, as
    ``{sectors: {name: weight}, holdings: [(symbol, name, weight), …]}``; ``{}`` for
    a non-fund or on failure."""
    try:
        fd = _ticker(symbol).funds_data
        sectors = dict(fd.sector_weightings or {})
        holdings = []
        th = fd.top_holdings
        # Typed as a DataFrame, returns None for funds without a holdings table.
        if th is not None and not getattr(th, "empty", True):  # pyright: ignore[reportUnnecessaryComparison]
            for idx, row in th.iterrows():
                holdings.append((
                    str(idx), str(row.get("Name") or ""), _num(row.get("Holding Percent")),
                ))
        return {"sectors": sectors, "holdings": holdings}
    except Exception:  # noqa: BLE001 — non-fund tickers raise; degrade to no-data
        return {}


def etf_exposure(symbol: str, as_of: str = "") -> str:
    """Look through an ETF/fund ``symbol`` to its underlying exposure: sector
    weightings and top holdings (with weights). Use for 'what's inside this ETF /
    VOO sector breakdown / top holdings / is this fund tech-heavy / do these two
    ETFs overlap' questions. Data source: Yahoo Finance (yfinance, keyless). Only
    works for funds/ETFs — a single stock returns a 'not a fund' note.
    The basket is published CURRENT ONLY: pass ``as_of`` (YYYY-MM-DD) for a past
    date and this tool says the historical composition isn't available, rather
    than implying today's holdings were the ones held then."""
    from .pointintime import snapshot_guard

    refusal = snapshot_guard("etf_exposure", as_of)
    if refusal:
        return refusal
    sym = symbol.strip().upper()
    data = _fetch_fund_data(sym)
    if not data or (not data.get("sectors") and not data.get("holdings")):
        return (
            f"No fund look-through data for {sym!r} — this works for ETFs/funds "
            f"(e.g. VOO, QQQ), not individual stocks. Check the ticker."
        )
    lines = [f"ETF look-through · {sym}"]
    sectors = data.get("sectors") or {}
    if sectors:
        top = sorted(sectors.items(), key=lambda kv: kv[1] or 0, reverse=True)
        lines.append("  sector weights:")
        for name, wt in top:
            lines.append(f"    {name.replace('_', ' '):<22} {float(wt or 0) * 100:5.1f}%")
    holdings = data.get("holdings") or []
    if holdings:
        lines.append("  top holdings:")
        for hsym, hname, wt in holdings:
            pct = f"{(wt or 0) * 100:5.2f}%" if wt is not None else "  n/a"
            lines.append(f"    {hsym:<6} {pct}  {hname}")
    lines.append("(Yahoo fund data — weights are as last reported.)")
    return "\n".join(lines)


# The yfinance-backed reference tools, appended to tools.TOOLS.
FUNDAMENTALS_TOOLS = [
    stock_fundamentals,
    analyst_ratings,
    earnings_calendar,
    dividend_projection,
    etf_exposure,
    compare_stocks,
]
