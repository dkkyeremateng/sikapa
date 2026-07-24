"""Portfolio analytics that combine the imported lot data with live prices:
tax-loss-harvesting candidates and a holdings correlation matrix.

Both read from the local statement store (no brokerage calls) and fetch current /
historical prices through the same keyless Yahoo helper the charts use, so they're
fully exercised offline in tests by monkeypatching that fetch.
"""

from __future__ import annotations

from datetime import date


def _current_price(sym: str, fallback: float | None = None) -> float | None:
    """Latest close for ``sym`` from Yahoo, falling back to a supplied value (e.g.
    the statement snapshot price) when live data isn't available."""
    from .tools import _fetch_daily

    series = _fetch_daily(sym, 5)
    if series:
        return series[-1][1]
    return fallback


def tax_loss_harvest(
    account: str = "",
    min_loss: float = 0.0,
    short_term_rate: float = 0.35,
    long_term_rate: float = 0.15,
    today: date | None = None,
) -> str:
    """Identify tax-loss-harvesting candidates among your open lots: shares now
    worth less than their cost basis, whose sale would realize a deductible loss.

    For each still-held FIFO lot it compares current price to cost basis, groups
    the losers per symbol into short-term (held < 1 year) and long-term, estimates
    the tax benefit at the given rates, and flags **wash-sale risk** when you
    bought the same symbol within the last 30 days (repurchasing within 30 days of
    the sale disallows the loss). ``min_loss`` hides symbols whose total loss is
    smaller. ``account`` scopes it. Read-only analysis, not tax advice — a loss is
    only realized if you actually sell."""
    from . import statements

    lots_by_sym = statements.open_lots(account=account or None)
    if not lots_by_sym:
        return (
            "No open lots found. Import statements with buys (and any sells) via "
            "`import_ibkr_statement`, then try again."
        )
    today = today or date.today()
    # Snapshot prices as a fallback when Yahoo has no live data for a symbol.
    snap = {
        (p.get("symbol") or "").upper(): p.get("close_price")
        for p in statements.query_positions(account=account or None)
    }
    cutoff = today.toordinal() - 30

    results = []
    total_loss = total_benefit = 0.0
    no_price = []
    for sym, lots in lots_by_sym.items():
        price = _current_price(sym, snap.get(sym))
        if not price:
            no_price.append(sym)
            continue
        st_loss = lt_loss = 0.0
        recent_buy = False
        for lot in lots:
            gain = (price - lot["cost_per_share"]) * lot["qty"]
            opened = lot["open_date"]
            try:
                opened_ord = date.fromisoformat(opened[:10]).toordinal()
            except ValueError:
                opened_ord = 0
            if opened_ord >= cutoff:
                recent_buy = True
            if gain < 0:
                if statements._days_between(opened, today.isoformat()) >= 365:
                    lt_loss += gain
                else:
                    st_loss += gain
        loss = st_loss + lt_loss
        if loss >= 0 or abs(loss) < min_loss:
            continue
        benefit = abs(st_loss) * short_term_rate + abs(lt_loss) * long_term_rate
        total_loss += loss
        total_benefit += benefit
        results.append({
            "symbol": sym, "loss": loss, "st_loss": st_loss, "lt_loss": lt_loss,
            "benefit": benefit, "wash": recent_buy, "price": price,
        })

    if not results:
        return (
            "No tax-loss-harvesting candidates: none of your open lots are currently "
            "at a loss beyond the threshold." + (
                f" (no price data for {', '.join(no_price)})" if no_price else ""
            )
        )
    results.sort(key=lambda r: r["loss"])  # biggest loss first (most negative)
    lines = [f"TAX-LOSS HARVESTING CANDIDATES (as of {today.isoformat()}):"]
    for r in results:
        wash = "  ⚠ wash-sale risk (bought within 30d)" if r["wash"] else ""
        split = []
        if r["st_loss"] < 0:
            split.append(f"ST {r['st_loss']:,.2f}")
        if r["lt_loss"] < 0:
            split.append(f"LT {r['lt_loss']:,.2f}")
        lines.append(
            f"  {r['symbol']:<6} loss {r['loss']:,.2f} ({' · '.join(split)}) "
            f"· est. tax benefit {r['benefit']:,.2f}{wash}"
        )
    lines.append(
        f"TOTAL harvestable loss {total_loss:,.2f} · est. tax benefit "
        f"~{total_benefit:,.2f} (ST {short_term_rate:.0%} / LT {long_term_rate:.0%})"
    )
    if no_price:
        lines.append(f"note: no price data for {', '.join(no_price)} — skipped.")
    lines.append(
        "(Estimate only — a loss is realized only if you sell; wash-sale rules also "
        "apply to repurchases within 30 days AFTER a sale. Not tax advice.)"
    )
    return "\n".join(lines)


def _returns(closes: list[float]) -> list[float]:
    return [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes)) if closes[i - 1]
    ]


def _pearson(a: list[float], b: list[float]) -> float | None:
    import statistics as st

    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n], b[:n]
    sa, sb = st.pstdev(a), st.pstdev(b)
    if not sa or not sb:
        return None
    ma, mb = st.fmean(a), st.fmean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
    return cov / (sa * sb)


def correlation_matrix(symbols: str = "", days: int = 180) -> str:
    """Correlation matrix of daily returns across several tickers (or your imported
    holdings when ``symbols`` is empty), over the last ``days``. Values near +1 move
    together, near 0 are unrelated, near −1 move oppositely — a quick read on how
    diversified (vs. concentrated in one factor) the set is. ``symbols`` is
    comma/space-separated (e.g. ``"AAPL, MSFT, SPY"``). Uses keyless Yahoo daily
    closes aligned on common dates."""
    from . import statements
    from .tools import _aligned_closes

    syms = [s.strip().upper() for s in symbols.replace(",", " ").split() if s.strip()]
    if not syms:
        syms = [
            (p.get("symbol") or "").upper()
            for p in statements.query_positions()
            if (p.get("symbol") or "").strip()
        ]
    syms = list(dict.fromkeys(syms))[:10]  # dedupe, cap for a readable grid
    if len(syms) < 2:
        return (
            "Give at least two tickers (e.g. correlation_matrix('AAPL, MSFT, SPY')), "
            "or import a portfolio with 2+ holdings first."
        )
    dates, closes = _aligned_closes(syms, days)
    present = [s for s in syms if s in closes]
    if len(dates) < 3 or len(present) < 2:
        return (
            f"Not enough overlapping price history for {', '.join(syms)}. Check the "
            f"tickers or widen the window."
        )
    rets = {s: _returns(closes[s]) for s in present}
    # Build the matrix.
    w = max(6, max(len(s) for s in present))
    header = " " * (w + 1) + " ".join(f"{s:>6}" for s in present)
    lines = [
        f"Return correlation · {dates[0]} → {dates[-1]} ({len(dates)} sessions):",
        header,
    ]
    for s in present:
        cells = []
        for t in present:
            c = 1.0 if s == t else _pearson(rets[s], rets[t])
            cells.append(f"{c:>6.2f}" if c is not None else "   n/a")
        lines.append(f"{s:<{w}} " + " ".join(cells))
    missing = [s for s in syms if s not in present]
    if missing:
        lines.append(f"note: no data for {', '.join(missing)} — omitted.")
    return "\n".join(lines)


def _max_drawdown(returns: list[float]) -> tuple[float, float]:
    """Reconstruct a growth-of-1 index from a return series and return
    ``(cumulative_return, max_drawdown)`` as fractions (e.g. -0.23 = -23%)."""
    idx, peak, max_dd = 1.0, 1.0, 0.0
    for r in returns:
        idx *= (1.0 + r)
        peak = max(peak, idx)
        if peak:
            max_dd = min(max_dd, (idx - peak) / peak)
    return idx - 1.0, max_dd


def _beta(port: list[float], bench: list[float]) -> float | None:
    """Beta of a portfolio return series against a benchmark's, over their common
    length; None when there isn't enough overlap or the benchmark has no variance."""
    import statistics as st

    n = min(len(port), len(bench))
    if n < 20:
        return None
    p, b = port[:n], bench[:n]
    var_b = st.pvariance(b)
    if not var_b:
        return None
    cov = st.fmean([p[i] * b[i] for i in range(n)]) - st.fmean(p) * st.fmean(b)
    return cov / var_b


def portfolio_risk(account: str = "", days: int = 365, benchmark: str = "SPY") -> str:
    """Risk/return metrics for your WHOLE portfolio as it stands now: annualized
    volatility, max drawdown, Sharpe ratio (risk-free 0), and beta vs a benchmark.

    Builds a synthetic daily portfolio return series by value-weighting each
    holding's daily return (weights = each position's current market value, so it
    answers 'how risky is the basket I hold TODAY'), then computes the metrics from
    it. This is the portfolio-level companion to the per-ticker `risk_metrics`. Use
    for 'how risky is my portfolio / my volatility / drawdown / Sharpe / beta /
    overall risk'. ``days`` is the lookback; ``benchmark`` the beta reference
    (default SPY); ``account`` scopes it.

    Approximation, stated in the output: current holdings are held CONSTANT over the
    window (it ignores past trades/rebalancing), and only holdings with fetchable
    price history are included — the covered share of portfolio value is reported,
    and weights are renormalized over the covered holdings. Positions without price
    data (cash, some non-US tickers) are excluded, not treated as risk-free."""
    import statistics as st

    from . import statements
    from .fundamentals import _num
    from .tools import BASE_CURRENCY, _aligned_closes, _fx_rate

    positions = statements.query_positions(account=account or None)
    if not positions:
        return (
            "No positions found. Import a statement with open positions using "
            "`import_ibkr_statement`, then try again. (For one ticker's risk, use "
            "`risk_metrics`.)"
        )
    # Current base-currency market value per symbol (the constant weights).
    value, fx_missing = {}, set()
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        val = _num(p.get("value")) or 0.0
        ccy = (p.get("currency") or BASE_CURRENCY).upper()
        r = _fx_rate(ccy)
        if r is None:
            fx_missing.add(ccy)
            r = 1.0
        base_val = val * r
        if sym and base_val > 0:
            value[sym] = value.get(sym, 0.0) + base_val
    if not value:
        return "Positions have no positive market value to analyze."

    total_value = sum(value.values())
    syms = list(value)
    bench = benchmark.strip().upper()
    dates, closes = _aligned_closes(syms + [bench], days)
    covered = [s for s in syms if s in closes]
    if len(dates) < 20 or not covered:
        return (
            "Not enough overlapping price history to compute portfolio risk. Widen "
            "the window, or check that your holdings have price data (US tickers, or "
            "Yahoo suffixes like VOD.L)."
        )
    # Renormalize the constant weights over the covered holdings, and record how much
    # of the portfolio's value that covers so the answer states its own scope.
    covered_value = sum(value[s] for s in covered)
    w = {s: value[s] / covered_value for s in covered}
    rets = {s: _returns(closes[s]) for s in covered}
    n = min(len(rets[s]) for s in covered)
    port = [sum(w[s] * rets[s][i] for s in covered) for i in range(n)]
    if len(port) < 19:
        return "Not enough overlapping return history to compute portfolio risk."

    sd = st.pstdev(port)
    vol = sd * (252 ** 0.5) * 100.0
    sharpe = (st.fmean(port) / sd * (252 ** 0.5)) if sd else 0.0
    cum_ret, max_dd = _max_drawdown(port)
    beta = _beta(port, _returns(closes[bench])) if bench in closes else None
    beta_txt = f"\n  beta vs {bench}          {beta:.2f}" if beta is not None else ""

    coverage = covered_value / total_value * 100.0 if total_value else 0.0
    excluded = [s for s in syms if s not in covered]
    lines = [
        f"PORTFOLIO RISK · {dates[0]} → {dates[-1]} ({len(port) + 1} sessions, "
        f"{len(covered)} holding(s)):",
        f"  annualized volatility  {vol:.1f}%",
        f"  max drawdown           {max_dd * 100.0:.1f}%",
        f"  cumulative return      {cum_ret * 100.0:+.1f}% (over the window)",
        f"  Sharpe (rf=0)          {sharpe:.2f}{beta_txt}",
    ]
    top = sorted(w.items(), key=lambda kv: kv[1], reverse=True)[:5]
    lines.append(
        "weights (current value): " + ", ".join(f"{s} {wt * 100:.0f}%" for s, wt in top)
        + (" …" if len(covered) > 5 else "")
    )
    lines.append(
        f"note: current holdings held constant; covers {coverage:.0f}% of portfolio "
        f"value (holdings with price history)."
    )
    if excluded:
        lines.append(f"note: no price history for {', '.join(excluded)} — excluded.")
    if fx_missing:
        lines.append(f"note: no FX rate for {', '.join(sorted(fx_missing))} — used raw values.")
    return "\n".join(lines)


def _norm_sector(name: str | None) -> str:
    """Canonical sector key so ETF weightings (``financial_services``) and stock
    ``.info`` sectors (``Financial Services``) merge into one bucket."""
    return (name or "Unknown").replace("_", " ").strip().lower() or "unknown"


def portfolio_lookthrough(account: str = "") -> str:
    """See your portfolio's TRUE exposure by looking through ETFs/funds to their
    underlying holdings.

    Weights each position by its value, then for funds distributes that weight
    across the fund's sector weightings and its top holdings (from Yahoo, keyless),
    while direct stocks contribute their whole weight to their own sector and
    themselves. Aggregates into a portfolio **sector breakdown** and a **true
    single-stock exposure** that reveals hidden concentration — e.g. a name you hold
    directly *and* inside several ETFs. Use for 'real/true exposure / am I
    over-concentrated / how much tech / hidden overlap / look-through' questions.
    ``account`` scopes it. (Sector exposure is complete; single-stock look-through
    covers each fund's top holdings only — the rest of the basket isn't itemized.)"""
    from collections import defaultdict

    from . import statements
    from .fundamentals import _fetch_fund_data, _fetch_info, _num
    from .tools import BASE_CURRENCY, _fx_rate

    positions = statements.query_positions(account=account or None)
    if not positions:
        return (
            "No positions found. Import a statement with open positions using "
            "`import_ibkr_statement`, then try again."
        )
    # Base-currency value per holding (weights the look-through).
    holdings, total, fx_missing = [], 0.0, set()
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        val = _num(p.get("value")) or 0.0
        ccy = (p.get("currency") or BASE_CURRENCY).upper()
        r = _fx_rate(ccy)
        if r is None:
            fx_missing.add(ccy)
            r = 1.0
        base_val = val * r
        if sym and base_val > 0:
            holdings.append((sym, base_val))
            total += base_val
    if total <= 0:
        return "Positions have no positive market value to look through."

    sector_exp: dict[str, float] = defaultdict(float)
    stock_exp: dict[str, float] = defaultdict(float)
    stock_src: dict[str, set] = defaultdict(set)
    covered = 0.0  # base $ attributable to named single stocks
    n_funds = 0
    for sym, base_val in holdings:
        fd = _fetch_fund_data(sym)
        if fd and (fd.get("sectors") or fd.get("holdings")):
            n_funds += 1
            sectors = fd.get("sectors") or {}
            ssum = sum(v or 0 for v in sectors.values()) or 1.0
            for name, wt in sectors.items():
                sector_exp[_norm_sector(name)] += base_val * (wt or 0) / ssum
            for hsym, _hname, wt in (fd.get("holdings") or []):
                if wt:
                    stock_exp[hsym] += base_val * wt
                    stock_src[hsym].add(sym)
                    covered += base_val * wt
        else:  # direct stock: whole weight to its own sector and itself
            info = _fetch_info(sym)
            sector_exp[_norm_sector(info.get("sector"))] += base_val
            stock_exp[sym] += base_val
            stock_src[sym].add("direct")
            covered += base_val

    lines = [
        f"PORTFOLIO LOOK-THROUGH (total {total:,.0f} {BASE_CURRENCY}, "
        f"{len(holdings)} holding(s) incl. {n_funds} fund(s)):"
    ]
    lines.append("true sector exposure (funds expanded to their sector weights):")
    for name, amt in sorted(sector_exp.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {name.title():<22} {amt / total * 100:5.1f}%")
    lines.append("true single-stock exposure (direct holdings + top ETF holdings):")
    top_stocks = sorted(stock_exp.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for sym, amt in top_stocks:
        src = sorted(stock_src[sym])
        via = ", ".join("held directly" if s == "direct" else f"via {s}" for s in src)
        lines.append(f"  {sym:<6} {amt / total * 100:5.1f}%  ({via})")
    lines.append(
        f"note: single-stock look-through covers {covered / total * 100:.0f}% of "
        f"assets (each fund's top holdings only; the rest of the basket isn't "
        f"itemized). Sector exposure is complete."
    )
    if fx_missing:
        lines.append(f"note: no FX rate for {', '.join(sorted(fx_missing))} — used raw values.")
    return "\n".join(lines)


ANALYTICS_TOOLS = [tax_loss_harvest, correlation_matrix, portfolio_lookthrough, portfolio_risk]
