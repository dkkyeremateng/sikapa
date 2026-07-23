"""Portfolio monitoring digest — a proactive "what changed / what's coming" report
over your imported holdings, with no model call.

It reads open positions from the statement store and, per holding, surfaces the
time-sensitive things a passive investor wants pushed to them:

- **Movers** — holdings whose price moved beyond a threshold over the lookback
  window (from the same keyless Yahoo daily closes the charts use).
- **Upcoming earnings** — next earnings date falling inside a window, with the
  consensus EPS when Yahoo has it.
- **Upcoming ex-dividends** — ex-dividend dates inside the window (so you know a
  payout is about to be captured).
- **Headlines** (opt-in) — a recent news line for each mover.

Because it only calls tools (no LLM), it is deterministic and free to run: the
``--digest`` CLI prints it and exits, so cron/launchd can mail or push it. Every
network access goes through the existing mockable fetch helpers, so the whole
thing is exercised offline in tests.
"""

from __future__ import annotations

from datetime import date, datetime


def _as_date(v):
    """Coerce a calendar value (``datetime.date``/``datetime``/ISO-ish string) to a
    ``date``, or None if it can't be parsed."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def _next_earnings(cal: dict):
    """The earliest earnings date from a yfinance calendar dict, or None."""
    ed = cal.get("Earnings Date")
    if isinstance(ed, (list, tuple)):
        dates = [d for d in (_as_date(x) for x in ed) if d]
        return min(dates) if dates else None
    return _as_date(ed)


def build_digest(
    account: str | None = None,
    lookback_days: int = 5,
    move_threshold: float = 5.0,
    earnings_within: int = 14,
    include_news: bool = False,
    today: date | None = None,
) -> str:
    """Build a plain-text/markdown monitoring digest over the imported holdings.

    ``lookback_days`` is the price-move window; ``move_threshold`` the ± percent
    that makes a holding a "mover"; ``earnings_within`` the day window for upcoming
    earnings/ex-dividends. ``include_news`` adds a headline per mover (slower — one
    web search each). ``today`` is injectable for deterministic tests. Returns a
    "no holdings" message when nothing has been imported."""
    from . import statements
    from .tools import _fetch_daily

    from .fundamentals import _fetch_calendar

    today = today or date.today()
    positions = statements.query_positions(account=account or None)
    if not positions:
        return (
            "No holdings to monitor. Import an IBKR statement with open positions "
            "using `import_ibkr_statement`, then run the digest again."
        )

    symbols = [
        (p.get("symbol") or "").upper() for p in positions if (p.get("symbol") or "").strip()
    ]
    horizon = today.toordinal() + earnings_within
    movers: list[dict] = []
    earnings: list[tuple] = []
    exdivs: list[tuple] = []

    for sym in symbols:
        series = _fetch_daily(sym, max(5, lookback_days))
        if len(series) >= 2:
            (d0, first), (d1, last) = series[0], series[-1]
            if first:
                chg = (last - first) / first * 100.0
                if abs(chg) >= move_threshold:
                    movers.append({
                        "symbol": sym, "pct": chg, "from_date": d0, "to_date": d1,
                        "first": first, "last": last,
                    })
        cal = _fetch_calendar(sym)
        if cal:
            ed = _next_earnings(cal)
            if ed and today <= ed and ed.toordinal() <= horizon:
                earnings.append((ed, sym, cal.get("Earnings Average")))
            xd = _as_date(cal.get("Ex-Dividend Date"))
            if xd and today <= xd and xd.toordinal() <= horizon:
                exdivs.append((xd, sym))

    movers.sort(key=lambda m: abs(m["pct"]), reverse=True)
    earnings.sort()
    exdivs.sort()

    acct = account or (positions[0].get("account") if positions else "") or "default"
    lines = [
        f"# Portfolio digest · {today.isoformat()} · account {acct}",
        f"{len(symbols)} holding(s) · price lookback ~{lookback_days}d · "
        f"event window {earnings_within}d",
        "",
        f"## 📈 Movers (±{move_threshold:g}% over the window)",
    ]
    if movers:
        for m in movers:
            lines.append(
                f"- {m['symbol']:<6} {m['pct']:+.2f}%  "
                f"({m['from_date']} → {m['to_date']}: {m['first']:.2f} → {m['last']:.2f})"
            )
    else:
        lines.append("- none beyond the threshold")

    lines += ["", f"## 📅 Upcoming earnings (next {earnings_within}d)"]
    if earnings:
        for ed, sym, est in earnings:
            est_txt = ""
            try:
                est_txt = f"  · consensus EPS {float(est):.2f}" if est is not None else ""
            except (TypeError, ValueError):
                est_txt = ""
            lines.append(f"- {sym:<6} {ed.isoformat()}{est_txt}")
    else:
        lines.append("- none scheduled in the window")

    lines += ["", f"## 💵 Upcoming ex-dividends (next {earnings_within}d)"]
    if exdivs:
        for xd, sym in exdivs:
            lines.append(f"- {sym:<6} {xd.isoformat()}")
    else:
        lines.append("- none in the window")

    if include_news and movers:
        from .tools import web_search

        lines += ["", "## 📰 Headlines for movers"]
        for m in movers:
            hit = web_search(f"{m['symbol']} stock news", max_results=1)
            first_line = next(
                (ln.strip() for ln in hit.splitlines() if ln.strip().startswith("1.")),
                "",
            )
            lines.append(f"- {m['symbol']:<6} {first_line[3:].strip() or '(no headline)'}")

    lines += [
        "",
        "(Prices/earnings are delayed Yahoo data; verify before acting. "
        "Not investment advice.)",
    ]
    return "\n".join(lines)


def portfolio_digest(
    account: str = "",
    lookback_days: int = 5,
    move_threshold: float = 5.0,
    earnings_within: int = 14,
    include_news: bool = False,
) -> str:
    """Generate a monitoring digest over your imported holdings: price movers over
    the last ``lookback_days`` (beyond ±``move_threshold``%), upcoming earnings and
    ex-dividend dates within ``earnings_within`` days, and — when ``include_news``
    is set — a recent headline per mover. Use for 'what's happening in my
    portfolio / anything I should know / what's coming up / any big moves' style
    check-ins. ``account`` scopes to one account (default: the newest import's).
    Needs positions imported via `import_ibkr_statement` first."""
    return build_digest(
        account=account or None,
        lookback_days=int(lookback_days or 5),
        move_threshold=float(move_threshold or 5.0),
        earnings_within=int(earnings_within or 14),
        include_news=bool(include_news),
    )


MONITOR_TOOLS = [portfolio_digest]
