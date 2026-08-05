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

from collections.abc import Callable
from typing import Any
from datetime import date, datetime

# The two market-data sources the digest reads through. They are parameters
# rather than direct imports so tests can drive the whole scan offline.
FetchDaily = Callable[[str, int], list[tuple[str, float]]]
FetchCalendar = Callable[[str], dict[str, Any]]
# A price mover, an upcoming earnings date, an upcoming ex-dividend date.
Mover = dict[str, Any]
Earnings = tuple[date, str, Any]
ExDiv = tuple[date, str]


def _as_date(v: Any) -> date | None:
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


def _next_earnings(cal: dict[str, Any]):
    """The earliest earnings date from a yfinance calendar dict, or None."""
    ed = cal.get("Earnings Date")
    if isinstance(ed, (list, tuple)):
        dates = [d for d in (_as_date(x) for x in ed) if d]
        return min(dates) if dates else None
    return _as_date(ed)


def _scan_one(
    sym: str, lookback_days: int, move_threshold: float, horizon: int, today: date,
    fetch_daily: FetchDaily, fetch_calendar: FetchCalendar,
) -> tuple[Mover | None, Earnings | None, ExDiv | None]:
    """Scan one holding → ``(mover|None, earnings|None, exdiv|None)``. Independent
    per symbol, so several run concurrently in ``_scan_holdings``."""
    mover = earn = exdiv = None
    series = fetch_daily(sym, max(5, lookback_days))
    if len(series) >= 2 and series[0][1]:
        (d0, first), (d1, last) = series[0], series[-1]
        chg = (last - first) / first * 100.0
        if abs(chg) >= move_threshold:
            mover = {"symbol": sym, "pct": chg, "from_date": d0, "to_date": d1,
                     "first": first, "last": last}
    cal = fetch_calendar(sym)
    if cal:
        ed = _next_earnings(cal)
        if ed and today <= ed and ed.toordinal() <= horizon:
            earn = (ed, sym, cal.get("Earnings Average"))
        xd = _as_date(cal.get("Ex-Dividend Date"))
        if xd and today <= xd and xd.toordinal() <= horizon:
            exdiv = (xd, sym)
    return mover, earn, exdiv


def _scan_holdings(
    symbols: list[str], lookback_days: int, move_threshold: float,
    earnings_within: int, today: date,
    fetch_daily: FetchDaily, fetch_calendar: FetchCalendar,
) -> tuple[list[Mover], list[Earnings], list[ExDiv]]:
    """Per-holding scan → ``(movers, earnings, exdivs)``, each already sorted for
    display (movers by absolute move, the dated lists chronologically). Holdings are
    scanned concurrently (network-bound) with output order preserved."""
    from concurrent.futures import ThreadPoolExecutor

    horizon = today.toordinal() + earnings_within
    movers, earnings, exdivs = [], [], []
    if symbols:
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
            scanned = pool.map(
                lambda s: _scan_one(s, lookback_days, move_threshold, horizon,
                                    today, fetch_daily, fetch_calendar),
                symbols,
            )
            for mover, earn, exdiv in scanned:
                if mover:
                    movers.append(mover)
                if earn:
                    earnings.append(earn)
                if exdiv:
                    exdivs.append(exdiv)
    movers.sort(key=lambda m: abs(m["pct"]), reverse=True)
    earnings.sort()
    exdivs.sort()
    return movers, earnings, exdivs


def _alert_section(symbols: list[str], lookback_days: int, today: date) -> list[str]:
    """The 🔔 triggered-alerts block, or ``[]`` when no rules are set."""
    from . import alerts

    rules = alerts.load_alerts()
    if not rules:
        return []
    fired = alerts.evaluate_alerts(symbols, lookback_days, today)
    out = ["", "## 🔔 Alerts triggered"]
    if fired:
        out += [f"- {a}" for a in fired]
    else:
        out.append(f"- none of your {len(rules)} alert rule(s) triggered")
    return out


def _movers_section(movers: list[Mover], move_threshold: float) -> list[str]:
    out = ["", f"## 📈 Movers (±{move_threshold:g}% over the window)"]
    if not movers:
        out.append("- none beyond the threshold")
        return out
    out += [
        f"- {m['symbol']:<6} {m['pct']:+.2f}%  "
        f"({m['from_date']} → {m['to_date']}: {m['first']:.2f} → {m['last']:.2f})"
        for m in movers
    ]
    return out


def _earnings_section(earnings: list[Earnings], earnings_within: int) -> list[str]:
    out = ["", f"## 📅 Upcoming earnings (next {earnings_within}d)"]
    if not earnings:
        out.append("- none scheduled in the window")
        return out
    for ed, sym, est in earnings:
        try:
            est_txt = f"  · consensus EPS {float(est):.2f}" if est is not None else ""
        except (TypeError, ValueError):
            est_txt = ""
        out.append(f"- {sym:<6} {ed.isoformat()}{est_txt}")
    return out


def _exdivs_section(exdivs: list[ExDiv], earnings_within: int) -> list[str]:
    out = ["", f"## 💵 Upcoming ex-dividends (next {earnings_within}d)"]
    if exdivs:
        out += [f"- {sym:<6} {xd.isoformat()}" for xd, sym in exdivs]
    else:
        out.append("- none in the window")
    return out


def _news_section(movers: list[Mover]) -> list[str]:
    from .tools import web_search

    out = ["", "## 📰 Headlines for movers"]
    for m in movers:
        hit = web_search(f"{m['symbol']} stock news", max_results=1)
        first_line = next(
            (ln.strip() for ln in hit.splitlines() if ln.strip().startswith("1.")), ""
        )
        out.append(f"- {m['symbol']:<6} {first_line[3:].strip() or '(no headline)'}")
    return out


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
    movers, earnings, exdivs = _scan_holdings(
        symbols, lookback_days, move_threshold, earnings_within, today,
        _fetch_daily, _fetch_calendar,
    )

    acct = account or (positions[0].get("account") if positions else "") or "default"
    lines = [
        f"# Portfolio digest · {today.isoformat()} · account {acct}",
        f"{len(symbols)} holding(s) · price lookback ~{lookback_days}d · "
        f"event window {earnings_within}d",
    ]
    # User-defined alert rules first — the personalized, act-on-it part of the
    # digest (shown only when rules exist), then the standing scans.
    lines += _alert_section(symbols, lookback_days, today)
    lines += _movers_section(movers, move_threshold)
    lines += _earnings_section(earnings, earnings_within)
    lines += _exdivs_section(exdivs, earnings_within)
    if include_news and movers:
        lines += _news_section(movers)
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
