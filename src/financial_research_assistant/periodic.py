"""Daily, weekly and monthly reports — the figures computed here, the meaning
written by the model, and nothing the model typed allowed to pass as a figure.

The same rule as ``reviews.py`` and ``stocks.py``, applied to the reports the
agent sends on its own. A scheduled task used to be a paragraph of instructions
("pull the YTD return, don't use the statement's TWR, label the window…") that
the model then followed or didn't; the wrong numbers on delivered sheets were
all numbers it had typed. So here a report is a JOB (``jobs.py``): code resolves
the period, fetches and computes every figure with its window written beside it,
and hands the model a finished sheet to comment on. Its commentary is then
checked (``unsupported_figures``) and dropped if it cites a number the sheet
doesn't contain.

**The period comes from the schedule, not the clock.** A report is for the
occurrence it was scheduled for (``jobs.scheduled_for``): the Saturday weekly
covers the week that ended on Friday even if it runs on Sunday after an outage,
and the ledger (``jobs.record_sent``) makes sure that week is sent once.

**Every figure says what it measured.** "Holdings +0.84%" is the price move of
the positions on file between two named closes — not the account's return,
which only the broker's statement can give (cash, trades since the statement,
and fees all sit outside it). Where the Flex file does cover the window, the
time-weighted return from it is used instead, and labelled as that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
import asyncio
import calendar
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from . import hooks, jobs

KINDS = ("daily", "weekly", "monthly")

INDEXES = (("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000"),
           ("DIA", "Dow Jones"))
SECTORS = (("XLK", "Technology"), ("XLF", "Financials"), ("XLE", "Energy"),
           ("XLV", "Health care"), ("XLY", "Consumer discretionary"),
           ("XLP", "Consumer staples"), ("XLI", "Industrials"), ("XLU", "Utilities"),
           ("XLB", "Materials"), ("XLRE", "Real estate"), ("XLC", "Communications"))
#: (symbol, label, unit) — "bp" for a yield quoted in percent, "pt" for an index
#: level whose points ARE the unit people quote, "%" for a price.
MACRO = (("^TNX", "US 10-year yield", "bp"), ("DX-Y.NYB", "US dollar index", "%"),
         ("CL=F", "WTI crude", "%"), ("GC=F", "Gold", "%"), ("^VIX", "VIX", "pt"))

#: A holding moving this much in a session is called out as a mover.
DEFAULT_MOVE_PCT = 3.0

#: Extra sections other modules add to a report: kind -> [builder(ctx) -> markdown].
#: The recommender adds its ideas here, the event watchers their recap.
SECTIONS: dict[str, list[Callable[[dict[str, Any]], str]]] = {k: [] for k in KINDS}


def register_section(kind: str, builder: Callable[[dict[str, Any]], str]) -> None:
    SECTIONS.setdefault(kind, []).append(builder)


@dataclass
class Brief:
    """One report, ready to render: every figure already in place."""

    kind: str
    period: str          # the ledger key's period: 2026-09-29, 2026-W39, 2026-09
    label: str           # human: "Tuesday 29 September 2026", "Week of 21 September"
    title: str
    subtitle: str
    highlights: str
    markdown: str
    message: str         # the short text pushed to the phone
    facts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --- figures -------------------------------------------------------------------------


def _fetch(symbol: str, days: int) -> list[tuple[str, float]]:
    from .tools import _fetch_daily  # pyright: ignore[reportPrivateUsage]

    return _fetch_daily(symbol, days)


def _fetch_many(symbols: list[str], days: int) -> dict[str, list[tuple[str, float]]]:
    syms = list(dict.fromkeys(symbols))
    if not syms:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(syms))) as pool:
        return dict(zip(syms, pool.map(lambda s: _fetch(s, days), syms)))


def _move(series: list[tuple[str, float]], start_after: str, end: str) -> dict[str, Any] | None:
    """Change from the last close ON OR BEFORE ``start_after`` to the last close on
    or before ``end``. None when either side is missing, or when the series has no
    close at ``end``'s session exactly (a stale source must not pass for today)."""
    upto = [(d, c) for d, c in series if d <= end]
    base = [(d, c) for d, c in series if d <= start_after]
    if not upto or not base or upto[-1][0] != end:
        return None
    (d0, p0), (d1, p1) = base[-1], upto[-1]
    if not p0 or d0 >= d1:
        return None
    return {"from": d0, "to": d1, "start": p0, "end": p1, "pct": (p1 / p0 - 1) * 100,
            "diff": p1 - p0}


def _prev_session(series: list[tuple[str, float]], day: str) -> str | None:
    earlier = [d for d, _c in series if d < day]
    return earlier[-1] if earlier else None


def _session_on_or_before(series: list[tuple[str, float]], day: str) -> str:
    """The session a window really starts from. "The close before Monday" is a
    Friday, and a label reading the Sunday it was computed from names a day with
    no close at all."""
    earlier = [d for d, _c in series if d <= day]
    return earlier[-1] if earlier else day


def _pct(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:+.{digits}f}%"


def _money(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "n/a"
    sign = "-" if v < 0 else ("+" if signed else "")
    return f"{sign}${abs(v):,.0f}" if abs(v) >= 1000 else f"{sign}${abs(v):,.2f}"


def _tile(label: str, value: str, note: str = "") -> str:
    return f"{label} | {value}" + (f" | {note}" if note else "")


def _pretty(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B %Y}"


# --- the book --------------------------------------------------------------------------


def _positions_as_of(account: str) -> str:
    """The date the positions on file are AS OF — the newest import's end date.
    A report built on a two-week-old statement must say so."""
    from . import statements

    for imp in statements.list_imports():
        if account and imp.get("account") != account:
            continue
        found = re.findall(r"[A-Z][a-z]+ \d{1,2}, \d{4}", str(imp.get("period") or ""))
        dates = []
        for text in found:
            try:
                dates.append(datetime.strptime(text, "%B %d, %Y").date())
            except ValueError:
                continue
        if dates:
            return max(dates).isoformat()
    return ""


#: IBKR asset categories priced from Yahoo, as the Activity CSV and the Flex XML
#: each spell them. The CSV says "Stocks"; Flex says "STK" — a filter that only
#: knew the first priced nothing at all for a Flex-imported book.
_EQUITY_CATEGORIES = frozenset({"stocks", "stock", "stk", "etf", "etfs"})
_CRYPTO_CATEGORIES = frozenset({"crypto", "cryptocurrency"})


def price_symbol(symbol: str, category: str, currency: str = "USD") -> str | None:
    """The Yahoo symbol for a holding, or None when Yahoo's series would not be the
    instrument held (an option, a bond, a future) — pricing those would put a
    wrong number on the sheet."""
    cat = (category or "").strip().lower()
    if cat in _EQUITY_CATEGORIES:
        return symbol
    if cat in _CRYPTO_CATEGORIES:
        # IBKR names a coin by its trading venue ("BTC.USD-PAXOS"); Yahoo by the
        # coin and the quote currency ("BTC-USD").
        coin = re.split(r"[.\-/ ]", symbol, maxsplit=1)[0]
        return f"{coin}-{(currency or 'USD').upper()}" if coin else None
    return None


def load_book(account: str = "") -> dict[str, Any]:
    """The positions a report measures, grouped by symbol.

    ``units`` is value / close price rather than the stored quantity, so a
    multiplier (options, some futures) is already inside it. Stocks, ETFs and
    crypto are priced; anything else is counted in ``unpriced_value``."""
    from . import statements

    account = account or statements.default_account() or ""
    rows = statements.query_positions(account=account or None)
    book: dict[str, dict[str, Any]] = {}
    other_value = 0.0
    for p in rows:
        raw = str(p.get("symbol") or "").strip().upper()
        value = float(p.get("value") or 0.0)
        close = float(p.get("close_price") or 0.0)
        sym = price_symbol(raw, str(p.get("asset_category") or ""),
                           str(p.get("currency") or "USD")) if raw else None
        if not sym or close <= 0:
            other_value += value
            continue
        entry = book.setdefault(sym, {"symbol": sym, "units": 0.0, "value": 0.0,
                                      "currency": p.get("currency") or "USD",
                                      "name": p.get("description") or ""})
        entry["units"] += value / close
        entry["value"] += value
    total = sum(e["value"] for e in book.values())
    for e in book.values():
        e["weight_pct"] = e["value"] / total * 100 if total else 0.0
    return {
        "account": account,
        "as_of": _positions_as_of(account),
        "holdings": sorted(book.values(), key=lambda e: e["value"], reverse=True),
        "total": total,
        "unpriced_value": other_value,
    }


def _book_moves(book: dict[str, Any], series: dict[str, list[tuple[str, float]]],
                start_after: str, end: str) -> dict[str, Any]:
    """Per-holding moves over one window and their dollar contribution."""
    rows = []
    for h in book["holdings"]:
        mv = _move(series.get(h["symbol"]) or [], start_after, end)
        if mv is None:
            continue
        contribution = h["units"] * mv["diff"] if h["currency"] == "USD" else None
        rows.append({**h, "move": mv, "pct": mv["pct"], "contribution": contribution})
    usd = [r for r in rows if r["contribution"] is not None]
    base = sum(r["units"] * r["move"]["start"] for r in usd)
    total = sum(r["contribution"] for r in usd)
    return {
        "rows": sorted(rows, key=lambda r: abs(r["pct"]), reverse=True),
        "priced": len(rows),
        "missing": [h["symbol"] for h in book["holdings"]
                    if h["symbol"] not in {r["symbol"] for r in rows}],
        "dollars": total if usd else None,
        "pct": (total / base * 100) if base else None,
    }


def _book_note(book: dict[str, Any]) -> str:
    return (f"price move of the {len(book['holdings'])} holdings on file "
            f"(positions as of {book['as_of'] or 'the last import'}); excludes cash, "
            "options and trades since")


# --- context the other modules supply --------------------------------------------------


def _calendar_ahead(
    symbols: list[str], today: date, days: int
) -> tuple[list[tuple[date, str, Any]], list[tuple[date, str]]]:
    from .fundamentals import _fetch_calendar  # pyright: ignore[reportPrivateUsage]
    from .monitor import _scan_holdings  # pyright: ignore[reportPrivateUsage]

    if not symbols:
        return [], []
    _movers, earnings, exdivs = _scan_holdings(
        symbols, 5, 1e9, days, today, _fetch, _fetch_calendar
    )
    return earnings, exdivs


def _calendar_symbols(book: dict[str, Any]) -> list[str]:
    """Holdings that can have an earnings date — not a coin."""
    return [h["symbol"] for h in book["holdings"] if not re.search(r"-[A-Z]{3}$", h["symbol"])]


def _open_calls_moving(threshold: float = 10.0) -> list[dict[str, Any]]:
    """Open journal calls whose price has moved past ±threshold since entry."""
    from . import journal

    out = []
    open_entries = [e for e in journal.load_entries() if e.get("status") == "open"]
    series = _fetch_many([str(e.get("symbol")) for e in open_entries], 10)
    for e in open_entries:
        rows = series.get(str(e.get("symbol"))) or []
        entry = float(e.get("entry_price") or 0)
        if not rows or entry <= 0:
            continue
        change = (rows[-1][1] / entry - 1) * 100
        if abs(change) >= threshold:
            out.append({**e, "since_entry_pct": change, "priced_on": rows[-1][0]})
    return out


def _run_sections(kind: str, ctx: dict[str, Any]) -> str:
    parts = []
    for builder in SECTIONS.get(kind, []):
        try:
            text = builder(ctx)
        except Exception as exc:  # noqa: BLE001 - one section must not sink a report
            text = f"_(a section could not be built: {type(exc).__name__})_"
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts)


# --- daily --------------------------------------------------------------------------------


def build_daily_brief(session: date, account: str = "",
                      move_pct: float = DEFAULT_MOVE_PCT) -> Brief:
    """The close of one session: market, sectors, macro, your holdings, what's next."""
    day = session.isoformat()
    book = load_book(account)
    symbols = ([s for s, _ in INDEXES] + [s for s, _ in SECTORS] + [s for s, _, _ in MACRO]
               + [h["symbol"] for h in book["holdings"]])
    series = _fetch_many(symbols, 15)

    def session_move(sym: str) -> dict[str, Any] | None:
        rows = series.get(sym) or []
        prev = _prev_session(rows, day)
        return _move(rows, prev, day) if prev else None

    idx = [(label, sym, session_move(sym)) for sym, label in INDEXES]
    spy = next((m for _l, s, m in idx if s == "SPY"), None)
    sectors = sorted(((label, session_move(sym)) for sym, label in SECTORS),
                     key=lambda r: r[1]["pct"] if r[1] else -1e9, reverse=True)
    sectors = [(label, mv) for label, mv in sectors if mv]
    macro = [(label, unit, session_move(sym)) for sym, label, unit in MACRO]

    prev_day = _prev_session(series.get("SPY") or [], day) or ""
    moves = _book_moves(book, series, prev_day, day) if prev_day else {
        "rows": [], "priced": 0, "missing": [], "dollars": None, "pct": None}
    movers = [r for r in moves["rows"] if abs(r["pct"]) >= move_pct]
    best = max(moves["rows"], key=lambda r: r["pct"], default=None)
    worst = min(moves["rows"], key=lambda r: r["pct"], default=None)
    earnings, exdivs = _calendar_ahead(_calendar_symbols(book), session, 7)
    calls = _open_calls_moving()

    def macro_value(unit: str, mv: dict[str, Any] | None) -> str:
        if mv is None:
            return "n/a"
        if unit == "bp":
            return f"{mv['end']:.2f}% ({mv['diff'] * 100:+.0f}bp)"
        if unit == "pt":
            return f"{mv['end']:.2f} ({mv['diff']:+.2f})"
        return f"{mv['end']:,.2f} ({mv['pct']:+.2f}%)"

    tiles = []
    if moves["pct"] is not None:
        tiles.append(_tile("Your holdings", _pct(moves["pct"]),
                           f"{prev_day} → {day} close, {_book_note(book)}"))
        tiles.append(_tile("Holdings P/L", _money(moves["dollars"]),
                           f"same window, USD positions"))
    if spy:
        tiles.append(_tile("S&P 500", _pct(spy["pct"]), f"SPY {spy['from']} → {spy['to']} close"))
    if best:
        tiles.append(_tile(f"Best: {best['symbol']}", _pct(best["pct"]), "this session"))
    if worst and worst is not best:
        tiles.append(_tile(f"Worst: {worst['symbol']}", _pct(worst["pct"]), "this session"))
    tnx = next((mv for label, unit, mv in macro if unit == "bp"), None)
    if tnx:
        tiles.append(_tile("10-year yield", f"{tnx['end']:.2f}%",
                           f"{tnx['diff'] * 100:+.0f}bp on the session"))

    md: list[str] = []
    md.append("## Market\n| Index | Session | Close |\n|---|---|---|")
    md += [f"| {label} | {_pct(m['pct'])} | {m['end']:,.2f} |" for label, _s, m in idx if m]
    if sectors:
        md.append("\n## Sectors\n" + "\n".join(
            f"- **{label}:** {_pct(mv['pct'])}" for label, mv in sectors))
    md.append("\n## Rates, Dollar and Commodities\n" + "\n".join(
        f"- **{label}:** {macro_value(unit, mv)}" for label, unit, mv in macro if mv))
    if moves["rows"]:
        md.append(f"\n## Your Holdings\n_{_book_note(book)}; session {prev_day} → {day}._\n\n"
                  "| Holding | Session | P/L | Weight |\n|---|---|---|---|")
        md += [f"| {r['symbol']} | {_pct(r['pct'])} | {_money(r['contribution'])} | "
               f"{r['weight_pct']:.1f}% |" for r in moves["rows"][:12]]
        if moves["missing"]:
            md.append(f"\nNo price for this session: {', '.join(moves['missing'])}.")
    elif book["holdings"]:
        md.append("\n## Your Holdings\nNo prices for this session yet.")
    if movers:
        md.append("\n## Movers\n" + "\n".join(
            f"- **{r['symbol']}:** {_pct(r['pct'])} ({r['move']['start']:,.2f} → "
            f"{r['move']['end']:,.2f})" for r in movers))
    ahead = [f"- **{sym}:** earnings {d.isoformat()}" for d, sym, _e in earnings]
    ahead += [f"- **{sym}:** ex-dividend {d.isoformat()}" for d, sym in exdivs]
    if ahead:
        md.append("\n## Coming Up (next 7 days)\n" + "\n".join(ahead))
    if calls:
        md.append("\n## Open Calls on the Move\n" + "\n".join(
            f"- **{c['symbol']}** ({c['verdict']}, opened {str(c['opened'])[:10]} at "
            f"{float(c['entry_price']):,.2f}): {_pct(c['since_entry_pct'])} since entry, "
            f"close {c['priced_on']}" for c in calls))
    ctx = {"kind": "daily", "session": session, "book": book, "moves": moves}
    extra = _run_sections("daily", ctx)
    if extra:
        md.append("\n" + extra)

    headline = []
    if moves["pct"] is not None:
        headline.append(f"Your holdings {_pct(moves['pct'])} ({_money(moves['dollars'])})")
    if spy:
        headline.append(f"S&P 500 {_pct(spy['pct'])}")
    message = [f"📊 Daily close · {_pretty(session)}", " · ".join(headline) or "Market recap"]
    if movers:
        message.append("Movers: " + ", ".join(f"{r['symbol']} {_pct(r['pct'], 1)}"
                                              for r in movers[:6]))
    if sectors:
        message.append(f"Sectors: best {sectors[0][0]} {_pct(sectors[0][1]['pct'], 1)}, "
                       f"worst {sectors[-1][0]} {_pct(sectors[-1][1]['pct'], 1)}")
    if tnx:
        message.append(f"10-year {tnx['end']:.2f}% ({tnx['diff'] * 100:+.0f}bp)")
    if ahead:
        message.append("Coming up: " + "; ".join(a[2:].replace("**", "") for a in ahead[:5]))
    if calls:
        message.append("Open calls moving: " + ", ".join(
            f"{c['symbol']} {_pct(c['since_entry_pct'], 1)}" for c in calls))

    facts = {
        "session": day, "prev_session": prev_day,
        "holdings_pct": moves["pct"], "holdings_dollars": moves["dollars"],
        "spy_pct": spy["pct"] if spy else None,
        "moves": {r["symbol"]: round(r["pct"], 2) for r in moves["rows"]},
        "sectors": {label: round(mv["pct"], 2) for label, mv in sectors},
        "macro": {label: (mv["end"], mv["diff"], mv["pct"]) for label, _u, mv in macro if mv},
        "positions_as_of": book["as_of"],
    }
    return Brief(
        kind="daily", period=day, label=_pretty(session),
        title=f"Daily Close — {_pretty(session)}",
        subtitle=f"Session {prev_day} → {day} · positions as of {book['as_of'] or 'n/a'}",
        highlights="\n".join(tiles[:6]), markdown="\n".join(md),
        message="\n".join(message), facts=facts,
    )


# --- weekly -----------------------------------------------------------------------------


def iso_week(day: date) -> str:
    y, w, _d = day.isocalendar()
    return f"{y}-W{w:02d}"


def build_weekly_brief(week_end: date, account: str = "") -> Brief:
    """The week ending ``week_end`` (its last session): portfolio, market, what
    carried it, what is coming, which calls were settled."""
    monday = week_end - timedelta(days=week_end.weekday())
    end = week_end.isoformat()
    base_day = (monday - timedelta(days=1)).isoformat()  # last close BEFORE Monday
    book = load_book(account)
    symbols = ([s for s, _ in INDEXES] + [s for s, _ in SECTORS]
               + [h["symbol"] for h in book["holdings"]])
    series = _fetch_many(symbols, 30)
    spy_rows = series.get("SPY") or []
    last = [d for d, _c in spy_rows if d <= end]
    if last:
        end = last[-1]  # a holiday Friday: the week ends on Thursday's close
    base_day = _session_on_or_before(spy_rows, base_day)
    moves = _book_moves(book, series, base_day, end)
    idx = [(label, _move(series.get(sym) or [], base_day, end)) for sym, label in INDEXES]
    spy = next((m for (label, m), (s, _l) in zip(idx, INDEXES) if s == "SPY"), None)
    sectors = sorted(((label, _move(series.get(sym) or [], base_day, end))
                      for sym, label in SECTORS),
                     key=lambda r: r[1]["pct"] if r[1] else -1e9, reverse=True)
    sectors = [(label, mv) for label, mv in sectors if mv]

    # The account's own time-weighted return when the Flex file reaches the week;
    # otherwise the holdings' price move, labelled as that.
    twr = None
    try:
        from . import flex

        r = flex.period_return(start=monday.isoformat(), end=end)
        if r["end"] == end:
            twr = r
    except Exception:  # noqa: BLE001 - no file, or it doesn't reach this week
        twr = None

    earnings, exdivs = _calendar_ahead(_calendar_symbols(book),
                                       week_end + timedelta(days=1), 7)
    from . import journal

    entries = journal.load_entries()
    settled = [e for e in entries if e.get("status") == "scored"
               and monday.isoformat() <= str(e.get("scored_on") or "") <= (week_end + timedelta(days=2)).isoformat()]
    opened = [e for e in entries
              if monday.isoformat() <= str(e.get("opened") or "")[:10] <= end]

    tiles = []
    if twr:
        tiles.append(_tile("Account return", _pct(twr["return_pct"]),
                           f"time-weighted, {twr['start']} → {twr['end']}, IBKR daily file"))
    if moves["pct"] is not None:
        tiles.append(_tile("Your holdings", _pct(moves["pct"]),
                           f"{base_day} → {end} close, {_book_note(book)}"))
        tiles.append(_tile("Holdings P/L", _money(moves["dollars"]), "same window, USD positions"))
    if spy:
        tiles.append(_tile("S&P 500", _pct(spy["pct"]), f"SPY {spy['from']} → {spy['to']} close"))
    if moves["pct"] is not None and spy:
        gap = moves["pct"] - spy["pct"]
        tiles.append(_tile("vs S&P 500", f"{gap:+.2f} pp", "holdings minus SPY, same closes"))
    if settled:
        hits = sum(1 for e in settled if e.get("hit"))
        tiles.append(_tile("Calls settled", f"{hits}/{len(settled)} right",
                           "journal calls whose horizon ended this week"))

    md: list[str] = []
    if moves["rows"]:
        contrib = sorted((r for r in moves["rows"] if r["contribution"] is not None),
                         key=lambda r: r["contribution"], reverse=True)
        md.append(f"## What Carried the Week\n_{_book_note(book)}; {base_day} → {end}._\n\n"
                  "| Holding | Week | P/L | Weight |\n|---|---|---|---|")
        md += [f"| {r['symbol']} | {_pct(r['pct'])} | {_money(r['contribution'])} | "
               f"{r['weight_pct']:.1f}% |" for r in contrib[:12]]
    md.append("\n## Market\n| Index | Week |\n|---|---|")
    md += [f"| {label} | {_pct(m['pct'])} |" for label, m in idx if m]
    if sectors:
        md.append("\n## Sectors This Week\n" + "\n".join(
            f"- **{label}:** {_pct(mv['pct'])}" for label, mv in sectors))
    if twr:
        md.append(f"\n## Account\nTime-weighted return {_pct(twr['return_pct'])} over "
                  f"{twr['sessions']} sessions ({twr['start']} → {twr['end']}); net asset value "
                  f"{_money(twr['nav_start'], signed=False)} → {_money(twr['nav_end'], signed=False)}, "
                  f"deposits {_money(twr['deposits'], signed=False)}, investment gain "
                  f"{_money(twr['investment_gain'])}.")
    ahead = [f"- **{sym}:** earnings {d.isoformat()}" for d, sym, _e in earnings]
    ahead += [f"- **{sym}:** ex-dividend {d.isoformat()}" for d, sym in exdivs]
    md.append("\n## Next Week\n" + ("\n".join(ahead) if ahead else
                                     "- No earnings or ex-dividend dates for your holdings."))
    if settled or opened:
        lines = [f"- **{e['symbol']}** {e['verdict']}: {_pct(e.get('change_pct'))} vs "
                 f"{e.get('benchmark')} {_pct(e.get('benchmark_change_pct'))} over "
                 f"{e.get('horizon_days')}d — {'right' if e.get('hit') else 'wrong'}"
                 for e in settled]
        lines += [f"- **{e['symbol']}** {e['verdict']} opened at "
                  f"{float(e['entry_price']):,.2f} ({e.get('horizon_days')}d)" for e in opened]
        md.append("\n## Calls\n" + "\n".join(lines))
    ctx = {"kind": "weekly", "week_end": week_end, "book": book, "moves": moves}
    extra = _run_sections("weekly", ctx)
    if extra:
        md.append("\n" + extra)

    label = f"week of {monday.day} {monday:%B %Y}"
    if end < (monday + timedelta(days=4)).isoformat():
        # Asked for mid-week: say it is the week SO FAR, in the title as well as
        # the subtitle, so a Tuesday snapshot never reads as the whole week.
        label += f" (to {date.fromisoformat(end):%A %d %B})"
    headline = []
    if twr:
        headline.append(f"Account {_pct(twr['return_pct'])}")
    if moves["pct"] is not None:
        headline.append(f"holdings {_pct(moves['pct'])} ({_money(moves['dollars'])})")
    if spy:
        headline.append(f"S&P 500 {_pct(spy['pct'])}")
    message = [f"🗓 Weekly report · {label}", " · ".join(headline) or "Market recap"]
    if moves["rows"]:
        top = max(moves["rows"], key=lambda r: r["pct"])
        bottom = min(moves["rows"], key=lambda r: r["pct"])
        message.append(f"Best {top['symbol']} {_pct(top['pct'], 1)}, "
                       f"worst {bottom['symbol']} {_pct(bottom['pct'], 1)}")
    if ahead:
        message.append("Next week: " + "; ".join(a[2:].replace("**", "") for a in ahead[:5]))
    message.append("Full report attached.")
    facts = {
        "week": iso_week(week_end), "from": base_day, "to": end,
        "holdings_pct": moves["pct"], "holdings_dollars": moves["dollars"],
        "spy_pct": spy["pct"] if spy else None,
        "twr_pct": twr["return_pct"] if twr else None,
        "moves": {r["symbol"]: round(r["pct"], 2) for r in moves["rows"]},
        "contributions": {r["symbol"]: round(r["contribution"], 2)
                          for r in moves["rows"] if r["contribution"] is not None},
        "sectors": {label: round(mv["pct"], 2) for label, mv in sectors},
        "settled": [(e["symbol"], e.get("change_pct"), e.get("hit")) for e in settled],
    }
    return Brief(
        kind="weekly", period=iso_week(week_end), label=label,
        title=f"Weekly Report — {label}",
        subtitle=f"{base_day} close → {end} close · positions as of {book['as_of'] or 'n/a'}",
        highlights="\n".join(tiles[:6]), markdown="\n".join(md),
        message="\n".join(message), facts=facts,
    )


# --- monthly -------------------------------------------------------------------------------


def _track_record(until: str) -> dict[str, Any]:
    """The recommender's scored calls up to ``until``: hit rate and excess return,
    overall and by conviction — the figure that says whether its ideas are any
    good."""
    from . import journal

    scored = [e for e in journal.load_entries()
              if e.get("source") == "recommender" and e.get("status") == "scored"
              and str(e.get("scored_on") or "") <= until]
    by_conv: dict[str, list[dict[str, Any]]] = {}
    for e in scored:
        by_conv.setdefault(str(e.get("conviction") or "?"), []).append(e)

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        alphas = [float(e["alpha_pct"]) for e in rows if e.get("alpha_pct") is not None]
        return {"n": len(rows), "hits": sum(1 for e in rows if e.get("hit")),
                "avg_alpha": sum(alphas) / len(alphas) if alphas else None}

    return {"all": stats(scored), "by_conviction": {k: stats(v) for k, v in sorted(by_conv.items())}}


def build_monthly_brief(year: int, month: int, account: str = "") -> Brief:
    """The month: the performance review (from the Flex file when it covers the
    month), the track record of the agent's own calls, and whatever sections the
    other modules add (the deep ideas run, the service's health)."""
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    period = f"{year}-{month:02d}"
    label = f"{calendar.month_name[month]} {year}"
    review: dict[str, Any] | None = None
    notes: list[str] = []
    try:
        from . import reviews

        review = reviews.build_review(period=f"{first.isoformat()}..{last.isoformat()}",
                                      account=account)
    except Exception as exc:  # noqa: BLE001 - no Flex file, or it misses the month
        notes.append(f"The account's own return is not available for {label}: {exc}")

    book = load_book(account)
    base_day = (first - timedelta(days=1)).isoformat()
    series = _fetch_many(["SPY"] + [h["symbol"] for h in book["holdings"]], 75)
    spy_rows = [d for d, _c in series.get("SPY") or [] if d <= last.isoformat()]
    end = spy_rows[-1] if spy_rows else last.isoformat()
    base_day = _session_on_or_before(series.get("SPY") or [], base_day)
    moves = _book_moves(book, series, base_day, end)
    spy = _move(series.get("SPY") or [], base_day, end)
    if review and review["facts"].get("truncated"):
        # The account figure covers only part of the month (the Flex file stops
        # short). The market beside it must cover the SAME days, or the sheet
        # compares two different windows and calls the gap performance.
        f = review["facts"]
        spy_series = series.get("SPY") or []
        before = _prev_session(spy_series, f["start"])
        spy = _move(spy_series, before, f["end"]) if before else None
        notes.append(f"The account return covers {f['start']} to {f['end']} only — the "
                     "saved IBKR file ends there; the S&P 500 figure is for the same days.")
    record = _track_record(end)

    tiles = []
    md: list[str] = []
    if review:
        tiles += [t for t in review["highlights"].splitlines() if t.strip()][:4]
        md.append(review["markdown"].strip())
    elif moves["pct"] is not None:
        tiles.append(_tile("Your holdings", _pct(moves["pct"]),
                           f"{base_day} → {end} close, {_book_note(book)}"))
    if spy:
        tiles.append(_tile("S&P 500", _pct(spy["pct"]), f"SPY {spy['from']} → {spy['to']} close"))
    if record["all"]["n"]:
        r = record["all"]
        tiles.append(_tile("Ideas track record", f"{r['hits']}/{r['n']} right",
                           "recommendations scored so far, vs their benchmarks"))
    if not review and moves["rows"]:
        md.append(f"## Holdings This Month\n_{_book_note(book)}; {base_day} → {end}._\n\n"
                  "| Holding | Month | P/L |\n|---|---|---|")
        md += [f"| {r['symbol']} | {_pct(r['pct'])} | {_money(r['contribution'])} |"
               for r in moves["rows"][:15]]
    if record["all"]["n"]:
        lines = ["## Track Record of the Agent's Ideas",
                 "| Conviction | Scored | Right | Avg vs benchmark |", "|---|---|---|---|"]
        for conv, st in record["by_conviction"].items():
            lines.append(f"| {conv} | {st['n']} | {st['hits']} | "
                         f"{_pct(st['avg_alpha']) if st['avg_alpha'] is not None else 'n/a'} |")
        a = record["all"]
        lines.append(f"| All | {a['n']} | {a['hits']} | "
                     f"{_pct(a['avg_alpha']) if a['avg_alpha'] is not None else 'n/a'} |")
        md.append("\n" + "\n".join(lines))
    ctx = {"kind": "monthly", "year": year, "month": month, "book": book, "moves": moves}
    extra = _run_sections("monthly", ctx)
    if extra:
        md.append("\n" + extra)

    if review and review["facts"].get("truncated"):
        label = review["period_label"]
    headline = []
    if review:
        f = review["facts"]
        headline.append(f"Return {_pct(f['return_pct'])} (time-weighted, "
                        f"{f['start']} → {f['end']})")
    elif moves["pct"] is not None:
        headline.append(f"Holdings {_pct(moves['pct'])}")
    if spy:
        headline.append(f"S&P 500 {_pct(spy['pct'])}")
    message = [f"📅 Monthly report · {label}", " · ".join(headline) or "Month in review"]
    if record["all"]["n"]:
        message.append(f"Ideas so far: {record['all']['hits']}/{record['all']['n']} right")
    message += notes
    message.append("Full report attached.")
    facts = {
        "month": period, "from": base_day, "to": end,
        "review": review["facts"] if review else None,
        "holdings_pct": moves["pct"], "spy_pct": spy["pct"] if spy else None,
        "track_record": record,
    }
    return Brief(
        kind="monthly", period=period, label=label,
        title=f"Monthly Report — {label}",
        subtitle=(review["subtitle"] if review else f"{base_day} close → {end} close"),
        highlights="\n".join(tiles[:6]), markdown="\n".join(md),
        message="\n".join(message), facts=facts, notes=notes,
    )


# --- commentary ------------------------------------------------------------------------------

#: Phrases whose digits are names, not figures ("S&P 500", "10-year").
_NAMED_NUMBERS = re.compile(
    r"S&P\s*500|Nasdaq[- ]?100|Russell\s*2000|Dow\s*30|\b\d{1,2}-year\b|\b\d{1,2}-day\b|"
    r"\bQ[1-4]\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{4}-W\d{2}\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_FIGURE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[tuple[str, float, int]]:
    out = []
    for m in _FIGURE.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        out.append((m.group(0), value, decimals))
    return out


def unsupported_figures(commentary: str, brief: Brief) -> list[str]:
    """Figures in ``commentary`` that appear nowhere in the brief.

    A figure is supported when some number on the sheet rounds to it at the
    precision it was written with — "0.8%" is the sheet's 0.84 — so the model may
    round, but not compute. Small whole numbers (counts: "3 of 5 holdings") and
    the digits of names and dates pass.
    """
    sheet = "\n".join([brief.highlights, brief.markdown, brief.message,
                       json.dumps(brief.facts, default=str)])
    allowed = [abs(v) for _s, v, _d in _numbers(_NAMED_NUMBERS.sub(" ", sheet))]
    bad = []
    for text, value, decimals in _numbers(_NAMED_NUMBERS.sub(" ", commentary)):
        if decimals == 0 and value <= 12:
            continue
        if any(round(a, decimals) == round(value, decimals) for a in allowed):
            continue
        bad.append(text)
    return bad


_SYSTEM = {
    "daily": (
        "You write the two-line commentary on a daily market close report for one "
        "investor. Say what mattered today for THEIR holdings, in plain words, in at "
        "most 120 words. Use only figures that appear in the report; you may round "
        "them but never compute new ones (no sums, differences or averages). No "
        "advice to trade. No preamble."
    ),
    "weekly": (
        "You write the Observations section of a weekly portfolio report: 3-5 "
        "markdown bullets, one insight each — what carried the week, how it compared "
        "with the market, what next week holds for these holdings, anything "
        "unflattering. Use only figures that appear in the report; you may round "
        "them but never compute new ones. No preamble, no headings."
    ),
    "monthly": (
        "You write the Observations section of a monthly portfolio review: 4-6 "
        "markdown bullets on what the month's figures MEAN — whether growth came "
        "from deposits or returns, which holdings carried it, concentration, how "
        "the agent's own recommendations have fared. Use only figures that appear in "
        "the report; you may round them but never compute new ones. Be candid about "
        "anything unflattering. No preamble, no headings."
    ),
}


async def write_commentary(brief: Brief, fake: bool = False) -> tuple[str, str]:
    """``(commentary, note)``: the model's reading of the brief, or "" with a note
    saying why there is none. Checked by ``unsupported_figures``; one retry naming
    the offending figures, then the report goes out without it."""
    from . import autonomy

    why = autonomy.blocked()
    if why and not fake:
        return "", f"No commentary: {why}."
    tier = "quick" if brief.kind == "daily" else "default"
    user = (f"REPORT: {brief.title}\n{brief.subtitle}\n\nHEADLINE FIGURES:\n"
            f"{brief.highlights}\n\n{brief.markdown}")
    fake_reply = "- " + (brief.highlights.splitlines()[0] if brief.highlights else "A quiet period.")
    text = await autonomy.ask(_SYSTEM[brief.kind], user, tier=tier,
                              purpose=f"report:{brief.kind}", fake=fake, fake_reply=fake_reply)
    if not text:
        return "", "No commentary: the model was unavailable."
    bad = unsupported_figures(text, brief)
    if bad:
        retry = await autonomy.ask(
            _SYSTEM[brief.kind],
            user + "\n\nYour previous draft cited figures that are not in the report: "
            + ", ".join(bad) + ". Rewrite it using ONLY figures shown above.",
            tier=tier, purpose=f"report:{brief.kind}:retry", fake=fake, fake_reply=fake_reply,
        )
        if retry and not unsupported_figures(retry, brief):
            return retry, ""
        return "", ("Commentary withheld: the model cited figures not in this report ("
                    + ", ".join(bad[:5]) + ").")
    return text, ""


# --- running a report ------------------------------------------------------------------------


def _render(brief: Brief, commentary: str, note: str) -> list[str]:
    """PDF + cover for the report; the files to send (cover first). Never raises."""
    from . import reports

    body = brief.markdown
    if commentary:
        lines = [ln.strip() for ln in commentary.splitlines() if ln.strip()]
        bullets = "\n".join(ln if ln.startswith(("-", "*")) else f"- {ln}" for ln in lines)
        body = f"## Observations\n{bullets}\n\n{body}"
    if note:
        body += f"\n\n_{note}_"
    content = {"title": brief.title, "subtitle": brief.subtitle, "highlights": brief.highlights,
               "markdown": body, "eyebrow": f"{brief.kind.title()} report"}
    try:
        paths = reports.render("", f"{brief.kind}-report-{brief.period}", content=content)
    except Exception:  # noqa: BLE001 - the text message still goes out
        return []
    return [p for p in (paths.get("png"), paths.get("pdf")) if p]


def _period_for(kind: str, when: datetime, tz: str | None) -> tuple[Any, str]:
    """What a report scheduled at ``when`` covers: ``(argument, period key)``."""
    from . import tasks

    local = when.astimezone(tasks.zone(tz)) if tz else when.astimezone()
    today = local.date()
    if kind == "daily":
        ny = when.astimezone(tasks.zone("America/New_York")).date()
        return ny, ny.isoformat()
    if kind == "weekly":
        end = today - timedelta(days=1)  # a Saturday run covers the week to Friday
        while end.weekday() >= 5:
            end -= timedelta(days=1)
        return end, iso_week(end)
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return (y, m), f"{y}-{m:02d}"


def _latest_complete_period(kind: str) -> tuple[Any, str]:
    """For an on-demand report: the most recent period that has data."""
    from . import market

    session = market.latest_session() or datetime.now(timezone.utc).date()
    if kind == "daily":
        return session, session.isoformat()
    if kind == "weekly":
        return session, iso_week(session)
    today = datetime.now().date()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return (y, m), f"{y}-{m:02d}"


async def produce(kind: str, arg: Any, period: str, fake: bool = False,
                  account: str = "") -> tuple[Brief, list[str], str]:
    """Build, comment and render one report: ``(brief, files, commentary note)``."""
    if kind == "daily":
        brief = await asyncio.to_thread(build_daily_brief, arg, account)
    elif kind == "weekly":
        brief = await asyncio.to_thread(build_weekly_brief, arg, account)
    else:
        brief = await asyncio.to_thread(build_monthly_brief, arg[0], arg[1], account)
    commentary, note = await write_commentary(brief, fake=fake)
    want_pdf = kind != "daily" or (os.environ.get("FRA_DAILY_PDF") or "").strip() in ("1", "true", "yes")
    files = await asyncio.to_thread(_render, brief, commentary, note) if want_pdf else []
    if commentary:
        head, _, rest = brief.message.partition("\n")
        brief.message = f"{head}\n{rest}\n\n{commentary}".strip()
    if note:
        brief.message += f"\n\n({note})"
    return brief, files, note


def _job(kind: str) -> Callable[[dict[str, Any], bool], Any]:
    async def run(task: dict[str, Any], fake: bool) -> jobs.JobResult:
        from . import guardrails, market

        paused = guardrails.paused_reason()
        if paused:
            return jobs.JobResult(True, f"{kind} report skipped: {paused}", notify=False)
        arg, period = _period_for(kind, jobs.scheduled_for(task), task.get("tz") or None)
        if jobs.already_sent(kind, period):
            return jobs.JobResult(True, f"{kind} report for {period} already sent", notify=False)
        if kind == "daily" and not fake:
            last = await asyncio.to_thread(market.latest_session)
            if last is None:
                # The price source is down: fail, so the retry path tries again.
                return jobs.JobResult(False, "no market data yet for the daily report")
            if last < arg:
                # A holiday or a closed market: nothing to report, and nothing to
                # enter in the ledger (the date had no session).
                return jobs.JobResult(True, f"no session on {period}", notify=False)
        options = task.get("options") or {}
        brief, files, _note = await produce(kind, arg, period, fake=fake,
                                            account=str(options.get("account") or ""))
        jobs.record_sent(kind, period, {"files": files, "title": brief.title})
        return jobs.JobResult(True, brief.message, files=files)

    return run


for _kind in KINDS:
    jobs.register_job(f"report-{_kind}", _job(_kind))

jobs.register_default_job(jobs.DefaultJob(
    "report-daily", "[job] Daily close report", "17:15", "weekdays", tz="America/New_York"))
jobs.register_default_job(jobs.DefaultJob(
    "report-weekly", "[job] Weekly report", "saturday 09:00", "weekly"))
jobs.register_default_job(jobs.DefaultJob(
    "report-monthly", "[job] Monthly report", "first weekday of the month 08:00",
    "monthly-first-weekday"))


# --- on demand -------------------------------------------------------------------------------


async def report_now(kind: str, deliver: bool = True, fake: bool = False) -> str:
    """Build the most recent complete report of ``kind`` now, ledger or not (the
    user asked), send its files if ``deliver``, and return its message."""
    from . import channels

    kind = (kind or "").strip().lower()
    if kind not in KINDS:
        return f"Which report? One of: {', '.join(KINDS)}."
    arg, period = await asyncio.to_thread(_latest_complete_period, kind)
    brief, files, _note = await produce(kind, arg, period, fake=fake)
    jobs.record_sent(kind, period, {"files": files, "title": brief.title, "on_demand": True})
    sent = []
    if deliver:
        for path in files:
            ok, _failed = await asyncio.to_thread(channels.deliver_file, path, "", "", True)
            sent += ok
    tail = ""
    if files:
        tail = ("\n\n(files sent)" if sent else "\n\nFiles: " + ", ".join(files))
    return brief.message + tail


async def _cmd_report(arg: str, fake: bool) -> str:
    return await report_now(arg or "daily", deliver=True, fake=fake)


hooks.register_command("report", _cmd_report, "a report now: /report daily|weekly|monthly")


async def periodic_report(kind: str = "daily", deliver: bool = True) -> str:
    """Build and send the daily, weekly or monthly report NOW. Use for 'send me
    today's report / the weekly report / last month's report'. Every figure is
    computed from market data and the imported statement; do not build these
    reports by hand with render_report. ``kind`` is daily (the latest session),
    weekly (the week of the latest session) or monthly (last calendar month).
    Returns the report's summary; the PDF goes to the delivery channels when
    ``deliver`` is true."""
    return await report_now(kind, deliver=deliver)


def _make_tools() -> list[Any]:
    from langchain_core.tools import StructuredTool

    return [StructuredTool.from_function(coroutine=periodic_report, name="periodic_report")]


PERIODIC_TOOLS = _make_tools()
