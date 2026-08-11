"""Stock report templates: a ticker in, a ready-to-render brief out.

The same lesson as ``reviews.py``, learned again on a different sheet. A
delivered earnings report put Fiserv's Q2 2026 revenue at $4.96B where the
as-reported figure is $5.29B, and called the stock down 63% in one bullet and
70% in another while the tile beside them read -66.3%.

Only some of that was wrong, and the interesting part is which. **-66.3% was
correct** — it is the maximum drawdown over the TRAILING TWELVE MONTHS, and it
came from a tool that computed it properly. But the sheet never said "trailing
twelve months", and measured from its actual high the stock is down 78%. So a
correct figure sat on the page answering a question nobody had asked, and the
model's own prose drifted around it precisely because "the decline" had no fixed
window to be checked against.

A number is therefore not enough on its own. Every figure built here travels
with the window it covers and the basis it is on, in the note printed beside it:
"trailing 12 months, peak to trough", "GAAP as-reported", "high since
2023-05-01". A reader can tell what was measured, and a later sentence that
contradicts it is visibly a different measurement rather than a second opinion.

What is left for the model is what the quarter MEANS, and none of what it
measured.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

#: How far either side of exactly one year a quarter-end may sit and still count
#: as the year-ago comparison. Fiscal quarters drift by a few days and a filer
#: occasionally reports a 53-week year; beyond this it is a different period.
_YOY_TOLERANCE_DAYS = 40

#: Calendar days of price history fetched. Long enough to reach a prior cycle's
#: high — the figure a reader means by "down from its peak" — without claiming to
#: be all-time, which this cannot know and therefore never says.
_PRICE_DAYS = 1200

#: The trailing window for drawdown and the 52-week range.
_YEAR_DAYS = 365


def _iso(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _money(amount: float | None) -> str:
    """Compact money, sign outside the currency mark: ``-$1.23B``, ``$5.29B``."""
    if amount is None:
        return "n/a"
    sign = "-" if amount < 0 else ""
    size = abs(amount)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if size >= scale:
            return f"{sign}${size / scale:.2f}{suffix}"
    return f"{sign}${size:,.0f}"


def _tile(label: str, value: str, note: str = "") -> str:
    return f"{label} | {value}" + (f" | {note}" if note else "")


def _change_pct(new: float | None, old: float | None) -> float | None:
    """Percent change, or None when either side is missing or the base is zero.

    A base of zero has no percentage change, and returning 0.0 for it would put a
    flat quarter on the sheet where the truth is that the comparison cannot be
    made.
    """
    if new is None or old is None or not old:
        return None
    return (new - old) / abs(old) * 100.0


def _ratio_pct(part: float | None, whole: float | None) -> float | None:
    """A LEVEL: ``part`` as a percentage OF ``whole``.

    Kept separate from `_change_pct` and named for what it is. Margin computed
    with the change formula reads -88.2% where the answer is 11.8% — the same
    level-for-a-change slip this whole module exists to stop, and it went onto a
    generated sheet from here before a live run caught it.
    """
    if part is None or whole is None or not whole:
        return None
    return part / whole * 100.0


def _pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


# --- the filings side ----------------------------------------------------------


def _value_at(rows: dict[str, Any], end: str) -> float | None:
    from . import edgar

    return edgar._num((rows.get(end) or {}).get("val"))


def _year_ago_end(ends: list[str], rows: dict[str, Any], newest: str) -> str | None:
    """The quarter-end closest to one year before ``newest``, within tolerance.

    Matched by DATE, never by position. A filer's fiscal-year-end quarter is
    absent from the 10-Q series — the 10-K reports that period as the full year —
    so counting four back through the list silently spans the gap and labels a
    fifteen-month change as year-over-year.
    """
    anchor = _iso(newest)
    if anchor is None:
        return None
    best, best_diff = None, _YOY_TOLERANCE_DAYS + 1
    for end in ends:
        if end == newest or _value_at(rows, end) is None:
            continue
        other = _iso(end)
        if other is None:
            continue
        diff = abs((anchor - other).days - _YEAR_DAYS)
        if diff < best_diff:
            best, best_diff = end, diff
    return best


def _prior_quarter_end(ends: list[str], rows: dict[str, Any], newest: str) -> str | None:
    """The immediately preceding quarter, only when it really is ~one quarter back."""
    anchor = _iso(newest)
    if anchor is None:
        return None
    for end in ends:
        other = _iso(end)
        if end == newest or other is None or _value_at(rows, end) is None:
            continue
        if 80 <= (anchor - other).days <= 100:
            return end
    return None


def _filing_facts(symbol: str, quarters: int) -> dict[str, Any]:
    """Latest reported quarter plus its year-ago and prior-quarter comparisons.

    Every figure is GAAP as-reported, straight from 10-Q XBRL. That basis is
    recorded here and printed in the tiles, because the non-GAAP "adjusted"
    figures a company headlines in its press release are genuinely different
    numbers — the run that reported $4.96B against an as-reported $5.29B was
    plausibly quoting one, and the sheet said neither.
    """
    from . import edgar

    cik = edgar._cik_for(symbol)
    if not cik:
        raise ValueError(edgar._no_cik(symbol))
    facts = edgar._fetch_company_facts(cik)
    name = str(facts.get("entityName") or symbol).strip() or symbol
    series, ends = edgar._quarterly_series(facts, quarters)
    revenue = series.get(edgar._REVENUE) or {}
    valued = [e for e in ends if _value_at(revenue, e) is not None]
    if not valued:
        raise ValueError(
            f"No quarterly (10-Q) XBRL revenue found for {symbol}. SEC EDGAR covers "
            "US filers; a foreign issuer filing 20-F/40-F has no 10-Q series."
        )
    newest = valued[0]
    prior = _prior_quarter_end(ends, revenue, newest)
    year_ago = _year_ago_end(ends, revenue, newest)

    def figures(end: str | None) -> dict[str, float | None]:
        if end is None:
            return {"revenue": None, "net_income": None, "eps": None, "margin": None}
        rev = _value_at(series.get(edgar._REVENUE) or {}, end)
        net = _value_at(series.get(edgar._NET_INCOME) or {}, end)
        return {
            "revenue": rev,
            "net_income": net,
            "eps": _value_at(series.get("Diluted EPS") or {}, end),
            # A margin is a LEVEL. It is computed rather than differenced on
            # purpose: a delivered chart of year-over-year CHANGES carried
            # "net margin compressed +11.8%", which is this number, sitting
            # among moves and drawn as the one thing that rose in a bad quarter.
            "margin": _ratio_pct(net, rev),
        }

    return {
        "symbol": symbol.upper(),
        "name": name,
        "basis": "GAAP as-reported (10-Q XBRL)",
        "quarter_end": newest,
        "prior_end": prior,
        "year_ago_end": year_ago,
        "latest": figures(newest),
        "prior": figures(prior),
        "year_ago": figures(year_ago),
        "history": [(e, _value_at(revenue, e)) for e in valued[:6]],
        "eps_history": [
            (e, _value_at(series.get("Diluted EPS") or {}, e)) for e in valued[:6]
        ],
        "margin_history": [
            (
                e,
                _ratio_pct(
                    _value_at(series.get(edgar._NET_INCOME) or {}, e),
                    _value_at(revenue, e),
                ),
            )
            for e in valued[:6]
        ],
    }


# --- the price side ------------------------------------------------------------


def _max_drawdown_pct(rows: list[tuple[str, float]]) -> tuple[float, str]:
    """Worst peak-to-trough fall inside ``rows``, and the date it bottomed."""
    if not rows:
        return 0.0, ""
    run, worst, at = rows[0][1], 0.0, rows[0][0]
    for when, close in rows:
        run = max(run, close)
        fall = close / run - 1.0 if run else 0.0
        if fall < worst:
            worst, at = fall, when
    return worst * 100.0, at


def _price_facts(symbol: str, days: int) -> dict[str, Any]:
    """Price level, the high this window reaches back to, and two named windows.

    The window is stated everywhere it is used. "Down 66%" and "down 78%" are
    both true of the same stock on the same day over different spans, and a sheet
    that prints one without saying which invites the other to be written beside
    it in prose.
    """
    from .tools import _fetch_daily

    rows = _fetch_daily(symbol, days=days)
    if not rows:
        raise ValueError(
            f"No price history for {symbol!r} — an unknown, delisted or "
            "non-US-listed ticker."
        )
    last_day, last_close = rows[-1]
    peak_day, peak = max(rows, key=lambda r: r[1])
    anchor = _iso(last_day)
    cutoff = (anchor - timedelta(days=_YEAR_DAYS)).isoformat() if anchor else ""
    trailing = [r for r in rows if r[0] >= cutoff] or rows
    drawdown, trough_day = _max_drawdown_pct(trailing)
    return {
        "last_close": last_close,
        "last_day": last_day,
        "peak": peak,
        "peak_day": peak_day,
        # NOT called all-time. This is the high of the fetched window and nothing
        # here can see further back, so the note names the window's own start
        # date rather than making a claim the data cannot support.
        "window_start": rows[0][0],
        "from_peak_pct": _change_pct(last_close, peak),
        "drawdown_pct": drawdown,
        "trough_day": trough_day,
        "year_high": max(c for _d, c in trailing),
        "year_low": min(c for _d, c in trailing),
        "sessions": len(rows),
    }


# --- the brief -----------------------------------------------------------------


def _quarter_label(end: str) -> str:
    """``2026-06-30`` → ``Q2 2026``, by calendar quarter of the period end."""
    when = _iso(end)
    if when is None:
        return end
    return f"Q{(when.month - 1) // 3 + 1} {when.year}"


def _table(title: str, header: str, rows: list[str]) -> str:
    """A markdown table, or nothing at all when there is too little to chart.

    Below three rows a table reads as a stub and cannot chart, so the section is
    omitted rather than shipped half-empty.
    """
    if len(rows) < 3:
        return ""
    body = "\n".join(rows)
    return f"## {title}\n| {header} |\n|---|---|\n{body}\n\n"


def build_stock_brief(
    symbol: str, quarters: int = 8, price_days: int = _PRICE_DAYS
) -> dict[str, Any]:
    """Every figure a single-stock report needs, computed and labelled.

    Returns ``{title, subtitle, highlights, markdown, facts}`` in the shape
    ``render_report`` consumes, the same contract ``reviews.build_review`` uses.
    """
    filings = _filing_facts(symbol, quarters)
    prices = _price_facts(symbol, price_days)
    latest, year_ago = filings["latest"], filings["year_ago"]
    quarter = _quarter_label(filings["quarter_end"])
    rev_yoy = _change_pct(latest["revenue"], year_ago["revenue"])
    eps_yoy = _change_pct(latest["eps"], year_ago["eps"])
    basis = filings["basis"]

    def against(end: str | None, shown: str) -> str:
        return f"vs {shown} in {_quarter_label(end)}" if end else "no year-ago quarter"

    highlights = "\n".join([
        _tile(
            f"Revenue {quarter}", _money(latest["revenue"]),
            f"{_pct(rev_yoy)} YoY · {basis}",
        ),
        _tile(
            "Diluted EPS", f"${latest['eps']:.2f}" if latest["eps"] is not None else "n/a",
            f"{_pct(eps_yoy)} YoY · " + against(
                filings["year_ago_end"],
                f"${year_ago['eps']:.2f}" if year_ago["eps"] is not None else "n/a",
            ),
        ),
        # Labelled a LEVEL in the note, which is the whole point: written bare it
        # gets restated as a move and charted among changes.
        _tile(
            "Net margin", _pct(latest["margin"], signed=False),
            f"level, not a change · was {_pct(year_ago['margin'], signed=False)} a year ago",
        ),
        _tile(
            "Price", f"${prices['last_close']:,.2f}", f"close, {prices['last_day']}",
        ),
        _tile(
            "From the high", _pct(prices["from_peak_pct"]),
            f"${prices['peak']:,.2f} on {prices['peak_day']} — high since "
            f"{prices['window_start']}",
        ),
        _tile(
            "Max drawdown", _pct(prices["drawdown_pct"]),
            f"trailing 12 months, peak to trough — trough {prices['trough_day']}",
        ),
    ])

    revenue_rows = [
        f"| {_quarter_label(e)} | {_money(v)} |" for e, v in filings["history"]
    ]
    eps_rows = [
        f"| {_quarter_label(e)} | ${v:.2f} |"
        for e, v in filings["eps_history"] if v is not None
    ]
    margin_rows = [
        f"| {_quarter_label(e)} | {m:.1f}% |"
        for e, m in filings["margin_history"] if m is not None
    ]
    markdown = (
        _table(f"Revenue by Quarter — {basis}", "Quarter | Revenue", revenue_rows)
        + _table("Diluted EPS by Quarter", "Quarter | EPS", eps_rows)
        + _table("Net Margin by Quarter (levels)", "Quarter | Net margin", margin_rows)
        + f"""## What the Window Is
Every price figure above is measured over a STATED span, because the same stock \
supports very different true statements at once. From its {prices['window_start']}-to-date \
high of ${prices['peak']:,.2f} on {prices['peak_day']}, {filings['symbol']} is \
{_pct(prices['from_peak_pct'])}. Its worst peak-to-trough fall in the trailing twelve \
months is {_pct(prices['drawdown_pct'])}. Those are different measurements, not \
competing estimates, and neither is "the" decline.

Earnings figures are {basis} — not the adjusted, non-GAAP numbers a company \
headlines in its press release, which are usually different and are not carried here.
"""
    )

    return {
        "symbol": filings["symbol"],
        "quarter": quarter,
        "title": f"{filings['name']} ({filings['symbol']}) — {quarter} Results",
        "subtitle": (
            f"{basis} · prices through {prices['last_day']}"
        ),
        "highlights": highlights,
        "markdown": markdown,
        "facts": {
            "basis": basis,
            "quarter_end": filings["quarter_end"],
            "year_ago_end": filings["year_ago_end"],
            "prior_end": filings["prior_end"],
            "revenue": latest["revenue"],
            "revenue_yoy_pct": rev_yoy,
            "revenue_qoq_pct": _change_pct(latest["revenue"], filings["prior"]["revenue"]),
            "net_income": latest["net_income"],
            "eps": latest["eps"],
            "eps_yoy_pct": eps_yoy,
            "net_margin_pct": latest["margin"],
            "net_margin_year_ago_pct": year_ago["margin"],
            "last_close": prices["last_close"],
            "last_day": prices["last_day"],
            "peak": prices["peak"],
            "peak_day": prices["peak_day"],
            "price_window_start": prices["window_start"],
            "from_peak_pct": prices["from_peak_pct"],
            "drawdown_pct": prices["drawdown_pct"],
            "trough_day": prices["trough_day"],
            "year_high": prices["year_high"],
            "year_low": prices["year_low"],
            "sessions": prices["sessions"],
        },
    }
