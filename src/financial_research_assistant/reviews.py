"""Report templates: a short request in, a ready-to-render brief out.

Four generated portfolio reviews, each wrong in a different way, taught the same
lesson twice over. First, the method has to travel with the request — pasted into
the prompt it was one dropped newline away from vanishing, and the reply looked
finished either way. Second, and less obvious: even with the method intact, every
figure the model retyped was a chance to lose one. The last review put
``$7,410.61`` on the page directly under eleven line items summing to
``$7,412.61``.

So a template here is not prose to paste. It RESOLVES the period, FETCHES every
figure, and hands back the exact ``highlights`` block and markdown skeleton with
the numbers already in place. What is left for the model is the part it is good
at — reading the shape and saying what it means — and none of the part it is bad
at.

The period is resolved against the DATA's last session, not today. A file pulled
on the 7th and read on the 10th still means August by "this month", and asking
for dates the statement cannot reach is refused rather than quietly answered for
a different span.
"""

from __future__ import annotations

from typing import Any
import calendar
import re

#: Named windows, resolved relative to the last session in the data.
_PERIOD_ALIASES = {
    "ytd": "ytd", "year to date": "ytd", "year-to-date": "ytd", "this year": "ytd",
    "": "ytd",
    "last month": "last-month", "previous month": "last-month",
    "past month": "last-month", "the last month": "last-month",
    "this month": "this-month", "month to date": "this-month",
    "mtd": "this-month",
    "last quarter": "last-quarter", "previous quarter": "last-quarter",
    "this quarter": "this-quarter", "quarter to date": "this-quarter",
    "qtd": "this-quarter",
    "last year": "last-year", "previous year": "last-year",
    "inception": "inception", "since inception": "inception",
    "all": "inception", "all time": "inception", "everything": "inception",
}

_QUARTER_RE = re.compile(r"^q([1-4])(?:\s+(\d{4}))?$|^(\d{4})\s+q([1-4])$")
_YEAR_RE = re.compile(r"^(\d{4})$")
_LAST_N_RE = re.compile(r"^(?:last\s+|past\s+)?(\d+)\s*(day|days|d|month|months|m)$")
_EXPLICIT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*(?:\.\.|to|—|–|-)\s*(\d{4}-\d{2}-\d{2})$")


def _month_end(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _shift_month(year: int, month: int, by: int) -> tuple[int, int]:
    index = (year * 12 + month - 1) + by
    return index // 12, index % 12 + 1


def resolve_period(period: str, last_session: str, first_session: str) -> tuple[str, str, str]:
    """``(start, end, label)`` for a period phrase, anchored on the DATA.

    Anchored on ``last_session`` rather than the clock because the statement is a
    snapshot: pulled on the 7th and read on the 10th, "this month" still means the
    month the data is in. Using today would silently shift every window by the age
    of the file.
    """
    text = re.sub(r"\s+", " ", (period or "").strip().lower()).strip(" .")
    year, month = int(last_session[:4]), int(last_session[5:7])

    explicit = _EXPLICIT_RE.match(text)
    if explicit:
        return explicit.group(1), explicit.group(2), f"{explicit.group(1)} to {explicit.group(2)}"

    named = _PERIOD_ALIASES.get(text)
    if named == "ytd":
        return f"{year}-01-01", last_session, f"{year} year to date"
    if named == "this-month":
        return f"{year}-{month:02d}-01", last_session, f"{calendar.month_name[month]} {year}"
    if named == "last-month":
        y, m = _shift_month(year, month, -1)
        return f"{y}-{m:02d}-01", _month_end(y, m), f"{calendar.month_name[m]} {y}"
    if named in ("this-quarter", "last-quarter"):
        q = (month - 1) // 3 + 1
        if named == "last-quarter":
            q -= 1
            if q == 0:
                q, year = 4, year - 1
        start_month = (q - 1) * 3 + 1
        end = last_session if named == "this-quarter" else _month_end(year, start_month + 2)
        return f"{year}-{start_month:02d}-01", end, f"Q{q} {year}"
    if named == "last-year":
        return f"{year - 1}-01-01", f"{year - 1}-12-31", str(year - 1)
    if named == "inception":
        return first_session, last_session, f"{first_session} to {last_session}"

    quarter = _QUARTER_RE.match(text)
    if quarter:
        q = int(quarter.group(1) or quarter.group(4))
        qy = int(quarter.group(2) or quarter.group(3) or year)
        start_month = (q - 1) * 3 + 1
        return f"{qy}-{start_month:02d}-01", _month_end(qy, start_month + 2), f"Q{q} {qy}"

    whole_year = _YEAR_RE.match(text)
    if whole_year:
        y = int(whole_year.group(1))
        return f"{y}-01-01", f"{y}-12-31", str(y)

    span = _LAST_N_RE.match(text)
    if span:
        n, unit = int(span.group(1)), span.group(2)
        if unit.startswith("m"):
            y, m = _shift_month(year, month, -n)
            return f"{y}-{m:02d}-01", last_session, f"the last {n} months"
        from datetime import date, timedelta

        start = date.fromisoformat(last_session) - timedelta(days=n)
        return start.isoformat(), last_session, f"the last {n} days"

    raise ValueError(
        f"unrecognised period {period!r}. Try 'ytd', 'last month', 'this month', "
        f"'last quarter', 'Q2 2026', '2025', 'last 90 days', or "
        f"'2026-01-01..2026-06-30'."
    )


def _money(amount: float, signed: bool = False) -> str:
    """``-$63.03``, not ``$-63.03``. The sign belongs outside the currency mark;
    interpolating ``${x:+,.2f}`` puts it inside and reads as a typo."""
    sign = "-" if amount < 0 else ("+" if signed else "")
    return f"{sign}${abs(amount):,.2f}"


def _tile(label: str, value: str, note: str = "") -> str:
    return f"{label} | {value}" + (f" | {note}" if note else "")


def _window_income(account: str, start: str, end: str) -> dict[str, float]:
    """Dividends, withholding and fees INSIDE the window.

    Year-scoped income on a one-month review is the same error that put an
    all-dates total on a sheet as "YTD", only smaller and harder to spot: the
    figure is real, it just answers a different question than its label. Reads
    through ``query_transactions``, so it inherits the cross-import dedupe.
    """
    from . import statements

    out = {"gross_dividends": 0.0, "withholding_tax": 0.0, "fees": 0.0, "net": 0.0}
    by_symbol: dict[str, float] = {}
    for kind in ("dividend", "withholding_tax", "fee"):
        for row in statements.query_transactions(
            kind=kind, account=account or None, limit=10000
        ):
            date = row.get("date") or ""
            if not (start <= date <= end) or (row.get("currency") or "USD") != "USD":
                continue
            amount = row.get("amount") or 0.0
            field = {"dividend": "gross_dividends", "withholding_tax": "withholding_tax",
                     "fee": "fees"}[kind]
            out[field] += amount
            out["net"] += amount
            if kind == "dividend":
                symbol = statements._symbol_from_description(row.get("description") or "")
                by_symbol[symbol] = by_symbol.get(symbol, 0.0) + amount
    out = {k: round(v, 2) for k, v in out.items()}
    out["_by_symbol"] = dict(  # type: ignore[assignment]
        sorted(by_symbol.items(), key=lambda kv: kv[1], reverse=True)
    )
    return out


def build_review(period: str = "", account: str = "") -> dict[str, Any]:
    """Every figure a performance review needs, pre-formatted for ``render_report``.

    Returns ``title``/``subtitle``/``highlights``/``markdown`` ready to pass
    straight through, plus ``facts`` for anything the caller wants to say in
    prose. Nothing here needs retyping, which is the point: the figures that were
    wrong on delivered sheets were wrong in transcription, not in the source.
    """
    from . import flex, statements

    account = account or statements.default_account() or ""
    bounds = flex.period_return()  # the file's own extent, for anchoring
    start, end, label = resolve_period(period, bounds["file_end"], bounds["file_start"])
    r = flex.period_return(start=start, end=end)
    # The statement clamps the window to what it holds. Asking for "2025" against a
    # file starting in August returns August-December — a real figure under a label
    # that claims the year, which is the same mislabelling that put a trailing
    # twelve months on a sheet as year-to-date. Say so in the label itself, since
    # that is what becomes the title.
    truncated = r["start"] > start or (end and r["end"] < end)
    if truncated:
        label = f"{label} (data covers {r['start']} to {r['end']})"

    year = r["start"][:4]
    usd = _window_income(account, r["start"], r["end"])
    alloc = statements.allocation(account=account or None)
    realized = statements.realized_gains(account=account or None, year=int(year))
    # From `allocation`'s own total, which sums the raw values once. Adding the
    # DISPLAYED rows here instead compounded eleven roundings into a cent that is
    # not in the data.
    unrealized = alloc.get("unrealized_pl")
    if unrealized is None:
        unrealized = sum(p.get("unrealized_pl") or 0.0 for p in alloc["positions"])
    nav_change = r["nav_end"] - r["nav_start"]
    deposit_share = (r["deposits"] / nav_change * 100) if nav_change else 0.0

    highlights = "\n".join([
        _tile("Return", f"{r['return_pct']:+.2f}%", f"time-weighted, {r['sessions']} sessions"),
        _tile("Portfolio value", _money(r["nav_end"]), f"at {r['end']}"),
        _tile("Investment gain", _money(r["investment_gain"], signed=True),
              f"NAV {_money(nav_change, signed=True)} less "
              f"{_money(r['deposits'])} deposits"),
        _tile("Max drawdown", f"{r['max_drawdown_pct']:.2f}%", "peak to trough, this window"),
        _tile("Net dividends", _money(usd["net"]), "in this window, after withholding"),
        _tile("Unrealised P/L", _money(unrealized, signed=True),
              "since purchase, open positions"),
    ])

    # A one- or two-row table cannot chart and reads as a stub, so a short window
    # simply has no monthly section rather than a table of one month.
    months_section = ""
    if len(r["monthly_pct"]) >= 3:
        rows = "\n".join(
            f"| {calendar.month_name[int(m[5:7])]} | {v:+.2f}% |"
            for m, v in r["monthly_pct"].items()
        )
        months_section = f"## Monthly Returns\n| Month | Return |\n|---|---|\n{rows}\n\n"

    weights = "\n".join(
        f"- **{p['symbol']}:** {p['weight_pct']:.1f}% — {_money(p['value'])}"
        for p in alloc["positions"][:8]
    )
    dividends = ", ".join(
        f"{s} {_money(a)}" for s, a in list(usd["_by_symbol"].items())[:6]  # type: ignore[union-attr]
    )
    markdown = f"""{months_section}## Holdings by Weight
{weights}

## Deposits and Investment Gain
Net asset value moved {_money(r['nav_start'])} to {_money(r['nav_end'])}, a change \
of {_money(nav_change, signed=True)}. Deposits of {_money(r['deposits'])} account \
for {deposit_share:.1f}% of that, leaving an investment gain of \
{_money(r['investment_gain'], signed=True)} — this, not the balance, is what the \
portfolio earned.

## Income and Realised Gains
Dividends {_money(usd['gross_dividends'])} gross, withholding \
{_money(usd['withholding_tax'])}, fees {_money(usd['fees'])}, net \
{_money(usd['net'])} over this window. By holding: {dividends or 'none recorded'}. \
Realised gains for the {year} calendar year: {_money(realized['total_realized'])} \
across {len(realized.get('rows') or [])} closed lots.

## Concentration
Top five holdings are {alloc['top5_concentration_pct']:.1f}% of the book; the \
largest single position is {alloc['largest_weight_pct']:.1f}%. Total unrealised \
P/L across open positions is {_money(unrealized, signed=True)}, measured since \
purchase.
"""
    return {
        "period_label": label,
        "title": f"Portfolio Performance — {label}",
        "subtitle": f"{r['start']} to {r['end']} · time-weighted, deposit-independent",
        "highlights": highlights,
        "markdown": markdown,
        "facts": {
            "return_pct": r["return_pct"],
            "monthly_pct": r["monthly_pct"],
            "nav_start": r["nav_start"],
            "nav_end": r["nav_end"],
            "deposits": r["deposits"],
            "deposit_share_pct": deposit_share,
            "investment_gain": r["investment_gain"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "up_sessions": r["up_sessions"],
            "sessions": r["sessions"],
            "net_dividends": usd["net"],
            "dividends_by_symbol": usd["_by_symbol"],
            "realized": realized["total_realized"],
            "closed_lots": len(realized.get("rows") or []),
            "unrealized": unrealized,
            "top5_pct": alloc["top5_concentration_pct"],
            "reconciled": f"{r['reconciled']}/{r['checked']}",
            "whole_file_pct": r["file_return_pct"],
            "whole_file_window": f"{r['file_start']} to {r['file_end']}",
        },
    }
