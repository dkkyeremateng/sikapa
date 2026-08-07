"""Answering a question about the past with data from the past.

Every market-data tool here reports what is true NOW. Asked "was NVDA expensive
in January 2025?", the agent called `stock_fundamentals` and got today's P/E,
then wrote a confident answer about January around it. Nothing failed, nothing
warned, and the number was off by a year — the worst combination a research tool
can produce, and the one the look-ahead-bias literature exists to name.

The fix is not one function, because the sources do not have one shape:

**Price-derived tools can be exact.** Volatility, drawdown, beta, correlation and
the charts are all computed from a daily close series, and a series can simply be
cut at a date. ``as_of_series`` does that cut, and the tools that use it report
the real window they covered, so an ``as_of`` on a market holiday reads back as
the last session on or before it rather than silently shifting.

**Snapshot tools cannot, and must say so.** Yahoo's `.info` is a single current
record: there is no January 2025 P/E in it to return, and no amount of parameter
plumbing creates one. Those tools take an ``as_of`` anyway — not to honour it, but
so the model has a way to *express* the intent and get told no. ``unsupported``
writes that refusal, and it names the tool that CAN answer (the SEC tools are
point-in-time by construction: a 10-K is filed once and never restated in place).

That asymmetry is the whole design. A tool that quietly ignores ``as_of`` would be
worse than one that never had it, because the model would believe it worked.
"""

from __future__ import annotations

from datetime import date, timedelta

#: How far back a request may reach. Not a data limit — it is the point past
#: which a keyless daily-close series stops being a fair record of what a
#: position was worth (splits and delistings accumulate, and Yahoo's adjustments
#: are applied retroactively rather than as-of).
_MAX_YEARS_BACK = 30


class AsOfError(ValueError):
    """An ``as_of`` that cannot be honoured — malformed, or in the future."""


def parse_as_of(raw: str | None) -> date | None:
    """``"2025-01-31"`` -> a date; ``""``/None -> None (meaning "now").

    Raises ``AsOfError`` rather than falling back to today. Falling back is the
    exact bug this module exists to prevent: the caller asked for a past date,
    and silently answering about the present is how a wrong figure gets written
    into a report with no trace.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError:
        raise AsOfError(
            f"Couldn't read {raw!r} as a date — use YYYY-MM-DD, e.g. '2025-01-31'."
        ) from None
    today = date.today()
    if parsed > today:
        raise AsOfError(
            f"as_of {parsed.isoformat()} is in the future; there is no data for it. "
            f"Omit as_of for the latest available figures."
        )
    if parsed < today.replace(year=today.year - _MAX_YEARS_BACK):
        raise AsOfError(
            f"as_of {parsed.isoformat()} is further back than {_MAX_YEARS_BACK} "
            f"years, where the keyless daily-close history stops being reliable."
        )
    return parsed


def lookback_days(days: int, as_of: date | None) -> int:
    """How many days of history to FETCH so ``days`` sessions still remain after
    the series is cut at ``as_of``.

    The sources return a window ending today, so asking for 180 days of history
    as of two years ago returns a window that ends 730 days after the point of
    interest — and cutting it leaves nothing. Widening by the gap first is what
    makes the cut land on real data.
    """
    if as_of is None:
        return days
    gap = (date.today() - as_of).days
    return max(days, days + max(0, gap))


def as_of_series(
    series: list[tuple[str, float]], as_of: date | None
) -> list[tuple[str, float]]:
    """``series`` cut to rows dated on or before ``as_of``.

    ``on or before``, not ``on``: an ``as_of`` landing on a weekend, a holiday or
    a halt has no row of its own, and the honest answer is the last session that
    did trade — which the caller then reports, so the shift is visible rather
    than assumed.
    """
    if as_of is None:
        return series
    cutoff = as_of.isoformat()
    return [row for row in series if row[0] <= cutoff]


def window_note(as_of: date | None, last_date: str = "") -> str:
    """The trailing phrase that says which world the figures come from.

    Always non-empty for a point-in-time answer. The figures look identical
    either way, so if the output doesn't say, nothing does.
    """
    if as_of is None:
        return ""
    stamp = as_of.isoformat()
    if last_date and last_date != stamp:
        return (
            f" · AS OF {stamp} (last session on or before it: {last_date}); "
            f"later data deliberately excluded"
        )
    return f" · AS OF {stamp}; later data deliberately excluded"


def unsupported(tool: str, as_of: date | None, instead: str) -> str | None:
    """The refusal a snapshot tool returns when asked for a past date, or None
    when there is no ``as_of`` and the tool should just run.

    Written as a redirect rather than an error: the model asked a reasonable
    question of the wrong source, and the useful reply names the right one.
    """
    if as_of is None:
        return None
    return (
        f"`{tool}` cannot answer as of {as_of.isoformat()}. It reads a CURRENT "
        f"snapshot from Yahoo, which keeps no history — there is no {as_of.year} "
        f"version of these figures to return, and reporting today's as though they "
        f"were {as_of.isoformat()}'s is the error this refusal exists to prevent.\n"
        f"For a point-in-time answer use: {instead}\n"
        f"Tell the user which figures are as-of and which are current, rather than "
        f"presenting a mix as one."
    )


#: Where to send each snapshot tool's caller. SEC XBRL is point-in-time by
#: construction — a 10-K covers a fiscal year and is not restated in place — and
#: the price series can be cut to any date, so between them most retrospective
#: questions have a real answer.
REDIRECTS = {
    "stock_fundamentals": (
        "`sec_financials` / `sec_quarterly_financials` for as-reported figures "
        "from the filing covering that date, and `price_history_chart(as_of=…)` "
        "for the price then"
    ),
    "compare_stocks": (
        "`compare_sec_financials` for an as-reported companies×metrics matrix "
        "from the filings covering that date"
    ),
    "analyst_ratings": (
        "`sec_filing_excerpt` for what the company itself disclosed then, or "
        "`web_search` for contemporaneous coverage. Ratings history is not "
        "available from this source"
    ),
    "etf_exposure": (
        "nothing keyless — fund holdings are published as a current basket only. "
        "Say the historical composition isn't available rather than implying "
        "today's is it"
    ),
    "dcf_valuation": (
        "`sec_financials` for the cash-flow history that was on file then. The "
        "model's net debt, share count and price come from a current snapshot, so "
        "a dated DCF would mix two eras"
    ),
    "screen_stocks": (
        "nothing keyless — the screen reads current fundamentals, so it cannot "
        "reconstruct which names would have passed on a past date. Say so; a "
        "screen run today is not a backtest"
    ),
}


def snapshot_guard(tool: str, raw_as_of: str | None) -> str | None:
    """One call for a snapshot tool: parse ``as_of``, and return the refusal text
    when one was given. ``None`` means "no as_of, carry on".

    Returns the message rather than raising so a tool stays a pure str-returning
    function — the model reads the refusal as the tool result and re-routes.
    """
    try:
        as_of = parse_as_of(raw_as_of)
    except AsOfError as exc:
        return str(exc)
    return unsupported(tool, as_of, REDIRECTS.get(tool, "a point-in-time source"))


#: The `as_of` argument description, shared so every tool documents it the same
#: way in its schema.
AS_OF_DOC = (
    "``as_of`` (YYYY-MM-DD) answers as of that date, excluding everything after "
    "it; omit it for the latest available."
)
