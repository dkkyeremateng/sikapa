"""Macro data from FRED — the layer every comparable agent has and this had none of.

Rates, inflation, employment, growth and credit spreads: the context that decides
whether a single company's numbers mean anything. Without it the assistant could
describe a stock's multiple in detail and had nothing at all to say about the
discount rate that multiple is priced against.

**Keyless, like the rest of the stack.** FRED's official API requires a key
(``api.stlouisfed.org`` answers 400 without one), but the graph CSV endpoint that
backs their charts does not, and it takes the same series ids and date bounds. So
this follows SEC EDGAR and Ken French rather than Tavily: no signup, nothing to
configure, works on a fresh clone.

**Series ids are an implementation detail.** Nobody asks for ``DGS10``; they ask
about the 10-year. ``ALIASES`` maps the language people actually use onto the ids,
so the model passes ``"10y"`` or ``"unemployment"`` and gets the right series —
and an unknown name lists what is available rather than silently fetching nothing.

**One caveat, stated wherever a date is involved.** ``as_of`` bounds the
OBSERVATION date, not the vintage. FRED revises: GDP and payrolls are restated for
months afterwards, so asking for June 2025 today returns June 2025 *as currently
revised*, not the number that was on the screen in June. Point-in-time vintages
need ALFRED, which is a different (keyed) service. That distinction matters for
exactly the same reason the rest of the point-in-time work does, so the output
says it rather than leaving the reader to assume.
"""

from __future__ import annotations

from typing import Any
import csv
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

from .pointintime import AsOfError, parse_as_of, window_note

_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_UA = "Mozilla/5.0 (compatible; financial-research-assistant)"

#: Same-process cache: (series_id, start, end) -> rows. A snapshot pulls a dozen
#: series and the digest may re-ask within a turn; FRED is a courtesy endpoint and
#: hammering it for identical data would be rude as well as slow.
_CACHE: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
_CACHED_AT: dict[tuple[str, str, str], float] = {}
#: FRED revises and extends series; an always-on process refetches after this long.
_CACHE_TTL = 6 * 3600.0
_CACHE_MAX = 256


class MacroDataUnavailable(RuntimeError):
    """FRED couldn't be reached — distinct from a series that simply has no data."""


#: What people say -> the FRED series id. Grouped by what a question is usually
#: about. Keys are matched case-insensitively after stripping spaces/underscores,
#: so "fed funds", "fed_funds" and "FedFunds" all land.
ALIASES: dict[str, tuple[str, str]] = {
    # --- policy & rates ---
    "fedfunds": ("DFF", "Effective federal funds rate (%)"),
    "policyrate": ("DFF", "Effective federal funds rate (%)"),
    "3m": ("DGS3MO", "3-month Treasury yield (%)"),
    "2y": ("DGS2", "2-year Treasury yield (%)"),
    "10y": ("DGS10", "10-year Treasury yield (%)"),
    "30y": ("DGS30", "30-year Treasury yield (%)"),
    "yieldcurve": ("T10Y2Y", "10y minus 2y Treasury spread (%)"),
    "realrate": ("DFII10", "10-year TIPS (real) yield (%)"),
    "mortgage": ("MORTGAGE30US", "30-year fixed mortgage rate (%)"),
    # --- inflation ---
    "cpi": ("CPIAUCSL", "CPI, all urban consumers (index)"),
    "coreCPI".lower(): ("CPILFESL", "Core CPI, ex food & energy (index)"),
    "pce": ("PCEPI", "PCE price index"),
    "corepce": ("PCEPILFE", "Core PCE price index — the Fed's target measure"),
    "breakeven": ("T10YIE", "10-year breakeven inflation rate (%)"),
    # --- labour ---
    "unemployment": ("UNRATE", "Unemployment rate (%)"),
    "payrolls": ("PAYEMS", "Nonfarm payrolls (thousands)"),
    "claims": ("ICSA", "Initial jobless claims"),
    "participation": ("CIVPART", "Labour force participation rate (%)"),
    # --- growth & activity ---
    "gdp": ("GDPC1", "Real GDP (chained 2017 $bn)"),
    "retail": ("RSAFS", "Retail sales ($m)"),
    "industrial": ("INDPRO", "Industrial production (index)"),
    "sentiment": ("UMCSENT", "U. Michigan consumer sentiment"),
    # --- credit & risk ---
    "highyield": ("BAMLH0A0HYM2", "ICE BofA US high-yield option-adjusted spread (%)"),
    "creditspread": ("BAMLH0A0HYM2", "ICE BofA US high-yield option-adjusted spread (%)"),
    "investmentgrade": ("BAMLC0A0CM", "ICE BofA US corporate OAS (%)"),
    "vix": ("VIXCLS", "CBOE volatility index"),
    "dollar": ("DTWEXBGS", "Trade-weighted US dollar index"),
    "oil": ("DCOILWTICO", "WTI crude oil ($/bbl)"),
}

#: Lookback for each snapshot line. Comfortably over a year on purpose: every line
#: is quoted against its value a year earlier, and the monthly series (CPI, PCE,
#: unemployment) are released with a lag — so a 400-day window reached back far
#: enough for daily yields but not for CPI, which silently fell back to printing a
#: meaningless index level.
_SNAPSHOT_DAYS = 800

#: The snapshot's fixed panel: one line each, chosen to answer "what is the
#: backdrop" without becoming a data dump. Order is the order a macro note reads.
_SNAPSHOT = (
    "fedfunds", "2y", "10y", "yieldcurve",
    "cpi", "corepce", "breakeven",
    "unemployment", "claims",
    "highyield", "vix", "oil",
)

#: Series where a level is meaningless and the CHANGE is the point. An index of
#: 322.1 says nothing; "+2.7% year over year" is the number people mean by "CPI".
_AS_YOY = frozenset({
    "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PAYEMS", "GDPC1", "RSAFS",
    "INDPRO",
})


def _normalize(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def resolve(name: str) -> tuple[str, str] | None:
    """``("10-year", …)`` -> ``("DGS10", label)``. A raw FRED id passes through, so
    anything in the library is still reachable even when it has no alias."""
    key = _normalize(name)
    if key in ALIASES:
        return ALIASES[key]
    raw = (name or "").strip().upper()
    # A bare id is uppercase alphanumerics; treat it as one and let the fetch say
    # if it doesn't exist. Guessing an alias for it would be worse.
    if raw and raw.replace("_", "").isalnum() and not raw.isdigit():
        return raw, raw
    return None


def _fetch(series_id: str, start: date, end: date | None) -> list[tuple[str, float]]:
    """``[(date, value)]`` oldest-first for one series, from the keyless CSV.

    Rows with an empty value are dropped — daily series carry blank rows for
    holidays, and a blank is "no observation", not zero. An unknown id returns an
    HTML error page rather than a 404, so the CSV header is what's checked.
    """
    params = {"id": series_id, "cosd": start.isoformat()}
    if end is not None:
        params["coed"] = end.isoformat()
    key = (series_id, params["cosd"], params.get("coed", ""))
    if key in _CACHE and time.time() - _CACHED_AT.get(key, 0.0) < _CACHE_TTL:
        return _CACHE[key]
    url = f"{_CSV_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (https)
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        raise MacroDataUnavailable(
            f"Couldn't reach FRED for {series_id} ({type(exc).__name__})."
        ) from None
    if not text.lstrip().lower().startswith("observation_date"):
        return []  # unknown id: FRED serves an HTML page, not a CSV
    rows: list[tuple[str, float]] = []
    for row in csv.DictReader(io.StringIO(text)):
        stamp = (row.get("observation_date") or "").strip()
        raw = (row.get(series_id) or "").strip()
        if not stamp or not raw or raw == ".":
            continue
        try:
            rows.append((stamp, float(raw)))
        except ValueError:
            continue
    # The key carries a start date that moves every day, so without a bound an
    # always-on process kept one copy of each series per day for as long as it ran.
    if len(_CACHE) >= _CACHE_MAX:
        for stale in sorted(_CACHED_AT, key=_CACHED_AT.__getitem__)[: len(_CACHE) // 2]:
            _CACHE.pop(stale, None)
            _CACHED_AT.pop(stale, None)
    _CACHE[key] = rows
    _CACHED_AT[key] = time.time()
    return rows


def _year_ago(rows: list[tuple[str, float]]) -> float | None:
    """The observation closest to a year before the last one, or None when the
    window doesn't reach back that far.

    Anniversary-based rather than "N rows back", because series here have wildly
    different frequencies — DGS10 is daily, UNRATE monthly, GDPC1 quarterly — so a
    fixed row offset would mean a different span for each.
    """
    if len(rows) < 2:
        return None
    last = date.fromisoformat(rows[-1][0])
    try:
        target = last.replace(year=last.year - 1)
    except ValueError:  # 29 Feb has no anniversary in a common year
        target = last.replace(month=2, day=28, year=last.year - 1)
    prior = [r for r in rows if r[0] <= target.isoformat()]
    return prior[-1][1] if prior else None


def _yoy(rows: list[tuple[str, float]]) -> float | None:
    """Percent change over a year — the form an index series is actually quoted
    in. A CPI level of 332.6 says nothing; "+2.7% y/y" is what people mean."""
    base = _year_ago(rows)
    if base is None or not base:
        return None
    return (rows[-1][1] - base) / base * 100.0


def _fmt(value: float) -> str:
    return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"


#: Said wherever a date is involved. FRED revises, so a past observation comes
#: back as CURRENTLY revised — not the figure that was on the screen at the time.
_VINTAGE_NOTE = (
    "note: as_of bounds the OBSERVATION date, not the data vintage — FRED revises, "
    "so this is that period as currently restated, not what was known then."
)


def _series_rows(
    series_id: str, days: int, stamp: date | None
) -> list[tuple[str, float]]:
    end = stamp
    start = (stamp or date.today()) - timedelta(days=max(1, days))
    return _fetch(series_id, start, end)


def macro_series(series: str, days: int = 730, as_of: str = "") -> str:
    """Fetch a US macroeconomic series from FRED (keyless) and chart it with summary
    stats — latest value, change over the window, and year-over-year for index
    series where the level alone means nothing.

    ``series`` takes plain names — 'fed funds', '10y', 'cpi', 'core pce',
    'unemployment', 'claims', 'yield curve', 'breakeven', 'high yield', 'vix',
    'oil', 'gdp', 'mortgage' — or any raw FRED series id. An unknown name lists
    what's available rather than guessing.

    Use for 'what are rates doing / where is inflation / how is the labour market /
    is the yield curve inverted / what's the macro backdrop for this'. ``days`` is
    the lookback; ``as_of`` (YYYY-MM-DD) ends the window at a past date.

    For the whole picture at once use `macro_snapshot`.
    """
    try:
        stamp = parse_as_of(as_of)
    except AsOfError as exc:
        return str(exc)
    found = resolve(series)
    if found is None:
        # ALL of them, not an alphabetical slice: truncating cut off `unemployment`
        # and `vix`, which are among the most asked-for. Thirty short names is a
        # cheap tool result and a complete answer.
        return (
            f"Don't know a macro series called {series!r}. Available names: "
            + ", ".join(sorted(ALIASES))
            + ". Or pass a FRED series id directly (e.g. 'DGS10')."
        )
    series_id, label = found
    try:
        rows = _series_rows(series_id, days, stamp)
    except MacroDataUnavailable as exc:
        return f"{exc} This is a data-source issue, not a problem with {series!r}."
    if not rows:
        return (
            f"No observations for {series_id} over that window. Check the series "
            f"name/id, or widen `days`."
        )

    first_d, first_v = rows[0]
    last_d, last_v = rows[-1]
    change = last_v - first_v
    lines = [
        f"{label} · {series_id} · {first_d} → {last_d} ({len(rows)} obs)"
        f"{window_note(stamp, last_d)}",
        f"latest {_fmt(last_v)} · {first_d} {_fmt(first_v)} · change {change:+,.2f}",
    ]
    yoy = _yoy(rows) if series_id in _AS_YOY else None
    if yoy is not None:
        lines.append(f"year over year {yoy:+.1f}%")
    lines.append(
        f"high {_fmt(max(v for _, v in rows))} · low {_fmt(min(v for _, v in rows))}"
    )
    if stamp:
        lines.append(_VINTAGE_NOTE)
    lines.append("")
    from .tools import ChartText, _render_multi_series

    chart = _render_multi_series(
        f"{label} · {first_d} → {last_d}", [(series_id, [v for _, v in rows])],
        xlabel="observation",
    )
    return ChartText("\n".join(lines), chart)


def macro_snapshot(as_of: str = "") -> str:
    """One-screen view of the US macro backdrop from FRED (keyless): policy rate,
    2y/10y yields and the curve spread, CPI and core PCE (year over year),
    breakeven inflation, unemployment and jobless claims, high-yield spreads, VIX
    and oil — each with its latest value and its move over the past year.

    Use for 'what's the macro picture / backdrop / environment', and before a
    valuation or allocation view where the rate and inflation regime is the thing
    that actually moves the answer. ``as_of`` (YYYY-MM-DD) reports the panel as of
    a past date. For one series in depth use `macro_series`.
    """
    try:
        stamp = parse_as_of(as_of)
    except AsOfError as exc:
        return str(exc)
    when = stamp or date.today()
    lines = [f"US MACRO SNAPSHOT · as of {when.isoformat()} (FRED, keyless):"]
    unreachable = False
    for alias in _SNAPSHOT:
        series_id, label = ALIASES[alias]
        try:
            rows = _series_rows(series_id, _SNAPSHOT_DAYS, stamp)
        except MacroDataUnavailable:
            unreachable = True
            continue
        if not rows:
            continue
        last_d, last_v = rows[-1]
        base = _year_ago(rows)
        if series_id in _AS_YOY:
            yoy = _yoy(rows)
            value = f"{yoy:+.1f}% y/y" if yoy is not None else f"{_fmt(last_v)} (level)"
        elif base is not None:
            value = f"{_fmt(last_v)}  ({last_v - base:+,.2f} y/y)"
        else:
            value = f"{_fmt(last_v)}"
        lines.append(f"  {label:<52} {value}   [{last_d}]")
    if len(lines) == 1:
        return (
            "Couldn't reach FRED for any series just now — this is a data-source "
            "issue. Please try again shortly."
        )
    if unreachable:
        lines.append("note: some series couldn't be fetched and are omitted.")
    if stamp:
        lines.append(_VINTAGE_NOTE)
    lines.append(
        "Each line is the latest observation on or before the date shown; release "
        "lags differ by series, so they are not all the same day."
    )
    return "\n".join(lines)


def _make_tools() -> list[Any]:
    """`macro_series` charts, so it goes through the chart wrapper — the model gets
    the stats and the UI gets the plot, matching the price charts."""
    from .tools import _chart_tool

    return [_chart_tool(macro_series), macro_snapshot]


MACRO_TOOLS = _make_tools()
