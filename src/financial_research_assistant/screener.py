"""Multi-criteria stock screener — evaluate a candidate universe against
quantitative filters and return the passers with the measured figures.

There is no bulk market-data feed here: every source is per-ticker (yfinance
``.info`` for market cap/sector, keyless Yahoo daily closes for price history and
all-time highs, and the earnings-surprise history). So this screens a *candidate
universe* — the tickers the caller supplies, or a built-in large-cap default —
rather than the whole market. For each candidate it measures the criteria it can
evaluate offline:

- **market cap** bounds (from ``.info['marketCap']``);
- **proximity to a high** — closest the price came, within a recent window, to its
  high over a lookback (pass a large ``high_lookback_days`` for a true all-time
  high) — optionally requiring that the near-high day coincided with a **down
  market** (relative strength), cross-referenced against a market index's return
  that day;
- an **earnings-beat streak** — consecutive most-recent quarters whose reported
  EPS exceeded the analyst estimate;
- a **sector** substring match.

Criteria the data can't support — forward-guidance beats, management commentary,
anything qualitative — are NOT silently treated as met: the tool returns the
quantitative passers and tells the model to finish those checks per name via
``web_search`` / ``research_report``.

Every network access goes through the same mockable ``_fetch_*`` helpers the other
tools use, so the whole screen is exercised offline in tests, and the per-symbol
evaluations run concurrently in a thread pool since they are network-bound.
"""

from __future__ import annotations

from typing import Any
import csv
import io
import urllib.request

# A curated large-cap US default universe (spread across sectors) used when the
# caller doesn't supply their own candidate list and doesn't name a universe. A
# fixed, readable fallback that needs no network — the caller can always pass an
# explicit `symbols` list (e.g. an ETF's holdings from `etf_exposure`, an IBKR
# theme, or their own watchlist), or name a broader universe (see `universe`).
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK-B", "BLK",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR",
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "PG", "KO", "PEP", "CL",
    "XOM", "CVX", "COP",
    "CAT", "BA", "GE", "HON", "UPS", "DE",
    "DIS", "NFLX", "CMCSA", "T", "VZ",
]

# yfinance `.info` round-trips are slow; cap the pool so a screen over the whole
# default universe stays parallel without hammering Yahoo.
_MAX_WORKERS = 8

# Keyless S&P 500 constituents (the datahub datasets mirror — a plain CSV, so no
# HTML scraping or API key). Cached per run; a fetch failure degrades to the
# built-in fallback with a note rather than aborting.
_SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
_UA = "Mozilla/5.0 (compatible; financial-research-assistant)"
_UNIVERSE_CACHE: dict[str, list[str]] = {}
_LARGECAP_LABEL = "built-in large-cap universe"


def _fetch_sp500() -> list[str]:
    """The current S&P 500 tickers from the keyless datahub CSV (ticker in the
    first column), normalized to Yahoo's dash form (``BRK.B`` -> ``BRK-B``) and
    deduped. Cached per run; ``[]`` on any network/parse failure."""
    if "sp500" in _UNIVERSE_CACHE:
        return _UNIVERSE_CACHE["sp500"]
    try:
        req = urllib.request.Request(_SP500_URL, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20.0) as resp:  # noqa: S310 (https)
            text = resp.read().decode("utf-8", "replace")
        rows = list(csv.reader(io.StringIO(text)))
        syms = [
            r[0].strip().upper().replace(".", "-")
            for r in rows[1:] if r and r[0].strip()
        ]
        syms = list(dict.fromkeys(syms))
    except Exception:  # noqa: BLE001 — network/parse failure degrades to fallback
        syms = []
    if syms:
        _UNIVERSE_CACHE["sp500"] = syms
    return syms


# Named universes the caller can request via ``universe=``. Each maps a set of
# normalized aliases to a (label, resolver) — resolvers are thunks so the S&P 500
# fetch only runs when actually asked for.
def _norm_universe_name(name: str) -> str:
    """Fold a universe name to a lookup key: lowercase, drop spaces/&/-/_/. so
    'S&P 500', 'sp-500', 's and p 500' all match 'sp500'."""
    return (
        (name or "").lower()
        .replace("&", "").replace("and", "").replace(" ", "")
        .replace("-", "").replace("_", "").replace(".", "")
    )


_NAMED_UNIVERSES = {
    "sp500": ("S&P 500", _fetch_sp500),
    "sandp500": ("S&P 500", _fetch_sp500),
    "largecap": (_LARGECAP_LABEL, lambda: list(_DEFAULT_UNIVERSE)),
    "default": (_LARGECAP_LABEL, lambda: list(_DEFAULT_UNIVERSE)),
}


def _resolve_universe(name: str):
    """Look up a named universe. Returns ``(symbols, label)`` — where ``symbols``
    is ``[]`` if the name is known but its source couldn't be fetched — or
    ``None`` when the name isn't recognized at all."""
    key = _norm_universe_name(name)
    entry = _NAMED_UNIVERSES.get(key)
    if entry is None:
        return None
    label, resolver = entry
    return resolver(), label


def _select_universe(symbols: str, universe: str):
    """Pick the candidate list. An explicit ``symbols`` list wins; else a named
    ``universe`` (S&P 500, large-cap); else the built-in large-cap default.
    Returns ``(symbols_list, source_label, error_message)`` — ``error_message``
    is set (and the list empty) when a named universe is unknown or unfetchable."""
    listed = [s.strip().upper() for s in symbols.replace(",", " ").split() if s.strip()]
    if listed:
        return list(dict.fromkeys(listed)), "your list", None
    name = (universe or "").strip()
    if not name:
        return list(_DEFAULT_UNIVERSE), _LARGECAP_LABEL, None
    resolved = _resolve_universe(name)
    if resolved is None:
        return [], "", (
            f"Unknown universe {name!r}. Supported: 'sp500' (S&P 500) or "
            f"'largecap' (built-in default). Or pass an explicit `symbols` list."
        )
    syms, label = resolved
    if not syms:
        return [], label, (
            f"Couldn't fetch the {label} constituents right now (network issue). "
            f"Try again, pass an explicit `symbols` list, or use universe='largecap'."
        )
    return list(dict.fromkeys(syms)), label, None


def _pct_return_by_date(symbol: str, days: int) -> dict[str, float]:
    """Map each date to the day's percent close-to-close return for ``symbol``,
    over roughly ``days`` sessions — used to test the 'market was down that day'
    condition against the near-high days. ``{}`` if history is unavailable."""
    from .tools import _fetch_daily

    series = _fetch_daily(symbol, max(5, days))
    out: dict[str, float] = {}
    for i in range(1, len(series)):
        (_d0, prev), (d1, cur) = series[i - 1], series[i]
        if prev:
            out[d1] = (cur - prev) / prev * 100.0
    return out


def _near_high(
    symbol: str,
    high_lookback_days: int,
    within_days: int,
    within_pct: float,
    market_returns: dict[str, float] | None,
    market_down_pct: float,
) -> dict[str, Any] | None:
    """Evaluate the proximity-to-high criterion for one ``symbol``.

    Finds the highest close over ``high_lookback_days`` (pass a large value for a
    true all-time high), then scans the last ``within_days`` sessions for the day
    the close came closest to that high. When ``market_down_pct`` > 0, a near-high
    day only qualifies if the market index fell at least that percent that day
    (relative strength). Returns ``{high, last, dist_pct, date, market_ret}`` for
    the best qualifying day, or None if no session in the window came within
    ``within_pct`` of the high (or none did so on a down-market day)."""
    from .tools import _fetch_daily

    series = _fetch_daily(symbol, max(30, high_lookback_days))
    if len(series) < 2:
        return None
    closes = [c for _d, c in series]
    high = max(closes)
    if high <= 0:
        return None
    window = series[-max(1, within_days):]
    best: dict[str, Any] | None = None
    for d, close in window:
        dist = (high - close) / high * 100.0  # 0 = at the high; smaller = closer
        if dist > within_pct:
            continue
        mret = market_returns.get(d) if market_returns is not None else None
        if market_down_pct > 0.0:
            # Require the market to have fallen at least market_down_pct that day.
            if mret is None or mret > -market_down_pct:
                continue
        if best is None or dist < best["dist_pct"]:
            best = {
                "high": high, "last": close, "dist_pct": dist,
                "date": d, "market_ret": mret,
            }
    return best


def _beat_streak(symbol: str, need: int) -> int:
    """Count consecutive most-recent REPORTED quarters whose EPS beat the analyst
    estimate. Rows without a reported figure (upcoming quarters) are skipped; the
    streak stops at the first non-beat. Fetches a couple extra quarters beyond
    ``need`` so a full streak can be confirmed."""
    from .fundamentals import _fetch_earnings_history, _num

    history = _fetch_earnings_history(symbol, limit=max(need + 2, 4))
    streak = 0
    for row in history:  # newest first
        reported, estimate = _num(row.get("reported")), _num(row.get("estimate"))
        if reported is None or estimate is None:
            continue  # not yet reported — skip, don't break the streak
        if reported > estimate:
            streak += 1
        else:
            break
    return streak


def _evaluate(sym: str, params: dict[str, Any], market_returns: dict[str, float] | None) -> dict[str, Any] | None:
    """Measure one candidate against every active criterion. Returns a result dict
    (with the measured figures and a per-criterion pass flag) or None if the
    ticker has no usable data at all. Actual filtering happens in ``screen_stocks``
    so failures can be counted."""
    from .fundamentals import _fetch_info, _num, _price

    info = _fetch_info(sym)
    name = info.get("longName") or info.get("shortName")
    mcap = _num(info.get("marketCap"))
    sector = info.get("sector") or ""
    # No name and no market cap means Yahoo has nothing for this ticker.
    if not name and mcap is None:
        return None

    res: dict[str, Any] = {
        "symbol": sym, "name": name or sym, "market_cap": mcap,
        "sector": sector, "price": _price(info),
        "high": None, "dist_pct": None, "near_high_date": None,
        "market_ret": None, "beats": None, "passes": True, "fail": [],
    }

    def fail(reason: str) -> None:
        res["passes"] = False
        res["fail"].append(reason)

    # Market cap bounds (billions -> dollars).
    if params["min_cap"] > 0 or params["max_cap"] > 0:
        if mcap is None:
            fail("no market cap")
        else:
            if params["min_cap"] > 0 and mcap < params["min_cap"]:
                fail("cap below min")
            if params["max_cap"] > 0 and mcap > params["max_cap"]:
                fail("cap above max")

    # Sector substring match (case-insensitive).
    if params["sector"]:
        if params["sector"].lower() not in sector.lower():
            fail("sector mismatch")

    # Proximity to high (optionally on a down-market day).
    if params["near_high_pct"] > 0:
        nh = _near_high(
            sym, params["high_lookback_days"], params["within_days"],
            params["near_high_pct"], market_returns, params["market_down_pct"],
        )
        if nh is None:
            fail("not near high in window")
        else:
            res.update(high=nh["high"], dist_pct=nh["dist_pct"],
                       near_high_date=nh["date"], market_ret=nh["market_ret"])

    # Earnings-beat streak.
    if params["min_beats"] > 0:
        streak = _beat_streak(sym, params["min_beats"])
        res["beats"] = streak
        if streak < params["min_beats"]:
            fail(f"only {streak} beat(s)")

    return res


def _parse_params(
    min_market_cap_b: float, max_market_cap_b: float, near_high_pct: float,
    near_high_within_days: int, high_lookback_days: int, market_down_pct: float,
    min_earnings_beats: int, sector: str,
) -> dict[str, Any]:
    """Coerce and clamp the raw criteria arguments into the internal params dict
    (market caps to dollars, negatives to 0, ints floored at sane minimums)."""
    return {
        "min_cap": max(0.0, float(min_market_cap_b or 0.0)) * 1e9,
        "max_cap": max(0.0, float(max_market_cap_b or 0.0)) * 1e9,
        "near_high_pct": max(0.0, float(near_high_pct or 0.0)),
        "within_days": max(1, int(near_high_within_days or 10)),
        "high_lookback_days": max(30, int(high_lookback_days or 1825)),
        "market_down_pct": max(0.0, float(market_down_pct or 0.0)),
        "min_beats": max(0, int(min_earnings_beats or 0)),
        "sector": (sector or "").strip(),
    }


def _no_criteria(params: dict[str, Any]) -> bool:
    return (params["min_cap"] == 0 and params["max_cap"] == 0
            and params["near_high_pct"] == 0 and params["min_beats"] == 0
            and not params["sector"])


def _criteria_summary(params: dict[str, Any], market_symbol: str) -> str:
    """One-line, human-readable description of the active criteria for the header."""
    crit: list[str] = []
    if params["min_cap"] > 0:
        crit.append(f"cap ≥ ${params['min_cap'] / 1e9:g}B")
    if params["max_cap"] > 0:
        crit.append(f"cap ≤ ${params['max_cap'] / 1e9:g}B")
    if params["near_high_pct"] > 0:
        c = (f"within {params['near_high_pct']:g}% of the high "
             f"(over ~{params['high_lookback_days']}d) in the last "
             f"{params['within_days']} sessions")
        if params["market_down_pct"] > 0:
            c += f", on a day {market_symbol.upper()} fell ≥{params['market_down_pct']:g}%"
        crit.append(c)
    if params["min_beats"] > 0:
        crit.append(f"≥{params['min_beats']} consecutive EPS beats")
    if params["sector"]:
        crit.append(f"sector contains '{params['sector']}'")
    return " · ".join(crit)


def _format_row(r: dict[str, Any]) -> str:
    """Render one passing candidate as a table line with only its measured legs."""
    from .fundamentals import _money

    parts = [f"{r['symbol']:<6} {_money(r['market_cap']):>8}"]
    if r["dist_pct"] is not None:
        md = f", mkt {r['market_ret']:+.2f}%" if r["market_ret"] is not None else ""
        parts.append(f"{r['dist_pct']:.2f}% from high on {r['near_high_date']}{md}")
    if r["beats"] is not None:
        parts.append(f"{r['beats']} EPS beat(s)")
    if r["sector"]:
        parts.append(r["sector"])
    return "  " + " · ".join(parts) + f"  {r['name']}"


def screen_stocks(
    symbols: str = "",
    universe: str = "",
    min_market_cap_b: float = 0.0,
    max_market_cap_b: float = 0.0,
    near_high_pct: float = 0.0,
    near_high_within_days: int = 10,
    high_lookback_days: int = 1825,
    market_down_pct: float = 0.0,
    market_symbol: str = "SPY",
    min_earnings_beats: int = 0,
    sector: str = "",
    max_symbols: int = 40,
    as_of: str = "",
) -> str:
    """Screen stocks against quantitative criteria and return the ones that pass,
    with the measured figures. Screens a CANDIDATE UNIVERSE, not the whole market.

    Pick the universe (first match wins):
    - ``symbols`` — an explicit comma/space-separated ticker list (e.g. an ETF's
      holdings from `etf_exposure`, a theme, or a watchlist); OR
    - ``universe`` — a named set: ``"sp500"`` (the current S&P 500, fetched
      keyless) or ``"largecap"`` (the built-in large-cap default); OR
    - neither — defaults to the built-in large-cap universe.

    Active criteria (a value of 0 / "" turns a criterion OFF, so only the filters
    you set are applied):
    - ``min_market_cap_b`` / ``max_market_cap_b`` — market-cap bounds in USD
      BILLIONS (e.g. 10 = only companies above $10B).
    - ``near_high_pct`` — keep only stocks whose price came within this percent of
      its high (e.g. 1.0 = within 1% of the high). The high is the max close over
      ``high_lookback_days`` (default ~5y; pass a large value like 9000 for a true
      all-time high). ``near_high_within_days`` restricts WHEN that near-high touch
      happened to the last N trading sessions (default 10 ≈ two weeks).
    - ``market_down_pct`` — additionally require the near-high day to be a day the
      market fell at least this percent (e.g. 0.5 = market down ≥0.5%), i.e. the
      stock showed relative strength; cross-referenced against ``market_symbol``
      (default SPY). Only meaningful together with ``near_high_pct``.
    - ``min_earnings_beats`` — require this many consecutive most-recent reported
      quarters to have beaten the analyst EPS estimate (e.g. 2).
    - ``sector`` — case-insensitive substring the company's sector must contain.
    - ``max_symbols`` — cap on how many candidates are evaluated (bounds runtime;
      the universe is truncated with a note if larger). Screening a big universe
      like the S&P 500 is slow (one Yahoo lookup per name) — raise this
      deliberately (e.g. 500) when you want the whole set.

    Returns a ranked plain-text table of the passers (symbol, name, market cap,
    distance-to-high and its date, market move that day, beat streak, sector),
    plus a count of how many were screened/passed. Data is delayed Yahoo data —
    verify before acting; this is not investment advice.

    IMPORTANT: this evaluates only the QUANTITATIVE criteria above. Qualitative
    criteria — forward-guidance beats, raised guidance, management commentary — are
    NOT screened here; the tool reminds you to confirm those per passing name with
    `web_search` / `earnings_calendar` / `research_report` before concluding.

    A screen runs against TODAY's fundamentals and today's index membership, so it
    cannot reconstruct which names would have passed on a past date. Pass ``as_of``
    (YYYY-MM-DD) and it declines instead of presenting a current screen as a
    historical one — a screen run today is not a backtest.
    """
    from .pointintime import snapshot_guard

    refusal = snapshot_guard("screen_stocks", as_of)
    if refusal:
        return refusal
    from concurrent.futures import ThreadPoolExecutor

    candidates, src, err = _select_universe(symbols, universe)
    if err:
        return err
    cap = max(1, int(max_symbols or 40))
    truncated = len(candidates) > cap
    candidates = candidates[:cap]

    params = _parse_params(
        min_market_cap_b, max_market_cap_b, near_high_pct, near_high_within_days,
        high_lookback_days, market_down_pct, min_earnings_beats, sector,
    )
    if _no_criteria(params):
        return (
            "No screening criteria set. Set at least one: min_market_cap_b, "
            "near_high_pct (with near_high_within_days / high_lookback_days, and "
            "optionally market_down_pct), min_earnings_beats, or sector. Pass a "
            "`symbols` list or `universe` (e.g. 'sp500') to choose what to screen."
        )

    # The market-return map is shared across all candidates (fetched once) and
    # only needed for the down-market condition.
    market_returns = None
    if params["near_high_pct"] > 0 and params["market_down_pct"] > 0:
        market_returns = _pct_return_by_date(
            market_symbol.strip().upper() or "SPY", params["within_days"] + 5
        )

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        evaluated = list(pool.map(lambda s: _evaluate(s, params, market_returns), candidates))

    results = [r for r in evaluated if r is not None]
    passers = [r for r in results if r["passes"]]
    no_data = len(candidates) - len(results)

    # Rank: closest to the high first when that criterion is active; otherwise by
    # market cap (largest first).
    if params["near_high_pct"] > 0:
        passers.sort(key=lambda r: (r["dist_pct"] if r["dist_pct"] is not None else 1e9))
    else:
        passers.sort(key=lambda r: (r["market_cap"] or 0.0), reverse=True)

    lines = [
        f"STOCK SCREEN · {len(candidates)} candidate(s) from the {src} · "
        f"{len(passers)} passed",
        "criteria: " + _criteria_summary(params, market_symbol),
    ]
    if truncated:
        lines.append(
            f"note: universe truncated to {cap} (max_symbols) — raise max_symbols "
            f"or pass a shorter `symbols` list to screen the rest."
        )
    lines.append("")

    if passers:
        lines.extend(_format_row(r) for r in passers)
    else:
        lines.append(
            "No stocks in this universe met all the criteria. Loosen a threshold, "
            "widen the universe, or check fewer criteria at once."
        )

    if no_data:
        lines.append("")
        lines.append(f"note: {no_data} ticker(s) had no usable Yahoo data and were skipped.")

    lines.append("")
    lines.append(
        "QUALITATIVE criteria are NOT screened above. If you need forward-guidance "
        "beats, raised guidance, or management commentary, confirm those per "
        "passing name with `earnings_calendar`, `web_search`, or `research_report` "
        "before concluding — an EPS beat is not the same as a guidance beat."
    )
    lines.append("(Delayed Yahoo data — verify before acting; not investment advice.)")
    return "\n".join(lines)


SCREENER_TOOLS = [screen_stocks]
