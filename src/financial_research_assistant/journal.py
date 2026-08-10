"""The thesis journal — what the assistant said would happen, and what did.

Three self-learning layers already exist and none of them learn this. Fact memory
(``memory.py``) learns about the USER. Reflection (``reflection.py``) learns about
the WORK — "that source came back thin, check it earlier next time". The eval loop
(``improve.py``) learns from a fixed regression set. Every one of them can be fully
satisfied while the agent's actual market calls are consistently wrong, because
nothing ever checks them.

So: when the agent takes a directional view — a bull/bear verdict, a DCF that says
undervalued, a "this looks cheap" — it records the call here with the price at the
time and a horizon. Later, a runner tick fetches the price then and now and scores
it. That is the whole loop.

Four decisions worth stating, because each is a trap avoided:

**The price is captured here, not reported by the model.** A call scored against a
price the model typed is scored against the model's memory of a price, which is
the thing least worth trusting. ``record_thesis`` fetches it.

**Scoring is model-free.** It is two price lookups and a subtraction, so it runs on
an ordinary scheduler tick at no cost and cannot itself hallucinate. A tick that
finds nothing due does no work at all.

**The benchmark travels with the score.** "Bullish, +8%" is not a good call if the
index did +14% over the same window. Both numbers are always reported; the verdict
is judged on direction (that is what was actually claimed) and the benchmark tells
you whether direction was worth anything.

**A track record is not a forecast.** This measures calibration on a handful of
past calls — a sample far too small to predict the next one, and drawn from
whatever the user happened to ask about. Every surface that reports it says so.
The point is to stop the agent from repeating a confident line it has already been
wrong about, not to market a hit rate.
"""

from __future__ import annotations

from typing import Any
import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .storage import write_private

#: The directional views a call can take. Deliberately not buy/sell: this
#: assistant cannot trade, and recording "buy" would document a recommendation it
#: is not making. A view on direction is what the research actually supports.
VERDICTS = ("bullish", "bearish", "neutral")

#: How far a "neutral" call may drift and still count as neutral. Wider than it
#: looks: a single name moving under 5% over a quarter genuinely is the flat case,
#: and a tighter band would score neutral calls as misses almost always, making
#: the verdict useless rather than calibrated.
_NEUTRAL_BAND_PCT = 5.0

#: Default horizon. A quarter — long enough that a call is about the thesis rather
#: than the week's noise, short enough to close the loop while the reasoning is
#: still worth revisiting.
DEFAULT_HORIZON_DAYS = 90
_MIN_HORIZON_DAYS = 7
_MAX_HORIZON_DAYS = 1095

#: What the score is measured against when the caller names nothing else.
DEFAULT_BENCHMARK = "SPY"

#: Below this many scored calls, a hit rate is noise. Reported anyway (hiding it
#: would be its own distortion) but always labelled as too small to read.
_MIN_FOR_RATE = 5


def journal_file() -> Path:
    raw = os.environ.get("FINANCIAL_RESEARCH_JOURNAL_FILE")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "journal.json"


@contextmanager
def _locked():
    """Serialize read-modify-write across processes — same trade as ``tasks.py``:
    a scheduler tick scoring calls and an interactive turn recording one can
    interleave, and without this one silently drops the other's write."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    path = journal_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_entries() -> list[dict[str, Any]]:
    """Every recorded call. A missing or corrupt file reads as "none" rather than
    raising — a bad hand-edit must not take down the tick that would have scored
    the others."""
    path = journal_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and e.get("id") and e.get("symbol")]


def save_entries(items: list[dict[str, Any]]) -> None:
    """Write the store ``0600``, atomically — it names tickers the user is acting
    on and the reasoning behind it."""
    write_private(journal_file(), json.dumps(items, indent=2) + "\n", prefix=".journal-")


def _next_id(items: list[dict[str, Any]]) -> str:
    """``t1``, ``t2``, … — matching the ``a1``/``s1`` scheme alerts and tasks use."""
    n = 0
    for e in items:
        eid = str(e.get("id", ""))
        if eid.startswith("t") and eid[1:].isdigit():
            n = max(n, int(eid[1:]))
    return f"t{n + 1}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- recording ------------------------------------------------------------------


def _spot(symbol: str) -> tuple[float | None, str]:
    """``(close, date)`` for the latest session — the price the call is anchored
    to. Returns ``(None, "")`` when the source has nothing, which the caller turns
    into a refusal: a thesis with no entry price can never be scored, so recording
    one would just accumulate junk that looks like a track record."""
    from .tools import _fetch_daily

    series = _fetch_daily(symbol, 10)
    if not series:
        return None, ""
    return series[-1][1], series[-1][0]


def _price_on(symbol: str, when: date) -> tuple[float | None, str]:
    """``(close, date)`` for the last session on or before ``when`` — the same
    on-or-before rule the point-in-time tools use, so a horizon landing on a
    weekend scores against the Friday rather than not at all."""
    from .pointintime import as_of_series
    from .tools import _fetch_daily

    series = as_of_series(_fetch_daily(symbol, 30, as_of=when), when)
    if not series:
        return None, ""
    return series[-1][1], series[-1][0]


def record_thesis(
    symbol: str,
    verdict: str,
    thesis: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    benchmark: str = DEFAULT_BENCHMARK,
) -> str:
    """Record a directional call so it can be scored against what actually happened.

    Call this whenever you commit to a VIEW on a ticker — a bull/bear verdict, a
    DCF that concludes over- or undervalued, "this looks cheap/expensive", a
    recommendation to watch or avoid. Not for neutral summaries of data: record a
    call only when you have actually taken a side, because an unscoreable entry
    dilutes the record.

    ``verdict`` is bullish, bearish, or neutral (expecting no material move).
    ``thesis`` is one or two sentences on WHY — it is what you read back later when
    the call is scored, so make it the reasoning, not the conclusion.
    ``horizon_days`` is when to judge it (default 90).

    The entry price is captured here from market data, not from you. Later a
    background tick compares it against the price at the horizon and against
    ``benchmark``, and the result becomes a lesson you see next time this ticker
    comes up. Review with `review_theses`.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return "Which ticker? record_thesis needs a symbol."
    view = (verdict or "").strip().lower()
    if view not in VERDICTS:
        return f"verdict must be one of: {', '.join(VERDICTS)} (got {verdict!r})."
    if not (thesis or "").strip():
        return (
            "A call needs its reasoning — pass `thesis` with a sentence or two on "
            "why, since that is what gets read back when the call is scored."
        )
    horizon = max(_MIN_HORIZON_DAYS, min(int(horizon_days or DEFAULT_HORIZON_DAYS),
                                          _MAX_HORIZON_DAYS))
    price, priced_on = _spot(sym)
    if price is None:
        return (
            f"No price data for {sym!r}, so this call could never be scored — not "
            f"recording it. Check the ticker."
        )
    due = (_now() + timedelta(days=horizon)).date()
    with _locked():
        items = load_entries()
        entry = {
            "id": _next_id(items),
            "symbol": sym,
            "verdict": view,
            "thesis": thesis.strip(),
            "opened": _now().isoformat(),
            "entry_price": price,
            "entry_date": priced_on,
            "horizon_days": horizon,
            "due": due.isoformat(),
            "benchmark": (benchmark or DEFAULT_BENCHMARK).strip().upper(),
            "status": "open",
        }
        items.append(entry)
        save_entries(items)
    return (
        f"Recorded [{entry['id']}] {view.upper()} {sym} at {price:,.2f} "
        f"(close {priced_on}), to be scored on {due:%Y-%m-%d} ({horizon}d) against "
        f"{entry['benchmark']}. Say that you have logged the call and will be "
        f"held to it; do not present this as a recommendation."
    )


# --- scoring --------------------------------------------------------------------


def _hit(verdict: str, change_pct: float) -> bool:
    """Whether the call went the way it said. Judged on DIRECTION because direction
    is what was claimed — the benchmark comparison sits alongside it and answers
    the separate question of whether being right was worth anything."""
    if verdict == "bullish":
        return change_pct > 0
    if verdict == "bearish":
        return change_pct < 0
    return abs(change_pct) < _NEUTRAL_BAND_PCT


def _due_date(entry: dict[str, Any]) -> date | None:
    """The date a call was to be judged on, or None when it can't be read."""
    try:
        return date.fromisoformat(str(entry.get("due"))[:10])
    except ValueError:
        return None


def due_entries(now: date | None = None) -> list[dict[str, Any]]:
    """Open calls whose horizon has passed."""
    today = now or _now().date()
    out = []
    for e in load_entries():
        if e.get("status") != "open":
            continue
        due = _due_date(e)
        # An unreadable due date leaves the call open rather than scoring junk.
        if due is not None and due <= today:
            out.append(e)
    return out


def score_entry(entry: dict[str, Any], now: date | None = None) -> dict[str, Any] | None:
    """Score one call. Returns the updated entry, or None when it can't be scored
    yet (no price at the horizon) — which leaves it open to retry on a later tick
    rather than burning it as a miss for a data outage."""
    today = now or _now().date()
    sym = str(entry.get("symbol"))
    entry_price = float(entry.get("entry_price") or 0)
    if entry_price <= 0:
        return None
    # Score at the HORIZON, not at whenever the tick happened to run. Scoring is a
    # background job: the machine sleeps, the scheduler misses a day, a horizon
    # falls over a holiday. Pricing at `today` would then judge a 90-day call over
    # 104 days — and the result is written into memory as a lesson stamped "90d",
    # so the mismatch is permanent and invisible from the record itself.
    at = min(today, _due_date(entry) or today)
    price, priced_on = _price_on(sym, at)
    if price is None:
        return None
    change = (price - entry_price) / entry_price * 100.0

    # The benchmark over the SAME window, so the comparison is like for like.
    bench_change: float | None = None
    bench = str(entry.get("benchmark") or DEFAULT_BENCHMARK)
    try:
        opened = date.fromisoformat(str(entry.get("entry_date"))[:10])
    except ValueError:
        opened = None
    if opened:
        b_start, _ = _price_on(bench, opened)
        b_end, _ = _price_on(bench, at)
        if b_start and b_end:
            bench_change = (b_end - b_start) / b_start * 100.0

    scored = dict(entry)
    scored.update({
        "status": "scored",
        "scored_on": today.isoformat(),
        "exit_price": price,
        "exit_date": priced_on,
        "change_pct": round(change, 2),
        "benchmark_change_pct": (
            round(bench_change, 2) if bench_change is not None else None
        ),
        "alpha_pct": (
            round(change - bench_change, 2) if bench_change is not None else None
        ),
        "hit": _hit(str(entry.get("verdict")), change),
    })
    return scored


def score_due(now: date | None = None) -> list[dict[str, Any]]:
    """Score every call whose horizon has passed. Model-free, so a scheduler tick
    can run it unconditionally; returns the entries it scored (empty on a quiet
    tick). Prices are fetched OUTSIDE the lock — the network call is slow and
    holding a cross-process lock across it would stall an interactive turn trying
    to record a call."""
    pending = due_entries(now)
    if not pending:
        return []
    scored = [s for s in (score_entry(e, now) for e in pending) if s is not None]
    if not scored:
        return []
    by_id = {s["id"]: s for s in scored}
    with _locked():
        items = load_entries()
        for i, e in enumerate(items):
            if e.get("id") in by_id and e.get("status") == "open":
                items[i] = by_id[e["id"]]
        save_entries(items)
    for entry in scored:
        _remember(entry)
    return scored


def _remember(entry: dict[str, Any]) -> None:
    """File a scored call as a lesson, so the existing recall path surfaces it the
    next time this ticker comes up — the agent reads what it said and how it went
    before saying something similar again. No-op when memory is off."""
    from .memory import get_memory

    mem = get_memory()
    if mem is None:
        return
    try:
        mem.save(describe_outcome(entry), "lesson")
    except Exception:  # noqa: BLE001 - a lesson that won't store must not fail a tick
        pass


def describe_outcome(entry: dict[str, Any]) -> str:
    """One line stating what was called, what happened, and how it compared."""
    verdict = str(entry.get("verdict", "")).upper()
    sym = entry.get("symbol")
    change = entry.get("change_pct")
    verb = "was RIGHT" if entry.get("hit") else "was WRONG"
    line = (
        f"Earlier {verdict} call on {sym} ({entry.get('entry_date')} → "
        f"{entry.get('exit_date')}, {entry.get('horizon_days')}d) {verb}: "
        f"{sym} {change:+.1f}%"
    )
    alpha = entry.get("alpha_pct")
    if alpha is not None:
        line += (
            f" vs {entry.get('benchmark')} {entry.get('benchmark_change_pct'):+.1f}% "
            f"({alpha:+.1f}% relative)"
        )
    thesis = str(entry.get("thesis") or "").strip()
    if thesis:
        line += f". The reasoning was: {thesis}"
    return line


# --- reporting ------------------------------------------------------------------


def track_record(symbol: str = "") -> dict[str, Any]:
    """Aggregate the scored calls, overall or for one ticker."""
    sym = (symbol or "").strip().upper()
    scored = [
        e for e in load_entries()
        if e.get("status") == "scored" and (not sym or e.get("symbol") == sym)
    ]
    hits = [e for e in scored if e.get("hit")]
    alphas = [e["alpha_pct"] for e in scored if e.get("alpha_pct") is not None]
    return {
        "scored": len(scored),
        "hits": len(hits),
        "hit_rate": (len(hits) / len(scored) * 100.0) if scored else None,
        "mean_alpha_pct": (sum(alphas) / len(alphas)) if alphas else None,
        "enough_to_read": len(scored) >= _MIN_FOR_RATE,
    }


def _describe_open(entry: dict[str, Any]) -> str:
    return (
        f"  [{entry['id']}] {str(entry['verdict']).upper():<7} {entry['symbol']:<6} "
        f"from {entry.get('entry_price', 0):,.2f} on {entry.get('entry_date')} "
        f"· scores {entry.get('due')}"
    )


def _describe_scored(entry: dict[str, Any]) -> str:
    mark = "✓" if entry.get("hit") else "✗"
    alpha = entry.get("alpha_pct")
    tail = f" · {alpha:+.1f}% vs {entry.get('benchmark')}" if alpha is not None else ""
    return (
        f"  {mark} [{entry['id']}] {str(entry['verdict']).upper():<7} "
        f"{entry['symbol']:<6} {entry.get('change_pct', 0):+.1f}% "
        f"over {entry.get('horizon_days')}d{tail}"
    )


def review_theses(symbol: str = "") -> str:
    """Review the directional calls previously recorded with `record_thesis`: which
    are still open, how the scored ones turned out, and the running hit rate.

    Use for 'how have your calls done / what did you say about X before / are you
    any good at this', and check it BEFORE making a fresh call on a ticker you have
    covered — repeating a view that has already been wrong, without saying so, is
    the failure this exists to prevent. ``symbol`` narrows it to one ticker.

    The record is a small sample of whatever the user happened to ask about, so
    report it as calibration, never as evidence the next call is right.
    """
    sym = (symbol or "").strip().upper()
    entries = [e for e in load_entries() if not sym or e.get("symbol") == sym]
    if not entries:
        scope = f" for {sym}" if sym else ""
        return (
            f"No recorded calls{scope} yet. `record_thesis` logs a directional view "
            f"so it can be scored against what actually happens."
        )
    openv = [e for e in entries if e.get("status") == "open"]
    scored = [e for e in entries if e.get("status") == "scored"]
    out: list[str] = []
    if openv:
        out.append(f"Open calls ({len(openv)}) — not yet scored:")
        out += [_describe_open(e) for e in sorted(openv, key=lambda e: e.get("due") or "")]
    if scored:
        out.append(f"Scored calls ({len(scored)}):")
        out += [
            _describe_scored(e)
            for e in sorted(scored, key=lambda e: e.get("scored_on") or "")[-12:]
        ]
        rec = track_record(sym)
        rate = rec["hit_rate"]
        line = f"Track record: {rec['hits']}/{rec['scored']} directionally right"
        if rate is not None:
            line += f" ({rate:.0f}%)"
        if rec["mean_alpha_pct"] is not None:
            line += f" · mean {rec['mean_alpha_pct']:+.1f}% vs benchmark"
        out.append(line)
        if not rec["enough_to_read"]:
            out.append(
                f"  (only {rec['scored']} scored call(s) — far too few to mean "
                f"anything; report it as such)"
            )
    out.append(
        "This is a record of past calls on whatever was asked about, not evidence "
        "about the next one. Cite it for calibration — especially to flag a view "
        "you have already been wrong on — never as a reason to trust a new call."
    )
    return "\n".join(out)


JOURNAL_TOOLS = [record_thesis, review_theses]
