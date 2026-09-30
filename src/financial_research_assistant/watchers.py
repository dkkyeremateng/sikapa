"""Event watchers — what makes the always-on agent notice things on its own.

Between scheduled reports nothing used to look at the holdings: a position
falling 8% at 11am surfaced in the evening digest, a new 8-K whenever someone
asked. This module runs inside ``--serve`` as its own loop (every 5 minutes
while New York is open, every 30 otherwise) and does two separate things.

**Detection is model-free.** Each watcher is a function over market data, SEC
filings, the alert rules and the journal that returns *events* — facts with a
stable key. A quiet market costs a few cached price lookups and no tokens.

**Routing decides what an event is worth.** The key dedupes it for good (one
filing is one message, however many passes see it). Severity comes from what
happened AND how much of the book it touches: a 6% move in a 1% position is
background, the same move in a 20% position is news. Then:

- ``high`` is pushed now, with a short analysis from the cheap model when the
  budget allows (facts only when it doesn't);
- ``medium`` is pushed as facts, no model;
- ``low`` waits for the daily report;
- during quiet hours only ``high`` goes out; the rest is held and summarised
  when the quiet ends;
- a symbol pushed within the cooldown is only pushed again if the news is worse.

**Shadow mode** (``FRA_EVENTS_SHADOW=1``) records every decision in
``events.jsonl`` and pushes nothing — the week of watching before trusting the
thresholds that the plan asks for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import date, datetime, timedelta, timezone

from . import guardrails, hooks, market, periodic
from .storage import append_jsonl, locked, read_json, read_jsonl, state_file, write_private

SEVERITIES = ("low", "medium", "high")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Event:
    key: str
    symbol: str
    kind: str
    severity: str
    title: str
    facts: dict[str, Any] = field(default_factory=dict)
    detected_at: str = ""


# --- configuration -------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def shadow() -> bool:
    return (os.environ.get("FRA_EVENTS_SHADOW") or "").strip().lower() in ("1", "true", "yes", "on")


def floor_pct() -> float:
    """The smallest session move ever called an event (``FRA_EVENT_FLOOR_PCT``, 3)."""
    return _env_float("FRA_EVENT_FLOOR_PCT", 3.0)


def sigmas() -> float:
    """How many of a symbol's own daily standard deviations make a move unusual
    (``FRA_EVENT_SIGMA``, 2). A 3% day is noise for a volatile name and news for
    a utility; the floor keeps a sleepy stock's 0.8% "two-sigma" day quiet."""
    return _env_float("FRA_EVENT_SIGMA", 2.0)


def cooldown_minutes() -> float:
    return _env_float("FRA_EVENT_COOLDOWN_MIN", 120.0)


def events_log():
    return state_file("events.jsonl", "FRA_EVENTS_LOG")


def _state_path():
    return state_file("events-state.json", "FRA_EVENTS_STATE")


def _load_state() -> dict[str, Any]:
    data = read_json(_state_path(), {})
    if not isinstance(data, dict):
        data = {}
    for k, empty in (("seen", {}), ("last_push", {}), ("primed", []), ("held", []),
                     ("last_run", {})):
        data.setdefault(k, empty)
    return data


def _save_state(state: dict[str, Any]) -> None:
    # Forget keys after 30 days: long enough that a filing is never re-announced,
    # short enough that the file doesn't grow for the life of the server.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    state["seen"] = {k: v for k, v in state["seen"].items() if str(v) >= cutoff}
    write_private(_state_path(), json.dumps(state, indent=1), prefix=".events-")


# --- watchers --------------------------------------------------------------------------------

#: A watcher: ``(ctx) -> events``. ``ctx`` carries ``now`` (aware UTC), ``book``,
#: ``state`` and ``fake``. Registered in order; each runs even if another fails.
Watcher = Callable[[dict[str, Any]], list[Event]]
WATCHERS: dict[str, Watcher] = {}


def watcher(name: str) -> Callable[[Watcher], Watcher]:
    def register(fn: Watcher) -> Watcher:
        WATCHERS[name] = fn
        return fn

    return register


def _watchlist() -> list[str]:
    """Symbols watched but not held (the investor profile's watchlist, when one
    exists)."""
    try:
        from . import profile

        return list(profile.load().get("watchlist") or [])
    except Exception:  # noqa: BLE001 - no profile module or no profile yet
        return []


@watcher("price_move")
def price_moves(ctx: dict[str, Any]) -> list[Event]:
    """A holding (or watched symbol) moving more than max(floor, k·σ) on the day."""
    book = ctx["book"]
    held = {h["symbol"]: h for h in book["holdings"]}
    symbols = list(held) + [s for s in _watchlist() if s not in held]
    series = periodic._fetch_many(symbols, 120)  # pyright: ignore[reportPrivateUsage]
    today = market.ny_now(ctx["now"]).date().isoformat()
    out: list[Event] = []
    for sym in symbols:
        rows = series.get(sym) or []
        if len(rows) < 22 or rows[-1][0] != today:
            continue  # no bar for today: not a session, or not open yet
        closes = [c for _d, c in rows]
        prev, last = closes[-2], closes[-1]
        if not prev:
            continue
        move = (last / prev - 1) * 100
        rets = [(closes[i] / closes[i - 1] - 1) * 100 for i in range(max(1, len(closes) - 61),
                                                                    len(closes) - 1)
                if closes[i - 1]]
        sigma = statistics.pstdev(rets) if len(rets) >= 20 else 0.0
        threshold = max(floor_pct(), sigmas() * sigma)
        if abs(move) < threshold:
            continue
        band = int(abs(move) // threshold)
        h = held.get(sym)
        weight = float(h["weight_pct"]) if h else 0.0
        dollars = h["units"] * (last - prev) if h and h["currency"] == "USD" else None
        impact = weight * abs(move) / 100  # percent of the book this move is worth
        if h and (impact >= 0.5 or band >= 2):
            severity = "high"
        elif h:
            severity = "medium"
        else:
            severity = "medium" if band >= 2 else "low"
        out.append(Event(
            key=f"price:{sym}:{today}:{band}", symbol=sym, kind="price_move",
            severity=severity,
            title=f"{sym} {move:+.1f}% today" + (f" (your {weight:.0f}% position)" if h else
                                                  " (watchlist)"),
            facts={"move_pct": round(move, 2), "prev_close": prev, "prev_date": rows[-2][0],
                   "price": last, "date": today, "sigma_pct": round(sigma, 2),
                   "threshold_pct": round(threshold, 2), "weight_pct": round(weight, 2),
                   "dollars": round(dollars, 2) if dollars is not None else None,
                   "book_impact_pct": round(impact, 3), "band": band},
        ))
    return out


@watcher("alert_level")
def alert_rules(ctx: dict[str, Any]) -> list[Event]:
    """The user's own alert rules, checked on every pass instead of only when a
    digest happened to be built."""
    from . import alerts

    if not alerts.load_alerts():
        return []
    held = [h["symbol"] for h in ctx["book"]["holdings"]]
    today = market.ny_now(ctx["now"]).date()
    out = []
    for line in alerts.evaluate_alerts(held, 5, today):
        rule = line[1:line.index("]")] if line.startswith("[") and "]" in line else "?"
        text = line.split("] ", 1)[-1]
        sym = text.split(" ", 1)[0]
        out.append(Event(key=f"alert:{rule}:{today.isoformat()}", symbol=sym,
                         kind="alert_level", severity="medium", title=f"🔔 {text}",
                         facts={"rule": rule, "text": text}))
    return out


#: Forms worth a message, and what they are called on one.
_FORMS = {"8-K": "8-K", "10-Q": "quarterly report (10-Q)", "10-K": "annual report (10-K)",
          "SC 13D": "activist stake (13D)", "SC 13D/A": "activist stake amended (13D/A)"}
_FILINGS_EVERY = 30 * 60.0


@watcher("sec_filing")
def filings(ctx: dict[str, Any]) -> list[Event]:
    """New SEC filings by the companies held. The first time a company is seen its
    existing filings are recorded as known, so starting the watcher doesn't
    announce a year of history."""
    from . import edgar

    state = ctx["state"]
    if time.time() - float(state["last_run"].get("sec_filing") or 0) < _FILINGS_EVERY:
        return []
    state["last_run"]["sec_filing"] = time.time()
    today = market.ny_now(ctx["now"]).date()
    recent_from = (today - timedelta(days=3)).isoformat()
    out: list[Event] = []
    for h in ctx["book"]["holdings"]:
        sym = h["symbol"]
        cik = edgar._cik_for(sym)  # pyright: ignore[reportPrivateUsage]
        if not cik:
            continue  # an ETF, a coin, a foreign listing
        _name, recent = edgar._submission_recent(cik)  # pyright: ignore[reportPrivateUsage]
        if cik not in state["primed"]:
            # Only the window that could ever be announced: anything older is
            # skipped by date below anyway, and recording a company's whole
            # history made a state file of thousands of keys rewritten every pass.
            for f in recent:
                if str(f["date"]) < recent_from:
                    break
                state["seen"].setdefault(f"filing:{f['accession']}", ctx["now"].isoformat())
            state["primed"].append(cik)
            continue
        form4: dict[str, int] = {}
        for f in recent:
            if str(f["date"]) < recent_from:
                break
            form = str(f["form"]).upper()
            if form == "4":
                form4[str(f["date"])] = form4.get(str(f["date"]), 0) + 1
                continue
            if form not in _FORMS:
                continue
            items = str(f.get("items") or "")
            url = edgar._filing_url(cik, str(f["accession"]), str(f["doc"]))  # pyright: ignore[reportPrivateUsage]
            if form == "8-K" and "2.02" in items:
                out.append(Event(key=f"filing:{f['accession']}", symbol=sym,
                                 kind="earnings_out", severity="high",
                                 title=f"{sym} reported results (8-K item 2.02, filed {f['date']})",
                                 facts={"form": form, "items": items, "date": f["date"],
                                        "url": url}))
                continue
            label = _FORMS[form]
            detail = edgar._decode_items(items) if items else ""  # pyright: ignore[reportPrivateUsage]
            out.append(Event(key=f"filing:{f['accession']}", symbol=sym, kind="sec_filing",
                             severity="medium",
                             title=f"{sym} filed {label}" + (f": {detail}" if detail else ""),
                             facts={"form": form, "items": items, "date": f["date"], "url": url}))
        for day, n in form4.items():
            if n >= 3:
                out.append(Event(key=f"form4:{sym}:{day}", symbol=sym, kind="insider_cluster",
                                 severity="low",
                                 title=f"{sym}: {n} insider (Form 4) filings on {day}",
                                 facts={"count": n, "date": day}))
    return out


@watcher("thesis_break")
def theses(ctx: dict[str, Any]) -> list[Event]:
    """Open journal calls that have been proved wrong (the invalidation level the
    call named was crossed) or have moved a long way either side."""
    from . import journal

    open_entries = [e for e in journal.load_entries() if e.get("status") == "open"]
    if not open_entries:
        return []
    series = periodic._fetch_many([str(e["symbol"]) for e in open_entries], 10)  # pyright: ignore[reportPrivateUsage]
    out = []
    for e in open_entries:
        rows = series.get(str(e["symbol"])) or []
        entry = float(e.get("entry_price") or 0)
        if not rows or entry <= 0:
            continue
        price = rows[-1][1]
        change = (price / entry - 1) * 100
        raw_level = e.get("invalidation_price")
        level = float(raw_level) if isinstance(raw_level, (int, float)) else 0.0
        verdict = str(e.get("verdict"))
        broken = level > 0 and (
            (verdict == "bullish" and price <= level) or (verdict == "bearish" and price >= level))
        base = {"id": e.get("id"), "verdict": verdict, "entry_price": entry,
                "entry_date": e.get("entry_date"), "price": price, "priced_on": rows[-1][0],
                "since_entry_pct": round(change, 2), "invalidation_price": level or None}
        if broken:
            out.append(Event(key=f"thesis:{e.get('id')}:invalidated", symbol=str(e["symbol"]),
                             kind="thesis_break", severity="high",
                             title=f"{e['symbol']}: the {verdict} call is invalidated — "
                                   f"{price:,.2f} crossed {level:,.2f}",
                             facts=base))
        elif abs(change) >= 15:
            out.append(Event(key=f"thesis:{e.get('id')}:move15", symbol=str(e["symbol"]),
                             kind="thesis_move", severity="medium",
                             title=f"{e['symbol']}: {change:+.1f}% since the {verdict} call",
                             facts=base))
    return out


@watcher("data_stale")
def stale_data(ctx: dict[str, Any]) -> list[Event]:
    """The positions on file are old — almost always a failing Flex sync. Said
    once per statement date, not on every pass."""
    book = ctx["book"]
    warning = periodic.staleness(book, market.ny_now(ctx["now"]).date())
    if not warning:
        return []
    return [Event(key=f"stale:{book['as_of']}", symbol="", kind="data_stale",
                  severity="medium", title=warning, facts={"as_of": book["as_of"]})]


# --- routing ----------------------------------------------------------------------------------


def _decide(ev: Event, state: dict[str, Any], now: datetime) -> str:
    if shadow():
        return "shadow"
    if ev.severity == "low":
        return "digest"
    last = state["last_push"].get(ev.symbol) if ev.symbol else None
    if last:
        at, sev = last[0], last[1]
        last_band = int(last[2]) if len(last) > 2 else 0
        try:
            recent = now - datetime.fromisoformat(at) < timedelta(minutes=cooldown_minutes())
        except ValueError:
            recent = False
        # Worse news is never held back by the cooldown: a stock down 6% that is
        # now down 13% is a second message, even though both were "high".
        worse = _RANK[ev.severity] > _RANK.get(sev, 0) or (
            ev.kind == "price_move" and int(ev.facts.get("band") or 0) > last_band)
        if recent and not worse:
            return "cooldown"
    if ev.severity != "high" and guardrails.quiet_now():
        return "held"
    return "push"


_ANALYSIS_SYSTEM = (
    "You write a two-sentence note on one market event for the investor who holds "
    "the position: what likely happened and what it means for them. Use only the "
    "figures given; you may round them but never compute new ones. No advice to "
    "trade. No preamble."
)


async def _analysis(ev: Event, fake: bool) -> str:
    """The cheap model's take on a high-severity event, or "" (budget, pause,
    provider down, or a figure it made up)."""
    from . import autonomy

    cap = int(_env_float("FRA_AUTONOMY_EVENT_TOKENS", 0))
    if cap:
        spent_today = sum(int(r.get("tokens") or 0) for r in read_jsonl(autonomy.usage_log())
                          if str(r.get("purpose", "")).startswith("event")
                          and str(r.get("at", "")).startswith(datetime.now().strftime("%Y-%m-%d")))
        if spent_today >= cap:
            return ""
    facts = json.dumps(ev.facts, default=str)
    text = await autonomy.ask(_ANALYSIS_SYSTEM, f"{ev.title}\nFacts: {facts}", tier="quick",
                              purpose=f"event:{ev.kind}", fake=fake,
                              fake_reply=f"{ev.title}." if fake else "")
    if not text:
        return ""
    sheet = periodic.Brief(kind="daily", period="", label="", title=ev.title, subtitle="",
                           highlights="", markdown="", message=ev.title, facts=ev.facts)
    return "" if periodic.unsupported_figures(text, sheet) else text


def format_event(ev: Event) -> str:
    icon = {"high": "⚡", "medium": "•", "low": "·"}[ev.severity]
    lines = [f"{icon} {ev.title}"]
    f = ev.facts
    if ev.kind == "price_move":
        bits = [f"{f['prev_close']:,.2f} → {f['price']:,.2f} ({f['prev_date']} close → now)",
                f"usual daily move ±{f['sigma_pct']:.1f}%"]
        if f.get("dollars") is not None:
            sign = "-" if f["dollars"] < 0 else "+"
            bits.append(f"≈ {sign}${abs(f['dollars']):,.0f} on your position")
        lines.append(" · ".join(bits))
    if f.get("url"):
        lines.append(str(f["url"]))
    return "\n".join(lines)


def _log(ev: Event, decision: str, text: str = "") -> None:
    rec = {"at": datetime.now().isoformat(timespec="seconds"), **asdict(ev), "decision": decision}
    if text:
        rec["pushed"] = text
    append_jsonl(events_log(), rec)


async def route(events: list[Event], state: dict[str, Any], now: datetime,
                fake: bool = False) -> list[tuple[Event, str]]:
    """Dedupe, decide, push, log. Returns ``(event, decision)`` for new events."""
    from . import channels

    done: list[tuple[Event, str]] = []
    for ev in events:
        if ev.key in state["seen"]:
            continue
        state["seen"][ev.key] = now.isoformat()
        ev.detected_at = now.isoformat(timespec="seconds")
        decision = _decide(ev, state, now)
        text = ""
        if decision == "push":
            text = format_event(ev)
            if ev.severity == "high":
                note = await _analysis(ev, fake)
                if note:
                    text += f"\n{note}"
            delivered, _failed = await asyncio.to_thread(channels.deliver, text)
            if not channels.carried_full_text(delivered):
                decision = "undelivered"
            elif ev.symbol:
                state["last_push"][ev.symbol] = [now.isoformat(), ev.severity,
                                                 int(ev.facts.get("band") or 0)]
        elif decision == "held":
            state["held"].append(asdict(ev))
        _log(ev, decision, text)
        done.append((ev, decision))
    return done


async def _release_held(state: dict[str, Any]) -> None:
    """Once quiet is over, one message summarising what waited."""
    from . import channels

    if not state["held"] or guardrails.quiet_now():
        return
    held, state["held"] = state["held"], []
    text = "While you were quiet:\n" + "\n".join(f"• {e['title']}" for e in held[:15])
    if len(held) > 15:
        text += f"\n…and {len(held) - 15} more (/events)"
    if not shadow():
        await asyncio.to_thread(channels.deliver, text)


# --- the pass and the loop ---------------------------------------------------------------------


async def run_pass(fake: bool = False, now: datetime | None = None) -> list[tuple[Event, str]]:
    """One pass of every watcher, then routing. Each watcher is isolated: a data
    source being down costs that watcher's events, not the pass."""
    now = now or datetime.now(timezone.utc)
    book = await asyncio.to_thread(periodic.load_book)
    path = _state_path()
    with locked(path):
        state = _load_state()
        ctx = {"now": now, "book": book, "state": state, "fake": fake}
        events: list[Event] = []
        for name, fn in WATCHERS.items():
            try:
                events += await asyncio.to_thread(fn, ctx)
            except Exception as exc:  # noqa: BLE001 - one source down, not the pass
                print(f"watcher {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        decided = await route(events, state, now, fake)
        await _release_held(state)
        state["last_run"]["pass"] = now.isoformat()
        _save_state(state)
    return decided


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=max(0.0, seconds))
    except asyncio.TimeoutError:
        pass


def interval_seconds(now: datetime | None = None) -> float:
    if market.regular_hours(now):
        return _env_float("FRA_EVENTS_INTERVAL_OPEN", 300.0)
    return _env_float("FRA_EVENTS_INTERVAL_CLOSED", 1800.0)


async def events_loop(stop: asyncio.Event, fake: bool) -> None:
    """The service loop: a pass, then sleep until the next one. Paused means no
    passes at all — watching is autonomous work too."""
    while not stop.is_set():
        if not guardrails.paused_reason():
            try:
                decided = await run_pass(fake)
                pushed = sum(1 for _e, d in decided if d == "push")
                if decided:
                    print(f"events: {len(decided)} new, {pushed} pushed", file=sys.stderr)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the next pass tries again
                print(f"event pass failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        await _sleep_or_stop(stop, interval_seconds())


if (os.environ.get("FRA_EVENTS") or "").strip().lower() not in ("0", "off", "false", "no"):
    hooks.register_service_loop("events", events_loop)


# --- reading the record ----------------------------------------------------------------------


def events_on(day: str) -> list[dict[str, Any]]:
    return [e for e in read_jsonl(events_log()) if str(e.get("detected_at", ""))[:10] == day
            or str(e.get("at", ""))[:10] == day]


def _daily_section(ctx: dict[str, Any]) -> str:
    """Events of the session that weren't pushed on their own."""
    day = ctx["session"].isoformat()
    rows = [e for e in events_on(day) if e.get("decision") in ("digest", "held", "cooldown", "shadow")]
    if not rows:
        return ""
    return "## Also Noticed Today\n" + "\n".join(f"- {e['title']}" for e in rows[:12])


def _weekly_section(ctx: dict[str, Any]) -> str:
    """The week's pushed price events and what the stock did afterwards — the
    first read on whether the thresholds are right."""
    end: date = ctx["week_end"]
    start = end - timedelta(days=end.weekday())
    pushed = [e for e in read_jsonl(events_log())
              if e.get("decision") in ("push", "shadow") and e.get("kind") == "price_move"
              and start.isoformat() <= str(e.get("detected_at", ""))[:10] <= end.isoformat()]
    if not pushed:
        return ""
    series = periodic._fetch_many(sorted({e["symbol"] for e in pushed}), 30)  # pyright: ignore[reportPrivateUsage]
    lines = ["## Alerts This Week, and What Followed",
             "| Event | Move then | Since, to week end |", "|---|---|---|"]
    for e in pushed[:12]:
        rows = [(d, c) for d, c in series.get(e["symbol"]) or [] if d <= end.isoformat()]
        price = e["facts"].get("price")
        after = (rows[-1][1] / price - 1) * 100 if rows and price else None
        lines.append(f"| {e['symbol']} {str(e['detected_at'])[:10]} | "
                     f"{e['facts'].get('move_pct', 0):+.1f}% | "
                     + (f"{after:+.1f}%" if after is not None else "n/a") + " |")
    return "\n".join(lines)


periodic.register_section("daily", _daily_section)
periodic.register_section("weekly", _weekly_section)


async def _cmd_events(arg: str, _fake: bool) -> str:
    n = int(arg) if (arg or "").isdigit() else 10
    rows = read_jsonl(events_log(), limit=min(n, 50))
    if not rows:
        return "No events recorded yet." + (" (shadow mode is on)" if shadow() else "")
    head = "Recent events" + (" — shadow mode, nothing is pushed" if shadow() else "") + ":"
    return head + "\n" + "\n".join(
        f"{str(e.get('detected_at', ''))[5:16].replace('T', ' ')}  [{e.get('severity')}"
        f"→{e.get('decision')}] {e.get('title')}" for e in rows)


def status_line() -> str:
    state = _load_state()
    last = str(state["last_run"].get("pass") or "")
    today = datetime.now().strftime("%Y-%m-%d")
    n = len(events_on(today))
    if not last:
        return "watchers: not run yet" + (" (shadow mode)" if shadow() else "")
    return (f"watchers: last pass {last[11:16]} UTC · {n} event(s) today"
            + (" · shadow mode" if shadow() else ""))


hooks.register_command("events", _cmd_events, "what the watchers noticed: /events 20")
hooks.register_status_line("watchers", status_line)
