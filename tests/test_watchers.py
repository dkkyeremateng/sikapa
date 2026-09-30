"""The event watchers: detection is model-free, routing decides what is worth a push.

The price fake is keyed on the DATE and carries a bar for "today" only when the
test says the session is open — the property the price watcher is about.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from financial_research_assistant import (
    alerts, channels, edgar, guardrails, hooks, journal, periodic, statements, tools, watchers,
)

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 9, 29, 14, 0, tzinfo=NY)  # a Tuesday, mid-session
TODAY = NOW.date()


def _series(daily_pct: float, today_pct: float | None, base: float = 100.0):
    """Weekday closes alternating ±daily_pct (so σ ≈ daily_pct), ending yesterday,
    plus today's bar moved by today_pct when given."""
    days = []
    d = TODAY - timedelta(days=130)
    while d < TODAY:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows, price = [], base
    for i, d in enumerate(days):
        price *= 1 + (daily_pct if i % 2 else -daily_pct) / 100
        rows.append((d.isoformat(), round(price, 4)))
    if today_pct is not None:
        rows.append((TODAY.isoformat(), round(price * (1 + today_pct / 100), 4)))
    return rows


@pytest.fixture
def world(monkeypatch):
    """A book (AAPL ~94%, KO ~6%), a price script, recorded pushes, no SEC."""
    prices: dict[str, list] = {}
    pushed: list[str] = []
    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **_kw: prices.get(sym, []))
    monkeypatch.setattr(statements, "query_positions", lambda account=None: [
        {"symbol": "AAPL", "asset_category": "STK", "currency": "USD",
         "close_price": 100.0, "value": 6000.0},
        {"symbol": "KO", "asset_category": "STK", "currency": "USD",
         "close_price": 50.0, "value": 400.0},
    ])
    monkeypatch.setattr(statements, "default_account", lambda: "U1")
    monkeypatch.setattr(statements, "list_imports", lambda: [
        {"account": "U1", "period": "August 29, 2026 - September 28, 2026"}])
    monkeypatch.setattr(channels, "deliver",
                        lambda text, prefer="": (pushed.append(text), (["telegram"], []))[1])
    monkeypatch.setattr(edgar, "_cik_for", lambda sym: None)
    monkeypatch.delenv("FRA_EVENTS_SHADOW", raising=False)
    monkeypatch.delenv("FRA_QUIET_HOURS", raising=False)
    return {"prices": prices, "pushed": pushed}


def _pass(fake=True, now=NOW):
    return asyncio.run(watchers.run_pass(fake=fake, now=now.astimezone(timezone.utc)))


# --- detection ------------------------------------------------------------------------------


def test_a_big_move_in_a_big_position_is_pushed_now(world):
    world["prices"]["AAPL"] = _series(1.0, -6.0)
    world["prices"]["KO"] = _series(0.5, 0.2)
    decided = _pass()
    assert [(e.symbol, e.severity, d) for e, d in decided] == [("AAPL", "high", "push")]
    text = world["pushed"][0]
    assert text.startswith("⚡ AAPL -6.0% today (your 94% position)")
    assert "usual daily move ±1.0%" in text and "≈ -$" in text


def test_a_move_is_judged_against_the_stocks_own_volatility(world):
    """4% is noise for a stock that moves 3% a day, and news for one that moves
    0.5%. The floor keeps a sleepy stock's tiny "two-sigma" day quiet."""
    world["prices"]["AAPL"] = _series(3.0, 4.0)
    world["prices"]["KO"] = _series(0.5, 1.5)
    assert _pass() == []
    world["prices"]["KO"] = _series(0.5, 3.4)
    assert [e.symbol for e, _d in _pass()] == ["KO"]


def test_no_bar_for_today_is_no_event(world):
    """Before the open (or on a holiday) the last bar is yesterday's; yesterday's
    move was already reported, and must not be reported again as today's."""
    world["prices"]["AAPL"] = _series(1.0, None)
    assert _pass() == []


def test_one_move_is_one_message_unless_it_gets_worse(world):
    world["prices"]["AAPL"] = _series(1.0, -6.0)
    assert len(_pass()) == 1
    assert _pass() == [], "the same move on the next pass is not news"
    world["prices"]["AAPL"] = _series(1.0, -13.0)  # a new band: worse news
    decided = _pass()
    assert [(e.key, d) for e, d in decided] == [("price:AAPL:2026-09-29:4", "push")]
    assert len(world["pushed"]) == 2, "worse news gets through the cooldown"


def test_a_medium_event_within_the_cooldown_waits_for_the_digest(world):
    world["prices"]["AAPL"] = _series(1.0, -6.0)
    _pass()  # a high push on AAPL
    ev = watchers.Event("x:1", "AAPL", "sec_filing", "medium", "AAPL filed a 10-Q")
    state = watchers._load_state()
    state["last_push"]["AAPL"] = [NOW.astimezone(timezone.utc).isoformat(), "high"]
    assert watchers._decide(ev, state, NOW.astimezone(timezone.utc)) == "cooldown"


# --- routing ----------------------------------------------------------------------------------


def test_shadow_mode_records_and_pushes_nothing(world, monkeypatch):
    monkeypatch.setenv("FRA_EVENTS_SHADOW", "1")
    world["prices"]["AAPL"] = _series(1.0, -6.0)
    assert [d for _e, d in _pass()] == ["shadow"]
    assert world["pushed"] == []
    assert "shadow mode" in asyncio.run(watchers._cmd_events("", False))


def test_quiet_holds_all_but_the_urgent_then_summarises(world, monkeypatch):
    guardrails.set_quiet(60)
    world["prices"]["AAPL"] = _series(1.0, -6.0)   # high: goes through
    world["prices"]["KO"] = _series(0.5, 3.4)      # medium: waits
    decided = {e.symbol: d for e, d in _pass()}
    assert decided == {"AAPL": "push", "KO": "held"}
    assert len(world["pushed"]) == 1
    guardrails.clear_quiet()
    _pass()
    assert world["pushed"][-1].startswith("While you were quiet:\n• KO +3.4% today")


def test_an_invented_figure_drops_the_analysis_not_the_alert(world, monkeypatch):
    from financial_research_assistant import autonomy

    async def ask(system, user, **kw):
        return "AAPL fell because its P/E hit 41.7."

    monkeypatch.setattr(autonomy, "ask", ask)
    world["prices"]["AAPL"] = _series(1.0, -6.0)
    _pass(fake=False)
    assert "41.7" not in world["pushed"][0] and world["pushed"][0].startswith("⚡ AAPL")


def test_an_undelivered_push_is_recorded_as_such(world, monkeypatch):
    monkeypatch.setattr(channels, "deliver", lambda text, prefer="": ([], ["telegram"]))
    world["prices"]["AAPL"] = _series(1.0, -6.0)
    assert [d for _e, d in _pass()] == ["undelivered"]


# --- the other watchers -----------------------------------------------------------------------


def test_filings_are_primed_then_announced(world, monkeypatch):
    feed = [{"date": "2026-09-01", "form": "10-Q", "accession": "old-1", "doc": "q.htm",
             "desc": "", "items": ""}]
    monkeypatch.setattr(edgar, "_cik_for", lambda sym: "0000320193" if sym == "AAPL" else None)
    monkeypatch.setattr(edgar, "_submission_recent", lambda cik: ("Apple", list(feed)))
    assert _pass() == [], "first sight: history is recorded, not announced"
    feed[:0] = [
        {"date": "2026-09-29", "form": "8-K", "accession": "new-1", "doc": "e.htm",
         "desc": "", "items": "2.02,9.01"},
        {"date": "2026-09-28", "form": "10-Q", "accession": "new-2", "doc": "q2.htm",
         "desc": "", "items": ""},
    ] + [{"date": "2026-09-28", "form": "4", "accession": f"f4-{i}", "doc": "x",
          "desc": "", "items": ""} for i in range(3)]
    state = watchers._load_state()
    assert _pass() == [], "throttled: filings are checked every 30 minutes"
    state = watchers._load_state()
    state["last_run"]["sec_filing"] = 0
    watchers._save_state(state)
    got = {(e.kind, e.severity) for e, _d in _pass()}
    assert got == {("earnings_out", "high"), ("sec_filing", "medium"), ("insider_cluster", "low")}


def test_a_user_alert_rule_fires_on_the_pass(world):
    world["prices"]["AAPL"] = _series(1.0, 0.1)
    world["prices"]["KO"] = _series(0.5, 0.1)
    alerts.add_alert("KO", "above", 1.0)
    decided = _pass()
    assert [(e.kind, d) for e, d in decided] == [("alert_level", "push")]
    assert "KO at" in world["pushed"][0]


def test_an_invalidated_call_is_urgent(world):
    world["prices"]["AAPL"] = _series(1.0, 0.1)
    world["prices"]["NVDA"] = [("2026-09-28", 100.0), ("2026-09-29", 88.0)]
    journal.save_entries([{"id": "t1", "symbol": "NVDA", "verdict": "bullish",
                           "status": "open", "entry_price": 110.0, "entry_date": "2026-09-01",
                           "invalidation_price": 90.0}])
    decided = [(e.kind, e.severity) for e, _d in _pass()]
    assert ("thesis_break", "high") in decided


def test_stale_positions_are_flagged_once_per_statement(world, monkeypatch):
    monkeypatch.setattr(statements, "list_imports", lambda: [
        {"account": "U1", "period": "July 1, 2026 - August 7, 2026"}])
    kinds = [e.kind for e, _d in _pass()]
    assert kinds.count("data_stale") == 1
    assert [e.kind for e, _d in _pass()].count("data_stale") == 0


# --- wiring -----------------------------------------------------------------------------------


def test_the_watchers_run_as_a_service_loop_and_stand_down_when_paused(world, monkeypatch):
    hooks.load_feature_modules()
    assert hooks.SERVICE_LOOPS.get("events") is watchers.events_loop
    passes: list[int] = []

    async def counted(fake=False, now=None):
        passes.append(1)
        return []

    monkeypatch.setattr(watchers, "run_pass", counted)
    monkeypatch.setattr(watchers, "interval_seconds", lambda now=None: 0.01)

    async def run(paused: bool):
        passes.clear()
        (guardrails.pause if paused else guardrails.resume)()
        stop = asyncio.Event()
        task = asyncio.create_task(watchers.events_loop(stop, True))
        await asyncio.sleep(0.05)
        stop.set()
        await task
        return len(passes)

    assert asyncio.run(run(paused=False)) >= 2
    assert asyncio.run(run(paused=True)) == 0


def test_what_was_not_pushed_appears_in_the_daily_report(world):
    world["prices"]["AAPL"] = _series(1.0, 0.1)
    world["prices"]["KO"] = _series(0.5, 0.1)
    watchers._log(watchers.Event("form4:KO:1", "KO", "insider_cluster", "low",
                                 "KO: 3 insider (Form 4) filings", detected_at=f"{TODAY}T15:00:00"),
                  "digest")
    section = watchers._daily_section({"session": TODAY})
    assert section.startswith("## Also Noticed Today") and "KO: 3 insider" in section
